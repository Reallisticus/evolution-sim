from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from evolution_sim.config import WorldConfig
from evolution_sim.mind.evaluation_harness import (
    CONTROLLED_FIXTURE_NAMES,
    MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
    MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
    mind_v3_fixture_gate_config,
    mind_v3_fixture_gate_status,
    run_mind_v3_fixture_suite,
)
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.mind.carrion_autopsy import (
    build_carrion_autopsy_report,
    load_carrion_autopsy_trajectory_jsonl,
)
from evolution_sim.mind.carrion_objective_audit import (
    MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
    MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY,
    build_fixture_rerank_recovery_probe_archive_report,
)
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_ARCHITECTURE,
    MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
    MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE,
    MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
    MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
    MIND_V3_SPECIALIZATION_PROFILES,
    founder_mind_v3_metadata,
    inherit_mind_v3_metadata,
    mind_v3_controller_architecture_is_promotion_eligible,
)
from evolution_sim.mind.v3_policy import (
    MIND_V3_FOUNDER_TEMPLATE_ASSIGNMENT_POLICY,
    MIND_V3_REPRODUCTION_READINESS_GOALS,
    MindV3EvolutionPolicy,
)

MIND_V3_EVOLUTION_SEARCH_SCHEMA_VERSION = "mind_v3_evolution_search_v1"
MIND_V3_EVOLUTION_SCORE_POLICY = (
    "need_gated_navigation_visible_movement_fixture_selection_qd_v26"
)
MIND_V3_ARCHIVE_POLICY = "quality_diversity_archive_v2"
MIND_V3_CURRICULUM_POLICY = "holdout_gated_tick_curriculum_v1"
MIND_V3_ROLLOUT_EXECUTION_POLICY = "mind_v3_candidate_process_pool_rollouts_v1"
MIND_V3_WARM_START_POLICY = "fixture_archive_report_warm_start_v1"
MIND_V3_CONTROLLER_SELECTION_POLICY = (
    "runtime_lineage_elite_controller_selection_v1"
)
MIND_V3_FIXTURE_RERANK_POLICY = (
    "fixture_holdout_top_k_multi_horizon_scavenger_lane_rerank_v7"
)
MIND_V3_GENERATION_FIXTURE_SELECTION_POLICY = (
    "generation_fixture_blocker_severity_parent_selection_pressure_v3"
)
MIND_V3_GENERATION_FIXTURE_SELECTION_NOMINEE_POLICY = (
    "generation_fixture_diverse_nominee_pool_v1"
)
MIND_V3_FIXTURE_REPAIR_POLICY = "fixture_blocker_composite_founder_pool_repair_v5"
MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY = "initial-only"
MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_AND_REPAIR = "initial-and-repair"
MIND_V3_SUSTAINED_READINESS_POLICY = (
    "trajectory_observed_temporal_bottleneck_readiness_v3"
)
MIND_V3_FOUNDER_TEMPLATE_POOL_POLICY = (
    "archive_diverse_founder_template_pool_with_identity_diagnostics_v2"
)
MIND_V3_CONTROLLER_LINEAGE_ELITE_LIMIT = 8
MIND_V3_FOUNDER_TEMPLATE_POOL_LIMIT = 8
MIND_V3_COMPOSITE_FOUNDER_TEMPLATE_POOL_LIMIT = 16
MIND_V3_FIXTURE_REPAIR_LIMIT = 4
MIND_V3_FIXTURE_BRIDGE_REPAIR_LIMIT = 4
MIND_V3_FIXTURE_BRIDGE_DONOR_TEMPLATE_LIMITS = (1, 2, 4)
MIND_V3_FIXTURE_WARM_START_CANDIDATE_LIMIT = 4
MIND_V3_GENERATION_FIXTURE_SELECTION_DEFAULT_TOP_K = 4
MIND_V3_FOUNDER_TEMPLATE_POOL_MIN_ALIVE_MEAN = 2.0
MIND_V3_TERMINAL_REPRODUCTION_DEAD_END_PENALTY = 5.0
MIND_V3_SEED_BRITTLE_BIRTH_PENALTY = 4.0
MIND_V3_TERMINAL_REPRODUCTION_BRITTLENESS_PENALTY = 6.0
MIND_V3_SUSTAINED_READY_DEAD_END_PENALTY = 4.0
MIND_V3_SUSTAINED_READY_PAIR_DEAD_END_PENALTY = 2.0
MIND_V3_BEHAVIOR_NICHE_LIMIT = 32
MIND_V3_FIXTURE_ECOLOGY_LANE_CANDIDATE_LIMIT = 3
MIND_V3_DOMINANT_ACTION_COLLAPSE_SHARE = 0.65
MIND_V3_DOMINANT_ACTION_COLLAPSE_MAX_PENALTY = 6.0
MIND_V3_ACTIVE_BEHAVIOR_NICHE_BONUS = 0.25
MIND_V3_ACTIVE_BEHAVIOR_NICHE_BONUS_CAP = 6
MIND_V3_REPRODUCTION_VIABILITY_SCORE_POLICY = (
    "terminal_reproduction_viability_pressure_v3"
)
ARCHIVE_ORDER = (
    "fixture_selection",
    "balanced",
    "survival",
    "births",
    "resource_use",
    "scavenger_lane",
    "movement",
    "low_death",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run heuristic-free Mind v3 population search over inherited "
            "controller templates."
        )
    )
    parser.add_argument(
        "--seeds",
        default="5,13",
        help="Comma-separated simulator seeds used to evaluate each candidate.",
    )
    parser.add_argument(
        "--holdout-seeds",
        help="Optional comma-separated seeds used only after search selection.",
    )
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=311)
    parser.add_argument(
        "--controller-architecture",
        choices=[
            MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
            MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE,
            MIND_V3_CONTROLLER_ARCHITECTURE,
            MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        ],
        default=MIND_V3_CONTROLLER_ARCHITECTURE,
        help=(
            "Controller architecture used only for newly created founder "
            "candidates. The default preserves the current v4 controller."
        ),
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help=(
            "Number of CPU worker processes used for candidate and holdout "
            "rollouts. The default keeps the previous serial execution path."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from a previous mind_v3_evolution_search_v1 report.",
    )
    parser.add_argument(
        "--warm-start-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "Seed the initial population from one or more previous Mind v3 "
            "search reports. Warm-start candidates preserve report-derived "
            "founder template pools and are forced into fixture rerank."
        ),
    )
    parser.add_argument(
        "--comparison-baseline-report",
        type=Path,
        help=(
            "Optional existing Mind v3 evolution report used only to emit "
            "matched holdout seed diagnostic deltas. It does not affect "
            "selection, scoring, or gates."
        ),
    )
    parser.add_argument(
        "--warm-start-candidate-limit",
        type=int,
        default=4,
        help=(
            "Maximum total candidates imported from --warm-start-report. "
            "Imported candidates replace the tail of generation 0."
        ),
    )
    parser.add_argument(
        "--curriculum-ticks",
        help=(
            "Optional comma-separated tick horizons. Each horizon runs a "
            "full search stage and must pass holdout floors before the next "
            "horizon is attempted."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-alive",
        type=float,
        default=1.0,
        help="Minimum holdout alive-agent mean required to advance a curriculum stage.",
    )
    parser.add_argument(
        "--curriculum-min-holdout-births",
        type=float,
        default=0.0,
        help="Minimum holdout births mean required to advance a curriculum stage.",
    )
    parser.add_argument(
        "--curriculum-min-holdout-movement-rate",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout movement-event rate required to advance a "
            "curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-unique-actions",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout unique requested actions mean required to "
            "advance a curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-behavior-niches",
        type=int,
        default=0,
        help=(
            "Minimum holdout behavior niche count required to advance a "
            "curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-max-holdout-dominant-action-share",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed holdout share for the most common requested "
            "action. Values below 1.0 reject single-action collapse."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-active-behavior-niches",
        type=int,
        default=0,
        help=(
            "Minimum active holdout behavior niche count required to advance "
            "a curriculum stage. Active niches require live, moving, or "
            "reproducing lineage evidence."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-energy-viability",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout terminal energy-viability share required to "
            "advance a curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-hydration-viability",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout terminal hydration-viability share required to "
            "advance a curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-health-viability",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout terminal health-viability share required to "
            "advance a curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-matched-diet-viability",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout terminal matched-diet viability share required "
            "to advance a curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-biologically-ready",
        type=float,
        default=0.0,
        help=(
            "Minimum holdout biologically reproduction-ready agent mean "
            "required to advance a curriculum stage."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-ready-agent-tick-share",
        type=float,
        default=0.0,
        help=(
            "Minimum share of holdout trajectory records whose finalized "
            "state is reproduction-ready."
        ),
    )
    parser.add_argument(
        "--curriculum-min-holdout-ready-pair-tick-share",
        type=float,
        default=0.0,
        help=(
            "Minimum share of holdout ticks with at least two "
            "reproduction-ready finalized agent states."
        ),
    )
    parser.add_argument(
        "--fixture-suite",
        choices=["none", "basic"],
        default="none",
        help=(
            "Optionally evaluate the selected candidate on controlled ecology "
            "fixtures and record a hard regression gate."
        ),
    )
    parser.add_argument(
        "--fixture-seeds",
        help=(
            "Comma-separated fixture seeds. Defaults to holdout seeds when "
            "present, otherwise --seeds."
        ),
    )
    parser.add_argument(
        "--fixture-names",
        help=(
            "Optional comma-separated fixture subset for --fixture-suite. "
            "Defaults to all fixtures in the selected suite."
        ),
    )
    parser.add_argument(
        "--fixture-ticks",
        type=int,
        help="Fixture tick horizon. Defaults to the active search horizon.",
    )
    parser.add_argument(
        "--fixture-min-alive",
        type=float,
        default=1.0,
        help="Minimum v3 alive-agent mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-births",
        type=float,
        default=0.0,
        help="Minimum v3 births mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-mixed-stable-births",
        type=float,
        default=0.0,
        help="Minimum v3 births mean required specifically on mixed_stable.",
    )
    parser.add_argument(
        "--fixture-min-energy-viability",
        type=float,
        default=0.0,
        help="Minimum terminal energy viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-hydration-viability",
        type=float,
        default=0.0,
        help="Minimum terminal hydration viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-health-viability",
        type=float,
        default=0.0,
        help="Minimum terminal health viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-matched-diet-viability",
        type=float,
        default=0.0,
        help="Minimum terminal matched-diet viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-biologically-ready",
        type=float,
        default=0.0,
        help=(
            "Minimum biologically-ready terminal agent mean required for "
            "every fixture."
        ),
    )
    parser.add_argument(
        "--fixture-rerank-top-k",
        type=int,
        default=1,
        help=(
            "When --fixture-suite is enabled, evaluate the top K search "
            "candidates on holdout and fixtures, then select by fixture gate "
            "blockers plus holdout/fixture outcomes. Default 1 preserves the "
            "previous selected-candidate path."
        ),
    )
    parser.add_argument(
        "--fixture-selection-top-k",
        type=int,
        default=MIND_V3_GENERATION_FIXTURE_SELECTION_DEFAULT_TOP_K,
        help=(
            "When --fixture-suite is enabled, evaluate a bounded generation "
            "candidate set on controlled fixtures before archive/parent "
            "selection. The set preserves broad-score, ecology-lane, "
            "current-architecture, and controlled-readiness nominees before "
            "filling by the scalar prefilter. "
            "Use 0 to disable fixture-in-generation pressure."
        ),
    )
    parser.add_argument(
        "--fixture-rerank-ticks",
        help=(
            "Optional comma-separated fixture tick horizons used only for "
            "top-K fixture rerank. Defaults to --fixture-ticks or the active "
            "search horizon. Use values such as 80,120 to reject candidates "
            "that only pass the short fixture horizon."
        ),
    )
    parser.add_argument(
        "--fixture-rerank-selector-probe",
        choices=[MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY],
        help=(
            "Opt-in JSON-only carrion recovery diagnostics for bounded "
            "fixture-rerank candidates. This never changes active selection."
        ),
    )
    parser.add_argument(
        "--fixture-rerank-selector-probe-scope",
        choices=[
            MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY,
            MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_AND_REPAIR,
        ],
        default=MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY,
        help=(
            "Candidate scope for --fixture-rerank-selector-probe. The default "
            "probes only the initial fixture-rerank pool; initial-and-repair "
            "also probes standard repair and bridge-repair candidates."
        ),
    )
    parser.add_argument(
        "--fixture-rerank-selector-probe-trajectory-output-dir",
        type=Path,
        help=(
            "Optional trajectory directory for --fixture-rerank-selector-probe. "
            "Defaults to a directory next to --output."
        ),
    )
    parser.add_argument(
        "--fixture-recovery-archive-retention",
        choices=[MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY],
        help=(
            "Opt-in report-backed carrion recovery archive retention for "
            "future parent diversity. This consumes completed fixture-rerank "
            "recovery probe fields and never changes current fixture-rerank "
            "selection."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-evolution-search-report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.rollout_workers < 1:
        raise SystemExit("--rollout-workers must be >= 1")
    if args.fixture_rerank_top_k < 1:
        raise SystemExit("--fixture-rerank-top-k must be >= 1")
    if args.fixture_selection_top_k < 0:
        raise SystemExit("--fixture-selection-top-k must be >= 0")
    if args.warm_start_candidate_limit < 0:
        raise SystemExit("--warm-start-candidate-limit must be >= 0")
    if args.fixture_rerank_selector_probe and args.fixture_suite == "none":
        raise SystemExit(
            "--fixture-rerank-selector-probe requires --fixture-suite"
        )
    if args.fixture_rerank_selector_probe and args.fixture_rerank_top_k <= 1:
        raise SystemExit(
            "--fixture-rerank-selector-probe requires --fixture-rerank-top-k > 1"
        )
    if (
        args.fixture_rerank_selector_probe_scope
        != MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY
        and not args.fixture_rerank_selector_probe
    ):
        raise SystemExit(
            "--fixture-rerank-selector-probe-scope requires "
            "--fixture-rerank-selector-probe"
        )
    if args.fixture_recovery_archive_retention and args.fixture_suite == "none":
        raise SystemExit(
            "--fixture-recovery-archive-retention requires --fixture-suite"
        )
    if args.fixture_recovery_archive_retention and args.fixture_rerank_top_k <= 1:
        raise SystemExit(
            "--fixture-recovery-archive-retention requires "
            "--fixture-rerank-top-k > 1"
        )
    if (
        args.fixture_recovery_archive_retention
        and not args.fixture_rerank_selector_probe
    ):
        raise SystemExit(
            "--fixture-recovery-archive-retention requires "
            "--fixture-rerank-selector-probe"
        )
    if not mind_v3_controller_architecture_is_promotion_eligible(
        str(args.controller_architecture)
    ):
        raise SystemExit("--controller-architecture must be promotion eligible")
    if args.resume_from is not None and args.warm_start_report:
        raise SystemExit("--warm-start-report cannot be combined with --resume-from")
    if args.curriculum_ticks is not None:
        report = _run_curriculum_search(args)
        _attach_comparison_baseline_report(
            report,
            args.comparison_baseline_report,
        )
        _write_report(args.output, report)
        _print_report_summary(args.output, report)
        return
    if args.resume_from is not None:
        state = _load_resume_state(args.resume_from)
        seeds = list(state["seeds"])
        ticks = int(state["ticks"])
        population_size = int(state["population_size"])
        search_seed = int(state["search_seed"])
        rng = Random()
        rng.setstate(_decode_rng_state(state["rng_state"]))
        candidates = list(state["next_candidates"])
        generations = list(state["generations"])
        start_generation_index = int(state["next_generation_index"])
        resumed_from = str(args.resume_from)
        holdout_seeds = _parse_optional_seeds(args.holdout_seeds) or list(
            state["holdout_seeds"]
        )
    else:
        seeds = _parse_seeds(args.seeds)
        if args.population_size < 1:
            raise SystemExit("--population-size must be >= 1")
        ticks = int(args.ticks)
        population_size = int(args.population_size)
        search_seed = int(args.search_seed)
        rng = Random(search_seed)
        candidates = _founder_candidates(
            population_size=population_size,
            rng=rng,
            controller_architecture=str(args.controller_architecture),
        )
        warm_start_report = _warm_start_report_from_args(
            args,
            population_size=population_size,
            candidates=candidates,
        )
        candidates = _inject_warm_start_candidates(
            candidates,
            warm_start_candidates=(
                list(warm_start_report["candidates"])
                if warm_start_report is not None
                else []
            ),
        )
        generations: list[dict[str, object]] = []
        start_generation_index = 0
        resumed_from = None
        holdout_seeds = _parse_optional_seeds(args.holdout_seeds)
    if args.resume_from is not None:
        warm_start_report = None
    report = _run_search_stage(
        search_seed=search_seed,
        seeds=seeds,
        holdout_seeds=holdout_seeds,
        ticks=ticks,
        population_size=population_size,
        requested_generations=args.generations,
        rollout_workers=int(args.rollout_workers),
        resumed_from=resumed_from,
        generations=generations,
        start_generation_index=start_generation_index,
        candidates=candidates,
        rng=rng,
        checkpoint_output=args.output,
        fixture_config=_fixture_config_from_args(
            args,
            search_seeds=seeds,
            holdout_seeds=holdout_seeds,
            default_ticks=ticks,
        ),
        fixture_rerank_top_k=int(args.fixture_rerank_top_k),
        fixture_rerank_ticks=_fixture_rerank_ticks_from_args(
            args,
            default_ticks=ticks,
        ),
        fixture_selection_top_k=(
            int(args.fixture_selection_top_k)
            if args.fixture_suite != "none"
            else 0
        ),
        fixture_rerank_selector_probe=args.fixture_rerank_selector_probe,
        fixture_rerank_selector_probe_scope=args.fixture_rerank_selector_probe_scope,
        fixture_rerank_selector_probe_trajectory_output_dir=(
            _fixture_rerank_selector_probe_trajectory_output_dir(args)
        ),
        fixture_recovery_archive_retention=args.fixture_recovery_archive_retention,
        warm_start=warm_start_report,
    )
    _attach_comparison_baseline_report(
        report,
        args.comparison_baseline_report,
    )
    _write_report(args.output, report)
    _print_report_summary(args.output, report)


def _run_search_stage(
    *,
    search_seed: int,
    seeds: list[int],
    holdout_seeds: list[int],
    ticks: int,
    population_size: int,
    requested_generations: int,
    rollout_workers: int,
    resumed_from: str | None,
    generations: list[dict[str, object]],
    start_generation_index: int,
    candidates: list[dict[str, object]],
    rng: Random,
    checkpoint_output: Path | None,
    fixture_config: dict[str, object] | None = None,
    fixture_rerank_top_k: int = 1,
    fixture_rerank_ticks: list[int] | None = None,
    fixture_selection_top_k: int = 0,
    fixture_rerank_selector_probe: str | None = None,
    fixture_rerank_selector_probe_scope: str = (
        MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY
    ),
    fixture_rerank_selector_probe_trajectory_output_dir: Path | None = None,
    fixture_recovery_archive_retention: str | None = None,
    warm_start: dict[str, object] | None = None,
) -> dict[str, object]:
    if start_generation_index > requested_generations:
        raise SystemExit(
            "--generations must be greater than or equal to completed generations"
        )
    if not candidates and start_generation_index < requested_generations:
        raise SystemExit("search state does not contain next candidates")
    archive = _archive_from_generations(generations)
    best_candidate = _best_candidate_from_archive(archive)
    next_candidates = list(candidates)
    for generation_index in range(start_generation_index, requested_generations):
        evaluated = _evaluate_generation_candidates(
            candidates,
            generation_index=generation_index,
            seeds=seeds,
            ticks=ticks,
            rollout_workers=rollout_workers,
        )
        if fixture_config is not None and fixture_selection_top_k > 0:
            evaluated = _apply_generation_fixture_selection_pressure(
                evaluated,
                limit=fixture_selection_top_k,
                ticks=ticks,
                rollout_workers=rollout_workers,
                fixture_config=fixture_config,
                fixture_ticks=fixture_rerank_ticks
                or [int(fixture_config.get("ticks") or ticks)],
            )
        generation_best = max(
            evaluated,
            key=lambda candidate: (
                float(candidate["score"]),
                float(candidate["alive_agents_mean"]),
                float(candidate["births_mean"]),
                -int(candidate["candidate_index"]),
            ),
        )
        generation_report = {
            "generation_index": generation_index,
            "best_candidate_id": generation_best["candidate_id"],
            "best_score": generation_best["score"],
            "candidates": evaluated,
            "archive": _archive_from_candidates(evaluated),
        }
        fixture_selection_summary = _fixture_selection_generation_summary(
            evaluated
        )
        if fixture_selection_summary is not None:
            generation_report["fixture_selection"] = fixture_selection_summary
        generations.append(generation_report)
        archive = _archive_from_generations(generations)
        best_candidate = _best_candidate_from_archive(archive)
        if best_candidate is None:
            raise SystemExit("no candidates were evaluated")
        next_candidates = _next_generation(
            _archive_parent_candidates(archive),
            generation_index=generation_index + 1,
            population_size=population_size,
            rng=rng,
        )
        candidates = next_candidates
        if checkpoint_output is not None:
            _write_report(
                checkpoint_output,
                _build_report(
                    search_seed=search_seed,
                    seeds=seeds,
                    holdout_seeds=holdout_seeds,
                    ticks=ticks,
                    population_size=population_size,
                    requested_generations=requested_generations,
                    rollout_workers=rollout_workers,
                    resumed_from=resumed_from,
                    generations=generations,
                    archive=archive,
                    best_candidate=best_candidate,
                    next_candidates=next_candidates,
                    rng=rng,
                    holdout_evaluation=None,
                    fixture_suite=None,
                    fixture_gate=None,
                    fixture_rerank=None,
                    fixture_selection_top_k=fixture_selection_top_k,
                    warm_start=warm_start,
                ),
            )
    if best_candidate is None:
        raise SystemExit("no candidates were evaluated")
    best_candidate = _candidate_with_founder_template_pool(
        best_candidate,
        archive=archive,
        candidates=_generation_candidates(generations),
    )
    fixture_rerank = None
    if fixture_config is not None and fixture_rerank_top_k > 1:
        rerank = _run_fixture_rerank(
            archive=archive,
            generations=generations,
            limit=fixture_rerank_top_k,
            holdout_seeds=holdout_seeds,
            ticks=ticks,
            rollout_workers=rollout_workers,
            fixture_config=fixture_config,
            fixture_ticks=fixture_rerank_ticks or [int(fixture_config.get("ticks") or ticks)],
            selector_probe=fixture_rerank_selector_probe,
            selector_probe_scope=fixture_rerank_selector_probe_scope,
            selector_probe_trajectory_output_dir=(
                fixture_rerank_selector_probe_trajectory_output_dir
            ),
            recovery_archive_retention=fixture_recovery_archive_retention,
        )
        best_candidate = dict(rerank["selected_candidate"])
        holdout_evaluation = rerank["holdout_evaluation"]
        fixture_suite = rerank["fixture_suite"]
        fixture_gate = rerank["fixture_gate"]
        fixture_rerank = dict(rerank["report"])
        retention = rerank.get("recovery_archive_retention")
        if isinstance(retention, Mapping):
            retention_archive = _archive_with_recovery_archive_retention(
                archive,
                retention=retention,
            )
            retention_report = retention.get("report")
            retention_added_count = (
                int(retention_report.get("retention_added_count", 0))
                if isinstance(retention_report, Mapping)
                else 0
            )
            if retention_added_count > 0:
                archive = retention_archive
                next_candidates = _next_generation(
                    _archive_parent_candidates(archive),
                    generation_index=requested_generations,
                    population_size=population_size,
                    rng=rng,
                )
            else:
                archive = retention_archive
    else:
        holdout_evaluation = (
            _evaluate_holdout(
                seeds=holdout_seeds,
                ticks=ticks,
                metadata=_best_candidate_founder_template(best_candidate),
                rollout_workers=rollout_workers,
            )
            if holdout_seeds
            else None
        )
        fixture_suite = None
        fixture_gate = None
        if fixture_config is not None:
            fixture_suite = run_mind_v3_fixture_suite(
                suite=str(fixture_config["suite"]),
                fixture_names=_fixture_names_from_config(fixture_config),
                seeds=[int(seed) for seed in list(fixture_config["seeds"])],
                ticks=int(fixture_config.get("ticks") or ticks),
                founder_template=_best_candidate_founder_template(best_candidate),
            )
            fixture_gate = mind_v3_fixture_gate_status(
                fixture_suite=fixture_suite,
                fixture_config=fixture_config,
            )
    report = _build_report(
        search_seed=search_seed,
        seeds=seeds,
        holdout_seeds=holdout_seeds,
        ticks=ticks,
        population_size=population_size,
        requested_generations=requested_generations,
        rollout_workers=rollout_workers,
        resumed_from=resumed_from,
        generations=generations,
        archive=archive,
        best_candidate=best_candidate,
        next_candidates=next_candidates,
        rng=rng,
        holdout_evaluation=holdout_evaluation,
        fixture_suite=fixture_suite,
        fixture_gate=fixture_gate,
        fixture_rerank=fixture_rerank,
        warm_start=warm_start,
        fixture_selection_top_k=fixture_selection_top_k,
    )
    return report


def _run_curriculum_search(args: argparse.Namespace) -> dict[str, object]:
    if args.resume_from is not None:
        raise SystemExit("--curriculum-ticks cannot be combined with --resume-from")
    if args.population_size < 1:
        raise SystemExit("--population-size must be >= 1")
    seeds = _parse_seeds(args.seeds)
    holdout_seeds = _parse_optional_seeds(args.holdout_seeds)
    if not holdout_seeds:
        raise SystemExit("--curriculum-ticks requires --holdout-seeds")
    fixture_config = _fixture_config_from_args(
        args,
        search_seeds=seeds,
        holdout_seeds=holdout_seeds,
        default_ticks=None,
    )
    tick_schedule = _parse_positive_ints(
        args.curriculum_ticks,
        flag_name="--curriculum-ticks",
    )
    search_seed = int(args.search_seed)
    rng = Random(search_seed)
    candidates = _founder_candidates(
        population_size=args.population_size,
        rng=rng,
        controller_architecture=str(args.controller_architecture),
    )
    warm_start_report = _warm_start_report_from_args(
        args,
        population_size=int(args.population_size),
        candidates=candidates,
    )
    candidates = _inject_warm_start_candidates(
        candidates,
        warm_start_candidates=(
            list(warm_start_report["candidates"])
            if warm_start_report is not None
            else []
        ),
    )
    stages: list[dict[str, object]] = []
    selected_report: dict[str, object] | None = None
    selected_stage_index: int | None = None
    selected_stage_passed = False
    stopped_reason = "complete"

    for stage_index, ticks in enumerate(tick_schedule):
        stage_report = _run_search_stage(
            search_seed=search_seed,
            seeds=seeds,
            holdout_seeds=holdout_seeds,
            ticks=ticks,
            population_size=int(args.population_size),
            requested_generations=int(args.generations),
            rollout_workers=int(args.rollout_workers),
            resumed_from=None,
            generations=[],
            start_generation_index=0,
            candidates=candidates,
            rng=rng,
            checkpoint_output=None,
            fixture_config=fixture_config,
            fixture_rerank_top_k=int(args.fixture_rerank_top_k),
            fixture_rerank_ticks=_fixture_rerank_ticks_from_args(
                args,
                default_ticks=ticks,
            ),
            fixture_selection_top_k=(
                int(args.fixture_selection_top_k)
                if args.fixture_suite != "none"
                else 0
            ),
            fixture_rerank_selector_probe=args.fixture_rerank_selector_probe,
            fixture_rerank_selector_probe_scope=(
                args.fixture_rerank_selector_probe_scope
            ),
            fixture_rerank_selector_probe_trajectory_output_dir=(
                _fixture_rerank_selector_probe_trajectory_output_dir(args)
            ),
            fixture_recovery_archive_retention=args.fixture_recovery_archive_retention,
            warm_start=warm_start_report if stage_index == 0 else None,
        )
        stage_status = _curriculum_stage_status(
            stage_report=stage_report,
            stage_index=stage_index,
            ticks=ticks,
            min_holdout_alive=float(args.curriculum_min_holdout_alive),
            min_holdout_births=float(args.curriculum_min_holdout_births),
            min_holdout_movement_rate=float(
                args.curriculum_min_holdout_movement_rate
            ),
            min_holdout_unique_actions=float(
                args.curriculum_min_holdout_unique_actions
            ),
            min_holdout_behavior_niches=int(
                args.curriculum_min_holdout_behavior_niches
            ),
            max_holdout_dominant_action_share=float(
                args.curriculum_max_holdout_dominant_action_share
            ),
            min_holdout_active_behavior_niches=int(
                args.curriculum_min_holdout_active_behavior_niches
            ),
            min_holdout_energy_viability=float(
                args.curriculum_min_holdout_energy_viability
            ),
            min_holdout_hydration_viability=float(
                args.curriculum_min_holdout_hydration_viability
            ),
            min_holdout_health_viability=float(
                args.curriculum_min_holdout_health_viability
            ),
            min_holdout_matched_diet_viability=float(
                args.curriculum_min_holdout_matched_diet_viability
            ),
            min_holdout_biologically_ready=float(
                args.curriculum_min_holdout_biologically_ready
            ),
            min_holdout_ready_agent_tick_share=float(
                args.curriculum_min_holdout_ready_agent_tick_share
            ),
            min_holdout_ready_pair_tick_share=float(
                args.curriculum_min_holdout_ready_pair_tick_share
            ),
        )
        stage_report["curriculum_stage"] = stage_status
        stages.append(stage_report)

        if bool(stage_status["passed"]):
            selected_report = stage_report
            selected_stage_index = stage_index
            selected_stage_passed = True
        else:
            stopped_reason = str(stage_status["stopped_reason"])
            if selected_report is None:
                selected_report = stage_report
                selected_stage_index = stage_index
                selected_stage_passed = False
            break

        if stage_index < len(tick_schedule) - 1:
            candidates = _next_generation(
                _archive_parent_candidates(dict(stage_report["archive"])),
                generation_index=0,
                population_size=int(args.population_size),
                rng=rng,
            )

    if selected_report is None or selected_stage_index is None:
        raise SystemExit("curriculum search did not evaluate any stages")
    final_report = dict(selected_report)
    final_report["curriculum"] = {
        "policy": MIND_V3_CURRICULUM_POLICY,
        "tick_schedule": tick_schedule,
        "completed_stage_count": len(stages),
        "passed_stage_count": sum(
            1
            for stage in stages
            if bool(dict(stage.get("curriculum_stage", {})).get("passed", False))
        ),
        "selected_stage_index": selected_stage_index,
        "selected_stage_passed": selected_stage_passed,
        "stopped_reason": stopped_reason,
        "min_holdout_alive": float(args.curriculum_min_holdout_alive),
        "min_holdout_births": float(args.curriculum_min_holdout_births),
        "min_holdout_movement_rate": float(
            args.curriculum_min_holdout_movement_rate
        ),
        "min_holdout_unique_actions": float(
            args.curriculum_min_holdout_unique_actions
        ),
        "min_holdout_behavior_niches": int(
            args.curriculum_min_holdout_behavior_niches
        ),
        "max_holdout_dominant_action_share": float(
            args.curriculum_max_holdout_dominant_action_share
        ),
        "min_holdout_active_behavior_niches": int(
            args.curriculum_min_holdout_active_behavior_niches
        ),
        "min_holdout_energy_viability": float(
            args.curriculum_min_holdout_energy_viability
        ),
        "min_holdout_hydration_viability": float(
            args.curriculum_min_holdout_hydration_viability
        ),
        "min_holdout_health_viability": float(
            args.curriculum_min_holdout_health_viability
        ),
        "min_holdout_matched_diet_viability": float(
            args.curriculum_min_holdout_matched_diet_viability
        ),
        "min_holdout_biologically_ready": float(
            args.curriculum_min_holdout_biologically_ready
        ),
        "min_holdout_ready_agent_tick_share": float(
            args.curriculum_min_holdout_ready_agent_tick_share
        ),
        "min_holdout_ready_pair_tick_share": float(
            args.curriculum_min_holdout_ready_pair_tick_share
        ),
        "fixture_gate_enabled": fixture_config is not None,
        "fixture_rerank_top_k": int(args.fixture_rerank_top_k),
        "fixture_selection_top_k": (
            int(args.fixture_selection_top_k)
            if fixture_config is not None
            else 0
        ),
    }
    final_report["stages"] = stages
    return final_report


def _curriculum_stage_status(
    *,
    stage_report: dict[str, object],
    stage_index: int,
    ticks: int,
    min_holdout_alive: float,
    min_holdout_births: float,
    min_holdout_movement_rate: float,
    min_holdout_unique_actions: float,
    min_holdout_behavior_niches: int,
    max_holdout_dominant_action_share: float = 1.0,
    min_holdout_active_behavior_niches: int = 0,
    min_holdout_energy_viability: float = 0.0,
    min_holdout_hydration_viability: float = 0.0,
    min_holdout_health_viability: float = 0.0,
    min_holdout_matched_diet_viability: float = 0.0,
    min_holdout_biologically_ready: float = 0.0,
    min_holdout_ready_agent_tick_share: float = 0.0,
    min_holdout_ready_pair_tick_share: float = 0.0,
) -> dict[str, object]:
    holdout = stage_report.get("holdout_evaluation")
    if not isinstance(holdout, dict):
        raise SystemExit("curriculum stage is missing holdout evaluation")
    aggregate = holdout.get("aggregate")
    if not isinstance(aggregate, dict):
        raise SystemExit("curriculum stage is missing holdout aggregate")
    alive_mean = float(aggregate["alive_agents_mean"])
    births_mean = float(aggregate["births_mean"])
    movement_rate = float(aggregate.get("movement_event_rate", 0.0))
    unique_actions_mean = float(aggregate.get("unique_requested_actions_mean", 0.0))
    behavior_niche_count = int(aggregate.get("behavior_niche_count", 0))
    active_behavior_niche_count = int(
        aggregate.get("active_behavior_niche_count", 0)
    )
    dominant_requested_action = aggregate.get("dominant_requested_action")
    dominant_action_share = float(
        aggregate.get("dominant_requested_action_share", 0.0)
    )
    terminal_energy_viability = float(
        aggregate.get("terminal_energy_viability_share_mean", 0.0)
    )
    terminal_energy_requirement_satisfaction = float(
        aggregate.get("terminal_energy_requirement_satisfaction_mean", 0.0)
    )
    terminal_balanced_readiness = float(
        aggregate.get(
            "terminal_balanced_reproduction_readiness_mean",
            _balanced_readiness_score(
                [
                    terminal_energy_requirement_satisfaction,
                    float(
                        aggregate.get(
                            "terminal_hydration_viability_share_mean",
                            0.0,
                        )
                    ),
                    float(aggregate.get("terminal_health_viability_share_mean", 0.0)),
                    float(
                        aggregate.get(
                            "terminal_matched_diet_viability_share_mean",
                            0.0,
                        )
                    ),
                ]
            ),
        )
    )
    terminal_energy_hydration_balance = float(
        aggregate.get(
            "terminal_energy_hydration_balance_mean",
            min(
                terminal_energy_requirement_satisfaction,
                float(
                    aggregate.get("terminal_hydration_viability_share_mean", 0.0)
                ),
            ),
        )
    )
    terminal_energy_hydration_gap_abs = float(
        aggregate.get(
            "terminal_energy_hydration_gap_abs_mean",
            abs(
                terminal_energy_requirement_satisfaction
                - float(
                    aggregate.get("terminal_hydration_viability_share_mean", 0.0)
                )
            ),
        )
    )
    terminal_hydration_viability = float(
        aggregate.get("terminal_hydration_viability_share_mean", 0.0)
    )
    terminal_health_viability = float(
        aggregate.get("terminal_health_viability_share_mean", 0.0)
    )
    terminal_matched_diet_viability = float(
        aggregate.get("terminal_matched_diet_viability_share_mean", 0.0)
    )
    biologically_ready_mean = float(
        aggregate.get("biologically_reproduction_ready_agents_mean", 0.0)
    )
    ready_agent_tick_share = float(
        aggregate.get("reproduction_ready_agent_tick_share_mean", 0.0)
    )
    ready_pair_tick_share = float(
        aggregate.get("reproduction_ready_pair_tick_share_mean", 0.0)
    )
    stopped_reason: str | None = None
    if alive_mean < min_holdout_alive:
        stopped_reason = "holdout_alive_floor"
    elif births_mean < min_holdout_births:
        stopped_reason = "holdout_birth_floor"
    elif movement_rate < min_holdout_movement_rate:
        stopped_reason = "holdout_movement_floor"
    elif unique_actions_mean < min_holdout_unique_actions:
        stopped_reason = "holdout_action_diversity_floor"
    elif dominant_action_share > max_holdout_dominant_action_share:
        stopped_reason = "holdout_dominant_action_share_ceiling"
    elif behavior_niche_count < min_holdout_behavior_niches:
        stopped_reason = "holdout_behavior_niche_floor"
    elif active_behavior_niche_count < min_holdout_active_behavior_niches:
        stopped_reason = "holdout_active_behavior_niche_floor"
    elif terminal_energy_viability < min_holdout_energy_viability:
        stopped_reason = "holdout_energy_viability_floor"
    elif terminal_hydration_viability < min_holdout_hydration_viability:
        stopped_reason = "holdout_hydration_viability_floor"
    elif terminal_health_viability < min_holdout_health_viability:
        stopped_reason = "holdout_health_viability_floor"
    elif (
        terminal_matched_diet_viability
        < min_holdout_matched_diet_viability
    ):
        stopped_reason = "holdout_matched_diet_viability_floor"
    elif biologically_ready_mean < min_holdout_biologically_ready:
        stopped_reason = "holdout_biological_reproduction_readiness_floor"
    elif ready_agent_tick_share < min_holdout_ready_agent_tick_share:
        stopped_reason = "holdout_sustained_ready_agent_tick_floor"
    elif ready_pair_tick_share < min_holdout_ready_pair_tick_share:
        stopped_reason = "holdout_sustained_ready_pair_tick_floor"
    fixture_gate = stage_report.get("fixture_gate")
    fixture_gate_passed = None
    fixture_gate_blockers: list[object] = []
    if isinstance(fixture_gate, Mapping):
        fixture_gate_passed = bool(fixture_gate.get("passed", False))
        raw_blockers = fixture_gate.get("blockers", [])
        if isinstance(raw_blockers, list):
            fixture_gate_blockers = list(raw_blockers)
        if stopped_reason is None and not fixture_gate_passed:
            stopped_reason = "fixture_gate_floor"
    return {
        "stage_index": stage_index,
        "ticks": ticks,
        "holdout_alive_agents_mean": alive_mean,
        "holdout_births_mean": births_mean,
        "holdout_movement_event_rate": movement_rate,
        "holdout_unique_requested_actions_mean": unique_actions_mean,
        "holdout_dominant_requested_action": (
            str(dominant_requested_action)
            if isinstance(dominant_requested_action, str)
            else None
        ),
        "holdout_dominant_requested_action_share": dominant_action_share,
        "holdout_behavior_niche_count": behavior_niche_count,
        "holdout_active_behavior_niche_count": active_behavior_niche_count,
        "holdout_terminal_energy_viability_share_mean": (
            terminal_energy_viability
        ),
        "holdout_terminal_hydration_viability_share_mean": (
            terminal_hydration_viability
        ),
        "holdout_terminal_health_viability_share_mean": (
            terminal_health_viability
        ),
        "holdout_terminal_matched_diet_viability_share_mean": (
            terminal_matched_diet_viability
        ),
        "holdout_biologically_reproduction_ready_agents_mean": (
            biologically_ready_mean
        ),
        "holdout_reproduction_ready_agent_tick_share_mean": (
            ready_agent_tick_share
        ),
        "holdout_reproduction_ready_pair_tick_share_mean": (
            ready_pair_tick_share
        ),
        "min_holdout_alive": min_holdout_alive,
        "min_holdout_births": min_holdout_births,
        "min_holdout_movement_rate": min_holdout_movement_rate,
        "min_holdout_unique_actions": min_holdout_unique_actions,
        "min_holdout_behavior_niches": min_holdout_behavior_niches,
        "max_holdout_dominant_action_share": max_holdout_dominant_action_share,
        "min_holdout_active_behavior_niches": (
            min_holdout_active_behavior_niches
        ),
        "min_holdout_energy_viability": min_holdout_energy_viability,
        "min_holdout_hydration_viability": (
            min_holdout_hydration_viability
        ),
        "min_holdout_health_viability": min_holdout_health_viability,
        "min_holdout_matched_diet_viability": (
            min_holdout_matched_diet_viability
        ),
        "min_holdout_biologically_ready": min_holdout_biologically_ready,
        "min_holdout_ready_agent_tick_share": (
            min_holdout_ready_agent_tick_share
        ),
        "min_holdout_ready_pair_tick_share": (
            min_holdout_ready_pair_tick_share
        ),
        "fixture_gate_passed": fixture_gate_passed,
        "fixture_gate_blockers": fixture_gate_blockers,
        "passed": stopped_reason is None,
        "stopped_reason": stopped_reason,
    }


def _print_report_summary(path: Path, report: dict[str, object]) -> None:
    best_candidate = dict(report["best_candidate"])
    print(f"mind_v3_evolution_report={path}")
    print(f"best_candidate_id={best_candidate['candidate_id']}")
    print(f"best_score={best_candidate['score']}")
    print(
        "best_heuristic_action_source_count="
        f"{best_candidate['heuristic_action_source_count']}"
    )
    print(f"best_alive_agents_mean={best_candidate['alive_agents_mean']}")
    print(f"best_births_mean={best_candidate['births_mean']}")
    fixture_gate = report.get("fixture_gate")
    if isinstance(fixture_gate, dict):
        print(f"fixture_gate_passed={fixture_gate['passed']}")


def _founder_candidates(
    *,
    population_size: int,
    rng: Random,
    controller_architecture: str = MIND_V3_CONTROLLER_ARCHITECTURE,
) -> list[dict[str, object]]:
    return [
        _new_candidate(
            candidate_id=f"g0-c{index}",
            metadata=founder_mind_v3_metadata(
                agent_id=index,
                rng=rng,
                architecture=controller_architecture,
                specialization_profile=MIND_V3_SPECIALIZATION_PROFILES[
                    index % len(MIND_V3_SPECIALIZATION_PROFILES)
                ],
            ),
            parent_candidate_id=None,
        )
        for index in range(population_size)
    ]


def _warm_start_report_from_args(
    args: argparse.Namespace,
    *,
    population_size: int,
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    paths = [Path(path) for path in list(args.warm_start_report or [])]
    if not paths:
        return None
    imported = _warm_start_candidates_from_reports(
        paths,
        candidate_limit=int(args.warm_start_candidate_limit),
        population_size=population_size,
        start_index=max(
            0,
            min(
                len(candidates),
                population_size
                - min(
                    int(args.warm_start_candidate_limit),
                    population_size,
                ),
            ),
        ),
    )
    return {
        "policy": MIND_V3_WARM_START_POLICY,
        "source_reports": [str(path) for path in paths],
        "candidate_limit": int(args.warm_start_candidate_limit),
        "candidate_count": len(imported),
        "candidates": imported,
    }


def _inject_warm_start_candidates(
    candidates: list[dict[str, object]],
    *,
    warm_start_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not warm_start_candidates:
        return list(candidates)
    if len(warm_start_candidates) > len(candidates):
        return list(warm_start_candidates[: len(candidates)])
    keep_count = len(candidates) - len(warm_start_candidates)
    return [*candidates[:keep_count], *warm_start_candidates]


def _warm_start_candidates_from_reports(
    paths: list[Path],
    *,
    candidate_limit: int,
    population_size: int,
    start_index: int,
) -> list[dict[str, object]]:
    if candidate_limit <= 0 or population_size <= 0:
        return []
    imported: list[dict[str, object]] = []
    seen_fingerprints: set[str] = set()
    report_sources = [
        (path, _warm_start_sources_from_report(_load_warm_start_report(path)))
        for path in paths
    ]
    max_source_count = max((len(sources) for _, sources in report_sources), default=0)
    for source_rank in range(max_source_count):
        for path, sources in report_sources:
            if len(imported) >= min(candidate_limit, population_size):
                return _renumber_warm_start_candidates(
                    imported,
                    start_index=start_index,
                )
            if source_rank >= len(sources):
                continue
            source = sources[source_rank]
            metadata = source.get("controller_metadata")
            if not isinstance(metadata, dict):
                continue
            fingerprint = _metadata_fingerprint(metadata)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            candidate = {
                "candidate_id": "",
                "parent_candidate_id": source.get("source_candidate_id"),
                "controller_metadata": dict(metadata),
                "warm_start": {
                    "policy": MIND_V3_WARM_START_POLICY,
                    "source_report": str(path),
                    "source_candidate_id": source.get("source_candidate_id"),
                    "source_kind": source.get("source_kind"),
                    "source_rank": source_rank,
                    "selection_rank": len(imported),
                },
            }
            pool = source.get("founder_template_pool")
            if isinstance(pool, list) and pool:
                candidate["founder_template_pool"] = [
                    dict(template)
                    for template in pool
                    if isinstance(template, dict)
                ]
            imported.append(candidate)
    return _renumber_warm_start_candidates(imported, start_index=start_index)


def _renumber_warm_start_candidates(
    candidates: list[dict[str, object]],
    *,
    start_index: int,
) -> list[dict[str, object]]:
    renumbered: list[dict[str, object]] = []
    for offset, candidate in enumerate(candidates):
        updated = dict(candidate)
        candidate_index = int(start_index) + offset
        updated["candidate_id"] = f"g0-c{candidate_index}"
        warm_start = dict(updated.get("warm_start", {}))
        warm_start["candidate_index"] = candidate_index
        updated["warm_start"] = warm_start
        renumbered.append(updated)
    return renumbered


def _load_warm_start_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--warm-start-report must contain a JSON object")
    if payload.get("schema_version") != MIND_V3_EVOLUTION_SEARCH_SCHEMA_VERSION:
        raise SystemExit("--warm-start-report must point to a Mind v3 evolution report")
    return payload


def _warm_start_sources_from_report(
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    by_id = _report_generation_candidates_by_id(report)
    ordered_ids = _warm_start_fixture_candidate_ids(report)
    sources: list[dict[str, object]] = []
    for candidate_id in ordered_ids:
        candidate = by_id.get(candidate_id)
        if candidate is not None:
            sources.append(
                _warm_start_source_from_candidate(
                    candidate,
                    source_kind="fixture_rerank",
                )
            )
    best_candidate = report.get("best_candidate")
    if isinstance(best_candidate, Mapping):
        sources.append(
            _warm_start_source_from_candidate(
                best_candidate,
                source_kind="best_candidate",
            )
        )
    archive = report.get("archive")
    if isinstance(archive, Mapping):
        elites = archive.get("elites")
        if isinstance(elites, Mapping):
            for name in ARCHIVE_ORDER:
                candidate = elites.get(name)
                if isinstance(candidate, Mapping):
                    sources.append(
                        _warm_start_source_from_candidate(
                            candidate,
                            source_kind=f"archive:{name}",
                        )
                    )
        niches = archive.get("behavior_niches")
        if isinstance(niches, Mapping):
            for niche_key in sorted(niches):
                cell = niches[niche_key]
                if not isinstance(cell, Mapping):
                    continue
                elite = cell.get("lineage_elite")
                if isinstance(elite, Mapping):
                    sources.append(
                        _warm_start_source_from_candidate(
                            elite,
                            source_kind=f"behavior_niche:{niche_key}",
                        )
                    )
    return sources


def _warm_start_fixture_candidate_ids(report: Mapping[str, object]) -> list[str]:
    fixture_rerank = report.get("fixture_rerank")
    payload = fixture_rerank if isinstance(fixture_rerank, Mapping) else {}
    raw_candidates = payload.get("candidates")
    entries = [
        entry for entry in raw_candidates
        if isinstance(entry, Mapping)
    ] if isinstance(raw_candidates, list) else []
    entries.sort(key=_fixture_rerank_selection_key, reverse=True)
    selected = payload.get("selected_candidate_id")
    ordered: list[str] = []
    if isinstance(selected, str) and selected:
        ordered.append(selected)
    for entry in entries:
        candidate_id = entry.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id and candidate_id not in ordered:
            ordered.append(candidate_id)
    return ordered


def _report_generation_candidates_by_id(
    report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    by_id: dict[str, Mapping[str, object]] = {}
    generations = report.get("generations")
    if not isinstance(generations, list):
        return by_id
    for generation in generations:
        if not isinstance(generation, Mapping):
            continue
        candidates = generation.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            candidate_id = candidate.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                by_id[candidate_id] = candidate
    return by_id


def _warm_start_source_from_candidate(
    candidate: Mapping[str, object],
    *,
    source_kind: str,
) -> dict[str, object]:
    source: dict[str, object] = {
        "source_kind": source_kind,
        "source_candidate_id": candidate.get("candidate_id"),
    }
    metadata = candidate.get("controller_metadata")
    if isinstance(metadata, dict):
        source["controller_metadata"] = dict(metadata)
    pool = candidate.get("founder_template_pool")
    if isinstance(pool, list) and pool:
        source["founder_template_pool"] = [
            dict(template)
            for template in pool
            if isinstance(template, dict)
        ]
    return source


def _new_candidate(
    *,
    candidate_id: str,
    metadata: dict[str, object],
    parent_candidate_id: str | None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_candidate_id,
        "controller_metadata": metadata,
    }


def _copy_founder_template(
    metadata: dict[str, object] | list[dict[str, object]],
) -> dict[str, object] | list[dict[str, object]]:
    if isinstance(metadata, list):
        return [dict(template) for template in metadata if isinstance(template, dict)]
    return dict(metadata)


def _next_generation(
    parent_candidates: list[dict[str, object]],
    *,
    generation_index: int,
    population_size: int,
    rng: Random,
) -> list[dict[str, object]]:
    if not parent_candidates:
        raise ValueError("at least one parent candidate is required")
    balanced_parent = parent_candidates[0]
    next_candidates = [
        _new_candidate(
            candidate_id=f"g{generation_index}-c0",
            metadata=dict(balanced_parent["controller_metadata"]),
            parent_candidate_id=str(balanced_parent["candidate_id"]),
        )
    ]
    for index in range(1, population_size):
        parent = parent_candidates[(index - 1) % len(parent_candidates)]
        parent_metadata = _parent_controller_metadata_for_child(
            parent,
            child_index=index,
        )
        next_candidates.append(
            _new_candidate(
                candidate_id=f"g{generation_index}-c{index}",
                metadata=inherit_mind_v3_metadata(
                    primary_parent_metadata=parent_metadata,
                    secondary_parent_metadata=None,
                    child_agent_id=generation_index * 10_000 + index,
                    rng=rng,
                ),
                parent_candidate_id=str(parent["candidate_id"]),
            )
        )
    return next_candidates


def _evaluate_candidate(
    candidate: dict[str, object],
    *,
    generation_index: int,
    seeds: list[int],
    ticks: int,
) -> dict[str, object]:
    runs = [
        _run_candidate(
            seed=seed,
            ticks=ticks,
            metadata=_candidate_evaluation_template(candidate),
        )
        for seed in seeds
    ]
    aggregate = _aggregate_runs(runs)
    score_components = _score_components(aggregate)
    score = _score_components_total(score_components)
    candidate_index = int(str(candidate["candidate_id"]).split("-c")[-1])
    controller_lineage_elites = _candidate_controller_lineage_elites(runs)
    selected_controller_metadata = _selected_controller_metadata(
        fallback=dict(candidate["controller_metadata"]),
        controller_lineage_elites=controller_lineage_elites,
    )
    result = {
        "candidate_id": candidate["candidate_id"],
        "candidate_index": candidate_index,
        "parent_candidate_id": candidate["parent_candidate_id"],
        "generation_index": generation_index,
        "score": score,
        "score_components": score_components,
        "alive_agents_mean": aggregate["alive_agents_mean"],
        "births_mean": aggregate["births_mean"],
        "deaths_mean": aggregate["deaths_mean"],
        "alive_agent_ticks_mean": aggregate["alive_agent_ticks_mean"],
        "alive_agent_ticks_per_tick_mean": aggregate[
            "alive_agent_ticks_per_tick_mean"
        ],
        "terminal_energy_viability_share_mean": aggregate[
            "terminal_energy_viability_share_mean"
        ],
        "terminal_energy_requirement_satisfaction_mean": aggregate[
            "terminal_energy_requirement_satisfaction_mean"
        ],
        "terminal_balanced_reproduction_readiness_mean": aggregate[
            "terminal_balanced_reproduction_readiness_mean"
        ],
        "terminal_balanced_reproduction_readiness_min": aggregate[
            "terminal_balanced_reproduction_readiness_min"
        ],
        "terminal_energy_hydration_balance_mean": aggregate[
            "terminal_energy_hydration_balance_mean"
        ],
        "terminal_energy_hydration_balance_min": aggregate[
            "terminal_energy_hydration_balance_min"
        ],
        "terminal_energy_hydration_gap_abs_mean": aggregate[
            "terminal_energy_hydration_gap_abs_mean"
        ],
        "terminal_energy_shortfall_share_mean": aggregate[
            "terminal_energy_shortfall_share_mean"
        ],
        "terminal_hydration_viability_share_mean": aggregate[
            "terminal_hydration_viability_share_mean"
        ],
        "terminal_health_viability_share_mean": aggregate[
            "terminal_health_viability_share_mean"
        ],
        "terminal_matched_diet_viability_share_mean": aggregate[
            "terminal_matched_diet_viability_share_mean"
        ],
        "terminal_reproduction_viability_run_share": aggregate[
            "terminal_reproduction_viability_run_share"
        ],
        "terminal_reproduction_viability_min": aggregate[
            "terminal_reproduction_viability_min"
        ],
        "birth_positive_run_share": aggregate["birth_positive_run_share"],
        "reproduction_ready_agents_mean": aggregate[
            "reproduction_ready_agents_mean"
        ],
        "biologically_reproduction_ready_agents_mean": aggregate[
            "biologically_reproduction_ready_agents_mean"
        ],
        "terminal_biologically_ready_group_count_mean": aggregate[
            "terminal_biologically_ready_group_count_mean"
        ],
        "terminal_ready_group_count_mean": aggregate[
            "terminal_ready_group_count_mean"
        ],
        "core_readiness_agent_tick_mean": aggregate[
            "core_readiness_agent_tick_mean"
        ],
        "core_ready_agent_tick_share_mean": aggregate[
            "core_ready_agent_tick_share_mean"
        ],
        "core_ready_tick_share_mean": aggregate["core_ready_tick_share_mean"],
        "core_ready_pair_tick_share_mean": aggregate[
            "core_ready_pair_tick_share_mean"
        ],
        "core_ready_distinct_agent_count_mean": aggregate[
            "core_ready_distinct_agent_count_mean"
        ],
        "reproduction_ready_agent_tick_share_mean": aggregate[
            "reproduction_ready_agent_tick_share_mean"
        ],
        "reproduction_ready_tick_share_mean": aggregate[
            "reproduction_ready_tick_share_mean"
        ],
        "reproduction_ready_pair_tick_share_mean": aggregate[
            "reproduction_ready_pair_tick_share_mean"
        ],
        "reproduction_ready_distinct_agent_count_mean": aggregate[
            "reproduction_ready_distinct_agent_count_mean"
        ],
        "resource_event_rate": aggregate["resource_event_rate"],
        "animal_resource_event_rate": aggregate["animal_resource_event_rate"],
        "fresh_kill_event_rate": aggregate["fresh_kill_event_rate"],
        "carcass_event_rate": aggregate["carcass_event_rate"],
        "animal_resource_gain_total_mean": aggregate[
            "animal_resource_gain_total_mean"
        ],
        "movement_event_rate": aggregate["movement_event_rate"],
        "unique_requested_actions_mean": aggregate["unique_requested_actions_mean"],
        "requested_action_counts": aggregate["requested_action_counts"],
        "resolved_action_counts": aggregate["resolved_action_counts"],
        "unsupported_requested_action_count": aggregate[
            "unsupported_requested_action_count"
        ],
        "unsupported_resolved_action_count": aggregate[
            "unsupported_resolved_action_count"
        ],
        "unsupported_requested_action_breakdown": aggregate[
            "unsupported_requested_action_breakdown"
        ],
        "unsupported_resolved_action_breakdown": aggregate[
            "unsupported_resolved_action_breakdown"
        ],
        "rollout_context_decision_count": aggregate[
            "rollout_context_decision_count"
        ],
        "rollout_context_non_empty_count": aggregate[
            "rollout_context_non_empty_count"
        ],
        "rollout_context_non_empty_share": aggregate[
            "rollout_context_non_empty_share"
        ],
        "rollout_context_selected_score_delta_nonzero_count": aggregate[
            "rollout_context_selected_score_delta_nonzero_count"
        ],
        "rollout_context_selected_score_delta_mean": aggregate[
            "rollout_context_selected_score_delta_mean"
        ],
        "rollout_context_selected_score_delta_abs_mean": aggregate[
            "rollout_context_selected_score_delta_abs_mean"
        ],
        "rollout_context_selected_score_delta_abs_max": aggregate[
            "rollout_context_selected_score_delta_abs_max"
        ],
        "rollout_context_post_carrion_context_count": aggregate[
            "rollout_context_post_carrion_context_count"
        ],
        "rollout_context_post_carrion_context_share": aggregate[
            "rollout_context_post_carrion_context_share"
        ],
        "rollout_context_selected_score_delta_by_requested_action": aggregate[
            "rollout_context_selected_score_delta_by_requested_action"
        ],
        "recovery_context_decision_count": aggregate[
            "recovery_context_decision_count"
        ],
        "recovery_context_non_empty_count": aggregate[
            "recovery_context_non_empty_count"
        ],
        "recovery_context_non_empty_share": aggregate[
            "recovery_context_non_empty_share"
        ],
        "recovery_context_selected_score_delta_nonzero_count": aggregate[
            "recovery_context_selected_score_delta_nonzero_count"
        ],
        "recovery_context_selected_score_delta_mean": aggregate[
            "recovery_context_selected_score_delta_mean"
        ],
        "recovery_context_selected_score_delta_abs_mean": aggregate[
            "recovery_context_selected_score_delta_abs_mean"
        ],
        "recovery_context_selected_score_delta_abs_max": aggregate[
            "recovery_context_selected_score_delta_abs_max"
        ],
        "recovery_context_post_carrion_context_count": aggregate[
            "recovery_context_post_carrion_context_count"
        ],
        "recovery_context_post_carrion_context_share": aggregate[
            "recovery_context_post_carrion_context_share"
        ],
        "recovery_context_drink_available_count": aggregate[
            "recovery_context_drink_available_count"
        ],
        "recovery_context_selected_score_delta_by_requested_action": aggregate[
            "recovery_context_selected_score_delta_by_requested_action"
        ],
        "dominant_requested_action": aggregate["dominant_requested_action"],
        "dominant_requested_action_count": aggregate[
            "dominant_requested_action_count"
        ],
        "dominant_requested_action_share": aggregate[
            "dominant_requested_action_share"
        ],
        "core_blocker_agent_tick_counts": aggregate[
            "core_blocker_agent_tick_counts"
        ],
        "core_blocker_agent_tick_shares": aggregate[
            "core_blocker_agent_tick_shares"
        ],
        "primary_core_blocker_agent_tick_counts": aggregate[
            "primary_core_blocker_agent_tick_counts"
        ],
        "primary_core_blocker_agent_tick_shares": aggregate[
            "primary_core_blocker_agent_tick_shares"
        ],
        "primary_core_blocker": aggregate["primary_core_blocker"],
        "heuristic_action_source_count": aggregate[
            "heuristic_action_source_count"
        ],
        "trajectory_record_count": aggregate["trajectory_record_count"],
        "behavior_descriptors": aggregate["behavior_descriptors"],
        "behavior_niche_count": aggregate["behavior_niche_count"],
        "behavior_niche_keys": aggregate["behavior_niche_keys"],
        "active_behavior_niche_count": aggregate["active_behavior_niche_count"],
        "active_behavior_niche_keys": aggregate["active_behavior_niche_keys"],
        "controller_selection_policy": MIND_V3_CONTROLLER_SELECTION_POLICY,
        "controller_lineage_elites": controller_lineage_elites,
        "action_source_counts": aggregate["action_source_counts"],
        "policy_id_counts": aggregate["policy_id_counts"],
        "runs": runs,
        "founder_template_metadata": candidate["controller_metadata"],
        "controller_metadata": selected_controller_metadata,
    }
    existing_pool = candidate.get("founder_template_pool")
    if isinstance(existing_pool, list) and existing_pool:
        result["founder_template_pool"] = [
            dict(template)
            for template in existing_pool
            if isinstance(template, dict)
        ]
        result["founder_template_pool_size"] = len(result["founder_template_pool"])
        result["founder_template_pool_specialization_profile_counts"] = (
            _metadata_specialization_profile_counts(
                list(result["founder_template_pool"])
            )
        )
        result.update(
            _founder_template_pool_identity_diagnostics(
                list(result["founder_template_pool"])
            )
        )
    warm_start = candidate.get("warm_start")
    if isinstance(warm_start, Mapping):
        result["warm_start"] = dict(warm_start)
    result["ecology_lane_tags"] = _candidate_ecology_lane_tags(result)
    result["scavenger_lane_score"] = _round(
        _candidate_scavenger_lane_signal(result)
    )
    return result


def _candidate_evaluation_template(
    candidate: Mapping[str, object],
) -> dict[str, object] | list[dict[str, object]]:
    pool = candidate.get("founder_template_pool")
    if isinstance(pool, list) and pool:
        templates = [
            dict(template)
            for template in pool
            if isinstance(template, dict)
        ]
        if templates:
            return templates
    metadata = candidate.get("controller_metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    raise ValueError("candidate is missing controller_metadata")


def _evaluate_generation_candidates(
    candidates: list[dict[str, object]],
    *,
    generation_index: int,
    seeds: list[int],
    ticks: int,
    rollout_workers: int,
) -> list[dict[str, object]]:
    worker_count = _resolved_rollout_worker_count(
        rollout_workers,
        work_item_count=len(candidates),
    )
    if worker_count <= 1:
        return [
            _evaluate_candidate(
                candidate,
                generation_index=generation_index,
                seeds=seeds,
                ticks=ticks,
            )
            for candidate in candidates
        ]
    tasks = [
        {
            "candidate": candidate,
            "generation_index": generation_index,
            "seeds": seeds,
            "ticks": ticks,
        }
        for candidate in candidates
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_evaluate_candidate_task, tasks))


def _apply_generation_fixture_selection_pressure(
    candidates: list[dict[str, object]],
    *,
    limit: int,
    ticks: int,
    rollout_workers: int,
    fixture_config: dict[str, object],
    fixture_ticks: list[int],
) -> list[dict[str, object]]:
    if limit <= 0 or not candidates:
        return candidates
    nominees = _fixture_selection_candidate_pool(candidates, limit=limit)
    if not nominees:
        return candidates
    runtimes, worker_count = _evaluate_fixture_rerank_candidates(
        [
            (rank, candidate)
            for rank, candidate in enumerate(nominees)
        ],
        holdout_seeds=[],
        ticks=ticks,
        rollout_workers=rollout_workers,
        fixture_config=fixture_config,
        fixture_ticks=fixture_ticks,
    )
    runtimes_by_candidate_id = {
        str(dict(runtime["candidate"]).get("candidate_id", "")): runtime
        for runtime in runtimes
        if isinstance(runtime.get("candidate"), Mapping)
    }
    updated: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        runtime = runtimes_by_candidate_id.get(candidate_id)
        if runtime is None:
            updated.append(candidate)
            continue
        updated.append(
            _candidate_with_fixture_selection_pressure(
                candidate,
                runtime=runtime,
                worker_count=worker_count,
            )
        )
    return updated


def _fixture_selection_candidate_pool(
    candidates: list[dict[str, object]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()

    def append_candidate(candidate: dict[str, object]) -> None:
        if len(selected) >= limit:
            return
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            return
        seen.add(candidate_id)
        selected.append(candidate)

    if limit <= 0:
        return selected
    ordered_candidates = list(candidates)
    if not ordered_candidates:
        return selected

    append_candidate(
        max(ordered_candidates, key=_fixture_rerank_prefilter_key)
    )
    append_candidate(
        max(ordered_candidates, key=_scavenger_lane_key)
    )
    append_candidate(
        max(ordered_candidates, key=_fixture_current_architecture_key)
    )
    append_candidate(
        max(ordered_candidates, key=_fixture_controlled_readiness_key)
    )
    for candidate in sorted(
        ordered_candidates,
        key=_fixture_rerank_prefilter_key,
        reverse=True,
    ):
        append_candidate(candidate)
        if len(selected) >= limit:
            break
    return selected


def _candidate_with_fixture_selection_pressure(
    candidate: dict[str, object],
    *,
    runtime: Mapping[str, object],
    worker_count: int,
) -> dict[str, object]:
    entry = dict(runtime.get("entry", {}))
    selection_key = _fixture_rerank_selection_key(entry)
    entry["selection_key"] = list(selection_key)
    score_delta = _fixture_selection_score_delta(entry)
    pre_fixture_score = float(candidate.get("score", 0.0))
    raw_components = candidate.get("score_components")
    score_components = (
        dict(raw_components) if isinstance(raw_components, Mapping) else {}
    )
    score_components["fixture_selection_pressure"] = score_delta
    fixture_gate = entry.get("fixture_gate")
    gate_payload = fixture_gate if isinstance(fixture_gate, Mapping) else {}
    blockers = gate_payload.get("blockers", [])
    pressure_summary = entry.get("fixture_blocker_pressure")
    pressure_payload = (
        pressure_summary if isinstance(pressure_summary, Mapping) else {}
    )
    if not pressure_payload:
        pressure_payload = _fixture_blocker_pressure_summary(gate_payload)
    horizon_summary = entry.get("fixture_horizon_summary")
    horizon_payload = (
        horizon_summary if isinstance(horizon_summary, Mapping) else {}
    )
    updated = dict(candidate)
    updated["pre_fixture_selection_score"] = _round(pre_fixture_score)
    updated["score_components"] = score_components
    updated["score"] = _score_components_total(score_components)
    updated["fixture_selection"] = {
        "policy": MIND_V3_GENERATION_FIXTURE_SELECTION_POLICY,
        "worker_count": int(worker_count),
        "score_delta": score_delta,
        "pre_fixture_score": _round(pre_fixture_score),
        "post_fixture_score": updated["score"],
        "fixture_gate_passed": bool(gate_payload.get("passed", False)),
        "blocker_count": len(blockers) if isinstance(blockers, list) else 0,
        "weighted_blocker_pressure": float(
            pressure_payload.get("weighted_blocker_pressure", 0.0)
        ),
        "carrion_only_blocker_count": int(
            pressure_payload.get("carrion_only_blocker_count", 0)
        ),
        "carrion_only_weighted_blocker_pressure": float(
            pressure_payload.get("carrion_only_weighted_blocker_pressure", 0.0)
        ),
        "worst_fixture": pressure_payload.get("worst_fixture", "unknown"),
        "worst_reason": pressure_payload.get("worst_reason", "unknown"),
        "blocker_counts_by_fixture": dict(
            pressure_payload.get("blocker_counts_by_fixture", {})
        )
        if isinstance(pressure_payload.get("blocker_counts_by_fixture"), Mapping)
        else {},
        "blocker_counts_by_reason": dict(
            pressure_payload.get("blocker_counts_by_reason", {})
        )
        if isinstance(pressure_payload.get("blocker_counts_by_reason"), Mapping)
        else {},
        "passed_horizon_count": int(
            horizon_payload.get("passed_horizon_count", 0)
        ),
        "first_horizon_passed": _fixture_first_horizon_passed(entry),
        "selection_key": list(selection_key),
        "entry": entry,
    }
    return updated


def _fixture_selection_generation_summary(
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    fixture_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate.get("fixture_selection"), Mapping)
    ]
    if not fixture_candidates:
        return None
    blocker_counts_by_fixture: Counter[str] = Counter()
    blocker_counts_by_reason: Counter[str] = Counter()
    candidate_entries: list[dict[str, object]] = []
    passed_count = 0
    for candidate in fixture_candidates:
        fixture_selection = candidate["fixture_selection"]
        payload = fixture_selection if isinstance(fixture_selection, Mapping) else {}
        if bool(payload.get("fixture_gate_passed", False)):
            passed_count += 1
        fixture_counts = payload.get("blocker_counts_by_fixture")
        if isinstance(fixture_counts, Mapping):
            for key, value in fixture_counts.items():
                blocker_counts_by_fixture[str(key)] += int(value)
        reason_counts = payload.get("blocker_counts_by_reason")
        if isinstance(reason_counts, Mapping):
            for key, value in reason_counts.items():
                blocker_counts_by_reason[str(key)] += int(value)
        candidate_entries.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "score_delta": float(payload.get("score_delta", 0.0)),
                "fixture_gate_passed": bool(
                    payload.get("fixture_gate_passed", False)
                ),
                "blocker_count": int(payload.get("blocker_count", 0)),
                "weighted_blocker_pressure": float(
                    payload.get("weighted_blocker_pressure", 0.0)
                ),
                "carrion_only_blocker_count": int(
                    payload.get("carrion_only_blocker_count", 0)
                ),
                "carrion_only_weighted_blocker_pressure": float(
                    payload.get(
                        "carrion_only_weighted_blocker_pressure",
                        0.0,
                    )
                ),
                "worst_fixture": str(payload.get("worst_fixture", "unknown")),
                "worst_reason": str(payload.get("worst_reason", "unknown")),
            }
        )
    return {
        "policy": MIND_V3_GENERATION_FIXTURE_SELECTION_POLICY,
        "nominee_policy": MIND_V3_GENERATION_FIXTURE_SELECTION_NOMINEE_POLICY,
        "candidate_count": len(fixture_candidates),
        "candidate_ids": [
            str(candidate.get("candidate_id", ""))
            for candidate in fixture_candidates
        ],
        "fixture_gate_pass_count": passed_count,
        "blocker_counts_by_fixture": dict(sorted(blocker_counts_by_fixture.items())),
        "blocker_counts_by_reason": dict(sorted(blocker_counts_by_reason.items())),
        "candidates": candidate_entries,
    }


def _fixture_selection_score_delta(entry: Mapping[str, object]) -> float:
    fixture_gate = entry.get("fixture_gate")
    gate_payload = fixture_gate if isinstance(fixture_gate, Mapping) else {}
    blockers = gate_payload.get("blockers", [])
    blocker_count = len(blockers) if isinstance(blockers, list) else 0
    pressure_summary = _fixture_blocker_pressure_summary(gate_payload)
    weighted_blocker_pressure = float(
        pressure_summary["weighted_blocker_pressure"]
    )
    carrion_weighted_blocker_pressure = float(
        pressure_summary["carrion_only_weighted_blocker_pressure"]
    )
    horizon_summary = entry.get("fixture_horizon_summary")
    horizon_payload = (
        horizon_summary if isinstance(horizon_summary, Mapping) else {}
    )
    fixture_summary = entry.get("fixture_summary")
    fixture_payload = (
        fixture_summary if isinstance(fixture_summary, Mapping) else {}
    )
    carrion = _fixture_summary_named_fixture(fixture_payload, "carrion_only")
    passed_horizon_count = int(
        horizon_payload.get(
            "passed_horizon_count",
            1 if bool(gate_payload.get("passed", False)) else 0,
        )
    )
    first_horizon_bonus = 5.0 if _fixture_first_horizon_passed(entry) else 0.0
    gate_bonus = 18.0 if bool(gate_payload.get("passed", False)) else 0.0
    carrion_energy_requirement = float(
        horizon_payload.get(
            "carrion_only_terminal_energy_requirement_satisfaction_min",
            carrion.get("terminal_energy_requirement_satisfaction_mean", 0.0),
        )
    )
    carrion_energy_viability = float(
        horizon_payload.get(
            "carrion_only_terminal_energy_viability_share_min",
            carrion.get("terminal_energy_viability_share_mean", 0.0),
        )
    )
    carrion_hydration_viability = float(
        horizon_payload.get(
            "carrion_only_terminal_hydration_viability_share_min",
            carrion.get("terminal_hydration_viability_share_mean", 0.0),
        )
    )
    carrion_matched_viability = float(
        horizon_payload.get(
            "carrion_only_terminal_matched_diet_viability_share_min",
            carrion.get("terminal_matched_diet_viability_share_mean", 0.0),
        )
    )
    carrion_resource_events = float(
        horizon_payload.get(
            "carrion_only_animal_resource_consumption_events_min",
            carrion.get("animal_resource_consumption_events_mean", 0.0),
        )
    )
    carrion_resource_gain = float(
        horizon_payload.get(
            "carrion_only_animal_resource_gained_energy_min",
            carrion.get("animal_resource_gained_energy_mean", 0.0),
        )
    )
    carrion_alive = float(
        horizon_payload.get(
            "carrion_only_alive_agents_min",
            carrion.get("alive_agents_mean", 0.0),
        )
    )
    carrion_alive_ticks = float(
        horizon_payload.get(
            "carrion_only_alive_agent_ticks_per_tick_min",
            carrion.get("alive_agent_ticks_per_tick_mean", 0.0),
        )
    )
    carrion_births = float(
        horizon_payload.get(
            "carrion_only_births_min",
            carrion.get("births_mean", 0.0),
        )
    )
    fixture_reproduction_viability = float(
        horizon_payload.get(
            "terminal_reproduction_viability_min",
            fixture_payload.get("terminal_reproduction_viability_min", 0.0),
        )
    )
    fixture_energy_hydration_balance = float(
        horizon_payload.get(
            "terminal_energy_hydration_balance_min",
            fixture_payload.get("terminal_energy_hydration_balance_min", 0.0),
        )
    )
    fixture_births = float(
        horizon_payload.get(
            "births_min",
            fixture_payload.get("births_mean", 0.0),
        )
    )
    fixture_alive = float(
        horizon_payload.get(
            "alive_agents_min",
            fixture_payload.get("alive_agents_mean", 0.0),
        )
    )
    return _round(
        gate_bonus
        + 4.0 * passed_horizon_count
        + first_horizon_bonus
        - 1.5 * blocker_count
        - 4.0 * weighted_blocker_pressure
        - 2.0 * carrion_weighted_blocker_pressure
        + 4.0 * carrion_energy_requirement
        + 3.0 * carrion_energy_viability
        + 2.5 * carrion_hydration_viability
        + 2.5 * carrion_matched_viability
        + 0.25 * min(20.0, carrion_resource_events)
        + 0.1 * min(50.0, carrion_resource_gain)
        + 0.5 * carrion_alive
        + 0.25 * carrion_alive_ticks
        + 0.3 * carrion_births
        + 2.0 * fixture_reproduction_viability
        + 2.0 * fixture_energy_hydration_balance
        + 0.2 * fixture_births
        + 0.1 * fixture_alive
    )


def _fixture_blocker_pressure_summary(
    fixture_gate: Mapping[str, object],
) -> dict[str, object]:
    raw_blockers = fixture_gate.get("blockers")
    blockers = raw_blockers if isinstance(raw_blockers, list) else []
    fixture_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    fixture_pressure: Counter[str] = Counter()
    reason_pressure: Counter[str] = Counter()
    weighted_pressure = 0.0
    carrion_count = 0
    carrion_weighted_pressure = 0.0
    for raw_blocker in blockers:
        if not isinstance(raw_blocker, Mapping):
            continue
        fixture = str(raw_blocker.get("fixture") or "unknown")
        reason = str(raw_blocker.get("reason") or "unknown")
        pressure = _fixture_blocker_floor_gap(raw_blocker)
        weighted = pressure * _fixture_blocker_weight(raw_blocker)
        fixture_counts.update([fixture])
        reason_counts.update([reason])
        fixture_pressure[fixture] += weighted
        reason_pressure[reason] += weighted
        weighted_pressure += weighted
        if fixture == "carrion_only":
            carrion_count += 1
            carrion_weighted_pressure += weighted
    return {
        "blocker_count": sum(fixture_counts.values()),
        "weighted_blocker_pressure": _round(weighted_pressure),
        "carrion_only_blocker_count": carrion_count,
        "carrion_only_weighted_blocker_pressure": _round(
            carrion_weighted_pressure
        ),
        "blocker_counts_by_fixture": dict(sorted(fixture_counts.items())),
        "blocker_counts_by_reason": dict(sorted(reason_counts.items())),
        "weighted_pressure_by_fixture": {
            key: _round(value)
            for key, value in sorted(fixture_pressure.items())
        },
        "weighted_pressure_by_reason": {
            key: _round(value)
            for key, value in sorted(reason_pressure.items())
        },
        "worst_fixture": _dominant_counter_key(fixture_pressure),
        "worst_reason": _dominant_counter_key(reason_pressure),
    }


def _fixture_blocker_floor_gap(blocker: Mapping[str, object]) -> float:
    floor = _fixture_blocker_numeric_value(blocker.get("floor"))
    value = _fixture_blocker_numeric_value(blocker.get("value"))
    if floor is None or value is None:
        return 0.0
    return max(0.0, floor - value)


def _fixture_blocker_numeric_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _fixture_blocker_weight(blocker: Mapping[str, object]) -> float:
    fixture = str(blocker.get("fixture") or "")
    reason = str(blocker.get("reason") or "")
    weight = 1.0
    if fixture == "carrion_only":
        weight += 0.75
    if reason == "fixture_alive_floor":
        weight += 1.25
    elif reason in {
        "fixture_energy_viability_floor",
        "fixture_hydration_viability_floor",
        "fixture_health_viability_floor",
        "fixture_matched_diet_viability_floor",
    }:
        weight += 0.5
    return weight


def _evaluate_candidate_task(task: dict[str, object]) -> dict[str, object]:
    return _evaluate_candidate(
        dict(task["candidate"]),
        generation_index=int(task["generation_index"]),
        seeds=[int(seed) for seed in list(task["seeds"])],
        ticks=int(task["ticks"]),
    )


def _run_candidate(
    *,
    seed: int,
    ticks: int,
    metadata: dict[str, object] | list[dict[str, object]],
) -> dict[str, object]:
    policy = MindV3EvolutionPolicy(
        seed=seed,
        founder_template_metadata=metadata,
    )
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=policy,
    )
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    policy_id_counts = Counter(
        str(record.get("policy_id", "unknown"))
        for record in world.trajectory_records
    )
    controller_population_metadata = _controller_population_metadata(policy)
    agent_controller_stats = _agent_controller_stats(
        world,
        controller_population_metadata,
    )
    resource_event_count = 0
    movement_event_count = 0
    animal_resource_event_count = 0
    fresh_kill_event_count = 0
    carcass_event_count = 0
    animal_resource_gain_total = 0.0
    requested_actions: set[str] = set()
    requested_action_counts: Counter[str] = Counter()
    resolved_action_counts: Counter[str] = Counter()
    unsupported_requested_action_count = 0
    unsupported_resolved_action_count = 0
    unsupported_requested_action_breakdown = _unsupported_action_breakdown(
        seed=seed,
        records=world.trajectory_records,
        validity_key="action_valid",
    )
    unsupported_resolved_action_breakdown = _unsupported_action_breakdown(
        seed=seed,
        records=world.trajectory_records,
        validity_key="resolution_action_valid",
    )
    rollout_context_diagnostics = _rollout_context_decision_summary(
        diagnostics_records=world.policy_decision_diagnostics_records,
        trajectory_records=world.trajectory_records,
    )
    recovery_context_diagnostics = _recovery_context_decision_summary(
        diagnostics_records=world.policy_decision_diagnostics_records,
        trajectory_records=world.trajectory_records,
    )
    for record in world.trajectory_records:
        requested_action = record.get("requested_action")
        if requested_action is not None:
            action_name = str(requested_action)
            requested_actions.add(action_name)
            requested_action_counts.update([action_name])
            if record.get("action_valid") is False:
                unsupported_requested_action_count += 1
        resolved_action = record.get("resolved_action")
        if resolved_action is not None:
            resolved_action_counts.update([str(resolved_action)])
            if record.get("resolution_action_valid") is False:
                unsupported_resolved_action_count += 1
        if bool(record.get("moved", False)):
            movement_event_count += 1
        outcome = record.get("outcome")
        if not isinstance(outcome, dict):
            continue
        feeding = outcome.get("feeding")
        drinking = outcome.get("drinking")
        if isinstance(feeding, dict) and bool(feeding.get("ate", False)):
            resource_event_count += 1
            food_source = str(feeding.get("food_source", ""))
            gained_energy = _float_value(feeding.get("gained_energy")) or 0.0
            if food_source in {"carcass", "fresh_kill"}:
                animal_resource_event_count += 1
                animal_resource_gain_total += gained_energy
                if food_source == "carcass":
                    carcass_event_count += 1
                else:
                    fresh_kill_event_count += 1
        if isinstance(drinking, dict) and bool(drinking.get("drank", False)):
            resource_event_count += 1
    trajectory_record_count = len(world.trajectory_records)
    sustained_readiness = _run_sustained_readiness_metrics(
        world.trajectory_records,
        ticks_executed=int(result.summary["ticks_executed"]),
    )
    reproduction_viability = _run_reproduction_viability(result.summary)
    return {
        "seed": seed,
        "ticks": ticks,
        "ticks_executed": int(result.summary["ticks_executed"]),
        "alive_agents": int(result.summary["alive_agents"]),
        "births": int(result.summary["births"]),
        "deaths": int(result.summary["deaths"]),
        "trajectory_record_count": trajectory_record_count,
        "alive_agent_ticks": trajectory_record_count,
        "alive_agent_ticks_per_tick": _round(trajectory_record_count / max(1, ticks)),
        "resource_event_count": resource_event_count,
        "resource_event_rate": _round(
            resource_event_count / max(1, trajectory_record_count)
        ),
        "animal_resource_event_count": animal_resource_event_count,
        "animal_resource_event_rate": _round(
            animal_resource_event_count / max(1, trajectory_record_count)
        ),
        "fresh_kill_event_count": fresh_kill_event_count,
        "fresh_kill_event_rate": _round(
            fresh_kill_event_count / max(1, trajectory_record_count)
        ),
        "carcass_event_count": carcass_event_count,
        "carcass_event_rate": _round(
            carcass_event_count / max(1, trajectory_record_count)
        ),
        "animal_resource_gain_total": _round(animal_resource_gain_total),
        "movement_event_count": movement_event_count,
        "movement_event_rate": _round(
            movement_event_count / max(1, trajectory_record_count)
        ),
        "unique_requested_actions": len(requested_actions),
        "sustained_readiness": sustained_readiness,
        "core_readiness_agent_tick_mean": sustained_readiness[
            "core_readiness_agent_tick_mean"
        ],
        "core_ready_agent_tick_share": sustained_readiness[
            "core_ready_agent_tick_share"
        ],
        "core_ready_tick_share": sustained_readiness["core_ready_tick_share"],
        "core_ready_pair_tick_share": sustained_readiness[
            "core_ready_pair_tick_share"
        ],
        "core_ready_distinct_agent_count": sustained_readiness[
            "core_ready_distinct_agent_count"
        ],
        "reproduction_ready_agent_tick_count": sustained_readiness[
            "reproduction_ready_agent_tick_count"
        ],
        "reproduction_ready_agent_tick_share": sustained_readiness[
            "reproduction_ready_agent_tick_share"
        ],
        "reproduction_ready_tick_count": sustained_readiness[
            "reproduction_ready_tick_count"
        ],
        "reproduction_ready_tick_share": sustained_readiness[
            "reproduction_ready_tick_share"
        ],
        "reproduction_ready_pair_tick_count": sustained_readiness[
            "reproduction_ready_pair_tick_count"
        ],
        "reproduction_ready_pair_tick_share": sustained_readiness[
            "reproduction_ready_pair_tick_share"
        ],
        "reproduction_ready_distinct_agent_count": sustained_readiness[
            "reproduction_ready_distinct_agent_count"
        ],
        "reproduction_viability": reproduction_viability,
        "terminal_energy_viability_share": reproduction_viability[
            "terminal_energy_viability_share"
        ],
        "terminal_energy_requirement_satisfaction": reproduction_viability[
            "terminal_energy_requirement_satisfaction"
        ],
        "terminal_balanced_reproduction_readiness": reproduction_viability[
            "terminal_balanced_reproduction_readiness"
        ],
        "terminal_energy_hydration_balance": reproduction_viability[
            "terminal_energy_hydration_balance"
        ],
        "terminal_energy_hydration_gap_abs": reproduction_viability[
            "terminal_energy_hydration_gap_abs"
        ],
        "terminal_energy_shortfall_share": reproduction_viability[
            "terminal_energy_shortfall_share"
        ],
        "terminal_energy_required_total": reproduction_viability[
            "terminal_energy_required_total"
        ],
        "terminal_energy_gap_total": reproduction_viability[
            "terminal_energy_gap_total"
        ],
        "terminal_hydration_viability_share": reproduction_viability[
            "terminal_hydration_viability_share"
        ],
        "terminal_health_viability_share": reproduction_viability[
            "terminal_health_viability_share"
        ],
        "terminal_matched_diet_viability_share": reproduction_viability[
            "terminal_matched_diet_viability_share"
        ],
        "reproduction_ready_agents": reproduction_viability[
            "reproduction_ready_agents"
        ],
        "biologically_reproduction_ready_agents": reproduction_viability[
            "biologically_reproduction_ready_agents"
        ],
        "terminal_biologically_ready_group_count": reproduction_viability[
            "terminal_biologically_ready_group_count"
        ],
        "terminal_ready_group_count": reproduction_viability[
            "terminal_ready_group_count"
        ],
        "core_blocker_agent_tick_counts": sustained_readiness[
            "core_blocker_agent_tick_counts"
        ],
        "core_blocker_agent_tick_shares": sustained_readiness[
            "core_blocker_agent_tick_shares"
        ],
        "primary_core_blocker_agent_tick_counts": sustained_readiness[
            "primary_core_blocker_agent_tick_counts"
        ],
        "primary_core_blocker_agent_tick_shares": sustained_readiness[
            "primary_core_blocker_agent_tick_shares"
        ],
        "primary_core_blocker": sustained_readiness["primary_core_blocker"],
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(
            unsupported_resolved_action_count
        ),
        "unsupported_requested_action_breakdown": unsupported_requested_action_breakdown,
        "unsupported_resolved_action_breakdown": unsupported_resolved_action_breakdown,
        **rollout_context_diagnostics,
        **recovery_context_diagnostics,
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "behavior_descriptors": _run_behavior_descriptors(
            agent_controller_stats,
            action_source_counts=action_source_counts,
        ),
        "controller_lineage_elites": _run_controller_lineage_elites(
            agent_controller_stats
        ),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }


def _rollout_context_decision_summary(
    *,
    diagnostics_records: list[dict[str, object] | None],
    trajectory_records: list[dict[str, object]],
) -> dict[str, object]:
    count = 0
    non_empty_count = 0
    post_carrion_count = 0
    nonzero_count = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    per_action: dict[str, dict[str, float | int]] = {}
    for index, diagnostics in enumerate(diagnostics_records):
        if not isinstance(diagnostics, Mapping):
            continue
        if "rollout_context_schema_version" not in diagnostics:
            continue
        count += 1
        if bool(diagnostics.get("rollout_context_non_empty", False)):
            non_empty_count += 1
        if bool(diagnostics.get("rollout_context_post_carrion_context", False)):
            post_carrion_count += 1
        delta = _float_value(
            diagnostics.get("rollout_context_selected_score_delta")
        )
        if delta is None:
            delta = 0.0
        delta = _round(delta)
        delta_abs = abs(delta)
        delta_sum += delta
        delta_abs_sum += delta_abs
        delta_abs_max = max(delta_abs_max, delta_abs)
        if delta != 0.0:
            nonzero_count += 1
        action = "unknown"
        if index < len(trajectory_records):
            action = str(trajectory_records[index].get("requested_action", "unknown"))
        action_stats = per_action.setdefault(
            action,
            {
                "count": 0,
                "nonzero_count": 0,
                "delta_sum": 0.0,
                "delta_abs_sum": 0.0,
                "delta_abs_max": 0.0,
            },
        )
        action_stats["count"] = int(action_stats["count"]) + 1
        action_stats["delta_sum"] = float(action_stats["delta_sum"]) + delta
        action_stats["delta_abs_sum"] = (
            float(action_stats["delta_abs_sum"]) + delta_abs
        )
        action_stats["delta_abs_max"] = max(
            float(action_stats["delta_abs_max"]),
            delta_abs,
        )
        if delta != 0.0:
            action_stats["nonzero_count"] = int(action_stats["nonzero_count"]) + 1
    return _rollout_context_summary_from_totals(
        count=count,
        non_empty_count=non_empty_count,
        post_carrion_count=post_carrion_count,
        nonzero_count=nonzero_count,
        delta_sum=delta_sum,
        delta_abs_sum=delta_abs_sum,
        delta_abs_max=delta_abs_max,
        per_action=per_action,
    )


def _rollout_context_summary_from_totals(
    *,
    count: int,
    non_empty_count: int,
    post_carrion_count: int,
    nonzero_count: int,
    delta_sum: float,
    delta_abs_sum: float,
    delta_abs_max: float,
    per_action: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "rollout_context_decision_count": int(count),
        "rollout_context_non_empty_count": int(non_empty_count),
        "rollout_context_non_empty_share": _round(
            int(non_empty_count) / max(1, int(count))
        ),
        "rollout_context_selected_score_delta_nonzero_count": int(nonzero_count),
        "rollout_context_selected_score_delta_mean": _round(
            float(delta_sum) / max(1, int(count))
        ),
        "rollout_context_selected_score_delta_abs_mean": _round(
            float(delta_abs_sum) / max(1, int(count))
        ),
        "rollout_context_selected_score_delta_abs_max": _round(delta_abs_max),
        "rollout_context_post_carrion_context_count": int(post_carrion_count),
        "rollout_context_post_carrion_context_share": _round(
            int(post_carrion_count) / max(1, int(count))
        ),
        "rollout_context_selected_score_delta_by_requested_action": (
            _rollout_context_per_action_summary(per_action)
        ),
    }


def _rollout_context_per_action_summary(
    per_action: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for action in sorted(per_action):
        stats = per_action[action]
        count = int(stats.get("count", 0))
        summary[str(action)] = {
            "count": count,
            "nonzero_count": int(stats.get("nonzero_count", 0)),
            "mean": _round(float(stats.get("delta_sum", 0.0)) / max(1, count)),
            "abs_mean": _round(
                float(stats.get("delta_abs_sum", 0.0)) / max(1, count)
            ),
            "abs_max": _round(float(stats.get("delta_abs_max", 0.0))),
        }
    return summary


def _recovery_context_decision_summary(
    *,
    diagnostics_records: list[dict[str, object] | None],
    trajectory_records: list[dict[str, object]],
) -> dict[str, object]:
    count = 0
    non_empty_count = 0
    post_carrion_count = 0
    drink_available_count = 0
    nonzero_count = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    per_action: dict[str, dict[str, float | int]] = {}
    for index, diagnostics in enumerate(diagnostics_records):
        if not isinstance(diagnostics, Mapping):
            continue
        if "recovery_context_schema_version" not in diagnostics:
            continue
        count += 1
        if bool(diagnostics.get("recovery_context_non_empty", False)):
            non_empty_count += 1
        if bool(diagnostics.get("recovery_context_post_carrion_contact", False)):
            post_carrion_count += 1
        if bool(diagnostics.get("recovery_context_drink_available", False)):
            drink_available_count += 1
        delta = _float_value(
            diagnostics.get("recovery_context_selected_score_delta")
        )
        if delta is None:
            delta = 0.0
        delta = _round(delta)
        delta_abs = abs(delta)
        delta_sum += delta
        delta_abs_sum += delta_abs
        delta_abs_max = max(delta_abs_max, delta_abs)
        if delta != 0.0:
            nonzero_count += 1
        action = "unknown"
        if index < len(trajectory_records):
            action = str(trajectory_records[index].get("requested_action", "unknown"))
        action_stats = per_action.setdefault(
            action,
            {
                "count": 0,
                "nonzero_count": 0,
                "delta_sum": 0.0,
                "delta_abs_sum": 0.0,
                "delta_abs_max": 0.0,
            },
        )
        action_stats["count"] = int(action_stats["count"]) + 1
        action_stats["delta_sum"] = float(action_stats["delta_sum"]) + delta
        action_stats["delta_abs_sum"] = (
            float(action_stats["delta_abs_sum"]) + delta_abs
        )
        action_stats["delta_abs_max"] = max(
            float(action_stats["delta_abs_max"]),
            delta_abs,
        )
        if delta != 0.0:
            action_stats["nonzero_count"] = int(action_stats["nonzero_count"]) + 1
    return _recovery_context_summary_from_totals(
        count=count,
        non_empty_count=non_empty_count,
        post_carrion_count=post_carrion_count,
        drink_available_count=drink_available_count,
        nonzero_count=nonzero_count,
        delta_sum=delta_sum,
        delta_abs_sum=delta_abs_sum,
        delta_abs_max=delta_abs_max,
        per_action=per_action,
    )


def _recovery_context_summary_from_totals(
    *,
    count: int,
    non_empty_count: int,
    post_carrion_count: int,
    drink_available_count: int,
    nonzero_count: int,
    delta_sum: float,
    delta_abs_sum: float,
    delta_abs_max: float,
    per_action: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "recovery_context_decision_count": int(count),
        "recovery_context_non_empty_count": int(non_empty_count),
        "recovery_context_non_empty_share": _round(
            int(non_empty_count) / max(1, int(count))
        ),
        "recovery_context_selected_score_delta_nonzero_count": int(nonzero_count),
        "recovery_context_selected_score_delta_mean": _round(
            float(delta_sum) / max(1, int(count))
        ),
        "recovery_context_selected_score_delta_abs_mean": _round(
            float(delta_abs_sum) / max(1, int(count))
        ),
        "recovery_context_selected_score_delta_abs_max": _round(delta_abs_max),
        "recovery_context_post_carrion_context_count": int(post_carrion_count),
        "recovery_context_post_carrion_context_share": _round(
            int(post_carrion_count) / max(1, int(count))
        ),
        "recovery_context_drink_available_count": int(drink_available_count),
        "recovery_context_selected_score_delta_by_requested_action": (
            _rollout_context_per_action_summary(per_action)
        ),
    }


def _unsupported_action_breakdown(
    *,
    seed: int,
    records: list[dict[str, object]],
    validity_key: str,
) -> dict[str, object]:
    by_requested: Counter[str] = Counter()
    by_resolved: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_tuple: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if record.get(validity_key) is not False:
            continue
        requested_action = str(record.get("requested_action", "unknown"))
        resolved_action = str(record.get("resolved_action", "unknown"))
        reason = _record_invalid_reason(record)
        by_requested.update([requested_action])
        by_resolved.update([resolved_action])
        by_reason.update([reason])
        by_tuple.update([(requested_action, resolved_action, reason)])
    return _unsupported_action_breakdown_from_counters(
        seed=seed,
        by_requested=by_requested,
        by_resolved=by_resolved,
        by_reason=by_reason,
        by_tuple=by_tuple,
    )


def _unsupported_action_breakdown_from_counters(
    *,
    seed: int | None,
    by_requested: Counter[str],
    by_resolved: Counter[str],
    by_reason: Counter[str],
    by_tuple: Counter[tuple[str, str, str]],
    by_seed: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "by_requested_action": dict(sorted(by_requested.items())),
        "by_resolved_action": dict(sorted(by_resolved.items())),
        "by_invalid_reason": dict(sorted(by_reason.items())),
        "by_requested_resolved_invalid_reason": [
            {
                "requested_action": requested,
                "resolved_action": resolved,
                "invalid_reason": reason,
                "count": int(count),
            }
            for (requested, resolved, reason), count in sorted(by_tuple.items())
        ],
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if by_seed is not None:
        payload["by_seed"] = by_seed
    return payload


def _record_invalid_reason(record: Mapping[str, object]) -> str:
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping):
        reason = outcome.get("invalid_reason")
        if isinstance(reason, str) and reason:
            return reason
    reason = record.get("invalid_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "unknown"


def _run_reproduction_viability(
    summary: Mapping[str, object],
) -> dict[str, float | int | str]:
    alive_agents = int(summary.get("alive_agents", 0))
    reproduction = summary.get("reproduction_end")
    if not isinstance(reproduction, Mapping):
        return _empty_reproduction_viability(alive_agents)
    blockers = reproduction.get("biological_blocker_counts")
    if not isinstance(blockers, Mapping):
        blockers = {}
    energy_blockers = _bounded_blocker_count(blockers, "energy", alive_agents)
    hydration_blockers = _bounded_blocker_count(
        blockers,
        "hydration",
        alive_agents,
    )
    health_blockers = _bounded_blocker_count(blockers, "health", alive_agents)
    matched_diet_blockers = _bounded_blocker_count(
        blockers,
        "matched_diet",
        alive_agents,
    )
    energy_readiness = _terminal_energy_readiness(reproduction)
    terminal_energy_viability = _viability_share(alive_agents, energy_blockers)
    terminal_hydration_viability = _viability_share(
        alive_agents,
        hydration_blockers,
    )
    terminal_health_viability = _viability_share(alive_agents, health_blockers)
    terminal_matched_diet_viability = _viability_share(
        alive_agents,
        matched_diet_blockers,
    )
    terminal_energy_requirement_satisfaction = float(
        energy_readiness["energy_requirement_satisfaction"]
    )
    terminal_balanced_readiness = _balanced_readiness_score(
        [
            terminal_energy_requirement_satisfaction,
            terminal_hydration_viability,
            terminal_health_viability,
            terminal_matched_diet_viability,
        ]
    )
    terminal_energy_hydration_balance = min(
        terminal_energy_requirement_satisfaction,
        terminal_hydration_viability,
    )
    return {
        "policy": MIND_V3_REPRODUCTION_VIABILITY_SCORE_POLICY,
        "alive_agents": alive_agents,
        "reproduction_ready_agents": _bounded_count(
            reproduction.get("ready_agents"),
            alive_agents,
        ),
        "biologically_reproduction_ready_agents": _bounded_count(
            reproduction.get("biologically_ready_agents"),
            alive_agents,
        ),
        "terminal_energy_blocker_agents": energy_blockers,
        "terminal_hydration_blocker_agents": hydration_blockers,
        "terminal_health_blocker_agents": health_blockers,
        "terminal_matched_diet_blocker_agents": matched_diet_blockers,
        "terminal_energy_viability_share": terminal_energy_viability,
        "terminal_energy_requirement_satisfaction": (
            terminal_energy_requirement_satisfaction
        ),
        "terminal_balanced_reproduction_readiness": terminal_balanced_readiness,
        "terminal_energy_hydration_balance": _round(
            terminal_energy_hydration_balance
        ),
        "terminal_energy_hydration_gap_abs": _round(
            abs(
                terminal_energy_requirement_satisfaction
                - terminal_hydration_viability
            )
        ),
        "terminal_energy_shortfall_share": energy_readiness[
            "energy_shortfall_share"
        ],
        "terminal_energy_shortfall_agents": energy_readiness[
            "energy_shortfall_agents"
        ],
        "terminal_energy_total": energy_readiness["energy_total"],
        "terminal_energy_required_total": energy_readiness[
            "energy_required_total"
        ],
        "terminal_energy_gap_total": energy_readiness["energy_gap_total"],
        "terminal_hydration_viability_share": terminal_hydration_viability,
        "terminal_health_viability_share": terminal_health_viability,
        "terminal_matched_diet_viability_share": terminal_matched_diet_viability,
        "terminal_biologically_ready_group_count": _bounded_count(
            reproduction.get("biologically_ready_group_count"),
            alive_agents,
        ),
        "terminal_ready_group_count": _bounded_count(
            reproduction.get("ready_group_count"),
            alive_agents,
        ),
    }


def _terminal_energy_readiness(
    reproduction: Mapping[str, object],
) -> dict[str, float | int]:
    by_mode = reproduction.get("energy_readiness_by_meat_mode")
    grouped = by_mode if isinstance(by_mode, Mapping) else {}
    totals = {
        "alive_agents": 0,
        "energy_shortfall_agents": 0,
        "energy_total": 0.0,
        "energy_required_total": 0.0,
        "energy_gap_total": 0.0,
    }
    for raw_counts in grouped.values():
        if not isinstance(raw_counts, Mapping):
            continue
        totals["alive_agents"] = int(totals["alive_agents"]) + _nonnegative_int(
            raw_counts.get("alive_agents")
        )
        totals["energy_shortfall_agents"] = int(
            totals["energy_shortfall_agents"]
        ) + _nonnegative_int(raw_counts.get("energy_shortfall_agents"))
        totals["energy_total"] = float(totals["energy_total"]) + _nonnegative_float(
            raw_counts.get("energy_total")
        )
        totals["energy_required_total"] = float(
            totals["energy_required_total"]
        ) + _nonnegative_float(raw_counts.get("energy_required_total"))
        totals["energy_gap_total"] = float(
            totals["energy_gap_total"]
        ) + _nonnegative_float(raw_counts.get("energy_gap_total"))
    alive_agents = int(totals["alive_agents"])
    required_total = float(totals["energy_required_total"])
    gap_total = float(totals["energy_gap_total"])
    return {
        "alive_agents": alive_agents,
        "energy_shortfall_agents": int(totals["energy_shortfall_agents"]),
        "energy_total": _round(float(totals["energy_total"])),
        "energy_required_total": _round(required_total),
        "energy_gap_total": _round(gap_total),
        "energy_shortfall_share": _count_share(
            int(totals["energy_shortfall_agents"]),
            alive_agents,
        ),
        "energy_requirement_satisfaction": (
            _round(max(0.0, min(1.0, 1.0 - gap_total / required_total)))
            if required_total > 0.0
            else 0.0
        ),
    }


def _balanced_readiness_score(values: list[float]) -> float:
    bounded = [max(0.0, min(1.0, float(value))) for value in values]
    if not bounded or any(value <= 0.0 for value in bounded):
        return 0.0
    return _round(len(bounded) / sum(1.0 / value for value in bounded))


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, int(value))


def _nonnegative_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value))


def _run_sustained_readiness_metrics(
    records: list[dict[str, object]],
    *,
    ticks_executed: int,
) -> dict[str, float | int | str]:
    ready_agents: set[int] = set()
    ready_by_tick: Counter[int] = Counter()
    core_ready_agents: set[int] = set()
    core_ready_by_tick: Counter[int] = Counter()
    core_readiness_values: list[float] = []
    ready_agent_tick_count = 0
    core_ready_agent_tick_count = 0
    blocker_counts: Counter[str] = Counter()
    primary_blocker_counts: Counter[str] = Counter()
    for record in records:
        agent_id = _int_value(record.get("agent_id"))
        tick = _int_value(record.get("tick"))
        if agent_id is None or tick is None:
            continue
        after = record.get("after")
        if not isinstance(after, Mapping) or not bool(after.get("alive", False)):
            continue
        core_quality = _core_readiness_quality(after)
        if core_quality is not None:
            core_readiness_values.append(core_quality)
            blockers = _core_readiness_blockers(after)
            blocker_counts.update(blockers)
            primary_blocker = _primary_core_readiness_blocker(after)
            if primary_blocker is not None:
                primary_blocker_counts.update([primary_blocker])
            if core_quality >= 1.0:
                core_ready_agents.add(agent_id)
                core_ready_by_tick.update([tick])
                core_ready_agent_tick_count += 1
        outcome = record.get("outcome")
        if isinstance(outcome, Mapping) and bool(
            outcome.get("reproduction_ready_after", False)
        ):
            ready_agents.add(agent_id)
            ready_by_tick.update([tick])
            ready_agent_tick_count += 1
    ready_tick_count = len(ready_by_tick)
    ready_pair_tick_count = sum(
        1 for count in ready_by_tick.values() if int(count) >= 2
    )
    core_ready_tick_count = len(core_ready_by_tick)
    core_ready_pair_tick_count = sum(
        1 for count in core_ready_by_tick.values() if int(count) >= 2
    )
    record_count = len(records)
    alive_record_count = len(core_readiness_values)
    tick_count = max(1, int(ticks_executed))
    return {
        "policy": MIND_V3_SUSTAINED_READINESS_POLICY,
        "core_readiness_agent_tick_mean": _round(
            _mean(core_readiness_values)
        ),
        "core_ready_agent_tick_count": core_ready_agent_tick_count,
        "core_ready_agent_tick_share": _round(
            core_ready_agent_tick_count / max(1, record_count)
        ),
        "core_ready_tick_count": core_ready_tick_count,
        "core_ready_tick_share": _round(core_ready_tick_count / tick_count),
        "core_ready_pair_tick_count": core_ready_pair_tick_count,
        "core_ready_pair_tick_share": _round(
            core_ready_pair_tick_count / tick_count
        ),
        "core_ready_distinct_agent_count": len(core_ready_agents),
        "reproduction_ready_agent_tick_count": ready_agent_tick_count,
        "reproduction_ready_agent_tick_share": _round(
            ready_agent_tick_count / max(1, record_count)
        ),
        "reproduction_ready_tick_count": ready_tick_count,
        "reproduction_ready_tick_share": _round(ready_tick_count / tick_count),
        "reproduction_ready_pair_tick_count": ready_pair_tick_count,
        "reproduction_ready_pair_tick_share": _round(
            ready_pair_tick_count / tick_count
        ),
        "reproduction_ready_distinct_agent_count": len(ready_agents),
        "core_blocker_agent_tick_counts": dict(
            sorted((field, int(blocker_counts[field])) for field in _core_fields())
        ),
        "core_blocker_agent_tick_shares": {
            field: _round(int(blocker_counts[field]) / max(1, alive_record_count))
            for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_counts": dict(
            sorted(
                (field, int(primary_blocker_counts[field]))
                for field in _core_fields()
            )
        ),
        "primary_core_blocker_agent_tick_shares": {
            field: _round(
                int(primary_blocker_counts[field]) / max(1, alive_record_count)
            )
            for field in _core_fields()
        },
        "primary_core_blocker": _dominant_counter_key(primary_blocker_counts),
    }


def _core_readiness_quality(after: Mapping[str, object]) -> float | None:
    progress: list[float] = []
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        value = _float_value(after.get(field))
        if value is None:
            return None
        progress.append(min(1.0, max(0.0, value / max(float(goal), 1e-9))))
    return _round(min(progress)) if progress else None


def _core_readiness_blockers(after: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        value = _float_value(after.get(field))
        if value is None:
            continue
        if value < float(goal):
            blockers.append(_core_blocker_name(field))
    return blockers


def _primary_core_readiness_blocker(after: Mapping[str, object]) -> str | None:
    gaps: dict[str, float] = {}
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        value = _float_value(after.get(field))
        if value is None:
            continue
        gap = max(0.0, 1.0 - value / max(float(goal), 1e-9))
        if gap > 0.0:
            gaps[_core_blocker_name(field)] = gap
    if not gaps:
        return None
    return max(gaps, key=lambda field: (gaps[field], field))


def _core_fields() -> tuple[str, ...]:
    return tuple(
        _core_blocker_name(field)
        for field in MIND_V3_REPRODUCTION_READINESS_GOALS
    )


def _core_blocker_name(field: str) -> str:
    return field.removesuffix("_ratio")


def _empty_reproduction_viability(
    alive_agents: int,
) -> dict[str, float | int | str]:
    return {
        "policy": MIND_V3_REPRODUCTION_VIABILITY_SCORE_POLICY,
        "alive_agents": alive_agents,
        "reproduction_ready_agents": 0,
        "biologically_reproduction_ready_agents": 0,
        "terminal_energy_blocker_agents": alive_agents,
        "terminal_hydration_blocker_agents": alive_agents,
        "terminal_health_blocker_agents": alive_agents,
        "terminal_matched_diet_blocker_agents": alive_agents,
        "terminal_energy_viability_share": 0.0,
        "terminal_energy_requirement_satisfaction": 0.0,
        "terminal_balanced_reproduction_readiness": 0.0,
        "terminal_energy_hydration_balance": 0.0,
        "terminal_energy_hydration_gap_abs": 0.0,
        "terminal_energy_shortfall_share": 0.0,
        "terminal_energy_shortfall_agents": alive_agents,
        "terminal_energy_total": 0.0,
        "terminal_energy_required_total": 0.0,
        "terminal_energy_gap_total": 0.0,
        "terminal_hydration_viability_share": 0.0,
        "terminal_health_viability_share": 0.0,
        "terminal_matched_diet_viability_share": 0.0,
        "terminal_biologically_ready_group_count": 0,
        "terminal_ready_group_count": 0,
    }


def _bounded_blocker_count(
    blockers: Mapping[str, object],
    key: str,
    alive_agents: int,
) -> int:
    return _bounded_count(blockers.get(key), alive_agents)


def _bounded_count(value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, min(maximum, int(value)))


def _viability_share(alive_agents: int, blocker_agents: int) -> float:
    if alive_agents <= 0:
        return 0.0
    return _round((alive_agents - blocker_agents) / float(alive_agents))


def _count_share(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return _round(max(0, int(count)) / float(total))


def _controller_population_metadata(
    policy: MindV3EvolutionPolicy,
) -> dict[int, dict[str, object]]:
    population_metadata = getattr(policy, "controller_population_metadata", None)
    if not callable(population_metadata):
        return {}
    raw = population_metadata()
    if not isinstance(raw, dict):
        return {}
    return {
        int(agent_id): dict(metadata)
        for agent_id, metadata in raw.items()
        if isinstance(agent_id, int) and isinstance(metadata, dict)
    }


def _agent_controller_stats(
    world: SimulationWorld,
    controller_population_metadata: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    stats_by_agent: dict[int, dict[str, object]] = {}
    terrain_counts_by_agent: dict[int, Counter[str]] = {}
    action_counts_by_agent: dict[int, Counter[str]] = {}
    for record in world.trajectory_records:
        agent_id = _int_value(record.get("agent_id"))
        if agent_id is None:
            continue
        stats = stats_by_agent.setdefault(
            agent_id,
            {
                "agent_id": agent_id,
                "lineage_id": _int_value(record.get("lineage_id")),
                "runtime_species_id": _int_value(record.get("runtime_species_id")),
                "runtime_ecotype_id": _int_value(record.get("runtime_ecotype_id")),
                "record_count": 0,
                "first_tick": _int_value(record.get("tick")) or 0,
                "last_tick": _int_value(record.get("tick")) or 0,
                "reproduced_count": 0,
                "resource_event_count": 0,
                "movement_event_count": 0,
                "alive": False,
            },
        )
        tick = _int_value(record.get("tick"))
        if tick is not None:
            stats["first_tick"] = min(int(stats["first_tick"]), tick)
            stats["last_tick"] = max(int(stats["last_tick"]), tick)
        stats["record_count"] = int(stats["record_count"]) + 1
        after = record.get("after")
        if isinstance(after, Mapping):
            stats["alive"] = bool(after.get("alive", False))
            stats["last_energy_ratio"] = _float_value(after.get("energy_ratio"))
            stats["last_hydration_ratio"] = _float_value(
                after.get("hydration_ratio")
            )
            stats["last_health_ratio"] = _float_value(after.get("health_ratio"))
        outcome = record.get("outcome")
        if isinstance(outcome, Mapping):
            if bool(outcome.get("reproduced", False)):
                stats["reproduced_count"] = int(stats["reproduced_count"]) + 1
            feeding = outcome.get("feeding")
            drinking = outcome.get("drinking")
            if isinstance(feeding, Mapping) and bool(feeding.get("ate", False)):
                stats["resource_event_count"] = int(
                    stats["resource_event_count"]
                ) + 1
            if isinstance(drinking, Mapping) and bool(drinking.get("drank", False)):
                stats["resource_event_count"] = int(
                    stats["resource_event_count"]
                ) + 1
        if bool(record.get("moved", False)):
            stats["movement_event_count"] = int(stats["movement_event_count"]) + 1
        terrain = _terrain_for_record(world, record)
        terrain_counts_by_agent.setdefault(agent_id, Counter()).update([terrain])
        requested_action = record.get("requested_action")
        if isinstance(requested_action, str):
            action_counts_by_agent.setdefault(agent_id, Counter()).update(
                [requested_action]
            )

    stats: list[dict[str, object]] = []
    for agent_id, metadata in controller_population_metadata.items():
        agent_stats = stats_by_agent.get(agent_id)
        if agent_stats is None:
            continue
        agent = world.agents.get(agent_id)
        if agent is not None:
            agent_stats["alive"] = bool(agent.alive)
            agent_stats["trophic_role"] = _agent_trophic_role(world, agent)
            agent_stats["meat_mode"] = _agent_meat_mode(world, agent)
            agent_stats["lineage_id"] = agent.lineage_id
        terrain_counts = terrain_counts_by_agent.get(agent_id, Counter())
        action_counts = action_counts_by_agent.get(agent_id, Counter())
        record_count = max(1, int(agent_stats["record_count"]))
        agent_stats["dominant_terrain"] = _dominant_counter_key(terrain_counts)
        agent_stats["terrain_counts"] = dict(sorted(terrain_counts.items()))
        agent_stats["action_counts"] = dict(sorted(action_counts.items()))
        agent_stats["resource_event_rate"] = _round(
            int(agent_stats["resource_event_count"]) / record_count
        )
        agent_stats["movement_event_rate"] = _round(
            int(agent_stats["movement_event_count"]) / record_count
        )
        agent_stats["specialization_profile"] = _metadata_specialization_profile(
            metadata
        )
        agent_stats["controller_metadata"] = dict(metadata)
        stats.append(agent_stats)
    return stats


def _run_behavior_descriptors(
    agent_controller_stats: list[dict[str, object]],
    *,
    action_source_counts: Counter[str],
) -> dict[str, object]:
    terrain_occupancy: Counter[str] = Counter()
    dominant_terrain_counts: Counter[str] = Counter()
    trophic_role_counts: Counter[str] = Counter()
    meat_mode_counts: Counter[str] = Counter()
    specialization_profile_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    lineage_ids: set[int] = set()
    runtime_species_ids: set[int] = set()
    runtime_ecotype_ids: set[int] = set()
    for stats in agent_controller_stats:
        terrain_occupancy.update(
            {
                str(terrain): int(count)
                for terrain, count in dict(stats.get("terrain_counts", {})).items()
            }
        )
        dominant_terrain = stats.get("dominant_terrain")
        if isinstance(dominant_terrain, str):
            dominant_terrain_counts.update([dominant_terrain])
        trophic_role = stats.get("trophic_role")
        if isinstance(trophic_role, str):
            trophic_role_counts.update([trophic_role])
        meat_mode = stats.get("meat_mode")
        if isinstance(meat_mode, str):
            meat_mode_counts.update([meat_mode])
        specialization_profile = stats.get("specialization_profile")
        if isinstance(specialization_profile, str):
            specialization_profile_counts.update([specialization_profile])
        action_counts.update(
            {
                str(action): int(count)
                for action, count in dict(stats.get("action_counts", {})).items()
            }
        )
        lineage_id = _int_value(stats.get("lineage_id"))
        if lineage_id is not None:
            lineage_ids.add(lineage_id)
        species_id = _int_value(stats.get("runtime_species_id"))
        if species_id is not None:
            runtime_species_ids.add(species_id)
        ecotype_id = _int_value(stats.get("runtime_ecotype_id"))
        if ecotype_id is not None:
            runtime_ecotype_ids.add(ecotype_id)
    return {
        "policy": "mind_v3_behavior_descriptor_v1",
        "terrain_occupancy": dict(sorted(terrain_occupancy.items())),
        "dominant_terrain_counts": dict(sorted(dominant_terrain_counts.items())),
        "trophic_role_counts": dict(sorted(trophic_role_counts.items())),
        "meat_mode_counts": dict(sorted(meat_mode_counts.items())),
        "specialization_profile_counts": dict(
            sorted(specialization_profile_counts.items())
        ),
        "requested_action_counts": dict(sorted(action_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "lineage_count": len(lineage_ids),
        "runtime_species_count": len(runtime_species_ids),
        "runtime_ecotype_count": len(runtime_ecotype_ids),
    }


def _run_controller_lineage_elites(
    agent_controller_stats: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not agent_controller_stats:
        return []
    selected: list[dict[str, object]] = []
    seen_agents: set[int] = set()

    def add_best(
        reason: str,
        candidates: list[dict[str, object]],
    ) -> None:
        available = [
            candidate
            for candidate in candidates
            if int(candidate["agent_id"]) not in seen_agents
        ]
        if not available:
            return
        best = max(available, key=_agent_controller_stats_key)
        seen_agents.add(int(best["agent_id"]))
        selected.append(_lineage_elite(reason, best))

    add_best(
        "reproducer",
        [
            stats
            for stats in agent_controller_stats
            if int(stats.get("reproduced_count", 0)) > 0
        ],
    )
    add_best(
        "survivor",
        [stats for stats in agent_controller_stats if bool(stats.get("alive"))],
    )
    add_best("long_lived", agent_controller_stats)
    add_best(
        "movement",
        [
            stats
            for stats in agent_controller_stats
            if int(stats.get("movement_event_count", 0)) > 0
        ],
    )

    for terrain in sorted(
        {
            str(stats.get("dominant_terrain"))
            for stats in agent_controller_stats
            if isinstance(stats.get("dominant_terrain"), str)
        }
    ):
        add_best(
            f"terrain:{terrain}",
            [
                stats
                for stats in agent_controller_stats
                if stats.get("dominant_terrain") == terrain
            ],
        )
    for role in sorted(
        {
            str(stats.get("trophic_role"))
            for stats in agent_controller_stats
            if isinstance(stats.get("trophic_role"), str)
        }
    ):
        add_best(
            f"trophic_role:{role}",
            [
                stats
                for stats in agent_controller_stats
                if stats.get("trophic_role") == role
            ],
        )
    return _diverse_lineage_elites(selected)


def _lineage_elite(
    reason: str,
    stats: dict[str, object],
) -> dict[str, object]:
    return {
        "selection_reason": reason,
        "agent_id": int(stats["agent_id"]),
        "lineage_id": _int_value(stats.get("lineage_id")),
        "runtime_species_id": _int_value(stats.get("runtime_species_id")),
        "runtime_ecotype_id": _int_value(stats.get("runtime_ecotype_id")),
        "alive": bool(stats.get("alive", False)),
        "record_count": int(stats.get("record_count", 0)),
        "first_tick": int(stats.get("first_tick", 0)),
        "last_tick": int(stats.get("last_tick", 0)),
        "reproduced_count": int(stats.get("reproduced_count", 0)),
        "resource_event_rate": float(stats.get("resource_event_rate", 0.0)),
        "movement_event_rate": float(stats.get("movement_event_rate", 0.0)),
        "dominant_terrain": str(stats.get("dominant_terrain", "unknown")),
        "terrain_counts": dict(stats.get("terrain_counts", {})),
        "trophic_role": str(stats.get("trophic_role", "unknown")),
        "meat_mode": str(stats.get("meat_mode", "unknown")),
        "controller_metadata": dict(stats["controller_metadata"]),
    }


def _candidate_controller_lineage_elites(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    elites: list[dict[str, object]] = []
    for run in runs:
        raw_elites = run.get("controller_lineage_elites")
        if not isinstance(raw_elites, list):
            continue
        for raw_elite in raw_elites:
            if not isinstance(raw_elite, Mapping):
                continue
            metadata = raw_elite.get("controller_metadata")
            if not isinstance(metadata, dict):
                continue
            elite = dict(raw_elite)
            elite["seed"] = int(run["seed"])
            elite["controller_metadata"] = dict(metadata)
            elites.append(elite)
    elites.sort(key=_lineage_elite_sort_key, reverse=True)
    return _diverse_lineage_elites(elites)


def _diverse_lineage_elites(
    elites: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    for elite in elites:
        signature = _lineage_elite_signature(elite)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(elite)
        if len(selected) >= MIND_V3_CONTROLLER_LINEAGE_ELITE_LIMIT:
            return selected
    for elite in elites:
        if elite in selected:
            continue
        selected.append(elite)
        if len(selected) >= MIND_V3_CONTROLLER_LINEAGE_ELITE_LIMIT:
            break
    return selected


def _lineage_elite_signature(elite: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(elite.get("dominant_terrain", "unknown")),
        str(elite.get("trophic_role", "unknown")),
        str(elite.get("meat_mode", "unknown")),
    )


def _selected_controller_metadata(
    *,
    fallback: dict[str, object],
    controller_lineage_elites: list[dict[str, object]],
) -> dict[str, object]:
    if not controller_lineage_elites:
        return dict(fallback)
    metadata = controller_lineage_elites[0].get("controller_metadata")
    return dict(metadata) if isinstance(metadata, dict) else dict(fallback)


def _parent_controller_metadata_for_child(
    parent: dict[str, object],
    *,
    child_index: int,
) -> dict[str, object]:
    elites = [
        elite
        for elite in list(parent.get("controller_lineage_elites", []))
        if isinstance(elite, dict) and isinstance(elite.get("controller_metadata"), dict)
    ]
    if not elites:
        return dict(parent["controller_metadata"])
    elite = elites[(child_index - 1) % len(elites)]
    return dict(elite["controller_metadata"])


def _aggregate_run_action_counts(
    runs: list[dict[str, object]],
    *,
    key: str,
    descriptor_fallback_key: str | None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for run in runs:
        run_counts = _counts_from_mapping(run.get(key))
        if not run_counts and descriptor_fallback_key is not None:
            descriptors = run.get("behavior_descriptors")
            if isinstance(descriptors, Mapping):
                run_counts = _counts_from_mapping(
                    descriptors.get(descriptor_fallback_key)
                )
        counts.update(run_counts)
    return counts


def _aggregate_rollout_context_diagnostics(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    count = 0
    non_empty_count = 0
    post_carrion_count = 0
    nonzero_count = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    per_action: dict[str, dict[str, float | int]] = {}
    for run in runs:
        run_count = int(run.get("rollout_context_decision_count", 0))
        count += run_count
        non_empty_count += int(run.get("rollout_context_non_empty_count", 0))
        post_carrion_count += int(
            run.get("rollout_context_post_carrion_context_count", 0)
        )
        nonzero_count += int(
            run.get("rollout_context_selected_score_delta_nonzero_count", 0)
        )
        delta_sum += (
            float(run.get("rollout_context_selected_score_delta_mean", 0.0))
            * run_count
        )
        delta_abs_sum += (
            float(run.get("rollout_context_selected_score_delta_abs_mean", 0.0))
            * run_count
        )
        delta_abs_max = max(
            delta_abs_max,
            float(run.get("rollout_context_selected_score_delta_abs_max", 0.0)),
        )
        raw_per_action = run.get(
            "rollout_context_selected_score_delta_by_requested_action"
        )
        if not isinstance(raw_per_action, Mapping):
            continue
        for action, raw_stats in raw_per_action.items():
            if not isinstance(raw_stats, Mapping):
                continue
            action_count = int(raw_stats.get("count", 0))
            action_stats = per_action.setdefault(
                str(action),
                {
                    "count": 0,
                    "nonzero_count": 0,
                    "delta_sum": 0.0,
                    "delta_abs_sum": 0.0,
                    "delta_abs_max": 0.0,
                },
            )
            action_stats["count"] = int(action_stats["count"]) + action_count
            action_stats["nonzero_count"] = int(action_stats["nonzero_count"]) + int(
                raw_stats.get("nonzero_count", 0)
            )
            action_stats["delta_sum"] = float(action_stats["delta_sum"]) + (
                float(raw_stats.get("mean", 0.0)) * action_count
            )
            action_stats["delta_abs_sum"] = float(action_stats["delta_abs_sum"]) + (
                float(raw_stats.get("abs_mean", 0.0)) * action_count
            )
            action_stats["delta_abs_max"] = max(
                float(action_stats["delta_abs_max"]),
                float(raw_stats.get("abs_max", 0.0)),
            )
    return _rollout_context_summary_from_totals(
        count=count,
        non_empty_count=non_empty_count,
        post_carrion_count=post_carrion_count,
        nonzero_count=nonzero_count,
        delta_sum=delta_sum,
        delta_abs_sum=delta_abs_sum,
        delta_abs_max=delta_abs_max,
        per_action=per_action,
    )


def _aggregate_recovery_context_diagnostics(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    count = 0
    non_empty_count = 0
    post_carrion_count = 0
    drink_available_count = 0
    nonzero_count = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    per_action: dict[str, dict[str, float | int]] = {}
    for run in runs:
        run_count = int(run.get("recovery_context_decision_count", 0))
        count += run_count
        non_empty_count += int(run.get("recovery_context_non_empty_count", 0))
        post_carrion_count += int(
            run.get("recovery_context_post_carrion_context_count", 0)
        )
        drink_available_count += int(
            run.get("recovery_context_drink_available_count", 0)
        )
        nonzero_count += int(
            run.get("recovery_context_selected_score_delta_nonzero_count", 0)
        )
        delta_sum += (
            float(run.get("recovery_context_selected_score_delta_mean", 0.0))
            * run_count
        )
        delta_abs_sum += (
            float(run.get("recovery_context_selected_score_delta_abs_mean", 0.0))
            * run_count
        )
        delta_abs_max = max(
            delta_abs_max,
            float(run.get("recovery_context_selected_score_delta_abs_max", 0.0)),
        )
        raw_per_action = run.get(
            "recovery_context_selected_score_delta_by_requested_action"
        )
        if not isinstance(raw_per_action, Mapping):
            continue
        for action, raw_stats in raw_per_action.items():
            if not isinstance(raw_stats, Mapping):
                continue
            action_count = int(raw_stats.get("count", 0))
            action_stats = per_action.setdefault(
                str(action),
                {
                    "count": 0,
                    "nonzero_count": 0,
                    "delta_sum": 0.0,
                    "delta_abs_sum": 0.0,
                    "delta_abs_max": 0.0,
                },
            )
            action_stats["count"] = int(action_stats["count"]) + action_count
            action_stats["nonzero_count"] = int(action_stats["nonzero_count"]) + int(
                raw_stats.get("nonzero_count", 0)
            )
            action_stats["delta_sum"] = float(action_stats["delta_sum"]) + (
                float(raw_stats.get("mean", 0.0)) * action_count
            )
            action_stats["delta_abs_sum"] = float(action_stats["delta_abs_sum"]) + (
                float(raw_stats.get("abs_mean", 0.0)) * action_count
            )
            action_stats["delta_abs_max"] = max(
                float(action_stats["delta_abs_max"]),
                float(raw_stats.get("abs_max", 0.0)),
            )
    return _recovery_context_summary_from_totals(
        count=count,
        non_empty_count=non_empty_count,
        post_carrion_count=post_carrion_count,
        drink_available_count=drink_available_count,
        nonzero_count=nonzero_count,
        delta_sum=delta_sum,
        delta_abs_sum=delta_abs_sum,
        delta_abs_max=delta_abs_max,
        per_action=per_action,
    )


def _aggregate_unsupported_action_breakdowns(
    runs: list[dict[str, object]],
    *,
    key: str,
) -> dict[str, object]:
    by_requested: Counter[str] = Counter()
    by_resolved: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_tuple: Counter[tuple[str, str, str]] = Counter()
    by_seed: dict[str, object] = {}
    for run in runs:
        raw = run.get(key)
        if not isinstance(raw, Mapping):
            continue
        seed = str(raw.get("seed", run.get("seed", "unknown")))
        by_seed[seed] = dict(raw)
        by_requested.update(_counts_from_mapping(raw.get("by_requested_action")))
        by_resolved.update(_counts_from_mapping(raw.get("by_resolved_action")))
        by_reason.update(_counts_from_mapping(raw.get("by_invalid_reason")))
        raw_tuples = raw.get("by_requested_resolved_invalid_reason")
        if isinstance(raw_tuples, list):
            for item in raw_tuples:
                if not isinstance(item, Mapping):
                    continue
                by_tuple[
                    (
                        str(item.get("requested_action", "unknown")),
                        str(item.get("resolved_action", "unknown")),
                        str(item.get("invalid_reason", "unknown")),
                    )
                ] += int(item.get("count", 0))
    return _unsupported_action_breakdown_from_counters(
        seed=None,
        by_requested=by_requested,
        by_resolved=by_resolved,
        by_reason=by_reason,
        by_tuple=by_tuple,
        by_seed=dict(sorted(by_seed.items())),
    )


def _counts_from_mapping(payload: object) -> Counter[str]:
    if not isinstance(payload, Mapping):
        return Counter()
    return Counter({str(name): int(count) for name, count in payload.items()})


def _dominant_action_summary(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(
        sorted(counts.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return {
        "action": str(action),
        "count": int(count),
        "share": _round(int(count) / float(total)),
    }


def _aggregate_behavior_descriptors(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    terrain_occupancy: Counter[str] = Counter()
    dominant_terrain_counts: Counter[str] = Counter()
    trophic_role_counts: Counter[str] = Counter()
    meat_mode_counts: Counter[str] = Counter()
    specialization_profile_counts: Counter[str] = Counter()
    requested_action_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    lineage_count = 0
    runtime_species_count = 0
    runtime_ecotype_count = 0
    for run in runs:
        descriptors = run.get("behavior_descriptors")
        if not isinstance(descriptors, Mapping):
            continue
        terrain_occupancy.update(_counter_from_mapping(descriptors, "terrain_occupancy"))
        dominant_terrain_counts.update(
            _counter_from_mapping(descriptors, "dominant_terrain_counts")
        )
        trophic_role_counts.update(
            _counter_from_mapping(descriptors, "trophic_role_counts")
        )
        meat_mode_counts.update(_counter_from_mapping(descriptors, "meat_mode_counts"))
        specialization_profile_counts.update(
            _counter_from_mapping(descriptors, "specialization_profile_counts")
        )
        requested_action_counts.update(
            _counter_from_mapping(descriptors, "requested_action_counts")
        )
        action_source_counts.update(
            _counter_from_mapping(descriptors, "action_source_counts")
        )
        lineage_count += int(descriptors.get("lineage_count", 0))
        runtime_species_count += int(descriptors.get("runtime_species_count", 0))
        runtime_ecotype_count += int(descriptors.get("runtime_ecotype_count", 0))
    return {
        "policy": "mind_v3_behavior_descriptor_v1",
        "terrain_occupancy": dict(sorted(terrain_occupancy.items())),
        "dominant_terrain_counts": dict(sorted(dominant_terrain_counts.items())),
        "trophic_role_counts": dict(sorted(trophic_role_counts.items())),
        "meat_mode_counts": dict(sorted(meat_mode_counts.items())),
        "specialization_profile_counts": dict(
            sorted(specialization_profile_counts.items())
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "lineage_count": lineage_count,
        "runtime_species_count": runtime_species_count,
        "runtime_ecotype_count": runtime_ecotype_count,
    }


def _behavior_niche_keys_from_runs(runs: list[dict[str, object]]) -> list[str]:
    keys: set[str] = set()
    for run in runs:
        raw_elites = run.get("controller_lineage_elites")
        if not isinstance(raw_elites, list):
            continue
        for raw_elite in raw_elites:
            if not isinstance(raw_elite, Mapping):
                continue
            signature = _lineage_elite_signature(raw_elite)
            if signature == ("unknown", "unknown", "unknown"):
                continue
            keys.add(_behavior_niche_key(signature))
    return sorted(keys)


def _active_behavior_niche_keys_from_runs(runs: list[dict[str, object]]) -> list[str]:
    keys: set[str] = set()
    for run in runs:
        raw_elites = run.get("controller_lineage_elites")
        if not isinstance(raw_elites, list):
            continue
        for raw_elite in raw_elites:
            if not isinstance(raw_elite, Mapping):
                continue
            if not _lineage_elite_is_active(raw_elite):
                continue
            signature = _lineage_elite_signature(raw_elite)
            if signature == ("unknown", "unknown", "unknown"):
                continue
            keys.add(_behavior_niche_key(signature))
    return sorted(keys)


def _lineage_elite_is_active(elite: Mapping[str, object]) -> bool:
    return (
        bool(elite.get("alive", False))
        or int(elite.get("reproduced_count", 0)) > 0
        or int(elite.get("movement_event_count", 0)) > 0
        or float(elite.get("movement_event_rate", 0.0)) > 0.0
    )


def _counter_from_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Counter[str]:
    raw = payload.get(key)
    if not isinstance(raw, Mapping):
        return Counter()
    return Counter({str(name): int(count) for name, count in raw.items()})


def _agent_controller_stats_key(stats: Mapping[str, object]) -> tuple:
    return (
        bool(stats.get("alive", False)),
        int(stats.get("reproduced_count", 0)),
        int(stats.get("last_tick", 0)),
        int(stats.get("record_count", 0)),
        int(stats.get("resource_event_count", 0)),
        int(stats.get("movement_event_count", 0)),
        -int(stats.get("agent_id", 0)),
    )


def _lineage_elite_sort_key(elite: Mapping[str, object]) -> tuple:
    return (
        bool(elite.get("alive", False)),
        int(elite.get("reproduced_count", 0)),
        int(elite.get("last_tick", 0)),
        int(elite.get("record_count", 0)),
        float(elite.get("resource_event_rate", 0.0)),
        float(elite.get("movement_event_rate", 0.0)),
        -int(elite.get("agent_id", 0)),
    )


def _terrain_for_record(
    world: SimulationWorld,
    record: Mapping[str, object],
) -> str:
    before = record.get("before")
    if not isinstance(before, Mapping):
        return "unknown"
    x = _int_value(before.get("x"))
    y = _int_value(before.get("y"))
    if x is None or y is None:
        return "unknown"
    if y < 0 or y >= len(world.grid) or x < 0 or x >= len(world.grid[y]):
        return "unknown"
    return str(world.grid[y][x].terrain)


def _agent_trophic_role(world: SimulationWorld, agent: object) -> str:
    trophic_role = getattr(world, "_trophic_role", None)
    return str(trophic_role(agent)) if callable(trophic_role) else "unknown"


def _agent_meat_mode(world: SimulationWorld, agent: object) -> str:
    meat_mode = getattr(world, "_meat_mode", None)
    return str(meat_mode(agent)) if callable(meat_mode) else "unknown"


def _dominant_counter_key(counts: Counter[str]) -> str:
    if not counts:
        return "unknown"
    return max(
        sorted(counts),
        key=lambda key: (counts[key], key),
    )


def _int_value(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    action_source_counts: Counter[str] = Counter()
    policy_id_counts: Counter[str] = Counter()
    trajectory_record_count = sum(int(run["trajectory_record_count"]) for run in runs)
    resource_event_count = sum(int(run["resource_event_count"]) for run in runs)
    animal_resource_event_count = sum(
        int(run.get("animal_resource_event_count", 0)) for run in runs
    )
    fresh_kill_event_count = sum(
        int(run.get("fresh_kill_event_count", 0)) for run in runs
    )
    carcass_event_count = sum(
        int(run.get("carcass_event_count", 0)) for run in runs
    )
    animal_resource_gain_total = sum(
        float(run.get("animal_resource_gain_total", 0.0)) for run in runs
    )
    movement_event_count = sum(int(run["movement_event_count"]) for run in runs)
    behavior_descriptors = _aggregate_behavior_descriptors(runs)
    behavior_niche_keys = _behavior_niche_keys_from_runs(runs)
    active_behavior_niche_keys = _active_behavior_niche_keys_from_runs(runs)
    requested_action_counts = _aggregate_run_action_counts(
        runs,
        key="requested_action_counts",
        descriptor_fallback_key="requested_action_counts",
    )
    resolved_action_counts = _aggregate_run_action_counts(
        runs,
        key="resolved_action_counts",
        descriptor_fallback_key=None,
    )
    unsupported_requested_action_count = sum(
        int(run.get("unsupported_requested_action_count", 0)) for run in runs
    )
    unsupported_resolved_action_count = sum(
        int(run.get("unsupported_resolved_action_count", 0)) for run in runs
    )
    unsupported_requested_action_breakdown = _aggregate_unsupported_action_breakdowns(
        runs,
        key="unsupported_requested_action_breakdown",
    )
    unsupported_resolved_action_breakdown = _aggregate_unsupported_action_breakdowns(
        runs,
        key="unsupported_resolved_action_breakdown",
    )
    rollout_context_diagnostics = _aggregate_rollout_context_diagnostics(runs)
    recovery_context_diagnostics = _aggregate_recovery_context_diagnostics(runs)
    core_blocker_counts = _aggregate_run_action_counts(
        runs,
        key="core_blocker_agent_tick_counts",
        descriptor_fallback_key=None,
    )
    primary_core_blocker_counts = _aggregate_run_action_counts(
        runs,
        key="primary_core_blocker_agent_tick_counts",
        descriptor_fallback_key=None,
    )
    dominant_action = _dominant_action_summary(requested_action_counts)
    terminal_reproduction_viabilities = [
        min(
            float(run.get("terminal_energy_viability_share", 0.0)),
            float(run.get("terminal_hydration_viability_share", 0.0)),
            float(run.get("terminal_health_viability_share", 0.0)),
            float(run.get("terminal_matched_diet_viability_share", 0.0)),
        )
        for run in runs
    ]
    birth_counts = [int(run["births"]) for run in runs]
    for run in runs:
        action_source_counts.update(
            {
                str(source): int(count)
                for source, count in dict(run["action_source_counts"]).items()
            }
        )
        policy_id_counts.update(
            {
                str(policy_id): int(count)
                for policy_id, count in dict(run["policy_id_counts"]).items()
            }
        )
    return {
        "alive_agents_mean": _mean([int(run["alive_agents"]) for run in runs]),
        "births_mean": _mean([int(run["births"]) for run in runs]),
        "deaths_mean": _mean([int(run["deaths"]) for run in runs]),
        "alive_agent_ticks_mean": _mean(
            [int(run["alive_agent_ticks"]) for run in runs]
        ),
        "alive_agent_ticks_per_tick_mean": _mean(
            [float(run["alive_agent_ticks_per_tick"]) for run in runs]
        ),
        "terminal_energy_viability_share_mean": _mean(
            [float(run.get("terminal_energy_viability_share", 0.0)) for run in runs]
        ),
        "terminal_energy_requirement_satisfaction_mean": _mean(
            [
                float(run.get("terminal_energy_requirement_satisfaction", 0.0))
                for run in runs
            ]
        ),
        "terminal_balanced_reproduction_readiness_mean": _mean(
            [
                float(run.get("terminal_balanced_reproduction_readiness", 0.0))
                for run in runs
            ]
        ),
        "terminal_balanced_reproduction_readiness_min": (
            min(
                float(run.get("terminal_balanced_reproduction_readiness", 0.0))
                for run in runs
            )
            if runs
            else 0.0
        ),
        "terminal_energy_hydration_balance_mean": _mean(
            [
                float(run.get("terminal_energy_hydration_balance", 0.0))
                for run in runs
            ]
        ),
        "terminal_energy_hydration_balance_min": (
            min(
                float(run.get("terminal_energy_hydration_balance", 0.0))
                for run in runs
            )
            if runs
            else 0.0
        ),
        "terminal_energy_hydration_gap_abs_mean": _mean(
            [
                float(run.get("terminal_energy_hydration_gap_abs", 0.0))
                for run in runs
            ]
        ),
        "terminal_energy_shortfall_share_mean": _mean(
            [
                float(run.get("terminal_energy_shortfall_share", 0.0))
                for run in runs
            ]
        ),
        "terminal_energy_required_total_mean": _mean(
            [
                float(run.get("terminal_energy_required_total", 0.0))
                for run in runs
            ]
        ),
        "terminal_energy_gap_total_mean": _mean(
            [float(run.get("terminal_energy_gap_total", 0.0)) for run in runs]
        ),
        "terminal_hydration_viability_share_mean": _mean(
            [
                float(run.get("terminal_hydration_viability_share", 0.0))
                for run in runs
            ]
        ),
        "terminal_health_viability_share_mean": _mean(
            [float(run.get("terminal_health_viability_share", 0.0)) for run in runs]
        ),
        "terminal_matched_diet_viability_share_mean": _mean(
            [
                float(run.get("terminal_matched_diet_viability_share", 0.0))
                for run in runs
            ]
        ),
        "terminal_reproduction_viability_run_share": _mean(
            [1 if viability > 0.0 else 0 for viability in terminal_reproduction_viabilities]
        ),
        "terminal_reproduction_viability_min": (
            min(terminal_reproduction_viabilities)
            if terminal_reproduction_viabilities
            else 0.0
        ),
        "birth_positive_run_share": _mean(
            [1 if birth_count > 0 else 0 for birth_count in birth_counts]
        ),
        "reproduction_ready_agents_mean": _mean(
            [int(run.get("reproduction_ready_agents", 0)) for run in runs]
        ),
        "biologically_reproduction_ready_agents_mean": _mean(
            [
                int(run.get("biologically_reproduction_ready_agents", 0))
                for run in runs
            ]
        ),
        "terminal_biologically_ready_group_count_mean": _mean(
            [
                int(run.get("terminal_biologically_ready_group_count", 0))
                for run in runs
            ]
        ),
        "terminal_ready_group_count_mean": _mean(
            [int(run.get("terminal_ready_group_count", 0)) for run in runs]
        ),
        "core_readiness_agent_tick_mean": _mean(
            [
                float(run.get("core_readiness_agent_tick_mean", 0.0))
                for run in runs
            ]
        ),
        "core_ready_agent_tick_share_mean": _mean(
            [
                float(run.get("core_ready_agent_tick_share", 0.0))
                for run in runs
            ]
        ),
        "core_ready_tick_share_mean": _mean(
            [float(run.get("core_ready_tick_share", 0.0)) for run in runs]
        ),
        "core_ready_pair_tick_share_mean": _mean(
            [
                float(run.get("core_ready_pair_tick_share", 0.0))
                for run in runs
            ]
        ),
        "core_ready_distinct_agent_count_mean": _mean(
            [
                int(run.get("core_ready_distinct_agent_count", 0))
                for run in runs
            ]
        ),
        "reproduction_ready_agent_tick_share_mean": _mean(
            [
                float(run.get("reproduction_ready_agent_tick_share", 0.0))
                for run in runs
            ]
        ),
        "reproduction_ready_tick_share_mean": _mean(
            [
                float(run.get("reproduction_ready_tick_share", 0.0))
                for run in runs
            ]
        ),
        "reproduction_ready_pair_tick_share_mean": _mean(
            [
                float(run.get("reproduction_ready_pair_tick_share", 0.0))
                for run in runs
            ]
        ),
        "reproduction_ready_distinct_agent_count_mean": _mean(
            [
                int(run.get("reproduction_ready_distinct_agent_count", 0))
                for run in runs
            ]
        ),
        "resource_event_count": resource_event_count,
        "resource_event_rate": _round(
            resource_event_count / max(1, trajectory_record_count)
        ),
        "animal_resource_event_count": animal_resource_event_count,
        "animal_resource_event_rate": _round(
            animal_resource_event_count / max(1, trajectory_record_count)
        ),
        "fresh_kill_event_count": fresh_kill_event_count,
        "fresh_kill_event_rate": _round(
            fresh_kill_event_count / max(1, trajectory_record_count)
        ),
        "carcass_event_count": carcass_event_count,
        "carcass_event_rate": _round(
            carcass_event_count / max(1, trajectory_record_count)
        ),
        "animal_resource_gain_total_mean": _round(
            animal_resource_gain_total / max(1, len(runs))
        ),
        "movement_event_count": movement_event_count,
        "movement_event_rate": _round(
            movement_event_count / max(1, trajectory_record_count)
        ),
        "unique_requested_actions_mean": _mean(
            [int(run["unique_requested_actions"]) for run in runs]
        ),
        "trajectory_record_count": trajectory_record_count,
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "behavior_descriptors": behavior_descriptors,
        "behavior_niche_count": len(behavior_niche_keys),
        "behavior_niche_keys": behavior_niche_keys,
        "active_behavior_niche_count": len(active_behavior_niche_keys),
        "active_behavior_niche_keys": active_behavior_niche_keys,
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(
            unsupported_resolved_action_count
        ),
        "unsupported_requested_action_breakdown": unsupported_requested_action_breakdown,
        "unsupported_resolved_action_breakdown": unsupported_resolved_action_breakdown,
        **rollout_context_diagnostics,
        **recovery_context_diagnostics,
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "core_blocker_agent_tick_counts": dict(sorted(core_blocker_counts.items())),
        "core_blocker_agent_tick_shares": {
            field: _round(
                int(core_blocker_counts[field]) / max(1, trajectory_record_count)
            )
            for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_counts": dict(
            sorted(primary_core_blocker_counts.items())
        ),
        "primary_core_blocker_agent_tick_shares": {
            field: _round(
                int(primary_core_blocker_counts[field])
                / max(1, trajectory_record_count)
            )
            for field in _core_fields()
        },
        "primary_core_blocker": _dominant_counter_key(primary_core_blocker_counts),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }


def _score_components(aggregate: dict[str, object]) -> dict[str, float]:
    dominant_action_share = float(
        aggregate.get("dominant_requested_action_share", 0.0)
    )
    collapse_fraction = 0.0
    if dominant_action_share > MIND_V3_DOMINANT_ACTION_COLLAPSE_SHARE:
        collapse_fraction = min(
            1.0,
            (
                dominant_action_share
                - MIND_V3_DOMINANT_ACTION_COLLAPSE_SHARE
            )
            / max(1e-9, 1.0 - MIND_V3_DOMINANT_ACTION_COLLAPSE_SHARE),
        )
    active_behavior_niche_count = int(
        aggregate.get("active_behavior_niche_count", 0)
    )
    terminal_energy_viability = float(
        aggregate.get("terminal_energy_viability_share_mean", 0.0)
    )
    terminal_energy_requirement_satisfaction = float(
        aggregate.get("terminal_energy_requirement_satisfaction_mean", 0.0)
    )
    terminal_hydration_viability = float(
        aggregate.get("terminal_hydration_viability_share_mean", 0.0)
    )
    terminal_health_viability = float(
        aggregate.get("terminal_health_viability_share_mean", 0.0)
    )
    terminal_matched_diet_viability = float(
        aggregate.get("terminal_matched_diet_viability_share_mean", 0.0)
    )
    terminal_balanced_readiness = float(
        aggregate.get(
            "terminal_balanced_reproduction_readiness_mean",
            _balanced_readiness_score(
                [
                    terminal_energy_requirement_satisfaction,
                    terminal_hydration_viability,
                    terminal_health_viability,
                    terminal_matched_diet_viability,
                ]
            ),
        )
    )
    terminal_energy_hydration_balance = float(
        aggregate.get(
            "terminal_energy_hydration_balance_mean",
            min(
                terminal_energy_requirement_satisfaction,
                terminal_hydration_viability,
            ),
        )
    )
    terminal_energy_hydration_gap_abs = float(
        aggregate.get(
            "terminal_energy_hydration_gap_abs_mean",
            abs(
                terminal_energy_requirement_satisfaction
                - terminal_hydration_viability
            ),
        )
    )
    terminal_reproduction_viability = min(
        terminal_energy_viability,
        terminal_hydration_viability,
        terminal_health_viability,
        terminal_matched_diet_viability,
    )
    alive_agents_mean = float(aggregate["alive_agents_mean"])
    births_mean = float(aggregate["births_mean"])
    birth_positive_run_share = float(
        aggregate.get("birth_positive_run_share", 0.0)
    )
    terminal_reproduction_viability_run_share = float(
        aggregate.get("terminal_reproduction_viability_run_share", 0.0)
    )
    sustained_ready_agent_share = float(
        aggregate.get("reproduction_ready_agent_tick_share_mean", 0.0)
    )
    sustained_ready_tick_share = float(
        aggregate.get("reproduction_ready_tick_share_mean", 0.0)
    )
    sustained_ready_pair_tick_share = float(
        aggregate.get("reproduction_ready_pair_tick_share_mean", 0.0)
    )
    sustained_core_readiness = float(
        aggregate.get("core_readiness_agent_tick_mean", 0.0)
    )
    sustained_core_agent_share = float(
        aggregate.get("core_ready_agent_tick_share_mean", 0.0)
    )
    sustained_core_pair_tick_share = float(
        aggregate.get("core_ready_pair_tick_share_mean", 0.0)
    )
    brittleness_applies = (
        births_mean > 0.0 or terminal_reproduction_viability > 0.0
    )
    return {
        "alive_agents": _round(3.0 * alive_agents_mean),
        "births": _round(8.0 * births_mean),
        "deaths": _round(-0.35 * float(aggregate["deaths_mean"])),
        "alive_agent_ticks": _round(
            0.15 * float(aggregate["alive_agent_ticks_per_tick_mean"])
        ),
        "terminal_energy_viability": _round(
            4.0 * terminal_energy_viability
        ),
        "terminal_energy_requirement_satisfaction": _round(
            6.0 * terminal_energy_requirement_satisfaction
        ),
        "terminal_balanced_reproduction_readiness": _round(
            10.0 * terminal_balanced_readiness
        ),
        "terminal_energy_hydration_balance": _round(
            8.0 * terminal_energy_hydration_balance
        ),
        "terminal_energy_hydration_imbalance": _round(
            -3.0 * terminal_energy_hydration_gap_abs
        ),
        "terminal_hydration_viability": _round(
            4.0 * terminal_hydration_viability
        ),
        "terminal_health_viability": _round(
            2.0 * terminal_health_viability
        ),
        "terminal_matched_diet_viability": _round(
            3.0 * terminal_matched_diet_viability
        ),
        "terminal_reproduction_viability": _round(
            10.0 * terminal_reproduction_viability
        ),
        "terminal_reproduction_viability_coverage": _round(
            3.0 * terminal_reproduction_viability_run_share
        ),
        "reproduction_readiness": _round(
            4.0
            * float(
                aggregate.get("biologically_reproduction_ready_agents_mean", 0.0)
            )
            + 8.0 * float(aggregate.get("reproduction_ready_agents_mean", 0.0))
            + 2.0
            * float(
                aggregate.get("terminal_biologically_ready_group_count_mean", 0.0)
            )
            + 4.0 * float(aggregate.get("terminal_ready_group_count_mean", 0.0))
        ),
        "sustained_reproduction_readiness": _round(
            12.0 * sustained_ready_agent_share
            + 5.0 * sustained_ready_tick_share
            + 8.0 * sustained_ready_pair_tick_share
            + 4.0 * sustained_core_readiness
            + 4.0 * sustained_core_agent_share
            + 4.0 * sustained_core_pair_tick_share
        ),
        "resource_events": _round(
            0.75 * min(0.75, float(aggregate["resource_event_rate"]))
        ),
        "animal_resource_events": _round(
            0.85 * min(0.75, float(aggregate.get("animal_resource_event_rate", 0.0)))
        ),
        "carrion_resource_events": _round(
            0.45 * min(0.5, float(aggregate.get("carcass_event_rate", 0.0)))
        ),
        "movement_events": _round(
            0.5 * min(0.5, float(aggregate["movement_event_rate"]))
        ),
        "action_diversity": _round(
            0.05 * float(aggregate["unique_requested_actions_mean"])
        ),
        "active_behavior_niches": _round(
            MIND_V3_ACTIVE_BEHAVIOR_NICHE_BONUS
            * min(
                MIND_V3_ACTIVE_BEHAVIOR_NICHE_BONUS_CAP,
                active_behavior_niche_count,
            )
        ),
        "dominant_action_collapse": _round(
            -MIND_V3_DOMINANT_ACTION_COLLAPSE_MAX_PENALTY * collapse_fraction
        ),
        "terminal_reproduction_dead_end": (
            -MIND_V3_TERMINAL_REPRODUCTION_DEAD_END_PENALTY
            if alive_agents_mean > 0.0 and terminal_reproduction_viability <= 0.0
            else 0.0
        ),
        "sustained_ready_dead_end": (
            -MIND_V3_SUSTAINED_READY_DEAD_END_PENALTY
            if (
                births_mean > 0.0
                and sustained_ready_agent_share <= 0.0
                and sustained_core_agent_share <= 0.0
            )
            else 0.0
        ),
        "sustained_ready_pair_dead_end": (
            -MIND_V3_SUSTAINED_READY_PAIR_DEAD_END_PENALTY
            if (
                births_mean > 1.0
                and alive_agents_mean > 1.0
                and sustained_ready_pair_tick_share <= 0.0
                and sustained_core_pair_tick_share <= 0.0
            )
            else 0.0
        ),
        "seed_brittle_birth": (
            _round(
                -MIND_V3_SEED_BRITTLE_BIRTH_PENALTY
                * (1.0 - birth_positive_run_share)
            )
            if births_mean > 0.0
            else 0.0
        ),
        "terminal_reproduction_brittleness": (
            _round(
                -MIND_V3_TERMINAL_REPRODUCTION_BRITTLENESS_PENALTY
                * (1.0 - terminal_reproduction_viability_run_share)
            )
            if brittleness_applies
            else 0.0
        ),
    }


def _score_components_total(components: dict[str, float]) -> float:
    return _round(sum(float(value) for value in components.values()))


def _archive_from_generations(
    generations: list[dict[str, object]],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for generation in generations:
        candidates.extend(list(generation.get("candidates", [])))
    return _archive_from_candidates(candidates)


def _archive_from_candidates(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        return {
            "policy": MIND_V3_ARCHIVE_POLICY,
            "elites": {},
            "behavior_niches": {},
            "niche_count": 0,
        }
    elites = {
        "balanced": max(candidates, key=_balanced_key),
        "survival": max(candidates, key=_survival_key),
        "births": max(candidates, key=_births_key),
        "resource_use": max(candidates, key=_resource_key),
        "movement": max(candidates, key=_movement_key),
        "low_death": max(candidates, key=_low_death_key),
    }
    scavenger_candidates = [
        candidate
        for candidate in candidates
        if _candidate_scavenger_lane_signal(candidate) > 0.0
    ]
    if scavenger_candidates:
        elites["scavenger_lane"] = max(
            scavenger_candidates,
            key=_scavenger_lane_key,
        )
    fixture_selection_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate.get("fixture_selection"), Mapping)
    ]
    if fixture_selection_candidates:
        elites["fixture_selection"] = max(
            fixture_selection_candidates,
            key=_fixture_selection_elite_key,
        )
    behavior_niches = _behavior_niches_from_candidates(candidates)
    return {
        "policy": MIND_V3_ARCHIVE_POLICY,
        "elites": {
            name: dict(candidate)
            for name, candidate in elites.items()
        },
        "behavior_niches": behavior_niches,
        "niche_count": len(behavior_niches),
    }


def _behavior_niches_from_candidates(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        raw_elites = candidate.get("controller_lineage_elites")
        if not isinstance(raw_elites, list):
            continue
        for raw_elite in raw_elites:
            if not isinstance(raw_elite, Mapping):
                continue
            metadata = raw_elite.get("controller_metadata")
            if not isinstance(metadata, dict):
                continue
            signature = _lineage_elite_signature(raw_elite)
            if signature == ("unknown", "unknown", "unknown"):
                continue
            niche_key = _behavior_niche_key(signature)
            elite = dict(raw_elite)
            elite["controller_metadata"] = dict(metadata)
            cell = {
                "signature": _behavior_niche_signature_dict(signature),
                "candidate": _archive_candidate_reference(candidate),
                "lineage_elite": elite,
                "score": float(candidate.get("score", 0.0)),
                "candidate_id": str(candidate.get("candidate_id", "")),
            }
            existing = cells.get(niche_key)
            if existing is None or _behavior_niche_cell_key(
                cell
            ) > _behavior_niche_cell_key(existing):
                cells[niche_key] = cell
    ordered = dict(
        sorted(
            cells.items(),
            key=lambda item: (
                _behavior_niche_cell_key(item[1]),
                item[0],
            ),
            reverse=True,
        )[:MIND_V3_BEHAVIOR_NICHE_LIMIT]
    )
    return dict(sorted(ordered.items()))


def _archive_candidate_reference(candidate: Mapping[str, object]) -> dict[str, object]:
    reference: dict[str, object] = {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "candidate_index": int(candidate.get("candidate_index", 0)),
        "parent_candidate_id": candidate.get("parent_candidate_id"),
        "generation_index": int(candidate.get("generation_index", 0)),
        "score": float(candidate.get("score", 0.0)),
        "alive_agents_mean": float(candidate.get("alive_agents_mean", 0.0)),
        "births_mean": float(candidate.get("births_mean", 0.0)),
        "deaths_mean": float(candidate.get("deaths_mean", 0.0)),
        "alive_agent_ticks_per_tick_mean": float(
            candidate.get("alive_agent_ticks_per_tick_mean", 0.0)
        ),
        "resource_event_rate": float(candidate.get("resource_event_rate", 0.0)),
        "movement_event_rate": float(candidate.get("movement_event_rate", 0.0)),
        "heuristic_action_source_count": int(
            candidate.get("heuristic_action_source_count", 0)
        ),
        "controller_architecture": _candidate_controller_architecture(candidate),
    }
    lane_tags = candidate.get("ecology_lane_tags")
    if isinstance(lane_tags, list):
        reference["ecology_lane_tags"] = [str(tag) for tag in lane_tags]
    reference["scavenger_lane_score"] = _round(
        _candidate_scavenger_lane_signal(candidate)
    )
    metadata = candidate.get("controller_metadata")
    if isinstance(metadata, dict):
        reference["controller_metadata"] = dict(metadata)
    behavior_descriptors = candidate.get("behavior_descriptors")
    if isinstance(behavior_descriptors, dict):
        reference["behavior_descriptors"] = dict(behavior_descriptors)
    lineage_elites = candidate.get("controller_lineage_elites")
    if isinstance(lineage_elites, list):
        reference["controller_lineage_elites"] = [
            dict(elite)
            for elite in lineage_elites
            if isinstance(elite, dict)
        ]
    return reference


def _behavior_niche_key(signature: tuple[str, str, str]) -> str:
    terrain, trophic_role, meat_mode = signature
    return (
        f"terrain={terrain}|trophic_role={trophic_role}|meat_mode={meat_mode}"
    )


def _behavior_niche_signature_dict(
    signature: tuple[str, str, str],
) -> dict[str, str]:
    terrain, trophic_role, meat_mode = signature
    return {
        "dominant_terrain": terrain,
        "trophic_role": trophic_role,
        "meat_mode": meat_mode,
    }


def _behavior_niche_cell_key(cell: Mapping[str, object]) -> tuple:
    candidate = cell.get("candidate")
    elite = cell.get("lineage_elite")
    candidate_payload = candidate if isinstance(candidate, Mapping) else {}
    elite_payload = elite if isinstance(elite, Mapping) else {}
    return (
        bool(elite_payload.get("alive", False)),
        int(elite_payload.get("reproduced_count", 0)),
        int(elite_payload.get("record_count", 0)),
        float(candidate_payload.get("alive_agents_mean", 0.0)),
        float(candidate_payload.get("births_mean", 0.0)),
        float(candidate_payload.get("score", 0.0)),
        -int(candidate_payload.get("candidate_index", 0)),
    )


def _balanced_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(candidate["score"]),
        float(candidate["alive_agents_mean"]),
        float(candidate["births_mean"]),
        -int(candidate["candidate_index"]),
    )


def _survival_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(candidate["alive_agents_mean"]),
        float(candidate["alive_agent_ticks_per_tick_mean"]),
        float(candidate["score"]),
        -int(candidate["candidate_index"]),
    )


def _births_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(candidate["births_mean"]),
        float(candidate["alive_agents_mean"]),
        float(candidate["score"]),
        -int(candidate["candidate_index"]),
    )


def _resource_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(candidate["resource_event_rate"]),
        float(candidate["alive_agents_mean"]),
        float(candidate["score"]),
        -int(candidate["candidate_index"]),
    )


def _movement_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(candidate["movement_event_rate"]),
        float(candidate["alive_agents_mean"]),
        float(candidate["score"]),
        -int(candidate["candidate_index"]),
    )


def _low_death_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        -float(candidate["deaths_mean"]),
        float(candidate["alive_agents_mean"]),
        float(candidate["score"]),
        -int(candidate["candidate_index"]),
    )


def _scavenger_lane_key(candidate: Mapping[str, object]) -> tuple:
    return (
        _candidate_scavenger_lane_signal(candidate),
        float(candidate.get("alive_agents_mean", 0.0)),
        float(candidate.get("births_mean", 0.0)),
        float(candidate.get("terminal_energy_hydration_balance_mean", 0.0)),
        float(
            candidate.get(
                "terminal_balanced_reproduction_readiness_mean",
                0.0,
            )
        ),
        float(candidate.get("terminal_matched_diet_viability_share_mean", 0.0)),
        float(candidate.get("resource_event_rate", 0.0)),
        float(candidate.get("score", 0.0)),
        -int(candidate.get("candidate_index", 0)),
    )


def _fixture_selection_elite_key(candidate: Mapping[str, object]) -> tuple:
    fixture_selection = candidate.get("fixture_selection")
    payload = fixture_selection if isinstance(fixture_selection, Mapping) else {}
    return (
        bool(payload.get("fixture_gate_passed", False)),
        int(payload.get("passed_horizon_count", 0)),
        bool(payload.get("first_horizon_passed", False)),
        -int(payload.get("blocker_count", 0)),
        float(payload.get("score_delta", 0.0)),
        float(candidate.get("score", 0.0)),
        float(candidate.get("alive_agents_mean", 0.0)),
        float(candidate.get("births_mean", 0.0)),
        -int(candidate.get("candidate_index", 0)),
    )


def _candidate_ecology_lane_tags(candidate: Mapping[str, object]) -> list[str]:
    tags: list[str] = []
    if _candidate_scavenger_lane_signal(candidate) > 0.0:
        tags.append("scavenger")
    return tags


def _candidate_scavenger_lane_signal(candidate: Mapping[str, object]) -> float:
    signal = 0.0
    metadata = candidate.get("controller_metadata")
    if isinstance(metadata, Mapping):
        profile = _metadata_specialization_profile(metadata)
        if profile == "scavenger":
            signal += 4.0
        elif profile == "predator_scavenger":
            signal += 1.5
    profile_counts = candidate.get("founder_template_pool_specialization_profile_counts")
    if isinstance(profile_counts, Mapping):
        signal += 3.0 * int(profile_counts.get("scavenger", 0))
        signal += 1.0 * int(profile_counts.get("predator_scavenger", 0))
    descriptors = candidate.get("behavior_descriptors")
    if isinstance(descriptors, Mapping):
        meat_modes = _counter_from_mapping(descriptors, "meat_mode_counts")
        profiles = _counter_from_mapping(
            descriptors,
            "specialization_profile_counts",
        )
        signal += 3.0 * int(meat_modes.get("scavenger", 0))
        signal += 2.0 * int(profiles.get("scavenger", 0))
        signal += 0.75 * int(profiles.get("predator_scavenger", 0))
    raw_elites = candidate.get("controller_lineage_elites")
    if isinstance(raw_elites, list):
        for raw_elite in raw_elites:
            if not isinstance(raw_elite, Mapping):
                continue
            if str(raw_elite.get("meat_mode", "")) == "scavenger":
                signal += 4.0
                if _lineage_elite_is_active(raw_elite):
                    signal += 2.0
            elite_metadata = raw_elite.get("controller_metadata")
            if isinstance(elite_metadata, Mapping):
                profile = _metadata_specialization_profile(elite_metadata)
                if profile == "scavenger":
                    signal += 3.0
                elif profile == "predator_scavenger":
                    signal += 1.0
    return float(signal)


def _best_candidate_from_archive(
    archive: dict[str, object],
) -> dict[str, object] | None:
    elites = archive.get("elites")
    if not isinstance(elites, dict) or not elites:
        return None
    balanced = elites.get("balanced")
    if isinstance(balanced, dict):
        return balanced
    candidate_values = [
        candidate for candidate in elites.values() if isinstance(candidate, dict)
    ]
    if not candidate_values:
        return None
    return max(candidate_values, key=_balanced_key)


def _archive_parent_candidates(archive: dict[str, object]) -> list[dict[str, object]]:
    elites = archive.get("elites")
    if not isinstance(elites, dict):
        elites = {}
    parents: list[dict[str, object]] = []
    seen: set[str] = set()

    def append_parent(parent: dict[str, object], *, parent_key: str) -> None:
        if parent_key in seen:
            return
        seen.add(parent_key)
        parents.append(parent)

    def append_retained_archive_parents() -> None:
        retained = archive.get("fixture_recovery_archive_retained_candidates")
        if not isinstance(retained, list):
            return
        for raw_candidate in retained:
            if not isinstance(raw_candidate, dict):
                continue
            candidate_id = str(raw_candidate.get("candidate_id", ""))
            if not candidate_id:
                continue
            if any(
                str(parent.get("candidate_id", "")) == candidate_id
                for parent in parents
            ):
                continue
            append_parent(
                dict(raw_candidate),
                parent_key=f"fixture_recovery_archive:{candidate_id}",
            )

    for name in ARCHIVE_ORDER:
        candidate = elites.get(name)
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        append_parent(candidate, parent_key=f"metric:{candidate_id}")
        if name == "balanced":
            append_retained_archive_parents()
    append_retained_archive_parents()
    behavior_niches = archive.get("behavior_niches")
    if not isinstance(behavior_niches, dict):
        behavior_niches = {}
    for niche_key in sorted(behavior_niches):
        raw_cell = behavior_niches[niche_key]
        if not isinstance(raw_cell, dict):
            continue
        parent = _niche_parent_candidate(niche_key, raw_cell)
        if parent is None:
            continue
        append_parent(parent, parent_key=f"niche:{niche_key}")
    return parents


def _niche_parent_candidate(
    niche_key: str,
    cell: Mapping[str, object],
) -> dict[str, object] | None:
    raw_candidate = cell.get("candidate")
    raw_elite = cell.get("lineage_elite")
    if not isinstance(raw_candidate, Mapping) or not isinstance(raw_elite, Mapping):
        return None
    metadata = raw_elite.get("controller_metadata")
    if not isinstance(metadata, dict):
        return None
    elite = dict(raw_elite)
    elite["controller_metadata"] = dict(metadata)
    parent = dict(raw_candidate)
    parent["controller_metadata"] = dict(metadata)
    parent["controller_lineage_elites"] = [elite]
    parent["archive_parent_niche"] = niche_key
    return parent


def _candidate_with_founder_template_pool(
    candidate: dict[str, object],
    *,
    archive: dict[str, object],
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    enriched = dict(candidate)
    existing_pool = candidate.get("founder_template_pool")
    if (
        isinstance(existing_pool, list)
        and existing_pool
        and (
            isinstance(candidate.get("fixture_repair"), Mapping)
            or isinstance(candidate.get("warm_start"), Mapping)
        )
    ):
        pool = [
            dict(metadata)
            for metadata in existing_pool
            if isinstance(metadata, dict)
        ]
        if pool:
            enriched["founder_template_pool"] = pool
            enriched["founder_template_pool_size"] = len(pool)
            enriched["founder_template_pool_specialization_profile_counts"] = (
                _metadata_specialization_profile_counts(pool)
            )
            enriched.update(_founder_template_pool_identity_diagnostics(pool))
            return enriched
    pool = _founder_template_pool_from_archive(
        archive=archive,
        fallback=candidate,
        candidates=candidates or [],
    )
    enriched["founder_template_pool_policy"] = MIND_V3_FOUNDER_TEMPLATE_POOL_POLICY
    enriched["founder_template_pool_min_alive_agents_mean"] = (
        MIND_V3_FOUNDER_TEMPLATE_POOL_MIN_ALIVE_MEAN
    )
    enriched["founder_template_pool"] = pool
    enriched["founder_template_pool_size"] = len(pool)
    enriched["founder_template_pool_specialization_profile_counts"] = (
        _metadata_specialization_profile_counts(pool)
    )
    enriched.update(_founder_template_pool_identity_diagnostics(pool))
    return enriched


def _best_candidate_founder_template(
    candidate: Mapping[str, object],
) -> dict[str, object] | list[dict[str, object]]:
    pool = candidate.get("founder_template_pool")
    if isinstance(pool, list) and pool:
        templates = [
            dict(metadata)
            for metadata in pool
            if isinstance(metadata, dict)
        ]
        if templates:
            return templates
    metadata = candidate.get("controller_metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    raise ValueError("best candidate is missing controller_metadata")


def _founder_template_pool_from_archive(
    *,
    archive: Mapping[str, object],
    fallback: Mapping[str, object],
    candidates: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []

    def collect_metadata(metadata: object) -> None:
        if isinstance(metadata, dict):
            templates.append(dict(metadata))

    collect_metadata(fallback.get("controller_metadata"))
    raw_elites = fallback.get("controller_lineage_elites")
    if isinstance(raw_elites, list):
        for raw_elite in raw_elites:
            if isinstance(raw_elite, Mapping):
                collect_metadata(raw_elite.get("controller_metadata"))
    for candidate in _candidate_pool_order(candidates or []):
        if not _candidate_is_founder_pool_eligible(candidate):
            continue
        collect_metadata(candidate.get("controller_metadata"))
        candidate_elites = candidate.get("controller_lineage_elites")
        if not isinstance(candidate_elites, list):
            continue
        for raw_elite in candidate_elites:
            if isinstance(raw_elite, Mapping):
                collect_metadata(raw_elite.get("controller_metadata"))
    for parent in _archive_parent_candidates(dict(archive)):
        if not _candidate_is_founder_pool_eligible(parent):
            continue
        collect_metadata(parent.get("controller_metadata"))
        parent_elites = parent.get("controller_lineage_elites")
        if not isinstance(parent_elites, list):
            continue
        for raw_elite in parent_elites:
            if isinstance(raw_elite, Mapping):
                collect_metadata(raw_elite.get("controller_metadata"))
    return _diverse_founder_template_pool(templates)


def _generation_candidates(
    generations: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for generation in generations:
        raw_candidates = generation.get("candidates")
        if not isinstance(raw_candidates, list):
            continue
        candidates.extend(
            candidate
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        )
    return candidates


def _run_fixture_rerank(
    *,
    archive: dict[str, object],
    generations: list[dict[str, object]],
    limit: int,
    holdout_seeds: list[int],
    ticks: int,
    rollout_workers: int,
    fixture_config: dict[str, object],
    fixture_ticks: list[int],
    selector_probe: str | None = None,
    selector_probe_scope: str = MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY,
    selector_probe_trajectory_output_dir: Path | None = None,
    recovery_archive_retention: str | None = None,
) -> dict[str, object]:
    generation_candidates = _generation_candidates(generations)
    candidates = [
        _candidate_with_founder_template_pool(
            candidate,
            archive=archive,
            candidates=generation_candidates,
        )
        for candidate in _fixture_rerank_candidate_pool(
            generation_candidates,
            limit=limit,
        )
    ]
    if not candidates:
        raise SystemExit("fixture rerank did not find candidate nominees")
    evaluated: list[dict[str, object]] = []
    evaluated_runtimes: list[dict[str, object]] = []
    selected_runtime: dict[str, object] | None = None
    selected_key: tuple | None = None

    def consider_runtime(runtime: dict[str, object]) -> None:
        nonlocal selected_key, selected_runtime
        entry = dict(runtime["entry"])
        key = _fixture_rerank_selection_key(entry)
        entry["selection_key"] = list(key)
        runtime["entry"] = entry
        evaluated.append(entry)
        evaluated_runtimes.append(runtime)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_runtime = runtime

    initial_selector_probe = _fixture_rerank_selector_probe_for_phase(
        selector_probe,
        selector_probe_scope=selector_probe_scope,
        phase="initial",
    )
    repair_selector_probe = _fixture_rerank_selector_probe_for_phase(
        selector_probe,
        selector_probe_scope=selector_probe_scope,
        phase="repair",
    )
    bridge_repair_selector_probe = _fixture_rerank_selector_probe_for_phase(
        selector_probe,
        selector_probe_scope=selector_probe_scope,
        phase="bridge_repair",
    )

    initial_runtimes, initial_worker_count = _evaluate_fixture_rerank_candidates(
        [
            (rank, candidate)
            for rank, candidate in enumerate(candidates)
        ],
        holdout_seeds=holdout_seeds,
        ticks=ticks,
        rollout_workers=rollout_workers,
        fixture_config=fixture_config,
        fixture_ticks=fixture_ticks,
        selector_probe=initial_selector_probe,
        selector_probe_trajectory_output_dir=selector_probe_trajectory_output_dir,
    )
    for runtime in initial_runtimes:
        consider_runtime(runtime)

    repair_candidates = _fixture_repair_candidates(evaluated_runtimes)
    repair_runtimes, repair_worker_count = _evaluate_fixture_rerank_candidates(
        repair_candidates,
        holdout_seeds=holdout_seeds,
        ticks=ticks,
        rollout_workers=rollout_workers,
        fixture_config=fixture_config,
        fixture_ticks=fixture_ticks,
        selector_probe=repair_selector_probe,
        selector_probe_trajectory_output_dir=(
            selector_probe_trajectory_output_dir
            if repair_selector_probe is not None
            else None
        ),
    )
    for runtime in repair_runtimes:
        consider_runtime(runtime)

    bridge_repair_candidates = _fixture_promotion_bridge_repair_candidates(
        evaluated_runtimes
    )
    (
        bridge_repair_runtimes,
        bridge_repair_worker_count,
    ) = _evaluate_fixture_rerank_candidates(
        bridge_repair_candidates,
        holdout_seeds=holdout_seeds,
        ticks=ticks,
        rollout_workers=rollout_workers,
        fixture_config=fixture_config,
        fixture_ticks=fixture_ticks,
        selector_probe=bridge_repair_selector_probe,
        selector_probe_trajectory_output_dir=(
            selector_probe_trajectory_output_dir
            if bridge_repair_selector_probe is not None
            else None
        ),
    )
    for runtime in bridge_repair_runtimes:
        consider_runtime(runtime)

    if selected_runtime is None:
        raise SystemExit("fixture rerank did not select a candidate")
    selected_candidate = dict(selected_runtime["candidate"])
    selected_entry = dict(selected_runtime["entry"])
    repair_count = sum(
        1
        for entry in evaluated
        if isinstance(entry.get("fixture_repair"), Mapping)
    )
    selector_probe_summary = _fixture_rerank_selector_probe_summary(
        evaluated,
        selected_candidate_id=str(selected_candidate["candidate_id"]),
        selector_probe=selector_probe,
        selector_probe_scope=selector_probe_scope,
        initial_candidate_count=len(candidates),
        standard_repair_candidate_count=len(repair_candidates),
        bridge_repair_candidate_count=len(bridge_repair_candidates),
    )
    warm_start_candidate_count = sum(
        1
        for candidate in candidates
        if isinstance(candidate.get("warm_start"), Mapping)
    )
    ecology_lane_candidate_count = sum(
        1
        for candidate in candidates
        if "scavenger" in _candidate_ecology_lane_tags(candidate)
    )
    report = {
        "policy": MIND_V3_FIXTURE_RERANK_POLICY,
        "top_k": limit,
        "candidate_count": len(evaluated),
        "initial_candidate_count": len(candidates),
        "warm_start_candidate_count": warm_start_candidate_count,
        "warm_start_candidate_limit": MIND_V3_FIXTURE_WARM_START_CANDIDATE_LIMIT,
        "ecology_lane_candidate_count": ecology_lane_candidate_count,
        "ecology_lane_candidate_limit": MIND_V3_FIXTURE_ECOLOGY_LANE_CANDIDATE_LIMIT,
        "fixture_rerank_ticks": [int(value) for value in fixture_ticks],
        "execution": {
            "policy": "fixture_rerank_candidate_process_pool_v1",
            "requested_workers": rollout_workers,
            "initial_workers": initial_worker_count,
            "repair_workers": repair_worker_count,
            "bridge_repair_workers": bridge_repair_worker_count,
        },
        "repair_policy": MIND_V3_FIXTURE_REPAIR_POLICY,
        "selector_probe_policy": selector_probe,
        "selector_probe_enabled": selector_probe is not None,
        "selector_probe_scope": (
            selector_probe_scope if selector_probe is not None else None
        ),
        "selector_probe_candidate_scope": (
            _fixture_rerank_selector_probe_scope_label(selector_probe_scope)
            if selector_probe is not None
            else None
        ),
        "selector_probe_summary": selector_probe_summary,
        "selector_probe_trajectory_output_dir": (
            str(selector_probe_trajectory_output_dir)
            if selector_probe_trajectory_output_dir is not None
            else None
        ),
        "repair_candidate_count": repair_count,
        "standard_repair_candidate_count": len(repair_candidates),
        "bridge_repair_candidate_count": len(bridge_repair_candidates),
        "selected_candidate_id": str(selected_candidate["candidate_id"]),
        "selected_prefilter_rank": int(selected_entry["prefilter_rank"]),
        "selected_selection_key": list(selected_key or ()),
        "candidates": evaluated,
    }
    retention = _fixture_recovery_archive_retention_from_rerank(
        recovery_archive_retention,
        rerank_report=report,
        evaluated_runtimes=evaluated_runtimes,
    )
    if retention is not None:
        report["fixture_recovery_archive_retention_policy"] = (
            recovery_archive_retention
        )
        report["fixture_recovery_archive_retention_enabled"] = True
        report["fixture_recovery_archive_retention"] = dict(
            retention["report"]
        )
    return {
        "selected_candidate": selected_candidate,
        "holdout_evaluation": selected_runtime["holdout_evaluation"],
        "fixture_suite": selected_runtime["fixture_suite"],
        "fixture_gate": selected_runtime["fixture_gate"],
        "report": report,
        "recovery_archive_retention": retention,
    }


def _fixture_recovery_archive_retention_from_rerank(
    policy: str | None,
    *,
    rerank_report: Mapping[str, object],
    evaluated_runtimes: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if policy is None:
        return None
    if policy != MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY:
        raise ValueError(f"unsupported recovery archive retention policy: {policy}")
    audit_report = build_fixture_rerank_recovery_probe_archive_report(
        search_reports=[
            (
                "fixture_rerank",
                {
                    "schema_version": MIND_V3_EVOLUTION_SEARCH_SCHEMA_VERSION,
                    "fixture_rerank": rerank_report,
                },
                None,
            )
        ]
    )
    input_summary = dict(audit_report.get("input_summary", {}))
    selected_ids = [
        str(candidate_id)
        for candidate_id in input_summary.get("selected_candidate_ids", [])
    ]
    selected_candidate_id = selected_ids[0] if selected_ids else str(
        rerank_report.get("selected_candidate_id", "")
    )
    missing_probe_count = int(input_summary.get("missing_probe_count", 0))
    selected_probe_completed = bool(
        input_summary.get("selected_candidate_probe_completed", False)
    )
    blocked_reasons: list[str] = []
    if int(input_summary.get("complete_probe_count", 0)) <= 0:
        blocked_reasons.append("no_complete_recovery_probe_candidates")
    if missing_probe_count > 0:
        blocked_reasons.append("missing_or_incomplete_recovery_probe_candidates")
    if not selected_probe_completed:
        blocked_reasons.append("selected_candidate_recovery_probe_incomplete")

    source_candidates = _fixture_rerank_runtime_candidates_by_id(evaluated_runtimes)
    source_cells = _fixture_recovery_archive_retention_source_cells(audit_report)
    candidate_ids_that_would_be_added = [
        str(candidate_id)
        for candidate_id in dict(audit_report.get("retention", {})).get(
            "retained_candidate_ids_that_would_be_added",
            [],
        )
    ]
    if not candidate_ids_that_would_be_added:
        blocked_reasons.append("no_non_selected_cell_elites")

    retained_candidates: list[dict[str, object]] = []
    retained_source_cells: list[dict[str, object]] = []
    missing_candidate_ids: list[str] = []
    if not blocked_reasons:
        for candidate_id in candidate_ids_that_would_be_added:
            source = source_candidates.get(candidate_id)
            cells = source_cells.get(candidate_id, [])
            if source is None:
                missing_candidate_ids.append(candidate_id)
                continue
            retained = dict(source)
            retained["fixture_recovery_archive_retention"] = {
                "policy": policy,
                "source_mode": (
                    MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE
                ),
                "selected_candidate_id": selected_candidate_id,
                "source_candidate_id": candidate_id,
                "source_cells": cells,
            }
            retained["archive_parent_source"] = policy
            retained_candidates.append(retained)
            retained_source_cells.extend(cells)
    if missing_candidate_ids:
        blocked_reasons.append("retained_candidate_metadata_missing")
        retained_candidates = []
        retained_source_cells = []

    retained_candidate_ids = [
        str(candidate["candidate_id"])
        for candidate in retained_candidates
        if "candidate_id" in candidate
    ]
    source_type_counts = Counter(
        str(cell.get("source_type", "unknown")) for cell in retained_source_cells
    )
    report = {
        "policy": policy,
        "enabled": True,
        "source_mode": MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
        "changes_runtime_policy": False,
        "changes_fixture_rerank_selection": False,
        "changes_final_selected_candidate": False,
        "bounded_to_fixture_rerank_candidates": True,
        "archive_cell_count": int(
            dict(audit_report.get("archive", {})).get("cell_count", 0)
        ),
        "selected_candidate_id": selected_candidate_id,
        "selected_dominates_non_selected_cells": bool(
            dict(audit_report.get("selected", {})).get(
                "dominates_all_non_selected_recovery_cells",
                False,
            )
        ),
        "input_summary": input_summary,
        "retention_added_count": len(retained_candidate_ids),
        "retained_candidate_ids": retained_candidate_ids,
        "retained_source_cells": retained_source_cells,
        "retention_blocked_reasons": sorted(set(blocked_reasons)),
        "missing_retained_candidate_ids": sorted(missing_candidate_ids),
        "parent_archive_source_diagnostics": {
            "policy": policy,
            "source_candidate_count": len(source_candidates),
            "source_candidate_ids": sorted(source_candidates),
            "retained_candidate_count": len(retained_candidate_ids),
            "retained_source_type_counts": dict(sorted(source_type_counts.items())),
            "retention_pool": (
                "fixture_rerank_initial_repair_bridge_repair_candidates_only"
            ),
        },
        "audit": {
            "archive_policy": audit_report.get("archive_policy"),
            "archive_contract_digest": dict(
                audit_report.get("provenance", {})
            ).get("archive_contract_digest"),
            "retained_candidate_ids_that_would_be_added": (
                candidate_ids_that_would_be_added
            ),
            "non_selected_recovery_better_than_selected": bool(
                dict(audit_report.get("retention", {})).get(
                    "non_selected_recovery_better_than_selected",
                    False,
                )
            ),
        },
    }
    return {
        "policy": policy,
        "report": report,
        "retained_candidates": retained_candidates,
    }


def _fixture_rerank_runtime_candidates_by_id(
    evaluated_runtimes: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}
    for runtime in evaluated_runtimes:
        candidate = runtime.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id:
            continue
        candidates[candidate_id] = dict(candidate)
    return candidates


def _fixture_recovery_archive_retention_source_cells(
    audit_report: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    archive = audit_report.get("archive")
    cells = dict(archive).get("cells", []) if isinstance(archive, Mapping) else []
    source_cells: dict[str, list[dict[str, object]]] = {}
    if not isinstance(cells, list):
        return source_cells
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            continue
        elite = raw_cell.get("elite")
        if not isinstance(elite, Mapping):
            continue
        candidate_id = str(elite.get("candidate_id", ""))
        if not candidate_id:
            continue
        source_cells.setdefault(candidate_id, []).append(
            {
                "cell_key": str(raw_cell.get("cell_key", "")),
                "descriptor": dict(raw_cell.get("descriptor", {}))
                if isinstance(raw_cell.get("descriptor"), Mapping)
                else {},
                "facet_cells": [
                    str(value)
                    for value in elite.get("facet_cells", [])
                    if isinstance(value, str)
                ],
                "source_type": str(elite.get("source_type", "unknown")),
                "origin_source_type": str(
                    elite.get("origin_source_type", "unknown")
                ),
                "post_contact_survival_rate": float(
                    elite.get("post_contact_survival_rate", 0.0)
                ),
                "drink_after_carrion_rate": float(
                    elite.get("drink_after_carrion_rate", 0.0)
                ),
                "mean_hydration_delta_after_carrion": float(
                    elite.get("mean_hydration_delta_after_carrion", 0.0)
                ),
                "mean_water_distance": float(
                    elite.get("mean_water_distance", 0.0)
                ),
                "unsupported_resolved_action_count": int(
                    elite.get("unsupported_resolved_action_count", 0)
                ),
                "dominant_requested_action_share": float(
                    elite.get("dominant_requested_action_share", 0.0)
                ),
            }
        )
    for candidate_cells in source_cells.values():
        candidate_cells.sort(
            key=lambda cell: (
                str(cell.get("cell_key", "")),
                str(cell.get("source_type", "")),
            )
        )
    return source_cells


def _archive_with_recovery_archive_retention(
    archive: Mapping[str, object],
    *,
    retention: Mapping[str, object],
) -> dict[str, object]:
    enriched = dict(archive)
    report = retention.get("report")
    retained_candidates = retention.get("retained_candidates")
    if isinstance(report, Mapping):
        enriched["fixture_recovery_archive_retention"] = dict(report)
    if isinstance(retained_candidates, list):
        enriched["fixture_recovery_archive_retained_candidates"] = [
            dict(candidate)
            for candidate in retained_candidates
            if isinstance(candidate, Mapping)
        ]
    return enriched


def _fixture_rerank_selector_probe_for_phase(
    selector_probe: str | None,
    *,
    selector_probe_scope: str,
    phase: str,
) -> str | None:
    if selector_probe is None:
        return None
    if phase == "initial":
        return selector_probe
    if (
        selector_probe_scope
        == MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_AND_REPAIR
        and phase in {"repair", "bridge_repair"}
    ):
        return selector_probe
    return None


def _fixture_rerank_selector_probe_scope_label(selector_probe_scope: str) -> str:
    if selector_probe_scope == MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_AND_REPAIR:
        return "initial_and_repair_fixture_rerank_candidates"
    return "initial_fixture_rerank_candidate_pool_only"


def _fixture_rerank_selector_probe_summary(
    evaluated: list[dict[str, object]],
    *,
    selected_candidate_id: str,
    selector_probe: str | None,
    selector_probe_scope: str,
    initial_candidate_count: int,
    standard_repair_candidate_count: int,
    bridge_repair_candidate_count: int,
) -> dict[str, object] | None:
    if selector_probe is None:
        return None
    probed_by_group = {
        "initial": 0,
        "repair": 0,
        "bridge_repair": 0,
    }
    total_by_group = {
        "initial": int(initial_candidate_count),
        "repair": int(standard_repair_candidate_count),
        "bridge_repair": int(bridge_repair_candidate_count),
    }
    missing_by_reason: dict[str, int] = {}
    complete_probe_count = 0
    selected_probe_present = False
    selected_probe_completed = False

    for entry in evaluated:
        group = _fixture_rerank_entry_probe_group(entry)
        probe = entry.get("carrion_recovery_probe")
        probe_present = isinstance(probe, Mapping)
        if probe_present:
            probed_by_group[group] = probed_by_group.get(group, 0) + 1
            if _fixture_rerank_recovery_probe_completed(probe):
                complete_probe_count += 1
            else:
                reason = _fixture_rerank_recovery_probe_missing_reason(probe)
                missing_by_reason[reason] = missing_by_reason.get(reason, 0) + 1
        else:
            reason = (
                "not_probed_initial_only_scope"
                if group in {"repair", "bridge_repair"}
                and selector_probe_scope
                == MIND_V3_FIXTURE_SELECTOR_PROBE_SCOPE_INITIAL_ONLY
                else "probe_missing"
            )
            missing_by_reason[reason] = missing_by_reason.get(reason, 0) + 1
        if str(entry.get("candidate_id", "")) == selected_candidate_id:
            selected_probe_present = probe_present
            selected_probe_completed = bool(
                probe_present
                and _fixture_rerank_recovery_probe_completed(probe)
            )

    missing_probe_count = sum(missing_by_reason.values())
    return {
        "policy": selector_probe,
        "scope": selector_probe_scope,
        "report_only": True,
        "changes_candidate_selection": False,
        "initial_candidate_count": total_by_group["initial"],
        "repair_candidate_count": total_by_group["repair"],
        "bridge_repair_candidate_count": total_by_group["bridge_repair"],
        "initial_candidates_probed": probed_by_group["initial"],
        "repair_candidates_probed": probed_by_group["repair"],
        "bridge_repair_candidates_probed": probed_by_group["bridge_repair"],
        "complete_probe_count": complete_probe_count,
        "missing_probe_count": missing_probe_count,
        "missing_probe_count_by_reason": dict(sorted(missing_by_reason.items())),
        "selected_candidate_probe_present": selected_probe_present,
        "selected_candidate_probe_completed": selected_probe_completed,
    }


def _fixture_rerank_entry_probe_group(entry: Mapping[str, object]) -> str:
    repair = entry.get("fixture_repair")
    if isinstance(repair, Mapping):
        if repair.get("donor_selection_reason") == "promotion_safe_bridge":
            return "bridge_repair"
        return "repair"
    return "initial"


def _fixture_rerank_recovery_probe_completed(
    probe: Mapping[str, object],
) -> bool:
    if not bool(probe.get("available", False)):
        return False
    missing_fields = probe.get("missing_fields")
    return isinstance(missing_fields, list) and not missing_fields


def _fixture_rerank_recovery_probe_missing_reason(
    probe: Mapping[str, object],
) -> str:
    if not bool(probe.get("available", False)):
        reason = probe.get("missing_reason")
        return str(reason) if isinstance(reason, str) and reason else "unavailable"
    missing_fields = probe.get("missing_fields")
    if isinstance(missing_fields, list) and missing_fields:
        return "missing_fields"
    return "incomplete_probe"


def _evaluate_fixture_rerank_candidates(
    ranked_candidates: list[tuple[int, dict[str, object]]],
    *,
    holdout_seeds: list[int],
    ticks: int,
    rollout_workers: int,
    fixture_config: dict[str, object],
    fixture_ticks: list[int],
    selector_probe: str | None = None,
    selector_probe_trajectory_output_dir: Path | None = None,
) -> tuple[list[dict[str, object]], int]:
    worker_count = _resolved_rollout_worker_count(
        rollout_workers,
        work_item_count=len(ranked_candidates),
    )
    if not ranked_candidates:
        return [], worker_count
    if worker_count <= 1:
        return (
            [
                _evaluate_fixture_rerank_candidate(
                    candidate=candidate,
                    prefilter_rank=prefilter_rank,
                    holdout_seeds=holdout_seeds,
                    ticks=ticks,
                    rollout_workers=rollout_workers,
                    fixture_config=fixture_config,
                    fixture_ticks=fixture_ticks,
                    selector_probe=selector_probe,
                    selector_probe_trajectory_output_dir=(
                        selector_probe_trajectory_output_dir
                    ),
                )
                for prefilter_rank, candidate in ranked_candidates
            ],
            worker_count,
        )
    tasks = [
        {
            "candidate": candidate,
            "prefilter_rank": prefilter_rank,
            "holdout_seeds": holdout_seeds,
            "ticks": ticks,
            "fixture_config": fixture_config,
            "fixture_ticks": fixture_ticks,
            "selector_probe": selector_probe,
            "selector_probe_trajectory_output_dir": (
                str(selector_probe_trajectory_output_dir)
                if selector_probe_trajectory_output_dir is not None
                else None
            ),
        }
        for prefilter_rank, candidate in ranked_candidates
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_evaluate_fixture_rerank_task, tasks)), worker_count


def _evaluate_fixture_rerank_task(task: dict[str, object]) -> dict[str, object]:
    return _evaluate_fixture_rerank_candidate(
        candidate=dict(task["candidate"]),
        prefilter_rank=int(task["prefilter_rank"]),
        holdout_seeds=[int(seed) for seed in list(task["holdout_seeds"])],
        ticks=int(task["ticks"]),
        rollout_workers=1,
        fixture_config=dict(task["fixture_config"]),
        fixture_ticks=[int(tick) for tick in list(task["fixture_ticks"])],
        selector_probe=(
            str(task["selector_probe"])
            if task.get("selector_probe") is not None
            else None
        ),
        selector_probe_trajectory_output_dir=(
            Path(str(task["selector_probe_trajectory_output_dir"]))
            if task.get("selector_probe_trajectory_output_dir") is not None
            else None
        ),
    )


def _evaluate_fixture_rerank_candidate(
    *,
    candidate: dict[str, object],
    prefilter_rank: int,
    holdout_seeds: list[int],
    ticks: int,
    rollout_workers: int,
    fixture_config: dict[str, object],
    fixture_ticks: list[int],
    selector_probe: str | None = None,
    selector_probe_trajectory_output_dir: Path | None = None,
) -> dict[str, object]:
    founder_template = _best_candidate_founder_template(candidate)
    holdout_evaluation = (
        _evaluate_holdout(
            seeds=holdout_seeds,
            ticks=ticks,
            metadata=founder_template,
            rollout_workers=rollout_workers,
        )
        if holdout_seeds
        else None
    )
    horizon_runtimes: list[dict[str, object]] = []
    for fixture_tick in fixture_ticks:
        horizon_config = _fixture_config_for_ticks(
            fixture_config,
            ticks=fixture_tick,
        )
        horizon_suite = run_mind_v3_fixture_suite(
            suite=str(horizon_config["suite"]),
            fixture_names=_fixture_names_from_config(horizon_config),
            seeds=[int(seed) for seed in list(horizon_config["seeds"])],
            ticks=int(horizon_config.get("ticks") or ticks),
            founder_template=founder_template,
        )
        horizon_gate = mind_v3_fixture_gate_status(
            fixture_suite=horizon_suite,
            fixture_config=horizon_config,
        )
        horizon_runtimes.append(
            {
                "ticks": int(fixture_tick),
                "fixture_suite": horizon_suite,
                "fixture_gate": horizon_gate,
                "fixture_summary": _fixture_rerank_fixture_summary(
                    horizon_suite
                ),
            }
        )
    primary_runtime = horizon_runtimes[0]
    fixture_suite = dict(primary_runtime["fixture_suite"])
    fixture_gate = _combined_fixture_horizon_gate(horizon_runtimes)
    recovery_probe = _fixture_rerank_candidate_recovery_probe(
        selector_probe=selector_probe,
        candidate=candidate,
        prefilter_rank=prefilter_rank,
        founder_template=founder_template,
        fixture_config=fixture_config,
        fixture_ticks=fixture_ticks,
        fallback_fixture_gate=fixture_gate,
        fallback_fixture_summary=_fixture_rerank_fixture_summary(fixture_suite),
        trajectory_output_dir=selector_probe_trajectory_output_dir,
    )
    return {
        "candidate": candidate,
        "holdout_evaluation": holdout_evaluation,
        "fixture_suite": fixture_suite,
        "fixture_gate": fixture_gate,
        "fixture_horizons": horizon_runtimes,
        "entry": _fixture_rerank_entry(
            candidate=candidate,
            prefilter_rank=prefilter_rank,
            holdout_evaluation=holdout_evaluation,
            fixture_suite=fixture_suite,
            fixture_gate=fixture_gate,
            fixture_horizons=horizon_runtimes,
            recovery_probe=recovery_probe,
        ),
    }


def _fixture_rerank_candidate_recovery_probe(
    *,
    selector_probe: str | None,
    candidate: Mapping[str, object],
    prefilter_rank: int,
    founder_template: dict[str, object] | list[dict[str, object]],
    fixture_config: Mapping[str, object],
    fixture_ticks: list[int],
    fallback_fixture_gate: Mapping[str, object],
    fallback_fixture_summary: Mapping[str, object],
    trajectory_output_dir: Path | None,
) -> dict[str, object] | None:
    if selector_probe is None:
        return None
    if selector_probe != MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY:
        raise ValueError(f"unsupported selector probe policy: {selector_probe}")
    configured_fixture_names = _fixture_names_from_config(fixture_config)
    if configured_fixture_names is not None and (
        "carrion_only" not in configured_fixture_names
    ):
        return _fixture_rerank_recovery_probe_unavailable(
            candidate=candidate,
            prefilter_rank=prefilter_rank,
            reason="carrion_only_not_in_fixture_config",
            fixture_gate=fallback_fixture_gate,
            fixture_summary=fallback_fixture_summary,
        )
    if trajectory_output_dir is None:
        return _fixture_rerank_recovery_probe_unavailable(
            candidate=candidate,
            prefilter_rank=prefilter_rank,
            reason="trajectory_output_dir_missing",
            fixture_gate=fallback_fixture_gate,
            fixture_summary=fallback_fixture_summary,
        )
    probe_ticks = max(int(value) for value in (fixture_ticks or [0]))
    if probe_ticks < 1:
        probe_ticks = int(fixture_config.get("ticks") or 1)
    horizon_config = _fixture_config_for_ticks(fixture_config, ticks=probe_ticks)
    horizon_config["fixture_names"] = ["carrion_only"]
    trajectory_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = str(candidate.get("candidate_id", "candidate"))
    prefix = (
        "selector_probe_"
        f"{_selector_probe_path_token(candidate_id)}_rank{int(prefilter_rank)}"
        f"_t{int(probe_ticks)}"
    )
    fixture_suite = run_mind_v3_fixture_suite(
        suite=str(horizon_config["suite"]),
        fixture_names=["carrion_only"],
        seeds=[int(seed) for seed in list(horizon_config["seeds"])],
        ticks=int(horizon_config.get("ticks") or probe_ticks),
        founder_template=founder_template,
        trajectory_output_dir=trajectory_output_dir,
        trajectory_prefix=prefix,
    )
    fixture_gate = mind_v3_fixture_gate_status(
        fixture_suite=fixture_suite,
        fixture_config=horizon_config,
    )
    trajectory_paths = _mind_v3_fixture_trajectory_paths(fixture_suite)
    datasets = [
        load_carrion_autopsy_trajectory_jsonl(path)
        for path in trajectory_paths
    ]
    trace_report = build_carrion_autopsy_report(datasets)
    return _fixture_rerank_recovery_probe_report(
        candidate=candidate,
        prefilter_rank=prefilter_rank,
        fixture_gate=fixture_gate,
        fixture_summary=_fixture_rerank_fixture_summary(fixture_suite),
        trace_report=trace_report,
        trajectory_paths=trajectory_paths,
        ticks=probe_ticks,
    )


def _fixture_rerank_recovery_probe_unavailable(
    *,
    candidate: Mapping[str, object],
    prefilter_rank: int,
    reason: str,
    fixture_gate: Mapping[str, object],
    fixture_summary: Mapping[str, object],
) -> dict[str, object]:
    report = _fixture_rerank_recovery_probe_report(
        candidate=candidate,
        prefilter_rank=prefilter_rank,
        fixture_gate=fixture_gate,
        fixture_summary=fixture_summary,
        trace_report=None,
        trajectory_paths=[],
        ticks=None,
    )
    report["available"] = False
    report["missing_reason"] = reason
    return report


def _fixture_rerank_recovery_probe_report(
    *,
    candidate: Mapping[str, object],
    prefilter_rank: int,
    fixture_gate: Mapping[str, object],
    fixture_summary: Mapping[str, object],
    trace_report: Mapping[str, object] | None,
    trajectory_paths: list[Path],
    ticks: int | None,
) -> dict[str, object]:
    carrion_fixture = _fixture_summary_named_fixture(
        fixture_summary,
        "carrion_only",
    )
    blockers = fixture_gate.get("blockers")
    blocker_list = (
        [item for item in blockers if isinstance(item, Mapping)]
        if isinstance(blockers, list)
        else []
    )
    trace_aggregate = _mapping(
        _mapping(trace_report.get("fixture_trace") if trace_report else None).get(
            "aggregate"
        )
    )
    navigation = _mapping(trace_aggregate.get("navigation_target_observations"))
    water_navigation = _mapping(navigation.get("water"))
    requested_counts = _int_counter(trace_aggregate.get("requested_action_counts"))
    dominant_count = max(requested_counts.values(), default=0)
    total_requested = sum(requested_counts.values())
    report: dict[str, object] = {
        "policy": MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
        "available": trace_report is not None,
        "report_only": True,
        "changes_candidate_selection": False,
        "candidate_id": str(candidate.get("candidate_id", "")),
        "prefilter_rank": int(prefilter_rank),
        "ticks": int(ticks) if ticks is not None else None,
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "fixture_gate_passed": bool(fixture_gate.get("passed", False)),
        "fixture_blocker_count": len(blocker_list),
        "carrion_only_blocker_count": sum(
            1 for blocker in blocker_list if blocker.get("fixture") == "carrion_only"
        ),
        "fixture_blockers": [dict(blocker) for blocker in blocker_list],
        "post_contact_survival_rate": _probe_optional_number(
            trace_aggregate.get("survival_after_carrion_rate")
        ),
        "drink_after_carrion_rate": _probe_optional_number(
            trace_aggregate.get("drink_after_carrion_rate")
        ),
        "mean_hydration_delta_after_carrion": _probe_optional_number(
            trace_aggregate.get("mean_hydration_delta_after_carrion")
        ),
        "mean_energy_delta_after_carrion": _probe_optional_number(
            trace_aggregate.get("mean_energy_delta_after_carrion")
        ),
        "mean_health_delta_after_carrion": _probe_optional_number(
            trace_aggregate.get("mean_health_delta_after_carrion")
        ),
        "mean_water_distance": _probe_optional_number(
            water_navigation.get("mean_distance")
        ),
        "unsupported_requested_action_count": _probe_optional_number(
            trace_aggregate.get("unsupported_requested_action_count")
        ),
        "unsupported_resolved_action_count": _probe_optional_number(
            trace_aggregate.get("unsupported_resolved_action_count")
        ),
        "dominant_requested_action_share": _round(
            dominant_count / max(1, total_requested)
        )
        if total_requested > 0
        else None,
        "carrion_only_alive_agents_mean": _probe_optional_number(
            carrion_fixture.get("alive_agents_mean")
        ),
        "carrion_only_births_mean": _probe_optional_number(
            carrion_fixture.get("births_mean")
        ),
        "carrion_only_terminal_hydration_viability_share_mean": (
            _probe_optional_number(
                carrion_fixture.get("terminal_hydration_viability_share_mean")
            )
        ),
        "carrion_only_terminal_energy_viability_share_mean": (
            _probe_optional_number(
                carrion_fixture.get("terminal_energy_viability_share_mean")
            )
        ),
        "carrion_only_terminal_health_viability_share_mean": _probe_optional_number(
            carrion_fixture.get("terminal_health_viability_share_mean")
        ),
        "carrion_only_terminal_matched_diet_viability_share_mean": (
            _probe_optional_number(
                carrion_fixture.get("terminal_matched_diet_viability_share_mean")
            )
        ),
    }
    missing_fields = [
        field
        for field in (
            "post_contact_survival_rate",
            "drink_after_carrion_rate",
            "mean_hydration_delta_after_carrion",
            "mean_energy_delta_after_carrion",
            "mean_health_delta_after_carrion",
            "mean_water_distance",
            "unsupported_requested_action_count",
            "unsupported_resolved_action_count",
            "dominant_requested_action_share",
        )
        if report.get(field) is None
    ]
    report["missing_fields"] = missing_fields
    return report


def _mind_v3_fixture_trajectory_paths(fixture_suite: Mapping[str, object]) -> list[Path]:
    paths: list[Path] = []
    for fixture in _list_of_mappings(fixture_suite.get("fixtures")):
        comparison = _mapping(fixture.get("comparison"))
        mind_v3 = _mapping(comparison.get("mind_v3"))
        for run in _list_of_mappings(mind_v3.get("runs")):
            path = run.get("trajectory_path")
            if isinstance(path, str) and path:
                paths.append(Path(path))
    return paths


def _selector_probe_path_token(value: str) -> str:
    chars = [
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in value
    ]
    token = "".join(chars).strip("-_")
    return token or "candidate"


def _probe_optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _round(float(value))
    return None


def _int_counter(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        result[str(key)] = int(raw)
    return result


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _fixture_repair_candidates(
    evaluated_runtimes: list[dict[str, object]],
) -> list[tuple[int, dict[str, object]]]:
    donors = [
        runtime
        for runtime in evaluated_runtimes
        if _fixture_runtime_gate_passed(runtime)
    ]
    failures = [
        runtime
        for runtime in evaluated_runtimes
        if not _fixture_runtime_gate_passed(runtime)
    ]
    if not failures:
        return []
    donor_reason = "fixture_gate_pass"
    if donors:
        donor = max(donors, key=_fixture_repair_donor_key)
    else:
        lane_donors = [
            runtime
            for runtime in evaluated_runtimes
            if _fixture_runtime_scavenger_lane_signal(runtime) > 0.0
        ]
        if not lane_donors:
            return []
        donor = max(lane_donors, key=_fixture_repair_scavenger_donor_key)
        donor_reason = "scavenger_lane"
    repairs: list[tuple[int, dict[str, object]]] = []
    seen_ids: set[str] = set()
    for primary in sorted(failures, key=_fixture_repair_primary_key, reverse=True):
        if len(repairs) >= MIND_V3_FIXTURE_REPAIR_LIMIT:
            break
        primary_id = str(dict(primary["candidate"]).get("candidate_id", ""))
        donor_id = str(dict(donor["candidate"]).get("candidate_id", ""))
        if primary_id == donor_id:
            continue
        repaired_candidate = _fixture_repaired_candidate(
            dict(primary["candidate"]),
            donor_candidate=dict(donor["candidate"]),
            donor_selection_reason=donor_reason,
        )
        candidate_id = str(repaired_candidate["candidate_id"])
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        repairs.append(
            (
                int(dict(primary.get("entry", {})).get("prefilter_rank", 0)),
                repaired_candidate,
            )
        )
    return repairs


def _fixture_promotion_bridge_repair_candidates(
    evaluated_runtimes: list[dict[str, object]],
) -> list[tuple[int, dict[str, object]]]:
    safe_primaries = [
        runtime
        for runtime in evaluated_runtimes
        if _fixture_runtime_first_horizon_passed(runtime)
        and not _fixture_runtime_gate_passed(runtime)
    ]
    unsafe_donors = [
        runtime
        for runtime in evaluated_runtimes
        if not _fixture_runtime_first_horizon_passed(runtime)
        and _fixture_runtime_long_horizon_signal(runtime) > 0.0
    ]
    if not safe_primaries or not unsafe_donors:
        return []
    existing_ids = {
        str(dict(runtime.get("candidate", {})).get("candidate_id", ""))
        for runtime in evaluated_runtimes
    }
    repairs: list[tuple[int, dict[str, object]]] = []
    seen_ids: set[str] = set()
    ranked_donors = sorted(
        unsafe_donors,
        key=_fixture_bridge_donor_key,
        reverse=True,
    )
    for primary in sorted(
        safe_primaries,
        key=_fixture_bridge_primary_key,
        reverse=True,
    ):
        if len(repairs) >= MIND_V3_FIXTURE_BRIDGE_REPAIR_LIMIT:
            break
        primary_id = str(dict(primary["candidate"]).get("candidate_id", ""))
        for donor in ranked_donors:
            donor_id = str(dict(donor["candidate"]).get("candidate_id", ""))
            if not primary_id or primary_id == donor_id:
                continue
            for donor_template_limit in MIND_V3_FIXTURE_BRIDGE_DONOR_TEMPLATE_LIMITS:
                if len(repairs) >= MIND_V3_FIXTURE_BRIDGE_REPAIR_LIMIT:
                    break
                repaired_candidate = _fixture_repaired_candidate(
                    dict(primary["candidate"]),
                    donor_candidate=dict(donor["candidate"]),
                    donor_selection_reason="promotion_safe_bridge",
                    donor_template_limit=donor_template_limit,
                    candidate_id_suffix=f"donor{donor_template_limit}",
                )
                candidate_id = str(repaired_candidate["candidate_id"])
                if candidate_id in existing_ids or candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                repairs.append(
                    (
                        int(dict(primary.get("entry", {})).get("prefilter_rank", 0)),
                        repaired_candidate,
                    )
                )
            break
    return repairs


def _fixture_runtime_gate_passed(runtime: Mapping[str, object]) -> bool:
    gate = runtime.get("fixture_gate")
    return isinstance(gate, Mapping) and bool(gate.get("passed", False))


def _fixture_runtime_first_horizon_passed(runtime: Mapping[str, object]) -> bool:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    return _fixture_first_horizon_passed(payload)


def _fixture_runtime_long_horizon_signal(runtime: Mapping[str, object]) -> float:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    summary = payload.get("fixture_horizon_summary")
    horizon = summary if isinstance(summary, Mapping) else {}
    return max(
        float(
            horizon.get(
                "carrion_only_terminal_energy_requirement_satisfaction_min",
                0.0,
            )
        ),
        float(horizon.get("carrion_only_alive_agents_min", 0.0)),
        float(horizon.get("carrion_only_births_min", 0.0)),
    )


def _fixture_repair_donor_key(runtime: Mapping[str, object]) -> tuple:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    return _fixture_rerank_selection_key(payload)


def _fixture_repair_scavenger_donor_key(runtime: Mapping[str, object]) -> tuple:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    fixture_summary = payload.get("fixture_summary")
    fixture_payload = fixture_summary if isinstance(fixture_summary, Mapping) else {}
    carrion = _fixture_summary_named_fixture(fixture_payload, "carrion_only")
    return (
        _fixture_runtime_scavenger_lane_signal(runtime),
        float(
            carrion.get(
                "terminal_energy_requirement_satisfaction_mean",
                0.0,
            )
        ),
        float(carrion.get("animal_resource_consumption_events_mean", 0.0)),
        float(carrion.get("animal_resource_gained_energy_mean", 0.0)),
        float(carrion.get("alive_agents_mean", 0.0)),
        float(carrion.get("terminal_energy_hydration_balance_mean", 0.0)),
        float(carrion.get("terminal_reproduction_viability_min", 0.0)),
        float(carrion.get("births_mean", 0.0)),
        float(fixture_payload.get("mixed_stable_births_mean", 0.0)),
        float(payload.get("search_score", 0.0)),
        -int(payload.get("prefilter_rank", 0)),
    )


def _fixture_bridge_primary_key(runtime: Mapping[str, object]) -> tuple:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    return _fixture_rerank_selection_key(payload)


def _fixture_bridge_donor_key(runtime: Mapping[str, object]) -> tuple:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    fixture_gate = payload.get("fixture_gate")
    gate_payload = fixture_gate if isinstance(fixture_gate, Mapping) else {}
    blockers = gate_payload.get("blockers", [])
    blocker_count = len(blockers) if isinstance(blockers, list) else 0
    summary = payload.get("fixture_horizon_summary")
    horizon = summary if isinstance(summary, Mapping) else {}
    holdout = payload.get("holdout_aggregate")
    holdout_payload = holdout if isinstance(holdout, Mapping) else {}
    return (
        -blocker_count,
        float(
            horizon.get(
                "carrion_only_terminal_energy_requirement_satisfaction_min",
                0.0,
            )
        ),
        float(
            horizon.get(
                "carrion_only_terminal_energy_viability_share_min",
                0.0,
            )
        ),
        float(
            horizon.get(
                "carrion_only_terminal_hydration_viability_share_min",
                0.0,
            )
        ),
        float(
            horizon.get(
                "carrion_only_terminal_matched_diet_viability_share_min",
                0.0,
            )
        ),
        float(horizon.get("carrion_only_alive_agents_min", 0.0)),
        float(horizon.get("carrion_only_births_min", 0.0)),
        float(
            horizon.get(
                "carrion_only_animal_resource_consumption_events_min",
                0.0,
            )
        ),
        float(horizon.get("carrion_only_animal_resource_gained_energy_min", 0.0)),
        float(holdout_payload.get("births_mean", 0.0)),
        float(holdout_payload.get("alive_agents_mean", 0.0)),
        float(payload.get("search_score", 0.0)),
        -int(payload.get("prefilter_rank", 0)),
    )


def _fixture_runtime_scavenger_lane_signal(runtime: Mapping[str, object]) -> float:
    candidate = runtime.get("candidate")
    if isinstance(candidate, Mapping):
        return _candidate_scavenger_lane_signal(candidate)
    return 0.0


def _fixture_repair_primary_key(runtime: Mapping[str, object]) -> tuple:
    entry = runtime.get("entry")
    payload = entry if isinstance(entry, Mapping) else {}
    holdout = payload.get("holdout_aggregate")
    holdout_payload = holdout if isinstance(holdout, Mapping) else {}
    fixture_summary = payload.get("fixture_summary")
    fixture_payload = fixture_summary if isinstance(fixture_summary, Mapping) else {}
    return (
        float(holdout_payload.get("births_mean", 0.0)),
        float(holdout_payload.get("alive_agents_mean", 0.0)),
        float(
            holdout_payload.get(
                "terminal_energy_requirement_satisfaction_mean",
                0.0,
            )
        ),
        float(holdout_payload.get("core_readiness_agent_tick_mean", 0.0)),
        float(holdout_payload.get("core_ready_pair_tick_share_mean", 0.0)),
        float(fixture_payload.get("mixed_stable_births_mean", 0.0)),
        float(fixture_payload.get("births_mean", 0.0)),
        float(payload.get("search_score", 0.0)),
        -int(payload.get("prefilter_rank", 0)),
    )


def _fixture_repaired_candidate(
    primary_candidate: dict[str, object],
    *,
    donor_candidate: dict[str, object],
    donor_selection_reason: str = "fixture_gate_pass",
    donor_template_limit: int | None = None,
    candidate_id_suffix: str | None = None,
) -> dict[str, object]:
    primary_id = str(primary_candidate.get("candidate_id", "primary"))
    donor_id = str(donor_candidate.get("candidate_id", "donor"))
    primary_templates = _template_list(
        _best_candidate_founder_template(primary_candidate)
    )
    available_donor_templates = _template_list(
        _best_candidate_founder_template(donor_candidate)
    )
    donor_templates = (
        available_donor_templates[: max(0, int(donor_template_limit))]
        if donor_template_limit is not None
        else available_donor_templates
    )
    pool = _combined_founder_template_pool(
        [*primary_templates, *donor_templates],
        limit=MIND_V3_COMPOSITE_FOUNDER_TEMPLATE_POOL_LIMIT,
    )
    repaired = dict(primary_candidate)
    suffix = f"+{candidate_id_suffix}" if candidate_id_suffix else ""
    repaired["candidate_id"] = f"{primary_id}+repair-{donor_id}{suffix}"
    repaired["parent_candidate_id"] = primary_id
    repaired["founder_template_pool_policy"] = MIND_V3_FIXTURE_REPAIR_POLICY
    repaired["founder_template_pool"] = pool
    repaired["founder_template_pool_size"] = len(pool)
    repaired["founder_template_pool_specialization_profile_counts"] = (
        _metadata_specialization_profile_counts(pool)
    )
    repaired["fixture_repair"] = {
        "policy": MIND_V3_FIXTURE_REPAIR_POLICY,
        "primary_candidate_id": primary_id,
        "donor_candidate_id": donor_id,
        "donor_selection_reason": donor_selection_reason,
        "pool_limit": MIND_V3_COMPOSITE_FOUNDER_TEMPLATE_POOL_LIMIT,
        "primary_template_count": len(primary_templates),
        "donor_template_count": len(donor_templates),
        "donor_template_available_count": len(available_donor_templates),
        "donor_template_limit": (
            int(donor_template_limit)
            if donor_template_limit is not None
            else None
        ),
        "founder_template_pool_size": len(pool),
    }
    return repaired


def _template_list(
    templates: dict[str, object] | list[dict[str, object]],
) -> list[dict[str, object]]:
    if isinstance(templates, dict):
        return [dict(templates)]
    return [dict(template) for template in templates if isinstance(template, dict)]


def _combined_founder_template_pool(
    templates: list[dict[str, object]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for template in templates:
        fingerprint = _metadata_fingerprint(template)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(dict(template))
        if len(selected) >= limit:
            break
    return selected


def _fixture_rerank_candidate_pool(
    candidates: list[dict[str, object]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()

    def append_candidate(candidate: dict[str, object]) -> bool:
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            return False
        seen.add(candidate_id)
        selected.append(candidate)
        return True

    for candidate in sorted(candidates, key=_fixture_rerank_prefilter_key, reverse=True):
        append_candidate(candidate)
        if len(selected) >= limit:
            break

    warm_start_count = 0
    warm_start_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate.get("warm_start"), Mapping)
    ]
    for candidate in sorted(
        warm_start_candidates,
        key=_fixture_rerank_prefilter_key,
        reverse=True,
    ):
        if append_candidate(candidate):
            warm_start_count += 1
        if warm_start_count >= MIND_V3_FIXTURE_WARM_START_CANDIDATE_LIMIT:
            break

    lane_count = 0
    for candidate in sorted(candidates, key=_scavenger_lane_key, reverse=True):
        if _candidate_scavenger_lane_signal(candidate) <= 0.0:
            break
        if append_candidate(candidate):
            lane_count += 1
        if lane_count >= MIND_V3_FIXTURE_ECOLOGY_LANE_CANDIDATE_LIMIT:
            break
    return selected


def _fixture_rerank_prefilter_key(candidate: Mapping[str, object]) -> tuple:
    return (
        float(candidate.get("score", 0.0)),
        float(candidate.get("terminal_reproduction_viability_min", 0.0)),
        float(
            candidate.get("terminal_energy_requirement_satisfaction_mean", 0.0)
        ),
        float(candidate.get("core_readiness_agent_tick_mean", 0.0)),
        float(candidate.get("core_ready_agent_tick_share_mean", 0.0)),
        float(candidate.get("core_ready_pair_tick_share_mean", 0.0)),
        float(candidate.get("reproduction_ready_agent_tick_share_mean", 0.0)),
        float(candidate.get("reproduction_ready_pair_tick_share_mean", 0.0)),
        float(candidate.get("terminal_ready_group_count_mean", 0.0)),
        float(candidate.get("births_mean", 0.0)),
        float(candidate.get("alive_agents_mean", 0.0)),
        int(candidate.get("active_behavior_niche_count", 0)),
        -int(candidate.get("candidate_index", 0)),
    )


def _fixture_current_architecture_key(candidate: Mapping[str, object]) -> tuple:
    return (
        _candidate_controller_architecture(candidate)
        == MIND_V3_CONTROLLER_ARCHITECTURE,
        not isinstance(candidate.get("warm_start"), Mapping),
        float(candidate.get("terminal_energy_hydration_balance_mean", 0.0)),
        float(
            candidate.get(
                "terminal_balanced_reproduction_readiness_mean",
                0.0,
            )
        ),
        float(candidate.get("terminal_matched_diet_viability_share_mean", 0.0)),
        float(candidate.get("score", 0.0)),
        -int(candidate.get("candidate_index", 0)),
    )


def _fixture_controlled_readiness_key(candidate: Mapping[str, object]) -> tuple:
    return (
        float(candidate.get("terminal_energy_hydration_balance_mean", 0.0)),
        float(
            candidate.get(
                "terminal_energy_requirement_satisfaction_mean",
                0.0,
            )
        ),
        float(candidate.get("terminal_energy_viability_share_mean", 0.0)),
        float(candidate.get("terminal_hydration_viability_share_mean", 0.0)),
        float(candidate.get("terminal_matched_diet_viability_share_mean", 0.0)),
        float(candidate.get("movement_event_rate", 0.0)),
        -float(candidate.get("dominant_requested_action_share", 1.0)),
        float(candidate.get("score", 0.0)),
        -int(candidate.get("candidate_index", 0)),
    )


def _fixture_rerank_entry(
    *,
    candidate: Mapping[str, object],
    prefilter_rank: int,
    holdout_evaluation: dict[str, object] | None,
    fixture_suite: dict[str, object],
    fixture_gate: dict[str, object],
    fixture_horizons: list[dict[str, object]] | None = None,
    recovery_probe: dict[str, object] | None = None,
) -> dict[str, object]:
    holdout_aggregate = (
        dict(holdout_evaluation["aggregate"])
        if isinstance(holdout_evaluation, dict)
        and isinstance(holdout_evaluation.get("aggregate"), dict)
        else None
    )
    entry = {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "prefilter_rank": prefilter_rank,
        "search_score": float(candidate.get("score", 0.0)),
        "search_alive_agents_mean": float(candidate.get("alive_agents_mean", 0.0)),
        "search_births_mean": float(candidate.get("births_mean", 0.0)),
        "search_reproduction_ready_agent_tick_share_mean": float(
            candidate.get("reproduction_ready_agent_tick_share_mean", 0.0)
        ),
        "search_reproduction_ready_pair_tick_share_mean": float(
            candidate.get("reproduction_ready_pair_tick_share_mean", 0.0)
        ),
        "search_terminal_energy_requirement_satisfaction_mean": float(
            candidate.get("terminal_energy_requirement_satisfaction_mean", 0.0)
        ),
        "search_terminal_balanced_reproduction_readiness_mean": float(
            candidate.get("terminal_balanced_reproduction_readiness_mean", 0.0)
        ),
        "search_terminal_energy_hydration_balance_mean": float(
            candidate.get("terminal_energy_hydration_balance_mean", 0.0)
        ),
        "search_terminal_energy_hydration_gap_abs_mean": float(
            candidate.get("terminal_energy_hydration_gap_abs_mean", 0.0)
        ),
        "search_scavenger_lane_score": _round(
            _candidate_scavenger_lane_signal(candidate)
        ),
        "search_ecology_lane_tags": _candidate_ecology_lane_tags(candidate),
        "controller_architecture": _candidate_controller_architecture(candidate),
        "founder_template_pool_size": int(
            candidate.get("founder_template_pool_size", 0)
        ),
        "founder_template_pool_specialization_profile_counts": dict(
            candidate.get("founder_template_pool_specialization_profile_counts", {})
        )
        if isinstance(
            candidate.get("founder_template_pool_specialization_profile_counts"),
            Mapping,
        )
        else {},
        "founder_template_pool_fingerprints": [
            str(value)
            for value in list(candidate.get("founder_template_pool_fingerprints", []))
        ]
        if isinstance(candidate.get("founder_template_pool_fingerprints"), list)
        else [],
        "founder_template_pool_distinct_fingerprint_count": int(
            candidate.get("founder_template_pool_distinct_fingerprint_count", 0)
        ),
        "search_terminal_energy_shortfall_share_mean": float(
            candidate.get("terminal_energy_shortfall_share_mean", 0.0)
        ),
        "search_terminal_ready_group_count_mean": float(
            candidate.get("terminal_ready_group_count_mean", 0.0)
        ),
        "search_core_readiness_agent_tick_mean": float(
            candidate.get("core_readiness_agent_tick_mean", 0.0)
        ),
        "search_core_ready_agent_tick_share_mean": float(
            candidate.get("core_ready_agent_tick_share_mean", 0.0)
        ),
        "search_core_ready_pair_tick_share_mean": float(
            candidate.get("core_ready_pair_tick_share_mean", 0.0)
        ),
        "search_dominant_requested_action": candidate.get(
            "dominant_requested_action"
        ),
        "search_dominant_requested_action_share": float(
            candidate.get("dominant_requested_action_share", 0.0)
        ),
        "holdout_aggregate": holdout_aggregate,
        "fixture_summary": _fixture_rerank_fixture_summary(fixture_suite),
        "fixture_gate": fixture_gate,
        "fixture_blocker_pressure": _fixture_blocker_pressure_summary(
            fixture_gate
        ),
    }
    if fixture_horizons is not None:
        entry["fixture_horizons"] = [
            _fixture_horizon_report_entry(horizon)
            for horizon in fixture_horizons
        ]
        entry["fixture_horizon_summary"] = _fixture_horizon_summary(
            fixture_horizons
        )
    if recovery_probe is not None:
        entry["gate_aligned_carrion_recovery_probe_v1"] = recovery_probe
        entry["carrion_recovery_probe"] = recovery_probe
    repair = candidate.get("fixture_repair")
    if isinstance(repair, Mapping):
        entry["fixture_repair"] = dict(repair)
    return entry


def _fixture_config_for_ticks(
    fixture_config: Mapping[str, object],
    *,
    ticks: int,
) -> dict[str, object]:
    config = dict(fixture_config)
    config["ticks"] = int(ticks)
    return config


def _combined_fixture_horizon_gate(
    fixture_horizons: list[dict[str, object]],
) -> dict[str, object]:
    if not fixture_horizons:
        return {
            "policy": MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
            "fixture_suite_policy": MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
            "passed": False,
            "blockers": [
                {
                    "fixture": None,
                    "reason": "fixture_horizon_suite_empty",
                    "metric": "fixture_horizon_count",
                    "value": 0.0,
                    "floor": 1.0,
                }
            ],
        }
    if len(fixture_horizons) == 1:
        return dict(fixture_horizons[0]["fixture_gate"])
    first_gate = dict(fixture_horizons[0]["fixture_gate"])
    blockers: list[dict[str, object]] = []
    per_horizon: dict[str, object] = {}
    horizon_ticks: list[int] = []
    for horizon in fixture_horizons:
        ticks = int(horizon["ticks"])
        horizon_ticks.append(ticks)
        gate = dict(horizon["fixture_gate"])
        horizon_blockers = [
            {**dict(blocker), "ticks": ticks}
            for blocker in gate.get("blockers", [])
            if isinstance(blocker, Mapping)
        ]
        blockers.extend(horizon_blockers)
        per_horizon[str(ticks)] = {
            "passed": bool(gate.get("passed", False)),
            "blockers": horizon_blockers,
            "per_fixture": gate.get("per_fixture", {}),
        }
    combined = dict(first_gate)
    combined["ticks"] = list(horizon_ticks)
    combined["primary_ticks"] = horizon_ticks[0]
    combined["horizon_ticks"] = list(horizon_ticks)
    combined["passed"] = not blockers
    combined["blockers"] = blockers
    combined["per_horizon"] = dict(sorted(per_horizon.items()))
    return combined


def _fixture_horizon_report_entry(
    horizon: Mapping[str, object],
) -> dict[str, object]:
    return {
        "ticks": int(horizon["ticks"]),
        "fixture_gate": dict(horizon["fixture_gate"]),
        "fixture_summary": dict(horizon["fixture_summary"]),
    }


def _fixture_horizon_summary(
    fixture_horizons: list[dict[str, object]],
) -> dict[str, object]:
    entries = [_fixture_horizon_report_entry(horizon) for horizon in fixture_horizons]
    if not entries:
        return {
            "horizon_count": 0,
            "ticks": [],
            "all_horizons_passed": False,
            "passed_horizon_count": 0,
        }
    summaries = [
        dict(entry["fixture_summary"])
        for entry in entries
        if isinstance(entry.get("fixture_summary"), Mapping)
    ]
    carrion_summaries = [
        _fixture_summary_named_fixture(summary, "carrion_only")
        for summary in summaries
    ]
    carrion_summaries = [
        summary for summary in carrion_summaries if isinstance(summary, Mapping)
    ]

    def values(key: str, payloads: list[Mapping[str, object]]) -> list[float]:
        return [float(payload.get(key, 0.0)) for payload in payloads]

    gates = [
        dict(entry["fixture_gate"])
        for entry in entries
        if isinstance(entry.get("fixture_gate"), Mapping)
    ]
    return {
        "horizon_count": len(entries),
        "ticks": [int(entry["ticks"]) for entry in entries],
        "all_horizons_passed": all(
            bool(gate.get("passed", False)) for gate in gates
        ),
        "passed_horizon_count": sum(
            1 for gate in gates if bool(gate.get("passed", False))
        ),
        "blocker_count": sum(
            len(gate.get("blockers", []))
            for gate in gates
            if isinstance(gate.get("blockers", []), list)
        ),
        "alive_agents_min": _round(min(values("alive_agents_mean", summaries)))
        if summaries
        else 0.0,
        "births_min": _round(min(values("births_mean", summaries)))
        if summaries
        else 0.0,
        "mixed_stable_births_min": _round(
            min(values("mixed_stable_births_mean", summaries))
        )
        if summaries
        else 0.0,
        "terminal_reproduction_viability_min": _round(
            min(values("terminal_reproduction_viability_min", summaries))
        )
        if summaries
        else 0.0,
        "terminal_energy_hydration_balance_min": _round(
            min(values("terminal_energy_hydration_balance_min", summaries))
        )
        if summaries
        else 0.0,
        "carrion_only_alive_agents_min": _round(
            min(values("alive_agents_mean", carrion_summaries))
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_births_min": _round(
            min(values("births_mean", carrion_summaries))
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_alive_agent_ticks_per_tick_min": _round(
            min(values("alive_agent_ticks_per_tick_mean", carrion_summaries))
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_terminal_energy_requirement_satisfaction_min": _round(
            min(
                values(
                    "terminal_energy_requirement_satisfaction_mean",
                    carrion_summaries,
                )
            )
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_terminal_energy_viability_share_min": _round(
            min(values("terminal_energy_viability_share_mean", carrion_summaries))
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_terminal_hydration_viability_share_min": _round(
            min(
                values(
                    "terminal_hydration_viability_share_mean",
                    carrion_summaries,
                )
            )
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_terminal_matched_diet_viability_share_min": _round(
            min(
                values(
                    "terminal_matched_diet_viability_share_mean",
                    carrion_summaries,
                )
            )
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_animal_resource_consumption_events_min": _round(
            min(
                values(
                    "animal_resource_consumption_events_mean",
                    carrion_summaries,
                )
            )
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_animal_resource_gained_energy_min": _round(
            min(values("animal_resource_gained_energy_mean", carrion_summaries))
        )
        if carrion_summaries
        else 0.0,
        "carrion_only_dominant_requested_action_share_max": _round(
            max(values("dominant_requested_action_share", carrion_summaries))
        )
        if carrion_summaries
        else 1.0,
    }


def _fixture_rerank_selection_key(entry: Mapping[str, object]) -> tuple:
    fixture_gate = entry.get("fixture_gate")
    gate_payload = fixture_gate if isinstance(fixture_gate, Mapping) else {}
    blockers = gate_payload.get("blockers", [])
    blocker_count = len(blockers) if isinstance(blockers, list) else 0
    fixture_summary = entry.get("fixture_summary")
    fixture_payload = fixture_summary if isinstance(fixture_summary, Mapping) else {}
    horizon_summary = entry.get("fixture_horizon_summary")
    horizon_payload = horizon_summary if isinstance(horizon_summary, Mapping) else {}
    carrion_fixture = _fixture_summary_named_fixture(
        fixture_payload,
        "carrion_only",
    )
    carrion_blocker_pressure = (
        1.0 if _fixture_gate_has_fixture_blocker(gate_payload, "carrion_only") else 0.0
    )
    holdout = entry.get("holdout_aggregate")
    holdout_payload = holdout if isinstance(holdout, Mapping) else {}
    holdout_viability = min(
        float(holdout_payload.get("terminal_energy_viability_share_mean", 0.0)),
        float(
            holdout_payload.get(
                "terminal_hydration_viability_share_mean",
                0.0,
            )
        ),
        float(holdout_payload.get("terminal_health_viability_share_mean", 0.0)),
        float(
            holdout_payload.get(
                "terminal_matched_diet_viability_share_mean",
                0.0,
            )
        ),
    )
    holdout_balanced_readiness = float(
        holdout_payload.get(
            "terminal_balanced_reproduction_readiness_mean",
            _balanced_readiness_score(
                [
                    float(
                        holdout_payload.get(
                            "terminal_energy_requirement_satisfaction_mean",
                            0.0,
                        )
                    ),
                    float(
                        holdout_payload.get(
                            "terminal_hydration_viability_share_mean",
                            0.0,
                        )
                    ),
                    float(
                        holdout_payload.get(
                            "terminal_health_viability_share_mean",
                            0.0,
                        )
                    ),
                    float(
                        holdout_payload.get(
                            "terminal_matched_diet_viability_share_mean",
                            0.0,
                        )
                    ),
                ]
            ),
        )
    )
    holdout_energy_hydration_balance = float(
        holdout_payload.get(
            "terminal_energy_hydration_balance_mean",
            min(
                float(
                    holdout_payload.get(
                        "terminal_energy_requirement_satisfaction_mean",
                        0.0,
                    )
                ),
                float(
                    holdout_payload.get(
                        "terminal_hydration_viability_share_mean",
                        0.0,
                    )
                ),
            ),
        )
    )
    carrion_energy_requirement_satisfaction = float(
        horizon_payload.get(
            "carrion_only_terminal_energy_requirement_satisfaction_min",
            carrion_fixture.get(
                "terminal_energy_requirement_satisfaction_mean",
                0.0,
            ),
        )
    )
    carrion_energy_viability = float(
        horizon_payload.get(
            "carrion_only_terminal_energy_viability_share_min",
            carrion_fixture.get("terminal_energy_viability_share_mean", 0.0),
        )
    )
    carrion_hydration_viability = float(
        horizon_payload.get(
            "carrion_only_terminal_hydration_viability_share_min",
            carrion_fixture.get("terminal_hydration_viability_share_mean", 0.0),
        )
    )
    carrion_matched_diet_viability = float(
        horizon_payload.get(
            "carrion_only_terminal_matched_diet_viability_share_min",
            carrion_fixture.get(
                "terminal_matched_diet_viability_share_mean",
                0.0,
            ),
        )
    )
    carrion_resource_events = float(
        horizon_payload.get(
            "carrion_only_animal_resource_consumption_events_min",
            carrion_fixture.get(
                "animal_resource_consumption_events_mean",
                0.0,
            ),
        )
    )
    carrion_resource_gain = float(
        horizon_payload.get(
            "carrion_only_animal_resource_gained_energy_min",
            carrion_fixture.get("animal_resource_gained_energy_mean", 0.0),
        )
    )
    carrion_dominant_action_share = float(
        horizon_payload.get(
            "carrion_only_dominant_requested_action_share_max",
            carrion_fixture.get("dominant_requested_action_share", 1.0),
        )
    )
    carrion_alive = float(
        horizon_payload.get(
            "carrion_only_alive_agents_min",
            carrion_fixture.get("alive_agents_mean", 0.0),
        )
    )
    carrion_alive_ticks = float(
        horizon_payload.get(
            "carrion_only_alive_agent_ticks_per_tick_min",
            carrion_fixture.get("alive_agent_ticks_per_tick_mean", 0.0),
        )
    )
    carrion_births = float(
        horizon_payload.get(
            "carrion_only_births_min",
            carrion_fixture.get("births_mean", 0.0),
        )
    )
    fixture_reproduction_viability = float(
        horizon_payload.get(
            "terminal_reproduction_viability_min",
            fixture_payload.get("terminal_reproduction_viability_min", 0.0),
        )
    )
    fixture_energy_hydration_balance = float(
        horizon_payload.get(
            "terminal_energy_hydration_balance_min",
            fixture_payload.get("terminal_energy_hydration_balance_min", 0.0),
        )
    )
    fixture_mixed_stable_births = float(
        horizon_payload.get(
            "mixed_stable_births_min",
            fixture_payload.get("mixed_stable_births_mean", 0.0),
        )
    )
    fixture_births = float(
        horizon_payload.get("births_min", fixture_payload.get("births_mean", 0.0))
    )
    fixture_alive = float(
        horizon_payload.get(
            "alive_agents_min",
            fixture_payload.get("alive_agents_mean", 0.0),
        )
    )
    passed_horizon_count = int(
        horizon_payload.get(
            "passed_horizon_count",
            1 if bool(gate_payload.get("passed", False)) else 0,
        )
    )
    first_horizon_passed = _fixture_first_horizon_passed(entry)
    return (
        bool(gate_payload.get("passed", False)),
        passed_horizon_count,
        first_horizon_passed,
        -blocker_count,
        carrion_blocker_pressure * carrion_energy_requirement_satisfaction,
        carrion_blocker_pressure * carrion_energy_viability,
        carrion_blocker_pressure * carrion_hydration_viability,
        carrion_blocker_pressure * carrion_matched_diet_viability,
        carrion_blocker_pressure * carrion_resource_events,
        carrion_blocker_pressure * carrion_resource_gain,
        carrion_blocker_pressure * -carrion_dominant_action_share,
        carrion_blocker_pressure * carrion_alive,
        carrion_blocker_pressure * carrion_alive_ticks,
        carrion_blocker_pressure * fixture_energy_hydration_balance,
        carrion_blocker_pressure * fixture_reproduction_viability,
        carrion_blocker_pressure * carrion_births,
        holdout_energy_hydration_balance,
        holdout_balanced_readiness,
        float(holdout_payload.get("births_mean", 0.0)),
        float(holdout_payload.get("alive_agents_mean", 0.0)),
        float(
            holdout_payload.get(
                "terminal_energy_requirement_satisfaction_mean",
                0.0,
            )
        ),
        float(
            holdout_payload.get("reproduction_ready_agent_tick_share_mean", 0.0)
        ),
        float(
            holdout_payload.get("reproduction_ready_pair_tick_share_mean", 0.0)
        ),
        float(holdout_payload.get("core_readiness_agent_tick_mean", 0.0)),
        float(holdout_payload.get("core_ready_agent_tick_share_mean", 0.0)),
        float(holdout_payload.get("core_ready_pair_tick_share_mean", 0.0)),
        float(
            holdout_payload.get("terminal_ready_group_count_mean", 0.0)
        ),
        holdout_viability,
        -float(holdout_payload.get("dominant_requested_action_share", 1.0)),
        fixture_mixed_stable_births,
        fixture_births,
        fixture_alive,
        fixture_energy_hydration_balance,
        float(
            fixture_payload.get(
                "terminal_balanced_reproduction_readiness_mean",
                0.0,
            )
        ),
        fixture_reproduction_viability,
        float(fixture_payload.get("biologically_ready_agents_mean", 0.0)),
        float(entry.get("search_score", 0.0)),
        -int(entry.get("prefilter_rank", 0)),
    )


def _fixture_first_horizon_passed(entry: Mapping[str, object]) -> bool:
    horizons = entry.get("fixture_horizons")
    if not isinstance(horizons, list) or not horizons:
        gate = entry.get("fixture_gate")
        return isinstance(gate, Mapping) and bool(gate.get("passed", False))
    first = horizons[0]
    if not isinstance(first, Mapping):
        return False
    gate = first.get("fixture_gate")
    return isinstance(gate, Mapping) and bool(gate.get("passed", False))


def _fixture_gate_has_fixture_blocker(
    fixture_gate: Mapping[str, object],
    fixture_name: str,
) -> bool:
    blockers = fixture_gate.get("blockers")
    if not isinstance(blockers, list):
        return False
    for blocker in blockers:
        if isinstance(blocker, Mapping) and blocker.get("fixture") == fixture_name:
            return True
    return False


def _fixture_summary_named_fixture(
    fixture_summary: Mapping[str, object],
    fixture_name: str,
) -> Mapping[str, object]:
    per_fixture = fixture_summary.get("per_fixture")
    if not isinstance(per_fixture, Mapping):
        return {}
    payload = per_fixture.get(fixture_name)
    return payload if isinstance(payload, Mapping) else {}


def _fixture_observed_animal_resource_summary(
    mind_v3: Mapping[str, object],
) -> dict[str, float]:
    raw_runs = mind_v3.get("runs")
    runs = raw_runs if isinstance(raw_runs, list) else []
    fresh_kill_events: list[float] = []
    fresh_kill_energy: list[float] = []
    carcass_events: list[float] = []
    carcass_energy: list[float] = []
    for raw_run in runs:
        if not isinstance(raw_run, Mapping):
            continue
        fresh_kill = raw_run.get("fresh_kill_end")
        fresh_kill_payload = fresh_kill if isinstance(fresh_kill, Mapping) else {}
        carcass = raw_run.get("carcass_end")
        carcass_payload = carcass if isinstance(carcass, Mapping) else {}
        fresh_kill_events.append(
            float(fresh_kill_payload.get("consumption_events", 0.0))
        )
        fresh_kill_energy.append(float(fresh_kill_payload.get("gained_energy", 0.0)))
        carcass_events.append(float(carcass_payload.get("consumption_events", 0.0)))
        carcass_energy.append(float(carcass_payload.get("gained_energy", 0.0)))
    animal_events = [
        fresh + carcass
        for fresh, carcass in zip(fresh_kill_events, carcass_events)
    ]
    animal_energy = [
        fresh + carcass
        for fresh, carcass in zip(fresh_kill_energy, carcass_energy)
    ]
    return {
        "fresh_kill_consumption_events_mean": _round(_mean(fresh_kill_events)),
        "fresh_kill_gained_energy_mean": _round(_mean(fresh_kill_energy)),
        "carcass_consumption_events_mean": _round(_mean(carcass_events)),
        "carcass_gained_energy_mean": _round(_mean(carcass_energy)),
        "animal_resource_consumption_events_mean": _round(_mean(animal_events)),
        "animal_resource_gained_energy_mean": _round(_mean(animal_energy)),
    }


def _fixture_alive_agent_tick_summary(
    mind_v3: Mapping[str, object],
) -> dict[str, float]:
    aggregate = mind_v3.get("aggregate")
    if isinstance(aggregate, Mapping):
        aggregate_alive_ticks = aggregate.get("alive_agent_ticks_per_tick_mean")
        if isinstance(aggregate_alive_ticks, (int, float)) and not isinstance(
            aggregate_alive_ticks,
            bool,
        ):
            return {
                "alive_agent_ticks_per_tick_mean": _round(
                    float(aggregate_alive_ticks)
                )
            }
    raw_runs = mind_v3.get("runs")
    runs = raw_runs if isinstance(raw_runs, list) else []
    values: list[float] = []
    for raw_run in runs:
        if not isinstance(raw_run, Mapping):
            continue
        trajectory_record_count = raw_run.get("trajectory_record_count")
        if isinstance(trajectory_record_count, bool) or not isinstance(
            trajectory_record_count,
            (int, float),
        ):
            continue
        ticks_value = raw_run.get("ticks")
        if isinstance(ticks_value, bool) or not isinstance(ticks_value, (int, float)):
            ticks_value = raw_run.get("ticks_executed")
        if isinstance(ticks_value, bool) or not isinstance(ticks_value, (int, float)):
            continue
        values.append(float(trajectory_record_count) / max(1.0, float(ticks_value)))
    return {"alive_agent_ticks_per_tick_mean": _round(_mean(values))}


def _fixture_rerank_fixture_summary(
    fixture_suite: Mapping[str, object],
) -> dict[str, object]:
    raw_fixtures = fixture_suite.get("fixtures")
    fixtures = raw_fixtures if isinstance(raw_fixtures, list) else []
    alive_values: list[float] = []
    birth_values: list[float] = []
    viability_mins: list[float] = []
    balanced_values: list[float] = []
    energy_hydration_balance_values: list[float] = []
    energy_hydration_gap_values: list[float] = []
    biologically_ready_values: list[float] = []
    animal_resource_event_values: list[float] = []
    animal_resource_gain_values: list[float] = []
    alive_agent_ticks_per_tick_values: list[float] = []
    per_fixture: dict[str, object] = {}
    mixed_stable_births = 0.0
    carrion_only_summary: dict[str, float] = {}
    for raw_fixture in fixtures:
        if not isinstance(raw_fixture, Mapping):
            continue
        fixture_name = str(raw_fixture.get("fixture", "unknown"))
        comparison = raw_fixture.get("comparison")
        mind_v3 = comparison.get("mind_v3") if isinstance(comparison, Mapping) else None
        aggregate = mind_v3.get("aggregate") if isinstance(mind_v3, Mapping) else None
        if not isinstance(aggregate, Mapping):
            continue
        animal_resources = _fixture_observed_animal_resource_summary(mind_v3)
        alive_agent_ticks = _fixture_alive_agent_tick_summary(mind_v3)
        attribution = aggregate.get("reproduction_failure_attribution")
        attr_payload = attribution if isinstance(attribution, Mapping) else {}
        viability = attr_payload.get("terminal_viability_shares_mean")
        viability_payload = viability if isinstance(viability, Mapping) else {}
        alive = float(aggregate.get("alive_agents_mean", 0.0))
        births = float(aggregate.get("births_mean", 0.0))
        viability_min = min(
            float(viability_payload.get("energy", 0.0)),
            float(viability_payload.get("hydration", 0.0)),
            float(viability_payload.get("health", 0.0)),
            float(viability_payload.get("matched_diet", 0.0)),
        )
        energy_requirement = float(
            aggregate.get(
                "terminal_energy_requirement_satisfaction_mean",
                attr_payload.get(
                    "terminal_energy_requirement_satisfaction_mean",
                    0.0,
                ),
            )
        )
        hydration_viability = float(viability_payload.get("hydration", 0.0))
        health_viability = float(viability_payload.get("health", 0.0))
        matched_diet_viability = float(viability_payload.get("matched_diet", 0.0))
        balanced_readiness = float(
            aggregate.get(
                "terminal_balanced_reproduction_readiness_mean",
                _balanced_readiness_score(
                    [
                        energy_requirement,
                        hydration_viability,
                        health_viability,
                        matched_diet_viability,
                    ]
                ),
            )
        )
        energy_hydration_balance = float(
            aggregate.get(
                "terminal_energy_hydration_balance_mean",
                min(energy_requirement, hydration_viability),
            )
        )
        energy_hydration_gap = float(
            aggregate.get(
                "terminal_energy_hydration_gap_abs_mean",
                abs(energy_requirement - hydration_viability),
            )
        )
        biologically_ready = float(
            attr_payload.get("biologically_ready_agents_mean", 0.0)
        )
        energy_viability = float(viability_payload.get("energy", 0.0))
        animal_resource_events = float(
            animal_resources["animal_resource_consumption_events_mean"]
        )
        animal_resource_gain = float(
            animal_resources["animal_resource_gained_energy_mean"]
        )
        alive_agent_ticks_per_tick = float(
            alive_agent_ticks["alive_agent_ticks_per_tick_mean"]
        )
        alive_values.append(alive)
        birth_values.append(births)
        viability_mins.append(viability_min)
        balanced_values.append(balanced_readiness)
        energy_hydration_balance_values.append(energy_hydration_balance)
        energy_hydration_gap_values.append(energy_hydration_gap)
        biologically_ready_values.append(biologically_ready)
        animal_resource_event_values.append(animal_resource_events)
        animal_resource_gain_values.append(animal_resource_gain)
        alive_agent_ticks_per_tick_values.append(alive_agent_ticks_per_tick)
        if fixture_name == "mixed_stable":
            mixed_stable_births = births
        if fixture_name == "carrion_only":
            carrion_only_summary = {
                "alive_agents_mean": alive,
                "births_mean": births,
                "terminal_reproduction_viability_min": viability_min,
                "terminal_energy_viability_share_mean": energy_viability,
                "terminal_hydration_viability_share_mean": hydration_viability,
                "terminal_health_viability_share_mean": health_viability,
                "terminal_matched_diet_viability_share_mean": matched_diet_viability,
                "terminal_energy_requirement_satisfaction_mean": (
                    energy_requirement
                ),
                "terminal_energy_hydration_balance_mean": energy_hydration_balance,
                "terminal_balanced_reproduction_readiness_mean": balanced_readiness,
                "animal_resource_consumption_events_mean": (
                    animal_resource_events
                ),
                "animal_resource_gained_energy_mean": animal_resource_gain,
                "alive_agent_ticks_per_tick_mean": alive_agent_ticks_per_tick,
            }
        per_fixture[fixture_name] = {
            "alive_agents_mean": _round(alive),
            "births_mean": _round(births),
            **alive_agent_ticks,
            "terminal_reproduction_viability_min": _round(viability_min),
            "terminal_energy_viability_share_mean": _round(energy_viability),
            "terminal_hydration_viability_share_mean": _round(
                hydration_viability
            ),
            "terminal_health_viability_share_mean": _round(health_viability),
            "terminal_matched_diet_viability_share_mean": _round(
                matched_diet_viability
            ),
            "terminal_energy_requirement_satisfaction_mean": _round(
                energy_requirement
            ),
            "terminal_balanced_reproduction_readiness_mean": _round(
                balanced_readiness
            ),
            "terminal_energy_hydration_balance_mean": _round(
                energy_hydration_balance
            ),
            "terminal_energy_hydration_gap_abs_mean": _round(
                energy_hydration_gap
            ),
            "biologically_ready_agents_mean": _round(biologically_ready),
            "dominant_requested_action": aggregate.get("dominant_requested_action"),
            "dominant_requested_action_share": _round(
                float(aggregate.get("dominant_requested_action_share", 0.0))
            ),
            **animal_resources,
        }
    return {
        "fixture_count": len(per_fixture),
        "alive_agents_mean": _round(_mean(alive_values)),
        "births_mean": _round(_mean(birth_values)),
        "mixed_stable_births_mean": _round(mixed_stable_births),
        "carrion_only_alive_agents_mean": _round(
            carrion_only_summary.get("alive_agents_mean", 0.0)
        ),
        "carrion_only_births_mean": _round(
            carrion_only_summary.get("births_mean", 0.0)
        ),
        "carrion_only_alive_agent_ticks_per_tick_mean": _round(
            carrion_only_summary.get("alive_agent_ticks_per_tick_mean", 0.0)
        ),
        "carrion_only_terminal_reproduction_viability_min": _round(
            carrion_only_summary.get("terminal_reproduction_viability_min", 0.0)
        ),
        "carrion_only_terminal_energy_viability_share_mean": _round(
            carrion_only_summary.get("terminal_energy_viability_share_mean", 0.0)
        ),
        "carrion_only_terminal_energy_requirement_satisfaction_mean": _round(
            carrion_only_summary.get(
                "terminal_energy_requirement_satisfaction_mean",
                0.0,
            )
        ),
        "carrion_only_terminal_energy_hydration_balance_mean": _round(
            carrion_only_summary.get("terminal_energy_hydration_balance_mean", 0.0)
        ),
        "carrion_only_terminal_balanced_reproduction_readiness_mean": _round(
            carrion_only_summary.get(
                "terminal_balanced_reproduction_readiness_mean",
                0.0,
            )
        ),
        "carrion_only_animal_resource_consumption_events_mean": _round(
            carrion_only_summary.get("animal_resource_consumption_events_mean", 0.0)
        ),
        "carrion_only_animal_resource_gained_energy_mean": _round(
            carrion_only_summary.get("animal_resource_gained_energy_mean", 0.0)
        ),
        "terminal_reproduction_viability_min": (
            _round(min(viability_mins)) if viability_mins else 0.0
        ),
        "terminal_balanced_reproduction_readiness_mean": _round(
            _mean(balanced_values)
        ),
        "terminal_energy_hydration_balance_mean": _round(
            _mean(energy_hydration_balance_values)
        ),
        "terminal_energy_hydration_balance_min": (
            _round(min(energy_hydration_balance_values))
            if energy_hydration_balance_values
            else 0.0
        ),
        "terminal_energy_hydration_gap_abs_mean": _round(
            _mean(energy_hydration_gap_values)
        ),
        "biologically_ready_agents_mean": _round(_mean(biologically_ready_values)),
        "animal_resource_consumption_events_mean": _round(
            _mean(animal_resource_event_values)
        ),
        "animal_resource_gained_energy_mean": _round(
            _mean(animal_resource_gain_values)
        ),
        "alive_agent_ticks_per_tick_mean": _round(
            _mean(alive_agent_ticks_per_tick_values)
        ),
        "per_fixture": dict(sorted(per_fixture.items())),
    }


def _candidate_pool_order(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get("score", 0.0)),
            float(candidate.get("alive_agents_mean", 0.0)),
            float(candidate.get("births_mean", 0.0)),
            -int(candidate.get("candidate_index", 0)),
        ),
        reverse=True,
    )


def _candidate_is_founder_pool_eligible(candidate: Mapping[str, object]) -> bool:
    return (
        float(candidate.get("alive_agents_mean", 0.0))
        >= MIND_V3_FOUNDER_TEMPLATE_POOL_MIN_ALIVE_MEAN
        or float(candidate.get("births_mean", 0.0)) > 0.0
    )


def _diverse_founder_template_pool(
    templates: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_fingerprints: set[str] = set()
    seen_profiles: set[str] = set()

    def append_template(template: dict[str, object]) -> None:
        fingerprint = _metadata_fingerprint(template)
        if fingerprint in seen_fingerprints:
            return
        seen_fingerprints.add(fingerprint)
        selected.append(dict(template))

    for template in templates:
        profile = _metadata_specialization_profile(template)
        if profile in seen_profiles:
            continue
        seen_profiles.add(profile)
        append_template(template)
        if len(selected) >= MIND_V3_FOUNDER_TEMPLATE_POOL_LIMIT:
            return selected
    for template in templates:
        append_template(template)
        if len(selected) >= MIND_V3_FOUNDER_TEMPLATE_POOL_LIMIT:
            break
    return selected


def _metadata_fingerprint(metadata: Mapping[str, object]) -> str:
    return json.dumps(
        {
            "architecture": metadata.get("architecture"),
            "specialization_profile": metadata.get("specialization_profile"),
            "action_head_weights": metadata.get("action_head_weights"),
            "action_head_bias": metadata.get("action_head_bias"),
        },
        sort_keys=True,
    )


def _founder_template_pool_identity_diagnostics(
    templates: list[dict[str, object]],
) -> dict[str, object]:
    fingerprints = [
        _metadata_fingerprint_digest(template)
        for template in templates
        if isinstance(template, Mapping)
    ]
    return {
        "founder_template_pool_fingerprints": fingerprints,
        "founder_template_pool_distinct_fingerprint_count": len(set(fingerprints)),
    }


def _metadata_fingerprint_digest(metadata: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _metadata_fingerprint(metadata).encode("utf-8")
    ).hexdigest()[:16]


def _metadata_specialization_profile(metadata: Mapping[str, object]) -> str:
    profile = metadata.get("specialization_profile")
    return str(profile) if isinstance(profile, str) and profile else "unknown"


def _candidate_controller_architecture(candidate: Mapping[str, object]) -> str:
    metadata = candidate.get("controller_metadata")
    if not isinstance(metadata, Mapping):
        return "unknown"
    architecture = metadata.get("architecture")
    return str(architecture) if isinstance(architecture, str) and architecture else "unknown"


def _metadata_specialization_profile_counts(
    templates: list[dict[str, object]],
) -> dict[str, int]:
    counts = Counter(
        _metadata_specialization_profile(metadata)
        for metadata in templates
    )
    return dict(sorted(counts.items()))


def _candidate_controller_architecture_counts(
    candidates: list[dict[str, object]],
) -> dict[str, int]:
    counts = Counter(
        _candidate_controller_architecture(candidate)
        for candidate in candidates
    )
    return dict(sorted(counts.items()))


def _evaluate_holdout(
    *,
    seeds: list[int],
    ticks: int,
    metadata: dict[str, object] | list[dict[str, object]],
    rollout_workers: int = 1,
) -> dict[str, object]:
    runs = _run_candidate_batch(
        [
            {
                "seed": seed,
                "ticks": ticks,
                "metadata": _copy_founder_template(metadata),
            }
            for seed in seeds
        ],
        rollout_workers=rollout_workers,
    )
    return {
        "seeds": seeds,
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }


def _run_candidate_batch(
    tasks: list[dict[str, object]],
    *,
    rollout_workers: int,
) -> list[dict[str, object]]:
    worker_count = _resolved_rollout_worker_count(
        rollout_workers,
        work_item_count=len(tasks),
    )
    if worker_count <= 1:
        return [_run_candidate_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_run_candidate_task, tasks))


def _run_candidate_task(task: dict[str, object]) -> dict[str, object]:
    return _run_candidate(
        seed=int(task["seed"]),
        ticks=int(task["ticks"]),
        metadata=_copy_founder_template(task["metadata"]),
    )


def _build_report(
    *,
    search_seed: int,
    seeds: list[int],
    holdout_seeds: list[int],
    ticks: int,
    population_size: int,
    requested_generations: int,
    rollout_workers: int,
    resumed_from: str | None,
    generations: list[dict[str, object]],
    archive: dict[str, object],
    best_candidate: dict[str, object],
    next_candidates: list[dict[str, object]],
    rng: Random,
    holdout_evaluation: dict[str, object] | None,
    fixture_suite: dict[str, object] | None,
    fixture_gate: dict[str, object] | None,
    fixture_rerank: dict[str, object] | None,
    fixture_selection_top_k: int,
    warm_start: dict[str, object] | None,
) -> dict[str, object]:
    generation_candidates = _generation_candidates(generations)
    best_candidate = _candidate_with_founder_template_pool(
        best_candidate,
        archive=archive,
        candidates=generation_candidates,
    )
    search = {
        "seed": search_seed,
        "seeds": seeds,
        "holdout_seeds": holdout_seeds,
        "ticks": ticks,
        "population_size": population_size,
        "generation_count": requested_generations,
        "completed_generation_count": len(generations),
        "score_policy": MIND_V3_EVOLUTION_SCORE_POLICY,
        "founder_template_assignment_policy": (
            MIND_V3_FOUNDER_TEMPLATE_ASSIGNMENT_POLICY
        ),
        "heuristic_free": True,
        "rollout_execution": _rollout_execution_metadata(
            requested_workers=rollout_workers,
            population_size=population_size,
            holdout_seed_count=len(holdout_seeds),
        ),
        "controller_architecture_counts": (
            _candidate_controller_architecture_counts(generation_candidates)
        ),
        "best_candidate_controller_architecture": (
            _candidate_controller_architecture(best_candidate)
        ),
    }
    if resumed_from is not None:
        search["resumed_from"] = resumed_from
    if warm_start is not None:
        search["warm_start_policy"] = MIND_V3_WARM_START_POLICY
        search["warm_start_source_reports"] = list(
            warm_start.get("source_reports", [])
        )
        search["warm_start_candidate_count"] = int(
            warm_start.get("candidate_count", 0)
        )
        search["warm_start_candidate_limit"] = int(
            warm_start.get("candidate_limit", 0)
        )
    if fixture_gate is not None:
        search["fixture_gate_policy"] = MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY
        search["fixture_suite_policy"] = MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY
    if fixture_rerank is not None:
        search["fixture_rerank_policy"] = MIND_V3_FIXTURE_RERANK_POLICY
        search["fixture_rerank_top_k"] = int(fixture_rerank["top_k"])
    archive_retention = archive.get("fixture_recovery_archive_retention")
    if isinstance(archive_retention, Mapping):
        search["fixture_recovery_archive_retention_policy"] = str(
            archive_retention.get("policy", "")
        )
        search["fixture_recovery_archive_retention_added_count"] = int(
            archive_retention.get("retention_added_count", 0)
        )
    if fixture_selection_top_k > 0:
        search["fixture_selection_policy"] = (
            MIND_V3_GENERATION_FIXTURE_SELECTION_POLICY
        )
        search["fixture_selection_nominee_policy"] = (
            MIND_V3_GENERATION_FIXTURE_SELECTION_NOMINEE_POLICY
        )
        search["fixture_selection_top_k"] = int(fixture_selection_top_k)
    return {
        "schema_version": MIND_V3_EVOLUTION_SEARCH_SCHEMA_VERSION,
        "search": search,
        "archive": archive,
        "best_candidate": best_candidate,
        "holdout_evaluation": holdout_evaluation,
        "fixture_suite": fixture_suite,
        "fixture_gate": fixture_gate,
        "fixture_rerank": fixture_rerank,
        "warm_start": warm_start,
        "generations": generations,
        "resume": {
            "completed_generations": len(generations),
            "next_generation_index": len(generations),
            "next_candidates": next_candidates,
            "rng_state": _encode_rng_state(rng.getstate()),
        },
    }


def _comparison_baseline_report(
    report: Mapping[str, object],
    baseline_path: Path,
) -> dict[str, object]:
    if not baseline_path.exists():
        raise SystemExit(
            f"--comparison-baseline-report does not exist: {baseline_path}"
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_runs = _comparison_holdout_runs(report)
    baseline_runs = _comparison_holdout_runs(baseline)
    baseline_by_seed = {
        int(run["seed"]): run
        for run in baseline_runs
        if isinstance(run.get("seed"), int)
    }
    deltas: list[dict[str, object]] = []
    for current in current_runs:
        seed = current.get("seed")
        if not isinstance(seed, int) or seed not in baseline_by_seed:
            continue
        baseline_run = baseline_by_seed[seed]
        deltas.append(
            _comparison_seed_delta(
                seed=seed,
                current=current,
                baseline=baseline_run,
            )
        )
    current_heuristic_count = _comparison_heuristic_action_source_count(report)
    baseline_heuristic_count = _comparison_heuristic_action_source_count(baseline)
    current_fixture_gate = _comparison_fixture_gate_summary(report)
    baseline_fixture_gate = _comparison_fixture_gate_summary(baseline)
    return {
        "policy": "mind_v3_matched_holdout_baseline_report_delta_v1",
        "baseline_report": str(baseline_path),
        "matched_seed_count": len(deltas),
        "matched_holdout_seed_deltas": deltas,
        "heuristic_action_source_count_current": current_heuristic_count,
        "heuristic_action_source_count_baseline": baseline_heuristic_count,
        "heuristic_action_source_count_delta": _optional_numeric_delta(
            current_heuristic_count,
            baseline_heuristic_count,
        ),
        "fixture_gate": {
            "current_passed": current_fixture_gate["passed"],
            "baseline_passed": baseline_fixture_gate["passed"],
            "current_blocker_count": current_fixture_gate["blocker_count"],
            "baseline_blocker_count": baseline_fixture_gate["blocker_count"],
            "blocker_count_delta": (
                int(current_fixture_gate["blocker_count"])
                - int(baseline_fixture_gate["blocker_count"])
            ),
            "current_blocker_counts_by_fixture": current_fixture_gate[
                "blocker_counts_by_fixture"
            ],
            "baseline_blocker_counts_by_fixture": baseline_fixture_gate[
                "blocker_counts_by_fixture"
            ],
            "blocker_count_delta_by_fixture": _count_mapping_delta(
                current_fixture_gate["blocker_counts_by_fixture"],
                baseline_fixture_gate["blocker_counts_by_fixture"],
            ),
            "current_blocker_counts_by_reason": current_fixture_gate[
                "blocker_counts_by_reason"
            ],
            "baseline_blocker_counts_by_reason": baseline_fixture_gate[
                "blocker_counts_by_reason"
            ],
            "blocker_count_delta_by_reason": _count_mapping_delta(
                current_fixture_gate["blocker_counts_by_reason"],
                baseline_fixture_gate["blocker_counts_by_reason"],
            ),
            "current_blocker_counts_by_metric": current_fixture_gate[
                "blocker_counts_by_metric"
            ],
            "baseline_blocker_counts_by_metric": baseline_fixture_gate[
                "blocker_counts_by_metric"
            ],
            "blocker_count_delta_by_metric": _count_mapping_delta(
                current_fixture_gate["blocker_counts_by_metric"],
                baseline_fixture_gate["blocker_counts_by_metric"],
            ),
        },
    }


def _attach_comparison_baseline_report(
    report: dict[str, object],
    baseline_path: Path | None,
) -> None:
    if baseline_path is None:
        return
    report["comparison_baseline"] = _comparison_baseline_report(
        report,
        baseline_path,
    )


def _comparison_holdout_runs(report: Mapping[str, object]) -> list[dict[str, object]]:
    holdout = report.get("holdout_evaluation")
    if isinstance(holdout, Mapping) and isinstance(holdout.get("runs"), list):
        return [
            dict(run)
            for run in holdout["runs"]
            if isinstance(run, Mapping)
        ]
    best = report.get("best_candidate")
    if isinstance(best, Mapping) and isinstance(best.get("runs"), list):
        return [dict(run) for run in best["runs"] if isinstance(run, Mapping)]
    return []


def _comparison_seed_delta(
    *,
    seed: int,
    current: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    current_dominant = _dominant_action_summary(
        _counts_from_mapping(current.get("requested_action_counts"))
    )
    baseline_dominant = _dominant_action_summary(
        _counts_from_mapping(baseline.get("requested_action_counts"))
    )
    return {
        "seed": seed,
        "alive_agents_delta": _numeric_delta(current, baseline, "alive_agents"),
        "births_delta": _numeric_delta(current, baseline, "births"),
        "deaths_delta": _numeric_delta(current, baseline, "deaths"),
        "unsupported_requested_action_count_delta": _numeric_delta(
            current,
            baseline,
            "unsupported_requested_action_count",
        ),
        "unsupported_resolved_action_count_delta": _numeric_delta(
            current,
            baseline,
            "unsupported_resolved_action_count",
        ),
        "heuristic_action_source_count_delta": _numeric_delta(
            current,
            baseline,
            "heuristic_action_source_count",
        ),
        "current_dominant_requested_action": current_dominant["action"],
        "baseline_dominant_requested_action": baseline_dominant["action"],
        "dominant_requested_action_changed": (
            current_dominant["action"] != baseline_dominant["action"]
        ),
        "current_dominant_requested_action_share": current_dominant["share"],
        "baseline_dominant_requested_action_share": baseline_dominant["share"],
        "dominant_requested_action_share_delta": _round(
            float(current_dominant["share"]) - float(baseline_dominant["share"])
        ),
    }


def _comparison_heuristic_action_source_count(
    report: Mapping[str, object],
) -> float | None:
    holdout = report.get("holdout_evaluation")
    if isinstance(holdout, Mapping):
        aggregate = holdout.get("aggregate")
        if isinstance(aggregate, Mapping):
            value = _float_value(aggregate.get("heuristic_action_source_count"))
            if value is not None:
                return value
    best = report.get("best_candidate")
    if isinstance(best, Mapping):
        value = _float_value(best.get("heuristic_action_source_count"))
        if value is not None:
            return value
    runs = _comparison_holdout_runs(report)
    values = [
        _float_value(run.get("heuristic_action_source_count"))
        for run in runs
    ]
    parsed = [value for value in values if value is not None]
    if parsed:
        return _round(sum(parsed))
    return None


def _comparison_fixture_gate_summary(
    report: Mapping[str, object],
) -> dict[str, object]:
    fixture_gate = report.get("fixture_gate")
    if not isinstance(fixture_gate, Mapping):
        return {
            "passed": None,
            "blocker_count": 0,
            "blocker_counts_by_fixture": {},
            "blocker_counts_by_reason": {},
            "blocker_counts_by_metric": {},
        }
    blockers = fixture_gate.get("blockers")
    blocker_list = (
        [blocker for blocker in blockers if isinstance(blocker, Mapping)]
        if isinstance(blockers, list)
        else []
    )
    return {
        "passed": (
            bool(fixture_gate.get("passed"))
            if "passed" in fixture_gate
            else None
        ),
        "blocker_count": len(blocker_list),
        "blocker_counts_by_fixture": _comparison_blocker_counts(
            blocker_list,
            "fixture",
        ),
        "blocker_counts_by_reason": _comparison_blocker_counts(
            blocker_list,
            "reason",
        ),
        "blocker_counts_by_metric": _comparison_blocker_counts(
            blocker_list,
            "metric",
        ),
    }


def _comparison_blocker_counts(
    blockers: Sequence[Mapping[str, object]],
    key: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for blocker in blockers:
        counts.update([str(blocker.get(key) or "unknown")])
    return dict(sorted((name, int(count)) for name, count in counts.items()))


def _count_mapping_delta(
    current: object,
    baseline: object,
) -> dict[str, int]:
    current_mapping = current if isinstance(current, Mapping) else {}
    baseline_mapping = baseline if isinstance(baseline, Mapping) else {}
    keys = set(str(key) for key in current_mapping) | set(
        str(key) for key in baseline_mapping
    )
    return {
        key: int(current_mapping.get(key, 0)) - int(baseline_mapping.get(key, 0))
        for key in sorted(keys)
    }


def _optional_numeric_delta(
    current_value: float | None,
    baseline_value: float | None,
) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    return _round(current_value - baseline_value)


def _numeric_delta(
    current: Mapping[str, object],
    baseline: Mapping[str, object],
    key: str,
) -> float | None:
    current_value = _float_value(current.get(key))
    baseline_value = _float_value(baseline.get(key))
    if current_value is None or baseline_value is None:
        return None
    return _round(current_value - baseline_value)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_resume_state(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != MIND_V3_EVOLUTION_SEARCH_SCHEMA_VERSION:
        raise SystemExit("--resume-from must point to a Mind v3 evolution report")
    search = report.get("search")
    resume = report.get("resume")
    if not isinstance(search, dict) or not isinstance(resume, dict):
        raise SystemExit("resume report is missing search/resume state")
    next_candidates = resume.get("next_candidates")
    if not isinstance(next_candidates, list):
        raise SystemExit("resume report is missing next candidates")
    generations = report.get("generations")
    if not isinstance(generations, list):
        raise SystemExit("resume report is missing generations")
    return {
        "search_seed": int(search["seed"]),
        "seeds": [int(seed) for seed in list(search["seeds"])],
        "holdout_seeds": [
            int(seed) for seed in list(search.get("holdout_seeds", []))
        ],
        "ticks": int(search["ticks"]),
        "population_size": int(search["population_size"]),
        "generations": generations,
        "next_generation_index": int(resume["next_generation_index"]),
        "next_candidates": next_candidates,
        "rng_state": resume["rng_state"],
    }


def _encode_rng_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_encode_rng_state(item) for item in value]
    return value


def _decode_rng_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_decode_rng_state(item) for item in value)
    return value


def _heuristic_action_source_count(counts: Counter[str]) -> int:
    return sum(count for source, count in counts.items() if "heuristic" in source)


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seeds must include at least one integer seed")
    return seeds


def _parse_optional_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return []
    return _parse_seeds(raw)


def _fixture_config_from_args(
    args: argparse.Namespace,
    *,
    search_seeds: list[int],
    holdout_seeds: list[int],
    default_ticks: int | None,
) -> dict[str, object] | None:
    if args.fixture_suite == "none":
        return None
    fixture_seeds = (
        _parse_seeds(args.fixture_seeds)
        if args.fixture_seeds is not None
        else list(holdout_seeds or search_seeds)
    )
    fixture_ticks = (
        int(args.fixture_ticks)
        if args.fixture_ticks is not None
        else default_ticks
    )
    fixture_names = _parse_fixture_names(
        args.fixture_names,
        suite=str(args.fixture_suite),
    )
    config = mind_v3_fixture_gate_config(
        suite=str(args.fixture_suite),
        seeds=fixture_seeds,
        ticks=fixture_ticks,
        min_alive=float(args.fixture_min_alive),
        min_births=float(args.fixture_min_births),
        min_mixed_stable_births=float(args.fixture_min_mixed_stable_births),
        min_energy_viability=float(args.fixture_min_energy_viability),
        min_hydration_viability=float(args.fixture_min_hydration_viability),
        min_health_viability=float(args.fixture_min_health_viability),
        min_matched_diet_viability=float(args.fixture_min_matched_diet_viability),
        min_biologically_ready=float(args.fixture_min_biologically_ready),
    )
    if fixture_names is not None:
        config["fixture_names"] = fixture_names
    return config


def _parse_fixture_names(raw: str | None, *, suite: str) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    if suite != "basic":
        raise SystemExit(f"unsupported fixture suite: {suite}")
    supported = set(CONTROLLED_FIXTURE_NAMES)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise SystemExit("--fixture-names must include at least one fixture name")
    unsupported = sorted(name for name in names if name not in supported)
    if unsupported:
        raise SystemExit(
            "--fixture-names includes unsupported fixture(s): "
            + ", ".join(unsupported)
        )
    return list(dict.fromkeys(names))


def _fixture_names_from_config(
    fixture_config: Mapping[str, object],
) -> list[str] | None:
    raw = fixture_config.get("fixture_names")
    if not isinstance(raw, list):
        return None
    names = [str(name) for name in raw if isinstance(name, str) and name]
    return names or None


def _fixture_rerank_ticks_from_args(
    args: argparse.Namespace,
    *,
    default_ticks: int,
) -> list[int] | None:
    if args.fixture_rerank_ticks is None:
        return None
    ticks = _parse_positive_ints(
        args.fixture_rerank_ticks,
        flag_name="--fixture-rerank-ticks",
    )
    selected: list[int] = []
    seen: set[int] = set()
    for value in ticks:
        if value in seen:
            continue
        seen.add(value)
        selected.append(value)
    active_ticks = int(default_ticks)
    if active_ticks in seen:
        selected = [active_ticks, *[value for value in selected if value != active_ticks]]
    else:
        selected.insert(0, active_ticks)
    return selected


def _fixture_rerank_selector_probe_trajectory_output_dir(
    args: argparse.Namespace,
) -> Path | None:
    if not args.fixture_rerank_selector_probe:
        return None
    if args.fixture_rerank_selector_probe_trajectory_output_dir is not None:
        return Path(args.fixture_rerank_selector_probe_trajectory_output_dir)
    output = Path(args.output)
    return output.with_suffix("").with_name(
        f"{output.with_suffix('').name}-selector-probe-trajectories"
    )


def _parse_positive_ints(raw: str, *, flag_name: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{flag_name} must include at least one integer value")
    if any(value < 1 for value in values):
        raise SystemExit(f"{flag_name} values must be >= 1")
    return values


def _rollout_execution_metadata(
    *,
    requested_workers: int,
    population_size: int,
    holdout_seed_count: int,
) -> dict[str, object]:
    return {
        "policy": MIND_V3_ROLLOUT_EXECUTION_POLICY,
        "requested_workers": requested_workers,
        "generation_workers": _resolved_rollout_worker_count(
            requested_workers,
            work_item_count=population_size,
        ),
        "holdout_workers": (
            _resolved_rollout_worker_count(
                requested_workers,
                work_item_count=holdout_seed_count,
            )
            if holdout_seed_count > 0
            else 0
        ),
    }


def _resolved_rollout_worker_count(
    requested_workers: int,
    *,
    work_item_count: int,
) -> int:
    if requested_workers < 1:
        raise ValueError("rollout workers must be >= 1")
    if work_item_count <= 0:
        return 1
    return max(1, min(int(requested_workers), int(work_item_count)))


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / float(len(values)))


def _round(value: float) -> float:
    return round(float(value), 4)


if __name__ == "__main__":
    main()
