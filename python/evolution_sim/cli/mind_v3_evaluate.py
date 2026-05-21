from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any

from evolution_sim.config import ClimateConfig, ResourceRegrowthConfig, WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime import reproduction as runtime_reproduction
from evolution_sim.env.runtime import trajectory as runtime_trajectory
from evolution_sim.env.runtime.state import Agent
from evolution_sim.genome import Genome
from evolution_sim.genome.species import genome_vector
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.evolution import (
    load_mind_v3_founder_template,
    require_mind_v3_founder_template_promotion_eligible,
)
from evolution_sim.mind.outcome_metrics import (
    aggregate_run_outcome_metrics,
    build_run_outcome_metrics,
)
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_MODEL_TYPE,
    load_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_policy import (
    MIND_V3_REPRODUCTION_READINESS_GOALS,
    MindV3EvolutionPolicy,
)
from evolution_sim.mind.v3_planner_distilled import (
    MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
    load_mind_v3_planner_distilled_artifact,
)

MIND_V3_EVALUATION_SCHEMA_VERSION = (
    "mind_v3_autonomous_evolution_evaluation_v1"
)
MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY = (
    "terminal_reproduction_failure_attribution_v2"
)
MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY = (
    "trajectory_temporal_reproduction_blocker_attribution_v1"
)
MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY = (
    "mind_v3_controlled_ecology_fixture_suite_v1"
)
MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY = (
    "mind_v3_controlled_fixture_hard_gate_v1"
)
MIND_V3_NEURAL_ANCHOR_DIAGNOSTICS_POLICY = (
    "mind_v3_neural_anchor_diagnostics_v1"
)
MIND_V3_V97_PLANNER_DISTILLED_PROMOTION_POLICY = (
    "mind_v3_v97_planner_distilled_runtime_promotion_gate_v1"
)
V97_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
V97_CARRION_FIXTURE_SEEDS = (13, 19, 29, 37, 41, 43)
V97_TICKS = 120
V97_MAX_DOMINANT_REQUESTED_ACTION_SHARE = 0.50
CONTROLLED_FIXTURE_NAMES = (
    "plant_only",
    "carrion_only",
    "prey_rich",
    "mixed_stable",
)
BIOLOGICAL_BLOCKER_REASONS = (
    "age",
    "cooldown",
    "energy",
    "hydration",
    "health",
    "matched_diet",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Mind v3 autonomous evolution against the heuristic "
            "baseline without heuristic fallback inside the v3 policy."
        )
    )
    parser.add_argument(
        "--seeds",
        default="5,13,19,29",
        help="Comma-separated deterministic seed list.",
    )
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--founder-template",
        type=Path,
        help=(
            "Optional raw Mind v3 controller metadata or evolution-search "
            "report whose best candidate initializes founders."
        ),
    )
    parser.add_argument(
        "--neural-artifact",
        type=Path,
        help=(
            "Optional frozen Mind v3 neural policy artifact. The artifact "
            "uses ecological policy inputs and does not mutate neural weights "
            "inside a run."
        ),
    )
    parser.add_argument(
        "--anchored-neural-artifact",
        type=Path,
        help=(
            "Optional current anchored neural artifact to evaluate beside the "
            "primary --neural-artifact. This is intended for comparing an "
            "experimental direct artifact against the existing linear-anchor "
            "neural path in the same report."
        ),
    )
    parser.add_argument(
        "--planner-distilled-artifact",
        type=Path,
        help=(
            "Optional frozen v96 planner-distilled Mind v3 action scorer "
            "artifact or v96 report containing distilled_artifact. This is a "
            "runtime-feasibility artifact and is evaluated without planner "
            "outcome tables or global assignment."
        ),
    )
    parser.add_argument(
        "--compare-linear-baseline",
        action="store_true",
        help=(
            "When --neural-artifact or --planner-distilled-artifact is set, "
            "also evaluate the current linear Mind v3 controller on the same "
            "seeds/ticks and report primary-minus-linear deltas."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-autonomous-evolution-report.json"),
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        help=(
            "Optional directory for per-run trajectory JSONL.gz files. "
            "This preserves broad and controlled-fixture runs for horizon "
            "label generation without building full replay payloads."
        ),
    )
    parser.add_argument(
        "--fixture-suite",
        choices=["none", "basic"],
        default="none",
        help=(
            "Optionally run controlled ecology fixtures through the same "
            "simulator and policies. Defaults to disabled."
        ),
    )
    parser.add_argument(
        "--fixture-seeds",
        help=(
            "Comma-separated deterministic seed list for --fixture-suite. "
            "Defaults to --seeds."
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
        help="Tick horizon for --fixture-suite. Defaults to --ticks.",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    founder_template = (
        load_mind_v3_founder_template(args.founder_template)
        if args.founder_template is not None
        else None
    )
    if founder_template is not None:
        try:
            require_mind_v3_founder_template_promotion_eligible(founder_template)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    neural_artifact = (
        load_mind_v3_neural_artifact(args.neural_artifact)
        if args.neural_artifact is not None
        else None
    )
    planner_distilled_source_payload = (
        _load_json_mapping(args.planner_distilled_artifact)
        if args.planner_distilled_artifact is not None
        else None
    )
    planner_distilled_artifact = (
        load_mind_v3_planner_distilled_artifact(
            planner_distilled_source_payload
            if planner_distilled_source_payload is not None
            else args.planner_distilled_artifact
        )
        if args.planner_distilled_artifact is not None
        else None
    )
    anchored_neural_artifact = (
        load_mind_v3_neural_artifact(args.anchored_neural_artifact)
        if args.anchored_neural_artifact is not None
        else None
    )
    if planner_distilled_artifact is not None and neural_artifact is not None:
        raise SystemExit(
            "--planner-distilled-artifact and --neural-artifact are mutually exclusive"
        )
    if (
        planner_distilled_artifact is not None
        and anchored_neural_artifact is not None
    ):
        raise SystemExit(
            "--planner-distilled-artifact cannot be combined with "
            "--anchored-neural-artifact"
        )
    if (
        anchored_neural_artifact is not None
        and anchored_neural_artifact.get("model_type") != MIND_V3_NEURAL_MODEL_TYPE
    ):
        raise SystemExit(
            "--anchored-neural-artifact must use the current anchored neural "
            f"model_type {MIND_V3_NEURAL_MODEL_TYPE}"
        )
    if (
        args.compare_linear_baseline
        and neural_artifact is None
        and planner_distilled_artifact is None
    ):
        raise SystemExit(
            "--compare-linear-baseline requires --neural-artifact or "
            "--planner-distilled-artifact"
        )
    heuristic_runs = [
        _run_once(
            seed=seed,
            ticks=args.ticks,
            policy=None,
            trajectory_output_path=_trajectory_output_path(
                args.trajectory_output_dir,
                "open",
                "heuristic",
                seed,
                args.ticks,
            ),
            trajectory_split_id="mind_v3_evaluate_open_heuristic",
        )
        for seed in seeds
    ]
    linear_runs = (
        [
            _run_once(
                seed=seed,
                ticks=args.ticks,
                policy=_mind_v3_policy(
                    seed=seed,
                    founder_template=founder_template,
                ),
                trajectory_output_path=_trajectory_output_path(
                    args.trajectory_output_dir,
                    "open",
                    "mind_v3_linear",
                    seed,
                    args.ticks,
                ),
                trajectory_split_id="mind_v3_evaluate_open_linear",
            )
            for seed in seeds
        ]
        if args.compare_linear_baseline
        else None
    )
    mind_v3_runs = [
        _run_once(
            seed=seed,
            ticks=args.ticks,
            policy=_mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=neural_artifact,
                planner_distilled_artifact=planner_distilled_artifact,
            ),
            trajectory_output_path=_trajectory_output_path(
                args.trajectory_output_dir,
                "open",
                "mind_v3",
                seed,
                args.ticks,
            ),
            trajectory_split_id="mind_v3_evaluate_open_mind_v3",
        )
        for seed in seeds
    ]
    anchored_neural_runs = (
        [
            _run_once(
                seed=seed,
                ticks=args.ticks,
                policy=_mind_v3_policy(
                    seed=seed,
                    founder_template=founder_template,
                    neural_artifact=anchored_neural_artifact,
                ),
                trajectory_output_path=_trajectory_output_path(
                    args.trajectory_output_dir,
                    "open",
                    "mind_v3_anchored_neural",
                    seed,
                    args.ticks,
                ),
                trajectory_split_id="mind_v3_evaluate_open_anchored_neural",
            )
            for seed in seeds
        ]
        if anchored_neural_artifact is not None
        else None
    )
    report = {
        "schema_version": MIND_V3_EVALUATION_SCHEMA_VERSION,
        "policy": {
            "policy_id": "mind_v3_autonomous_evolution_policy",
            "policy_version": "mind_v3_autonomous_evolution_policy_v1",
            "heuristic_free": True,
            "founder_template_source": (
                str(args.founder_template)
                if args.founder_template is not None
                else None
            ),
            "founder_template_count": _founder_template_count(founder_template),
            "founder_template_specialization_profile_counts": (
                _founder_template_specialization_profile_counts(founder_template)
            ),
            "neural_artifact_source": (
                str(args.neural_artifact)
                if args.neural_artifact is not None
                else None
            ),
            "neural_artifact_schema_version": (
                neural_artifact.get("schema_version")
                if isinstance(neural_artifact, dict)
                else None
            ),
            "neural_model_type": (
                neural_artifact.get("model_type")
                if isinstance(neural_artifact, dict)
                else None
            ),
            "neural_input_policy": (
                neural_artifact.get("input_policy")
                if isinstance(neural_artifact, dict)
                else None
            ),
            "anchored_neural_artifact_source": (
                str(args.anchored_neural_artifact)
                if args.anchored_neural_artifact is not None
                else None
            ),
            "anchored_neural_model_type": (
                anchored_neural_artifact.get("model_type")
                if isinstance(anchored_neural_artifact, dict)
                else None
            ),
            "planner_distilled_artifact_source": (
                str(args.planner_distilled_artifact)
                if args.planner_distilled_artifact is not None
                else None
            ),
            "planner_distilled_artifact_schema_version": (
                planner_distilled_artifact.get("schema_version")
                if isinstance(planner_distilled_artifact, dict)
                else None
            ),
            "planner_distilled_model_type": (
                planner_distilled_artifact.get("model_type")
                if isinstance(planner_distilled_artifact, dict)
                else None
            ),
            "planner_distilled_source_reload_actions_match": (
                _planner_distilled_source_reload_actions_match(
                    planner_distilled_source_payload
                )
            ),
            "linear_baseline_compared": bool(args.compare_linear_baseline),
        },
        "comparison": {
            "heuristic": {
                "runs": heuristic_runs,
                "aggregate": _aggregate_runs(heuristic_runs),
            },
            "mind_v3": {
                "runs": mind_v3_runs,
                "aggregate": _aggregate_runs(mind_v3_runs),
            },
        },
    }
    if linear_runs is not None:
        report["comparison"]["mind_v3_linear"] = {
            "runs": linear_runs,
            "aggregate": _aggregate_runs(linear_runs),
        }
    if anchored_neural_runs is not None:
        report["comparison"]["mind_v3_anchored_neural"] = {
            "runs": anchored_neural_runs,
            "aggregate": _aggregate_runs(anchored_neural_runs),
        }
    report["comparison"]["delta"] = _comparison_delta(
        heuristic=report["comparison"]["heuristic"]["aggregate"],
        mind_v3=report["comparison"]["mind_v3"]["aggregate"],
    )
    if linear_runs is not None:
        primary_vs_linear_delta = _comparison_delta(
            heuristic=report["comparison"]["mind_v3_linear"]["aggregate"],
            mind_v3=report["comparison"]["mind_v3"]["aggregate"],
        )
        if neural_artifact is not None:
            report["comparison"]["neural_vs_linear_delta"] = (
                primary_vs_linear_delta
            )
        if planner_distilled_artifact is not None:
            report["comparison"]["planner_vs_linear_delta"] = (
                primary_vs_linear_delta
            )
    if anchored_neural_runs is not None:
        report["comparison"]["primary_vs_anchored_neural_delta"] = (
            _comparison_delta(
                heuristic=report["comparison"]["mind_v3_anchored_neural"][
                    "aggregate"
                ],
                mind_v3=report["comparison"]["mind_v3"]["aggregate"],
            )
        )
    if args.fixture_suite != "none":
        fixture_seeds = (
            _parse_seeds(args.fixture_seeds)
            if args.fixture_seeds is not None
            else seeds
        )
        fixture_ticks = (
            int(args.fixture_ticks)
            if args.fixture_ticks is not None
            else int(args.ticks)
        )
        fixture_names = _parse_fixture_names(
            args.fixture_names,
            suite=args.fixture_suite,
        )
        fixture_config = mind_v3_fixture_gate_config(
            suite=args.fixture_suite,
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
        report["fixture_suite"] = run_mind_v3_fixture_suite(
            suite=args.fixture_suite,
            fixture_names=fixture_names,
            seeds=fixture_seeds,
            ticks=fixture_ticks,
            founder_template=founder_template,
            neural_artifact=neural_artifact,
            planner_distilled_artifact=planner_distilled_artifact,
            trajectory_output_dir=args.trajectory_output_dir,
            trajectory_prefix="fixture",
        )
        report["fixture_gate"] = mind_v3_fixture_gate_status(
            fixture_suite=report["fixture_suite"],
            fixture_config=fixture_config,
        )
        if args.compare_linear_baseline:
            report["linear_baseline_fixture_suite"] = run_mind_v3_fixture_suite(
                suite=args.fixture_suite,
                fixture_names=fixture_names,
                seeds=fixture_seeds,
                ticks=fixture_ticks,
                founder_template=founder_template,
                trajectory_output_dir=args.trajectory_output_dir,
                trajectory_prefix="fixture_linear_baseline",
            )
            report["linear_baseline_fixture_gate"] = mind_v3_fixture_gate_status(
                fixture_suite=report["linear_baseline_fixture_suite"],
                fixture_config=fixture_config,
            )
            primary_vs_linear_fixture_delta = _fixture_gate_comparison_delta(
                linear_gate=report["linear_baseline_fixture_gate"],
                neural_gate=report["fixture_gate"],
            )
            if neural_artifact is not None:
                report["neural_vs_linear_fixture_delta"] = (
                    primary_vs_linear_fixture_delta
                )
            if planner_distilled_artifact is not None:
                report["planner_vs_linear_fixture_delta"] = (
                    primary_vs_linear_fixture_delta
                )
        if anchored_neural_artifact is not None:
            report["anchored_neural_fixture_suite"] = run_mind_v3_fixture_suite(
                suite=args.fixture_suite,
                fixture_names=fixture_names,
                seeds=fixture_seeds,
                ticks=fixture_ticks,
                founder_template=founder_template,
                neural_artifact=anchored_neural_artifact,
                trajectory_output_dir=args.trajectory_output_dir,
                trajectory_prefix="fixture_anchored_neural",
            )
            report["anchored_neural_fixture_gate"] = mind_v3_fixture_gate_status(
                fixture_suite=report["anchored_neural_fixture_suite"],
                fixture_config=fixture_config,
            )
            report["primary_vs_anchored_neural_fixture_delta"] = (
                _fixture_gate_comparison_delta(
                    linear_gate=report["anchored_neural_fixture_gate"],
                    neural_gate=report["fixture_gate"],
                )
            )
    if planner_distilled_artifact is not None:
        report["v97_planner_distilled_promotion"] = (
            _v97_planner_distilled_promotion_report(report)
        )
        _attach_v97_promotion_ledger_metrics(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"mind_v3_report={args.output}")
    print(
        "mind_v3_heuristic_action_source_count="
        f"{report['comparison']['mind_v3']['aggregate']['heuristic_action_source_count']}"
    )
    print(
        "mind_v3_alive_agents_mean="
        f"{report['comparison']['mind_v3']['aggregate']['alive_agents_mean']}"
    )
    print(
        "mind_v3_births_mean="
        f"{report['comparison']['mind_v3']['aggregate']['births_mean']}"
    )
    if "fixture_suite" in report:
        print(
            "mind_v3_fixture_count="
            f"{len(report['fixture_suite']['fixtures'])}"
        )
    if "fixture_gate" in report:
        print(f"mind_v3_fixture_gate_passed={report['fixture_gate']['passed']}")
    if "comparison" in report and "neural_vs_linear_delta" in report["comparison"]:
        delta = report["comparison"]["neural_vs_linear_delta"]
        print(f"mind_v3_neural_minus_linear_alive={delta['alive_agents_mean']}")
        print(f"mind_v3_neural_minus_linear_births={delta['births_mean']}")
    if "comparison" in report and "planner_vs_linear_delta" in report["comparison"]:
        delta = report["comparison"]["planner_vs_linear_delta"]
        print(f"mind_v3_planner_minus_linear_alive={delta['alive_agents_mean']}")
        print(f"mind_v3_planner_minus_linear_births={delta['births_mean']}")
    if "v97_planner_distilled_promotion" in report:
        promotion = report["v97_planner_distilled_promotion"]
        print(
            "mind_v3_v97_promotion_candidate_passed="
            f"{promotion['promotion_candidate_passed']}"
        )


def _mind_v3_policy(
    *,
    seed: int,
    founder_template: dict[str, object] | list[dict[str, object]] | None,
    neural_artifact: dict[str, object] | None = None,
    planner_distilled_artifact: dict[str, object] | None = None,
) -> MindV3EvolutionPolicy:
    return MindV3EvolutionPolicy(
        seed=seed,
        founder_template_metadata=founder_template,
        neural_artifact=neural_artifact,
        planner_distilled_artifact=planner_distilled_artifact,
    )


def _run_once(
    *,
    seed: int,
    ticks: int,
    policy: object | None,
    trajectory_output_path: Path | None = None,
    trajectory_split_id: str = "mind_v3_evaluate",
) -> dict[str, object]:
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=policy,
    )
    return _run_world(
        world=world,
        seed=seed,
        ticks=ticks,
        trajectory_output_path=trajectory_output_path,
        trajectory_split_id=trajectory_split_id,
    )


def _run_world(
    *,
    world: SimulationWorld,
    seed: int,
    ticks: int,
    trajectory_output_path: Path | None = None,
    trajectory_split_id: str = "mind_v3_evaluate",
) -> dict[str, object]:
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    policy_id_counts = Counter(
        str(record.get("policy_id", "unknown"))
        for record in world.trajectory_records
    )
    requested_action_counts = Counter(
        str(record["requested_action"])
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    resolved_action_counts = Counter(
        str(record["resolved_action"])
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
    )
    unsupported_requested_action_count = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
        and record.get("action_valid") is False
    )
    unsupported_resolved_action_count = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
        and record.get("resolution_action_valid") is False
    )
    dominant_action = _dominant_action_summary(requested_action_counts)
    summary = result.summary
    if trajectory_output_path is not None:
        _write_trajectory_records(
            world=world,
            summary=summary,
            output_path=trajectory_output_path,
            seed=seed,
            split_id=trajectory_split_id,
        )
    run = {
        "seed": seed,
        "ticks": ticks,
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "trophic_role_counts_at_end": _json_ready(
            summary.get("trophic_role_counts_at_end", {})
        ),
        "meat_mode_counts_at_end": _json_ready(
            summary.get("meat_mode_counts_at_end", {})
        ),
        "animal_resource_opportunity_by_meat_mode_end": _json_ready(
            summary.get("animal_resource_opportunity_by_meat_mode_end", {})
        ),
        "diet_by_trophic_role_end": _json_ready(
            summary.get("diet_by_trophic_role_end", {})
        ),
        "diet_by_meat_mode_end": _json_ready(
            summary.get("diet_by_meat_mode_end", {})
        ),
        "combat_end": _json_ready(summary.get("combat_end", {})),
        "fresh_kill_end": _json_ready(summary.get("fresh_kill_end", {})),
        "carcass_end": _json_ready(summary.get("carcass_end", {})),
        "outcome_metrics": build_run_outcome_metrics(
            summary=summary,
            trajectory_records=world.trajectory_records,
        ),
        "reproduction_failure_attribution": (
            _reproduction_failure_attribution(summary)
        ),
        "temporal_readiness_attribution": _temporal_readiness_attribution(
            world.trajectory_records
        ),
        "neural_anchor_diagnostics": _neural_anchor_diagnostics(
            world.policy_decision_diagnostics_records
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "unique_requested_actions": len(requested_action_counts),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved_action_count),
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }
    if trajectory_output_path is not None:
        run["trajectory_path"] = str(trajectory_output_path)
    return run


def _write_trajectory_records(
    *,
    world: SimulationWorld,
    summary: Mapping[str, object],
    output_path: Path,
    seed: int,
    split_id: str,
) -> None:
    writer = JsonlTrajectoryWriter(
        output_path,
        source_seeds=[seed],
        split_id=split_id,
        include_policy_decision_diagnostics=True,
        include_policy_update_trace=True,
    )
    writer.begin(
        run_id=world.run_id,
        config=world.config.to_dict(),
        contract=runtime_trajectory.trajectory_contract(world.config.signals),
    )
    try:
        for record in _trajectory_records_with_optional_metadata(world):
            writer.write_record(record)
        writer.finish(summary=dict(summary))
    except Exception:
        writer.abort()
        raise


def _trajectory_records_with_optional_metadata(
    world: SimulationWorld,
) -> list[dict[str, object]]:
    records = list(world.trajectory_records)
    diagnostics = list(world.policy_decision_diagnostics_records)
    update_traces = list(
        getattr(world, "policy_update_trace_records_by_trajectory_record", [])
    )
    diagnostics_aligned = len(diagnostics) == len(records)
    update_traces_aligned = len(update_traces) == len(records)
    enriched_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        enriched = dict(record)
        if (
            "policy_decision_diagnostics" not in enriched
            and diagnostics_aligned
            and isinstance(diagnostics[index], dict)
        ):
            enriched["policy_decision_diagnostics"] = dict(diagnostics[index])
        if (
            "policy_update_trace" not in enriched
            and update_traces_aligned
            and isinstance(update_traces[index], dict)
        ):
            enriched["policy_update_trace"] = dict(update_traces[index])
        enriched_records.append(enriched)
    return enriched_records


def _trajectory_output_path(
    output_dir: Path | None,
    *parts: object,
) -> Path | None:
    if output_dir is None:
        return None
    stem = "-".join(_safe_path_part(part) for part in parts if part is not None)
    return output_dir / f"{stem}.jsonl.gz"


def _safe_path_part(value: object) -> str:
    text = str(value).strip().replace("_", "-")
    safe = [
        character.lower()
        if character.isalnum() or character in {"-", "."}
        else "-"
        for character in text
    ]
    return "".join(safe).strip("-") or "run"


def run_mind_v3_fixture_suite(
    *,
    suite: str,
    fixture_names: list[str] | None = None,
    seeds: list[int],
    ticks: int,
    founder_template: dict[str, object] | list[dict[str, object]] | None,
    neural_artifact: dict[str, object] | None = None,
    planner_distilled_artifact: dict[str, object] | None = None,
    trajectory_output_dir: Path | None = None,
    trajectory_prefix: str = "fixture",
) -> dict[str, object]:
    return run_controlled_fixture_policy_suite(
        suite=suite,
        fixture_names=fixture_names,
        seeds=seeds,
        ticks=ticks,
        learned_policy_factory=(
            lambda fixture_name, seed: _mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=neural_artifact,
                planner_distilled_artifact=planner_distilled_artifact,
            )
        ),
        learned_policy_key="mind_v3",
        learned_policy_name="mind_v3",
        trajectory_output_dir=trajectory_output_dir,
        trajectory_prefix=trajectory_prefix,
    )


def run_controlled_fixture_policy_suite(
    *,
    suite: str,
    fixture_names: list[str] | None = None,
    seeds: list[int],
    ticks: int,
    learned_policy_factory: Callable[[str, int], object | None],
    learned_policy_key: str,
    learned_policy_name: str | None = None,
    trajectory_output_dir: Path | None = None,
    trajectory_prefix: str = "fixture",
) -> dict[str, object]:
    if not learned_policy_key or not learned_policy_key.strip():
        raise ValueError("learned_policy_key must be non-empty")
    fixtures = []
    selected_fixture_names = (
        tuple(fixture_names) if fixture_names else _fixture_names(suite)
    )
    for fixture_name in selected_fixture_names:
        heuristic_runs = [
            _run_fixture_once(
                fixture_name=fixture_name,
                seed=seed,
                ticks=ticks,
                policy=None,
                trajectory_output_path=_trajectory_output_path(
                    trajectory_output_dir,
                    trajectory_prefix,
                    fixture_name,
                    "heuristic",
                    seed,
                    ticks,
                ),
                trajectory_split_id=(
                    f"mind_v3_evaluate_{trajectory_prefix}_{fixture_name}_heuristic"
                ),
            )
            for seed in seeds
        ]
        learned_runs = [
            _run_fixture_once(
                fixture_name=fixture_name,
                seed=seed,
                ticks=ticks,
                policy=learned_policy_factory(fixture_name, seed),
                trajectory_output_path=_trajectory_output_path(
                    trajectory_output_dir,
                    trajectory_prefix,
                    fixture_name,
                    learned_policy_key,
                    seed,
                    ticks,
                ),
                trajectory_split_id=(
                    "mind_v3_evaluate_"
                    f"{trajectory_prefix}_{fixture_name}_{learned_policy_key}"
                ),
            )
            for seed in seeds
        ]
        heuristic_aggregate = _aggregate_runs(heuristic_runs)
        learned_aggregate = _aggregate_runs(learned_runs)
        fixtures.append(
            {
                "fixture": fixture_name,
                "scenario_config": _fixture_scenario_config(fixture_name),
                "comparison": {
                    "heuristic": {
                        "runs": heuristic_runs,
                        "aggregate": heuristic_aggregate,
                    },
                    learned_policy_key: {
                        "runs": learned_runs,
                        "aggregate": learned_aggregate,
                    },
                    "delta": _comparison_delta(
                        heuristic=heuristic_aggregate,
                        mind_v3=learned_aggregate,
                    ),
                },
            }
        )
    return {
        "policy": MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        "suite": suite,
        "fixture_names": list(selected_fixture_names),
        "evaluated_policy_key": learned_policy_key,
        "evaluated_policy_name": learned_policy_name or learned_policy_key,
        "seeds": list(seeds),
        "ticks": int(ticks),
        "fixtures": fixtures,
    }


def mind_v3_fixture_gate_config(
    *,
    suite: str,
    seeds: list[int],
    ticks: int | None,
    min_alive: float,
    min_births: float,
    min_mixed_stable_births: float,
    min_energy_viability: float,
    min_hydration_viability: float,
    min_health_viability: float,
    min_matched_diet_viability: float,
    min_biologically_ready: float,
) -> dict[str, object]:
    if suite not in {"basic"}:
        raise SystemExit(f"unsupported fixture suite: {suite}")
    if not seeds:
        raise SystemExit("--fixture-seeds must include at least one integer seed")
    if ticks is not None and int(ticks) < 1:
        raise SystemExit("--fixture-ticks must be >= 1")
    floors = {
        "min_alive": min_alive,
        "min_births": min_births,
        "min_mixed_stable_births": min_mixed_stable_births,
        "min_energy_viability": min_energy_viability,
        "min_hydration_viability": min_hydration_viability,
        "min_health_viability": min_health_viability,
        "min_matched_diet_viability": min_matched_diet_viability,
        "min_biologically_ready": min_biologically_ready,
    }
    for name, value in floors.items():
        if float(value) < 0.0:
            raise SystemExit(f"--fixture-{name.removeprefix('min_')} must be >= 0")
    return {
        "policy": MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
        "suite": suite,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks) if ticks is not None else None,
        **{name: float(value) for name, value in floors.items()},
    }


def mind_v3_fixture_gate_status(
    *,
    fixture_suite: Mapping[str, object],
    fixture_config: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    per_fixture: dict[str, object] = {}
    evaluated_policy_key = str(fixture_suite.get("evaluated_policy_key", "mind_v3"))
    fixtures = fixture_suite.get("fixtures")
    fixture_items = fixtures if isinstance(fixtures, list) else []
    for raw_fixture in fixture_items:
        if not isinstance(raw_fixture, Mapping):
            continue
        fixture_name = str(raw_fixture.get("fixture", "unknown"))
        comparison = raw_fixture.get("comparison")
        evaluated_policy = (
            comparison.get(evaluated_policy_key)
            if isinstance(comparison, Mapping)
            else None
        )
        aggregate = (
            evaluated_policy.get("aggregate")
            if isinstance(evaluated_policy, Mapping)
            else None
        )
        if not isinstance(aggregate, Mapping):
            blockers.append(
                _fixture_gate_blocker(
                    fixture=fixture_name,
                    reason="fixture_evaluated_policy_aggregate_missing",
                    metric="aggregate",
                    value=0.0,
                    floor=1.0,
                )
            )
            continue
        metrics = _fixture_gate_metrics(aggregate)
        fixture_blockers = _fixture_gate_blockers(
            fixture_name=fixture_name,
            metrics=metrics,
            fixture_config=fixture_config,
        )
        blockers.extend(fixture_blockers)
        per_fixture[fixture_name] = {
            "metrics": metrics,
            "passed": not fixture_blockers,
            "blockers": fixture_blockers,
        }
    if not fixture_items:
        blockers.append(
            _fixture_gate_blocker(
                fixture=None,
                reason="fixture_suite_empty",
                metric="fixture_count",
                value=0.0,
                floor=1.0,
            )
        )
    return {
        "policy": MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
        "fixture_suite_policy": MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        "evaluated_policy_key": evaluated_policy_key,
        "evaluated_policy_name": str(
            fixture_suite.get("evaluated_policy_name", evaluated_policy_key)
        ),
        "suite": str(fixture_config["suite"]),
        "fixture_names": [
            str(name)
            for name in fixture_suite.get("fixture_names", [])
            if isinstance(name, str)
        ],
        "seeds": [int(seed) for seed in list(fixture_config["seeds"])],
        "ticks": (
            fixture_config.get("ticks")
            if fixture_config.get("ticks") is not None
            else fixture_suite.get("ticks")
        ),
        "min_alive": float(fixture_config["min_alive"]),
        "min_births": float(fixture_config["min_births"]),
        "min_mixed_stable_births": float(
            fixture_config["min_mixed_stable_births"]
        ),
        "min_energy_viability": float(fixture_config["min_energy_viability"]),
        "min_hydration_viability": float(
            fixture_config["min_hydration_viability"]
        ),
        "min_health_viability": float(fixture_config["min_health_viability"]),
        "min_matched_diet_viability": float(
            fixture_config["min_matched_diet_viability"]
        ),
        "min_biologically_ready": float(
            fixture_config["min_biologically_ready"]
        ),
        "passed": not blockers,
        "blockers": blockers,
        "per_fixture": dict(sorted(per_fixture.items())),
    }


def _fixture_gate_comparison_delta(
    *,
    linear_gate: Mapping[str, object],
    neural_gate: Mapping[str, object],
) -> dict[str, object]:
    linear_blockers = linear_gate.get("blockers")
    neural_blockers = neural_gate.get("blockers")
    linear_blocker_count = (
        len(linear_blockers) if isinstance(linear_blockers, list) else 0
    )
    neural_blocker_count = (
        len(neural_blockers) if isinstance(neural_blockers, list) else 0
    )
    return {
        "linear_passed": bool(linear_gate.get("passed", False)),
        "neural_passed": bool(neural_gate.get("passed", False)),
        "blocker_count_delta": int(neural_blocker_count - linear_blocker_count),
        "linear_blocker_count": int(linear_blocker_count),
        "neural_blocker_count": int(neural_blocker_count),
    }


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read JSON report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON report {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON report must be an object: {path}")
    return payload


def _planner_distilled_source_reload_actions_match(
    payload: Mapping[str, object] | None,
) -> bool:
    if payload is None:
        return False
    reload_check = payload.get("runtime_reload_check")
    if isinstance(reload_check, Mapping):
        return reload_check.get("actions_match") is True
    return False


def _v97_planner_distilled_promotion_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    comparison = _mapping(report.get("comparison"))
    candidate = _mapping(comparison.get("mind_v3"))
    linear = _mapping(comparison.get("mind_v3_linear"))
    candidate_aggregate = _mapping(candidate.get("aggregate"))
    linear_aggregate = _mapping(linear.get("aggregate"))
    policy = _mapping(report.get("policy"))
    fixture_suite = _mapping(report.get("fixture_suite"))
    linear_fixture_suite = _mapping(report.get("linear_baseline_fixture_suite"))
    fixture_gate = _mapping(report.get("fixture_gate"))
    linear_fixture_gate = _mapping(report.get("linear_baseline_fixture_gate"))
    carrion_fixture = _fixture_report_by_name(fixture_suite, "carrion_only")
    linear_carrion_fixture = _fixture_report_by_name(
        linear_fixture_suite,
        "carrion_only",
    )
    carrion_candidate = _mapping(
        _mapping(_mapping(carrion_fixture.get("comparison")).get("mind_v3")).get(
            "aggregate"
        )
    )
    carrion_linear = _mapping(
        _mapping(
            _mapping(linear_carrion_fixture.get("comparison")).get("mind_v3")
        ).get("aggregate")
    )
    broad_delta = (
        _comparison_delta(
            heuristic=dict(linear_aggregate),
            mind_v3=dict(candidate_aggregate),
        )
        if candidate_aggregate and linear_aggregate
        else {
            "alive_agents_mean": 0.0,
            "births_mean": 0.0,
            "biologically_ready_agents_mean": 0.0,
            "ready_agents_mean": 0.0,
        }
    )
    carrion_delta = (
        _comparison_delta(
            heuristic=dict(carrion_linear),
            mind_v3=dict(carrion_candidate),
        )
        if carrion_candidate and carrion_linear
        else {
            "alive_agents_mean": 0.0,
            "births_mean": 0.0,
            "biologically_ready_agents_mean": 0.0,
            "ready_agents_mean": 0.0,
        }
    )
    per_seed_delta = _per_seed_deltas(
        _list_of_mappings(candidate.get("runs")),
        _list_of_mappings(linear.get("runs")),
    )
    carrion_blocker_count = _fixture_blocker_count(fixture_gate, "carrion_only")
    linear_carrion_blocker_count = _fixture_blocker_count(
        linear_fixture_gate,
        "carrion_only",
    )
    broad_unsupported = int(
        candidate_aggregate.get("unsupported_requested_action_count", 0)
    ) + int(candidate_aggregate.get("unsupported_resolved_action_count", 0))
    carrion_unsupported = int(
        carrion_candidate.get("unsupported_requested_action_count", 0)
    ) + int(carrion_candidate.get("unsupported_resolved_action_count", 0))
    broad_share = _float_metric(
        candidate_aggregate.get("dominant_requested_action_share")
    )
    carrion_share = _float_metric(
        carrion_candidate.get("dominant_requested_action_share")
    )
    max_dominant_share = max(broad_share, carrion_share)
    blockers: list[dict[str, object]] = []
    _append_exact_sequence_blocker(
        blockers,
        name="broad_seeds",
        actual=tuple(_int(run.get("seed")) for run in _list_of_mappings(candidate.get("runs"))),
        expected=V97_BROAD_SEEDS,
    )
    _append_exact_sequence_blocker(
        blockers,
        name="fixture_seeds",
        actual=tuple(int(seed) for seed in fixture_suite.get("seeds", []) or ()),
        expected=V97_CARRION_FIXTURE_SEEDS,
    )
    if int(fixture_suite.get("ticks", 0)) != V97_TICKS:
        blockers.append(
            _v97_blocker(
                reason="fixture_ticks_not_strict_v97",
                metric="fixture_ticks",
                value=_float_metric(fixture_suite.get("ticks")),
                floor=float(V97_TICKS),
            )
        )
    if policy.get("planner_distilled_artifact_schema_version") != (
        MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION
    ):
        blockers.append(
            _v97_blocker(
                reason="planner_distilled_artifact_missing",
                metric="planner_distilled_artifact_schema_version_present",
                value=0.0,
                floor=1.0,
            )
        )
    if policy.get("planner_distilled_source_reload_actions_match") is not True:
        blockers.append(
            _v97_blocker(
                reason="artifact_reload_action_choices_not_verified",
                metric="artifact_reload_actions_match",
                value=0.0,
                floor=1.0,
            )
        )
    if int(candidate_aggregate.get("heuristic_action_source_count", 0)) != 0:
        blockers.append(
            _v97_blocker(
                reason="heuristic_action_source_count_nonzero",
                metric="heuristic_action_source_count",
                value=_float_metric(
                    candidate_aggregate.get("heuristic_action_source_count")
                ),
                floor=0.0,
                comparator="eq",
            )
        )
    if max_dominant_share > V97_MAX_DOMINANT_REQUESTED_ACTION_SHARE:
        blockers.append(
            _v97_blocker(
                reason="dominant_requested_action_share_above_cap",
                metric="max_broad_or_carrion_dominant_requested_action_share",
                value=max_dominant_share,
                floor=V97_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
                comparator="le",
            )
        )
    if broad_unsupported + carrion_unsupported != 0:
        blockers.append(
            _v97_blocker(
                reason="unsupported_requested_or_resolved_action_nonzero",
                metric="unsupported_action_count",
                value=float(broad_unsupported + carrion_unsupported),
                floor=0.0,
                comparator="eq",
            )
        )
    _append_nonnegative_blocker(
        blockers,
        reason="broad_alive_mean_regression_vs_linear",
        metric="broad_alive_mean_delta_vs_linear",
        value=_float_metric(broad_delta.get("alive_agents_mean")),
    )
    _append_nonnegative_blocker(
        blockers,
        reason="broad_births_mean_regression_vs_linear",
        metric="broad_births_mean_delta_vs_linear",
        value=_float_metric(broad_delta.get("births_mean")),
    )
    for item in per_seed_delta:
        seed = int(item["seed"])
        _append_nonnegative_blocker(
            blockers,
            reason=f"broad_seed_{seed}_alive_regression_vs_linear",
            metric="broad_seed_alive_delta_vs_linear",
            value=_float_metric(item.get("alive_delta_vs_linear")),
        )
        _append_nonnegative_blocker(
            blockers,
            reason=f"broad_seed_{seed}_births_regression_vs_linear",
            metric="broad_seed_births_delta_vs_linear",
            value=_float_metric(item.get("births_delta_vs_linear")),
        )
    _append_nonnegative_blocker(
        blockers,
        reason="carrion_alive_mean_regression_vs_linear",
        metric="carrion_only_alive_mean_delta_vs_linear",
        value=_float_metric(carrion_delta.get("alive_agents_mean")),
    )
    _append_nonnegative_blocker(
        blockers,
        reason="carrion_births_mean_regression_vs_linear",
        metric="carrion_only_births_mean_delta_vs_linear",
        value=_float_metric(carrion_delta.get("births_mean")),
    )
    blocker_delta = carrion_blocker_count - linear_carrion_blocker_count
    if blocker_delta > 0:
        blockers.append(
            _v97_blocker(
                reason="carrion_only_fixture_blocker_regression_vs_linear",
                metric="carrion_only_blocker_count_delta_vs_linear",
                value=float(blocker_delta),
                floor=0.0,
                comparator="le",
            )
        )
    return {
        "policy": MIND_V3_V97_PLANNER_DISTILLED_PROMOTION_POLICY,
        "promotion_candidate_passed": not blockers,
        "runtime_policy_status": (
            "strict_runtime_policy_pass"
            if not blockers
            else "rejected_no_promotion"
        ),
        "criteria": {
            "broad_seeds": list(V97_BROAD_SEEDS),
            "fixture": "carrion_only",
            "fixture_seeds": list(V97_CARRION_FIXTURE_SEEDS),
            "ticks": V97_TICKS,
            "max_dominant_requested_action_share": (
                V97_MAX_DOMINANT_REQUESTED_ACTION_SHARE
            ),
        },
        "metrics": {
            "heuristic_action_source_count": int(
                candidate_aggregate.get("heuristic_action_source_count", 0)
            ),
            "broad_dominant_requested_action_share": _round(broad_share),
            "carrion_only_dominant_requested_action_share": _round(carrion_share),
            "max_broad_or_carrion_dominant_requested_action_share": _round(
                max_dominant_share
            ),
            "broad_alive_mean_delta_vs_linear": _float_metric(
                broad_delta.get("alive_agents_mean")
            ),
            "broad_births_mean_delta_vs_linear": _float_metric(
                broad_delta.get("births_mean")
            ),
            "carrion_only_alive_mean_delta_vs_linear": _float_metric(
                carrion_delta.get("alive_agents_mean")
            ),
            "carrion_only_births_mean_delta_vs_linear": _float_metric(
                carrion_delta.get("births_mean")
            ),
            "carrion_only_blocker_count": carrion_blocker_count,
            "linear_carrion_only_blocker_count": linear_carrion_blocker_count,
            "carrion_only_blocker_count_delta_vs_linear": blocker_delta,
            "unsupported_action_count": broad_unsupported + carrion_unsupported,
            "artifact_reload_actions_match": bool(
                policy.get("planner_distilled_source_reload_actions_match")
            ),
        },
        "per_seed_delta_vs_linear": per_seed_delta,
        "blockers": blockers,
    }


def _attach_v97_promotion_ledger_metrics(report: dict[str, object]) -> None:
    promotion = _mapping(report.get("v97_planner_distilled_promotion"))
    metrics = _mapping(promotion.get("metrics"))
    blockers = promotion.get("blockers")
    blocker_count = len(blockers) if isinstance(blockers, list) else 0
    report["promotion_candidate_passed"] = bool(
        promotion.get("promotion_candidate_passed")
    )
    report["promotion_blocker_count"] = blocker_count
    report["blocker_count"] = blocker_count
    report["candidate_alive_delta_vs_linear"] = metrics.get(
        "broad_alive_mean_delta_vs_linear"
    )
    report["candidate_births_delta_vs_linear"] = metrics.get(
        "broad_births_mean_delta_vs_linear"
    )
    report["candidate_dominant_requested_action_share"] = metrics.get(
        "max_broad_or_carrion_dominant_requested_action_share"
    )
    report["candidate_heuristic_action_source_count"] = metrics.get(
        "heuristic_action_source_count"
    )
    per_seed = _list_of_mappings(promotion.get("per_seed_delta_vs_linear"))
    alive_deltas = [
        _float_metric(item.get("alive_delta_vs_linear")) for item in per_seed
    ]
    birth_deltas = [
        _float_metric(item.get("births_delta_vs_linear")) for item in per_seed
    ]
    if alive_deltas:
        report["candidate_min_seed_alive_delta_vs_linear"] = min(alive_deltas)
    if birth_deltas:
        report["candidate_min_seed_births_delta_vs_linear"] = min(birth_deltas)


def _fixture_report_by_name(
    fixture_suite: Mapping[str, object],
    fixture_name: str,
) -> Mapping[str, object]:
    for item in _list_of_mappings(fixture_suite.get("fixtures")):
        if str(item.get("fixture", "")) == fixture_name:
            return item
    return {}


def _fixture_blocker_count(
    fixture_gate: Mapping[str, object],
    fixture_name: str,
) -> int:
    blockers = fixture_gate.get("blockers")
    if not isinstance(blockers, list):
        return 0
    return sum(
        1
        for item in blockers
        if isinstance(item, Mapping) and item.get("fixture") == fixture_name
    )


def _per_seed_deltas(
    candidate_runs: Sequence[Mapping[str, object]],
    linear_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    linear_by_seed = {_int(run.get("seed")): run for run in linear_runs}
    deltas = []
    for run in sorted(candidate_runs, key=lambda item: _int(item.get("seed"))):
        seed = _int(run.get("seed"))
        linear = _mapping(linear_by_seed.get(seed))
        deltas.append(
            {
                "seed": seed,
                "alive_delta_vs_linear": _round(
                    _float_metric(run.get("alive_agents"))
                    - _float_metric(linear.get("alive_agents"))
                ),
                "births_delta_vs_linear": _round(
                    _float_metric(run.get("births"))
                    - _float_metric(linear.get("births"))
                ),
                "candidate_alive": _int(run.get("alive_agents")),
                "linear_alive": _int(linear.get("alive_agents")),
                "candidate_births": _int(run.get("births")),
                "linear_births": _int(linear.get("births")),
            }
        )
    return deltas


def _append_exact_sequence_blocker(
    blockers: list[dict[str, object]],
    *,
    name: str,
    actual: Sequence[int],
    expected: Sequence[int],
) -> None:
    if tuple(actual) == tuple(expected):
        return
    blockers.append(
        {
            "reason": f"{name}_not_strict_v97",
            "metric": name,
            "value": list(actual),
            "floor": list(expected),
            "comparator": "eq",
        }
    )


def _append_nonnegative_blocker(
    blockers: list[dict[str, object]],
    *,
    reason: str,
    metric: str,
    value: float,
) -> None:
    if float(value) >= 0.0:
        return
    blockers.append(
        _v97_blocker(
            reason=reason,
            metric=metric,
            value=float(value),
            floor=0.0,
        )
    )


def _v97_blocker(
    *,
    reason: str,
    metric: str,
    value: float,
    floor: float,
    comparator: str = "ge",
) -> dict[str, object]:
    return {
        "reason": reason,
        "metric": metric,
        "value": _round(value),
        "floor": _round(floor),
        "comparator": comparator,
    }


def _fixture_gate_metrics(aggregate: Mapping[str, object]) -> dict[str, float]:
    attribution = aggregate.get("reproduction_failure_attribution")
    if not isinstance(attribution, Mapping):
        attribution = {}
    viability = attribution.get("terminal_viability_shares_mean")
    if not isinstance(viability, Mapping):
        viability = {}
    return {
        "alive_agents_mean": _float_metric(aggregate.get("alive_agents_mean")),
        "births_mean": _float_metric(aggregate.get("births_mean")),
        "energy_viability_share_mean": _float_metric(viability.get("energy")),
        "energy_requirement_satisfaction_mean": _float_metric(
            aggregate.get("terminal_energy_requirement_satisfaction_mean")
        ),
        "balanced_reproduction_readiness_mean": _float_metric(
            aggregate.get("terminal_balanced_reproduction_readiness_mean")
        ),
        "energy_hydration_balance_mean": _float_metric(
            aggregate.get("terminal_energy_hydration_balance_mean")
        ),
        "energy_hydration_gap_abs_mean": _float_metric(
            aggregate.get("terminal_energy_hydration_gap_abs_mean")
        ),
        "hydration_viability_share_mean": _float_metric(
            viability.get("hydration")
        ),
        "health_viability_share_mean": _float_metric(viability.get("health")),
        "matched_diet_viability_share_mean": _float_metric(
            viability.get("matched_diet")
        ),
        "biologically_ready_agents_mean": _float_metric(
            attribution.get("biologically_ready_agents_mean")
        ),
    }


def _fixture_gate_blockers(
    *,
    fixture_name: str,
    metrics: Mapping[str, float],
    fixture_config: Mapping[str, object],
) -> list[dict[str, object]]:
    checks = [
        ("fixture_alive_floor", "alive_agents_mean", "min_alive"),
        ("fixture_birth_floor", "births_mean", "min_births"),
        (
            "fixture_energy_viability_floor",
            "energy_viability_share_mean",
            "min_energy_viability",
        ),
        (
            "fixture_hydration_viability_floor",
            "hydration_viability_share_mean",
            "min_hydration_viability",
        ),
        (
            "fixture_health_viability_floor",
            "health_viability_share_mean",
            "min_health_viability",
        ),
        (
            "fixture_matched_diet_viability_floor",
            "matched_diet_viability_share_mean",
            "min_matched_diet_viability",
        ),
        (
            "fixture_biological_readiness_floor",
            "biologically_ready_agents_mean",
            "min_biologically_ready",
        ),
    ]
    blockers = []
    for reason, metric_name, floor_name in checks:
        value = float(metrics[metric_name])
        floor = float(fixture_config[floor_name])
        if value < floor:
            blockers.append(
                _fixture_gate_blocker(
                    fixture=fixture_name,
                    reason=reason,
                    metric=metric_name,
                    value=value,
                    floor=floor,
                )
            )
    mixed_stable_floor = float(fixture_config["min_mixed_stable_births"])
    if fixture_name == "mixed_stable" and (
        float(metrics["births_mean"]) < mixed_stable_floor
    ):
        blockers.append(
            _fixture_gate_blocker(
                fixture=fixture_name,
                reason="fixture_mixed_stable_birth_floor",
                metric="births_mean",
                value=float(metrics["births_mean"]),
                floor=mixed_stable_floor,
            )
        )
    return blockers


def _fixture_gate_blocker(
    *,
    fixture: str | None,
    reason: str,
    metric: str,
    value: float,
    floor: float,
) -> dict[str, object]:
    return {
        "fixture": fixture,
        "reason": reason,
        "metric": metric,
        "value": _round(value),
        "floor": _round(floor),
    }


def _float_metric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _run_fixture_once(
    *,
    fixture_name: str,
    seed: int,
    ticks: int,
    policy: object | None,
    trajectory_output_path: Path | None = None,
    trajectory_split_id: str = "mind_v3_evaluate_fixture",
) -> dict[str, object]:
    world = _fixture_world(
        fixture_name=fixture_name,
        seed=seed,
        ticks=ticks,
        policy=policy,
    )
    run = _run_world(
        world=world,
        seed=seed,
        ticks=ticks,
        trajectory_output_path=trajectory_output_path,
        trajectory_split_id=trajectory_split_id,
    )
    run["fixture"] = fixture_name
    run["evaluation_context"] = "controlled_ecology_fixture"
    return run


def _fixture_names(suite: str) -> tuple[str, ...]:
    if suite == "basic":
        return CONTROLLED_FIXTURE_NAMES
    raise SystemExit(f"unsupported fixture suite: {suite}")


def _parse_fixture_names(raw: str | None, *, suite: str) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    supported = set(_fixture_names(suite))
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


def _fixture_world(
    *,
    fixture_name: str,
    seed: int,
    ticks: int,
    policy: object | None,
) -> SimulationWorld:
    config = WorldConfig(
        seed=seed,
        width=18,
        height=12,
        max_ticks=ticks,
        initial_agents=0,
        max_agents=180,
        resources=_fixture_resources(fixture_name),
        climate=ClimateConfig(season_length=60),
    )
    world = SimulationWorld(config, policy=policy)
    genomes = _archetype_genomes()
    scenario = _fixture_scenario_config(fixture_name)
    _configure_uniform_arena(
        world,
        terrain=str(scenario["terrain"]),
        food=float(scenario["food"]),
        vegetation=float(scenario["vegetation"]),
        shelter=float(scenario["shelter"]),
        recovery_debt=float(scenario["recovery_debt"]),
        fertility=float(scenario["fertility"]),
        moisture=float(scenario["moisture"]),
        heat=float(scenario["heat"]),
    )
    _set_water_border(world)
    _seed_archetypes(
        world,
        counts=dict(scenario["initial_archetype_counts"]),
        genomes=genomes,
    )
    for deposit in scenario["carcass_deposits"]:
        x = int(deposit["x"])
        y = int(deposit["y"])
        world._deposit_carcass(
            world.grid[y][x],
            x=x,
            y=y,
            energy=float(deposit["energy"]),
            source_species=deposit.get("source_species"),
            source_agent_id=None,
            cause="controlled_fixture",
            killer_id=None,
        )
    world.reset_derived_caches()
    return world


def _fixture_resources(fixture_name: str) -> ResourceRegrowthConfig:
    if fixture_name == "plant_only":
        return ResourceRegrowthConfig(
            plain_food_rate=0.038,
            forest_food_rate=0.062,
            wetland_food_rate=0.052,
            rocky_food_rate=0.006,
        )
    if fixture_name == "carrion_only":
        return ResourceRegrowthConfig(
            plain_food_rate=0.003,
            forest_food_rate=0.004,
            wetland_food_rate=0.004,
            rocky_food_rate=0.002,
        )
    if fixture_name == "prey_rich":
        return ResourceRegrowthConfig(
            plain_food_rate=0.044,
            forest_food_rate=0.026,
            wetland_food_rate=0.026,
            rocky_food_rate=0.008,
        )
    if fixture_name == "mixed_stable":
        return ResourceRegrowthConfig(
            plain_food_rate=0.024,
            forest_food_rate=0.038,
            wetland_food_rate=0.034,
            rocky_food_rate=0.01,
        )
    raise SystemExit(f"unsupported fixture: {fixture_name}")


def _fixture_scenario_config(fixture_name: str) -> dict[str, object]:
    scenarios: dict[str, dict[str, object]] = {
        "plant_only": {
            "terrain": "forest",
            "food": 0.92,
            "vegetation": 0.88,
            "shelter": 0.42,
            "recovery_debt": 0.02,
            "fertility": 0.88,
            "moisture": 0.82,
            "heat": 0.28,
            "initial_archetype_counts": {
                "herbivore": 4,
                "hunter": 2,
                "scavenger": 2,
                "omnivore": 4,
            },
            "carcass_deposits": [],
        },
        "carrion_only": {
            "terrain": "rocky",
            "food": 0.02,
            "vegetation": 0.06,
            "shelter": 0.18,
            "recovery_debt": 0.08,
            "fertility": 0.16,
            "moisture": 0.58,
            "heat": 0.42,
            "initial_archetype_counts": {
                "herbivore": 2,
                "hunter": 2,
                "scavenger": 5,
                "omnivore": 3,
            },
            "carcass_deposits": [
                {"x": 3, "y": 7, "energy": 1.1, "source_species": 101},
                {"x": 4, "y": 8, "energy": 1.1, "source_species": 101},
                {"x": 5, "y": 7, "energy": 0.9, "source_species": 101},
                {"x": 14, "y": 8, "energy": 0.7, "source_species": 101},
            ],
        },
        "prey_rich": {
            "terrain": "plain",
            "food": 0.58,
            "vegetation": 0.46,
            "shelter": 0.16,
            "recovery_debt": 0.03,
            "fertility": 0.62,
            "moisture": 0.64,
            "heat": 0.34,
            "initial_archetype_counts": {
                "herbivore": 8,
                "hunter": 4,
                "scavenger": 2,
                "omnivore": 2,
            },
            "carcass_deposits": [],
        },
        "mixed_stable": {
            "terrain": "wetland",
            "food": 0.5,
            "vegetation": 0.58,
            "shelter": 0.3,
            "recovery_debt": 0.02,
            "fertility": 0.68,
            "moisture": 0.86,
            "heat": 0.24,
            "initial_archetype_counts": {
                "herbivore": 3,
                "hunter": 3,
                "scavenger": 3,
                "omnivore": 3,
            },
            "carcass_deposits": [
                {"x": 3, "y": 8, "energy": 0.65, "source_species": 101},
                {"x": 14, "y": 8, "energy": 0.65, "source_species": 101},
            ],
        },
    }
    try:
        return _json_ready(scenarios[fixture_name])
    except KeyError as exc:
        raise SystemExit(f"unsupported fixture: {fixture_name}") from exc


def _configure_uniform_arena(
    world: SimulationWorld,
    *,
    terrain: str,
    food: float,
    vegetation: float,
    shelter: float,
    recovery_debt: float,
    fertility: float,
    moisture: float,
    heat: float,
) -> None:
    for row in world.grid:
        for tile in row:
            tile.terrain = terrain
            tile.food = food
            tile.water = 1.0 if terrain == "water" else 0.0
            tile.fertility = fertility
            tile.moisture = moisture
            tile.heat = heat
            tile.vegetation = vegetation
            tile.shelter = shelter
            tile.recovery_debt = recovery_debt
            tile.fresh_kill_deposits = []
            tile.carcass_deposits = []
            tile.occupant_id = None
    world.reset_derived_caches()


def _set_water_border(world: SimulationWorld) -> None:
    for x in range(world.config.width):
        _set_water_tile(world, x=x, y=0)


def _set_water_tile(world: SimulationWorld, *, x: int, y: int) -> None:
    tile = world.grid[y][x]
    tile.terrain = "water"
    tile.food = 0.0
    tile.water = 1.0
    tile.fertility = 0.0
    tile.moisture = 1.0
    tile.heat = 0.28
    tile.vegetation = 0.0
    tile.shelter = 0.0
    tile.recovery_debt = 0.0
    tile.fresh_kill_deposits = []
    tile.carcass_deposits = []
    tile.occupant_id = None


def _archetype_genomes() -> dict[str, Genome]:
    base = Genome.sample_initial(Random(13))
    return {
        "herbivore": replace(
            base,
            attack_power=0.44,
            attack_cost_multiplier=1.24,
            defense_rating=0.76,
            meat_efficiency=0.24,
            food_efficiency=1.58,
            water_efficiency=1.22,
            plant_bias=1.72,
            carrion_bias=0.22,
            live_prey_bias=0.2,
            reproduction_threshold=0.68,
            mutation_scale=0.03,
        ),
        "hunter": replace(
            base,
            attack_power=1.6,
            attack_cost_multiplier=0.76,
            defense_rating=1.18,
            meat_efficiency=1.56,
            food_efficiency=0.5,
            water_efficiency=0.94,
            plant_bias=0.46,
            carrion_bias=0.74,
            live_prey_bias=1.58,
            reproduction_threshold=0.72,
            mutation_scale=0.03,
        ),
        "scavenger": replace(
            base,
            attack_power=0.82,
            attack_cost_multiplier=0.96,
            defense_rating=0.98,
            meat_efficiency=1.5,
            food_efficiency=0.56,
            water_efficiency=1.0,
            plant_bias=0.48,
            carrion_bias=1.6,
            live_prey_bias=0.38,
            reproduction_threshold=0.72,
            mutation_scale=0.03,
        ),
        "omnivore": replace(
            base,
            attack_power=1.0,
            attack_cost_multiplier=0.92,
            defense_rating=0.94,
            meat_efficiency=1.02,
            food_efficiency=1.12,
            water_efficiency=1.06,
            plant_bias=1.06,
            carrion_bias=0.94,
            live_prey_bias=0.9,
            reproduction_threshold=0.69,
            mutation_scale=0.03,
        ),
    }


def _seed_archetypes(
    world: SimulationWorld,
    *,
    counts: dict[str, int],
    genomes: dict[str, Genome],
) -> None:
    placements = {
        "herbivore": (2, 2),
        "hunter": (world.config.width - 5, 2),
        "scavenger": (2, world.config.height - 5),
        "omnivore": (world.config.width - 5, world.config.height - 5),
    }
    lineage_ids = {
        "herbivore": 101,
        "hunter": 202,
        "scavenger": 303,
        "omnivore": 404,
    }
    for archetype, count in sorted(counts.items()):
        start_x, start_y = placements[archetype]
        for offset in range(int(count)):
            x = start_x + (offset % 3)
            y = start_y + (offset // 3)
            _add_fixture_agent(
                world,
                genome=genomes[archetype],
                x=min(max(x, 0), world.config.width - 1),
                y=min(max(y, 1), world.config.height - 1),
                lineage_id=lineage_ids[archetype],
            )
    world.current_species_map = {
        agent.agent_id: agent.lineage_id for agent in world.alive_agents()
    }
    world.agent_last_species_map = world.current_species_map.copy()
    world.reset_derived_caches()


def _add_fixture_agent(
    world: SimulationWorld,
    *,
    genome: Genome,
    x: int,
    y: int,
    lineage_id: int,
    energy_ratio: float = 0.9,
    hydration_ratio: float = 0.9,
    health_ratio: float = 0.96,
    age: int = 32,
) -> int:
    agent_id = world.next_agent_id
    reproductive_state = runtime_reproduction.founder_reproductive_state(lineage_id)
    agent = Agent(
        agent_id=agent_id,
        parent_id=None,
        lineage_id=lineage_id,
        birth_tick=0,
        death_tick=None,
        x=x,
        y=y,
        energy=genome.max_energy * energy_ratio,
        hydration=genome.max_hydration * hydration_ratio,
        health=genome.max_health * health_ratio,
        max_health=genome.max_health,
        injury_load=0.0,
        age=age,
        alive=True,
        last_reproduction_tick=-10_000,
        last_damage_source="none",
        recent_plant_energy=0.0,
        recent_fresh_kill_energy=0.0,
        recent_carcass_energy=0.0,
        genome_vector=genome_vector(genome),
        genome=genome,
        reproductive_group_id=reproductive_state.group_id,
        reproductive_stage=reproductive_state.stage,
        reproductive_expression=reproductive_state.expression,
        mind_inheritance_metadata=world._founder_mind_metadata(
            agent_id,
            genome=genome,
        ),
    )
    world._place_agent(agent)
    runtime_reproduction.register_founder_group(
        world.reproductive_groups,
        agent,
        tick=0,
    )
    world.next_agent_id += 1
    return agent_id


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    alive_values = [int(run["alive_agents"]) for run in runs]
    birth_values = [int(run["births"]) for run in runs]
    death_values = [int(run["deaths"]) for run in runs]
    action_source_counts: Counter[str] = Counter()
    policy_id_counts: Counter[str] = Counter()
    requested_action_counts: Counter[str] = Counter()
    resolved_action_counts: Counter[str] = Counter()
    unsupported_requested_action_count = 0
    unsupported_resolved_action_count = 0
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
        requested_action_counts.update(
            {
                str(action): int(count)
                for action, count in dict(
                    run.get("requested_action_counts", {})
                ).items()
            }
        )
        resolved_action_counts.update(
            {
                str(action): int(count)
                for action, count in dict(
                    run.get("resolved_action_counts", {})
                ).items()
            }
        )
        unsupported_requested_action_count += int(
            run.get("unsupported_requested_action_count", 0)
        )
        unsupported_resolved_action_count += int(
            run.get("unsupported_resolved_action_count", 0)
        )
    dominant_action = _dominant_action_summary(requested_action_counts)
    reproduction_attribution = _aggregate_reproduction_failure_attribution(runs)
    temporal_readiness = _aggregate_temporal_readiness_attribution(runs)
    neural_anchor_diagnostics = _aggregate_neural_anchor_diagnostics(runs)
    terminal_viability = reproduction_attribution[
        "terminal_viability_shares_mean"
    ]
    return {
        "run_count": len(runs),
        "alive_agents_mean": _mean(alive_values),
        "births_mean": _mean(birth_values),
        "deaths_mean": _mean(death_values),
        "trajectory_record_count": sum(
            int(run["trajectory_record_count"]) for run in runs
        ),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_source_counts
        ),
        "unique_requested_actions_mean": _mean(
            [int(run.get("unique_requested_actions", 0)) for run in runs]
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved_action_count),
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
        "reproduction_ready_agents_mean": reproduction_attribution[
            "ready_agents_mean"
        ],
        "ready_agents_mean": reproduction_attribution["ready_agents_mean"],
        "biologically_reproduction_ready_agents_mean": (
            reproduction_attribution["biologically_ready_agents_mean"]
        ),
        "biologically_ready_agents_mean": reproduction_attribution[
            "biologically_ready_agents_mean"
        ],
        "terminal_ready_share_mean": reproduction_attribution[
            "terminal_ready_share_mean"
        ],
        "terminal_biologically_ready_share_mean": reproduction_attribution[
            "terminal_biologically_ready_share_mean"
        ],
        "terminal_energy_requirement_satisfaction_mean": (
            reproduction_attribution[
                "terminal_energy_requirement_satisfaction_mean"
            ]
        ),
        "terminal_balanced_reproduction_readiness_mean": (
            reproduction_attribution[
                "terminal_balanced_reproduction_readiness_mean"
            ]
        ),
        "terminal_energy_hydration_balance_mean": (
            reproduction_attribution["terminal_energy_hydration_balance_mean"]
        ),
        "terminal_energy_hydration_gap_abs_mean": (
            reproduction_attribution["terminal_energy_hydration_gap_abs_mean"]
        ),
        "terminal_energy_shortfall_share_mean": reproduction_attribution[
            "terminal_energy_shortfall_share_mean"
        ],
        "terminal_energy_shortfall_agents_mean": reproduction_attribution[
            "terminal_energy_shortfall_agents_mean"
        ],
        "terminal_energy_total_mean": reproduction_attribution[
            "terminal_energy_total_mean"
        ],
        "terminal_energy_required_total_mean": reproduction_attribution[
            "terminal_energy_required_total_mean"
        ],
        "terminal_energy_gap_total_mean": reproduction_attribution[
            "terminal_energy_gap_total_mean"
        ],
        "terminal_energy_viability_share_mean": float(
            dict(terminal_viability).get("energy", 0.0)
        ),
        "terminal_hydration_viability_share_mean": float(
            dict(terminal_viability).get("hydration", 0.0)
        ),
        "terminal_health_viability_share_mean": float(
            dict(terminal_viability).get("health", 0.0)
        ),
        "terminal_matched_diet_viability_share_mean": float(
            dict(terminal_viability).get("matched_diet", 0.0)
        ),
        "reproduction_failure_attribution": reproduction_attribution,
        "temporal_readiness_attribution": temporal_readiness,
        "neural_anchor_diagnostics": neural_anchor_diagnostics,
        "outcome_metrics": aggregate_run_outcome_metrics(runs),
        "primary_temporal_readiness_blocker": temporal_readiness[
            "primary_temporal_readiness_blocker"
        ],
    }


def _neural_anchor_diagnostics(
    decision_diagnostics: list[dict[str, object] | None],
) -> dict[str, object]:
    stats = _empty_neural_anchor_diagnostic_stats()
    stats["total_decision_count"] = len(decision_diagnostics)
    for diagnostic in decision_diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        if not isinstance(diagnostic.get("neural_linear_anchor_policy"), str):
            continue
        _update_neural_anchor_diagnostic_stats(stats, diagnostic)
    return _finalize_neural_anchor_diagnostic_stats(stats, run_count=1)


def _aggregate_neural_anchor_diagnostics(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    stats = _empty_neural_anchor_diagnostic_stats()
    for run in runs:
        diagnostics = run.get("neural_anchor_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        stats["total_decision_count"] = int(stats["total_decision_count"]) + int(
            diagnostics.get("total_decision_count", 0)
        )
        stats["decision_count"] = int(stats["decision_count"]) + int(
            diagnostics.get("decision_count", 0)
        )
        for target_key in (
            "residual_applied_count",
            "residual_shadowed_count",
            "changed_linear_action_count",
            "matched_linear_action_count",
            "matched_neural_top_action_count",
        ):
            stats[target_key] = int(stats[target_key]) + int(
                diagnostics.get(target_key, 0)
            )
        for target_key in (
            "neural_score_margin_total",
            "linear_anchor_score_margin_total",
            "anchored_score_margin_total",
            "changed_neural_score_margin_total",
            "changed_linear_anchor_score_margin_total",
        ):
            mean_key = target_key.removesuffix("_total") + "_mean"
            denominator_key = (
                "changed_linear_action_count"
                if target_key.startswith("changed_")
                else "decision_count"
            )
            stats[target_key] = float(stats[target_key]) + (
                float(diagnostics.get(mean_key, 0.0))
                * float(diagnostics.get(denominator_key, 0))
            )
        for source_key, target_key in (
            ("neural_top_action_counts", "neural_top_action_counts"),
            ("linear_anchor_action_counts", "linear_anchor_action_counts"),
            ("anchored_action_counts", "anchored_action_counts"),
            (
                "linear_to_anchored_action_counts",
                "linear_to_anchored_action_counts",
            ),
            (
                "neural_to_anchored_action_counts",
                "neural_to_anchored_action_counts",
            ),
            ("shadow_reason_counts", "shadow_reason_counts"),
        ):
            counter = stats[target_key]
            if isinstance(counter, Counter):
                counter.update(_int_counter(diagnostics.get(source_key, {})))
    return _finalize_neural_anchor_diagnostic_stats(
        stats,
        run_count=len(runs),
    )


def _empty_neural_anchor_diagnostic_stats() -> dict[str, object]:
    return {
        "total_decision_count": 0,
        "decision_count": 0,
        "residual_applied_count": 0,
        "residual_shadowed_count": 0,
        "changed_linear_action_count": 0,
        "matched_linear_action_count": 0,
        "matched_neural_top_action_count": 0,
        "neural_top_action_counts": Counter(),
        "linear_anchor_action_counts": Counter(),
        "anchored_action_counts": Counter(),
        "linear_to_anchored_action_counts": Counter(),
        "neural_to_anchored_action_counts": Counter(),
        "neural_score_margin_total": 0.0,
        "linear_anchor_score_margin_total": 0.0,
        "anchored_score_margin_total": 0.0,
        "changed_neural_score_margin_total": 0.0,
        "changed_linear_anchor_score_margin_total": 0.0,
        "shadow_reason_counts": Counter(),
    }


def _update_neural_anchor_diagnostic_stats(
    stats: dict[str, object],
    diagnostic: Mapping[str, object],
) -> None:
    stats["decision_count"] = int(stats["decision_count"]) + 1
    if diagnostic.get("neural_residual_applied") is True:
        stats["residual_applied_count"] = int(stats["residual_applied_count"]) + 1
    if diagnostic.get("neural_residual_shadowed") is True:
        stats["residual_shadowed_count"] = int(stats["residual_shadowed_count"]) + 1
    if diagnostic.get("neural_residual_changed_linear_action") is True:
        stats["changed_linear_action_count"] = (
            int(stats["changed_linear_action_count"]) + 1
        )
    anchored_action = _diagnostic_action(diagnostic, "anchored_action")
    linear_action = _diagnostic_action(diagnostic, "linear_anchor_action")
    neural_action = _diagnostic_action(diagnostic, "neural_top_action")
    if anchored_action is not None and anchored_action == linear_action:
        stats["matched_linear_action_count"] = (
            int(stats["matched_linear_action_count"]) + 1
        )
    if anchored_action is not None and anchored_action == neural_action:
        stats["matched_neural_top_action_count"] = (
            int(stats["matched_neural_top_action_count"]) + 1
        )
    changed_linear_action = (
        anchored_action is not None
        and linear_action is not None
        and anchored_action != linear_action
    )
    neural_margin = _diagnostic_float(diagnostic, "neural_score_margin")
    linear_margin = _diagnostic_float(diagnostic, "linear_anchor_score_margin")
    anchored_margin = _diagnostic_float(diagnostic, "anchored_score_margin")
    stats["neural_score_margin_total"] = (
        float(stats["neural_score_margin_total"]) + neural_margin
    )
    stats["linear_anchor_score_margin_total"] = (
        float(stats["linear_anchor_score_margin_total"]) + linear_margin
    )
    stats["anchored_score_margin_total"] = (
        float(stats["anchored_score_margin_total"]) + anchored_margin
    )
    if changed_linear_action:
        stats["changed_neural_score_margin_total"] = (
            float(stats["changed_neural_score_margin_total"]) + neural_margin
        )
        stats["changed_linear_anchor_score_margin_total"] = (
            float(stats["changed_linear_anchor_score_margin_total"])
            + linear_margin
        )
    for action, key in (
        (neural_action, "neural_top_action_counts"),
        (linear_action, "linear_anchor_action_counts"),
        (anchored_action, "anchored_action_counts"),
    ):
        if action is None:
            continue
        counter = stats[key]
        if isinstance(counter, Counter):
            counter.update([action])
    if linear_action is not None and anchored_action is not None:
        counter = stats["linear_to_anchored_action_counts"]
        if isinstance(counter, Counter):
            counter.update([f"{linear_action}->{anchored_action}"])
    if neural_action is not None and anchored_action is not None:
        counter = stats["neural_to_anchored_action_counts"]
        if isinstance(counter, Counter):
            counter.update([f"{neural_action}->{anchored_action}"])
    shadow_reason = diagnostic.get("neural_residual_shadow_reason")
    if isinstance(shadow_reason, str) and shadow_reason:
        counter = stats["shadow_reason_counts"]
        if isinstance(counter, Counter):
            counter.update([shadow_reason])


def _finalize_neural_anchor_diagnostic_stats(
    stats: Mapping[str, object],
    *,
    run_count: int,
) -> dict[str, object]:
    decision_count = int(stats.get("decision_count", 0))
    residual_applied_count = int(stats.get("residual_applied_count", 0))
    residual_shadowed_count = int(stats.get("residual_shadowed_count", 0))
    changed_linear_action_count = int(
        stats.get("changed_linear_action_count", 0)
    )
    matched_linear_action_count = int(
        stats.get("matched_linear_action_count", 0)
    )
    matched_neural_top_action_count = int(
        stats.get("matched_neural_top_action_count", 0)
    )
    return {
        "policy": MIND_V3_NEURAL_ANCHOR_DIAGNOSTICS_POLICY,
        "run_count": int(run_count),
        "total_decision_count": int(stats.get("total_decision_count", 0)),
        "decision_count": decision_count,
        "residual_applied_count": residual_applied_count,
        "residual_applied_share": _share(residual_applied_count, decision_count),
        "residual_shadowed_count": residual_shadowed_count,
        "residual_shadowed_share": _share(residual_shadowed_count, decision_count),
        "changed_linear_action_count": changed_linear_action_count,
        "changed_linear_action_share": _share(
            changed_linear_action_count,
            decision_count,
        ),
        "matched_linear_action_count": matched_linear_action_count,
        "matched_linear_action_share": _share(
            matched_linear_action_count,
            decision_count,
        ),
        "matched_neural_top_action_count": matched_neural_top_action_count,
        "matched_neural_top_action_share": _share(
            matched_neural_top_action_count,
            decision_count,
        ),
        "neural_score_margin_mean": _safe_mean_total(
            stats.get("neural_score_margin_total"),
            decision_count,
        ),
        "linear_anchor_score_margin_mean": _safe_mean_total(
            stats.get("linear_anchor_score_margin_total"),
            decision_count,
        ),
        "anchored_score_margin_mean": _safe_mean_total(
            stats.get("anchored_score_margin_total"),
            decision_count,
        ),
        "changed_neural_score_margin_mean": _safe_mean_total(
            stats.get("changed_neural_score_margin_total"),
            changed_linear_action_count,
        ),
        "changed_linear_anchor_score_margin_mean": _safe_mean_total(
            stats.get("changed_linear_anchor_score_margin_total"),
            changed_linear_action_count,
        ),
        "neural_top_action_counts": dict(
            sorted(_int_counter(stats.get("neural_top_action_counts", {})).items())
        ),
        "linear_anchor_action_counts": dict(
            sorted(_int_counter(stats.get("linear_anchor_action_counts", {})).items())
        ),
        "anchored_action_counts": dict(
            sorted(_int_counter(stats.get("anchored_action_counts", {})).items())
        ),
        "linear_to_anchored_action_counts": dict(
            sorted(
                _int_counter(
                    stats.get("linear_to_anchored_action_counts", {})
                ).items()
            )
        ),
        "neural_to_anchored_action_counts": dict(
            sorted(
                _int_counter(
                    stats.get("neural_to_anchored_action_counts", {})
                ).items()
            )
        ),
        "shadow_reason_counts": dict(
            sorted(_int_counter(stats.get("shadow_reason_counts", {})).items())
        ),
    }


def _diagnostic_action(
    diagnostic: Mapping[str, object],
    key: str,
) -> str | None:
    value = diagnostic.get(key)
    if not isinstance(value, str) or not value or value == "none":
        return None
    return value


def _diagnostic_float(
    diagnostic: Mapping[str, object],
    key: str,
) -> float:
    value = diagnostic.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _safe_mean_total(value: object, count: int) -> float:
    if count <= 0 or isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return _round(float(value) / float(count))


def _comparison_delta(
    *,
    heuristic: dict[str, object],
    mind_v3: dict[str, object],
) -> dict[str, object]:
    heuristic_attr = dict(heuristic["reproduction_failure_attribution"])
    mind_v3_attr = dict(mind_v3["reproduction_failure_attribution"])
    return {
        "alive_agents_mean": _round(
            float(mind_v3["alive_agents_mean"])
            - float(heuristic["alive_agents_mean"])
        ),
        "births_mean": _round(
            float(mind_v3["births_mean"]) - float(heuristic["births_mean"])
        ),
        "biologically_ready_agents_mean": _round(
            float(mind_v3_attr["biologically_ready_agents_mean"])
            - float(heuristic_attr["biologically_ready_agents_mean"])
        ),
        "ready_agents_mean": _round(
            float(mind_v3_attr["ready_agents_mean"])
            - float(heuristic_attr["ready_agents_mean"])
        ),
    }


def _reproduction_failure_attribution(
    summary: dict[str, object],
) -> dict[str, object]:
    alive = int(summary.get("alive_agents", 0))
    reproduction_end = summary.get("reproduction_end", {})
    if not isinstance(reproduction_end, dict):
        reproduction_end = {}
    blocker_counts = _int_counter(
        reproduction_end.get("biological_blocker_counts", {})
    )
    blocker_shares = {
        reason: _share(blocker_counts.get(reason, 0), alive)
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    terminal_viability_shares = {
        reason: _share(max(0, alive - blocker_counts.get(reason, 0)), alive)
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    primary_blocker = _dominant_count_key(blocker_counts)
    ready_agents = int(reproduction_end.get("ready_agents", 0))
    biologically_ready_agents = int(
        reproduction_end.get("biologically_ready_agents", 0)
    )
    energy_readiness = _terminal_energy_readiness(reproduction_end)
    energy_requirement_satisfaction = float(
        energy_readiness["energy_requirement_satisfaction"]
    )
    terminal_hydration_viability = float(terminal_viability_shares["hydration"])
    terminal_balanced_readiness = _balanced_readiness_score(
        [
            energy_requirement_satisfaction,
            terminal_hydration_viability,
            float(terminal_viability_shares["health"]),
            float(terminal_viability_shares["matched_diet"]),
        ]
    )
    return {
        "policy": MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
        "alive_agents": alive,
        "ready_agents": ready_agents,
        "biologically_ready_agents": biologically_ready_agents,
        "terminal_ready_share": _share(ready_agents, alive),
        "terminal_biologically_ready_share": _share(
            biologically_ready_agents,
            alive,
        ),
        "blocked_by_max_population_agents": int(
            reproduction_end.get("blocked_by_max_population_agents", 0)
        ),
        "blocked_by_local_crowding_agents": int(
            reproduction_end.get("blocked_by_local_crowding_agents", 0)
        ),
        "biological_blocker_counts": dict(sorted(blocker_counts.items())),
        "biological_blocker_share_by_alive": blocker_shares,
        "terminal_viability_shares": terminal_viability_shares,
        "terminal_energy_requirement_satisfaction": (
            energy_requirement_satisfaction
        ),
        "terminal_balanced_reproduction_readiness": terminal_balanced_readiness,
        "terminal_energy_hydration_balance": _round(
            min(energy_requirement_satisfaction, terminal_hydration_viability)
        ),
        "terminal_energy_hydration_gap_abs": _round(
            abs(energy_requirement_satisfaction - terminal_hydration_viability)
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
        "primary_terminal_blocker": primary_blocker,
        "reproductive_stage_counts": _json_ready(
            reproduction_end.get("reproductive_stage_counts", {})
        ),
        "reproductive_expression_counts": _json_ready(
            reproduction_end.get("reproductive_expression_counts", {})
        ),
        "biological_blocker_counts_by_trophic_role": _json_ready(
            reproduction_end.get("biological_blocker_counts_by_trophic_role", {})
        ),
        "biological_blocker_counts_by_meat_mode": _json_ready(
            reproduction_end.get("biological_blocker_counts_by_meat_mode", {})
        ),
        "energy_readiness_by_trophic_role": _json_ready(
            reproduction_end.get("energy_readiness_by_trophic_role", {})
        ),
        "energy_readiness_by_meat_mode": _json_ready(
            reproduction_end.get("energy_readiness_by_meat_mode", {})
        ),
        "by_trophic_role": _json_ready(reproduction_end.get("by_trophic_role", {})),
        "by_meat_mode": _json_ready(reproduction_end.get("by_meat_mode", {})),
    }


def _aggregate_reproduction_failure_attribution(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    attributions = [
        dict(run["reproduction_failure_attribution"])
        for run in runs
        if isinstance(run.get("reproduction_failure_attribution"), dict)
    ]
    alive_total = sum(int(item["alive_agents"]) for item in attributions)
    blocker_counts: Counter[str] = Counter()
    for item in attributions:
        blocker_counts.update(_int_counter(item["biological_blocker_counts"]))
    blocker_share_by_alive = {
        reason: _share(blocker_counts.get(reason, 0), alive_total)
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    blocker_share_by_alive_mean = {
        reason: _mean(
            [
                float(
                    dict(item["biological_blocker_share_by_alive"]).get(
                        reason,
                        0.0,
                    )
                )
                for item in attributions
            ]
        )
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    terminal_viability_shares_mean = {
        reason: _mean(
            [
                float(dict(item["terminal_viability_shares"]).get(reason, 0.0))
                for item in attributions
            ]
        )
        for reason in BIOLOGICAL_BLOCKER_REASONS
    }
    return {
        "policy": MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
        "run_count": len(attributions),
        "alive_agents_total": alive_total,
        "ready_agents_mean": _mean(
            [int(item["ready_agents"]) for item in attributions]
        ),
        "biologically_ready_agents_mean": _mean(
            [int(item["biologically_ready_agents"]) for item in attributions]
        ),
        "terminal_ready_share_mean": _mean(
            [float(item["terminal_ready_share"]) for item in attributions]
        ),
        "terminal_biologically_ready_share_mean": _mean(
            [
                float(item["terminal_biologically_ready_share"])
                for item in attributions
            ]
        ),
        "blocked_by_max_population_agents_mean": _mean(
            [
                int(item["blocked_by_max_population_agents"])
                for item in attributions
            ]
        ),
        "blocked_by_local_crowding_agents_mean": _mean(
            [
                int(item["blocked_by_local_crowding_agents"])
                for item in attributions
            ]
        ),
        "biological_blocker_counts": dict(sorted(blocker_counts.items())),
        "biological_blocker_share_by_alive": blocker_share_by_alive,
        "biological_blocker_share_by_alive_mean": blocker_share_by_alive_mean,
        "terminal_viability_shares_mean": terminal_viability_shares_mean,
        "terminal_energy_requirement_satisfaction_mean": _mean(
            [
                float(
                    item.get("terminal_energy_requirement_satisfaction", 0.0)
                )
                for item in attributions
            ]
        ),
        "terminal_balanced_reproduction_readiness_mean": _mean(
            [
                float(
                    item.get("terminal_balanced_reproduction_readiness", 0.0)
                )
                for item in attributions
            ]
        ),
        "terminal_energy_hydration_balance_mean": _mean(
            [
                float(item.get("terminal_energy_hydration_balance", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_hydration_gap_abs_mean": _mean(
            [
                float(item.get("terminal_energy_hydration_gap_abs", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_shortfall_share_mean": _mean(
            [
                float(item.get("terminal_energy_shortfall_share", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_shortfall_agents_mean": _mean(
            [
                int(item.get("terminal_energy_shortfall_agents", 0))
                for item in attributions
            ]
        ),
        "terminal_energy_total_mean": _mean(
            [
                float(item.get("terminal_energy_total", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_required_total_mean": _mean(
            [
                float(item.get("terminal_energy_required_total", 0.0))
                for item in attributions
            ]
        ),
        "terminal_energy_gap_total_mean": _mean(
            [
                float(item.get("terminal_energy_gap_total", 0.0))
                for item in attributions
            ]
        ),
        "primary_terminal_blocker": _dominant_count_key(blocker_counts),
        "biological_blocker_counts_by_trophic_role": _sum_grouped_int_counts(
            attributions,
            "biological_blocker_counts_by_trophic_role",
        ),
        "biological_blocker_counts_by_meat_mode": _sum_grouped_int_counts(
            attributions,
            "biological_blocker_counts_by_meat_mode",
        ),
        "terminal_readiness_by_trophic_role": _sum_grouped_int_counts(
            attributions,
            "by_trophic_role",
        ),
        "terminal_readiness_by_meat_mode": _sum_grouped_int_counts(
            attributions,
            "by_meat_mode",
        ),
    }


def _temporal_readiness_attribution(
    records: list[dict[str, object]],
) -> dict[str, object]:
    blocker_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    alive_record_count = 0
    for record in records:
        after = record.get("after")
        if not isinstance(after, Mapping) or not bool(after.get("alive", False)):
            continue
        alive_record_count += 1
        blockers = _core_readiness_blockers(after)
        blocker_counts.update(blockers)
        primary = _primary_core_readiness_blocker(after)
        if primary is not None:
            primary_counts.update([primary])
    return {
        "policy": MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY,
        "alive_agent_tick_count": alive_record_count,
        "core_blocker_agent_tick_counts": {
            field: int(blocker_counts[field]) for field in _core_fields()
        },
        "core_blocker_agent_tick_shares": {
            field: _round(int(blocker_counts[field]) / max(1, alive_record_count))
            for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_counts": {
            field: int(primary_counts[field]) for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_shares": {
            field: _round(int(primary_counts[field]) / max(1, alive_record_count))
            for field in _core_fields()
        },
        "primary_temporal_readiness_blocker": _dominant_count_key(primary_counts),
    }


def _aggregate_temporal_readiness_attribution(
    runs: list[dict[str, object]],
) -> dict[str, object]:
    alive_tick_count = 0
    blocker_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    for run in runs:
        attribution = run.get("temporal_readiness_attribution")
        if not isinstance(attribution, Mapping):
            continue
        alive_tick_count += int(attribution.get("alive_agent_tick_count", 0))
        blocker_counts.update(
            _int_counter(attribution.get("core_blocker_agent_tick_counts", {}))
        )
        primary_counts.update(
            _int_counter(
                attribution.get("primary_core_blocker_agent_tick_counts", {})
            )
        )
    return {
        "policy": MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY,
        "alive_agent_tick_count": alive_tick_count,
        "core_blocker_agent_tick_counts": {
            field: int(blocker_counts[field]) for field in _core_fields()
        },
        "core_blocker_agent_tick_shares": {
            field: _round(int(blocker_counts[field]) / max(1, alive_tick_count))
            for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_counts": {
            field: int(primary_counts[field]) for field in _core_fields()
        },
        "primary_core_blocker_agent_tick_shares": {
            field: _round(int(primary_counts[field]) / max(1, alive_tick_count))
            for field in _core_fields()
        },
        "primary_temporal_readiness_blocker": _dominant_count_key(primary_counts),
    }


def _core_readiness_blockers(after: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        value = _float_value(after.get(field))
        if value is not None and value < float(goal):
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


def _heuristic_action_source_count(counts: Counter[str]) -> int:
    return sum(count for source, count in counts.items() if "heuristic" in source)


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


def _dominant_count_key(counts: Counter[str]) -> str | None:
    positive = {key: count for key, count in counts.items() if int(count) > 0}
    if not positive:
        return None
    key, _ = max(
        sorted(positive.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return str(key)


def _int_counter(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        counts[str(key)] += int(count)
    return counts


def _sum_grouped_int_counts(
    attributions: list[dict[str, object]],
    key: str,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for attribution in attributions:
        raw_groups = attribution.get(key, {})
        if not isinstance(raw_groups, dict):
            continue
        for raw_group, raw_counts in raw_groups.items():
            if not isinstance(raw_counts, dict):
                continue
            group = str(raw_group)
            if group not in grouped:
                grouped[group] = Counter()
            grouped[group].update(_int_counter(raw_counts))
    return {
        group: dict(sorted(counts.items()))
        for group, counts in sorted(grouped.items())
    }


def _terminal_energy_readiness(
    reproduction_end: Mapping[str, object],
) -> dict[str, float | int]:
    raw_by_mode = reproduction_end.get("energy_readiness_by_meat_mode", {})
    by_mode = raw_by_mode if isinstance(raw_by_mode, Mapping) else {}
    alive_agents = 0
    shortfall_agents = 0
    energy_total = 0.0
    required_total = 0.0
    gap_total = 0.0
    for raw_counts in by_mode.values():
        if not isinstance(raw_counts, Mapping):
            continue
        alive_agents += _nonnegative_int(raw_counts.get("alive_agents"))
        shortfall_agents += _nonnegative_int(
            raw_counts.get("energy_shortfall_agents")
        )
        energy_total += _nonnegative_float(raw_counts.get("energy_total"))
        required_total += _nonnegative_float(
            raw_counts.get("energy_required_total")
        )
        gap_total += _nonnegative_float(raw_counts.get("energy_gap_total"))
    return {
        "alive_agents": alive_agents,
        "energy_shortfall_agents": shortfall_agents,
        "energy_total": _round(energy_total),
        "energy_required_total": _round(required_total),
        "energy_gap_total": _round(gap_total),
        "energy_shortfall_share": _share(shortfall_agents, alive_agents),
        "energy_requirement_satisfaction": _round(
            max(0.0, min(1.0, 1.0 - gap_total / required_total))
            if required_total > 0.0
            else 0.0
        ),
    }


def _balanced_readiness_score(values: list[float]) -> float:
    bounded = [max(0.0, min(1.0, float(value))) for value in values]
    if not bounded or any(value <= 0.0 for value in bounded):
        return 0.0
    return _round(len(bounded) / sum(1.0 / value for value in bounded))


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round(numerator / float(denominator))


def _founder_template_count(
    template: dict[str, object] | list[dict[str, object]] | None,
) -> int:
    if template is None:
        return 0
    if isinstance(template, list):
        return len(template)
    return 1


def _founder_template_specialization_profile_counts(
    template: dict[str, object] | list[dict[str, object]] | None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    templates = template if isinstance(template, list) else [template]
    for metadata in templates:
        if not isinstance(metadata, dict):
            continue
        profile = metadata.get("specialization_profile")
        counts.update([str(profile) if isinstance(profile, str) else "unknown"])
    return dict(sorted(counts.items()))


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seeds must include at least one integer seed")
    return seeds


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / float(len(values)))


def _round(value: float) -> float:
    return round(float(value), 4)


if __name__ == "__main__":
    main()
