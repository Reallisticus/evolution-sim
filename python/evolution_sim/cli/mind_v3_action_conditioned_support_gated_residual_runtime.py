from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_INPUT_VECTOR_SIZE
from evolution_sim.mind.broad_branch_residual_distillation_example import load_json_report
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
    MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
    build_action_conditioned_support_gated_residual_runtime_artifact,
    build_v104_branch_replay_feasibility_report,
)
from evolution_sim.mind.v3_planner_distilled import (
    _compact_state_from_values,
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy
from evolution_sim.cli.mind_v3_support_gated_residual_runtime import (
    V103_DEFAULT_TICKS,
    V103_MAX_DOMINANT_SHADOW_OVERRIDE_ACTION_SHARE,
    V103_NONSTRICT_LIVE_SEEDS,
    V103_STRICT_BROAD_SEEDS,
    _aggregate_runs,
    _parse_seeds,
    _reject_strict_leakage,
    _run_policy,
    _shadow_gate,
)

V104_SHADOW_FAILURE_AUDIT_SCHEMA_VERSION = (
    "mind_v3_v104_shadow_failure_audit_v1"
)
V104_RUNTIME_REPORT_SCHEMA_VERSION = (
    "mind_v3_v104_action_conditioned_support_gated_residual_runtime_report_v1"
)
V104_SHADOW_STRICT_REPORT_SCHEMA_VERSION = (
    "mind_v3_v104_shadow_strict_report_v1"
)
V104_NONSTRICT_LIVE_REPORT_SCHEMA_VERSION = (
    "mind_v3_v104_nonstrict_live_feasibility_report_v1"
)
V104_LEDGER_SCHEMA_VERSION = (
    "mind_v3_v104_action_conditioned_support_gated_residual_ledger_v1"
)
V104_MAX_DOMINANT_LIVE_ACTION_SHARE = 0.50

_FEATURE_SUMMARY_KEYS = (
    "self_energy_ratio",
    "self_hydration_ratio",
    "self_health_ratio",
    "center_water",
    "center_food",
    "center_carrion",
    "radius1_water",
    "radius1_food",
    "radius1_carrion",
    "radius2_water",
    "radius2_food",
    "radius2_carrion",
    "navigation_water_distance",
    "navigation_water_strength",
    "navigation_plant_distance",
    "navigation_plant_strength",
    "navigation_carrion_distance",
    "navigation_carrion_strength",
    "navigation_prey_distance",
    "navigation_prey_strength",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate the v104 action-conditioned support-gated "
            "residual runtime feasibility slice. This is audit-only and never "
            "runs strict live promotion."
        )
    )
    parser.add_argument(
        "--v102-report",
        type=Path,
        default=Path("output/mind/mind-v3-v102-expanded-broad-residual-training.json"),
    )
    parser.add_argument(
        "--v102-artifact",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v102-expanded-broad-residual-training-artifact.json"
        ),
    )
    parser.add_argument(
        "--v99-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v102-expanded-broad-residual-oracle-source.json"
        ),
    )
    parser.add_argument(
        "--v100-report",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v102-expanded-broad-residual-constrained-source.json"
        ),
    )
    parser.add_argument(
        "--v103-artifact",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v103-support-gated-residual-runtime-artifact.json"
        ),
    )
    parser.add_argument(
        "--shadow-failure-output",
        type=Path,
        default=Path("output/mind/mind-v3-v104-shadow-failure-audit.json"),
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v104-action-conditioned-support-gated-residual-runtime-artifact.json"
        ),
    )
    parser.add_argument(
        "--branch-output",
        type=Path,
        default=Path("output/mind/mind-v3-v104-branch-replay-feasibility.json"),
    )
    parser.add_argument(
        "--shadow-output",
        type=Path,
        default=Path("output/mind/mind-v3-v104-shadow-strict-report.json"),
    )
    parser.add_argument(
        "--live-output",
        type=Path,
        default=Path("output/mind/mind-v3-v104-nonstrict-live-feasibility.json"),
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v104-action-conditioned-support-gated-residual-ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v104-action-conditioned-support-gated-residual-report.json"
        ),
    )
    parser.add_argument(
        "--strict-seeds",
        default=",".join(str(seed) for seed in V103_STRICT_BROAD_SEEDS),
    )
    parser.add_argument(
        "--non-strict-seeds",
        default=",".join(str(seed) for seed in V103_NONSTRICT_LIVE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=V103_DEFAULT_TICKS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    v102_report = load_json_report(args.v102_report)
    v102_artifact = load_json_report(args.v102_artifact)
    v99_report = load_json_report(args.v99_report)
    v100_report = load_json_report(args.v100_report)
    v103_artifact = load_json_report(args.v103_artifact)
    strict_seeds = _parse_seeds(args.strict_seeds)
    non_strict_seeds = _parse_seeds(args.non_strict_seeds)
    _reject_strict_leakage(non_strict_seeds)

    shadow_failure_audit = build_v104_shadow_failure_audit_report(
        v103_artifact=v103_artifact,
        v102_artifact=v102_artifact,
        seeds=strict_seeds,
        ticks=int(args.ticks),
    )
    _write_json(args.shadow_failure_output, shadow_failure_audit)

    runtime_artifact, threshold_report = (
        build_action_conditioned_support_gated_residual_runtime_artifact(
            v102_report=v102_report,
            v102_artifact=v102_artifact,
            v99_report=v99_report,
            v100_report=v100_report,
        )
    )
    _write_json(args.artifact_output, runtime_artifact)

    ledger_entries = []
    branch_report = build_v104_branch_replay_feasibility_report(
        runtime_artifact=runtime_artifact,
        v99_report=v99_report,
        v100_report=v100_report,
    )
    _write_json(args.branch_output, branch_report)
    ledger_entries.append(
        _ledger_entry(
            stage="branch_replay",
            passed=bool(branch_report["v104_branch_replay_feasibility_passed"]),
            metrics=branch_report["summary"],
            blockers=branch_report["safety_gate"]["blockers"],
        )
    )

    shadow_report = None
    live_report = None
    if branch_report["v104_branch_replay_feasibility_passed"] is True:
        shadow_report = build_v104_shadow_strict_report(
            runtime_artifact=runtime_artifact,
            seeds=strict_seeds,
            ticks=int(args.ticks),
        )
        _write_json(args.shadow_output, shadow_report)
        ledger_entries.append(
            _ledger_entry(
                stage="shadow_strict",
                passed=bool(shadow_report["shadow_gate"]["passed"]),
                metrics=shadow_report["aggregate"]["support_residual_diagnostics"],
                blockers=shadow_report["shadow_gate"]["blockers"],
            )
        )
        if shadow_report["shadow_gate"]["passed"] is True:
            live_report = build_v104_nonstrict_live_feasibility_report(
                runtime_artifact=runtime_artifact,
                seeds=non_strict_seeds,
                ticks=int(args.ticks),
            )
            _write_json(args.live_output, live_report)
            ledger_entries.append(
                _ledger_entry(
                    stage="nonstrict_live",
                    passed=bool(live_report["live_feasibility_gate"]["passed"]),
                    metrics=live_report["delta"],
                    blockers=live_report["live_feasibility_gate"]["blockers"],
                )
            )

    final_passed = (
        branch_report["v104_branch_replay_feasibility_passed"] is True
        and shadow_report is not None
        and shadow_report["shadow_gate"]["passed"] is True
        and live_report is not None
        and live_report["live_feasibility_gate"]["passed"] is True
    )
    report = {
        "schema_version": V104_RUNTIME_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_promotion_allowed": False,
        "strict_live_promotion_executed": False,
        "v105_strict_promotion_allowed": bool(final_passed),
        "shadow_failure_audit_output": str(args.shadow_failure_output),
        "artifact_output": str(args.artifact_output),
        "branch_replay_output": str(args.branch_output),
        "shadow_strict_output": str(args.shadow_output) if shadow_report else None,
        "nonstrict_live_output": str(args.live_output) if live_report else None,
        "ledger_output": str(args.ledger_output),
        "threshold_report": threshold_report,
        "branch_replay_passed": bool(
            branch_report["v104_branch_replay_feasibility_passed"]
        ),
        "shadow_strict_passed": (
            bool(shadow_report["shadow_gate"]["passed"])
            if shadow_report is not None
            else False
        ),
        "nonstrict_live_feasibility_passed": (
            bool(live_report["live_feasibility_gate"]["passed"])
            if live_report is not None
            else False
        ),
        "active_shadow_failure_mining": {
            "executed": False,
            "reason": (
                "action_conditioned_gate_passed_strict_shadow"
                if shadow_report is not None and shadow_report["shadow_gate"]["passed"] is True
                else "not_reached_or_not_needed_after_action_conditioned_gate"
            ),
        },
        "v104_runtime_feasibility_passed": bool(final_passed),
        "decision": (
            "v105_strict_promotion_allowed"
            if final_passed
            else "rejected_or_stopped"
        ),
    }
    _write_json(args.report_output, report)
    _write_jsonl(args.ledger_output, ledger_entries)

    print(f"v104_shadow_failure_audit={args.shadow_failure_output}")
    print(f"v104_runtime_artifact={args.artifact_output}")
    print(f"v104_branch_replay_report={args.branch_output}")
    print(
        "v104_branch_replay_passed="
        f"{branch_report['v104_branch_replay_feasibility_passed']}"
    )
    if shadow_report is not None:
        print(f"v104_shadow_strict_report={args.shadow_output}")
        print(f"v104_shadow_strict_passed={shadow_report['shadow_gate']['passed']}")
    if live_report is not None:
        print(f"v104_nonstrict_live_report={args.live_output}")
        print(
            "v104_nonstrict_live_feasibility_passed="
            f"{live_report['live_feasibility_gate']['passed']}"
        )
    print(f"v104_runtime_report={args.report_output}")
    print(f"v104_ledger={args.ledger_output}")
    print(f"v104_runtime_feasibility_passed={final_passed}")
    print(f"v105_strict_promotion_allowed={final_passed}")


def build_v104_shadow_failure_audit_report(
    *,
    v103_artifact: Mapping[str, object],
    v102_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    decisions = []
    for seed in seeds:
        decisions.extend(
            _collect_shadow_decision_rows(
                seed=seed,
                ticks=ticks,
                runtime_artifact=v103_artifact,
            )
        )
    return build_shadow_failure_audit_from_decisions(
        decisions=decisions,
        v102_artifact=v102_artifact,
        seeds=seeds,
        ticks=ticks,
    )


def build_shadow_failure_audit_from_decisions(
    *,
    decisions: Sequence[Mapping[str, object]],
    v102_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    proposed_counts: Counter[str] = Counter()
    proposed_override_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    accepted_by_seed: dict[int, Counter[str]] = {
        int(seed): Counter() for seed in seeds
    }
    distance_by_action: dict[str, list[float]] = {}
    margin_by_action: dict[str, list[float]] = {}
    accepted_drink_rows = []
    for row in decisions:
        proposed = _optional_action(row.get("proposed_action"))
        if proposed is not None:
            proposed_counts.update([proposed])
        if row.get("override_proposed") is True and proposed is not None:
            proposed_override_counts.update([proposed])
        if row.get("override_allowed") is True and proposed is not None:
            accepted_counts.update([proposed])
            seed = int(row.get("seed", 0))
            accepted_by_seed.setdefault(seed, Counter()).update([proposed])
            distance = _optional_float(row.get("nearest_support_distance"))
            margin = _optional_float(row.get("score_margin"))
            if distance is not None:
                distance_by_action.setdefault(proposed, []).append(distance)
            if margin is not None:
                margin_by_action.setdefault(proposed, []).append(margin)
            if proposed == "drink":
                accepted_drink_rows.append(row)
    accepted_distance_margin_by_action = {
        action: {
            "distance_stats": _float_summary(distance_by_action.get(action, [])),
            "margin_stats": _float_summary(margin_by_action.get(action, [])),
        }
        for action in sorted(accepted_counts)
    }
    accepted_drink_rows = sorted(
        accepted_drink_rows,
        key=lambda item: (
            _float(item.get("selected_score")),
            _float(item.get("score_margin")),
            -_float(item.get("nearest_support_distance")),
        ),
        reverse=True,
    )
    drink_support = [
        item
        for item in _list_of_mappings(v102_artifact.get("support_examples"))
        if item.get("action") == "drink"
    ]
    support_comparison = _drink_support_comparison(
        accepted_drink_rows=accepted_drink_rows,
        drink_support=drink_support,
    )
    repeated = _accepted_drink_repetition_summary(accepted_drink_rows)
    dominant = _dominant_count_share(accepted_counts)
    return {
        "schema_version": V104_SHADOW_FAILURE_AUDIT_SCHEMA_VERSION,
        "source_policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        "repair_policy": MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_mode": "shadow",
        "strict_live_promotion_executed": False,
        "runtime_promotion_allowed": False,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "decision_count": len(decisions),
        "proposed_action_counts": dict(sorted(proposed_counts.items())),
        "proposed_override_action_counts": dict(sorted(proposed_override_counts.items())),
        "gate_accepted_override_action_counts": dict(sorted(accepted_counts.items())),
        "per_strict_seed_gate_accepted_action_counts": {
            str(seed): dict(sorted(accepted_by_seed.get(int(seed), Counter()).items()))
            for seed in sorted(int(seed) for seed in seeds)
        },
        "accepted_distance_margin_stats_by_action": accepted_distance_margin_by_action,
        "dominant_gate_accepted_override_action": dominant["key"],
        "dominant_gate_accepted_override_action_share": dominant["share"],
        "accepted_drink_repetition": repeated,
        "top_accepted_drink_examples": [
            _top_drink_example(row, drink_support=drink_support)
            for row in accepted_drink_rows[:12]
        ],
        "strict_accepted_drink_vs_v102_support": support_comparison,
        "diagnosis": (
            "The v103 global gate admits many strict shadow drink overrides "
            "because drink distances and margins are inside the global p75 "
            "support envelope across most strict seeds; the accepted drink "
            "rows are broad across agents rather than a single repeated tick."
        ),
    }


def build_v104_shadow_strict_report(
    *,
    runtime_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    runs = [
        _run_policy(
            seed=seed,
            ticks=ticks,
            runtime_artifact=runtime_artifact,
            runtime_mode="shadow",
        )
        for seed in seeds
    ]
    aggregate = _aggregate_runs(runs)
    gate = _shadow_gate(aggregate["support_residual_diagnostics"])
    return {
        "schema_version": V104_SHADOW_STRICT_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_mode": "shadow",
        "strict_live_promotion_executed": False,
        "runtime_promotion_allowed": False,
        "seeds": list(seeds),
        "ticks": int(ticks),
        "runs": runs,
        "aggregate": aggregate,
        "shadow_gate": gate,
        "v104_shadow_strict_passed": bool(gate["passed"]),
    }


def build_v104_nonstrict_live_feasibility_report(
    *,
    runtime_artifact: Mapping[str, object],
    seeds: Sequence[int],
    ticks: int,
) -> dict[str, object]:
    linear_runs = [
        _run_policy(seed=seed, ticks=ticks, runtime_artifact=None)
        for seed in seeds
    ]
    residual_runs = [
        _run_policy(
            seed=seed,
            ticks=ticks,
            runtime_artifact=runtime_artifact,
            runtime_mode="live",
        )
        for seed in seeds
    ]
    linear_aggregate = _aggregate_runs(linear_runs)
    residual_aggregate = _aggregate_runs(residual_runs)
    delta = {
        "alive_agents_mean": _round(
            _float(residual_aggregate.get("alive_agents_mean"))
            - _float(linear_aggregate.get("alive_agents_mean"))
        ),
        "births_mean": _round(
            _float(residual_aggregate.get("births_mean"))
            - _float(linear_aggregate.get("births_mean"))
        ),
        "deaths_mean": _round(
            _float(residual_aggregate.get("deaths_mean"))
            - _float(linear_aggregate.get("deaths_mean"))
        ),
    }
    paired_seed_deltas = _paired_seed_deltas(
        linear_runs=linear_runs,
        residual_runs=residual_runs,
    )
    gate = _v104_live_feasibility_gate(
        linear_runs=linear_runs,
        residual_runs=residual_runs,
        residual_aggregate=residual_aggregate,
        delta=delta,
    )
    return {
        "schema_version": V104_NONSTRICT_LIVE_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_mode": "live",
        "strict_live_promotion_executed": False,
        "runtime_promotion_allowed": False,
        "seeds": list(seeds),
        "ticks": int(ticks),
        "linear": {
            "runs": linear_runs,
            "aggregate": linear_aggregate,
        },
        "support_gated_residual": {
            "runs": residual_runs,
            "aggregate": residual_aggregate,
        },
        "paired_seed_deltas": paired_seed_deltas,
        "delta": delta,
        "live_feasibility_gate": gate,
        "v104_nonstrict_live_feasibility_passed": bool(gate["passed"]),
    }


def _collect_shadow_decision_rows(
    *,
    seed: int,
    ticks: int,
    runtime_artifact: Mapping[str, object],
) -> list[dict[str, object]]:
    policy = MindV3EvolutionPolicy(
        seed=seed,
        support_residual_artifact=runtime_artifact,
        support_residual_runtime_mode="shadow",
    )
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=policy,
    )
    world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    rows = []
    for record, diagnostic in zip(
        world.trajectory_records,
        world.policy_decision_diagnostics_records,
    ):
        if not isinstance(diagnostic, Mapping):
            continue
        if diagnostic.get("support_residual_policy") not in {
            MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
            MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        }:
            continue
        proposed = _optional_action(diagnostic.get("support_residual_proposed_action"))
        feature_vector = _candidate_vector_from_record(record, proposed or "stay")
        rows.append(
            {
                "seed": int(seed),
                "tick": int(record.get("tick", 0)),
                "agent_id": int(record.get("agent_id", 0)),
                "linear_action": diagnostic.get("support_residual_linear_action"),
                "proposed_action": proposed,
                "final_action": diagnostic.get("support_residual_final_action"),
                "override_proposed": bool(
                    diagnostic.get("support_residual_override_proposed")
                ),
                "override_allowed": bool(
                    diagnostic.get("support_residual_override_allowed")
                ),
                "nearest_support_distance": diagnostic.get(
                    "support_residual_nearest_support_distance"
                ),
                "score_margin": diagnostic.get("support_residual_score_margin"),
                "selected_score": diagnostic.get("support_residual_selected_score"),
                "candidate_scores_top": list(
                    diagnostic.get("support_residual_candidate_scores_top", [])
                ),
                "policy_visible_features": _feature_summary_from_record(record),
                "feature_vector": feature_vector,
            }
        )
    return rows


def _v104_live_feasibility_gate(
    *,
    linear_runs: Sequence[Mapping[str, object]],
    residual_runs: Sequence[Mapping[str, object]],
    residual_aggregate: Mapping[str, object],
    delta: Mapping[str, object],
) -> dict[str, object]:
    blockers = []

    def block(reason: str, field: str, observed: object, required: object, comparator: str) -> None:
        blockers.append(
            {
                "reason": reason,
                "field": field,
                "observed": observed,
                "required": required,
                "comparator": comparator,
            }
        )

    diagnostics = _mapping(residual_aggregate.get("support_residual_diagnostics"))
    unsupported_total = (
        int(residual_aggregate.get("unsupported_requested_action_count", 0))
        + int(residual_aggregate.get("unsupported_resolved_action_count", 0))
        + int(diagnostics.get("unsupported_proposed_action_count", 0))
    )
    if unsupported_total != 0:
        block("unsupported_action", "unsupported_action_total", unsupported_total, 0, "eq")
    if _float(delta.get("alive_agents_mean")) < 0.0:
        block(
            "mean_alive_regression",
            "delta.alive_agents_mean",
            delta.get("alive_agents_mean"),
            0.0,
            "ge",
        )
    if _float(delta.get("births_mean")) < 0.0:
        block(
            "mean_birth_regression",
            "delta.births_mean",
            delta.get("births_mean"),
            0.0,
            "ge",
        )
    requested_dominant = _dominant_count_share(
        _int_counter(residual_aggregate.get("requested_action_counts"))
    )
    if requested_dominant["share"] > V104_MAX_DOMINANT_LIVE_ACTION_SHARE:
        block(
            "dominant_requested_action_share_above_cap",
            "dominant_requested_action_share",
            requested_dominant["share"],
            V104_MAX_DOMINANT_LIVE_ACTION_SHARE,
            "le",
        )
    if (
        _float(diagnostics.get("dominant_applied_override_action_share"))
        > V104_MAX_DOMINANT_LIVE_ACTION_SHARE
    ):
        block(
            "dominant_applied_override_action_share_above_cap",
            "dominant_applied_override_action_share",
            diagnostics.get("dominant_applied_override_action_share"),
            V104_MAX_DOMINANT_LIVE_ACTION_SHARE,
            "le",
        )
    return {
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "dominant_requested_action": requested_dominant["key"],
        "dominant_requested_action_share": requested_dominant["share"],
        "first_failing_seed_action_pattern": _first_failing_seed_pattern(
            linear_runs=linear_runs,
            residual_runs=residual_runs,
        ),
        "runtime_promotion_allowed": False,
    }


def _paired_seed_deltas(
    *,
    linear_runs: Sequence[Mapping[str, object]],
    residual_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    residual_by_seed = {int(run.get("seed", 0)): run for run in residual_runs}
    rows = []
    for linear in sorted(linear_runs, key=lambda item: int(item.get("seed", 0))):
        seed = int(linear.get("seed", 0))
        residual = residual_by_seed.get(seed)
        if residual is None:
            continue
        diagnostics = _mapping(residual.get("support_residual_diagnostics"))
        rows.append(
            {
                "seed": seed,
                "linear_alive_agents": linear.get("alive_agents"),
                "residual_alive_agents": residual.get("alive_agents"),
                "alive_agents_delta": int(residual.get("alive_agents", 0))
                - int(linear.get("alive_agents", 0)),
                "linear_births": linear.get("births"),
                "residual_births": residual.get("births"),
                "births_delta": int(residual.get("births", 0))
                - int(linear.get("births", 0)),
                "applied_override_action_counts": diagnostics.get(
                    "applied_override_action_counts"
                ),
                "dominant_applied_override_action": diagnostics.get(
                    "dominant_applied_override_action"
                ),
                "dominant_applied_override_action_share": diagnostics.get(
                    "dominant_applied_override_action_share"
                ),
            }
        )
    return rows


def _first_failing_seed_pattern(
    *,
    linear_runs: Sequence[Mapping[str, object]],
    residual_runs: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    residual_by_seed = {int(run.get("seed", 0)): run for run in residual_runs}
    for linear in sorted(linear_runs, key=lambda item: int(item.get("seed", 0))):
        seed = int(linear.get("seed", 0))
        residual = residual_by_seed.get(seed)
        if residual is None:
            continue
        diagnostics = _mapping(residual.get("support_residual_diagnostics"))
        unsupported_total = (
            int(residual.get("unsupported_requested_action_count", 0))
            + int(residual.get("unsupported_resolved_action_count", 0))
            + int(diagnostics.get("unsupported_proposed_action_count", 0))
        )
        alive_regressed = int(residual.get("alive_agents", 0)) < int(
            linear.get("alive_agents", 0)
        )
        births_regressed = int(residual.get("births", 0)) < int(
            linear.get("births", 0)
        )
        dominant_override_share = _float(
            diagnostics.get("dominant_applied_override_action_share")
        )
        if (
            unsupported_total
            or alive_regressed
            or births_regressed
            or dominant_override_share > V104_MAX_DOMINANT_LIVE_ACTION_SHARE
        ):
            return {
                "seed": seed,
                "linear_alive_agents": linear.get("alive_agents"),
                "residual_alive_agents": residual.get("alive_agents"),
                "linear_births": linear.get("births"),
                "residual_births": residual.get("births"),
                "unsupported_action_total": unsupported_total,
                "dominant_applied_override_action": diagnostics.get(
                    "dominant_applied_override_action"
                ),
                "dominant_applied_override_action_share": diagnostics.get(
                    "dominant_applied_override_action_share"
                ),
                "applied_override_action_counts": diagnostics.get(
                    "applied_override_action_counts"
                ),
                "abstention_reason_counts": diagnostics.get(
                    "abstention_reason_counts"
                ),
            }
    return None


def _drink_support_comparison(
    *,
    accepted_drink_rows: Sequence[Mapping[str, object]],
    drink_support: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    support_features = [
        _float_list(item.get("feature_vector")) for item in drink_support
    ]
    support_features = [item for item in support_features if item]
    accepted_summaries = [
        _mapping(item.get("policy_visible_features"))
        for item in accepted_drink_rows
    ]
    support_summaries = [
        _feature_summary_from_feature_vector(item)
        for item in support_features
    ]
    nearest_distances = []
    gate_nearest_distances = []
    nearest_indices: Counter[str] = Counter()
    for row in accepted_drink_rows:
        gate_distance = _optional_float(row.get("nearest_support_distance"))
        if gate_distance is not None:
            gate_nearest_distances.append(gate_distance)
        vector = _float_list(row.get("feature_vector"))
        if not vector:
            continue
        best_index = None
        best_distance = None
        for index, support_vector in enumerate(support_features):
            distance = _squared_distance(vector, support_vector)
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        if best_distance is not None:
            nearest_distances.append(best_distance)
        if best_index is not None:
            nearest_indices.update([str(best_index)])
    return {
        "accepted_drink_count": len(accepted_drink_rows),
        "v102_drink_support_count": len(drink_support),
        "accepted_drink_feature_stats": _feature_stats(accepted_summaries),
        "v102_drink_support_feature_stats": _feature_stats(support_summaries),
        "nearest_v102_drink_support_distance_stats": _float_summary(
            gate_nearest_distances
        ),
        "observation_only_nearest_v102_drink_support_distance_stats": _float_summary(
            nearest_distances
        ),
        "nearest_v102_drink_support_example_counts_top": [
            {"support_example_rank": int(key), "count": int(count)}
            for key, count in nearest_indices.most_common(10)
        ],
    }


def _accepted_drink_repetition_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    agent_counts = Counter(
        f"{int(row.get('seed', 0))}:{int(row.get('agent_id', 0))}" for row in rows
    )
    tick_counts = Counter(int(row.get("tick", 0)) for row in rows)
    seed_counts = Counter(int(row.get("seed", 0)) for row in rows)
    dominant_agent = _dominant_count_share(agent_counts)
    dominant_tick = _dominant_count_share(Counter({str(k): v for k, v in tick_counts.items()}))
    return {
        "accepted_drink_count": len(rows),
        "unique_seed_agent_count": len(agent_counts),
        "unique_tick_count": len(tick_counts),
        "per_seed_drink_counts": dict(sorted(seed_counts.items())),
        "max_seed_agent": dominant_agent["key"],
        "max_seed_agent_count": dominant_agent["count"],
        "max_seed_agent_share": dominant_agent["share"],
        "max_tick": dominant_tick["key"],
        "max_tick_count": dominant_tick["count"],
        "max_tick_share": dominant_tick["share"],
        "top_seed_agent_counts": [
            {"seed_agent": key, "count": int(count)}
            for key, count in agent_counts.most_common(10)
        ],
    }


def _top_drink_example(
    row: Mapping[str, object],
    *,
    drink_support: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    vector = _float_list(row.get("feature_vector"))
    nearest = _nearest_support_index(vector, drink_support)
    return {
        "seed": row.get("seed"),
        "tick": row.get("tick"),
        "agent_id": row.get("agent_id"),
        "linear_action": row.get("linear_action"),
        "proposed_action": row.get("proposed_action"),
        "selected_score": row.get("selected_score"),
        "score_margin": row.get("score_margin"),
        "nearest_support_distance": row.get("nearest_support_distance"),
        "nearest_v102_drink_support_example_index": nearest["index"],
        "nearest_v102_drink_support_distance": nearest["distance"],
        "candidate_scores_top": row.get("candidate_scores_top"),
        "policy_visible_features": row.get("policy_visible_features"),
    }


def _nearest_support_index(
    vector: Sequence[float],
    support: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    best_index = None
    best_distance = None
    for item in support:
        support_vector = _float_list(item.get("feature_vector"))
        if not support_vector:
            continue
        distance = _squared_distance(vector, support_vector)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = int(item.get("example_index", 0))
    return {
        "index": best_index,
        "distance": _round(best_distance) if best_distance is not None else None,
    }


def _feature_summary_from_record(record: Mapping[str, object]) -> dict[str, object]:
    row = planner_distilled_runtime_row(
        observation_input=_mapping(record.get("observation_input")),
        action_mask=_mapping(record.get("action_mask")),
        public_history_trace=[],
    )
    compact = _mapping(row.get("compact_state"))
    summary = _feature_summary_from_compact_state(compact)
    before = _mapping(record.get("before"))
    if before:
        summary.update(
            {
                "before_energy_ratio": before.get("energy_ratio"),
                "before_hydration_ratio": before.get("hydration_ratio"),
                "before_health_ratio": before.get("health_ratio"),
            }
        )
    mask = _mapping(record.get("action_mask"))
    summary.update(
        {
            "legal_drink": bool(mask.get("drink", False)),
            "legal_eat": bool(mask.get("eat", False)),
            "legal_stay": bool(mask.get("stay", False)),
        }
    )
    return summary


def _feature_summary_from_feature_vector(vector: Sequence[float]) -> dict[str, object]:
    observation_values = [float(value) for value in vector[:OBSERVATION_INPUT_VECTOR_SIZE]]
    compact = _compact_state_from_values(observation_values)
    return _feature_summary_from_compact_state(compact)


def _feature_summary_from_compact_state(compact: Mapping[str, object]) -> dict[str, object]:
    self_state = _mapping(compact.get("self"))
    center = _mapping(compact.get("center"))
    local = _mapping(compact.get("local"))
    navigation = _mapping(compact.get("navigation"))
    water_nav = _mapping(navigation.get("water"))
    plant_nav = _mapping(navigation.get("plant"))
    carrion_nav = _mapping(navigation.get("carrion"))
    prey_nav = _mapping(navigation.get("prey"))
    center_carrion = _float(center.get("fresh_kill")) + _float(center.get("carcass"))
    return {
        "self_energy_ratio": self_state.get("energy_ratio"),
        "self_hydration_ratio": self_state.get("hydration_ratio"),
        "self_health_ratio": self_state.get("health_ratio"),
        "center_water": center.get("water"),
        "center_food": center.get("food"),
        "center_carrion": _round(center_carrion),
        "radius1_water": local.get("radius1_water"),
        "radius1_food": local.get("radius1_food"),
        "radius1_carrion": local.get("radius1_carrion"),
        "radius2_water": local.get("radius2_water"),
        "radius2_food": local.get("radius2_food"),
        "radius2_carrion": local.get("radius2_carrion"),
        "navigation_water_distance": water_nav.get("distance"),
        "navigation_water_strength": water_nav.get("strength"),
        "navigation_plant_distance": plant_nav.get("distance"),
        "navigation_plant_strength": plant_nav.get("strength"),
        "navigation_carrion_distance": carrion_nav.get("distance"),
        "navigation_carrion_strength": carrion_nav.get("strength"),
        "navigation_prey_distance": prey_nav.get("distance"),
        "navigation_prey_strength": prey_nav.get("strength"),
    }


def _candidate_vector_from_record(
    record: Mapping[str, object],
    action: str,
) -> list[float]:
    if action not in ACTION_NAMES:
        return []
    row = planner_distilled_runtime_row(
        observation_input=_mapping(record.get("observation_input")),
        action_mask=_mapping(record.get("action_mask")),
        public_history_trace=[],
    )
    return [float(value) for value in candidate_feature_vector(row, action)]


def _feature_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        key: _float_summary(
            [
                _float(row.get(key))
                for row in rows
                if _optional_float(row.get(key)) is not None
            ]
        )
        for key in _FEATURE_SUMMARY_KEYS
    }


def _ledger_entry(
    *,
    stage: str,
    passed: bool,
    metrics: Mapping[str, object],
    blockers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": V104_LEDGER_SCHEMA_VERSION,
        "policy": MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        "stage": stage,
        "passed": bool(passed),
        "runtime_promotion_allowed": False,
        "metrics": dict(metrics),
        "blocker_count": len(blockers),
        "blockers": [dict(item) for item in blockers],
    }


def _optional_action(value: object) -> str | None:
    if isinstance(value, str) and value and value != "none":
        return value
    return None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _int_counter(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        if isinstance(count, int) and not isinstance(count, bool):
            counter[str(key)] = int(count)
    return counter


def _float(value: object, *, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else default


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        result.append(float(item))
    return result


def _float_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": 0.0,
            "p25": None,
            "p50": None,
            "p75": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": _round(ordered[0]),
        "max": _round(ordered[-1]),
        "mean": _round(sum(ordered) / float(len(ordered))),
        "p25": _round(_quantile(ordered, 0.25)),
        "p50": _round(_quantile(ordered, 0.50)),
        "p75": _round(_quantile(ordered, 0.75)),
    }


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(sorted(counts.items()), key=lambda item: (item[1], item[0]))
    return {"key": key, "count": int(count), "share": _round(count / float(total))}


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _round(value: float | None) -> float:
    return round(float(value or 0.0), 6)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, entries: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    main()
