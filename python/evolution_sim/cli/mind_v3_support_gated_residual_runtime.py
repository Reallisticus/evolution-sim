from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.mind.broad_transfer_residual_audit import V98_STRICT_EXCLUDED_SEEDS
from evolution_sim.mind.broad_branch_residual_distillation_example import load_json_report
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
    aggregate_support_gated_residual_runtime_diagnostics,
    build_branch_replay_feasibility_report,
    build_support_gated_residual_runtime_artifact,
    support_gated_residual_runtime_diagnostics,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

V103_RUNTIME_REPORT_SCHEMA_VERSION = (
    "mind_v3_v103_support_gated_residual_runtime_report_v1"
)
V103_SHADOW_STRICT_REPORT_SCHEMA_VERSION = (
    "mind_v3_v103_shadow_strict_report_v1"
)
V103_NONSTRICT_LIVE_REPORT_SCHEMA_VERSION = (
    "mind_v3_v103_nonstrict_live_feasibility_report_v1"
)
V103_LEDGER_SCHEMA_VERSION = "mind_v3_v103_support_gated_residual_ledger_v1"
V103_STRICT_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
V103_NONSTRICT_LIVE_SEEDS = (2, 3, 7, 11, 17, 23, 31, 47, 53, 59)
V103_DEFAULT_TICKS = 120
V103_MAX_DOMINANT_SHADOW_OVERRIDE_ACTION_SHARE = 0.50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate the v103 opt-in support-gated residual runtime "
            "over the existing linear Mind v3 controller. This is a feasibility "
            "gate only; it never runs strict live promotion."
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
        "--artifact-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v103-support-gated-residual-runtime-artifact.json"
        ),
    )
    parser.add_argument(
        "--branch-output",
        type=Path,
        default=Path("output/mind/mind-v3-v103-branch-replay-feasibility.json"),
    )
    parser.add_argument(
        "--shadow-output",
        type=Path,
        default=Path("output/mind/mind-v3-v103-shadow-strict-report.json"),
    )
    parser.add_argument(
        "--live-output",
        type=Path,
        default=Path("output/mind/mind-v3-v103-nonstrict-live-feasibility.json"),
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v103-support-gated-residual-runtime-ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v103-support-gated-residual-runtime-report.json"
        ),
    )
    parser.add_argument(
        "--strict-seeds",
        default=",".join(str(seed) for seed in V103_STRICT_BROAD_SEEDS),
        help="Strict broad seeds for shadow-only evaluation.",
    )
    parser.add_argument(
        "--non-strict-seeds",
        default=",".join(str(seed) for seed in V103_NONSTRICT_LIVE_SEEDS),
        help="Non-strict seeds for live feasibility after branch/shadow gates pass.",
    )
    parser.add_argument("--ticks", type=int, default=V103_DEFAULT_TICKS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    v102_report = load_json_report(args.v102_report)
    v102_artifact = load_json_report(args.v102_artifact)
    v99_report = load_json_report(args.v99_report)
    v100_report = load_json_report(args.v100_report)
    strict_seeds = _parse_seeds(args.strict_seeds)
    non_strict_seeds = _parse_seeds(args.non_strict_seeds)
    _reject_strict_leakage(non_strict_seeds)

    runtime_artifact, threshold_report = build_support_gated_residual_runtime_artifact(
        v102_report=v102_report,
        v102_artifact=v102_artifact,
        v99_report=v99_report,
        v100_report=v100_report,
    )
    _write_json(args.artifact_output, runtime_artifact)

    ledger_entries = []
    branch_report = build_branch_replay_feasibility_report(
        runtime_artifact=runtime_artifact,
        v99_report=v99_report,
        v100_report=v100_report,
    )
    _write_json(args.branch_output, branch_report)
    ledger_entries.append(
        _ledger_entry(
            stage="branch_replay",
            passed=bool(branch_report["v103_branch_replay_feasibility_passed"]),
            metrics=branch_report["summary"],
            blockers=branch_report["safety_gate"]["blockers"],
        )
    )

    shadow_report = None
    live_report = None
    if branch_report["v103_branch_replay_feasibility_passed"] is True:
        shadow_report = build_shadow_strict_report(
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
            live_report = build_nonstrict_live_feasibility_report(
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
        branch_report["v103_branch_replay_feasibility_passed"] is True
        and shadow_report is not None
        and shadow_report["shadow_gate"]["passed"] is True
        and live_report is not None
        and live_report["live_feasibility_gate"]["passed"] is True
    )
    report = {
        "schema_version": V103_RUNTIME_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_promotion_allowed": False,
        "strict_live_promotion_executed": False,
        "artifact_output": str(args.artifact_output),
        "branch_replay_output": str(args.branch_output),
        "shadow_strict_output": str(args.shadow_output) if shadow_report else None,
        "nonstrict_live_output": str(args.live_output) if live_report else None,
        "ledger_output": str(args.ledger_output),
        "threshold_report": threshold_report,
        "branch_replay_passed": bool(
            branch_report["v103_branch_replay_feasibility_passed"]
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
        "v103_runtime_feasibility_passed": final_passed,
        "decision": "accepted_for_v104_strict_promotion" if final_passed else "rejected_or_stopped",
    }
    _write_json(args.report_output, report)
    _write_jsonl(args.ledger_output, ledger_entries)

    print(f"v103_runtime_artifact={args.artifact_output}")
    print(f"v103_branch_replay_report={args.branch_output}")
    print(
        "v103_branch_replay_passed="
        f"{branch_report['v103_branch_replay_feasibility_passed']}"
    )
    if shadow_report is not None:
        print(f"v103_shadow_strict_report={args.shadow_output}")
        print(f"v103_shadow_strict_passed={shadow_report['shadow_gate']['passed']}")
    if live_report is not None:
        print(f"v103_nonstrict_live_report={args.live_output}")
        print(
            "v103_nonstrict_live_feasibility_passed="
            f"{live_report['live_feasibility_gate']['passed']}"
        )
    print(f"v103_runtime_report={args.report_output}")
    print(f"v103_ledger={args.ledger_output}")
    print(f"v103_runtime_feasibility_passed={final_passed}")


def build_shadow_strict_report(
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
        "schema_version": V103_SHADOW_STRICT_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        "runtime_mode": "shadow",
        "strict_live_promotion_executed": False,
        "runtime_promotion_allowed": False,
        "seeds": list(seeds),
        "ticks": int(ticks),
        "runs": runs,
        "aggregate": aggregate,
        "shadow_gate": gate,
        "v103_shadow_strict_passed": bool(gate["passed"]),
    }


def build_nonstrict_live_feasibility_report(
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
    gate = _live_feasibility_gate(
        linear_runs=linear_runs,
        residual_runs=residual_runs,
        linear_aggregate=linear_aggregate,
        residual_aggregate=residual_aggregate,
        delta=delta,
    )
    return {
        "schema_version": V103_NONSTRICT_LIVE_REPORT_SCHEMA_VERSION,
        "policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
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
        "delta": delta,
        "live_feasibility_gate": gate,
        "v103_nonstrict_live_feasibility_passed": bool(gate["passed"]),
    }


def _run_policy(
    *,
    seed: int,
    ticks: int,
    runtime_artifact: Mapping[str, object] | None,
    runtime_mode: str = "live",
) -> dict[str, object]:
    policy = MindV3EvolutionPolicy(
        seed=seed,
        support_residual_artifact=runtime_artifact,
        support_residual_runtime_mode=runtime_mode,
    )
    world = SimulationWorld(
        WorldConfig(seed=seed, max_ticks=ticks),
        policy=policy,
    )
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    summary = result.summary
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
    unsupported_requested = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
        and record.get("action_valid") is False
    )
    unsupported_resolved = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
        and record.get("resolution_action_valid") is False
    )
    return {
        "seed": int(seed),
        "ticks": int(ticks),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "trajectory_record_count": len(world.trajectory_records),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(unsupported_requested),
        "unsupported_resolved_action_count": int(unsupported_resolved),
        "support_residual_diagnostics": support_gated_residual_runtime_diagnostics(
            world.policy_decision_diagnostics_records,
        ),
    }


def _aggregate_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    requested_counts: Counter[str] = Counter()
    resolved_counts: Counter[str] = Counter()
    unsupported_requested = 0
    unsupported_resolved = 0
    for run in runs:
        requested_counts.update(_int_counter(run.get("requested_action_counts")))
        resolved_counts.update(_int_counter(run.get("resolved_action_counts")))
        unsupported_requested += int(run.get("unsupported_requested_action_count", 0))
        unsupported_resolved += int(run.get("unsupported_resolved_action_count", 0))
    return {
        "run_count": len(runs),
        "alive_agents_mean": _round(
            _mean([_float(run.get("alive_agents")) for run in runs])
        ),
        "births_mean": _round(_mean([_float(run.get("births")) for run in runs])),
        "deaths_mean": _round(_mean([_float(run.get("deaths")) for run in runs])),
        "trajectory_record_count": sum(
            int(run.get("trajectory_record_count", 0)) for run in runs
        ),
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_counts.items())),
        "unsupported_requested_action_count": int(unsupported_requested),
        "unsupported_resolved_action_count": int(unsupported_resolved),
        "support_residual_diagnostics": aggregate_support_gated_residual_runtime_diagnostics(
            runs
        ),
    }


def _shadow_gate(diagnostics: Mapping[str, object]) -> dict[str, object]:
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

    if int(diagnostics.get("unsupported_proposed_action_count", 0)) != 0:
        block(
            "unsupported_proposed_action",
            "unsupported_proposed_action_count",
            diagnostics.get("unsupported_proposed_action_count"),
            0,
            "eq",
        )
    if int(diagnostics.get("gate_accepted_override_count", 0)) <= 0:
        block(
            "no_gate_accepted_shadow_overrides",
            "gate_accepted_override_count",
            diagnostics.get("gate_accepted_override_count"),
            0,
            "gt",
        )
    if (
        _float(diagnostics.get("dominant_gate_accepted_override_action_share"))
        > V103_MAX_DOMINANT_SHADOW_OVERRIDE_ACTION_SHARE
    ):
        block(
            "shadow_action_collapse",
            "dominant_gate_accepted_override_action_share",
            diagnostics.get("dominant_gate_accepted_override_action_share"),
            V103_MAX_DOMINANT_SHADOW_OVERRIDE_ACTION_SHARE,
            "le",
        )
    return {
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "runtime_promotion_allowed": False,
    }


def _live_feasibility_gate(
    *,
    linear_runs: Sequence[Mapping[str, object]],
    residual_runs: Sequence[Mapping[str, object]],
    linear_aggregate: Mapping[str, object],
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

    unsupported_total = (
        int(residual_aggregate.get("unsupported_requested_action_count", 0))
        + int(residual_aggregate.get("unsupported_resolved_action_count", 0))
        + int(
            _mapping(residual_aggregate.get("support_residual_diagnostics")).get(
                "unsupported_proposed_action_count", 0
            )
        )
    )
    if unsupported_total != 0:
        block(
            "unsupported_action",
            "unsupported_action_total",
            unsupported_total,
            0,
            "eq",
        )
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
    first_failing = _first_failing_seed_pattern(
        linear_runs=linear_runs,
        residual_runs=residual_runs,
    )
    return {
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "first_failing_seed_action_pattern": first_failing,
        "runtime_promotion_allowed": False,
    }


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
        unsupported_total = (
            int(residual.get("unsupported_requested_action_count", 0))
            + int(residual.get("unsupported_resolved_action_count", 0))
            + int(
                _mapping(residual.get("support_residual_diagnostics")).get(
                    "unsupported_proposed_action_count", 0
                )
            )
        )
        alive_regressed = int(residual.get("alive_agents", 0)) < int(
            linear.get("alive_agents", 0)
        )
        births_regressed = int(residual.get("births", 0)) < int(
            linear.get("births", 0)
        )
        if unsupported_total or alive_regressed or births_regressed:
            diagnostics = _mapping(residual.get("support_residual_diagnostics"))
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


def _ledger_entry(
    *,
    stage: str,
    passed: bool,
    metrics: Mapping[str, object],
    blockers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": V103_LEDGER_SCHEMA_VERSION,
        "policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        "stage": stage,
        "passed": bool(passed),
        "runtime_promotion_allowed": False,
        "metrics": dict(metrics),
        "blocker_count": len(blockers),
        "blockers": [dict(item) for item in blockers],
    }


def _parse_seeds(raw: str) -> list[int]:
    seeds = []
    for item in raw.split(","):
        text = item.strip()
        if text:
            seeds.append(int(text))
    if not seeds:
        raise SystemExit("seed list must not be empty")
    return seeds


def _reject_strict_leakage(seeds: Sequence[int]) -> None:
    leaked = sorted(set(int(seed) for seed in seeds) & set(V98_STRICT_EXCLUDED_SEEDS))
    if leaked:
        raise SystemExit(
            "--non-strict-seeds must not include strict seeds: "
            + ",".join(str(seed) for seed in leaked)
        )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, entries: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int_counter(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        if isinstance(count, int) and not isinstance(count, bool):
            counter[str(key)] = int(count)
    return counter


def _float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    main()
