from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.cli.evaluate import parse_seed_selection
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.learned_policy import (
    MIND_RUNTIME_MODE_AUTONOMOUS_ONLINE,
    MIND_RUNTIME_MODE_GUARDED,
    MIND_RUNTIME_MODES,
    load_learned_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare heuristic and learned policies on summary-only seeds."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--enable-mind",
        action="store_true",
        help="Required to run learned-policy inference.",
    )
    parser.add_argument("--seeds", help="Comma-separated seed list.")
    parser.add_argument("--seed", action="append", type=int, help="Add one seed.")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--mind-runtime-mode",
        choices=sorted(MIND_RUNTIME_MODES),
        default=MIND_RUNTIME_MODE_GUARDED,
        help="Runtime mode for the primary learned-policy evaluation.",
    )
    parser.add_argument(
        "--compare-runtime-mode",
        choices=sorted(MIND_RUNTIME_MODES),
        action="append",
        default=[],
        help="Evaluate an additional runtime mode on the same seeds and ticks.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--experiment-ledger-output",
        type=Path,
        help="Append a compact policy-evaluation ledger entry to this JSONL file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = load_learned_policy(
        args.artifact,
        enable_mind=args.enable_mind,
        runtime_mode=args.mind_runtime_mode,
    )
    seeds = parse_seed_selection(args.seed, args.seeds)
    primary_eval_kwargs: dict[str, object] = {
        "seeds": seeds,
        "ticks": args.ticks,
    }
    if args.mind_runtime_mode == MIND_RUNTIME_MODE_AUTONOMOUS_ONLINE:
        primary_eval_kwargs["learned_policy_factory"] = lambda: load_learned_policy(
            args.artifact,
            enable_mind=args.enable_mind,
            runtime_mode=args.mind_runtime_mode,
        )
        primary_eval_kwargs["learned_policy_name"] = policy.policy_id
    else:
        primary_eval_kwargs["learned_policy"] = policy
    report = compare_heuristic_and_learned(**primary_eval_kwargs)
    report["protocol"]["runtime_mode"] = args.mind_runtime_mode
    runtime_mode_comparisons = _dedupe_runtime_modes(args.compare_runtime_mode)
    report["protocol"]["runtime_mode_comparisons"] = runtime_mode_comparisons
    report["runtime_mode_evaluation_matrix"] = [
        {
            "runtime_mode": runtime_mode,
            "evaluation": compare_heuristic_and_learned(
                learned_policy_factory=(
                    lambda runtime_mode=runtime_mode: load_learned_policy(
                        args.artifact,
                        enable_mind=args.enable_mind,
                        runtime_mode=runtime_mode,
                    )
                ),
                learned_policy_name=f"{policy.policy_id}:{runtime_mode}",
                seeds=seeds,
                ticks=args.ticks,
            ),
        }
        for runtime_mode in runtime_mode_comparisons
    ]
    ledger_entry = _build_policy_eval_ledger_entry(
        report,
        artifact_path=args.artifact,
        seeds=seeds,
        ticks=args.ticks,
    )
    report["experiment_ledger_entry"] = ledger_entry
    if args.experiment_ledger_output is not None:
        _append_ledger_entry(args.experiment_ledger_output, ledger_entry)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def _dedupe_runtime_modes(runtime_modes: list[str]) -> list[str]:
    seen: set[str] = set()
    resolved: list[str] = []
    for runtime_mode in runtime_modes:
        if runtime_mode in seen:
            continue
        seen.add(runtime_mode)
        resolved.append(runtime_mode)
    return resolved


def _build_policy_eval_ledger_entry(
    report: Mapping[str, object],
    *,
    artifact_path: Path,
    seeds: list[int],
    ticks: int,
) -> dict[str, object]:
    gates = _mapping(report.get("mind_v1_gates"))
    learned = _mapping(report.get("learned"))
    aggregate = _mapping(learned.get("aggregate"))
    diagnostics = _mapping(aggregate.get("policy_diagnostics"))
    comparison = _mapping(report.get("comparison"))
    guard_rate = _float_or_none(diagnostics.get("guard_intervention_rate"))
    delegate_rate = _float_or_none(diagnostics.get("heuristic_delegate_rate"))
    return {
        "schema_version": "mind_policy_eval_ledger_v1",
        "artifact_path": str(artifact_path),
        "runtime_mode": _mapping(report.get("protocol")).get("runtime_mode"),
        "runtime_mode_comparisons": _runtime_mode_ledger_summary(
            report.get("runtime_mode_evaluation_matrix")
        ),
        "seeds": list(seeds),
        "ticks": ticks,
        "status": gates.get("status"),
        "hard_guard": guard_rate,
        "heuristic_delegate": delegate_rate,
        "total_fallback": (
            round(guard_rate + delegate_rate, 4)
            if guard_rate is not None and delegate_rate is not None
            else None
        ),
        "alive_delta": _float_or_none(comparison.get("alive_agents_mean_delta")),
        "births_delta": _float_or_none(comparison.get("births_mean_delta")),
    }


def _runtime_mode_ledger_summary(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list):
        return []
    summaries: list[dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        evaluation = _mapping(entry.get("evaluation"))
        learned = _mapping(evaluation.get("learned"))
        aggregate = _mapping(learned.get("aggregate"))
        diagnostics = _mapping(aggregate.get("policy_diagnostics"))
        comparison = _mapping(evaluation.get("comparison"))
        update_trace = _mapping(aggregate.get("policy_update_trace"))
        guard_rate = _float_or_none(diagnostics.get("guard_intervention_rate"))
        delegate_rate = _float_or_none(diagnostics.get("heuristic_delegate_rate"))
        summaries.append(
            {
                "runtime_mode": entry.get("runtime_mode"),
                "status": _mapping(evaluation.get("mind_v1_gates")).get("status"),
                "hard_guard": guard_rate,
                "heuristic_delegate": delegate_rate,
                "total_fallback": (
                    round(guard_rate + delegate_rate, 4)
                    if guard_rate is not None and delegate_rate is not None
                    else None
                ),
                "alive_delta": _float_or_none(
                    comparison.get("alive_agents_mean_delta")
                ),
                "births_delta": _float_or_none(
                    comparison.get("births_mean_delta")
                ),
                "policy_update_trace_count": update_trace.get("record_count"),
                "max_online_update_count": update_trace.get(
                    "max_online_update_count"
                ),
            }
        )
    return summaries


def _append_ledger_entry(path: Path, entry: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True, allow_nan=False) + "\n")


def _mapping(payload: object) -> Mapping[str, object]:
    return payload if isinstance(payload, Mapping) else {}


def _float_or_none(payload: object) -> float | None:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return None
    return round(float(payload), 4)


if __name__ == "__main__":
    main()
