from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path

from evolution_sim.cli.evaluate import parse_seed_selection
from evolution_sim.mind.evaluation_harness import (
    _aggregate_runs,
    _comparison_delta,
    _mind_v3_policy,
    _parse_fixture_names,
    _run_once,
    mind_v3_fixture_gate_config,
    mind_v3_fixture_gate_status,
    run_controlled_fixture_policy_suite,
)
from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.learned_policy import (
    MIND_RUNTIME_MODE_AUTONOMOUS,
    MIND_RUNTIME_MODES,
    load_learned_policy,
)
from evolution_sim.mind.v3_neural import load_mind_v3_neural_artifact

MIND_V3_LABELED_IQL_SLICE_SCHEMA_VERSION = "mind_v3_labeled_iql_slice_v1"
MIND_V3_LABELED_IQL_ACCEPTANCE_POLICY = (
    "mind_v3_labeled_iql_broad_carrion_acceptance_v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a labeled torch/IQL Mind artifact beside the current "
            "Mind v3 linear baseline and optional anchored-neural baseline."
        )
    )
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument(
        "--enable-mind",
        action="store_true",
        help="Required to load learned-policy artifacts for inference.",
    )
    parser.add_argument(
        "--mind-runtime-mode",
        choices=sorted(MIND_RUNTIME_MODES),
        default=MIND_RUNTIME_MODE_AUTONOMOUS,
        help=(
            "Runtime mode for the labeled torch/IQL candidate. Acceptance "
            "expects autonomous mode with no heuristic runtime actions."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="5,13,19,29",
        help="Comma-separated broad-world seed list.",
    )
    parser.add_argument("--seed", action="append", type=int, help="Add one seed.")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--founder-template",
        type=Path,
        help="Optional Mind v3 founder template for the linear and anchored baselines.",
    )
    parser.add_argument(
        "--anchored-neural-artifact",
        type=Path,
        help="Optional anchored Mind v3 neural artifact to evaluate beside linear.",
    )
    parser.add_argument(
        "--fixture-suite",
        choices=["basic"],
        default="basic",
        help="Controlled fixture suite for the acceptance slice.",
    )
    parser.add_argument(
        "--fixture-names",
        default="carrion_only",
        help="Comma-separated fixture subset. Defaults to the carrion blocker.",
    )
    parser.add_argument(
        "--fixture-seeds",
        help="Comma-separated fixture seed list. Defaults to --seeds.",
    )
    parser.add_argument(
        "--fixture-ticks",
        type=int,
        help="Tick horizon for fixtures. Defaults to --ticks.",
    )
    parser.add_argument(
        "--fixture-min-alive",
        type=float,
        default=1.0,
        help="Minimum alive-agent mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-births",
        type=float,
        default=0.0,
        help="Minimum births mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-mixed-stable-births",
        type=float,
        default=0.0,
        help="Minimum births mean required specifically on mixed_stable.",
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
        help="Minimum biologically-ready terminal agent mean for every fixture.",
    )
    parser.add_argument(
        "--max-broad-alive-regression-vs-linear",
        type=float,
        default=1.0,
        help="Maximum allowed candidate alive-agent mean regression vs linear.",
    )
    parser.add_argument(
        "--min-broad-births-delta-vs-linear",
        type=float,
        default=0.0,
        help="Minimum allowed candidate births mean delta vs linear.",
    )
    parser.add_argument(
        "--max-dominant-action-share",
        type=float,
        default=0.5,
        help="Maximum candidate dominant requested-action share.",
    )
    parser.add_argument(
        "--max-heuristic-action-source-count",
        type=int,
        default=0,
        help="Maximum heuristic runtime action-source count for the candidate.",
    )
    parser.add_argument(
        "--carrion-fixture-name",
        default="carrion_only",
        help="Fixture used for carrion acceptance movement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-labeled-iql-slice-report.json"),
    )
    parser.add_argument(
        "--experiment-ledger-output",
        type=Path,
        help="Append a compact labeled-IQL slice ledger entry to this JSONL file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seed_selection(args.seed, args.seeds)
    fixture_seeds = (
        parse_seed_selection(None, args.fixture_seeds)
        if args.fixture_seeds is not None
        else list(seeds)
    )
    fixture_ticks = int(args.fixture_ticks) if args.fixture_ticks is not None else args.ticks
    fixture_names = _parse_fixture_names(
        args.fixture_names,
        suite=args.fixture_suite,
    )
    founder_template = (
        load_mind_v3_founder_template(args.founder_template)
        if args.founder_template is not None
        else None
    )
    anchored_neural_artifact = (
        load_mind_v3_neural_artifact(args.anchored_neural_artifact)
        if args.anchored_neural_artifact is not None
        else None
    )
    candidate_policy_factory = _candidate_policy_factory(
        artifact_path=args.candidate_artifact,
        enable_mind=bool(args.enable_mind),
        runtime_mode=args.mind_runtime_mode,
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

    evaluations: dict[str, dict[str, object]] = {
        "linear_default": _evaluate_policy_slice(
            policy_key="linear_default",
            policy_name="mind_v3_linear_default",
            seeds=seeds,
            ticks=args.ticks,
            fixture_suite=args.fixture_suite,
            fixture_names=fixture_names,
            fixture_seeds=fixture_seeds,
            fixture_ticks=fixture_ticks,
            fixture_config=fixture_config,
            policy_factory=lambda seed: _mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=None,
            ),
            fixture_policy_factory=lambda fixture_name, seed: _mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=None,
            ),
        ),
        "candidate": _evaluate_policy_slice(
            policy_key="candidate",
            policy_name="labeled_torch_iql_candidate",
            seeds=seeds,
            ticks=args.ticks,
            fixture_suite=args.fixture_suite,
            fixture_names=fixture_names,
            fixture_seeds=fixture_seeds,
            fixture_ticks=fixture_ticks,
            fixture_config=fixture_config,
            policy_factory=lambda seed: candidate_policy_factory(),
            fixture_policy_factory=lambda fixture_name, seed: candidate_policy_factory(),
        ),
    }
    if anchored_neural_artifact is not None:
        evaluations["anchored_neural"] = _evaluate_policy_slice(
            policy_key="anchored_neural",
            policy_name="mind_v3_anchored_neural",
            seeds=seeds,
            ticks=args.ticks,
            fixture_suite=args.fixture_suite,
            fixture_names=fixture_names,
            fixture_seeds=fixture_seeds,
            fixture_ticks=fixture_ticks,
            fixture_config=fixture_config,
            policy_factory=lambda seed: _mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=anchored_neural_artifact,
            ),
            fixture_policy_factory=lambda fixture_name, seed: _mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=anchored_neural_artifact,
            ),
        )

    report: dict[str, object] = {
        "schema_version": MIND_V3_LABELED_IQL_SLICE_SCHEMA_VERSION,
        "protocol": {
            "candidate_artifact": str(args.candidate_artifact),
            "candidate_runtime_mode": args.mind_runtime_mode,
            "enable_mind": bool(args.enable_mind),
            "founder_template": (
                str(args.founder_template)
                if args.founder_template is not None
                else None
            ),
            "anchored_neural_artifact": (
                str(args.anchored_neural_artifact)
                if args.anchored_neural_artifact is not None
                else None
            ),
            "seeds": list(seeds),
            "ticks": int(args.ticks),
            "fixture_suite": args.fixture_suite,
            "fixture_names": fixture_names,
            "fixture_seeds": list(fixture_seeds),
            "fixture_ticks": int(fixture_ticks),
            "fixture_gate_config": fixture_config,
        },
        "evaluations": evaluations,
        "comparison": _build_comparison(evaluations),
    }
    report["acceptance_gate"] = build_labeled_iql_acceptance_gate(
        evaluations=evaluations,
        carrion_fixture_name=args.carrion_fixture_name,
        max_broad_alive_regression_vs_linear=float(
            args.max_broad_alive_regression_vs_linear
        ),
        min_broad_births_delta_vs_linear=float(args.min_broad_births_delta_vs_linear),
        max_dominant_action_share=float(args.max_dominant_action_share),
        max_heuristic_action_source_count=int(args.max_heuristic_action_source_count),
    )
    report["experiment_ledger_entry"] = _build_ledger_entry(report)
    if args.experiment_ledger_output is not None:
        _append_ledger_entry(args.experiment_ledger_output, report["experiment_ledger_entry"])

    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def build_labeled_iql_acceptance_gate(
    *,
    evaluations: Mapping[str, object],
    carrion_fixture_name: str = "carrion_only",
    max_broad_alive_regression_vs_linear: float = 1.0,
    min_broad_births_delta_vs_linear: float = 0.0,
    max_dominant_action_share: float = 0.5,
    max_heuristic_action_source_count: int = 0,
) -> dict[str, object]:
    linear_broad = _evaluation_broad_aggregate(evaluations, "linear_default")
    candidate_broad = _evaluation_broad_aggregate(evaluations, "candidate")
    candidate_vs_linear_per_seed = _candidate_vs_linear_per_seed(evaluations)
    linear_fixture_gate = _evaluation_fixture_gate(evaluations, "linear_default")
    candidate_fixture_gate = _evaluation_fixture_gate(evaluations, "candidate")
    linear_alive = _float_metric(linear_broad.get("alive_agents_mean"))
    candidate_alive = _float_metric(candidate_broad.get("alive_agents_mean"))
    candidate_alive_regression = round(linear_alive - candidate_alive, 4)
    linear_births = _float_metric(linear_broad.get("births_mean"))
    candidate_births = _float_metric(candidate_broad.get("births_mean"))
    candidate_births_delta = round(candidate_births - linear_births, 4)
    candidate_dominant_share = _float_metric(
        candidate_broad.get("dominant_requested_action_share")
    )
    candidate_heuristic_actions = _int_metric(
        candidate_broad.get("heuristic_action_source_count")
    )
    min_seed_alive_delta = _min_seed_delta(
        candidate_vs_linear_per_seed,
        field="alive_delta_vs_linear",
    )
    min_seed_births_delta = _min_seed_delta(
        candidate_vs_linear_per_seed,
        field="births_delta_vs_linear",
    )
    linear_blocker_count = _blocker_count(linear_fixture_gate)
    candidate_blocker_count = _blocker_count(candidate_fixture_gate)
    candidate_carrion_alive = _fixture_metric(
        candidate_fixture_gate,
        fixture_name=carrion_fixture_name,
        metric_name="alive_agents_mean",
    )

    blockers: list[dict[str, object]] = []
    if candidate_alive_regression > max_broad_alive_regression_vs_linear:
        blockers.append(
            _acceptance_blocker(
                reason="broad_alive_regression_vs_linear",
                metric="candidate_alive_regression_vs_linear",
                value=candidate_alive_regression,
                limit=max_broad_alive_regression_vs_linear,
            )
        )
    if candidate_births_delta < min_broad_births_delta_vs_linear:
        blockers.append(
            _acceptance_blocker(
                reason="broad_births_regression_vs_linear",
                metric="candidate_births_delta_vs_linear",
                value=candidate_births_delta,
                limit=min_broad_births_delta_vs_linear,
            )
        )
    if candidate_dominant_share > max_dominant_action_share:
        blockers.append(
            _acceptance_blocker(
                reason="dominant_action_share_too_high",
                metric="candidate_dominant_requested_action_share",
                value=candidate_dominant_share,
                limit=max_dominant_action_share,
            )
        )
    if candidate_heuristic_actions > max_heuristic_action_source_count:
        blockers.append(
            _acceptance_blocker(
                reason="heuristic_runtime_actions_present",
                metric="candidate_heuristic_action_source_count",
                value=float(candidate_heuristic_actions),
                limit=float(max_heuristic_action_source_count),
            )
        )
    for seed_delta in candidate_vs_linear_per_seed:
        seed = _int_metric(seed_delta.get("seed"))
        alive_delta = _float_metric(seed_delta.get("alive_delta_vs_linear"))
        births_delta = _float_metric(seed_delta.get("births_delta_vs_linear"))
        if alive_delta < 0.0:
            blockers.append(
                _acceptance_blocker(
                    reason=f"open_seed_{seed}_alive_regression_vs_linear",
                    metric="candidate_seed_alive_delta_vs_linear",
                    value=alive_delta,
                    limit=0.0,
                )
            )
        if births_delta < 0.0:
            blockers.append(
                _acceptance_blocker(
                    reason=f"open_seed_{seed}_birth_regression_vs_linear",
                    metric="candidate_seed_births_delta_vs_linear",
                    value=births_delta,
                    limit=0.0,
                )
            )
    carrion_moved = (
        candidate_carrion_alive > 0.0
        or candidate_blocker_count < linear_blocker_count
    )
    if not carrion_moved:
        blockers.append(
            _acceptance_blocker(
                reason="carrion_fixture_not_moved",
                metric="candidate_carrion_alive_or_blocker_reduction",
                value=0.0,
                limit=1.0,
            )
        )

    return {
        "policy": MIND_V3_LABELED_IQL_ACCEPTANCE_POLICY,
        "passed": not blockers,
        "blockers": blockers,
        "criteria": {
            "max_broad_alive_regression_vs_linear": float(
                max_broad_alive_regression_vs_linear
            ),
            "min_broad_births_delta_vs_linear": float(
                min_broad_births_delta_vs_linear
            ),
            "max_dominant_action_share": float(max_dominant_action_share),
            "max_heuristic_action_source_count": int(
                max_heuristic_action_source_count
            ),
            "carrion_fixture_name": carrion_fixture_name,
            "carrion_acceptance": (
                "candidate_alive_mean_gt_zero_or_blocker_count_below_linear"
            ),
        },
        "metrics": {
            "linear_broad_alive_mean": linear_alive,
            "candidate_broad_alive_mean": candidate_alive,
            "candidate_alive_regression_vs_linear": candidate_alive_regression,
            "linear_broad_births_mean": linear_births,
            "candidate_broad_births_mean": candidate_births,
            "candidate_births_delta_vs_linear": candidate_births_delta,
            "candidate_dominant_requested_action_share": candidate_dominant_share,
            "candidate_heuristic_action_source_count": candidate_heuristic_actions,
            "candidate_min_seed_alive_delta_vs_linear": min_seed_alive_delta,
            "candidate_min_seed_births_delta_vs_linear": min_seed_births_delta,
            "candidate_vs_linear_per_seed": candidate_vs_linear_per_seed,
            "linear_fixture_blocker_count": linear_blocker_count,
            "candidate_fixture_blocker_count": candidate_blocker_count,
            "candidate_fixture_blocker_count_delta_vs_linear": (
                candidate_blocker_count - linear_blocker_count
            ),
            "candidate_carrion_alive_agents_mean": candidate_carrion_alive,
        },
    }


def _candidate_policy_factory(
    *,
    artifact_path: Path,
    enable_mind: bool,
    runtime_mode: str,
) -> Callable[[], object]:
    return lambda: load_learned_policy(
        artifact_path,
        enable_mind=enable_mind,
        runtime_mode=runtime_mode,
    )


def _evaluate_policy_slice(
    *,
    policy_key: str,
    policy_name: str,
    seeds: list[int],
    ticks: int,
    fixture_suite: str,
    fixture_names: list[str] | None,
    fixture_seeds: list[int],
    fixture_ticks: int,
    fixture_config: Mapping[str, object],
    policy_factory: Callable[[int], object | None],
    fixture_policy_factory: Callable[[str, int], object | None],
) -> dict[str, object]:
    broad_runs = [
        _run_once(
            seed=seed,
            ticks=ticks,
            policy=policy_factory(seed),
        )
        for seed in seeds
    ]
    broad_aggregate = _aggregate_runs(broad_runs)
    fixture_report = run_controlled_fixture_policy_suite(
        suite=fixture_suite,
        fixture_names=fixture_names,
        seeds=fixture_seeds,
        ticks=fixture_ticks,
        learned_policy_factory=fixture_policy_factory,
        learned_policy_key=policy_key,
        learned_policy_name=policy_name,
    )
    fixture_gate = mind_v3_fixture_gate_status(
        fixture_suite=fixture_report,
        fixture_config=fixture_config,
    )
    return {
        "policy_key": policy_key,
        "policy_name": policy_name,
        "broad": {
            "runs": broad_runs,
            "aggregate": broad_aggregate,
        },
        "fixture_suite": fixture_report,
        "fixture_gate": fixture_gate,
    }


def _build_comparison(
    evaluations: Mapping[str, object],
) -> dict[str, object]:
    linear = _evaluation_broad_aggregate(evaluations, "linear_default")
    comparison: dict[str, object] = {}
    for policy_key in ("candidate", "anchored_neural"):
        policy_eval = _mapping(evaluations.get(policy_key))
        if not policy_eval:
            continue
        comparison[f"{policy_key}_vs_linear"] = _comparison_delta(
            heuristic=linear,
            mind_v3=_evaluation_broad_aggregate(evaluations, policy_key),
        )
    return comparison


def _build_ledger_entry(report: Mapping[str, object]) -> dict[str, object]:
    protocol = _mapping(report.get("protocol"))
    gate = _mapping(report.get("acceptance_gate"))
    metrics = _mapping(gate.get("metrics"))
    return {
        "schema_version": "mind_v3_labeled_iql_slice_ledger_v1",
        "candidate_artifact": protocol.get("candidate_artifact"),
        "candidate_runtime_mode": protocol.get("candidate_runtime_mode"),
        "seeds": protocol.get("seeds"),
        "ticks": protocol.get("ticks"),
        "fixture_names": protocol.get("fixture_names"),
        "fixture_seeds": protocol.get("fixture_seeds"),
        "fixture_ticks": protocol.get("fixture_ticks"),
        "status": "pass" if gate.get("passed") else "fail",
        "blocker_count": len(gate.get("blockers", []))
        if isinstance(gate.get("blockers"), list)
        else None,
        "candidate_alive_regression_vs_linear": metrics.get(
            "candidate_alive_regression_vs_linear"
        ),
        "candidate_births_delta_vs_linear": metrics.get(
            "candidate_births_delta_vs_linear"
        ),
        "candidate_dominant_requested_action_share": metrics.get(
            "candidate_dominant_requested_action_share"
        ),
        "candidate_heuristic_action_source_count": metrics.get(
            "candidate_heuristic_action_source_count"
        ),
        "candidate_min_seed_alive_delta_vs_linear": metrics.get(
            "candidate_min_seed_alive_delta_vs_linear"
        ),
        "candidate_min_seed_births_delta_vs_linear": metrics.get(
            "candidate_min_seed_births_delta_vs_linear"
        ),
        "candidate_fixture_blocker_count_delta_vs_linear": metrics.get(
            "candidate_fixture_blocker_count_delta_vs_linear"
        ),
        "candidate_carrion_alive_agents_mean": metrics.get(
            "candidate_carrion_alive_agents_mean"
        ),
    }


def _append_ledger_entry(path: Path, entry: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True, allow_nan=False) + "\n")


def _evaluation_broad_aggregate(
    evaluations: Mapping[str, object],
    policy_key: str,
) -> Mapping[str, object]:
    evaluation = _mapping(evaluations.get(policy_key))
    broad = _mapping(evaluation.get("broad"))
    return _mapping(broad.get("aggregate"))


def _evaluation_fixture_gate(
    evaluations: Mapping[str, object],
    policy_key: str,
) -> Mapping[str, object]:
    evaluation = _mapping(evaluations.get(policy_key))
    return _mapping(evaluation.get("fixture_gate"))


def _evaluation_broad_runs(
    evaluations: Mapping[str, object],
    policy_key: str,
) -> list[Mapping[str, object]]:
    evaluation = _mapping(evaluations.get(policy_key))
    broad = _mapping(evaluation.get("broad"))
    runs = broad.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, Mapping)]


def _candidate_vs_linear_per_seed(
    evaluations: Mapping[str, object],
) -> list[dict[str, object]]:
    linear_runs = {
        _int_metric(run.get("seed")): run
        for run in _evaluation_broad_runs(evaluations, "linear_default")
    }
    deltas: list[dict[str, object]] = []
    for candidate_run in _evaluation_broad_runs(evaluations, "candidate"):
        seed = _int_metric(candidate_run.get("seed"))
        linear_run = linear_runs.get(seed)
        if linear_run is None:
            continue
        candidate_alive = _int_metric(candidate_run.get("alive_agents"))
        linear_alive = _int_metric(linear_run.get("alive_agents"))
        candidate_births = _int_metric(candidate_run.get("births"))
        linear_births = _int_metric(linear_run.get("births"))
        deltas.append(
            {
                "seed": seed,
                "candidate_alive_agents": candidate_alive,
                "linear_alive_agents": linear_alive,
                "alive_delta_vs_linear": candidate_alive - linear_alive,
                "candidate_births": candidate_births,
                "linear_births": linear_births,
                "births_delta_vs_linear": candidate_births - linear_births,
            }
        )
    return sorted(deltas, key=lambda item: _int_metric(item.get("seed")))


def _min_seed_delta(
    deltas: list[Mapping[str, object]],
    *,
    field: str,
) -> float | None:
    if not deltas:
        return None
    return min(_float_metric(delta.get(field)) for delta in deltas)


def _fixture_metric(
    fixture_gate: Mapping[str, object],
    *,
    fixture_name: str,
    metric_name: str,
) -> float:
    per_fixture = _mapping(fixture_gate.get("per_fixture"))
    fixture = _mapping(per_fixture.get(fixture_name))
    metrics = _mapping(fixture.get("metrics"))
    return _float_metric(metrics.get(metric_name))


def _blocker_count(fixture_gate: Mapping[str, object]) -> int:
    blockers = fixture_gate.get("blockers")
    return len(blockers) if isinstance(blockers, list) else 0


def _acceptance_blocker(
    *,
    reason: str,
    metric: str,
    value: float,
    limit: float,
) -> dict[str, object]:
    return {
        "reason": reason,
        "metric": metric,
        "value": round(float(value), 4),
        "limit": round(float(limit), 4),
    }


def _mapping(payload: object) -> Mapping[str, object]:
    return payload if isinstance(payload, Mapping) else {}


def _float_metric(payload: object) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return 0.0
    return round(float(payload), 4)


def _int_metric(payload: object) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        return 0
    return int(payload)


if __name__ == "__main__":
    main()
