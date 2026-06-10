from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind import evaluation_harness as evaluate_cli

MIND_V3_SEQUENCE_HISTORY_LIVE_AB_SCHEMA_VERSION = (
    "mind_v3_v141_sequence_history_live_ab_v1"
)
MIND_V3_SEQUENCE_HISTORY_LIVE_AB_POLICY = (
    "controlled_v141_sequence_history_action_override_live_ab_v1"
)
DEFAULT_SCORER_PATH = Path(
    "output/mind/mind-v3-v139-sequence-history-shadow-scorer.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v141-sequence-history-live-ab.json"
)
STRICT_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
STRICT_CARRION_FIXTURE_SEEDS = (13, 19, 29, 37, 41, 43)
STRICT_TICKS = 120
MAX_DOMINANT_REQUESTED_ACTION_SHARE = 0.50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v141 controlled live A/B experiment for the Mind v3 "
            "sequence-history action override candidate."
        )
    )
    parser.add_argument(
        "--sequence-history-shadow-scorer",
        type=Path,
        default=DEFAULT_SCORER_PATH,
        help="Full v139 sequence-history shadow scorer report.",
    )
    parser.add_argument(
        "--broad-seeds",
        default=",".join(str(seed) for seed in STRICT_BROAD_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=STRICT_TICKS)
    parser.add_argument(
        "--fixture-seeds",
        default=",".join(str(seed) for seed in STRICT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--fixture-ticks", type=int, default=STRICT_TICKS)
    parser.add_argument("--trajectory-output-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = build_sequence_history_live_ab_report(
            sequence_history_shadow_scorer_path=args.sequence_history_shadow_scorer,
            broad_seeds=evaluate_cli._parse_seeds(args.broad_seeds),
            ticks=int(args.ticks),
            fixture_seeds=evaluate_cli._parse_seeds(args.fixture_seeds),
            fixture_ticks=int(args.fixture_ticks),
            trajectory_output_dir=args.trajectory_output_dir,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"failed to run sequence-history live A/B: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_summary(report, args.output)


def build_sequence_history_live_ab_report(
    *,
    sequence_history_shadow_scorer_path: str | Path = DEFAULT_SCORER_PATH,
    broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = STRICT_TICKS,
    fixture_seeds: Sequence[int] = STRICT_CARRION_FIXTURE_SEEDS,
    fixture_ticks: int = STRICT_TICKS,
    trajectory_output_dir: str | Path | None = None,
) -> dict[str, object]:
    scorer_path = Path(sequence_history_shadow_scorer_path)
    payload = evaluate_cli._load_json_mapping(scorer_path)
    source_integrity = evaluate_cli._sequence_history_shadow_scorer_source_integrity(
        payload
    )
    if source_integrity.get("passed") is not True:
        raise ValueError(
            "v141 action override requires ready v139 source integrity; "
            f"failures={source_integrity.get('failures')}"
        )
    scorer = evaluate_cli._load_sequence_history_shadow_scorer_artifact(payload)
    broad_seed_values = tuple(int(seed) for seed in broad_seeds)
    fixture_seed_values = tuple(int(seed) for seed in fixture_seeds)
    output_dir = Path(trajectory_output_dir) if trajectory_output_dir else None

    baseline_runs = [
        evaluate_cli._run_once(
            seed=seed,
            ticks=int(ticks),
            policy=evaluate_cli._mind_v3_policy(
                seed=seed,
                founder_template=None,
            ),
            trajectory_output_path=evaluate_cli._trajectory_output_path(
                output_dir,
                "v141",
                "broad",
                "baseline",
                seed,
                ticks,
            ),
            trajectory_split_id="mind_v3_v141_broad_baseline",
        )
        for seed in broad_seed_values
    ]
    override_runs = [
        evaluate_cli._run_once(
            seed=seed,
            ticks=int(ticks),
            policy=evaluate_cli._mind_v3_policy(
                seed=seed,
                founder_template=None,
                sequence_history_shadow_scorer=scorer,
                sequence_history_action_override=True,
                sequence_history_action_override_source_integrity_passed=True,
            ),
            trajectory_output_path=evaluate_cli._trajectory_output_path(
                output_dir,
                "v141",
                "broad",
                "override",
                seed,
                ticks,
            ),
            trajectory_split_id="mind_v3_v141_broad_override",
        )
        for seed in broad_seed_values
    ]
    baseline_aggregate = evaluate_cli._aggregate_runs(baseline_runs)
    override_aggregate = evaluate_cli._aggregate_runs(override_runs)

    baseline_fixture_suite = _run_carrion_fixture_suite(
        seeds=fixture_seed_values,
        ticks=int(fixture_ticks),
        trajectory_output_dir=output_dir,
        trajectory_prefix="v141-carrion-baseline",
        policy_factory=lambda seed: evaluate_cli._mind_v3_policy(
            seed=seed,
            founder_template=None,
        ),
        policy_name="mind_v3_baseline",
    )
    override_fixture_suite = _run_carrion_fixture_suite(
        seeds=fixture_seed_values,
        ticks=int(fixture_ticks),
        trajectory_output_dir=output_dir,
        trajectory_prefix="v141-carrion-override",
        policy_factory=lambda seed: evaluate_cli._mind_v3_policy(
            seed=seed,
            founder_template=None,
            sequence_history_shadow_scorer=scorer,
            sequence_history_action_override=True,
            sequence_history_action_override_source_integrity_passed=True,
        ),
        policy_name="sequence_history_override",
    )
    fixture_config = evaluate_cli.mind_v3_fixture_gate_config(
        suite="basic",
        seeds=list(fixture_seed_values),
        ticks=int(fixture_ticks),
        min_alive=1.0,
        min_births=0.0,
        min_mixed_stable_births=0.0,
        min_energy_viability=0.0,
        min_hydration_viability=0.0,
        min_health_viability=0.0,
        min_matched_diet_viability=0.0,
        min_biologically_ready=0.0,
    )
    baseline_fixture_gate = evaluate_cli.mind_v3_fixture_gate_status(
        fixture_suite=baseline_fixture_suite,
        fixture_config=fixture_config,
    )
    override_fixture_gate = evaluate_cli.mind_v3_fixture_gate_status(
        fixture_suite=override_fixture_suite,
        fixture_config=fixture_config,
    )

    broad = _comparison_section(
        baseline_runs=baseline_runs,
        override_runs=override_runs,
        baseline_aggregate=baseline_aggregate,
        override_aggregate=override_aggregate,
    )
    carrion_baseline_aggregate = _fixture_aggregate(baseline_fixture_suite)
    carrion_override_aggregate = _fixture_aggregate(override_fixture_suite)
    carrion = {
        "fixture": "carrion_only",
        "seeds": list(fixture_seed_values),
        "ticks": int(fixture_ticks),
        "baseline_fixture_suite": baseline_fixture_suite,
        "override_fixture_suite": override_fixture_suite,
        "baseline_fixture_gate": baseline_fixture_gate,
        "override_fixture_gate": override_fixture_gate,
        **_comparison_section(
            baseline_runs=_fixture_runs(baseline_fixture_suite),
            override_runs=_fixture_runs(override_fixture_suite),
            baseline_aggregate=carrion_baseline_aggregate,
            override_aggregate=carrion_override_aggregate,
        ),
    }
    acceptance = _acceptance(
        broad=broad,
        carrion=carrion,
        broad_seeds=broad_seed_values,
        fixture_seeds=fixture_seed_values,
        ticks=int(ticks),
        fixture_ticks=int(fixture_ticks),
    )
    classification = {
        "primary": (
            "sequence_history_live_override_passed_controlled_ab"
            if acceptance["passed"]
            else "sequence_history_live_override_blocked_non_promotable"
        ),
        "labels": [
            (
                "sequence_history_live_override_passed_controlled_ab"
                if acceptance["passed"]
                else "sequence_history_live_override_blocked_non_promotable"
            )
        ],
    }
    return {
        "schema_version": MIND_V3_SEQUENCE_HISTORY_LIVE_AB_SCHEMA_VERSION,
        "policy": MIND_V3_SEQUENCE_HISTORY_LIVE_AB_POLICY,
        "contract": {
            "default_runtime_behavior_changed": False,
            "explicit_opt_in_required": True,
            "runtime_action_selection_changed_when_enabled": True,
            "trainer_effect": "none",
            "gate_effect": "none",
            "viewer_effect": "none",
            "replay_golden_effect": "none",
            "trajectory_schema_changed": False,
            "runtime_promotion_authorized": False,
        },
        "source_integrity": source_integrity,
        "scorer_source": str(scorer_path),
        "matrix": {
            "broad_seeds": list(broad_seed_values),
            "ticks": int(ticks),
            "fixture": "carrion_only",
            "fixture_seeds": list(fixture_seed_values),
            "fixture_ticks": int(fixture_ticks),
        },
        "broad": broad,
        "carrion_only": carrion,
        "acceptance": acceptance,
        "classification": classification,
        "non_default_runtime": True,
        "non_promoted": True,
    }


def _run_carrion_fixture_suite(
    *,
    seeds: Sequence[int],
    ticks: int,
    trajectory_output_dir: Path | None,
    trajectory_prefix: str,
    policy_factory: object,
    policy_name: str,
) -> dict[str, object]:
    return evaluate_cli.run_controlled_fixture_policy_suite(
        suite="basic",
        fixture_names=["carrion_only"],
        seeds=[int(seed) for seed in seeds],
        ticks=int(ticks),
        learned_policy_factory=lambda _fixture_name, seed: policy_factory(seed),
        learned_policy_key="mind_v3",
        learned_policy_name=policy_name,
        trajectory_output_dir=trajectory_output_dir,
        trajectory_prefix=trajectory_prefix,
    )


def _comparison_section(
    *,
    baseline_runs: Sequence[Mapping[str, object]],
    override_runs: Sequence[Mapping[str, object]],
    baseline_aggregate: Mapping[str, object],
    override_aggregate: Mapping[str, object],
) -> dict[str, object]:
    return {
        "baseline": {
            "runs": [dict(run) for run in baseline_runs],
            "aggregate": dict(baseline_aggregate),
        },
        "override": {
            "runs": [dict(run) for run in override_runs],
            "aggregate": dict(override_aggregate),
        },
        "aggregate_delta": _aggregate_delta(
            baseline=baseline_aggregate,
            override=override_aggregate,
        ),
        "per_seed_delta": _per_seed_delta(
            baseline_runs=baseline_runs,
            override_runs=override_runs,
        ),
    }


def _aggregate_delta(
    *,
    baseline: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    return {
        "alive_agents_mean": _round(
            _float(override.get("alive_agents_mean"))
            - _float(baseline.get("alive_agents_mean"))
        ),
        "births_mean": _round(
            _float(override.get("births_mean")) - _float(baseline.get("births_mean"))
        ),
        "deaths_mean": _round(
            _float(override.get("deaths_mean")) - _float(baseline.get("deaths_mean"))
        ),
        "requested_action_count_delta": _count_delta(
            baseline.get("requested_action_counts"),
            override.get("requested_action_counts"),
        ),
        "resolved_action_count_delta": _count_delta(
            baseline.get("resolved_action_counts"),
            override.get("resolved_action_counts"),
        ),
        "unsupported_requested_action_count": int(
            _float(override.get("unsupported_requested_action_count"))
            - _float(baseline.get("unsupported_requested_action_count"))
        ),
        "unsupported_resolved_action_count": int(
            _float(override.get("unsupported_resolved_action_count"))
            - _float(baseline.get("unsupported_resolved_action_count"))
        ),
        "heuristic_action_source_count": int(
            _float(override.get("heuristic_action_source_count"))
            - _float(baseline.get("heuristic_action_source_count"))
        ),
        "dominant_requested_action_share": _round(
            _float(override.get("dominant_requested_action_share"))
            - _float(baseline.get("dominant_requested_action_share"))
        ),
        "override_applied_count": _shadow_count(override, "override_applied_count"),
        "override_applied_share": _shadow_float(override, "override_applied_share"),
    }


def _per_seed_delta(
    *,
    baseline_runs: Sequence[Mapping[str, object]],
    override_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline_by_seed = {_int(run.get("seed")): run for run in baseline_runs}
    rows: list[dict[str, object]] = []
    for run in sorted(override_runs, key=lambda item: _int(item.get("seed"))):
        seed = _int(run.get("seed"))
        baseline = _mapping(baseline_by_seed.get(seed))
        rows.append(
            {
                "seed": seed,
                "alive_delta": int(
                    _float(run.get("alive_agents"))
                    - _float(baseline.get("alive_agents"))
                ),
                "births_delta": int(
                    _float(run.get("births")) - _float(baseline.get("births"))
                ),
                "deaths_delta": int(
                    _float(run.get("deaths")) - _float(baseline.get("deaths"))
                ),
                "baseline_alive": _int(baseline.get("alive_agents")),
                "override_alive": _int(run.get("alive_agents")),
                "baseline_births": _int(baseline.get("births")),
                "override_births": _int(run.get("births")),
                "baseline_deaths": _int(baseline.get("deaths")),
                "override_deaths": _int(run.get("deaths")),
                "requested_action_count_delta": _count_delta(
                    baseline.get("requested_action_counts"),
                    run.get("requested_action_counts"),
                ),
                "resolved_action_count_delta": _count_delta(
                    baseline.get("resolved_action_counts"),
                    run.get("resolved_action_counts"),
                ),
                "unsupported_requested_action_count_delta": int(
                    _float(run.get("unsupported_requested_action_count"))
                    - _float(baseline.get("unsupported_requested_action_count"))
                ),
                "unsupported_resolved_action_count_delta": int(
                    _float(run.get("unsupported_resolved_action_count"))
                    - _float(baseline.get("unsupported_resolved_action_count"))
                ),
                "heuristic_action_source_count_delta": int(
                    _float(run.get("heuristic_action_source_count"))
                    - _float(baseline.get("heuristic_action_source_count"))
                ),
                "dominant_requested_action_share_delta": _round(
                    _float(run.get("dominant_requested_action_share"))
                    - _float(baseline.get("dominant_requested_action_share"))
                ),
                "override_applied_count": _shadow_count(
                    run,
                    "override_applied_count",
                ),
                "override_applied_share": _shadow_float(
                    run,
                    "override_applied_share",
                ),
            }
        )
    return rows


def _acceptance(
    *,
    broad: Mapping[str, object],
    carrion: Mapping[str, object],
    broad_seeds: Sequence[int],
    fixture_seeds: Sequence[int],
    ticks: int,
    fixture_ticks: int,
) -> dict[str, object]:
    broad_override = _mapping(_mapping(broad.get("override")).get("aggregate"))
    carrion_override = _mapping(_mapping(carrion.get("override")).get("aggregate"))
    broad_per_seed = _list_of_mappings(broad.get("per_seed_delta"))
    broad_override_shadow = _mapping(
        broad_override.get("sequence_history_shadow_scorer_diagnostics")
    )
    carrion_override_shadow = _mapping(
        carrion_override.get("sequence_history_shadow_scorer_diagnostics")
    )
    baseline_gate = _mapping(carrion.get("baseline_fixture_gate"))
    override_gate = _mapping(carrion.get("override_fixture_gate"))
    baseline_blocker_count = len(_list_of_mappings(baseline_gate.get("blockers")))
    override_blocker_count = len(_list_of_mappings(override_gate.get("blockers")))
    carrion_delta = _mapping(carrion.get("aggregate_delta"))
    override_applied_count = _shadow_count(
        broad_override,
        "override_applied_count",
    ) + _shadow_count(carrion_override, "override_applied_count")
    floors = [
        _floor(
            "strict_broad_seed_set",
            tuple(int(seed) for seed in broad_seeds) == STRICT_BROAD_SEEDS,
            observed=list(broad_seeds),
            required=list(STRICT_BROAD_SEEDS),
        ),
        _floor(
            "strict_broad_ticks",
            int(ticks) == STRICT_TICKS,
            observed=int(ticks),
            required=STRICT_TICKS,
        ),
        _floor(
            "strict_carrion_fixture_seed_set",
            tuple(int(seed) for seed in fixture_seeds)
            == STRICT_CARRION_FIXTURE_SEEDS,
            observed=list(fixture_seeds),
            required=list(STRICT_CARRION_FIXTURE_SEEDS),
            fixture="carrion_only",
        ),
        _floor(
            "strict_carrion_fixture_ticks",
            int(fixture_ticks) == STRICT_TICKS,
            observed=int(fixture_ticks),
            required=STRICT_TICKS,
            fixture="carrion_only",
        ),
        _floor(
            "zero_heuristic_action_source_count",
            _int(broad_override.get("heuristic_action_source_count"))
            + _int(carrion_override.get("heuristic_action_source_count"))
            == 0,
            observed=_int(broad_override.get("heuristic_action_source_count"))
            + _int(carrion_override.get("heuristic_action_source_count")),
            required=0,
        ),
        _floor(
            "zero_unsupported_requested_actions",
            _int(broad_override.get("unsupported_requested_action_count"))
            + _int(carrion_override.get("unsupported_requested_action_count"))
            == 0,
            observed=_int(broad_override.get("unsupported_requested_action_count"))
            + _int(carrion_override.get("unsupported_requested_action_count")),
            required=0,
        ),
        _floor(
            "dominant_requested_action_share_lte_0_50",
            max(
                _float(broad_override.get("dominant_requested_action_share")),
                _float(carrion_override.get("dominant_requested_action_share")),
            )
            <= MAX_DOMINANT_REQUESTED_ACTION_SHARE,
            observed=max(
                _float(broad_override.get("dominant_requested_action_share")),
                _float(carrion_override.get("dominant_requested_action_share")),
            ),
            required=MAX_DOMINANT_REQUESTED_ACTION_SHARE,
        ),
        _floor(
            "override_applied_count_nonzero",
            override_applied_count > 0,
            observed=override_applied_count,
            required=">0",
        ),
    ]
    for item in broad_per_seed:
        seed = _int(item.get("seed"))
        floors.append(
            _floor(
                f"broad_seed_{seed}_alive_no_regression",
                _int(item.get("alive_delta")) >= 0,
                observed=item.get("alive_delta"),
                required=0,
                seed=seed,
            )
        )
        floors.append(
            _floor(
                f"broad_seed_{seed}_births_no_regression",
                _int(item.get("births_delta")) >= 0,
                observed=item.get("births_delta"),
                required=0,
                seed=seed,
            )
        )
    floors.append(
        _floor(
            "carrion_alive_improves_or_blocker_count_reduces",
            _float(carrion_delta.get("alive_agents_mean")) > 0.0
            or override_blocker_count < baseline_blocker_count,
            observed={
                "alive_delta": carrion_delta.get("alive_agents_mean"),
                "baseline_blocker_count": baseline_blocker_count,
                "override_blocker_count": override_blocker_count,
            },
            required=(
                "carrion alive mean delta > 0 or override blocker count "
                "< baseline blocker count"
            ),
            fixture="carrion_only",
        )
    )
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    first_seed_failure = next(
        (
            floor
            for floor in floors
            if floor["passed"] is not True and "seed" in floor
        ),
        None,
    )
    first_fixture_failure = next(
        (
            floor
            for floor in floors
            if floor["passed"] is not True and "fixture" in floor
        ),
        None,
    )
    return {
        "policy": "v141_sequence_history_live_override_acceptance_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "first_failing_seed": (
            None if first_seed_failure is None else first_seed_failure.get("seed")
        ),
        "first_failing_fixture": (
            None
            if first_fixture_failure is None
            else first_fixture_failure.get("fixture")
        ),
        "override_applied_count": override_applied_count,
        "override_applied_share": _round(
            _share(
                override_applied_count,
                _int(broad_override_shadow.get("decision_count"))
                + _int(carrion_override_shadow.get("decision_count")),
            )
        ),
        "floors": floors,
    }


def _floor(
    name: str,
    passed: bool,
    *,
    observed: object,
    required: object,
    seed: int | None = None,
    fixture: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if fixture is not None:
        payload["fixture"] = fixture
    return payload


def _fixture_aggregate(fixture_suite: Mapping[str, object]) -> Mapping[str, object]:
    fixture = evaluate_cli._fixture_report_by_name(fixture_suite, "carrion_only")
    key = str(fixture_suite.get("evaluated_policy_key", "mind_v3"))
    return _mapping(_mapping(_mapping(fixture.get("comparison")).get(key)).get("aggregate"))


def _fixture_runs(fixture_suite: Mapping[str, object]) -> list[Mapping[str, object]]:
    fixture = evaluate_cli._fixture_report_by_name(fixture_suite, "carrion_only")
    key = str(fixture_suite.get("evaluated_policy_key", "mind_v3"))
    return _list_of_mappings(_mapping(_mapping(fixture.get("comparison")).get(key)).get("runs"))


def _shadow_count(run_or_aggregate: Mapping[str, object], key: str) -> int:
    return _int(
        _mapping(run_or_aggregate.get("sequence_history_shadow_scorer_diagnostics")).get(
            key
        )
    )


def _shadow_float(run_or_aggregate: Mapping[str, object], key: str) -> float:
    return _round(
        _float(
            _mapping(
                run_or_aggregate.get("sequence_history_shadow_scorer_diagnostics")
            ).get(key)
        )
    )


def _count_delta(baseline_counts: object, override_counts: object) -> dict[str, int]:
    baseline = Counter(
        {str(key): int(value) for key, value in _mapping(baseline_counts).items()}
    )
    override = Counter(
        {str(key): int(value) for key, value in _mapping(override_counts).items()}
    )
    keys = sorted(set(baseline) | set(override))
    return {key: int(override.get(key, 0) - baseline.get(key, 0)) for key in keys}


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    acceptance = _mapping(report.get("acceptance"))
    classification = _mapping(report.get("classification"))
    print(f"sequence_history_live_ab_report={output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"acceptance_passed={acceptance.get('passed')}")
    print(f"first_failed_floor={acceptance.get('first_failed_floor')}")
    print(f"override_applied_count={acceptance.get('override_applied_count')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    main()
