from __future__ import annotations

import argparse
import ast
import sys
import time
import unittest
from pathlib import Path


SUITE_RUNTIME_BUDGET_SECONDS: dict[str, float] = {
    "fast": 900.0,
    "full": 2_700.0,
}

SLOW_TEST_SUFFIXES: frozenset[str] = frozenset(
    {
        "HeadlessSimulationTests.test_compact_full_replay_has_turnover_species_and_surfaces",
        "HeadlessSimulationTests.test_hidden_forest_water_access_does_not_exist",
        "HeadlessSimulationTests.test_shoreline_adjacent_water_niche_exists_across_targeted_seeds",
        "HeadlessSimulationTests.test_soft_refuge_is_selective_and_changes_over_time",
        (
            "HeadlessSimulationTests."
            "test_full_replay_taxonomy_surfaces_remain_available_on_golden_speciation_seed"
        ),
        "HeadlessSimulationTests.test_species_ids_only_change_on_logged_speciation_events",
        "HeadlessSimulationTests.test_multi_seed_runs_remain_viable_and_reproductive",
        "HeadlessSimulationTests.test_production_readiness_mixed_world_sweep",
        "test_summary_only_release_span_is_byte_deterministic_under_repeated_runs",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_cases(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def _filter_fast_suite(suite: unittest.TestSuite) -> unittest.TestSuite:
    filtered = unittest.TestSuite()
    for case in _iter_cases(suite):
        test_id = case.id()
        if any(test_id.endswith(suffix) for suffix in SLOW_TEST_SUFFIXES):
            continue
        filtered.addTest(case)
    return filtered


def _assert_no_direct_cache_poking() -> None:
    test_dir = _repo_root() / "python" / "tests"
    failures: list[str] = []
    for path in sorted(test_dir.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                else:
                    targets.append(node.target)
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr.startswith("cached_")
                    ):
                        failures.append(f"{path}:{node.lineno}: cached attribute assignment")
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_invalidate_biotic_state"
                ):
                    failures.append(f"{path}:{node.lineno}: _invalidate_biotic_state call")
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                    and node.args[1].value.startswith("cached_")
                ):
                    failures.append(f"{path}:{node.lineno}: cached attribute setattr")
    if failures:
        joined = "\n".join(failures)
        raise SystemExit(f"Direct cache poking detected in tests:\n{joined}")


def _discover_suite() -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    root = _repo_root()
    return loader.discover(str(root / "python" / "tests"), pattern="test_*.py")


def _suite_runtime_budget(
    suite_name: str,
    override_budget_seconds: float | None,
) -> float | None:
    if override_budget_seconds is not None:
        if override_budget_seconds <= 0:
            return None
        return override_budget_seconds
    return SUITE_RUNTIME_BUDGET_SECONDS.get(suite_name)


def _duration_report(
    *,
    suite_name: str,
    tests_run: int,
    wall_seconds: float,
    budget_seconds: float | None,
) -> str:
    budget = "none" if budget_seconds is None else f"{budget_seconds:.3f}"
    return (
        "[test-runner] "
        f"suite={suite_name} tests={tests_run} "
        f"wall_seconds={wall_seconds:.3f} budget_seconds={budget}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulator test suites.")
    parser.add_argument("--suite", choices=("fast", "full"), default="fast")
    parser.add_argument(
        "--budget-seconds",
        type=float,
        help=(
            "Override the suite runtime budget. Use 0 to disable budget enforcement."
        ),
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        default=None,
        help="Stop at the first failure.",
    )
    parser.add_argument(
        "--no-failfast",
        action="store_false",
        dest="failfast",
        help="Continue running after failures.",
    )
    args = parser.parse_args()

    _assert_no_direct_cache_poking()
    suite = _discover_suite()
    if args.suite == "fast":
        suite = _filter_fast_suite(suite)

    failfast = args.failfast if args.failfast is not None else args.suite == "full"
    runner = unittest.TextTestRunner(verbosity=2, failfast=failfast)
    started = time.perf_counter()
    result = runner.run(suite)
    wall_seconds = time.perf_counter() - started
    budget_seconds = _suite_runtime_budget(args.suite, args.budget_seconds)
    print(
        _duration_report(
            suite_name=args.suite,
            tests_run=result.testsRun,
            wall_seconds=wall_seconds,
            budget_seconds=budget_seconds,
        ),
        file=sys.stderr,
    )
    if not result.wasSuccessful():
        sys.exit(1)
    if budget_seconds is not None and wall_seconds > budget_seconds:
        print(
            (
                "[test-runner] runtime budget exceeded: "
                f"{wall_seconds:.3f}s > {budget_seconds:.3f}s"
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
