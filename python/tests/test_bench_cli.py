from __future__ import annotations

import json
import unittest

from evolution_sim.cli.bench import (
    BenchScenario,
    _ru_maxrss_to_kib,
    _run_once_isolated,
    _scenario_stats,
)
from evolution_sim.env import RunMode


class BenchCliTests(unittest.TestCase):
    def test_ru_maxrss_is_normalized_to_kib_by_platform(self) -> None:
        self.assertEqual(_ru_maxrss_to_kib(91_701_248, system="Darwin"), 89_552)
        self.assertEqual(_ru_maxrss_to_kib(89_552, system="Linux"), 89_552)

    def test_scenario_stats_report_normalized_rss_fields(self) -> None:
        scenario = BenchScenario("unit", 7, 1, RunMode.SUMMARY_ONLY)

        stats = _scenario_stats(
            scenario,
            [
                {
                    "wall_seconds": 0.1,
                    "peak_rss_kib": 90_000,
                    "replay_size_bytes": None,
                    "trajectory_record_count": 0,
                    "trajectory_output_bytes": None,
                    "runtime_cost_counters": {
                        "observation_builds": 10,
                        "biotic_state_builds": 2,
                    },
                },
                {
                    "wall_seconds": 0.2,
                    "peak_rss_kib": 100_000,
                    "replay_size_bytes": None,
                    "trajectory_record_count": 4,
                    "trajectory_output_bytes": None,
                    "runtime_cost_counters": {
                        "observation_builds": 20,
                        "biotic_state_builds": 4,
                    },
                },
            ],
        )

        self.assertEqual(stats["median_peak_rss_kib"], 95_000)
        self.assertEqual(stats["p95_peak_rss_kib"], 99_500)
        self.assertIsNone(stats["median_replay_size_bytes"])
        self.assertEqual(stats["median_trajectory_record_count"], 2)
        self.assertEqual(
            stats["median_runtime_cost_counters"]["observation_builds"],
            15,
        )
        self.assertEqual(
            stats["median_runtime_cost_counters"]["biotic_state_builds"],
            3,
        )
        json.dumps(stats)

    def test_isolated_run_reports_plausible_summary_memory(self) -> None:
        result = _run_once_isolated(
            BenchScenario("unit_summary_seed7_ticks1", 7, 1, RunMode.SUMMARY_ONLY),
            timeout_seconds=30.0,
        )

        self.assertGreater(result["wall_seconds"], 0)
        self.assertGreater(result["peak_rss_kib"], 1_000)
        self.assertLess(result["peak_rss_kib"], 10 * 1024 * 1024)
        self.assertIsNone(result["replay_size_bytes"])
        self.assertIsInstance(result["runtime_cost_counters"], dict)
        self.assertGreater(result["runtime_cost_counters"]["observation_builds"], 0)


if __name__ == "__main__":
    unittest.main()
