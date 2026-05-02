from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from evolution_sim.cli.bench import (
    BenchScenario,
    _ru_maxrss_to_kib,
    _run_multi_process_summary_rollout,
    _run_once_isolated,
    _scenario_stats,
)
from evolution_sim.env import RunMode


class BenchCliTests(unittest.TestCase):
    def assertBenchmarkReportSchema(
        self,
        payload: dict[str, object],
        *,
        complete: bool,
    ) -> None:
        self.assertIsInstance(payload["complete"], bool)
        self.assertEqual(payload["complete"], complete)
        self.assertIsInstance(payload["scenarios"], list)
        self.assertIn("multi_process_summary_rollout", payload)
        self.assertBenchmarkProtocolSchema(payload["protocol"])
        for scenario in payload["scenarios"]:
            self.assertScenarioSchema(scenario)
        if complete:
            self.assertNotIn("error", payload)
        else:
            self.assertErrorSchema(payload["error"])

    def assertBenchmarkProtocolSchema(self, protocol: object) -> None:
        self.assertIsInstance(protocol, dict)
        assert isinstance(protocol, dict)
        self.assertIsInstance(protocol["warmup_runs"], int)
        self.assertGreaterEqual(protocol["warmup_runs"], 0)
        self.assertIsInstance(protocol["measured_runs"], int)
        self.assertGreater(protocol["measured_runs"], 0)
        self.assertEqual(protocol["rss_unit"], "KiB")
        self.assertEqual(protocol["scenario_repetition_isolation"], "fresh_process")
        machine = protocol["machine_profile"]
        self.assertIsInstance(machine, dict)
        assert isinstance(machine, dict)
        for key in ("cpu", "os", "python"):
            self.assertIsInstance(machine[key], str)
        self.assertIn("ram_gib", machine)

    def assertScenarioSchema(self, scenario: object) -> None:
        self.assertIsInstance(scenario, dict)
        assert isinstance(scenario, dict)
        for key in ("name", "mode"):
            self.assertIsInstance(scenario[key], str)
        for key in ("seed", "ticks", "median_peak_rss_kib", "p95_peak_rss_kib"):
            self.assertIsInstance(scenario[key], int)
        for key in ("median_wall_seconds", "p95_wall_seconds"):
            self.assertIsInstance(scenario[key], (int, float))
            self.assertGreaterEqual(float(scenario[key]), 0.0)
        self.assertIn("median_replay_size_bytes", scenario)
        self.assertIsInstance(scenario["median_trajectory_record_count"], int)
        self.assertIn("median_trajectory_output_bytes", scenario)
        self.assertIsInstance(scenario["median_runtime_cost_counters"], dict)
        self.assertIsInstance(scenario["stream_trajectory"], bool)
        self.assertIn("record_trajectory", scenario)

    def assertSummaryOnlyMaskBuildBudget(self, scenario: object) -> None:
        self.assertIsInstance(scenario, dict)
        assert isinstance(scenario, dict)
        self.assertEqual(scenario["mode"], RunMode.SUMMARY_ONLY)
        counters = scenario["median_runtime_cost_counters"]
        self.assertIsInstance(counters, dict)
        assert isinstance(counters, dict)
        observation_builds = int(counters["observation_builds"])
        action_mask_builds = int(counters["action_mask_builds"])
        self.assertGreater(observation_builds, 0)
        self.assertGreater(action_mask_builds, observation_builds)
        self.assertLessEqual(action_mask_builds, observation_builds * 2)

    def assertErrorSchema(self, error: object) -> None:
        self.assertIsInstance(error, dict)
        assert isinstance(error, dict)
        for key in ("phase", "type", "message"):
            self.assertIsInstance(error[key], str)
        for key in ("completed_scenarios", "requested_scenarios"):
            self.assertIsInstance(error[key], int)
            self.assertGreaterEqual(error[key], 0)

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
                        "action_mask_builds": 19,
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
                        "action_mask_builds": 39,
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
            stats["median_runtime_cost_counters"]["action_mask_builds"],
            29,
        )
        self.assertEqual(
            stats["median_runtime_cost_counters"]["biotic_state_builds"],
            3,
        )
        self.assertSummaryOnlyMaskBuildBudget(stats)
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

    def test_isolated_run_times_out_without_waiting_for_child_exit(self) -> None:
        with self.assertRaises(TimeoutError):
            _run_once_isolated(
                BenchScenario(
                    "unit_summary_seed7_ticks100",
                    7,
                    100,
                    RunMode.SUMMARY_ONLY,
                ),
                timeout_seconds=0.001,
            )

    def test_multi_process_rollout_reports_timeout_before_all_workers_finish(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "completed=0/1"):
            _run_multi_process_summary_rollout(
                timeout_seconds=0.001,
                worker_count=1,
                ticks=100,
            )

    def test_multi_process_rollout_rejects_invalid_protocol_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker_count"):
            _run_multi_process_summary_rollout(
                timeout_seconds=1.0,
                worker_count=0,
                ticks=1,
            )
        with self.assertRaisesRegex(ValueError, "ticks"):
            _run_multi_process_summary_rollout(
                timeout_seconds=1.0,
                worker_count=1,
                ticks=0,
            )

    def test_cli_timeout_emits_partial_json_report(self) -> None:
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "python",
        }

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "evolution_sim.cli.bench",
                "--warmup",
                "0",
                "--runs",
                "1",
                "--scenario",
                "summary_seed7_ticks100",
                "--scenario-timeout-seconds",
                "0.001",
                "--skip-multiprocess",
            ],
            check=False,
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertBenchmarkReportSchema(payload, complete=False)
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["scenarios"], [])
        self.assertEqual(payload["error"]["type"], "TimeoutError")
        self.assertEqual(payload["error"]["completed_scenarios"], 0)
        self.assertIn("summary_seed7_ticks100", payload["error"]["phase"])

    def test_cli_rejects_invalid_protocol_arguments(self) -> None:
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "python",
        }
        cases = (
            (("--warmup", "-1"), "--warmup must be greater than or equal to 0"),
            (("--runs", "0"), "--runs must be greater than 0"),
            (
                ("--scenario-timeout-seconds", "0"),
                "--scenario-timeout-seconds must be a finite positive number",
            ),
            (
                ("--multiprocess-timeout-seconds", "nan"),
                "--multiprocess-timeout-seconds must be a finite positive number",
            ),
        )

        for arguments, expected_message in cases:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "evolution_sim.cli.bench",
                        *arguments,
                    ],
                    check=False,
                    cwd=os.getcwd(),
                    env=environment,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertIn(expected_message, completed.stderr)

    def test_cli_success_marks_report_complete(self) -> None:
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "python",
        }

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "evolution_sim.cli.bench",
                "--warmup",
                "0",
                "--runs",
                "1",
                "--scenario",
                "summary_seed7_ticks20",
                "--skip-multiprocess",
            ],
            check=False,
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertBenchmarkReportSchema(payload, complete=True)
        self.assertTrue(payload["complete"])
        self.assertNotIn("error", payload)
        self.assertEqual(len(payload["scenarios"]), 1)
        self.assertSummaryOnlyMaskBuildBudget(payload["scenarios"][0])

    def test_cli_multiprocess_timeout_emits_partial_json_report(self) -> None:
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "python",
        }

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "evolution_sim.cli.bench",
                "--warmup",
                "0",
                "--runs",
                "1",
                "--scenario",
                "summary_seed7_ticks20",
                "--multiprocess-timeout-seconds",
                "0.001",
            ],
            check=False,
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertBenchmarkReportSchema(payload, complete=False)
        self.assertFalse(payload["complete"])
        self.assertEqual(len(payload["scenarios"]), 1)
        self.assertEqual(payload["error"]["phase"], "multiprocess summary rollout")
        self.assertEqual(payload["error"]["type"], "TimeoutError")
        self.assertEqual(payload["error"]["completed_scenarios"], 1)


if __name__ == "__main__":
    unittest.main()
