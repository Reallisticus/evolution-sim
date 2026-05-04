from __future__ import annotations

import unittest

from evolution_sim.cli import test_runner


class TestRunnerTests(unittest.TestCase):
    def test_suite_budget_defaults_and_override(self) -> None:
        self.assertGreater(test_runner._suite_runtime_budget("fast", None), 0)
        self.assertEqual(test_runner._suite_runtime_budget("fast", 12.5), 12.5)
        self.assertIsNone(test_runner._suite_runtime_budget("fast", 0.0))

    def test_duration_report_includes_suite_count_time_and_budget(self) -> None:
        report = test_runner._duration_report(
            suite_name="fast",
            tests_run=12,
            wall_seconds=1.23456,
            budget_seconds=30.0,
        )

        self.assertIn("suite=fast", report)
        self.assertIn("tests=12", report)
        self.assertIn("wall_seconds=1.235", report)
        self.assertIn("budget_seconds=30.000", report)


if __name__ == "__main__":
    unittest.main()
