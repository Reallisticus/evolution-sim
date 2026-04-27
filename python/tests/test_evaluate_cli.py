from __future__ import annotations

import json
import unittest

from evolution_sim.cli.evaluate import build_evaluation_report, parse_seed_selection
from evolution_sim.env import RunMode


class EvaluateCliTests(unittest.TestCase):
    def test_parse_seed_selection_dedupes_in_order(self) -> None:
        self.assertEqual(
            parse_seed_selection([3, 2, 3], "1, 2, 1"),
            [1, 2, 3],
        )

    def test_summary_only_evaluation_compares_runs_without_species_surfaces(self) -> None:
        report = build_evaluation_report(
            seeds=[7, 8],
            ticks=12,
            mode=RunMode.SUMMARY_ONLY,
            min_births=0,
        )

        self.assertEqual(report["protocol"]["mode"], RunMode.SUMMARY_ONLY.value)
        self.assertEqual(report["protocol"]["run_count"], 2)
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(report["aggregate"]["run_count"], 2)
        self.assertIn("hazard_counts_at_end", report["aggregate"])
        self.assertIn("trophic_role_counts_at_end", report["aggregate"])
        self.assertNotIn("species", report["runs"][0])
        json.dumps(report)

    def test_full_replay_evaluation_includes_compact_species_summary(self) -> None:
        report = build_evaluation_report(
            seeds=[7],
            ticks=8,
            mode=RunMode.FULL_REPLAY,
            min_births=0,
        )

        self.assertIn("species", report["runs"][0])
        self.assertIn("alive_species_count", report["aggregate"])
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
