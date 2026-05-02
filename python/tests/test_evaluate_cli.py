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
        self.assertIn("carrying_capacity", report["runs"][0])
        self.assertIn("at_cap_tick_share", report["runs"][0]["carrying_capacity"])
        self.assertIn("carrying_capacity", report["aggregate"])
        self.assertIn("at_cap_tick_share", report["aggregate"]["carrying_capacity"])
        self.assertIn("resource_pressure", report["runs"][0])
        self.assertIn("plant_budget", report["runs"][0]["resource_pressure"])
        self.assertIn("energy_spend", report["runs"][0]["resource_pressure"])
        self.assertIn("resource_pressure", report["aggregate"])
        self.assertIn("plant_budget", report["aggregate"]["resource_pressure"])
        self.assertIn("selection_heredity", report["runs"][0])
        self.assertIn(
            "terminal_minus_initial_mean",
            report["runs"][0]["selection_heredity"],
        )
        self.assertIn("selection_heredity", report["aggregate"])
        self.assertIn(
            "terminal_minus_initial_mean",
            report["aggregate"]["selection_heredity"],
        )
        self.assertIn("trophic_role_counts_at_end", report["aggregate"])
        self.assertIn("trophic_lifecycle", report["runs"][0])
        lifecycle = report["runs"][0]["trophic_lifecycle"]
        self.assertIn("births_by_parent_trophic_role", lifecycle)
        self.assertIn("deaths_by_trophic_role", lifecycle)
        self.assertIn("death_causes_by_trophic_role", lifecycle)
        self.assertIn("late_window", lifecycle)
        self.assertIn(
            "death_causes_by_meat_mode",
            report["aggregate"]["trophic_lifecycle"],
        )
        self.assertIn(
            "death_causes_by_meat_mode_by_tick_band",
            report["aggregate"]["trophic_lifecycle"]["meat_mode_persistence"],
        )
        self.assertIn(
            "late_window_trophic_role_presence_runs",
            report["aggregate"]["trophic_lifecycle"],
        )
        reproduction = report["runs"][0]["reproduction"]
        self.assertIn("by_trophic_role", reproduction)
        self.assertIn("by_meat_mode", reproduction)
        self.assertIn("biological_blocker_counts_by_trophic_role", reproduction)
        self.assertIn("energy_readiness_by_trophic_role", reproduction)
        self.assertIn("energy_readiness_by_meat_mode", reproduction)
        self.assertIn("blocked_run_counts_by_meat_mode", reproduction)
        self.assertIn("herbivore", reproduction["by_trophic_role"])
        self.assertIn("hunter", reproduction["by_meat_mode"])
        self.assertIn(
            "reproduction_biological_blockers_by_trophic_role_at_end",
            report["aggregate"],
        )
        self.assertIn(
            "reproduction_energy_readiness_by_meat_mode_at_end",
            report["aggregate"],
        )
        self.assertIn("diet_by_meat_mode_at_end", report["aggregate"])
        self.assertIn(
            "animal_energy",
            report["aggregate"]["diet_by_meat_mode_at_end"]["hunter"]["total"],
        )
        self.assertIn(
            "animal_resource_opportunity_by_meat_mode",
            report["runs"][0]["trophic"],
        )
        self.assertIn(
            "animal_resource_opportunity_by_meat_mode_at_end",
            report["aggregate"],
        )
        self.assertIn(
            "animal_resource_opportunity_run_counts_by_meat_mode",
            report["aggregate"],
        )
        opportunity_run_counts = report["aggregate"][
            "animal_resource_opportunity_run_counts_by_meat_mode"
        ]
        if opportunity_run_counts:
            first_counts = next(iter(opportunity_run_counts.values()))
            self.assertIn("animal_resource_present_unreachable_runs", first_counts)
            self.assertIn("animal_resource_reachable_unconsumed_runs", first_counts)
            self.assertIn("animal_resource_policy_actionable_runs", first_counts)
            self.assertIn(
                "animal_resource_reachable_policy_blocked_runs",
                first_counts,
            )
        self.assertIn(
            "reproduction_blocked_run_counts_by_meat_mode",
            report["aggregate"],
        )
        self.assertNotIn("species", report["runs"][0])
        json.dumps(report)

    def test_release_seed_scavenger_opportunity_diagnostics_are_present(self) -> None:
        report = build_evaluation_report(
            seeds=[3, 11],
            ticks=80,
            mode=RunMode.SUMMARY_ONLY,
            min_births=0,
        )

        for run in report["runs"]:
            scavenger = run["trophic"]["animal_resource_opportunity_by_meat_mode"][
                "scavenger"
            ]
            self.assertIn("animal_resource_policy_actionable_ticks", scavenger)
            self.assertIn("carcass_policy_actionable_ticks", scavenger)
            self.assertIn("fresh_kill_policy_actionable_ticks", scavenger)
            self.assertIn(
                "animal_resource_reachable_policy_blocked_ticks",
                scavenger,
            )
            self.assertIn(
                "carcass_policy_blocked_by_occupant_agent_ticks",
                scavenger,
            )
            self.assertIn(
                "fresh_kill_policy_blocked_by_water_agent_ticks",
                scavenger,
            )

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
