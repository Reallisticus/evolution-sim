from __future__ import annotations

import json
import unittest

from evolution_sim.cli.evaluate import build_evaluation_report, parse_seed_selection
from evolution_sim.env import RunMode
from evolution_sim.env.contracts import SUMMARY_SCHEMA_VERSION


def assert_series_stats_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    for key in ("count", "min", "median", "mean", "max"):
        testcase.assertIn(key, payload)


def assert_numeric_totals_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    testcase.assertIsInstance(payload["total"], dict)
    testcase.assertIsInstance(payload["per_run_mean"], dict)


def assert_carrying_capacity_run_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    for key in ("near_cap_ticks", "at_cap_ticks", "saturation_births", "saturation_deaths"):
        testcase.assertIsInstance(payload[key], int)
    for key in ("near_cap_saturation_threshold", "near_cap_tick_share", "at_cap_tick_share"):
        testcase.assertIsInstance(payload[key], (int, float))


def assert_carrying_capacity_aggregate_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    for key in (
        "near_cap_ticks",
        "at_cap_ticks",
        "near_cap_tick_share",
        "at_cap_tick_share",
        "saturation_births",
        "saturation_deaths",
    ):
        assert_series_stats_schema(testcase, payload[key])


def assert_resource_pressure_run_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    plant_budget = payload["plant_budget"]
    energy_spend = payload["energy_spend"]
    testcase.assertIsInstance(plant_budget, dict)
    testcase.assertIsInstance(energy_spend, dict)
    for key in (
        "energy_created",
        "energy_removed",
        "energy_lost",
        "net_created_minus_removed_lost",
        "energy_available_at_end",
    ):
        testcase.assertIsInstance(plant_budget[key], (int, float))
    for key in ("metabolism", "movement", "attack", "reproduction", "signal", "total"):
        testcase.assertIsInstance(energy_spend[key], (int, float))


def assert_resource_pressure_aggregate_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    assert_numeric_totals_schema(testcase, payload["plant_budget"])
    assert_numeric_totals_schema(testcase, payload["energy_spend"])


def assert_selection_heredity_run_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    for key in (
        "initial_trait_distributions",
        "terminal_alive_trait_distributions",
        "terminal_minus_initial_mean",
    ):
        testcase.assertIsInstance(payload[key], dict)
    initial = payload["initial_trait_distributions"]
    testcase.assertIn("max_energy", initial)
    max_energy_distribution = initial["max_energy"]
    testcase.assertIsInstance(max_energy_distribution, dict)
    for key in ("count", "min", "p10", "median", "mean", "p90", "max"):
        testcase.assertIn(key, max_energy_distribution)


def assert_selection_heredity_aggregate_schema(
    testcase: unittest.TestCase,
    payload: object,
) -> None:
    testcase.assertIsInstance(payload, dict)
    assert isinstance(payload, dict)
    for key in (
        "initial_trait_mean",
        "terminal_alive_trait_mean",
        "terminal_minus_initial_mean",
    ):
        testcase.assertIsInstance(payload[key], dict)
        testcase.assertIn("max_energy", payload[key])
        assert_series_stats_schema(testcase, payload[key]["max_energy"])


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
        self.assertEqual(
            report["protocol"]["summary_schema_version"],
            SUMMARY_SCHEMA_VERSION,
        )
        self.assertEqual(report["protocol"]["run_count"], 2)
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(report["aggregate"]["run_count"], 2)
        self.assertEqual(
            report["aggregate"]["summary_schema_versions"],
            [SUMMARY_SCHEMA_VERSION],
        )
        self.assertEqual(
            report["runs"][0]["summary_schema_version"],
            SUMMARY_SCHEMA_VERSION,
        )
        self.assertIsInstance(report["runs"][0]["land_tile_count"], int)
        assert_series_stats_schema(self, report["aggregate"]["land_tile_count"])
        self.assertIn("hazard_counts_at_end", report["aggregate"])
        assert_carrying_capacity_run_schema(self, report["runs"][0]["carrying_capacity"])
        assert_carrying_capacity_aggregate_schema(
            self,
            report["aggregate"]["carrying_capacity"],
        )
        assert_resource_pressure_run_schema(self, report["runs"][0]["resource_pressure"])
        assert_resource_pressure_aggregate_schema(
            self,
            report["aggregate"]["resource_pressure"],
        )
        assert_selection_heredity_run_schema(
            self,
            report["runs"][0]["selection_heredity"],
        )
        assert_selection_heredity_aggregate_schema(
            self,
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
