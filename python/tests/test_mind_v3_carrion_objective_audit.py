from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_objective_audit
from evolution_sim.cli import mind_v3_evolve
from evolution_sim.mind.carrion_objective_audit import (
    MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
    MIND_V3_CARRION_OBJECTIVE_AUDIT_SCHEMA_VERSION,
    build_carrion_objective_audit_report,
    foundation_heuristic_baseline_contract,
)


class MindV3CarrionObjectiveAuditTests(unittest.TestCase):
    def test_objective_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-objective-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_objective_audit"
            ),
        )

    def test_high_score_contact_gain_low_recovery_flags_objective_mismatch(
        self,
    ) -> None:
        report = build_carrion_objective_audit_report(
            search_reports=[
                (
                    "low_score_recovered",
                    _search_report(
                        candidate_id="low",
                        score=10.0,
                        births=1.0,
                        animal_gain=0.2,
                    ),
                    "low.json",
                ),
                (
                    "mid_score_mixed",
                    _search_report(
                        candidate_id="mid",
                        score=20.0,
                        births=3.0,
                        animal_gain=1.5,
                    ),
                    "mid.json",
                ),
                (
                    "high_score_collapsed",
                    _search_report(
                        candidate_id="high",
                        score=30.0,
                        births=5.0,
                        animal_gain=3.0,
                    ),
                    "high.json",
                ),
            ],
            trace_reports=[
                (
                    "low_score_recovered",
                    _trace_report(
                        contact_agents=1,
                        drink_rate=0.9,
                        hydration_delta=0.2,
                        survival_rate=1.0,
                        water_distance=1.0,
                    ),
                    "low-trace.json",
                ),
                (
                    "mid_score_mixed",
                    _trace_report(
                        contact_agents=2,
                        drink_rate=0.5,
                        hydration_delta=-0.1,
                        survival_rate=0.5,
                        water_distance=2.0,
                    ),
                    "mid-trace.json",
                ),
                (
                    "high_score_collapsed",
                    _trace_report(
                        contact_agents=3,
                        drink_rate=0.1,
                        hydration_delta=-0.5,
                        survival_rate=0.0,
                        water_distance=5.0,
                    ),
                    "high-trace.json",
                ),
            ],
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_OBJECTIVE_AUDIT_SCHEMA_VERSION,
        )
        diagnosis = report["diagnosis"]
        self.assertEqual(
            diagnosis["objective_pressure_assessment"],
            "contact_gain_birth_pressure_over_recovery_likely",
        )
        self.assertEqual(
            diagnosis["ranked_findings"][0]["finding"],
            "objective_mismatch_contact_gain_over_recovery",
        )
        correlations = report["correlations"]["selected_candidates"]
        self.assertEqual(
            correlations[
                "search_score_vs_animal_resource_gain_total_mean"
            ]["direction"],
            "positive",
        )
        self.assertEqual(
            correlations["search_score_vs_drink_after_carrion_rate"]["direction"],
            "negative",
        )

    def test_missing_trace_and_report_fields_are_reported(self) -> None:
        report = build_carrion_objective_audit_report(
            search_reports=[("broken", {"best_candidate": {}}, "broken.json")],
            trace_reports=[],
            missing_inputs=[
                {
                    "label": "missing_trace",
                    "source": "trace_report",
                    "path": "missing.json",
                    "field": "trace_report",
                    "reason": "input_path_missing",
                }
            ],
        )

        missing = report["missing_data"]
        self.assertGreater(missing["total_count"], 1)
        self.assertIn("search_score", missing["counts_by_field"])
        self.assertEqual(missing["counts_by_field"]["trace_report"], 1)
        self.assertIn(
            "metric_missing_or_non_numeric",
            missing["counts_by_reason"],
        )

    def test_gate_aligned_retrospective_selector_reports_changed_candidate(
        self,
    ) -> None:
        report = build_carrion_objective_audit_report(
            search_reports=[
                (
                    "probe",
                    _selector_search_report(
                        selected_candidate_id="current",
                        alternate_candidate_id="alternate",
                    ),
                    "search.json",
                )
            ],
            trace_reports=[],
            selector_probe=MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
        )

        probe = report["selector_probe"]
        self.assertTrue(probe["report_only"])
        self.assertFalse(probe["active_selector_implemented"])
        self.assertEqual(probe["candidate_count_inspected"], 2)
        comparison = probe["comparisons"][0]
        self.assertTrue(comparison["selection_changes"])
        self.assertEqual(
            comparison["current_selected_candidate"]["candidate_id"],
            "current",
        )
        self.assertEqual(
            comparison["report_only_gate_aligned_selected_candidate"][
                "candidate_id"
            ],
            "alternate",
        )
        self.assertIn(
            "post_contact_survival_rate",
            comparison["missing_fields_preventing_full_recovery_selection"][
                "counts_by_field"
            ],
        )
        self.assertFalse(comparison["full_recovery_aware_selection_available"])
        self.assertEqual(
            comparison["plausible_alternates"][0]["candidate_id"],
            "alternate",
        )

    def test_cli_selector_probe_writes_retrospective_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            search = root / "search.json"
            trace = root / "trace.json"
            output = root / "audit.json"
            search.write_text(
                json.dumps(
                    _selector_search_report(
                        selected_candidate_id="current",
                        alternate_candidate_id="alternate",
                    )
                ),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    _trace_report(
                        contact_agents=1,
                        drink_rate=0.0,
                        hydration_delta=-0.2,
                        survival_rate=0.0,
                        water_distance=4.0,
                    )
                ),
                encoding="utf-8",
            )
            argv = [
                "mind_v3_carrion_objective_audit",
                "--search-report",
                f"probe={search}",
                "--trace-report",
                f"probe={trace}",
                "--selector-probe",
                MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
                "--output",
                str(output),
            ]

            with patch.object(sys, "argv", argv):
                mind_v3_carrion_objective_audit.main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("selector_probe", payload)
            self.assertEqual(
                payload["selector_probe"]["selection_change_count"],
                1,
            )

    def test_evolve_recovery_probe_report_is_report_only(self) -> None:
        probe = mind_v3_evolve._fixture_rerank_recovery_probe_report(
            candidate={"candidate_id": "g0-c0"},
            prefilter_rank=0,
            fixture_gate={
                "passed": False,
                "blockers": [
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_alive_floor",
                    }
                ],
            },
            fixture_summary={
                "per_fixture": {
                    "carrion_only": {
                        "alive_agents_mean": 0.0,
                        "births_mean": 1.0,
                        "terminal_hydration_viability_share_mean": 0.0,
                        "terminal_energy_viability_share_mean": 0.2,
                        "terminal_health_viability_share_mean": 0.3,
                        "terminal_matched_diet_viability_share_mean": 0.4,
                    }
                }
            },
            trace_report=_trace_report(
                contact_agents=2,
                drink_rate=0.25,
                hydration_delta=-0.4,
                survival_rate=0.5,
                water_distance=3.0,
            ),
            trajectory_paths=[],
            ticks=120,
        )

        self.assertTrue(probe["report_only"])
        self.assertFalse(probe["changes_candidate_selection"])
        self.assertEqual(probe["candidate_id"], "g0-c0")
        self.assertEqual(probe["fixture_blocker_count"], 1)
        self.assertEqual(probe["carrion_only_blocker_count"], 1)
        self.assertEqual(probe["drink_after_carrion_rate"], 0.25)
        self.assertEqual(probe["mean_water_distance"], 3.0)

    def test_foundation_baseline_contract_is_explicitly_non_mind(self) -> None:
        contract = foundation_heuristic_baseline_contract(
            seeds=(13, 19),
            ticks=120,
            output_path="baseline.json",
            trajectory_output_dir="trajectories",
            trace_output_path="trace.json",
        )

        self.assertTrue(contract["available"])
        self.assertFalse(contract["uses_founder_template"])
        self.assertFalse(contract["uses_mind_policy"])
        self.assertEqual(contract["policy_source"], "foundation_default_policy_no_mind")
        self.assertNotIn("--founder-template", contract["command"])

    def test_cli_can_generate_small_foundation_heuristic_baseline(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "baseline.json"
            trace = root / "trace.json"
            trajectory_dir = root / "trajectories"
            argv = [
                "mind_v3_carrion_objective_audit",
                "--generate-heuristic-baseline",
                "--baseline-seeds",
                "13",
                "--baseline-ticks",
                "1",
                "--baseline-output",
                str(output),
                "--baseline-trajectory-output-dir",
                str(trajectory_dir),
                "--baseline-trace-output",
                str(trace),
            ]

            with patch.object(sys, "argv", argv):
                mind_v3_carrion_objective_audit.main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["policy"]["policy_source"],
                "foundation_default_policy_no_mind",
            )
            self.assertFalse(payload["policy"]["uses_founder_template"])
            self.assertFalse(payload["policy"]["uses_mind_policy"])
            self.assertEqual(payload["contract"]["output_path"], str(output))
            self.assertEqual(payload["fixture"], "carrion_only")
            self.assertTrue(trace.exists())
            self.assertTrue(list(trajectory_dir.glob("*.jsonl.gz")))


def _search_report(
    *,
    candidate_id: str,
    score: float,
    births: float,
    animal_gain: float,
) -> dict[str, object]:
    return {
        "best_candidate": {
            "candidate_id": candidate_id,
            "score": score,
            "alive_agents_mean": 1.0,
            "alive_agent_ticks_mean": 100.0,
            "births_mean": births,
            "animal_resource_gain_total_mean": animal_gain,
            "unsupported_resolved_action_count": 0,
            "dominant_requested_action_share": 0.4,
        },
        "fixture_gate": {
            "passed": False,
            "blockers": [
                {
                    "fixture": "carrion_only",
                    "reason": "fixture_hydration_viability_floor",
                    "metric": "hydration_viability_share_mean",
                }
            ],
        },
        "generations": [
            {
                "generation_index": 0,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "score": score,
                        "births_mean": births,
                        "animal_resource_gain_total_mean": animal_gain,
                        "alive_agents_mean": 1.0,
                        "alive_agent_ticks_mean": 100.0,
                        "dominant_requested_action_share": 0.4,
                    }
                ],
            }
        ],
    }


def _selector_search_report(
    *,
    selected_candidate_id: str,
    alternate_candidate_id: str,
) -> dict[str, object]:
    current = _fixture_candidate(
        candidate_id=selected_candidate_id,
        score=30.0,
        blocker_count=4,
        carrion_blocker_count=3,
        passed_horizon_count=0,
        first_horizon_passed=False,
        carrion_alive=0.0,
        carrion_births=1.0,
        hydration_viability=0.0,
        energy_viability=0.0,
        health_viability=0.0,
        matched_diet_viability=0.0,
        unsupported_resolved=4,
        dominant_share=0.7,
    )
    alternate = _fixture_candidate(
        candidate_id=alternate_candidate_id,
        score=20.0,
        blocker_count=2,
        carrion_blocker_count=1,
        passed_horizon_count=1,
        first_horizon_passed=True,
        carrion_alive=1.0,
        carrion_births=2.0,
        hydration_viability=0.4,
        energy_viability=0.3,
        health_viability=0.2,
        matched_diet_viability=0.5,
        unsupported_resolved=1,
        dominant_share=0.45,
    )
    return {
        "best_candidate": {
            "candidate_id": selected_candidate_id,
            "score": 30.0,
            "alive_agents_mean": 1.0,
            "alive_agent_ticks_mean": 100.0,
            "births_mean": 1.0,
            "animal_resource_gain_total_mean": 1.0,
            "unsupported_resolved_action_count": 4,
            "dominant_requested_action_share": 0.7,
        },
        "holdout_evaluation": {
            "aggregate": {
                "unsupported_resolved_action_count": 4,
                "dominant_requested_action_share": 0.7,
            }
        },
        "fixture_gate": {
            "passed": False,
            "blockers": [
                {
                    "fixture": "carrion_only",
                    "reason": "fixture_alive_floor",
                }
            ],
        },
        "fixture_rerank": {
            "top_k": 2,
            "selected_candidate_id": selected_candidate_id,
            "candidates": [current, alternate],
        },
        "generations": [],
    }


def _fixture_candidate(
    *,
    candidate_id: str,
    score: float,
    blocker_count: int,
    carrion_blocker_count: int,
    passed_horizon_count: int,
    first_horizon_passed: bool,
    carrion_alive: float,
    carrion_births: float,
    hydration_viability: float,
    energy_viability: float,
    health_viability: float,
    matched_diet_viability: float,
    unsupported_resolved: int,
    dominant_share: float,
) -> dict[str, object]:
    blockers = [
        {
            "fixture": "carrion_only" if index < carrion_blocker_count else "mixed_stable",
            "reason": "fixture_alive_floor",
        }
        for index in range(blocker_count)
    ]
    return {
        "candidate_id": candidate_id,
        "search_score": score,
        "prefilter_rank": 0,
        "holdout_aggregate": {
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": unsupported_resolved,
        },
        "fixture_gate": {
            "passed": blocker_count == 0,
            "blockers": blockers,
        },
        "fixture_blocker_pressure": {
            "blocker_count": blocker_count,
            "carrion_only_blocker_count": carrion_blocker_count,
        },
        "fixture_horizons": [
            {"fixture_gate": {"passed": first_horizon_passed, "blockers": []}}
        ],
        "fixture_horizon_summary": {
            "passed_horizon_count": passed_horizon_count,
            "horizon_count": 2,
            "all_horizons_passed": False,
            "carrion_only_alive_agents_min": carrion_alive,
            "carrion_only_terminal_hydration_viability_share_min": (
                hydration_viability
            ),
            "carrion_only_terminal_energy_viability_share_min": energy_viability,
            "carrion_only_terminal_matched_diet_viability_share_min": (
                matched_diet_viability
            ),
            "carrion_only_dominant_requested_action_share_max": dominant_share,
        },
        "fixture_summary": {
            "carrion_only_alive_agents_mean": carrion_alive,
            "carrion_only_births_mean": carrion_births,
            "carrion_only_terminal_hydration_viability_share_mean": (
                hydration_viability
            ),
            "carrion_only_terminal_energy_viability_share_mean": energy_viability,
            "carrion_only_terminal_health_viability_share_mean": health_viability,
            "carrion_only_terminal_matched_diet_viability_share_mean": (
                matched_diet_viability
            ),
            "per_fixture": {
                "carrion_only": {
                    "dominant_requested_action_share": dominant_share,
                }
            },
        },
    }


def _trace_report(
    *,
    contact_agents: int,
    drink_rate: float,
    hydration_delta: float,
    survival_rate: float,
    water_distance: float,
) -> dict[str, object]:
    return {
        "fixture_trace": {
            "aggregate": {
                "contact_agent_count": contact_agents,
                "animal_resource_successful_eat_count": contact_agents,
                "drink_after_carrion_rate": drink_rate,
                "mean_hydration_delta_after_carrion": hydration_delta,
                "mean_energy_delta_after_carrion": 0.0,
                "mean_health_delta_after_carrion": 0.0,
                "survival_after_carrion_rate": survival_rate,
                "unsupported_resolved_action_count": 0,
                "requested_action_counts": {
                    "eat": 3,
                    "drink": 1,
                    "stay": 1,
                },
                "navigation_target_observations": {
                    "water": {
                        "mean_distance": water_distance,
                    }
                },
            }
        }
    }


if __name__ == "__main__":
    unittest.main()
