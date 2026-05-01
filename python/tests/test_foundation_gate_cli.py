from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from evolution_sim.cli.foundation_gate import (
    ECOLOGY_PROFILE,
    PROFILES,
    QUICK_PROFILE,
    FullReplayProbe,
    GateProfile,
    build_foundation_gate_report,
    _ecology_failure_rollup,
    _mind_contract_flags,
    _replay_size_bytes,
    _reproductive_role_readiness_flags,
    _summary_gate_flags,
)
from evolution_sim.config import SignalConfig, WorldConfig
from evolution_sim.env import SimulationWorld


class FoundationGateCliTests(unittest.TestCase):
    def test_ecology_profile_is_summary_only_seed_bank(self) -> None:
        self.assertIs(PROFILES["ecology"], ECOLOGY_PROFILE)
        self.assertEqual(ECOLOGY_PROFILE.summary_seeds, tuple(range(1, 21)))
        self.assertEqual(ECOLOGY_PROFILE.summary_ticks, 120)
        self.assertEqual(ECOLOGY_PROFILE.min_trophic_roles, 2)
        self.assertEqual(ECOLOGY_PROFILE.min_meat_modes, 1)
        self.assertEqual(ECOLOGY_PROFILE.min_aggregate_meat_modes, 2)
        self.assertEqual(
            ECOLOGY_PROFILE.min_animal_resource_consumption_run_share_by_mode,
            0.5,
        )
        self.assertTrue(ECOLOGY_PROFILE.use_late_window_population_floor)
        self.assertEqual(ECOLOGY_PROFILE.full_replay_probes, ())

    def test_release_profile_hardens_seed_3_and_11_terminal_timelines(self) -> None:
        release = PROFILES["release"]

        self.assertEqual(release.dominance_warning_share, 0.85)
        self.assertEqual(
            release.required_terminal_meat_mode_alternatives_by_seed,
            {3: ("hunter", "mixed"), 11: ("hunter", "mixed")},
        )

    def test_replay_size_budget_uses_compact_json_bytes(self) -> None:
        payload = {"run_id": "x", "viewer": {"frames": [{"agents": [[1, 2, 3]]}]}}

        self.assertEqual(
            _replay_size_bytes(payload),
            len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        )

    def test_ecology_gate_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:gate:ecology"]

        self.assertIn("--profile ecology", script)
        self.assertIn("--output output/evaluations/foundation-ecology-current.json", script)
        self.assertIn("--scenario-timeout-seconds", script)

    def test_quick_gate_report_is_json_serializable(self) -> None:
        profile = GateProfile(
            name="unit",
            summary_seeds=(7,),
            summary_ticks=8,
            min_alive_agents=1,
            min_births=0,
            min_last_birth_tick=0,
            min_trophic_roles=1,
            min_meat_modes=1,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(
                FullReplayProbe(
                    name="unit_full_replay",
                    seed=7,
                    ticks=8,
                    min_alive_species=1,
                    min_species_created=1,
                ),
            ),
        )

        report = build_foundation_gate_report(profile)

        self.assertEqual(report["protocol"]["profile"], "unit")
        self.assertEqual(report["summary_evaluation"]["protocol"]["mode"], "summary_only")
        self.assertEqual(len(report["full_replay_probes"]), 1)
        self.assertIn("ecology_failure_rollup", report)
        self.assertIn("terminal_role_presence_runs", report["ecology_failure_rollup"])
        self.assertIn("diet_by_meat_mode_at_end", report["ecology_failure_rollup"])
        self.assertIn("death_causes_by_meat_mode", report["ecology_failure_rollup"])
        self.assertIn(
            "reproduction_biological_blockers_by_trophic_role_at_end",
            report["ecology_failure_rollup"],
        )
        self.assertIn(
            "reproduction_energy_readiness_by_meat_mode_at_end",
            report["ecology_failure_rollup"],
        )
        self.assertIn(
            "reproduction_blocked_run_counts_by_meat_mode",
            report["ecology_failure_rollup"],
        )
        self.assertIn(
            "animal_resource_opportunity_by_meat_mode_at_end",
            report["ecology_failure_rollup"],
        )
        self.assertIn(
            "no_animal_consumption_seeds_by_meat_mode",
            report["ecology_failure_rollup"],
        )
        self.assertIn(
            "animal_resource_reachable_policy_blocked_seeds_by_meat_mode",
            report["ecology_failure_rollup"],
        )
        self.assertIn(
            "animal_resource_policy_blocker_seeds_by_meat_mode",
            report["ecology_failure_rollup"],
        )
        self.assertIn(report["readiness"]["status"], {"pass", "review", "fail"})
        self.assertTrue(report["complete"])
        self.assertIn("timings", report)
        json.dumps(report)

    def test_ecology_rollup_splits_absent_from_present_unconsumed_resources(self) -> None:
        evaluation = {
            "runs": [
                {
                    "seed": 4,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 1, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 0,
                            "hunter": 1,
                            "mixed": 0,
                        },
                        "animal_resource_opportunity_by_meat_mode": {
                            "hunter": {
                                "alive_ticks": 10,
                                "animal_resource_present_ticks": 0,
                                "animal_resource_consumption_events": 0,
                            }
                        },
                    },
                    "fresh_kill": {"energy_consumed": 0.0},
                    "carrion": {"energy_consumed": 0.0},
                },
                {
                    "seed": 10,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 1, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 1,
                            "hunter": 0,
                            "mixed": 0,
                        },
                        "animal_resource_opportunity_by_meat_mode": {
                            "scavenger": {
                                "alive_ticks": 10,
                                "animal_resource_present_ticks": 4,
                                "animal_resource_reachable_ticks": 0,
                                "animal_resource_reachable_policy_blocked_ticks": 0,
                                "animal_resource_consumption_events": 0,
                            }
                        },
                    },
                    "fresh_kill": {"energy_consumed": 0.0},
                    "carrion": {"energy_consumed": 0.0},
                },
            ],
            "aggregate": {},
        }

        rollup = _ecology_failure_rollup(evaluation, QUICK_PROFILE)

        self.assertEqual(
            rollup["no_animal_consumption_seeds_by_meat_mode"],
            {"hunter": [4], "scavenger": [10]},
        )
        self.assertEqual(
            rollup["animal_resource_absent_seeds_by_meat_mode"],
            {"hunter": [4]},
        )
        self.assertEqual(
            rollup["animal_resource_present_unconsumed_seeds_by_meat_mode"],
            {"scavenger": [10]},
        )
        self.assertEqual(
            rollup["animal_resource_present_unreachable_seeds_by_meat_mode"],
            {"scavenger": [10]},
        )
        self.assertEqual(
            rollup["animal_resource_reachable_unconsumed_seeds_by_meat_mode"],
            {},
        )
        self.assertEqual(
            rollup["animal_resource_reachable_policy_blocked_seeds_by_meat_mode"],
            {},
        )

    def test_gate_report_tracks_progress_timing_and_incremental_output(self) -> None:
        profile = GateProfile(
            name="unit-progress",
            summary_seeds=(7,),
            summary_ticks=4,
            min_alive_agents=1,
            min_births=0,
            min_last_birth_tick=0,
            min_trophic_roles=0,
            min_meat_modes=0,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
        )
        progress_messages: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gate.json"

            def capture_progress(message: str) -> None:
                progress_messages.append(message)
                if "summary seed 7 complete" in message:
                    self.assertTrue(output_path.exists())
                    partial = json.loads(output_path.read_text(encoding="utf-8"))
                    self.assertFalse(partial["complete"])
                    self.assertEqual(partial["readiness"]["status"], "running")

            report = build_foundation_gate_report(
                profile,
                progress=capture_progress,
                incremental_output_path=output_path,
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(report["complete"])
        self.assertTrue(written["complete"])
        self.assertTrue(report["timings"]["summary_seed_wall_seconds"])
        self.assertIsNotNone(report["timings"]["summary_sweep_wall_seconds"])
        self.assertIsNotNone(report["timings"]["total_wall_seconds"])
        self.assertTrue(
            any("summary sweep start" in message for message in progress_messages)
        )
        self.assertTrue(any("gate complete" in message for message in progress_messages))

    def test_summary_timeout_errors_block_gate_without_successful_runs(self) -> None:
        profile = GateProfile(
            name="unit-timeout",
            summary_seeds=(7,),
            summary_ticks=8,
            min_alive_agents=1,
            min_births=0,
            min_last_birth_tick=0,
            min_trophic_roles=0,
            min_meat_modes=0,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
        )

        def fake_summary_seed(
            *,
            seed: int,
            ticks: int,
            timeout_seconds: float | None,
        ) -> tuple[dict[str, object] | None, dict[str, object] | None, float]:
            return (
                None,
                {
                    "severity": "error",
                    "seed": seed,
                    "field": "scenario_timeout",
                    "message": f"Summary seed {seed} timed out.",
                },
                1.25,
            )

        with mock.patch(
            "evolution_sim.cli.foundation_gate._run_summary_seed",
            side_effect=fake_summary_seed,
        ):
            report = build_foundation_gate_report(
                profile,
                scenario_timeout_seconds=0.01,
            )

        self.assertEqual(report["readiness"]["status"], "fail")
        self.assertEqual(report["summary_evaluation"]["protocol"]["run_count"], 0)
        self.assertEqual(report["timings"]["summary_seed_wall_seconds"][0]["seed"], 7)
        self.assertTrue(
            any(
                flag["field"] == "scenario_timeout"
                for flag in report["readiness"]["blockers"]
            )
        )

    def test_impossible_alive_floor_blocks_gate(self) -> None:
        strict_profile = replace(
            QUICK_PROFILE,
            name="impossible",
            summary_seeds=(7,),
            summary_ticks=8,
            min_alive_agents=999,
            min_births=0,
            min_last_birth_tick=0,
            min_trophic_roles=1,
            min_meat_modes=1,
            full_replay_probes=(),
        )

        report = build_foundation_gate_report(strict_profile)

        self.assertEqual(report["readiness"]["status"], "fail")
        self.assertTrue(report["readiness"]["blockers"])

    def test_per_run_trophic_collapse_blocks_gate_even_when_aggregate_has_roles(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            name="strict-unit",
            min_trophic_roles=2,
            min_meat_modes=2,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 1,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 10, "omnivore": 0, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 10,
                            "scavenger": 0,
                            "hunter": 0,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.0},
                    },
                    "carrion": {"energy_deposited": 0.0, "energy_consumed": 0.0},
                },
                {
                    "seed": 2,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 0, "carnivore": 2},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 0,
                            "hunter": 2,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.2},
                    },
                    "carrion": {"energy_deposited": 1.0, "energy_consumed": 0.2},
                },
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 18, "omnivore": 0, "carnivore": 2}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 18, "scavenger": 0, "hunter": 2, "mixed": 0}
                },
                "ecology_state_counts_at_end": {
                    "total": {"stable": 1, "lush": 0, "recovering": 0, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        blockers = [flag for flag in flags if flag["severity"] == "error"]
        self.assertTrue(
            any(
                flag["scope"] == "summary_seed_1"
                and flag["field"] == "trophic.role_counts"
                for flag in blockers
            )
        )

    def test_release_gate_blocks_unused_carrion_pressure(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            name="strict-carrion",
            min_trophic_roles=1,
            min_meat_modes=1,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            min_animal_energy_share=0.01,
            min_carrion_consumed_deposited_ratio=0.01,
            full_replay_probes=(),
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 7,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 0, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 0,
                            "hunter": 0,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.0},
                    },
                    "carrion": {"energy_deposited": 100.0, "energy_consumed": 0.2},
                }
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 8, "omnivore": 0, "carnivore": 0}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 8, "scavenger": 0, "hunter": 0, "mixed": 0}
                },
                "ecology_state_counts_at_end": {
                    "total": {"stable": 1, "lush": 0, "recovering": 0, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "trophic.diet.animal_energy_share"
                for flag in flags
            )
        )
        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "carrion.energy_consumed_ratio"
                for flag in flags
            )
        )

    def test_min_meat_modes_uses_declared_floor(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            name="strict-meat-modes",
            min_trophic_roles=1,
            min_meat_modes=2,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 7,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 0, "carnivore": 2},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 0,
                            "hunter": 2,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.2},
                    },
                    "carrion": {"energy_deposited": 1.0, "energy_consumed": 0.2},
                }
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 8, "omnivore": 0, "carnivore": 2}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 8, "scavenger": 0, "hunter": 2, "mixed": 0}
                },
                "ecology_state_counts_at_end": {
                    "total": {"stable": 1, "lush": 0, "recovering": 0, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}
        self.assertIn("trophic.meat_mode_counts", error_fields)
        self.assertIn("meat_mode_counts_at_end", error_fields)

    def test_animal_resource_consuming_run_share_floor_blocks_regression(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            name="animal-resource-run-share",
            min_trophic_roles=1,
            min_meat_modes=1,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            min_animal_resource_consumption_run_share_by_mode=0.5,
            full_replay_probes=(),
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 7,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 0, "carnivore": 2},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 0,
                            "hunter": 2,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.2},
                    },
                    "carrion": {"energy_deposited": 1.0, "energy_consumed": 0.2},
                }
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 8, "omnivore": 0, "carnivore": 2}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 8, "scavenger": 0, "hunter": 2, "mixed": 0}
                },
                "animal_resource_opportunity_run_counts_by_meat_mode": {
                    "hunter": {
                        "alive_runs": 4,
                        "no_animal_consumption_runs": 3,
                    },
                },
                "ecology_state_counts_at_end": {
                    "total": {"stable": 1, "lush": 0, "recovering": 0, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"]
                == (
                    "animal_resource_opportunity_run_counts_by_meat_mode."
                    "hunter.consuming_run_share"
                )
                for flag in flags
            )
        )

    def test_ecology_rollup_reports_mode_persistence_and_live_blockers(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            min_trophic_roles=1,
            min_meat_modes=1,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
        )
        evaluation = {
            "runs": [
                {
                    "seed": 7,
                    "trophic": {
                        "role_counts": {"herbivore": 5, "omnivore": 1, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 5,
                            "scavenger": 1,
                            "hunter": 0,
                            "mixed": 0,
                        },
                        "animal_resource_opportunity_by_meat_mode": {
                            "scavenger": {
                                "alive_ticks": 3,
                                "animal_resource_present_ticks": 2,
                                "animal_resource_reachable_ticks": 1,
                                "animal_resource_reachable_policy_blocked_ticks": 1,
                                "animal_resource_policy_blocked_by_hazard_ticks": 1,
                                "carcass_policy_blocked_by_water_ticks": 1,
                                "fresh_kill_policy_blocked_by_movement_mask_ticks": 1,
                                "animal_resource_consumption_events": 0,
                            }
                        },
                    },
                    "carrion": {"energy_consumed": 0.1},
                    "fresh_kill": {"energy_consumed": 0.0},
                    "trophic_lifecycle": {
                        "meat_mode_persistence": {
                            "last_alive_tick_by_meat_mode": {
                                "none": 39,
                                "hunter": 20,
                                "scavenger": 39,
                                "mixed": None,
                            },
                            "deaths_by_meat_mode_by_tick_band": {
                                "early": {
                                    "none": 0,
                                    "hunter": 0,
                                    "scavenger": 0,
                                    "mixed": 0,
                                },
                                "mid": {
                                    "none": 0,
                                    "hunter": 1,
                                    "scavenger": 0,
                                    "mixed": 0,
                                },
                            },
                            "death_causes_by_meat_mode_by_tick_band": {
                                "mid": {
                                    "hunter": {"energy_depletion": 1},
                                    "scavenger": {},
                                }
                            },
                            "births_by_parent_meat_mode_by_tick_band": {},
                            "births_by_child_meat_mode_by_tick_band": {},
                        },
                        "late_window": {
                            "presence_ticks_by_trophic_role": {
                                "herbivore": 3,
                                "omnivore": 3,
                                "carnivore": 0,
                            },
                            "presence_ticks_by_meat_mode": {
                                "none": 3,
                                "hunter": 0,
                                "scavenger": 3,
                                "mixed": 0,
                            },
                        },
                    },
                }
            ],
            "aggregate": {
                "trophic_lifecycle": {
                    "death_causes": {},
                    "death_causes_by_trophic_role": {},
                    "death_causes_by_meat_mode": {},
                    "meat_mode_persistence": {
                        "last_alive_tick_by_meat_mode": {
                            "hunter": {"count": 1, "min": 20},
                            "scavenger": {"count": 1, "min": 39},
                        },
                        "death_causes_by_meat_mode_by_tick_band": {
                            "mid": {
                                "hunter": {
                                    "total": {"energy_depletion": 1},
                                    "per_run_mean": {"energy_depletion": 1.0},
                                }
                            }
                        },
                    },
                },
                "reproduction_by_meat_mode_at_end": {
                    "hunter": {"total": {"alive_agents": 0}},
                    "scavenger": {"total": {"alive_agents": 1}},
                    "mixed": {"total": {"alive_agents": 0}},
                    "none": {"total": {"alive_agents": 5}},
                },
                "reproduction_biological_blockers_by_meat_mode_at_end": {
                    "hunter": {"total": {"energy": 0}},
                    "scavenger": {"total": {"energy": 1, "matched_diet": 1}},
                },
            },
        }

        rollup = _ecology_failure_rollup(evaluation, profile)

        self.assertEqual(
            rollup["meat_mode_persistence_by_seed"]["7"][
                "last_alive_tick_by_meat_mode"
            ]["hunter"],
            20,
        )
        live_blockers = rollup["terminal_biological_blockers_by_live_meat_mode"]
        self.assertNotIn("hunter", live_blockers)
        self.assertIn("scavenger", live_blockers)
        self.assertEqual(
            rollup["meat_mode_persistence_by_seed"]["7"][
                "death_causes_by_meat_mode_by_tick_band"
            ]["mid"]["hunter"]["energy_depletion"],
            1,
        )
        self.assertEqual(
            rollup["animal_resource_opportunity_by_seed"]["7"]["scavenger"][
                "animal_resource_reachable_ticks"
            ],
            1,
        )
        self.assertEqual(
            rollup["animal_resource_reachable_policy_blocked_seeds_by_meat_mode"],
            {"scavenger": [7]},
        )
        self.assertEqual(
            rollup["animal_resource_policy_blocker_seeds_by_meat_mode"][
                "scavenger"
            ]["animal_resource"]["hazard"],
            [7],
        )
        self.assertEqual(
            rollup["animal_resource_policy_blocker_seeds_by_meat_mode"][
                "scavenger"
            ]["carcass"]["water"],
            [7],
        )
        self.assertEqual(
            rollup["animal_resource_policy_blocker_seeds_by_meat_mode"][
                "scavenger"
            ]["fresh_kill"]["movement_mask"],
            [7],
        )
        self.assertEqual(
            rollup["meat_mode_persistence"][
                "death_causes_by_meat_mode_by_tick_band"
            ]["mid"]["hunter"]["total"]["energy_depletion"],
            1,
        )

    def test_late_window_population_floor_ignores_single_terminal_tick_loss(self) -> None:
        profile = replace(
            ECOLOGY_PROFILE,
            name="late-window-unit",
            summary_seeds=(7,),
            min_trophic_roles=2,
            min_meat_modes=2,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 7,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 0, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 1,
                            "hunter": 0,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.1},
                    },
                    "trophic_lifecycle": {
                        "late_window": {
                            "presence_ticks_by_trophic_role": {
                                "herbivore": 3,
                                "omnivore": 3,
                                "carnivore": 0,
                            },
                            "presence_ticks_by_meat_mode": {
                                "none": 3,
                                "scavenger": 3,
                                "hunter": 1,
                                "mixed": 0,
                            },
                        }
                    },
                    "carrion": {"energy_deposited": 1.0, "energy_consumed": 0.2},
                }
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 8, "omnivore": 0, "carnivore": 0}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 8, "scavenger": 1, "hunter": 0, "mixed": 0}
                },
                "trophic_lifecycle": {
                    "late_window_trophic_role_presence_runs": {
                        "herbivore": 1,
                        "omnivore": 1,
                        "carnivore": 0,
                    },
                    "late_window_meat_mode_presence_runs": {
                        "none": 1,
                        "scavenger": 1,
                        "hunter": 1,
                        "mixed": 0,
                    },
                },
                "ecology_state_counts_at_end": {
                    "total": {"stable": 1, "lush": 0, "recovering": 0, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}
        self.assertNotIn(
            "trophic_lifecycle.late_window.presence_ticks_by_meat_mode",
            error_fields,
        )
        self.assertNotIn(
            "trophic_lifecycle.late_window_meat_mode_presence_runs",
            error_fields,
        )

    def test_required_terminal_meat_mode_alternatives_use_last_alive_timelines(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            name="terminal-timeline-unit",
            min_trophic_roles=1,
            min_meat_modes=1,
            min_last_birth_tick=0,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=0,
            full_replay_probes=(),
            required_terminal_meat_mode_alternatives_by_seed={
                3: ("hunter", "mixed"),
            },
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 3,
                    "ticks_executed": 800,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 1},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 1,
                            "hunter": 1,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.1},
                    },
                    "trophic_lifecycle": {
                        "meat_mode_persistence": {
                            "last_alive_tick_by_meat_mode": {
                                "none": 799,
                                "scavenger": 799,
                                "hunter": 790,
                                "mixed": None,
                            },
                        },
                    },
                    "carrion": {"energy_deposited": 1.0, "energy_consumed": 0.2},
                }
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 8, "omnivore": 1}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 8, "scavenger": 1, "hunter": 1, "mixed": 0}
                },
                "trophic_lifecycle": {},
                "ecology_state_counts_at_end": {
                    "total": {"stable": 0, "lush": 0, "recovering": 1, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"]
                == (
                    "trophic_lifecycle.meat_mode_persistence."
                    "last_alive_tick_by_meat_mode"
                )
                for flag in flags
            )
        )

        passing_evaluation = copy.deepcopy(evaluation)
        passing_evaluation["runs"][0]["trophic_lifecycle"][
            "meat_mode_persistence"
        ]["last_alive_tick_by_meat_mode"]["hunter"] = 799

        passing_flags = _summary_gate_flags(passing_evaluation, profile)

        self.assertFalse(
            any(
                flag["field"]
                == (
                    "trophic_lifecycle.meat_mode_persistence."
                    "last_alive_tick_by_meat_mode"
                )
                for flag in passing_flags
            )
        )

    def test_missing_ecology_pressure_is_blocking(self) -> None:
        profile = replace(
            QUICK_PROFILE,
            name="strict-ecology",
            min_trophic_roles=1,
            min_meat_modes=1,
            min_hazardous_tiles=0,
            min_ecology_pressure_tiles=1,
            full_replay_probes=(),
        )
        evaluation = {
            "flags": [],
            "runs": [
                {
                    "seed": 7,
                    "last_birth_tick": 8,
                    "trophic": {
                        "role_counts": {"herbivore": 8, "omnivore": 0, "carnivore": 0},
                        "meat_mode_counts": {
                            "none": 8,
                            "scavenger": 1,
                            "hunter": 0,
                            "mixed": 0,
                        },
                        "diet": {"animal_energy_share": 0.1},
                    },
                    "carrion": {"energy_deposited": 1.0, "energy_consumed": 0.2},
                }
            ],
            "aggregate": {
                "hazardous_tiles": {"min": 1},
                "trophic_role_counts_at_end": {
                    "total": {"herbivore": 8, "omnivore": 0, "carnivore": 0}
                },
                "meat_mode_counts_at_end": {
                    "total": {"none": 8, "scavenger": 1, "hunter": 0, "mixed": 0}
                },
                "ecology_state_counts_at_end": {
                    "total": {"stable": 100, "lush": 0, "recovering": 0, "depleted": 0}
                },
                "carrion_energy_consumed": {"max": 0.2},
                "fresh_kill_energy_consumed": {"max": 0.0},
            },
        }

        flags = _summary_gate_flags(evaluation, profile)

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "ecology_state_counts_at_end"
                for flag in flags
            )
        )

    def test_missing_mind_contract_blocks_full_replay_probe(self) -> None:
        flags = _mind_contract_flags(scope="unit", summary={}, viewer={})

        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}
        self.assertIn("summary.mind_contracts", error_fields)
        self.assertIn("viewer.trajectory", error_fields)

    def test_stale_observation_input_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        viewer["trajectory"]["records"][0]["observation_input"] = {
            "schema_version": "stale",
        }

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "viewer.trajectory.records.observation_input"
                for flag in flags
            )
        )

    def test_stale_signal_contract_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        viewer["trajectory"]["observation_contract"]["signal_contract"][
            "schema_version"
        ] = "stale"

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"]
                == "viewer.trajectory.observation_contract.signal_contract"
                for flag in flags
            )
        )

    def test_stale_stage1_contract_versions_block_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        summary = copy.deepcopy(result.summary)
        viewer = copy.deepcopy(result.viewer)
        summary["mind_contracts"]["action_contract_version"] = "stale"
        summary["mind_contracts"]["reproductive_group_contract_version"] = "stale"
        viewer["trajectory"]["observation_schema_version"] = "stale"
        viewer["trajectory"]["genome_recombination_contract_version"] = "stale"
        viewer["trajectory"]["reward_schema_version"] = "stale"
        viewer["trajectory"]["action_outcome_schema_version"] = "stale"
        viewer["trajectory"]["action_contract"]["schema_version"] = "stale"
        viewer["trajectory"]["action_contract"]["communication"][
            "emission_enabled"
        ] = True
        viewer["trajectory"]["action_contract"]["active_action_keys"].append(
            "signal_0_profile_0"
        )
        viewer["trajectory"]["observation_contract"]["action_contract"][
            "schema_version"
        ] = "stale"
        viewer["trajectory"]["reproductive_group_contract"]["schema_version"] = "stale"
        viewer["trajectory"]["reward_contract"]["schema_version"] = "stale"
        viewer["reproductive_group_catalog"]["schema_version"] = "stale"
        viewer["trajectory"]["records"][0]["observation_schema"] = "stale"
        viewer["trajectory"]["records"][0]["reward"]["schema_version"] = "stale"
        viewer["trajectory"]["records"][0]["outcome"]["schema_version"] = "stale"
        del viewer["trajectory"]["records"][0]["outcome"]["signal"]

        flags = _mind_contract_flags(
            scope="unit",
            summary=summary,
            viewer=viewer,
        )
        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}

        self.assertIn("summary.mind_contracts.action_contract_version", error_fields)
        self.assertIn(
            "summary.mind_contracts.reproductive_group_contract_version",
            error_fields,
        )
        self.assertIn(
            "viewer.trajectory.observation_schema_version",
            error_fields,
        )
        self.assertIn(
            "viewer.trajectory.genome_recombination_contract_version",
            error_fields,
        )
        self.assertIn("viewer.trajectory.reward_schema_version", error_fields)
        self.assertIn("viewer.trajectory.action_outcome_schema_version", error_fields)
        self.assertIn(
            "viewer.trajectory.action_contract.schema_version",
            error_fields,
        )
        self.assertIn("viewer.trajectory.action_contract", error_fields)
        self.assertIn(
            "viewer.trajectory.observation_contract.action_contract.schema_version",
            error_fields,
        )
        self.assertIn(
            "viewer.trajectory.reproductive_group_contract.schema_version",
            error_fields,
        )
        self.assertIn("viewer.trajectory.reward_contract", error_fields)
        self.assertIn("viewer.reproductive_group_catalog", error_fields)
        self.assertIn("viewer.trajectory.records.observation_schema", error_fields)
        self.assertIn("viewer.trajectory.records.reward", error_fields)
        self.assertIn("viewer.trajectory.records.outcome", error_fields)
        self.assertIn("viewer.trajectory.records.outcome.signal", error_fields)

    def test_stale_later_trajectory_record_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        self.assertGreater(len(viewer["trajectory"]["records"]), 1)
        viewer["trajectory"]["records"][1]["outcome"]["schema_version"] = "stale"

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "viewer.trajectory.records.outcome"
                and "record 1" in flag["message"]
                for flag in flags
            )
        )

    def test_mismatched_reproductive_group_catalog_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        first_group = next(iter(viewer["reproductive_group_catalog"]["groups"].values()))
        first_group["member_count"] += 1

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"]
                == "viewer.reproductive_group_catalog.groups.member_count"
                for flag in flags
            )
        )

    def test_mismatched_reproductive_group_birth_events_block_full_replay_probe(
        self,
    ) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=40)).run()
        viewer = copy.deepcopy(result.viewer)
        reproduction_event = next(
            event for event in result.events if event.get("type") == "agent_reproduced"
        )
        group_key = str(reproduction_event["data"]["child_reproductive_group_id"])
        viewer["reproductive_group_catalog"]["groups"][group_key]["asexual_births"] += 1

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
            events=result.events,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"]
                == "viewer.reproductive_group_catalog.groups.asexual_births"
                for flag in flags
            )
        )

    def test_stale_reproductive_birth_event_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=40)).run()
        events = copy.deepcopy(result.events)
        reproduction_event = next(
            event for event in events if event.get("type") == "agent_reproduced"
        )
        reproduction_event["data"]["schema_version"] = "stale"

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=result.viewer,
            events=events,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "events.agent_reproduced.schema_version"
                for flag in flags
            )
        )

    def test_mismatched_reproductive_summary_birth_events_block_full_replay_probe(
        self,
    ) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=40)).run()
        summary = copy.deepcopy(result.summary)
        summary["births"] += 1
        summary["reproductive_groups_end"]["asexual_births"] += 1
        summary["reproductive_groups_end"]["group_count"] += 1
        summary["reproductive_groups_end"]["alive_expression_counts"] = {
            "asexual": 999,
        }

        flags = _mind_contract_flags(
            scope="unit",
            summary=summary,
            viewer=result.viewer,
            events=result.events,
        )
        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}

        self.assertIn("summary.births", error_fields)
        self.assertIn("summary.reproductive_groups_end.asexual_births", error_fields)
        self.assertIn("summary.reproductive_groups_end.group_count", error_fields)
        self.assertIn(
            "summary.reproductive_groups_end.alive_expression_counts",
            error_fields,
        )

    def test_unknown_reproductive_catalog_expression_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        first_agent = next(iter(viewer["agent_catalog"].values()))
        first_agent["reproductive_expression"] = "mystery"

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "viewer.agent_catalog.reproductive_expression"
                for flag in flags
            )
        )

    def test_invalid_reproductive_agent_catalog_fields_block_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        first_agent = next(iter(viewer["agent_catalog"].values()))
        first_agent["reproductive_group_id"] = None
        first_agent["death_tick"] = "later"

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )
        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}

        self.assertIn("viewer.agent_catalog.reproductive_group_id", error_fields)
        self.assertIn("viewer.agent_catalog.death_tick", error_fields)

    def test_malformed_reproductive_group_ids_block_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        first_agent = next(iter(viewer["agent_catalog"].values()))
        first_agent["reproductive_group_id"] = "1"
        first_group = next(iter(viewer["reproductive_group_catalog"]["groups"].values()))
        first_group["group_id"] = "1"

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )
        error_fields = {flag["field"] for flag in flags if flag["severity"] == "error"}

        self.assertIn("viewer.agent_catalog.reproductive_group_id", error_fields)
        self.assertIn(
            "viewer.reproductive_group_catalog.groups.group_id",
            error_fields,
        )

    def test_role_stage_scarcity_emits_readiness_warnings(self) -> None:
        flags = _reproductive_role_readiness_flags(
            scope="unit",
            reproduction={
                "reproductive_stage_counts": {"stage3_x_y_z": 2},
                "reproductive_expression_counts": {"x": 2},
                "reproductive_capability_counts": {
                    "proto_role_differentiation": 2,
                    "xyz_expression": 2,
                },
                "ready_by_reproductive_stage": {},
                "mate_search_run_counts": {
                    "sexual_searches": 3,
                    "sexual_successes": 0,
                    "fallback_expression_incompatible": 2,
                    "fallback_no_compatible_partner": 1,
                    "constraint_expression_incompatible": 3,
                },
            },
        )
        warning_fields = {
            flag["field"] for flag in flags if flag["severity"] == "warning"
        }

        self.assertIn("reproduction.reproductive_expression_counts", warning_fields)
        self.assertIn(
            "reproduction.reproductive_expression_counts.z_plastic",
            warning_fields,
        )
        self.assertIn("reproduction.ready_by_reproductive_stage", warning_fields)
        self.assertIn("reproduction.mate_search_run_counts", warning_fields)
        self.assertIn(
            "reproduction.mate_search_run_counts.constraint_expression_incompatible",
            warning_fields,
        )

    def test_stage0_reproduction_does_not_emit_role_readiness_warnings(self) -> None:
        flags = _reproductive_role_readiness_flags(
            scope="unit",
            reproduction={
                "reproductive_stage_counts": {"stage0_asexual": 20},
                "reproductive_expression_counts": {"asexual": 20},
                "reproductive_capability_counts": {
                    "proto_role_differentiation": 0,
                    "xyz_expression": 0,
                },
                "ready_by_reproductive_stage": {},
                "mate_search_run_counts": {},
            },
        )

        self.assertEqual(flags, [])

    def test_mismatched_action_signal_capacity_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=4,
                signals=SignalConfig(
                    communication_token_count=2,
                    communication_profiles_per_token=3,
                ),
            )
        ).run()
        viewer = copy.deepcopy(result.viewer)
        viewer["trajectory"]["action_contract"]["communication"]["token_count"] = 3

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "viewer.trajectory.action_contract"
                for flag in flags
            )
        )

    def test_missing_signal_emission_metadata_blocks_full_replay_probe(self) -> None:
        result = SimulationWorld(WorldConfig(seed=7, max_ticks=4)).run()
        viewer = copy.deepcopy(result.viewer)
        del viewer["frames"][-1]["signal_emissions"]

        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=viewer,
        )

        self.assertTrue(
            any(
                flag["severity"] == "error"
                and flag["field"] == "viewer.frames.signal_emissions"
                for flag in flags
            )
        )


if __name__ == "__main__":
    unittest.main()
