from __future__ import annotations

import json
import unittest
from dataclasses import replace

from evolution_sim.cli.foundation_gate import (
    QUICK_PROFILE,
    FullReplayProbe,
    GateProfile,
    build_foundation_gate_report,
    _mind_contract_flags,
    _summary_gate_flags,
)


class FoundationGateCliTests(unittest.TestCase):
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
        self.assertIn(report["readiness"]["status"], {"pass", "review", "fail"})
        json.dumps(report)

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


if __name__ == "__main__":
    unittest.main()
