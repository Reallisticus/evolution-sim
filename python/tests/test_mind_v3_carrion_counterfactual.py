from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_counterfactual
from evolution_sim.mind.carrion_counterfactual import (
    MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
    CarrionCounterfactualPolicy,
    build_carrion_counterfactual_report,
)


class MindV3CarrionCounterfactualTests(unittest.TestCase):
    def test_counterfactual_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-counterfactual"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_counterfactual"
            ),
        )

    def test_policy_drinks_when_water_accessible_and_hydration_low(self) -> None:
        policy = CarrionCounterfactualPolicy("water_first_recovery")

        decision = policy.decide(
            _observation(
                energy=0.7,
                hydration=0.45,
                center_animal_resource=0.4,
                water_dx=0,
                water_dy=0,
                water_distance=0,
            ),
            _action_mask(drink=True),
        )

        self.assertEqual(decision.requested_action, "drink")
        self.assertEqual(
            decision.source,
            "counterfactual_script:water_first_recovery",
        )
        self.assertNotIn("heuristic", decision.source)

    def test_report_records_same_fixture_seed_scope(self) -> None:
        report = build_carrion_counterfactual_report(
            seeds=(29,),
            ticks=1,
            scripts=("conserve_after_carrion",),
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
        )
        self.assertFalse(report["scope"]["state_restore_available"])
        self.assertEqual(report["aggregate"]["script_count"], 1)
        self.assertEqual(report["aggregate"]["run_count"], 1)
        script = report["scripts"][0]
        self.assertEqual(script["script_name"], "conserve_after_carrion")
        self.assertEqual(
            script["acceptance"]["heuristic_action_source_count"],
            0,
        )
        self.assertTrue(script["acceptance"]["diagnostic_acceptance_passed"])

    def test_counterfactual_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "counterfactual.json"

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_counterfactual",
                    "--seeds",
                    "29",
                    "--ticks",
                    "1",
                    "--script",
                    "conserve_after_carrion",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_counterfactual.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["script_count"], 1)
        self.assertEqual(
            payload["scripts"][0]["acceptance"]["heuristic_action_source_count"],
            0,
        )


def _observation(
    *,
    energy: float,
    hydration: float,
    center_animal_resource: float,
    water_dx: int = 0,
    water_dy: int = -1,
    water_distance: int = 1,
) -> dict[str, object]:
    return {
        "self": {
            "energy_ratio": energy,
            "hydration_ratio": hydration,
            "health_ratio": 0.9,
            "matched_diet_ratio": 0.7,
        },
        "local_patch": [
            {
                "dx": 0,
                "dy": 0,
                "in_bounds": True,
                "terrain": "rocky",
                "occupant": "self",
                "food": 0.0,
                "fresh_kill_energy": center_animal_resource,
                "carcass_energy": 0.0,
            }
        ],
        "navigation": {
            "water": {
                "dx": water_dx,
                "dy": water_dy,
                "distance": water_distance,
                "strength": 1.0,
            },
            "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
        },
    }


def _action_mask(*, drink: bool = False) -> dict[str, bool]:
    return {
        "stay": True,
        "eat": True,
        "drink": drink,
        "move_north": True,
        "move_south": True,
        "move_east": True,
        "move_west": True,
        "attack_north": False,
        "attack_south": False,
        "attack_east": False,
        "attack_west": False,
        "mate": False,
    }
