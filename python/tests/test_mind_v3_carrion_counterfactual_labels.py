from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_counterfactual_labels
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.carrion_counterfactual_labels import (
    MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
    build_carrion_counterfactual_label_report,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset


class MindV3CarrionCounterfactualLabelTests(unittest.TestCase):
    def test_counterfactual_labels_have_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-counterfactual-labels"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_counterfactual_labels"
            ),
        )

    def test_label_report_preserves_action_support_and_primary_targets(self) -> None:
        dataset = _dataset(
            (
                _record(
                    tick=0,
                    action="eat",
                    before_energy=0.62,
                    after_energy=0.78,
                    food_source="carcass",
                    gained_energy=0.4,
                ),
                _record(
                    tick=1,
                    action="drink",
                    before_energy=0.78,
                    after_energy=0.74,
                    after_hydration=0.92,
                    reproduced=True,
                ),
                _record(
                    tick=0,
                    agent_id=2,
                    action="stay",
                    before_energy=0.3,
                    after_energy=0.29,
                    source_script="carrion_then_water",
                ),
            )
        )

        report = build_carrion_counterfactual_label_report(
            [dataset],
            horizons=(1,),
            primary_horizon=1,
            source_counterfactual_report={
                "aggregate": {
                    "successful_scripts": ["hydration_safe_carrion_cycle"],
                }
            },
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        )
        self.assertEqual(report["aggregate"]["label_count"], 2)
        self.assertEqual(
            report["aggregate"]["source_script_counts"],
            {"hydration_safe_carrion_cycle": 2},
        )
        first = report["labels"][0]
        self.assertEqual(first["source_script"], "hydration_safe_carrion_cycle")
        self.assertEqual(first["fixture_seed"], 29)
        self.assertEqual(first["action_support"]["logged_action"], "eat")
        self.assertTrue(first["action_support"]["logged_action_legal"])
        self.assertIn("drink", first["action_support"]["legal_actions"])
        target = first["primary_target"]
        self.assertTrue(target["terminal_alive"])
        self.assertEqual(target["terminal_state"]["hydration_ratio"], 0.92)
        self.assertEqual(target["animal_resource"]["gained_energy"], 0.4)
        self.assertGreater(target["action_value"]["score"], 0.7)
        rollout = first["rollout_terminal_target"]
        self.assertTrue(rollout["terminal_alive"])
        self.assertEqual(rollout["animal_resource_gain_to_terminal"], 0.4)
        self.assertGreater(rollout["action_value"]["score"], 0.7)

    def test_counterfactual_labels_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "trajectory.jsonl.gz"
            output_path = tmp_path / "labels.json"
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="counterfactual-label-smoke",
            )
            SimulationWorld(WorldConfig(seed=7, max_ticks=1)).run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_counterfactual_labels",
                    "--trajectory",
                    str(trajectory_path),
                    "--horizons",
                    "1",
                    "--primary-horizon",
                    "1",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_counterfactual_labels.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        )
        self.assertGreater(payload["aggregate"]["label_count"], 0)
        self.assertEqual(payload["aggregate"]["primary_horizon"], 1)


def _dataset(records: tuple[dict[str, object], ...]) -> TrajectoryJsonlDataset:
    provenance = {
        "source_seeds": [29],
        "config_digest": "synthetic-config",
        "contract_digest": "synthetic-contract",
        "split_id": "synthetic-counterfactual",
        "trajectory_paths": ["synthetic-counterfactual.jsonl.gz"],
        "record_count": len(records),
    }
    return TrajectoryJsonlDataset(
        path=Path("synthetic-counterfactual.jsonl.gz"),
        header={"provenance": provenance},
        records=records,
        footer={"provenance": provenance},
    )


def _record(
    *,
    tick: int,
    action: str,
    before_energy: float,
    after_energy: float,
    agent_id: int = 1,
    after_hydration: float = 0.82,
    food_source: str | None = None,
    gained_energy: float = 0.0,
    reproduced: bool = False,
    source_script: str = "hydration_safe_carrion_cycle",
) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": agent_id,
        "lineage_id": agent_id,
        "runtime_species_id": agent_id + 10,
        "runtime_ecotype_id": None,
        "before": {
            "alive": True,
            "energy_ratio": before_energy,
            "hydration_ratio": after_hydration,
            "health_ratio": 0.88,
        },
        "after": {
            "alive": True,
            "energy_ratio": after_energy,
            "hydration_ratio": after_hydration,
            "health_ratio": 0.86,
        },
        "outcome": {
            "reproduced": reproduced,
            "died": False,
            "resource_gain": gained_energy,
            "feeding": {
                "ate": food_source is not None,
                "food_source": food_source,
                "gained_energy": gained_energy,
            },
            "passive": {
                "hazard_damage_taken": 0.0,
                "attack_damage_taken": 0.0,
            },
        },
        "requested_action": action,
        "resolved_action": action,
        "action_valid": True,
        "resolution_action_valid": True,
        "action_mask": {
            "stay": True,
            "eat": True,
            "drink": True,
            "move_north": True,
            "move_south": False,
            "move_east": False,
            "move_west": False,
            "attack_north": False,
            "attack_south": False,
            "attack_east": False,
            "attack_west": False,
            "mate": False,
        },
        "action_source": f"counterfactual_script:{source_script}",
        "policy_id": f"mind_v3_counterfactual_{source_script}",
        "policy_version": "scripted_policy_visible_carrion_water_recovery_v1",
    }
