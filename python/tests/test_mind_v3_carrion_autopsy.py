from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_autopsy
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.carrion_autopsy import (
    MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
    build_carrion_autopsy_report,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset


class MindV3CarrionAutopsyTests(unittest.TestCase):
    def test_autopsy_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-autopsy"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_autopsy"
            ),
        )

    def test_autopsy_identifies_post_contact_movement_energy_death(self) -> None:
        records = (
            _trajectory_record(
                tick=0,
                action="eat",
                before_energy=0.4,
                after_energy=0.7,
                after_hydration=0.9,
                after_health=1.0,
                food_source="carcass",
                gained_energy=0.31,
            ),
            _trajectory_record(
                tick=1,
                action="move_east",
                before_energy=0.7,
                after_energy=0.08,
                after_hydration=0.6,
                after_health=0.9,
            ),
            _trajectory_record(
                tick=2,
                action="move_east",
                before_energy=0.08,
                after_energy=-0.02,
                after_hydration=0.55,
                after_health=0.88,
                died=True,
                death_cause="energy_depletion",
            ),
        )
        dataset = _dataset(records)

        report = build_carrion_autopsy_report(
            [dataset],
            post_contact_window_ticks=120,
            sequence_record_limit=4,
            max_examples=2,
        )

        aggregate = report["aggregate"]
        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertEqual(aggregate["contact_episode_count"], 1)
        self.assertEqual(aggregate["death_after_contact_count"], 1)
        self.assertEqual(
            aggregate["dominant_death_path"]["path"],
            "movement_energy_depletion_after_carrion_contact",
        )
        self.assertEqual(
            aggregate["terminal_bottleneck_counts"],
            {"energy": 1},
        )
        example = report["examples"][0]
        self.assertEqual(example["death_tick"], 2)
        self.assertEqual(example["animal_resource_event_count"], 1)
        self.assertEqual(example["movement_action_count"], 2)
        self.assertEqual(len(example["sequence_excerpt"]), 3)
        self.assertIn(
            "continued spending movement energy",
            example["dominant_failure_hypothesis"],
        )

    def test_autopsy_cli_writes_empty_contact_report_for_valid_trajectory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            output_path = Path(tmpdir) / "autopsy.json"
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="autopsy-smoke",
            )
            SimulationWorld(WorldConfig(seed=7, max_ticks=1)).run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_autopsy",
                    "--trajectory",
                    str(trajectory_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_autopsy.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["contact_episode_count"], 0)
        self.assertEqual(payload["examples"], [])


def _dataset(records: tuple[dict[str, object], ...]) -> TrajectoryJsonlDataset:
    provenance = {
        "source_seeds": [1],
        "config_digest": "synthetic-config",
        "contract_digest": "synthetic-contract",
        "split_id": "synthetic-carrion",
        "trajectory_paths": ["synthetic-carrion.jsonl.gz"],
        "record_count": len(records),
    }
    return TrajectoryJsonlDataset(
        path=Path("synthetic-carrion.jsonl.gz"),
        header={"provenance": provenance},
        records=records,
        footer={"provenance": provenance},
    )


def _trajectory_record(
    *,
    tick: int,
    action: str,
    before_energy: float,
    after_energy: float,
    after_hydration: float,
    after_health: float,
    food_source: str | None = None,
    gained_energy: float = 0.0,
    died: bool = False,
    death_cause: str | None = None,
) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": 1,
        "requested_action": action,
        "resolved_action": action,
        "before": {
            "alive": True,
            "energy_ratio": before_energy,
            "hydration_ratio": after_hydration,
            "health_ratio": after_health,
            "x": tick,
            "y": 0,
        },
        "after": {
            "alive": not died,
            "energy_ratio": after_energy,
            "hydration_ratio": after_hydration,
            "health_ratio": after_health,
            "x": tick + (1 if action.startswith("move_") else 0),
            "y": 0,
        },
        "outcome": {
            "died": died,
            "reproduced": False,
            "feeding": {
                "ate": food_source is not None,
                "food_source": food_source,
                "gained_energy": gained_energy,
            },
            "drinking": {"drank": False},
            "movement": {"moved": action.startswith("move_")},
            "passive": {
                "death_cause": death_cause,
                "died_after_action": died,
            },
        },
    }
