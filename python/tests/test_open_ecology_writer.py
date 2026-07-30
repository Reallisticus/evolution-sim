from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evolution_sim.config.schema import SignalConfig
from evolution_sim.config.schema import WorldConfig
from evolution_sim.env.runtime.collectors import RunMode
from evolution_sim.env.runtime.trajectory import trajectory_contract
from evolution_sim.env.world import SimulationWorld
from evolution_sim.io.open_ecology_writer import (
    BoundedOpenEcologyTrajectoryWriter,
    ReplayWindow,
)


class OpenEcologyWriterTests(unittest.TestCase):
    def test_deterministic_stream_has_meaningful_ecology_action_and_social_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "first.jsonl"
            second_path = Path(tmpdir) / "second.jsonl"
            records = self._meaningful_records()

            for path in (first_path, second_path):
                writer = self._writer(path)
                writer.begin(
                    run_id="deterministic-run",
                    config={"seed": 17, "max_ticks": 3},
                    contract=self._contract(),
                )
                for record in records:
                    writer.write_record(deepcopy(record))
                writer.finish(summary={"ticks_executed": 3, "alive_agents": 2})

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            lines = self._read_lines(first_path)

        self.assertEqual(lines[0]["type"], "header")
        periods = [line for line in lines if line["type"] == "period_summary"]
        replay = [line for line in lines if line["type"] == "replay_record"]
        footer = lines[-1]
        self.assertEqual(len(periods), 2)
        self.assertEqual(len(replay), 3)
        self.assertEqual(footer["type"], "footer")

        lifetime = footer["lifetime_summary"]
        self.assertEqual(lifetime["decision_record_count"], 4)
        self.assertEqual(lifetime["ecology"]["reproduction_event_record_count"], 1)
        self.assertEqual(lifetime["ecology"]["death_event_record_count"], 1)
        self.assertEqual(lifetime["ecology"]["feeding_event_count"], 1)
        self.assertEqual(
            lifetime["ecology"]["food_source_counts"]["counts"],
            {"fresh_kill": 1},
        )
        self.assertEqual(
            lifetime["ecology"]["death_cause_counts"]["counts"],
            {"hydration_depletion": 1},
        )
        self.assertEqual(lifetime["social"]["mate_requested_count"], 1)
        self.assertEqual(lifetime["social"]["mate_resolved_count"], 1)
        self.assertEqual(lifetime["social"]["attack_attempt_count"], 1)
        self.assertEqual(lifetime["social"]["attack_success_count"], 1)
        self.assertEqual(lifetime["social"]["attack_kill_count"], 1)
        self.assertEqual(lifetime["social"]["signal_emission_count"], 1)
        self.assertEqual(
            lifetime["social"]["signal_token_counts"]["counts"],
            {"2": 1},
        )
        self.assertEqual(
            lifetime["actions"]["requested_action_counts"]["counts"],
            {
                "attack_north": 1,
                "mate": 1,
                "signal_2_profile_0": 1,
                "stay": 1,
            },
        )
        self.assertEqual(
            footer["bounded_output_diagnostics"]["in_memory_full_record_count"],
            0,
        )

        pre_footer_bytes = b"".join(self._canonical_line(line) for line in lines[:-1])
        self.assertEqual(
            footer["source_binding"]["pre_footer_content_sha256"],
            hashlib.sha256(pre_footer_bytes).hexdigest(),
        )

    def test_gzip_output_is_byte_deterministic_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = [
                Path(tmpdir) / "one.jsonl.gz",
                Path(tmpdir) / "two.jsonl.gz",
            ]
            for output_path in outputs:
                writer = self._writer(output_path)
                writer.begin(
                    run_id="gzip-run",
                    config={"seed": 3},
                    contract=self._contract(),
                )
                self.assertFalse(output_path.exists())
                writer.write_record(self._record(tick=0, agent_id=1))
                writer.finish(summary={"ticks_executed": 1})
                self.assertTrue(output_path.exists())

            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            lines = self._read_lines(outputs[0])

        self.assertEqual(lines[0]["compression"], "gzip")
        self.assertEqual(
            lines[-1]["bounded_output_diagnostics"]["input_record_count"], 1
        )

    def test_real_summary_only_world_streams_without_world_record_retention(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "world.jsonl.gz"
            writer = BoundedOpenEcologyTrajectoryWriter(
                output_path,
                summary_interval_ticks=2,
                replay_windows=(ReplayWindow("opening", 0, 0),),
                max_replay_rows=1_000,
                max_replay_bytes=10_000_000,
            )
            world = SimulationWorld(WorldConfig(seed=7, max_ticks=3))

            result = world.run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )
            lines = self._read_lines(output_path)

        self.assertIsNone(result.viewer)
        self.assertEqual(world.trajectory_records, [])
        self.assertGreater(writer.record_count, 0)
        self.assertEqual(writer.diagnostics["in_memory_full_record_count"], 0)
        self.assertEqual(lines[-1]["type"], "footer")
        self.assertEqual(
            lines[-1]["lifetime_summary"]["decision_record_count"],
            writer.record_count,
        )

    def test_replay_row_and_byte_limits_truncate_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            row_path = Path(tmpdir) / "row-limit.jsonl"
            writer = BoundedOpenEcologyTrajectoryWriter(
                row_path,
                summary_interval_ticks=10,
                replay_windows=(ReplayWindow("bounded", 0, 9),),
                max_replay_rows=2,
                max_replay_bytes=1_000_000,
            )
            writer.begin(run_id="row-limit", config={}, contract=self._contract())
            for tick in range(5):
                writer.write_record(self._record(tick=tick, agent_id=tick + 1))
            writer.finish(summary={})
            row_lines = self._read_lines(row_path)

            byte_path = Path(tmpdir) / "byte-limit.jsonl"
            byte_writer = BoundedOpenEcologyTrajectoryWriter(
                byte_path,
                replay_windows=(ReplayWindow("too-small", 0, 0),),
                max_replay_rows=10,
                max_replay_bytes=32,
            )
            byte_writer.begin(
                run_id="byte-limit",
                config={},
                contract=self._contract(),
            )
            byte_writer.write_record(self._record(tick=0, agent_id=1))
            byte_writer.finish(summary={})
            byte_lines = self._read_lines(byte_path)

        row_replays = [line for line in row_lines if line["type"] == "replay_record"]
        row_diagnostics = row_lines[-1]["bounded_output_diagnostics"]
        self.assertEqual(len(row_replays), 2)
        self.assertEqual(row_diagnostics["replay_rows_written"], 2)
        self.assertEqual(row_diagnostics["replay_rows_dropped_by_hard_bounds"], 3)
        self.assertLessEqual(
            row_diagnostics["replay_bytes_written"],
            row_diagnostics["max_replay_bytes"],
        )

        byte_replays = [line for line in byte_lines if line["type"] == "replay_record"]
        byte_diagnostics = byte_lines[-1]["bounded_output_diagnostics"]
        self.assertEqual(byte_replays, [])
        self.assertEqual(byte_diagnostics["replay_rows_written"], 0)
        self.assertEqual(byte_diagnostics["replay_rows_dropped_by_hard_bounds"], 1)

    def test_malformed_record_aborts_and_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "existing.jsonl"
            output_path.write_bytes(b"previous-complete-output\n")
            writer = self._writer(output_path)
            writer.begin(
                run_id="malformed",
                config={},
                contract=self._contract(),
            )
            malformed = self._record(tick=0, agent_id=1)
            del malformed["reward"]

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                writer.write_record(malformed)

            self.assertEqual(output_path.read_bytes(), b"previous-complete-output\n")
            self.assertEqual(writer.diagnostics["status"], "aborted")
            self.assertEqual(
                list(output_path.parent.glob(f".{output_path.name}.*.tmp")),
                [],
            )

    def test_nonmonotonic_ticks_and_nonfinite_numbers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tick_path = Path(tmpdir) / "ticks.jsonl"
            tick_writer = self._writer(tick_path)
            tick_writer.begin(run_id="ticks", config={}, contract=self._contract())
            tick_writer.write_record(self._record(tick=2, agent_id=1))
            with self.assertRaisesRegex(ValueError, "monotonically"):
                tick_writer.write_record(self._record(tick=1, agent_id=2))
            self.assertFalse(tick_path.exists())

            number_path = Path(tmpdir) / "number.jsonl"
            number_writer = self._writer(number_path)
            number_writer.begin(
                run_id="number",
                config={},
                contract=self._contract(),
            )
            invalid = self._record(tick=0, agent_id=1)
            invalid["reward"]["total"] = float("nan")
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                number_writer.write_record(invalid)
            self.assertFalse(number_path.exists())

    def test_inconsistent_public_outcomes_fail_closed(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        missing_signal_token = self._record(
            tick=0,
            agent_id=1,
            action="signal_2_profile_0",
        )
        missing_signal_token["outcome"]["signal"] = {
            "emitted": True,
            "token_id": None,
            "energy_cost": 0.1,
        }
        cases.append(("signal", missing_signal_token, "token_id"))

        impossible_attack_kill = self._record(
            tick=0,
            agent_id=1,
            action="attack_north",
        )
        impossible_attack_kill["outcome"]["attack"] = {
            "attempted": True,
            "success": False,
            "kill": True,
            "damage": 0.0,
        }
        cases.append(("attack", impossible_attack_kill, "without success"))

        inconsistent_death = self._record(tick=0, agent_id=1)
        inconsistent_death["after"]["alive"] = False
        cases.append(("death", inconsistent_death, "inverse"))

        with tempfile.TemporaryDirectory() as tmpdir:
            for label, record, error_pattern in cases:
                with self.subTest(label=label):
                    output_path = Path(tmpdir) / f"{label}.jsonl"
                    writer = self._writer(output_path)
                    writer.begin(
                        run_id=f"invalid-{label}",
                        config={},
                        contract=self._contract(),
                    )
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        writer.write_record(record)
                    self.assertFalse(output_path.exists())
                    self.assertEqual(writer.diagnostics["status"], "aborted")

    def test_abort_removes_partial_output_and_full_records_are_not_retained(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "aborted.jsonl"
            writer = BoundedOpenEcologyTrajectoryWriter(
                output_path,
                summary_interval_ticks=1_000,
                replay_windows=(),
                max_counter_keys=2,
            )
            writer.begin(run_id="abort", config={}, contract=self._contract())
            for index in range(100):
                record = self._record(tick=index, agent_id=index)
                record["action_source"] = f"source-{index}"
                writer.write_record(record)

            diagnostics = writer.diagnostics
            self.assertEqual(diagnostics["input_record_count"], 100)
            self.assertEqual(diagnostics["in_memory_full_record_count"], 0)
            self.assertLessEqual(
                diagnostics["peak_counter_key_count"],
                diagnostics["maximum_counter_key_count"],
            )
            writer.abort()

            self.assertFalse(output_path.exists())
            self.assertEqual(writer.diagnostics["status"], "aborted")
            self.assertEqual(
                list(output_path.parent.glob(f".{output_path.name}.*.tmp")),
                [],
            )

    def test_contract_and_window_configuration_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "max_replay_window_ticks"):
                BoundedOpenEcologyTrajectoryWriter(
                    Path(tmpdir) / "too-long.jsonl",
                    replay_windows=(ReplayWindow("long", 0, 10),),
                    max_replay_window_ticks=10,
                )
            with self.assertRaisesRegex(ValueError, "unique"):
                BoundedOpenEcologyTrajectoryWriter(
                    Path(tmpdir) / "duplicate.jsonl",
                    replay_windows=(
                        ReplayWindow("same", 0, 1),
                        ReplayWindow("same", 2, 3),
                    ),
                )

            invalid_contract = self._contract()
            invalid_contract["record_fields"] = ["tick"]
            writer = self._writer(Path(tmpdir) / "contract.jsonl")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                writer.begin(run_id="contract", config={}, contract=invalid_contract)
            self.assertEqual(
                list(Path(tmpdir).glob(".contract.jsonl.*.tmp")),
                [],
            )

    def _writer(self, path: Path) -> BoundedOpenEcologyTrajectoryWriter:
        return BoundedOpenEcologyTrajectoryWriter(
            path,
            summary_interval_ticks=2,
            replay_windows=(ReplayWindow("opening", 0, 1),),
            max_replay_rows=10,
            max_replay_bytes=1_000_000,
        )

    def _contract(self) -> dict[str, object]:
        return trajectory_contract(
            SignalConfig(
                communication_token_count=4,
                communication_profiles_per_token=2,
                communication_signal_emission_enabled=True,
            )
        )

    def _meaningful_records(self) -> list[dict[str, object]]:
        attack = self._record(
            tick=0,
            agent_id=1,
            action="attack_north",
            reward=0.2,
        )
        attack["outcome"]["attack"] = {
            "attempted": True,
            "success": True,
            "kill": True,
            "target_id": 9,
            "damage": 0.75,
        }
        attack["outcome"]["feeding"] = {
            "ate": True,
            "food_source": "fresh_kill",
        }
        attack["outcome"]["resource_gain"] = 0.4

        signal = self._record(
            tick=0,
            agent_id=2,
            action="signal_2_profile_0",
            reward=-0.05,
        )
        signal["outcome"]["signal"] = {
            "emitted": True,
            "token_id": 2,
            "profile_index": 0,
            "energy_cost": 0.05,
        }

        mate = self._record(
            tick=1,
            agent_id=3,
            action="mate",
            reward=1.0,
        )
        mate["outcome"]["reproduced"] = True
        mate["outcome"]["reproduction_ready_after"] = True

        death = self._record(tick=2, agent_id=4, action="stay", reward=-1.0)
        death["after"]["alive"] = False
        death["outcome"]["died"] = True
        death["outcome"]["passive"] = {
            "killed": False,
            "death_cause": "hydration_depletion",
            "attack_damage_taken": 0.0,
        }
        return [attack, signal, mate, death]

    def _record(
        self,
        *,
        tick: int,
        agent_id: int,
        action: str = "stay",
        reward: float = 0.02,
    ) -> dict[str, object]:
        return {
            "tick": tick,
            "agent_id": agent_id,
            "lineage_id": 1,
            "runtime_species_id": agent_id % 2,
            "runtime_ecotype_id": agent_id % 3,
            "action_mask": {action: True},
            "resolution_action_mask": {action: True},
            "requested_action": action,
            "resolved_action": action,
            "action_valid": True,
            "resolution_action_valid": True,
            "action_source": "test_public_policy",
            "moved": action.startswith("move_"),
            "before": {
                "alive": True,
                "energy_ratio": 0.8,
                "hydration_ratio": 0.7,
                "health_ratio": 0.9,
            },
            "after": {
                "alive": True,
                "energy_ratio": 0.79,
                "hydration_ratio": 0.69,
                "health_ratio": 0.9,
            },
            "outcome": {
                "requested_action": action,
                "resolved_action": action,
                "attack": {"attempted": False},
                "feeding": {"ate": False},
                "drinking": {"drank": False},
                "signal": {"emitted": False, "energy_cost": 0.0},
                "passive": {
                    "killed": False,
                    "death_cause": None,
                    "attack_damage_taken": 0.0,
                },
                "resource_gain": 0.0,
                "reproduced": False,
                "died": False,
                "reproduction_ready_after": False,
            },
            "reward": {"total": reward},
        }

    def _read_lines(self, path: Path) -> list[dict[str, object]]:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        if path.suffix == ".gz":
            with opener(path, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle]
        with opener(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def _canonical_line(self, payload: dict[str, object]) -> bytes:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
