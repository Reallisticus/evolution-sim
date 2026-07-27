from __future__ import annotations

import gc
import gzip
import json
import multiprocessing
import os
import tempfile
import tracemalloc
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evolution_sim.io.open_ecology_rotating_writer import (
    OPEN_ECOLOGY_EVIDENCE_CHAIN_GENESIS,
    OPEN_ECOLOGY_EVIDENCE_LOCK_NAME,
    OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME,
    OPEN_ECOLOGY_EVIDENCE_TRANSACTION_NAME,
    BirthEvidence,
    CongestionEvidence,
    DeathEvidence,
    DyadicInteractionEvidence,
    InterventionEvidence,
    LineageEvidence,
    OpenEcologyEvidenceError,
    RotatingEvidenceConfig,
    RotatingOpenEcologyEvidenceWriter,
    SignalContributorEvidence,
    load_open_ecology_evidence_manifest,
    validate_open_ecology_event_record,
    validate_open_ecology_evidence_continuation_state,
)


def _hold_resumed_writer(
    output: str,
    config_payload: dict[str, int],
    source_contract: dict[str, object],
    continuation_state: dict[str, object],
    ready_connection: object,
    release_connection: object,
) -> None:
    writer: RotatingOpenEcologyEvidenceWriter | None = None
    try:
        writer = RotatingOpenEcologyEvidenceWriter(
            output,
            run_id="open-ecology-run",
            source_contract=source_contract,
            config=RotatingEvidenceConfig.from_dict(config_payload),
            continuation_state=continuation_state,
        )
        ready_connection.send(("ready", None))  # type: ignore[attr-defined]
        release_connection.recv()  # type: ignore[attr-defined]
    except Exception as error:
        ready_connection.send(  # type: ignore[attr-defined]
            ("error", f"{type(error).__name__}: {error}")
        )
    finally:
        if writer is not None:
            writer.abort()
        ready_connection.close()  # type: ignore[attr-defined]
        release_connection.close()  # type: ignore[attr-defined]


class RotatingOpenEcologyEvidenceWriterTests(unittest.TestCase):
    def test_all_typed_events_rotate_with_exact_bounds_and_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "evidence"
            config = self._config(
                max_rows_per_shard=3,
                max_ticks_per_shard=2,
            )
            writer = self._writer(output, config=config)
            for event in self._all_events():
                writer.append(event)
            manifest = writer.finish()
            loaded = load_open_ecology_evidence_manifest(output)
            event_types = self._event_types(output, loaded)

        self.assertEqual(manifest, loaded)
        self.assertEqual(loaded["status"], "complete")
        self.assertEqual(loaded["total_event_count"], 7)
        self.assertEqual(
            event_types,
            [
                "birth",
                "lineage",
                "dyadic_interaction",
                "signal_contributor",
                "congestion",
                "intervention",
                "death",
            ],
        )
        previous = OPEN_ECOLOGY_EVIDENCE_CHAIN_GENESIS
        for index, entry in enumerate(loaded["completed_shards"]):
            self.assertEqual(entry["shard_index"], index)
            self.assertEqual(entry["previous_shard_sha256"], previous)
            self.assertLessEqual(entry["event_count"], config.max_rows_per_shard)
            self.assertLessEqual(
                entry["last_tick"] - entry["first_tick"] + 1,
                config.max_ticks_per_shard,
            )
            self.assertLessEqual(
                entry["uncompressed_bytes"],
                config.max_uncompressed_bytes_per_shard,
            )
            self.assertLessEqual(
                entry["compressed_bytes"],
                config.max_compressed_bytes_per_shard,
            )
            previous = entry["file_sha256"]
        self.assertEqual(loaded["chain_head_sha256"], previous)
        self.assertEqual(writer.diagnostics["in_memory_event_count"], 0)

    def test_gzip_shards_and_manifest_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            roots = [Path(tmpdir) / "first", Path(tmpdir) / "second"]
            for root in roots:
                writer = self._writer(
                    root,
                    config=self._config(
                        max_rows_per_shard=2,
                        max_ticks_per_shard=8,
                    ),
                )
                for event in self._all_events():
                    writer.append(event)
                writer.finish()

            first_files = self._artifact_bytes(roots[0])
            second_files = self._artifact_bytes(roots[1])

        self.assertEqual(first_files, second_files)
        self.assertEqual(
            sorted(first_files),
            [
                "manifest.json",
                "shard-00000000.jsonl.gz",
                "shard-00000001.jsonl.gz",
                "shard-00000002.jsonl.gz",
                "shard-00000003.jsonl.gz",
            ],
        )

    def test_resume_at_completed_shard_is_byte_and_semantically_equivalent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            control_root = Path(tmpdir) / "control"
            resumed_root = Path(tmpdir) / "resumed"
            config = self._config(
                max_rows_per_shard=2,
                max_ticks_per_shard=16,
            )
            events = self._all_events()

            control = self._writer(control_root, config=config)
            for event in events:
                control.append(event)
            control.finish()

            first_process = self._writer(resumed_root, config=config)
            for event in events[:2]:
                first_process.append(event)
            self.assertEqual(
                first_process.diagnostics["active_shard_event_count"],
                0,
            )
            state = first_process.checkpoint()
            serialized_state = json.loads(self._canonical(state))
            validate_open_ecology_evidence_continuation_state(serialized_state)

            second_process = RotatingOpenEcologyEvidenceWriter(
                resumed_root,
                run_id="open-ecology-run",
                source_contract=self._source_contract(),
                config=config,
                continuation_state=serialized_state,
            )
            for event in events[2:]:
                second_process.append(event)
            second_process.finish()

            control_bytes = self._artifact_bytes(control_root)
            resumed_bytes = self._artifact_bytes(resumed_root)
            control_manifest = load_open_ecology_evidence_manifest(control_root)
            resumed_manifest = load_open_ecology_evidence_manifest(resumed_root)

        self.assertEqual(control_bytes, resumed_bytes)
        self.assertEqual(control_manifest, resumed_manifest)

    def test_exclusive_stream_lock_rejects_dual_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "locked"
            config = self._config(max_rows_per_shard=1)
            initial = self._writer(output, config=config)
            initial.append(self._all_events()[0])
            state = initial.checkpoint()

            first_resume = RotatingOpenEcologyEvidenceWriter(
                output,
                run_id="open-ecology-run",
                source_contract=self._source_contract(),
                config=config,
                continuation_state=state,
            )
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "active writer",
            ):
                RotatingOpenEcologyEvidenceWriter(
                    output,
                    run_id="open-ecology-run",
                    source_contract=self._source_contract(),
                    config=config,
                    continuation_state=state,
                )
            first_resume.abort()

    def test_exclusive_stream_lock_rejects_a_second_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "multiprocess-locked"
            config = self._config(max_rows_per_shard=1)
            initial = self._writer(output, config=config)
            initial.append(self._all_events()[0])
            state = initial.checkpoint()

            context = multiprocessing.get_context("fork")
            child_ready, parent_ready = context.Pipe(duplex=False)
            child_release, parent_release = context.Pipe(duplex=False)
            process = context.Process(
                target=_hold_resumed_writer,
                args=(
                    str(output),
                    config.to_dict(),
                    self._source_contract(),
                    state,
                    parent_ready,
                    child_release,
                ),
            )
            process.start()
            parent_ready.close()
            child_release.close()
            try:
                status, detail = child_ready.recv()
                self.assertEqual((status, detail), ("ready", None))
                with self.assertRaisesRegex(
                    OpenEcologyEvidenceError,
                    "active writer",
                ):
                    RotatingOpenEcologyEvidenceWriter(
                        output,
                        run_id="open-ecology-run",
                        source_contract=self._source_contract(),
                        config=config,
                        continuation_state=state,
                    )
            finally:
                parent_release.send("release")
                parent_release.close()
                child_ready.close()
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)

    def test_completed_shards_publish_atomically_and_checkpoint_is_sealed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "atomic"
            writer = self._writer(
                output,
                config=self._config(max_rows_per_shard=10),
            )
            writer.append(self._all_events()[0])

            self.assertFalse((output / "shard-00000000.jsonl.gz").exists())
            self.assertEqual(
                list(output.glob("shard-*.jsonl.gz")),
                [],
            )
            state = writer.checkpoint()

            self.assertTrue((output / "shard-00000000.jsonl.gz").exists())
            self.assertEqual(list(output.glob(".*.tmp")), [])
            loaded = load_open_ecology_evidence_manifest(output)

        self.assertEqual(loaded["status"], "open")
        self.assertEqual(loaded["completed_shard_count"], 1)
        self.assertEqual(state["manifest_sha256"], loaded["manifest_sha256"])
        with self.assertRaisesRegex(RuntimeError, "checkpointed"):
            writer.append(replace(self._all_events()[1], event_index=1))

    def test_failed_shard_publication_preserves_the_last_complete_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "replace-failure"
            writer = self._writer(output)
            initial_manifest = (
                output / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME
            ).read_bytes()
            writer.append(self._all_events()[0])

            with patch(
                "evolution_sim.io.open_ecology_rotating_writer.os.replace",
                side_effect=OSError("simulated shard replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    writer.checkpoint()

            self.assertEqual(
                (output / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME).read_bytes(),
                initial_manifest,
            )
            self.assertEqual(list(output.glob("shard-*.jsonl.gz")), [])
            self.assertEqual(list(output.glob(".*.tmp")), [])
            loaded = load_open_ecology_evidence_manifest(output)

        self.assertEqual(loaded["completed_shard_count"], 0)
        self.assertEqual(loaded["total_event_count"], 0)

    def test_manifest_publish_failure_rolls_back_the_published_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest-failure"
            writer = self._writer(output)
            initial_manifest = (
                output / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME
            ).read_bytes()
            writer.append(self._all_events()[0])
            real_replace = os.replace
            replacement_count = 0

            def fail_manifest_replace(source: object, destination: object) -> None:
                nonlocal replacement_count
                replacement_count += 1
                if replacement_count == 3:
                    raise OSError("simulated manifest replace failure")
                real_replace(source, destination)

            with patch(
                "evolution_sim.io.open_ecology_rotating_writer.os.replace",
                side_effect=fail_manifest_replace,
            ):
                with self.assertRaisesRegex(OSError, "simulated manifest"):
                    writer.checkpoint()

            self.assertEqual(
                (output / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME).read_bytes(),
                initial_manifest,
            )
            self.assertEqual(list(output.glob("shard-*.jsonl.gz")), [])
            self.assertFalse((output / OPEN_ECOLOGY_EVIDENCE_TRANSACTION_NAME).exists())
            self.assertEqual(list(output.glob(".*.tmp")), [])
            loaded = load_open_ecology_evidence_manifest(output)

        self.assertEqual(loaded["completed_shard_count"], 0)
        self.assertEqual(loaded["total_event_count"], 0)

    def test_pending_transaction_recovers_to_external_checkpoint_after_crash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "crash-recovery"
            config = self._config()
            initial = self._writer(output, config=config)
            state = initial.checkpoint()
            writer = RotatingOpenEcologyEvidenceWriter(
                output,
                run_id="open-ecology-run",
                source_contract=self._source_contract(),
                config=config,
                continuation_state=state,
            )
            writer.append(self._death(0))
            real_replace = os.replace
            replacement_count = 0

            def crash_during_manifest_publish(
                source: object,
                destination: object,
            ) -> None:
                nonlocal replacement_count
                replacement_count += 1
                if replacement_count == 3:
                    raise SystemExit("simulated hard crash")
                real_replace(source, destination)

            with patch(
                "evolution_sim.io.open_ecology_rotating_writer.os.replace",
                side_effect=crash_during_manifest_publish,
            ):
                with self.assertRaisesRegex(SystemExit, "hard crash"):
                    writer.checkpoint()
            writer.abort()

            self.assertTrue((output / OPEN_ECOLOGY_EVIDENCE_TRANSACTION_NAME).exists())
            self.assertTrue((output / "shard-00000000.jsonl.gz").exists())
            recovered = RotatingOpenEcologyEvidenceWriter(
                output,
                run_id="open-ecology-run",
                source_contract=self._source_contract(),
                config=config,
                continuation_state=state,
            )
            recovered.append(self._death(0))
            manifest = recovered.finish()

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_shard_count"], 1)
        self.assertEqual(manifest["total_event_count"], 1)

    def test_tamper_unknown_nonfinite_and_out_of_order_inputs_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(OpenEcologyEvidenceError, "finite"):
            SignalContributorEvidence(
                tick=0,
                event_index=0,
                event_id="bad-signal:0",
                emitter_agent_id=1,
                receiver_agent_id=2,
                token_id=0,
                contribution=float("nan"),
            )
        with self.assertRaisesRegex(OpenEcologyEvidenceError, "suffix"):
            DeathEvidence(
                tick=0,
                event_index=1,
                event_id="death:0",
                agent_id=1,
                cause="energy_depletion",
            )

        unknown_event = self._all_events()[0].to_record()
        unknown_event["event_type"] = "unsupported"
        with self.assertRaisesRegex(OpenEcologyEvidenceError, "unknown"):
            validate_open_ecology_event_record(unknown_event)

        extra_field = self._all_events()[0].to_record()
        extra_field["payload"]["inferred_society"] = True
        with self.assertRaisesRegex(OpenEcologyEvidenceError, "keys mismatch"):
            validate_open_ecology_event_record(extra_field)

        with tempfile.TemporaryDirectory() as tmpdir:
            wrong_index = self._writer(Path(tmpdir) / "index")
            with self.assertRaisesRegex(OpenEcologyEvidenceError, "contiguous"):
                wrong_index.append(
                    replace(
                        self._all_events()[0],
                        event_index=1,
                        event_id="birth:1",
                    )
                )
            self.assertEqual(wrong_index.diagnostics["status"], "aborted")

            wrong_tick = self._writer(Path(tmpdir) / "tick")
            wrong_tick.append(replace(self._all_events()[0], tick=2, event_index=0))
            with self.assertRaisesRegex(OpenEcologyEvidenceError, "monotonically"):
                wrong_tick.append(replace(self._all_events()[1], tick=1, event_index=1))
            self.assertEqual(wrong_tick.diagnostics["status"], "aborted")

            unknown_class = self._writer(Path(tmpdir) / "unknown")
            with self.assertRaisesRegex(TypeError, "typed event"):
                unknown_class.append({"event_type": "birth"})  # type: ignore[arg-type]

    def test_shard_and_continuation_tampering_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "tampered"
            writer = self._writer(
                output,
                config=self._config(max_rows_per_shard=2),
            )
            for event in self._all_events()[:2]:
                writer.append(event)
            state = writer.checkpoint()
            shard_path = output / "shard-00000000.jsonl.gz"
            original = shard_path.read_bytes()
            shard_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(OpenEcologyEvidenceError, "SHA256"):
                load_open_ecology_evidence_manifest(output)

            tampered_state = dict(state)
            tampered_state["next_event_index"] = 99
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "total_event_count|SHA256",
            ):
                validate_open_ecology_evidence_continuation_state(tampered_state)

            orphan_output = Path(tmpdir) / "orphan"
            orphan_writer = self._writer(
                orphan_output,
                config=self._config(max_rows_per_shard=1),
            )
            orphan_writer.append(self._death(0))
            (orphan_output / ".stale-shard.tmp").write_bytes(b"partial")
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "directory entries",
            ):
                load_open_ecology_evidence_manifest(orphan_output)

    def test_lock_control_file_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "lock-tamper"
            writer = self._writer(output)
            manifest = writer.finish()
            lock_path = output / OPEN_ECOLOGY_EVIDENCE_LOCK_NAME
            exact_lock = lock_path.read_bytes()

            lock_path.write_bytes(exact_lock + b"x")
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "byte ceiling",
            ):
                load_open_ecology_evidence_manifest(output)

            lock_path.write_bytes(b"wrong\n")
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "control contract",
            ):
                load_open_ecology_evidence_manifest(output)
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "control contract",
            ):
                self._writer(output)
            self.assertEqual(lock_path.read_bytes(), b"wrong\n")

            lock_path.unlink()
            target = output.parent / "lock-target"
            target.write_bytes(exact_lock)
            lock_path.symlink_to(target)
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "safely open",
            ):
                load_open_ecology_evidence_manifest(output)

            lock_path.unlink()
            os.link(target, lock_path)
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "not regular",
            ):
                load_open_ecology_evidence_manifest(output)
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "one regular file",
            ):
                self._writer(output)
            self.assertEqual(target.read_bytes(), exact_lock)

            lock_path.unlink()
            lock_path.write_bytes(exact_lock)
            loaded = load_open_ecology_evidence_manifest(
                output,
                expected_manifest_sha256=manifest["manifest_sha256"],
                expected_status="complete",
            )

        self.assertEqual(loaded, manifest)

    def test_external_manifest_pin_detects_a_self_consistent_truncated_prefix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest-authority"
            config = self._config(max_rows_per_shard=1)
            first = self._writer(output, config=config)
            first.append(self._death(0))
            state = first.checkpoint()
            prefix_manifest = (
                output / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME
            ).read_bytes()

            resumed = RotatingOpenEcologyEvidenceWriter(
                output,
                run_id="open-ecology-run",
                source_contract=self._source_contract(),
                config=config,
                continuation_state=state,
            )
            resumed.append(self._death(1))
            complete_manifest = resumed.finish()
            complete_digest = str(complete_manifest["manifest_sha256"])

            (output / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME).write_bytes(prefix_manifest)
            (output / "shard-00000001.jsonl.gz").unlink()
            self.assertEqual(
                load_open_ecology_evidence_manifest(output)["status"],
                "open",
            )
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "externally expected",
            ):
                load_open_ecology_evidence_manifest(
                    output,
                    expected_manifest_sha256=complete_digest,
                )

    def test_hard_event_and_shard_count_limits_reject_without_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_limited = self._writer(
                Path(tmpdir) / "event-limit",
                config=self._config(max_event_bytes=128),
            )
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "max_event_bytes",
            ):
                event_limited.append(self._all_events()[0])

            shard_limited = self._writer(
                Path(tmpdir) / "shard-limit",
                config=self._config(
                    max_rows_per_shard=1,
                    max_shards=1,
                ),
            )
            shard_limited.append(self._all_events()[0])
            with self.assertRaisesRegex(OpenEcologyEvidenceError, "max_shards"):
                shard_limited.append(self._all_events()[1])
            manifest = load_open_ecology_evidence_manifest(Path(tmpdir) / "shard-limit")

            byte_total_limited = self._writer(
                Path(tmpdir) / "total-byte-limit",
                config=self._config(
                    max_rows_per_shard=1,
                    max_compressed_bytes_per_shard=4_096,
                    max_total_compressed_bytes=700,
                ),
            )
            byte_total_limited.append(self._death(0))
            with self.assertRaisesRegex(
                OpenEcologyEvidenceError,
                "max_total_compressed_bytes",
            ):
                byte_total_limited.append(self._death(1))
            total_limited_manifest = load_open_ecology_evidence_manifest(
                Path(tmpdir) / "total-byte-limit"
            )

        self.assertEqual(manifest["completed_shard_count"], 1)
        self.assertEqual(manifest["total_event_count"], 1)
        self.assertEqual(total_limited_manifest["completed_shard_count"], 1)
        self.assertLessEqual(
            total_limited_manifest["total_compressed_bytes"],
            700,
        )

    def test_uncompressed_byte_ceiling_rotates_before_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "byte-rotation"
            config = self._config(
                max_ticks_per_shard=100,
                max_rows_per_shard=10,
                max_uncompressed_bytes_per_shard=1_400,
                max_compressed_bytes_per_shard=4_096,
                max_event_bytes=512,
            )
            writer = self._writer(output, config=config)
            for index in range(3):
                writer.append(self._death(index))
            manifest = writer.finish()

        self.assertEqual(
            [entry["event_count"] for entry in manifest["completed_shards"]],
            [2, 1],
        )
        self.assertTrue(
            all(
                entry["uncompressed_bytes"] <= config.max_uncompressed_bytes_per_shard
                for entry in manifest["completed_shards"]
            )
        )

    def test_event_ingestion_does_not_retain_rows_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(
                max_rows_per_shard=10_000,
                max_ticks_per_shard=10_000,
                max_uncompressed_bytes_per_shard=16 * 1024 * 1024,
                max_compressed_bytes_per_shard=18 * 1024 * 1024,
            )
            writer = self._writer(Path(tmpdir) / "bounded", config=config)
            tracemalloc.start()
            try:
                for index in range(500):
                    writer.append(self._death(index))
                gc.collect()
                baseline_current, _ = tracemalloc.get_traced_memory()
                for index in range(500, 5_000):
                    writer.append(self._death(index))
                gc.collect()
                final_current, _ = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            diagnostics = writer.diagnostics
            manifest = writer.finish()

        self.assertEqual(diagnostics["in_memory_event_count"], 0)
        self.assertEqual(diagnostics["active_shard_event_count"], 5_000)
        self.assertLess(final_current - baseline_current, 512 * 1024)
        self.assertEqual(manifest["total_event_count"], 5_000)
        self.assertEqual(manifest["completed_shard_count"], 1)

    def _writer(
        self,
        output: Path,
        *,
        config: RotatingEvidenceConfig | None = None,
    ) -> RotatingOpenEcologyEvidenceWriter:
        return RotatingOpenEcologyEvidenceWriter(
            output,
            run_id="open-ecology-run",
            source_contract=self._source_contract(),
            config=config or self._config(),
        )

    def _config(self, **overrides: int) -> RotatingEvidenceConfig:
        values = {
            "max_ticks_per_shard": 4,
            "max_rows_per_shard": 16,
            "max_uncompressed_bytes_per_shard": 64 * 1024,
            "max_compressed_bytes_per_shard": 72 * 1024,
            "max_event_bytes": 4 * 1024,
            "max_shards": 64,
            "max_manifest_bytes": 256 * 1024,
            "max_source_contract_bytes": 16 * 1024,
            "max_total_compressed_bytes": 8 * 1024 * 1024 * 1024,
        }
        values.update(overrides)
        return RotatingEvidenceConfig(**values)

    def _source_contract(self) -> dict[str, object]:
        return {
            "campaign_contract_sha256": "a" * 64,
            "event_producer": "explicit-public-runtime-events-v1",
            "source_git_sha": "b" * 40,
        }

    def _all_events(self) -> list[object]:
        return [
            BirthEvidence(
                tick=0,
                event_index=0,
                event_id="birth:0",
                child_agent_id=10,
                parent_agent_ids=(1, 2),
                lineage_id="lineage-10",
                genome_sha256="1" * 64,
            ),
            LineageEvidence(
                tick=0,
                event_index=1,
                event_id="lineage:1",
                agent_id=10,
                lineage_id="lineage-10",
                parent_lineage_ids=("lineage-1", "lineage-2"),
                transition_kind="birth",
            ),
            DyadicInteractionEvidence(
                tick=1,
                event_index=2,
                event_id="dyad:2",
                actor_agent_id=10,
                target_agent_id=11,
                interaction_kind="resource_transfer",
                outcome="completed",
                magnitude=0.25,
            ),
            SignalContributorEvidence(
                tick=1,
                event_index=3,
                event_id="signal:3",
                emitter_agent_id=10,
                receiver_agent_id=11,
                token_id=2,
                contribution=0.75,
            ),
            CongestionEvidence(
                tick=2,
                event_index=4,
                event_id="congestion:4",
                target_x=4,
                target_y=5,
                contender_agent_ids=(10, 11),
                winner_agent_id=10,
                outcome="one_moved",
            ),
            InterventionEvidence(
                tick=3,
                event_index=5,
                event_id="intervention:5",
                intervention_id="genome-zero",
                intervention_kind="genome_ablation",
                branch_id="branch-a",
                subject_agent_id=10,
                assigned_action=None,
                value_sha256="2" * 64,
            ),
            DeathEvidence(
                tick=3,
                event_index=6,
                event_id="death:6",
                agent_id=11,
                cause="energy_depletion",
                source_agent_id=None,
            ),
        ]

    def _death(self, index: int) -> DeathEvidence:
        return DeathEvidence(
            tick=index,
            event_index=index,
            event_id=f"death:{index}",
            agent_id=index,
            cause="energy_depletion",
        )

    def _artifact_bytes(self, root: Path) -> dict[str, bytes]:
        return {
            path.name: path.read_bytes()
            for path in sorted(root.iterdir())
            if path.name == OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME
            or path.name.endswith(".jsonl.gz")
        }

    def _event_types(
        self,
        root: Path,
        manifest: dict[str, object],
    ) -> list[str]:
        result: list[str] = []
        for entry in manifest["completed_shards"]:
            with gzip.open(root / entry["file_name"], "rb") as handle:
                lines = [json.loads(line) for line in handle]
            result.extend(
                line["event"]["event_type"] for line in lines if line["type"] == "event"
            )
        return result

    def _canonical(self, value: object) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


if __name__ == "__main__":
    unittest.main()
