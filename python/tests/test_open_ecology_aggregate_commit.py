from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evolution_sim.io.open_ecology_aggregate_commit import (
    OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME,
    OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME,
    OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY,
    OPEN_ECOLOGY_AGGREGATE_LOCK_NAME,
    OpenEcologyAggregateCommitError,
    OpenEcologyAggregateIdentityPins,
    OpenEcologyAggregateResumePins,
    inspect_current_open_ecology_aggregate_generation,
    load_current_open_ecology_aggregate_generation,
    load_or_recover_open_ecology_genesis_generation,
    load_open_ecology_aggregate_generation,
    publish_open_ecology_aggregate_generation,
    select_open_ecology_aggregate_attempt_generation,
)
from evolution_sim.io.open_ecology_checkpoint import (
    VersionedCheckpointState,
    write_open_ecology_checkpoint,
)
from evolution_sim.io.open_ecology_rotating_writer import (
    OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA,
    BirthEvidence,
    RotatingEvidenceConfig,
    RotatingOpenEcologyEvidenceWriter,
    load_open_ecology_evidence_manifest,
)

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_RUNTIME_BINDING_SCHEMA = "open_ecology_runtime_checkpoint_binding_v2"
_RUNTIME_EVIDENCE_ADAPTER_SCHEMA = (
    "open_ecology_evidence_writer_continuation_adapter_v1"
)


class OpenEcologyAggregateCommitTests(unittest.TestCase):
    def test_two_generations_preserve_old_evidence_prefix_and_advance_current(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            first_commit_bytes = self._commit_path(fixture.root, 0).read_bytes()
            first_snapshot_bytes = self._evidence_snapshot_path(
                fixture.root,
                0,
            ).read_bytes()

            second = self._publish_second(fixture, first=first)
            current = load_current_open_ecology_aggregate_generation(
                fixture.root,
                evidence_directory=fixture.evidence,
                pins=self._resume_pins(second),
            )
            historical = load_open_ecology_aggregate_generation(
                fixture.root,
                aggregate_generation_index=0,
                evidence_directory=fixture.evidence,
                pins=self._resume_pins(first),
            )

            self.assertEqual(
                current["commit"]["aggregate_generation_index"],
                1,
            )
            self.assertEqual(
                historical["commit"]["aggregate_generation_index"],
                0,
            )
            self.assertEqual(
                first_commit_bytes,
                self._commit_path(fixture.root, 0).read_bytes(),
            )
            self.assertEqual(
                first_snapshot_bytes,
                self._evidence_snapshot_path(fixture.root, 0).read_bytes(),
            )
            self.assertEqual(
                sorted(
                    path.name
                    for path in (
                        fixture.root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
                    ).iterdir()
                ),
                [
                    "generation-0000000000000000",
                    "generation-0000000000000001",
                ],
            )

    def test_pointer_crash_recovers_complete_successor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            second_inputs = self._second_inputs(fixture)
            real_replace = os.replace

            def fail_current(source: object, destination: object) -> None:
                if Path(destination).name == OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME:
                    raise OSError("simulated CURRENT publication crash")
                real_replace(source, destination)

            with patch(
                "evolution_sim.io.open_ecology_aggregate_commit.os.replace",
                side_effect=fail_current,
            ):
                with self.assertRaisesRegex(OSError, "simulated CURRENT"):
                    self._publish(
                        fixture,
                        aggregate_generation_index=1,
                        checkpoint=second_inputs.checkpoint,
                        identity=second_inputs.identity,
                        manifest=second_inputs.manifest,
                        previous=first,
                    )

            current_before_recovery = json.loads(
                (fixture.root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                current_before_recovery["aggregate_generation_index"],
                0,
            )
            orphan_commit = self._load_json(self._commit_path(fixture.root, 1))
            expected_second = {
                "checkpoint": second_inputs.checkpoint,
                "commit": orphan_commit,
                "evidence_manifest": second_inputs.manifest,
            }
            recovered = load_current_open_ecology_aggregate_generation(
                fixture.root,
                evidence_directory=fixture.evidence,
                pins=self._resume_pins(expected_second),
            )
            self.assertEqual(
                recovered["commit"]["aggregate_generation_index"],
                1,
            )

    def test_attempt_selection_delegates_unpublished_genesis_to_genesis_api(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first_inputs = self._first_inputs(fixture)
            real_replace = os.replace

            def fail_current(source: object, destination: object) -> None:
                if Path(destination).name == OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME:
                    raise OSError("simulated genesis CURRENT publication crash")
                real_replace(source, destination)

            with patch(
                "evolution_sim.io.open_ecology_aggregate_commit.os.replace",
                side_effect=fail_current,
            ):
                with self.assertRaisesRegex(OSError, "genesis CURRENT"):
                    self._publish(
                        fixture,
                        aggregate_generation_index=0,
                        checkpoint=first_inputs.checkpoint,
                        identity=first_inputs.identity,
                        manifest=first_inputs.manifest,
                        previous=None,
                    )

            orphan = {
                "checkpoint": first_inputs.checkpoint,
                "commit": self._load_json(self._commit_path(fixture.root, 0)),
                "evidence_manifest": first_inputs.manifest,
            }
            pins = self._resume_pins(orphan)
            with self.assertRaisesRegex(
                OpenEcologyAggregateCommitError,
                "recover genesis through the genesis API",
            ):
                select_open_ecology_aggregate_attempt_generation(
                    fixture.root,
                    evidence_directory=fixture.evidence,
                    pins=pins,
                    expected_previous_commit_sha256=None,
                )

            recovered = load_or_recover_open_ecology_genesis_generation(
                fixture.root,
                evidence_directory=fixture.evidence,
                identity=first_inputs.identity,
            )
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered["commit"], orphan["commit"])
            self.assertTrue(
                (fixture.root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME).is_file()
            )

    def test_pinned_current_preserves_complete_unselected_successor_for_retry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            second_inputs = self._second_inputs(fixture)
            real_replace = os.replace

            def fail_current(source: object, destination: object) -> None:
                if Path(destination).name == OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME:
                    raise OSError("simulated CURRENT publication crash")
                real_replace(source, destination)

            with patch(
                "evolution_sim.io.open_ecology_aggregate_commit.os.replace",
                side_effect=fail_current,
            ):
                with self.assertRaisesRegex(OSError, "simulated CURRENT"):
                    self._publish(
                        fixture,
                        aggregate_generation_index=1,
                        checkpoint=second_inputs.checkpoint,
                        identity=second_inputs.identity,
                        manifest=second_inputs.manifest,
                        previous=first,
                    )

            orphan_commit_sha256 = self._load_json(self._commit_path(fixture.root, 1))[
                "commit_sha256"
            ]
            inspected = inspect_current_open_ecology_aggregate_generation(
                fixture.root,
                evidence_directory=fixture.evidence,
                pins=self._resume_pins(first),
            )
            self.assertEqual(
                inspected["commit"]["aggregate_generation_index"],
                0,
            )
            self.assertTrue(self._commit_path(fixture.root, 1).is_file())
            preservation = fixture.root.parent / "attempts"
            retained = load_current_open_ecology_aggregate_generation(
                fixture.root,
                evidence_directory=fixture.evidence,
                pins=self._resume_pins(first),
                uncommitted_successor_preservation_directory=preservation,
            )
            self.assertEqual(
                retained["commit"]["aggregate_generation_index"],
                0,
            )
            self.assertFalse(
                (
                    fixture.root
                    / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
                    / "generation-0000000000000001"
                ).exists()
            )
            preserved = list(preservation.iterdir())
            self.assertEqual(len(preserved), 1)
            self.assertEqual(
                self._load_json(preserved[0] / OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME)[
                    "commit_sha256"
                ],
                orphan_commit_sha256,
            )

            retried = self._publish(
                fixture,
                aggregate_generation_index=1,
                checkpoint=second_inputs.checkpoint,
                identity=second_inputs.identity,
                manifest=second_inputs.manifest,
                previous=first,
            )
            self.assertEqual(
                retried["commit"]["aggregate_generation_index"],
                1,
            )

    def test_generation_directory_crash_retains_previous_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            second_inputs = self._second_inputs(fixture)

            with patch(
                "evolution_sim.io.open_ecology_aggregate_commit.os.rename",
                side_effect=OSError("simulated directory publication crash"),
            ):
                with self.assertRaisesRegex(OSError, "simulated directory"):
                    self._publish(
                        fixture,
                        aggregate_generation_index=1,
                        checkpoint=second_inputs.checkpoint,
                        identity=second_inputs.identity,
                        manifest=second_inputs.manifest,
                        previous=first,
                    )

            retained = load_current_open_ecology_aggregate_generation(
                fixture.root,
                evidence_directory=fixture.evidence,
                pins=self._resume_pins(first),
            )
            self.assertEqual(
                retained["commit"]["aggregate_generation_index"],
                0,
            )
            self.assertFalse(
                (
                    fixture.root
                    / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
                    / "generation-0000000000000001"
                ).exists()
            )

    def test_failed_generation_cleanup_preserves_replaced_stage_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            staged_path: Path | None = None
            validated_backup = Path(tmpdir) / "validated-stage-backup"
            attacker_bytes = b"replacement-stage-must-not-be-removed\n"

            def replace_stage_then_fail(path: Path, encoded: bytes) -> None:
                nonlocal staged_path
                del encoded
                staged_path = path.parent
                staged_path.rename(validated_backup)
                staged_path.mkdir()
                (staged_path / "attacker-sentinel").write_bytes(attacker_bytes)
                raise OSError("simulated staged-file write failure")

            with patch(
                "evolution_sim.io.open_ecology_aggregate_commit._write_new_file",
                side_effect=replace_stage_then_fail,
            ):
                with self.assertRaises(Exception):
                    self._publish_first(fixture)

            self.assertIsNotNone(staged_path)
            assert staged_path is not None
            self.assertEqual(
                (staged_path / "attacker-sentinel").read_bytes(),
                attacker_bytes,
            )
            self.assertTrue(validated_backup.is_dir())

    def test_current_temp_cleanup_preserves_postcheck_replacement(self) -> None:
        from evolution_sim.io import open_ecology_aggregate_commit as aggregate_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "aggregate"
            root.mkdir()
            victim = root.parent / ".aggregate.CURRENT.injected.tmp"
            victim.write_bytes(b"validated-current-temp\n")
            identity = aggregate_module._regular_file_identity(
                victim,
                field="CURRENT cleanup victim",
            )
            validated_backup = root.parent / "validated-current-open-inode"
            attacker_bytes = b"replacement-must-survive-ftruncate\n"
            real_ftruncate = os.ftruncate
            injected = False

            def replace_after_descriptor_validation(
                descriptor: int,
                length: int,
            ) -> None:
                nonlocal injected
                if not injected:
                    tombstone = next(
                        (
                            root
                            / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
                        ).glob("*/candidate")
                    )
                    tombstone.rename(validated_backup)
                    tombstone.write_bytes(attacker_bytes)
                    injected = True
                real_ftruncate(descriptor, length)

            with patch.object(
                aggregate_module.os,
                "ftruncate",
                side_effect=replace_after_descriptor_validation,
            ):
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "changed during cleanup",
                ):
                    aggregate_module._unlink_owned_regular_file(
                        victim,
                        expected_identity=identity,
                        field="CURRENT cleanup victim",
                        cleanup_root=(
                            root
                            / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
                        ),
                        max_tombstones=8,
                    )

            self.assertTrue(injected)
            replacement = next(
                (
                    root
                    / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
                ).glob("*/candidate")
            )
            self.assertEqual(replacement.read_bytes(), attacker_bytes)
            self.assertEqual(validated_backup.read_bytes(), b"")

    def test_stage_cleanup_namespace_replacement_cannot_redirect_move(self) -> None:
        from evolution_sim.io import open_ecology_aggregate_commit as aggregate_module

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            root = parent / "aggregate"
            root.mkdir()
            stage_prefix = ".aggregate.generation-0000000000000000.stage-"
            stage = parent / f"{stage_prefix}probe"
            stage.mkdir()
            (stage / "checkpoint.json").write_bytes(b"validated-stage\n")
            identity = aggregate_module._directory_identity(
                stage,
                field="stage cleanup victim",
            )
            cleanup = (
                root / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
            )
            displaced = root / "displaced-cleanup-root"
            outside = root / "attacker-selected"
            outside.mkdir()
            real_rename = os.rename
            injected = False

            def replace_cleanup_namespace(
                source: object,
                destination: object,
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal injected
                if (
                    Path(os.fsdecode(source)).name == stage.name
                    and Path(os.fsdecode(destination)).name == "candidate"
                    and not injected
                ):
                    cleanup.rename(displaced)
                    cleanup.symlink_to(outside, target_is_directory=True)
                    injected = True
                real_rename(source, destination, *args, **kwargs)

            with patch.object(
                aggregate_module.os,
                "rename",
                side_effect=replace_cleanup_namespace,
            ):
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "namespace changed",
                ):
                    aggregate_module._remove_owned_stage(
                        stage,
                        parent=parent,
                        prefix=stage_prefix,
                        expected_identity=identity,
                        cleanup_root=cleanup,
                        max_tombstones=8,
                    )

            self.assertTrue(injected)
            self.assertEqual(list(outside.iterdir()), [])
            retained = next(displaced.glob("*/candidate/checkpoint.json"))
            self.assertEqual(retained.read_bytes(), b"validated-stage\n")

    def test_aggregate_tombstones_are_bounded_and_crash_reconciled(self) -> None:
        from evolution_sim.io import open_ecology_aggregate_commit as aggregate_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "aggregate"
            root.mkdir()
            cleanup = (
                root / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
            )
            victim = root.parent / ".aggregate.CURRENT.injected.tmp"
            victim.write_bytes(b"crash-leftover\n")
            identity = aggregate_module._regular_file_identity(
                victim,
                field="crash cleanup victim",
            )
            with patch.object(
                aggregate_module.os,
                "ftruncate",
                side_effect=SystemExit("simulated truncation crash"),
            ):
                with self.assertRaisesRegex(SystemExit, "truncation crash"):
                    aggregate_module._unlink_owned_regular_file(
                        victim,
                        expected_identity=identity,
                        field="crash cleanup victim",
                        cleanup_root=cleanup,
                        max_tombstones=8,
                    )

            candidate = next(cleanup.glob("*/candidate"))
            self.assertEqual(candidate.read_bytes(), b"crash-leftover\n")
            self.assertEqual(cleanup.stat().st_dev, root.stat().st_dev)
            aggregate_module._reconcile_aggregate_cleanup_tombstones(
                root,
                max_tombstones=8,
            )
            self.assertEqual(candidate.read_bytes(), b"")

            second = root.parent / ".aggregate.CURRENT.second.tmp"
            second.write_bytes(b"second\n")
            second_identity = aggregate_module._regular_file_identity(
                second,
                field="second cleanup victim",
            )
            with self.assertRaisesRegex(
                OpenEcologyAggregateCommitError,
                "tombstone limit",
            ):
                aggregate_module._unlink_owned_regular_file(
                    second,
                    expected_identity=second_identity,
                    field="second cleanup victim",
                    cleanup_root=cleanup,
                    max_tombstones=1,
                )
            self.assertEqual(second.read_bytes(), b"second\n")

    def test_aggregate_cleanup_rejects_cross_filesystem_quarantine(self) -> None:
        from evolution_sim.io import open_ecology_aggregate_commit as aggregate_module

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "aggregate"
            root.mkdir()
            victim = root.parent / ".aggregate.CURRENT.cross-device.tmp"
            victim.write_bytes(b"same-filesystem-required\n")
            identity = aggregate_module._regular_file_identity(
                victim,
                field="cross-device cleanup victim",
            )
            real_open_parent = aggregate_module._open_owned_cleanup_parent

            def report_distinct_source_device(
                path: Path,
                *,
                field: str,
            ) -> tuple[int, object]:
                descriptor, metadata = real_open_parent(path, field=field)
                if path == victim.parent:
                    return (
                        descriptor,
                        SimpleNamespace(st_dev=metadata.st_dev + 1),
                    )
                return descriptor, metadata

            with patch.object(
                aggregate_module,
                "_open_owned_cleanup_parent",
                side_effect=report_distinct_source_device,
            ):
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "same-filesystem quarantine",
                ):
                    aggregate_module._unlink_owned_regular_file(
                        victim,
                        expected_identity=identity,
                        field="cross-device cleanup victim",
                        cleanup_root=(
                            root
                            / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
                        ),
                        max_tombstones=8,
                    )

            self.assertEqual(victim.read_bytes(), b"same-filesystem-required\n")

    def test_relocated_aggregate_tombstones_are_inert_and_bounded(self) -> None:
        from evolution_sim.io import open_ecology_aggregate_commit as aggregate_module

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            root = parent / "aggregate"
            root.mkdir()
            cleanup = (
                root / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
            )
            file_victim = parent / ".aggregate.CURRENT.archive-copy.tmp"
            file_victim.write_bytes(b"temporary-pointer\n")
            aggregate_module._unlink_owned_regular_file(
                file_victim,
                expected_identity=aggregate_module._regular_file_identity(
                    file_victim,
                    field="archive-copy file victim",
                ),
                field="archive-copy file victim",
                cleanup_root=cleanup,
                max_tombstones=8,
            )
            stage_prefix = ".aggregate.generation-0000000000000000.stage-"
            stage = parent / f"{stage_prefix}archive-copy"
            stage.mkdir()
            (stage / "checkpoint.json").write_bytes(b"retained-stage\n")
            aggregate_module._remove_owned_stage(
                stage,
                parent=parent,
                prefix=stage_prefix,
                expected_identity=aggregate_module._directory_identity(
                    stage,
                    field="archive-copy stage victim",
                ),
                cleanup_root=cleanup,
                max_tombstones=8,
            )

            relocated = parent / "aggregate-relocated"
            shutil.copytree(root, relocated)
            self.assertEqual(
                aggregate_module._reconcile_aggregate_cleanup_tombstones(
                    relocated,
                    max_tombstones=8,
                ),
                2,
            )
            relocated_cleanup = (
                relocated
                / aggregate_module.OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
            )
            retained_stage = next(
                relocated_cleanup.glob("stage-*/candidate/checkpoint.json")
            )
            self.assertEqual(retained_stage.read_bytes(), b"retained-stage\n")

            relocated_file = next(relocated_cleanup.glob("file-*/candidate"))
            relocated_file.write_bytes(b"relocated-nonzero-replacement\n")
            with self.assertRaisesRegex(
                OpenEcologyAggregateCommitError,
                "changed before descriptor-bound truncation",
            ):
                aggregate_module._reconcile_aggregate_cleanup_tombstones(
                    relocated,
                    max_tombstones=8,
                )
            self.assertEqual(
                relocated_file.read_bytes(),
                b"relocated-nonzero-replacement\n",
            )

    def test_cross_process_writer_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            lock_path = fixture.root / OPEN_ECOLOGY_AGGREGATE_LOCK_NAME
            script = (
                "import fcntl,sys\n"
                "f=open(sys.argv[1],'r+b',buffering=0)\n"
                "fcntl.flock(f.fileno(),fcntl.LOCK_EX)\n"
                "print('locked',flush=True)\n"
                "sys.stdin.readline()\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(lock_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert process.stdout is not None
                self.assertEqual(process.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "held by another process",
                ):
                    load_current_open_ecology_aggregate_generation(
                        fixture.root,
                        evidence_directory=fixture.evidence,
                        pins=self._resume_pins(first),
                    )
            finally:
                if process.stdin is not None:
                    process.stdin.write("\n")
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=10)
                if process.returncode != 0:
                    stderr = "" if process.stderr is None else process.stderr.read()
                    self.fail(f"lock helper failed: {stderr}")
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()

    def test_active_evidence_writer_blocks_aggregate_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            active_writer = RotatingOpenEcologyEvidenceWriter(
                fixture.evidence,
                run_id="persistent-island-0",
                source_contract=fixture.source_contract,
                config=fixture.writer_config,
                continuation_state=fixture.continuation,
            )
            try:
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "evidence writer lock is held",
                ):
                    self._load_current(fixture, first)
            finally:
                active_writer.abort()

    def test_resume_requires_every_exact_external_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first = self._publish_first(fixture)
            correct = self._resume_pins(first)
            cases = {
                "source_git_sha": OpenEcologyAggregateResumePins(
                    identity=OpenEcologyAggregateIdentityPins(
                        source_git_sha="b" * 40,
                        config_contract_sha256=correct.identity.config_contract_sha256,
                        seed_contract_sha256=correct.identity.seed_contract_sha256,
                        run_generation_id=correct.identity.run_generation_id,
                        island_id=correct.identity.island_id,
                        simulation_generation_index=(
                            correct.identity.simulation_generation_index
                        ),
                        tick=correct.identity.tick,
                    ),
                    aggregate_generation_index=correct.aggregate_generation_index,
                    commit_sha256=correct.commit_sha256,
                    checkpoint_sha256=correct.checkpoint_sha256,
                    checkpoint_generation_identity_sha256=(
                        correct.checkpoint_generation_identity_sha256
                    ),
                    evidence_manifest_sha256=correct.evidence_manifest_sha256,
                    evidence_manifest_status=correct.evidence_manifest_status,
                ),
                "commit_sha256": self._replace_resume_pin(
                    correct,
                    commit_sha256="f" * 64,
                ),
                "checkpoint_sha256": self._replace_resume_pin(
                    correct,
                    checkpoint_sha256="f" * 64,
                ),
                "checkpoint_generation_identity_sha256": (
                    self._replace_resume_pin(
                        correct,
                        checkpoint_generation_identity_sha256="f" * 64,
                    )
                ),
                "evidence_manifest_sha256": self._replace_resume_pin(
                    correct,
                    evidence_manifest_sha256="f" * 64,
                ),
            }
            for label, pins in cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        OpenEcologyAggregateCommitError,
                        "pin mismatch",
                    ):
                        load_current_open_ecology_aggregate_generation(
                            fixture.root,
                            evidence_directory=fixture.evidence,
                            pins=pins,
                        )

    def test_surplus_symlink_duplicate_and_path_traversal_fail_closed(self) -> None:
        with self.subTest(case="surplus"):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                first = self._publish_first(fixture)
                generation = self._generation_path(fixture.root, 0)
                (generation / "surplus.txt").write_text(
                    "not declared",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "surplus",
                ):
                    self._load_current(fixture, first)

        with self.subTest(case="symlink"):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                first = self._publish_first(fixture)
                checkpoint_path = (
                    self._generation_path(fixture.root, 0) / "checkpoint.json"
                )
                checkpoint_copy = Path(tmpdir) / "checkpoint-copy.json"
                checkpoint_copy.write_bytes(checkpoint_path.read_bytes())
                checkpoint_path.unlink()
                checkpoint_path.symlink_to(checkpoint_copy)
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "single-link regular",
                ):
                    self._load_current(fixture, first)

        with self.subTest(case="duplicate-current-key"):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                first = self._publish_first(fixture)
                pointer_path = fixture.root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
                pointer_path.write_text(
                    '{"schema_version":"a","schema_version":"b"}\n',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "duplicate JSON key",
                ):
                    self._load_current(fixture, first)

        with self.subTest(case="path-traversal"):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                first = self._publish_first(fixture)
                commit_path = self._commit_path(fixture.root, 0)
                commit = self._load_json(commit_path)
                commit["checkpoint"]["file_name"] = "../checkpoint.json"
                self._redigest(commit, "commit_sha256")
                commit_path.write_bytes(self._canonical_line(commit))
                pointer_path = fixture.root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
                pointer = self._load_json(pointer_path)
                pointer["commit_sha256"] = commit["commit_sha256"]
                self._redigest(pointer, "pointer_sha256")
                pointer_path.write_bytes(self._canonical_line(pointer))
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "file name is not canonical",
                ):
                    self._load_current(fixture, first)

    def test_external_shard_mutation_and_snapshot_mutation_fail_closed(self) -> None:
        with self.subTest(case="external-shard"):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                first = self._publish_first(fixture)
                second = self._publish_second(fixture, first=first)
                manifest = second["evidence_manifest"]
                shard_name = manifest["completed_shards"][0]["file_name"]
                shard_path = fixture.evidence / shard_name
                shard_path.write_bytes(shard_path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "does not match snapshot",
                ):
                    self._load_current(fixture, second)

        with self.subTest(case="manifest-snapshot"):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                first = self._publish_first(fixture)
                snapshot = self._evidence_snapshot_path(fixture.root, 0)
                snapshot.write_bytes(snapshot.read_bytes() + b" ")
                with self.assertRaisesRegex(
                    OpenEcologyAggregateCommitError,
                    "file binding mismatch",
                ):
                    self._load_current(fixture, first)

    def test_metadata_bounds_and_contiguous_generation_policy_preserve_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(Path(tmpdir))
            first_inputs = self._first_inputs(fixture)
            with self.assertRaisesRegex(
                OpenEcologyAggregateCommitError,
                "max_commit_bytes",
            ):
                self._publish(
                    fixture,
                    aggregate_generation_index=0,
                    checkpoint=first_inputs.checkpoint,
                    identity=first_inputs.identity,
                    manifest=first_inputs.manifest,
                    previous=None,
                    max_commit_bytes=64,
                )
            generations = fixture.root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
            self.assertEqual(list(generations.iterdir()), [])

            first = self._publish(
                fixture,
                aggregate_generation_index=0,
                checkpoint=first_inputs.checkpoint,
                identity=first_inputs.identity,
                manifest=first_inputs.manifest,
                previous=None,
            )
            second_inputs = self._second_inputs(fixture)
            with self.assertRaisesRegex(
                OpenEcologyAggregateCommitError,
                "contiguous",
            ):
                self._publish(
                    fixture,
                    aggregate_generation_index=2,
                    checkpoint=second_inputs.checkpoint,
                    identity=second_inputs.identity,
                    manifest=second_inputs.manifest,
                    previous=first,
                )
            self.assertTrue(self._generation_path(fixture.root, 0).is_dir())
            self.assertFalse(self._generation_path(fixture.root, 1).exists())

    def test_nonfinite_and_invalid_evidence_status_are_rejected_early(self) -> None:
        with self.assertRaisesRegex(
            OpenEcologyAggregateCommitError,
            "nonnegative integer",
        ):
            OpenEcologyAggregateIdentityPins(
                source_git_sha="a" * 40,
                config_contract_sha256="b" * 64,
                seed_contract_sha256="c" * 64,
                run_generation_id="generation-0",
                island_id="island-0",
                simulation_generation_index=0,
                tick=float("nan"),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(
            OpenEcologyAggregateCommitError,
            "exactly open",
        ):
            OpenEcologyAggregateResumePins(
                identity=self._identity_stub(),
                aggregate_generation_index=0,
                commit_sha256="1" * 64,
                checkpoint_sha256="2" * 64,
                checkpoint_generation_identity_sha256="3" * 64,
                evidence_manifest_sha256="4" * 64,
                evidence_manifest_status="complete",
            )

    def test_unrelated_evidence_source_contract_cannot_be_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self._fixture(
                Path(tmpdir),
                source_contract_overrides={"island_id": "island-other"},
            )
            inputs = self._first_inputs(fixture)
            with self.assertRaisesRegex(
                OpenEcologyAggregateCommitError,
                "evidence source contract island_id",
            ):
                self._publish(
                    fixture,
                    aggregate_generation_index=0,
                    checkpoint=inputs.checkpoint,
                    identity=inputs.identity,
                    manifest=inputs.manifest,
                    previous=None,
                )

    @unittest.skipUnless(_TORCH_AVAILABLE, "runtime adapter requires torch")
    def test_runtime_adapter_continuation_is_strictly_extracted_and_bound(
        self,
    ) -> None:
        cases = {
            "valid": (0, None),
            "valid_lag": (1, None),
            "stale_schema": (0, "schema is unsupported"),
            "wrong_binding": (0, "does not match checkpoint"),
            "malformed_continuation": (
                0,
                "runtime evidence adapter is invalid",
            ),
        }
        for case, (checkpoint_tick, expected_error) in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmpdir:
                fixture = self._fixture(Path(tmpdir))
                writer = RotatingOpenEcologyEvidenceWriter(
                    fixture.evidence,
                    run_id="persistent-island-0",
                    source_contract=fixture.source_contract,
                    config=fixture.writer_config,
                    continuation_state=fixture.continuation,
                )
                writer.append(
                    BirthEvidence(
                        tick=0,
                        event_index=0,
                        event_id="birth:adapter:0",
                        child_agent_id=9,
                        parent_agent_ids=(1, 2),
                        lineage_id="lineage-9",
                        genome_sha256="e" * 64,
                    )
                )
                fixture.continuation = writer.checkpoint()
                checkpoint, identity = self._checkpoint(
                    fixture.root.parent / "checkpoint-0.json",
                    continuation=fixture.continuation,
                    simulation_generation_index=0,
                    tick=checkpoint_tick,
                    runtime_adapter_case=case,
                )
                manifest = load_open_ecology_evidence_manifest(
                    fixture.evidence,
                    verify_shards=True,
                    expected_status="open",
                )
                if expected_error is None:
                    published = self._publish(
                        fixture,
                        aggregate_generation_index=0,
                        checkpoint=checkpoint,
                        identity=identity,
                        manifest=manifest,
                        previous=None,
                    )
                    self.assertEqual(
                        published["commit"]["evidence"]["continuation_state_sha256"],
                        fixture.continuation["state_sha256"],
                    )
                else:
                    with self.assertRaisesRegex(
                        OpenEcologyAggregateCommitError,
                        expected_error,
                    ):
                        self._publish(
                            fixture,
                            aggregate_generation_index=0,
                            checkpoint=checkpoint,
                            identity=identity,
                            manifest=manifest,
                            previous=None,
                        )

    def _fixture(
        self,
        root: Path,
        *,
        source_contract_overrides: dict[str, object] | None = None,
    ) -> _Fixture:
        evidence = root / "evidence"
        aggregate = root / "aggregate"
        source_contract = {
            "campaign_contract_sha256": "d" * 64,
            "config_contract_sha256": self._state_digest(
                "open_ecology_world_config_v1",
                self._config_payload(),
            ),
            "event_producer": "test_explicit_runtime_events_v1",
            "island_id": "island-0",
            "run_generation_id": self._run_generation_id(),
            "seed_contract_sha256": self._state_digest(
                "open_ecology_seed_contract_v1",
                self._seed_payload(),
            ),
            "source_git_sha": self._source_sha(),
        }
        if source_contract_overrides is not None:
            source_contract.update(source_contract_overrides)
        config = RotatingEvidenceConfig(
            max_ticks_per_shard=8,
            max_rows_per_shard=4,
            max_uncompressed_bytes_per_shard=64 * 1024,
            max_compressed_bytes_per_shard=64 * 1024,
            max_event_bytes=8 * 1024,
            max_shards=16,
            max_manifest_bytes=64 * 1024,
            max_source_contract_bytes=8 * 1024,
            max_total_compressed_bytes=1024 * 1024,
        )
        writer = RotatingOpenEcologyEvidenceWriter(
            evidence,
            run_id="persistent-island-0",
            source_contract=source_contract,
            config=config,
        )
        continuation = writer.checkpoint()
        return _Fixture(
            root=aggregate,
            evidence=evidence,
            source_contract=source_contract,
            writer_config=config,
            continuation=continuation,
        )

    def _first_inputs(self, fixture: _Fixture) -> _Inputs:
        checkpoint, identity = self._checkpoint(
            fixture.root.parent / "checkpoint-0.json",
            continuation=fixture.continuation,
            simulation_generation_index=0,
            tick=0,
        )
        manifest = load_open_ecology_evidence_manifest(
            fixture.evidence,
            verify_shards=True,
            expected_status="open",
        )
        return _Inputs(checkpoint=checkpoint, identity=identity, manifest=manifest)

    def _second_inputs(self, fixture: _Fixture) -> _Inputs:
        writer = RotatingOpenEcologyEvidenceWriter(
            fixture.evidence,
            run_id="persistent-island-0",
            source_contract=fixture.source_contract,
            config=fixture.writer_config,
            continuation_state=fixture.continuation,
        )
        writer.append(
            BirthEvidence(
                tick=5,
                event_index=0,
                event_id="birth:0",
                child_agent_id=9,
                parent_agent_ids=(1, 2),
                lineage_id="lineage-9",
                genome_sha256="e" * 64,
            )
        )
        fixture.continuation = writer.checkpoint()
        checkpoint, identity = self._checkpoint(
            fixture.root.parent / "checkpoint-1.json",
            continuation=fixture.continuation,
            simulation_generation_index=1,
            tick=8,
        )
        manifest = load_open_ecology_evidence_manifest(
            fixture.evidence,
            verify_shards=True,
            expected_status="open",
        )
        return _Inputs(checkpoint=checkpoint, identity=identity, manifest=manifest)

    def _publish_first(self, fixture: _Fixture) -> dict[str, object]:
        inputs = self._first_inputs(fixture)
        return self._publish(
            fixture,
            aggregate_generation_index=0,
            checkpoint=inputs.checkpoint,
            identity=inputs.identity,
            manifest=inputs.manifest,
            previous=None,
        )

    def _publish_second(
        self,
        fixture: _Fixture,
        *,
        first: dict[str, object],
    ) -> dict[str, object]:
        inputs = self._second_inputs(fixture)
        return self._publish(
            fixture,
            aggregate_generation_index=1,
            checkpoint=inputs.checkpoint,
            identity=inputs.identity,
            manifest=inputs.manifest,
            previous=first,
        )

    def _publish(
        self,
        fixture: _Fixture,
        *,
        aggregate_generation_index: int,
        checkpoint: dict[str, object],
        identity: OpenEcologyAggregateIdentityPins,
        manifest: dict[str, object],
        previous: dict[str, object] | None,
        max_commit_bytes: int | None = None,
    ) -> dict[str, object]:
        checkpoint_path = (
            fixture.root.parent
            / f"checkpoint-{identity.simulation_generation_index}.json"
        )
        kwargs: dict[str, object] = {}
        if max_commit_bytes is not None:
            kwargs["max_commit_bytes"] = max_commit_bytes
        return publish_open_ecology_aggregate_generation(
            fixture.root,
            checkpoint_path=checkpoint_path,
            evidence_directory=fixture.evidence,
            identity=identity,
            aggregate_generation_index=aggregate_generation_index,
            expected_previous_commit_sha256=(
                None if previous is None else str(previous["commit"]["commit_sha256"])
            ),
            expected_checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
            expected_checkpoint_generation_identity_sha256=str(
                checkpoint["generation_identity"]["identity_sha256"]
            ),
            expected_evidence_manifest_sha256=str(manifest["manifest_sha256"]),
            expected_evidence_manifest_status=str(manifest["status"]),
            **kwargs,
        )

    def _checkpoint(
        self,
        path: Path,
        *,
        continuation: dict[str, object],
        simulation_generation_index: int,
        tick: int,
        runtime_adapter_case: str | None = None,
    ) -> tuple[dict[str, object], OpenEcologyAggregateIdentityPins]:
        state = self._state
        config_contract = state(
            "open_ecology_world_config_v1",
            self._config_payload(),
        )
        seed_contract = state(
            "open_ecology_seed_contract_v1",
            self._seed_payload(),
        )
        evidence_state = state(
            OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA,
            continuation,
        )
        if runtime_adapter_case is not None:
            binding_payload: dict[str, object] = {
                "schema_version": _RUNTIME_BINDING_SCHEMA,
                "source_git_sha": self._source_sha(),
                "source_manifest_sha256": "f" * 64,
                "config_contract_sha256": self._state_digest(
                    config_contract.schema_version,
                    dict(config_contract.payload),
                ),
                "seed_contract_sha256": self._state_digest(
                    seed_contract.schema_version,
                    dict(seed_contract.payload),
                ),
                "run_generation_id": self._run_generation_id(),
                "island_id": "island-0",
                "generation_index": simulation_generation_index,
                "completed_tick": tick,
            }
            nested_continuation: object = continuation
            adapter_schema = _RUNTIME_EVIDENCE_ADAPTER_SCHEMA
            if runtime_adapter_case == "stale_schema":
                adapter_schema = "open_ecology_evidence_writer_continuation_adapter_v0"
            elif runtime_adapter_case == "wrong_binding":
                binding_payload["source_git_sha"] = "b" * 40
            elif runtime_adapter_case == "malformed_continuation":
                nested_continuation = {}
            elif runtime_adapter_case not in {"valid", "valid_lag"}:
                raise AssertionError(
                    f"unknown runtime adapter test case: {runtime_adapter_case}"
                )
            evidence_state = state(
                adapter_schema,
                {
                    "binding": binding_payload,
                    "state": {"continuation_state": nested_continuation},
                },
            )
        checkpoint = write_open_ecology_checkpoint(
            path,
            source_git_sha=self._source_sha(),
            config_contract=config_contract,
            seed_contract=seed_contract,
            run_generation_id=self._run_generation_id(),
            island_id="island-0",
            generation_index=simulation_generation_index,
            tick=tick,
            world_state=state(
                "simulation_world_state_adapter_v1",
                {"agents": [{"alive": True, "id": 7}], "tick": tick},
            ),
            environment_rng_state=state(
                "python_random_state_json_v1",
                {"state": [11, 29, 47]},
            ),
            recurrent_policy_state=state(
                "frozen_recurrent_policy_state_v1",
                {"artifact_sha256": "1" * 64},
            ),
            public_feedback_history=state(
                "public_feedback_history_v1",
                {"previous_feedback_by_agent": {"7": [0.0, 1.0]}},
            ),
            sampling_rng_state=state(
                "torch_generator_state_json_v1",
                {"state_bytes_hex": "00a1ff"},
            ),
            genome_population_snapshot=state(
                "recurrent_genome_population_v1",
                {"agent_genomes": {"7": "2" * 64}},
            ),
            evidence_writer_continuation_state=evidence_state,
        )
        source = checkpoint["source"]
        generation_identity = checkpoint["generation_identity"]
        identity = OpenEcologyAggregateIdentityPins(
            source_git_sha=self._source_sha(),
            config_contract_sha256=str(source["config_contract"]["state_sha256"]),
            seed_contract_sha256=str(source["seed_contract"]["state_sha256"]),
            run_generation_id=str(generation_identity["run_generation_id"]),
            island_id=str(generation_identity["island_id"]),
            simulation_generation_index=simulation_generation_index,
            tick=tick,
        )
        return checkpoint, identity

    def _state(
        self,
        schema_version: str,
        payload: dict[str, object],
    ) -> VersionedCheckpointState:
        return VersionedCheckpointState(schema_version, payload)

    def _resume_pins(
        self,
        loaded: dict[str, object],
    ) -> OpenEcologyAggregateResumePins:
        commit = loaded["commit"]
        identity = commit["identity"]
        checkpoint = commit["checkpoint"]
        evidence = commit["evidence"]
        return OpenEcologyAggregateResumePins(
            identity=OpenEcologyAggregateIdentityPins(
                source_git_sha=str(identity["source_git_sha"]),
                config_contract_sha256=str(identity["config_contract_sha256"]),
                seed_contract_sha256=str(identity["seed_contract_sha256"]),
                run_generation_id=str(identity["run_generation_id"]),
                island_id=str(identity["island_id"]),
                simulation_generation_index=int(
                    identity["simulation_generation_index"]
                ),
                tick=int(identity["tick"]),
            ),
            aggregate_generation_index=int(commit["aggregate_generation_index"]),
            commit_sha256=str(commit["commit_sha256"]),
            checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
            checkpoint_generation_identity_sha256=str(
                checkpoint["generation_identity_sha256"]
            ),
            evidence_manifest_sha256=str(evidence["manifest_sha256"]),
            evidence_manifest_status=str(evidence["status"]),
        )

    def _replace_resume_pin(
        self,
        pins: OpenEcologyAggregateResumePins,
        **changes: object,
    ) -> OpenEcologyAggregateResumePins:
        payload = {
            "identity": pins.identity,
            "aggregate_generation_index": pins.aggregate_generation_index,
            "commit_sha256": pins.commit_sha256,
            "checkpoint_sha256": pins.checkpoint_sha256,
            "checkpoint_generation_identity_sha256": (
                pins.checkpoint_generation_identity_sha256
            ),
            "evidence_manifest_sha256": pins.evidence_manifest_sha256,
            "evidence_manifest_status": pins.evidence_manifest_status,
        }
        payload.update(changes)
        return OpenEcologyAggregateResumePins(**payload)

    def _load_current(
        self,
        fixture: _Fixture,
        loaded: dict[str, object],
    ) -> dict[str, object]:
        return load_current_open_ecology_aggregate_generation(
            fixture.root,
            evidence_directory=fixture.evidence,
            pins=self._resume_pins(loaded),
        )

    def _generation_path(self, root: Path, index: int) -> Path:
        return (
            root
            / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
            / f"generation-{index:016d}"
        )

    def _commit_path(self, root: Path, index: int) -> Path:
        return self._generation_path(root, index) / OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME

    def _evidence_snapshot_path(self, root: Path, index: int) -> Path:
        return self._generation_path(root, index) / "evidence-manifest.json"

    def _load_json(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _redigest(self, payload: dict[str, object], digest_key: str) -> None:
        body = copy.deepcopy(payload)
        body.pop(digest_key, None)
        payload[digest_key] = hashlib.sha256(self._canonical(body)).hexdigest()

    def _canonical(self, payload: object) -> bytes:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")

    def _canonical_line(self, payload: object) -> bytes:
        return self._canonical(payload) + b"\n"

    def _source_sha(self) -> str:
        return "a" * 40

    def _config_payload(self) -> dict[str, object]:
        return {
            "height": 32,
            "island_count": 1,
            "source_manifest_sha256": "f" * 64,
            "width": 48,
        }

    def _seed_payload(self) -> dict[str, object]:
        return {
            "environment_seed": 101,
            "genome_founder_seed": 303,
            "policy_sampling_seed": 202,
        }

    def _run_generation_id(self) -> str:
        return "persistent-run-generation"

    def _state_digest(
        self,
        schema_version: str,
        payload: dict[str, object],
    ) -> str:
        return hashlib.sha256(
            self._canonical(
                {
                    "payload": payload,
                    "schema_version": schema_version,
                }
            )
        ).hexdigest()

    def _identity_stub(self) -> OpenEcologyAggregateIdentityPins:
        return OpenEcologyAggregateIdentityPins(
            source_git_sha="a" * 40,
            config_contract_sha256="b" * 64,
            seed_contract_sha256="c" * 64,
            run_generation_id="generation-0",
            island_id="island-0",
            simulation_generation_index=0,
            tick=0,
        )


class _Fixture:
    def __init__(
        self,
        *,
        root: Path,
        evidence: Path,
        source_contract: dict[str, object],
        writer_config: RotatingEvidenceConfig,
        continuation: dict[str, object],
    ):
        self.root = root
        self.evidence = evidence
        self.source_contract = source_contract
        self.writer_config = writer_config
        self.continuation = continuation


class _Inputs:
    def __init__(
        self,
        *,
        checkpoint: dict[str, object],
        identity: OpenEcologyAggregateIdentityPins,
        manifest: dict[str, object],
    ):
        self.checkpoint = checkpoint
        self.identity = identity
        self.manifest = manifest


if __name__ == "__main__":
    unittest.main()
