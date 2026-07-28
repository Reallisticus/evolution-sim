from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from evolution_sim.mind.open_ecology_process_workers import (
    OPEN_ECOLOGY_PROCESS_WORKER_FATAL_MARKER_NAME,
    OpenEcologyProcessWorkerError,
    PersistentProcessWorkerLauncher,
    ProcessWorkerExpectedState,
    ProcessWorkerFrontierBatch,
    ProcessWorkerSlot,
    build_process_worker_slots,
    local_process_worker_host_identity,
    process_worker_assignment_sha256,
    _run_bounded_git,
    strict_open_ecology_source_snapshot,
)
from evolution_sim.mind.open_ecology_persistent_island import (
    PersistentArtifactBinding,
    build_persistent_island_task_matrix,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_SEED_REGISTRY,
)


REQUIRES_MIND_ML = True
_SOURCE_COMMIT = "a" * 40
_SOURCE_MANIFEST = "b" * 64


class OpenEcologyProcessWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.campaign_root = self.base / "campaign"
        self.campaign_root.mkdir()
        self.repository_root = Path(__file__).resolve().parents[2]
        self.tasks = build_persistent_island_task_matrix(
            self._bindings(),
            selected_density=64,
        )
        self.host_identity = local_process_worker_host_identity()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_processes_reuse_runners_and_restore_external_pins(self) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=2,
            host_identity=self.host_identity,
        )
        launcher = self._launcher(slots=slots)
        original_pids = dict(launcher.worker_pids)
        self.assertEqual(len(set(original_pids.values())), 2)
        self.assertNotIn(os.getpid(), original_pids.values())

        selected = tuple(task.task_id for task in self.tasks[:3])
        first = launcher.advance_frontier(
            selected_task_ids=selected,
            target_tick=5_000,
            expected_states=self._pending_states(selected),
        )
        self.assertEqual(first.selected_task_ids, selected)
        self.assertEqual(first.assignment_sha256, launcher.assignment_sha256)
        self.assertEqual(
            {result.worker_pid for result in first.results_by_task.values()},
            set(original_pids.values()),
        )
        views = first.runner_views({task.task_id: task for task in self.tasks})
        self.assertEqual(tuple(views), selected)
        self.assertEqual(
            [views[task_id].observed_tick for task_id in selected], [5_000] * 3
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "read-only"):
            views[selected[0]].advance_to(10_000)

        second = launcher.advance_frontier(
            selected_task_ids=selected,
            target_tick=10_000,
            expected_states=self._states_from_batch(first),
        )
        self.assertEqual(dict(launcher.worker_pids), original_pids)
        self.assertEqual(
            [second.results_by_task[task_id].observed_tick for task_id in selected],
            [10_000] * 3,
        )
        launcher.close()
        self.assertTrue(launcher.closed)
        self.assertFalse(launcher.poisoned)
        self.assertEqual(
            len(tuple(self.campaign_root.glob("*.preserved"))),
            4,
        )

        resumed = self._launcher(slots=slots)
        third = resumed.advance_frontier(
            selected_task_ids=selected,
            target_tick=15_000,
            expected_states=self._states_from_batch(second),
        )
        self.assertEqual(
            [third.results_by_task[task_id].observed_tick for task_id in selected],
            [15_000] * 3,
        )
        self.assertTrue(
            all(
                third.results_by_task[task_id].aggregate_resume_pins.identity.tick
                == 14_999
                for task_id in selected
            )
        )
        resumed.close()
        self.assertEqual(
            len(tuple(self.campaign_root.glob("*.preserved"))),
            6,
        )

    def test_duplicate_or_nonstatic_scope_fails_before_process_mutation(self) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=2,
            host_identity=self.host_identity,
        )
        launcher = self._launcher(slots=slots)
        first_id = self.tasks[0].task_id
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "duplicated"):
            launcher.advance_frontier(
                selected_task_ids=(first_id, first_id),
                target_tick=5_000,
                expected_states={first_id: self._pending_state(first_id)},
            )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "matrix order"):
            launcher.advance_frontier(
                selected_task_ids=(self.tasks[1].task_id, first_id),
                target_tick=5_000,
                expected_states={
                    first_id: self._pending_state(first_id),
                    self.tasks[1].task_id: self._pending_state(self.tasks[1].task_id),
                },
            )
        self.assertFalse(tuple(self.campaign_root.glob("*.preserved")))
        launcher.close()

    def test_runner_views_rejects_missing_selected_task_even_with_extra_key(
        self,
    ) -> None:
        selected = (self.tasks[0].task_id, self.tasks[1].task_id)
        batch = object.__new__(ProcessWorkerFrontierBatch)
        object.__setattr__(batch, "selected_task_ids", selected)
        object.__setattr__(batch, "results_by_task", {})
        hostile_task_map = {
            self.tasks[0].task_id: self.tasks[0],
            self.tasks[2].task_id: self.tasks[2],
        }

        with self.assertRaisesRegex(
            OpenEcologyProcessWorkerError,
            f"runner view task map is incomplete: {selected[1]}",
        ):
            batch.runner_views(hostile_task_map)

    def test_worker_death_fails_closed_and_preserves_partial_evidence(self) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=1,
            host_identity=self.host_identity,
        )
        launcher = self._launcher(
            slots=slots,
            behaviors={0: "die_on_advance"},
            response_timeout_seconds=5.0,
        )
        task_id = self.tasks[0].task_id
        with self.assertRaisesRegex(
            OpenEcologyProcessWorkerError,
            "died|receive bounded",
        ):
            launcher.advance_frontier(
                selected_task_ids=(task_id,),
                target_tick=5_000,
                expected_states={task_id: self._pending_state(task_id)},
            )
        self.assertTrue(launcher.poisoned)
        self.assertTrue(launcher.closed)
        self.assertEqual(
            len(tuple(self.campaign_root.glob("*.preserved"))),
            1,
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "closed"):
            launcher.advance_frontier(
                selected_task_ids=(task_id,),
                target_tick=5_000,
                expected_states={task_id: self._pending_state(task_id)},
            )

    def test_worker_timeout_is_bounded_and_preserves_files(self) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=1,
            host_identity=self.host_identity,
        )
        launcher = self._launcher(
            slots=slots,
            behaviors={0: "sleep_on_advance"},
            response_timeout_seconds=0.25,
        )
        task_id = self.tasks[0].task_id
        started = __import__("time").monotonic()
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "timed out"):
            launcher.advance_frontier(
                selected_task_ids=(task_id,),
                target_tick=5_000,
                expected_states={task_id: self._pending_state(task_id)},
            )
        self.assertLess(__import__("time").monotonic() - started, 5.0)
        self.assertTrue(launcher.poisoned)
        self.assertEqual(
            len(tuple(self.campaign_root.glob("*.preserved"))),
            1,
        )

    def test_partial_frontier_and_source_drift_fail_closed(self) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=1,
            host_identity=self.host_identity,
        )
        task_id = self.tasks[0].task_id
        partial = self._launcher(
            slots=slots,
            behaviors={0: "partial_frontier"},
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "partial frontier"):
            partial.advance_frontier(
                selected_task_ids=(task_id,),
                target_tick=5_000,
                expected_states={task_id: self._pending_state(task_id)},
            )
        self.assertTrue(partial.poisoned)

        drift = self._launcher(
            slots=slots,
            behaviors={0: "source_mismatch_after"},
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "source"):
            drift.advance_frontier(
                selected_task_ids=(task_id,),
                target_tick=5_000,
                expected_states={task_id: self._pending_state(task_id)},
            )
        self.assertTrue(drift.poisoned)

    def test_reconciliation_binds_original_authorities_and_rejects_forgery(
        self,
    ) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=1,
            host_identity=self.host_identity,
        )
        selected = tuple(task.task_id for task in self.tasks[:3])
        authorities = {
            task_id: f"{index + 1:x}" * 64 for index, task_id in enumerate(selected)
        }
        launcher = self._launcher(
            slots=slots,
            behaviors={0: "reconcile_mixed"},
        )
        batch = launcher.reconcile_frontier(
            selected_task_ids=selected,
            target_tick=5_000,
            expected_states=self._pending_states(selected),
            intent_sha256="f" * 64,
            attempt_authority_sha256_by_task=authorities,
        )
        self.assertEqual(batch.intent_sha256, "f" * 64)
        self.assertEqual(
            dict(batch.attempt_authority_sha256_by_task),
            authorities,
        )
        self.assertEqual(
            tuple(batch.disposition_by_task.values()),
            (
                "already_committed_target",
                "advanced_from_exact_predecessor",
                "recovered_original_precurrent_attempt",
            ),
        )
        self.assertEqual(
            [batch.results_by_task[task_id].observed_tick for task_id in selected],
            [5_000, 5_000, 5_000],
        )
        launcher.close()

        forged = self._launcher(
            slots=slots,
            behaviors={0: "reconcile_forged"},
        )
        with self.assertRaisesRegex(
            OpenEcologyProcessWorkerError,
            "digest mismatch|operational result drifted",
        ):
            forged.reconcile_frontier(
                selected_task_ids=selected,
                target_tick=5_000,
                expected_states=self._pending_states(selected),
                intent_sha256="f" * 64,
                attempt_authority_sha256_by_task=authorities,
            )
        self.assertTrue(forged.poisoned)

    def test_startup_source_mismatch_and_hidden_oversubscription_are_rejected(
        self,
    ) -> None:
        slots = build_process_worker_slots(
            self.tasks,
            worker_count=1,
            host_identity=self.host_identity,
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "source"):
            self._launcher(
                slots=slots,
                behaviors={0: "source_mismatch_startup"},
            )

        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "CPU-only"):
            ProcessWorkerSlot(
                worker_index=0,
                host_identity=self.host_identity,
                device_kind="cuda",
                device_index=0,
                torch_threads=1,
                task_ids=tuple(task.task_id for task in self.tasks),
            )

        wrong_host_slots = (
            ProcessWorkerSlot(
                worker_index=0,
                host_identity="different-host",
                device_kind="cpu",
                device_index=None,
                torch_threads=1,
                task_ids=tuple(task.task_id for task in self.tasks),
            ),
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "current host"):
            self._launcher(slots=wrong_host_slots)

        too_many_threads = (
            ProcessWorkerSlot(
                worker_index=0,
                host_identity=self.host_identity,
                device_kind="cpu",
                device_index=None,
                torch_threads=(os.cpu_count() or 1) + 1,
                task_ids=tuple(task.task_id for task in self.tasks),
            ),
        )
        with self.assertRaisesRegex(OpenEcologyProcessWorkerError, "oversubscribe"):
            self._launcher(slots=too_many_threads)

    def test_source_probe_uses_absolute_pinned_git_and_rejects_replacement(
        self,
    ) -> None:
        git_executable = self.base / "git"
        git_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        git_executable.chmod(0o700)
        git_sha256 = hashlib.sha256(git_executable.read_bytes()).hexdigest()
        calls = 0

        def successful_git(command, *, environment):
            nonlocal calls
            calls += 1
            self.assertEqual(command[0], str(git_executable))
            self.assertNotEqual(environment["PATH"], "/hostile")
            stdout = (_SOURCE_COMMIT + "\n").encode() if calls == 1 else b""
            return stdout, b"", 0

        with (
            mock.patch.dict(os.environ, {"PATH": "/hostile"}),
            mock.patch(
                "evolution_sim.mind.open_ecology_process_workers."
                "source_file_hash_manifest",
                return_value={"aggregate_sha256": _SOURCE_MANIFEST},
            ),
            mock.patch(
                "evolution_sim.mind.open_ecology_process_workers._run_bounded_git",
                side_effect=successful_git,
            ),
        ):
            snapshot = strict_open_ecology_source_snapshot(
                repository_root=self.repository_root,
                source_git_sha=_SOURCE_COMMIT,
                source_manifest_sha256=_SOURCE_MANIFEST,
                git_executable=git_executable,
                git_executable_sha256=git_sha256,
            )
        self.assertTrue(snapshot["git_clean"])
        self.assertEqual(calls, 2)

        replacement = self.base / "replacement-git"
        replacement.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        replacement.chmod(0o700)

        def replace_during_git(command, *, environment):
            del command, environment
            os.replace(replacement, git_executable)
            return (_SOURCE_COMMIT + "\n").encode(), b"", 0

        with (
            mock.patch(
                "evolution_sim.mind.open_ecology_process_workers."
                "source_file_hash_manifest",
                return_value={"aggregate_sha256": _SOURCE_MANIFEST},
            ),
            mock.patch(
                "evolution_sim.mind.open_ecology_process_workers._run_bounded_git",
                side_effect=replace_during_git,
            ),
            self.assertRaisesRegex(
                OpenEcologyProcessWorkerError,
                "identity|SHA256",
            ),
        ):
            strict_open_ecology_source_snapshot(
                repository_root=self.repository_root,
                source_git_sha=_SOURCE_COMMIT,
                source_manifest_sha256=_SOURCE_MANIFEST,
                git_executable=git_executable,
                git_executable_sha256=git_sha256,
            )

    def test_git_source_probe_output_is_streaming_bounded(self) -> None:
        noisy = self.base / "noisy-git"
        noisy.write_text(
            "#!/bin/sh\nprintf '0123456789abcdef0123456789abcdef'\n",
            encoding="utf-8",
        )
        noisy.chmod(0o700)
        with (
            mock.patch(
                "evolution_sim.mind.open_ecology_process_workers."
                "OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES",
                16,
            ),
            self.assertRaisesRegex(
                OpenEcologyProcessWorkerError,
                "output exceeded",
            ),
        ):
            _run_bounded_git(
                (str(noisy),),
                environment={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": "/usr/bin:/bin",
                },
            )

    def test_git_breach_kills_descendant_after_leader_exit(self) -> None:
        child_pid_path = self.base / "git-child.pid"
        child_body = (
            "import os,signal,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "while True:\n"
            " os.write(1,b'x'*4096)\n"
            " time.sleep(0.001)\n"
        )
        parent_code = (
            "import pathlib,subprocess,sys\n"
            f"child=subprocess.Popen([sys.executable,'-c',{child_body!r}])\n"
            f"pathlib.Path({str(child_pid_path)!r}).write_text("
            "str(child.pid),encoding='ascii')\n"
        )
        with (
            mock.patch(
                "evolution_sim.mind.open_ecology_process_workers."
                "OPEN_ECOLOGY_GIT_OUTPUT_MAX_BYTES",
                8192,
            ),
            self.assertRaisesRegex(
                OpenEcologyProcessWorkerError,
                "output exceeded",
            ),
        ):
            _run_bounded_git(
                (sys.executable, "-c", parent_code),
                environment={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": "/usr/bin:/bin",
                },
            )
        _assert_process_gone(self, child_pid_path)

    def test_unkillable_worker_writes_fatal_marker_and_blocks_restart(self) -> None:
        class _UnkillableProcess:
            pid = 424_242

            def __init__(self) -> None:
                self.terminate_called = False
                self.kill_called = False

            def is_alive(self) -> bool:
                return True

            def terminate(self) -> None:
                self.terminate_called = True

            def kill(self) -> None:
                self.kill_called = True

            def join(self, *, timeout: float) -> None:
                self.join_timeout = timeout

        process = _UnkillableProcess()
        launcher = object.__new__(PersistentProcessWorkerLauncher)
        launcher._processes = {0: process}
        launcher.shutdown_timeout_seconds = 0.01
        launcher.campaign_root = self.campaign_root
        launcher.campaign_id = "process-worker-test"
        launcher.source_git_sha = _SOURCE_COMMIT
        launcher.source_manifest_sha256 = _SOURCE_MANIFEST

        with self.assertRaisesRegex(
            OpenEcologyProcessWorkerError,
            "survived terminate and kill",
        ):
            launcher._join_or_terminate_all()

        self.assertTrue(process.terminate_called)
        self.assertTrue(process.kill_called)
        marker = self.campaign_root / OPEN_ECOLOGY_PROCESS_WORKER_FATAL_MARKER_NAME
        self.assertTrue(marker.is_file())
        self.assertEqual(marker.stat().st_mode & 0o777, 0o400)
        with self.assertRaisesRegex(
            OpenEcologyProcessWorkerError,
            "fatal worker PID marker",
        ):
            self._launcher(
                slots=build_process_worker_slots(
                    self.tasks,
                    worker_count=1,
                    host_identity=self.host_identity,
                )
            )

    def _launcher(
        self,
        *,
        slots,
        behaviors=None,
        response_timeout_seconds: float = 10.0,
    ) -> PersistentProcessWorkerLauncher:
        return PersistentProcessWorkerLauncher(
            tasks=self.tasks,
            campaign_root=self.campaign_root,
            campaign_id="process-worker-test",
            repository_root=self.repository_root,
            source_git_sha=_SOURCE_COMMIT,
            source_manifest_sha256=_SOURCE_MANIFEST,
            slots=slots,
            assignment_sha256=process_worker_assignment_sha256(
                self.tasks,
                slots,
            ),
            response_timeout_seconds=response_timeout_seconds,
            startup_timeout_seconds=15.0,
            shutdown_timeout_seconds=5.0,
            _test_behaviors={} if behaviors is None else behaviors,
        )

    def _pending_state(self, task_id: str) -> ProcessWorkerExpectedState:
        return ProcessWorkerExpectedState(
            task_id=task_id,
            observed_tick=0,
            terminal_extinct=False,
            aggregate_resume_pins=None,
        )

    def _pending_states(
        self,
        task_ids,
    ) -> dict[str, ProcessWorkerExpectedState]:
        return {task_id: self._pending_state(task_id) for task_id in task_ids}

    def _states_from_batch(self, batch):
        return {
            task_id: ProcessWorkerExpectedState(
                task_id=task_id,
                observed_tick=result.observed_tick,
                terminal_extinct=result.extinct,
                aggregate_resume_pins=result.aggregate_resume_pins,
            )
            for task_id, result in batch.results_by_task.items()
        }

    def _bindings(self) -> tuple[PersistentArtifactBinding, ...]:
        return tuple(
            PersistentArtifactBinding(
                learner_index=index,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][index],
                artifact_path=str(self.base / f"artifact-{index}.json"),
                artifact_sha256=f"{index + 1:x}" * 64,
                artifact_file_sha256=f"{index + 5:x}" * 64,
                source_commit=_SOURCE_COMMIT,
                terminal_authority_sha256=f"{index + 9:x}" * 64,
            )
            for index in range(4)
        )


def _assert_process_gone(
    case: unittest.TestCase,
    child_pid_path: Path,
) -> None:
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 3.0
    state = ""
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ("ps", "-p", str(child_pid), "-o", "stat="),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        state = completed.stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    case.assertTrue(
        not state or state.startswith("Z"),
        f"Git descendant survived cleanup: {state}",
    )


if __name__ == "__main__":
    unittest.main()
