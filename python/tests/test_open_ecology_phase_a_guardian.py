from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from evolution_sim.cli import open_ecology_phase_a_guardian as guardian_cli
from evolution_sim.io import open_ecology_runtime_venv_authority as runtime_authority
from evolution_sim.mind import open_ecology_phase_a_guardian as guardian


_SHA = "a" * 64
_GIT_SHA = "b" * 40


def _bindings(**changes: str) -> guardian.PhaseAAuthorityBindings:
    values = {
        "source_git_sha": _GIT_SHA,
        "source_manifest_sha256": "c" * 64,
        "preregistration_digest": "d" * 64,
        "evidence_index_digest": "e" * 64,
        "launch_authorization_digest": "f" * 64,
        "runtime_venv_authority_sha256": "1" * 64,
        "ssh_target": "trainer-node",
    }
    values.update(changes)
    return guardian.PhaseAAuthorityBindings(**values)


def _preregistration() -> dict[str, object]:
    bindings = _bindings()
    return {
        "exact_digest": bindings.preregistration_digest,
        "source": {
            "commit": bindings.source_git_sha,
            "manifest_sha256": bindings.source_manifest_sha256,
        },
    }


class OpenEcologyPhaseATwoPartyGuardianTests(unittest.TestCase):
    def test_public_serve_revalidates_runtime_before_process_or_live_gates(
        self,
    ) -> None:
        arguments = [
            "serve",
            "--preregistration",
            "/authority/preregistration.json",
            "--launch-authorization",
            "/authority/launch.json",
            "--output-root",
            "/runs",
            "--source-git-sha",
            "b" * 40,
            "--source-manifest-sha256",
            "c" * 64,
            "--preregistration-digest",
            "d" * 64,
            "--evidence-index-digest",
            "e" * 64,
            "--launch-authorization-digest",
            "f" * 64,
            "--runtime-venv-authority",
            "/authority/runtime.json",
            "--runtime-venv-authority-sha256",
            "1" * 64,
            "--archive-authority-sha256",
            "2" * 64,
            "--ssh-connection-sha256",
            "3" * 64,
            "--git-executable",
            "/usr/bin/git",
            "--git-executable-sha256",
            "4" * 64,
            "--ssh-target",
            "trainer-node",
        ]
        with mock.patch.object(
            runtime_authority,
            "revalidate_running_guardian_runtime",
            side_effect=runtime_authority.RuntimeVenvAuthorityError(
                "direct serve rejected"
            ),
        ) as revalidate:
            with self.assertRaisesRegex(
                runtime_authority.RuntimeVenvAuthorityError,
                "direct serve rejected",
            ):
                guardian_cli.main(arguments)
        revalidate.assert_called_once()

    def test_remote_gates_run_exactly_once_and_issue_nonserializable_capability(
        self,
    ) -> None:
        calls: list[str] = []
        source_calls: list[str] = []
        termination: list[str] = []
        channel = guardian.GuardianChannelLiveness(terminate=termination.append)
        verifiers = {
            name: (
                lambda _path, _preregistration, _time, name=name: (
                    calls.append(name) or {"gate": name}
                )
            )
            for name in ("d10", "throughput", "output_lock")
        }
        authority = guardian.RemoteGuardianAuthority(
            bindings=_bindings(),
            channel=channel,
            source_probe=lambda: source_calls.append("source"),
            verifiers=verifiers,
        )
        token = bytearray(b"g" * 32)

        capability = authority.activate(
            report_paths={
                name: Path(f"/tmp/{name}.json")
                for name in ("d10", "throughput", "output_lock")
            },
            preregistration=_preregistration(),
            authorization_time=None,
            github_token=token,
        )

        self.assertEqual(calls, ["d10", "throughput", "output_lock"])
        self.assertEqual(source_calls, ["source", "source"])
        self.assertEqual(token, bytearray(32))
        with self.assertRaisesRegex(TypeError, "cannot be serialized"):
            pickle.dumps(capability)
        capability.begin_cell(cell_id="A0", learner_index=0)
        capability.admit(
            preregistration=_preregistration(),
            launch_authorization_digest=_bindings().launch_authorization_digest,
            cell_id="A0",
            learner_index=0,
            stage="before_update",
            update_index=0,
        )
        capability.finish_cell(cell_id="A0", learner_index=0)
        capability.close()
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "closed or replayed",
        ):
            capability.begin_cell(cell_id="A0", learner_index=0)
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "exactly once",
        ):
            authority.activate(
                report_paths={
                    name: Path(f"/tmp/{name}.json")
                    for name in ("d10", "throughput", "output_lock")
                },
                preregistration=_preregistration(),
                authorization_time=None,
                github_token=bytearray(b"h" * 32),
            )
        self.assertEqual(calls, ["d10", "throughput", "output_lock"])
        self.assertEqual(termination, [])

    def test_forged_persisted_capability_and_missing_storage_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            Exception,
            "persisted launch JSON alone",
        ):
            guardian.require_live_phase_a_update_admission(
                {"passed": True, "exact_digest": _SHA},
                preregistration=_preregistration(),
                launch_authorization_digest=(_bindings().launch_authorization_digest),
                cell_id="A0",
                learner_index=0,
                stage="before_update",
                update_index=0,
            )
        storage_calls: list[str] = []
        storage = guardian.MacStorageAuthority(
            bindings=_bindings(),
            source_probe=lambda: storage_calls.append("source"),
            verifier=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("storage offline")
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "storage offline"):
            storage.acquire(
                report_path=Path("/tmp/storage.json"),
                preregistration=_preregistration(),
                authorization_time=None,
            )
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "has not been acquired",
        ):
            storage.secret_bytes()
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "exactly once",
        ):
            storage.acquire(
                report_path=Path("/tmp/storage.json"),
                preregistration=_preregistration(),
                authorization_time=None,
            )
        self.assertEqual(storage_calls, ["source"])

    def test_channel_eof_and_parent_loss_share_fatal_process_group_path(self) -> None:
        reasons: list[str] = []
        eof = guardian.GuardianChannelLiveness(terminate=reasons.append)
        eof.fail("ssh_channel_eof")
        eof.fail("second")
        self.assertEqual(reasons, ["ssh_channel_eof"])
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "ssh_channel_eof",
        ):
            eof.require_alive()

        parent = guardian.GuardianChannelLiveness(terminate=reasons.append)
        parent.fail("ssh_parent_lost")
        self.assertEqual(reasons, ["ssh_channel_eof", "ssh_parent_lost"])

    def test_background_thread_termination_really_kills_its_process_group(
        self,
    ) -> None:
        probe = (
            "import threading,time\n"
            "from evolution_sim.mind.open_ecology_phase_a_guardian "
            "import _terminate_own_process_group\n"
            "threading.Thread("
            "target=lambda: _terminate_own_process_group('probe')"
            ").start()\n"
            "time.sleep(10)\n"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": "python",
            }
        )
        result = subprocess.run(
            (sys.executable, "-c", probe),
            cwd=Path.cwd(),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=10.0,
        )
        self.assertIn(
            result.returncode,
            {-signal.SIGKILL, 128 + signal.SIGKILL},
        )

    def test_hmac_frame_binds_full_hello_and_rejects_replay(self) -> None:
        secret = b"s" * 32
        stream = io.BytesIO()
        hello = {
            "bindings": _bindings().as_dict(),
            "frame_type": "hello",
            "github_token": "not-a-real-token",
            "mac_storage_facts_digest": _SHA,
            "mac_storage_ready": True,
            "remote_challenge": __import__("base64").b64encode(b"r" * 32).decode(),
            "schema_version": guardian.TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
            "sequence": 0,
            "session_secret": "not-a-real-secret",
        }
        guardian._write_authenticated_frame(stream, hello, secret=secret)
        payload = stream.getvalue()
        frame = guardian._strict_json_mapping(payload[:-1])
        guardian._verify_frame_authentication(
            frame,
            secret=secret,
            expected_sequence=0,
        )
        self.assertEqual(frame["session_secret"], "not-a-real-secret")

        replay = guardian._strict_json_mapping(payload[:-1])
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "replayed or reordered",
        ):
            guardian._verify_frame_authentication(
                replay,
                secret=secret,
                expected_sequence=1,
            )

    def test_remote_rejects_a_replayed_hello_with_the_wrong_challenge(
        self,
    ) -> None:
        client_socket, server_socket = socket.socketpair()
        self.addCleanup(client_socket.close)
        self.addCleanup(server_socket.close)
        client_read = client_socket.makefile("rb")
        client_write = client_socket.makefile("wb")
        server_read = server_socket.makefile("rb")
        server_write = server_socket.makefile("wb")
        self.addCleanup(client_read.close)
        self.addCleanup(client_write.close)
        self.addCleanup(server_read.close)
        self.addCleanup(server_write.close)
        bindings = _bindings()
        channel = guardian.GuardianChannelLiveness(terminate=lambda _reason: None)
        errors: list[BaseException] = []

        def serve() -> None:
            try:
                guardian.run_remote_guardian_session(
                    stdin=server_read,
                    stdout=server_write,
                    preregistration_path=Path("/remote/prereg.json"),
                    launch_authorization_path=Path("/remote/auth.json"),
                    output_root=Path("/remote/output"),
                    expected_bindings=bindings,
                    channel=channel,
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve)
        thread.start()
        challenge = guardian._read_guardian_challenge_with_timeout(
            client_read,
            expected_bindings=bindings,
            timeout_seconds=5.0,
        )
        self.assertNotEqual(
            challenge["remote_challenge"],
            __import__("base64").b64encode(b"x" * 32).decode(),
        )
        secret = b"s" * 32
        guardian._write_authenticated_frame(
            client_write,
            {
                "bindings": bindings.as_dict(),
                "coordinator_pgid": os.getpgrp(),
                "coordinator_pid": os.getpid(),
                "frame_type": "hello",
                "github_token": __import__("base64").b64encode(b"t" * 32).decode(),
                "mac_storage_facts_digest": _SHA,
                "mac_storage_ready": True,
                "remote_challenge": __import__("base64").b64encode(b"x" * 32).decode(),
                "schema_version": guardian.TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                "sequence": 0,
                "session_secret": __import__("base64").b64encode(secret).decode(),
            },
            secret=secret,
        )
        thread.join(5.0)
        client_socket.shutdown(socket.SHUT_WR)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("fresh remote challenge", str(errors[0]))

    def test_hello_flush_scrubs_the_one_shot_credential_before_ready_wait(
        self,
    ) -> None:
        secret = b"s" * 32
        token = bytearray(b"one-shot-github-token-value")
        encoded_token = __import__("base64").b64encode(token).decode("ascii")
        hello = {
            "frame_type": "hello",
            "github_token": encoded_token,
            "schema_version": guardian.TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
            "sequence": 0,
        }
        stream = io.BytesIO()

        guardian._write_hello_and_scrub_token(
            stream,
            hello=hello,
            secret=secret,
            github_token=token,
        )

        self.assertEqual(token, bytearray(len(token)))
        self.assertNotIn("github_token", hello)
        transmitted = guardian._strict_json_mapping(stream.getvalue()[:-1])
        self.assertEqual(transmitted["github_token"], encoded_token)

    def test_stderr_overflow_invokes_the_fatal_process_callback(self) -> None:
        errors: list[BaseException] = []
        with mock.patch.object(guardian, "MAX_SSH_STDERR_BYTES", 8):
            capture = guardian._BoundedStderrCapture(
                io.BytesIO(b"x" * 9),
                on_error=errors.append,
            )
            capture.start()
            capture.join(1.0)
        self.assertEqual(len(errors), 1)
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "capture failed",
        ):
            capture.snapshot()

    def test_short_stderr_write_is_visible_while_pipe_remains_open(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        stream = os.fdopen(read_descriptor, "rb")
        writer = os.fdopen(write_descriptor, "wb", buffering=0)
        self.addCleanup(stream.close)
        self.addCleanup(writer.close)
        errors: list[BaseException] = []
        capture = guardian._BoundedStderrCapture(
            stream,
            on_error=errors.append,
        )
        capture.start()
        writer.write(b"short authenticated endpoint line\n")
        deadline = time.monotonic() + 1.0
        while (
            capture.snapshot() != b"short authenticated endpoint line\n"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.assertEqual(
            capture.snapshot(),
            b"short authenticated endpoint line\n",
        )
        self.assertEqual(errors, [])
        writer.close()
        capture.join(1.0)

    def test_partial_protocol_frame_cannot_defeat_the_deadline(self) -> None:
        sender, receiver = socket.socketpair()
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        stream = receiver.makefile("rb")
        self.addCleanup(stream.close)
        sender.sendall(b"{")
        started = time.monotonic()
        with self.assertRaisesRegex(
            guardian.OpenEcologyTwoPartyAuthorityError,
            "timed out",
        ):
            guardian._read_authenticated_frame_with_timeout(
                stream,
                secret=b"s" * 32,
                expected_sequence=0,
                timeout_seconds=0.05,
            )
        self.assertLess(time.monotonic() - started, 0.5)

    def test_endpoint_is_authenticated_before_the_mac_live_storage_gate(
        self,
    ) -> None:
        events: list[str] = []
        bindings = _bindings()
        preregistration = _preregistration()

        class Storage:
            facts_digest = None

            def acquire(self, **_kwargs: object) -> None:
                events.append("storage")
                raise RuntimeError("stop after ordering proof")

            def close(self) -> None:
                events.append("storage_close")

        class Process:
            stdin = io.BytesIO()
            stdout = io.BytesIO()
            stderr = io.BytesIO()
            pid = 999_999

            def poll(self) -> int:
                return 0

        token = bytearray(b"g" * 32)
        with (
            mock.patch.object(
                guardian,
                "load_archive_tool_authority",
                return_value=object(),
            ),
            mock.patch.object(
                guardian,
                "_bindings_from_archive_and_bundle",
                return_value=bindings,
            ),
            mock.patch.object(
                guardian,
                "load_phase_a_authority_bundle",
                return_value=(
                    preregistration,
                    {},
                    datetime.now(timezone.utc),
                    bindings,
                    {
                        name: Path(f"/tmp/{name}.json")
                        for name in (
                            "d10",
                            "throughput",
                            "storage",
                            "output_lock",
                        )
                    },
                ),
            ),
            mock.patch.object(
                guardian,
                "verify_local_authority_files",
                side_effect=lambda _authority: events.append("archive"),
            ),
            mock.patch.object(
                guardian,
                "_remote_guardian_ssh_command",
                side_effect=lambda **_kwargs: events.append("command") or ["ssh"],
            ),
            mock.patch.object(
                guardian,
                "_wait_for_authenticated_endpoint",
                side_effect=lambda *_args, **_kwargs: (
                    events.append("endpoint") or object()
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "ordering proof"),
        ):
            guardian.run_mac_coordinator(
                local_preregistration_path=Path("/local/prereg.json"),
                local_launch_authorization_path=Path("/local/auth.json"),
                remote_preregistration_path=Path("/remote/prereg.json"),
                remote_launch_authorization_path=Path("/remote/auth.json"),
                remote_output_root=Path("/remote/output"),
                archive_authority_path=Path("/local/archive.json"),
                expected_archive_authority_sha256="a" * 64,
                remote_runtime_venv_authority_path=Path("/remote/runtime.json"),
                expected_remote_runtime_venv_authority_sha256="1" * 64,
                transcript_path=Path("/local/transcript.json"),
                github_token=token,
                popen=lambda *_args, **_kwargs: events.append("popen") or Process(),
                storage_authority_factory=lambda **_kwargs: Storage(),
            )
        self.assertLess(events.index("endpoint"), events.index("storage"))
        self.assertEqual(Process.stdin.getvalue(), b"")
        self.assertEqual(token, bytearray(len(token)))

    def test_compact_transcript_is_non_authorizing_and_single_framed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "guardian-transcript.json"
            transcript = guardian._write_compact_transcript(
                path,
                bindings=_bindings(),
                endpoint={"host": "gpu.example", "port": 22},
                coordinator_pid=101,
                coordinator_pgid=100,
                remote_pid=201,
                remote_pgid=200,
                completed=[],
                storage_facts_digest="2" * 64,
                remote_verifier_digests={
                    name: "3" * 64 for name in ("d10", "throughput", "output_lock")
                },
                complete=False,
            )
            payload = path.read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            self.assertFalse(payload.endswith(b"\n\n"))
            self.assertFalse(
                transcript["authority_semantics"]["offline_json_authorizes_updates"]
            )
            self.assertFalse(transcript["authority_semantics"]["selection_authorized"])
            self.assertFalse(transcript["transcript_is_resume_or_launch_authority"])
            self.assertEqual(
                transcript["bindings"]["runtime_venv_authority_sha256"],
                _bindings().runtime_venv_authority_sha256,
            )

    def test_pinned_descriptor_bootstrap_rejects_tampered_bytes_before_exec(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "executed"
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                (
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    guardian._PINNED_STDLIB_EXEC_CODE,
                    str(bootstrap),
                    "0" * 64,
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA256 drifted", result.stderr)
            self.assertFalse(marker.exists())

    def test_d10_is_resolved_only_from_dependency_ten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preregistration_path = root / "preregistration.json"
            authorization_path = root / "authorization.json"
            index_path = root / "index.json"
            preregistration_path.write_text(
                json.dumps(
                    {
                        "exact_digest": "d" * 64,
                        "source": {
                            "commit": "b" * 40,
                            "manifest_sha256": "c" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            authorization_path.write_text(
                json.dumps(
                    {
                        "authorized_at_utc": "2026-07-28T00:00:00Z",
                        "evidence_index": {
                            "exact_digest": "e" * 64,
                            "file": {"relative_path": "index.json"},
                        },
                        "exact_digest": "f" * 64,
                    }
                ),
                encoding="utf-8",
            )
            for name in ("d10", "throughput", "storage", "output-lock"):
                (root / f"{name}.json").write_text("{}", encoding="utf-8")

            def index(dependency_id: str) -> dict[str, object]:
                return {
                    "exact_digest": "e" * 64,
                    "dependency_reports": [
                        {
                            "dependency_id": dependency_id,
                            "reports": [
                                {
                                    "evidence_kind": (
                                        "exact_sha_phase_a_training_and_torch_ci"
                                    ),
                                    "file": {"relative_path": "d10.json"},
                                }
                            ],
                        }
                    ],
                    "operational_reports": [
                        {
                            "report": {
                                "evidence_kind": kind,
                                "file": {"relative_path": f"{name}.json"},
                            }
                        }
                        for kind, name in (
                            ("phase_a_training_throughput", "throughput"),
                            ("campaign_storage_capacity", "storage"),
                            ("output_lock_contention", "output-lock"),
                        )
                    ],
                }

            with (
                mock.patch.object(
                    guardian,
                    "validate_open_ecology_phase_a_preregistration",
                ),
                mock.patch.object(
                    guardian,
                    "validate_open_ecology_phase_a_launch_authorization",
                ),
            ):
                index_path.write_text(
                    json.dumps(index("readiness_dependency_10")),
                    encoding="utf-8",
                )
                bundle = guardian.load_phase_a_authority_bundle(
                    preregistration_path=preregistration_path,
                    launch_authorization_path=authorization_path,
                )
                self.assertEqual(
                    bundle[-1]["d10"],
                    (root / "d10.json").resolve(),
                )

                index_path.write_text(
                    json.dumps(index("readiness_dependency_09")),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    guardian.OpenEcologyTwoPartyAuthorityError,
                    "dependency-10 D10 report",
                ):
                    guardian.load_phase_a_authority_bundle(
                        preregistration_path=preregistration_path,
                        launch_authorization_path=authorization_path,
                    )

    def test_end_to_end_guardian_pipe_waits_for_both_authorities_and_runs_ordered(
        self,
    ) -> None:
        client_socket, server_socket = socket.socketpair()
        self.addCleanup(client_socket.close)
        self.addCleanup(server_socket.close)
        client_read = client_socket.makefile("rb")
        client_write = client_socket.makefile("wb")
        server_read = server_socket.makefile("rb")
        server_write = server_socket.makefile("wb")
        self.addCleanup(client_read.close)
        self.addCleanup(client_write.close)
        self.addCleanup(server_read.close)
        self.addCleanup(server_write.close)
        events: list[str] = []
        termination: list[str] = []
        channel = guardian.GuardianChannelLiveness(terminate=termination.append)
        bindings = _bindings()
        preregistration = _preregistration()
        reports = {
            name: Path(f"/tmp/{name}.json")
            for name in ("d10", "throughput", "storage", "output_lock")
        }

        def factory(**kwargs: object) -> guardian.RemoteGuardianAuthority:
            return guardian.RemoteGuardianAuthority(
                bindings=bindings,
                channel=channel,
                source_probe=lambda: events.append("source"),
                verifiers={
                    name: (
                        lambda *_args, name=name: (
                            events.append(f"verify:{name}") or {"gate": name}
                        )
                    )
                    for name in ("d10", "throughput", "output_lock")
                },
            )

        def run_cell(
            _preregistration: object,
            **kwargs: object,
        ) -> dict[str, object]:
            cell = str(kwargs["cell_id"])
            learner = int(kwargs["learner_index"])
            events.append(f"run:{cell}:{learner}")
            guardian.require_live_phase_a_update_admission(
                kwargs["live_launch_capability"],
                preregistration=preregistration,
                launch_authorization_digest=(bindings.launch_authorization_digest),
                cell_id=cell,
                learner_index=learner,
                stage="before_update",
                update_index=0,
            )
            return {
                "run_id": f"{cell}-{learner}",
                "exact_digest": hashlib_for_test(f"{cell}-{learner}"),
            }

        error: list[BaseException] = []

        def serve() -> None:
            try:
                with mock.patch.object(
                    guardian,
                    "load_phase_a_authority_bundle",
                    return_value=(
                        preregistration,
                        {},
                        datetime.now(timezone.utc),
                        bindings,
                        reports,
                    ),
                ):
                    guardian.run_remote_guardian_session(
                        stdin=server_read,
                        stdout=server_write,
                        preregistration_path=Path("/remote/prereg.json"),
                        launch_authorization_path=Path("/remote/auth.json"),
                        output_root=Path("/remote/output"),
                        expected_bindings=bindings,
                        channel=channel,
                        run_cell=run_cell,
                        remote_authority_factory=factory,
                    )
            except BaseException as caught:
                error.append(caught)

        thread = threading.Thread(target=serve)
        thread.start()
        secret = b"q" * 32
        token = bytearray(b"t" * 32)
        challenge = guardian._read_guardian_challenge_with_timeout(
            client_read,
            expected_bindings=bindings,
            timeout_seconds=5.0,
        )
        guardian._write_authenticated_frame(
            client_write,
            {
                "bindings": bindings.as_dict(),
                "coordinator_pgid": os.getpgrp(),
                "coordinator_pid": os.getpid(),
                "frame_type": "hello",
                "github_token": __import__("base64").b64encode(token).decode(),
                "mac_storage_facts_digest": _SHA,
                "mac_storage_ready": True,
                "remote_challenge": challenge["remote_challenge"],
                "schema_version": guardian.TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                "sequence": 0,
                "session_secret": __import__("base64").b64encode(secret).decode(),
            },
            secret=secret,
        )
        try:
            ready = guardian._read_authenticated_frame_with_timeout(
                client_read,
                secret=secret,
                expected_sequence=0,
                timeout_seconds=5.0,
            )
        except BaseException:
            client_write.close()
            thread.join(1.0)
            if error:
                raise error[0]
            raise
        self.assertEqual(ready["frame_type"], "ready")
        self.assertEqual(ready["remote_pid"], challenge["remote_pid"])
        self.assertEqual(ready["remote_pgid"], challenge["remote_pgid"])
        self.assertEqual(
            events[:5],
            [
                "source",
                "verify:d10",
                "verify:throughput",
                "verify:output_lock",
                "source",
            ],
        )
        for sequence, (cell, learner) in enumerate(
            guardian._canonical_phase_a_matrix(),
            start=1,
        ):
            guardian._write_authenticated_frame(
                client_write,
                {
                    "cell_id": cell,
                    "frame_type": "run_cell",
                    "learner_index": learner,
                    "resume": True,
                    "schema_version": guardian.TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                    "sequence": sequence,
                },
                secret=secret,
            )
            response = guardian._read_authenticated_frame_with_timeout(
                client_read,
                secret=secret,
                expected_sequence=sequence,
                timeout_seconds=5.0,
            )
            self.assertEqual(response["frame_type"], "cell_complete")
        stop_sequence = 17
        guardian._write_authenticated_frame(
            client_write,
            {
                "frame_type": "stop",
                "schema_version": guardian.TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                "sequence": stop_sequence,
            },
            secret=secret,
        )
        stopped = guardian._read_authenticated_frame_with_timeout(
            client_read,
            secret=secret,
            expected_sequence=stop_sequence,
            timeout_seconds=5.0,
        )
        self.assertEqual(stopped["frame_type"], "stopped")
        # socket.makefile().close() does not deliver EOF while another
        # makefile still references the same socket. Deliver a real half-close
        # so the guardian reader can leave its blocking read before cleanup.
        client_socket.shutdown(socket.SHUT_WR)
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(error, [])
        self.assertEqual(termination, [])
        runs = [event for event in events if event.startswith("run:")]
        self.assertEqual(
            runs,
            [
                f"run:{cell}:{learner}"
                for cell, learner in guardian._canonical_phase_a_matrix()
            ],
        )

    def test_wrong_endpoint_binding_is_rejected_before_live_gates(self) -> None:
        expected = _bindings()
        observed = guardian._bindings_from_mapping(
            _bindings(ssh_target="wrong-host").as_dict()
        )
        self.assertNotEqual(observed, expected)
        calls: list[str] = []
        authority = guardian.RemoteGuardianAuthority(
            bindings=expected,
            channel=guardian.GuardianChannelLiveness(terminate=lambda _reason: None),
            source_probe=lambda: None,
            verifiers={
                name: (lambda *_args, name=name: calls.append(name) or {"gate": name})
                for name in ("d10", "throughput", "output_lock")
            },
        )
        self.assertEqual(calls, [])
        del authority


def hashlib_for_test(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    unittest.main()
