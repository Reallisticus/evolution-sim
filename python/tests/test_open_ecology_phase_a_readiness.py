from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import evolution_sim.cli.open_ecology_phase_a as phase_a_cli
import evolution_sim.io.open_ecology_archive_authority as archive_authority
import evolution_sim.mind.open_ecology_phase_a as phase_a
import evolution_sim.mind.open_ecology_phase_a_qualification as qualification
import evolution_sim.mind.open_ecology_phase_a_readiness as readiness
from evolution_sim.mind import open_ecology_selection
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_open_ecology_phase_a import _campaign


REQUIRES_MIND_ML = True

_PRODUCED_AT = "2026-07-27T10:00:00Z"
_AUTHORIZED_AT = "2026-07-27T11:00:00Z"
_SHA = "c" * 64
_D4_BUCKET_MATRIX_ACTIVE_ROWS = (
    1,
    2,
    3,
    5,
    9,
    17,
    33,
    64,
    65,
    129,
    257,
    319,
    320,
)
_D4_BUCKET_MATRIX_EXECUTION_ROWS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    64,
    128,
    256,
    320,
    320,
    320,
)
_D4_BUCKET_MATRIX_CASES = [
    {
        "active_rows": active_rows,
        "execution_rows": execution_rows,
        "observed_recurrent_input_shape": [1, execution_rows, 256],
    }
    for active_rows, execution_rows in zip(
        _D4_BUCKET_MATRIX_ACTIVE_ROWS,
        _D4_BUCKET_MATRIX_EXECUTION_ROWS,
        strict=True,
    )
]
_D4_BUCKET_MATRIX_COMPARISON_COUNT = sum(_D4_BUCKET_MATRIX_ACTIVE_ROWS)


def _reconstruct_report(
    report_path: Path,
    _preregistration: dict[str, object],
    _authorization_time: object,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "evidence_kind": report["evidence_kind"],
        "campaign_digest": report["campaign_digest"],
        "configuration_sha256": report["configuration_sha256"],
        "source": report["source"],
        "facts": report["facts"],
    }


_ALL_AUTHORITY_VERIFIERS = {
    kind: _reconstruct_report for kind in readiness.PROOF_PRODUCER_AVAILABILITY
}


class OpenEcologyPhaseAReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = _campaign()

    def test_readiness_reports_available_authority_but_no_supplied_evidence(
        self,
    ) -> None:
        report = phase_a.open_ecology_phase_a_launch_readiness()
        self.assertIs(report["authorization_assembler_available"], True)
        self.assertIs(
            report["dependency_specific_semantic_validators_available"],
            True,
        )
        self.assertIs(
            report["dependency_specific_proof_producers_available"],
            True,
        )
        self.assertIs(
            report["proof_producer_availability"][
                "capture_noninterference_reexecution"
            ],
            True,
        )
        self.assertIs(
            report["proof_producer_availability"][
                "preregistration_roundtrip_fail_closed"
            ],
            True,
        )
        self.assertIs(
            report["authority_verifier_availability"][
                "preregistration_roundtrip_fail_closed"
            ],
            True,
        )
        self.assertIs(
            report["dependency_specific_authority_verifiers_available"],
            True,
        )
        self.assertIs(report["phase_a_training_authorized"], False)
        self.assertEqual(
            report["blockers"],
            ["launch_evidence_index_required"],
        )
        self.assertEqual(
            set(report["dependencies"]),
            set(phase_a.OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES),
        )

    def test_authority_json_publication_never_overwrites_existing_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "authority.json"
            phase_a_cli._write_atomic_json(output, {"generation": 1})
            original = output.read_bytes()
            self.assertEqual(output.stat().st_nlink, 1)
            with self.assertRaises(FileExistsError):
                phase_a_cli._write_atomic_json(output, {"generation": 2})
            self.assertEqual(output.read_bytes(), original)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                phase_a_cli._write_atomic_json(
                    linked_parent / "forbidden.json",
                    {"generation": 1},
                )

    def test_git_source_inspection_authority_errors_fail_closed(self) -> None:
        timeout = phase_a_cli.OpenEcologyGitAuthorityError(
            "bounded Git authority command timed out"
        )
        with (
            patch.object(
                phase_a_cli,
                "discover_pinned_git_executable",
                return_value=object(),
            ),
            patch.object(phase_a_cli, "run_pinned_git", side_effect=timeout),
            self.assertRaisesRegex(RuntimeError, "inspection failed closed"),
        ):
            phase_a_cli._git_source_state()

        timeout = phase_a.OpenEcologyGitAuthorityError(
            "bounded Git authority command timed out"
        )
        with (
            patch.object(
                phase_a,
                "discover_pinned_git_executable",
                return_value=object(),
            ),
            patch.object(phase_a, "run_pinned_git", side_effect=timeout),
            self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "failed to inspect Phase A Git source",
            ),
        ):
            phase_a._require_live_source(self.campaign)

    def test_preregister_main_translates_raw_git_authority_failure(self) -> None:
        arguments = [
            "preregister",
            "--source-commit",
            "a" * 40,
            "--heritable-benchmark",
            "/missing/heritable.json",
            "--zero-all-benchmark",
            "/missing/zero-all.json",
            "--resource-envelope",
            "/missing/resources.json",
            "--expected-archive-tool-authority-sha256",
            "b" * 64,
            "--output",
            "/missing/preregistration.json",
        ]
        with (
            patch.object(
                phase_a_cli,
                "discover_pinned_git_executable",
                return_value=object(),
            ),
            patch.object(
                phase_a_cli,
                "_git_source_state",
                side_effect=phase_a_cli.OpenEcologyGitAuthorityError(
                    "hostile Git authority failure"
                ),
            ),
            self.assertRaisesRegex(
                SystemExit,
                "Git source inspection failed closed",
            ),
        ):
            phase_a_cli.main(arguments)

    def test_resource_envelope_cli_is_reproducible(self) -> None:
        source_commit = "a" * 40
        source_manifest = {
            "aggregate_sha256": "b" * 64,
            "files": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = root / "first.json"
            second = root / "second.json"
            with (
                patch.object(
                    phase_a_cli,
                    "discover_pinned_git_executable",
                    return_value=object(),
                ),
                patch.object(
                    phase_a_cli,
                    "_git_source_state",
                    return_value=(source_commit, True),
                ),
                patch.object(
                    phase_a_cli,
                    "source_file_hash_manifest",
                    return_value=source_manifest,
                ),
                patch.object(
                    phase_a_cli,
                    "_preregistration_document_sha256",
                    return_value=(phase_a.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256),
                ),
            ):
                self.assertEqual(
                    phase_a_cli.main(
                        [
                            "build-resource-envelope",
                            "--source-commit",
                            source_commit,
                            "--output",
                            str(first),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    phase_a_cli.main(
                        [
                            "build-resource-envelope",
                            "--source-commit",
                            source_commit,
                            "--output",
                            str(second),
                        ]
                    ),
                    0,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8")),
                phase_a.build_open_ecology_phase_a_resource_envelope(
                    source_commit=source_commit,
                ),
            )

    def test_resource_envelope_cli_rejects_dirty_or_stale_source(self) -> None:
        source_commit = "a" * 40
        hostile_states = (
            (source_commit, False),
            ("b" * 40, True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for index, state in enumerate(hostile_states):
                with (
                    self.subTest(state=state),
                    patch.object(
                        phase_a_cli,
                        "discover_pinned_git_executable",
                        return_value=object(),
                    ),
                    patch.object(
                        phase_a_cli,
                        "_git_source_state",
                        return_value=state,
                    ),
                    self.assertRaisesRegex(
                        SystemExit,
                        "match a clean Git HEAD",
                    ),
                ):
                    phase_a_cli.main(
                        [
                            "build-resource-envelope",
                            "--source-commit",
                            source_commit,
                            "--output",
                            str(root / f"hostile-{index}.json"),
                        ]
                    )

    def test_resource_envelope_cli_rejects_tampered_protocol_document(self) -> None:
        source_commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "resource-envelope.json"
            with (
                patch.object(
                    phase_a_cli,
                    "discover_pinned_git_executable",
                    return_value=object(),
                ),
                patch.object(
                    phase_a_cli,
                    "_git_source_state",
                    return_value=(source_commit, True),
                ),
                patch.object(
                    phase_a_cli,
                    "source_file_hash_manifest",
                    return_value={"aggregate_sha256": "b" * 64},
                ),
                patch.object(
                    phase_a_cli,
                    "_preregistration_document_sha256",
                    return_value="c" * 64,
                ),
                self.assertRaisesRegex(
                    SystemExit,
                    "differs from the sealed preregistration document",
                ),
            ):
                phase_a_cli.main(
                    [
                        "build-resource-envelope",
                        "--source-commit",
                        source_commit,
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_resource_envelope_cli_rejects_source_change_during_build(self) -> None:
        source_commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "resource-envelope.json"
            with (
                patch.object(
                    phase_a_cli,
                    "discover_pinned_git_executable",
                    return_value=object(),
                ),
                patch.object(
                    phase_a_cli,
                    "_git_source_state",
                    side_effect=[
                        (source_commit, True),
                        ("b" * 40, True),
                    ],
                ),
                patch.object(
                    phase_a_cli,
                    "source_file_hash_manifest",
                    return_value={"aggregate_sha256": "c" * 64},
                ),
                patch.object(
                    phase_a_cli,
                    "_preregistration_document_sha256",
                    return_value=(phase_a.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256),
                ),
                self.assertRaisesRegex(
                    SystemExit,
                    "source changed while the resource envelope",
                ),
            ):
                phase_a_cli.main(
                    [
                        "build-resource-envelope",
                        "--source-commit",
                        source_commit,
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_preregister_rejects_noncanonical_resource_envelope_bytes(self) -> None:
        source_commit = "a" * 40
        envelope = phase_a.build_open_ecology_phase_a_resource_envelope(
            source_commit=source_commit,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            heritable = root / "heritable.json"
            zero_all = root / "zero-all.json"
            resources = root / "resources.json"
            output = root / "preregistration.json"
            heritable.write_text("{}\n", encoding="utf-8")
            zero_all.write_text("{}\n", encoding="utf-8")
            resources.write_text(
                json.dumps(envelope, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    phase_a_cli,
                    "discover_pinned_git_executable",
                    return_value=object(),
                ),
                patch.object(
                    phase_a_cli,
                    "_git_source_state",
                    return_value=(source_commit, True),
                ),
                patch.object(
                    phase_a_cli,
                    "source_file_hash_manifest",
                    return_value={"aggregate_sha256": "b" * 64},
                ),
                patch.object(
                    phase_a_cli,
                    "_preregistration_document_sha256",
                    return_value=(phase_a.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256),
                ),
                patch.object(
                    phase_a_cli,
                    "configure_open_ecology_phase_a_determinism",
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "byte-canonical resource envelope",
                ),
            ):
                phase_a_cli.main(
                    [
                        "preregister",
                        "--source-commit",
                        source_commit,
                        "--heritable-benchmark",
                        str(heritable),
                        "--zero-all-benchmark",
                        str(zero_all),
                        "--resource-envelope",
                        str(resources),
                        "--expected-archive-tool-authority-sha256",
                        "c" * 64,
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_d9_directory_publication_never_replaces_existing_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "pending"
            source.mkdir()
            (source / "new-authority").write_bytes(b"new")
            destination = root / "authority"
            destination.mkdir()
            (destination / "prior-authority").write_bytes(b"prior")

            with self.assertRaises(FileExistsError):
                readiness._rename_path_no_replace(source, destination)

            self.assertEqual(
                (destination / "prior-authority").read_bytes(),
                b"prior",
            )
            self.assertFalse((destination / "new-authority").exists())
            self.assertEqual((source / "new-authority").read_bytes(), b"new")

    def test_no_replace_publication_closes_parent_on_libc_load_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "pending"
            destination = root / "authority"
            source.write_bytes(b"pending")
            real_close = os.close
            closed: list[int] = []

            def close_and_record(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with (
                patch.object(
                    readiness.ctypes,
                    "CDLL",
                    side_effect=OSError("injected libc failure"),
                ),
                patch.object(
                    readiness.os,
                    "close",
                    side_effect=close_and_record,
                ),
                self.assertRaisesRegex(OSError, "injected libc failure"),
            ):
                readiness._rename_path_no_replace(source, destination)
            self.assertEqual(len(closed), 1)

    def test_generic_bundle_publish_preserves_staging_namespace_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            destination = root / "authority"
            observed_staging: list[Path] = []
            preserved_owned: list[Path] = []

            def hostile_producer(staging: Path) -> dict[str, object]:
                self.assertTrue(staging.is_dir())
                observed_staging.append(staging)
                owned = staging.with_name(f"{staging.name}.owned")
                os.rename(staging, owned)
                preserved_owned.append(owned)
                staging.mkdir(mode=0o700)
                (staging / "replacement-marker").write_bytes(b"replacement")
                return {}

            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "staging directory namespace changed",
            ):
                readiness._publish_new_authority_bundle(
                    self.campaign,
                    output_directory=destination,
                    field="test authority",
                    producer=hostile_producer,
                )

            self.assertEqual(len(observed_staging), 1)
            self.assertEqual(len(preserved_owned), 1)
            self.assertFalse(destination.exists())
            self.assertEqual(
                (observed_staging[0] / "replacement-marker").read_bytes(),
                b"replacement",
            )
            self.assertTrue(preserved_owned[0].is_dir())

    def test_d9_publish_preserves_staging_namespace_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            destination = root / "d9-authority"
            observed_staging: list[Path] = []
            preserved_owned: list[Path] = []

            def hostile_producer(
                _preregistration: dict[str, object],
                *,
                output_directory: str | Path,
                **_kwargs: object,
            ) -> dict[str, object]:
                staging = Path(output_directory)
                self.assertTrue(staging.is_dir())
                observed_staging.append(staging)
                owned = staging.with_name(f"{staging.name}.owned")
                os.rename(staging, owned)
                preserved_owned.append(owned)
                staging.mkdir(mode=0o700)
                (staging / "replacement-marker").write_bytes(b"replacement")
                return {}

            with (
                patch.object(
                    readiness,
                    "_produce_preregistration_roundtrip_fail_closed_report_in_directory",
                    side_effect=hostile_producer,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "staging directory namespace changed",
                ),
            ):
                readiness.produce_preregistration_roundtrip_fail_closed_report(
                    self.campaign,
                    output_directory=destination,
                )

            self.assertEqual(len(observed_staging), 1)
            self.assertEqual(len(preserved_owned), 1)
            self.assertFalse(destination.exists())
            self.assertEqual(
                (observed_staging[0] / "replacement-marker").read_bytes(),
                b"replacement",
            )
            self.assertTrue(preserved_owned[0].is_dir())

    def test_failed_bundle_production_retains_owned_staging_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            destination = root / "authority"
            observed_staging: list[Path] = []
            created_authority: list[readiness._OwnedStagingDirectory] = []
            real_create = readiness._create_owned_staging_directory

            def capture_authority(
                parent: Path,
                *,
                destination_name: str,
                field: str,
            ) -> readiness._OwnedStagingDirectory:
                authority = real_create(
                    parent,
                    destination_name=destination_name,
                    field=field,
                )
                created_authority.append(authority)
                return authority

            def failing_producer(staging: Path) -> dict[str, object]:
                self.assertTrue(staging.is_dir())
                observed_staging.append(staging)
                (staging / "failure-evidence").write_bytes(b"preserve")
                raise RuntimeError("injected producer failure")

            with (
                patch.object(
                    readiness,
                    "_create_owned_staging_directory",
                    side_effect=capture_authority,
                ),
                self.assertRaisesRegex(RuntimeError, "injected producer failure"),
            ):
                readiness._publish_new_authority_bundle(
                    self.campaign,
                    output_directory=destination,
                    field="test authority",
                    producer=failing_producer,
                )

            self.assertEqual(len(observed_staging), 1)
            self.assertEqual(len(created_authority), 1)
            self.assertFalse(destination.exists())
            self.assertEqual(
                (observed_staging[0] / "failure-evidence").read_bytes(),
                b"preserve",
            )
            with self.assertRaises(OSError):
                os.fstat(created_authority[0].directory_descriptor)
            with self.assertRaises(OSError):
                os.fstat(created_authority[0].parent_descriptor)

    def test_bundle_publish_rejects_writable_staging_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            root.chmod(0o777)
            try:
                with self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "process-owned and not group/world-writable",
                ):
                    readiness._publish_new_authority_bundle(
                        self.campaign,
                        output_directory=root / "authority",
                        field="test authority",
                        producer=lambda _staging: {},
                    )
            finally:
                root.chmod(0o700)
            self.assertEqual(list(root.iterdir()), [])

    def test_descriptor_capture_rejects_concurrent_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "evidence.json"
            replacement = root / "replacement.json"
            target.write_bytes(b"a" * 4096)
            replacement.write_bytes(b"b" * 4096)
            root_fd = readiness._open_real_directory(root, field="test bundle")
            real_read = os.read
            swapped = False

            def swap_after_first_read(file_descriptor: int, count: int) -> bytes:
                nonlocal swapped
                payload = real_read(file_descriptor, count)
                if not swapped:
                    os.replace(replacement, target)
                    swapped = True
                return payload

            try:
                with (
                    patch.object(
                        readiness.os,
                        "read",
                        side_effect=swap_after_first_read,
                    ),
                    self.assertRaisesRegex(
                        phase_a.OpenEcologyPhaseAError,
                        "changed identity or bytes",
                    ),
                ):
                    readiness._capture_regular_file_at(
                        root_fd,
                        root,
                        target.name,
                        field="hostile evidence",
                        maximum_bytes=8192,
                    )
            finally:
                os.close(root_fd)
            self.assertTrue(swapped)

    def test_command_receipt_bounds_noisy_descendant_and_kills_process_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            child_pid_path = root / "noisy-child.pid"
            parent_code = (
                "import pathlib,subprocess,sys\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import os,signal,time\\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\\n"
                "while True:\\n"
                ' os.write(1,b"x"*4096)\\n time.sleep(0.001)\'])\n'
                f"pathlib.Path({str(child_pid_path)!r}).write_text("
                "str(child.pid),encoding='ascii')\n"
                "child.wait()\n"
            )
            with (
                patch.object(qualification, "COMMAND_OUTPUT_LIMIT_BYTES", 8192),
                self.assertRaisesRegex(
                    RuntimeError,
                    "output exceeded the qualification byte limit",
                ),
            ):
                qualification.run_command_receipt(
                    (sys.executable, "-c", parent_code),
                    cwd=root,
                    timeout_seconds=10.0,
                )
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            deadline = time.monotonic() + 3.0
            active_state = ""
            while time.monotonic() < deadline:
                status = subprocess.run(
                    ("ps", "-p", str(child_pid), "-o", "stat="),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                active_state = status.stdout.strip()
                if not active_state or active_state.startswith("Z"):
                    break
                time.sleep(0.05)
            self.assertTrue(
                not active_state or active_state.startswith("Z"),
                f"noisy descendant survived group termination: {active_state}",
            )

    def test_benchmark_health_samples_bracket_child_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            events: list[str] = []
            process_environments: list[dict[str, str]] = []
            host = {
                "boot_id": "test-boot",
                "memory": {"ram_used_share": 0.1, "swap_used_bytes": 0},
                "gpus": [],
                "xid_count": 0,
                "oom_count": 0,
            }

            def fake_process(
                *_args: object,
                **kwargs: object,
            ) -> tuple[int, bytes, bytes]:
                events.append("spawn-and-exit")
                process_environments.append(dict(kwargs["environment"]))
                return 0, b"{}", b""

            def observe_host(_root: Path) -> dict[str, object]:
                events.append("health")
                return host

            with patch.object(
                qualification,
                "_run_bounded_subprocess",
                side_effect=fake_process,
            ):
                _receipt, report, samples = qualification._run_monitored_benchmark(
                    (sys.executable, "-c", "raise SystemExit(0)"),
                    cwd=root,
                    campaign_root=root,
                    population_mode="heritable",
                    host_observer=observe_host,
                )
            self.assertEqual(events, ["health", "spawn-and-exit", "health"])
            self.assertEqual(report, {})
            self.assertEqual(len(samples), 2)
            self.assertEqual(process_environments[0]["PYTHONHASHSEED"], "0")
            self.assertEqual(
                process_environments[0]["PYTHONPATH"],
                str(root / "python"),
            )
            self.assertEqual(
                process_environments[0]["EVOLUTION_SIM_REPOSITORY_ROOT"],
                str(root),
            )

    def test_qualification_source_commit_is_exact_lowercase_sha(self) -> None:
        self.assertEqual(qualification._source_commit("a" * 40), "a" * 40)
        for invalid in ("a" * 39, "A" * 40, "../" + "a" * 40, 7):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "lowercase 40-character SHA",
                ):
                    qualification._source_commit(invalid)

    def test_descriptor_capture_rejects_symlinked_directory_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "one real directory",
            ):
                readiness._open_real_directory(
                    linked,
                    field="hostile bundle",
                )

    def test_staged_authority_keeps_phase_d_proofs_out_of_phase_a(self) -> None:
        stage_by_id = {
            stage["stage_id"]: stage
            for stage in self.campaign["launch_authority"]["stages"]
        }
        phase_a_stage = stage_by_id["phase_a_training"]
        phase_a_dependencies = {
            entry["dependency_id"]
            for entry in phase_a_stage["required_dependency_evidence"]
        }
        self.assertEqual(
            phase_a_dependencies,
            {
                "readiness_dependency_01",
                "readiness_dependency_02",
                "readiness_dependency_03",
                "readiness_dependency_04",
                "readiness_dependency_09",
                "readiness_dependency_10",
            },
        )
        self.assertNotIn(
            "persistent_island_50000",
            {
                kind
                for entry in phase_a_stage["required_dependency_evidence"]
                for kind in entry["evidence_kinds"]
            },
        )
        phase_d_stage = stage_by_id["phase_d_persistent_ecology"]
        self.assertIn(
            "persistent_island_50000",
            {
                kind
                for entry in phase_d_stage["required_dependency_evidence"]
                for kind in entry["evidence_kinds"]
            },
        )
        self.assertNotIn(
            "verification_before_prune",
            phase_d_stage["required_operational_evidence"],
        )
        self.assertIs(
            self.campaign["launch_authority"]["local_pruning_supported"],
            False,
        )

    def test_behavioral_gate_validators_reject_partial_or_nonproduction_proofs(
        self,
    ) -> None:
        d1 = _facts("cross_surface_and_self_echo", self.campaign)
        readiness._validate_cross_surface(d1, self.campaign, None)
        partial_d1 = json.loads(json.dumps(d1))
        partial_d1["self_echo_case_count"] = 1
        partial_d1["signal_emission_success_count"] = 1
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "self-echo semantics",
        ):
            readiness._validate_cross_surface(partial_d1, self.campaign, None)

        d2 = _facts("runtime_genome_and_action_source", self.campaign)
        readiness._validate_genome_runtime(d2, self.campaign, None)
        forged_d2 = json.loads(json.dumps(d2))
        forged_d2["exact_child_derivation_mismatch_count"] = 1
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "runtime genome/action-source behavior",
        ):
            readiness._validate_genome_runtime(forged_d2, self.campaign, None)

        d3 = _facts("critic_gradient_and_density_schedule", self.campaign)
        readiness._validate_critic_gradient(d3, self.campaign, None)
        forged_d3 = json.loads(json.dumps(d3))
        forged_d3["observed_cell_contracts"][2]["critic_genome_conditioning"] = "none"
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "four critic/gradient cell contracts drifted",
        ):
            readiness._validate_critic_gradient(forged_d3, self.campaign, None)

        d4 = _facts("fixed_batch_equivalence_and_speed", self.campaign)
        readiness._validate_fixed_batch(d4, self.campaign, None)
        forged_kernel = json.loads(json.dumps(d4))
        forged_kernel["numeric_kernel"] = "unsealed_dense_kernel"
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "recurrent model or numeric-kernel contract drifted",
        ):
            readiness._validate_fixed_batch(forged_kernel, self.campaign, None)
        for field, replacement in (
            ("device", "cpu"),
            (
                "timing_order",
                [["scalar", "batched"], ["scalar", "batched"]],
            ),
            ("max_scalar_reference_logit_tolerance_ratio", 1.01),
            ("collector_numeric_transition_comparison_count", 999),
            ("collector_hidden_component_comparison_count", 255_999),
            ("collector_bootstrap_value_comparison_count", 1_001),
            ("collector_identity_mismatch_count", 1),
            ("collector_hidden_shape_mismatch_count", 1),
            ("collector_bootstrap_none_mismatch_count", 1),
            ("collector_max_input_hidden_tolerance_ratio", 1.01),
            ("collector_max_logprob_tolerance_ratio", 1.01),
            ("collector_max_entropy_tolerance_ratio", 1.01),
            ("collector_max_value_tolerance_ratio", 1.01),
            ("collector_max_bootstrap_value_tolerance_ratio", 1.01),
            ("collector_max_abs_value_error", -0.01),
        ):
            forged_d4 = json.loads(json.dumps(d4))
            forged_d4[field] = replacement
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "fixed batch equivalence or measured speed benefit failed",
            ):
                readiness._validate_fixed_batch(forged_d4, self.campaign, None)
        zero_bootstrap_coverage = json.loads(json.dumps(d4))
        zero_bootstrap_coverage["collector_bootstrap_value_comparison_count"] = 0
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "bootstrap-value comparison count must be an integer >= 1",
        ):
            readiness._validate_fixed_batch(
                zero_bootstrap_coverage,
                self.campaign,
                None,
            )
        forged_bucket_topology = json.loads(json.dumps(d4))
        forged_bucket_topology["batched_fixed_batch_execution_bucket_histogram"][
            "64"
        ] = 99
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "fixed batch equivalence or measured speed benefit failed",
        ):
            readiness._validate_fixed_batch(
                forged_bucket_topology,
                self.campaign,
                None,
            )
        forged_repeat_path = json.loads(json.dumps(d4))
        forged_repeat_path["batched_repeat_collector_path_sha256"][1] = "e" * 64
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "fixed batch equivalence or measured speed benefit failed",
        ):
            readiness._validate_fixed_batch(
                forged_repeat_path,
                self.campaign,
                None,
            )
        for mutate_matrix in (
            lambda matrix: next(
                case for case in matrix["cases"] if case["active_rows"] == 129
            ).update({"execution_rows": 320}),
            lambda matrix: matrix.update({"action_mismatch_count": 1}),
            lambda matrix: matrix.update({"max_hidden_tolerance_ratio": 1.01}),
            lambda matrix: matrix.update({"distinct_action_mask_count": 1}),
            lambda matrix: matrix.update({"distinct_feedback_vector_count": 1}),
            lambda matrix: matrix.update(
                {"nonzero_feedback_row_count": (_D4_BUCKET_MATRIX_COMPARISON_COUNT - 1)}
            ),
            lambda matrix: matrix.update(
                {"feedback_input_sha256": matrix["action_mask_input_sha256"]}
            ),
        ):
            forged_matrix = json.loads(json.dumps(d4))
            mutate_matrix(forged_matrix["bounded_bucket_numeric_matrix"])
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "fixed batch equivalence or measured speed benefit failed",
            ):
                readiness._validate_fixed_batch(
                    forged_matrix,
                    self.campaign,
                    None,
                )
        for digest_field in (
            "action_mask_input_sha256",
            "feedback_input_sha256",
        ):
            forged_matrix = json.loads(json.dumps(d4))
            forged_matrix["bounded_bucket_numeric_matrix"][digest_field] = "0" * 63
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "must be lowercase SHA-256",
            ):
                readiness._validate_fixed_batch(
                    forged_matrix,
                    self.campaign,
                    None,
                )
        forged_determinism = json.loads(json.dumps(d4))
        forged_determinism["deterministic_cuda_contract"][
            "deterministic_algorithms_enabled"
        ] = False
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "deterministic CUDA contract drifted",
        ):
            readiness._validate_fixed_batch(
                forged_determinism,
                self.campaign,
                None,
            )
        forged_training_seed_access = json.loads(json.dumps(d4))
        forged_training_seed_access["proof_seed_contract"][
            "scientific_training_seed_access_count"
        ] = 1
        with self.assertRaisesRegex(
            phase_a.OpenEcologyPhaseAError,
            "engineering proof seed contract drifted",
        ):
            readiness._validate_fixed_batch(
                forged_training_seed_access,
                self.campaign,
                None,
            )

    def test_producer_and_verifier_registries_are_immutable_and_exact(self) -> None:
        expected_kinds = {
            kind
            for catalog in readiness._DEPENDENCY_EVIDENCE_CATALOG
            for kinds in catalog.values()
            for kind in kinds
        } | {
            kind
            for catalog in readiness._OPERATIONAL_EVIDENCE_CATALOG
            for kind in catalog.values()
        }
        self.assertEqual(
            set(readiness.PROOF_PRODUCER_AVAILABILITY),
            expected_kinds,
        )
        self.assertEqual(
            set(readiness.PROOF_PRODUCERS),
            {
                "campaign_storage_capacity",
                "capture_noninterference_reexecution",
                "critic_gradient_and_density_schedule",
                "cross_surface_and_self_echo",
                "exact_sha_phase_a_training_and_torch_ci",
                "fixed_batch_equivalence_and_speed",
                "output_lock_contention",
                "phase_a_training_throughput",
                "preregistration_roundtrip_fail_closed",
                "runtime_genome_and_action_source",
            },
        )
        self.assertEqual(
            set(readiness.REPORT_AUTHORITY_VERIFIERS),
            {
                "campaign_storage_capacity",
                "critic_gradient_and_density_schedule",
                "cross_surface_and_self_echo",
                "exact_sha_phase_a_training_and_torch_ci",
                "fixed_batch_equivalence_and_speed",
                "output_lock_contention",
                "phase_a_training_throughput",
                "preregistration_roundtrip_fail_closed",
                "runtime_genome_and_action_source",
            },
        )
        with self.assertRaises(TypeError):
            readiness.PROOF_PRODUCERS["forged"] = _reconstruct_report  # type: ignore[index]
        with self.assertRaises(TypeError):
            readiness.REPORT_AUTHORITY_VERIFIERS["forged"] = (  # type: ignore[index]
                _reconstruct_report
            )

    def test_generic_self_signed_boolean_is_not_behavioral_evidence(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
        ):
            root = Path(temporary).resolve()
            forged = root / "forged.json"
            forged.write_text('{"proved":true}\n', encoding="utf-8")
            dependencies, gates = _report_paths(root, self.campaign)
            dependencies["readiness_dependency_01"]["cross_surface_and_self_echo"] = (
                forged
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "schema is not an accepted authority envelope",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )

    def test_partial_index_reports_only_the_exact_missing_evidence(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _AUTHORIZED_AT,
                    field="test.authorized_at",
                ),
            ),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, _gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports={
                    "readiness_dependency_01": dependencies["readiness_dependency_01"]
                },
                operational_reports={},
            )
            index_path = root / "partial-index.json"
            _write_json(index_path, index)
            report = phase_a.open_ecology_phase_a_launch_readiness(
                preregistration=self.campaign,
                evidence_index=index,
                evidence_index_path=index_path,
            )

        self.assertEqual(
            report["dependencies"]["readiness_dependency_01"]["status"],
            "semantically_valid",
        )
        self.assertEqual(
            report["dependencies"]["readiness_dependency_02"]["status"],
            "missing",
        )
        self.assertEqual(len(report["blockers"]), 8)

    def test_registered_verifiers_allow_semantic_reports_to_build_authority(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _AUTHORIZED_AT,
                    field="test.authorized_at",
                ),
            ),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports=dependencies,
                operational_reports=gates,
            )
            index_path = root / "evidence-index.json"
            _write_json(index_path, index)
            inspected = phase_a.open_ecology_phase_a_launch_readiness(
                preregistration=self.campaign,
                evidence_index=index,
                evidence_index_path=index_path,
            )
            self.assertIs(inspected["phase_a_training_authorized"], True)
            self.assertEqual(inspected["blockers"], [])
            authorization_path = root / "launch-authorization.json"
            authorization = phase_a.build_open_ecology_phase_a_launch_authorization(
                self.campaign,
                evidence_index=index,
                evidence_index_path=index_path,
                authorization_path=authorization_path,
            )
            _write_json(authorization_path, authorization)
            phase_a.validate_open_ecology_phase_a_launch_authorization(
                authorization,
                preregistration=self.campaign,
                authorization_path=authorization_path,
            )

        self.assertEqual(
            authorization["authorization"]["authorization_scope"],
            "phase_a_training_only",
        )
        self.assertIs(authorization["authorization"]["phase_b_authorized"], False)

    def test_semantically_plausible_self_digested_report_cannot_self_sign(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "raw producer identity drifted",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )

    def test_boolean_verifier_flag_cannot_create_authority(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                {"cross_surface_and_self_echo": True},
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "no independent raw-evidence authority verifier",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )

    def test_verifier_reconstruction_must_match_report_identity_and_facts(
        self,
    ) -> None:
        def mismatched_reconstruction(
            report_path: Path,
            preregistration: dict[str, object],
            authorization_time: object,
        ) -> dict[str, object]:
            reconstructed = _reconstruct_report(
                report_path,
                preregistration,
                authorization_time,
            )
            reconstructed["facts"] = {"forged": True}
            return reconstructed

        verifiers = dict(_ALL_AUTHORITY_VERIFIERS)
        verifiers["cross_surface_and_self_echo"] = mismatched_reconstruction
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                verifiers,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "independently reconstructed facts or identity differ",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )

    def test_semantically_forged_short_persistent_run_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            path = root / "persistent_island_50000" / "report.json"
            _write_report(
                path,
                campaign=self.campaign,
                kind="persistent_island_50000",
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            report["facts"]["terminal_tick"] = 10_000
            _resign(report)
            _write_json(path, report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "50,000-tick",
            ):
                readiness._validate_report(
                    report,
                    expected_kind="persistent_island_50000",
                    preregistration=self.campaign,
                    authorization_time=None,
                    report_path=path,
                    require_authority_verifier=False,
                )

    def test_persistent_runner_proof_cannot_consume_post_training_artifacts(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            path = root / "persistent_island_50000" / "report.json"
            _write_report(
                path,
                campaign=self.campaign,
                kind="persistent_island_50000",
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            report["facts"]["phase_a_artifact_access_count"] = 1
            _resign(report)
            _write_json(path, report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "50,000-tick",
            ):
                readiness._validate_report(
                    report,
                    expected_kind="persistent_island_50000",
                    preregistration=self.campaign,
                    authorization_time=None,
                    report_path=path,
                    require_authority_verifier=False,
                )

    def test_source_bound_report_from_stale_commit_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            path = dependencies["readiness_dependency_03"][
                "critic_gradient_and_density_schedule"
            ]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["source"]["commit"] = "d" * 40
            _resign(report)
            _write_json(path, report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "not exact-campaign source-bound",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )

    def test_report_bytes_changed_after_index_are_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _AUTHORIZED_AT,
                    field="test.authorized_at",
                ),
            ),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports=dependencies,
                operational_reports=gates,
            )
            index_path = root / "evidence-index.json"
            _write_json(index_path, index)
            target = dependencies["readiness_dependency_03"][
                "critic_gradient_and_density_schedule"
            ]
            target.write_text(
                target.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "reference or bytes drifted",
            ):
                phase_a.build_open_ecology_phase_a_launch_authorization(
                    self.campaign,
                    evidence_index=index,
                    evidence_index_path=index_path,
                    authorization_path=root / "launch-authorization.json",
                )

    def test_supplied_index_must_equal_exact_index_file_bytes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports=dependencies,
                operational_reports=gates,
            )
            index_path = root / "evidence-index.json"
            substituted = dict(index)
            substituted["dependency_reports"] = []
            _resign(substituted)
            _write_json(index_path, substituted)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "supplied payload differs from its exact file bytes",
            ):
                phase_a.open_ecology_phase_a_launch_readiness(
                    preregistration=self.campaign,
                    evidence_index=index,
                    evidence_index_path=index_path,
                )

    def test_authorization_cannot_substitute_report_outside_its_index(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _AUTHORIZED_AT,
                    field="test.authorized_at",
                ),
            ),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports=dependencies,
                operational_reports=gates,
            )
            index_path = root / "evidence-index.json"
            authorization_path = root / "launch-authorization.json"
            _write_json(index_path, index)
            authorization = phase_a.build_open_ecology_phase_a_launch_authorization(
                self.campaign,
                evidence_index=index,
                evidence_index_path=index_path,
                authorization_path=authorization_path,
            )
            authorization["readiness_dependencies"][0]["evidence"][0] = authorization[
                "readiness_dependencies"
            ][1]["evidence"][0]
            _resign(authorization)
            _write_json(authorization_path, authorization)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "differs from its index",
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )

    def test_authorization_argument_must_equal_authorization_file_bytes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _AUTHORIZED_AT,
                    field="test.authorized_at",
                ),
            ),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports=dependencies,
                operational_reports=gates,
            )
            index_path = root / "evidence-index.json"
            authorization_path = root / "launch-authorization.json"
            _write_json(index_path, index)
            authorization = phase_a.build_open_ecology_phase_a_launch_authorization(
                self.campaign,
                evidence_index=index,
                evidence_index_path=index_path,
                authorization_path=authorization_path,
            )
            _write_json(authorization_path, {"substituted": True})
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "supplied payload differs from its exact file bytes",
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )

    def test_storage_attestation_must_be_fresh_at_authorization(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    "2026-07-29T11:00:00Z",
                    field="test.authorized_at",
                ),
            ),
            patch.object(
                readiness,
                "REPORT_AUTHORITY_VERIFIERS",
                _ALL_AUTHORITY_VERIFIERS,
            ),
        ):
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            index = phase_a.build_open_ecology_phase_a_evidence_index(
                self.campaign,
                evidence_root=root,
                dependency_reports=dependencies,
                operational_reports=gates,
            )
            index_path = root / "evidence-index.json"
            _write_json(index_path, index)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "storage capacity attestation is stale",
            ):
                phase_a.build_open_ecology_phase_a_launch_authorization(
                    self.campaign,
                    evidence_index=index,
                    evidence_index_path=index_path,
                    authorization_path=root / "launch-authorization.json",
                )

    def test_old_authorization_cannot_replay_stale_drive_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _AUTHORIZED_AT,
                        field="test.authorized_at",
                    ),
                ),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    _ALL_AUTHORITY_VERIFIERS,
                ),
            ):
                index = phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )
                index_path = root / "evidence-index.json"
                authorization_path = root / "launch-authorization.json"
                _write_json(index_path, index)
                authorization = phase_a.build_open_ecology_phase_a_launch_authorization(
                    self.campaign,
                    evidence_index=index,
                    evidence_index_path=index_path,
                    authorization_path=authorization_path,
                )
                _write_json(authorization_path, authorization)
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        "2026-07-29T11:00:00Z",
                        field="test.validation_time",
                    ),
                ),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    _ALL_AUTHORITY_VERIFIERS,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "storage capacity attestation is stale",
                ),
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        "2026-07-29T11:00:00Z",
                        field="test.closure_time",
                    ),
                ),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    _ALL_AUTHORITY_VERIFIERS,
                ),
            ):
                readiness.validate_launch_authorization(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                    require_current_storage_freshness=False,
                )

    def test_static_authorization_keeps_semantics_but_skips_host_reexecution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _AUTHORIZED_AT,
                        field="test.authorized_at",
                    ),
                ),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    _ALL_AUTHORITY_VERIFIERS,
                ),
            ):
                index = phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )
                index_path = root / "evidence-index.json"
                authorization_path = root / "launch-authorization.json"
                _write_json(index_path, index)
                authorization = phase_a.build_open_ecology_phase_a_launch_authorization(
                    self.campaign,
                    evidence_index=index,
                    evidence_index_path=index_path,
                    authorization_path=authorization_path,
                )
                _write_json(authorization_path, authorization)

            semantic_calls: list[str] = []
            semantic_validators = dict(readiness._SEMANTIC_VALIDATORS)
            semantic_kind = "fixed_batch_equivalence_and_speed"
            original_semantic_validator = semantic_validators[semantic_kind]

            def record_semantic_validation(
                facts: object,
                preregistration: object,
                authorization_time: object,
            ) -> None:
                semantic_calls.append(semantic_kind)
                original_semantic_validator(
                    facts,
                    preregistration,
                    authorization_time,
                )

            semantic_validators[semantic_kind] = record_semantic_validation

            def reject_host_reexecution(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("host-specific authority verifier ran")

            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _AUTHORIZED_AT,
                        field="test.static_validation_time",
                    ),
                ),
                patch.object(
                    readiness,
                    "_SEMANTIC_VALIDATORS",
                    semantic_validators,
                ),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    {
                        kind: reject_host_reexecution
                        for kind in readiness.PROOF_PRODUCER_AVAILABILITY
                    },
                ),
            ):
                phase_a.validate_open_ecology_phase_a_launch_authorization_static(
                    authorization,
                    preregistration=self.campaign,
                    authorization_path=authorization_path,
                )

            self.assertEqual(semantic_calls, [semantic_kind])

    def test_live_source_drift_after_authorization_assembly_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dependencies, gates = _report_paths(root, self.campaign)
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    _ALL_AUTHORITY_VERIFIERS,
                ),
            ):
                index = phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
                )
            index_path = root / "evidence-index.json"
            _write_json(index_path, index)
            with (
                patch.object(
                    phase_a,
                    "_require_live_source",
                    side_effect=[
                        None,
                        phase_a.OpenEcologyPhaseAError("source changed"),
                    ],
                ),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _AUTHORIZED_AT,
                        field="test.authorized_at",
                    ),
                ),
                patch.object(
                    readiness,
                    "REPORT_AUTHORITY_VERIFIERS",
                    _ALL_AUTHORITY_VERIFIERS,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "source drifted after",
                ),
            ):
                phase_a.build_open_ecology_phase_a_launch_authorization(
                    self.campaign,
                    evidence_index=index,
                    evidence_index_path=index_path,
                    authorization_path=root / "launch-authorization.json",
                )

    def test_capture_report_producer_reexecutes_and_compares_canonical_bytes(
        self,
    ) -> None:
        primary: dict[str, object] = {
            "contract": {
                "cell_order": ["A0", "A1", "A2", "A3"],
                "density_levels": [32, 64, 128],
                "case_count": 12,
                "ticks_per_case": 128,
            },
            "dynamics": {"total_births": 5, "total_deaths": 901},
            "exact_digest": _SHA,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                open_ecology_selection,
                "verify_open_ecology_capture_noninterference_proof_by_reexecution",
                return_value=primary,
            ) as reexecute,
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _AUTHORIZED_AT,
                    field="test.produced_at",
                ),
            ),
        ):
            path = Path(temporary) / "primary.json"
            _write_json(path, primary)
            report = readiness.produce_capture_noninterference_reexecution_report(
                self.campaign,
                primary_proof_path=path,
            )

        reexecute.assert_called_once_with(
            primary,
            preregistration=self.campaign,
        )
        self.assertEqual(
            report["facts"]["primary_report_sha256"],
            report["facts"]["reexecution_report_sha256"],
        )
        self.assertEqual(report["facts"]["case_count"], 12)

    def test_d9_raw_bundle_is_independent_and_rejects_hostile_storage(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(phase_a, "_require_live_source"),
            patch.object(
                readiness,
                "_utc_now",
                return_value=readiness._parse_utc(
                    _PRODUCED_AT,
                    field="test.produced_at",
                ),
            ),
        ):
            root = Path(temporary).resolve()
            bundle = root / "d9-authoritative"
            report = readiness.produce_preregistration_roundtrip_fail_closed_report(
                self.campaign,
                output_directory=bundle,
            )
            self.assertEqual(
                report["schema_version"],
                readiness.OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION,
            )
            self.assertEqual(
                report["facts"]["roundtrip_exact_digest"],
                self.campaign["exact_digest"],
            )
            self.assertEqual(
                report["facts"]["unknown_input_rejection_count"],
                1,
            )
            with patch.object(
                readiness,
                "_validate_raw_evidence_manifest_path",
                wraps=readiness._validate_raw_evidence_manifest_path,
            ) as capture_bundle:
                reconstructed = (
                    readiness.verify_preregistration_roundtrip_fail_closed_report(
                        bundle / "report.json",
                        self.campaign,
                        None,
                    )
                )
            self.assertEqual(capture_bundle.call_count, 1)
            self.assertEqual(reconstructed["facts"], report["facts"])

            symlink_parent_target = root / "d9-output-parent-target"
            symlink_parent_target.mkdir()
            symlink_parent = root / "d9-output-parent-link"
            symlink_parent.symlink_to(
                symlink_parent_target,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "real existing directory",
            ):
                readiness.produce_preregistration_roundtrip_fail_closed_report(
                    self.campaign,
                    output_directory=symlink_parent / "forbidden-bundle",
                )

            tampered = _copy_bundle(bundle, root / "d9-tampered")
            with (tampered / "raw" / "builder-inputs.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "reference or bytes drifted",
            ):
                readiness.verify_preregistration_roundtrip_fail_closed_report(
                    tampered / "report.json",
                    self.campaign,
                    None,
                )

            symlinked = _copy_bundle(bundle, root / "d9-symlinked")
            symlink_target = root / "symlink-target.json"
            shutil.copy2(
                symlinked / "raw" / "preregistration.json",
                symlink_target,
            )
            (symlinked / "raw" / "preregistration.json").unlink()
            (symlinked / "raw" / "preregistration.json").symlink_to(symlink_target)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "symbolic link",
            ):
                readiness.verify_preregistration_roundtrip_fail_closed_report(
                    symlinked / "report.json",
                    self.campaign,
                    None,
                )

            hardlinked = _copy_bundle(bundle, root / "d9-hardlinked")
            hardlink_target = root / "hardlink-target.json"
            shutil.copy2(
                hardlinked / "raw" / "preregistration.json",
                hardlink_target,
            )
            (hardlinked / "raw" / "preregistration.json").unlink()
            os.link(
                hardlink_target,
                hardlinked / "raw" / "preregistration.json",
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "regular file",
            ):
                readiness.verify_preregistration_roundtrip_fail_closed_report(
                    hardlinked / "report.json",
                    self.campaign,
                    None,
                )

            surplus = _copy_bundle(bundle, root / "d9-surplus")
            _write_json(surplus / "raw" / "surplus.json", {"forged": True})
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "missing or surplus",
            ):
                readiness.verify_preregistration_roundtrip_fail_closed_report(
                    surplus / "report.json",
                    self.campaign,
                    None,
                )

            escaped = _copy_bundle(bundle, root / "d9-path-escape")
            escaped_target = root / "escaped-target.json"
            shutil.copy2(
                escaped / "raw" / "builder-inputs.json",
                escaped_target,
            )
            escaped_manifest_path = escaped / "raw-evidence-manifest.json"
            escaped_manifest = json.loads(
                escaped_manifest_path.read_text(encoding="utf-8")
            )
            escaped_entry = next(
                entry
                for entry in escaped_manifest["files"]
                if entry["role"] == "builder_inputs"
            )
            escaped_reference = phase_a._file_reference(
                escaped_target,
                base=root,
            )
            escaped_reference["relative_path"] = (
                f"../{escaped_reference['relative_path']}"
            )
            escaped_entry.update(escaped_reference)
            _resign(escaped_manifest)
            _write_json(escaped_manifest_path, escaped_manifest)
            escaped_report_path = escaped / "report.json"
            escaped_report = json.loads(escaped_report_path.read_text(encoding="utf-8"))
            escaped_report["raw_evidence_manifest"] = phase_a._file_reference(
                escaped_manifest_path,
                base=escaped,
            )
            _resign(escaped_report)
            _write_json(escaped_report_path, escaped_report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "path is unsafe",
            ):
                readiness.verify_preregistration_roundtrip_fail_closed_report(
                    escaped_report_path,
                    self.campaign,
                    None,
                )

            forged = _copy_bundle(bundle, root / "d9-report-facts-forged")
            forged_report_path = forged / "report.json"
            forged_report = json.loads(forged_report_path.read_text(encoding="utf-8"))
            forged_report["facts"]["unknown_input_rejection_count"] = 2
            _resign(forged_report)
            _write_json(forged_report_path, forged_report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "independently reconstructed facts",
            ):
                readiness._validate_report(
                    forged_report,
                    expected_kind="preregistration_roundtrip_fail_closed",
                    preregistration=self.campaign,
                    authorization_time=None,
                    report_path=forged_report_path,
                )

    def test_expensive_live_operational_verifiers_are_explicit_only(self) -> None:
        expected_static_verifiers = {
            "exact_sha_phase_a_training_and_torch_ci": (
                readiness.verify_exact_sha_phase_a_training_and_torch_ci_report
            ),
            "phase_a_training_throughput": (
                readiness.verify_phase_a_training_throughput_report
            ),
            "campaign_storage_capacity": (
                readiness.verify_campaign_storage_capacity_report
            ),
            "output_lock_contention": readiness.verify_output_lock_contention_report,
        }
        for evidence_kind, verifier in expected_static_verifiers.items():
            with self.subTest(evidence_kind=evidence_kind):
                self.assertIs(
                    readiness.REPORT_AUTHORITY_VERIFIERS[evidence_kind],
                    verifier,
                )

    def test_d10_reconstructs_exact_source_gpu_torch_and_github_ci(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            preliminary = self.campaign["throughput_gate"]["reports"]
            heritable = root / "heritable.json"
            zero_all = root / "zero-all.json"
            _write_json(heritable, preliminary["heritable"])
            _write_json(zero_all, preliminary["zero_all"])
            source_manifest = _qualification_source_manifest(self.campaign)
            source_observation = _qualification_source_observation(
                self.campaign,
                source_manifest=source_manifest,
            )
            github_receipt = _qualification_github_receipt(self.campaign)

            def write_torch_report(
                _repository_root: Path,
                *,
                output_path: Path,
            ) -> dict[str, object]:
                _write_json(output_path, _qualification_torch_report(self.campaign))
                return _qualification_command_receipt(
                    [
                        sys.executable,
                        str(
                            Path(readiness.__file__).resolve().parents[3]
                            / "scripts"
                            / "validate_mind_torch.py"
                        ),
                        "--require-cuda",
                        "--output",
                        str(output_path),
                    ],
                    stdout="mind_torch_validation_passed=True\n",
                )

            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _PRODUCED_AT,
                        field="test.produced_at",
                    ),
                ),
                patch.object(
                    qualification,
                    "collect_exact_source_observation",
                    return_value=source_observation,
                ) as source_observation_probe,
                patch.object(
                    qualification,
                    "collect_github_check_runs",
                    return_value=github_receipt,
                ) as github_probe,
                patch.object(
                    qualification,
                    "run_torch_cuda_validation",
                    side_effect=write_torch_report,
                ) as torch_probe,
                patch(
                    "evolution_sim.io.source_manifest.source_file_hash_manifest",
                    return_value=source_manifest,
                ),
            ):
                bundle = root / "d10"
                report = (
                    readiness.produce_exact_sha_phase_a_training_and_torch_ci_report(
                        self.campaign,
                        output_directory=bundle,
                        host_class="trainer-class-a",
                        heritable_benchmark_path=heritable,
                        zero_all_benchmark_path=zero_all,
                    )
                )
                captured_reconstruction = (
                    readiness.verify_exact_sha_phase_a_training_and_torch_ci_report(
                        bundle / "report.json",
                        self.campaign,
                        None,
                    )
                )
                self.assertEqual(source_observation_probe.call_count, 1)
                self.assertEqual(github_probe.call_count, 1)
                self.assertEqual(torch_probe.call_count, 1)
                reconstructed = readiness.live_verify_exact_sha_phase_a_training_and_torch_ci_report(
                    bundle / "report.json",
                    self.campaign,
                    None,
                )

            self.assertEqual(captured_reconstruction["facts"], report["facts"])
            self.assertEqual(reconstructed["facts"], report["facts"])
            self.assertEqual(source_observation_probe.call_count, 2)
            self.assertEqual(github_probe.call_count, 2)
            self.assertEqual(torch_probe.call_count, 2)
            self.assertEqual(
                report["facts"]["tested_training_host_classes"],
                ["trainer-class-a"],
            )
            self.assertGreater(report["facts"]["torch_ci_test_count"], 0)
            with (
                patch.object(phase_a, "_require_live_source"),
                patch(
                    "evolution_sim.io.source_manifest.source_file_hash_manifest",
                    return_value=source_manifest,
                ),
                patch.object(
                    qualification,
                    "collect_exact_source_observation",
                    side_effect=RuntimeError("fresh source probe unavailable"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "fresh source probe unavailable",
                ),
            ):
                readiness.live_verify_exact_sha_phase_a_training_and_torch_ci_report(
                    bundle / "report.json",
                    self.campaign,
                    None,
                )

            tampered = _copy_bundle(bundle, root / "d10-tampered")
            github_path = tampered / "raw" / "github-check-runs.json"
            github = json.loads(github_path.read_text(encoding="utf-8"))
            payload = json.loads(github["stdout"])
            payload["check_runs"][0]["conclusion"] = "failure"
            github["stdout"] = json.dumps(
                payload, separators=(",", ":"), sort_keys=True
            )
            _rewrite_raw_and_resign_bundle(
                tampered,
                role="github_check_runs",
                payload=github,
            )
            with (
                patch(
                    "evolution_sim.io.source_manifest.source_file_hash_manifest",
                    return_value=source_manifest,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "GitHub Torch check did not succeed",
                ),
            ):
                readiness.verify_exact_sha_phase_a_training_and_torch_ci_report(
                    tampered / "report.json",
                    self.campaign,
                    None,
                )

            third_party = _copy_bundle(bundle, root / "d10-third-party-check")
            github_path = third_party / "raw" / "github-check-runs.json"
            github = json.loads(github_path.read_text(encoding="utf-8"))
            payload = json.loads(github["stdout"])
            payload["check_runs"][0]["app"]["slug"] = "untrusted-check-writer"
            github["stdout"] = json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
            )
            _rewrite_raw_and_resign_bundle(
                third_party,
                role="github_check_runs",
                payload=github,
            )
            with (
                patch(
                    "evolution_sim.io.source_manifest.source_file_hash_manifest",
                    return_value=source_manifest,
                ),
                self.assertRaisesRegex(
                    phase_a.OpenEcologyPhaseAError,
                    "GitHub Torch check did not succeed",
                ),
            ):
                readiness.verify_exact_sha_phase_a_training_and_torch_ci_report(
                    third_party / "report.json",
                    self.campaign,
                    None,
                )

    def test_throughput_reconstructs_fresh_full_shape_and_live_resources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            benchmark_call_count = 0

            def observe_host(_root: Path) -> dict[str, object]:
                return {"injected": True}

            def fresh_benchmarks(
                _repository_root: Path,
                *,
                source_commit: str,
                host_class: str,
                campaign_root: Path,
                host_observer: qualification.HostObserver,
            ) -> dict[str, object]:
                nonlocal benchmark_call_count
                del campaign_root
                self.assertIs(host_observer, observe_host)
                benchmark_call_count += 1
                reports = self.campaign["throughput_gate"]["reports"]
                receipts: dict[str, object] = {}
                for mode in ("heritable", "zero_all"):
                    payload = readiness._canonical_json_file_bytes(reports[mode])
                    receipts[mode] = {
                        "schema_version": qualification.COMMAND_RECEIPT_SCHEMA_VERSION,
                        "argv": list(
                            qualification._phase_a_benchmark_command(
                                Path(readiness.__file__).resolve().parents[3],
                                population_mode=mode,
                            )
                        ),
                        "started_at_utc": _PRODUCED_AT,
                        "finished_at_utc": _PRODUCED_AT,
                        "elapsed_ns": 10,
                        "returncode": 0,
                        "stdout_sha256": hashlib.sha256(payload).hexdigest(),
                        "stdout_byte_length": len(payload),
                        "stderr": "",
                    }
                gpu_delta = 0.05 if benchmark_call_count == 2 else 0.0
                ram_delta = 0.04 if benchmark_call_count == 2 else 0.0
                samples = [
                    _resource_sample(
                        "heritable",
                        gpu_share=0.40 - gpu_delta,
                        ram_share=0.30 - ram_delta,
                    ),
                    _resource_sample(
                        "zero_all",
                        gpu_share=0.60 - gpu_delta,
                        ram_share=0.50 - ram_delta,
                    ),
                ]
                return {
                    "schema_version": qualification.RESOURCE_TELEMETRY_SCHEMA_VERSION,
                    "source_commit": source_commit,
                    "host_class": host_class,
                    "sample_interval_seconds": 5.0,
                    "benchmark_receipts": receipts,
                    "benchmark_reports": reports,
                    "resource_samples": samples,
                }

            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _PRODUCED_AT,
                        field="test.produced_at",
                    ),
                ),
                patch.object(
                    qualification,
                    "run_fresh_phase_a_benchmarks",
                    side_effect=fresh_benchmarks,
                ) as benchmark_probe,
            ):
                bundle = root / "throughput"
                report = readiness.produce_phase_a_training_throughput_report(
                    self.campaign,
                    output_directory=bundle,
                    host_class="trainer-class-a",
                    host_observer=observe_host,
                )
                captured_reconstruction = (
                    readiness.verify_phase_a_training_throughput_report(
                        bundle / "report.json",
                        self.campaign,
                        None,
                    )
                )
                self.assertEqual(benchmark_probe.call_count, 1)
                reconstructed = (
                    readiness.live_verify_phase_a_training_throughput_report(
                        bundle / "report.json",
                        self.campaign,
                        None,
                        host_observer=observe_host,
                    )
                )

            self.assertEqual(captured_reconstruction["facts"], report["facts"])
            self.assertEqual(reconstructed["facts"], report["facts"])
            self.assertEqual(benchmark_probe.call_count, 2)
            self.assertEqual(report["facts"]["maximum_gpu_memory_share"], 0.60)
            self.assertEqual(report["facts"]["swap_delta_bytes"], 0)
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    qualification,
                    "run_fresh_phase_a_benchmarks",
                    side_effect=RuntimeError("fresh benchmark unavailable"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "fresh benchmark unavailable",
                ),
            ):
                readiness.live_verify_phase_a_training_throughput_report(
                    bundle / "report.json",
                    self.campaign,
                    None,
                    host_observer=observe_host,
                )

            tampered = _copy_bundle(bundle, root / "throughput-tampered")
            telemetry_path = tampered / "raw" / "resource-telemetry.json"
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            telemetry["resource_samples"][1]["host"]["memory"]["swap_used_bytes"] = 1
            _rewrite_raw_and_resign_bundle(
                tampered,
                role="resource_telemetry",
                payload=telemetry,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "throughput/resource gate failed",
            ):
                readiness.verify_phase_a_training_throughput_report(
                    tampered / "report.json",
                    self.campaign,
                    None,
                )

            stale = _copy_bundle(bundle, root / "throughput-stale")
            stale_telemetry_path = stale / "raw" / "resource-telemetry.json"
            stale_telemetry = json.loads(
                stale_telemetry_path.read_text(encoding="utf-8")
            )
            stale_telemetry["benchmark_receipts"]["zero_all"]["started_at_utc"] = (
                "2026-07-27T09:00:00Z"
            )
            stale_telemetry["benchmark_receipts"]["zero_all"]["finished_at_utc"] = (
                "2026-07-27T09:01:00Z"
            )
            _rewrite_raw_and_resign_bundle(
                stale,
                role="resource_telemetry",
                payload=stale_telemetry,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "stale or detached",
            ):
                readiness.verify_phase_a_training_throughput_report(
                    stale / "report.json",
                    self.campaign,
                    None,
                )

            nonfinite = _copy_bundle(bundle, root / "throughput-nonfinite")
            nonfinite_path = nonfinite / "raw" / "resource-telemetry.json"
            original_bytes = nonfinite_path.read_bytes()
            nonfinite_bytes = original_bytes.replace(
                b'"ram_used_share":0.3',
                b'"ram_used_share":1e10000',
                1,
            )
            self.assertNotEqual(nonfinite_bytes, original_bytes)
            _rewrite_raw_bytes_and_resign_bundle(
                nonfinite,
                role="resource_telemetry",
                payload=nonfinite_bytes,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "finite numeric",
            ):
                readiness.verify_phase_a_training_throughput_report(
                    nonfinite / "report.json",
                    self.campaign,
                    None,
                )

    def test_storage_uses_descriptor_and_live_rclone_measurements(self) -> None:
        capacity = 1_000 * 1024**3
        free = 300 * 1024**3
        fragment_size = 4096
        target_directory = "/srv/evolution-sim/runs"
        rclone_config_path = "/Users/test/.config/rclone/rclone.conf"
        ssh_config_stdout = "hostname trainer-node.example.invalid\nuser runner\n"
        host_key = "ssh-ed25519 SHA256:" + "A" * 43
        ssh_connection = {
            "address": "10.0.0.1",
            "authenticated_host": "trainer-node.example.invalid",
            "authentication": "publickey",
            "host_key": host_key,
            "port": 22,
        }
        ssh_verbose_stderr = (
            f"debug1: Server host key: {host_key}\n"
            'Authenticated to trainer-node.example.invalid ([10.0.0.1]:22) using "publickey".\n'
        )
        source = self.campaign["source"]
        local_tool_paths = {
            "git": "/usr/bin/git",
            "rclone": "/opt/homebrew/bin/rclone",
            "ssh": "/usr/bin/ssh",
            "zstd": "/usr/bin/zstd",
        }
        remote_tool_paths = {
            "env": "/usr/bin/env",
            "git": "/usr/bin/git",
            "python": "/usr/bin/python3.12",
            "sha256sum": "/usr/bin/sha256sum",
            "zstd": "/usr/bin/zstd",
        }
        authority_payload = {
            "schema_version": archive_authority.ARCHIVE_TOOL_AUTHORITY_SCHEMA_VERSION,
            "source": {
                "git_sha": source["commit"],
                "manifest_sha256": source["manifest_sha256"],
                "remote_repository_root": "/srv/evolution-sim/checkout",
            },
            "endpoint": {
                "rclone_base": "gdrive:evolution-sim-backups",
                "rclone_config": {
                    "path": rclone_config_path,
                    "sha256": _SHA,
                },
                "ssh_effective_config_sha256": hashlib.sha256(
                    ssh_config_stdout.encode("utf-8")
                ).hexdigest(),
                "ssh_connection": ssh_connection,
                "ssh_target": "trainer-node",
            },
            "local_tools": {
                name: {"path": path, "sha256": _SHA}
                for name, path in local_tool_paths.items()
            },
            "remote_tools": {
                name: {"path": path, "sha256": _SHA}
                for name, path in remote_tool_paths.items()
            },
            "remote_helpers": {
                path: _SHA for path in archive_authority.REMOTE_HELPER_PATHS
            },
        }
        remote_probe = {
            "schema_version": qualification.REMOTE_STORAGE_PROBE_SCHEMA_VERSION,
            "target_directory": target_directory,
            "remote_hostname": "trainer-host",
            "remote_boot_id": "87350a1b-e992-490b-bcd5-5e95ac29dc89",
            "remote_python": {
                "path": "/usr/bin/python3.12",
                "sha256": _SHA,
            },
            "target_filesystem": {
                "device": 1,
                "inode": 2,
                "fragment_size": fragment_size,
                "blocks": capacity // fragment_size,
                "available_blocks": free // fragment_size,
            },
            "filesystem_measurement_contract": (
                "remote_python_open_nofollow_directory_fstatvfs_fstat_identity_v1"
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            authority_path = root / "archive-authority.json"
            _write_json(authority_path, authority_payload)
            authority_sha256 = hashlib.sha256(authority_path.read_bytes()).hexdigest()
            campaign = json.loads(json.dumps(self.campaign))
            campaign["source"]["archive_tool_authority_sha256"] = authority_sha256
            _resign(campaign)
            authority_value = archive_authority.ArchiveToolAuthority(
                authority_path=str(authority_path),
                authority_sha256=authority_sha256,
                source_git_sha=str(source["commit"]),
                source_manifest_sha256=str(source["manifest_sha256"]),
                remote_repository_root="/srv/evolution-sim/checkout",
                ssh_target="trainer-node",
                ssh_effective_config_sha256=hashlib.sha256(
                    ssh_config_stdout.encode("utf-8")
                ).hexdigest(),
                ssh_connection=archive_authority.SshConnectionIdentity(
                    authenticated_host="trainer-node.example.invalid",
                    address="10.0.0.1",
                    port=22,
                    authentication="publickey",
                    host_key=host_key,
                ),
                rclone_base="gdrive:evolution-sim-backups",
                rclone_config=archive_authority.FilePin(
                    path=rclone_config_path,
                    sha256=_SHA,
                ),
                local_tools=tuple(
                    (
                        name,
                        archive_authority.FilePin(path=path, sha256=_SHA),
                    )
                    for name, path in local_tool_paths.items()
                ),
                remote_tools=tuple(
                    (
                        name,
                        archive_authority.FilePin(path=path, sha256=_SHA),
                    )
                    for name, path in remote_tool_paths.items()
                ),
                remote_helpers=tuple(
                    (path, _SHA) for path in archive_authority.REMOTE_HELPER_PATHS
                ),
            )
            raw_receipts = [
                _qualification_command_receipt(
                    [
                        "/usr/bin/ssh",
                        "-G",
                        *qualification.SEALED_SSH_OPTIONS,
                        "trainer-node",
                    ],
                    stdout=ssh_config_stdout,
                ),
                _qualification_command_receipt(
                    [
                        "/usr/bin/ssh",
                        "-v",
                        *qualification.SEALED_SSH_OPTIONS,
                        "trainer-node",
                        qualification.storage_remote_probe_command(
                            target_directory,
                            remote_python_path="/usr/bin/python3.12",
                        ),
                    ],
                    stdout=json.dumps(
                        remote_probe,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    stderr=ssh_verbose_stderr,
                ),
                _qualification_command_receipt(
                    [
                        "/opt/homebrew/bin/rclone",
                        "--config",
                        rclone_config_path,
                        "about",
                        "gdrive:",
                        "--json",
                    ],
                    stdout=json.dumps(
                        {"free": 400 * 1024**3},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            ]
            with (
                patch.object(
                    qualification,
                    "verify_local_authority_files",
                ),
                patch.object(
                    qualification,
                    "run_command_receipt",
                    side_effect=raw_receipts,
                ),
            ):
                collected_measurement = qualification.measure_campaign_storage(
                    target_directory,
                    archive_authority=authority_value,
                    drive_remote="gdrive:",
                )
            self.assertEqual(
                collected_measurement["ssh_config_receipt"]["stdout"],
                qualification.SSH_EFFECTIVE_CONFIG_REDACTION_MARKER,
            )
            self.assertEqual(
                collected_measurement["remote_probe_receipt"]["stderr"],
                qualification.SSH_VERBOSE_LOG_REDACTION_MARKER,
            )
            self.assertEqual(
                collected_measurement["ssh_verbose_stderr_sha256"],
                hashlib.sha256(ssh_verbose_stderr.encode("utf-8")).hexdigest(),
            )
            self.assertNotIn("user runner", json.dumps(collected_measurement))
            self.assertNotIn("10.0.0.1]:22", json.dumps(collected_measurement))
            measurement = {
                "schema_version": qualification.STORAGE_MEASUREMENT_SCHEMA_VERSION,
                "checked_at_utc": _PRODUCED_AT,
                "measurement_location": "remote_ssh_target_with_local_drive_v1",
                "target_ssh": "trainer-node",
                "target_directory": target_directory,
                "archive_authority_sha256": authority_sha256,
                "ssh_effective_config_sha256": hashlib.sha256(
                    ssh_config_stdout.encode("utf-8")
                ).hexdigest(),
                "ssh_verbose_stderr_sha256": hashlib.sha256(
                    ssh_verbose_stderr.encode("utf-8")
                ).hexdigest(),
                "ssh_connection": ssh_connection,
                "ssh_config_receipt": _qualification_command_receipt(
                    [
                        "/usr/bin/ssh",
                        "-G",
                        *qualification.SEALED_SSH_OPTIONS,
                        "trainer-node",
                    ],
                    stdout=qualification.SSH_EFFECTIVE_CONFIG_REDACTION_MARKER,
                ),
                "remote_probe_code_sha256": (
                    qualification.remote_storage_probe_code_sha256()
                ),
                "remote_probe_receipt": _qualification_command_receipt(
                    [
                        "/usr/bin/ssh",
                        "-v",
                        *qualification.SEALED_SSH_OPTIONS,
                        "trainer-node",
                        qualification.storage_remote_probe_command(
                            target_directory,
                            remote_python_path="/usr/bin/python3.12",
                        ),
                    ],
                    stdout=json.dumps(
                        remote_probe,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    stderr=qualification.SSH_VERBOSE_LOG_REDACTION_MARKER,
                ),
                "drive_remote": "gdrive:",
                "rclone_config": {
                    "path": rclone_config_path,
                    "sha256": _SHA,
                },
                "drive_receipt": _qualification_command_receipt(
                    [
                        "/opt/homebrew/bin/rclone",
                        "--config",
                        rclone_config_path,
                        "about",
                        "gdrive:",
                        "--json",
                    ],
                    stdout=json.dumps(
                        {"free": 400 * 1024**3},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            }
            bundle = root / "storage"
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _PRODUCED_AT,
                        field="test.produced_at",
                    ),
                ),
                patch.object(
                    qualification,
                    "measure_campaign_storage",
                    return_value=measurement,
                ) as storage_probe,
                patch.object(
                    archive_authority,
                    "load_archive_tool_authority",
                    return_value=authority_value,
                ) as authority_probe,
            ):
                report = readiness.produce_campaign_storage_capacity_report(
                    campaign,
                    output_directory=bundle,
                    target_filesystem=target_directory,
                    archive_tool_authority_path=authority_path,
                    expected_archive_tool_authority_sha256=authority_sha256,
                )
                captured_reconstruction = (
                    readiness.verify_campaign_storage_capacity_report(
                        bundle / "report.json",
                        campaign,
                        None,
                    )
                )
                self.assertEqual(storage_probe.call_count, 1)
                self.assertEqual(authority_probe.call_count, 1)
                reconstructed = readiness.live_verify_campaign_storage_capacity_report(
                    bundle / "report.json",
                    campaign,
                    None,
                )

            self.assertEqual(captured_reconstruction["facts"], report["facts"])
            self.assertEqual(reconstructed["facts"], report["facts"])
            self.assertEqual(storage_probe.call_count, 2)
            self.assertEqual(authority_probe.call_count, 2)
            self.assertEqual(report["facts"]["target_filesystem_free_bytes"], free)
            self.assertEqual(
                report["facts"]["target_filesystem_path"],
                target_directory,
            )
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    archive_authority,
                    "load_archive_tool_authority",
                    return_value=authority_value,
                ),
                patch.object(
                    qualification,
                    "measure_campaign_storage",
                    side_effect=RuntimeError("fresh storage probe unavailable"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "fresh storage probe unavailable",
                ),
            ):
                readiness.live_verify_campaign_storage_capacity_report(
                    bundle / "report.json",
                    campaign,
                    None,
                )
            raw_ssh_config = _copy_bundle(
                bundle,
                root / "storage-raw-ssh-config",
            )
            raw_ssh_config_path = raw_ssh_config / "raw" / "storage-measurement.json"
            raw_ssh_config_measurement = json.loads(
                raw_ssh_config_path.read_text(encoding="utf-8")
            )
            raw_ssh_config_measurement["ssh_config_receipt"]["stdout"] = (
                ssh_config_stdout
            )
            _rewrite_raw_and_resign_bundle(
                raw_ssh_config,
                role="storage_measurement",
                payload=raw_ssh_config_measurement,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "effective configuration drifted",
            ):
                readiness.verify_campaign_storage_capacity_report(
                    raw_ssh_config / "report.json",
                    campaign,
                    None,
                )

            raw_ssh_verbose = _copy_bundle(
                bundle,
                root / "storage-raw-ssh-verbose",
            )
            raw_ssh_verbose_path = raw_ssh_verbose / "raw" / "storage-measurement.json"
            raw_ssh_verbose_measurement = json.loads(
                raw_ssh_verbose_path.read_text(encoding="utf-8")
            )
            raw_ssh_verbose_measurement["remote_probe_receipt"]["stderr"] = (
                ssh_verbose_stderr
            )
            _rewrite_raw_and_resign_bundle(
                raw_ssh_verbose,
                role="storage_measurement",
                payload=raw_ssh_verbose_measurement,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "remote filesystem command drifted",
            ):
                readiness.verify_campaign_storage_capacity_report(
                    raw_ssh_verbose / "report.json",
                    campaign,
                    None,
                )
            tampered = _copy_bundle(bundle, root / "storage-tampered")
            storage_path = tampered / "raw" / "storage-measurement.json"
            storage = json.loads(storage_path.read_text(encoding="utf-8"))
            storage["drive_receipt"]["returncode"] = 1
            _rewrite_raw_and_resign_bundle(
                tampered,
                role="storage_measurement",
                payload=storage,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "Drive measurement command failed",
            ):
                readiness.verify_campaign_storage_capacity_report(
                    tampered / "report.json",
                    campaign,
                    None,
                )

            local_substitute = _copy_bundle(
                bundle,
                root / "storage-local-substitute",
            )
            local_measurement_path = (
                local_substitute / "raw" / "storage-measurement.json"
            )
            local_measurement = json.loads(
                local_measurement_path.read_text(encoding="utf-8")
            )
            local_measurement["measurement_location"] = "local_calling_host_filesystem"
            _rewrite_raw_and_resign_bundle(
                local_substitute,
                role="storage_measurement",
                payload=local_measurement,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "measurement contract drifted",
            ):
                readiness.verify_campaign_storage_capacity_report(
                    local_substitute / "report.json",
                    campaign,
                    None,
                )

            replaced_ssh = _copy_bundle(bundle, root / "storage-replaced-ssh")
            replaced_measurement_path = (
                replaced_ssh / "raw" / "storage-measurement.json"
            )
            replaced_measurement = json.loads(
                replaced_measurement_path.read_text(encoding="utf-8")
            )
            replaced_remote = json.loads(
                replaced_measurement["remote_probe_receipt"]["stdout"]
            )
            replaced_remote["remote_hostname"] = "local mac"
            replaced_measurement["remote_probe_receipt"]["stdout"] = json.dumps(
                replaced_remote,
                separators=(",", ":"),
                sort_keys=True,
            )
            _rewrite_raw_and_resign_bundle(
                replaced_ssh,
                role="storage_measurement",
                payload=replaced_measurement,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "remote hostname must be one bounded host identifier",
            ):
                readiness.verify_campaign_storage_capacity_report(
                    replaced_ssh / "report.json",
                    campaign,
                    None,
                )

            replaced_authority = _copy_bundle(
                bundle,
                root / "storage-replaced-authority",
            )
            replacement_payload = json.loads(json.dumps(authority_payload))
            replacement_payload["endpoint"]["ssh_connection"]["address"] = "10.0.0.2"
            _rewrite_raw_and_resign_bundle(
                replaced_authority,
                role="archive_tool_authority",
                payload=replacement_payload,
            )
            replacement_authority_bytes = (
                replaced_authority / "raw" / "archive-tool-authority.json"
            ).read_bytes()
            replacement_authority_sha256 = hashlib.sha256(
                replacement_authority_bytes
            ).hexdigest()
            replacement_measurement_path = (
                replaced_authority / "raw" / "storage-measurement.json"
            )
            replacement_measurement = json.loads(
                replacement_measurement_path.read_text(encoding="utf-8")
            )
            replacement_measurement["archive_authority_sha256"] = (
                replacement_authority_sha256
            )
            replacement_measurement["ssh_connection"]["address"] = "10.0.0.2"
            _rewrite_raw_and_resign_bundle(
                replaced_authority,
                role="storage_measurement",
                payload=replacement_measurement,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "archive authority source drifted",
            ):
                readiness.verify_campaign_storage_capacity_report(
                    replaced_authority / "report.json",
                    campaign,
                    None,
                )

    def test_output_lock_runs_real_process_contention_and_rejects_forgery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bundle = root / "output-lock"
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    qualification,
                    "run_output_lock_probe",
                    wraps=qualification.run_output_lock_probe,
                ) as lock_probe,
                patch.object(
                    readiness,
                    "_utc_now",
                    return_value=readiness._parse_utc(
                        _PRODUCED_AT,
                        field="test.produced_at",
                    ),
                ),
            ):
                report = readiness.produce_output_lock_contention_report(
                    self.campaign,
                    output_directory=bundle,
                )
                captured_reconstruction = (
                    readiness.verify_output_lock_contention_report(
                        bundle / "report.json",
                        self.campaign,
                        None,
                    )
                )
                self.assertEqual(lock_probe.call_count, 1)
                reconstructed = readiness.live_verify_output_lock_contention_report(
                    bundle / "report.json",
                    self.campaign,
                    None,
                )

            self.assertEqual(captured_reconstruction["facts"], report["facts"])
            self.assertEqual(reconstructed["facts"], report["facts"])
            self.assertEqual(lock_probe.call_count, 2)
            first, second = json.loads(
                (bundle / "raw" / "lock-probe.json").read_text(encoding="utf-8")
            )["concurrent_contenders"]
            self.assertEqual(first["event"]["outcome"], "admitted")
            self.assertEqual(second["event"]["outcome"], "rejected")
            with (
                patch.object(phase_a, "_require_live_source"),
                patch.object(
                    qualification,
                    "run_output_lock_probe",
                    side_effect=RuntimeError("fresh lock probe unavailable"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "fresh lock probe unavailable",
                ),
            ):
                readiness.live_verify_output_lock_contention_report(
                    bundle / "report.json",
                    self.campaign,
                    None,
                )

            tampered = _copy_bundle(bundle, root / "output-lock-tampered")
            probe_path = tampered / "raw" / "lock-probe.json"
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            probe["concurrent_contenders"][1]["event"]["error"] = "forged rejection"
            _rewrite_raw_and_resign_bundle(
                tampered,
                role="lock_probe",
                payload=probe,
            )
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "causal contract",
            ):
                readiness.verify_output_lock_contention_report(
                    tampered / "report.json",
                    self.campaign,
                    None,
                )

    def test_output_lock_probe_cleans_holder_and_restores_identity_on_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            lock = root / "campaign.lock"
            real_start = qualification._start_lock_worker
            holder: list[subprocess.Popen[bytes]] = []

            def start_then_fail(
                lock_path: Path,
                *,
                campaign_id: str,
                source_git_sha: str,
                hold_seconds: float,
            ) -> subprocess.Popen[bytes]:
                if holder:
                    raise RuntimeError("injected second contender failure")
                process = real_start(
                    lock_path,
                    campaign_id=campaign_id,
                    source_git_sha=source_git_sha,
                    hold_seconds=hold_seconds,
                )
                holder.append(process)
                return process

            with (
                patch.object(
                    qualification,
                    "_start_lock_worker",
                    side_effect=start_then_fail,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "injected second contender failure",
                ),
            ):
                qualification.run_output_lock_probe(
                    lock,
                    campaign_id="phase-a-lock-error",
                    source_git_sha="a" * 40,
                )

            self.assertIsNotNone(holder[0].poll())
            self.assertIsNotNone(holder[0].stdout)
            self.assertIsNotNone(holder[0].stderr)
            self.assertTrue(holder[0].stdout.closed)
            self.assertTrue(holder[0].stderr.closed)
            self.assertEqual(
                lock.read_bytes(),
                qualification.canonical_json_bytes(
                    {
                        "campaign_id": "phase-a-lock-error",
                        "schema_version": ("open_ecology_campaign_storage_lock_v1"),
                        "source_git_sha": "a" * 40,
                    }
                ),
            )


def _report_paths(
    root: Path,
    campaign: dict[str, object],
) -> tuple[dict[str, dict[str, Path]], dict[str, Path]]:
    dependency_reports: dict[str, dict[str, Path]] = {}
    for dependency_id, kinds in readiness.DEPENDENCY_EVIDENCE_KINDS.items():
        dependency_reports[dependency_id] = {}
        for kind in kinds:
            path = root / kind / "report.json"
            _write_report(path, campaign=campaign, kind=kind)
            dependency_reports[dependency_id][kind] = path
    gate_reports: dict[str, Path] = {}
    for gate_name, kind in readiness.OPERATIONAL_EVIDENCE_KINDS.items():
        path = root / kind / "report.json"
        _write_report(path, campaign=campaign, kind=kind)
        gate_reports[gate_name] = path
    return dependency_reports, gate_reports


def _write_report(
    path: Path,
    *,
    campaign: dict[str, object],
    kind: str,
) -> None:
    path.parent.mkdir()
    raw_root = path.parent / "raw"
    raw_root.mkdir()
    placeholder = raw_root / "placeholder.json"
    _write_json(placeholder, {"test_only": True})
    manifest = readiness.build_open_ecology_raw_evidence_manifest(
        campaign,
        evidence_kind=kind,
        bundle_root=path.parent,
        produced_at_utc=_PRODUCED_AT,
        producer_name="test_semantic_report_fixture",
        producer_contract="non_authoritative_test_fixture_v1",
        raw_files={"placeholder": placeholder},
    )
    manifest_path = path.parent / "raw-evidence-manifest.json"
    _write_json(manifest_path, manifest)
    report: dict[str, object] = {
        "schema_version": (
            readiness.OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION
        ),
        "evidence_kind": kind,
        "campaign_digest": campaign["exact_digest"],
        "configuration_sha256": campaign["configuration_sha256"],
        "source": campaign["source"],
        "produced_at_utc": _PRODUCED_AT,
        "raw_evidence_manifest": phase_a._file_reference(
            manifest_path,
            base=path.parent,
        ),
        "facts": _facts(kind, campaign),
    }
    _resign(report)
    _write_json(path, report)


def _facts(kind: str, campaign: dict[str, object]) -> dict[str, object]:
    surfaces = {
        surface: _SHA
        for surface in (
            "observation",
            "action",
            "model_artifact",
            "replay",
            "checkpoint",
            "viewer",
        )
    }
    if kind == "cross_surface_and_self_echo":
        return {
            "field_order_sha256_by_surface": surfaces,
            "token_profile_count_by_surface": {surface: 4 for surface in surfaces},
            "profiles_per_token_by_surface": {surface: 2 for surface in surfaces},
            "action_count_by_surface": {surface: 20 for surface in surfaces},
            "self_echo_case_count": 8,
            "emitter_own_signal_self_projection_nonzero_count": 0,
            "emitter_own_signal_local_patch_projection_nonzero_count": 0,
            "adjacent_receiver_self_projection_positive_count": 8,
            "adjacent_receiver_local_patch_projection_positive_count": 8,
            "signal_emission_success_count": 8,
            "global_signal_field_positive_count": 8,
            "global_signal_emission_event_count": 8,
            "signal_action_identity_mismatch_count": 0,
        }
    if kind == "runtime_genome_and_action_source":
        return {
            "heritable_founder_count": 8,
            "heritable_child_count": 4,
            "parent_lineage_match_count": 4,
            "zero_all_founder_count": 8,
            "zero_all_child_count": 4,
            "zero_all_nonzero_genome_count": 0,
            "birth_state_reset_count": 4,
            "genome_digest_replay_mismatch_count": 0,
            "missing_genome_rejection_count": 1,
            "decision_count": 100,
            "heuristic_action_source_count": 0,
            "parent_pre_birth_nonzero_state_count": 2,
            "exact_child_derivation_mismatch_count": 0,
            "inheritance_bound_violation_count": 0,
        }
    if kind == "critic_gradient_and_density_schedule":
        return {
            "cell_contracts": campaign["architecture"]["critic_gradient_cells"],
            "observed_cell_contracts": campaign["architecture"][
                "critic_gradient_cells"
            ],
            "phase_a_density_schedule": [64],
            "phase_b_density_cycle": [32, 64, 128, 64],
            "shared_value_trunk_gradient_l2": 1.0,
            "stop_gradient_value_trunk_gradient_l2": 0.0,
            "stop_gradient_value_head_gradient_l2": 1.0,
            "actor_loss_shared_trunk_gradient_l2": 1.0,
            "critic_conditioned_genome_sensitivity_min_abs_delta": 0.5,
            "critic_unconditioned_genome_sensitivity_max_abs_delta": 0.0,
            "critic_conditioned_parameter_gradient_l2": 1.0,
        }
    if kind == "fixed_batch_equivalence_and_speed":
        from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
            _d4_proof_seed_contract,
            _sealed_d4_deterministic_cuda_contract,
        )

        return {
            "declared_shape": {
                "worlds": 16,
                "rollout_ticks": 128,
                "initial_agents": 64,
            },
            "batch_capacity": 320,
            "repeat_count": 2,
            "device": "cuda",
            "model_contract_version": campaign["architecture"][
                "model_contract_version"
            ],
            "numeric_kernel": campaign["architecture"]["numeric_kernel"],
            "deterministic_cuda_contract": (_sealed_d4_deterministic_cuda_contract()),
            "proof_seed_contract": _d4_proof_seed_contract(
                shape={
                    "worlds": 16,
                    "rollout_ticks": 128,
                    "initial_agents": 64,
                }
            ),
            "collector_device": "cpu",
            "rollout_workers": campaign["runtime_contract"]["rollout_workers"],
            "torch_version": "test",
            "timing_order": [
                ["scalar", "batched"],
                ["batched", "scalar"],
            ],
            "scalar_timing_sample_count": 2,
            "batched_timing_sample_count": 2,
            "reference_semantic_sha256": _SHA,
            "core_scalar_semantic_sha256": _SHA,
            "core_batched_semantic_sha256": _SHA,
            "scalar_semantic_sha256": _SHA,
            "scalar_repeat_semantic_sha256": [_SHA, _SHA],
            "scalar_repeat_collector_path_sha256": [_SHA, _SHA],
            "batched_semantic_sha256": _SHA,
            "batched_repeat_semantic_sha256": [_SHA, _SHA],
            "batched_repeat_collector_path_sha256": ["d" * 64, "d" * 64],
            "ordered_merge_semantic_sha256": _SHA,
            "collector_transition_count": 1_000,
            "collector_batched_transition_count": 1_000,
            "collector_semantic_mismatch_count": 0,
            "collector_paired_transition_count": 1_000,
            "collector_numeric_transition_comparison_count": 1_000,
            "collector_hidden_component_comparison_count": 256_000,
            "collector_bootstrap_value_comparison_count": 100,
            "collector_identity_mismatch_count": 0,
            "collector_hidden_shape_mismatch_count": 0,
            "collector_bootstrap_none_mismatch_count": 0,
            "collector_max_abs_input_hidden_error": 0.0,
            "collector_max_abs_logprob_error": 0.0,
            "collector_max_abs_entropy_error": 0.0,
            "collector_max_abs_value_error": 0.0,
            "collector_max_abs_bootstrap_value_error": 0.0,
            "collector_max_input_hidden_tolerance_ratio": 0.0,
            "collector_max_logprob_tolerance_ratio": 0.0,
            "collector_max_entropy_tolerance_ratio": 0.0,
            "collector_max_value_tolerance_ratio": 0.0,
            "collector_max_bootstrap_value_tolerance_ratio": 0.0,
            "scalar_fixed_batch_step_count": 0,
            "batched_fixed_batch_step_count": 1_000,
            "batched_fixed_batch_runtime_sha256": [_SHA],
            "fixed_batch_execution_buckets": [
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                320,
            ],
            "batched_fixed_batch_call_count": 100,
            "batched_fixed_batch_execution_bucket_histogram": {
                "1": 0,
                "2": 0,
                "4": 0,
                "8": 0,
                "16": 0,
                "32": 0,
                "64": 100,
                "128": 0,
                "256": 0,
                "320": 0,
            },
            "batched_fixed_batch_active_row_slots": 5_000,
            "batched_fixed_batch_execution_row_slots": 6_400,
            "batched_fixed_batch_slot_utilization": 0.78125,
            "batched_fixed_batch_topology_mismatch_count": 0,
            "bounded_bucket_numeric_matrix": {
                "cases": _D4_BUCKET_MATRIX_CASES,
                "genome_conditioning_mode": "actor_film_v1",
                "genome_conditioned_row_count": (_D4_BUCKET_MATRIX_COMPARISON_COUNT),
                "comparison_count": _D4_BUCKET_MATRIX_COMPARISON_COUNT,
                "reference_comparison_count": (_D4_BUCKET_MATRIX_COMPARISON_COUNT),
                "action_mismatch_count": 0,
                "scalar_reference_action_mismatch_count": 0,
                "batched_reference_action_mismatch_count": 0,
                "distinct_action_mask_count": 2,
                "distinct_feedback_vector_count": 2,
                "nonzero_feedback_row_count": (_D4_BUCKET_MATRIX_COMPARISON_COUNT),
                "max_abs_logit_error": 0.0,
                "max_abs_value_error": 0.0,
                "max_abs_hidden_error": 0.0,
                "max_logit_tolerance_ratio": 0.0,
                "max_value_tolerance_ratio": 0.0,
                "max_hidden_tolerance_ratio": 0.0,
                "max_scalar_reference_logit_tolerance_ratio": 0.0,
                "max_scalar_reference_value_tolerance_ratio": 0.0,
                "max_scalar_reference_hidden_tolerance_ratio": 0.0,
                "max_batched_reference_logit_tolerance_ratio": 0.0,
                "max_batched_reference_value_tolerance_ratio": 0.0,
                "max_batched_reference_hidden_tolerance_ratio": 0.0,
                "scalar_semantic_sha256": _SHA,
                "batched_semantic_sha256": _SHA,
                "reference_semantic_sha256": _SHA,
                "action_mask_input_sha256": "a" * 64,
                "feedback_input_sha256": "b" * 64,
            },
            "numeric_comparison_count": 16 * 128 * 64,
            "reference_numeric_comparison_count": 16 * 128 * 64,
            "action_mismatch_count": 0,
            "scalar_reference_action_mismatch_count": 0,
            "batched_reference_action_mismatch_count": 0,
            "max_abs_logit_error": 0.0,
            "max_abs_value_error": 0.0,
            "max_abs_hidden_error": 0.0,
            "max_logit_tolerance_ratio": 0.0,
            "max_value_tolerance_ratio": 0.0,
            "max_hidden_tolerance_ratio": 0.0,
            "max_scalar_reference_logit_tolerance_ratio": 0.0,
            "max_scalar_reference_value_tolerance_ratio": 0.0,
            "max_scalar_reference_hidden_tolerance_ratio": 0.0,
            "max_batched_reference_logit_tolerance_ratio": 0.0,
            "max_batched_reference_value_tolerance_ratio": 0.0,
            "max_batched_reference_hidden_tolerance_ratio": 0.0,
            "scalar_median_elapsed_ns": 20,
            "batched_median_elapsed_ns": 10,
        }
    if kind == "persistent_island_50000":
        return {
            "scheduled_task_count": 48,
            "completed_proof_task_count": 3,
            "terminal_tick": 50_000,
            "same_world_identity_count": 3,
            "episode_reset_count": 0,
            "backbone_initial_sha256": _SHA,
            "backbone_final_sha256": _SHA,
            "tick_10000_report_count": 3,
            "resource_sample_count": 500,
            "arms": ["H", "Z", "R"],
            "proof_scope": "engineering_runner_endurance_only",
            "proof_seed_role": "open_ecology_proof",
            "proof_artifact_source": (
                "source_initialized_frozen_width256_actor_film_v1"
            ),
            "phase_a_artifact_access_count": 0,
            "phase_b_artifact_access_count": 0,
            "scientific_selection_seed_access_count": 0,
            "scientific_outcome_use_count": 0,
        }
    if kind == "checkpoint_continuation_equivalence":
        return {
            "arms": ["H", "Z", "R"],
            "checkpoint_tick": 5_000,
            "continuation_ticks": 100,
            "component_schema_versions": {
                "world_state": "v1",
                "environment_rng_state": "v1",
                "recurrent_policy_state": "v1",
                "public_feedback_history": "v1",
                "sampling_rng_state": "v1",
                "genome_population_snapshot": "v1",
                "evidence_writer_continuation_state": "v1",
            },
            "world_state_match": True,
            "action_stream_match": True,
            "environment_rng_draw_match": True,
            "sampling_rng_draw_match": True,
            "hidden_state_match": True,
            "public_feedback_history_match": True,
            "genome_population_match": True,
            "evidence_digest_match": True,
        }
    if kind == "bounded_writer_event_coverage":
        event_kinds = [
            "birth",
            "death",
            "lineage",
            "dyadic_interaction",
            "signal_contributor",
            "congestion",
            "intervention",
        ]
        return {
            "completed_shard_count": 2,
            "resumable_prefix_match": True,
            "event_kinds_observed": event_kinds,
            "event_count_by_kind": {event_kind: 1 for event_kind in event_kinds},
            "replay_rows_within_bound": True,
            "replay_bytes_within_bound": True,
            "campaign_quota_enforced": True,
            "active_writer_archive_rejection_count": 1,
        }
    if kind == "causal_evaluator_rejection_battery":
        return {
            "real_frozen_policy_state_count": 10,
            "controlled_action_mask_count": 10,
            "controlled_sampling_rng_count": 10,
            "genome_intervention_count": 10,
            "signal_intervention_count": 10,
            "entropy_only_rejection_count": 1,
            "correlation_only_rejection_count": 1,
            "self_echo_rejection_count": 1,
            "missing_contributor_rejection_count": 1,
            "pseudo_replication_rejection_count": 1,
        }
    if kind == "capture_noninterference_reexecution":
        return {
            "cell_order": ["A0", "A1", "A2", "A3"],
            "density_levels": [32, 64, 128],
            "case_count": 12,
            "ticks_per_case": 128,
            "birth_count": 5,
            "death_count": 901,
            "primary_report_sha256": _SHA,
            "reexecution_report_sha256": _SHA,
            "report_bytes_match": True,
            "scientific_selection_seed_access_count": 0,
            "model_state_mismatch_count": 0,
            "trajectory_mismatch_count": 0,
            "action_rng_mismatch_count": 0,
            "genome_provenance_mismatch_count": 0,
        }
    if kind == "preregistration_roundtrip_fail_closed":
        run_matrix = campaign["training"]["run_matrix"]
        return {
            "roundtrip_exact_digest": campaign["exact_digest"],
            "run_matrix_row_count": len(run_matrix),
            "unique_task_id_count": len(run_matrix),
            "seed_registry_sha256": campaign["seed_contract"]["registry_sha256"],
            "configuration_sha256": campaign["configuration_sha256"],
            "unknown_input_rejection_count": 1,
            "dirty_source_rejection_count": 1,
            "stale_source_rejection_count": 1,
            "duplicate_key_rejection_count": 1,
        }
    if kind == "exact_sha_phase_a_training_and_torch_ci":
        return {
            "fresh_checkout_count": 1,
            "dirty_checkout_count": 0,
            "source_manifest_match_count": 1,
            "import_root_match_count": 1,
            "intended_training_host_classes": ["host-a"],
            "tested_training_host_classes": ["host-a"],
            "full_shape_training_benchmark_count_by_host_class": {
                "host-a": 3,
            },
            "runtime_contract_sha256_by_host_class": {
                "host-a": "d" * 64,
            },
            "semantic_evidence_sha256_by_host_class": {
                "host-a": _SHA,
            },
            "cross_host_semantic_mismatch_count": 0,
            "torch_ci_lane_required": True,
            "torch_ci_test_count": 10,
            "torch_ci_failure_count": 0,
            "torch_ci_skip_count": 0,
        }
    if kind == "phase_d_host_equivalence_or_homogeneous_contract":
        return {
            "host_contract_mode": "single_homogeneous_host_class",
            "intended_host_class_count": 1,
            "tested_host_class_count": 1,
            "runtime_contract_sha256_by_host_class": {
                "host-a": "d" * 64,
            },
            "semantic_evidence_sha256_by_host_class": {
                "host-a": _SHA,
            },
            "cross_host_semantic_mismatch_count": 0,
        }
    if kind == "phase_a_training_throughput":
        return {
            "preliminary_throughput_gate_digest": campaign["throughput_gate"][
                "exact_digest"
            ],
            "measured_host_class": "host-a",
            "fresh_full_shape_repetition_count": 3,
            "phase_a_training_projection_authoritative": True,
            "selection_projection_included": False,
            "semantic_mismatch_count": 0,
            "projected_phase_a_training_seconds": 60 * 60,
            "maximum_gpu_memory_share": 0.5,
            "maximum_host_ram_share": 0.5,
            "swap_delta_bytes": 0,
            "temperature_below_throttle": True,
            "xid_count": 0,
            "oom_count": 0,
            "nonfinite_count": 0,
        }
    if kind == "campaign_storage_capacity":
        capacity = 1_000 * 1024**3
        return {
            "checked_at_utc": _PRODUCED_AT,
            "measurement_location": "remote_ssh_target_with_local_drive_v1",
            "target_ssh": "trainer-node",
            "remote_hostname": "trainer-host",
            "remote_boot_id": "87350a1b-e992-490b-bcd5-5e95ac29dc89",
            "archive_authority_sha256": campaign["source"][
                "archive_tool_authority_sha256"
            ],
            "ssh_effective_config_sha256": _SHA,
            "ssh_verbose_stderr_sha256": _SHA,
            "ssh_connection": {
                "address": "10.0.0.1",
                "authenticated_host": "trainer-node.example.invalid",
                "authentication": "publickey",
                "host_key": "ssh-ed25519 SHA256:" + "A" * 43,
                "port": 22,
            },
            "rclone_config_path": "/Users/test/.config/rclone/rclone.conf",
            "rclone_config_sha256": _SHA,
            "google_drive_free_bytes": 400 * 1024**3,
            "projected_active_storage_bytes": 100 * 1024**3,
            "target_filesystem_path": "/srv/evolution-sim/runs",
            "target_filesystem_capacity_bytes": capacity,
            "target_filesystem_free_bytes": 300 * 1024**3,
            "required_target_free_bytes": 200 * 1024**3,
            "drive_measurement_command_sha256": _SHA,
            "filesystem_measurement_command_sha256": _SHA,
        }
    if kind == "output_lock_contention":
        return {
            "concurrent_contender_count": 2,
            "admitted_writer_count": 1,
            "contention_rejection_count": 1,
            "identity_drift_rejection_count": 1,
            "lock_release_reacquire_count": 1,
        }
    if kind == "immutable_drive_uploader":
        return {
            "closed_bundle_count": 1,
            "active_bundle_rejection_count": 1,
            "remote_object_names": [
                "bundle.tar.zst",
                "bundle.manifest.json",
                "bundle.sha256",
            ],
            "remote_unique_id_count": 3,
            "stream_readback_sha256_match_count": 3,
            "rclone_check_match_count": 3,
            "rclone_check_difference_count": 0,
            "symlink_rejection_count": 1,
            "hardlink_rejection_count": 1,
            "special_file_rejection_count": 1,
            "mount_crossing_rejection_count": 1,
            "source_bundle_retained_after_verification": True,
            "local_pruning_supported": False,
            "local_prune_attempt_count": 0,
        }
    if kind == "terminal_aggregate_validator":
        return {
            "canonical_terminal_count": 16,
            "validated_terminal_count": 16,
            "missing_terminal_rejection_count": 1,
            "surplus_terminal_rejection_count": 1,
            "digest_tamper_rejection_count": 1,
            "source_drift_rejection_count": 1,
            "prefix_drift_rejection_count": 1,
            "fresh_exact_cpu_reexecution_count": 16,
            "proof_scope": "engineering_terminal_matrix_fixture_only",
            "scientific_terminal_access_count": 0,
        }
    raise AssertionError(f"missing test facts for {kind}")


def _resign(payload: dict[str, object]) -> None:
    payload.pop("exact_digest", None)
    payload["exact_digest"] = stable_payload_digest(payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _copy_bundle(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _qualification_source_manifest(
    campaign: dict[str, object],
) -> dict[str, object]:
    init_path = (
        Path(readiness.__file__).resolve().parents[3]
        / "python"
        / "evolution_sim"
        / "__init__.py"
    )
    return {
        "hash_algorithm": "sha256",
        "path_contract": (
            "repository_relative_sorted_runtime_python_package_and_mind_requirements"
        ),
        "file_count": 1,
        "files": {
            "python/evolution_sim/__init__.py": hashlib.sha256(
                init_path.read_bytes()
            ).hexdigest()
        },
        "aggregate_sha256": campaign["source"]["manifest_sha256"],
    }


def _qualification_command_receipt(
    argv: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": qualification.COMMAND_RECEIPT_SCHEMA_VERSION,
        "argv": argv,
        "executable": {
            "device": 1,
            "inode": 2,
            "size": 3,
            "sha256": _SHA,
        },
        "started_at_utc": _PRODUCED_AT,
        "finished_at_utc": _PRODUCED_AT,
        "elapsed_ns": 1,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _qualification_source_observation(
    campaign: dict[str, object],
    *,
    source_manifest: dict[str, object],
) -> dict[str, object]:
    source_commit = campaign["source"]["commit"]
    import_result = {
        "module_relative_path": "python/evolution_sim/__init__.py",
        "module_sha256": source_manifest["files"]["python/evolution_sim/__init__.py"],
        "python_version": "3.test",
    }
    return {
        "schema_version": qualification.SOURCE_OBSERVATION_SCHEMA_VERSION,
        "head": _qualification_command_receipt(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            stdout=f"{source_commit}\n",
        ),
        "status": _qualification_command_receipt(
            [
                "/usr/bin/git",
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ]
        ),
        "source_manifest": source_manifest,
        "import_probe": _qualification_command_receipt(
            [sys.executable, "-c", qualification._IMPORT_PROBE_CODE],
            stdout=json.dumps(import_result, separators=(",", ":"), sort_keys=True)
            + "\n",
        ),
        "import_probe_code_sha256": qualification.import_probe_code_sha256(),
    }


def _qualification_github_receipt(
    campaign: dict[str, object],
) -> dict[str, object]:
    source_commit = campaign["source"]["commit"]
    endpoint = (
        f"repos/{qualification.GITHUB_REPOSITORY}/commits/{source_commit}/"
        "check-runs?per_page=100"
    )
    return _qualification_command_receipt(
        [
            "/opt/homebrew/bin/gh",
            "api",
            "--method",
            "GET",
            "-H",
            "Accept: application/vnd.github+json",
            endpoint,
        ],
        stdout=json.dumps(
            {
                "total_count": 1,
                "check_runs": [
                    {
                        "id": 10,
                        "name": qualification.GITHUB_TORCH_CHECK_NAME,
                        "head_sha": source_commit,
                        "status": "completed",
                        "conclusion": "success",
                        "app": {"slug": "github-actions"},
                    }
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _qualification_torch_report(
    campaign: dict[str, object],
) -> dict[str, object]:
    test_source = (
        Path(readiness.__file__).resolve().parents[3]
        / "python"
        / "tests"
        / "test_mind_v1.py"
    ).read_bytes()
    selected = list(readiness._discover_torch_gated_tests(test_source))
    return {
        "schema_version": "mind_torch_validation_v1",
        "validation_policy": "explicit_optional_mind_ml_stack_validation_v1",
        "require_cuda": True,
        "runtime_environment": {
            "git_dirty": False,
            "git_head": str(campaign["source"]["commit"])[:7],
        },
        "dependency_versions": {"torch": "2.11.0"},
        "torch_environment": {
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "RTX 4070",
        },
        "unittest": {
            "test_module": "tests.test_mind_v1",
            "test_class": "MindV1Tests",
            "test_count": len(selected),
            "selected_tests": selected,
            "seconds": 1.0,
            "passed": True,
            "failure_count": 0,
            "error_count": 0,
            "skip_count": 0,
            "failures": [],
            "errors": [],
            "skips": [],
            "runner_output": "OK",
        },
        "cuda_training_smoke": {
            "passed": True,
            "requested_device": "cuda",
            "resolved_device": "cuda",
            "cuda_available": True,
            "cuda_device_count": 1,
        },
        "passed": True,
    }


def _resource_sample(
    mode: str,
    *,
    gpu_share: float,
    ram_share: float,
) -> dict[str, object]:
    return {
        "population_mode": mode,
        "sampled_at_utc": _PRODUCED_AT,
        "host": {
            "boot_id": "test-boot",
            "memory": {
                "ram_used_share": ram_share,
                "swap_used_bytes": 0,
            },
            "gpus": [
                {
                    "memory_used_share": gpu_share,
                    "temperature_c": 60,
                    "slowdown_temperature_c": 90,
                }
            ],
            "xid_count": 0,
            "oom_count": 0,
        },
    }


def _rewrite_raw_and_resign_bundle(
    bundle: Path,
    *,
    role: str,
    payload: dict[str, object],
) -> None:
    manifest_path = bundle / "raw-evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["role"] == role)
    raw_path = bundle / entry["relative_path"]
    _write_json(raw_path, payload)
    raw_bytes = raw_path.read_bytes()
    entry["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    entry["byte_length"] = len(raw_bytes)
    _resign(manifest)
    _write_json(manifest_path, manifest)
    report_path = bundle / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["raw_evidence_manifest"] = phase_a._file_reference(
        manifest_path,
        base=bundle,
    )
    _resign(report)
    _write_json(report_path, report)


def _rewrite_raw_bytes_and_resign_bundle(
    bundle: Path,
    *,
    role: str,
    payload: bytes,
) -> None:
    manifest_path = bundle / "raw-evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["role"] == role)
    raw_path = bundle / entry["relative_path"]
    raw_path.write_bytes(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["byte_length"] = len(payload)
    _resign(manifest)
    _write_json(manifest_path, manifest)
    report_path = bundle / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["raw_evidence_manifest"] = phase_a._file_reference(
        manifest_path,
        base=bundle,
    )
    _resign(report)
    _write_json(report_path, report)


if __name__ == "__main__":
    unittest.main()
