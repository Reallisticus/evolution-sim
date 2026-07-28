from __future__ import annotations

from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from evolution_sim.io.open_ecology_campaign_storage import canonical_json_bytes
from evolution_sim.mind import open_ecology_phase_a_archive as archive
from evolution_sim.mind.open_ecology_phase_a import phase_a_run_id
from evolution_sim.mind.open_ecology_phase_a_guardian import (
    PhaseAAuthorityBindings,
    _write_compact_transcript,
)
from evolution_sim.mind.open_ecology_phase_a_readiness import (
    OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest


_GIT_SHA = "a" * 40
_MANIFEST_SHA = "b" * 64
_CONFIGURATION_SHA = "c" * 64
_ARCHIVE_AUTHORITY_SHA = "d" * 64


def _signed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["exact_digest"] = stable_payload_digest(result)
    return result


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _file_reference(path: Path, *, base: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "relative_path": str(path.relative_to(base)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
    }


class _PhaseAFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority = root / "authority"
        self.active = root / "active"
        self.closed_parent = root / "closed"
        self.receipts = root / "receipts"
        for path in (
            self.authority,
            self.active,
            self.closed_parent,
            self.receipts,
        ):
            path.mkdir()

        self.runtime_authority = self.authority / "runtime-venv-authority.json"
        self.runtime_authority.write_bytes(b'{"runtime":"sealed"}\n')
        self.runtime_authority_sha = hashlib.sha256(
            self.runtime_authority.read_bytes()
        ).hexdigest()
        self.preregistration = _signed(
            {
                "configuration_sha256": _CONFIGURATION_SHA,
                "source": {
                    "archive_tool_authority_sha256": _ARCHIVE_AUTHORITY_SHA,
                    "commit": _GIT_SHA,
                    "manifest_sha256": _MANIFEST_SHA,
                },
            }
        )
        self.preregistration_path = self.authority / "preregistration.json"
        _write_json(self.preregistration_path, self.preregistration)

        self.report_path = self.authority / "reports" / "proof.json"
        self.report_path.parent.mkdir()
        _write_json(self.report_path, _signed({"proof": "sealed"}))
        self.guardian_report_paths: dict[str, Path] = {}
        guardian_evidence_kinds = {
            "d10": "exact_sha_phase_a_training_and_torch_ci",
            "throughput": "phase_a_training_throughput",
            "storage": "campaign_storage_capacity",
            "output_lock": "output_lock_contention",
        }
        self.verifier_payloads: dict[str, dict[str, object]] = {}
        for gate, evidence_kind in guardian_evidence_kinds.items():
            report_path = self.report_path.parent / f"{gate}.json"
            _write_json(
                report_path,
                _signed({"evidence_kind": evidence_kind, "proof": "sealed"}),
            )
            self.guardian_report_paths[gate] = report_path
            self.verifier_payloads[gate] = {
                "evidence_kind": evidence_kind,
                "facts": {"fixture": gate},
            }
        self.index = _signed(
            {
                "schema_version": (OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION),
                "campaign_digest": self.preregistration["exact_digest"],
                "configuration_sha256": _CONFIGURATION_SHA,
                "source": self.preregistration["source"],
                "dependency_reports": [
                    {
                        "dependency_id": "readiness_dependency_01",
                        "reports": [
                            {
                                "evidence_kind": "test-proof",
                                "file": _file_reference(
                                    self.report_path,
                                    base=self.authority,
                                ),
                            }
                        ],
                    },
                    {
                        "dependency_id": "readiness_dependency_10",
                        "reports": [
                            {
                                "evidence_kind": (
                                    "exact_sha_phase_a_training_and_torch_ci"
                                ),
                                "file": _file_reference(
                                    self.guardian_report_paths["d10"],
                                    base=self.authority,
                                ),
                            }
                        ],
                    },
                ],
                "operational_reports": [
                    {
                        "gate": gate,
                        "report": {
                            "evidence_kind": guardian_evidence_kinds[gate],
                            "file": _file_reference(
                                self.guardian_report_paths[gate],
                                base=self.authority,
                            ),
                        },
                    }
                    for gate in ("throughput", "storage", "output_lock")
                ],
            }
        )
        self.index_path = self.authority / "evidence-index.json"
        _write_json(self.index_path, self.index)
        self.authorization = _signed(
            {
                "schema_version": (
                    archive.OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
                ),
                "authorized_at_utc": "2026-07-28T00:00:00Z",
                "campaign_digest": self.preregistration["exact_digest"],
                "configuration_sha256": _CONFIGURATION_SHA,
                "source": self.preregistration["source"],
                "evidence_index": {
                    "exact_digest": self.index["exact_digest"],
                    "file": _file_reference(
                        self.index_path,
                        base=self.authority,
                    ),
                },
                "readiness_dependencies": [],
                "operational_gates": {
                    "storage": {
                        "target_ssh": "trainer-node",
                        "ssh_connection": {
                            "address": "192.0.2.10",
                            "authenticated_host": "trainer-node.example.invalid",
                            "authentication": "publickey",
                            "host_key": "SHA256:test-host-key",
                            "port": 22,
                        },
                    }
                },
                "authorization": {
                    "phase_a_training_authorized": True,
                    "authorization_basis": (
                        "stage_gated_phase_a_training_dependencies_and_operations_v2"
                    ),
                    "authorization_scope": "phase_a_training_only",
                    "phase_a_selection_authorized": False,
                    "phase_b_authorized": False,
                    "phase_c_authorized": False,
                    "phase_d_authorized": False,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                },
            }
        )
        self.authorization_path = self.authority / "launch-authorization.json"
        _write_json(self.authorization_path, self.authorization)

        self.terminal_digests: dict[str, str] = {}
        completed: list[dict[str, object]] = []
        for cell_id, learner_index in archive._canonical_matrix():
            run_id = phase_a_run_id(
                cell_id=cell_id,
                learner_index=learner_index,
            )
            run = self.active / run_id
            (run / "terminal").mkdir(parents=True)
            (run / ".run.lock").write_bytes(b"")
            (run / "terminal" / "terminal.json").write_bytes(b"{}\n")
            digest = hashlib.sha256(f"terminal:{run_id}".encode()).hexdigest()
            self.terminal_digests[run_id] = digest
            completed.append(
                {
                    "cell_id": cell_id,
                    "exact_digest": digest,
                    "learner_index": learner_index,
                    "run_id": run_id,
                }
            )
        self.transcript_path = self.authority / "guardian-transcript.json"
        self.transcript = _write_compact_transcript(
            self.transcript_path,
            bindings=PhaseAAuthorityBindings(
                evidence_index_digest=self.index["exact_digest"],
                launch_authorization_digest=self.authorization["exact_digest"],
                preregistration_digest=self.preregistration["exact_digest"],
                runtime_venv_authority_sha256=self.runtime_authority_sha,
                source_git_sha=_GIT_SHA,
                source_manifest_sha256=_MANIFEST_SHA,
                ssh_target="trainer-node",
            ),
            endpoint={
                "address": "192.0.2.10",
                "authenticated_host": "trainer-node.example.invalid",
                "authentication": "publickey",
                "host_key": "SHA256:test-host-key",
                "port": 22,
            },
            coordinator_pgid=100,
            coordinator_pid=101,
            remote_pgid=200,
            remote_pid=201,
            storage_facts_digest=stable_payload_digest(
                self.verifier_payloads["storage"]
            ),
            remote_verifier_digests={
                gate: stable_payload_digest(self.verifier_payloads[gate])
                for gate in ("d10", "throughput", "output_lock")
            },
            completed=completed,
            complete=True,
        )

    @property
    def bundle(self) -> Path:
        return self.closed_parent / "phase-a-attempt-1"

    @property
    def receipt(self) -> Path:
        return self.receipts / "closure.json"

    def close_arguments(self) -> dict[str, object]:
        return {
            "preregistration_path": self.preregistration_path,
            "launch_authorization_path": self.authorization_path,
            "guardian_transcript_path": self.transcript_path,
            "runtime_venv_authority_path": self.runtime_authority,
            "active_output_root": self.active,
            "authority_root": self.authority,
            "closed_bundle_parent": self.closed_parent,
            "bundle_id": self.bundle.name,
            "closure_receipt_path": self.receipt,
        }

    def verified_request(
        self,
        _preregistration: object,
        *,
        cell_id: str,
        learner_index: int,
        terminal_path: Path,
        evaluation_workers: int,
    ) -> SimpleNamespace:
        self._assert_terminal_call(
            terminal_path,
            cell_id=cell_id,
            learner_index=learner_index,
            evaluation_workers=evaluation_workers,
        )
        run_id = phase_a_run_id(
            cell_id=cell_id,
            learner_index=learner_index,
        )
        return SimpleNamespace(
            training_authority={
                "terminal_exact_digest": self.terminal_digests[run_id],
                "exact_digest": hashlib.sha256(
                    f"authority:{run_id}".encode()
                ).hexdigest(),
                "artifact_sha256": hashlib.sha256(
                    f"artifact:{run_id}".encode()
                ).hexdigest(),
            }
        )

    def _assert_terminal_call(
        self,
        terminal_path: Path,
        *,
        cell_id: str,
        learner_index: int,
        evaluation_workers: int,
    ) -> None:
        expected_run_id = phase_a_run_id(
            cell_id=cell_id,
            learner_index=learner_index,
        )
        if terminal_path.parent.parent.name != expected_run_id:
            raise AssertionError("terminal verifier received a wrong run path")
        if evaluation_workers != 1:
            raise AssertionError("closure verifier changed evaluation workers")

    def fake_seal(
        self,
        active_root: Path,
        bundle_root: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        if active_root != self.active:
            raise AssertionError("sealer received a wrong active root")
        if kwargs != {
            "campaign_id": (
                f"phase-a-{str(self.preregistration['exact_digest'])[:24]}"
            ),
            "bundle_id": self.bundle.name,
            "source_git_sha": _GIT_SHA,
            "source_manifest_sha256": _MANIFEST_SHA,
            "limits": archive.CampaignStorageLimits(),
        }:
            raise AssertionError("sealer authority binding drifted")
        return SimpleNamespace(
            root=bundle_root,
            marker_sha256="5" * 64,
            scan=SimpleNamespace(
                entries=(),
                file_count=0,
                directory_count=0,
                total_file_bytes=0,
            ),
        )

    def validation_patches(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                archive,
                "validate_open_ecology_phase_a_preregistration",
            )
        )
        stack.enter_context(mock.patch.object(archive, "_require_live_source"))
        stack.enter_context(mock.patch.object(archive, "validate_launch_authorization"))
        stack.enter_context(mock.patch.object(archive, "_validate_runtime_authority"))
        for gate, verifier_name in (
            ("d10", "verify_exact_sha_phase_a_training_and_torch_ci_report"),
            ("throughput", "verify_phase_a_training_throughput_report"),
            ("storage", "verify_campaign_storage_capacity_report"),
            ("output_lock", "verify_output_lock_contention_report"),
        ):
            stack.enter_context(
                mock.patch.object(
                    archive,
                    verifier_name,
                    return_value=self.verifier_payloads[gate],
                )
            )
        stack.enter_context(
            mock.patch.object(
                archive,
                "build_verified_phase_a_selection_request",
                side_effect=self.verified_request,
            )
        )
        stack.enter_context(
            mock.patch.object(
                archive,
                "seal_closed_bundle",
                side_effect=self.fake_seal,
            )
        )
        return stack


class OpenEcologyPhaseAArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = _PhaseAFixture(Path(self.temporary.name).resolve())

    def test_complete_matrix_is_copied_sealed_and_receipted_without_deletion(
        self,
    ) -> None:
        with self.fixture.validation_patches():
            result = archive.close_phase_a_terminal_matrix(
                **self.fixture.close_arguments()
            )

        self.assertEqual(result["closed_bundle_root"], str(self.fixture.bundle))
        self.assertTrue(self.fixture.receipt.is_file())
        validated = archive.validate_phase_a_terminal_matrix_closure(
            self.fixture.receipt
        )
        self.assertEqual(validated, result)
        self.assertTrue(self.fixture.active.is_dir())
        self.assertTrue(self.fixture.authority.is_dir())
        self.assertTrue(
            (
                self.fixture.bundle
                / "training"
                / next(iter(self.fixture.terminal_digests))
                / "terminal"
                / "terminal.json"
            ).is_file()
        )
        manifest_envelope = json.loads(
            (
                self.fixture.bundle / archive.PHASE_A_TERMINAL_MATRIX_MANIFEST_NAME
            ).read_text()
        )
        manifest = manifest_envelope["payload"]
        self.assertEqual(len(manifest["terminal_matrix"]), 16)
        self.assertEqual(
            manifest["authority"]["runtime_venv_authority_sha256"],
            self.fixture.runtime_authority_sha,
        )
        self.assertFalse(result["drive_archive_route"]["source_deletion_authorized"])
        self.assertFalse(result["drive_archive_route"]["source_pruning_authorized"])

    def test_incomplete_or_surplus_guardian_transcript_fails_before_copy(
        self,
    ) -> None:
        for mutation in ("incomplete", "surplus"):
            with self.subTest(mutation=mutation):
                fixture = _PhaseAFixture(
                    (Path(self.temporary.name) / mutation).resolve()
                )
                payload = dict(fixture.transcript)
                payload.pop("exact_digest")
                if mutation == "incomplete":
                    payload["completed_cells"] = list(payload["completed_cells"])[:-1]
                    payload["completed_cell_count"] = 15
                    payload["matrix_complete"] = False
                else:
                    payload["unexpected"] = True
                fixture.transcript = _signed(payload)
                _write_json(fixture.transcript_path, fixture.transcript)
                with fixture.validation_patches():
                    with self.assertRaises(archive.OpenEcologyPhaseAArchiveError):
                        archive.close_phase_a_terminal_matrix(
                            **fixture.close_arguments()
                        )
                self.assertFalse(fixture.bundle.exists())
                self.assertFalse(fixture.receipt.exists())

    def test_runtime_authority_and_terminal_digests_are_not_interchangeable(
        self,
    ) -> None:
        for attack, expected_error in (
            ("runtime", "bindings drifted"),
            ("terminal", "differs from a verified terminal chain"),
        ):
            with self.subTest(attack=attack):
                fixture = _PhaseAFixture((Path(self.temporary.name) / attack).resolve())
                payload = dict(fixture.transcript)
                payload.pop("exact_digest")
                if attack == "runtime":
                    bindings = dict(payload["bindings"])
                    bindings["runtime_venv_authority_sha256"] = "9" * 64
                    payload["bindings"] = bindings
                else:
                    payload["completed_cells"] = [
                        ({**row, "exact_digest": "8" * 64} if index == 0 else row)
                        for index, row in enumerate(payload["completed_cells"])
                    ]
                fixture.transcript = _signed(payload)
                _write_json(fixture.transcript_path, fixture.transcript)
                with fixture.validation_patches():
                    with self.assertRaisesRegex(
                        archive.OpenEcologyPhaseAArchiveError,
                        expected_error,
                    ):
                        archive.close_phase_a_terminal_matrix(
                            **fixture.close_arguments()
                        )
                self.assertFalse(fixture.bundle.exists())

    def test_referenced_evidence_byte_drift_fails_before_copy(self) -> None:
        self.fixture.report_path.write_bytes(b'{"tampered":true}\n')
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "file reference bytes drifted",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertFalse(self.fixture.bundle.exists())
        self.assertFalse(self.fixture.receipt.exists())

    def test_unreferenced_authority_secret_fails_before_bundle_or_seal(self) -> None:
        secret = self.fixture.authority / "github-token.txt"
        secret.write_text("not-for-publication", encoding="utf-8")
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "surplus or missing unreferenced",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertFalse(self.fixture.bundle.exists())
        self.assertFalse(self.fixture.receipt.exists())
        self.assertEqual(secret.read_text(encoding="utf-8"), "not-for-publication")

    def test_forged_guardian_verifier_digest_fails_before_copy(self) -> None:
        payload = dict(self.fixture.transcript)
        payload.pop("exact_digest")
        digests = dict(payload["remote_verifier_digests"])
        digests["d10"] = "9" * 64
        payload["remote_verifier_digests"] = digests
        self.fixture.transcript = _signed(payload)
        _write_json(self.fixture.transcript_path, self.fixture.transcript)
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "lacks captured evidence provenance",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertFalse(self.fixture.bundle.exists())
        self.assertFalse(self.fixture.receipt.exists())

    def test_forged_guardian_ssh_endpoint_fails_before_copy(self) -> None:
        payload = dict(self.fixture.transcript)
        payload.pop("exact_digest")
        endpoint = dict(payload["ssh_endpoint"])
        endpoint["address"] = "192.0.2.99"
        payload["ssh_endpoint"] = endpoint
        self.fixture.transcript = _signed(payload)
        _write_json(self.fixture.transcript_path, self.fixture.transcript)
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "differs from launch storage authority",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertFalse(self.fixture.bundle.exists())
        self.assertFalse(self.fixture.receipt.exists())

    def test_runtime_authority_semantics_bind_every_launch_authority(self) -> None:
        expected_connection_digest = stable_payload_digest(
            self.fixture.authorization["operational_gates"]["storage"]["ssh_connection"]
        )
        with mock.patch.object(
            archive,
            "load_runtime_venv_authority",
            return_value={"validated": True},
        ) as loader:
            archive._validate_runtime_authority(
                self.fixture.runtime_authority,
                file_sha256=self.fixture.runtime_authority_sha,
                preregistration=self.fixture.preregistration,
                launch_authorization=self.fixture.authorization,
            )
        loader.assert_called_once_with(
            self.fixture.runtime_authority,
            expected_sha256=self.fixture.runtime_authority_sha,
            expected_source_git_sha=_GIT_SHA,
            expected_source_manifest_sha256=_MANIFEST_SHA,
            expected_archive_authority_sha256=_ARCHIVE_AUTHORITY_SHA,
            expected_ssh_target="trainer-node",
            expected_ssh_connection_sha256=expected_connection_digest,
        )

    def test_surplus_run_or_existing_bundle_is_never_overwritten(self) -> None:
        (self.fixture.active / "surplus-run").mkdir()
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "surplus",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertFalse(self.fixture.bundle.exists())

        self.fixture.active.joinpath("surplus-run").rmdir()
        self.fixture.bundle.mkdir()
        sentinel = self.fixture.bundle / "sentinel"
        sentinel.write_text("preserve")
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "already exists",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_descriptor_copy_rejects_symlink_hardlink_and_fifo(self) -> None:
        for attack in ("symlink", "hardlink", "fifo"):
            with self.subTest(attack=attack):
                fixture = _PhaseAFixture((Path(self.temporary.name) / attack).resolve())
                target_run = fixture.active / next(iter(fixture.terminal_digests))
                if attack == "symlink":
                    (target_run / "escape").symlink_to(fixture.preregistration_path)
                elif attack == "hardlink":
                    original = target_run / "hardlink-original"
                    original.write_bytes(b"same inode")
                    os.link(original, target_run / "hardlink-alias")
                else:
                    os.mkfifo(target_run / "special-fifo")
                with fixture.validation_patches():
                    with self.assertRaisesRegex(
                        archive.OpenEcologyPhaseAArchiveError,
                        "single-link regular files/directories",
                    ):
                        archive.close_phase_a_terminal_matrix(
                            **fixture.close_arguments()
                        )
                self.assertFalse(fixture.receipt.exists())
                self.assertTrue(fixture.active.exists())

    def test_held_run_lock_fails_before_destination_materialization(self) -> None:
        run_id = next(iter(self.fixture.terminal_digests))
        lock_path = self.fixture.active / run_id / ".run.lock"
        descriptor = os.open(lock_path, os.O_RDWR)
        self.addCleanup(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(fcntl.flock, descriptor, fcntl.LOCK_UN)
        with self.fixture.validation_patches():
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "run lock is unavailable",
            ):
                archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        self.assertFalse(self.fixture.bundle.exists())
        self.assertFalse(self.fixture.receipt.exists())

    def test_archive_delegates_exact_closure_to_sealed_uploader(self) -> None:
        with self.fixture.validation_patches():
            closure = archive.close_phase_a_terminal_matrix(
                **self.fixture.close_arguments()
            )
        authority = SimpleNamespace(
            authority_sha256=_ARCHIVE_AUTHORITY_SHA,
            ssh_target="trainer-node",
            source_git_sha=_GIT_SHA,
            source_manifest_sha256=_MANIFEST_SHA,
            remote_repository_root=f"/checkouts/{_GIT_SHA}",
        )
        authority_path = self.fixture.receipts / "archive-authority.json"
        authority_path.write_bytes(b"sealed")
        drive_receipt = self.fixture.receipts / "drive-receipt.json"
        with (
            mock.patch.object(
                archive,
                "load_archive_tool_authority",
                return_value=authority,
            ),
            mock.patch(
                "scripts.archive_open_ecology_campaign.execute_remote_archive",
                return_value={"uploaded": True},
            ) as execute,
        ):
            result = archive.archive_phase_a_terminal_matrix(
                closure_receipt_path=self.fixture.receipt,
                expected_closure_receipt_sha256=hashlib.sha256(
                    self.fixture.receipt.read_bytes()
                ).hexdigest(),
                archive_tool_authority_path=authority_path,
                expected_archive_tool_authority_sha256=_ARCHIVE_AUTHORITY_SHA,
                remote_staging_directory=Path("/remote/staging"),
                drive_receipt_path=drive_receipt,
            )
        self.assertEqual(result, {"uploaded": True})
        options = execute.call_args.args[0]
        self.assertEqual(options.remote_active_campaign_root, str(self.fixture.active))
        self.assertEqual(options.remote_closed_bundle_dir, str(self.fixture.bundle))
        self.assertEqual(options.campaign_id, closure["campaign_id"])
        self.assertEqual(options.bundle_id, self.fixture.bundle.name)
        self.assertEqual(options.source_git_sha, _GIT_SHA)
        self.assertEqual(
            options.expected_marker_sha256,
            closure["closed_bundle"]["marker_sha256"],
        )
        self.assertEqual(
            options.expected_entry_count,
            closure["closed_bundle"]["entry_count"],
        )
        self.assertEqual(
            options.expected_total_file_bytes,
            closure["closed_bundle"]["total_file_bytes"],
        )
        self.assertEqual(
            options.tool_authority_sha256,
            _ARCHIVE_AUTHORITY_SHA,
        )

    def test_archive_requires_external_closure_receipt_transfer_digest(self) -> None:
        with self.fixture.validation_patches():
            archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        authority_path = self.fixture.receipts / "archive-authority.json"
        authority_path.write_bytes(b"sealed")
        with mock.patch.object(archive, "load_archive_tool_authority") as loader:
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "external transfer digest",
            ):
                archive.archive_phase_a_terminal_matrix(
                    closure_receipt_path=self.fixture.receipt,
                    expected_closure_receipt_sha256="0" * 64,
                    archive_tool_authority_path=authority_path,
                    expected_archive_tool_authority_sha256=(_ARCHIVE_AUTHORITY_SHA),
                    remote_staging_directory=Path("/remote/staging"),
                    drive_receipt_path=(
                        self.fixture.receipts / "wrong-digest-drive.json"
                    ),
                )
        loader.assert_not_called()

    def test_archive_authority_replacement_cannot_retarget_upload(self) -> None:
        with self.fixture.validation_patches():
            archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        authority_path = self.fixture.receipts / "replacement-authority.json"
        authority_path.write_bytes(b"same-source-different-rclone-authority")
        with mock.patch.object(archive, "load_archive_tool_authority") as loader:
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "external archive-tool authority digest differ",
            ):
                archive.archive_phase_a_terminal_matrix(
                    closure_receipt_path=self.fixture.receipt,
                    expected_closure_receipt_sha256=hashlib.sha256(
                        self.fixture.receipt.read_bytes()
                    ).hexdigest(),
                    archive_tool_authority_path=authority_path,
                    expected_archive_tool_authority_sha256="e" * 64,
                    remote_staging_directory=Path("/remote/staging"),
                    drive_receipt_path=(
                        self.fixture.receipts / "retargeted-drive.json"
                    ),
                )
        loader.assert_not_called()

    def test_archive_rejects_authority_source_or_endpoint_drift(self) -> None:
        with self.fixture.validation_patches():
            archive.close_phase_a_terminal_matrix(**self.fixture.close_arguments())
        authority_path = self.fixture.receipts / "archive-authority.json"
        authority_path.write_bytes(b"sealed")
        for changes in (
            {"source_git_sha": "f" * 40},
            {"ssh_target": "another-host"},
        ):
            with self.subTest(changes=changes):
                values = {
                    "authority_sha256": _ARCHIVE_AUTHORITY_SHA,
                    "ssh_target": "trainer-node",
                    "source_git_sha": _GIT_SHA,
                    "source_manifest_sha256": _MANIFEST_SHA,
                    "remote_repository_root": f"/checkouts/{_GIT_SHA}",
                }
                values.update(changes)
                with mock.patch.object(
                    archive,
                    "load_archive_tool_authority",
                    return_value=SimpleNamespace(**values),
                ):
                    with self.assertRaisesRegex(
                        archive.OpenEcologyPhaseAArchiveError,
                        "source/endpoint differ",
                    ):
                        archive.archive_phase_a_terminal_matrix(
                            closure_receipt_path=self.fixture.receipt,
                            expected_closure_receipt_sha256=hashlib.sha256(
                                self.fixture.receipt.read_bytes()
                            ).hexdigest(),
                            archive_tool_authority_path=authority_path,
                            expected_archive_tool_authority_sha256=(
                                _ARCHIVE_AUTHORITY_SHA
                            ),
                            remote_staging_directory=Path("/remote/staging"),
                            drive_receipt_path=(
                                self.fixture.receipts / f"drive-{len(changes)}.json"
                            ),
                        )

        with mock.patch.object(
            archive,
            "load_archive_tool_authority",
            return_value=SimpleNamespace(
                authority_sha256="e" * 64,
                ssh_target="trainer-node",
                source_git_sha=_GIT_SHA,
                source_manifest_sha256=_MANIFEST_SHA,
                remote_repository_root=f"/checkouts/{_GIT_SHA}",
            ),
        ):
            with self.assertRaisesRegex(
                archive.OpenEcologyPhaseAArchiveError,
                "source/endpoint differ",
            ):
                archive.archive_phase_a_terminal_matrix(
                    closure_receipt_path=self.fixture.receipt,
                    expected_closure_receipt_sha256=hashlib.sha256(
                        self.fixture.receipt.read_bytes()
                    ).hexdigest(),
                    archive_tool_authority_path=authority_path,
                    expected_archive_tool_authority_sha256=(_ARCHIVE_AUTHORITY_SHA),
                    remote_staging_directory=Path("/remote/staging"),
                    drive_receipt_path=(
                        self.fixture.receipts / "drive-authority-replaced.json"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
