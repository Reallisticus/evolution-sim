from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evolution_sim.mind.open_ecology_phase_a as phase_a
import evolution_sim.mind.open_ecology_phase_a_readiness as readiness
from evolution_sim.mind import open_ecology_selection
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_open_ecology_phase_a import _campaign


REQUIRES_MIND_ML = True

_PRODUCED_AT = "2026-07-27T10:00:00Z"
_AUTHORIZED_AT = "2026-07-27T11:00:00Z"
_SHA = "c" * 64


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
            False,
        )
        self.assertIs(
            report["proof_producer_availability"][
                "capture_noninterference_reexecution"
            ],
            True,
        )
        self.assertIs(
            report["dependency_specific_authority_verifiers_available"],
            False,
        )
        self.assertIs(report["phase_a_training_authorized"], False)
        self.assertEqual(
            report["blockers"],
            [
                "launch_evidence_index_required",
                "independent_report_authority_verifiers_required",
            ],
        )
        self.assertEqual(
            set(report["dependencies"]),
            set(phase_a.OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES),
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
                "fields differ",
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
        self.assertEqual(len(report["blockers"]), 15)

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
                "no independent raw-evidence authority verifier",
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
            dependencies, gates = _report_paths(root, self.campaign)
            path = dependencies["readiness_dependency_05"]["persistent_island_50000"]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["facts"]["terminal_tick"] = 10_000
            _resign(report)
            _write_json(path, report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "50,000-tick",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
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
            dependencies, gates = _report_paths(root, self.campaign)
            path = dependencies["readiness_dependency_05"]["persistent_island_50000"]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["facts"]["phase_a_artifact_access_count"] = 1
            _resign(report)
            _write_json(path, report)
            with self.assertRaisesRegex(
                phase_a.OpenEcologyPhaseAError,
                "50,000-tick",
            ):
                phase_a.build_open_ecology_phase_a_evidence_index(
                    self.campaign,
                    evidence_root=root,
                    dependency_reports=dependencies,
                    operational_reports=gates,
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
            target = dependencies["readiness_dependency_07"][
                "bounded_writer_event_coverage"
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


def _report_paths(
    root: Path,
    campaign: dict[str, object],
) -> tuple[dict[str, dict[str, Path]], dict[str, Path]]:
    dependency_reports: dict[str, dict[str, Path]] = {}
    for dependency_id, kinds in readiness.DEPENDENCY_EVIDENCE_KINDS.items():
        dependency_reports[dependency_id] = {}
        for kind in kinds:
            path = root / f"{kind}.json"
            _write_report(path, campaign=campaign, kind=kind)
            dependency_reports[dependency_id][kind] = path
    gate_reports: dict[str, Path] = {}
    for gate_name, kind in readiness.OPERATIONAL_EVIDENCE_KINDS.items():
        path = root / f"{kind}.json"
        _write_report(path, campaign=campaign, kind=kind)
        gate_reports[gate_name] = path
    return dependency_reports, gate_reports


def _write_report(
    path: Path,
    *,
    campaign: dict[str, object],
    kind: str,
) -> None:
    report: dict[str, object] = {
        "schema_version": (
            readiness.OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION
        ),
        "evidence_kind": kind,
        "campaign_digest": campaign["exact_digest"],
        "configuration_sha256": campaign["configuration_sha256"],
        "source": campaign["source"],
        "produced_at_utc": _PRODUCED_AT,
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
            "action_count_by_surface": {surface: 20 for surface in surfaces},
            "self_echo_case_count": 4,
            "receiver_own_emission_projection_nonzero_count": 0,
            "global_own_emission_report_count": 4,
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
        }
    if kind == "critic_gradient_and_density_schedule":
        return {
            "cell_contracts": campaign["architecture"]["critic_gradient_cells"],
            "phase_a_density_schedule": [64],
            "phase_b_density_cycle": [32, 64, 128, 64],
            "shared_value_trunk_gradient_l2": 1.0,
            "stop_gradient_value_trunk_gradient_l2": 0.0,
            "stop_gradient_value_head_gradient_l2": 1.0,
            "actor_loss_shared_trunk_gradient_l2": 1.0,
        }
    if kind == "fixed_batch_equivalence_and_speed":
        return {
            "declared_shape": {
                "worlds": 16,
                "rollout_ticks": 128,
                "initial_agents": 64,
            },
            "repeat_count": 2,
            "scalar_semantic_sha256": _SHA,
            "batched_semantic_sha256": _SHA,
            "batched_repeat_semantic_sha256": [_SHA, _SHA],
            "ordered_merge_semantic_sha256": _SHA,
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
    if kind == "exact_sha_operational_path":
        return {
            "fresh_checkout_count": 1,
            "dirty_checkout_count": 0,
            "source_manifest_match_count": 1,
            "import_root_match_count": 1,
            "end_to_end_training_benchmark_count": 1,
            "selection_512_benchmark_count": 1,
            "selection_2000_benchmark_count": 1,
            "output_lock_proof_count": 1,
            "drive_quota_proof_count": 1,
            "uploader_proof_count": 1,
            "verification_before_prune_proof_count": 1,
            "terminal_aggregate_proof_count": 1,
        }
    if kind == "multi_host_equivalence_and_torch_ci":
        return {
            "intended_host_class_count": 2,
            "tested_host_class_count": 2,
            "runtime_contract_sha256_by_host_class": {
                "host-a": "d" * 64,
                "host-b": "d" * 64,
            },
            "semantic_evidence_sha256_by_host_class": {
                "host-a": _SHA,
                "host-b": _SHA,
            },
            "cross_host_semantic_mismatch_count": 0,
            "torch_ci_lane_required": True,
            "torch_ci_test_count": 10,
            "torch_ci_failure_count": 0,
            "torch_ci_skip_count": 0,
        }
    if kind == "phase_a_full_throughput":
        return {
            "preliminary_throughput_gate_digest": campaign["throughput_gate"][
                "exact_digest"
            ],
            "phase_a_training_projection_authoritative": True,
            "selection_horizons": [512, 2000],
            "selection_primary_and_replay_measured": True,
            "semantic_mismatch_count": 0,
            "projected_full_matrix_median_seconds": 60 * 60,
            "projected_full_matrix_slow_plus_margin_seconds": 2 * 60 * 60,
            "maximum_gpu_memory_share": 0.5,
            "maximum_host_ram_share": 0.5,
            "swap_delta_bytes": 0,
            "temperature_below_throttle": True,
            "xid_count": 0,
            "oom_count": 0,
            "nonfinite_count": 0,
            "evidence_write_error_count": 0,
        }
    if kind == "campaign_storage_capacity":
        capacity = 1_000 * 1024**3
        return {
            "checked_at_utc": _PRODUCED_AT,
            "google_drive_free_bytes": 400 * 1024**3,
            "projected_active_storage_bytes": 100 * 1024**3,
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
        }
    if kind == "verification_before_prune":
        return {
            "preverification_prune_attempt_count": 1,
            "preverification_prune_rejection_count": 1,
            "postverification_prune_count": 1,
            "failed_verification_source_retained_count": 1,
            "remote_inventory_exact_count": 3,
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
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
