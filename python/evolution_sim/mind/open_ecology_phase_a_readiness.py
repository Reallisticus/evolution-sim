"""Fail-closed evidence authority for the open-ecology Phase-A launch.

This module does not manufacture behavioral evidence. It defines the exact
machine-readable reports that independent producers must emit, validates their
substantive observations, and assembles a launch authorization only when every
dependency and operational gate is present and source-bound.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from evolution_sim.mind.provenance import stable_payload_digest

if TYPE_CHECKING:
    from evolution_sim.mind import open_ecology_phase_a as phase_a_contract


OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_readiness_report_v1"
)
OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_evidence_index_v1"
)
OPEN_ECOLOGY_PHASE_A_LAUNCH_READINESS_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_launch_readiness_v2"
)

_SURFACES = (
    "observation",
    "action",
    "model_artifact",
    "replay",
    "checkpoint",
    "viewer",
)
_EVENT_KINDS = (
    "birth",
    "death",
    "lineage",
    "dyadic_interaction",
    "signal_contributor",
    "congestion",
    "intervention",
)
_CELLS = ("A0", "A1", "A2", "A3")
_ARMS = ("H", "Z", "R")
_GATES = (
    "throughput",
    "storage",
    "output_lock",
    "immutable_uploader",
    "verification_before_prune",
    "terminal_aggregate_validator",
)

DEPENDENCY_EVIDENCE_KINDS: Mapping[str, tuple[str, ...]] = {
    "readiness_dependency_01": ("cross_surface_and_self_echo",),
    "readiness_dependency_02": ("runtime_genome_and_action_source",),
    "readiness_dependency_03": ("critic_gradient_and_density_schedule",),
    "readiness_dependency_04": ("fixed_batch_equivalence_and_speed",),
    "readiness_dependency_05": ("persistent_island_50000",),
    "readiness_dependency_06": ("checkpoint_continuation_equivalence",),
    "readiness_dependency_07": ("bounded_writer_event_coverage",),
    "readiness_dependency_08": (
        "causal_evaluator_rejection_battery",
        "capture_noninterference_reexecution",
    ),
    "readiness_dependency_09": ("preregistration_roundtrip_fail_closed",),
    "readiness_dependency_10": (
        "exact_sha_operational_path",
        "multi_host_equivalence_and_torch_ci",
    ),
}

OPERATIONAL_EVIDENCE_KINDS: Mapping[str, str] = {
    "throughput": "phase_a_full_throughput",
    "storage": "campaign_storage_capacity",
    "output_lock": "output_lock_contention",
    "immutable_uploader": "immutable_drive_uploader",
    "verification_before_prune": "verification_before_prune",
    "terminal_aggregate_validator": "terminal_aggregate_validator",
}
PROOF_PRODUCER_AVAILABILITY: Mapping[str, bool] = {
    kind: kind == "capture_noninterference_reexecution"
    for kind in (
        *(
            evidence_kind
            for evidence_kinds in DEPENDENCY_EVIDENCE_KINDS.values()
            for evidence_kind in evidence_kinds
        ),
        *OPERATIONAL_EVIDENCE_KINDS.values(),
    )
}
# A producer can write a report without making that report authoritative.  An
# authority verifier must independently reconstruct the report's substantive
# facts from exact, referenced raw evidence.  A canonical digest is only an
# integrity checksum and must never be treated as a signature.
ReportAuthorityVerifier = Callable[
    [Path, Mapping[str, object], datetime | None],
    Mapping[str, object],
]
REPORT_AUTHORITY_VERIFIERS: Mapping[str, ReportAuthorityVerifier] = {}


def produce_capture_noninterference_reexecution_report(
    preregistration: Mapping[str, object],
    *,
    primary_proof_path: str | Path,
) -> dict[str, object]:
    """Reexecute the real capture proof and emit dependency-08 evidence."""

    from evolution_sim.mind import open_ecology_selection

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    proof_path = Path(primary_proof_path)
    if (
        not proof_path.is_file()
        or proof_path.is_symlink()
        or proof_path.stat().st_nlink != 1
    ):
        raise contract.OpenEcologyPhaseAError(
            "capture proof must be one regular non-hardlinked file"
        )
    _require_live_source_twice(preregistration, before=True)
    primary = contract._load_strict_json(proof_path)
    try:
        reproduced = open_ecology_selection.verify_open_ecology_capture_noninterference_proof_by_reexecution(
            primary,
            preregistration=preregistration,
        )
    except (ValueError, RuntimeError) as error:
        raise contract.OpenEcologyPhaseAError(
            f"capture proof reexecution failed: {error}"
        ) from error
    primary_bytes = _canonical_json_bytes(primary)
    reproduced_bytes = _canonical_json_bytes(reproduced)
    if primary_bytes != reproduced_bytes:
        raise contract.OpenEcologyPhaseAError(
            "capture proof canonical bytes changed under reexecution"
        )
    proof_contract = _mapping(primary["contract"], field="capture proof contract")
    dynamics = _mapping(primary["dynamics"], field="capture proof dynamics")
    report_sha256 = hashlib.sha256(primary_bytes).hexdigest()
    report: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION,
        "evidence_kind": "capture_noninterference_reexecution",
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _source_binding(preregistration),
        "produced_at_utc": _format_utc(_utc_now()),
        "facts": {
            "cell_order": proof_contract["cell_order"],
            "density_levels": proof_contract["density_levels"],
            "case_count": proof_contract["case_count"],
            "ticks_per_case": proof_contract["ticks_per_case"],
            "birth_count": dynamics["total_births"],
            "death_count": dynamics["total_deaths"],
            "primary_report_sha256": report_sha256,
            "reexecution_report_sha256": report_sha256,
            "report_bytes_match": True,
            "scientific_selection_seed_access_count": 0,
            "model_state_mismatch_count": 0,
            "trajectory_mismatch_count": 0,
            "action_rng_mismatch_count": 0,
            "genome_provenance_mismatch_count": 0,
        },
    }
    report["exact_digest"] = stable_payload_digest(report)
    _validate_report(
        report,
        expected_kind="capture_noninterference_reexecution",
        preregistration=preregistration,
        authorization_time=None,
        require_authority_verifier=False,
    )
    _require_live_source_twice(preregistration, before=False)
    return report


def launch_readiness(
    *,
    preregistration: Mapping[str, object] | None = None,
    evidence_index: Mapping[str, object] | None = None,
    evidence_index_path: str | Path | None = None,
) -> dict[str, object]:
    """Describe executable authority separately from available evidence."""

    contract = _contract()
    verifier_availability = _authority_verifier_availability()
    if (
        preregistration is None
        and evidence_index is None
        and evidence_index_path is None
    ):
        blockers = [
            "launch_evidence_index_required",
            "independent_report_authority_verifiers_required",
        ]
        dependencies = {
            dependency_id: {
                "status": "evidence_not_supplied",
                "required_evidence_kinds": list(kinds),
            }
            for dependency_id, kinds in DEPENDENCY_EVIDENCE_KINDS.items()
        }
        gates = {
            gate: {
                "status": "evidence_not_supplied",
                "required_evidence_kind": kind,
            }
            for gate, kind in OPERATIONAL_EVIDENCE_KINDS.items()
        }
    elif (
        preregistration is None or evidence_index is None or evidence_index_path is None
    ):
        raise contract.OpenEcologyPhaseAError(
            "readiness inspection requires preregistration, evidence index, "
            "and evidence-index path together"
        )
    else:
        contract.validate_open_ecology_phase_a_preregistration(preregistration)
        dependencies, gates, blockers = _inspect_evidence_index(
            preregistration,
            evidence_index=evidence_index,
            evidence_index_path=Path(evidence_index_path),
            authorization_time=_utc_now(),
            allow_missing=True,
        )
        try:
            contract._require_live_source(preregistration)
        except contract.OpenEcologyPhaseAError as error:
            blockers.append(f"live_source:{error}")
    authorized = not blockers
    readiness: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_LAUNCH_READINESS_SCHEMA_VERSION,
        "phase_a_training_authorized": authorized,
        "authorization_assembler_available": True,
        "dependency_specific_semantic_validators_available": True,
        "dependency_specific_proof_producers_available": all(
            PROOF_PRODUCER_AVAILABILITY.values()
        ),
        "dependency_specific_authority_verifiers_available": all(
            verifier_availability.values()
        ),
        "proof_producer_availability": dict(PROOF_PRODUCER_AVAILABILITY),
        "authority_verifier_availability": verifier_availability,
        "blockers": blockers,
        "dependencies": dependencies,
        "operational_gates": gates,
        "reason": (
            "all_required_evidence_semantically_valid"
            if authorized
            else "required_behavioral_or_operational_evidence_unavailable"
        ),
        "claim_boundary": {
            "training_launch": authorized,
            "phase_b": False,
            "runtime_integration": False,
            "promotion": False,
        },
    }
    readiness["exact_digest"] = stable_payload_digest(readiness)
    return readiness


def build_evidence_index(
    preregistration: Mapping[str, object],
    *,
    evidence_root: str | Path,
    dependency_reports: Mapping[str, Mapping[str, str | Path]],
    operational_reports: Mapping[str, str | Path],
) -> dict[str, object]:
    """Index already-produced reports; never infer or manufacture their facts."""

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    raw_root = Path(evidence_root)
    if raw_root.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            "Phase A evidence root must be one real directory"
        )
    root = raw_root.resolve()
    if not root.is_dir():
        raise contract.OpenEcologyPhaseAError(
            "Phase A evidence root must be one real directory"
        )
    _require_live_source_twice(preregistration, before=True)
    source = _source_binding(preregistration)
    unknown_dependencies = set(dependency_reports) - set(DEPENDENCY_EVIDENCE_KINDS)
    if unknown_dependencies:
        raise contract.OpenEcologyPhaseAError(
            "dependency_reports contains unknown dependencies "
            f"{sorted(unknown_dependencies)}"
        )
    indexed_dependencies: list[dict[str, object]] = []
    for dependency_id in contract.OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES:
        supplied = dependency_reports.get(dependency_id)
        if supplied is None:
            continue
        if not isinstance(supplied, Mapping):
            raise contract.OpenEcologyPhaseAError(
                f"report mapping for {dependency_id} is malformed"
            )
        expected_kinds = DEPENDENCY_EVIDENCE_KINDS[dependency_id]
        _exact_key_set(
            supplied,
            set(expected_kinds),
            field=f"dependency_reports.{dependency_id}",
        )
        reports = [
            _index_report(
                root=root,
                path=Path(supplied[kind]),
                expected_kind=kind,
                preregistration=preregistration,
                authorization_time=None,
            )
            for kind in expected_kinds
        ]
        indexed_dependencies.append(
            {
                "dependency_id": dependency_id,
                "reports": reports,
            }
        )
    unknown_gates = set(operational_reports) - set(_GATES)
    if unknown_gates:
        raise contract.OpenEcologyPhaseAError(
            f"operational_reports contains unknown gates {sorted(unknown_gates)}"
        )
    indexed_gates = [
        {
            "gate_name": gate,
            "report": _index_report(
                root=root,
                path=Path(operational_reports[gate]),
                expected_kind=OPERATIONAL_EVIDENCE_KINDS[gate],
                preregistration=preregistration,
                authorization_time=None,
            ),
        }
        for gate in _GATES
        if gate in operational_reports
    ]
    index: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": source,
        "dependency_reports": indexed_dependencies,
        "operational_reports": indexed_gates,
    }
    index["exact_digest"] = stable_payload_digest(index)
    _require_live_source_twice(preregistration, before=False)
    return index


def build_launch_authorization(
    preregistration: Mapping[str, object],
    *,
    evidence_index: Mapping[str, object],
    evidence_index_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, object]:
    """Assemble authority only from semantically validated exact-source reports."""

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    raw_index_path = Path(evidence_index_path)
    raw_destination = Path(authorization_path)
    if raw_index_path.is_symlink() or raw_destination.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            "evidence index and launch authorization paths cannot be symbolic links"
        )
    index_path = raw_index_path.resolve()
    destination = raw_destination.resolve()
    if index_path.parent != destination.parent:
        raise contract.OpenEcologyPhaseAError(
            "evidence index and launch authorization must share one sealed root"
        )
    _require_live_source_twice(preregistration, before=True)
    authorized_at = _utc_now()
    dependencies, gates, blockers = _inspect_evidence_index(
        preregistration,
        evidence_index=evidence_index,
        evidence_index_path=index_path,
        authorization_time=authorized_at,
        allow_missing=False,
    )
    if blockers:
        raise contract.OpenEcologyPhaseAError(
            "Phase A launch evidence remains incomplete: " + ", ".join(blockers)
        )
    source = _source_binding(preregistration)
    indexed_dependencies = _dependency_entries(evidence_index)
    dependency_proofs: list[dict[str, object]] = []
    for dependency_id, status in dependencies.items():
        entry = indexed_dependencies[dependency_id]
        dependency_proofs.append(
            {
                "dependency_id": dependency_id,
                "status": "behaviorally_proved",
                "source_commit": source["commit"],
                "source_manifest_sha256": source["manifest_sha256"],
                "assertions": contract._phase_a_readiness_assertions(dependency_id),
                "semantic_report_digests": status["semantic_report_digests"],
                "evidence": [report["file"] for report in _reports_from_entry(entry)],
            }
        )
    indexed_gates = _gate_entries(evidence_index)
    operational_gates: dict[str, object] = {}
    for gate_name in _GATES:
        entry = indexed_gates[gate_name]
        report_entry = _mapping(entry["report"], field=f"{gate_name}.report")
        report = _load_indexed_report(
            index_path.parent,
            report_entry,
            expected_kind=OPERATIONAL_EVIDENCE_KINDS[gate_name],
            preregistration=preregistration,
            authorization_time=authorized_at,
        )
        gate: dict[str, object] = {
            "passed": True,
            "evidence_kind": OPERATIONAL_EVIDENCE_KINDS[gate_name],
            "semantic_report_digest": report["exact_digest"],
            "evidence": [report_entry["file"]],
        }
        if gate_name == "throughput":
            gate["preliminary_throughput_gate_digest"] = _mapping(
                preregistration["throughput_gate"],
                field="throughput_gate",
            )["exact_digest"]
        elif gate_name == "storage":
            facts = _mapping(report["facts"], field="storage.facts")
            for key in (
                "checked_at_utc",
                "google_drive_free_bytes",
                "projected_active_storage_bytes",
                "target_filesystem_capacity_bytes",
                "target_filesystem_free_bytes",
                "required_target_free_bytes",
            ):
                gate[key] = facts[key]
        operational_gates[gate_name] = gate
    authorization: dict[str, object] = {
        "schema_version": (
            contract.OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
        ),
        "authorized_at_utc": _format_utc(authorized_at),
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": source,
        "evidence_index": {
            "exact_digest": evidence_index["exact_digest"],
            "file": contract._file_reference(index_path, base=destination.parent),
        },
        "readiness_dependencies": dependency_proofs,
        "operational_gates": operational_gates,
        "authorization": {
            "phase_a_training_authorized": True,
            "authorization_basis": (
                "all_10_behavioral_dependencies_plus_operational_gates"
            ),
            "authorization_scope": "phase_a_training_only",
            "phase_b_authorized": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
        },
    }
    authorization["exact_digest"] = stable_payload_digest(authorization)
    _require_bound_payload_file(
        raw_index_path,
        evidence_index,
        field="Phase A evidence index",
    )
    _require_live_source_twice(preregistration, before=False)
    return authorization


def validate_launch_authorization(
    authorization: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    authorization_path: str | Path,
) -> None:
    """Reopen and semantically validate every report referenced by authority."""

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    raw_path = Path(authorization_path)
    _require_bound_payload_file(
        raw_path,
        authorization,
        field="Phase A launch authorization",
    )
    _exact_key_set(
        authorization,
        {
            "schema_version",
            "authorized_at_utc",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "evidence_index",
            "readiness_dependencies",
            "operational_gates",
            "authorization",
            "exact_digest",
        },
        field="Phase A launch authorization",
    )
    contract._validate_signed_payload(
        authorization,
        field="Phase A launch authorization",
    )
    if (
        authorization.get("schema_version")
        != contract.OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("campaign_digest") != preregistration.get("exact_digest")
        or authorization.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or authorization.get("source") != preregistration.get("source")
    ):
        raise contract.OpenEcologyPhaseAError(
            "Phase A launch authorization is detached from the campaign"
        )
    _require_live_source_twice(preregistration, before=True)
    authorized_at = _parse_utc(
        authorization.get("authorized_at_utc"),
        field="authorization.authorized_at_utc",
    )
    validation_time = _utc_now()
    if authorized_at > validation_time:
        raise contract.OpenEcologyPhaseAError(
            "Phase A launch authorization postdates current validation time"
        )
    path = raw_path.resolve()
    base = path.parent
    indexed = _mapping(
        authorization.get("evidence_index"),
        field="authorization.evidence_index",
    )
    _exact_key_set(
        indexed,
        {"exact_digest", "file"},
        field="authorization.evidence_index",
    )
    _sha256(indexed.get("exact_digest"), field="evidence_index.exact_digest")
    contract._verify_evidence_references(
        [indexed.get("file")],
        base=base,
        field="authorization.evidence_index.file",
    )
    index_reference = _mapping(
        indexed.get("file"),
        field="authorization.evidence_index.file",
    )
    index_payload = contract._load_strict_json(
        _regular_bundle_file(
            base,
            str(index_reference["relative_path"]),
            field="authorization.evidence_index.file",
        )
    )
    _validate_index_header(index_payload, preregistration=preregistration)
    if indexed.get("exact_digest") != index_payload.get("exact_digest"):
        raise contract.OpenEcologyPhaseAError(
            "Phase A evidence-index digest binding drifted"
        )
    indexed_dependencies = _dependency_entries(index_payload)
    indexed_gates = _gate_entries(index_payload)
    source = _source_binding(preregistration)
    raw_proofs = _sequence(
        authorization.get("readiness_dependencies"),
        field="authorization.readiness_dependencies",
    )
    if len(raw_proofs) != len(DEPENDENCY_EVIDENCE_KINDS):
        raise contract.OpenEcologyPhaseAError(
            "Phase A launch authorization requires all 10 readiness proofs"
        )
    for dependency_id, raw_proof in zip(
        contract.OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES,
        raw_proofs,
        strict=True,
    ):
        proof = _mapping(raw_proof, field=f"proof.{dependency_id}")
        _exact_key_set(
            proof,
            {
                "dependency_id",
                "status",
                "source_commit",
                "source_manifest_sha256",
                "assertions",
                "semantic_report_digests",
                "evidence",
            },
            field=f"proof.{dependency_id}",
        )
        if (
            proof.get("dependency_id") != dependency_id
            or proof.get("status") != "behaviorally_proved"
            or proof.get("source_commit") != source["commit"]
            or proof.get("source_manifest_sha256") != source["manifest_sha256"]
            or proof.get("assertions")
            != contract._phase_a_readiness_assertions(dependency_id)
        ):
            raise contract.OpenEcologyPhaseAError(
                f"Phase A readiness proof {dependency_id} drifted"
            )
        references = _sequence(
            proof.get("evidence"),
            field=f"proof.{dependency_id}.evidence",
        )
        kinds = DEPENDENCY_EVIDENCE_KINDS[dependency_id]
        expected_index_reports = _reports_from_entry(
            indexed_dependencies[dependency_id]
        )
        if len(references) != len(kinds):
            raise contract.OpenEcologyPhaseAError(
                f"Phase A readiness proof {dependency_id} evidence count drifted"
            )
        if list(references) != [report["file"] for report in expected_index_reports]:
            raise contract.OpenEcologyPhaseAError(
                f"Phase A readiness proof {dependency_id} differs from its index"
            )
        digests: dict[str, str] = {}
        for report_index, (kind, reference) in enumerate(
            zip(kinds, references, strict=True)
        ):
            report = _load_report_reference(
                base,
                reference,
                expected_kind=kind,
                preregistration=preregistration,
                authorization_time=authorized_at,
            )
            indexed_report = expected_index_reports[report_index]
            if indexed_report.get("evidence_kind") != kind or indexed_report.get(
                "semantic_report_digest"
            ) != report.get("exact_digest"):
                raise contract.OpenEcologyPhaseAError(
                    f"Phase A readiness proof {dependency_id} index digest drifted"
                )
            digests[kind] = str(report["exact_digest"])
        if proof.get("semantic_report_digests") != digests:
            raise contract.OpenEcologyPhaseAError(
                f"Phase A readiness proof {dependency_id} report digests drifted"
            )
    gates = _mapping(
        authorization.get("operational_gates"),
        field="authorization.operational_gates",
    )
    _exact_key_set(gates, set(_GATES), field="authorization.operational_gates")
    for gate_name in _GATES:
        gate = _mapping(gates[gate_name], field=f"gate.{gate_name}")
        expected_fields = {
            "passed",
            "evidence_kind",
            "semantic_report_digest",
            "evidence",
        }
        if gate_name == "throughput":
            expected_fields.add("preliminary_throughput_gate_digest")
        elif gate_name == "storage":
            expected_fields.update(
                {
                    "checked_at_utc",
                    "google_drive_free_bytes",
                    "projected_active_storage_bytes",
                    "target_filesystem_capacity_bytes",
                    "target_filesystem_free_bytes",
                    "required_target_free_bytes",
                }
            )
        _exact_key_set(gate, expected_fields, field=f"gate.{gate_name}")
        kind = OPERATIONAL_EVIDENCE_KINDS[gate_name]
        references = _sequence(gate["evidence"], field=f"gate.{gate_name}.evidence")
        indexed_gate_report = _mapping(
            indexed_gates[gate_name]["report"],
            field=f"indexed gate {gate_name}",
        )
        if (
            gate.get("passed") is not True
            or gate.get("evidence_kind") != kind
            or len(references) != 1
            or references[0] != indexed_gate_report.get("file")
        ):
            raise contract.OpenEcologyPhaseAError(
                f"Phase A operational gate {gate_name} failed"
            )
        report = _load_report_reference(
            base,
            references[0],
            expected_kind=kind,
            preregistration=preregistration,
            authorization_time=authorized_at,
        )
        if indexed_gate_report.get("evidence_kind") != kind or indexed_gate_report.get(
            "semantic_report_digest"
        ) != report.get("exact_digest"):
            raise contract.OpenEcologyPhaseAError(
                f"Phase A operational gate {gate_name} index digest drifted"
            )
        if gate.get("semantic_report_digest") != report.get("exact_digest"):
            raise contract.OpenEcologyPhaseAError(
                f"Phase A operational gate {gate_name} report digest drifted"
            )
        if gate_name == "throughput":
            preliminary = _mapping(
                preregistration["throughput_gate"],
                field="throughput_gate",
            )
            if gate.get("preliminary_throughput_gate_digest") != preliminary.get(
                "exact_digest"
            ):
                raise contract.OpenEcologyPhaseAError(
                    "Phase A throughput preliminary-gate binding drifted"
                )
        elif gate_name == "storage":
            facts = _mapping(report["facts"], field="storage.facts")
            # The authorization can be replayed later, but Drive capacity
            # cannot. Require a capacity attestation fresh at this validation,
            # forcing a new report and authorization when it expires.
            _validate_storage(
                facts,
                preregistration,
                validation_time,
            )
            for key in (
                "checked_at_utc",
                "google_drive_free_bytes",
                "projected_active_storage_bytes",
                "target_filesystem_capacity_bytes",
                "target_filesystem_free_bytes",
                "required_target_free_bytes",
            ):
                if gate.get(key) != facts.get(key):
                    raise contract.OpenEcologyPhaseAError(
                        "Phase A storage gate fact projection drifted"
                    )
    if authorization.get("authorization") != {
        "phase_a_training_authorized": True,
        "authorization_basis": (
            "all_10_behavioral_dependencies_plus_operational_gates"
        ),
        "authorization_scope": "phase_a_training_only",
        "phase_b_authorized": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
    }:
        raise contract.OpenEcologyPhaseAError(
            "Phase A launch authorization claim drifted"
        )
    _require_bound_payload_file(
        raw_path,
        authorization,
        field="Phase A launch authorization",
    )
    _require_live_source_twice(preregistration, before=False)


def _inspect_evidence_index(
    preregistration: Mapping[str, object],
    *,
    evidence_index: Mapping[str, object],
    evidence_index_path: Path,
    authorization_time: datetime | None,
    allow_missing: bool,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    contract = _contract()
    _require_bound_payload_file(
        evidence_index_path,
        evidence_index,
        field="Phase A evidence index",
    )
    index_path = evidence_index_path.resolve()
    _validate_index_header(evidence_index, preregistration=preregistration)
    dependencies: dict[str, object] = {}
    gates: dict[str, object] = {}
    blockers: list[str] = []
    try:
        indexed_dependencies = _dependency_entries(
            evidence_index,
            require_complete=not allow_missing,
        )
    except contract.OpenEcologyPhaseAError:
        if not allow_missing:
            raise
        indexed_dependencies = {}
    for dependency_id, kinds in DEPENDENCY_EVIDENCE_KINDS.items():
        entry = indexed_dependencies.get(dependency_id)
        if entry is None:
            blockers.append(f"{dependency_id}:missing_dependency_entry")
            dependencies[dependency_id] = {
                "status": "missing",
                "required_evidence_kinds": list(kinds),
            }
            continue
        try:
            reports = _reports_from_entry(entry)
            if tuple(report.get("evidence_kind") for report in reports) != kinds:
                raise contract.OpenEcologyPhaseAError(
                    f"{dependency_id} evidence-kind ordering drifted"
                )
            digests: dict[str, str] = {}
            for kind, report_entry in zip(kinds, reports, strict=True):
                report = _load_indexed_report(
                    index_path.parent,
                    report_entry,
                    expected_kind=kind,
                    preregistration=preregistration,
                    authorization_time=authorization_time,
                )
                digests[kind] = str(report["exact_digest"])
            dependencies[dependency_id] = {
                "status": "semantically_valid",
                "required_evidence_kinds": list(kinds),
                "semantic_report_digests": digests,
            }
        except contract.OpenEcologyPhaseAError as error:
            if not allow_missing:
                raise
            blockers.append(f"{dependency_id}:{error}")
            dependencies[dependency_id] = {
                "status": "invalid",
                "required_evidence_kinds": list(kinds),
                "error": str(error),
            }
    try:
        indexed_gates = _gate_entries(
            evidence_index,
            require_complete=not allow_missing,
        )
    except contract.OpenEcologyPhaseAError:
        if not allow_missing:
            raise
        indexed_gates = {}
    for gate_name, kind in OPERATIONAL_EVIDENCE_KINDS.items():
        entry = indexed_gates.get(gate_name)
        if entry is None:
            blockers.append(f"operational_gate_{gate_name}:missing_gate_entry")
            gates[gate_name] = {
                "status": "missing",
                "required_evidence_kind": kind,
            }
            continue
        try:
            report_entry = _mapping(entry["report"], field=f"{gate_name}.report")
            report = _load_indexed_report(
                index_path.parent,
                report_entry,
                expected_kind=kind,
                preregistration=preregistration,
                authorization_time=authorization_time,
            )
            gates[gate_name] = {
                "status": "semantically_valid",
                "required_evidence_kind": kind,
                "semantic_report_digest": report["exact_digest"],
            }
        except contract.OpenEcologyPhaseAError as error:
            if not allow_missing:
                raise
            blockers.append(f"operational_gate_{gate_name}:{error}")
            gates[gate_name] = {
                "status": "invalid",
                "required_evidence_kind": kind,
                "error": str(error),
            }
    _require_bound_payload_file(
        evidence_index_path,
        evidence_index,
        field="Phase A evidence index",
    )
    return dependencies, gates, blockers


def _validate_index_header(
    evidence_index: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
) -> None:
    contract = _contract()
    _exact_key_set(
        evidence_index,
        {
            "schema_version",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "dependency_reports",
            "operational_reports",
            "exact_digest",
        },
        field="Phase A evidence index",
    )
    contract._validate_signed_payload(evidence_index, field="Phase A evidence index")
    if (
        evidence_index.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION
        or evidence_index.get("campaign_digest") != preregistration.get("exact_digest")
        or evidence_index.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or evidence_index.get("source") != preregistration.get("source")
    ):
        raise contract.OpenEcologyPhaseAError(
            "Phase A evidence index is detached from the campaign"
        )


def _index_report(
    *,
    root: Path,
    path: Path,
    expected_kind: str,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> dict[str, object]:
    contract = _contract()
    candidate = path if path.is_absolute() else root / path
    current = candidate
    while current.resolve() != root:
        if current.is_symlink():
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind}.file contains a symbolic link"
            )
        if current.parent == current:
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind}.file escaped its sealed root"
            )
        current = current.parent
    try:
        relative = candidate.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind}.file escaped its sealed root"
        ) from error
    resolved = _regular_bundle_file(
        root,
        relative,
        field=f"{expected_kind}.file",
    )
    report = contract._load_strict_json(resolved)
    _validate_report(
        report,
        expected_kind=expected_kind,
        preregistration=preregistration,
        authorization_time=authorization_time,
        report_path=resolved,
    )
    return {
        "evidence_kind": expected_kind,
        "semantic_report_digest": report["exact_digest"],
        "file": contract._file_reference(resolved, base=root),
    }


def _load_indexed_report(
    base: Path,
    entry: Mapping[str, object],
    *,
    expected_kind: str,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> dict[str, object]:
    _exact_key_set(
        entry,
        {"evidence_kind", "semantic_report_digest", "file"},
        field=f"indexed report {expected_kind}",
    )
    if entry.get("evidence_kind") != expected_kind:
        raise _error(f"indexed report expected {expected_kind}")
    report = _load_report_reference(
        base,
        entry["file"],
        expected_kind=expected_kind,
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    if entry.get("semantic_report_digest") != report.get("exact_digest"):
        raise _error(f"{expected_kind} indexed report digest drifted")
    return report


def _load_report_reference(
    base: Path,
    raw_reference: object,
    *,
    expected_kind: str,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> dict[str, object]:
    contract = _contract()
    reference = _mapping(raw_reference, field=f"{expected_kind}.file")
    contract._verify_evidence_references(
        [reference],
        base=base,
        field=f"{expected_kind}.evidence",
    )
    relative = str(reference["relative_path"])
    report_path = _regular_bundle_file(
        base,
        relative,
        field=f"{expected_kind}.evidence",
    )
    report = contract._load_strict_json(report_path)
    _validate_report(
        report,
        expected_kind=expected_kind,
        preregistration=preregistration,
        authorization_time=authorization_time,
        report_path=report_path,
    )
    return report


def _validate_report(
    report: Mapping[str, object],
    *,
    expected_kind: str,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
    report_path: Path | None = None,
    require_authority_verifier: bool = True,
) -> None:
    contract = _contract()
    _exact_key_set(
        report,
        {
            "schema_version",
            "evidence_kind",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "produced_at_utc",
            "facts",
            "exact_digest",
        },
        field=f"{expected_kind} report",
    )
    contract._validate_signed_payload(report, field=f"{expected_kind} report")
    if (
        report.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION
        or report.get("evidence_kind") != expected_kind
        or report.get("campaign_digest") != preregistration.get("exact_digest")
        or report.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or report.get("source") != preregistration.get("source")
    ):
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} report is not exact-campaign source-bound"
        )
    produced_at = _parse_utc(
        report.get("produced_at_utc"),
        field=f"{expected_kind}.produced_at_utc",
    )
    if authorization_time is not None and produced_at > authorization_time:
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} report postdates authorization"
        )
    facts = _mapping(report.get("facts"), field=f"{expected_kind}.facts")
    validator = _SEMANTIC_VALIDATORS.get(expected_kind)
    if validator is None:
        raise contract.OpenEcologyPhaseAError(
            f"no semantic validator for {expected_kind}"
        )
    validator(facts, preregistration, authorization_time)
    if not require_authority_verifier:
        return
    verifier = REPORT_AUTHORITY_VERIFIERS.get(expected_kind)
    if not callable(verifier):
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} has no independent raw-evidence authority verifier; "
            "a self-digested semantic report cannot authorize training"
        )
    if report_path is None:
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} authority verification requires its exact report path"
        )
    reconstructed = verifier(
        report_path,
        preregistration,
        authorization_time,
    )
    reconstructed_mapping = _mapping(
        reconstructed,
        field=f"{expected_kind} reconstructed authority evidence",
    )
    _exact_key_set(
        reconstructed_mapping,
        {
            "evidence_kind",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "facts",
        },
        field=f"{expected_kind} reconstructed authority evidence",
    )
    expected_reconstruction = {
        "evidence_kind": expected_kind,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _source_binding(preregistration),
        "facts": dict(facts),
    }
    if dict(reconstructed_mapping) != expected_reconstruction:
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} independently reconstructed facts or identity "
            "differ from the semantic report"
        )


def _validate_cross_surface(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "field_order_sha256_by_surface",
            "token_profile_count_by_surface",
            "action_count_by_surface",
            "self_echo_case_count",
            "receiver_own_emission_projection_nonzero_count",
            "global_own_emission_report_count",
        },
        field="cross_surface facts",
    )
    field_digests = _surface_mapping(
        facts["field_order_sha256_by_surface"],
        field="field_order_sha256_by_surface",
    )
    if (
        len(
            {
                _sha256(value, field="field_order_sha256")
                for value in field_digests.values()
            }
        )
        != 1
    ):
        raise _error("cross-surface field order differs")
    token_counts = _surface_mapping(
        facts["token_profile_count_by_surface"],
        field="token_profile_count_by_surface",
    )
    action_counts = _surface_mapping(
        facts["action_count_by_surface"],
        field="action_count_by_surface",
    )
    if any(
        _integer(value, field="token count", minimum=0) != 4
        for value in token_counts.values()
    ):
        raise _error("cross-surface token/profile count differs from four")
    if any(
        _integer(value, field="action count", minimum=0) != 20
        for value in action_counts.values()
    ):
        raise _error("cross-surface action count differs from twenty")
    if (
        _integer(facts["self_echo_case_count"], field="self_echo_case_count", minimum=1)
        < 1
        or _integer(
            facts["receiver_own_emission_projection_nonzero_count"],
            field="receiver self echo",
            minimum=0,
        )
        != 0
        or _integer(
            facts["global_own_emission_report_count"],
            field="global own emission reports",
            minimum=1,
        )
        < 1
    ):
        raise _error("communication self-echo semantics were not proved")


def _validate_genome_runtime(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    fields = {
        "heritable_founder_count",
        "heritable_child_count",
        "parent_lineage_match_count",
        "zero_all_founder_count",
        "zero_all_child_count",
        "zero_all_nonzero_genome_count",
        "birth_state_reset_count",
        "genome_digest_replay_mismatch_count",
        "missing_genome_rejection_count",
        "decision_count",
        "heuristic_action_source_count",
    }
    _exact_key_set(facts, fields, field="genome runtime facts")
    child_count = _integer(
        facts["heritable_child_count"],
        field="heritable_child_count",
        minimum=1,
    )
    if (
        _integer(
            facts["heritable_founder_count"], field="heritable founders", minimum=1
        )
        < 1
        or _integer(
            facts["parent_lineage_match_count"], field="lineage matches", minimum=0
        )
        != child_count
        or _integer(facts["zero_all_founder_count"], field="zero founders", minimum=1)
        < 1
        or _integer(facts["zero_all_child_count"], field="zero children", minimum=1) < 1
        or _integer(
            facts["zero_all_nonzero_genome_count"],
            field="zero genome errors",
            minimum=0,
        )
        != 0
        or _integer(facts["birth_state_reset_count"], field="birth resets", minimum=0)
        < child_count
        or _integer(
            facts["genome_digest_replay_mismatch_count"],
            field="replay mismatches",
            minimum=0,
        )
        != 0
        or _integer(
            facts["missing_genome_rejection_count"],
            field="missing genome rejections",
            minimum=1,
        )
        < 1
        or _integer(facts["decision_count"], field="decision_count", minimum=1) < 1
        or _integer(
            facts["heuristic_action_source_count"], field="heuristic actions", minimum=0
        )
        != 0
    ):
        raise _error("runtime genome/action-source behavior was not proved")


def _validate_critic_gradient(
    facts: Mapping[str, object],
    preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "cell_contracts",
            "phase_a_density_schedule",
            "phase_b_density_cycle",
            "shared_value_trunk_gradient_l2",
            "stop_gradient_value_trunk_gradient_l2",
            "stop_gradient_value_head_gradient_l2",
            "actor_loss_shared_trunk_gradient_l2",
        },
        field="critic gradient facts",
    )
    expected_cells = _mapping(preregistration["architecture"], field="architecture")[
        "critic_gradient_cells"
    ]
    if facts["cell_contracts"] != expected_cells:
        raise _error("four critic/gradient cell contracts drifted")
    if (
        facts["phase_a_density_schedule"] != [64]
        or facts["phase_b_density_cycle"] != [32, 64, 128, 64]
        or _positive_number(
            facts["shared_value_trunk_gradient_l2"],
            field="shared value gradient",
        )
        <= 0
        or _number(
            facts["stop_gradient_value_trunk_gradient_l2"],
            field="stopped value gradient",
        )
        != 0
        or _positive_number(
            facts["stop_gradient_value_head_gradient_l2"],
            field="value head gradient",
        )
        <= 0
        or _positive_number(
            facts["actor_loss_shared_trunk_gradient_l2"],
            field="actor gradient",
        )
        <= 0
    ):
        raise _error("critic gradient boundary or density schedule was not proved")


def _validate_fixed_batch(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "declared_shape",
            "repeat_count",
            "scalar_semantic_sha256",
            "batched_semantic_sha256",
            "batched_repeat_semantic_sha256",
            "ordered_merge_semantic_sha256",
            "scalar_median_elapsed_ns",
            "batched_median_elapsed_ns",
        },
        field="fixed batch facts",
    )
    if facts["declared_shape"] != {
        "worlds": 16,
        "rollout_ticks": 128,
        "initial_agents": 64,
    }:
        raise _error("fixed batch proof did not use the Phase-A shape")
    scalar = _sha256(facts["scalar_semantic_sha256"], field="scalar semantics")
    batched = _sha256(facts["batched_semantic_sha256"], field="batch semantics")
    repeats = _sequence(
        facts["batched_repeat_semantic_sha256"],
        field="batch repeat digests",
    )
    if (
        _integer(facts["repeat_count"], field="repeat_count", minimum=2) != len(repeats)
        or any(
            _sha256(value, field="batch repeat digest") != scalar for value in repeats
        )
        or batched != scalar
        or _sha256(facts["ordered_merge_semantic_sha256"], field="ordered merge")
        != scalar
        or _positive_integer(facts["batched_median_elapsed_ns"], field="batch elapsed")
        >= _positive_integer(facts["scalar_median_elapsed_ns"], field="scalar elapsed")
    ):
        raise _error("fixed batch equivalence or measured speed benefit failed")


def _validate_persistent_50000(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "scheduled_task_count",
            "completed_proof_task_count",
            "terminal_tick",
            "same_world_identity_count",
            "episode_reset_count",
            "backbone_initial_sha256",
            "backbone_final_sha256",
            "tick_10000_report_count",
            "resource_sample_count",
            "arms",
            "proof_scope",
            "proof_seed_role",
            "proof_artifact_source",
            "phase_a_artifact_access_count",
            "phase_b_artifact_access_count",
            "scientific_selection_seed_access_count",
            "scientific_outcome_use_count",
        },
        field="persistent island facts",
    )
    if (
        _integer(facts["scheduled_task_count"], field="scheduled tasks", minimum=0)
        != 48
        or _integer(
            facts["completed_proof_task_count"],
            field="completed proof tasks",
            minimum=0,
        )
        != 3
        or _integer(facts["terminal_tick"], field="terminal tick", minimum=0) != 50_000
        or _integer(facts["same_world_identity_count"], field="same worlds", minimum=0)
        != 3
        or _integer(facts["episode_reset_count"], field="episode resets", minimum=0)
        != 0
        or _sha256(facts["backbone_initial_sha256"], field="initial backbone")
        != _sha256(facts["backbone_final_sha256"], field="final backbone")
        or _integer(facts["tick_10000_report_count"], field="10k reports", minimum=0)
        != 3
        or _integer(facts["resource_sample_count"], field="resource samples", minimum=1)
        < 1
        or facts["arms"] != list(_ARMS)
        or facts["proof_scope"] != "engineering_runner_endurance_only"
        or facts["proof_seed_role"] != "open_ecology_proof"
        or facts["proof_artifact_source"]
        != "source_initialized_frozen_width256_actor_film_v1"
        or any(
            _integer(facts[field], field=field, minimum=0) != 0
            for field in (
                "phase_a_artifact_access_count",
                "phase_b_artifact_access_count",
                "scientific_selection_seed_access_count",
                "scientific_outcome_use_count",
            )
        )
    ):
        raise _error("50,000-tick persistent-island execution was not proved")


def _validate_checkpoint_continuation(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "arms",
            "checkpoint_tick",
            "continuation_ticks",
            "component_schema_versions",
            "world_state_match",
            "action_stream_match",
            "environment_rng_draw_match",
            "sampling_rng_draw_match",
            "hidden_state_match",
            "public_feedback_history_match",
            "genome_population_match",
            "evidence_digest_match",
        },
        field="checkpoint continuation facts",
    )
    components = _mapping(
        facts["component_schema_versions"],
        field="checkpoint components",
    )
    if set(components) != {
        "world_state",
        "environment_rng_state",
        "recurrent_policy_state",
        "public_feedback_history",
        "sampling_rng_state",
        "genome_population_snapshot",
        "evidence_writer_continuation_state",
    } or any(not isinstance(value, str) or not value for value in components.values()):
        raise _error("checkpoint does not cover all seven concrete adapters")
    exact_matches = (
        "world_state_match",
        "action_stream_match",
        "environment_rng_draw_match",
        "sampling_rng_draw_match",
        "hidden_state_match",
        "public_feedback_history_match",
        "genome_population_match",
        "evidence_digest_match",
    )
    if (
        facts["arms"] != list(_ARMS)
        or _positive_integer(facts["checkpoint_tick"], field="checkpoint tick") < 1
        or _positive_integer(facts["continuation_ticks"], field="continuation ticks")
        < 1
        or any(facts[field] is not True for field in exact_matches)
    ):
        raise _error("checkpoint continuation equivalence failed")


def _validate_bounded_writer(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "completed_shard_count",
            "resumable_prefix_match",
            "event_kinds_observed",
            "event_count_by_kind",
            "replay_rows_within_bound",
            "replay_bytes_within_bound",
            "campaign_quota_enforced",
            "active_writer_archive_rejection_count",
        },
        field="bounded writer facts",
    )
    counts = _mapping(facts["event_count_by_kind"], field="event counts")
    if (
        _positive_integer(facts["completed_shard_count"], field="shard count") < 2
        or facts["resumable_prefix_match"] is not True
        or facts["event_kinds_observed"] != list(_EVENT_KINDS)
        or set(counts) != set(_EVENT_KINDS)
        or any(
            _positive_integer(counts[kind], field=f"{kind} count") < 1
            for kind in _EVENT_KINDS
        )
        or facts["replay_rows_within_bound"] is not True
        or facts["replay_bytes_within_bound"] is not True
        or facts["campaign_quota_enforced"] is not True
        or _positive_integer(
            facts["active_writer_archive_rejection_count"],
            field="active writer rejections",
        )
        < 1
    ):
        raise _error("bounded writer/event coverage proof failed")


def _validate_causal_rejections(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "real_frozen_policy_state_count",
            "controlled_action_mask_count",
            "controlled_sampling_rng_count",
            "genome_intervention_count",
            "signal_intervention_count",
            "entropy_only_rejection_count",
            "correlation_only_rejection_count",
            "self_echo_rejection_count",
            "missing_contributor_rejection_count",
            "pseudo_replication_rejection_count",
        },
        field="causal evaluator facts",
    )
    if any(_positive_integer(facts[field], field=field) < 1 for field in facts):
        raise _error("causal evaluator rejection battery is incomplete")


def _validate_capture_reexecution(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "cell_order",
            "density_levels",
            "case_count",
            "ticks_per_case",
            "birth_count",
            "death_count",
            "primary_report_sha256",
            "reexecution_report_sha256",
            "report_bytes_match",
            "scientific_selection_seed_access_count",
            "model_state_mismatch_count",
            "trajectory_mismatch_count",
            "action_rng_mismatch_count",
            "genome_provenance_mismatch_count",
        },
        field="capture reexecution facts",
    )
    mismatch_fields = (
        "model_state_mismatch_count",
        "trajectory_mismatch_count",
        "action_rng_mismatch_count",
        "genome_provenance_mismatch_count",
    )
    if (
        facts["cell_order"] != list(_CELLS)
        or facts["density_levels"] != [32, 64, 128]
        or _integer(facts["case_count"], field="case count", minimum=0) != 12
        or _integer(facts["ticks_per_case"], field="ticks", minimum=0) != 128
        or _positive_integer(facts["birth_count"], field="birth count") < 1
        or _positive_integer(facts["death_count"], field="death count") < 1
        or _sha256(facts["primary_report_sha256"], field="primary report")
        != _sha256(facts["reexecution_report_sha256"], field="reexecution report")
        or facts["report_bytes_match"] is not True
        or _integer(
            facts["scientific_selection_seed_access_count"],
            field="selection seed accesses",
            minimum=0,
        )
        != 0
        or any(
            _integer(facts[field], field=field, minimum=0) != 0
            for field in mismatch_fields
        )
    ):
        raise _error("capture noninterference reexecution failed")


def _validate_preregistration_roundtrip(
    facts: Mapping[str, object],
    preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "roundtrip_exact_digest",
            "run_matrix_row_count",
            "unique_task_id_count",
            "seed_registry_sha256",
            "configuration_sha256",
            "unknown_input_rejection_count",
            "dirty_source_rejection_count",
            "stale_source_rejection_count",
            "duplicate_key_rejection_count",
        },
        field="preregistration roundtrip facts",
    )
    training = _mapping(preregistration["training"], field="training")
    run_matrix = _sequence(training["run_matrix"], field="training.run_matrix")
    seed_contract = _mapping(preregistration["seed_contract"], field="seed_contract")
    if (
        facts["roundtrip_exact_digest"] != preregistration["exact_digest"]
        or _integer(facts["run_matrix_row_count"], field="matrix rows", minimum=0)
        != len(run_matrix)
        or _integer(facts["unique_task_id_count"], field="task ids", minimum=0)
        != len(run_matrix)
        or facts["seed_registry_sha256"] != seed_contract["registry_sha256"]
        or facts["configuration_sha256"] != preregistration["configuration_sha256"]
        or any(
            _positive_integer(facts[field], field=field) < 1
            for field in (
                "unknown_input_rejection_count",
                "dirty_source_rejection_count",
                "stale_source_rejection_count",
                "duplicate_key_rejection_count",
            )
        )
    ):
        raise _error("machine preregistration roundtrip/fail-closed proof failed")


def _validate_exact_sha_path(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "fresh_checkout_count",
            "dirty_checkout_count",
            "source_manifest_match_count",
            "import_root_match_count",
            "end_to_end_training_benchmark_count",
            "selection_512_benchmark_count",
            "selection_2000_benchmark_count",
            "output_lock_proof_count",
            "drive_quota_proof_count",
            "uploader_proof_count",
            "verification_before_prune_proof_count",
            "terminal_aggregate_proof_count",
        },
        field="exact SHA path facts",
    )
    if (
        _positive_integer(facts["fresh_checkout_count"], field="fresh checkouts") < 1
        or _integer(facts["dirty_checkout_count"], field="dirty checkouts", minimum=0)
        != 0
        or any(
            _positive_integer(facts[field], field=field) < 1
            for field in facts
            if field not in {"fresh_checkout_count", "dirty_checkout_count"}
        )
    ):
        raise _error("exact-SHA end-to-end operational path is incomplete")


def _validate_multi_host_ci(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "intended_host_class_count",
            "tested_host_class_count",
            "runtime_contract_sha256_by_host_class",
            "semantic_evidence_sha256_by_host_class",
            "cross_host_semantic_mismatch_count",
            "torch_ci_lane_required",
            "torch_ci_test_count",
            "torch_ci_failure_count",
            "torch_ci_skip_count",
        },
        field="multi-host/CI facts",
    )
    intended = _positive_integer(
        facts["intended_host_class_count"],
        field="intended host classes",
    )
    tested = _positive_integer(
        facts["tested_host_class_count"],
        field="tested host classes",
    )
    runtime = _mapping(
        facts["runtime_contract_sha256_by_host_class"],
        field="runtime digests",
    )
    semantics = _mapping(
        facts["semantic_evidence_sha256_by_host_class"],
        field="semantic digests",
    )
    if (
        intended < 2
        or tested != intended
        or len(runtime) != intended
        or set(runtime) != set(semantics)
        or any(
            _sha256(value, field="runtime digest") == "" for value in runtime.values()
        )
        or len({_sha256(value, field="runtime digest") for value in runtime.values()})
        != 1
        or len(
            {_sha256(value, field="semantic digest") for value in semantics.values()}
        )
        != 1
        or _integer(
            facts["cross_host_semantic_mismatch_count"],
            field="cross-host mismatches",
            minimum=0,
        )
        != 0
        or facts["torch_ci_lane_required"] is not True
        or _positive_integer(facts["torch_ci_test_count"], field="torch test count") < 1
        or _integer(facts["torch_ci_failure_count"], field="torch failures", minimum=0)
        != 0
        or _integer(facts["torch_ci_skip_count"], field="torch skips", minimum=0) != 0
    ):
        raise _error("multi-host equivalence or non-optional Torch CI failed")


def _validate_throughput(
    facts: Mapping[str, object],
    preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "preliminary_throughput_gate_digest",
            "phase_a_training_projection_authoritative",
            "selection_horizons",
            "selection_primary_and_replay_measured",
            "semantic_mismatch_count",
            "projected_full_matrix_median_seconds",
            "projected_full_matrix_slow_plus_margin_seconds",
            "maximum_gpu_memory_share",
            "maximum_host_ram_share",
            "swap_delta_bytes",
            "temperature_below_throttle",
            "xid_count",
            "oom_count",
            "nonfinite_count",
            "evidence_write_error_count",
        },
        field="throughput facts",
    )
    preliminary = _mapping(
        preregistration["throughput_gate"],
        field="throughput gate",
    )
    if (
        facts["preliminary_throughput_gate_digest"] != preliminary["exact_digest"]
        or facts["phase_a_training_projection_authoritative"] is not True
        or facts["selection_horizons"] != [512, 2000]
        or facts["selection_primary_and_replay_measured"] is not True
        or _integer(
            facts["semantic_mismatch_count"], field="semantic mismatches", minimum=0
        )
        != 0
        or _positive_number(
            facts["projected_full_matrix_median_seconds"],
            field="median projection",
        )
        > 72 * 60 * 60
        or _positive_number(
            facts["projected_full_matrix_slow_plus_margin_seconds"],
            field="slow projection",
        )
        > 96 * 60 * 60
        or not 0
        <= _number(facts["maximum_gpu_memory_share"], field="GPU memory share")
        < 0.80
        or not 0 <= _number(facts["maximum_host_ram_share"], field="RAM share") < 0.80
        or _integer(facts["swap_delta_bytes"], field="swap delta", minimum=0) != 0
        or facts["temperature_below_throttle"] is not True
        or any(
            _integer(facts[field], field=field, minimum=0) != 0
            for field in (
                "xid_count",
                "oom_count",
                "nonfinite_count",
                "evidence_write_error_count",
            )
        )
    ):
        raise _error("full throughput/resource gate failed")


def _validate_storage(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "checked_at_utc",
            "google_drive_free_bytes",
            "projected_active_storage_bytes",
            "target_filesystem_capacity_bytes",
            "target_filesystem_free_bytes",
            "required_target_free_bytes",
            "drive_measurement_command_sha256",
            "filesystem_measurement_command_sha256",
        },
        field="storage facts",
    )
    checked = _parse_utc(facts["checked_at_utc"], field="storage.checked_at_utc")
    if authorization_time is not None and (
        checked > authorization_time
        or authorization_time - checked > timedelta(hours=24)
    ):
        raise _error("storage capacity attestation is stale")
    capacity = _positive_integer(
        facts["target_filesystem_capacity_bytes"],
        field="target capacity",
    )
    target_free = _positive_integer(
        facts["target_filesystem_free_bytes"],
        field="target free",
    )
    required = max(100 * 1024**3, (capacity + 4) // 5)
    if (
        _positive_integer(facts["google_drive_free_bytes"], field="Drive free")
        < 300 * 1024**3
        or _positive_integer(
            facts["projected_active_storage_bytes"],
            field="projected active storage",
        )
        > 200 * 1024**3
        or facts["required_target_free_bytes"] != required
        or target_free < required
        or target_free > capacity
    ):
        raise _error("storage capacity thresholds failed")
    _sha256(facts["drive_measurement_command_sha256"], field="Drive measurement")
    _sha256(
        facts["filesystem_measurement_command_sha256"],
        field="filesystem measurement",
    )


def _validate_output_lock(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "concurrent_contender_count",
            "admitted_writer_count",
            "contention_rejection_count",
            "identity_drift_rejection_count",
            "lock_release_reacquire_count",
        },
        field="output lock facts",
    )
    if (
        _positive_integer(facts["concurrent_contender_count"], field="contenders") < 2
        or _integer(facts["admitted_writer_count"], field="admitted", minimum=0) != 1
        or _positive_integer(
            facts["contention_rejection_count"], field="contention rejection"
        )
        < 1
        or _positive_integer(
            facts["identity_drift_rejection_count"], field="identity rejection"
        )
        < 1
        or _positive_integer(facts["lock_release_reacquire_count"], field="reacquire")
        < 1
    ):
        raise _error("output lock contention semantics failed")


def _validate_uploader(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "closed_bundle_count",
            "active_bundle_rejection_count",
            "remote_object_names",
            "remote_unique_id_count",
            "stream_readback_sha256_match_count",
            "rclone_check_match_count",
            "rclone_check_difference_count",
            "symlink_rejection_count",
            "hardlink_rejection_count",
            "special_file_rejection_count",
            "mount_crossing_rejection_count",
        },
        field="uploader facts",
    )
    if (
        _positive_integer(facts["closed_bundle_count"], field="closed bundles") < 1
        or _positive_integer(
            facts["active_bundle_rejection_count"], field="active rejection"
        )
        < 1
        or facts["remote_object_names"]
        != ["bundle.tar.zst", "bundle.manifest.json", "bundle.sha256"]
        or _integer(facts["remote_unique_id_count"], field="unique objects", minimum=0)
        != 3
        or _integer(
            facts["stream_readback_sha256_match_count"],
            field="readback matches",
            minimum=0,
        )
        != 3
        or _integer(
            facts["rclone_check_match_count"], field="rclone matches", minimum=0
        )
        != 3
        or _integer(
            facts["rclone_check_difference_count"],
            field="rclone differences",
            minimum=0,
        )
        != 0
        or any(
            _positive_integer(facts[field], field=field) < 1
            for field in (
                "symlink_rejection_count",
                "hardlink_rejection_count",
                "special_file_rejection_count",
                "mount_crossing_rejection_count",
            )
        )
    ):
        raise _error("immutable Drive uploader proof failed")


def _validate_prune(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "preverification_prune_attempt_count",
            "preverification_prune_rejection_count",
            "postverification_prune_count",
            "failed_verification_source_retained_count",
            "remote_inventory_exact_count",
        },
        field="verification-before-prune facts",
    )
    attempts = _positive_integer(
        facts["preverification_prune_attempt_count"],
        field="preverification attempts",
    )
    if (
        _integer(
            facts["preverification_prune_rejection_count"],
            field="preverification rejections",
            minimum=0,
        )
        != attempts
        or _positive_integer(
            facts["postverification_prune_count"], field="verified prunes"
        )
        < 1
        or _positive_integer(
            facts["failed_verification_source_retained_count"],
            field="retained sources",
        )
        < 1
        or _integer(
            facts["remote_inventory_exact_count"], field="inventory count", minimum=0
        )
        != 3
    ):
        raise _error("verification-before-prune ordering failed")


def _validate_terminal_aggregate(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "canonical_terminal_count",
            "validated_terminal_count",
            "missing_terminal_rejection_count",
            "surplus_terminal_rejection_count",
            "digest_tamper_rejection_count",
            "source_drift_rejection_count",
            "prefix_drift_rejection_count",
            "fresh_exact_cpu_reexecution_count",
            "proof_scope",
            "scientific_terminal_access_count",
        },
        field="terminal aggregate facts",
    )
    if (
        _integer(
            facts["canonical_terminal_count"], field="canonical terminals", minimum=0
        )
        != 16
        or _integer(
            facts["validated_terminal_count"], field="validated terminals", minimum=0
        )
        != 16
        or any(
            _positive_integer(facts[field], field=field) < 1
            for field in (
                "missing_terminal_rejection_count",
                "surplus_terminal_rejection_count",
                "digest_tamper_rejection_count",
                "source_drift_rejection_count",
                "prefix_drift_rejection_count",
            )
        )
        or _integer(
            facts["fresh_exact_cpu_reexecution_count"],
            field="fresh exact CPU reexecutions",
            minimum=0,
        )
        != 16
        or facts["proof_scope"] != "engineering_terminal_matrix_fixture_only"
        or _integer(
            facts["scientific_terminal_access_count"],
            field="scientific terminal accesses",
            minimum=0,
        )
        != 0
    ):
        raise _error("terminal aggregate validator proof failed")


_SEMANTIC_VALIDATORS = {
    "cross_surface_and_self_echo": _validate_cross_surface,
    "runtime_genome_and_action_source": _validate_genome_runtime,
    "critic_gradient_and_density_schedule": _validate_critic_gradient,
    "fixed_batch_equivalence_and_speed": _validate_fixed_batch,
    "persistent_island_50000": _validate_persistent_50000,
    "checkpoint_continuation_equivalence": _validate_checkpoint_continuation,
    "bounded_writer_event_coverage": _validate_bounded_writer,
    "causal_evaluator_rejection_battery": _validate_causal_rejections,
    "capture_noninterference_reexecution": _validate_capture_reexecution,
    "preregistration_roundtrip_fail_closed": _validate_preregistration_roundtrip,
    "exact_sha_operational_path": _validate_exact_sha_path,
    "multi_host_equivalence_and_torch_ci": _validate_multi_host_ci,
    "phase_a_full_throughput": _validate_throughput,
    "campaign_storage_capacity": _validate_storage,
    "output_lock_contention": _validate_output_lock,
    "immutable_drive_uploader": _validate_uploader,
    "verification_before_prune": _validate_prune,
    "terminal_aggregate_validator": _validate_terminal_aggregate,
}


def _dependency_entries(
    evidence_index: Mapping[str, object],
    *,
    require_complete: bool = True,
) -> dict[str, Mapping[str, object]]:
    entries = _sequence(
        evidence_index.get("dependency_reports"),
        field="evidence_index.dependency_reports",
    )
    result: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        parsed = _mapping(entry, field="dependency report entry")
        _exact_key_set(
            parsed,
            {"dependency_id", "reports"},
            field="dependency report entry",
        )
        dependency_id = parsed.get("dependency_id")
        if (
            not isinstance(dependency_id, str)
            or dependency_id not in DEPENDENCY_EVIDENCE_KINDS
            or dependency_id in result
        ):
            raise _error("dependency evidence index identity drifted")
        result[dependency_id] = parsed
    expected_order = tuple(
        dependency_id
        for dependency_id in DEPENDENCY_EVIDENCE_KINDS
        if dependency_id in result
    )
    if tuple(result) != expected_order or (
        require_complete and tuple(result) != tuple(DEPENDENCY_EVIDENCE_KINDS)
    ):
        raise _error("dependency evidence index is incomplete or unordered")
    return result


def _gate_entries(
    evidence_index: Mapping[str, object],
    *,
    require_complete: bool = True,
) -> dict[str, Mapping[str, object]]:
    entries = _sequence(
        evidence_index.get("operational_reports"),
        field="evidence_index.operational_reports",
    )
    result: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        parsed = _mapping(entry, field="operational report entry")
        _exact_key_set(
            parsed,
            {"gate_name", "report"},
            field="operational report entry",
        )
        gate_name = parsed.get("gate_name")
        if (
            not isinstance(gate_name, str)
            or gate_name not in OPERATIONAL_EVIDENCE_KINDS
            or gate_name in result
        ):
            raise _error("operational evidence index identity drifted")
        result[gate_name] = parsed
    expected_order = tuple(gate for gate in _GATES if gate in result)
    if tuple(result) != expected_order or (
        require_complete and tuple(result) != _GATES
    ):
        raise _error("operational evidence index is incomplete or unordered")
    return result


def _reports_from_entry(
    entry: Mapping[str, object],
) -> list[Mapping[str, object]]:
    return [
        _mapping(report, field="indexed dependency report")
        for report in _sequence(entry.get("reports"), field="dependency reports")
    ]


def _source_binding(
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    return dict(_mapping(preregistration["source"], field="preregistration.source"))


def _authority_verifier_availability() -> dict[str, bool]:
    return {
        kind: callable(REPORT_AUTHORITY_VERIFIERS.get(kind))
        for kind in PROOF_PRODUCER_AVAILABILITY
    }


def _surface_mapping(value: object, *, field: str) -> Mapping[str, object]:
    parsed = _mapping(value, field=field)
    _exact_key_set(parsed, set(_SURFACES), field=field)
    return parsed


def _regular_bundle_file(base: Path, relative: str, *, field: str) -> Path:
    root = base.resolve()
    candidate_relative = Path(relative)
    if (
        candidate_relative.is_absolute()
        or not candidate_relative.parts
        or any(part in {"", ".", ".."} for part in candidate_relative.parts)
    ):
        raise _error(f"{field} path is unsafe")
    current = root
    for part in candidate_relative.parts:
        current = current / part
        if current.is_symlink():
            raise _error(f"{field} contains a symbolic link")
    resolved = current.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise _error(f"{field} escaped its sealed root") from error
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_nlink != 1:
        raise _error(f"{field} must be a regular file")
    return resolved


def _require_bound_payload_file(
    path: Path,
    payload: Mapping[str, object],
    *,
    field: str,
) -> Path:
    contract = _contract()
    if path.is_symlink():
        raise contract.OpenEcologyPhaseAError(f"{field} must not be a symbolic link")
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_nlink != 1:
        raise contract.OpenEcologyPhaseAError(
            f"{field} must be one regular non-hardlinked file"
        )
    disk_payload = contract._load_strict_json(resolved)
    if disk_payload != dict(payload):
        raise contract.OpenEcologyPhaseAError(
            f"{field} supplied payload differs from its exact file bytes"
        )
    return resolved


def _exact_key_set(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise _error(
            f"{field} fields differ: missing={sorted(expected - observed)}, "
            f"surplus={sorted(observed - expected)}"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(f"{field} must be a sequence")
    return value


def _integer(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(f"{field} must be an integer >= {minimum}")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    return _integer(value, field=field, minimum=1)


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{field} must be finite numeric")
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise _error(f"{field} must be finite numeric")
    return parsed


def _positive_number(value: object, *, field: str) -> float:
    parsed = _number(value, field=field)
    if parsed <= 0:
        raise _error(f"{field} must be positive")
    return parsed


def _sha256(value: object, *, field: str) -> str:
    contract = _contract()
    return contract._sha256(value, field=field)


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise _error(f"{field} must be UTC RFC3339 seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise _error(f"{field} must be UTC RFC3339 seconds") from error
    return parsed


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _error("capture proof is not canonical JSON") from error


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _require_live_source_twice(
    preregistration: Mapping[str, object],
    *,
    before: bool,
) -> None:
    contract = _contract()
    try:
        contract._require_live_source(preregistration)
    except contract.OpenEcologyPhaseAError as error:
        position = "before" if before else "after"
        raise contract.OpenEcologyPhaseAError(
            f"Phase A source drifted {position} readiness evidence assembly"
        ) from error


def _error(message: str) -> Exception:
    return _contract().OpenEcologyPhaseAError(message)


def _contract() -> "phase_a_contract":
    from evolution_sim.mind import open_ecology_phase_a

    return open_ecology_phase_a


__all__ = [
    "DEPENDENCY_EVIDENCE_KINDS",
    "OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_LAUNCH_READINESS_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION",
    "OPERATIONAL_EVIDENCE_KINDS",
    "PROOF_PRODUCER_AVAILABILITY",
    "REPORT_AUTHORITY_VERIFIERS",
    "ReportAuthorityVerifier",
    "build_evidence_index",
    "build_launch_authorization",
    "launch_readiness",
    "produce_capture_noninterference_reexecution_report",
    "validate_launch_authorization",
]
