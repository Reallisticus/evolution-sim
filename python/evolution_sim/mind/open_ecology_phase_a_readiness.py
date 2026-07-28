"""Fail-closed evidence authority for the open-ecology Phase-A launch.

This module does not manufacture behavioral evidence. It defines the exact
machine-readable reports that independent producers must emit, validates their
substantive observations, and assembles a launch authorization only when every
dependency and operational gate is present and source-bound.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING

from evolution_sim.mind.open_ecology_phase_a_contract import (
    OpenEcologyPhaseAError,
    require_lowercase_sha256,
)
from evolution_sim.mind.provenance import stable_payload_digest

if TYPE_CHECKING:
    from evolution_sim.mind import open_ecology_phase_a as phase_a_contract


OPEN_ECOLOGY_PHASE_A_LEGACY_READINESS_REPORT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_readiness_report_v1"
)
OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_readiness_report_v2"
)
OPEN_ECOLOGY_RAW_EVIDENCE_MANIFEST_SCHEMA_VERSION = (
    "mind_v3_open_ecology_raw_evidence_manifest_v1"
)
OPEN_ECOLOGY_RAW_EVIDENCE_BYTE_CONTRACT = (
    "sha256_exact_bytes_sorted_manifest_inventory_v1"
)
OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_evidence_index_v2"
)
OPEN_ECOLOGY_PHASE_A_LAUNCH_READINESS_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_launch_readiness_v3"
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
DEPENDENCY_EVIDENCE_KINDS: Mapping[str, tuple[str, ...]] = {
    "readiness_dependency_01": ("cross_surface_and_self_echo",),
    "readiness_dependency_02": ("runtime_genome_and_action_source",),
    "readiness_dependency_03": ("critic_gradient_and_density_schedule",),
    "readiness_dependency_04": ("fixed_batch_equivalence_and_speed",),
    "readiness_dependency_09": ("preregistration_roundtrip_fail_closed",),
    "readiness_dependency_10": ("exact_sha_phase_a_training_and_torch_ci",),
}

SELECTION_DEPENDENCY_EVIDENCE_KINDS: Mapping[str, tuple[str, ...]] = {
    "readiness_dependency_08": (
        "causal_evaluator_rejection_battery",
        "capture_noninterference_reexecution",
    ),
}

PHASE_D_DEPENDENCY_EVIDENCE_KINDS: Mapping[str, tuple[str, ...]] = {
    "readiness_dependency_05": ("persistent_island_50000",),
    "readiness_dependency_06": ("checkpoint_continuation_equivalence",),
    "readiness_dependency_07": ("bounded_writer_event_coverage",),
    "readiness_dependency_10": ("phase_d_host_equivalence_or_homogeneous_contract",),
}

OPERATIONAL_EVIDENCE_KINDS: Mapping[str, str] = {
    "throughput": "phase_a_training_throughput",
    "storage": "campaign_storage_capacity",
    "output_lock": "output_lock_contention",
}

SELECTION_OPERATIONAL_EVIDENCE_KINDS: Mapping[str, str] = {
    "selection_throughput": "phase_a_selection_throughput",
}

PHASE_D_OPERATIONAL_EVIDENCE_KINDS: Mapping[str, str] = {
    "phase_d_throughput": "phase_d_full_throughput",
    "storage": "campaign_storage_capacity",
    "output_lock": "output_lock_contention",
    "immutable_uploader": "immutable_drive_uploader",
    "terminal_aggregate_validator": "terminal_aggregate_validator",
}
_GATES = tuple(OPERATIONAL_EVIDENCE_KINDS)

_DEPENDENCY_EVIDENCE_CATALOG = (
    DEPENDENCY_EVIDENCE_KINDS,
    SELECTION_DEPENDENCY_EVIDENCE_KINDS,
    PHASE_D_DEPENDENCY_EVIDENCE_KINDS,
)
_OPERATIONAL_EVIDENCE_CATALOG = (
    OPERATIONAL_EVIDENCE_KINDS,
    SELECTION_OPERATIONAL_EVIDENCE_KINDS,
    PHASE_D_OPERATIONAL_EVIDENCE_KINDS,
)
_ALL_EVIDENCE_KINDS = tuple(
    (
        *(
            evidence_kind
            for catalog in _DEPENDENCY_EVIDENCE_CATALOG
            for evidence_kinds in catalog.values()
            for evidence_kind in evidence_kinds
        ),
        *(
            evidence_kind
            for catalog in _OPERATIONAL_EVIDENCE_CATALOG
            for evidence_kind in catalog.values()
        ),
    )
)
_ALL_EVIDENCE_KIND_SET = frozenset(_ALL_EVIDENCE_KINDS)
_PHASE_A_REQUIRED_EVIDENCE_KINDS = frozenset(
    (
        *(kind for kinds in DEPENDENCY_EVIDENCE_KINDS.values() for kind in kinds),
        *OPERATIONAL_EVIDENCE_KINDS.values(),
    )
)
_MAX_AUTHORITY_JSON_BYTES = 16 * 1024 * 1024
_MAX_RAW_EVIDENCE_FILE_BYTES = 256 * 1024 * 1024
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_D10_LIVE_STABLE_FACT_FIELDS = frozenset(
    {
        "fresh_checkout_count",
        "dirty_checkout_count",
        "source_manifest_match_count",
        "import_root_match_count",
        "intended_training_host_classes",
        "tested_training_host_classes",
        "full_shape_training_benchmark_count_by_host_class",
        "runtime_contract_sha256_by_host_class",
        "semantic_evidence_sha256_by_host_class",
        "cross_host_semantic_mismatch_count",
        "torch_ci_lane_required",
        "torch_ci_test_count",
        "torch_ci_failure_count",
        "torch_ci_skip_count",
    }
)
_THROUGHPUT_LIVE_STABLE_FACT_FIELDS = frozenset(
    {
        "preliminary_throughput_gate_digest",
        "measured_host_class",
        "fresh_full_shape_repetition_count",
        "phase_a_training_projection_authoritative",
        "selection_projection_included",
        "semantic_mismatch_count",
    }
)
_STORAGE_LIVE_STABLE_FACT_FIELDS = frozenset(
    {
        "measurement_location",
        "target_ssh",
        "remote_hostname",
        "archive_authority_sha256",
        "ssh_effective_config_sha256",
        "ssh_connection",
        "rclone_config_path",
        "rclone_config_sha256",
        "projected_active_storage_bytes",
        "target_filesystem_path",
        "target_filesystem_capacity_bytes",
        "required_target_free_bytes",
    }
)
_OUTPUT_LOCK_LIVE_STABLE_FACT_FIELDS = frozenset(
    {
        "concurrent_contender_count",
        "admitted_writer_count",
        "contention_rejection_count",
        "identity_drift_rejection_count",
        "lock_release_reacquire_count",
    }
)


@dataclass(frozen=True, slots=True)
class _CapturedRegularFile:
    path: Path
    relative_path: str
    payload: bytes
    sha256: str
    byte_length: int


@dataclass(frozen=True, slots=True)
class _OwnedStagingDirectory:
    path: Path
    parent_descriptor: int
    directory_descriptor: int
    parent_device: int
    parent_inode: int
    device: int
    inode: int


# A producer can write a report without making that report authoritative.  An
# authority verifier must independently reconstruct the report's substantive
# facts from exact, referenced raw evidence.  A canonical digest is only an
# integrity checksum and must never be treated as a signature.
ProofProducer = Callable[..., Mapping[str, object]]
ReportAuthorityVerifier = Callable[
    [Path, Mapping[str, object], datetime | None],
    Mapping[str, object],
]


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
        "schema_version": (OPEN_ECOLOGY_PHASE_A_LEGACY_READINESS_REPORT_SCHEMA_VERSION),
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


def build_open_ecology_raw_evidence_manifest(
    preregistration: Mapping[str, object],
    *,
    evidence_kind: str,
    bundle_root: str | Path,
    produced_at_utc: str,
    producer_name: str,
    producer_contract: str,
    raw_files: Mapping[str, str | Path],
) -> dict[str, object]:
    """Bind an exact, closed raw-file inventory for one evidence report."""

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    if evidence_kind not in _ALL_EVIDENCE_KINDS:
        raise contract.OpenEcologyPhaseAError(
            f"unknown raw-evidence kind {evidence_kind!r}"
        )
    _parse_utc(produced_at_utc, field="raw evidence produced_at_utc")
    if not isinstance(producer_name, str) or not producer_name:
        raise contract.OpenEcologyPhaseAError(
            "raw evidence producer_name must be non-empty"
        )
    if not isinstance(producer_contract, str) or not producer_contract:
        raise contract.OpenEcologyPhaseAError(
            "raw evidence producer_contract must be non-empty"
        )
    raw_bundle_root = Path(bundle_root)
    if raw_bundle_root.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            "raw-evidence bundle root must be one real directory"
        )
    root = raw_bundle_root.resolve()
    if not root.is_dir():
        raise contract.OpenEcologyPhaseAError(
            "raw-evidence bundle root must be one real directory"
        )
    if not raw_files:
        raise contract.OpenEcologyPhaseAError(
            "raw-evidence manifest requires concrete raw files"
        )
    entries: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    for role in sorted(raw_files):
        if (
            not isinstance(role, str)
            or not role
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in role
            )
        ):
            raise contract.OpenEcologyPhaseAError(
                "raw-evidence roles must use lowercase safe identifiers"
            )
        raw_path = Path(raw_files[role])
        try:
            relative = (
                raw_path.relative_to(root) if raw_path.is_absolute() else raw_path
            )
        except ValueError as error:
            raise contract.OpenEcologyPhaseAError(
                f"raw-evidence role {role!r} escaped its bundle"
            ) from error
        relative_text = relative.as_posix()
        if not relative.parts or relative.parts[0] != "raw":
            raise contract.OpenEcologyPhaseAError(
                f"raw-evidence role {role!r} is not beneath raw/"
            )
        if relative_text in observed_paths:
            raise contract.OpenEcologyPhaseAError(
                "raw-evidence roles must reference distinct files"
            )
        resolved = _regular_bundle_file(
            root,
            relative_text,
            field=f"raw evidence {role}",
        )
        reference = contract._file_reference(resolved, base=root)
        entries.append({"role": role, **reference})
        observed_paths.add(relative_text)
    actual_paths = _raw_bundle_file_inventory(root)
    if observed_paths != actual_paths:
        raise contract.OpenEcologyPhaseAError(
            "raw-evidence manifest inventory differs from exact raw/ contents"
        )
    manifest: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_RAW_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _source_binding(preregistration),
        "produced_at_utc": produced_at_utc,
        "producer": {
            "name": producer_name,
            "contract": producer_contract,
        },
        "canonical_preimage_contract": OPEN_ECOLOGY_RAW_EVIDENCE_BYTE_CONTRACT,
        "files": entries,
    }
    manifest["exact_digest"] = stable_payload_digest(manifest)
    return manifest


def _produce_preregistration_roundtrip_fail_closed_report_in_directory(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    _precreated_root: bool = False,
) -> dict[str, object]:
    """Produce a sealed D9 bundle and its independently reconstructable report."""

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source_twice(preregistration, before=True)
    destination = Path(output_directory)
    raw_parent = destination.parent
    if raw_parent.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            "D9 output parent must be one real existing directory"
        )
    parent = raw_parent.resolve()
    if not parent.is_dir():
        raise contract.OpenEcologyPhaseAError(
            "D9 output parent must be one real existing directory"
        )
    root = parent / destination.name
    if _precreated_root:
        if root.is_symlink() or not root.is_dir():
            raise contract.OpenEcologyPhaseAError(
                "D9 precreated output directory authority drifted"
            )
    else:
        if root.exists() or root.is_symlink():
            raise contract.OpenEcologyPhaseAError(
                "D9 output directory must be new and absent"
            )
        root.mkdir()
    raw_root = root / "raw"
    raw_root.mkdir()
    produced_at_utc = _format_utc(_utc_now())

    builder_inputs = {
        "source_commit": _mapping(
            preregistration["source"],
            field="preregistration.source",
        )["commit"],
        "source_manifest_sha256": _mapping(
            preregistration["source"],
            field="preregistration.source",
        )["manifest_sha256"],
        "archive_tool_authority_sha256": _mapping(
            preregistration["source"],
            field="preregistration.source",
        )["archive_tool_authority_sha256"],
        "runtime_contract": dict(
            _mapping(
                preregistration["runtime_contract"],
                field="preregistration.runtime_contract",
            )
        ),
        "throughput_gate": dict(
            _mapping(
                preregistration["throughput_gate"],
                field="preregistration.throughput_gate",
            )
        ),
    }
    unknown_input = json.loads(_canonical_json_bytes(preregistration))
    if not isinstance(unknown_input, dict):
        raise contract.OpenEcologyPhaseAError(
            "D9 preregistration did not normalize to an object"
        )
    unknown_input["unexpected_authority_input"] = True
    unknown_input.pop("exact_digest", None)
    unknown_input["exact_digest"] = stable_payload_digest(unknown_input)
    source = _mapping(preregistration["source"], field="preregistration.source")
    stale_commit = "0" * 40 if source["commit"] != "0" * 40 else "1" * 40
    source_observations = {
        "clean": {
            "observed_commit": source["commit"],
            "observed_status_porcelain": "",
            "observed_manifest_sha256": source["manifest_sha256"],
            "observed_preregistration_sha256": (
                contract.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256
            ),
        },
        "dirty": {
            "observed_commit": source["commit"],
            "observed_status_porcelain": " M d9-hostile-dirty-source",
            "observed_manifest_sha256": source["manifest_sha256"],
            "observed_preregistration_sha256": (
                contract.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256
            ),
        },
        "stale": {
            "observed_commit": stale_commit,
            "observed_status_porcelain": "",
            "observed_manifest_sha256": source["manifest_sha256"],
            "observed_preregistration_sha256": (
                contract.OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256
            ),
        },
    }
    raw_files = {
        "builder_inputs": raw_root / "builder-inputs.json",
        "duplicate_key_input": raw_root / "duplicate-key-input.json",
        "preregistration": raw_root / "preregistration.json",
        "source_observations": raw_root / "source-observations.json",
        "unknown_input": raw_root / "unknown-input.json",
    }
    _write_new_json(raw_files["builder_inputs"], builder_inputs)
    _write_new_bytes(
        raw_files["duplicate_key_input"],
        b'{"schema_version":"first","schema_version":"second"}\n',
    )
    _write_new_json(raw_files["preregistration"], preregistration)
    _write_new_json(raw_files["source_observations"], source_observations)
    _write_new_json(raw_files["unknown_input"], unknown_input)
    manifest = build_open_ecology_raw_evidence_manifest(
        preregistration,
        evidence_kind="preregistration_roundtrip_fail_closed",
        bundle_root=root,
        produced_at_utc=produced_at_utc,
        producer_name="produce_preregistration_roundtrip_fail_closed_report",
        producer_contract=("open_ecology_preregistration_roundtrip_raw_producer_v1"),
        raw_files=raw_files,
    )
    manifest_path = root / "raw-evidence-manifest.json"
    _write_new_json(manifest_path, manifest)
    reconstructed = _reconstruct_preregistration_roundtrip_from_manifest(
        manifest_path,
        preregistration=preregistration,
        authorization_time=None,
        expected_produced_at_utc=produced_at_utc,
    )
    report: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION,
        "evidence_kind": "preregistration_roundtrip_fail_closed",
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _source_binding(preregistration),
        "produced_at_utc": produced_at_utc,
        "raw_evidence_manifest": contract._file_reference(
            manifest_path,
            base=root,
        ),
        "facts": reconstructed["facts"],
    }
    report["exact_digest"] = stable_payload_digest(report)
    report_path = root / "report.json"
    _write_new_json(report_path, report)
    _validate_report(
        report,
        expected_kind="preregistration_roundtrip_fail_closed",
        preregistration=preregistration,
        authorization_time=None,
        report_path=report_path,
    )
    _require_live_source_twice(preregistration, before=False)
    return report


def produce_preregistration_roundtrip_fail_closed_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    """Atomically publish one independently reconstructable D9 bundle."""

    return _publish_new_authority_bundle(
        preregistration,
        output_directory=output_directory,
        field="D9",
        producer=lambda staging: (
            _produce_preregistration_roundtrip_fail_closed_report_in_directory(
                preregistration,
                output_directory=staging,
                _precreated_root=True,
            )
        ),
    )


def produce_cross_surface_and_self_echo_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    """Run and publish the independently reproducible D1 behavioral probe."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        produce_cross_surface_and_self_echo_report as produce,
    )

    return produce(preregistration, output_directory=output_directory)


def verify_cross_surface_and_self_echo_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reexecute D1 and reconstruct authority from its captured observations."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        verify_cross_surface_and_self_echo_report as verify,
    )

    return verify(report_path, preregistration, authorization_time)


def produce_runtime_genome_and_action_source_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    """Run and publish the independently reproducible D2 behavioral probe."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        produce_runtime_genome_and_action_source_report as produce,
    )

    return produce(preregistration, output_directory=output_directory)


def verify_runtime_genome_and_action_source_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reexecute D2 and reconstruct authority from its captured observations."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        verify_runtime_genome_and_action_source_report as verify,
    )

    return verify(report_path, preregistration, authorization_time)


def produce_critic_gradient_and_density_schedule_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    """Run and publish the independently reproducible D3 behavioral probe."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        produce_critic_gradient_and_density_schedule_report as produce,
    )

    return produce(preregistration, output_directory=output_directory)


def verify_critic_gradient_and_density_schedule_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reexecute D3 and reconstruct authority from its captured observations."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        verify_critic_gradient_and_density_schedule_report as verify,
    )

    return verify(report_path, preregistration, authorization_time)


def produce_fixed_batch_equivalence_and_speed_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    device: str = "cuda",
) -> dict[str, object]:
    """Run and publish the independently reproducible D4 behavioral probe."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        produce_fixed_batch_equivalence_and_speed_report as produce,
    )

    return produce(
        preregistration,
        output_directory=output_directory,
        device=device,
    )


def verify_fixed_batch_equivalence_and_speed_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reexecute D4 and reconstruct authority from its captured observations."""

    from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
        verify_fixed_batch_equivalence_and_speed_report as verify,
    )

    return verify(report_path, preregistration, authorization_time)


def produce_exact_sha_phase_a_training_and_torch_ci_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    host_class: str,
    heritable_benchmark_path: str | Path,
    zero_all_benchmark_path: str | Path,
) -> dict[str, object]:
    """Publish exact-source GPU/Torch plus exact-SHA GitHub CI evidence."""

    return _publish_new_authority_bundle(
        preregistration,
        output_directory=output_directory,
        field="D10",
        producer=lambda staging: (
            _produce_exact_sha_phase_a_training_and_torch_ci_in_directory(
                preregistration,
                output_directory=staging,
                host_class=host_class,
                heritable_benchmark_path=heritable_benchmark_path,
                zero_all_benchmark_path=zero_all_benchmark_path,
                _precreated_root=True,
            )
        ),
    )


def _produce_exact_sha_phase_a_training_and_torch_ci_in_directory(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    host_class: str,
    heritable_benchmark_path: str | Path,
    zero_all_benchmark_path: str | Path,
    _precreated_root: bool = False,
) -> dict[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    parsed_host_class = _host_class(host_class)
    _require_live_source_twice(preregistration, before=True)
    root, raw_root = _new_bundle_root(
        output_directory,
        field="D10",
        _precreated_root=_precreated_root,
    )

    source_observation = qualification.collect_exact_source_observation(
        _REPOSITORY_ROOT
    )
    github_check_runs = qualification.collect_github_check_runs(
        _REPOSITORY_ROOT,
        source_commit=str(
            _mapping(preregistration["source"], field="source")["commit"]
        ),
    )
    raw_files: dict[str, Path] = {
        "benchmark_heritable": raw_root / "benchmark-heritable.json",
        "benchmark_zero_all": raw_root / "benchmark-zero-all.json",
        "ci_workflow": raw_root / "ci-workflow.yml",
        "github_check_runs": raw_root / "github-check-runs.json",
        "qualification_identity": raw_root / "qualification-identity.json",
        "source_observation": raw_root / "source-observation.json",
        "torch_command": raw_root / "torch-command.json",
        "torch_test_source": raw_root / "test-mind-v1.py",
        "torch_validation": raw_root / "torch-validation.json",
        "torch_validator_source": raw_root / "validate-mind-torch.py",
    }
    _copy_captured_regular_file(
        heritable_benchmark_path,
        raw_files["benchmark_heritable"],
        field="D10 heritable benchmark",
    )
    _copy_captured_regular_file(
        zero_all_benchmark_path,
        raw_files["benchmark_zero_all"],
        field="D10 zero-all benchmark",
    )
    _copy_captured_regular_file(
        _REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml",
        raw_files["ci_workflow"],
        field="D10 CI workflow",
    )
    _copy_captured_regular_file(
        _REPOSITORY_ROOT / "python" / "tests" / "test_mind_v1.py",
        raw_files["torch_test_source"],
        field="D10 Torch test source",
    )
    _copy_captured_regular_file(
        _REPOSITORY_ROOT / "scripts" / "validate_mind_torch.py",
        raw_files["torch_validator_source"],
        field="D10 Torch validator source",
    )
    _write_new_json(raw_files["source_observation"], source_observation)
    _write_new_json(raw_files["github_check_runs"], github_check_runs)
    _write_new_json(
        raw_files["qualification_identity"],
        {
            "host_class": parsed_host_class,
            "github_repository": qualification.GITHUB_REPOSITORY,
            "github_torch_check_name": qualification.GITHUB_TORCH_CHECK_NAME,
        },
    )
    torch_command = qualification.run_torch_cuda_validation(
        _REPOSITORY_ROOT,
        output_path=raw_files["torch_validation"],
    )
    _write_new_json(raw_files["torch_command"], torch_command)
    produced_at_utc = _format_utc(_utc_now())
    report = _seal_reconstructed_authority_report(
        preregistration,
        root=root,
        produced_at_utc=produced_at_utc,
        evidence_kind="exact_sha_phase_a_training_and_torch_ci",
        producer_name="produce_exact_sha_phase_a_training_and_torch_ci_report",
        producer_contract=(
            "exact_source_full_shape_gpu_torch_and_github_check_runs_v1"
        ),
        raw_files=raw_files,
        reconstruct=lambda manifest, captures: (
            _reconstruct_exact_sha_phase_a_training_from_captures(
                manifest,
                captures,
                preregistration=preregistration,
            )
        ),
    )
    _require_live_source_twice(preregistration, before=False)
    return report


def produce_phase_a_training_throughput_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    host_class: str,
    host_observer: Callable[[Path], Mapping[str, object]],
) -> dict[str, object]:
    """Rerun both full benchmark arms under live resource sampling."""

    return _publish_new_authority_bundle(
        preregistration,
        output_directory=output_directory,
        field="Phase-A throughput",
        producer=lambda staging: _produce_phase_a_training_throughput_in_directory(
            preregistration,
            output_directory=staging,
            host_class=host_class,
            host_observer=host_observer,
            _precreated_root=True,
        ),
    )


def _produce_phase_a_training_throughput_in_directory(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    host_class: str,
    host_observer: Callable[[Path], Mapping[str, object]],
    _precreated_root: bool = False,
) -> dict[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    parsed_host_class = _host_class(host_class)
    _require_live_source_twice(preregistration, before=True)
    root, raw_root = _new_bundle_root(
        output_directory,
        field="Phase-A throughput",
        _precreated_root=_precreated_root,
    )
    from evolution_sim.mind.open_ecology_seed_registry import (
        OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        OPEN_ECOLOGY_SEED_REGISTRY,
    )

    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
    expected_genome_seed = int(
        benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX]
    )
    preliminary_reports = _mapping(
        _mapping(preregistration["throughput_gate"], field="throughput gate")[
            "reports"
        ],
        field="throughput reports",
    )
    preliminary_heritable = _mapping(
        preliminary_reports["heritable"],
        field="preliminary heritable benchmark",
    )
    preliminary_protocol = _mapping(
        preliminary_heritable["protocol"],
        field="preliminary heritable protocol",
    )
    if preliminary_protocol["genome_stream_seed"] != expected_genome_seed:
        raise contract.OpenEcologyPhaseAError(
            "Phase-A benchmark genome stream detached from preregistration"
        )
    fresh = qualification.run_fresh_phase_a_benchmarks(
        _REPOSITORY_ROOT,
        source_commit=str(
            _mapping(preregistration["source"], field="source")["commit"]
        ),
        host_class=parsed_host_class,
        campaign_root=root,
        host_observer=host_observer,
    )
    reports = _mapping(fresh["benchmark_reports"], field="fresh benchmark reports")
    telemetry = dict(fresh)
    telemetry.pop("benchmark_reports")
    telemetry["benchmark_report_sha256_by_mode"] = {
        mode: stable_payload_digest(
            _mapping(reports[mode], field=f"fresh benchmark {mode}")
        )
        for mode in ("heritable", "zero_all")
    }
    raw_files = {
        "benchmark_heritable": raw_root / "benchmark-heritable.json",
        "benchmark_zero_all": raw_root / "benchmark-zero-all.json",
        "resource_telemetry": raw_root / "resource-telemetry.json",
    }
    _write_new_json(
        raw_files["benchmark_heritable"],
        _mapping(reports["heritable"], field="fresh heritable benchmark"),
    )
    _write_new_json(
        raw_files["benchmark_zero_all"],
        _mapping(reports["zero_all"], field="fresh zero-all benchmark"),
    )
    _write_new_json(raw_files["resource_telemetry"], telemetry)
    produced_at_utc = _format_utc(_utc_now())
    report = _seal_reconstructed_authority_report(
        preregistration,
        root=root,
        produced_at_utc=produced_at_utc,
        evidence_kind="phase_a_training_throughput",
        producer_name="produce_phase_a_training_throughput_report",
        producer_contract=(
            "fresh_full_shape_pair_with_live_gpu_host_kernel_sampling_v1"
        ),
        raw_files=raw_files,
        reconstruct=lambda manifest, captures: (
            _reconstruct_phase_a_training_throughput_from_captures(
                manifest,
                captures,
                preregistration=preregistration,
            )
        ),
    )
    _require_live_source_twice(preregistration, before=False)
    return report


def produce_campaign_storage_capacity_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    target_filesystem: str | Path,
    archive_tool_authority_path: str | Path,
    expected_archive_tool_authority_sha256: str,
    drive_remote: str = "gdrive:",
) -> dict[str, object]:
    """Publish live descriptor-bound local and rclone Drive capacity evidence."""

    return _publish_new_authority_bundle(
        preregistration,
        output_directory=output_directory,
        field="campaign storage capacity",
        producer=lambda staging: _produce_campaign_storage_capacity_in_directory(
            preregistration,
            output_directory=staging,
            target_filesystem=target_filesystem,
            archive_tool_authority_path=archive_tool_authority_path,
            expected_archive_tool_authority_sha256=(
                expected_archive_tool_authority_sha256
            ),
            drive_remote=drive_remote,
            _precreated_root=True,
        ),
    )


def _produce_campaign_storage_capacity_in_directory(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    target_filesystem: str | Path,
    archive_tool_authority_path: str | Path,
    expected_archive_tool_authority_sha256: str,
    drive_remote: str,
    _precreated_root: bool = False,
) -> dict[str, object]:
    from evolution_sim.io.open_ecology_archive_authority import (
        load_archive_tool_authority,
    )
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    if drive_remote != "gdrive:":
        raise contract.OpenEcologyPhaseAError(
            "campaign storage authority requires the configured gdrive: remote"
        )
    _require_live_source_twice(preregistration, before=True)
    root, raw_root = _new_bundle_root(
        output_directory,
        field="campaign storage capacity",
        _precreated_root=_precreated_root,
    )
    authority = load_archive_tool_authority(
        Path(archive_tool_authority_path),
        expected_sha256=expected_archive_tool_authority_sha256,
    )
    _host_identifier(
        authority.ssh_target,
        field="campaign storage archive authority SSH target",
    )
    source = _mapping(preregistration["source"], field="storage source")
    if (
        expected_archive_tool_authority_sha256
        != source["archive_tool_authority_sha256"]
        or authority.authority_sha256 != source["archive_tool_authority_sha256"]
        or authority.source_git_sha != source["commit"]
        or authority.source_manifest_sha256 != source["manifest_sha256"]
    ):
        raise contract.OpenEcologyPhaseAError(
            "campaign storage archive authority is not exact-source/host bound"
        )
    measurement = qualification.measure_campaign_storage(
        target_filesystem,
        archive_authority=authority,
        drive_remote=drive_remote,
    )
    raw_files = {
        "archive_tool_authority": raw_root / "archive-tool-authority.json",
        "storage_measurement": raw_root / "storage-measurement.json",
    }
    _copy_captured_regular_file(
        archive_tool_authority_path,
        raw_files["archive_tool_authority"],
        field="campaign storage archive authority",
    )
    _write_new_json(raw_files["storage_measurement"], measurement)
    produced_at_utc = _format_utc(_utc_now())
    report = _seal_reconstructed_authority_report(
        preregistration,
        root=root,
        produced_at_utc=produced_at_utc,
        evidence_kind="campaign_storage_capacity",
        producer_name="produce_campaign_storage_capacity_report",
        producer_contract=("sealed_archive_authority_remote_statvfs_local_drive_v3"),
        raw_files=raw_files,
        reconstruct=lambda manifest, captures: (
            _reconstruct_campaign_storage_capacity_from_captures(
                manifest,
                captures,
                preregistration=preregistration,
            )
        ),
    )
    _require_live_source_twice(preregistration, before=False)
    return report


def produce_output_lock_contention_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    """Publish a real multi-process lock contention and identity probe."""

    return _publish_new_authority_bundle(
        preregistration,
        output_directory=output_directory,
        field="output lock contention",
        producer=lambda staging: _produce_output_lock_contention_in_directory(
            preregistration,
            output_directory=staging,
            _precreated_root=True,
        ),
    )


def _produce_output_lock_contention_in_directory(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    _precreated_root: bool = False,
) -> dict[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source_twice(preregistration, before=True)
    root, raw_root = _new_bundle_root(
        output_directory,
        field="output lock contention",
        _precreated_root=_precreated_root,
    )
    campaign_id = f"phase-a-{str(preregistration['exact_digest'])[:24]}"
    probe_path = raw_root / "output.lock"
    probe = qualification.run_output_lock_probe(
        probe_path,
        campaign_id=campaign_id,
        source_git_sha=str(
            _mapping(preregistration["source"], field="source")["commit"]
        ),
    )
    raw_files = {
        "lock_identity": probe_path,
        "lock_probe": raw_root / "lock-probe.json",
    }
    _write_new_json(raw_files["lock_probe"], probe)
    produced_at_utc = _format_utc(_utc_now())
    report = _seal_reconstructed_authority_report(
        preregistration,
        root=root,
        produced_at_utc=produced_at_utc,
        evidence_kind="output_lock_contention",
        producer_name="produce_output_lock_contention_report",
        producer_contract=(
            "real_multiprocess_storage_lock_contention_identity_reacquire_v1"
        ),
        raw_files=raw_files,
        reconstruct=lambda manifest, captures: (
            _reconstruct_output_lock_contention_from_captures(
                manifest,
                captures,
                preregistration=preregistration,
            )
        ),
    )
    _require_live_source_twice(preregistration, before=False)
    return report


def verify_exact_sha_phase_a_training_and_torch_ci_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reconstruct captured D10 authority evidence without running host probes."""

    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="exact_sha_phase_a_training_and_torch_ci",
        authorization_time=authorization_time,
    )
    reconstructed = _reconstruct_exact_sha_phase_a_training_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        reconstructed,
        evidence_kind="exact_sha_phase_a_training_and_torch_ci",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    return reconstructed


def live_verify_exact_sha_phase_a_training_and_torch_ci_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="exact_sha_phase_a_training_and_torch_ci",
        authorization_time=authorization_time,
    )
    captured = _reconstruct_exact_sha_phase_a_training_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        captured,
        evidence_kind="exact_sha_phase_a_training_and_torch_ci",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    captured_facts = _mapping(captured["facts"], field="captured D10 facts")
    intended_hosts = _sequence(
        captured_facts["intended_training_host_classes"],
        field="captured D10 intended hosts",
    )
    if len(intended_hosts) != 1:
        raise _error("D10 live reexecution requires one exact measured host class")
    with tempfile.TemporaryDirectory(prefix="open-ecology-d10-live-") as temporary:
        temporary_root = Path(temporary).resolve()
        captured_heritable = _write_new_bytes(
            temporary_root / "captured-heritable.json",
            captures["benchmark_heritable"].payload,
        )
        captured_zero_all = _write_new_bytes(
            temporary_root / "captured-zero-all.json",
            captures["benchmark_zero_all"].payload,
        )
        fresh_root = temporary_root / "d10"
        _produce_exact_sha_phase_a_training_and_torch_ci_in_directory(
            preregistration,
            output_directory=fresh_root,
            host_class=_host_class(intended_hosts[0]),
            heritable_benchmark_path=captured_heritable,
            zero_all_benchmark_path=captured_zero_all,
        )
        _fresh_report, fresh_manifest, fresh_captures = (
            _capture_authority_report_bundle(
                fresh_root / "report.json",
                preregistration=preregistration,
                expected_kind="exact_sha_phase_a_training_and_torch_ci",
                authorization_time=None,
            )
        )
        fresh = _reconstruct_exact_sha_phase_a_training_from_captures(
            fresh_manifest,
            fresh_captures,
            preregistration=preregistration,
        )
        _validate_live_reexecution_pair(
            captured,
            fresh,
            evidence_kind="exact_sha_phase_a_training_and_torch_ci",
            stable_fact_fields=_D10_LIVE_STABLE_FACT_FIELDS,
            preregistration=preregistration,
            captured_authorization_time=authorization_time,
        )
    return captured


def verify_phase_a_training_throughput_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reconstruct captured throughput authority without rerunning benchmarks."""

    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="phase_a_training_throughput",
        authorization_time=authorization_time,
    )
    reconstructed = _reconstruct_phase_a_training_throughput_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        reconstructed,
        evidence_kind="phase_a_training_throughput",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    return reconstructed


def live_verify_phase_a_training_throughput_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
    *,
    host_observer: Callable[[Path], Mapping[str, object]],
) -> Mapping[str, object]:
    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="phase_a_training_throughput",
        authorization_time=authorization_time,
    )
    captured = _reconstruct_phase_a_training_throughput_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        captured,
        evidence_kind="phase_a_training_throughput",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    captured_facts = _mapping(
        captured["facts"],
        field="captured Phase-A throughput facts",
    )
    with tempfile.TemporaryDirectory(
        prefix="open-ecology-throughput-live-"
    ) as temporary:
        fresh_root = Path(temporary).resolve() / "throughput"
        _produce_phase_a_training_throughput_in_directory(
            preregistration,
            output_directory=fresh_root,
            host_class=_host_class(captured_facts["measured_host_class"]),
            host_observer=host_observer,
        )
        _fresh_report, fresh_manifest, fresh_captures = (
            _capture_authority_report_bundle(
                fresh_root / "report.json",
                preregistration=preregistration,
                expected_kind="phase_a_training_throughput",
                authorization_time=None,
            )
        )
        fresh = _reconstruct_phase_a_training_throughput_from_captures(
            fresh_manifest,
            fresh_captures,
            preregistration=preregistration,
        )
        _validate_live_reexecution_pair(
            captured,
            fresh,
            evidence_kind="phase_a_training_throughput",
            stable_fact_fields=_THROUGHPUT_LIVE_STABLE_FACT_FIELDS,
            preregistration=preregistration,
            captured_authorization_time=authorization_time,
        )
    return captured


def verify_campaign_storage_capacity_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reconstruct captured storage authority without contacting remote systems."""

    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="campaign_storage_capacity",
        authorization_time=authorization_time,
    )
    reconstructed = _reconstruct_campaign_storage_capacity_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        reconstructed,
        evidence_kind="campaign_storage_capacity",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    return reconstructed


def live_verify_campaign_storage_capacity_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="campaign_storage_capacity",
        authorization_time=authorization_time,
    )
    captured = _reconstruct_campaign_storage_capacity_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        captured,
        evidence_kind="campaign_storage_capacity",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    captured_facts = _mapping(
        captured["facts"],
        field="captured campaign storage facts",
    )
    expected_authority_sha256 = _sha256(
        _mapping(
            preregistration["source"],
            field="campaign storage source",
        )["archive_tool_authority_sha256"],
        field="campaign storage archive authority",
    )
    with tempfile.TemporaryDirectory(prefix="open-ecology-storage-live-") as temporary:
        temporary_root = Path(temporary).resolve()
        captured_authority = _write_new_bytes(
            temporary_root / "captured-archive-tool-authority.json",
            captures["archive_tool_authority"].payload,
        )
        fresh_root = temporary_root / "storage"
        _produce_campaign_storage_capacity_in_directory(
            preregistration,
            output_directory=fresh_root,
            target_filesystem=str(captured_facts["target_filesystem_path"]),
            archive_tool_authority_path=captured_authority,
            expected_archive_tool_authority_sha256=expected_authority_sha256,
            drive_remote="gdrive:",
        )
        _fresh_report, fresh_manifest, fresh_captures = (
            _capture_authority_report_bundle(
                fresh_root / "report.json",
                preregistration=preregistration,
                expected_kind="campaign_storage_capacity",
                authorization_time=None,
            )
        )
        fresh = _reconstruct_campaign_storage_capacity_from_captures(
            fresh_manifest,
            fresh_captures,
            preregistration=preregistration,
        )
        _validate_live_reexecution_pair(
            captured,
            fresh,
            evidence_kind="campaign_storage_capacity",
            stable_fact_fields=_STORAGE_LIVE_STABLE_FACT_FIELDS,
            preregistration=preregistration,
            captured_authorization_time=authorization_time,
        )
    return captured


def verify_output_lock_contention_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reconstruct captured lock authority without spawning contenders."""

    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="output_lock_contention",
        authorization_time=authorization_time,
    )
    reconstructed = _reconstruct_output_lock_contention_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        reconstructed,
        evidence_kind="output_lock_contention",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    return reconstructed


def live_verify_output_lock_contention_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    _report, manifest, captures = _capture_authority_report_bundle(
        report_path,
        preregistration=preregistration,
        expected_kind="output_lock_contention",
        authorization_time=authorization_time,
    )
    captured = _reconstruct_output_lock_contention_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )
    _validate_reconstructed_authority_semantics(
        captured,
        evidence_kind="output_lock_contention",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    with tempfile.TemporaryDirectory(prefix="open-ecology-lock-live-") as temporary:
        fresh_root = Path(temporary).resolve() / "output-lock"
        _produce_output_lock_contention_in_directory(
            preregistration,
            output_directory=fresh_root,
        )
        _fresh_report, fresh_manifest, fresh_captures = (
            _capture_authority_report_bundle(
                fresh_root / "report.json",
                preregistration=preregistration,
                expected_kind="output_lock_contention",
                authorization_time=None,
            )
        )
        fresh = _reconstruct_output_lock_contention_from_captures(
            fresh_manifest,
            fresh_captures,
            preregistration=preregistration,
        )
        _validate_live_reexecution_pair(
            captured,
            fresh,
            evidence_kind="output_lock_contention",
            stable_fact_fields=_OUTPUT_LOCK_LIVE_STABLE_FACT_FIELDS,
            preregistration=preregistration,
            captured_authorization_time=authorization_time,
        )
    return captured


def _validate_live_reexecution_pair(
    captured: Mapping[str, object],
    fresh: Mapping[str, object],
    *,
    evidence_kind: str,
    stable_fact_fields: frozenset[str],
    preregistration: Mapping[str, object],
    captured_authorization_time: datetime | None,
) -> None:
    """Require independent live facts before captured evidence gains authority."""

    expected_reconstruction_keys = {
        "evidence_kind",
        "campaign_digest",
        "configuration_sha256",
        "source",
        "facts",
    }
    captured_mapping = _mapping(
        captured,
        field=f"{evidence_kind} captured reconstruction",
    )
    fresh_mapping = _mapping(
        fresh,
        field=f"{evidence_kind} live reconstruction",
    )
    _exact_key_set(
        captured_mapping,
        expected_reconstruction_keys,
        field=f"{evidence_kind} captured reconstruction",
    )
    _exact_key_set(
        fresh_mapping,
        expected_reconstruction_keys,
        field=f"{evidence_kind} live reconstruction",
    )
    expected_identity = {
        "evidence_kind": evidence_kind,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _source_binding(preregistration),
    }
    for reconstruction_name, reconstruction in (
        ("captured", captured_mapping),
        ("live", fresh_mapping),
    ):
        identity = {
            key: reconstruction[key]
            for key in (
                "evidence_kind",
                "campaign_digest",
                "configuration_sha256",
                "source",
            )
        }
        if identity != expected_identity:
            raise _error(
                f"{evidence_kind} {reconstruction_name} reexecution identity drifted"
            )
    captured_facts = _mapping(
        captured_mapping["facts"],
        field=f"{evidence_kind} captured facts",
    )
    fresh_facts = _mapping(
        fresh_mapping["facts"],
        field=f"{evidence_kind} live facts",
    )
    if set(captured_facts) != set(fresh_facts) or not stable_fact_fields.issubset(
        captured_facts
    ):
        raise _error(f"{evidence_kind} live fact contract drifted")
    validator = _SEMANTIC_VALIDATORS.get(evidence_kind)
    if validator is None:
        raise _error(f"{evidence_kind} has no live semantic validator")
    validator(captured_facts, preregistration, captured_authorization_time)
    validator(fresh_facts, preregistration, _utc_now())
    captured_stable = {
        field: captured_facts[field] for field in sorted(stable_fact_fields)
    }
    fresh_stable = {field: fresh_facts[field] for field in sorted(stable_fact_fields)}
    if captured_stable != fresh_stable:
        raise _error(f"{evidence_kind} live reexecution stable contract drifted")


def _validate_reconstructed_authority_semantics(
    reconstructed: Mapping[str, object],
    *,
    evidence_kind: str,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> None:
    facts = _mapping(
        reconstructed.get("facts"),
        field=f"{evidence_kind} reconstructed facts",
    )
    validator = _SEMANTIC_VALIDATORS.get(evidence_kind)
    if validator is None:
        raise _error(f"{evidence_kind} has no semantic validator")
    validator(facts, preregistration, authorization_time)


def _rename_name_no_replace(
    parent_descriptor: int,
    *,
    source_name: str,
    destination_name: str,
    destination_path: Path,
) -> None:
    """Atomically rename two entries relative to one held parent descriptor."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        # RENAME_EXCL prevents replacement; RENAME_NOFOLLOW_ANY prevents
        # any symlink traversal in the source or destination paths.
        result = rename(
            parent_descriptor,
            encoded_source,
            parent_descriptor,
            encoded_destination,
            0x00000004 | 0x00000010,
        )
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        # Linux RENAME_NOREPLACE is the exact create-if-absent contract.
        result = rename(
            parent_descriptor,
            encoded_source,
            parent_descriptor,
            encoded_destination,
            1,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "exclusive directory publication is unsupported",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                destination_path,
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_path,
        )


def _rename_path_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one file or directory without replacing authority."""

    if source.parent != destination.parent:
        raise ValueError("exclusive directory publication requires one parent")
    parent_flags = os.O_RDONLY
    parent_flags |= getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(source.parent, parent_flags)
    try:
        _rename_name_no_replace(
            parent_fd,
            source_name=source.name,
            destination_name=destination.name,
            destination_path=destination,
        )
    finally:
        os.close(parent_fd)


def verify_preregistration_roundtrip_fail_closed_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> Mapping[str, object]:
    """Reconstruct D9 identity and facts from sealed raw bytes, not report facts."""

    root = report_path.parent.absolute()
    root_fd = _open_real_directory(root, field="D9 report bundle")
    try:
        report_capture = _capture_regular_file_at(
            root_fd,
            root,
            report_path.name,
            field="D9 report",
            maximum_bytes=_MAX_AUTHORITY_JSON_BYTES,
        )
    finally:
        os.close(root_fd)
    report = _strict_json_from_captured_bytes(
        report_capture,
        field="D9 report",
    )
    _manifest_path, manifest, captures = _load_report_raw_evidence_manifest(
        report_capture.path,
        report,
        expected_kind="preregistration_roundtrip_fail_closed",
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    return _reconstruct_preregistration_roundtrip_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )


PROOF_PRODUCERS: Mapping[str, ProofProducer] = MappingProxyType(
    {
        "cross_surface_and_self_echo": (produce_cross_surface_and_self_echo_report),
        "runtime_genome_and_action_source": (
            produce_runtime_genome_and_action_source_report
        ),
        "critic_gradient_and_density_schedule": (
            produce_critic_gradient_and_density_schedule_report
        ),
        "fixed_batch_equivalence_and_speed": (
            produce_fixed_batch_equivalence_and_speed_report
        ),
        "capture_noninterference_reexecution": (
            produce_capture_noninterference_reexecution_report
        ),
        "campaign_storage_capacity": (produce_campaign_storage_capacity_report),
        "exact_sha_phase_a_training_and_torch_ci": (
            produce_exact_sha_phase_a_training_and_torch_ci_report
        ),
        "output_lock_contention": produce_output_lock_contention_report,
        "phase_a_training_throughput": produce_phase_a_training_throughput_report,
        "preregistration_roundtrip_fail_closed": (
            produce_preregistration_roundtrip_fail_closed_report
        ),
    }
)
REPORT_AUTHORITY_VERIFIERS: Mapping[str, ReportAuthorityVerifier] = MappingProxyType(
    {
        "cross_surface_and_self_echo": verify_cross_surface_and_self_echo_report,
        "runtime_genome_and_action_source": (
            verify_runtime_genome_and_action_source_report
        ),
        "critic_gradient_and_density_schedule": (
            verify_critic_gradient_and_density_schedule_report
        ),
        "fixed_batch_equivalence_and_speed": (
            verify_fixed_batch_equivalence_and_speed_report
        ),
        "campaign_storage_capacity": verify_campaign_storage_capacity_report,
        "exact_sha_phase_a_training_and_torch_ci": (
            verify_exact_sha_phase_a_training_and_torch_ci_report
        ),
        "output_lock_contention": verify_output_lock_contention_report,
        "phase_a_training_throughput": verify_phase_a_training_throughput_report,
        "preregistration_roundtrip_fail_closed": (
            verify_preregistration_roundtrip_fail_closed_report
        ),
    }
)
if not set(PROOF_PRODUCERS).issubset(_ALL_EVIDENCE_KINDS):
    raise RuntimeError("Phase A proof-producer registry contains unknown kinds")
if not set(REPORT_AUTHORITY_VERIFIERS).issubset(_ALL_EVIDENCE_KINDS):
    raise RuntimeError("Phase A authority-verifier registry contains unknown kinds")
PROOF_PRODUCER_AVAILABILITY: Mapping[str, bool] = MappingProxyType(
    {kind: kind in PROOF_PRODUCERS for kind in _ALL_EVIDENCE_KINDS}
)


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
        blockers = ["launch_evidence_index_required"]
        if not all(
            verifier_availability[kind] for kind in _PHASE_A_REQUIRED_EVIDENCE_KINDS
        ):
            blockers.append("independent_report_authority_verifiers_required")
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
            PROOF_PRODUCER_AVAILABILITY[kind]
            for kind in _PHASE_A_REQUIRED_EVIDENCE_KINDS
        ),
        "dependency_specific_authority_verifiers_available": all(
            verifier_availability[kind] for kind in _PHASE_A_REQUIRED_EVIDENCE_KINDS
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
            "phase_a_selection": False,
            "phase_b": False,
            "phase_c": False,
            "phase_d": False,
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
                "measurement_location",
                "target_ssh",
                "remote_hostname",
                "remote_boot_id",
                "archive_authority_sha256",
                "ssh_effective_config_sha256",
                "ssh_connection",
                "rclone_config_path",
                "rclone_config_sha256",
                "google_drive_free_bytes",
                "projected_active_storage_bytes",
                "target_filesystem_path",
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
    require_current_storage_freshness: bool = True,
) -> None:
    """Reopen and semantically validate every report referenced by authority."""

    contract = _contract()
    if not isinstance(require_current_storage_freshness, bool):
        raise contract.OpenEcologyPhaseAError(
            "storage-freshness policy must be boolean"
        )
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
            "Phase A launch authorization requires its six stage-specific "
            "readiness proofs"
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
                    "measurement_location",
                    "target_ssh",
                    "remote_hostname",
                    "remote_boot_id",
                    "archive_authority_sha256",
                    "ssh_effective_config_sha256",
                    "ssh_connection",
                    "rclone_config_path",
                    "rclone_config_sha256",
                    "google_drive_free_bytes",
                    "projected_active_storage_bytes",
                    "target_filesystem_path",
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
                validation_time if require_current_storage_freshness else authorized_at,
            )
            for key in (
                "checked_at_utc",
                "measurement_location",
                "target_ssh",
                "remote_hostname",
                "remote_boot_id",
                "archive_authority_sha256",
                "ssh_effective_config_sha256",
                "ssh_connection",
                "rclone_config_path",
                "rclone_config_sha256",
                "google_drive_free_bytes",
                "projected_active_storage_bytes",
                "target_filesystem_path",
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
            "stage_gated_phase_a_training_dependencies_and_operations_v2"
        ),
        "authorization_scope": "phase_a_training_only",
        "phase_a_selection_authorized": False,
        "phase_b_authorized": False,
        "phase_c_authorized": False,
        "phase_d_authorized": False,
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
    schema_version = report.get("schema_version")
    legacy_capture_report = (
        expected_kind == "capture_noninterference_reexecution"
        and schema_version
        == OPEN_ECOLOGY_PHASE_A_LEGACY_READINESS_REPORT_SCHEMA_VERSION
    )
    if schema_version == OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION:
        expected_report_keys = {
            "schema_version",
            "evidence_kind",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "produced_at_utc",
            "raw_evidence_manifest",
            "facts",
            "exact_digest",
        }
    elif legacy_capture_report:
        expected_report_keys = {
            "schema_version",
            "evidence_kind",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "produced_at_utc",
            "facts",
            "exact_digest",
        }
    else:
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} report schema is not an accepted authority envelope"
        )
    _exact_key_set(
        report,
        expected_report_keys,
        field=f"{expected_kind} report",
    )
    contract._validate_signed_payload(report, field=f"{expected_kind} report")
    if (
        report.get("evidence_kind") != expected_kind
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
    if schema_version == OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION:
        if report_path is None:
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} v2 authority report requires its exact file path"
            )
        _load_report_raw_evidence_manifest(
            report_path,
            report,
            expected_kind=expected_kind,
            preregistration=preregistration,
            authorization_time=authorization_time,
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
            "profiles_per_token_by_surface",
            "action_count_by_surface",
            "self_echo_case_count",
            "emitter_own_signal_self_projection_nonzero_count",
            "emitter_own_signal_local_patch_projection_nonzero_count",
            "adjacent_receiver_self_projection_positive_count",
            "adjacent_receiver_local_patch_projection_positive_count",
            "signal_emission_success_count",
            "global_signal_field_positive_count",
            "global_signal_emission_event_count",
            "signal_action_identity_mismatch_count",
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
    profile_counts = _surface_mapping(
        facts["profiles_per_token_by_surface"],
        field="profiles_per_token_by_surface",
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
        _integer(value, field="profiles per token", minimum=0) != 2
        for value in profile_counts.values()
    ):
        raise _error("cross-surface profiles-per-token count differs from two")
    if any(
        _integer(value, field="action count", minimum=0) != 20
        for value in action_counts.values()
    ):
        raise _error("cross-surface action count differs from twenty")
    if (
        _integer(facts["self_echo_case_count"], field="self_echo_case_count", minimum=0)
        != 8
        or _integer(
            facts["emitter_own_signal_self_projection_nonzero_count"],
            field="emitter own-signal self projection",
            minimum=0,
        )
        != 0
        or _integer(
            facts["emitter_own_signal_local_patch_projection_nonzero_count"],
            field="emitter own-signal patch projection",
            minimum=0,
        )
        != 0
        or _integer(
            facts["adjacent_receiver_self_projection_positive_count"],
            field="adjacent receiver self projection",
            minimum=0,
        )
        != 8
        or _integer(
            facts["adjacent_receiver_local_patch_projection_positive_count"],
            field="adjacent receiver patch projection",
            minimum=0,
        )
        != 8
        or _integer(
            facts["signal_emission_success_count"],
            field="signal emission successes",
            minimum=0,
        )
        != 8
        or _integer(
            facts["global_signal_field_positive_count"],
            field="positive global signal fields",
            minimum=0,
        )
        != 8
        or _integer(
            facts["global_signal_emission_event_count"],
            field="global signal emission events",
            minimum=0,
        )
        != 8
        or _integer(
            facts["signal_action_identity_mismatch_count"],
            field="signal action identity mismatches",
            minimum=0,
        )
        != 0
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
        "parent_pre_birth_nonzero_state_count",
        "exact_child_derivation_mismatch_count",
        "inheritance_bound_violation_count",
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
        or _integer(
            facts["parent_pre_birth_nonzero_state_count"],
            field="parents with live pre-birth recurrent state",
            minimum=0,
        )
        != 2
        or _integer(
            facts["exact_child_derivation_mismatch_count"],
            field="exact child derivation mismatches",
            minimum=0,
        )
        != 0
        or _integer(
            facts["inheritance_bound_violation_count"],
            field="inheritance bound violations",
            minimum=0,
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
            "observed_cell_contracts",
            "phase_a_density_schedule",
            "phase_b_density_cycle",
            "shared_value_trunk_gradient_l2",
            "stop_gradient_value_trunk_gradient_l2",
            "stop_gradient_value_head_gradient_l2",
            "actor_loss_shared_trunk_gradient_l2",
            "critic_conditioned_genome_sensitivity_min_abs_delta",
            "critic_unconditioned_genome_sensitivity_max_abs_delta",
            "critic_conditioned_parameter_gradient_l2",
        },
        field="critic gradient facts",
    )
    expected_cells = _mapping(preregistration["architecture"], field="architecture")[
        "critic_gradient_cells"
    ]
    if (
        facts["cell_contracts"] != expected_cells
        or facts["observed_cell_contracts"] != expected_cells
    ):
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
        or _positive_number(
            facts["critic_conditioned_genome_sensitivity_min_abs_delta"],
            field="critic-conditioned genome sensitivity",
        )
        <= 0
        or _number(
            facts["critic_unconditioned_genome_sensitivity_max_abs_delta"],
            field="critic-unconditioned genome sensitivity",
        )
        != 0
        or _positive_number(
            facts["critic_conditioned_parameter_gradient_l2"],
            field="critic-conditioned parameter gradient",
        )
        <= 0
    ):
        raise _error("critic gradient boundary or density schedule was not proved")


def _validate_fixed_batch(
    facts: Mapping[str, object],
    preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "declared_shape",
            "batch_capacity",
            "repeat_count",
            "device",
            "deterministic_cuda_contract",
            "proof_seed_contract",
            "collector_device",
            "rollout_workers",
            "torch_version",
            "timing_order",
            "scalar_timing_sample_count",
            "batched_timing_sample_count",
            "reference_semantic_sha256",
            "core_scalar_semantic_sha256",
            "core_batched_semantic_sha256",
            "scalar_semantic_sha256",
            "scalar_repeat_semantic_sha256",
            "batched_semantic_sha256",
            "batched_repeat_semantic_sha256",
            "ordered_merge_semantic_sha256",
            "collector_transition_count",
            "collector_batched_transition_count",
            "collector_semantic_mismatch_count",
            "scalar_fixed_batch_step_count",
            "batched_fixed_batch_step_count",
            "batched_fixed_batch_runtime_sha256",
            "numeric_comparison_count",
            "reference_numeric_comparison_count",
            "action_mismatch_count",
            "scalar_reference_action_mismatch_count",
            "batched_reference_action_mismatch_count",
            "max_abs_logit_error",
            "max_abs_value_error",
            "max_abs_hidden_error",
            "max_logit_tolerance_ratio",
            "max_value_tolerance_ratio",
            "max_hidden_tolerance_ratio",
            "max_scalar_reference_logit_tolerance_ratio",
            "max_scalar_reference_value_tolerance_ratio",
            "max_scalar_reference_hidden_tolerance_ratio",
            "max_batched_reference_logit_tolerance_ratio",
            "max_batched_reference_value_tolerance_ratio",
            "max_batched_reference_hidden_tolerance_ratio",
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
    deterministic_cuda_contract = _mapping(
        facts["deterministic_cuda_contract"],
        field="D4 deterministic CUDA contract",
    )
    expected_deterministic_cuda_contract = {
        "device": "cuda",
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "default_dtype": "torch.float32",
    }
    if dict(deterministic_cuda_contract) != expected_deterministic_cuda_contract:
        raise _error("D4 deterministic CUDA contract drifted")
    proof_seed_contract = _mapping(
        facts["proof_seed_contract"],
        field="D4 engineering proof seed contract",
    )
    _exact_key_set(
        proof_seed_contract,
        {
            "seed_registry_version",
            "seed_registry_sha256",
            "engineering_seed_role",
            "environment_seed_indices",
            "environment_seeds",
            "model_initialization_seed_index",
            "model_initialization_seed",
            "genome_stream_seed_index",
            "genome_stream_seed",
            "core_genome_seed_indices",
            "core_genome_seeds",
            "policy_sampling_identity_contract",
            "task_ids",
            "policy_sampling_identities",
            "policy_sampling_seeds",
            "scientific_training_seed_access_count",
            "scientific_selection_seed_access_count",
        },
        field="D4 engineering proof seed contract",
    )
    from evolution_sim.mind.open_ecology_seed_registry import (
        OPEN_ECOLOGY_CANONICAL_SHA256,
        OPEN_ECOLOGY_SEED_REGISTRY,
        OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
    )
    from evolution_sim.mind.recurrent_rollout import (
        derive_recurrent_policy_sampling_seed,
    )

    proof_seeds = tuple(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"])
    environment_indices = list(range(16))
    policy_identities = [
        (
            "mind_v3_open_ecology_engineering_proof_task_v1|"
            f"registry={OPEN_ECOLOGY_CANONICAL_SHA256}|"
            "role=open_ecology_proof|"
            "model_seed_index=00|genome_stream_seed_index=01|"
            f"environment_seed_index={index:02d}|world={index:02d}|"
            "initial_agents=64"
        )
        for index in environment_indices
    ]
    expected_proof_seed_contract = {
        "seed_registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
        "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        "engineering_seed_role": "open_ecology_proof",
        "environment_seed_indices": environment_indices,
        "environment_seeds": list(proof_seeds),
        "model_initialization_seed_index": 0,
        "model_initialization_seed": proof_seeds[0],
        "genome_stream_seed_index": 1,
        "genome_stream_seed": proof_seeds[1],
        "core_genome_seed_indices": [index % 16 for index in range(64)],
        "core_genome_seeds": [proof_seeds[index % 16] for index in range(64)],
        "policy_sampling_identity_contract": (
            "mind_v3_open_ecology_engineering_proof_task_v1"
        ),
        "task_ids": [
            f"open-ecology-proof-world-{index:02d}-seed-{proof_seeds[index]}-density-64"
            for index in environment_indices
        ],
        "policy_sampling_identities": policy_identities,
        "policy_sampling_seeds": [
            derive_recurrent_policy_sampling_seed(task_identity=identity)
            for identity in policy_identities
        ],
        "scientific_training_seed_access_count": 0,
        "scientific_selection_seed_access_count": 0,
    }
    if dict(proof_seed_contract) != expected_proof_seed_contract:
        raise _error("D4 engineering proof seed contract drifted")
    scalar = _sha256(facts["scalar_semantic_sha256"], field="scalar semantics")
    batched = _sha256(facts["batched_semantic_sha256"], field="batch semantics")
    reference = _sha256(
        facts["reference_semantic_sha256"],
        field="direct-model reference semantics",
    )
    core_scalar = _sha256(
        facts["core_scalar_semantic_sha256"],
        field="scalar core semantics",
    )
    core_batched = _sha256(
        facts["core_batched_semantic_sha256"],
        field="batched core semantics",
    )
    scalar_repeats = _sequence(
        facts["scalar_repeat_semantic_sha256"],
        field="scalar repeat digests",
    )
    batched_repeats = _sequence(
        facts["batched_repeat_semantic_sha256"],
        field="batch repeat digests",
    )
    tolerance_ratios = [
        _number(facts[field], field=field)
        for field in (
            "max_logit_tolerance_ratio",
            "max_value_tolerance_ratio",
            "max_hidden_tolerance_ratio",
            "max_scalar_reference_logit_tolerance_ratio",
            "max_scalar_reference_value_tolerance_ratio",
            "max_scalar_reference_hidden_tolerance_ratio",
            "max_batched_reference_logit_tolerance_ratio",
            "max_batched_reference_value_tolerance_ratio",
            "max_batched_reference_hidden_tolerance_ratio",
        )
    ]
    absolute_errors = [
        _number(facts[field], field=field)
        for field in (
            "max_abs_logit_error",
            "max_abs_value_error",
            "max_abs_hidden_error",
        )
    ]
    runtime_digests = _sequence(
        facts["batched_fixed_batch_runtime_sha256"],
        field="fixed batch runtime digests",
    )
    runtime_contract = _mapping(
        preregistration["runtime_contract"],
        field="runtime contract",
    )
    collector_transition_count = _positive_integer(
        facts["collector_transition_count"],
        field="collector transition count",
    )
    if (
        facts["device"] != "cuda"
        or facts["collector_device"] != "cpu"
        or _positive_integer(
            facts["rollout_workers"],
            field="collector rollout workers",
        )
        != _positive_integer(
            runtime_contract["rollout_workers"],
            field="runtime rollout workers",
        )
        or not isinstance(facts["torch_version"], str)
        or not facts["torch_version"]
        or facts["timing_order"] != [["scalar", "batched"], ["batched", "scalar"]]
        or _positive_integer(
            facts["scalar_timing_sample_count"],
            field="scalar timing sample count",
        )
        != 2
        or _positive_integer(
            facts["batched_timing_sample_count"],
            field="batched timing sample count",
        )
        != 2
        or _positive_integer(facts["batch_capacity"], field="batch capacity") != 320
        or _integer(facts["repeat_count"], field="repeat_count", minimum=2) != 2
        or len(scalar_repeats) != 2
        or len(batched_repeats) != 2
        or any(
            _sha256(value, field="scalar repeat digest") != scalar
            for value in scalar_repeats
        )
        or any(
            _sha256(value, field="batch repeat digest") != scalar
            for value in batched_repeats
        )
        or batched != scalar
        or _sha256(facts["ordered_merge_semantic_sha256"], field="ordered merge")
        != scalar
        or core_batched != core_scalar
        or reference != core_scalar
        or _positive_integer(
            facts["collector_batched_transition_count"],
            field="batched collector transition count",
        )
        != collector_transition_count
        or _integer(
            facts["collector_semantic_mismatch_count"],
            field="collector semantic mismatch count",
            minimum=0,
        )
        != 0
        or _integer(
            facts["scalar_fixed_batch_step_count"],
            field="scalar fixed batch step count",
            minimum=0,
        )
        != 0
        or _positive_integer(
            facts["batched_fixed_batch_step_count"],
            field="batched fixed batch step count",
        )
        != collector_transition_count
        or len(runtime_digests) != 1
        or any(
            _sha256(value, field="fixed batch runtime digest") != value
            for value in runtime_digests
        )
        or _positive_integer(
            facts["numeric_comparison_count"],
            field="numeric comparison count",
        )
        != 16 * 128 * 64
        or _positive_integer(
            facts["reference_numeric_comparison_count"],
            field="reference numeric comparison count",
        )
        != 16 * 128 * 64
        or _integer(
            facts["action_mismatch_count"],
            field="action mismatch count",
            minimum=0,
        )
        != 0
        or _integer(
            facts["scalar_reference_action_mismatch_count"],
            field="scalar/reference action mismatch count",
            minimum=0,
        )
        != 0
        or _integer(
            facts["batched_reference_action_mismatch_count"],
            field="batched/reference action mismatch count",
            minimum=0,
        )
        != 0
        or any(value < 0.0 for value in absolute_errors)
        or any(value < 0.0 or value > 1.0 for value in tolerance_ratios)
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


def _validate_exact_sha_phase_a_training(
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
            "intended_training_host_classes",
            "tested_training_host_classes",
            "full_shape_training_benchmark_count_by_host_class",
            "runtime_contract_sha256_by_host_class",
            "semantic_evidence_sha256_by_host_class",
            "cross_host_semantic_mismatch_count",
            "torch_ci_lane_required",
            "torch_ci_test_count",
            "torch_ci_failure_count",
            "torch_ci_skip_count",
        },
        field="exact SHA Phase-A training facts",
    )
    intended = _sequence(
        facts["intended_training_host_classes"],
        field="intended Phase-A training host classes",
    )
    tested = _sequence(
        facts["tested_training_host_classes"],
        field="tested Phase-A training host classes",
    )
    if (
        not intended
        or any(not isinstance(value, str) or not value for value in intended)
        or len(set(intended)) != len(intended)
        or list(tested) != list(intended)
    ):
        raise _error("Phase-A intended training host classes were not all tested")
    benchmarks = _mapping(
        facts["full_shape_training_benchmark_count_by_host_class"],
        field="Phase-A training benchmark counts",
    )
    runtime = _mapping(
        facts["runtime_contract_sha256_by_host_class"],
        field="Phase-A runtime digests",
    )
    semantics = _mapping(
        facts["semantic_evidence_sha256_by_host_class"],
        field="Phase-A semantic digests",
    )
    host_classes = set(intended)
    if (
        _positive_integer(facts["fresh_checkout_count"], field="fresh checkouts") < 1
        or _integer(facts["dirty_checkout_count"], field="dirty checkouts", minimum=0)
        != 0
        or _positive_integer(
            facts["source_manifest_match_count"],
            field="source-manifest matches",
        )
        < 1
        or _positive_integer(
            facts["import_root_match_count"],
            field="import-root matches",
        )
        < 1
        or set(benchmarks) != host_classes
        or any(
            _positive_integer(value, field="full-shaped benchmark count") < 1
            for value in benchmarks.values()
        )
        or set(runtime) != host_classes
        or set(semantics) != host_classes
        or len({_sha256(value, field="runtime digest") for value in runtime.values()})
        != 1
        or len(
            {_sha256(value, field="semantic digest") for value in semantics.values()}
        )
        != 1
        or _integer(
            facts["cross_host_semantic_mismatch_count"],
            field="cross-host semantic mismatches",
            minimum=0,
        )
        != 0
        or facts["torch_ci_lane_required"] is not True
        or _positive_integer(facts["torch_ci_test_count"], field="Torch test count") < 1
        or _integer(facts["torch_ci_failure_count"], field="Torch failures", minimum=0)
        != 0
        or _integer(facts["torch_ci_skip_count"], field="Torch skips", minimum=0) != 0
    ):
        raise _error(
            "exact-SHA Phase-A training host path or non-optional Torch CI failed"
        )


def _validate_phase_d_host_contract(
    facts: Mapping[str, object],
    _preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "host_contract_mode",
            "intended_host_class_count",
            "tested_host_class_count",
            "runtime_contract_sha256_by_host_class",
            "semantic_evidence_sha256_by_host_class",
            "cross_host_semantic_mismatch_count",
        },
        field="Phase-D host-contract facts",
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
        tested != intended
        or len(runtime) != intended
        or set(runtime) != set(semantics)
        or (
            intended == 1
            and facts["host_contract_mode"] != "single_homogeneous_host_class"
        )
        or (
            intended > 1
            and facts["host_contract_mode"] != "cross_host_exact_equivalence"
        )
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
    ):
        raise _error("Phase-D homogeneous or cross-host exact contract failed")


def _validate_phase_a_training_throughput(
    facts: Mapping[str, object],
    preregistration: Mapping[str, object],
    _authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "preliminary_throughput_gate_digest",
            "measured_host_class",
            "fresh_full_shape_repetition_count",
            "phase_a_training_projection_authoritative",
            "selection_projection_included",
            "semantic_mismatch_count",
            "projected_phase_a_training_seconds",
            "maximum_gpu_memory_share",
            "maximum_host_ram_share",
            "swap_delta_bytes",
            "temperature_below_throttle",
            "xid_count",
            "oom_count",
            "nonfinite_count",
        },
        field="throughput facts",
    )
    preliminary = _mapping(
        preregistration["throughput_gate"],
        field="throughput gate",
    )
    projection = _mapping(
        preliminary["resource_projection"],
        field="throughput resource projection",
    )
    envelope = _mapping(
        projection["resource_envelope"],
        field="throughput resource envelope",
    )
    if (
        facts["preliminary_throughput_gate_digest"] != preliminary["exact_digest"]
        or not isinstance(facts["measured_host_class"], str)
        or not facts["measured_host_class"]
        or _integer(
            facts["fresh_full_shape_repetition_count"],
            field="fresh full-shape repetitions",
            minimum=0,
        )
        < 3
        or facts["phase_a_training_projection_authoritative"] is not True
        or facts["selection_projection_included"] is not False
        or _integer(
            facts["semantic_mismatch_count"], field="semantic mismatches", minimum=0
        )
        != 0
        or _positive_number(
            facts["projected_phase_a_training_seconds"],
            field="Phase-A training projection",
        )
        > _positive_number(
            envelope["maximum_wall_seconds"],
            field="Phase-A resource-envelope wall seconds",
        )
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
            )
        )
    ):
        raise _error("Phase-A training throughput/resource gate failed")


def _validate_storage(
    facts: Mapping[str, object],
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> None:
    _exact_key_set(
        facts,
        {
            "checked_at_utc",
            "measurement_location",
            "target_ssh",
            "remote_hostname",
            "remote_boot_id",
            "archive_authority_sha256",
            "ssh_effective_config_sha256",
            "ssh_verbose_stderr_sha256",
            "ssh_connection",
            "rclone_config_path",
            "rclone_config_sha256",
            "google_drive_free_bytes",
            "projected_active_storage_bytes",
            "target_filesystem_path",
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
    target_path = facts["target_filesystem_path"]
    rclone_config_path = facts["rclone_config_path"]
    ssh_connection = _mapping(
        facts["ssh_connection"],
        field="storage SSH connection",
    )
    _exact_key_set(
        ssh_connection,
        {
            "address",
            "authenticated_host",
            "authentication",
            "host_key",
            "port",
        },
        field="storage SSH connection",
    )
    required = max(100 * 1024**3, (capacity + 4) // 5)
    expected_archive_authority_sha256 = _mapping(
        preregistration["source"],
        field="storage preregistration source",
    )["archive_tool_authority_sha256"]
    _host_identifier(facts["target_ssh"], field="storage SSH target")
    _host_identifier(facts["remote_hostname"], field="storage remote hostname")
    if (
        facts["measurement_location"] != "remote_ssh_target_with_local_drive_v1"
        or facts["archive_authority_sha256"] != expected_archive_authority_sha256
        or ssh_connection.get("authentication") != "publickey"
        or not isinstance(ssh_connection.get("host_key"), str)
        or not str(ssh_connection["host_key"]).startswith(("ssh-", "ecdsa-"))
        or not isinstance(ssh_connection.get("address"), str)
        or not str(ssh_connection["address"])
        or not isinstance(ssh_connection.get("authenticated_host"), str)
        or not str(ssh_connection["authenticated_host"])
        or not 1
        <= _integer(
            ssh_connection.get("port"),
            field="storage SSH port",
            minimum=1,
        )
        <= 65535
        or not isinstance(facts["remote_boot_id"], str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}",
            facts["remote_boot_id"],
        )
        is None
        or not isinstance(target_path, str)
        or not target_path
        or not Path(target_path).is_absolute()
        or str(Path(target_path)) != target_path
        or not isinstance(rclone_config_path, str)
        or not Path(rclone_config_path).is_absolute()
        or str(Path(rclone_config_path)) != rclone_config_path
        or _positive_integer(facts["google_drive_free_bytes"], field="Drive free")
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
        facts["archive_authority_sha256"],
        field="archive authority",
    )
    _sha256(
        facts["ssh_effective_config_sha256"],
        field="SSH effective config",
    )
    _sha256(
        facts["ssh_verbose_stderr_sha256"],
        field="SSH verbose log",
    )
    _sha256(
        facts["rclone_config_sha256"],
        field="rclone config",
    )
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
            "source_retained_after_verification_count",
            "local_pruning_supported",
            "local_prune_attempt_count",
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
        or _positive_integer(
            facts["source_retained_after_verification_count"],
            field="retained verified source bundles",
        )
        < 1
        or facts["local_pruning_supported"] is not False
        or _integer(
            facts["local_prune_attempt_count"],
            field="local prune attempts",
            minimum=0,
        )
        != 0
    ):
        raise _error("immutable Drive uploader proof failed")


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
    "exact_sha_phase_a_training_and_torch_ci": (_validate_exact_sha_phase_a_training),
    "phase_d_host_equivalence_or_homogeneous_contract": (
        _validate_phase_d_host_contract
    ),
    "phase_a_training_throughput": _validate_phase_a_training_throughput,
    "campaign_storage_capacity": _validate_storage,
    "output_lock_contention": _validate_output_lock,
    "immutable_drive_uploader": _validate_uploader,
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


def _load_report_raw_evidence_manifest(
    report_path: Path,
    report: Mapping[str, object],
    *,
    expected_kind: str,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
) -> tuple[
    Path,
    Mapping[str, object],
    dict[str, _CapturedRegularFile],
]:
    contract = _contract()
    if report_path.parent.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} report bundle must not be a symbolic link"
        )
    root = report_path.parent.absolute()
    reference = _mapping(
        report.get("raw_evidence_manifest"),
        field=f"{expected_kind}.raw_evidence_manifest",
    )
    _exact_key_set(
        reference,
        {"relative_path", "sha256", "byte_length"},
        field=f"{expected_kind}.raw_evidence_manifest",
    )
    if reference.get("relative_path") != "raw-evidence-manifest.json":
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} raw-evidence manifest must be bundle-local"
        )
    manifest_path = root / "raw-evidence-manifest.json"
    manifest, captures, manifest_capture = _validate_raw_evidence_manifest_path(
        manifest_path,
        preregistration=preregistration,
        expected_kind=expected_kind,
        authorization_time=authorization_time,
        expected_produced_at_utc=report.get("produced_at_utc"),
    )
    if dict(reference) != _captured_file_reference(manifest_capture):
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} raw-evidence manifest reference or bytes drifted"
        )
    return manifest_path, manifest, captures


def _validate_raw_evidence_manifest_path(
    manifest_path: Path,
    *,
    preregistration: Mapping[str, object],
    expected_kind: str,
    authorization_time: datetime | None,
    expected_produced_at_utc: object,
) -> tuple[
    Mapping[str, object],
    dict[str, _CapturedRegularFile],
    _CapturedRegularFile,
]:
    contract = _contract()
    if manifest_path.parent.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{expected_kind} raw-evidence bundle must not be a symbolic link"
        )
    root = manifest_path.parent.absolute()
    root_fd = _open_real_directory(
        root,
        field=f"{expected_kind} raw-evidence bundle",
    )
    try:
        manifest_capture = _capture_regular_file_at(
            root_fd,
            root,
            manifest_path.name,
            field=f"{expected_kind} raw-evidence manifest",
            maximum_bytes=_MAX_AUTHORITY_JSON_BYTES,
        )
        manifest = _strict_json_from_captured_bytes(
            manifest_capture,
            field=f"{expected_kind} raw-evidence manifest",
        )
        _exact_key_set(
            manifest,
            {
                "schema_version",
                "evidence_kind",
                "campaign_digest",
                "configuration_sha256",
                "source",
                "produced_at_utc",
                "producer",
                "canonical_preimage_contract",
                "files",
                "exact_digest",
            },
            field=f"{expected_kind} raw-evidence manifest",
        )
        contract._validate_signed_payload(
            manifest,
            field=f"{expected_kind} raw-evidence manifest",
        )
        if (
            manifest.get("schema_version")
            != OPEN_ECOLOGY_RAW_EVIDENCE_MANIFEST_SCHEMA_VERSION
            or manifest.get("evidence_kind") != expected_kind
            or manifest.get("campaign_digest") != preregistration.get("exact_digest")
            or manifest.get("configuration_sha256")
            != preregistration.get("configuration_sha256")
            or manifest.get("source") != preregistration.get("source")
            or manifest.get("canonical_preimage_contract")
            != OPEN_ECOLOGY_RAW_EVIDENCE_BYTE_CONTRACT
        ):
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw-evidence manifest identity drifted"
            )
        produced_at = _parse_utc(
            manifest.get("produced_at_utc"),
            field=f"{expected_kind}.raw_evidence.produced_at_utc",
        )
        expected_produced_at = _parse_utc(
            expected_produced_at_utc,
            field=f"{expected_kind}.report.produced_at_utc",
        )
        if produced_at != expected_produced_at:
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw evidence and report times differ"
            )
        if authorization_time is not None and produced_at > authorization_time:
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw evidence postdates authorization"
            )
        producer = _mapping(
            manifest.get("producer"),
            field=f"{expected_kind}.raw_evidence.producer",
        )
        _exact_key_set(
            producer,
            {"name", "contract"},
            field=f"{expected_kind}.raw_evidence.producer",
        )
        if any(
            not isinstance(producer.get(field), str) or not producer.get(field)
            for field in ("name", "contract")
        ):
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw-evidence producer identity is malformed"
            )
        file_entries = _sequence(
            manifest.get("files"),
            field=f"{expected_kind}.raw_evidence.files",
        )
        if not file_entries:
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw-evidence manifest has no files"
            )
        roles: list[str] = []
        captures: dict[str, _CapturedRegularFile] = {}
        declared_paths: set[str] = set()
        for index, raw_entry in enumerate(file_entries):
            entry = _mapping(
                raw_entry,
                field=f"{expected_kind}.raw_evidence.files[{index}]",
            )
            _exact_key_set(
                entry,
                {"role", "relative_path", "sha256", "byte_length"},
                field=f"{expected_kind}.raw_evidence.files[{index}]",
            )
            role = entry.get("role")
            relative = entry.get("relative_path")
            if not isinstance(role, str) or not role or role in captures:
                raise contract.OpenEcologyPhaseAError(
                    f"{expected_kind} raw-evidence roles are malformed or duplicated"
                )
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or not Path(relative).parts
                or Path(relative).parts[0] != "raw"
            ):
                raise contract.OpenEcologyPhaseAError(
                    f"{expected_kind} raw-evidence file path is unsafe"
                )
            if relative in declared_paths:
                raise contract.OpenEcologyPhaseAError(
                    f"{expected_kind} raw-evidence file paths are duplicated"
                )
            capture = _capture_regular_file_at(
                root_fd,
                root,
                relative,
                field=f"{expected_kind}.raw_evidence.files[{index}]",
                maximum_bytes=_MAX_RAW_EVIDENCE_FILE_BYTES,
            )
            if {
                key: entry[key] for key in ("relative_path", "sha256", "byte_length")
            } != _captured_file_reference(capture):
                raise contract.OpenEcologyPhaseAError(
                    f"{expected_kind} raw-evidence file reference or bytes drifted"
                )
            roles.append(role)
            captures[role] = capture
            declared_paths.add(relative)
        if roles != sorted(roles):
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw-evidence roles are not canonically ordered"
            )
        if declared_paths != _raw_bundle_file_inventory_at(root_fd):
            raise contract.OpenEcologyPhaseAError(
                f"{expected_kind} raw-evidence bundle contains missing or "
                "surplus entries"
            )
        return manifest, captures, manifest_capture
    finally:
        os.close(root_fd)


def _directory_matches_identity(
    metadata: os.stat_result,
    *,
    device: int,
    inode: int,
) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_dev == device
        and metadata.st_ino == inode
    )


def _create_owned_staging_directory(
    parent: Path,
    *,
    destination_name: str,
    field: str,
) -> _OwnedStagingDirectory:
    """Create and pin a private staging inode before producer code runs."""

    parent_descriptor = _open_real_directory(
        parent,
        field=f"{field} output parent",
    )
    directory_descriptor: int | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if (
            parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise _error(
                f"{field} output parent must be process-owned and not "
                "group/world-writable"
            )
        for _attempt in range(128):
            staging_name = (
                f".{destination_name}.pending-{os.getpid()}-{os.urandom(8).hex()}"
            )
            try:
                os.mkdir(
                    staging_name,
                    mode=0o700,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as error:
                raise _error(
                    f"{field} staging directory could not be created safely"
                ) from error
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                directory_descriptor = os.open(
                    staging_name,
                    flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise _error(
                    f"{field} staging directory could not be pinned"
                ) from error
            metadata = os.fstat(directory_descriptor)
            named_metadata = os.stat(
                staging_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not _directory_matches_identity(
                    metadata,
                    device=named_metadata.st_dev,
                    inode=named_metadata.st_ino,
                )
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent_metadata.st_dev
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise _error(f"{field} staging directory authority is invalid")
            current_parent = os.lstat(parent)
            if not _directory_matches_identity(
                current_parent,
                device=parent_metadata.st_dev,
                inode=parent_metadata.st_ino,
            ):
                raise _error(f"{field} output parent namespace changed")
            os.fsync(parent_descriptor)
            return _OwnedStagingDirectory(
                path=parent / staging_name,
                parent_descriptor=parent_descriptor,
                directory_descriptor=directory_descriptor,
                parent_device=parent_metadata.st_dev,
                parent_inode=parent_metadata.st_ino,
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
    except BaseException:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(parent_descriptor)
        raise
    os.close(parent_descriptor)
    raise _error(f"{field} staging namespace could not be allocated")


def _require_owned_staging_namespace(
    staging: _OwnedStagingDirectory,
    *,
    field: str,
) -> None:
    """Require both held descriptors to retain their original named identities."""

    try:
        current_parent = os.lstat(staging.path.parent)
        current_directory = os.stat(
            staging.path.name,
            dir_fd=staging.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise _error(f"{field} staging directory namespace changed") from error
    held_parent = os.fstat(staging.parent_descriptor)
    held_directory = os.fstat(staging.directory_descriptor)
    if (
        not _directory_matches_identity(
            held_parent,
            device=staging.parent_device,
            inode=staging.parent_inode,
        )
        or not _directory_matches_identity(
            current_parent,
            device=staging.parent_device,
            inode=staging.parent_inode,
        )
        or not _directory_matches_identity(
            held_directory,
            device=staging.device,
            inode=staging.inode,
        )
        or not _directory_matches_identity(
            current_directory,
            device=staging.device,
            inode=staging.inode,
        )
    ):
        raise _error(f"{field} staging directory namespace changed")


def _require_returned_report_matches_staging(
    staging: _OwnedStagingDirectory,
    report: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    field: str,
) -> None:
    capture = _capture_regular_file_at(
        staging.directory_descriptor,
        staging.path,
        "report.json",
        field=f"{field} staged report",
        maximum_bytes=_MAX_AUTHORITY_JSON_BYTES,
    )
    if capture.payload != _canonical_json_file_bytes(report):
        raise _error(f"{field} staged report bytes differ from producer result")
    evidence_kind = report.get("evidence_kind")
    if not isinstance(evidence_kind, str) or not evidence_kind:
        raise _error(f"{field} staged report evidence kind is malformed")
    _validate_report(
        report,
        expected_kind=evidence_kind,
        preregistration=preregistration,
        authorization_time=None,
        report_path=capture.path,
    )
    _require_owned_staging_namespace(staging, field=field)


def _publish_owned_staging_directory(
    staging: _OwnedStagingDirectory,
    destination: Path,
    *,
    field: str,
) -> None:
    """Exclusively publish the still-owned staged inode through its held parent."""

    _require_owned_staging_namespace(staging, field=field)
    try:
        os.stat(
            destination.name,
            dir_fd=staging.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as error:
        raise _error(
            f"{field} output destination could not be inspected safely"
        ) from error
    else:
        raise _error(f"{field} output destination appeared during production")
    try:
        _rename_name_no_replace(
            staging.parent_descriptor,
            source_name=staging.path.name,
            destination_name=destination.name,
            destination_path=destination,
        )
    except FileExistsError as error:
        raise _error(
            f"{field} output destination appeared during production"
        ) from error
    try:
        published = os.stat(
            destination.name,
            dir_fd=staging.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise _error(f"{field} staging directory namespace changed") from error
    held = os.fstat(staging.directory_descriptor)
    if not _directory_matches_identity(
        held,
        device=staging.device,
        inode=staging.inode,
    ) or not _directory_matches_identity(
        published,
        device=staging.device,
        inode=staging.inode,
    ):
        raise _error(f"{field} staging directory namespace changed")
    try:
        os.stat(
            staging.path.name,
            dir_fd=staging.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise _error(f"{field} staging directory namespace changed")
    current_parent = os.lstat(staging.path.parent)
    if not _directory_matches_identity(
        current_parent,
        device=staging.parent_device,
        inode=staging.parent_inode,
    ):
        raise _error(f"{field} output parent namespace changed")
    os.fsync(staging.parent_descriptor)


def _publish_new_authority_bundle(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    field: str,
    producer: Callable[[Path], dict[str, object]],
) -> dict[str, object]:
    """Publish a fully validated authority directory without replacement."""

    contract = _contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    requested = Path(output_directory)
    if requested.exists() or requested.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{field} output directory must be new and absent"
        )
    raw_parent = requested.parent
    if raw_parent.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{field} output parent must be one real existing directory"
        )
    parent = raw_parent.resolve()
    if not parent.is_dir():
        raise contract.OpenEcologyPhaseAError(
            f"{field} output parent must be one real existing directory"
        )
    destination = parent / requested.name
    staging = _create_owned_staging_directory(
        parent,
        destination_name=destination.name,
        field=field,
    )
    try:
        report = producer(staging.path)
        _require_owned_staging_namespace(staging, field=field)
        _require_returned_report_matches_staging(
            staging,
            report,
            preregistration=preregistration,
            field=field,
        )
        os.fsync(staging.directory_descriptor)
        _publish_owned_staging_directory(
            staging,
            destination,
            field=field,
        )
        return report
    finally:
        os.close(staging.directory_descriptor)
        os.close(staging.parent_descriptor)


def _new_bundle_root(
    output_directory: str | Path,
    *,
    field: str,
    _precreated_root: bool = False,
) -> tuple[Path, Path]:
    destination = Path(output_directory)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise _error(f"{field} output parent must be one real directory")
    root = destination.parent.resolve() / destination.name
    if _precreated_root:
        if root.is_symlink() or not root.is_dir():
            raise _error(f"{field} precreated output directory authority drifted")
    else:
        if root.exists() or root.is_symlink():
            raise _error(f"{field} output directory must be new and absent")
        root.mkdir()
    raw_root = root / "raw"
    raw_root.mkdir()
    return root, raw_root


def _seal_reconstructed_authority_report(
    preregistration: Mapping[str, object],
    *,
    root: Path,
    produced_at_utc: str,
    evidence_kind: str,
    producer_name: str,
    producer_contract: str,
    raw_files: Mapping[str, Path],
    reconstruct: Callable[
        [Mapping[str, object], Mapping[str, _CapturedRegularFile]],
        Mapping[str, object],
    ],
) -> dict[str, object]:
    contract = _contract()
    manifest = build_open_ecology_raw_evidence_manifest(
        preregistration,
        evidence_kind=evidence_kind,
        bundle_root=root,
        produced_at_utc=produced_at_utc,
        producer_name=producer_name,
        producer_contract=producer_contract,
        raw_files=raw_files,
    )
    manifest_path = root / "raw-evidence-manifest.json"
    _write_new_json(manifest_path, manifest)
    captured_manifest, captures, _manifest_capture = (
        _validate_raw_evidence_manifest_path(
            manifest_path,
            preregistration=preregistration,
            expected_kind=evidence_kind,
            authorization_time=None,
            expected_produced_at_utc=produced_at_utc,
        )
    )
    reconstructed = _mapping(
        reconstruct(captured_manifest, captures),
        field=f"{evidence_kind} reconstructed evidence",
    )
    facts = _mapping(
        reconstructed["facts"],
        field=f"{evidence_kind} reconstructed facts",
    )
    report: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _source_binding(preregistration),
        "produced_at_utc": produced_at_utc,
        "raw_evidence_manifest": contract._file_reference(
            manifest_path,
            base=root,
        ),
        "facts": dict(facts),
    }
    report["exact_digest"] = stable_payload_digest(report)
    report_path = root / "report.json"
    _write_new_json(report_path, report)
    _validate_report(
        report,
        expected_kind=evidence_kind,
        preregistration=preregistration,
        authorization_time=None,
        report_path=report_path,
        require_authority_verifier=False,
    )
    return report


def _capture_authority_report_bundle(
    report_path: Path,
    *,
    preregistration: Mapping[str, object],
    expected_kind: str,
    authorization_time: datetime | None,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, _CapturedRegularFile],
]:
    root = report_path.parent.absolute()
    root_fd = _open_real_directory(
        root,
        field=f"{expected_kind} report bundle",
    )
    try:
        report_capture = _capture_regular_file_at(
            root_fd,
            root,
            report_path.name,
            field=f"{expected_kind} report",
            maximum_bytes=_MAX_AUTHORITY_JSON_BYTES,
        )
    finally:
        os.close(root_fd)
    report = _strict_json_from_captured_bytes(
        report_capture,
        field=f"{expected_kind} report",
    )
    _manifest_path, manifest, captures = _load_report_raw_evidence_manifest(
        report_capture.path,
        report,
        expected_kind=expected_kind,
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    return report, manifest, captures


def _copy_captured_regular_file(
    source: str | Path,
    destination: Path,
    *,
    field: str,
) -> Path:
    candidate = Path(source)
    if candidate.parent.is_symlink():
        raise _error(f"{field} parent must be one real directory")
    parent = candidate.parent.resolve()
    parent_fd = _open_real_directory(parent, field=f"{field} parent")
    try:
        captured = _capture_regular_file_at(
            parent_fd,
            parent,
            candidate.name,
            field=field,
            maximum_bytes=_MAX_RAW_EVIDENCE_FILE_BYTES,
        )
    finally:
        os.close(parent_fd)
    return _write_new_bytes(destination, captured.payload)


def _reconstruct_preregistration_roundtrip_from_manifest(
    manifest_path: Path,
    *,
    preregistration: Mapping[str, object],
    authorization_time: datetime | None,
    expected_produced_at_utc: object,
) -> dict[str, object]:
    manifest, captures, _manifest_capture = _validate_raw_evidence_manifest_path(
        manifest_path,
        preregistration=preregistration,
        expected_kind="preregistration_roundtrip_fail_closed",
        authorization_time=authorization_time,
        expected_produced_at_utc=expected_produced_at_utc,
    )
    return _reconstruct_preregistration_roundtrip_from_captures(
        manifest,
        captures,
        preregistration=preregistration,
    )


def _reconstruct_preregistration_roundtrip_from_captures(
    manifest: Mapping[str, object],
    captures: Mapping[str, _CapturedRegularFile],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct D9 from the exact bundle snapshot bound to the report."""

    contract = _contract()
    if manifest.get("producer") != {
        "name": "produce_preregistration_roundtrip_fail_closed_report",
        "contract": "open_ecology_preregistration_roundtrip_raw_producer_v1",
    }:
        raise contract.OpenEcologyPhaseAError("D9 raw producer identity drifted")
    expected_roles = {
        "builder_inputs",
        "duplicate_key_input",
        "preregistration",
        "source_observations",
        "unknown_input",
    }
    if set(captures) != expected_roles:
        raise contract.OpenEcologyPhaseAError("D9 raw evidence roles are incomplete")

    raw_preregistration = _strict_json_from_captured_bytes(
        captures["preregistration"],
        field="D9 raw preregistration",
    )
    contract.validate_open_ecology_phase_a_preregistration(raw_preregistration)
    if raw_preregistration != dict(preregistration):
        raise contract.OpenEcologyPhaseAError(
            "D9 raw preregistration differs from the supplied campaign"
        )
    builder_inputs = _strict_json_from_captured_bytes(
        captures["builder_inputs"],
        field="D9 builder inputs",
    )
    _exact_key_set(
        builder_inputs,
        {
            "source_commit",
            "source_manifest_sha256",
            "archive_tool_authority_sha256",
            "runtime_contract",
            "throughput_gate",
        },
        field="D9 builder inputs",
    )
    rebuilt = contract.build_open_ecology_phase_a_preregistration(
        source_commit=builder_inputs["source_commit"],
        source_manifest_sha256=builder_inputs["source_manifest_sha256"],
        archive_tool_authority_sha256=builder_inputs["archive_tool_authority_sha256"],
        runtime_contract=_mapping(
            builder_inputs["runtime_contract"],
            field="D9 runtime contract",
        ),
        throughput_gate=_mapping(
            builder_inputs["throughput_gate"],
            field="D9 throughput gate",
        ),
    )
    if rebuilt != raw_preregistration or (
        captures["preregistration"].payload != _canonical_json_file_bytes(rebuilt)
    ):
        raise contract.OpenEcologyPhaseAError(
            "D9 machine preregistration did not roundtrip exact canonical bytes"
        )

    unknown_input = _strict_json_from_captured_bytes(
        captures["unknown_input"],
        field="D9 unknown input",
    )
    unknown_rejections = _require_expected_rejection(
        lambda: contract.validate_open_ecology_phase_a_preregistration(unknown_input),
        expected_message="fields differ",
        field="D9 unknown input",
    )
    source_observations = _strict_json_from_captured_bytes(
        captures["source_observations"],
        field="D9 source observations",
    )
    _exact_key_set(
        source_observations,
        {"clean", "dirty", "stale"},
        field="D9 source observations",
    )
    parsed_observations: dict[str, Mapping[str, object]] = {}
    for case_id in ("clean", "dirty", "stale"):
        observation = _mapping(
            source_observations[case_id],
            field=f"D9 source observation {case_id}",
        )
        _exact_key_set(
            observation,
            {
                "observed_commit",
                "observed_status_porcelain",
                "observed_manifest_sha256",
                "observed_preregistration_sha256",
            },
            field=f"D9 source observation {case_id}",
        )
        parsed_observations[case_id] = observation
    contract.validate_open_ecology_phase_a_source_observation(
        rebuilt,
        **parsed_observations["clean"],
    )
    dirty_rejections = _require_expected_rejection(
        lambda: contract.validate_open_ecology_phase_a_source_observation(
            rebuilt,
            **parsed_observations["dirty"],
        ),
        expected_message="exact clean preregistered Git source",
        field="D9 dirty source",
    )
    stale_rejections = _require_expected_rejection(
        lambda: contract.validate_open_ecology_phase_a_source_observation(
            rebuilt,
            **parsed_observations["stale"],
        ),
        expected_message="exact clean preregistered Git source",
        field="D9 stale source",
    )
    duplicate_rejections = _require_expected_rejection(
        lambda: _strict_json_from_captured_bytes(
            captures["duplicate_key_input"],
            field="D9 duplicate-key input",
        ),
        expected_message="duplicate JSON key",
        field="D9 duplicate key",
    )

    training = _mapping(rebuilt["training"], field="D9 rebuilt training")
    run_matrix = _sequence(
        training["run_matrix"],
        field="D9 rebuilt training.run_matrix",
    )
    task_ids: list[str] = []
    for index, raw_row in enumerate(run_matrix):
        row = _mapping(raw_row, field=f"D9 run matrix row {index}")
        task_id = row.get("run_id")
        if not isinstance(task_id, str) or not task_id:
            raise contract.OpenEcologyPhaseAError(
                "D9 run matrix contains an invalid run_id"
            )
        task_ids.append(task_id)
    seed_contract = _mapping(
        rebuilt["seed_contract"],
        field="D9 rebuilt seed contract",
    )
    facts = {
        "roundtrip_exact_digest": rebuilt["exact_digest"],
        "run_matrix_row_count": len(run_matrix),
        "unique_task_id_count": len(set(task_ids)),
        "seed_registry_sha256": seed_contract["registry_sha256"],
        "configuration_sha256": rebuilt["configuration_sha256"],
        "unknown_input_rejection_count": unknown_rejections,
        "dirty_source_rejection_count": dirty_rejections,
        "stale_source_rejection_count": stale_rejections,
        "duplicate_key_rejection_count": duplicate_rejections,
    }
    return {
        "evidence_kind": manifest["evidence_kind"],
        "campaign_digest": manifest["campaign_digest"],
        "configuration_sha256": manifest["configuration_sha256"],
        "source": manifest["source"],
        "facts": facts,
    }


def _reconstruct_exact_sha_phase_a_training_from_captures(
    manifest: Mapping[str, object],
    captures: Mapping[str, _CapturedRegularFile],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    from evolution_sim.io.source_manifest import source_file_hash_manifest
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    contract = _contract()
    if manifest.get("producer") != {
        "name": "produce_exact_sha_phase_a_training_and_torch_ci_report",
        "contract": "exact_source_full_shape_gpu_torch_and_github_check_runs_v1",
    }:
        raise contract.OpenEcologyPhaseAError("D10 raw producer identity drifted")
    expected_roles = {
        "benchmark_heritable",
        "benchmark_zero_all",
        "ci_workflow",
        "github_check_runs",
        "qualification_identity",
        "source_observation",
        "torch_command",
        "torch_test_source",
        "torch_validation",
        "torch_validator_source",
    }
    if set(captures) != expected_roles:
        raise contract.OpenEcologyPhaseAError("D10 raw evidence roles are incomplete")
    for role, path in (
        ("ci_workflow", _REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"),
        (
            "torch_test_source",
            _REPOSITORY_ROOT / "python" / "tests" / "test_mind_v1.py",
        ),
        (
            "torch_validator_source",
            _REPOSITORY_ROOT / "scripts" / "validate_mind_torch.py",
        ),
    ):
        _require_capture_matches_live_file(
            captures[role],
            path,
            field=f"D10 {role}",
        )

    source = _mapping(preregistration["source"], field="D10 source")
    source_commit = str(source["commit"])
    source_observation = _strict_json_from_captured_bytes(
        captures["source_observation"],
        field="D10 source observation",
    )
    _exact_key_set(
        source_observation,
        {
            "schema_version",
            "head",
            "status",
            "source_manifest",
            "import_probe",
            "import_probe_code_sha256",
        },
        field="D10 source observation",
    )
    if (
        source_observation["schema_version"]
        != qualification.SOURCE_OBSERVATION_SCHEMA_VERSION
        or source_observation["import_probe_code_sha256"]
        != qualification.import_probe_code_sha256()
    ):
        raise contract.OpenEcologyPhaseAError("D10 source probe contract drifted")
    head_receipt = _validate_qualification_command_receipt(
        source_observation["head"],
        field="D10 git HEAD",
    )
    status_receipt = _validate_qualification_command_receipt(
        source_observation["status"],
        field="D10 git status",
    )
    if (
        Path(str(_sequence(head_receipt["argv"], field="D10 HEAD argv")[0])).name
        != "git"
        or list(_sequence(head_receipt["argv"], field="D10 HEAD argv"))[-2:]
        != ["rev-parse", "HEAD"]
        or head_receipt["returncode"] != 0
        or str(head_receipt["stderr"]).strip()
        or str(head_receipt["stdout"]).strip() != source_commit
        or list(_sequence(status_receipt["argv"], field="D10 status argv"))[-3:]
        != ["status", "--porcelain=v1", "--untracked-files=normal"]
        or status_receipt["returncode"] != 0
        or str(status_receipt["stderr"]).strip()
        or str(status_receipt["stdout"])
    ):
        raise contract.OpenEcologyPhaseAError(
            "D10 did not observe one clean exact Git checkout"
        )
    observed_manifest = _mapping(
        source_observation["source_manifest"],
        field="D10 source manifest",
    )
    live_manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
    if (
        dict(observed_manifest) != live_manifest
        or observed_manifest.get("aggregate_sha256") != source["manifest_sha256"]
    ):
        raise contract.OpenEcologyPhaseAError(
            "D10 source manifest does not match exact live source"
        )
    import_receipt = _validate_qualification_command_receipt(
        source_observation["import_probe"],
        field="D10 import probe",
    )
    import_argv = list(_sequence(import_receipt["argv"], field="D10 import argv"))
    if (
        import_receipt["returncode"] != 0
        or str(import_receipt["stderr"]).strip()
        or len(import_argv) != 3
        or import_argv[1] != "-c"
        or hashlib.sha256(str(import_argv[2]).encode("utf-8")).hexdigest()
        != qualification.import_probe_code_sha256()
    ):
        raise contract.OpenEcologyPhaseAError("D10 import probe command drifted")
    import_result = _strict_json_text(
        str(import_receipt["stdout"]),
        field="D10 import probe output",
    )
    _exact_key_set(
        import_result,
        {"module_relative_path", "module_sha256", "python_version"},
        field="D10 import probe output",
    )
    manifest_files = _mapping(
        observed_manifest["files"],
        field="D10 source manifest files",
    )
    if (
        import_result["module_relative_path"] != "python/evolution_sim/__init__.py"
        or import_result["module_sha256"]
        != manifest_files.get("python/evolution_sim/__init__.py")
        or not isinstance(import_result["python_version"], str)
        or not import_result["python_version"]
    ):
        raise contract.OpenEcologyPhaseAError(
            "D10 imported evolution_sim from a detached root"
        )

    identity = _strict_json_from_captured_bytes(
        captures["qualification_identity"],
        field="D10 qualification identity",
    )
    _exact_key_set(
        identity,
        {"host_class", "github_repository", "github_torch_check_name"},
        field="D10 qualification identity",
    )
    host_class = _host_class(identity["host_class"])
    if (
        identity["github_repository"] != qualification.GITHUB_REPOSITORY
        or identity["github_torch_check_name"] != qualification.GITHUB_TORCH_CHECK_NAME
    ):
        raise contract.OpenEcologyPhaseAError("D10 GitHub identity drifted")
    workflow_text = _utf8_text(
        captures["ci_workflow"].payload,
        field="D10 CI workflow",
    )
    _validate_nonoptional_torch_ci_workflow(workflow_text)
    raw_github_receipt = _strict_json_from_captured_bytes(
        captures["github_check_runs"],
        field="D10 GitHub check receipt",
    )
    expected_endpoint = (
        f"repos/{qualification.GITHUB_REPOSITORY}/commits/{source_commit}/"
        "check-runs?per_page=100"
    )
    if (
        raw_github_receipt.get("schema_version")
        == qualification.AUTHENTICATED_HTTPS_RECEIPT_SCHEMA_VERSION
    ):
        github_receipt = _validate_authenticated_github_https_receipt(
            raw_github_receipt,
            field="D10 GitHub HTTPS receipt",
        )
        request = _mapping(
            github_receipt["request"],
            field="D10 GitHub HTTPS request",
        )
        response = _mapping(
            github_receipt["response"],
            field="D10 GitHub HTTPS response",
        )
        if (
            request["host"] != "api.github.com"
            or request["path"] != f"/{expected_endpoint}"
            or response["status"] != 200
        ):
            raise contract.OpenEcologyPhaseAError(
                "D10 GitHub exact-SHA HTTPS query failed"
            )
        github_output = str(response["body"])
    else:
        github_receipt = _validate_qualification_command_receipt(
            raw_github_receipt,
            field="D10 GitHub check receipt",
        )
        github_argv = list(
            _sequence(github_receipt["argv"], field="D10 GitHub check argv")
        )
        if (
            len(github_argv) != 7
            or Path(str(github_argv[0])).name != "gh"
            or github_argv[1:]
            != [
                "api",
                "--method",
                "GET",
                "-H",
                "Accept: application/vnd.github+json",
                expected_endpoint,
            ]
            or github_receipt["returncode"] != 0
            or str(github_receipt["stderr"]).strip()
        ):
            raise contract.OpenEcologyPhaseAError(
                "D10 GitHub exact-SHA check query failed"
            )
        github_output = str(github_receipt["stdout"])
    check_payload = _strict_json_text(
        github_output,
        field="D10 GitHub check-runs output",
    )
    check_runs = _sequence(
        check_payload.get("check_runs"),
        field="D10 GitHub check-runs",
    )
    matching_checks: list[Mapping[str, object]] = []
    for index, raw_check in enumerate(check_runs):
        check = _mapping(raw_check, field=f"D10 GitHub check-run {index}")
        if (
            check.get("name") == qualification.GITHUB_TORCH_CHECK_NAME
            and check.get("head_sha") == source_commit
        ):
            matching_checks.append(check)
    if not matching_checks:
        raise contract.OpenEcologyPhaseAError(
            "D10 exact-SHA GitHub Torch check is missing"
        )
    latest_check = max(
        matching_checks,
        key=lambda value: _integer(
            value.get("id"),
            field="D10 GitHub check id",
            minimum=1,
        ),
    )
    github_app = _mapping(
        latest_check.get("app"),
        field="D10 GitHub check app",
    )
    if (
        latest_check.get("status") != "completed"
        or latest_check.get("conclusion") != "success"
        or github_app.get("slug") != "github-actions"
    ):
        raise contract.OpenEcologyPhaseAError(
            "D10 latest exact-SHA GitHub Torch check did not succeed"
        )

    preliminary_reports = _mapping(
        _mapping(preregistration["throughput_gate"], field="throughput gate")[
            "reports"
        ],
        field="preliminary reports",
    )
    semantic_pairs: dict[str, object] = {}
    runtime_pairs: dict[str, object] = {}
    repeat_counts: list[int] = []
    for mode, role in (
        ("heritable", "benchmark_heritable"),
        ("zero_all", "benchmark_zero_all"),
    ):
        report = _strict_json_from_captured_bytes(
            captures[role],
            field=f"D10 {mode} benchmark",
        )
        contract._validate_phase_a_pipeline_benchmark(
            report,
            expected_source_commit=source_commit,
            expected_population_mode=mode,
        )
        protocol = _mapping(report["protocol"], field=f"D10 {mode} protocol")
        repeat_counts.append(
            _positive_integer(protocol["repeats"], field=f"D10 {mode} repeats")
        )
        determinism = _mapping(
            report["determinism"],
            field=f"D10 {mode} determinism",
        )
        preliminary = _mapping(
            preliminary_reports[mode],
            field=f"D10 preliminary {mode}",
        )
        if determinism != _mapping(
            preliminary["determinism"],
            field=f"D10 preliminary {mode} determinism",
        ):
            raise contract.OpenEcologyPhaseAError(
                "D10 fresh benchmark semantic evidence differs from preregistration"
            )
        semantic_pairs[mode] = dict(determinism)
        runtime_pairs[mode] = dict(
            _mapping(report["runtime"], field=f"D10 {mode} runtime")
        )

    torch_command = _validate_qualification_command_receipt(
        _strict_json_from_captured_bytes(
            captures["torch_command"],
            field="D10 Torch command",
        ),
        field="D10 Torch command",
    )
    torch_argv = list(_sequence(torch_command["argv"], field="D10 Torch argv"))
    torch_output_index = (
        torch_argv.index("--output") if "--output" in torch_argv else -1
    )
    if (
        torch_command["returncode"] != 0
        or str(torch_command["stderr"]).strip()
        or len(torch_argv) != 5
        or Path(str(torch_argv[1])).name != "validate_mind_torch.py"
        or torch_argv[2] != "--require-cuda"
        or torch_output_index != 3
        or Path(str(torch_argv[4])).name
        != Path(captures["torch_validation"].relative_path).name
    ):
        raise contract.OpenEcologyPhaseAError(
            "D10 local Torch/CUDA validation command failed"
        )
    torch_report = _strict_json_from_captured_bytes(
        captures["torch_validation"],
        field="D10 Torch validation",
    )
    _exact_key_set(
        torch_report,
        {
            "schema_version",
            "validation_policy",
            "require_cuda",
            "runtime_environment",
            "dependency_versions",
            "torch_environment",
            "unittest",
            "cuda_training_smoke",
            "passed",
        },
        field="D10 Torch validation",
    )
    unittest_result = _mapping(
        torch_report["unittest"],
        field="D10 Torch unittest result",
    )
    _exact_key_set(
        unittest_result,
        {
            "test_module",
            "test_class",
            "test_count",
            "selected_tests",
            "seconds",
            "passed",
            "failure_count",
            "error_count",
            "skip_count",
            "failures",
            "errors",
            "skips",
            "runner_output",
        },
        field="D10 Torch unittest result",
    )
    discovered_tests = _discover_torch_gated_tests(
        captures["torch_test_source"].payload
    )
    selected_tests = list(
        _sequence(
            unittest_result["selected_tests"],
            field="D10 selected Torch tests",
        )
    )
    runtime_environment = _mapping(
        torch_report["runtime_environment"],
        field="D10 Torch runtime environment",
    )
    torch_environment = _mapping(
        torch_report["torch_environment"],
        field="D10 Torch environment",
    )
    smoke = _mapping(
        torch_report["cuda_training_smoke"],
        field="D10 CUDA smoke",
    )
    failure_count = _integer(
        unittest_result["failure_count"],
        field="D10 Torch failures",
        minimum=0,
    ) + _integer(
        unittest_result["error_count"],
        field="D10 Torch errors",
        minimum=0,
    )
    skip_count = _integer(
        unittest_result["skip_count"],
        field="D10 Torch skips",
        minimum=0,
    )
    runtime_git_head = runtime_environment.get("git_head")
    if (
        torch_report["schema_version"] != "mind_torch_validation_v1"
        or torch_report["validation_policy"]
        != "explicit_optional_mind_ml_stack_validation_v1"
        or torch_report["require_cuda"] is not True
        or torch_report["passed"] is not True
        or runtime_environment.get("git_dirty") is not False
        or not isinstance(runtime_git_head, str)
        or re.fullmatch(r"[0-9a-f]{7,40}", runtime_git_head) is None
        or not source_commit.startswith(runtime_git_head)
        or torch_environment.get("cuda_available") is not True
        or _positive_integer(
            torch_environment.get("cuda_device_count"),
            field="D10 CUDA device count",
        )
        < 1
        or smoke.get("passed") is not True
        or smoke.get("requested_device") != "cuda"
        or smoke.get("resolved_device") != "cuda"
        or smoke.get("cuda_available") is not True
        or _positive_integer(
            smoke.get("cuda_device_count"),
            field="D10 CUDA smoke device count",
        )
        < 1
        or unittest_result["passed"] is not True
        or selected_tests != list(discovered_tests)
        or unittest_result["test_count"] != len(discovered_tests)
        or failure_count != 0
        or skip_count != 0
    ):
        raise contract.OpenEcologyPhaseAError(
            "D10 local Torch/CUDA suite did not pass its complete exact-source contract"
        )
    facts = {
        "fresh_checkout_count": 1,
        "dirty_checkout_count": 0,
        "source_manifest_match_count": 1,
        "import_root_match_count": 1,
        "intended_training_host_classes": [host_class],
        "tested_training_host_classes": [host_class],
        "full_shape_training_benchmark_count_by_host_class": {
            host_class: min(repeat_counts),
        },
        "runtime_contract_sha256_by_host_class": {
            host_class: stable_payload_digest(
                {
                    "benchmark_runtime_by_mode": runtime_pairs,
                    "dependency_versions": torch_report["dependency_versions"],
                    "python_version": import_result["python_version"],
                    "torch_environment": torch_environment,
                }
            ),
        },
        "semantic_evidence_sha256_by_host_class": {
            host_class: stable_payload_digest(semantic_pairs),
        },
        "cross_host_semantic_mismatch_count": 0,
        "torch_ci_lane_required": True,
        "torch_ci_test_count": len(discovered_tests),
        "torch_ci_failure_count": failure_count,
        "torch_ci_skip_count": skip_count,
    }
    return _authority_reconstruction(manifest, facts)


def _reconstruct_phase_a_training_throughput_from_captures(
    manifest: Mapping[str, object],
    captures: Mapping[str, _CapturedRegularFile],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    contract = _contract()
    if manifest.get("producer") != {
        "name": "produce_phase_a_training_throughput_report",
        "contract": "fresh_full_shape_pair_with_live_gpu_host_kernel_sampling_v1",
    }:
        raise contract.OpenEcologyPhaseAError(
            "Phase-A throughput raw producer identity drifted"
        )
    if set(captures) != {
        "benchmark_heritable",
        "benchmark_zero_all",
        "resource_telemetry",
    }:
        raise contract.OpenEcologyPhaseAError(
            "Phase-A throughput raw evidence roles are incomplete"
        )
    source_commit = str(
        _mapping(preregistration["source"], field="throughput source")["commit"]
    )
    telemetry = _strict_json_from_captured_bytes(
        captures["resource_telemetry"],
        field="Phase-A resource telemetry",
    )
    _exact_key_set(
        telemetry,
        {
            "schema_version",
            "source_commit",
            "host_class",
            "sample_interval_seconds",
            "benchmark_receipts",
            "benchmark_report_sha256_by_mode",
            "resource_samples",
        },
        field="Phase-A resource telemetry",
    )
    host_class = _host_class(telemetry["host_class"])
    if (
        telemetry["schema_version"] != qualification.RESOURCE_TELEMETRY_SCHEMA_VERSION
        or telemetry["source_commit"] != source_commit
        or _positive_number(
            telemetry["sample_interval_seconds"],
            field="resource sample interval",
        )
        > 30.0
    ):
        raise contract.OpenEcologyPhaseAError(
            "Phase-A resource telemetry contract drifted"
        )
    receipts = _mapping(
        telemetry["benchmark_receipts"],
        field="Phase-A benchmark receipts",
    )
    report_digests = _mapping(
        telemetry["benchmark_report_sha256_by_mode"],
        field="Phase-A benchmark report digests",
    )
    _exact_key_set(
        receipts,
        {"heritable", "zero_all"},
        field="Phase-A benchmark receipts",
    )
    _exact_key_set(
        report_digests,
        {"heritable", "zero_all"},
        field="Phase-A benchmark report digests",
    )
    reports: dict[str, Mapping[str, object]] = {}
    elapsed_by_mode: dict[str, dict[int, float]] = {}
    command_windows: dict[str, tuple[datetime, datetime]] = {}
    preliminary_reports = _mapping(
        _mapping(preregistration["throughput_gate"], field="throughput gate")[
            "reports"
        ],
        field="preliminary reports",
    )
    semantic_mismatch_count = 0
    repeat_counts: list[int] = []
    for mode, role in (
        ("heritable", "benchmark_heritable"),
        ("zero_all", "benchmark_zero_all"),
    ):
        report = _strict_json_from_captured_bytes(
            captures[role],
            field=f"fresh throughput {mode} benchmark",
        )
        reports[mode] = report
        elapsed_by_mode[mode] = contract._validate_phase_a_pipeline_benchmark(
            report,
            expected_source_commit=source_commit,
            expected_population_mode=mode,
        )
        if report_digests[mode] != stable_payload_digest(report):
            raise contract.OpenEcologyPhaseAError(
                "fresh throughput benchmark digest drifted"
            )
        receipt = _validate_benchmark_command_receipt(
            receipts[mode],
            mode=mode,
            source_bytes=captures[role].payload,
        )
        command_windows[mode] = (
            _parse_utc(
                receipt["started_at_utc"],
                field=f"{mode} benchmark started",
            ),
            _parse_utc(
                receipt["finished_at_utc"],
                field=f"{mode} benchmark finished",
            ),
        )
        if receipt["returncode"] != 0:
            raise contract.OpenEcologyPhaseAError(
                "fresh throughput benchmark returned nonzero"
            )
        protocol = _mapping(report["protocol"], field=f"throughput {mode} protocol")
        repeat_counts.append(
            _positive_integer(protocol["repeats"], field=f"throughput {mode} repeats")
        )
        if _mapping(
            report["determinism"],
            field=f"throughput {mode} determinism",
        ) != _mapping(
            _mapping(
                preliminary_reports[mode],
                field=f"preliminary {mode} report",
            )["determinism"],
            field=f"preliminary {mode} determinism",
        ):
            semantic_mismatch_count += 1
    produced_at = _parse_utc(
        manifest["produced_at_utc"],
        field="Phase-A throughput produced_at_utc",
    )
    heritable_window = command_windows["heritable"]
    zero_window = command_windows["zero_all"]
    if (
        heritable_window[1] > zero_window[0]
        or zero_window[0] - heritable_window[1] > timedelta(minutes=5)
        or zero_window[1] > produced_at
        or produced_at - zero_window[1] > timedelta(minutes=5)
    ):
        raise contract.OpenEcologyPhaseAError(
            "Phase-A benchmark command series is stale or detached"
        )
    samples = _sequence(
        telemetry["resource_samples"],
        field="Phase-A resource samples",
    )
    if not samples:
        raise contract.OpenEcologyPhaseAError(
            "Phase-A throughput has no live resource samples"
        )
    observed_modes: set[str] = set()
    boot_ids: set[str] = set()
    ram_shares: list[float] = []
    gpu_memory_shares: list[float] = []
    temperatures_below_throttle: list[bool] = []
    swap_values: list[int] = []
    xid_values: list[int] = []
    oom_values: list[int] = []
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, field=f"resource sample {index}")
        _exact_key_set(
            sample,
            {"population_mode", "sampled_at_utc", "host"},
            field=f"resource sample {index}",
        )
        mode = sample["population_mode"]
        if mode not in {"heritable", "zero_all"}:
            raise contract.OpenEcologyPhaseAError(
                "resource sample population mode drifted"
            )
        observed_modes.add(str(mode))
        sampled_at = _parse_flexible_utc(
            sample["sampled_at_utc"],
            field=f"resource sample {index} time",
        )
        mode_window = command_windows[str(mode)]
        if sampled_at < mode_window[0] or sampled_at > mode_window[1]:
            raise contract.OpenEcologyPhaseAError(
                "resource sample is outside its benchmark command window"
            )
        host = _mapping(sample["host"], field=f"resource sample {index} host")
        memory = _mapping(
            host.get("memory"),
            field=f"resource sample {index} memory",
        )
        ram_shares.append(
            _number(
                memory.get("ram_used_share"),
                field=f"resource sample {index} RAM share",
            )
        )
        swap_values.append(
            _integer(
                memory.get("swap_used_bytes"),
                field=f"resource sample {index} swap",
                minimum=0,
            )
        )
        xid_values.append(
            _integer(
                host.get("xid_count"),
                field=f"resource sample {index} Xid count",
                minimum=0,
            )
        )
        oom_values.append(
            _integer(
                host.get("oom_count"),
                field=f"resource sample {index} OOM count",
                minimum=0,
            )
        )
        boot_id = host.get("boot_id")
        if not isinstance(boot_id, str) or not boot_id:
            raise contract.OpenEcologyPhaseAError(
                "resource sample boot identity is malformed"
            )
        boot_ids.add(boot_id)
        gpus = _sequence(host.get("gpus"), field=f"resource sample {index} GPUs")
        if not gpus:
            raise contract.OpenEcologyPhaseAError("resource sample contains no GPU")
        for raw_gpu in gpus:
            gpu = _mapping(raw_gpu, field=f"resource sample {index} GPU")
            gpu_memory_shares.append(
                _number(
                    gpu.get("memory_used_share"),
                    field=f"resource sample {index} GPU memory share",
                )
            )
            current = _integer(
                gpu.get("temperature_c"),
                field=f"resource sample {index} GPU temperature",
                minimum=0,
            )
            slowdown = _positive_integer(
                gpu.get("slowdown_temperature_c"),
                field=f"resource sample {index} slowdown temperature",
            )
            temperatures_below_throttle.append(current < slowdown)
    if observed_modes != {"heritable", "zero_all"} or len(boot_ids) != 1:
        raise contract.OpenEcologyPhaseAError(
            "resource sampling did not span both benchmark arms on one boot"
        )
    baseline_swap = swap_values[0]
    baseline_xid = xid_values[0]
    baseline_oom = oom_values[0]
    if (
        min(swap_values) < baseline_swap
        or min(xid_values) < baseline_xid
        or min(oom_values) < baseline_oom
    ):
        raise contract.OpenEcologyPhaseAError(
            "resource counters moved backwards during qualification"
        )
    preliminary = _mapping(
        preregistration["throughput_gate"],
        field="throughput gate",
    )
    preliminary_projection = _mapping(
        preliminary["resource_projection"],
        field="preliminary resource projection",
    )
    selected_workers = _positive_integer(
        preliminary["selected_rollout_workers"],
        field="selected rollout workers",
    )
    projection = contract._build_phase_a_resource_projection(
        source_commit=source_commit,
        elapsed_by_mode=elapsed_by_mode,
        selected_workers=selected_workers,
        resource_envelope=_mapping(
            preliminary_projection["resource_envelope"],
            field="resource envelope",
        ),
    )
    observed_nonfinite_count = _count_nonfinite_numbers(
        {
            "benchmark_reports": reports,
            "resource_telemetry": telemetry,
        }
    )
    facts = {
        "preliminary_throughput_gate_digest": preliminary["exact_digest"],
        "measured_host_class": host_class,
        "fresh_full_shape_repetition_count": min(repeat_counts),
        "phase_a_training_projection_authoritative": projection[
            "phase_a_training_projection_authoritative"
        ],
        "selection_projection_included": False,
        "semantic_mismatch_count": semantic_mismatch_count,
        "projected_phase_a_training_seconds": projection[
            "projected_phase_a_training_seconds"
        ],
        "maximum_gpu_memory_share": max(gpu_memory_shares),
        "maximum_host_ram_share": max(ram_shares),
        "swap_delta_bytes": max(swap_values) - baseline_swap,
        "temperature_below_throttle": all(temperatures_below_throttle),
        "xid_count": max(xid_values) - baseline_xid,
        "oom_count": max(oom_values) - baseline_oom,
        "nonfinite_count": observed_nonfinite_count,
    }
    return _authority_reconstruction(manifest, facts)


def _reconstruct_campaign_storage_capacity_from_captures(
    manifest: Mapping[str, object],
    captures: Mapping[str, _CapturedRegularFile],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    from evolution_sim.io.open_ecology_campaign_storage import (
        ACTIVE_CAMPAIGN_BUDGET_BYTES,
    )
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    if manifest.get("producer") != {
        "name": "produce_campaign_storage_capacity_report",
        "contract": "sealed_archive_authority_remote_statvfs_local_drive_v3",
    }:
        raise _error("campaign storage raw producer identity drifted")
    if set(captures) != {"archive_tool_authority", "storage_measurement"}:
        raise _error("campaign storage raw evidence roles are incomplete")
    archive_authority = _strict_json_from_captured_bytes(
        captures["archive_tool_authority"],
        field="campaign storage archive authority",
    )
    _exact_key_set(
        archive_authority,
        {
            "endpoint",
            "local_tools",
            "remote_helpers",
            "remote_tools",
            "schema_version",
            "source",
        },
        field="campaign storage archive authority",
    )
    if archive_authority["schema_version"] != "open_ecology_archive_tool_authority_v1":
        raise _error("campaign storage archive authority schema drifted")
    authority_source = _mapping(
        archive_authority["source"],
        field="campaign storage archive authority source",
    )
    _exact_key_set(
        authority_source,
        {"git_sha", "manifest_sha256", "remote_repository_root"},
        field="campaign storage archive authority source",
    )
    preregistration_source = _mapping(
        preregistration["source"],
        field="campaign storage preregistration source",
    )
    if (
        captures["archive_tool_authority"].sha256
        != preregistration_source["archive_tool_authority_sha256"]
        or authority_source["git_sha"] != preregistration_source["commit"]
        or authority_source["manifest_sha256"]
        != preregistration_source["manifest_sha256"]
        or not isinstance(authority_source["remote_repository_root"], str)
        or not Path(str(authority_source["remote_repository_root"])).is_absolute()
    ):
        raise _error("campaign storage archive authority source drifted")
    endpoint = _mapping(
        archive_authority["endpoint"],
        field="campaign storage archive endpoint",
    )
    _exact_key_set(
        endpoint,
        {
            "rclone_base",
            "rclone_config",
            "ssh_effective_config_sha256",
            "ssh_connection",
            "ssh_target",
        },
        field="campaign storage archive endpoint",
    )
    endpoint_ssh_target = _host_identifier(
        endpoint["ssh_target"],
        field="campaign storage archive endpoint SSH target",
    )
    if not isinstance(endpoint["rclone_base"], str) or not str(
        endpoint["rclone_base"]
    ).startswith("gdrive:"):
        raise _error("campaign storage archive endpoint drifted")
    _sha256(
        endpoint["ssh_effective_config_sha256"],
        field="campaign storage authority SSH config",
    )
    authority_connection = _mapping(
        endpoint["ssh_connection"],
        field="campaign storage authority SSH connection",
    )
    _exact_key_set(
        authority_connection,
        {
            "address",
            "authenticated_host",
            "authentication",
            "host_key",
            "port",
        },
        field="campaign storage authority SSH connection",
    )
    if (
        not all(
            isinstance(authority_connection[field], str)
            and bool(authority_connection[field])
            for field in (
                "address",
                "authenticated_host",
                "authentication",
                "host_key",
            )
        )
        or authority_connection["authentication"] != "publickey"
        or not 1
        <= _integer(
            authority_connection["port"],
            field="campaign storage authority SSH port",
            minimum=1,
        )
        <= 65535
    ):
        raise _error("campaign storage authority SSH connection is malformed")
    authority_rclone_config = _archive_authority_file_pin(
        endpoint["rclone_config"],
        field="campaign storage authority rclone config",
    )
    local_tools = _mapping(
        archive_authority["local_tools"],
        field="campaign storage local tools",
    )
    _exact_key_set(
        local_tools,
        {"git", "rclone", "ssh", "zstd"},
        field="campaign storage local tools",
    )
    local_tool_pins = {
        name: _archive_authority_file_pin(
            local_tools[name],
            field=f"campaign storage local tool {name}",
        )
        for name in ("git", "rclone", "ssh", "zstd")
    }
    remote_tools = _mapping(
        archive_authority["remote_tools"],
        field="campaign storage remote tools",
    )
    _exact_key_set(
        remote_tools,
        {"env", "git", "python", "sha256sum", "zstd"},
        field="campaign storage remote tools",
    )
    remote_tool_pins = {
        name: _archive_authority_file_pin(
            remote_tools[name],
            field=f"campaign storage remote tool {name}",
        )
        for name in ("env", "git", "python", "sha256sum", "zstd")
    }
    remote_helpers = _mapping(
        archive_authority["remote_helpers"],
        field="campaign storage remote helpers",
    )
    from evolution_sim.io.open_ecology_archive_authority import REMOTE_HELPER_PATHS

    _exact_key_set(
        remote_helpers,
        set(REMOTE_HELPER_PATHS),
        field="campaign storage remote helpers",
    )
    for path, digest in remote_helpers.items():
        _sha256(digest, field=f"campaign storage remote helper {path}")
    measurement = _strict_json_from_captured_bytes(
        captures["storage_measurement"],
        field="campaign storage measurement",
    )
    _exact_key_set(
        measurement,
        {
            "schema_version",
            "checked_at_utc",
            "measurement_location",
            "target_ssh",
            "target_directory",
            "archive_authority_sha256",
            "ssh_effective_config_sha256",
            "ssh_verbose_stderr_sha256",
            "ssh_connection",
            "ssh_config_receipt",
            "remote_probe_code_sha256",
            "remote_probe_receipt",
            "drive_remote",
            "rclone_config",
            "drive_receipt",
        },
        field="campaign storage measurement",
    )
    checked_at = _parse_utc(
        measurement["checked_at_utc"],
        field="campaign storage checked_at_utc",
    )
    produced_at = _parse_utc(
        manifest["produced_at_utc"],
        field="campaign storage produced_at_utc",
    )
    if checked_at > produced_at or produced_at - checked_at > timedelta(minutes=5):
        raise _error("campaign storage measurement time is detached from production")
    if (
        measurement["schema_version"]
        != qualification.STORAGE_MEASUREMENT_SCHEMA_VERSION
        or measurement["measurement_location"]
        != "remote_ssh_target_with_local_drive_v1"
        or measurement["target_ssh"] != endpoint_ssh_target
        or measurement["archive_authority_sha256"]
        != captures["archive_tool_authority"].sha256
        or measurement["ssh_effective_config_sha256"]
        != endpoint["ssh_effective_config_sha256"]
        or measurement["ssh_connection"] != authority_connection
        or measurement["drive_remote"] != "gdrive:"
    ):
        raise _error("campaign storage measurement contract drifted")
    target_ssh = endpoint_ssh_target
    target_directory = measurement["target_directory"]
    if (
        not isinstance(target_directory, str)
        or not target_directory
        or not Path(target_directory).is_absolute()
        or str(Path(target_directory)) != target_directory
    ):
        raise _error("campaign storage target directory is malformed")
    ssh_config_receipt = _validate_qualification_command_receipt(
        measurement["ssh_config_receipt"],
        field="campaign SSH effective config",
    )
    ssh_config_argv = list(
        _sequence(
            ssh_config_receipt["argv"],
            field="campaign SSH effective config argv",
        )
    )
    expected_ssh_config_tail = [
        "-G",
        *qualification.SEALED_SSH_OPTIONS,
        target_ssh,
    ]
    ssh_config_executable = _mapping(
        ssh_config_receipt["executable"],
        field="campaign SSH config executable",
    )
    ssh_config_stdout = str(ssh_config_receipt["stdout"])
    ssh_config_sha256 = _sha256(
        measurement["ssh_effective_config_sha256"],
        field="campaign SSH effective configuration SHA256",
    )
    if (
        len(ssh_config_argv) != len(expected_ssh_config_tail) + 1
        or ssh_config_argv[0] != local_tool_pins["ssh"]["path"]
        or ssh_config_executable["sha256"] != local_tool_pins["ssh"]["sha256"]
        or ssh_config_argv[1:] != expected_ssh_config_tail
        or ssh_config_receipt["returncode"] != 0
        or str(ssh_config_receipt["stderr"]).strip()
        or ssh_config_stdout != qualification.SSH_EFFECTIVE_CONFIG_REDACTION_MARKER
        or ssh_config_sha256 != endpoint["ssh_effective_config_sha256"]
    ):
        raise _error("campaign SSH effective configuration drifted")
    if (
        measurement["remote_probe_code_sha256"]
        != qualification.remote_storage_probe_code_sha256()
    ):
        raise _error("campaign remote storage probe code drifted")
    remote_receipt = _validate_qualification_command_receipt(
        measurement["remote_probe_receipt"],
        field="campaign remote filesystem measurement",
    )
    remote_argv = list(
        _sequence(remote_receipt["argv"], field="campaign remote probe argv")
    )
    expected_remote_tail = [
        "-v",
        *qualification.SEALED_SSH_OPTIONS,
        target_ssh,
        qualification.storage_remote_probe_command(
            target_directory,
            remote_python_path=str(remote_tool_pins["python"]["path"]),
        ),
    ]
    remote_executable = _mapping(
        remote_receipt["executable"],
        field="campaign remote probe executable",
    )
    if (
        len(remote_argv) != len(expected_remote_tail) + 1
        or remote_argv[0] != local_tool_pins["ssh"]["path"]
        or remote_executable["sha256"] != local_tool_pins["ssh"]["sha256"]
        or remote_argv[1:] != expected_remote_tail
        or remote_receipt["returncode"] != 0
        or remote_receipt["stderr"] != qualification.SSH_VERBOSE_LOG_REDACTION_MARKER
    ):
        raise _error("campaign remote filesystem command drifted")
    _sha256(
        measurement["ssh_verbose_stderr_sha256"],
        field="campaign SSH verbose log SHA256",
    )
    remote = _strict_json_text(
        str(remote_receipt["stdout"]),
        field="campaign remote filesystem output",
    )
    _exact_key_set(
        remote,
        {
            "schema_version",
            "target_directory",
            "remote_hostname",
            "remote_boot_id",
            "remote_python",
            "target_filesystem",
            "filesystem_measurement_contract",
        },
        field="campaign remote filesystem output",
    )
    remote_hostname = _host_identifier(
        remote["remote_hostname"],
        field="campaign remote hostname",
    )
    remote_boot_id = remote["remote_boot_id"]
    if (
        remote["schema_version"] != qualification.REMOTE_STORAGE_PROBE_SCHEMA_VERSION
        or remote["target_directory"] != target_directory
        or remote["filesystem_measurement_contract"]
        != "remote_python_open_nofollow_directory_fstatvfs_fstat_identity_v1"
        or not isinstance(remote_boot_id, str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}",
            remote_boot_id,
        )
        is None
    ):
        raise _error("campaign remote filesystem identity drifted")
    remote_python = _mapping(
        remote["remote_python"],
        field="campaign remote Python",
    )
    _exact_key_set(
        remote_python,
        {"path", "sha256"},
        field="campaign remote Python",
    )
    if (
        not isinstance(remote_python["path"], str)
        or not Path(str(remote_python["path"])).is_absolute()
        or remote_python["path"] != remote_tool_pins["python"]["path"]
        or remote_python["sha256"] != remote_tool_pins["python"]["sha256"]
    ):
        raise _error("campaign remote Python path is malformed")
    _sha256(remote_python["sha256"], field="campaign remote Python SHA256")
    filesystem = _mapping(
        remote["target_filesystem"],
        field="campaign target filesystem",
    )
    _exact_key_set(
        filesystem,
        {"device", "inode", "fragment_size", "blocks", "available_blocks"},
        field="campaign target filesystem",
    )
    fragment_size = _positive_integer(
        filesystem["fragment_size"],
        field="filesystem fragment size",
    )
    blocks = _positive_integer(filesystem["blocks"], field="filesystem blocks")
    available_blocks = _integer(
        filesystem["available_blocks"],
        field="filesystem available blocks",
        minimum=0,
    )
    if available_blocks > blocks:
        raise _error("filesystem available blocks exceed capacity")
    rclone_config = _mapping(
        measurement["rclone_config"],
        field="campaign rclone config",
    )
    _exact_key_set(
        rclone_config,
        {"path", "sha256"},
        field="campaign rclone config",
    )
    rclone_config_path = rclone_config["path"]
    if (
        not isinstance(rclone_config_path, str)
        or not Path(rclone_config_path).is_absolute()
        or str(Path(rclone_config_path)) != rclone_config_path
    ):
        raise _error("campaign rclone config path is malformed")
    rclone_config_sha256 = _sha256(
        rclone_config["sha256"],
        field="campaign rclone config SHA256",
    )
    if dict(rclone_config) != dict(authority_rclone_config):
        raise _error("campaign rclone config differs from archive authority")
    drive_receipt = _validate_qualification_command_receipt(
        measurement["drive_receipt"],
        field="campaign Drive measurement",
    )
    drive_argv = list(_sequence(drive_receipt["argv"], field="campaign Drive argv"))
    drive_executable = _mapping(
        drive_receipt["executable"],
        field="campaign Drive executable",
    )
    if (
        len(drive_argv) != 6
        or drive_argv[0] != local_tool_pins["rclone"]["path"]
        or drive_executable["sha256"] != local_tool_pins["rclone"]["sha256"]
        or drive_argv[1:]
        != [
            "--config",
            rclone_config_path,
            "about",
            "gdrive:",
            "--json",
        ]
        or drive_receipt["returncode"] != 0
        or str(drive_receipt["stderr"]).strip()
    ):
        raise _error("campaign Drive measurement command failed")
    drive = _strict_json_text(
        str(drive_receipt["stdout"]),
        field="campaign Drive measurement output",
    )
    drive_free = _positive_integer(
        drive.get("free"),
        field="Google Drive free bytes",
    )
    capacity = blocks * fragment_size
    free = available_blocks * fragment_size
    required = max(100 * 1024**3, (capacity + 4) // 5)
    facts = {
        "checked_at_utc": measurement["checked_at_utc"],
        "measurement_location": measurement["measurement_location"],
        "target_ssh": target_ssh,
        "remote_hostname": remote_hostname,
        "remote_boot_id": remote_boot_id,
        "archive_authority_sha256": captures["archive_tool_authority"].sha256,
        "ssh_effective_config_sha256": ssh_config_sha256,
        "ssh_verbose_stderr_sha256": measurement["ssh_verbose_stderr_sha256"],
        "ssh_connection": dict(authority_connection),
        "rclone_config_path": rclone_config_path,
        "rclone_config_sha256": rclone_config_sha256,
        "google_drive_free_bytes": drive_free,
        "projected_active_storage_bytes": ACTIVE_CAMPAIGN_BUDGET_BYTES,
        "target_filesystem_path": target_directory,
        "target_filesystem_capacity_bytes": capacity,
        "target_filesystem_free_bytes": free,
        "required_target_free_bytes": required,
        "drive_measurement_command_sha256": stable_payload_digest(drive_receipt),
        "filesystem_measurement_command_sha256": stable_payload_digest(remote_receipt),
    }
    return _authority_reconstruction(manifest, facts)


def _reconstruct_output_lock_contention_from_captures(
    manifest: Mapping[str, object],
    captures: Mapping[str, _CapturedRegularFile],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    from evolution_sim.io.open_ecology_campaign_storage import canonical_json_bytes
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    if manifest.get("producer") != {
        "name": "produce_output_lock_contention_report",
        "contract": ("real_multiprocess_storage_lock_contention_identity_reacquire_v1"),
    }:
        raise _error("output lock raw producer identity drifted")
    if set(captures) != {"lock_identity", "lock_probe"}:
        raise _error("output lock raw evidence roles are incomplete")
    probe = _strict_json_from_captured_bytes(
        captures["lock_probe"],
        field="output lock probe",
    )
    _exact_key_set(
        probe,
        {
            "schema_version",
            "worker_program_sha256",
            "campaign_id",
            "source_git_sha",
            "lock_identity_sha256",
            "concurrent_contenders",
            "identity_drift_attempt",
            "post_release_reacquire",
        },
        field="output lock probe",
    )
    expected_campaign_id = f"phase-a-{str(preregistration['exact_digest'])[:24]}"
    source_commit = str(
        _mapping(preregistration["source"], field="output lock source")["commit"]
    )
    expected_identity = canonical_json_bytes(
        {
            "campaign_id": expected_campaign_id,
            "schema_version": "open_ecology_campaign_storage_lock_v1",
            "source_git_sha": source_commit,
        }
    )
    if (
        probe["schema_version"] != qualification.OUTPUT_LOCK_PROBE_SCHEMA_VERSION
        or probe["worker_program_sha256"] != qualification.lock_probe_code_sha256()
        or probe["campaign_id"] != expected_campaign_id
        or probe["source_git_sha"] != source_commit
        or probe["lock_identity_sha256"]
        != hashlib.sha256(expected_identity).hexdigest()
        or captures["lock_identity"].payload != expected_identity
    ):
        raise _error("output lock probe identity drifted")
    contenders = _sequence(
        probe["concurrent_contenders"],
        field="output lock contenders",
    )
    if len(contenders) != 2:
        raise _error("output lock probe did not run exactly two contenders")
    first = _validate_lock_worker_receipt(
        contenders[0],
        field="output lock first contender",
    )
    second = _validate_lock_worker_receipt(
        contenders[1],
        field="output lock second contender",
    )
    identity_attempt = _validate_lock_worker_receipt(
        probe["identity_drift_attempt"],
        field="output lock identity attempt",
    )
    reacquire = _validate_lock_worker_receipt(
        probe["post_release_reacquire"],
        field="output lock reacquire",
    )
    first_event = _mapping(first["event"], field="first lock event")
    second_event = _mapping(second["event"], field="second lock event")
    identity_event = _mapping(identity_attempt["event"], field="identity lock event")
    reacquire_event = _mapping(reacquire["event"], field="reacquire lock event")
    first_pid = _positive_integer(first_event.get("pid"), field="first lock PID")
    second_pid = _positive_integer(second_event.get("pid"), field="second lock PID")
    if (
        first["returncode"] != 0
        or first_event.get("outcome") != "admitted"
        or second["returncode"] != 3
        or second_event.get("outcome") != "rejected"
        or "locked by another process" not in str(second_event.get("error"))
        or first_pid == second_pid
        or _positive_integer(
            first_event.get("acquired_monotonic_ns"),
            field="first lock acquisition",
        )
        > _positive_integer(
            second_event.get("started_monotonic_ns"),
            field="second lock start",
        )
        or _positive_integer(
            first["observed_finished_monotonic_ns"],
            field="first lock finish",
        )
        <= _positive_integer(
            second_event.get("started_monotonic_ns"),
            field="second lock start",
        )
        or identity_attempt["returncode"] != 3
        or identity_event.get("outcome") != "rejected"
        or "identity drifted" not in str(identity_event.get("error"))
        or reacquire["returncode"] != 0
        or reacquire_event.get("outcome") != "admitted"
    ):
        raise _error("output lock contention transcript failed its causal contract")
    facts = {
        "concurrent_contender_count": 2,
        "admitted_writer_count": 1,
        "contention_rejection_count": 1,
        "identity_drift_rejection_count": 1,
        "lock_release_reacquire_count": 1,
    }
    return _authority_reconstruction(manifest, facts)


def _authority_reconstruction(
    manifest: Mapping[str, object],
    facts: Mapping[str, object],
) -> dict[str, object]:
    return {
        "evidence_kind": manifest["evidence_kind"],
        "campaign_digest": manifest["campaign_digest"],
        "configuration_sha256": manifest["configuration_sha256"],
        "source": manifest["source"],
        "facts": dict(facts),
    }


def _validate_qualification_command_receipt(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    receipt = _mapping(value, field=field)
    _exact_key_set(
        receipt,
        {
            "schema_version",
            "argv",
            "executable",
            "started_at_utc",
            "finished_at_utc",
            "elapsed_ns",
            "returncode",
            "stdout",
            "stderr",
        },
        field=field,
    )
    if receipt["schema_version"] != qualification.COMMAND_RECEIPT_SCHEMA_VERSION:
        raise _error(f"{field} schema drifted")
    argv = _sequence(receipt["argv"], field=f"{field}.argv")
    if (
        not argv
        or any(not isinstance(argument, str) or not argument for argument in argv)
        or not Path(str(argv[0])).is_absolute()
    ):
        raise _error(f"{field} argv is malformed")
    executable = _mapping(receipt["executable"], field=f"{field}.executable")
    _exact_key_set(
        executable,
        {"device", "inode", "size", "sha256"},
        field=f"{field}.executable",
    )
    _integer(executable["device"], field=f"{field} executable device", minimum=0)
    _positive_integer(executable["inode"], field=f"{field} executable inode")
    _positive_integer(executable["size"], field=f"{field} executable size")
    _sha256(executable["sha256"], field=f"{field} executable SHA256")
    started = _parse_utc(receipt["started_at_utc"], field=f"{field}.started_at_utc")
    finished = _parse_utc(
        receipt["finished_at_utc"],
        field=f"{field}.finished_at_utc",
    )
    if finished < started:
        raise _error(f"{field} finished before it started")
    _integer(receipt["elapsed_ns"], field=f"{field}.elapsed_ns", minimum=0)
    _integer(receipt["returncode"], field=f"{field}.returncode", minimum=0)
    if not isinstance(receipt["stdout"], str) or not isinstance(receipt["stderr"], str):
        raise _error(f"{field} output must be UTF-8 text")
    return receipt


def _validate_authenticated_github_https_receipt(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification

    receipt = _mapping(value, field=field)
    _exact_key_set(
        receipt,
        {
            "schema_version",
            "request",
            "response",
            "started_at_utc",
            "finished_at_utc",
            "elapsed_ns",
        },
        field=field,
    )
    if (
        receipt["schema_version"]
        != qualification.AUTHENTICATED_HTTPS_RECEIPT_SCHEMA_VERSION
    ):
        raise _error(f"{field} schema drifted")
    request = _mapping(receipt["request"], field=f"{field}.request")
    expected_request = {
        "authentication": "bearer_one_shot_in_memory_redacted",
        "credential_in_argv": False,
        "credential_in_environment": False,
        "credential_in_receipt": False,
        "credential_on_disk": False,
        "host": "api.github.com",
        "method": "GET",
        "response_byte_ceiling": qualification.COMMAND_OUTPUT_LIMIT_BYTES,
        "tls_context": "python_default_verified_context",
    }
    _exact_key_set(
        request,
        {*expected_request, "path"},
        field=f"{field}.request",
    )
    if (
        {key: request[key] for key in expected_request} != expected_request
        or not isinstance(request["path"], str)
        or not str(request["path"]).startswith("/repos/")
        or any(
            key.lower() in json.dumps(receipt, sort_keys=True).lower()
            for key in ("ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_")
        )
    ):
        raise _error(f"{field} credential-redaction contract drifted")
    response = _mapping(receipt["response"], field=f"{field}.response")
    _exact_key_set(
        response,
        {"body", "body_sha256", "reason", "status"},
        field=f"{field}.response",
    )
    if (
        not isinstance(response["body"], str)
        or not isinstance(response["reason"], str)
        or not response["reason"]
        or hashlib.sha256(str(response["body"]).encode("utf-8")).hexdigest()
        != _sha256(
            response["body_sha256"],
            field=f"{field}.response.body_sha256",
        )
    ):
        raise _error(f"{field} response binding drifted")
    _integer(response["status"], field=f"{field}.response.status", minimum=100)
    started = _parse_utc(receipt["started_at_utc"], field=f"{field}.started_at_utc")
    finished = _parse_utc(
        receipt["finished_at_utc"],
        field=f"{field}.finished_at_utc",
    )
    if finished < started:
        raise _error(f"{field} finished before it started")
    _integer(receipt["elapsed_ns"], field=f"{field}.elapsed_ns", minimum=0)
    return receipt


def _archive_authority_file_pin(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    pin = _mapping(value, field=field)
    _exact_key_set(pin, {"path", "sha256"}, field=field)
    path = pin["path"]
    if (
        not isinstance(path, str)
        or not path
        or not Path(path).is_absolute()
        or str(Path(path)) != path
    ):
        raise _error(f"{field} path is malformed")
    _sha256(pin["sha256"], field=f"{field} SHA256")
    return pin


def _validate_benchmark_command_receipt(
    value: object,
    *,
    mode: str,
    source_bytes: bytes,
) -> Mapping[str, object]:
    from evolution_sim.mind import open_ecology_phase_a_qualification as qualification
    from evolution_sim.mind.open_ecology_seed_registry import (
        OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX,
        OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        OPEN_ECOLOGY_SEED_REGISTRY,
    )

    receipt = _mapping(value, field=f"{mode} benchmark receipt")
    _exact_key_set(
        receipt,
        {
            "schema_version",
            "argv",
            "started_at_utc",
            "finished_at_utc",
            "elapsed_ns",
            "returncode",
            "stdout_sha256",
            "stdout_byte_length",
            "stderr",
        },
        field=f"{mode} benchmark receipt",
    )
    argv = list(_sequence(receipt["argv"], field=f"{mode} benchmark argv"))
    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
    required_pairs = {
        "--worker-counts": "1,2,4,8,16",
        "--repeats": "3",
        "--updates": "1",
        "--worlds-per-update": "16",
        "--rollout-ticks": "128",
        "--scenarios": "broad",
        "--device": "cuda",
        "--input-contract": "tokenized",
        "--genome-conditioning": "actor_film_v1",
        "--genome-population-mode": mode,
        "--genome-stream-seed": str(
            benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX]
        ),
        "--encoder-size": "256",
        "--hidden-size": "256",
        "--recurrent-layers": "1",
        "--learner-seed": str(
            benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]
        ),
        "--update-epochs": "4",
        "--sequence-minibatch-size": "16",
        "--tbptt-steps": "128",
        "--burn-in-steps": "16",
        "--fixed-batch-capacity": "320",
    }
    if (
        receipt["schema_version"] != qualification.COMMAND_RECEIPT_SCHEMA_VERSION
        or len(argv) < 3
        or Path(str(argv[1])).name != "benchmark_recurrent_pipeline.py"
        or "--preregistered-gate-member" not in argv
        or any(
            option not in argv
            or argv.index(option) + 1 >= len(argv)
            or argv[argv.index(option) + 1] != expected
            for option, expected in required_pairs.items()
        )
        or receipt["returncode"] != 0
        or str(receipt["stderr"]).strip()
        or receipt["stdout_sha256"] != hashlib.sha256(source_bytes).hexdigest()
        or receipt["stdout_byte_length"] != len(source_bytes)
    ):
        raise _error(f"{mode} benchmark command receipt drifted")
    started = _parse_utc(
        receipt["started_at_utc"],
        field=f"{mode} benchmark started",
    )
    finished = _parse_utc(
        receipt["finished_at_utc"],
        field=f"{mode} benchmark finished",
    )
    if finished < started:
        raise _error(f"{mode} benchmark finished before it started")
    _positive_integer(receipt["elapsed_ns"], field=f"{mode} benchmark elapsed")
    return receipt


def _validate_lock_worker_receipt(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    receipt = _mapping(value, field=field)
    _exact_key_set(
        receipt,
        {"returncode", "stderr", "observed_finished_monotonic_ns", "event"},
        field=field,
    )
    _integer(receipt["returncode"], field=f"{field} returncode", minimum=0)
    _positive_integer(
        receipt["observed_finished_monotonic_ns"],
        field=f"{field} finish time",
    )
    if not isinstance(receipt["stderr"], str) or str(receipt["stderr"]).strip():
        raise _error(f"{field} emitted stderr")
    event = _mapping(receipt["event"], field=f"{field} event")
    outcome = event.get("outcome")
    expected_keys = (
        {"outcome", "pid", "started_monotonic_ns", "acquired_monotonic_ns"}
        if outcome == "admitted"
        else {"outcome", "pid", "started_monotonic_ns", "error"}
    )
    _exact_key_set(event, expected_keys, field=f"{field} event")
    _positive_integer(event["pid"], field=f"{field} PID")
    _positive_integer(
        event["started_monotonic_ns"],
        field=f"{field} start time",
    )
    if outcome == "admitted":
        acquired = _positive_integer(
            event["acquired_monotonic_ns"],
            field=f"{field} acquisition time",
        )
        if acquired < int(event["started_monotonic_ns"]):
            raise _error(f"{field} acquired before it started")
    elif outcome == "rejected":
        if not isinstance(event["error"], str) or not event["error"]:
            raise _error(f"{field} rejection reason is malformed")
    else:
        raise _error(f"{field} outcome drifted")
    return receipt


def _validate_nonoptional_torch_ci_workflow(workflow: str) -> None:
    match = re.search(
        r"(?ms)^  mind-recurrent-cpu:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    if match is None:
        raise _error("D10 CI workflow omits mind-recurrent-cpu")
    body = match.group("body")
    required = (
        "torch==2.11.0",
        "Run recurrent Mind contracts with Torch enabled",
        "Run open-ecology contracts with Torch enabled",
        "python -m unittest discover -s python/tests -p 'test_recurrent_*.py' -v",
        "python -m unittest discover -s python/tests -p 'test_open_ecology_*.py' -v",
    )
    if re.search(r"(?m)^    if:", body) is not None or any(
        value not in body for value in required
    ):
        raise _error("D10 Torch CI lane is optional or incomplete")


def _discover_torch_gated_tests(source_bytes: bytes) -> tuple[str, ...]:
    source = _utf8_text(source_bytes, field="D10 Torch test source")
    try:
        tree = ast.parse(source, filename="python/tests/test_mind_v1.py")
    except SyntaxError as error:
        raise _error(f"D10 Torch test source is invalid: {error}")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "MindV1Tests":
            continue
        names: list[str] = []
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith(
                "test_"
            ):
                continue
            decorators = [
                ast.get_source_segment(source, dec) or "" for dec in item.decorator_list
            ]
            if any(
                "skipUnless" in decorator
                and "find_spec" in decorator
                and ('"torch"' in decorator or "'torch'" in decorator)
                for decorator in decorators
            ):
                names.append(item.name)
        if not names:
            raise _error("D10 Torch source has no Torch-gated tests")
        return tuple(names)
    raise _error("D10 Torch source omits MindV1Tests")


def _require_capture_matches_live_file(
    capture: _CapturedRegularFile,
    path: Path,
    *,
    field: str,
) -> None:
    parent = path.parent.resolve()
    parent_fd = _open_real_directory(parent, field=f"{field} live parent")
    try:
        live = _capture_regular_file_at(
            parent_fd,
            parent,
            path.name,
            field=f"{field} live source",
            maximum_bytes=_MAX_RAW_EVIDENCE_FILE_BYTES,
        )
    finally:
        os.close(parent_fd)
    if capture.payload != live.payload:
        raise _error(f"{field} differs from the exact live checkout")


def _strict_json_text(value: str, *, field: str) -> dict[str, object]:
    contract = _contract()
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=contract._reject_duplicate_keys,
            parse_constant=contract._reject_nonfinite_json,
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise contract.OpenEcologyPhaseAError(
            f"{field} is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise contract.OpenEcologyPhaseAError(f"{field} root must be an object")
    return parsed


def _utf8_text(value: bytes, *, field: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _error(f"{field} is not UTF-8: {error}")


def _parse_flexible_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise _error(f"{field} must be UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise _error(f"{field} is malformed") from error
    if parsed.tzinfo != timezone.utc:
        raise _error(f"{field} must be UTC")
    return parsed


def _host_class(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
    ):
        raise _error("host class must be one bounded identifier")
    return value


def _host_identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", value) is None
    ):
        raise _error(f"{field} must be one bounded host identifier")
    return value


def _require_expected_rejection(
    operation: Callable[[], object],
    *,
    expected_message: str,
    field: str,
) -> int:
    contract = _contract()
    try:
        operation()
    except contract.OpenEcologyPhaseAError as error:
        if expected_message not in str(error):
            raise contract.OpenEcologyPhaseAError(
                f"{field} rejected for the wrong reason: {error}"
            ) from error
        return 1
    raise contract.OpenEcologyPhaseAError(f"{field} did not fail closed")


def _raw_bundle_file_inventory(bundle_root: Path) -> set[str]:
    root = bundle_root.resolve()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as error:
        raise _error(f"raw-evidence bundle root cannot be pinned: {error}")
    try:
        return _raw_bundle_file_inventory_at(root_fd)
    finally:
        os.close(root_fd)


def _raw_bundle_file_inventory_at(root_fd: int) -> set[str]:
    """Inventory raw/ through one pinned directory descriptor."""

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        raw_fd = os.open("raw", directory_flags, dir_fd=root_fd)
    except OSError as error:
        raise _error(f"raw-evidence bundle requires one real raw/ directory: {error}")
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory_fd: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as error:
            raise _error(f"raw-evidence directory could not be inventoried: {error}")
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise _error("raw-evidence directory contains an unsafe name")
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise _error(f"raw-evidence entry could not be inspected: {error}")
            relative = f"{prefix}/{name}"
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as error:
                    raise _error(
                        f"raw-evidence directory changed during inventory: {error}"
                    )
                try:
                    child_metadata = os.fstat(child_fd)
                    if (
                        child_metadata.st_dev != metadata.st_dev
                        or child_metadata.st_ino != metadata.st_ino
                    ):
                        raise _error("raw-evidence directory changed during inventory")
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise _error(
                    "raw-evidence bundle entries must be regular non-hardlinked files"
                )
            files.add(relative)

    try:
        visit(raw_fd, "raw")
    finally:
        os.close(raw_fd)
    expected_directories: set[str] = set()
    for relative in files:
        parent = Path(relative).parent
        while parent.as_posix() != "raw":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != expected_directories:
        raise _error("raw-evidence bundle contains surplus directories")
    return files


def _canonical_json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _write_new_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    return _write_new_bytes(Path(path), _canonical_json_file_bytes(payload))


def _write_new_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    if (
        destination.exists()
        or destination.is_symlink()
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
    ):
        raise _error("raw-evidence producer writes only new files in real directories")
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise _error(f"failed to write raw-evidence file: {error}") from error
    if (
        not destination.is_file()
        or destination.is_symlink()
        or destination.stat().st_nlink != 1
    ):
        raise _error("raw-evidence producer did not create one regular file")
    return destination


def _surface_mapping(value: object, *, field: str) -> Mapping[str, object]:
    parsed = _mapping(value, field=field)
    _exact_key_set(parsed, set(_SURFACES), field=field)
    return parsed


def _open_real_directory(path: Path, *, field: str) -> int:
    """Open an absolute directory path component-by-component without symlinks."""

    absolute = path.absolute()
    if not absolute.is_absolute():
        raise _error(f"{field} must be an absolute directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise _error(f"{field} must be one real directory")
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise _error(f"{field} must be one real directory: {error}") from error
    except BaseException:
        os.close(descriptor)
        raise


def _capture_regular_file_at(
    root_fd: int,
    root_path: Path,
    relative: str,
    *,
    field: str,
    maximum_bytes: int,
) -> _CapturedRegularFile:
    """Read one fd-pinned regular file once and validate stable inode metadata."""

    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise _error(f"{field} path is unsafe")
    parent_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
        for part in candidate.parts[:-1]:
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        file_fd = os.open(
            candidate.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise _error(
                f"{field} must be one bounded regular file with exactly one link"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                raise _error(f"{field} ended before its declared byte length")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            raise _error(f"{field} grew while it was being read")
        after = os.fstat(file_fd)
        path_after = os.stat(
            candidate.parts[-1],
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(after, name) for name in identity_fields
        ) or any(
            getattr(after, name) != getattr(path_after, name)
            for name in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size")
        ):
            raise _error(f"{field} changed identity or bytes while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise _error(f"{field} byte length changed while being read")
        return _CapturedRegularFile(
            path=root_path / candidate,
            relative_path=candidate.as_posix(),
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_length=len(payload),
        )
    except OSError as error:
        raise _error(f"{field} could not be read safely: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _strict_json_from_captured_bytes(
    captured: _CapturedRegularFile,
    *,
    field: str,
) -> dict[str, object]:
    contract = _contract()
    try:
        payload = json.loads(
            captured.payload.decode("utf-8"),
            object_pairs_hook=contract._reject_duplicate_keys,
            parse_constant=contract._reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise contract.OpenEcologyPhaseAError(
            f"{field} is not strict UTF-8 JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise contract.OpenEcologyPhaseAError(
            f"{field} strict JSON root must be an object"
        )
    return payload


def _captured_file_reference(
    captured: _CapturedRegularFile,
) -> dict[str, object]:
    return {
        "relative_path": captured.relative_path,
        "sha256": captured.sha256,
        "byte_length": captured.byte_length,
    }


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


def _count_nonfinite_numbers(value: object) -> int:
    """Count non-finite numeric observations without trusting declared facts."""

    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return 0
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return int(value != value or value in {float("inf"), float("-inf")})
    if isinstance(value, Mapping):
        return sum(_count_nonfinite_numbers(item) for item in value.values())
    if isinstance(value, Sequence):
        return sum(_count_nonfinite_numbers(item) for item in value)
    return 0


def _positive_number(value: object, *, field: str) -> float:
    parsed = _number(value, field=field)
    if parsed <= 0:
        raise _error(f"{field} must be positive")
    return parsed


def _sha256(value: object, *, field: str) -> str:
    return require_lowercase_sha256(value, field=field)


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
    return OpenEcologyPhaseAError(message)


def _contract() -> "phase_a_contract":
    from evolution_sim.mind import open_ecology_phase_a

    return open_ecology_phase_a


__all__ = [
    "DEPENDENCY_EVIDENCE_KINDS",
    "OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_LEGACY_READINESS_REPORT_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_LAUNCH_READINESS_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION",
    "OPEN_ECOLOGY_RAW_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "OPERATIONAL_EVIDENCE_KINDS",
    "PROOF_PRODUCERS",
    "PROOF_PRODUCER_AVAILABILITY",
    "REPORT_AUTHORITY_VERIFIERS",
    "ProofProducer",
    "ReportAuthorityVerifier",
    "build_open_ecology_raw_evidence_manifest",
    "build_evidence_index",
    "build_launch_authorization",
    "launch_readiness",
    "live_verify_campaign_storage_capacity_report",
    "live_verify_exact_sha_phase_a_training_and_torch_ci_report",
    "live_verify_output_lock_contention_report",
    "live_verify_phase_a_training_throughput_report",
    "produce_campaign_storage_capacity_report",
    "produce_capture_noninterference_reexecution_report",
    "produce_cross_surface_and_self_echo_report",
    "produce_critic_gradient_and_density_schedule_report",
    "produce_exact_sha_phase_a_training_and_torch_ci_report",
    "produce_fixed_batch_equivalence_and_speed_report",
    "produce_output_lock_contention_report",
    "produce_phase_a_training_throughput_report",
    "produce_preregistration_roundtrip_fail_closed_report",
    "produce_runtime_genome_and_action_source_report",
    "validate_launch_authorization",
    "verify_campaign_storage_capacity_report",
    "verify_cross_surface_and_self_echo_report",
    "verify_critic_gradient_and_density_schedule_report",
    "verify_exact_sha_phase_a_training_and_torch_ci_report",
    "verify_fixed_batch_equivalence_and_speed_report",
    "verify_output_lock_contention_report",
    "verify_phase_a_training_throughput_report",
    "verify_preregistration_roundtrip_fail_closed_report",
    "verify_runtime_genome_and_action_source_report",
]
