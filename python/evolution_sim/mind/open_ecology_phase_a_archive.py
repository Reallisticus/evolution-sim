"""Close and archive one complete Phase-A terminal matrix.

The active training root is never an archive input.  This module verifies the
completed live-guardian transcript and all sixteen restart/terminal/artifact
chains, holds every run lock, descriptor-copies the active matrix and its
authority tree into a new external bundle, then delegates immutability to the
existing campaign storage seal.  The Drive step reuses the existing
exact-source archive authority and uploader; no pruning or deletion exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Final

from evolution_sim.io.open_ecology_archive_authority import (
    load_archive_tool_authority,
)
from evolution_sim.io.open_ecology_campaign_storage import (
    CampaignStorageLimits,
    RECEIPT_SCHEMA_VERSION,
    canonical_json_bytes,
    ensure_real_directory_tree,
    seal_closed_bundle,
    sha256_bytes,
    write_verified_receipt,
)
from evolution_sim.io.open_ecology_runtime_venv_authority import (
    RuntimeVenvAuthorityError,
    load_runtime_venv_authority,
)
from evolution_sim.mind.open_ecology_phase_a import (
    OPEN_ECOLOGY_PHASE_A_CELL_ORDER,
    OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION,
    OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    _require_live_source,
    build_verified_phase_a_selection_request,
    phase_a_run_id,
    validate_open_ecology_phase_a_preregistration,
)
from evolution_sim.mind.open_ecology_phase_a_guardian import (
    TWO_PARTY_TRANSCRIPT_SCHEMA_VERSION,
)
from evolution_sim.mind.open_ecology_phase_a_readiness import (
    OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION,
    OPEN_ECOLOGY_RAW_EVIDENCE_MANIFEST_SCHEMA_VERSION,
    validate_launch_authorization,
    verify_campaign_storage_capacity_report,
    verify_exact_sha_phase_a_training_and_torch_ci_report,
    verify_output_lock_contention_report,
    verify_phase_a_training_throughput_report,
)
from evolution_sim.mind.provenance import stable_payload_digest


PHASE_A_TERMINAL_MATRIX_MANIFEST_SCHEMA_VERSION: Final = (
    "mind_v3_open_ecology_phase_a_terminal_matrix_bundle_v1"
)
PHASE_A_TERMINAL_MATRIX_CLOSURE_SCHEMA_VERSION: Final = (
    "mind_v3_open_ecology_phase_a_terminal_matrix_closure_v1"
)
PHASE_A_TERMINAL_MATRIX_MANIFEST_NAME: Final = "phase-a-terminal-matrix.json"
_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_COPY_BLOCK_BYTES: Final = 8 * 1024 * 1024
_IDENTIFIER_CHARACTERS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_IDENTIFIER_INITIAL_CHARACTERS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


class OpenEcologyPhaseAArchiveError(RuntimeError):
    """The Phase-A matrix cannot be closed or archived authoritatively."""


@dataclass(frozen=True, slots=True)
class _CopiedTree:
    entry_count: int
    file_count: int
    directory_count: int
    total_file_bytes: int
    inventory_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "directory_count": self.directory_count,
            "entry_count": self.entry_count,
            "file_count": self.file_count,
            "inventory_sha256": self.inventory_sha256,
            "total_file_bytes": self.total_file_bytes,
        }


def close_phase_a_terminal_matrix(
    *,
    preregistration_path: Path,
    launch_authorization_path: Path,
    guardian_transcript_path: Path,
    runtime_venv_authority_path: Path,
    active_output_root: Path,
    authority_root: Path,
    closed_bundle_parent: Path,
    bundle_id: str,
    closure_receipt_path: Path,
) -> dict[str, object]:
    """Materialize and seal one complete matrix without replacing any path."""

    parsed_bundle_id = _identifier(bundle_id, field="bundle ID")
    preregistration_file = _regular_file(
        preregistration_path,
        field="preregistration",
    )
    authorization_file = _regular_file(
        launch_authorization_path,
        field="launch authorization",
    )
    transcript_file = _regular_file(
        guardian_transcript_path,
        field="guardian transcript",
    )
    runtime_venv_authority_file = _regular_file(
        runtime_venv_authority_path,
        field="runtime venv authority",
    )
    authority = _canonical_existing_directory(authority_root, field="authority root")
    for field, path in (
        ("preregistration", preregistration_file),
        ("launch authorization", authorization_file),
        ("guardian transcript", transcript_file),
        ("runtime venv authority", runtime_venv_authority_file),
    ):
        if not _is_within(path, authority):
            raise OpenEcologyPhaseAArchiveError(
                f"{field} must be inside the copied authority root"
            )
    active = _canonical_existing_directory(
        active_output_root,
        field="active Phase-A output root",
    )
    destination_parent = ensure_real_directory_tree(
        closed_bundle_parent,
        field="closed Phase-A bundle parent",
    )
    destination = destination_parent / parsed_bundle_id
    receipt = _new_output_path(closure_receipt_path, field="closure receipt")
    _require_pairwise_distinct_trees(
        (
            ("active Phase-A output root", active),
            ("authority root", authority),
            ("closed bundle", destination),
        )
    )
    if destination.exists() or destination.is_symlink():
        raise OpenEcologyPhaseAArchiveError("closed Phase-A bundle path already exists")
    if (
        _is_within(receipt, active)
        or _is_within(receipt, authority)
        or _is_within(receipt, destination)
    ):
        raise OpenEcologyPhaseAArchiveError(
            "closure receipt must remain outside active, authority, and closed trees"
        )

    preregistration = _load_strict_json(preregistration_file)
    validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source(preregistration)
    (
        authorization,
        evidence_index_digest,
        authorization_time,
        report_paths,
    ) = _validate_launch_binding(
        authorization_file,
        preregistration=preregistration,
    )
    storage_ssh_target = _launch_storage_ssh_target(authorization)
    transcript = _load_strict_json(transcript_file)
    runtime_venv_authority_sha256 = _sha256_file(runtime_venv_authority_file)
    source = _mapping(preregistration.get("source"), field="source")
    _validate_runtime_authority(
        runtime_venv_authority_file,
        file_sha256=runtime_venv_authority_sha256,
        preregistration=preregistration,
        launch_authorization=authorization,
    )
    authority_files, authority_directories = _build_authority_allowlist(
        authority,
        explicit_files=(
            preregistration_file,
            authorization_file,
            transcript_file,
            runtime_venv_authority_file,
        ),
    )
    _require_exact_authority_tree(
        authority,
        expected_files=authority_files,
        expected_directories=authority_directories,
    )
    terminal_rows = _validate_guardian_and_terminal_matrix(
        transcript,
        preregistration=preregistration,
        launch_authorization=authorization,
        evidence_index_digest=evidence_index_digest,
        runtime_venv_authority_sha256=runtime_venv_authority_sha256,
        active_output_root=active,
    )
    _validate_transcript_verifier_digests(
        transcript,
        preregistration=preregistration,
        authorization_time=authorization_time,
        report_paths=report_paths,
    )
    campaign_id = f"phase-a-{str(preregistration['exact_digest'])[:24]}"

    lock_paths = tuple(
        active
        / phase_a_run_id(cell_id=cell_id, learner_index=learner_index)
        / ".run.lock"
        for cell_id, learner_index in _canonical_matrix()
    )
    with ExitStack() as locks:
        for path in lock_paths:
            locks.enter_context(_held_run_lock(path))
        _require_live_source(preregistration)
        terminal_rows = _validate_guardian_and_terminal_matrix(
            transcript,
            preregistration=preregistration,
            launch_authorization=authorization,
            evidence_index_digest=evidence_index_digest,
            runtime_venv_authority_sha256=runtime_venv_authority_sha256,
            active_output_root=active,
        )
        os.mkdir(destination, mode=0o700)
        _fsync_directory(destination_parent)
        training = destination / "training"
        copied_authority = destination / "authority"
        os.mkdir(training, mode=0o700)
        os.mkdir(copied_authority, mode=0o700)
        training_copy = _copy_tree_descriptor_safe(active, training)
        authority_copy = _copy_tree_descriptor_safe(
            authority,
            copied_authority,
            expected_files=authority_files,
            expected_directories=authority_directories,
        )

        copied_preregistration = copied_authority / preregistration_file.relative_to(
            authority
        )
        copied_authorization = copied_authority / authorization_file.relative_to(
            authority
        )
        copied_transcript = copied_authority / transcript_file.relative_to(authority)
        copied_runtime_venv_authority = (
            copied_authority / runtime_venv_authority_file.relative_to(authority)
        )
        copied_preregistration_payload = _load_strict_json(copied_preregistration)
        validate_open_ecology_phase_a_preregistration(copied_preregistration_payload)
        if copied_preregistration_payload != preregistration:
            raise OpenEcologyPhaseAArchiveError(
                "copied preregistration differs from the validated authority"
            )
        (
            copied_authorization_payload,
            copied_index_digest,
            copied_authorization_time,
            copied_report_paths,
        ) = _validate_launch_binding(
            copied_authorization,
            preregistration=copied_preregistration_payload,
        )
        if (
            copied_authorization_payload != authorization
            or copied_index_digest != evidence_index_digest
        ):
            raise OpenEcologyPhaseAArchiveError(
                "copied launch authority differs from the validated authority"
            )
        copied_transcript_payload = _load_strict_json(copied_transcript)
        if copied_transcript_payload != transcript:
            raise OpenEcologyPhaseAArchiveError(
                "copied guardian transcript differs from the validated transcript"
            )
        copied_runtime_venv_authority_sha256 = _sha256_file(
            copied_runtime_venv_authority
        )
        _validate_runtime_authority(
            copied_runtime_venv_authority,
            file_sha256=copied_runtime_venv_authority_sha256,
            preregistration=copied_preregistration_payload,
            launch_authorization=copied_authorization_payload,
        )
        copied_terminal_rows = _validate_guardian_and_terminal_matrix(
            copied_transcript_payload,
            preregistration=copied_preregistration_payload,
            launch_authorization=copied_authorization_payload,
            evidence_index_digest=copied_index_digest,
            runtime_venv_authority_sha256=(copied_runtime_venv_authority_sha256),
            active_output_root=training,
        )
        _validate_transcript_verifier_digests(
            copied_transcript_payload,
            preregistration=copied_preregistration_payload,
            authorization_time=copied_authorization_time,
            report_paths=copied_report_paths,
        )
        if copied_terminal_rows != terminal_rows:
            raise OpenEcologyPhaseAArchiveError(
                "copied terminal matrix differs from the locked active matrix"
            )

        materialization: dict[str, object] = {
            "schema_version": PHASE_A_TERMINAL_MATRIX_MANIFEST_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "bundle_id": parsed_bundle_id,
            "source": {
                "git_sha": source["commit"],
                "manifest_sha256": source["manifest_sha256"],
            },
            "authority": {
                "archive_tool_authority_sha256": source[
                    "archive_tool_authority_sha256"
                ],
                "preregistration_digest": preregistration["exact_digest"],
                "evidence_index_digest": evidence_index_digest,
                "launch_authorization_digest": authorization["exact_digest"],
                "guardian_transcript_digest": transcript["exact_digest"],
                "guardian_transcript_file_sha256": _sha256_file(copied_transcript),
                "runtime_venv_authority_sha256": (runtime_venv_authority_sha256),
                "ssh_target": storage_ssh_target,
            },
            "terminal_matrix": terminal_rows,
            "copies": {
                "training": training_copy.as_dict(),
                "authority": authority_copy.as_dict(),
            },
            "materialization_contract": {
                "source_run_locks_held": True,
                "descriptor_relative_no_follow_copy": True,
                "destination_created_exclusively": True,
                "source_deleted": False,
                "source_pruned": False,
                "path_replaced": False,
            },
            "lifecycle": {
                "phase_a_training_complete": True,
                "phase_a_selection_authorized": False,
                "phase_b_authorized": False,
                "runtime_integration_authorized": False,
                "promotion_authorized": False,
            },
        }
        materialization["exact_digest"] = _payload_digest(materialization)
        write_verified_receipt(
            destination / PHASE_A_TERMINAL_MATRIX_MANIFEST_NAME,
            materialization,
        )
        _fsync_directory(destination)
        snapshot = seal_closed_bundle(
            active,
            destination,
            campaign_id=campaign_id,
            bundle_id=parsed_bundle_id,
            source_git_sha=str(source["commit"]),
            source_manifest_sha256=str(source["manifest_sha256"]),
            limits=CampaignStorageLimits(),
        )
        _require_live_source(preregistration)

    closure: dict[str, object] = {
        "schema_version": PHASE_A_TERMINAL_MATRIX_CLOSURE_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "bundle_id": parsed_bundle_id,
        "active_campaign_root": str(active),
        "closed_bundle_root": str(snapshot.root),
        "source": {
            "git_sha": source["commit"],
            "manifest_sha256": source["manifest_sha256"],
        },
        "authority": {
            "archive_tool_authority_sha256": source["archive_tool_authority_sha256"],
            "preregistration_digest": preregistration["exact_digest"],
            "evidence_index_digest": evidence_index_digest,
            "launch_authorization_digest": authorization["exact_digest"],
            "guardian_transcript_digest": transcript["exact_digest"],
            "materialization_manifest_digest": materialization["exact_digest"],
            "runtime_venv_authority_sha256": runtime_venv_authority_sha256,
            "ssh_target": storage_ssh_target,
        },
        "terminal_matrix": terminal_rows,
        "closed_bundle": {
            "marker_sha256": snapshot.marker_sha256,
            "entry_count": len(snapshot.scan.entries),
            "file_count": snapshot.scan.file_count,
            "directory_count": snapshot.scan.directory_count,
            "total_file_bytes": snapshot.scan.total_file_bytes,
        },
        "drive_archive_route": {
            "destination_prefix": (
                "gdrive:evolution-sim-backups/archives/open-ecology/"
                f"{campaign_id}/{parsed_bundle_id}/"
            ),
            "existing_verified_uploader_required": True,
            "local_payload_staged": False,
            "source_deletion_authorized": False,
            "source_pruning_authorized": False,
        },
    }
    closure["exact_digest"] = _payload_digest(closure)
    write_verified_receipt(receipt, closure)
    return closure


def archive_phase_a_terminal_matrix(
    *,
    closure_receipt_path: Path,
    expected_closure_receipt_sha256: str,
    archive_tool_authority_path: Path,
    expected_archive_tool_authority_sha256: str,
    remote_staging_directory: Path,
    drive_receipt_path: Path,
) -> dict[str, object]:
    """Delegate one sealed Phase-A bundle to the verified Drive uploader."""

    from scripts.archive_open_ecology_campaign import (
        RemoteArchiveOptions,
        execute_remote_archive,
    )

    closure = validate_phase_a_terminal_matrix_closure(
        closure_receipt_path,
        expected_file_sha256=expected_closure_receipt_sha256,
    )
    closure_authority = _mapping(
        closure.get("authority"),
        field="closure authority",
    )
    expected_archive_authority = _sha256(
        expected_archive_tool_authority_sha256,
        field="expected archive-tool authority",
    )
    if (
        closure_authority.get("archive_tool_authority_sha256")
        != expected_archive_authority
    ):
        raise OpenEcologyPhaseAArchiveError(
            "closure and external archive-tool authority digest differ"
        )
    archive_authority = load_archive_tool_authority(
        archive_tool_authority_path,
        expected_sha256=expected_archive_authority,
    )
    source = _mapping(closure.get("source"), field="closure source")
    closed_bundle = _mapping(
        closure.get("closed_bundle"),
        field="closure closed bundle",
    )
    if (
        archive_authority.authority_sha256 != expected_archive_authority
        or archive_authority.ssh_target != closure_authority.get("ssh_target")
        or archive_authority.source_git_sha != source.get("git_sha")
        or archive_authority.source_manifest_sha256 != source.get("manifest_sha256")
    ):
        raise OpenEcologyPhaseAArchiveError(
            "closure and archive-tool authority source/endpoint differ"
        )
    options = RemoteArchiveOptions(
        ssh_target=archive_authority.ssh_target,
        remote_repository_root=archive_authority.remote_repository_root,
        remote_active_campaign_root=str(closure["active_campaign_root"]),
        remote_closed_bundle_dir=str(closure["closed_bundle_root"]),
        remote_staging_dir=str(remote_staging_directory),
        campaign_id=str(closure["campaign_id"]),
        bundle_id=str(closure["bundle_id"]),
        source_git_sha=str(source["git_sha"]),
        source_manifest_sha256=str(source["manifest_sha256"]),
        expected_marker_sha256=str(closed_bundle["marker_sha256"]),
        expected_entry_count=int(closed_bundle["entry_count"]),
        expected_file_count=int(closed_bundle["file_count"]),
        expected_directory_count=int(closed_bundle["directory_count"]),
        expected_total_file_bytes=int(closed_bundle["total_file_bytes"]),
        receipt_path=drive_receipt_path,
        tool_authority_path=archive_tool_authority_path,
        tool_authority_sha256=expected_archive_tool_authority_sha256,
        limits=CampaignStorageLimits(),
    )
    return execute_remote_archive(options)


def validate_phase_a_terminal_matrix_closure(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, object]:
    """Validate a compact closure receipt before the remote archive recheck."""

    receipt_file = _regular_file(path, field="closure receipt")
    raw = _read_bounded(receipt_file, maximum_bytes=_MAX_JSON_BYTES)
    if expected_file_sha256 is not None and sha256_bytes(raw) != _sha256(
        expected_file_sha256,
        field="closure receipt file SHA256",
    ):
        raise OpenEcologyPhaseAArchiveError(
            "closure receipt file differs from its external transfer digest"
        )
    receipt = _load_strict_json_bytes(raw, field="closure receipt")
    _require_exact_keys(
        receipt,
        {"payload", "payload_sha256", "schema_version"},
        field="closure receipt envelope",
    )
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise OpenEcologyPhaseAArchiveError("closure receipt envelope schema drifted")
    payload = dict(_mapping(receipt.get("payload"), field="closure receipt payload"))
    if receipt.get("payload_sha256") != sha256_bytes(canonical_json_bytes(payload)):
        raise OpenEcologyPhaseAArchiveError(
            "closure receipt envelope payload digest drifted"
        )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "campaign_id",
            "bundle_id",
            "active_campaign_root",
            "closed_bundle_root",
            "source",
            "authority",
            "terminal_matrix",
            "closed_bundle",
            "drive_archive_route",
            "exact_digest",
        },
        field="closure receipt",
    )
    _require_signed(payload, field="closure receipt")
    if payload.get("schema_version") != PHASE_A_TERMINAL_MATRIX_CLOSURE_SCHEMA_VERSION:
        raise OpenEcologyPhaseAArchiveError("closure receipt schema drifted")
    campaign_id = _identifier(payload.get("campaign_id"), field="campaign ID")
    bundle_id = _identifier(payload.get("bundle_id"), field="bundle ID")
    source = _mapping(payload.get("source"), field="closure source")
    _require_exact_keys(
        source,
        {"git_sha", "manifest_sha256"},
        field="closure source",
    )
    _git_sha(source.get("git_sha"), field="closure source Git SHA")
    _sha256(source.get("manifest_sha256"), field="closure source manifest")
    authority = _mapping(payload.get("authority"), field="closure authority")
    _require_exact_keys(
        authority,
        {
            "preregistration_digest",
            "archive_tool_authority_sha256",
            "evidence_index_digest",
            "launch_authorization_digest",
            "guardian_transcript_digest",
            "materialization_manifest_digest",
            "runtime_venv_authority_sha256",
            "ssh_target",
        },
        field="closure authority",
    )
    _identifier(authority.get("ssh_target"), field="closure authority SSH target")
    for key, value in authority.items():
        if key == "ssh_target":
            continue
        _sha256(value, field=f"closure authority {key}")
    if campaign_id != (f"phase-a-{str(authority['preregistration_digest'])[:24]}"):
        raise OpenEcologyPhaseAArchiveError(
            "closure campaign ID is detached from preregistration"
        )
    matrix = _sequence(payload.get("terminal_matrix"), field="terminal matrix")
    if len(matrix) != 16:
        raise OpenEcologyPhaseAArchiveError(
            "closure receipt does not bind all 16 terminal rows"
        )
    _validate_terminal_row_order(matrix)
    closed = _mapping(payload.get("closed_bundle"), field="closed bundle")
    _require_exact_keys(
        closed,
        {
            "marker_sha256",
            "entry_count",
            "file_count",
            "directory_count",
            "total_file_bytes",
        },
        field="closed bundle",
    )
    _sha256(closed.get("marker_sha256"), field="closed bundle marker")
    counts = {
        key: _nonnegative_int(closed.get(key), field=f"closed bundle {key}")
        for key in (
            "entry_count",
            "file_count",
            "directory_count",
            "total_file_bytes",
        )
    }
    if counts["entry_count"] != counts["file_count"] + counts["directory_count"]:
        raise OpenEcologyPhaseAArchiveError("closed bundle counts are inconsistent")
    route = _mapping(payload.get("drive_archive_route"), field="Drive archive route")
    expected_prefix = (
        f"gdrive:evolution-sim-backups/archives/open-ecology/{campaign_id}/{bundle_id}/"
    )
    if route != {
        "destination_prefix": expected_prefix,
        "existing_verified_uploader_required": True,
        "local_payload_staged": False,
        "source_deletion_authorized": False,
        "source_pruning_authorized": False,
    }:
        raise OpenEcologyPhaseAArchiveError("Drive archive route drifted")
    roots: dict[str, PurePosixPath] = {}
    for key in ("active_campaign_root", "closed_bundle_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
            raise OpenEcologyPhaseAArchiveError(
                f"closure {key} must be one absolute remote path"
            )
        parsed = PurePosixPath(value)
        if str(parsed) != value or parsed == PurePosixPath("/"):
            raise OpenEcologyPhaseAArchiveError(
                f"closure {key} must be one canonical remote path"
            )
        roots[key] = parsed
    if roots["closed_bundle_root"].name != bundle_id:
        raise OpenEcologyPhaseAArchiveError(
            "closure bundle path is detached from bundle ID"
        )
    if _posix_trees_overlap(
        roots["active_campaign_root"],
        roots["closed_bundle_root"],
    ):
        raise OpenEcologyPhaseAArchiveError("closure active and closed roots overlap")
    return payload


def _validate_launch_binding(
    path: Path,
    *,
    preregistration: Mapping[str, object],
) -> tuple[dict[str, object], str, datetime, dict[str, Path]]:
    authorization = _load_strict_json(path)
    _require_exact_keys(
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
        field="launch authorization",
    )
    _require_signed(authorization, field="launch authorization")
    if (
        authorization.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("campaign_digest") != preregistration.get("exact_digest")
        or authorization.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or authorization.get("source") != preregistration.get("source")
    ):
        raise OpenEcologyPhaseAArchiveError(
            "launch authorization is detached from preregistration"
        )
    scope = _mapping(
        authorization.get("authorization"),
        field="launch authorization scope",
    )
    if dict(scope) != {
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
        raise OpenEcologyPhaseAArchiveError(
            "launch authorization scope is not Phase-A training only"
        )
    _verify_payload_file_references(authorization, base=path.parent)
    index = _mapping(
        authorization.get("evidence_index"),
        field="launch authorization evidence index",
    )
    _require_exact_keys(
        index,
        {"exact_digest", "file"},
        field="launch authorization evidence index",
    )
    index_digest = _sha256(index.get("exact_digest"), field="evidence index digest")
    reference = _mapping(index.get("file"), field="evidence index file")
    evidence_path = _resolve_file_reference(path.parent, reference)
    evidence = _load_strict_json(evidence_path)
    _require_exact_keys(
        evidence,
        {
            "schema_version",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "dependency_reports",
            "operational_reports",
            "exact_digest",
        },
        field="evidence index",
    )
    _require_signed(evidence, field="evidence index")
    if (
        evidence.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_EVIDENCE_INDEX_SCHEMA_VERSION
        or evidence.get("exact_digest") != index_digest
        or evidence.get("campaign_digest") != preregistration.get("exact_digest")
        or evidence.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or evidence.get("source") != preregistration.get("source")
    ):
        raise OpenEcologyPhaseAArchiveError(
            "evidence index is detached from launch authorization"
        )
    _verify_payload_file_references(evidence, base=path.parent)
    validate_launch_authorization(
        authorization,
        preregistration=preregistration,
        authorization_path=path,
        require_current_storage_freshness=False,
    )
    authorization_time = _parse_utc(
        authorization.get("authorized_at_utc"),
        field="launch authorization time",
    )
    reports = _resolve_guardian_report_paths(
        evidence,
        authority_root=path.parent,
    )
    return authorization, index_digest, authorization_time, reports


def _resolve_guardian_report_paths(
    evidence_index: Mapping[str, object],
    *,
    authority_root: Path,
) -> dict[str, Path]:
    dependency_rows = _sequence(
        evidence_index.get("dependency_reports"),
        field="evidence index dependency reports",
    )
    d10_matches: list[Mapping[str, object]] = []
    for raw_dependency in dependency_rows:
        dependency = _mapping(raw_dependency, field="dependency report")
        if dependency.get("dependency_id") != "readiness_dependency_10":
            continue
        reports = _sequence(
            dependency.get("reports"),
            field="D10 dependency reports",
        )
        d10_matches.extend(
            report
            for raw_report in reports
            if (
                (report := _mapping(raw_report, field="D10 dependency report")).get(
                    "evidence_kind"
                )
                == "exact_sha_phase_a_training_and_torch_ci"
            )
        )
    if len(d10_matches) != 1:
        raise OpenEcologyPhaseAArchiveError(
            "authority requires exactly one dependency-10 D10 report"
        )
    resolved: dict[str, Path] = {
        "d10": _resolve_file_reference(
            authority_root,
            _mapping(d10_matches[0].get("file"), field="D10 report file"),
        )
    }
    kind_to_gate = {
        "phase_a_training_throughput": "throughput",
        "campaign_storage_capacity": "storage",
        "output_lock_contention": "output_lock",
    }
    operational_rows = _sequence(
        evidence_index.get("operational_reports"),
        field="evidence index operational reports",
    )
    for raw_row in operational_rows:
        row = _mapping(raw_row, field="operational report")
        report = _mapping(row.get("report"), field="operational report payload")
        evidence_kind = report.get("evidence_kind")
        gate = (
            kind_to_gate.get(evidence_kind) if isinstance(evidence_kind, str) else None
        )
        if gate is None or gate in resolved:
            raise OpenEcologyPhaseAArchiveError(
                "guardian operational report topology drifted"
            )
        resolved[gate] = _resolve_file_reference(
            authority_root,
            _mapping(report.get("file"), field=f"{gate} report file"),
        )
    if set(resolved) != {"d10", "throughput", "storage", "output_lock"}:
        raise OpenEcologyPhaseAArchiveError(
            "authority lacks the complete guardian report set"
        )
    return resolved


def _validate_transcript_verifier_digests(
    transcript: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    authorization_time: datetime,
    report_paths: Mapping[str, Path],
) -> None:
    expected_verifiers = {
        "d10": verify_exact_sha_phase_a_training_and_torch_ci_report,
        "throughput": verify_phase_a_training_throughput_report,
        "storage": verify_campaign_storage_capacity_report,
        "output_lock": verify_output_lock_contention_report,
    }
    if set(report_paths) != set(expected_verifiers):
        raise OpenEcologyPhaseAArchiveError(
            "guardian capture-verifier report topology drifted"
        )
    observed = {
        gate: stable_payload_digest(
            verifier(
                report_paths[gate],
                preregistration,
                authorization_time,
            )
        )
        for gate, verifier in expected_verifiers.items()
    }
    verifier_digests = _mapping(
        transcript.get("remote_verifier_digests"),
        field="guardian remote verifier digests",
    )
    for gate in ("d10", "throughput", "output_lock"):
        if verifier_digests.get(gate) != observed[gate]:
            raise OpenEcologyPhaseAArchiveError(
                f"guardian {gate} verifier digest lacks captured evidence provenance"
            )
    if transcript.get("storage_facts_digest") != observed["storage"]:
        raise OpenEcologyPhaseAArchiveError(
            "guardian storage verifier digest lacks captured evidence provenance"
        )


def _launch_storage_ssh_target(
    launch_authorization: Mapping[str, object],
) -> str:
    operational_gates = _mapping(
        launch_authorization.get("operational_gates"),
        field="launch operational gates",
    )
    storage = _mapping(
        operational_gates.get("storage"),
        field="launch storage gate",
    )
    return _identifier(
        storage.get("target_ssh"),
        field="launch storage SSH target",
    )


def _validate_runtime_authority(
    path: Path,
    *,
    file_sha256: str,
    preregistration: Mapping[str, object],
    launch_authorization: Mapping[str, object],
) -> None:
    source = _mapping(preregistration.get("source"), field="source")
    storage_ssh_target = _launch_storage_ssh_target(launch_authorization)
    operational_gates = _mapping(
        launch_authorization.get("operational_gates"),
        field="launch operational gates",
    )
    storage = _mapping(
        operational_gates.get("storage"),
        field="launch storage gate",
    )
    ssh_connection = _mapping(
        storage.get("ssh_connection"),
        field="launch storage SSH connection",
    )
    try:
        load_runtime_venv_authority(
            path,
            expected_sha256=file_sha256,
            expected_source_git_sha=_git_sha(
                source.get("commit"),
                field="runtime source Git SHA",
            ),
            expected_source_manifest_sha256=_sha256(
                source.get("manifest_sha256"),
                field="runtime source manifest",
            ),
            expected_archive_authority_sha256=_sha256(
                source.get("archive_tool_authority_sha256"),
                field="runtime archive authority",
            ),
            expected_ssh_target=storage_ssh_target,
            expected_ssh_connection_sha256=stable_payload_digest(dict(ssh_connection)),
        )
    except RuntimeVenvAuthorityError as error:
        raise OpenEcologyPhaseAArchiveError(
            f"runtime venv authority is not live-valid: {error}"
        ) from error


def _validate_guardian_and_terminal_matrix(
    transcript: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    launch_authorization: Mapping[str, object],
    evidence_index_digest: str,
    runtime_venv_authority_sha256: str,
    active_output_root: Path,
) -> list[dict[str, object]]:
    _require_exact_keys(
        transcript,
        {
            "schema_version",
            "authority_semantics",
            "bindings",
            "processes",
            "ssh_endpoint",
            "storage_facts_digest",
            "remote_verifier_digests",
            "completed_cells",
            "completed_cell_count",
            "matrix_complete",
            "transcript_is_resume_or_launch_authority",
            "exact_digest",
        },
        field="guardian transcript",
    )
    _require_signed(transcript, field="guardian transcript")
    if (
        transcript.get("schema_version") != TWO_PARTY_TRANSCRIPT_SCHEMA_VERSION
        or transcript.get("matrix_complete") is not True
        or transcript.get("completed_cell_count") != 16
        or transcript.get("transcript_is_resume_or_launch_authority") is not False
    ):
        raise OpenEcologyPhaseAArchiveError(
            "guardian transcript is incomplete or claims launch authority"
        )
    bindings = _mapping(transcript.get("bindings"), field="guardian bindings")
    source = _mapping(preregistration.get("source"), field="source")
    storage_ssh_target = _launch_storage_ssh_target(launch_authorization)
    expected_bindings = {
        "evidence_index_digest": evidence_index_digest,
        "launch_authorization_digest": launch_authorization["exact_digest"],
        "preregistration_digest": preregistration["exact_digest"],
        "runtime_venv_authority_sha256": runtime_venv_authority_sha256,
        "source_git_sha": source["commit"],
        "source_manifest_sha256": source["manifest_sha256"],
        "ssh_target": storage_ssh_target,
    }
    if dict(bindings) != expected_bindings:
        raise OpenEcologyPhaseAArchiveError(
            "guardian transcript source/authority bindings drifted"
        )
    semantics = _mapping(
        transcript.get("authority_semantics"),
        field="guardian authority semantics",
    )
    if dict(semantics) != {
        "offline_json_authorizes_updates": False,
        "capability_serialized": False,
        "cryptographic_mac_attestation": False,
        "hello_freshness": ("remote_nonce_bound_hmac_over_authenticated_pinned_ssh"),
        "mac_attestation_model": (
            "cooperative_exact_source_process_over_authenticated_pinned_ssh"
        ),
        "live_mac_storage_verifier_count": 1,
        "live_remote_verifier_counts": {
            "d10": 1,
            "throughput": 1,
            "output_lock": 1,
        },
        "selection_authorized": False,
        "phase_b_authorized": False,
        "phase_c_authorized": False,
        "phase_d_authorized": False,
    }:
        raise OpenEcologyPhaseAArchiveError(
            "guardian transcript authority semantics drifted"
        )
    processes = _mapping(transcript.get("processes"), field="guardian processes")
    _require_exact_keys(
        processes,
        {
            "coordinator_pgid",
            "coordinator_pid",
            "remote_pgid",
            "remote_pid",
        },
        field="guardian processes",
    )
    for key, value in processes.items():
        _positive_int(value, field=f"guardian process {key}")
    endpoint = _mapping(transcript.get("ssh_endpoint"), field="guardian SSH endpoint")
    _require_exact_keys(
        endpoint,
        {
            "address",
            "authenticated_host",
            "authentication",
            "host_key",
            "port",
        },
        field="guardian SSH endpoint",
    )
    for key in ("address", "authenticated_host", "authentication", "host_key"):
        _nonempty_text(endpoint.get(key), field=f"guardian SSH endpoint {key}")
    port = _positive_int(endpoint.get("port"), field="guardian SSH endpoint port")
    if port > 65_535:
        raise OpenEcologyPhaseAArchiveError("guardian SSH endpoint port is invalid")
    launch_storage = _mapping(
        _mapping(
            launch_authorization.get("operational_gates"),
            field="launch operational gates",
        ).get("storage"),
        field="launch storage gate",
    )
    launch_connection = _mapping(
        launch_storage.get("ssh_connection"),
        field="launch storage SSH connection",
    )
    if dict(endpoint) != dict(launch_connection):
        raise OpenEcologyPhaseAArchiveError(
            "guardian SSH endpoint differs from launch storage authority"
        )
    _sha256(
        transcript.get("storage_facts_digest"),
        field="guardian storage facts digest",
    )
    verifier_digests = _mapping(
        transcript.get("remote_verifier_digests"),
        field="guardian remote verifier digests",
    )
    if set(verifier_digests) != {"d10", "throughput", "output_lock"}:
        raise OpenEcologyPhaseAArchiveError(
            "guardian transcript remote verifier topology drifted"
        )
    for key, value in verifier_digests.items():
        _sha256(value, field=f"guardian verifier {key}")
    completed = _sequence(
        transcript.get("completed_cells"),
        field="guardian completed cells",
    )
    if len(completed) != 16:
        raise OpenEcologyPhaseAArchiveError(
            "guardian transcript lacks the complete cell matrix"
        )
    expected = _canonical_matrix()
    rows: list[dict[str, object]] = []
    observed_transcript: list[dict[str, object]] = []
    root_entries = _directory_names(active_output_root)
    expected_run_ids = {
        phase_a_run_id(cell_id=cell_id, learner_index=learner_index)
        for cell_id, learner_index in expected
    }
    if root_entries != expected_run_ids:
        raise OpenEcologyPhaseAArchiveError(
            "active Phase-A root is incomplete or contains surplus entries"
        )
    for index, ((cell_id, learner_index), raw_completed) in enumerate(
        zip(expected, completed, strict=True)
    ):
        completed_row = _mapping(
            raw_completed,
            field=f"guardian completed cell {index}",
        )
        run_id = phase_a_run_id(
            cell_id=cell_id,
            learner_index=learner_index,
        )
        if (
            completed_row.get("cell_id") != cell_id
            or completed_row.get("learner_index") != learner_index
            or completed_row.get("run_id") != run_id
        ):
            raise OpenEcologyPhaseAArchiveError(
                "guardian completed-cell order or identity drifted"
            )
        transcript_terminal_digest = _sha256(
            completed_row.get("exact_digest"),
            field=f"guardian terminal digest {index}",
        )
        terminal_path = active_output_root / run_id / "terminal" / "terminal.json"
        request = build_verified_phase_a_selection_request(
            preregistration,
            cell_id=cell_id,
            learner_index=learner_index,
            terminal_path=terminal_path,
            evaluation_workers=1,
        )
        terminal_authority = _mapping(
            request.training_authority,
            field=f"terminal authority {index}",
        )
        observed_digest = _sha256(
            terminal_authority.get("terminal_exact_digest"),
            field=f"terminal authority digest {index}",
        )
        if observed_digest != transcript_terminal_digest:
            raise OpenEcologyPhaseAArchiveError(
                "guardian transcript differs from a verified terminal chain"
            )
        observed_transcript.append(
            {
                "cell_id": cell_id,
                "exact_digest": transcript_terminal_digest,
                "learner_index": learner_index,
                "run_id": run_id,
            }
        )
        rows.append(
            {
                "cell_id": cell_id,
                "learner_index": learner_index,
                "run_id": run_id,
                "terminal_exact_digest": observed_digest,
                "terminal_authority_exact_digest": _sha256(
                    terminal_authority.get("exact_digest"),
                    field=f"terminal selection authority digest {index}",
                ),
                "artifact_sha256": _sha256(
                    terminal_authority.get("artifact_sha256"),
                    field=f"terminal artifact digest {index}",
                ),
            }
        )
    if observed_transcript != [
        dict(_mapping(row, field="completed cell")) for row in completed
    ]:
        raise OpenEcologyPhaseAArchiveError(
            "guardian completed-cell rows contain surplus or changed fields"
        )
    return rows


class _held_run_lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> _held_run_lock:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OpenEcologyPhaseAArchiveError(
                    "Phase-A run lock is not one single-link regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OpenEcologyPhaseAArchiveError:
            if "descriptor" in locals():
                os.close(descriptor)
            raise
        except OSError as error:
            if "descriptor" in locals():
                os.close(descriptor)
            raise OpenEcologyPhaseAArchiveError(
                f"Phase-A run lock is unavailable: {self.path.name}"
            ) from error
        self.descriptor = descriptor
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.descriptor is None:
            return
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _build_authority_allowlist(
    authority_root: Path,
    *,
    explicit_files: Sequence[Path],
) -> tuple[frozenset[str], frozenset[str]]:
    """Derive the only authority files eligible for durable publication."""

    root = authority_root.resolve()
    pending: list[Path] = []
    allowed_files: set[str] = set()
    parsed_files: set[str] = set()

    def add_file(path: Path, *, parse_json: bool) -> None:
        canonical = _regular_file(path, field="authority allowlist file")
        if not _is_within(canonical, root):
            raise OpenEcologyPhaseAArchiveError(
                "authority reference escaped the copied authority root"
            )
        relative = canonical.relative_to(root).as_posix()
        allowed_files.add(relative)
        if parse_json and relative not in parsed_files:
            pending.append(canonical)

    for explicit in explicit_files:
        add_file(explicit, parse_json=True)

    while pending:
        current = pending.pop()
        relative = current.relative_to(root).as_posix()
        if relative in parsed_files:
            continue
        parsed_files.add(relative)
        payload = _load_strict_json(current)
        is_raw_manifest = (
            payload.get("schema_version")
            == OPEN_ECOLOGY_RAW_EVIDENCE_MANIFEST_SCHEMA_VERSION
        )
        for reference in _nested_file_references(payload):
            referenced = _resolve_file_reference(current.parent, reference)
            add_file(
                referenced,
                parse_json=(referenced.suffix == ".json" and not is_raw_manifest),
            )

    allowed_directories: set[str] = set()
    for relative in allowed_files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(allowed_files), frozenset(allowed_directories)


def _nested_file_references(
    value: object,
) -> list[Mapping[str, object]]:
    references: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        if set(value) == {"relative_path", "sha256", "byte_length"}:
            references.append(_mapping(value, field="file reference"))
        else:
            for child in value.values():
                references.extend(_nested_file_references(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            references.extend(_nested_file_references(child))
    return references


def _require_exact_authority_tree(
    source: Path,
    *,
    expected_files: frozenset[str],
    expected_directories: frozenset[str],
) -> None:
    files, directories = _tree_paths_descriptor_safe(source)
    if files != expected_files or directories != expected_directories:
        raise OpenEcologyPhaseAArchiveError(
            "authority root contains surplus or missing unreferenced entries"
        )


def _tree_paths_descriptor_safe(
    source: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    root_before = os.fstat(descriptor)
    files: set[str] = set()
    directories: set[str] = set()
    try:
        _scan_tree_paths_fd(
            descriptor,
            relative="",
            root_device=root_before.st_dev,
            files=files,
            directories=directories,
        )
        _require_same_identity(
            root_before,
            os.fstat(descriptor),
            field=source.name,
        )
    finally:
        os.close(descriptor)
    return frozenset(files), frozenset(directories)


def _scan_tree_paths_fd(
    descriptor: int,
    *,
    relative: str,
    root_device: int,
    files: set[str],
    directories: set[str],
) -> None:
    for name in sorted(os.listdir(descriptor)):
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise OpenEcologyPhaseAArchiveError("unsafe authority entry name")
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if before.st_dev != root_device:
            raise OpenEcologyPhaseAArchiveError(
                "Phase-A closure refuses authority mount crossings"
            )
        path = f"{relative}/{name}" if relative else name
        if stat.S_ISDIR(before.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                _require_same_identity(before, opened, field=path)
                directories.add(path)
                _scan_tree_paths_fd(
                    child,
                    relative=path,
                    root_device=root_device,
                    files=files,
                    directories=directories,
                )
                _require_same_identity(opened, os.fstat(child), field=path)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OpenEcologyPhaseAArchiveError(
                "authority accepts only single-link regular files/directories"
            )
        files.add(path)


def _copy_tree_descriptor_safe(
    source: Path,
    destination: Path,
    *,
    expected_files: frozenset[str] | None = None,
    expected_directories: frozenset[str] | None = None,
) -> _CopiedTree:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    destination_fd = os.open(destination, flags)
    records: list[dict[str, object]] = []
    root_before = os.fstat(source_fd)
    root_device = root_before.st_dev
    try:
        _copy_directory_fd(
            source_fd,
            destination_fd,
            relative="",
            root_device=root_device,
            records=records,
            expected_files=expected_files,
            expected_directories=expected_directories,
        )
        _require_same_identity(
            root_before,
            os.fstat(source_fd),
            field=source.name,
        )
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)
    files = [record for record in records if record["kind"] == "file"]
    directories = [record for record in records if record["kind"] == "directory"]
    if (
        expected_files is not None
        and {str(record["path"]) for record in files} != expected_files
    ):
        raise OpenEcologyPhaseAArchiveError(
            "authority files changed during descriptor copy"
        )
    if (
        expected_directories is not None
        and {str(record["path"]) for record in directories} != expected_directories
    ):
        raise OpenEcologyPhaseAArchiveError(
            "authority directories changed during descriptor copy"
        )
    inventory = {"entries": records}
    return _CopiedTree(
        entry_count=len(records),
        file_count=len(files),
        directory_count=len(directories),
        total_file_bytes=sum(int(record["size"]) for record in files),
        inventory_sha256=_payload_digest(inventory),
    )


def _copy_directory_fd(
    source_fd: int,
    destination_fd: int,
    *,
    relative: str,
    root_device: int,
    records: list[dict[str, object]],
    expected_files: frozenset[str] | None,
    expected_directories: frozenset[str] | None,
) -> None:
    for name in sorted(os.listdir(source_fd)):
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise OpenEcologyPhaseAArchiveError("unsafe source entry name")
        before = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if before.st_dev != root_device:
            raise OpenEcologyPhaseAArchiveError(
                "Phase-A closure refuses source mount crossings"
            )
        path = f"{relative}/{name}" if relative else name
        mode = stat.S_IMODE(before.st_mode)
        if stat.S_ISDIR(before.st_mode):
            if expected_directories is not None and path not in expected_directories:
                raise OpenEcologyPhaseAArchiveError(
                    "authority root contains surplus directories during copy"
                )
            os.mkdir(name, mode=0o700, dir_fd=destination_fd)
            source_child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=source_fd,
            )
            destination_child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=destination_fd,
            )
            try:
                opened = os.fstat(source_child)
                _require_same_identity(before, opened, field=path)
                records.append({"kind": "directory", "mode": mode, "path": path})
                _copy_directory_fd(
                    source_child,
                    destination_child,
                    relative=path,
                    root_device=root_device,
                    records=records,
                    expected_files=expected_files,
                    expected_directories=expected_directories,
                )
                after = os.fstat(source_child)
                _require_same_identity(opened, after, field=path)
                os.fchmod(destination_child, mode)
                os.fsync(destination_child)
            finally:
                os.close(destination_child)
                os.close(source_child)
            continue
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OpenEcologyPhaseAArchiveError(
                "Phase-A closure accepts only single-link regular files/directories"
            )
        if expected_files is not None and path not in expected_files:
            raise OpenEcologyPhaseAArchiveError(
                "authority root contains surplus files during copy"
            )
        source_file = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_fd,
        )
        destination_file = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_fd,
        )
        digest = hashlib.sha256()
        copied = 0
        try:
            opened = os.fstat(source_file)
            _require_same_identity(before, opened, field=path)
            with os.fdopen(os.dup(source_file), "rb", closefd=True) as source_handle:
                with os.fdopen(
                    os.dup(destination_file),
                    "wb",
                    closefd=True,
                ) as destination_handle:
                    while block := source_handle.read(_COPY_BLOCK_BYTES):
                        digest.update(block)
                        destination_handle.write(block)
                        copied += len(block)
                    destination_handle.flush()
            if copied != opened.st_size:
                raise OpenEcologyPhaseAArchiveError(
                    "source file size changed during Phase-A closure"
                )
            after = os.fstat(source_file)
            _require_same_identity(opened, after, field=path)
            os.fchmod(destination_file, mode)
            os.fsync(destination_file)
        finally:
            os.close(destination_file)
            os.close(source_file)
        records.append(
            {
                "kind": "file",
                "mode": mode,
                "path": path,
                "sha256": digest.hexdigest(),
                "size": copied,
            }
        )


def _require_same_identity(
    before: os.stat_result,
    after: os.stat_result,
    *,
    field: str,
) -> None:
    if any(
        getattr(before, attribute) != getattr(after, attribute)
        for attribute in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    ):
        raise OpenEcologyPhaseAArchiveError(
            f"source entry changed during Phase-A closure: {field}"
        )


def _validate_terminal_row_order(rows: Sequence[object]) -> None:
    for index, ((cell_id, learner_index), raw_row) in enumerate(
        zip(_canonical_matrix(), rows, strict=True)
    ):
        row = _mapping(raw_row, field=f"terminal row {index}")
        _require_exact_keys(
            row,
            {
                "cell_id",
                "learner_index",
                "run_id",
                "terminal_exact_digest",
                "terminal_authority_exact_digest",
                "artifact_sha256",
            },
            field=f"terminal row {index}",
        )
        if (
            row.get("cell_id") != cell_id
            or row.get("learner_index") != learner_index
            or row.get("run_id")
            != phase_a_run_id(cell_id=cell_id, learner_index=learner_index)
        ):
            raise OpenEcologyPhaseAArchiveError(
                "terminal matrix row order or identity drifted"
            )
        for key in (
            "terminal_exact_digest",
            "terminal_authority_exact_digest",
            "artifact_sha256",
        ):
            _sha256(row.get(key), field=f"terminal row {index} {key}")


def _resolve_file_reference(base: Path, reference: Mapping[str, object]) -> Path:
    _require_exact_keys(
        reference,
        {"relative_path", "sha256", "byte_length"},
        field="file reference",
    )
    relative = reference.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or "\x00" in relative
    ):
        raise OpenEcologyPhaseAArchiveError("file reference path is invalid")
    parsed = PurePosixPath(relative)
    if parsed.as_posix() != relative or any(
        part in {".", ".."} for part in parsed.parts
    ):
        raise OpenEcologyPhaseAArchiveError("file reference path is non-canonical")
    path = base.joinpath(*parsed.parts).resolve()
    if not _is_within(path, base.resolve()):
        raise OpenEcologyPhaseAArchiveError("file reference escaped authority root")
    path = _regular_file(path, field="file reference")
    if path.stat().st_size != _nonnegative_int(
        reference.get("byte_length"), field="file byte length"
    ) or _sha256_file(path) != _sha256(
        reference.get("sha256"), field="file reference SHA256"
    ):
        raise OpenEcologyPhaseAArchiveError("file reference bytes drifted")
    return path


def _verify_payload_file_references(
    value: object,
    *,
    base: Path,
) -> None:
    """Verify every canonical file reference nested in one sealed payload."""

    if isinstance(value, Mapping):
        if set(value) == {"relative_path", "sha256", "byte_length"}:
            _resolve_file_reference(base, _mapping(value, field="file reference"))
            return
        for child in value.values():
            _verify_payload_file_references(child, base=base)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _verify_payload_file_references(child, base=base)


def _load_strict_json(path: Path) -> dict[str, object]:
    raw = _read_bounded(path, maximum_bytes=_MAX_JSON_BYTES)
    return _load_strict_json_bytes(raw, field=path.name)


def _load_strict_json_bytes(
    raw: bytes,
    *,
    field: str,
) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OpenEcologyPhaseAArchiveError(
            f"failed to load strict JSON {field}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise OpenEcologyPhaseAArchiveError("strict JSON root must be an object")
    return payload


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise OpenEcologyPhaseAArchiveError(f"{field} must be UTC RFC3339 seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise OpenEcologyPhaseAArchiveError(
            f"{field} must be UTC RFC3339 seconds"
        ) from error


def _read_bounded(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise OpenEcologyPhaseAArchiveError(
                "authority JSON is not one bounded single-link regular file"
            )
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        _require_same_identity(before, after, field=path.name)
        if len(payload) != before.st_size or len(payload) > maximum_bytes:
            raise OpenEcologyPhaseAArchiveError("authority JSON changed while read")
        return payload
    finally:
        os.close(descriptor)


def _new_output_path(path: Path, *, field: str) -> Path:
    if not path.is_absolute():
        raise OpenEcologyPhaseAArchiveError(f"{field} must be absolute")
    parent = _canonical_existing_directory(path.parent, field=f"{field} parent")
    destination = parent / path.name
    if destination.exists() or destination.is_symlink():
        raise OpenEcologyPhaseAArchiveError(f"{field} already exists")
    return destination


def _regular_file(path: Path, *, field: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise OpenEcologyPhaseAArchiveError(
            f"{field} must be one absolute non-symlink file"
        )
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise OpenEcologyPhaseAArchiveError(f"{field} is unavailable") from error
    if resolved != path or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OpenEcologyPhaseAArchiveError(
            f"{field} must be canonical and single-link"
        )
    return resolved


def _canonical_existing_directory(path: Path, *, field: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise OpenEcologyPhaseAArchiveError(
            f"{field} must be one absolute non-symlink directory"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise OpenEcologyPhaseAArchiveError(f"{field} is unavailable") from error
    if resolved != path or not resolved.is_dir():
        raise OpenEcologyPhaseAArchiveError(f"{field} must be canonical")
    return resolved


def _require_pairwise_distinct_trees(
    trees: Sequence[tuple[str, Path]],
) -> None:
    for index, (left_field, left) in enumerate(trees):
        for right_field, right in trees[index + 1 :]:
            if _is_within(left, right) or _is_within(right, left):
                raise OpenEcologyPhaseAArchiveError(
                    f"{left_field} and {right_field} overlap"
                )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _posix_trees_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    try:
        left.relative_to(right)
    except ValueError:
        pass
    else:
        return True
    try:
        right.relative_to(left)
    except ValueError:
        return False
    return True


def _directory_names(path: Path) -> set[str]:
    try:
        return {entry.name for entry in path.iterdir()}
    except OSError as error:
        raise OpenEcologyPhaseAArchiveError(
            "Phase-A terminal root cannot be inventoried"
        ) from error


def _canonical_matrix() -> tuple[tuple[str, int], ...]:
    return tuple(
        (cell_id, learner_index)
        for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OpenEcologyPhaseAArchiveError(f"{field} must be a string-key mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OpenEcologyPhaseAArchiveError(f"{field} must be a sequence")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(value) != expected:
        raise OpenEcologyPhaseAArchiveError(f"{field} keys drifted")


def _require_signed(payload: Mapping[str, object], *, field: str) -> None:
    exact = _sha256(payload.get("exact_digest"), field=f"{field} exact digest")
    unsigned = dict(payload)
    unsigned.pop("exact_digest")
    if _payload_digest(unsigned) != exact:
        raise OpenEcologyPhaseAArchiveError(f"{field} digest drifted")


def _payload_digest(payload: Mapping[str, object]) -> str:
    return stable_payload_digest(dict(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenEcologyPhaseAArchiveError(f"{field} must be a lowercase SHA256")
    return value


def _git_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenEcologyPhaseAArchiveError(f"{field} must be a lowercase Git SHA")
    return value


def _identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or value[0] not in _IDENTIFIER_INITIAL_CHARACTERS
        or any(character not in _IDENTIFIER_CHARACTERS for character in value)
    ):
        raise OpenEcologyPhaseAArchiveError(f"{field} is not a valid identifier")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyPhaseAArchiveError(f"{field} must be non-negative")
    return value


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise OpenEcologyPhaseAArchiveError(f"{field} must be positive")
    return parsed


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        raise OpenEcologyPhaseAArchiveError(f"{field} must be bounded text")
    return value


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologyPhaseAArchiveError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise OpenEcologyPhaseAArchiveError(f"non-finite JSON constant {value!r}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "OpenEcologyPhaseAArchiveError",
    "PHASE_A_TERMINAL_MATRIX_CLOSURE_SCHEMA_VERSION",
    "PHASE_A_TERMINAL_MATRIX_MANIFEST_NAME",
    "PHASE_A_TERMINAL_MATRIX_MANIFEST_SCHEMA_VERSION",
    "archive_phase_a_terminal_matrix",
    "close_phase_a_terminal_matrix",
    "validate_phase_a_terminal_matrix_closure",
]
