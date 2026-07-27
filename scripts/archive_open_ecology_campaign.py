#!/usr/bin/env python3
"""Archive one sealed open-ecology bundle and verify it on Google Drive.

The active campaign root is never an archive input.  A remote helper running
from the exact clean source checkout validates a separately sealed bundle,
uses ``archive_evolution_outputs.py`` to create its deterministic ``tar.zst``
triple, and leaves those source objects on the compute host.  The coordinator
streams only those three objects to the immutable campaign-scoped Drive path,
reads every byte back, runs an independent ``rclone check`` from a tiny local
SHA256 sum file, and writes one small receipt only after every gate passes.

There is deliberately no deletion or pruning path in this tool.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tempfile


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPOSITORY_ROOT / "python"
sys.path = [
    entry for entry in sys.path if Path(entry or os.curdir).resolve() != _PYTHON_ROOT
]
sys.path.insert(0, str(_PYTHON_ROOT))

from evolution_sim.io.open_ecology_campaign_storage import (  # noqa: E402
    CampaignStorageError,
    CampaignStorageLimits,
    CampaignStorageLock,
    ClosedBundleSnapshot,
    DEFAULT_REMOTE_FREE_BYTES,
    canonical_json_bytes,
    check_campaign_storage,
    default_storage_lock_path,
    validate_closed_bundle,
    write_verified_receipt,
)
from evolution_sim.io.source_manifest import (  # noqa: E402
    source_file_hash_manifest,
)


DEFAULT_RCLONE_BASE = "gdrive:evolution-sim-backups/archives/open-ecology"
REMOTE_BUILD_SCHEMA_VERSION = "open_ecology_closed_bundle_archive_build_v1"
REMOTE_RECEIPT_SCHEMA_VERSION = "open_ecology_closed_bundle_drive_receipt_v1"
_SSH_TARGET_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_DRIVE_ID_PATTERN = re.compile(r"[^\x00-\x20/\\]+")
_READ_CHUNK_SIZE = 8 * 1024 * 1024


class OpenEcologyArchiveError(CampaignStorageError):
    """The closed-bundle archive or Drive verification failed closed."""


@dataclass(frozen=True, slots=True)
class RemoteArchiveOptions:
    ssh_target: str
    remote_repository_root: str
    remote_active_campaign_root: str
    remote_closed_bundle_dir: str
    remote_staging_dir: str
    campaign_id: str
    bundle_id: str
    source_git_sha: str
    source_manifest_sha256: str
    receipt_path: Path
    limits: CampaignStorageLimits = CampaignStorageLimits()
    rclone_base: str = DEFAULT_RCLONE_BASE

    def validate(self) -> None:
        if (
            not isinstance(self.ssh_target, str)
            or not self.ssh_target
            or self.ssh_target.startswith("-")
            or _SSH_TARGET_PATTERN.fullmatch(self.ssh_target) is None
        ):
            raise OpenEcologyArchiveError("ssh target contains unsupported characters")
        repository = _absolute_posix_path(
            self.remote_repository_root,
            field="remote repository root",
        )
        active = _absolute_posix_path(
            self.remote_active_campaign_root,
            field="remote active campaign root",
        )
        bundle = _absolute_posix_path(
            self.remote_closed_bundle_dir,
            field="remote closed bundle",
        )
        staging = _absolute_posix_path(
            self.remote_staging_dir,
            field="remote staging directory",
        )
        if repository == PurePosixPath("/"):
            raise OpenEcologyArchiveError(
                "remote repository root must identify an exact checkout"
            )
        for left_name, left, right_name, right in (
            ("repository", repository, "active campaign", active),
            ("repository", repository, "closed bundle", bundle),
            ("repository", repository, "staging directory", staging),
            ("active campaign", active, "closed bundle", bundle),
            ("active campaign", active, "staging directory", staging),
            ("closed bundle", bundle, "staging directory", staging),
        ):
            if _trees_overlap(left, right):
                raise OpenEcologyArchiveError(
                    f"{left_name} and {right_name} must be separate trees"
                )
        _identifier(self.campaign_id, field="campaign id")
        _identifier(self.bundle_id, field="bundle id")
        if bundle.name != self.bundle_id:
            raise OpenEcologyArchiveError(
                "remote closed-bundle directory name must equal bundle id"
            )
        _git_sha(self.source_git_sha)
        _sha256(self.source_manifest_sha256, field="source manifest SHA256")
        self.limits.validate()
        if self.rclone_base != DEFAULT_RCLONE_BASE:
            raise OpenEcologyArchiveError(
                "open-ecology Drive base is sealed and may not be changed"
            )
        receipt = Path(self.receipt_path)
        if not receipt.is_absolute():
            raise OpenEcologyArchiveError("receipt path must be absolute")
        _require_canonical_directory(receipt.parent, field="receipt parent")
        if receipt.exists() or receipt.is_symlink():
            raise OpenEcologyArchiveError(
                f"refusing to overwrite immutable local receipt: {receipt}"
            )


def execute_remote_archive(options: RemoteArchiveOptions) -> dict[str, object]:
    """Run every upload and verification gate, then write one receipt."""

    options.validate()
    drive_free_before_bytes = _require_drive_quota(options.limits.min_remote_free_bytes)
    build = _remote_build(options)
    objects = _validated_remote_build(build, options=options)
    expected = {str(row["name"]): row for row in objects}
    expected_names = tuple(sorted(expected))
    destination_prefix = (
        f"{DEFAULT_RCLONE_BASE}/{options.campaign_id}/{options.bundle_id}"
    )

    initial = _inspect_remote_inventory(
        destination_prefix,
        expected_names=expected_names,
        require_complete=False,
    )
    initial_by_name = {str(row["name"]): row for row in initial}
    for name, inventory_row in initial_by_name.items():
        observed = _rclone_readback(f"{destination_prefix}/{name}")
        _require_object_match(
            observed,
            expected[name],
            field=f"existing Drive object {name}",
        )
        if inventory_row["size"] != expected[name]["size"]:
            raise OpenEcologyArchiveError(
                f"existing Drive inventory size mismatch for {name}"
            )

    for name in expected_names:
        if name in initial_by_name:
            continue
        source = expected[name]
        _stream_remote_object(
            ssh_target=options.ssh_target,
            source_path=str(source["source_path"]),
            source_size=int(source["size"]),
            destination_path=f"{destination_prefix}/{name}",
        )

    verification_inventory = _inspect_remote_inventory(
        destination_prefix,
        expected_names=expected_names,
        require_complete=True,
    )
    readbacks: dict[str, dict[str, object]] = {}
    for name in expected_names:
        observed = _rclone_readback(f"{destination_prefix}/{name}")
        _require_object_match(
            observed,
            expected[name],
            field=f"Drive readback {name}",
        )
        readbacks[name] = observed

    check_rows = _run_independent_rclone_check(
        destination_prefix,
        expected=expected,
    )
    final_inventory = _inspect_remote_inventory(
        destination_prefix,
        expected_names=expected_names,
        require_complete=True,
    )
    if final_inventory != verification_inventory:
        raise OpenEcologyArchiveError(
            "Drive object IDs, names, or sizes changed during verification"
        )
    final_by_name = {str(row["name"]): row for row in final_inventory}
    for name in expected_names:
        if final_by_name[name]["size"] != expected[name]["size"]:
            raise OpenEcologyArchiveError(
                f"final Drive inventory size mismatch for {name}"
            )
    drive_free_after_bytes = _require_drive_quota(options.limits.min_remote_free_bytes)

    payload: dict[str, object] = {
        "archive": {
            "archive_name": build["archive_name"],
            "marker_sha256": build["marker_sha256"],
            "objects": [
                {
                    "drive_id": final_by_name[name]["id"],
                    "name": name,
                    "sha256": expected[name]["sha256"],
                    "size": expected[name]["size"],
                }
                for name in expected_names
            ],
        },
        "bundle_id": options.bundle_id,
        "campaign_id": options.campaign_id,
        "destination_prefix": destination_prefix,
        "drive_quota": {
            "free_after_bytes": drive_free_after_bytes,
            "free_before_bytes": drive_free_before_bytes,
            "minimum_free_bytes": options.limits.min_remote_free_bytes,
        },
        "local_payload_staged": False,
        "pruning_available": False,
        "rclone_check": {
            "checkfile": "SHA-256",
            "combined_rows": check_rows,
            "matched_objects": 3,
            "one_way": False,
        },
        "readback": {
            name: {
                "sha256": readbacks[name]["sha256"],
                "size": readbacks[name]["size"],
            }
            for name in expected_names
        },
        "remote_source_bytes_deleted": False,
        "schema_version": REMOTE_RECEIPT_SCHEMA_VERSION,
        "source": {
            "git_sha": options.source_git_sha,
            "manifest_sha256": options.source_manifest_sha256,
            "repository_root": options.remote_repository_root,
        },
        "status": "three_objects_byte_readback_and_rclone_check_verified",
    }
    envelope = write_verified_receipt(options.receipt_path, payload)
    return {
        "destination_prefix": destination_prefix,
        "receipt_path": str(options.receipt_path),
        "receipt_payload_sha256": envelope["payload_sha256"],
        "status": payload["status"],
    }


def build_remote_closed_bundle_archive(
    *,
    repository_root: Path,
    active_campaign_root: Path,
    closed_bundle_dir: Path,
    staging_dir: Path,
    campaign_id: str,
    bundle_id: str,
    source_git_sha: str,
    source_manifest_sha256: str,
    limits: CampaignStorageLimits,
) -> dict[str, object]:
    """Remote helper: validate source/bundle and produce the exact triple."""

    repository = _require_exact_source_binding(
        repository_root,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
    )
    active = _require_canonical_directory(
        active_campaign_root,
        field="active campaign root",
    )
    bundle = _require_canonical_directory(
        closed_bundle_dir,
        field="closed bundle",
    )
    _require_separate_local_trees(repository, active)
    _require_separate_local_trees(repository, bundle)
    _require_separate_local_trees(active, bundle)
    staging = _prepare_staging_directory(staging_dir, active=active, bundle=bundle)
    _require_separate_local_trees(repository, staging)
    limits.validate()

    with CampaignStorageLock(
        default_storage_lock_path(active),
        campaign_id=campaign_id,
        source_git_sha=source_git_sha,
    ):
        active_scan = check_campaign_storage(
            active,
            campaign_id=campaign_id,
            source_git_sha=source_git_sha,
            limits=limits,
        )
        snapshot = validate_closed_bundle(
            active,
            bundle,
            campaign_id=campaign_id,
            bundle_id=bundle_id,
            source_git_sha=source_git_sha,
            source_manifest_sha256=source_manifest_sha256,
            limits=limits,
        )
        if (
            active_scan.total_file_bytes + snapshot.scan.total_file_bytes
            > limits.max_campaign_bytes
        ):
            raise OpenEcologyArchiveError(
                "active root plus closed bundle exceeds the global campaign budget"
            )
        archive_name = f"{bundle_id}-{snapshot.marker_sha256[:16]}.tar.zst"
        object_paths = (
            staging / archive_name,
            staging / f"{archive_name}.manifest.json",
            staging / f"{archive_name}.sha256",
        )
        present = _staging_object_names(staging)
        expected_names = {path.name for path in object_paths}
        if present and present != expected_names:
            raise OpenEcologyArchiveError(
                "remote staging prefix is partial or contains surplus evidence"
            )
        if not present:
            _run_archive_producer(
                repository=repository,
                bundle=bundle,
                staging=staging,
                archive_name=archive_name,
            )
        records = _validate_producer_objects(snapshot, object_paths)
        repeated = validate_closed_bundle(
            active,
            bundle,
            campaign_id=campaign_id,
            bundle_id=bundle_id,
            source_git_sha=source_git_sha,
            source_manifest_sha256=source_manifest_sha256,
            limits=limits,
        )
        if repeated.producer_manifest != snapshot.producer_manifest:
            raise OpenEcologyArchiveError(
                "closed bundle changed during archive production"
            )
    _require_exact_source_binding(
        repository,
        source_git_sha=source_git_sha,
        source_manifest_sha256=source_manifest_sha256,
    )
    return {
        "archive_name": archive_name,
        "bundle_id": bundle_id,
        "campaign_id": campaign_id,
        "marker_sha256": snapshot.marker_sha256,
        "objects": records,
        "remote_input_pruned": False,
        "remote_staging_pruned": False,
        "schema_version": REMOTE_BUILD_SCHEMA_VERSION,
        "source_git_sha": source_git_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "status": "deterministic_tar_zst_triple_validated",
    }


def _run_archive_producer(
    *,
    repository: Path,
    bundle: Path,
    staging: Path,
    archive_name: str,
) -> None:
    tool_path = repository / "scripts" / "archive_evolution_outputs.py"
    completed = _run(
        (
            sys.executable,
            str(tool_path),
            "--input-dir",
            str(bundle),
            "--output-dir",
            str(staging),
            "--archive-name",
            archive_name,
            "--archive-only",
        )
    )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError(
            "archive_evolution_outputs.py did not emit valid JSON"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("archive_name") != archive_name
        or payload.get("status") != "local_archive_verified"
        or payload.get("prune_requested") is not False
        or payload.get("archive_only") is not True
    ):
        raise OpenEcologyArchiveError(
            "archive_evolution_outputs.py result contract mismatch"
        )


def _validate_producer_objects(
    snapshot: ClosedBundleSnapshot,
    object_paths: Sequence[Path],
) -> list[dict[str, object]]:
    archive_path, manifest_path, sidecar_path = object_paths
    for path in object_paths:
        if path.is_symlink():
            raise OpenEcologyArchiveError("producer object may not be a symlink")
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
        ):
            raise OpenEcologyArchiveError(
                "producer output is not one positive regular file"
            )
    manifest_bytes = _read_regular_file(manifest_path)
    try:
        producer_manifest = json.loads(
            manifest_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("producer manifest is not strict JSON") from exc
    if producer_manifest != snapshot.producer_manifest:
        raise OpenEcologyArchiveError(
            "producer manifest differs from descriptor-safe preflight snapshot"
        )
    if manifest_bytes != canonical_json_bytes(snapshot.producer_manifest):
        raise OpenEcologyArchiveError(
            "producer manifest bytes are not the expected canonical form"
        )
    archive_sha256 = _sha256_path(archive_path)
    expected_sidecar = f"{archive_sha256}  {archive_path.name}\n".encode()
    if _read_regular_file(sidecar_path) != expected_sidecar:
        raise OpenEcologyArchiveError("archive SHA sidecar is missing or mismatched")
    _run(("zstd", "--test", "--quiet", str(archive_path)))

    records: list[dict[str, object]] = []
    for path in object_paths:
        os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o222)
        records.append(
            {
                "name": path.name,
                "sha256": _sha256_path(path),
                "size": path.stat().st_size,
                "source_path": str(path),
            }
        )
    return records


def _remote_build(options: RemoteArchiveOptions) -> dict[str, object]:
    command = (
        "python3",
        str(
            PurePosixPath(options.remote_repository_root)
            / "scripts"
            / "archive_open_ecology_campaign.py"
        ),
        "remote-build",
        "--repository-root",
        options.remote_repository_root,
        "--active-campaign-root",
        options.remote_active_campaign_root,
        "--closed-bundle-dir",
        options.remote_closed_bundle_dir,
        "--staging-dir",
        options.remote_staging_dir,
        "--campaign-id",
        options.campaign_id,
        "--bundle-id",
        options.bundle_id,
        "--source-git-sha",
        options.source_git_sha,
        "--source-manifest-sha256",
        options.source_manifest_sha256,
        "--max-campaign-bytes",
        str(options.limits.max_campaign_bytes),
        "--min-campaign-free-bytes",
        str(options.limits.min_campaign_free_bytes),
        "--min-remote-free-bytes",
        str(options.limits.min_remote_free_bytes),
        "--max-entries",
        str(options.limits.max_entries),
    )
    completed = _run(_ssh_command(options.ssh_target, command))
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("remote build did not emit valid JSON") from exc
    if not isinstance(payload, dict):
        raise OpenEcologyArchiveError("remote build JSON root must be an object")
    return payload


def _validated_remote_build(
    payload: Mapping[str, object],
    *,
    options: RemoteArchiveOptions,
) -> list[dict[str, object]]:
    if (
        payload.get("schema_version") != REMOTE_BUILD_SCHEMA_VERSION
        or payload.get("status") != "deterministic_tar_zst_triple_validated"
        or payload.get("campaign_id") != options.campaign_id
        or payload.get("bundle_id") != options.bundle_id
        or payload.get("source_git_sha") != options.source_git_sha
        or payload.get("source_manifest_sha256") != options.source_manifest_sha256
        or payload.get("remote_input_pruned") is not False
        or payload.get("remote_staging_pruned") is not False
    ):
        raise OpenEcologyArchiveError("remote build provenance mismatch")
    marker_sha256 = payload.get("marker_sha256")
    _sha256(marker_sha256, field="remote marker SHA256")
    archive_name = payload.get("archive_name")
    expected_archive_name = f"{options.bundle_id}-{str(marker_sha256)[:16]}.tar.zst"
    if not isinstance(archive_name, str) or archive_name != expected_archive_name:
        raise OpenEcologyArchiveError("remote archive name is invalid")
    expected_names = {
        archive_name,
        f"{archive_name}.manifest.json",
        f"{archive_name}.sha256",
    }
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or len(raw_objects) != 3:
        raise OpenEcologyArchiveError(
            "remote build must describe exactly three objects"
        )
    objects: list[dict[str, object]] = []
    seen: set[str] = set()
    expected_staging = PurePosixPath(options.remote_staging_dir)
    for raw in raw_objects:
        if not isinstance(raw, dict):
            raise OpenEcologyArchiveError("remote build object must be a mapping")
        name = raw.get("name")
        source_path = raw.get("source_path")
        size = raw.get("size")
        sha256 = raw.get("sha256")
        parsed_source_path = (
            _absolute_posix_path(source_path, field=f"remote object {name} source path")
            if isinstance(source_path, str)
            else None
        )
        if (
            not isinstance(name, str)
            or name not in expected_names
            or name in seen
            or parsed_source_path is None
            or parsed_source_path.name != name
            or parsed_source_path.parent != expected_staging
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise OpenEcologyArchiveError("remote build object evidence is invalid")
        _sha256(sha256, field=f"remote object {name} SHA256")
        seen.add(name)
        objects.append(
            {
                "name": name,
                "sha256": sha256,
                "size": size,
                "source_path": str(parsed_source_path),
            }
        )
    if seen != expected_names:
        raise OpenEcologyArchiveError("remote build object set is incomplete")
    return sorted(objects, key=lambda row: str(row["name"]))


def _inspect_remote_inventory(
    destination_prefix: str,
    *,
    expected_names: Sequence[str],
    require_complete: bool,
) -> list[dict[str, object]]:
    completed = _run(
        (
            "rclone",
            "lsjson",
            destination_prefix,
            "--max-depth",
            "1",
        ),
        allow_missing_remote=True,
    )
    if completed.returncode != 0:
        payload: object = []
    else:
        try:
            payload = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenEcologyArchiveError("Drive inventory is not valid JSON") from exc
    if not isinstance(payload, list):
        raise OpenEcologyArchiveError("Drive inventory root must be a list")
    allowed = set(expected_names)
    records: list[dict[str, object]] = []
    names: set[str] = set()
    ids: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise OpenEcologyArchiveError("Drive inventory row must be an object")
        name = raw.get("Name")
        inventory_path = raw.get("Path")
        size = raw.get("Size")
        drive_id = raw.get("ID")
        if (
            raw.get("IsDir") is not False
            or raw.get("IsLink") is True
            or not isinstance(name, str)
            or inventory_path != name
            or "/" in name
            or name not in allowed
            or name in names
        ):
            raise OpenEcologyArchiveError(
                "Drive inventory has a duplicate or surplus object name"
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(drive_id, str)
            or _DRIVE_ID_PATTERN.fullmatch(drive_id) is None
            or drive_id in ids
        ):
            raise OpenEcologyArchiveError(
                "Drive inventory has invalid size or duplicate/empty object ID"
            )
        names.add(name)
        ids.add(drive_id)
        records.append({"id": drive_id, "name": name, "size": size})
    if require_complete and names != allowed:
        raise OpenEcologyArchiveError(
            "Drive inventory is not the exact three-object bundle"
        )
    return sorted(records, key=lambda row: str(row["name"]))


def _stream_remote_object(
    *,
    ssh_target: str,
    source_path: str,
    source_size: int,
    destination_path: str,
) -> None:
    reader = (
        "import os,sys\n"
        "p=sys.argv[1]\n"
        "fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))\n"
        "try:\n"
        " s=os.fstat(fd)\n"
        " if not __import__('stat').S_ISREG(s.st_mode) or s.st_nlink!=1:"
        "  raise SystemExit(3)\n"
        " while True:\n"
        "  b=os.read(fd,8*1024*1024)\n"
        "  if not b: break\n"
        "  sys.stdout.buffer.write(b)\n"
        "finally: os.close(fd)\n"
    )
    try:
        ssh_process = subprocess.Popen(
            _ssh_command(ssh_target, ("python3", "-c", reader, source_path)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise OpenEcologyArchiveError(f"cannot start SSH object stream: {exc}") from exc
    if ssh_process.stdout is None:
        ssh_process.kill()
        raise OpenEcologyArchiveError("SSH object stream did not expose stdout")
    try:
        rclone_process = subprocess.Popen(
            (
                "rclone",
                "rcat",
                destination_path,
                "--size",
                str(source_size),
                "--immutable",
            ),
            stdin=ssh_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        ssh_process.kill()
        raise OpenEcologyArchiveError(f"cannot start rclone upload: {exc}") from exc
    ssh_process.stdout.close()
    _, rclone_stderr = rclone_process.communicate()
    _, ssh_stderr = ssh_process.communicate()
    if ssh_process.returncode != 0 or rclone_process.returncode != 0:
        raise OpenEcologyArchiveError(
            "immutable object upload failed: "
            f"ssh_exit={ssh_process.returncode} "
            f"rclone_exit={rclone_process.returncode} "
            f"ssh_stderr={ssh_stderr.decode(errors='replace').strip()!r} "
            f"rclone_stderr={rclone_stderr.decode(errors='replace').strip()!r}"
        )


def _rclone_readback(destination_path: str) -> dict[str, object]:
    try:
        process = subprocess.Popen(
            ("rclone", "cat", destination_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise OpenEcologyArchiveError(f"cannot start Drive readback: {exc}") from exc
    if process.stdout is None:
        process.kill()
        raise OpenEcologyArchiveError("Drive readback did not expose stdout")
    digest = hashlib.sha256()
    size = 0
    while chunk := process.stdout.read(_READ_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        raise OpenEcologyArchiveError(
            "Drive full-byte readback failed: "
            + stderr.decode(errors="replace").strip()
        )
    return {"sha256": digest.hexdigest(), "size": size}


def _run_independent_rclone_check(
    destination_prefix: str,
    *,
    expected: Mapping[str, Mapping[str, object]],
) -> list[str]:
    names = sorted(expected)
    if len(names) != 3:
        raise OpenEcologyArchiveError("rclone check requires exactly three objects")
    sum_bytes = "".join(
        f"{expected[name]['sha256']}  {name}\n" for name in names
    ).encode()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="open-ecology-sha256-",
        suffix=".sum",
    ) as handle:
        handle.write(sum_bytes)
        handle.flush()
        os.fsync(handle.fileno())
        command = (
            "rclone",
            "check",
            handle.name,
            destination_prefix,
            "--checkfile",
            "SHA-256",
            "--combined",
            "-",
        )
        if "--one-way" in command:
            raise OpenEcologyArchiveError("independent rclone check may not be one-way")
        completed = _run(command)
    try:
        rows = [line for line in completed.stdout.decode("utf-8").splitlines() if line]
    except UnicodeDecodeError as exc:
        raise OpenEcologyArchiveError(
            "rclone combined check output is not UTF-8"
        ) from exc
    expected_rows = [f"= {name}" for name in names]
    if sorted(rows) != sorted(expected_rows) or len(rows) != 3:
        raise OpenEcologyArchiveError(
            "rclone check did not emit exactly three canonical match rows"
        )
    return sorted(rows)


def _require_drive_quota(minimum_free_bytes: int) -> int:
    completed = _run(("rclone", "about", "gdrive:", "--json"))
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenEcologyArchiveError("rclone about did not emit valid JSON") from exc
    free = payload.get("free") if isinstance(payload, dict) else None
    if isinstance(free, bool) or not isinstance(free, int) or free < minimum_free_bytes:
        raise OpenEcologyArchiveError(
            "Google Drive free space is below the sealed campaign floor"
        )
    return free


def _require_object_match(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    field: str,
) -> None:
    if observed.get("sha256") != expected.get("sha256") or observed.get(
        "size"
    ) != expected.get("size"):
        raise OpenEcologyArchiveError(f"{field} SHA256 or size mismatch")


def _require_exact_source_binding(
    repository_root: Path,
    *,
    source_git_sha: str,
    source_manifest_sha256: str,
) -> Path:
    repository = _require_canonical_directory(
        repository_root,
        field="repository root",
    )
    if repository != _REPOSITORY_ROOT:
        raise OpenEcologyArchiveError(
            "archive helper is not running from the claimed repository root"
        )
    storage_module = sys.modules["evolution_sim.io.open_ecology_campaign_storage"]
    storage_path = Path(str(storage_module.__file__)).resolve()
    if not storage_path.is_relative_to(repository / "python"):
        raise OpenEcologyArchiveError(
            "open-ecology storage import root differs from exact checkout"
        )
    manifest_module = sys.modules["evolution_sim.io.source_manifest"]
    manifest_module_path = Path(str(manifest_module.__file__)).resolve()
    if not manifest_module_path.is_relative_to(repository / "python"):
        raise OpenEcologyArchiveError(
            "source-manifest import root differs from exact checkout"
        )
    producer_path = repository / "scripts" / "archive_evolution_outputs.py"
    if not producer_path.is_file() or producer_path.is_symlink():
        raise OpenEcologyArchiveError(
            "exact archive_evolution_outputs.py producer is missing"
        )
    observed_head = _git_output(repository, ("rev-parse", "HEAD")).strip()
    if observed_head != source_git_sha:
        raise OpenEcologyArchiveError("exact checkout HEAD differs from source pin")
    observed_branch = _git_output(
        repository,
        ("rev-parse", "--abbrev-ref", "HEAD"),
    ).strip()
    if observed_branch != "HEAD":
        raise OpenEcologyArchiveError(
            "exact checkout must be detached at the source pin"
        )
    status = _git_output(
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    )
    if status:
        raise OpenEcologyArchiveError("exact checkout is dirty")
    for relative_path in (
        PurePosixPath("scripts/archive_open_ecology_campaign.py"),
        PurePosixPath("scripts/archive_evolution_outputs.py"),
    ):
        _require_committed_file_bytes(
            repository,
            source_git_sha=source_git_sha,
            relative_path=relative_path,
        )
    observed_manifest = source_file_hash_manifest(repository).get("aggregate_sha256")
    if observed_manifest != source_manifest_sha256:
        raise OpenEcologyArchiveError("exact checkout source manifest drifted")
    return repository


def _git_output(repository: Path, arguments: Sequence[str]) -> str:
    completed = _run(("git", "-C", str(repository), *arguments))
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpenEcologyArchiveError("git output is not UTF-8") from exc


def _require_committed_file_bytes(
    repository: Path,
    *,
    source_git_sha: str,
    relative_path: PurePosixPath,
) -> None:
    worktree_path = repository / relative_path
    worktree_bytes = _read_regular_file(worktree_path)
    committed = _run(
        (
            "git",
            "-C",
            str(repository),
            "show",
            f"{source_git_sha}:{relative_path.as_posix()}",
        )
    ).stdout
    if worktree_bytes != committed:
        raise OpenEcologyArchiveError(
            f"exact checkout file differs from committed bytes: {relative_path}"
        )


def _prepare_staging_directory(
    path: Path,
    *,
    active: Path,
    bundle: Path,
) -> Path:
    raw = Path(path)
    if raw.exists():
        staging = _require_canonical_directory(raw, field="staging directory")
    else:
        parent = _require_canonical_directory(raw.parent, field="staging parent")
        raw.mkdir(mode=0o700)
        staging = _require_canonical_directory(raw, field="staging directory")
        _fsync_directory(parent)
    _require_separate_local_trees(active, staging)
    _require_separate_local_trees(bundle, staging)
    return staging


def _staging_object_names(path: Path) -> set[str]:
    names: set[str] = set()
    for child in os.scandir(path):
        if child.name in names:
            raise OpenEcologyArchiveError("duplicate staging object name")
        metadata = os.lstat(child.path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OpenEcologyArchiveError(
                "staging directory contains non-regular evidence"
            )
        names.add(child.name)
    return names


def _require_canonical_directory(path: Path, *, field: str) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise OpenEcologyArchiveError(f"{field} does not exist") from exc
    if raw != resolved:
        raise OpenEcologyArchiveError(
            f"{field} raw/resolved identity differs or has a symlink ancestor"
        )
    current = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OpenEcologyArchiveError(
                f"{field} ancestor is a symlink or non-directory"
            )
    if resolved == Path(resolved.anchor):
        raise OpenEcologyArchiveError(f"{field} may not be a filesystem root")
    return resolved


def _require_separate_local_trees(left: Path, right: Path) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise OpenEcologyArchiveError("archive trees must not overlap")


def _trees_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def _absolute_posix_path(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise OpenEcologyArchiveError(f"{field} must be one safe absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value.rstrip("/"):
        raise OpenEcologyArchiveError(f"{field} must be normalized and absolute")
    return path


def _ssh_command(target: str, arguments: Sequence[str]) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        target,
        " ".join(shlex.quote(argument) for argument in arguments),
    )


def _run(
    command: Sequence[str],
    *,
    allow_missing_remote: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            tuple(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise OpenEcologyArchiveError(f"cannot execute {command[0]}: {exc}") from exc
    if completed.returncode == 0:
        return completed
    if (
        allow_missing_remote
        and completed.returncode == 3
        and completed.stdout.strip() in {b"", b"[", b"[]"}
        and b"not found" in completed.stderr.lower()
    ):
        return completed
    raise OpenEcologyArchiveError(
        f"{command[0]} failed with exit {completed.returncode}: "
        + completed.stderr.decode(errors="replace").strip()
    )


def _read_regular_file(path: Path) -> bytes:
    initial = os.lstat(path)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size <= 0
    ):
        raise OpenEcologyArchiveError("evidence object is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not _same_regular_file(initial, metadata):
            raise OpenEcologyArchiveError("evidence object is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            chunks.append(chunk)
        payload = b"".join(chunks)
        finished = os.fstat(descriptor)
        if len(payload) != metadata.st_size or not _same_regular_file(
            metadata,
            finished,
        ):
            raise OpenEcologyArchiveError("evidence object changed while reading")
        return payload
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    initial = os.lstat(path)
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size <= 0
    ):
        raise OpenEcologyArchiveError("evidence object is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not _same_regular_file(initial, opened):
            raise OpenEcologyArchiveError("evidence object changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if not _same_regular_file(opened, finished):
            raise OpenEcologyArchiveError("evidence object changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _same_regular_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_nlink == right.st_nlink == 1
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologyArchiveError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise OpenEcologyArchiveError(f"non-finite JSON constant: {value}")


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise OpenEcologyArchiveError(f"{field} is not a conservative identifier")
    return value


def _git_sha(value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA_PATTERN.fullmatch(value) is None:
        raise OpenEcologyArchiveError("source git SHA must be a full lowercase commit")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise OpenEcologyArchiveError(f"{field} must be lowercase SHA256")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _limits_from_args(args: argparse.Namespace) -> CampaignStorageLimits:
    return CampaignStorageLimits(
        max_campaign_bytes=args.max_campaign_bytes,
        min_campaign_free_bytes=args.min_campaign_free_bytes,
        min_remote_free_bytes=args.min_remote_free_bytes,
        max_entries=args.max_entries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive one sealed open-ecology bundle without pruning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive = subparsers.add_parser("archive")
    archive.add_argument("--ssh-target", required=True)
    archive.add_argument("--remote-repository-root", required=True)
    archive.add_argument("--remote-active-campaign-root", required=True)
    archive.add_argument("--remote-closed-bundle-dir", required=True)
    archive.add_argument("--remote-staging-dir", required=True)
    archive.add_argument("--campaign-id", required=True)
    archive.add_argument("--bundle-id", required=True)
    archive.add_argument("--source-git-sha", required=True)
    archive.add_argument("--source-manifest-sha256", required=True)
    archive.add_argument("--receipt-path", type=Path, required=True)

    remote = subparsers.add_parser("remote-build")
    remote.add_argument("--repository-root", type=Path, required=True)
    remote.add_argument("--active-campaign-root", type=Path, required=True)
    remote.add_argument("--closed-bundle-dir", type=Path, required=True)
    remote.add_argument("--staging-dir", type=Path, required=True)
    remote.add_argument("--campaign-id", required=True)
    remote.add_argument("--bundle-id", required=True)
    remote.add_argument("--source-git-sha", required=True)
    remote.add_argument("--source-manifest-sha256", required=True)

    for command in (archive, remote):
        command.add_argument(
            "--max-campaign-bytes",
            type=int,
            default=CampaignStorageLimits().max_campaign_bytes,
        )
        command.add_argument(
            "--min-campaign-free-bytes",
            type=int,
            default=CampaignStorageLimits().min_campaign_free_bytes,
        )
        command.add_argument(
            "--min-remote-free-bytes",
            type=int,
            default=DEFAULT_REMOTE_FREE_BYTES,
        )
        command.add_argument(
            "--max-entries",
            type=int,
            default=CampaignStorageLimits().max_entries,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        limits = _limits_from_args(args)
        if args.command == "remote-build":
            result = build_remote_closed_bundle_archive(
                repository_root=args.repository_root,
                active_campaign_root=args.active_campaign_root,
                closed_bundle_dir=args.closed_bundle_dir,
                staging_dir=args.staging_dir,
                campaign_id=args.campaign_id,
                bundle_id=args.bundle_id,
                source_git_sha=args.source_git_sha,
                source_manifest_sha256=args.source_manifest_sha256,
                limits=limits,
            )
        else:
            result = execute_remote_archive(
                RemoteArchiveOptions(
                    ssh_target=args.ssh_target,
                    remote_repository_root=args.remote_repository_root,
                    remote_active_campaign_root=args.remote_active_campaign_root,
                    remote_closed_bundle_dir=args.remote_closed_bundle_dir,
                    remote_staging_dir=args.remote_staging_dir,
                    campaign_id=args.campaign_id,
                    bundle_id=args.bundle_id,
                    source_git_sha=args.source_git_sha,
                    source_manifest_sha256=args.source_manifest_sha256,
                    receipt_path=args.receipt_path,
                    limits=limits,
                )
            )
    except (CampaignStorageError, OSError, ValueError) as exc:
        print(f"open-ecology archive failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
