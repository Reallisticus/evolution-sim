"""Ephemeral two-party launch authority for the Phase-A training matrix.

Persisted readiness reports remain evidence, not a capability.  A Mac-side
coordinator statically validates the sealed bundle, independently reexecutes
storage authority, and keeps one authenticated SSH channel open.  The
exact-source GPU guardian fully reexecutes launch authority, then independently
reexecutes D10, throughput, and output-lock authority.  Only an in-memory
capability issued by that live guardian may admit training updates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import base64
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import secrets
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Final
import weakref

import torch

from evolution_sim.io.open_ecology_archive_authority import (
    SEALED_SSH_OPTIONS,
    ArchiveAuthorityError,
    ArchiveToolAuthority,
    load_archive_tool_authority,
    parse_ssh_connection_identity,
    verify_effective_ssh_config,
    verify_local_authority_files,
    verify_pinned_file,
)
from evolution_sim.io.open_ecology_campaign_storage import canonical_json_bytes
from evolution_sim.mind.open_ecology_phase_a import (
    OPEN_ECOLOGY_PHASE_A_CELL_ORDER,
    OpenEcologyPhaseAError,
    _load_strict_json,
    _require_live_source,
    run_open_ecology_phase_a_cell,
    validate_open_ecology_phase_a_launch_authorization,
    validate_open_ecology_phase_a_launch_authorization_static,
    validate_open_ecology_phase_a_preregistration,
)
from evolution_sim.mind.open_ecology_phase_a_readiness import (
    live_verify_campaign_storage_capacity_report,
    live_verify_exact_sha_phase_a_training_and_torch_ci_report,
    live_verify_output_lock_contention_report,
    live_verify_phase_a_training_throughput_report,
)
from evolution_sim.mind.open_ecology_phase_a_qualification import (
    ephemeral_github_credential,
)
from evolution_sim.mind.provenance import stable_payload_digest


TWO_PARTY_PROTOCOL_SCHEMA_VERSION: Final = (
    "mind_v3_open_ecology_phase_a_two_party_guardian_protocol_v1"
)
TWO_PARTY_TRANSCRIPT_SCHEMA_VERSION: Final = (
    "mind_v3_open_ecology_phase_a_two_party_guardian_transcript_v1"
)
MAX_PROTOCOL_FRAME_BYTES: Final = 256 * 1024
MAX_SSH_STDERR_BYTES: Final = 4 * 1024 * 1024
DEFAULT_GUARDIAN_READY_TIMEOUT_SECONDS: Final = 14 * 60 * 60.0
DEFAULT_CELL_TIMEOUT_SECONDS: Final = 24 * 60 * 60.0
_CAPABILITY_BYTES: Final = 32
_REMOTE_CHALLENGE_BYTES: Final = 32
_MAC_GATE_NAME: Final = "storage"
_REMOTE_GATE_NAMES: Final = ("d10", "throughput", "output_lock")
_RUNTIME_BOOTSTRAP_RELATIVE_PATH: Final = (
    "python/evolution_sim/io/open_ecology_runtime_venv_authority.py"
)
_PINNED_STDLIB_EXEC_CODE: Final = r"""
import hashlib
import os
import stat
import sys

path = sys.argv[1]
expected = sys.argv[2]
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("runtime bootstrap is not regular")
    digest = hashlib.sha256()
    chunks = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        chunks.append(block)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_mode, value.st_size,
    value.st_mtime_ns, value.st_ctime_ns,
)
if identity(before) != identity(after) or digest.hexdigest() != expected:
    raise SystemExit("runtime bootstrap identity or SHA256 drifted")
source = b"".join(chunks)
sys.argv = [path, *sys.argv[3:]]
namespace = {
    "__builtins__": __builtins__,
    "__file__": path,
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(source, path, "exec"), namespace, namespace)
""".strip()
_ISSUER = object()
_FULL_VALIDATION_ISSUER = object()
_ISSUED_CAPABILITIES: weakref.WeakSet[LivePhaseAUpdateAdmission] = weakref.WeakSet()
_ISSUED_FULL_AUTHORITY_RECEIPTS: weakref.WeakSet[
    _FullAuthorityValidationReceipt
] = weakref.WeakSet()


class OpenEcologyTwoPartyAuthorityError(RuntimeError):
    """Two-party launch authority was absent, stale, replayed, or lost."""


@dataclass(frozen=True, slots=True)
class PhaseAAuthorityBindings:
    source_git_sha: str
    source_manifest_sha256: str
    preregistration_digest: str
    evidence_index_digest: str
    launch_authorization_digest: str
    runtime_venv_authority_sha256: str
    ssh_target: str

    def as_dict(self) -> dict[str, str]:
        return {
            "evidence_index_digest": self.evidence_index_digest,
            "launch_authorization_digest": self.launch_authorization_digest,
            "preregistration_digest": self.preregistration_digest,
            "runtime_venv_authority_sha256": self.runtime_venv_authority_sha256,
            "source_git_sha": self.source_git_sha,
            "source_manifest_sha256": self.source_manifest_sha256,
            "ssh_target": self.ssh_target,
        }


class _FullAuthorityValidationReceipt:
    """Opaque one-shot proof that this process ran full authority validation."""

    __slots__ = (
        "_authorization_path",
        "_authorization_time",
        "_bindings",
        "_consumed",
        "_guardian_pgid",
        "_guardian_pid",
        "_issuer",
        "_preregistration",
        "_report_paths",
        "__weakref__",
    )

    def __init__(
        self,
        issuer: object,
        *,
        bindings: PhaseAAuthorityBindings,
        preregistration: Mapping[str, object],
        authorization_time: datetime,
        authorization_path: Path,
        report_paths: Mapping[str, Path],
    ) -> None:
        if issuer is not _FULL_VALIDATION_ISSUER:
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipts cannot be constructed by callers"
            )
        if set(report_paths) != {
            "d10",
            "throughput",
            "storage",
            "output_lock",
        }:
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipt report topology drifted"
            )
        cloned_preregistration = json.loads(canonical_json_bytes(preregistration))
        if not isinstance(cloned_preregistration, dict):
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipt preregistration is malformed"
            )
        self._issuer = issuer
        self._bindings = bindings
        self._preregistration = cloned_preregistration
        self._authorization_time = authorization_time
        self._authorization_path = authorization_path.resolve()
        self._report_paths = tuple(
            (name, Path(report_paths[name]).resolve())
            for name in ("d10", "throughput", "storage", "output_lock")
        )
        self._guardian_pid = os.getpid()
        self._guardian_pgid = os.getpgrp()
        self._consumed = False
        _ISSUED_FULL_AUTHORITY_RECEIPTS.add(self)

    def __reduce__(self) -> object:
        raise TypeError("full authority validation receipt cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("full authority validation receipt cannot be serialized")

    def __copy__(self) -> object:
        raise TypeError("full authority validation receipt cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("full authority validation receipt cannot be copied")

    def consume(
        self,
        *,
        expected_bindings: PhaseAAuthorityBindings,
    ) -> tuple[dict[str, object], datetime, dict[str, Path]]:
        if (
            self._consumed
            or self not in _ISSUED_FULL_AUTHORITY_RECEIPTS
            or self._issuer is not _FULL_VALIDATION_ISSUER
        ):
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipt is closed or replayed"
            )
        self._consumed = True
        _ISSUED_FULL_AUTHORITY_RECEIPTS.discard(self)
        if os.getpid() != self._guardian_pid or os.getpgrp() != self._guardian_pgid:
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipt crossed its process boundary"
            )
        if self._bindings != expected_bindings:
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipt binding drifted"
            )
        if not self._authorization_path.is_absolute():
            raise OpenEcologyTwoPartyAuthorityError(
                "full authority validation receipt authorization path drifted"
            )
        return (
            dict(self._preregistration),
            self._authorization_time,
            dict(self._report_paths),
        )

class GuardianChannelLiveness:
    """One-way liveness latch shared by the guardian and update capability."""

    def __init__(self, *, terminate: Callable[[str], None]) -> None:
        self._alive = threading.Event()
        self._alive.set()
        self._terminate = terminate
        self._reason: str | None = None
        self._clean_shutdown = False
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._alive.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def fail(self, reason: str) -> None:
        with self._lock:
            if self._clean_shutdown:
                return
            if not self._alive.is_set():
                return
            self._reason = reason
            self._alive.clear()
        self._terminate(reason)

    def require_alive(self) -> None:
        if not self._alive.is_set():
            raise OpenEcologyTwoPartyAuthorityError(
                f"two-party guardian channel is not alive: {self._reason or 'unknown'}"
            )

    def allow_clean_shutdown(self) -> None:
        with self._lock:
            self._clean_shutdown = True
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, 0, 0, 0, 0) != 0:
                errno = ctypes.get_errno()
                raise OpenEcologyTwoPartyAuthorityError(
                    f"failed to disarm guardian parent-death signal: errno={errno}"
                )


class LivePhaseAUpdateAdmission:
    """Non-pickleable, process-local update authority issued after live gates."""

    __slots__ = (
        "_active_cell",
        "_bindings",
        "_channel",
        "_closed",
        "_guardian_pgid",
        "_guardian_pid",
        "_issuer",
        "_launch_authorization_digest",
        "_source_probe",
        "__weakref__",
    )

    def __init__(
        self,
        issuer: object,
        *,
        bindings: PhaseAAuthorityBindings,
        channel: GuardianChannelLiveness,
        source_probe: Callable[[], None],
        launch_authorization_digest: str,
    ) -> None:
        if issuer is not _ISSUER:
            raise OpenEcologyTwoPartyAuthorityError(
                "live update admission cannot be constructed by callers"
            )
        self._issuer = issuer
        self._bindings = bindings
        self._channel = channel
        self._source_probe = source_probe
        self._launch_authorization_digest = launch_authorization_digest
        self._guardian_pid = os.getpid()
        self._guardian_pgid = os.getpgrp()
        self._active_cell: tuple[str, int] | None = None
        self._closed = False
        _ISSUED_CAPABILITIES.add(self)

    def __reduce__(self) -> object:
        raise TypeError("live Phase-A update admission cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("live Phase-A update admission cannot be serialized")

    def begin_cell(self, *, cell_id: str, learner_index: int) -> None:
        self._require_process_and_channel()
        if self._active_cell is not None:
            raise OpenEcologyTwoPartyAuthorityError(
                "a guardian capability cannot admit concurrent Phase-A cells"
            )
        _require_cell_identity(cell_id, learner_index)
        self._source_probe()
        self._active_cell = (cell_id, learner_index)

    def finish_cell(self, *, cell_id: str, learner_index: int) -> None:
        self._require_process_and_channel()
        if self._active_cell != (cell_id, learner_index):
            raise OpenEcologyTwoPartyAuthorityError(
                "guardian cell completion identity drifted"
            )
        self._source_probe()
        self._active_cell = None

    def admit(
        self,
        *,
        preregistration: Mapping[str, object],
        launch_authorization_digest: str,
        cell_id: str,
        learner_index: int,
        stage: str,
        update_index: int | None,
    ) -> None:
        self._require_process_and_channel()
        if self._active_cell != (cell_id, learner_index):
            raise OpenEcologyTwoPartyAuthorityError(
                "Phase-A update is outside the guardian's active cell"
            )
        if preregistration.get("exact_digest") != self._bindings.preregistration_digest:
            raise OpenEcologyTwoPartyAuthorityError(
                "guardian preregistration digest binding drifted"
            )
        source = _strict_mapping(preregistration.get("source"), field="source")
        if (
            source.get("commit") != self._bindings.source_git_sha
            or source.get("manifest_sha256") != self._bindings.source_manifest_sha256
            or launch_authorization_digest != self._launch_authorization_digest
        ):
            raise OpenEcologyTwoPartyAuthorityError(
                "guardian source or launch-authorization binding drifted"
            )
        allowed_stages = {
            "before_activation",
            "after_activation",
            "before_update",
            "before_update_commit",
            "before_terminal_publish",
            "terminal_resume",
        }
        if stage not in allowed_stages:
            raise OpenEcologyTwoPartyAuthorityError(
                f"unknown Phase-A guardian admission stage {stage!r}"
            )
        if stage in {"before_update", "before_update_commit"}:
            if (
                isinstance(update_index, bool)
                or not isinstance(update_index, int)
                or update_index < 0
            ):
                raise OpenEcologyTwoPartyAuthorityError(
                    "update admission requires a non-negative update index"
                )
        elif update_index is not None:
            raise OpenEcologyTwoPartyAuthorityError(
                "non-update admission cannot carry an update index"
            )
        self._source_probe()

    def close(self) -> None:
        self._closed = True
        self._active_cell = None
        _ISSUED_CAPABILITIES.discard(self)

    def _require_process_and_channel(self) -> None:
        if (
            self._closed
            or self not in _ISSUED_CAPABILITIES
            or self._issuer is not _ISSUER
        ):
            raise OpenEcologyTwoPartyAuthorityError(
                "Phase-A live guardian capability is closed or replayed"
            )
        if os.getpid() != self._guardian_pid or os.getpgrp() != self._guardian_pgid:
            raise OpenEcologyTwoPartyAuthorityError(
                "Phase-A live guardian capability crossed its process boundary"
            )
        self._channel.require_alive()


def require_live_phase_a_update_admission(
    capability: object | None,
    *,
    preregistration: Mapping[str, object],
    launch_authorization_digest: str,
    cell_id: str,
    learner_index: int,
    stage: str,
    update_index: int | None = None,
) -> None:
    """Fail closed unless ``capability`` is a currently issued live authority."""

    if not isinstance(capability, LivePhaseAUpdateAdmission):
        raise OpenEcologyPhaseAError(
            "Phase-A training requires a live two-party guardian; persisted "
            "launch JSON alone cannot authorize updates"
        )
    try:
        capability.admit(
            preregistration=preregistration,
            launch_authorization_digest=launch_authorization_digest,
            cell_id=cell_id,
            learner_index=learner_index,
            stage=stage,
            update_index=update_index,
        )
    except OpenEcologyTwoPartyAuthorityError as error:
        raise OpenEcologyPhaseAError(str(error)) from error


class MacStorageAuthority:
    """Mac-local storage authority that can be acquired only once."""

    def __init__(
        self,
        *,
        bindings: PhaseAAuthorityBindings,
        source_probe: Callable[[], None],
        verifier: Callable[
            [Path, Mapping[str, object], datetime | None],
            Mapping[str, object],
        ] = live_verify_campaign_storage_capacity_report,
    ) -> None:
        self.bindings = bindings
        self._source_probe = source_probe
        self._verifier = verifier
        self._attempted = False
        self._secret: bytearray | None = None
        self.facts_digest: str | None = None

    def acquire(
        self,
        *,
        report_path: Path,
        preregistration: Mapping[str, object],
        authorization_time: datetime | None,
    ) -> None:
        if self._attempted:
            raise OpenEcologyTwoPartyAuthorityError(
                "Mac storage authority may be verified exactly once per coordinator"
            )
        self._attempted = True
        self._source_probe()
        reconstructed = self._verifier(
            report_path,
            preregistration,
            authorization_time,
        )
        self._source_probe()
        self.facts_digest = stable_payload_digest(reconstructed)
        self._secret = bytearray(secrets.token_bytes(_CAPABILITY_BYTES))

    def secret_bytes(self) -> bytes:
        if self._secret is None:
            raise OpenEcologyTwoPartyAuthorityError(
                "Mac storage authority has not been acquired"
            )
        return bytes(self._secret)

    def close(self) -> None:
        if self._secret is not None:
            for index in range(len(self._secret)):
                self._secret[index] = 0
        self._secret = None


class RemoteGuardianAuthority:
    """GPU-local live gates entered only through one full-validation receipt."""

    def __init__(
        self,
        *,
        bindings: PhaseAAuthorityBindings,
        channel: GuardianChannelLiveness,
        source_probe: Callable[[], None],
        host_observer: Callable[[Path], Mapping[str, object]],
        verifiers: Mapping[
            str,
            Callable[
                [Path, Mapping[str, object], datetime | None],
                Mapping[str, object],
            ],
        ]
        | None = None,
    ) -> None:
        def verify_throughput(
            report_path: Path,
            preregistration: Mapping[str, object],
            authorization_time: datetime | None,
        ) -> Mapping[str, object]:
            return live_verify_phase_a_training_throughput_report(
                report_path,
                preregistration,
                authorization_time,
                host_observer=host_observer,
            )

        self.bindings = bindings
        self.channel = channel
        self._source_probe = source_probe
        self._verifiers = dict(
            verifiers
            or {
                "d10": live_verify_exact_sha_phase_a_training_and_torch_ci_report,
                "throughput": verify_throughput,
                "output_lock": live_verify_output_lock_contention_report,
            }
        )
        self._attempted = False
        self.verifier_digests: dict[str, str] = {}

    def activate(
        self,
        *,
        full_validation_receipt: object,
        github_token: bytearray,
    ) -> LivePhaseAUpdateAdmission:
        if self._attempted:
            raise OpenEcologyTwoPartyAuthorityError(
                "remote operational gates may run exactly once per guardian lifetime"
            )
        self._attempted = True
        if type(full_validation_receipt) is not _FullAuthorityValidationReceipt:
            raise OpenEcologyTwoPartyAuthorityError(
                "remote guardian requires a full authority validation receipt"
            )
        preregistration, authorization_time, all_report_paths = (
            full_validation_receipt.consume(expected_bindings=self.bindings)
        )
        report_paths = {
            name: all_report_paths[name] for name in _REMOTE_GATE_NAMES
        }
        if tuple(self._verifiers) != _REMOTE_GATE_NAMES:
            raise OpenEcologyTwoPartyAuthorityError(
                "remote guardian verifier topology drifted"
            )
        if set(report_paths) != set(_REMOTE_GATE_NAMES):
            raise OpenEcologyTwoPartyAuthorityError(
                "remote guardian report topology drifted"
            )
        self.channel.require_alive()
        self._source_probe()
        for gate_name in _REMOTE_GATE_NAMES:
            if gate_name == "d10":
                with ephemeral_github_credential(github_token):
                    reconstructed = self._verifiers[gate_name](
                        report_paths[gate_name],
                        preregistration,
                        authorization_time,
                    )
                if any(github_token):
                    raise OpenEcologyTwoPartyAuthorityError(
                        "one-shot GitHub credential was not zeroed after D10"
                    )
            else:
                reconstructed = self._verifiers[gate_name](
                    report_paths[gate_name],
                    preregistration,
                    authorization_time,
                )
            self.channel.require_alive()
            self.verifier_digests[gate_name] = stable_payload_digest(reconstructed)
        self._source_probe()
        return LivePhaseAUpdateAdmission(
            _ISSUER,
            bindings=self.bindings,
            channel=self.channel,
            source_probe=self._source_probe,
            launch_authorization_digest=self.bindings.launch_authorization_digest,
        )


def load_phase_a_authority_bundle(
    *,
    preregistration_path: Path,
    launch_authorization_path: Path,
    expected_bindings: PhaseAAuthorityBindings | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    datetime,
    PhaseAAuthorityBindings,
    dict[str, Path],
]:
    """Fully reexecute authority and resolve its exact operational reports."""

    return _load_phase_a_authority_bundle(
        preregistration_path=preregistration_path,
        launch_authorization_path=launch_authorization_path,
        expected_bindings=expected_bindings,
        authorization_validator=(
            validate_open_ecology_phase_a_launch_authorization
        ),
    )


def _load_phase_a_authority_bundle_for_remote(
    *,
    preregistration_path: Path,
    launch_authorization_path: Path,
    expected_bindings: PhaseAAuthorityBindings,
) -> tuple[
    dict[str, object],
    dict[str, object],
    datetime,
    PhaseAAuthorityBindings,
    dict[str, Path],
    _FullAuthorityValidationReceipt,
]:
    """Mint remote launch proof only after the full authority path succeeds."""

    bundle = load_phase_a_authority_bundle(
        preregistration_path=preregistration_path,
        launch_authorization_path=launch_authorization_path,
        expected_bindings=expected_bindings,
    )
    preregistration, _authorization, authorization_time, bindings, reports = bundle
    receipt = _FullAuthorityValidationReceipt(
        _FULL_VALIDATION_ISSUER,
        bindings=bindings,
        preregistration=preregistration,
        authorization_time=authorization_time,
        authorization_path=launch_authorization_path,
        report_paths=reports,
    )
    return (*bundle, receipt)


def _load_phase_a_authority_bundle_static(
    *,
    preregistration_path: Path,
    launch_authorization_path: Path,
    expected_bindings: PhaseAAuthorityBindings | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    datetime,
    PhaseAAuthorityBindings,
    dict[str, Path],
]:
    """Load the Mac's static half without host-specific proof reexecution."""

    return _load_phase_a_authority_bundle(
        preregistration_path=preregistration_path,
        launch_authorization_path=launch_authorization_path,
        expected_bindings=expected_bindings,
        authorization_validator=(
            validate_open_ecology_phase_a_launch_authorization_static
        ),
    )


def _load_phase_a_authority_bundle(
    *,
    preregistration_path: Path,
    launch_authorization_path: Path,
    expected_bindings: PhaseAAuthorityBindings | None,
    authorization_validator: Callable[..., None],
) -> tuple[
    dict[str, object],
    dict[str, object],
    datetime,
    PhaseAAuthorityBindings,
    dict[str, Path],
]:
    """Resolve one bundle after the caller selected a closed validation path."""

    preregistration = _load_strict_json(preregistration_path)
    authorization = _load_strict_json(launch_authorization_path)
    validate_open_ecology_phase_a_preregistration(preregistration)
    authorization_validator(
        authorization,
        preregistration=preregistration,
        authorization_path=launch_authorization_path,
    )
    source = _strict_mapping(preregistration.get("source"), field="source")
    index_reference = _strict_mapping(
        _strict_mapping(
            authorization.get("evidence_index"),
            field="authorization.evidence_index",
        ).get("file"),
        field="authorization.evidence_index.file",
    )
    root = launch_authorization_path.resolve().parent
    index_path = _resolve_regular_relative(
        root,
        index_reference.get("relative_path"),
        field="evidence index",
    )
    evidence_index = _load_strict_json(index_path)
    indexed = _strict_mapping(
        authorization.get("evidence_index"),
        field="authorization.evidence_index",
    )
    if evidence_index.get("exact_digest") != indexed.get("exact_digest"):
        raise OpenEcologyTwoPartyAuthorityError(
            "launch authorization evidence-index digest drifted"
        )
    bindings = PhaseAAuthorityBindings(
        source_git_sha=_git_sha(source.get("commit"), field="source commit"),
        source_manifest_sha256=_sha256(
            source.get("manifest_sha256"),
            field="source manifest",
        ),
        preregistration_digest=_sha256(
            preregistration.get("exact_digest"),
            field="preregistration digest",
        ),
        evidence_index_digest=_sha256(
            evidence_index.get("exact_digest"),
            field="evidence index digest",
        ),
        launch_authorization_digest=_sha256(
            authorization.get("exact_digest"),
            field="launch authorization digest",
        ),
        runtime_venv_authority_sha256=(
            expected_bindings.runtime_venv_authority_sha256
            if expected_bindings is not None
            else "0" * 64
        ),
        ssh_target=(
            expected_bindings.ssh_target if expected_bindings is not None else "unbound"
        ),
    )
    if expected_bindings is not None and bindings != expected_bindings:
        raise OpenEcologyTwoPartyAuthorityError(
            "Phase-A authority bundle differs from the live session bindings"
        )
    dependency_rows = evidence_index.get("dependency_reports")
    if not isinstance(dependency_rows, list):
        raise OpenEcologyTwoPartyAuthorityError(
            "evidence index dependency reports are malformed"
        )
    d10_matches: list[Mapping[str, object]] = []
    for raw_dependency in dependency_rows:
        dependency = _strict_mapping(raw_dependency, field="dependency report")
        if dependency.get("dependency_id") != "readiness_dependency_10":
            continue
        dependency_reports = dependency.get("reports")
        if not isinstance(dependency_reports, list):
            raise OpenEcologyTwoPartyAuthorityError(
                "D10 dependency report list is malformed"
            )
        d10_matches.extend(
            _strict_mapping(raw_report, field="D10 report")
            for raw_report in dependency_reports
            if _strict_mapping(
                raw_report,
                field="D10 report",
            ).get("evidence_kind")
            == "exact_sha_phase_a_training_and_torch_ci"
        )
    if len(d10_matches) != 1:
        raise OpenEcologyTwoPartyAuthorityError(
            "Phase-A authority requires exactly one dependency-10 D10 report"
        )
    d10_reference = _strict_mapping(
        d10_matches[0].get("file"),
        field="d10.file",
    )
    reports: dict[str, Path] = {
        "d10": _resolve_regular_relative(
            root,
            d10_reference.get("relative_path"),
            field="d10 report",
        )
    }
    rows = evidence_index.get("operational_reports")
    if not isinstance(rows, list):
        raise OpenEcologyTwoPartyAuthorityError(
            "evidence index operational reports are malformed"
        )
    kind_to_gate = {
        "phase_a_training_throughput": "throughput",
        "campaign_storage_capacity": "storage",
        "output_lock_contention": "output_lock",
    }
    for row in rows:
        entry = _strict_mapping(row, field="operational report")
        report = _strict_mapping(entry.get("report"), field="operational report.file")
        kind = report.get("evidence_kind")
        gate = kind_to_gate.get(kind) if isinstance(kind, str) else None
        if gate is None or gate in reports:
            raise OpenEcologyTwoPartyAuthorityError(
                "operational report kind is unknown or duplicated"
            )
        reference = _strict_mapping(report.get("file"), field=f"{gate}.file")
        reports[gate] = _resolve_regular_relative(
            root,
            reference.get("relative_path"),
            field=f"{gate} report",
        )
    if set(reports) != {"d10", "throughput", "storage", "output_lock"}:
        raise OpenEcologyTwoPartyAuthorityError(
            "Phase-A authority bundle lacks the complete operational report set"
        )
    authorization_time = _parse_utc(authorization.get("authorized_at_utc"))
    return preregistration, authorization, authorization_time, bindings, reports


def run_remote_guardian_session(
    *,
    stdin: BinaryIO,
    stdout: BinaryIO,
    preregistration_path: Path,
    launch_authorization_path: Path,
    output_root: Path,
    expected_bindings: PhaseAAuthorityBindings,
    channel: GuardianChannelLiveness,
    host_observer: Callable[[Path], Mapping[str, object]],
    device: torch.device | str = "cuda",
    run_cell: Callable[..., Mapping[str, object]] = run_open_ecology_phase_a_cell,
    remote_authority_factory: Callable[..., RemoteGuardianAuthority] = (
        RemoteGuardianAuthority
    ),
) -> dict[str, object]:
    """Serve one authenticated session; EOF invalidates all update authority."""

    remote_pid = os.getpid()
    remote_pgid = os.getpgrp()
    remote_challenge = bytearray(secrets.token_bytes(_REMOTE_CHALLENGE_BYTES))
    _write_plain_frame(
        stdout,
        {
            "bindings": expected_bindings.as_dict(),
            "frame_type": "challenge",
            "remote_challenge": base64.b64encode(remote_challenge).decode("ascii"),
            "remote_pgid": remote_pgid,
            "remote_pid": remote_pid,
            "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
            "sequence": -1,
        },
    )
    messages: queue.Queue[dict[str, object] | BaseException | None] = queue.Queue()
    reader = threading.Thread(
        target=_protocol_reader,
        kwargs={"stream": stdin, "messages": messages, "channel": channel},
        name="open-ecology-guardian-stdin",
        daemon=True,
    )
    reader.start()
    hello = _next_message(messages, channel=channel)
    _require_frame_type(hello, "hello")
    secret = _decode_session_secret(hello.get("session_secret"))
    _verify_frame_authentication(hello, secret=secret, expected_sequence=0)
    github_token = _decode_github_token(hello.get("github_token"))
    echoed_challenge = _decode_remote_challenge(hello.get("remote_challenge"))
    hello.pop("session_secret", None)
    hello.pop("github_token", None)
    hello.pop("remote_challenge", None)
    try:
        if not hmac.compare_digest(remote_challenge, echoed_challenge):
            raise OpenEcologyTwoPartyAuthorityError(
                "coordinator hello did not bind the fresh remote challenge"
            )
    finally:
        _scrub_bytearray(echoed_challenge)
        _scrub_bytearray(remote_challenge)
    hello_bindings = _bindings_from_mapping(hello.get("bindings"))
    if hello_bindings != expected_bindings:
        raise OpenEcologyTwoPartyAuthorityError(
            "coordinator hello source or SSH endpoint binding drifted"
        )
    if hello.get("mac_storage_ready") is not True:
        raise OpenEcologyTwoPartyAuthorityError(
            "remote guardian requires a live Mac storage authority"
        )
    _sha256(hello.get("mac_storage_facts_digest"), field="Mac storage facts digest")
    try:
        (
            preregistration,
            _authorization,
            _authorization_time,
            bindings,
            _reports,
            full_validation_receipt,
        ) = _load_phase_a_authority_bundle_for_remote(
            preregistration_path=preregistration_path,
            launch_authorization_path=launch_authorization_path,
            expected_bindings=expected_bindings,
        )
        if bindings != expected_bindings:
            raise OpenEcologyTwoPartyAuthorityError(
                "remote bundle bindings changed before live verification"
            )
        remote = remote_authority_factory(
            bindings=bindings,
            channel=channel,
            source_probe=lambda: _require_live_source(preregistration),
            host_observer=host_observer,
        )
        capability = remote.activate(
            full_validation_receipt=full_validation_receipt,
            github_token=github_token,
        )
    finally:
        _scrub_bytearray(github_token)
    ready = {
        "bindings": bindings.as_dict(),
        "frame_type": "ready",
        "remote_pgid": remote_pgid,
        "remote_pid": remote_pid,
        "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
        "sequence": 0,
        "verifier_counts": {name: 1 for name in _REMOTE_GATE_NAMES},
        "verifier_digests": dict(remote.verifier_digests),
    }
    _write_authenticated_frame(stdout, ready, secret=secret)
    completed: list[dict[str, object]] = []
    try:
        for sequence, (cell_id, learner_index) in enumerate(
            _canonical_phase_a_matrix(),
            start=1,
        ):
            request = _next_message(messages, channel=channel)
            _require_frame_type(request, "run_cell")
            _verify_frame_authentication(
                request,
                secret=secret,
                expected_sequence=sequence,
            )
            if (
                request.get("cell_id") != cell_id
                or request.get("learner_index") != learner_index
                or request.get("resume") is not True
            ):
                raise OpenEcologyTwoPartyAuthorityError(
                    "guardian run request is not the canonical resumable matrix order"
                )
            channel.require_alive()
            capability.begin_cell(cell_id=cell_id, learner_index=learner_index)
            try:
                report = run_cell(
                    preregistration,
                    expected_preregistration_digest=(bindings.preregistration_digest),
                    launch_authorization_path=launch_authorization_path,
                    cell_id=cell_id,
                    learner_index=learner_index,
                    output_root=output_root,
                    device=device,
                    resume=True,
                    live_launch_capability=capability,
                )
            finally:
                capability.finish_cell(
                    cell_id=cell_id,
                    learner_index=learner_index,
                )
            result = {
                "cell_id": cell_id,
                "exact_digest": _sha256(
                    report.get("exact_digest"),
                    field="cell report digest",
                ),
                "frame_type": "cell_complete",
                "learner_index": learner_index,
                "run_id": str(report.get("run_id")),
                "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                "sequence": sequence,
            }
            completed.append(dict(result))
            _write_authenticated_frame(stdout, result, secret=secret)
        stop_sequence = len(_canonical_phase_a_matrix()) + 1
        stop = _next_message(messages, channel=channel)
        _require_frame_type(stop, "stop")
        _verify_frame_authentication(
            stop,
            secret=secret,
            expected_sequence=stop_sequence,
        )
        capability.close()
        channel.allow_clean_shutdown()
        final = {
            "completed_cell_count": len(completed),
            "frame_type": "stopped",
            "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
            "sequence": stop_sequence,
        }
        _write_authenticated_frame(stdout, final, secret=secret)
        return {
            "completed": completed,
            "remote_pgid": remote_pgid,
            "remote_pid": remote_pid,
            "verifier_digests": dict(remote.verifier_digests),
        }
    finally:
        capability.close()
        secret = b"\x00" * len(secret)


def run_mac_coordinator(
    *,
    local_preregistration_path: Path,
    local_launch_authorization_path: Path,
    remote_preregistration_path: Path,
    remote_launch_authorization_path: Path,
    remote_output_root: Path,
    archive_authority_path: Path,
    expected_archive_authority_sha256: str,
    remote_runtime_venv_authority_path: Path,
    expected_remote_runtime_venv_authority_sha256: str,
    transcript_path: Path,
    github_token: bytearray,
    device: str = "cuda",
    ready_timeout_seconds: float = DEFAULT_GUARDIAN_READY_TIMEOUT_SECONDS,
    cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    storage_authority_factory: Callable[..., MacStorageAuthority] = (
        MacStorageAuthority
    ),
) -> dict[str, object]:
    """Run the fixed matrix while Mac authority and one SSH channel stay alive."""

    authority = load_archive_tool_authority(
        archive_authority_path,
        expected_sha256=expected_archive_authority_sha256,
    )
    expected_bindings = _bindings_from_archive_and_bundle(
        authority=authority,
        preregistration_path=local_preregistration_path,
        launch_authorization_path=local_launch_authorization_path,
        runtime_venv_authority_sha256=(expected_remote_runtime_venv_authority_sha256),
    )
    preregistration, _authorization, authorization_time, bindings, reports = (
        _load_phase_a_authority_bundle_static(
            preregistration_path=local_preregistration_path,
            launch_authorization_path=local_launch_authorization_path,
            expected_bindings=expected_bindings,
        )
    )
    storage = storage_authority_factory(
        bindings=bindings,
        source_probe=lambda: _require_live_source(preregistration),
    )
    ssh_process: subprocess.Popen[bytes] | None = None
    stderr_capture: _BoundedStderrCapture | None = None
    completed: list[dict[str, object]] = []
    try:
        _require_github_token(github_token)
        _revalidate_local_guardian_authority(authority)
        command = _remote_guardian_ssh_command(
            authority=authority,
            remote_runtime_venv_authority_path=(remote_runtime_venv_authority_path),
            remote_preregistration_path=remote_preregistration_path,
            remote_launch_authorization_path=remote_launch_authorization_path,
            remote_output_root=remote_output_root,
            bindings=bindings,
            device=device,
        )
        ssh_process = popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=_coordinator_environment(),
        )
        if (
            ssh_process.stdin is None
            or ssh_process.stdout is None
            or ssh_process.stderr is None
        ):
            raise OpenEcologyTwoPartyAuthorityError(
                "SSH guardian did not expose all protocol pipes"
            )
        stderr_capture = _BoundedStderrCapture(
            ssh_process.stderr,
            on_error=lambda _error: _terminate_local_ssh_process_group(ssh_process),
        )
        stderr_capture.start()
        endpoint = _wait_for_authenticated_endpoint(
            stderr_capture,
            expected=authority,
            deadline=time.monotonic() + 30.0,
        )
        storage.acquire(
            report_path=reports[_MAC_GATE_NAME],
            preregistration=preregistration,
            authorization_time=authorization_time,
        )
        challenge = _read_guardian_challenge_with_timeout(
            ssh_process.stdout,
            expected_bindings=bindings,
            timeout_seconds=ready_timeout_seconds,
        )
        secret = storage.secret_bytes()
        hello = {
            "bindings": bindings.as_dict(),
            "coordinator_pgid": os.getpgrp(),
            "coordinator_pid": os.getpid(),
            "frame_type": "hello",
            "github_token": base64.b64encode(github_token).decode("ascii"),
            "mac_storage_facts_digest": storage.facts_digest,
            "mac_storage_ready": True,
            "remote_challenge": challenge["remote_challenge"],
            "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
            "sequence": 0,
            "session_secret": base64.b64encode(secret).decode("ascii"),
        }
        _write_hello_and_scrub_token(
            ssh_process.stdin,
            hello=hello,
            secret=secret,
            github_token=github_token,
        )
        ready = _read_authenticated_frame_with_timeout(
            ssh_process.stdout,
            secret=secret,
            expected_sequence=0,
            timeout_seconds=ready_timeout_seconds,
        )
        _require_frame_type(ready, "ready")
        if _bindings_from_mapping(ready.get("bindings")) != bindings:
            raise OpenEcologyTwoPartyAuthorityError("remote ready binding drifted")
        if ready.get("verifier_counts") != {name: 1 for name in _REMOTE_GATE_NAMES}:
            raise OpenEcologyTwoPartyAuthorityError(
                "remote guardian did not run each expensive verifier exactly once"
            )
        remote_pid = _positive_int(ready.get("remote_pid"), field="remote PID")
        remote_pgid = _positive_int(ready.get("remote_pgid"), field="remote PGID")
        if (
            remote_pid != challenge["remote_pid"]
            or remote_pgid != challenge["remote_pgid"]
        ):
            raise OpenEcologyTwoPartyAuthorityError(
                "remote guardian process identity changed after its challenge"
            )
        for sequence, (cell_id, learner_index) in enumerate(
            _canonical_phase_a_matrix(),
            start=1,
        ):
            _revalidate_local_guardian_authority(authority)
            _require_live_source(preregistration)
            request = {
                "cell_id": cell_id,
                "frame_type": "run_cell",
                "learner_index": learner_index,
                "resume": True,
                "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                "sequence": sequence,
            }
            _write_authenticated_frame(ssh_process.stdin, request, secret=secret)
            response = _read_authenticated_frame_with_timeout(
                ssh_process.stdout,
                secret=secret,
                expected_sequence=sequence,
                timeout_seconds=cell_timeout_seconds,
            )
            _require_frame_type(response, "cell_complete")
            if (
                response.get("cell_id") != cell_id
                or response.get("learner_index") != learner_index
            ):
                raise OpenEcologyTwoPartyAuthorityError(
                    "remote cell response identity drifted"
                )
            completed.append(
                {
                    "cell_id": cell_id,
                    "exact_digest": _sha256(
                        response.get("exact_digest"),
                        field="cell report digest",
                    ),
                    "learner_index": learner_index,
                    "run_id": str(response.get("run_id")),
                }
            )
            _write_compact_transcript(
                transcript_path,
                bindings=bindings,
                endpoint=endpoint.receipt_record(),
                coordinator_pid=os.getpid(),
                coordinator_pgid=os.getpgrp(),
                remote_pid=remote_pid,
                remote_pgid=remote_pgid,
                completed=completed,
                storage_facts_digest=storage.facts_digest,
                remote_verifier_digests=_strict_mapping(
                    ready.get("verifier_digests"),
                    field="remote verifier digests",
                ),
                complete=False,
            )
        stop_sequence = len(_canonical_phase_a_matrix()) + 1
        _write_authenticated_frame(
            ssh_process.stdin,
            {
                "frame_type": "stop",
                "schema_version": TWO_PARTY_PROTOCOL_SCHEMA_VERSION,
                "sequence": stop_sequence,
            },
            secret=secret,
        )
        stopped = _read_authenticated_frame_with_timeout(
            ssh_process.stdout,
            secret=secret,
            expected_sequence=stop_sequence,
            timeout_seconds=60.0,
        )
        _require_frame_type(stopped, "stopped")
        return_code = ssh_process.wait(timeout=60.0)
        if return_code != 0:
            raise OpenEcologyTwoPartyAuthorityError(
                f"remote guardian exited with status {return_code}"
            )
        ssh_process.stdin.close()
        _revalidate_local_guardian_authority(authority)
        _require_live_source(preregistration)
        transcript = _write_compact_transcript(
            transcript_path,
            bindings=bindings,
            endpoint=endpoint.receipt_record(),
            coordinator_pid=os.getpid(),
            coordinator_pgid=os.getpgrp(),
            remote_pid=remote_pid,
            remote_pgid=remote_pgid,
            completed=completed,
            storage_facts_digest=storage.facts_digest,
            remote_verifier_digests=_strict_mapping(
                ready.get("verifier_digests"),
                field="remote verifier digests",
            ),
            complete=True,
        )
        return transcript
    finally:
        _scrub_bytearray(github_token)
        storage.close()
        if ssh_process is not None and ssh_process.poll() is None:
            if ssh_process.stdin is not None:
                ssh_process.stdin.close()
            try:
                os.killpg(ssh_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                ssh_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(ssh_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                ssh_process.wait(timeout=10.0)
        if stderr_capture is not None:
            stderr_capture.join(timeout=5.0)


def _revalidate_local_guardian_authority(
    authority: ArchiveToolAuthority,
) -> None:
    try:
        verify_local_authority_files(authority)
        verify_effective_ssh_config(authority)
    except ArchiveAuthorityError as error:
        raise OpenEcologyTwoPartyAuthorityError(
            f"local guardian archive authority drifted: {error}"
        ) from error


def install_guardian_process_safety(
    *,
    terminate: Callable[[str], None] | None = None,
) -> GuardianChannelLiveness:
    """Create an isolated process group and arm Linux parent-death handling."""

    if os.getpgrp() != os.getpid():
        os.setpgrp()
    if os.getpgrp() != os.getpid():
        raise OpenEcologyTwoPartyAuthorityError(
            "remote guardian could not isolate its process group"
        )
    terminator = terminate or _terminate_own_process_group
    liveness = GuardianChannelLiveness(terminate=terminator)
    if sys.platform.startswith("linux"):
        _arm_linux_parent_death_signal(liveness)
    return liveness


def _arm_linux_parent_death_signal(liveness: GuardianChannelLiveness) -> None:
    original_parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_pdeathsig = 1
    if libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OpenEcologyTwoPartyAuthorityError(
            f"failed to arm guardian parent-death signal: errno={errno}"
        )

    def parent_lost(signum: int, frame: object) -> None:
        del signum, frame
        liveness.fail("ssh_parent_lost")

    signal.signal(signal.SIGTERM, parent_lost)
    if os.getppid() != original_parent:
        liveness.fail("ssh_parent_lost_during_arm")


def _terminate_own_process_group(reason: str) -> None:
    del reason
    try:
        # EOF and protocol failures originate on the reader thread.  Python
        # forbids changing signal handlers there, and the guardian's installed
        # SIGTERM handler would otherwise swallow a second liveness failure.
        # SIGKILL is process-directed, thread-safe, and cannot be intercepted.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    finally:
        # The own process group should always exist, but do not continue
        # training if killpg itself is denied or unexpectedly returns first.
        os._exit(128 + signal.SIGKILL)


def _terminate_local_ssh_process_group(
    process: subprocess.Popen[bytes],
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _protocol_reader(
    *,
    stream: BinaryIO,
    messages: queue.Queue[dict[str, object] | BaseException | None],
    channel: GuardianChannelLiveness,
) -> None:
    try:
        while True:
            line = stream.readline(MAX_PROTOCOL_FRAME_BYTES + 1)
            if not line:
                messages.put(None)
                channel.fail("ssh_channel_eof")
                return
            if len(line) > MAX_PROTOCOL_FRAME_BYTES or not line.endswith(b"\n"):
                error = OpenEcologyTwoPartyAuthorityError(
                    "guardian protocol frame exceeded its byte ceiling"
                )
                messages.put(error)
                channel.fail("protocol_frame_oversized")
                return
            message = _strict_json_mapping(line[:-1])
            line = b""
            messages.put(message)
            message = None
    except BaseException as error:
        messages.put(error)
        channel.fail("protocol_reader_failed")


def _next_message(
    messages: queue.Queue[dict[str, object] | BaseException | None],
    *,
    channel: GuardianChannelLiveness,
) -> dict[str, object]:
    message = messages.get()
    if message is None:
        raise OpenEcologyTwoPartyAuthorityError("guardian SSH channel reached EOF")
    if isinstance(message, BaseException):
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol reader failed"
        ) from message
    channel.require_alive()
    return message


def _write_hello_and_scrub_token(
    stream: BinaryIO,
    *,
    hello: dict[str, object],
    secret: bytes,
    github_token: bytearray,
) -> None:
    """Flush the sole credential-bearing frame, then drop all owned copies."""

    try:
        _write_authenticated_frame(stream, hello, secret=secret)
    finally:
        hello.pop("github_token", None)
        _scrub_bytearray(github_token)


def _write_authenticated_frame(
    stream: BinaryIO,
    payload: Mapping[str, object],
    *,
    secret: bytes,
) -> None:
    frame = dict(payload)
    if "authentication" in frame:
        raise OpenEcologyTwoPartyAuthorityError(
            "protocol payload already contains authentication"
        )
    frame["authentication"] = hmac.new(
        secret,
        _protocol_canonical_bytes(frame),
        hashlib.sha256,
    ).hexdigest()
    encoded = _protocol_canonical_bytes(frame) + b"\n"
    if len(encoded) > MAX_PROTOCOL_FRAME_BYTES:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol frame exceeded its byte ceiling"
        )
    stream.write(encoded)
    stream.flush()


def _write_plain_frame(
    stream: BinaryIO,
    payload: Mapping[str, object],
) -> None:
    """Write the pre-authentication nonce challenge on the pinned SSH stream."""

    if "authentication" in payload:
        raise OpenEcologyTwoPartyAuthorityError(
            "plain challenge must not claim protocol authentication"
        )
    encoded = _protocol_canonical_bytes(payload) + b"\n"
    if len(encoded) > MAX_PROTOCOL_FRAME_BYTES:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol frame exceeded its byte ceiling"
        )
    stream.write(encoded)
    stream.flush()


def _verify_frame_authentication(
    frame: dict[str, object],
    *,
    secret: bytes,
    expected_sequence: int,
) -> None:
    observed = frame.pop("authentication", None)
    if not isinstance(observed, str) or len(observed) != 64:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol authentication is malformed"
        )
    expected = hmac.new(
        secret,
        _protocol_canonical_bytes(frame),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol authentication mismatched"
        )
    if frame.get("sequence") != expected_sequence:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol sequence was replayed or reordered"
        )
    if frame.get("schema_version") != TWO_PARTY_PROTOCOL_SCHEMA_VERSION:
        raise OpenEcologyTwoPartyAuthorityError("guardian protocol schema drifted")


def _read_authenticated_frame_with_timeout(
    stream: BinaryIO,
    *,
    secret: bytes,
    expected_sequence: int,
    timeout_seconds: float,
) -> dict[str, object]:
    frame = _read_protocol_frame_with_timeout(
        stream,
        timeout_seconds=timeout_seconds,
    )
    _verify_frame_authentication(
        frame,
        secret=secret,
        expected_sequence=expected_sequence,
    )
    return frame


def _read_protocol_frame_with_timeout(
    stream: BinaryIO,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise OpenEcologyTwoPartyAuthorityError("protocol timeout must be positive")
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError) as error:
        raise OpenEcologyTwoPartyAuthorityError(
            "remote guardian protocol stream lacks a live descriptor"
        ) from error
    deadline = time.monotonic() + float(timeout_seconds)
    payload = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise OpenEcologyTwoPartyAuthorityError(
                    "timed out waiting for remote guardian protocol frame"
                )
            try:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, MAX_PROTOCOL_FRAME_BYTES - len(payload)),
                )
            except OSError as error:
                raise OpenEcologyTwoPartyAuthorityError(
                    "remote guardian protocol read failed"
                ) from error
            if not chunk:
                raise OpenEcologyTwoPartyAuthorityError(
                    "remote guardian closed its protocol channel"
                )
            payload.extend(chunk)
            newline = payload.find(b"\n")
            if newline >= 0:
                if newline != len(payload) - 1:
                    raise OpenEcologyTwoPartyAuthorityError(
                        "remote guardian sent surplus protocol frame bytes"
                    )
                break
            if len(payload) >= MAX_PROTOCOL_FRAME_BYTES:
                raise OpenEcologyTwoPartyAuthorityError(
                    "remote guardian protocol frame exceeded its byte ceiling"
                )
    finally:
        selector.close()
    return _strict_json_mapping(bytes(payload[:-1]))


def _read_guardian_challenge_with_timeout(
    stream: BinaryIO,
    *,
    expected_bindings: PhaseAAuthorityBindings,
    timeout_seconds: float,
) -> dict[str, object]:
    frame = _read_protocol_frame_with_timeout(
        stream,
        timeout_seconds=timeout_seconds,
    )
    if set(frame) != {
        "bindings",
        "frame_type",
        "remote_challenge",
        "remote_pgid",
        "remote_pid",
        "schema_version",
        "sequence",
    }:
        raise OpenEcologyTwoPartyAuthorityError(
            "remote guardian challenge shape drifted"
        )
    _require_frame_type(frame, "challenge")
    if (
        frame.get("schema_version") != TWO_PARTY_PROTOCOL_SCHEMA_VERSION
        or frame.get("sequence") != -1
        or _bindings_from_mapping(frame.get("bindings")) != expected_bindings
    ):
        raise OpenEcologyTwoPartyAuthorityError(
            "remote guardian challenge binding drifted"
        )
    decoded = _decode_remote_challenge(frame.get("remote_challenge"))
    _scrub_bytearray(decoded)
    frame["remote_pid"] = _positive_int(
        frame.get("remote_pid"),
        field="challenge remote PID",
    )
    frame["remote_pgid"] = _positive_int(
        frame.get("remote_pgid"),
        field="challenge remote PGID",
    )
    return frame


class _BoundedStderrCapture:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        on_error: Callable[[BaseException], None],
    ) -> None:
        self.stream = stream
        self._on_error = on_error
        self._payload = bytearray()
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="open-ecology-guardian-stderr",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def snapshot(self) -> bytes:
        with self._lock:
            if self._error is not None:
                raise OpenEcologyTwoPartyAuthorityError(
                    "SSH stderr capture failed"
                ) from self._error
            return bytes(self._payload)

    def _run(self) -> None:
        try:
            read = getattr(self.stream, "read1", self.stream.read)
            while chunk := read(8192):
                with self._lock:
                    if len(self._payload) + len(chunk) > MAX_SSH_STDERR_BYTES:
                        raise OpenEcologyTwoPartyAuthorityError(
                            "SSH stderr exceeded its byte ceiling"
                        )
                    self._payload.extend(chunk)
        except BaseException as error:
            with self._lock:
                self._error = error
            self._on_error(error)


def _wait_for_authenticated_endpoint(
    capture: _BoundedStderrCapture,
    *,
    expected: ArchiveToolAuthority,
    deadline: float,
) -> Any:
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        payload = capture.snapshot()
        try:
            observed = parse_ssh_connection_identity(payload)
        except BaseException as error:
            last_error = error
            time.sleep(0.05)
            continue
        if observed != expected.ssh_connection:
            raise OpenEcologyTwoPartyAuthorityError(
                "live guardian SSH endpoint differs from the pinned authority"
            )
        return observed
    raise OpenEcologyTwoPartyAuthorityError(
        "live guardian SSH endpoint could not be authenticated"
    ) from last_error


def _remote_guardian_ssh_command(
    *,
    authority: ArchiveToolAuthority,
    remote_runtime_venv_authority_path: Path,
    remote_preregistration_path: Path,
    remote_launch_authorization_path: Path,
    remote_output_root: Path,
    bindings: PhaseAAuthorityBindings,
    device: str,
) -> list[str]:
    verify_pinned_file(
        authority.local_tool("ssh"),
        executable=True,
        require_nonempty=True,
    )
    remote_root = Path(authority.remote_repository_root)
    if not remote_root.is_absolute():
        raise OpenEcologyTwoPartyAuthorityError(
            "remote repository root is not absolute"
        )
    ssh_connection_sha256 = stable_payload_digest(
        authority.ssh_connection.receipt_record()
    )
    bootstrap_path = remote_root / _RUNTIME_BOOTSTRAP_RELATIVE_PATH
    remote_command = [
        authority.remote_tool("env").path,
        "PYTHONHASHSEED=0",
        authority.remote_tool("python").path,
        "-I",
        "-S",
        "-c",
        _PINNED_STDLIB_EXEC_CODE,
        str(bootstrap_path),
        authority.remote_helper_sha256(_RUNTIME_BOOTSTRAP_RELATIVE_PATH),
        "exec",
        "--authority",
        str(remote_runtime_venv_authority_path),
        "--expected-authority-sha256",
        bindings.runtime_venv_authority_sha256,
        "--source-git-sha",
        bindings.source_git_sha,
        "--source-manifest-sha256",
        bindings.source_manifest_sha256,
        "--archive-authority-sha256",
        authority.authority_sha256,
        "--ssh-target",
        authority.ssh_target,
        "--ssh-connection-sha256",
        ssh_connection_sha256,
        "--git-executable",
        authority.remote_tool("git").path,
        "--git-executable-sha256",
        authority.remote_tool("git").sha256,
        "--",
        "serve",
        "--preregistration",
        str(remote_preregistration_path),
        "--launch-authorization",
        str(remote_launch_authorization_path),
        "--output-root",
        str(remote_output_root),
        "--source-git-sha",
        bindings.source_git_sha,
        "--source-manifest-sha256",
        bindings.source_manifest_sha256,
        "--preregistration-digest",
        bindings.preregistration_digest,
        "--evidence-index-digest",
        bindings.evidence_index_digest,
        "--launch-authorization-digest",
        bindings.launch_authorization_digest,
        "--runtime-venv-authority",
        str(remote_runtime_venv_authority_path),
        "--runtime-venv-authority-sha256",
        bindings.runtime_venv_authority_sha256,
        "--archive-authority-sha256",
        authority.authority_sha256,
        "--ssh-connection-sha256",
        ssh_connection_sha256,
        "--git-executable",
        authority.remote_tool("git").path,
        "--git-executable-sha256",
        authority.remote_tool("git").sha256,
        "--ssh-target",
        bindings.ssh_target,
        "--device",
        device,
    ]
    return [
        authority.local_tool("ssh").path,
        "-v",
        *SEALED_SSH_OPTIONS,
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        authority.ssh_target,
        "--",
        shlex.join(remote_command),
    ]


def _bindings_from_archive_and_bundle(
    *,
    authority: ArchiveToolAuthority,
    preregistration_path: Path,
    launch_authorization_path: Path,
    runtime_venv_authority_sha256: str,
) -> PhaseAAuthorityBindings:
    preregistration = _load_strict_json(preregistration_path)
    authorization = _load_strict_json(launch_authorization_path)
    source = _strict_mapping(preregistration.get("source"), field="source")
    evidence = _strict_mapping(
        authorization.get("evidence_index"),
        field="authorization.evidence_index",
    )
    bindings = PhaseAAuthorityBindings(
        source_git_sha=_git_sha(source.get("commit"), field="source commit"),
        source_manifest_sha256=_sha256(
            source.get("manifest_sha256"),
            field="source manifest",
        ),
        preregistration_digest=_sha256(
            preregistration.get("exact_digest"),
            field="preregistration digest",
        ),
        evidence_index_digest=_sha256(
            evidence.get("exact_digest"),
            field="evidence index digest",
        ),
        launch_authorization_digest=_sha256(
            authorization.get("exact_digest"),
            field="launch authorization digest",
        ),
        runtime_venv_authority_sha256=_sha256(
            runtime_venv_authority_sha256,
            field="runtime venv authority SHA256",
        ),
        ssh_target=authority.ssh_target,
    )
    if (
        bindings.source_git_sha != authority.source_git_sha
        or bindings.source_manifest_sha256 != authority.source_manifest_sha256
    ):
        raise OpenEcologyTwoPartyAuthorityError(
            "archive authority and Phase-A source bindings differ"
        )
    return bindings


def _write_compact_transcript(
    path: Path,
    *,
    bindings: PhaseAAuthorityBindings,
    endpoint: Mapping[str, object],
    coordinator_pid: int,
    coordinator_pgid: int,
    remote_pid: int,
    remote_pgid: int,
    completed: Sequence[Mapping[str, object]],
    storage_facts_digest: str | None,
    remote_verifier_digests: Mapping[str, object],
    complete: bool,
) -> dict[str, object]:
    if storage_facts_digest is None:
        raise OpenEcologyTwoPartyAuthorityError(
            "storage authority digest is unavailable"
        )
    transcript: dict[str, object] = {
        "schema_version": TWO_PARTY_TRANSCRIPT_SCHEMA_VERSION,
        "authority_semantics": {
            "offline_json_authorizes_updates": False,
            "capability_serialized": False,
            "cryptographic_mac_attestation": False,
            "hello_freshness": (
                "remote_nonce_bound_hmac_over_authenticated_pinned_ssh"
            ),
            "mac_attestation_model": (
                "cooperative_exact_source_process_over_authenticated_pinned_ssh"
            ),
            "live_mac_storage_verifier_count": 1,
            "live_remote_verifier_counts": {name: 1 for name in _REMOTE_GATE_NAMES},
            "selection_authorized": False,
            "phase_b_authorized": False,
            "phase_c_authorized": False,
            "phase_d_authorized": False,
        },
        "bindings": bindings.as_dict(),
        "processes": {
            "coordinator_pgid": coordinator_pgid,
            "coordinator_pid": coordinator_pid,
            "remote_pgid": remote_pgid,
            "remote_pid": remote_pid,
        },
        "ssh_endpoint": dict(endpoint),
        "storage_facts_digest": storage_facts_digest,
        "remote_verifier_digests": dict(remote_verifier_digests),
        "completed_cells": [dict(row) for row in completed],
        "completed_cell_count": len(completed),
        "matrix_complete": complete,
        "transcript_is_resume_or_launch_authority": False,
    }
    transcript["exact_digest"] = stable_payload_digest(transcript)
    _write_atomic_json_replace(path, transcript)
    return transcript


def _write_atomic_json_replace(path: Path, payload: Mapping[str, object]) -> None:
    parent = path.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian transcript cannot be a symbolic link"
        )
    temporary = parent / f".{path.name}.pending-{os.getpid()}-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload_bytes = canonical_json_bytes(payload)
        view = memoryview(payload_bytes)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _canonical_phase_a_matrix() -> tuple[tuple[str, int], ...]:
    return tuple(
        (cell_id, learner_index)
        for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        for learner_index in range(4)
    )


def _bindings_from_mapping(value: object) -> PhaseAAuthorityBindings:
    mapping = _strict_mapping(value, field="authority bindings")
    expected = {
        "evidence_index_digest",
        "launch_authorization_digest",
        "preregistration_digest",
        "runtime_venv_authority_sha256",
        "source_git_sha",
        "source_manifest_sha256",
        "ssh_target",
    }
    if set(mapping) != expected:
        raise OpenEcologyTwoPartyAuthorityError("authority binding keys drifted")
    target = mapping.get("ssh_target")
    if (
        not isinstance(target, str)
        or not target
        or any(character.isspace() for character in target)
    ):
        raise OpenEcologyTwoPartyAuthorityError("SSH target binding is malformed")
    return PhaseAAuthorityBindings(
        source_git_sha=_git_sha(
            mapping.get("source_git_sha"),
            field="source Git SHA",
        ),
        source_manifest_sha256=_sha256(
            mapping.get("source_manifest_sha256"),
            field="source manifest",
        ),
        preregistration_digest=_sha256(
            mapping.get("preregistration_digest"),
            field="preregistration digest",
        ),
        evidence_index_digest=_sha256(
            mapping.get("evidence_index_digest"),
            field="evidence index digest",
        ),
        launch_authorization_digest=_sha256(
            mapping.get("launch_authorization_digest"),
            field="launch authorization digest",
        ),
        runtime_venv_authority_sha256=_sha256(
            mapping.get("runtime_venv_authority_sha256"),
            field="runtime venv authority SHA256",
        ),
        ssh_target=target,
    )


def _decode_session_secret(value: object) -> bytes:
    if not isinstance(value, str):
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian hello lacks its ephemeral session secret"
        )
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian session secret is malformed"
        ) from error
    if len(decoded) != _CAPABILITY_BYTES:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian session secret length drifted"
        )
    return decoded


def _decode_github_token(value: object) -> bytearray:
    if not isinstance(value, str):
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian hello lacks its one-shot GitHub credential"
        )
    try:
        decoded = bytearray(base64.b64decode(value, validate=True))
    except (ValueError, TypeError) as error:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian GitHub credential is malformed"
        ) from error
    _require_github_token(decoded)
    return decoded


def _decode_remote_challenge(value: object) -> bytearray:
    if not isinstance(value, str):
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian session lacks its remote freshness challenge"
        )
    try:
        decoded = bytearray(base64.b64decode(value, validate=True))
    except (ValueError, TypeError) as error:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian remote freshness challenge is malformed"
        ) from error
    if len(decoded) != _REMOTE_CHALLENGE_BYTES:
        _scrub_bytearray(decoded)
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian remote freshness challenge length drifted"
        )
    return decoded


def _require_github_token(value: bytearray) -> None:
    if (
        not isinstance(value, bytearray)
        or len(value) < 20
        or len(value) > 4096
        or any(byte < 0x21 or byte > 0x7E for byte in value)
    ):
        raise OpenEcologyTwoPartyAuthorityError(
            "one-shot GitHub credential is absent or malformed"
        )


def _scrub_bytearray(value: bytearray) -> None:
    for index in range(len(value)):
        value[index] = 0


def _strict_json_mapping(payload: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OpenEcologyTwoPartyAuthorityError(
                    "guardian protocol JSON contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: _raise_json_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian protocol frame is not strict JSON"
        ) from error
    return dict(_strict_mapping(value, field="guardian protocol frame"))


def _protocol_canonical_bytes(payload: Mapping[str, object]) -> bytes:
    encoded = canonical_json_bytes(payload)
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise OpenEcologyTwoPartyAuthorityError(
            "canonical protocol JSON framing drifted"
        )
    return encoded[:-1]


def _raise_json_constant(value: str) -> object:
    raise OpenEcologyTwoPartyAuthorityError(
        f"guardian protocol JSON constant {value!r} is forbidden"
    )


def _resolve_regular_relative(root: Path, value: object, *, field: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or "\\" in value
    ):
        raise OpenEcologyTwoPartyAuthorityError(
            f"{field} path is not one canonical relative path"
        )
    candidate = root / value
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise OpenEcologyTwoPartyAuthorityError(
            f"{field} escaped its authority root"
        ) from error
    current = candidate
    while current != root:
        if current.is_symlink():
            raise OpenEcologyTwoPartyAuthorityError(f"{field} contains a symbolic link")
        current = current.parent
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OpenEcologyTwoPartyAuthorityError(f"{field} is not a regular file")
    return resolved


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OpenEcologyTwoPartyAuthorityError(
            "launch authorization time is malformed"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise OpenEcologyTwoPartyAuthorityError(
            "launch authorization time is malformed"
        ) from error
    return parsed


def _require_frame_type(frame: Mapping[str, object], expected: str) -> None:
    if frame.get("frame_type") != expected:
        raise OpenEcologyTwoPartyAuthorityError(f"expected guardian frame {expected!r}")


def _require_cell_identity(cell_id: str, learner_index: int) -> None:
    if (
        cell_id not in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        or isinstance(learner_index, bool)
        or not isinstance(learner_index, int)
        or not 0 <= learner_index < 4
    ):
        raise OpenEcologyTwoPartyAuthorityError(
            "guardian cell identity is outside the fixed Phase-A matrix"
        )


def _strict_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OpenEcologyTwoPartyAuthorityError(f"{field} is not a string-key mapping")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenEcologyTwoPartyAuthorityError(f"{field} is not a lowercase SHA256")
    return value


def _git_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpenEcologyTwoPartyAuthorityError(f"{field} is not a lowercase Git SHA")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyTwoPartyAuthorityError(f"{field} must be a positive integer")
    return value


def _coordinator_environment() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }


__all__ = [
    "DEFAULT_CELL_TIMEOUT_SECONDS",
    "DEFAULT_GUARDIAN_READY_TIMEOUT_SECONDS",
    "GuardianChannelLiveness",
    "LivePhaseAUpdateAdmission",
    "MacStorageAuthority",
    "MAX_PROTOCOL_FRAME_BYTES",
    "OpenEcologyTwoPartyAuthorityError",
    "PhaseAAuthorityBindings",
    "RemoteGuardianAuthority",
    "TWO_PARTY_PROTOCOL_SCHEMA_VERSION",
    "TWO_PARTY_TRANSCRIPT_SCHEMA_VERSION",
    "install_guardian_process_safety",
    "load_phase_a_authority_bundle",
    "require_live_phase_a_update_admission",
    "run_mac_coordinator",
    "run_remote_guardian_session",
]
