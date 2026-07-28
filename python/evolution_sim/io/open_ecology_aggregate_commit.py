from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Iterator, Mapping

from evolution_sim.io.open_ecology_campaign_storage import (
    CampaignStorageError,
    ensure_real_directory_tree,
)
from evolution_sim.io.open_ecology_checkpoint import (
    OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    load_open_ecology_checkpoint,
)
from evolution_sim.io.open_ecology_rotating_writer import (
    OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA,
    OPEN_ECOLOGY_EVIDENCE_LOCK_NAME,
    OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME,
    load_open_ecology_evidence_manifest,
    validate_open_ecology_evidence_continuation_state,
)

if TYPE_CHECKING:
    from evolution_sim.io.open_ecology_runtime_checkpoint import (
        ExtractedRuntimeEvidenceContinuation,
    )


OPEN_ECOLOGY_AGGREGATE_COMMIT_SCHEMA = (
    "evolution_sim_open_ecology_aggregate_generation_commit_v1"
)
OPEN_ECOLOGY_AGGREGATE_IDENTITY_SCHEMA = (
    "evolution_sim_open_ecology_aggregate_generation_identity_v1"
)
OPEN_ECOLOGY_AGGREGATE_POINTER_SCHEMA = (
    "evolution_sim_open_ecology_aggregate_current_pointer_v1"
)
OPEN_ECOLOGY_AGGREGATE_COMMIT_DIGEST_POLICY = (
    "sha256_canonical_json_without_commit_sha256_v1"
)
OPEN_ECOLOGY_AGGREGATE_IDENTITY_DIGEST_POLICY = (
    "sha256_canonical_json_without_identity_sha256_v1"
)
OPEN_ECOLOGY_AGGREGATE_POINTER_DIGEST_POLICY = (
    "sha256_canonical_json_without_pointer_sha256_v1"
)

OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY = "generations"
OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME = "CURRENT"
OPEN_ECOLOGY_AGGREGATE_LOCK_NAME = ".aggregate.lock"
OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME = ".cleanup-tombstones-v1"
OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME = "checkpoint.json"
OPEN_ECOLOGY_AGGREGATE_EVIDENCE_MANIFEST_NAME = "evidence-manifest.json"
OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME = "commit.json"

OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES = 1024 * 1024
OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES = 64 * 1024
OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES = 16 * 1024 * 1024
OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS = 100_000

_AGGREGATE_LOCK_BYTES = b"evolution_sim_open_ecology_aggregate_writer_lock_v1\n"
_EVIDENCE_LOCK_BYTES = b"evolution_sim_open_ecology_writer_lock_v1\n"
_GENERATION_DIRECTORY_RE = re.compile(r"^generation-([0-9]{16})$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_CLEANUP_SLOT_RE = re.compile(
    r"^(?P<kind>file|stage)-d(?P<device>[0-9a-f]+)-"
    r"i(?P<inode>[0-9a-f]+)-(?P<nonce>[0-9a-f]{16})$"
)
_MAX_JSON_NESTING_DEPTH = 128
_CLEANUP_TOMBSTONE_MULTIPLIER = 2
_CLEANUP_TOMBSTONE_RESERVE = 16
_OPEN_ECOLOGY_RUNTIME_EVIDENCE_ADAPTER_SCHEMA_VERSION = (
    "open_ecology_evidence_writer_continuation_adapter_v1"
)
_GENERATION_FILES = frozenset(
    {
        OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME,
        OPEN_ECOLOGY_AGGREGATE_EVIDENCE_MANIFEST_NAME,
        OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME,
    }
)
_ROOT_ENTRIES_WITHOUT_CURRENT = frozenset(
    {
        OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY,
        OPEN_ECOLOGY_AGGREGATE_LOCK_NAME,
    }
)
_ROOT_ENTRIES_WITH_CURRENT = frozenset(
    {
        *_ROOT_ENTRIES_WITHOUT_CURRENT,
        OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME,
    }
)
_FORMAT_CONTRACT = {
    "commit_encoding": "canonical_ascii_json_with_single_trailing_lf",
    "commit_digest_policy": OPEN_ECOLOGY_AGGREGATE_COMMIT_DIGEST_POLICY,
    "current_pointer_encoding": "canonical_ascii_json_with_single_trailing_lf",
    "current_pointer_digest_policy": OPEN_ECOLOGY_AGGREGATE_POINTER_DIGEST_POLICY,
    "evidence_manifest_encoding": "canonical_utf8_json_with_single_trailing_lf",
    "generation_directory_policy": (
        "immutable_after_atomic_directory_publication; existing_generations_are_"
        "never_replaced_or_pruned"
    ),
    "publication_policy": (
        "complete_generation_directory_then_atomic_current_pointer_v1"
    ),
}
_SUBSTRATE_STATUS = (
    "container_and_evidence_prefix_complete; world_adapters_and_continuation_"
    "equivalence_not_proven"
)


class OpenEcologyAggregateCommitError(ValueError):
    """Raised when an aggregate checkpoint generation fails closed."""


@dataclass(frozen=True, slots=True)
class _PathIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class OpenEcologyAggregateIdentityPins:
    """Exact source and simulation identity expected at publication or resume."""

    source_git_sha: str
    config_contract_sha256: str
    seed_contract_sha256: str
    run_generation_id: str
    island_id: str
    simulation_generation_index: int
    tick: int

    def __post_init__(self) -> None:
        _git_sha(self.source_git_sha, field="identity.source_git_sha")
        _sha256(
            self.config_contract_sha256,
            field="identity.config_contract_sha256",
        )
        _sha256(
            self.seed_contract_sha256,
            field="identity.seed_contract_sha256",
        )
        _identifier(self.run_generation_id, field="identity.run_generation_id")
        _identifier(self.island_id, field="identity.island_id")
        _nonnegative_int(
            self.simulation_generation_index,
            field="identity.simulation_generation_index",
        )
        _nonnegative_int(self.tick, field="identity.tick")


@dataclass(frozen=True, slots=True)
class OpenEcologyAggregateResumePins:
    """External authority required before a generation may be used to resume."""

    identity: OpenEcologyAggregateIdentityPins
    aggregate_generation_index: int
    commit_sha256: str
    checkpoint_sha256: str
    checkpoint_generation_identity_sha256: str
    evidence_manifest_sha256: str
    evidence_manifest_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, OpenEcologyAggregateIdentityPins):
            raise OpenEcologyAggregateCommitError(
                "resume identity must be OpenEcologyAggregateIdentityPins"
            )
        _nonnegative_int(
            self.aggregate_generation_index,
            field="resume.aggregate_generation_index",
        )
        for field_name in (
            "commit_sha256",
            "checkpoint_sha256",
            "checkpoint_generation_identity_sha256",
            "evidence_manifest_sha256",
        ):
            _sha256(getattr(self, field_name), field=f"resume.{field_name}")
        if self.evidence_manifest_status != "open":
            raise OpenEcologyAggregateCommitError(
                "resume evidence manifest status must be exactly open"
            )


def publish_open_ecology_aggregate_generation(
    aggregate_root: str | Path,
    *,
    checkpoint_path: str | Path,
    evidence_directory: str | Path,
    identity: OpenEcologyAggregateIdentityPins,
    aggregate_generation_index: int,
    expected_previous_commit_sha256: str | None,
    expected_checkpoint_sha256: str,
    expected_checkpoint_generation_identity_sha256: str,
    expected_evidence_manifest_sha256: str,
    expected_evidence_manifest_status: str,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object]:
    """Publish one immutable aggregate generation, then atomically advance CURRENT.

    The evidence writer must first be checkpointed so its process lock is
    available and its continuation state names an ``open`` manifest head.
    """

    limits = _limits(
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )
    if not isinstance(identity, OpenEcologyAggregateIdentityPins):
        raise OpenEcologyAggregateCommitError(
            "identity must be OpenEcologyAggregateIdentityPins"
        )
    selected_index = _nonnegative_int(
        aggregate_generation_index,
        field="aggregate_generation_index",
    )
    if selected_index >= limits.max_generations:
        raise OpenEcologyAggregateCommitError(
            "aggregate_generation_index exceeds max_generations"
        )
    expected_checkpoint_digest = _sha256(
        expected_checkpoint_sha256,
        field="expected_checkpoint_sha256",
    )
    expected_checkpoint_identity = _sha256(
        expected_checkpoint_generation_identity_sha256,
        field="expected_checkpoint_generation_identity_sha256",
    )
    expected_manifest_digest = _sha256(
        expected_evidence_manifest_sha256,
        field="expected_evidence_manifest_sha256",
    )
    if expected_evidence_manifest_status != "open":
        raise OpenEcologyAggregateCommitError(
            "aggregate resume generations require an open evidence manifest"
        )
    if expected_previous_commit_sha256 is not None:
        expected_previous_commit_sha256 = _sha256(
            expected_previous_commit_sha256,
            field="expected_previous_commit_sha256",
        )

    root = _prepare_root(Path(aggregate_root))
    evidence_path = _require_directory(Path(evidence_directory), field="evidence")
    with _aggregate_writer_lock(root, limits=limits):
        with _evidence_writer_lock(evidence_path):
            _recover_complete_successor(
                root,
                limits=limits,
                evidence_directory=evidence_path,
            )
            previous = _load_current_internal(root, limits=limits)
            _validate_next_generation_request(
                selected_index=selected_index,
                expected_previous_commit_sha256=expected_previous_commit_sha256,
                previous=previous,
            )
            checkpoint = load_open_ecology_checkpoint(
                checkpoint_path,
                max_checkpoint_bytes=limits.max_checkpoint_bytes,
                expected_source_git_sha=identity.source_git_sha,
                expected_generation_identity_sha256=expected_checkpoint_identity,
                require_restartable=True,
            )
            _validate_checkpoint_identity(checkpoint, identity=identity)
            if checkpoint["checkpoint_sha256"] != expected_checkpoint_digest:
                raise OpenEcologyAggregateCommitError(
                    "checkpoint SHA256 does not match the expected publication pin"
                )
            checkpoint_bytes = _canonical_ascii_line(checkpoint)
            if len(checkpoint_bytes) > limits.max_checkpoint_bytes:
                raise OpenEcologyAggregateCommitError(
                    "checkpoint exceeds max_checkpoint_bytes"
                )

            evidence_manifest = load_open_ecology_evidence_manifest(
                evidence_path,
                verify_shards=True,
                expected_manifest_sha256=expected_manifest_digest,
                expected_status=expected_evidence_manifest_status,
            )
            evidence_manifest_bytes = _canonical_utf8_line(evidence_manifest)
            if len(evidence_manifest_bytes) > limits.max_evidence_manifest_bytes:
                raise OpenEcologyAggregateCommitError(
                    "evidence manifest exceeds max_evidence_manifest_bytes"
                )
            _bind_manifest_source_contract_to_identity(
                evidence_manifest,
                identity=identity,
            )
            continuation = _checkpoint_evidence_continuation(checkpoint)
            _bind_continuation_to_manifest(
                continuation,
                evidence_manifest,
                checkpoint_tick=identity.tick,
            )

            directory_name = _generation_directory_name(selected_index)
            commit = _build_commit(
                aggregate_generation_index=selected_index,
                directory_name=directory_name,
                identity=identity,
                checkpoint=checkpoint,
                checkpoint_bytes=checkpoint_bytes,
                evidence_manifest=evidence_manifest,
                evidence_manifest_bytes=evidence_manifest_bytes,
                continuation=continuation,
                previous=previous,
            )
            commit_bytes = _canonical_ascii_line(commit)
            if len(commit_bytes) > limits.max_commit_bytes:
                raise OpenEcologyAggregateCommitError(
                    "aggregate commit exceeds max_commit_bytes"
                )
            _publish_generation_directory(
                root,
                directory_name=directory_name,
                checkpoint_bytes=checkpoint_bytes,
                evidence_manifest_bytes=evidence_manifest_bytes,
                commit_bytes=commit_bytes,
                max_cleanup_tombstones=_aggregate_cleanup_tombstone_limit(
                    limits.max_generations
                ),
            )
            _publish_current_pointer(
                root,
                _pointer_for_commit(commit),
                max_pointer_bytes=limits.max_pointer_bytes,
                max_cleanup_tombstones=_aggregate_cleanup_tombstone_limit(
                    limits.max_generations
                ),
            )
            loaded = _load_generation_internal(
                root,
                generation_index=selected_index,
                limits=limits,
            )
            _validate_external_evidence_prefix(
                evidence_path,
                evidence_manifest=_mapping(
                    loaded["evidence_manifest"],
                    field="loaded.evidence_manifest",
                ),
            )
            return loaded


def load_current_open_ecology_aggregate_generation(
    aggregate_root: str | Path,
    *,
    evidence_directory: str | Path,
    pins: OpenEcologyAggregateResumePins,
    uncommitted_successor_preservation_directory: str | Path | None = None,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object]:
    """Load CURRENT under exact pins.

    By default the legacy aggregate-container recovery contract advances
    ``CURRENT`` across one atomically complete successor.  A persistent
    campaign restore instead supplies
    ``uncommitted_successor_preservation_directory``: externally pinned
    ``CURRENT`` remains the sole authority and a complete but unselected
    successor is moved intact into that attempt directory.
    """

    return _load_pinned_generation(
        aggregate_root,
        evidence_directory=evidence_directory,
        pins=pins,
        requested_generation_index=None,
        complete_successor_policy=(
            "recover"
            if uncommitted_successor_preservation_directory is None
            else "preserve"
        ),
        uncommitted_successor_preservation_directory=(
            uncommitted_successor_preservation_directory
        ),
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )


def inspect_current_open_ecology_aggregate_generation(
    aggregate_root: str | Path,
    *,
    evidence_directory: str | Path,
    pins: OpenEcologyAggregateResumePins,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object]:
    """Read CURRENT under exact pins without recovering or preserving a successor."""

    return _load_pinned_generation(
        aggregate_root,
        evidence_directory=evidence_directory,
        pins=pins,
        requested_generation_index=None,
        complete_successor_policy="inspect",
        uncommitted_successor_preservation_directory=None,
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )


def load_or_recover_open_ecology_genesis_generation(
    aggregate_root: str | Path,
    *,
    evidence_directory: str | Path,
    identity: OpenEcologyAggregateIdentityPins,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object] | None:
    """Load or adopt one complete generation-zero attempt under exact identity.

    ``None`` means no immutable genesis generation exists yet.  A complete
    generation-zero directory whose ``CURRENT`` publication was interrupted is
    validated before the pointer is advanced.  No later generation is accepted
    by this bootstrap-only API.
    """

    if not isinstance(identity, OpenEcologyAggregateIdentityPins):
        raise OpenEcologyAggregateCommitError(
            "genesis identity must be OpenEcologyAggregateIdentityPins"
        )
    if identity.simulation_generation_index != 0 or identity.tick != 0:
        raise OpenEcologyAggregateCommitError(
            "genesis identity must name simulation generation zero at tick zero"
        )
    limits = _limits(
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )
    root_path = Path(aggregate_root)
    if not root_path.exists() and not root_path.is_symlink():
        return None
    root = _require_directory(root_path, field="aggregate_root")
    evidence_path = _require_directory(Path(evidence_directory), field="evidence")
    # A process may die after creating the aggregate root or zero-length lock
    # but before the first generation is durable.  Genesis recovery owns the
    # only bootstrap mutation, so it may finish that control-file setup.
    with _aggregate_writer_lock(root, create=True, limits=limits):
        current = _load_current_internal(
            root,
            limits=limits,
            allow_complete_successor=True,
        )
        indexes = _generation_indexes(
            root,
            max_generations=limits.max_generations,
        )
        # No immutable aggregate exists yet, so the evidence stream is only a
        # failed attempt.  In particular, its writer lock may be zero-length
        # after process death during lock initialization; the caller preserves
        # that directory intact before rematerializing.
        if current is None and not indexes:
            return None
        with _evidence_writer_lock(evidence_path):
            if current is None:
                if indexes != [0]:
                    raise OpenEcologyAggregateCommitError(
                        "genesis recovery found a non-genesis aggregate frontier"
                    )
                current = _load_generation_internal(
                    root,
                    generation_index=0,
                    limits=limits,
                )
                _validate_genesis_generation(current, identity=identity)
                _validate_external_evidence_prefix(
                    evidence_path,
                    evidence_manifest=_mapping(
                        current["evidence_manifest"],
                        field="genesis.evidence_manifest",
                    ),
                )
                _publish_current_pointer(
                    root,
                    _pointer_for_commit(
                        _mapping(current["commit"], field="genesis.commit")
                    ),
                    max_pointer_bytes=limits.max_pointer_bytes,
                    max_cleanup_tombstones=_aggregate_cleanup_tombstone_limit(
                        limits.max_generations
                    ),
                )
            if indexes != [0]:
                raise OpenEcologyAggregateCommitError(
                    "genesis API refuses a post-genesis aggregate frontier"
                )
            _validate_genesis_generation(current, identity=identity)
            _validate_external_evidence_prefix(
                evidence_path,
                evidence_manifest=_mapping(
                    current["evidence_manifest"],
                    field="genesis.evidence_manifest",
                ),
            )
            retained = _load_current_internal(root, limits=limits)
            if retained is None:
                raise OpenEcologyAggregateCommitError(
                    "genesis CURRENT disappeared during validated recovery"
                )
            if retained != current:
                raise OpenEcologyAggregateCommitError(
                    "genesis CURRENT changed during validated recovery"
                )
            return retained


def inspect_open_ecology_aggregate_attempt_frontier(
    aggregate_root: str | Path,
    *,
    evidence_directory: str | Path,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object]:
    """Inspect CURRENT and at most one complete successor without selecting it.

    This is intentionally not resume authority.  A caller must validate a
    returned generation against an external attempt contract before using
    :func:`select_open_ecology_aggregate_attempt_generation`.
    """

    limits = _limits(
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )
    root_path = Path(aggregate_root)
    if not root_path.exists() and not root_path.is_symlink():
        return {
            "complete_successor": None,
            "current": None,
            "current_predecessor": None,
        }
    root = _require_directory(root_path, field="aggregate_root")
    evidence_path = _require_directory(Path(evidence_directory), field="evidence")
    with _aggregate_writer_lock(root, create=True, limits=limits):
        with _evidence_writer_lock(evidence_path):
            current = _load_current_internal(
                root,
                limits=limits,
                allow_complete_successor=True,
            )
            indexes = _generation_indexes(
                root,
                max_generations=limits.max_generations,
            )
            current_index = (
                None
                if current is None
                else int(
                    _mapping(current["commit"], field="current.commit")[
                        "aggregate_generation_index"
                    ]
                )
            )
            current_predecessor = (
                None
                if current_index is None or current_index == 0
                else _load_generation_internal(
                    root,
                    generation_index=current_index - 1,
                    limits=limits,
                )
            )
            successor_index = 0 if current_index is None else current_index + 1
            successor = (
                _load_generation_internal(
                    root,
                    generation_index=successor_index,
                    limits=limits,
                )
                if indexes and indexes[-1] == successor_index
                else None
            )
            for label, loaded in (
                ("current", current),
                ("current_predecessor", current_predecessor),
                ("complete_successor", successor),
            ):
                if loaded is not None:
                    _validate_external_evidence_prefix(
                        evidence_path,
                        evidence_manifest=_mapping(
                            loaded["evidence_manifest"],
                            field=f"{label}.evidence_manifest",
                        ),
                    )
            return {
                "complete_successor": successor,
                "current": current,
                "current_predecessor": current_predecessor,
            }


def select_open_ecology_aggregate_attempt_generation(
    aggregate_root: str | Path,
    *,
    evidence_directory: str | Path,
    pins: OpenEcologyAggregateResumePins,
    expected_previous_commit_sha256: str | None,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object]:
    """Select one externally validated complete attempt generation."""

    if not isinstance(pins, OpenEcologyAggregateResumePins):
        raise OpenEcologyAggregateCommitError(
            "attempt generation pins must be OpenEcologyAggregateResumePins"
        )
    previous_digest = (
        None
        if expected_previous_commit_sha256 is None
        else _sha256(
            expected_previous_commit_sha256,
            field="expected_previous_commit_sha256",
        )
    )
    if (pins.aggregate_generation_index == 0) != (previous_digest is None):
        raise OpenEcologyAggregateCommitError(
            "attempt predecessor presence does not match generation index"
        )
    limits = _limits(
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )
    root = _require_directory(Path(aggregate_root), field="aggregate_root")
    evidence_path = _require_directory(Path(evidence_directory), field="evidence")
    with _aggregate_writer_lock(root, create=False, limits=limits):
        with _evidence_writer_lock(evidence_path):
            current = _load_current_internal(
                root,
                limits=limits,
                allow_complete_successor=True,
            )
            current_index = (
                None
                if current is None
                else int(
                    _mapping(current["commit"], field="current.commit")[
                        "aggregate_generation_index"
                    ]
                )
            )
            target_index = pins.aggregate_generation_index
            if current_index == target_index:
                assert current is not None
                _validate_resume_pins(current, pins=pins)
                selected = current
            else:
                if current is None:
                    raise OpenEcologyAggregateCommitError(
                        "attempt selection requires a published CURRENT "
                        "predecessor; recover genesis through the genesis API"
                    )
                expected_current_index = target_index - 1
                if current_index != expected_current_index:
                    raise OpenEcologyAggregateCommitError(
                        "attempt selection CURRENT is not the exact predecessor"
                    )
                current_commit = _mapping(
                    current["commit"],
                    field="attempt predecessor.commit",
                )
                if current_commit["commit_sha256"] != previous_digest:
                    raise OpenEcologyAggregateCommitError(
                        "attempt predecessor commit pin mismatch"
                    )
                selected = _load_generation_internal(
                    root,
                    generation_index=target_index,
                    limits=limits,
                )
                _validate_resume_pins(selected, pins=pins)
                selected_commit = _mapping(
                    selected["commit"],
                    field="attempt target.commit",
                )
                selected_previous = selected_commit["previous"]
                if previous_digest is None:
                    if selected_previous is not None:
                        raise OpenEcologyAggregateCommitError(
                            "genesis attempt target unexpectedly names a predecessor"
                        )
                else:
                    previous = _mapping(
                        selected_previous,
                        field="attempt target.previous",
                    )
                    if (
                        previous["aggregate_generation_index"] != expected_current_index
                        or previous["commit_sha256"] != previous_digest
                    ):
                        raise OpenEcologyAggregateCommitError(
                            "attempt target does not extend the exact predecessor"
                        )
                _validate_external_evidence_prefix(
                    evidence_path,
                    evidence_manifest=_mapping(
                        selected["evidence_manifest"],
                        field="attempt target.evidence_manifest",
                    ),
                )
                _publish_current_pointer(
                    root,
                    _pointer_for_commit(selected_commit),
                    max_pointer_bytes=limits.max_pointer_bytes,
                    max_cleanup_tombstones=_aggregate_cleanup_tombstone_limit(
                        limits.max_generations
                    ),
                )
            retained = _load_current_internal(root, limits=limits)
            if retained is None:
                raise OpenEcologyAggregateCommitError(
                    "attempt selection failed to retain CURRENT"
                )
            _validate_resume_pins(retained, pins=pins)
            if retained != selected:
                raise OpenEcologyAggregateCommitError(
                    "attempt selection readback changed"
                )
            return retained


def load_open_ecology_aggregate_generation(
    aggregate_root: str | Path,
    *,
    aggregate_generation_index: int,
    evidence_directory: str | Path,
    pins: OpenEcologyAggregateResumePins,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    max_commit_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES,
    max_pointer_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_POINTER_BYTES,
    max_evidence_manifest_bytes: int = (
        OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
    ),
    max_generations: int = OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_GENERATIONS,
) -> dict[str, object]:
    """Load an immutable historical generation under exact external resume pins."""

    return _load_pinned_generation(
        aggregate_root,
        evidence_directory=evidence_directory,
        pins=pins,
        requested_generation_index=aggregate_generation_index,
        complete_successor_policy="recover",
        uncommitted_successor_preservation_directory=None,
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )


def _load_pinned_generation(
    aggregate_root: str | Path,
    *,
    evidence_directory: str | Path,
    pins: OpenEcologyAggregateResumePins,
    requested_generation_index: int | None,
    complete_successor_policy: str,
    uncommitted_successor_preservation_directory: str | Path | None,
    max_checkpoint_bytes: int,
    max_commit_bytes: int,
    max_pointer_bytes: int,
    max_evidence_manifest_bytes: int,
    max_generations: int,
) -> dict[str, object]:
    if not isinstance(pins, OpenEcologyAggregateResumePins):
        raise OpenEcologyAggregateCommitError(
            "pins must be OpenEcologyAggregateResumePins"
        )
    limits = _limits(
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_commit_bytes=max_commit_bytes,
        max_pointer_bytes=max_pointer_bytes,
        max_evidence_manifest_bytes=max_evidence_manifest_bytes,
        max_generations=max_generations,
    )
    root = _require_directory(Path(aggregate_root), field="aggregate_root")
    evidence_path = _require_directory(Path(evidence_directory), field="evidence")
    preservation_path = (
        None
        if uncommitted_successor_preservation_directory is None
        else Path(uncommitted_successor_preservation_directory)
    )
    if complete_successor_policy not in {"recover", "preserve", "inspect"}:
        raise OpenEcologyAggregateCommitError("unknown complete successor load policy")
    if (complete_successor_policy == "preserve") != (preservation_path is not None):
        raise OpenEcologyAggregateCommitError(
            "complete successor preservation policy/path mismatch"
        )
    if preservation_path is not None:
        if requested_generation_index is not None:
            raise OpenEcologyAggregateCommitError(
                "historical generation loads cannot preserve a CURRENT successor"
            )
        if not preservation_path.is_absolute():
            raise OpenEcologyAggregateCommitError(
                "uncommitted successor preservation directory must be absolute"
            )
        if preservation_path == root or root in preservation_path.parents:
            raise OpenEcologyAggregateCommitError(
                "uncommitted successor preservation directory must be outside "
                "the aggregate root"
            )
    if requested_generation_index is not None:
        requested_generation_index = _nonnegative_int(
            requested_generation_index,
            field="aggregate_generation_index",
        )
        if requested_generation_index != pins.aggregate_generation_index:
            raise OpenEcologyAggregateCommitError(
                "requested generation does not match resume pins"
            )
    with _aggregate_writer_lock(root, create=False, limits=limits):
        with _evidence_writer_lock(evidence_path):
            if complete_successor_policy == "recover":
                _recover_complete_successor(
                    root,
                    limits=limits,
                    evidence_directory=evidence_path,
                )
                current = _load_current_internal(root, limits=limits)
            else:
                current = _load_current_internal(
                    root,
                    limits=limits,
                    allow_complete_successor=True,
                )
            if current is None:
                raise OpenEcologyAggregateCommitError(
                    "aggregate store has no current generation"
                )
            current_index = int(
                _mapping(current["commit"], field="current.commit")[
                    "aggregate_generation_index"
                ]
            )
            selected_index = (
                current_index
                if requested_generation_index is None
                else requested_generation_index
            )
            if requested_generation_index is None and (
                pins.aggregate_generation_index != current_index
            ):
                raise OpenEcologyAggregateCommitError(
                    "resume pins do not name the CURRENT generation"
                )
            if selected_index > current_index:
                raise OpenEcologyAggregateCommitError(
                    "requested generation is newer than CURRENT"
                )
            loaded = _load_generation_internal(
                root,
                generation_index=selected_index,
                limits=limits,
            )
            _validate_resume_pins(loaded, pins=pins)
            _validate_external_evidence_prefix(
                evidence_path,
                evidence_manifest=_mapping(
                    loaded["evidence_manifest"],
                    field="loaded.evidence_manifest",
                ),
            )
            if complete_successor_policy == "preserve":
                assert preservation_path is not None
                _preserve_complete_successor(
                    root,
                    current=current,
                    limits=limits,
                    evidence_directory=evidence_path,
                    preservation_directory=preservation_path,
                )
                retained = _load_current_internal(root, limits=limits)
                if retained is None:
                    raise OpenEcologyAggregateCommitError(
                        "aggregate CURRENT disappeared during successor preservation"
                    )
                _validate_resume_pins(retained, pins=pins)
        return loaded


@dataclass(frozen=True, slots=True)
class _Limits:
    max_checkpoint_bytes: int
    max_commit_bytes: int
    max_pointer_bytes: int
    max_evidence_manifest_bytes: int
    max_generations: int


def _limits(
    *,
    max_checkpoint_bytes: int,
    max_commit_bytes: int,
    max_pointer_bytes: int,
    max_evidence_manifest_bytes: int,
    max_generations: int,
) -> _Limits:
    return _Limits(
        max_checkpoint_bytes=_positive_int(
            max_checkpoint_bytes,
            field="max_checkpoint_bytes",
        ),
        max_commit_bytes=_positive_int(
            max_commit_bytes,
            field="max_commit_bytes",
        ),
        max_pointer_bytes=_positive_int(
            max_pointer_bytes,
            field="max_pointer_bytes",
        ),
        max_evidence_manifest_bytes=_positive_int(
            max_evidence_manifest_bytes,
            field="max_evidence_manifest_bytes",
        ),
        max_generations=_positive_int(max_generations, field="max_generations"),
    )


def _prepare_root(root: Path) -> Path:
    if root.exists() or root.is_symlink():
        root = _require_directory(root, field="aggregate_root")
    else:
        try:
            root = ensure_real_directory_tree(
                root,
                field="aggregate root",
            )
        except CampaignStorageError as error:
            raise OpenEcologyAggregateCommitError(str(error)) from error
        root = _require_directory(root, field="aggregate_root")
    generations = root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
    if generations.exists() or generations.is_symlink():
        _require_directory(generations, field="aggregate generations")
    else:
        generations.mkdir(exist_ok=True)
        _require_directory(generations, field="aggregate generations")
        _fsync_directory(root)
    return root


@contextmanager
def _aggregate_writer_lock(
    root: Path,
    *,
    create: bool = True,
    limits: _Limits,
) -> Iterator[None]:
    if create:
        root = _prepare_root(root)
    else:
        root = _require_directory(root, field="aggregate_root")
    lock_path = root / OPEN_ECOLOGY_AGGREGATE_LOCK_NAME
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise OpenEcologyAggregateCommitError(
            f"cannot safely open aggregate lock: {error}"
        ) from error
    try:
        _lock_descriptor(
            descriptor,
            expected_bytes=_AGGREGATE_LOCK_BYTES,
            initialize=create,
            field="aggregate lock",
        )
        _reconcile_aggregate_cleanup_tombstones(
            root,
            max_tombstones=_aggregate_cleanup_tombstone_limit(limits.max_generations),
            limits=limits,
        )
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _evidence_writer_lock(evidence_directory: Path) -> Iterator[None]:
    lock_path = evidence_directory / OPEN_ECOLOGY_EVIDENCE_LOCK_NAME
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as error:
        raise OpenEcologyAggregateCommitError(
            f"cannot safely open evidence writer lock: {error}"
        ) from error
    try:
        _lock_descriptor(
            descriptor,
            expected_bytes=_EVIDENCE_LOCK_BYTES,
            initialize=False,
            field="evidence writer lock",
        )
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _lock_descriptor(
    descriptor: int,
    *,
    expected_bytes: bytes,
    initialize: bool,
    field: str,
) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise OpenEcologyAggregateCommitError(
            f"{field} is held by another process"
        ) from error
    file_stat = os.fstat(descriptor)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise OpenEcologyAggregateCommitError(
            f"{field} must be a single-link regular file"
        )
    if file_stat.st_size == 0 and initialize:
        pending = memoryview(expected_bytes)
        while pending:
            written = os.write(descriptor, pending)
            pending = pending[written:]
        os.fsync(descriptor)
        file_stat = os.fstat(descriptor)
    if file_stat.st_size != len(expected_bytes):
        raise OpenEcologyAggregateCommitError(
            f"{field} does not match its exact control contract"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    observed = os.read(descriptor, len(expected_bytes) + 1)
    if observed != expected_bytes:
        raise OpenEcologyAggregateCommitError(
            f"{field} does not match its exact control contract"
        )


def _recover_complete_successor(
    root: Path,
    *,
    limits: _Limits,
    evidence_directory: Path,
) -> None:
    current = _load_current_internal(
        root,
        limits=limits,
        allow_complete_successor=True,
    )
    indexes = _generation_indexes(root, max_generations=limits.max_generations)
    current_index = (
        None
        if current is None
        else int(
            _mapping(current["commit"], field="current.commit")[
                "aggregate_generation_index"
            ]
        )
    )
    candidate_index = 0 if current_index is None else current_index + 1
    if indexes and indexes[-1] == candidate_index:
        candidate = _load_generation_internal(
            root,
            generation_index=candidate_index,
            limits=limits,
        )
        commit = _mapping(candidate["commit"], field="candidate.commit")
        previous = commit["previous"]
        if current is None:
            if candidate_index != 0 or previous is not None:
                raise OpenEcologyAggregateCommitError(
                    "orphan initial generation has invalid predecessor"
                )
        else:
            previous_payload = _mapping(previous, field="candidate.previous")
            current_commit = _mapping(current["commit"], field="current.commit")
            if (
                previous_payload["aggregate_generation_index"] != current_index
                or previous_payload["commit_sha256"] != current_commit["commit_sha256"]
            ):
                raise OpenEcologyAggregateCommitError(
                    "complete successor does not extend CURRENT"
                )
        _validate_external_evidence_prefix(
            evidence_directory,
            evidence_manifest=_mapping(
                candidate["evidence_manifest"],
                field="candidate.evidence_manifest",
            ),
        )
        _publish_current_pointer(
            root,
            _pointer_for_commit(commit),
            max_pointer_bytes=limits.max_pointer_bytes,
            max_cleanup_tombstones=_aggregate_cleanup_tombstone_limit(
                limits.max_generations
            ),
        )
    _load_current_internal(root, limits=limits)


def _preserve_complete_successor(
    root: Path,
    *,
    current: Mapping[str, object],
    limits: _Limits,
    evidence_directory: Path,
    preservation_directory: Path,
) -> None:
    """Move one complete generation not selected by CURRENT out of authority."""

    indexes = _generation_indexes(root, max_generations=limits.max_generations)
    current_commit = _mapping(current["commit"], field="current.commit")
    current_index = int(current_commit["aggregate_generation_index"])
    candidate_index = current_index + 1
    if not indexes or indexes[-1] != candidate_index:
        return
    candidate = _load_generation_internal(
        root,
        generation_index=candidate_index,
        limits=limits,
    )
    candidate_commit = _mapping(candidate["commit"], field="candidate.commit")
    previous = _mapping(candidate_commit["previous"], field="candidate.previous")
    if (
        previous["aggregate_generation_index"] != current_index
        or previous["commit_sha256"] != current_commit["commit_sha256"]
    ):
        raise OpenEcologyAggregateCommitError(
            "complete successor does not extend pinned CURRENT"
        )
    _validate_external_evidence_prefix(
        evidence_directory,
        evidence_manifest=_mapping(
            candidate["evidence_manifest"],
            field="candidate.evidence_manifest",
        ),
    )

    try:
        preservation_directory = ensure_real_directory_tree(
            preservation_directory,
            field="aggregate attempt preservation directory",
        )
    except CampaignStorageError as error:
        raise OpenEcologyAggregateCommitError(str(error)) from error
    preservation_directory = _require_directory(
        preservation_directory,
        field="aggregate attempt preservation directory",
    )
    source = (
        root
        / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
        / _generation_directory_name(candidate_index)
    )
    commit_sha256 = _sha256(
        candidate_commit["commit_sha256"],
        field="candidate.commit_sha256",
    )
    base_name = f"{source.name}-{commit_sha256}"
    destination = preservation_directory / base_name
    attempt_index = 0
    while destination.exists() or destination.is_symlink():
        attempt_index += 1
        destination = preservation_directory / (
            f"{base_name}-attempt-{attempt_index:04d}"
        )
    os.rename(source, destination)
    _fsync_directory(root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY)
    _fsync_directory(preservation_directory)


def _load_current_internal(
    root: Path,
    *,
    limits: _Limits,
    allow_complete_successor: bool = False,
) -> dict[str, object] | None:
    _validate_root_layout(
        root,
        limits=limits,
        allow_complete_successor=allow_complete_successor,
    )
    pointer_path = root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
    if not pointer_path.exists() and not pointer_path.is_symlink():
        return None
    pointer = _load_pointer(pointer_path, max_bytes=limits.max_pointer_bytes)
    generation_index = int(pointer["aggregate_generation_index"])
    loaded = _load_generation_internal(
        root,
        generation_index=generation_index,
        limits=limits,
        skip_root_layout=True,
    )
    commit = _mapping(loaded["commit"], field="current.commit")
    if (
        pointer["directory_name"] != commit["directory_name"]
        or pointer["commit_sha256"] != commit["commit_sha256"]
    ):
        raise OpenEcologyAggregateCommitError(
            "CURRENT pointer does not match its generation commit"
        )
    return loaded


def _load_generation_internal(
    root: Path,
    *,
    generation_index: int,
    limits: _Limits,
    skip_root_layout: bool = False,
) -> dict[str, object]:
    selected_index = _nonnegative_int(
        generation_index,
        field="aggregate_generation_index",
    )
    if selected_index >= limits.max_generations:
        raise OpenEcologyAggregateCommitError(
            "aggregate_generation_index exceeds max_generations"
        )
    if not skip_root_layout:
        _validate_root_layout(
            root,
            limits=limits,
            allow_complete_successor=True,
        )
    directory_name = _generation_directory_name(selected_index)
    generation_path = _require_directory(
        root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY / directory_name,
        field="aggregate generation",
    )
    _validate_generation_directory_entries(generation_path)
    commit_path = generation_path / OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME
    commit = _load_ascii_json(
        commit_path,
        max_bytes=limits.max_commit_bytes,
        field="aggregate commit",
    )
    normalized_commit = _validate_commit(
        commit,
        expected_generation_index=selected_index,
        expected_directory_name=directory_name,
    )

    identity = _mapping(normalized_commit["identity"], field="commit.identity")
    checkpoint_binding = _mapping(
        normalized_commit["checkpoint"],
        field="commit.checkpoint",
    )
    checkpoint_path = generation_path / str(checkpoint_binding["file_name"])
    checkpoint_file_sha256, checkpoint_file_bytes = _sha256_regular_file(
        checkpoint_path,
        max_bytes=limits.max_checkpoint_bytes,
        field="aggregate checkpoint",
    )
    if (
        checkpoint_file_sha256 != checkpoint_binding["file_sha256"]
        or checkpoint_file_bytes != checkpoint_binding["byte_count"]
    ):
        raise OpenEcologyAggregateCommitError(
            "aggregate checkpoint file binding mismatch"
        )
    checkpoint = load_open_ecology_checkpoint(
        checkpoint_path,
        max_checkpoint_bytes=limits.max_checkpoint_bytes,
        expected_source_git_sha=str(identity["source_git_sha"]),
        expected_generation_identity_sha256=str(
            checkpoint_binding["generation_identity_sha256"]
        ),
        require_restartable=True,
    )
    if checkpoint["checkpoint_sha256"] != checkpoint_binding["checkpoint_sha256"]:
        raise OpenEcologyAggregateCommitError(
            "aggregate checkpoint internal SHA256 mismatch"
        )
    _validate_checkpoint_identity(
        checkpoint,
        identity=_identity_pins_from_commit(identity),
    )

    evidence_binding = _mapping(
        normalized_commit["evidence"],
        field="commit.evidence",
    )
    evidence_manifest_path = generation_path / str(
        evidence_binding["manifest_file_name"]
    )
    evidence_file_sha256, evidence_file_bytes = _sha256_regular_file(
        evidence_manifest_path,
        max_bytes=limits.max_evidence_manifest_bytes,
        field="aggregate evidence manifest snapshot",
    )
    if (
        evidence_file_sha256 != evidence_binding["manifest_file_sha256"]
        or evidence_file_bytes != evidence_binding["manifest_byte_count"]
    ):
        raise OpenEcologyAggregateCommitError(
            "aggregate evidence manifest file binding mismatch"
        )
    evidence_manifest_bytes = _read_regular_file(
        evidence_manifest_path,
        max_bytes=limits.max_evidence_manifest_bytes,
        field="aggregate evidence manifest snapshot",
    )
    evidence_manifest = _validate_evidence_manifest_snapshot(
        evidence_manifest_bytes,
        expected_manifest_sha256=str(evidence_binding["manifest_sha256"]),
        expected_status=str(evidence_binding["status"]),
    )
    _validate_manifest_binding(evidence_manifest, evidence_binding=evidence_binding)
    _bind_manifest_source_contract_to_identity(
        evidence_manifest,
        identity=_identity_pins_from_commit(identity),
    )
    continuation = _checkpoint_evidence_continuation(checkpoint)
    _bind_continuation_to_manifest(
        continuation,
        evidence_manifest,
        checkpoint_tick=int(identity["tick"]),
    )
    if continuation["state_sha256"] != evidence_binding["continuation_state_sha256"]:
        raise OpenEcologyAggregateCommitError(
            "checkpoint continuation digest does not match aggregate commit"
        )
    return {
        "checkpoint": checkpoint,
        "commit": normalized_commit,
        "evidence_manifest": evidence_manifest,
    }


def _validate_next_generation_request(
    *,
    selected_index: int,
    expected_previous_commit_sha256: str | None,
    previous: Mapping[str, object] | None,
) -> None:
    if previous is None:
        if selected_index != 0:
            raise OpenEcologyAggregateCommitError(
                "first aggregate generation index must be zero"
            )
        if expected_previous_commit_sha256 is not None:
            raise OpenEcologyAggregateCommitError(
                "first aggregate generation must not name a predecessor"
            )
        return
    previous_commit = _mapping(previous["commit"], field="previous.commit")
    expected_index = int(previous_commit["aggregate_generation_index"]) + 1
    if selected_index != expected_index:
        raise OpenEcologyAggregateCommitError(
            "aggregate generation indexes must be contiguous"
        )
    if expected_previous_commit_sha256 != previous_commit["commit_sha256"]:
        raise OpenEcologyAggregateCommitError(
            "expected_previous_commit_sha256 does not match CURRENT"
        )


def _validate_genesis_generation(
    loaded: Mapping[str, object],
    *,
    identity: OpenEcologyAggregateIdentityPins,
) -> None:
    commit = _mapping(loaded["commit"], field="genesis.commit")
    if (
        commit["aggregate_generation_index"] != 0
        or commit["previous"] is not None
        or _identity_pins_from_commit(
            _mapping(commit["identity"], field="genesis.commit.identity")
        )
        != identity
    ):
        raise OpenEcologyAggregateCommitError(
            "aggregate generation is not the expected exact genesis authority"
        )


def _build_commit(
    *,
    aggregate_generation_index: int,
    directory_name: str,
    identity: OpenEcologyAggregateIdentityPins,
    checkpoint: Mapping[str, object],
    checkpoint_bytes: bytes,
    evidence_manifest: Mapping[str, object],
    evidence_manifest_bytes: bytes,
    continuation: Mapping[str, object],
    previous: Mapping[str, object] | None,
) -> dict[str, object]:
    checkpoint_identity = _mapping(
        checkpoint["generation_identity"],
        field="checkpoint.generation_identity",
    )
    identity_payload = _identity_payload(
        identity,
        checkpoint_generation_identity_sha256=str(
            checkpoint_identity["identity_sha256"]
        ),
    )
    previous_payload: dict[str, object] | None = None
    if previous is not None:
        previous_commit = _mapping(previous["commit"], field="previous.commit")
        previous_payload = {
            "aggregate_generation_index": previous_commit["aggregate_generation_index"],
            "commit_sha256": previous_commit["commit_sha256"],
            "directory_name": previous_commit["directory_name"],
        }
    body: dict[str, object] = {
        "aggregate_generation_index": aggregate_generation_index,
        "checkpoint": {
            "byte_count": len(checkpoint_bytes),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "file_name": OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME,
            "file_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "generation_identity_sha256": checkpoint_identity["identity_sha256"],
            "restartable": True,
        },
        "directory_name": directory_name,
        "evidence": {
            "chain_head_sha256": evidence_manifest["chain_head_sha256"],
            "completed_shard_count": evidence_manifest["completed_shard_count"],
            "continuation_state_sha256": continuation["state_sha256"],
            "last_tick": evidence_manifest["last_tick"],
            "manifest_byte_count": len(evidence_manifest_bytes),
            "manifest_file_name": (OPEN_ECOLOGY_AGGREGATE_EVIDENCE_MANIFEST_NAME),
            "manifest_file_sha256": hashlib.sha256(evidence_manifest_bytes).hexdigest(),
            "manifest_sha256": evidence_manifest["manifest_sha256"],
            "next_event_index": evidence_manifest["next_event_index"],
            "run_id": evidence_manifest["run_id"],
            "source_contract_sha256": evidence_manifest["source_contract_sha256"],
            "status": evidence_manifest["status"],
            "total_compressed_bytes": evidence_manifest["total_compressed_bytes"],
            "total_event_count": evidence_manifest["total_event_count"],
            "writer_config_sha256": evidence_manifest["writer_config_sha256"],
        },
        "format_contract": _canonical_clone(
            _FORMAT_CONTRACT,
            field="format_contract",
        ),
        "identity": identity_payload,
        "previous": previous_payload,
        "schema_version": OPEN_ECOLOGY_AGGREGATE_COMMIT_SCHEMA,
        "substrate_status": _SUBSTRATE_STATUS,
    }
    return {**body, "commit_sha256": _digest(body)}


def _identity_payload(
    identity: OpenEcologyAggregateIdentityPins,
    *,
    checkpoint_generation_identity_sha256: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "checkpoint_generation_identity_sha256": _sha256(
            checkpoint_generation_identity_sha256,
            field="checkpoint_generation_identity_sha256",
        ),
        "config_contract_sha256": identity.config_contract_sha256,
        "digest_policy": OPEN_ECOLOGY_AGGREGATE_IDENTITY_DIGEST_POLICY,
        "island_id": identity.island_id,
        "run_generation_id": identity.run_generation_id,
        "schema_version": OPEN_ECOLOGY_AGGREGATE_IDENTITY_SCHEMA,
        "seed_contract_sha256": identity.seed_contract_sha256,
        "simulation_generation_index": identity.simulation_generation_index,
        "source_git_sha": identity.source_git_sha,
        "tick": identity.tick,
    }
    return {**body, "identity_sha256": _digest(body)}


def _pointer_for_commit(commit: Mapping[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "aggregate_generation_index": commit["aggregate_generation_index"],
        "commit_sha256": commit["commit_sha256"],
        "digest_policy": OPEN_ECOLOGY_AGGREGATE_POINTER_DIGEST_POLICY,
        "directory_name": commit["directory_name"],
        "schema_version": OPEN_ECOLOGY_AGGREGATE_POINTER_SCHEMA,
    }
    return {**body, "pointer_sha256": _digest(body)}


def _publish_generation_directory(
    root: Path,
    *,
    directory_name: str,
    checkpoint_bytes: bytes,
    evidence_manifest_bytes: bytes,
    commit_bytes: bytes,
    max_cleanup_tombstones: int,
) -> None:
    generations = root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
    destination = generations / directory_name
    if destination.exists() or destination.is_symlink():
        raise OpenEcologyAggregateCommitError(
            "aggregate generation directory already exists and is immutable"
        )
    stage_prefix = f".{root.name}.{directory_name}.stage-"
    stage, stage_identity = _create_owned_stage_directory(
        root.parent,
        prefix=stage_prefix,
    )
    renamed = False
    try:
        _write_new_file(
            stage / OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME,
            checkpoint_bytes,
        )
        _write_new_file(
            stage / OPEN_ECOLOGY_AGGREGATE_EVIDENCE_MANIFEST_NAME,
            evidence_manifest_bytes,
        )
        _write_new_file(
            stage / OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME,
            commit_bytes,
        )
        _fsync_directory(stage)
        if (
            _directory_identity(
                stage,
                field="aggregate staging directory before publication",
            )
            != stage_identity
        ):
            raise OpenEcologyAggregateCommitError(
                "aggregate staging directory changed before publication"
            )
        os.rename(stage, destination)
        if (
            _directory_identity(
                destination,
                field="published aggregate generation directory",
            )
            != stage_identity
        ):
            raise OpenEcologyAggregateCommitError(
                "aggregate staging directory changed during publication"
            )
        renamed = True
        _fsync_directory(generations)
    finally:
        if not renamed:
            _remove_owned_stage(
                stage,
                parent=root.parent,
                prefix=stage_prefix,
                expected_identity=stage_identity,
                cleanup_root=(root / OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME),
                max_tombstones=max_cleanup_tombstones,
            )


def _publish_current_pointer(
    root: Path,
    pointer: Mapping[str, object],
    *,
    max_pointer_bytes: int,
    max_cleanup_tombstones: int,
) -> None:
    validated = _validate_pointer(pointer)
    encoded = _canonical_ascii_line(validated)
    if len(encoded) > max_pointer_bytes:
        raise OpenEcologyAggregateCommitError(
            "CURRENT pointer exceeds max_pointer_bytes"
        )
    destination = root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
    temp_path: Path | None = None
    temp_identity: _PathIdentity | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=root.parent,
            prefix=f".{root.name}.CURRENT.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_identity = _path_identity_from_stat(os.fstat(temp_file.fileno()))
            temp_file.write(encoded)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, destination)
        temp_path = None
        _fsync_directory(root)
    finally:
        if temp_path is not None and temp_identity is not None:
            _unlink_owned_regular_file(
                temp_path,
                expected_identity=temp_identity,
                field="aggregate CURRENT temp",
                missing_ok=True,
                cleanup_root=(root / OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME),
                max_tombstones=max_cleanup_tombstones,
            )


def _validate_commit(
    commit: Mapping[str, object],
    *,
    expected_generation_index: int,
    expected_directory_name: str,
) -> dict[str, object]:
    _exact_keys(
        commit,
        frozenset(
            {
                "aggregate_generation_index",
                "checkpoint",
                "commit_sha256",
                "directory_name",
                "evidence",
                "format_contract",
                "identity",
                "previous",
                "schema_version",
                "substrate_status",
            }
        ),
        field="aggregate commit",
    )
    if commit["schema_version"] != OPEN_ECOLOGY_AGGREGATE_COMMIT_SCHEMA:
        raise OpenEcologyAggregateCommitError("aggregate commit schema is stale")
    if not _canonical_equal(commit["format_contract"], _FORMAT_CONTRACT):
        raise OpenEcologyAggregateCommitError(
            "aggregate commit format contract mismatch"
        )
    if commit["substrate_status"] != _SUBSTRATE_STATUS:
        raise OpenEcologyAggregateCommitError(
            "aggregate commit substrate status mismatch"
        )
    index = _nonnegative_int(
        commit["aggregate_generation_index"],
        field="commit.aggregate_generation_index",
    )
    if index != expected_generation_index:
        raise OpenEcologyAggregateCommitError(
            "aggregate commit generation index mismatch"
        )
    if commit["directory_name"] != expected_directory_name:
        raise OpenEcologyAggregateCommitError(
            "aggregate commit directory name mismatch"
        )
    identity = _validate_identity(commit["identity"])
    checkpoint = _mapping(commit["checkpoint"], field="commit.checkpoint")
    _exact_keys(
        checkpoint,
        frozenset(
            {
                "byte_count",
                "checkpoint_sha256",
                "file_name",
                "file_sha256",
                "generation_identity_sha256",
                "restartable",
            }
        ),
        field="commit.checkpoint",
    )
    if checkpoint["file_name"] != OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME:
        raise OpenEcologyAggregateCommitError(
            "aggregate checkpoint file name is not canonical"
        )
    _positive_int(checkpoint["byte_count"], field="checkpoint.byte_count")
    for field_name in (
        "checkpoint_sha256",
        "file_sha256",
        "generation_identity_sha256",
    ):
        _sha256(checkpoint[field_name], field=f"checkpoint.{field_name}")
    if checkpoint["restartable"] is not True:
        raise OpenEcologyAggregateCommitError(
            "aggregate checkpoint must be container-restartable"
        )
    if (
        checkpoint["generation_identity_sha256"]
        != identity["checkpoint_generation_identity_sha256"]
    ):
        raise OpenEcologyAggregateCommitError(
            "commit checkpoint identity binding mismatch"
        )

    evidence = _mapping(commit["evidence"], field="commit.evidence")
    _exact_keys(
        evidence,
        frozenset(
            {
                "chain_head_sha256",
                "completed_shard_count",
                "continuation_state_sha256",
                "last_tick",
                "manifest_byte_count",
                "manifest_file_name",
                "manifest_file_sha256",
                "manifest_sha256",
                "next_event_index",
                "run_id",
                "source_contract_sha256",
                "status",
                "total_compressed_bytes",
                "total_event_count",
                "writer_config_sha256",
            }
        ),
        field="commit.evidence",
    )
    if evidence["manifest_file_name"] != OPEN_ECOLOGY_AGGREGATE_EVIDENCE_MANIFEST_NAME:
        raise OpenEcologyAggregateCommitError(
            "aggregate evidence manifest file name is not canonical"
        )
    if evidence["status"] != "open":
        raise OpenEcologyAggregateCommitError(
            "aggregate evidence manifest status must be open"
        )
    _identifier(evidence["run_id"], field="evidence.run_id")
    for field_name in (
        "chain_head_sha256",
        "continuation_state_sha256",
        "manifest_file_sha256",
        "manifest_sha256",
        "source_contract_sha256",
        "writer_config_sha256",
    ):
        _sha256(evidence[field_name], field=f"evidence.{field_name}")
    for field_name in (
        "completed_shard_count",
        "manifest_byte_count",
        "next_event_index",
        "total_compressed_bytes",
        "total_event_count",
    ):
        _nonnegative_int(evidence[field_name], field=f"evidence.{field_name}")
    _optional_nonnegative_int(evidence["last_tick"], field="evidence.last_tick")

    previous = commit["previous"]
    if index == 0:
        if previous is not None:
            raise OpenEcologyAggregateCommitError(
                "initial aggregate commit must not have a predecessor"
            )
    else:
        previous_payload = _mapping(previous, field="commit.previous")
        _exact_keys(
            previous_payload,
            frozenset(
                {
                    "aggregate_generation_index",
                    "commit_sha256",
                    "directory_name",
                }
            ),
            field="commit.previous",
        )
        previous_index = _nonnegative_int(
            previous_payload["aggregate_generation_index"],
            field="previous.aggregate_generation_index",
        )
        if previous_index != index - 1:
            raise OpenEcologyAggregateCommitError(
                "aggregate predecessor index is not contiguous"
            )
        if previous_payload["directory_name"] != _generation_directory_name(
            previous_index
        ):
            raise OpenEcologyAggregateCommitError(
                "aggregate predecessor directory name mismatch"
            )
        _sha256(
            previous_payload["commit_sha256"],
            field="previous.commit_sha256",
        )

    observed_digest = _sha256(
        commit["commit_sha256"],
        field="commit.commit_sha256",
    )
    body = dict(commit)
    body.pop("commit_sha256")
    if _digest(body) != observed_digest:
        raise OpenEcologyAggregateCommitError("aggregate commit SHA256 mismatch")
    return _canonical_clone(commit, field="aggregate commit")


def _validate_identity(identity: object) -> dict[str, object]:
    payload = _mapping(identity, field="commit.identity")
    _exact_keys(
        payload,
        frozenset(
            {
                "checkpoint_generation_identity_sha256",
                "config_contract_sha256",
                "digest_policy",
                "identity_sha256",
                "island_id",
                "run_generation_id",
                "schema_version",
                "seed_contract_sha256",
                "simulation_generation_index",
                "source_git_sha",
                "tick",
            }
        ),
        field="commit.identity",
    )
    if payload["schema_version"] != OPEN_ECOLOGY_AGGREGATE_IDENTITY_SCHEMA:
        raise OpenEcologyAggregateCommitError("aggregate identity schema is stale")
    if payload["digest_policy"] != OPEN_ECOLOGY_AGGREGATE_IDENTITY_DIGEST_POLICY:
        raise OpenEcologyAggregateCommitError(
            "aggregate identity digest policy mismatch"
        )
    _git_sha(payload["source_git_sha"], field="identity.source_git_sha")
    for field_name in (
        "checkpoint_generation_identity_sha256",
        "config_contract_sha256",
        "seed_contract_sha256",
    ):
        _sha256(payload[field_name], field=f"identity.{field_name}")
    _identifier(payload["run_generation_id"], field="identity.run_generation_id")
    _identifier(payload["island_id"], field="identity.island_id")
    _nonnegative_int(
        payload["simulation_generation_index"],
        field="identity.simulation_generation_index",
    )
    _nonnegative_int(payload["tick"], field="identity.tick")
    observed_digest = _sha256(
        payload["identity_sha256"],
        field="identity.identity_sha256",
    )
    body = dict(payload)
    body.pop("identity_sha256")
    if _digest(body) != observed_digest:
        raise OpenEcologyAggregateCommitError("aggregate identity SHA256 mismatch")
    return _canonical_clone(payload, field="aggregate identity")


def _validate_pointer(pointer: Mapping[str, object]) -> dict[str, object]:
    _exact_keys(
        pointer,
        frozenset(
            {
                "aggregate_generation_index",
                "commit_sha256",
                "digest_policy",
                "directory_name",
                "pointer_sha256",
                "schema_version",
            }
        ),
        field="CURRENT pointer",
    )
    if pointer["schema_version"] != OPEN_ECOLOGY_AGGREGATE_POINTER_SCHEMA:
        raise OpenEcologyAggregateCommitError("CURRENT pointer schema is stale")
    if pointer["digest_policy"] != OPEN_ECOLOGY_AGGREGATE_POINTER_DIGEST_POLICY:
        raise OpenEcologyAggregateCommitError("CURRENT pointer digest policy mismatch")
    index = _nonnegative_int(
        pointer["aggregate_generation_index"],
        field="CURRENT.aggregate_generation_index",
    )
    if pointer["directory_name"] != _generation_directory_name(index):
        raise OpenEcologyAggregateCommitError("CURRENT pointer directory name mismatch")
    _sha256(pointer["commit_sha256"], field="CURRENT.commit_sha256")
    observed_digest = _sha256(
        pointer["pointer_sha256"],
        field="CURRENT.pointer_sha256",
    )
    body = dict(pointer)
    body.pop("pointer_sha256")
    if _digest(body) != observed_digest:
        raise OpenEcologyAggregateCommitError("CURRENT pointer SHA256 mismatch")
    return _canonical_clone(pointer, field="CURRENT pointer")


def _load_pointer(path: Path, *, max_bytes: int) -> dict[str, object]:
    return _validate_pointer(
        _load_ascii_json(path, max_bytes=max_bytes, field="CURRENT pointer")
    )


def _validate_checkpoint_identity(
    checkpoint: Mapping[str, object],
    *,
    identity: OpenEcologyAggregateIdentityPins,
) -> None:
    source = _mapping(checkpoint["source"], field="checkpoint.source")
    generation_identity = _mapping(
        checkpoint["generation_identity"],
        field="checkpoint.generation_identity",
    )
    config_contract = _mapping(
        source["config_contract"],
        field="checkpoint.source.config_contract",
    )
    seed_contract = _mapping(
        source["seed_contract"],
        field="checkpoint.source.seed_contract",
    )
    comparisons = {
        "source_git_sha": (source["git_sha"], identity.source_git_sha),
        "config_contract_sha256": (
            config_contract["state_sha256"],
            identity.config_contract_sha256,
        ),
        "seed_contract_sha256": (
            seed_contract["state_sha256"],
            identity.seed_contract_sha256,
        ),
        "run_generation_id": (
            generation_identity["run_generation_id"],
            identity.run_generation_id,
        ),
        "island_id": (generation_identity["island_id"], identity.island_id),
        "simulation_generation_index": (
            generation_identity["generation_index"],
            identity.simulation_generation_index,
        ),
        "tick": (checkpoint["tick"], identity.tick),
    }
    for field_name, (observed, expected) in comparisons.items():
        if observed != expected:
            raise OpenEcologyAggregateCommitError(
                f"checkpoint {field_name} does not match aggregate identity"
            )


def _checkpoint_evidence_continuation(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    components = _mapping(checkpoint["components"], field="checkpoint.components")
    envelope = _mapping(
        components["evidence_writer_continuation_state"],
        field="checkpoint evidence continuation envelope",
    )
    if envelope["present"] is not True:
        raise OpenEcologyAggregateCommitError(
            "checkpoint must contain an evidence continuation envelope"
        )
    payload = _mapping(
        envelope["payload"],
        field="checkpoint evidence continuation payload",
    )
    schema_version = envelope["schema_version"]
    if schema_version == _OPEN_ECOLOGY_RUNTIME_EVIDENCE_ADAPTER_SCHEMA_VERSION:
        from evolution_sim.io.open_ecology_runtime_checkpoint import (
            OpenEcologyRuntimeCheckpointError,
            extract_open_ecology_runtime_evidence_continuation,
        )

        try:
            extracted = extract_open_ecology_runtime_evidence_continuation(
                adapter_schema_version=schema_version,
                payload=payload,
            )
        except OpenEcologyRuntimeCheckpointError as error:
            raise OpenEcologyAggregateCommitError(
                f"checkpoint runtime evidence adapter is invalid: {error}"
            ) from error
        _bind_runtime_evidence_adapter_to_checkpoint(
            extracted,
            checkpoint=checkpoint,
        )
        return extracted.continuation_state
    if schema_version != OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA:
        raise OpenEcologyAggregateCommitError(
            "checkpoint evidence continuation schema is unsupported"
        )
    try:
        continuation = validate_open_ecology_evidence_continuation_state(payload)
    except ValueError as error:
        raise OpenEcologyAggregateCommitError(
            f"checkpoint evidence continuation is invalid: {error}"
        ) from error
    return continuation


def _bind_runtime_evidence_adapter_to_checkpoint(
    extracted: ExtractedRuntimeEvidenceContinuation,
    *,
    checkpoint: Mapping[str, object],
) -> None:
    source = _mapping(checkpoint["source"], field="checkpoint.source")
    config_contract = _mapping(
        source["config_contract"],
        field="checkpoint.source.config_contract",
    )
    seed_contract = _mapping(
        source["seed_contract"],
        field="checkpoint.source.seed_contract",
    )
    generation_identity = _mapping(
        checkpoint["generation_identity"],
        field="checkpoint.generation_identity",
    )
    binding = extracted.binding
    comparisons = {
        "source_git_sha": (binding.source_git_sha, source["git_sha"]),
        "config_contract_sha256": (
            binding.config_contract_sha256,
            config_contract["state_sha256"],
        ),
        "seed_contract_sha256": (
            binding.seed_contract_sha256,
            seed_contract["state_sha256"],
        ),
        "run_generation_id": (
            binding.run_generation_id,
            generation_identity["run_generation_id"],
        ),
        "island_id": (binding.island_id, generation_identity["island_id"]),
        "generation_index": (
            binding.generation_index,
            generation_identity["generation_index"],
        ),
        "completed_tick": (binding.completed_tick, checkpoint["tick"]),
    }
    for field_name, (observed, expected) in comparisons.items():
        if observed != expected:
            raise OpenEcologyAggregateCommitError(
                f"runtime evidence adapter {field_name} does not match checkpoint"
            )
    config_payload = _mapping(
        config_contract["payload"],
        field="checkpoint.source.config_contract.payload",
    )
    if config_payload.get("source_manifest_sha256") != (binding.source_manifest_sha256):
        raise OpenEcologyAggregateCommitError(
            "runtime evidence adapter source manifest does not match checkpoint"
        )


def _bind_continuation_to_manifest(
    continuation: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    checkpoint_tick: int,
) -> None:
    comparisons = (
        "chain_head_sha256",
        "completed_shard_count",
        "last_tick",
        "manifest_sha256",
        "next_event_index",
        "next_shard_index",
        "run_id",
        "source_contract_sha256",
        "total_compressed_bytes",
        "total_event_count",
        "writer_config_sha256",
    )
    for field_name in comparisons:
        if continuation[field_name] != manifest[field_name]:
            raise OpenEcologyAggregateCommitError(
                f"evidence continuation {field_name} does not match manifest"
            )
    if manifest["status"] != "open":
        raise OpenEcologyAggregateCommitError(
            "evidence continuation requires an open manifest"
        )
    last_tick = _optional_nonnegative_int(
        manifest["last_tick"],
        field="manifest.last_tick",
    )
    if last_tick is not None and last_tick > checkpoint_tick:
        raise OpenEcologyAggregateCommitError(
            "evidence manifest extends beyond the checkpoint tick"
        )


def _validate_manifest_binding(
    manifest: Mapping[str, object],
    *,
    evidence_binding: Mapping[str, object],
) -> None:
    comparisons = (
        "chain_head_sha256",
        "completed_shard_count",
        "last_tick",
        "manifest_sha256",
        "next_event_index",
        "run_id",
        "source_contract_sha256",
        "status",
        "total_compressed_bytes",
        "total_event_count",
        "writer_config_sha256",
    )
    for field_name in comparisons:
        if manifest[field_name] != evidence_binding[field_name]:
            raise OpenEcologyAggregateCommitError(
                f"evidence manifest {field_name} does not match aggregate commit"
            )


def _bind_manifest_source_contract_to_identity(
    manifest: Mapping[str, object],
    *,
    identity: OpenEcologyAggregateIdentityPins,
) -> None:
    source_contract = _mapping(
        manifest["source_contract"],
        field="evidence manifest source_contract",
    )
    required = {
        "config_contract_sha256": identity.config_contract_sha256,
        "island_id": identity.island_id,
        "run_generation_id": identity.run_generation_id,
        "seed_contract_sha256": identity.seed_contract_sha256,
        "source_git_sha": identity.source_git_sha,
    }
    for field_name, expected in required.items():
        if source_contract.get(field_name) != expected:
            raise OpenEcologyAggregateCommitError(
                "evidence source contract "
                f"{field_name} does not match aggregate identity"
            )


def _validate_evidence_manifest_snapshot(
    encoded: bytes,
    *,
    expected_manifest_sha256: str,
    expected_status: str,
) -> dict[str, object]:
    parsed = _parse_json(
        encoded,
        field="aggregate evidence manifest snapshot",
    )
    if not isinstance(parsed, dict):
        raise OpenEcologyAggregateCommitError(
            "aggregate evidence manifest root must be an object"
        )
    if encoded != _canonical_utf8_line(parsed):
        raise OpenEcologyAggregateCommitError(
            "aggregate evidence manifest snapshot is not canonical"
        )
    completed_shards = parsed.get("completed_shards")
    if not isinstance(completed_shards, list):
        raise OpenEcologyAggregateCommitError(
            "aggregate evidence manifest completed_shards must be a list"
        )
    with TemporaryDirectory(prefix="open-ecology-manifest-validate-") as tmpdir:
        temp = Path(tmpdir)
        _write_new_file(temp / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME, encoded)
        _write_new_file(temp / OPEN_ECOLOGY_EVIDENCE_LOCK_NAME, _EVIDENCE_LOCK_BYTES)
        declared_files: set[str] = set()
        for entry in completed_shards:
            entry_payload = _mapping(entry, field="manifest shard entry")
            file_name = entry_payload.get("file_name")
            if (
                not isinstance(file_name, str)
                or Path(file_name).name != file_name
                or file_name
                in {
                    OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME,
                    OPEN_ECOLOGY_EVIDENCE_LOCK_NAME,
                }
            ):
                raise OpenEcologyAggregateCommitError(
                    "evidence manifest shard path is not a safe basename"
                )
            if file_name in declared_files:
                raise OpenEcologyAggregateCommitError(
                    "evidence manifest contains duplicate shard file names"
                )
            declared_files.add(file_name)
            _write_new_file(temp / file_name, b"")
        try:
            return load_open_ecology_evidence_manifest(
                temp,
                verify_shards=False,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_status=expected_status,
            )
        except ValueError as error:
            raise OpenEcologyAggregateCommitError(
                f"aggregate evidence manifest is invalid: {error}"
            ) from error


def _validate_external_evidence_prefix(
    evidence_directory: Path,
    *,
    evidence_manifest: Mapping[str, object],
) -> None:
    lock_bytes = _read_regular_file(
        evidence_directory / OPEN_ECOLOGY_EVIDENCE_LOCK_NAME,
        max_bytes=len(_EVIDENCE_LOCK_BYTES),
        field="external evidence lock",
        require_single_link=True,
    )
    if lock_bytes != _EVIDENCE_LOCK_BYTES:
        raise OpenEcologyAggregateCommitError(
            "external evidence lock does not match its exact control contract"
        )
    writer_config = _mapping(
        evidence_manifest["writer_config"],
        field="evidence_manifest.writer_config",
    )
    max_shard_bytes = _positive_int(
        writer_config["max_compressed_bytes_per_shard"],
        field="writer_config.max_compressed_bytes_per_shard",
    )
    completed_shards = evidence_manifest["completed_shards"]
    if not isinstance(completed_shards, list):
        raise OpenEcologyAggregateCommitError(
            "evidence manifest completed_shards must be a list"
        )
    for entry in completed_shards:
        payload = _mapping(entry, field="evidence manifest shard")
        file_name = payload["file_name"]
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise OpenEcologyAggregateCommitError(
                "evidence shard file name is not a safe basename"
            )
        expected_size = _positive_int(
            payload["compressed_bytes"],
            field="evidence shard compressed_bytes",
        )
        observed_sha256, observed_size = _sha256_regular_file(
            evidence_directory / file_name,
            max_bytes=max_shard_bytes,
            field=f"external evidence shard {file_name}",
        )
        if observed_size != expected_size or observed_sha256 != payload["file_sha256"]:
            raise OpenEcologyAggregateCommitError(
                f"external evidence shard {file_name} does not match snapshot"
            )


def _validate_resume_pins(
    loaded: Mapping[str, object],
    *,
    pins: OpenEcologyAggregateResumePins,
) -> None:
    commit = _mapping(loaded["commit"], field="loaded.commit")
    identity = _mapping(commit["identity"], field="loaded.commit.identity")
    checkpoint_binding = _mapping(
        commit["checkpoint"],
        field="loaded.commit.checkpoint",
    )
    evidence = _mapping(commit["evidence"], field="loaded.commit.evidence")
    expected_identity = pins.identity
    comparisons = {
        "aggregate_generation_index": (
            commit["aggregate_generation_index"],
            pins.aggregate_generation_index,
        ),
        "commit_sha256": (commit["commit_sha256"], pins.commit_sha256),
        "source_git_sha": (
            identity["source_git_sha"],
            expected_identity.source_git_sha,
        ),
        "config_contract_sha256": (
            identity["config_contract_sha256"],
            expected_identity.config_contract_sha256,
        ),
        "seed_contract_sha256": (
            identity["seed_contract_sha256"],
            expected_identity.seed_contract_sha256,
        ),
        "run_generation_id": (
            identity["run_generation_id"],
            expected_identity.run_generation_id,
        ),
        "island_id": (identity["island_id"], expected_identity.island_id),
        "simulation_generation_index": (
            identity["simulation_generation_index"],
            expected_identity.simulation_generation_index,
        ),
        "tick": (identity["tick"], expected_identity.tick),
        "checkpoint_sha256": (
            checkpoint_binding["checkpoint_sha256"],
            pins.checkpoint_sha256,
        ),
        "checkpoint_generation_identity_sha256": (
            checkpoint_binding["generation_identity_sha256"],
            pins.checkpoint_generation_identity_sha256,
        ),
        "evidence_manifest_sha256": (
            evidence["manifest_sha256"],
            pins.evidence_manifest_sha256,
        ),
        "evidence_manifest_status": (
            evidence["status"],
            pins.evidence_manifest_status,
        ),
    }
    for field_name, (observed, expected) in comparisons.items():
        if observed != expected:
            raise OpenEcologyAggregateCommitError(f"resume {field_name} pin mismatch")


def _identity_pins_from_commit(
    identity: Mapping[str, object],
) -> OpenEcologyAggregateIdentityPins:
    return OpenEcologyAggregateIdentityPins(
        source_git_sha=str(identity["source_git_sha"]),
        config_contract_sha256=str(identity["config_contract_sha256"]),
        seed_contract_sha256=str(identity["seed_contract_sha256"]),
        run_generation_id=str(identity["run_generation_id"]),
        island_id=str(identity["island_id"]),
        simulation_generation_index=int(identity["simulation_generation_index"]),
        tick=int(identity["tick"]),
    )


def _validate_root_layout(
    root: Path,
    *,
    limits: _Limits,
    allow_complete_successor: bool,
) -> None:
    root = _require_directory(root, field="aggregate_root")
    names = {entry.name for entry in root.iterdir()}
    cleanup_present = OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME in names
    authority_names = names - {OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME}
    if authority_names not in {
        _ROOT_ENTRIES_WITHOUT_CURRENT,
        _ROOT_ENTRIES_WITH_CURRENT,
    }:
        raise OpenEcologyAggregateCommitError(
            "aggregate root contains missing or surplus entries"
        )
    if cleanup_present:
        _reconcile_aggregate_cleanup_tombstones(
            root,
            max_tombstones=_aggregate_cleanup_tombstone_limit(limits.max_generations),
            repair=False,
            limits=limits,
        )
    generations = _require_directory(
        root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY,
        field="aggregate generations",
    )
    indexes = _generation_indexes(
        root,
        max_generations=limits.max_generations,
    )
    pointer_path = root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
    if pointer_path.exists() or pointer_path.is_symlink():
        pointer = _load_pointer(
            pointer_path,
            max_bytes=limits.max_pointer_bytes,
        )
        current_index = int(pointer["aggregate_generation_index"])
        maximum_allowed = current_index + (1 if allow_complete_successor else 0)
        if not indexes or indexes[-1] > maximum_allowed:
            raise OpenEcologyAggregateCommitError(
                "aggregate root contains uncommitted generation directories"
            )
    elif len(indexes) > (1 if allow_complete_successor else 0):
        raise OpenEcologyAggregateCommitError(
            "aggregate root contains generations without CURRENT"
        )
    for entry in generations.iterdir():
        _require_directory(entry, field=f"aggregate generation {entry.name}")
        _validate_generation_directory_entries(entry)


def _generation_indexes(root: Path, *, max_generations: int) -> list[int]:
    generations = _require_directory(
        root / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY,
        field="aggregate generations",
    )
    entries = list(generations.iterdir())
    if len(entries) > max_generations:
        raise OpenEcologyAggregateCommitError("aggregate store exceeds max_generations")
    indexes: list[int] = []
    for entry in entries:
        match = _GENERATION_DIRECTORY_RE.fullmatch(entry.name)
        if match is None:
            raise OpenEcologyAggregateCommitError(
                "aggregate generations directory contains a surplus entry"
            )
        indexes.append(int(match.group(1)))
    indexes.sort()
    if indexes != list(range(len(indexes))):
        raise OpenEcologyAggregateCommitError(
            "aggregate generation directories must be contiguous from zero"
        )
    return indexes


def _validate_generation_directory_entries(path: Path) -> None:
    names = {entry.name for entry in path.iterdir()}
    if names != _GENERATION_FILES:
        raise OpenEcologyAggregateCommitError(
            "aggregate generation contains missing or surplus files"
        )
    for name in _GENERATION_FILES:
        file_path = path / name
        try:
            file_stat = file_path.lstat()
        except OSError as error:
            raise OpenEcologyAggregateCommitError(
                f"cannot inspect aggregate generation file {name}: {error}"
            ) from error
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise OpenEcologyAggregateCommitError(
                f"aggregate generation file {name} must be single-link regular"
            )


def _generation_directory_name(index: int) -> str:
    if index > 9_999_999_999_999_999:
        raise OpenEcologyAggregateCommitError(
            "aggregate generation index exceeds directory-name capacity"
        )
    return f"generation-{index:016d}"


def _load_ascii_json(
    path: Path,
    *,
    max_bytes: int,
    field: str,
) -> dict[str, object]:
    encoded = _read_regular_file(path, max_bytes=max_bytes, field=field)
    parsed = _parse_json(encoded, field=field)
    if not isinstance(parsed, dict):
        raise OpenEcologyAggregateCommitError(f"{field} root must be an object")
    if encoded != _canonical_ascii_line(parsed):
        raise OpenEcologyAggregateCommitError(f"{field} is not canonical")
    return parsed


def _parse_json(encoded: bytes, *, field: str) -> object:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenEcologyAggregateCommitError(f"{field} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except OpenEcologyAggregateCommitError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise OpenEcologyAggregateCommitError(
            f"{field} is not valid JSON: {error}"
        ) from error


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int,
    field: str,
    require_single_link: bool = True,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OpenEcologyAggregateCommitError(
            f"cannot safely open {field}: {error}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OpenEcologyAggregateCommitError(f"{field} must be a regular file")
        if require_single_link and file_stat.st_nlink != 1:
            raise OpenEcologyAggregateCommitError(
                f"{field} must have exactly one hard link"
            )
        if file_stat.st_size > max_bytes:
            raise OpenEcologyAggregateCommitError(f"{field} exceeds its byte ceiling")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > max_bytes:
            raise OpenEcologyAggregateCommitError(
                f"{field} exceeds its byte ceiling while reading"
            )
        return encoded
    finally:
        os.close(descriptor)


def _sha256_regular_file(
    path: Path,
    *,
    max_bytes: int,
    field: str,
) -> tuple[str, int]:
    encoded = _read_regular_file(
        path,
        max_bytes=max_bytes,
        field=field,
        require_single_link=False,
    )
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _write_new_file(path: Path, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_identity_from_stat(path_stat: os.stat_result) -> _PathIdentity:
    return _PathIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
    )


def _aggregate_cleanup_tombstone_limit(max_generations: int) -> int:
    return (
        _positive_int(max_generations, field="max_generations")
        * _CLEANUP_TOMBSTONE_MULTIPLIER
        + _CLEANUP_TOMBSTONE_RESERVE
    )


def _directory_identity(path: Path, *, field: str) -> _PathIdentity:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise OpenEcologyAggregateCommitError(f"cannot inspect {field}") from error
    if not stat.S_ISDIR(path_stat.st_mode):
        raise OpenEcologyAggregateCommitError(f"{field} must be a real directory")
    return _path_identity_from_stat(path_stat)


def _directory_descriptor_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise OpenEcologyAggregateCommitError(
            "descriptor-relative cleanup requires O_DIRECTORY and O_NOFOLLOW"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_owned_cleanup_parent(
    path: Path,
    *,
    field: str,
) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(path, _directory_descriptor_flags())
    except OSError as error:
        raise OpenEcologyAggregateCommitError(
            f"cannot safely open {field} directory"
        ) from error
    metadata = os.fstat(descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or mode & 0o022
    ):
        os.close(descriptor)
        raise OpenEcologyAggregateCommitError(
            f"{field} directory must be real, process-owned, and not "
            "group/world-writable"
        )
    try:
        current = os.lstat(path)
    except OSError as error:
        os.close(descriptor)
        raise OpenEcologyAggregateCommitError(
            f"cannot revalidate {field} directory"
        ) from error
    if not stat.S_ISDIR(current.st_mode) or _path_identity_from_stat(
        current
    ) != _path_identity_from_stat(metadata):
        os.close(descriptor)
        raise OpenEcologyAggregateCommitError(f"{field} directory namespace changed")
    return descriptor, metadata


def _create_owned_stage_directory(
    parent: Path,
    *,
    prefix: str,
) -> tuple[Path, _PathIdentity]:
    parent_descriptor, parent_metadata = _open_owned_cleanup_parent(
        parent,
        field="aggregate staging parent",
    )
    try:
        for _attempt in range(128):
            name = f"{prefix}{secrets.token_hex(8)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            except OSError as error:
                raise OpenEcologyAggregateCommitError(
                    "cannot create aggregate staging directory"
                ) from error
            metadata = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent_metadata.st_dev
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise OpenEcologyAggregateCommitError(
                    "aggregate staging directory authority is invalid"
                )
            current_parent = os.lstat(parent)
            if not stat.S_ISDIR(current_parent.st_mode) or _path_identity_from_stat(
                current_parent
            ) != _path_identity_from_stat(parent_metadata):
                raise OpenEcologyAggregateCommitError(
                    "aggregate staging parent namespace changed"
                )
            _fsync_directory_descriptor_best_effort(parent_descriptor)
            return parent / name, _path_identity_from_stat(metadata)
    finally:
        os.close(parent_descriptor)
    raise OpenEcologyAggregateCommitError(
        "cannot allocate a unique aggregate staging directory"
    )


def _open_aggregate_cleanup_directory(
    root_descriptor: int,
    *,
    root_metadata: os.stat_result,
    create: bool,
) -> tuple[int, os.stat_result] | None:
    name = OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=root_descriptor)
        except FileExistsError:
            pass
        except OSError as error:
            raise OpenEcologyAggregateCommitError(
                "cannot create aggregate cleanup tombstone directory"
            ) from error
    try:
        descriptor = os.open(
            name,
            _directory_descriptor_flags(),
            dir_fd=root_descriptor,
        )
    except FileNotFoundError:
        if not create:
            return None
        raise
    except OSError as error:
        raise OpenEcologyAggregateCommitError(
            "cannot safely open aggregate cleanup tombstone directory"
        ) from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) not in {0o500, 0o700}
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != root_metadata.st_dev
    ):
        os.close(descriptor)
        raise OpenEcologyAggregateCommitError(
            "aggregate cleanup tombstone directory authority is invalid"
        )
    try:
        namespace_metadata = os.stat(
            name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        os.close(descriptor)
        raise OpenEcologyAggregateCommitError(
            "cannot revalidate aggregate cleanup tombstone namespace"
        ) from error
    if not stat.S_ISDIR(namespace_metadata.st_mode) or _path_identity_from_stat(
        namespace_metadata
    ) != _path_identity_from_stat(metadata):
        os.close(descriptor)
        raise OpenEcologyAggregateCommitError(
            "aggregate cleanup tombstone namespace changed"
        )
    return descriptor, metadata


def _aggregate_cleanup_slot_contract(
    name: str,
) -> tuple[str, _PathIdentity]:
    match = _CLEANUP_SLOT_RE.fullmatch(name)
    if match is None:
        raise OpenEcologyAggregateCommitError(
            "aggregate cleanup tombstone directory contains a surplus entry"
        )
    return (
        match.group("kind"),
        _PathIdentity(
            device=int(match.group("device"), 16),
            inode=int(match.group("inode"), 16),
        ),
    )


def _validate_aggregate_cleanup_directory(
    metadata: os.stat_result,
    *,
    cleanup_device: int,
    field: str,
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) not in {0o500, 0o700}
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != cleanup_device
    ):
        raise OpenEcologyAggregateCommitError(f"{field} authority is invalid")


def _open_aggregate_cleanup_slot(
    cleanup_descriptor: int,
    *,
    cleanup_metadata: os.stat_result,
    kind: str,
    expected_identity: _PathIdentity,
) -> tuple[str, int, os.stat_result]:
    if kind not in {"file", "stage"}:
        raise OpenEcologyAggregateCommitError(
            "aggregate cleanup tombstone kind is invalid"
        )
    for _attempt in range(128):
        slot_name = (
            f"{kind}-d{expected_identity.device:x}-i{expected_identity.inode:x}-"
            f"{secrets.token_hex(8)}"
        )
        try:
            os.mkdir(slot_name, mode=0o700, dir_fd=cleanup_descriptor)
        except FileExistsError:
            continue
        except OSError as error:
            raise OpenEcologyAggregateCommitError(
                "cannot create aggregate cleanup tombstone slot"
            ) from error
        try:
            slot_descriptor = os.open(
                slot_name,
                _directory_descriptor_flags(),
                dir_fd=cleanup_descriptor,
            )
        except OSError as error:
            raise OpenEcologyAggregateCommitError(
                "cannot safely open aggregate cleanup tombstone slot"
            ) from error
        slot_metadata = os.fstat(slot_descriptor)
        _validate_aggregate_cleanup_directory(
            slot_metadata,
            cleanup_device=cleanup_metadata.st_dev,
            field="aggregate cleanup tombstone slot",
        )
        return slot_name, slot_descriptor, slot_metadata
    raise OpenEcologyAggregateCommitError(
        "cannot allocate a unique aggregate cleanup tombstone slot"
    )


def _opened_aggregate_cleanup_file_identity(
    descriptor: int,
    *,
    expected_identity: _PathIdentity,
    cleanup_device: int,
    field: str,
    allow_relocated_zero: bool = False,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != cleanup_device
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise OpenEcologyAggregateCommitError(
            f"{field} changed before descriptor-bound truncation"
        )
    if _path_identity_from_stat(metadata) != expected_identity and not (
        allow_relocated_zero and metadata.st_size == 0
    ):
        raise OpenEcologyAggregateCommitError(
            f"{field} changed before descriptor-bound truncation"
        )
    return metadata


def _stage_file_byte_ceilings(limits: _Limits | None) -> dict[str, int]:
    return {
        OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME: (
            OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES
            if limits is None
            else limits.max_checkpoint_bytes
        ),
        OPEN_ECOLOGY_AGGREGATE_EVIDENCE_MANIFEST_NAME: (
            OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_EVIDENCE_MANIFEST_BYTES
            if limits is None
            else limits.max_evidence_manifest_bytes
        ),
        OPEN_ECOLOGY_AGGREGATE_COMMIT_NAME: (
            OPEN_ECOLOGY_DEFAULT_MAX_AGGREGATE_COMMIT_BYTES
            if limits is None
            else limits.max_commit_bytes
        ),
    }


def _validate_opened_aggregate_stage(
    descriptor: int,
    *,
    expected_identity: _PathIdentity,
    cleanup_device: int,
    limits: _Limits | None,
    field: str,
    allow_relocated: bool = False,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != cleanup_device
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise OpenEcologyAggregateCommitError(f"{field} authority is invalid")
    if _path_identity_from_stat(metadata) != expected_identity and not allow_relocated:
        raise OpenEcologyAggregateCommitError(f"{field} authority is invalid")
    ceilings = _stage_file_byte_ceilings(limits)
    entries = os.listdir(descriptor)
    if len(entries) > len(ceilings) or not set(entries).issubset(ceilings):
        raise OpenEcologyAggregateCommitError(f"{field} contains surplus entries")
    for name in entries:
        try:
            candidate = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise OpenEcologyAggregateCommitError(
                f"cannot inspect {field} file {name}"
            ) from error
        if (
            not stat.S_ISREG(candidate.st_mode)
            or candidate.st_nlink != 1
            or candidate.st_uid != os.geteuid()
            or candidate.st_dev != cleanup_device
            or stat.S_IMODE(candidate.st_mode) & 0o022
            or candidate.st_size > ceilings[name]
        ):
            raise OpenEcologyAggregateCommitError(
                f"{field} file {name} violates its authority or byte ceiling"
            )
    return metadata


def _reconcile_aggregate_cleanup_directory_descriptor(
    cleanup_descriptor: int,
    *,
    cleanup_metadata: os.stat_result,
    max_tombstones: int,
    repair: bool,
    limits: _Limits | None,
) -> int:
    names = sorted(os.listdir(cleanup_descriptor))
    if len(names) > max_tombstones:
        raise OpenEcologyAggregateCommitError(
            "aggregate cleanup tombstone limit exceeded"
        )
    for slot_name in names:
        kind, expected_identity = _aggregate_cleanup_slot_contract(slot_name)
        try:
            slot_descriptor = os.open(
                slot_name,
                _directory_descriptor_flags(),
                dir_fd=cleanup_descriptor,
            )
        except OSError as error:
            raise OpenEcologyAggregateCommitError(
                "cannot safely open aggregate cleanup tombstone slot"
            ) from error
        try:
            slot_metadata = os.fstat(slot_descriptor)
            _validate_aggregate_cleanup_directory(
                slot_metadata,
                cleanup_device=cleanup_metadata.st_dev,
                field="aggregate cleanup tombstone slot",
            )
            entries = os.listdir(slot_descriptor)
            if not entries:
                continue
            if entries != ["candidate"]:
                raise OpenEcologyAggregateCommitError(
                    "aggregate cleanup tombstone slot contains surplus entries"
                )
            if kind == "file":
                flags = os.O_RDWR if repair else os.O_RDONLY
                flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
                    os,
                    "O_NOFOLLOW",
                    0,
                )
                try:
                    candidate_descriptor = os.open(
                        "candidate",
                        flags,
                        dir_fd=slot_descriptor,
                    )
                except OSError as error:
                    raise OpenEcologyAggregateCommitError(
                        "cannot safely open aggregate cleanup tombstone candidate"
                    ) from error
                try:
                    candidate_metadata = _opened_aggregate_cleanup_file_identity(
                        candidate_descriptor,
                        expected_identity=expected_identity,
                        cleanup_device=cleanup_metadata.st_dev,
                        field="aggregate cleanup tombstone candidate",
                        allow_relocated_zero=True,
                    )
                    if candidate_metadata.st_size:
                        if not repair:
                            raise OpenEcologyAggregateCommitError(
                                "aggregate cleanup tombstone candidate is not zero-byte"
                            )
                        os.ftruncate(candidate_descriptor, 0)
                        os.fsync(candidate_descriptor)
                    final_metadata = _opened_aggregate_cleanup_file_identity(
                        candidate_descriptor,
                        expected_identity=expected_identity,
                        cleanup_device=cleanup_metadata.st_dev,
                        field="aggregate cleanup tombstone candidate",
                        allow_relocated_zero=True,
                    )
                    if final_metadata.st_size != 0:
                        raise OpenEcologyAggregateCommitError(
                            "aggregate cleanup tombstone truncation did not persist"
                        )
                    named_metadata = os.stat(
                        "candidate",
                        dir_fd=slot_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        _path_identity_from_stat(named_metadata)
                        != _path_identity_from_stat(final_metadata)
                        or named_metadata.st_size != 0
                    ):
                        raise OpenEcologyAggregateCommitError(
                            "aggregate cleanup tombstone candidate changed "
                            "during cleanup"
                        )
                finally:
                    os.close(candidate_descriptor)
            else:
                try:
                    candidate_descriptor = os.open(
                        "candidate",
                        _directory_descriptor_flags(),
                        dir_fd=slot_descriptor,
                    )
                except OSError as error:
                    raise OpenEcologyAggregateCommitError(
                        "cannot safely open aggregate stage tombstone candidate"
                    ) from error
                try:
                    candidate_metadata = _validate_opened_aggregate_stage(
                        candidate_descriptor,
                        expected_identity=expected_identity,
                        cleanup_device=cleanup_metadata.st_dev,
                        limits=limits,
                        field="aggregate stage tombstone candidate",
                        allow_relocated=True,
                    )
                    named_metadata = os.stat(
                        "candidate",
                        dir_fd=slot_descriptor,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(
                        named_metadata.st_mode
                    ) or _path_identity_from_stat(
                        named_metadata
                    ) != _path_identity_from_stat(candidate_metadata):
                        raise OpenEcologyAggregateCommitError(
                            "aggregate stage tombstone candidate changed during cleanup"
                        )
                finally:
                    os.close(candidate_descriptor)
            current_slot = os.stat(
                slot_name,
                dir_fd=cleanup_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(current_slot.st_mode) or _path_identity_from_stat(
                current_slot
            ) != _path_identity_from_stat(slot_metadata):
                raise OpenEcologyAggregateCommitError(
                    "aggregate cleanup tombstone slot namespace changed"
                )
        finally:
            os.close(slot_descriptor)
    return len(names)


def _reconcile_aggregate_cleanup_tombstones(
    root: Path,
    *,
    max_tombstones: int,
    repair: bool = True,
    limits: _Limits | None = None,
) -> int:
    _positive_int(max_tombstones, field="max_tombstones")
    root_descriptor, root_metadata = _open_owned_cleanup_parent(
        root,
        field="aggregate cleanup root",
    )
    try:
        opened = _open_aggregate_cleanup_directory(
            root_descriptor,
            root_metadata=root_metadata,
            create=False,
        )
        if opened is None:
            return 0
        cleanup_descriptor, cleanup_metadata = opened
        try:
            count = _reconcile_aggregate_cleanup_directory_descriptor(
                cleanup_descriptor,
                cleanup_metadata=cleanup_metadata,
                max_tombstones=max_tombstones,
                repair=repair,
                limits=limits,
            )
            current_cleanup = os.stat(
                OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(current_cleanup.st_mode) or _path_identity_from_stat(
                current_cleanup
            ) != _path_identity_from_stat(cleanup_metadata):
                raise OpenEcologyAggregateCommitError(
                    "aggregate cleanup tombstone namespace changed"
                )
            current_root = os.lstat(root)
            if not stat.S_ISDIR(current_root.st_mode) or _path_identity_from_stat(
                current_root
            ) != _path_identity_from_stat(root_metadata):
                raise OpenEcologyAggregateCommitError(
                    "aggregate cleanup root namespace changed"
                )
            return count
        finally:
            os.close(cleanup_descriptor)
    finally:
        os.close(root_descriptor)


def _quarantine_owned_path(
    path: Path,
    *,
    expected_identity: _PathIdentity,
    field: str,
    kind: str,
    cleanup_root: Path,
    max_tombstones: int,
    missing_ok: bool,
    limits: _Limits | None = None,
) -> bool:
    _positive_int(max_tombstones, field="max_tombstones")
    if cleanup_root.name != OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME:
        raise OpenEcologyAggregateCommitError(
            "aggregate cleanup root name is not authoritative"
        )
    aggregate_root = cleanup_root.parent
    source_descriptor, source_metadata = _open_owned_cleanup_parent(
        path.parent,
        field=f"{field} parent",
    )
    try:
        try:
            observed = os.stat(
                path.name,
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if kind == "file":
            source_valid = (
                stat.S_ISREG(observed.st_mode)
                and observed.st_nlink == 1
                and observed.st_uid == os.geteuid()
            )
        elif kind == "stage":
            source_valid = (
                stat.S_ISDIR(observed.st_mode)
                and observed.st_uid == os.geteuid()
                and not stat.S_IMODE(observed.st_mode) & 0o022
            )
        else:
            raise OpenEcologyAggregateCommitError(
                "aggregate cleanup tombstone kind is invalid"
            )
        if not source_valid or _path_identity_from_stat(observed) != expected_identity:
            raise OpenEcologyAggregateCommitError(
                f"{field} changed before cleanup; replacement preserved"
            )
        root_descriptor, root_metadata = _open_owned_cleanup_parent(
            aggregate_root,
            field="aggregate cleanup root",
        )
        try:
            if source_metadata.st_dev != root_metadata.st_dev:
                raise OpenEcologyAggregateCommitError(
                    f"{field} cleanup requires a same-filesystem quarantine"
                )
            opened = _open_aggregate_cleanup_directory(
                root_descriptor,
                root_metadata=root_metadata,
                create=True,
            )
            assert opened is not None
            cleanup_descriptor, cleanup_metadata = opened
            try:
                count = _reconcile_aggregate_cleanup_directory_descriptor(
                    cleanup_descriptor,
                    cleanup_metadata=cleanup_metadata,
                    max_tombstones=max_tombstones,
                    repair=True,
                    limits=limits,
                )
                if count >= max_tombstones:
                    raise OpenEcologyAggregateCommitError(
                        "aggregate cleanup tombstone limit reached"
                    )
                slot_name, slot_descriptor, slot_metadata = (
                    _open_aggregate_cleanup_slot(
                        cleanup_descriptor,
                        cleanup_metadata=cleanup_metadata,
                        kind=kind,
                        expected_identity=expected_identity,
                    )
                )
                try:
                    if os.listdir(slot_descriptor):
                        raise OpenEcologyAggregateCommitError(
                            "new aggregate cleanup tombstone slot is not empty"
                        )
                    current = os.stat(
                        path.name,
                        dir_fd=source_descriptor,
                        follow_symlinks=False,
                    )
                    if kind == "file":
                        current_valid = (
                            stat.S_ISREG(current.st_mode)
                            and current.st_nlink == 1
                            and current.st_uid == os.geteuid()
                        )
                    else:
                        current_valid = (
                            stat.S_ISDIR(current.st_mode)
                            and current.st_uid == os.geteuid()
                            and not stat.S_IMODE(current.st_mode) & 0o022
                        )
                    if (
                        not current_valid
                        or _path_identity_from_stat(current) != expected_identity
                    ):
                        raise OpenEcologyAggregateCommitError(
                            f"{field} changed before cleanup; replacement preserved"
                        )
                    try:
                        os.rename(
                            path.name,
                            "candidate",
                            src_dir_fd=source_descriptor,
                            dst_dir_fd=slot_descriptor,
                        )
                    except FileNotFoundError:
                        if missing_ok:
                            return False
                        raise
                    if kind == "file":
                        flags = (
                            os.O_RDWR
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0)
                        )
                        candidate_descriptor = os.open(
                            "candidate",
                            flags,
                            dir_fd=slot_descriptor,
                        )
                        try:
                            _opened_aggregate_cleanup_file_identity(
                                candidate_descriptor,
                                expected_identity=expected_identity,
                                cleanup_device=cleanup_metadata.st_dev,
                                field=f"{field} quarantine candidate",
                            )
                            os.ftruncate(candidate_descriptor, 0)
                            os.fsync(candidate_descriptor)
                            final_metadata = _opened_aggregate_cleanup_file_identity(
                                candidate_descriptor,
                                expected_identity=expected_identity,
                                cleanup_device=cleanup_metadata.st_dev,
                                field=f"{field} quarantine candidate",
                            )
                            if final_metadata.st_size != 0:
                                raise OpenEcologyAggregateCommitError(
                                    f"{field} descriptor-bound truncation failed"
                                )
                            named_metadata = os.stat(
                                "candidate",
                                dir_fd=slot_descriptor,
                                follow_symlinks=False,
                            )
                            if (
                                _path_identity_from_stat(named_metadata)
                                != expected_identity
                                or named_metadata.st_size != 0
                            ):
                                raise OpenEcologyAggregateCommitError(
                                    f"{field} changed during cleanup; "
                                    "replacement preserved"
                                )
                        finally:
                            os.close(candidate_descriptor)
                    else:
                        candidate_descriptor = os.open(
                            "candidate",
                            _directory_descriptor_flags(),
                            dir_fd=slot_descriptor,
                        )
                        try:
                            candidate_metadata = _validate_opened_aggregate_stage(
                                candidate_descriptor,
                                expected_identity=expected_identity,
                                cleanup_device=cleanup_metadata.st_dev,
                                limits=limits,
                                field=f"{field} quarantine candidate",
                            )
                            named_metadata = os.stat(
                                "candidate",
                                dir_fd=slot_descriptor,
                                follow_symlinks=False,
                            )
                            if not stat.S_ISDIR(
                                named_metadata.st_mode
                            ) or _path_identity_from_stat(
                                named_metadata
                            ) != _path_identity_from_stat(candidate_metadata):
                                raise OpenEcologyAggregateCommitError(
                                    f"{field} changed during cleanup; "
                                    "replacement preserved"
                                )
                        finally:
                            os.close(candidate_descriptor)
                    current_slot = os.stat(
                        slot_name,
                        dir_fd=cleanup_descriptor,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(
                        current_slot.st_mode
                    ) or _path_identity_from_stat(
                        current_slot
                    ) != _path_identity_from_stat(slot_metadata):
                        raise OpenEcologyAggregateCommitError(
                            f"{field} cleanup slot namespace changed"
                        )
                    current_cleanup = os.stat(
                        OPEN_ECOLOGY_AGGREGATE_CLEANUP_DIRECTORY_NAME,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(
                        current_cleanup.st_mode
                    ) or _path_identity_from_stat(
                        current_cleanup
                    ) != _path_identity_from_stat(cleanup_metadata):
                        raise OpenEcologyAggregateCommitError(
                            f"{field} cleanup namespace changed"
                        )
                    current_root = os.lstat(aggregate_root)
                    if not stat.S_ISDIR(
                        current_root.st_mode
                    ) or _path_identity_from_stat(
                        current_root
                    ) != _path_identity_from_stat(root_metadata):
                        raise OpenEcologyAggregateCommitError(
                            f"{field} cleanup root namespace changed"
                        )
                    current_source_parent = os.lstat(path.parent)
                    if not stat.S_ISDIR(
                        current_source_parent.st_mode
                    ) or _path_identity_from_stat(
                        current_source_parent
                    ) != _path_identity_from_stat(source_metadata):
                        raise OpenEcologyAggregateCommitError(
                            f"{field} parent namespace changed"
                        )
                    _fsync_directory_descriptor_best_effort(slot_descriptor)
                    _fsync_directory_descriptor_best_effort(cleanup_descriptor)
                    _fsync_directory_descriptor_best_effort(root_descriptor)
                    _fsync_directory_descriptor_best_effort(source_descriptor)
                    return True
                finally:
                    os.close(slot_descriptor)
            finally:
                os.close(cleanup_descriptor)
        finally:
            os.close(root_descriptor)
    finally:
        os.close(source_descriptor)


def _remove_owned_stage(
    stage: Path,
    *,
    parent: Path,
    prefix: str,
    expected_identity: _PathIdentity,
    cleanup_root: Path,
    max_tombstones: int,
    limits: _Limits | None = None,
) -> None:
    if stage.parent != parent or not stage.name.startswith(prefix):
        raise OpenEcologyAggregateCommitError(
            "refusing to remove an unrecognized aggregate staging directory"
        )
    _quarantine_owned_path(
        stage,
        expected_identity=expected_identity,
        field="aggregate staging directory",
        kind="stage",
        cleanup_root=cleanup_root,
        max_tombstones=max_tombstones,
        missing_ok=True,
        limits=limits,
    )


def _regular_file_identity(path: Path, *, field: str) -> _PathIdentity:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise OpenEcologyAggregateCommitError(f"cannot inspect {field}") from error
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
        raise OpenEcologyAggregateCommitError(
            f"{field} must be a single-link regular file"
        )
    return _path_identity_from_stat(path_stat)


def _unlink_owned_regular_file(
    path: Path,
    *,
    expected_identity: _PathIdentity,
    field: str,
    missing_ok: bool = False,
    cleanup_root: Path,
    max_tombstones: int,
) -> bool:
    return _quarantine_owned_path(
        path,
        expected_identity=expected_identity,
        field=field,
        kind="file",
        cleanup_root=cleanup_root,
        max_tombstones=max_tombstones,
        missing_ok=missing_ok,
    )


def _require_directory(path: Path, *, field: str) -> Path:
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise OpenEcologyAggregateCommitError(
            f"cannot inspect {field}: {error}"
        ) from error
    if not stat.S_ISDIR(path_stat.st_mode):
        raise OpenEcologyAggregateCommitError(
            f"{field} must be a real directory, not a link or other file"
        )
    return path


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologyAggregateCommitError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise OpenEcologyAggregateCommitError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _canonical_ascii_line(value: object) -> bytes:
    return _canonical_ascii_bytes(value) + b"\n"


def _canonical_ascii_bytes(value: object) -> bytes:
    _validate_json_value(value, field="value", depth=0)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise OpenEcologyAggregateCommitError(
            f"value is not canonical JSON: {error}"
        ) from error


def _canonical_utf8_line(value: object) -> bytes:
    _validate_json_value(value, field="value", depth=0)
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise OpenEcologyAggregateCommitError(
            f"value is not canonical UTF-8 JSON: {error}"
        ) from error


def _canonical_clone(value: object, *, field: str) -> dict[str, object]:
    parsed = _parse_json(_canonical_ascii_bytes(value), field=field)
    if not isinstance(parsed, dict):
        raise OpenEcologyAggregateCommitError(f"{field} must be an object")
    return parsed


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_ascii_bytes(value)).hexdigest()


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_ascii_bytes(left) == _canonical_ascii_bytes(right)


def _validate_json_value(value: object, *, field: str, depth: int) -> None:
    if depth > _MAX_JSON_NESTING_DEPTH:
        raise OpenEcologyAggregateCommitError(
            f"{field} exceeds maximum JSON nesting depth"
        )
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise OpenEcologyAggregateCommitError(
                f"{field} contains a non-finite float"
            )
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                field=f"{field}[{index}]",
                depth=depth + 1,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise OpenEcologyAggregateCommitError(
                    f"{field} contains a non-string key"
                )
            _validate_json_value(
                item,
                field=f"{field}.{key}",
                depth=depth + 1,
            )
        return
    raise OpenEcologyAggregateCommitError(
        f"{field} contains unsupported JSON type {type(value).__name__}"
    )


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OpenEcologyAggregateCommitError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise OpenEcologyAggregateCommitError(
            f"{field} keys mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _git_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise OpenEcologyAggregateCommitError(
            f"{field} must be an exact lowercase 40-hex commit SHA"
        )
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyAggregateCommitError(f"{field} must be lowercase SHA-256 hex")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise OpenEcologyAggregateCommitError(f"{field} must be a safe identifier")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyAggregateCommitError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyAggregateCommitError(f"{field} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field=field)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor_best_effort(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError:
        pass
