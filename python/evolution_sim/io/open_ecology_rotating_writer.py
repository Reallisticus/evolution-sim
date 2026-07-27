from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Mapping, Sequence, TypeAlias


OPEN_ECOLOGY_EVIDENCE_FORMAT = "evolution_sim_open_ecology_evidence_shard_v1"
OPEN_ECOLOGY_EVIDENCE_MANIFEST_SCHEMA = (
    "evolution_sim_open_ecology_evidence_manifest_v1"
)
OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA = (
    "evolution_sim_open_ecology_evidence_continuation_v1"
)
OPEN_ECOLOGY_EVIDENCE_EVENT_SCHEMA = "evolution_sim_open_ecology_event_v1"
OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME = "manifest.json"
OPEN_ECOLOGY_EVIDENCE_LOCK_NAME = ".writer.lock"
OPEN_ECOLOGY_EVIDENCE_TRANSACTION_NAME = ".pending-transaction.json"
OPEN_ECOLOGY_EVIDENCE_TRANSACTION_SCHEMA = (
    "evolution_sim_open_ecology_evidence_transaction_v1"
)
OPEN_ECOLOGY_EVIDENCE_CHAIN_GENESIS = "0" * 64
_OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES = b"evolution_sim_open_ecology_writer_lock_v1\n"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_NESTING_DEPTH = 128
_DEFAULT_SAFE_MANIFEST_READ_BYTES = 16 * 1024 * 1024
_DEFAULT_SAFE_TRANSACTION_READ_BYTES = 40 * 1024 * 1024
_ABSOLUTE_SAFE_SHARD_READ_BYTES = 512 * 1024 * 1024


class OpenEcologyEvidenceError(ValueError):
    """Raised when rotating open-ecology evidence fails closed."""


@dataclass(frozen=True, slots=True)
class RotatingEvidenceConfig:
    """Hard resource ceilings for one opt-in evidence stream."""

    gzip_compresslevel: int = 9
    max_ticks_per_shard: int = 512
    max_rows_per_shard: int = 100_000
    max_uncompressed_bytes_per_shard: int = 64 * 1024 * 1024
    max_compressed_bytes_per_shard: int = 72 * 1024 * 1024
    max_event_bytes: int = 256 * 1024
    max_shards: int = 4_096
    max_manifest_bytes: int = 8 * 1024 * 1024
    max_source_contract_bytes: int = 1024 * 1024
    max_total_compressed_bytes: int = 8 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            isinstance(self.gzip_compresslevel, bool)
            or not isinstance(self.gzip_compresslevel, int)
            or not 0 <= self.gzip_compresslevel <= 9
        ):
            raise OpenEcologyEvidenceError(
                "gzip_compresslevel must be an integer from 0 through 9"
            )
        for field_name in (
            "max_ticks_per_shard",
            "max_rows_per_shard",
            "max_uncompressed_bytes_per_shard",
            "max_compressed_bytes_per_shard",
            "max_event_bytes",
            "max_shards",
            "max_manifest_bytes",
            "max_source_contract_bytes",
            "max_total_compressed_bytes",
        ):
            _positive_int(getattr(self, field_name), field=field_name)
        if self.max_event_bytes >= self.max_uncompressed_bytes_per_shard:
            raise OpenEcologyEvidenceError(
                "max_event_bytes must be smaller than max_uncompressed_bytes_per_shard"
            )
        if self.max_manifest_bytes > _DEFAULT_SAFE_MANIFEST_READ_BYTES:
            raise OpenEcologyEvidenceError(
                "max_manifest_bytes exceeds the absolute safe manifest read ceiling"
            )
        if self.max_source_contract_bytes >= self.max_manifest_bytes:
            raise OpenEcologyEvidenceError(
                "max_source_contract_bytes must be smaller than max_manifest_bytes"
            )
        if (
            self.max_uncompressed_bytes_per_shard > _ABSOLUTE_SAFE_SHARD_READ_BYTES
            or self.max_compressed_bytes_per_shard > _ABSOLUTE_SAFE_SHARD_READ_BYTES
        ):
            raise OpenEcologyEvidenceError(
                "per-shard byte ceilings exceed the absolute safe shard read ceiling"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "gzip_compresslevel": self.gzip_compresslevel,
            "max_compressed_bytes_per_shard": (self.max_compressed_bytes_per_shard),
            "max_event_bytes": self.max_event_bytes,
            "max_manifest_bytes": self.max_manifest_bytes,
            "max_rows_per_shard": self.max_rows_per_shard,
            "max_shards": self.max_shards,
            "max_source_contract_bytes": self.max_source_contract_bytes,
            "max_ticks_per_shard": self.max_ticks_per_shard,
            "max_total_compressed_bytes": self.max_total_compressed_bytes,
            "max_uncompressed_bytes_per_shard": (self.max_uncompressed_bytes_per_shard),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RotatingEvidenceConfig:
        expected = frozenset(cls().to_dict())
        _exact_keys(payload, expected, field="writer_config")
        return cls(
            gzip_compresslevel=_bounded_int(
                payload["gzip_compresslevel"],
                field="writer_config.gzip_compresslevel",
                minimum=0,
                maximum=9,
            ),
            max_ticks_per_shard=_positive_int(
                payload["max_ticks_per_shard"],
                field="writer_config.max_ticks_per_shard",
            ),
            max_rows_per_shard=_positive_int(
                payload["max_rows_per_shard"],
                field="writer_config.max_rows_per_shard",
            ),
            max_uncompressed_bytes_per_shard=_positive_int(
                payload["max_uncompressed_bytes_per_shard"],
                field="writer_config.max_uncompressed_bytes_per_shard",
            ),
            max_compressed_bytes_per_shard=_positive_int(
                payload["max_compressed_bytes_per_shard"],
                field="writer_config.max_compressed_bytes_per_shard",
            ),
            max_event_bytes=_positive_int(
                payload["max_event_bytes"],
                field="writer_config.max_event_bytes",
            ),
            max_shards=_positive_int(
                payload["max_shards"],
                field="writer_config.max_shards",
            ),
            max_manifest_bytes=_positive_int(
                payload["max_manifest_bytes"],
                field="writer_config.max_manifest_bytes",
            ),
            max_source_contract_bytes=_positive_int(
                payload["max_source_contract_bytes"],
                field="writer_config.max_source_contract_bytes",
            ),
            max_total_compressed_bytes=_positive_int(
                payload["max_total_compressed_bytes"],
                field="writer_config.max_total_compressed_bytes",
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BirthEvidence:
    tick: int
    event_index: int
    event_id: str
    child_agent_id: int
    parent_agent_ids: tuple[int, ...]
    lineage_id: str
    genome_sha256: str | None = None

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _nonnegative_int(self.child_agent_id, field="child_agent_id")
        _agent_id_tuple(
            self.parent_agent_ids,
            field="parent_agent_ids",
            allow_empty=False,
        )
        if self.child_agent_id in self.parent_agent_ids:
            raise OpenEcologyEvidenceError(
                "child_agent_id cannot also be a parent_agent_id"
            )
        _identifier(self.lineage_id, field="lineage_id")
        _optional_sha256(self.genome_sha256, field="genome_sha256")

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="birth",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "child_agent_id": self.child_agent_id,
                "genome_sha256": self.genome_sha256,
                "lineage_id": self.lineage_id,
                "parent_agent_ids": list(self.parent_agent_ids),
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DeathEvidence:
    tick: int
    event_index: int
    event_id: str
    agent_id: int
    cause: str
    source_agent_id: int | None = None

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _nonnegative_int(self.agent_id, field="agent_id")
        _identifier(self.cause, field="cause")
        _optional_nonnegative_int(self.source_agent_id, field="source_agent_id")
        if self.source_agent_id == self.agent_id:
            raise OpenEcologyEvidenceError(
                "source_agent_id must differ from the dead agent_id"
            )

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="death",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "agent_id": self.agent_id,
                "cause": self.cause,
                "source_agent_id": self.source_agent_id,
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LineageEvidence:
    tick: int
    event_index: int
    event_id: str
    agent_id: int
    lineage_id: str
    parent_lineage_ids: tuple[str, ...]
    transition_kind: str

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _nonnegative_int(self.agent_id, field="agent_id")
        _identifier(self.lineage_id, field="lineage_id")
        _identifier_tuple(
            self.parent_lineage_ids,
            field="parent_lineage_ids",
            allow_empty=True,
        )
        if self.lineage_id in self.parent_lineage_ids:
            raise OpenEcologyEvidenceError(
                "lineage_id cannot also be a parent_lineage_id"
            )
        _identifier(self.transition_kind, field="transition_kind")

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="lineage",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "agent_id": self.agent_id,
                "lineage_id": self.lineage_id,
                "parent_lineage_ids": list(self.parent_lineage_ids),
                "transition_kind": self.transition_kind,
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DyadicInteractionEvidence:
    tick: int
    event_index: int
    event_id: str
    actor_agent_id: int
    target_agent_id: int
    interaction_kind: str
    outcome: str
    magnitude: float | None = None

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _nonnegative_int(self.actor_agent_id, field="actor_agent_id")
        _nonnegative_int(self.target_agent_id, field="target_agent_id")
        if self.actor_agent_id == self.target_agent_id:
            raise OpenEcologyEvidenceError(
                "dyadic interaction actor and target must differ"
            )
        _identifier(self.interaction_kind, field="interaction_kind")
        _identifier(self.outcome, field="outcome")
        _optional_finite_number(self.magnitude, field="magnitude")

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="dyadic_interaction",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "actor_agent_id": self.actor_agent_id,
                "interaction_kind": self.interaction_kind,
                "magnitude": self.magnitude,
                "outcome": self.outcome,
                "target_agent_id": self.target_agent_id,
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalContributorEvidence:
    tick: int
    event_index: int
    event_id: str
    emitter_agent_id: int
    receiver_agent_id: int
    token_id: int
    contribution: float

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _nonnegative_int(self.emitter_agent_id, field="emitter_agent_id")
        _nonnegative_int(self.receiver_agent_id, field="receiver_agent_id")
        if self.emitter_agent_id == self.receiver_agent_id:
            raise OpenEcologyEvidenceError(
                "signal contributor emitter and receiver must differ"
            )
        _nonnegative_int(self.token_id, field="token_id")
        value = _finite_number(self.contribution, field="contribution")
        if value < 0.0:
            raise OpenEcologyEvidenceError("contribution must be nonnegative")

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="signal_contributor",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "contribution": self.contribution,
                "emitter_agent_id": self.emitter_agent_id,
                "receiver_agent_id": self.receiver_agent_id,
                "token_id": self.token_id,
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CongestionEvidence:
    tick: int
    event_index: int
    event_id: str
    target_x: int
    target_y: int
    contender_agent_ids: tuple[int, ...]
    winner_agent_id: int | None
    outcome: str

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _nonnegative_int(self.target_x, field="target_x")
        _nonnegative_int(self.target_y, field="target_y")
        _agent_id_tuple(
            self.contender_agent_ids,
            field="contender_agent_ids",
            allow_empty=False,
        )
        _optional_nonnegative_int(self.winner_agent_id, field="winner_agent_id")
        if (
            self.winner_agent_id is not None
            and self.winner_agent_id not in self.contender_agent_ids
        ):
            raise OpenEcologyEvidenceError(
                "winner_agent_id must be one of contender_agent_ids"
            )
        _identifier(self.outcome, field="outcome")

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="congestion",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "contender_agent_ids": list(self.contender_agent_ids),
                "outcome": self.outcome,
                "target_x": self.target_x,
                "target_y": self.target_y,
                "winner_agent_id": self.winner_agent_id,
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class InterventionEvidence:
    tick: int
    event_index: int
    event_id: str
    intervention_id: str
    intervention_kind: str
    branch_id: str
    subject_agent_id: int | None = None
    assigned_action: str | None = None
    value_sha256: str | None = None

    def __post_init__(self) -> None:
        _event_identity(self.tick, self.event_index, self.event_id)
        _identifier(self.intervention_id, field="intervention_id")
        _identifier(self.intervention_kind, field="intervention_kind")
        _identifier(self.branch_id, field="branch_id")
        _optional_nonnegative_int(self.subject_agent_id, field="subject_agent_id")
        _optional_identifier(self.assigned_action, field="assigned_action")
        _optional_sha256(self.value_sha256, field="value_sha256")

    def to_record(self) -> dict[str, object]:
        return _event_record(
            event_type="intervention",
            tick=self.tick,
            event_index=self.event_index,
            event_id=self.event_id,
            payload={
                "assigned_action": self.assigned_action,
                "branch_id": self.branch_id,
                "intervention_id": self.intervention_id,
                "intervention_kind": self.intervention_kind,
                "subject_agent_id": self.subject_agent_id,
                "value_sha256": self.value_sha256,
            },
        )


OpenEcologyEvidenceEvent: TypeAlias = (
    BirthEvidence
    | DeathEvidence
    | LineageEvidence
    | DyadicInteractionEvidence
    | SignalContributorEvidence
    | CongestionEvidence
    | InterventionEvidence
)
_EVENT_TYPES = (
    BirthEvidence,
    DeathEvidence,
    LineageEvidence,
    DyadicInteractionEvidence,
    SignalContributorEvidence,
    CongestionEvidence,
    InterventionEvidence,
)


@dataclass(slots=True)
class _ActiveShard:
    index: int
    previous_shard_sha256: str
    temp_path: Path
    raw_handle: BinaryIO
    gzip_handle: gzip.GzipFile
    first_tick: int
    last_tick: int
    first_event_index: int
    last_event_index: int
    event_count: int
    uncompressed_bytes_before_footer: int
    content_hash: object
    event_stream_hash: object


class RotatingOpenEcologyEvidenceWriter:
    """Write typed ecology events to bounded, atomic, hash-chained shards.

    This writer is deliberately not installed in ``SimulationWorld``. A caller
    must opt in and supply already-observed typed events. It does not infer
    births, alliances, meanings, or causal effects from trajectory rows.
    """

    def __init__(
        self,
        output_directory: str | Path,
        *,
        run_id: str,
        source_contract: Mapping[str, object],
        config: RotatingEvidenceConfig | None = None,
        continuation_state: Mapping[str, object] | None = None,
    ):
        self.output_directory = Path(output_directory)
        self.run_id = _identifier(run_id, field="run_id")
        self.config = config or RotatingEvidenceConfig()
        self._writer_config = self.config.to_dict()
        self._writer_config_sha256 = _digest(self._writer_config)
        self._source_contract = _canonical_mapping_clone(
            source_contract,
            field="source_contract",
            max_bytes=self.config.max_source_contract_bytes,
        )
        self._source_contract_sha256 = _digest(self._source_contract)
        self._manifest_path = (
            self.output_directory / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME
        )
        self._transaction_path = (
            self.output_directory / OPEN_ECOLOGY_EVIDENCE_TRANSACTION_NAME
        )
        self._lock_path = self.output_directory / OPEN_ECOLOGY_EVIDENCE_LOCK_NAME
        self._lock_fd: int | None = None
        self._entries: list[dict[str, object]] = []
        self._chain_head_sha256 = OPEN_ECOLOGY_EVIDENCE_CHAIN_GENESIS
        self._next_shard_index = 0
        self._next_event_index = 0
        self._last_tick: int | None = None
        self._total_event_count = 0
        self._active: _ActiveShard | None = None
        self._manifest_sha256: str | None = None
        self._durable_manifest: dict[str, object] | None = None
        self._finished = False
        self._aborted = False
        self._checkpointed = False

        if self.output_directory.exists() and not self.output_directory.is_dir():
            raise OpenEcologyEvidenceError("output_directory must be a directory")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            if continuation_state is None:
                self._initialize_new_stream()
            else:
                self._initialize_resume(continuation_state)
        except Exception:
            self._release_lock()
            raise

    @property
    def diagnostics(self) -> dict[str, object]:
        status = (
            "finished"
            if self._finished
            else "aborted"
            if self._aborted
            else "checkpointed"
            if self._checkpointed
            else "open"
        )
        return {
            "active_shard_event_count": (
                0 if self._active is None else self._active.event_count
            ),
            "chain_head_sha256": self._chain_head_sha256,
            "completed_shard_count": len(self._entries),
            "in_memory_event_count": 0,
            "last_tick": self._last_tick,
            "manifest_entry_count": len(self._entries),
            "maximum_manifest_entry_count": self.config.max_shards,
            "next_event_index": self._next_event_index,
            "status": status,
            "total_compressed_bytes": sum(
                int(entry["compressed_bytes"]) for entry in self._entries
            ),
            "total_event_count": self._total_event_count,
        }

    def append(self, event: OpenEcologyEvidenceEvent) -> None:
        self._ensure_writable()
        try:
            if not isinstance(event, _EVENT_TYPES):
                raise TypeError(
                    "open ecology evidence ingestion accepts only declared "
                    "typed event classes"
                )
            record = event.to_record()
            validate_open_ecology_event_record(record)
            tick = event.tick
            event_index = event.event_index
            if event_index != self._next_event_index:
                raise OpenEcologyEvidenceError(
                    "event_index must be contiguous and ordered "
                    f"(expected {self._next_event_index}, got {event_index})"
                )
            if self._last_tick is not None and tick < self._last_tick:
                raise OpenEcologyEvidenceError(
                    "event ticks must be monotonically nondecreasing"
                )

            record_bytes = _canonical_bytes(record)
            line_bytes = _canonical_line_bytes({"event": record, "type": "event"})
            if len(line_bytes) > self.config.max_event_bytes:
                raise OpenEcologyEvidenceError(
                    "canonical event row exceeds max_event_bytes "
                    f"({len(line_bytes)} > {self.config.max_event_bytes})"
                )

            if self._active is None:
                self._open_shard(tick=tick, event_index=event_index)
            elif not self._can_append(
                tick=tick,
                event_index=event_index,
                record_bytes=record_bytes,
                line_bytes=line_bytes,
            ):
                self._seal_active_shard()
                self._open_shard(tick=tick, event_index=event_index)

            if not self._can_append(
                tick=tick,
                event_index=event_index,
                record_bytes=record_bytes,
                line_bytes=line_bytes,
            ):
                raise OpenEcologyEvidenceError(
                    "one canonical event cannot fit within the configured shard "
                    "row/tick/byte ceilings"
                )
            self._write_event(
                tick=tick,
                event_index=event_index,
                record_bytes=record_bytes,
                line_bytes=line_bytes,
            )
            if (
                self._active is not None
                and self._active.event_count >= self.config.max_rows_per_shard
            ):
                self._seal_active_shard()
        except Exception:
            self.abort()
            raise

    def checkpoint(self) -> dict[str, object]:
        """Seal the current shard and return a JSON checkpoint component.

        The instance becomes read-only. Resume by constructing a new writer
        with this state and the same run/source/config. This intentional shard
        boundary is the only portable continuation point.
        """

        self._ensure_writable()
        try:
            self._seal_active_shard()
            self._write_manifest(status="open")
            state = self._build_continuation_state()
            self._checkpointed = True
            self._release_lock()
            return state
        except Exception:
            self.abort()
            raise

    def finish(self) -> dict[str, object]:
        if self._finished:
            return load_open_ecology_evidence_manifest(
                self.output_directory,
                verify_shards=True,
            )
        self._ensure_writable()
        try:
            self._seal_active_shard()
            manifest = self._write_manifest(status="complete")
            self._finished = True
            self._release_lock()
            return manifest
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        active = self._active
        self._active = None
        if active is not None:
            try:
                active.gzip_handle.close()
            except OSError:
                pass
            try:
                active.raw_handle.close()
            except OSError:
                pass
            try:
                active.temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if not self._finished and not self._checkpointed:
            self._aborted = True
        self._release_lock()

    def __enter__(self) -> RotatingOpenEcologyEvidenceWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or not self._finished:
            self.abort()

    def _initialize_new_stream(self) -> None:
        unexpected_entries = {
            path.name
            for path in self.output_directory.iterdir()
            if path.name != OPEN_ECOLOGY_EVIDENCE_LOCK_NAME
        }
        if unexpected_entries:
            raise OpenEcologyEvidenceError(
                "new evidence output_directory must be empty"
            )
        self._write_manifest(status="open")

    def _initialize_resume(self, state: Mapping[str, object]) -> None:
        normalized_state = validate_open_ecology_evidence_continuation_state(state)
        self._recover_pending_transaction(normalized_state)
        manifest = load_open_ecology_evidence_manifest(
            self.output_directory,
            verify_shards=True,
            expected_manifest_sha256=str(normalized_state["manifest_sha256"]),
        )
        if manifest["status"] != "open":
            raise OpenEcologyEvidenceError(
                "only an open evidence manifest can be resumed"
            )
        comparisons = {
            "run_id": self.run_id,
            "writer_config_sha256": self._writer_config_sha256,
            "source_contract_sha256": self._source_contract_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "chain_head_sha256": manifest["chain_head_sha256"],
            "completed_shard_count": manifest["completed_shard_count"],
            "next_shard_index": manifest["next_shard_index"],
            "next_event_index": manifest["next_event_index"],
            "last_tick": manifest["last_tick"],
            "total_compressed_bytes": manifest["total_compressed_bytes"],
            "total_event_count": manifest["total_event_count"],
        }
        for field_name, expected in comparisons.items():
            if normalized_state[field_name] != expected:
                raise OpenEcologyEvidenceError(
                    f"continuation {field_name} does not match the manifest"
                )
        if manifest["writer_config"] != self._writer_config:
            raise OpenEcologyEvidenceError(
                "resume writer config does not match the manifest"
            )
        if manifest["source_contract"] != self._source_contract:
            raise OpenEcologyEvidenceError(
                "resume source contract does not match the manifest"
            )

        self._entries = [
            dict(entry) for entry in _mapping_sequence(manifest["completed_shards"])
        ]
        self._chain_head_sha256 = str(manifest["chain_head_sha256"])
        self._next_shard_index = int(manifest["next_shard_index"])
        self._next_event_index = int(manifest["next_event_index"])
        self._last_tick = _optional_int(manifest["last_tick"], field="last_tick")
        self._total_event_count = int(manifest["total_event_count"])
        self._manifest_sha256 = str(manifest["manifest_sha256"])
        self._durable_manifest = _canonical_mapping_clone(
            manifest,
            field="manifest",
            max_bytes=self.config.max_manifest_bytes,
        )

    def _open_shard(self, *, tick: int, event_index: int) -> None:
        if len(self._entries) >= self.config.max_shards:
            raise OpenEcologyEvidenceError("evidence stream exceeds max_shards")
        if self._next_shard_index != len(self._entries):
            raise OpenEcologyEvidenceError(
                "next shard index does not match completed manifest entries"
            )
        with NamedTemporaryFile(
            "wb",
            dir=self.output_directory,
            prefix=f".shard-{self._next_shard_index:08d}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        raw_handle = temp_path.open("wb")
        gzip_handle = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
            compresslevel=self.config.gzip_compresslevel,
        )
        content_hash = hashlib.sha256()
        event_stream_hash = hashlib.sha256()
        header = {
            "format": OPEN_ECOLOGY_EVIDENCE_FORMAT,
            "previous_shard_sha256": self._chain_head_sha256,
            "run_id": self.run_id,
            "shard_index": self._next_shard_index,
            "source_contract_sha256": self._source_contract_sha256,
            "type": "header",
            "writer_config_sha256": self._writer_config_sha256,
        }
        header_line = _canonical_line_bytes(header)
        gzip_handle.write(header_line)
        content_hash.update(header_line)
        self._active = _ActiveShard(
            index=self._next_shard_index,
            previous_shard_sha256=self._chain_head_sha256,
            temp_path=temp_path,
            raw_handle=raw_handle,
            gzip_handle=gzip_handle,
            first_tick=tick,
            last_tick=tick,
            first_event_index=event_index,
            last_event_index=event_index - 1,
            event_count=0,
            uncompressed_bytes_before_footer=len(header_line),
            content_hash=content_hash,
            event_stream_hash=event_stream_hash,
        )

    def _can_append(
        self,
        *,
        tick: int,
        event_index: int,
        record_bytes: bytes,
        line_bytes: bytes,
    ) -> bool:
        active = self._active
        if active is None:
            return False
        prospective_count = active.event_count + 1
        if prospective_count > self.config.max_rows_per_shard:
            return False
        if tick - active.first_tick + 1 > self.config.max_ticks_per_shard:
            return False

        content_hash = active.content_hash.copy()
        content_hash.update(line_bytes)
        event_stream_hash = active.event_stream_hash.copy()
        event_stream_hash.update(len(record_bytes).to_bytes(8, "big"))
        event_stream_hash.update(record_bytes)
        bytes_before_footer = active.uncompressed_bytes_before_footer + len(line_bytes)
        footer_line = _canonical_line_bytes(
            self._footer_payload(
                active=active,
                event_count=prospective_count,
                last_tick=tick,
                last_event_index=event_index,
                bytes_before_footer=bytes_before_footer,
                content_sha256=content_hash.hexdigest(),
                event_stream_sha256=event_stream_hash.hexdigest(),
            )
        )
        return (
            bytes_before_footer + len(footer_line)
            <= self.config.max_uncompressed_bytes_per_shard
        )

    def _write_event(
        self,
        *,
        tick: int,
        event_index: int,
        record_bytes: bytes,
        line_bytes: bytes,
    ) -> None:
        active = self._active
        if active is None:
            raise RuntimeError("active evidence shard is missing")
        active.gzip_handle.write(line_bytes)
        active.content_hash.update(line_bytes)
        active.event_stream_hash.update(len(record_bytes).to_bytes(8, "big"))
        active.event_stream_hash.update(record_bytes)
        active.last_tick = tick
        active.last_event_index = event_index
        active.event_count += 1
        active.uncompressed_bytes_before_footer += len(line_bytes)
        self._last_tick = tick
        self._next_event_index += 1
        self._total_event_count += 1

    def _seal_active_shard(self) -> None:
        active = self._active
        if active is None:
            return
        if active.event_count <= 0:
            raise OpenEcologyEvidenceError("cannot publish an empty evidence shard")
        footer_line = _canonical_line_bytes(
            self._footer_payload(
                active=active,
                event_count=active.event_count,
                last_tick=active.last_tick,
                last_event_index=active.last_event_index,
                bytes_before_footer=active.uncompressed_bytes_before_footer,
                content_sha256=active.content_hash.hexdigest(),
                event_stream_sha256=active.event_stream_hash.hexdigest(),
            )
        )
        uncompressed_bytes = active.uncompressed_bytes_before_footer + len(footer_line)
        if uncompressed_bytes > self.config.max_uncompressed_bytes_per_shard:
            raise OpenEcologyEvidenceError(
                "completed shard exceeds max_uncompressed_bytes_per_shard"
            )
        active.gzip_handle.write(footer_line)
        active.gzip_handle.close()
        active.raw_handle.flush()
        os.fsync(active.raw_handle.fileno())
        active.raw_handle.close()

        compressed_bytes = active.temp_path.stat().st_size
        if compressed_bytes > self.config.max_compressed_bytes_per_shard:
            active.temp_path.unlink(missing_ok=True)
            self._active = None
            raise OpenEcologyEvidenceError(
                "completed shard exceeds max_compressed_bytes_per_shard "
                f"({compressed_bytes} > "
                f"{self.config.max_compressed_bytes_per_shard})"
            )
        prospective_total_compressed_bytes = compressed_bytes + sum(
            int(entry["compressed_bytes"]) for entry in self._entries
        )
        if prospective_total_compressed_bytes > self.config.max_total_compressed_bytes:
            active.temp_path.unlink(missing_ok=True)
            self._active = None
            raise OpenEcologyEvidenceError(
                "evidence stream exceeds max_total_compressed_bytes "
                f"({prospective_total_compressed_bytes} > "
                f"{self.config.max_total_compressed_bytes})"
            )
        file_sha256 = _sha256_file(
            active.temp_path,
            max_bytes=self.config.max_compressed_bytes_per_shard,
        )
        file_name = f"shard-{active.index:08d}.jsonl.gz"
        destination = self.output_directory / file_name
        if destination.exists():
            raise OpenEcologyEvidenceError(
                f"refusing to overwrite completed shard {file_name}"
            )
        entry: dict[str, object] = {
            "compressed_bytes": compressed_bytes,
            "event_count": active.event_count,
            "event_stream_sha256": active.event_stream_hash.hexdigest(),
            "file_name": file_name,
            "file_sha256": file_sha256,
            "first_event_index": active.first_event_index,
            "first_tick": active.first_tick,
            "last_event_index": active.last_event_index,
            "last_tick": active.last_tick,
            "pre_footer_content_sha256": active.content_hash.hexdigest(),
            "previous_shard_sha256": active.previous_shard_sha256,
            "shard_index": active.index,
            "uncompressed_bytes": uncompressed_bytes,
        }
        prospective_entries = [*self._entries, entry]
        prospective_manifest = self._manifest_payload(
            status="open",
            entries=prospective_entries,
            chain_head_sha256=file_sha256,
            next_shard_index=active.index + 1,
        )
        self._ensure_manifest_fits(prospective_manifest)
        if self._durable_manifest is None:
            raise OpenEcologyEvidenceError("durable evidence manifest is unavailable")
        previous_manifest = _canonical_mapping_clone(
            self._durable_manifest,
            field="durable manifest",
            max_bytes=self.config.max_manifest_bytes,
        )
        if previous_manifest["status"] != "open":
            raise OpenEcologyEvidenceError(
                "cannot append a shard to a non-open evidence manifest"
            )
        if previous_manifest["manifest_sha256"] != self._manifest_sha256:
            raise OpenEcologyEvidenceError(
                "in-memory manifest does not match the durable manifest digest"
            )
        transaction = self._transaction_payload(
            previous_manifest=previous_manifest,
            prospective_manifest=prospective_manifest,
            shard_entry=entry,
            temp_file_name=active.temp_path.name,
        )
        transaction_bytes = _canonical_line_bytes(transaction)
        if len(transaction_bytes) > _DEFAULT_SAFE_TRANSACTION_READ_BYTES:
            raise OpenEcologyEvidenceError(
                "pending evidence transaction exceeds its safe byte ceiling"
            )
        _atomic_write(self._transaction_path, transaction_bytes)
        try:
            os.replace(active.temp_path, destination)
            _fsync_directory_best_effort(self.output_directory)
            self._publish_manifest(prospective_manifest)
            self._transaction_path.unlink()
            _fsync_directory_best_effort(self.output_directory)
        except Exception:
            self._rollback_handled_transaction(
                previous_manifest=previous_manifest,
                destination=destination,
                expected_file_sha256=file_sha256,
            )
            raise

        self._active = None
        self._entries = prospective_entries
        self._chain_head_sha256 = file_sha256
        self._next_shard_index = active.index + 1

    def _footer_payload(
        self,
        *,
        active: _ActiveShard,
        event_count: int,
        last_tick: int,
        last_event_index: int,
        bytes_before_footer: int,
        content_sha256: str,
        event_stream_sha256: str,
    ) -> dict[str, object]:
        return {
            "event_count": event_count,
            "event_stream_sha256": event_stream_sha256,
            "first_event_index": active.first_event_index,
            "first_tick": active.first_tick,
            "format": OPEN_ECOLOGY_EVIDENCE_FORMAT,
            "last_event_index": last_event_index,
            "last_tick": last_tick,
            "pre_footer_content_sha256": content_sha256,
            "previous_shard_sha256": active.previous_shard_sha256,
            "shard_index": active.index,
            "type": "footer",
            "uncompressed_bytes_before_footer": bytes_before_footer,
        }

    def _build_continuation_state(self) -> dict[str, object]:
        if self._active is not None:
            raise OpenEcologyEvidenceError(
                "continuation state requires a sealed shard boundary"
            )
        if self._manifest_sha256 is None:
            raise OpenEcologyEvidenceError("manifest digest is unavailable")
        body: dict[str, object] = {
            "chain_head_sha256": self._chain_head_sha256,
            "completed_shard_count": len(self._entries),
            "last_tick": self._last_tick,
            "manifest_sha256": self._manifest_sha256,
            "next_event_index": self._next_event_index,
            "next_shard_index": self._next_shard_index,
            "run_id": self.run_id,
            "schema_version": OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA,
            "source_contract_sha256": self._source_contract_sha256,
            "total_compressed_bytes": sum(
                int(entry["compressed_bytes"]) for entry in self._entries
            ),
            "total_event_count": self._total_event_count,
            "writer_config_sha256": self._writer_config_sha256,
        }
        return {**body, "state_sha256": _digest(body)}

    def _transaction_payload(
        self,
        *,
        previous_manifest: Mapping[str, object],
        prospective_manifest: Mapping[str, object],
        shard_entry: Mapping[str, object],
        temp_file_name: str,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "previous_manifest": dict(previous_manifest),
            "prospective_manifest": dict(prospective_manifest),
            "run_id": self.run_id,
            "schema_version": OPEN_ECOLOGY_EVIDENCE_TRANSACTION_SCHEMA,
            "shard_entry": dict(shard_entry),
            "source_contract_sha256": self._source_contract_sha256,
            "temp_file_name": temp_file_name,
            "writer_config_sha256": self._writer_config_sha256,
        }
        return {**body, "transaction_sha256": _digest(body)}

    def _recover_pending_transaction(
        self,
        authoritative_state: Mapping[str, object],
    ) -> None:
        if not self._transaction_path.exists():
            self._cleanup_unpublished_temp_files()
            return
        transaction = _load_pending_transaction(self._transaction_path)
        if transaction["run_id"] != self.run_id:
            raise OpenEcologyEvidenceError(
                "pending transaction run_id does not match resume"
            )
        if transaction["source_contract_sha256"] != self._source_contract_sha256:
            raise OpenEcologyEvidenceError(
                "pending transaction source contract does not match resume"
            )
        if transaction["writer_config_sha256"] != self._writer_config_sha256:
            raise OpenEcologyEvidenceError(
                "pending transaction writer config does not match resume"
            )
        previous_manifest = _mapping(
            transaction["previous_manifest"],
            field="transaction.previous_manifest",
        )
        prospective_manifest = _mapping(
            transaction["prospective_manifest"],
            field="transaction.prospective_manifest",
        )
        previous_sha256 = _embedded_manifest_sha256(
            previous_manifest,
            field="transaction.previous_manifest",
        )
        prospective_sha256 = _embedded_manifest_sha256(
            prospective_manifest,
            field="transaction.prospective_manifest",
        )
        authoritative_sha256 = str(authoritative_state["manifest_sha256"])
        current_manifest = _parse_canonical_json_line(
            _safe_read_file(
                self._manifest_path,
                max_bytes=_DEFAULT_SAFE_MANIFEST_READ_BYTES,
            ),
            field="current manifest",
        )
        current_sha256 = _embedded_manifest_sha256(
            current_manifest,
            field="current manifest",
        )
        if current_sha256 not in {previous_sha256, prospective_sha256}:
            raise OpenEcologyEvidenceError(
                "durable manifest matches neither side of the pending transaction"
            )

        entry = _mapping(
            transaction["shard_entry"],
            field="transaction.shard_entry",
        )
        previous_entries = _mapping_sequence(previous_manifest.get("completed_shards"))
        previous_completed_count = _nonnegative_int(
            previous_manifest.get("completed_shard_count"),
            field="transaction.previous_manifest.completed_shard_count",
        )
        if previous_completed_count != len(previous_entries):
            raise OpenEcologyEvidenceError(
                "pending transaction previous manifest shard count mismatch"
            )
        previous_next_event_index = _nonnegative_int(
            previous_manifest.get("next_event_index"),
            field="transaction.previous_manifest.next_event_index",
        )
        previous_chain_head = _sha256(
            previous_manifest.get("chain_head_sha256"),
            field="transaction.previous_manifest.chain_head_sha256",
        )
        normalized_entry = _validate_manifest_entry(
            entry,
            expected_index=previous_completed_count,
            expected_previous=previous_chain_head,
            expected_first_event_index=previous_next_event_index,
            config=self.config,
        )
        if dict(entry) != normalized_entry:
            raise OpenEcologyEvidenceError(
                "pending transaction shard entry is not normalized"
            )
        for label, candidate in (
            ("previous", previous_manifest),
            ("prospective", prospective_manifest),
        ):
            if candidate.get("status") != "open":
                raise OpenEcologyEvidenceError(
                    f"pending transaction {label} manifest must be open"
                )
            if (
                candidate.get("run_id") != self.run_id
                or candidate.get("source_contract_sha256")
                != self._source_contract_sha256
                or candidate.get("writer_config_sha256") != self._writer_config_sha256
                or candidate.get("source_contract") != self._source_contract
                or candidate.get("writer_config") != self._writer_config
            ):
                raise OpenEcologyEvidenceError(
                    f"pending transaction {label} manifest contract mismatch"
                )
        if (
            prospective_manifest.get("completed_shard_count")
            != previous_completed_count + 1
            or prospective_manifest.get("next_shard_index")
            != previous_completed_count + 1
            or prospective_manifest.get("chain_head_sha256")
            != normalized_entry["file_sha256"]
            or prospective_manifest.get("next_event_index")
            != previous_next_event_index + int(normalized_entry["event_count"])
            or prospective_manifest.get("total_event_count")
            != prospective_manifest.get("next_event_index")
            or prospective_manifest.get("last_tick") != normalized_entry["last_tick"]
            or prospective_manifest.get("total_compressed_bytes")
            != _nonnegative_int(
                previous_manifest.get("total_compressed_bytes"),
                field=("transaction.previous_manifest.total_compressed_bytes"),
            )
            + int(normalized_entry["compressed_bytes"])
        ):
            raise OpenEcologyEvidenceError(
                "pending transaction prospective manifest counters mismatch"
            )
        file_name = entry.get("file_name")
        if not isinstance(file_name, str):
            raise OpenEcologyEvidenceError(
                "pending transaction shard file_name must be a string"
            )
        destination = self.output_directory / file_name
        temp_file_name = _transaction_temp_file_name(
            transaction["temp_file_name"],
            expected_shard_index=int(entry["shard_index"]),
        )
        temp_path = self.output_directory / temp_file_name
        expected_file_sha256 = _sha256(
            entry["file_sha256"],
            field="transaction.shard_entry.file_sha256",
        )

        if authoritative_sha256 == previous_sha256:
            self._publish_manifest(previous_manifest)
            self._unlink_transaction_shard_if_owned(
                destination,
                expected_file_sha256=expected_file_sha256,
            )
            temp_path.unlink(missing_ok=True)
        elif authoritative_sha256 == prospective_sha256:
            if destination.exists():
                _require_file_sha256(
                    destination,
                    expected_sha256=expected_file_sha256,
                    max_bytes=self.config.max_compressed_bytes_per_shard,
                )
            else:
                _require_file_sha256(
                    temp_path,
                    expected_sha256=expected_file_sha256,
                    max_bytes=self.config.max_compressed_bytes_per_shard,
                )
                os.replace(temp_path, destination)
                _fsync_directory_best_effort(self.output_directory)
            self._publish_manifest(prospective_manifest)
            temp_path.unlink(missing_ok=True)
        else:
            raise OpenEcologyEvidenceError(
                "continuation manifest digest does not authorize either side "
                "of the pending transaction"
            )
        self._transaction_path.unlink()
        _fsync_directory_best_effort(self.output_directory)
        self._cleanup_unpublished_temp_files()

    def _rollback_handled_transaction(
        self,
        *,
        previous_manifest: Mapping[str, object],
        destination: Path,
        expected_file_sha256: str,
    ) -> None:
        try:
            self._publish_manifest(previous_manifest)
            self._unlink_transaction_shard_if_owned(
                destination,
                expected_file_sha256=expected_file_sha256,
            )
            self._transaction_path.unlink(missing_ok=True)
            _fsync_directory_best_effort(self.output_directory)
        except Exception as error:
            raise OpenEcologyEvidenceError(
                "evidence transaction failed and could not be rolled back"
            ) from error

    def _unlink_transaction_shard_if_owned(
        self,
        path: Path,
        *,
        expected_file_sha256: str,
    ) -> None:
        if not path.exists():
            return
        _require_file_sha256(
            path,
            expected_sha256=expected_file_sha256,
            max_bytes=self.config.max_compressed_bytes_per_shard,
        )
        path.unlink()

    def _cleanup_unpublished_temp_files(self) -> None:
        for path in self.output_directory.iterdir():
            name = path.name
            is_atomic_temp = (
                name.startswith(f".{OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME}.")
                or name.startswith(f".{OPEN_ECOLOGY_EVIDENCE_TRANSACTION_NAME}.")
            ) and name.endswith(".tmp")
            is_shard_temp = (
                name.startswith(".shard-")
                and name.endswith(".tmp")
                and ".jsonl.gz" not in name
            )
            if is_atomic_temp or is_shard_temp:
                if path.is_symlink() or not path.is_file():
                    raise OpenEcologyEvidenceError(
                        "unpublished evidence temp path is not a regular file"
                    )
                path.unlink()
        _fsync_directory_best_effort(self.output_directory)

    def _manifest_payload(
        self,
        *,
        status: str,
        entries: Sequence[Mapping[str, object]] | None = None,
        chain_head_sha256: str | None = None,
        next_shard_index: int | None = None,
    ) -> dict[str, object]:
        selected_entries = list(entries if entries is not None else self._entries)
        selected_chain_head = (
            self._chain_head_sha256 if chain_head_sha256 is None else chain_head_sha256
        )
        selected_next_shard_index = (
            self._next_shard_index if next_shard_index is None else next_shard_index
        )
        body: dict[str, object] = {
            "chain_head_sha256": selected_chain_head,
            "completed_shard_count": len(selected_entries),
            "completed_shards": [dict(entry) for entry in selected_entries],
            "format": OPEN_ECOLOGY_EVIDENCE_FORMAT,
            "last_tick": self._last_tick,
            "next_event_index": self._next_event_index,
            "next_shard_index": selected_next_shard_index,
            "run_id": self.run_id,
            "schema_version": OPEN_ECOLOGY_EVIDENCE_MANIFEST_SCHEMA,
            "source_contract": self._source_contract,
            "source_contract_sha256": self._source_contract_sha256,
            "status": status,
            "total_compressed_bytes": sum(
                int(entry["compressed_bytes"]) for entry in selected_entries
            ),
            "total_event_count": self._total_event_count,
            "writer_config": self._writer_config,
            "writer_config_sha256": self._writer_config_sha256,
        }
        return {**body, "manifest_sha256": _digest(body)}

    def _write_manifest(self, *, status: str) -> dict[str, object]:
        manifest = self._manifest_payload(status=status)
        self._publish_manifest(manifest)
        return manifest

    def _publish_manifest(self, manifest: Mapping[str, object]) -> None:
        self._ensure_manifest_fits(manifest)
        _atomic_write(self._manifest_path, _canonical_line_bytes(manifest))
        self._manifest_sha256 = str(manifest["manifest_sha256"])
        self._durable_manifest = _canonical_mapping_clone(
            manifest,
            field="manifest",
            max_bytes=self.config.max_manifest_bytes,
        )

    def _ensure_manifest_fits(self, manifest: Mapping[str, object]) -> None:
        encoded = _canonical_line_bytes(manifest)
        if len(encoded) > self.config.max_manifest_bytes:
            raise OpenEcologyEvidenceError(
                "evidence manifest exceeds max_manifest_bytes "
                f"({len(encoded)} > {self.config.max_manifest_bytes})"
            )

    def _ensure_writable(self) -> None:
        if self._finished:
            raise RuntimeError("open ecology evidence writer is finished")
        if self._aborted:
            raise RuntimeError("open ecology evidence writer is aborted")
        if self._checkpointed:
            raise RuntimeError(
                "open ecology evidence writer was checkpointed; construct a "
                "new writer with its continuation state"
            )
        if self._lock_fd is None:
            raise RuntimeError("open ecology evidence writer has no stream lock")

    def _acquire_lock(self) -> None:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(
                self._lock_path,
                flags,
                0o600,
            )
        except OSError as error:
            raise OpenEcologyEvidenceError(
                "cannot create the evidence stream lock"
            ) from error
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise OpenEcologyEvidenceError(
                    "evidence stream lock must be one regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            os.close(descriptor)
            raise OpenEcologyEvidenceError(
                "evidence stream already has an active writer"
            ) from error
        try:
            if file_stat.st_size > len(_OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES):
                raise OpenEcologyEvidenceError(
                    "evidence stream lock exceeds its byte ceiling"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            existing = os.read(
                descriptor,
                len(_OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES) + 1,
            )
            if existing not in {b"", _OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES}:
                raise OpenEcologyEvidenceError(
                    "evidence stream lock does not match its control contract"
                )
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, _OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES)
            os.fsync(descriptor)
        except (OSError, OpenEcologyEvidenceError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        self._lock_fd = descriptor

    def _release_lock(self) -> None:
        descriptor = self._lock_fd
        if descriptor is None:
            return
        self._lock_fd = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def validate_open_ecology_event_record(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate one exact serialized event and reject unknown fields/types."""

    _exact_keys(
        payload,
        frozenset(
            {
                "event_id",
                "event_index",
                "event_type",
                "payload",
                "schema_version",
                "tick",
            }
        ),
        field="event",
    )
    if payload["schema_version"] != OPEN_ECOLOGY_EVIDENCE_EVENT_SCHEMA:
        raise OpenEcologyEvidenceError("unknown evidence event schema_version")
    event_type = payload["event_type"]
    if not isinstance(event_type, str):
        raise OpenEcologyEvidenceError("event_type must be a string")
    common = {
        "tick": _nonnegative_int(payload["tick"], field="event.tick"),
        "event_index": _nonnegative_int(
            payload["event_index"],
            field="event.event_index",
        ),
        "event_id": _identifier(payload["event_id"], field="event.event_id"),
    }
    event_payload = _mapping(payload["payload"], field="event.payload")

    if event_type == "birth":
        _exact_keys(
            event_payload,
            frozenset(
                {
                    "child_agent_id",
                    "genome_sha256",
                    "lineage_id",
                    "parent_agent_ids",
                }
            ),
            field="event.payload",
        )
        event: OpenEcologyEvidenceEvent = BirthEvidence(
            **common,
            child_agent_id=event_payload["child_agent_id"],
            parent_agent_ids=_int_tuple(event_payload["parent_agent_ids"]),
            lineage_id=event_payload["lineage_id"],
            genome_sha256=event_payload["genome_sha256"],
        )
    elif event_type == "death":
        _exact_keys(
            event_payload,
            frozenset({"agent_id", "cause", "source_agent_id"}),
            field="event.payload",
        )
        event = DeathEvidence(
            **common,
            agent_id=event_payload["agent_id"],
            cause=event_payload["cause"],
            source_agent_id=event_payload["source_agent_id"],
        )
    elif event_type == "lineage":
        _exact_keys(
            event_payload,
            frozenset(
                {
                    "agent_id",
                    "lineage_id",
                    "parent_lineage_ids",
                    "transition_kind",
                }
            ),
            field="event.payload",
        )
        event = LineageEvidence(
            **common,
            agent_id=event_payload["agent_id"],
            lineage_id=event_payload["lineage_id"],
            parent_lineage_ids=_str_tuple(event_payload["parent_lineage_ids"]),
            transition_kind=event_payload["transition_kind"],
        )
    elif event_type == "dyadic_interaction":
        _exact_keys(
            event_payload,
            frozenset(
                {
                    "actor_agent_id",
                    "interaction_kind",
                    "magnitude",
                    "outcome",
                    "target_agent_id",
                }
            ),
            field="event.payload",
        )
        event = DyadicInteractionEvidence(
            **common,
            actor_agent_id=event_payload["actor_agent_id"],
            target_agent_id=event_payload["target_agent_id"],
            interaction_kind=event_payload["interaction_kind"],
            outcome=event_payload["outcome"],
            magnitude=event_payload["magnitude"],
        )
    elif event_type == "signal_contributor":
        _exact_keys(
            event_payload,
            frozenset(
                {
                    "contribution",
                    "emitter_agent_id",
                    "receiver_agent_id",
                    "token_id",
                }
            ),
            field="event.payload",
        )
        event = SignalContributorEvidence(
            **common,
            emitter_agent_id=event_payload["emitter_agent_id"],
            receiver_agent_id=event_payload["receiver_agent_id"],
            token_id=event_payload["token_id"],
            contribution=event_payload["contribution"],
        )
    elif event_type == "congestion":
        _exact_keys(
            event_payload,
            frozenset(
                {
                    "contender_agent_ids",
                    "outcome",
                    "target_x",
                    "target_y",
                    "winner_agent_id",
                }
            ),
            field="event.payload",
        )
        event = CongestionEvidence(
            **common,
            target_x=event_payload["target_x"],
            target_y=event_payload["target_y"],
            contender_agent_ids=_int_tuple(event_payload["contender_agent_ids"]),
            winner_agent_id=event_payload["winner_agent_id"],
            outcome=event_payload["outcome"],
        )
    elif event_type == "intervention":
        _exact_keys(
            event_payload,
            frozenset(
                {
                    "assigned_action",
                    "branch_id",
                    "intervention_id",
                    "intervention_kind",
                    "subject_agent_id",
                    "value_sha256",
                }
            ),
            field="event.payload",
        )
        event = InterventionEvidence(
            **common,
            intervention_id=event_payload["intervention_id"],
            intervention_kind=event_payload["intervention_kind"],
            branch_id=event_payload["branch_id"],
            subject_agent_id=event_payload["subject_agent_id"],
            assigned_action=event_payload["assigned_action"],
            value_sha256=event_payload["value_sha256"],
        )
    else:
        raise OpenEcologyEvidenceError(f"unknown evidence event_type {event_type!r}")

    normalized = event.to_record()
    if normalized != payload:
        raise OpenEcologyEvidenceError(
            "event payload is not normalized to the exact typed contract"
        )
    return normalized


def _load_pending_transaction(path: Path) -> dict[str, object]:
    transaction = _parse_canonical_json_line(
        _safe_read_file(
            path,
            max_bytes=_DEFAULT_SAFE_TRANSACTION_READ_BYTES,
        ),
        field="pending transaction",
    )
    _exact_keys(
        transaction,
        frozenset(
            {
                "previous_manifest",
                "prospective_manifest",
                "run_id",
                "schema_version",
                "shard_entry",
                "source_contract_sha256",
                "temp_file_name",
                "transaction_sha256",
                "writer_config_sha256",
            }
        ),
        field="pending transaction",
    )
    if transaction["schema_version"] != OPEN_ECOLOGY_EVIDENCE_TRANSACTION_SCHEMA:
        raise OpenEcologyEvidenceError("unknown pending evidence transaction schema")
    _identifier(transaction["run_id"], field="pending transaction.run_id")
    for field_name in (
        "source_contract_sha256",
        "transaction_sha256",
        "writer_config_sha256",
    ):
        _sha256(
            transaction[field_name],
            field=f"pending transaction.{field_name}",
        )
    body = dict(transaction)
    supplied_digest = body.pop("transaction_sha256")
    if _digest(body) != supplied_digest:
        raise OpenEcologyEvidenceError("pending transaction SHA256 mismatch")

    previous_manifest = _mapping(
        transaction["previous_manifest"],
        field="pending transaction.previous_manifest",
    )
    prospective_manifest = _mapping(
        transaction["prospective_manifest"],
        field="pending transaction.prospective_manifest",
    )
    previous_entries = _mapping_sequence(previous_manifest.get("completed_shards"))
    prospective_entries = _mapping_sequence(
        prospective_manifest.get("completed_shards")
    )
    shard_entry = _mapping(
        transaction["shard_entry"],
        field="pending transaction.shard_entry",
    )
    if prospective_entries != [*previous_entries, shard_entry]:
        raise OpenEcologyEvidenceError(
            "pending transaction prospective manifest is not one-shard append"
        )
    if prospective_manifest.get("completed_shard_count") != len(prospective_entries):
        raise OpenEcologyEvidenceError(
            "pending transaction prospective shard count mismatch"
        )
    _embedded_manifest_sha256(
        previous_manifest,
        field="pending transaction.previous_manifest",
    )
    _embedded_manifest_sha256(
        prospective_manifest,
        field="pending transaction.prospective_manifest",
    )
    return transaction


def _embedded_manifest_sha256(
    manifest: Mapping[str, object],
    *,
    field: str,
) -> str:
    supplied = _sha256(
        manifest.get("manifest_sha256"),
        field=f"{field}.manifest_sha256",
    )
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if _digest(body) != supplied:
        raise OpenEcologyEvidenceError(f"{field} SHA256 mismatch")
    return supplied


def _transaction_temp_file_name(
    value: object,
    *,
    expected_shard_index: int,
) -> str:
    if not isinstance(value, str) or Path(value).name != value:
        raise OpenEcologyEvidenceError(
            "pending transaction temp_file_name must be a basename"
        )
    expected_prefix = f".shard-{expected_shard_index:08d}."
    if (
        not value.startswith(expected_prefix)
        or not value.endswith(".tmp")
        or len(value) > 255
    ):
        raise OpenEcologyEvidenceError(
            "pending transaction temp_file_name is not canonical"
        )
    return value


def _require_file_sha256(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> None:
    if _sha256_file(path, max_bytes=max_bytes) != expected_sha256:
        raise OpenEcologyEvidenceError(
            f"transaction-owned file {path.name} SHA256 mismatch"
        )


def validate_open_ecology_evidence_continuation_state(
    payload: Mapping[str, object],
) -> dict[str, object]:
    _exact_keys(
        payload,
        frozenset(
            {
                "chain_head_sha256",
                "completed_shard_count",
                "last_tick",
                "manifest_sha256",
                "next_event_index",
                "next_shard_index",
                "run_id",
                "schema_version",
                "source_contract_sha256",
                "state_sha256",
                "total_compressed_bytes",
                "total_event_count",
                "writer_config_sha256",
            }
        ),
        field="continuation_state",
    )
    normalized = _canonical_mapping_clone(
        payload,
        field="continuation_state",
        max_bytes=1024 * 1024,
    )
    if normalized["schema_version"] != OPEN_ECOLOGY_EVIDENCE_CONTINUATION_SCHEMA:
        raise OpenEcologyEvidenceError("unknown evidence continuation schema_version")
    for field_name in (
        "chain_head_sha256",
        "manifest_sha256",
        "source_contract_sha256",
        "state_sha256",
        "writer_config_sha256",
    ):
        _sha256(normalized[field_name], field=f"continuation_state.{field_name}")
    _identifier(normalized["run_id"], field="continuation_state.run_id")
    for field_name in (
        "completed_shard_count",
        "next_event_index",
        "next_shard_index",
        "total_compressed_bytes",
        "total_event_count",
    ):
        _nonnegative_int(
            normalized[field_name],
            field=f"continuation_state.{field_name}",
        )
    if normalized["completed_shard_count"] != normalized["next_shard_index"]:
        raise OpenEcologyEvidenceError(
            "continuation shard count must equal next_shard_index"
        )
    if normalized["next_event_index"] != normalized["total_event_count"]:
        raise OpenEcologyEvidenceError(
            "continuation next_event_index must equal total_event_count"
        )
    _optional_nonnegative_int(
        normalized["last_tick"],
        field="continuation_state.last_tick",
    )
    body = dict(normalized)
    supplied_digest = body.pop("state_sha256")
    if _digest(body) != supplied_digest:
        raise OpenEcologyEvidenceError("continuation state SHA256 mismatch")
    return normalized


def load_open_ecology_evidence_manifest(
    output_directory: str | Path,
    *,
    verify_shards: bool = True,
    expected_manifest_sha256: str | None = None,
    expected_status: str | None = None,
) -> dict[str, object]:
    directory = Path(output_directory)
    manifest_path = directory / OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME
    raw = _safe_read_file(
        manifest_path,
        max_bytes=_DEFAULT_SAFE_MANIFEST_READ_BYTES,
    )
    manifest = _parse_canonical_json_line(raw, field="manifest")
    _exact_keys(
        manifest,
        frozenset(
            {
                "chain_head_sha256",
                "completed_shard_count",
                "completed_shards",
                "format",
                "last_tick",
                "manifest_sha256",
                "next_event_index",
                "next_shard_index",
                "run_id",
                "schema_version",
                "source_contract",
                "source_contract_sha256",
                "status",
                "total_compressed_bytes",
                "total_event_count",
                "writer_config",
                "writer_config_sha256",
            }
        ),
        field="manifest",
    )
    if manifest["schema_version"] != OPEN_ECOLOGY_EVIDENCE_MANIFEST_SCHEMA:
        raise OpenEcologyEvidenceError("unknown evidence manifest schema_version")
    if manifest["format"] != OPEN_ECOLOGY_EVIDENCE_FORMAT:
        raise OpenEcologyEvidenceError("unknown evidence shard format")
    if manifest["status"] not in {"open", "complete"}:
        raise OpenEcologyEvidenceError("manifest status must be open or complete")
    if expected_status is not None:
        if expected_status not in {"open", "complete"}:
            raise OpenEcologyEvidenceError("expected_status must be open or complete")
        if manifest["status"] != expected_status:
            raise OpenEcologyEvidenceError(
                "manifest status does not match expected_status"
            )
    _identifier(manifest["run_id"], field="manifest.run_id")
    config = RotatingEvidenceConfig.from_dict(
        _mapping(manifest["writer_config"], field="manifest.writer_config")
    )
    if len(raw) > config.max_manifest_bytes:
        raise OpenEcologyEvidenceError(
            "manifest exceeds its declared max_manifest_bytes"
        )
    source_contract = _canonical_mapping_clone(
        manifest["source_contract"],
        field="manifest.source_contract",
        max_bytes=config.max_source_contract_bytes,
    )
    if _digest(source_contract) != manifest["source_contract_sha256"]:
        raise OpenEcologyEvidenceError("source contract SHA256 mismatch")
    if _digest(config.to_dict()) != manifest["writer_config_sha256"]:
        raise OpenEcologyEvidenceError("writer config SHA256 mismatch")
    body = dict(manifest)
    supplied_manifest_digest = body.pop("manifest_sha256")
    _sha256(supplied_manifest_digest, field="manifest.manifest_sha256")
    if _digest(body) != supplied_manifest_digest:
        raise OpenEcologyEvidenceError("manifest SHA256 mismatch")
    if expected_manifest_sha256 is not None:
        expected_digest = _sha256(
            expected_manifest_sha256,
            field="expected_manifest_sha256",
        )
        if supplied_manifest_digest != expected_digest:
            raise OpenEcologyEvidenceError(
                "manifest SHA256 does not match the externally expected digest"
            )

    entries = _mapping_sequence(manifest["completed_shards"])
    completed_count = _nonnegative_int(
        manifest["completed_shard_count"],
        field="manifest.completed_shard_count",
    )
    if len(entries) != completed_count:
        raise OpenEcologyEvidenceError(
            "manifest completed_shard_count does not match completed_shards"
        )
    if completed_count > config.max_shards:
        raise OpenEcologyEvidenceError("manifest exceeds max_shards")
    next_shard_index = _nonnegative_int(
        manifest["next_shard_index"],
        field="manifest.next_shard_index",
    )
    if next_shard_index != completed_count:
        raise OpenEcologyEvidenceError(
            "manifest next_shard_index must equal completed_shard_count"
        )
    total_event_count = _nonnegative_int(
        manifest["total_event_count"],
        field="manifest.total_event_count",
    )
    declared_total_compressed_bytes = _nonnegative_int(
        manifest["total_compressed_bytes"],
        field="manifest.total_compressed_bytes",
    )
    if declared_total_compressed_bytes > config.max_total_compressed_bytes:
        raise OpenEcologyEvidenceError("manifest exceeds max_total_compressed_bytes")
    next_event_index = _nonnegative_int(
        manifest["next_event_index"],
        field="manifest.next_event_index",
    )
    if next_event_index != total_event_count:
        raise OpenEcologyEvidenceError(
            "manifest next_event_index must equal total_event_count"
        )
    last_tick = _optional_nonnegative_int(
        manifest["last_tick"],
        field="manifest.last_tick",
    )
    chain_head = _sha256(
        manifest["chain_head_sha256"],
        field="manifest.chain_head_sha256",
    )

    expected_previous = OPEN_ECOLOGY_EVIDENCE_CHAIN_GENESIS
    expected_event_index = 0
    summed_events = 0
    summed_compressed_bytes = 0
    observed_last_tick: int | None = None
    declared_files: set[str] = set()
    for index, entry in enumerate(entries):
        normalized_entry = _validate_manifest_entry(
            entry,
            expected_index=index,
            expected_previous=expected_previous,
            expected_first_event_index=expected_event_index,
            config=config,
        )
        if (
            observed_last_tick is not None
            and int(normalized_entry["first_tick"]) < observed_last_tick
        ):
            raise OpenEcologyEvidenceError(
                "manifest shard ticks are globally out of order"
            )
        file_name = str(normalized_entry["file_name"])
        if file_name in declared_files:
            raise OpenEcologyEvidenceError("manifest has duplicate shard file names")
        declared_files.add(file_name)
        if verify_shards:
            _verify_shard(
                directory / file_name,
                entry=normalized_entry,
                run_id=str(manifest["run_id"]),
                source_contract_sha256=str(manifest["source_contract_sha256"]),
                writer_config_sha256=str(manifest["writer_config_sha256"]),
                config=config,
            )
        event_count = int(normalized_entry["event_count"])
        summed_events += event_count
        summed_compressed_bytes += int(normalized_entry["compressed_bytes"])
        expected_event_index += event_count
        observed_last_tick = int(normalized_entry["last_tick"])
        expected_previous = str(normalized_entry["file_sha256"])

    if summed_events != total_event_count:
        raise OpenEcologyEvidenceError(
            "manifest total_event_count does not equal shard event counts"
        )
    if summed_compressed_bytes != declared_total_compressed_bytes:
        raise OpenEcologyEvidenceError(
            "manifest total_compressed_bytes does not equal shard byte counts"
        )
    if expected_event_index != next_event_index:
        raise OpenEcologyEvidenceError(
            "manifest next_event_index does not follow the shard sequence"
        )
    if observed_last_tick != last_tick:
        raise OpenEcologyEvidenceError(
            "manifest last_tick does not match the final shard"
        )
    if expected_previous != chain_head:
        raise OpenEcologyEvidenceError(
            "manifest chain_head_sha256 does not match the final shard"
        )
    lock_bytes = _safe_read_file(
        directory / OPEN_ECOLOGY_EVIDENCE_LOCK_NAME,
        max_bytes=len(_OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES),
        require_single_link=True,
    )
    if lock_bytes != _OPEN_ECOLOGY_EVIDENCE_LOCK_BYTES:
        raise OpenEcologyEvidenceError(
            "evidence stream lock does not match its exact control contract"
        )
    expected_directory_entries = {
        OPEN_ECOLOGY_EVIDENCE_LOCK_NAME,
        OPEN_ECOLOGY_EVIDENCE_MANIFEST_NAME,
        *declared_files,
    }
    actual_directory_entries = {path.name for path in directory.iterdir()}
    if actual_directory_entries != expected_directory_entries:
        raise OpenEcologyEvidenceError(
            "evidence directory entries do not exactly match the manifest"
        )
    return manifest


def _validate_manifest_entry(
    entry: Mapping[str, object],
    *,
    expected_index: int,
    expected_previous: str,
    expected_first_event_index: int,
    config: RotatingEvidenceConfig,
) -> dict[str, object]:
    _exact_keys(
        entry,
        frozenset(
            {
                "compressed_bytes",
                "event_count",
                "event_stream_sha256",
                "file_name",
                "file_sha256",
                "first_event_index",
                "first_tick",
                "last_event_index",
                "last_tick",
                "pre_footer_content_sha256",
                "previous_shard_sha256",
                "shard_index",
                "uncompressed_bytes",
            }
        ),
        field="manifest shard entry",
    )
    normalized = dict(entry)
    shard_index = _nonnegative_int(
        normalized["shard_index"],
        field="entry.shard_index",
    )
    if shard_index != expected_index:
        raise OpenEcologyEvidenceError("manifest shard indexes are not contiguous")
    file_name = normalized["file_name"]
    if (
        not isinstance(file_name, str)
        or file_name != f"shard-{shard_index:08d}.jsonl.gz"
    ):
        raise OpenEcologyEvidenceError("manifest shard file_name is not canonical")
    event_count = _positive_int(
        normalized["event_count"],
        field="entry.event_count",
    )
    if event_count > config.max_rows_per_shard:
        raise OpenEcologyEvidenceError("shard event_count exceeds row ceiling")
    first_event_index = _nonnegative_int(
        normalized["first_event_index"],
        field="entry.first_event_index",
    )
    last_event_index = _nonnegative_int(
        normalized["last_event_index"],
        field="entry.last_event_index",
    )
    if first_event_index != expected_first_event_index:
        raise OpenEcologyEvidenceError(
            "manifest shard event indexes are not contiguous"
        )
    if last_event_index != first_event_index + event_count - 1:
        raise OpenEcologyEvidenceError(
            "manifest shard last_event_index does not match event_count"
        )
    first_tick = _nonnegative_int(
        normalized["first_tick"],
        field="entry.first_tick",
    )
    last_tick = _nonnegative_int(
        normalized["last_tick"],
        field="entry.last_tick",
    )
    if last_tick < first_tick:
        raise OpenEcologyEvidenceError("manifest shard ticks are reversed")
    if last_tick - first_tick + 1 > config.max_ticks_per_shard:
        raise OpenEcologyEvidenceError("manifest shard exceeds tick ceiling")
    compressed_bytes = _positive_int(
        normalized["compressed_bytes"],
        field="entry.compressed_bytes",
    )
    if compressed_bytes > config.max_compressed_bytes_per_shard:
        raise OpenEcologyEvidenceError("manifest shard exceeds compressed ceiling")
    uncompressed_bytes = _positive_int(
        normalized["uncompressed_bytes"],
        field="entry.uncompressed_bytes",
    )
    if uncompressed_bytes > config.max_uncompressed_bytes_per_shard:
        raise OpenEcologyEvidenceError("manifest shard exceeds uncompressed ceiling")
    for field_name in (
        "event_stream_sha256",
        "file_sha256",
        "pre_footer_content_sha256",
        "previous_shard_sha256",
    ):
        _sha256(normalized[field_name], field=f"entry.{field_name}")
    if normalized["previous_shard_sha256"] != expected_previous:
        raise OpenEcologyEvidenceError("manifest shard hash chain is broken")
    return normalized


def _verify_shard(
    path: Path,
    *,
    entry: Mapping[str, object],
    run_id: str,
    source_contract_sha256: str,
    writer_config_sha256: str,
    config: RotatingEvidenceConfig,
) -> None:
    compressed = _safe_read_file(
        path,
        max_bytes=config.max_compressed_bytes_per_shard,
    )
    if len(compressed) != entry["compressed_bytes"]:
        raise OpenEcologyEvidenceError("shard compressed byte count mismatch")
    if hashlib.sha256(compressed).hexdigest() != entry["file_sha256"]:
        raise OpenEcologyEvidenceError("shard file SHA256 mismatch")
    uncompressed = _bounded_gzip_decompress(
        compressed,
        max_bytes=config.max_uncompressed_bytes_per_shard,
    )
    if len(uncompressed) != entry["uncompressed_bytes"]:
        raise OpenEcologyEvidenceError("shard uncompressed byte count mismatch")
    if (
        _deterministic_gzip_bytes(
            uncompressed,
            compresslevel=config.gzip_compresslevel,
        )
        != compressed
    ):
        raise OpenEcologyEvidenceError("shard gzip bytes are not canonical")
    if not uncompressed.endswith(b"\n"):
        raise OpenEcologyEvidenceError("shard JSONL must end with a newline")
    raw_lines = uncompressed.splitlines(keepends=True)
    if len(raw_lines) < 3:
        raise OpenEcologyEvidenceError("shard must contain a header, event, and footer")
    parsed_lines = [
        _parse_canonical_json_line(line, field=f"shard line {index}")
        for index, line in enumerate(raw_lines)
    ]
    expected_header = {
        "format": OPEN_ECOLOGY_EVIDENCE_FORMAT,
        "previous_shard_sha256": entry["previous_shard_sha256"],
        "run_id": run_id,
        "shard_index": entry["shard_index"],
        "source_contract_sha256": source_contract_sha256,
        "type": "header",
        "writer_config_sha256": writer_config_sha256,
    }
    if parsed_lines[0] != expected_header:
        raise OpenEcologyEvidenceError("shard header does not match the manifest")
    event_lines = parsed_lines[1:-1]
    if len(event_lines) != entry["event_count"]:
        raise OpenEcologyEvidenceError("shard event row count mismatch")

    content_hash = hashlib.sha256()
    event_stream_hash = hashlib.sha256()
    content_hash.update(raw_lines[0])
    expected_event_index = int(entry["first_event_index"])
    previous_tick: int | None = None
    observed_first_tick: int | None = None
    for offset, (wrapper, raw_line) in enumerate(
        zip(event_lines, raw_lines[1:-1], strict=True)
    ):
        _exact_keys(wrapper, frozenset({"event", "type"}), field="event row")
        if wrapper["type"] != "event":
            raise OpenEcologyEvidenceError("unknown shard row type")
        record = validate_open_ecology_event_record(
            _mapping(wrapper["event"], field="event row event")
        )
        if record["event_index"] != expected_event_index + offset:
            raise OpenEcologyEvidenceError("shard event indexes are not contiguous")
        tick = int(record["tick"])
        if previous_tick is not None and tick < previous_tick:
            raise OpenEcologyEvidenceError("shard event ticks are out of order")
        if observed_first_tick is None:
            observed_first_tick = tick
        previous_tick = tick
        record_bytes = _canonical_bytes(record)
        event_stream_hash.update(len(record_bytes).to_bytes(8, "big"))
        event_stream_hash.update(record_bytes)
        content_hash.update(raw_line)
    if observed_first_tick != entry["first_tick"]:
        raise OpenEcologyEvidenceError(
            "shard first_tick does not match its first event"
        )
    if previous_tick != entry["last_tick"]:
        raise OpenEcologyEvidenceError("shard last_tick does not match its final event")

    footer = parsed_lines[-1]
    expected_footer = {
        "event_count": entry["event_count"],
        "event_stream_sha256": event_stream_hash.hexdigest(),
        "first_event_index": entry["first_event_index"],
        "first_tick": entry["first_tick"],
        "format": OPEN_ECOLOGY_EVIDENCE_FORMAT,
        "last_event_index": entry["last_event_index"],
        "last_tick": entry["last_tick"],
        "pre_footer_content_sha256": content_hash.hexdigest(),
        "previous_shard_sha256": entry["previous_shard_sha256"],
        "shard_index": entry["shard_index"],
        "type": "footer",
        "uncompressed_bytes_before_footer": sum(len(line) for line in raw_lines[:-1]),
    }
    if footer != expected_footer:
        raise OpenEcologyEvidenceError("shard footer does not match canonical content")
    if footer["event_stream_sha256"] != entry["event_stream_sha256"]:
        raise OpenEcologyEvidenceError("shard event stream SHA256 mismatch")
    if footer["pre_footer_content_sha256"] != entry["pre_footer_content_sha256"]:
        raise OpenEcologyEvidenceError("shard pre-footer SHA256 mismatch")


def _event_record(
    *,
    event_type: str,
    tick: int,
    event_index: int,
    event_id: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_index": event_index,
        "event_type": event_type,
        "payload": dict(payload),
        "schema_version": OPEN_ECOLOGY_EVIDENCE_EVENT_SCHEMA,
        "tick": tick,
    }


def _event_identity(tick: object, event_index: object, event_id: object) -> None:
    _nonnegative_int(tick, field="tick")
    normalized_index = _nonnegative_int(event_index, field="event_index")
    normalized_id = _identifier(event_id, field="event_id")
    if not normalized_id.endswith(f":{normalized_index}"):
        raise OpenEcologyEvidenceError(
            "event_id must end with the canonical ':<event_index>' suffix"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OpenEcologyEvidenceError(f"{field} must be an object with string keys")
    return value


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise OpenEcologyEvidenceError("completed_shards must be a list")
    return [
        _mapping(item, field=f"completed_shards[{index}]")
        for index, item in enumerate(value)
    ]


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OpenEcologyEvidenceError(f"{field} must be an object with string keys")
    actual = frozenset(value)
    if actual != expected:
        raise OpenEcologyEvidenceError(
            f"{field} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise OpenEcologyEvidenceError(f"{field} must match {_IDENTIFIER_RE.pattern}")
    return value


def _optional_identifier(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyEvidenceError(f"{field} must be lowercase SHA-256 hex")
    return value


def _optional_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field=field)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyEvidenceError(f"{field} must be a positive integer")
    return value


def _bounded_int(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise OpenEcologyEvidenceError(
            f"{field} must be an integer from {minimum} through {maximum}"
        )
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyEvidenceError(f"{field} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field=field)


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenEcologyEvidenceError(f"{field} must be an integer or null")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenEcologyEvidenceError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise OpenEcologyEvidenceError(f"{field} must be a finite number")
    return normalized


def _optional_finite_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field=field)


def _agent_id_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise OpenEcologyEvidenceError(f"{field} must be a tuple")
    if not value and not allow_empty:
        raise OpenEcologyEvidenceError(f"{field} must not be empty")
    if len(value) > 32:
        raise OpenEcologyEvidenceError(f"{field} exceeds 32 entries")
    normalized = tuple(
        _nonnegative_int(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise OpenEcologyEvidenceError(f"{field} must not contain duplicates")
    if normalized != tuple(sorted(normalized)):
        raise OpenEcologyEvidenceError(f"{field} must be sorted")
    return normalized


def _identifier_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise OpenEcologyEvidenceError(f"{field} must be a tuple")
    if not value and not allow_empty:
        raise OpenEcologyEvidenceError(f"{field} must not be empty")
    if len(value) > 32:
        raise OpenEcologyEvidenceError(f"{field} exceeds 32 entries")
    normalized = tuple(
        _identifier(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise OpenEcologyEvidenceError(f"{field} must not contain duplicates")
    if normalized != tuple(sorted(normalized)):
        raise OpenEcologyEvidenceError(f"{field} must be sorted")
    return normalized


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise OpenEcologyEvidenceError("event integer tuple must serialize as a list")
    return tuple(value)


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OpenEcologyEvidenceError("event string tuple must serialize as a list")
    return tuple(value)


def _canonical_mapping_clone(
    value: object,
    *,
    field: str,
    max_bytes: int,
) -> dict[str, object]:
    mapping = _mapping(value, field=field)
    encoded = _canonical_bytes(mapping)
    if len(encoded) > max_bytes:
        raise OpenEcologyEvidenceError(
            f"{field} exceeds its canonical byte ceiling ({len(encoded)} > {max_bytes})"
        )
    normalized = _parse_json(encoded, field=field)
    if not isinstance(normalized, dict):
        raise OpenEcologyEvidenceError(f"{field} must normalize to an object")
    return normalized


def _canonical_bytes(value: object) -> bytes:
    _validate_json_shape(value, field="value", depth=0)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise OpenEcologyEvidenceError(
            f"value is not canonical JSON: {error}"
        ) from error
    return text.encode("utf-8")


def _canonical_line_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _validate_json_shape(value: object, *, field: str, depth: int) -> None:
    if depth > _MAX_JSON_NESTING_DEPTH:
        raise OpenEcologyEvidenceError(f"{field} exceeds maximum JSON nesting depth")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        _finite_number(value, field=field)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_shape(
                item,
                field=f"{field}[{index}]",
                depth=depth + 1,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise OpenEcologyEvidenceError(
                    f"{field} contains a non-string object key"
                )
            _validate_json_shape(
                item,
                field=f"{field}.{key}",
                depth=depth + 1,
            )
        return
    raise OpenEcologyEvidenceError(
        f"{field} contains unsupported JSON type {type(value).__name__}"
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parse_json(data: bytes, *, field: str) -> object:
    def reject_constant(value: str) -> object:
        raise OpenEcologyEvidenceError(
            f"{field} contains non-finite JSON constant {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OpenEcologyEvidenceError(
                    f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except UnicodeDecodeError as error:
        raise OpenEcologyEvidenceError(f"{field} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise OpenEcologyEvidenceError(f"{field} is not valid JSON") from error


def _parse_canonical_json_line(data: bytes, *, field: str) -> dict[str, object]:
    parsed = _parse_json(data, field=field)
    if not isinstance(parsed, dict):
        raise OpenEcologyEvidenceError(f"{field} must be a JSON object")
    if _canonical_line_bytes(parsed) != data:
        raise OpenEcologyEvidenceError(f"{field} is not canonical JSONL")
    return parsed


def _safe_read_file(
    path: Path,
    *,
    max_bytes: int,
    require_single_link: bool = False,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OpenEcologyEvidenceError(
            f"cannot safely open evidence file {path.name}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size > max_bytes
            or (require_single_link and file_stat.st_nlink != 1)
        ):
            raise OpenEcologyEvidenceError(
                f"evidence file {path.name} exceeds its byte ceiling or is not regular"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise OpenEcologyEvidenceError(
                f"evidence file {path.name} exceeds its byte ceiling"
            )
        return data
    finally:
        os.close(descriptor)


def _sha256_file(path: Path, *, max_bytes: int) -> str:
    return hashlib.sha256(_safe_read_file(path, max_bytes=max_bytes)).hexdigest()


def _bounded_gzip_decompress(data: bytes, *, max_bytes: int) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as handle:
            result = handle.read(max_bytes + 1)
    except (OSError, EOFError) as error:
        raise OpenEcologyEvidenceError("shard is not valid gzip") from error
    if len(result) > max_bytes:
        raise OpenEcologyEvidenceError("shard exceeds max_uncompressed_bytes_per_shard")
    return result


def _deterministic_gzip_bytes(data: bytes, *, compresslevel: int) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=buffer,
        mtime=0,
        compresslevel=compresslevel,
    ) as handle:
        handle.write(data)
    return buffer.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory_best_effort(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _fsync_directory_best_effort(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)
