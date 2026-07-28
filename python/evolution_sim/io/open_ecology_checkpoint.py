from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping


OPEN_ECOLOGY_CHECKPOINT_SCHEMA_VERSION = "open_ecology_checkpoint_v1"
OPEN_ECOLOGY_GENERATION_IDENTITY_SCHEMA_VERSION = "open_ecology_generation_identity_v1"
OPEN_ECOLOGY_STATE_ENVELOPE_SCHEMA_VERSION = "open_ecology_state_envelope_v1"
OPEN_ECOLOGY_CHECKPOINT_DIGEST_POLICY = (
    "sha256_canonical_json_without_checkpoint_sha256_v1"
)
OPEN_ECOLOGY_STATE_DIGEST_POLICY = (
    "sha256_canonical_json_of_schema_version_and_payload_v1"
)
OPEN_ECOLOGY_GENERATION_IDENTITY_DIGEST_POLICY = (
    "sha256_canonical_json_without_identity_sha256_v1"
)
OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024

REQUIRED_CHECKPOINT_COMPONENTS = (
    "world_state",
    "environment_rng_state",
    "recurrent_policy_state",
    "public_feedback_history",
    "sampling_rng_state",
    "genome_population_snapshot",
    "evidence_writer_continuation_state",
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "format_contract",
        "source",
        "generation_identity",
        "tick",
        "components",
        "restartability",
        "checkpoint_sha256",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "git_sha",
        "config_contract",
        "seed_contract",
    }
)
_STATE_ENVELOPE_KEYS = frozenset(
    {
        "envelope_schema_version",
        "present",
        "schema_version",
        "payload",
        "state_sha256",
    }
)
_GENERATION_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "run_generation_id",
        "island_id",
        "generation_index",
        "source_git_sha",
        "config_contract_sha256",
        "seed_contract_sha256",
        "digest_policy",
        "identity_sha256",
    }
)
_RESTARTABILITY_KEYS = frozenset(
    {
        "restartable",
        "required_components",
        "validated_components",
        "missing_components",
        "policy",
    }
)
_FORMAT_CONTRACT = {
    "encoding": "canonical_json_utf8_with_single_trailing_lf",
    "compression": "none",
    "checkpoint_digest_policy": OPEN_ECOLOGY_CHECKPOINT_DIGEST_POLICY,
    "state_digest_policy": OPEN_ECOLOGY_STATE_DIGEST_POLICY,
    "generation_identity_digest_policy": (
        OPEN_ECOLOGY_GENERATION_IDENTITY_DIGEST_POLICY
    ),
    "duplicate_json_keys_allowed": False,
    "pickle_or_executable_payload_allowed": False,
    "required_components": list(REQUIRED_CHECKPOINT_COMPONENTS),
    "restartable_policy": (
        "true_only_when_every_required_component_is_present_and_container_valid"
    ),
}
_RESTARTABILITY_POLICY = (
    "container_complete_only; end_to_end_resume_requires_external_adapter_and_"
    "continuation_equivalence_validation"
)
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SCHEMA_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_MAX_JSON_NESTING_DEPTH = 128


class OpenEcologyCheckpointError(ValueError):
    """Raised when a persistent-island checkpoint fails closed."""


@dataclass(frozen=True, slots=True)
class VersionedCheckpointState:
    """Caller-owned JSON state bound to an explicit adapter schema version."""

    schema_version: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, field="schema_version")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise OpenEcologyCheckpointError(
                "checkpoint state payload must be a non-empty mapping"
            )
        _canonical_clone(dict(self.payload), field="checkpoint state payload")


def build_open_ecology_checkpoint(
    *,
    source_git_sha: str,
    config_contract: VersionedCheckpointState,
    seed_contract: VersionedCheckpointState,
    run_generation_id: str,
    island_id: str,
    generation_index: int,
    tick: int,
    world_state: VersionedCheckpointState | None,
    environment_rng_state: VersionedCheckpointState | None,
    recurrent_policy_state: VersionedCheckpointState | None,
    public_feedback_history: VersionedCheckpointState | None,
    sampling_rng_state: VersionedCheckpointState | None,
    genome_population_snapshot: VersionedCheckpointState | None,
    evidence_writer_continuation_state: VersionedCheckpointState | None,
) -> dict[str, object]:
    """Build a validated JSON container without claiming world-level resumability."""

    parsed_git_sha = _git_sha(source_git_sha)
    parsed_generation_id = _identifier(
        run_generation_id,
        field="run_generation_id",
    )
    parsed_island_id = _identifier(island_id, field="island_id")
    parsed_generation_index = _nonnegative_int(
        generation_index,
        field="generation_index",
    )
    parsed_tick = _nonnegative_int(tick, field="tick")
    config_envelope = _build_present_envelope(
        config_contract,
        field="config_contract",
    )
    seed_envelope = _build_present_envelope(
        seed_contract,
        field="seed_contract",
    )
    components = {
        "world_state": _build_optional_envelope(
            world_state,
            field="world_state",
        ),
        "environment_rng_state": _build_optional_envelope(
            environment_rng_state,
            field="environment_rng_state",
        ),
        "recurrent_policy_state": _build_optional_envelope(
            recurrent_policy_state,
            field="recurrent_policy_state",
        ),
        "public_feedback_history": _build_optional_envelope(
            public_feedback_history,
            field="public_feedback_history",
        ),
        "sampling_rng_state": _build_optional_envelope(
            sampling_rng_state,
            field="sampling_rng_state",
        ),
        "genome_population_snapshot": _build_optional_envelope(
            genome_population_snapshot,
            field="genome_population_snapshot",
        ),
        "evidence_writer_continuation_state": _build_optional_envelope(
            evidence_writer_continuation_state,
            field="evidence_writer_continuation_state",
        ),
    }
    generation_identity = _build_generation_identity(
        run_generation_id=parsed_generation_id,
        island_id=parsed_island_id,
        generation_index=parsed_generation_index,
        source_git_sha=parsed_git_sha,
        config_contract_sha256=str(config_envelope["state_sha256"]),
        seed_contract_sha256=str(seed_envelope["state_sha256"]),
    )
    payload: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_CHECKPOINT_SCHEMA_VERSION,
        "format_contract": _canonical_clone(
            _FORMAT_CONTRACT,
            field="format contract",
        ),
        "source": {
            "git_sha": parsed_git_sha,
            "config_contract": config_envelope,
            "seed_contract": seed_envelope,
        },
        "generation_identity": generation_identity,
        "tick": parsed_tick,
        "components": components,
        "restartability": _restartability_payload(components),
    }
    payload["checkpoint_sha256"] = _digest(payload)
    return validate_open_ecology_checkpoint(payload)


def write_open_ecology_checkpoint(
    output_path: str | Path,
    *,
    source_git_sha: str,
    config_contract: VersionedCheckpointState,
    seed_contract: VersionedCheckpointState,
    run_generation_id: str,
    island_id: str,
    generation_index: int,
    tick: int,
    world_state: VersionedCheckpointState | None,
    environment_rng_state: VersionedCheckpointState | None,
    recurrent_policy_state: VersionedCheckpointState | None,
    public_feedback_history: VersionedCheckpointState | None,
    sampling_rng_state: VersionedCheckpointState | None,
    genome_population_snapshot: VersionedCheckpointState | None,
    evidence_writer_continuation_state: VersionedCheckpointState | None,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
) -> dict[str, object]:
    """Atomically write one bounded, non-executable checkpoint container."""

    byte_ceiling = _positive_int(
        max_checkpoint_bytes,
        field="max_checkpoint_bytes",
    )
    checkpoint = build_open_ecology_checkpoint(
        source_git_sha=source_git_sha,
        config_contract=config_contract,
        seed_contract=seed_contract,
        run_generation_id=run_generation_id,
        island_id=island_id,
        generation_index=generation_index,
        tick=tick,
        world_state=world_state,
        environment_rng_state=environment_rng_state,
        recurrent_policy_state=recurrent_policy_state,
        public_feedback_history=public_feedback_history,
        sampling_rng_state=sampling_rng_state,
        genome_population_snapshot=genome_population_snapshot,
        evidence_writer_continuation_state=evidence_writer_continuation_state,
    )
    encoded = _canonical_bytes(checkpoint) + b"\n"
    if len(encoded) > byte_ceiling:
        raise OpenEcologyCheckpointError(
            f"checkpoint exceeds max_checkpoint_bytes ({len(encoded)} > {byte_ceiling})"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(encoded)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, destination)
        temp_path = None
        _fsync_directory_best_effort(destination.parent)
    except Exception as error:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise OpenEcologyCheckpointError(
                    "checkpoint write failed and temporary-file cleanup also "
                    f"failed: {cleanup_error}"
                ) from error
        raise
    return checkpoint


def load_open_ecology_checkpoint(
    checkpoint_path: str | Path,
    *,
    max_checkpoint_bytes: int = OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    expected_source_git_sha: str | None = None,
    expected_generation_identity_sha256: str | None = None,
    require_restartable: bool = False,
    require_canonical: bool = True,
) -> dict[str, object]:
    """Safely load JSON only, rejecting links, duplicate keys, and stale identity."""

    byte_ceiling = _positive_int(
        max_checkpoint_bytes,
        field="max_checkpoint_bytes",
    )
    encoded = _read_bounded_regular_file(Path(checkpoint_path), byte_ceiling)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenEcologyCheckpointError(
            "checkpoint is not valid UTF-8 JSON"
        ) from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except OpenEcologyCheckpointError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise OpenEcologyCheckpointError(
            f"checkpoint is not valid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise OpenEcologyCheckpointError("checkpoint root must be an object")
    if require_canonical and encoded != _canonical_bytes(parsed) + b"\n":
        raise OpenEcologyCheckpointError("checkpoint JSON is not canonical")
    return validate_open_ecology_checkpoint(
        parsed,
        expected_source_git_sha=expected_source_git_sha,
        expected_generation_identity_sha256=(expected_generation_identity_sha256),
        require_restartable=require_restartable,
    )


def validate_open_ecology_checkpoint(
    checkpoint: object,
    *,
    expected_source_git_sha: str | None = None,
    expected_generation_identity_sha256: str | None = None,
    require_restartable: bool = False,
) -> dict[str, object]:
    """Validate exact container keys, state digests, identity, and restartability."""

    payload = _object(checkpoint, field="checkpoint")
    _exact_keys(payload, _TOP_LEVEL_KEYS, field="checkpoint")
    if payload["schema_version"] != OPEN_ECOLOGY_CHECKPOINT_SCHEMA_VERSION:
        raise OpenEcologyCheckpointError("checkpoint schema_version is stale")
    if not _canonical_equal(payload["format_contract"], _FORMAT_CONTRACT):
        raise OpenEcologyCheckpointError("checkpoint format_contract mismatch")
    _nonnegative_int(payload["tick"], field="checkpoint.tick")

    source = _object(payload["source"], field="checkpoint.source")
    _exact_keys(source, _SOURCE_KEYS, field="checkpoint.source")
    source_git_sha = _git_sha(source["git_sha"])
    if expected_source_git_sha is not None and source_git_sha != _git_sha(
        expected_source_git_sha
    ):
        raise OpenEcologyCheckpointError("checkpoint source git SHA mismatch")
    config_contract = _validate_present_envelope(
        source["config_contract"],
        field="checkpoint.source.config_contract",
    )
    seed_contract = _validate_present_envelope(
        source["seed_contract"],
        field="checkpoint.source.seed_contract",
    )

    generation_identity = _validate_generation_identity(
        payload["generation_identity"],
        source_git_sha=source_git_sha,
        config_contract_sha256=str(config_contract["state_sha256"]),
        seed_contract_sha256=str(seed_contract["state_sha256"]),
    )
    identity_sha256 = str(generation_identity["identity_sha256"])
    if expected_generation_identity_sha256 is not None and identity_sha256 != _sha256(
        expected_generation_identity_sha256,
        field="expected_generation_identity_sha256",
    ):
        raise OpenEcologyCheckpointError(
            "checkpoint generation identity SHA256 mismatch"
        )

    components = _object(payload["components"], field="checkpoint.components")
    _exact_keys(
        components,
        frozenset(REQUIRED_CHECKPOINT_COMPONENTS),
        field="checkpoint.components",
    )
    normalized_components: dict[str, dict[str, object]] = {}
    for component_name in REQUIRED_CHECKPOINT_COMPONENTS:
        normalized_components[component_name] = _validate_optional_envelope(
            components[component_name],
            field=f"checkpoint.components.{component_name}",
        )
    expected_restartability = _restartability_payload(normalized_components)
    restartability = _object(
        payload["restartability"],
        field="checkpoint.restartability",
    )
    _exact_keys(
        restartability,
        _RESTARTABILITY_KEYS,
        field="checkpoint.restartability",
    )
    if not _canonical_equal(restartability, expected_restartability):
        raise OpenEcologyCheckpointError(
            "checkpoint restartability truth gate mismatch"
        )
    if require_restartable and not bool(restartability["restartable"]):
        raise OpenEcologyCheckpointError(
            "checkpoint is observational/partial and is not restartable"
        )

    observed_checkpoint_sha256 = _sha256(
        payload["checkpoint_sha256"],
        field="checkpoint.checkpoint_sha256",
    )
    digest_payload = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    if _digest(digest_payload) != observed_checkpoint_sha256:
        raise OpenEcologyCheckpointError("checkpoint SHA256 mismatch")
    return _canonical_clone(payload, field="checkpoint")


def checkpoint_generation_identity_sha256(checkpoint: object) -> str:
    validated = validate_open_ecology_checkpoint(checkpoint)
    identity = _object(
        validated["generation_identity"],
        field="checkpoint.generation_identity",
    )
    return str(identity["identity_sha256"])


def _build_present_envelope(
    state: VersionedCheckpointState,
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(state, VersionedCheckpointState):
        raise OpenEcologyCheckpointError(f"{field} must be a VersionedCheckpointState")
    schema_version = _schema_version(
        state.schema_version,
        field=f"{field}.schema_version",
    )
    payload = _canonical_clone(
        dict(state.payload),
        field=f"{field}.payload",
    )
    digest_payload = {
        "schema_version": schema_version,
        "payload": payload,
    }
    return {
        "envelope_schema_version": OPEN_ECOLOGY_STATE_ENVELOPE_SCHEMA_VERSION,
        "present": True,
        "schema_version": schema_version,
        "payload": payload,
        "state_sha256": _digest(digest_payload),
    }


def _build_optional_envelope(
    state: VersionedCheckpointState | None,
    *,
    field: str,
) -> dict[str, object]:
    if state is None:
        return {
            "envelope_schema_version": OPEN_ECOLOGY_STATE_ENVELOPE_SCHEMA_VERSION,
            "present": False,
            "schema_version": None,
            "payload": None,
            "state_sha256": None,
        }
    return _build_present_envelope(state, field=field)


def _validate_present_envelope(
    envelope: object,
    *,
    field: str,
) -> dict[str, object]:
    validated = _validate_optional_envelope(envelope, field=field)
    if not bool(validated["present"]):
        raise OpenEcologyCheckpointError(f"{field} must be present")
    return validated


def _validate_optional_envelope(
    envelope: object,
    *,
    field: str,
) -> dict[str, object]:
    parsed = _object(envelope, field=field)
    _exact_keys(parsed, _STATE_ENVELOPE_KEYS, field=field)
    if parsed["envelope_schema_version"] != OPEN_ECOLOGY_STATE_ENVELOPE_SCHEMA_VERSION:
        raise OpenEcologyCheckpointError(f"{field} envelope schema is stale")
    if type(parsed["present"]) is not bool:
        raise OpenEcologyCheckpointError(f"{field}.present must be a bool")
    if not parsed["present"]:
        if any(
            parsed[key] is not None
            for key in ("schema_version", "payload", "state_sha256")
        ):
            raise OpenEcologyCheckpointError(
                f"{field} absent envelope must contain only null state fields"
            )
        return dict(parsed)

    schema_version = _schema_version(
        parsed["schema_version"],
        field=f"{field}.schema_version",
    )
    state_payload = _object(parsed["payload"], field=f"{field}.payload")
    if not state_payload:
        raise OpenEcologyCheckpointError(f"{field}.payload must be non-empty")
    observed_sha256 = _sha256(
        parsed["state_sha256"],
        field=f"{field}.state_sha256",
    )
    expected_sha256 = _digest(
        {
            "schema_version": schema_version,
            "payload": state_payload,
        }
    )
    if observed_sha256 != expected_sha256:
        raise OpenEcologyCheckpointError(f"{field} state SHA256 mismatch")
    return dict(parsed)


def _build_generation_identity(
    *,
    run_generation_id: str,
    island_id: str,
    generation_index: int,
    source_git_sha: str,
    config_contract_sha256: str,
    seed_contract_sha256: str,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_GENERATION_IDENTITY_SCHEMA_VERSION,
        "run_generation_id": run_generation_id,
        "island_id": island_id,
        "generation_index": generation_index,
        "source_git_sha": source_git_sha,
        "config_contract_sha256": config_contract_sha256,
        "seed_contract_sha256": seed_contract_sha256,
        "digest_policy": OPEN_ECOLOGY_GENERATION_IDENTITY_DIGEST_POLICY,
    }
    identity["identity_sha256"] = _digest(identity)
    return identity


def _validate_generation_identity(
    identity: object,
    *,
    source_git_sha: str,
    config_contract_sha256: str,
    seed_contract_sha256: str,
) -> dict[str, object]:
    parsed = _object(identity, field="checkpoint.generation_identity")
    _exact_keys(
        parsed,
        _GENERATION_IDENTITY_KEYS,
        field="checkpoint.generation_identity",
    )
    if parsed["schema_version"] != OPEN_ECOLOGY_GENERATION_IDENTITY_SCHEMA_VERSION:
        raise OpenEcologyCheckpointError("generation identity schema is stale")
    if parsed["digest_policy"] != OPEN_ECOLOGY_GENERATION_IDENTITY_DIGEST_POLICY:
        raise OpenEcologyCheckpointError("generation identity digest policy mismatch")
    _identifier(
        parsed["run_generation_id"],
        field="generation_identity.run_generation_id",
    )
    _identifier(parsed["island_id"], field="generation_identity.island_id")
    _nonnegative_int(
        parsed["generation_index"],
        field="generation_identity.generation_index",
    )
    if parsed["source_git_sha"] != source_git_sha:
        raise OpenEcologyCheckpointError("generation identity source git SHA mismatch")
    if parsed["config_contract_sha256"] != config_contract_sha256:
        raise OpenEcologyCheckpointError(
            "generation identity config contract SHA256 mismatch"
        )
    if parsed["seed_contract_sha256"] != seed_contract_sha256:
        raise OpenEcologyCheckpointError(
            "generation identity seed contract SHA256 mismatch"
        )
    observed_sha256 = _sha256(
        parsed["identity_sha256"],
        field="generation_identity.identity_sha256",
    )
    expected_sha256 = _digest(
        {key: value for key, value in parsed.items() if key != "identity_sha256"}
    )
    if observed_sha256 != expected_sha256:
        raise OpenEcologyCheckpointError("generation identity SHA256 mismatch")
    return dict(parsed)


def _restartability_payload(
    components: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    validated = [
        component_name
        for component_name in REQUIRED_CHECKPOINT_COMPONENTS
        if bool(components[component_name]["present"])
    ]
    missing = [
        component_name
        for component_name in REQUIRED_CHECKPOINT_COMPONENTS
        if component_name not in validated
    ]
    return {
        "restartable": not missing,
        "required_components": list(REQUIRED_CHECKPOINT_COMPONENTS),
        "validated_components": validated,
        "missing_components": missing,
        "policy": _RESTARTABILITY_POLICY,
    }


def _read_bounded_regular_file(path: Path, byte_ceiling: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OpenEcologyCheckpointError(
            f"cannot safely open checkpoint: {error}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OpenEcologyCheckpointError("checkpoint must be a regular file")
        if file_stat.st_size > byte_ceiling:
            raise OpenEcologyCheckpointError(
                "checkpoint exceeds max_checkpoint_bytes "
                f"({file_stat.st_size} > {byte_ceiling})"
            )
        chunks: list[bytes] = []
        remaining = byte_ceiling + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > byte_ceiling:
            raise OpenEcologyCheckpointError(
                "checkpoint exceeds max_checkpoint_bytes while reading"
            )
        return encoded
    finally:
        os.close(descriptor)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologyCheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise OpenEcologyCheckpointError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_clone(value: object, *, field: str) -> dict[str, object]:
    _validate_json_value(value, field=field, depth=0)
    if type(value) is not dict:
        raise OpenEcologyCheckpointError(f"{field} must be an object")
    return deepcopy(value)


def _canonical_bytes(value: object) -> bytes:
    _validate_json_value(value, field="value", depth=0)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise OpenEcologyCheckpointError(
            f"value is not canonical JSON: {error}"
        ) from error


def _validate_json_value(value: object, *, field: str, depth: int) -> None:
    # This validation is on the checkpoint hot path and may visit millions of
    # scalar tensor values.  Keep one bounded iterative walk rather than
    # recursively constructing a new diagnostic path string for every scalar.
    # The accepted type contract is unchanged; the root field still identifies
    # the offending value without turning validation into quadratic allocation.
    pending: list[tuple[object, int]] = [(value, depth)]
    while pending:
        current, current_depth = pending.pop()
        if current_depth > _MAX_JSON_NESTING_DEPTH:
            raise OpenEcologyCheckpointError(
                f"{field} exceeds maximum JSON nesting depth"
            )
        current_type = type(current)
        if current is None or current_type in (bool, int, str):
            continue
        if current_type is float:
            if not math.isfinite(current):
                raise OpenEcologyCheckpointError(f"{field} contains a non-finite float")
            continue
        if current_type is list:
            child_depth = current_depth + 1
            pending.extend((item, child_depth) for item in current)
            continue
        if current_type is dict:
            child_depth = current_depth + 1
            for key, item in current.items():
                if not isinstance(key, str):
                    raise OpenEcologyCheckpointError(
                        f"{field} contains a non-string object key"
                    )
                pending.append((item, child_depth))
            continue
        raise OpenEcologyCheckpointError(
            f"{field} contains unsupported JSON type {current_type.__name__}"
        )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OpenEcologyCheckpointError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise OpenEcologyCheckpointError(
            f"{field} keys mismatch; missing={missing}, extra={extra}"
        )


def _schema_version(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SCHEMA_VERSION_RE.fullmatch(value):
        raise OpenEcologyCheckpointError(f"{field} is not a valid schema version")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise OpenEcologyCheckpointError(f"{field} is not a valid identifier")
    return value


def _git_sha(value: object) -> str:
    if not isinstance(value, str) or not _GIT_SHA_RE.fullmatch(value):
        raise OpenEcologyCheckpointError(
            "source_git_sha must be an exact lowercase 40-hex commit SHA"
        )
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise OpenEcologyCheckpointError(f"{field} must be a lowercase SHA256")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyCheckpointError(f"{field} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyCheckpointError(f"{field} must be a positive integer")
    return value


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
