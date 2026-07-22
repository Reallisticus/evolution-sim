from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor

from evolution_sim.env.runtime.action_contract import (
    ACTION_MASK_CONTRACT_VERSION,
    ACTION_NAMES,
)
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
)
from evolution_sim.mind.recurrent_actor_critic import (
    ACTION_COUNT,
    LEARNED_ENCODER_INPUT_SIZE,
    PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    PUBLIC_INPUT_SIZE,
    RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
    recurrent_actor_critic_contract,
)


RECURRENT_ARTIFACT_SCHEMA_VERSION = "mind_public_recurrent_actor_critic_artifact_v1"
RECURRENT_TENSOR_ENCODING = "base64_raw"
RECURRENT_TENSOR_DTYPE = "float32_le"
RECURRENT_MODEL_DIGEST_POLICY = "length_prefixed_tensor_records_and_bytes_v1"
RECURRENT_ARTIFACT_DIGEST_POLICY = "canonical_json_without_artifact_sha256_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "serialization",
        "model",
        "provenance",
        "tensors",
        "artifact_sha256",
    }
)
_SERIALIZATION_KEYS = frozenset(
    {
        "format",
        "tensor_encoding",
        "tensor_dtype",
        "tensor_byte_order",
        "tensor_layout",
        "tensor_count",
        "whole_model_digest_policy",
        "whole_model_sha256",
        "artifact_digest_policy",
    }
)
_MODEL_KEYS = frozenset(
    {
        "contract_version",
        "config",
        "public_input_schema_version",
        "public_input_size",
        "previous_public_feedback_schema_version",
        "previous_public_feedback_size",
        "learned_encoder_input_size",
        "action_mask_contract_version",
        "action_ordering",
        "action_count",
        "architecture_contract",
        "runtime_integrated",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "training_config",
        "seed_registry_digest",
        "source_commit",
        "data_metadata",
        "run_metadata",
        "learner_seed",
        "learner_device",
    }
)
_TENSOR_KEYS = frozenset(
    {"name", "shape", "dtype", "encoding", "byte_length", "sha256", "data"}
)
_CONFIG_KEYS = frozenset({"encoder_size", "hidden_size", "recurrent_layers"})


class RecurrentArtifactError(ValueError):
    """Raised when a recurrent artifact fails its fail-closed contract."""


@dataclass(frozen=True, slots=True)
class LoadedRecurrentArtifact:
    artifact: dict[str, object]
    model: PublicRecurrentActorCritic


@dataclass(frozen=True, slots=True)
class _ValidatedArtifact:
    config: RecurrentActorCriticConfig
    state_dict: dict[str, Tensor]


def build_recurrent_artifact(
    model: PublicRecurrentActorCritic,
    *,
    training_config: Mapping[str, object],
    seed_registry_digest: str,
    source_commit: str,
    data_metadata: Mapping[str, object],
    run_metadata: Mapping[str, object],
    learner_seed: int,
    learner_device: str,
) -> dict[str, object]:
    """Build a deterministic JSON-safe artifact without pickle or code objects."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise RecurrentArtifactError("model must be a PublicRecurrentActorCritic")
    config = model.config
    safe_training_config = _json_mapping_copy(
        training_config,
        field="training_config",
    )
    safe_data_metadata = _json_mapping_copy(
        data_metadata,
        field="data_metadata",
    )
    safe_run_metadata = _json_mapping_copy(
        run_metadata,
        field="run_metadata",
    )
    _validate_sha256(seed_registry_digest, field="seed_registry_digest")
    _validate_source_commit(source_commit)
    parsed_seed = _validate_seed(learner_seed)
    parsed_device = _validate_device_label(learner_device)

    tensor_records = [
        _encode_tensor(name, tensor)
        for name, tensor in sorted(model.state_dict().items())
    ]
    whole_model_sha256 = _whole_model_sha256(tensor_records)
    artifact: dict[str, object] = {
        "schema_version": RECURRENT_ARTIFACT_SCHEMA_VERSION,
        "serialization": {
            "format": "json_tensor_artifact_v1",
            "tensor_encoding": RECURRENT_TENSOR_ENCODING,
            "tensor_dtype": RECURRENT_TENSOR_DTYPE,
            "tensor_byte_order": "little",
            "tensor_layout": "contiguous_c_order",
            "tensor_count": len(tensor_records),
            "whole_model_digest_policy": RECURRENT_MODEL_DIGEST_POLICY,
            "whole_model_sha256": whole_model_sha256,
            "artifact_digest_policy": RECURRENT_ARTIFACT_DIGEST_POLICY,
        },
        "model": {
            "contract_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
            "config": asdict(config),
            "public_input_schema_version": ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
            "public_input_size": PUBLIC_INPUT_SIZE,
            "previous_public_feedback_schema_version": (
                PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
            ),
            "previous_public_feedback_size": PREVIOUS_PUBLIC_FEEDBACK_SIZE,
            "learned_encoder_input_size": LEARNED_ENCODER_INPUT_SIZE,
            "action_mask_contract_version": ACTION_MASK_CONTRACT_VERSION,
            "action_ordering": list(ACTION_NAMES),
            "action_count": ACTION_COUNT,
            "architecture_contract": recurrent_actor_critic_contract(config),
            "runtime_integrated": False,
        },
        "provenance": {
            "training_config": safe_training_config,
            "seed_registry_digest": seed_registry_digest,
            "source_commit": source_commit,
            "data_metadata": safe_data_metadata,
            "run_metadata": safe_run_metadata,
            "learner_seed": parsed_seed,
            "learner_device": parsed_device,
        },
        "tensors": tensor_records,
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    validate_recurrent_artifact(artifact)
    return artifact


def validate_recurrent_artifact(artifact: Mapping[str, object]) -> None:
    _validate_and_decode(artifact)


def model_from_recurrent_artifact(
    artifact: Mapping[str, object],
) -> PublicRecurrentActorCritic:
    """Reconstruct a strict CPU/float32 inference model from verified tensors."""

    validated = _validate_and_decode(artifact)
    model = PublicRecurrentActorCritic(
        validated.config,
        initialization_seed=0,
    ).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(validated.state_dict, strict=True)
    model.eval()
    return model


def write_recurrent_artifact(
    path: str | Path,
    artifact: Mapping[str, object],
) -> Path:
    """Validate and atomically replace one artifact in its destination directory."""

    validate_recurrent_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        temp_path = None
        _fsync_directory(destination.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return destination


def save_recurrent_artifact(
    path: str | Path,
    model: PublicRecurrentActorCritic,
    *,
    training_config: Mapping[str, object],
    seed_registry_digest: str,
    source_commit: str,
    data_metadata: Mapping[str, object],
    run_metadata: Mapping[str, object],
    learner_seed: int,
    learner_device: str,
) -> dict[str, object]:
    artifact = build_recurrent_artifact(
        model,
        training_config=training_config,
        seed_registry_digest=seed_registry_digest,
        source_commit=source_commit,
        data_metadata=data_metadata,
        run_metadata=run_metadata,
        learner_seed=learner_seed,
        learner_device=learner_device,
    )
    write_recurrent_artifact(path, artifact)
    return artifact


def load_recurrent_artifact(path: str | Path) -> LoadedRecurrentArtifact:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrentArtifactError(
            f"failed to read recurrent artifact: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RecurrentArtifactError("artifact root must be a JSON object")
    model = model_from_recurrent_artifact(payload)
    return LoadedRecurrentArtifact(artifact=payload, model=model)


def _validate_and_decode(
    artifact: Mapping[str, object],
) -> _ValidatedArtifact:
    if not isinstance(artifact, Mapping):
        raise RecurrentArtifactError("artifact must be a mapping")
    _require_exact_keys(artifact, _TOP_LEVEL_KEYS, field="artifact")
    if artifact.get("schema_version") != RECURRENT_ARTIFACT_SCHEMA_VERSION:
        raise RecurrentArtifactError("artifact schema_version is missing or stale")

    serialization = _required_mapping(artifact.get("serialization"), "serialization")
    _require_exact_keys(serialization, _SERIALIZATION_KEYS, field="serialization")
    expected_serialization = {
        "format": "json_tensor_artifact_v1",
        "tensor_encoding": RECURRENT_TENSOR_ENCODING,
        "tensor_dtype": RECURRENT_TENSOR_DTYPE,
        "tensor_byte_order": "little",
        "tensor_layout": "contiguous_c_order",
        "whole_model_digest_policy": RECURRENT_MODEL_DIGEST_POLICY,
        "artifact_digest_policy": RECURRENT_ARTIFACT_DIGEST_POLICY,
    }
    for key, expected in expected_serialization.items():
        if serialization.get(key) != expected:
            raise RecurrentArtifactError(f"serialization.{key} is missing or stale")

    model_payload = _required_mapping(artifact.get("model"), "model")
    _require_exact_keys(model_payload, _MODEL_KEYS, field="model")
    config_payload = _required_mapping(model_payload.get("config"), "model.config")
    _require_exact_keys(config_payload, _CONFIG_KEYS, field="model.config")
    try:
        config = RecurrentActorCriticConfig(
            encoder_size=_strict_int(
                config_payload.get("encoder_size"), "encoder_size"
            ),
            hidden_size=_strict_int(config_payload.get("hidden_size"), "hidden_size"),
            recurrent_layers=_strict_int(
                config_payload.get("recurrent_layers"),
                "recurrent_layers",
            ),
        )
    except ValueError as exc:
        raise RecurrentArtifactError(f"invalid model config: {exc}") from exc
    expected_model = {
        "contract_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        "config": asdict(config),
        "public_input_schema_version": ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        "public_input_size": PUBLIC_INPUT_SIZE,
        "previous_public_feedback_schema_version": (
            PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
        ),
        "previous_public_feedback_size": PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        "learned_encoder_input_size": LEARNED_ENCODER_INPUT_SIZE,
        "action_mask_contract_version": ACTION_MASK_CONTRACT_VERSION,
        "action_ordering": list(ACTION_NAMES),
        "action_count": ACTION_COUNT,
        "architecture_contract": recurrent_actor_critic_contract(config),
        "runtime_integrated": False,
    }
    if dict(model_payload) != expected_model:
        raise RecurrentArtifactError(
            "model contract, public inputs, mask contract, or action ordering drifted"
        )

    provenance = _required_mapping(artifact.get("provenance"), "provenance")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, field="provenance")
    _json_mapping_copy(provenance.get("training_config"), field="training_config")
    _json_mapping_copy(provenance.get("data_metadata"), field="data_metadata")
    _json_mapping_copy(provenance.get("run_metadata"), field="run_metadata")
    _validate_sha256(
        provenance.get("seed_registry_digest"),
        field="seed_registry_digest",
    )
    _validate_source_commit(provenance.get("source_commit"))
    _validate_seed(provenance.get("learner_seed"))
    _validate_device_label(provenance.get("learner_device"))

    tensor_payloads = artifact.get("tensors")
    if not isinstance(tensor_payloads, list):
        raise RecurrentArtifactError("tensors must be a JSON array")
    if serialization.get("tensor_count") != len(tensor_payloads):
        raise RecurrentArtifactError(
            "serialization.tensor_count does not match tensors"
        )
    expected_model_for_shapes = PublicRecurrentActorCritic(
        config,
        initialization_seed=0,
    ).to(device="cpu", dtype=torch.float32)
    expected_state = expected_model_for_shapes.state_dict()
    expected_names = sorted(expected_state)
    observed_names: list[str] = []
    state_dict: dict[str, Tensor] = {}
    validated_records: list[dict[str, object]] = []
    for index, value in enumerate(tensor_payloads):
        record = _required_mapping(value, f"tensors[{index}]")
        _require_exact_keys(record, _TENSOR_KEYS, field=f"tensors[{index}]")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise RecurrentArtifactError(f"tensors[{index}].name must be non-empty")
        observed_names.append(name)
        if name not in expected_state:
            raise RecurrentArtifactError(f"unexpected tensor name: {name}")
        expected_shape = list(expected_state[name].shape)
        shape = record.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        ):
            raise RecurrentArtifactError(f"tensor {name} has an invalid shape")
        if shape != expected_shape:
            raise RecurrentArtifactError(
                f"tensor {name} shape {shape} does not match {expected_shape}"
            )
        if record.get("dtype") != RECURRENT_TENSOR_DTYPE:
            raise RecurrentArtifactError(f"tensor {name} dtype must be float32_le")
        if record.get("encoding") != RECURRENT_TENSOR_ENCODING:
            raise RecurrentArtifactError(f"tensor {name} encoding is missing or stale")
        expected_byte_length = math.prod(shape) * 4
        if record.get("byte_length") != expected_byte_length:
            raise RecurrentArtifactError(
                f"tensor {name} byte_length does not match shape"
            )
        raw = _decode_tensor_bytes(record, name=name)
        values = np.frombuffer(raw, dtype="<f4")
        if values.size != math.prod(shape):
            raise RecurrentArtifactError(
                f"tensor {name} element count does not match shape"
            )
        if not bool(np.isfinite(values).all()):
            raise RecurrentArtifactError(f"tensor {name} contains a non-finite value")
        copied = np.array(values, dtype=np.float32, copy=True).reshape(shape)
        state_dict[name] = torch.from_numpy(copied)
        validated_records.append(dict(record))
    if observed_names != expected_names:
        raise RecurrentArtifactError(
            "tensor names must exactly match the sorted model state_dict"
        )
    observed_whole_model_sha256 = serialization.get("whole_model_sha256")
    _validate_sha256(
        observed_whole_model_sha256,
        field="serialization.whole_model_sha256",
    )
    if observed_whole_model_sha256 != _whole_model_sha256(validated_records):
        raise RecurrentArtifactError("whole-model SHA256 mismatch")

    observed_artifact_sha256 = artifact.get("artifact_sha256")
    _validate_sha256(observed_artifact_sha256, field="artifact_sha256")
    if observed_artifact_sha256 != _artifact_sha256(artifact):
        raise RecurrentArtifactError("artifact SHA256 mismatch")
    return _ValidatedArtifact(config=config, state_dict=state_dict)


def _encode_tensor(name: str, tensor: Tensor) -> dict[str, object]:
    if not isinstance(tensor, Tensor):
        raise RecurrentArtifactError(f"state_dict value {name} is not a tensor")
    if tensor.dtype != torch.float32:
        raise RecurrentArtifactError(
            f"state_dict tensor {name} must be float32 before serialization"
        )
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(cpu_tensor).all().item()):
        raise RecurrentArtifactError(f"state_dict tensor {name} is non-finite")
    raw = cpu_tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
    return {
        "name": name,
        "shape": list(cpu_tensor.shape),
        "dtype": RECURRENT_TENSOR_DTYPE,
        "encoding": RECURRENT_TENSOR_ENCODING,
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _decode_tensor_bytes(record: Mapping[str, object], *, name: str) -> bytes:
    data = record.get("data")
    if not isinstance(data, str):
        raise RecurrentArtifactError(f"tensor {name} data must be base64 text")
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise RecurrentArtifactError(f"tensor {name} data is invalid base64") from exc
    if len(raw) != record.get("byte_length"):
        raise RecurrentArtifactError(f"tensor {name} decoded byte length mismatch")
    observed_sha256 = record.get("sha256")
    _validate_sha256(observed_sha256, field=f"tensor {name} sha256")
    if hashlib.sha256(raw).hexdigest() != observed_sha256:
        raise RecurrentArtifactError(f"tensor {name} SHA256 mismatch")
    return raw


def _whole_model_sha256(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        name = str(record.get("name") or "")
        raw = _decode_tensor_bytes(record, name=name)
        descriptor = {
            key: record[key]
            for key in ("name", "shape", "dtype", "encoding", "byte_length", "sha256")
        }
        descriptor_bytes = _canonical_json_bytes(descriptor)
        digest.update(len(descriptor_bytes).to_bytes(8, "big"))
        digest.update(descriptor_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _artifact_sha256(artifact: Mapping[str, object]) -> str:
    without_digest = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    return hashlib.sha256(_canonical_json_bytes(without_digest)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecurrentArtifactError(f"value is not canonical JSON: {exc}") from exc


def _json_mapping_copy(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentArtifactError(f"{field} must be a mapping")
    encoded = _canonical_json_bytes(value)
    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RecurrentArtifactError(f"{field} must encode as a JSON object")
    return decoded


def _required_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentArtifactError(f"{field} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise RecurrentArtifactError(
            f"{field} keys do not match schema; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentArtifactError(f"{field} must be an integer")
    return value


def _validate_seed(value: object) -> int:
    parsed = _strict_int(value, "learner_seed")
    if parsed < 0 or parsed > (2**63 - 1):
        raise RecurrentArtifactError("learner_seed must be in [0, 2**63 - 1]")
    return parsed


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RecurrentArtifactError(f"{field} must be a lowercase SHA256 digest")
    return value


def _validate_source_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise RecurrentArtifactError(
            "source_commit must be a lowercase 40- or 64-character Git object id"
        )
    return value


def _validate_device_label(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentArtifactError(
            "learner_device must be a non-empty trimmed string"
        )
    if len(value) > 128 or any(ord(character) < 32 for character in value):
        raise RecurrentArtifactError("learner_device contains invalid characters")
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RecurrentArtifactError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise RecurrentArtifactError(f"non-finite JSON constant is forbidden: {value}")


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "RECURRENT_ARTIFACT_SCHEMA_VERSION",
    "RECURRENT_TENSOR_DTYPE",
    "RECURRENT_TENSOR_ENCODING",
    "LoadedRecurrentArtifact",
    "RecurrentArtifactError",
    "build_recurrent_artifact",
    "load_recurrent_artifact",
    "model_from_recurrent_artifact",
    "save_recurrent_artifact",
    "validate_recurrent_artifact",
    "write_recurrent_artifact",
]
