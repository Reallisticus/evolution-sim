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
from evolution_sim.mind.recurrent_actor_critic import (
    ACTION_COUNT,
    CRITIC_GENOME_CONDITIONING_NONE,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    GENOME_CONDITIONING_DISABLED,
    PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
    recurrent_actor_critic_contract,
)
from evolution_sim.mind.recurrent_genome import RECURRENT_CONTROLLER_GENOME_SIZE


RECURRENT_ARTIFACT_SCHEMA_VERSION = "mind_public_recurrent_actor_critic_artifact_v4"
FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION = (
    "mind_public_recurrent_frozen_policy_artifact_v5"
)
FROZEN_RECURRENT_POLICY_ARTIFACT_KIND = "frozen_recurrent_policy"
RECURRENT_REPLAY_PROBE_CONTRACT_VERSION = "mind_public_recurrent_cpu_replay_probe_v3"
FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION = (
    "mind_public_recurrent_full_world_replay_manifest_v1"
)
RECURRENT_TRAINING_CRASH_CHECKPOINT_SCHEMA_VERSION = (
    "mind_public_recurrent_training_crash_checkpoint_v4"
)
RECURRENT_TRAINING_CRASH_CHECKPOINT_KIND = "optimizer_rng_crash_checkpoint"
RECURRENT_TENSOR_ENCODING = "base64_raw"
RECURRENT_TENSOR_DTYPE = "float32_le"
RECURRENT_MODEL_DIGEST_POLICY = "length_prefixed_tensor_records_and_bytes_v1"
RECURRENT_ARTIFACT_DIGEST_POLICY = "canonical_json_without_artifact_sha256_v2"
FROZEN_RECURRENT_POLICY_ARTIFACT_DIGEST_POLICY = (
    "canonical_json_without_artifact_sha256_v3"
)
RECURRENT_CRASH_CHECKPOINT_DIGEST_POLICY = "canonical_json_without_checkpoint_sha256_v2"
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
_CONFIG_KEYS = frozenset(
    {
        "encoder_size",
        "hidden_size",
        "recurrent_layers",
        "public_input_schema_version",
        "public_input_size",
        "genome_conditioning_mode",
        "critic_genome_conditioning",
        "value_shared_trunk_gradient",
    }
)
_FROZEN_POLICY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "serialization",
        "model",
        "provenance",
        "integrity",
        "verification",
        "tensors",
        "artifact_sha256",
    }
)
_FROZEN_POLICY_PROVENANCE_KEYS = frozenset(
    {
        *_PROVENANCE_KEYS,
        "experiment_config",
        "source_manifest_sha256",
    }
)
_FROZEN_POLICY_INTEGRITY_KEYS = frozenset(
    {
        "parameters_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "experiment_config_sha256",
        "configuration_sha256",
        "source_commit_sha256",
        "source_manifest_sha256",
        "seed_registry_digest",
    }
)
_FROZEN_POLICY_VERIFICATION_KEYS = frozenset(
    {
        "replay_probe_contract_version",
        "verified_on_device",
        "probe",
        "probe_input_sha256",
        "probe_output_sha256",
        "full_world_replay_manifest",
        "full_world_replay_manifest_metadata_sha256",
    }
)
_REPLAY_PROBE_KEYS = frozenset(
    {
        "deterministic",
        "observations",
        "action_masks",
        "previous_feedback",
        "genome_values",
        "zero_genome_evidence",
        "expected_raw_logits",
        "expected_values",
        "expected_actions",
        "expected_next_state",
    }
)
_ZERO_GENOME_EVIDENCE_KEYS = frozenset(
    {
        "genome_values",
        "expected_raw_logits",
        "expected_values",
        "expected_actions",
        "expected_next_state",
        "neutral_against_disabled_path",
    }
)
_GENOME_FILM_BUFFER_NAMES = frozenset(
    {
        "_genome_film_scale_coefficients",
        "_genome_film_bias_coefficients",
    }
)
_CRITIC_GENOME_FILM_PARAMETER_NAMES = frozenset(
    {
        "critic_genome_film_scale_coefficients",
        "critic_genome_film_bias_coefficients",
    }
)
_FULL_WORLD_REPLAY_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_sha256",
        "replay_engine_contract_sha256",
        "environment_seed_registry_sha256",
        "environment_seed_roles",
        "scenario_names",
        "tick_horizons",
        "world_count",
        "replay_verified_world_count",
        "policy_sampling_stream_count",
        "all_replays_exact",
        "verification_runner",
        "verification_runner_sha256",
    }
)
_CRASH_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_kind",
        "runtime_policy_eligible",
        "resumable_training_state",
        "digest_policy",
        "source",
        "configuration",
        "progress",
        "model",
        "model_tensors",
        "parameters_sha256",
        "optimizer_state",
        "optimizer_state_sha256",
        "rng_state",
        "rng_state_sha256",
        "checkpoint_sha256",
    }
)


class RecurrentArtifactError(ValueError):
    """Raised when a recurrent artifact fails its fail-closed contract."""


@dataclass(frozen=True, slots=True)
class LoadedRecurrentArtifact:
    artifact: dict[str, object]
    model: PublicRecurrentActorCritic


@dataclass(frozen=True, slots=True)
class LoadedFrozenRecurrentPolicyArtifact:
    artifact: dict[str, object]
    model: PublicRecurrentActorCritic


@dataclass(frozen=True, slots=True)
class LoadedRecurrentTrainingCrashCheckpoint:
    checkpoint: dict[str, object]
    model: PublicRecurrentActorCritic
    optimizer_state: object
    rng_state: object


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
            "format": "json_tensor_artifact_v2",
            "tensor_encoding": RECURRENT_TENSOR_ENCODING,
            "tensor_dtype": RECURRENT_TENSOR_DTYPE,
            "tensor_byte_order": "little",
            "tensor_layout": "contiguous_c_order",
            "tensor_count": len(tensor_records),
            "whole_model_digest_policy": RECURRENT_MODEL_DIGEST_POLICY,
            "whole_model_sha256": whole_model_sha256,
            "artifact_digest_policy": RECURRENT_ARTIFACT_DIGEST_POLICY,
        },
        "model": _model_contract_payload(config),
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


def build_frozen_recurrent_policy_artifact(
    model: PublicRecurrentActorCritic,
    *,
    training_config: Mapping[str, object],
    experiment_config: Mapping[str, object],
    seed_registry_digest: str,
    source_commit: str,
    source_manifest_sha256: str,
    data_metadata: Mapping[str, object],
    run_metadata: Mapping[str, object],
    full_world_replay_manifest: Mapping[str, object],
    learner_seed: int,
    learner_device: str,
) -> dict[str, object]:
    """Build a frozen policy candidate with executable CPU replay evidence.

    This is deliberately a different schema from resumable optimizer/RNG
    checkpoints.  It contains only inference parameters plus provenance and
    verification evidence required to replay the frozen policy.
    """

    legacy = build_recurrent_artifact(
        model,
        training_config=training_config,
        seed_registry_digest=seed_registry_digest,
        source_commit=source_commit,
        data_metadata=data_metadata,
        run_metadata=run_metadata,
        learner_seed=learner_seed,
        learner_device=learner_device,
    )
    safe_training_config = _json_mapping_copy(training_config, field="training_config")
    safe_experiment_config = _json_mapping_copy(
        experiment_config, field="experiment_config"
    )
    safe_manifest = _validated_full_world_replay_manifest(full_world_replay_manifest)
    parsed_source_commit = _validate_source_commit(source_commit)
    parsed_source_manifest = _validate_sha256(
        source_manifest_sha256,
        field="source_manifest_sha256",
    )
    cpu_model = model_from_recurrent_artifact(legacy)
    probe = _build_cpu_replay_probe(cpu_model)

    serialization = dict(legacy["serialization"])
    serialization["format"] = "json_frozen_policy_tensor_artifact_v3"
    serialization["artifact_digest_policy"] = (
        FROZEN_RECURRENT_POLICY_ARTIFACT_DIGEST_POLICY
    )
    model_payload = dict(legacy["model"])
    provenance = dict(legacy["provenance"])
    provenance["experiment_config"] = safe_experiment_config
    provenance["source_manifest_sha256"] = parsed_source_manifest

    model_config_sha256 = _json_sha256(model_payload["config"])
    training_config_sha256 = _json_sha256(safe_training_config)
    experiment_config_sha256 = _json_sha256(safe_experiment_config)
    configuration_sha256 = _json_sha256(
        {
            "model_config_sha256": model_config_sha256,
            "training_config_sha256": training_config_sha256,
            "experiment_config_sha256": experiment_config_sha256,
        }
    )
    parameters_sha256 = str(serialization["whole_model_sha256"])
    integrity = {
        "parameters_sha256": parameters_sha256,
        "model_config_sha256": model_config_sha256,
        "training_config_sha256": training_config_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "configuration_sha256": configuration_sha256,
        "source_commit_sha256": hashlib.sha256(
            parsed_source_commit.encode("ascii")
        ).hexdigest(),
        "source_manifest_sha256": parsed_source_manifest,
        "seed_registry_digest": seed_registry_digest,
    }
    verification = {
        "replay_probe_contract_version": RECURRENT_REPLAY_PROBE_CONTRACT_VERSION,
        "verified_on_device": "cpu",
        "probe": probe,
        "probe_input_sha256": _replay_probe_input_sha256(probe),
        "probe_output_sha256": _replay_probe_output_sha256(probe),
        "full_world_replay_manifest": safe_manifest,
        "full_world_replay_manifest_metadata_sha256": _json_sha256(safe_manifest),
    }
    artifact: dict[str, object] = {
        "schema_version": FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": FROZEN_RECURRENT_POLICY_ARTIFACT_KIND,
        "serialization": serialization,
        "model": model_payload,
        "provenance": provenance,
        "integrity": integrity,
        "verification": verification,
        "tensors": legacy["tensors"],
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    validate_frozen_recurrent_policy_artifact(artifact)
    return artifact


def validate_frozen_recurrent_policy_artifact(
    artifact: Mapping[str, object],
) -> None:
    _validate_frozen_policy_and_decode(artifact)


def model_from_frozen_recurrent_policy_artifact(
    artifact: Mapping[str, object],
) -> PublicRecurrentActorCritic:
    validated = _validate_frozen_policy_and_decode(artifact)
    model = PublicRecurrentActorCritic(
        validated.config,
        initialization_seed=0,
    ).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(validated.state_dict, strict=True)
    model.eval()
    return model


def write_frozen_recurrent_policy_artifact(
    path: str | Path,
    artifact: Mapping[str, object],
) -> Path:
    validate_frozen_recurrent_policy_artifact(artifact)
    return _write_atomic_json(path, artifact)


def save_frozen_recurrent_policy_artifact(
    path: str | Path,
    model: PublicRecurrentActorCritic,
    **metadata: object,
) -> dict[str, object]:
    artifact = build_frozen_recurrent_policy_artifact(model, **metadata)
    write_frozen_recurrent_policy_artifact(path, artifact)
    return artifact


def load_frozen_recurrent_policy_artifact(
    path: str | Path,
) -> LoadedFrozenRecurrentPolicyArtifact:
    payload = _read_strict_json_mapping(path, label="frozen recurrent policy artifact")
    model = model_from_frozen_recurrent_policy_artifact(payload)
    return LoadedFrozenRecurrentPolicyArtifact(artifact=payload, model=model)


def build_recurrent_training_crash_checkpoint(
    model: PublicRecurrentActorCritic,
    *,
    optimizer_state: Mapping[object, object],
    rng_state: Mapping[object, object],
    optimizer_type: str,
    training_config: Mapping[str, object],
    seed_registry_digest: str,
    source_commit: str,
    source_manifest_sha256: str,
    learner_seed: int,
    completed_updates: int,
    run_id: str,
) -> dict[str, object]:
    """Build JSON-safe resumable state that can never masquerade as a policy."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise RecurrentArtifactError("model must be a PublicRecurrentActorCritic")
    safe_training_config = _json_mapping_copy(training_config, field="training_config")
    _validate_sha256(seed_registry_digest, field="seed_registry_digest")
    parsed_source_commit = _validate_source_commit(source_commit)
    parsed_source_manifest = _validate_sha256(
        source_manifest_sha256,
        field="source_manifest_sha256",
    )
    parsed_seed = _validate_seed(learner_seed)
    parsed_optimizer_type = _validate_trimmed_text(
        optimizer_type,
        field="optimizer_type",
        maximum=256,
    )
    parsed_updates = _strict_nonnegative_int(completed_updates, "completed_updates")
    parsed_run_id = _validate_trimmed_text(run_id, field="run_id", maximum=256)
    encoded_optimizer = _encode_checkpoint_state(
        optimizer_state, field="optimizer_state"
    )
    encoded_rng = _encode_checkpoint_state(rng_state, field="rng_state")
    tensor_records = [
        _encode_tensor(name, tensor)
        for name, tensor in sorted(model.state_dict().items())
    ]
    parameters_sha256 = _whole_model_sha256(tensor_records)
    model_payload = _model_contract_payload(model.config)
    checkpoint: dict[str, object] = {
        "schema_version": RECURRENT_TRAINING_CRASH_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": RECURRENT_TRAINING_CRASH_CHECKPOINT_KIND,
        "runtime_policy_eligible": False,
        "resumable_training_state": True,
        "digest_policy": RECURRENT_CRASH_CHECKPOINT_DIGEST_POLICY,
        "source": {
            "source_commit": parsed_source_commit,
            "source_commit_sha256": hashlib.sha256(
                parsed_source_commit.encode("ascii")
            ).hexdigest(),
            "source_manifest_sha256": parsed_source_manifest,
            "seed_registry_digest": seed_registry_digest,
        },
        "configuration": {
            "model_config": asdict(model.config),
            "model_config_sha256": _json_sha256(asdict(model.config)),
            "optimizer_type": parsed_optimizer_type,
            "training_config": safe_training_config,
            "training_config_sha256": _json_sha256(safe_training_config),
        },
        "progress": {
            "learner_seed": parsed_seed,
            "completed_updates": parsed_updates,
            "run_id": parsed_run_id,
            "model_training": bool(model.training),
        },
        "model": model_payload,
        "model_tensors": tensor_records,
        "parameters_sha256": parameters_sha256,
        "optimizer_state": encoded_optimizer,
        "optimizer_state_sha256": _json_sha256(encoded_optimizer),
        "rng_state": encoded_rng,
        "rng_state_sha256": _json_sha256(encoded_rng),
    }
    checkpoint["checkpoint_sha256"] = _checkpoint_sha256(checkpoint)
    validate_recurrent_training_crash_checkpoint(checkpoint)
    return checkpoint


def validate_recurrent_training_crash_checkpoint(
    checkpoint: Mapping[str, object],
) -> None:
    _validate_crash_checkpoint_and_decode(checkpoint)


def write_recurrent_training_crash_checkpoint(
    path: str | Path,
    checkpoint: Mapping[str, object],
) -> Path:
    validate_recurrent_training_crash_checkpoint(checkpoint)
    return _write_atomic_json(path, checkpoint)


def save_recurrent_training_crash_checkpoint(
    path: str | Path,
    model: PublicRecurrentActorCritic,
    **state: object,
) -> dict[str, object]:
    checkpoint = build_recurrent_training_crash_checkpoint(model, **state)
    write_recurrent_training_crash_checkpoint(path, checkpoint)
    return checkpoint


def load_recurrent_training_crash_checkpoint(
    path: str | Path,
) -> LoadedRecurrentTrainingCrashCheckpoint:
    payload = _read_strict_json_mapping(path, label="recurrent crash checkpoint")
    validated, optimizer_state, rng_state = _validate_crash_checkpoint_and_decode(
        payload
    )
    model = PublicRecurrentActorCritic(
        validated.config,
        initialization_seed=0,
    ).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(validated.state_dict, strict=True)
    progress = _required_mapping(payload.get("progress"), "progress")
    model.train(progress.get("model_training") is True)
    return LoadedRecurrentTrainingCrashCheckpoint(
        checkpoint=payload,
        model=model,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
    )


def _validate_frozen_policy_and_decode(
    artifact: Mapping[str, object],
) -> _ValidatedArtifact:
    if not isinstance(artifact, Mapping):
        raise RecurrentArtifactError("frozen policy artifact must be a mapping")
    _require_exact_keys(
        artifact,
        _FROZEN_POLICY_TOP_LEVEL_KEYS,
        field="frozen policy artifact",
    )
    if artifact.get("schema_version") != (
        FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION
    ):
        raise RecurrentArtifactError("frozen policy schema_version is missing or stale")
    if artifact.get("artifact_kind") != FROZEN_RECURRENT_POLICY_ARTIFACT_KIND:
        raise RecurrentArtifactError("frozen policy artifact_kind is missing or stale")

    serialization = _required_mapping(artifact.get("serialization"), "serialization")
    _require_exact_keys(serialization, _SERIALIZATION_KEYS, field="serialization")
    expected_serialization = {
        "format": "json_frozen_policy_tensor_artifact_v3",
        "tensor_encoding": RECURRENT_TENSOR_ENCODING,
        "tensor_dtype": RECURRENT_TENSOR_DTYPE,
        "tensor_byte_order": "little",
        "tensor_layout": "contiguous_c_order",
        "whole_model_digest_policy": RECURRENT_MODEL_DIGEST_POLICY,
        "artifact_digest_policy": FROZEN_RECURRENT_POLICY_ARTIFACT_DIGEST_POLICY,
    }
    for key, expected in expected_serialization.items():
        if serialization.get(key) != expected:
            raise RecurrentArtifactError(f"serialization.{key} is missing or stale")

    provenance = _required_mapping(artifact.get("provenance"), "provenance")
    _require_exact_keys(
        provenance,
        _FROZEN_POLICY_PROVENANCE_KEYS,
        field="provenance",
    )
    training_config = _json_mapping_copy(
        provenance.get("training_config"), field="training_config"
    )
    experiment_config = _json_mapping_copy(
        provenance.get("experiment_config"), field="experiment_config"
    )
    _json_mapping_copy(provenance.get("data_metadata"), field="data_metadata")
    _json_mapping_copy(provenance.get("run_metadata"), field="run_metadata")
    seed_registry_digest = _validate_sha256(
        provenance.get("seed_registry_digest"), field="seed_registry_digest"
    )
    source_commit = _validate_source_commit(provenance.get("source_commit"))
    source_manifest_sha256 = _validate_sha256(
        provenance.get("source_manifest_sha256"), field="source_manifest_sha256"
    )
    _validate_seed(provenance.get("learner_seed"))
    _validate_device_label(provenance.get("learner_device"))

    validated = _validated_model_from_extended_artifact(
        model_payload=artifact.get("model"),
        serialization=serialization,
        provenance=provenance,
        tensors=artifact.get("tensors"),
    )
    model_payload = _required_mapping(artifact.get("model"), "model")
    integrity = _required_mapping(artifact.get("integrity"), "integrity")
    _require_exact_keys(integrity, _FROZEN_POLICY_INTEGRITY_KEYS, field="integrity")
    expected_integrity = {
        "parameters_sha256": serialization.get("whole_model_sha256"),
        "model_config_sha256": _json_sha256(model_payload.get("config")),
        "training_config_sha256": _json_sha256(training_config),
        "experiment_config_sha256": _json_sha256(experiment_config),
        "source_commit_sha256": hashlib.sha256(
            source_commit.encode("ascii")
        ).hexdigest(),
        "source_manifest_sha256": source_manifest_sha256,
        "seed_registry_digest": seed_registry_digest,
    }
    expected_integrity["configuration_sha256"] = _json_sha256(
        {
            "model_config_sha256": expected_integrity["model_config_sha256"],
            "training_config_sha256": expected_integrity["training_config_sha256"],
            "experiment_config_sha256": expected_integrity["experiment_config_sha256"],
        }
    )
    if dict(integrity) != expected_integrity:
        raise RecurrentArtifactError(
            "frozen policy parameter, configuration, source, or registry hashes drifted"
        )

    verification = _required_mapping(artifact.get("verification"), "verification")
    _require_exact_keys(
        verification,
        _FROZEN_POLICY_VERIFICATION_KEYS,
        field="verification",
    )
    if verification.get("replay_probe_contract_version") != (
        RECURRENT_REPLAY_PROBE_CONTRACT_VERSION
    ):
        raise RecurrentArtifactError("replay probe contract is missing or stale")
    if verification.get("verified_on_device") != "cpu":
        raise RecurrentArtifactError("frozen policy replay probe must be CPU verified")
    probe = _required_mapping(verification.get("probe"), "verification.probe")
    _require_exact_keys(probe, _REPLAY_PROBE_KEYS, field="verification.probe")
    if verification.get("probe_input_sha256") != _replay_probe_input_sha256(probe):
        raise RecurrentArtifactError("replay probe input SHA256 mismatch")
    if verification.get("probe_output_sha256") != _replay_probe_output_sha256(probe):
        raise RecurrentArtifactError("replay probe output SHA256 mismatch")
    manifest = _validated_full_world_replay_manifest(
        verification.get("full_world_replay_manifest")
    )
    if verification.get("full_world_replay_manifest_metadata_sha256") != (
        _json_sha256(manifest)
    ):
        raise RecurrentArtifactError(
            "full-world replay manifest metadata SHA256 mismatch"
        )

    probe_model = PublicRecurrentActorCritic(
        validated.config,
        initialization_seed=0,
    ).to(device="cpu", dtype=torch.float32)
    probe_model.load_state_dict(validated.state_dict, strict=True)
    probe_model.eval()
    _verify_cpu_replay_probe(probe_model, probe)

    observed_artifact_sha256 = artifact.get("artifact_sha256")
    _validate_sha256(observed_artifact_sha256, field="artifact_sha256")
    if observed_artifact_sha256 != _artifact_sha256(artifact):
        raise RecurrentArtifactError("artifact SHA256 mismatch")
    return validated


def _validated_model_from_extended_artifact(
    *,
    model_payload: object,
    serialization: Mapping[str, object],
    provenance: Mapping[str, object],
    tensors: object,
) -> _ValidatedArtifact:
    legacy_serialization = dict(serialization)
    legacy_serialization["format"] = "json_tensor_artifact_v2"
    legacy_serialization["artifact_digest_policy"] = RECURRENT_ARTIFACT_DIGEST_POLICY
    legacy_provenance = {key: provenance[key] for key in _PROVENANCE_KEYS}
    legacy: dict[str, object] = {
        "schema_version": RECURRENT_ARTIFACT_SCHEMA_VERSION,
        "serialization": legacy_serialization,
        "model": model_payload,
        "provenance": legacy_provenance,
        "tensors": tensors,
    }
    legacy["artifact_sha256"] = _artifact_sha256(legacy)
    return _validate_and_decode(legacy)


def _canonical_probe_genome_values(*, batch_size: int) -> Tensor:
    positions = torch.arange(
        batch_size * RECURRENT_CONTROLLER_GENOME_SIZE,
        dtype=torch.float32,
        device="cpu",
    ).reshape(batch_size, RECURRENT_CONTROLLER_GENOME_SIZE)
    return ((positions.remainder(31.0) - 15.0) / 15.0).contiguous()


def _disabled_genome_reference_model(
    model: PublicRecurrentActorCritic,
) -> PublicRecurrentActorCritic:
    config = model.config
    disabled_config = RecurrentActorCriticConfig(
        encoder_size=config.encoder_size,
        hidden_size=config.hidden_size,
        recurrent_layers=config.recurrent_layers,
        public_input_schema_version=config.public_input_schema_version,
        public_input_size=config.public_input_size,
        genome_conditioning_mode=GENOME_CONDITIONING_DISABLED,
        critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
        value_shared_trunk_gradient=config.value_shared_trunk_gradient,
    )
    reference = PublicRecurrentActorCritic(
        disabled_config,
        initialization_seed=0,
    ).to(device="cpu", dtype=torch.float32)
    source_state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name not in _GENOME_FILM_BUFFER_NAMES | _CRITIC_GENOME_FILM_PARAMETER_NAMES
    }
    if set(source_state) != set(reference.state_dict()):
        raise RecurrentArtifactError(
            "enabled model parameters do not match the disabled neutral reference"
        )
    reference.load_state_dict(source_state, strict=True)
    reference.eval()
    return reference


def _build_cpu_replay_probe(
    model: PublicRecurrentActorCritic,
) -> dict[str, object]:
    batch_size = 4
    public_input_size = model.config.public_input_size
    positions = torch.arange(
        batch_size * public_input_size,
        dtype=torch.float32,
        device="cpu",
    ).reshape(batch_size, public_input_size)
    observations = ((positions.remainder(257.0) - 128.0) / 128.0).contiguous()
    action_masks = torch.zeros(batch_size, ACTION_COUNT, dtype=torch.bool)
    for row in range(batch_size):
        action_masks[row, row] = True
        action_masks[row, (row * 5 + 7) % ACTION_COUNT] = True
        action_masks[row, ACTION_COUNT - 1 - row] = True
    previous_feedback = torch.zeros(
        batch_size,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        dtype=torch.float32,
    )
    genome_values: Tensor | None = None
    zero_genome_evidence: dict[str, object] | None = None
    act_kwargs: dict[str, object] = {}
    if model.config.genome_conditioning_mode == GENOME_CONDITIONING_ACTOR_FILM_V1:
        genome_values = _canonical_probe_genome_values(batch_size=batch_size)
        zero_genome_values = torch.zeros_like(genome_values)
        act_kwargs["genome_values"] = genome_values
    with torch.no_grad():
        selected = model.act(
            observations,
            action_masks,
            previous_feedback,
            deterministic=True,
            **act_kwargs,
        )
        if genome_values is not None:
            zero_selected = model.act(
                observations,
                action_masks,
                previous_feedback,
                genome_values=zero_genome_values,
                deterministic=True,
            )
            disabled_reference = _disabled_genome_reference_model(model)
            disabled_selected = disabled_reference.act(
                observations,
                action_masks,
                previous_feedback,
                deterministic=True,
            )
            for label, actual, expected in (
                (
                    "raw logits",
                    zero_selected.raw_logits,
                    disabled_selected.raw_logits,
                ),
                ("values", zero_selected.values, disabled_selected.values),
                ("actions", zero_selected.actions, disabled_selected.actions),
                (
                    "next state",
                    zero_selected.next_state,
                    disabled_selected.next_state,
                ),
            ):
                if not torch.equal(actual, expected):
                    raise RecurrentArtifactError(
                        f"zero genome is not neutral for replay probe {label}"
                    )
            zero_genome_evidence = {
                "genome_values": _encode_tensor(
                    "probe.zero_genome_values",
                    zero_genome_values,
                ),
                "expected_raw_logits": _encode_tensor(
                    "probe.zero_expected_raw_logits",
                    zero_selected.raw_logits,
                ),
                "expected_values": _encode_tensor(
                    "probe.zero_expected_values",
                    zero_selected.values,
                ),
                "expected_actions": [
                    int(value) for value in zero_selected.actions.tolist()
                ],
                "expected_next_state": _encode_tensor(
                    "probe.zero_expected_next_state",
                    zero_selected.next_state,
                ),
                "neutral_against_disabled_path": True,
            }
    return {
        "deterministic": True,
        "observations": _encode_tensor("probe.observations", observations),
        "action_masks": action_masks.tolist(),
        "previous_feedback": _encode_tensor(
            "probe.previous_feedback", previous_feedback
        ),
        "genome_values": (
            None
            if genome_values is None
            else _encode_tensor("probe.genome_values", genome_values)
        ),
        "zero_genome_evidence": zero_genome_evidence,
        "expected_raw_logits": _encode_tensor(
            "probe.expected_raw_logits", selected.raw_logits
        ),
        "expected_values": _encode_tensor("probe.expected_values", selected.values),
        "expected_actions": [int(value) for value in selected.actions.tolist()],
        "expected_next_state": _encode_tensor(
            "probe.expected_next_state", selected.next_state
        ),
    }


def _replay_probe_input_sha256(probe: Mapping[str, object]) -> str:
    zero_genome_evidence = probe.get("zero_genome_evidence")
    return _json_sha256(
        {
            "deterministic": probe.get("deterministic"),
            "observations": probe.get("observations"),
            "action_masks": probe.get("action_masks"),
            "previous_feedback": probe.get("previous_feedback"),
            "genome_values": probe.get("genome_values"),
            "zero_genome_values": (
                zero_genome_evidence.get("genome_values")
                if isinstance(zero_genome_evidence, Mapping)
                else None
            ),
        }
    )


def _replay_probe_output_sha256(probe: Mapping[str, object]) -> str:
    zero_genome_evidence = probe.get("zero_genome_evidence")
    return _json_sha256(
        {
            "expected_raw_logits": probe.get("expected_raw_logits"),
            "expected_values": probe.get("expected_values"),
            "expected_actions": probe.get("expected_actions"),
            "expected_next_state": probe.get("expected_next_state"),
            "zero_genome_evidence": (
                {
                    key: zero_genome_evidence.get(key)
                    for key in (
                        "expected_raw_logits",
                        "expected_values",
                        "expected_actions",
                        "expected_next_state",
                        "neutral_against_disabled_path",
                    )
                }
                if isinstance(zero_genome_evidence, Mapping)
                else None
            ),
        }
    )


def _verify_cpu_replay_probe(
    model: PublicRecurrentActorCritic,
    probe: Mapping[str, object],
) -> None:
    if probe.get("deterministic") is not True:
        raise RecurrentArtifactError("replay probe must be deterministic")
    observations = _decode_float32_record(
        probe.get("observations"), expected_name="probe.observations"
    )
    previous_feedback = _decode_float32_record(
        probe.get("previous_feedback"),
        expected_name="probe.previous_feedback",
    )
    genome_values: Tensor | None = None
    zero_genome_expected: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
    if model.config.genome_conditioning_mode == GENOME_CONDITIONING_ACTOR_FILM_V1:
        if probe.get("genome_values") is None:
            raise RecurrentArtifactError(
                "enabled replay probe requires explicit genome_values"
            )
        genome_values = _decode_float32_record(
            probe.get("genome_values"),
            expected_name="probe.genome_values",
        )
        expected_genomes = _canonical_probe_genome_values(
            batch_size=observations.shape[0]
        )
        if not torch.equal(genome_values, expected_genomes):
            raise RecurrentArtifactError(
                "replay probe genome_values are not the canonical nonzero probe"
            )
        if not bool((genome_values != 0.0).any().item()):
            raise RecurrentArtifactError(
                "enabled replay probe genome_values must be nonzero"
            )
        zero_evidence = _required_mapping(
            probe.get("zero_genome_evidence"),
            "verification.probe.zero_genome_evidence",
        )
        _require_exact_keys(
            zero_evidence,
            _ZERO_GENOME_EVIDENCE_KEYS,
            field="verification.probe.zero_genome_evidence",
        )
        if zero_evidence.get("neutral_against_disabled_path") is not True:
            raise RecurrentArtifactError("zero-genome replay evidence is not neutral")
        zero_genome_values = _decode_float32_record(
            zero_evidence.get("genome_values"),
            expected_name="probe.zero_genome_values",
        )
        if zero_genome_values.shape != genome_values.shape or not torch.equal(
            zero_genome_values,
            torch.zeros_like(genome_values),
        ):
            raise RecurrentArtifactError(
                "zero-genome replay evidence must contain exact zero rows"
            )
        zero_actions_value = zero_evidence.get("expected_actions")
        if not isinstance(zero_actions_value, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in zero_actions_value
        ):
            raise RecurrentArtifactError(
                "zero-genome replay expected_actions are invalid"
            )
        zero_genome_expected = (
            _decode_float32_record(
                zero_evidence.get("expected_raw_logits"),
                expected_name="probe.zero_expected_raw_logits",
            ),
            _decode_float32_record(
                zero_evidence.get("expected_values"),
                expected_name="probe.zero_expected_values",
            ),
            torch.tensor(zero_actions_value, dtype=torch.long),
            _decode_float32_record(
                zero_evidence.get("expected_next_state"),
                expected_name="probe.zero_expected_next_state",
            ),
        )
    elif (
        probe.get("genome_values") is not None
        or probe.get("zero_genome_evidence") is not None
    ):
        raise RecurrentArtifactError(
            "disabled replay probe must not contain genome evidence"
        )
    raw_logits = _decode_float32_record(
        probe.get("expected_raw_logits"),
        expected_name="probe.expected_raw_logits",
    )
    values = _decode_float32_record(
        probe.get("expected_values"),
        expected_name="probe.expected_values",
    )
    next_state = _decode_float32_record(
        probe.get("expected_next_state"),
        expected_name="probe.expected_next_state",
    )
    masks_value = probe.get("action_masks")
    if not isinstance(masks_value, list):
        raise RecurrentArtifactError("replay probe action_masks must be an array")
    if any(
        not isinstance(row, list)
        or len(row) != ACTION_COUNT
        or any(type(value) is not bool for value in row)
        for row in masks_value
    ):
        raise RecurrentArtifactError(
            "replay probe action_masks must contain fixed-width boolean rows"
        )
    try:
        action_masks = torch.tensor(masks_value, dtype=torch.bool)
    except (TypeError, ValueError) as exc:
        raise RecurrentArtifactError("replay probe action_masks are invalid") from exc
    if action_masks.shape != (observations.shape[0], ACTION_COUNT):
        raise RecurrentArtifactError("replay probe action_masks shape drifted")
    actions_value = probe.get("expected_actions")
    if not isinstance(actions_value, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in actions_value
    ):
        raise RecurrentArtifactError("replay probe expected_actions are invalid")
    actions = torch.tensor(actions_value, dtype=torch.long)
    try:
        with torch.no_grad():
            observed = model.act(
                observations,
                action_masks,
                previous_feedback,
                genome_values=genome_values,
                deterministic=True,
            )
    except (ValueError, RuntimeError) as exc:
        raise RecurrentArtifactError(
            f"CPU replay probe inputs violate the model contract: {exc}"
        ) from exc
    comparisons = (
        ("raw logits", observed.raw_logits, raw_logits),
        ("values", observed.values, values),
        ("actions", observed.actions, actions),
        ("next state", observed.next_state, next_state),
    )
    for label, actual, expected in comparisons:
        if not torch.equal(actual, expected):
            raise RecurrentArtifactError(f"CPU replay probe {label} mismatch")
    if genome_values is not None:
        assert zero_genome_expected is not None
        with torch.no_grad():
            zero_observed = model.act(
                observations,
                action_masks,
                previous_feedback,
                genome_values=torch.zeros_like(genome_values),
                deterministic=True,
            )
            disabled_observed = _disabled_genome_reference_model(model).act(
                observations,
                action_masks,
                previous_feedback,
                deterministic=True,
            )
        zero_comparisons = (
            (
                "zero-genome raw logits",
                zero_observed.raw_logits,
                zero_genome_expected[0],
            ),
            (
                "zero-genome values",
                zero_observed.values,
                zero_genome_expected[1],
            ),
            (
                "zero-genome actions",
                zero_observed.actions,
                zero_genome_expected[2],
            ),
            (
                "zero-genome next state",
                zero_observed.next_state,
                zero_genome_expected[3],
            ),
            (
                "zero-genome neutral raw logits",
                zero_observed.raw_logits,
                disabled_observed.raw_logits,
            ),
            (
                "zero-genome neutral values",
                zero_observed.values,
                disabled_observed.values,
            ),
            (
                "zero-genome neutral actions",
                zero_observed.actions,
                disabled_observed.actions,
            ),
            (
                "zero-genome neutral next state",
                zero_observed.next_state,
                disabled_observed.next_state,
            ),
        )
        for label, actual, expected in zero_comparisons:
            if not torch.equal(actual, expected):
                raise RecurrentArtifactError(f"CPU replay probe {label} mismatch")


def _validated_full_world_replay_manifest(value: object) -> dict[str, object]:
    manifest = _json_mapping_copy(value, field="full_world_replay_manifest")
    _require_exact_keys(
        manifest,
        _FULL_WORLD_REPLAY_MANIFEST_KEYS,
        field="full_world_replay_manifest",
    )
    if manifest.get("schema_version") != FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION:
        raise RecurrentArtifactError(
            "full-world replay manifest schema is missing or stale"
        )
    for field in (
        "manifest_sha256",
        "replay_engine_contract_sha256",
        "environment_seed_registry_sha256",
        "verification_runner_sha256",
    ):
        _validate_sha256(
            manifest.get(field), field=f"full_world_replay_manifest.{field}"
        )
    roles = _validated_text_list(
        manifest.get("environment_seed_roles"),
        field="full_world_replay_manifest.environment_seed_roles",
    )
    if any("validation" in role.lower() or "lockbox" in role.lower() for role in roles):
        raise RecurrentArtifactError(
            "full-world replay manifest must not consume validation or lockbox roles"
        )
    _validated_text_list(
        manifest.get("scenario_names"),
        field="full_world_replay_manifest.scenario_names",
    )
    tick_horizons = manifest.get("tick_horizons")
    if not isinstance(tick_horizons, list) or not tick_horizons:
        raise RecurrentArtifactError(
            "full_world_replay_manifest.tick_horizons must be non-empty"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in tick_horizons
    ):
        raise RecurrentArtifactError(
            "full_world_replay_manifest.tick_horizons must be positive integers"
        )
    if len(tick_horizons) != len(set(tick_horizons)):
        raise RecurrentArtifactError(
            "full_world_replay_manifest.tick_horizons must be unique"
        )
    world_count = _strict_positive_int(
        manifest.get("world_count"), "full_world_replay_manifest.world_count"
    )
    replay_count = _strict_positive_int(
        manifest.get("replay_verified_world_count"),
        "full_world_replay_manifest.replay_verified_world_count",
    )
    if replay_count != world_count:
        raise RecurrentArtifactError(
            "full-world replay manifest must verify every world"
        )
    _strict_positive_int(
        manifest.get("policy_sampling_stream_count"),
        "full_world_replay_manifest.policy_sampling_stream_count",
    )
    if manifest.get("all_replays_exact") is not True:
        raise RecurrentArtifactError("full-world replay manifest is not exact")
    _validate_trimmed_text(
        manifest.get("verification_runner"),
        field="full_world_replay_manifest.verification_runner",
        maximum=256,
    )
    return manifest


def _validate_crash_checkpoint_and_decode(
    checkpoint: Mapping[str, object],
) -> tuple[_ValidatedArtifact, object, object]:
    if not isinstance(checkpoint, Mapping):
        raise RecurrentArtifactError("crash checkpoint must be a mapping")
    _require_exact_keys(checkpoint, _CRASH_CHECKPOINT_KEYS, field="crash checkpoint")
    if checkpoint.get("schema_version") != (
        RECURRENT_TRAINING_CRASH_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RecurrentArtifactError("crash checkpoint schema is missing or stale")
    if checkpoint.get("checkpoint_kind") != RECURRENT_TRAINING_CRASH_CHECKPOINT_KIND:
        raise RecurrentArtifactError("crash checkpoint kind is missing or stale")
    if checkpoint.get("runtime_policy_eligible") is not False:
        raise RecurrentArtifactError(
            "crash checkpoint cannot be runtime-policy eligible"
        )
    if checkpoint.get("resumable_training_state") is not True:
        raise RecurrentArtifactError("crash checkpoint is not resumable training state")
    if checkpoint.get("digest_policy") != RECURRENT_CRASH_CHECKPOINT_DIGEST_POLICY:
        raise RecurrentArtifactError("crash checkpoint digest policy is stale")

    source = _required_mapping(checkpoint.get("source"), "source")
    _require_exact_keys(
        source,
        frozenset(
            {
                "source_commit",
                "source_commit_sha256",
                "source_manifest_sha256",
                "seed_registry_digest",
            }
        ),
        field="source",
    )
    source_commit = _validate_source_commit(source.get("source_commit"))
    expected_source_sha = hashlib.sha256(source_commit.encode("ascii")).hexdigest()
    if source.get("source_commit_sha256") != expected_source_sha:
        raise RecurrentArtifactError("crash checkpoint source commit hash drifted")
    _validate_sha256(
        source.get("source_manifest_sha256"), field="source_manifest_sha256"
    )
    _validate_sha256(source.get("seed_registry_digest"), field="seed_registry_digest")

    configuration = _required_mapping(checkpoint.get("configuration"), "configuration")
    _require_exact_keys(
        configuration,
        frozenset(
            {
                "model_config",
                "model_config_sha256",
                "optimizer_type",
                "training_config",
                "training_config_sha256",
            }
        ),
        field="configuration",
    )
    model_config = _required_mapping(
        configuration.get("model_config"), "configuration.model_config"
    )
    _require_exact_keys(model_config, _CONFIG_KEYS, field="configuration.model_config")
    training_config = _json_mapping_copy(
        configuration.get("training_config"), field="training_config"
    )
    _validate_trimmed_text(
        configuration.get("optimizer_type"),
        field="optimizer_type",
        maximum=256,
    )
    if configuration.get("model_config_sha256") != _json_sha256(model_config):
        raise RecurrentArtifactError("crash checkpoint model config hash drifted")
    if configuration.get("training_config_sha256") != _json_sha256(training_config):
        raise RecurrentArtifactError("crash checkpoint training config hash drifted")

    progress = _required_mapping(checkpoint.get("progress"), "progress")
    _require_exact_keys(
        progress,
        frozenset({"learner_seed", "completed_updates", "run_id", "model_training"}),
        field="progress",
    )
    _validate_seed(progress.get("learner_seed"))
    _strict_nonnegative_int(progress.get("completed_updates"), "completed_updates")
    _validate_trimmed_text(progress.get("run_id"), field="run_id", maximum=256)
    if type(progress.get("model_training")) is not bool:
        raise RecurrentArtifactError("model_training must be an exact boolean")

    model_payload = checkpoint.get("model")
    tensors = checkpoint.get("model_tensors")
    serialization = {
        "format": "json_tensor_artifact_v2",
        "tensor_encoding": RECURRENT_TENSOR_ENCODING,
        "tensor_dtype": RECURRENT_TENSOR_DTYPE,
        "tensor_byte_order": "little",
        "tensor_layout": "contiguous_c_order",
        "tensor_count": len(tensors) if isinstance(tensors, list) else -1,
        "whole_model_digest_policy": RECURRENT_MODEL_DIGEST_POLICY,
        "whole_model_sha256": checkpoint.get("parameters_sha256"),
        "artifact_digest_policy": RECURRENT_ARTIFACT_DIGEST_POLICY,
    }
    dummy_provenance = {
        "training_config": training_config,
        "seed_registry_digest": source.get("seed_registry_digest"),
        "source_commit": source_commit,
        "data_metadata": {},
        "run_metadata": {},
        "learner_seed": progress.get("learner_seed"),
        "learner_device": "checkpoint",
    }
    validated = _validated_model_from_extended_artifact(
        model_payload=model_payload,
        serialization=serialization,
        provenance=dummy_provenance,
        tensors=tensors,
    )
    if dict(model_config) != asdict(validated.config):
        raise RecurrentArtifactError("crash checkpoint model configurations disagree")

    optimizer_encoded = checkpoint.get("optimizer_state")
    rng_encoded = checkpoint.get("rng_state")
    if checkpoint.get("optimizer_state_sha256") != _json_sha256(optimizer_encoded):
        raise RecurrentArtifactError("optimizer state SHA256 mismatch")
    if checkpoint.get("rng_state_sha256") != _json_sha256(rng_encoded):
        raise RecurrentArtifactError("RNG state SHA256 mismatch")
    optimizer_state = _decode_checkpoint_state(
        optimizer_encoded, field="optimizer_state"
    )
    rng_state = _decode_checkpoint_state(rng_encoded, field="rng_state")
    if _encode_checkpoint_state(optimizer_state, field="optimizer_state") != (
        optimizer_encoded
    ):
        raise RecurrentArtifactError("optimizer state is not canonically encoded")
    if _encode_checkpoint_state(rng_state, field="rng_state") != rng_encoded:
        raise RecurrentArtifactError("RNG state is not canonically encoded")

    observed_checkpoint_sha256 = checkpoint.get("checkpoint_sha256")
    _validate_sha256(observed_checkpoint_sha256, field="checkpoint_sha256")
    if observed_checkpoint_sha256 != _checkpoint_sha256(checkpoint):
        raise RecurrentArtifactError("checkpoint SHA256 mismatch")
    return validated, optimizer_state, rng_state


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
        "format": "json_tensor_artifact_v2",
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
            public_input_schema_version=_strict_text(
                config_payload.get("public_input_schema_version"),
                "public_input_schema_version",
            ),
            public_input_size=_strict_int(
                config_payload.get("public_input_size"),
                "public_input_size",
            ),
            genome_conditioning_mode=_strict_text(
                config_payload.get("genome_conditioning_mode"),
                "genome_conditioning_mode",
            ),
            critic_genome_conditioning=_strict_text(
                config_payload.get("critic_genome_conditioning"),
                "critic_genome_conditioning",
            ),
            value_shared_trunk_gradient=_strict_text(
                config_payload.get("value_shared_trunk_gradient"),
                "value_shared_trunk_gradient",
            ),
        )
    except ValueError as exc:
        raise RecurrentArtifactError(f"invalid model config: {exc}") from exc
    expected_model = _model_contract_payload(config)
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
        decoded_tensor = torch.from_numpy(copied)
        if name in _GENOME_FILM_BUFFER_NAMES and not torch.equal(
            decoded_tensor,
            expected_state[name],
        ):
            raise RecurrentArtifactError(
                f"fixed genome conditioning buffer {name} drifted"
            )
        state_dict[name] = decoded_tensor
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


def _strict_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentArtifactError(f"{field} must be non-empty trimmed text")
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


def _model_contract_payload(
    config: RecurrentActorCriticConfig,
) -> dict[str, object]:
    return {
        "contract_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        "config": asdict(config),
        "public_input_schema_version": config.public_input_schema_version,
        "public_input_size": config.public_input_size,
        "previous_public_feedback_schema_version": (
            PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
        ),
        "previous_public_feedback_size": PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        "learned_encoder_input_size": config.learned_encoder_input_size,
        "action_mask_contract_version": ACTION_MASK_CONTRACT_VERSION,
        "action_ordering": list(ACTION_NAMES),
        "action_count": ACTION_COUNT,
        "architecture_contract": recurrent_actor_critic_contract(config),
        "runtime_integrated": False,
    }


def _decode_float32_record(value: object, *, expected_name: str) -> Tensor:
    record = _required_mapping(value, expected_name)
    _require_exact_keys(record, _TENSOR_KEYS, field=expected_name)
    if record.get("name") != expected_name:
        raise RecurrentArtifactError(f"{expected_name} tensor name drifted")
    if record.get("dtype") != RECURRENT_TENSOR_DTYPE:
        raise RecurrentArtifactError(f"{expected_name} tensor dtype drifted")
    if record.get("encoding") != RECURRENT_TENSOR_ENCODING:
        raise RecurrentArtifactError(f"{expected_name} tensor encoding drifted")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise RecurrentArtifactError(f"{expected_name} tensor shape is invalid")
    expected_byte_length = math.prod(shape) * 4
    if record.get("byte_length") != expected_byte_length:
        raise RecurrentArtifactError(f"{expected_name} tensor byte length drifted")
    raw = _decode_tensor_bytes(record, name=expected_name)
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != math.prod(shape) or not bool(np.isfinite(values).all()):
        raise RecurrentArtifactError(f"{expected_name} tensor values are invalid")
    return torch.from_numpy(
        np.array(values, dtype=np.float32, copy=True).reshape(shape)
    )


def _validated_text_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RecurrentArtifactError(f"{field} must be a non-empty array")
    result = [
        _validate_trimmed_text(item, field=f"{field}[]", maximum=256) for item in value
    ]
    if len(result) != len(set(result)):
        raise RecurrentArtifactError(f"{field} must contain unique values")
    return result


def _validate_trimmed_text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise RecurrentArtifactError(f"{field} must be valid trimmed text")
    return value


def _strict_nonnegative_int(value: object, field: str) -> int:
    parsed = _strict_int(value, field)
    if parsed < 0:
        raise RecurrentArtifactError(f"{field} must be nonnegative")
    return parsed


def _strict_positive_int(value: object, field: str) -> int:
    parsed = _strict_int(value, field)
    if parsed <= 0:
        raise RecurrentArtifactError(f"{field} must be positive")
    return parsed


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _checkpoint_sha256(checkpoint: Mapping[str, object]) -> str:
    without_digest = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    return _json_sha256(without_digest)


def _write_atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
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


def _read_strict_json_mapping(
    path: str | Path,
    *,
    label: str,
) -> dict[str, object]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrentArtifactError(f"failed to read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecurrentArtifactError(f"{label} root must be a JSON object")
    return payload


_STATE_NUMPY_DTYPES: dict[str, np.dtype[Any]] = {
    "float16_le": np.dtype("<f2"),
    "float32_le": np.dtype("<f4"),
    "float64_le": np.dtype("<f8"),
    "int32_le": np.dtype("<i4"),
    "int64_le": np.dtype("<i8"),
    "uint8": np.dtype("u1"),
    "uint32_le": np.dtype("<u4"),
    "bool": np.dtype("?"),
}
_TORCH_STATE_DTYPE_LABELS: dict[torch.dtype, str] = {
    torch.float16: "float16_le",
    torch.float32: "float32_le",
    torch.float64: "float64_le",
    torch.int32: "int32_le",
    torch.int64: "int64_le",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STATE_TORCH_DTYPES: dict[str, torch.dtype] = {
    label: dtype for dtype, label in _TORCH_STATE_DTYPE_LABELS.items()
}


def _encode_checkpoint_state(value: object, *, field: str) -> dict[str, object]:
    return _encode_checkpoint_value(value, field=field, depth=0)


def _encode_checkpoint_value(
    value: object,
    *,
    field: str,
    depth: int,
) -> dict[str, object]:
    if depth > 64:
        raise RecurrentArtifactError(f"{field} exceeds checkpoint nesting limit")
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecurrentArtifactError(f"{field} contains a non-finite float")
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, np.generic):
        return _encode_checkpoint_value(value.item(), field=field, depth=depth + 1)
    if isinstance(value, Tensor):
        label = _TORCH_STATE_DTYPE_LABELS.get(value.dtype)
        if label is None:
            raise RecurrentArtifactError(
                f"{field} tensor dtype {value.dtype} is unsupported"
            )
        array = value.detach().to(device="cpu").contiguous().numpy()
        return _encode_state_array(array, kind="torch_tensor", dtype_label=label)
    if isinstance(value, np.ndarray):
        label = _numpy_dtype_label(value.dtype)
        return _encode_state_array(value, kind="numpy_array", dtype_label=label)
    if isinstance(value, Mapping):
        encoded_items: list[dict[str, object]] = []
        for key, item in value.items():
            encoded_key = _encode_checkpoint_key(key, field=field)
            encoded_items.append(
                {
                    "key": encoded_key,
                    "value": _encode_checkpoint_value(
                        item,
                        field=f"{field}[{key!r}]",
                        depth=depth + 1,
                    ),
                }
            )
        encoded_items.sort(key=lambda item: _canonical_json_bytes(item["key"]))
        return {"kind": "mapping", "items": encoded_items}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _encode_checkpoint_value(
                    item,
                    field=f"{field}[{index}]",
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _encode_checkpoint_value(
                    item,
                    field=f"{field}[{index}]",
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ],
        }
    raise RecurrentArtifactError(
        f"{field} contains unsupported checkpoint value {type(value).__name__}"
    )


def _encode_checkpoint_key(value: object, *, field: str) -> dict[str, object]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"kind": "int", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    raise RecurrentArtifactError(
        f"{field} mapping keys must be strings or non-boolean integers"
    )


def _numpy_dtype_label(dtype: np.dtype[Any]) -> str:
    normalized = np.dtype(dtype).newbyteorder("<")
    for label, candidate in _STATE_NUMPY_DTYPES.items():
        if normalized == candidate:
            return label
    raise RecurrentArtifactError(f"numpy dtype {dtype} is unsupported")


def _encode_state_array(
    value: np.ndarray,
    *,
    kind: str,
    dtype_label: str,
) -> dict[str, object]:
    dtype = _STATE_NUMPY_DTYPES[dtype_label]
    array = np.asarray(value).astype(dtype, copy=False)
    if array.dtype.kind in {"f", "c"} and not bool(np.isfinite(array).all()):
        raise RecurrentArtifactError("checkpoint tensor contains a non-finite value")
    raw = np.ascontiguousarray(array).tobytes(order="C")
    return {
        "kind": kind,
        "dtype": dtype_label,
        "shape": list(array.shape),
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _decode_checkpoint_state(value: object, *, field: str) -> object:
    return _decode_checkpoint_value(value, field=field, depth=0)


def _decode_checkpoint_value(value: object, *, field: str, depth: int) -> object:
    if depth > 64:
        raise RecurrentArtifactError(f"{field} exceeds checkpoint nesting limit")
    payload = _required_mapping(value, field)
    kind = payload.get("kind")
    if kind == "none":
        _require_exact_keys(payload, frozenset({"kind"}), field=field)
        return None
    if kind in {"bool", "int", "float", "str"}:
        _require_exact_keys(payload, frozenset({"kind", "value"}), field=field)
        item = payload.get("value")
        valid = (
            (kind == "bool" and type(item) is bool)
            or (kind == "int" and isinstance(item, int) and not isinstance(item, bool))
            or (kind == "float" and isinstance(item, float) and math.isfinite(item))
            or (kind == "str" and isinstance(item, str))
        )
        if not valid:
            raise RecurrentArtifactError(f"{field} scalar kind/value mismatch")
        return item
    if kind in {"torch_tensor", "numpy_array"}:
        _require_exact_keys(
            payload,
            frozenset({"kind", "dtype", "shape", "byte_length", "sha256", "data"}),
            field=field,
        )
        dtype_label = payload.get("dtype")
        if not isinstance(dtype_label, str) or dtype_label not in _STATE_NUMPY_DTYPES:
            raise RecurrentArtifactError(f"{field} checkpoint dtype is unsupported")
        if kind == "torch_tensor" and dtype_label not in _STATE_TORCH_DTYPES:
            raise RecurrentArtifactError(f"{field} torch dtype is unsupported")
        shape = payload.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        ):
            raise RecurrentArtifactError(f"{field} checkpoint shape is invalid")
        dtype = _STATE_NUMPY_DTYPES[dtype_label]
        expected_length = math.prod(shape) * dtype.itemsize
        if payload.get("byte_length") != expected_length:
            raise RecurrentArtifactError(f"{field} checkpoint byte length mismatch")
        data = payload.get("data")
        if not isinstance(data, str):
            raise RecurrentArtifactError(f"{field} checkpoint data must be base64")
        try:
            raw = base64.b64decode(data.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise RecurrentArtifactError(f"{field} checkpoint data is invalid") from exc
        if len(raw) != expected_length:
            raise RecurrentArtifactError(f"{field} decoded byte length mismatch")
        _validate_sha256(payload.get("sha256"), field=f"{field}.sha256")
        if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
            raise RecurrentArtifactError(f"{field} checkpoint SHA256 mismatch")
        array = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        if array.dtype.kind in {"f", "c"} and not bool(np.isfinite(array).all()):
            raise RecurrentArtifactError(f"{field} contains non-finite values")
        if kind == "numpy_array":
            return array
        return torch.from_numpy(array).to(dtype=_STATE_TORCH_DTYPES[dtype_label])
    if kind == "mapping":
        _require_exact_keys(payload, frozenset({"kind", "items"}), field=field)
        items = payload.get("items")
        if not isinstance(items, list):
            raise RecurrentArtifactError(f"{field}.items must be an array")
        result: dict[object, object] = {}
        canonical_keys: list[bytes] = []
        for index, item in enumerate(items):
            entry = _required_mapping(item, f"{field}.items[{index}]")
            _require_exact_keys(
                entry,
                frozenset({"key", "value"}),
                field=f"{field}.items[{index}]",
            )
            key = _decode_checkpoint_key(
                entry.get("key"), field=f"{field}.items[{index}].key"
            )
            if key in result:
                raise RecurrentArtifactError(f"{field} contains a duplicate key")
            canonical_keys.append(_canonical_json_bytes(entry.get("key")))
            result[key] = _decode_checkpoint_value(
                entry.get("value"),
                field=f"{field}[{key!r}]",
                depth=depth + 1,
            )
        if canonical_keys != sorted(canonical_keys):
            raise RecurrentArtifactError(f"{field} mapping keys are not canonical")
        return result
    if kind in {"tuple", "list"}:
        _require_exact_keys(payload, frozenset({"kind", "items"}), field=field)
        items = payload.get("items")
        if not isinstance(items, list):
            raise RecurrentArtifactError(f"{field}.items must be an array")
        decoded = [
            _decode_checkpoint_value(
                item,
                field=f"{field}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(items)
        ]
        return tuple(decoded) if kind == "tuple" else decoded
    raise RecurrentArtifactError(f"{field} checkpoint kind is missing or stale")


def _decode_checkpoint_key(value: object, *, field: str) -> object:
    payload = _required_mapping(value, field)
    _require_exact_keys(payload, frozenset({"kind", "value"}), field=field)
    kind = payload.get("kind")
    item = payload.get("value")
    if kind == "str" and isinstance(item, str):
        return item
    if kind == "int" and isinstance(item, int) and not isinstance(item, bool):
        return item
    raise RecurrentArtifactError(f"{field} checkpoint mapping key is invalid")


__all__ = [
    "FROZEN_RECURRENT_POLICY_ARTIFACT_KIND",
    "FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION",
    "FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION",
    "RECURRENT_ARTIFACT_SCHEMA_VERSION",
    "RECURRENT_REPLAY_PROBE_CONTRACT_VERSION",
    "RECURRENT_TENSOR_DTYPE",
    "RECURRENT_TENSOR_ENCODING",
    "RECURRENT_TRAINING_CRASH_CHECKPOINT_KIND",
    "RECURRENT_TRAINING_CRASH_CHECKPOINT_SCHEMA_VERSION",
    "LoadedFrozenRecurrentPolicyArtifact",
    "LoadedRecurrentArtifact",
    "LoadedRecurrentTrainingCrashCheckpoint",
    "RecurrentArtifactError",
    "build_frozen_recurrent_policy_artifact",
    "build_recurrent_artifact",
    "build_recurrent_training_crash_checkpoint",
    "load_frozen_recurrent_policy_artifact",
    "load_recurrent_artifact",
    "load_recurrent_training_crash_checkpoint",
    "model_from_frozen_recurrent_policy_artifact",
    "model_from_recurrent_artifact",
    "save_frozen_recurrent_policy_artifact",
    "save_recurrent_artifact",
    "save_recurrent_training_crash_checkpoint",
    "validate_frozen_recurrent_policy_artifact",
    "validate_recurrent_artifact",
    "validate_recurrent_training_crash_checkpoint",
    "write_frozen_recurrent_policy_artifact",
    "write_recurrent_artifact",
    "write_recurrent_training_crash_checkpoint",
]
