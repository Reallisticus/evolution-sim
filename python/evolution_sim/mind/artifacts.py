from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.observations import OBSERVATION_SCHEMA_VERSION
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.trajectory import TRAJECTORY_SCHEMA_VERSION
from evolution_sim.mind.contracts import (
    MIND_MODEL_ARTIFACT_VERSION,
    MIND_RUNTIME_ENABLED_DEFAULT,
)


class MindArtifactError(ValueError):
    pass


def require_mind_enabled(*, enable_mind: bool) -> None:
    if not enable_mind:
        raise MindArtifactError(
            "Mind runtime inference is disabled by default; pass enable_mind=True."
        )


def load_model_artifact(
    path: str | Path,
    *,
    enable_mind: bool = MIND_RUNTIME_ENABLED_DEFAULT,
) -> dict[str, object]:
    require_mind_enabled(enable_mind=enable_mind)
    artifact_path = Path(path)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MindArtifactError(f"failed to read model artifact: {artifact_path}") from exc
    except json.JSONDecodeError as exc:
        raise MindArtifactError(f"model artifact is not valid JSON: {exc.msg}") from exc
    if not isinstance(artifact, dict):
        raise MindArtifactError("model artifact must be a JSON object")
    validate_model_artifact_manifest(artifact)
    return artifact


def validate_model_artifact_manifest(artifact: dict[str, object]) -> None:
    manifest = artifact.get("manifest")
    if not isinstance(manifest, dict):
        raise MindArtifactError("model artifact is missing manifest")
    expected_versions = {
        "artifact_version": MIND_MODEL_ARTIFACT_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
    }
    for field, expected in expected_versions.items():
        if manifest.get(field) != expected:
            raise MindArtifactError(
                f"model artifact manifest {field} expected {expected}, found {manifest.get(field)!r}"
            )
    if not isinstance(artifact.get("model"), dict):
        raise MindArtifactError("model artifact is missing model payload")
    model_type = manifest.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise MindArtifactError("model artifact manifest is missing model_type")


def write_model_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    validate_model_artifact_manifest(artifact)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
