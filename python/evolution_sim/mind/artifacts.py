from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_SCHEMA_VERSION
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.trajectory import TRAJECTORY_SCHEMA_VERSION
from evolution_sim.mind.contracts import (
    MIND_MODEL_ARTIFACT_VERSION,
    MIND_RUNTIME_ENABLED_DEFAULT,
)
from evolution_sim.mind.feature_policy import FEATURE_POLICY_VERSION
from evolution_sim.mind.provenance import validate_dataset_provenance


class MindArtifactError(ValueError):
    pass


BEHAVIOR_CLONING_BASELINE_MODEL_TYPE = "guarded_contextual_local_prior_bc_v2"
REWARD_WEIGHTED_BASELINE_MODEL_TYPE = (
    "guarded_reward_weighted_contextual_prior_bc_v1"
)
CONDITIONAL_SCORE_POLICY = "smoothed_contextual_action_prior_v1"
CONTEXTUAL_PRIOR_TRAINER = "contextual-prior"
REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER = "reward-weighted-contextual-prior"
UNIFORM_SAMPLE_WEIGHT_POLICY = "uniform_v1"
REWARD_TOTAL_SHIFTED_SAMPLE_WEIGHT_POLICY = "reward_total_shifted_clamp_v1"
HEURISTIC_GUARD_POLICY = "observation_heuristic_safety_floor_v1"
HEURISTIC_DELEGATE_POLICY = "observation_heuristic_confidence_delegate_v1"
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        BEHAVIOR_CLONING_BASELINE_MODEL_TYPE,
        REWARD_WEIGHTED_BASELINE_MODEL_TYPE,
    }
)


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
                (
                    f"model artifact manifest {field} expected {expected}, "
                    f"found {manifest.get(field)!r}"
                )
            )
    model = artifact.get("model")
    if not isinstance(model, dict):
        raise MindArtifactError("model artifact is missing model payload")
    model_type = manifest.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise MindArtifactError("model artifact manifest is missing model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise MindArtifactError(
            f"model artifact manifest model_type is unsupported: {model_type!r}"
        )
    try:
        validate_dataset_provenance(manifest.get("provenance"))
    except ValueError as exc:
        raise MindArtifactError(
            f"model artifact manifest provenance is invalid: {exc}"
        ) from exc
    trained_record_count = _required_non_negative_int(
        manifest,
        "trained_record_count",
        location="manifest",
    )
    _validate_behavior_cloning_model_payload(
        model,
        model_type=model_type,
        trained_record_count=trained_record_count,
    )


def _validate_behavior_cloning_model_payload(
    model: dict[str, object],
    *,
    model_type: str,
    trained_record_count: int,
) -> None:
    action_scores = _required_mapping(model, "action_scores", location="model")
    _validate_action_score_map(
        action_scores,
        location="model.action_scores",
    )
    action_score_metadata = _required_mapping(
        model,
        "action_score_metadata",
        location="model",
    )
    _validate_score_metadata(
        action_score_metadata,
        location="model.action_score_metadata",
        min_record_count=0,
        max_record_count=trained_record_count,
        expected_record_count=trained_record_count,
    )
    conditional_scores = _required_mapping(
        model,
        "conditional_action_scores",
        location="model",
    )
    conditional_metadata = _required_mapping(
        model,
        "conditional_action_metadata",
        location="model",
    )
    if set(conditional_scores) != set(conditional_metadata):
        raise MindArtifactError(
            (
                "model.conditional_action_scores keys must match "
                "model.conditional_action_metadata keys"
            )
        )
    conditional_min_records = _required_positive_int(
        model,
        "conditional_min_records",
        location="model",
    )
    conditional_score_policy = model.get("conditional_score_policy")
    if conditional_score_policy != CONDITIONAL_SCORE_POLICY:
        raise MindArtifactError(
            (
                "model.conditional_score_policy expected "
                f"{CONDITIONAL_SCORE_POLICY}, found {conditional_score_policy!r}"
            )
        )
    prior_exponent = _required_finite_number(
        model,
        "conditional_prior_correction_exponent",
        location="model",
        minimum=0.0,
    )
    if prior_exponent > 1.0:
        raise MindArtifactError(
            "model.conditional_prior_correction_exponent must be <= 1.0"
        )
    smoothing_alpha = _required_finite_number(
        model,
        "conditional_score_smoothing_alpha",
        location="model",
        minimum=0.0,
    )
    if smoothing_alpha <= 0.0:
        raise MindArtifactError(
            "model.conditional_score_smoothing_alpha must be positive"
        )
    _validate_training_weight_metadata(
        model,
        model_type=model_type,
        trained_record_count=trained_record_count,
    )
    for feature_key, scores in conditional_scores.items():
        if not isinstance(feature_key, str) or not feature_key:
            raise MindArtifactError(
                "model.conditional_action_scores keys must be non-empty strings"
            )
        if not isinstance(scores, dict):
            raise MindArtifactError(
                f"model.conditional_action_scores[{feature_key!r}] must be an object"
            )
        _validate_action_score_map(
            scores,
            location=f"model.conditional_action_scores[{feature_key!r}]",
        )
        metadata = conditional_metadata[feature_key]
        if not isinstance(metadata, dict):
            raise MindArtifactError(
                f"model.conditional_action_metadata[{feature_key!r}] must be an object"
            )
        _validate_score_metadata(
            metadata,
            location=f"model.conditional_action_metadata[{feature_key!r}]",
            min_record_count=conditional_min_records,
            max_record_count=trained_record_count,
            expected_record_count=None,
        )

    fallback_action = model.get("fallback_action")
    if fallback_action not in ACTION_NAMES:
        raise MindArtifactError(
            "model.fallback_action must be a known action from the action contract"
        )
    feature_policy_version = model.get("feature_policy_version")
    if feature_policy_version != FEATURE_POLICY_VERSION:
        raise MindArtifactError(
            (
                "model.feature_policy_version expected "
                f"{FEATURE_POLICY_VERSION}, found {feature_policy_version!r}"
            )
        )
    heuristic_guard_policy = model.get("heuristic_guard_policy")
    if heuristic_guard_policy != HEURISTIC_GUARD_POLICY:
        raise MindArtifactError(
            (
                "model.heuristic_guard_policy expected "
                f"{HEURISTIC_GUARD_POLICY}, found {heuristic_guard_policy!r}"
            )
        )
    _required_finite_number(
        model,
        "heuristic_confidence_threshold",
        location="model",
        minimum=0.0,
    )
    _required_finite_number(
        model,
        "heuristic_override_min_margin",
        location="model",
        minimum=0.0,
    )
    heuristic_delegate_policy = model.get("heuristic_delegate_policy")
    if heuristic_delegate_policy != HEURISTIC_DELEGATE_POLICY:
        raise MindArtifactError(
            (
                "model.heuristic_delegate_policy expected "
                f"{HEURISTIC_DELEGATE_POLICY}, found {heuristic_delegate_policy!r}"
            )
        )
    _required_finite_number(
        model,
        "heuristic_delegate_max_training_score_margin",
        location="model",
        minimum=0.0,
    )
    _required_nullable_finite_number(
        model,
        "heuristic_safe_local_eat_min_score",
        location="model",
        minimum=0.0,
    )
    _required_nullable_finite_number(
        model,
        "heuristic_safe_local_eat_min_food",
        location="model",
        minimum=0.0,
    )
    _required_nullable_finite_number(
        model,
        "heuristic_safe_local_eat_min_plant_ratio",
        location="model",
        minimum=0.0,
    )
    _required_nullable_finite_number(
        model,
        "heuristic_safe_plant_move_min_score",
        location="model",
        minimum=0.0,
    )
    _required_nullable_finite_number(
        model,
        "heuristic_safe_plant_move_min_strength",
        location="model",
        minimum=0.0,
    )
    _required_nullable_finite_number(
        model,
        "heuristic_safe_plant_move_max_local_food_ratio",
        location="model",
        minimum=0.0,
    )
    _required_nullable_positive_int(
        model,
        "heuristic_safe_plant_move_max_distance",
        location="model",
    )


def _validate_action_score_map(
    scores: dict[str, object],
    *,
    location: str,
) -> None:
    keys = set(scores)
    expected = set(ACTION_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise MindArtifactError(
            f"{location} must cover the full action vocabulary ({'; '.join(detail)})"
        )
    for action in ACTION_NAMES:
        _finite_number(scores.get(action), f"{location}.{action}")
    total = sum(float(scores[action]) for action in ACTION_NAMES)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise MindArtifactError(f"{location} scores must sum to 1.0")


def _validate_score_metadata(
    metadata: dict[str, object],
    *,
    location: str,
    min_record_count: int,
    max_record_count: int,
    expected_record_count: int | None,
) -> None:
    record_count = _required_non_negative_int(metadata, "record_count", location=location)
    if record_count < min_record_count:
        raise MindArtifactError(
            (
                f"{location}.record_count {record_count} is below the required "
                f"support count {min_record_count}"
            )
        )
    if record_count > max_record_count:
        raise MindArtifactError(
            f"{location}.record_count exceeds trained_record_count"
        )
    if expected_record_count is not None and record_count != expected_record_count:
        raise MindArtifactError(
            (
                f"{location}.record_count expected {expected_record_count}, "
                f"found {record_count}"
            )
        )
    top_action = metadata.get("top_action")
    runner_up_action = metadata.get("runner_up_action")
    if top_action not in ACTION_NAMES:
        raise MindArtifactError(f"{location}.top_action is not a known action")
    if runner_up_action not in ACTION_NAMES:
        raise MindArtifactError(f"{location}.runner_up_action is not a known action")
    top_score = _required_finite_number(metadata, "top_score", location=location)
    runner_up_score = _required_finite_number(
        metadata,
        "runner_up_score",
        location=location,
    )
    score_margin = _required_finite_number(
        metadata,
        "score_margin",
        location=location,
    )
    if top_score < runner_up_score:
        raise MindArtifactError(f"{location}.top_score must be >= runner_up_score")
    if not math.isclose(score_margin, top_score - runner_up_score, abs_tol=1e-9):
        raise MindArtifactError(
            f"{location}.score_margin must equal top_score - runner_up_score"
        )
    if "weighted_record_count" in metadata:
        _required_finite_number(
            metadata,
            "weighted_record_count",
            location=location,
            minimum=0.0,
        )


def _validate_training_weight_metadata(
    model: dict[str, object],
    *,
    model_type: str,
    trained_record_count: int,
) -> None:
    trainer = model.get("trainer")
    sample_weight_policy = model.get("sample_weight_policy")
    if model_type == BEHAVIOR_CLONING_BASELINE_MODEL_TYPE:
        expected_trainer = CONTEXTUAL_PRIOR_TRAINER
        expected_sample_weight_policy = UNIFORM_SAMPLE_WEIGHT_POLICY
    elif model_type == REWARD_WEIGHTED_BASELINE_MODEL_TYPE:
        expected_trainer = REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER
        expected_sample_weight_policy = REWARD_TOTAL_SHIFTED_SAMPLE_WEIGHT_POLICY
    else:
        raise MindArtifactError(
            f"model artifact manifest model_type is unsupported: {model_type!r}"
        )
    if trainer != expected_trainer:
        raise MindArtifactError(
            f"model.trainer expected {expected_trainer}, found {trainer!r}"
        )
    if sample_weight_policy != expected_sample_weight_policy:
        raise MindArtifactError(
            (
                "model.sample_weight_policy expected "
                f"{expected_sample_weight_policy}, found {sample_weight_policy!r}"
            )
        )
    sample_weight_base = _required_finite_number(
        model,
        "sample_weight_base",
        location="model",
        minimum=0.0,
    )
    sample_weight_min = _required_finite_number(
        model,
        "sample_weight_min",
        location="model",
        minimum=0.0,
    )
    sample_weight_total = _required_finite_number(
        model,
        "sample_weight_total",
        location="model",
        minimum=0.0,
    )
    if sample_weight_min > sample_weight_base:
        raise MindArtifactError("model.sample_weight_min must be <= sample_weight_base")
    if model_type == BEHAVIOR_CLONING_BASELINE_MODEL_TYPE and not math.isclose(
        sample_weight_total,
        float(trained_record_count),
        abs_tol=1e-9,
    ):
        raise MindArtifactError(
            "uniform model.sample_weight_total must equal trained_record_count"
        )


def _required_mapping(
    payload: dict[str, object],
    field: str,
    *,
    location: str,
) -> dict[str, object]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise MindArtifactError(f"{location}.{field} must be an object")
    return value


def _required_non_negative_int(
    payload: dict[str, object],
    field: str,
    *,
    location: str,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MindArtifactError(f"{location}.{field} must be an integer")
    if value < 0:
        raise MindArtifactError(f"{location}.{field} must be non-negative")
    return value


def _required_positive_int(
    payload: dict[str, object],
    field: str,
    *,
    location: str,
) -> int:
    value = _required_non_negative_int(payload, field, location=location)
    if value <= 0:
        raise MindArtifactError(f"{location}.{field} must be positive")
    return value


def _required_nullable_positive_int(
    payload: dict[str, object],
    field: str,
    *,
    location: str,
) -> int | None:
    if field not in payload:
        raise MindArtifactError(f"{location}.{field} is required")
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MindArtifactError(f"{location}.{field} must be an integer or null")
    if value <= 0:
        raise MindArtifactError(f"{location}.{field} must be positive")
    return value


def _required_finite_number(
    payload: dict[str, object],
    field: str,
    *,
    location: str,
    minimum: float | None = None,
) -> float:
    return _finite_number(
        payload.get(field),
        f"{location}.{field}",
        minimum=minimum,
    )


def _required_nullable_finite_number(
    payload: dict[str, object],
    field: str,
    *,
    location: str,
    minimum: float | None = None,
) -> float | None:
    if field not in payload:
        raise MindArtifactError(f"{location}.{field} is required")
    value = payload.get(field)
    if value is None:
        return None
    return _finite_number(value, f"{location}.{field}", minimum=minimum)


def _finite_number(
    value: object,
    location: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MindArtifactError(f"{location} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise MindArtifactError(f"{location} must be finite")
    if minimum is not None and parsed < minimum:
        raise MindArtifactError(f"{location} must be >= {minimum}")
    return parsed


def write_model_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    validate_model_artifact_manifest(artifact)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
