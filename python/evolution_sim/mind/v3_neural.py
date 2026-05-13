from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
)
from evolution_sim.mind.dataset import (
    TrajectoryJsonlDataset,
    combined_dataset_provenance,
    records_with_trajectory_context,
)
from evolution_sim.mind.fixture_labels import MIND_FIXTURE_LABEL_SCHEMA_VERSION
from evolution_sim.mind.horizon_labels import MIND_HORIZON_LABEL_SCHEMA_VERSION
from evolution_sim.mind.policy_inputs import (
    CONTROLLER_DIAGNOSTIC_SELF_FIELDS,
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    ecological_policy_input_contract,
    ecological_policy_input_values,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION = "mind_v3_neural_policy_artifact_v1"
MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED = "anchored-neural"
MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE = "horizon-fixture"
MIND_V3_NEURAL_MODEL_TYPE = "deterministic_ecological_mlp_policy_v1"
MIND_V3_HORIZON_FIXTURE_MODEL_TYPE = "deterministic_horizon_fixture_policy_v2"
MIND_V3_NEURAL_TRAINER = "horizon_fixture_weighted_ecological_mlp_v1"
MIND_V3_HORIZON_FIXTURE_TRAINER = (
    "horizon_fixture_action_value_projection_v1"
)
MIND_V3_NEURAL_BACKEND = "pure_python_deterministic_v1"
MIND_V3_NEURAL_ARCHITECTURE = "fixed_projection_mlp_action_horizon_heads_v1"
MIND_V3_HORIZON_FIXTURE_ARCHITECTURE = (
    "fixed_projection_action_value_horizon_fixture_heads_v1"
)
MIND_V3_NEURAL_INPUT_POLICY = ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
MIND_V3_NEURAL_DEFAULT_HIDDEN_UNITS = 32
MIND_V3_NEURAL_DEFAULT_SEED = 43
MIND_V3_NEURAL_FIXTURE_BIAS_POLICY = (
    "contextual_fixture_floor_gap_action_bias_v2"
)
MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_BIAS_POLICY = (
    "policy_visible_carrion_water_context_bias_v1"
)
MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_MAX_SCALE = 0.18
MIND_V3_NEURAL_SAMPLE_WEIGHT_POLICY = "horizon_survival_reproduction_viability_v1"
MIND_V3_NEURAL_ACTION_PRIOR_LOG_WEIGHT = 0.18
MIND_V3_NEURAL_PROTOTYPE_WEIGHT_SCALE = 2.0
MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE = "none"
MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_SCAVENGER = (
    "visible_carrion_scavenger_v1"
)
MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_OR_RECOVERY_PHASE = (
    "visible_carrion_or_recovery_phase_v1"
)
MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATES = frozenset(
    {
        MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE,
        MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_SCAVENGER,
        MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_OR_RECOVERY_PHASE,
    }
)
MIND_V3_NEURAL_DEFAULT_RECOVERY_PHASE_TICKS = 8
MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY = (
    "branch_survivor_failure_action_log_odds_v1"
)
MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_DEFAULT_SCALE = 0.18
MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_DEFAULT_MAX_ABS = 0.16
MIND_V3_HORIZON_FIXTURE_SCORE_POLICY = (
    "action_conditioned_horizon_fixture_value_v1"
)
MIND_V3_HORIZON_FIXTURE_ACTION_PRIOR_LOG_WEIGHT = 0.05
MIND_V3_HORIZON_FIXTURE_VALUE_WEIGHT_SCALE = 1.65
MIND_V3_HORIZON_FIXTURE_BIAS_SCALE = 0.75
MIND_V3_HORIZON_FIXTURE_BEHAVIOR_SUPPORT_WEIGHT = 0.35
_ECOLOGICAL_SELF_FIELDS = tuple(
    field
    for field in SELF_INPUT_FIELDS
    if field not in CONTROLLER_DIAGNOSTIC_SELF_FIELDS
)
_ECOLOGICAL_PATCH_START = len(_ECOLOGICAL_SELF_FIELDS)
_ECOLOGICAL_NAVIGATION_START = _ECOLOGICAL_PATCH_START + (
    PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
)


class MindV3NeuralArtifactError(ValueError):
    pass


def train_mind_v3_neural_artifact(
    datasets: Sequence[TrajectoryJsonlDataset],
    *,
    horizon_label_report: Mapping[str, object],
    fixture_label_report: Mapping[str, object] | None = None,
    hidden_units: int = MIND_V3_NEURAL_DEFAULT_HIDDEN_UNITS,
    seed: int = MIND_V3_NEURAL_DEFAULT_SEED,
    trajectory_weight_multipliers: Sequence[float] | None = None,
    artifact_mode: str = MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
    neural_residual_scale: float | None = None,
    neural_residual_max_linear_override_margin: float | None = None,
    neural_residual_context_gate: str | None = None,
    neural_residual_recovery_phase_ticks: int | None = None,
    recovery_phase_action_bias: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not datasets:
        raise MindV3NeuralArtifactError("at least one trajectory dataset is required")
    if hidden_units < 1:
        raise MindV3NeuralArtifactError("hidden_units must be positive")
    _validate_horizon_label_report(horizon_label_report)
    if fixture_label_report is not None:
        _validate_fixture_label_report(fixture_label_report)
    mode_config = _artifact_mode_config(artifact_mode)

    contextual_records = tuple(records_with_trajectory_context(datasets))
    trajectory_weights = _trajectory_weight_multipliers(
        datasets,
        trajectory_weight_multipliers,
    )
    source_weight_by_index = _source_weight_by_index(datasets, trajectory_weights)
    labels_by_index = _horizon_labels_by_source_index(horizon_label_report)
    samples = _training_samples(
        contextual_records,
        labels_by_index,
        source_weight_by_index=source_weight_by_index,
    )
    if not samples:
        raise MindV3NeuralArtifactError(
            "no trajectory records could be paired with horizon labels"
        )

    hidden_weights = _deterministic_hidden_weights(
        hidden_units=hidden_units,
        seed=seed,
    )
    hidden_bias = _deterministic_hidden_bias(hidden_units=hidden_units, seed=seed)
    hidden_samples = [
        {
            **sample,
            "hidden": _hidden_activations(
                sample["values"],  # type: ignore[arg-type]
                hidden_weights,
                hidden_bias,
            ),
        }
        for sample in samples
    ]
    global_mean = _weighted_mean_hidden(hidden_samples)
    global_utility = _weighted_mean_sample_utility(hidden_samples)
    action_output_weights = _action_output_weights(
        hidden_samples,
        global_mean=global_mean,
        hidden_units=hidden_units,
    )
    action_output_bias = _action_output_bias(hidden_samples)
    action_value_weights = _action_value_weights(
        hidden_samples,
        global_mean=global_mean,
        global_utility=global_utility,
        hidden_units=hidden_units,
    )
    action_value_bias = _action_value_bias(
        hidden_samples,
        global_utility=global_utility,
    )
    fixture_bias_delta = _fixture_action_bias_delta(fixture_label_report)
    fixture_context_bias = _fixture_context_bias(fixture_label_report)
    horizon_ticks = _horizon_ticks(horizon_label_report)
    survival_heads, reproduction_heads = _horizon_heads(
        hidden_samples,
        hidden_units=hidden_units,
        horizon_ticks=horizon_ticks,
    )
    fixture_summary = _fixture_pressure_summary(fixture_label_report)
    recovery_phase_action_bias_payload = _recovery_phase_action_bias_payload(
        recovery_phase_action_bias
    )
    input_contract = ecological_policy_input_contract()
    residual_config = _neural_residual_config(
        neural_residual_scale=neural_residual_scale,
        neural_residual_max_linear_override_margin=(
            neural_residual_max_linear_override_margin
        ),
        neural_residual_context_gate=neural_residual_context_gate,
        neural_residual_recovery_phase_ticks=(
            neural_residual_recovery_phase_ticks
        ),
    )
    training_contract = {
        "schema_version": MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
        "artifact_mode": mode_config["artifact_mode"],
        "model_type": mode_config["model_type"],
        "trainer": mode_config["trainer"],
        "backend": MIND_V3_NEURAL_BACKEND,
        "architecture": mode_config["architecture"],
        "input_policy": MIND_V3_NEURAL_INPUT_POLICY,
        "sample_weight_policy": MIND_V3_NEURAL_SAMPLE_WEIGHT_POLICY,
        "horizon_fixture_score_policy": MIND_V3_HORIZON_FIXTURE_SCORE_POLICY,
        "horizon_fixture_behavior_support_weight": (
            MIND_V3_HORIZON_FIXTURE_BEHAVIOR_SUPPORT_WEIGHT
        ),
        "fixture_bias_policy": MIND_V3_NEURAL_FIXTURE_BIAS_POLICY,
        "action_prior_log_weight": MIND_V3_NEURAL_ACTION_PRIOR_LOG_WEIGHT,
        "prototype_weight_scale": MIND_V3_NEURAL_PROTOTYPE_WEIGHT_SCALE,
        "trajectory_weight_multipliers": list(trajectory_weights),
        "hidden_units": hidden_units,
        "seed": seed,
        "horizon_ticks": horizon_ticks,
        **residual_config,
    }
    if recovery_phase_action_bias is not None:
        training_contract["recovery_phase_action_bias"] = (
            recovery_phase_action_bias_payload
        )
    artifact = {
        "schema_version": MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
        "artifact_mode": mode_config["artifact_mode"],
        "model_type": mode_config["model_type"],
        "trainer": mode_config["trainer"],
        "backend": MIND_V3_NEURAL_BACKEND,
        "architecture": mode_config["architecture"],
        "input_policy": MIND_V3_NEURAL_INPUT_POLICY,
        "input_contract": input_contract,
        "sample_weight_policy": MIND_V3_NEURAL_SAMPLE_WEIGHT_POLICY,
        "horizon_fixture_score_policy": MIND_V3_HORIZON_FIXTURE_SCORE_POLICY,
        "horizon_fixture_behavior_support_weight": (
            MIND_V3_HORIZON_FIXTURE_BEHAVIOR_SUPPORT_WEIGHT
        ),
        "fixture_bias_policy": MIND_V3_NEURAL_FIXTURE_BIAS_POLICY,
        "action_prior_log_weight": MIND_V3_NEURAL_ACTION_PRIOR_LOG_WEIGHT,
        "prototype_weight_scale": MIND_V3_NEURAL_PROTOTYPE_WEIGHT_SCALE,
        "trajectory_weight_multipliers": list(trajectory_weights),
        "hidden_units": hidden_units,
        "seed": seed,
        "trained_record_count": len(samples),
        "horizon_ticks": horizon_ticks,
        **residual_config,
        "provenance": {
            **combined_dataset_provenance(datasets),
            "horizon_label_schema_version": horizon_label_report.get(
                "schema_version"
            ),
            "horizon_label_digest": stable_payload_digest(
                {
                    "schema_version": horizon_label_report.get("schema_version"),
                    "source": horizon_label_report.get("source"),
                    "aggregate": horizon_label_report.get("aggregate"),
                    "label_count": len(
                        list(horizon_label_report.get("labels", []))
                        if isinstance(horizon_label_report.get("labels"), list)
                        else []
                    ),
                }
            ),
            "fixture_label_schema_version": (
                fixture_label_report.get("schema_version")
                if fixture_label_report is not None
                else None
            ),
            "fixture_label_digest": (
                stable_payload_digest(
                    {
                        "schema_version": fixture_label_report.get(
                            "schema_version"
                        ),
                        "source": fixture_label_report.get("source"),
                        "aggregate": fixture_label_report.get("aggregate"),
                    }
                )
                if fixture_label_report is not None
                else None
            ),
            "training_contract_digest": stable_payload_digest(training_contract),
            "input_contract_digest": stable_payload_digest(input_contract),
        },
        "training_contract": training_contract,
        "training_summary": _training_summary(hidden_samples),
        "fixture_pressure_summary": fixture_summary,
        "hidden_weights": hidden_weights,
        "hidden_bias": hidden_bias,
        "action_output_weights": action_output_weights,
        "action_output_bias": action_output_bias,
        "action_value_weights": action_value_weights,
        "action_value_bias": action_value_bias,
        "action_value_summary": _action_value_summary(
            hidden_samples,
            global_utility=global_utility,
        ),
        "fixture_action_bias_delta": fixture_bias_delta,
        "fixture_context_bias": fixture_context_bias,
        "survival_heads": survival_heads,
        "reproduction_heads": reproduction_heads,
    }
    if recovery_phase_action_bias is not None:
        artifact["recovery_phase_action_bias"] = recovery_phase_action_bias_payload
    validate_mind_v3_neural_artifact(artifact)
    return artifact


def score_mind_v3_neural_artifact(
    *,
    artifact: Mapping[str, object],
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, bool],
    recovery_phase_remaining: int = 0,
) -> dict[str, float]:
    compiled = _compiled_artifact(artifact)
    values = ecological_policy_input_values(dict(observation_input))
    hidden = _hidden_activations(
        values,
        compiled["hidden_weights"],  # type: ignore[arg-type]
        compiled["hidden_bias"],  # type: ignore[arg-type]
    )
    if compiled["model_type"] == MIND_V3_HORIZON_FIXTURE_MODEL_TYPE:
        return _score_horizon_fixture_artifact(
            compiled=compiled,
            values=values,
            hidden=hidden,
            action_mask=action_mask,
        )
    weights = compiled["action_output_weights"]
    bias = compiled["action_output_bias"]
    fixture_delta = compiled["fixture_action_bias_delta"]
    fixture_context_delta = _fixture_context_action_bias(
        compiled["fixture_context_bias"],  # type: ignore[arg-type]
        values,
    )
    recovery_phase_delta = _recovery_phase_action_bias_scores(
        compiled["recovery_phase_action_bias"],  # type: ignore[arg-type]
        recovery_phase_remaining=recovery_phase_remaining,
        action_mask=action_mask,
    )
    scores: dict[str, float] = {}
    for action in ACTION_NAMES:
        if not bool(action_mask.get(action, False)):
            continue
        scores[action] = _round(
            float(bias[action])
            + float(fixture_delta[action])
            + float(fixture_context_delta[action])
            + float(recovery_phase_delta[action])
            + _dot(weights[action], hidden)
        )
    return scores


def is_horizon_fixture_neural_artifact(artifact: Mapping[str, object]) -> bool:
    return artifact.get("model_type") == MIND_V3_HORIZON_FIXTURE_MODEL_TYPE


def mind_v3_neural_head_predictions(
    *,
    artifact: Mapping[str, object],
    observation_input: Mapping[str, object],
) -> dict[str, object]:
    compiled = _compiled_artifact(artifact)
    values = ecological_policy_input_values(dict(observation_input))
    hidden = _hidden_activations(
        values,
        compiled["hidden_weights"],  # type: ignore[arg-type]
        compiled["hidden_bias"],  # type: ignore[arg-type]
    )
    return {
        "survival": _head_predictions(compiled["survival_heads"], hidden),
        "reproduction": _head_predictions(compiled["reproduction_heads"], hidden),
    }


def load_mind_v3_neural_artifact(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise MindV3NeuralArtifactError(
            f"failed to read Mind v3 neural artifact: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MindV3NeuralArtifactError(
            f"Mind v3 neural artifact is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise MindV3NeuralArtifactError("Mind v3 neural artifact must be an object")
    validate_mind_v3_neural_artifact(payload)
    return payload


def write_mind_v3_neural_artifact(
    artifact: Mapping[str, object],
    output_path: str | Path,
) -> None:
    validate_mind_v3_neural_artifact(artifact)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(
            artifact,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")


def validate_mind_v3_neural_artifact(artifact: Mapping[str, object]) -> None:
    if artifact.get("schema_version") != MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION:
        raise MindV3NeuralArtifactError("Mind v3 neural artifact has stale schema")
    model_type = artifact.get("model_type")
    if model_type not in {
        MIND_V3_NEURAL_MODEL_TYPE,
        MIND_V3_HORIZON_FIXTURE_MODEL_TYPE,
    }:
        raise MindV3NeuralArtifactError("Mind v3 neural artifact has wrong model_type")
    if artifact.get("input_policy") != MIND_V3_NEURAL_INPUT_POLICY:
        raise MindV3NeuralArtifactError("Mind v3 neural artifact has wrong input_policy")
    hidden_units = _required_positive_int(artifact.get("hidden_units"), "hidden_units")
    input_contract = artifact.get("input_contract")
    if not isinstance(input_contract, Mapping):
        raise MindV3NeuralArtifactError("Mind v3 neural artifact lacks input_contract")
    if input_contract.get("schema_version") != ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION:
        raise MindV3NeuralArtifactError(
            "Mind v3 neural artifact input_contract has stale schema"
        )
    if int(input_contract.get("ecological_vector_size", -1)) != (
        ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
    ):
        raise MindV3NeuralArtifactError(
            "Mind v3 neural artifact input_contract has wrong vector size"
        )
    _matrix(
        artifact.get("hidden_weights"),
        rows=hidden_units,
        columns=ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        field="hidden_weights",
    )
    _vector(artifact.get("hidden_bias"), length=hidden_units, field="hidden_bias")
    _action_matrix(
        artifact.get("action_output_weights"),
        length=hidden_units,
        field="action_output_weights",
    )
    _action_vector(artifact.get("action_output_bias"), field="action_output_bias")
    if model_type == MIND_V3_HORIZON_FIXTURE_MODEL_TYPE:
        _action_matrix(
            artifact.get("action_value_weights"),
            length=hidden_units,
            field="action_value_weights",
        )
        _action_vector(artifact.get("action_value_bias"), field="action_value_bias")
    elif "action_value_weights" in artifact:
        _action_matrix(
            artifact.get("action_value_weights"),
            length=hidden_units,
            field="action_value_weights",
        )
    if "action_value_bias" in artifact:
        _action_vector(artifact.get("action_value_bias"), field="action_value_bias")
    _action_vector(
        artifact.get("fixture_action_bias_delta"),
        field="fixture_action_bias_delta",
    )
    if "fixture_context_bias" in artifact:
        _fixture_context_bias_payload(artifact.get("fixture_context_bias"))
    if "recovery_phase_action_bias" in artifact:
        _recovery_phase_action_bias_payload(
            artifact.get("recovery_phase_action_bias")
        )
    if "neural_residual_scale" in artifact:
        _nonnegative_finite_float(
            artifact.get("neural_residual_scale"),
            "neural_residual_scale",
        )
    if "neural_residual_max_linear_override_margin" in artifact:
        _nonnegative_finite_float(
            artifact.get("neural_residual_max_linear_override_margin"),
            "neural_residual_max_linear_override_margin",
        )
    if "neural_residual_context_gate" in artifact:
        _neural_residual_context_gate_value(
            artifact.get("neural_residual_context_gate")
        )
    if "neural_residual_recovery_phase_ticks" in artifact:
        _required_non_negative_int(
            artifact.get("neural_residual_recovery_phase_ticks"),
            "neural_residual_recovery_phase_ticks",
        )
    _head_mapping(
        artifact.get("survival_heads"),
        hidden_units=hidden_units,
        field="survival_heads",
    )
    _head_mapping(
        artifact.get("reproduction_heads"),
        hidden_units=hidden_units,
        field="reproduction_heads",
    )


def _artifact_mode_config(artifact_mode: str) -> dict[str, str]:
    mode = str(artifact_mode).strip()
    if mode == MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED:
        return {
            "artifact_mode": MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
            "model_type": MIND_V3_NEURAL_MODEL_TYPE,
            "trainer": MIND_V3_NEURAL_TRAINER,
            "architecture": MIND_V3_NEURAL_ARCHITECTURE,
        }
    if mode == MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE:
        return {
            "artifact_mode": MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
            "model_type": MIND_V3_HORIZON_FIXTURE_MODEL_TYPE,
            "trainer": MIND_V3_HORIZON_FIXTURE_TRAINER,
            "architecture": MIND_V3_HORIZON_FIXTURE_ARCHITECTURE,
        }
    raise MindV3NeuralArtifactError(f"unsupported Mind v3 artifact mode: {mode}")


def _neural_residual_config(
    *,
    neural_residual_scale: float | None,
    neural_residual_max_linear_override_margin: float | None,
    neural_residual_context_gate: str | None,
    neural_residual_recovery_phase_ticks: int | None,
) -> dict[str, object]:
    config: dict[str, object] = {}
    if neural_residual_scale is not None:
        config["neural_residual_scale"] = _round(
            _nonnegative_finite_float(
                neural_residual_scale,
                "neural_residual_scale",
            )
        )
    if neural_residual_max_linear_override_margin is not None:
        config["neural_residual_max_linear_override_margin"] = _round(
            _nonnegative_finite_float(
                neural_residual_max_linear_override_margin,
                "neural_residual_max_linear_override_margin",
            )
        )
    if neural_residual_context_gate is not None:
        config["neural_residual_context_gate"] = _neural_residual_context_gate_value(
            neural_residual_context_gate
        )
    if neural_residual_recovery_phase_ticks is not None:
        config["neural_residual_recovery_phase_ticks"] = (
            _required_non_negative_int(
                neural_residual_recovery_phase_ticks,
                "neural_residual_recovery_phase_ticks",
            )
        )
    return config


def _validate_horizon_label_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_HORIZON_LABEL_SCHEMA_VERSION:
        raise MindV3NeuralArtifactError("horizon labels have stale schema_version")
    labels = report.get("labels")
    if not isinstance(labels, list) or not labels:
        raise MindV3NeuralArtifactError("horizon label report must include labels")


def _validate_fixture_label_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_FIXTURE_LABEL_SCHEMA_VERSION:
        raise MindV3NeuralArtifactError("fixture labels have stale schema_version")


def _horizon_labels_by_source_index(
    report: Mapping[str, object],
) -> dict[int, Mapping[str, object]]:
    labels = report.get("labels")
    if not isinstance(labels, list):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        index = label.get("source_record_index")
        if isinstance(index, int) and not isinstance(index, bool):
            result[int(index)] = label
    return result


def _training_samples(
    records: Sequence[Mapping[str, object]],
    labels_by_index: Mapping[int, Mapping[str, object]],
    *,
    source_weight_by_index: Mapping[int, float] | None = None,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for source_index, record in enumerate(records):
        label = labels_by_index.get(source_index)
        if label is None:
            continue
        observation_input = record.get("observation_input")
        if not isinstance(observation_input, Mapping):
            continue
        action = _record_action(record)
        if action not in ACTION_NAMES:
            continue
        values = ecological_policy_input_values(dict(observation_input))
        horizons = _observed_horizon_payloads(label)
        if not horizons:
            continue
        horizon_weight = _sample_weight(horizons)
        horizon_utility = _horizon_fixture_utility(horizons)
        trajectory_weight = (
            float(source_weight_by_index.get(source_index, 1.0))
            if source_weight_by_index is not None
            else 1.0
        )
        samples.append(
            {
                "source_record_index": source_index,
                "values": values,
                "action": action,
                "weight": _round(horizon_weight * trajectory_weight),
                "horizon_weight": horizon_weight,
                "horizon_utility": horizon_utility,
                "trajectory_weight": trajectory_weight,
                "horizons": horizons,
            }
        )
    return samples


def _trajectory_weight_multipliers(
    datasets: Sequence[TrajectoryJsonlDataset],
    supplied: Sequence[float] | None,
) -> tuple[float, ...]:
    if supplied is None:
        return tuple(1.0 for _dataset in datasets)
    if len(supplied) != len(datasets):
        raise MindV3NeuralArtifactError(
            "trajectory_weight_multipliers length must match datasets"
        )
    weights: list[float] = []
    for index, value in enumerate(supplied):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MindV3NeuralArtifactError(
                f"trajectory_weight_multipliers[{index}] must be finite"
            )
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise MindV3NeuralArtifactError(
                f"trajectory_weight_multipliers[{index}] must be positive"
            )
        weights.append(_round(parsed))
    return tuple(weights)


def _source_weight_by_index(
    datasets: Sequence[TrajectoryJsonlDataset],
    trajectory_weights: Sequence[float],
) -> dict[int, float]:
    source_weights: dict[int, float] = {}
    source_index = 0
    for dataset, weight in zip(datasets, trajectory_weights, strict=True):
        for _record in dataset.records:
            source_weights[source_index] = float(weight)
            source_index += 1
    return source_weights


def _record_action(record: Mapping[str, object]) -> str:
    requested = record.get("requested_action")
    resolved = record.get("resolved_action")
    action_valid = record.get("action_valid", record.get("resolution_action_valid"))
    if isinstance(requested, str) and bool(action_valid):
        return requested
    if isinstance(resolved, str):
        return resolved
    return str(requested) if isinstance(requested, str) else ""


def _observed_horizon_payloads(
    label: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    horizons = label.get("horizons")
    if not isinstance(horizons, Mapping):
        return {}
    return {
        str(key): payload
        for key, payload in horizons.items()
        if isinstance(payload, Mapping) and bool(payload.get("observed", False))
    }


def _sample_weight(horizons: Mapping[str, Mapping[str, object]]) -> float:
    components = []
    for payload in horizons.values():
        survived = 1.0 if payload.get("survived") is True else 0.0
        reproduced = 1.0 if payload.get("reproduced") is True else 0.0
        viability = payload.get("viability")
        balanced = (
            _optional_float(viability.get("balanced_core_min"))
            if isinstance(viability, Mapping)
            else None
        )
        animal = payload.get("animal_resource")
        animal_bonus = (
            0.35
            if isinstance(animal, Mapping)
            and animal.get("survived_after_first_contact") is True
            else 0.0
        )
        components.append(
            0.25
            + 0.75 * survived
            + 1.5 * reproduced
            + 0.5 * (balanced if balanced is not None else 0.0)
            + animal_bonus
        )
    if not components:
        return 0.25
    return _round(max(0.05, sum(components) / float(len(components))))


def _horizon_fixture_utility(
    horizons: Mapping[str, Mapping[str, object]],
) -> float:
    utilities = []
    for payload in horizons.values():
        survived = 1.0 if payload.get("survived") is True else -1.0
        reproduced = 1.0 if payload.get("reproduced") is True else 0.0
        viability = payload.get("viability")
        viability_payload = viability if isinstance(viability, Mapping) else {}
        balanced = _optional_float(viability_payload.get("balanced_core_min"))
        matched = viability_payload.get("matched_diet")
        matched_score = 0.0
        if matched is True:
            matched_score = 1.0
        elif matched is False:
            matched_score = -1.0
        animal = payload.get("animal_resource")
        animal_payload = animal if isinstance(animal, Mapping) else {}
        consumed_animal = animal_payload.get("animal_resource_consumed") is True
        survived_after_contact = animal_payload.get("survived_after_first_contact")
        animal_score = 0.0
        if consumed_animal:
            animal_score += 0.65
            animal_score += 0.55 if survived_after_contact is True else -0.35
        utilities.append(
            0.90 * survived
            + 1.20 * reproduced
            + 0.75 * (balanced if balanced is not None else 0.0)
            + 0.30 * matched_score
            + animal_score
        )
    if not utilities:
        return 0.0
    return _round(sum(utilities) / float(len(utilities)))


def _deterministic_hidden_weights(*, hidden_units: int, seed: int) -> list[list[float]]:
    return [
        [
            _round(
                0.035
                * math.sin(
                    (seed + 1) * 0.019
                    + (unit + 1) * 0.073
                    + (index + 1) * 0.011
                )
                + 0.015
                * math.cos(
                    (seed + 1) * 0.007
                    + (unit + 1) * 0.031
                    + (index + 1) * 0.017
                )
            )
            for index in range(ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE)
        ]
        for unit in range(hidden_units)
    ]


def _deterministic_hidden_bias(*, hidden_units: int, seed: int) -> list[float]:
    return [
        _round(0.04 * math.sin((seed + unit + 1) * 0.13))
        for unit in range(hidden_units)
    ]


def _hidden_activations(
    values: Sequence[float],
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> list[float]:
    return [
        math.tanh(_dot(row, values) + float(bias[index]))
        for index, row in enumerate(weights)
    ]


def _weighted_mean_hidden(samples: Sequence[Mapping[str, object]]) -> list[float]:
    if not samples:
        return []
    hidden_units = len(samples[0]["hidden"])  # type: ignore[arg-type]
    total = [0.0] * hidden_units
    weight_total = 0.0
    for sample in samples:
        weight = float(sample["weight"])
        hidden = sample["hidden"]
        for index, value in enumerate(hidden):  # type: ignore[assignment]
            total[index] += weight * float(value)
        weight_total += weight
    if weight_total <= 0.0:
        return [0.0] * hidden_units
    return [_round(value / weight_total) for value in total]


def _weighted_mean_hidden_with_weights(
    samples: Sequence[Mapping[str, object]],
    weights: Sequence[float],
    *,
    hidden_units: int,
) -> list[float]:
    total = [0.0] * hidden_units
    weight_total = 0.0
    for sample, sample_weight in zip(samples, weights, strict=True):
        weight = max(0.0, float(sample_weight))
        if weight <= 0.0:
            continue
        hidden = sample["hidden"]
        for index, value in enumerate(hidden):  # type: ignore[assignment]
            total[index] += weight * float(value)
        weight_total += weight
    if weight_total <= 0.0:
        return [0.0] * hidden_units
    return [_round(value / weight_total) for value in total]


def _weighted_mean_sample_utility(samples: Sequence[Mapping[str, object]]) -> float:
    total = 0.0
    weight_total = 0.0
    for sample in samples:
        weight = float(sample["weight"])
        total += weight * float(sample.get("horizon_utility", 0.0))
        weight_total += weight
    if weight_total <= 0.0:
        return 0.0
    return _round(total / weight_total)


def _action_output_weights(
    samples: Sequence[Mapping[str, object]],
    *,
    global_mean: Sequence[float],
    hidden_units: int,
) -> dict[str, list[float]]:
    weights: dict[str, list[float]] = {}
    for action in ACTION_NAMES:
        action_samples = [sample for sample in samples if sample["action"] == action]
        if not action_samples:
            weights[action] = [0.0] * hidden_units
            continue
        mean = _weighted_mean_hidden(action_samples)
        weights[action] = [
            _round(
                MIND_V3_NEURAL_PROTOTYPE_WEIGHT_SCALE
                * (mean[index] - global_mean[index])
            )
            for index in range(hidden_units)
        ]
    return weights


def _action_output_bias(samples: Sequence[Mapping[str, object]]) -> dict[str, float]:
    totals = Counter()
    total = 0.0
    for sample in samples:
        weight = float(sample["weight"])
        totals[str(sample["action"])] += weight
        total += weight
    denominator = total + 0.1 * len(ACTION_NAMES)
    return {
        action: _round(
            MIND_V3_NEURAL_ACTION_PRIOR_LOG_WEIGHT
            * math.log((float(totals.get(action, 0.0)) + 0.1) / denominator)
        )
        for action in ACTION_NAMES
    }


def _action_value_weights(
    samples: Sequence[Mapping[str, object]],
    *,
    global_mean: Sequence[float],
    global_utility: float,
    hidden_units: int,
) -> dict[str, list[float]]:
    weights: dict[str, list[float]] = {}
    for action in ACTION_NAMES:
        action_samples = [sample for sample in samples if sample["action"] == action]
        if not action_samples:
            weights[action] = [0.0] * hidden_units
            continue
        positive_weights = [
            float(sample["weight"])
            * max(0.0, float(sample.get("horizon_utility", 0.0)) - global_utility)
            for sample in action_samples
        ]
        negative_weights = [
            float(sample["weight"])
            * max(0.0, global_utility - float(sample.get("horizon_utility", 0.0)))
            for sample in action_samples
        ]
        if sum(positive_weights) <= 0.0:
            positive_weights = [float(sample["weight"]) for sample in action_samples]
        if sum(negative_weights) <= 0.0:
            negative_weights = [float(sample["weight"]) for sample in action_samples]
        positive_mean = _weighted_mean_hidden_with_weights(
            action_samples,
            positive_weights,
            hidden_units=hidden_units,
        )
        negative_mean = _weighted_mean_hidden_with_weights(
            action_samples,
            negative_weights,
            hidden_units=hidden_units,
        )
        action_mean = _weighted_mean_hidden(action_samples)
        weights[action] = [
            _round(
                MIND_V3_HORIZON_FIXTURE_VALUE_WEIGHT_SCALE
                * (positive_mean[index] - negative_mean[index])
                + 0.20 * (action_mean[index] - global_mean[index])
            )
            for index in range(hidden_units)
        ]
    return weights


def _action_value_bias(
    samples: Sequence[Mapping[str, object]],
    *,
    global_utility: float,
) -> dict[str, float]:
    totals = Counter()
    utility_totals = Counter()
    total = 0.0
    for sample in samples:
        weight = float(sample["weight"])
        action = str(sample["action"])
        totals[action] += weight
        utility_totals[action] += weight * float(sample.get("horizon_utility", 0.0))
        total += weight
    denominator = total + 0.1 * len(ACTION_NAMES)
    biases: dict[str, float] = {}
    for action in ACTION_NAMES:
        action_weight = float(totals.get(action, 0.0))
        action_utility = (
            float(utility_totals.get(action, 0.0)) / action_weight
            if action_weight > 0.0
            else global_utility - 0.75
        )
        prior = math.log((action_weight + 0.1) / denominator)
        biases[action] = _round(
            MIND_V3_HORIZON_FIXTURE_BIAS_SCALE
            * (action_utility - global_utility)
            + MIND_V3_HORIZON_FIXTURE_ACTION_PRIOR_LOG_WEIGHT * prior
        )
    return biases


def _horizon_heads(
    samples: Sequence[Mapping[str, object]],
    *,
    hidden_units: int,
    horizon_ticks: Sequence[int],
) -> tuple[dict[str, object], dict[str, object]]:
    horizon_keys = sorted(
        {
            str(key)
            for sample in samples
            for key in dict(sample["horizons"]).keys()  # type: ignore[arg-type]
        }
        | {str(horizon) for horizon in horizon_ticks},
        key=lambda value: int(value),
    )
    survival: dict[str, object] = {}
    reproduction: dict[str, object] = {}
    for horizon_key in horizon_keys:
        survival[horizon_key] = _binary_head(
            samples,
            horizon_key=horizon_key,
            target="survived",
            hidden_units=hidden_units,
        )
        reproduction[horizon_key] = _binary_head(
            samples,
            horizon_key=horizon_key,
            target="reproduced",
            hidden_units=hidden_units,
        )
    return survival, reproduction


def _binary_head(
    samples: Sequence[Mapping[str, object]],
    *,
    horizon_key: str,
    target: str,
    hidden_units: int,
) -> dict[str, object]:
    positive: list[Mapping[str, object]] = []
    negative: list[Mapping[str, object]] = []
    for sample in samples:
        horizons = sample["horizons"]
        payload = dict(horizons).get(horizon_key)  # type: ignore[arg-type]
        if not isinstance(payload, Mapping):
            continue
        if payload.get(target) is True:
            positive.append(sample)
        elif payload.get(target) is False:
            negative.append(sample)
    pos_mean = _weighted_mean_hidden(positive) if positive else [0.0] * hidden_units
    neg_mean = _weighted_mean_hidden(negative) if negative else [0.0] * hidden_units
    count = len(positive) + len(negative)
    probability = (len(positive) + 0.5) / float(count + 1) if count else 0.5
    return {
        "positive_count": len(positive),
        "negative_count": len(negative),
        "weights": [
            _round(0.6 * (pos_mean[index] - neg_mean[index]))
            for index in range(hidden_units)
        ],
        "bias": _round(_logit(probability)),
    }


def _fixture_action_bias_delta(
    fixture_label_report: Mapping[str, object] | None,
) -> dict[str, float]:
    deltas = {action: 0.0 for action in ACTION_NAMES}
    pressures = _fixture_pressures(fixture_label_report)
    carrion_pressure = pressures["carrion_pressure"]
    mixed_birth_pressure = pressures["mixed_birth_pressure"]

    carrion_scale = min(0.25, 0.03 * carrion_pressure)
    mixed_scale = min(0.18, 0.04 * mixed_birth_pressure)
    if carrion_scale:
        deltas["eat"] += 0.45 * carrion_scale
        deltas["drink"] += 0.35 * carrion_scale
        for action in ("move_north", "move_south", "move_east", "move_west"):
            deltas[action] += 0.25 * carrion_scale
        for action in ("attack_north", "attack_south", "attack_east", "attack_west"):
            deltas[action] -= 0.10 * carrion_scale
        deltas["mate"] -= 0.20 * carrion_scale
    if mixed_scale:
        deltas["mate"] += 0.55 * mixed_scale
        deltas["signal_0_profile_0"] += 0.20 * mixed_scale
        deltas["signal_0_profile_1"] += 0.10 * mixed_scale
    return {action: _round(value) for action, value in deltas.items()}


def _fixture_context_bias(
    fixture_label_report: Mapping[str, object] | None,
) -> dict[str, object]:
    pressures = _fixture_pressures(fixture_label_report)
    return {
        "policy": MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_BIAS_POLICY,
        "carrion_pressure": _round(pressures["carrion_pressure"]),
        "mixed_birth_pressure": _round(pressures["mixed_birth_pressure"]),
        "max_scale": MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_MAX_SCALE,
    }


def _fixture_pressures(
    fixture_label_report: Mapping[str, object] | None,
) -> dict[str, float]:
    if fixture_label_report is None:
        return {"carrion_pressure": 0.0, "mixed_birth_pressure": 0.0}
    carrion_pressure = 0.0
    mixed_birth_pressure = 0.0
    labels = fixture_label_report.get("labels")
    for label in labels if isinstance(labels, list) else []:
        if not isinstance(label, Mapping) or bool(label.get("passed", False)):
            continue
        pressure = _optional_float(label.get("pressure")) or 0.0
        fixture = str(label.get("fixture", ""))
        reason = str(label.get("reason", ""))
        if fixture == "carrion_only":
            carrion_pressure += pressure
        if fixture == "mixed_stable" and "birth" in reason:
            mixed_birth_pressure += pressure
    return {
        "carrion_pressure": carrion_pressure,
        "mixed_birth_pressure": mixed_birth_pressure,
    }


def _fixture_context_action_bias(
    fixture_context_bias: Mapping[str, object],
    values: Sequence[float],
) -> dict[str, float]:
    deltas = {action: 0.0 for action in ACTION_NAMES}
    carrion_pressure = _optional_float(
        fixture_context_bias.get("carrion_pressure")
    ) or 0.0
    if carrion_pressure <= 0.0:
        return deltas
    max_scale = _optional_float(fixture_context_bias.get("max_scale"))
    if max_scale is None:
        max_scale = MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_MAX_SCALE
    scale = min(max(0.0, max_scale), 0.015 * carrion_pressure)
    if scale <= 0.0:
        return deltas

    energy_need = 1.0 - _unit_value(_self_value(values, "energy_ratio"))
    hydration_need = 1.0 - _unit_value(_self_value(values, "hydration_ratio"))
    matched_need = 1.0 - _unit_value(_self_value(values, "matched_diet_ratio"))
    local_carrion = max(
        _positive_value(_center_patch_value(values, "fresh_kill_energy")),
        _positive_value(_center_patch_value(values, "carcass_energy")),
    )
    carrion_dx = _navigation_value(values, "carrion", "dx")
    carrion_dy = _navigation_value(values, "carrion", "dy")
    carrion_strength = _positive_value(
        _navigation_value(values, "carrion", "strength")
    )
    water_dx = _navigation_value(values, "water", "dx")
    water_dy = _navigation_value(values, "water", "dy")
    water_strength = _positive_value(_navigation_value(values, "water", "strength"))
    carrion_need = max(energy_need, matched_need) * max(carrion_strength, local_carrion)
    water_need = hydration_need * water_strength
    local_eat_need = max(energy_need, matched_need) * local_carrion

    deltas["eat"] += 0.80 * scale * local_eat_need
    deltas["drink"] += 0.75 * scale * hydration_need * max(water_strength, 0.25)
    deltas["stay"] -= 0.35 * scale * max(carrion_need, water_need)
    for action in ("move_north", "move_south", "move_east", "move_west"):
        deltas[action] += scale * (
            0.65 * carrion_need * _move_alignment(action, carrion_dx, carrion_dy)
            + 0.75 * water_need * _move_alignment(action, water_dx, water_dy)
        )
    for action in ("attack_north", "attack_south", "attack_east", "attack_west"):
        deltas[action] -= 0.10 * scale * max(carrion_need, water_need)
    deltas["mate"] -= 0.20 * scale * max(carrion_need, water_need)
    return {action: _round(value) for action, value in deltas.items()}


def _score_horizon_fixture_artifact(
    *,
    compiled: Mapping[str, object],
    values: Sequence[float],
    hidden: Sequence[float],
    action_mask: Mapping[str, bool],
) -> dict[str, float]:
    value_weights = compiled["action_value_weights"]
    value_bias = compiled["action_value_bias"]
    support_weights = compiled["action_output_weights"]
    support_bias = compiled["action_output_bias"]
    fixture_delta = compiled["fixture_action_bias_delta"]
    fixture_context_delta = _fixture_context_action_bias(
        compiled["fixture_context_bias"],  # type: ignore[arg-type]
        values,
    )
    value_scores: dict[str, float] = {}
    support_scores: dict[str, float] = {}
    for action in ACTION_NAMES:
        if not bool(action_mask.get(action, False)):
            continue
        value_scores[action] = _round(
            float(value_bias[action]) + _dot(value_weights[action], hidden)
        )
        support_scores[action] = _round(
            float(support_bias[action]) + _dot(support_weights[action], hidden)
        )
    value_normalized = _centered_legal_scores(value_scores)
    support_normalized = _centered_legal_scores(support_scores)
    return {
        action: _round(
            value_normalized[action]
            + MIND_V3_HORIZON_FIXTURE_BEHAVIOR_SUPPORT_WEIGHT
            * support_normalized[action]
            + float(fixture_delta[action])
            + float(fixture_context_delta[action])
        )
        for action in value_scores
    }


def _centered_legal_scores(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mean = sum(float(value) for value in scores.values()) / float(len(scores))
    centered = {
        action: _finite_value(float(value) - mean)
        for action, value in scores.items()
    }
    scale = max((abs(value) for value in centered.values()), default=0.0)
    if scale <= 1e-9:
        return {action: 0.0 for action in scores}
    return {action: _round(value / scale) for action, value in centered.items()}


def _move_alignment(action: str, dx: float, dy: float) -> float:
    if action == "move_east":
        return _positive_value(dx)
    if action == "move_west":
        return _positive_value(-dx)
    if action == "move_south":
        return _positive_value(dy)
    if action == "move_north":
        return _positive_value(-dy)
    return 0.0


def _fixture_context_bias_payload(payload: object) -> dict[str, object]:
    if payload is None:
        return _fixture_context_bias(None)
    if not isinstance(payload, Mapping):
        raise MindV3NeuralArtifactError("fixture_context_bias must be an object")
    return {
        "policy": str(
            payload.get("policy", MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_BIAS_POLICY)
        ),
        "carrion_pressure": _finite_float(
            payload.get("carrion_pressure", 0.0),
            "fixture_context_bias.carrion_pressure",
        ),
        "mixed_birth_pressure": _finite_float(
            payload.get("mixed_birth_pressure", 0.0),
            "fixture_context_bias.mixed_birth_pressure",
        ),
        "max_scale": _finite_float(
            payload.get(
                "max_scale",
                MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_MAX_SCALE,
            ),
            "fixture_context_bias.max_scale",
        ),
    }


def _recovery_phase_action_bias_payload(payload: object) -> dict[str, object]:
    if payload is None:
        return {
            "policy": MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY,
            "scale": 0.0,
            "max_abs_bias": 0.0,
            "action_bias": {action: 0.0 for action in ACTION_NAMES},
            "survivor_action_weight": {action: 0.0 for action in ACTION_NAMES},
            "failure_action_weight": {action: 0.0 for action in ACTION_NAMES},
            "survivor_total_weight": 0.0,
            "failure_total_weight": 0.0,
            "record_count": 0,
        }
    if not isinstance(payload, Mapping):
        raise MindV3NeuralArtifactError("recovery_phase_action_bias must be an object")
    policy = str(payload.get("policy", "")).strip()
    if policy != MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY:
        raise MindV3NeuralArtifactError(
            "recovery_phase_action_bias has unsupported policy"
        )
    return {
        "policy": policy,
        "scale": _round(
            _nonnegative_finite_float(
                payload.get("scale", 0.0),
                "recovery_phase_action_bias.scale",
            )
        ),
        "max_abs_bias": _round(
            _nonnegative_finite_float(
                payload.get("max_abs_bias", 0.0),
                "recovery_phase_action_bias.max_abs_bias",
            )
        ),
        "action_bias": _action_vector(
            payload.get("action_bias"),
            field="recovery_phase_action_bias.action_bias",
        ),
        "survivor_action_weight": _action_weight_vector(
            payload.get("survivor_action_weight"),
            field="recovery_phase_action_bias.survivor_action_weight",
        ),
        "failure_action_weight": _action_weight_vector(
            payload.get("failure_action_weight"),
            field="recovery_phase_action_bias.failure_action_weight",
        ),
        "survivor_total_weight": _round(
            _nonnegative_finite_float(
                payload.get("survivor_total_weight", 0.0),
                "recovery_phase_action_bias.survivor_total_weight",
            )
        ),
        "failure_total_weight": _round(
            _nonnegative_finite_float(
                payload.get("failure_total_weight", 0.0),
                "recovery_phase_action_bias.failure_total_weight",
            )
        ),
        "record_count": _required_non_negative_int(
            payload.get("record_count", 0),
            "recovery_phase_action_bias.record_count",
        ),
    }


def _recovery_phase_action_bias_scores(
    payload: Mapping[str, object],
    *,
    recovery_phase_remaining: int,
    action_mask: Mapping[str, bool],
) -> dict[str, float]:
    if int(recovery_phase_remaining) <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    action_bias = payload.get("action_bias")
    bias = action_bias if isinstance(action_bias, Mapping) else {}
    return {
        action: _round(float(bias.get(action, 0.0)))
        if bool(action_mask.get(action, False))
        else 0.0
        for action in ACTION_NAMES
    }


def _action_weight_vector(payload: object, *, field: str) -> dict[str, float]:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise MindV3NeuralArtifactError(f"{field} must be an object")
    return {
        action: _round(
            _nonnegative_finite_float(payload.get(action, 0.0), f"{field}.{action}")
        )
        for action in ACTION_NAMES
    }


def _self_value(values: Sequence[float], field: str) -> float:
    try:
        return _finite_value(values[_ECOLOGICAL_SELF_FIELDS.index(field)])
    except (ValueError, IndexError):
        return 0.0


def _center_patch_value(values: Sequence[float], field: str) -> float:
    try:
        index = (
            _ECOLOGICAL_PATCH_START
            + (PATCH_CELL_COUNT // 2) * len(PATCH_INPUT_FIELDS)
            + PATCH_INPUT_FIELDS.index(field)
        )
        return _finite_value(values[index])
    except (ValueError, IndexError):
        return 0.0


def _navigation_value(values: Sequence[float], target: str, field: str) -> float:
    try:
        index = (
            _ECOLOGICAL_NAVIGATION_START
            + NAVIGATION_TARGETS.index(target) * len(NAVIGATION_INPUT_FIELDS)
            + NAVIGATION_INPUT_FIELDS.index(field)
        )
        return _finite_value(values[index])
    except (ValueError, IndexError):
        return 0.0


def _unit_value(value: float) -> float:
    return min(1.0, max(0.0, _finite_value(value)))


def _positive_value(value: float) -> float:
    return max(0.0, _finite_value(value))


def _finite_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _fixture_pressure_summary(
    fixture_label_report: Mapping[str, object] | None,
) -> dict[str, object]:
    if fixture_label_report is None:
        return {
            "fixture_label_schema_version": None,
            "failed_label_count": 0,
            "pressure_total": 0.0,
            "pressure_by_fixture": {},
        }
    aggregate = fixture_label_report.get("aggregate")
    payload = aggregate if isinstance(aggregate, Mapping) else {}
    pressure_by_fixture = payload.get("pressure_by_fixture")
    return {
        "fixture_label_schema_version": fixture_label_report.get("schema_version"),
        "failed_label_count": int(payload.get("failed_label_count", 0)),
        "pressure_total": _round(_optional_float(payload.get("pressure_total")) or 0.0),
        "pressure_by_fixture": (
            dict(pressure_by_fixture)
            if isinstance(pressure_by_fixture, Mapping)
            else {}
        ),
    }


def _training_summary(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    actions = Counter(str(sample["action"]) for sample in samples)
    weights = [float(sample["weight"]) for sample in samples]
    horizon_weights = [float(sample["horizon_weight"]) for sample in samples]
    trajectory_weights = [float(sample["trajectory_weight"]) for sample in samples]
    return {
        "record_count": len(samples),
        "action_counts": dict(sorted(actions.items())),
        "sample_weight_min": _round(min(weights) if weights else 0.0),
        "sample_weight_max": _round(max(weights) if weights else 0.0),
        "sample_weight_mean": _round(
            sum(weights) / float(len(weights)) if weights else 0.0
        ),
        "horizon_weight_min": _round(
            min(horizon_weights) if horizon_weights else 0.0
        ),
        "horizon_weight_max": _round(
            max(horizon_weights) if horizon_weights else 0.0
        ),
        "horizon_weight_mean": _round(
            sum(horizon_weights) / float(len(horizon_weights))
            if horizon_weights
            else 0.0
        ),
        "trajectory_weight_min": _round(
            min(trajectory_weights) if trajectory_weights else 0.0
        ),
        "trajectory_weight_max": _round(
            max(trajectory_weights) if trajectory_weights else 0.0
        ),
        "trajectory_weight_mean": _round(
            sum(trajectory_weights) / float(len(trajectory_weights))
            if trajectory_weights
            else 0.0
        ),
    }


def _action_value_summary(
    samples: Sequence[Mapping[str, object]],
    *,
    global_utility: float,
) -> dict[str, object]:
    utilities = [float(sample.get("horizon_utility", 0.0)) for sample in samples]
    action_utility_totals = Counter()
    action_weight_totals = Counter()
    for sample in samples:
        action = str(sample["action"])
        weight = float(sample["weight"])
        action_utility_totals[action] += weight * float(
            sample.get("horizon_utility", 0.0)
        )
        action_weight_totals[action] += weight
    return {
        "policy": MIND_V3_HORIZON_FIXTURE_SCORE_POLICY,
        "global_utility_mean": _round(global_utility),
        "utility_min": _round(min(utilities) if utilities else 0.0),
        "utility_max": _round(max(utilities) if utilities else 0.0),
        "utility_mean": _round(
            sum(utilities) / float(len(utilities)) if utilities else 0.0
        ),
        "action_utility_mean": {
            action: _round(
                float(action_utility_totals.get(action, 0.0))
                / float(action_weight_totals[action])
            )
            for action in sorted(action_weight_totals)
            if float(action_weight_totals[action]) > 0.0
        },
    }


def _horizon_ticks(report: Mapping[str, object]) -> list[int]:
    contract = report.get("label_contract")
    ticks = contract.get("horizon_ticks") if isinstance(contract, Mapping) else None
    if not isinstance(ticks, list):
        return []
    return [
        int(value)
        for value in ticks
        if isinstance(value, int) and not isinstance(value, bool)
    ]


def _compiled_artifact(artifact: Mapping[str, object]) -> dict[str, object]:
    validate_mind_v3_neural_artifact(artifact)
    hidden_units = int(artifact["hidden_units"])
    action_value_weights = artifact.get("action_value_weights")
    action_value_bias = artifact.get("action_value_bias")
    return {
        "model_type": str(artifact.get("model_type")),
        "hidden_weights": _matrix(
            artifact.get("hidden_weights"),
            rows=hidden_units,
            columns=ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
            field="hidden_weights",
        ),
        "hidden_bias": _vector(
            artifact.get("hidden_bias"),
            length=hidden_units,
            field="hidden_bias",
        ),
        "action_output_weights": _action_matrix(
            artifact.get("action_output_weights"),
            length=hidden_units,
            field="action_output_weights",
        ),
        "action_output_bias": _action_vector(
            artifact.get("action_output_bias"),
            field="action_output_bias",
        ),
        "action_value_weights": (
            _action_matrix(
                action_value_weights,
                length=hidden_units,
                field="action_value_weights",
            )
            if isinstance(action_value_weights, Mapping)
            else {action: [0.0] * hidden_units for action in ACTION_NAMES}
        ),
        "action_value_bias": (
            _action_vector(action_value_bias, field="action_value_bias")
            if isinstance(action_value_bias, Mapping)
            else {action: 0.0 for action in ACTION_NAMES}
        ),
        "fixture_action_bias_delta": _action_vector(
            artifact.get("fixture_action_bias_delta"),
            field="fixture_action_bias_delta",
        ),
        "fixture_context_bias": _fixture_context_bias_payload(
            artifact.get("fixture_context_bias")
        ),
        "recovery_phase_action_bias": _recovery_phase_action_bias_payload(
            artifact.get("recovery_phase_action_bias")
        ),
        "survival_heads": _head_mapping(
            artifact.get("survival_heads"),
            hidden_units=hidden_units,
            field="survival_heads",
        ),
        "reproduction_heads": _head_mapping(
            artifact.get("reproduction_heads"),
            hidden_units=hidden_units,
            field="reproduction_heads",
        ),
    }


def _head_predictions(
    heads: Mapping[str, Mapping[str, object]],
    hidden: Sequence[float],
) -> dict[str, float]:
    return {
        key: _round(_sigmoid(float(head["bias"]) + _dot(head["weights"], hidden)))
        for key, head in sorted(heads.items(), key=lambda item: int(item[0]))
    }


def _head_mapping(
    payload: object,
    *,
    hidden_units: int,
    field: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(payload, Mapping):
        raise MindV3NeuralArtifactError(f"{field} must be an object")
    parsed: dict[str, dict[str, object]] = {}
    for key, head in payload.items():
        if not isinstance(key, str) or not key.isdigit():
            raise MindV3NeuralArtifactError(f"{field} keys must be horizon ticks")
        if not isinstance(head, Mapping):
            raise MindV3NeuralArtifactError(f"{field}.{key} must be an object")
        parsed[key] = {
            "positive_count": _required_non_negative_int(
                head.get("positive_count"),
                f"{field}.{key}.positive_count",
            ),
            "negative_count": _required_non_negative_int(
                head.get("negative_count"),
                f"{field}.{key}.negative_count",
            ),
            "weights": _vector(
                head.get("weights"),
                length=hidden_units,
                field=f"{field}.{key}.weights",
            ),
            "bias": _finite_float(head.get("bias"), f"{field}.{key}.bias"),
        }
    return parsed


def _action_matrix(payload: object, *, length: int, field: str) -> dict[str, list[float]]:
    if not isinstance(payload, Mapping):
        raise MindV3NeuralArtifactError(f"{field} must be an object")
    return {
        action: _vector(payload.get(action), length=length, field=f"{field}.{action}")
        for action in ACTION_NAMES
    }


def _action_vector(payload: object, *, field: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise MindV3NeuralArtifactError(f"{field} must be an object")
    return {
        action: _finite_float(payload.get(action), f"{field}.{action}")
        for action in ACTION_NAMES
    }


def _matrix(
    payload: object,
    *,
    rows: int,
    columns: int,
    field: str,
) -> list[list[float]]:
    if not isinstance(payload, list) or len(payload) != rows:
        raise MindV3NeuralArtifactError(f"{field} has invalid row count")
    return [
        _vector(row, length=columns, field=f"{field}[{index}]")
        for index, row in enumerate(payload)
    ]


def _vector(payload: object, *, length: int, field: str) -> list[float]:
    if not isinstance(payload, list) or len(payload) != length:
        raise MindV3NeuralArtifactError(f"{field} has invalid length")
    return [
        _finite_float(value, f"{field}[{index}]")
        for index, value in enumerate(payload)
    ]


def _required_positive_int(value: object, field: str) -> int:
    parsed = _required_non_negative_int(value, field)
    if parsed <= 0:
        raise MindV3NeuralArtifactError(f"{field} must be positive")
    return parsed


def _required_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MindV3NeuralArtifactError(f"{field} must be an integer")
    if value < 0:
        raise MindV3NeuralArtifactError(f"{field} must be non-negative")
    return int(value)


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MindV3NeuralArtifactError(f"{field} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise MindV3NeuralArtifactError(f"{field} must be finite")
    return parsed


def _nonnegative_finite_float(value: object, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed < 0.0:
        raise MindV3NeuralArtifactError(f"{field} must be non-negative")
    return parsed


def _string_value(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MindV3NeuralArtifactError("neural_residual_context_gate must be a string")
    return value.strip()


def _neural_residual_context_gate_value(value: object) -> str:
    gate = _string_value(value)
    if gate not in MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATES:
        raise MindV3NeuralArtifactError(
            f"unsupported neural_residual_context_gate: {gate}"
        )
    return gate


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    parsed = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-parsed))


def _logit(value: float) -> float:
    clamped = max(1e-6, min(1.0 - 1e-6, float(value)))
    return math.log(clamped / (1.0 - clamped))


def _round(value: float) -> float:
    return round(float(value), 6)


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    with _open_input(resolved) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise MindV3NeuralArtifactError(f"report must be a JSON object: {resolved}")
    return payload


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
