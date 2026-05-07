from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Iterable

from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.action_space import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.trajectory import TRAJECTORY_SCHEMA_VERSION
from evolution_sim.mind.contracts import MIND_MODEL_ARTIFACT_VERSION
from evolution_sim.mind.feature_policy import (
    FEATURE_POLICY_VERSION,
    feature_keys_from_record,
)
from evolution_sim.mind.neural import (
    NEURAL_ACTION_VALUE_POLICY,
    NEURAL_ADVANTAGE_BLENDED_ACTOR_PRIOR_POLICY,
    NEURAL_ACTOR_CRITIC_MODEL_TYPE,
    NEURAL_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY,
    NEURAL_ACTOR_CRITIC_TRAINER,
    NEURAL_ARCHITECTURE,
    NEURAL_ACTOR_PRIOR_BLEND_WEIGHT,
    NEURAL_ACTOR_PRIOR_POLICY,
    NEURAL_BACKEND,
    NEURAL_HIDDEN_ACTIVATION,
    NEURAL_HIDDEN_UNITS,
    NEURAL_INPUT_NORMALIZATION,
    NEURAL_SEED,
    NEURAL_STATE_VALUE_POLICY,
    NEURAL_TRAINING_POLICY,
    TORCH_NEURAL_ACTOR_CRITIC_MODEL_TYPE,
    TORCH_NEURAL_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY,
    TORCH_NEURAL_ACTOR_CRITIC_TRAINER,
    TORCH_ADVANTAGE_ACTOR_CRITIC_MODEL_TYPE,
    TORCH_ADVANTAGE_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY,
    TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER,
    TORCH_ADVANTAGE_TRAINING_POLICY,
    TORCH_DISCRETE_IQL_MODEL_TYPE,
    TORCH_DISCRETE_IQL_SAMPLE_WEIGHT_POLICY,
    TORCH_DISCRETE_IQL_TRAINER,
    TORCH_DISCRETE_IQL_TRAINING_POLICY,
    TORCH_NEURAL_ARCHITECTURE,
    TORCH_NEURAL_BACKEND,
    TORCH_NEURAL_HIDDEN_UNITS,
    TORCH_NEURAL_SEED,
    TORCH_NEURAL_TRAINING_POLICY,
    train_neural_actor_critic_network,
)
from evolution_sim.mind.provenance import validate_dataset_provenance
from evolution_sim.mind.torch_trainer import (
    train_torch_actor_critic_network,
    train_torch_advantage_actor_critic_network,
    train_torch_discrete_iql_network,
)

CONTEXTUAL_PRIOR_TRAINER = "contextual-prior"
REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER = "reward-weighted-contextual-prior"
ADVANTAGE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER = (
    "advantage-calibrated-contextual-prior"
)
ADVANTAGE_BLENDED_CONTEXTUAL_PRIOR_TRAINER = (
    "advantage-blended-contextual-prior"
)
VALUE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER = (
    "value-calibrated-contextual-prior"
)
TRAINER_CHOICES: tuple[str, ...] = (
    CONTEXTUAL_PRIOR_TRAINER,
    REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER,
    ADVANTAGE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER,
    ADVANTAGE_BLENDED_CONTEXTUAL_PRIOR_TRAINER,
    VALUE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER,
    NEURAL_ACTOR_CRITIC_TRAINER,
    TORCH_NEURAL_ACTOR_CRITIC_TRAINER,
    TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER,
    TORCH_DISCRETE_IQL_TRAINER,
)
BEHAVIOR_CLONING_BASELINE_MODEL_TYPE = "guarded_contextual_local_prior_bc_v2"
REWARD_WEIGHTED_BASELINE_MODEL_TYPE = (
    "guarded_reward_weighted_contextual_prior_bc_v1"
)
ADVANTAGE_CALIBRATED_BASELINE_MODEL_TYPE = (
    "guarded_advantage_calibrated_contextual_prior_bc_v1"
)
ADVANTAGE_BLENDED_BASELINE_MODEL_TYPE = (
    "guarded_advantage_blended_contextual_prior_bc_v1"
)
VALUE_CALIBRATED_BASELINE_MODEL_TYPE = (
    "guarded_value_calibrated_contextual_prior_bc_v1"
)
NEURAL_ACTOR_CRITIC_BASELINE_MODEL_TYPE = NEURAL_ACTOR_CRITIC_MODEL_TYPE
TORCH_NEURAL_ACTOR_CRITIC_BASELINE_MODEL_TYPE = (
    TORCH_NEURAL_ACTOR_CRITIC_MODEL_TYPE
)
TORCH_ADVANTAGE_ACTOR_CRITIC_BASELINE_MODEL_TYPE = (
    TORCH_ADVANTAGE_ACTOR_CRITIC_MODEL_TYPE
)
TORCH_DISCRETE_IQL_BASELINE_MODEL_TYPE = TORCH_DISCRETE_IQL_MODEL_TYPE
CONDITIONAL_MIN_RECORDS = 3
CONDITIONAL_SCORE_POLICY = "smoothed_contextual_action_prior_v1"
CONDITIONAL_PRIOR_CORRECTION_EXPONENT = 0.0
CONDITIONAL_SCORE_SMOOTHING_ALPHA = 0.1
UNIFORM_SAMPLE_WEIGHT_POLICY = "uniform_v1"
REWARD_TOTAL_SHIFTED_SAMPLE_WEIGHT_POLICY = "reward_total_shifted_clamp_v1"
CONTEXTUAL_REWARD_ADVANTAGE_SAMPLE_WEIGHT_POLICY = (
    "contextual_reward_advantage_adjusted_counts_v1"
)
CONTEXTUAL_REWARD_ADVANTAGE_BLEND_SAMPLE_WEIGHT_POLICY = (
    "contextual_reward_advantage_blended_counts_v1"
)
CONTEXTUAL_VALUE_CALIBRATED_SAMPLE_WEIGHT_POLICY = (
    "contextual_value_calibrated_score_blend_v1"
)
CONTEXTUAL_REWARD_ADVANTAGE_POLICY = "contextual_reward_advantage_lift_v1"
CONTEXTUAL_REWARD_ADVANTAGE_BLEND_POLICY = (
    "contextual_reward_advantage_score_blend_v1"
)
MEAN_REWARD_ACTION_VALUE_POLICY = "mean_reward_action_value_v1"
PRIOR_VALUE_SCORE_BLEND_POLICY = "prior_value_score_blend_v1"
VALUE_SUPPORTED_DEVIATION_POLICY = "positive_value_safe_deviation_v1"
UNIFORM_SAMPLE_WEIGHT_BASE = 1.0
UNIFORM_SAMPLE_WEIGHT_MIN = 1.0
REWARD_TOTAL_SAMPLE_WEIGHT_BASE = 1.0
REWARD_TOTAL_SAMPLE_WEIGHT_MIN = 0.05
ADVANTAGE_SAMPLE_WEIGHT_BASE = 1.0
ADVANTAGE_SAMPLE_WEIGHT_MIN = 0.25
ADVANTAGE_SAMPLE_WEIGHT_MAX = 2.0
ADVANTAGE_REWARD_SCALE = 2.0
ADVANTAGE_MIN_ACTION_SUPPORT = 2
ADVANTAGE_BLEND_WEIGHT = 0.2
VALUE_MIN_ACTION_SUPPORT = 2
VALUE_SCORE_BLEND_WEIGHT = 0.05
VALUE_SCORE_EPSILON = 0.001
VALUE_SUPPORTED_DEVIATION_MIN_SUPPORT = 32
VALUE_SUPPORTED_DEVIATION_MIN_VALUE_MARGIN = 0.04
VALUE_SUPPORTED_DEVIATION_MIN_LEARNED_VALUE = 0.02
VALUE_SUPPORTED_DEVIATION_MIN_SCORE_MARGIN = 0.25
VALUE_SUPPORTED_DEVIATION_MIN_PREDICTED_ADVANTAGE = 0.24
HEURISTIC_CONFIDENCE_THRESHOLD = 0.5
HEURISTIC_OVERRIDE_MIN_MARGIN = 1.0
HEURISTIC_DELEGATE_POLICY = "observation_heuristic_confidence_delegate_v1"
HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN = 0.221
IQL_HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN = 0.25
HEURISTIC_SAFE_LOCAL_EAT_MIN_SCORE = None
HEURISTIC_SAFE_LOCAL_EAT_MIN_FOOD = None
HEURISTIC_SAFE_LOCAL_EAT_MIN_PLANT_RATIO = None
HEURISTIC_SAFE_PLANT_MOVE_MIN_SCORE = None
HEURISTIC_SAFE_PLANT_MOVE_MIN_STRENGTH = None
HEURISTIC_SAFE_PLANT_MOVE_MAX_LOCAL_FOOD_RATIO = None
HEURISTIC_SAFE_PLANT_MOVE_MAX_DISTANCE = None
HEURISTIC_GUARD_POLICY = "observation_heuristic_safety_floor_v1"


@dataclass(frozen=True, slots=True)
class BehaviorCloningBaseline:
    action_scores: dict[str, float]
    action_score_metadata: dict[str, object]
    conditional_action_scores: dict[str, dict[str, float]]
    conditional_action_metadata: dict[str, dict[str, object]]
    record_count: int
    provenance: dict[str, object]
    model_type: str = BEHAVIOR_CLONING_BASELINE_MODEL_TYPE
    trainer: str = CONTEXTUAL_PRIOR_TRAINER
    sample_weight_policy: str = UNIFORM_SAMPLE_WEIGHT_POLICY
    sample_weight_base: float = UNIFORM_SAMPLE_WEIGHT_BASE
    sample_weight_min: float = UNIFORM_SAMPLE_WEIGHT_MIN
    sample_weight_total: float = 0.0
    sample_weight_max: float | None = None
    reward_advantage_policy: str | None = None
    reward_advantage_scale: float | None = None
    reward_advantage_min_action_support: int | None = None
    reward_advantage_blend_policy: str | None = None
    reward_advantage_blend_weight: float | None = None
    action_value_estimates: dict[str, float] | None = None
    conditional_action_value_estimates: dict[str, dict[str, float]] | None = None
    value_estimation_policy: str | None = None
    value_score_blend_policy: str | None = None
    value_score_blend_weight: float | None = None
    value_min_action_support: int | None = None
    value_score_epsilon: float | None = None
    value_supported_deviation_policy: str | None = None
    value_supported_deviation_min_support: int | None = None
    value_supported_deviation_min_value_margin: float | None = None
    value_supported_deviation_min_learned_value: float | None = None
    value_supported_deviation_min_score_margin: float | None = None
    value_supported_deviation_min_predicted_advantage: float | None = None
    neural_network: dict[str, object] | None = None
    neural_backend: str | None = None
    neural_architecture: str | None = None
    neural_training_policy: str | None = None
    neural_input_size: int | None = None
    neural_hidden_units: int | None = None
    neural_hidden_activation: str | None = None
    neural_input_normalization: str | None = None
    neural_seed: int | None = None
    neural_state_value_policy: str | None = None
    neural_actor_prior_policy: str | None = None
    neural_actor_prior_blend_weight: float | None = None
    heuristic_delegate_max_training_score_margin: float = (
        HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN
    )

    def to_artifact(self) -> dict[str, object]:
        provenance = validate_dataset_provenance(self.provenance)
        model = {
            "action_scores": dict(sorted(self.action_scores.items())),
            "action_score_metadata": dict(
                sorted(self.action_score_metadata.items())
            ),
            "conditional_action_scores": {
                key: dict(sorted(scores.items()))
                for key, scores in sorted(self.conditional_action_scores.items())
            },
            "conditional_action_metadata": {
                key: dict(sorted(metadata.items()))
                for key, metadata in sorted(
                    self.conditional_action_metadata.items()
                )
            },
            "fallback_action": "stay",
            "feature_policy_version": FEATURE_POLICY_VERSION,
            "conditional_min_records": CONDITIONAL_MIN_RECORDS,
            "conditional_score_policy": CONDITIONAL_SCORE_POLICY,
            "conditional_prior_correction_exponent": (
                CONDITIONAL_PRIOR_CORRECTION_EXPONENT
            ),
            "conditional_score_smoothing_alpha": (
                CONDITIONAL_SCORE_SMOOTHING_ALPHA
            ),
            "trainer": self.trainer,
            "sample_weight_policy": self.sample_weight_policy,
            "sample_weight_base": self.sample_weight_base,
            "sample_weight_min": self.sample_weight_min,
            "sample_weight_total": round(self.sample_weight_total, 4),
            "heuristic_guard_policy": HEURISTIC_GUARD_POLICY,
            "heuristic_confidence_threshold": HEURISTIC_CONFIDENCE_THRESHOLD,
            "heuristic_override_min_margin": HEURISTIC_OVERRIDE_MIN_MARGIN,
            "heuristic_delegate_policy": HEURISTIC_DELEGATE_POLICY,
            "heuristic_delegate_max_training_score_margin": (
                self.heuristic_delegate_max_training_score_margin
            ),
            "heuristic_safe_local_eat_min_score": (
                HEURISTIC_SAFE_LOCAL_EAT_MIN_SCORE
            ),
            "heuristic_safe_local_eat_min_food": (
                HEURISTIC_SAFE_LOCAL_EAT_MIN_FOOD
            ),
            "heuristic_safe_local_eat_min_plant_ratio": (
                HEURISTIC_SAFE_LOCAL_EAT_MIN_PLANT_RATIO
            ),
            "heuristic_safe_plant_move_min_score": (
                HEURISTIC_SAFE_PLANT_MOVE_MIN_SCORE
            ),
            "heuristic_safe_plant_move_min_strength": (
                HEURISTIC_SAFE_PLANT_MOVE_MIN_STRENGTH
            ),
            "heuristic_safe_plant_move_max_local_food_ratio": (
                HEURISTIC_SAFE_PLANT_MOVE_MAX_LOCAL_FOOD_RATIO
            ),
            "heuristic_safe_plant_move_max_distance": (
                HEURISTIC_SAFE_PLANT_MOVE_MAX_DISTANCE
            ),
        }
        if self.sample_weight_max is not None:
            model["sample_weight_max"] = self.sample_weight_max
        if self.reward_advantage_policy is not None:
            model["reward_advantage_policy"] = self.reward_advantage_policy
        if self.reward_advantage_scale is not None:
            model["reward_advantage_scale"] = self.reward_advantage_scale
        if self.reward_advantage_min_action_support is not None:
            model["reward_advantage_min_action_support"] = (
                self.reward_advantage_min_action_support
            )
        if self.reward_advantage_blend_policy is not None:
            model["reward_advantage_blend_policy"] = (
                self.reward_advantage_blend_policy
            )
        if self.reward_advantage_blend_weight is not None:
            model["reward_advantage_blend_weight"] = (
                self.reward_advantage_blend_weight
            )
        if self.action_value_estimates is not None:
            model["action_value_estimates"] = dict(
                sorted(self.action_value_estimates.items())
            )
        if self.conditional_action_value_estimates is not None:
            model["conditional_action_value_estimates"] = {
                key: dict(sorted(estimates.items()))
                for key, estimates in sorted(
                    self.conditional_action_value_estimates.items()
                )
            }
        if self.value_estimation_policy is not None:
            model["value_estimation_policy"] = self.value_estimation_policy
        if self.value_score_blend_policy is not None:
            model["value_score_blend_policy"] = self.value_score_blend_policy
        if self.value_score_blend_weight is not None:
            model["value_score_blend_weight"] = self.value_score_blend_weight
        if self.value_min_action_support is not None:
            model["value_min_action_support"] = self.value_min_action_support
        if self.value_score_epsilon is not None:
            model["value_score_epsilon"] = self.value_score_epsilon
        if self.value_supported_deviation_policy is not None:
            model["value_supported_deviation_policy"] = (
                self.value_supported_deviation_policy
            )
        if self.value_supported_deviation_min_support is not None:
            model["value_supported_deviation_min_support"] = (
                self.value_supported_deviation_min_support
            )
        if self.value_supported_deviation_min_value_margin is not None:
            model["value_supported_deviation_min_value_margin"] = (
                self.value_supported_deviation_min_value_margin
            )
        if self.value_supported_deviation_min_learned_value is not None:
            model["value_supported_deviation_min_learned_value"] = (
                self.value_supported_deviation_min_learned_value
            )
        if self.value_supported_deviation_min_score_margin is not None:
            model["value_supported_deviation_min_score_margin"] = (
                self.value_supported_deviation_min_score_margin
            )
        if self.value_supported_deviation_min_predicted_advantage is not None:
            model["value_supported_deviation_min_predicted_advantage"] = (
                self.value_supported_deviation_min_predicted_advantage
            )
        if self.neural_network is not None:
            model["neural_backend"] = self.neural_backend
            model["neural_architecture"] = self.neural_architecture
            model["neural_training_policy"] = self.neural_training_policy
            model["neural_input_size"] = self.neural_input_size
            model["neural_hidden_units"] = self.neural_hidden_units
            model["neural_hidden_activation"] = self.neural_hidden_activation
            model["neural_input_normalization"] = self.neural_input_normalization
            model["neural_seed"] = self.neural_seed
            model["neural_state_value_policy"] = self.neural_state_value_policy
            model["neural_actor_prior_policy"] = self.neural_actor_prior_policy
            model["neural_actor_prior_blend_weight"] = (
                self.neural_actor_prior_blend_weight
            )
            model["neural_network"] = self.neural_network
        return {
            "manifest": {
                "artifact_version": MIND_MODEL_ARTIFACT_VERSION,
                "model_type": self.model_type,
                "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "policy_interface_version": POLICY_INTERFACE_VERSION,
                "action_contract_version": ACTION_CONTRACT_VERSION,
                "trained_record_count": self.record_count,
                "provenance": provenance,
            },
            "model": model,
        }


def train_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_contextual_prior_baseline(
        records,
        provenance=provenance,
        model_type=BEHAVIOR_CLONING_BASELINE_MODEL_TYPE,
        trainer=CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=UNIFORM_SAMPLE_WEIGHT_POLICY,
        sample_weight_base=UNIFORM_SAMPLE_WEIGHT_BASE,
        sample_weight_min=UNIFORM_SAMPLE_WEIGHT_MIN,
        sample_weight_fn=_uniform_sample_weight,
    )


def train_reward_weighted_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_contextual_prior_baseline(
        records,
        provenance=provenance,
        model_type=REWARD_WEIGHTED_BASELINE_MODEL_TYPE,
        trainer=REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=REWARD_TOTAL_SHIFTED_SAMPLE_WEIGHT_POLICY,
        sample_weight_base=REWARD_TOTAL_SAMPLE_WEIGHT_BASE,
        sample_weight_min=REWARD_TOTAL_SAMPLE_WEIGHT_MIN,
        sample_weight_fn=_reward_total_shifted_sample_weight,
    )


def train_advantage_calibrated_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_advantage_calibrated_contextual_prior_baseline(
        records,
        provenance=provenance,
    )


def train_advantage_blended_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_advantage_blended_contextual_prior_baseline(
        records,
        provenance=provenance,
    )


def train_value_calibrated_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_value_calibrated_contextual_prior_baseline(
        records,
        provenance=provenance,
    )


def train_neural_actor_critic_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_neural_actor_critic_behavior_cloning_baseline(
        records,
        provenance=provenance,
        model_type=NEURAL_ACTOR_CRITIC_BASELINE_MODEL_TYPE,
        trainer=NEURAL_ACTOR_CRITIC_TRAINER,
        sample_weight_policy=NEURAL_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY,
        neural_backend=NEURAL_BACKEND,
        neural_architecture=NEURAL_ARCHITECTURE,
        neural_training_policy=NEURAL_TRAINING_POLICY,
        neural_hidden_units=NEURAL_HIDDEN_UNITS,
        neural_seed=NEURAL_SEED,
        train_network_fn=train_neural_actor_critic_network,
    )


def train_torch_neural_actor_critic_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_neural_actor_critic_behavior_cloning_baseline(
        records,
        provenance=provenance,
        model_type=TORCH_NEURAL_ACTOR_CRITIC_BASELINE_MODEL_TYPE,
        trainer=TORCH_NEURAL_ACTOR_CRITIC_TRAINER,
        sample_weight_policy=TORCH_NEURAL_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY,
        neural_backend=TORCH_NEURAL_BACKEND,
        neural_architecture=TORCH_NEURAL_ARCHITECTURE,
        neural_training_policy=TORCH_NEURAL_TRAINING_POLICY,
        neural_hidden_units=TORCH_NEURAL_HIDDEN_UNITS,
        neural_seed=TORCH_NEURAL_SEED,
        train_network_fn=train_torch_actor_critic_network,
    )


def train_torch_advantage_actor_critic_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_neural_actor_critic_behavior_cloning_baseline(
        records,
        provenance=provenance,
        model_type=TORCH_ADVANTAGE_ACTOR_CRITIC_BASELINE_MODEL_TYPE,
        trainer=TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER,
        sample_weight_policy=TORCH_ADVANTAGE_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY,
        neural_backend=TORCH_NEURAL_BACKEND,
        neural_architecture=TORCH_NEURAL_ARCHITECTURE,
        neural_training_policy=TORCH_ADVANTAGE_TRAINING_POLICY,
        neural_hidden_units=TORCH_NEURAL_HIDDEN_UNITS,
        neural_seed=TORCH_NEURAL_SEED,
        train_network_fn=train_torch_advantage_actor_critic_network,
    )


def train_torch_discrete_iql_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_neural_actor_critic_behavior_cloning_baseline(
        records,
        provenance=provenance,
        model_type=TORCH_DISCRETE_IQL_BASELINE_MODEL_TYPE,
        trainer=TORCH_DISCRETE_IQL_TRAINER,
        sample_weight_policy=TORCH_DISCRETE_IQL_SAMPLE_WEIGHT_POLICY,
        neural_backend=TORCH_NEURAL_BACKEND,
        neural_architecture=TORCH_NEURAL_ARCHITECTURE,
        neural_training_policy=TORCH_DISCRETE_IQL_TRAINING_POLICY,
        neural_hidden_units=TORCH_NEURAL_HIDDEN_UNITS,
        neural_seed=TORCH_NEURAL_SEED,
        train_network_fn=train_torch_discrete_iql_network,
        enable_value_supported_deviation=False,
        heuristic_delegate_max_training_score_margin=(
            IQL_HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN
        ),
    )


def _train_neural_actor_critic_behavior_cloning_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
    model_type: str,
    trainer: str,
    sample_weight_policy: str,
    neural_backend: str,
    neural_architecture: str,
    neural_training_policy: str,
    neural_hidden_units: int,
    neural_seed: int,
    train_network_fn: Callable[
        [tuple[dict[str, object], ...]],
        dict[str, object],
    ],
    enable_value_supported_deviation: bool = True,
    neural_actor_prior_policy: str = NEURAL_ACTOR_PRIOR_POLICY,
    heuristic_delegate_max_training_score_margin: float = (
        HEURISTIC_DELEGATE_MAX_TRAINING_SCORE_MARGIN
    ),
) -> BehaviorCloningBaseline:
    materialized_records = tuple(records)
    baseline = _train_neural_actor_prior_baseline(
        materialized_records,
        provenance=provenance,
        model_type=model_type,
        trainer=trainer,
        sample_weight_policy=sample_weight_policy,
        neural_actor_prior_policy=neural_actor_prior_policy,
    )
    neural_network = train_network_fn(materialized_records)
    if not isinstance(neural_network, dict):
        raise ValueError("neural trainer must return a network object")
    action_value_output_bias = neural_network.get("action_value_output_bias")
    if not isinstance(action_value_output_bias, dict):
        raise ValueError(
            "neural network action_value_output_bias must be an object"
        )
    action_value_estimates = {
        action: _required_action_value_bias(action_value_output_bias, action)
        for action in ACTION_NAMES
    }
    return replace(
        baseline,
        action_score_metadata=_with_delegate_score_margin(
            baseline.action_score_metadata
        ),
        conditional_action_metadata={
            key: _with_delegate_score_margin(metadata)
            for key, metadata in baseline.conditional_action_metadata.items()
        },
        action_value_estimates=action_value_estimates,
        value_estimation_policy=NEURAL_ACTION_VALUE_POLICY,
        value_supported_deviation_policy=(
            VALUE_SUPPORTED_DEVIATION_POLICY
            if enable_value_supported_deviation
            else None
        ),
        value_supported_deviation_min_support=(
            VALUE_SUPPORTED_DEVIATION_MIN_SUPPORT
            if enable_value_supported_deviation
            else None
        ),
        value_supported_deviation_min_value_margin=(
            VALUE_SUPPORTED_DEVIATION_MIN_VALUE_MARGIN
            if enable_value_supported_deviation
            else None
        ),
        value_supported_deviation_min_learned_value=(
            VALUE_SUPPORTED_DEVIATION_MIN_LEARNED_VALUE
            if enable_value_supported_deviation
            else None
        ),
        value_supported_deviation_min_score_margin=(
            VALUE_SUPPORTED_DEVIATION_MIN_SCORE_MARGIN
            if enable_value_supported_deviation
            else None
        ),
        value_supported_deviation_min_predicted_advantage=(
            VALUE_SUPPORTED_DEVIATION_MIN_PREDICTED_ADVANTAGE
            if enable_value_supported_deviation
            else None
        ),
        neural_network=neural_network,
        neural_backend=neural_backend,
        neural_architecture=neural_architecture,
        neural_training_policy=neural_training_policy,
        neural_input_size=OBSERVATION_INPUT_VECTOR_SIZE,
        neural_hidden_units=neural_hidden_units,
        neural_hidden_activation=NEURAL_HIDDEN_ACTIVATION,
        neural_input_normalization=NEURAL_INPUT_NORMALIZATION,
        neural_seed=neural_seed,
        neural_state_value_policy=NEURAL_STATE_VALUE_POLICY,
        neural_actor_prior_policy=neural_actor_prior_policy,
        neural_actor_prior_blend_weight=NEURAL_ACTOR_PRIOR_BLEND_WEIGHT,
        heuristic_delegate_max_training_score_margin=(
            heuristic_delegate_max_training_score_margin
        ),
    )


def _train_neural_actor_prior_baseline(
    records: tuple[dict[str, object], ...],
    *,
    provenance: dict[str, object],
    model_type: str,
    trainer: str,
    sample_weight_policy: str,
    neural_actor_prior_policy: str,
) -> BehaviorCloningBaseline:
    if neural_actor_prior_policy == NEURAL_ADVANTAGE_BLENDED_ACTOR_PRIOR_POLICY:
        baseline = _train_reward_advantage_contextual_prior_baseline(
            records,
            provenance=provenance,
            model_type=model_type,
            trainer=trainer,
            sample_weight_policy=sample_weight_policy,
            advantage_blend_weight=ADVANTAGE_BLEND_WEIGHT,
        )
        return replace(
            baseline,
            sample_weight_base=UNIFORM_SAMPLE_WEIGHT_BASE,
            sample_weight_min=UNIFORM_SAMPLE_WEIGHT_MIN,
            sample_weight_max=None,
            sample_weight_total=float(baseline.record_count),
        )
    if neural_actor_prior_policy == NEURAL_ACTOR_PRIOR_POLICY:
        return _train_contextual_prior_baseline(
            records,
            provenance=provenance,
            model_type=model_type,
            trainer=trainer,
            sample_weight_policy=sample_weight_policy,
            sample_weight_base=UNIFORM_SAMPLE_WEIGHT_BASE,
            sample_weight_min=UNIFORM_SAMPLE_WEIGHT_MIN,
            sample_weight_fn=_uniform_sample_weight,
        )
    raise ValueError(
        f"unsupported neural actor prior policy: {neural_actor_prior_policy!r}"
    )


def _required_action_value_bias(
    action_value_output_bias: dict[object, object],
    action: str,
) -> float:
    value = action_value_output_bias.get(action)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            "neural network action_value_output_bias must include finite "
            f"numeric value for action {action!r}"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            "neural network action_value_output_bias must include finite "
            f"numeric value for action {action!r}"
        )
    return float(value)


def train_baseline_with_trainer(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
    trainer: str,
) -> BehaviorCloningBaseline:
    if trainer == CONTEXTUAL_PRIOR_TRAINER:
        return train_behavior_cloning_baseline(records, provenance=provenance)
    if trainer == REWARD_WEIGHTED_CONTEXTUAL_PRIOR_TRAINER:
        return train_reward_weighted_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == ADVANTAGE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER:
        return train_advantage_calibrated_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == ADVANTAGE_BLENDED_CONTEXTUAL_PRIOR_TRAINER:
        return train_advantage_blended_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == VALUE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER:
        return train_value_calibrated_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == NEURAL_ACTOR_CRITIC_TRAINER:
        return train_neural_actor_critic_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == TORCH_NEURAL_ACTOR_CRITIC_TRAINER:
        return train_torch_neural_actor_critic_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER:
        return train_torch_advantage_actor_critic_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    if trainer == TORCH_DISCRETE_IQL_TRAINER:
        return train_torch_discrete_iql_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
    raise ValueError(f"unsupported Mind baseline trainer: {trainer!r}")


def _train_contextual_prior_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
    model_type: str,
    trainer: str,
    sample_weight_policy: str,
    sample_weight_base: float,
    sample_weight_min: float,
    sample_weight_fn: Callable[[dict[str, object]], float],
) -> BehaviorCloningBaseline:
    support_counts: Counter[str] = Counter()
    weighted_counts: Counter[str] = Counter()
    conditional_support_counts: dict[str, Counter[str]] = {}
    conditional_weighted_counts: dict[str, Counter[str]] = {}
    record_count = 0
    sample_weight_total = 0.0
    for record in records:
        record_count += 1
        requested_action = str(record["requested_action"])
        resolved_action = str(record["resolved_action"])
        label = (
            requested_action
            if bool(record.get("resolution_action_valid", False))
            else resolved_action
        )
        weight = sample_weight_fn(record)
        support_counts[label] += 1
        weighted_counts[label] += weight
        sample_weight_total += weight
        for feature_key in feature_keys_from_record(record):
            conditional_support_counts.setdefault(feature_key, Counter())[label] += 1
            conditional_weighted_counts.setdefault(feature_key, Counter())[label] += (
                weight
            )
    total = sum(support_counts.values())
    if total <= 0:
        raise ValueError("behavior-cloning baseline requires at least one record")
    action_scores = _normalize_scores(weighted_counts)
    action_score_metadata = _score_metadata(
        support_counts,
        scores=action_scores,
        weighted_record_count=sum(weighted_counts.values()),
    )
    conditional_action_scores = {
        key: _normalize_prior_corrected_scores(
            conditional_weighted_counts[key],
            global_counts=weighted_counts,
            alpha=CONDITIONAL_SCORE_SMOOTHING_ALPHA,
            prior_exponent=CONDITIONAL_PRIOR_CORRECTION_EXPONENT,
        )
        for key, action_counts in conditional_support_counts.items()
        if sum(action_counts.values()) >= CONDITIONAL_MIN_RECORDS
    }
    conditional_action_metadata = {
        key: _score_metadata(
            action_counts,
            scores=conditional_action_scores[key],
            weighted_record_count=sum(conditional_weighted_counts[key].values()),
        )
        for key, action_counts in conditional_support_counts.items()
        if key in conditional_action_scores
    }
    return BehaviorCloningBaseline(
        action_scores=action_scores,
        action_score_metadata=action_score_metadata,
        conditional_action_scores=conditional_action_scores,
        conditional_action_metadata=conditional_action_metadata,
        record_count=record_count,
        provenance=provenance,
        model_type=model_type,
        trainer=trainer,
        sample_weight_policy=sample_weight_policy,
        sample_weight_base=sample_weight_base,
        sample_weight_min=sample_weight_min,
        sample_weight_total=sample_weight_total,
    )


def _train_advantage_calibrated_contextual_prior_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_reward_advantage_contextual_prior_baseline(
        records,
        provenance=provenance,
        model_type=ADVANTAGE_CALIBRATED_BASELINE_MODEL_TYPE,
        trainer=ADVANTAGE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=CONTEXTUAL_REWARD_ADVANTAGE_SAMPLE_WEIGHT_POLICY,
        advantage_blend_weight=None,
    )


def _train_advantage_blended_contextual_prior_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    return _train_reward_advantage_contextual_prior_baseline(
        records,
        provenance=provenance,
        model_type=ADVANTAGE_BLENDED_BASELINE_MODEL_TYPE,
        trainer=ADVANTAGE_BLENDED_CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=CONTEXTUAL_REWARD_ADVANTAGE_BLEND_SAMPLE_WEIGHT_POLICY,
        advantage_blend_weight=ADVANTAGE_BLEND_WEIGHT,
    )


def _train_reward_advantage_contextual_prior_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
    model_type: str,
    trainer: str,
    sample_weight_policy: str,
    advantage_blend_weight: float | None,
) -> BehaviorCloningBaseline:
    support_counts: Counter[str] = Counter()
    reward_sums: Counter[str] = Counter()
    conditional_support_counts: dict[str, Counter[str]] = {}
    conditional_reward_sums: dict[str, Counter[str]] = {}
    record_count = 0
    for record in records:
        record_count += 1
        requested_action = str(record["requested_action"])
        resolved_action = str(record["resolved_action"])
        label = (
            requested_action
            if bool(record.get("resolution_action_valid", False))
            else resolved_action
        )
        reward_total = _record_reward_total(record)
        support_counts[label] += 1
        reward_sums[label] += reward_total
        for feature_key in feature_keys_from_record(record):
            conditional_support_counts.setdefault(feature_key, Counter())[label] += 1
            conditional_reward_sums.setdefault(feature_key, Counter())[label] += (
                reward_total
            )
    total = sum(support_counts.values())
    if total <= 0:
        raise ValueError("behavior-cloning baseline requires at least one record")

    weighted_counts = _advantage_adjusted_counts(
        support_counts,
        reward_sums=reward_sums,
    )
    raw_action_scores = _normalize_scores(support_counts)
    advantage_action_scores = _normalize_scores(weighted_counts)
    action_scores = _blend_scores(
        raw_action_scores,
        advantage_action_scores,
        advantage_blend_weight,
    )
    action_score_metadata = _score_metadata(
        support_counts,
        scores=action_scores,
        delegate_scores=raw_action_scores,
        weighted_record_count=_blended_weighted_count(
            raw_record_count=sum(support_counts.values()),
            advantage_record_count=sum(weighted_counts.values()),
            advantage_blend_weight=advantage_blend_weight,
        ),
    )
    conditional_weighted_counts = {
        key: _advantage_adjusted_counts(
            action_counts,
            reward_sums=conditional_reward_sums[key],
        )
        for key, action_counts in conditional_support_counts.items()
    }
    raw_conditional_action_scores = {
        key: _normalize_prior_corrected_scores(
            action_counts,
            global_counts=support_counts,
            alpha=CONDITIONAL_SCORE_SMOOTHING_ALPHA,
            prior_exponent=CONDITIONAL_PRIOR_CORRECTION_EXPONENT,
        )
        for key, action_counts in conditional_support_counts.items()
        if sum(action_counts.values()) >= CONDITIONAL_MIN_RECORDS
    }
    advantage_conditional_action_scores = {
        key: _normalize_prior_corrected_scores(
            conditional_weighted_counts[key],
            global_counts=weighted_counts,
            alpha=CONDITIONAL_SCORE_SMOOTHING_ALPHA,
            prior_exponent=CONDITIONAL_PRIOR_CORRECTION_EXPONENT,
        )
        for key in raw_conditional_action_scores
    }
    conditional_action_scores = {
        key: _blend_scores(
            raw_conditional_action_scores[key],
            advantage_conditional_action_scores[key],
            advantage_blend_weight,
        )
        for key in raw_conditional_action_scores
    }
    conditional_action_metadata = {
        key: _score_metadata(
            action_counts,
            scores=conditional_action_scores[key],
            delegate_scores=raw_conditional_action_scores[key],
            weighted_record_count=_blended_weighted_count(
                raw_record_count=sum(action_counts.values()),
                advantage_record_count=sum(
                    conditional_weighted_counts[key].values()
                ),
                advantage_blend_weight=advantage_blend_weight,
            ),
        )
        for key, action_counts in conditional_support_counts.items()
        if key in conditional_action_scores
    }
    return BehaviorCloningBaseline(
        action_scores=action_scores,
        action_score_metadata=action_score_metadata,
        conditional_action_scores=conditional_action_scores,
        conditional_action_metadata=conditional_action_metadata,
        record_count=record_count,
        provenance=provenance,
        model_type=model_type,
        trainer=trainer,
        sample_weight_policy=sample_weight_policy,
        sample_weight_base=ADVANTAGE_SAMPLE_WEIGHT_BASE,
        sample_weight_min=ADVANTAGE_SAMPLE_WEIGHT_MIN,
        sample_weight_total=_blended_weighted_count(
            raw_record_count=sum(support_counts.values()),
            advantage_record_count=sum(weighted_counts.values()),
            advantage_blend_weight=advantage_blend_weight,
        ),
        sample_weight_max=ADVANTAGE_SAMPLE_WEIGHT_MAX,
        reward_advantage_policy=CONTEXTUAL_REWARD_ADVANTAGE_POLICY,
        reward_advantage_scale=ADVANTAGE_REWARD_SCALE,
        reward_advantage_min_action_support=ADVANTAGE_MIN_ACTION_SUPPORT,
        reward_advantage_blend_policy=(
            CONTEXTUAL_REWARD_ADVANTAGE_BLEND_POLICY
            if advantage_blend_weight is not None
            else None
        ),
        reward_advantage_blend_weight=advantage_blend_weight,
    )


def _train_value_calibrated_contextual_prior_baseline(
    records: Iterable[dict[str, object]],
    *,
    provenance: dict[str, object],
) -> BehaviorCloningBaseline:
    support_counts: Counter[str] = Counter()
    reward_sums: Counter[str] = Counter()
    conditional_support_counts: dict[str, Counter[str]] = {}
    conditional_reward_sums: dict[str, Counter[str]] = {}
    record_count = 0
    for record in records:
        record_count += 1
        requested_action = str(record["requested_action"])
        resolved_action = str(record["resolved_action"])
        label = (
            requested_action
            if bool(record.get("resolution_action_valid", False))
            else resolved_action
        )
        reward_total = _record_reward_total(record)
        support_counts[label] += 1
        reward_sums[label] += reward_total
        for feature_key in feature_keys_from_record(record):
            conditional_support_counts.setdefault(feature_key, Counter())[label] += 1
            conditional_reward_sums.setdefault(feature_key, Counter())[label] += (
                reward_total
            )
    total = sum(support_counts.values())
    if total <= 0:
        raise ValueError("behavior-cloning baseline requires at least one record")

    weighted_counts = _advantage_adjusted_counts(
        support_counts,
        reward_sums=reward_sums,
    )
    raw_action_scores = _normalize_scores(support_counts)
    advantage_action_scores = _normalize_scores(weighted_counts)
    anchor_action_scores = _blend_scores(
        raw_action_scores,
        advantage_action_scores,
        ADVANTAGE_BLEND_WEIGHT,
    )
    action_value_estimates = _mean_reward_estimates(
        support_counts,
        reward_sums=reward_sums,
    )
    value_action_scores = _value_score_distribution(
        support_counts,
        value_estimates=action_value_estimates,
        raw_scores=anchor_action_scores,
    )
    action_scores = _blend_scores(
        anchor_action_scores,
        value_action_scores,
        VALUE_SCORE_BLEND_WEIGHT,
    )
    action_score_metadata = _score_metadata(
        support_counts,
        scores=action_scores,
        delegate_scores=raw_action_scores,
        weighted_record_count=_blended_weighted_count(
            raw_record_count=sum(support_counts.values()),
            advantage_record_count=sum(weighted_counts.values()),
            advantage_blend_weight=ADVANTAGE_BLEND_WEIGHT,
        ),
    )
    conditional_weighted_counts = {
        key: _advantage_adjusted_counts(
            action_counts,
            reward_sums=conditional_reward_sums[key],
        )
        for key, action_counts in conditional_support_counts.items()
    }
    raw_conditional_action_scores = {
        key: _normalize_prior_corrected_scores(
            action_counts,
            global_counts=support_counts,
            alpha=CONDITIONAL_SCORE_SMOOTHING_ALPHA,
            prior_exponent=CONDITIONAL_PRIOR_CORRECTION_EXPONENT,
        )
        for key, action_counts in conditional_support_counts.items()
        if sum(action_counts.values()) >= CONDITIONAL_MIN_RECORDS
    }
    advantage_conditional_action_scores = {
        key: _normalize_prior_corrected_scores(
            conditional_weighted_counts[key],
            global_counts=weighted_counts,
            alpha=CONDITIONAL_SCORE_SMOOTHING_ALPHA,
            prior_exponent=CONDITIONAL_PRIOR_CORRECTION_EXPONENT,
        )
        for key in raw_conditional_action_scores
    }
    anchor_conditional_action_scores = {
        key: _blend_scores(
            raw_conditional_action_scores[key],
            advantage_conditional_action_scores[key],
            ADVANTAGE_BLEND_WEIGHT,
        )
        for key in raw_conditional_action_scores
    }
    conditional_action_value_estimates = {
        key: _mean_reward_estimates(
            conditional_support_counts[key],
            reward_sums=conditional_reward_sums[key],
        )
        for key in raw_conditional_action_scores
    }
    value_conditional_action_scores = {
        key: _value_score_distribution(
            conditional_support_counts[key],
            value_estimates=conditional_action_value_estimates[key],
            raw_scores=anchor_conditional_action_scores[key],
        )
        for key in raw_conditional_action_scores
    }
    conditional_action_scores = {
        key: _blend_scores(
            anchor_conditional_action_scores[key],
            value_conditional_action_scores[key],
            VALUE_SCORE_BLEND_WEIGHT,
        )
        for key in raw_conditional_action_scores
    }
    conditional_action_metadata = {
        key: _score_metadata(
            action_counts,
            scores=conditional_action_scores[key],
            delegate_scores=raw_conditional_action_scores[key],
            weighted_record_count=_blended_weighted_count(
                raw_record_count=sum(action_counts.values()),
                advantage_record_count=sum(
                    conditional_weighted_counts[key].values()
                ),
                advantage_blend_weight=ADVANTAGE_BLEND_WEIGHT,
            ),
        )
        for key, action_counts in conditional_support_counts.items()
        if key in conditional_action_scores
    }
    return BehaviorCloningBaseline(
        action_scores=action_scores,
        action_score_metadata=action_score_metadata,
        conditional_action_scores=conditional_action_scores,
        conditional_action_metadata=conditional_action_metadata,
        record_count=record_count,
        provenance=provenance,
        model_type=VALUE_CALIBRATED_BASELINE_MODEL_TYPE,
        trainer=VALUE_CALIBRATED_CONTEXTUAL_PRIOR_TRAINER,
        sample_weight_policy=CONTEXTUAL_VALUE_CALIBRATED_SAMPLE_WEIGHT_POLICY,
        sample_weight_base=ADVANTAGE_SAMPLE_WEIGHT_BASE,
        sample_weight_min=ADVANTAGE_SAMPLE_WEIGHT_MIN,
        sample_weight_total=_blended_weighted_count(
            raw_record_count=sum(support_counts.values()),
            advantage_record_count=sum(weighted_counts.values()),
            advantage_blend_weight=ADVANTAGE_BLEND_WEIGHT,
        ),
        sample_weight_max=ADVANTAGE_SAMPLE_WEIGHT_MAX,
        reward_advantage_policy=CONTEXTUAL_REWARD_ADVANTAGE_POLICY,
        reward_advantage_scale=ADVANTAGE_REWARD_SCALE,
        reward_advantage_min_action_support=ADVANTAGE_MIN_ACTION_SUPPORT,
        reward_advantage_blend_policy=CONTEXTUAL_REWARD_ADVANTAGE_BLEND_POLICY,
        reward_advantage_blend_weight=ADVANTAGE_BLEND_WEIGHT,
        action_value_estimates=action_value_estimates,
        conditional_action_value_estimates=conditional_action_value_estimates,
        value_estimation_policy=MEAN_REWARD_ACTION_VALUE_POLICY,
        value_score_blend_policy=PRIOR_VALUE_SCORE_BLEND_POLICY,
        value_score_blend_weight=VALUE_SCORE_BLEND_WEIGHT,
        value_min_action_support=VALUE_MIN_ACTION_SUPPORT,
        value_score_epsilon=VALUE_SCORE_EPSILON,
        value_supported_deviation_policy=VALUE_SUPPORTED_DEVIATION_POLICY,
        value_supported_deviation_min_support=(
            VALUE_SUPPORTED_DEVIATION_MIN_SUPPORT
        ),
        value_supported_deviation_min_value_margin=(
            VALUE_SUPPORTED_DEVIATION_MIN_VALUE_MARGIN
        ),
        value_supported_deviation_min_learned_value=(
            VALUE_SUPPORTED_DEVIATION_MIN_LEARNED_VALUE
        ),
        value_supported_deviation_min_score_margin=(
            VALUE_SUPPORTED_DEVIATION_MIN_SCORE_MARGIN
        ),
        value_supported_deviation_min_predicted_advantage=(
            VALUE_SUPPORTED_DEVIATION_MIN_PREDICTED_ADVANTAGE
        ),
    )


def _normalize_scores(counts: Counter[str]) -> dict[str, float]:
    total = sum(float(value) for value in counts.values())
    if total <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    return {
        action: float(counts.get(action, 0.0)) / total
        for action in ACTION_NAMES
    }


def _normalize_prior_corrected_scores(
    counts: Counter[str],
    *,
    global_counts: Counter[str],
    alpha: float,
    prior_exponent: float,
) -> dict[str, float]:
    total = sum(float(value) for value in counts.values())
    global_total = sum(float(value) for value in global_counts.values())
    if total <= 0 or global_total <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    action_count = len(ACTION_NAMES)
    raw_scores = {
        action: (float(counts.get(action, 0.0)) + alpha)
        / (total + alpha * action_count)
        for action in ACTION_NAMES
    }
    global_priors = {
        action: (float(global_counts.get(action, 0.0)) + alpha)
        / (global_total + alpha * action_count)
        for action in ACTION_NAMES
    }
    corrected_scores = {
        action: raw_scores[action] / (global_priors[action] ** prior_exponent)
        for action in ACTION_NAMES
    }
    corrected_total = sum(corrected_scores.values())
    if corrected_total <= 0:
        return {action: 0.0 for action in ACTION_NAMES}
    return {
        action: corrected_scores[action] / corrected_total
        for action in ACTION_NAMES
    }


def _advantage_adjusted_counts(
    counts: Counter[str],
    *,
    reward_sums: Counter[str],
) -> Counter[str]:
    total = sum(counts.values())
    if total <= 0:
        return Counter()
    supported_actions = [
        action
        for action, count in counts.items()
        if count >= ADVANTAGE_MIN_ACTION_SUPPORT
    ]
    if len(supported_actions) < 2:
        return Counter({action: float(count) for action, count in counts.items()})
    context_mean_reward = (
        sum(float(value) for value in reward_sums.values()) / float(total)
    )
    adjusted: Counter[str] = Counter()
    for action, count in counts.items():
        count_value = float(count)
        if count_value <= 0.0:
            continue
        if count < ADVANTAGE_MIN_ACTION_SUPPORT:
            multiplier = ADVANTAGE_SAMPLE_WEIGHT_BASE
        else:
            action_mean_reward = float(reward_sums.get(action, 0.0)) / count_value
            advantage = action_mean_reward - context_mean_reward
            multiplier = ADVANTAGE_SAMPLE_WEIGHT_BASE + (
                ADVANTAGE_REWARD_SCALE * advantage
            )
            multiplier = max(
                ADVANTAGE_SAMPLE_WEIGHT_MIN,
                min(ADVANTAGE_SAMPLE_WEIGHT_MAX, multiplier),
            )
        adjusted[action] = count_value * multiplier
    return adjusted


def _blend_scores(
    raw_scores: dict[str, float],
    advantage_scores: dict[str, float],
    advantage_blend_weight: float | None,
) -> dict[str, float]:
    if advantage_blend_weight is None:
        return dict(advantage_scores)
    return {
        action: (
            (1.0 - advantage_blend_weight) * float(raw_scores.get(action, 0.0))
            + advantage_blend_weight * float(advantage_scores.get(action, 0.0))
        )
        for action in ACTION_NAMES
    }


def _mean_reward_estimates(
    counts: Counter[str],
    *,
    reward_sums: Counter[str],
) -> dict[str, float]:
    total = sum(counts.values())
    context_mean_reward = (
        sum(float(value) for value in reward_sums.values()) / float(total)
        if total > 0
        else 0.0
    )
    estimates: dict[str, float] = {}
    for action in ACTION_NAMES:
        count = float(counts.get(action, 0.0))
        if count <= 0.0:
            estimates[action] = context_mean_reward
        else:
            estimates[action] = float(reward_sums.get(action, 0.0)) / count
    return estimates


def _value_score_distribution(
    counts: Counter[str],
    *,
    value_estimates: dict[str, float],
    raw_scores: dict[str, float],
) -> dict[str, float]:
    supported_actions = [
        action
        for action in ACTION_NAMES
        if counts.get(action, 0) >= VALUE_MIN_ACTION_SUPPORT
    ]
    if len(supported_actions) < 2:
        return dict(raw_scores)
    supported_values = [float(value_estimates[action]) for action in supported_actions]
    minimum_value = min(supported_values)
    maximum_value = max(supported_values)
    if maximum_value - minimum_value <= VALUE_SCORE_EPSILON:
        return dict(raw_scores)
    shifted = {
        action: (
            max(0.0, float(value_estimates[action]) - minimum_value)
            + VALUE_SCORE_EPSILON
            if action in supported_actions
            else 0.0
        )
        for action in ACTION_NAMES
    }
    total = sum(shifted.values())
    if total <= 0.0:
        return dict(raw_scores)
    return {
        action: float(shifted[action]) / total
        for action in ACTION_NAMES
    }


def _blended_weighted_count(
    *,
    raw_record_count: float,
    advantage_record_count: float,
    advantage_blend_weight: float | None,
) -> float:
    if advantage_blend_weight is None:
        return float(advantage_record_count)
    return (
        (1.0 - advantage_blend_weight) * float(raw_record_count)
        + advantage_blend_weight * float(advantage_record_count)
    )


def _score_metadata(
    counts: Counter[str],
    *,
    scores: dict[str, float] | None = None,
    delegate_scores: dict[str, float] | None = None,
    weighted_record_count: float | None = None,
) -> dict[str, object]:
    total = int(sum(counts.values()))
    if total <= 0:
        metadata = {
            "record_count": 0,
            "top_action": "stay",
            "top_score": 0.0,
            "runner_up_action": "stay",
            "runner_up_score": 0.0,
            "score_margin": 0.0,
        }
        if delegate_scores is not None:
            metadata["delegate_score_margin"] = 0.0
        if weighted_record_count is not None:
            metadata["weighted_record_count"] = round(weighted_record_count, 4)
        return metadata
    ranking_scores = scores if scores is not None else _normalize_scores(counts)
    ranked = sorted(
        (
            (float(ranking_scores.get(action, 0.0)), action)
            for action in sorted(ACTION_NAMES)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    top_score, top_action = ranked[0]
    runner_up_score, runner_up_action = (
        ranked[1] if len(ranked) > 1 else (0.0, "stay")
    )
    score_margin = top_score - runner_up_score
    metadata = {
        "record_count": total,
        "top_action": top_action,
        "top_score": top_score,
        "runner_up_action": runner_up_action,
        "runner_up_score": runner_up_score,
        "score_margin": score_margin,
    }
    if delegate_scores is not None:
        metadata["delegate_score_margin"] = _delegate_score_margin(
            delegate_scores,
            top_action,
            max_margin=score_margin,
        )
    if weighted_record_count is not None:
        metadata["weighted_record_count"] = round(weighted_record_count, 4)
    return metadata


def _delegate_score_margin(
    scores: dict[str, float],
    top_action: str,
    *,
    max_margin: float,
) -> float:
    target_score = float(scores.get(top_action, 0.0))
    runner_up_score = max(
        (
            float(scores.get(action, 0.0))
            for action in ACTION_NAMES
            if action != top_action
        ),
        default=0.0,
    )
    return max(0.0, min(max_margin, target_score - runner_up_score))


def _with_delegate_score_margin(metadata: dict[str, object]) -> dict[str, object]:
    updated = dict(metadata)
    delegate_margin = updated.get("delegate_score_margin")
    if isinstance(delegate_margin, (int, float)) and not isinstance(
        delegate_margin,
        bool,
    ):
        return updated
    score_margin = updated.get("score_margin")
    updated["delegate_score_margin"] = (
        float(score_margin)
        if isinstance(score_margin, (int, float)) and not isinstance(score_margin, bool)
        else 0.0
    )
    return updated


def _uniform_sample_weight(record: dict[str, object]) -> float:
    return UNIFORM_SAMPLE_WEIGHT_BASE


def _reward_total_shifted_sample_weight(record: dict[str, object]) -> float:
    return max(
        REWARD_TOTAL_SAMPLE_WEIGHT_MIN,
        REWARD_TOTAL_SAMPLE_WEIGHT_BASE + _record_reward_total(record),
    )


def _record_reward_total(record: dict[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return 0.0
    total = reward.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    reward_total = float(total)
    if not math.isfinite(reward_total):
        return 0.0
    return reward_total
