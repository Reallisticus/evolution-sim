from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    decode_observation_input,
    encode_observation_input,
)
from evolution_sim.env.runtime.trajectory import REWARD_TOTAL_BOUNDS

NEURAL_ACTOR_CRITIC_TRAINER = "neural-actor-critic-bc"
NEURAL_ACTOR_CRITIC_MODEL_TYPE = "guarded_neural_actor_critic_bc_v1"
NEURAL_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY = "neural_actor_critic_bc_uniform_v1"
NEURAL_BACKEND = "pure_python_deterministic_v1"
NEURAL_ARCHITECTURE = "fixed_random_feature_mlp_actor_critic_v1"
NEURAL_TRAINING_POLICY = "one_pass_hidden_prototype_actor_critic_bc_v1"
NEURAL_INPUT_NORMALIZATION = "observation_encoder_identity_clipped_v1"
NEURAL_HIDDEN_ACTIVATION = "tanh"
NEURAL_HIDDEN_UNITS = 8
NEURAL_SEED = 17
NEURAL_ACTION_VALUE_POLICY = "neural_action_value_head_v1"
NEURAL_STATE_VALUE_POLICY = "neural_state_value_head_v1"
NEURAL_CONTEXTUAL_ACTOR_PRIOR_POLICY = "contextual_prior_score_anchor_v1"
NEURAL_ADVANTAGE_BLENDED_ACTOR_PRIOR_POLICY = (
    "advantage_blended_contextual_prior_score_anchor_v1"
)
NEURAL_ACTOR_PRIOR_POLICY = NEURAL_CONTEXTUAL_ACTOR_PRIOR_POLICY
NEURAL_ACTOR_PRIOR_POLICIES: frozenset[str] = frozenset(
    {
        NEURAL_CONTEXTUAL_ACTOR_PRIOR_POLICY,
        NEURAL_ADVANTAGE_BLENDED_ACTOR_PRIOR_POLICY,
    }
)
NEURAL_ACTOR_PRIOR_BLEND_WEIGHT = 0.9

TORCH_NEURAL_ACTOR_CRITIC_TRAINER = "torch-actor-critic-bc"
TORCH_NEURAL_ACTOR_CRITIC_MODEL_TYPE = "guarded_torch_actor_critic_bc_v1"
TORCH_NEURAL_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY = (
    "torch_actor_critic_bc_uniform_v1"
)
TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER = "torch-advantage-actor-critic-bc"
TORCH_ADVANTAGE_ACTOR_CRITIC_MODEL_TYPE = (
    "guarded_torch_advantage_actor_critic_bc_v1"
)
TORCH_ADVANTAGE_ACTOR_CRITIC_SAMPLE_WEIGHT_POLICY = (
    "torch_advantage_actor_critic_bc_contextual_advantage_weighted_v1"
)
TORCH_DISCRETE_IQL_TRAINER = "torch-discrete-iql"
TORCH_DISCRETE_IQL_MODEL_TYPE = "guarded_torch_discrete_iql_v1"
TORCH_DISCRETE_IQL_SAMPLE_WEIGHT_POLICY = (
    "torch_discrete_iql_transition_expectile_awbc_v1"
)
TORCH_NEURAL_BACKEND = "pytorch_optional_v1"
TORCH_NEURAL_ARCHITECTURE = "torch_mlp_actor_critic_v1"
TORCH_NEURAL_TRAINING_POLICY = (
    "adamw_balanced_cross_entropy_td0_actor_critic_v1"
)
TORCH_ADVANTAGE_TRAINING_POLICY = (
    "adamw_contextual_advantage_weighted_actor_critic_v1"
)
TORCH_DISCRETE_IQL_TRAINING_POLICY = (
    "adamw_discrete_iql_expectile_advantage_weighted_v1"
)
TORCH_NEURAL_HIDDEN_UNITS = 256
TORCH_NEURAL_SEED = 23
TORCH_NEURAL_EPOCHS = 224
TORCH_NEURAL_LEARNING_RATE = 0.01
TORCH_NEURAL_WEIGHT_DECAY = 0.0001
TORCH_NEURAL_VALUE_LOSS_WEIGHT = 0.25
TORCH_NEURAL_CLASS_WEIGHT_MIN = 0.25
TORCH_NEURAL_CLASS_WEIGHT_MAX = 3.0
TORCH_ADVANTAGE_TEMPERATURE = 0.08
TORCH_ADVANTAGE_WEIGHT_BLEND = 0.25
TORCH_ADVANTAGE_WEIGHT_MIN = 0.25
TORCH_ADVANTAGE_WEIGHT_MAX = 2.5
TORCH_ADVANTAGE_MIN_CONTEXT_RECORDS = 16
TORCH_ADVANTAGE_MIN_ACTION_SUPPORT = 3
TORCH_IQL_DISCOUNT = 0.98
TORCH_IQL_EXPECTILE = 0.7
TORCH_IQL_ADVANTAGE_TEMPERATURE = 0.1
TORCH_IQL_ADVANTAGE_WEIGHT_MAX = 5.0
TORCH_IQL_Q_LOSS_WEIGHT = 1.0
TORCH_IQL_VALUE_LOSS_WEIGHT = 1.0
TORCH_IQL_ACTOR_LOSS_WEIGHT = 1.0

NEURAL_ACTOR_CRITIC_MODEL_TYPES: frozenset[str] = frozenset(
    {
        NEURAL_ACTOR_CRITIC_MODEL_TYPE,
        TORCH_NEURAL_ACTOR_CRITIC_MODEL_TYPE,
        TORCH_ADVANTAGE_ACTOR_CRITIC_MODEL_TYPE,
        TORCH_DISCRETE_IQL_MODEL_TYPE,
    }
)


@dataclass(frozen=True, slots=True)
class CompiledNeuralActorCriticNetwork:
    hidden_weights: list[list[float]]
    hidden_bias: list[float]
    actor_output_weights: dict[str, list[float]]
    actor_output_bias: dict[str, float]
    action_value_output_weights: dict[str, list[float]]
    action_value_output_bias: dict[str, float]
    state_value_weights: list[float]
    state_value_bias: float


NeuralActorCriticNetworkPayload = (
    Mapping[str, object] | CompiledNeuralActorCriticNetwork
)


def is_neural_actor_critic_model_type(model_type: object) -> bool:
    return (
        isinstance(model_type, str)
        and model_type in NEURAL_ACTOR_CRITIC_MODEL_TYPES
    )


def train_neural_actor_critic_network(
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if not records:
        raise ValueError("neural actor-critic trainer requires at least one record")

    hidden_weights = _deterministic_hidden_weights()
    hidden_bias = _deterministic_hidden_bias()
    samples: list[tuple[list[float], list[float], str, float]] = []
    action_counts: Counter[str] = Counter()
    action_reward_sums: Counter[str] = Counter()
    total_reward = 0.0

    for record in records:
        values = _record_observation_values(record)
        hidden = _hidden_activations(values, hidden_weights, hidden_bias)
        label = _record_label(record)
        reward_total = _record_reward_total(record)
        samples.append((values, hidden, label, reward_total))
        action_counts[label] += 1
        action_reward_sums[label] += reward_total
        total_reward += reward_total

    record_count = len(samples)
    global_hidden_mean = _mean_vectors(hidden for _, hidden, _, _ in samples)
    global_reward_mean = total_reward / float(record_count)
    action_hidden_means = {
        action: _mean_vectors(
            hidden
            for _, hidden, label, _ in samples
            if label == action
        )
        for action in ACTION_NAMES
    }
    actor_output_weights = {
        action: [
            _round(0.5 * (action_hidden_means[action][index] - global_hidden_mean[index]))
            for index in range(NEURAL_HIDDEN_UNITS)
        ]
        for action in ACTION_NAMES
    }
    actor_output_bias = {
        action: _round(
            math.log(
                (float(action_counts.get(action, 0)) + 0.1)
                / (float(record_count) + 0.1 * len(ACTION_NAMES))
            )
        )
        for action in ACTION_NAMES
    }

    state_value_weights = [
        _round(
            0.2
            * sum((reward - global_reward_mean) * hidden[index] for _, hidden, _, reward in samples)
            / float(record_count)
        )
        for index in range(NEURAL_HIDDEN_UNITS)
    ]
    action_value_output_bias = {
        action: _round(
            float(action_reward_sums.get(action, 0.0)) / float(action_counts[action])
            if action_counts.get(action, 0) > 0
            else global_reward_mean
        )
        for action in ACTION_NAMES
    }
    action_value_output_weights = {
        action: _action_value_weights(
            samples,
            action=action,
            action_mean=float(action_value_output_bias[action]),
        )
        for action in ACTION_NAMES
    }

    return {
        "hidden_weights": hidden_weights,
        "hidden_bias": hidden_bias,
        "actor_output_weights": actor_output_weights,
        "actor_output_bias": actor_output_bias,
        "action_value_output_weights": action_value_output_weights,
        "action_value_output_bias": action_value_output_bias,
        "state_value_weights": state_value_weights,
        "state_value_bias": _round(global_reward_mean),
    }


def score_neural_actor_critic(
    *,
    network: NeuralActorCriticNetworkPayload,
    observation: dict[str, object],
) -> tuple[dict[str, float], dict[str, float], float]:
    values = decode_observation_input(encode_observation_input(observation))
    return score_neural_actor_critic_values(network=network, values=values)


def compile_neural_actor_critic_network(
    network: Mapping[str, object],
) -> CompiledNeuralActorCriticNetwork:
    return CompiledNeuralActorCriticNetwork(
        hidden_weights=_matrix(network["hidden_weights"]),
        hidden_bias=_vector(network["hidden_bias"]),
        actor_output_weights=_action_matrix(network["actor_output_weights"]),
        actor_output_bias=_action_vector(network["actor_output_bias"]),
        action_value_output_weights=_action_matrix(
            network["action_value_output_weights"]
        ),
        action_value_output_bias=_action_vector(
            network["action_value_output_bias"]
        ),
        state_value_weights=_vector(network["state_value_weights"]),
        state_value_bias=float(network["state_value_bias"]),
    )


def score_neural_actor_critic_values(
    *,
    network: NeuralActorCriticNetworkPayload,
    values: Sequence[float],
) -> tuple[dict[str, float], dict[str, float], float]:
    compiled_network = (
        network
        if isinstance(network, CompiledNeuralActorCriticNetwork)
        else compile_neural_actor_critic_network(network)
    )
    hidden = _hidden_activations(
        values,
        compiled_network.hidden_weights,
        compiled_network.hidden_bias,
    )
    logits = {
        action: compiled_network.actor_output_bias[action]
        + _dot(compiled_network.actor_output_weights[action], hidden)
        for action in ACTION_NAMES
    }
    action_scores = _softmax(logits)
    lower, upper = REWARD_TOTAL_BOUNDS
    action_values = {
        action: _clamp(
            compiled_network.action_value_output_bias[action]
            + _dot(compiled_network.action_value_output_weights[action], hidden),
            lower,
            upper,
        )
        for action in ACTION_NAMES
    }
    state_value = _clamp(
        compiled_network.state_value_bias
        + _dot(compiled_network.state_value_weights, hidden),
        lower,
        upper,
    )
    return action_scores, action_values, state_value


def _record_observation_values(record: dict[str, object]) -> list[float]:
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        raise ValueError("trajectory record observation_input must be an object")
    return decode_observation_input(observation_input)


def _record_label(record: dict[str, object]) -> str:
    requested_action = str(record["requested_action"])
    resolved_action = str(record["resolved_action"])
    return (
        requested_action
        if bool(record.get("resolution_action_valid", False))
        else resolved_action
    )


def _record_reward_total(record: dict[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return 0.0
    total = reward.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    parsed = float(total)
    return parsed if math.isfinite(parsed) else 0.0


def _deterministic_hidden_weights() -> list[list[float]]:
    return [
        [
            _round(
                0.025
                * math.sin(
                    (NEURAL_SEED + 1) * 0.011
                    + (unit + 1) * 0.071
                    + (index + 1) * 0.013
                )
            )
            for index in range(OBSERVATION_INPUT_VECTOR_SIZE)
        ]
        for unit in range(NEURAL_HIDDEN_UNITS)
    ]


def _deterministic_hidden_bias() -> list[float]:
    return [
        _round(0.05 * math.sin((NEURAL_SEED + unit + 1) * 0.17))
        for unit in range(NEURAL_HIDDEN_UNITS)
    ]


def _hidden_activations(
    values: Sequence[float],
    hidden_weights: Sequence[Sequence[float]],
    hidden_bias: Sequence[float],
) -> list[float]:
    return [
        math.tanh(_dot(weights, values) + hidden_bias[index])
        for index, weights in enumerate(hidden_weights)
    ]


def _mean_vectors(vectors: Any) -> list[float]:
    total = [0.0 for _ in range(NEURAL_HIDDEN_UNITS)]
    count = 0
    for vector in vectors:
        count += 1
        for index, value in enumerate(vector):
            total[index] += float(value)
    if count <= 0:
        return [0.0 for _ in range(NEURAL_HIDDEN_UNITS)]
    return [_round(value / float(count)) for value in total]


def _action_value_weights(
    samples: list[tuple[list[float], list[float], str, float]],
    *,
    action: str,
    action_mean: float,
) -> list[float]:
    matching = [
        (hidden, reward)
        for _, hidden, label, reward in samples
        if label == action
    ]
    if not matching:
        return [0.0 for _ in range(NEURAL_HIDDEN_UNITS)]
    return [
        _round(
            0.2
            * sum((reward - action_mean) * hidden[index] for hidden, reward in matching)
            / float(len(matching))
        )
        for index in range(NEURAL_HIDDEN_UNITS)
    ]


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    maximum = max(float(value) for value in logits.values())
    exps = {
        action: math.exp(float(value) - maximum)
        for action, value in logits.items()
    }
    total = sum(exps.values())
    if total <= 0.0:
        return {action: 1.0 / len(ACTION_NAMES) for action in ACTION_NAMES}
    return {action: exps[action] / total for action in ACTION_NAMES}


def _action_matrix(payload: object) -> dict[str, list[float]]:
    if not isinstance(payload, dict):
        raise ValueError("neural action matrix must be an object")
    return {action: _vector(payload[action]) for action in ACTION_NAMES}


def _action_vector(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("neural action vector must be an object")
    return {action: float(payload[action]) for action in ACTION_NAMES}


def _matrix(payload: object) -> list[list[float]]:
    if not isinstance(payload, list):
        raise ValueError("neural matrix must be a list")
    return [_vector(row) for row in payload]


def _vector(payload: object) -> list[float]:
    if not isinstance(payload, list):
        raise ValueError("neural vector must be a list")
    return [float(value) for value in payload]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round(value: float) -> float:
    return round(float(value), 6)
