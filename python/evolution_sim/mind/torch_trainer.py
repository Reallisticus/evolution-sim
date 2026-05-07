from __future__ import annotations

import importlib.metadata as importlib_metadata
import math
import platform
from collections import Counter
from typing import Any

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    decode_observation_input,
)
from evolution_sim.env.runtime.trajectory import REWARD_TOTAL_BOUNDS
from evolution_sim.mind.dataset import build_trajectory_transitions
from evolution_sim.mind.feature_policy import feature_keys_from_record
from evolution_sim.mind.neural import (
    TORCH_ADVANTAGE_MIN_ACTION_SUPPORT,
    TORCH_ADVANTAGE_MIN_CONTEXT_RECORDS,
    TORCH_ADVANTAGE_TEMPERATURE,
    TORCH_ADVANTAGE_WEIGHT_BLEND,
    TORCH_ADVANTAGE_WEIGHT_MAX,
    TORCH_ADVANTAGE_WEIGHT_MIN,
    TORCH_IQL_ACTOR_LOSS_WEIGHT,
    TORCH_IQL_ADVANTAGE_TEMPERATURE,
    TORCH_IQL_ADVANTAGE_WEIGHT_MAX,
    TORCH_IQL_DISCOUNT,
    TORCH_IQL_EXPECTILE,
    TORCH_IQL_Q_LOSS_WEIGHT,
    TORCH_IQL_VALUE_LOSS_WEIGHT,
    TORCH_NEURAL_CLASS_WEIGHT_MAX,
    TORCH_NEURAL_CLASS_WEIGHT_MIN,
    TORCH_NEURAL_EPOCHS,
    TORCH_NEURAL_HIDDEN_UNITS,
    TORCH_NEURAL_LEARNING_RATE,
    TORCH_NEURAL_SEED,
    TORCH_NEURAL_VALUE_LOSS_WEIGHT,
    TORCH_NEURAL_WEIGHT_DECAY,
    _record_label,
    _record_observation_values,
    _record_reward_total,
)

MIND_ML_INSTALL_HINT = (
    "PyTorch Mind training requires optional ML dependencies. "
    "Install them with: python3 -m pip install -r requirements-mind-ml.txt"
)


def train_torch_actor_critic_network(
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return _train_torch_actor_critic_network(
        records,
        advantage_weighted=False,
    )


def train_torch_advantage_actor_critic_network(
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return _train_torch_actor_critic_network(
        records,
        advantage_weighted=True,
    )


def train_torch_discrete_iql_network(
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    torch = _load_torch()
    transitions = build_trajectory_transitions(records)
    if not transitions:
        raise ValueError("torch discrete IQL trainer requires at least one transition")

    torch.manual_seed(TORCH_NEURAL_SEED)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    action_index = {action: index for index, action in enumerate(ACTION_NAMES)}
    features = [
        decode_observation_input(transition.observation_input)
        for transition in transitions
    ]
    next_features = [
        (
            [0.0 for _ in range(OBSERVATION_INPUT_VECTOR_SIZE)]
            if transition.next_observation_input is None
            else decode_observation_input(transition.next_observation_input)
        )
        for transition in transitions
    ]
    labels = [action_index[transition.action] for transition in transitions]
    rewards = [transition.reward_total for transition in transitions]
    dones = [1.0 if transition.done else 0.0 for transition in transitions]
    action_masks = [
        _action_mask_values(transition.action_mask)
        for transition in transitions
    ]

    x = torch.tensor(features, dtype=torch.float32)
    next_x = torch.tensor(next_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    reward_target = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    done_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    mask_tensor = torch.tensor(action_masks, dtype=torch.bool)
    class_weights = _balanced_class_weights(torch, labels)
    model = _TorchActorCritic(torch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TORCH_NEURAL_LEARNING_RATE,
        weight_decay=TORCH_NEURAL_WEIGHT_DECAY,
    )

    final_loss = 0.0
    final_actor_loss = 0.0
    final_q_loss = 0.0
    final_value_loss = 0.0
    for _ in range(TORCH_NEURAL_EPOCHS):
        optimizer.zero_grad()
        logits, action_values, state_values = model(x)
        masked_logits = _masked_logits(torch, logits, mask_tensor)
        selected_action_values = action_values.gather(1, y.unsqueeze(1))
        with torch.no_grad():
            _, _, next_state_values = model(next_x)
            q_target = reward_target + (
                TORCH_IQL_DISCOUNT * (1.0 - done_tensor) * next_state_values
            )
        q_loss = torch.nn.functional.mse_loss(selected_action_values, q_target)
        value_error = selected_action_values.detach() - state_values
        expectile_weights = torch.where(
            value_error > 0,
            torch.full_like(value_error, TORCH_IQL_EXPECTILE),
            torch.full_like(value_error, 1.0 - TORCH_IQL_EXPECTILE),
        )
        value_loss = torch.mean(expectile_weights * value_error.pow(2))
        with torch.no_grad():
            actor_advantage = selected_action_values - state_values
            actor_weights = torch.exp(
                actor_advantage / TORCH_IQL_ADVANTAGE_TEMPERATURE
            ).clamp(max=TORCH_IQL_ADVANTAGE_WEIGHT_MAX)
        actor_losses = torch.nn.functional.cross_entropy(
            masked_logits,
            y,
            weight=class_weights,
            reduction="none",
        )
        actor_loss = (
            (actor_losses * actor_weights.squeeze(1)).sum()
            / actor_weights.sum().clamp_min(1e-9)
        )
        loss = (
            TORCH_IQL_ACTOR_LOSS_WEIGHT * actor_loss
            + TORCH_IQL_Q_LOSS_WEIGHT * q_loss
            + TORCH_IQL_VALUE_LOSS_WEIGHT * value_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        final_actor_loss = float(actor_loss.detach().cpu().item())
        final_q_loss = float(q_loss.detach().cpu().item())
        final_value_loss = float(value_loss.detach().cpu().item())

    training_metrics = {
        "epochs": TORCH_NEURAL_EPOCHS,
        "final_loss": _round(final_loss),
        "final_actor_loss": _round(final_actor_loss),
        "final_q_loss": _round(final_q_loss),
        "final_value_loss": _round(final_value_loss),
        "learning_rate": TORCH_NEURAL_LEARNING_RATE,
        "weight_decay": TORCH_NEURAL_WEIGHT_DECAY,
        "class_weight_min": TORCH_NEURAL_CLASS_WEIGHT_MIN,
        "class_weight_max": TORCH_NEURAL_CLASS_WEIGHT_MAX,
        "class_weights": {
            action: _round(float(class_weights[index].detach().cpu().item()))
            for index, action in enumerate(ACTION_NAMES)
        },
        "transition_count": len(transitions),
        "terminal_transition_rate": _round(
            sum(dones) / float(len(transitions)) if transitions else 0.0
        ),
        "critic_policy": "td0_expectile_q_v_v1",
        "actor_weighting_policy": "iql_masked_advantage_weighted_bc_v1",
        "iql_discount": TORCH_IQL_DISCOUNT,
        "iql_expectile": TORCH_IQL_EXPECTILE,
        "iql_advantage_temperature": TORCH_IQL_ADVANTAGE_TEMPERATURE,
        "iql_advantage_weight_max": TORCH_IQL_ADVANTAGE_WEIGHT_MAX,
        "q_loss_weight": TORCH_IQL_Q_LOSS_WEIGHT,
        "value_loss_weight": TORCH_IQL_VALUE_LOSS_WEIGHT,
        "actor_loss_weight": TORCH_IQL_ACTOR_LOSS_WEIGHT,
    }
    training_metrics.update(
        _iql_training_diagnostics(
            torch,
            model,
            x,
            next_x,
            y,
            reward_target,
            done_tensor,
            mask_tensor,
        )
    )
    return _serialize_model(
        model,
        training_metrics=training_metrics,
    )


def _train_torch_actor_critic_network(
    records: tuple[dict[str, object], ...],
    *,
    advantage_weighted: bool,
) -> dict[str, object]:
    torch = _load_torch()
    if not records:
        raise ValueError("torch actor-critic trainer requires at least one record")

    torch.manual_seed(TORCH_NEURAL_SEED)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    features: list[list[float]] = []
    labels: list[int] = []
    rewards: list[float] = []
    action_labels: list[str] = []
    action_index = {action: index for index, action in enumerate(ACTION_NAMES)}
    for record in records:
        label = _record_label(record)
        features.append(_record_observation_values(record))
        labels.append(action_index[label])
        action_labels.append(label)
        rewards.append(_record_reward_total(record))

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    reward_target = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    class_weights = _balanced_class_weights(torch, labels)
    advantage_stats: dict[str, object] = {
        "actor_weighting_policy": "class_balanced_cross_entropy_v1",
        "advantage_weighted": False,
    }
    advantage_weights = None
    if advantage_weighted:
        raw_advantage_weights, advantage_stats = _contextual_advantage_weights(
            records,
            action_labels=action_labels,
            rewards=rewards,
        )
        advantage_weights = torch.tensor(raw_advantage_weights, dtype=torch.float32)
    model = _TorchActorCritic(torch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TORCH_NEURAL_LEARNING_RATE,
        weight_decay=TORCH_NEURAL_WEIGHT_DECAY,
    )
    final_loss = 0.0
    final_actor_loss = 0.0
    final_value_loss = 0.0
    for _ in range(TORCH_NEURAL_EPOCHS):
        optimizer.zero_grad()
        logits, action_values, state_values = model(x)
        selected_action_values = action_values.gather(1, y.unsqueeze(1))
        if advantage_weights is None:
            actor_loss = torch.nn.functional.cross_entropy(
                logits,
                y,
                weight=class_weights,
            )
        else:
            actor_losses = torch.nn.functional.cross_entropy(
                logits,
                y,
                weight=class_weights,
                reduction="none",
            )
            actor_loss = (
                (actor_losses * advantage_weights).sum()
                / advantage_weights.sum().clamp_min(1e-9)
            )
        action_value_loss = torch.nn.functional.mse_loss(
            selected_action_values,
            reward_target,
        )
        state_value_loss = torch.nn.functional.mse_loss(
            state_values,
            reward_target,
        )
        value_loss = action_value_loss + state_value_loss
        loss = actor_loss + TORCH_NEURAL_VALUE_LOSS_WEIGHT * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        final_actor_loss = float(actor_loss.detach().cpu().item())
        final_value_loss = float(value_loss.detach().cpu().item())

    training_metrics = {
        "epochs": TORCH_NEURAL_EPOCHS,
        "final_loss": _round(final_loss),
        "final_actor_loss": _round(final_actor_loss),
        "final_value_loss": _round(final_value_loss),
        "learning_rate": TORCH_NEURAL_LEARNING_RATE,
        "weight_decay": TORCH_NEURAL_WEIGHT_DECAY,
        "value_loss_weight": TORCH_NEURAL_VALUE_LOSS_WEIGHT,
        "class_weight_min": TORCH_NEURAL_CLASS_WEIGHT_MIN,
        "class_weight_max": TORCH_NEURAL_CLASS_WEIGHT_MAX,
        "class_weights": {
            action: _round(float(class_weights[index].detach().cpu().item()))
            for index, action in enumerate(ACTION_NAMES)
        },
    }
    training_metrics.update(advantage_stats)
    training_metrics.update(
        _training_diagnostics(
            torch,
            model,
            x,
            y,
            reward_target,
        )
    )

    return _serialize_model(
        model,
        training_metrics=training_metrics,
    )


class _TorchActorCritic:
    def __new__(cls, torch: Any) -> Any:
        class TorchActorCritic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden = torch.nn.Linear(
                    OBSERVATION_INPUT_VECTOR_SIZE,
                    TORCH_NEURAL_HIDDEN_UNITS,
                )
                self.actor = torch.nn.Linear(
                    TORCH_NEURAL_HIDDEN_UNITS,
                    len(ACTION_NAMES),
                )
                self.action_value = torch.nn.Linear(
                    TORCH_NEURAL_HIDDEN_UNITS,
                    len(ACTION_NAMES),
                )
                self.state_value = torch.nn.Linear(TORCH_NEURAL_HIDDEN_UNITS, 1)

            def forward(self, x: Any) -> tuple[Any, Any, Any]:
                hidden = torch.tanh(self.hidden(x))
                return (
                    self.actor(hidden),
                    self.action_value(hidden),
                    self.state_value(hidden),
                )

        return TorchActorCritic()


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(MIND_ML_INSTALL_HINT) from exc
    return torch


def _balanced_class_weights(torch: Any, labels: list[int]) -> Any:
    counts = [0 for _ in ACTION_NAMES]
    for label in labels:
        counts[int(label)] += 1
    total = float(sum(counts))
    action_count = float(len(ACTION_NAMES))
    weights = [
        _clamp(
            (total / (action_count * float(max(count, 1)))) ** 0.5,
            TORCH_NEURAL_CLASS_WEIGHT_MIN,
            TORCH_NEURAL_CLASS_WEIGHT_MAX,
        )
        for count in counts
    ]
    mean_weight = sum(weights) / float(len(weights))
    if mean_weight > 0.0:
        weights = [weight / mean_weight for weight in weights]
    return torch.tensor(weights, dtype=torch.float32)


def _action_mask_values(action_mask: dict[str, bool]) -> list[bool]:
    return [bool(action_mask.get(action, False)) for action in ACTION_NAMES]


def _masked_logits(torch: Any, logits: Any, action_masks: Any) -> Any:
    return logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)


def _contextual_advantage_weights(
    records: tuple[dict[str, object], ...],
    *,
    action_labels: list[str],
    rewards: list[float],
) -> tuple[list[float], dict[str, object]]:
    context_counts: Counter[str] = Counter()
    context_reward_sums: Counter[str] = Counter()
    context_action_counts: dict[str, Counter[str]] = {}
    context_action_reward_sums: dict[str, Counter[str]] = {}
    record_keys: list[tuple[str, ...]] = []
    for record, label, reward in zip(records, action_labels, rewards, strict=True):
        try:
            keys = feature_keys_from_record(record)
        except ValueError:
            keys = ()
        record_keys.append(keys)
        for key in keys:
            context_counts[key] += 1
            context_reward_sums[key] += reward
            context_action_counts.setdefault(key, Counter())[label] += 1
            context_action_reward_sums.setdefault(key, Counter())[label] += reward

    global_count = len(records)
    global_reward_sum = sum(float(reward) for reward in rewards)
    global_action_counts = Counter(action_labels)
    global_action_reward_sums: Counter[str] = Counter()
    for label, reward in zip(action_labels, rewards, strict=True):
        global_action_reward_sums[label] += reward

    weights: list[float] = []
    contextual_count = 0
    for keys, label in zip(record_keys, action_labels, strict=True):
        context_key = _select_advantage_context_key(
            keys,
            label=label,
            context_counts=context_counts,
            context_action_counts=context_action_counts,
        )
        if context_key is None:
            total = global_count
            reward_sum = global_reward_sum
            action_count = global_action_counts[label]
            action_reward_sum = global_action_reward_sums[label]
        else:
            contextual_count += 1
            total = context_counts[context_key]
            reward_sum = context_reward_sums[context_key]
            action_count = context_action_counts[context_key][label]
            action_reward_sum = context_action_reward_sums[context_key][label]
        context_mean_reward = reward_sum / float(max(total, 1))
        action_mean_reward = (
            action_reward_sum / float(action_count)
            if action_count >= TORCH_ADVANTAGE_MIN_ACTION_SUPPORT
            else context_mean_reward
        )
        advantage = action_mean_reward - context_mean_reward
        weight = math.exp(advantage / TORCH_ADVANTAGE_TEMPERATURE)
        clipped_weight = _clamp(
            weight,
            TORCH_ADVANTAGE_WEIGHT_MIN,
            TORCH_ADVANTAGE_WEIGHT_MAX,
        )
        weights.append(
            (1.0 - TORCH_ADVANTAGE_WEIGHT_BLEND)
            + TORCH_ADVANTAGE_WEIGHT_BLEND * clipped_weight
        )

    mean_weight = sum(weights) / float(len(weights)) if weights else 1.0
    if mean_weight > 0.0:
        weights = [weight / mean_weight for weight in weights]
    return weights, {
        "actor_weighting_policy": (
            "contextual_advantage_weighted_cross_entropy_v1"
        ),
        "advantage_weighted": True,
        "advantage_temperature": TORCH_ADVANTAGE_TEMPERATURE,
        "advantage_weight_blend": TORCH_ADVANTAGE_WEIGHT_BLEND,
        "advantage_weight_min": TORCH_ADVANTAGE_WEIGHT_MIN,
        "advantage_weight_max": TORCH_ADVANTAGE_WEIGHT_MAX,
        "advantage_min_context_records": TORCH_ADVANTAGE_MIN_CONTEXT_RECORDS,
        "advantage_min_action_support": TORCH_ADVANTAGE_MIN_ACTION_SUPPORT,
        "advantage_contextual_record_rate": _round(
            contextual_count / float(len(weights)) if weights else 0.0
        ),
        "advantage_sample_weight_mean": _round(
            sum(weights) / float(len(weights)) if weights else 0.0
        ),
        "advantage_sample_weight_min_observed": _round(
            min(weights) if weights else 0.0
        ),
        "advantage_sample_weight_max_observed": _round(
            max(weights) if weights else 0.0
        ),
    }


def _select_advantage_context_key(
    keys: tuple[str, ...],
    *,
    label: str,
    context_counts: Counter[str],
    context_action_counts: dict[str, Counter[str]],
) -> str | None:
    for key in keys:
        if context_counts[key] < TORCH_ADVANTAGE_MIN_CONTEXT_RECORDS:
            continue
        action_counts = context_action_counts.get(key, Counter())
        if action_counts[label] < TORCH_ADVANTAGE_MIN_ACTION_SUPPORT:
            continue
        supported_actions = [
            action
            for action, count in action_counts.items()
            if count >= TORCH_ADVANTAGE_MIN_ACTION_SUPPORT
        ]
        if len(supported_actions) < 2:
            continue
        return key
    return None


def _training_diagnostics(
    torch: Any,
    model: Any,
    x: Any,
    y: Any,
    reward_target: Any,
) -> dict[str, object]:
    with torch.no_grad():
        logits, action_values, state_values = model(x)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        top_values = probabilities.topk(k=2, dim=1).values
        selected_action_values = action_values.gather(1, y.unsqueeze(1))

        action_counts: dict[str, int] = {}
        action_accuracy: dict[str, float | None] = {}
        for index, action in enumerate(ACTION_NAMES):
            action_mask = y == index
            count = int(action_mask.sum().detach().cpu().item())
            action_counts[action] = count
            if count <= 0:
                action_accuracy[action] = None
                continue
            correct = (predictions[action_mask] == y[action_mask]).float()
            action_accuracy[action] = _round(float(correct.mean().cpu().item()))

        return {
            "python_version": platform.python_version(),
            "ml_dependency_versions": _optional_dependency_versions(),
            "actor_accuracy": _round(
                float((predictions == y).float().mean().cpu().item())
            ),
            "actor_mean_top_score": _round(
                float(top_values[:, 0].mean().cpu().item())
            ),
            "actor_mean_top_margin": _round(
                float((top_values[:, 0] - top_values[:, 1]).mean().cpu().item())
            ),
            "action_counts": action_counts,
            "action_accuracy": action_accuracy,
            "action_value_mean_abs_error": _round(
                float(
                    torch.mean(
                        torch.abs(selected_action_values - reward_target)
                    )
                    .cpu()
                    .item()
                )
            ),
            "state_value_mean_abs_error": _round(
                float(
                    torch.mean(torch.abs(state_values - reward_target))
                    .cpu()
                    .item()
                )
            ),
        }


def _iql_training_diagnostics(
    torch: Any,
    model: Any,
    x: Any,
    next_x: Any,
    y: Any,
    reward_target: Any,
    done_tensor: Any,
    mask_tensor: Any,
) -> dict[str, object]:
    with torch.no_grad():
        logits, action_values, state_values = model(x)
        masked_logits = _masked_logits(torch, logits, mask_tensor)
        probabilities = torch.nn.functional.softmax(masked_logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        top_values = probabilities.topk(k=2, dim=1).values
        selected_action_values = action_values.gather(1, y.unsqueeze(1))
        _, _, next_state_values = model(next_x)
        q_target = reward_target + (
            TORCH_IQL_DISCOUNT * (1.0 - done_tensor) * next_state_values
        )

        action_counts: dict[str, int] = {}
        action_accuracy: dict[str, float | None] = {}
        for index, action in enumerate(ACTION_NAMES):
            action_mask = y == index
            count = int(action_mask.sum().detach().cpu().item())
            action_counts[action] = count
            if count <= 0:
                action_accuracy[action] = None
                continue
            correct = (predictions[action_mask] == y[action_mask]).float()
            action_accuracy[action] = _round(float(correct.mean().cpu().item()))

        return {
            "python_version": platform.python_version(),
            "ml_dependency_versions": _optional_dependency_versions(),
            "actor_accuracy": _round(
                float((predictions == y).float().mean().cpu().item())
            ),
            "actor_mean_top_score": _round(
                float(top_values[:, 0].mean().cpu().item())
            ),
            "actor_mean_top_margin": _round(
                float((top_values[:, 0] - top_values[:, 1]).mean().cpu().item())
            ),
            "action_counts": action_counts,
            "action_accuracy": action_accuracy,
            "q_value_mean_abs_error": _round(
                float(
                    torch.mean(torch.abs(selected_action_values - q_target))
                    .cpu()
                    .item()
                )
            ),
            "state_value_mean_abs_error": _round(
                float(
                    torch.mean(torch.abs(state_values - q_target))
                    .cpu()
                    .item()
                )
            ),
        }


def _optional_dependency_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "torchrl",
        "tensordict",
        "gymnasium",
        "pettingzoo",
        "minari",
        "d3rlpy",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _serialize_model(
    model: Any,
    *,
    training_metrics: dict[str, object],
) -> dict[str, object]:
    lower, upper = REWARD_TOTAL_BOUNDS
    return {
        "hidden_weights": _round_matrix(model.hidden.weight.detach().cpu().tolist()),
        "hidden_bias": _round_vector(model.hidden.bias.detach().cpu().tolist()),
        "actor_output_weights": _action_weight_map(model.actor.weight),
        "actor_output_bias": _action_bias_map(
            model.actor.bias,
            bounded_by_reward=False,
        ),
        "action_value_output_weights": _action_weight_map(model.action_value.weight),
        "action_value_output_bias": _action_bias_map(
            model.action_value.bias,
            bounded_by_reward=True,
            lower=lower,
            upper=upper,
        ),
        "state_value_weights": _round_vector(
            model.state_value.weight.detach().cpu().tolist()[0]
        ),
        "state_value_bias": _round(
            _clamp(
                float(model.state_value.bias.detach().cpu().tolist()[0]),
                lower,
                upper,
            )
        ),
        "training_metrics": training_metrics,
    }


def _action_weight_map(parameter: Any) -> dict[str, list[float]]:
    rows = parameter.detach().cpu().tolist()
    return {
        action: _round_vector(rows[index])
        for index, action in enumerate(ACTION_NAMES)
    }


def _action_bias_map(
    parameter: Any,
    *,
    bounded_by_reward: bool,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float]:
    values = parameter.detach().cpu().tolist()
    return {
        action: _round(
            _clamp(float(values[index]), float(lower), float(upper))
            if bounded_by_reward and lower is not None and upper is not None
            else float(values[index])
        )
        for index, action in enumerate(ACTION_NAMES)
    }


def _round_matrix(rows: list[list[float]]) -> list[list[float]]:
    return [_round_vector(row) for row in rows]


def _round_vector(values: list[float]) -> list[float]:
    return [_round(value) for value in values]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round(value: float) -> float:
    return round(float(value), 6)
