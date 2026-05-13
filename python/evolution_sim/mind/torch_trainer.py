from __future__ import annotations

import importlib.metadata as importlib_metadata
import math
import platform
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    decode_observation_input,
)
from evolution_sim.env.runtime.trajectory import REWARD_TOTAL_BOUNDS
from evolution_sim.mind.carrion_counterfactual_labels import (
    MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
    build_trajectory_transitions,
    discounted_return_targets,
)
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
    TORCH_IQL_BEHAVIOR_ANCHOR_LOSS_WEIGHT,
    TORCH_IQL_BEHAVIOR_ANCHOR_POLICY,
    TORCH_IQL_CQL_LOSS_WEIGHT,
    TORCH_IQL_CQL_REGULARIZATION_POLICY,
    TORCH_IQL_CQL_TEMPERATURE,
    TORCH_IQL_DISCOUNT,
    TORCH_IQL_EXPECTILE,
    TORCH_IQL_GUARD_FEEDBACK_LOSS_WEIGHT,
    TORCH_IQL_GUARD_FEEDBACK_MARGIN,
    TORCH_IQL_GUARD_FEEDBACK_POLICY,
    TORCH_IQL_HARD_GUARD_FEEDBACK_WEIGHT,
    TORCH_IQL_HEURISTIC_DELEGATE_FEEDBACK_WEIGHT,
    TORCH_IQL_LEARNED_REPLAY_GUARDED_ACTION_WEIGHT,
    TORCH_IQL_LEARNED_REPLAY_SELF_ACTION_WEIGHT,
    TORCH_IQL_LEARNED_REPLAY_WEIGHT_POLICY,
    TORCH_IQL_ONLINE_UPDATE_FEEDBACK_LOSS_WEIGHT,
    TORCH_IQL_ONLINE_UPDATE_FEEDBACK_MARGIN,
    TORCH_IQL_ONLINE_UPDATE_FEEDBACK_POLICY,
    TORCH_IQL_Q_LOSS_WEIGHT,
    TORCH_IQL_VALUE_LOSS_WEIGHT,
    TORCH_IQL_DETACHED_VIABILITY_REPRESENTATION_POLICY,
    TORCH_IQL_SHARED_VIABILITY_REPRESENTATION_POLICY,
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
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.learned_policy import (
    HEURISTIC_DELEGATE_POLICY,
    HEURISTIC_GUARD_POLICY,
)
from evolution_sim.mind.viability import (
    VIABILITY_FLOOR_RISK_RATIO,
    VIABILITY_HEALTH_FLOOR_RISK_RATIO,
    VIABILITY_ACTION_HEAD_POLICY,
    VIABILITY_ACTION_SUPERVISION_POLICY,
    VIABILITY_COMPONENT_NAMES,
    VIABILITY_HEAD_POLICY,
    VIABILITY_SUPPRESSION_COMPONENT,
    build_viability_component_targets,
)

MIND_ML_INSTALL_HINT = (
    "PyTorch Mind training requires optional ML dependencies. "
    "Install them with: python3 -m pip install -r requirements-mind-ml.txt"
)
TORCH_DEVICE_POLICY = "torch_device_resolution_v1"
TORCH_DEVICE_CHOICES = {"cpu", "cuda", "mps", "auto"}
TORCH_IQL_VIABILITY_LOSS_WEIGHT = 0.25
TORCH_IQL_ACTION_VIABILITY_LOSS_WEIGHT = 0.25
TORCH_IQL_VIABILITY_POS_WEIGHT_MAX = 8.0
TORCH_IQL_VIABILITY_POS_WEIGHT_POLICY = "observed_state_component_balance_v1"
TORCH_IQL_ACTION_VIABILITY_POS_WEIGHT_POLICY = (
    "observed_action_component_balance_v1"
)
TORCH_IQL_STATE_SUPPRESSION_SUPERVISION_POLICY = (
    "suppression_owned_by_action_viability_head_v1"
)
TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_POLICY = (
    "viability_safe_logged_action_margin_anchor_v1"
)
TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_FINETUNE_POLICY = (
    "viability_safe_logged_action_margin_finetune_carryover_v1"
)
TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_TARGET = 1.0
TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_LOSS_WEIGHT = 0.15
TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_FINETUNE_LOSS_WEIGHT = (
    TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_LOSS_WEIGHT
)
TORCH_IQL_ACTOR_WEIGHTING_POLICY = "iql_masked_advantage_weighted_bc_v1"
TORCH_IQL_CALIBRATED_ACTOR_WEIGHTING_POLICY = (
    "behavior_anchored_batch_standardized_iql_advantage_weighted_bc_v2"
)
TORCH_IQL_CONSTRAINT_AWARE_ACTOR_WEIGHTING_POLICY = (
    "observed_viability_safe_iql_actor_weight_filter_v1"
)
TORCH_IQL_CALIBRATED_ADVANTAGE_TEMPERATURE = 1.0
TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON = 1e-6
TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_BLEND = 0.35
TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_MIN = 0.25
TORCH_IQL_CONSTRAINT_AWARE_ACTOR_WEIGHT_MIN = 0.35
TORCH_IQL_CONSTRAINT_AWARE_ACTOR_COMPONENT_POLICY = (
    "logged_action_observed_non_suppression_components_v1"
)
TORCH_IQL_RISK_ADJUSTED_ACTOR_EXTRACTION_POLICY = (
    "detached_q_minus_action_viability_risk_actor_distillation_v1"
)
TORCH_IQL_ACTION_RISK_SCORE_POLICY = (
    "detached_max_non_suppression_action_viability_risk_v1"
)
TORCH_IQL_RISK_ADJUSTED_ACTOR_LOSS_WEIGHT = 0.15
TORCH_IQL_RISK_ADJUSTED_ACTOR_RISK_PENALTY = 0.75
TORCH_IQL_RISK_ADJUSTED_ACTOR_TEMPERATURE = 0.35
TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EXTRACTION_POLICY = (
    "calibrated_supported_actor_extraction_v1"
)
TORCH_IQL_CONTEXTUAL_BEHAVIOR_SUPPORTED_ACTOR_EXTRACTION_POLICY = (
    "contextual_behavior_proximity_supported_actor_extraction_v2"
)
TORCH_IQL_CALIBRATED_SUPPORTED_RISK_CALIBRATION_POLICY = (
    "calibration_bank_action_component_bias_v1"
)
TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_POLICY = (
    "calibration_bank_standardized_legal_q_minus_v_v1"
)
TORCH_IQL_CONTEXTUAL_SUPPORT_POLICY = (
    "calibration_bank_context_action_support_fallback_v1"
)
TORCH_IQL_BEHAVIOR_PROXIMITY_POLICY = (
    "contextual_logged_action_distribution_proximity_v1"
)
TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_REGULARIZATION_POLICY = (
    "contextual_behavior_prior_cross_entropy_actor_regularization_v1"
)
TORCH_IQL_ACTION_DISTRIBUTION_REGULARIZATION_POLICY = (
    "batch_logged_sharp_action_marginal_kl_v2"
)
TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY = (
    "calibration_bank_top1_actor_bias_control_v1"
)
TORCH_IQL_GUARD_FEEDBACK_FINETUNE_POLICY = (
    "runtime_suppressed_learned_action_margin_finetune_carryover_v1"
)
TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_LOSS_WEIGHT = 0.18
TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_LOSS_WEIGHT = 0.08
TORCH_IQL_ACTION_DISTRIBUTION_LOSS_WEIGHT = 0.35
TORCH_IQL_ACTION_DISTRIBUTION_TEMPERATURE = 0.25
TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_SHARE = 0.5
TORCH_IQL_ROLLOUT_STATE_ACTION_BIAS_STEP = 0.08
TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_BIAS_DELTA = 1.25
TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_ITERATIONS = 64
TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_MIN_MASS = 1e-6
TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EPOCHS = 64
TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT = 16
TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT = 8
TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_BEHAVIOR_PROBABILITY = 0.04
TORCH_IQL_CONTEXTUAL_SUPPORTED_FAMILY_CONVERSION_RATIO = 0.5
TORCH_IQL_CONTEXTUAL_SUPPORTED_RESOURCE_CONVERSION_MIN_PROBABILITY = 0.12
TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_TARGET_ACTION_EXPANSION_RATIO = 1.25
TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_FAMILY_CONVERSION_RATE = 0.22
TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_MOVEMENT_STAY_RESOURCE_RATE = 0.18
TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD = 0.35
TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD = 0.0
TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY = 0.75
TORCH_IQL_RETURN_CALIBRATION_POLICY = "discounted_return_q_v_auxiliary_v1"
TORCH_IQL_RETURN_CALIBRATION_LOSS_WEIGHT = 0.05
TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_POLICY = (
    "runtime_suppressed_action_q_v_margin_calibration_v1"
)
TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_LOSS_WEIGHT = 0.12
TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_MARGIN = 0.05
TORCH_IQL_SUPPRESSION_CRITIC_STATE_ADVANTAGE_WEIGHT = 0.5
TORCH_IQL_COUNTERFACTUAL_LABEL_SUPERVISION_POLICY = (
    "carrion_counterfactual_terminal_action_value_supervision_v1"
)
TORCH_IQL_COUNTERFACTUAL_LABEL_WEIGHT_POLICY = (
    "one_plus_terminal_value_weight_scale_v1"
)
TORCH_IQL_COUNTERFACTUAL_LABEL_DEFAULT_WEIGHT_SCALE = 2.0
TORCH_IQL_COUNTERFACTUAL_ACTION_VALUE_LOSS_WEIGHT = 0.08
TORCH_IQL_COUNTERFACTUAL_VIABILITY_LOSS_WEIGHT = 0.05
TORCH_IQL_COUNTERFACTUAL_ACTION_VIABILITY_LOSS_WEIGHT = 0.05


def train_torch_actor_critic_network(
    records: tuple[dict[str, object], ...],
    *,
    torch_device: str = "cpu",
) -> dict[str, object]:
    return _train_torch_actor_critic_network(
        records,
        advantage_weighted=False,
        torch_device=torch_device,
    )


def train_torch_advantage_actor_critic_network(
    records: tuple[dict[str, object], ...],
    *,
    torch_device: str = "cpu",
) -> dict[str, object]:
    return _train_torch_actor_critic_network(
        records,
        advantage_weighted=True,
        torch_device=torch_device,
    )


def train_torch_discrete_iql_network(
    records: tuple[dict[str, object], ...],
    *,
    detach_viability_heads: bool = False,
    behavior_margin_anchor: bool = False,
    calibrated_actor_extraction: bool = False,
    constraint_aware_actor_extraction: bool = False,
    risk_adjusted_actor_extraction: bool = False,
    calibrated_supported_actor_extraction: bool = False,
    contextual_behavior_supported_actor_extraction: bool = False,
    contextual_behavior_prior_regularization: bool = False,
    action_distribution_regularization: bool = False,
    rollout_state_action_calibration: bool = False,
    contextual_behavior_prior_loss_weight: float | None = None,
    action_distribution_loss_weight: float | None = None,
    action_distribution_temperature: float | None = None,
    rollout_state_action_max_share: float | None = None,
    rollout_state_action_bias_step: float | None = None,
    rollout_state_action_max_bias_delta: float | None = None,
    calibration_records: tuple[dict[str, object], ...] | None = None,
    calibration_validation_records: tuple[dict[str, object], ...] | None = None,
    return_calibration: bool = False,
    suppression_critic_calibration: bool = False,
    counterfactual_label_report: Mapping[str, object] | None = None,
    counterfactual_label_weight_scale: float = (
        TORCH_IQL_COUNTERFACTUAL_LABEL_DEFAULT_WEIGHT_SCALE
    ),
    torch_device: str = "cpu",
) -> dict[str, object]:
    torch = _load_torch()
    torch_device_metadata = _resolve_torch_device_metadata(
        torch,
        requested_device=torch_device,
    )
    device = torch.device(torch_device_metadata["resolved_device"])
    transitions = build_trajectory_transitions(records)
    if not transitions:
        raise ValueError("torch discrete IQL trainer requires at least one transition")
    uses_calibrated_supported_extraction = (
        calibrated_supported_actor_extraction
        or contextual_behavior_supported_actor_extraction
    )
    uses_contextual_prior_regularization = (
        contextual_behavior_prior_regularization
    )
    uses_actor_finetune = (
        uses_calibrated_supported_extraction
        or uses_contextual_prior_regularization
    )
    if (
        uses_calibrated_supported_extraction
        or uses_contextual_prior_regularization
        or rollout_state_action_calibration
    ) and not calibration_records:
        raise ValueError(
            "calibrated supported actor extraction and contextual behavior "
            "prior regularization and rollout-state action calibration "
            "require a separate calibration trajectory bank"
        )
    if (
        contextual_behavior_supported_actor_extraction
        and not calibration_validation_records
    ):
        raise ValueError(
            "contextual behavior supported actor extraction requires a "
            "separate calibration validation trajectory bank"
        )
    resolved_contextual_behavior_prior_loss_weight = _nonnegative_float_override(
        contextual_behavior_prior_loss_weight,
        default=TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_LOSS_WEIGHT,
        field="contextual_behavior_prior_loss_weight",
    )
    resolved_action_distribution_loss_weight = _nonnegative_float_override(
        action_distribution_loss_weight,
        default=TORCH_IQL_ACTION_DISTRIBUTION_LOSS_WEIGHT,
        field="action_distribution_loss_weight",
    )
    resolved_action_distribution_temperature = _positive_float_override(
        action_distribution_temperature,
        default=TORCH_IQL_ACTION_DISTRIBUTION_TEMPERATURE,
        field="action_distribution_temperature",
    )
    resolved_rollout_state_action_max_share = _probability_float_override(
        rollout_state_action_max_share,
        default=TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_SHARE,
        field="rollout_state_action_max_share",
    )
    resolved_rollout_state_action_bias_step = _positive_float_override(
        rollout_state_action_bias_step,
        default=TORCH_IQL_ROLLOUT_STATE_ACTION_BIAS_STEP,
        field="rollout_state_action_bias_step",
    )
    resolved_rollout_state_action_max_bias_delta = _nonnegative_float_override(
        rollout_state_action_max_bias_delta,
        default=TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_BIAS_DELTA,
        field="rollout_state_action_max_bias_delta",
    )

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
    returns = discounted_return_targets(
        transitions,
        discount=TORCH_IQL_DISCOUNT,
    )
    dones = [1.0 if transition.done else 0.0 for transition in transitions]
    action_masks = [
        _action_mask_values(transition.action_mask)
        for transition in transitions
    ]
    counterfactual_supervision = _counterfactual_label_supervision(
        records,
        transitions,
        counterfactual_label_report,
        action_index=action_index,
        weight_scale=counterfactual_label_weight_scale,
    )
    guard_feedback = [
        _guard_feedback_record(transition, action_index=action_index)
        for transition in transitions
    ]
    base_replay_weights = [
        _replay_weight_record(transition)
        for transition in transitions
    ]
    replay_weights = [
        base_weight * counterfactual_weight
        for base_weight, counterfactual_weight in zip(
            base_replay_weights,
            counterfactual_supervision["sample_weights"],  # type: ignore[index]
            strict=True,
        )
    ]
    online_feedback = [
        _online_update_feedback_record(transition, action_index=action_index)
        for transition in transitions
    ]
    component_targets, _aggregate_targets, survival_horizons = (
        build_viability_component_targets(records)
    )
    if len(component_targets) != len(transitions):
        raise ValueError(
            "torch discrete IQL viability targets must align with transitions"
        )
    viability_targets = [
        [
            1.0 if components[component] else 0.0
            for component in VIABILITY_COMPONENT_NAMES
        ]
        for components in component_targets
    ]
    viability_observed = _state_viability_observed(viability_targets)
    behavior_margin_anchor_weights = _behavior_margin_anchor_weights(
        viability_targets,
        labels=labels,
        action_masks=action_masks,
    )
    viability_positive_counts = _observed_component_positive_counts(
        viability_targets,
        viability_observed,
    )
    viability_observed_counts = _observed_component_counts(viability_observed)
    action_viability_targets, action_viability_observed = (
        _action_viability_supervision(
            transitions,
            component_targets,
            action_index=action_index,
        )
    )
    action_viability_positive_counts = (
        _action_viability_component_positive_counts(
            action_viability_targets,
            action_viability_observed,
        )
    )
    action_viability_observed_counts = (
        _action_viability_component_observed_counts(action_viability_observed)
    )
    constraint_actor_weights, constraint_actor_stats = (
        _constraint_aware_actor_weights(
            action_viability_targets,
            labels=labels,
        )
    )
    viability_representation_policy = (
        TORCH_IQL_DETACHED_VIABILITY_REPRESENTATION_POLICY
        if detach_viability_heads
        else TORCH_IQL_SHARED_VIABILITY_REPRESENTATION_POLICY
    )

    x = torch.tensor(features, dtype=torch.float32, device=device)
    next_x = torch.tensor(next_features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    reward_target = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    return_target = torch.tensor(
        returns,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    done_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=device)
    guard_feedback_actions = torch.tensor(
        [feedback[0] for feedback in guard_feedback],
        dtype=torch.long,
        device=device,
    )
    guard_feedback_weights = torch.tensor(
        [feedback[1] for feedback in guard_feedback],
        dtype=torch.float32,
        device=device,
    )
    replay_weight_tensor = torch.tensor(
        replay_weights,
        dtype=torch.float32,
        device=device,
    )
    behavior_margin_anchor_weight_tensor = torch.tensor(
        behavior_margin_anchor_weights,
        dtype=torch.float32,
        device=device,
    )
    online_feedback_actions = torch.tensor(
        [feedback[0] for feedback in online_feedback],
        dtype=torch.long,
        device=device,
    )
    online_feedback_signals = torch.tensor(
        [feedback[1] for feedback in online_feedback],
        dtype=torch.float32,
        device=device,
    )
    viability_target_tensor = torch.tensor(
        viability_targets,
        dtype=torch.float32,
        device=device,
    )
    viability_observed_tensor = torch.tensor(
        viability_observed,
        dtype=torch.float32,
        device=device,
    )
    action_viability_target_tensor = torch.tensor(
        action_viability_targets,
        dtype=torch.float32,
        device=device,
    )
    action_viability_observed_tensor = torch.tensor(
        action_viability_observed,
        dtype=torch.float32,
        device=device,
    )
    constraint_actor_weight_tensor = torch.tensor(
        constraint_actor_weights,
        dtype=torch.float32,
        device=device,
    )
    counterfactual_loss_weight_tensor = torch.tensor(
        counterfactual_supervision["loss_weights"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    )
    counterfactual_action_value_target_tensor = torch.tensor(
        counterfactual_supervision["action_value_targets"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    counterfactual_viability_target_tensor = torch.tensor(
        counterfactual_supervision["viability_targets"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    )
    counterfactual_viability_observed_tensor = torch.tensor(
        counterfactual_supervision["viability_observed"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    )
    counterfactual_action_viability_target_tensor = torch.tensor(
        counterfactual_supervision["action_viability_targets"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    )
    counterfactual_action_viability_observed_tensor = torch.tensor(
        counterfactual_supervision["action_viability_observed"],  # type: ignore[index]
        dtype=torch.float32,
        device=device,
    )
    viability_pos_weights = _viability_positive_weights(
        torch,
        viability_targets,
        viability_observed,
        device=device,
    )
    action_viability_pos_weights = _action_viability_positive_weights(
        torch,
        action_viability_targets,
        action_viability_observed,
        device=device,
    )
    class_weights = _balanced_class_weights(torch, labels, device=device)
    model = _TorchActorCritic(torch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TORCH_NEURAL_LEARNING_RATE,
        weight_decay=TORCH_NEURAL_WEIGHT_DECAY,
    )

    final_loss = 0.0
    final_actor_loss = 0.0
    final_q_loss = 0.0
    final_value_loss = 0.0
    final_cql_loss = 0.0
    final_behavior_anchor_loss = 0.0
    final_behavior_margin_anchor_loss = 0.0
    final_guard_feedback_loss = 0.0
    final_online_update_feedback_loss = 0.0
    final_return_calibration_loss = 0.0
    final_suppression_critic_calibration_loss = 0.0
    final_counterfactual_action_value_loss = 0.0
    final_counterfactual_viability_loss = 0.0
    final_counterfactual_action_viability_loss = 0.0
    final_action_distribution_loss = 0.0
    final_risk_adjusted_actor_loss = 0.0
    final_calibrated_supported_actor_loss = 0.0
    final_contextual_behavior_prior_loss = 0.0
    final_actor_finetune_guard_feedback_loss = 0.0
    final_actor_finetune_behavior_margin_anchor_loss = 0.0
    final_viability_loss = 0.0
    final_action_viability_loss = 0.0
    final_risk_adjusted_actor_stats: dict[str, float] = (
        _empty_risk_adjusted_actor_stats()
    )
    calibrated_supported_report: dict[str, object] = (
        _empty_calibrated_supported_report()
    )
    calibrated_supported_validation_report: dict[str, object] = (
        _empty_calibrated_supported_report()
    )
    contextual_supported_report: dict[str, object] = (
        _empty_contextual_supported_report()
    )
    calibrated_supported_stats: dict[str, object] = (
        _empty_calibrated_supported_stats()
    )
    contextual_behavior_prior_stats: dict[str, object] = (
        _empty_contextual_behavior_prior_stats()
    )
    action_distribution_stats: dict[str, object] = (
        _empty_action_distribution_stats(
            row_count=len(transitions),
            temperature=resolved_action_distribution_temperature,
        )
    )
    rollout_state_action_calibration_report: dict[str, object] = (
        _empty_rollout_state_action_calibration_report()
    )
    rollout_state_action_calibration_validation_report: dict[str, object] = (
        _empty_rollout_state_action_top1_report()
    )
    final_actor_weight_mean = 0.0
    final_actor_weight_min = 0.0
    final_actor_weight_max = 0.0
    final_actor_advantage_mean = 0.0
    final_actor_advantage_scale = 0.0
    for _ in range(TORCH_NEURAL_EPOCHS):
        optimizer.zero_grad()
        hidden = torch.tanh(model.hidden(x))
        logits = model.actor(hidden)
        action_values = model.action_value(hidden)
        state_values = model.state_value(hidden)
        viability_hidden = hidden.detach() if detach_viability_heads else hidden
        viability_logits = model.viability(viability_hidden)
        action_viability_logits = model.action_viability(viability_hidden).view(
            -1,
            len(ACTION_NAMES),
            len(VIABILITY_COMPONENT_NAMES),
        )
        masked_logits = _masked_logits(torch, logits, mask_tensor)
        selected_action_values = action_values.gather(1, y.unsqueeze(1))
        with torch.no_grad():
            _, _, next_state_values = model(next_x)
            q_target = reward_target + (
                TORCH_IQL_DISCOUNT * (1.0 - done_tensor) * next_state_values
            )
        q_loss = _weighted_mean(
            torch.nn.functional.mse_loss(
                selected_action_values,
                q_target,
                reduction="none",
            ),
            replay_weight_tensor.unsqueeze(1),
        )
        value_error = selected_action_values.detach() - state_values
        expectile_weights = torch.where(
            value_error > 0,
            torch.full_like(value_error, TORCH_IQL_EXPECTILE),
            torch.full_like(value_error, 1.0 - TORCH_IQL_EXPECTILE),
        )
        value_loss = _weighted_mean(
            expectile_weights * value_error.pow(2),
            replay_weight_tensor.unsqueeze(1),
        )
        cql_loss = _cql_conservative_loss(
            torch,
            model.action_value(hidden.detach()),
            y,
            mask_tensor,
            replay_weight_tensor,
        )
        with torch.no_grad():
            actor_advantage = selected_action_values - state_values
            actor_weights, actor_weight_stats = _iql_actor_weights(
                torch,
                actor_advantage,
                replay_weight_tensor,
                calibrated_actor_extraction=calibrated_actor_extraction,
            )
            actor_extraction_weights = (
                actor_weights * constraint_actor_weight_tensor.view(-1, 1)
                if constraint_aware_actor_extraction
                else actor_weights
            )
        actor_losses = torch.nn.functional.cross_entropy(
            masked_logits,
            y,
            weight=class_weights,
            reduction="none",
        )
        actor_sample_weights = (
            actor_extraction_weights.squeeze(1) * replay_weight_tensor
        )
        actor_loss = (
            (actor_losses * actor_sample_weights).sum()
            / actor_sample_weights.sum().clamp_min(1e-9)
        )
        behavior_anchor_loss = (
            (actor_losses * replay_weight_tensor).sum()
            / replay_weight_tensor.sum().clamp_min(1e-9)
        )
        behavior_margin_anchor_loss = _behavior_margin_anchor_loss(
            torch,
            masked_logits,
            y,
            mask_tensor,
            replay_weight_tensor,
            behavior_margin_anchor_weight_tensor,
        )
        action_distribution_loss, action_distribution_stats = (
            _action_distribution_actor_loss(
                torch,
                masked_logits,
                mask_tensor,
                y,
                replay_weight_tensor,
                temperature=resolved_action_distribution_temperature,
            )
        )
        guard_feedback_loss = _guard_feedback_loss(
            torch,
            masked_logits,
            y,
            guard_feedback_actions,
            guard_feedback_weights,
        )
        online_update_feedback_loss = _online_update_feedback_loss(
            torch,
            logits,
            mask_tensor,
            online_feedback_actions,
            online_feedback_signals,
        )
        risk_adjusted_actor_loss, risk_adjusted_actor_stats = (
            _risk_adjusted_actor_loss(
                torch,
                masked_logits,
                action_values,
                state_values,
                mask_tensor,
                action_viability_logits,
                y,
                replay_weight_tensor,
            )
        )
        return_calibration_loss = _return_calibration_loss(
            torch,
            selected_action_values,
            state_values,
            return_target,
            replay_weight_tensor,
        )
        suppression_critic_calibration_loss = (
            _suppression_critic_calibration_loss(
                torch,
                action_values,
                state_values,
                y,
                guard_feedback_actions,
                guard_feedback_weights,
            )
        )
        counterfactual_action_value_loss = _weighted_mean(
            torch.nn.functional.mse_loss(
                selected_action_values,
                counterfactual_action_value_target_tensor,
                reduction="none",
            ),
            counterfactual_loss_weight_tensor.unsqueeze(1),
        )
        viability_component_losses = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                viability_logits,
                viability_target_tensor,
                pos_weight=viability_pos_weights,
                reduction="none",
            )
        )
        viability_loss = _masked_component_mean(
            viability_component_losses,
            viability_observed_tensor,
            replay_weight_tensor,
        )
        counterfactual_viability_component_losses = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                viability_logits,
                counterfactual_viability_target_tensor,
                reduction="none",
            )
        )
        counterfactual_viability_loss = _masked_component_mean(
            counterfactual_viability_component_losses,
            counterfactual_viability_observed_tensor,
            counterfactual_loss_weight_tensor,
        )
        action_viability_component_losses = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                action_viability_logits,
                action_viability_target_tensor,
                pos_weight=action_viability_pos_weights.view(1, 1, -1),
                reduction="none",
            )
        )
        action_viability_loss = _masked_action_component_mean(
            action_viability_component_losses,
            action_viability_observed_tensor,
            replay_weight_tensor,
        )
        counterfactual_action_viability_component_losses = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                action_viability_logits,
                counterfactual_action_viability_target_tensor,
                reduction="none",
            )
        )
        counterfactual_action_viability_loss = _masked_action_component_mean(
            counterfactual_action_viability_component_losses,
            counterfactual_action_viability_observed_tensor,
            counterfactual_loss_weight_tensor,
        )
        loss = (
            TORCH_IQL_ACTOR_LOSS_WEIGHT * actor_loss
            + TORCH_IQL_GUARD_FEEDBACK_LOSS_WEIGHT * guard_feedback_loss
            + TORCH_IQL_ONLINE_UPDATE_FEEDBACK_LOSS_WEIGHT
            * online_update_feedback_loss
            + (
                TORCH_IQL_RISK_ADJUSTED_ACTOR_LOSS_WEIGHT
                if risk_adjusted_actor_extraction
                else 0.0
            )
            * risk_adjusted_actor_loss
            + (
                TORCH_IQL_RETURN_CALIBRATION_LOSS_WEIGHT
                if return_calibration
                else 0.0
            )
            * return_calibration_loss
            + (
                TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_LOSS_WEIGHT
                if suppression_critic_calibration
                else 0.0
            )
            * suppression_critic_calibration_loss
            + TORCH_IQL_COUNTERFACTUAL_ACTION_VALUE_LOSS_WEIGHT
            * counterfactual_action_value_loss
            + TORCH_IQL_COUNTERFACTUAL_VIABILITY_LOSS_WEIGHT
            * counterfactual_viability_loss
            + TORCH_IQL_COUNTERFACTUAL_ACTION_VIABILITY_LOSS_WEIGHT
            * counterfactual_action_viability_loss
            + TORCH_IQL_Q_LOSS_WEIGHT * q_loss
            + TORCH_IQL_VALUE_LOSS_WEIGHT * value_loss
            + TORCH_IQL_CQL_LOSS_WEIGHT * cql_loss
            + TORCH_IQL_BEHAVIOR_ANCHOR_LOSS_WEIGHT * behavior_anchor_loss
            + (
                TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_LOSS_WEIGHT
                if behavior_margin_anchor
                else 0.0
            )
            * behavior_margin_anchor_loss
            + (
                resolved_action_distribution_loss_weight
                if action_distribution_regularization
                else 0.0
            )
            * action_distribution_loss
            + TORCH_IQL_VIABILITY_LOSS_WEIGHT * viability_loss
            + TORCH_IQL_ACTION_VIABILITY_LOSS_WEIGHT * action_viability_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        final_actor_loss = float(actor_loss.detach().cpu().item())
        final_q_loss = float(q_loss.detach().cpu().item())
        final_value_loss = float(value_loss.detach().cpu().item())
        final_cql_loss = float(cql_loss.detach().cpu().item())
        final_behavior_anchor_loss = float(
            behavior_anchor_loss.detach().cpu().item()
        )
        final_behavior_margin_anchor_loss = float(
            behavior_margin_anchor_loss.detach().cpu().item()
        )
        final_action_distribution_loss = float(
            action_distribution_loss.detach().cpu().item()
        )
        final_guard_feedback_loss = float(
            guard_feedback_loss.detach().cpu().item()
        )
        final_online_update_feedback_loss = float(
            online_update_feedback_loss.detach().cpu().item()
        )
        final_return_calibration_loss = float(
            return_calibration_loss.detach().cpu().item()
        )
        final_suppression_critic_calibration_loss = float(
            suppression_critic_calibration_loss.detach().cpu().item()
        )
        final_counterfactual_action_value_loss = float(
            counterfactual_action_value_loss.detach().cpu().item()
        )
        final_counterfactual_viability_loss = float(
            counterfactual_viability_loss.detach().cpu().item()
        )
        final_counterfactual_action_viability_loss = float(
            counterfactual_action_viability_loss.detach().cpu().item()
        )
        final_risk_adjusted_actor_loss = float(
            risk_adjusted_actor_loss.detach().cpu().item()
        )
        final_risk_adjusted_actor_stats = risk_adjusted_actor_stats
        final_viability_loss = float(viability_loss.detach().cpu().item())
        final_action_viability_loss = float(
            action_viability_loss.detach().cpu().item()
        )
        final_actor_weight_mean = float(
            actor_extraction_weights.mean().detach().cpu().item()
        )
        final_actor_weight_min = float(
            actor_extraction_weights.min().detach().cpu().item()
        )
        final_actor_weight_max = float(
            actor_extraction_weights.max().detach().cpu().item()
        )
        final_actor_advantage_mean = actor_weight_stats["advantage_mean"]
        final_actor_advantage_scale = actor_weight_stats["advantage_scale"]

    if uses_actor_finetune:
        action_risk_bias_tensor = None
        action_support_tensor = None
        action_component_support_tensor = None
        advantage_mean_tensor = None
        advantage_scale_tensor = None
        if uses_calibrated_supported_extraction:
            calibrated_supported_report = _fit_calibrated_supported_actor_report(
                torch,
                model,
                tuple(calibration_records or ()),
                action_index=action_index,
            )
            if calibration_validation_records:
                calibrated_supported_validation_report = (
                    _evaluate_calibrated_supported_actor_report(
                        torch,
                        model,
                        tuple(calibration_validation_records),
                        action_index=action_index,
                        fit_report=calibrated_supported_report,
                    )
                )
            action_risk_bias_tensor = torch.tensor(
                calibrated_supported_report["action_component_bias_matrix"],
                dtype=torch.float32,
                device=device,
            )
            action_support_tensor = torch.tensor(
                [
                    float(
                        calibrated_supported_report[
                            "action_support_counts"
                        ][action]
                    )
                    for action in ACTION_NAMES
                ],
                dtype=torch.float32,
                device=device,
            )
            action_component_support_tensor = torch.tensor(
                calibrated_supported_report["action_component_support_matrix"],
                dtype=torch.float32,
                device=device,
            )
            advantage_mean_tensor = torch.tensor(
                float(calibrated_supported_report["advantage_mean"]),
                dtype=torch.float32,
                device=device,
            )
            advantage_scale_tensor = torch.tensor(
                float(calibrated_supported_report["advantage_scale"]),
                dtype=torch.float32,
                device=device,
            )
        context_action_support_tensor = None
        context_action_probability_tensor = None
        action_family_tensor = None
        row_context_labels: tuple[str, ...] = ()
        contextual_actor_active_rows = None
        contextual_actor_target_actions = None
        if (
            contextual_behavior_supported_actor_extraction
            or uses_contextual_prior_regularization
        ):
            contextual_supported_context = _contextual_supported_context(
                records,
                tuple(calibration_records or ()),
                action_index=action_index,
            )
            contextual_supported_report = contextual_supported_context["report"]
            context_action_support_tensor = torch.tensor(
                contextual_supported_context["row_action_support_matrix"],
                dtype=torch.float32,
                device=device,
            )
            context_action_probability_tensor = torch.tensor(
                contextual_supported_context["row_action_probability_matrix"],
                dtype=torch.float32,
                device=device,
            )
            action_family_tensor = torch.tensor(
                [_action_family_index(action) for action in ACTION_NAMES],
                dtype=torch.long,
                device=device,
            )
            row_context_labels = tuple(
                str(label)
                for label in contextual_supported_context["row_context_labels"]
            )
        if contextual_behavior_supported_actor_extraction:
            if (
                action_risk_bias_tensor is None
                or advantage_mean_tensor is None
                or advantage_scale_tensor is None
            ):
                raise ValueError(
                    "contextual behavior supported actor extraction requires "
                    "calibrated supported actor tensors"
                )
            with torch.no_grad():
                target_hidden = torch.tanh(model.hidden(x))
                action_values = model.action_value(target_hidden)
                state_values = model.state_value(target_hidden)
                action_viability_logits = model.action_viability(
                    target_hidden,
                ).view(
                    -1,
                    len(ACTION_NAMES),
                    len(VIABILITY_COMPONENT_NAMES),
                )
                (
                    contextual_actor_active_rows,
                    contextual_actor_target_actions,
                    calibrated_supported_stats,
                ) = _contextual_behavior_supported_actor_targets(
                    torch,
                    action_values,
                    state_values,
                    mask_tensor,
                    action_viability_logits,
                    y,
                    action_risk_bias_tensor,
                    context_action_support_tensor,
                    context_action_probability_tensor,
                    action_family_tensor,
                    advantage_mean_tensor,
                    advantage_scale_tensor,
                    row_context_labels,
                )
        for _ in range(TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EPOCHS):
            optimizer.zero_grad()
            hidden = torch.tanh(model.hidden(x)).detach()
            logits = model.actor(hidden)
            masked_logits = _masked_logits(torch, logits, mask_tensor)
            if contextual_behavior_supported_actor_extraction:
                if (
                    context_action_support_tensor is None
                    or context_action_probability_tensor is None
                    or action_family_tensor is None
                    or action_risk_bias_tensor is None
                    or advantage_mean_tensor is None
                    or advantage_scale_tensor is None
                ):
                    raise ValueError(
                        "contextual behavior supported actor extraction "
                        "requires contextual support tensors"
                    )
                if (
                    contextual_actor_active_rows is None
                    or contextual_actor_target_actions is None
                ):
                    raise ValueError(
                        "contextual behavior supported actor extraction "
                        "requires precomputed actor targets"
                    )
                if int(contextual_actor_active_rows.numel()) <= 0:
                    calibrated_supported_actor_loss = masked_logits.new_zeros(())
                else:
                    losses = torch.nn.functional.cross_entropy(
                        masked_logits[contextual_actor_active_rows],
                        contextual_actor_target_actions[
                            contextual_actor_active_rows
                        ],
                        reduction="none",
                    )
                    active_weights = replay_weight_tensor[
                        contextual_actor_active_rows
                    ]
                    calibrated_supported_actor_loss = (
                        (losses * active_weights).sum()
                        / active_weights.sum().clamp_min(1e-9)
                    )
            elif calibrated_supported_actor_extraction:
                if (
                    action_risk_bias_tensor is None
                    or action_support_tensor is None
                    or action_component_support_tensor is None
                    or advantage_mean_tensor is None
                    or advantage_scale_tensor is None
                ):
                    raise ValueError(
                        "calibrated supported actor extraction requires "
                        "calibrated support tensors"
                    )
                with torch.no_grad():
                    target_hidden = torch.tanh(model.hidden(x))
                    action_values = model.action_value(target_hidden)
                    state_values = model.state_value(target_hidden)
                    action_viability_logits = model.action_viability(
                        target_hidden,
                    ).view(
                        -1,
                        len(ACTION_NAMES),
                        len(VIABILITY_COMPONENT_NAMES),
                    )
                calibrated_supported_actor_loss, calibrated_supported_stats = (
                    _calibrated_supported_actor_loss(
                        torch,
                        masked_logits,
                        action_values,
                        state_values,
                        mask_tensor,
                        action_viability_logits,
                        replay_weight_tensor,
                        action_risk_bias_tensor,
                        action_support_tensor,
                        action_component_support_tensor,
                        advantage_mean_tensor,
                        advantage_scale_tensor,
                    )
                )
            else:
                calibrated_supported_actor_loss = masked_logits.new_zeros(())
            actor_losses = torch.nn.functional.cross_entropy(
                masked_logits,
                y,
                weight=class_weights,
                reduction="none",
            )
            actor_loss = (
                (actor_losses * replay_weight_tensor).sum()
                / replay_weight_tensor.sum().clamp_min(1e-9)
            )
            if uses_contextual_prior_regularization:
                if context_action_probability_tensor is None:
                    raise ValueError(
                        "contextual behavior prior regularization requires "
                        "contextual behavior probabilities"
                    )
                (
                    contextual_behavior_prior_loss,
                    contextual_behavior_prior_stats,
                ) = _contextual_behavior_prior_actor_loss(
                    torch,
                    masked_logits,
                    mask_tensor,
                    context_action_probability_tensor,
                    replay_weight_tensor,
                    logged_actions=y,
                )
            else:
                contextual_behavior_prior_loss = masked_logits.new_zeros(())
            action_distribution_loss, action_distribution_stats = (
                _action_distribution_actor_loss(
                    torch,
                    masked_logits,
                    mask_tensor,
                    y,
                    replay_weight_tensor,
                    temperature=resolved_action_distribution_temperature,
                )
            )
            actor_finetune_guard_feedback_loss = _guard_feedback_loss(
                torch,
                masked_logits,
                y,
                guard_feedback_actions,
                guard_feedback_weights,
            )
            actor_finetune_behavior_margin_anchor_loss = (
                _behavior_margin_anchor_loss(
                    torch,
                    masked_logits,
                    y,
                    mask_tensor,
                    replay_weight_tensor,
                    behavior_margin_anchor_weight_tensor,
                )
                if behavior_margin_anchor
                else masked_logits.new_zeros(())
            )
            loss = (
                0.35 * actor_loss
                + TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_LOSS_WEIGHT
                * calibrated_supported_actor_loss
                + resolved_contextual_behavior_prior_loss_weight
                * contextual_behavior_prior_loss
                + TORCH_IQL_GUARD_FEEDBACK_LOSS_WEIGHT
                * actor_finetune_guard_feedback_loss
                + (
                    TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_FINETUNE_LOSS_WEIGHT
                    if behavior_margin_anchor
                    else 0.0
                )
                * actor_finetune_behavior_margin_anchor_loss
                + (
                    resolved_action_distribution_loss_weight
                    if action_distribution_regularization
                    else 0.0
                )
                * action_distribution_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=1.0)
            optimizer.step()
            final_actor_loss = float(actor_loss.detach().cpu().item())
            final_calibrated_supported_actor_loss = float(
                calibrated_supported_actor_loss.detach().cpu().item()
            )
            final_contextual_behavior_prior_loss = float(
                contextual_behavior_prior_loss.detach().cpu().item()
            )
            final_actor_finetune_guard_feedback_loss = float(
                actor_finetune_guard_feedback_loss.detach().cpu().item()
            )
            final_actor_finetune_behavior_margin_anchor_loss = float(
                actor_finetune_behavior_margin_anchor_loss.detach().cpu().item()
            )
            final_action_distribution_loss = float(
                action_distribution_loss.detach().cpu().item()
            )

    if rollout_state_action_calibration:
        rollout_state_action_calibration_report = (
            _apply_rollout_state_action_bias_calibration(
                torch,
                model,
                tuple(calibration_records or ()),
                action_index=action_index,
                device=device,
                max_action_share=resolved_rollout_state_action_max_share,
                bias_step=resolved_rollout_state_action_bias_step,
                max_bias_delta=resolved_rollout_state_action_max_bias_delta,
            )
        )
        if calibration_validation_records:
            rollout_state_action_calibration_validation_report = (
                _rollout_state_action_top1_report(
                    torch,
                    model,
                    tuple(calibration_validation_records),
                    action_index=action_index,
                    device=device,
                )
            )

    training_metrics = {
        "epochs": TORCH_NEURAL_EPOCHS,
        "final_loss": _round(final_loss),
        "final_actor_loss": _round(final_actor_loss),
        "final_q_loss": _round(final_q_loss),
        "final_value_loss": _round(final_value_loss),
        "final_cql_loss": _round(final_cql_loss),
        "final_behavior_anchor_loss": _round(final_behavior_anchor_loss),
        "final_behavior_margin_anchor_loss": _round(
            final_behavior_margin_anchor_loss
        ),
        "final_guard_feedback_loss": _round(final_guard_feedback_loss),
        "final_online_update_feedback_loss": _round(
            final_online_update_feedback_loss
        ),
        "final_return_calibration_loss": _round(final_return_calibration_loss),
        "final_suppression_critic_calibration_loss": _round(
            final_suppression_critic_calibration_loss
        ),
        "final_counterfactual_action_value_loss": _round(
            final_counterfactual_action_value_loss
        ),
        "final_counterfactual_viability_loss": _round(
            final_counterfactual_viability_loss
        ),
        "final_counterfactual_action_viability_loss": _round(
            final_counterfactual_action_viability_loss
        ),
        "final_action_distribution_loss": _round(
            final_action_distribution_loss
        ),
        "final_risk_adjusted_actor_loss": _round(
            final_risk_adjusted_actor_loss
        ),
        "final_calibrated_supported_actor_loss": _round(
            final_calibrated_supported_actor_loss
        ),
        "final_contextual_behavior_prior_loss": _round(
            final_contextual_behavior_prior_loss
        ),
        "final_actor_finetune_guard_feedback_loss": _round(
            final_actor_finetune_guard_feedback_loss
        ),
        "final_actor_finetune_behavior_margin_anchor_loss": _round(
            final_actor_finetune_behavior_margin_anchor_loss
        ),
        "final_viability_loss": _round(final_viability_loss),
        "final_action_viability_loss": _round(final_action_viability_loss),
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
        "critic_return_calibration_policy": TORCH_IQL_RETURN_CALIBRATION_POLICY,
        "critic_return_calibration_enabled": return_calibration,
        "critic_return_calibration_loss_weight": (
            TORCH_IQL_RETURN_CALIBRATION_LOSS_WEIGHT
            if return_calibration
            else 0.0
        ),
        "critic_return_calibration_discount": TORCH_IQL_DISCOUNT,
        "critic_return_target_mean": _round(
            sum(returns) / float(len(returns)) if returns else 0.0
        ),
        "critic_return_target_min": _round(min(returns) if returns else 0.0),
        "critic_return_target_max": _round(max(returns) if returns else 0.0),
        "critic_suppression_calibration_policy": (
            TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_POLICY
        ),
        "critic_suppression_calibration_enabled": (
            suppression_critic_calibration
        ),
        "critic_suppression_calibration_loss_weight": (
            TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_LOSS_WEIGHT
            if suppression_critic_calibration
            else 0.0
        ),
        "critic_suppression_calibration_margin": (
            TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_MARGIN
        ),
        "critic_suppression_calibration_state_advantage_weight": (
            TORCH_IQL_SUPPRESSION_CRITIC_STATE_ADVANTAGE_WEIGHT
        ),
        "critic_suppression_calibration_count": int(
            sum(1 for _, weight in guard_feedback if weight > 0.0)
        ),
        "critic_suppression_calibration_rate": _round(
            (
                sum(1 for _, weight in guard_feedback if weight > 0.0)
                / float(len(guard_feedback))
            )
            if guard_feedback
            else 0.0
        ),
        "counterfactual_label_supervision": (
            counterfactual_supervision["diagnostics"]
        ),
        "counterfactual_label_supervision_enabled": (
            counterfactual_label_report is not None
        ),
        "counterfactual_label_supervision_policy": (
            TORCH_IQL_COUNTERFACTUAL_LABEL_SUPERVISION_POLICY
        ),
        "counterfactual_label_weight_policy": (
            TORCH_IQL_COUNTERFACTUAL_LABEL_WEIGHT_POLICY
        ),
        "counterfactual_label_weight_scale": _round(
            counterfactual_label_weight_scale
        ),
        "counterfactual_action_value_loss_weight": (
            TORCH_IQL_COUNTERFACTUAL_ACTION_VALUE_LOSS_WEIGHT
        ),
        "counterfactual_viability_loss_weight": (
            TORCH_IQL_COUNTERFACTUAL_VIABILITY_LOSS_WEIGHT
        ),
        "counterfactual_action_viability_loss_weight": (
            TORCH_IQL_COUNTERFACTUAL_ACTION_VIABILITY_LOSS_WEIGHT
        ),
        "critic_regularization_policy": TORCH_IQL_CQL_REGULARIZATION_POLICY,
        "critic_regularization_enabled": TORCH_IQL_CQL_LOSS_WEIGHT > 0.0,
        "cql_temperature": TORCH_IQL_CQL_TEMPERATURE,
        "cql_loss_weight": TORCH_IQL_CQL_LOSS_WEIGHT,
        "actor_weighting_policy": _actor_weighting_policy(
            calibrated_actor_extraction=calibrated_actor_extraction,
            constraint_aware_actor_extraction=(
                constraint_aware_actor_extraction
            ),
            risk_adjusted_actor_extraction=risk_adjusted_actor_extraction,
            calibrated_supported_actor_extraction=(
                calibrated_supported_actor_extraction
            ),
            contextual_behavior_supported_actor_extraction=(
                contextual_behavior_supported_actor_extraction
            ),
            contextual_behavior_prior_regularization=(
                uses_contextual_prior_regularization
            ),
            action_distribution_regularization=(
                action_distribution_regularization
            ),
            rollout_state_action_calibration=(
                rollout_state_action_calibration
            ),
        ),
        "actor_advantage_calibration_enabled": calibrated_actor_extraction,
        "actor_advantage_calibration_policy": (
            TORCH_IQL_CALIBRATED_ACTOR_WEIGHTING_POLICY
            if calibrated_actor_extraction
            else None
        ),
        "actor_advantage_calibration_temperature": (
            TORCH_IQL_CALIBRATED_ADVANTAGE_TEMPERATURE
            if calibrated_actor_extraction
            else None
        ),
        "actor_advantage_calibration_scale_epsilon": (
            TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON
            if calibrated_actor_extraction
            else None
        ),
        "actor_advantage_calibration_weight_blend": (
            TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_BLEND
            if calibrated_actor_extraction
            else None
        ),
        "actor_advantage_calibration_weight_min": (
            TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_MIN
            if calibrated_actor_extraction
            else None
        ),
        "actor_sample_weight_mean": _round(final_actor_weight_mean),
        "actor_sample_weight_min_observed": _round(final_actor_weight_min),
        "actor_sample_weight_max_observed": _round(final_actor_weight_max),
        "actor_advantage_mean": _round(final_actor_advantage_mean),
        "actor_advantage_scale": _round(final_actor_advantage_scale),
        "actor_constraint_awareness_enabled": constraint_aware_actor_extraction,
        "actor_constraint_awareness_policy": (
            TORCH_IQL_CONSTRAINT_AWARE_ACTOR_WEIGHTING_POLICY
        ),
        "actor_constraint_component_policy": (
            TORCH_IQL_CONSTRAINT_AWARE_ACTOR_COMPONENT_POLICY
        ),
        "actor_constraint_weight_min": (
            TORCH_IQL_CONSTRAINT_AWARE_ACTOR_WEIGHT_MIN
            if constraint_aware_actor_extraction
            else 1.0
        ),
        "actor_constraint_weight_mean": _round(
            constraint_actor_stats["weight_mean"]
        ),
        "actor_constraint_weight_min_observed": _round(
            constraint_actor_stats["weight_min"]
        ),
        "actor_constraint_weight_max_observed": _round(
            constraint_actor_stats["weight_max"]
        ),
        "actor_constraint_risky_logged_action_count": int(
            constraint_actor_stats["risky_count"]
        ),
        "actor_constraint_risky_logged_action_rate": _round(
            constraint_actor_stats["risky_rate"]
        ),
        "actor_constraint_risky_logged_action_component_counts": (
            constraint_actor_stats["component_counts"]
        ),
        "actor_risk_adjusted_extraction_enabled": (
            risk_adjusted_actor_extraction
        ),
        "actor_risk_adjusted_extraction_policy": (
            TORCH_IQL_RISK_ADJUSTED_ACTOR_EXTRACTION_POLICY
        ),
        "actor_risk_score_policy": TORCH_IQL_ACTION_RISK_SCORE_POLICY,
        "actor_risk_adjusted_loss_weight": (
            TORCH_IQL_RISK_ADJUSTED_ACTOR_LOSS_WEIGHT
            if risk_adjusted_actor_extraction
            else 0.0
        ),
        "actor_risk_adjusted_risk_penalty": (
            TORCH_IQL_RISK_ADJUSTED_ACTOR_RISK_PENALTY
        ),
        "actor_risk_adjusted_temperature": (
            TORCH_IQL_RISK_ADJUSTED_ACTOR_TEMPERATURE
        ),
        "actor_risk_adjusted_target_entropy_mean": _round(
            final_risk_adjusted_actor_stats["target_entropy_mean"]
        ),
        "actor_risk_adjusted_target_logged_probability_mean": _round(
            final_risk_adjusted_actor_stats["target_logged_probability_mean"]
        ),
        "actor_risk_adjusted_target_logged_top1_rate": _round(
            final_risk_adjusted_actor_stats["target_logged_top1_rate"]
        ),
        "actor_risk_adjusted_target_top_risk_mean": _round(
            final_risk_adjusted_actor_stats["target_top_risk_mean"]
        ),
        "actor_risk_adjusted_target_top_advantage_mean": _round(
            final_risk_adjusted_actor_stats["target_top_advantage_mean"]
        ),
        "actor_calibrated_supported_extraction_enabled": (
            uses_calibrated_supported_extraction
        ),
        "actor_calibrated_supported_extraction_policy": (
            TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EXTRACTION_POLICY
        ),
        "actor_contextual_behavior_supported_extraction_enabled": (
            contextual_behavior_supported_actor_extraction
        ),
        "actor_contextual_behavior_supported_extraction_policy": (
            TORCH_IQL_CONTEXTUAL_BEHAVIOR_SUPPORTED_ACTOR_EXTRACTION_POLICY
        ),
        "actor_contextual_support_policy": TORCH_IQL_CONTEXTUAL_SUPPORT_POLICY,
        "actor_behavior_proximity_policy": TORCH_IQL_BEHAVIOR_PROXIMITY_POLICY,
        "actor_contextual_behavior_prior_regularization_enabled": (
            uses_contextual_prior_regularization
        ),
        "actor_contextual_behavior_prior_regularization_policy": (
            TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_REGULARIZATION_POLICY
        ),
        "actor_contextual_behavior_prior_regularization_loss_weight": (
            resolved_contextual_behavior_prior_loss_weight
            if uses_contextual_prior_regularization
            else 0.0
        ),
        "actor_contextual_behavior_prior_regularization_min_mass": (
            TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_MIN_MASS
        ),
        "actor_contextual_behavior_prior_regularization_finetune_epochs": (
            TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EPOCHS
            if uses_contextual_prior_regularization
            else 0
        ),
        "actor_contextual_behavior_prior": contextual_behavior_prior_stats,
        "actor_action_distribution_regularization_enabled": (
            action_distribution_regularization
        ),
        "actor_action_distribution_regularization_policy": (
            TORCH_IQL_ACTION_DISTRIBUTION_REGULARIZATION_POLICY
        ),
        "actor_action_distribution_loss_weight": (
            resolved_action_distribution_loss_weight
            if action_distribution_regularization
            else 0.0
        ),
        "actor_action_distribution": action_distribution_stats,
        "actor_rollout_state_action_calibration_enabled": (
            rollout_state_action_calibration
        ),
        "actor_rollout_state_action_calibration_policy": (
            TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY
        ),
        "actor_rollout_state_action_calibration_max_share": (
            resolved_rollout_state_action_max_share
        ),
        "actor_rollout_state_action_calibration_bias_step": (
            resolved_rollout_state_action_bias_step
        ),
        "actor_rollout_state_action_calibration_max_bias_delta": (
            resolved_rollout_state_action_max_bias_delta
        ),
        "actor_rollout_state_action_calibration_max_iterations": (
            TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_ITERATIONS
        ),
        "actor_rollout_state_action_calibration": (
            rollout_state_action_calibration_report
        ),
        "actor_rollout_state_action_calibration_validation": (
            rollout_state_action_calibration_validation_report
        ),
        "actor_finetune_guard_feedback_enabled": uses_actor_finetune,
        "actor_finetune_guard_feedback_policy": (
            TORCH_IQL_GUARD_FEEDBACK_FINETUNE_POLICY
        ),
        "actor_finetune_guard_feedback_loss_weight": (
            TORCH_IQL_GUARD_FEEDBACK_LOSS_WEIGHT if uses_actor_finetune else 0.0
        ),
        "actor_finetune_guard_feedback_margin": TORCH_IQL_GUARD_FEEDBACK_MARGIN,
        "actor_finetune_guard_feedback_count": (
            int(sum(1 for _, weight in guard_feedback if weight > 0.0))
            if uses_actor_finetune
            else 0
        ),
        "actor_finetune_guard_feedback_rate": (
            _round(
                (
                    sum(1 for _, weight in guard_feedback if weight > 0.0)
                    / float(len(guard_feedback))
                )
                if guard_feedback and uses_actor_finetune
                else 0.0
            )
        ),
        "actor_finetune_behavior_margin_anchor_enabled": (
            uses_actor_finetune and behavior_margin_anchor
        ),
        "actor_finetune_behavior_margin_anchor_policy": (
            TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_FINETUNE_POLICY
        ),
        "actor_finetune_behavior_margin_anchor_loss_weight": (
            TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_FINETUNE_LOSS_WEIGHT
            if uses_actor_finetune and behavior_margin_anchor
            else 0.0
        ),
        "actor_finetune_behavior_margin_anchor_target": (
            TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_TARGET
        ),
        "actor_finetune_behavior_margin_anchor_eligible_count": (
            int(
                sum(
                    1
                    for weight in behavior_margin_anchor_weights
                    if weight > 0.0
                )
            )
            if uses_actor_finetune and behavior_margin_anchor
            else 0
        ),
        "actor_finetune_behavior_margin_anchor_eligible_rate": (
            _round(
                (
                    sum(
                        1
                        for weight in behavior_margin_anchor_weights
                        if weight > 0.0
                    )
                    / float(len(behavior_margin_anchor_weights))
                )
                if behavior_margin_anchor_weights
                and uses_actor_finetune
                and behavior_margin_anchor
                else 0.0
            )
        ),
        "actor_calibrated_supported_risk_calibration_policy": (
            TORCH_IQL_CALIBRATED_SUPPORTED_RISK_CALIBRATION_POLICY
        ),
        "actor_calibrated_supported_advantage_policy": (
            TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_POLICY
        ),
        "actor_calibrated_supported_loss_weight": (
            TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_LOSS_WEIGHT
            if uses_calibrated_supported_extraction
            else 0.0
        ),
        "actor_calibrated_supported_finetune_epochs": (
            TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EPOCHS
            if uses_calibrated_supported_extraction
            else 0
        ),
        "actor_calibrated_supported_min_action_support": (
            TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT
        ),
        "actor_contextual_supported_min_action_support": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT
        ),
        "actor_contextual_supported_min_behavior_probability": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_BEHAVIOR_PROBABILITY
        ),
        "actor_contextual_supported_family_conversion_ratio": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_FAMILY_CONVERSION_RATIO
        ),
        "actor_contextual_supported_resource_conversion_min_probability": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_RESOURCE_CONVERSION_MIN_PROBABILITY
        ),
        "actor_contextual_supported_max_target_action_expansion_ratio": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_TARGET_ACTION_EXPANSION_RATIO
        ),
        "actor_contextual_supported_max_family_conversion_rate": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_FAMILY_CONVERSION_RATE
        ),
        "actor_contextual_supported_max_movement_stay_resource_rate": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_MOVEMENT_STAY_RESOURCE_RATE
        ),
        "actor_calibrated_supported_risk_threshold": (
            TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD
        ),
        "actor_calibrated_supported_advantage_threshold": (
            TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD
        ),
        "actor_calibrated_supported_risk_penalty": (
            TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY
        ),
        "actor_calibrated_supported_calibration": calibrated_supported_report,
        "actor_calibrated_supported_calibration_validation": (
            calibrated_supported_validation_report
        ),
        "actor_contextual_supported_support": contextual_supported_report,
        "actor_calibrated_supported_extraction": calibrated_supported_stats,
        "actor_calibrated_supported_selected_count": int(
            calibrated_supported_stats["selected_count"]
        ),
        "actor_calibrated_supported_selected_rate": _round(
            float(calibrated_supported_stats["selected_rate"])
        ),
        "actor_calibrated_supported_rejected_count": int(
            calibrated_supported_stats["rejected_count"]
        ),
        "actor_calibrated_supported_rejection_reasons": (
            calibrated_supported_stats["rejection_reasons"]
        ),
        "actor_calibrated_supported_low_support_count": int(
            calibrated_supported_stats["low_support_count"]
        ),
        "actor_calibrated_supported_high_risk_count": int(
            calibrated_supported_stats["high_risk_count"]
        ),
        "actor_calibrated_supported_negative_advantage_count": int(
            calibrated_supported_stats["negative_advantage_count"]
        ),
        "actor_calibrated_supported_behavior_proximity_count": int(
            calibrated_supported_stats.get("behavior_proximity_count", 0)
        ),
        "actor_calibrated_supported_behavior_proximity_cap_counts": (
            calibrated_supported_stats.get("behavior_proximity_cap_counts", {})
        ),
        "actor_calibrated_supported_no_legal_candidate_count": int(
            calibrated_supported_stats["no_legal_candidate_count"]
        ),
        "behavior_anchor_policy": TORCH_IQL_BEHAVIOR_ANCHOR_POLICY,
        "behavior_anchor_enabled": (
            TORCH_IQL_BEHAVIOR_ANCHOR_LOSS_WEIGHT > 0.0
        ),
        "behavior_anchor_loss_weight": TORCH_IQL_BEHAVIOR_ANCHOR_LOSS_WEIGHT,
        "behavior_margin_anchor_policy": TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_POLICY,
        "behavior_margin_anchor_enabled": behavior_margin_anchor,
        "behavior_margin_anchor_target": TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_TARGET,
        "behavior_margin_anchor_loss_weight": (
            TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_LOSS_WEIGHT
            if behavior_margin_anchor
            else 0.0
        ),
        "behavior_margin_anchor_eligible_count": int(
            sum(1 for weight in behavior_margin_anchor_weights if weight > 0.0)
        ),
        "behavior_margin_anchor_eligible_rate": _round(
            (
                sum(1 for weight in behavior_margin_anchor_weights if weight > 0.0)
                / float(len(behavior_margin_anchor_weights))
            )
            if behavior_margin_anchor_weights
            else 0.0
        ),
        "guard_feedback_policy": TORCH_IQL_GUARD_FEEDBACK_POLICY,
        "guard_feedback_count": int(
            sum(1 for _, weight in guard_feedback if weight > 0.0)
        ),
        "guard_feedback_rate": _round(
            (
                sum(1 for _, weight in guard_feedback if weight > 0.0)
                / float(len(guard_feedback))
            )
            if guard_feedback
            else 0.0
        ),
        "guard_feedback_margin": TORCH_IQL_GUARD_FEEDBACK_MARGIN,
        "guard_feedback_loss_weight": TORCH_IQL_GUARD_FEEDBACK_LOSS_WEIGHT,
        "hard_guard_feedback_weight": TORCH_IQL_HARD_GUARD_FEEDBACK_WEIGHT,
        "heuristic_delegate_feedback_weight": (
            TORCH_IQL_HEURISTIC_DELEGATE_FEEDBACK_WEIGHT
        ),
        "learned_replay_weight_policy": TORCH_IQL_LEARNED_REPLAY_WEIGHT_POLICY,
        "learned_replay_self_action_weight": (
            TORCH_IQL_LEARNED_REPLAY_SELF_ACTION_WEIGHT
        ),
        "learned_replay_guarded_action_weight": (
            TORCH_IQL_LEARNED_REPLAY_GUARDED_ACTION_WEIGHT
        ),
        "online_update_feedback_policy": TORCH_IQL_ONLINE_UPDATE_FEEDBACK_POLICY,
        "online_update_feedback_count": int(
            sum(1 for _, signal in online_feedback if signal != 0.0)
        ),
        "online_update_feedback_positive_count": int(
            sum(1 for _, signal in online_feedback if signal > 0.0)
        ),
        "online_update_feedback_negative_count": int(
            sum(1 for _, signal in online_feedback if signal < 0.0)
        ),
        "online_update_feedback_margin": TORCH_IQL_ONLINE_UPDATE_FEEDBACK_MARGIN,
        "online_update_feedback_loss_weight": (
            TORCH_IQL_ONLINE_UPDATE_FEEDBACK_LOSS_WEIGHT
        ),
        "learned_replay_record_count": int(
            sum(
                1
                for transition in transitions
                if _is_learned_policy_action_source(transition.action_source)
            )
        ),
        "learned_replay_self_action_count": int(
            sum(
                1
                for transition in transitions
                if _is_learned_policy_action_source(transition.action_source)
                and _guard_feedback_record(transition, action_index=action_index)[1]
                <= 0.0
            )
        ),
        "replay_sample_weight_mean": _round(
            sum(replay_weights) / float(len(replay_weights))
            if replay_weights
            else 0.0
        ),
        "replay_sample_weight_min_observed": _round(
            min(replay_weights) if replay_weights else 0.0
        ),
        "iql_discount": TORCH_IQL_DISCOUNT,
        "iql_expectile": TORCH_IQL_EXPECTILE,
        "iql_advantage_temperature": TORCH_IQL_ADVANTAGE_TEMPERATURE,
        "iql_advantage_weight_max": TORCH_IQL_ADVANTAGE_WEIGHT_MAX,
        "q_loss_weight": TORCH_IQL_Q_LOSS_WEIGHT,
        "value_loss_weight": TORCH_IQL_VALUE_LOSS_WEIGHT,
        "actor_loss_weight": TORCH_IQL_ACTOR_LOSS_WEIGHT,
        "viability_head_policy": VIABILITY_HEAD_POLICY,
        "viability_head_enabled": True,
        "viability_representation_policy": viability_representation_policy,
        "viability_component_names": list(VIABILITY_COMPONENT_NAMES),
        "viability_loss_weight": TORCH_IQL_VIABILITY_LOSS_WEIGHT,
        "viability_pos_weight_policy": TORCH_IQL_VIABILITY_POS_WEIGHT_POLICY,
        "viability_pos_weights": _component_scalar_map(
            viability_pos_weights,
        ),
        "viability_component_positive_counts": viability_positive_counts,
        "viability_component_observed_counts": viability_observed_counts,
        "viability_suppression_supervision_policy": (
            TORCH_IQL_STATE_SUPPRESSION_SUPERVISION_POLICY
        ),
        "viability_suppression_observed_count": (
            viability_observed_counts[VIABILITY_SUPPRESSION_COMPONENT]
        ),
        "action_viability_head_policy": VIABILITY_ACTION_HEAD_POLICY,
        "action_viability_head_enabled": True,
        "action_viability_supervision_policy": (
            VIABILITY_ACTION_SUPERVISION_POLICY
        ),
        "action_viability_loss_weight": TORCH_IQL_ACTION_VIABILITY_LOSS_WEIGHT,
        "action_viability_pos_weight_policy": (
            TORCH_IQL_ACTION_VIABILITY_POS_WEIGHT_POLICY
        ),
        "action_viability_pos_weights": _component_scalar_map(
            action_viability_pos_weights,
        ),
        "action_viability_observed_component_count": int(
            sum(
                sum(sum(1 for observed in components if observed) for components in row)
                for row in action_viability_observed
            )
        ),
        "action_viability_observed_component_rate": _round(
            (
                sum(
                    sum(
                        sum(1 for observed in components if observed)
                        for components in row
                    )
                    for row in action_viability_observed
                )
                / float(
                    len(action_viability_observed)
                    * len(ACTION_NAMES)
                    * len(VIABILITY_COMPONENT_NAMES)
                )
            )
            if action_viability_observed
            else 0.0
        ),
        "action_viability_component_positive_counts": (
            action_viability_positive_counts
        ),
        "action_viability_component_observed_counts": (
            action_viability_observed_counts
        ),
        "action_viability_suppression_positive_count": (
            action_viability_positive_counts[VIABILITY_SUPPRESSION_COMPONENT]
        ),
        "action_viability_suppression_observed_count": (
            action_viability_observed_counts[VIABILITY_SUPPRESSION_COMPONENT]
        ),
        "action_viability_logged_suppression_positive_count": (
            _logged_action_component_positive_count(
                action_viability_targets,
                labels,
                component=VIABILITY_SUPPRESSION_COMPONENT,
            )
        ),
        "viability_pos_weight_max": TORCH_IQL_VIABILITY_POS_WEIGHT_MAX,
        "viability_target_component_positive_counts": _viability_component_counts(
            component_targets
        ),
        "viability_target_component_positive_rates": _viability_component_rates(
            component_targets
        ),
        "viability_component_positive_rates": _observed_component_rates(
            viability_positive_counts,
            viability_observed_counts,
        ),
        "viability_survival_horizon_observed_mean": _round(
            sum(survival_horizons) / float(len(survival_horizons))
            if survival_horizons
            else 0.0
        ),
    }
    training_metrics.update(
        _iql_training_diagnostics(
            torch,
            model,
            x,
            next_x,
            y,
            reward_target,
            return_target,
            done_tensor,
            mask_tensor,
        )
    )
    training_metrics["torch_device_metadata"] = torch_device_metadata
    return _serialize_model(
        model,
        training_metrics=training_metrics,
        include_viability_head=True,
        torch_device_metadata=torch_device_metadata,
    )


def _train_torch_actor_critic_network(
    records: tuple[dict[str, object], ...],
    *,
    advantage_weighted: bool,
    torch_device: str,
) -> dict[str, object]:
    torch = _load_torch()
    torch_device_metadata = _resolve_torch_device_metadata(
        torch,
        requested_device=torch_device,
    )
    device = torch.device(torch_device_metadata["resolved_device"])
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

    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    reward_target = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    class_weights = _balanced_class_weights(torch, labels, device=device)
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
        advantage_weights = torch.tensor(
            raw_advantage_weights,
            dtype=torch.float32,
            device=device,
        )
    model = _TorchActorCritic(torch).to(device)
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
    training_metrics["torch_device_metadata"] = torch_device_metadata

    return _serialize_model(
        model,
        training_metrics=training_metrics,
        torch_device_metadata=torch_device_metadata,
    )


def _counterfactual_label_supervision(
    records: tuple[dict[str, object], ...],
    transitions: tuple[Any, ...],
    label_report: Mapping[str, object] | None,
    *,
    action_index: Mapping[str, int],
    weight_scale: float,
) -> dict[str, object]:
    row_count = len(records)
    _validate_counterfactual_weight_scale(weight_scale)
    empty = _empty_counterfactual_label_supervision(
        row_count,
        weight_scale=weight_scale,
    )
    if label_report is None:
        return empty
    if (
        label_report.get("schema_version")
        != MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "counterfactual label report has unsupported schema_version"
        )
    labels_payload = label_report.get("labels")
    if not isinstance(labels_payload, list):
        raise ValueError("counterfactual label report labels must be a list")

    label_lookup: dict[tuple[object, ...], Mapping[str, object]] = {}
    duplicate_label_count = 0
    usable_label_count = 0
    for payload in labels_payload:
        if not isinstance(payload, Mapping):
            continue
        keys = _counterfactual_label_lookup_keys(payload)
        if not keys:
            continue
        usable_label_count += 1
        for key in keys:
            if key in label_lookup:
                duplicate_label_count += 1
                continue
            label_lookup[key] = payload

    sample_weights = list(empty["sample_weights"])  # type: ignore[arg-type]
    loss_weights = list(empty["loss_weights"])  # type: ignore[arg-type]
    action_value_targets = list(
        empty["action_value_targets"]  # type: ignore[arg-type]
    )
    viability_targets = [
        list(row)
        for row in empty["viability_targets"]  # type: ignore[union-attr]
    ]
    viability_observed = [
        list(row)
        for row in empty["viability_observed"]  # type: ignore[union-attr]
    ]
    action_viability_targets = [
        [list(components) for components in row]
        for row in empty["action_viability_targets"]  # type: ignore[union-attr]
    ]
    action_viability_observed = [
        [list(components) for components in row]
        for row in empty["action_viability_observed"]  # type: ignore[union-attr]
    ]

    matched_count = 0
    action_mismatch_count = 0
    illegal_logged_action_count = 0
    terminal_alive_count = 0
    value_scores: list[float] = []
    sample_weight_values: list[float] = []
    animal_resource_gain_total = 0.0
    action_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    viability_positive_counts: Counter[str] = Counter()
    record_label_misses = 0
    for row_index, (record, transition) in enumerate(
        zip(records, transitions, strict=True)
    ):
        label = _counterfactual_label_for_record(record, label_lookup)
        if label is None:
            record_label_misses += 1
            continue
        action_support = _mapping(label.get("action_support"))
        logged_action = str(action_support.get("logged_action", ""))
        if logged_action != str(transition.action):
            action_mismatch_count += 1
            continue
        if action_support.get("logged_action_legal") is not True:
            illegal_logged_action_count += 1
            continue
        target = _counterfactual_terminal_target(label)
        value_score = _counterfactual_target_value_score(target)
        row_weight = 1.0 + float(weight_scale) * value_score
        sample_weights[row_index] = row_weight
        loss_weights[row_index] = row_weight
        action_value_targets[row_index] = value_score
        matched_count += 1
        value_scores.append(value_score)
        sample_weight_values.append(row_weight)
        action_counts[logged_action] += 1
        script = label.get("source_script")
        if isinstance(script, str) and script:
            script_counts[script] += 1
        if target.get("terminal_alive") is True:
            terminal_alive_count += 1
        animal_resource_gain_total += _finite_float(
            target.get("animal_resource_gain_to_terminal"),
            default=0.0,
        )
        component_targets, component_observed = (
            _counterfactual_viability_components(target, action_support)
        )
        logged_action_index = action_index[logged_action]
        for component_index, component in enumerate(VIABILITY_COMPONENT_NAMES):
            if not component_observed[component_index]:
                continue
            target_value = component_targets[component_index]
            viability_targets[row_index][component_index] = target_value
            viability_observed[row_index][component_index] = 1.0
            action_viability_targets[row_index][logged_action_index][
                component_index
            ] = target_value
            action_viability_observed[row_index][logged_action_index][
                component_index
            ] = 1.0
            if target_value > 0.0:
                viability_positive_counts[component] += 1

    if matched_count <= 0:
        raise ValueError(
            "counterfactual label report did not match any training records"
        )

    diagnostics = {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        "policy": TORCH_IQL_COUNTERFACTUAL_LABEL_SUPERVISION_POLICY,
        "label_report_digest": stable_payload_digest(
            {
                "schema_version": label_report.get("schema_version"),
                "source": label_report.get("source"),
                "aggregate": label_report.get("aggregate"),
                "label_count": len(labels_payload),
            }
        ),
        "label_report_label_count": len(labels_payload),
        "usable_label_count": usable_label_count,
        "duplicate_lookup_key_count": duplicate_label_count,
        "training_record_count": row_count,
        "matched_record_count": matched_count,
        "matched_record_rate": _round(matched_count / float(row_count))
        if row_count
        else 0.0,
        "unmatched_record_count": record_label_misses,
        "action_mismatch_count": action_mismatch_count,
        "illegal_logged_action_count": illegal_logged_action_count,
        "terminal_alive_label_count": terminal_alive_count,
        "terminal_alive_label_rate": _round(
            terminal_alive_count / float(matched_count)
        ),
        "animal_resource_gain_to_terminal_total": _round(
            animal_resource_gain_total
        ),
        "action_value_score_mean": _safe_mean(value_scores),
        "action_value_score_min": _round(min(value_scores)),
        "action_value_score_max": _round(max(value_scores)),
        "sample_weight_mean": _safe_mean(sample_weight_values),
        "sample_weight_min": _round(min(sample_weight_values)),
        "sample_weight_max": _round(max(sample_weight_values)),
        "action_counts": dict(sorted(action_counts.items())),
        "source_script_counts": dict(sorted(script_counts.items())),
        "viability_positive_counts": dict(
            sorted(viability_positive_counts.items())
        ),
        "weight_policy": TORCH_IQL_COUNTERFACTUAL_LABEL_WEIGHT_POLICY,
        "weight_scale": _round(weight_scale),
    }
    return {
        "sample_weights": tuple(sample_weights),
        "loss_weights": tuple(loss_weights),
        "action_value_targets": tuple(action_value_targets),
        "viability_targets": tuple(tuple(row) for row in viability_targets),
        "viability_observed": tuple(tuple(row) for row in viability_observed),
        "action_viability_targets": tuple(
            tuple(tuple(components) for components in row)
            for row in action_viability_targets
        ),
        "action_viability_observed": tuple(
            tuple(tuple(components) for components in row)
            for row in action_viability_observed
        ),
        "diagnostics": diagnostics,
    }


def _empty_counterfactual_label_supervision(
    row_count: int,
    *,
    weight_scale: float,
) -> dict[str, object]:
    component_count = len(VIABILITY_COMPONENT_NAMES)
    action_count = len(ACTION_NAMES)
    return {
        "sample_weights": tuple(1.0 for _ in range(row_count)),
        "loss_weights": tuple(0.0 for _ in range(row_count)),
        "action_value_targets": tuple(0.0 for _ in range(row_count)),
        "viability_targets": tuple(
            tuple(0.0 for _ in range(component_count))
            for _ in range(row_count)
        ),
        "viability_observed": tuple(
            tuple(0.0 for _ in range(component_count))
            for _ in range(row_count)
        ),
        "action_viability_targets": tuple(
            tuple(
                tuple(0.0 for _ in range(component_count))
                for _ in range(action_count)
            )
            for _ in range(row_count)
        ),
        "action_viability_observed": tuple(
            tuple(
                tuple(0.0 for _ in range(component_count))
                for _ in range(action_count)
            )
            for _ in range(row_count)
        ),
        "diagnostics": {
            "schema_version": None,
            "policy": TORCH_IQL_COUNTERFACTUAL_LABEL_SUPERVISION_POLICY,
            "label_report_digest": None,
            "label_report_label_count": 0,
            "usable_label_count": 0,
            "duplicate_lookup_key_count": 0,
            "training_record_count": row_count,
            "matched_record_count": 0,
            "matched_record_rate": 0.0,
            "unmatched_record_count": row_count,
            "action_mismatch_count": 0,
            "illegal_logged_action_count": 0,
            "terminal_alive_label_count": 0,
            "terminal_alive_label_rate": 0.0,
            "animal_resource_gain_to_terminal_total": 0.0,
            "action_value_score_mean": None,
            "action_value_score_min": None,
            "action_value_score_max": None,
            "sample_weight_mean": None,
            "sample_weight_min": None,
            "sample_weight_max": None,
            "action_counts": {},
            "source_script_counts": {},
            "viability_positive_counts": {},
            "weight_policy": TORCH_IQL_COUNTERFACTUAL_LABEL_WEIGHT_POLICY,
            "weight_scale": _round(weight_scale),
        },
    }


def _counterfactual_label_for_record(
    record: Mapping[str, object],
    label_lookup: Mapping[tuple[object, ...], Mapping[str, object]],
) -> Mapping[str, object] | None:
    for key in _counterfactual_record_lookup_keys(record):
        label = label_lookup.get(key)
        if label is not None:
            return label
    return None


def _counterfactual_label_lookup_keys(
    label: Mapping[str, object],
) -> tuple[tuple[object, ...], ...]:
    keys: list[tuple[object, ...]] = []
    dataset_record_index = _optional_int(label.get("dataset_record_index"))
    trajectory_path = _optional_string(label.get("trajectory_path"))
    if trajectory_path is not None and dataset_record_index is not None:
        for path in _path_lookup_variants(trajectory_path):
            keys.append(("path-record", path, dataset_record_index))
    episode_id = _optional_string(label.get("episode_id"))
    tick = _optional_int(label.get("tick"))
    agent_id = _optional_int(label.get("agent_id"))
    if episode_id is not None and tick is not None and agent_id is not None:
        keys.append(("episode-tick-agent", episode_id, tick, agent_id))
    return tuple(dict.fromkeys(keys))


def _counterfactual_record_lookup_keys(
    record: Mapping[str, object],
) -> tuple[tuple[object, ...], ...]:
    keys: list[tuple[object, ...]] = []
    dataset_record_index = _optional_int(
        record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD)
    )
    trajectory_path = _optional_string(record.get(TRAJECTORY_SOURCE_PATH_FIELD))
    if trajectory_path is not None and dataset_record_index is not None:
        for path in _path_lookup_variants(trajectory_path):
            keys.append(("path-record", path, dataset_record_index))
    episode_id = _optional_string(record.get(TRAJECTORY_EPISODE_ID_FIELD))
    tick = _optional_int(record.get("tick"))
    agent_id = _optional_int(record.get("agent_id"))
    if episode_id is not None and tick is not None and agent_id is not None:
        keys.append(("episode-tick-agent", episode_id, tick, agent_id))
    return tuple(dict.fromkeys(keys))


def _path_lookup_variants(value: str) -> tuple[str, ...]:
    raw = str(value)
    try:
        resolved = str(Path(raw).resolve(strict=False))
    except OSError:
        return (raw,)
    return tuple(dict.fromkeys((raw, resolved)))


def _counterfactual_terminal_target(
    label: Mapping[str, object],
) -> Mapping[str, object]:
    rollout = _mapping(label.get("rollout_terminal_target"))
    if rollout:
        return rollout
    return _mapping(label.get("primary_target"))


def _counterfactual_target_value_score(target: Mapping[str, object]) -> float:
    action_value = _mapping(target.get("action_value"))
    score = _finite_float(action_value.get("score"), default=0.0)
    return _clamp01(score)


def _counterfactual_viability_components(
    target: Mapping[str, object],
    action_support: Mapping[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    terminal_state = _mapping(target.get("terminal_state"))
    terminal_alive = target.get("terminal_alive")
    components: dict[str, tuple[float, float]] = {
        "death_or_survival_horizon_risk": (
            1.0 if terminal_alive is False else 0.0,
            1.0 if isinstance(terminal_alive, bool) else 0.0,
        ),
        "energy_floor_risk": _ratio_risk_component(
            terminal_state.get("energy_ratio"),
            floor=VIABILITY_FLOOR_RISK_RATIO,
        ),
        "hydration_floor_risk": _ratio_risk_component(
            terminal_state.get("hydration_ratio"),
            floor=VIABILITY_FLOOR_RISK_RATIO,
        ),
        "health_floor_risk": _ratio_risk_component(
            terminal_state.get("health_ratio"),
            floor=VIABILITY_HEALTH_FLOOR_RISK_RATIO,
        ),
        "invalid_action": (
            0.0 if action_support.get("logged_action_legal") is True else 1.0,
            1.0,
        ),
        VIABILITY_SUPPRESSION_COMPONENT: (0.0, 1.0),
    }
    return (
        tuple(components[component][0] for component in VIABILITY_COMPONENT_NAMES),
        tuple(components[component][1] for component in VIABILITY_COMPONENT_NAMES),
    )


def _ratio_risk_component(value: object, *, floor: float) -> tuple[float, float]:
    parsed = _optional_float(value)
    if parsed is None:
        return (0.0, 0.0)
    return (1.0 if parsed <= floor else 0.0, 1.0)


def _mapping(payload: object) -> Mapping[str, object]:
    return payload if isinstance(payload, Mapping) else {}


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _finite_float(value: object, *, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _safe_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return _round(sum(finite) / float(len(finite)))


def _validate_counterfactual_weight_scale(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("counterfactual_label_weight_scale must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError("counterfactual_label_weight_scale must be finite")
    if float(value) < 0.0:
        raise ValueError("counterfactual_label_weight_scale must be non-negative")


def _nonnegative_float_override(
    value: float | None,
    *,
    default: float,
    field: str,
) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _positive_float_override(
    value: float | None,
    *,
    default: float,
    field: str,
) -> float:
    parsed = _nonnegative_float_override(value, default=default, field=field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _probability_float_override(
    value: float | None,
    *,
    default: float,
    field: str,
) -> float:
    parsed = _positive_float_override(value, default=default, field=field)
    if parsed > 1.0:
        raise ValueError(f"{field} must be less than or equal to 1")
    return parsed


def _actor_weighting_policy(
    *,
    calibrated_actor_extraction: bool,
    constraint_aware_actor_extraction: bool,
    risk_adjusted_actor_extraction: bool,
    calibrated_supported_actor_extraction: bool,
    contextual_behavior_supported_actor_extraction: bool,
    contextual_behavior_prior_regularization: bool,
    action_distribution_regularization: bool,
    rollout_state_action_calibration: bool,
) -> str:
    policies = [
        TORCH_IQL_CALIBRATED_ACTOR_WEIGHTING_POLICY
        if calibrated_actor_extraction
        else TORCH_IQL_ACTOR_WEIGHTING_POLICY
    ]
    if constraint_aware_actor_extraction:
        policies.append(TORCH_IQL_CONSTRAINT_AWARE_ACTOR_WEIGHTING_POLICY)
    if risk_adjusted_actor_extraction:
        policies.append(TORCH_IQL_RISK_ADJUSTED_ACTOR_EXTRACTION_POLICY)
    if calibrated_supported_actor_extraction:
        policies.append(TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EXTRACTION_POLICY)
    if contextual_behavior_supported_actor_extraction:
        policies.append(
            TORCH_IQL_CONTEXTUAL_BEHAVIOR_SUPPORTED_ACTOR_EXTRACTION_POLICY
        )
    if contextual_behavior_prior_regularization:
        policies.append(
            TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_REGULARIZATION_POLICY
        )
    if action_distribution_regularization:
        policies.append(TORCH_IQL_ACTION_DISTRIBUTION_REGULARIZATION_POLICY)
    if rollout_state_action_calibration:
        policies.append(TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY)
    return "+".join(policies)


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
                self.viability = torch.nn.Linear(
                    TORCH_NEURAL_HIDDEN_UNITS,
                    len(VIABILITY_COMPONENT_NAMES),
                )
                self.action_viability = torch.nn.Linear(
                    TORCH_NEURAL_HIDDEN_UNITS,
                    len(ACTION_NAMES) * len(VIABILITY_COMPONENT_NAMES),
                )

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


def _resolve_torch_device_metadata(
    torch: Any,
    *,
    requested_device: str,
) -> dict[str, object]:
    normalized = str(requested_device).strip().lower()
    if normalized not in TORCH_DEVICE_CHOICES:
        raise ValueError("torch_device must be one of cpu, cuda, mps, auto")
    cuda_available = _torch_cuda_available(torch)
    mps_available = _torch_mps_available(torch)
    if normalized == "auto":
        if cuda_available:
            resolved = "cuda"
        elif mps_available:
            resolved = "mps"
        else:
            resolved = "cpu"
    elif normalized == "cuda":
        if not cuda_available:
            raise ValueError("requested torch_device 'cuda' is not available")
        resolved = "cuda"
    elif normalized == "mps":
        if not mps_available:
            raise ValueError("requested torch_device 'mps' is not available")
        resolved = "mps"
    else:
        resolved = "cpu"

    return {
        "policy": TORCH_DEVICE_POLICY,
        "requested_device": normalized,
        "resolved_device": resolved,
        "cuda_available": cuda_available,
        "cuda_device_count": _torch_cuda_device_count(torch),
        "cuda_device_name": _torch_cuda_device_name(torch),
        "mps_available": mps_available,
    }


def _torch_cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _torch_cuda_device_count(torch: Any) -> int:
    cuda = getattr(torch, "cuda", None)
    device_count = getattr(cuda, "device_count", None)
    if not callable(device_count):
        return 0
    try:
        return int(device_count())
    except (TypeError, RuntimeError):
        return 0


def _torch_cuda_device_name(torch: Any) -> str | None:
    if not _torch_cuda_available(torch):
        return None
    cuda = getattr(torch, "cuda", None)
    get_device_name = getattr(cuda, "get_device_name", None)
    if not callable(get_device_name):
        return None
    try:
        return str(get_device_name(0))
    except (TypeError, RuntimeError):
        return None


def _torch_mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _balanced_class_weights(
    torch: Any,
    labels: list[int],
    *,
    device: Any | None = None,
) -> Any:
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
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _viability_positive_weights(
    torch: Any,
    targets: list[list[float]],
    observed: list[list[float]],
    *,
    device: Any | None = None,
) -> Any:
    if not targets:
        return torch.ones(
            len(VIABILITY_COMPONENT_NAMES),
            dtype=torch.float32,
            device=device,
        )
    weights: list[float] = []
    for index, _component in enumerate(VIABILITY_COMPONENT_NAMES):
        positives = 0
        observed_count = 0
        for target, observed_components in zip(targets, observed, strict=True):
            if observed_components[index] <= 0.0:
                continue
            observed_count += 1
            if target[index] > 0.0:
                positives += 1
        weights.append(_positive_weight(observed_count, positives))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _action_viability_positive_weights(
    torch: Any,
    targets: list[list[list[float]]],
    observed: list[list[list[float]]],
    *,
    device: Any | None = None,
) -> Any:
    if not targets:
        return torch.ones(
            len(VIABILITY_COMPONENT_NAMES),
            dtype=torch.float32,
            device=device,
        )
    weights: list[float] = []
    for index, _component in enumerate(VIABILITY_COMPONENT_NAMES):
        positives = 0
        observed_count = 0
        for target_row, observed_row in zip(targets, observed, strict=True):
            for action_targets, action_observed in zip(
                target_row,
                observed_row,
                strict=True,
            ):
                if action_observed[index] <= 0.0:
                    continue
                observed_count += 1
                if action_targets[index] > 0.0:
                    positives += 1
        weights.append(_positive_weight(observed_count, positives))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _positive_weight(observed_count: int, positives: int) -> float:
    if observed_count <= 0 or positives <= 0:
        return 1.0
    negatives = float(max(observed_count - positives, 0))
    return _clamp(
        negatives / float(positives),
        1.0,
        TORCH_IQL_VIABILITY_POS_WEIGHT_MAX,
    )


def _state_viability_observed(
    targets: list[list[float]],
) -> list[list[float]]:
    suppression_index = VIABILITY_COMPONENT_NAMES.index(
        VIABILITY_SUPPRESSION_COMPONENT
    )
    return [
        [
            0.0 if index == suppression_index else 1.0
            for index, _component in enumerate(VIABILITY_COMPONENT_NAMES)
        ]
        for _target in targets
    ]


def _observed_component_positive_counts(
    targets: list[list[float]],
    observed: list[list[float]],
) -> dict[str, int]:
    return {
        component: int(
            sum(
                1
                for target, observed_components in zip(targets, observed, strict=True)
                if observed_components[index] > 0.0 and target[index] > 0.0
            )
        )
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _observed_component_counts(
    observed: list[list[float]],
) -> dict[str, int]:
    return {
        component: int(
            sum(
                1
                for observed_components in observed
                if observed_components[index] > 0.0
            )
        )
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _observed_component_rates(
    positives: dict[str, int],
    observed: dict[str, int],
) -> dict[str, float]:
    return {
        component: _round(
            float(positives[component]) / float(observed[component])
        )
        if observed[component] > 0
        else 0.0
        for component in VIABILITY_COMPONENT_NAMES
    }


def _behavior_margin_anchor_weights(
    viability_targets: list[list[float]],
    *,
    labels: list[int],
    action_masks: list[list[bool]],
) -> list[float]:
    suppression_index = VIABILITY_COMPONENT_NAMES.index(
        VIABILITY_SUPPRESSION_COMPONENT
    )
    weights: list[float] = []
    for target, label, action_mask in zip(
        viability_targets,
        labels,
        action_masks,
        strict=True,
    ):
        if label < 0 or label >= len(action_mask) or not action_mask[label]:
            weights.append(0.0)
            continue
        has_non_suppression_risk = any(
            value > 0.0 and index != suppression_index
            for index, value in enumerate(target)
        )
        weights.append(0.0 if has_non_suppression_risk else 1.0)
    return weights


def _constraint_aware_actor_weights(
    action_viability_targets: list[list[list[float]]],
    *,
    labels: list[int],
) -> tuple[list[float], dict[str, object]]:
    if not action_viability_targets:
        return [], _constraint_actor_stats([])
    suppression_index = VIABILITY_COMPONENT_NAMES.index(
        VIABILITY_SUPPRESSION_COMPONENT
    )
    weights: list[float] = []
    component_counts: Counter[str] = Counter()
    risky_count = 0
    for target_row, label in zip(action_viability_targets, labels, strict=True):
        logged_components = target_row[label]
        risky_components = [
            component
            for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
            if index != suppression_index and logged_components[index] > 0.0
        ]
        if risky_components:
            risky_count += 1
            for component in risky_components:
                component_counts[component] += 1
            weights.append(TORCH_IQL_CONSTRAINT_AWARE_ACTOR_WEIGHT_MIN)
        else:
            weights.append(1.0)
    stats = _constraint_actor_stats(weights)
    stats["risky_count"] = risky_count
    stats["risky_rate"] = (
        float(risky_count) / float(len(weights)) if weights else 0.0
    )
    stats["component_counts"] = {
        component: int(component_counts.get(component, 0))
        for component in VIABILITY_COMPONENT_NAMES
        if component != VIABILITY_SUPPRESSION_COMPONENT
    }
    return weights, stats


def _constraint_actor_stats(weights: list[float]) -> dict[str, object]:
    if not weights:
        return {
            "weight_mean": 1.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
            "risky_count": 0,
            "risky_rate": 0.0,
            "component_counts": {
                component: 0
                for component in VIABILITY_COMPONENT_NAMES
                if component != VIABILITY_SUPPRESSION_COMPONENT
            },
        }
    return {
        "weight_mean": sum(weights) / float(len(weights)),
        "weight_min": min(weights),
        "weight_max": max(weights),
        "risky_count": 0,
        "risky_rate": 0.0,
        "component_counts": {
            component: 0
            for component in VIABILITY_COMPONENT_NAMES
            if component != VIABILITY_SUPPRESSION_COMPONENT
        },
    }


def _viability_component_counts(
    targets: tuple[dict[str, bool], ...],
) -> dict[str, int]:
    return {
        component: int(
            sum(1 for target in targets if bool(target.get(component, False)))
        )
        for component in VIABILITY_COMPONENT_NAMES
    }


def _viability_component_rates(
    targets: tuple[dict[str, bool], ...],
) -> dict[str, float]:
    denominator = float(len(targets))
    if denominator <= 0.0:
        return {component: 0.0 for component in VIABILITY_COMPONENT_NAMES}
    counts = _viability_component_counts(targets)
    return {
        component: _round(float(counts[component]) / denominator)
        for component in VIABILITY_COMPONENT_NAMES
    }


def _action_viability_supervision(
    transitions: tuple[Any, ...],
    component_targets: tuple[dict[str, bool], ...],
    *,
    action_index: dict[str, int],
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    component_index = {
        component: index
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }
    suppression_index = component_index[VIABILITY_SUPPRESSION_COMPONENT]
    targets: list[list[list[float]]] = []
    observed: list[list[list[float]]] = []
    for transition, components in zip(transitions, component_targets, strict=True):
        row_targets = [
            [0.0 for _component in VIABILITY_COMPONENT_NAMES]
            for _action in ACTION_NAMES
        ]
        row_observed = [
            [0.0 for _component in VIABILITY_COMPONENT_NAMES]
            for _action in ACTION_NAMES
        ]
        logged_action_index = action_index[transition.action]
        for component, active in components.items():
            if component == VIABILITY_SUPPRESSION_COMPONENT:
                continue
            index = component_index[component]
            row_targets[logged_action_index][index] = 1.0 if active else 0.0
            row_observed[logged_action_index][index] = 1.0
        suppressed_action_index, feedback_weight = _guard_feedback_record(
            transition,
            action_index=action_index,
        )
        if feedback_weight > 0.0 and suppressed_action_index >= 0:
            row_targets[suppressed_action_index][suppression_index] = 1.0
            row_observed[suppressed_action_index][suppression_index] = 1.0
        elif _is_learned_policy_action_source(transition.action_source):
            row_targets[logged_action_index][suppression_index] = 0.0
            row_observed[logged_action_index][suppression_index] = 1.0
        targets.append(row_targets)
        observed.append(row_observed)
    return targets, observed


def _action_viability_component_positive_counts(
    targets: list[list[list[float]]],
    observed: list[list[list[float]]],
) -> dict[str, int]:
    return {
        component: int(
            sum(
                1
                for target_row, observed_row in zip(targets, observed, strict=True)
                for action_targets, action_observed in zip(
                    target_row,
                    observed_row,
                    strict=True,
                )
                if action_observed[index] > 0.0 and action_targets[index] > 0.0
            )
        )
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _action_viability_component_observed_counts(
    observed: list[list[list[float]]],
) -> dict[str, int]:
    return {
        component: int(
            sum(
                1
                for observed_row in observed
                for action_observed in observed_row
                if action_observed[index] > 0.0
            )
        )
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _logged_action_component_positive_count(
    targets: list[list[list[float]]],
    labels: list[int],
    *,
    component: str,
) -> int:
    component_index = VIABILITY_COMPONENT_NAMES.index(component)
    return int(
        sum(
            1
            for row, label in zip(targets, labels, strict=True)
            if row[label][component_index] > 0.0
        )
    )


def _action_mask_values(action_mask: dict[str, bool]) -> list[bool]:
    return [bool(action_mask.get(action, False)) for action in ACTION_NAMES]


def _masked_logits(torch: Any, logits: Any, action_masks: Any) -> Any:
    return logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)


def _guard_feedback_record(
    transition: Any,
    *,
    action_index: dict[str, int],
) -> tuple[int, float]:
    diagnostics = transition.policy_decision_diagnostics
    hard_guard_used = HEURISTIC_GUARD_POLICY in transition.action_source or (
        _diagnostic_bool(diagnostics, "guard_used")
    )
    heuristic_delegate_used = (
        HEURISTIC_DELEGATE_POLICY in transition.action_source
        or _diagnostic_bool(diagnostics, "heuristic_delegate_used")
    )
    if not hard_guard_used and not heuristic_delegate_used:
        return -1, 0.0

    learned_action = _diagnostic_action(diagnostics, "learned_action")
    if learned_action is None or learned_action == transition.action:
        return -1, 0.0
    if not bool(transition.action_mask.get(learned_action, False)):
        return -1, 0.0
    suppressed_index = action_index.get(learned_action)
    if suppressed_index is None:
        return -1, 0.0
    if hard_guard_used:
        return suppressed_index, TORCH_IQL_HARD_GUARD_FEEDBACK_WEIGHT
    return suppressed_index, TORCH_IQL_HEURISTIC_DELEGATE_FEEDBACK_WEIGHT


def _replay_weight_record(transition: Any) -> float:
    if not _is_learned_policy_action_source(transition.action_source):
        return 1.0
    diagnostics = transition.policy_decision_diagnostics
    hard_guard_used = HEURISTIC_GUARD_POLICY in transition.action_source or (
        _diagnostic_bool(diagnostics, "guard_used")
    )
    heuristic_delegate_used = (
        HEURISTIC_DELEGATE_POLICY in transition.action_source
        or _diagnostic_bool(diagnostics, "heuristic_delegate_used")
    )
    if hard_guard_used or heuristic_delegate_used:
        return TORCH_IQL_LEARNED_REPLAY_GUARDED_ACTION_WEIGHT
    return TORCH_IQL_LEARNED_REPLAY_SELF_ACTION_WEIGHT


def _is_learned_policy_action_source(action_source: object) -> bool:
    if not isinstance(action_source, str):
        return False
    return action_source.startswith(
        "mind_v1_learned_policy"
    ) or action_source.startswith("mind_v2_neural_policy")


def _weighted_mean(values: Any, weights: Any) -> Any:
    return (values * weights).sum() / (
        weights.sum() * float(values.shape[1] if values.dim() > 1 else 1)
    ).clamp_min(1e-9)


def _masked_component_mean(values: Any, observed: Any, row_weights: Any) -> Any:
    weights = observed * row_weights.view(-1, 1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-9)


def _masked_action_component_mean(values: Any, observed: Any, row_weights: Any) -> Any:
    weights = observed * row_weights.view(-1, 1, 1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-9)


def _behavior_margin_anchor_loss(
    torch: Any,
    masked_logits: Any,
    y: Any,
    action_masks: Any,
    sample_weights: Any,
    anchor_weights: Any,
) -> Any:
    active_weights = sample_weights * anchor_weights
    if not bool((active_weights > 0.0).any().detach().cpu().item()):
        return masked_logits.new_zeros(())
    target_logits = masked_logits.gather(1, y.unsqueeze(1)).squeeze(1)
    target_mask = torch.nn.functional.one_hot(
        y,
        num_classes=len(ACTION_NAMES),
    ).bool()
    competitor_logits = masked_logits.masked_fill(
        target_mask | ~action_masks,
        torch.finfo(masked_logits.dtype).min,
    ).max(dim=1).values
    margin_losses = torch.nn.functional.relu(
        competitor_logits
        - target_logits
        + TORCH_IQL_BEHAVIOR_MARGIN_ANCHOR_TARGET
    )
    return (
        (margin_losses * active_weights).sum()
        / active_weights.sum().clamp_min(1e-9)
    )


def _iql_actor_weights(
    torch: Any,
    actor_advantage: Any,
    replay_weights: Any,
    *,
    calibrated_actor_extraction: bool,
) -> tuple[Any, dict[str, float]]:
    if calibrated_actor_extraction:
        row_weights = replay_weights.view(-1, 1)
        weight_total = row_weights.sum().clamp_min(1e-9)
        advantage_mean = (actor_advantage * row_weights).sum() / weight_total
        centered_advantage = actor_advantage - advantage_mean
        advantage_scale = torch.sqrt(
            ((centered_advantage.pow(2) * row_weights).sum() / weight_total)
        ).clamp_min(TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON)
        weighted_advantage = centered_advantage / advantage_scale
        calibrated_weights = torch.exp(
            weighted_advantage / TORCH_IQL_CALIBRATED_ADVANTAGE_TEMPERATURE
        ).clamp(
            min=TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_MIN,
            max=TORCH_IQL_ADVANTAGE_WEIGHT_MAX,
        )
        calibrated_weight_mean = (
            calibrated_weights * row_weights
        ).sum() / weight_total
        normalized_weights = calibrated_weights / calibrated_weight_mean.clamp_min(
            1e-9
        )
        actor_weights = (
            1.0 - TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_BLEND
        ) + TORCH_IQL_CALIBRATED_ADVANTAGE_WEIGHT_BLEND * normalized_weights
    else:
        advantage_mean = actor_advantage.mean()
        advantage_scale = torch.sqrt(
            (actor_advantage - advantage_mean).pow(2).mean()
        ).clamp_min(TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON)
        actor_weights = torch.exp(
            actor_advantage / TORCH_IQL_ADVANTAGE_TEMPERATURE
        ).clamp(max=TORCH_IQL_ADVANTAGE_WEIGHT_MAX)
    return actor_weights, {
        "mean": float(actor_weights.mean().detach().cpu().item()),
        "min": float(actor_weights.min().detach().cpu().item()),
        "max": float(actor_weights.max().detach().cpu().item()),
        "advantage_mean": float(advantage_mean.detach().cpu().item()),
        "advantage_scale": float(advantage_scale.detach().cpu().item()),
    }


def _cql_conservative_loss(
    torch: Any,
    action_values: Any,
    y: Any,
    action_masks: Any,
    sample_weights: Any,
) -> Any:
    gaps = _cql_conservative_gaps(torch, action_values, y, action_masks)
    return _weighted_mean(gaps, sample_weights.unsqueeze(1))


def _cql_conservative_gaps(
    torch: Any,
    action_values: Any,
    y: Any,
    action_masks: Any,
) -> Any:
    masked_action_values = _masked_logits(torch, action_values, action_masks)
    legal_logsumexp = (
        torch.logsumexp(masked_action_values / TORCH_IQL_CQL_TEMPERATURE, dim=1)
        * TORCH_IQL_CQL_TEMPERATURE
    ).unsqueeze(1)
    selected_action_values = action_values.gather(1, y.unsqueeze(1))
    return legal_logsumexp - selected_action_values


def _guard_feedback_loss(
    torch: Any,
    masked_logits: Any,
    y: Any,
    guard_feedback_actions: Any,
    guard_feedback_weights: Any,
    *,
    margin: float = TORCH_IQL_GUARD_FEEDBACK_MARGIN,
) -> Any:
    active_mask = guard_feedback_weights > 0.0
    if not bool(active_mask.any().detach().cpu().item()):
        return guard_feedback_weights.sum() * 0.0
    active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
    target_logits = masked_logits[active_rows, y[active_rows]]
    suppressed_logits = masked_logits[
        active_rows,
        guard_feedback_actions[active_rows],
    ]
    margin_losses = torch.nn.functional.relu(
        suppressed_logits
        - target_logits
        + margin
    )
    active_weights = guard_feedback_weights[active_rows]
    return (
        (margin_losses * active_weights).sum()
        / active_weights.sum().clamp_min(1e-9)
    )


def _online_update_feedback_record(
    transition: Any,
    *,
    action_index: dict[str, int],
) -> tuple[int, float]:
    trace = transition.policy_update_trace
    if not isinstance(trace, dict):
        return -1, 0.0
    if trace.get("schema_version") != "mind_policy_update_trace_v1":
        return -1, 0.0
    if trace.get("policy") != "in_run_contextual_bandit_adapter_v1":
        return -1, 0.0
    action = trace.get("action")
    if action != transition.action:
        return -1, 0.0
    index = action_index.get(str(action))
    if index is None:
        return -1, 0.0
    reward_signal = trace.get("reward_signal")
    if isinstance(reward_signal, bool) or not isinstance(
        reward_signal,
        (int, float),
    ):
        return -1, 0.0
    parsed_signal = float(reward_signal)
    if not math.isfinite(parsed_signal):
        return -1, 0.0
    return index, max(-1.0, min(1.0, parsed_signal))


def _online_update_feedback_loss(
    torch: Any,
    logits: Any,
    action_masks: Any,
    feedback_actions: Any,
    feedback_signals: Any,
) -> Any:
    active_mask = feedback_signals != 0.0
    if not bool(active_mask.any().detach().cpu().item()):
        return logits.sum() * 0.0
    active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
    active_logits = logits[active_rows]
    active_masks = action_masks[active_rows]
    active_actions = feedback_actions[active_rows]
    active_signals = feedback_signals[active_rows]
    selected_logits = active_logits[
        torch.arange(active_rows.numel(), device=active_logits.device),
        active_actions,
    ]
    legal_counts = active_masks.float().sum(dim=1).clamp_min(1.0)
    legal_mean_logits = (
        active_logits.masked_fill(~active_masks, 0.0).sum(dim=1) / legal_counts
    )
    positive_margin = torch.nn.functional.relu(
        legal_mean_logits
        - selected_logits
        + TORCH_IQL_ONLINE_UPDATE_FEEDBACK_MARGIN
    )
    negative_margin = torch.nn.functional.relu(
        selected_logits
        - legal_mean_logits
        + TORCH_IQL_ONLINE_UPDATE_FEEDBACK_MARGIN
    )
    margin_losses = torch.where(
        active_signals > 0.0,
        positive_margin,
        negative_margin,
    )
    return (
        (margin_losses * active_signals.abs()).sum()
        / active_signals.abs().sum().clamp_min(1e-9)
    )


def _return_calibration_loss(
    torch: Any,
    selected_action_values: Any,
    state_values: Any,
    return_target: Any,
    replay_weights: Any,
) -> Any:
    action_return_losses = torch.nn.functional.smooth_l1_loss(
        selected_action_values,
        return_target,
        reduction="none",
    )
    state_return_losses = torch.nn.functional.smooth_l1_loss(
        state_values,
        return_target,
        reduction="none",
    )
    return _weighted_mean(
        action_return_losses + state_return_losses,
        replay_weights.unsqueeze(1),
    )


def _suppression_critic_calibration_loss(
    torch: Any,
    action_values: Any,
    state_values: Any,
    y: Any,
    guard_feedback_actions: Any,
    guard_feedback_weights: Any,
    *,
    margin: float = TORCH_IQL_SUPPRESSION_CRITIC_CALIBRATION_MARGIN,
) -> Any:
    active_mask = (guard_feedback_weights > 0.0) & (guard_feedback_actions >= 0)
    if not bool(active_mask.any().detach().cpu().item()):
        return action_values.sum() * 0.0
    active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
    selected_values = action_values[active_rows, y[active_rows]]
    suppressed_values = action_values[
        active_rows,
        guard_feedback_actions[active_rows],
    ]
    active_state_values = state_values[active_rows].squeeze(1)
    q_margin_losses = torch.nn.functional.relu(
        suppressed_values - selected_values + margin
    )
    suppressed_advantage_losses = torch.nn.functional.relu(
        suppressed_values - active_state_values
    )
    active_weights = guard_feedback_weights[active_rows]
    losses = (
        q_margin_losses
        + TORCH_IQL_SUPPRESSION_CRITIC_STATE_ADVANTAGE_WEIGHT
        * suppressed_advantage_losses
    )
    return (
        (losses * active_weights).sum()
        / active_weights.sum().clamp_min(1e-9)
    )


def _risk_adjusted_actor_loss(
    torch: Any,
    masked_logits: Any,
    action_values: Any,
    state_values: Any,
    action_masks: Any,
    action_viability_logits: Any,
    y: Any,
    replay_weights: Any,
) -> tuple[Any, dict[str, float]]:
    risk_scores = _action_viability_risk_scores(
        torch,
        action_viability_logits.detach(),
    )
    advantages = action_values.detach() - state_values.detach()
    risk_adjusted_scores = (
        advantages - TORCH_IQL_RISK_ADJUSTED_ACTOR_RISK_PENALTY * risk_scores
    )
    masked_risk_adjusted_scores = _masked_logits(
        torch,
        risk_adjusted_scores,
        action_masks,
    )
    target_probabilities = torch.nn.functional.softmax(
        masked_risk_adjusted_scores / TORCH_IQL_RISK_ADJUSTED_ACTOR_TEMPERATURE,
        dim=1,
    ).detach()
    log_probabilities = torch.nn.functional.log_softmax(masked_logits, dim=1)
    losses = -(target_probabilities * log_probabilities).sum(dim=1)
    loss = (
        (losses * replay_weights).sum()
        / replay_weights.sum().clamp_min(1e-9)
    )

    with torch.no_grad():
        top_actions = target_probabilities.argmax(dim=1)
        row_count = float(max(int(y.numel()), 1))
        target_logged_probability = target_probabilities.gather(
            1,
            y.unsqueeze(1),
        ).squeeze(1)
        entropy = -(
            target_probabilities
            * target_probabilities.clamp_min(1e-9).log()
        ).sum(dim=1)
        top_risk = risk_scores.gather(1, top_actions.unsqueeze(1)).squeeze(1)
        top_advantage = advantages.gather(1, top_actions.unsqueeze(1)).squeeze(1)
        stats = {
            "target_entropy_mean": float(entropy.mean().detach().cpu().item()),
            "target_logged_probability_mean": float(
                target_logged_probability.mean().detach().cpu().item()
            ),
            "target_logged_top1_rate": float(
                (top_actions == y).float().sum().detach().cpu().item() / row_count
            ),
            "target_top_risk_mean": float(top_risk.mean().detach().cpu().item()),
            "target_top_advantage_mean": float(
                top_advantage.mean().detach().cpu().item()
            ),
        }
    return loss, stats


def _action_viability_risk_scores(torch: Any, action_viability_logits: Any) -> Any:
    probabilities = torch.sigmoid(action_viability_logits)
    risk_component_indices = [
        index
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
        if component != VIABILITY_SUPPRESSION_COMPONENT
    ]
    component_probabilities = torch.stack(
        [probabilities[:, :, index] for index in risk_component_indices],
        dim=2,
    )
    return component_probabilities.max(dim=2).values


def _empty_risk_adjusted_actor_stats() -> dict[str, float]:
    return {
        "target_entropy_mean": 0.0,
        "target_logged_probability_mean": 0.0,
        "target_logged_top1_rate": 0.0,
        "target_top_risk_mean": 0.0,
        "target_top_advantage_mean": 0.0,
    }


def _fit_calibrated_supported_actor_report(
    torch: Any,
    model: Any,
    calibration_records: tuple[dict[str, object], ...],
    *,
    action_index: dict[str, int],
) -> dict[str, object]:
    calibration_transitions = build_trajectory_transitions(calibration_records)
    if not calibration_transitions:
        raise ValueError(
            "calibrated supported actor extraction requires a non-empty "
            "calibration trajectory bank"
        )
    component_targets, _aggregate_targets, _survival_horizons = (
        build_viability_component_targets(calibration_records)
    )
    if len(component_targets) != len(calibration_transitions):
        raise ValueError(
            "calibrated supported actor calibration targets must align with "
            "calibration transitions"
        )
    action_viability_targets, action_viability_observed = (
        _action_viability_supervision(
            calibration_transitions,
            component_targets,
            action_index=action_index,
        )
    )
    x = torch.tensor(
        [
            decode_observation_input(transition.observation_input)
            for transition in calibration_transitions
        ],
        dtype=torch.float32,
        device=_model_device(model),
    )
    action_masks = [
        _action_mask_values(transition.action_mask)
        for transition in calibration_transitions
    ]
    with torch.no_grad():
        hidden = torch.tanh(model.hidden(x))
        action_values = model.action_value(hidden)
        state_values = model.state_value(hidden)
        action_viability_logits = model.action_viability(hidden).view(
            -1,
            len(ACTION_NAMES),
            len(VIABILITY_COMPONENT_NAMES),
        )
        probability_rows = (
            torch.sigmoid(action_viability_logits).detach().cpu().tolist()
        )
        advantage_rows = (
            (action_values - state_values).detach().cpu().tolist()
        )

    action_count = len(ACTION_NAMES)
    component_count = len(VIABILITY_COMPONENT_NAMES)
    raw_by_pair: list[list[list[float]]] = [
        [[] for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    target_by_pair: list[list[list[bool]]] = [
        [[] for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    action_support_counts = [0 for _action in ACTION_NAMES]
    action_component_support_counts = [
        [0 for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    legal_advantages: list[float] = []

    for row_index, (target_row, observed_row, action_mask) in enumerate(
        zip(
            action_viability_targets,
            action_viability_observed,
            action_masks,
            strict=True,
        )
    ):
        for action_idx, legal in enumerate(action_mask):
            if legal:
                legal_advantages.append(float(advantage_rows[row_index][action_idx]))
        for action_idx in range(action_count):
            action_observed = False
            for component_idx in range(component_count):
                if observed_row[action_idx][component_idx] <= 0.0:
                    continue
                action_observed = True
                raw_by_pair[action_idx][component_idx].append(
                    float(probability_rows[row_index][action_idx][component_idx])
                )
                target_by_pair[action_idx][component_idx].append(
                    target_row[action_idx][component_idx] > 0.0
                )
                action_component_support_counts[action_idx][component_idx] += 1
            if action_observed:
                action_support_counts[action_idx] += 1

    global_raw_by_component: list[list[float]] = [
        [] for _component in VIABILITY_COMPONENT_NAMES
    ]
    global_target_by_component: list[list[bool]] = [
        [] for _component in VIABILITY_COMPONENT_NAMES
    ]
    for component_idx in range(component_count):
        for action_idx in range(action_count):
            global_raw_by_component[component_idx].extend(
                raw_by_pair[action_idx][component_idx]
            )
            global_target_by_component[component_idx].extend(
                target_by_pair[action_idx][component_idx]
            )

    bias_matrix: list[list[float]] = [
        [0.0 for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    for action_idx in range(action_count):
        for component_idx in range(component_count):
            raw_scores = raw_by_pair[action_idx][component_idx]
            targets = target_by_pair[action_idx][component_idx]
            if raw_scores:
                bias_matrix[action_idx][component_idx] = _calibration_bias(
                    raw_scores,
                    targets,
                )
            else:
                bias_matrix[action_idx][component_idx] = _calibration_bias(
                    global_raw_by_component[component_idx],
                    global_target_by_component[component_idx],
                )

    raw_scores_all: list[float] = []
    calibrated_scores_all: list[float] = []
    targets_all: list[bool] = []
    raw_by_action: dict[str, list[float]] = {action: [] for action in ACTION_NAMES}
    calibrated_by_action: dict[str, list[float]] = {
        action: [] for action in ACTION_NAMES
    }
    target_by_action: dict[str, list[bool]] = {
        action: [] for action in ACTION_NAMES
    }
    raw_by_component: dict[str, list[float]] = {
        component: [] for component in VIABILITY_COMPONENT_NAMES
    }
    calibrated_by_component: dict[str, list[float]] = {
        component: [] for component in VIABILITY_COMPONENT_NAMES
    }
    target_by_component: dict[str, list[bool]] = {
        component: [] for component in VIABILITY_COMPONENT_NAMES
    }
    for action_idx, action in enumerate(ACTION_NAMES):
        for component_idx, component in enumerate(VIABILITY_COMPONENT_NAMES):
            bias = bias_matrix[action_idx][component_idx]
            for raw_score, target in zip(
                raw_by_pair[action_idx][component_idx],
                target_by_pair[action_idx][component_idx],
                strict=True,
            ):
                calibrated_score = _clamp(raw_score + bias, 0.0, 1.0)
                raw_scores_all.append(raw_score)
                calibrated_scores_all.append(calibrated_score)
                targets_all.append(target)
                raw_by_action[action].append(raw_score)
                calibrated_by_action[action].append(calibrated_score)
                target_by_action[action].append(target)
                raw_by_component[component].append(raw_score)
                calibrated_by_component[component].append(calibrated_score)
                target_by_component[component].append(target)

    advantage_mean = (
        sum(legal_advantages) / float(len(legal_advantages))
        if legal_advantages
        else 0.0
    )
    advantage_scale = (
        math.sqrt(
            sum((advantage - advantage_mean) ** 2 for advantage in legal_advantages)
            / float(len(legal_advantages))
        )
        if legal_advantages
        else 1.0
    )
    advantage_scale = max(
        advantage_scale,
        TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON,
    )

    aggregate_metrics = _calibrated_supported_metric_report(
        raw_scores_all,
        calibrated_scores_all,
        targets_all,
    )
    return {
        "schema_version": "mind_calibrated_supported_actor_extraction_v1",
        "policy": TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EXTRACTION_POLICY,
        "risk_calibration_policy": (
            TORCH_IQL_CALIBRATED_SUPPORTED_RISK_CALIBRATION_POLICY
        ),
        "advantage_policy": TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_POLICY,
        "calibration_record_count": len(calibration_records),
        "calibration_transition_count": len(calibration_transitions),
        "component_names": list(VIABILITY_COMPONENT_NAMES),
        "action_names": list(ACTION_NAMES),
        "min_action_support": TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT,
        "risk_threshold": TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD,
        "advantage_threshold": TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD,
        "risk_penalty": TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY,
        "actor_loss_weight": TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_LOSS_WEIGHT,
        "actor_finetune_epochs": TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EPOCHS,
        "advantage_mean": _round(advantage_mean),
        "advantage_scale": _round(advantage_scale),
        "action_support_counts": {
            action: int(action_support_counts[index])
            for index, action in enumerate(ACTION_NAMES)
        },
        "action_component_support_counts": {
            action: {
                component: int(action_component_support_counts[action_idx][component_idx])
                for component_idx, component in enumerate(VIABILITY_COMPONENT_NAMES)
            }
            for action_idx, action in enumerate(ACTION_NAMES)
        },
        "action_component_support_matrix": action_component_support_counts,
        "action_component_bias": {
            action: {
                component: _round(bias_matrix[action_idx][component_idx])
                for component_idx, component in enumerate(VIABILITY_COMPONENT_NAMES)
            }
            for action_idx, action in enumerate(ACTION_NAMES)
        },
        "action_component_bias_matrix": _round_matrix(bias_matrix),
        **aggregate_metrics,
        "by_action": {
            action: _calibrated_supported_metric_report(
                raw_by_action[action],
                calibrated_by_action[action],
                target_by_action[action],
            )
            for action in ACTION_NAMES
        },
        "by_component": {
            component: _calibrated_supported_metric_report(
                raw_by_component[component],
                calibrated_by_component[component],
                target_by_component[component],
            )
            for component in VIABILITY_COMPONENT_NAMES
        },
    }


def _evaluate_calibrated_supported_actor_report(
    torch: Any,
    model: Any,
    calibration_validation_records: tuple[dict[str, object], ...],
    *,
    action_index: dict[str, int],
    fit_report: dict[str, object],
) -> dict[str, object]:
    tables = _calibrated_supported_observation_tables(
        torch,
        model,
        calibration_validation_records,
        action_index=action_index,
    )
    return _calibrated_supported_report_from_tables(
        tables,
        fit_report["action_component_bias_matrix"],
        schema_version=(
            "mind_calibrated_supported_actor_calibration_validation_v1"
        ),
        policy=TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EXTRACTION_POLICY,
        risk_calibration_policy=(
            TORCH_IQL_CALIBRATED_SUPPORTED_RISK_CALIBRATION_POLICY
        ),
        advantage_policy=TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_POLICY,
        advantage_mean=float(fit_report["advantage_mean"]),
        advantage_scale=float(fit_report["advantage_scale"]),
    )


def _calibrated_supported_observation_tables(
    torch: Any,
    model: Any,
    records: tuple[dict[str, object], ...],
    *,
    action_index: dict[str, int],
) -> dict[str, object]:
    transitions = build_trajectory_transitions(records)
    if not transitions:
        raise ValueError("calibration report requires at least one transition")
    component_targets, _aggregate_targets, _survival_horizons = (
        build_viability_component_targets(records)
    )
    if len(component_targets) != len(transitions):
        raise ValueError(
            "calibration report targets must align with calibration transitions"
        )
    action_viability_targets, action_viability_observed = (
        _action_viability_supervision(
            transitions,
            component_targets,
            action_index=action_index,
        )
    )
    x = torch.tensor(
        [
            decode_observation_input(transition.observation_input)
            for transition in transitions
        ],
        dtype=torch.float32,
        device=_model_device(model),
    )
    action_masks = [
        _action_mask_values(transition.action_mask)
        for transition in transitions
    ]
    with torch.no_grad():
        hidden = torch.tanh(model.hidden(x))
        action_values = model.action_value(hidden)
        state_values = model.state_value(hidden)
        action_viability_logits = model.action_viability(hidden).view(
            -1,
            len(ACTION_NAMES),
            len(VIABILITY_COMPONENT_NAMES),
        )
        probability_rows = (
            torch.sigmoid(action_viability_logits).detach().cpu().tolist()
        )
        advantage_rows = (
            (action_values - state_values).detach().cpu().tolist()
        )
    raw_by_pair: list[list[list[float]]] = [
        [[] for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    target_by_pair: list[list[list[bool]]] = [
        [[] for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    action_support_counts = [0 for _action in ACTION_NAMES]
    action_component_support_counts = [
        [0 for _component in VIABILITY_COMPONENT_NAMES]
        for _action in ACTION_NAMES
    ]
    legal_advantages: list[float] = []
    for row_index, (target_row, observed_row, action_mask) in enumerate(
        zip(
            action_viability_targets,
            action_viability_observed,
            action_masks,
            strict=True,
        )
    ):
        for action_idx, legal in enumerate(action_mask):
            if legal:
                legal_advantages.append(float(advantage_rows[row_index][action_idx]))
        for action_idx in range(len(ACTION_NAMES)):
            action_observed = False
            for component_idx in range(len(VIABILITY_COMPONENT_NAMES)):
                if observed_row[action_idx][component_idx] <= 0.0:
                    continue
                action_observed = True
                raw_by_pair[action_idx][component_idx].append(
                    float(probability_rows[row_index][action_idx][component_idx])
                )
                target_by_pair[action_idx][component_idx].append(
                    target_row[action_idx][component_idx] > 0.0
                )
                action_component_support_counts[action_idx][component_idx] += 1
            if action_observed:
                action_support_counts[action_idx] += 1
    return {
        "record_count": len(records),
        "transition_count": len(transitions),
        "raw_by_pair": raw_by_pair,
        "target_by_pair": target_by_pair,
        "action_support_counts": action_support_counts,
        "action_component_support_counts": action_component_support_counts,
        "legal_advantages": legal_advantages,
    }


def _calibrated_supported_report_from_tables(
    tables: dict[str, object],
    bias_matrix: list[list[float]],
    *,
    schema_version: str,
    policy: str,
    risk_calibration_policy: str,
    advantage_policy: str,
    advantage_mean: float,
    advantage_scale: float,
) -> dict[str, object]:
    raw_by_pair = tables["raw_by_pair"]
    target_by_pair = tables["target_by_pair"]
    if not isinstance(raw_by_pair, list) or not isinstance(target_by_pair, list):
        raise ValueError("calibration report tables are malformed")
    raw_scores_all: list[float] = []
    calibrated_scores_all: list[float] = []
    targets_all: list[bool] = []
    raw_by_action: dict[str, list[float]] = {action: [] for action in ACTION_NAMES}
    calibrated_by_action: dict[str, list[float]] = {
        action: [] for action in ACTION_NAMES
    }
    target_by_action: dict[str, list[bool]] = {
        action: [] for action in ACTION_NAMES
    }
    raw_by_component: dict[str, list[float]] = {
        component: [] for component in VIABILITY_COMPONENT_NAMES
    }
    calibrated_by_component: dict[str, list[float]] = {
        component: [] for component in VIABILITY_COMPONENT_NAMES
    }
    target_by_component: dict[str, list[bool]] = {
        component: [] for component in VIABILITY_COMPONENT_NAMES
    }
    for action_idx, action in enumerate(ACTION_NAMES):
        for component_idx, component in enumerate(VIABILITY_COMPONENT_NAMES):
            bias = float(bias_matrix[action_idx][component_idx])
            for raw_score, target in zip(
                raw_by_pair[action_idx][component_idx],
                target_by_pair[action_idx][component_idx],
                strict=True,
            ):
                calibrated_score = _clamp(float(raw_score) + bias, 0.0, 1.0)
                parsed_target = bool(target)
                raw_scores_all.append(float(raw_score))
                calibrated_scores_all.append(calibrated_score)
                targets_all.append(parsed_target)
                raw_by_action[action].append(float(raw_score))
                calibrated_by_action[action].append(calibrated_score)
                target_by_action[action].append(parsed_target)
                raw_by_component[component].append(float(raw_score))
                calibrated_by_component[component].append(calibrated_score)
                target_by_component[component].append(parsed_target)
    aggregate_metrics = _calibrated_supported_metric_report(
        raw_scores_all,
        calibrated_scores_all,
        targets_all,
    )
    action_support_counts = tables["action_support_counts"]
    action_component_support_counts = tables["action_component_support_counts"]
    if not isinstance(action_support_counts, list) or not isinstance(
        action_component_support_counts,
        list,
    ):
        raise ValueError("calibration support tables are malformed")
    return {
        "schema_version": schema_version,
        "policy": policy,
        "risk_calibration_policy": risk_calibration_policy,
        "advantage_policy": advantage_policy,
        "calibration_record_count": int(tables["record_count"]),
        "calibration_transition_count": int(tables["transition_count"]),
        "component_names": list(VIABILITY_COMPONENT_NAMES),
        "action_names": list(ACTION_NAMES),
        "min_action_support": TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT,
        "risk_threshold": TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD,
        "advantage_threshold": TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD,
        "risk_penalty": TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY,
        "actor_loss_weight": TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_LOSS_WEIGHT,
        "actor_finetune_epochs": TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EPOCHS,
        "advantage_mean": _round(advantage_mean),
        "advantage_scale": _round(advantage_scale),
        "action_support_counts": {
            action: int(action_support_counts[index])
            for index, action in enumerate(ACTION_NAMES)
        },
        "action_component_support_counts": {
            action: {
                component: int(action_component_support_counts[action_idx][component_idx])
                for component_idx, component in enumerate(VIABILITY_COMPONENT_NAMES)
            }
            for action_idx, action in enumerate(ACTION_NAMES)
        },
        "action_component_support_matrix": action_component_support_counts,
        "action_component_bias": {
            action: {
                component: _round(bias_matrix[action_idx][component_idx])
                for component_idx, component in enumerate(VIABILITY_COMPONENT_NAMES)
            }
            for action_idx, action in enumerate(ACTION_NAMES)
        },
        "action_component_bias_matrix": _round_matrix(bias_matrix),
        **aggregate_metrics,
        "by_action": {
            action: _calibrated_supported_metric_report(
                raw_by_action[action],
                calibrated_by_action[action],
                target_by_action[action],
            )
            for action in ACTION_NAMES
        },
        "by_component": {
            component: _calibrated_supported_metric_report(
                raw_by_component[component],
                calibrated_by_component[component],
                target_by_component[component],
            )
            for component in VIABILITY_COMPONENT_NAMES
        },
    }


def _contextual_supported_context(
    records: tuple[dict[str, object], ...],
    calibration_records: tuple[dict[str, object], ...],
    *,
    action_index: dict[str, int],
) -> dict[str, object]:
    calibration_transitions = build_trajectory_transitions(calibration_records)
    context_action_counts: dict[str, Counter[str]] = {}
    context_totals: Counter[str] = Counter()
    for record, transition in zip(
        calibration_records,
        calibration_transitions,
        strict=True,
    ):
        for key in _safe_feature_keys_from_record(record):
            context_action_counts.setdefault(key, Counter())[transition.action] += 1
            context_totals[key] += 1

    row_action_support_matrix: list[list[int]] = []
    row_action_probability_matrix: list[list[float]] = []
    row_context_labels: list[str] = []
    fallback_depth_counts: Counter[str] = Counter()
    by_action_supported_counts: Counter[str] = Counter()
    by_action_probability_totals: Counter[str] = Counter()
    supported_row_action_count = 0
    total_row_action_count = 0
    top_context_counts: Counter[str] = Counter()
    for record in records:
        keys = _safe_feature_keys_from_record(record)
        row_context = _first_supported_context_key(keys, context_totals)
        row_context_labels.append(row_context or "<unsupported>")
        if row_context is not None:
            top_context_counts[row_context] += 1
        support_row: list[int] = []
        probability_row: list[float] = []
        for action in ACTION_NAMES:
            selected_key, support_count, probability, depth = (
                _context_action_support(
                    keys,
                    action,
                    context_action_counts,
                    context_totals,
                )
            )
            support_row.append(support_count)
            probability_row.append(probability)
            total_row_action_count += 1
            if selected_key is not None:
                supported_row_action_count += 1
                by_action_supported_counts[action] += 1
                by_action_probability_totals[action] += probability
                fallback_depth_counts[str(depth)] += 1
            else:
                fallback_depth_counts["unsupported"] += 1
        row_action_support_matrix.append(support_row)
        row_action_probability_matrix.append(probability_row)

    supported_rate = (
        supported_row_action_count / float(total_row_action_count)
        if total_row_action_count
        else 0.0
    )
    return {
        "row_action_support_matrix": row_action_support_matrix,
        "row_action_probability_matrix": row_action_probability_matrix,
        "row_context_labels": tuple(row_context_labels),
        "report": {
            "schema_version": "mind_contextual_behavior_support_v1",
            "support_policy": TORCH_IQL_CONTEXTUAL_SUPPORT_POLICY,
            "behavior_proximity_policy": TORCH_IQL_BEHAVIOR_PROXIMITY_POLICY,
            "calibration_record_count": len(calibration_records),
            "calibration_context_count": len(context_totals),
            "training_record_count": len(records),
            "min_context_action_support": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT
            ),
            "min_behavior_probability": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_BEHAVIOR_PROBABILITY
            ),
            "family_conversion_ratio": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_FAMILY_CONVERSION_RATIO
            ),
            "resource_conversion_min_probability": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_RESOURCE_CONVERSION_MIN_PROBABILITY
            ),
            "max_target_action_expansion_ratio": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_TARGET_ACTION_EXPANSION_RATIO
            ),
            "max_family_conversion_rate": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_FAMILY_CONVERSION_RATE
            ),
            "max_movement_stay_resource_conversion_rate": (
                TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_MOVEMENT_STAY_RESOURCE_RATE
            ),
            "fallback_requires_sparse_context": True,
            "supported_row_action_count": supported_row_action_count,
            "supported_row_action_rate": _round(supported_rate),
            "fallback_depth_counts": _sorted_counts(fallback_depth_counts),
            "top_context_counts": _top_counter(top_context_counts, limit=12),
            "by_action": {
                action: {
                    "supported_row_count": int(by_action_supported_counts[action]),
                    "supported_row_rate": _round(
                        by_action_supported_counts[action] / float(max(len(records), 1))
                    ),
                    "mean_behavior_probability": _round(
                        by_action_probability_totals[action]
                        / float(max(by_action_supported_counts[action], 1))
                    ),
                }
                for action in ACTION_NAMES
            },
        },
    }


def _context_action_support(
    keys: tuple[str, ...],
    action: str,
    context_action_counts: dict[str, Counter[str]],
    context_totals: Counter[str],
) -> tuple[str | None, int, float, int]:
    for depth, key in enumerate(keys):
        action_counts = context_action_counts.get(key)
        if action_counts is None:
            continue
        total = int(context_totals[key])
        if total < TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT:
            continue
        support_count = int(action_counts[action])
        if support_count < TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT:
            return None, support_count, support_count / float(max(total, 1)), depth
        probability = support_count / float(max(total, 1))
        return key, support_count, probability, depth
    return None, 0, 0.0, -1


def _first_supported_context_key(
    keys: tuple[str, ...],
    context_totals: Counter[str],
) -> str | None:
    for key in keys:
        if context_totals[key] >= TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT:
            return key
    return None


def _safe_feature_keys_from_record(record: dict[str, object]) -> tuple[str, ...]:
    try:
        return feature_keys_from_record(record)
    except ValueError:
        return ()


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _top_counter(counter: Counter[str], *, limit: int) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in counter.most_common(limit)
    }


def _calibration_bias(scores: list[float], targets: list[bool]) -> float:
    if not scores:
        return 0.0
    score_mean = sum(scores) / float(len(scores))
    target_rate = sum(1 for target in targets if target) / float(len(targets))
    return _clamp(target_rate - score_mean, -1.0, 1.0)


def _calibrated_supported_metric_report(
    raw_scores: list[float],
    calibrated_scores: list[float],
    targets: list[bool],
) -> dict[str, object]:
    observed_count = len(targets)
    positive_count = sum(1 for target in targets if target)
    buckets = _calibrated_supported_reliability_buckets(
        calibrated_scores,
        targets,
    )
    raw_buckets = _calibrated_supported_reliability_buckets(raw_scores, targets)
    return {
        "observed_count": observed_count,
        "constraint_risk_count": positive_count,
        "constraint_risk_rate": _optional_rate(positive_count, observed_count),
        "raw_mean_score": _optional_mean(raw_scores),
        "mean_score": _optional_mean(calibrated_scores),
        "raw_risk_brier_score": _brier_score(raw_scores, targets),
        "risk_brier_score": _brier_score(calibrated_scores, targets),
        "raw_risk_auc": _binary_auc_scores(raw_scores, targets),
        "risk_auc": _binary_auc_scores(calibrated_scores, targets),
        "raw_risk_ece": _ece_from_buckets(raw_buckets, observed_count),
        "risk_ece": _ece_from_buckets(buckets, observed_count),
        "raw_reliability_buckets": raw_buckets,
        "reliability_buckets": buckets,
    }


def _calibrated_supported_reliability_buckets(
    scores: list[float],
    targets: list[bool],
) -> dict[str, dict[str, object]]:
    bucket_ranges = (
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    )
    buckets: dict[str, dict[str, object]] = {
        _risk_bucket_label(lower, upper): {
            "observed_count": 0,
            "mean_score": None,
            "constraint_risk_rate": None,
            "absolute_calibration_error": None,
        }
        for lower, upper in bucket_ranges
    }
    score_totals: dict[str, float] = {key: 0.0 for key in buckets}
    positive_counts: dict[str, int] = {key: 0 for key in buckets}
    for score, target in zip(scores, targets, strict=True):
        key = _risk_bucket_for_score(score)
        buckets[key]["observed_count"] = int(buckets[key]["observed_count"]) + 1
        score_totals[key] += float(score)
        if target:
            positive_counts[key] += 1
    for key, bucket in buckets.items():
        count = int(bucket["observed_count"])
        if count <= 0:
            continue
        mean_score = score_totals[key] / float(count)
        risk_rate = positive_counts[key] / float(count)
        bucket["mean_score"] = _round(mean_score)
        bucket["constraint_risk_rate"] = _round(risk_rate)
        bucket["absolute_calibration_error"] = _round(abs(mean_score - risk_rate))
    return buckets


def _risk_bucket_for_score(score: float) -> str:
    value = _clamp(float(score), 0.0, 1.0)
    if value < 0.2:
        return "0.00-0.20"
    if value < 0.4:
        return "0.20-0.40"
    if value < 0.6:
        return "0.40-0.60"
    if value < 0.8:
        return "0.60-0.80"
    return "0.80-1.00"


def _risk_bucket_label(lower: float, upper: float) -> str:
    return f"{lower:.2f}-{upper:.2f}"


def _ece_from_buckets(
    buckets: dict[str, dict[str, object]],
    observed_count: int,
) -> float | None:
    if observed_count <= 0:
        return None
    total = 0.0
    for bucket in buckets.values():
        count = int(bucket["observed_count"])
        error = bucket["absolute_calibration_error"]
        if count <= 0 or not isinstance(error, (int, float)):
            continue
        total += (float(count) / float(observed_count)) * float(error)
    return _round(total)


def _brier_score(scores: list[float], targets: list[bool]) -> float | None:
    if not scores:
        return None
    return _round(
        sum(
            (score - (1.0 if target else 0.0)) ** 2
            for score, target in zip(scores, targets, strict=True)
        )
        / float(len(scores))
    )


def _optional_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / float(len(values)))


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(float(numerator) / float(denominator))


def _binary_auc_scores(
    scores: list[float],
    targets: list[bool],
) -> float | None:
    positive_count = sum(1 for target in targets if target)
    negative_count = len(targets) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        return None
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (float(item[1]), int(item[0])),
    )
    rank_sum_positive = 0.0
    rank = 1
    position = 0
    while position < len(ranked):
        next_position = position + 1
        while (
            next_position < len(ranked)
            and float(ranked[next_position][1]) == float(ranked[position][1])
        ):
            next_position += 1
        average_rank = (rank + rank + (next_position - position) - 1) / 2.0
        for ranked_index in range(position, next_position):
            original_index = int(ranked[ranked_index][0])
            if targets[original_index]:
                rank_sum_positive += average_rank
        rank += next_position - position
        position = next_position
    auc = (
        rank_sum_positive
        - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)
    return _round(auc)


def _calibrated_supported_actor_loss(
    torch: Any,
    masked_logits: Any,
    action_values: Any,
    state_values: Any,
    action_masks: Any,
    action_viability_logits: Any,
    replay_weights: Any,
    action_risk_bias: Any,
    action_support: Any,
    action_component_support: Any,
    advantage_mean: Any,
    advantage_scale: Any,
) -> tuple[Any, dict[str, object]]:
    calibrated_scores = torch.sigmoid(action_viability_logits.detach())
    calibrated_scores = (
        calibrated_scores + action_risk_bias.view(
            1,
            len(ACTION_NAMES),
            len(VIABILITY_COMPONENT_NAMES),
        )
    ).clamp(min=0.0, max=1.0)
    component_supported = action_component_support.view(
        1,
        len(ACTION_NAMES),
        len(VIABILITY_COMPONENT_NAMES),
    ) >= float(TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT)
    supported_component_scores = torch.where(
        component_supported,
        calibrated_scores,
        torch.zeros_like(calibrated_scores),
    )
    has_supported_component = component_supported.any(dim=2)
    risk_scores = supported_component_scores.max(dim=2).values
    risk_scores = torch.where(
        has_supported_component,
        risk_scores,
        torch.ones_like(risk_scores),
    )
    advantages = (
        action_values.detach()
        - state_values.detach()
        - advantage_mean
    ) / advantage_scale.clamp_min(TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON)
    action_supported = action_support.view(1, len(ACTION_NAMES)) >= float(
        TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT
    )
    supported_legal = action_masks & action_supported
    positive_supported = supported_legal & (
        advantages > TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD
    )
    candidate_mask = positive_supported & (
        risk_scores <= TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD
    )
    selected_rows = candidate_mask.any(dim=1)
    target_scores = (
        advantages - TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY * risk_scores
    )
    target_scores = target_scores.masked_fill(
        ~candidate_mask,
        torch.finfo(target_scores.dtype).min,
    )
    target_actions = target_scores.argmax(dim=1)
    active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
    if int(active_rows.numel()) <= 0:
        return masked_logits.new_zeros(()), _calibrated_supported_extraction_stats(
            torch,
            action_masks,
            supported_legal,
            positive_supported,
            candidate_mask,
            target_actions,
            risk_scores,
            advantages,
            action_support,
        )
    losses = torch.nn.functional.cross_entropy(
        masked_logits[active_rows],
        target_actions[active_rows],
        reduction="none",
    )
    active_weights = replay_weights[active_rows]
    loss = (
        (losses * active_weights).sum()
        / active_weights.sum().clamp_min(1e-9)
    )
    return loss, _calibrated_supported_extraction_stats(
        torch,
        action_masks,
        supported_legal,
        positive_supported,
        candidate_mask,
        target_actions,
        risk_scores,
        advantages,
        action_support,
    )


def _contextual_behavior_supported_actor_targets(
    torch: Any,
    action_values: Any,
    state_values: Any,
    action_masks: Any,
    action_viability_logits: Any,
    y: Any,
    action_risk_bias: Any,
    context_action_support: Any,
    context_action_probability: Any,
    action_family: Any,
    advantage_mean: Any,
    advantage_scale: Any,
    row_context_labels: tuple[str, ...],
) -> tuple[Any, Any, dict[str, object]]:
    calibrated_scores = torch.sigmoid(action_viability_logits.detach())
    calibrated_scores = (
        calibrated_scores + action_risk_bias.view(
            1,
            len(ACTION_NAMES),
            len(VIABILITY_COMPONENT_NAMES),
        )
    ).clamp(min=0.0, max=1.0)
    risk_component_indices = [
        index
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
        if component != VIABILITY_SUPPRESSION_COMPONENT
    ]
    risk_scores = torch.stack(
        [calibrated_scores[:, :, index] for index in risk_component_indices],
        dim=2,
    ).max(dim=2).values
    advantages = (
        action_values.detach()
        - state_values.detach()
        - advantage_mean
    ) / advantage_scale.clamp_min(TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON)
    context_supported = context_action_support >= float(
        TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT
    )
    behavior_probability_supported = context_action_probability >= float(
        TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_BEHAVIOR_PROBABILITY
    )
    supported_legal = (
        action_masks
        & context_supported
        & behavior_probability_supported
    )
    positive_supported = supported_legal & (
        advantages > TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD
    )
    low_risk_supported = positive_supported & (
        risk_scores <= TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD
    )
    action_families = action_family.view(1, len(ACTION_NAMES))
    logged_families = action_family[y].view(-1, 1)
    same_family = action_families == logged_families
    logged_behavior_probability = context_action_probability.gather(
        1,
        y.unsqueeze(1),
    ).clamp_min(1e-6)
    family_conversion_allowed = same_family | (
        context_action_probability
        >= logged_behavior_probability
        * TORCH_IQL_CONTEXTUAL_SUPPORTED_FAMILY_CONVERSION_RATIO
    )
    source_movement_or_stay = (
        (logged_families == _action_family_index("move_north"))
        | (logged_families == _action_family_index("stay"))
    )
    target_resource = action_families == _action_family_index("eat")
    resource_conversion_allowed = (~(source_movement_or_stay & target_resource)) | (
        context_action_probability
        >= TORCH_IQL_CONTEXTUAL_SUPPORTED_RESOURCE_CONVERSION_MIN_PROBABILITY
    )
    behavior_candidate = (
        low_risk_supported
        & family_conversion_allowed
        & resource_conversion_allowed
    )
    target_scores = (
        advantages - TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY * risk_scores
    )
    target_scores = target_scores.masked_fill(
        ~behavior_candidate,
        torch.finfo(target_scores.dtype).min,
    )
    behavior_candidate, target_actions, behavior_cap_masks = (
        _apply_behavior_proximity_caps(
            torch,
            y,
            behavior_candidate,
            target_scores,
            action_family,
        )
    )
    selected_rows = behavior_candidate.any(dim=1)
    target_scores = target_scores.masked_fill(
        ~behavior_candidate,
        torch.finfo(target_scores.dtype).min,
    )
    target_actions = target_scores.argmax(dim=1)
    active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
    stats = _contextual_supported_extraction_stats(
        torch,
        y,
        action_masks,
        supported_legal,
        positive_supported,
        low_risk_supported,
        behavior_candidate,
        target_actions,
        risk_scores,
        advantages,
        context_action_support,
        context_action_probability,
        action_family,
        row_context_labels,
        behavior_cap_masks,
    )
    return active_rows, target_actions, stats


def _contextual_behavior_supported_actor_loss(
    torch: Any,
    masked_logits: Any,
    action_values: Any,
    state_values: Any,
    action_masks: Any,
    action_viability_logits: Any,
    replay_weights: Any,
    y: Any,
    action_risk_bias: Any,
    context_action_support: Any,
    context_action_probability: Any,
    action_family: Any,
    advantage_mean: Any,
    advantage_scale: Any,
    row_context_labels: tuple[str, ...],
) -> tuple[Any, dict[str, object]]:
    calibrated_scores = torch.sigmoid(action_viability_logits.detach())
    calibrated_scores = (
        calibrated_scores + action_risk_bias.view(
            1,
            len(ACTION_NAMES),
            len(VIABILITY_COMPONENT_NAMES),
        )
    ).clamp(min=0.0, max=1.0)
    risk_component_indices = [
        index
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
        if component != VIABILITY_SUPPRESSION_COMPONENT
    ]
    risk_scores = torch.stack(
        [calibrated_scores[:, :, index] for index in risk_component_indices],
        dim=2,
    ).max(dim=2).values
    advantages = (
        action_values.detach()
        - state_values.detach()
        - advantage_mean
    ) / advantage_scale.clamp_min(TORCH_IQL_CALIBRATED_ADVANTAGE_SCALE_EPSILON)
    context_supported = context_action_support >= float(
        TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT
    )
    behavior_probability_supported = context_action_probability >= float(
        TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_BEHAVIOR_PROBABILITY
    )
    supported_legal = (
        action_masks
        & context_supported
        & behavior_probability_supported
    )
    positive_supported = supported_legal & (
        advantages > TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD
    )
    low_risk_supported = positive_supported & (
        risk_scores <= TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD
    )
    action_families = action_family.view(1, len(ACTION_NAMES))
    logged_families = action_family[y].view(-1, 1)
    same_family = action_families == logged_families
    logged_behavior_probability = context_action_probability.gather(
        1,
        y.unsqueeze(1),
    ).clamp_min(1e-6)
    family_conversion_allowed = same_family | (
        context_action_probability
        >= logged_behavior_probability
        * TORCH_IQL_CONTEXTUAL_SUPPORTED_FAMILY_CONVERSION_RATIO
    )
    source_movement_or_stay = (
        (logged_families == _action_family_index("move_north"))
        | (logged_families == _action_family_index("stay"))
    )
    target_resource = action_families == _action_family_index("eat")
    resource_conversion_allowed = (~(source_movement_or_stay & target_resource)) | (
        context_action_probability
        >= TORCH_IQL_CONTEXTUAL_SUPPORTED_RESOURCE_CONVERSION_MIN_PROBABILITY
    )
    behavior_candidate = (
        low_risk_supported
        & family_conversion_allowed
        & resource_conversion_allowed
    )
    target_scores = (
        advantages - TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY * risk_scores
    )
    target_scores = target_scores.masked_fill(
        ~behavior_candidate,
        torch.finfo(target_scores.dtype).min,
    )
    behavior_candidate, target_actions, behavior_cap_masks = (
        _apply_behavior_proximity_caps(
            torch,
            y,
            behavior_candidate,
            target_scores,
            action_family,
        )
    )
    selected_rows = behavior_candidate.any(dim=1)
    target_scores = target_scores.masked_fill(
        ~behavior_candidate,
        torch.finfo(target_scores.dtype).min,
    )
    target_actions = target_scores.argmax(dim=1)
    active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
    if int(active_rows.numel()) <= 0:
        return masked_logits.new_zeros(()), _contextual_supported_extraction_stats(
            torch,
            y,
            action_masks,
            supported_legal,
            positive_supported,
            low_risk_supported,
            behavior_candidate,
            target_actions,
            risk_scores,
            advantages,
            context_action_support,
            context_action_probability,
            action_family,
            row_context_labels,
            behavior_cap_masks,
        )
    losses = torch.nn.functional.cross_entropy(
        masked_logits[active_rows],
        target_actions[active_rows],
        reduction="none",
    )
    active_weights = replay_weights[active_rows]
    loss = (
        (losses * active_weights).sum()
        / active_weights.sum().clamp_min(1e-9)
    )
    return loss, _contextual_supported_extraction_stats(
        torch,
        y,
        action_masks,
        supported_legal,
        positive_supported,
        low_risk_supported,
        behavior_candidate,
        target_actions,
        risk_scores,
        advantages,
        context_action_support,
        context_action_probability,
        action_family,
        row_context_labels,
        behavior_cap_masks,
    )


def _contextual_behavior_prior_actor_loss(
    torch: Any,
    masked_logits: Any,
    action_masks: Any,
    context_action_probability: Any,
    replay_weights: Any,
    *,
    logged_actions: Any | None = None,
) -> tuple[Any, dict[str, object]]:
    legal_prior = (
        context_action_probability.to(dtype=masked_logits.dtype)
        * action_masks.to(dtype=masked_logits.dtype)
    )
    prior_mass = legal_prior.sum(dim=1)
    active_rows = prior_mass > TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_MIN_MASS
    active_count = int(active_rows.sum().detach().cpu().item())
    row_count = int(masked_logits.shape[0])
    if active_count <= 0:
        return (
            masked_logits.new_zeros(()),
            _empty_contextual_behavior_prior_stats(row_count=row_count),
        )

    active_prior = legal_prior[active_rows] / prior_mass[
        active_rows
    ].unsqueeze(1).clamp_min(TORCH_IQL_CONTEXTUAL_BEHAVIOR_PRIOR_MIN_MASS)
    log_policy = torch.nn.functional.log_softmax(
        masked_logits[active_rows],
        dim=1,
    )
    per_row_loss = -(active_prior * log_policy).sum(dim=1)
    active_weights = replay_weights[active_rows]
    loss = (
        (per_row_loss * active_weights).sum()
        / active_weights.sum().clamp_min(1e-9)
    )

    entropy = -(
        active_prior * active_prior.clamp_min(1e-9).log()
    ).sum(dim=1)
    max_probability = active_prior.max(dim=1).values
    logged_probability_mean: float | None = None
    logged_top1_rate: float | None = None
    if logged_actions is not None:
        active_logged_actions = logged_actions[active_rows]
        logged_probabilities = active_prior.gather(
            1,
            active_logged_actions.unsqueeze(1),
        ).squeeze(1)
        top_prior_actions = active_prior.argmax(dim=1)
        logged_probability_mean = float(
            logged_probabilities.mean().detach().cpu().item()
        )
        logged_top1_rate = float(
            (top_prior_actions == active_logged_actions)
            .to(dtype=masked_logits.dtype)
            .mean()
            .detach()
            .cpu()
            .item()
        )

    return loss, {
        "schema_version": (
            "mind_contextual_behavior_prior_actor_regularization_v1"
        ),
        "row_count": row_count,
        "active_row_count": active_count,
        "active_row_rate": _round(active_count / float(max(row_count, 1))),
        "behavior_prior_entropy_mean": _round(
            float(entropy.mean().detach().cpu().item())
        ),
        "behavior_prior_max_probability_mean": _round(
            float(max_probability.mean().detach().cpu().item())
        ),
        "behavior_prior_logged_probability_mean": (
            None
            if logged_probability_mean is None
            else _round(logged_probability_mean)
        ),
        "behavior_prior_logged_top1_rate": (
            None if logged_top1_rate is None else _round(logged_top1_rate)
        ),
    }


def _action_distribution_actor_loss(
    torch: Any,
    masked_logits: Any,
    action_masks: Any,
    labels: Any,
    replay_weights: Any,
    *,
    temperature: float = TORCH_IQL_ACTION_DISTRIBUTION_TEMPERATURE,
) -> tuple[Any, dict[str, object]]:
    row_count = int(labels.shape[0])
    if row_count <= 0:
        return (
            masked_logits.new_zeros(()),
            _empty_action_distribution_stats(temperature=temperature),
        )
    weights = replay_weights.to(dtype=masked_logits.dtype).clamp_min(0.0)
    weight_total = weights.sum().clamp_min(1e-9)
    probabilities = torch.nn.functional.softmax(
        masked_logits / float(temperature),
        dim=1,
    )
    predicted_distribution = (
        probabilities * weights.view(-1, 1)
    ).sum(dim=0) / weight_total
    target_counts = torch.zeros(
        len(ACTION_NAMES),
        dtype=masked_logits.dtype,
        device=masked_logits.device,
    )
    target_counts.scatter_add_(0, labels, weights)
    target_distribution = target_counts / weight_total
    legal_action_mass = action_masks.to(dtype=masked_logits.dtype).sum(dim=0) > 0.0
    active_actions = (target_counts > 0.0) | legal_action_mass
    safe_target = target_distribution.clamp_min(1e-9)
    safe_predicted = predicted_distribution.clamp_min(1e-9)
    kl_terms = target_distribution * (safe_target.log() - safe_predicted.log())
    loss = kl_terms[active_actions].sum()
    tvd = 0.5 * torch.abs(
        target_distribution - predicted_distribution
    )[active_actions].sum()
    return loss, {
        "schema_version": "mind_action_distribution_actor_regularization_v1",
        "policy": TORCH_IQL_ACTION_DISTRIBUTION_REGULARIZATION_POLICY,
        "row_count": row_count,
        "active_action_count": int(active_actions.sum().detach().cpu().item()),
        "action_distribution_temperature": _round(float(temperature)),
        "action_distribution_kl": _round(float(loss.detach().cpu().item())),
        "action_distribution_tvd": _round(float(tvd.detach().cpu().item())),
        "target_distribution": {
            action: _round(
                float(target_distribution[index].detach().cpu().item())
            )
            for index, action in enumerate(ACTION_NAMES)
        },
        "predicted_distribution": {
            action: _round(
                float(predicted_distribution[index].detach().cpu().item())
            )
            for index, action in enumerate(ACTION_NAMES)
        },
    }


def _apply_rollout_state_action_bias_calibration(
    torch: Any,
    model: Any,
    records: tuple[dict[str, object], ...],
    *,
    action_index: Mapping[str, int],
    device: Any,
    max_action_share: float,
    bias_step: float,
    max_bias_delta: float,
) -> dict[str, object]:
    tensors = _rollout_state_action_tensors(
        torch,
        records,
        action_index=action_index,
        device=device,
    )
    if tensors is None:
        return _empty_rollout_state_action_calibration_report()
    x, action_masks, labels = tensors
    original_bias = model.actor.bias.detach().clone()
    before = _rollout_state_action_top1_report_from_tensors(
        torch,
        model,
        x,
        action_masks,
        labels,
    )
    iteration_count = 0
    converged = bool(before["dominant_action_share"] <= max_action_share)
    with torch.no_grad():
        for iteration in range(TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_ITERATIONS):
            current = _rollout_state_action_top1_report_from_tensors(
                torch,
                model,
                x,
                action_masks,
                labels,
            )
            over_cap = [
                action
                for action, share in current["top1_action_shares"].items()
                if float(share) > max_action_share
            ]
            if not over_cap:
                converged = True
                iteration_count = iteration
                break
            changed = False
            for action in over_cap:
                action_idx = int(action_index[action])
                current_delta = float(
                    (model.actor.bias[action_idx] - original_bias[action_idx])
                    .detach()
                    .cpu()
                    .item()
                )
                remaining = max_bias_delta + current_delta
                if remaining <= 0.0:
                    continue
                model.actor.bias[action_idx] -= min(float(bias_step), remaining)
                changed = True
            iteration_count = iteration + 1
            if not changed:
                break
        after = _rollout_state_action_top1_report_from_tensors(
            torch,
            model,
            x,
            action_masks,
            labels,
        )
    bias_delta = {
        action: _round(
            float(
                (
                    model.actor.bias[int(action_index[action])]
                    - original_bias[int(action_index[action])]
                )
                .detach()
                .cpu()
                .item()
            )
        )
        for action in ACTION_NAMES
    }
    return {
        "schema_version": "mind_rollout_state_action_calibration_v1",
        "policy": TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY,
        "calibration_record_count": len(records),
        "calibration_transition_count": int(before["row_count"]),
        "max_action_share": _round(max_action_share),
        "bias_step": _round(bias_step),
        "max_bias_delta": _round(max_bias_delta),
        "max_iterations": TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_ITERATIONS,
        "iteration_count": iteration_count,
        "converged": bool(after["dominant_action_share"] <= max_action_share),
        "initially_converged": converged and iteration_count == 0,
        "action_bias_delta": bias_delta,
        "before": before,
        "after": after,
    }


def _rollout_state_action_top1_report(
    torch: Any,
    model: Any,
    records: tuple[dict[str, object], ...],
    *,
    action_index: Mapping[str, int],
    device: Any,
) -> dict[str, object]:
    tensors = _rollout_state_action_tensors(
        torch,
        records,
        action_index=action_index,
        device=device,
    )
    if tensors is None:
        return _empty_rollout_state_action_top1_report()
    x, action_masks, labels = tensors
    return _rollout_state_action_top1_report_from_tensors(
        torch,
        model,
        x,
        action_masks,
        labels,
    )


def _rollout_state_action_tensors(
    torch: Any,
    records: tuple[dict[str, object], ...],
    *,
    action_index: Mapping[str, int],
    device: Any,
) -> tuple[Any, Any, Any] | None:
    transitions = build_trajectory_transitions(records)
    if not transitions:
        return None
    features = [
        decode_observation_input(transition.observation_input)
        for transition in transitions
    ]
    action_masks = [
        _action_mask_values(transition.action_mask)
        for transition in transitions
    ]
    labels = [int(action_index[transition.action]) for transition in transitions]
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(action_masks, dtype=torch.bool, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _rollout_state_action_top1_report_from_tensors(
    torch: Any,
    model: Any,
    x: Any,
    action_masks: Any,
    labels: Any,
) -> dict[str, object]:
    row_count = int(labels.shape[0])
    if row_count <= 0:
        return _empty_rollout_state_action_top1_report()
    with torch.no_grad():
        hidden = torch.tanh(model.hidden(x))
        masked_logits = _masked_logits(torch, model.actor(hidden), action_masks)
        has_legal_action = action_masks.any(dim=1)
        active_rows = torch.nonzero(has_legal_action, as_tuple=False).squeeze(1)
        active_count = int(active_rows.numel())
        if active_count <= 0:
            return _empty_rollout_state_action_top1_report(row_count=row_count)
        top_actions = masked_logits[active_rows].argmax(dim=1)
        active_labels = labels[active_rows]
        top_counts = {action: 0 for action in ACTION_NAMES}
        logged_counts = {action: 0 for action in ACTION_NAMES}
        for action_idx in top_actions.detach().cpu().tolist():
            top_counts[ACTION_NAMES[int(action_idx)]] += 1
        for action_idx in active_labels.detach().cpu().tolist():
            logged_counts[ACTION_NAMES[int(action_idx)]] += 1
        top_shares = {
            action: _round(count / float(active_count))
            for action, count in top_counts.items()
        }
        logged_shares = {
            action: _round(count / float(active_count))
            for action, count in logged_counts.items()
        }
        dominant_action = max(
            ACTION_NAMES,
            key=lambda action: (top_counts[action], action),
        )
        match_rate = float(
            (top_actions == active_labels)
            .to(dtype=masked_logits.dtype)
            .mean()
            .detach()
            .cpu()
            .item()
        )
    return {
        "schema_version": "mind_rollout_state_action_top1_report_v1",
        "policy": TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY,
        "row_count": row_count,
        "active_row_count": active_count,
        "top1_action_counts": top_counts,
        "top1_action_shares": top_shares,
        "logged_action_counts": logged_counts,
        "logged_action_shares": logged_shares,
        "dominant_action": dominant_action,
        "dominant_action_share": top_shares[dominant_action],
        "top1_logged_match_rate": _round(match_rate),
        "top1_distribution_tvd_from_logged": _round(
            _action_distribution_tvd(top_counts, logged_counts)
        ),
    }


def _apply_behavior_proximity_caps(
    torch: Any,
    y: Any,
    candidate_mask: Any,
    target_scores: Any,
    action_family: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    selected_rows = candidate_mask.any(dim=1)
    target_actions = target_scores.argmax(dim=1)
    row_count = int(candidate_mask.shape[0])
    false_rows = torch.zeros(row_count, dtype=torch.bool, device=target_scores.device)
    cap_masks = {
        "target_action_expansion": false_rows.clone(),
        "family_conversion": false_rows.clone(),
        "movement_stay_resource_conversion": false_rows.clone(),
    }
    if int(torch.nonzero(selected_rows, as_tuple=False).numel()) <= 0:
        return candidate_mask, target_actions, cap_masks

    capped_selected_rows = selected_rows.clone()
    for _ in range(row_count + 1):
        changed = False
        changed = (
            _reject_target_action_expansion_rows(
                torch,
                capped_selected_rows,
                target_actions,
                target_scores,
                y,
                cap_masks["target_action_expansion"],
            )
            or changed
        )
        changed = (
            _reject_family_conversion_rows(
                torch,
                capped_selected_rows,
                target_actions,
                target_scores,
                y,
                action_family,
                cap_masks["family_conversion"],
                max_rate=TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_FAMILY_CONVERSION_RATE,
                movement_stay_resource_only=False,
            )
            or changed
        )
        changed = (
            _reject_family_conversion_rows(
                torch,
                capped_selected_rows,
                target_actions,
                target_scores,
                y,
                action_family,
                cap_masks["movement_stay_resource_conversion"],
                max_rate=(
                    TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_MOVEMENT_STAY_RESOURCE_RATE
                ),
                movement_stay_resource_only=True,
            )
            or changed
        )
        if not changed:
            break

    capped_candidate_mask = candidate_mask.clone()
    capped_candidate_mask[~capped_selected_rows] = False
    capped_target_scores = target_scores.masked_fill(
        ~capped_candidate_mask,
        torch.finfo(target_scores.dtype).min,
    )
    return capped_candidate_mask, capped_target_scores.argmax(dim=1), cap_masks


def _reject_target_action_expansion_rows(
    torch: Any,
    selected_rows: Any,
    target_actions: Any,
    target_scores: Any,
    y: Any,
    cap_mask: Any,
) -> bool:
    active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
    if int(active_rows.numel()) <= 0:
        return False
    selected_targets = target_actions[active_rows]
    selected_logged = y[active_rows]
    changed = False
    for action_idx in range(len(ACTION_NAMES)):
        target_rows = active_rows[selected_targets == action_idx]
        target_count = int(target_rows.numel())
        if target_count <= 0:
            continue
        logged_count = int((selected_logged == action_idx).sum().detach().cpu().item())
        allowed_count = int(
            math.floor(
                logged_count
                * TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_TARGET_ACTION_EXPANSION_RATIO
            )
        )
        if target_count <= allowed_count:
            continue
        reject_count = target_count - allowed_count
        conversion_rows = target_rows[y[target_rows] != action_idx]
        if int(conversion_rows.numel()) < reject_count:
            candidate_rows = target_rows
        else:
            candidate_rows = conversion_rows
        candidate_scores = target_scores[candidate_rows, action_idx]
        order = torch.argsort(candidate_scores, descending=True)
        rejected_rows = candidate_rows[order[-reject_count:]]
        if int(rejected_rows.numel()) <= 0:
            continue
        cap_mask[rejected_rows] = True
        selected_rows[rejected_rows] = False
        changed = True
    return changed


def _reject_family_conversion_rows(
    torch: Any,
    selected_rows: Any,
    target_actions: Any,
    target_scores: Any,
    y: Any,
    action_family: Any,
    cap_mask: Any,
    *,
    max_rate: float,
    movement_stay_resource_only: bool,
) -> bool:
    active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
    selected_count = int(active_rows.numel())
    if selected_count <= 0:
        return False
    selected_targets = target_actions[active_rows]
    source_families = action_family[y[active_rows]]
    target_families = action_family[selected_targets]
    conversion_mask = source_families != target_families
    if movement_stay_resource_only:
        movement_index = _action_family_index("move_north")
        stay_index = _action_family_index("stay")
        resource_index = _action_family_index("eat")
        conversion_mask = conversion_mask & (
            ((source_families == movement_index) | (source_families == stay_index))
            & (target_families == resource_index)
        )
    conversion_rows = active_rows[conversion_mask]
    conversion_count = int(conversion_rows.numel())
    allowed_count = int(math.floor(selected_count * max_rate))
    if conversion_count <= allowed_count:
        return False
    reject_count = conversion_count - allowed_count
    conversion_targets = target_actions[conversion_rows]
    conversion_scores = target_scores[conversion_rows, conversion_targets]
    order = torch.argsort(conversion_scores, descending=True)
    rejected_rows = conversion_rows[order[-reject_count:]]
    if int(rejected_rows.numel()) <= 0:
        return False
    cap_mask[rejected_rows] = True
    selected_rows[rejected_rows] = False
    return True


def _contextual_supported_extraction_stats(
    torch: Any,
    y: Any,
    action_masks: Any,
    supported_legal: Any,
    positive_supported: Any,
    low_risk_supported: Any,
    candidate_mask: Any,
    target_actions: Any,
    risk_scores: Any,
    advantages: Any,
    context_action_support: Any,
    context_action_probability: Any,
    action_family: Any,
    row_context_labels: tuple[str, ...],
    behavior_cap_masks: dict[str, Any],
) -> dict[str, object]:
    with torch.no_grad():
        row_count = int(action_masks.shape[0])
        selected_rows = candidate_mask.any(dim=1)
        selected_count = int(selected_rows.sum().detach().cpu().item())
        rejected_count = row_count - selected_count
        has_legal = action_masks.any(dim=1)
        has_supported = supported_legal.any(dim=1)
        has_positive = positive_supported.any(dim=1)
        has_low_risk = low_risk_supported.any(dim=1)
        no_legal = ~has_legal
        low_context_support = (~selected_rows) & has_legal & (~has_supported)
        negative_advantage = (
            (~selected_rows) & has_legal & has_supported & (~has_positive)
        )
        high_risk = (
            (~selected_rows)
            & has_legal
            & has_supported
            & has_positive
            & (~has_low_risk)
        )
        behavior_proximity = (
            (~selected_rows)
            & has_legal
            & has_supported
            & has_positive
            & has_low_risk
        )
        accounted = (
            no_legal
            | low_context_support
            | negative_advantage
            | high_risk
            | behavior_proximity
        )
        no_legal_candidate = no_legal | ((~selected_rows) & (~accounted))
        target_action_counts = {action: 0 for action in ACTION_NAMES}
        logged_action_counts = {action: 0 for action in ACTION_NAMES}
        target_risk_mean = 0.0
        target_advantage_mean = 0.0
        target_support_mean = 0.0
        target_behavior_probability_mean = 0.0
        family_conversion_counts: dict[str, dict[str, int]] = {
            family: {target_family: 0 for target_family in _action_family_names()}
            for family in _action_family_names()
        }
        movement_or_stay_to_resource_count = 0
        if selected_count > 0:
            active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
            selected_targets = target_actions[active_rows]
            selected_logged = y[active_rows]
            for logged, target in zip(
                selected_logged.detach().cpu().tolist(),
                selected_targets.detach().cpu().tolist(),
                strict=True,
            ):
                logged_action = ACTION_NAMES[int(logged)]
                target_action = ACTION_NAMES[int(target)]
                logged_action_counts[logged_action] += 1
                target_action_counts[target_action] += 1
                source_family = _action_family_name(logged_action)
                target_family = _action_family_name(target_action)
                family_conversion_counts[source_family][target_family] += 1
                if source_family in {"movement", "stay"} and target_family == "resource":
                    movement_or_stay_to_resource_count += 1
            target_risks = risk_scores[active_rows, selected_targets]
            target_advantages = advantages[active_rows, selected_targets]
            target_supports = context_action_support[active_rows, selected_targets]
            target_behavior_probabilities = context_action_probability[
                active_rows,
                selected_targets,
            ]
            target_risk_mean = float(target_risks.mean().detach().cpu().item())
            target_advantage_mean = float(
                target_advantages.mean().detach().cpu().item()
            )
            target_support_mean = float(target_supports.mean().detach().cpu().item())
            target_behavior_probability_mean = float(
                target_behavior_probabilities.mean().detach().cpu().item()
            )
        rejection_masks = {
            "low_context_support": low_context_support,
            "high_risk": high_risk,
            "negative_advantage": negative_advantage,
            "behavior_proximity": behavior_proximity,
            "no_legal_candidate": no_legal_candidate,
        }
        rejection_reasons = {
            reason: int(mask.sum().detach().cpu().item())
            for reason, mask in rejection_masks.items()
        }
        behavior_cap_counts = {
            reason: int(mask.sum().detach().cpu().item())
            for reason, mask in behavior_cap_masks.items()
        }
        denominator = float(max(row_count, 1))
        selected_denominator = float(max(selected_count, 1))
        return {
            "schema_version": "mind_contextual_behavior_supported_actor_targets_v2",
            "selected_count": selected_count,
            "selected_rate": _round(selected_count / denominator),
            "rejected_count": rejected_count,
            "rejected_rate": _round(rejected_count / denominator),
            "rejection_reasons": rejection_reasons,
            "low_support_count": rejection_reasons["low_context_support"],
            "high_risk_count": rejection_reasons["high_risk"],
            "negative_advantage_count": rejection_reasons["negative_advantage"],
            "behavior_proximity_count": rejection_reasons["behavior_proximity"],
            "behavior_proximity_cap_counts": behavior_cap_counts,
            "target_action_expansion_cap_count": (
                behavior_cap_counts["target_action_expansion"]
            ),
            "family_conversion_cap_count": (
                behavior_cap_counts["family_conversion"]
            ),
            "movement_stay_resource_conversion_cap_count": (
                behavior_cap_counts["movement_stay_resource_conversion"]
            ),
            "no_legal_candidate_count": (
                rejection_reasons["no_legal_candidate"]
            ),
            "target_action_counts": target_action_counts,
            "selected_logged_action_counts": logged_action_counts,
            "target_distribution_tvd_from_logged": _round(
                _action_distribution_tvd(
                    target_action_counts,
                    logged_action_counts,
                )
            ),
            "family_conversion_counts": family_conversion_counts,
            "movement_or_stay_to_resource_count": movement_or_stay_to_resource_count,
            "movement_or_stay_to_resource_rate": _round(
                movement_or_stay_to_resource_count / selected_denominator
            ),
            "target_risk_mean": _round(target_risk_mean),
            "target_advantage_mean": _round(target_advantage_mean),
            "target_support_mean": _round(target_support_mean),
            "target_behavior_probability_mean": _round(
                target_behavior_probability_mean
            ),
            "rejection_reasons_by_logged_action": (
                _rejection_reasons_by_logged_action(
                    y,
                    rejection_masks,
                )
            ),
            "rejection_reasons_by_context": _rejection_reasons_by_context(
                row_context_labels,
                rejection_masks,
            ),
        }


def _calibrated_supported_extraction_stats(
    torch: Any,
    action_masks: Any,
    supported_legal: Any,
    positive_supported: Any,
    candidate_mask: Any,
    target_actions: Any,
    risk_scores: Any,
    advantages: Any,
    action_support: Any,
) -> dict[str, object]:
    with torch.no_grad():
        row_count = int(action_masks.shape[0])
        selected_rows = candidate_mask.any(dim=1)
        selected_count = int(selected_rows.sum().detach().cpu().item())
        rejected_count = row_count - selected_count
        has_legal = action_masks.any(dim=1)
        has_supported = supported_legal.any(dim=1)
        has_positive = positive_supported.any(dim=1)
        has_candidate = selected_rows
        no_legal = ~has_legal
        low_support = (~has_candidate) & has_legal & (~has_supported)
        negative_advantage = (
            (~has_candidate) & has_legal & has_supported & (~has_positive)
        )
        high_risk = (
            (~has_candidate)
            & has_legal
            & has_supported
            & has_positive
        )
        accounted = no_legal | low_support | negative_advantage | high_risk
        no_legal_candidate = no_legal | ((~has_candidate) & (~accounted))
        target_action_counts = {action: 0 for action in ACTION_NAMES}
        target_risk_mean = 0.0
        target_advantage_mean = 0.0
        target_support_mean = 0.0
        if selected_count > 0:
            active_rows = torch.nonzero(selected_rows, as_tuple=False).squeeze(1)
            selected_targets = target_actions[active_rows]
            for target in selected_targets.detach().cpu().tolist():
                target_action_counts[ACTION_NAMES[int(target)]] += 1
            row_index = torch.arange(active_rows.numel())
            target_risks = risk_scores[active_rows, selected_targets]
            target_advantages = advantages[active_rows, selected_targets]
            target_supports = action_support[selected_targets]
            target_risk_mean = float(target_risks.mean().detach().cpu().item())
            target_advantage_mean = float(
                target_advantages.mean().detach().cpu().item()
            )
            target_support_mean = float(target_supports.mean().detach().cpu().item())
        denominator = float(max(row_count, 1))
        rejection_reasons = {
            "low_support": int(low_support.sum().detach().cpu().item()),
            "high_risk": int(high_risk.sum().detach().cpu().item()),
            "negative_advantage": int(
                negative_advantage.sum().detach().cpu().item()
            ),
            "no_legal_candidate": int(
                no_legal_candidate.sum().detach().cpu().item()
            ),
        }
        return {
            "schema_version": "mind_calibrated_supported_actor_targets_v1",
            "selected_count": selected_count,
            "selected_rate": _round(selected_count / denominator),
            "rejected_count": rejected_count,
            "rejected_rate": _round(rejected_count / denominator),
            "rejection_reasons": rejection_reasons,
            "low_support_count": rejection_reasons["low_support"],
            "high_risk_count": rejection_reasons["high_risk"],
            "negative_advantage_count": rejection_reasons["negative_advantage"],
            "no_legal_candidate_count": (
                rejection_reasons["no_legal_candidate"]
            ),
            "target_action_counts": target_action_counts,
            "target_risk_mean": _round(target_risk_mean),
            "target_advantage_mean": _round(target_advantage_mean),
            "target_support_mean": _round(target_support_mean),
        }


def _action_family_index(action: str) -> int:
    return _action_family_names().index(_action_family_name(action))


def _action_family_name(action: str) -> str:
    if action in {"eat", "drink"}:
        return "resource"
    if action.startswith("move_"):
        return "movement"
    if action == "stay":
        return "stay"
    if action.startswith("attack_"):
        return "attack"
    return "other"


def _action_family_names() -> tuple[str, ...]:
    return ("resource", "movement", "stay", "attack", "other")


def _action_distribution_tvd(
    target_counts: dict[str, int],
    logged_counts: dict[str, int],
) -> float:
    target_total = float(max(sum(target_counts.values()), 1))
    logged_total = float(max(sum(logged_counts.values()), 1))
    return 0.5 * sum(
        abs(
            float(target_counts[action]) / target_total
            - float(logged_counts[action]) / logged_total
        )
        for action in ACTION_NAMES
    )


def _rejection_reasons_by_logged_action(
    y: Any,
    rejection_masks: dict[str, Any],
) -> dict[str, dict[str, int]]:
    labels = [int(label) for label in y.detach().cpu().tolist()]
    reason_rows = {
        reason: set(int(index) for index in torch_rows.detach().cpu().tolist())
        for reason, mask in rejection_masks.items()
        for torch_rows in [mask.nonzero(as_tuple=False).squeeze(1)]
    }
    return {
        action: {
            reason: int(
                sum(
                    1
                    for row_index, label in enumerate(labels)
                    if label == action_index and row_index in rows
                )
            )
            for reason, rows in reason_rows.items()
        }
        for action_index, action in enumerate(ACTION_NAMES)
    }


def _rejection_reasons_by_context(
    row_context_labels: tuple[str, ...],
    rejection_masks: dict[str, Any],
) -> dict[str, dict[str, int]]:
    if not row_context_labels:
        return {reason: {} for reason in rejection_masks}
    payload: dict[str, dict[str, int]] = {}
    for reason, mask in rejection_masks.items():
        rows = [int(index) for index in mask.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()]
        counts: Counter[str] = Counter()
        for row_index in rows:
            if row_index < len(row_context_labels):
                counts[row_context_labels[row_index]] += 1
        payload[reason] = _top_counter(counts, limit=12)
    return payload


def _empty_calibrated_supported_report() -> dict[str, object]:
    return {
        "schema_version": "mind_calibrated_supported_actor_extraction_v1",
        "policy": TORCH_IQL_CALIBRATED_SUPPORTED_ACTOR_EXTRACTION_POLICY,
        "risk_calibration_policy": (
            TORCH_IQL_CALIBRATED_SUPPORTED_RISK_CALIBRATION_POLICY
        ),
        "advantage_policy": TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_POLICY,
        "calibration_record_count": 0,
        "calibration_transition_count": 0,
        "component_names": list(VIABILITY_COMPONENT_NAMES),
        "action_names": list(ACTION_NAMES),
        "min_action_support": TORCH_IQL_CALIBRATED_SUPPORTED_MIN_ACTION_SUPPORT,
        "risk_threshold": TORCH_IQL_CALIBRATED_SUPPORTED_RISK_THRESHOLD,
        "advantage_threshold": TORCH_IQL_CALIBRATED_SUPPORTED_ADVANTAGE_THRESHOLD,
        "risk_penalty": TORCH_IQL_CALIBRATED_SUPPORTED_RISK_PENALTY,
        "actor_loss_weight": 0.0,
        "actor_finetune_epochs": 0,
        "advantage_mean": 0.0,
        "advantage_scale": 1.0,
        "action_support_counts": {action: 0 for action in ACTION_NAMES},
        "action_component_support_counts": {
            action: {component: 0 for component in VIABILITY_COMPONENT_NAMES}
            for action in ACTION_NAMES
        },
        "action_component_support_matrix": [
            [0 for _component in VIABILITY_COMPONENT_NAMES]
            for _action in ACTION_NAMES
        ],
        "action_component_bias": {
            action: {component: 0.0 for component in VIABILITY_COMPONENT_NAMES}
            for action in ACTION_NAMES
        },
        "action_component_bias_matrix": [
            [0.0 for _component in VIABILITY_COMPONENT_NAMES]
            for _action in ACTION_NAMES
        ],
        **_calibrated_supported_metric_report([], [], []),
        "by_action": {
            action: _calibrated_supported_metric_report([], [], [])
            for action in ACTION_NAMES
        },
        "by_component": {
            component: _calibrated_supported_metric_report([], [], [])
            for component in VIABILITY_COMPONENT_NAMES
        },
    }


def _empty_contextual_supported_report() -> dict[str, object]:
    return {
        "schema_version": "mind_contextual_behavior_support_v1",
        "support_policy": TORCH_IQL_CONTEXTUAL_SUPPORT_POLICY,
        "behavior_proximity_policy": TORCH_IQL_BEHAVIOR_PROXIMITY_POLICY,
        "calibration_record_count": 0,
        "calibration_context_count": 0,
        "training_record_count": 0,
        "min_context_action_support": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_ACTION_SUPPORT
        ),
        "min_behavior_probability": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MIN_BEHAVIOR_PROBABILITY
        ),
        "family_conversion_ratio": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_FAMILY_CONVERSION_RATIO
        ),
        "resource_conversion_min_probability": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_RESOURCE_CONVERSION_MIN_PROBABILITY
        ),
        "max_target_action_expansion_ratio": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_TARGET_ACTION_EXPANSION_RATIO
        ),
        "max_family_conversion_rate": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_FAMILY_CONVERSION_RATE
        ),
        "max_movement_stay_resource_conversion_rate": (
            TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_MOVEMENT_STAY_RESOURCE_RATE
        ),
        "fallback_requires_sparse_context": True,
        "supported_row_action_count": 0,
        "supported_row_action_rate": 0.0,
        "fallback_depth_counts": {},
        "top_context_counts": {},
        "by_action": {
            action: {
                "supported_row_count": 0,
                "supported_row_rate": 0.0,
                "mean_behavior_probability": 0.0,
            }
            for action in ACTION_NAMES
        },
    }


def _empty_calibrated_supported_stats() -> dict[str, object]:
    return {
        "schema_version": "mind_calibrated_supported_actor_targets_v1",
        "selected_count": 0,
        "selected_rate": 0.0,
        "rejected_count": 0,
        "rejected_rate": 0.0,
        "rejection_reasons": {
            "low_support": 0,
            "low_context_support": 0,
            "high_risk": 0,
            "negative_advantage": 0,
            "behavior_proximity": 0,
            "no_legal_candidate": 0,
        },
        "low_support_count": 0,
        "high_risk_count": 0,
        "negative_advantage_count": 0,
        "behavior_proximity_count": 0,
        "behavior_proximity_cap_counts": {
            "target_action_expansion": 0,
            "family_conversion": 0,
            "movement_stay_resource_conversion": 0,
        },
        "target_action_expansion_cap_count": 0,
        "family_conversion_cap_count": 0,
        "movement_stay_resource_conversion_cap_count": 0,
        "no_legal_candidate_count": 0,
        "target_action_counts": {action: 0 for action in ACTION_NAMES},
        "selected_logged_action_counts": {action: 0 for action in ACTION_NAMES},
        "target_distribution_tvd_from_logged": 0.0,
        "family_conversion_counts": {
            family: {target_family: 0 for target_family in _action_family_names()}
            for family in _action_family_names()
        },
        "movement_or_stay_to_resource_count": 0,
        "movement_or_stay_to_resource_rate": 0.0,
        "target_risk_mean": 0.0,
        "target_advantage_mean": 0.0,
        "target_support_mean": 0.0,
        "target_behavior_probability_mean": 0.0,
        "rejection_reasons_by_logged_action": {
            action: {} for action in ACTION_NAMES
        },
        "rejection_reasons_by_context": {},
    }


def _empty_contextual_behavior_prior_stats(
    *,
    row_count: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": (
            "mind_contextual_behavior_prior_actor_regularization_v1"
        ),
        "row_count": row_count,
        "active_row_count": 0,
        "active_row_rate": 0.0,
        "behavior_prior_entropy_mean": 0.0,
        "behavior_prior_max_probability_mean": 0.0,
        "behavior_prior_logged_probability_mean": None,
        "behavior_prior_logged_top1_rate": None,
    }


def _empty_action_distribution_stats(
    *,
    row_count: int = 0,
    temperature: float = TORCH_IQL_ACTION_DISTRIBUTION_TEMPERATURE,
) -> dict[str, object]:
    return {
        "schema_version": "mind_action_distribution_actor_regularization_v1",
        "policy": TORCH_IQL_ACTION_DISTRIBUTION_REGULARIZATION_POLICY,
        "row_count": row_count,
        "active_action_count": 0,
        "action_distribution_temperature": _round(float(temperature)),
        "action_distribution_kl": 0.0,
        "action_distribution_tvd": 0.0,
        "target_distribution": {action: 0.0 for action in ACTION_NAMES},
        "predicted_distribution": {action: 0.0 for action in ACTION_NAMES},
    }


def _empty_rollout_state_action_calibration_report() -> dict[str, object]:
    return {
        "schema_version": "mind_rollout_state_action_calibration_v1",
        "policy": TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY,
        "calibration_record_count": 0,
        "calibration_transition_count": 0,
        "max_action_share": TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_SHARE,
        "bias_step": TORCH_IQL_ROLLOUT_STATE_ACTION_BIAS_STEP,
        "max_bias_delta": TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_BIAS_DELTA,
        "max_iterations": TORCH_IQL_ROLLOUT_STATE_ACTION_MAX_ITERATIONS,
        "iteration_count": 0,
        "converged": False,
        "initially_converged": False,
        "action_bias_delta": {action: 0.0 for action in ACTION_NAMES},
        "before": _empty_rollout_state_action_top1_report(),
        "after": _empty_rollout_state_action_top1_report(),
    }


def _empty_rollout_state_action_top1_report(
    *,
    row_count: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": "mind_rollout_state_action_top1_report_v1",
        "policy": TORCH_IQL_ROLLOUT_STATE_ACTION_CALIBRATION_POLICY,
        "row_count": row_count,
        "active_row_count": 0,
        "top1_action_counts": {action: 0 for action in ACTION_NAMES},
        "top1_action_shares": {action: 0.0 for action in ACTION_NAMES},
        "logged_action_counts": {action: 0 for action in ACTION_NAMES},
        "logged_action_shares": {action: 0.0 for action in ACTION_NAMES},
        "dominant_action": None,
        "dominant_action_share": 0.0,
        "top1_logged_match_rate": 0.0,
        "top1_distribution_tvd_from_logged": 0.0,
    }


def _diagnostic_bool(diagnostics: object, key: str) -> bool:
    return isinstance(diagnostics, dict) and diagnostics.get(key) is True


def _diagnostic_action(diagnostics: object, key: str) -> str | None:
    if not isinstance(diagnostics, dict):
        return None
    value = diagnostics.get(key)
    if isinstance(value, str) and value in ACTION_NAMES:
        return value
    return None


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
    return_target: Any,
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
        cql_gaps = _cql_conservative_gaps(
            torch,
            action_values,
            y,
            mask_tensor,
        )
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
            "q_value_return_mean_abs_error": _round(
                float(
                    torch.mean(torch.abs(selected_action_values - return_target))
                    .cpu()
                    .item()
                )
            ),
            "cql_gap_mean": _round(float(cql_gaps.mean().cpu().item())),
            "cql_gap_min": _round(float(cql_gaps.min().cpu().item())),
            "cql_gap_max": _round(float(cql_gaps.max().cpu().item())),
            "state_value_mean_abs_error": _round(
                float(
                    torch.mean(torch.abs(state_values - q_target))
                    .cpu()
                    .item()
                )
            ),
            "state_value_return_mean_abs_error": _round(
                float(
                    torch.mean(torch.abs(state_values - return_target))
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
    include_viability_head: bool = False,
    torch_device_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    lower, upper = REWARD_TOTAL_BOUNDS
    payload = {
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
    if torch_device_metadata is not None:
        payload["torch_device_metadata"] = dict(torch_device_metadata)
    if include_viability_head:
        payload["viability_component_output_weights"] = _component_weight_map(
            model.viability.weight,
        )
        payload["viability_component_output_bias"] = _component_bias_map(
            model.viability.bias,
        )
        payload["action_viability_component_output_weights"] = (
            _action_component_weight_map(model.action_viability.weight)
        )
        payload["action_viability_component_output_bias"] = (
            _action_component_bias_map(model.action_viability.bias)
        )
    return payload


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


def _component_weight_map(parameter: Any) -> dict[str, list[float]]:
    rows = parameter.detach().cpu().tolist()
    return {
        component: _round_vector(rows[index])
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _component_scalar_map(parameter: Any) -> dict[str, float]:
    values = parameter.detach().cpu().tolist()
    return {
        component: _round(float(values[index]))
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _component_bias_map(parameter: Any) -> dict[str, float]:
    values = parameter.detach().cpu().tolist()
    return {
        component: _round(float(values[index]))
        for index, component in enumerate(VIABILITY_COMPONENT_NAMES)
    }


def _action_component_weight_map(parameter: Any) -> dict[str, dict[str, list[float]]]:
    rows = parameter.detach().cpu().tolist()
    component_count = len(VIABILITY_COMPONENT_NAMES)
    return {
        action: {
            component: _round_vector(
                rows[action_index * component_count + component_index]
            )
            for component_index, component in enumerate(VIABILITY_COMPONENT_NAMES)
        }
        for action_index, action in enumerate(ACTION_NAMES)
    }


def _action_component_bias_map(parameter: Any) -> dict[str, dict[str, float]]:
    values = parameter.detach().cpu().tolist()
    component_count = len(VIABILITY_COMPONENT_NAMES)
    return {
        action: {
            component: _round(
                float(values[action_index * component_count + component_index])
            )
            for component_index, component in enumerate(VIABILITY_COMPONENT_NAMES)
        }
        for action_index, action in enumerate(ACTION_NAMES)
    }


def _round_matrix(rows: list[list[float]]) -> list[list[float]]:
    return [_round_vector(row) for row in rows]


def _round_vector(values: list[float]) -> list[float]:
    return [_round(value) for value in values]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round(value: float) -> float:
    return round(float(value), 6)
