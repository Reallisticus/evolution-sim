from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import math
import random

import torch
from torch import Tensor

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    ACTION_COUNT,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    PublicRecurrentActorCritic,
    strict_action_mask_tensor,
)
from evolution_sim.mind.recurrent_counterfactual_branch import (
    RECURRENT_COUNTERFACTUAL_TRAINING_USE,
    RecurrentCounterfactualBranchError,
    reconstruct_current_model_hidden_from_branch_row,
    validate_recurrent_counterfactual_branch_row,
)
from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256


RECURRENT_COUNTERFACTUAL_AUXILIARY_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_soft_policy_improvement_v2"
)
RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION = (
    "mind_v3_recurrent_counterfactual_transactional_auxiliary_step_v1"
)
RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED = "disabled"
RECURRENT_COUNTERFACTUAL_TERMINAL_VALUE_TARGET = (
    "terminal_behavior_expected_discounted_return"
)
_VALUE_TARGET_MODES = {
    RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED,
    RECURRENT_COUNTERFACTUAL_TERMINAL_VALUE_TARGET,
}
RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED = "disabled"
RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS = (
    "deterministic_context_keyed_valid_action_scalarized_value_permutation_v2"
)
_TARGET_PERMUTATION_MODES = {
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED,
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS,
}


class RecurrentCounterfactualAuxiliaryError(ValueError):
    """Raised when counterfactual auxiliary learning cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class CounterfactualHorizonScalarization:
    """Explicit scalarization of exact action outcomes across branch horizons.

    Horizon weights must be unique, positive, and sum to one. Outcome weights
    are deliberately separate: this prevents an implicit survival, population,
    or birth objective from being smuggled into a nominal reward-return target.
    Deaths can be penalized with a negative ``deaths_during_horizon_weight``.
    """

    horizon_weights: tuple[tuple[int, float], ...]
    focal_discounted_return_weight: float = 1.0
    focal_terminal_alive_weight: float = 0.0
    population_alive_weight: float = 0.0
    births_during_horizon_weight: float = 0.0
    deaths_during_horizon_weight: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.horizon_weights, tuple) or not self.horizon_weights:
            raise RecurrentCounterfactualAuxiliaryError(
                "horizon_weights must be a non-empty tuple"
            )
        seen: set[int] = set()
        total = 0.0
        for index, item in enumerate(self.horizon_weights):
            if not isinstance(item, tuple) or len(item) != 2:
                raise RecurrentCounterfactualAuxiliaryError(
                    f"horizon weight {index} must be a (ticks, weight) tuple"
                )
            horizon, raw_weight = item
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon <= 0
            ):
                raise RecurrentCounterfactualAuxiliaryError(
                    "scalarization horizons must be unique positive integers"
                )
            if horizon in seen:
                raise RecurrentCounterfactualAuxiliaryError(
                    "scalarization horizons must be unique"
                )
            seen.add(horizon)
            weight = _finite_number(raw_weight, field="horizon weight")
            if weight <= 0.0:
                raise RecurrentCounterfactualAuxiliaryError(
                    "horizon weights must be strictly positive"
                )
            total += weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise RecurrentCounterfactualAuxiliaryError(
                "horizon weights must sum to one"
            )
        outcome_weights = self.outcome_weights()
        if not any(weight != 0.0 for weight in outcome_weights.values()):
            raise RecurrentCounterfactualAuxiliaryError(
                "at least one outcome scalarization weight must be non-zero"
            )

    def outcome_weights(self) -> dict[str, float]:
        return {
            "focal_discounted_return": _finite_number(
                self.focal_discounted_return_weight,
                field="focal discounted return weight",
            ),
            "focal_terminal_alive": _finite_number(
                self.focal_terminal_alive_weight,
                field="focal terminal alive weight",
            ),
            "population_alive": _finite_number(
                self.population_alive_weight,
                field="population alive weight",
            ),
            "births_during_horizon": _finite_number(
                self.births_during_horizon_weight,
                field="births during horizon weight",
            ),
            "deaths_during_horizon": _finite_number(
                self.deaths_during_horizon_weight,
                field="deaths during horizon weight",
            ),
        }

    def as_contract(self) -> dict[str, object]:
        return {
            "horizon_weights": [
                {"horizon_ticks": horizon, "weight": float(weight)}
                for horizon, weight in self.horizon_weights
            ],
            "outcome_weights": self.outcome_weights(),
            "missing_terminal_resource_policy": "not_used",
            "implicit_weight_normalization": False,
        }


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualAuxiliaryConfig:
    scalarization: CounterfactualHorizonScalarization
    temperature: float = 1.0
    advantage_clip: float = 5.0
    policy_improvement_coefficient: float = 1.0
    behavior_kl_coefficient: float = 0.1
    value_loss_coefficient: float = 0.0
    value_target_mode: str = RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED
    target_permutation_mode: str = RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
    target_permutation_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scalarization, CounterfactualHorizonScalarization):
            raise RecurrentCounterfactualAuxiliaryError(
                "scalarization must be a CounterfactualHorizonScalarization"
            )
        temperature = _finite_number(self.temperature, field="temperature")
        advantage_clip = _finite_number(
            self.advantage_clip,
            field="advantage clip",
        )
        if temperature <= 0.0:
            raise RecurrentCounterfactualAuxiliaryError(
                "temperature must be strictly positive"
            )
        if advantage_clip <= 0.0:
            raise RecurrentCounterfactualAuxiliaryError(
                "advantage clip must be strictly positive"
            )
        for field, value in (
            (
                "policy improvement coefficient",
                self.policy_improvement_coefficient,
            ),
            ("behavior KL coefficient", self.behavior_kl_coefficient),
            ("value loss coefficient", self.value_loss_coefficient),
        ):
            if _finite_number(value, field=field) < 0.0:
                raise RecurrentCounterfactualAuxiliaryError(
                    f"{field} must be non-negative"
                )
        if self.policy_improvement_coefficient == 0.0:
            raise RecurrentCounterfactualAuxiliaryError(
                "policy improvement coefficient must be positive"
            )
        if self.value_target_mode not in _VALUE_TARGET_MODES:
            raise RecurrentCounterfactualAuxiliaryError(
                "value target mode is unsupported"
            )
        if self.value_target_mode == RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED:
            if self.value_loss_coefficient != 0.0:
                raise RecurrentCounterfactualAuxiliaryError(
                    "disabled value target requires a zero value loss coefficient"
                )
        elif self.value_loss_coefficient <= 0.0:
            raise RecurrentCounterfactualAuxiliaryError(
                "terminal value targets require a positive value loss coefficient"
            )
        if self.target_permutation_mode not in _TARGET_PERMUTATION_MODES:
            raise RecurrentCounterfactualAuxiliaryError(
                "target permutation mode is unsupported"
            )
        if (
            self.target_permutation_mode
            == RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
        ):
            if self.target_permutation_seed is not None:
                raise RecurrentCounterfactualAuxiliaryError(
                    "disabled target permutation requires a null seed"
                )
        elif (
            isinstance(self.target_permutation_seed, bool)
            or not isinstance(self.target_permutation_seed, int)
            or self.target_permutation_seed < 0
            or self.target_permutation_seed > 2**63 - 1
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "enabled target permutation requires an integer seed in [0, 2**63 - 1]"
            )

    def as_contract(self) -> dict[str, object]:
        return {
            "schema_version": RECURRENT_COUNTERFACTUAL_AUXILIARY_SCHEMA_VERSION,
            "scalarization": self.scalarization.as_contract(),
            "advantage_baseline": "source_behavior_probability_weighted_action_value",
            "soft_target": ("normalize(pi_old * exp(clip(advantage) / temperature))"),
            "temperature": float(self.temperature),
            "advantage_clip": float(self.advantage_clip),
            "policy_loss": "forward_kl(soft_improvement_target || current_actor)",
            "behavior_regularizer": "forward_kl(pi_old || current_actor)",
            "policy_improvement_coefficient": float(
                self.policy_improvement_coefficient
            ),
            "behavior_kl_coefficient": float(self.behavior_kl_coefficient),
            "value_target_mode": self.value_target_mode,
            "value_loss_coefficient": float(self.value_loss_coefficient),
            "scientific_negative_control": {
                "enabled": self.target_permutation_mode
                != RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED,
                "mode": self.target_permutation_mode,
                "seed": self.target_permutation_seed,
                "applied_after_exact_row_validation": True,
                "permuted_field": "scalarized_policy_action_values_only",
                "permutation_seed_derivation": (
                    "sha256(master_seed,trainable_public_context_sha256)"
                ),
                "same_positional_permutation_reused_for_every_group": False,
                "branch_rows_mutated": False,
                "exact_branch_labels_claimed_after_permutation": False,
            },
            "model_input_fields": [
                "public_history_prefix",
                "current_public_observation",
                "current_public_action_mask",
                "previous_public_feedback",
            ],
            "metadata_seed_fixture_private_or_provenance_used_as_model_input": False,
            "outcome_labels_used_as_model_input": False,
            "stored_source_hidden_used_as_model_input": False,
            "current_hidden_state_policy": (
                "reconstruct_from_public_history_with_exact_current_model"
            ),
            "ppo_ratio_data_use": False,
            "branch_label_statistical_semantics": (
                "single_exact_rollout_target_not_expected_causal_effect"
            ),
            "branch_outcome_uncertainty_estimated": False,
            "optimizer_step_performed": False,
            "runtime_integrated": False,
        }


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualAuxiliaryTarget:
    source_model_state_sha256: str
    source_artifact_digest: str
    row_exact_digests: tuple[str, ...]
    public_context_sha256: str
    horizons: tuple[int, ...]
    action_mask: Tensor
    behavior_probabilities: Tensor
    scalarized_action_values: Tensor
    behavior_centered_advantages: Tensor
    clipped_advantages: Tensor
    soft_policy_target: Tensor
    value_target: Tensor | None
    contract: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualAuxiliaryLoss:
    loss: Tensor
    policy_improvement_kl: Tensor
    behavior_kl: Tensor
    value_loss: Tensor
    current_value: Tensor
    current_probabilities: Tensor
    target: RecurrentCounterfactualAuxiliaryTarget


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualAuxiliaryStepConfig:
    """Safety boundary for one and only one auxiliary Adam step."""

    learning_rate_multiplier: float = 0.1
    max_gradient_norm: float = 0.5
    mean_behavior_kl_limit: float = 0.002
    max_state_behavior_kl_limit: float = 0.010

    def __post_init__(self) -> None:
        if (
            _finite_number(
                self.learning_rate_multiplier,
                field="auxiliary learning-rate multiplier",
            )
            <= 0.0
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary learning-rate multiplier must be strictly positive"
            )
        if (
            _finite_number(
                self.max_gradient_norm,
                field="auxiliary maximum gradient norm",
            )
            <= 0.0
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary maximum gradient norm must be strictly positive"
            )
        for field, value in (
            ("mean behavior KL limit", self.mean_behavior_kl_limit),
            ("maximum state behavior KL limit", self.max_state_behavior_kl_limit),
        ):
            if _finite_number(value, field=field) < 0.0:
                raise RecurrentCounterfactualAuxiliaryError(
                    f"{field} must be non-negative"
                )

    def as_contract(self) -> dict[str, object]:
        return {
            "schema_version": (RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION),
            "optimizer": "shared_ppo_adam",
            "optimizer_steps": 1,
            "backward_passes": 1,
            "retry_on_rejection_or_error": False,
            "learning_rate_multiplier": float(self.learning_rate_multiplier),
            "max_gradient_norm": float(self.max_gradient_norm),
            "post_step_public_policy_audit": {
                "mean_forward_kl_pi_old_to_pi_post_limit": float(
                    self.mean_behavior_kl_limit
                ),
                "max_state_forward_kl_pi_old_to_pi_post_limit": float(
                    self.max_state_behavior_kl_limit
                ),
            },
            "runtime_integrated": False,
            "promotion_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualAuxiliaryBatchLoss:
    """One mean loss over independently reconstructed public branch states."""

    loss: Tensor
    policy_improvement_kl: Tensor
    behavior_kl: Tensor
    value_loss: Tensor
    group_losses: tuple[RecurrentCounterfactualAuxiliaryLoss, ...]
    targets: tuple[RecurrentCounterfactualAuxiliaryTarget, ...]


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualAuxiliaryStepDiagnostics:
    schema_version: str
    accepted: bool
    rejection_reason: str | None
    retry_authorized: bool
    bundle_digest: str
    pre_model_state_sha256: str
    attempted_post_model_state_sha256: str
    final_model_state_sha256: str
    ppo_update_index: int
    auxiliary_update_count: int
    group_count: int
    row_count: int
    min_public_prefix_length: int
    max_public_prefix_length: int
    optimizer_step_count: int
    backward_pass_count: int
    learning_rate_multiplier: float
    parameter_group_learning_rates: tuple[float, ...]
    total_loss: float
    policy_improvement_kl: float
    behavior_kl: float
    value_loss: float
    gradient_norm_before_clip: float
    gradient_norm_after_clip: float
    parameter_delta_l2: float
    mean_behavior_kl_old_to_post: float
    max_state_behavior_kl_old_to_post: float
    mean_behavior_kl_limit: float
    max_state_behavior_kl_limit: float
    rollback_performed: bool
    optimizer_state_restored: bool
    parameter_group_learning_rates_restored: bool
    update_counters_restored: bool
    rows_stale_after_step: bool
    rows_exact_model_valid_after_step: bool
    runtime_artifact_created: bool
    runtime_action_selection_changed: bool
    promotion_authorized: bool


def build_recurrent_counterfactual_auxiliary_target(
    model: PublicRecurrentActorCritic,
    rows: Sequence[Mapping[str, object]],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> RecurrentCounterfactualAuxiliaryTarget:
    """Build a detached soft-improvement target without mutating the model.

    Rows are accepted only from the exact current model and exact artifact.
    The current recurrent state is recomputed from the actor-public prefix; the
    serialized source hidden tensor is never read as a training input.
    """

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    if not isinstance(config, RecurrentCounterfactualAuxiliaryConfig):
        raise TypeError("config must be a RecurrentCounterfactualAuxiliaryConfig")
    resolved_artifact = _nonempty_string(
        artifact_digest,
        field="artifact_digest",
    )
    ordered_rows = _validated_ordered_rows(
        model,
        rows,
        artifact_digest=resolved_artifact,
        config=config,
    )
    first = ordered_rows[0]
    context = _mapping(
        first.get("trainable_public_context"),
        field="trainable_public_context",
    )
    public_context_sha256 = stable_payload_digest(context)
    row_exact_digests = tuple(
        _nonempty_string(row.get("exact_digest"), field="row exact digest")
        for row in ordered_rows
    )
    permutation_identity_sha256 = stable_payload_digest(
        {
            "policy": "counterfactual_negative_control_group_permutation_v2",
            "master_seed": config.target_permutation_seed,
            "public_context_sha256": public_context_sha256,
        }
    )
    reference = next(model.parameters())
    action_mask = strict_action_mask_tensor(
        _mapping(
            context.get("current_public_action_mask"),
            field="current public action mask",
        ),
        device=reference.device,
    )
    behavior = _behavior_probability_tensor(
        first,
        device=reference.device,
        dtype=reference.dtype,
    )
    parameter_versions_before = tuple(
        parameter._version for parameter in model.parameters()
    )
    model_digest_before = recurrent_model_state_sha256(model)
    with torch.no_grad():
        current_probabilities, _ = _current_actor_distribution_and_value(
            model,
            first,
        )
        _require_probability_match(
            current_probabilities,
            behavior,
            action_mask=action_mask,
        )
        scalarized_values, pure_return_values = _scalarized_action_values(
            ordered_rows,
            config=config,
            action_mask=action_mask,
            device=reference.device,
            dtype=reference.dtype,
        )
        scalarized_values = _maybe_permute_scalarized_action_values(
            scalarized_values,
            action_mask=action_mask,
            config=config,
            permutation_identity_sha256=permutation_identity_sha256,
        )
        baseline = torch.sum(behavior * scalarized_values)
        advantages = torch.where(
            action_mask,
            scalarized_values - baseline,
            torch.zeros_like(scalarized_values),
        )
        clipped = torch.where(
            action_mask,
            torch.clamp(
                advantages,
                min=-float(config.advantage_clip),
                max=float(config.advantage_clip),
            ),
            torch.zeros_like(advantages),
        )
        support = action_mask & (behavior > 0.0)
        if not bool(support.any().item()):
            raise RecurrentCounterfactualAuxiliaryError(
                "source behavior has no positive probability on a valid action"
            )
        target_logits = torch.full_like(behavior, -torch.inf)
        target_logits[support] = torch.log(behavior[support]) + clipped[
            support
        ] / float(config.temperature)
        soft_target = torch.softmax(target_logits, dim=-1)
        value_target = _terminal_value_target(
            ordered_rows,
            behavior=behavior,
            pure_return_values=pure_return_values,
            config=config,
        )

    parameter_versions_after = tuple(
        parameter._version for parameter in model.parameters()
    )
    model_digest_after = recurrent_model_state_sha256(model)
    if (
        parameter_versions_after != parameter_versions_before
        or model_digest_after != model_digest_before
    ):
        raise RecurrentCounterfactualAuxiliaryError(
            "target construction mutated current model parameters"
        )
    _require_finite_masked_vector(
        scalarized_values,
        action_mask=action_mask,
        field="scalarized action values",
    )
    _require_finite_masked_vector(
        advantages,
        action_mask=action_mask,
        field="behavior-centered advantages",
    )
    _require_probability_vector(
        soft_target,
        action_mask=action_mask,
        field="soft policy target",
    )
    target_contract = copy.deepcopy(config.as_contract())
    negative_control = _mapping(
        target_contract.get("scientific_negative_control"),
        field="scientific_negative_control",
    )
    negative_control["group_permutation_identity_sha256"] = permutation_identity_sha256
    return RecurrentCounterfactualAuxiliaryTarget(
        source_model_state_sha256=model_digest_before,
        source_artifact_digest=resolved_artifact,
        row_exact_digests=row_exact_digests,
        public_context_sha256=public_context_sha256,
        horizons=tuple(
            _positive_int(
                _mapping(row.get("metadata"), field="metadata").get("horizon_ticks"),
                field="horizon_ticks",
            )
            for row in ordered_rows
        ),
        action_mask=action_mask.detach().clone(),
        behavior_probabilities=behavior.detach().clone(),
        scalarized_action_values=scalarized_values.detach().clone(),
        behavior_centered_advantages=advantages.detach().clone(),
        clipped_advantages=clipped.detach().clone(),
        soft_policy_target=soft_target.detach().clone(),
        value_target=(None if value_target is None else value_target.detach().clone()),
        contract=target_contract,
    )


def recurrent_counterfactual_auxiliary_loss(
    model: PublicRecurrentActorCritic,
    rows: Sequence[Mapping[str, object]],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> RecurrentCounterfactualAuxiliaryLoss:
    """Return a differentiable auxiliary loss; never step an optimizer here."""

    batch = recurrent_counterfactual_auxiliary_batch_loss(
        model,
        (rows,),
        artifact_digest=artifact_digest,
        config=config,
    )
    return batch.group_losses[0]


def recurrent_counterfactual_auxiliary_batch_loss(
    model: PublicRecurrentActorCritic,
    row_groups: Sequence[Sequence[Mapping[str, object]]],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> RecurrentCounterfactualAuxiliaryBatchLoss:
    """Build all frozen targets, then one mean loss against one model state.

    Groups are evaluated independently rather than padded.  This is deliberate:
    each exact branch state can have a different public-history length, and no
    padding token is part of the actor-public contract.
    """

    groups = _normalized_row_groups(row_groups)
    model_digest_before = recurrent_model_state_sha256(model)
    parameter_versions_before = tuple(
        parameter._version for parameter in model.parameters()
    )
    targets = tuple(
        build_recurrent_counterfactual_auxiliary_target(
            model,
            group,
            artifact_digest=artifact_digest,
            config=config,
        )
        for group in groups
    )
    if (
        recurrent_model_state_sha256(model) != model_digest_before
        or tuple(parameter._version for parameter in model.parameters())
        != parameter_versions_before
    ):
        raise RecurrentCounterfactualAuxiliaryError(
            "batch target construction mutated the current model"
        )
    group_losses = tuple(
        _loss_against_frozen_target(
            model,
            _ordered_rows_without_revalidation(group, config=config)[0],
            target,
            config=config,
        )
        for group, target in zip(groups, targets, strict=True)
    )
    if recurrent_model_state_sha256(model) != model_digest_before:
        raise RecurrentCounterfactualAuxiliaryError(
            "current model changed while building the auxiliary batch loss"
        )
    return RecurrentCounterfactualAuxiliaryBatchLoss(
        loss=torch.stack([result.loss for result in group_losses]).mean(),
        policy_improvement_kl=torch.stack(
            [result.policy_improvement_kl for result in group_losses]
        ).mean(),
        behavior_kl=torch.stack([result.behavior_kl for result in group_losses]).mean(),
        value_loss=torch.stack([result.value_loss for result in group_losses]).mean(),
        group_losses=group_losses,
        targets=targets,
    )


def recurrent_counterfactual_auxiliary_bundle_digest(
    row_groups: Sequence[Sequence[Mapping[str, object]]],
    *,
    artifact_digest: str,
) -> str:
    """Return a configuration-independent identity for one-use row groups."""

    groups = _normalized_row_groups(row_groups)
    resolved_artifact = _nonempty_string(
        artifact_digest,
        field="artifact_digest",
    )
    canonical_groups: list[tuple[str, ...]] = []
    for group in groups:
        row_digests = tuple(
            sorted(
                _nonempty_string(row.get("exact_digest"), field="row exact digest")
                for row in group
            )
        )
        canonical_groups.append(row_digests)
    canonical_groups.sort()
    return stable_payload_digest(
        {
            "artifact_digest": resolved_artifact,
            "row_groups": [list(group) for group in canonical_groups],
        }
    )


def _transactional_recurrent_counterfactual_auxiliary_step(
    model: PublicRecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    row_groups: Sequence[Sequence[Mapping[str, object]]],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualAuxiliaryConfig,
    step_config: RecurrentCounterfactualAuxiliaryStepConfig | None = None,
) -> RecurrentCounterfactualAuxiliaryStepDiagnostics:
    """Attempt exactly one auxiliary Adam step and fail closed on audit drift.

    This low-level transaction does not authorize retries.  The PPO trainer
    owns durable in-process one-use tracking and update counters.
    """

    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    if not isinstance(optimizer, torch.optim.Adam):
        raise RecurrentCounterfactualAuxiliaryError(
            "counterfactual auxiliary learning requires the PPO-owned Adam optimizer"
        )
    if not isinstance(config, RecurrentCounterfactualAuxiliaryConfig):
        raise TypeError("config must be a RecurrentCounterfactualAuxiliaryConfig")
    resolved_step_config = step_config or RecurrentCounterfactualAuxiliaryStepConfig()
    if not isinstance(
        resolved_step_config,
        RecurrentCounterfactualAuxiliaryStepConfig,
    ):
        raise TypeError(
            "step_config must be a RecurrentCounterfactualAuxiliaryStepConfig"
        )
    groups = _normalized_row_groups(row_groups)
    bundle_digest = recurrent_counterfactual_auxiliary_bundle_digest(
        groups,
        artifact_digest=artifact_digest,
    )
    prefix_lengths = tuple(_public_prefix_length(group[0]) for group in groups)
    model_snapshot = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    gradient_snapshot = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    )
    learning_rate_snapshot = tuple(
        copy.deepcopy(group.get("lr")) for group in optimizer.param_groups
    )
    learning_rates = tuple(
        _finite_positive_learning_rate(value) for value in learning_rate_snapshot
    )
    pre_digest = recurrent_model_state_sha256(model)

    try:
        batch = recurrent_counterfactual_auxiliary_batch_loss(
            model,
            groups,
            artifact_digest=artifact_digest,
            config=config,
        )
        if recurrent_model_state_sha256(model) != pre_digest:
            raise RecurrentCounterfactualAuxiliaryError(
                "current model changed before the auxiliary optimizer step"
            )
        optimizer.zero_grad(set_to_none=True)
        batch.loss.backward()
        _require_finite_gradients(model)
        gradient_norm_before = _gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(resolved_step_config.max_gradient_norm),
            error_if_nonfinite=True,
        )
        _require_finite_gradients(model)
        gradient_norm_after = _gradient_norm(model)
        for parameter_group, base_learning_rate in zip(
            optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            parameter_group["lr"] = base_learning_rate * float(
                resolved_step_config.learning_rate_multiplier
            )
        optimizer.step()
        for parameter_group, original_learning_rate in zip(
            optimizer.param_groups,
            learning_rate_snapshot,
            strict=True,
        ):
            parameter_group["lr"] = original_learning_rate
        _require_finite_model_and_optimizer(model, optimizer)
        attempted_post_digest = recurrent_model_state_sha256(model)
        post_probabilities: list[Tensor] = []
        state_kls: list[float] = []
        with torch.no_grad():
            for group, target in zip(groups, batch.targets, strict=True):
                probabilities, _ = _current_actor_distribution_and_value(
                    model,
                    _ordered_rows_without_revalidation(group, config=config)[0],
                    differentiable_prefix=True,
                )
                _require_probability_vector(
                    probabilities,
                    action_mask=target.action_mask,
                    field="post-step current actor probabilities",
                )
                post_probabilities.append(probabilities)
                kl = _forward_kl(
                    target.behavior_probabilities[target.action_mask],
                    probabilities[target.action_mask],
                )
                parsed_kl = _finite_tensor_scalar(
                    kl,
                    field="post-step state behavior KL",
                )
                state_kls.append(max(0.0, parsed_kl))
        del post_probabilities
        mean_kl = math.fsum(state_kls) / len(state_kls)
        max_kl = max(state_kls)
        rejection_reason: str | None = None
        if attempted_post_digest == pre_digest:
            rejection_reason = "optimizer_step_did_not_change_model_state"
        elif mean_kl > float(resolved_step_config.mean_behavior_kl_limit):
            rejection_reason = "mean_public_policy_kl_limit_exceeded"
        elif max_kl > float(resolved_step_config.max_state_behavior_kl_limit):
            rejection_reason = "max_state_public_policy_kl_limit_exceeded"
        accepted = rejection_reason is None
        parameter_delta = _model_delta_l2(model, model_snapshot)
        rollback_performed = not accepted
        if rollback_performed:
            _restore_training_transaction(
                model,
                optimizer,
                model_snapshot=model_snapshot,
                optimizer_snapshot=optimizer_snapshot,
                gradient_snapshot=gradient_snapshot,
                learning_rate_snapshot=learning_rate_snapshot,
            )
        else:
            optimizer.zero_grad(set_to_none=True)
        final_digest = recurrent_model_state_sha256(model)
        optimizer_restored = rollback_performed and _nested_state_equal(
            optimizer.state_dict(),
            optimizer_snapshot,
        )
        learning_rates_restored = all(
            _learning_rate_equal(group.get("lr"), expected)
            for group, expected in zip(
                optimizer.param_groups,
                learning_rate_snapshot,
                strict=True,
            )
        )
        if not learning_rates_restored:
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary parameter-group learning rates were not restored"
            )
        if rollback_performed and (
            final_digest != pre_digest or not optimizer_restored
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "rejected auxiliary transaction did not restore exact state"
            )
        return RecurrentCounterfactualAuxiliaryStepDiagnostics(
            schema_version=RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION,
            accepted=accepted,
            rejection_reason=rejection_reason,
            retry_authorized=False,
            bundle_digest=bundle_digest,
            pre_model_state_sha256=pre_digest,
            attempted_post_model_state_sha256=attempted_post_digest,
            final_model_state_sha256=final_digest,
            ppo_update_index=-1,
            auxiliary_update_count=-1,
            group_count=len(groups),
            row_count=sum(len(group) for group in groups),
            min_public_prefix_length=min(prefix_lengths),
            max_public_prefix_length=max(prefix_lengths),
            optimizer_step_count=1,
            backward_pass_count=1,
            learning_rate_multiplier=float(
                resolved_step_config.learning_rate_multiplier
            ),
            parameter_group_learning_rates=learning_rates,
            total_loss=_finite_tensor_scalar(batch.loss, field="batch total loss"),
            policy_improvement_kl=_finite_tensor_scalar(
                batch.policy_improvement_kl,
                field="batch policy improvement KL",
            ),
            behavior_kl=_finite_tensor_scalar(
                batch.behavior_kl,
                field="batch behavior KL",
            ),
            value_loss=_finite_tensor_scalar(
                batch.value_loss,
                field="batch value loss",
            ),
            gradient_norm_before_clip=gradient_norm_before,
            gradient_norm_after_clip=gradient_norm_after,
            parameter_delta_l2=parameter_delta,
            mean_behavior_kl_old_to_post=mean_kl,
            max_state_behavior_kl_old_to_post=max_kl,
            mean_behavior_kl_limit=float(resolved_step_config.mean_behavior_kl_limit),
            max_state_behavior_kl_limit=float(
                resolved_step_config.max_state_behavior_kl_limit
            ),
            rollback_performed=rollback_performed,
            optimizer_state_restored=optimizer_restored,
            parameter_group_learning_rates_restored=learning_rates_restored,
            update_counters_restored=False,
            rows_stale_after_step=accepted,
            rows_exact_model_valid_after_step=not accepted,
            runtime_artifact_created=False,
            runtime_action_selection_changed=False,
            promotion_authorized=False,
        )
    except Exception as error:
        _restore_training_transaction(
            model,
            optimizer,
            model_snapshot=model_snapshot,
            optimizer_snapshot=optimizer_snapshot,
            gradient_snapshot=gradient_snapshot,
            learning_rate_snapshot=learning_rate_snapshot,
        )
        if recurrent_model_state_sha256(model) != pre_digest or not _nested_state_equal(
            optimizer.state_dict(), optimizer_snapshot
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary error rollback could not restore exact training state"
            ) from error
        if isinstance(error, RecurrentCounterfactualAuxiliaryError):
            raise
        raise RecurrentCounterfactualAuxiliaryError(
            "auxiliary optimizer step failed closed and restored exact state"
        ) from error


def _loss_against_frozen_target(
    model: PublicRecurrentActorCritic,
    first_row: Mapping[str, object],
    target: RecurrentCounterfactualAuxiliaryTarget,
    *,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> RecurrentCounterfactualAuxiliaryLoss:
    if recurrent_model_state_sha256(model) != target.source_model_state_sha256:
        raise RecurrentCounterfactualAuxiliaryError(
            "current model changed after target construction"
        )
    current_probabilities, current_value = _current_actor_distribution_and_value(
        model,
        first_row,
        differentiable_prefix=True,
    )
    valid = target.action_mask
    current_valid = current_probabilities[valid]
    target_valid = target.soft_policy_target[valid]
    behavior_valid = target.behavior_probabilities[valid]
    policy_improvement_kl = _forward_kl(target_valid, current_valid)
    behavior_kl = _forward_kl(behavior_valid, current_valid)
    if target.value_target is None:
        value_loss = current_value.new_zeros(())
    else:
        value_loss = torch.square(current_value - target.value_target)
    loss = (
        float(config.policy_improvement_coefficient) * policy_improvement_kl
        + float(config.behavior_kl_coefficient) * behavior_kl
        + float(config.value_loss_coefficient) * value_loss
    )
    for field, value in (
        ("total auxiliary loss", loss),
        ("policy improvement KL", policy_improvement_kl),
        ("behavior KL", behavior_kl),
        ("value loss", value_loss),
        ("current value", current_value),
    ):
        if value.numel() != 1 or not bool(torch.isfinite(value).item()):
            raise RecurrentCounterfactualAuxiliaryError(f"{field} is non-finite")
    _require_probability_vector(
        current_probabilities,
        action_mask=valid,
        field="current actor probabilities",
    )
    return RecurrentCounterfactualAuxiliaryLoss(
        loss=loss,
        policy_improvement_kl=policy_improvement_kl,
        behavior_kl=behavior_kl,
        value_loss=value_loss,
        current_value=current_value,
        current_probabilities=current_probabilities,
        target=target,
    )


def _validated_ordered_rows(
    model: PublicRecurrentActorCritic,
    rows: Sequence[Mapping[str, object]],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> tuple[Mapping[str, object], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise RecurrentCounterfactualAuxiliaryError(
            "rows must be a non-empty sequence of branch-row mappings"
        )
    by_horizon: dict[int, Mapping[str, object]] = {}
    current_model_digest = recurrent_model_state_sha256(model)
    context_digest: str | None = None
    behavior_digest: str | None = None
    source_action: object = None
    source_state_identity: tuple[object, ...] | None = None
    for index, row in enumerate(rows):
        try:
            validate_recurrent_counterfactual_branch_row(row)
        except RecurrentCounterfactualBranchError as error:
            raise RecurrentCounterfactualAuxiliaryError(
                f"counterfactual row {index} failed its exact branch contract"
            ) from error
        contract = _mapping(row.get("contract"), field="contract")
        if contract.get("exact_replay_required") is not True:
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary learning requires exact replay-verified branch rows"
            )
        if (
            contract.get("historical_row_training_use")
            != (RECURRENT_COUNTERFACTUAL_TRAINING_USE)
            or contract.get("ppo_ratio_data_use_forbidden") is not True
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "branch rows are auxiliary-only and forbidden as PPO-ratio data"
            )
        optimizer_context = _mapping(
            row.get("optimizer_context"),
            field="optimizer_context",
        )
        if optimizer_context.get("source_artifact_digest") != artifact_digest:
            raise RecurrentCounterfactualAuxiliaryError(
                "counterfactual source artifact does not match the requested artifact"
            )
        if optimizer_context.get("source_model_state_sha256") != current_model_digest:
            raise RecurrentCounterfactualAuxiliaryError(
                "counterfactual row is stale for the current model state"
            )
        if optimizer_context.get("current_model_state_policy") != (
            "reconstruct_from_trainable_public_history_prefix"
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "current-model hidden state is not public-history reconstructable"
            )
        metadata = _mapping(row.get("metadata"), field="metadata")
        if metadata.get("source_artifact_digest") != artifact_digest:
            raise RecurrentCounterfactualAuxiliaryError(
                "counterfactual metadata artifact identity drifted"
            )
        if metadata.get("source_model_state_sha256") != current_model_digest:
            raise RecurrentCounterfactualAuxiliaryError(
                "counterfactual metadata model identity is stale"
            )
        horizon = _positive_int(
            metadata.get("horizon_ticks"),
            field="horizon_ticks",
        )
        if horizon in by_horizon:
            raise RecurrentCounterfactualAuxiliaryError(
                "counterfactual horizon rows must be unique"
            )
        by_horizon[horizon] = row
        context = _mapping(
            row.get("trainable_public_context"),
            field="trainable_public_context",
        )
        observed_context_digest = stable_payload_digest(context)
        labels = _mapping(row.get("labels"), field="labels")
        observed_behavior_digest = stable_payload_digest(
            labels.get("source_behavior_distribution")
        )
        observed_state_identity = (
            metadata.get("seed_role"),
            metadata.get("environment_seed"),
            metadata.get("scenario"),
            metadata.get("branch_tick"),
            metadata.get("focal_agent_id"),
            metadata.get("policy_sampling_seed"),
            metadata.get("source_observation_digest"),
            metadata.get("source_action_mask_digest"),
            metadata.get("source_sampling_state_sha256"),
            metadata.get("source_public_history_prefix_sha256"),
        )
        if context_digest is None:
            context_digest = observed_context_digest
            behavior_digest = observed_behavior_digest
            source_action = labels.get("source_requested_action")
            source_state_identity = observed_state_identity
        elif (
            observed_context_digest != context_digest
            or observed_behavior_digest != behavior_digest
            or labels.get("source_requested_action") != source_action
            or observed_state_identity != source_state_identity
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "horizon rows do not describe one exact public branch state"
            )
    expected_horizons = tuple(
        horizon for horizon, _ in config.scalarization.horizon_weights
    )
    if set(by_horizon) != set(expected_horizons):
        raise RecurrentCounterfactualAuxiliaryError(
            "counterfactual horizons do not exactly match scalarization horizons"
        )
    return tuple(by_horizon[horizon] for horizon in expected_horizons)


def _ordered_rows_without_revalidation(
    rows: Sequence[Mapping[str, object]],
    *,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> tuple[Mapping[str, object], ...]:
    by_horizon = {
        int(_mapping(row.get("metadata"), field="metadata")["horizon_ticks"]): row
        for row in rows
    }
    return tuple(
        by_horizon[horizon] for horizon, _ in config.scalarization.horizon_weights
    )


def _current_actor_distribution_and_value(
    model: PublicRecurrentActorCritic,
    row: Mapping[str, object],
    *,
    differentiable_prefix: bool = False,
) -> tuple[Tensor, Tensor]:
    context = _mapping(
        row.get("trainable_public_context"),
        field="trainable_public_context",
    )
    reference = next(model.parameters())
    observation_payload = _mapping(
        context.get("current_public_observation"),
        field="current public observation",
    )
    observation_values = observation_payload.get("values")
    if not isinstance(observation_values, list) or len(observation_values) != (
        ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
    ):
        raise RecurrentCounterfactualAuxiliaryError(
            "current public observation vector drifted"
        )
    feedback_payload = _mapping(
        context.get("previous_public_feedback"),
        field="previous public feedback",
    )
    feedback_values = feedback_payload.get("values")
    if not isinstance(feedback_values, list) or len(feedback_values) != (
        PREVIOUS_PUBLIC_FEEDBACK_SIZE
    ):
        raise RecurrentCounterfactualAuxiliaryError(
            "previous public feedback vector drifted"
        )
    observation = torch.tensor(
        observation_values,
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, 1, -1)
    action_mask = strict_action_mask_tensor(
        _mapping(
            context.get("current_public_action_mask"),
            field="current public action mask",
        ),
        device=reference.device,
    ).reshape(1, 1, -1)
    previous_feedback = torch.tensor(
        feedback_values,
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, 1, -1)
    if differentiable_prefix:
        recurrent_state = _differentiable_public_prefix_hidden(model, context)
    else:
        try:
            recurrent_state = reconstruct_current_model_hidden_from_branch_row(
                model,
                row,
            )
        except RecurrentCounterfactualBranchError as error:
            raise RecurrentCounterfactualAuxiliaryError(
                "failed to reconstruct current-model hidden state from public history"
            ) from error
    output = model.forward_sequence(
        observation,
        action_mask,
        previous_feedback,
        initial_state=recurrent_state,
    )
    probabilities = torch.softmax(output.masked_logits[0, 0], dim=-1)
    return probabilities, output.values[0, 0]


def _differentiable_public_prefix_hidden(
    model: PublicRecurrentActorCritic,
    context: Mapping[str, object],
) -> Tensor:
    """Replay the complete actor-public prefix with gradients intact."""

    prefix = _mapping(
        context.get("public_history_prefix"),
        field="public history prefix",
    )
    records = prefix.get("records")
    if not isinstance(records, list):
        raise RecurrentCounterfactualAuxiliaryError(
            "public history prefix records must be a list"
        )
    if prefix.get("record_count") != len(records):
        raise RecurrentCounterfactualAuxiliaryError(
            "public history prefix record count drifted"
        )
    reference = next(model.parameters())
    state = model.initial_state(1)
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, field=f"public history record {index}")
        reset = record.get("recurrent_state_reset_before_decision")
        if type(reset) is not bool:
            raise RecurrentCounterfactualAuxiliaryError(
                "public history reset flags must be exact booleans"
            )
        if reset:
            state = model.initial_state(1)
        observation_payload = _mapping(
            record.get("public_observation"),
            field=f"public history observation {index}",
        )
        observation_values = observation_payload.get("values")
        if not isinstance(observation_values, list) or len(observation_values) != (
            ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "public history observation vector drifted"
            )
        feedback_payload = _mapping(
            record.get("previous_public_feedback"),
            field=f"public history feedback {index}",
        )
        feedback_values = feedback_payload.get("values")
        if not isinstance(feedback_values, list) or len(feedback_values) != (
            PREVIOUS_PUBLIC_FEEDBACK_SIZE
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "public history feedback vector drifted"
            )
        observation = torch.tensor(
            observation_values,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, -1)
        action_mask_payload = _mapping(
            record.get("public_action_mask"),
            field=f"public history action mask {index}",
        )
        if action_mask_payload.get("action_order") != list(ACTION_NAMES):
            raise RecurrentCounterfactualAuxiliaryError(
                "public history action ordering drifted"
            )
        action_mask_values = action_mask_payload.get("values")
        if (
            not isinstance(action_mask_values, list)
            or len(action_mask_values) != ACTION_COUNT
            or any(type(value) is not bool for value in action_mask_values)
            or not any(action_mask_values)
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                "public history action-mask values drifted"
            )
        action_mask = torch.tensor(
            action_mask_values,
            dtype=torch.bool,
            device=reference.device,
        ).reshape(1, 1, -1)
        previous_feedback = torch.tensor(
            feedback_values,
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, -1)
        state = model.forward_sequence(
            observation,
            action_mask,
            previous_feedback,
            initial_state=state,
        ).final_state
    return state


def _behavior_probability_tensor(
    row: Mapping[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    labels = _mapping(row.get("labels"), field="labels")
    distribution = _mapping(
        labels.get("source_behavior_distribution"),
        field="source behavior distribution",
    )
    probabilities = _mapping(
        distribution.get("probabilities"),
        field="source behavior probabilities",
    )
    return torch.tensor(
        [
            _finite_number(
                probabilities.get(action),
                field=f"source behavior probability {action}",
            )
            for action in ACTION_NAMES
        ],
        device=device,
        dtype=dtype,
    )


def _scalarized_action_values(
    rows: Sequence[Mapping[str, object]],
    *,
    config: RecurrentCounterfactualAuxiliaryConfig,
    action_mask: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    scalarized = torch.zeros(ACTION_COUNT, device=device, dtype=dtype)
    pure_returns = torch.zeros(ACTION_COUNT, device=device, dtype=dtype)
    outcome_weights = config.scalarization.outcome_weights()
    for row, (expected_horizon, horizon_weight) in zip(
        rows,
        config.scalarization.horizon_weights,
        strict=True,
    ):
        metadata = _mapping(row.get("metadata"), field="metadata")
        if metadata.get("horizon_ticks") != expected_horizon:
            raise RecurrentCounterfactualAuxiliaryError(
                "ordered horizon row does not match scalarization"
            )
        labels = _mapping(row.get("labels"), field="labels")
        outcomes = labels.get("action_outcomes")
        if not isinstance(outcomes, list):
            raise RecurrentCounterfactualAuxiliaryError(
                "action outcomes must be a list"
            )
        for outcome in outcomes:
            parsed = _mapping(outcome, field="action outcome")
            action = parsed.get("action")
            if not isinstance(action, str) or action not in ACTION_NAMES:
                raise RecurrentCounterfactualAuxiliaryError(
                    "action outcome does not use a stable action"
                )
            index = ACTION_NAMES.index(action)
            terminal = _mapping(
                parsed.get("focal_terminal"),
                field="focal terminal",
            )
            terminal_alive = terminal.get("alive")
            if type(terminal_alive) is not bool:
                raise RecurrentCounterfactualAuxiliaryError(
                    "focal terminal alive must be an exact boolean"
                )
            focal_return = _finite_number(
                parsed.get("focal_discounted_return"),
                field="focal discounted return",
            )
            vector = {
                "focal_discounted_return": focal_return,
                "focal_terminal_alive": float(terminal_alive),
                "population_alive": _finite_number(
                    parsed.get("population_alive"),
                    field="population alive",
                ),
                "births_during_horizon": _finite_number(
                    parsed.get("births_during_horizon"),
                    field="births during horizon",
                ),
                "deaths_during_horizon": _finite_number(
                    parsed.get("deaths_during_horizon"),
                    field="deaths during horizon",
                ),
            }
            score = math.fsum(
                outcome_weights[name] * value for name, value in vector.items()
            )
            scalarized[index] += float(horizon_weight) * score
            pure_returns[index] += float(horizon_weight) * focal_return
    scalarized = torch.where(action_mask, scalarized, torch.zeros_like(scalarized))
    pure_returns = torch.where(
        action_mask, pure_returns, torch.zeros_like(pure_returns)
    )
    return scalarized, pure_returns


def _maybe_permute_scalarized_action_values(
    values: Tensor,
    *,
    action_mask: Tensor,
    config: RecurrentCounterfactualAuxiliaryConfig,
    permutation_identity_sha256: str,
) -> Tensor:
    if (
        config.target_permutation_mode
        == RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
    ):
        return values
    if config.target_permutation_mode != (
        RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
    ):
        raise AssertionError("validated target permutation mode drifted")
    valid_indices = torch.nonzero(action_mask, as_tuple=False).flatten().tolist()
    if len(valid_indices) < 2:
        raise RecurrentCounterfactualAuxiliaryError(
            "target permutation requires at least two currently valid actions"
        )
    seed = config.target_permutation_seed
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AssertionError("validated target permutation seed drifted")
    if (
        not isinstance(permutation_identity_sha256, str)
        or len(permutation_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in permutation_identity_sha256
        )
    ):
        raise RecurrentCounterfactualAuxiliaryError(
            "target permutation group identity must be a lowercase SHA256"
        )
    shuffled = list(valid_indices)
    derived_seed = int(permutation_identity_sha256[:16], 16)
    random.Random(derived_seed).shuffle(shuffled)
    if shuffled == valid_indices:
        shuffled = valid_indices[1:] + valid_indices[:1]
    permuted = values.clone()
    source_indices = torch.tensor(
        shuffled,
        dtype=torch.long,
        device=values.device,
    )
    destination_indices = torch.tensor(
        valid_indices,
        dtype=torch.long,
        device=values.device,
    )
    permuted[destination_indices] = values[source_indices]
    return torch.where(action_mask, permuted, torch.zeros_like(permuted))


def _terminal_value_target(
    rows: Sequence[Mapping[str, object]],
    *,
    behavior: Tensor,
    pure_return_values: Tensor,
    config: RecurrentCounterfactualAuxiliaryConfig,
) -> Tensor | None:
    if config.value_target_mode == RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED:
        return None
    if config.value_target_mode != RECURRENT_COUNTERFACTUAL_TERMINAL_VALUE_TARGET:
        raise AssertionError("validated value target mode drifted")
    for row in rows:
        labels = _mapping(row.get("labels"), field="labels")
        outcomes = labels.get("action_outcomes")
        if not isinstance(outcomes, list):
            raise RecurrentCounterfactualAuxiliaryError(
                "action outcomes must be a list"
            )
        for outcome in outcomes:
            terminal = _mapping(
                _mapping(outcome, field="action outcome").get("focal_terminal"),
                field="focal terminal",
            )
            if terminal.get("alive") is not False:
                raise RecurrentCounterfactualAuxiliaryError(
                    "critic auxiliary target requires every branch return to be terminal"
                )
    return torch.sum(behavior * pure_return_values)


def _forward_kl(target: Tensor, current: Tensor) -> Tensor:
    support = target > 0.0
    if not bool(support.any().item()):
        raise RecurrentCounterfactualAuxiliaryError(
            "KL target has no positive-probability support"
        )
    target_support = target[support]
    current_support = current[support]
    if not bool((current_support > 0.0).all().item()):
        raise RecurrentCounterfactualAuxiliaryError(
            "current actor assigns zero probability to target support"
        )
    return torch.sum(
        target_support * (torch.log(target_support) - torch.log(current_support))
    )


def _require_probability_match(
    observed: Tensor,
    expected: Tensor,
    *,
    action_mask: Tensor,
) -> None:
    _require_probability_vector(
        observed,
        action_mask=action_mask,
        field="current actor probabilities",
    )
    _require_probability_vector(
        expected,
        action_mask=action_mask,
        field="source behavior probabilities",
    )
    if not bool(torch.allclose(observed, expected, rtol=0.0, atol=2.0e-6)):
        raise RecurrentCounterfactualAuxiliaryError(
            "current-model public-history reconstruction does not reproduce source behavior"
        )


def _require_probability_vector(
    value: Tensor,
    *,
    action_mask: Tensor,
    field: str,
) -> None:
    if value.shape != (ACTION_COUNT,) or action_mask.shape != (ACTION_COUNT,):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} shape drifted")
    if not bool(torch.isfinite(value).all().item()):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} is non-finite")
    if not bool((value >= 0.0).all().item()):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} is negative")
    if not bool((value[~action_mask] == 0.0).all().item()):
        raise RecurrentCounterfactualAuxiliaryError(
            f"{field} assigns mass to a masked action"
        )
    if not math.isclose(
        float(value.sum().item()),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-6,
    ):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} does not sum to one")


def _require_finite_masked_vector(
    value: Tensor,
    *,
    action_mask: Tensor,
    field: str,
) -> None:
    if value.shape != (ACTION_COUNT,):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} shape drifted")
    if not bool(torch.isfinite(value).all().item()):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} is non-finite")
    if not bool((value[~action_mask] == 0.0).all().item()):
        raise RecurrentCounterfactualAuxiliaryError(
            f"{field} must be zero outside the exact public action mask"
        )


def _normalized_row_groups(
    row_groups: Sequence[Sequence[Mapping[str, object]]],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    if (
        isinstance(row_groups, (str, bytes))
        or not isinstance(row_groups, Sequence)
        or not row_groups
    ):
        raise RecurrentCounterfactualAuxiliaryError(
            "row_groups must be a non-empty sequence of branch-row groups"
        )
    normalized: list[tuple[Mapping[str, object], ...]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for group_index, group in enumerate(row_groups):
        if (
            isinstance(group, (str, bytes))
            or not isinstance(group, Sequence)
            or not group
        ):
            raise RecurrentCounterfactualAuxiliaryError(
                f"row group {group_index} must be a non-empty sequence"
            )
        rows: list[Mapping[str, object]] = []
        exact_digests: list[str] = []
        for row_index, row in enumerate(group):
            if not isinstance(row, Mapping):
                raise RecurrentCounterfactualAuxiliaryError(
                    f"row group {group_index} row {row_index} must be a mapping"
                )
            rows.append(row)
            exact_digests.append(
                _nonempty_string(row.get("exact_digest"), field="row exact digest")
            )
        group_identity = tuple(sorted(exact_digests))
        if group_identity in seen_groups:
            raise RecurrentCounterfactualAuxiliaryError(
                "one auxiliary batch cannot contain a duplicate branch-row group"
            )
        seen_groups.add(group_identity)
        normalized.append(tuple(rows))
    return tuple(normalized)


def _public_prefix_length(row: Mapping[str, object]) -> int:
    context = _mapping(
        row.get("trainable_public_context"),
        field="trainable_public_context",
    )
    prefix = _mapping(
        context.get("public_history_prefix"),
        field="public history prefix",
    )
    records = prefix.get("records")
    if not isinstance(records, list) or prefix.get("record_count") != len(records):
        raise RecurrentCounterfactualAuxiliaryError(
            "public history prefix record count drifted"
        )
    return len(records)


def _require_finite_gradients(model: PublicRecurrentActorCritic) -> None:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary gradients are non-finite"
            )
    if not found:
        raise RecurrentCounterfactualAuxiliaryError(
            "auxiliary backward pass produced no gradients"
        )


def _gradient_norm(model: PublicRecurrentActorCritic) -> float:
    squared_norms: list[float] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = float(torch.linalg.vector_norm(parameter.grad.detach()).item())
        if not math.isfinite(norm):
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary gradient norm is non-finite"
            )
        squared_norms.append(norm * norm)
    result = math.sqrt(math.fsum(squared_norms))
    if not math.isfinite(result):
        raise RecurrentCounterfactualAuxiliaryError(
            "auxiliary gradient norm is non-finite"
        )
    return result


def _require_finite_model_and_optimizer(
    model: PublicRecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    for name, value in model.state_dict().items():
        if not bool(torch.isfinite(value).all().item()):
            raise RecurrentCounterfactualAuxiliaryError(
                f"post-step model tensor {name!r} is non-finite"
            )
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, Tensor) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise RecurrentCounterfactualAuxiliaryError(
                    f"post-step Adam state {name!r} is non-finite"
                )


def _restore_training_transaction(
    model: PublicRecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    model_snapshot: Mapping[str, Tensor],
    optimizer_snapshot: Mapping[str, object],
    gradient_snapshot: Sequence[Tensor | None],
    learning_rate_snapshot: Sequence[object],
) -> None:
    model.load_state_dict(model_snapshot)
    optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
    for parameter_group, learning_rate in zip(
        optimizer.param_groups,
        learning_rate_snapshot,
        strict=True,
    ):
        parameter_group["lr"] = copy.deepcopy(learning_rate)
    for parameter, gradient in zip(
        model.parameters(),
        gradient_snapshot,
        strict=True,
    ):
        parameter.grad = None if gradient is None else gradient.detach().clone()


def _model_delta_l2(
    model: PublicRecurrentActorCritic,
    snapshot: Mapping[str, Tensor],
) -> float:
    squared_norms: list[float] = []
    for name, value in model.state_dict().items():
        prior = snapshot.get(name)
        if not isinstance(prior, Tensor):
            raise RecurrentCounterfactualAuxiliaryError(
                "model snapshot tensor set drifted"
            )
        delta = float(torch.linalg.vector_norm(value.detach() - prior).item())
        if not math.isfinite(delta):
            raise RecurrentCounterfactualAuxiliaryError(
                "auxiliary parameter delta is non-finite"
            )
        squared_norms.append(delta * delta)
    return math.sqrt(math.fsum(squared_norms))


def _nested_state_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _nested_state_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return bool(left == right)


def _finite_positive_learning_rate(value: object) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise RecurrentCounterfactualAuxiliaryError(
                "Adam parameter-group learning rates must be scalar"
            )
        parsed = float(value.detach().item())
    else:
        parsed = _finite_number(value, field="Adam parameter-group learning rate")
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RecurrentCounterfactualAuxiliaryError(
            "Adam parameter-group learning rates must be finite and positive"
        )
    return parsed


def _learning_rate_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return _nested_state_equal(left, right)
    return type(left) is type(right) and left == right


def _finite_tensor_scalar(value: Tensor, *, field: str) -> float:
    if value.numel() != 1:
        raise RecurrentCounterfactualAuxiliaryError(f"{field} must be scalar")
    parsed = float(value.detach().item())
    if not math.isfinite(parsed):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} must be finite")
    return parsed


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} must be a mapping")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecurrentCounterfactualAuxiliaryError(
            f"{field} must be a non-empty string"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentCounterfactualAuxiliaryError(
            f"{field} must be a positive integer"
        )
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentCounterfactualAuxiliaryError(f"{field} must be finite")
    return parsed


__all__ = [
    "CounterfactualHorizonScalarization",
    "RECURRENT_COUNTERFACTUAL_AUXILIARY_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED",
    "RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS",
    "RECURRENT_COUNTERFACTUAL_TERMINAL_VALUE_TARGET",
    "RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED",
    "RecurrentCounterfactualAuxiliaryBatchLoss",
    "RecurrentCounterfactualAuxiliaryConfig",
    "RecurrentCounterfactualAuxiliaryError",
    "RecurrentCounterfactualAuxiliaryLoss",
    "RecurrentCounterfactualAuxiliaryStepConfig",
    "RecurrentCounterfactualAuxiliaryStepDiagnostics",
    "RecurrentCounterfactualAuxiliaryTarget",
    "build_recurrent_counterfactual_auxiliary_target",
    "recurrent_counterfactual_auxiliary_batch_loss",
    "recurrent_counterfactual_auxiliary_bundle_digest",
    "recurrent_counterfactual_auxiliary_loss",
]
