from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from evolution_sim.mind.recurrent_actor_critic import (
    ACTION_COUNT,
    PublicRecurrentActorCritic,
    validate_action_mask_tensor,
    validate_previous_feedback_tensor,
    validate_public_input_tensor,
)
from evolution_sim.mind.recurrent_rollout import (
    RecurrentAdvantageRow,
    RecurrentRolloutBuffer,
    RecurrentRolloutStep,
)

if TYPE_CHECKING:
    from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
        RecurrentCounterfactualAuxiliaryConfig,
        RecurrentCounterfactualAuxiliaryStepConfig,
        RecurrentCounterfactualAuxiliaryStepDiagnostics,
    )


RECURRENT_PPO_CONTRACT_VERSION = "mind_public_recurrent_ppo_v1"


class RecurrentPPOError(ValueError):
    """Raised when a PPO input or update violates the fail-closed contract."""


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    learning_rate: float = 3.0e-4
    adam_epsilon: float = 1.0e-8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    update_epochs: int = 4
    sequence_minibatch_size: int = 8
    tbptt_steps: int = 32
    burn_in_steps: int = 8
    max_gradient_norm: float = 0.5
    normalize_advantages: bool = True
    advantage_epsilon: float = 1.0e-8
    target_kl: float | None = None
    learner_seed: int = 0
    feed_forward_history_ablation: bool = False
    world_balanced_loss: bool = False

    def __post_init__(self) -> None:
        _positive_float(self.learning_rate, field="learning_rate")
        _positive_float(self.adam_epsilon, field="adam_epsilon")
        _unit_interval(self.gamma, field="gamma", allow_zero=True)
        _unit_interval(self.gae_lambda, field="gae_lambda", allow_zero=True)
        _unit_interval(
            self.policy_clip_range,
            field="policy_clip_range",
            allow_zero=False,
        )
        _positive_float(self.value_clip_range, field="value_clip_range")
        _nonnegative_float(
            self.value_loss_coefficient,
            field="value_loss_coefficient",
        )
        _nonnegative_float(
            self.entropy_coefficient,
            field="entropy_coefficient",
        )
        _positive_int(self.update_epochs, field="update_epochs")
        _positive_int(
            self.sequence_minibatch_size,
            field="sequence_minibatch_size",
        )
        _positive_int(self.tbptt_steps, field="tbptt_steps")
        _nonnegative_int(self.burn_in_steps, field="burn_in_steps")
        _positive_float(self.max_gradient_norm, field="max_gradient_norm")
        if type(self.normalize_advantages) is not bool:
            raise RecurrentPPOError("normalize_advantages must be an exact boolean")
        _positive_float(self.advantage_epsilon, field="advantage_epsilon")
        if self.target_kl is not None:
            _positive_float(self.target_kl, field="target_kl")
        if (
            isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or self.learner_seed < 0
            or self.learner_seed > 2**63 - 1
        ):
            raise RecurrentPPOError("learner_seed must be an integer in [0, 2**63 - 1]")
        if type(self.feed_forward_history_ablation) is not bool:
            raise RecurrentPPOError(
                "feed_forward_history_ablation must be an exact boolean"
            )
        if type(self.world_balanced_loss) is not bool:
            raise RecurrentPPOError("world_balanced_loss must be an exact boolean")


@dataclass(frozen=True, slots=True)
class PPOTrainingSequence:
    """One ordered agent episode with behavior-policy statistics frozen.

    ``recurrent_states[t]`` is the behavior policy's hidden state immediately
    before decision ``t``. It is used only as the detached initial state of a
    TBPTT chunk; rows within a chunk are always evaluated in temporal order.
    """

    world_id: str
    observations: Tensor
    action_masks: Tensor
    previous_feedback: Tensor
    recurrent_states: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    return_targets: Tensor
    episode_starts: Tensor

    @property
    def length(self) -> int:
        if not isinstance(self.observations, Tensor) or self.observations.ndim == 0:
            return 0
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class PPOUpdateDiagnostics:
    contract_version: str
    update_index: int
    sequence_count: int
    transition_count: int
    chunk_count: int
    burn_in_steps: int
    burn_in_transition_count: int
    burn_in_reconstructed_chunk_count: int
    burn_in_state_delta_mean: float
    minibatch_count: int
    epochs_completed: int
    early_stopped_for_kl: bool
    feed_forward_history_ablation: bool
    world_balanced_loss: bool
    world_count: int
    min_world_transition_count: int
    max_world_transition_count: int
    min_effective_world_total_weight: float
    max_effective_world_total_weight: float
    mean_transition_loss_weight: float
    old_statistics_frozen: bool
    advantage_mean: float
    advantage_std: float
    initial_policy_objective: float
    final_policy_objective: float
    policy_objective_delta: float
    final_total_loss: float
    final_policy_loss: float
    final_value_loss: float
    final_entropy: float
    approximate_kl: float
    policy_clip_fraction: float
    value_clip_fraction: float
    explained_variance: float
    gradient_norm_mean: float
    gradient_norm_max: float
    gradient_clip_fraction: float
    parameter_delta_l2: float
    minibatch_order_sha256: str


@dataclass(frozen=True, slots=True)
class _RecurrentPPOTrainerTransactionSnapshot:
    """Exact mutable trainer state at a whole-experiment update boundary."""

    model_state: dict[str, Tensor]
    optimizer_state: dict[str, object]
    update_index: int
    counterfactual_auxiliary_update_count: int
    attempted_counterfactual_auxiliary_bundles: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ChunkReference:
    sequence_index: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _EvaluatedRows:
    new_log_probs: Tensor
    new_values: Tensor
    entropy: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    return_targets: Tensor
    loss_weights: Tensor
    burn_in_transition_count: int
    burn_in_reconstructed_chunk_count: int
    burn_in_state_delta_sum: float


@dataclass(frozen=True, slots=True)
class _LossTerms:
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    policy_clip_fraction: Tensor
    value_clip_fraction: Tensor


@dataclass(frozen=True, slots=True)
class _WorldLossWeighting:
    transition_counts: dict[str, int]
    transition_weights: dict[str, float]
    effective_total_weights: dict[str, float]
    mean_transition_weight: float


def recurrent_ppo_contract(
    config: RecurrentPPOConfig | None = None,
) -> dict[str, object]:
    resolved = config or RecurrentPPOConfig()
    return {
        "schema_version": RECURRENT_PPO_CONTRACT_VERSION,
        "optimizer": "adam",
        "algorithm": "parameter_shared_recurrent_ippo_clipped_ppo",
        "gae": {
            "gamma": resolved.gamma,
            "lambda": resolved.gae_lambda,
            "terminated_bootstrap": 0.0,
            "truncated_bootstrap": "required_from_frozen_behavior_value",
        },
        "sequence_batching": {
            "variable_length": True,
            "row_shuffle": False,
            "tbptt_steps": resolved.tbptt_steps,
            "burn_in_steps": resolved.burn_in_steps,
            "minibatch_unit": "ordered_contiguous_sequence_chunk",
            "minibatch_order": "learner_seed_deterministic",
            "burn_in": (
                "preceding_prefix_recomputed_with_current_model_without_loss_or_gradient"
            ),
            "burn_in_initial_state": (
                "stored_behavior_hidden_at_prefix_start_or_zero_at_episode_start"
            ),
        },
        "likelihood_contract": {
            "action_masks": "stored_observation_time_masks_only",
            "old_log_probs": "frozen_behavior_policy_values",
            "old_values": "frozen_behavior_policy_values",
            "resolution_masks_used_for_likelihood": False,
        },
        "losses": {
            "policy": "clipped_surrogate",
            "value": "max_unclipped_and_clipped_squared_error",
            "entropy_coefficient": resolved.entropy_coefficient,
            "value_loss_coefficient": resolved.value_loss_coefficient,
            "advantage_normalization": resolved.normalize_advantages,
        },
        "world_balancing": {
            "enabled": resolved.world_balanced_loss,
            "default_enabled": False,
            "group_key": "world_id",
            "transition_weight": (
                "total_transitions / (world_count * world_transition_count)"
            ),
            "normalization": "global_mean_transition_weight_one",
            "applies_to": (
                "advantage_normalization_policy_value_entropy_kl_clip_"
                "diagnostics_and_explained_variance"
            ),
        },
        "history": {
            "primary": "stored_behavior_hidden_at_chunk_start_then_recurrent_tbptt",
            "feed_forward_ablation": "reset_hidden_before_every_decision",
            "feed_forward_ablation_enabled": (resolved.feed_forward_history_ablation),
        },
        "fail_closed": {
            "nonfinite_inputs_losses_gradients_parameters_or_optimizer_state": True,
            "transactional_restore_on_failure": True,
            "illegal_behavior_action": True,
            "empty_action_mask": True,
        },
        "runtime_integrated": False,
        "hardcoded_action_selection": False,
        "private_world_features": False,
    }


def ppo_sequences_from_rollout_buffer(
    buffer: RecurrentRolloutBuffer,
    *,
    model: PublicRecurrentActorCritic,
    config: RecurrentPPOConfig,
) -> tuple[PPOTrainingSequence, ...]:
    """Freeze a closed rollout buffer into per-agent PPO sequences."""

    if not isinstance(buffer, RecurrentRolloutBuffer):
        raise RecurrentPPOError("buffer must be a RecurrentRolloutBuffer")
    if not isinstance(model, PublicRecurrentActorCritic):
        raise RecurrentPPOError("model must be a PublicRecurrentActorCritic")
    if not isinstance(config, RecurrentPPOConfig):
        raise RecurrentPPOError("config must be a RecurrentPPOConfig")

    advantage_rows = buffer.compute_gae(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    row_by_key: dict[tuple[str, int, int], RecurrentAdvantageRow] = {}
    for row in advantage_rows:
        key = _step_key(row.step)
        if key in row_by_key:
            raise RecurrentPPOError(f"duplicate rollout decision key: {key!r}")
        row_by_key[key] = row

    reference = next(model.parameters())
    recurrent_layers = model.config.recurrent_layers
    hidden_size = model.config.hidden_size
    expected_hidden_values = recurrent_layers * hidden_size
    sequences: list[PPOTrainingSequence] = []
    consumed_keys: set[tuple[str, int, int]] = set()
    for steps in buffer.sequences():
        if not steps:
            continue
        world_id = steps[0].world_id
        if any(step.world_id != world_id for step in steps):
            raise RecurrentPPOError(
                "one agent sequence cannot span multiple rollout worlds"
            )
        rows: list[RecurrentAdvantageRow] = []
        for step in steps:
            key = _step_key(step)
            row = row_by_key.get(key)
            if row is None:
                raise RecurrentPPOError(
                    f"rollout sequence is missing GAE row for {key!r}"
                )
            consumed_keys.add(key)
            rows.append(row)
            if len(step.hidden) != expected_hidden_values:
                raise RecurrentPPOError(
                    "stored behavior hidden state has unexpected size: "
                    f"{len(step.hidden)} != {expected_hidden_values}"
                )
            if any(type(value) is not bool for value in step.action_mask):
                raise RecurrentPPOError(
                    "stored observation-time action mask must contain exact booleans"
                )

        length = len(rows)
        observations = torch.tensor(
            [row.step.observation for row in rows],
            dtype=reference.dtype,
            device=reference.device,
        )
        action_masks = torch.tensor(
            [row.step.action_mask for row in rows],
            dtype=torch.bool,
            device=reference.device,
        )
        previous_feedback = torch.tensor(
            [row.step.previous_feedback.vector() for row in rows],
            dtype=reference.dtype,
            device=reference.device,
        )
        recurrent_states = torch.tensor(
            [row.step.hidden for row in rows],
            dtype=reference.dtype,
            device=reference.device,
        ).reshape(length, recurrent_layers, hidden_size)
        episode_starts = torch.zeros(
            length,
            dtype=torch.bool,
            device=reference.device,
        )
        episode_starts[0] = True
        sequences.append(
            PPOTrainingSequence(
                world_id=world_id,
                observations=observations,
                action_masks=action_masks,
                previous_feedback=previous_feedback,
                recurrent_states=recurrent_states,
                actions=torch.tensor(
                    [row.step.action_index for row in rows],
                    dtype=torch.long,
                    device=reference.device,
                ),
                old_log_probs=torch.tensor(
                    [row.step.logprob for row in rows],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                old_values=torch.tensor(
                    [row.step.value for row in rows],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                advantages=torch.tensor(
                    [row.advantage for row in rows],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                return_targets=torch.tensor(
                    [row.return_target for row in rows],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                episode_starts=episode_starts,
            )
        )

    if consumed_keys != set(row_by_key):
        missing = sorted(set(row_by_key) - consumed_keys)
        raise RecurrentPPOError(
            f"GAE rows were not assigned to an agent sequence: {missing[:8]!r}"
        )
    if not sequences:
        raise RecurrentPPOError("rollout buffer contains no trainable decisions")
    return tuple(sequences)


class RecurrentPPOTrainer:
    """Transactional sequence-preserving recurrent IPPO/PPO optimizer."""

    def __init__(
        self,
        model: PublicRecurrentActorCritic,
        config: RecurrentPPOConfig | None = None,
    ) -> None:
        if not isinstance(model, PublicRecurrentActorCritic):
            raise RecurrentPPOError("model must be a PublicRecurrentActorCritic")
        self.model = model
        self.config = config or RecurrentPPOConfig()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )
        self._update_index = 0
        self._counterfactual_auxiliary_update_count = 0
        self._attempted_counterfactual_auxiliary_bundles: set[str] = set()
        _assert_finite_model(self.model)

    @property
    def update_index(self) -> int:
        return self._update_index

    @property
    def counterfactual_auxiliary_update_count(self) -> int:
        return self._counterfactual_auxiliary_update_count

    def _snapshot_experiment_transaction(
        self,
    ) -> _RecurrentPPOTrainerTransactionSnapshot:
        """Capture all state mutated by PPO and its optional auxiliary stage.

        The experiment runner uses this private boundary to make one scheduled
        update atomic.  The auxiliary's one-use evidence ledger is included so
        restore can distinguish pre-existing evidence from evidence consumed by
        the failed transaction.
        """

        _assert_finite_model(self.model)
        return _RecurrentPPOTrainerTransactionSnapshot(
            model_state={
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            },
            optimizer_state=copy.deepcopy(self.optimizer.state_dict()),
            update_index=self._update_index,
            counterfactual_auxiliary_update_count=(
                self._counterfactual_auxiliary_update_count
            ),
            attempted_counterfactual_auxiliary_bundles=frozenset(
                self._attempted_counterfactual_auxiliary_bundles
            ),
        )

    def _restore_experiment_transaction(
        self,
        snapshot: _RecurrentPPOTrainerTransactionSnapshot,
        *,
        preserve_new_evidence_attempts: bool = True,
    ) -> None:
        """Restore a whole-update snapshot without laundering used evidence."""

        if not isinstance(snapshot, _RecurrentPPOTrainerTransactionSnapshot):
            raise RecurrentPPOError("invalid experiment transaction snapshot")
        if type(preserve_new_evidence_attempts) is not bool:
            raise RecurrentPPOError(
                "preserve_new_evidence_attempts must be an exact boolean"
            )
        attempted_after_snapshot = set(self._attempted_counterfactual_auxiliary_bundles)
        self.model.load_state_dict(snapshot.model_state, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer_state))
        self.optimizer.zero_grad(set_to_none=True)
        self._update_index = snapshot.update_index
        self._counterfactual_auxiliary_update_count = (
            snapshot.counterfactual_auxiliary_update_count
        )
        restored_attempts = set(snapshot.attempted_counterfactual_auxiliary_bundles)
        if preserve_new_evidence_attempts:
            restored_attempts.update(attempted_after_snapshot)
        self._attempted_counterfactual_auxiliary_bundles = restored_attempts
        _assert_finite_model(self.model)

    def counterfactual_auxiliary_update(
        self,
        row_groups: Sequence[Sequence[Mapping[str, object]]],
        *,
        artifact_digest: str,
        config: RecurrentCounterfactualAuxiliaryConfig,
        step_config: RecurrentCounterfactualAuxiliaryStepConfig | None = None,
    ) -> RecurrentCounterfactualAuxiliaryStepDiagnostics:
        """Attempt one non-retryable auxiliary step on the shared PPO Adam.

        Every exact row bundle is one-use for this trainer, regardless of
        acceptance.  A rejected transaction restores the rows' source model
        exactly, but the caller must collect or explicitly construct a fresh
        bundle instead of silently tuning and retrying the same evidence.
        """

        from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
            RecurrentCounterfactualAuxiliaryError,
            _transactional_recurrent_counterfactual_auxiliary_step,
            recurrent_counterfactual_auxiliary_bundle_digest,
        )

        bundle_digest = recurrent_counterfactual_auxiliary_bundle_digest(
            row_groups,
            artifact_digest=artifact_digest,
        )
        if bundle_digest in self._attempted_counterfactual_auxiliary_bundles:
            raise RecurrentCounterfactualAuxiliaryError(
                "counterfactual branch-row bundle was already attempted; retry is forbidden"
            )
        self._attempted_counterfactual_auxiliary_bundles.add(bundle_digest)
        update_index_snapshot = self._update_index
        auxiliary_count_snapshot = self._counterfactual_auxiliary_update_count
        try:
            diagnostics = _transactional_recurrent_counterfactual_auxiliary_step(
                self.model,
                self.optimizer,
                row_groups,
                artifact_digest=artifact_digest,
                config=config,
                step_config=step_config,
            )
        except Exception:
            self._update_index = update_index_snapshot
            self._counterfactual_auxiliary_update_count = auxiliary_count_snapshot
            raise
        if diagnostics.accepted:
            self._counterfactual_auxiliary_update_count += 1
            counters_restored = False
        else:
            self._update_index = update_index_snapshot
            self._counterfactual_auxiliary_update_count = auxiliary_count_snapshot
            counters_restored = True
        return replace(
            diagnostics,
            ppo_update_index=self._update_index,
            auxiliary_update_count=self._counterfactual_auxiliary_update_count,
            update_counters_restored=counters_restored,
        )

    def update(
        self,
        sequences: Sequence[PPOTrainingSequence],
    ) -> PPOUpdateDiagnostics:
        """Run one transactional PPO update over frozen behavior sequences."""

        if not isinstance(sequences, Sequence) or isinstance(sequences, (str, bytes)):
            raise RecurrentPPOError("sequences must be an ordered sequence")
        if not sequences:
            raise RecurrentPPOError("PPO update requires at least one sequence")
        _assert_finite_model(self.model)
        model_snapshot = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        try:
            diagnostics = self._update_impl(sequences, model_snapshot)
        except Exception as exc:
            self.model.load_state_dict(model_snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            if isinstance(exc, RecurrentPPOError):
                raise
            raise RecurrentPPOError(
                "PPO update failed closed and was restored"
            ) from exc
        self._update_index += 1
        return diagnostics

    def _update_impl(
        self,
        sequences: Sequence[PPOTrainingSequence],
        model_snapshot: dict[str, Tensor],
    ) -> PPOUpdateDiagnostics:
        frozen = tuple(
            _freeze_and_validate_sequence(sequence, model=self.model)
            for sequence in sequences
        )
        world_weighting = _world_loss_weighting(
            frozen,
            enabled=self.config.world_balanced_loss,
        )
        normalized, advantage_mean, advantage_std = _normalize_advantages(
            frozen,
            config=self.config,
            world_weighting=world_weighting,
        )
        chunks = _chunk_references(normalized, tbptt_steps=self.config.tbptt_steps)
        if not chunks:
            raise RecurrentPPOError("PPO update produced no sequence chunks")

        with torch.no_grad():
            initial_rows = self._evaluate_chunks(
                chunks,
                normalized,
                world_weighting=world_weighting,
            )
            initial_terms = _loss_terms(initial_rows, config=self.config)
            initial_objective = -_finite_scalar(
                initial_terms.policy_loss,
                field="initial policy objective",
            )

        rng = random.Random(self.config.learner_seed + self._update_index)
        order_digest = hashlib.sha256()
        minibatch_count = 0
        epochs_completed = 0
        early_stopped = False
        gradient_norms: list[float] = []
        gradient_clipped = 0
        for epoch in range(self.config.update_epochs):
            epoch_chunks = list(chunks)
            rng.shuffle(epoch_chunks)
            epochs_completed = epoch + 1
            for offset in range(
                0,
                len(epoch_chunks),
                self.config.sequence_minibatch_size,
            ):
                minibatch = tuple(
                    epoch_chunks[offset : offset + self.config.sequence_minibatch_size]
                )
                order_digest.update(
                    (
                        f"epoch={epoch};batch={minibatch_count};"
                        + ",".join(
                            f"{chunk.sequence_index}:{chunk.start}:{chunk.stop}"
                            for chunk in minibatch
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                evaluated = self._evaluate_chunks(
                    minibatch,
                    normalized,
                    world_weighting=world_weighting,
                )
                terms = _loss_terms(evaluated, config=self.config)
                self.optimizer.zero_grad(set_to_none=True)
                terms.total_loss.backward()
                _assert_finite_gradients(self.model)
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_gradient_norm,
                    error_if_nonfinite=True,
                )
                gradient_norm = _finite_scalar(
                    gradient_norm_tensor,
                    field="gradient norm",
                )
                gradient_norms.append(gradient_norm)
                if gradient_norm > self.config.max_gradient_norm:
                    gradient_clipped += 1
                self.optimizer.step()
                _assert_finite_model(self.model)
                _assert_finite_optimizer_state(self.optimizer)
                minibatch_count += 1

                approximate_kl = _finite_scalar(
                    terms.approximate_kl,
                    field="approximate KL",
                )
                if (
                    self.config.target_kl is not None
                    and approximate_kl > self.config.target_kl
                ):
                    early_stopped = True
                    break
            if early_stopped:
                break

        with torch.no_grad():
            final_rows = self._evaluate_chunks(
                chunks,
                normalized,
                world_weighting=world_weighting,
            )
            final_terms = _loss_terms(final_rows, config=self.config)
        final_objective = -_finite_scalar(
            final_terms.policy_loss,
            field="final policy objective",
        )
        explained_variance = _explained_variance(
            final_rows.return_targets,
            final_rows.new_values,
            loss_weights=final_rows.loss_weights,
            weighted=self.config.world_balanced_loss,
            epsilon=self.config.advantage_epsilon,
        )
        parameter_delta_l2 = _parameter_delta_l2(
            self.model,
            model_snapshot,
        )
        transition_count = sum(sequence.length for sequence in normalized)
        burn_in_state_delta_mean = (
            final_rows.burn_in_state_delta_sum
            / final_rows.burn_in_reconstructed_chunk_count
            if final_rows.burn_in_reconstructed_chunk_count
            else 0.0
        )
        diagnostics = PPOUpdateDiagnostics(
            contract_version=RECURRENT_PPO_CONTRACT_VERSION,
            update_index=self._update_index,
            sequence_count=len(normalized),
            transition_count=transition_count,
            chunk_count=len(chunks),
            burn_in_steps=self.config.burn_in_steps,
            burn_in_transition_count=final_rows.burn_in_transition_count,
            burn_in_reconstructed_chunk_count=(
                final_rows.burn_in_reconstructed_chunk_count
            ),
            burn_in_state_delta_mean=burn_in_state_delta_mean,
            minibatch_count=minibatch_count,
            epochs_completed=epochs_completed,
            early_stopped_for_kl=early_stopped,
            feed_forward_history_ablation=(self.config.feed_forward_history_ablation),
            world_balanced_loss=self.config.world_balanced_loss,
            world_count=len(world_weighting.transition_counts),
            min_world_transition_count=min(world_weighting.transition_counts.values()),
            max_world_transition_count=max(world_weighting.transition_counts.values()),
            min_effective_world_total_weight=min(
                world_weighting.effective_total_weights.values()
            ),
            max_effective_world_total_weight=max(
                world_weighting.effective_total_weights.values()
            ),
            mean_transition_loss_weight=(world_weighting.mean_transition_weight),
            old_statistics_frozen=True,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            initial_policy_objective=initial_objective,
            final_policy_objective=final_objective,
            policy_objective_delta=final_objective - initial_objective,
            final_total_loss=_finite_scalar(
                final_terms.total_loss,
                field="final total loss",
            ),
            final_policy_loss=_finite_scalar(
                final_terms.policy_loss,
                field="final policy loss",
            ),
            final_value_loss=_finite_scalar(
                final_terms.value_loss,
                field="final value loss",
            ),
            final_entropy=_finite_scalar(
                final_terms.entropy,
                field="final entropy",
            ),
            approximate_kl=_finite_scalar(
                final_terms.approximate_kl,
                field="final approximate KL",
            ),
            policy_clip_fraction=_finite_scalar(
                final_terms.policy_clip_fraction,
                field="final policy clip fraction",
            ),
            value_clip_fraction=_finite_scalar(
                final_terms.value_clip_fraction,
                field="final value clip fraction",
            ),
            explained_variance=explained_variance,
            gradient_norm_mean=(
                sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
            ),
            gradient_norm_max=max(gradient_norms, default=0.0),
            gradient_clip_fraction=(
                gradient_clipped / len(gradient_norms) if gradient_norms else 0.0
            ),
            parameter_delta_l2=parameter_delta_l2,
            minibatch_order_sha256=order_digest.hexdigest(),
        )
        _assert_finite_diagnostics(diagnostics)
        return diagnostics

    def _evaluate_chunks(
        self,
        chunks: Sequence[_ChunkReference],
        sequences: Sequence[PPOTrainingSequence],
        *,
        world_weighting: _WorldLossWeighting,
    ) -> _EvaluatedRows:
        new_log_probs: list[Tensor] = []
        new_values: list[Tensor] = []
        entropies: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        old_values: list[Tensor] = []
        advantages: list[Tensor] = []
        return_targets: list[Tensor] = []
        loss_weights: list[Tensor] = []
        burn_in_transition_count = 0
        burn_in_reconstructed_chunk_count = 0
        burn_in_state_delta_sum = 0.0
        for chunk in chunks:
            sequence = sequences[chunk.sequence_index]
            selected = slice(chunk.start, chunk.stop)
            if self.config.feed_forward_history_ablation:
                transition_count = chunk.stop - chunk.start
                evaluation = self.model.evaluate_sequence(
                    sequence.observations[selected].unsqueeze(0),
                    sequence.action_masks[selected].unsqueeze(0),
                    sequence.previous_feedback[selected].unsqueeze(0),
                    sequence.actions[selected].unsqueeze(0),
                    initial_state=self.model.initial_state(transition_count),
                    episode_starts=torch.zeros(
                        (1, transition_count),
                        dtype=torch.bool,
                        device=sequence.observations.device,
                    ),
                )
                new_log_probs.append(evaluation.log_probs.squeeze(0))
                new_values.append(evaluation.values.squeeze(0))
                entropies.append(evaluation.entropy.squeeze(0))
            else:
                (
                    initial_state,
                    burn_in_count,
                    state_delta,
                ) = self._reconstruct_chunk_initial_state(sequence, chunk)
                burn_in_transition_count += burn_in_count
                if burn_in_count:
                    burn_in_reconstructed_chunk_count += 1
                    burn_in_state_delta_sum += state_delta
                episode_starts = sequence.episode_starts[selected]
                evaluation = self.model.evaluate_sequence(
                    sequence.observations[selected].unsqueeze(1),
                    sequence.action_masks[selected].unsqueeze(1),
                    sequence.previous_feedback[selected].unsqueeze(1),
                    sequence.actions[selected].unsqueeze(1),
                    initial_state=initial_state,
                    episode_starts=episode_starts.unsqueeze(1),
                )
                new_log_probs.append(evaluation.log_probs.squeeze(1))
                new_values.append(evaluation.values.squeeze(1))
                entropies.append(evaluation.entropy.squeeze(1))
            old_log_probs.append(sequence.old_log_probs[selected])
            old_values.append(sequence.old_values[selected])
            advantages.append(sequence.advantages[selected])
            return_targets.append(sequence.return_targets[selected])
            loss_weights.append(
                torch.full(
                    (chunk.stop - chunk.start,),
                    world_weighting.transition_weights[sequence.world_id],
                    dtype=sequence.observations.dtype,
                    device=sequence.observations.device,
                )
            )
        return _EvaluatedRows(
            new_log_probs=torch.cat(new_log_probs),
            new_values=torch.cat(new_values),
            entropy=torch.cat(entropies),
            old_log_probs=torch.cat(old_log_probs),
            old_values=torch.cat(old_values),
            advantages=torch.cat(advantages),
            return_targets=torch.cat(return_targets),
            loss_weights=torch.cat(loss_weights),
            burn_in_transition_count=burn_in_transition_count,
            burn_in_reconstructed_chunk_count=(burn_in_reconstructed_chunk_count),
            burn_in_state_delta_sum=burn_in_state_delta_sum,
        )

    def _reconstruct_chunk_initial_state(
        self,
        sequence: PPOTrainingSequence,
        chunk: _ChunkReference,
    ) -> tuple[Tensor, int, float]:
        if chunk.start == 0:
            return self.model.initial_state(1), 0, 0.0
        behavior_state = sequence.recurrent_states[chunk.start].unsqueeze(1)
        if self.config.burn_in_steps == 0:
            return behavior_state, 0, 0.0

        prefix_start = max(0, chunk.start - self.config.burn_in_steps)
        prefix = slice(prefix_start, chunk.start)
        if prefix_start == 0:
            prefix_state = self.model.initial_state(1)
        else:
            prefix_state = sequence.recurrent_states[prefix_start].unsqueeze(1)
        with torch.no_grad():
            prefix_output = self.model.forward_sequence(
                sequence.observations[prefix].unsqueeze(1),
                sequence.action_masks[prefix].unsqueeze(1),
                sequence.previous_feedback[prefix].unsqueeze(1),
                initial_state=prefix_state,
                episode_starts=sequence.episode_starts[prefix].unsqueeze(1),
            )
            reconstructed = prefix_output.final_state.detach()
        state_delta = _finite_scalar(
            torch.linalg.vector_norm(reconstructed - behavior_state),
            field="burn-in state delta",
        )
        return reconstructed, chunk.start - prefix_start, state_delta


def _freeze_and_validate_sequence(
    sequence: PPOTrainingSequence,
    *,
    model: PublicRecurrentActorCritic,
) -> PPOTrainingSequence:
    if not isinstance(sequence, PPOTrainingSequence):
        raise RecurrentPPOError("every sequence must be a PPOTrainingSequence")
    if (
        not isinstance(sequence.world_id, str)
        or not sequence.world_id
        or sequence.world_id.strip() != sequence.world_id
    ):
        raise RecurrentPPOError(
            "world_id must be a nonempty string without surrounding whitespace"
        )
    reference = next(model.parameters())
    float_fields = (
        "observations",
        "previous_feedback",
        "recurrent_states",
        "old_log_probs",
        "old_values",
        "advantages",
        "return_targets",
    )
    for field_name in float_fields:
        value = getattr(sequence, field_name)
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise RecurrentPPOError(f"{field_name} must be a floating torch.Tensor")
    if (
        not isinstance(sequence.action_masks, Tensor)
        or sequence.action_masks.dtype != torch.bool
    ):
        raise RecurrentPPOError("action_masks must use torch.bool")
    if not isinstance(sequence.actions, Tensor) or sequence.actions.dtype != torch.long:
        raise RecurrentPPOError("actions must use torch.long")
    if (
        not isinstance(sequence.episode_starts, Tensor)
        or sequence.episode_starts.dtype != torch.bool
    ):
        raise RecurrentPPOError("episode_starts must use torch.bool")

    frozen = PPOTrainingSequence(
        world_id=sequence.world_id,
        observations=sequence.observations.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        action_masks=sequence.action_masks.detach()
        .to(
            device=reference.device,
        )
        .clone(),
        previous_feedback=sequence.previous_feedback.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        recurrent_states=sequence.recurrent_states.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        actions=sequence.actions.detach().to(device=reference.device).clone(),
        old_log_probs=sequence.old_log_probs.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        old_values=sequence.old_values.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        advantages=sequence.advantages.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        return_targets=sequence.return_targets.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone(),
        episode_starts=sequence.episode_starts.detach()
        .to(
            device=reference.device,
        )
        .clone(),
    )
    length = frozen.length
    if length <= 0:
        raise RecurrentPPOError("training sequences cannot be empty")
    validate_public_input_tensor(frozen.observations, ranks=(2,))
    validate_action_mask_tensor(frozen.action_masks, leading_shape=(length,))
    validate_previous_feedback_tensor(
        frozen.previous_feedback,
        leading_shape=(length,),
    )
    expected_state_shape = (
        length,
        model.config.recurrent_layers,
        model.config.hidden_size,
    )
    if tuple(frozen.recurrent_states.shape) != expected_state_shape:
        raise RecurrentPPOError(
            "recurrent_states must have shape "
            f"{expected_state_shape}; got {tuple(frozen.recurrent_states.shape)}"
        )
    expected_row_shape = (length,)
    for field_name in (
        "actions",
        "old_log_probs",
        "old_values",
        "advantages",
        "return_targets",
        "episode_starts",
    ):
        value = getattr(frozen, field_name)
        if tuple(value.shape) != expected_row_shape:
            raise RecurrentPPOError(
                f"{field_name} must have shape {expected_row_shape}; "
                f"got {tuple(value.shape)}"
            )
    if not bool(frozen.episode_starts[0].item()):
        raise RecurrentPPOError("each agent sequence must start at an episode boundary")
    if length > 1 and bool(frozen.episode_starts[1:].any().item()):
        raise RecurrentPPOError(
            "one agent sequence cannot contain an internal episode boundary"
        )
    if not bool((frozen.previous_feedback[frozen.episode_starts] == 0.0).all().item()):
        raise RecurrentPPOError("episode-start previous feedback must be all zero")
    if not bool(torch.isfinite(frozen.recurrent_states).all().item()):
        raise RecurrentPPOError("recurrent_states must be finite")
    if not bool((frozen.recurrent_states[0] == 0.0).all().item()):
        raise RecurrentPPOError(
            "the first behavior recurrent state must be the zero episode state"
        )
    for field_name in (
        "old_log_probs",
        "old_values",
        "advantages",
        "return_targets",
    ):
        value = getattr(frozen, field_name)
        if not bool(torch.isfinite(value).all().item()):
            raise RecurrentPPOError(f"{field_name} must be finite")
    if not bool((frozen.old_log_probs <= 1.0e-6).all().item()):
        raise RecurrentPPOError("old_log_probs cannot be positive")
    if not bool(((frozen.actions >= 0) & (frozen.actions < ACTION_COUNT)).all().item()):
        raise RecurrentPPOError("actions contain an id outside the stable action space")
    legal = frozen.action_masks.gather(
        -1,
        frozen.actions.unsqueeze(-1),
    ).squeeze(-1)
    if not bool(legal.all().item()):
        raise RecurrentPPOError(
            "behavior actions must be legal under their stored observation-time masks"
        )
    return frozen


def _world_loss_weighting(
    sequences: tuple[PPOTrainingSequence, ...],
    *,
    enabled: bool,
) -> _WorldLossWeighting:
    transition_counts: dict[str, int] = {}
    for sequence in sequences:
        transition_counts[sequence.world_id] = (
            transition_counts.get(sequence.world_id, 0) + sequence.length
        )
    if not transition_counts:
        raise RecurrentPPOError("world loss weighting requires training sequences")

    total_transitions = sum(transition_counts.values())
    world_count = len(transition_counts)
    if enabled:
        effective_total_per_world = total_transitions / world_count
        transition_weights = {
            world_id: effective_total_per_world / transition_count
            for world_id, transition_count in transition_counts.items()
        }
    else:
        transition_weights = {world_id: 1.0 for world_id in transition_counts}
    effective_total_weights = {
        world_id: transition_counts[world_id] * transition_weights[world_id]
        for world_id in transition_counts
    }
    mean_transition_weight = sum(effective_total_weights.values()) / total_transitions
    if not all(
        math.isfinite(value) and value > 0.0 for value in transition_weights.values()
    ):
        raise RecurrentPPOError(
            "world transition loss weights must be finite and positive"
        )
    if not math.isfinite(mean_transition_weight):
        raise RecurrentPPOError("mean transition loss weight must be finite")
    return _WorldLossWeighting(
        transition_counts=transition_counts,
        transition_weights=transition_weights,
        effective_total_weights=effective_total_weights,
        mean_transition_weight=mean_transition_weight,
    )


def _normalize_advantages(
    sequences: tuple[PPOTrainingSequence, ...],
    *,
    config: RecurrentPPOConfig,
    world_weighting: _WorldLossWeighting,
) -> tuple[tuple[PPOTrainingSequence, ...], float, float]:
    all_advantages = torch.cat([sequence.advantages for sequence in sequences])
    if config.world_balanced_loss:
        all_weights = torch.cat(
            [
                torch.full_like(
                    sequence.advantages,
                    world_weighting.transition_weights[sequence.world_id],
                )
                for sequence in sequences
            ]
        )
        weight_sum = all_weights.sum()
        weighted_mean = (all_advantages * all_weights).sum() / weight_sum
        weighted_variance = (
            all_weights * (all_advantages - weighted_mean).square()
        ).sum() / weight_sum
        mean = _finite_scalar(weighted_mean, field="advantage mean")
        std = _finite_scalar(
            torch.sqrt(weighted_variance),
            field="advantage standard deviation",
        )
    else:
        mean = _finite_scalar(all_advantages.mean(), field="advantage mean")
        std = _finite_scalar(
            all_advantages.std(unbiased=False),
            field="advantage standard deviation",
        )
    if not config.normalize_advantages:
        return sequences, mean, std
    normalized_values = (all_advantages - mean) / max(
        std,
        config.advantage_epsilon,
    )
    if not bool(torch.isfinite(normalized_values).all().item()):
        raise RecurrentPPOError("normalized advantages must be finite")
    normalized: list[PPOTrainingSequence] = []
    offset = 0
    for sequence in sequences:
        stop = offset + sequence.length
        normalized.append(
            replace(
                sequence,
                advantages=normalized_values[offset:stop].clone(),
            )
        )
        offset = stop
    return tuple(normalized), mean, std


def _chunk_references(
    sequences: Sequence[PPOTrainingSequence],
    *,
    tbptt_steps: int,
) -> tuple[_ChunkReference, ...]:
    chunks: list[_ChunkReference] = []
    for sequence_index, sequence in enumerate(sequences):
        for start in range(0, sequence.length, tbptt_steps):
            chunks.append(
                _ChunkReference(
                    sequence_index=sequence_index,
                    start=start,
                    stop=min(start + tbptt_steps, sequence.length),
                )
            )
    return tuple(chunks)


def _loss_terms(
    rows: _EvaluatedRows,
    *,
    config: RecurrentPPOConfig,
) -> _LossTerms:
    for field_name in (
        "new_log_probs",
        "new_values",
        "entropy",
        "old_log_probs",
        "old_values",
        "advantages",
        "return_targets",
        "loss_weights",
    ):
        value = getattr(rows, field_name)
        if not bool(torch.isfinite(value).all().item()):
            raise RecurrentPPOError(f"{field_name} became nonfinite")
    if tuple(rows.loss_weights.shape) != tuple(rows.advantages.shape):
        raise RecurrentPPOError(
            "loss_weights must match the evaluated transition shape"
        )
    if not bool((rows.loss_weights > 0.0).all().item()):
        raise RecurrentPPOError("loss_weights must be positive")

    def loss_mean(values: Tensor) -> Tensor:
        if config.world_balanced_loss:
            return (values * rows.loss_weights).mean()
        return values.mean()

    log_ratio = rows.new_log_probs - rows.old_log_probs
    ratio = torch.exp(log_ratio)
    if not bool(torch.isfinite(ratio).all().item()):
        raise RecurrentPPOError("policy likelihood ratio became nonfinite")
    clipped_ratio = ratio.clamp(
        1.0 - config.policy_clip_range,
        1.0 + config.policy_clip_range,
    )
    surrogate = torch.minimum(
        ratio * rows.advantages,
        clipped_ratio * rows.advantages,
    )
    policy_loss = -loss_mean(surrogate)

    value_delta = rows.new_values - rows.old_values
    clipped_values = rows.old_values + value_delta.clamp(
        -config.value_clip_range,
        config.value_clip_range,
    )
    value_error = (rows.new_values - rows.return_targets).square()
    clipped_value_error = (clipped_values - rows.return_targets).square()
    value_loss = 0.5 * loss_mean(torch.maximum(value_error, clipped_value_error))
    entropy = loss_mean(rows.entropy)
    total_loss = (
        policy_loss
        + config.value_loss_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    approximate_kl = loss_mean((ratio - 1.0) - log_ratio)
    policy_clip_fraction = loss_mean(
        (torch.abs(ratio - 1.0) > config.policy_clip_range).to(rows.new_values.dtype)
    )
    value_clip_fraction = loss_mean(
        (torch.abs(value_delta) > config.value_clip_range).to(rows.new_values.dtype)
    )
    terms = _LossTerms(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        policy_clip_fraction=policy_clip_fraction,
        value_clip_fraction=value_clip_fraction,
    )
    for field_name in terms.__dataclass_fields__:
        _finite_scalar(getattr(terms, field_name), field=field_name)
    return terms


def _explained_variance(
    return_targets: Tensor,
    values: Tensor,
    *,
    loss_weights: Tensor,
    weighted: bool,
    epsilon: float,
) -> float:
    if weighted:
        weight_sum = loss_weights.sum()
        target_mean = (return_targets * loss_weights).sum() / weight_sum
        target_variance = (
            loss_weights * (return_targets - target_mean).square()
        ).sum() / weight_sum
    else:
        target_variance = return_targets.var(unbiased=False)
    target_variance_value = _finite_scalar(
        target_variance,
        field="return-target variance",
    )
    if target_variance_value <= epsilon:
        return 0.0
    residual = return_targets - values
    if weighted:
        residual_mean = (residual * loss_weights).sum() / weight_sum
        residual_variance = (
            loss_weights * (residual - residual_mean).square()
        ).sum() / weight_sum
    else:
        residual_variance = residual.var(unbiased=False)
    return _finite_scalar(
        1.0 - residual_variance / target_variance,
        field="explained variance",
    )


def _parameter_delta_l2(
    model: PublicRecurrentActorCritic,
    snapshot: dict[str, Tensor],
) -> float:
    squared = 0.0
    for key, current in model.state_dict().items():
        previous = snapshot[key]
        if current.is_floating_point():
            squared += float((current.detach() - previous).square().sum().item())
    value = math.sqrt(squared)
    if not math.isfinite(value):
        raise RecurrentPPOError("parameter delta became nonfinite")
    return value


def _assert_finite_model(model: PublicRecurrentActorCritic) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise RecurrentPPOError(f"model state {name!r} is nonfinite")


def _assert_finite_gradients(model: PublicRecurrentActorCritic) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all().item()
        ):
            raise RecurrentPPOError(f"gradient for {name!r} is nonfinite")


def _assert_finite_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if (
                isinstance(value, Tensor)
                and value.is_floating_point()
                and not bool(torch.isfinite(value).all().item())
            ):
                raise RecurrentPPOError(f"optimizer state {key!r} is nonfinite")


def _assert_finite_diagnostics(diagnostics: PPOUpdateDiagnostics) -> None:
    for field_name in (
        "advantage_mean",
        "advantage_std",
        "initial_policy_objective",
        "final_policy_objective",
        "policy_objective_delta",
        "final_total_loss",
        "final_policy_loss",
        "final_value_loss",
        "final_entropy",
        "approximate_kl",
        "policy_clip_fraction",
        "value_clip_fraction",
        "explained_variance",
        "gradient_norm_mean",
        "gradient_norm_max",
        "gradient_clip_fraction",
        "parameter_delta_l2",
        "burn_in_state_delta_mean",
        "min_effective_world_total_weight",
        "max_effective_world_total_weight",
        "mean_transition_loss_weight",
    ):
        value = getattr(diagnostics, field_name)
        if not math.isfinite(value):
            raise RecurrentPPOError(f"diagnostic {field_name!r} is nonfinite")


def _step_key(step: RecurrentRolloutStep) -> tuple[str, int, int]:
    return (step.world_id, step.agent_id, step.decision_index)


def _finite_scalar(value: Tensor, *, field: str) -> float:
    parsed = float(value.detach().item())
    if not math.isfinite(parsed):
        raise RecurrentPPOError(f"{field} must be finite")
    return parsed


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentPPOError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentPPOError(f"{field} must be a nonnegative integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    parsed = _nonnegative_float(value, field=field)
    if parsed <= 0.0:
        raise RecurrentPPOError(f"{field} must be greater than zero")
    return parsed


def _nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentPPOError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RecurrentPPOError(f"{field} must be finite and nonnegative")
    return parsed


def _unit_interval(
    value: object,
    *,
    field: str,
    allow_zero: bool,
) -> float:
    parsed = _nonnegative_float(value, field=field)
    lower_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not lower_ok or parsed > 1.0:
        boundary = "[0, 1]" if allow_zero else "(0, 1]"
        raise RecurrentPPOError(f"{field} must be in {boundary}")
    return parsed
