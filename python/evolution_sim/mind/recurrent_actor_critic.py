from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType

import torch
from torch import Tensor, nn

from evolution_sim.env.runtime.action_contract import (
    ACTION_MASK_CONTRACT_VERSION,
    ACTION_NAMES,
)
from evolution_sim.env.runtime.trajectory import REWARD_TOTAL_BOUNDS
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    ecological_policy_input_values,
    ecological_policy_values_from_decoded,
)


PUBLIC_INPUT_SIZE = ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
ACTION_COUNT = len(ACTION_NAMES)
PREVIOUS_PUBLIC_FEEDBACK_SIZE = ACTION_COUNT * 2 + 3
LEARNED_ENCODER_INPUT_SIZE = (
    PUBLIC_INPUT_SIZE + ACTION_COUNT + PREVIOUS_PUBLIC_FEEDBACK_SIZE
)
PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION = "mind_previous_public_feedback_v1"
_REWARD_NORMALIZATION_SCALE = max(abs(bound) for bound in REWARD_TOTAL_BOUNDS)
ACTION_INDEX: Mapping[str, int] = MappingProxyType(
    {action: index for index, action in enumerate(ACTION_NAMES)}
)
RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION = "mind_public_recurrent_masked_actor_critic_v1"


class RecurrentPolicyContractError(ValueError):
    """Base error for a fail-closed recurrent-policy contract violation."""


class PublicInputError(RecurrentPolicyContractError):
    """Raised when a tensor is not a valid public ecological policy input."""


class ActionMaskError(RecurrentPolicyContractError):
    """Raised when an action mask is incomplete, ambiguous, or empty."""


class RecurrentStateError(RecurrentPolicyContractError):
    """Raised when recurrent state does not match the model invocation."""


class RecurrentContextError(RecurrentPolicyContractError):
    """Raised when aligned previous public feedback violates its contract."""


@dataclass(frozen=True, slots=True)
class RecurrentActorCriticConfig:
    encoder_size: int = 128
    hidden_size: int = 128
    recurrent_layers: int = 1

    def __post_init__(self) -> None:
        for field_name in ("encoder_size", "hidden_size", "recurrent_layers"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PreviousPublicFeedbackInput:
    """Previous finalized same-agent public outcome supplied to the learner."""

    requested_action_id: int | None
    resolved_action_id: int | None
    resolution_action_valid: bool
    moved: bool
    reward_total: float

    def __post_init__(self) -> None:
        for field_name in ("requested_action_id", "resolved_action_id"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= ACTION_COUNT
            ):
                raise RecurrentContextError(f"{field_name} is outside the action space")
        if (self.requested_action_id is None) != (self.resolved_action_id is None):
            raise RecurrentContextError(
                "requested and resolved action ids must both be present or both absent"
            )
        if type(self.resolution_action_valid) is not bool:
            raise RecurrentContextError(
                "resolution_action_valid must be an exact boolean"
            )
        if type(self.moved) is not bool:
            raise RecurrentContextError("moved must be an exact boolean")
        if isinstance(self.reward_total, bool) or not isinstance(
            self.reward_total, (int, float)
        ):
            raise RecurrentContextError("reward_total must be numeric")
        reward_total = float(self.reward_total)
        if not math.isfinite(reward_total):
            raise RecurrentContextError("reward_total must be finite")
        if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
            raise RecurrentContextError(
                "reward_total is outside the public trajectory reward bounds"
            )
        if self.requested_action_id is None and (
            self.resolution_action_valid or self.moved or reward_total != 0.0
        ):
            raise RecurrentContextError("birth/reset feedback must be entirely zero")
        if self.requested_action_id is not None:
            if (
                self.resolution_action_valid
                and self.resolved_action_id != self.requested_action_id
            ):
                raise RecurrentContextError(
                    "valid feedback must resolve the requested action"
                )
            if (
                not self.resolution_action_valid
                and self.resolved_action_id != ACTION_INDEX["stay"]
            ):
                raise RecurrentContextError(
                    "invalid feedback must fail closed to stay"
                )
            resolved_action = ACTION_NAMES[self.resolved_action_id]
            expected_moved = (
                self.resolution_action_valid
                and resolved_action.startswith("move_")
            )
            if self.moved != expected_moved:
                raise RecurrentContextError(
                    "moved feedback must exactly match a valid resolved movement"
                )

    @classmethod
    def zero(cls) -> PreviousPublicFeedbackInput:
        return cls(
            requested_action_id=None,
            resolved_action_id=None,
            resolution_action_valid=False,
            moved=False,
            reward_total=0.0,
        )

    def values(self) -> tuple[float, ...]:
        requested = [0.0] * ACTION_COUNT
        resolved = [0.0] * ACTION_COUNT
        if self.requested_action_id is not None:
            requested[self.requested_action_id] = 1.0
        if self.resolved_action_id is not None:
            resolved[self.resolved_action_id] = 1.0
        values = (
            *requested,
            *resolved,
            float(self.resolution_action_valid),
            float(self.moved),
            float(self.reward_total) / _REWARD_NORMALIZATION_SCALE,
        )
        if len(values) != PREVIOUS_PUBLIC_FEEDBACK_SIZE:
            raise AssertionError("previous public feedback size drifted")
        return values


@dataclass(frozen=True, slots=True)
class ActorCriticSequenceOutput:
    raw_logits: Tensor
    masked_logits: Tensor
    values: Tensor
    final_state: Tensor


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    raw_logits: Tensor
    masked_logits: Tensor
    values: Tensor
    log_probs: Tensor
    entropy: Tensor
    final_state: Tensor


@dataclass(frozen=True, slots=True)
class ActionSelection:
    actions: Tensor
    raw_logits: Tensor
    masked_logits: Tensor
    values: Tensor
    log_probs: Tensor
    entropy: Tensor
    next_state: Tensor


def recurrent_actor_critic_contract(
    config: RecurrentActorCriticConfig | None = None,
) -> dict[str, object]:
    resolved = config or RecurrentActorCriticConfig()
    return {
        "schema_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        "public_input_schema_version": ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        "public_input_size": PUBLIC_INPUT_SIZE,
        "previous_public_feedback_schema_version": (
            PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
        ),
        "previous_public_feedback_size": PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        "action_mask_contract_version": ACTION_MASK_CONTRACT_VERSION,
        "action_names": list(ACTION_NAMES),
        "action_count": ACTION_COUNT,
        "action_id_encoding": "zero_based_stable_action_contract_order",
        "architecture": {
            "parameter_sharing": "one_policy_for_all_agents",
            "learned_encoder_input_size": LEARNED_ENCODER_INPUT_SIZE,
            "learned_encoder_inputs": [
                "current_ecological_observation_541",
                "current_action_mask_20",
                "previous_requested_action_one_hot_20",
                "previous_resolved_action_one_hot_20",
                "previous_resolution_valid_1",
                "previous_moved_1",
                "previous_public_reward_total_normalized_1",
            ],
            "input_normalization": "explicit_layer_norm",
            "encoder": "linear_tanh",
            "memory": "gru",
            "actor": "linear_20_logits",
            "critic": "linear_scalar_value",
            **asdict(resolved),
        },
        "selection": {
            "policy": "strict_current_action_mask_constrained_distribution",
            "empty_mask": "error_fail_closed",
            "invalid_mask": "error_fail_closed",
            "invalid_training_action": "error_fail_closed",
            "stochastic_sampling": "caller_supplied_seeded_torch_generator",
            "deterministic_evaluation": "masked_argmax_stable_action_id_tie_break",
        },
        "recurrent_state": {
            "scope": "one_hidden_state_per_agent",
            "birth": "zero_state",
            "episode_or_world_reset": "zero_state",
            "death": "discard_state",
            "birth_or_world_reset_previous_feedback": "all_zero",
        },
        "runtime_integrated": False,
        "heuristic_action_source": False,
    }


def public_policy_tensor_from_observation_input(
    observation_input: dict[str, object],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Project a versioned raw observation payload onto the safe 541 features."""

    values = ecological_policy_input_values(observation_input)
    return _public_values_tensor(values, device=device, dtype=dtype)


def public_policy_tensor_from_decoded(
    decoded_observation: Sequence[float],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Project a decoded raw observation vector onto the safe 541 features."""

    values = ecological_policy_values_from_decoded(decoded_observation)
    return _public_values_tensor(values, device=device, dtype=dtype)


def strict_action_mask_tensor(
    action_mask: Mapping[str, object],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode one complete Foundation action mask without truthy coercion."""

    if not isinstance(action_mask, Mapping):
        raise ActionMaskError("action_mask must be a mapping")
    observed_keys = set(action_mask)
    expected_keys = set(ACTION_NAMES)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys, key=str)
        raise ActionMaskError(
            "action_mask keys must exactly match the stable action contract; "
            f"missing={missing}, extra={extra}"
        )
    invalid_types = [
        action for action in ACTION_NAMES if type(action_mask[action]) is not bool
    ]
    if invalid_types:
        raise ActionMaskError(
            f"action_mask values must be exact booleans; invalid={invalid_types}"
        )
    encoded = torch.tensor(
        [action_mask[action] for action in ACTION_NAMES],
        dtype=torch.bool,
        device=device,
    )
    validate_action_mask_tensor(encoded, leading_shape=())
    return encoded


def previous_public_feedback_tensor(
    feedback: PreviousPublicFeedbackInput,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if not isinstance(feedback, PreviousPublicFeedbackInput):
        raise RecurrentContextError("feedback must be a PreviousPublicFeedbackInput")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise RecurrentContextError("previous feedback dtype must be floating point")
    tensor = torch.tensor(feedback.values(), device=device, dtype=dtype)
    validate_previous_feedback_tensor(tensor, leading_shape=())
    return tensor


def validate_public_input_tensor(
    observations: Tensor,
    *,
    ranks: tuple[int, ...] = (1, 2, 3),
) -> None:
    if not isinstance(observations, Tensor):
        raise PublicInputError("public observations must be a torch.Tensor")
    if observations.ndim not in ranks:
        raise PublicInputError(
            f"public observations must have rank in {ranks}; got {observations.ndim}"
        )
    if observations.shape[-1] != PUBLIC_INPUT_SIZE:
        raise PublicInputError(
            "public observations have unexpected final dimension: "
            f"{observations.shape[-1]}; expected {PUBLIC_INPUT_SIZE}"
        )
    if any(size <= 0 for size in observations.shape):
        raise PublicInputError("public observations cannot contain an empty dimension")
    if not observations.is_floating_point():
        raise PublicInputError("public observations must use a floating-point dtype")
    if not bool(torch.isfinite(observations).all().item()):
        raise PublicInputError("public observations must be finite")
    if not bool(((observations >= -1.0) & (observations <= 1.0)).all().item()):
        raise PublicInputError("public observations must be in [-1, 1]")


def validate_action_mask_tensor(
    action_masks: Tensor,
    *,
    leading_shape: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(action_masks, Tensor):
        raise ActionMaskError("action masks must be a torch.Tensor")
    if action_masks.dtype != torch.bool:
        raise ActionMaskError("action masks must use torch.bool without coercion")
    if action_masks.ndim < 1 or action_masks.shape[-1] != ACTION_COUNT:
        raise ActionMaskError(
            "action masks must end with the stable action dimension "
            f"{ACTION_COUNT}; got {tuple(action_masks.shape)}"
        )
    if leading_shape is not None and tuple(action_masks.shape[:-1]) != leading_shape:
        raise ActionMaskError(
            "action-mask leading shape does not match observations: "
            f"{tuple(action_masks.shape[:-1])} != {leading_shape}"
        )
    if action_masks.numel() == 0:
        raise ActionMaskError("action masks cannot be empty")
    legal_counts = action_masks.sum(dim=-1)
    if not bool((legal_counts > 0).all().item()):
        empty_indices = torch.nonzero(legal_counts == 0, as_tuple=False).tolist()
        raise ActionMaskError(
            "every action mask must expose at least one valid action; "
            f"empty_rows={empty_indices[:16]}"
        )


def validate_previous_feedback_tensor(
    previous_feedback: Tensor,
    *,
    leading_shape: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(previous_feedback, Tensor):
        raise RecurrentContextError("previous feedback must be a torch.Tensor")
    if not previous_feedback.is_floating_point():
        raise RecurrentContextError("previous feedback must use a floating dtype")
    if (
        previous_feedback.ndim < 1
        or previous_feedback.shape[-1] != PREVIOUS_PUBLIC_FEEDBACK_SIZE
    ):
        raise RecurrentContextError(
            "previous feedback must end with dimension "
            f"{PREVIOUS_PUBLIC_FEEDBACK_SIZE}; got {tuple(previous_feedback.shape)}"
        )
    if (
        leading_shape is not None
        and tuple(previous_feedback.shape[:-1]) != leading_shape
    ):
        raise RecurrentContextError(
            "previous-feedback leading shape does not match observations: "
            f"{tuple(previous_feedback.shape[:-1])} != {leading_shape}"
        )
    if previous_feedback.numel() == 0:
        raise RecurrentContextError("previous feedback cannot be empty")
    if not bool(torch.isfinite(previous_feedback).all().item()):
        raise RecurrentContextError("previous feedback must be finite")

    requested = previous_feedback[..., :ACTION_COUNT]
    resolved = previous_feedback[..., ACTION_COUNT : ACTION_COUNT * 2]
    resolution_valid = previous_feedback[..., -3]
    moved = previous_feedback[..., -2]
    normalized_reward = previous_feedback[..., -1]
    for field_name, values in (
        ("requested action one-hot", requested),
        ("resolved action one-hot", resolved),
        ("resolution-valid scalar", resolution_valid),
        ("moved scalar", moved),
    ):
        if not bool(((values == 0.0) | (values == 1.0)).all().item()):
            raise RecurrentContextError(f"{field_name} must contain only 0 or 1")
    requested_count = requested.sum(dim=-1)
    resolved_count = resolved.sum(dim=-1)
    if not bool(((requested_count == 0.0) | (requested_count == 1.0)).all().item()):
        raise RecurrentContextError("requested action encoding must be zero or one-hot")
    if not bool(((resolved_count == 0.0) | (resolved_count == 1.0)).all().item()):
        raise RecurrentContextError("resolved action encoding must be zero or one-hot")
    if not bool((requested_count == resolved_count).all().item()):
        raise RecurrentContextError(
            "requested and resolved action availability must match"
        )
    present = requested_count == 1.0
    resolution_is_valid = resolution_valid == 1.0
    same_action = (requested == resolved).all(dim=-1)
    if not bool((~present | ~resolution_is_valid | same_action).all().item()):
        raise RecurrentContextError(
            "valid feedback must resolve the requested action"
        )
    resolved_stay = resolved[..., ACTION_INDEX["stay"]] == 1.0
    if not bool(
        (~present | resolution_is_valid | resolved_stay).all().item()
    ):
        raise RecurrentContextError("invalid feedback must fail closed to stay")
    resolved_move = torch.zeros_like(resolution_is_valid)
    for action_index, action in enumerate(ACTION_NAMES):
        if action.startswith("move_"):
            resolved_move |= resolved[..., action_index] == 1.0
    expected_moved = present & resolution_is_valid & resolved_move
    if not bool(((moved == 1.0) == expected_moved).all().item()):
        raise RecurrentContextError(
            "moved feedback must exactly match a valid resolved movement"
        )
    normalized_lower = REWARD_TOTAL_BOUNDS[0] / _REWARD_NORMALIZATION_SCALE
    normalized_upper = REWARD_TOTAL_BOUNDS[1] / _REWARD_NORMALIZATION_SCALE
    if not bool(
        (
            (normalized_reward >= normalized_lower)
            & (normalized_reward <= normalized_upper)
        )
        .all()
        .item()
    ):
        raise RecurrentContextError(
            "normalized reward is outside the public trajectory reward bounds"
        )
    absent = requested_count == 0.0
    absent_scalars_are_zero = (
        (resolution_valid == 0.0) & (moved == 0.0) & (normalized_reward == 0.0)
    )
    if not bool((~absent | absent_scalars_are_zero).all().item()):
        raise RecurrentContextError("birth/reset feedback must be entirely zero")


def seeded_torch_generator(
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    parsed_seed = _validated_seed(seed)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(parsed_seed)
    return generator


class BackendStableLayerNorm(nn.Module):
    """Layer normalization without a backend-fused affine backward kernel.

    PyTorch's MPS fused ``LayerNorm`` backward can emit non-finite affine
    gradients when one module is reused across multiple real agent sequences
    in a PPO minibatch.  The explicit formulation is mathematically identical
    to LayerNorm's biased-variance definition, preserves the conventional
    ``weight``/``bias`` state-dict keys, and is deterministic across CPU, CUDA,
    and MPS backends.
    """

    def __init__(self, normalized_size: int, *, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        if (
            isinstance(normalized_size, bool)
            or not isinstance(normalized_size, int)
            or normalized_size <= 0
        ):
            raise ValueError("normalized_size must be a positive integer")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            raise ValueError("epsilon must be a positive finite number")
        self.normalized_size = normalized_size
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.normalized_size:
            raise PublicInputError(
                "layer-normalization input must end with dimension "
                f"{self.normalized_size}; got {tuple(inputs.shape)}"
            )
        centered = inputs - inputs.mean(dim=-1, keepdim=True)
        inverse_standard_deviation = torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return centered * inverse_standard_deviation * self.weight + self.bias


class PublicRecurrentActorCritic(nn.Module):
    """Parameter-shared masked recurrent actor-critic for public Mind inputs."""

    def __init__(
        self,
        config: RecurrentActorCriticConfig | None = None,
        *,
        initialization_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RecurrentActorCriticConfig()
        seed = (
            None
            if initialization_seed is None
            else _validated_seed(initialization_seed)
        )
        rng_context = (
            torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        )
        with rng_context:
            if seed is not None:
                torch.manual_seed(seed)
            self.input_norm = BackendStableLayerNorm(LEARNED_ENCODER_INPUT_SIZE)
            self.encoder = nn.Sequential(
                nn.Linear(LEARNED_ENCODER_INPUT_SIZE, self.config.encoder_size),
                nn.Tanh(),
            )
            self.recurrent = nn.GRU(
                self.config.encoder_size,
                self.config.hidden_size,
                num_layers=self.config.recurrent_layers,
            )
            self.actor = nn.Linear(self.config.hidden_size, ACTION_COUNT)
            self.value = nn.Linear(self.config.hidden_size, 1)
            self._initialize_parameters()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise RecurrentStateError("batch_size must be a positive integer")
        reference = next(self.parameters())
        resolved_device = reference.device if device is None else torch.device(device)
        resolved_dtype = reference.dtype if dtype is None else dtype
        if resolved_device != reference.device:
            raise RecurrentStateError(
                f"recurrent state device {resolved_device} does not match model device "
                f"{reference.device}"
            )
        if resolved_dtype != reference.dtype:
            raise RecurrentStateError(
                f"recurrent state dtype {resolved_dtype} does not match model dtype "
                f"{reference.dtype}"
            )
        if not torch.empty((), dtype=resolved_dtype).is_floating_point():
            raise RecurrentStateError("recurrent state dtype must be floating point")
        return torch.zeros(
            self.config.recurrent_layers,
            batch_size,
            self.config.hidden_size,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    def forward_sequence(
        self,
        observations: Tensor,
        action_masks: Tensor,
        previous_feedback: Tensor,
        *,
        initial_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> ActorCriticSequenceOutput:
        """Evaluate a time-major ``[time, batch, feature]`` sequence.

        ``episode_starts[t, b]`` resets that batch member before observation
        ``t``. This makes concatenated PPO chunks equivalent to separately
        evaluated episodes while retaining gradients within each segment.
        """

        validate_public_input_tensor(observations, ranks=(3,))
        self._validate_model_input_placement(observations)
        time_steps, batch_size, _ = observations.shape
        validate_action_mask_tensor(
            action_masks,
            leading_shape=(time_steps, batch_size),
        )
        if action_masks.device != observations.device:
            raise ActionMaskError("action masks and observations must share a device")
        validate_previous_feedback_tensor(
            previous_feedback,
            leading_shape=(time_steps, batch_size),
        )
        if previous_feedback.device != observations.device:
            raise RecurrentContextError(
                "previous feedback and observations must share a device"
            )
        if previous_feedback.dtype != observations.dtype:
            raise RecurrentContextError(
                "previous feedback and observations must share a dtype"
            )
        starts = self._validated_episode_starts(
            episode_starts,
            time_steps=time_steps,
            batch_size=batch_size,
            device=observations.device,
        )
        if bool(starts.any().item()) and not bool(
            (previous_feedback[starts] == 0.0).all().item()
        ):
            raise RecurrentContextError(
                "episode-start previous feedback must be entirely zero"
            )
        state = self._validated_or_initial_state(
            initial_state,
            batch_size=batch_size,
            observations=observations,
        )

        learned_inputs = torch.cat(
            (
                observations,
                action_masks.to(dtype=observations.dtype),
                previous_feedback,
            ),
            dim=-1,
        )
        if learned_inputs.shape[-1] != LEARNED_ENCODER_INPUT_SIZE:
            raise AssertionError("learned encoder input size drifted")
        encoded = self.encoder(self.input_norm(learned_inputs))
        outputs: list[Tensor] = []
        for step in range(time_steps):
            keep_state = (~starts[step]).to(observations.dtype).view(1, batch_size, 1)
            state = state * keep_state
            recurrent_output, state = self.recurrent(encoded[step : step + 1], state)
            outputs.append(recurrent_output)
        recurrent_outputs = torch.cat(outputs, dim=0)
        raw_logits = self.actor(recurrent_outputs)
        masked_logits = raw_logits.masked_fill(~action_masks, -torch.inf)
        values = self.value(recurrent_outputs).squeeze(-1)
        return ActorCriticSequenceOutput(
            raw_logits=raw_logits,
            masked_logits=masked_logits,
            values=values,
            final_state=state,
        )

    def evaluate_sequence(
        self,
        observations: Tensor,
        action_masks: Tensor,
        previous_feedback: Tensor,
        actions: Tensor,
        *,
        initial_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> SequenceEvaluation:
        """Return differentiable recurrent-PPO log-probs, entropy, and values."""

        output = self.forward_sequence(
            observations,
            action_masks,
            previous_feedback,
            initial_state=initial_state,
            episode_starts=episode_starts,
        )
        expected_shape = tuple(observations.shape[:2])
        self._validate_actions(
            actions,
            action_masks=action_masks,
            expected_shape=expected_shape,
        )
        distribution = torch.distributions.Categorical(logits=output.masked_logits)
        return SequenceEvaluation(
            raw_logits=output.raw_logits,
            masked_logits=output.masked_logits,
            values=output.values,
            log_probs=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            final_state=output.final_state,
        )

    def act(
        self,
        observations: Tensor,
        action_masks: Tensor,
        previous_feedback: Tensor,
        *,
        recurrent_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> ActionSelection:
        """Select a legal action for one decision step per batch member."""

        validate_public_input_tensor(observations, ranks=(1, 2))
        if not isinstance(action_masks, Tensor):
            raise ActionMaskError("action masks must be a torch.Tensor")
        if type(deterministic) is not bool:
            raise RecurrentPolicyContractError("deterministic must be an exact boolean")
        batched_observations = (
            observations.unsqueeze(0) if observations.ndim == 1 else observations
        )
        batched_masks = (
            action_masks.unsqueeze(0) if action_masks.ndim == 1 else action_masks
        )
        if not isinstance(previous_feedback, Tensor):
            raise RecurrentContextError("previous feedback must be a torch.Tensor")
        batched_feedback = (
            previous_feedback.unsqueeze(0)
            if previous_feedback.ndim == 1
            else previous_feedback
        )
        batch_size = batched_observations.shape[0]
        validate_action_mask_tensor(batched_masks, leading_shape=(batch_size,))
        if batched_masks.device != batched_observations.device:
            raise ActionMaskError("action masks and observations must share a device")
        if episode_starts is None:
            starts = None
        else:
            if not isinstance(episode_starts, Tensor):
                raise RecurrentStateError("episode_starts must be a torch.Tensor")
            starts = (
                episode_starts.reshape(1)
                if episode_starts.ndim == 0
                else episode_starts
            )
            if starts.shape != (batch_size,):
                raise RecurrentStateError(
                    f"episode_starts must have shape {(batch_size,)}; got {tuple(starts.shape)}"
                )
        output = self.forward_sequence(
            batched_observations.unsqueeze(0),
            batched_masks.unsqueeze(0),
            batched_feedback.unsqueeze(0),
            initial_state=recurrent_state,
            episode_starts=None if starts is None else starts.unsqueeze(0),
        )
        logits = output.masked_logits.squeeze(0)
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            if generator is None:
                raise RecurrentPolicyContractError(
                    "stochastic action selection requires an explicit seeded generator"
                )
            if not isinstance(generator, torch.Generator):
                raise RecurrentPolicyContractError(
                    "generator must be a torch.Generator created for the model device"
                )
            if torch.device(generator.device) != logits.device:
                raise RecurrentPolicyContractError(
                    "sampling generator and model outputs must share a device"
                )
            actions = torch.multinomial(
                distribution.probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
        return ActionSelection(
            actions=actions,
            raw_logits=output.raw_logits.squeeze(0),
            masked_logits=logits,
            values=output.values.squeeze(0),
            log_probs=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            next_state=output.final_state,
        )

    def validate_recurrent_state(
        self,
        state: Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(state, Tensor):
            raise RecurrentStateError("recurrent state must be a torch.Tensor")
        expected_shape = (
            self.config.recurrent_layers,
            batch_size,
            self.config.hidden_size,
        )
        if tuple(state.shape) != expected_shape:
            raise RecurrentStateError(
                f"recurrent state shape must be {expected_shape}; got {tuple(state.shape)}"
            )
        if state.device != device:
            raise RecurrentStateError(
                f"recurrent state device {state.device} does not match {device}"
            )
        if state.dtype != dtype:
            raise RecurrentStateError(
                f"recurrent state dtype {state.dtype} does not match {dtype}"
            )
        if not bool(torch.isfinite(state).all().item()):
            raise RecurrentStateError("recurrent state must be finite")

    def _validated_or_initial_state(
        self,
        state: Tensor | None,
        *,
        batch_size: int,
        observations: Tensor,
    ) -> Tensor:
        if state is None:
            return self.initial_state(
                batch_size,
                device=observations.device,
                dtype=observations.dtype,
            )
        self.validate_recurrent_state(
            state,
            batch_size=batch_size,
            device=observations.device,
            dtype=observations.dtype,
        )
        return state

    def _validate_model_input_placement(self, observations: Tensor) -> None:
        reference = next(self.parameters())
        if observations.device != reference.device:
            raise PublicInputError(
                f"public observations device {observations.device} does not match "
                f"model device {reference.device}"
            )
        if observations.dtype != reference.dtype:
            raise PublicInputError(
                f"public observations dtype {observations.dtype} does not match "
                f"model dtype {reference.dtype}"
            )

    @staticmethod
    def _validated_episode_starts(
        episode_starts: Tensor | None,
        *,
        time_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if episode_starts is None:
            return torch.zeros(
                time_steps,
                batch_size,
                dtype=torch.bool,
                device=device,
            )
        if not isinstance(episode_starts, Tensor):
            raise RecurrentStateError("episode_starts must be a torch.Tensor")
        if episode_starts.dtype != torch.bool:
            raise RecurrentStateError("episode_starts must use torch.bool")
        if episode_starts.shape != (time_steps, batch_size):
            raise RecurrentStateError(
                "episode_starts shape must match sequence leading shape: "
                f"{tuple(episode_starts.shape)} != {(time_steps, batch_size)}"
            )
        if episode_starts.device != device:
            raise RecurrentStateError(
                "episode_starts and observations must share a device"
            )
        return episode_starts

    @staticmethod
    def _validate_actions(
        actions: Tensor,
        *,
        action_masks: Tensor,
        expected_shape: tuple[int, ...],
    ) -> None:
        if not isinstance(actions, Tensor):
            raise ActionMaskError("actions must be a torch.Tensor")
        if actions.dtype != torch.long:
            raise ActionMaskError("actions must use torch.long action ids")
        if tuple(actions.shape) != expected_shape:
            raise ActionMaskError(
                f"actions shape must be {expected_shape}; got {tuple(actions.shape)}"
            )
        if actions.device != action_masks.device:
            raise ActionMaskError("actions and action masks must share a device")
        if not bool(((actions >= 0) & (actions < ACTION_COUNT)).all().item()):
            raise ActionMaskError(
                "actions contain an id outside the stable action space"
            )
        selected_is_legal = action_masks.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if not bool(selected_is_legal.all().item()):
            invalid_indices = torch.nonzero(~selected_is_legal, as_tuple=False).tolist()
            raise ActionMaskError(
                "actions contain ids forbidden by their current action masks; "
                f"invalid_rows={invalid_indices[:16]}"
            )

    def _initialize_parameters(self) -> None:
        encoder_linear = self.encoder[0]
        assert isinstance(encoder_linear, nn.Linear)
        nn.init.orthogonal_(encoder_linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(encoder_linear.bias)
        for name, parameter in self.recurrent.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)


class PerAgentRecurrentStateStore:
    """Host-side state routing; agent identity is never exposed to the model."""

    def __init__(self, model: PublicRecurrentActorCritic) -> None:
        if not isinstance(model, PublicRecurrentActorCritic):
            raise TypeError("model must be a PublicRecurrentActorCritic")
        self._model = model
        self._states: dict[Hashable, Tensor] = {}

    def state_for(
        self,
        agent_id: Hashable,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        _validate_agent_id(agent_id)
        existing = self._states.get(agent_id)
        if existing is not None:
            if device is not None and existing.device != torch.device(device):
                raise RecurrentStateError("stored state is on a different device")
            if dtype is not None and existing.dtype != dtype:
                raise RecurrentStateError("stored state uses a different dtype")
            return existing.clone()
        return self._model.initial_state(1, device=device, dtype=dtype)

    def update(self, agent_id: Hashable, next_state: Tensor) -> None:
        _validate_agent_id(agent_id)
        if not isinstance(next_state, Tensor):
            raise RecurrentStateError("next_state must be a torch.Tensor")
        reference = next(self._model.parameters())
        self._model.validate_recurrent_state(
            next_state,
            batch_size=1,
            device=reference.device,
            dtype=reference.dtype,
        )
        self._states[agent_id] = next_state.detach().clone()

    def reset_agent(self, agent_id: Hashable) -> bool:
        """Discard one state on death; the next lookup returns birth-zero state."""

        _validate_agent_id(agent_id)
        return self._states.pop(agent_id, None) is not None

    def retain_agents(self, active_agent_ids: Sequence[Hashable]) -> int:
        """Discard states for agents absent from the current alive-agent set."""

        active = set(active_agent_ids)
        for agent_id in active:
            _validate_agent_id(agent_id)
        stale = [agent_id for agent_id in self._states if agent_id not in active]
        for agent_id in stale:
            del self._states[agent_id]
        return len(stale)

    def reset_world(self) -> int:
        """Discard every state at episode/world reset and return the removed count."""

        removed = len(self._states)
        self._states.clear()
        return removed

    @property
    def tracked_agent_ids(self) -> tuple[Hashable, ...]:
        return tuple(self._states)

    def __len__(self) -> int:
        return len(self._states)


def _public_values_tensor(
    values: Sequence[float],
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> Tensor:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise PublicInputError("public policy tensor dtype must be floating point")
    tensor = torch.tensor(values, dtype=dtype, device=device)
    validate_public_input_tensor(tensor, ranks=(1,))
    return tensor


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seed < 0 or seed > (2**63 - 1):
        raise ValueError("seed must be in [0, 2**63 - 1]")
    return seed


def _validate_agent_id(agent_id: Hashable) -> None:
    try:
        hash(agent_id)
    except TypeError as exc:
        raise TypeError("agent_id must be hashable") from exc


__all__ = [
    "ACTION_COUNT",
    "ACTION_INDEX",
    "ACTION_NAMES",
    "LEARNED_ENCODER_INPUT_SIZE",
    "PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION",
    "PREVIOUS_PUBLIC_FEEDBACK_SIZE",
    "PUBLIC_INPUT_SIZE",
    "RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION",
    "ActionMaskError",
    "ActionSelection",
    "ActorCriticSequenceOutput",
    "BackendStableLayerNorm",
    "PerAgentRecurrentStateStore",
    "PreviousPublicFeedbackInput",
    "PublicInputError",
    "PublicRecurrentActorCritic",
    "RecurrentActorCriticConfig",
    "RecurrentContextError",
    "RecurrentPolicyContractError",
    "RecurrentStateError",
    "SequenceEvaluation",
    "public_policy_tensor_from_decoded",
    "public_policy_tensor_from_observation_input",
    "previous_public_feedback_tensor",
    "recurrent_actor_critic_contract",
    "seeded_torch_generator",
    "strict_action_mask_tensor",
    "validate_action_mask_tensor",
    "validate_public_input_tensor",
    "validate_previous_feedback_tensor",
]
