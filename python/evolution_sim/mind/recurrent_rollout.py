from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    encode_observation_input,
)
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.trajectory import (
    REWARD_COMPONENT_BOUNDS,
    REWARD_SCHEMA_VERSION,
    REWARD_TOTAL_BOUNDS,
)
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ecological_policy_input_values,
)


RECURRENT_ROLLOUT_POLICY_ID = "mind_v3_recurrent_on_policy"
RECURRENT_ROLLOUT_POLICY_VERSION = "mind_v3_recurrent_on_policy_v1"
RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION = "mind_v3_recurrent_rollout_decision_v2"
RECURRENT_ROLLOUT_ACTION_SOURCE = "learned_recurrent_on_policy"
RECURRENT_ROLLOUT_ACTIONS: tuple[str, ...] = tuple(ACTION_NAMES)
RECURRENT_REWARD_COMPONENTS: tuple[str, ...] = tuple(REWARD_COMPONENT_BOUNDS)
RECURRENT_PUBLIC_FEEDBACK_SCHEMA_VERSION = "mind_previous_public_feedback_v1"
RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE = len(RECURRENT_ROLLOUT_ACTIONS) * 2 + 3
RECURRENT_LEARNED_INPUT_VECTOR_SIZE = (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
    + len(RECURRENT_ROLLOUT_ACTIONS)
    + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
)
RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ippo|policy-action-sampling-v1|2026-07-21"
)
MAX_RECURRENT_POLICY_SAMPLING_SEED = 2**63 - 1
_MIN_DERIVED_POLICY_SAMPLING_SEED = 2**31
_FEEDBACK_REWARD_SCALE = max(abs(bound) for bound in REWARD_TOTAL_BOUNDS)


class RecurrentRolloutError(ValueError):
    pass


def derive_recurrent_policy_sampling_seed(*, task_identity: str) -> int:
    """Derive a deterministic signed-63-bit action-sampling seed.

    The namespace and caller-supplied task identity are the complete derivation
    material. Environment seeds are deliberately absent. Derived seeds live
    above the simulator's signed-31-bit seed range, making the two seed axes
    disjoint by construction rather than by chance.
    """

    if (
        not isinstance(task_identity, str)
        or not task_identity
        or task_identity != task_identity.strip()
    ):
        raise RecurrentRolloutError(
            "policy sampling task_identity must be non-empty and trimmed"
        )
    material = (f"{RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE}|{task_identity}").encode(
        "utf-8"
    )
    digest_value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    derived_span = (
        MAX_RECURRENT_POLICY_SAMPLING_SEED - _MIN_DERIVED_POLICY_SAMPLING_SEED + 1
    )
    return _MIN_DERIVED_POLICY_SAMPLING_SEED + (digest_value % derived_span)


@dataclass(frozen=True, slots=True)
class RecurrentCoreOutput:
    """Detached inference output for one public observation.

    The core owns representation learning. The collector owns the stable action
    mask and seeded sampling, so a training rollout cannot silently fall back to
    a heuristic or an invalid action.
    """

    logits: tuple[float, ...]
    value: float
    next_hidden: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreviousPublicFeedback:
    """Previous finalized same-agent feedback visible to the recurrent core."""

    requested_action_index: int | None
    resolved_action_index: int | None
    resolution_action_valid: bool
    moved: bool
    reward_total: float

    def __post_init__(self) -> None:
        for field_name in ("requested_action_index", "resolved_action_index"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= len(RECURRENT_ROLLOUT_ACTIONS)
            ):
                raise RecurrentRolloutError(f"{field_name} is out of range")
        if (self.requested_action_index is None) != (
            self.resolved_action_index is None
        ):
            raise RecurrentRolloutError(
                "requested and resolved feedback actions must both be present "
                "or both be absent"
            )
        if type(self.resolution_action_valid) is not bool:
            raise RecurrentRolloutError("resolution_action_valid must be a boolean")
        if type(self.moved) is not bool:
            raise RecurrentRolloutError("moved must be a boolean")
        reward_total = _finite_float(self.reward_total, field="feedback reward_total")
        if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
            raise RecurrentRolloutError(
                "feedback reward_total is outside the public reward contract"
            )
        if self.requested_action_index is None and (
            self.resolution_action_valid or self.moved or reward_total != 0.0
        ):
            raise RecurrentRolloutError("birth/reset feedback must be entirely zero")
        if self.moved and not self.resolution_action_valid:
            raise RecurrentRolloutError(
                "moved feedback requires a resolution-valid action"
            )

    @classmethod
    def zero(cls) -> PreviousPublicFeedback:
        return cls(
            requested_action_index=None,
            resolved_action_index=None,
            resolution_action_valid=False,
            moved=False,
            reward_total=0.0,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PreviousPublicFeedback:
        requested_action = _action_name(
            record.get("requested_action"),
            field="requested_action",
        )
        resolved_action = _action_name(
            record.get("resolved_action"),
            field="resolved_action",
        )
        return cls(
            requested_action_index=RECURRENT_ROLLOUT_ACTIONS.index(requested_action),
            resolved_action_index=RECURRENT_ROLLOUT_ACTIONS.index(resolved_action),
            resolution_action_valid=_strict_bool(
                record.get("resolution_action_valid"),
                field="resolution_action_valid",
            ),
            moved=_strict_bool(record.get("moved"), field="moved"),
            reward_total=_bounded_reward_total(record),
        )

    @property
    def available(self) -> bool:
        return self.requested_action_index is not None

    def vector(self) -> tuple[float, ...]:
        requested = [0.0] * len(RECURRENT_ROLLOUT_ACTIONS)
        resolved = [0.0] * len(RECURRENT_ROLLOUT_ACTIONS)
        if self.requested_action_index is not None:
            requested[self.requested_action_index] = 1.0
        if self.resolved_action_index is not None:
            resolved[self.resolved_action_index] = 1.0
        values = (
            *requested,
            *resolved,
            float(self.resolution_action_valid),
            float(self.moved),
            self.reward_total / _FEEDBACK_REWARD_SCALE,
        )
        if len(values) != RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE:
            raise AssertionError("previous public feedback vector size drifted")
        return values


class RecurrentPolicyCore(Protocol):
    hidden_size: int
    public_input_schema_version: str
    public_input_size: int
    learned_input_size: int

    def initial_hidden(self) -> Sequence[float]: ...

    def forward_step(
        self,
        observation: Sequence[float],
        current_action_mask: Sequence[bool],
        previous_feedback: PreviousPublicFeedback,
        hidden: Sequence[float],
    ) -> RecurrentCoreOutput: ...


class TorchRecurrentPolicyCore:
    """No-grad adapter from the shared torch model to the rollout protocol."""

    def __init__(self, model: Any) -> None:
        try:
            import torch
            from evolution_sim.mind.recurrent_actor_critic import (
                PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION,
                PREVIOUS_PUBLIC_FEEDBACK_SIZE,
                PublicRecurrentActorCritic,
            )
        except ModuleNotFoundError as exc:
            raise RecurrentRolloutError(
                "TorchRecurrentPolicyCore requires the optional Mind ML stack"
            ) from exc
        if not isinstance(model, PublicRecurrentActorCritic):
            raise RecurrentRolloutError("model must be a PublicRecurrentActorCritic")
        if PREVIOUS_PUBLIC_FEEDBACK_SIZE != RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE:
            raise RecurrentRolloutError(
                "torch and rollout previous-feedback contracts disagree"
            )
        if (
            PREVIOUS_PUBLIC_FEEDBACK_SCHEMA_VERSION
            != RECURRENT_PUBLIC_FEEDBACK_SCHEMA_VERSION
        ):
            raise RecurrentRolloutError(
                "torch and rollout previous-feedback schema versions disagree"
            )
        expected_learned_input_size = (
            model.config.public_input_size
            + len(RECURRENT_ROLLOUT_ACTIONS)
            + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
        )
        if model.config.learned_encoder_input_size != expected_learned_input_size:
            raise RecurrentRolloutError(
                "torch and rollout learned input sizes disagree"
            )
        self._torch = torch
        self._model = model
        self._model.eval()
        self.public_input_schema_version = model.config.public_input_schema_version
        self.public_input_size = model.config.public_input_size
        self.learned_input_size = model.config.learned_encoder_input_size
        self._layers = int(model.config.recurrent_layers)
        self._layer_hidden_size = int(model.config.hidden_size)
        self.hidden_size = self._layers * self._layer_hidden_size

    def initial_hidden(self) -> tuple[float, ...]:
        state = self._model.initial_state(1)
        return tuple(
            float(value) for value in state.detach().cpu().reshape(-1).tolist()
        )

    def forward_step(
        self,
        observation: Sequence[float],
        current_action_mask: Sequence[bool],
        previous_feedback: PreviousPublicFeedback,
        hidden: Sequence[float],
    ) -> RecurrentCoreOutput:
        torch = self._torch
        reference = next(self._model.parameters())
        observations = torch.tensor(
            tuple(observation),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, self.public_input_size)
        action_masks = torch.tensor(
            tuple(current_action_mask),
            device=reference.device,
            dtype=torch.bool,
        ).reshape(1, 1, len(RECURRENT_ROLLOUT_ACTIONS))
        feedback = torch.tensor(
            previous_feedback.vector(),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(1, 1, RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE)
        state = torch.tensor(
            tuple(hidden),
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(self._layers, 1, self._layer_hidden_size)
        with torch.no_grad():
            output = self._model.forward_sequence(
                observations,
                action_masks,
                feedback,
                initial_state=state,
            )
        return RecurrentCoreOutput(
            logits=tuple(
                float(value)
                for value in output.raw_logits[0, 0].detach().cpu().tolist()
            ),
            value=float(output.values[0, 0].detach().cpu().item()),
            next_hidden=tuple(
                float(value)
                for value in output.final_state.detach().cpu().reshape(-1).tolist()
            ),
        )


@dataclass(frozen=True, slots=True)
class PendingDecision:
    world_id: str
    environment_seed: int
    policy_sampling_seed: int
    tick_phase: str
    agent_id: int
    decision_index: int
    observation: tuple[float, ...]
    previous_feedback: PreviousPublicFeedback
    action_mask: tuple[bool, ...]
    action_index: int
    requested_action: str
    logprob: float
    entropy: float
    value: float
    hidden: tuple[float, ...]
    next_hidden: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RecurrentRolloutStep:
    """One policy decision aligned to its finalized simulator transition."""

    world_id: str
    world_seed: int
    tick: int
    agent_id: int
    decision_index: int
    observation: tuple[float, ...]
    previous_feedback: PreviousPublicFeedback
    action_mask: tuple[bool, ...]
    hidden: tuple[float, ...]
    action_index: int
    requested_action: str
    logprob: float
    entropy: float
    value: float
    reward: float
    reward_components: dict[str, float]
    resolved_action_index: int
    resolved_action: str
    resolution_action_mask: tuple[bool, ...]
    action_valid: bool
    resolution_action_valid: bool
    moved: bool
    outcome: dict[str, object]
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: float | None = None
    passive_terminal_reward: float = 0.0
    passive_terminal_reward_components: dict[str, float] = field(default_factory=dict)
    passive_terminal_tick: int | None = None
    environment_seed: int | None = None
    policy_sampling_seed: int | None = None

    def __post_init__(self) -> None:
        world_seed = _strict_int(self.world_seed, field="world_seed")
        environment_seed = (
            world_seed
            if self.environment_seed is None
            else _strict_int(self.environment_seed, field="environment_seed")
        )
        if environment_seed != world_seed:
            raise RecurrentRolloutError(
                "environment_seed must match the legacy world_seed alias"
            )
        policy_sampling_seed = self.policy_sampling_seed
        if policy_sampling_seed is None:
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=f"legacy-rollout-step:{self.world_id}"
            )
        policy_sampling_seed = _policy_sampling_seed(
            policy_sampling_seed,
            field="policy_sampling_seed",
        )
        object.__setattr__(self, "world_seed", world_seed)
        object.__setattr__(self, "environment_seed", environment_seed)
        object.__setattr__(self, "policy_sampling_seed", policy_sampling_seed)
        reward, reward_components = _validated_reward_values(
            self.reward,
            self.reward_components,
            field="rollout reward",
        )
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "reward_components", reward_components)
        if self.passive_terminal_tick is None:
            if self.passive_terminal_reward != 0.0:
                raise RecurrentRolloutError(
                    "passive terminal reward requires a passive terminal tick"
                )
            if self.passive_terminal_reward_components:
                raise RecurrentRolloutError(
                    "passive terminal reward components require a passive terminal tick"
                )
        else:
            passive_reward, passive_components = _validated_reward_values(
                self.passive_terminal_reward,
                self.passive_terminal_reward_components,
                field="passive terminal reward",
            )
            object.__setattr__(self, "passive_terminal_reward", passive_reward)
            object.__setattr__(
                self,
                "passive_terminal_reward_components",
                passive_components,
            )


@dataclass(frozen=True, slots=True)
class RecurrentAdvantageRow:
    step: RecurrentRolloutStep
    advantage: float
    return_target: float


class RecurrentRolloutBuffer:
    """Sequence-preserving rollout storage with agent-local GAE boundaries."""

    def __init__(self) -> None:
        self._steps: list[RecurrentRolloutStep] = []
        self._indices_by_agent: dict[tuple[str, int], list[int]] = {}
        self._world_ids: set[str] = set()
        self._world_seed_provenance: dict[str, tuple[int, int] | None] = {}

    @property
    def steps(self) -> tuple[RecurrentRolloutStep, ...]:
        return tuple(self._steps)

    @property
    def world_seed_provenance(self) -> dict[str, dict[str, int]]:
        return {
            world_id: {
                "environment_seed": provenance[0],
                "policy_sampling_seed": provenance[1],
            }
            for world_id, provenance in self._world_seed_provenance.items()
            if provenance is not None
        }

    def register_world(
        self,
        world_id: str,
        *,
        environment_seed: int | None = None,
        policy_sampling_seed: int | None = None,
    ) -> None:
        if not world_id:
            raise RecurrentRolloutError("world_id must not be empty")
        if world_id in self._world_ids:
            raise RecurrentRolloutError(f"world_id already collected: {world_id!r}")
        if (environment_seed is None) != (policy_sampling_seed is None):
            raise RecurrentRolloutError(
                "world seed provenance requires both environment and policy seeds"
            )
        provenance: tuple[int, int] | None = None
        if environment_seed is not None and policy_sampling_seed is not None:
            provenance = (
                _strict_int(environment_seed, field="environment_seed"),
                _policy_sampling_seed(
                    policy_sampling_seed,
                    field="policy_sampling_seed",
                ),
            )
        self._world_ids.add(world_id)
        self._world_seed_provenance[world_id] = provenance

    def append(self, step: RecurrentRolloutStep) -> None:
        if step.world_id not in self._world_ids:
            raise RecurrentRolloutError(
                f"rollout world was not registered: {step.world_id!r}"
            )
        observed_provenance = (
            int(step.environment_seed),
            int(step.policy_sampling_seed),
        )
        registered_provenance = self._world_seed_provenance[step.world_id]
        if registered_provenance is None:
            self._world_seed_provenance[step.world_id] = observed_provenance
        elif observed_provenance != registered_provenance:
            raise RecurrentRolloutError(
                "rollout step seed provenance does not match its registered world"
            )
        key = (step.world_id, step.agent_id)
        indices = self._indices_by_agent.setdefault(key, [])
        if indices:
            previous = self._steps[indices[-1]]
            if previous.terminated or previous.truncated:
                raise RecurrentRolloutError(
                    "cannot append after an agent episode boundary: "
                    f"world={step.world_id!r} agent={step.agent_id}"
                )
            if step.tick <= previous.tick:
                raise RecurrentRolloutError(
                    "agent rollout ticks must increase strictly: "
                    f"previous={previous.tick} current={step.tick}"
                )
        if step.terminated and step.truncated:
            raise RecurrentRolloutError("a rollout step cannot terminate and truncate")
        indices.append(len(self._steps))
        self._steps.append(step)

    def mark_passive_terminal(
        self,
        *,
        world_id: str,
        agent_id: int,
        tick: int,
        reward: float,
        reward_components: Mapping[str, object],
    ) -> bool:
        """Attach a no-action terminal transition without fabricating an action.

        An agent can be killed before its turn. The simulator then emits a
        passive terminal trajectory record, but there is no policy log-prob for
        that record. Its reward is kept separately and discounted after the
        agent's last real action during GAE.
        """

        key = (world_id, agent_id)
        indices = self._indices_by_agent.get(key)
        if not indices:
            return False
        index = indices[-1]
        step = self._steps[index]
        if step.terminated or step.truncated:
            raise RecurrentRolloutError(
                "passive terminal arrived after a closed agent rollout"
            )
        if tick != step.tick + 1:
            raise RecurrentRolloutError(
                "passive terminal tick must immediately follow the last policy "
                f"decision: previous={step.tick} terminal={tick}"
            )
        reward_total, components = _validated_reward_values(
            reward,
            reward_components,
            field="passive terminal reward",
        )
        self._steps[index] = replace(
            step,
            terminated=True,
            passive_terminal_reward=reward_total,
            passive_terminal_reward_components=components,
            passive_terminal_tick=int(tick),
        )
        return True

    def mark_truncated(
        self,
        *,
        world_id: str,
        agent_id: int,
        bootstrap_value: float,
    ) -> bool:
        key = (world_id, agent_id)
        indices = self._indices_by_agent.get(key)
        if not indices:
            return False
        index = indices[-1]
        step = self._steps[index]
        if step.terminated:
            return False
        if step.truncated:
            raise RecurrentRolloutError("agent rollout was already truncated")
        self._steps[index] = replace(
            step,
            truncated=True,
            bootstrap_value=_finite_float(
                bootstrap_value,
                field="bootstrap value",
            ),
        )
        return True

    def validate_world_closed(self, world_id: str) -> None:
        open_agents = [
            agent_id
            for (
                candidate_world_id,
                agent_id,
            ), indices in self._indices_by_agent.items()
            if candidate_world_id == world_id
            and indices
            and not (
                self._steps[indices[-1]].terminated
                or self._steps[indices[-1]].truncated
            )
        ]
        if open_agents:
            raise RecurrentRolloutError(
                "world rollout has unclosed agent sequences; run one bootstrap "
                "tick beyond rollout_ticks with record_trajectory=True: "
                f"{sorted(open_agents)}"
            )

    def sequences(self) -> tuple[tuple[RecurrentRolloutStep, ...], ...]:
        ordered = sorted(
            self._indices_by_agent.values(),
            key=lambda indices: indices[0],
        )
        return tuple(
            tuple(self._steps[index] for index in indices) for indices in ordered
        )

    def compute_gae(
        self,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[RecurrentAdvantageRow, ...]:
        gamma = _unit_interval(gamma, field="gamma", allow_zero=True)
        gae_lambda = _unit_interval(
            gae_lambda,
            field="gae_lambda",
            allow_zero=True,
        )
        advantages: dict[int, float] = {}
        returns: dict[int, float] = {}
        for indices in self._indices_by_agent.values():
            if not indices:
                continue
            last = self._steps[indices[-1]]
            if not (last.terminated or last.truncated):
                raise RecurrentRolloutError(
                    "cannot compute GAE for an open agent sequence"
                )
            next_advantage = 0.0
            for position in range(len(indices) - 1, -1, -1):
                index = indices[position]
                step = self._steps[index]
                is_last = position == len(indices) - 1
                if is_last and step.truncated:
                    if step.bootstrap_value is None:
                        raise RecurrentRolloutError(
                            "truncated rollout is missing a bootstrap value"
                        )
                    next_value = step.bootstrap_value
                    continuation = 1.0
                elif is_last:
                    next_value = 0.0
                    continuation = 0.0
                else:
                    next_value = self._steps[indices[position + 1]].value
                    continuation = 1.0

                effective_reward = step.reward
                if is_last and step.passive_terminal_tick is not None:
                    effective_reward += gamma * step.passive_terminal_reward
                delta = (
                    effective_reward + gamma * continuation * next_value - step.value
                )
                advantage = delta + gamma * gae_lambda * continuation * next_advantage
                advantages[index] = advantage
                returns[index] = advantage + step.value
                next_advantage = advantage

        return tuple(
            RecurrentAdvantageRow(
                step=step,
                advantage=advantages[index],
                return_target=returns[index],
            )
            for index, step in enumerate(self._steps)
        )


class RecurrentOnPolicyCollector:
    """Policy-induced recurrent rollout collector for shared-policy PPO/IPPO.

    `SimulationWorld` finalizes transitions only after the complete multi-agent
    tick. `observe_transition` is therefore the alignment boundary: requested
    action, old log-prob/value, public input, reward, and resolved outcome are
    joined only there.

    For an unbiased finite-horizon truncation, configure the world for exactly
    `rollout_ticks + 1` ticks. The final tick is used only to evaluate V(s_T)
    from the next policy-visible observation. Its actions and rewards are not
    inserted into the rollout. `finish_world` fails closed if that bootstrap
    tick was omitted.
    """

    policy_id = RECURRENT_ROLLOUT_POLICY_ID
    policy_version = RECURRENT_ROLLOUT_POLICY_VERSION

    def __init__(
        self,
        core: RecurrentPolicyCore,
        *,
        buffer: RecurrentRolloutBuffer | None = None,
        reset_recurrent_state_each_decision: bool = False,
    ) -> None:
        if isinstance(core.hidden_size, bool) or int(core.hidden_size) <= 0:
            raise RecurrentRolloutError("core.hidden_size must be positive")
        if type(reset_recurrent_state_each_decision) is not bool:
            raise RecurrentRolloutError(
                "reset_recurrent_state_each_decision must be an exact boolean"
            )
        self._core = core
        self._public_input_schema_version = getattr(
            core,
            "public_input_schema_version",
            ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        )
        self._public_input_size = getattr(
            core,
            "public_input_size",
            ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        )
        self._learned_input_size = getattr(
            core,
            "learned_input_size",
            (
                self._public_input_size
                + len(RECURRENT_ROLLOUT_ACTIONS)
                + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
            ),
        )
        if self._public_input_schema_version not in {
            ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
            TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        }:
            raise RecurrentRolloutError("core public input schema is unsupported")
        if (
            isinstance(self._public_input_size, bool)
            or not isinstance(self._public_input_size, int)
            or self._public_input_size <= 0
        ):
            raise RecurrentRolloutError("core public input size must be positive")
        expected_learned_input_size = (
            self._public_input_size
            + len(RECURRENT_ROLLOUT_ACTIONS)
            + RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE
        )
        if self._learned_input_size != expected_learned_input_size:
            raise RecurrentRolloutError(
                "core learned input size does not match its public input contract"
            )
        self.buffer = buffer if buffer is not None else RecurrentRolloutBuffer()
        self._reset_recurrent_state_each_decision = reset_recurrent_state_each_decision
        self._active_world_id: str | None = None
        self._environment_seed = 0
        self._policy_sampling_seed = 0
        self._rollout_ticks = 0
        self._bootstrap_phase = False
        self._rng = random.Random()
        self._decision_index = 0
        self._hidden_by_agent: dict[int, tuple[float, ...]] = {}
        self._feedback_by_agent: dict[int, PreviousPublicFeedback] = {}
        self._pending_by_agent: dict[int, PendingDecision] = {}

    def start_world(
        self,
        *,
        world_id: str,
        rollout_ticks: int,
        environment_seed: int | None = None,
        policy_sampling_seed: int | None = None,
        seed: int | None = None,
    ) -> None:
        if self._active_world_id is not None:
            raise RecurrentRolloutError(
                "finish the active world before starting another"
            )
        if environment_seed is None:
            environment_seed = seed
        elif seed is not None and seed != environment_seed:
            raise RecurrentRolloutError(
                "environment_seed and legacy seed alias disagree"
            )
        if isinstance(environment_seed, bool) or not isinstance(
            environment_seed,
            int,
        ):
            raise RecurrentRolloutError("environment_seed must be an integer")
        if policy_sampling_seed is None:
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=f"legacy-world-id:{world_id}"
            )
        policy_sampling_seed = _policy_sampling_seed(
            policy_sampling_seed,
            field="policy_sampling_seed",
        )
        if isinstance(rollout_ticks, bool) or int(rollout_ticks) <= 0:
            raise RecurrentRolloutError("rollout_ticks must be positive")
        self.buffer.register_world(
            world_id,
            environment_seed=environment_seed,
            policy_sampling_seed=policy_sampling_seed,
        )
        self._active_world_id = world_id
        self._environment_seed = int(environment_seed)
        self._policy_sampling_seed = policy_sampling_seed
        self._rollout_ticks = int(rollout_ticks)
        self._bootstrap_phase = False
        self._rng = random.Random(policy_sampling_seed)
        self._decision_index = 0
        self._hidden_by_agent.clear()
        self._feedback_by_agent.clear()
        self._pending_by_agent.clear()

    def finish_world(self) -> None:
        world_id = self._require_active_world()
        if self._pending_by_agent:
            raise RecurrentRolloutError(
                "world finished with decisions that have no finalized transition"
            )
        self.buffer.validate_world_closed(world_id)
        self._hidden_by_agent.clear()
        self._feedback_by_agent.clear()
        self._pending_by_agent.clear()
        self._active_world_id = None
        self._environment_seed = 0
        self._policy_sampling_seed = 0
        self._bootstrap_phase = False

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        world_id = self._require_active_world()
        agent_id = _agent_id_from_observation(observation)
        if agent_id in self._pending_by_agent:
            raise RecurrentRolloutError(
                f"agent {agent_id} has an unfinalized previous decision"
            )
        policy_input = ecological_policy_input_values(
            encode_observation_input(observation)
        )
        observed_schema_version = (
            TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
            if observation.get("schema_version")
            == TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
            else ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
        )
        if observed_schema_version != self._public_input_schema_version:
            raise RecurrentRolloutError(
                "ecological policy input schema does not match the recurrent core"
            )
        if len(policy_input) != self._public_input_size:
            raise RecurrentRolloutError(
                "ecological policy input size does not match the recurrent core: "
                f"{len(policy_input)} != {self._public_input_size}"
            )
        mask = _action_mask_tuple(action_mask)
        hidden = (
            None
            if self._reset_recurrent_state_each_decision
            else self._hidden_by_agent.get(agent_id)
        )
        if hidden is None:
            hidden = self._initial_hidden()
        previous_feedback = self._feedback_by_agent.get(
            agent_id,
            PreviousPublicFeedback.zero(),
        )
        output = self._validated_core_output(
            self._core.forward_step(
                policy_input,
                mask,
                previous_feedback,
                hidden,
            )
        )
        action_index, logprob, entropy = _sample_masked_action(
            output.logits,
            mask,
            rng=self._rng,
        )
        requested_action = RECURRENT_ROLLOUT_ACTIONS[action_index]
        tick_phase = "bootstrap" if self._bootstrap_phase else "rollout"
        decision_index = self._decision_index
        self._decision_index += 1
        pending = PendingDecision(
            world_id=world_id,
            environment_seed=self._environment_seed,
            policy_sampling_seed=self._policy_sampling_seed,
            tick_phase=tick_phase,
            agent_id=agent_id,
            decision_index=decision_index,
            observation=tuple(policy_input),
            previous_feedback=previous_feedback,
            action_mask=mask,
            action_index=action_index,
            requested_action=requested_action,
            logprob=logprob,
            entropy=entropy,
            value=output.value,
            hidden=hidden,
            next_hidden=output.next_hidden,
        )
        self._pending_by_agent[agent_id] = pending
        if not self._reset_recurrent_state_each_decision:
            self._hidden_by_agent[agent_id] = output.next_hidden
        return ActionDecision(
            requested_action=requested_action,
            source=RECURRENT_ROLLOUT_ACTION_SOURCE,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics={
                "schema_version": RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION,
                "world_id": world_id,
                "environment_seed": self._environment_seed,
                "policy_sampling_seed": self._policy_sampling_seed,
                "policy_sampling_seed_namespace": (
                    RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
                ),
                "decision_index": decision_index,
                "agent_id": agent_id,
                "phase": tick_phase,
                "policy_input_schema_version": self._public_input_schema_version,
                "policy_input_size": self._public_input_size,
                "learned_input_size": self._learned_input_size,
                "previous_feedback_schema_version": (
                    RECURRENT_PUBLIC_FEEDBACK_SCHEMA_VERSION
                ),
                "previous_feedback_available": previous_feedback.available,
                "recurrent_state_reset_each_decision": (
                    self._reset_recurrent_state_each_decision
                ),
                "action_index": action_index,
                "logprob": round(logprob, 12),
                "value": round(output.value, 12),
            },
        )

    def observe_transition(
        self,
        record: dict[str, object],
    ) -> dict[str, object] | None:
        world_id = self._require_active_world()
        agent_id = _record_agent_id(record)
        tick = _record_tick(record)
        pending = self._pending_by_agent.pop(agent_id, None)

        if pending is None:
            self._observe_passive_record(
                record,
                world_id=world_id,
                agent_id=agent_id,
                tick=tick,
            )
            if tick == self._rollout_ticks - 1:
                self._bootstrap_phase = True
            return None

        self._validate_alignment(record, pending=pending, tick=tick)
        if pending.tick_phase == "bootstrap":
            self.buffer.mark_truncated(
                world_id=world_id,
                agent_id=agent_id,
                bootstrap_value=pending.value,
            )
            self._hidden_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)
            return {
                "schema_version": RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION,
                "decision_index": pending.decision_index,
                "phase": "bootstrap",
                "collected": False,
                "environment_seed": pending.environment_seed,
                "policy_sampling_seed": pending.policy_sampling_seed,
            }

        if tick >= self._rollout_ticks:
            raise RecurrentRolloutError(
                "rollout decision finalized beyond configured rollout horizon"
            )
        reward, reward_components = _reward_payload(record)
        terminated = _record_terminated(record)
        outcome = _mapping(record.get("outcome"), field="outcome")
        resolved_action = _action_name(
            record.get("resolved_action"),
            field="resolved_action",
        )
        resolution_action_mask = _action_mask_tuple(
            _mapping(
                record.get("resolution_action_mask"),
                field="resolution_action_mask",
            )
        )
        step = RecurrentRolloutStep(
            world_id=world_id,
            world_seed=pending.environment_seed,
            tick=tick,
            agent_id=agent_id,
            decision_index=pending.decision_index,
            observation=pending.observation,
            previous_feedback=pending.previous_feedback,
            action_mask=pending.action_mask,
            hidden=pending.hidden,
            action_index=pending.action_index,
            requested_action=pending.requested_action,
            logprob=pending.logprob,
            entropy=pending.entropy,
            value=pending.value,
            reward=reward,
            reward_components=reward_components,
            resolved_action_index=RECURRENT_ROLLOUT_ACTIONS.index(resolved_action),
            resolved_action=resolved_action,
            resolution_action_mask=resolution_action_mask,
            action_valid=_strict_bool(record.get("action_valid"), field="action_valid"),
            resolution_action_valid=_strict_bool(
                record.get("resolution_action_valid"),
                field="resolution_action_valid",
            ),
            moved=_strict_bool(record.get("moved"), field="moved"),
            outcome=copy.deepcopy(dict(outcome)),
            terminated=terminated,
            environment_seed=pending.environment_seed,
            policy_sampling_seed=pending.policy_sampling_seed,
        )
        self.buffer.append(step)
        if terminated:
            self._hidden_by_agent.pop(agent_id, None)
            self._feedback_by_agent.pop(agent_id, None)
        else:
            self._feedback_by_agent[agent_id] = PreviousPublicFeedback.from_record(
                record
            )
        if tick == self._rollout_ticks - 1:
            self._bootstrap_phase = True
        return {
            "schema_version": RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION,
            "decision_index": pending.decision_index,
            "phase": "rollout",
            "collected": True,
            "terminated": terminated,
            "environment_seed": pending.environment_seed,
            "policy_sampling_seed": pending.policy_sampling_seed,
        }

    def _observe_passive_record(
        self,
        record: Mapping[str, object],
        *,
        world_id: str,
        agent_id: int,
        tick: int,
    ) -> None:
        if str(record.get("action_source")) != "passive":
            raise RecurrentRolloutError(
                "transition has no pending policy decision and is not passive"
            )
        if not _record_terminated(record):
            raise RecurrentRolloutError("passive transition must terminate the agent")

        if tick > self._rollout_ticks:
            raise RecurrentRolloutError(
                "passive terminal arrived beyond the one-tick bootstrap boundary"
            )
        reward, reward_components = _reward_payload(record)
        self.buffer.mark_passive_terminal(
            world_id=world_id,
            agent_id=agent_id,
            tick=tick,
            reward=reward,
            reward_components=reward_components,
        )
        self._hidden_by_agent.pop(agent_id, None)
        self._feedback_by_agent.pop(agent_id, None)

    def _validate_alignment(
        self,
        record: Mapping[str, object],
        *,
        pending: PendingDecision,
        tick: int,
    ) -> None:
        diagnostics = _mapping(
            record.get("policy_decision_diagnostics"),
            field="policy_decision_diagnostics",
        )
        if (
            diagnostics.get("schema_version")
            != RECURRENT_ROLLOUT_DIAGNOSTIC_SCHEMA_VERSION
        ):
            raise RecurrentRolloutError(
                "transition decision diagnostic schema mismatch"
            )
        if diagnostics.get("world_id") != pending.world_id:
            raise RecurrentRolloutError("transition world_id does not match decision")
        if diagnostics.get("environment_seed") != pending.environment_seed:
            raise RecurrentRolloutError(
                "transition environment_seed does not match decision"
            )
        if diagnostics.get("policy_sampling_seed") != pending.policy_sampling_seed:
            raise RecurrentRolloutError(
                "transition policy_sampling_seed does not match decision"
            )
        if (
            diagnostics.get("policy_sampling_seed_namespace")
            != RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
        ):
            raise RecurrentRolloutError(
                "transition policy sampling namespace does not match collector"
            )
        if diagnostics.get("decision_index") != pending.decision_index:
            raise RecurrentRolloutError("transition decision_index does not match")
        if diagnostics.get("phase") != pending.tick_phase:
            raise RecurrentRolloutError("transition phase does not match decision")
        if record.get("policy_id") != self.policy_id:
            raise RecurrentRolloutError("transition policy_id does not match collector")
        if record.get("policy_version") != self.policy_version:
            raise RecurrentRolloutError(
                "transition policy_version does not match collector"
            )
        if record.get("action_source") != RECURRENT_ROLLOUT_ACTION_SOURCE:
            raise RecurrentRolloutError(
                "transition action source does not match collector"
            )
        if record.get("requested_action") != pending.requested_action:
            raise RecurrentRolloutError("transition requested action does not match")
        if (
            _action_mask_tuple(_mapping(record.get("action_mask"), field="action_mask"))
            != pending.action_mask
        ):
            raise RecurrentRolloutError(
                "transition action mask does not match decision"
            )
        expected_bootstrap_tick = self._rollout_ticks
        if pending.tick_phase == "bootstrap" and tick != expected_bootstrap_tick:
            raise RecurrentRolloutError(
                "bootstrap decision finalized at unexpected tick: "
                f"expected={expected_bootstrap_tick} actual={tick}"
            )

    def _initial_hidden(self) -> tuple[float, ...]:
        hidden = tuple(
            _finite_float(value, field="initial hidden state")
            for value in self._core.initial_hidden()
        )
        if len(hidden) != int(self._core.hidden_size):
            raise RecurrentRolloutError(
                "initial hidden state length does not match core.hidden_size"
            )
        return hidden

    def _validated_core_output(
        self,
        output: RecurrentCoreOutput,
    ) -> RecurrentCoreOutput:
        logits = tuple(
            _finite_float(value, field="action logit") for value in output.logits
        )
        if len(logits) != len(RECURRENT_ROLLOUT_ACTIONS):
            raise RecurrentRolloutError(
                "core must emit one logit per stable action: "
                f"expected={len(RECURRENT_ROLLOUT_ACTIONS)} actual={len(logits)}"
            )
        next_hidden = tuple(
            _finite_float(value, field="next hidden state")
            for value in output.next_hidden
        )
        if len(next_hidden) != int(self._core.hidden_size):
            raise RecurrentRolloutError(
                "next hidden state length does not match core.hidden_size"
            )
        return RecurrentCoreOutput(
            logits=logits,
            value=_finite_float(output.value, field="value estimate"),
            next_hidden=next_hidden,
        )

    def _require_active_world(self) -> str:
        if self._active_world_id is None:
            raise RecurrentRolloutError("start_world must be called before collection")
        return self._active_world_id


def _sample_masked_action(
    logits: Sequence[float],
    mask: Sequence[bool],
    *,
    rng: random.Random,
) -> tuple[int, float, float]:
    valid_indices = [index for index, valid in enumerate(mask) if valid]
    if not valid_indices:
        raise RecurrentRolloutError("action mask contains no valid action")
    max_logit = max(logits[index] for index in valid_indices)
    weights = [math.exp(logits[index] - max_logit) for index in valid_indices]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise RecurrentRolloutError("masked action distribution is not finite")
    threshold = rng.random() * total
    cumulative = 0.0
    selected_offset = len(valid_indices) - 1
    for offset, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            selected_offset = offset
            break
    probabilities = [weight / total for weight in weights]
    probability = probabilities[selected_offset]
    logprob = math.log(probability)
    entropy = -sum(
        candidate * math.log(candidate)
        for candidate in probabilities
        if candidate > 0.0
    )
    return valid_indices[selected_offset], logprob, entropy


def _action_mask_tuple(action_mask: Mapping[str, object]) -> tuple[bool, ...]:
    if set(action_mask) != set(RECURRENT_ROLLOUT_ACTIONS):
        missing = sorted(set(RECURRENT_ROLLOUT_ACTIONS) - set(action_mask))
        extra = sorted(set(action_mask) - set(RECURRENT_ROLLOUT_ACTIONS))
        raise RecurrentRolloutError(
            f"action mask keys drifted; missing={missing} extra={extra}"
        )
    return tuple(
        _strict_bool(action_mask[action], field=f"action_mask.{action}")
        for action in RECURRENT_ROLLOUT_ACTIONS
    )


def _agent_id_from_observation(observation: Mapping[str, object]) -> int:
    metadata = _mapping(observation.get("metadata"), field="observation.metadata")
    return _strict_int(metadata.get("agent_id"), field="observation.metadata.agent_id")


def _record_agent_id(record: Mapping[str, object]) -> int:
    return _strict_int(record.get("agent_id"), field="record.agent_id")


def _record_tick(record: Mapping[str, object]) -> int:
    tick = _strict_int(record.get("tick"), field="record.tick")
    if tick < 0:
        raise RecurrentRolloutError("record.tick must not be negative")
    return tick


def _record_terminated(record: Mapping[str, object]) -> bool:
    outcome = _mapping(record.get("outcome"), field="outcome")
    died = _strict_bool(outcome.get("died"), field="outcome.died")
    after = _mapping(record.get("after"), field="after")
    alive = _strict_bool(after.get("alive"), field="after.alive")
    if died == alive:
        raise RecurrentRolloutError("outcome.died and after.alive disagree")
    return died


def _reward_total(record: Mapping[str, object]) -> float:
    reward_total, _components = _reward_payload(record)
    return reward_total


def _reward_payload(
    record: Mapping[str, object],
) -> tuple[float, dict[str, float]]:
    reward = _mapping(record.get("reward"), field="reward")
    if reward.get("schema_version") != REWARD_SCHEMA_VERSION:
        raise RecurrentRolloutError(
            "reward.schema_version does not match the canonical reward contract"
        )
    return _validated_reward_values(
        reward.get("total"),
        _mapping(reward.get("components"), field="reward.components"),
        field="reward",
    )


def _validated_reward_values(
    total: object,
    components: Mapping[str, object],
    *,
    field: str,
) -> tuple[float, dict[str, float]]:
    observed_keys = set(components)
    expected_keys = set(RECURRENT_REWARD_COMPONENTS)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RecurrentRolloutError(
            f"{field} component keys drifted; missing={missing} extra={extra}"
        )
    parsed: dict[str, float] = {}
    for name in RECURRENT_REWARD_COMPONENTS:
        value = _finite_float(
            components[name],
            field=f"{field}.components.{name}",
        )
        lower, upper = REWARD_COMPONENT_BOUNDS[name]
        if not lower <= value <= upper:
            raise RecurrentRolloutError(
                f"{field}.components.{name} is outside canonical bounds "
                f"[{lower}, {upper}]"
            )
        parsed[name] = value

    reward_total = _finite_float(total, field=f"{field}.total")
    if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
        raise RecurrentRolloutError(
            f"{field}.total is outside the public reward contract bounds"
        )
    expected_total = round(math.fsum(parsed.values()), 4)
    if not math.isclose(reward_total, expected_total, rel_tol=0.0, abs_tol=1.0e-9):
        raise RecurrentRolloutError(
            f"{field}.total does not equal its canonical rounded component sum: "
            f"{reward_total} != {expected_total}"
        )
    return reward_total, parsed


def _bounded_reward_total(record: Mapping[str, object]) -> float:
    reward_total = _reward_total(record)
    if not REWARD_TOTAL_BOUNDS[0] <= reward_total <= REWARD_TOTAL_BOUNDS[1]:
        raise RecurrentRolloutError(
            "reward.total is outside the public reward contract bounds"
        )
    return reward_total


def _action_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in RECURRENT_ROLLOUT_ACTIONS:
        raise RecurrentRolloutError(f"{field} is not a stable action")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentRolloutError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise RecurrentRolloutError(f"{field} keys must be strings")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise RecurrentRolloutError(f"{field} must be a boolean")
    return value


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentRolloutError(f"{field} must be an integer")
    return int(value)


def _policy_sampling_seed(value: object, *, field: str) -> int:
    parsed = _strict_int(value, field=field)
    if parsed < 1 or parsed > MAX_RECURRENT_POLICY_SAMPLING_SEED:
        raise RecurrentRolloutError(
            f"{field} must be in [1, {MAX_RECURRENT_POLICY_SAMPLING_SEED}]"
        )
    return parsed


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentRolloutError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentRolloutError(f"{field} must be finite")
    return parsed


def _unit_interval(
    value: object,
    *,
    field: str,
    allow_zero: bool,
) -> float:
    parsed = _finite_float(value, field=field)
    lower_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not lower_ok or parsed > 1.0:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        raise RecurrentRolloutError(f"{field} must be in {bracket}")
    return parsed
