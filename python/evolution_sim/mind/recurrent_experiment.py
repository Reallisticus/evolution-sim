from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import torch

from evolution_sim.config.schema import WorldConfig
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.evaluation_harness import (
    CONTROLLED_FIXTURE_NAMES,
    _fixture_world,
)
from evolution_sim.mind.recurrent_actor_critic import (
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
    RecurrentCounterfactualAggregateGroup,
    RecurrentCounterfactualAuxiliaryConfig,
    RecurrentCounterfactualAuxiliaryStepConfig,
    RecurrentCounterfactualAuxiliaryStepDiagnostics,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RecurrentCounterfactualCollectionConfig,
    RecurrentCounterfactualCollectionResult,
    RecurrentCounterfactualCollectionTask,
    collect_recurrent_counterfactual_bundles,
    recurrent_counterfactual_collection_config_payload,
)
from evolution_sim.mind.recurrent_policy import (
    frozen_cpu_model_copy,
    recurrent_model_state_sha256,
)
from evolution_sim.mind.recurrent_ppo import (
    PPOUpdateDiagnostics,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    ppo_sequences_from_rollout_buffer,
)
from evolution_sim.mind.recurrent_rollout import (
    MAX_RECURRENT_POLICY_SAMPLING_SEED,
    RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
    RECURRENT_REWARD_COMPONENTS,
    RECURRENT_ROLLOUT_ACTIONS,
    RecurrentOnPolicyCollector,
    RecurrentRolloutBuffer,
    RecurrentRolloutStep,
    TorchRecurrentPolicyCore,
    derive_recurrent_policy_sampling_seed,
)
from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_SEED_REGISTRY,
)


RECURRENT_EXPERIMENT_CONTRACT_VERSION = "mind_public_recurrent_ippo_experiment_v4"
RECURRENT_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_public_recurrent_training_seed_provenance_v1"
)
RECURRENT_SCALE_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_public_recurrent_scale_training_seed_provenance_v1"
)
RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION = (
    "mind_public_recurrent_ippo_training_schedule_task_v1"
)
RECURRENT_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION = (
    "mind_public_recurrent_counterfactual_collection_task_v1"
)
RECURRENT_SCALE_POLICY_SAMPLING_TASK_IDENTITY_VERSION = (
    "mind_public_recurrent_ippo_scale_training_schedule_task_v1"
)
RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL = "canonical_v1"
RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT = "scale_development_v1"
RECURRENT_TRAINING_SEED_REGISTRY_CONTRACTS = (
    RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL,
    RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
)
RECURRENT_BROAD_SCENARIO = "broad"
RECURRENT_TRAINING_SCENARIOS: tuple[str, ...] = (
    RECURRENT_BROAD_SCENARIO,
    *CONTROLLED_FIXTURE_NAMES,
)
MAX_RECURRENT_ROLLOUT_WORKERS = 64
RECURRENT_ROLLOUT_WORKER_START_METHOD = "spawn"
RECURRENT_ROLLOUT_WORKER_TORCH_THREADS = 1


_WORKER_BEHAVIOR_MODEL: PublicRecurrentActorCritic | None = None
_WORKER_FEED_FORWARD_HISTORY_ABLATION: bool | None = None


class RecurrentExperimentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualExperimentConfig:
    """Opt-in exact-branch auxiliary stage applied after each PPO update.

    This configuration is deliberately absent by default. Branch evidence is
    collected from the exact post-PPO model and may be consumed by one
    transactional auxiliary step only; it is never PPO ratio data.
    """

    collection: RecurrentCounterfactualCollectionConfig
    auxiliary: RecurrentCounterfactualAuxiliaryConfig
    step: RecurrentCounterfactualAuxiliaryStepConfig = (
        RecurrentCounterfactualAuxiliaryStepConfig()
    )
    bundles_per_update: int = 2
    branch_tick_candidates: tuple[int, ...] = (16, 32, 48, 64)
    workers: int = 1

    def __post_init__(self) -> None:
        if not isinstance(
            self.collection,
            RecurrentCounterfactualCollectionConfig,
        ):
            raise RecurrentExperimentError(
                "counterfactual collection config has the wrong type"
            )
        if not isinstance(
            self.auxiliary,
            RecurrentCounterfactualAuxiliaryConfig,
        ):
            raise RecurrentExperimentError(
                "counterfactual auxiliary config has the wrong type"
            )
        if not isinstance(self.step, RecurrentCounterfactualAuxiliaryStepConfig):
            raise RecurrentExperimentError(
                "counterfactual auxiliary step config has the wrong type"
            )
        _positive_int(self.bundles_per_update, field="bundles_per_update")
        if (
            not isinstance(self.branch_tick_candidates, tuple)
            or not self.branch_tick_candidates
        ):
            raise RecurrentExperimentError(
                "branch_tick_candidates must be a non-empty tuple"
            )
        seen: set[int] = set()
        for tick in self.branch_tick_candidates:
            if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
                raise RecurrentExperimentError(
                    "branch tick candidates must be non-negative integers"
                )
            if tick in seen:
                raise RecurrentExperimentError("branch tick candidates must be unique")
            seen.add(tick)
        if tuple(sorted(seen)) != self.branch_tick_candidates:
            raise RecurrentExperimentError(
                "branch tick candidates must be strictly increasing"
            )
        _rollout_worker_count(self.workers)
        collection_horizons = tuple(self.collection.horizons)
        scalarization_horizons = tuple(
            horizon for horizon, _weight in self.auxiliary.scalarization.horizon_weights
        )
        if collection_horizons != scalarization_horizons:
            raise RecurrentExperimentError(
                "counterfactual collection and scalarization horizons must match"
            )
        if (
            self.auxiliary.terminal_target_weight > 0.0
            and self.collection.terminal_target_world_tick is None
        ):
            raise RecurrentExperimentError(
                "terminal target weight requires an absolute collection target"
            )
        if (
            not self.collection.multi_tape_enabled
            and self.auxiliary.terminal_target_weight != 0.0
        ):
            raise RecurrentExperimentError(
                "legacy one-tape collection cannot use aggregate terminal targets"
            )


@dataclass(frozen=True, slots=True)
class RecurrentRolloutTask:
    task_id: str
    scenario: str
    environment_seed: int
    rollout_ticks: int
    policy_sampling_identity: str | None = None
    policy_sampling_seed: int | None = None
    seed_role: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or self.task_id != self.task_id.strip():
            raise RecurrentExperimentError("task_id must be non-empty and trimmed")
        if self.scenario not in RECURRENT_TRAINING_SCENARIOS:
            raise RecurrentExperimentError(
                f"unsupported recurrent training scenario: {self.scenario!r}"
            )
        environment_seed = _positive_seed(
            self.environment_seed,
            field="environment_seed",
        )
        legacy_seed_role = (
            "train" if self.scenario == RECURRENT_BROAD_SCENARIO else "curriculum"
        )
        scale_seed_role = (
            "scale_train"
            if self.scenario == RECURRENT_BROAD_SCENARIO
            else "scale_curriculum"
        )
        expected_seed_role = self.seed_role or legacy_seed_role
        if expected_seed_role not in {legacy_seed_role, scale_seed_role}:
            raise RecurrentExperimentError(
                "rollout task seed_role does not match its training scenario"
            )
        seed_registry = (
            RECURRENT_SEED_REGISTRY
            if expected_seed_role == legacy_seed_role
            else SCALE_DEVELOPMENT_SEED_REGISTRY
        )
        if environment_seed not in seed_registry[expected_seed_role]:
            raise RecurrentExperimentError(
                f"{self.scenario} training requires a canonical "
                f"{expected_seed_role} environment seed"
            )
        _positive_int(self.rollout_ticks, field="rollout_ticks")
        policy_sampling_identity = self.policy_sampling_identity
        if policy_sampling_identity is None:
            policy_sampling_identity = f"ad-hoc-task:{self.task_id}"
        if (
            not isinstance(policy_sampling_identity, str)
            or not policy_sampling_identity
            or policy_sampling_identity != policy_sampling_identity.strip()
        ):
            raise RecurrentExperimentError(
                "policy_sampling_identity must be non-empty and trimmed"
            )
        expected_policy_sampling_seed = derive_recurrent_policy_sampling_seed(
            task_identity=policy_sampling_identity
        )
        policy_sampling_seed = self.policy_sampling_seed
        if policy_sampling_seed is None:
            policy_sampling_seed = expected_policy_sampling_seed
        policy_sampling_seed = _policy_sampling_seed(
            policy_sampling_seed,
            field="policy_sampling_seed",
        )
        if policy_sampling_seed != expected_policy_sampling_seed:
            raise RecurrentExperimentError(
                "policy_sampling_seed must match its namespaced task identity"
            )
        object.__setattr__(
            self,
            "policy_sampling_identity",
            policy_sampling_identity,
        )
        object.__setattr__(self, "policy_sampling_seed", policy_sampling_seed)
        object.__setattr__(self, "seed_role", expected_seed_role)


@dataclass(frozen=True, slots=True)
class RecurrentRolloutScopeDiagnostics:
    """Action and reward measurements for one explicit rollout stratum.

    Reward totals include a passive terminal reward attached to the final real
    policy decision. Passive records never fabricate an action or mask row, so
    action counts and mask availability remain policy-decision counts.
    """

    world_count: int
    transition_count: int
    passive_terminal_count: int
    total_reward: float
    mean_reward: float
    reward_component_totals: dict[str, float]
    action_mask_availability_counts: dict[str, int]
    chosen_action_counts: dict[str, int]
    chosen_given_valid_rates: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class RecurrentRolloutBatchDiagnostics:
    world_count: int
    transition_count: int
    sequence_count: int
    terminated_sequence_count: int
    truncated_sequence_count: int
    total_reward: float
    mean_reward: float
    reward_component_totals: dict[str, float]
    action_mask_availability_counts: dict[str, int]
    chosen_action_counts: dict[str, int]
    chosen_given_valid_rates: dict[str, float | None]
    requested_action_counts: dict[str, int]
    dominant_requested_action: str
    dominant_requested_action_share: float
    world_diagnostics: dict[str, RecurrentRolloutScopeDiagnostics]
    scenario_diagnostics: dict[str, RecurrentRolloutScopeDiagnostics]
    policy_sampling_seed_namespace: str
    world_seed_provenance: dict[str, dict[str, object]]
    world_summaries: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class RecurrentTrainingUpdateResult:
    update_index: int
    tasks: tuple[RecurrentRolloutTask, ...]
    rollout: RecurrentRolloutBatchDiagnostics
    optimizer: PPOUpdateDiagnostics
    rollout_workers_requested: int
    rollout_workers_resolved: int
    counterfactual_collection: RecurrentCounterfactualCollectionResult | None
    counterfactual_auxiliary: RecurrentCounterfactualAuxiliaryStepDiagnostics | None


@dataclass(frozen=True, slots=True)
class RecurrentTrainingRunResult:
    contract_version: str
    learner_seed: int
    device: str
    deterministic_algorithms_enabled: bool
    model_config: dict[str, object]
    ppo_config: dict[str, object]
    rollout_execution: dict[str, object]
    training_scenarios: tuple[str, ...]
    environment_seed_provenance: dict[str, object]
    updates: tuple[RecurrentTrainingUpdateResult, ...]
    total_worlds: int
    total_transitions: int
    counterfactual_experiment: dict[str, object] | None


def build_recurrent_training_schedule(
    *,
    update_count: int,
    worlds_per_update: int,
    rollout_ticks: int,
    scenarios: Sequence[str] = RECURRENT_TRAINING_SCENARIOS,
    seed_registry_contract: str = RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL,
    scale_learner_seed: int | None = None,
) -> tuple[tuple[RecurrentRolloutTask, ...], ...]:
    """Build a deterministic policy-optimization schedule from train-only roles.

    The scale-development registry is an explicit opt-in.  It is disjoint from
    every historical development, validation, and lockbox seed and therefore
    cannot be selected by an existing caller accidentally.
    """

    _positive_int(update_count, field="update_count")
    _positive_int(worlds_per_update, field="worlds_per_update")
    _positive_int(rollout_ticks, field="rollout_ticks")
    selected_scenarios = tuple(scenarios)
    if not selected_scenarios:
        raise RecurrentExperimentError("training scenarios cannot be empty")
    if len(set(selected_scenarios)) != len(selected_scenarios):
        raise RecurrentExperimentError("training scenarios cannot contain duplicates")
    unsupported = [
        scenario
        for scenario in selected_scenarios
        if scenario not in RECURRENT_TRAINING_SCENARIOS
    ]
    if unsupported:
        raise RecurrentExperimentError(f"unsupported training scenarios: {unsupported}")

    if seed_registry_contract not in RECURRENT_TRAINING_SEED_REGISTRY_CONTRACTS:
        raise RecurrentExperimentError(
            "seed_registry_contract must be one of "
            + ", ".join(RECURRENT_TRAINING_SEED_REGISTRY_CONTRACTS)
        )
    scale_development = (
        seed_registry_contract == RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT
    )
    if scale_development:
        if scale_learner_seed is None:
            raise RecurrentExperimentError(
                "scale_learner_seed is required for the scale-development registry"
            )
        resolved_scale_learner_seed = _positive_seed(
            scale_learner_seed,
            field="scale_learner_seed",
        )
        if (
            resolved_scale_learner_seed
            not in (SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"])
        ):
            raise RecurrentExperimentError(
                "scale_learner_seed must be registered for the scale_learner role"
            )
    else:
        if scale_learner_seed is not None:
            raise RecurrentExperimentError(
                "scale_learner_seed is only valid with the scale-development registry"
            )
        resolved_scale_learner_seed = None
    seed_registry = (
        SCALE_DEVELOPMENT_SEED_REGISTRY
        if scale_development
        else RECURRENT_SEED_REGISTRY
    )
    broad_role = "scale_train" if scale_development else "train"
    fixture_role = "scale_curriculum" if scale_development else "curriculum"
    sampling_identity_version = (
        RECURRENT_SCALE_POLICY_SAMPLING_TASK_IDENTITY_VERSION
        if scale_development
        else RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION
    )

    broad_seeds = seed_registry[broad_role]
    fixture_seeds = seed_registry[fixture_role]
    broad_index = 0
    fixture_indices = {
        scenario: 0
        for scenario in selected_scenarios
        if scenario != RECURRENT_BROAD_SCENARIO
    }
    schedule: list[tuple[RecurrentRolloutTask, ...]] = []
    policy_sampling_seeds: set[int] = set()
    global_index = 0
    for update_index in range(update_count):
        tasks: list[RecurrentRolloutTask] = []
        for _ in range(worlds_per_update):
            scenario = selected_scenarios[global_index % len(selected_scenarios)]
            if scenario == RECURRENT_BROAD_SCENARIO:
                seed = broad_seeds[broad_index % len(broad_seeds)]
                broad_index += 1
            else:
                fixture_index = fixture_indices[scenario]
                seed = fixture_seeds[fixture_index % len(fixture_seeds)]
                fixture_indices[scenario] = fixture_index + 1
            if scale_development:
                policy_sampling_identity = (
                    f"{sampling_identity_version}|"
                    f"learner={resolved_scale_learner_seed}|"
                    f"update={update_index:04d}|world={global_index:06d}|"
                    f"scenario={scenario}"
                )
                task_id = (
                    f"scale-learner-{resolved_scale_learner_seed}-"
                    f"update-{update_index:04d}-world-{global_index:06d}-"
                    f"{scenario}-seed-{seed}"
                )
            else:
                policy_sampling_identity = (
                    f"{sampling_identity_version}|"
                    f"update={update_index:04d}|world={global_index:06d}|"
                    f"scenario={scenario}"
                )
                task_id = (
                    f"update-{update_index:04d}-world-{global_index:06d}-"
                    f"{scenario}-seed-{seed}"
                )
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=policy_sampling_identity
            )
            if policy_sampling_seed in policy_sampling_seeds:
                raise RecurrentExperimentError(
                    "policy sampling seed collision in recurrent training schedule"
                )
            policy_sampling_seeds.add(policy_sampling_seed)
            tasks.append(
                RecurrentRolloutTask(
                    task_id=task_id,
                    scenario=scenario,
                    environment_seed=seed,
                    rollout_ticks=rollout_ticks,
                    policy_sampling_identity=policy_sampling_identity,
                    policy_sampling_seed=policy_sampling_seed,
                    seed_role=(
                        broad_role
                        if scenario == RECURRENT_BROAD_SCENARIO
                        else fixture_role
                    ),
                )
            )
            global_index += 1
        schedule.append(tuple(tasks))
    return tuple(schedule)


def build_recurrent_counterfactual_collection_tasks(
    training_tasks: Sequence[RecurrentRolloutTask],
    *,
    update_index: int,
    bundles_per_update: int,
    branch_tick_candidates: tuple[int, ...],
) -> tuple[RecurrentCounterfactualCollectionTask, ...]:
    """Select learner-induced branch worlds without using future outcomes.

    Carrion receives one fixed-priority slot when present because it is the
    preregistered target fixture. Remaining slots rotate through the ordered
    non-carrion task list by update index. Selection depends only on the
    schedule, never on rollout rewards, deaths, survival, or future labels.
    """

    if not isinstance(training_tasks, Sequence) or isinstance(
        training_tasks,
        (str, bytes),
    ):
        raise RecurrentExperimentError("training_tasks must be an ordered sequence")
    tasks = tuple(training_tasks)
    if not tasks or any(not isinstance(task, RecurrentRolloutTask) for task in tasks):
        raise RecurrentExperimentError(
            "training_tasks must contain RecurrentRolloutTask values"
        )
    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or update_index < 0
    ):
        raise RecurrentExperimentError(
            "counterfactual update_index must be a non-negative integer"
        )
    resolved_count = _positive_int(
        bundles_per_update,
        field="bundles_per_update",
    )
    if resolved_count > len(tasks):
        raise RecurrentExperimentError(
            "bundles_per_update cannot exceed the number of training tasks"
        )
    if not isinstance(branch_tick_candidates, tuple) or not branch_tick_candidates:
        raise RecurrentExperimentError(
            "branch_tick_candidates must be a non-empty tuple"
        )

    carrion = tuple(task for task in tasks if task.scenario == "carrion_only")
    ordered: list[RecurrentRolloutTask] = []
    if carrion:
        ordered.append(carrion[update_index % len(carrion)])
    remaining = tuple(task for task in tasks if task not in ordered)
    if remaining:
        offset = update_index % len(remaining)
        ordered.extend((*remaining[offset:], *remaining[:offset]))
    selected = tuple(ordered[:resolved_count])
    if len(selected) != resolved_count or len(set(selected)) != resolved_count:
        raise RecurrentExperimentError(
            "counterfactual task selection did not produce unique source worlds"
        )

    collection_tasks: list[RecurrentCounterfactualCollectionTask] = []
    for slot, task in enumerate(selected):
        scale_task = str(task.seed_role).startswith("scale_")
        identity_prefix = (
            f"{RECURRENT_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION}|"
            f"update={update_index:04d}|slot={slot:02d}|"
            f"source_task={task.task_id}"
        )
        collection_task_id = (
            (f"counterfactual-{task.task_id}-update-{update_index:04d}-slot-{slot:02d}")
            if scale_task
            else (
                f"counterfactual-update-{update_index:04d}-slot-{slot:02d}-"
                f"{task.scenario}-seed-{task.environment_seed}"
            )
        )
        collection_tasks.append(
            RecurrentCounterfactualCollectionTask(
                task_id=collection_task_id,
                seed_role=str(task.seed_role),
                scenario=task.scenario,
                environment_seed=task.environment_seed,
                branch_tick_candidates=branch_tick_candidates,
                branch_tick_stratum_index=(
                    update_index % len(branch_tick_candidates) if scale_task else None
                ),
                source_policy_sampling_identity=f"{identity_prefix}|source-policy",
                branch_selection_identity=f"{identity_prefix}|branch-selection",
            )
        )
    return tuple(collection_tasks)


def collect_recurrent_rollout_batch(
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentRolloutTask],
    *,
    feed_forward_history_ablation: bool = False,
    rollout_workers: int = 1,
) -> tuple[RecurrentRolloutBuffer, RecurrentRolloutBatchDiagnostics]:
    """Collect one frozen-policy batch from real canonical simulator worlds."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise RecurrentExperimentError("model must be a PublicRecurrentActorCritic")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        raise RecurrentExperimentError("tasks must be a non-empty ordered sequence")
    task_tuple = tuple(tasks)
    if any(not isinstance(task, RecurrentRolloutTask) for task in task_tuple):
        raise RecurrentExperimentError("every task must be a RecurrentRolloutTask")
    if type(feed_forward_history_ablation) is not bool:
        raise RecurrentExperimentError(
            "feed_forward_history_ablation must be an exact boolean"
        )
    rollout_workers = _rollout_worker_count(rollout_workers)
    task_ids = [task.task_id for task in task_tuple]
    if len(task_ids) != len(set(task_ids)):
        raise RecurrentExperimentError("rollout task ids must be unique")

    behavior_model = frozen_cpu_model_copy(model)
    if rollout_workers == 1 or len(task_tuple) == 1:
        task_results = tuple(
            _collect_recurrent_rollout_task(
                behavior_model,
                task,
                feed_forward_history_ablation=feed_forward_history_ablation,
            )
            for task in task_tuple
        )
    else:
        task_results = _collect_recurrent_rollout_tasks_parallel(
            behavior_model,
            task_tuple,
            feed_forward_history_ablation=feed_forward_history_ablation,
            rollout_workers=rollout_workers,
        )

    buffer = RecurrentRolloutBuffer()
    summaries: list[dict[str, object]] = []
    for task, task_result in zip(task_tuple, task_results, strict=True):
        steps, summary = task_result
        if summary.get("task_id") != task.task_id:
            raise RecurrentExperimentError(
                "rollout worker result task order or identity drifted"
            )
        buffer.register_world(
            task.task_id,
            environment_seed=task.environment_seed,
            policy_sampling_seed=task.policy_sampling_seed,
        )
        for step in steps:
            buffer.append(step)
        buffer.validate_world_closed(task.task_id)
        summaries.append(summary)
    diagnostics = _rollout_diagnostics(buffer, world_summaries=summaries)
    return buffer, diagnostics


def _collect_recurrent_rollout_task(
    behavior_model: PublicRecurrentActorCritic,
    task: RecurrentRolloutTask,
    *,
    feed_forward_history_ablation: bool,
) -> tuple[tuple[RecurrentRolloutStep, ...], dict[str, object]]:
    """Collect one isolated world so sequential and worker paths share code."""

    core = TorchRecurrentPolicyCore(behavior_model)
    task_buffer = RecurrentRolloutBuffer()
    collector = RecurrentOnPolicyCollector(
        core,
        buffer=task_buffer,
        reset_recurrent_state_each_decision=feed_forward_history_ablation,
    )
    collector.start_world(
        world_id=task.task_id,
        rollout_ticks=task.rollout_ticks,
        environment_seed=task.environment_seed,
        policy_sampling_seed=task.policy_sampling_seed,
    )
    world = _world_for_task(task, policy=collector)
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    collector.finish_world()
    return task_buffer.steps, {
        "task_id": task.task_id,
        "scenario": task.scenario,
        "environment_seed": task.environment_seed,
        "seed_role": task.seed_role,
        "policy_sampling_identity": task.policy_sampling_identity,
        "policy_sampling_seed": task.policy_sampling_seed,
        "policy_sampling_seed_namespace": RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
        "rollout_ticks": task.rollout_ticks,
        "summary": _compact_world_summary(result.summary),
    }


def _collect_recurrent_rollout_tasks_parallel(
    behavior_model: PublicRecurrentActorCritic,
    tasks: tuple[RecurrentRolloutTask, ...],
    *,
    feed_forward_history_ablation: bool,
    rollout_workers: int,
) -> tuple[tuple[tuple[RecurrentRolloutStep, ...], dict[str, object]], ...]:
    """Run independent real worlds in spawned workers and merge by task order.

    The worker receives one frozen float32 state snapshot in its initializer.
    Results are returned by ``executor.map``, whose iteration order is input
    order rather than completion order. Any worker or serialization failure is
    propagated; this path never silently falls back to a different collector.
    """

    worker_count = min(_rollout_worker_count(rollout_workers), len(tasks))
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in behavior_model.state_dict().items()
    }
    context = mp.get_context(RECURRENT_ROLLOUT_WORKER_START_METHOD)
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_recurrent_rollout_worker,
            initargs=(
                behavior_model.config,
                model_state,
                feed_forward_history_ablation,
            ),
        ) as executor:
            results = tuple(executor.map(_collect_recurrent_rollout_worker, tasks))
    except (OSError, RuntimeError) as error:
        raise RecurrentExperimentError(
            "process-parallel recurrent rollout collection failed closed"
        ) from error
    if len(results) != len(tasks):
        raise RecurrentExperimentError(
            "process-parallel recurrent rollout result count drifted"
        )
    return results


def _initialize_recurrent_rollout_worker(
    model_config: RecurrentActorCriticConfig,
    model_state: dict[str, torch.Tensor],
    feed_forward_history_ablation: bool,
) -> None:
    """Initialize one inference-only CPU model and bounded Torch thread pool."""

    global _WORKER_BEHAVIOR_MODEL
    global _WORKER_FEED_FORWARD_HISTORY_ABLATION
    torch.set_num_threads(RECURRENT_ROLLOUT_WORKER_TORCH_THREADS)
    torch.set_num_interop_threads(RECURRENT_ROLLOUT_WORKER_TORCH_THREADS)
    model = PublicRecurrentActorCritic(
        model_config,
        initialization_seed=1,
    ).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _WORKER_BEHAVIOR_MODEL = model
    _WORKER_FEED_FORWARD_HISTORY_ABLATION = feed_forward_history_ablation


def _collect_recurrent_rollout_worker(
    task: RecurrentRolloutTask,
) -> tuple[tuple[RecurrentRolloutStep, ...], dict[str, object]]:
    model = _WORKER_BEHAVIOR_MODEL
    ablation = _WORKER_FEED_FORWARD_HISTORY_ABLATION
    if model is None or type(ablation) is not bool:
        raise RecurrentExperimentError("recurrent rollout worker was not initialized")
    return _collect_recurrent_rollout_task(
        model,
        task,
        feed_forward_history_ablation=ablation,
    )


class RecurrentExperimentRunner:
    """Own a learned model and optimizer while worlds receive frozen snapshots."""

    def __init__(
        self,
        *,
        learner_seed: int,
        device: torch.device | str,
        model_config: RecurrentActorCriticConfig | None = None,
        ppo_config: RecurrentPPOConfig | None = None,
        rollout_workers: int = 1,
        counterfactual_config: RecurrentCounterfactualExperimentConfig | None = None,
    ) -> None:
        _positive_seed(learner_seed, field="learner_seed")
        self.learner_seed = learner_seed
        self.device = torch.device(device)
        self.rollout_workers = _rollout_worker_count(rollout_workers)
        configure_recurrent_training_determinism(
            learner_seed=learner_seed,
            device=self.device,
        )
        resolved_ppo = ppo_config or RecurrentPPOConfig(learner_seed=learner_seed)
        if resolved_ppo.learner_seed != learner_seed:
            raise RecurrentExperimentError(
                "PPO learner_seed must match the experiment learner_seed"
            )
        if counterfactual_config is not None and not isinstance(
            counterfactual_config,
            RecurrentCounterfactualExperimentConfig,
        ):
            raise RecurrentExperimentError(
                "counterfactual_config must be a "
                "RecurrentCounterfactualExperimentConfig or None"
            )
        if (
            counterfactual_config is not None
            and resolved_ppo.feed_forward_history_ablation
        ):
            raise RecurrentExperimentError(
                "counterfactual recurrent-history auxiliary is incompatible with "
                "the feed-forward history ablation"
            )
        self.model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=learner_seed,
        ).to(device=self.device, dtype=torch.float32)
        self.ppo_config = resolved_ppo
        self.trainer = RecurrentPPOTrainer(self.model, self.ppo_config)
        self.counterfactual_config = counterfactual_config
        self._updates: list[RecurrentTrainingUpdateResult] = []
        self._completed_update_count = 0
        self._scheduled_run_state = "not_started"

    @property
    def completed_update_count(self) -> int:
        return self._completed_update_count

    def export_training_checkpoint_state(self) -> dict[str, object]:
        """Return JSON-safe-encodable optimizer, RNG, and trainer state.

        The durable checkpoint serializer owns the wire format.  This method
        exposes only resumable training state and never an inference artifact.
        """

        if self._scheduled_run_state == "failed":
            raise RecurrentExperimentError(
                "failed experiment runner cannot create a resumable checkpoint"
            )
        rng_state: dict[str, object] = {
            "schema_version": "mind_public_recurrent_runner_rng_state_v1",
            "python_random_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "trainer_update_index": self.trainer.update_index,
            "counterfactual_auxiliary_update_count": (
                self.trainer.counterfactual_auxiliary_update_count
            ),
            "attempted_counterfactual_auxiliary_bundles": sorted(
                self.trainer._attempted_counterfactual_auxiliary_bundles
            ),
        }
        if (
            torch.backends.mps.is_available()
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state")
        ):
            rng_state["torch_mps_rng_state"] = torch.mps.get_rng_state()
        return {
            "optimizer_state": copy.deepcopy(self.trainer.optimizer.state_dict()),
            "rng_state": rng_state,
        }

    def restore_training_checkpoint_state(
        self,
        *,
        model_state: Mapping[str, torch.Tensor],
        optimizer_state: Mapping[object, object],
        rng_state: Mapping[object, object],
        completed_updates: int,
    ) -> None:
        """Restore an externally verified crash checkpoint exactly once."""

        if (
            self._scheduled_run_state != "not_started"
            or self._updates
            or self._completed_update_count != 0
            or self.trainer.update_index != 0
        ):
            raise RecurrentExperimentError(
                "training checkpoint restore requires a pristine runner"
            )
        if (
            isinstance(completed_updates, bool)
            or not isinstance(completed_updates, int)
            or completed_updates < 0
        ):
            raise RecurrentExperimentError(
                "completed_updates must be a non-negative integer"
            )
        if rng_state.get("schema_version") != (
            "mind_public_recurrent_runner_rng_state_v1"
        ):
            raise RecurrentExperimentError("runner RNG checkpoint schema drifted")
        trainer_update_index = rng_state.get("trainer_update_index")
        if trainer_update_index != completed_updates:
            raise RecurrentExperimentError(
                "checkpoint trainer update index disagrees with completed updates"
            )
        auxiliary_update_count = rng_state.get("counterfactual_auxiliary_update_count")
        if (
            isinstance(auxiliary_update_count, bool)
            or not isinstance(auxiliary_update_count, int)
            or not 0 <= auxiliary_update_count <= completed_updates
        ):
            raise RecurrentExperimentError(
                "checkpoint auxiliary update count is invalid"
            )
        attempted_value = rng_state.get("attempted_counterfactual_auxiliary_bundles")
        if not isinstance(attempted_value, list) or any(
            not isinstance(value, str) or not value for value in attempted_value
        ):
            raise RecurrentExperimentError(
                "checkpoint attempted auxiliary bundle ledger is invalid"
            )
        if attempted_value != sorted(set(attempted_value)):
            raise RecurrentExperimentError(
                "checkpoint attempted auxiliary bundle ledger is not canonical"
            )
        python_random_state = rng_state.get("python_random_state")
        torch_cpu_rng_state = rng_state.get("torch_cpu_rng_state")
        if not isinstance(torch_cpu_rng_state, torch.Tensor):
            raise RecurrentExperimentError(
                "checkpoint torch CPU RNG state must be a tensor"
            )
        try:
            python_probe = random.Random()
            python_probe.setstate(python_random_state)  # type: ignore[arg-type]
            cpu_rng_state = torch_cpu_rng_state.to(device="cpu", dtype=torch.uint8)
            torch.Generator(device="cpu").set_state(cpu_rng_state)
        except (TypeError, ValueError, RuntimeError) as error:
            raise RecurrentExperimentError(
                "checkpoint CPU RNG state is invalid"
            ) from error
        cuda_rng_state = rng_state.get("torch_cuda_rng_state_all")
        if cuda_rng_state is not None:
            if (
                not torch.cuda.is_available()
                or not isinstance(cuda_rng_state, list)
                or len(cuda_rng_state) != torch.cuda.device_count()
                or any(not isinstance(value, torch.Tensor) for value in cuda_rng_state)
            ):
                raise RecurrentExperimentError(
                    "CUDA RNG checkpoint cannot be restored on this runtime"
                )
            try:
                for device_index, state in enumerate(cuda_rng_state):
                    torch.Generator(device=f"cuda:{device_index}").set_state(
                        state.to(device="cpu", dtype=torch.uint8)
                    )
            except RuntimeError as error:
                raise RecurrentExperimentError(
                    "checkpoint CUDA RNG state is invalid"
                ) from error
        mps_rng_state = rng_state.get("torch_mps_rng_state")
        if mps_rng_state is not None:
            if (
                not isinstance(mps_rng_state, torch.Tensor)
                or not hasattr(torch, "mps")
                or not hasattr(torch.mps, "set_rng_state")
            ):
                raise RecurrentExperimentError(
                    "MPS RNG checkpoint cannot be restored on this runtime"
                )

        model_snapshot = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        optimizer_snapshot = copy.deepcopy(self.trainer.optimizer.state_dict())
        python_rng_snapshot = random.getstate()
        cpu_rng_snapshot = torch.get_rng_state()
        cuda_rng_snapshot = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        mps_rng_snapshot = (
            torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state")
            else None
        )
        try:
            self.model.load_state_dict(dict(model_state), strict=True)
            self.model.to(device=self.device, dtype=torch.float32)
            self.trainer.optimizer.load_state_dict(copy.deepcopy(dict(optimizer_state)))
            self.trainer.optimizer.zero_grad(set_to_none=True)
            random.setstate(python_random_state)  # type: ignore[arg-type]
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if mps_rng_state is not None:
                torch.mps.set_rng_state(mps_rng_state)
            self.trainer._update_index = completed_updates
            self.trainer._counterfactual_auxiliary_update_count = auxiliary_update_count
            self.trainer._attempted_counterfactual_auxiliary_bundles = set(
                attempted_value
            )
            self._completed_update_count = completed_updates
        except Exception as error:
            self.model.load_state_dict(model_snapshot, strict=True)
            self.trainer.optimizer.load_state_dict(optimizer_snapshot)
            self.trainer.optimizer.zero_grad(set_to_none=True)
            random.setstate(python_rng_snapshot)
            torch.set_rng_state(cpu_rng_snapshot)
            if cuda_rng_snapshot is not None:
                torch.cuda.set_rng_state_all(cuda_rng_snapshot)
            if mps_rng_snapshot is not None:
                torch.mps.set_rng_state(mps_rng_snapshot)
            self.trainer._update_index = 0
            self.trainer._counterfactual_auxiliary_update_count = 0
            self.trainer._attempted_counterfactual_auxiliary_bundles = set()
            self._completed_update_count = 0
            raise RecurrentExperimentError(
                "checkpoint restore failed transactionally"
            ) from error

    def train_update(
        self,
        tasks: Sequence[RecurrentRolloutTask],
    ) -> RecurrentTrainingUpdateResult:
        if self._scheduled_run_state in {"completed", "failed"}:
            raise RecurrentExperimentError(
                "scheduled experiment runner is terminal and cannot train more updates"
            )
        task_tuple = tuple(tasks)
        buffer, rollout_diagnostics = collect_recurrent_rollout_batch(
            self.model,
            task_tuple,
            feed_forward_history_ablation=(
                self.ppo_config.feed_forward_history_ablation
            ),
            rollout_workers=self.rollout_workers,
        )
        sequences = ppo_sequences_from_rollout_buffer(
            buffer,
            model=self.model,
            config=self.ppo_config,
        )
        transaction_snapshot = self.trainer._snapshot_experiment_transaction()
        try:
            optimizer_diagnostics = self.trainer.update(sequences)
            counterfactual_collection: (
                RecurrentCounterfactualCollectionResult | None
            ) = None
            counterfactual_auxiliary: (
                RecurrentCounterfactualAuxiliaryStepDiagnostics | None
            ) = None
            if self.counterfactual_config is not None:
                post_ppo_model_sha256 = recurrent_model_state_sha256(self.model)
                ephemeral_artifact_sha256 = stable_payload_digest(
                    {
                        "policy": "in_memory_post_ppo_counterfactual_source_v1",
                        "learner_seed": self.learner_seed,
                        "ppo_update_index": self.trainer.update_index,
                        "model_state_sha256": post_ppo_model_sha256,
                        "durable_artifact_created": False,
                    }
                )
                branch_tasks = build_recurrent_counterfactual_collection_tasks(
                    task_tuple,
                    update_index=self._completed_update_count,
                    bundles_per_update=(self.counterfactual_config.bundles_per_update),
                    branch_tick_candidates=(
                        self.counterfactual_config.branch_tick_candidates
                    ),
                )
                counterfactual_collection = collect_recurrent_counterfactual_bundles(
                    self.model,
                    branch_tasks,
                    artifact_digest=ephemeral_artifact_sha256,
                    config=self.counterfactual_config.collection,
                    workers=self.counterfactual_config.workers,
                )
                if (
                    counterfactual_collection.source_model_state_sha256
                    != post_ppo_model_sha256
                    or counterfactual_collection.source_artifact_digest
                    != ephemeral_artifact_sha256
                ):
                    raise RecurrentExperimentError(
                        "counterfactual collection source identity drifted"
                    )
                if self.counterfactual_config.collection.multi_tape_enabled:
                    aggregate_groups: list[RecurrentCounterfactualAggregateGroup] = []
                    for bundle in counterfactual_collection.bundles:
                        if bundle.aggregate_rows is None:
                            raise RecurrentExperimentError(
                                "multi-tape collection omitted aggregate rows"
                            )
                        aggregate_groups.append(
                            RecurrentCounterfactualAggregateGroup(
                                aggregate_rows=tuple(bundle.aggregate_rows),
                                terminal_target=bundle.terminal_target,
                            )
                        )
                    if len(aggregate_groups) != len(branch_tasks):
                        raise RecurrentExperimentError(
                            "counterfactual collector did not return one aggregate "
                            "bundle per task"
                        )
                    counterfactual_auxiliary = (
                        self.trainer.counterfactual_aggregate_auxiliary_update(
                            tuple(aggregate_groups),
                            artifact_digest=ephemeral_artifact_sha256,
                            config=self.counterfactual_config.auxiliary,
                            step_config=self.counterfactual_config.step,
                        )
                    )
                else:
                    row_groups = tuple(
                        bundle.rows for bundle in counterfactual_collection.bundles
                    )
                    if len(row_groups) != len(branch_tasks):
                        raise RecurrentExperimentError(
                            "counterfactual collector did not return one bundle per task"
                        )
                    counterfactual_auxiliary = (
                        self.trainer.counterfactual_auxiliary_update(
                            row_groups,
                            artifact_digest=ephemeral_artifact_sha256,
                            config=self.counterfactual_config.auxiliary,
                            step_config=self.counterfactual_config.step,
                        )
                    )
                if (
                    counterfactual_auxiliary.pre_model_state_sha256
                    != post_ppo_model_sha256
                ):
                    raise RecurrentExperimentError(
                        "counterfactual auxiliary did not consume the exact collected model"
                    )
            result = RecurrentTrainingUpdateResult(
                update_index=self._completed_update_count,
                tasks=task_tuple,
                rollout=rollout_diagnostics,
                optimizer=optimizer_diagnostics,
                rollout_workers_requested=self.rollout_workers,
                rollout_workers_resolved=min(self.rollout_workers, len(task_tuple)),
                counterfactual_collection=counterfactual_collection,
                counterfactual_auxiliary=counterfactual_auxiliary,
            )
            self._updates.append(result)
            self._completed_update_count += 1
        except Exception:
            self.trainer._restore_experiment_transaction(
                transaction_snapshot,
                preserve_new_evidence_attempts=True,
            )
            raise
        return result

    def run(
        self,
        schedule: Sequence[Sequence[RecurrentRolloutTask]],
    ) -> RecurrentTrainingRunResult:
        if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
            raise RecurrentExperimentError("schedule must be an ordered sequence")
        if self._scheduled_run_state != "not_started":
            raise RecurrentExperimentError(
                "scheduled experiment run is single-use and already started"
            )
        if self._updates or self._completed_update_count != 0:
            raise RecurrentExperimentError(
                "scheduled experiment run requires a pristine update journal"
            )
        schedule_tuple: list[tuple[RecurrentRolloutTask, ...]] = []
        for tasks in schedule:
            if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
                raise RecurrentExperimentError(
                    "every scheduled update must be an ordered sequence"
                )
            task_tuple = tuple(tasks)
            if not task_tuple:
                raise RecurrentExperimentError("scheduled updates cannot be empty")
            if any(not isinstance(task, RecurrentRolloutTask) for task in task_tuple):
                raise RecurrentExperimentError(
                    "every scheduled task must be a RecurrentRolloutTask"
                )
            schedule_tuple.append(task_tuple)
        if not schedule_tuple:
            raise RecurrentExperimentError("schedule cannot be empty")
        scheduled_tasks = tuple(task for tasks in schedule_tuple for task in tasks)
        _require_unique_schedule_values(
            tuple(task.task_id for task in scheduled_tasks),
            field="task_id",
        )
        _require_unique_schedule_values(
            tuple(str(task.policy_sampling_identity) for task in scheduled_tasks),
            field="policy_sampling_identity",
        )
        _require_unique_schedule_values(
            tuple(int(task.policy_sampling_seed) for task in scheduled_tasks),
            field="policy_sampling_seed",
        )
        self._scheduled_run_state = "in_progress"
        try:
            for tasks in schedule_tuple:
                self.train_update(tasks)
        except Exception:
            self._scheduled_run_state = "failed"
            raise
        self._scheduled_run_state = "completed"
        scenario_order: list[str] = []
        for update in self._updates:
            for task in update.tasks:
                if task.scenario not in scenario_order:
                    scenario_order.append(task.scenario)
        return RecurrentTrainingRunResult(
            contract_version=RECURRENT_EXPERIMENT_CONTRACT_VERSION,
            learner_seed=self.learner_seed,
            device=str(self.device),
            deterministic_algorithms_enabled=(
                torch.are_deterministic_algorithms_enabled()
            ),
            model_config=asdict(self.model.config),
            ppo_config=asdict(self.ppo_config),
            rollout_execution={
                "requested_workers": self.rollout_workers,
                "start_method": RECURRENT_ROLLOUT_WORKER_START_METHOD,
                "torch_threads_per_worker": RECURRENT_ROLLOUT_WORKER_TORCH_THREADS,
                "ordered_merge": True,
                "sequential_default": True,
            },
            training_scenarios=tuple(scenario_order),
            environment_seed_provenance=_training_seed_provenance(self._updates),
            updates=tuple(self._updates),
            total_worlds=sum(update.rollout.world_count for update in self._updates),
            total_transitions=sum(
                update.rollout.transition_count for update in self._updates
            ),
            counterfactual_experiment=(
                None
                if self.counterfactual_config is None
                else {
                    "enabled": True,
                    "collection": recurrent_counterfactual_collection_config_payload(
                        self.counterfactual_config.collection
                    ),
                    "auxiliary": self.counterfactual_config.auxiliary.as_contract(),
                    "step": self.counterfactual_config.step.as_contract(),
                    "bundles_per_update": (
                        self.counterfactual_config.bundles_per_update
                    ),
                    "branch_tick_candidates": list(
                        self.counterfactual_config.branch_tick_candidates
                    ),
                    "workers": self.counterfactual_config.workers,
                    "source_artifact_policy": (
                        "ephemeral_in_memory_post_ppo_model_identity_only"
                    ),
                    "durable_artifact_created": False,
                    "runtime_integrated": False,
                    "promotion_authorized": False,
                }
            ),
        )


def recurrent_training_run_payload(
    result: RecurrentTrainingRunResult,
) -> dict[str, object]:
    if not isinstance(result, RecurrentTrainingRunResult):
        raise RecurrentExperimentError("result must be a RecurrentTrainingRunResult")
    return asdict(result)


def _training_seed_provenance(
    updates: Sequence[RecurrentTrainingUpdateResult],
) -> dict[str, object]:
    role_contracts = {
        RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL: (
            ("train", "curriculum"),
            RECURRENT_SEED_REGISTRY,
            RECURRENT_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION,
        ),
        RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT: (
            ("scale_train", "scale_curriculum"),
            SCALE_DEVELOPMENT_SEED_REGISTRY,
            RECURRENT_SCALE_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION,
        ),
    }
    observed_roles = {
        str(task.seed_role) for update in updates for task in update.tasks
    }
    selected_contract: str | None = None
    selected_roles: tuple[str, str] | None = None
    selected_registry: Mapping[str, Sequence[int]] | None = None
    schema_version: str | None = None
    for contract, (roles, registry, candidate_schema) in role_contracts.items():
        if observed_roles and observed_roles.issubset(set(roles)):
            selected_contract = contract
            selected_roles = roles
            selected_registry = registry
            schema_version = candidate_schema
            break
    if (
        selected_contract is None
        or selected_roles is None
        or selected_registry is None
        or schema_version is None
    ):
        raise RecurrentExperimentError(
            "committed updates mix seed registries or contain unsupported roles"
        )

    observed: dict[str, set[int]] = {role: set() for role in selected_roles}
    for update in updates:
        for task in update.tasks:
            role = task.seed_role
            if role not in observed:
                raise RecurrentExperimentError(
                    "committed update contains a non-training environment seed role"
                )
            if task.environment_seed not in selected_registry[role]:
                raise RecurrentExperimentError(
                    "committed update contains an environment seed outside its role"
                )
            observed[role].add(task.environment_seed)
    seeds_by_role = {
        role: [seed for seed in selected_registry[role] if seed in observed[role]]
        for role in selected_roles
    }
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "environment_seed_roles": list(selected_roles),
        "environment_seeds_by_role": seeds_by_role,
        "observed_environment_seed_count": sum(
            len(seeds) for seeds in seeds_by_role.values()
        ),
        "canonical_registry_membership_valid": True,
        "non_training_environment_seed_roles_accessed": [],
        "selection_seeds_accessed": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
    }
    if selected_contract == RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT:
        payload["seed_registry_contract"] = selected_contract
    return payload


def _require_unique_schedule_values(
    values: Sequence[object],
    *,
    field: str,
) -> None:
    if len(values) != len(set(values)):
        raise RecurrentExperimentError(
            f"scheduled {field} values must be globally unique"
        )


def configure_recurrent_training_determinism(
    *,
    learner_seed: int,
    device: torch.device | str,
) -> None:
    """Fail closed to deterministic Torch kernels for reproducible experiments."""

    _positive_seed(learner_seed, field="learner_seed")
    resolved = torch.device(device)
    if resolved.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(learner_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(learner_seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _world_for_task(
    task: RecurrentRolloutTask,
    *,
    policy: object,
) -> SimulationWorld:
    world_ticks = task.rollout_ticks + 1
    if task.scenario == RECURRENT_BROAD_SCENARIO:
        return SimulationWorld(
            WorldConfig(seed=task.environment_seed, max_ticks=world_ticks),
            policy=policy,
        )
    return _fixture_world(
        fixture_name=task.scenario,
        seed=task.environment_seed,
        ticks=world_ticks,
        policy=policy,
    )


def _rollout_diagnostics(
    buffer: RecurrentRolloutBuffer,
    *,
    world_summaries: Sequence[dict[str, object]],
) -> RecurrentRolloutBatchDiagnostics:
    steps = buffer.steps
    if not steps:
        raise RecurrentExperimentError("rollout batch contains no policy decisions")
    sequences = buffer.sequences()
    world_scenario: dict[str, str] = {}
    world_seed_provenance: dict[str, dict[str, object]] = {}
    for summary in world_summaries:
        world_id = summary.get("task_id")
        scenario = summary.get("scenario")
        if not isinstance(world_id, str) or not world_id:
            raise RecurrentExperimentError(
                "every rollout world summary must have a non-empty task_id"
            )
        if world_id in world_scenario:
            raise RecurrentExperimentError(
                f"duplicate rollout world summary task_id: {world_id!r}"
            )
        if (
            not isinstance(scenario, str)
            or scenario not in RECURRENT_TRAINING_SCENARIOS
        ):
            raise RecurrentExperimentError(
                f"rollout world summary has an invalid scenario: {scenario!r}"
            )
        environment_seed = _positive_seed(
            summary.get("environment_seed"),
            field="world summary environment_seed",
        )
        seed_role = summary.get("seed_role")
        legacy_seed_role = (
            "train" if scenario == RECURRENT_BROAD_SCENARIO else "curriculum"
        )
        scale_seed_role = (
            "scale_train"
            if scenario == RECURRENT_BROAD_SCENARIO
            else "scale_curriculum"
        )
        if seed_role not in {legacy_seed_role, scale_seed_role}:
            raise RecurrentExperimentError(
                "world summary seed role does not match its training scenario"
            )
        seed_registry = (
            RECURRENT_SEED_REGISTRY
            if seed_role == legacy_seed_role
            else SCALE_DEVELOPMENT_SEED_REGISTRY
        )
        if environment_seed not in seed_registry[seed_role]:
            raise RecurrentExperimentError(
                "world summary environment seed is outside its canonical training role"
            )
        policy_sampling_seed = _policy_sampling_seed(
            summary.get("policy_sampling_seed"),
            field="world summary policy_sampling_seed",
        )
        policy_sampling_identity = summary.get("policy_sampling_identity")
        if (
            not isinstance(policy_sampling_identity, str)
            or not policy_sampling_identity
            or policy_sampling_identity != policy_sampling_identity.strip()
        ):
            raise RecurrentExperimentError(
                "world summary policy_sampling_identity must be non-empty and trimmed"
            )
        if (
            summary.get("policy_sampling_seed_namespace")
            != RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
        ):
            raise RecurrentExperimentError(
                "world summary policy sampling namespace drifted"
            )
        expected_policy_sampling_seed = derive_recurrent_policy_sampling_seed(
            task_identity=policy_sampling_identity
        )
        if policy_sampling_seed != expected_policy_sampling_seed:
            raise RecurrentExperimentError(
                "world summary policy seed does not match its task identity"
            )
        world_scenario[world_id] = scenario
        world_seed_provenance[world_id] = {
            "environment_seed": environment_seed,
            "seed_role": seed_role,
            "policy_sampling_identity": policy_sampling_identity,
            "policy_sampling_seed": policy_sampling_seed,
        }
    unknown_world_ids = sorted({step.world_id for step in steps} - set(world_scenario))
    if unknown_world_ids:
        raise RecurrentExperimentError(
            f"rollout steps have no matching world summary: {unknown_world_ids[:8]}"
        )
    for step in steps:
        provenance = world_seed_provenance[step.world_id]
        if step.environment_seed != provenance["environment_seed"]:
            raise RecurrentExperimentError(
                "rollout step environment seed does not match its world summary"
            )
        if step.policy_sampling_seed != provenance["policy_sampling_seed"]:
            raise RecurrentExperimentError(
                "rollout step policy seed does not match its world summary"
            )
    buffer_provenance = buffer.world_seed_provenance
    if set(buffer_provenance) != set(world_seed_provenance):
        raise RecurrentExperimentError(
            "rollout buffer and world summaries cover different seed provenance"
        )
    for world_id, provenance in world_seed_provenance.items():
        if buffer_provenance[world_id] != {
            "environment_seed": provenance["environment_seed"],
            "policy_sampling_seed": provenance["policy_sampling_seed"],
        }:
            raise RecurrentExperimentError(
                "rollout buffer seed provenance does not match world summary"
            )

    batch_scope = _scope_diagnostics(steps, world_count=len(world_summaries))
    world_diagnostics = {
        world_id: _scope_diagnostics(
            tuple(step for step in steps if step.world_id == world_id),
            world_count=1,
        )
        for world_id in world_scenario
    }
    scenario_diagnostics = {
        scenario: _scope_diagnostics(
            tuple(step for step in steps if world_scenario[step.world_id] == scenario),
            world_count=sum(
                1 for candidate in world_scenario.values() if candidate == scenario
            ),
        )
        for scenario in RECURRENT_TRAINING_SCENARIOS
        if scenario in world_scenario.values()
    }
    _validate_scope_partition(
        batch_scope,
        tuple(world_diagnostics.values()),
        field="world diagnostics",
    )
    _validate_scope_partition(
        batch_scope,
        tuple(scenario_diagnostics.values()),
        field="scenario diagnostics",
    )

    action_counts = Counter(step.requested_action for step in steps)
    dominant_action, dominant_count = min(
        action_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    terminated = sum(1 for sequence in sequences if sequence[-1].terminated)
    truncated = sum(1 for sequence in sequences if sequence[-1].truncated)
    if terminated + truncated != len(sequences):
        raise RecurrentExperimentError("rollout contains an open agent sequence")
    return RecurrentRolloutBatchDiagnostics(
        world_count=len(world_summaries),
        transition_count=len(steps),
        sequence_count=len(sequences),
        terminated_sequence_count=terminated,
        truncated_sequence_count=truncated,
        total_reward=batch_scope.total_reward,
        mean_reward=batch_scope.mean_reward,
        reward_component_totals=batch_scope.reward_component_totals,
        action_mask_availability_counts=(batch_scope.action_mask_availability_counts),
        chosen_action_counts=batch_scope.chosen_action_counts,
        chosen_given_valid_rates=batch_scope.chosen_given_valid_rates,
        requested_action_counts=dict(sorted(action_counts.items())),
        dominant_requested_action=dominant_action,
        dominant_requested_action_share=dominant_count / len(steps),
        world_diagnostics=world_diagnostics,
        scenario_diagnostics=scenario_diagnostics,
        policy_sampling_seed_namespace=RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
        world_seed_provenance=world_seed_provenance,
        world_summaries=tuple(world_summaries),
    )


def _scope_diagnostics(
    steps: Sequence[RecurrentRolloutStep],
    *,
    world_count: int,
) -> RecurrentRolloutScopeDiagnostics:
    if (
        isinstance(world_count, bool)
        or not isinstance(world_count, int)
        or world_count <= 0
    ):
        raise RecurrentExperimentError("scope world_count must be a positive integer")
    availability = {action: 0 for action in RECURRENT_ROLLOUT_ACTIONS}
    chosen = {action: 0 for action in RECURRENT_ROLLOUT_ACTIONS}
    component_values = {component: [] for component in RECURRENT_REWARD_COMPONENTS}
    reward_values: list[float] = []
    passive_terminal_count = 0
    for step in steps:
        if len(step.action_mask) != len(RECURRENT_ROLLOUT_ACTIONS):
            raise RecurrentExperimentError(
                "rollout action mask length does not match the stable action space"
            )
        if (
            step.action_index < 0
            or step.action_index >= len(RECURRENT_ROLLOUT_ACTIONS)
            or RECURRENT_ROLLOUT_ACTIONS[step.action_index] != step.requested_action
        ):
            raise RecurrentExperimentError(
                "rollout action index and requested action disagree"
            )
        if not step.action_mask[step.action_index]:
            raise RecurrentExperimentError(
                "rollout chosen action is unavailable in its observation-time mask"
            )
        for index, action in enumerate(RECURRENT_ROLLOUT_ACTIONS):
            if type(step.action_mask[index]) is not bool:
                raise RecurrentExperimentError(
                    "rollout action mask must contain exact booleans"
                )
            availability[action] += int(step.action_mask[index])
        chosen[step.requested_action] += 1
        reward_values.append(step.reward)
        for component in RECURRENT_REWARD_COMPONENTS:
            component_values[component].append(step.reward_components[component])
        if step.passive_terminal_tick is not None:
            passive_terminal_count += 1
            reward_values.append(step.passive_terminal_reward)
            for component in RECURRENT_REWARD_COMPONENTS:
                component_values[component].append(
                    step.passive_terminal_reward_components[component]
                )

    for action in RECURRENT_ROLLOUT_ACTIONS:
        if chosen[action] > availability[action]:
            raise RecurrentExperimentError(
                f"chosen action count exceeds mask availability for {action!r}"
            )
    total_reward = math.fsum(reward_values)
    component_totals = {
        component: math.fsum(values) for component, values in component_values.items()
    }
    if not math.isfinite(total_reward) or any(
        not math.isfinite(value) for value in component_totals.values()
    ):
        raise RecurrentExperimentError("rollout reward diagnostics are non-finite")
    if not math.isclose(
        total_reward,
        math.fsum(component_totals.values()),
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise RecurrentExperimentError(
            "rollout reward total and reward-component totals disagree"
        )
    rates = {
        action: (
            chosen[action] / availability[action] if availability[action] else None
        )
        for action in RECURRENT_ROLLOUT_ACTIONS
    }
    return RecurrentRolloutScopeDiagnostics(
        world_count=world_count,
        transition_count=len(steps),
        passive_terminal_count=passive_terminal_count,
        total_reward=total_reward,
        mean_reward=total_reward / len(steps) if steps else 0.0,
        reward_component_totals=component_totals,
        action_mask_availability_counts=availability,
        chosen_action_counts=chosen,
        chosen_given_valid_rates=rates,
    )


def _validate_scope_partition(
    batch: RecurrentRolloutScopeDiagnostics,
    scopes: Sequence[RecurrentRolloutScopeDiagnostics],
    *,
    field: str,
) -> None:
    if not scopes:
        raise RecurrentExperimentError(f"{field} partition cannot be empty")
    additive_int_fields = (
        "world_count",
        "transition_count",
        "passive_terminal_count",
    )
    for name in additive_int_fields:
        if sum(getattr(scope, name) for scope in scopes) != getattr(batch, name):
            raise RecurrentExperimentError(
                f"{field} {name} does not partition the batch"
            )
    for action in RECURRENT_ROLLOUT_ACTIONS:
        if (
            sum(scope.action_mask_availability_counts[action] for scope in scopes)
            != batch.action_mask_availability_counts[action]
        ):
            raise RecurrentExperimentError(
                f"{field} mask availability does not partition action {action!r}"
            )
        if (
            sum(scope.chosen_action_counts[action] for scope in scopes)
            != (batch.chosen_action_counts[action])
        ):
            raise RecurrentExperimentError(
                f"{field} chosen counts do not partition action {action!r}"
            )
    if not math.isclose(
        math.fsum(scope.total_reward for scope in scopes),
        batch.total_reward,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise RecurrentExperimentError(
            f"{field} reward total does not partition the batch"
        )
    for component in RECURRENT_REWARD_COMPONENTS:
        if not math.isclose(
            math.fsum(scope.reward_component_totals[component] for scope in scopes),
            batch.reward_component_totals[component],
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise RecurrentExperimentError(
                f"{field} does not partition reward component {component!r}"
            )


def _compact_world_summary(summary: dict[str, object]) -> dict[str, object]:
    required = (
        "run_id",
        "seed",
        "ticks_executed",
        "births",
        "deaths",
        "alive_agents",
        "peak_alive_agents",
        "extinct",
    )
    missing = [field for field in required if field not in summary]
    if missing:
        raise RecurrentExperimentError(
            f"world summary is missing required fields: {missing}"
        )
    return {field: summary[field] for field in required}


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentExperimentError(f"{field} must be a positive integer")
    return value


def _positive_seed(value: object, *, field: str) -> int:
    parsed = _positive_int(value, field=field)
    if parsed > 2**31 - 1:
        raise RecurrentExperimentError(f"{field} must fit the simulator seed range")
    return parsed


def _policy_sampling_seed(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentExperimentError(f"{field} must be an integer")
    if value < 1 or value > MAX_RECURRENT_POLICY_SAMPLING_SEED:
        raise RecurrentExperimentError(
            f"{field} must be in [1, {MAX_RECURRENT_POLICY_SAMPLING_SEED}]"
        )
    return value


def _rollout_worker_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentExperimentError("rollout_workers must be an integer")
    if value < 1 or value > MAX_RECURRENT_ROLLOUT_WORKERS:
        raise RecurrentExperimentError(
            f"rollout_workers must be in [1, {MAX_RECURRENT_ROLLOUT_WORKERS}]"
        )
    return value


__all__ = [
    "RECURRENT_BROAD_SCENARIO",
    "RECURRENT_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION",
    "RECURRENT_EXPERIMENT_CONTRACT_VERSION",
    "MAX_RECURRENT_ROLLOUT_WORKERS",
    "RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE",
    "RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION",
    "RECURRENT_ROLLOUT_WORKER_START_METHOD",
    "RECURRENT_ROLLOUT_WORKER_TORCH_THREADS",
    "RECURRENT_TRAINING_SCENARIOS",
    "RecurrentExperimentError",
    "RecurrentExperimentRunner",
    "RecurrentCounterfactualExperimentConfig",
    "RecurrentRolloutBatchDiagnostics",
    "RecurrentRolloutScopeDiagnostics",
    "RecurrentRolloutTask",
    "RecurrentTrainingRunResult",
    "RecurrentTrainingUpdateResult",
    "build_recurrent_counterfactual_collection_tasks",
    "build_recurrent_training_schedule",
    "collect_recurrent_rollout_batch",
    "configure_recurrent_training_determinism",
    "derive_recurrent_policy_sampling_seed",
    "recurrent_training_run_payload",
]
