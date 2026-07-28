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

from evolution_sim.config.schema import SignalConfig, WorldConfig
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.evaluation_harness import (
    CONTROLLED_FIXTURE_NAMES,
    _fixture_world,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT,
    OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
    OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX,
    OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_PHASE_A_BENCHMARK_ENVIRONMENT_SEED_INDICES,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
)
from evolution_sim.mind.policy_inputs import (
    ecological_policy_input_schema_version,
    ecological_policy_input_vector_size,
)
from evolution_sim.mind.recurrent_actor_critic import (
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationMode,
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
    MAX_RECURRENT_GENOME_STREAM_SEED,
    MAX_RECURRENT_POLICY_SAMPLING_SEED,
    RECURRENT_ACTION_FREE_BOOTSTRAP_SCHEMA_VERSION,
    RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1,
    RECURRENT_GENOME_CONDITIONING_DISABLED,
    OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
    RECURRENT_FIXED_BATCH_RELEASE_STATUS,
    RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
    RECURRENT_REWARD_COMPONENTS,
    RECURRENT_ROLLOUT_ACTIONS,
    RecurrentOnPolicyCollector,
    RecurrentFixedBatchRuntimeContract,
    RecurrentRolloutBuffer,
    RecurrentRolloutStep,
    TorchRecurrentPolicyCore,
    derive_recurrent_policy_sampling_seed,
)
from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
)


RECURRENT_EXPERIMENT_CONTRACT_VERSION = "mind_public_recurrent_ippo_experiment_v5"
RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION = (
    "mind_public_recurrent_ippo_experiment_fixed_batch_v1"
)
RECURRENT_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_public_recurrent_training_seed_provenance_v1"
)
RECURRENT_SCALE_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_public_recurrent_scale_training_seed_provenance_v1"
)
RECURRENT_SCALE_V2_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_public_recurrent_scale_training_seed_provenance_v2"
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
RECURRENT_SCALE_V2_POLICY_SAMPLING_TASK_IDENTITY_VERSION = (
    "mind_public_recurrent_ippo_scale_training_schedule_task_v2"
)
RECURRENT_SCALE_V2_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION = (
    "mind_public_recurrent_counterfactual_collection_task_v2"
)
RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL = "canonical_v1"
RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT = "scale_development_v1"
RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT_V2 = "scale_development_v2"
RECURRENT_TRAINING_SEED_REGISTRY_CONTRACTS = (
    RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL,
    RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
    RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT_V2,
)
RECURRENT_BROAD_SCENARIO = "broad"
RECURRENT_TRAINING_SCENARIOS: tuple[str, ...] = (
    RECURRENT_BROAD_SCENARIO,
    *CONTROLLED_FIXTURE_NAMES,
)
MAX_RECURRENT_ROLLOUT_WORKERS = 64
RECURRENT_ROLLOUT_WORKER_START_METHOD = "spawn"
RECURRENT_ROLLOUT_WORKER_TORCH_THREADS = 1
OPEN_ECOLOGY_TREATMENT_SCHEMA_VERSION = "mind_v3_open_ecology_broad_treatment_v1"
OPEN_ECOLOGY_TASK_PROVENANCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_rollout_task_provenance_v1"
)
OPEN_ECOLOGY_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_training_seed_provenance_v1"
)
OPEN_ECOLOGY_BENCHMARK_SEED_PROVENANCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_benchmark_seed_provenance_v1"
)
OPEN_ECOLOGY_POLICY_SAMPLING_TASK_IDENTITY_VERSION = (
    "mind_v3_open_ecology_ippo_training_schedule_task_v1"
)
OPEN_ECOLOGY_TRAINING_SEED_ROLE = "open_ecology_train"
OPEN_ECOLOGY_BENCHMARK_TASK_IDENTITY_VERSION = (
    "mind_v3_open_ecology_operational_benchmark_task_v1"
)
OPEN_ECOLOGY_PROOF_SEED_ROLE = "open_ecology_proof"
OPEN_ECOLOGY_PROOF_TASK_IDENTITY_VERSION = (
    "mind_v3_open_ecology_engineering_proof_task_v1"
)
OPEN_ECOLOGY_INITIAL_AGENT_DENSITIES: tuple[int, ...] = (32, 64, 128, 256)
OPEN_ECOLOGY_PHASE_A = "phase_a"
OPEN_ECOLOGY_PHASE_B = "phase_b"
OPEN_ECOLOGY_TRAINING_PHASES: tuple[str, ...] = (
    OPEN_ECOLOGY_PHASE_A,
    OPEN_ECOLOGY_PHASE_B,
)
OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE: tuple[int, ...] = (64,)
OPEN_ECOLOGY_PHASE_B_DENSITY_CYCLE: tuple[int, ...] = (32, 64, 128, 64)
OPEN_ECOLOGY_MAX_AGENTS = 320
_OPEN_ECOLOGY_SIGNAL_VALUES: dict[str, object] = {
    "enabled": True,
    "reproductive_signal_channels": 1,
    "reproductive_signal_emission_enabled": True,
    "reproductive_signal_radius": 3,
    "reproductive_signal_duration_ticks": 6,
    "reproductive_signal_decay_rate": 0.58,
    "reproductive_signal_base_intensity": 0.16,
    "reproductive_signal_trait_intensity_bonus": 0.24,
    "communication_token_count": 4,
    "communication_profiles_per_token": 2,
    "communication_signal_emission_enabled": True,
    "communication_signal_radius": 3,
    "communication_signal_duration_ticks": 6,
    "communication_signal_decay_rate": 0.58,
    "communication_signal_base_intensity": 0.10,
    "communication_signal_trait_intensity_bonus": 0.20,
    "max_signal_radius": 8,
    "max_duration_ticks": 24,
    "max_intensity": 1.0,
    "base_emission_energy_cost": 0.005,
}
_ROLLOUT_SUMMARY_KEYS = frozenset(
    {
        "task_id",
        "scenario",
        "environment_seed",
        "seed_role",
        "policy_sampling_identity",
        "policy_sampling_seed",
        "policy_sampling_seed_namespace",
        "rollout_ticks",
        "terminal_bootstrap",
        "summary",
    }
)
_CONDITIONED_WORLD_SEED_PROVENANCE_KEYS = frozenset(
    {
        "environment_seed",
        "policy_sampling_seed",
        "genome_conditioning_mode",
        "genome_population_mode",
        "genome_stream_seed",
        "genome_population_binding_sha256",
        "genome_population_pre_founder_state_sha256",
        "genome_population_final_state_sha256",
        "genome_population_reset_state_sha256",
    }
)
_LEGACY_WORLD_SEED_PROVENANCE_KEYS = frozenset(
    {
        "environment_seed",
        "policy_sampling_seed",
    }
)
_OPEN_ECOLOGY_TASK_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "seed_registry_version",
        "seed_registry_sha256",
        "learner_seed",
        "environment_seed",
        "environment_seed_index",
        "genome_stream_seed",
        "genome_stream_seed_index",
        "genome_population_mode",
        "training_phase",
        "update_index",
        "world_index",
        "treatment",
    }
)


_WORKER_BEHAVIOR_MODEL: PublicRecurrentActorCritic | None = None
_WORKER_FEED_FORWARD_HISTORY_ABLATION: bool | None = None
_WORKER_FIXED_BATCH_CONTRACT: RecurrentFixedBatchRuntimeContract | None = None


class RecurrentExperimentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OpenEcologySignalTreatment:
    """Deeply immutable signal settings for the first open-ecology treatment."""

    enabled: bool = True
    reproductive_signal_channels: int = 1
    reproductive_signal_emission_enabled: bool = True
    reproductive_signal_radius: int = 3
    reproductive_signal_duration_ticks: int = 6
    reproductive_signal_decay_rate: float = 0.58
    reproductive_signal_base_intensity: float = 0.16
    reproductive_signal_trait_intensity_bonus: float = 0.24
    communication_token_count: int = 4
    communication_profiles_per_token: int = 2
    communication_signal_emission_enabled: bool = True
    communication_signal_radius: int = 3
    communication_signal_duration_ticks: int = 6
    communication_signal_decay_rate: float = 0.58
    communication_signal_base_intensity: float = 0.10
    communication_signal_trait_intensity_bonus: float = 0.20
    max_signal_radius: int = 8
    max_duration_ticks: int = 24
    max_intensity: float = 1.0
    base_emission_energy_cost: float = 0.005

    def __post_init__(self) -> None:
        if asdict(self) != _OPEN_ECOLOGY_SIGNAL_VALUES:
            raise RecurrentExperimentError(
                "open-ecology signal treatment must match the canonical contract"
            )
        # Reuse Foundation validation rather than duplicating its numeric limits.
        self.as_signal_config()

    def as_signal_config(self) -> SignalConfig:
        return SignalConfig(**asdict(self))


@dataclass(frozen=True, slots=True)
class OpenEcologyLaunchDependencies:
    """Truthful boundary for functionality outside this bounded implementation."""

    self_echo_exclusion_implemented: bool = True
    self_echo_exclusion_required_before_authoritative_launch: bool = True

    def __post_init__(self) -> None:
        if (
            self.self_echo_exclusion_implemented is not True
            or self.self_echo_exclusion_required_before_authoritative_launch is not True
        ):
            raise RecurrentExperimentError(
                "open-ecology launch dependencies must satisfy the preregistered "
                "self-echo requirement"
            )


@dataclass(frozen=True, slots=True)
class OpenEcologyBroadWorldTreatment:
    """Canonical broad-world treatment; only initial population may vary."""

    initial_agents: int
    schema_version: str = OPEN_ECOLOGY_TREATMENT_SCHEMA_VERSION
    max_agents: int = OPEN_ECOLOGY_MAX_AGENTS
    signals: OpenEcologySignalTreatment = OpenEcologySignalTreatment()
    reward_contract: str = "foundation_reward_contract_unchanged_v1"
    reward_shaping_added: bool = False
    action_heuristics_added: bool = False
    launch_dependencies: OpenEcologyLaunchDependencies = OpenEcologyLaunchDependencies()

    def __post_init__(self) -> None:
        if (
            isinstance(self.initial_agents, bool)
            or not isinstance(self.initial_agents, int)
            or self.initial_agents not in OPEN_ECOLOGY_INITIAL_AGENT_DENSITIES
        ):
            raise RecurrentExperimentError(
                "open-ecology initial_agents must be one of "
                f"{OPEN_ECOLOGY_INITIAL_AGENT_DENSITIES}"
            )
        if self.schema_version != OPEN_ECOLOGY_TREATMENT_SCHEMA_VERSION:
            raise RecurrentExperimentError("open-ecology treatment schema drifted")
        if (
            isinstance(self.max_agents, bool)
            or not isinstance(self.max_agents, int)
            or self.max_agents != OPEN_ECOLOGY_MAX_AGENTS
        ):
            raise RecurrentExperimentError("open-ecology max_agents must be 320")
        if not isinstance(self.signals, OpenEcologySignalTreatment):
            raise RecurrentExperimentError(
                "open-ecology treatment signals have the wrong type"
            )
        if (
            self.reward_contract != "foundation_reward_contract_unchanged_v1"
            or self.reward_shaping_added is not False
            or self.action_heuristics_added is not False
        ):
            raise RecurrentExperimentError(
                "open-ecology treatment cannot change rewards or add action heuristics"
            )
        if not isinstance(self.launch_dependencies, OpenEcologyLaunchDependencies):
            raise RecurrentExperimentError(
                "open-ecology launch dependencies have the wrong type"
            )

    def as_contract(self) -> dict[str, object]:
        return asdict(self)


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
    genome_stream_seed: int | None = None
    genome_population_mode: str = RecurrentGenomePopulationMode.DISABLED.value

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
        scale_v2_seed_role = (
            "scale_v2_train"
            if self.scenario == RECURRENT_BROAD_SCENARIO
            else "scale_v2_curriculum"
        )
        expected_seed_role = self.seed_role or legacy_seed_role
        open_ecology_task = isinstance(self, OpenEcologyRolloutTask)
        open_ecology_proof_task = isinstance(
            self,
            OpenEcologyProofRolloutTask,
        )
        open_ecology_benchmark_task = isinstance(
            self,
            OpenEcologyBenchmarkRolloutTask,
        )
        if expected_seed_role not in {
            legacy_seed_role,
            scale_seed_role,
            scale_v2_seed_role,
            *(
                (
                    OPEN_ECOLOGY_TRAINING_SEED_ROLE,
                    *(
                        (OPEN_ECOLOGY_PROOF_SEED_ROLE,)
                        if open_ecology_proof_task
                        else ()
                    ),
                    *(
                        (OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,)
                        if open_ecology_benchmark_task
                        else ()
                    ),
                )
                if self.scenario == RECURRENT_BROAD_SCENARIO and open_ecology_task
                else ()
            ),
        }:
            raise RecurrentExperimentError(
                "rollout task seed_role does not match its training scenario"
            )
        if expected_seed_role in {
            OPEN_ECOLOGY_TRAINING_SEED_ROLE,
            OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            OPEN_ECOLOGY_PROOF_SEED_ROLE,
        }:
            seed_registry = OPEN_ECOLOGY_SEED_REGISTRY
        elif expected_seed_role == legacy_seed_role:
            seed_registry = RECURRENT_SEED_REGISTRY
        elif expected_seed_role == scale_seed_role:
            seed_registry = SCALE_DEVELOPMENT_SEED_REGISTRY
        else:
            seed_registry = SCALE_DEVELOPMENT_V2_SEED_REGISTRY
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
        genome_population_mode, genome_stream_seed = _genome_population_configuration(
            population_mode=self.genome_population_mode,
            stream_seed=self.genome_stream_seed,
        )
        object.__setattr__(
            self,
            "genome_population_mode",
            genome_population_mode,
        )
        object.__setattr__(self, "genome_stream_seed", genome_stream_seed)


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenEcologyRolloutTask(RecurrentRolloutTask):
    """A broad training task with exact registry and treatment provenance."""

    open_ecology_treatment: OpenEcologyBroadWorldTreatment
    open_ecology_learner_seed: int
    open_ecology_environment_seed_index: int
    open_ecology_genome_stream_seed_index: int
    open_ecology_training_phase: str
    open_ecology_update_index: int
    open_ecology_world_index: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scenario != RECURRENT_BROAD_SCENARIO:
            raise RecurrentExperimentError(
                "open-ecology rollout tasks must use the broad scenario"
            )
        if self.seed_role != OPEN_ECOLOGY_TRAINING_SEED_ROLE:
            raise RecurrentExperimentError(
                "open-ecology rollout tasks must use open_ecology_train seeds"
            )
        if not isinstance(self.open_ecology_treatment, OpenEcologyBroadWorldTreatment):
            raise RecurrentExperimentError(
                "open-ecology rollout task treatment has the wrong type"
            )
        learner_seed = _positive_seed(
            self.open_ecology_learner_seed,
            field="open_ecology_learner_seed",
        )
        if learner_seed not in OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"]:
            raise RecurrentExperimentError(
                "open_ecology_learner_seed is not registered"
            )
        environment_index = _nonnegative_int(
            self.open_ecology_environment_seed_index,
            field="open_ecology_environment_seed_index",
        )
        train_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_TRAINING_SEED_ROLE]
        if (
            environment_index >= len(train_seeds)
            or train_seeds[environment_index] != self.environment_seed
        ):
            raise RecurrentExperimentError(
                "open-ecology environment seed index does not bind its seed"
            )
        genome_index = _nonnegative_int(
            self.open_ecology_genome_stream_seed_index,
            field="open_ecology_genome_stream_seed_index",
        )
        genome_seeds = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"]
        if (
            genome_index >= len(genome_seeds)
            or genome_seeds[genome_index] != self.genome_stream_seed
        ):
            raise RecurrentExperimentError(
                "open-ecology genome stream index does not bind its seed"
            )
        if self.genome_population_mode not in {
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        }:
            raise RecurrentExperimentError(
                "open-ecology rollout tasks require heritable or zero_all genomes"
            )
        training_phase = _open_ecology_training_phase(self.open_ecology_training_phase)
        update_index = _nonnegative_int(
            self.open_ecology_update_index,
            field="open_ecology_update_index",
        )
        world_index = _nonnegative_int(
            self.open_ecology_world_index,
            field="open_ecology_world_index",
        )
        expected_density_cycle = _open_ecology_density_cycle(training_phase)
        expected_density = expected_density_cycle[
            world_index % len(expected_density_cycle)
        ]
        if self.open_ecology_treatment.initial_agents != expected_density:
            raise RecurrentExperimentError(
                "open-ecology rollout task density does not match its training phase"
            )
        expected_identity = _open_ecology_policy_sampling_identity(
            learner_seed=learner_seed,
            genome_stream_seed_index=genome_index,
            genome_population_mode=self.genome_population_mode,
            training_phase=training_phase,
            update_index=update_index,
            world_index=world_index,
            environment_seed_index=environment_index,
            initial_agents=self.open_ecology_treatment.initial_agents,
        )
        if self.policy_sampling_identity != expected_identity:
            raise RecurrentExperimentError(
                "open-ecology policy sampling identity is not canonical"
            )
        expected_task_id = _open_ecology_task_id(
            learner_seed=learner_seed,
            genome_stream_seed_index=genome_index,
            genome_population_mode=self.genome_population_mode,
            training_phase=training_phase,
            update_index=update_index,
            world_index=world_index,
            environment_seed_index=environment_index,
            environment_seed=self.environment_seed,
            initial_agents=self.open_ecology_treatment.initial_agents,
        )
        if self.task_id != expected_task_id:
            raise RecurrentExperimentError("open-ecology task_id is not canonical")

    def open_ecology_task_provenance(self) -> dict[str, object]:
        return {
            "schema_version": OPEN_ECOLOGY_TASK_PROVENANCE_SCHEMA_VERSION,
            "seed_registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
            "learner_seed": self.open_ecology_learner_seed,
            "environment_seed": self.environment_seed,
            "environment_seed_index": self.open_ecology_environment_seed_index,
            "genome_stream_seed": self.genome_stream_seed,
            "genome_stream_seed_index": self.open_ecology_genome_stream_seed_index,
            "genome_population_mode": self.genome_population_mode,
            "training_phase": self.open_ecology_training_phase,
            "update_index": self.open_ecology_update_index,
            "world_index": self.open_ecology_world_index,
            "treatment": self.open_ecology_treatment.as_contract(),
        }


def _open_ecology_proof_policy_sampling_identity(
    *,
    environment_seed_index: int,
    initial_agents: int,
) -> str:
    return (
        f"{OPEN_ECOLOGY_PROOF_TASK_IDENTITY_VERSION}|"
        f"registry={OPEN_ECOLOGY_CANONICAL_SHA256}|"
        f"role={OPEN_ECOLOGY_PROOF_SEED_ROLE}|"
        f"model_seed_index=00|genome_stream_seed_index=01|"
        f"environment_seed_index={environment_seed_index:02d}|"
        f"world={environment_seed_index:02d}|"
        f"initial_agents={initial_agents}"
    )


def _open_ecology_proof_task_id(
    *,
    environment_seed_index: int,
    environment_seed: int,
    initial_agents: int,
) -> str:
    return (
        f"open-ecology-proof-world-{environment_seed_index:02d}-"
        f"seed-{environment_seed}-density-{initial_agents}"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenEcologyProofRolloutTask(OpenEcologyRolloutTask):
    """Production open-ecology path bound only to engineering proof seeds."""

    def __post_init__(self) -> None:
        RecurrentRolloutTask.__post_init__(self)
        proof_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_PROOF_SEED_ROLE]
        environment_index = _nonnegative_int(
            self.open_ecology_environment_seed_index,
            field="open_ecology proof environment_seed_index",
        )
        genome_index = _nonnegative_int(
            self.open_ecology_genome_stream_seed_index,
            field="open_ecology proof genome_stream_seed_index",
        )
        if (
            self.scenario != RECURRENT_BROAD_SCENARIO
            or self.seed_role != OPEN_ECOLOGY_PROOF_SEED_ROLE
            or environment_index >= len(proof_seeds)
            or self.environment_seed != proof_seeds[environment_index]
            or self.open_ecology_learner_seed != proof_seeds[0]
            or genome_index != 1
            or self.genome_stream_seed != proof_seeds[genome_index]
            or self.genome_population_mode
            != RecurrentGenomePopulationMode.HERITABLE.value
            or self.open_ecology_training_phase != OPEN_ECOLOGY_PHASE_A
            or self.open_ecology_update_index != 0
            or self.open_ecology_world_index != environment_index
            or self.open_ecology_treatment.initial_agents
            != OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE[0]
            or self.open_ecology_treatment.max_agents != OPEN_ECOLOGY_MAX_AGENTS
        ):
            raise RecurrentExperimentError(
                "open-ecology proof task is detached from its proof-only contract"
            )
        expected_identity = _open_ecology_proof_policy_sampling_identity(
            environment_seed_index=environment_index,
            initial_agents=self.open_ecology_treatment.initial_agents,
        )
        if self.policy_sampling_identity != expected_identity:
            raise RecurrentExperimentError(
                "open-ecology proof task policy identity drifted"
            )
        expected_task_id = _open_ecology_proof_task_id(
            environment_seed_index=environment_index,
            environment_seed=self.environment_seed,
            initial_agents=self.open_ecology_treatment.initial_agents,
        )
        if self.task_id != expected_task_id:
            raise RecurrentExperimentError("open-ecology proof task_id drifted")


def _open_ecology_benchmark_policy_sampling_identity(
    *,
    genome_population_mode: str,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
    initial_agents: int,
) -> str:
    return (
        f"{OPEN_ECOLOGY_BENCHMARK_TASK_IDENTITY_VERSION}|"
        f"registry={OPEN_ECOLOGY_CANONICAL_SHA256}|"
        f"role={OPEN_ECOLOGY_BENCHMARK_SEED_ROLE}|"
        f"model_seed_index={OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX:02d}|"
        "genome_stream_seed_index="
        f"{OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX:02d}|"
        f"genome_mode={genome_population_mode}|phase={OPEN_ECOLOGY_PHASE_A}|"
        f"update={update_index:04d}|world={world_index:06d}|"
        f"environment_seed_index={environment_seed_index:02d}|"
        f"initial_agents={initial_agents}"
    )


def _open_ecology_benchmark_task_id(
    *,
    genome_population_mode: str,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
    environment_seed: int,
    initial_agents: int,
) -> str:
    return (
        f"open-ecology-benchmark-{genome_population_mode}-"
        f"update-{update_index:04d}-world-{world_index:06d}-"
        f"environment-{environment_seed_index:02d}-seed-{environment_seed}-"
        f"density-{initial_agents}"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenEcologyBenchmarkRolloutTask(OpenEcologyRolloutTask):
    """Production open-ecology path bound only to operational benchmark seeds."""

    def __post_init__(self) -> None:
        RecurrentRolloutTask.__post_init__(self)
        benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
        environment_index = _nonnegative_int(
            self.open_ecology_environment_seed_index,
            field="open_ecology benchmark environment_seed_index",
        )
        update_index = _nonnegative_int(
            self.open_ecology_update_index,
            field="open_ecology benchmark update_index",
        )
        world_index = _nonnegative_int(
            self.open_ecology_world_index,
            field="open_ecology benchmark world_index",
        )
        if (
            self.scenario != RECURRENT_BROAD_SCENARIO
            or self.seed_role != OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
            or environment_index
            not in OPEN_ECOLOGY_PHASE_A_BENCHMARK_ENVIRONMENT_SEED_INDICES
            or self.environment_seed != benchmark_seeds[environment_index]
            or self.open_ecology_learner_seed
            != benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]
            or self.open_ecology_genome_stream_seed_index
            != OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
            or self.genome_stream_seed
            != benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX]
            or self.genome_population_mode
            not in {
                RecurrentGenomePopulationMode.HERITABLE.value,
                RecurrentGenomePopulationMode.ZERO_ALL.value,
            }
            or self.open_ecology_training_phase != OPEN_ECOLOGY_PHASE_A
            or self.open_ecology_treatment.initial_agents
            != OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE[0]
            or self.open_ecology_treatment.max_agents != OPEN_ECOLOGY_MAX_AGENTS
        ):
            raise RecurrentExperimentError(
                "open-ecology benchmark task is detached from its "
                "operational-only contract"
            )
        expected_identity = _open_ecology_benchmark_policy_sampling_identity(
            genome_population_mode=self.genome_population_mode,
            update_index=update_index,
            world_index=world_index,
            environment_seed_index=environment_index,
            initial_agents=self.open_ecology_treatment.initial_agents,
        )
        if self.policy_sampling_identity != expected_identity:
            raise RecurrentExperimentError(
                "open-ecology benchmark task policy identity drifted"
            )
        expected_task_id = _open_ecology_benchmark_task_id(
            genome_population_mode=self.genome_population_mode,
            update_index=update_index,
            world_index=world_index,
            environment_seed_index=environment_index,
            environment_seed=self.environment_seed,
            initial_agents=self.open_ecology_treatment.initial_agents,
        )
        if self.task_id != expected_task_id:
            raise RecurrentExperimentError("open-ecology benchmark task_id drifted")


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
    genome_stream_seed: int | None = None,
    genome_population_mode: RecurrentGenomePopulationMode | str = (
        RecurrentGenomePopulationMode.DISABLED
    ),
) -> tuple[tuple[RecurrentRolloutTask, ...], ...]:
    """Build a deterministic policy-optimization schedule from train-only roles.

    The scale-development registry is an explicit opt-in.  It is disjoint from
    every historical development, validation, and lockbox seed and therefore
    cannot be selected by an existing caller accidentally.
    """

    _positive_int(update_count, field="update_count")
    _positive_int(worlds_per_update, field="worlds_per_update")
    _positive_int(rollout_ticks, field="rollout_ticks")
    resolved_genome_population_mode, resolved_genome_stream_seed = (
        _genome_population_configuration(
            population_mode=genome_population_mode,
            stream_seed=genome_stream_seed,
        )
    )
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
    scale_v1_development = (
        seed_registry_contract == RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT
    )
    scale_v2_development = (
        seed_registry_contract == RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT_V2
    )
    scale_development = scale_v1_development or scale_v2_development
    if scale_development:
        if scale_learner_seed is None:
            raise RecurrentExperimentError(
                "scale_learner_seed is required for the scale-development registry"
            )
        resolved_scale_learner_seed = _positive_seed(
            scale_learner_seed,
            field="scale_learner_seed",
        )
        learner_registry = (
            SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"]
            if scale_v2_development
            else SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
        )
        if resolved_scale_learner_seed not in learner_registry:
            raise RecurrentExperimentError(
                "scale_learner_seed must be registered for the scale_learner role"
            )
    else:
        if scale_learner_seed is not None:
            raise RecurrentExperimentError(
                "scale_learner_seed is only valid with the scale-development registry"
            )
        resolved_scale_learner_seed = None
    if scale_v2_development:
        seed_registry = SCALE_DEVELOPMENT_V2_SEED_REGISTRY
        broad_role = "scale_v2_train"
        fixture_role = "scale_v2_curriculum"
        sampling_identity_version = (
            RECURRENT_SCALE_V2_POLICY_SAMPLING_TASK_IDENTITY_VERSION
        )
    elif scale_v1_development:
        seed_registry = SCALE_DEVELOPMENT_SEED_REGISTRY
        broad_role = "scale_train"
        fixture_role = "scale_curriculum"
        sampling_identity_version = (
            RECURRENT_SCALE_POLICY_SAMPLING_TASK_IDENTITY_VERSION
        )
    else:
        seed_registry = RECURRENT_SEED_REGISTRY
        broad_role = "train"
        fixture_role = "curriculum"
        sampling_identity_version = RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION

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
                scale_task_prefix = (
                    "scale-v2-learner" if scale_v2_development else "scale-learner"
                )
                task_id = (
                    f"{scale_task_prefix}-{resolved_scale_learner_seed}-"
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
                    genome_stream_seed=resolved_genome_stream_seed,
                    genome_population_mode=resolved_genome_population_mode,
                )
            )
            global_index += 1
        schedule.append(tuple(tasks))
    return tuple(schedule)


def build_open_ecology_benchmark_schedule(
    *,
    update_count: int,
    worlds_per_update: int,
    rollout_ticks: int,
    genome_population_mode: RecurrentGenomePopulationMode | str,
    environment_seed_offset: int = 0,
) -> tuple[tuple[OpenEcologyBenchmarkRolloutTask, ...], ...]:
    """Build a Phase-A-shaped schedule using only operational benchmark seeds."""

    _positive_int(update_count, field="update_count")
    _positive_int(worlds_per_update, field="worlds_per_update")
    _positive_int(rollout_ticks, field="rollout_ticks")
    try:
        resolved_mode = RecurrentGenomePopulationMode(genome_population_mode)
    except ValueError as error:
        raise RecurrentExperimentError(
            "open-ecology benchmark genome population mode is unsupported"
        ) from error
    if resolved_mode not in {
        RecurrentGenomePopulationMode.HERITABLE,
        RecurrentGenomePopulationMode.ZERO_ALL,
    }:
        raise RecurrentExperimentError(
            "open-ecology benchmark requires heritable or zero_all genomes"
        )
    offset = _nonnegative_int(
        environment_seed_offset,
        field="environment_seed_offset",
    )
    total_worlds = update_count * worlds_per_update
    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
    if offset + total_worlds > OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT:
        raise RecurrentExperimentError(
            "open-ecology benchmark environment range must remain below its "
            "dedicated model and genome seed indices"
        )
    learner_seed = benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]
    genome_stream_seed = benchmark_seeds[
        OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
    ]
    initial_agents = OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE[0]
    treatment = OpenEcologyBroadWorldTreatment(initial_agents=initial_agents)

    schedule: list[tuple[OpenEcologyBenchmarkRolloutTask, ...]] = []
    policy_sampling_seeds: set[int] = set()
    global_index = 0
    for update_index in range(update_count):
        tasks: list[OpenEcologyBenchmarkRolloutTask] = []
        for _ in range(worlds_per_update):
            environment_seed_index = offset + global_index
            environment_seed = benchmark_seeds[environment_seed_index]
            policy_sampling_identity = _open_ecology_benchmark_policy_sampling_identity(
                genome_population_mode=resolved_mode.value,
                update_index=update_index,
                world_index=global_index,
                environment_seed_index=environment_seed_index,
                initial_agents=initial_agents,
            )
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=policy_sampling_identity,
            )
            if policy_sampling_seed in policy_sampling_seeds:
                raise RecurrentExperimentError(
                    "open-ecology benchmark policy sampling seeds must be unique"
                )
            policy_sampling_seeds.add(policy_sampling_seed)
            tasks.append(
                OpenEcologyBenchmarkRolloutTask(
                    task_id=_open_ecology_benchmark_task_id(
                        genome_population_mode=resolved_mode.value,
                        update_index=update_index,
                        world_index=global_index,
                        environment_seed_index=environment_seed_index,
                        environment_seed=environment_seed,
                        initial_agents=initial_agents,
                    ),
                    scenario=RECURRENT_BROAD_SCENARIO,
                    environment_seed=environment_seed,
                    rollout_ticks=rollout_ticks,
                    policy_sampling_identity=policy_sampling_identity,
                    policy_sampling_seed=policy_sampling_seed,
                    seed_role=OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
                    genome_stream_seed=genome_stream_seed,
                    genome_population_mode=resolved_mode.value,
                    open_ecology_treatment=treatment,
                    open_ecology_learner_seed=learner_seed,
                    open_ecology_environment_seed_index=environment_seed_index,
                    open_ecology_genome_stream_seed_index=(
                        OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
                    ),
                    open_ecology_training_phase=OPEN_ECOLOGY_PHASE_A,
                    open_ecology_update_index=update_index,
                    open_ecology_world_index=global_index,
                )
            )
            global_index += 1
        schedule.append(tuple(tasks))
    return tuple(schedule)


def build_open_ecology_training_schedule(
    *,
    training_phase: str,
    update_count: int,
    worlds_per_update: int,
    rollout_ticks: int,
    learner_seed: int,
    genome_stream_seed: int,
    genome_population_mode: RecurrentGenomePopulationMode | str,
    environment_seed_offset: int = 0,
) -> tuple[tuple[OpenEcologyRolloutTask, ...], ...]:
    """Build one learner-namespaced, broad-only open-ecology schedule.

    Environment seeds are consumed once from a contiguous registry slice. The
    explicit offset permits preregistered disjoint ranges while avoiding the
    modulo wraparound used by legacy schedules.
    """

    resolved_training_phase = _open_ecology_training_phase(training_phase)
    density_cycle = _open_ecology_density_cycle(resolved_training_phase)
    _positive_int(update_count, field="update_count")
    _positive_int(worlds_per_update, field="worlds_per_update")
    _positive_int(rollout_ticks, field="rollout_ticks")
    resolved_learner_seed = _positive_seed(
        learner_seed,
        field="learner_seed",
    )
    if resolved_learner_seed not in OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"]:
        raise RecurrentExperimentError("open-ecology learner_seed must be registered")
    resolved_mode, resolved_genome_seed = _genome_population_configuration(
        population_mode=genome_population_mode,
        stream_seed=genome_stream_seed,
    )
    if resolved_mode not in {
        RecurrentGenomePopulationMode.HERITABLE.value,
        RecurrentGenomePopulationMode.ZERO_ALL.value,
    }:
        raise RecurrentExperimentError(
            "open-ecology schedule requires heritable or zero_all genomes"
        )
    genome_seeds = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"]
    if resolved_genome_seed not in genome_seeds:
        raise RecurrentExperimentError(
            "open-ecology genome_stream_seed must be registered"
        )
    genome_stream_seed_index = genome_seeds.index(resolved_genome_seed)
    offset = _nonnegative_int(
        environment_seed_offset,
        field="environment_seed_offset",
    )
    total_worlds = update_count * worlds_per_update
    train_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_TRAINING_SEED_ROLE]
    if offset + total_worlds > len(train_seeds):
        raise RecurrentExperimentError(
            "open-ecology environment seed range exceeds the non-reused registry"
        )

    schedule: list[tuple[OpenEcologyRolloutTask, ...]] = []
    policy_sampling_seeds: set[int] = set()
    global_index = 0
    for update_index in range(update_count):
        tasks: list[OpenEcologyRolloutTask] = []
        for _ in range(worlds_per_update):
            environment_seed_index = offset + global_index
            environment_seed = train_seeds[environment_seed_index]
            initial_agents = density_cycle[global_index % len(density_cycle)]
            treatment = OpenEcologyBroadWorldTreatment(
                initial_agents=initial_agents,
            )
            policy_sampling_identity = _open_ecology_policy_sampling_identity(
                learner_seed=resolved_learner_seed,
                genome_stream_seed_index=genome_stream_seed_index,
                genome_population_mode=resolved_mode,
                training_phase=resolved_training_phase,
                update_index=update_index,
                world_index=global_index,
                environment_seed_index=environment_seed_index,
                initial_agents=initial_agents,
            )
            policy_sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=policy_sampling_identity
            )
            if policy_sampling_seed in policy_sampling_seeds:
                raise RecurrentExperimentError(
                    "policy sampling seed collision in open-ecology schedule"
                )
            policy_sampling_seeds.add(policy_sampling_seed)
            task_id = _open_ecology_task_id(
                learner_seed=resolved_learner_seed,
                genome_stream_seed_index=genome_stream_seed_index,
                genome_population_mode=resolved_mode,
                training_phase=resolved_training_phase,
                update_index=update_index,
                world_index=global_index,
                environment_seed_index=environment_seed_index,
                environment_seed=environment_seed,
                initial_agents=initial_agents,
            )
            tasks.append(
                OpenEcologyRolloutTask(
                    task_id=task_id,
                    scenario=RECURRENT_BROAD_SCENARIO,
                    environment_seed=environment_seed,
                    rollout_ticks=rollout_ticks,
                    policy_sampling_identity=policy_sampling_identity,
                    policy_sampling_seed=policy_sampling_seed,
                    seed_role=OPEN_ECOLOGY_TRAINING_SEED_ROLE,
                    genome_stream_seed=resolved_genome_seed,
                    genome_population_mode=resolved_mode,
                    open_ecology_treatment=treatment,
                    open_ecology_learner_seed=resolved_learner_seed,
                    open_ecology_environment_seed_index=environment_seed_index,
                    open_ecology_genome_stream_seed_index=(genome_stream_seed_index),
                    open_ecology_training_phase=resolved_training_phase,
                    open_ecology_update_index=update_index,
                    open_ecology_world_index=global_index,
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
    if any(
        task.genome_population_mode != RecurrentGenomePopulationMode.DISABLED.value
        for task in tasks
    ):
        raise RecurrentExperimentError(
            "conditioned counterfactual collection is blocked until exact branch "
            "rows carry genotype provenance"
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
        identity_version = (
            RECURRENT_SCALE_V2_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION
            if str(task.seed_role).startswith("scale_v2_")
            else RECURRENT_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION
        )
        identity_prefix = (
            f"{identity_version}|"
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


def _fixed_batch_contract_for_tasks(
    tasks: Sequence[RecurrentRolloutTask],
    *,
    fixed_batch_capacity: int | None,
) -> RecurrentFixedBatchRuntimeContract | None:
    if fixed_batch_capacity is None:
        return None
    open_ecology = any(isinstance(task, OpenEcologyRolloutTask) for task in tasks)
    if open_ecology:
        if not all(isinstance(task, OpenEcologyRolloutTask) for task in tasks):
            raise RecurrentExperimentError(
                "fixed recurrent rollout batching cannot mix open-ecology and "
                "legacy tasks"
            )
        resolved_capacity = _positive_int(
            fixed_batch_capacity,
            field="fixed_batch_capacity",
        )
        if resolved_capacity != OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY:
            raise RecurrentExperimentError(
                "open-ecology rollout batching requires the preregistered "
                f"capacity {OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY}"
            )
        return RecurrentFixedBatchRuntimeContract(batch_capacity=resolved_capacity)
    raise RecurrentExperimentError(
        "fixed recurrent rollout batching is currently restricted to open-ecology tasks"
    )


def collect_recurrent_rollout_batch(
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentRolloutTask],
    *,
    feed_forward_history_ablation: bool = False,
    rollout_workers: int = 1,
    fixed_batch_capacity: int | None = None,
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
    _validate_rollout_task_genome_bindings(model=model, tasks=task_tuple)
    _validate_rollout_task_treatment_bindings(model=model, tasks=task_tuple)
    fixed_batch_contract = _fixed_batch_contract_for_tasks(
        task_tuple,
        fixed_batch_capacity=fixed_batch_capacity,
    )

    behavior_model = frozen_cpu_model_copy(model)
    if fixed_batch_contract is not None:
        task_results = _collect_recurrent_rollout_tasks_parallel(
            behavior_model,
            task_tuple,
            feed_forward_history_ablation=feed_forward_history_ablation,
            rollout_workers=rollout_workers,
            fixed_batch_contract=fixed_batch_contract,
        )
    elif rollout_workers == 1 or len(task_tuple) == 1:
        task_results = tuple(
            _collect_recurrent_rollout_task(
                behavior_model,
                task,
                feed_forward_history_ablation=feed_forward_history_ablation,
                fixed_batch_contract=fixed_batch_contract,
            )
            for task in task_tuple
        )
    else:
        task_results = _collect_recurrent_rollout_tasks_parallel(
            behavior_model,
            task_tuple,
            feed_forward_history_ablation=feed_forward_history_ablation,
            rollout_workers=rollout_workers,
            fixed_batch_contract=fixed_batch_contract,
        )
    if len(task_results) != len(task_tuple):
        raise RecurrentExperimentError(
            "rollout worker result count does not match the task batch"
        )

    buffer = RecurrentRolloutBuffer()
    summaries: list[dict[str, object]] = []
    for task, task_result in zip(task_tuple, task_results, strict=True):
        steps, summary = task_result
        worker_provenance = _validate_rollout_worker_result(
            task=task,
            summary=summary,
            steps=steps,
            genome_conditioning_mode=model.config.genome_conditioning_mode,
            expected_fixed_batch_contract=fixed_batch_contract,
        )
        genome_registration: dict[str, object] = {}
        if task.genome_population_mode != RecurrentGenomePopulationMode.DISABLED.value:
            genome_registration = {
                "genome_conditioning_mode": worker_provenance[
                    "genome_conditioning_mode"
                ],
                "genome_population_mode": worker_provenance["genome_population_mode"],
                "genome_stream_seed": worker_provenance["genome_stream_seed"],
                "genome_population_binding_sha256": worker_provenance[
                    "genome_population_binding_sha256"
                ],
                "genome_population_pre_founder_state_sha256": worker_provenance[
                    "genome_population_pre_founder_state_sha256"
                ],
            }
            if "genome_world_identity" in worker_provenance:
                genome_registration["genome_world_identity"] = worker_provenance[
                    "genome_world_identity"
                ]
        fixed_batch_registration: dict[str, object] = {}
        if "fixed_batch_runtime" in worker_provenance:
            fixed_batch_registration = {
                "fixed_batch_runtime": worker_provenance["fixed_batch_runtime"]
            }
        buffer.register_world(
            task.task_id,
            environment_seed=task.environment_seed,
            policy_sampling_seed=task.policy_sampling_seed,
            **fixed_batch_registration,
            **genome_registration,
        )
        for step in steps:
            buffer.append(step)
        if genome_registration:
            buffer.finalize_world_genome_provenance(
                task.task_id,
                final_state_sha256=str(
                    worker_provenance["genome_population_final_state_sha256"]
                ),
                reset_state_sha256=str(
                    worker_provenance["genome_population_reset_state_sha256"]
                ),
            )
        buffer.validate_world_closed(task.task_id)
        buffer_worker_provenance = {
            key: value
            for key, value in worker_provenance.items()
            if key != "open_ecology_task_provenance"
        }
        if buffer.world_seed_provenance.get(task.task_id) != buffer_worker_provenance:
            raise RecurrentExperimentError(
                "rollout worker and merged buffer genome provenance drifted"
            )
        summaries.append(summary)
    diagnostics = _rollout_diagnostics(buffer, world_summaries=summaries)
    return buffer, diagnostics


def _collect_recurrent_rollout_task(
    behavior_model: PublicRecurrentActorCritic,
    task: RecurrentRolloutTask,
    *,
    feed_forward_history_ablation: bool,
    fixed_batch_contract: RecurrentFixedBatchRuntimeContract | None = None,
) -> tuple[tuple[RecurrentRolloutStep, ...], dict[str, object]]:
    """Collect one isolated world so sequential and worker paths share code."""

    core = TorchRecurrentPolicyCore(behavior_model)
    task_buffer = RecurrentRolloutBuffer()
    collector = RecurrentOnPolicyCollector(
        core,
        buffer=task_buffer,
        reset_recurrent_state_each_decision=feed_forward_history_ablation,
        fixed_batch_contract=fixed_batch_contract,
    )
    collector.start_world(
        world_id=task.task_id,
        genome_world_identity=getattr(
            task,
            "phase_a_genome_world_identity",
            None,
        ),
        rollout_ticks=task.rollout_ticks,
        environment_seed=task.environment_seed,
        policy_sampling_seed=task.policy_sampling_seed,
        genome_stream_seed=task.genome_stream_seed,
        genome_population_mode=task.genome_population_mode,
    )
    world = _world_for_task(task, policy=collector)
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    alive_agents = world.alive_agents()
    if alive_agents:
        prepared = world.prepare_policy_visible_tick_start_on_clone(
            tick=task.rollout_ticks
        )
        terminal_bootstrap = collector.finalize_action_free_bootstrap(
            tick=task.rollout_ticks,
            ordered_agent_ids=prepared.ordered_agent_ids,
            observations_by_agent=prepared.observation_snapshots,
        )
    else:
        terminal_bootstrap = collector.finalize_action_free_bootstrap(
            tick=task.rollout_ticks,
            ordered_agent_ids=(),
            observations_by_agent={},
        )
    collector.finish_world()
    world_seed_provenance = task_buffer.world_seed_provenance
    if set(world_seed_provenance) != {task.task_id}:
        raise RecurrentExperimentError(
            "isolated rollout task emitted invalid world provenance coverage"
        )
    summary: dict[str, object] = {
        "task_id": task.task_id,
        "scenario": task.scenario,
        "environment_seed": task.environment_seed,
        "seed_role": task.seed_role,
        "policy_sampling_identity": task.policy_sampling_identity,
        "policy_sampling_seed": task.policy_sampling_seed,
        "policy_sampling_seed_namespace": RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
        "rollout_ticks": task.rollout_ticks,
        "terminal_bootstrap": terminal_bootstrap,
        "summary": _compact_world_summary(result.summary),
    }
    if task.genome_population_mode != RecurrentGenomePopulationMode.DISABLED.value:
        summary["world_seed_provenance"] = world_seed_provenance[task.task_id]
    if isinstance(task, OpenEcologyRolloutTask):
        summary["open_ecology_task_provenance"] = task.open_ecology_task_provenance()
    return task_buffer.steps, summary


def _collect_recurrent_rollout_tasks_parallel(
    behavior_model: PublicRecurrentActorCritic,
    tasks: tuple[RecurrentRolloutTask, ...],
    *,
    feed_forward_history_ablation: bool,
    rollout_workers: int,
    fixed_batch_contract: RecurrentFixedBatchRuntimeContract | None,
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
    fixed_batch_torch_runtime = (
        _fixed_batch_worker_torch_runtime_contract()
        if fixed_batch_contract is not None
        else None
    )
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_recurrent_rollout_worker,
            initargs=(
                behavior_model.config,
                model_state,
                feed_forward_history_ablation,
                fixed_batch_contract,
                fixed_batch_torch_runtime,
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
    fixed_batch_contract: RecurrentFixedBatchRuntimeContract | None,
    fixed_batch_torch_runtime: Mapping[str, object] | None,
) -> None:
    """Initialize one inference-only CPU model and bounded Torch thread pool."""

    global _WORKER_BEHAVIOR_MODEL
    global _WORKER_FEED_FORWARD_HISTORY_ABLATION
    global _WORKER_FIXED_BATCH_CONTRACT
    if fixed_batch_contract is not None:
        _apply_torch_determinism_runtime_state(fixed_batch_torch_runtime)
    else:
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
    _WORKER_FIXED_BATCH_CONTRACT = fixed_batch_contract


def _collect_recurrent_rollout_worker(
    task: RecurrentRolloutTask,
) -> tuple[tuple[RecurrentRolloutStep, ...], dict[str, object]]:
    model = _WORKER_BEHAVIOR_MODEL
    ablation = _WORKER_FEED_FORWARD_HISTORY_ABLATION
    fixed_batch_contract = _WORKER_FIXED_BATCH_CONTRACT
    if model is None or type(ablation) is not bool:
        raise RecurrentExperimentError("recurrent rollout worker was not initialized")
    return _collect_recurrent_rollout_task(
        model,
        task,
        feed_forward_history_ablation=ablation,
        fixed_batch_contract=fixed_batch_contract,
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
        fixed_batch_capacity: int | None = None,
        counterfactual_config: RecurrentCounterfactualExperimentConfig | None = None,
    ) -> None:
        _positive_seed(learner_seed, field="learner_seed")
        self.learner_seed = learner_seed
        self.device = torch.device(device)
        self.rollout_workers = _rollout_worker_count(rollout_workers)
        self.fixed_batch_capacity = (
            None
            if fixed_batch_capacity is None
            else _positive_int(
                fixed_batch_capacity,
                field="fixed_batch_capacity",
            )
        )
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
        resolved_model_config = model_config or RecurrentActorCriticConfig()
        if (
            counterfactual_config is not None
            and resolved_model_config.genome_conditioning_mode
            != RECURRENT_GENOME_CONDITIONING_DISABLED
        ):
            raise RecurrentExperimentError(
                "conditioned counterfactual auxiliary training is blocked until "
                "exact branch rows carry genotype provenance"
            )
        self.model = PublicRecurrentActorCritic(
            resolved_model_config,
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
        open_ecology_tasks = tuple(
            task for task in task_tuple if isinstance(task, OpenEcologyRolloutTask)
        )
        if open_ecology_tasks and any(
            task.open_ecology_learner_seed != self.learner_seed
            for task in open_ecology_tasks
        ):
            raise RecurrentExperimentError(
                "open-ecology task learner seed must match the experiment runner"
            )
        if self.counterfactual_config is not None and any(
            task.genome_population_mode != RecurrentGenomePopulationMode.DISABLED.value
            for task in task_tuple
        ):
            raise RecurrentExperimentError(
                "conditioned counterfactual auxiliary training is blocked until "
                "exact branch rows carry genotype provenance"
            )
        buffer, rollout_diagnostics = collect_recurrent_rollout_batch(
            self.model,
            task_tuple,
            feed_forward_history_ablation=(
                self.ppo_config.feed_forward_history_ablation
            ),
            rollout_workers=self.rollout_workers,
            fixed_batch_capacity=self.fixed_batch_capacity,
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
        schedule_genome_configuration = _validate_rollout_task_genome_bindings(
            model=self.model,
            tasks=scheduled_tasks,
        )
        schedule_open_ecology_treatments = _validate_rollout_task_treatment_bindings(
            model=self.model,
            tasks=scheduled_tasks,
        )
        open_ecology_tasks = tuple(
            task for task in scheduled_tasks if isinstance(task, OpenEcologyRolloutTask)
        )
        if open_ecology_tasks and any(
            task.open_ecology_learner_seed != self.learner_seed
            for task in open_ecology_tasks
        ):
            raise RecurrentExperimentError(
                "open-ecology schedule learner seed must match the experiment runner"
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
        rollout_execution: dict[str, object] = {
            "requested_workers": self.rollout_workers,
            "start_method": RECURRENT_ROLLOUT_WORKER_START_METHOD,
            "torch_threads_per_worker": RECURRENT_ROLLOUT_WORKER_TORCH_THREADS,
            "ordered_merge": True,
            "sequential_default": True,
        }
        genome_population_mode, genome_stream_seed = schedule_genome_configuration
        if genome_population_mode != RecurrentGenomePopulationMode.DISABLED.value:
            rollout_execution["genome_conditioning"] = {
                "conditioning_mode": self.model.config.genome_conditioning_mode,
                "population_mode": genome_population_mode,
                "stream_seed": genome_stream_seed,
                "per_world_full_seed_provenance_recorded": True,
            }
        if schedule_open_ecology_treatments is not None:
            unique_treatments: list[dict[str, object]] = []
            for treatment in schedule_open_ecology_treatments:
                payload = treatment.as_contract()
                if payload not in unique_treatments:
                    unique_treatments.append(payload)
            rollout_execution["open_ecology"] = {
                "treatment_schema_version": OPEN_ECOLOGY_TREATMENT_SCHEMA_VERSION,
                "broad_world_only": True,
                "full_task_worker_merge_diagnostic_provenance": True,
                "fixed_batch_runtime_contract": (
                    None
                    if self.fixed_batch_capacity is None
                    else RecurrentFixedBatchRuntimeContract(
                        batch_capacity=self.fixed_batch_capacity,
                    ).as_contract()
                ),
                "fixed_batch_execution_scope": (
                    None
                    if self.fixed_batch_capacity is None
                    else (
                        "experimental_opt_in_intra_world_cpu_worker_only_"
                        "cross_world_gpu_batching_excluded_v1"
                    )
                ),
                "fixed_batch_enabled": self.fixed_batch_capacity is not None,
                "fixed_batch_default_enabled": False,
                "fixed_batch_release_status": RECURRENT_FIXED_BATCH_RELEASE_STATUS,
                "fixed_batch_authoritative_launch_gate_satisfied": False,
                "treatments": unique_treatments,
                "treatment_runtime_ready": True,
                "authoritative_campaign_launch_claimed": False,
                "launch_dependencies": asdict(OpenEcologyLaunchDependencies()),
            }
        return RecurrentTrainingRunResult(
            contract_version=(
                RECURRENT_EXPERIMENT_CONTRACT_VERSION
                if self.fixed_batch_capacity is None
                else RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION
            ),
            learner_seed=self.learner_seed,
            device=str(self.device),
            deterministic_algorithms_enabled=(
                torch.are_deterministic_algorithms_enabled()
            ),
            model_config=asdict(self.model.config),
            ppo_config=asdict(self.ppo_config),
            rollout_execution=rollout_execution,
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
    observed_roles = {
        str(task.seed_role) for update in updates for task in update.tasks
    }
    if observed_roles == {OPEN_ECOLOGY_TRAINING_SEED_ROLE}:
        return _open_ecology_training_seed_provenance(updates)
    if observed_roles == {OPEN_ECOLOGY_BENCHMARK_SEED_ROLE}:
        return _open_ecology_benchmark_seed_provenance(updates)
    if {
        OPEN_ECOLOGY_TRAINING_SEED_ROLE,
        OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
    }.intersection(observed_roles):
        raise RecurrentExperimentError(
            "committed updates mix open-ecology and legacy seed registries"
        )
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
        RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT_V2: (
            ("scale_v2_train", "scale_v2_curriculum"),
            SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
            RECURRENT_SCALE_V2_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION,
        ),
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
    if selected_contract in {
        RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
        RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT_V2,
    }:
        payload["seed_registry_contract"] = selected_contract
    return payload


def _open_ecology_training_seed_provenance(
    updates: Sequence[RecurrentTrainingUpdateResult],
) -> dict[str, object]:
    tasks = tuple(task for update in updates for task in update.tasks)
    if not tasks or any(not isinstance(task, OpenEcologyRolloutTask) for task in tasks):
        raise RecurrentExperimentError(
            "open-ecology updates require exact open-ecology rollout tasks"
        )
    open_tasks = tuple(
        task for task in tasks if isinstance(task, OpenEcologyRolloutTask)
    )
    learner_seeds = {task.open_ecology_learner_seed for task in open_tasks}
    genome_bindings = {
        (
            task.genome_stream_seed,
            task.open_ecology_genome_stream_seed_index,
            task.genome_population_mode,
        )
        for task in open_tasks
    }
    training_phases = {task.open_ecology_training_phase for task in open_tasks}
    if (
        len(learner_seeds) != 1
        or len(genome_bindings) != 1
        or len(training_phases) != 1
    ):
        raise RecurrentExperimentError(
            "open-ecology updates mix learner, genome, or training-phase axes"
        )
    environment_indices = [
        task.open_ecology_environment_seed_index for task in open_tasks
    ]
    if environment_indices != list(
        range(environment_indices[0], environment_indices[0] + len(open_tasks))
    ):
        raise RecurrentExperimentError(
            "open-ecology environment seeds must form one ordered non-reused range"
        )
    environment_seeds = [task.environment_seed for task in open_tasks]
    if len(environment_seeds) != len(set(environment_seeds)):
        raise RecurrentExperimentError(
            "open-ecology environment seeds cannot be reused in one schedule"
        )
    training_phase = _open_ecology_training_phase(next(iter(training_phases)))
    density_cycle = _open_ecology_density_cycle(training_phase)
    for global_index, task in enumerate(open_tasks):
        if (
            task.open_ecology_world_index != global_index
            or task.open_ecology_treatment.initial_agents
            != density_cycle[global_index % len(density_cycle)]
        ):
            raise RecurrentExperimentError(
                "open-ecology task order or density cycle drifted"
            )
    for update_index, update in enumerate(updates):
        if any(
            not isinstance(task, OpenEcologyRolloutTask)
            or task.open_ecology_update_index != update_index
            for task in update.tasks
        ):
            raise RecurrentExperimentError(
                "open-ecology update index provenance drifted"
            )

    learner_seed = next(iter(learner_seeds))
    learner_seed_index = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"].index(
        learner_seed
    )
    genome_stream_seed, genome_stream_seed_index, genome_population_mode = next(
        iter(genome_bindings)
    )
    return {
        "schema_version": OPEN_ECOLOGY_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION,
        "seed_registry_contract": {
            "version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        },
        "environment_seed_roles": [OPEN_ECOLOGY_TRAINING_SEED_ROLE],
        "environment_seeds_by_role": {
            OPEN_ECOLOGY_TRAINING_SEED_ROLE: environment_seeds,
        },
        "observed_environment_seed_count": len(environment_seeds),
        "canonical_registry_membership_valid": True,
        "registry_ordered_non_reused_range": {
            "offset": environment_indices[0],
            "count": len(environment_indices),
            "exclusive_stop": environment_indices[-1] + 1,
        },
        "learner": {
            "seed": learner_seed,
            "registry_index": learner_seed_index,
        },
        "genome_stream": {
            "seed": genome_stream_seed,
            "registry_index": genome_stream_seed_index,
        },
        "genome_population_mode": genome_population_mode,
        "training_phase": training_phase,
        "initial_agent_density_cycle": list(density_cycle),
        "non_training_seed_values_exposed": False,
    }


def _open_ecology_benchmark_seed_provenance(
    updates: Sequence[RecurrentTrainingUpdateResult],
) -> dict[str, object]:
    tasks = tuple(task for update in updates for task in update.tasks)
    if not tasks or any(
        not isinstance(task, OpenEcologyBenchmarkRolloutTask) for task in tasks
    ):
        raise RecurrentExperimentError(
            "open-ecology benchmark updates require exact benchmark rollout tasks"
        )
    benchmark_tasks = tuple(
        task for task in tasks if isinstance(task, OpenEcologyBenchmarkRolloutTask)
    )
    environment_indices = [
        task.open_ecology_environment_seed_index for task in benchmark_tasks
    ]
    if environment_indices != list(
        range(environment_indices[0], environment_indices[0] + len(benchmark_tasks))
    ):
        raise RecurrentExperimentError(
            "open-ecology benchmark seeds must form one ordered non-reused range"
        )
    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
    environment_seeds = [task.environment_seed for task in benchmark_tasks]
    if environment_seeds != [
        benchmark_seeds[index] for index in environment_indices
    ] or len(environment_seeds) != len(set(environment_seeds)):
        raise RecurrentExperimentError(
            "open-ecology benchmark environment seed provenance drifted"
        )
    learner_seed = benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]
    genome_stream_seed = benchmark_seeds[
        OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
    ]
    population_modes = {task.genome_population_mode for task in benchmark_tasks}
    if (
        {task.open_ecology_learner_seed for task in benchmark_tasks} != {learner_seed}
        or {task.genome_stream_seed for task in benchmark_tasks} != {genome_stream_seed}
        or {task.open_ecology_genome_stream_seed_index for task in benchmark_tasks}
        != {OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX}
        or len(population_modes) != 1
    ):
        raise RecurrentExperimentError(
            "open-ecology benchmark model or genome seed provenance drifted"
        )
    global_index = 0
    for expected_update_index, update in enumerate(updates):
        if not update.tasks:
            raise RecurrentExperimentError(
                "open-ecology benchmark updates cannot be empty"
            )
        for task in update.tasks:
            if (
                task.open_ecology_update_index != expected_update_index
                or task.open_ecology_world_index != global_index
                or task.open_ecology_training_phase != OPEN_ECOLOGY_PHASE_A
                or task.open_ecology_treatment.initial_agents
                != OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE[0]
            ):
                raise RecurrentExperimentError(
                    "open-ecology benchmark task order or Phase-A shape drifted"
                )
            global_index += 1
    return {
        "schema_version": OPEN_ECOLOGY_BENCHMARK_SEED_PROVENANCE_SCHEMA_VERSION,
        "seed_registry_contract": {
            "version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        },
        "environment_seed_roles": [OPEN_ECOLOGY_BENCHMARK_SEED_ROLE],
        "environment_seeds_by_role": {
            OPEN_ECOLOGY_BENCHMARK_SEED_ROLE: environment_seeds,
        },
        "environment_seed_indices": environment_indices,
        "observed_environment_seed_count": len(environment_seeds),
        "canonical_registry_membership_valid": True,
        "registry_ordered_non_reused_range": {
            "offset": environment_indices[0],
            "count": len(environment_indices),
            "exclusive_stop": environment_indices[-1] + 1,
        },
        "model_initialization": {
            "seed": learner_seed,
            "registry_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "registry_index": OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX,
        },
        "genome_stream": {
            "seed": genome_stream_seed,
            "registry_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "registry_index": OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        },
        "genome_population_mode": next(iter(population_modes)),
        "training_phase": OPEN_ECOLOGY_PHASE_A,
        "initial_agent_density_cycle": list(OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE),
        "scientific_environment_seed_roles_accessed": [],
        "training_seeds_accessed": False,
        "selection_seeds_accessed": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
    }


def _require_unique_schedule_values(
    values: Sequence[object],
    *,
    field: str,
) -> None:
    if len(values) != len(set(values)):
        raise RecurrentExperimentError(
            f"scheduled {field} values must be globally unique"
        )


def _open_ecology_training_phase(value: object) -> str:
    if not isinstance(value, str) or value not in OPEN_ECOLOGY_TRAINING_PHASES:
        raise RecurrentExperimentError(
            "open-ecology training_phase must be phase_a or phase_b"
        )
    return value


def _open_ecology_density_cycle(training_phase: object) -> tuple[int, ...]:
    resolved = _open_ecology_training_phase(training_phase)
    if resolved == OPEN_ECOLOGY_PHASE_A:
        return OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE
    return OPEN_ECOLOGY_PHASE_B_DENSITY_CYCLE


def _genome_population_configuration(
    *,
    population_mode: object,
    stream_seed: object,
) -> tuple[str, int | None]:
    try:
        resolved_mode = RecurrentGenomePopulationMode(population_mode)
    except (TypeError, ValueError) as error:
        raise RecurrentExperimentError(
            "genome_population_mode must be disabled, heritable, or zero_all"
        ) from error
    if resolved_mode is RecurrentGenomePopulationMode.DISABLED:
        if stream_seed is not None:
            raise RecurrentExperimentError(
                "disabled genome population cannot define genome_stream_seed"
            )
        return resolved_mode.value, None
    if isinstance(stream_seed, bool) or not isinstance(stream_seed, int):
        raise RecurrentExperimentError(
            "active genome population requires an unsigned 64-bit genome_stream_seed"
        )
    if stream_seed < 0 or stream_seed > MAX_RECURRENT_GENOME_STREAM_SEED:
        raise RecurrentExperimentError(
            "genome_stream_seed must be an unsigned 64-bit integer"
        )
    return resolved_mode.value, stream_seed


def _validate_rollout_task_treatment_bindings(
    *,
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentRolloutTask],
) -> tuple[OpenEcologyBroadWorldTreatment, ...] | None:
    open_tasks = tuple(
        task for task in tasks if isinstance(task, OpenEcologyRolloutTask)
    )
    if not open_tasks:
        return None
    if len(open_tasks) != len(tasks):
        raise RecurrentExperimentError(
            "open-ecology and legacy rollout tasks cannot share one batch"
        )
    expected_signal_config = open_tasks[
        0
    ].open_ecology_treatment.signals.as_signal_config()
    expected_schema = ecological_policy_input_schema_version(expected_signal_config)
    expected_size = ecological_policy_input_vector_size(expected_signal_config)
    if (
        model.config.public_input_schema_version != expected_schema
        or model.config.public_input_size != expected_size
    ):
        raise RecurrentExperimentError(
            "recurrent model public input schema/size does not match the "
            "open-ecology signal treatment"
        )
    treatments: list[OpenEcologyBroadWorldTreatment] = []
    for task in open_tasks:
        treatment = task.open_ecology_treatment
        signal_config = treatment.signals.as_signal_config()
        if (
            ecological_policy_input_schema_version(signal_config) != expected_schema
            or ecological_policy_input_vector_size(signal_config) != expected_size
        ):
            raise RecurrentExperimentError(
                "open-ecology tasks disagree on the public signal input contract"
            )
        treatments.append(treatment)
    return tuple(treatments)


def _open_ecology_treatment_from_payload(
    value: object,
) -> OpenEcologyBroadWorldTreatment:
    if not isinstance(value, Mapping):
        raise RecurrentExperimentError(
            "open-ecology treatment provenance must be a mapping"
        )
    canonical_field_names = set(OpenEcologyBroadWorldTreatment(32).as_contract())
    if set(value) != canonical_field_names:
        raise RecurrentExperimentError(
            "open-ecology treatment provenance fields do not match the exact contract"
        )
    raw_signals = value.get("signals")
    raw_dependencies = value.get("launch_dependencies")
    if not isinstance(raw_signals, Mapping) or set(raw_signals) != set(
        _OPEN_ECOLOGY_SIGNAL_VALUES
    ):
        raise RecurrentExperimentError(
            "open-ecology signal provenance fields do not match the exact contract"
        )
    dependency_fields = {
        "self_echo_exclusion_implemented",
        "self_echo_exclusion_required_before_authoritative_launch",
    }
    if (
        not isinstance(raw_dependencies, Mapping)
        or set(raw_dependencies) != dependency_fields
    ):
        raise RecurrentExperimentError(
            "open-ecology launch dependency fields do not match the exact contract"
        )
    try:
        treatment = OpenEcologyBroadWorldTreatment(
            initial_agents=value.get("initial_agents"),  # type: ignore[arg-type]
            schema_version=value.get("schema_version"),  # type: ignore[arg-type]
            max_agents=value.get("max_agents"),  # type: ignore[arg-type]
            signals=OpenEcologySignalTreatment(**dict(raw_signals)),
            reward_contract=value.get("reward_contract"),  # type: ignore[arg-type]
            reward_shaping_added=value.get("reward_shaping_added"),  # type: ignore[arg-type]
            action_heuristics_added=value.get("action_heuristics_added"),  # type: ignore[arg-type]
            launch_dependencies=OpenEcologyLaunchDependencies(**dict(raw_dependencies)),
        )
    except (TypeError, ValueError) as error:
        raise RecurrentExperimentError(
            "open-ecology treatment provenance is invalid"
        ) from error
    if treatment.as_contract() != dict(value):
        raise RecurrentExperimentError(
            "open-ecology treatment provenance is not canonical"
        )
    return treatment


def _open_ecology_task_provenance_from_payload(
    value: object,
    *,
    seed_role: object,
    environment_seed: int,
    genome_stream_seed: int | None,
    genome_population_mode: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentExperimentError("open-ecology task provenance must be a mapping")
    if set(value) != _OPEN_ECOLOGY_TASK_PROVENANCE_KEYS:
        raise RecurrentExperimentError(
            "open-ecology task provenance fields do not match the exact contract"
        )
    if (
        value.get("schema_version") != OPEN_ECOLOGY_TASK_PROVENANCE_SCHEMA_VERSION
        or value.get("seed_registry_version") != OPEN_ECOLOGY_SEED_REGISTRY_VERSION
        or value.get("seed_registry_sha256") != OPEN_ECOLOGY_CANONICAL_SHA256
    ):
        raise RecurrentExperimentError(
            "open-ecology task provenance registry contract drifted"
        )
    resolved_seed_role = str(seed_role)
    if resolved_seed_role not in {
        OPEN_ECOLOGY_TRAINING_SEED_ROLE,
        OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        OPEN_ECOLOGY_PROOF_SEED_ROLE,
    }:
        raise RecurrentExperimentError(
            "open-ecology task provenance seed role is not authorized"
        )
    learner_seed = _positive_seed(
        value.get("learner_seed"),
        field="open-ecology task learner_seed",
    )
    expected_learner_role = {
        OPEN_ECOLOGY_TRAINING_SEED_ROLE: "open_ecology_learner",
        OPEN_ECOLOGY_BENCHMARK_SEED_ROLE: OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        OPEN_ECOLOGY_PROOF_SEED_ROLE: OPEN_ECOLOGY_PROOF_SEED_ROLE,
    }[resolved_seed_role]
    if learner_seed not in OPEN_ECOLOGY_SEED_REGISTRY[expected_learner_role]:
        raise RecurrentExperimentError(
            "open-ecology task learner seed is not registered"
        )
    environment_seed_index = _nonnegative_int(
        value.get("environment_seed_index"),
        field="open-ecology task environment_seed_index",
    )
    environment_seeds = OPEN_ECOLOGY_SEED_REGISTRY[resolved_seed_role]
    if (
        value.get("environment_seed") != environment_seed
        or environment_seed_index >= len(environment_seeds)
        or environment_seeds[environment_seed_index] != environment_seed
    ):
        raise RecurrentExperimentError(
            "open-ecology task environment seed provenance drifted"
        )
    genome_stream_seed_index = _nonnegative_int(
        value.get("genome_stream_seed_index"),
        field="open-ecology task genome_stream_seed_index",
    )
    genome_seed_role = {
        OPEN_ECOLOGY_TRAINING_SEED_ROLE: "open_ecology_genome_stream",
        OPEN_ECOLOGY_BENCHMARK_SEED_ROLE: OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        OPEN_ECOLOGY_PROOF_SEED_ROLE: OPEN_ECOLOGY_PROOF_SEED_ROLE,
    }[resolved_seed_role]
    genome_seeds = OPEN_ECOLOGY_SEED_REGISTRY[genome_seed_role]
    if (
        value.get("genome_stream_seed") != genome_stream_seed
        or genome_stream_seed_index >= len(genome_seeds)
        or genome_seeds[genome_stream_seed_index] != genome_stream_seed
        or value.get("genome_population_mode") != genome_population_mode
    ):
        raise RecurrentExperimentError("open-ecology task genome provenance drifted")
    training_phase = _open_ecology_training_phase(value.get("training_phase"))
    update_index = _nonnegative_int(
        value.get("update_index"),
        field="open-ecology task update_index",
    )
    world_index = _nonnegative_int(
        value.get("world_index"),
        field="open-ecology task world_index",
    )
    treatment = _open_ecology_treatment_from_payload(value.get("treatment"))
    density_cycle = _open_ecology_density_cycle(training_phase)
    if treatment.initial_agents != density_cycle[world_index % len(density_cycle)]:
        raise RecurrentExperimentError(
            "open-ecology task treatment density differs from its training phase"
        )
    canonical = {
        **dict(value),
        "training_phase": training_phase,
        "update_index": update_index,
        "world_index": world_index,
        "treatment": treatment.as_contract(),
    }
    if canonical != dict(value):
        raise RecurrentExperimentError("open-ecology task provenance is not canonical")
    return canonical


def _open_ecology_policy_sampling_identity(
    *,
    learner_seed: int,
    genome_stream_seed_index: int,
    genome_population_mode: str,
    training_phase: str,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
    initial_agents: int,
) -> str:
    return (
        f"{OPEN_ECOLOGY_POLICY_SAMPLING_TASK_IDENTITY_VERSION}|"
        f"registry={OPEN_ECOLOGY_CANONICAL_SHA256}|learner={learner_seed}|"
        f"genome_stream_index={genome_stream_seed_index:02d}|"
        f"genome_mode={genome_population_mode}|phase={training_phase}|"
        f"update={update_index:04d}|"
        f"world={world_index:06d}|environment_index={environment_seed_index:03d}|"
        f"initial_agents={initial_agents}"
    )


def _open_ecology_task_id(
    *,
    learner_seed: int,
    genome_stream_seed_index: int,
    genome_population_mode: str,
    training_phase: str,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
    environment_seed: int,
    initial_agents: int,
) -> str:
    return (
        f"open-ecology-learner-{learner_seed}-genome-{genome_stream_seed_index:02d}-"
        f"{genome_population_mode}-{training_phase}-update-{update_index:04d}-"
        f"world-{world_index:06d}-environment-{environment_seed_index:03d}-"
        f"seed-{environment_seed}-density-{initial_agents}"
    )


def _validate_rollout_task_genome_bindings(
    *,
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentRolloutTask],
) -> tuple[str, int | None]:
    configurations = {
        _genome_population_configuration(
            population_mode=task.genome_population_mode,
            stream_seed=task.genome_stream_seed,
        )
        for task in tasks
    }
    if len(configurations) != 1:
        raise RecurrentExperimentError(
            "rollout tasks must share one exact genome population mode and stream seed"
        )
    population_mode, stream_seed = next(iter(configurations))
    conditioning_mode = model.config.genome_conditioning_mode
    if conditioning_mode == RECURRENT_GENOME_CONDITIONING_DISABLED:
        if (
            population_mode != RecurrentGenomePopulationMode.DISABLED.value
            or stream_seed is not None
        ):
            raise RecurrentExperimentError(
                "disabled recurrent model requires disabled rollout genome tasks"
            )
    elif conditioning_mode == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1:
        if population_mode not in {
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        }:
            raise RecurrentExperimentError(
                "actor_film_v1 recurrent model requires heritable or zero_all "
                "rollout tasks"
            )
        if stream_seed is None:
            raise RecurrentExperimentError(
                "actor_film_v1 recurrent model requires a genome stream seed"
            )
    else:
        raise RecurrentExperimentError(
            "recurrent model genome conditioning mode is unsupported"
        )
    return population_mode, stream_seed


def _summary_world_seed_provenance(
    summary: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        raise RecurrentExperimentError("rollout worker summary must be a mapping")
    conditioned = "world_seed_provenance" in summary
    open_ecology = "open_ecology_task_provenance" in summary
    expected_summary_keys = set(_ROLLOUT_SUMMARY_KEYS)
    if conditioned:
        expected_summary_keys.add("world_seed_provenance")
    if open_ecology:
        expected_summary_keys.add("open_ecology_task_provenance")
    if set(summary) != expected_summary_keys:
        raise RecurrentExperimentError(
            "rollout worker summary fields do not match the exact contract"
        )
    world_id = summary.get("task_id")
    if not isinstance(world_id, str) or not world_id:
        raise RecurrentExperimentError(
            "rollout worker summary task_id must be non-empty"
        )
    environment_seed = _positive_seed(
        summary.get("environment_seed"),
        field="world summary environment_seed",
    )
    policy_sampling_seed = _policy_sampling_seed(
        summary.get("policy_sampling_seed"),
        field="world summary policy_sampling_seed",
    )
    compact_summary = summary.get("summary")
    if not isinstance(compact_summary, Mapping):
        raise RecurrentExperimentError(
            "rollout worker compact simulator summary must be a mapping"
        )
    if set(compact_summary) != {
        "run_id",
        "seed",
        "ticks_executed",
        "births",
        "deaths",
        "alive_agents",
        "peak_alive_agents",
        "extinct",
    }:
        raise RecurrentExperimentError(
            "rollout worker compact simulator summary fields drifted"
        )
    if compact_summary.get("seed") != environment_seed:
        raise RecurrentExperimentError("rollout worker simulator summary seed drifted")
    _validate_rollout_terminal_bootstrap(
        summary.get("terminal_bootstrap"),
        world_id=world_id,
        rollout_ticks=_positive_int(
            summary.get("rollout_ticks"),
            field="world summary rollout_ticks",
        ),
    )
    if open_ecology and not conditioned:
        raise RecurrentExperimentError(
            "open-ecology rollout worker provenance requires genome conditioning"
        )
    if not conditioned:
        return {
            "environment_seed": environment_seed,
            "policy_sampling_seed": policy_sampling_seed,
        }

    raw_provenance = summary.get("world_seed_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise RecurrentExperimentError(
            "conditioned rollout worker provenance must be a mapping"
        )
    observed_provenance_keys = set(raw_provenance)
    expected_provenance_keys = _CONDITIONED_WORLD_SEED_PROVENANCE_KEYS
    paired_provenance_keys = expected_provenance_keys | {"genome_world_identity"}
    fixed_batch_provenance_keys = expected_provenance_keys | {"fixed_batch_runtime"}
    paired_fixed_batch_provenance_keys = paired_provenance_keys | {
        "fixed_batch_runtime"
    }
    if observed_provenance_keys not in {
        expected_provenance_keys,
        paired_provenance_keys,
        fixed_batch_provenance_keys,
        paired_fixed_batch_provenance_keys,
    } or (
        not open_ecology
        and observed_provenance_keys
        not in {expected_provenance_keys, paired_provenance_keys}
    ):
        raise RecurrentExperimentError(
            "conditioned rollout worker provenance fields do not match the "
            "exact contract"
        )
    population_mode, stream_seed = _genome_population_configuration(
        population_mode=raw_provenance.get("genome_population_mode"),
        stream_seed=raw_provenance.get("genome_stream_seed"),
    )
    if population_mode == RecurrentGenomePopulationMode.DISABLED.value:
        raise RecurrentExperimentError(
            "conditioned rollout worker provenance requires an active population"
        )
    if (
        raw_provenance.get("environment_seed") != environment_seed
        or raw_provenance.get("policy_sampling_seed") != policy_sampling_seed
        or raw_provenance.get("genome_conditioning_mode")
        != RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
    ):
        raise RecurrentExperimentError(
            "conditioned rollout worker provenance disagrees with its summary"
        )
    probe = RecurrentRolloutBuffer()
    try:
        probe.register_world(
            world_id,
            genome_world_identity=raw_provenance.get("genome_world_identity"),
            environment_seed=environment_seed,
            policy_sampling_seed=policy_sampling_seed,
            genome_conditioning_mode=raw_provenance.get("genome_conditioning_mode"),
            genome_population_mode=population_mode,
            genome_stream_seed=stream_seed,
            genome_population_binding_sha256=raw_provenance.get(
                "genome_population_binding_sha256"
            ),
            genome_population_pre_founder_state_sha256=raw_provenance.get(
                "genome_population_pre_founder_state_sha256"
            ),
            fixed_batch_runtime=raw_provenance.get("fixed_batch_runtime"),
        )
        probe.finalize_world_genome_provenance(
            world_id,
            final_state_sha256=raw_provenance.get(
                "genome_population_final_state_sha256"
            ),
            reset_state_sha256=raw_provenance.get(
                "genome_population_reset_state_sha256"
            ),
        )
    except ValueError as error:
        raise RecurrentExperimentError(
            "conditioned rollout worker provenance is invalid"
        ) from error
    provenance = probe.world_seed_provenance[world_id]
    if provenance != dict(raw_provenance):
        raise RecurrentExperimentError(
            "conditioned rollout worker provenance is not canonical"
        )
    if open_ecology:
        fixed_batch_runtime = provenance.get("fixed_batch_runtime")
        if fixed_batch_runtime is not None and (
            not isinstance(fixed_batch_runtime, Mapping)
            or fixed_batch_runtime.get("batch_capacity")
            != OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
        ):
            raise RecurrentExperimentError(
                "open-ecology rollout worker fixed batch capacity drifted"
            )
        provenance["open_ecology_task_provenance"] = (
            _open_ecology_task_provenance_from_payload(
                summary.get("open_ecology_task_provenance"),
                seed_role=summary.get("seed_role"),
                environment_seed=environment_seed,
                genome_stream_seed=stream_seed,
                genome_population_mode=population_mode,
            )
        )
    return provenance


def _validate_rollout_terminal_bootstrap(
    raw_evidence: object,
    *,
    world_id: str,
    rollout_ticks: int,
) -> dict[str, object]:
    if not isinstance(raw_evidence, Mapping):
        raise RecurrentExperimentError(
            "rollout terminal bootstrap evidence must be a mapping"
        )
    evidence = copy.deepcopy(dict(raw_evidence))
    if set(evidence) != {
        "schema_version",
        "world_id",
        "tick",
        "boundary",
        "action_sampled",
        "action_committed",
        "action_resolved",
        "alive_agent_count",
        "target_eligible_agent_count",
        "zero_decision_alive_agent_count",
        "zero_decision_alive_agent_ids",
        "values",
        "exact_digest",
    }:
        raise RecurrentExperimentError(
            "rollout terminal bootstrap evidence fields drifted"
        )
    if (
        evidence.get("schema_version") != RECURRENT_ACTION_FREE_BOOTSTRAP_SCHEMA_VERSION
        or evidence.get("world_id") != world_id
        or evidence.get("tick") != rollout_ticks
    ):
        raise RecurrentExperimentError("rollout terminal bootstrap identity drifted")
    if evidence.get("boundary") not in {
        "exact_policy_visible_tick_start_before_action_v1",
        "terminal_before_horizon_no_bootstrap_v1",
    }:
        raise RecurrentExperimentError("rollout terminal bootstrap boundary drifted")
    if any(
        evidence.get(field) is not False
        for field in ("action_sampled", "action_committed", "action_resolved")
    ):
        raise RecurrentExperimentError(
            "rollout terminal bootstrap must remain action-free"
        )
    raw_rows = evidence.get("values")
    if not isinstance(raw_rows, list):
        raise RecurrentExperimentError(
            "rollout terminal bootstrap values must be a list"
        )
    agent_ids: list[int] = []
    eligible_ids: list[int] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise RecurrentExperimentError(
                "rollout terminal bootstrap value row must be a mapping"
            )
        row = dict(raw_row)
        if set(row) != {
            "agent_id",
            "policy_input_sha256",
            "value",
            "target_eligible",
            "genome_sha256",
            "exact_digest",
        }:
            raise RecurrentExperimentError(
                "rollout terminal bootstrap value row fields drifted"
            )
        agent_id = row.get("agent_id")
        if isinstance(agent_id, bool) or not isinstance(agent_id, int):
            raise RecurrentExperimentError(
                "rollout terminal bootstrap agent_id must be an integer"
            )
        for field in ("policy_input_sha256", "exact_digest"):
            _validate_sha256(row.get(field), field=f"terminal bootstrap {field}")
        genome_sha256 = row.get("genome_sha256")
        if genome_sha256 is not None:
            _validate_sha256(
                genome_sha256,
                field="terminal bootstrap genome_sha256",
            )
        value = row.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RecurrentExperimentError(
                "rollout terminal bootstrap value must be finite"
            )
        if type(row.get("target_eligible")) is not bool:
            raise RecurrentExperimentError(
                "rollout terminal bootstrap target_eligible must be boolean"
            )
        row_without_digest = {
            key: value for key, value in row.items() if key != "exact_digest"
        }
        if row["exact_digest"] != stable_payload_digest(row_without_digest):
            raise RecurrentExperimentError(
                "rollout terminal bootstrap value digest mismatched"
            )
        agent_ids.append(agent_id)
        if row["target_eligible"] is True:
            eligible_ids.append(agent_id)
    if agent_ids != sorted(set(agent_ids)):
        raise RecurrentExperimentError(
            "rollout terminal bootstrap value rows must be unique and sorted"
        )
    zero_decision_ids = [
        agent_id for agent_id in agent_ids if agent_id not in set(eligible_ids)
    ]
    if (
        evidence.get("alive_agent_count") != len(agent_ids)
        or evidence.get("target_eligible_agent_count") != len(eligible_ids)
        or evidence.get("zero_decision_alive_agent_count") != len(zero_decision_ids)
        or evidence.get("zero_decision_alive_agent_ids") != zero_decision_ids
    ):
        raise RecurrentExperimentError(
            "rollout terminal bootstrap agent counts drifted"
        )
    evidence_without_digest = {
        key: value for key, value in evidence.items() if key != "exact_digest"
    }
    if evidence["exact_digest"] != stable_payload_digest(evidence_without_digest):
        raise RecurrentExperimentError(
            "rollout terminal bootstrap evidence digest mismatched"
        )
    return evidence


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RecurrentExperimentError(f"{field} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise RecurrentExperimentError(
            f"{field} must be a SHA256 hex digest"
        ) from error
    return value


def _validate_rollout_worker_result(
    *,
    task: RecurrentRolloutTask,
    summary: Mapping[str, object],
    steps: Sequence[RecurrentRolloutStep],
    genome_conditioning_mode: str,
    expected_fixed_batch_contract: RecurrentFixedBatchRuntimeContract | None,
) -> dict[str, object]:
    provenance = _summary_world_seed_provenance(summary)
    exact_task_fields = {
        "task_id": task.task_id,
        "scenario": task.scenario,
        "environment_seed": task.environment_seed,
        "seed_role": task.seed_role,
        "policy_sampling_identity": task.policy_sampling_identity,
        "policy_sampling_seed": task.policy_sampling_seed,
        "policy_sampling_seed_namespace": RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
        "rollout_ticks": task.rollout_ticks,
    }
    if any(summary.get(key) != value for key, value in exact_task_fields.items()):
        raise RecurrentExperimentError(
            "rollout worker result task order, identity, or configuration drifted"
        )
    conditioned = (
        task.genome_population_mode != RecurrentGenomePopulationMode.DISABLED.value
    )
    if conditioned != ("world_seed_provenance" in summary):
        raise RecurrentExperimentError(
            "rollout worker summary conditioning shape disagrees with its task"
        )
    open_ecology = isinstance(task, OpenEcologyRolloutTask)
    if open_ecology != ("open_ecology_task_provenance" in summary):
        raise RecurrentExperimentError(
            "rollout worker open-ecology provenance shape disagrees with its task"
        )
    if (
        open_ecology
        and provenance.get("open_ecology_task_provenance")
        != task.open_ecology_task_provenance()
    ):
        raise RecurrentExperimentError(
            "rollout worker open-ecology provenance disagrees with its task"
        )
    if conditioned:
        expected_genome_world_identity = getattr(
            task,
            "phase_a_genome_world_identity",
            None,
        )
        if (
            genome_conditioning_mode != RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
            or provenance.get("genome_population_mode") != task.genome_population_mode
            or provenance.get("genome_stream_seed") != task.genome_stream_seed
            or (
                expected_genome_world_identity is not None
                and provenance.get("genome_world_identity")
                != expected_genome_world_identity
            )
        ):
            raise RecurrentExperimentError(
                "rollout worker genome provenance disagrees with its task or model"
            )
    elif genome_conditioning_mode != RECURRENT_GENOME_CONDITIONING_DISABLED:
        raise RecurrentExperimentError(
            "rollout worker omitted genome provenance for a conditioned model"
        )
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise RecurrentExperimentError(
            "rollout worker result must contain policy decision steps"
        )
    if any(not isinstance(step, RecurrentRolloutStep) for step in steps):
        raise RecurrentExperimentError(
            "rollout worker result contains an invalid rollout step"
        )
    fixed_batch_runtime = provenance.get("fixed_batch_runtime")
    if expected_fixed_batch_contract is None:
        if fixed_batch_runtime is not None:
            raise RecurrentExperimentError(
                "scalar rollout worker unexpectedly used fixed batching"
            )
    elif not open_ecology:
        raise RecurrentExperimentError(
            "legacy rollout worker cannot use open-ecology fixed batching"
        )
    elif not isinstance(fixed_batch_runtime, Mapping):
        raise RecurrentExperimentError(
            "fixed-batch rollout worker omitted its runtime provenance"
        )
    elif (
        fixed_batch_runtime.get("batch_capacity")
        != expected_fixed_batch_contract.batch_capacity
        or fixed_batch_runtime.get("contract")
        != expected_fixed_batch_contract.as_contract()
    ):
        raise RecurrentExperimentError(
            "fixed-batch rollout worker runtime disagrees with the requested contract"
        )
    for step in steps:
        if (
            step.world_id != task.task_id
            or step.environment_seed != task.environment_seed
            or step.policy_sampling_seed != task.policy_sampling_seed
        ):
            raise RecurrentExperimentError(
                "rollout worker step identity or seed provenance drifted"
            )
        step_conditioned = step.genome_values is not None
        if step_conditioned != conditioned:
            raise RecurrentExperimentError(
                "rollout worker step conditioning disagrees with its task"
            )
        if conditioned and step.genome_stream_seed != task.genome_stream_seed:
            raise RecurrentExperimentError(
                "rollout worker step genome stream seed drifted"
            )
        if expected_fixed_batch_contract is not None:
            if not isinstance(fixed_batch_runtime, Mapping):
                raise AssertionError("validated fixed-batch runtime disappeared")
            if (
                step.fixed_batch_runtime_sha256
                != fixed_batch_runtime.get("exact_digest")
                or step.fixed_batch_capacity
                != expected_fixed_batch_contract.batch_capacity
            ):
                raise RecurrentExperimentError(
                    "open-ecology rollout worker step fixed batch provenance drifted"
                )
        elif step.fixed_batch_runtime_sha256 is not None:
            raise RecurrentExperimentError(
                "rollout worker step unexpectedly used fixed batching"
            )
    return provenance


def _torch_determinism_runtime_state() -> dict[str, object]:
    return {
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(
            getattr(getattr(torch.backends, "cudnn", object()), "benchmark", False)
        ),
        "cudnn_deterministic": bool(
            getattr(
                getattr(torch.backends, "cudnn", object()),
                "deterministic",
                False,
            )
        ),
        "cudnn_allow_tf32": bool(
            getattr(getattr(torch.backends, "cudnn", object()), "allow_tf32", False)
        ),
        "cuda_matmul_allow_tf32": bool(
            getattr(
                getattr(getattr(torch.backends, "cuda", object()), "matmul", object()),
                "allow_tf32",
                False,
            )
        ),
    }


def _fixed_batch_worker_torch_runtime_contract() -> dict[str, object]:
    return {
        **_torch_determinism_runtime_state(),
        "torch_num_threads": RECURRENT_ROLLOUT_WORKER_TORCH_THREADS,
        "torch_num_interop_threads": RECURRENT_ROLLOUT_WORKER_TORCH_THREADS,
    }


def _observed_fixed_batch_worker_torch_runtime() -> dict[str, object]:
    return {
        **_torch_determinism_runtime_state(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def _apply_torch_determinism_runtime_state(
    value: Mapping[str, object] | None,
) -> None:
    boolean_keys = {
        "deterministic_algorithms_enabled",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
    }
    integer_keys = {
        "torch_num_threads",
        "torch_num_interop_threads",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != boolean_keys | integer_keys
        or any(type(value[key]) is not bool for key in boolean_keys)
        or any(
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or int(value[key]) <= 0
            for key in integer_keys
        )
    ):
        raise RecurrentExperimentError(
            "fixed-batch worker Torch runtime binding is invalid"
        )
    torch.set_num_threads(int(value["torch_num_threads"]))
    torch.set_num_interop_threads(int(value["torch_num_interop_threads"]))
    torch.use_deterministic_algorithms(
        bool(value["deterministic_algorithms_enabled"]),
        warn_only=False,
    )
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(value["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = bool(value["cudnn_deterministic"])
        torch.backends.cudnn.allow_tf32 = bool(value["cudnn_allow_tf32"])
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(value["cuda_matmul_allow_tf32"])
    if _observed_fixed_batch_worker_torch_runtime() != dict(value):
        raise RecurrentExperimentError(
            "fixed-batch worker Torch runtime binding could not be applied"
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
        expected_workspace = ":4096:8"
        observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed_workspace not in {None, expected_workspace}:
            raise RecurrentExperimentError(
                "CUBLAS_WORKSPACE_CONFIG must be unset or exactly :4096:8"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
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
    if not torch.are_deterministic_algorithms_enabled():
        raise RecurrentExperimentError(
            "Torch deterministic algorithms did not remain enabled"
        )
    warn_only_enabled = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    if callable(warn_only_enabled) and warn_only_enabled():
        raise RecurrentExperimentError(
            "Torch deterministic algorithms cannot use warn-only mode"
        )
    if hasattr(torch.backends, "cudnn") and (
        torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.allow_tf32
    ):
        raise RecurrentExperimentError(
            "cuDNN deterministic runtime flags did not match the contract"
        )
    if hasattr(torch.backends, "cuda") and torch.backends.cuda.matmul.allow_tf32:
        raise RecurrentExperimentError(
            "CUDA matrix multiplication TF32 remained enabled"
        )
    if (
        resolved.type == "cuda"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise RecurrentExperimentError(
            "CUBLAS_WORKSPACE_CONFIG did not match the deterministic contract"
        )


def _world_for_task(
    task: RecurrentRolloutTask,
    *,
    policy: object,
) -> SimulationWorld:
    world_ticks = task.rollout_ticks
    if task.scenario == RECURRENT_BROAD_SCENARIO:
        if isinstance(task, OpenEcologyRolloutTask):
            treatment = task.open_ecology_treatment
            return SimulationWorld(
                WorldConfig(
                    seed=task.environment_seed,
                    max_ticks=world_ticks,
                    initial_agents=treatment.initial_agents,
                    max_agents=treatment.max_agents,
                    signals=treatment.signals.as_signal_config(),
                ),
                policy=policy,
            )
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
        summary_seed_provenance = _summary_world_seed_provenance(summary)
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
        environment_seed = int(summary_seed_provenance["environment_seed"])
        seed_role = summary.get("seed_role")
        legacy_seed_role = (
            "train" if scenario == RECURRENT_BROAD_SCENARIO else "curriculum"
        )
        scale_seed_role = (
            "scale_train"
            if scenario == RECURRENT_BROAD_SCENARIO
            else "scale_curriculum"
        )
        scale_v2_seed_role = (
            "scale_v2_train"
            if scenario == RECURRENT_BROAD_SCENARIO
            else "scale_v2_curriculum"
        )
        open_ecology_seed_roles = (
            {
                OPEN_ECOLOGY_TRAINING_SEED_ROLE,
                OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
                OPEN_ECOLOGY_PROOF_SEED_ROLE,
            }
            if scenario == RECURRENT_BROAD_SCENARIO
            else set()
        )
        if seed_role not in {
            legacy_seed_role,
            scale_seed_role,
            scale_v2_seed_role,
            *open_ecology_seed_roles,
        }:
            raise RecurrentExperimentError(
                "world summary seed role does not match its training scenario"
            )
        if seed_role in open_ecology_seed_roles:
            seed_registry = OPEN_ECOLOGY_SEED_REGISTRY
        elif seed_role == legacy_seed_role:
            seed_registry = RECURRENT_SEED_REGISTRY
        elif seed_role == scale_seed_role:
            seed_registry = SCALE_DEVELOPMENT_SEED_REGISTRY
        else:
            seed_registry = SCALE_DEVELOPMENT_V2_SEED_REGISTRY
        if environment_seed not in seed_registry[seed_role]:
            raise RecurrentExperimentError(
                "world summary environment seed is outside its canonical training role"
            )
        open_ecology_provenance = summary_seed_provenance.get(
            "open_ecology_task_provenance"
        )
        if (seed_role in open_ecology_seed_roles) != (
            open_ecology_provenance is not None
        ):
            raise RecurrentExperimentError(
                "world summary open-ecology role and treatment provenance disagree"
            )
        policy_sampling_seed = int(summary_seed_provenance["policy_sampling_seed"])
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
        if summary_seed_provenance.keys() != _LEGACY_WORLD_SEED_PROVENANCE_KEYS:
            world_seed_provenance[world_id].update(
                {
                    key: value
                    for key, value in summary_seed_provenance.items()
                    if key
                    not in {
                        "environment_seed",
                        "policy_sampling_seed",
                    }
                }
            )
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
        step_conditioned = step.genome_values is not None
        provenance_conditioned = (
            provenance.get("genome_conditioning_mode")
            == RECURRENT_GENOME_CONDITIONING_ACTOR_FILM_V1
        )
        if step_conditioned != provenance_conditioned:
            raise RecurrentExperimentError(
                "rollout step conditioning does not match its world summary"
            )
        if step_conditioned and step.genome_stream_seed != provenance.get(
            "genome_stream_seed"
        ):
            raise RecurrentExperimentError(
                "rollout step genome stream seed does not match its world summary"
            )
        fixed_batch_runtime = provenance.get("fixed_batch_runtime")
        if fixed_batch_runtime is None:
            if step.fixed_batch_runtime_sha256 is not None:
                raise RecurrentExperimentError(
                    "rollout step fixed batch provenance has no world binding"
                )
        elif (
            not isinstance(fixed_batch_runtime, Mapping)
            or step.fixed_batch_runtime_sha256
            != fixed_batch_runtime.get("exact_digest")
            or step.fixed_batch_capacity != fixed_batch_runtime.get("batch_capacity")
        ):
            raise RecurrentExperimentError(
                "rollout step fixed batch provenance does not match world summary"
            )
    buffer_provenance = buffer.world_seed_provenance
    if set(buffer_provenance) != set(world_seed_provenance):
        raise RecurrentExperimentError(
            "rollout buffer and world summaries cover different seed provenance"
        )
    for world_id, provenance in world_seed_provenance.items():
        expected_buffer_provenance = {
            key: value
            for key, value in provenance.items()
            if key
            not in {
                "seed_role",
                "policy_sampling_identity",
                "open_ecology_task_provenance",
            }
        }
        if buffer_provenance[world_id] != expected_buffer_provenance:
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


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentExperimentError(f"{field} must be a non-negative integer")
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
    "OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT",
    "OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX",
    "OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX",
    "OPEN_ECOLOGY_BENCHMARK_SEED_PROVENANCE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_BENCHMARK_SEED_ROLE",
    "OPEN_ECOLOGY_BENCHMARK_TASK_IDENTITY_VERSION",
    "OPEN_ECOLOGY_INITIAL_AGENT_DENSITIES",
    "OPEN_ECOLOGY_MAX_AGENTS",
    "OPEN_ECOLOGY_PHASE_A",
    "OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE",
    "OPEN_ECOLOGY_PHASE_B",
    "OPEN_ECOLOGY_PHASE_B_DENSITY_CYCLE",
    "OPEN_ECOLOGY_POLICY_SAMPLING_TASK_IDENTITY_VERSION",
    "OPEN_ECOLOGY_PROOF_SEED_ROLE",
    "OPEN_ECOLOGY_PROOF_TASK_IDENTITY_VERSION",
    "OPEN_ECOLOGY_TASK_PROVENANCE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_TRAINING_SEED_ROLE",
    "OPEN_ECOLOGY_TREATMENT_SCHEMA_VERSION",
    "OPEN_ECOLOGY_TRAINING_PHASES",
    "RECURRENT_BROAD_SCENARIO",
    "RECURRENT_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION",
    "RECURRENT_EXPERIMENT_CONTRACT_VERSION",
    "RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION",
    "MAX_RECURRENT_ROLLOUT_WORKERS",
    "RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE",
    "RECURRENT_POLICY_SAMPLING_TASK_IDENTITY_VERSION",
    "RECURRENT_SCALE_POLICY_SAMPLING_TASK_IDENTITY_VERSION",
    "RECURRENT_SCALE_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION",
    "RECURRENT_SCALE_V2_COUNTERFACTUAL_COLLECTION_TASK_IDENTITY_VERSION",
    "RECURRENT_SCALE_V2_POLICY_SAMPLING_TASK_IDENTITY_VERSION",
    "RECURRENT_SCALE_V2_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION",
    "RECURRENT_ROLLOUT_WORKER_START_METHOD",
    "RECURRENT_ROLLOUT_WORKER_TORCH_THREADS",
    "RECURRENT_TRAINING_SEED_REGISTRY_CANONICAL",
    "RECURRENT_TRAINING_SEED_REGISTRY_CONTRACTS",
    "RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT",
    "RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT_V2",
    "RECURRENT_TRAINING_SEED_PROVENANCE_SCHEMA_VERSION",
    "RECURRENT_TRAINING_SCENARIOS",
    "RecurrentExperimentError",
    "RecurrentExperimentRunner",
    "RecurrentCounterfactualExperimentConfig",
    "RecurrentRolloutBatchDiagnostics",
    "RecurrentRolloutScopeDiagnostics",
    "RecurrentRolloutTask",
    "OpenEcologyBroadWorldTreatment",
    "OpenEcologyBenchmarkRolloutTask",
    "OpenEcologyLaunchDependencies",
    "OpenEcologyProofRolloutTask",
    "OpenEcologyRolloutTask",
    "OpenEcologySignalTreatment",
    "RecurrentTrainingRunResult",
    "RecurrentTrainingUpdateResult",
    "build_recurrent_counterfactual_collection_tasks",
    "build_recurrent_training_schedule",
    "build_open_ecology_benchmark_schedule",
    "build_open_ecology_training_schedule",
    "collect_recurrent_rollout_batch",
    "configure_recurrent_training_determinism",
    "derive_recurrent_policy_sampling_seed",
    "recurrent_training_run_payload",
]
