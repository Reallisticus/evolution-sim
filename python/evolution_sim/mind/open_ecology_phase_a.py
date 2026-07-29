from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import torch

from evolution_sim.io.open_ecology_git_authority import (
    OpenEcologyGitAuthorityError,
    discover_pinned_git_executable,
    run_pinned_git,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT,
    OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
    OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX,
    OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
)
from evolution_sim.mind.open_ecology_phase_a_contract import (
    OpenEcologyPhaseAError,
    require_lowercase_sha256,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    CRITIC_GENOME_CONDITIONING_FILM_V1,
    CRITIC_GENOME_CONDITIONING_NONE,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
    RECURRENT_NUMERIC_KERNEL_VERSION,
    VALUE_SHARED_TRUNK_GRADIENT_SHARED,
    VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_artifact import (
    LoadedRecurrentTrainingCrashCheckpoint,
    build_recurrent_training_crash_checkpoint,
    load_recurrent_artifact,
    load_recurrent_training_crash_checkpoint,
    save_recurrent_artifact,
    write_recurrent_training_crash_checkpoint,
)
from evolution_sim.mind.recurrent_experiment import (
    OPEN_ECOLOGY_BENCHMARK_SEED_PROVENANCE_SCHEMA_VERSION,
    OPEN_ECOLOGY_MAX_AGENTS,
    OPEN_ECOLOGY_PHASE_A,
    RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION,
    OPEN_ECOLOGY_TRAINING_SEED_ROLE,
    OpenEcologyBroadWorldTreatment,
    OpenEcologyRolloutTask,
    RecurrentExperimentRunner,
    RecurrentRolloutTask,
    RecurrentTrainingUpdateResult,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationMode,
)
from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig
from evolution_sim.mind.recurrent_rollout import (
    OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
    RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS,
    RecurrentFixedBatchRuntimeContract,
    derive_recurrent_policy_sampling_seed,
)
from evolution_sim.mind.recurrent_scale_campaign import source_file_hash_manifest

if TYPE_CHECKING:
    from evolution_sim.mind.open_ecology_selection import (
        OpenEcologySelectionRequest,
    )


OPEN_ECOLOGY_PHASE_A_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_campaign_v5"
OPEN_ECOLOGY_PHASE_A_RUN_CONTRACT_VERSION = (
    "mind_v3_open_ecology_phase_a_run_contract_v3"
)
OPEN_ECOLOGY_PHASE_A_TASK_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_rollout_task_v2"
)
OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_runtime_v2"
OPEN_ECOLOGY_PHASE_A_THROUGHPUT_GATE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_throughput_gate_v4"
)
OPEN_ECOLOGY_PHASE_A_RESOURCE_PROJECTION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_resource_projection_v4"
)
OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_resource_envelope_v1"
)
OPEN_ECOLOGY_PHASE_A_MAXIMUM_WALL_SECONDS = 7 * 24 * 60 * 60
OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_launch_authorization_v3"
)
OPEN_ECOLOGY_PHASE_A_SELECTION_AUTHORIZATION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_selection_authorization_v2"
)
OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_update_v2"
OPEN_ECOLOGY_PHASE_A_COMMIT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_evidence_commit_v1"
)
OPEN_ECOLOGY_PHASE_A_TERMINAL_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_training_terminal_v3"
)
OPEN_ECOLOGY_PHASE_A_LEARNER_EVIDENCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_learner_selection_evidence_v4"
)
OPEN_ECOLOGY_PHASE_A_PAIRED_RETURN_AGGREGATION = (
    "four_tapes_within_environment_then_equal_weight_32_environment_median"
)
OPEN_ECOLOGY_PHASE_A_SELECTION_TARGET_CONTRACT = (
    "discounted_gamma_0.997_exact_tick_t_action_free_alive_value_"
    "bootstrap_passive_terminal_reward_at_gamma_boundary_death_zero_v3"
)
OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_terminal_selection_authority_v2"
)
OPEN_ECOLOGY_PHASE_A_SELECTION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_cell_selection_v2"
)
OPEN_ECOLOGY_PHASE_A_AUTHORITATIVE_SELECTION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_authoritative_cell_selection_v2"
)
OPEN_ECOLOGY_PHASE_A_POLICY_SAMPLING_NAMESPACE = (
    "evolution-sim|mind-v3-open-ecology|phase-a|policy-sampling-v2"
)
OPEN_ECOLOGY_PHASE_A_GENOME_WORLD_IDENTITY_NAMESPACE = (
    "evolution-sim|mind-v3-open-ecology|phase-a|genome-world-identity-v1"
)
OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH = (
    "docs/research/open-ecology-campaign-preregistration-v1.md"
)
OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256 = (
    "38a4216494cae162661bc575f6beff437898498e8ce140691ab0ece2ec2f1721"
)
OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SCHEMA_VERSION = (
    "mind_v3_open_ecology_launch_authority_amendment_v2"
)
OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH = (
    "docs/research/open-ecology-launch-authority-amendment-v2.md"
)
OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256 = (
    "80e5185ffedf2edd4f4850c7b475fabbaff9c192fcd4b70228477f0f5e0ebfd3"
)
OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_launch_authority_amendment_v3"
)
OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH = (
    "docs/research/open-ecology-launch-authority-amendment-v3.md"
)
OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256 = (
    "f46b34ab55a930d5b558d2b72b5279178c37acf274422986e26b437d34247813"
)

OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT = 8
OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE = 16
OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS = 128
OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS = 64
OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT = 4
OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT = 32
OPEN_ECOLOGY_PHASE_A_SELECTION_TICKS = 512
OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT = 4
OPEN_ECOLOGY_PHASE_A_CAUSAL_STATES_PER_WORLD = 64
OPEN_ECOLOGY_PHASE_A_CAUSAL_STATE_COUNT = (
    OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
    * OPEN_ECOLOGY_PHASE_A_CAUSAL_STATES_PER_WORLD
)
OPEN_ECOLOGY_PHASE_A_TOTAL_TRAINING_WORLDS = (
    4
    * OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
    * OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
    * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
)
OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS = (1, 2, 4, 8, 16)
OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS = 3
OPEN_ECOLOGY_PHASE_A_BENCHMARK_SCHEMA_VERSION = (
    "mind_recurrent_end_to_end_pipeline_benchmark_v2"
)
OPEN_ECOLOGY_PHASE_A_ALL_TRAINING_WORLDS = 6_144
OPEN_ECOLOGY_PHASE_A_MATRIX_TRAINING_WORLDS = 2_048
OPEN_ECOLOGY_PHASE_B_MATRIX_TRAINING_WORLDS = 4_096
OPEN_ECOLOGY_PHASE_B_ROLLOUT_TICKS = 256
OPEN_ECOLOGY_PHASE_A_TRAINED_SELECTION_EXECUTIONS = 2_560
OPEN_ECOLOGY_PHASE_A_INITIALIZED_BASELINE_SELECTION_EXECUTIONS = 2_048
OPEN_ECOLOGY_PHASE_A_PRIMARY_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_A_TRAINED_SELECTION_EXECUTIONS
    + OPEN_ECOLOGY_PHASE_A_INITIALIZED_BASELINE_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_PHASE_A_REPLAY_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_A_PRIMARY_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_PHASE_A_PHYSICAL_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_A_PRIMARY_SELECTION_EXECUTIONS
    + OPEN_ECOLOGY_PHASE_A_REPLAY_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_PHASE_B_TRAINED_SELECTION_EXECUTIONS = 1_280
OPEN_ECOLOGY_PHASE_B_INITIALIZED_BASELINE_SELECTION_EXECUTIONS = 1_024
OPEN_ECOLOGY_PHASE_B_PRIMARY_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_B_TRAINED_SELECTION_EXECUTIONS
    + OPEN_ECOLOGY_PHASE_B_INITIALIZED_BASELINE_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_PHASE_B_REPLAY_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_B_PRIMARY_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_PHASE_B_PHYSICAL_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_B_PRIMARY_SELECTION_EXECUTIONS
    + OPEN_ECOLOGY_PHASE_B_REPLAY_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_RESOURCE_PROJECTION_SAFETY_FACTOR = 1.20
OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES = (
    "readiness_dependency_01",
    "readiness_dependency_02",
    "readiness_dependency_03",
    "readiness_dependency_04",
    "readiness_dependency_09",
    "readiness_dependency_10",
)
OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_PROOF_PRODUCERS_AVAILABLE = True
OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_BLOCKERS = ("launch_evidence_index_required",)

OPEN_ECOLOGY_PHASE_A_CELL_ORDER = ("A0", "A1", "A2", "A3")
OPEN_ECOLOGY_PHASE_A_CELLS: Mapping[str, tuple[str, str]] = {
    "A0": (
        CRITIC_GENOME_CONDITIONING_NONE,
        VALUE_SHARED_TRUNK_GRADIENT_SHARED,
    ),
    "A1": (
        CRITIC_GENOME_CONDITIONING_NONE,
        VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
    ),
    "A2": (
        CRITIC_GENOME_CONDITIONING_FILM_V1,
        VALUE_SHARED_TRUNK_GRADIENT_SHARED,
    ),
    "A3": (
        CRITIC_GENOME_CONDITIONING_FILM_V1,
        VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
    ),
}

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_DIRECTORY_RE = re.compile(r"^phase-a-(a[0-3])-learner-([0-3])-([1-9][0-9]*)$")
_UPDATE_DIRECTORY_RE = re.compile(r"^update-([0-9]{4})$")
_UPDATE_STAGING_DIRECTORY_RE = re.compile(
    r"^\.update-([0-9]{4})\.pending-([1-9][0-9]*)-([0-9a-f]{32})$"
)
_TERMINAL_STAGING_DIRECTORY_RE = re.compile(
    r"^\.terminal\.pending-([1-9][0-9]*)-([0-9a-f]{32})$"
)
_COMMON_PARAMETER_EXCLUSIONS = ("critic_genome_",)
_MAX_SELECTION_AUTHORIZATION_BYTES = 16 * 1024 * 1024


def _open_ecology_staged_launch_authority() -> dict[str, object]:
    """Return the prospective, stage-specific evidence boundary."""

    return {
        "schema_version": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION,
        "amendment_adopted_before_scientific_outcome_evidence": True,
        "scientific_protocol_changed": False,
        "stages": [
            {
                "stage_id": "phase_a_training",
                "required_prior_authorizations": [],
                "required_dependency_evidence": [
                    {
                        "dependency_id": "readiness_dependency_01",
                        "evidence_kinds": ["cross_surface_and_self_echo"],
                    },
                    {
                        "dependency_id": "readiness_dependency_02",
                        "evidence_kinds": ["runtime_genome_and_action_source"],
                    },
                    {
                        "dependency_id": "readiness_dependency_03",
                        "evidence_kinds": [
                            "critic_gradient_and_density_schedule",
                        ],
                    },
                    {
                        "dependency_id": "readiness_dependency_04",
                        "evidence_kinds": [
                            "fixed_batch_equivalence_and_speed",
                        ],
                    },
                    {
                        "dependency_id": "readiness_dependency_09",
                        "evidence_kinds": [
                            "preregistration_roundtrip_fail_closed",
                        ],
                    },
                    {
                        "dependency_id": "readiness_dependency_10",
                        "evidence_kinds": [
                            "exact_sha_phase_a_training_and_torch_ci",
                        ],
                    },
                ],
                "required_operational_evidence": [
                    "phase_a_training_throughput",
                    "campaign_storage_capacity",
                    "output_lock_contention",
                ],
                "authorization_scope": "phase_a_training_only",
            },
            {
                "stage_id": "phase_a_selection",
                "required_prior_authorizations": [
                    "phase_a_training_terminal_matrix",
                ],
                "required_dependency_evidence": [
                    {
                        "dependency_id": "readiness_dependency_08",
                        "evidence_kinds": [
                            "causal_evaluator_rejection_battery",
                            "capture_noninterference_reexecution",
                        ],
                    },
                ],
                "required_operational_evidence": [
                    "phase_a_selection_throughput",
                ],
                "authorization_scope": "phase_a_selection_only",
            },
            {
                "stage_id": "phase_b_training",
                "required_prior_authorizations": [
                    "phase_a_authoritative_cell_selection",
                ],
                "required_dependency_evidence": [],
                "required_operational_evidence": [
                    "phase_b_mixed_density_training_throughput",
                ],
                "authorization_scope": "phase_b_training_only",
            },
            {
                "stage_id": "phase_c_density_selection",
                "required_prior_authorizations": [
                    "phase_b_terminal_selection",
                ],
                "required_dependency_evidence": [],
                "required_operational_evidence": [
                    "phase_c_density_selection_throughput",
                ],
                "authorization_scope": "phase_c_density_selection_only",
            },
            {
                "stage_id": "phase_d_persistent_ecology",
                "required_prior_authorizations": [
                    "phase_b_terminal_selection",
                    "phase_c_density_selection",
                ],
                "required_dependency_evidence": [
                    {
                        "dependency_id": "readiness_dependency_05",
                        "evidence_kinds": ["persistent_island_50000"],
                    },
                    {
                        "dependency_id": "readiness_dependency_06",
                        "evidence_kinds": [
                            "checkpoint_continuation_equivalence",
                        ],
                    },
                    {
                        "dependency_id": "readiness_dependency_07",
                        "evidence_kinds": ["bounded_writer_event_coverage"],
                    },
                    {
                        "dependency_id": "readiness_dependency_10",
                        "evidence_kinds": [
                            "phase_d_host_equivalence_or_homogeneous_contract",
                        ],
                    },
                ],
                "required_operational_evidence": [
                    "phase_d_full_throughput",
                    "campaign_storage_capacity",
                    "output_lock_contention",
                    "immutable_drive_uploader",
                    "terminal_aggregate_validator",
                ],
                "authorization_scope": "phase_d_persistent_ecology_only",
            },
        ],
        "local_pruning_supported": False,
        "local_deletion_authorized": False,
    }


def configure_open_ecology_phase_a_determinism() -> None:
    """Configure the exact float32/CUDA contract before any campaign work."""

    required_workspace = ":4096:8"
    observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed_workspace not in {None, required_workspace}:
        raise OpenEcologyPhaseAError(
            "preexisting CUBLAS_WORKSPACE_CONFIG contradicts Phase A"
        )
    if observed_workspace is None and torch.cuda.is_initialized():
        raise OpenEcologyPhaseAError(
            "CUDA initialized before Phase A CUBLAS determinism was configured"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_workspace
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseAOpenEcologyRolloutTask(OpenEcologyRolloutTask):
    """Phase- and cell-domain-separated task on the existing broad-world path."""

    phase_a_cell: str
    phase_a_learner_index: int
    phase_a_genome_world_identity: str
    phase_a_task_schema_version: str = OPEN_ECOLOGY_PHASE_A_TASK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # OpenEcologyRolloutTask's task identity predates the sealed requirement
        # to domain-separate policy draws by phase and arm. Reuse only the base
        # task validation, then enforce the stricter Phase A identity here.
        RecurrentRolloutTask.__post_init__(self)
        if self.phase_a_task_schema_version != OPEN_ECOLOGY_PHASE_A_TASK_SCHEMA_VERSION:
            raise OpenEcologyPhaseAError("Phase A rollout task schema drifted")
        cell_id = _cell_id(self.phase_a_cell)
        learner_index = _index(
            self.phase_a_learner_index,
            field="phase_a_learner_index",
            upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
        )
        learner_seed = _phase_a_learner_seeds()[learner_index]
        if (
            self.open_ecology_learner_seed != learner_seed
            or self.scenario != "broad"
            or self.seed_role != OPEN_ECOLOGY_TRAINING_SEED_ROLE
            or self.open_ecology_training_phase != OPEN_ECOLOGY_PHASE_A
            or not isinstance(
                self.open_ecology_treatment,
                OpenEcologyBroadWorldTreatment,
            )
            or self.open_ecology_treatment.initial_agents
            != OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS
            or self.open_ecology_treatment.max_agents != OPEN_ECOLOGY_MAX_AGENTS
        ):
            raise OpenEcologyPhaseAError(
                "Phase A task learner, broad-world role, or fixed density drifted"
            )
        if self.genome_population_mode != RecurrentGenomePopulationMode.HERITABLE.value:
            raise OpenEcologyPhaseAError(
                "Phase A requires heritable controller-genome populations"
            )
        train_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_TRAINING_SEED_ROLE]
        environment_index = _index(
            self.open_ecology_environment_seed_index,
            field="open_ecology_environment_seed_index",
            upper=OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
            * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
        )
        if (
            self.environment_seed != train_seeds[environment_index]
            or self.open_ecology_world_index != environment_index
            or self.open_ecology_update_index
            != environment_index // OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
        ):
            raise OpenEcologyPhaseAError(
                "Phase A task environment ordering or update partition drifted"
            )
        genome_seeds = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"]
        if (
            self.open_ecology_genome_stream_seed_index != learner_index
            or self.genome_stream_seed != genome_seeds[learner_index]
        ):
            raise OpenEcologyPhaseAError(
                "Phase A learner/genome-stream pairing drifted"
            )
        expected_identity = _phase_a_policy_sampling_identity(
            learner_index=learner_index,
            learner_seed=learner_seed,
            genome_stream_seed_index=learner_index,
            update_index=self.open_ecology_update_index,
            world_index=self.open_ecology_world_index,
            environment_seed_index=environment_index,
        )
        if self.policy_sampling_identity != expected_identity:
            raise OpenEcologyPhaseAError(
                "Phase A policy sampling identity is not paired-world canonical"
            )
        expected_genome_world_identity = _phase_a_genome_world_identity(
            learner_index=learner_index,
            learner_seed=learner_seed,
            genome_stream_seed_index=learner_index,
            update_index=self.open_ecology_update_index,
            world_index=self.open_ecology_world_index,
            environment_seed_index=environment_index,
            environment_seed=self.environment_seed,
        )
        if self.phase_a_genome_world_identity != expected_genome_world_identity:
            raise OpenEcologyPhaseAError(
                "Phase A genome world identity is not paired-world canonical"
            )
        if self.task_id != _phase_a_task_id(
            cell_id=cell_id,
            learner_index=learner_index,
            learner_seed=learner_seed,
            update_index=self.open_ecology_update_index,
            world_index=self.open_ecology_world_index,
            environment_seed_index=environment_index,
            environment_seed=self.environment_seed,
        ):
            raise OpenEcologyPhaseAError("Phase A task_id is not canonical")


@dataclass(frozen=True, slots=True)
class PhaseAEvidencePrefix:
    completed_updates: int
    commit_digests: tuple[str, ...]
    terminal_checkpoint: LoadedRecurrentTrainingCrashCheckpoint | None
    initial_model_state_sha256: str | None = None
    terminal_model_state_sha256: str | None = None
    cumulative_accepted_ppo_minibatches: int = 0
    cumulative_post_step_kl_rejected_steps: int = 0


@dataclass(frozen=True, slots=True)
class _PhaseAEvidenceNode:
    commit_digest: str
    checkpoint: LoadedRecurrentTrainingCrashCheckpoint
    model_state_sha256_after_update: str
    accepted_ppo_minibatches: int
    post_step_kl_rejected_steps: int


def build_open_ecology_phase_a_runtime_contract(
    *,
    device: torch.device | str,
    rollout_workers: int,
) -> dict[str, object]:
    resolved_device = torch.device(device)
    workers = _positive_int(rollout_workers, field="rollout_workers")
    if resolved_device.type != "cuda":
        raise OpenEcologyPhaseAError(
            "the authoritative Phase A runtime contract requires CUDA"
        )
    if not torch.cuda.is_available():
        raise OpenEcologyPhaseAError("CUDA is unavailable on the Phase A host")
    device_index = resolved_device.index if resolved_device.index is not None else 0
    properties = torch.cuda.get_device_properties(device_index)
    runtime: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if hasattr(torch.backends, "cudnn")
                else None
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "device": {
            "type": resolved_device.type,
            "name": properties.name,
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory_bytes": int(properties.total_memory),
        },
        "determinism": {
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "rollout_workers": workers,
        "ordered_worker_merge_required": True,
        "fixed_batch_runtime_contract": (
            RecurrentFixedBatchRuntimeContract.open_ecology().as_contract()
        ),
    }
    runtime["exact_digest"] = stable_payload_digest(runtime)
    validate_open_ecology_phase_a_runtime_contract(runtime)
    return runtime


def validate_open_ecology_phase_a_runtime_contract(
    runtime: Mapping[str, object],
) -> None:
    _require_exact_keys(
        runtime,
        {
            "schema_version",
            "python",
            "platform",
            "torch",
            "device",
            "determinism",
            "rollout_workers",
            "ordered_worker_merge_required",
            "fixed_batch_runtime_contract",
            "exact_digest",
        },
        field="Phase A runtime contract",
    )
    if runtime.get("schema_version") != OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION:
        raise OpenEcologyPhaseAError("Phase A runtime schema drifted")
    _validate_signed_payload(runtime, field="Phase A runtime contract")
    device = _mapping(runtime.get("device"), field="runtime.device")
    if device.get("type") != "cuda":
        raise OpenEcologyPhaseAError("Phase A runtime must be CUDA")
    _positive_int(runtime.get("rollout_workers"), field="runtime.rollout_workers")
    if runtime.get("ordered_worker_merge_required") is not True:
        raise OpenEcologyPhaseAError("Phase A requires ordered rollout-worker merge")
    expected_fixed_batch_contract = (
        RecurrentFixedBatchRuntimeContract.open_ecology().as_contract()
    )
    if runtime.get("fixed_batch_runtime_contract") != expected_fixed_batch_contract:
        raise OpenEcologyPhaseAError(
            "Phase A fixed-batch runtime contract drifted"
        )
    determinism = _mapping(runtime.get("determinism"), field="runtime.determinism")
    expected_determinism = {
        "deterministic_algorithms_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "default_dtype": "torch.float32",
    }
    if dict(determinism) != expected_determinism:
        raise OpenEcologyPhaseAError("Phase A deterministic runtime contract drifted")
    torch_contract = _mapping(runtime.get("torch"), field="runtime.torch")
    if torch_contract.get("float32_matmul_precision") != "highest":
        raise OpenEcologyPhaseAError("Phase A float32 matmul precision drifted")


def build_open_ecology_phase_a_resource_envelope(
    *,
    source_commit: str,
) -> dict[str, object]:
    """Build the only resource envelope accepted by the sealed protocol."""

    envelope: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION,
        "source_commit": _git_sha(source_commit),
        "maximum_wall_seconds": OPEN_ECOLOGY_PHASE_A_MAXIMUM_WALL_SECONDS,
        "evidence_sha256": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
    }
    envelope["exact_digest"] = stable_payload_digest(envelope)
    return envelope


def validate_open_ecology_phase_a_resource_envelope(
    resource_envelope: Mapping[str, object],
    *,
    expected_source_commit: str,
) -> None:
    """Reject post-hoc budgets or evidence detached from the sealed protocol."""

    _require_exact_keys(
        resource_envelope,
        {
            "schema_version",
            "source_commit",
            "maximum_wall_seconds",
            "evidence_sha256",
            "exact_digest",
        },
        field="Phase A resource envelope",
    )
    _validate_signed_payload(
        resource_envelope,
        field="Phase A resource envelope",
    )
    maximum_wall_seconds = _positive_int(
        resource_envelope.get("maximum_wall_seconds"),
        field="resource_envelope.maximum_wall_seconds",
    )
    evidence_sha256 = _sha256(
        resource_envelope.get("evidence_sha256"),
        field="resource_envelope.evidence_sha256",
    )
    expected = build_open_ecology_phase_a_resource_envelope(
        source_commit=expected_source_commit,
    )
    if (
        resource_envelope.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION
        or resource_envelope.get("source_commit") != expected["source_commit"]
        or maximum_wall_seconds != OPEN_ECOLOGY_PHASE_A_MAXIMUM_WALL_SECONDS
        or evidence_sha256 != OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256
        or dict(resource_envelope) != expected
    ):
        raise OpenEcologyPhaseAError(
            "Phase A resource envelope differs from the prospectively sealed "
            "seven-day protocol"
        )


def build_open_ecology_phase_a_throughput_gate(
    *,
    source_commit: str,
    heritable_report: Mapping[str, object],
    zero_all_report: Mapping[str, object],
    resource_envelope: Mapping[str, object],
) -> dict[str, object]:
    """Bind the two exact-source pipeline benchmarks to one frozen topology."""

    commit = _git_sha(source_commit)
    reports = {
        RecurrentGenomePopulationMode.HERITABLE.value: _json_clone(
            heritable_report,
            field="heritable_report",
        ),
        RecurrentGenomePopulationMode.ZERO_ALL.value: _json_clone(
            zero_all_report,
            field="zero_all_report",
        ),
    }
    elapsed_by_mode = {
        mode: _validate_phase_a_pipeline_benchmark(
            report,
            expected_source_commit=commit,
            expected_population_mode=mode,
        )
        for mode, report in reports.items()
    }
    selected_workers = min(
        OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS,
        key=lambda workers: (
            max(
                elapsed_by_mode[mode][workers]
                for mode in (
                    RecurrentGenomePopulationMode.HERITABLE.value,
                    RecurrentGenomePopulationMode.ZERO_ALL.value,
                )
            ),
            workers,
        ),
    )
    projection = _build_phase_a_resource_projection(
        source_commit=commit,
        elapsed_by_mode=elapsed_by_mode,
        selected_workers=selected_workers,
        resource_envelope=resource_envelope,
    )
    gate: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_THROUGHPUT_GATE_SCHEMA_VERSION,
        "source_commit": commit,
        "benchmark_contract": {
            "worker_counts": list(OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS),
            "repeats": OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS,
            "updates": 1,
            "worlds_per_update": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
            "rollout_ticks": OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS,
            "environment_seed_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "environment_seed_indices": list(
                range(OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT)
            ),
            "model_initialization_seed_index": (
                OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
            ),
            "genome_stream_seed_index": (
                OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ),
            "scientific_environment_seed_roles_accessed": [],
            "fixed_batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            "fixed_batch_execution_buckets": list(
                RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
            ),
            "conditioning_modes": [
                RecurrentGenomePopulationMode.HERITABLE.value,
                RecurrentGenomePopulationMode.ZERO_ALL.value,
            ],
            "equivalence_required": (
                "exact_model_state_and_semantic_evidence_vs_one_worker"
            ),
            "topology_selection_rule": (
                "minimize_worst_mode_median_elapsed_ns_then_smaller_worker_count"
            ),
        },
        "reports": reports,
        "report_digests": {
            mode: stable_payload_digest(report) for mode, report in reports.items()
        },
        "median_elapsed_ns_by_mode_and_workers": {
            mode: {
                str(workers): elapsed_by_mode[mode][workers]
                for workers in OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS
            }
            for mode in (
                RecurrentGenomePopulationMode.HERITABLE.value,
                RecurrentGenomePopulationMode.ZERO_ALL.value,
            )
        },
        "selected_rollout_workers": selected_workers,
        "resource_projection": projection,
        "training_topology_gate_passed": True,
        "selection_resource_gate_passed": False,
        "gate_passed": False,
    }
    gate["exact_digest"] = stable_payload_digest(gate)
    validate_open_ecology_phase_a_throughput_gate(gate)
    return gate


def validate_open_ecology_phase_a_throughput_gate(
    gate: Mapping[str, object],
) -> None:
    _require_exact_keys(
        gate,
        {
            "schema_version",
            "source_commit",
            "benchmark_contract",
            "reports",
            "report_digests",
            "median_elapsed_ns_by_mode_and_workers",
            "selected_rollout_workers",
            "resource_projection",
            "training_topology_gate_passed",
            "selection_resource_gate_passed",
            "gate_passed",
            "exact_digest",
        },
        field="Phase A throughput gate",
    )
    _validate_signed_payload(gate, field="Phase A throughput gate")
    if (
        gate.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_THROUGHPUT_GATE_SCHEMA_VERSION
    ):
        raise OpenEcologyPhaseAError("Phase A throughput gate schema drifted")
    if (
        gate.get("training_topology_gate_passed") is not True
        or gate.get("selection_resource_gate_passed") is not False
        or gate.get("gate_passed") is not False
    ):
        raise OpenEcologyPhaseAError(
            "Phase A preliminary throughput gate status drifted"
        )
    commit = _git_sha(gate.get("source_commit"))
    benchmark_contract = _mapping(
        gate.get("benchmark_contract"),
        field="throughput_gate.benchmark_contract",
    )
    expected_benchmark_contract = {
        "worker_counts": list(OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS),
        "repeats": OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS,
        "updates": 1,
        "worlds_per_update": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
        "rollout_ticks": OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS,
        "environment_seed_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        "environment_seed_indices": list(
            range(OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT)
        ),
        "model_initialization_seed_index": (OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX),
        "genome_stream_seed_index": (OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX),
        "scientific_environment_seed_roles_accessed": [],
        "fixed_batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        "fixed_batch_execution_buckets": list(
            RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
        ),
        "conditioning_modes": [
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        ],
        "equivalence_required": (
            "exact_model_state_and_semantic_evidence_vs_one_worker"
        ),
        "topology_selection_rule": (
            "minimize_worst_mode_median_elapsed_ns_then_smaller_worker_count"
        ),
    }
    if dict(benchmark_contract) != expected_benchmark_contract:
        raise OpenEcologyPhaseAError("Phase A throughput benchmark contract drifted")
    reports = _mapping(gate.get("reports"), field="throughput_gate.reports")
    _require_exact_keys(
        reports,
        {
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        },
        field="throughput_gate.reports",
    )
    elapsed_by_mode: dict[str, dict[int, int]] = {}
    for mode in (
        RecurrentGenomePopulationMode.HERITABLE.value,
        RecurrentGenomePopulationMode.ZERO_ALL.value,
    ):
        elapsed_by_mode[mode] = _validate_phase_a_pipeline_benchmark(
            _mapping(reports.get(mode), field=f"throughput_gate.reports.{mode}"),
            expected_source_commit=commit,
            expected_population_mode=mode,
        )
    expected_report_digests = {
        mode: stable_payload_digest(_mapping(reports.get(mode), field=mode))
        for mode in (
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        )
    }
    if gate.get("report_digests") != expected_report_digests:
        raise OpenEcologyPhaseAError("Phase A benchmark report digest drifted")
    expected_elapsed = {
        mode: {
            str(workers): elapsed_by_mode[mode][workers]
            for workers in OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS
        }
        for mode in (
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        )
    }
    if gate.get("median_elapsed_ns_by_mode_and_workers") != expected_elapsed:
        raise OpenEcologyPhaseAError("Phase A benchmark timing summary drifted")
    selected_workers = min(
        OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS,
        key=lambda workers: (
            max(
                elapsed_by_mode[mode][workers]
                for mode in (
                    RecurrentGenomePopulationMode.HERITABLE.value,
                    RecurrentGenomePopulationMode.ZERO_ALL.value,
                )
            ),
            workers,
        ),
    )
    if gate.get("selected_rollout_workers") != selected_workers:
        raise OpenEcologyPhaseAError(
            "Phase A did not freeze the fastest eligible worker topology"
        )
    expected_projection = _build_phase_a_resource_projection(
        source_commit=commit,
        elapsed_by_mode=elapsed_by_mode,
        selected_workers=selected_workers,
        resource_envelope=_mapping(
            _mapping(
                gate.get("resource_projection"),
                field="throughput_gate.resource_projection",
            ).get("resource_envelope"),
            field="throughput_gate.resource_projection.resource_envelope",
        ),
    )
    if gate.get("resource_projection") != expected_projection:
        raise OpenEcologyPhaseAError("Phase A resource projection drifted")


def open_ecology_phase_a_launch_readiness(
    *,
    preregistration: Mapping[str, object] | None = None,
    evidence_index: Mapping[str, object] | None = None,
    evidence_index_path: str | Path | None = None,
) -> dict[str, object]:
    """Report authority-code readiness and exact outstanding evidence."""

    from evolution_sim.mind.open_ecology_phase_a_readiness import (
        launch_readiness,
    )

    return launch_readiness(
        preregistration=preregistration,
        evidence_index=evidence_index,
        evidence_index_path=evidence_index_path,
    )


def build_open_ecology_phase_a_evidence_index(
    preregistration: Mapping[str, object],
    *,
    evidence_root: str | Path,
    dependency_reports: Mapping[str, Mapping[str, str | Path]],
    operational_reports: Mapping[str, str | Path],
) -> dict[str, object]:
    """Index concrete dependency reports without manufacturing their facts."""

    from evolution_sim.mind.open_ecology_phase_a_readiness import (
        build_evidence_index,
    )

    return build_evidence_index(
        preregistration,
        evidence_root=evidence_root,
        dependency_reports=dependency_reports,
        operational_reports=operational_reports,
    )


def build_open_ecology_phase_a_launch_authorization(
    preregistration: Mapping[str, object],
    *,
    evidence_index: Mapping[str, object],
    evidence_index_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, object]:
    """Assemble launch authority from all semantically valid exact-source reports."""

    from evolution_sim.mind.open_ecology_phase_a_readiness import (
        build_launch_authorization,
    )

    return build_launch_authorization(
        preregistration,
        evidence_index=evidence_index,
        evidence_index_path=evidence_index_path,
        authorization_path=authorization_path,
    )


def validate_open_ecology_phase_a_launch_authorization(
    authorization: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    authorization_path: str | Path,
) -> None:
    """Verify every sealed readiness proof and its concrete evidence bytes."""

    from evolution_sim.mind.open_ecology_phase_a_readiness import (
        validate_launch_authorization,
    )

    validate_launch_authorization(
        authorization,
        preregistration=preregistration,
        authorization_path=authorization_path,
    )


def build_open_ecology_phase_a_preregistration(
    *,
    source_commit: str,
    source_manifest_sha256: str,
    archive_tool_authority_sha256: str,
    runtime_contract: Mapping[str, object],
    throughput_gate: Mapping[str, object],
) -> dict[str, object]:
    commit = _git_sha(source_commit)
    manifest_sha256 = _sha256(
        source_manifest_sha256,
        field="source_manifest_sha256",
    )
    archive_authority_sha256 = _sha256(
        archive_tool_authority_sha256,
        field="archive_tool_authority_sha256",
    )
    validate_open_ecology_phase_a_runtime_contract(runtime_contract)
    validate_open_ecology_phase_a_throughput_gate(throughput_gate)
    runtime = _json_clone(runtime_contract, field="runtime_contract")
    throughput = _json_clone(throughput_gate, field="throughput_gate")
    if throughput.get("source_commit") != commit or runtime.get(
        "rollout_workers"
    ) != throughput.get("selected_rollout_workers"):
        raise OpenEcologyPhaseAError(
            "Phase A source or runtime is detached from the throughput gate"
        )
    train_seeds = list(
        OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_TRAINING_SEED_ROLE][
            : OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
        ]
    )
    selection_seeds = list(
        OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"][
            :OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
        ]
    )
    learners = list(_phase_a_learner_seeds())
    run_matrix: list[dict[str, object]] = []
    common_parameter_digests: dict[int, set[str]] = {
        index: set() for index in range(len(learners))
    }
    for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER:
        critic, gradient = OPEN_ECOLOGY_PHASE_A_CELLS[cell_id]
        for learner_index, learner_seed in enumerate(learners):
            model_config, _ppo_config, schedule = build_phase_a_run_components(
                cell_id=cell_id,
                learner_index=learner_index,
            )
            model = PublicRecurrentActorCritic(
                model_config,
                initialization_seed=learner_seed,
            ).to(device="cpu", dtype=torch.float32)
            common_digest = _common_parameter_sha256(model)
            common_parameter_digests[learner_index].add(common_digest)
            run_matrix.append(
                {
                    "run_id": phase_a_run_id(
                        cell_id=cell_id,
                        learner_index=learner_index,
                    ),
                    "cell_id": cell_id,
                    "learner_index": learner_index,
                    "learner_seed": learner_seed,
                    "genome_stream_seed_index": learner_index,
                    "genome_stream_seed": OPEN_ECOLOGY_SEED_REGISTRY[
                        "open_ecology_genome_stream"
                    ][learner_index],
                    "critic_genome_conditioning": critic,
                    "value_shared_trunk_gradient": gradient,
                    "initial_common_parameter_sha256": common_digest,
                    "initial_full_model_sha256": recurrent_model_state_sha256(model),
                    "schedule_sha256": _schedule_sha256(schedule),
                }
            )
    if any(len(values) != 1 for values in common_parameter_digests.values()):
        raise OpenEcologyPhaseAError(
            "Phase A cell initialization changed actor/backbone tensors"
        )

    preregistration: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_SCHEMA_VERSION,
        "campaign": "open_ecology_phase_a_critic_gradient_ablation",
        "claim_boundary": {
            "development_only": True,
            "phase_b_authorized_by_this_record": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
            "society_claim_authorized": False,
        },
        "sealed_document": {
            "base_protocol": {
                "path": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH,
                "file_sha256": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
            },
            "prior_launch_authority_amendment": {
                "schema_version": (
                    OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SCHEMA_VERSION
                ),
                "path": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH,
                "file_sha256": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256,
            },
            "launch_authority_amendment": {
                "schema_version": (
                    OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION
                ),
                "path": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH,
                "file_sha256": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256,
            },
            "editing_after_evidence_allowed": False,
        },
        "launch_authority": _open_ecology_staged_launch_authority(),
        "source": {
            "commit": commit,
            "manifest_sha256": manifest_sha256,
            "archive_tool_authority_sha256": archive_authority_sha256,
            "clean_tree_required": True,
            "stable_through_every_cell_required": True,
        },
        "runtime_contract": runtime,
        "throughput_gate": throughput,
        "seed_contract": {
            "registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
            "learner_role": "open_ecology_learner",
            "learner_indices": list(range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)),
            "learner_seeds": learners,
            "training_role": OPEN_ECOLOGY_TRAINING_SEED_ROLE,
            "training_seed_indices": list(
                range(
                    OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                    * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                )
            ),
            "training_seeds": train_seeds,
            "selection_role": "open_ecology_selection",
            "selection_seed_indices": list(
                range(OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT)
            ),
            "selection_seeds": selection_seeds,
            "genome_stream_role": "open_ecology_genome_stream",
            "genome_stream_pairing": "learner_index_selects_same_index",
            "phase_a_cell_pairing": (
                "same_learner_world_environment_genome_identity_and_policy_rng_"
                "across_A0_A3_v1"
            ),
            "benchmark_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "benchmark_seed_indices": list(
                range(len(OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]))
            ),
            "benchmark_seeds": list(
                OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
            ),
            "benchmark_environment_seed_indices": list(
                range(OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT)
            ),
            "benchmark_model_initialization_seed_index": (
                OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
            ),
            "benchmark_genome_stream_seed_index": (
                OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ),
            "benchmark_scientific_environment_seed_roles_accessed": [],
            "benchmark_scientific_outcomes_allowed": False,
            "engineering_proof_role": "open_ecology_proof",
            "engineering_proof_seed_indices": list(
                range(len(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"]))
            ),
            "engineering_proof_seeds": list(
                OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"]
            ),
            "engineering_proof_scientific_outcomes_allowed": False,
            "validation_available": False,
            "validation_accessed": False,
            "lockbox_available": False,
            "lockbox_accessed": False,
        },
        "architecture": {
            "model_contract_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
            "numeric_kernel": RECURRENT_NUMERIC_KERNEL_VERSION,
            "public_observation": "tokenized_four_opaque_communication_channels",
            "action_count": 20,
            "encoder_size": 256,
            "hidden_size": 256,
            "recurrent_layers": 1,
            "actor_genome_conditioning": GENOME_CONDITIONING_ACTOR_FILM_V1,
            "controller_genome_size": 16,
            "critic_gradient_cells": [
                {
                    "cell_id": cell_id,
                    "critic_genome_conditioning": (
                        OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][0]
                    ),
                    "value_shared_trunk_gradient": (
                        OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][1]
                    ),
                }
                for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
            ],
        },
        "training": {
            "updates_per_run": OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT,
            "worlds_per_update": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
            "worlds_per_run": (
                OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
            ),
            "total_worlds": OPEN_ECOLOGY_PHASE_A_TOTAL_TRAINING_WORLDS,
            "rollout_ticks": OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS,
            "initial_agents": OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS,
            "scenario_order": ["broad"],
            "fixture_names": [],
            "carrion_training_objective": False,
            "reward_shaping_added": False,
            "heuristic_action_selection": False,
            "counterfactual_auxiliary": False,
            "fixed_batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            "fixed_batch_execution_buckets": list(
                RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
            ),
            "update_commit_acceptance": {
                "accepted_ppo_minibatches_minimum_per_update": 1,
                "positive_parameter_delta_required": True,
                "model_state_change_required": True,
                "mixed_post_step_kl_rejection_evidence_preserved": True,
                "prefix_chain_initial_to_terminal_change_required": True,
            },
            "ppo": asdict(_phase_a_ppo_config(learner_seed=learners[0])),
            "run_matrix": run_matrix,
        },
        "selection": {
            "training_weights_reused": False,
            "environment_seed_role": "open_ecology_selection",
            "environment_count": OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT,
            "ticks_per_execution": OPEN_ECOLOGY_PHASE_A_SELECTION_TICKS,
            "stochastic_tape_identities": [
                f"phase-a-selection-tape-{index:02d}"
                for index in range(OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT)
            ],
            "argmax_diagnostic": True,
            "trained_policy_executions_per_artifact": (
                OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "initialized_baseline_executions_per_artifact": (
                OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT
            ),
            "primary_executions_per_artifact": (
                OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "independent_replay_executions_per_artifact": (
                OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "physical_world_runs_per_artifact": (
                2
                * OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "execution_count_semantics": (
                "trained_plus_same_tape_initialized_baseline_then_equal_full_"
                "authority_reexecution"
            ),
            "initialized_baseline": {
                "model": "exact_learner_seed_initialized_run_model",
                "tapes": "same_four_stochastic_selection_tapes",
                "argmax_or_causal_duplicate_executions": 0,
                "learner_eligibility": (
                    "paired_median_normalized_return_improvement_strictly_positive"
                ),
            },
            "action_collapse": {
                "rolling_span_decisions": 3_000,
                "minimum_decisions_for_eligibility": 3_000,
                "requested_action_share_threshold": 0.80,
                "represented_lineage_share_threshold": 0.90,
            },
            "fixtures": [],
            "causal_genome": {
                "states_per_environment": (
                    OPEN_ECOLOGY_PHASE_A_CAUSAL_STATES_PER_WORLD
                ),
                "states_per_artifact": OPEN_ECOLOGY_PHASE_A_CAUSAL_STATE_COUNT,
                "interventions": [
                    "original",
                    "zero",
                    "donor",
                    "single_locus_plus_or_minus_0.05",
                ],
                "original_vs_donor_js_share_threshold": 0.10,
                "original_vs_donor_js_state_threshold": 0.01,
                "mean_original_vs_donor_js_threshold": 0.002,
                "single_locus_total_variation_p99_max": 0.25,
                "capture_noninterference_proof": {
                    "schema_version": (
                        "mind_v3_open_ecology_capture_noninterference_proof_v1"
                    ),
                    "producer": (
                        "scripts/prove_open_ecology_capture_noninterference.py"
                    ),
                    "authority_verification": (
                        "exact_clean_source_full_reexecution_and_exact_report_match_v1"
                    ),
                    "environment_seed_role": "open_ecology_proof",
                    "environment_seed_indices": list(range(12)),
                    "scientific_selection_seed_accessed": False,
                    "cell_order": list(OPEN_ECOLOGY_PHASE_A_CELL_ORDER),
                    "case_matrix": "four_cells_by_three_densities_cartesian_v1",
                    "density_levels": [32, 64, 128],
                    "case_count": 12,
                    "ticks_per_case": 128,
                    "births_and_deaths_required": True,
                },
            },
            "cell_eligibility_minimum_passing_learners": 3,
            "selection_rule": (
                "lexicographic_median_normalized_return_1pct_tie_then_"
                "value_rmse_then_advantage_variance_then_stop_gradient_then_"
                "critic_none"
            ),
        },
        "evidence": {
            "update_journal": "atomic_directory_commit_hash_chain_v1",
            "checkpoint_per_committed_update": True,
            "resume_policy": "verified_contiguous_evidence_prefix_only",
            "terminal_training_artifact": (
                "strict_json_tensor_artifact_for_later_exact_cpu_reevaluation"
            ),
            "selection_evidence_required_before_cell_selection": True,
        },
    }
    preregistration["configuration_sha256"] = stable_payload_digest(
        {
            "architecture": preregistration["architecture"],
            "training": preregistration["training"],
            "selection": preregistration["selection"],
            "seed_contract": preregistration["seed_contract"],
            "throughput_gate": preregistration["throughput_gate"],
            "launch_authority": preregistration["launch_authority"],
        }
    )
    preregistration["exact_digest"] = stable_payload_digest(preregistration)
    validate_open_ecology_phase_a_preregistration(preregistration)
    return preregistration


def validate_open_ecology_phase_a_preregistration(
    preregistration: Mapping[str, object],
) -> None:
    _require_exact_keys(
        preregistration,
        {
            "schema_version",
            "campaign",
            "claim_boundary",
            "sealed_document",
            "launch_authority",
            "source",
            "runtime_contract",
            "throughput_gate",
            "seed_contract",
            "architecture",
            "training",
            "selection",
            "evidence",
            "configuration_sha256",
            "exact_digest",
        },
        field="Phase A preregistration",
    )
    if preregistration.get("schema_version") != OPEN_ECOLOGY_PHASE_A_SCHEMA_VERSION:
        raise OpenEcologyPhaseAError("Phase A preregistration schema drifted")
    _validate_signed_payload(preregistration, field="Phase A preregistration")
    source = _mapping(preregistration.get("source"), field="source")
    source_commit = _git_sha(source.get("commit"))
    _sha256(
        source.get("manifest_sha256"),
        field="source.manifest_sha256",
    )
    _sha256(
        source.get("archive_tool_authority_sha256"),
        field="source.archive_tool_authority_sha256",
    )
    if source.get("clean_tree_required") is not True:
        raise OpenEcologyPhaseAError("Phase A clean-source requirement drifted")
    runtime = _mapping(
        preregistration.get("runtime_contract"),
        field="runtime_contract",
    )
    validate_open_ecology_phase_a_runtime_contract(runtime)
    throughput_gate = _mapping(
        preregistration.get("throughput_gate"),
        field="throughput_gate",
    )
    validate_open_ecology_phase_a_throughput_gate(throughput_gate)
    if throughput_gate.get("source_commit") != source_commit or throughput_gate.get(
        "selected_rollout_workers"
    ) != runtime.get("rollout_workers"):
        raise OpenEcologyPhaseAError(
            "Phase A source/runtime detached from throughput evidence"
        )
    sealed = _mapping(preregistration.get("sealed_document"), field="sealed_document")
    if dict(sealed) != {
        "base_protocol": {
            "path": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH,
            "file_sha256": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
        },
        "prior_launch_authority_amendment": {
            "schema_version": (
                OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SCHEMA_VERSION
            ),
            "path": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH,
            "file_sha256": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256,
        },
        "launch_authority_amendment": {
            "schema_version": (OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION),
            "path": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH,
            "file_sha256": OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256,
        },
        "editing_after_evidence_allowed": False,
    }:
        raise OpenEcologyPhaseAError("sealed Phase A document binding drifted")
    if preregistration.get("launch_authority") != (
        _open_ecology_staged_launch_authority()
    ):
        raise OpenEcologyPhaseAError("staged open-ecology launch authority drifted")
    _validate_phase_a_matrix(preregistration)


def _validate_phase_a_matrix(preregistration: Mapping[str, object]) -> None:
    if preregistration.get("campaign") != (
        "open_ecology_phase_a_critic_gradient_ablation"
    ):
        raise OpenEcologyPhaseAError("Phase A campaign identity drifted")
    expected_claim_boundary = {
        "development_only": True,
        "phase_b_authorized_by_this_record": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
        "society_claim_authorized": False,
    }
    if preregistration.get("claim_boundary") != expected_claim_boundary:
        raise OpenEcologyPhaseAError("Phase A claim boundary drifted")
    source = _mapping(preregistration.get("source"), field="source")
    _require_exact_keys(
        source,
        {
            "commit",
            "manifest_sha256",
            "archive_tool_authority_sha256",
            "clean_tree_required",
            "stable_through_every_cell_required",
        },
        field="source",
    )
    if (
        source.get("clean_tree_required") is not True
        or source.get("stable_through_every_cell_required") is not True
    ):
        raise OpenEcologyPhaseAError("Phase A source stability contract drifted")
    seed_contract = _mapping(
        preregistration.get("seed_contract"), field="seed_contract"
    )
    expected_seed_contract = {
        "registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
        "registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        "learner_role": "open_ecology_learner",
        "learner_indices": list(range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)),
        "learner_seeds": list(_phase_a_learner_seeds()),
        "training_role": OPEN_ECOLOGY_TRAINING_SEED_ROLE,
        "training_seed_indices": list(
            range(
                OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
            )
        ),
        "training_seeds": list(
            OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_TRAINING_SEED_ROLE][
                : OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
            ]
        ),
        "selection_role": "open_ecology_selection",
        "selection_seed_indices": list(
            range(OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT)
        ),
        "selection_seeds": list(
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"][
                :OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            ]
        ),
        "genome_stream_role": "open_ecology_genome_stream",
        "genome_stream_pairing": "learner_index_selects_same_index",
        "phase_a_cell_pairing": (
            "same_learner_world_environment_genome_identity_and_policy_rng_"
            "across_A0_A3_v1"
        ),
        "benchmark_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        "benchmark_seed_indices": list(
            range(len(OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]))
        ),
        "benchmark_seeds": list(
            OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
        ),
        "benchmark_environment_seed_indices": list(
            range(OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT)
        ),
        "benchmark_model_initialization_seed_index": (
            OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
        ),
        "benchmark_genome_stream_seed_index": (
            OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
        ),
        "benchmark_scientific_environment_seed_roles_accessed": [],
        "benchmark_scientific_outcomes_allowed": False,
        "engineering_proof_role": "open_ecology_proof",
        "engineering_proof_seed_indices": list(
            range(len(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"]))
        ),
        "engineering_proof_seeds": list(
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"]
        ),
        "engineering_proof_scientific_outcomes_allowed": False,
        "validation_available": False,
        "validation_accessed": False,
        "lockbox_available": False,
        "lockbox_accessed": False,
    }
    if dict(seed_contract) != expected_seed_contract:
        raise OpenEcologyPhaseAError("Phase A seed boundary drifted")
    expected_architecture = {
        "model_contract_version": RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        "numeric_kernel": RECURRENT_NUMERIC_KERNEL_VERSION,
        "public_observation": "tokenized_four_opaque_communication_channels",
        "action_count": 20,
        "encoder_size": 256,
        "hidden_size": 256,
        "recurrent_layers": 1,
        "actor_genome_conditioning": GENOME_CONDITIONING_ACTOR_FILM_V1,
        "controller_genome_size": 16,
        "critic_gradient_cells": [
            {
                "cell_id": cell_id,
                "critic_genome_conditioning": OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][0],
                "value_shared_trunk_gradient": OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][1],
            }
            for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        ],
    }
    if preregistration.get("architecture") != expected_architecture:
        raise OpenEcologyPhaseAError("Phase A architecture contract drifted")
    training = _mapping(preregistration.get("training"), field="training")
    _require_exact_keys(
        training,
        {
            "updates_per_run",
            "worlds_per_update",
            "worlds_per_run",
            "total_worlds",
            "rollout_ticks",
            "initial_agents",
            "scenario_order",
            "fixture_names",
            "carrion_training_objective",
            "reward_shaping_added",
            "heuristic_action_selection",
            "counterfactual_auxiliary",
            "fixed_batch_capacity",
            "fixed_batch_execution_buckets",
            "update_commit_acceptance",
            "ppo",
            "run_matrix",
        },
        field="training",
    )
    if (
        training.get("updates_per_run") != OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
        or training.get("worlds_per_update") != OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
        or training.get("worlds_per_run")
        != (OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE)
        or training.get("total_worlds") != OPEN_ECOLOGY_PHASE_A_TOTAL_TRAINING_WORLDS
        or training.get("rollout_ticks") != OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS
        or training.get("initial_agents") != OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS
        or training.get("scenario_order") != ["broad"]
        or training.get("fixture_names") != []
        or training.get("carrion_training_objective") is not False
        or training.get("reward_shaping_added") is not False
        or training.get("heuristic_action_selection") is not False
        or training.get("counterfactual_auxiliary") is not False
        or training.get("fixed_batch_capacity")
        != OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
        or training.get("fixed_batch_execution_buckets")
        != list(RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS)
        or training.get("update_commit_acceptance")
        != {
            "accepted_ppo_minibatches_minimum_per_update": 1,
            "positive_parameter_delta_required": True,
            "model_state_change_required": True,
            "mixed_post_step_kl_rejection_evidence_preserved": True,
            "prefix_chain_initial_to_terminal_change_required": True,
        }
    ):
        raise OpenEcologyPhaseAError("Phase A training matrix drifted")
    ppo_reference = asdict(
        _phase_a_ppo_config(learner_seed=_phase_a_learner_seeds()[0])
    )
    observed_ppo = _mapping(training.get("ppo"), field="training.ppo")
    if dict(observed_ppo) != ppo_reference:
        raise OpenEcologyPhaseAError("Phase A PPO contract drifted")
    rows = _sequence(training.get("run_matrix"), field="training.run_matrix")
    if len(rows) != 4 * OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT:
        raise OpenEcologyPhaseAError("Phase A run matrix does not contain 16 cells")
    expected_rows: list[dict[str, object]] = []
    common_by_learner: dict[int, set[str]] = {
        index: set() for index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
    }
    for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER:
        critic, gradient = OPEN_ECOLOGY_PHASE_A_CELLS[cell_id]
        for learner_index, learner_seed in enumerate(_phase_a_learner_seeds()):
            model_config, _ppo, schedule = build_phase_a_run_components(
                cell_id=cell_id,
                learner_index=learner_index,
            )
            model = PublicRecurrentActorCritic(
                model_config,
                initialization_seed=learner_seed,
            ).to(device="cpu", dtype=torch.float32)
            common_digest = _common_parameter_sha256(model)
            common_by_learner[learner_index].add(common_digest)
            expected_rows.append(
                {
                    "run_id": phase_a_run_id(
                        cell_id=cell_id,
                        learner_index=learner_index,
                    ),
                    "cell_id": cell_id,
                    "learner_index": learner_index,
                    "learner_seed": learner_seed,
                    "genome_stream_seed_index": learner_index,
                    "genome_stream_seed": OPEN_ECOLOGY_SEED_REGISTRY[
                        "open_ecology_genome_stream"
                    ][learner_index],
                    "critic_genome_conditioning": critic,
                    "value_shared_trunk_gradient": gradient,
                    "initial_common_parameter_sha256": common_digest,
                    "initial_full_model_sha256": recurrent_model_state_sha256(model),
                    "schedule_sha256": _schedule_sha256(schedule),
                }
            )
    if [dict(_mapping(row, field="run_matrix[]")) for row in rows] != expected_rows:
        raise OpenEcologyPhaseAError("Phase A run matrix rows are not canonical")
    if any(len(values) != 1 for values in common_by_learner.values()):
        raise OpenEcologyPhaseAError(
            "Phase A initial actor/backbone tensors differ across cells"
        )
    expected_selection = {
        "training_weights_reused": False,
        "environment_seed_role": "open_ecology_selection",
        "environment_count": OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT,
        "ticks_per_execution": OPEN_ECOLOGY_PHASE_A_SELECTION_TICKS,
        "stochastic_tape_identities": [
            f"phase-a-selection-tape-{index:02d}"
            for index in range(OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT)
        ],
        "argmax_diagnostic": True,
        "trained_policy_executions_per_artifact": (
            OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "initialized_baseline_executions_per_artifact": (
            OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT
        ),
        "primary_executions_per_artifact": (
            OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "independent_replay_executions_per_artifact": (
            OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "physical_world_runs_per_artifact": (
            2
            * OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "execution_count_semantics": (
            "trained_plus_same_tape_initialized_baseline_then_equal_full_"
            "authority_reexecution"
        ),
        "initialized_baseline": {
            "model": "exact_learner_seed_initialized_run_model",
            "tapes": "same_four_stochastic_selection_tapes",
            "argmax_or_causal_duplicate_executions": 0,
            "learner_eligibility": (
                "paired_median_normalized_return_improvement_strictly_positive"
            ),
        },
        "action_collapse": {
            "rolling_span_decisions": 3_000,
            "minimum_decisions_for_eligibility": 3_000,
            "requested_action_share_threshold": 0.80,
            "represented_lineage_share_threshold": 0.90,
        },
        "fixtures": [],
        "causal_genome": {
            "states_per_environment": OPEN_ECOLOGY_PHASE_A_CAUSAL_STATES_PER_WORLD,
            "states_per_artifact": OPEN_ECOLOGY_PHASE_A_CAUSAL_STATE_COUNT,
            "interventions": [
                "original",
                "zero",
                "donor",
                "single_locus_plus_or_minus_0.05",
            ],
            "original_vs_donor_js_share_threshold": 0.10,
            "original_vs_donor_js_state_threshold": 0.01,
            "mean_original_vs_donor_js_threshold": 0.002,
            "single_locus_total_variation_p99_max": 0.25,
            "capture_noninterference_proof": {
                "schema_version": (
                    "mind_v3_open_ecology_capture_noninterference_proof_v1"
                ),
                "producer": "scripts/prove_open_ecology_capture_noninterference.py",
                "authority_verification": (
                    "exact_clean_source_full_reexecution_and_exact_report_match_v1"
                ),
                "environment_seed_role": "open_ecology_proof",
                "environment_seed_indices": list(range(12)),
                "scientific_selection_seed_accessed": False,
                "cell_order": list(OPEN_ECOLOGY_PHASE_A_CELL_ORDER),
                "case_matrix": "four_cells_by_three_densities_cartesian_v1",
                "density_levels": [32, 64, 128],
                "case_count": 12,
                "ticks_per_case": 128,
                "births_and_deaths_required": True,
            },
        },
        "cell_eligibility_minimum_passing_learners": 3,
        "selection_rule": (
            "lexicographic_median_normalized_return_1pct_tie_then_"
            "value_rmse_then_advantage_variance_then_stop_gradient_then_"
            "critic_none"
        ),
    }
    if preregistration.get("selection") != expected_selection:
        raise OpenEcologyPhaseAError("Phase A selection contract drifted")
    expected_evidence = {
        "update_journal": "atomic_directory_commit_hash_chain_v1",
        "checkpoint_per_committed_update": True,
        "resume_policy": "verified_contiguous_evidence_prefix_only",
        "terminal_training_artifact": (
            "strict_json_tensor_artifact_for_later_exact_cpu_reevaluation"
        ),
        "selection_evidence_required_before_cell_selection": True,
    }
    if preregistration.get("evidence") != expected_evidence:
        raise OpenEcologyPhaseAError("Phase A evidence contract drifted")
    expected_configuration = stable_payload_digest(
        {
            "architecture": preregistration["architecture"],
            "training": preregistration["training"],
            "selection": preregistration["selection"],
            "seed_contract": preregistration["seed_contract"],
            "throughput_gate": preregistration["throughput_gate"],
            "launch_authority": preregistration["launch_authority"],
        }
    )
    if preregistration.get("configuration_sha256") != expected_configuration:
        raise OpenEcologyPhaseAError("Phase A configuration digest mismatched")


def build_phase_a_run_components(
    *,
    cell_id: str,
    learner_index: int,
) -> tuple[
    RecurrentActorCriticConfig,
    RecurrentPPOConfig,
    tuple[tuple[PhaseAOpenEcologyRolloutTask, ...], ...],
]:
    resolved_cell = _cell_id(cell_id)
    resolved_learner_index = _index(
        learner_index,
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    learner_seed = _phase_a_learner_seeds()[resolved_learner_index]
    critic, gradient = OPEN_ECOLOGY_PHASE_A_CELLS[resolved_cell]
    treatment = OpenEcologyBroadWorldTreatment(
        initial_agents=OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS
    )
    model_config = RecurrentActorCriticConfig.for_signal_config(
        treatment.signals.as_signal_config(),
        encoder_size=256,
        hidden_size=256,
        recurrent_layers=1,
        genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        critic_genome_conditioning=critic,
        value_shared_trunk_gradient=gradient,
    )
    ppo_config = _phase_a_ppo_config(learner_seed=learner_seed)
    train_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_TRAINING_SEED_ROLE]
    genome_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
        resolved_learner_index
    ]
    schedule: list[tuple[PhaseAOpenEcologyRolloutTask, ...]] = []
    sampling_seeds: set[int] = set()
    for update_index in range(OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT):
        tasks: list[PhaseAOpenEcologyRolloutTask] = []
        for local_world_index in range(OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE):
            world_index = (
                update_index * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                + local_world_index
            )
            environment_seed = train_seeds[world_index]
            identity = _phase_a_policy_sampling_identity(
                learner_index=resolved_learner_index,
                learner_seed=learner_seed,
                genome_stream_seed_index=resolved_learner_index,
                update_index=update_index,
                world_index=world_index,
                environment_seed_index=world_index,
            )
            sampling_seed = derive_recurrent_policy_sampling_seed(
                task_identity=identity
            )
            genome_world_identity = _phase_a_genome_world_identity(
                learner_index=resolved_learner_index,
                learner_seed=learner_seed,
                genome_stream_seed_index=resolved_learner_index,
                update_index=update_index,
                world_index=world_index,
                environment_seed_index=world_index,
                environment_seed=environment_seed,
            )
            if sampling_seed in sampling_seeds:
                raise OpenEcologyPhaseAError("Phase A policy sampling seed collision")
            sampling_seeds.add(sampling_seed)
            tasks.append(
                PhaseAOpenEcologyRolloutTask(
                    task_id=_phase_a_task_id(
                        cell_id=resolved_cell,
                        learner_index=resolved_learner_index,
                        learner_seed=learner_seed,
                        update_index=update_index,
                        world_index=world_index,
                        environment_seed_index=world_index,
                        environment_seed=environment_seed,
                    ),
                    scenario="broad",
                    environment_seed=environment_seed,
                    rollout_ticks=OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS,
                    policy_sampling_identity=identity,
                    policy_sampling_seed=sampling_seed,
                    seed_role=OPEN_ECOLOGY_TRAINING_SEED_ROLE,
                    genome_stream_seed=genome_seed,
                    genome_population_mode=(
                        RecurrentGenomePopulationMode.HERITABLE.value
                    ),
                    open_ecology_treatment=treatment,
                    open_ecology_learner_seed=learner_seed,
                    open_ecology_environment_seed_index=world_index,
                    open_ecology_genome_stream_seed_index=resolved_learner_index,
                    open_ecology_training_phase=OPEN_ECOLOGY_PHASE_A,
                    open_ecology_update_index=update_index,
                    open_ecology_world_index=world_index,
                    phase_a_cell=resolved_cell,
                    phase_a_learner_index=resolved_learner_index,
                    phase_a_genome_world_identity=genome_world_identity,
                )
            )
        schedule.append(tuple(tasks))
    return model_config, ppo_config, tuple(schedule)


def phase_a_run_id(*, cell_id: str, learner_index: int) -> str:
    resolved_cell = _cell_id(cell_id)
    resolved_index = _index(
        learner_index,
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    return (
        f"phase-a-{resolved_cell.lower()}-learner-{resolved_index}-"
        f"{_phase_a_learner_seeds()[resolved_index]}"
    )


def build_phase_a_run_contract(
    preregistration: Mapping[str, object],
    *,
    cell_id: str,
    learner_index: int,
) -> dict[str, object]:
    validate_open_ecology_phase_a_preregistration(preregistration)
    resolved_cell = _cell_id(cell_id)
    resolved_index = _index(
        learner_index,
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    learner_seed = _phase_a_learner_seeds()[resolved_index]
    model_config, ppo_config, schedule = build_phase_a_run_components(
        cell_id=resolved_cell,
        learner_index=resolved_index,
    )
    model = PublicRecurrentActorCritic(
        model_config,
        initialization_seed=learner_seed,
    ).to(device="cpu", dtype=torch.float32)
    contract: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_RUN_CONTRACT_VERSION,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": _json_clone(preregistration["source"], field="source"),
        "runtime_contract_digest": _mapping(
            preregistration["runtime_contract"],
            field="runtime_contract",
        )["exact_digest"],
        "run_id": phase_a_run_id(
            cell_id=resolved_cell,
            learner_index=resolved_index,
        ),
        "cell_id": resolved_cell,
        "learner_index": resolved_index,
        "learner_seed": learner_seed,
        "model": asdict(model_config),
        "ppo": asdict(ppo_config),
        "genome_population": {
            "mode": RecurrentGenomePopulationMode.HERITABLE.value,
            "stream_seed_index": resolved_index,
            "stream_seed": OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
                resolved_index
            ],
        },
        "training": {
            "scenario_order": ["broad"],
            "fixture_names": [],
            "environment_seed_role": OPEN_ECOLOGY_TRAINING_SEED_ROLE,
            "environment_seed_indices": list(
                range(
                    OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                    * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                )
            ),
            "update_count": OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT,
            "worlds_per_update": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
            "rollout_ticks": OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS,
            "initial_agents": OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS,
            "collector_device": "cpu",
            "fixed_batch_capacity": (OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY),
            "fixed_batch_execution_buckets": list(
                RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
            ),
            "counterfactual_auxiliary": False,
            "validation_accessed": False,
            "lockbox_accessed": False,
            "carrion_training_objective": False,
        },
        "schedule_sha256": _schedule_sha256(schedule),
        "initial_common_parameter_sha256": _common_parameter_sha256(model),
        "initial_full_model_sha256": recurrent_model_state_sha256(model),
        "terminal_policy": {
            "update_count": OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT,
            "strict_cpu_load_required": True,
            "selection_evaluation_pending": True,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
        },
    }
    contract["exact_digest"] = stable_payload_digest(contract)
    return contract


def build_verified_phase_a_selection_request(
    preregistration: Mapping[str, object],
    *,
    cell_id: str,
    learner_index: int,
    terminal_path: str | Path,
    evaluation_workers: int = 1,
) -> OpenEcologySelectionRequest:
    """Derive every Phase-A selection pin from one verified terminal chain."""

    from evolution_sim.mind import open_ecology_selection as selection

    validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source(preregistration)
    resolved_cell = _cell_id(cell_id)
    resolved_index = _index(
        learner_index,
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    path = Path(terminal_path)
    if not path.is_file() or path.is_symlink() or path.name != "terminal.json":
        raise OpenEcologyPhaseAError(
            "Phase A selection terminal must be a regular terminal.json file"
        )
    terminal_directory = path.parent
    run_directory = terminal_directory.parent
    if (
        terminal_directory.name != "terminal"
        or not terminal_directory.is_dir()
        or terminal_directory.is_symlink()
        or not run_directory.is_dir()
        or run_directory.is_symlink()
        or run_directory.name
        != phase_a_run_id(
            cell_id=resolved_cell,
            learner_index=resolved_index,
        )
        or {entry.name for entry in terminal_directory.iterdir()}
        != {
            "terminal.json",
            "terminal-training-artifact.json",
            "run-contract.json",
        }
    ):
        raise OpenEcologyPhaseAError(
            "Phase A selection terminal directory is not the canonical bundle"
        )
    preview = _load_strict_json(path)
    if (
        preview.get("campaign_digest") != preregistration.get("exact_digest")
        or preview.get("cell_id") != resolved_cell
        or preview.get("learner_index") != resolved_index
    ):
        raise OpenEcologyPhaseAError(
            "Phase A selection terminal identity differs from its request"
        )

    def bound_path(
        reference: Mapping[str, object],
        *,
        field: str,
        logical_name: str,
    ) -> Path:
        file_reference = _mapping(
            reference.get("file"),
            field=f"{field}.file",
        )
        relative = file_reference.get("path")
        if (
            relative != logical_name
            or not isinstance(relative, str)
            or Path(relative).name != relative
        ):
            raise OpenEcologyPhaseAError(
                f"{field} does not resolve one canonical relative file"
            )
        resolved = terminal_directory / relative
        if not resolved.is_file() or resolved.is_symlink():
            raise OpenEcologyPhaseAError(
                f"{field} does not resolve a regular terminal file"
            )
        return resolved

    run_contract_path = bound_path(
        _mapping(preview.get("run_contract"), field="terminal.run_contract"),
        field="terminal.run_contract",
        logical_name="run-contract.json",
    )
    artifact_path = bound_path(
        _mapping(preview.get("artifact"), field="terminal.artifact"),
        field="terminal.artifact",
        logical_name="terminal-training-artifact.json",
    )
    run_contract = build_phase_a_run_contract(
        preregistration,
        cell_id=resolved_cell,
        learner_index=resolved_index,
    )
    _model_config, _ppo_config, schedule = build_phase_a_run_components(
        cell_id=resolved_cell,
        learner_index=resolved_index,
    )
    terminal = _verify_phase_a_terminal(
        path,
        artifact_path=artifact_path,
        run_contract_path=run_contract_path,
        run_contract=run_contract,
        run_directory=run_directory,
        schedule=schedule,
    )
    training = _mapping(terminal.get("training"), field="terminal.training")
    commits = training.get("commit_exact_digests")
    if (
        not isinstance(commits, Sequence)
        or isinstance(commits, (str, bytes))
        or len(commits) != OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
    ):
        raise OpenEcologyPhaseAError(
            "Phase A terminal does not expose the complete verified prefix"
        )
    artifact = _mapping(terminal.get("artifact"), field="terminal.artifact")
    source = _mapping(run_contract.get("source"), field="run.source")
    authority: dict[str, object] = {
        "schema_version": (
            selection.OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION
        ),
        "campaign_digest": run_contract["campaign_digest"],
        "source_commit": source["commit"],
        "source_manifest_sha256": source["manifest_sha256"],
        "run_id": run_contract["run_id"],
        "cell_id": resolved_cell,
        "learner_index": resolved_index,
        "learner_seed": run_contract["learner_seed"],
        "terminal_logical_name": "terminal.json",
        "terminal_exact_digest": terminal["exact_digest"],
        "terminal_file_sha256": _file_sha256(path),
        "final_prefix_commit_exact_digest": _sha256(
            commits[-1],
            field="terminal final prefix commit",
        ),
        "terminal_checkpoint_model_state_sha256": _sha256(
            training.get("terminal_model_state_sha256"),
            field="terminal model state",
        ),
        "initial_model_state_sha256": _sha256(
            training.get("initial_model_state_sha256"),
            field="initial model state",
        ),
        "initial_to_terminal_model_changed": (
            training.get("initial_to_terminal_model_changed")
        ),
        "cumulative_accepted_ppo_minibatches": _positive_int(
            training.get("cumulative_accepted_ppo_minibatches"),
            field="terminal cumulative accepted PPO minibatches",
        ),
        "cumulative_post_step_kl_rejected_steps": int(
            training.get("cumulative_post_step_kl_rejected_steps", 0)
        ),
        "run_contract_logical_name": "run-contract.json",
        "run_contract_exact_digest": run_contract["exact_digest"],
        "run_contract_file_sha256": _file_sha256(run_contract_path),
        "artifact_logical_name": "terminal-training-artifact.json",
        "artifact_sha256": _sha256(
            artifact.get("artifact_sha256"),
            field="terminal artifact SHA",
        ),
        "artifact_file_sha256": _file_sha256(artifact_path),
        "selection_binding_required": True,
    }
    authority["exact_digest"] = stable_payload_digest(authority)
    if (
        authority["schema_version"]
        != OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION
    ):
        raise OpenEcologyPhaseAError("Phase A and selection authority schemas differ")
    return selection.OpenEcologySelectionRequest(
        artifact_path=artifact_path,
        run_contract_path=run_contract_path,
        expected_artifact_sha256=str(authority["artifact_sha256"]),
        expected_source_commit=str(source["commit"]),
        expected_source_manifest_sha256=str(source["manifest_sha256"]),
        campaign_digest=str(run_contract["campaign_digest"]),
        run_contract_digest=str(run_contract["exact_digest"]),
        phase=selection.OpenEcologySelectionPhase.PHASE_A,
        cell_id=resolved_cell,
        learner_index=resolved_index,
        learner_seed=int(run_contract["learner_seed"]),
        evaluation_workers=evaluation_workers,
        training_authority=authority,
        source_repository_root=_REPOSITORY_ROOT,
    )


def authorize_phase_a_terminal_selection_report(
    preregistration: Mapping[str, object],
    report: Mapping[str, object],
    *,
    cell_id: str,
    learner_index: int,
    terminal_path: str | Path,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    """Rebuild and reexecute one canonical terminal chain before authorization.

    This is the only public path that may emit Phase-A learner selection
    evidence. Portable request objects, report mappings, and unkeyed digests
    remain provisional integrity data and cannot authorize this transition.
    """

    from evolution_sim.mind.open_ecology_selection import (
        _phase_a_learner_evidence_from_verified_selection_report,
    )

    request = build_verified_phase_a_selection_request(
        preregistration,
        cell_id=cell_id,
        learner_index=learner_index,
        terminal_path=terminal_path,
        evaluation_workers=evaluation_workers,
    )
    return _phase_a_learner_evidence_from_verified_selection_report(
        report,
        request=request,
    )


def validate_open_ecology_phase_a_selection_authorization(
    authorization: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    authorization_path: str | Path,
    terminal_root: str | Path,
    terminal_requests: Sequence[OpenEcologySelectionRequest],
) -> None:
    """Validate a separately sealed authority before Phase-A selection.

    This validator deliberately has no paired producer. Selection remains
    blocked until the causal-rejection, capture-reexecution, and full-shape
    selection-throughput reports each have independent raw-evidence verifiers.
    """

    from evolution_sim.mind.open_ecology_phase_a_readiness import (
        _load_report_reference,
        _parse_utc,
        _require_bound_payload_file,
        _utc_now,
    )

    validate_open_ecology_phase_a_preregistration(preregistration)
    raw_authorization_path = Path(authorization_path)
    resolved_authorization_path = _require_bound_payload_file(
        raw_authorization_path,
        authorization,
        field="Phase A selection authorization",
    )
    _require_exact_keys(
        authorization,
        {
            "schema_version",
            "authorized_at_utc",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "dependency_evidence",
            "selection_throughput_evidence",
            "terminal_matrix",
            "authorization",
            "exact_digest",
        },
        field="Phase A selection authorization",
    )
    _validate_signed_payload(
        authorization,
        field="Phase A selection authorization",
    )
    if (
        authorization.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_SELECTION_AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("campaign_digest") != preregistration.get("exact_digest")
        or authorization.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or authorization.get("source") != preregistration.get("source")
    ):
        raise OpenEcologyPhaseAError(
            "Phase A selection authorization is stale or detached from the campaign"
        )
    authorized_at = _parse_utc(
        authorization.get("authorized_at_utc"),
        field="selection_authorization.authorized_at_utc",
    )
    if authorized_at > _utc_now():
        raise OpenEcologyPhaseAError(
            "Phase A selection authorization postdates current validation time"
        )
    scope = _mapping(
        authorization.get("authorization"),
        field="selection_authorization.authorization",
    )
    _require_exact_keys(
        scope,
        {
            "authorization_scope",
            "phase_a_selection_authorized",
            "phase_b_authorized",
            "phase_c_authorized",
            "phase_d_authorized",
            "runtime_integration_authorized",
            "promotion_authorized",
        },
        field="selection_authorization.authorization",
    )
    if scope != {
        "authorization_scope": "phase_a_selection_only",
        "phase_a_selection_authorized": True,
        "phase_b_authorized": False,
        "phase_c_authorized": False,
        "phase_d_authorized": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
    }:
        raise OpenEcologyPhaseAError("Phase A selection authorization scope drifted")

    root = _canonical_phase_a_terminal_root(terminal_root)
    raw_matrix = _sequence(
        authorization.get("terminal_matrix"),
        field="selection_authorization.terminal_matrix",
    )
    expected_count = (
        len(OPEN_ECOLOGY_PHASE_A_CELL_ORDER) * OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
    )
    if len(raw_matrix) != expected_count or len(terminal_requests) != expected_count:
        raise OpenEcologyPhaseAError(
            "Phase A selection authorization requires all 16 terminal bundles"
        )
    canonical_identities = [
        (cell_id, learner_index)
        for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
    ]
    observed_identities: list[tuple[str, int]] = []
    for index, (raw_entry, request) in enumerate(
        zip(raw_matrix, terminal_requests, strict=True)
    ):
        entry = _mapping(
            raw_entry,
            field=f"selection_authorization.terminal_matrix[{index}]",
        )
        _require_exact_keys(
            entry,
            {
                "cell_id",
                "learner_index",
                "run_id",
                "terminal_authority",
                "terminal_file",
                "run_contract_file",
                "artifact_file",
            },
            field=f"selection_authorization.terminal_matrix[{index}]",
        )
        identity = (
            _cell_id(entry.get("cell_id")),
            _index(
                entry.get("learner_index"),
                field=f"terminal_matrix[{index}].learner_index",
                upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
            ),
        )
        observed_identities.append(identity)
        if identity != canonical_identities[index]:
            raise OpenEcologyPhaseAError(
                "Phase A selection terminal matrix is missing, duplicate, or reordered"
            )
        expected_entry = _phase_a_selection_terminal_authority_entry(
            request,
            terminal_root=root,
        )
        if dict(entry) != expected_entry:
            raise OpenEcologyPhaseAError(
                "Phase A selection terminal authority is stale, replaced, or "
                "detached from its canonical bundle"
            )
    if observed_identities != canonical_identities:
        raise OpenEcologyPhaseAError(
            "Phase A selection terminal matrix is missing, duplicate, or reordered"
        )

    base = resolved_authorization_path.parent
    raw_dependencies = _sequence(
        authorization.get("dependency_evidence"),
        field="selection_authorization.dependency_evidence",
    )
    required_dependency_kinds = (
        "causal_evaluator_rejection_battery",
        "capture_noninterference_reexecution",
    )
    if len(raw_dependencies) != len(required_dependency_kinds):
        raise OpenEcologyPhaseAError(
            "Phase A selection dependency evidence is missing or duplicated"
        )
    for index, (raw_entry, expected_kind) in enumerate(
        zip(raw_dependencies, required_dependency_kinds, strict=True)
    ):
        entry = _selection_evidence_entry(
            raw_entry,
            expected_kind=expected_kind,
            field=f"selection_authorization.dependency_evidence[{index}]",
        )
        report = _load_report_reference(
            base,
            entry["file"],
            expected_kind=expected_kind,
            preregistration=preregistration,
            authorization_time=authorized_at,
        )
        if entry["semantic_report_digest"] != report.get("exact_digest"):
            raise OpenEcologyPhaseAError(
                f"{expected_kind} selection evidence digest drifted"
            )
    throughput_entry = _selection_evidence_entry(
        authorization.get("selection_throughput_evidence"),
        expected_kind="phase_a_selection_throughput",
        field="selection_authorization.selection_throughput_evidence",
    )
    throughput_report = _load_report_reference(
        base,
        throughput_entry["file"],
        expected_kind="phase_a_selection_throughput",
        preregistration=preregistration,
        authorization_time=authorized_at,
    )
    if throughput_entry["semantic_report_digest"] != throughput_report.get(
        "exact_digest"
    ):
        raise OpenEcologyPhaseAError(
            "phase_a_selection_throughput evidence digest drifted"
        )
    _require_live_source(preregistration)


def _preflight_open_ecology_phase_a_selection(
    preregistration: Mapping[str, object],
    *,
    terminal_root: str | Path,
    selection_authorization_path: str | Path,
    evaluation_workers: int,
) -> tuple[
    dict[str, object],
    tuple[OpenEcologySelectionRequest, ...],
    tuple[Path, ...],
]:
    """Validate the complete stage boundary before the first selection seed."""

    validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source(preregistration)
    root = _canonical_phase_a_terminal_root(terminal_root)
    terminal_paths = _canonical_phase_a_terminal_paths(root)
    authorization_path = Path(selection_authorization_path)
    authorization = _load_sealed_selection_json(
        authorization_path,
        field="Phase A selection authorization",
    )
    requests = tuple(
        build_verified_phase_a_selection_request(
            preregistration,
            cell_id=cell_id,
            learner_index=learner_index,
            terminal_path=terminal_path,
            evaluation_workers=evaluation_workers,
        )
        for (cell_id, learner_index), terminal_path in zip(
            (
                (cell_id, learner_index)
                for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
                for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
            ),
            terminal_paths,
            strict=True,
        )
    )
    validate_open_ecology_phase_a_selection_authorization(
        authorization,
        preregistration=preregistration,
        authorization_path=authorization_path,
        terminal_root=root,
        terminal_requests=requests,
    )
    reloaded = _load_sealed_selection_json(
        authorization_path,
        field="Phase A selection authorization",
    )
    if reloaded != authorization:
        raise OpenEcologyPhaseAError(
            "Phase A selection authorization was replaced during preflight"
        )
    validate_open_ecology_phase_a_selection_authorization(
        reloaded,
        preregistration=preregistration,
        authorization_path=authorization_path,
        terminal_root=root,
        terminal_requests=requests,
    )
    _require_live_source(preregistration)
    return reloaded, requests, terminal_paths


def authorize_open_ecology_phase_a_cell_selection(
    preregistration: Mapping[str, object],
    *,
    terminal_root: str | Path,
    selection_authorization_path: str | Path,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    """Reopen and independently evaluate every Phase-A terminal before ranking.

    The caller supplies no learner summaries or artifact digests. This function
    derives the canonical 4 x 4 terminal paths, produces one fresh primary
    selection report per artifact, rebuilds each terminal chain, independently
    reexecutes the full selection matrix, and only then applies the sealed
    ranking rule. Any incomplete, replaced, or nondeterministic terminal fails
    the entire matrix closed.
    """

    from evolution_sim.mind.open_ecology_selection import (
        evaluate_open_ecology_selection_artifact,
    )

    selection_authorization, requests, terminal_paths = (
        _preflight_open_ecology_phase_a_selection(
            preregistration,
            terminal_root=terminal_root,
            selection_authorization_path=selection_authorization_path,
            evaluation_workers=evaluation_workers,
        )
    )

    learner_evidence: list[dict[str, object]] = []
    primary_report_digests: list[str] = []
    for request, terminal_path in zip(requests, terminal_paths, strict=True):
        primary_report = evaluate_open_ecology_selection_artifact(request)
        evidence = authorize_phase_a_terminal_selection_report(
            preregistration,
            primary_report,
            cell_id=request.cell_id,
            learner_index=request.learner_index,
            terminal_path=terminal_path,
            evaluation_workers=evaluation_workers,
        )
        learner_evidence.append(evidence)
        primary_report_digests.append(
            _sha256(
                primary_report.get("exact_digest"),
                field="primary selection report exact_digest",
            )
        )

    _require_live_source(preregistration)
    preview = preview_open_ecology_phase_a_cell_selection(
        preregistration,
        learner_evidence,
    )
    selected_cell = _cell_id(preview.get("provisional_selected_cell_id"))
    selected_contract = _mapping(
        preview.get("provisional_selected_contract"),
        field="provisional selected contract",
    )
    terminal_authority_digests = [
        _sha256(
            _mapping(
                evidence.get("terminal_authority"),
                field="learner terminal authority",
            ).get("exact_digest"),
            field="learner terminal authority exact_digest",
        )
        for evidence in learner_evidence
    ]
    result: dict[str, object] = {
        "schema_version": (OPEN_ECOLOGY_PHASE_A_AUTHORITATIVE_SELECTION_SCHEMA_VERSION),
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "authority": {
            "terminal_bundle_count": len(learner_evidence),
            "terminal_matrix": "four_cells_by_four_learners_canonical_paths_v1",
            "primary_report_producer": (
                "fresh_exact_cpu_open_ecology_selection_execution_v1"
            ),
            "independent_verification": (
                "rebuild_terminal_prefix_checkpoint_and_artifact_then_"
                "full_exact_cpu_reexecution_v1"
            ),
            "caller_supplied_learner_summaries": False,
            "all_terminal_bundles_reopened": True,
            "all_primary_reports_independently_reexecuted": True,
            "selection_authorization_exact_digest": (
                selection_authorization["exact_digest"]
            ),
            "selection_preflight_completed_before_seed_access": True,
        },
        "primary_selection_report_exact_digests": primary_report_digests,
        "terminal_authority_exact_digests": terminal_authority_digests,
        "learner_evidence": learner_evidence,
        "learner_evidence_exact_digests": [
            evidence["exact_digest"] for evidence in learner_evidence
        ],
        "cell_summaries": preview["cell_summaries"],
        "selected_cell_id": selected_cell,
        "selected_contract": dict(selected_contract),
        "phase_b_authorized": True,
        "selection_rule": preview["selection_rule"],
        "lifecycle": {
            "development_only": True,
            "phase_a_selection_complete": True,
            "phase_b_training_authorized": True,
            "validation_accessed": False,
            "lockbox_accessed": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
            "society_claim_authorized": False,
        },
    }
    result["exact_digest"] = stable_payload_digest(result)
    _require_live_source(preregistration)
    return result


def run_open_ecology_phase_a_cell(
    preregistration: Mapping[str, object],
    *,
    expected_preregistration_digest: str,
    launch_authorization_path: str | Path,
    cell_id: str,
    learner_index: int,
    output_root: str | Path,
    device: torch.device | str,
    resume: bool,
    live_launch_capability: object | None = None,
) -> dict[str, object]:
    """Train exactly one Phase A cell/learner with atomic evidence commits."""

    validate_open_ecology_phase_a_preregistration(preregistration)
    expected_digest = _sha256(
        expected_preregistration_digest,
        field="expected_preregistration_digest",
    )
    if preregistration.get("exact_digest") != expected_digest:
        raise OpenEcologyPhaseAError("Phase A preregistration digest pin mismatched")
    if type(resume) is not bool:
        raise OpenEcologyPhaseAError("resume must be an exact boolean")
    authorization_path = Path(launch_authorization_path).resolve()
    launch_authorization = _load_strict_json(authorization_path)
    validate_open_ecology_phase_a_launch_authorization(
        launch_authorization,
        preregistration=preregistration,
        authorization_path=authorization_path,
    )
    resolved_cell = _cell_id(cell_id)
    resolved_index = _index(
        learner_index,
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    from evolution_sim.mind.open_ecology_phase_a_guardian import (
        require_live_phase_a_update_admission,
    )

    require_live_phase_a_update_admission(
        live_launch_capability,
        preregistration=preregistration,
        launch_authorization_digest=_sha256(
            launch_authorization.get("exact_digest"),
            field="launch_authorization.exact_digest",
        ),
        cell_id=resolved_cell,
        learner_index=resolved_index,
        stage="before_activation",
    )
    configure_open_ecology_phase_a_determinism()
    _require_live_source(preregistration)
    _require_live_runtime(preregistration, device=device)
    _require_launch_dependencies()
    run_contract = build_phase_a_run_contract(
        preregistration,
        cell_id=resolved_cell,
        learner_index=resolved_index,
    )
    model_config, ppo_config, schedule = build_phase_a_run_components(
        cell_id=resolved_cell,
        learner_index=resolved_index,
    )
    run_directory = Path(output_root) / phase_a_run_id(
        cell_id=resolved_cell, learner_index=resolved_index
    )
    _require_live_output_storage(
        Path(output_root),
        launch_authorization=launch_authorization,
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(run_directory, run_contract=run_contract):
        terminal_directory = run_directory / "terminal"
        terminal_path = terminal_directory / "terminal.json"
        artifact_path = terminal_directory / "terminal-training-artifact.json"
        run_contract_path = terminal_directory / "run-contract.json"
        _recover_unpublished_terminal_staging(
            run_directory,
            run_contract=run_contract,
            schedule=schedule,
            resume=resume,
        )
        if terminal_directory.exists():
            if not resume:
                raise OpenEcologyPhaseAError(
                    "terminal evidence exists but resume was not enabled"
                )
            if (
                not terminal_directory.is_dir()
                or terminal_directory.is_symlink()
                or {path.name for path in terminal_directory.iterdir()}
                != {
                    "terminal.json",
                    "terminal-training-artifact.json",
                    "run-contract.json",
                }
            ):
                raise OpenEcologyPhaseAError(
                    "Phase A terminal directory is incomplete or contains surplus"
                )
            require_live_phase_a_update_admission(
                live_launch_capability,
                preregistration=preregistration,
                launch_authorization_digest=_sha256(
                    launch_authorization.get("exact_digest"),
                    field="launch_authorization.exact_digest",
                ),
                cell_id=resolved_cell,
                learner_index=resolved_index,
                stage="terminal_resume",
            )
            return _verify_phase_a_terminal(
                terminal_path,
                artifact_path=artifact_path,
                run_contract_path=run_contract_path,
                run_contract=run_contract,
                run_directory=run_directory,
                schedule=schedule,
            )
        if (
            (run_directory / "terminal.json").exists()
            or (run_directory / "terminal-training-artifact.json").exists()
            or (run_directory / "run-contract.json").exists()
        ):
            raise OpenEcologyPhaseAError(
                "legacy non-transactional terminal evidence is not accepted"
            )

        _recover_unpublished_update_staging(
            run_directory,
            run_contract=run_contract,
            schedule=schedule,
            resume=resume,
        )
        prefix = verify_phase_a_evidence_prefix(
            run_directory,
            run_contract=run_contract,
            schedule=schedule,
        )
        if prefix.completed_updates and not resume:
            raise OpenEcologyPhaseAError(
                "committed update evidence exists but resume was not enabled"
            )
        learner_seed = _phase_a_learner_seeds()[resolved_index]
        workers = _positive_int(
            _mapping(
                preregistration.get("runtime_contract"),
                field="runtime_contract",
            ).get("rollout_workers"),
            field="runtime.rollout_workers",
        )
        runner = RecurrentExperimentRunner(
            learner_seed=learner_seed,
            device=device,
            model_config=model_config,
            ppo_config=ppo_config,
            rollout_workers=workers,
            fixed_batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            counterfactual_config=None,
        )
        _require_live_source(preregistration)
        require_live_phase_a_update_admission(
            live_launch_capability,
            preregistration=preregistration,
            launch_authorization_digest=_sha256(
                launch_authorization.get("exact_digest"),
                field="launch_authorization.exact_digest",
            ),
            cell_id=resolved_cell,
            learner_index=resolved_index,
            stage="after_activation",
        )
        if prefix.terminal_checkpoint is not None:
            loaded = prefix.terminal_checkpoint
            runner.restore_training_checkpoint_state(
                model_state=loaded.model.state_dict(),
                optimizer_state=_mapping(
                    loaded.optimizer_state,
                    field="checkpoint.optimizer_state",
                ),
                rng_state=_mapping(
                    loaded.rng_state,
                    field="checkpoint.rng_state",
                ),
                completed_updates=prefix.completed_updates,
            )

        for update_index in range(prefix.completed_updates, len(schedule)):
            _require_live_source(preregistration)
            require_live_phase_a_update_admission(
                live_launch_capability,
                preregistration=preregistration,
                launch_authorization_digest=_sha256(
                    launch_authorization.get("exact_digest"),
                    field="launch_authorization.exact_digest",
                ),
                cell_id=resolved_cell,
                learner_index=resolved_index,
                stage="before_update",
                update_index=update_index,
            )
            model_state_sha256_before_update = recurrent_model_state_sha256(
                runner.model
            )
            expected_model_state_sha256_before_update = (
                prefix.terminal_model_state_sha256 or prefix.initial_model_state_sha256
            )
            if (
                expected_model_state_sha256_before_update is None
                or model_state_sha256_before_update
                != expected_model_state_sha256_before_update
            ):
                raise OpenEcologyPhaseAError(
                    "Phase A runner model is detached from its committed prefix"
                )
            update = runner.train_update(schedule[update_index])
            model_state_sha256_after_update = recurrent_model_state_sha256(runner.model)
            _validate_phase_a_update_transition(
                optimizer=asdict(update.optimizer),
                model_state_sha256_before_update=(model_state_sha256_before_update),
                model_state_sha256_after_update=(model_state_sha256_after_update),
            )
            _require_live_source(preregistration)
            require_live_phase_a_update_admission(
                live_launch_capability,
                preregistration=preregistration,
                launch_authorization_digest=_sha256(
                    launch_authorization.get("exact_digest"),
                    field="launch_authorization.exact_digest",
                ),
                cell_id=resolved_cell,
                learner_index=resolved_index,
                stage="before_update_commit",
                update_index=update_index,
            )
            update_payload = _phase_a_update_payload(
                update,
                run_contract=run_contract,
                previous_commit_exact_digest=(
                    prefix.commit_digests[-1] if prefix.commit_digests else None
                ),
                model=runner.model,
                model_state_sha256_before_update=(model_state_sha256_before_update),
            )
            checkpoint_state = runner.export_training_checkpoint_state()
            rng_state = dict(
                _mapping(
                    checkpoint_state.get("rng_state"),
                    field="checkpoint rng_state",
                )
            )
            rng_state.update(
                {
                    "phase_a_campaign_digest": preregistration["exact_digest"],
                    "phase_a_run_contract_digest": run_contract["exact_digest"],
                    "phase_a_update_exact_digest": update_payload["exact_digest"],
                    "phase_a_previous_commit_exact_digest": (
                        prefix.commit_digests[-1] if prefix.commit_digests else None
                    ),
                }
            )
            checkpoint = build_recurrent_training_crash_checkpoint(
                runner.model,
                optimizer_state=_mapping(
                    checkpoint_state.get("optimizer_state"),
                    field="checkpoint optimizer_state",
                ),
                rng_state=rng_state,
                optimizer_type="torch.optim.Adam",
                training_config=run_contract,
                seed_registry_digest=OPEN_ECOLOGY_CANONICAL_SHA256,
                source_commit=_mapping(
                    preregistration["source"],
                    field="source",
                )["commit"],  # type: ignore[arg-type]
                source_manifest_sha256=_mapping(
                    preregistration["source"],
                    field="source",
                )["manifest_sha256"],  # type: ignore[arg-type]
                learner_seed=learner_seed,
                completed_updates=runner.completed_update_count,
                run_id=run_contract["run_id"],  # type: ignore[arg-type]
            )
            write_phase_a_update_commit(
                run_directory,
                update_index=update_index,
                update_payload=update_payload,
                checkpoint=checkpoint,
                run_contract=run_contract,
                previous_commit_exact_digest=(
                    prefix.commit_digests[-1] if prefix.commit_digests else None
                ),
            )
            prefix = verify_phase_a_evidence_prefix(
                run_directory,
                run_contract=run_contract,
                schedule=schedule,
            )

        if prefix.completed_updates != OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT:
            raise OpenEcologyPhaseAError("Phase A terminal update prefix is incomplete")
        _require_live_source(preregistration)
        staging = run_directory / (
            f".terminal.pending-{os.getpid()}-{uuid.uuid4().hex}"
        )
        staging.mkdir(mode=0o700)
        try:
            staging_artifact_path = staging / "terminal-training-artifact.json"
            staging_terminal_path = staging / "terminal.json"
            staging_run_contract_path = staging / "run-contract.json"
            _write_atomic_json(staging_run_contract_path, run_contract)
            artifact = save_recurrent_artifact(
                staging_artifact_path,
                runner.model,
                training_config=asdict(ppo_config),
                seed_registry_digest=OPEN_ECOLOGY_CANONICAL_SHA256,
                source_commit=_mapping(
                    preregistration["source"],
                    field="source",
                )["commit"],  # type: ignore[arg-type]
                data_metadata={
                    "campaign_digest": preregistration["exact_digest"],
                    "run_contract_digest": run_contract["exact_digest"],
                    "environment_seed_role": OPEN_ECOLOGY_TRAINING_SEED_ROLE,
                    "environment_seed_indices": list(
                        range(
                            OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                            * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                        )
                    ),
                    "training_world_count": (
                        OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                        * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                    ),
                    "training_scenarios": ["broad"],
                    "fixture_names": [],
                    "validation_accessed": False,
                    "lockbox_accessed": False,
                },
                run_metadata={
                    "purpose": "preregistered_open_ecology_phase_a_training_cell",
                    "cell_id": resolved_cell,
                    "learner_index": resolved_index,
                    "run_id": run_contract["run_id"],
                    "terminal_commit_exact_digest": prefix.commit_digests[-1],
                    "source_manifest_sha256": _mapping(
                        preregistration["source"],
                        field="source",
                    )["manifest_sha256"],
                    "selection_evaluation_pending": True,
                    "exact_cpu_reevaluation_required": True,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                },
                learner_seed=learner_seed,
                learner_device=str(torch.device(device)),
            )
            loaded_artifact = load_recurrent_artifact(staging_artifact_path)
            if recurrent_model_state_sha256(
                loaded_artifact.model
            ) != recurrent_model_state_sha256(runner.model):
                raise OpenEcologyPhaseAError(
                    "terminal artifact CPU reconstruction changed model state"
                )
            terminal: dict[str, object] = {
                "schema_version": OPEN_ECOLOGY_PHASE_A_TERMINAL_SCHEMA_VERSION,
                "campaign_digest": preregistration["exact_digest"],
                "run_contract_digest": run_contract["exact_digest"],
                "run_id": run_contract["run_id"],
                "cell_id": resolved_cell,
                "learner_index": resolved_index,
                "learner_seed": learner_seed,
                "run_contract": {
                    "file": _file_reference(
                        staging_run_contract_path,
                        base=staging,
                    ),
                    "exact_digest": run_contract["exact_digest"],
                    "selection_binding_required": True,
                },
                "training": {
                    "completed_updates": prefix.completed_updates,
                    "training_world_count": (
                        OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                        * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                    ),
                    "commit_exact_digests": list(prefix.commit_digests),
                    "initial_model_state_sha256": (prefix.initial_model_state_sha256),
                    "terminal_model_state_sha256": recurrent_model_state_sha256(
                        runner.model
                    ),
                    "initial_to_terminal_model_changed": (
                        prefix.initial_model_state_sha256
                        != recurrent_model_state_sha256(runner.model)
                    ),
                    "cumulative_accepted_ppo_minibatches": (
                        prefix.cumulative_accepted_ppo_minibatches
                    ),
                    "cumulative_post_step_kl_rejected_steps": (
                        prefix.cumulative_post_step_kl_rejected_steps
                    ),
                },
                "artifact": {
                    "file": _file_reference(
                        staging_artifact_path,
                        base=staging,
                    ),
                    "artifact_sha256": artifact["artifact_sha256"],
                    "strict_cpu_reconstruction_verified": True,
                    "selection_evaluation_pending": True,
                },
                "lifecycle": {
                    "development_only": True,
                    "training_complete": True,
                    "selection_complete": False,
                    "cell_selection_authorized": False,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                    "validation_accessed": False,
                    "lockbox_accessed": False,
                },
            }
            terminal["exact_digest"] = stable_payload_digest(terminal)
            _write_atomic_json(staging_terminal_path, terminal)
            _fsync_directory(staging)
            _require_live_source(preregistration)
            require_live_phase_a_update_admission(
                live_launch_capability,
                preregistration=preregistration,
                launch_authorization_digest=_sha256(
                    launch_authorization.get("exact_digest"),
                    field="launch_authorization.exact_digest",
                ),
                cell_id=resolved_cell,
                learner_index=resolved_index,
                stage="before_terminal_publish",
            )
            source_identity = _phase_a_stat_identity(staging.lstat())
            entry_identity = _phase_a_terminal_entry_identity(staging)
            _publish_phase_a_staging_directory(
                run_directory,
                source_name=staging.name,
                destination_name=terminal_directory.name,
                expected_source_identity=source_identity,
            )
            if _phase_a_terminal_entry_identity(terminal_directory) != entry_identity:
                raise OpenEcologyPhaseAError(
                    "Phase A terminal identity changed during publication"
                )
        except BaseException:
            # Failed stages are evidence. Never remove a pathname that may have
            # been substituted by a racing same-host process.
            raise
        return _verify_phase_a_terminal(
            terminal_path,
            artifact_path=artifact_path,
            run_contract_path=run_contract_path,
            run_contract=run_contract,
            run_directory=run_directory,
            schedule=schedule,
        )


def write_phase_a_update_commit(
    run_directory: str | Path,
    *,
    update_index: int,
    update_payload: Mapping[str, object],
    checkpoint: Mapping[str, object],
    run_contract: Mapping[str, object],
    previous_commit_exact_digest: str | None,
) -> dict[str, object]:
    """Atomically publish one journal/checkpoint pair as the next hash-chain node."""

    run_root = Path(run_directory)
    index = _index(
        update_index,
        field="update_index",
        upper=OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT,
    )
    _validate_signed_payload(update_payload, field="Phase A update payload")
    if (
        update_payload.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION
        or update_payload.get("update_index") != index
        or update_payload.get("run_contract_digest") != run_contract.get("exact_digest")
        or update_payload.get("previous_commit_exact_digest")
        != previous_commit_exact_digest
    ):
        raise OpenEcologyPhaseAError("Phase A update payload binding drifted")
    _validate_phase_a_update_transition(
        optimizer=_mapping(
            update_payload.get("optimizer"),
            field="Phase A update payload optimizer",
        ),
        model_state_sha256_before_update=_sha256(
            update_payload.get("model_state_sha256_before_update"),
            field="Phase A update payload pre-update model state",
        ),
        model_state_sha256_after_update=_sha256(
            update_payload.get("model_state_sha256_after_update"),
            field="Phase A update payload post-update model state",
        ),
    )
    if previous_commit_exact_digest is None and update_payload.get(
        "model_state_sha256_before_update"
    ) != run_contract.get("initial_full_model_sha256"):
        raise OpenEcologyPhaseAError(
            "first Phase A update is detached from initialized model state"
        )
    if previous_commit_exact_digest is not None:
        _sha256(
            previous_commit_exact_digest,
            field="previous_commit_exact_digest",
        )
    updates_root = run_root / "updates"
    updates_root.mkdir(parents=True, exist_ok=True)
    final_directory = updates_root / f"update-{index:04d}"
    if os.path.lexists(final_directory):
        raise OpenEcologyPhaseAError("Phase A update commit already exists")
    staging = updates_root / (
        f".update-{index:04d}.pending-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    update_path = staging / "update.json"
    checkpoint_path = staging / "checkpoint.json"
    _write_atomic_json(update_path, update_payload)
    write_recurrent_training_crash_checkpoint(checkpoint_path, checkpoint)
    commit: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_COMMIT_SCHEMA_VERSION,
        "run_id": run_contract["run_id"],
        "run_contract_digest": run_contract["exact_digest"],
        "update_index": index,
        "completed_updates": index + 1,
        "previous_commit_exact_digest": previous_commit_exact_digest,
        "update": _file_reference(update_path, base=staging),
        "checkpoint": _file_reference(checkpoint_path, base=staging),
    }
    commit["exact_digest"] = stable_payload_digest(commit)
    _write_atomic_json(staging / "commit.json", commit)
    _fsync_directory(staging)
    source_identity = _phase_a_stat_identity(staging.lstat())
    entry_identity = _phase_a_update_entry_identity(staging)
    _publish_phase_a_staging_directory(
        updates_root,
        source_name=staging.name,
        destination_name=final_directory.name,
        expected_source_identity=source_identity,
    )
    if _phase_a_update_entry_identity(final_directory) != entry_identity:
        raise OpenEcologyPhaseAError(
            "Phase A update identity changed during commit publication"
        )
    return commit


def verify_phase_a_evidence_prefix(
    run_directory: str | Path,
    *,
    run_contract: Mapping[str, object],
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
) -> PhaseAEvidencePrefix:
    return _verify_phase_a_evidence_prefix(
        run_directory,
        run_contract=run_contract,
        schedule=schedule,
        ignored_update_entry_names=frozenset(),
    )


def _verify_phase_a_evidence_prefix(
    run_directory: str | Path,
    *,
    run_contract: Mapping[str, object],
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
    ignored_update_entry_names: frozenset[str],
) -> PhaseAEvidencePrefix:
    run_root = Path(run_directory)
    updates_root = run_root / "updates"
    initial_model_state_sha256 = _sha256(
        run_contract.get("initial_full_model_sha256"),
        field="run_contract.initial_full_model_sha256",
    )
    if not updates_root.exists():
        return PhaseAEvidencePrefix(
            completed_updates=0,
            commit_digests=(),
            terminal_checkpoint=None,
            initial_model_state_sha256=initial_model_state_sha256,
            terminal_model_state_sha256=None,
            cumulative_accepted_ppo_minibatches=0,
            cumulative_post_step_kl_rejected_steps=0,
        )
    if not updates_root.is_dir() or updates_root.is_symlink():
        raise OpenEcologyPhaseAError("Phase A updates path is not a real directory")
    entries = sorted(updates_root.iterdir(), key=lambda value: value.name)
    indices: list[int] = []
    for entry in entries:
        if entry.name in ignored_update_entry_names:
            continue
        match = _UPDATE_DIRECTORY_RE.fullmatch(entry.name)
        if match is None or not entry.is_dir() or entry.is_symlink():
            raise OpenEcologyPhaseAError(
                "Phase A evidence contains an uncommitted or unknown update entry"
            )
        indices.append(int(match.group(1)))
    if indices != list(range(len(indices))):
        raise OpenEcologyPhaseAError(
            "Phase A evidence prefix is not contiguous from update zero"
        )
    if len(indices) > len(schedule):
        raise OpenEcologyPhaseAError(
            "Phase A evidence prefix exceeds the preregistered schedule"
        )
    commit_digests: list[str] = []
    terminal_checkpoint: LoadedRecurrentTrainingCrashCheckpoint | None = None
    previous_digest: str | None = None
    expected_model_state_sha256_before_update = initial_model_state_sha256
    cumulative_accepted_ppo_minibatches = 0
    cumulative_post_step_kl_rejected_steps = 0
    for index in indices:
        directory = updates_root / f"update-{index:04d}"
        node = _verify_phase_a_evidence_node(
            directory,
            index=index,
            previous_commit_exact_digest=previous_digest,
            expected_model_state_sha256_before_update=(
                expected_model_state_sha256_before_update
            ),
            run_contract=run_contract,
            scheduled_tasks=schedule[index],
        )
        previous_digest = node.commit_digest
        commit_digests.append(previous_digest)
        terminal_checkpoint = node.checkpoint
        expected_model_state_sha256_before_update = node.model_state_sha256_after_update
        cumulative_accepted_ppo_minibatches += node.accepted_ppo_minibatches
        cumulative_post_step_kl_rejected_steps += node.post_step_kl_rejected_steps
    return PhaseAEvidencePrefix(
        completed_updates=len(indices),
        commit_digests=tuple(commit_digests),
        terminal_checkpoint=terminal_checkpoint,
        initial_model_state_sha256=initial_model_state_sha256,
        terminal_model_state_sha256=(
            expected_model_state_sha256_before_update if indices else None
        ),
        cumulative_accepted_ppo_minibatches=(cumulative_accepted_ppo_minibatches),
        cumulative_post_step_kl_rejected_steps=(cumulative_post_step_kl_rejected_steps),
    )


def _verify_phase_a_evidence_node(
    directory: Path,
    *,
    index: int,
    previous_commit_exact_digest: str | None,
    expected_model_state_sha256_before_update: str,
    run_contract: Mapping[str, object],
    scheduled_tasks: Sequence[PhaseAOpenEcologyRolloutTask],
) -> _PhaseAEvidenceNode:
    expected_files = {
        "update.json",
        "checkpoint.json",
        "commit.json",
    }
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or {path.name for path in directory.iterdir()} != expected_files
        or any(
            not (directory / name).is_file() or (directory / name).is_symlink()
            for name in expected_files
        )
    ):
        raise OpenEcologyPhaseAError(
            "Phase A update commit files are incomplete, unsafe, or surplus"
        )
    commit = _load_strict_json(directory / "commit.json")
    _require_exact_keys(
        commit,
        {
            "schema_version",
            "run_id",
            "run_contract_digest",
            "update_index",
            "completed_updates",
            "previous_commit_exact_digest",
            "update",
            "checkpoint",
            "exact_digest",
        },
        field="Phase A evidence commit",
    )
    _validate_signed_payload(commit, field="Phase A evidence commit")
    if (
        commit.get("schema_version") != OPEN_ECOLOGY_PHASE_A_COMMIT_SCHEMA_VERSION
        or commit.get("run_id") != run_contract.get("run_id")
        or commit.get("run_contract_digest") != run_contract.get("exact_digest")
        or commit.get("update_index") != index
        or commit.get("completed_updates") != index + 1
        or commit.get("previous_commit_exact_digest") != previous_commit_exact_digest
    ):
        raise OpenEcologyPhaseAError("Phase A update commit hash-chain binding drifted")
    _verify_file_reference(
        _mapping(commit.get("update"), field="commit.update"),
        path=directory / "update.json",
        base=directory,
    )
    _verify_file_reference(
        _mapping(commit.get("checkpoint"), field="commit.checkpoint"),
        path=directory / "checkpoint.json",
        base=directory,
    )
    update = _load_strict_json(directory / "update.json")
    _require_exact_keys(
        update,
        {
            "schema_version",
            "run_id",
            "run_contract_digest",
            "update_index",
            "previous_commit_exact_digest",
            "tasks",
            "rollout",
            "optimizer",
            "counterfactual_collection",
            "counterfactual_auxiliary",
            "model_state_sha256_before_update",
            "model_state_sha256_after_update",
            "exact_digest",
        },
        field="Phase A update",
    )
    _validate_signed_payload(update, field="Phase A update")
    if (
        update.get("schema_version") != OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION
        or update.get("run_id") != run_contract.get("run_id")
        or update.get("run_contract_digest") != run_contract.get("exact_digest")
        or update.get("update_index") != index
        or update.get("previous_commit_exact_digest") != previous_commit_exact_digest
        or update.get("tasks") != [asdict(task) for task in scheduled_tasks]
    ):
        raise OpenEcologyPhaseAError(
            "Phase A update journal detached from its exact schedule"
        )
    optimizer = _mapping(update.get("optimizer"), field="update.optimizer")
    before_update = _sha256(
        update.get("model_state_sha256_before_update"),
        field="update.model_state_sha256_before_update",
    )
    after_update = _sha256(
        update.get("model_state_sha256_after_update"),
        field="update.model_state_sha256_after_update",
    )
    if before_update != expected_model_state_sha256_before_update:
        raise OpenEcologyPhaseAError(
            "Phase A update model-state chain is not contiguous"
        )
    _validate_phase_a_update_transition(
        optimizer=optimizer,
        model_state_sha256_before_update=before_update,
        model_state_sha256_after_update=after_update,
    )
    loaded = load_recurrent_training_crash_checkpoint(directory / "checkpoint.json")
    checkpoint = loaded.checkpoint
    source = _mapping(checkpoint.get("source"), field="checkpoint.source")
    configuration = _mapping(
        checkpoint.get("configuration"),
        field="checkpoint.configuration",
    )
    progress = _mapping(checkpoint.get("progress"), field="checkpoint.progress")
    expected_source = _mapping(run_contract.get("source"), field="run.source")
    rng_state = _mapping(loaded.rng_state, field="checkpoint.rng_state")
    if (
        source.get("source_commit") != expected_source.get("commit")
        or source.get("source_manifest_sha256")
        != expected_source.get("manifest_sha256")
        or source.get("seed_registry_digest") != OPEN_ECOLOGY_CANONICAL_SHA256
        or configuration.get("training_config") != dict(run_contract)
        or progress.get("run_id") != run_contract.get("run_id")
        or progress.get("learner_seed") != run_contract.get("learner_seed")
        or progress.get("completed_updates") != index + 1
        or rng_state.get("phase_a_campaign_digest")
        != run_contract.get("campaign_digest")
        or rng_state.get("phase_a_run_contract_digest")
        != run_contract.get("exact_digest")
        or rng_state.get("phase_a_update_exact_digest") != update.get("exact_digest")
        or rng_state.get("phase_a_previous_commit_exact_digest")
        != previous_commit_exact_digest
        or recurrent_model_state_sha256(loaded.model)
        != update.get("model_state_sha256_after_update")
    ):
        raise OpenEcologyPhaseAError(
            "Phase A checkpoint detached from its evidence prefix"
        )
    return _PhaseAEvidenceNode(
        commit_digest=_sha256(
            commit.get("exact_digest"),
            field="commit.exact_digest",
        ),
        checkpoint=loaded,
        model_state_sha256_after_update=after_update,
        accepted_ppo_minibatches=int(optimizer["minibatch_count"]),
        post_step_kl_rejected_steps=int(
            optimizer.get("post_step_kl_rejected_step_count", 0)
        ),
    )


def preview_open_ecology_phase_a_cell_selection(
    preregistration: Mapping[str, object],
    learner_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compute a nonauthoritative selector preview from portable summaries.

    Mappings and SHA-256 values are not authority. This function therefore
    remains hard-blocked from selecting or launching Phase B even when every
    supplied learner summary is structurally valid. A future authoritative
    orchestrator must reopen all 16 canonical terminal bundles and reexecute
    their artifacts before applying the same frozen ranking rule.
    """

    validate_open_ecology_phase_a_preregistration(preregistration)
    if len(learner_evidence) != (
        len(OPEN_ECOLOGY_PHASE_A_CELL_ORDER) * OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
    ):
        raise OpenEcologyPhaseAError(
            "Phase A selection requires exactly 16 learner evidence reports"
        )
    reports: dict[tuple[str, int], Mapping[str, object]] = {}
    for report in learner_evidence:
        validate_open_ecology_phase_a_learner_evidence(
            report,
            preregistration=preregistration,
        )
        key = (
            _cell_id(report.get("cell_id")),
            _index(
                report.get("learner_index"),
                field="learner_index",
                upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
            ),
        )
        if key in reports:
            raise OpenEcologyPhaseAError("duplicate Phase A learner evidence report")
        reports[key] = report
    expected_keys = {
        (cell_id, learner_index)
        for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
    }
    if set(reports) != expected_keys:
        raise OpenEcologyPhaseAError("Phase A learner evidence matrix is incomplete")

    cell_summaries: dict[str, dict[str, object]] = {}
    eligible_cells: list[str] = []
    for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER:
        cell_reports = [
            reports[(cell_id, learner_index)]
            for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
        ]
        passing_count = sum(
            _mapping(report.get("gates"), field="gates").get("eligible") is True
            for report in cell_reports
        )
        returns = [
            _finite(
                _mapping(report.get("metrics"), field="metrics").get(
                    "median_per_decision_normalized_individual_return"
                ),
                field="normalized individual return",
            )
            for report in cell_reports
        ]
        value_rmses = [
            _finite(
                _mapping(report.get("metrics"), field="metrics").get(
                    "heldout_value_rmse"
                ),
                field="heldout value RMSE",
            )
            for report in cell_reports
        ]
        advantage_variances = [
            _finite(
                _mapping(report.get("metrics"), field="metrics").get(
                    "advantage_variance"
                ),
                field="advantage variance",
            )
            for report in cell_reports
        ]
        summary = {
            "passing_learner_count": passing_count,
            "eligible": passing_count >= 3,
            "median_normalized_return": median(returns),
            "median_heldout_value_rmse": median(value_rmses),
            "median_advantage_variance": median(advantage_variances),
        }
        cell_summaries[cell_id] = summary
        if summary["eligible"] is True:
            eligible_cells.append(cell_id)
    if not eligible_cells:
        raise OpenEcologyPhaseAError(
            "no Phase A cell is eligible; Phase B remains blocked"
        )

    leading_return = max(
        float(cell_summaries[cell_id]["median_normalized_return"])
        for cell_id in eligible_cells
    )
    tied = [
        cell_id
        for cell_id in eligible_cells
        if _within_one_percent(
            float(cell_summaries[cell_id]["median_normalized_return"]),
            leading_return,
        )
    ]
    best_rmse = min(
        float(cell_summaries[cell_id]["median_heldout_value_rmse"]) for cell_id in tied
    )
    tied = [
        cell_id
        for cell_id in tied
        if _within_one_percent(
            float(cell_summaries[cell_id]["median_heldout_value_rmse"]),
            best_rmse,
        )
    ]
    best_variance = min(
        float(cell_summaries[cell_id]["median_advantage_variance"]) for cell_id in tied
    )
    tied = [
        cell_id
        for cell_id in tied
        if _within_one_percent(
            float(cell_summaries[cell_id]["median_advantage_variance"]),
            best_variance,
        )
    ]
    selected = min(
        tied,
        key=lambda cell_id: (
            0
            if OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][1]
            == VALUE_SHARED_TRUNK_GRADIENT_STOP_V1
            else 1,
            0
            if OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][0] == CRITIC_GENOME_CONDITIONING_NONE
            else 1,
            OPEN_ECOLOGY_PHASE_A_CELL_ORDER.index(cell_id),
        ),
    )
    result: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_SELECTION_SCHEMA_VERSION,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "learner_evidence_exact_digests": [
            reports[(cell_id, learner_index)]["exact_digest"]
            for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
            for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
        ],
        "cell_summaries": cell_summaries,
        "provisional_selected_cell_id": selected,
        "provisional_selected_contract": {
            "critic_genome_conditioning": OPEN_ECOLOGY_PHASE_A_CELLS[selected][0],
            "value_shared_trunk_gradient": OPEN_ECOLOGY_PHASE_A_CELLS[selected][1],
        },
        "selected_cell_id": None,
        "selected_contract": None,
        "phase_b_authorized": False,
        "authorization_blocker": (
            "fresh_exact_cpu_open_ecology_evidence_producer_unavailable"
        ),
        "input_evidence_authority": (
            "caller_summary_only_not_artifact_trajectory_replay_causal_backed"
        ),
        "selection_rule": _mapping(
            preregistration.get("selection"),
            field="selection",
        )["selection_rule"],
        "lifecycle": {
            "development_only": True,
            "validation_accessed": False,
            "lockbox_accessed": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
        },
    }
    result["exact_digest"] = stable_payload_digest(result)
    return result


def validate_open_ecology_phase_a_learner_evidence(
    report: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
) -> None:
    _require_exact_keys(
        report,
        {
            "schema_version",
            "campaign_digest",
            "run_contract_digest",
            "cell_id",
            "learner_index",
            "learner_seed",
            "artifact_sha256",
            "terminal_authority",
            "evaluation",
            "causal_genome",
            "gates",
            "metrics",
            "lifecycle",
            "exact_digest",
        },
        field="Phase A learner evidence",
    )
    _validate_signed_payload(report, field="Phase A learner evidence")
    if report.get(
        "schema_version"
    ) != OPEN_ECOLOGY_PHASE_A_LEARNER_EVIDENCE_SCHEMA_VERSION or report.get(
        "campaign_digest"
    ) != preregistration.get("exact_digest"):
        raise OpenEcologyPhaseAError("Phase A learner evidence campaign drifted")
    cell_id = _cell_id(report.get("cell_id"))
    learner_index = _index(
        report.get("learner_index"),
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    if report.get("learner_seed") != _phase_a_learner_seeds()[learner_index]:
        raise OpenEcologyPhaseAError("Phase A learner evidence seed drifted")
    artifact_sha256 = _sha256(
        report.get("artifact_sha256"),
        field="artifact_sha256",
    )
    expected_run_contract = build_phase_a_run_contract(
        preregistration,
        cell_id=cell_id,
        learner_index=learner_index,
    )
    if report.get("run_contract_digest") != expected_run_contract["exact_digest"]:
        raise OpenEcologyPhaseAError("Phase A learner evidence run contract drifted")
    evaluation = _mapping(report.get("evaluation"), field="evaluation")
    _require_exact_keys(
        evaluation,
        {
            "environment_seed_role",
            "environment_seed_indices",
            "environment_seeds",
            "ticks_per_execution",
            "stochastic_tape_count",
            "argmax_diagnostic",
            "fixture_names",
            "exact_cpu_artifact_replay",
            "selection_report_schema_version",
            "selection_report_exact_digest",
            "artifact_file_sha256",
            "per_environment_then_per_learner_aggregation",
            "paired_return_aggregation",
            "authoritative_full_artifact_reexecution_verified",
            "authorization_verification_exact_digest",
            "initialized_baseline_model_state_sha256",
            "initialized_baseline_report_exact_digest",
            "initialized_baseline_execution_count",
            "producer_primary_world_run_count",
            "authorization_reexecution_world_run_count",
            "total_physical_world_run_count_through_authorization",
        },
        field="evaluation",
    )
    terminal_authority = _mapping(
        report.get("terminal_authority"),
        field="terminal_authority",
    )
    _require_exact_keys(
        terminal_authority,
        {
            "schema_version",
            "campaign_digest",
            "source_commit",
            "source_manifest_sha256",
            "run_id",
            "cell_id",
            "learner_index",
            "learner_seed",
            "terminal_logical_name",
            "terminal_exact_digest",
            "terminal_file_sha256",
            "final_prefix_commit_exact_digest",
            "terminal_checkpoint_model_state_sha256",
            "initial_model_state_sha256",
            "initial_to_terminal_model_changed",
            "cumulative_accepted_ppo_minibatches",
            "cumulative_post_step_kl_rejected_steps",
            "run_contract_logical_name",
            "run_contract_exact_digest",
            "run_contract_file_sha256",
            "artifact_logical_name",
            "artifact_sha256",
            "artifact_file_sha256",
            "selection_binding_required",
            "exact_digest",
        },
        field="terminal_authority",
    )
    _validate_signed_payload(
        terminal_authority,
        field="terminal_authority",
    )
    source = _mapping(
        preregistration.get("source"),
        field="source",
    )
    for field in (
        "terminal_exact_digest",
        "terminal_file_sha256",
        "final_prefix_commit_exact_digest",
        "terminal_checkpoint_model_state_sha256",
        "initial_model_state_sha256",
        "run_contract_exact_digest",
        "run_contract_file_sha256",
        "artifact_sha256",
        "artifact_file_sha256",
    ):
        _sha256(
            terminal_authority.get(field),
            field=f"terminal_authority.{field}",
        )
    if (
        terminal_authority.get("schema_version")
        != OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION
        or terminal_authority.get("campaign_digest")
        != preregistration.get("exact_digest")
        or terminal_authority.get("source_commit") != source.get("commit")
        or terminal_authority.get("source_manifest_sha256")
        != source.get("manifest_sha256")
        or terminal_authority.get("run_id") != expected_run_contract.get("run_id")
        or terminal_authority.get("cell_id") != cell_id
        or terminal_authority.get("learner_index") != learner_index
        or terminal_authority.get("learner_seed")
        != _phase_a_learner_seeds()[learner_index]
        or terminal_authority.get("terminal_logical_name") != "terminal.json"
        or terminal_authority.get("run_contract_logical_name") != "run-contract.json"
        or terminal_authority.get("artifact_logical_name")
        != "terminal-training-artifact.json"
        or terminal_authority.get("run_contract_exact_digest")
        != expected_run_contract.get("exact_digest")
        or terminal_authority.get("artifact_sha256") != artifact_sha256
        or terminal_authority.get("artifact_file_sha256")
        != evaluation.get("artifact_file_sha256")
        or terminal_authority.get("selection_binding_required") is not True
        or terminal_authority.get("initial_to_terminal_model_changed") is not True
        or terminal_authority.get("initial_model_state_sha256")
        == terminal_authority.get("terminal_checkpoint_model_state_sha256")
        or not isinstance(
            terminal_authority.get("cumulative_accepted_ppo_minibatches"),
            int,
        )
        or terminal_authority.get("cumulative_accepted_ppo_minibatches", 0)
        < OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
        or not isinstance(
            terminal_authority.get("cumulative_post_step_kl_rejected_steps"),
            int,
        )
        or terminal_authority.get("cumulative_post_step_kl_rejected_steps", -1) < 0
    ):
        raise OpenEcologyPhaseAError("Phase A learner terminal authority drifted")
    seed_contract = _mapping(
        preregistration.get("seed_contract"), field="seed_contract"
    )
    if (
        evaluation.get("environment_seed_role") != "open_ecology_selection"
        or evaluation.get("environment_seed_indices")
        != list(range(OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT))
        or evaluation.get("environment_seeds") != seed_contract.get("selection_seeds")
        or evaluation.get("ticks_per_execution") != OPEN_ECOLOGY_PHASE_A_SELECTION_TICKS
        or evaluation.get("stochastic_tape_count")
        != OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT
        or evaluation.get("argmax_diagnostic") is not True
        or evaluation.get("fixture_names") != []
        or evaluation.get("exact_cpu_artifact_replay") is not True
        or evaluation.get("selection_report_schema_version")
        != "mind_v3_open_ecology_selection_evidence_v3"
        or evaluation.get("authoritative_full_artifact_reexecution_verified")
        is not True
        or evaluation.get("initialized_baseline_execution_count")
        != OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
        * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT
        or evaluation.get("paired_return_aggregation")
        != OPEN_ECOLOGY_PHASE_A_PAIRED_RETURN_AGGREGATION
        or evaluation.get("per_environment_then_per_learner_aggregation") is not True
        or evaluation.get("producer_primary_world_run_count")
        != OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
        * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        or evaluation.get("authorization_reexecution_world_run_count")
        != OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
        * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        or evaluation.get("total_physical_world_run_count_through_authorization")
        != 2
        * OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
        * (2 * OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
    ):
        raise OpenEcologyPhaseAError("Phase A selection execution drifted")
    _sha256(
        evaluation.get("initialized_baseline_model_state_sha256"),
        field="evaluation.initialized_baseline_model_state_sha256",
    )
    _sha256(
        evaluation.get("initialized_baseline_report_exact_digest"),
        field="evaluation.initialized_baseline_report_exact_digest",
    )
    for field in (
        "selection_report_exact_digest",
        "authorization_verification_exact_digest",
        "artifact_file_sha256",
    ):
        _sha256(evaluation.get(field), field=f"evaluation.{field}")
    causal = _mapping(report.get("causal_genome"), field="causal_genome")
    if causal.get("state_count") != OPEN_ECOLOGY_PHASE_A_CAUSAL_STATE_COUNT:
        raise OpenEcologyPhaseAError("Phase A causal-genome sample count drifted")
    js_share = _finite(
        causal.get("original_vs_donor_js_share_at_least_0.01"),
        field="causal JS share",
    )
    mean_js = _finite(
        causal.get("mean_original_vs_donor_js"),
        field="causal mean JS",
    )
    p99_tv = _finite(
        causal.get("single_locus_total_variation_p99"),
        field="causal p99 TV",
    )
    causal_pass = js_share >= 0.10 and mean_js >= 0.002 and p99_tv <= 0.25
    gates = _mapping(report.get("gates"), field="gates")
    exact_gate_fields = {
        "finite_outputs",
        "exact_action_mask_legality",
        "exact_same_contract_replay",
        "no_heuristic_action_source",
        "no_global_action_collapse",
        "action_collapse_evidence_sufficient",
        "positive_paired_initialized_baseline_improvement",
        "causal_genome_use",
        "bounded_single_locus_perturbation",
        "eligible",
    }
    _require_exact_keys(gates, exact_gate_fields, field="gates")
    component_pass = all(
        gates.get(field) is True for field in exact_gate_fields if field != "eligible"
    )
    if (
        gates.get("causal_genome_use") is not (js_share >= 0.10 and mean_js >= 0.002)
        or gates.get("bounded_single_locus_perturbation") is not (p99_tv <= 0.25)
        or gates.get("eligible") is not (component_pass and causal_pass)
    ):
        raise OpenEcologyPhaseAError("Phase A learner gate derivation drifted")
    metrics = _mapping(report.get("metrics"), field="metrics")
    _require_exact_keys(
        metrics,
        {
            "median_per_decision_normalized_individual_return",
            "heldout_value_rmse",
            "advantage_variance",
            "initialized_baseline_median_normalized_return",
            "paired_median_normalized_return_improvement",
            "target_contract",
            "paired_return_aggregation",
        },
        field="metrics",
    )
    for field in (
        "median_per_decision_normalized_individual_return",
        "heldout_value_rmse",
        "advantage_variance",
        "initialized_baseline_median_normalized_return",
        "paired_median_normalized_return_improvement",
    ):
        _finite(metrics.get(field), field=f"metrics.{field}")
    if (
        metrics.get("target_contract") != OPEN_ECOLOGY_PHASE_A_SELECTION_TARGET_CONTRACT
        or metrics.get("paired_return_aggregation")
        != OPEN_ECOLOGY_PHASE_A_PAIRED_RETURN_AGGREGATION
    ):
        raise OpenEcologyPhaseAError("Phase A learner metric contract drifted")
    if gates.get("positive_paired_initialized_baseline_improvement") is not (
        _finite(
            metrics.get("paired_median_normalized_return_improvement"),
            field="paired median normalized-return improvement",
        )
        > 0.0
    ):
        raise OpenEcologyPhaseAError(
            "Phase A initialized-baseline eligibility gate drifted"
        )
    lifecycle = _mapping(report.get("lifecycle"), field="lifecycle")
    if (
        lifecycle.get("development_only") is not True
        or lifecycle.get("validation_accessed") is not False
        or lifecycle.get("lockbox_accessed") is not False
        or lifecycle.get("runtime_integration_authorized") is not False
        or lifecycle.get("promotion_authorized") is not False
    ):
        raise OpenEcologyPhaseAError("Phase A learner lifecycle drifted")


def _validate_phase_a_pipeline_benchmark(
    report: Mapping[str, object],
    *,
    expected_source_commit: str,
    expected_population_mode: str,
) -> dict[int, float]:
    _require_exact_keys(
        report,
        {
            "schema_version",
            "scope",
            "source",
            "runtime",
            "seed_access",
            "collector",
            "protocol",
            "determinism",
            "preregistered_gate",
            "cases",
        },
        field=f"Phase A {expected_population_mode} pipeline benchmark",
    )
    if report.get("schema_version") != OPEN_ECOLOGY_PHASE_A_BENCHMARK_SCHEMA_VERSION:
        raise OpenEcologyPhaseAError("Phase A pipeline benchmark schema drifted")
    source = _mapping(report.get("source"), field="benchmark.source")
    _require_exact_keys(
        source,
        {"commit_sha", "dirty", "capture_contract"},
        field="benchmark.source",
    )
    if (
        source.get("commit_sha") != expected_source_commit
        or source.get("dirty") is not False
        or source.get("capture_contract") != "git_head_plus_porcelain_dirty_flag_v1"
    ):
        raise OpenEcologyPhaseAError(
            "Phase A pipeline benchmark was not run on clean exact source"
        )
    scope = _mapping(report.get("scope"), field="benchmark.scope")
    expected_scope = {
        "measured_operation": "simulation_rollout_merge_gae_and_ppo_update",
        "world_construction_included": True,
        "world_ticks_included": True,
        "observation_encoding_included": True,
        "policy_action_sampling_included": True,
        "ordered_worker_merge_included": True,
        "gae_included": True,
        "ppo_update_included": True,
        "fixed_batch_collection_included": True,
        "artifact_serialization_included": False,
        "evaluation_included": False,
        "policy_promotion_authorized": False,
    }
    if dict(scope) != expected_scope:
        raise OpenEcologyPhaseAError("Phase A pipeline benchmark scope drifted")
    benchmark_seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE]
    expected_environment_seed_indices = list(
        range(OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT)
    )
    expected_seed_access = {
        "schema_version": OPEN_ECOLOGY_BENCHMARK_SEED_PROVENANCE_SCHEMA_VERSION,
        "seed_registry_contract": {
            "version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        },
        "environment_seed_roles": [OPEN_ECOLOGY_BENCHMARK_SEED_ROLE],
        "environment_seeds_by_role": {
            OPEN_ECOLOGY_BENCHMARK_SEED_ROLE: [
                benchmark_seeds[index] for index in expected_environment_seed_indices
            ],
        },
        "environment_seed_indices": expected_environment_seed_indices,
        "observed_environment_seed_count": (
            OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT
        ),
        "canonical_registry_membership_valid": True,
        "registry_ordered_non_reused_range": {
            "offset": 0,
            "count": OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT,
            "exclusive_stop": OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT,
        },
        "model_initialization": {
            "seed": benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX],
            "registry_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "registry_index": OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX,
        },
        "genome_stream": {
            "seed": benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX],
            "registry_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
            "registry_index": OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        },
        "genome_population_mode": expected_population_mode,
        "training_phase": OPEN_ECOLOGY_PHASE_A,
        "initial_agent_density_cycle": [OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS],
        "scientific_environment_seed_roles_accessed": [],
        "training_seeds_accessed": False,
        "selection_seeds_accessed": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
    }
    if report.get("seed_access") != expected_seed_access:
        raise OpenEcologyPhaseAError(
            "Phase A benchmark accessed a scientific or noncanonical seed role"
        )
    expected_collector = {
        "experiment_contract_version": (
            RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION
        ),
        "fixed_batch_enabled": True,
        "fixed_batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        "fixed_batch_runtime_contract": RecurrentFixedBatchRuntimeContract(
            batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        ).as_contract(),
    }
    if report.get("collector") != expected_collector:
        raise OpenEcologyPhaseAError(
            "Phase A benchmark did not measure the canonical fixed-batch collector"
        )
    protocol = _mapping(report.get("protocol"), field="benchmark.protocol")
    required_protocol = {
        "worker_counts": list(OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS),
        "repeats": OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS,
        "updates": 1,
        "worlds_per_update": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
        "rollout_ticks": OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS,
        "scenarios": ["broad"],
        "input_contract": "tokenized",
        "genome_conditioning": GENOME_CONDITIONING_ACTOR_FILM_V1,
        "genome_population_mode": expected_population_mode,
        "genome_stream_seed": benchmark_seeds[
            OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
        ],
        "training_phase": OPEN_ECOLOGY_PHASE_A,
        "encoder_size": 256,
        "hidden_size": 256,
        "recurrent_layers": 1,
        "learner_seed": benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX],
        "update_epochs": 4,
        "sequence_minibatch_size": 16,
        "tbptt_steps": 128,
        "burn_in_steps": 16,
        "fixed_batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        "scheduled_worlds": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
        "scheduled_world_ticks": (
            OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
            * (OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS + 1)
        ),
        "schedule_contract": (
            "canonical_open_ecology_operational_benchmark_phase_a_v1"
        ),
        "timing_boundary": "runner.run_only",
        "fresh_identically_seeded_runner_per_sample": True,
        "preregistered_gate_member": True,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise OpenEcologyPhaseAError(
                f"Phase A pipeline benchmark protocol drifted at {key}"
            )
    repeats = _positive_int(protocol.get("repeats"), field="benchmark.repeats")
    if repeats != OPEN_ECOLOGY_PHASE_A_BENCHMARK_REPEATS:
        raise OpenEcologyPhaseAError("Phase A benchmark repeat count drifted")
    device = protocol.get("device_resolved")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise OpenEcologyPhaseAError("Phase A pipeline benchmark was not CUDA")
    model = _mapping(protocol.get("model_config"), field="benchmark.model_config")
    if (
        model.get("encoder_size") != 256
        or model.get("hidden_size") != 256
        or model.get("recurrent_layers") != 1
        or model.get("genome_conditioning_mode") != GENOME_CONDITIONING_ACTOR_FILM_V1
    ):
        raise OpenEcologyPhaseAError("Phase A benchmark model contract drifted")
    ppo = _mapping(protocol.get("ppo_config"), field="benchmark.ppo_config")
    if dict(ppo) != asdict(
        _phase_a_ppo_config(
            learner_seed=benchmark_seeds[OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX]
        )
    ):
        raise OpenEcologyPhaseAError("Phase A benchmark PPO contract drifted")
    determinism = _mapping(
        report.get("determinism"),
        field="benchmark.determinism",
    )
    _require_exact_keys(
        determinism,
        {
            "cross_worker_model_state_match",
            "cross_worker_semantic_evidence_match",
            "final_model_state_sha256",
            "semantic_evidence_sha256",
        },
        field="benchmark.determinism",
    )
    model_sha = _sha256(
        determinism.get("final_model_state_sha256"),
        field="benchmark.final_model_state_sha256",
    )
    semantic_sha = _sha256(
        determinism.get("semantic_evidence_sha256"),
        field="benchmark.semantic_evidence_sha256",
    )
    if (
        determinism.get("cross_worker_model_state_match") is not True
        or determinism.get("cross_worker_semantic_evidence_match") is not True
    ):
        raise OpenEcologyPhaseAError(
            "Phase A worker topology changed deterministic evidence"
        )
    preregistered_gate = _mapping(
        report.get("preregistered_gate"),
        field="benchmark.preregistered_gate",
    )
    if dict(preregistered_gate) != {
        "member_shape_valid": True,
        "member_claimed": True,
        "population_mode": expected_population_mode,
        "required_population_modes": [
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        ],
        "complete_pair_required": True,
        "complete_gate_claimed": False,
    }:
        raise OpenEcologyPhaseAError(
            "Phase A benchmark is not a canonical preregistered gate member"
        )
    cases = _sequence(report.get("cases"), field="benchmark.cases")
    if len(cases) != len(OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS):
        raise OpenEcologyPhaseAError("Phase A benchmark worker matrix is incomplete")
    elapsed_by_workers: dict[int, float] = {}
    for expected_workers, raw_case in zip(
        OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS,
        cases,
        strict=True,
    ):
        case = _mapping(raw_case, field=f"benchmark.case.{expected_workers}")
        if case.get("rollout_workers") != expected_workers:
            raise OpenEcologyPhaseAError("Phase A benchmark case ordering drifted")
        samples = _sequence(
            case.get("samples"),
            field=f"benchmark.case.{expected_workers}.samples",
        )
        if len(samples) != repeats:
            raise OpenEcologyPhaseAError("Phase A benchmark repeat count drifted")
        sample_elapsed: list[int] = []
        for repeat_index, raw_sample in enumerate(samples):
            sample = _mapping(
                raw_sample,
                field=f"benchmark.case.{expected_workers}.sample.{repeat_index}",
            )
            elapsed_ns = _positive_int(
                sample.get("elapsed_ns"),
                field="benchmark.sample.elapsed_ns",
            )
            if (
                sample.get("repeat_index") != repeat_index
                or sample.get("total_worlds") != OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
                or _positive_int(
                    sample.get("total_transitions"),
                    field="benchmark.sample.total_transitions",
                )
                <= 0
                or sample.get("final_model_state_sha256") != model_sha
                or sample.get("semantic_evidence_sha256") != semantic_sha
                or sample.get("seed_access_sha256")
                != stable_payload_digest(expected_seed_access)
                or sample.get("collector_contract_sha256")
                != stable_payload_digest(expected_collector)
            ):
                raise OpenEcologyPhaseAError(
                    "Phase A benchmark sample evidence drifted"
                )
            _finite(
                sample.get("worlds_per_second"),
                field="benchmark.sample.worlds_per_second",
            )
            _finite(
                sample.get("transitions_per_second"),
                field="benchmark.sample.transitions_per_second",
            )
            sample_elapsed.append(elapsed_ns)
        observed_distribution = _mapping(
            case.get("elapsed_ns"),
            field=f"benchmark.case.{expected_workers}.elapsed_ns",
        )
        expected_median = float(median(sample_elapsed))
        if (
            _finite(
                observed_distribution.get("median"),
                field="benchmark.case.elapsed_ns.median",
            )
            != expected_median
        ):
            raise OpenEcologyPhaseAError(
                "Phase A benchmark elapsed distribution drifted"
            )
        elapsed_by_workers[expected_workers] = expected_median
    return elapsed_by_workers


def _build_phase_a_resource_projection(
    *,
    source_commit: str,
    elapsed_by_mode: Mapping[str, Mapping[int, float]],
    selected_workers: int,
    resource_envelope: Mapping[str, object],
) -> dict[str, object]:
    envelope = _json_clone(resource_envelope, field="resource_envelope")
    validate_open_ecology_phase_a_resource_envelope(
        envelope,
        expected_source_commit=source_commit,
    )
    maximum_wall_seconds = _positive_int(
        envelope["maximum_wall_seconds"],
        field="resource_envelope.maximum_wall_seconds",
    )
    worst_mode_median_ns = max(
        elapsed_by_mode[mode][selected_workers]
        for mode in (
            RecurrentGenomePopulationMode.HERITABLE.value,
            RecurrentGenomePopulationMode.ZERO_ALL.value,
        )
    )
    benchmark_world_ticks = OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE * (
        OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS + 1
    )
    conservative_world_ticks_per_second = benchmark_world_ticks / (
        worst_mode_median_ns / 1_000_000_000.0
    )
    phase_a_training_update_count = (
        OPEN_ECOLOGY_PHASE_A_MATRIX_TRAINING_WORLDS
        // OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
    )
    phase_b_training_update_count = (
        OPEN_ECOLOGY_PHASE_B_MATRIX_TRAINING_WORLDS
        // OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
    )
    training_update_count = (
        phase_a_training_update_count + phase_b_training_update_count
    )
    phase_a_training_world_ticks = OPEN_ECOLOGY_PHASE_A_MATRIX_TRAINING_WORLDS * (
        OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS + 1
    )
    phase_b_training_world_ticks = OPEN_ECOLOGY_PHASE_B_MATRIX_TRAINING_WORLDS * (
        OPEN_ECOLOGY_PHASE_B_ROLLOUT_TICKS + 1
    )
    training_world_ticks = phase_a_training_world_ticks + phase_b_training_world_ticks
    phase_a_training_seconds = (
        phase_a_training_update_count
        * (worst_mode_median_ns / 1_000_000_000.0)
        * OPEN_ECOLOGY_RESOURCE_PROJECTION_SAFETY_FACTOR
    )
    phase_b_training_seconds_proxy = (
        phase_b_training_world_ticks
        / conservative_world_ticks_per_second
        * OPEN_ECOLOGY_RESOURCE_PROJECTION_SAFETY_FACTOR
    )
    training_seconds_proxy = phase_a_training_seconds + phase_b_training_seconds_proxy
    selection_world_ticks = (
        OPEN_ECOLOGY_PHASE_A_PHYSICAL_SELECTION_EXECUTIONS
        * OPEN_ECOLOGY_PHASE_A_SELECTION_TICKS
        + OPEN_ECOLOGY_PHASE_B_PHYSICAL_SELECTION_EXECUTIONS * 2_000
    )
    selection_seconds_proxy = (
        selection_world_ticks
        / conservative_world_ticks_per_second
        * OPEN_ECOLOGY_RESOURCE_PROJECTION_SAFETY_FACTOR
    )
    projected_total_seconds_proxy = training_seconds_proxy + selection_seconds_proxy
    proxy_inside_recorded_resource_envelope = (
        projected_total_seconds_proxy <= maximum_wall_seconds
    )
    return {
        "schema_version": OPEN_ECOLOGY_PHASE_A_RESOURCE_PROJECTION_SCHEMA_VERSION,
        "method": (
            "phase_a_training_from_worst_h_or_z_full_update_rate_with_non_"
            "authoritative_phase_b_training_and_selection_tick_rate_proxies_v3"
        ),
        "selected_rollout_workers": selected_workers,
        "worst_mode_median_update_elapsed_ns": worst_mode_median_ns,
        "benchmark_world_ticks_per_update": benchmark_world_ticks,
        "conservative_world_ticks_per_second": (conservative_world_ticks_per_second),
        "projection_safety_factor": (OPEN_ECOLOGY_RESOURCE_PROJECTION_SAFETY_FACTOR),
        "training_worlds": OPEN_ECOLOGY_PHASE_A_ALL_TRAINING_WORLDS,
        "phase_a_training_update_count": phase_a_training_update_count,
        "phase_b_training_update_count": phase_b_training_update_count,
        "training_update_count": training_update_count,
        "phase_a_training_world_ticks": phase_a_training_world_ticks,
        "phase_b_training_world_ticks": phase_b_training_world_ticks,
        "training_world_ticks": training_world_ticks,
        "phase_a_training_rate_source": ("measured_phase_a_fixed_density_full_update"),
        "projected_phase_a_training_seconds": phase_a_training_seconds,
        "phase_a_training_projection_authoritative": True,
        "phase_b_training_rate_source": (
            "phase_a_training_tick_rate_proxy_non_authoritative"
        ),
        "projected_phase_b_training_seconds_proxy": (phase_b_training_seconds_proxy),
        "projected_training_seconds_proxy": training_seconds_proxy,
        "phase_b_training_projection_authoritative": False,
        "phase_b_mixed_density_full_update_benchmark_required": True,
        "phase_a_primary_selection_executions": (
            OPEN_ECOLOGY_PHASE_A_PRIMARY_SELECTION_EXECUTIONS
        ),
        "phase_a_replay_selection_executions": (
            OPEN_ECOLOGY_PHASE_A_REPLAY_SELECTION_EXECUTIONS
        ),
        "phase_a_physical_selection_executions": (
            OPEN_ECOLOGY_PHASE_A_PHYSICAL_SELECTION_EXECUTIONS
        ),
        "phase_b_primary_selection_executions": (
            OPEN_ECOLOGY_PHASE_B_PRIMARY_SELECTION_EXECUTIONS
        ),
        "phase_b_replay_selection_executions": (
            OPEN_ECOLOGY_PHASE_B_REPLAY_SELECTION_EXECUTIONS
        ),
        "phase_b_physical_selection_executions": (
            OPEN_ECOLOGY_PHASE_B_PHYSICAL_SELECTION_EXECUTIONS
        ),
        "selection_world_ticks": selection_world_ticks,
        "selection_rate_source": ("training_update_tick_rate_proxy_non_authoritative"),
        "projected_selection_seconds_proxy": selection_seconds_proxy,
        "projected_total_seconds_proxy": projected_total_seconds_proxy,
        "proxy_inside_recorded_resource_envelope": (
            proxy_inside_recorded_resource_envelope
        ),
        "selection_projection_authoritative": False,
        "long_horizon_selection_benchmark_required": True,
        "resource_envelope": envelope,
        "inside_recorded_resource_envelope": False,
    }


def _phase_a_readiness_assertions(dependency_id: str) -> dict[str, object]:
    assertions: dict[str, dict[str, object]] = {
        "readiness_dependency_01": {
            "cross_surface_field_order_and_token_count_proved": True,
            "communication_self_echo_excluded": True,
            "receiver_projection_schema_version": (
                "foundation_communication_receiver_projection_v1"
            ),
            "receiver_observation_policy": (
                "exclude_receiver_own_communication_emissions_across_local_patch_v1"
            ),
            "global_reporting_policy": "include_all_emitters_v1",
        },
        "readiness_dependency_02": {
            "runtime_genome_lineage_and_zero_control_proved": True,
            "missing_genome_fails_closed": True,
            "heuristic_action_fallback_absent": True,
        },
        "readiness_dependency_03": {
            "four_critic_gradient_cells_proved": True,
            "phase_density_schedules_proved": True,
            "stop_gradient_probe_proved": True,
        },
        "readiness_dependency_04": {
            "fixed_shape_batching_same_contract_repeatability_proved": True,
            "phase_a_shape_scalar_behavior_regression_proved": True,
            "ordered_merge_and_semantic_evidence_equivalence_proved": True,
            "measured_topology_gate_required": True,
        },
        "readiness_dependency_05": {
            "persistent_island_50000_tick_runner_proved": True,
            "frozen_backbone_and_nonreset_world_proved": True,
        },
        "readiness_dependency_06": {
            "all_seven_checkpoint_envelopes_proved": True,
            "uninterrupted_vs_resume_equivalence_proved": True,
        },
        "readiness_dependency_07": {
            "bounded_rotating_writer_proved": True,
            "declared_event_coverage_proved": True,
        },
        "readiness_dependency_08": {
            "frozen_state_causal_evaluators_proved": True,
            "pseudo_replication_and_self_echo_rejections_proved": True,
            "capture_noninterference_exact_reexecution_proved": True,
            "scientific_selection_seed_access_absent": True,
        },
        "readiness_dependency_09": {
            "machine_preregistration_roundtrip_proved": True,
            "dirty_unknown_stale_inputs_fail_closed": True,
        },
        "readiness_dependency_10": {
            "exact_clean_phase_a_training_host_path_proved": True,
            "every_intended_phase_a_training_host_class_tested": True,
            "single_homogeneous_host_class_allowed": True,
            "torch_ci_lane_nonoptional": True,
        },
    }
    try:
        return assertions[dependency_id]
    except KeyError as error:
        raise OpenEcologyPhaseAError(
            f"unknown Phase A readiness dependency {dependency_id!r}"
        ) from error


def _phase_a_ppo_config(*, learner_seed: int) -> RecurrentPPOConfig:
    return RecurrentPPOConfig(
        learning_rate=0.0002,
        adam_epsilon=1.0e-8,
        gamma=0.997,
        gae_lambda=0.97,
        policy_clip_range=0.15,
        value_clip_range=0.20,
        value_loss_coefficient=0.50,
        entropy_coefficient=0.02,
        update_epochs=4,
        sequence_minibatch_size=16,
        tbptt_steps=128,
        burn_in_steps=16,
        max_gradient_norm=0.50,
        normalize_advantages=True,
        advantage_epsilon=1.0e-8,
        target_kl=0.02,
        learner_seed=learner_seed,
        feed_forward_history_ablation=False,
        world_balanced_loss=True,
    )


def _phase_a_policy_sampling_identity(
    *,
    learner_index: int,
    learner_seed: int,
    genome_stream_seed_index: int,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
) -> str:
    return (
        f"{OPEN_ECOLOGY_PHASE_A_POLICY_SAMPLING_NAMESPACE}|"
        f"registry={OPEN_ECOLOGY_CANONICAL_SHA256}|phase=phase_a|"
        f"learner_index={learner_index:02d}|"
        f"learner={learner_seed}|genome_stream_index="
        f"{genome_stream_seed_index:02d}|"
        f"genome_mode={RecurrentGenomePopulationMode.HERITABLE.value}|"
        f"update={update_index:04d}|world={world_index:06d}|"
        f"environment_index={environment_seed_index:03d}|"
        f"tape=training_rollout_v1|"
        f"initial_agents={OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS}"
    )


def _phase_a_genome_world_identity(
    *,
    learner_index: int,
    learner_seed: int,
    genome_stream_seed_index: int,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
    environment_seed: int,
) -> str:
    return (
        f"{OPEN_ECOLOGY_PHASE_A_GENOME_WORLD_IDENTITY_NAMESPACE}|"
        f"registry={OPEN_ECOLOGY_CANONICAL_SHA256}|phase=phase_a|"
        f"learner_index={learner_index:02d}|learner={learner_seed}|"
        f"genome_stream_index={genome_stream_seed_index:02d}|"
        f"genome_mode={RecurrentGenomePopulationMode.HERITABLE.value}|"
        f"update={update_index:04d}|world={world_index:06d}|"
        f"environment_index={environment_seed_index:03d}|"
        f"environment_seed={environment_seed}|"
        f"initial_agents={OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS}"
    )


def _phase_a_task_id(
    *,
    cell_id: str,
    learner_index: int,
    learner_seed: int,
    update_index: int,
    world_index: int,
    environment_seed_index: int,
    environment_seed: int,
) -> str:
    return (
        f"open-ecology-phase-a-{cell_id.lower()}-learner-{learner_index:02d}-"
        f"{learner_seed}-update-{update_index:04d}-world-{world_index:06d}-"
        f"environment-{environment_seed_index:03d}-seed-{environment_seed}-"
        f"density-{OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS}"
    )


def _phase_a_update_payload(
    update: RecurrentTrainingUpdateResult,
    *,
    run_contract: Mapping[str, object],
    previous_commit_exact_digest: str | None,
    model: PublicRecurrentActorCritic,
    model_state_sha256_before_update: str,
) -> dict[str, object]:
    if update.counterfactual_collection is not None or update.counterfactual_auxiliary:
        raise OpenEcologyPhaseAError(
            "Phase A cannot consume counterfactual auxiliary training"
        )
    model_state_sha256_after_update = recurrent_model_state_sha256(model)
    optimizer = asdict(update.optimizer)
    _validate_phase_a_update_transition(
        optimizer=optimizer,
        model_state_sha256_before_update=model_state_sha256_before_update,
        model_state_sha256_after_update=model_state_sha256_after_update,
    )
    payload: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION,
        "run_id": run_contract["run_id"],
        "run_contract_digest": run_contract["exact_digest"],
        "update_index": update.update_index,
        "previous_commit_exact_digest": previous_commit_exact_digest,
        "tasks": [asdict(task) for task in update.tasks],
        "rollout": asdict(update.rollout),
        "optimizer": optimizer,
        "counterfactual_collection": None,
        "counterfactual_auxiliary": None,
        "model_state_sha256_before_update": model_state_sha256_before_update,
        "model_state_sha256_after_update": model_state_sha256_after_update,
    }
    payload["exact_digest"] = stable_payload_digest(payload)
    return payload


def _validate_phase_a_update_transition(
    *,
    optimizer: Mapping[str, object],
    model_state_sha256_before_update: str,
    model_state_sha256_after_update: str,
) -> None:
    """Fail closed before evidence publication when one update learned nothing.

    A later KL-rejected minibatch is valid mixed evidence only when at least
    one earlier minibatch was accepted and the full model state actually
    changed. The optimizer diagnostics remain unmodified in the update journal.
    """

    before = _sha256(
        model_state_sha256_before_update,
        field="model_state_sha256_before_update",
    )
    after = _sha256(
        model_state_sha256_after_update,
        field="model_state_sha256_after_update",
    )
    accepted = optimizer.get("minibatch_count")
    if isinstance(accepted, bool) or not isinstance(accepted, int) or accepted <= 0:
        raise OpenEcologyPhaseAError(
            "Phase A update committed no accepted PPO minibatch"
        )
    parameter_delta = optimizer.get("parameter_delta_l2")
    if (
        isinstance(parameter_delta, bool)
        or not isinstance(parameter_delta, (int, float))
        or not math.isfinite(float(parameter_delta))
        or float(parameter_delta) <= 0.0
    ):
        raise OpenEcologyPhaseAError(
            "Phase A accepted PPO minibatches produced no positive model delta"
        )
    if before == after:
        raise OpenEcologyPhaseAError(
            "Phase A accepted PPO minibatches left the model state unchanged"
        )
    required_kl_fields = {
        "post_step_kl_audit_count",
        "post_step_kl_rejected_step_count",
        "post_step_kl_rollback_performed",
        "post_step_kl_rejection_reason",
    }
    if not required_kl_fields.issubset(optimizer):
        raise OpenEcologyPhaseAError(
            "Phase A optimizer omitted post-step KL audit evidence"
        )
    audit_count = optimizer.get("post_step_kl_audit_count")
    rejected = optimizer.get("post_step_kl_rejected_step_count")
    if (
        isinstance(audit_count, bool)
        or not isinstance(audit_count, int)
        or audit_count < 0
    ):
        raise OpenEcologyPhaseAError("Phase A post-step KL audit count is invalid")
    if isinstance(rejected, bool) or not isinstance(rejected, int) or rejected < 0:
        raise OpenEcologyPhaseAError(
            "Phase A post-step KL rejection evidence is invalid"
        )
    if audit_count != accepted + rejected:
        raise OpenEcologyPhaseAError(
            "Phase A post-step KL audit count must equal accepted plus rejected "
            "optimizer steps"
        )
    rollback = optimizer.get("post_step_kl_rollback_performed")
    reason = optimizer.get("post_step_kl_rejection_reason")
    if (
        rejected > audit_count
        or rollback is not (rejected > 0)
        or (
            rejected > 0
            and (not isinstance(reason, str) or not reason or reason != reason.strip())
        )
        or (rejected == 0 and reason is not None)
    ):
        raise OpenEcologyPhaseAError(
            "Phase A mixed post-step KL rejection evidence is inconsistent"
        )


def _verify_phase_a_terminal(
    terminal_path: Path,
    *,
    artifact_path: Path,
    run_contract_path: Path,
    run_contract: Mapping[str, object],
    run_directory: Path,
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
) -> dict[str, object]:
    terminal = _load_strict_json(terminal_path)
    _require_exact_keys(
        terminal,
        {
            "schema_version",
            "campaign_digest",
            "run_contract_digest",
            "run_id",
            "cell_id",
            "learner_index",
            "learner_seed",
            "run_contract",
            "training",
            "artifact",
            "lifecycle",
            "exact_digest",
        },
        field="Phase A terminal",
    )
    _validate_signed_payload(terminal, field="Phase A terminal")
    if (
        terminal.get("schema_version") != OPEN_ECOLOGY_PHASE_A_TERMINAL_SCHEMA_VERSION
        or terminal.get("campaign_digest") != run_contract.get("campaign_digest")
        or terminal.get("run_contract_digest") != run_contract.get("exact_digest")
        or terminal.get("run_id") != run_contract.get("run_id")
        or terminal.get("cell_id") != run_contract.get("cell_id")
        or terminal.get("learner_index") != run_contract.get("learner_index")
        or terminal.get("learner_seed") != run_contract.get("learner_seed")
    ):
        raise OpenEcologyPhaseAError("Phase A terminal record drifted")
    run_contract_reference = _mapping(
        terminal.get("run_contract"),
        field="terminal.run_contract",
    )
    _require_exact_keys(
        run_contract_reference,
        {
            "file",
            "exact_digest",
            "selection_binding_required",
        },
        field="terminal.run_contract",
    )
    if (
        run_contract_reference.get("exact_digest") != run_contract.get("exact_digest")
        or run_contract_reference.get("selection_binding_required") is not True
    ):
        raise OpenEcologyPhaseAError("Phase A terminal run contract claim drifted")
    _verify_file_reference(
        _mapping(
            run_contract_reference.get("file"),
            field="terminal.run_contract.file",
        ),
        path=run_contract_path,
        base=terminal_path.parent,
    )
    if _load_strict_json(run_contract_path) != dict(run_contract):
        raise OpenEcologyPhaseAError("Phase A terminal run contract bytes drifted")
    prefix = verify_phase_a_evidence_prefix(
        run_directory,
        run_contract=run_contract,
        schedule=schedule,
    )
    if prefix.completed_updates != OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT:
        raise OpenEcologyPhaseAError("Phase A terminal evidence prefix is incomplete")
    if prefix.terminal_checkpoint is None:
        raise OpenEcologyPhaseAError(
            "Phase A terminal evidence prefix lacks its final checkpoint"
        )
    training = _mapping(terminal.get("training"), field="terminal.training")
    _require_exact_keys(
        training,
        {
            "completed_updates",
            "training_world_count",
            "commit_exact_digests",
            "initial_model_state_sha256",
            "terminal_model_state_sha256",
            "initial_to_terminal_model_changed",
            "cumulative_accepted_ppo_minibatches",
            "cumulative_post_step_kl_rejected_steps",
        },
        field="terminal.training",
    )
    if (
        training.get("completed_updates") != prefix.completed_updates
        or training.get("training_world_count")
        != (OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE)
        or training.get("commit_exact_digests") != list(prefix.commit_digests)
        or training.get("initial_model_state_sha256")
        != prefix.initial_model_state_sha256
        or training.get("terminal_model_state_sha256")
        != prefix.terminal_model_state_sha256
        or training.get("initial_to_terminal_model_changed") is not True
        or training.get("cumulative_accepted_ppo_minibatches")
        != prefix.cumulative_accepted_ppo_minibatches
        or type(training.get("cumulative_accepted_ppo_minibatches")) is not int
        or training.get("cumulative_post_step_kl_rejected_steps")
        != prefix.cumulative_post_step_kl_rejected_steps
        or type(training.get("cumulative_post_step_kl_rejected_steps")) is not int
        or prefix.cumulative_accepted_ppo_minibatches
        < OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
    ):
        raise OpenEcologyPhaseAError(
            "Phase A terminal training evidence does not match its prefix"
        )
    _sha256(
        training.get("initial_model_state_sha256"),
        field="terminal.training.initial_model_state_sha256",
    )
    _sha256(
        training.get("terminal_model_state_sha256"),
        field="terminal.training.terminal_model_state_sha256",
    )
    checkpoint_model_sha256 = recurrent_model_state_sha256(
        prefix.terminal_checkpoint.model
    )
    if training.get("terminal_model_state_sha256") != checkpoint_model_sha256:
        raise OpenEcologyPhaseAError(
            "Phase A terminal model differs from the final committed checkpoint"
        )
    if training.get("initial_model_state_sha256") == checkpoint_model_sha256:
        raise OpenEcologyPhaseAError(
            "Phase A terminal model did not change from initialization"
        )
    artifact_reference = _mapping(terminal.get("artifact"), field="terminal.artifact")
    _require_exact_keys(
        artifact_reference,
        {
            "file",
            "artifact_sha256",
            "strict_cpu_reconstruction_verified",
            "selection_evaluation_pending",
        },
        field="terminal.artifact",
    )
    if (
        artifact_reference.get("strict_cpu_reconstruction_verified") is not True
        or artifact_reference.get("selection_evaluation_pending") is not True
    ):
        raise OpenEcologyPhaseAError("Phase A terminal artifact claim drifted")
    _verify_file_reference(
        _mapping(artifact_reference.get("file"), field="terminal.artifact.file"),
        path=artifact_path,
        base=terminal_path.parent,
    )
    loaded = load_recurrent_artifact(artifact_path)
    if artifact_reference.get("artifact_sha256") != loaded.artifact.get(
        "artifact_sha256"
    ) or recurrent_model_state_sha256(loaded.model) != training.get(
        "terminal_model_state_sha256"
    ):
        raise OpenEcologyPhaseAError("Phase A terminal artifact evidence drifted")
    provenance = _mapping(
        loaded.artifact.get("provenance"),
        field="terminal artifact provenance",
    )
    expected_source = _mapping(run_contract.get("source"), field="run.source")
    expected_data_metadata = {
        "campaign_digest": run_contract["campaign_digest"],
        "run_contract_digest": run_contract["exact_digest"],
        "environment_seed_role": OPEN_ECOLOGY_TRAINING_SEED_ROLE,
        "environment_seed_indices": list(
            range(
                OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT
                * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
            )
        ),
        "training_world_count": (
            OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
        ),
        "training_scenarios": ["broad"],
        "fixture_names": [],
        "validation_accessed": False,
        "lockbox_accessed": False,
    }
    expected_run_metadata = {
        "purpose": "preregistered_open_ecology_phase_a_training_cell",
        "cell_id": run_contract["cell_id"],
        "learner_index": run_contract["learner_index"],
        "run_id": run_contract["run_id"],
        "terminal_commit_exact_digest": prefix.commit_digests[-1],
        "source_manifest_sha256": expected_source["manifest_sha256"],
        "selection_evaluation_pending": True,
        "exact_cpu_reevaluation_required": True,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
    }
    if (
        provenance.get("training_config") != run_contract.get("ppo")
        or provenance.get("seed_registry_digest") != OPEN_ECOLOGY_CANONICAL_SHA256
        or provenance.get("source_commit") != expected_source.get("commit")
        or provenance.get("learner_seed") != run_contract.get("learner_seed")
        or provenance.get("data_metadata") != expected_data_metadata
        or provenance.get("run_metadata") != expected_run_metadata
    ):
        raise OpenEcologyPhaseAError(
            "Phase A terminal artifact provenance detached from its evidence prefix"
        )
    lifecycle = _mapping(terminal.get("lifecycle"), field="terminal.lifecycle")
    if dict(lifecycle) != {
        "development_only": True,
        "training_complete": True,
        "selection_complete": False,
        "cell_selection_authorized": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
        "validation_accessed": False,
        "lockbox_accessed": False,
    }:
        raise OpenEcologyPhaseAError("Phase A terminal lifecycle drifted")
    return terminal


def validate_open_ecology_phase_a_source_observation(
    preregistration: Mapping[str, object],
    *,
    observed_commit: object,
    observed_status_porcelain: object,
    observed_manifest_sha256: object,
    observed_preregistration_sha256: object,
) -> None:
    """Validate one explicit source observation against the sealed campaign."""

    source = _mapping(preregistration.get("source"), field="source")
    expected_commit = _git_sha(source.get("commit"))
    expected_manifest = _sha256(
        source.get("manifest_sha256"),
        field="source.manifest_sha256",
    )
    commit = _git_sha(observed_commit)
    if not isinstance(observed_status_porcelain, str):
        raise OpenEcologyPhaseAError(
            "observed_status_porcelain must be an exact string"
        )
    manifest = _sha256(
        observed_manifest_sha256,
        field="observed_manifest_sha256",
    )
    preregistration_sha256 = _sha256(
        observed_preregistration_sha256,
        field="observed_preregistration_sha256",
    )
    if commit != expected_commit or observed_status_porcelain.strip():
        raise OpenEcologyPhaseAError(
            "Phase A requires the exact clean preregistered Git source"
        )
    if manifest != expected_manifest:
        raise OpenEcologyPhaseAError("Phase A runtime source manifest drifted")
    if preregistration_sha256 != OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256:
        raise OpenEcologyPhaseAError("sealed Phase A preregistration bytes drifted")


def _require_live_source(preregistration: Mapping[str, object]) -> None:
    source = _mapping(preregistration.get("source"), field="source")
    expected_manifest = _sha256(
        source.get("manifest_sha256"),
        field="source.manifest_sha256",
    )
    try:
        git_authority = discover_pinned_git_executable()
        observed_commit = run_pinned_git(
            git_authority,
            repository_root=_REPOSITORY_ROOT,
            arguments=("rev-parse", "HEAD"),
        )
        status = run_pinned_git(
            git_authority,
            repository_root=_REPOSITORY_ROOT,
            arguments=("status", "--porcelain", "--untracked-files=all"),
        )
    except OpenEcologyGitAuthorityError as error:
        raise OpenEcologyPhaseAError("failed to inspect Phase A Git source") from error
    validate_open_ecology_phase_a_source_observation(
        preregistration,
        observed_commit=observed_commit,
        observed_status_porcelain=status,
        observed_manifest_sha256=expected_manifest,
        observed_preregistration_sha256=(OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256),
    )
    prereg_path = _REPOSITORY_ROOT / OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH
    prior_amendment_path = (
        _REPOSITORY_ROOT / OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH
    )
    amendment_path = _REPOSITORY_ROOT / OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH
    observed_manifest = source_file_hash_manifest(_REPOSITORY_ROOT)["aggregate_sha256"]
    try:
        final_commit = run_pinned_git(
            git_authority,
            repository_root=_REPOSITORY_ROOT,
            arguments=("rev-parse", "HEAD"),
        )
        final_status = run_pinned_git(
            git_authority,
            repository_root=_REPOSITORY_ROOT,
            arguments=("status", "--porcelain", "--untracked-files=all"),
        )
    except OpenEcologyGitAuthorityError as error:
        raise OpenEcologyPhaseAError(
            "failed to reinspect Phase A Git source"
        ) from error
    if final_commit != observed_commit or final_status != status:
        raise OpenEcologyPhaseAError(
            "Phase A Git source changed during source-manifest verification"
        )
    validate_open_ecology_phase_a_source_observation(
        preregistration,
        observed_commit=final_commit,
        observed_status_porcelain=final_status,
        observed_manifest_sha256=observed_manifest,
        observed_preregistration_sha256=_file_sha256(prereg_path),
    )
    if (
        _file_sha256(prior_amendment_path)
        != OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256
    ):
        raise OpenEcologyPhaseAError(
            "sealed prior open-ecology launch-authority amendment bytes drifted"
        )
    if _file_sha256(amendment_path) != OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256:
        raise OpenEcologyPhaseAError(
            "sealed open-ecology launch-authority amendment bytes drifted"
        )


def _require_live_runtime(
    preregistration: Mapping[str, object],
    *,
    device: torch.device | str,
) -> None:
    expected = _mapping(
        preregistration.get("runtime_contract"),
        field="runtime_contract",
    )
    observed = build_open_ecology_phase_a_runtime_contract(
        device=device,
        rollout_workers=_positive_int(
            expected.get("rollout_workers"),
            field="runtime.rollout_workers",
        ),
    )
    if observed != dict(expected):
        raise OpenEcologyPhaseAError(
            "Phase A live runtime differs from the preregistered contract"
        )


def _require_launch_dependencies() -> None:
    treatment = OpenEcologyBroadWorldTreatment(
        initial_agents=OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS
    )
    dependencies = treatment.launch_dependencies
    if (
        dependencies.self_echo_exclusion_implemented is not True
        or dependencies.self_echo_exclusion_required_before_authoritative_launch
        is not True
    ):
        raise OpenEcologyPhaseAError(
            "Phase A launch remains blocked until communication self-echo "
            "exclusion is implemented and contract-bound"
        )


def _common_parameter_sha256(model: PublicRecurrentActorCritic) -> str:
    tensors: dict[str, object] = {}
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(_COMMON_PARAMETER_EXCLUSIONS):
            continue
        value = tensor.detach().to(device="cpu").contiguous()
        tensors[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest(),
        }
    return stable_payload_digest(tensors)


def _schedule_sha256(
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
) -> str:
    return stable_payload_digest(
        [[asdict(task) for task in tasks] for tasks in schedule]
    )


def _phase_a_learner_seeds() -> tuple[int, ...]:
    return tuple(
        OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
            :OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT
        ]
    )


def _cell_id(value: object) -> str:
    if not isinstance(value, str) or value not in OPEN_ECOLOGY_PHASE_A_CELLS:
        raise OpenEcologyPhaseAError(
            f"Phase A cell must be one of {OPEN_ECOLOGY_PHASE_A_CELL_ORDER}"
        )
    return value


def _index(value: object, *, field: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise OpenEcologyPhaseAError(f"{field} must be an integer in [0, {upper})")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyPhaseAError(f"{field} must be a positive integer")
    return value


def _git_sha(value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise OpenEcologyPhaseAError("source commit must be a full lowercase Git SHA")
    return value


def _sha256(value: object, *, field: str) -> str:
    return require_lowercase_sha256(value, field=field)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenEcologyPhaseAError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise OpenEcologyPhaseAError(f"{field} must be finite")
    return parsed


def _within_one_percent(value: float, reference: float) -> bool:
    return abs(value - reference) <= 0.01 * max(abs(reference), 1.0e-12)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpenEcologyPhaseAError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OpenEcologyPhaseAError(f"{field} must be a sequence")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise OpenEcologyPhaseAError(
            f"{field} fields differ: missing={sorted(expected - observed)}, "
            f"surplus={sorted(observed - expected)}"
        )


def _validate_signed_payload(value: Mapping[str, object], *, field: str) -> None:
    digest = _sha256(value.get("exact_digest"), field=f"{field}.exact_digest")
    unsigned = dict(value)
    unsigned.pop("exact_digest", None)
    if stable_payload_digest(unsigned) != digest:
        raise OpenEcologyPhaseAError(f"{field} exact digest mismatched")


def _json_clone(value: object, *, field: str) -> dict[str, object]:
    try:
        normalized = json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as error:
        raise OpenEcologyPhaseAError(f"{field} is not canonical JSON") from error
    if not isinstance(normalized, dict):
        raise OpenEcologyPhaseAError(f"{field} must normalize to an object")
    return normalized


def _load_strict_json(path: str | Path) -> dict[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OpenEcologyPhaseAError(f"failed to load strict JSON: {error}") from error
    if not isinstance(payload, dict):
        raise OpenEcologyPhaseAError("strict JSON root must be an object")
    return payload


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologyPhaseAError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise OpenEcologyPhaseAError(f"non-finite JSON constant: {value}")


def _canonical_phase_a_terminal_root(value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise OpenEcologyPhaseAError(
            "Phase A authoritative selection terminal root cannot be a symbolic link"
        )
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise OpenEcologyPhaseAError(
            "Phase A authoritative selection terminal root is missing"
        ) from error
    if not root.is_dir() or root.is_symlink():
        raise OpenEcologyPhaseAError(
            "Phase A authoritative selection requires one regular terminal root"
        )
    return root


def _canonical_phase_a_terminal_paths(root: Path) -> tuple[Path, ...]:
    expected_run_ids = tuple(
        phase_a_run_id(cell_id=cell_id, learner_index=learner_index)
        for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER
        for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT)
    )
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise OpenEcologyPhaseAError(
            "Phase A terminal matrix cannot be inventoried"
        ) from error
    if len(entries) != len(expected_run_ids) or {
        entry.name for entry in entries
    } != set(expected_run_ids):
        raise OpenEcologyPhaseAError(
            "Phase A terminal matrix is missing, duplicate, or contains surplus inputs"
        )
    terminal_paths: list[Path] = []
    for run_id in expected_run_ids:
        run_directory = root / run_id
        terminal_directory = run_directory / "terminal"
        if (
            run_directory.is_symlink()
            or not run_directory.is_dir()
            or terminal_directory.is_symlink()
            or not terminal_directory.is_dir()
        ):
            raise OpenEcologyPhaseAError(
                "Phase A terminal matrix contains a symbolic-link or non-directory input"
            )
        terminal_paths.append(terminal_directory / "terminal.json")
    return tuple(terminal_paths)


def _load_sealed_selection_json(path: Path, *, field: str) -> dict[str, object]:
    payload = _capture_selection_file_bytes(
        path,
        field=field,
        maximum_bytes=_MAX_SELECTION_AUTHORIZATION_BYTES,
    )
    try:
        decoded = payload.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OpenEcologyPhaseAError(f"{field} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise OpenEcologyPhaseAError(f"{field} strict JSON root must be an object")
    return parsed


def _capture_selection_file_bytes(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
) -> bytes:
    if path.is_symlink():
        raise OpenEcologyPhaseAError(f"{field} cannot be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OpenEcologyPhaseAError(f"{field} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise OpenEcologyPhaseAError(
                f"{field} must be one bounded regular non-hardlinked file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise OpenEcologyPhaseAError(f"{field} exceeds its byte bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise OpenEcologyPhaseAError(f"{field} disappeared during capture") from error

    def identity(observed: os.stat_result) -> tuple[int, ...]:
        return (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_uid,
            observed.st_gid,
            observed.st_nlink,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(after) != identity(path_stat):
        raise OpenEcologyPhaseAError(f"{field} was replaced during capture")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise OpenEcologyPhaseAError(f"{field} changed length during capture")
    return payload


def _selection_evidence_entry(
    value: object,
    *,
    expected_kind: str,
    field: str,
) -> Mapping[str, object]:
    entry = _mapping(value, field=field)
    _require_exact_keys(
        entry,
        {"evidence_kind", "semantic_report_digest", "file"},
        field=field,
    )
    if entry.get("evidence_kind") != expected_kind:
        raise OpenEcologyPhaseAError(f"{field} expected evidence kind {expected_kind}")
    _sha256(
        entry.get("semantic_report_digest"),
        field=f"{field}.semantic_report_digest",
    )
    _mapping(entry.get("file"), field=f"{field}.file")
    return entry


def _phase_a_selection_terminal_authority_entry(
    request: OpenEcologySelectionRequest,
    *,
    terminal_root: Path,
) -> dict[str, object]:
    from evolution_sim.mind.open_ecology_selection import (
        OpenEcologySelectionRequest as SelectionRequest,
    )

    if not isinstance(request, SelectionRequest):
        raise OpenEcologyPhaseAError(
            "Phase A selection preflight request has the wrong type"
        )
    terminal_root = terminal_root.resolve()
    cell_id = _cell_id(request.cell_id)
    learner_index = _index(
        request.learner_index,
        field="selection request learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
    run_id = phase_a_run_id(cell_id=cell_id, learner_index=learner_index)
    terminal_directory = terminal_root / run_id / "terminal"
    terminal_path = terminal_directory / "terminal.json"
    if (
        request.artifact_path.name != "terminal-training-artifact.json"
        or request.artifact_path.parent.resolve() != terminal_directory
        or request.run_contract_path.name != "run-contract.json"
        or request.run_contract_path.parent.resolve() != terminal_directory
    ):
        raise OpenEcologyPhaseAError(
            "Phase A selection request paths left the canonical terminal matrix"
        )
    authority = _mapping(
        request.training_authority,
        field="selection request terminal authority",
    )
    return {
        "cell_id": cell_id,
        "learner_index": learner_index,
        "run_id": run_id,
        "terminal_authority": dict(authority),
        "terminal_file": _sealed_selection_file_reference(
            terminal_path,
            base=terminal_root,
            field="Phase A terminal file",
        ),
        "run_contract_file": _sealed_selection_file_reference(
            request.run_contract_path,
            base=terminal_root,
            field="Phase A terminal run contract",
        ),
        "artifact_file": _sealed_selection_file_reference(
            request.artifact_path,
            base=terminal_root,
            field="Phase A terminal artifact",
        ),
    }


def _sealed_selection_file_reference(
    path: Path,
    *,
    base: Path,
    field: str,
) -> dict[str, object]:
    canonical_path = path.parent.resolve() / path.name
    try:
        relative = canonical_path.relative_to(base).as_posix()
    except ValueError as error:
        raise OpenEcologyPhaseAError(f"{field} escaped the terminal root") from error
    if any(part in {"", ".", ".."} for part in Path(relative).parts):
        raise OpenEcologyPhaseAError(f"{field} has an unsafe relative path")
    current = base
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise OpenEcologyPhaseAError(f"{field} contains a symbolic link")
    payload = _capture_selection_file_bytes(
        path,
        field=field,
        maximum_bytes=_MAX_SELECTION_AUTHORIZATION_BYTES,
    )
    return {
        "relative_path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
    }


def _write_atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _file_reference(path: Path, *, base: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise OpenEcologyPhaseAError("evidence reference requires a regular file")
    try:
        relative = path.relative_to(base).as_posix()
    except ValueError as error:
        raise OpenEcologyPhaseAError("evidence file escaped its commit root") from error
    return {
        "relative_path": relative,
        "sha256": _file_sha256(path),
        "byte_length": path.stat().st_size,
    }


def _verify_evidence_references(
    raw_references: object,
    *,
    base: Path,
    field: str,
) -> None:
    references = _sequence(raw_references, field=field)
    if not references:
        raise OpenEcologyPhaseAError(f"{field} must contain concrete evidence")
    for index, raw_reference in enumerate(references):
        reference = _mapping(raw_reference, field=f"{field}[{index}]")
        _require_exact_keys(
            reference,
            {"relative_path", "sha256", "byte_length"},
            field=f"{field}[{index}]",
        )
        relative = reference.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise OpenEcologyPhaseAError(f"{field}[{index}] path is unsafe")
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError as error:
            raise OpenEcologyPhaseAError(
                f"{field}[{index}] escaped its authorization bundle"
            ) from error
        _verify_file_reference(reference, path=candidate, base=base.resolve())


def _verify_file_reference(
    reference: Mapping[str, object],
    *,
    path: Path,
    base: Path,
) -> None:
    expected = _file_reference(path, base=base)
    if dict(reference) != expected:
        raise OpenEcologyPhaseAError("evidence file reference or bytes drifted")


def _phase_a_stat_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
    )


def _rename_phase_a_name_no_replace(
    parent_descriptor: int,
    *,
    source_name: str,
    destination_name: str,
    destination_path: Path,
) -> None:
    """Atomically rename two entries in one held directory without clobbering."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            encoded_source,
            parent_descriptor,
            encoded_destination,
            0x00000004 | 0x00000010,
        )
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            encoded_source,
            parent_descriptor,
            encoded_destination,
            1,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "exclusive Phase A evidence publication is unsupported",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_path,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_path,
    )


def _publish_phase_a_staging_directory(
    parent: Path,
    *,
    source_name: str,
    destination_name: str,
    expected_source_identity: tuple[int, int, int],
) -> None:
    """Publish the held source identity with atomic create-if-absent semantics."""

    if (
        Path(source_name).name != source_name
        or Path(destination_name).name != destination_name
        or source_name in {".", ".."}
        or destination_name in {".", ".."}
    ):
        raise OpenEcologyPhaseAError("Phase A publication entry name is unsafe")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor: int | None = None
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        parent_descriptor = os.open(parent, flags)
        parent_identity = _phase_a_stat_identity(os.fstat(parent_descriptor))
        if _phase_a_stat_identity(parent.lstat()) != parent_identity:
            raise OpenEcologyPhaseAError("Phase A publication parent identity changed")
        source_descriptor = os.open(
            source_name,
            flags,
            dir_fd=parent_descriptor,
        )
        source_identity = _phase_a_stat_identity(os.fstat(source_descriptor))
        if source_identity != expected_source_identity:
            raise OpenEcologyPhaseAError(
                "Phase A staging source identity changed before publication"
            )
        current_source = os.stat(
            source_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _phase_a_stat_identity(current_source) != source_identity:
            raise OpenEcologyPhaseAError(
                "Phase A staging source namespace changed before publication"
            )
        _rename_phase_a_name_no_replace(
            parent_descriptor,
            source_name=source_name,
            destination_name=destination_name,
            destination_path=parent / destination_name,
        )
        destination_descriptor = os.open(
            destination_name,
            flags,
            dir_fd=parent_descriptor,
        )
        if _phase_a_stat_identity(os.fstat(destination_descriptor)) != source_identity:
            raise OpenEcologyPhaseAError(
                "Phase A staging source identity changed during publication"
            )
        if _phase_a_stat_identity(parent.lstat()) != parent_identity:
            raise OpenEcologyPhaseAError("Phase A publication parent identity changed")
        os.fsync(parent_descriptor)
    except FileExistsError as error:
        raise OpenEcologyPhaseAError(
            "Phase A publication destination already exists"
        ) from error
    except OpenEcologyPhaseAError:
        raise
    except OSError as error:
        raise OpenEcologyPhaseAError(
            "Phase A exclusive evidence publication failed"
        ) from error
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _phase_a_directory_entry_identity(
    path: Path,
    *,
    expected_files: set[str],
    field: str,
) -> tuple[tuple[object, ...], ...]:
    try:
        root_stat = path.lstat()
        children = {child.name: child for child in path.iterdir()}
    except OSError as error:
        raise OpenEcologyPhaseAError(f"{field} could not be inspected") from error
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or path.is_symlink()
        or set(children) != expected_files
    ):
        raise OpenEcologyPhaseAError(f"{field} is incomplete, unsafe, or surplus")
    identity: list[tuple[object, ...]] = [
        (
            ".",
            root_stat.st_dev,
            root_stat.st_ino,
            root_stat.st_mode,
            root_stat.st_mtime_ns,
        )
    ]
    for name in sorted(expected_files):
        try:
            child_stat = children[name].lstat()
        except OSError as error:
            raise OpenEcologyPhaseAError(
                f"{field} changed during inspection"
            ) from error
        if not stat.S_ISREG(child_stat.st_mode) or children[name].is_symlink():
            raise OpenEcologyPhaseAError(f"{field} contains an unsafe file")
        identity.append(
            (
                name,
                child_stat.st_dev,
                child_stat.st_ino,
                child_stat.st_mode,
                child_stat.st_size,
                child_stat.st_mtime_ns,
                child_stat.st_ctime_ns,
            )
        )
    return tuple(identity)


def _phase_a_update_entry_identity(path: Path) -> tuple[tuple[object, ...], ...]:
    return _phase_a_directory_entry_identity(
        path,
        expected_files={
            "update.json",
            "checkpoint.json",
            "commit.json",
        },
        field="Phase A pending update staging",
    )


def _phase_a_terminal_entry_identity(path: Path) -> tuple[tuple[object, ...], ...]:
    return _phase_a_directory_entry_identity(
        path,
        expected_files={
            "terminal.json",
            "terminal-training-artifact.json",
            "run-contract.json",
        },
        field="Phase A terminal staging",
    )


def _recover_unpublished_update_staging(
    run_directory: Path,
    *,
    run_contract: Mapping[str, object],
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
    resume: bool,
) -> None:
    """Publish one crash-complete next update only after exact prefix proof."""

    updates_root = run_directory / "updates"
    if not updates_root.exists() and not updates_root.is_symlink():
        return
    if not updates_root.is_dir() or updates_root.is_symlink():
        raise OpenEcologyPhaseAError("Phase A updates path is not a real directory")
    pending: list[tuple[int, Path]] = []
    for entry in sorted(updates_root.iterdir(), key=lambda value: value.name):
        final_match = _UPDATE_DIRECTORY_RE.fullmatch(entry.name)
        if final_match is not None and entry.is_dir() and not entry.is_symlink():
            continue
        staging_match = _UPDATE_STAGING_DIRECTORY_RE.fullmatch(entry.name)
        if staging_match is None or not entry.is_dir() or entry.is_symlink():
            raise OpenEcologyPhaseAError(
                "Phase A recovery found an unknown update entry"
            )
        pending.append((int(staging_match.group(1)), entry))
    if not pending:
        return
    if not resume:
        raise OpenEcologyPhaseAError(
            "unpublished update staging exists but resume was not enabled"
        )
    if len(pending) != 1:
        raise OpenEcologyPhaseAError(
            "Phase A recovery requires exactly one pending update stage"
        )
    index, staging = pending[0]
    prefix = _verify_phase_a_evidence_prefix(
        run_directory,
        run_contract=run_contract,
        schedule=schedule,
        ignored_update_entry_names=frozenset({staging.name}),
    )
    if index != prefix.completed_updates or index >= len(schedule):
        raise OpenEcologyPhaseAError(
            "Phase A pending update is not the exact next committed update"
        )
    previous_digest = prefix.commit_digests[-1] if prefix.commit_digests else None
    expected_before = _sha256(
        prefix.terminal_model_state_sha256 or prefix.initial_model_state_sha256,
        field="Phase A pending update expected model state",
    )
    source_identity = _phase_a_stat_identity(staging.lstat())
    identity_before = _phase_a_update_entry_identity(staging)
    node = _verify_phase_a_evidence_node(
        staging,
        index=index,
        previous_commit_exact_digest=previous_digest,
        expected_model_state_sha256_before_update=expected_before,
        run_contract=run_contract,
        scheduled_tasks=schedule[index],
    )
    if _phase_a_update_entry_identity(staging) != identity_before:
        raise OpenEcologyPhaseAError(
            "Phase A pending update staging changed during verification"
        )
    final_directory = updates_root / f"update-{index:04d}"
    if os.path.lexists(final_directory):
        raise OpenEcologyPhaseAError(
            "Phase A pending update duplicates a committed update"
        )
    _publish_phase_a_staging_directory(
        updates_root,
        source_name=staging.name,
        destination_name=final_directory.name,
        expected_source_identity=source_identity,
    )
    if _phase_a_update_entry_identity(final_directory) != identity_before:
        raise OpenEcologyPhaseAError(
            "Phase A pending update identity changed during publication"
        )
    verified = verify_phase_a_evidence_prefix(
        run_directory,
        run_contract=run_contract,
        schedule=schedule,
    )
    if (
        verified.completed_updates != prefix.completed_updates + 1
        or verified.commit_digests[-1] != node.commit_digest
    ):
        raise OpenEcologyPhaseAError(
            "Phase A recovered update did not extend the exact evidence prefix"
        )


def _recover_unpublished_terminal_staging(
    run_directory: Path,
    *,
    run_contract: Mapping[str, object],
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
    resume: bool,
) -> None:
    pending = sorted(
        (
            entry
            for entry in run_directory.iterdir()
            if entry.name.startswith(".terminal.pending-")
        ),
        key=lambda value: value.name,
    )
    if pending and not resume:
        raise OpenEcologyPhaseAError(
            "unpublished terminal staging exists but resume was not enabled"
        )
    if not pending:
        return
    if len(pending) != 1:
        raise OpenEcologyPhaseAError(
            "Phase A recovery requires exactly one terminal stage"
        )
    staging = pending[0]
    if (
        _TERMINAL_STAGING_DIRECTORY_RE.fullmatch(staging.name) is None
        or not staging.is_dir()
        or staging.is_symlink()
    ):
        raise OpenEcologyPhaseAError("unsafe Phase A terminal staging entry")
    source_identity = _phase_a_stat_identity(staging.lstat())
    entry_identity = _phase_a_terminal_entry_identity(staging)
    _verify_phase_a_terminal(
        staging / "terminal.json",
        artifact_path=staging / "terminal-training-artifact.json",
        run_contract_path=staging / "run-contract.json",
        run_contract=run_contract,
        run_directory=run_directory,
        schedule=schedule,
    )
    terminal_directory = run_directory / "terminal"
    _publish_phase_a_staging_directory(
        run_directory,
        source_name=staging.name,
        destination_name=terminal_directory.name,
        expected_source_identity=source_identity,
    )
    if _phase_a_terminal_entry_identity(terminal_directory) != entry_identity:
        raise OpenEcologyPhaseAError(
            "Phase A terminal identity changed during recovery publication"
        )
    _verify_phase_a_terminal(
        terminal_directory / "terminal.json",
        artifact_path=terminal_directory / "terminal-training-artifact.json",
        run_contract_path=terminal_directory / "run-contract.json",
        run_contract=run_contract,
        run_directory=run_directory,
        schedule=schedule,
    )


def _require_live_output_storage(
    output_root: Path,
    *,
    launch_authorization: Mapping[str, object],
) -> None:
    probe = output_root.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise OpenEcologyPhaseAError("Phase A output filesystem is unavailable")
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    storage = _mapping(
        _mapping(
            launch_authorization.get("operational_gates"),
            field="operational_gates",
        ).get("storage"),
        field="operational_gates.storage",
    )
    required_free = _positive_int(
        storage.get("required_target_free_bytes"),
        field="storage.required_target_free_bytes",
    )
    if usage.total != storage.get("target_filesystem_capacity_bytes"):
        raise OpenEcologyPhaseAError(
            "Phase A output filesystem differs from the authorized target"
        )
    if usage.free < required_free:
        raise OpenEcologyPhaseAError(
            "Phase A output filesystem no longer satisfies its free-space gate"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise OpenEcologyPhaseAError(
            f"failed to hash evidence file: {error}"
        ) from error
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class _exclusive_run_lock:
    def __init__(
        self,
        run_directory: Path,
        *,
        run_contract: Mapping[str, object],
    ) -> None:
        self.path = run_directory / ".run.lock"
        self.run_contract = run_contract
        self.handle: object | None = None

    def __enter__(self) -> _exclusive_run_lock:
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise OpenEcologyPhaseAError(
                "Phase A run directory is locked by another process"
            ) from error
        handle.seek(0)
        existing = handle.read()
        identity = {
            "schema_version": "mind_v3_open_ecology_phase_a_output_lock_v1",
            "run_id": self.run_contract["run_id"],
            "run_contract_digest": self.run_contract["exact_digest"],
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        if existing and existing != canonical + "\n":
            handle.close()
            raise OpenEcologyPhaseAError("Phase A output lock identity drifted")
        if not existing:
            handle.seek(0)
            handle.write(canonical + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.handle = handle
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is None:
            return
        handle = self.handle
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        handle.close()  # type: ignore[union-attr]
        self.handle = None


__all__ = [
    "OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_BLOCKERS",
    "OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_PROOF_PRODUCERS_AVAILABLE",
    "OPEN_ECOLOGY_PHASE_A_BENCHMARK_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_BENCHMARK_WORKER_COUNTS",
    "OPEN_ECOLOGY_PHASE_A_CELL_ORDER",
    "OPEN_ECOLOGY_PHASE_A_CELLS",
    "OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_LEARNER_EVIDENCE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_MAXIMUM_WALL_SECONDS",
    "OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH",
    "OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256",
    "OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES",
    "OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_SELECTION_AUTHORIZATION_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_TOTAL_TRAINING_WORLDS",
    "OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_PATH",
    "OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SCHEMA_VERSION",
    "OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_SHA256",
    "OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_PATH",
    "OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SCHEMA_VERSION",
    "OPEN_ECOLOGY_LAUNCH_AUTHORITY_AMENDMENT_V2_SHA256",
    "OpenEcologyPhaseAError",
    "PhaseAEvidencePrefix",
    "PhaseAOpenEcologyRolloutTask",
    "authorize_open_ecology_phase_a_cell_selection",
    "authorize_phase_a_terminal_selection_report",
    "build_open_ecology_phase_a_evidence_index",
    "build_open_ecology_phase_a_launch_authorization",
    "build_open_ecology_phase_a_preregistration",
    "build_open_ecology_phase_a_resource_envelope",
    "build_open_ecology_phase_a_runtime_contract",
    "build_open_ecology_phase_a_throughput_gate",
    "build_phase_a_run_components",
    "build_phase_a_run_contract",
    "build_verified_phase_a_selection_request",
    "configure_open_ecology_phase_a_determinism",
    "open_ecology_phase_a_launch_readiness",
    "phase_a_run_id",
    "preview_open_ecology_phase_a_cell_selection",
    "run_open_ecology_phase_a_cell",
    "validate_open_ecology_phase_a_learner_evidence",
    "validate_open_ecology_phase_a_launch_authorization",
    "validate_open_ecology_phase_a_preregistration",
    "validate_open_ecology_phase_a_resource_envelope",
    "validate_open_ecology_phase_a_runtime_contract",
    "validate_open_ecology_phase_a_selection_authorization",
    "validate_open_ecology_phase_a_source_observation",
    "validate_open_ecology_phase_a_throughput_gate",
    "verify_phase_a_evidence_prefix",
    "write_phase_a_update_commit",
]
