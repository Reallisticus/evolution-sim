from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import torch

from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    CRITIC_GENOME_CONDITIONING_FILM_V1,
    CRITIC_GENOME_CONDITIONING_NONE,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
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
    OPEN_ECOLOGY_MAX_AGENTS,
    OPEN_ECOLOGY_PHASE_A,
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
    derive_recurrent_policy_sampling_seed,
)
from evolution_sim.mind.recurrent_scale_campaign import source_file_hash_manifest

if TYPE_CHECKING:
    from evolution_sim.mind.open_ecology_selection import (
        OpenEcologySelectionRequest,
    )


OPEN_ECOLOGY_PHASE_A_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_campaign_v1"
OPEN_ECOLOGY_PHASE_A_RUN_CONTRACT_VERSION = (
    "mind_v3_open_ecology_phase_a_run_contract_v1"
)
OPEN_ECOLOGY_PHASE_A_TASK_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_rollout_task_v1"
)
OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_runtime_v1"
OPEN_ECOLOGY_PHASE_A_THROUGHPUT_GATE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_throughput_gate_v3"
)
OPEN_ECOLOGY_PHASE_A_RESOURCE_PROJECTION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_resource_projection_v3"
)
OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_resource_envelope_v1"
)
OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_launch_authorization_v1"
)
OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION = "mind_v3_open_ecology_phase_a_update_v1"
OPEN_ECOLOGY_PHASE_A_COMMIT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_evidence_commit_v1"
)
OPEN_ECOLOGY_PHASE_A_TERMINAL_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_training_terminal_v2"
)
OPEN_ECOLOGY_PHASE_A_LEARNER_EVIDENCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_learner_selection_evidence_v2"
)
OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_terminal_selection_authority_v1"
)
OPEN_ECOLOGY_PHASE_A_SELECTION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_cell_selection_v1"
)
OPEN_ECOLOGY_PHASE_A_AUTHORITATIVE_SELECTION_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_authoritative_cell_selection_v1"
)
OPEN_ECOLOGY_PHASE_A_POLICY_SAMPLING_NAMESPACE = (
    "evolution-sim|mind-v3-open-ecology|phase-a|policy-sampling-v1"
)
OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH = (
    "docs/research/open-ecology-campaign-preregistration-v1.md"
)
OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256 = (
    "058b19dfadcb3cd4aef0c942d6b88be1b3a077701818774f40c162713bfdd8ad"
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
    "mind_recurrent_end_to_end_pipeline_benchmark_v1"
)
OPEN_ECOLOGY_PHASE_A_ALL_TRAINING_WORLDS = 6_144
OPEN_ECOLOGY_PHASE_A_MATRIX_TRAINING_WORLDS = 2_048
OPEN_ECOLOGY_PHASE_B_MATRIX_TRAINING_WORLDS = 4_096
OPEN_ECOLOGY_PHASE_B_ROLLOUT_TICKS = 256
OPEN_ECOLOGY_PHASE_A_PRIMARY_SELECTION_EXECUTIONS = 2_560
OPEN_ECOLOGY_PHASE_A_REPLAY_SELECTION_EXECUTIONS = 2_560
OPEN_ECOLOGY_PHASE_A_PHYSICAL_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_A_PRIMARY_SELECTION_EXECUTIONS
    + OPEN_ECOLOGY_PHASE_A_REPLAY_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_PHASE_B_PRIMARY_SELECTION_EXECUTIONS = 1_280
OPEN_ECOLOGY_PHASE_B_REPLAY_SELECTION_EXECUTIONS = 1_280
OPEN_ECOLOGY_PHASE_B_PHYSICAL_SELECTION_EXECUTIONS = (
    OPEN_ECOLOGY_PHASE_B_PRIMARY_SELECTION_EXECUTIONS
    + OPEN_ECOLOGY_PHASE_B_REPLAY_SELECTION_EXECUTIONS
)
OPEN_ECOLOGY_RESOURCE_PROJECTION_SAFETY_FACTOR = 1.20
OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES = tuple(
    f"readiness_dependency_{index:02d}" for index in range(1, 11)
)
OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_PROOF_PRODUCERS_AVAILABLE = False
OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_BLOCKERS = (
    *OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES,
    "dependency_specific_behavioral_evidence_parsers",
    "throughput_attestation_parser",
    "phase_b_mixed_density_training_throughput_benchmark_and_parser",
    "long_horizon_selection_throughput_benchmark_and_parser",
    "storage_attestation_parser",
    "output_lock_attestation_parser",
    "immutable_uploader_attestation_parser",
    "verification_before_prune_attestation_parser",
    "terminal_aggregate_validator_attestation_parser",
)

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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_DIRECTORY_RE = re.compile(r"^phase-a-(a[0-3])-learner-([0-3])-([1-9][0-9]*)$")
_UPDATE_DIRECTORY_RE = re.compile(r"^update-([0-9]{4})$")
_COMMON_PARAMETER_EXCLUSIONS = ("critic_genome_",)


class OpenEcologyPhaseAError(ValueError):
    """Raised when Phase A differs from its sealed causal-ablation contract."""


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
            cell_id=cell_id,
            learner_index=learner_index,
            learner_seed=learner_seed,
            genome_stream_seed_index=learner_index,
            update_index=self.open_ecology_update_index,
            world_index=self.open_ecology_world_index,
            environment_seed_index=environment_index,
        )
        if self.policy_sampling_identity != expected_identity:
            raise OpenEcologyPhaseAError(
                "Phase A policy sampling identity is not phase/cell canonical"
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


def open_ecology_phase_a_launch_readiness() -> dict[str, object]:
    """Return the truthful current launch boundary without inferring proof."""

    readiness: dict[str, object] = {
        "schema_version": "mind_v3_open_ecology_phase_a_launch_readiness_v1",
        "phase_a_training_authorized": False,
        "dependency_specific_proof_producers_available": (
            OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_PROOF_PRODUCERS_AVAILABLE
        ),
        "blockers": list(OPEN_ECOLOGY_PHASE_A_AUTHORIZATION_BLOCKERS),
        "reason": (
            "dependency_specific_machine_evidence_schemas_and_semantic_"
            "validators_are_not_yet_implemented"
        ),
        "claim_boundary": {
            "training_launch": False,
            "phase_b": False,
            "runtime_integration": False,
            "promotion": False,
        },
    }
    readiness["exact_digest"] = stable_payload_digest(readiness)
    return readiness


def validate_open_ecology_phase_a_launch_authorization(
    authorization: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    authorization_path: str | Path,
) -> None:
    """Verify every sealed readiness proof and its concrete evidence bytes."""

    readiness = open_ecology_phase_a_launch_readiness()
    if readiness["phase_a_training_authorized"] is not True:
        raise OpenEcologyPhaseAError(
            "Phase A launch remains blocked: dependency-specific behavioral "
            "proof producers and semantic validators are unavailable"
        )
    _require_exact_keys(
        authorization,
        {
            "schema_version",
            "campaign_digest",
            "configuration_sha256",
            "source",
            "readiness_dependencies",
            "operational_gates",
            "authorization",
            "exact_digest",
        },
        field="Phase A launch authorization",
    )
    _validate_signed_payload(
        authorization,
        field="Phase A launch authorization",
    )
    if (
        authorization.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("campaign_digest") != preregistration.get("exact_digest")
        or authorization.get("configuration_sha256")
        != preregistration.get("configuration_sha256")
        or authorization.get("source") != preregistration.get("source")
    ):
        raise OpenEcologyPhaseAError(
            "Phase A launch authorization is detached from the campaign"
        )
    record_path = Path(authorization_path).resolve()
    if not record_path.is_file() or record_path.is_symlink():
        raise OpenEcologyPhaseAError(
            "Phase A launch authorization must be a regular file"
        )
    evidence_root = record_path.parent
    proofs = _sequence(
        authorization.get("readiness_dependencies"),
        field="launch_authorization.readiness_dependencies",
    )
    if len(proofs) != len(OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES):
        raise OpenEcologyPhaseAError(
            "Phase A launch authorization requires all 10 readiness proofs"
        )
    source = _mapping(preregistration.get("source"), field="source")
    for expected_id, raw_proof in zip(
        OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES,
        proofs,
        strict=True,
    ):
        proof = _mapping(raw_proof, field=f"readiness_dependencies.{expected_id}")
        _require_exact_keys(
            proof,
            {
                "dependency_id",
                "status",
                "source_commit",
                "source_manifest_sha256",
                "assertions",
                "evidence",
            },
            field=f"readiness_dependencies.{expected_id}",
        )
        if (
            proof.get("dependency_id") != expected_id
            or proof.get("status") != "behaviorally_proved"
            or proof.get("source_commit") != source.get("commit")
            or proof.get("source_manifest_sha256") != source.get("manifest_sha256")
        ):
            raise OpenEcologyPhaseAError(
                f"Phase A readiness proof {expected_id} is not source-bound"
            )
        assertions = _mapping(
            proof.get("assertions"),
            field=f"readiness_dependencies.{expected_id}.assertions",
        )
        if dict(assertions) != _phase_a_readiness_assertions(expected_id):
            raise OpenEcologyPhaseAError(
                f"Phase A readiness proof {expected_id} assertions drifted"
            )
        _verify_evidence_references(
            proof.get("evidence"),
            base=evidence_root,
            field=f"readiness_dependencies.{expected_id}.evidence",
        )
    gates = _mapping(
        authorization.get("operational_gates"),
        field="launch_authorization.operational_gates",
    )
    _require_exact_keys(
        gates,
        {
            "throughput",
            "storage",
            "output_lock",
            "immutable_uploader",
            "verification_before_prune",
            "terminal_aggregate_validator",
        },
        field="launch_authorization.operational_gates",
    )
    throughput = _mapping(gates.get("throughput"), field="gates.throughput")
    _require_exact_keys(
        throughput,
        {"passed", "throughput_gate_digest", "evidence"},
        field="gates.throughput",
    )
    expected_throughput = _mapping(
        preregistration.get("throughput_gate"),
        field="throughput_gate",
    )
    if (
        expected_throughput.get("gate_passed") is not True
        or throughput.get("passed") is not True
        or throughput.get("throughput_gate_digest")
        != expected_throughput.get("exact_digest")
    ):
        raise OpenEcologyPhaseAError("Phase A throughput launch proof failed")
    _verify_evidence_references(
        throughput.get("evidence"),
        base=evidence_root,
        field="gates.throughput.evidence",
    )
    storage = _mapping(gates.get("storage"), field="gates.storage")
    _require_exact_keys(
        storage,
        {
            "passed",
            "checked_at_utc",
            "google_drive_free_bytes",
            "projected_active_storage_bytes",
            "target_filesystem_capacity_bytes",
            "target_filesystem_free_bytes",
            "required_target_free_bytes",
            "evidence",
        },
        field="gates.storage",
    )
    capacity = _positive_int(
        storage.get("target_filesystem_capacity_bytes"),
        field="gates.storage.target_filesystem_capacity_bytes",
    )
    required_free = max(100 * 1024**3, math.ceil(capacity * 0.20))
    if (
        storage.get("passed") is not True
        or not isinstance(storage.get("checked_at_utc"), str)
        or not str(storage.get("checked_at_utc")).strip()
        or _positive_int(
            storage.get("google_drive_free_bytes"),
            field="gates.storage.google_drive_free_bytes",
        )
        < 300 * 1024**3
        or _positive_int(
            storage.get("projected_active_storage_bytes"),
            field="gates.storage.projected_active_storage_bytes",
        )
        > 200 * 1024**3
        or storage.get("required_target_free_bytes") != required_free
        or _positive_int(
            storage.get("target_filesystem_free_bytes"),
            field="gates.storage.target_filesystem_free_bytes",
        )
        < required_free
    ):
        raise OpenEcologyPhaseAError("Phase A storage launch gate failed")
    _verify_evidence_references(
        storage.get("evidence"),
        base=evidence_root,
        field="gates.storage.evidence",
    )
    for gate_name in (
        "output_lock",
        "immutable_uploader",
        "verification_before_prune",
        "terminal_aggregate_validator",
    ):
        gate = _mapping(gates.get(gate_name), field=f"gates.{gate_name}")
        _require_exact_keys(
            gate,
            {"passed", "evidence"},
            field=f"gates.{gate_name}",
        )
        if gate.get("passed") is not True:
            raise OpenEcologyPhaseAError(f"Phase A operational gate {gate_name} failed")
        _verify_evidence_references(
            gate.get("evidence"),
            base=evidence_root,
            field=f"gates.{gate_name}.evidence",
        )
    authorization_claim = _mapping(
        authorization.get("authorization"),
        field="launch_authorization.authorization",
    )
    if dict(authorization_claim) != {
        "phase_a_training_authorized": True,
        "authorization_basis": (
            "all_10_behavioral_dependencies_plus_operational_gates"
        ),
        "authorization_scope": "phase_a_training_only",
        "phase_b_authorized": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
    }:
        raise OpenEcologyPhaseAError("Phase A launch authorization claim drifted")


def build_open_ecology_phase_a_preregistration(
    *,
    source_commit: str,
    source_manifest_sha256: str,
    runtime_contract: Mapping[str, object],
    throughput_gate: Mapping[str, object],
) -> dict[str, object]:
    commit = _git_sha(source_commit)
    manifest_sha256 = _sha256(
        source_manifest_sha256,
        field="source_manifest_sha256",
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
            "path": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH,
            "file_sha256": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
            "editing_after_evidence_allowed": False,
        },
        "source": {
            "commit": commit,
            "manifest_sha256": manifest_sha256,
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
            "benchmark_role": "open_ecology_benchmark",
            "benchmark_seed_indices": list(
                range(len(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_benchmark"]))
            ),
            "benchmark_seeds": list(
                OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_benchmark"]
            ),
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
            "primary_executions_per_artifact": (
                OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "independent_replay_executions_per_artifact": (
                OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "physical_world_runs_per_artifact": (
                2
                * OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
                * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
            ),
            "execution_count_semantics": (
                "primary_policy_executions_plus_equal_independent_replay_runs"
            ),
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
        "path": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH,
        "file_sha256": OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
        "editing_after_evidence_allowed": False,
    }:
        raise OpenEcologyPhaseAError("sealed Phase A document binding drifted")
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
        "benchmark_role": "open_ecology_benchmark",
        "benchmark_seed_indices": list(
            range(len(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_benchmark"]))
        ),
        "benchmark_seeds": list(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_benchmark"]),
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
        "primary_executions_per_artifact": (
            OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "independent_replay_executions_per_artifact": (
            OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "physical_world_runs_per_artifact": (
            2
            * OPEN_ECOLOGY_PHASE_A_SELECTION_SEED_COUNT
            * (OPEN_ECOLOGY_PHASE_A_STOCHASTIC_TAPE_COUNT + 1)
        ),
        "execution_count_semantics": (
            "primary_policy_executions_plus_equal_independent_replay_runs"
        ),
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
                cell_id=resolved_cell,
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


def authorize_open_ecology_phase_a_cell_selection(
    preregistration: Mapping[str, object],
    *,
    terminal_root: str | Path,
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

    validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source(preregistration)
    root = Path(terminal_root)
    if not root.is_dir() or root.is_symlink():
        raise OpenEcologyPhaseAError(
            "Phase A authoritative selection requires one regular terminal root"
        )

    learner_evidence: list[dict[str, object]] = []
    primary_report_digests: list[str] = []
    for cell_id in OPEN_ECOLOGY_PHASE_A_CELL_ORDER:
        for learner_index in range(OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT):
            terminal_path = (
                root
                / phase_a_run_id(
                    cell_id=cell_id,
                    learner_index=learner_index,
                )
                / "terminal"
                / "terminal.json"
            )
            request = build_verified_phase_a_selection_request(
                preregistration,
                cell_id=cell_id,
                learner_index=learner_index,
                terminal_path=terminal_path,
                evaluation_workers=evaluation_workers,
            )
            primary_report = evaluate_open_ecology_selection_artifact(request)
            evidence = authorize_phase_a_terminal_selection_report(
                preregistration,
                primary_report,
                cell_id=cell_id,
                learner_index=learner_index,
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
    readiness = open_ecology_phase_a_launch_readiness()
    if readiness["phase_a_training_authorized"] is not True:
        raise OpenEcologyPhaseAError(
            "Phase A launch remains blocked by unimplemented dependency-specific "
            "behavioral proof producers and validators"
        )
    authorization_path = Path(launch_authorization_path).resolve()
    launch_authorization = _load_strict_json(authorization_path)
    validate_open_ecology_phase_a_launch_authorization(
        launch_authorization,
        preregistration=preregistration,
        authorization_path=authorization_path,
    )
    configure_open_ecology_phase_a_determinism()
    _require_live_source(preregistration)
    _require_live_runtime(preregistration, device=device)
    _require_launch_dependencies()
    resolved_cell = _cell_id(cell_id)
    resolved_index = _index(
        learner_index,
        field="learner_index",
        upper=OPEN_ECOLOGY_PHASE_A_LEARNER_COUNT,
    )
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
            counterfactual_config=None,
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
            update = runner.train_update(schedule[update_index])
            update_payload = _phase_a_update_payload(
                update,
                run_contract=run_contract,
                previous_commit_exact_digest=(
                    prefix.commit_digests[-1] if prefix.commit_digests else None
                ),
                model=runner.model,
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
        staging = Path(
            tempfile.mkdtemp(
                prefix=".terminal.pending-",
                dir=run_directory,
            )
        )
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
                    "terminal_model_state_sha256": recurrent_model_state_sha256(
                        runner.model
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
            os.replace(staging, terminal_directory)
            _fsync_directory(run_directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
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
    if previous_commit_exact_digest is not None:
        _sha256(
            previous_commit_exact_digest,
            field="previous_commit_exact_digest",
        )
    updates_root = run_root / "updates"
    updates_root.mkdir(parents=True, exist_ok=True)
    final_directory = updates_root / f"update-{index:04d}"
    if final_directory.exists():
        raise OpenEcologyPhaseAError("Phase A update commit already exists")
    staging = updates_root / (
        f".update-{index:04d}.pending-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    try:
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
        os.replace(staging, final_directory)
        _fsync_directory(updates_root)
        return commit
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_phase_a_evidence_prefix(
    run_directory: str | Path,
    *,
    run_contract: Mapping[str, object],
    schedule: Sequence[Sequence[PhaseAOpenEcologyRolloutTask]],
) -> PhaseAEvidencePrefix:
    run_root = Path(run_directory)
    updates_root = run_root / "updates"
    if not updates_root.exists():
        return PhaseAEvidencePrefix(0, (), None)
    if not updates_root.is_dir() or updates_root.is_symlink():
        raise OpenEcologyPhaseAError("Phase A updates path is not a real directory")
    entries = sorted(updates_root.iterdir(), key=lambda value: value.name)
    indices: list[int] = []
    for entry in entries:
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
    for index in indices:
        directory = updates_root / f"update-{index:04d}"
        if {path.name for path in directory.iterdir()} != {
            "update.json",
            "checkpoint.json",
            "commit.json",
        }:
            raise OpenEcologyPhaseAError(
                "Phase A update commit files are incomplete or surplus"
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
            or commit.get("previous_commit_exact_digest") != previous_digest
        ):
            raise OpenEcologyPhaseAError(
                "Phase A update commit hash-chain binding drifted"
            )
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
        _validate_signed_payload(update, field="Phase A update")
        if (
            update.get("schema_version") != OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION
            or update.get("run_id") != run_contract.get("run_id")
            or update.get("run_contract_digest") != run_contract.get("exact_digest")
            or update.get("update_index") != index
            or update.get("previous_commit_exact_digest") != previous_digest
            or update.get("tasks") != [asdict(task) for task in schedule[index]]
        ):
            raise OpenEcologyPhaseAError(
                "Phase A update journal detached from its exact schedule"
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
            or rng_state.get("phase_a_update_exact_digest")
            != update.get("exact_digest")
            or rng_state.get("phase_a_previous_commit_exact_digest") != previous_digest
            or recurrent_model_state_sha256(loaded.model)
            != update.get("model_state_sha256_after_update")
        ):
            raise OpenEcologyPhaseAError(
                "Phase A checkpoint detached from its evidence prefix"
            )
        previous_digest = _sha256(
            commit.get("exact_digest"),
            field="commit.exact_digest",
        )
        commit_digests.append(previous_digest)
        terminal_checkpoint = loaded
    return PhaseAEvidencePrefix(
        completed_updates=len(indices),
        commit_digests=tuple(commit_digests),
        terminal_checkpoint=terminal_checkpoint,
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
    ):
        raise OpenEcologyPhaseAError("Phase A selection execution drifted")
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
    for field in (
        "median_per_decision_normalized_individual_return",
        "heldout_value_rmse",
        "advantage_variance",
    ):
        _finite(metrics.get(field), field=f"metrics.{field}")
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
        "artifact_serialization_included": False,
        "evaluation_included": False,
        "policy_promotion_authorized": False,
    }
    if dict(scope) != expected_scope:
        raise OpenEcologyPhaseAError("Phase A pipeline benchmark scope drifted")
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
        "genome_stream_seed": OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
            0
        ],
        "training_phase": OPEN_ECOLOGY_PHASE_A,
        "encoder_size": 256,
        "hidden_size": 256,
        "recurrent_layers": 1,
        "learner_seed": _phase_a_learner_seeds()[0],
        "update_epochs": 4,
        "sequence_minibatch_size": 16,
        "tbptt_steps": 128,
        "burn_in_steps": 16,
        "scheduled_worlds": OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
        "scheduled_world_ticks": (
            OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE
            * (OPEN_ECOLOGY_PHASE_A_ROLLOUT_TICKS + 1)
        ),
        "schedule_contract": "canonical_open_ecology_phase_a_broad_treatment_v1",
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
        _phase_a_ppo_config(learner_seed=_phase_a_learner_seeds()[0])
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
    _require_exact_keys(
        envelope,
        {
            "schema_version",
            "source_commit",
            "maximum_wall_seconds",
            "evidence_sha256",
            "exact_digest",
        },
        field="Phase A resource envelope",
    )
    _validate_signed_payload(envelope, field="Phase A resource envelope")
    maximum_wall_seconds = _finite(
        envelope.get("maximum_wall_seconds"),
        field="resource_envelope.maximum_wall_seconds",
    )
    if (
        envelope.get("schema_version")
        != OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION
        or envelope.get("source_commit") != source_commit
        or maximum_wall_seconds <= 0.0
    ):
        raise OpenEcologyPhaseAError("Phase A resource envelope drifted")
    _sha256(
        envelope.get("evidence_sha256"),
        field="resource_envelope.evidence_sha256",
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
            "exact_sha_operational_path_proved": True,
            "multi_host_equivalence_proved": True,
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
    cell_id: str,
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
        f"cell={cell_id}|learner_index={learner_index:02d}|"
        f"learner={learner_seed}|genome_stream_index="
        f"{genome_stream_seed_index:02d}|"
        f"genome_mode={RecurrentGenomePopulationMode.HERITABLE.value}|"
        f"update={update_index:04d}|world={world_index:06d}|"
        f"environment_index={environment_seed_index:03d}|"
        f"tape=training_rollout_v1|"
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
) -> dict[str, object]:
    if update.counterfactual_collection is not None or update.counterfactual_auxiliary:
        raise OpenEcologyPhaseAError(
            "Phase A cannot consume counterfactual auxiliary training"
        )
    payload: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_UPDATE_SCHEMA_VERSION,
        "run_id": run_contract["run_id"],
        "run_contract_digest": run_contract["exact_digest"],
        "update_index": update.update_index,
        "previous_commit_exact_digest": previous_commit_exact_digest,
        "tasks": [asdict(task) for task in update.tasks],
        "rollout": asdict(update.rollout),
        "optimizer": asdict(update.optimizer),
        "counterfactual_collection": None,
        "counterfactual_auxiliary": None,
        "model_state_sha256_after_update": recurrent_model_state_sha256(model),
    }
    payload["exact_digest"] = stable_payload_digest(payload)
    return payload


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
            "terminal_model_state_sha256",
        },
        field="terminal.training",
    )
    if (
        training.get("completed_updates") != prefix.completed_updates
        or training.get("training_world_count")
        != (OPEN_ECOLOGY_PHASE_A_UPDATE_COUNT * OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE)
        or training.get("commit_exact_digests") != list(prefix.commit_digests)
    ):
        raise OpenEcologyPhaseAError(
            "Phase A terminal training evidence does not match its prefix"
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


def _require_live_source(preregistration: Mapping[str, object]) -> None:
    source = _mapping(preregistration.get("source"), field="source")
    expected_commit = _git_sha(source.get("commit"))
    expected_manifest = _sha256(
        source.get("manifest_sha256"),
        field="source.manifest_sha256",
    )
    try:
        observed_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise OpenEcologyPhaseAError("failed to inspect Phase A Git source") from error
    if observed_commit != expected_commit or status.strip():
        raise OpenEcologyPhaseAError(
            "Phase A requires the exact clean preregistered Git source"
        )
    if (
        source_file_hash_manifest(_REPOSITORY_ROOT)["aggregate_sha256"]
        != expected_manifest
    ):
        raise OpenEcologyPhaseAError("Phase A runtime source manifest drifted")
    prereg_path = _REPOSITORY_ROOT / OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH
    if _file_sha256(prereg_path) != OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256:
        raise OpenEcologyPhaseAError("sealed Phase A preregistration bytes drifted")


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
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyPhaseAError(f"{field} must be lowercase SHA-256")
    return value


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


def _recover_unpublished_terminal_staging(
    run_directory: Path,
    *,
    resume: bool,
) -> None:
    pending = sorted(run_directory.glob(".terminal.pending-*"))
    if pending and not resume:
        raise OpenEcologyPhaseAError(
            "unpublished terminal staging exists but resume was not enabled"
        )
    for path in pending:
        if (
            not path.is_dir()
            or path.is_symlink()
            or not path.name.startswith(".terminal.pending-")
        ):
            raise OpenEcologyPhaseAError("unsafe Phase A terminal staging entry")
        # These bytes were never atomically published. The complete verified
        # update prefix remains authoritative and deterministically rebuilds
        # the terminal bundle.
        shutil.rmtree(path)


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
    "OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256",
    "OPEN_ECOLOGY_PHASE_A_READINESS_DEPENDENCIES",
    "OPEN_ECOLOGY_PHASE_A_RESOURCE_ENVELOPE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_RUNTIME_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_A_TOTAL_TRAINING_WORLDS",
    "OpenEcologyPhaseAError",
    "PhaseAEvidencePrefix",
    "PhaseAOpenEcologyRolloutTask",
    "authorize_open_ecology_phase_a_cell_selection",
    "authorize_phase_a_terminal_selection_report",
    "build_open_ecology_phase_a_preregistration",
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
    "validate_open_ecology_phase_a_runtime_contract",
    "validate_open_ecology_phase_a_throughput_gate",
    "verify_phase_a_evidence_prefix",
    "write_phase_a_update_commit",
]
