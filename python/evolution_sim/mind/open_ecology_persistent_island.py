from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Protocol

from evolution_sim.config import WorldConfig
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.world import SimulationWorld
from evolution_sim.io.open_ecology_campaign_storage import (
    CampaignStorageError,
    CampaignStorageLock,
    canonical_json_bytes,
    default_storage_lock_path,
    ensure_real_directory_tree,
)
from evolution_sim.io.open_ecology_aggregate_commit import (
    OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME,
    OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME,
    OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY,
    OPEN_ECOLOGY_AGGREGATE_POINTER_DIGEST_POLICY,
    OPEN_ECOLOGY_AGGREGATE_POINTER_SCHEMA,
    OpenEcologyAggregateIdentityPins,
    OpenEcologyAggregateResumePins,
    inspect_current_open_ecology_aggregate_generation,
    inspect_open_ecology_aggregate_attempt_frontier,
    load_current_open_ecology_aggregate_generation,
    load_or_recover_open_ecology_genesis_generation,
    publish_open_ecology_aggregate_generation,
    select_open_ecology_aggregate_attempt_generation,
)
from evolution_sim.io.open_ecology_checkpoint import (
    OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
    REQUIRED_CHECKPOINT_COMPONENTS,
    VersionedCheckpointState,
    load_open_ecology_checkpoint,
    write_open_ecology_checkpoint,
)
from evolution_sim.io.open_ecology_rotating_writer import (
    BirthEvidence,
    CongestionEvidence,
    DeathEvidence,
    DyadicInteractionEvidence,
    InterventionEvidence,
    LineageEvidence,
    OpenEcologyEvidenceEvent,
    RotatingEvidenceConfig,
    RotatingOpenEcologyEvidenceWriter,
    SignalContributorEvidence,
    load_open_ecology_evidence_manifest,
    preserve_open_ecology_evidence_attempt_directory,
    restore_open_ecology_evidence_manifest_snapshot,
)
from evolution_sim.io.open_ecology_runtime_checkpoint import (
    RuntimeCheckpointBinding,
    RuntimeCheckpointComponents,
    capture_open_ecology_runtime_checkpoint,
    restore_open_ecology_runtime_checkpoint,
)
from evolution_sim.io.open_ecology_git_authority import (
    OpenEcologyGitAuthorityError,
    discover_pinned_git_executable,
    run_pinned_git,
)
from evolution_sim.io.source_manifest import source_file_hash_manifest
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
)
from evolution_sim.mind.policy_inputs import (
    TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
)
from evolution_sim.mind.recurrent_actor_critic import (
    GENOME_CONDITIONING_ACTOR_FILM_V1,
)
from evolution_sim.mind.recurrent_artifact import (
    FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
    load_frozen_recurrent_policy_artifact,
)
from evolution_sim.mind.recurrent_experiment import (
    OPEN_ECOLOGY_MAX_AGENTS,
    OpenEcologySignalTreatment,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationMode,
)
from evolution_sim.mind.recurrent_policy import (
    DeterministicPublicRecurrentPolicy,
    recurrent_model_state_sha256,
)
from evolution_sim.mind.recurrent_rollout import (
    RECURRENT_ROLLOUT_ACTION_SOURCE,
    derive_recurrent_policy_sampling_seed,
)


OPEN_ECOLOGY_PERSISTENT_TASK_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_task_v1"
)
OPEN_ECOLOGY_PERSISTENT_RUNNER_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_runner_v1"
)
OPEN_ECOLOGY_PERSISTENT_SUMMARY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_summary_v1"
)
OPEN_ECOLOGY_PERSISTENT_MILESTONE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_milestone_v1"
)
OPEN_ECOLOGY_PERSISTENT_WRITER_SOURCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_writer_source_v1"
)
OPEN_ECOLOGY_PERSISTENT_CHECKPOINT_CONFIG_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_checkpoint_config_v1"
)
OPEN_ECOLOGY_PERSISTENT_CHECKPOINT_SEED_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_checkpoint_seeds_v1"
)
OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_persistent_island_runner_state_v1"
)
OPEN_ECOLOGY_PHASE_D_POLICY_SAMPLING_IDENTITY_VERSION = (
    "mind_v3_open_ecology_phase_d_policy_sampling_identity_v1"
)
OPEN_ECOLOGY_PHASE_D_ARM_ORDER = ("H", "Z", "R")
OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT = 4
OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT = 4
OPEN_ECOLOGY_PHASE_D_TASK_COUNT = 48
OPEN_ECOLOGY_PHASE_D_TARGET_TICKS = 50_000
OPEN_ECOLOGY_FIRST_MILESTONE_TICK = 10_000
OPEN_ECOLOGY_SUMMARY_INTERVAL_TICKS = 100
OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS = 5_000
# Policy target only. No checkpoint is deleted until an independently verified
# cold-storage receipt contract is implemented.
OPEN_ECOLOGY_RUNTIME_CHECKPOINT_RETENTION_COUNT = 2
OPEN_ECOLOGY_PHASE_D_DENSITIES = frozenset({64, 128, 256})
OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED = True
OPEN_ECOLOGY_PARALLEL_CHECKPOINT_PUBLICATION_AUTHORIZED = True
OPEN_ECOLOGY_CHECKPOINT_PUBLICATION_CONCURRENCY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_checkpoint_publication_concurrency_v1"
)
OPEN_ECOLOGY_GENESIS_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_genesis_attempt_authority_v1"
)
OPEN_ECOLOGY_PERSISTENT_LAUNCH_BLOCKERS = (
    "campaign_storage_interval_check_not_integrated",
    "exact_source_phase_d_throughput_gate_not_yet_passed",
)
_OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_ATTRIBUTE = (
    "_open_ecology_persistent_runner_state"
)
_OPEN_ECOLOGY_TASK_MUTATION_LOCK_SCHEMA_VERSION = (
    "mind_v3_open_ecology_task_mutation_lock_v1"
)
_OPEN_ECOLOGY_CAMPAIGN_BARRIER_FRONTIER_SCHEMA_VERSION = (
    "mind_v3_open_ecology_campaign_barrier_frontier_v1"
)
_OPEN_ECOLOGY_LOCK_MAX_BYTES = 16 * 1024
_OPEN_ECOLOGY_PHASE_D_TASK_ID_RE = re.compile(
    r"^phase-d-l(?:0[0-3])-i(?:0[0-3])-[hzr]$"
)
_OPEN_ECOLOGY_CURRENT_MAX_BYTES = 64 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MOVE_DELTAS = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}


class OpenEcologyPersistentIslandError(ValueError):
    """Raised when a persistent primary-island contract fails closed."""


def checkpoint_publication_concurrency_contract() -> dict[str, object]:
    """Describe the lock and frontier-latency contract without wall-clock claims."""

    return {
        "schema_version": (
            OPEN_ECOLOGY_CHECKPOINT_PUBLICATION_CONCURRENCY_SCHEMA_VERSION
        ),
        "task_lock_mode": "exclusive",
        "campaign_epoch_lock_mode": "shared",
        "global_storage_barrier_lock_mode": "exclusive",
        "distinct_task_publications_may_overlap": (
            OPEN_ECOLOGY_PARALLEL_CHECKPOINT_PUBLICATION_AUTHORIZED
        ),
        "same_task_publications_may_overlap": False,
        "storage_scan_may_overlap_publication": False,
        "frontier_publication_latency_model": (
            "maximum_task_pipeline_elapsed_not_sum_of_task_pipelines"
        ),
        "publication_parallelism_upper_bound": OPEN_ECOLOGY_PHASE_D_TASK_COUNT,
        "global_serial_publication_blocker_present": False,
        "exact_source_throughput_gate_still_required": True,
    }


class _IdentityFileLock:
    """One validated flock over a canonical identity file."""

    def __init__(
        self,
        path: Path,
        *,
        identity: Mapping[str, object],
        shared: bool,
        nonblocking: bool,
        initialize: bool = False,
    ) -> None:
        self._path = path
        self._expected = canonical_json_bytes(dict(identity))
        if len(self._expected) > _OPEN_ECOLOGY_LOCK_MAX_BYTES:
            raise CampaignStorageError("cooperative lock identity is too large")
        self._operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        self._nonblocking = nonblocking
        self._initialize = initialize
        self._descriptor: int | None = None

    def __enter__(self) -> _IdentityFileLock:
        if not self._path.is_absolute() or not self._path.parent.is_dir():
            raise CampaignStorageError("cooperative lock path is not an absolute child")
        flags = (
            os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(
                os,
                "O_CLOEXEC",
                0,
            )
        )
        if self._initialize:
            flags |= os.O_CREAT
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except OSError as error:
            raise CampaignStorageError(
                f"cannot safely open cooperative lock: {error}"
            ) from error
        try:
            operation = self._operation
            if self._nonblocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, operation)
            except BlockingIOError as error:
                raise CampaignStorageError(
                    "campaign or task mutation barrier is locked by another process"
                ) from error
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
                or metadata.st_size > _OPEN_ECOLOGY_LOCK_MAX_BYTES
            ):
                raise CampaignStorageError(
                    "cooperative lock is not one private small regular file"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = os.read(descriptor, _OPEN_ECOLOGY_LOCK_MAX_BYTES + 1)
            if not observed and self._initialize:
                if self._operation != fcntl.LOCK_EX:
                    raise CampaignStorageError(
                        "cooperative lock initialization requires exclusive mode"
                    )
                _write_lock_bytes(descriptor, self._expected)
                observed = self._expected
            if observed != self._expected:
                raise CampaignStorageError("cooperative lock identity drifted")
            self._descriptor = descriptor
            return self
        except BaseException:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class _EvidenceWriter(Protocol):
    @property
    def diagnostics(self) -> Mapping[str, object]: ...

    def append(self, event: OpenEcologyEvidenceEvent) -> None: ...

    def checkpoint(self) -> Mapping[str, object]: ...

    def finish(self) -> Mapping[str, object]: ...

    def abort(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PersistentArtifactBinding:
    learner_index: int
    learner_seed: int
    artifact_path: str
    artifact_sha256: str
    artifact_file_sha256: str
    source_commit: str
    terminal_authority_sha256: str

    def __post_init__(self) -> None:
        _index(
            self.learner_index,
            field="learner_index",
            upper=OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT,
        )
        expected_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
            self.learner_index
        ]
        if self.learner_seed != expected_seed:
            raise OpenEcologyPersistentIslandError(
                "artifact learner seed does not match registry order"
            )
        if (
            not isinstance(self.artifact_path, str)
            or not self.artifact_path
            or self.artifact_path != self.artifact_path.strip()
            or not Path(self.artifact_path).is_absolute()
        ):
            raise OpenEcologyPersistentIslandError(
                "artifact_path must be a non-empty absolute path"
            )
        _sha256(self.artifact_sha256, field="artifact_sha256")
        _sha256(self.artifact_file_sha256, field="artifact_file_sha256")
        _sha256(
            self.terminal_authority_sha256,
            field="terminal_authority_sha256",
        )
        if (
            not isinstance(self.source_commit, str)
            or _COMMIT_RE.fullmatch(self.source_commit) is None
        ):
            raise OpenEcologyPersistentIslandError(
                "source_commit must be a full hexadecimal Git commit"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "learner_index": self.learner_index,
            "learner_seed": self.learner_seed,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_file_sha256": self.artifact_file_sha256,
            "source_commit": self.source_commit,
            "terminal_authority_sha256": self.terminal_authority_sha256,
        }


@dataclass(frozen=True, slots=True)
class PersistentIslandTask:
    learner_index: int
    learner_seed: int
    island_index: int
    environment_seed: int
    genome_stream_seed_index: int
    genome_stream_seed: int
    arm: str
    genome_population_mode: str
    reset_recurrent_state_each_decision: bool
    selected_density: int
    policy_sampling_identity: str
    policy_sampling_seed: int
    world_identity: str
    task_id: str
    world_config_sha256: str
    artifact: PersistentArtifactBinding
    schema_version: str = OPEN_ECOLOGY_PERSISTENT_TASK_SCHEMA_VERSION
    target_ticks: int = OPEN_ECOLOGY_PHASE_D_TARGET_TICKS

    def __post_init__(self) -> None:
        if self.schema_version != OPEN_ECOLOGY_PERSISTENT_TASK_SCHEMA_VERSION:
            raise OpenEcologyPersistentIslandError(
                "persistent task schema version drifted"
            )
        _index(
            self.learner_index,
            field="learner_index",
            upper=OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT,
        )
        _index(
            self.island_index,
            field="island_index",
            upper=OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT,
        )
        if self.artifact.learner_index != self.learner_index:
            raise OpenEcologyPersistentIslandError(
                "task and artifact learner indices disagree"
            )
        if self.learner_seed != self.artifact.learner_seed:
            raise OpenEcologyPersistentIslandError(
                "task and artifact learner seeds disagree"
            )
        if (
            self.environment_seed
            != OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_island"][self.island_index]
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent environment seed does not match island index"
            )
        if (
            self.genome_stream_seed_index != self.island_index
            or self.genome_stream_seed
            != OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
                self.island_index
            ]
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent genome stream is not paired by island index"
            )
        expected_arm = _arm_contract(self.arm)
        if (
            self.genome_population_mode != expected_arm["genome_population_mode"]
            or self.reset_recurrent_state_each_decision
            is not expected_arm["reset_recurrent_state_each_decision"]
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent H/Z/R treatment fields drifted"
            )
        if self.selected_density not in OPEN_ECOLOGY_PHASE_D_DENSITIES:
            raise OpenEcologyPersistentIslandError(
                "selected_density must be one preregistered phase-D density"
            )
        if self.target_ticks != OPEN_ECOLOGY_PHASE_D_TARGET_TICKS:
            raise OpenEcologyPersistentIslandError(
                "persistent target must remain exactly 50,000 ticks"
            )
        expected_task_id = _task_id(
            learner_index=self.learner_index,
            island_index=self.island_index,
            arm=self.arm,
        )
        if self.task_id != expected_task_id or self.world_identity != expected_task_id:
            raise OpenEcologyPersistentIslandError(
                "persistent task/world identity drifted"
            )
        expected_sampling_identity = _policy_sampling_identity(
            learner_index=self.learner_index,
            learner_seed=self.learner_seed,
            island_index=self.island_index,
            environment_seed=self.environment_seed,
            arm=self.arm,
        )
        if self.policy_sampling_identity != expected_sampling_identity:
            raise OpenEcologyPersistentIslandError(
                "persistent policy-sampling identity drifted"
            )
        expected_sampling_seed = derive_recurrent_policy_sampling_seed(
            task_identity=expected_sampling_identity
        )
        if self.policy_sampling_seed != expected_sampling_seed:
            raise OpenEcologyPersistentIslandError(
                "persistent policy-sampling seed drifted"
            )
        expected_world_config_sha256 = _stable_digest(
            build_persistent_world_config(self).to_dict()
        )
        if self.world_config_sha256 != expected_world_config_sha256:
            raise OpenEcologyPersistentIslandError(
                "persistent world configuration digest drifted"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "world_identity": self.world_identity,
            "learner_index": self.learner_index,
            "learner_seed": self.learner_seed,
            "island_index": self.island_index,
            "environment_seed": self.environment_seed,
            "genome_stream_seed_index": self.genome_stream_seed_index,
            "genome_stream_seed": self.genome_stream_seed,
            "arm": self.arm,
            "genome_population_mode": self.genome_population_mode,
            "reset_recurrent_state_each_decision": (
                self.reset_recurrent_state_each_decision
            ),
            "selected_density": self.selected_density,
            "policy_sampling_identity": self.policy_sampling_identity,
            "policy_sampling_seed": self.policy_sampling_seed,
            "world_config_sha256": self.world_config_sha256,
            "target_ticks": self.target_ticks,
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PersistentAdvanceResult:
    task_id: str
    requested_target_tick: int
    observed_tick: int
    extinct: bool
    extinction_tick: int | None
    summaries: tuple[dict[str, object], ...]
    milestones: tuple[dict[str, object], ...]
    model_state_sha256: str
    same_world_instance: bool


@dataclass(slots=True)
class _IntervalCounters:
    births: int = 0
    deaths: int = 0
    attacks: int = 0
    successful_attacks: int = 0
    feeding_events: int = 0
    drinking_events: int = 0
    reproduction_events: int = 0
    signal_emissions: int = 0
    invalid_observation_actions: int = 0
    invalid_resolution_actions: int = 0
    congestion_events: int = 0
    population_auc: int = 0
    requested_actions: Counter[str] = field(default_factory=Counter)
    resolved_actions: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        return {
            "births": self.births,
            "deaths": self.deaths,
            "attacks": self.attacks,
            "successful_attacks": self.successful_attacks,
            "feeding_events": self.feeding_events,
            "drinking_events": self.drinking_events,
            "reproduction_events": self.reproduction_events,
            "signal_emissions": self.signal_emissions,
            "invalid_observation_actions": self.invalid_observation_actions,
            "invalid_resolution_actions": self.invalid_resolution_actions,
            "congestion_events": self.congestion_events,
            "population_auc": self.population_auc,
            "requested_actions": dict(sorted(self.requested_actions.items())),
            "resolved_actions": dict(sorted(self.resolved_actions.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> _IntervalCounters:
        count_fields = (
            "births",
            "deaths",
            "attacks",
            "successful_attacks",
            "feeding_events",
            "drinking_events",
            "reproduction_events",
            "signal_emissions",
            "invalid_observation_actions",
            "invalid_resolution_actions",
            "congestion_events",
            "population_auc",
        )
        _exact_keys(
            payload,
            {*count_fields, "requested_actions", "resolved_actions"},
            field="runner_state.interval",
        )
        requested = _action_counter(
            payload["requested_actions"],
            field="runner_state.interval.requested_actions",
        )
        resolved = _action_counter(
            payload["resolved_actions"],
            field="runner_state.interval.resolved_actions",
        )
        return cls(
            **{
                field_name: _nonnegative_int(
                    payload[field_name],
                    field=f"runner_state.interval.{field_name}",
                )
                for field_name in count_fields
            },
            requested_actions=requested,
            resolved_actions=resolved,
        )

    def reset(self) -> None:
        self.births = 0
        self.deaths = 0
        self.attacks = 0
        self.successful_attacks = 0
        self.feeding_events = 0
        self.drinking_events = 0
        self.reproduction_events = 0
        self.signal_emissions = 0
        self.invalid_observation_actions = 0
        self.invalid_resolution_actions = 0
        self.congestion_events = 0
        self.population_auc = 0
        self.requested_actions.clear()
        self.resolved_actions.clear()


def build_persistent_island_task_matrix(
    artifact_bindings: Sequence[PersistentArtifactBinding],
    *,
    selected_density: int,
) -> tuple[PersistentIslandTask, ...]:
    """Build the frozen 4 learners x 4 islands x H/Z/R phase-D order."""

    if selected_density not in OPEN_ECOLOGY_PHASE_D_DENSITIES:
        raise OpenEcologyPersistentIslandError(
            "selected_density must be exactly 64, 128, or 256"
        )
    if len(artifact_bindings) != OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT:
        raise OpenEcologyPersistentIslandError(
            "phase D requires exactly four registry-ordered artifacts"
        )
    bindings = tuple(artifact_bindings)
    if tuple(binding.learner_index for binding in bindings) != tuple(
        range(OPEN_ECOLOGY_PHASE_D_LEARNER_COUNT)
    ):
        raise OpenEcologyPersistentIslandError(
            "phase-D artifacts must use registry learner order 0..3"
        )
    if len({binding.source_commit for binding in bindings}) != 1:
        raise OpenEcologyPersistentIslandError(
            "phase-D artifacts must share one exact source commit"
        )

    tasks: list[PersistentIslandTask] = []
    for binding in bindings:
        learner_index = binding.learner_index
        learner_seed = binding.learner_seed
        for island_index in range(OPEN_ECOLOGY_PHASE_D_ISLAND_COUNT):
            environment_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_island"][
                island_index
            ]
            genome_stream_seed = OPEN_ECOLOGY_SEED_REGISTRY[
                "open_ecology_genome_stream"
            ][island_index]
            for arm in OPEN_ECOLOGY_PHASE_D_ARM_ORDER:
                arm_contract = _arm_contract(arm)
                task_id = _task_id(
                    learner_index=learner_index,
                    island_index=island_index,
                    arm=arm,
                )
                sampling_identity = _policy_sampling_identity(
                    learner_index=learner_index,
                    learner_seed=learner_seed,
                    island_index=island_index,
                    environment_seed=environment_seed,
                    arm=arm,
                )
                provisional = {
                    "learner_index": learner_index,
                    "learner_seed": learner_seed,
                    "island_index": island_index,
                    "environment_seed": environment_seed,
                    "genome_stream_seed_index": island_index,
                    "genome_stream_seed": genome_stream_seed,
                    "arm": arm,
                    "genome_population_mode": arm_contract["genome_population_mode"],
                    "reset_recurrent_state_each_decision": arm_contract[
                        "reset_recurrent_state_each_decision"
                    ],
                    "selected_density": selected_density,
                    "policy_sampling_identity": sampling_identity,
                    "policy_sampling_seed": derive_recurrent_policy_sampling_seed(
                        task_identity=sampling_identity
                    ),
                    "world_identity": task_id,
                    "task_id": task_id,
                    "artifact": binding,
                }
                world_config = WorldConfig(
                    seed=environment_seed,
                    width=48,
                    height=32,
                    max_ticks=OPEN_ECOLOGY_PHASE_D_TARGET_TICKS,
                    initial_agents=selected_density,
                    max_agents=OPEN_ECOLOGY_MAX_AGENTS,
                    signals=OpenEcologySignalTreatment().as_signal_config(),
                )
                tasks.append(
                    PersistentIslandTask(
                        **provisional,
                        world_config_sha256=_stable_digest(world_config.to_dict()),
                    )
                )
    if len(tasks) != OPEN_ECOLOGY_PHASE_D_TASK_COUNT:
        raise AssertionError("phase-D task matrix size drifted")
    return tuple(tasks)


def prioritized_persistent_island_triplet(
    tasks: Sequence[PersistentIslandTask],
) -> tuple[PersistentIslandTask, PersistentIslandTask, PersistentIslandTask]:
    """Return the preregistered first learner/island H/Z/R block."""

    if len(tasks) != OPEN_ECOLOGY_PHASE_D_TASK_COUNT:
        raise OpenEcologyPersistentIslandError(
            "prioritized triplet requires the complete 48-task matrix"
        )
    expected = tuple(
        _task_id(learner_index=0, island_index=0, arm=arm)
        for arm in OPEN_ECOLOGY_PHASE_D_ARM_ORDER
    )
    first = tuple(tasks[:3])
    if tuple(task.task_id for task in first) != expected:
        raise OpenEcologyPersistentIslandError(
            "phase-D matrix does not start with the preregistered H/Z/R triplet"
        )
    return first[0], first[1], first[2]


def build_persistent_world_config(task: PersistentIslandTask) -> WorldConfig:
    return WorldConfig(
        seed=task.environment_seed,
        width=48,
        height=32,
        max_ticks=OPEN_ECOLOGY_PHASE_D_TARGET_TICKS,
        initial_agents=task.selected_density,
        max_agents=OPEN_ECOLOGY_MAX_AGENTS,
        signals=OpenEcologySignalTreatment().as_signal_config(),
    )


def persistent_island_writer_source_contract(
    task: PersistentIslandTask,
    *,
    model_state_sha256: str,
    source_git_sha: str,
    config_contract_sha256: str,
    seed_contract_sha256: str,
    run_generation_id: str,
    island_id: str,
) -> dict[str, object]:
    _sha256(model_state_sha256, field="model_state_sha256")
    _sha256(config_contract_sha256, field="config_contract_sha256")
    _sha256(seed_contract_sha256, field="seed_contract_sha256")
    if source_git_sha != task.artifact.source_commit:
        raise OpenEcologyPersistentIslandError(
            "writer source Git SHA does not match the task artifact"
        )
    if (
        not isinstance(run_generation_id, str)
        or not run_generation_id
        or island_id != task.task_id
    ):
        raise OpenEcologyPersistentIslandError(
            "writer source aggregate identity does not match the task"
        )
    payload: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PERSISTENT_WRITER_SOURCE_SCHEMA_VERSION,
        "source_git_sha": source_git_sha,
        "config_contract_sha256": config_contract_sha256,
        "seed_contract_sha256": seed_contract_sha256,
        "run_generation_id": run_generation_id,
        "island_id": island_id,
        "seed_registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
        "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        "task": task.to_dict(),
        "task_sha256": _stable_digest(task.to_dict()),
        "model_state_sha256": model_state_sha256,
        "event_producers": [
            "simulation_birth_groups",
            "simulation_death_events",
            "founder_lineage_state",
            "simulation_attack_and_mating_events",
            "physical_signal_emission_receiver_projection",
            "same_target_movement_contention",
            "external_frozen_branch_intervention_receipts_only",
        ],
        "periodic_summary_interval_ticks": OPEN_ECOLOGY_SUMMARY_INTERVAL_TICKS,
        "first_milestone_tick": OPEN_ECOLOGY_FIRST_MILESTONE_TICK,
        "primary_world_mutation_from_evidence": False,
        "launch_readiness": False,
        "launch_blockers": list(OPEN_ECOLOGY_PERSISTENT_LAUNCH_BLOCKERS),
    }
    payload["contract_sha256"] = _stable_digest(payload)
    return payload


def persistent_genesis_attempt_authority_sha256(
    task: PersistentIslandTask,
    *,
    campaign_root: str | Path,
    campaign_id: str,
    evidence_directory: str | Path | None = None,
    checkpoint_directory: str | Path | None = None,
    aggregate_directory: str | Path | None = None,
    writer_config: RotatingEvidenceConfig | None = None,
    checkpoint_interval_ticks: int = (OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS),
) -> str:
    """Derive the exact coordinator-visible authorization for tick-zero work."""

    campaign_path = Path(campaign_root)
    if not campaign_path.is_absolute():
        raise OpenEcologyPersistentIslandError(
            "genesis authority campaign_root must be absolute"
        )
    evidence_path = (
        campaign_path / task.task_id
        if evidence_directory is None
        else Path(evidence_directory)
    )
    checkpoint_path = (
        campaign_path / "checkpoints" / task.task_id
        if checkpoint_directory is None
        else Path(checkpoint_directory)
    )
    aggregate_path = (
        campaign_path / "aggregates" / task.task_id
        if aggregate_directory is None
        else Path(aggregate_directory)
    )
    if not all(
        path.is_absolute() for path in (evidence_path, checkpoint_path, aggregate_path)
    ):
        raise OpenEcologyPersistentIslandError(
            "genesis authority paths must be absolute"
        )
    concrete_writer_config = writer_config or RotatingEvidenceConfig(
        max_ticks_per_shard=5_000
    )
    payload = {
        "schema_version": OPEN_ECOLOGY_GENESIS_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_path),
        "task_id": task.task_id,
        "task_sha256": _stable_digest(task.to_dict()),
        "source_git_sha": task.artifact.source_commit,
        "run_generation_id": _persistent_checkpoint_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        ),
        "predecessor": None,
        "target_observed_tick": 0,
        "evidence_directory": str(evidence_path),
        "checkpoint_directory": str(checkpoint_path),
        "aggregate_directory": str(aggregate_path),
        "checkpoint_interval_ticks": _positive_int(
            checkpoint_interval_ticks,
            field="checkpoint_interval_ticks",
        ),
        "writer_config": concrete_writer_config.to_dict(),
    }
    return _stable_digest(payload)


def persistent_interval_attempt_authority_sha256(
    task: PersistentIslandTask,
    *,
    campaign_id: str,
    predecessor: OpenEcologyAggregateResumePins | None,
    target_tick: int,
) -> str:
    """Derive the coordinator/worker authorization for one barrier attempt."""

    requested_target = _positive_int(target_tick, field="target_tick")
    if requested_target > task.target_ticks:
        raise OpenEcologyPersistentIslandError(
            "interval attempt target exceeds the task target"
        )
    if predecessor is not None and not isinstance(
        predecessor,
        OpenEcologyAggregateResumePins,
    ):
        raise OpenEcologyPersistentIslandError(
            "interval attempt predecessor must be aggregate resume pins"
        )
    run_generation_id = _persistent_checkpoint_generation_id(
        campaign_id=campaign_id,
        task_id=task.task_id,
    )
    if predecessor is not None:
        if predecessor.identity.run_generation_id != run_generation_id:
            raise OpenEcologyPersistentIslandError(
                "interval attempt predecessor run identity drifted"
            )
        predecessor_commit = predecessor.commit_sha256
    else:
        predecessor_commit = "genesis"
    return persistent_interval_attempt_authority_payload_sha256(
        task_id=task.task_id,
        run_generation_id=run_generation_id,
        predecessor_aggregate_commit_sha256=predecessor_commit,
        requested_target_tick=requested_target,
    )


def persistent_interval_attempt_authority_payload_sha256(
    *,
    task_id: str,
    run_generation_id: str,
    predecessor_aggregate_commit_sha256: str,
    requested_target_tick: int,
) -> str:
    """Hash the exact newline-canonical authority object stored in an intent."""

    if (
        not isinstance(task_id, str)
        or _OPEN_ECOLOGY_PHASE_D_TASK_ID_RE.fullmatch(task_id) is None
        or not isinstance(run_generation_id, str)
        or not run_generation_id
        or len(run_generation_id) > 512
    ):
        raise OpenEcologyPersistentIslandError(
            "interval attempt payload identity is invalid"
        )
    if predecessor_aggregate_commit_sha256 != "genesis":
        _sha256(
            predecessor_aggregate_commit_sha256,
            field="predecessor aggregate commit SHA256",
        )
    target_tick = _positive_int(
        requested_target_tick,
        field="requested_target_tick",
    )
    authority = {
        "task_id": task_id,
        "run_generation_id": run_generation_id,
        "predecessor_aggregate_commit_sha256": (predecessor_aggregate_commit_sha256),
        "requested_target_tick": target_tick,
    }
    return hashlib.sha256(canonical_json_bytes(authority)).hexdigest()


class PersistentIslandRunner:
    """Advance one real, frozen-policy world without episode or milestone reset."""

    def __init__(
        self,
        *,
        task: PersistentIslandTask,
        world: SimulationWorld,
        policy: DeterministicPublicRecurrentPolicy,
        evidence_writer: _EvidenceWriter | None,
        artifact: Mapping[str, object],
        evidence_directory: Path,
        writer_config: RotatingEvidenceConfig,
        writer_source_contract: Mapping[str, object],
        campaign_root: Path,
        campaign_id: str,
        checkpoint_directory: Path,
        aggregate_directory: Path,
        checkpoint_interval_ticks: int,
        checkpoint_config_contract: VersionedCheckpointState,
        checkpoint_seed_contract: VersionedCheckpointState,
        restored_runner_state: Mapping[str, object] | None = None,
        restored_evidence_continuation: Mapping[str, object] | None = None,
        restored_checkpoint_metadata: Mapping[str, object] | None = None,
        restored_aggregate_metadata: Mapping[str, object] | None = None,
        task_lock_already_held: bool = False,
    ) -> None:
        if world.config.to_dict() != build_persistent_world_config(task).to_dict():
            raise OpenEcologyPersistentIslandError(
                "persistent world does not match its task configuration"
            )
        if world.policy is not policy:
            raise OpenEcologyPersistentIslandError(
                "persistent world is not bound to the supplied frozen policy"
            )
        self.task = task
        self._world = world
        self._policy = policy
        self._writer: _EvidenceWriter | None = evidence_writer
        self._artifact = dict(artifact)
        self._evidence_directory = evidence_directory
        self._writer_config = writer_config
        self._writer_source_contract = dict(writer_source_contract)
        self._campaign_root = campaign_root
        self._campaign_id = campaign_id
        self._storage_lock_path = default_storage_lock_path(campaign_root)
        self._task_lock_path = (
            campaign_root.parent
            / f".{campaign_root.name}.{task.task_id}.open-ecology-task.lock"
        )
        self._task_lock_identity = {
            "campaign_id": campaign_id,
            "schema_version": _OPEN_ECOLOGY_TASK_MUTATION_LOCK_SCHEMA_VERSION,
            "source_git_sha": task.artifact.source_commit,
            "task_id": task.task_id,
        }
        if not task_lock_already_held:
            with self._task_mutation_lock(initialize=True, nonblocking=True):
                pass
        self._checkpoint_directory = checkpoint_directory
        self._aggregate_directory = aggregate_directory
        self._checkpoint_config = checkpoint_config_contract
        self._checkpoint_seeds = checkpoint_seed_contract
        self._checkpoint_interval_ticks = _positive_int(
            checkpoint_interval_ticks,
            field="checkpoint_interval_ticks",
        )
        if self._checkpoint_interval_ticks > task.target_ticks:
            raise OpenEcologyPersistentIslandError(
                "checkpoint_interval_ticks cannot exceed the task target"
            )
        self._continuation_state: dict[str, object] | None = None
        self._evidence_finished = False
        self._evidence_aborted = False
        self._latest_runtime_checkpoint: dict[str, object] | None = None
        self._latest_aggregate_generation: dict[str, object] | None = None
        self._aggregate_resume_authorized = (
            restored_runner_state is None or restored_aggregate_metadata is not None
        )
        self._world_object_identity = id(world)
        self._model_state_sha256 = recurrent_model_state_sha256(policy.model)
        expected_writer_source = persistent_island_writer_source_contract(
            task,
            model_state_sha256=self._model_state_sha256,
            source_git_sha=task.artifact.source_commit,
            config_contract_sha256=_versioned_state_sha256(checkpoint_config_contract),
            seed_contract_sha256=_versioned_state_sha256(checkpoint_seed_contract),
            run_generation_id=_persistent_checkpoint_generation_id(
                campaign_id=campaign_id,
                task_id=task.task_id,
            ),
            island_id=task.task_id,
        )
        if self._writer_source_contract != expected_writer_source:
            raise OpenEcologyPersistentIslandError(
                "persistent writer source contract drifted"
            )

        world.record_events = False
        world.record_tick_details = True
        world.record_trajectory = True
        world.retain_trajectory_records = False
        world.trajectory_sink = None
        world._has_run = True
        if restored_runner_state is None:
            if evidence_writer is None or restored_evidence_continuation is not None:
                raise OpenEcologyPersistentIslandError(
                    "fresh runner requires one active evidence writer"
                )
            self._next_tick = 0
            self._extinct = len(world.alive_agents()) == 0
            self._extinction_tick: int | None = 0 if self._extinct else None
            self._episode_reset_count = 0
            self._world_replacement_count = 0
            self._interval = _IntervalCounters()
            self._summaries: list[dict[str, object]] = []
            self._milestones: dict[int, dict[str, object]] = {}
            self._event_counts: Counter[str] = Counter()
            writer_count = evidence_writer.diagnostics.get("total_event_count")
            if (
                isinstance(writer_count, bool)
                or not isinstance(writer_count, int)
                or writer_count < 0
            ):
                raise OpenEcologyPersistentIslandError(
                    "evidence writer total_event_count is invalid"
                )
            self._writer_initial_event_count = writer_count
            self._next_event_index = _writer_next_event_index(evidence_writer)
            self._last_evidence_tick: int | None = None
            self._emit_founder_lineages()
            if restored_aggregate_metadata is not None:
                raise OpenEcologyPersistentIslandError(
                    "fresh runner cannot have restored aggregate metadata"
                )
        else:
            if evidence_writer is not None or restored_evidence_continuation is None:
                raise OpenEcologyPersistentIslandError(
                    "restored runner requires an inactive writer continuation"
                )
            self._restore_runner_state(
                restored_runner_state,
                evidence_continuation=restored_evidence_continuation,
            )
            if restored_checkpoint_metadata is None:
                raise OpenEcologyPersistentIslandError(
                    "restored runner checkpoint metadata is missing"
                )
            self._latest_runtime_checkpoint = _json_clone(restored_checkpoint_metadata)
            if restored_aggregate_metadata is not None:
                self._latest_aggregate_generation = _json_clone(
                    restored_aggregate_metadata
                )
        self._verify_invariants()

    @classmethod
    def open(
        cls,
        task: PersistentIslandTask,
        *,
        evidence_directory: str | Path,
        campaign_root: str | Path,
        campaign_id: str,
        writer_config: RotatingEvidenceConfig | None = None,
        checkpoint_directory: str | Path | None = None,
        aggregate_directory: str | Path | None = None,
        checkpoint_interval_ticks: int = (
            OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS
        ),
    ) -> PersistentIslandRunner:
        attempt_authority_sha256 = persistent_genesis_attempt_authority_sha256(
            task,
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            evidence_directory=evidence_directory,
            checkpoint_directory=checkpoint_directory,
            aggregate_directory=aggregate_directory,
            writer_config=writer_config,
            checkpoint_interval_ticks=checkpoint_interval_ticks,
        )
        runner, _ = cls.materialize_or_restore_genesis(
            task,
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            attempt_authority_sha256=attempt_authority_sha256,
            evidence_directory=evidence_directory,
            writer_config=writer_config,
            checkpoint_directory=checkpoint_directory,
            aggregate_directory=aggregate_directory,
            checkpoint_interval_ticks=checkpoint_interval_ticks,
        )
        return runner

    @classmethod
    def materialize_or_restore_genesis(
        cls,
        task: PersistentIslandTask,
        *,
        campaign_root: str | Path,
        campaign_id: str,
        attempt_authority_sha256: str,
        evidence_directory: str | Path | None = None,
        writer_config: RotatingEvidenceConfig | None = None,
        checkpoint_directory: str | Path | None = None,
        aggregate_directory: str | Path | None = None,
        checkpoint_interval_ticks: int = (
            OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS
        ),
    ) -> tuple[PersistentIslandRunner, bool]:
        campaign_path = Path(campaign_root)
        evidence_path = (
            campaign_path / task.task_id
            if evidence_directory is None
            else Path(evidence_directory)
        )
        if not campaign_path.is_absolute() or not evidence_path.is_absolute():
            raise OpenEcologyPersistentIslandError(
                "campaign_root and evidence_directory must be absolute paths"
            )
        try:
            relative_evidence = evidence_path.relative_to(campaign_path)
        except ValueError as error:
            raise OpenEcologyPersistentIslandError(
                "evidence_directory must be inside campaign_root"
            ) from error
        if not relative_evidence.parts or ".." in relative_evidence.parts:
            raise OpenEcologyPersistentIslandError(
                "evidence_directory must be a normalized campaign child"
            )
        checkpoint_path = (
            campaign_path / "checkpoints" / task.task_id
            if checkpoint_directory is None
            else Path(checkpoint_directory)
        )
        if not checkpoint_path.is_absolute():
            raise OpenEcologyPersistentIslandError(
                "checkpoint_directory must be an absolute path"
            )
        try:
            relative_checkpoints = checkpoint_path.relative_to(campaign_path)
        except ValueError as error:
            raise OpenEcologyPersistentIslandError(
                "checkpoint_directory must be inside campaign_root"
            ) from error
        if not relative_checkpoints.parts or ".." in relative_checkpoints.parts:
            raise OpenEcologyPersistentIslandError(
                "checkpoint_directory must be a normalized campaign child"
            )
        aggregate_path = (
            campaign_path / "aggregates" / task.task_id
            if aggregate_directory is None
            else Path(aggregate_directory)
        )
        if not aggregate_path.is_absolute():
            raise OpenEcologyPersistentIslandError(
                "aggregate_directory must be an absolute path"
            )
        try:
            relative_aggregate = aggregate_path.relative_to(campaign_path)
        except ValueError as error:
            raise OpenEcologyPersistentIslandError(
                "aggregate_directory must be inside campaign_root"
            ) from error
        if not relative_aggregate.parts or ".." in relative_aggregate.parts:
            raise OpenEcologyPersistentIslandError(
                "aggregate_directory must be a normalized campaign child"
            )
        parsed_checkpoint_interval = _positive_int(
            checkpoint_interval_ticks,
            field="checkpoint_interval_ticks",
        )
        if parsed_checkpoint_interval > task.target_ticks:
            raise OpenEcologyPersistentIslandError(
                "checkpoint_interval_ticks cannot exceed the task target"
            )
        expected_attempt_authority = persistent_genesis_attempt_authority_sha256(
            task,
            campaign_root=campaign_path,
            campaign_id=campaign_id,
            evidence_directory=evidence_path,
            checkpoint_directory=checkpoint_path,
            aggregate_directory=aggregate_path,
            writer_config=writer_config,
            checkpoint_interval_ticks=parsed_checkpoint_interval,
        )
        if (
            _sha256(
                attempt_authority_sha256,
                field="genesis attempt authority SHA256",
            )
            != expected_attempt_authority
        ):
            raise OpenEcologyPersistentIslandError(
                "genesis attempt authority does not match exact task/config scope"
            )
        if len(task.artifact.source_commit) != 40:
            raise OpenEcologyPersistentIslandError(
                "campaign storage lock currently requires a 40-character Git SHA"
            )
        artifact_path = Path(task.artifact.artifact_path)
        if _file_sha256(artifact_path) != task.artifact.artifact_file_sha256:
            raise OpenEcologyPersistentIslandError(
                "frozen artifact file SHA256 does not match the task binding"
            )
        loaded = load_frozen_recurrent_policy_artifact(artifact_path)
        artifact = loaded.artifact
        if (
            artifact.get("schema_version")
            != FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION
        ):
            raise OpenEcologyPersistentIslandError(
                "phase D requires the frozen recurrent policy artifact schema"
            )
        if artifact.get("artifact_sha256") != task.artifact.artifact_sha256:
            raise OpenEcologyPersistentIslandError(
                "frozen artifact internal digest does not match the task binding"
            )
        provenance = _mapping(artifact.get("provenance"), field="artifact provenance")
        if (
            provenance.get("learner_seed") != task.learner_seed
            or provenance.get("source_commit") != task.artifact.source_commit
            or provenance.get("seed_registry_digest") != OPEN_ECOLOGY_CANONICAL_SHA256
        ):
            raise OpenEcologyPersistentIslandError(
                "frozen artifact provenance does not match phase-D authority"
            )
        model = loaded.model
        config = model.config
        if (
            config.encoder_size != 256
            or config.hidden_size != 256
            or config.recurrent_layers != 1
            or config.genome_conditioning_mode != GENOME_CONDITIONING_ACTOR_FILM_V1
            or config.public_input_schema_version
            != TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
        ):
            raise OpenEcologyPersistentIslandError(
                "phase D requires the tokenized width-256 actor_film_v1 model"
            )
        _verify_live_source_authority(task, artifact=artifact)
        model_sha256 = recurrent_model_state_sha256(model)
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest=task.artifact.artifact_sha256,
            copy_to_cpu=True,
            reset_recurrent_state_each_decision=(
                task.reset_recurrent_state_each_decision
            ),
            sampling_seed=task.policy_sampling_seed,
        )
        policy.start_world(
            world_identity=task.world_identity,
            genome_stream_seed=task.genome_stream_seed,
            genome_population_mode=task.genome_population_mode,
        )
        concrete_writer_config = writer_config or RotatingEvidenceConfig(
            max_ticks_per_shard=5_000
        )
        seed_contract = _persistent_checkpoint_seed_contract(task)
        config_contract = _persistent_checkpoint_config_contract(
            task,
            campaign_root=campaign_path,
            campaign_id=campaign_id,
            evidence_directory=evidence_path,
            checkpoint_directory=checkpoint_path,
            aggregate_directory=aggregate_path,
            checkpoint_interval_ticks=parsed_checkpoint_interval,
            writer_config=concrete_writer_config,
            artifact=artifact,
            model_state_sha256=model_sha256,
        )
        run_generation_id = _persistent_checkpoint_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        )
        source_contract = persistent_island_writer_source_contract(
            task,
            model_state_sha256=model_sha256,
            source_git_sha=task.artifact.source_commit,
            config_contract_sha256=_versioned_state_sha256(config_contract),
            seed_contract_sha256=_versioned_state_sha256(seed_contract),
            run_generation_id=run_generation_id,
            island_id=task.task_id,
        )
        genesis_identity = OpenEcologyAggregateIdentityPins(
            source_git_sha=task.artifact.source_commit,
            config_contract_sha256=_versioned_state_sha256(config_contract),
            seed_contract_sha256=_versioned_state_sha256(seed_contract),
            run_generation_id=run_generation_id,
            island_id=task.task_id,
            simulation_generation_index=0,
            tick=0,
        )
        task_lock_path = (
            campaign_path.parent
            / f".{campaign_path.name}.{task.task_id}.open-ecology-task.lock"
        )
        task_lock_identity = {
            "campaign_id": campaign_id,
            "schema_version": _OPEN_ECOLOGY_TASK_MUTATION_LOCK_SCHEMA_VERSION,
            "source_git_sha": task.artifact.source_commit,
            "task_id": task.task_id,
        }
        storage_lock = CampaignStorageLock(
            default_storage_lock_path(campaign_path),
            campaign_id=campaign_id,
            source_git_sha=task.artifact.source_commit,
        )
        writer: RotatingOpenEcologyEvidenceWriter | None = None
        with _IdentityFileLock(
            task_lock_path,
            identity=task_lock_identity,
            shared=False,
            nonblocking=True,
            initialize=True,
        ):
            if not default_storage_lock_path(campaign_path).exists():
                with storage_lock:
                    pass
            with _IdentityFileLock(
                default_storage_lock_path(campaign_path),
                identity=storage_lock.identity,
                shared=True,
                nonblocking=True,
            ):
                if evidence_path.exists() or evidence_path.is_symlink():
                    genesis = load_or_recover_open_ecology_genesis_generation(
                        aggregate_path,
                        evidence_directory=evidence_path,
                        identity=genesis_identity,
                    )
                    if genesis is not None:
                        aggregate_metadata = _aggregate_metadata_from_loaded_generation(
                            genesis
                        )
                        pins = _aggregate_resume_pins_from_metadata(aggregate_metadata)
                        commit = _mapping(
                            genesis.get("commit"),
                            field="genesis aggregate commit",
                        )
                        checkpoint_file = (
                            aggregate_path
                            / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
                            / str(commit["directory_name"])
                            / OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME
                        )
                        restored = cls._restore_from_checkpoint(
                            task,
                            checkpoint_path=checkpoint_file,
                            campaign_root=campaign_path,
                            campaign_id=campaign_id,
                            allow_aggregate_snapshot=True,
                            aggregate_metadata=aggregate_metadata,
                            cooperative_restore_locks_held=True,
                        )
                        restore_open_ecology_evidence_manifest_snapshot(
                            evidence_path,
                            authoritative_manifest=_mapping(
                                genesis.get("evidence_manifest"),
                                field="genesis evidence manifest",
                            ),
                            preservation_directory=(
                                campaign_path / "attempts" / task.task_id / "evidence"
                            ),
                        )
                        if restored.observed_tick != 0:
                            raise OpenEcologyPersistentIslandError(
                                "genesis CURRENT does not restore at observed tick zero"
                            )
                        reread = inspect_current_open_ecology_aggregate_generation(
                            aggregate_path,
                            evidence_directory=evidence_path,
                            pins=pins,
                        )
                        if reread != genesis:
                            raise OpenEcologyPersistentIslandError(
                                "genesis CURRENT changed during locked restore"
                            )
                        return restored, True
                    preserve_open_ecology_evidence_attempt_directory(
                        evidence_path,
                        preservation_directory=(
                            campaign_path
                            / "attempts"
                            / task.task_id
                            / "genesis-evidence"
                        ),
                    )
                elif _aggregate_genesis_evidence_is_required(aggregate_path):
                    raise OpenEcologyPersistentIslandError(
                        "materialized genesis aggregate evidence is missing"
                    )
                if checkpoint_path.exists() or checkpoint_path.is_symlink():
                    _preserve_genesis_checkpoint_attempt_directory(
                        checkpoint_path,
                        preservation_directory=(
                            campaign_path
                            / "attempts"
                            / task.task_id
                            / "genesis-checkpoints"
                        ),
                    )
                _preserve_genesis_aggregate_stages(
                    aggregate_path,
                    preservation_directory=(
                        campaign_path
                        / "attempts"
                        / task.task_id
                        / "genesis-aggregate-stages"
                    ),
                )
                try:
                    writer = RotatingOpenEcologyEvidenceWriter(
                        evidence_path,
                        run_id=task.task_id,
                        source_contract=source_contract,
                        config=concrete_writer_config,
                    )
                    world = SimulationWorld(
                        build_persistent_world_config(task),
                        policy=policy,
                    )
                    runner = cls(
                        task=task,
                        world=world,
                        policy=policy,
                        evidence_writer=writer,
                        artifact=artifact,
                        evidence_directory=evidence_path,
                        writer_config=concrete_writer_config,
                        writer_source_contract=source_contract,
                        campaign_root=campaign_path,
                        campaign_id=campaign_id,
                        checkpoint_directory=checkpoint_path,
                        aggregate_directory=aggregate_path,
                        checkpoint_interval_ticks=parsed_checkpoint_interval,
                        checkpoint_config_contract=config_contract,
                        checkpoint_seed_contract=seed_contract,
                        task_lock_already_held=True,
                    )
                    runner._checkpoint_active_writer()
                    runner._write_genesis_runtime_checkpoint_locked()
                    if runner.latest_aggregate_resume_pins is None:
                        raise OpenEcologyPersistentIslandError(
                            "fresh genesis did not publish restartable CURRENT"
                        )
                    return runner, False
                except Exception:
                    if writer is not None:
                        writer.abort()
                    raise

    @classmethod
    def reconcile_interval_attempt(
        cls,
        task: PersistentIslandTask,
        *,
        campaign_root: str | Path,
        campaign_id: str,
        predecessor: OpenEcologyAggregateResumePins | None,
        predecessor_observed_tick: int,
        predecessor_terminal: bool,
        target_tick: int,
        attempt_authority_sha256: str,
        evidence_directory: str | Path | None = None,
        checkpoint_directory: str | Path | None = None,
        aggregate_directory: str | Path | None = None,
        writer_config: RotatingEvidenceConfig | None = None,
        checkpoint_interval_ticks: int = (
            OPEN_ECOLOGY_RUNTIME_CHECKPOINT_INTERVAL_TICKS
        ),
    ) -> tuple[PersistentIslandRunner, str]:
        """Reconcile one original coordinator attempt without authorizing a new one."""

        campaign_path = Path(campaign_root)
        evidence_path = (
            campaign_path / task.task_id
            if evidence_directory is None
            else Path(evidence_directory)
        )
        checkpoint_path = (
            campaign_path / "checkpoints" / task.task_id
            if checkpoint_directory is None
            else Path(checkpoint_directory)
        )
        aggregate_path = (
            campaign_path / "aggregates" / task.task_id
            if aggregate_directory is None
            else Path(aggregate_directory)
        )
        if not all(
            path.is_absolute()
            for path in (
                campaign_path,
                evidence_path,
                checkpoint_path,
                aggregate_path,
            )
        ):
            raise OpenEcologyPersistentIslandError(
                "interval reconciliation paths must be absolute"
            )
        for field_name, path in (
            ("evidence_directory", evidence_path),
            ("checkpoint_directory", checkpoint_path),
            ("aggregate_directory", aggregate_path),
        ):
            try:
                relative = path.relative_to(campaign_path)
            except ValueError as error:
                raise OpenEcologyPersistentIslandError(
                    f"{field_name} must be inside campaign_root"
                ) from error
            if not relative.parts or ".." in relative.parts:
                raise OpenEcologyPersistentIslandError(
                    f"{field_name} must be a normalized campaign child"
                )
        predecessor_tick = _nonnegative_int(
            predecessor_observed_tick,
            field="predecessor_observed_tick",
        )
        if not isinstance(predecessor_terminal, bool):
            raise OpenEcologyPersistentIslandError(
                "predecessor_terminal must be boolean"
            )
        requested_target = _positive_int(target_tick, field="target_tick")
        parsed_checkpoint_interval = _positive_int(
            checkpoint_interval_ticks,
            field="checkpoint_interval_ticks",
        )
        if (
            requested_target > task.target_ticks
            or requested_target % parsed_checkpoint_interval != 0
            or requested_target < predecessor_tick
            or (not predecessor_terminal and requested_target <= predecessor_tick)
        ):
            raise OpenEcologyPersistentIslandError(
                "interval reconciliation target is outside the exact barrier contract"
            )
        run_generation_id = _persistent_checkpoint_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        )
        if predecessor is None:
            if predecessor_tick != 0 or predecessor_terminal:
                raise OpenEcologyPersistentIslandError(
                    "unpinned interval predecessor must be pending tick zero"
                )
        else:
            if not isinstance(predecessor, OpenEcologyAggregateResumePins):
                raise OpenEcologyPersistentIslandError(
                    "interval predecessor must be aggregate resume pins"
                )
            identity = predecessor.identity
            expected_completed_tick = (
                0 if predecessor_tick == 0 else predecessor_tick - 1
            )
            if (
                identity.source_git_sha != task.artifact.source_commit
                or identity.run_generation_id != run_generation_id
                or identity.island_id != task.task_id
                or identity.simulation_generation_index != 0
                or identity.tick != expected_completed_tick
            ):
                raise OpenEcologyPersistentIslandError(
                    "interval predecessor identity does not match task frontier"
                )
        expected_attempt_authority = persistent_interval_attempt_authority_sha256(
            task,
            campaign_id=campaign_id,
            predecessor=predecessor,
            target_tick=requested_target,
        )
        if (
            _sha256(
                attempt_authority_sha256,
                field="interval attempt authority SHA256",
            )
            != expected_attempt_authority
        ):
            raise OpenEcologyPersistentIslandError(
                "interval attempt authority does not match exact predecessor/target"
            )

        expected_target_aggregate_index = (
            1 if predecessor is None else predecessor.aggregate_generation_index + 1
        )
        incomplete_before = _interval_attempt_has_loose_or_staged_bytes(
            evidence_directory=evidence_path,
            checkpoint_directory=checkpoint_path,
            aggregate_directory=aggregate_path,
            target_observed_tick=requested_target,
            target_aggregate_generation_index=(expected_target_aggregate_index),
            predecessor_manifest_sha256=(
                None if predecessor is None else predecessor.evidence_manifest_sha256
            ),
        )

        # A genuinely unmaterialized task has no aggregate state to inspect.
        # Genesis owns its bootstrap locks and returns a selected tick-zero
        # CURRENT before the long-lived advance begins.
        if predecessor is None and not _aggregate_genesis_evidence_is_required(
            aggregate_path
        ):
            genesis_authority = persistent_genesis_attempt_authority_sha256(
                task,
                campaign_root=campaign_path,
                campaign_id=campaign_id,
                evidence_directory=evidence_path,
                checkpoint_directory=checkpoint_path,
                aggregate_directory=aggregate_path,
                writer_config=writer_config,
                checkpoint_interval_ticks=parsed_checkpoint_interval,
            )
            runner, recovered_genesis = cls.materialize_or_restore_genesis(
                task,
                campaign_root=campaign_path,
                campaign_id=campaign_id,
                attempt_authority_sha256=genesis_authority,
                evidence_directory=evidence_path,
                checkpoint_directory=checkpoint_path,
                aggregate_directory=aggregate_path,
                writer_config=writer_config,
                checkpoint_interval_ticks=parsed_checkpoint_interval,
            )
            runner.advance_to(requested_target)
            disposition = (
                "recovered_original_precurrent_attempt"
                if incomplete_before or recovered_genesis
                else "advanced_from_exact_predecessor"
            )
            return runner, disposition

        task_lock_path = (
            campaign_path.parent
            / f".{campaign_path.name}.{task.task_id}.open-ecology-task.lock"
        )
        task_lock_identity = {
            "campaign_id": campaign_id,
            "schema_version": _OPEN_ECOLOGY_TASK_MUTATION_LOCK_SCHEMA_VERSION,
            "source_git_sha": task.artifact.source_commit,
            "task_id": task.task_id,
        }
        storage_lock = CampaignStorageLock(
            default_storage_lock_path(campaign_path),
            campaign_id=campaign_id,
            source_git_sha=task.artifact.source_commit,
        )
        with _IdentityFileLock(
            task_lock_path,
            identity=task_lock_identity,
            shared=False,
            nonblocking=True,
            initialize=True,
        ):
            if not default_storage_lock_path(campaign_path).exists():
                with storage_lock:
                    pass
            with _IdentityFileLock(
                default_storage_lock_path(campaign_path),
                identity=storage_lock.identity,
                shared=True,
                nonblocking=True,
            ):
                frontier = inspect_open_ecology_aggregate_attempt_frontier(
                    aggregate_path,
                    evidence_directory=evidence_path,
                )
                current = frontier["current"]
                successor = frontier["complete_successor"]
                current_predecessor = frontier["current_predecessor"]
                current_index = _loaded_aggregate_index(current)
                successor_index = _loaded_aggregate_index(successor)
                if current_index == expected_target_aggregate_index:
                    if successor is not None:
                        raise OpenEcologyPersistentIslandError(
                            "interval attempt contains a generation beyond its target"
                        )
                    target_loaded = _mapping(
                        current,
                        field="attempt target CURRENT",
                    )
                    target_was_current = True
                    base_loaded = current_predecessor
                elif successor_index == expected_target_aggregate_index:
                    target_loaded = _mapping(
                        successor,
                        field="attempt target successor",
                    )
                    target_was_current = False
                    base_loaded = current
                else:
                    target_loaded = None
                    target_was_current = False
                    if predecessor is None:
                        base_loaded = (
                            current
                            if current_index == 0
                            else successor
                            if current is None and successor_index == 0
                            else None
                        )
                    else:
                        base_loaded = (
                            current
                            if current_index == predecessor.aggregate_generation_index
                            else None
                        )

                if target_loaded is not None:
                    runner, target_pins = cls._restore_loaded_aggregate_task_locked(
                        task,
                        loaded=target_loaded,
                        campaign_root=campaign_path,
                        campaign_id=campaign_id,
                        evidence_directory=evidence_path,
                        aggregate_directory=aggregate_path,
                        preservation_directory=(
                            campaign_path / "attempts" / task.task_id / "evidence"
                        ),
                    )
                    _validate_reconciled_target_runner(
                        runner,
                        pins=target_pins,
                        task=task,
                        campaign_id=campaign_id,
                        predecessor=predecessor,
                        predecessor_observed_tick=predecessor_tick,
                        predecessor_terminal=predecessor_terminal,
                        requested_target_tick=requested_target,
                        expected_aggregate_generation_index=(
                            expected_target_aggregate_index
                        ),
                        base_loaded=base_loaded,
                        target_loaded=target_loaded,
                    )
                    if not target_was_current:
                        selected = select_open_ecology_aggregate_attempt_generation(
                            aggregate_path,
                            evidence_directory=evidence_path,
                            pins=target_pins,
                            expected_previous_commit_sha256=(
                                _loaded_aggregate_commit_sha256(base_loaded)
                            ),
                        )
                        if selected != target_loaded:
                            raise OpenEcologyPersistentIslandError(
                                "selected attempt target changed after validation"
                            )
                    retained = inspect_current_open_ecology_aggregate_generation(
                        aggregate_path,
                        evidence_directory=evidence_path,
                        pins=target_pins,
                    )
                    if retained != target_loaded:
                        raise OpenEcologyPersistentIslandError(
                            "reconciled target CURRENT changed during locked restore"
                        )
                    return (
                        runner,
                        (
                            "already_committed_target"
                            if target_was_current
                            else "recovered_original_precurrent_attempt"
                        ),
                    )

                if base_loaded is None:
                    raise OpenEcologyPersistentIslandError(
                        "interval attempt aggregate frontier is not predecessor/target"
                    )
                runner, base_pins = cls._restore_loaded_aggregate_task_locked(
                    task,
                    loaded=_mapping(
                        base_loaded,
                        field="interval attempt predecessor",
                    ),
                    campaign_root=campaign_path,
                    campaign_id=campaign_id,
                    evidence_directory=evidence_path,
                    aggregate_directory=aggregate_path,
                    preservation_directory=(
                        campaign_path / "attempts" / task.task_id / "evidence"
                    ),
                )
                if predecessor is None:
                    _validate_exact_genesis_runner(
                        runner,
                        pins=base_pins,
                        task=task,
                        campaign_id=campaign_id,
                    )
                    if current_index is None:
                        select_open_ecology_aggregate_attempt_generation(
                            aggregate_path,
                            evidence_directory=evidence_path,
                            pins=base_pins,
                            expected_previous_commit_sha256=None,
                        )
                elif (
                    base_pins != predecessor
                    or runner.observed_tick != predecessor_tick
                    or runner.extinct is not predecessor_terminal
                ):
                    raise OpenEcologyPersistentIslandError(
                        "restored interval predecessor does not match intent"
                    )
                loose_target = checkpoint_path / (
                    f"checkpoint-observed-{requested_target:08d}.json"
                )
                if loose_target.exists() or loose_target.is_symlink():
                    _preserve_loose_interval_checkpoint(
                        loose_target,
                        preservation_directory=(
                            campaign_path
                            / "attempts"
                            / task.task_id
                            / "interval-checkpoints"
                        ),
                    )
                _preserve_aggregate_attempt_stages(
                    aggregate_path,
                    aggregate_generation_index=(expected_target_aggregate_index),
                    preservation_directory=(
                        campaign_path / "attempts" / task.task_id / "aggregate-stages"
                    ),
                )

            runner._advance_to_task_locked(requested_target)
            return (
                runner,
                (
                    "recovered_original_precurrent_attempt"
                    if incomplete_before
                    else "advanced_from_exact_predecessor"
                ),
            )

    @classmethod
    def _restore_loaded_aggregate_task_locked(
        cls,
        task: PersistentIslandTask,
        *,
        loaded: Mapping[str, object],
        campaign_root: Path,
        campaign_id: str,
        evidence_directory: Path,
        aggregate_directory: Path,
        preservation_directory: Path,
    ) -> tuple[PersistentIslandRunner, OpenEcologyAggregateResumePins]:
        aggregate_metadata = _aggregate_metadata_from_loaded_generation(loaded)
        pins = _aggregate_resume_pins_from_metadata(aggregate_metadata)
        commit = _mapping(loaded.get("commit"), field="aggregate commit")
        directory_name = commit.get("directory_name")
        if not isinstance(directory_name, str) or not directory_name:
            raise OpenEcologyPersistentIslandError(
                "aggregate commit directory name is invalid"
            )
        checkpoint_file = (
            aggregate_directory
            / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
            / directory_name
            / OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME
        )
        runner = cls._restore_from_checkpoint(
            task,
            checkpoint_path=checkpoint_file,
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            allow_aggregate_snapshot=True,
            aggregate_metadata=aggregate_metadata,
            cooperative_restore_locks_held=True,
        )
        restore_open_ecology_evidence_manifest_snapshot(
            evidence_directory,
            authoritative_manifest=_mapping(
                loaded.get("evidence_manifest"),
                field="aggregate evidence manifest",
            ),
            preservation_directory=preservation_directory,
        )
        if (
            runner._evidence_directory != evidence_directory
            or runner._aggregate_directory != aggregate_directory
        ):
            raise OpenEcologyPersistentIslandError(
                "aggregate attempt paths do not match checkpoint configuration"
            )
        return runner, pins

    @classmethod
    def restore_from_checkpoint(
        cls,
        task: PersistentIslandTask,
        *,
        checkpoint_path: str | Path,
        campaign_root: str | Path,
        campaign_id: str,
    ) -> PersistentIslandRunner:
        """Restore one loose checkpoint without claiming aggregate authority."""

        return cls._restore_from_checkpoint(
            task,
            checkpoint_path=checkpoint_path,
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            allow_aggregate_snapshot=False,
            aggregate_metadata=None,
            cooperative_restore_locks_held=False,
        )

    @classmethod
    def restore_from_current(
        cls,
        task: PersistentIslandTask,
        *,
        campaign_root: str | Path,
        campaign_id: str,
        pins: OpenEcologyAggregateResumePins,
        evidence_directory: str | Path | None = None,
        aggregate_directory: str | Path | None = None,
    ) -> PersistentIslandRunner:
        """Restore only the immutable aggregate generation selected by CURRENT."""

        campaign_path = Path(campaign_root)
        evidence_path = (
            campaign_path / task.task_id
            if evidence_directory is None
            else Path(evidence_directory)
        )
        aggregate_path = (
            campaign_path / "aggregates" / task.task_id
            if aggregate_directory is None
            else Path(aggregate_directory)
        )
        if (
            not campaign_path.is_absolute()
            or not evidence_path.is_absolute()
            or not aggregate_path.is_absolute()
        ):
            raise OpenEcologyPersistentIslandError(
                "aggregate restore paths must be absolute"
            )
        task_lock_path = (
            campaign_path.parent
            / f".{campaign_path.name}.{task.task_id}.open-ecology-task.lock"
        )
        task_lock_identity = {
            "campaign_id": campaign_id,
            "schema_version": _OPEN_ECOLOGY_TASK_MUTATION_LOCK_SCHEMA_VERSION,
            "source_git_sha": task.artifact.source_commit,
            "task_id": task.task_id,
        }
        storage_lock = CampaignStorageLock(
            default_storage_lock_path(campaign_path),
            campaign_id=campaign_id,
            source_git_sha=task.artifact.source_commit,
        )
        attempt_root = campaign_path / "attempts" / task.task_id
        with _IdentityFileLock(
            task_lock_path,
            identity=task_lock_identity,
            shared=False,
            nonblocking=True,
        ):
            with _IdentityFileLock(
                default_storage_lock_path(campaign_path),
                identity=storage_lock.identity,
                shared=True,
                nonblocking=True,
            ):
                loaded = load_current_open_ecology_aggregate_generation(
                    aggregate_path,
                    evidence_directory=evidence_path,
                    pins=pins,
                    uncommitted_successor_preservation_directory=(
                        attempt_root / "aggregate-generations"
                    ),
                )
                commit = _mapping(loaded.get("commit"), field="aggregate commit")
                directory_name = commit.get("directory_name")
                if not isinstance(directory_name, str) or not directory_name:
                    raise OpenEcologyPersistentIslandError(
                        "aggregate commit directory name is invalid"
                    )
                checkpoint_path = (
                    aggregate_path
                    / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
                    / directory_name
                    / OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME
                )
                aggregate_metadata = _aggregate_metadata_from_loaded_generation(loaded)
                runner = cls._restore_from_checkpoint(
                    task,
                    checkpoint_path=checkpoint_path,
                    campaign_root=campaign_path,
                    campaign_id=campaign_id,
                    allow_aggregate_snapshot=True,
                    aggregate_metadata=aggregate_metadata,
                    cooperative_restore_locks_held=True,
                )
                restore_open_ecology_evidence_manifest_snapshot(
                    evidence_path,
                    authoritative_manifest=_mapping(
                        loaded.get("evidence_manifest"),
                        field="aggregate evidence manifest",
                    ),
                    preservation_directory=attempt_root / "evidence",
                )
                reread = load_current_open_ecology_aggregate_generation(
                    aggregate_path,
                    evidence_directory=evidence_path,
                    pins=pins,
                    uncommitted_successor_preservation_directory=(
                        attempt_root / "aggregate-generations"
                    ),
                )
                if reread != loaded:
                    raise OpenEcologyPersistentIslandError(
                        "aggregate CURRENT changed during locked restore"
                    )
                if (
                    runner._evidence_directory != evidence_path
                    or runner._aggregate_directory != aggregate_path
                ):
                    raise OpenEcologyPersistentIslandError(
                        "aggregate CURRENT paths do not match checkpoint configuration"
                    )
                return runner

    @classmethod
    def _restore_from_checkpoint(
        cls,
        task: PersistentIslandTask,
        *,
        checkpoint_path: str | Path,
        campaign_root: str | Path,
        campaign_id: str,
        allow_aggregate_snapshot: bool,
        aggregate_metadata: Mapping[str, object] | None,
        cooperative_restore_locks_held: bool,
    ) -> PersistentIslandRunner:
        """Construct a fresh runtime from one validated interval checkpoint."""

        campaign_path = Path(campaign_root)
        checkpoint_file = Path(checkpoint_path)
        if not campaign_path.is_absolute() or not checkpoint_file.is_absolute():
            raise OpenEcologyPersistentIslandError(
                "campaign_root and checkpoint_path must be absolute paths"
            )
        checkpoint = load_open_ecology_checkpoint(
            checkpoint_file,
            max_checkpoint_bytes=OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
            expected_source_git_sha=task.artifact.source_commit,
            require_restartable=True,
        )
        source = _mapping(checkpoint.get("source"), field="checkpoint.source")
        _exact_keys(
            source,
            {"git_sha", "config_contract", "seed_contract"},
            field="checkpoint.source",
        )
        config_envelope = _mapping(
            source["config_contract"],
            field="checkpoint.source.config_contract",
        )
        seed_envelope = _mapping(
            source["seed_contract"],
            field="checkpoint.source.seed_contract",
        )
        config_contract = _checkpoint_state_from_envelope(
            config_envelope,
            expected_schema_version=(
                OPEN_ECOLOGY_PERSISTENT_CHECKPOINT_CONFIG_SCHEMA_VERSION
            ),
            field="checkpoint.source.config_contract",
        )
        seed_contract = _checkpoint_state_from_envelope(
            seed_envelope,
            expected_schema_version=(
                OPEN_ECOLOGY_PERSISTENT_CHECKPOINT_SEED_SCHEMA_VERSION
            ),
            field="checkpoint.source.seed_contract",
        )
        config_payload = _mapping(
            config_contract.payload,
            field="checkpoint config payload",
        )
        _exact_keys(
            config_payload,
            {
                "aggregate_directory_relative",
                "artifact_file_sha256",
                "artifact_sha256",
                "campaign_id",
                "checkpoint_directory_relative",
                "checkpoint_interval_ticks",
                "evidence_directory_relative",
                "model_state_sha256",
                "source_manifest_sha256",
                "task",
                "task_sha256",
                "world_config",
                "world_config_sha256",
                "writer_config",
            },
            field="checkpoint config payload",
        )
        if config_payload["campaign_id"] != campaign_id:
            raise OpenEcologyPersistentIslandError(
                "checkpoint campaign id does not match the requested campaign"
            )
        task_payload = task.to_dict()
        if config_payload["task"] != task_payload or config_payload[
            "task_sha256"
        ] != _stable_digest(task_payload):
            raise OpenEcologyPersistentIslandError(
                "checkpoint task binding does not match the requested task"
            )
        world_config = build_persistent_world_config(task)
        if (
            config_payload["world_config"] != world_config.to_dict()
            or config_payload["world_config_sha256"] != task.world_config_sha256
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint world configuration binding drifted"
            )
        if (
            config_payload["artifact_sha256"] != task.artifact.artifact_sha256
            or config_payload["artifact_file_sha256"]
            != task.artifact.artifact_file_sha256
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint frozen-artifact binding drifted"
            )
        expected_seed_contract = _persistent_checkpoint_seed_contract(task)
        if seed_contract != expected_seed_contract:
            raise OpenEcologyPersistentIslandError(
                "checkpoint seed contract does not match the task registry binding"
            )

        evidence_path = _restore_campaign_child(
            campaign_path,
            config_payload["evidence_directory_relative"],
            field="checkpoint evidence directory",
        )
        checkpoint_directory = _restore_campaign_child(
            campaign_path,
            config_payload["checkpoint_directory_relative"],
            field="checkpoint directory",
        )
        aggregate_directory = _restore_campaign_child(
            campaign_path,
            config_payload["aggregate_directory_relative"],
            field="checkpoint aggregate directory",
        )
        if (
            checkpoint_file.parent != checkpoint_directory
            and not allow_aggregate_snapshot
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint file is outside its bound task checkpoint directory"
            )
        if allow_aggregate_snapshot:
            if aggregate_metadata is None:
                raise OpenEcologyPersistentIslandError(
                    "aggregate snapshot restore metadata is missing"
                )
            directory_name = aggregate_metadata.get("directory_name")
            expected_snapshot = (
                aggregate_directory
                / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
                / str(directory_name)
                / OPEN_ECOLOGY_AGGREGATE_CHECKPOINT_NAME
            )
            if checkpoint_file != expected_snapshot:
                raise OpenEcologyPersistentIslandError(
                    "aggregate snapshot path does not match validated CURRENT"
                )
        checkpoint_interval = _positive_int(
            config_payload["checkpoint_interval_ticks"],
            field="checkpoint checkpoint_interval_ticks",
        )
        writer_config = RotatingEvidenceConfig.from_dict(
            _mapping(config_payload["writer_config"], field="checkpoint writer config")
        )
        completed_world_tick = _nonnegative_int(
            checkpoint.get("tick"),
            field="checkpoint tick",
        )

        generation_identity = _mapping(
            checkpoint.get("generation_identity"),
            field="checkpoint generation identity",
        )
        expected_generation_id = _persistent_checkpoint_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        )
        if (
            generation_identity.get("run_generation_id") != expected_generation_id
            or generation_identity.get("island_id") != task.task_id
            or generation_identity.get("generation_index") != 0
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint generation identity does not match the task"
            )
        binding = RuntimeCheckpointBinding(
            source_git_sha=task.artifact.source_commit,
            source_manifest_sha256=_sha256(
                config_payload["source_manifest_sha256"],
                field="checkpoint source_manifest_sha256",
            ),
            config_contract_sha256=_versioned_state_sha256(config_contract),
            seed_contract_sha256=_versioned_state_sha256(seed_contract),
            run_generation_id=expected_generation_id,
            island_id=task.task_id,
            generation_index=0,
            completed_tick=completed_world_tick,
        )
        if (
            config_envelope.get("state_sha256") != binding.config_contract_sha256
            or seed_envelope.get("state_sha256") != binding.seed_contract_sha256
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint contract state digest binding drifted"
            )
        components = _runtime_components_from_checkpoint(checkpoint)

        artifact_path = Path(task.artifact.artifact_path)
        if _file_sha256(artifact_path) != task.artifact.artifact_file_sha256:
            raise OpenEcologyPersistentIslandError(
                "frozen artifact file SHA256 does not match the task binding"
            )
        loaded = load_frozen_recurrent_policy_artifact(artifact_path)
        artifact = loaded.artifact
        _validate_persistent_artifact(task, artifact=artifact, model=loaded.model)
        _verify_live_source_authority(task, artifact=artifact)
        model_sha256 = recurrent_model_state_sha256(loaded.model)
        if config_payload["model_state_sha256"] != model_sha256:
            raise OpenEcologyPersistentIslandError(
                "checkpoint frozen model-state digest drifted"
            )
        if config_payload["source_manifest_sha256"] != (
            _artifact_source_manifest_sha256(artifact)
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint source-manifest binding drifted"
            )
        expected_config_contract = _persistent_checkpoint_config_contract(
            task,
            campaign_root=campaign_path,
            campaign_id=campaign_id,
            evidence_directory=evidence_path,
            checkpoint_directory=checkpoint_directory,
            aggregate_directory=aggregate_directory,
            checkpoint_interval_ticks=checkpoint_interval,
            writer_config=writer_config,
            artifact=artifact,
            model_state_sha256=model_sha256,
        )
        if config_contract != expected_config_contract:
            raise OpenEcologyPersistentIslandError(
                "checkpoint static configuration contract drifted"
            )
        expected_writer_source = persistent_island_writer_source_contract(
            task,
            model_state_sha256=model_sha256,
            source_git_sha=task.artifact.source_commit,
            config_contract_sha256=_versioned_state_sha256(config_contract),
            seed_contract_sha256=_versioned_state_sha256(seed_contract),
            run_generation_id=expected_generation_id,
            island_id=task.task_id,
        )
        policy = DeterministicPublicRecurrentPolicy(
            loaded.model,
            artifact_digest=task.artifact.artifact_sha256,
            copy_to_cpu=True,
            reset_recurrent_state_each_decision=(
                task.reset_recurrent_state_each_decision
            ),
            sampling_seed=task.policy_sampling_seed,
        )
        policy.start_world(
            world_identity=task.world_identity,
            genome_stream_seed=task.genome_stream_seed,
            genome_population_mode=task.genome_population_mode,
        )
        world = SimulationWorld(world_config, policy=policy)
        setattr(world, _OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_ATTRIBUTE, {})
        restored_continuation = restore_open_ecology_runtime_checkpoint(
            world,
            policy,
            binding=binding,
            components=components,
        )
        runner_state = _mapping(
            getattr(world, _OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_ATTRIBUTE, None),
            field="checkpoint runner state",
        )
        observed_tick = _nonnegative_int(
            runner_state.get("observed_tick"),
            field="checkpoint runner observed tick",
        )
        genesis_boundary = observed_tick == 0 and completed_world_tick == 0
        completed_boundary = (
            observed_tick > 0 and completed_world_tick == observed_tick - 1
        )
        if (
            not (genesis_boundary or completed_boundary)
            or runner_state.get("completed_world_tick") != completed_world_tick
        ):
            raise OpenEcologyPersistentIslandError(
                "checkpoint runner/world tick boundary is inconsistent"
            )
        checkpoint_sha256 = _sha256(
            checkpoint.get("checkpoint_sha256"),
            field="checkpoint SHA256",
        )
        if aggregate_metadata is not None and (
            aggregate_metadata.get("checkpoint_sha256") != checkpoint_sha256
            or aggregate_metadata.get("tick") != completed_world_tick
            or aggregate_metadata.get("evidence_manifest_sha256")
            != restored_continuation.get("manifest_sha256")
        ):
            raise OpenEcologyPersistentIslandError(
                "validated aggregate metadata does not match restored checkpoint"
            )
        checkpoint_metadata = {
            "path": str(checkpoint_file),
            "observed_tick": observed_tick,
            "completed_world_tick": completed_world_tick,
            "checkpoint_sha256": checkpoint_sha256,
            "generation_identity_sha256": _sha256(
                generation_identity.get("identity_sha256"),
                field="checkpoint generation identity SHA256",
            ),
            "evidence_continuation_sha256": _sha256(
                runner_state.get("evidence_continuation_sha256"),
                field="checkpoint runner evidence continuation SHA256",
            ),
            "restartable": True,
            "byte_size": checkpoint_file.stat().st_size,
        }

        def construct_runner() -> PersistentIslandRunner:
            return cls(
                task=task,
                world=world,
                policy=policy,
                evidence_writer=None,
                artifact=artifact,
                evidence_directory=evidence_path,
                writer_config=writer_config,
                writer_source_contract=expected_writer_source,
                campaign_root=campaign_path,
                campaign_id=campaign_id,
                checkpoint_directory=checkpoint_directory,
                aggregate_directory=aggregate_directory,
                checkpoint_interval_ticks=checkpoint_interval,
                checkpoint_config_contract=config_contract,
                checkpoint_seed_contract=seed_contract,
                restored_runner_state=runner_state,
                restored_evidence_continuation=restored_continuation,
                restored_checkpoint_metadata=checkpoint_metadata,
                restored_aggregate_metadata=aggregate_metadata,
                task_lock_already_held=cooperative_restore_locks_held,
            )

        if cooperative_restore_locks_held:
            return construct_runner()
        with CampaignStorageLock(
            default_storage_lock_path(campaign_path),
            campaign_id=campaign_id,
            source_git_sha=task.artifact.source_commit,
        ):
            return construct_runner()

    @property
    def world(self) -> SimulationWorld:
        return self._world

    @property
    def observed_tick(self) -> int:
        return self._next_tick

    @property
    def extinct(self) -> bool:
        return self._extinct

    @property
    def model_state_sha256(self) -> str:
        return self._model_state_sha256

    @property
    def frozen_policy_diagnostics(self) -> dict[str, object]:
        return {
            "model_state_sha256": self._model_state_sha256,
            "model_training": self._policy.model.training,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self._policy.model.parameters()
                if parameter.requires_grad
            ),
            "ppo_updates": 0,
            "optimizer_state_present": False,
        }

    @property
    def runtime_history_diagnostics(self) -> dict[str, object]:
        return {
            "reset_recurrent_state_each_decision": (
                self.task.reset_recurrent_state_each_decision
            ),
            "recurrent_state_agent_count": len(self._policy._state_by_agent),
            "previous_public_feedback_agent_count": len(
                self._policy._feedback_by_agent
            ),
            "pending_decision_count": len(self._policy._pending_by_agent),
        }

    @property
    def arm_contract(self) -> dict[str, object]:
        return {
            "arm": self.task.arm,
            **_arm_contract(self.task.arm),
        }

    @property
    def summaries(self) -> tuple[dict[str, object], ...]:
        return tuple(self._summaries)

    @property
    def first_milestone(self) -> dict[str, object] | None:
        milestone = self._milestones.get(OPEN_ECOLOGY_FIRST_MILESTONE_TICK)
        return None if milestone is None else _json_clone(milestone)

    @property
    def event_counts(self) -> dict[str, int]:
        return dict(sorted(self._event_counts.items()))

    @property
    def latest_runtime_checkpoint(self) -> dict[str, object] | None:
        return (
            None
            if self._latest_runtime_checkpoint is None
            else _json_clone(self._latest_runtime_checkpoint)
        )

    @property
    def latest_aggregate_resume_pins(
        self,
    ) -> OpenEcologyAggregateResumePins | None:
        if self._latest_aggregate_generation is None:
            return None
        return _aggregate_resume_pins_from_metadata(self._latest_aggregate_generation)

    @property
    def quiescent_checkpoint(self) -> dict[str, object]:
        """Return the exact archive-safe evidence boundary."""

        with self._task_mutation_lock():
            return self._quiescent_checkpoint_task_locked()

    def _quiescent_checkpoint_task_locked(self) -> dict[str, object]:
        if self._writer is not None or self._continuation_state is None:
            raise OpenEcologyPersistentIslandError(
                "persistent evidence is not at a quiescent checkpoint boundary"
            )
        current_restartable_checkpoint = (
            self._latest_runtime_checkpoint is not None
            and self._latest_runtime_checkpoint.get("observed_tick") == self._next_tick
            and self._latest_runtime_checkpoint.get("evidence_continuation_sha256")
            == self._continuation_state.get("state_sha256")
            and not self._evidence_finished
            and not self._evidence_aborted
        )
        checkpoint_authority = (
            None
            if self._latest_runtime_checkpoint is None
            else _json_clone(
                {
                    key: value
                    for key, value in self._latest_runtime_checkpoint.items()
                    if key != "profile"
                }
            )
        )
        payload: dict[str, object] = {
            "task_id": self.task.task_id,
            "campaign_id": self._campaign_id,
            "campaign_root": str(self._campaign_root),
            "storage_lock_path": str(self._storage_lock_path),
            "task_mutation_lock_path": str(self._task_lock_path),
            "source_git_sha": self.task.artifact.source_commit,
            "observed_tick": self._next_tick,
            "world_object_identity": self._world_object_identity,
            "same_in_memory_world": id(self._world) == self._world_object_identity,
            "model_state_sha256": self._model_state_sha256,
            "evidence_continuation": _json_clone(self._continuation_state),
            "evidence_finished": self._evidence_finished,
            "evidence_aborted": self._evidence_aborted,
            "archive_safe_while_runner_idle": True,
            "restartable_world_checkpoint_present": current_restartable_checkpoint,
            "latest_runtime_checkpoint": checkpoint_authority,
            "latest_aggregate_generation": (
                None
                if self._latest_aggregate_generation is None
                else _json_clone(self._latest_aggregate_generation)
            ),
            "aggregate_current_restartable": (
                current_restartable_checkpoint
                and self._latest_aggregate_generation is not None
                and self._latest_aggregate_generation.get("tick") == self._world.tick
                and self._latest_aggregate_generation.get("checkpoint_sha256")
                == self._latest_runtime_checkpoint.get("checkpoint_sha256")
            ),
            "aggregate_resume_authorized": self._aggregate_resume_authorized,
            "parallel_campaign_advance_authorized": (
                OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED
            ),
            "checkpoint_publication_concurrency": (
                checkpoint_publication_concurrency_contract()
            ),
            "launch_readiness": False,
        }
        payload["boundary_sha256"] = _stable_digest(payload)
        return payload

    def campaign_barrier_frontier(self) -> dict[str, object]:
        """Pin one idle task frontier for a coordinator-owned campaign barrier."""

        with self._task_mutation_lock():
            boundary = self._quiescent_checkpoint_task_locked()
            runtime = self._latest_runtime_checkpoint
            aggregate = self._latest_aggregate_generation
            if (
                runtime is None
                or aggregate is None
                or not boundary["restartable_world_checkpoint_present"]
                or not boundary["aggregate_current_restartable"]
                or not boundary["aggregate_resume_authorized"]
            ):
                raise OpenEcologyPersistentIslandError(
                    "campaign barrier requires an authorized restartable CURRENT"
                )
            expected_config_sha256 = _versioned_state_sha256(self._checkpoint_config)
            expected_seed_sha256 = _versioned_state_sha256(self._checkpoint_seeds)
            if (
                aggregate.get("source_git_sha") != self.task.artifact.source_commit
                or aggregate.get("config_contract_sha256") != expected_config_sha256
                or aggregate.get("seed_contract_sha256") != expected_seed_sha256
                or aggregate.get("run_generation_id")
                != _persistent_checkpoint_generation_id(
                    campaign_id=self._campaign_id,
                    task_id=self.task.task_id,
                )
                or aggregate.get("island_id") != self.task.task_id
                or aggregate.get("simulation_generation_index") != 0
                or aggregate.get("tick") != self._world.tick
                or aggregate.get("evidence_manifest_status") != "open"
                or aggregate.get("evidence_manifest_sha256")
                != self._continuation_state.get("manifest_sha256")
                or aggregate.get("evidence_continuation_sha256")
                != self._continuation_state.get("state_sha256")
            ):
                raise OpenEcologyPersistentIslandError(
                    "campaign barrier aggregate identity drifted"
                )
            _assert_current_pointer_matches(
                self._aggregate_directory,
                aggregate=aggregate,
            )
            payload: dict[str, object] = {
                "schema_version": (
                    _OPEN_ECOLOGY_CAMPAIGN_BARRIER_FRONTIER_SCHEMA_VERSION
                ),
                "task_id": self.task.task_id,
                "task_sha256": _stable_digest(self.task.to_dict()),
                "campaign_id": self._campaign_id,
                "campaign_root": str(self._campaign_root),
                "source_git_sha": self.task.artifact.source_commit,
                "observed_tick": self._next_tick,
                "extinct": self._extinct,
                "extinction_tick": self._extinction_tick,
                "config_contract_sha256": expected_config_sha256,
                "seed_contract_sha256": expected_seed_sha256,
                "world_config_sha256": self.task.world_config_sha256,
                "aggregate_generation_index": aggregate["aggregate_generation_index"],
                "aggregate_completed_world_tick": aggregate["tick"],
                "aggregate_commit_sha256": aggregate["commit_sha256"],
                "checkpoint_sha256": aggregate["checkpoint_sha256"],
                "checkpoint_generation_identity_sha256": aggregate[
                    "checkpoint_generation_identity_sha256"
                ],
                "evidence_manifest_sha256": aggregate["evidence_manifest_sha256"],
                "evidence_manifest_status": aggregate["evidence_manifest_status"],
                "evidence_continuation_sha256": aggregate[
                    "evidence_continuation_sha256"
                ],
                "aggregate_current_restartable": True,
                "aggregate_resume_authorized": True,
            }
            if (
                runtime["checkpoint_sha256"] != payload["checkpoint_sha256"]
                or runtime["generation_identity_sha256"]
                != payload["checkpoint_generation_identity_sha256"]
                or runtime["evidence_continuation_sha256"]
                != payload["evidence_continuation_sha256"]
            ):
                raise OpenEcologyPersistentIslandError(
                    "campaign barrier runtime and aggregate frontier drifted"
                )
            payload["frontier_sha256"] = _stable_digest(payload)
            return payload

    @classmethod
    @contextmanager
    def campaign_storage_barrier(
        cls,
        *,
        campaign_root: str | Path,
        campaign_id: str,
        source_git_sha: str,
        frontiers: Sequence[Mapping[str, object]],
    ) -> Iterator[tuple[dict[str, object], ...]]:
        """Exclude all listed task mutation while a coordinator scans storage.

        The runner validates receipt structure and exact campaign identity.  The
        coordinator remains responsible for joining every worker, rereading the
        complete expected task matrix, and applying cross-task semantic gates.
        """

        campaign_path = Path(campaign_root)
        if not campaign_path.is_absolute():
            raise CampaignStorageError("campaign barrier root must be absolute")
        normalized = tuple(
            sorted(
                (
                    _validated_campaign_barrier_frontier(
                        frontier,
                        campaign_root=campaign_path,
                        campaign_id=campaign_id,
                        source_git_sha=source_git_sha,
                    )
                    for frontier in frontiers
                ),
                key=lambda frontier: str(frontier["task_id"]),
            )
        )
        if not normalized:
            raise CampaignStorageError(
                "campaign storage barrier requires at least one frontier"
            )
        task_ids = tuple(str(frontier["task_id"]) for frontier in normalized)
        if len(set(task_ids)) != len(task_ids):
            raise CampaignStorageError(
                "campaign storage barrier frontiers contain duplicate task ids"
            )

        with ExitStack() as locks:
            locks.enter_context(
                CampaignStorageLock(
                    default_storage_lock_path(campaign_path),
                    campaign_id=campaign_id,
                    source_git_sha=source_git_sha,
                )
            )
            for frontier in normalized:
                task_id = str(frontier["task_id"])
                locks.enter_context(
                    _IdentityFileLock(
                        campaign_path.parent
                        / (f".{campaign_path.name}.{task_id}.open-ecology-task.lock"),
                        identity={
                            "campaign_id": campaign_id,
                            "schema_version": (
                                _OPEN_ECOLOGY_TASK_MUTATION_LOCK_SCHEMA_VERSION
                            ),
                            "source_git_sha": source_git_sha,
                            "task_id": task_id,
                        },
                        shared=False,
                        nonblocking=True,
                    )
                )
            yield tuple(_json_clone(frontier) for frontier in normalized)

    def advance_to(self, target_tick: int) -> PersistentAdvanceResult:
        with self._task_mutation_lock():
            return self._advance_to_task_locked(target_tick)

    def _advance_to_task_locked(
        self,
        target_tick: int,
    ) -> PersistentAdvanceResult:
        if (
            isinstance(target_tick, bool)
            or not isinstance(target_tick, int)
            or target_tick < self._next_tick
            or target_tick > self.task.target_ticks
        ):
            raise OpenEcologyPersistentIslandError(
                "target_tick must advance monotonically within [current, 50,000]"
            )
        starting_summary_count = len(self._summaries)
        starting_milestones = set(self._milestones)
        self._verify_invariants()

        if self._next_tick < target_tick and not self._extinct:
            while self._next_tick < target_tick and not self._extinct:
                next_scheduled_boundary = (
                    (self._next_tick // self._checkpoint_interval_ticks) + 1
                ) * self._checkpoint_interval_ticks
                segment_target = min(target_tick, next_scheduled_boundary)
                with self._campaign_shared_advance_lock():
                    self._resume_writer()
                    try:
                        while self._next_tick < segment_target and not self._extinct:
                            if not self._world.alive_agents():
                                self._mark_extinct()
                                break
                            self._world.tick = self._next_tick
                            births, deaths = self._world._run_tick()
                            self._next_tick += 1
                            self._record_tick_facts(births=births, deaths=deaths)
                            self._interval.population_auc += len(
                                self._world.alive_agents()
                            )
                            if (
                                self._next_tick % OPEN_ECOLOGY_SUMMARY_INTERVAL_TICKS
                                == 0
                            ):
                                self._emit_sampled_spatial_associations()
                                self._summaries.append(
                                    self._build_summary(terminal=False)
                                )
                                self._interval.reset()
                            if (
                                self._next_tick == OPEN_ECOLOGY_FIRST_MILESTONE_TICK
                                and OPEN_ECOLOGY_FIRST_MILESTONE_TICK
                                not in self._milestones
                            ):
                                self._milestones[OPEN_ECOLOGY_FIRST_MILESTONE_TICK] = (
                                    self._build_milestone()
                                )
                            if not self._world.alive_agents():
                                self._mark_extinct()
                        self._checkpoint_active_writer()
                    except Exception:
                        self._abort_active_writer()
                        raise
                    if self._next_tick > 0:
                        self._write_runtime_checkpoint_locked()

        if (
            target_tick >= OPEN_ECOLOGY_FIRST_MILESTONE_TICK
            and OPEN_ECOLOGY_FIRST_MILESTONE_TICK not in self._milestones
            and self._extinct
        ):
            with self._campaign_shared_publication_lock():
                self._milestones[OPEN_ECOLOGY_FIRST_MILESTONE_TICK] = (
                    self._build_milestone()
                )
                try:
                    if self._next_tick > 0:
                        self._write_runtime_checkpoint_locked()
                except Exception:
                    del self._milestones[OPEN_ECOLOGY_FIRST_MILESTONE_TICK]
                    raise
        self._verify_invariants()
        return PersistentAdvanceResult(
            task_id=self.task.task_id,
            requested_target_tick=target_tick,
            observed_tick=self._next_tick,
            extinct=self._extinct,
            extinction_tick=self._extinction_tick,
            summaries=tuple(self._summaries[starting_summary_count:]),
            milestones=tuple(
                _json_clone(self._milestones[tick])
                for tick in sorted(set(self._milestones) - starting_milestones)
            ),
            model_state_sha256=self._model_state_sha256,
            same_world_instance=id(self._world) == self._world_object_identity,
        )

    def record_intervention(
        self,
        *,
        intervention_id: str,
        intervention_kind: str,
        branch_id: str,
        subject_agent_id: int | None = None,
        assigned_action: str | None = None,
        value_sha256: str | None = None,
    ) -> None:
        """Record an external frozen-branch receipt without mutating the island."""

        with self._task_mutation_lock():
            tick = max(0, self._next_tick - 1)
            if self._last_evidence_tick is not None and tick < self._last_evidence_tick:
                raise OpenEcologyPersistentIslandError(
                    "intervention receipt predates the durable evidence prefix"
                )
            with self._campaign_shared_advance_lock():
                self._resume_writer()
                try:
                    self._append_event(
                        InterventionEvidence(
                            tick=tick,
                            event_index=self._next_event_index,
                            event_id=self._event_id("intervention"),
                            intervention_id=intervention_id,
                            intervention_kind=intervention_kind,
                            branch_id=branch_id,
                            subject_agent_id=subject_agent_id,
                            assigned_action=assigned_action,
                            value_sha256=value_sha256,
                        )
                    )
                    self._checkpoint_active_writer()
                except Exception:
                    self._abort_active_writer()
                    raise
                if self._next_tick > 0:
                    self._write_runtime_checkpoint_locked()

    def finish_evidence(self) -> Mapping[str, object]:
        with self._task_mutation_lock():
            if not self._extinct and self._next_tick != self.task.target_ticks:
                raise OpenEcologyPersistentIslandError(
                    "primary evidence can finish only at extinction or tick 50,000"
                )
            self._verify_invariants()
            with self._campaign_shared_publication_lock():
                self._resume_writer()
                assert self._writer is not None
                try:
                    manifest = self._writer.finish()
                except Exception:
                    self._abort_active_writer()
                    raise
                self._writer = None
                self._evidence_finished = True
                return manifest

    def abort_evidence(self) -> None:
        with self._task_mutation_lock():
            if self._evidence_finished or self._evidence_aborted:
                return
            with self._campaign_shared_publication_lock():
                self._resume_writer()
                self._abort_active_writer()
                self._evidence_aborted = True

    def _mark_extinct(self) -> None:
        if not self._extinct:
            self._extinct = True
            self._extinction_tick = self._next_tick
            if (
                not self._summaries
                or self._summaries[-1].get("observed_tick") != self._next_tick
            ):
                self._summaries.append(self._build_summary(terminal=True))
                self._interval.reset()

    def _emit_founder_lineages(self) -> None:
        for agent in sorted(self._world.alive_agents(), key=lambda item: item.agent_id):
            self._append_event(
                LineageEvidence(
                    tick=0,
                    event_index=self._next_event_index,
                    event_id=self._event_id("lineage"),
                    agent_id=agent.agent_id,
                    lineage_id=_lineage_id(agent.lineage_id),
                    parent_lineage_ids=(),
                    transition_kind="founder",
                )
            )

    def _record_tick_facts(self, *, births: int, deaths: int) -> None:
        world = self._world
        self._interval.births += births
        self._interval.deaths += deaths
        self._interval.attacks += len(world.tick_attack_events)
        self._interval.successful_attacks += sum(
            bool(event["success"]) for event in world.tick_attack_events
        )
        self._interval.feeding_events += len(world.tick_feeding_events)
        self._interval.reproduction_events += len(
            world.tick_reproduction_parent_child_groups
        )
        self._interval.signal_emissions += len(world.tick_signal_emission_events)

        for parent_ids, child_id in world.tick_reproduction_parent_child_groups:
            child = world.agents[child_id]
            genome_sha256 = _optional_genome_sha256(child.mind_inheritance_metadata)
            child_lineage_id = _lineage_id(child.lineage_id)
            parent_lineage_ids = tuple(
                sorted(
                    {
                        _lineage_id(world.agents[parent_id].lineage_id)
                        for parent_id in parent_ids
                    }
                )
            )
            self._append_event(
                BirthEvidence(
                    tick=world.tick,
                    event_index=self._next_event_index,
                    event_id=self._event_id("birth"),
                    child_agent_id=child_id,
                    parent_agent_ids=tuple(parent_ids),
                    lineage_id=child_lineage_id,
                    genome_sha256=genome_sha256,
                )
            )
            if child_lineage_id not in parent_lineage_ids:
                self._append_event(
                    LineageEvidence(
                        tick=world.tick,
                        event_index=self._next_event_index,
                        event_id=self._event_id("lineage"),
                        agent_id=child_id,
                        lineage_id=child_lineage_id,
                        parent_lineage_ids=parent_lineage_ids,
                        transition_kind="new_child_lineage",
                    )
                )
            if len(parent_ids) == 2 and parent_ids[0] != parent_ids[1]:
                self._append_event(
                    DyadicInteractionEvidence(
                        tick=world.tick,
                        event_index=self._next_event_index,
                        event_id=self._event_id("dyadic"),
                        actor_agent_id=parent_ids[0],
                        target_agent_id=parent_ids[1],
                        interaction_kind="mating",
                        outcome="offspring_born",
                        magnitude=1.0,
                    )
                )

        for event in world.tick_death_events:
            self._append_event(
                DeathEvidence(
                    tick=world.tick,
                    event_index=self._next_event_index,
                    event_id=self._event_id("death"),
                    agent_id=int(event["agent_id"]),
                    cause=str(event["cause"]),
                    source_agent_id=(
                        None
                        if event.get("killer_id") is None
                        else int(event["killer_id"])
                    ),
                )
            )

        for event in world.tick_attack_events:
            self._append_event(
                DyadicInteractionEvidence(
                    tick=world.tick,
                    event_index=self._next_event_index,
                    event_id=self._event_id("dyadic"),
                    actor_agent_id=int(event["attacker_id"]),
                    target_agent_id=int(event["target_id"]),
                    interaction_kind="attack",
                    outcome=(
                        "kill"
                        if bool(event["kill"])
                        else "damage"
                        if bool(event["success"])
                        else "failed"
                    ),
                    magnitude=float(event["damage"]),
                )
            )

        self._emit_signal_contributors()
        self._emit_congestion_events()
        for record in world.tick_trajectory_records:
            requested = str(record["requested_action"])
            resolved = str(record["resolved_action"])
            if requested not in ACTION_NAMES or resolved not in ACTION_NAMES:
                raise OpenEcologyPersistentIslandError(
                    "trajectory action left the stable action contract"
                )
            if record.get("action_source") not in {
                RECURRENT_ROLLOUT_ACTION_SOURCE,
                "passive",
            }:
                raise OpenEcologyPersistentIslandError(
                    "persistent island observed a hidden action source"
                )
            self._interval.requested_actions[requested] += 1
            self._interval.resolved_actions[resolved] += 1
            self._interval.invalid_observation_actions += int(
                not bool(record["action_valid"])
            )
            self._interval.invalid_resolution_actions += int(
                not bool(record["resolution_action_valid"])
            )
            outcome = _mapping(record.get("outcome"), field="trajectory outcome")
            drinking = _mapping(outcome.get("drinking"), field="drinking outcome")
            self._interval.drinking_events += int(bool(drinking.get("drank")))

    def _emit_signal_contributors(self) -> None:
        alive = self._world.alive_agents()
        for emission in self._world.tick_signal_emission_events:
            token_id = emission.get("token_id")
            emitter_id = emission.get("source_agent_id")
            if token_id is None or emitter_id is None:
                continue
            emitter = int(emitter_id)
            radius = int(emission["radius"])
            intensity = float(emission["intensity"])
            source_x = int(emission["x"])
            source_y = int(emission["y"])
            for receiver in sorted(alive, key=lambda item: item.agent_id):
                if receiver.agent_id == emitter:
                    continue
                distance = abs(receiver.x - source_x) + abs(receiver.y - source_y)
                if distance > radius:
                    continue
                self._append_event(
                    SignalContributorEvidence(
                        tick=self._world.tick,
                        event_index=self._next_event_index,
                        event_id=self._event_id("signal"),
                        emitter_agent_id=emitter,
                        receiver_agent_id=receiver.agent_id,
                        token_id=int(token_id),
                        contribution=intensity / (distance + 1.0),
                    )
                )

    def _emit_congestion_events(self) -> None:
        contenders: dict[tuple[int, int], list[Mapping[str, object]]] = {}
        for record in self._world.tick_trajectory_records:
            requested = str(record["requested_action"])
            delta = _MOVE_DELTAS.get(requested)
            if delta is None:
                continue
            before = _mapping(record.get("before"), field="trajectory before")
            target = (int(before["x"]) + delta[0], int(before["y"]) + delta[1])
            contenders.setdefault(target, []).append(record)
        for (target_x, target_y), records in sorted(contenders.items()):
            if len(records) < 2:
                continue
            moved = [record for record in records if bool(record["moved"])]
            if len(moved) > 1:
                raise OpenEcologyPersistentIslandError(
                    "multiple agents moved into one congestion target"
                )
            winner = None if not moved else int(moved[0]["agent_id"])
            self._append_event(
                CongestionEvidence(
                    tick=self._world.tick,
                    event_index=self._next_event_index,
                    event_id=self._event_id("congestion"),
                    target_x=target_x,
                    target_y=target_y,
                    contender_agent_ids=tuple(
                        sorted(int(record["agent_id"]) for record in records)
                    ),
                    winner_agent_id=winner,
                    outcome="one_winner" if winner is not None else "all_blocked",
                )
            )
            self._interval.congestion_events += 1

    def _emit_sampled_spatial_associations(self) -> None:
        positions = {
            (agent.x, agent.y): agent.agent_id for agent in self._world.alive_agents()
        }
        pairs: set[tuple[int, int]] = set()
        for (x, y), agent_id in sorted(positions.items()):
            for target in ((x + 1, y), (x, y + 1)):
                other_id = positions.get(target)
                if other_id is not None:
                    pairs.add(tuple(sorted((agent_id, other_id))))
        for actor_id, target_id in sorted(pairs):
            self._append_event(
                DyadicInteractionEvidence(
                    tick=max(0, self._next_tick - 1),
                    event_index=self._next_event_index,
                    event_id=self._event_id("dyadic"),
                    actor_agent_id=actor_id,
                    target_agent_id=target_id,
                    interaction_kind="adjacent_at_100_tick_sample",
                    outcome="observed",
                    magnitude=1.0,
                )
            )

    def _build_summary(self, *, terminal: bool) -> dict[str, object]:
        alive = self._world.alive_agents()
        lineage_sizes = Counter(agent.lineage_id for agent in alive)
        controller_genomes = Counter(
            digest
            for agent in alive
            if (digest := _optional_genome_sha256(agent.mind_inheritance_metadata))
            is not None
        )
        generation_by_agent = _generation_depths(self._world)
        completed_generations = sum(
            all(
                not agent.alive
                for agent in self._world.agents.values()
                if generation_by_agent[agent.agent_id] == generation
            )
            for generation in set(generation_by_agent.values())
        )
        trophic_roles, meat_modes = self._world._population_trophic_counts(alive)
        summary: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_PERSISTENT_SUMMARY_SCHEMA_VERSION,
            "task_id": self.task.task_id,
            "arm": self.task.arm,
            "observed_tick": self._next_tick,
            "window_ticks": (
                self._next_tick % OPEN_ECOLOGY_SUMMARY_INTERVAL_TICKS
                or OPEN_ECOLOGY_SUMMARY_INTERVAL_TICKS
            ),
            "terminal": terminal,
            "extinct": len(alive) == 0,
            "population": len(alive),
            "population_auc": self._interval.population_auc,
            "births": self._interval.births,
            "deaths": self._interval.deaths,
            "vitality": {
                "energy_ratio": _series_stats(
                    agent.energy / max(agent.genome.max_energy, 1e-12)
                    for agent in alive
                ),
                "hydration_ratio": _series_stats(
                    agent.hydration / max(agent.genome.max_hydration, 1e-12)
                    for agent in alive
                ),
                "health_ratio": _series_stats(
                    agent.health / max(agent.max_health, 1e-12) for agent in alive
                ),
            },
            "resources": {
                "vegetation": _series_stats(
                    tile.vegetation
                    for row in self._world.grid
                    for tile in row
                    if tile.terrain != "water"
                ),
                "fresh_kill_energy": round(
                    sum(
                        tile.fresh_kill_energy
                        for row in self._world.grid
                        for tile in row
                    ),
                    8,
                ),
                "carcass_energy": round(
                    sum(
                        tile.carcass_energy for row in self._world.grid for tile in row
                    ),
                    8,
                ),
            },
            "lineages": {
                "alive_count": len(lineage_sizes),
                "hill_effective_q1": _hill_effective_q1(lineage_sizes),
                "maximum_generation_depth": max(
                    (generation_by_agent[agent.agent_id] for agent in alive),
                    default=0,
                ),
                "completed_generation_count": completed_generations,
                "founder_lineages_alive": sum(
                    1
                    for agent in self._world.agents.values()
                    if agent.parent_id is None and agent.lineage_id in lineage_sizes
                ),
            },
            "controller_genomes": {
                "population_mode": self.task.genome_population_mode,
                "unique_alive_digests": len(controller_genomes),
                "hill_effective_q1": _hill_effective_q1(controller_genomes),
            },
            "trophic_role_occupancy": dict(sorted(trophic_roles.items())),
            "meat_mode_occupancy": dict(sorted(meat_modes.items())),
            "actions": {
                "requested": dict(sorted(self._interval.requested_actions.items())),
                "resolved": dict(sorted(self._interval.resolved_actions.items())),
                "invalid_observation": self._interval.invalid_observation_actions,
                "invalid_resolution": self._interval.invalid_resolution_actions,
            },
            "interactions": {
                "attacks": self._interval.attacks,
                "successful_attacks": self._interval.successful_attacks,
                "feeding_events": self._interval.feeding_events,
                "drinking_events": self._interval.drinking_events,
                "reproduction_events": self._interval.reproduction_events,
                "signal_emissions": self._interval.signal_emissions,
                "congestion_events": self._interval.congestion_events,
            },
            "evidence_event_counts": self.event_counts,
            "model_state_sha256": self._model_state_sha256,
            "outcome_used_for_tuning": False,
        }
        summary["summary_sha256"] = _stable_digest(summary)
        return summary

    def _build_milestone(self) -> dict[str, object]:
        current_model_sha256 = recurrent_model_state_sha256(self._policy.model)
        latest_summary = (
            self._summaries[-1]
            if self._summaries
            else self._build_summary(terminal=self._extinct)
        )
        milestone: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_PERSISTENT_MILESTONE_SCHEMA_VERSION,
            "scheduled_tick": OPEN_ECOLOGY_FIRST_MILESTONE_TICK,
            "observed_tick": self._next_tick,
            "task": self.task.to_dict(),
            "task_sha256": _stable_digest(self.task.to_dict()),
            "summary": _json_clone(latest_summary),
            "extinct": self._extinct,
            "extinction_tick": self._extinction_tick,
            "world_continuity": {
                "same_in_memory_world": id(self._world) == self._world_object_identity,
                "episode_reset_count": self._episode_reset_count,
                "world_replacement_count": self._world_replacement_count,
                "continues_toward_tick": self.task.target_ticks,
                "restart_from_milestone_checkpoint": True,
                "reseed_after_extinction": False,
            },
            "frozen_policy": {
                "artifact_sha256": self.task.artifact.artifact_sha256,
                "initial_model_state_sha256": self._model_state_sha256,
                "milestone_model_state_sha256": current_model_sha256,
                "unchanged": current_model_sha256 == self._model_state_sha256,
                "ppo_updates": 0,
                "optimizer_state_present": False,
            },
            "evidence_event_counts": self.event_counts,
            "causal_intervention_receipts": self._event_counts["intervention"],
            "outcome_used_for_tuning": False,
            "interim_acceptance_gate": False,
            "launch_readiness": False,
            "launch_blockers": list(OPEN_ECOLOGY_PERSISTENT_LAUNCH_BLOCKERS),
            "atomic_restartable_bundle_emitted": True,
            "atomic_bundle_blocker": None,
        }
        milestone["milestone_sha256"] = _stable_digest(milestone)
        return milestone

    def _append_event(self, event: OpenEcologyEvidenceEvent) -> None:
        if self._writer is None:
            raise OpenEcologyPersistentIslandError(
                "evidence append attempted outside a locked mutable chunk"
            )
        self._writer.append(event)
        self._event_counts[event.to_record()["event_type"]] += 1
        self._last_evidence_tick = event.tick
        self._next_event_index += 1
        expected_total = self._writer_initial_event_count + sum(
            self._event_counts.values()
        )
        if self._writer.diagnostics.get("total_event_count") != expected_total:
            raise OpenEcologyPersistentIslandError(
                "evidence writer event accounting drifted"
            )

    def _runner_state_payload(self) -> dict[str, object]:
        if self._writer is not None or self._continuation_state is None:
            raise OpenEcologyPersistentIslandError(
                "runtime checkpoint requires a quiescent evidence writer"
            )
        continuation_sha256 = _sha256(
            self._continuation_state.get("state_sha256"),
            field="evidence continuation SHA256",
        )
        payload: dict[str, object] = {
            "schema_version": OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_SCHEMA_VERSION,
            "task_sha256": _stable_digest(self.task.to_dict()),
            "model_state_sha256": self._model_state_sha256,
            "observed_tick": self._next_tick,
            "completed_world_tick": self._world.tick,
            "extinct": self._extinct,
            "extinction_tick": self._extinction_tick,
            "episode_reset_count": self._episode_reset_count,
            "world_replacement_count": self._world_replacement_count,
            "interval": self._interval.to_dict(),
            "summaries": [_json_clone(summary) for summary in self._summaries],
            "milestones": [
                {
                    "scheduled_tick": tick,
                    "payload": _json_clone(self._milestones[tick]),
                }
                for tick in sorted(self._milestones)
            ],
            "event_counts": dict(sorted(self._event_counts.items())),
            "writer_initial_event_count": self._writer_initial_event_count,
            "next_event_index": self._next_event_index,
            "last_evidence_tick": self._last_evidence_tick,
            "evidence_continuation_sha256": continuation_sha256,
            "evidence_finished": self._evidence_finished,
            "evidence_aborted": self._evidence_aborted,
        }
        payload["runner_state_sha256"] = _stable_digest(payload)
        return payload

    def _restore_runner_state(
        self,
        payload: Mapping[str, object],
        *,
        evidence_continuation: Mapping[str, object],
    ) -> None:
        expected_keys = {
            "schema_version",
            "task_sha256",
            "model_state_sha256",
            "observed_tick",
            "completed_world_tick",
            "extinct",
            "extinction_tick",
            "episode_reset_count",
            "world_replacement_count",
            "interval",
            "summaries",
            "milestones",
            "event_counts",
            "writer_initial_event_count",
            "next_event_index",
            "last_evidence_tick",
            "evidence_continuation_sha256",
            "evidence_finished",
            "evidence_aborted",
            "runner_state_sha256",
        }
        _exact_keys(payload, expected_keys, field="runner_state")
        if (
            payload["schema_version"]
            != OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_SCHEMA_VERSION
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent runner-state schema version drifted"
            )
        observed_runner_sha256 = _sha256(
            payload["runner_state_sha256"],
            field="runner_state.runner_state_sha256",
        )
        if observed_runner_sha256 != _stable_digest(
            {
                key: value
                for key, value in payload.items()
                if key != "runner_state_sha256"
            }
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent runner-state digest mismatch"
            )
        if (
            payload["task_sha256"] != _stable_digest(self.task.to_dict())
            or payload["model_state_sha256"] != self._model_state_sha256
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent runner state is bound to a different task or model"
            )
        self._next_tick = _nonnegative_int(
            payload["observed_tick"],
            field="runner_state.observed_tick",
        )
        completed_world_tick = _nonnegative_int(
            payload["completed_world_tick"],
            field="runner_state.completed_world_tick",
        )
        genesis_boundary = self._next_tick == 0 and completed_world_tick == 0
        completed_boundary = (
            self._next_tick > 0 and completed_world_tick == self._next_tick - 1
        )
        if (
            not (genesis_boundary or completed_boundary)
            or self._world.tick != completed_world_tick
        ):
            raise OpenEcologyPersistentIslandError(
                "restored runner/world tick boundary is inconsistent"
            )
        self._extinct = _exact_bool(payload["extinct"], field="runner_state.extinct")
        extinction_tick = payload["extinction_tick"]
        if extinction_tick is not None:
            extinction_tick = _nonnegative_int(
                extinction_tick,
                field="runner_state.extinction_tick",
            )
        if (self._extinct and extinction_tick is None) or (
            not self._extinct and extinction_tick is not None
        ):
            raise OpenEcologyPersistentIslandError(
                "runner extinction flag and tick disagree"
            )
        self._extinction_tick = extinction_tick
        self._episode_reset_count = _nonnegative_int(
            payload["episode_reset_count"],
            field="runner_state.episode_reset_count",
        )
        self._world_replacement_count = _nonnegative_int(
            payload["world_replacement_count"],
            field="runner_state.world_replacement_count",
        )
        self._interval = _IntervalCounters.from_dict(
            _mapping(payload["interval"], field="runner_state.interval")
        )
        summaries = _sequence(payload["summaries"], field="runner_state.summaries")
        self._summaries = []
        previous_summary_tick = -1
        for index, value in enumerate(summaries):
            summary = _mapping(
                value,
                field=f"runner_state.summaries[{index}]",
            )
            if (
                summary.get("schema_version")
                != OPEN_ECOLOGY_PERSISTENT_SUMMARY_SCHEMA_VERSION
                or summary.get("task_id") != self.task.task_id
            ):
                raise OpenEcologyPersistentIslandError(
                    "restored periodic summary identity drifted"
                )
            _validate_embedded_digest(
                summary,
                digest_field="summary_sha256",
                field=f"runner_state.summaries[{index}]",
            )
            summary_tick = _nonnegative_int(
                summary.get("observed_tick"),
                field=f"runner_state.summaries[{index}].observed_tick",
            )
            if summary_tick < previous_summary_tick or summary_tick > self._next_tick:
                raise OpenEcologyPersistentIslandError(
                    "restored periodic summary tick order drifted"
                )
            previous_summary_tick = summary_tick
            self._summaries.append(_json_clone(summary))

        milestones = _sequence(
            payload["milestones"],
            field="runner_state.milestones",
        )
        self._milestones = {}
        for index, value in enumerate(milestones):
            entry = _mapping(value, field=f"runner_state.milestones[{index}]")
            _exact_keys(
                entry,
                {"scheduled_tick", "payload"},
                field=f"runner_state.milestones[{index}]",
            )
            scheduled_tick = _nonnegative_int(
                entry["scheduled_tick"],
                field=f"runner_state.milestones[{index}].scheduled_tick",
            )
            milestone = _mapping(
                entry["payload"],
                field=f"runner_state.milestones[{index}].payload",
            )
            if (
                scheduled_tick in self._milestones
                or milestone.get("schema_version")
                != OPEN_ECOLOGY_PERSISTENT_MILESTONE_SCHEMA_VERSION
                or milestone.get("scheduled_tick") != scheduled_tick
            ):
                raise OpenEcologyPersistentIslandError(
                    "restored milestone identity drifted"
                )
            _validate_embedded_digest(
                milestone,
                digest_field="milestone_sha256",
                field=f"runner_state.milestones[{index}].payload",
            )
            self._milestones[scheduled_tick] = _json_clone(milestone)

        self._event_counts = _event_counter(
            payload["event_counts"],
            field="runner_state.event_counts",
        )
        self._writer_initial_event_count = _nonnegative_int(
            payload["writer_initial_event_count"],
            field="runner_state.writer_initial_event_count",
        )
        self._next_event_index = _nonnegative_int(
            payload["next_event_index"],
            field="runner_state.next_event_index",
        )
        last_evidence_tick = payload["last_evidence_tick"]
        if last_evidence_tick is not None:
            last_evidence_tick = _nonnegative_int(
                last_evidence_tick,
                field="runner_state.last_evidence_tick",
            )
            if last_evidence_tick > completed_world_tick:
                raise OpenEcologyPersistentIslandError(
                    "restored evidence tick exceeds completed world tick"
                )
        self._last_evidence_tick = last_evidence_tick
        self._evidence_finished = _exact_bool(
            payload["evidence_finished"],
            field="runner_state.evidence_finished",
        )
        self._evidence_aborted = _exact_bool(
            payload["evidence_aborted"],
            field="runner_state.evidence_aborted",
        )
        if self._evidence_finished or self._evidence_aborted:
            raise OpenEcologyPersistentIslandError(
                "restartable checkpoint cannot contain a closed evidence stream"
            )
        continuation = _json_clone(evidence_continuation)
        if payload["evidence_continuation_sha256"] != continuation.get("state_sha256"):
            raise OpenEcologyPersistentIslandError(
                "runner state and evidence continuation digest disagree"
            )
        if continuation.get("source_contract_sha256") != _stable_digest(
            self._writer_source_contract
        ):
            raise OpenEcologyPersistentIslandError(
                "restored evidence source contract does not match exact authority"
            )
        continuation_total = _nonnegative_int(
            continuation.get("total_event_count"),
            field="evidence continuation total_event_count",
        )
        continuation_next = _nonnegative_int(
            continuation.get("next_event_index"),
            field="evidence continuation next_event_index",
        )
        if (
            continuation_total
            != self._writer_initial_event_count + sum(self._event_counts.values())
            or continuation_next != self._next_event_index
        ):
            raise OpenEcologyPersistentIslandError(
                "restored runner event accounting does not match evidence prefix"
            )
        self._continuation_state = continuation

    def _checkpoint_config_contract(self) -> VersionedCheckpointState:
        expected = _persistent_checkpoint_config_contract(
            self.task,
            campaign_root=self._campaign_root,
            campaign_id=self._campaign_id,
            evidence_directory=self._evidence_directory,
            checkpoint_directory=self._checkpoint_directory,
            aggregate_directory=self._aggregate_directory,
            checkpoint_interval_ticks=self._checkpoint_interval_ticks,
            writer_config=self._writer_config,
            artifact=self._artifact,
            model_state_sha256=self._model_state_sha256,
        )
        if expected != self._checkpoint_config:
            raise OpenEcologyPersistentIslandError(
                "persistent static checkpoint configuration drifted"
            )
        return self._checkpoint_config

    def _write_genesis_runtime_checkpoint_locked(self) -> None:
        if (
            self._next_tick != 0
            or self._world.tick != 0
            or self._latest_runtime_checkpoint is not None
            or self._latest_aggregate_generation is not None
        ):
            raise OpenEcologyPersistentIslandError(
                "genesis checkpoint requires a fresh tick-zero runner"
            )
        self._write_runtime_checkpoint_locked(allow_genesis=True)

    def _write_runtime_checkpoint_locked(
        self,
        *,
        allow_genesis: bool = False,
    ) -> None:
        if not self._aggregate_resume_authorized:
            raise OpenEcologyPersistentIslandError(
                "loose checkpoint restore cannot publish; resume through "
                "externally pinned aggregate CURRENT"
            )
        if self._next_tick <= 0 and not allow_genesis:
            raise OpenEcologyPersistentIslandError(
                "runtime checkpoint requires at least one completed tick"
            )
        if allow_genesis and (
            self._next_tick != 0
            or self._world.tick != 0
            or self._latest_runtime_checkpoint is not None
            or self._latest_aggregate_generation is not None
        ):
            raise OpenEcologyPersistentIslandError(
                "genesis checkpoint cannot overwrite an existing authority"
            )
        if self._writer is not None or self._continuation_state is None:
            raise OpenEcologyPersistentIslandError(
                "runtime checkpoint requires a sealed evidence boundary"
            )
        if self._evidence_finished or self._evidence_aborted:
            raise OpenEcologyPersistentIslandError(
                "runtime checkpoint cannot capture a closed evidence stream"
            )
        pipeline_started_ns = time.perf_counter_ns()
        self._verify_invariants()
        _verify_live_source_authority(self.task, artifact=self._artifact)
        config_contract = self._checkpoint_config_contract()
        seed_contract = _persistent_checkpoint_seed_contract(self.task)
        if seed_contract != self._checkpoint_seeds:
            raise OpenEcologyPersistentIslandError(
                "persistent static checkpoint seed contract drifted"
            )
        binding = RuntimeCheckpointBinding(
            source_git_sha=self.task.artifact.source_commit,
            source_manifest_sha256=_artifact_source_manifest_sha256(self._artifact),
            config_contract_sha256=_versioned_state_sha256(config_contract),
            seed_contract_sha256=_versioned_state_sha256(seed_contract),
            run_generation_id=_persistent_checkpoint_generation_id(
                campaign_id=self._campaign_id,
                task_id=self.task.task_id,
            ),
            island_id=self.task.task_id,
            generation_index=0,
            completed_tick=self._world.tick,
        )
        setattr(
            self._world,
            _OPEN_ECOLOGY_PERSISTENT_RUNNER_STATE_ATTRIBUTE,
            self._runner_state_payload(),
        )
        capture_started_ns = time.perf_counter_ns()
        components = capture_open_ecology_runtime_checkpoint(
            self._world,
            self._policy,
            binding=binding,
            evidence_writer_continuation_state=self._continuation_state,
        )
        capture_elapsed_ns = time.perf_counter_ns() - capture_started_ns
        try:
            ensure_real_directory_tree(
                self._checkpoint_directory,
                field="persistent-island checkpoint directory",
            )
        except CampaignStorageError as error:
            raise OpenEcologyPersistentIslandError(str(error)) from error
        destination = self._checkpoint_directory / (
            f"checkpoint-observed-{self._next_tick:08d}.json"
        )
        write_started_ns = time.perf_counter_ns()
        checkpoint = write_open_ecology_checkpoint(
            destination,
            source_git_sha=binding.source_git_sha,
            config_contract=config_contract,
            seed_contract=seed_contract,
            run_generation_id=binding.run_generation_id,
            island_id=binding.island_id,
            generation_index=binding.generation_index,
            tick=binding.completed_tick,
            world_state=components.world_state,
            environment_rng_state=components.environment_rng_state,
            recurrent_policy_state=components.recurrent_policy_state,
            public_feedback_history=components.public_feedback_history,
            sampling_rng_state=components.sampling_rng_state,
            genome_population_snapshot=components.genome_population_snapshot,
            evidence_writer_continuation_state=(
                components.evidence_writer_continuation_state
            ),
            max_checkpoint_bytes=OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
        )
        write_elapsed_ns = time.perf_counter_ns() - write_started_ns
        readback_started_ns = time.perf_counter_ns()
        validated = load_open_ecology_checkpoint(
            destination,
            max_checkpoint_bytes=OPEN_ECOLOGY_DEFAULT_MAX_CHECKPOINT_BYTES,
            expected_source_git_sha=binding.source_git_sha,
            require_restartable=True,
        )
        readback_elapsed_ns = time.perf_counter_ns() - readback_started_ns
        if validated != checkpoint:
            raise OpenEcologyPersistentIslandError(
                "written runtime checkpoint did not read back exactly"
            )
        generation_identity = _mapping(
            checkpoint["generation_identity"],
            field="written checkpoint generation identity",
        )
        checkpoint_sha256 = _sha256(
            checkpoint["checkpoint_sha256"],
            field="written checkpoint SHA256",
        )
        checkpoint_identity_sha256 = _sha256(
            generation_identity["identity_sha256"],
            field="written checkpoint generation identity SHA256",
        )
        aggregate_identity = OpenEcologyAggregateIdentityPins(
            source_git_sha=binding.source_git_sha,
            config_contract_sha256=binding.config_contract_sha256,
            seed_contract_sha256=binding.seed_contract_sha256,
            run_generation_id=binding.run_generation_id,
            island_id=binding.island_id,
            simulation_generation_index=binding.generation_index,
            tick=binding.completed_tick,
        )
        previous_commit_sha256 = (
            None
            if self._latest_aggregate_generation is None
            else _sha256(
                self._latest_aggregate_generation.get("commit_sha256"),
                field="previous aggregate commit SHA256",
            )
        )
        aggregate_index = (
            0
            if self._latest_aggregate_generation is None
            else _nonnegative_int(
                self._latest_aggregate_generation.get("aggregate_generation_index"),
                field="previous aggregate generation index",
            )
            + 1
        )
        aggregate_started_ns = time.perf_counter_ns()
        loaded_aggregate = publish_open_ecology_aggregate_generation(
            self._aggregate_directory,
            checkpoint_path=destination,
            evidence_directory=self._evidence_directory,
            identity=aggregate_identity,
            aggregate_generation_index=aggregate_index,
            expected_previous_commit_sha256=previous_commit_sha256,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_checkpoint_generation_identity_sha256=(checkpoint_identity_sha256),
            expected_evidence_manifest_sha256=_sha256(
                self._continuation_state.get("manifest_sha256"),
                field="evidence manifest SHA256",
            ),
            expected_evidence_manifest_status="open",
        )
        aggregate_elapsed_ns = time.perf_counter_ns() - aggregate_started_ns
        aggregate_metadata = _aggregate_metadata_from_loaded_generation(
            loaded_aggregate
        )
        if (
            aggregate_metadata["checkpoint_sha256"] != checkpoint_sha256
            or aggregate_metadata["tick"] != binding.completed_tick
        ):
            raise OpenEcologyPersistentIslandError(
                "aggregate generation readback does not match runtime checkpoint"
            )
        checkpoint_components = _mapping(
            checkpoint["components"],
            field="written checkpoint components",
        )
        component_canonical_bytes = {
            component_name: _canonical_json_byte_size(
                checkpoint_components[component_name]
            )
            for component_name in REQUIRED_CHECKPOINT_COMPONENTS
        }
        self._latest_runtime_checkpoint = {
            "path": str(destination),
            "observed_tick": self._next_tick,
            "completed_world_tick": self._world.tick,
            "checkpoint_sha256": checkpoint_sha256,
            "generation_identity_sha256": checkpoint_identity_sha256,
            "evidence_continuation_sha256": _sha256(
                self._continuation_state.get("state_sha256"),
                field="written checkpoint evidence continuation SHA256",
            ),
            "restartable": True,
            "byte_size": destination.stat().st_size,
            "profile": {
                "capture_elapsed_ns": capture_elapsed_ns,
                "write_elapsed_ns": write_elapsed_ns,
                "readback_elapsed_ns": readback_elapsed_ns,
                "aggregate_publish_elapsed_ns": aggregate_elapsed_ns,
                "checkpoint_pipeline_elapsed_ns": (
                    time.perf_counter_ns() - pipeline_started_ns
                ),
                "component_canonical_envelope_bytes": component_canonical_bytes,
                "component_canonical_envelope_total_bytes": sum(
                    component_canonical_bytes.values()
                ),
                "publication_concurrency": (
                    checkpoint_publication_concurrency_contract()
                ),
            },
        }
        self._latest_aggregate_generation = aggregate_metadata

    def _storage_lock(self) -> CampaignStorageLock:
        return CampaignStorageLock(
            self._storage_lock_path,
            campaign_id=self._campaign_id,
            source_git_sha=self.task.artifact.source_commit,
        )

    def _task_mutation_lock(
        self,
        *,
        initialize: bool = False,
        nonblocking: bool = True,
    ) -> _IdentityFileLock:
        return _IdentityFileLock(
            self._task_lock_path,
            identity=self._task_lock_identity,
            shared=False,
            nonblocking=nonblocking,
            initialize=initialize,
        )

    def _campaign_shared_advance_lock(self) -> _IdentityFileLock:
        return _IdentityFileLock(
            self._storage_lock_path,
            identity=self._storage_lock().identity,
            shared=True,
            nonblocking=True,
        )

    def _campaign_shared_publication_lock(self) -> _IdentityFileLock:
        """Join the campaign epoch while this task publishes its durable frontier."""

        return _IdentityFileLock(
            self._storage_lock_path,
            identity=self._storage_lock().identity,
            shared=True,
            nonblocking=True,
        )

    def _resume_writer(self) -> None:
        if not self._aggregate_resume_authorized:
            raise OpenEcologyPersistentIslandError(
                "loose checkpoint restore is inspection-only; resume through "
                "externally pinned aggregate CURRENT"
            )
        if self._evidence_finished or self._evidence_aborted:
            raise OpenEcologyPersistentIslandError(
                "evidence stream is no longer resumable"
            )
        if self._writer is not None:
            raise OpenEcologyPersistentIslandError("evidence writer is already active")
        if self._continuation_state is None:
            raise OpenEcologyPersistentIslandError(
                "evidence continuation state is unavailable"
            )
        writer = RotatingOpenEcologyEvidenceWriter(
            self._evidence_directory,
            run_id=self.task.task_id,
            source_contract=self._writer_source_contract,
            config=self._writer_config,
            continuation_state=self._continuation_state,
        )
        if _writer_next_event_index(writer) != self._next_event_index:
            writer.abort()
            raise OpenEcologyPersistentIslandError(
                "resumed evidence prefix does not match runner event cursor"
            )
        expected_total = self._writer_initial_event_count + sum(
            self._event_counts.values()
        )
        if writer.diagnostics.get("total_event_count") != expected_total:
            writer.abort()
            raise OpenEcologyPersistentIslandError(
                "resumed evidence prefix does not match runner event count"
            )
        self._writer = writer

    def _checkpoint_active_writer(self) -> None:
        if self._writer is None:
            raise OpenEcologyPersistentIslandError(
                "cannot checkpoint an inactive evidence writer"
            )
        writer = self._writer
        state = writer.checkpoint()
        self._continuation_state = _json_clone(state)
        self._writer = None

    def _abort_active_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.abort()

    def _event_id(self, event_kind: str) -> str:
        return f"{self.task.task_id}:{event_kind}:{self._next_event_index}"

    def _verify_invariants(self) -> None:
        if id(self._world) != self._world_object_identity:
            raise OpenEcologyPersistentIslandError(
                "persistent runner world instance was replaced"
            )
        current_model_sha256 = recurrent_model_state_sha256(self._policy.model)
        if current_model_sha256 != self._model_state_sha256:
            raise OpenEcologyPersistentIslandError(
                "persistent frozen model state changed"
            )
        if any(
            parameter.requires_grad for parameter in self._policy.model.parameters()
        ):
            raise OpenEcologyPersistentIslandError(
                "persistent frozen policy has trainable parameters"
            )
        if self._world.tick != max(0, self._next_tick - 1):
            raise OpenEcologyPersistentIslandError(
                "persistent world tick and runner cursor disagree"
            )
        if self._episode_reset_count != 0 or self._world_replacement_count != 0:
            raise OpenEcologyPersistentIslandError(
                "persistent world reset/replacement counter changed"
            )
        if self._extinct and self._world.alive_agents():
            raise OpenEcologyPersistentIslandError(
                "extinct persistent world regained living agents"
            )
        if self.task.arm == "R" and (
            self._policy._state_by_agent
            or self._policy._feedback_by_agent
            or self._policy._pending_by_agent
        ):
            raise OpenEcologyPersistentIslandError(
                "R retained recurrent state, previous feedback, or a pending decision"
            )


def _validate_persistent_artifact(
    task: PersistentIslandTask,
    *,
    artifact: Mapping[str, object],
    model: object,
) -> None:
    if (
        artifact.get("schema_version")
        != FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION
    ):
        raise OpenEcologyPersistentIslandError(
            "phase D requires the frozen recurrent policy artifact schema"
        )
    if artifact.get("artifact_sha256") != task.artifact.artifact_sha256:
        raise OpenEcologyPersistentIslandError(
            "frozen artifact internal digest does not match the task binding"
        )
    provenance = _mapping(artifact.get("provenance"), field="artifact provenance")
    if (
        provenance.get("learner_seed") != task.learner_seed
        or provenance.get("source_commit") != task.artifact.source_commit
        or provenance.get("seed_registry_digest") != OPEN_ECOLOGY_CANONICAL_SHA256
    ):
        raise OpenEcologyPersistentIslandError(
            "frozen artifact provenance does not match phase-D authority"
        )
    config = getattr(model, "config", None)
    if (
        config is None
        or config.encoder_size != 256
        or config.hidden_size != 256
        or config.recurrent_layers != 1
        or config.genome_conditioning_mode != GENOME_CONDITIONING_ACTOR_FILM_V1
        or config.public_input_schema_version
        != TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
    ):
        raise OpenEcologyPersistentIslandError(
            "phase D requires the tokenized width-256 actor_film_v1 model"
        )


def _artifact_source_manifest_sha256(artifact: Mapping[str, object]) -> str:
    provenance = _mapping(artifact.get("provenance"), field="artifact provenance")
    return _sha256(
        provenance.get("source_manifest_sha256"),
        field="artifact provenance source_manifest_sha256",
    )


def _verify_live_source_authority(
    task: PersistentIslandTask,
    *,
    artifact: Mapping[str, object],
) -> None:
    """Fail closed unless imports and tracked source match the frozen authority."""

    repository_root = Path(__file__).resolve().parents[3]
    python_root = (repository_root / "python").resolve()
    import_paths = (
        Path(__file__).resolve(),
        Path(capture_open_ecology_runtime_checkpoint.__code__.co_filename).resolve(),
        Path(source_file_hash_manifest.__code__.co_filename).resolve(),
    )
    for import_path in import_paths:
        try:
            import_path.relative_to(python_root)
        except ValueError as error:
            raise OpenEcologyPersistentIslandError(
                "persistent runtime import resolved outside exact repository python root"
            ) from error

    def git(*arguments: str) -> str:
        try:
            return run_pinned_git(
                git_authority,
                repository_root=repository_root,
                arguments=arguments,
            )
        except OpenEcologyGitAuthorityError as error:
            raise OpenEcologyPersistentIslandError(
                "cannot verify persistent runtime Git authority"
            ) from error

    try:
        git_authority = discover_pinned_git_executable()
    except OpenEcologyGitAuthorityError as error:
        raise OpenEcologyPersistentIslandError(
            "cannot pin persistent runtime Git authority"
        ) from error

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repository_root:
        raise OpenEcologyPersistentIslandError(
            "persistent runtime repository root does not match imported source"
        )
    if git("rev-parse", "HEAD") != task.artifact.source_commit:
        raise OpenEcologyPersistentIslandError(
            "persistent runtime HEAD does not match frozen artifact source"
        )
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise OpenEcologyPersistentIslandError(
            "persistent runtime source checkout is not clean"
        )
    expected_manifest_sha256 = _artifact_source_manifest_sha256(artifact)
    observed_manifest = source_file_hash_manifest(repository_root)
    if observed_manifest.get("aggregate_sha256") != expected_manifest_sha256:
        raise OpenEcologyPersistentIslandError(
            "persistent runtime source manifest does not match frozen artifact"
        )
    if git("rev-parse", "HEAD") != task.artifact.source_commit or git(
        "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise OpenEcologyPersistentIslandError(
            "persistent runtime source changed during authority verification"
        )


def _persistent_checkpoint_seed_contract(
    task: PersistentIslandTask,
) -> VersionedCheckpointState:
    return VersionedCheckpointState(
        OPEN_ECOLOGY_PERSISTENT_CHECKPOINT_SEED_SCHEMA_VERSION,
        {
            "seed_registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
            "task_id": task.task_id,
            "learner_index": task.learner_index,
            "learner_seed": task.learner_seed,
            "island_index": task.island_index,
            "environment_seed": task.environment_seed,
            "genome_stream_seed_index": task.genome_stream_seed_index,
            "genome_stream_seed": task.genome_stream_seed,
            "policy_sampling_identity": task.policy_sampling_identity,
            "policy_sampling_seed": task.policy_sampling_seed,
        },
    )


def _persistent_checkpoint_config_contract(
    task: PersistentIslandTask,
    *,
    campaign_root: Path,
    campaign_id: str,
    evidence_directory: Path,
    checkpoint_directory: Path,
    aggregate_directory: Path,
    checkpoint_interval_ticks: int,
    writer_config: RotatingEvidenceConfig,
    artifact: Mapping[str, object],
    model_state_sha256: str,
) -> VersionedCheckpointState:
    payload = {
        "campaign_id": campaign_id,
        "evidence_directory_relative": evidence_directory.relative_to(
            campaign_root
        ).as_posix(),
        "checkpoint_directory_relative": checkpoint_directory.relative_to(
            campaign_root
        ).as_posix(),
        "aggregate_directory_relative": aggregate_directory.relative_to(
            campaign_root
        ).as_posix(),
        "checkpoint_interval_ticks": _positive_int(
            checkpoint_interval_ticks,
            field="checkpoint_interval_ticks",
        ),
        "task": task.to_dict(),
        "task_sha256": _stable_digest(task.to_dict()),
        "world_config": build_persistent_world_config(task).to_dict(),
        "world_config_sha256": task.world_config_sha256,
        "writer_config": writer_config.to_dict(),
        "artifact_sha256": task.artifact.artifact_sha256,
        "artifact_file_sha256": task.artifact.artifact_file_sha256,
        "model_state_sha256": _sha256(
            model_state_sha256,
            field="model_state_sha256",
        ),
        "source_manifest_sha256": _artifact_source_manifest_sha256(artifact),
    }
    return VersionedCheckpointState(
        OPEN_ECOLOGY_PERSISTENT_CHECKPOINT_CONFIG_SCHEMA_VERSION,
        payload,
    )


def _persistent_checkpoint_generation_id(*, campaign_id: str, task_id: str) -> str:
    campaign_digest = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()[:16]
    return f"phase-d:{task_id}:{campaign_digest}"


def _aggregate_metadata_from_loaded_generation(
    loaded: Mapping[str, object],
) -> dict[str, object]:
    commit = _mapping(loaded.get("commit"), field="aggregate commit")
    identity = _mapping(commit.get("identity"), field="aggregate identity")
    checkpoint = _mapping(commit.get("checkpoint"), field="aggregate checkpoint")
    evidence = _mapping(commit.get("evidence"), field="aggregate evidence")
    metadata = {
        "aggregate_generation_index": _nonnegative_int(
            commit.get("aggregate_generation_index"),
            field="aggregate generation index",
        ),
        "directory_name": commit.get("directory_name"),
        "commit_sha256": _sha256(
            commit.get("commit_sha256"),
            field="aggregate commit SHA256",
        ),
        "source_git_sha": identity.get("source_git_sha"),
        "config_contract_sha256": _sha256(
            identity.get("config_contract_sha256"),
            field="aggregate config contract SHA256",
        ),
        "seed_contract_sha256": _sha256(
            identity.get("seed_contract_sha256"),
            field="aggregate seed contract SHA256",
        ),
        "run_generation_id": identity.get("run_generation_id"),
        "island_id": identity.get("island_id"),
        "simulation_generation_index": _nonnegative_int(
            identity.get("simulation_generation_index"),
            field="aggregate simulation generation index",
        ),
        "tick": _nonnegative_int(identity.get("tick"), field="aggregate tick"),
        "checkpoint_sha256": _sha256(
            checkpoint.get("checkpoint_sha256"),
            field="aggregate checkpoint SHA256",
        ),
        "checkpoint_generation_identity_sha256": _sha256(
            checkpoint.get("generation_identity_sha256"),
            field="aggregate checkpoint generation identity SHA256",
        ),
        "evidence_manifest_sha256": _sha256(
            evidence.get("manifest_sha256"),
            field="aggregate evidence manifest SHA256",
        ),
        "evidence_manifest_status": evidence.get("status"),
        "evidence_continuation_sha256": _sha256(
            evidence.get("continuation_state_sha256"),
            field="aggregate evidence continuation SHA256",
        ),
    }
    if (
        not isinstance(metadata["directory_name"], str)
        or not metadata["directory_name"]
        or not isinstance(metadata["source_git_sha"], str)
        or len(metadata["source_git_sha"]) != 40
        or _COMMIT_RE.fullmatch(metadata["source_git_sha"]) is None
        or not isinstance(metadata["run_generation_id"], str)
        or not metadata["run_generation_id"]
        or not isinstance(metadata["island_id"], str)
        or not metadata["island_id"]
        or metadata["evidence_manifest_status"] != "open"
    ):
        raise OpenEcologyPersistentIslandError(
            "aggregate generation metadata is incomplete"
        )
    return metadata


def _aggregate_resume_pins_from_metadata(
    metadata: Mapping[str, object],
) -> OpenEcologyAggregateResumePins:
    source_git_sha = metadata.get("source_git_sha")
    run_generation_id = metadata.get("run_generation_id")
    island_id = metadata.get("island_id")
    if (
        not isinstance(source_git_sha, str)
        or not isinstance(run_generation_id, str)
        or not isinstance(island_id, str)
    ):
        raise OpenEcologyPersistentIslandError(
            "aggregate resume identity metadata is invalid"
        )
    return OpenEcologyAggregateResumePins(
        identity=OpenEcologyAggregateIdentityPins(
            source_git_sha=source_git_sha,
            config_contract_sha256=_sha256(
                metadata.get("config_contract_sha256"),
                field="aggregate config contract SHA256",
            ),
            seed_contract_sha256=_sha256(
                metadata.get("seed_contract_sha256"),
                field="aggregate seed contract SHA256",
            ),
            run_generation_id=run_generation_id,
            island_id=island_id,
            simulation_generation_index=_nonnegative_int(
                metadata.get("simulation_generation_index"),
                field="aggregate simulation generation index",
            ),
            tick=_nonnegative_int(metadata.get("tick"), field="aggregate tick"),
        ),
        aggregate_generation_index=_nonnegative_int(
            metadata.get("aggregate_generation_index"),
            field="aggregate generation index",
        ),
        commit_sha256=_sha256(
            metadata.get("commit_sha256"),
            field="aggregate commit SHA256",
        ),
        checkpoint_sha256=_sha256(
            metadata.get("checkpoint_sha256"),
            field="aggregate checkpoint SHA256",
        ),
        checkpoint_generation_identity_sha256=_sha256(
            metadata.get("checkpoint_generation_identity_sha256"),
            field="aggregate checkpoint generation identity SHA256",
        ),
        evidence_manifest_sha256=_sha256(
            metadata.get("evidence_manifest_sha256"),
            field="aggregate evidence manifest SHA256",
        ),
        evidence_manifest_status=str(metadata.get("evidence_manifest_status")),
    )


def _versioned_state_sha256(state: VersionedCheckpointState) -> str:
    return _stable_digest(
        {
            "schema_version": state.schema_version,
            "payload": dict(state.payload),
        }
    )


def _checkpoint_state_from_envelope(
    envelope: Mapping[str, object],
    *,
    expected_schema_version: str,
    field: str,
) -> VersionedCheckpointState:
    _exact_keys(
        envelope,
        {
            "envelope_schema_version",
            "present",
            "schema_version",
            "payload",
            "state_sha256",
        },
        field=field,
    )
    if envelope["present"] is not True:
        raise OpenEcologyPersistentIslandError(f"{field} is not present")
    if envelope["schema_version"] != expected_schema_version:
        raise OpenEcologyPersistentIslandError(f"{field} schema version drifted")
    payload = _mapping(envelope["payload"], field=f"{field}.payload")
    state = VersionedCheckpointState(expected_schema_version, payload)
    if envelope["state_sha256"] != _versioned_state_sha256(state):
        raise OpenEcologyPersistentIslandError(f"{field} state digest mismatch")
    return state


def _runtime_components_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> RuntimeCheckpointComponents:
    envelopes = _mapping(checkpoint.get("components"), field="checkpoint.components")
    _exact_keys(
        envelopes,
        set(REQUIRED_CHECKPOINT_COMPONENTS),
        field="checkpoint.components",
    )
    states: dict[str, VersionedCheckpointState] = {}
    for component_name in REQUIRED_CHECKPOINT_COMPONENTS:
        envelope = _mapping(
            envelopes[component_name],
            field=f"checkpoint.components.{component_name}",
        )
        if envelope.get("present") is not True:
            raise OpenEcologyPersistentIslandError(
                f"checkpoint component {component_name} is absent"
            )
        schema_version = envelope.get("schema_version")
        if not isinstance(schema_version, str):
            raise OpenEcologyPersistentIslandError(
                f"checkpoint component {component_name} schema is invalid"
            )
        states[component_name] = VersionedCheckpointState(
            schema_version=schema_version,
            payload=_mapping(
                envelope.get("payload"),
                field=f"checkpoint.components.{component_name}.payload",
            ),
        )
    return RuntimeCheckpointComponents(**states)


def _restore_campaign_child(
    campaign_root: Path,
    value: object,
    *,
    field: str,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
    ):
        raise OpenEcologyPersistentIslandError(
            f"{field} must be a normalized relative path"
        )
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise OpenEcologyPersistentIslandError(
            f"{field} must remain inside campaign_root"
        )
    if relative.as_posix() != value or "." in relative.parts:
        raise OpenEcologyPersistentIslandError(
            f"{field} must use canonical POSIX path spelling"
        )
    return campaign_root / relative


def _validate_embedded_digest(
    payload: Mapping[str, object],
    *,
    digest_field: str,
    field: str,
) -> None:
    observed = _sha256(payload.get(digest_field), field=f"{field}.{digest_field}")
    expected = _stable_digest(
        {key: value for key, value in payload.items() if key != digest_field}
    )
    if observed != expected:
        raise OpenEcologyPersistentIslandError(f"{field} digest mismatch")


def _event_counter(value: object, *, field: str) -> Counter[str]:
    payload = _mapping(value, field=field)
    result: Counter[str] = Counter()
    for event_type, count in payload.items():
        if (
            not event_type
            or event_type != event_type.strip()
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", event_type) is None
        ):
            raise OpenEcologyPersistentIslandError(
                f"{field} contains an invalid event type"
            )
        result[event_type] = _nonnegative_int(
            count,
            field=f"{field}.{event_type}",
        )
    return result


def _action_counter(value: object, *, field: str) -> Counter[str]:
    payload = _mapping(value, field=field)
    result: Counter[str] = Counter()
    for action, count in payload.items():
        if action not in ACTION_NAMES:
            raise OpenEcologyPersistentIslandError(
                f"{field} contains an unknown action"
            )
        result[action] = _nonnegative_int(count, field=f"{field}.{action}")
    return result


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OpenEcologyPersistentIslandError(f"{field} must be a list")
    return value


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise OpenEcologyPersistentIslandError(f"{field} must be an exact boolean")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyPersistentIslandError(f"{field} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise OpenEcologyPersistentIslandError(f"{field} must be positive")
    return parsed


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(payload) != expected:
        raise OpenEcologyPersistentIslandError(f"{field} field set drifted")


def _arm_contract(arm: str) -> dict[str, object]:
    contracts: dict[str, dict[str, object]] = {
        "H": {
            "genome_population_mode": (RecurrentGenomePopulationMode.HERITABLE.value),
            "reset_recurrent_state_each_decision": False,
            "history_enabled": True,
            "description": "heritable_history",
        },
        "Z": {
            "genome_population_mode": RecurrentGenomePopulationMode.ZERO_ALL.value,
            "reset_recurrent_state_each_decision": False,
            "history_enabled": True,
            "description": "zero_genome_history",
        },
        "R": {
            "genome_population_mode": (RecurrentGenomePopulationMode.HERITABLE.value),
            "reset_recurrent_state_each_decision": True,
            "history_enabled": False,
            "description": "heritable_reset_each_decision",
        },
    }
    try:
        return dict(contracts[arm])
    except KeyError as error:
        raise OpenEcologyPersistentIslandError(
            "arm must be exactly H, Z, or R"
        ) from error


def _task_id(*, learner_index: int, island_index: int, arm: str) -> str:
    return f"phase-d-l{learner_index:02d}-i{island_index:02d}-{arm.lower()}"


def _policy_sampling_identity(
    *,
    learner_index: int,
    learner_seed: int,
    island_index: int,
    environment_seed: int,
    arm: str,
) -> str:
    return (
        f"{OPEN_ECOLOGY_PHASE_D_POLICY_SAMPLING_IDENTITY_VERSION}"
        f"|phase:phase_d"
        f"|learner_index:{learner_index}"
        f"|learner_seed:{learner_seed}"
        f"|environment_index:{island_index}"
        f"|environment_seed:{environment_seed}"
        f"|arm:{arm}"
        "|update:none|tape:primary"
    )


def _generation_depths(world: SimulationWorld) -> dict[int, int]:
    depths: dict[int, int] = {}

    def depth(agent_id: int) -> int:
        cached = depths.get(agent_id)
        if cached is not None:
            return cached
        agent = world.agents[agent_id]
        value = 0 if agent.parent_id is None else depth(agent.parent_id) + 1
        depths[agent_id] = value
        return value

    for agent_id in sorted(world.agents):
        depth(agent_id)
    return depths


def _series_stats(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "minimum": round(min(materialized), 8),
        "mean": round(sum(materialized) / len(materialized), 8),
        "maximum": round(max(materialized), 8),
    }


def _hill_effective_q1(counts: Mapping[object, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count > 0
    )
    return round(math.exp(entropy), 8)


def _optional_genome_sha256(metadata: object) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("genome_sha256")
    if value is None:
        return None
    return _sha256(value, field="agent controller genome SHA256")


def _lineage_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenEcologyPersistentIslandError("lineage id must be positive")
    return f"lineage-{value}"


def _writer_next_event_index(writer: _EvidenceWriter) -> int:
    value = writer.diagnostics.get("next_event_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologyPersistentIslandError(
            "evidence writer next_event_index is invalid"
        )
    return value


def _loaded_aggregate_index(loaded: object) -> int | None:
    if loaded is None:
        return None
    commit = _mapping(
        _mapping(loaded, field="loaded aggregate").get("commit"),
        field="loaded aggregate commit",
    )
    return _nonnegative_int(
        commit.get("aggregate_generation_index"),
        field="loaded aggregate generation index",
    )


def _loaded_aggregate_commit_sha256(loaded: object) -> str:
    if loaded is None:
        raise OpenEcologyPersistentIslandError("aggregate predecessor is missing")
    commit = _mapping(
        _mapping(loaded, field="loaded aggregate").get("commit"),
        field="loaded aggregate commit",
    )
    return _sha256(
        commit.get("commit_sha256"),
        field="loaded aggregate commit SHA256",
    )


def _validate_exact_genesis_runner(
    runner: PersistentIslandRunner,
    *,
    pins: OpenEcologyAggregateResumePins,
    task: PersistentIslandTask,
    campaign_id: str,
) -> None:
    identity = pins.identity
    if (
        runner.task != task
        or runner.observed_tick != 0
        or runner.extinct
        or runner.latest_aggregate_resume_pins != pins
        or pins.aggregate_generation_index != 0
        or identity.source_git_sha != task.artifact.source_commit
        or identity.run_generation_id
        != _persistent_checkpoint_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        )
        or identity.island_id != task.task_id
        or identity.simulation_generation_index != 0
        or identity.tick != 0
    ):
        raise OpenEcologyPersistentIslandError(
            "restored aggregate is not the exact task genesis"
        )


def _validate_reconciled_target_runner(
    runner: PersistentIslandRunner,
    *,
    pins: OpenEcologyAggregateResumePins,
    task: PersistentIslandTask,
    campaign_id: str,
    predecessor: OpenEcologyAggregateResumePins | None,
    predecessor_observed_tick: int,
    predecessor_terminal: bool,
    requested_target_tick: int,
    expected_aggregate_generation_index: int,
    base_loaded: object,
    target_loaded: Mapping[str, object],
) -> None:
    if predecessor_terminal:
        raise OpenEcologyPersistentIslandError(
            "terminal predecessor cannot have a later attempt generation"
        )
    observed_tick = runner.observed_tick
    identity = pins.identity
    if (
        runner.task != task
        or runner.latest_aggregate_resume_pins != pins
        or pins.aggregate_generation_index != expected_aggregate_generation_index
        or identity.source_git_sha != task.artifact.source_commit
        or identity.run_generation_id
        != _persistent_checkpoint_generation_id(
            campaign_id=campaign_id,
            task_id=task.task_id,
        )
        or identity.island_id != task.task_id
        or identity.simulation_generation_index != 0
        or observed_tick <= predecessor_observed_tick
        or observed_tick > requested_target_tick
        or identity.tick != observed_tick - 1
        or (observed_tick < requested_target_tick and not runner.extinct)
    ):
        raise OpenEcologyPersistentIslandError(
            "reconciled attempt target does not satisfy the original intent"
        )
    if base_loaded is None:
        raise OpenEcologyPersistentIslandError(
            "reconciled attempt target has no exact predecessor generation"
        )
    base_metadata = _aggregate_metadata_from_loaded_generation(
        _mapping(base_loaded, field="attempt target predecessor")
    )
    base_pins = _aggregate_resume_pins_from_metadata(base_metadata)
    if predecessor is None:
        base_identity = base_pins.identity
        if (
            base_pins.aggregate_generation_index != 0
            or base_identity.source_git_sha != identity.source_git_sha
            or base_identity.config_contract_sha256 != identity.config_contract_sha256
            or base_identity.seed_contract_sha256 != identity.seed_contract_sha256
            or base_identity.run_generation_id != identity.run_generation_id
            or base_identity.island_id != identity.island_id
            or base_identity.simulation_generation_index != 0
            or base_identity.tick != 0
        ):
            raise OpenEcologyPersistentIslandError(
                "initial attempt target does not extend exact task genesis"
            )
    elif (
        base_pins != predecessor
        or identity.config_contract_sha256
        != predecessor.identity.config_contract_sha256
        or identity.seed_contract_sha256 != predecessor.identity.seed_contract_sha256
    ):
        raise OpenEcologyPersistentIslandError(
            "attempt target does not extend the externally pinned predecessor"
        )
    target_commit = _mapping(
        target_loaded.get("commit"),
        field="reconciled target commit",
    )
    target_previous = _mapping(
        target_commit.get("previous"),
        field="reconciled target predecessor",
    )
    if (
        target_previous.get("aggregate_generation_index")
        != base_pins.aggregate_generation_index
        or target_previous.get("commit_sha256") != base_pins.commit_sha256
    ):
        raise OpenEcologyPersistentIslandError(
            "reconciled target commit predecessor chain drifted"
        )


def _interval_attempt_has_loose_or_staged_bytes(
    *,
    evidence_directory: Path,
    checkpoint_directory: Path,
    aggregate_directory: Path,
    target_observed_tick: int,
    target_aggregate_generation_index: int,
    predecessor_manifest_sha256: str | None,
) -> bool:
    loose_checkpoint = checkpoint_directory / (
        f"checkpoint-observed-{target_observed_tick:08d}.json"
    )
    if loose_checkpoint.exists() or loose_checkpoint.is_symlink():
        return True
    stage_prefix = (
        f".{aggregate_directory.name}."
        f"generation-{target_aggregate_generation_index:016d}.stage-"
    )
    stage_parent = aggregate_directory.parent
    if stage_parent.exists():
        try:
            if any(
                child.name.startswith(stage_prefix) for child in stage_parent.iterdir()
            ):
                return True
        except OSError as error:
            raise OpenEcologyPersistentIslandError(
                f"cannot inspect aggregate attempt stages: {error}"
            ) from error
    if not evidence_directory.exists() and not evidence_directory.is_symlink():
        return False
    if predecessor_manifest_sha256 is None:
        return True
    try:
        manifest = load_open_ecology_evidence_manifest(
            evidence_directory,
            verify_shards=True,
        )
    except (OSError, ValueError):
        return True
    return manifest.get("manifest_sha256") != predecessor_manifest_sha256


def _preserve_loose_interval_checkpoint(
    checkpoint_path: Path,
    *,
    preservation_directory: Path,
) -> Path:
    if (
        not checkpoint_path.is_absolute()
        or not preservation_directory.is_absolute()
        or checkpoint_path.parent in preservation_directory.parents
    ):
        raise OpenEcologyPersistentIslandError(
            "loose interval checkpoint preservation paths are invalid"
        )
    try:
        metadata = checkpoint_path.lstat()
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot inspect loose interval checkpoint: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OpenEcologyPersistentIslandError(
            "loose interval checkpoint must be one regular file"
        )
    try:
        preservation_directory = ensure_real_directory_tree(
            preservation_directory,
            field="interval checkpoint preservation directory",
        )
    except CampaignStorageError as error:
        raise OpenEcologyPersistentIslandError(str(error)) from error
    base_name = f"{checkpoint_path.name}-{_file_sha256(checkpoint_path)}"
    destination = _unused_attempt_path(
        preservation_directory,
        base_name=base_name,
    )
    try:
        os.rename(checkpoint_path, destination)
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot preserve loose interval checkpoint: {error}"
        ) from error
    _fsync_directory_best_effort(checkpoint_path.parent)
    _fsync_directory_best_effort(preservation_directory)
    return destination


def _aggregate_genesis_evidence_is_required(aggregate_directory: Path) -> bool:
    """Return whether an aggregate contains authority that needs live evidence."""

    if not aggregate_directory.exists() and not aggregate_directory.is_symlink():
        return False
    try:
        metadata = aggregate_directory.lstat()
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot inspect genesis aggregate directory: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise OpenEcologyPersistentIslandError(
            "genesis aggregate path must be a real directory"
        )
    current = aggregate_directory / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
    if current.exists() or current.is_symlink():
        return True
    generations = aggregate_directory / OPEN_ECOLOGY_AGGREGATE_GENERATIONS_DIRECTORY
    if not generations.exists() and not generations.is_symlink():
        return False
    try:
        generations_metadata = generations.lstat()
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot inspect genesis aggregate generations: {error}"
        ) from error
    if not stat.S_ISDIR(generations_metadata.st_mode):
        raise OpenEcologyPersistentIslandError(
            "genesis aggregate generations must be a real directory"
        )
    try:
        return any(generations.iterdir())
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot enumerate genesis aggregate generations: {error}"
        ) from error


def _preserve_genesis_checkpoint_attempt_directory(
    checkpoint_directory: Path,
    *,
    preservation_directory: Path,
) -> Path:
    """Move all loose files from one failed genesis attempt without replacement."""

    if (
        not checkpoint_directory.is_absolute()
        or not preservation_directory.is_absolute()
        or preservation_directory == checkpoint_directory
        or checkpoint_directory in preservation_directory.parents
    ):
        raise OpenEcologyPersistentIslandError(
            "genesis checkpoint attempt paths are invalid"
        )
    try:
        metadata = checkpoint_directory.lstat()
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot inspect genesis checkpoint attempt: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise OpenEcologyPersistentIslandError(
            "genesis checkpoint attempt must be a real directory"
        )
    entries: list[dict[str, object]] = []
    try:
        children = sorted(checkpoint_directory.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot enumerate genesis checkpoint attempt: {error}"
        ) from error
    for child in children:
        try:
            child_metadata = child.lstat()
        except OSError as error:
            raise OpenEcologyPersistentIslandError(
                f"cannot inspect genesis checkpoint attempt entry: {error}"
            ) from error
        if not stat.S_ISREG(child_metadata.st_mode):
            raise OpenEcologyPersistentIslandError(
                "genesis checkpoint attempt entries must be regular files"
            )
        entries.append(
            {
                "name": child.name,
                "size": child_metadata.st_size,
            }
        )
    try:
        preservation_directory = ensure_real_directory_tree(
            preservation_directory,
            field="genesis checkpoint preservation directory",
        )
    except CampaignStorageError as error:
        raise OpenEcologyPersistentIslandError(str(error)) from error
    base_name = f"genesis-checkpoints-{_stable_digest({'entries': entries})}"
    destination = _unused_attempt_path(
        preservation_directory,
        base_name=base_name,
    )
    try:
        os.rename(checkpoint_directory, destination)
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot preserve genesis checkpoint attempt: {error}"
        ) from error
    _fsync_directory_best_effort(checkpoint_directory.parent)
    _fsync_directory_best_effort(preservation_directory)
    return destination


def _preserve_genesis_aggregate_stages(
    aggregate_directory: Path,
    *,
    preservation_directory: Path,
) -> tuple[Path, ...]:
    return _preserve_aggregate_attempt_stages(
        aggregate_directory,
        aggregate_generation_index=0,
        preservation_directory=preservation_directory,
    )


def _preserve_aggregate_attempt_stages(
    aggregate_directory: Path,
    *,
    aggregate_generation_index: int,
    preservation_directory: Path,
) -> tuple[Path, ...]:
    """Move process-death aggregate stages into the attempt evidence namespace."""

    if (
        not aggregate_directory.is_absolute()
        or not preservation_directory.is_absolute()
        or preservation_directory == aggregate_directory.parent
        or aggregate_directory.parent in preservation_directory.parents
    ):
        raise OpenEcologyPersistentIslandError(
            "genesis aggregate stage preservation paths are invalid"
        )
    stage_parent = aggregate_directory.parent
    if not stage_parent.exists():
        return ()
    try:
        parent_metadata = stage_parent.lstat()
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot inspect genesis aggregate stage parent: {error}"
        ) from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise OpenEcologyPersistentIslandError(
            "genesis aggregate stage parent must be a real directory"
        )
    stage_prefix = (
        f".{aggregate_directory.name}."
        f"generation-{_nonnegative_int(aggregate_generation_index, field='stage index'):016d}."
        "stage-"
    )
    try:
        stages = sorted(
            (
                child
                for child in stage_parent.iterdir()
                if child.name.startswith(stage_prefix)
            ),
            key=lambda path: path.name,
        )
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot enumerate genesis aggregate stages: {error}"
        ) from error
    if not stages:
        return ()
    try:
        preservation_directory = ensure_real_directory_tree(
            preservation_directory,
            field="genesis aggregate stage preservation directory",
        )
    except CampaignStorageError as error:
        raise OpenEcologyPersistentIslandError(str(error)) from error
    preserved: list[Path] = []
    for stage in stages:
        try:
            stage_metadata = stage.lstat()
        except OSError as error:
            raise OpenEcologyPersistentIslandError(
                f"cannot inspect genesis aggregate stage: {error}"
            ) from error
        if not stat.S_ISDIR(stage_metadata.st_mode):
            raise OpenEcologyPersistentIslandError(
                "genesis aggregate stage must be a real directory"
            )
        destination = _unused_attempt_path(
            preservation_directory,
            base_name=stage.name,
        )
        try:
            os.rename(stage, destination)
        except OSError as error:
            raise OpenEcologyPersistentIslandError(
                f"cannot preserve genesis aggregate stage: {error}"
            ) from error
        preserved.append(destination)
    _fsync_directory_best_effort(stage_parent)
    _fsync_directory_best_effort(preservation_directory)
    return tuple(preserved)


def _unused_attempt_path(directory: Path, *, base_name: str) -> Path:
    destination = directory / base_name
    attempt_index = 0
    while destination.exists() or destination.is_symlink():
        attempt_index += 1
        destination = directory / f"{base_name}-attempt-{attempt_index:04d}"
    return destination


def _fsync_directory_best_effort(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            f"cannot read frozen artifact {path}"
        ) from error
    return digest.hexdigest()


def _validated_campaign_barrier_frontier(
    frontier: Mapping[str, object],
    *,
    campaign_root: Path,
    campaign_id: str,
    source_git_sha: str,
) -> dict[str, object]:
    if not isinstance(frontier, Mapping) or not all(
        isinstance(key, str) for key in frontier
    ):
        raise CampaignStorageError(
            "campaign barrier frontier must be an object with string keys"
        )
    normalized = dict(frontier)
    expected_keys = {
        "aggregate_commit_sha256",
        "aggregate_completed_world_tick",
        "aggregate_current_restartable",
        "aggregate_generation_index",
        "aggregate_resume_authorized",
        "campaign_id",
        "campaign_root",
        "checkpoint_generation_identity_sha256",
        "checkpoint_sha256",
        "config_contract_sha256",
        "evidence_continuation_sha256",
        "evidence_manifest_sha256",
        "evidence_manifest_status",
        "extinct",
        "extinction_tick",
        "frontier_sha256",
        "observed_tick",
        "schema_version",
        "seed_contract_sha256",
        "source_git_sha",
        "task_id",
        "task_sha256",
        "world_config_sha256",
    }
    if set(normalized) != expected_keys:
        raise CampaignStorageError("campaign barrier frontier field set drifted")
    if normalized["schema_version"] != (
        _OPEN_ECOLOGY_CAMPAIGN_BARRIER_FRONTIER_SCHEMA_VERSION
    ):
        raise CampaignStorageError("campaign barrier frontier schema drifted")
    task_id = normalized["task_id"]
    if (
        not isinstance(task_id, str)
        or _OPEN_ECOLOGY_PHASE_D_TASK_ID_RE.fullmatch(task_id) is None
    ):
        raise CampaignStorageError("campaign barrier task id is invalid")
    if (
        normalized["campaign_id"] != campaign_id
        or normalized["campaign_root"] != str(campaign_root)
        or normalized["source_git_sha"] != source_git_sha
    ):
        raise CampaignStorageError("campaign barrier identity does not match")
    observed_tick = normalized["observed_tick"]
    completed_tick = normalized["aggregate_completed_world_tick"]
    aggregate_generation_index = normalized["aggregate_generation_index"]
    genesis_boundary = (
        observed_tick == 0 and completed_tick == 0 and aggregate_generation_index == 0
    )
    completed_boundary = (
        isinstance(observed_tick, int)
        and not isinstance(observed_tick, bool)
        and observed_tick > 0
        and isinstance(completed_tick, int)
        and not isinstance(completed_tick, bool)
        and completed_tick == observed_tick - 1
    )
    if (
        isinstance(observed_tick, bool)
        or not isinstance(observed_tick, int)
        or observed_tick < 0
        or observed_tick > OPEN_ECOLOGY_PHASE_D_TARGET_TICKS
        or isinstance(completed_tick, bool)
        or not isinstance(completed_tick, int)
        or isinstance(aggregate_generation_index, bool)
        or not isinstance(aggregate_generation_index, int)
        or aggregate_generation_index < 0
        or not (genesis_boundary or completed_boundary)
    ):
        raise CampaignStorageError(
            "campaign barrier tick or generation frontier is invalid"
        )
    extinct = normalized["extinct"]
    extinction_tick = normalized["extinction_tick"]
    if (
        not isinstance(extinct, bool)
        or (
            extinct
            and (
                isinstance(extinction_tick, bool)
                or not isinstance(extinction_tick, int)
                or extinction_tick <= 0
                or extinction_tick != observed_tick
            )
        )
        or (not extinct and extinction_tick is not None)
    ):
        raise CampaignStorageError("campaign barrier extinction frontier is invalid")
    for field_name in (
        "aggregate_commit_sha256",
        "checkpoint_generation_identity_sha256",
        "checkpoint_sha256",
        "config_contract_sha256",
        "evidence_continuation_sha256",
        "evidence_manifest_sha256",
        "seed_contract_sha256",
        "task_sha256",
        "world_config_sha256",
    ):
        value = normalized[field_name]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise CampaignStorageError(
                f"campaign barrier {field_name} must be lowercase SHA256"
            )
    if (
        normalized["evidence_manifest_status"] != "open"
        or normalized["aggregate_current_restartable"] is not True
        or normalized["aggregate_resume_authorized"] is not True
    ):
        raise CampaignStorageError(
            "campaign barrier frontier is not an open authorized CURRENT"
        )
    claimed_digest = normalized.pop("frontier_sha256")
    if (
        not isinstance(claimed_digest, str)
        or _SHA256_RE.fullmatch(claimed_digest) is None
        or claimed_digest != _stable_digest(normalized)
    ):
        raise CampaignStorageError("campaign barrier frontier digest drifted")
    normalized["frontier_sha256"] = claimed_digest
    return normalized


def _assert_current_pointer_matches(
    aggregate_directory: Path,
    *,
    aggregate: Mapping[str, object],
) -> None:
    try:
        resolved_root = aggregate_directory.resolve(strict=True)
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            "campaign aggregate root is unavailable"
        ) from error
    if resolved_root != aggregate_directory or not resolved_root.is_dir():
        raise OpenEcologyPersistentIslandError(
            "campaign aggregate root is not canonical"
        )
    current_path = resolved_root / OPEN_ECOLOGY_AGGREGATE_CURRENT_NAME
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(
            os,
            "O_CLOEXEC",
            0,
        )
    )
    try:
        descriptor = os.open(current_path, flags)
    except OSError as error:
        raise OpenEcologyPersistentIslandError(
            "campaign aggregate CURRENT is unavailable"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size <= 0
            or metadata.st_size > _OPEN_ECOLOGY_CURRENT_MAX_BYTES
        ):
            raise OpenEcologyPersistentIslandError(
                "campaign aggregate CURRENT is not one private small regular file"
            )
        observed = bytearray()
        while len(observed) <= _OPEN_ECOLOGY_CURRENT_MAX_BYTES:
            block = os.read(
                descriptor,
                min(
                    8 * 1024,
                    _OPEN_ECOLOGY_CURRENT_MAX_BYTES + 1 - len(observed),
                ),
            )
            if not block:
                break
            observed.extend(block)
    finally:
        os.close(descriptor)
    body: dict[str, object] = {
        "aggregate_generation_index": aggregate["aggregate_generation_index"],
        "commit_sha256": aggregate["commit_sha256"],
        "digest_policy": OPEN_ECOLOGY_AGGREGATE_POINTER_DIGEST_POLICY,
        "directory_name": aggregate["directory_name"],
        "schema_version": OPEN_ECOLOGY_AGGREGATE_POINTER_SCHEMA,
    }
    expected = canonical_json_bytes({**body, "pointer_sha256": _stable_digest(body)})
    if bytes(observed) != expected:
        raise OpenEcologyPersistentIslandError(
            "campaign aggregate CURRENT does not match the live frontier"
        )


def _write_lock_bytes(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise CampaignStorageError("cooperative lock identity write stalled")
        view = view[written:]
    os.ftruncate(descriptor, len(payload))
    os.fsync(descriptor)


def _stable_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json_byte_size(value: object) -> int:
    return len(_canonical_json_bytes(value))


def _json_clone(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise OpenEcologyPersistentIslandError(
            f"{field} must be an object with string keys"
        )
    return value


def _index(value: object, *, field: str, upper: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= upper
    ):
        raise OpenEcologyPersistentIslandError(
            f"{field} must be an integer in [0, {upper})"
        )
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyPersistentIslandError(f"{field} must be lowercase SHA256")
    return value


__all__ = [
    "OPEN_ECOLOGY_CHECKPOINT_PUBLICATION_CONCURRENCY_SCHEMA_VERSION",
    "OPEN_ECOLOGY_FIRST_MILESTONE_TICK",
    "OPEN_ECOLOGY_GENESIS_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED",
    "OPEN_ECOLOGY_PARALLEL_CHECKPOINT_PUBLICATION_AUTHORIZED",
    "OPEN_ECOLOGY_PERSISTENT_LAUNCH_BLOCKERS",
    "OPEN_ECOLOGY_PERSISTENT_MILESTONE_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PERSISTENT_RUNNER_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PERSISTENT_SUMMARY_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PERSISTENT_TASK_SCHEMA_VERSION",
    "OPEN_ECOLOGY_PHASE_D_ARM_ORDER",
    "OPEN_ECOLOGY_PHASE_D_TASK_COUNT",
    "OPEN_ECOLOGY_PHASE_D_TARGET_TICKS",
    "OPEN_ECOLOGY_SUMMARY_INTERVAL_TICKS",
    "OpenEcologyPersistentIslandError",
    "PersistentAdvanceResult",
    "PersistentArtifactBinding",
    "PersistentIslandRunner",
    "PersistentIslandTask",
    "build_persistent_island_task_matrix",
    "build_persistent_world_config",
    "checkpoint_publication_concurrency_contract",
    "persistent_genesis_attempt_authority_sha256",
    "persistent_interval_attempt_authority_payload_sha256",
    "persistent_interval_attempt_authority_sha256",
    "persistent_island_writer_source_contract",
    "prioritized_persistent_island_triplet",
]
