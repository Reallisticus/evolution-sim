from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import multiprocessing as mp
import random

import torch

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_counterfactual_branch import (
    RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO,
    RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_SCALE_SEED_ROLES,
    RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES,
    RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES,
    RECURRENT_COUNTERFACTUAL_SCENARIOS,
    RECURRENT_COUNTERFACTUAL_SEED_ROLES,
    RecurrentCounterfactualBranchError,
    build_recurrent_counterfactual_nested_horizon_materialization,
    build_recurrent_counterfactual_nested_horizon_tape_chunk_materialization,
    merge_recurrent_counterfactual_nested_horizon_tape_chunks,
    validate_recurrent_counterfactual_aggregate_row,
    validate_recurrent_counterfactual_branch_row,
)
from evolution_sim.mind.recurrent_policy import (
    frozen_cpu_model_copy,
    recurrent_model_state_sha256,
)
from evolution_sim.mind.recurrent_seed_registry import (
    RECURRENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
)


RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION = (
    "mind_v3_recurrent_nested_counterfactual_collection_v1"
)
RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION = (
    "mind_v3_recurrent_nested_counterfactual_multi_tape_collection_v3"
)
RECURRENT_COUNTERFACTUAL_LEGACY_EXECUTION_CONTRACT_VERSION = (
    "mind_v3_recurrent_counterfactual_bundle_parallel_execution_v1"
)
RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION = (
    "mind_v3_recurrent_counterfactual_tape_parallel_execution_v2"
)
RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE = (
    "mind_v3_recurrent_counterfactual_source_policy_seed_v1"
)
RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE = (
    "mind_v3_recurrent_counterfactual_branch_selection_seed_v1"
)
RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD = "spawn"
RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS = 1
MAX_RECURRENT_COUNTERFACTUAL_COLLECTION_WORKERS = 64
MIN_RECURRENT_COUNTERFACTUAL_TAPES_PER_CHUNK = 2
MAX_RECURRENT_COUNTERFACTUAL_SEED = 2**63 - 1

_TAPE_PARALLEL_EXECUTION_COMPUTE_FIELDS = {
    "tape_chunk_work_item_count",
    "split_bundle_count",
    "maximum_chunks_per_bundle",
    "source_tick_executions",
    "canonical_source_tick_executions",
    "duplicated_source_tick_executions",
    "legacy_actual_continuation_tick_count",
    "canonical_legacy_actual_continuation_tick_count",
    "duplicated_legacy_actual_continuation_tick_count",
    "multi_tape_actual_continuation_tick_count",
    "total_actual_simulator_tick_executions",
}

_SELECTION_BASE_FIELDS = {
    "requested_branch_ticks",
    "selected_branch_tick",
    "fallback_index",
    "eligible_candidate_count",
    "candidate_list_digest",
    "selected_candidate_index",
    "selected_focal_agent_id",
    "selection_policy",
    "outcome_or_future_data_used",
}
_SELECTION_MULTI_TAPE_FIELDS = {
    "eligible_branch_ticks",
    "eligible_branch_tick_count",
    "selected_eligible_tick_index",
}
_SELECTION_ROTATION_FIELDS = {
    "requested_branch_tick_stratum_index",
    "branch_tick_strata_traversal",
    "selected_branch_tick_stratum_index",
}
_PREFIX_PROOF_FIELDS = {
    "schema_version",
    "horizons",
    "stream_count",
    "streams",
    "all_shorter_streams_are_exact_max_stream_prefixes",
    "raw_private_world_state_serialized",
}
_PREFIX_STREAM_FIELDS = {
    "horizons",
    "exact_prefix",
}
_PREFIX_HORIZON_FIELDS = {
    "behavior_digest",
    "evidence_digest",
    "executed_tick_count",
    "causal_record_digest_sequence",
    "policy_record_digest_sequence",
    "diagnostics_digest_sequence",
}
_COMPUTE_FIELDS = {
    "source_tick_executions",
    "selected_continuation_checkpoint_count",
    "valid_action_count",
    "horizon_count",
    "action_horizon_target_count",
    "max_horizon_continuation_count",
    "exact_replay_continuation_count",
    "maximum_horizon_ticks",
    "actual_continuation_tick_count",
    "maximum_continuation_tick_budget",
    "independent_horizon_tick_budget",
    "nested_tick_budget_saved",
}
_MULTI_TAPE_COMPUTE_FIELDS = {
    "continuation_tape_count",
    "continuation_count_per_tape",
    "max_horizon_continuation_count",
    "exact_replay_continuation_count",
    "maximum_horizon_ticks",
    "actual_continuation_tick_count",
    "maximum_continuation_tick_budget",
}


_WORKER_MODEL: PublicRecurrentActorCritic | None = None
_WORKER_ARTIFACT_DIGEST: str | None = None
_WORKER_CONFIG: RecurrentCounterfactualCollectionConfig | None = None


class RecurrentCounterfactualCollectionError(ValueError):
    """Raised when deterministic exact collection fails closed."""


def derive_recurrent_counterfactual_collection_seed(
    *,
    namespace: str,
    identity: str,
) -> int:
    if namespace not in {
        RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE,
        RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE,
    }:
        raise RecurrentCounterfactualCollectionError(
            "counterfactual seed namespace is unsupported"
        )
    resolved_identity = _nonempty_trimmed_string(identity, field="seed identity")
    payload = f"{namespace}|{resolved_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        MAX_RECURRENT_COUNTERFACTUAL_SEED
    )


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualCollectionTask:
    task_id: str
    seed_role: str
    scenario: str
    environment_seed: int
    branch_tick_candidates: tuple[int, ...]
    source_policy_sampling_identity: str
    branch_selection_identity: str
    source_policy_sampling_seed: int | None = None
    branch_selection_seed: int | None = None
    branch_tick_stratum_index: int | None = None

    def __post_init__(self) -> None:
        task_id = _nonempty_trimmed_string(self.task_id, field="task_id")
        if self.seed_role not in RECURRENT_COUNTERFACTUAL_SEED_ROLES:
            raise RecurrentCounterfactualCollectionError(
                "seed_role must be a legacy or scale training role"
            )
        if self.scenario not in RECURRENT_COUNTERFACTUAL_SCENARIOS:
            raise RecurrentCounterfactualCollectionError("scenario is unsupported")
        scale_v1_role = self.seed_role in RECURRENT_COUNTERFACTUAL_SCALE_V1_SEED_ROLES
        scale_v2_role = self.seed_role in RECURRENT_COUNTERFACTUAL_SCALE_V2_SEED_ROLES
        if scale_v2_role:
            expected_role = (
                "scale_v2_train"
                if self.scenario == RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO
                else "scale_v2_curriculum"
            )
        elif scale_v1_role:
            expected_role = (
                "scale_train"
                if self.scenario == RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO
                else "scale_curriculum"
            )
        else:
            expected_role = (
                "train"
                if self.scenario == RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO
                else "curriculum"
            )
        if self.seed_role != expected_role:
            raise RecurrentCounterfactualCollectionError(
                "broad tasks require train seeds and fixture tasks require "
                "curriculum seeds"
            )
        environment_seed = _positive_seed(
            self.environment_seed,
            field="environment_seed",
        )
        if scale_v2_role:
            registry = SCALE_DEVELOPMENT_V2_SEED_REGISTRY
        elif scale_v1_role:
            registry = SCALE_DEVELOPMENT_SEED_REGISTRY
        else:
            registry = RECURRENT_SEED_REGISTRY
        if environment_seed not in registry[self.seed_role]:
            raise RecurrentCounterfactualCollectionError(
                "environment_seed is not registered for its training role"
            )
        branch_ticks = _branch_tick_candidates(self.branch_tick_candidates)
        stratum_index = (
            None
            if self.branch_tick_stratum_index is None
            else _nonnegative_int(
                self.branch_tick_stratum_index,
                field="branch_tick_stratum_index",
            )
        )
        if stratum_index is not None and stratum_index >= len(branch_ticks):
            raise RecurrentCounterfactualCollectionError(
                "branch_tick_stratum_index must index branch_tick_candidates"
            )
        source_identity = _nonempty_trimmed_string(
            self.source_policy_sampling_identity,
            field="source_policy_sampling_identity",
        )
        selection_identity = _nonempty_trimmed_string(
            self.branch_selection_identity,
            field="branch_selection_identity",
        )
        if source_identity == selection_identity:
            raise RecurrentCounterfactualCollectionError(
                "source-policy and branch-selection identities must differ"
            )
        expected_source_seed = derive_recurrent_counterfactual_collection_seed(
            namespace=RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE,
            identity=source_identity,
        )
        expected_selection_seed = derive_recurrent_counterfactual_collection_seed(
            namespace=RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE,
            identity=selection_identity,
        )
        source_seed = (
            expected_source_seed
            if self.source_policy_sampling_seed is None
            else _sampling_seed(
                self.source_policy_sampling_seed,
                field="source_policy_sampling_seed",
            )
        )
        selection_seed = (
            expected_selection_seed
            if self.branch_selection_seed is None
            else _sampling_seed(
                self.branch_selection_seed,
                field="branch_selection_seed",
            )
        )
        if source_seed != expected_source_seed:
            raise RecurrentCounterfactualCollectionError(
                "source_policy_sampling_seed does not match its namespaced identity"
            )
        if selection_seed != expected_selection_seed:
            raise RecurrentCounterfactualCollectionError(
                "branch_selection_seed does not match its namespaced identity"
            )
        if len({environment_seed, source_seed, selection_seed}) != 3:
            raise RecurrentCounterfactualCollectionError(
                "environment, source-policy, and branch-selection seeds must differ"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "branch_tick_candidates", branch_ticks)
        object.__setattr__(self, "source_policy_sampling_identity", source_identity)
        object.__setattr__(self, "branch_selection_identity", selection_identity)
        object.__setattr__(self, "source_policy_sampling_seed", source_seed)
        object.__setattr__(self, "branch_selection_seed", selection_seed)
        object.__setattr__(self, "branch_tick_stratum_index", stratum_index)


@dataclass(frozen=True, slots=True)
class _RecurrentCounterfactualTapeChunkTask:
    task_index: int
    task: RecurrentCounterfactualCollectionTask
    tape_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualCollectionConfig:
    horizons: tuple[int, ...] = (8, 32, 64)
    gamma: float = 0.99
    continuation_tape_count: int = 1
    terminal_target_world_tick: int | None = None
    uncertainty_penalty: float = 0.0

    def __post_init__(self) -> None:
        horizons = _nested_horizons(self.horizons)
        gamma = _finite_number(self.gamma, field="gamma")
        if not 0.0 < gamma <= 1.0:
            raise RecurrentCounterfactualCollectionError("gamma must be in (0, 1]")
        tape_count = _positive_int(
            self.continuation_tape_count,
            field="continuation_tape_count",
        )
        terminal_target = (
            None
            if self.terminal_target_world_tick is None
            else _positive_int(
                self.terminal_target_world_tick,
                field="terminal_target_world_tick",
            )
        )
        uncertainty_penalty = _finite_number(
            self.uncertainty_penalty,
            field="uncertainty_penalty",
        )
        if uncertainty_penalty < 0.0:
            raise RecurrentCounterfactualCollectionError(
                "uncertainty_penalty must be non-negative"
            )
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "continuation_tape_count", tape_count)
        object.__setattr__(self, "terminal_target_world_tick", terminal_target)
        object.__setattr__(self, "uncertainty_penalty", uncertainty_penalty)

    @property
    def multi_tape_enabled(self) -> bool:
        return (
            self.continuation_tape_count != 1
            or self.terminal_target_world_tick is not None
            or self.uncertainty_penalty != 0.0
        )


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualCollectionBundle:
    task: RecurrentCounterfactualCollectionTask
    rows: tuple[dict[str, object], ...]
    selection: dict[str, object]
    prefix_proof: dict[str, object]
    compute: dict[str, object]
    source_model_state_sha256: str
    source_artifact_digest: str
    valid_actions: tuple[str, ...]
    horizons: tuple[int, ...]
    exact_digest: str
    aggregate_rows: tuple[dict[str, object], ...] | None = None
    terminal_target: dict[str, object] | None = None
    multi_tape_compute: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualCollectionResult:
    contract_version: str
    source_model_state_sha256: str
    source_artifact_digest: str
    config: RecurrentCounterfactualCollectionConfig
    bundles: tuple[RecurrentCounterfactualCollectionBundle, ...]
    workers_requested: int
    workers_resolved: int
    worker_start_method: str
    torch_threads_per_worker: int
    ordered_merge: bool
    aggregate_compute: dict[str, int]
    training_ran: bool
    runtime_action_selection_changed: bool
    promotion_authorized: bool
    exact_digest: str
    execution_contract_version: str = (
        RECURRENT_COUNTERFACTUAL_LEGACY_EXECUTION_CONTRACT_VERSION
    )
    execution_compute: dict[str, int] | None = None


def collect_recurrent_counterfactual_bundles(
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentCounterfactualCollectionTask],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualCollectionConfig | None = None,
    workers: int = 1,
) -> RecurrentCounterfactualCollectionResult:
    """Collect ordered exact branch bundles from one frozen post-PPO model."""

    if not isinstance(model, PublicRecurrentActorCritic):
        raise RecurrentCounterfactualCollectionError(
            "model must be a PublicRecurrentActorCritic"
        )
    resolved_artifact = _sha256(artifact_digest, field="artifact_digest")
    resolved_config = config or RecurrentCounterfactualCollectionConfig()
    if not isinstance(resolved_config, RecurrentCounterfactualCollectionConfig):
        raise RecurrentCounterfactualCollectionError(
            "config must be a RecurrentCounterfactualCollectionConfig"
        )
    task_tuple = _validated_tasks(tasks)
    _validate_task_selection_modes(task_tuple, config=resolved_config)
    worker_count = _worker_count(workers)
    source_model_digest = recurrent_model_state_sha256(model)
    behavior_model = frozen_cpu_model_copy(model)
    if recurrent_model_state_sha256(behavior_model) != source_model_digest:
        raise RecurrentCounterfactualCollectionError(
            "frozen CPU source model digest drifted"
        )

    workers_resolved = _resolved_worker_count(
        workers=worker_count,
        task_count=len(task_tuple),
        config=resolved_config,
    )
    tape_parallel_execution = _tape_parallelism_used(
        workers=worker_count,
        task_count=len(task_tuple),
        config=resolved_config,
    )
    execution_compute: dict[str, int] | None = None
    if worker_count == 1:
        materializations = tuple(
            _collect_one(
                behavior_model,
                task,
                artifact_digest=resolved_artifact,
                config=resolved_config,
            )
            for task in task_tuple
        )
    elif tape_parallel_execution:
        materializations, execution_compute = (
            _collect_parallel_tape_chunks(
                behavior_model,
                task_tuple,
                artifact_digest=resolved_artifact,
                config=resolved_config,
                workers=worker_count,
            )
        )
    elif len(task_tuple) == 1:
        materializations = (
            _collect_one(
                behavior_model,
                task_tuple[0],
                artifact_digest=resolved_artifact,
                config=resolved_config,
            ),
        )
    else:
        materializations = _collect_parallel(
            behavior_model,
            task_tuple,
            artifact_digest=resolved_artifact,
            config=resolved_config,
            workers=worker_count,
        )
    if recurrent_model_state_sha256(model) != source_model_digest:
        raise RecurrentCounterfactualCollectionError(
            "learner model changed during frozen counterfactual collection"
        )
    bundles = tuple(
        _bundle_from_materialization(task, materialization)
        for task, materialization in zip(
            task_tuple,
            materializations,
            strict=True,
        )
    )
    aggregate_compute = _aggregate_compute(bundles)
    contract_version = _collection_contract_version(resolved_config)
    execution_contract_version = (
        RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
        if tape_parallel_execution
        else RECURRENT_COUNTERFACTUAL_LEGACY_EXECUTION_CONTRACT_VERSION
    )
    result_without_digest = {
        "contract_version": contract_version,
        "source_model_state_sha256": source_model_digest,
        "source_artifact_digest": resolved_artifact,
        "config": recurrent_counterfactual_collection_config_payload(resolved_config),
        "bundles": [_bundle_payload(bundle) for bundle in bundles],
        "workers_requested": worker_count,
        "workers_resolved": workers_resolved,
        "worker_start_method": RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD,
        "torch_threads_per_worker": RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS,
        "ordered_merge": True,
        "aggregate_compute": aggregate_compute,
        "training_ran": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    if (
        execution_contract_version
        == RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
    ):
        if execution_compute is None:
            raise AssertionError("tape-parallel execution compute was not recorded")
        result_without_digest["execution_contract_version"] = (
            execution_contract_version
        )
        result_without_digest["execution_compute"] = execution_compute
    result = RecurrentCounterfactualCollectionResult(
        contract_version=contract_version,
        source_model_state_sha256=source_model_digest,
        source_artifact_digest=resolved_artifact,
        config=resolved_config,
        bundles=bundles,
        workers_requested=worker_count,
        workers_resolved=workers_resolved,
        worker_start_method=RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD,
        torch_threads_per_worker=RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS,
        ordered_merge=True,
        aggregate_compute=aggregate_compute,
        training_ran=False,
        runtime_action_selection_changed=False,
        promotion_authorized=False,
        exact_digest=stable_payload_digest(result_without_digest),
        execution_contract_version=execution_contract_version,
        execution_compute=execution_compute,
    )
    validate_recurrent_counterfactual_collection_result(
        result,
        model=model,
        artifact_digest=resolved_artifact,
    )
    return result


def recurrent_counterfactual_collection_result_payload(
    result: RecurrentCounterfactualCollectionResult,
) -> dict[str, object]:
    validate_recurrent_counterfactual_collection_result(result)
    payload = {
        "contract_version": result.contract_version,
        "source_model_state_sha256": result.source_model_state_sha256,
        "source_artifact_digest": result.source_artifact_digest,
        "config": recurrent_counterfactual_collection_config_payload(result.config),
        "bundles": [_bundle_payload(bundle) for bundle in result.bundles],
        "workers_requested": result.workers_requested,
        "workers_resolved": result.workers_resolved,
        "worker_start_method": result.worker_start_method,
        "torch_threads_per_worker": result.torch_threads_per_worker,
        "ordered_merge": result.ordered_merge,
        "aggregate_compute": dict(result.aggregate_compute),
        "training_ran": result.training_ran,
        "runtime_action_selection_changed": result.runtime_action_selection_changed,
        "promotion_authorized": result.promotion_authorized,
        "exact_digest": result.exact_digest,
    }
    if (
        result.execution_contract_version
        == RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
    ):
        payload["execution_contract_version"] = result.execution_contract_version
        payload["execution_compute"] = copy.deepcopy(result.execution_compute)
    return copy.deepcopy(payload)


def validate_recurrent_counterfactual_collection_result(
    result: RecurrentCounterfactualCollectionResult,
    *,
    model: PublicRecurrentActorCritic | None = None,
    artifact_digest: str | None = None,
) -> None:
    if not isinstance(result, RecurrentCounterfactualCollectionResult):
        raise RecurrentCounterfactualCollectionError(
            "result must be a RecurrentCounterfactualCollectionResult"
        )
    if not isinstance(result.config, RecurrentCounterfactualCollectionConfig):
        raise RecurrentCounterfactualCollectionError(
            "collection config type drifted"
        )
    normalized_config = RecurrentCounterfactualCollectionConfig(
        horizons=result.config.horizons,
        gamma=result.config.gamma,
        continuation_tape_count=result.config.continuation_tape_count,
        terminal_target_world_tick=result.config.terminal_target_world_tick,
        uncertainty_penalty=result.config.uncertainty_penalty,
    )
    if normalized_config != result.config:
        raise RecurrentCounterfactualCollectionError(
            "collection config failed exact normalization"
        )
    if result.contract_version != _collection_contract_version(result.config):
        raise RecurrentCounterfactualCollectionError(
            "collection contract version drifted"
        )
    if result.execution_contract_version not in {
        RECURRENT_COUNTERFACTUAL_LEGACY_EXECUTION_CONTRACT_VERSION,
        RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION,
    }:
        raise RecurrentCounterfactualCollectionError(
            "collection execution contract version drifted"
        )
    if (
        result.execution_contract_version
        == RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
        and (
            not result.config.multi_tape_enabled
            or result.config.continuation_tape_count <= 1
        )
    ):
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel execution requires multiple continuation tapes"
        )
    if (
        result.execution_contract_version
        == RECURRENT_COUNTERFACTUAL_LEGACY_EXECUTION_CONTRACT_VERSION
        and result.execution_compute is not None
    ):
        raise RecurrentCounterfactualCollectionError(
            "legacy execution cannot contain tape-parallel compute"
        )
    _sha256(result.source_model_state_sha256, field="source model digest")
    _sha256(result.source_artifact_digest, field="source artifact digest")
    if artifact_digest is not None and result.source_artifact_digest != _sha256(
        artifact_digest,
        field="artifact_digest",
    ):
        raise RecurrentCounterfactualCollectionError(
            "collection artifact digest does not match"
        )
    if model is not None:
        if not isinstance(model, PublicRecurrentActorCritic):
            raise RecurrentCounterfactualCollectionError(
                "model must be a PublicRecurrentActorCritic"
            )
        if recurrent_model_state_sha256(model) != result.source_model_state_sha256:
            raise RecurrentCounterfactualCollectionError(
                "collection source model digest is stale"
            )
    if not result.bundles:
        raise RecurrentCounterfactualCollectionError(
            "collection result must contain bundles"
        )
    normalized_tasks: list[RecurrentCounterfactualCollectionTask] = []
    for bundle in result.bundles:
        task = bundle.task
        if not isinstance(task, RecurrentCounterfactualCollectionTask):
            raise RecurrentCounterfactualCollectionError(
                "collection bundle task type drifted"
            )
        normalized_task = RecurrentCounterfactualCollectionTask(
            task_id=task.task_id,
            seed_role=task.seed_role,
            scenario=task.scenario,
            environment_seed=task.environment_seed,
            branch_tick_candidates=task.branch_tick_candidates,
            source_policy_sampling_identity=task.source_policy_sampling_identity,
            branch_selection_identity=task.branch_selection_identity,
            source_policy_sampling_seed=task.source_policy_sampling_seed,
            branch_selection_seed=task.branch_selection_seed,
            branch_tick_stratum_index=task.branch_tick_stratum_index,
        )
        if normalized_task != task:
            raise RecurrentCounterfactualCollectionError(
                "collection task failed exact normalization"
            )
        normalized_tasks.append(normalized_task)
    validated_tasks = _validated_tasks(tuple(normalized_tasks))
    _validate_task_selection_modes(
        validated_tasks,
        config=result.config,
    )
    task_ids: set[str] = set()
    for bundle in result.bundles:
        _validate_bundle(
            bundle,
            config=result.config,
            source_model_digest=result.source_model_state_sha256,
            artifact_digest=result.source_artifact_digest,
        )
        if bundle.task.task_id in task_ids:
            raise RecurrentCounterfactualCollectionError(
                "collection bundle task ids must be unique"
            )
        task_ids.add(bundle.task.task_id)
    expected_aggregate_compute = _aggregate_compute(result.bundles)
    if set(result.aggregate_compute) != set(expected_aggregate_compute):
        raise RecurrentCounterfactualCollectionError(
            "aggregate compute field set drifted"
        )
    parsed_aggregate_compute = {
        field: _nonnegative_int(
            result.aggregate_compute.get(field),
            field=f"aggregate compute {field}",
        )
        for field in expected_aggregate_compute
    }
    if parsed_aggregate_compute != expected_aggregate_compute:
        raise RecurrentCounterfactualCollectionError(
            "aggregate compute diagnostics drifted"
        )
    if result.workers_requested != _worker_count(result.workers_requested):
        raise RecurrentCounterfactualCollectionError("workers_requested drifted")
    workers_resolved = _positive_int(
        result.workers_resolved,
        field="workers_resolved",
    )
    expected_workers_resolved = (
        _resolved_worker_count(
            workers=result.workers_requested,
            task_count=len(result.bundles),
            config=result.config,
        )
        if result.execution_contract_version
        == RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
        else min(result.workers_requested, len(result.bundles))
    )
    if workers_resolved != expected_workers_resolved:
        raise RecurrentCounterfactualCollectionError("workers_resolved drifted")
    if (
        result.execution_contract_version
        == RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
    ):
        if not _tape_parallelism_used(
            workers=result.workers_requested,
            task_count=len(result.bundles),
            config=result.config,
        ):
            raise RecurrentCounterfactualCollectionError(
                "tape-parallel execution metadata does not describe the actual path"
            )
        _validate_tape_parallel_execution_compute(
            result.execution_compute,
            tasks=validated_tasks,
            config=result.config,
            bundles=result.bundles,
            workers=result.workers_requested,
        )
    torch_threads_per_worker = _positive_int(
        result.torch_threads_per_worker,
        field="torch_threads_per_worker",
    )
    if (
        result.worker_start_method != RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD
        or torch_threads_per_worker
        != RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS
        or result.ordered_merge is not True
    ):
        raise RecurrentCounterfactualCollectionError(
            "collection execution contract drifted"
        )
    for flag in (
        "training_ran",
        "runtime_action_selection_changed",
        "promotion_authorized",
    ):
        if getattr(result, flag) is not False:
            raise RecurrentCounterfactualCollectionError(
                f"closed lifecycle flag {flag} must remain false"
            )
    without_digest = _result_payload_without_validation(result)
    observed_digest = without_digest.pop("exact_digest")
    if observed_digest != stable_payload_digest(without_digest):
        raise RecurrentCounterfactualCollectionError("collection exact digest mismatch")


def _result_payload_without_validation(
    result: RecurrentCounterfactualCollectionResult,
) -> dict[str, object]:
    payload = {
        "contract_version": result.contract_version,
        "source_model_state_sha256": result.source_model_state_sha256,
        "source_artifact_digest": result.source_artifact_digest,
        "config": recurrent_counterfactual_collection_config_payload(result.config),
        "bundles": [_bundle_payload(bundle) for bundle in result.bundles],
        "workers_requested": result.workers_requested,
        "workers_resolved": result.workers_resolved,
        "worker_start_method": result.worker_start_method,
        "torch_threads_per_worker": result.torch_threads_per_worker,
        "ordered_merge": result.ordered_merge,
        "aggregate_compute": dict(result.aggregate_compute),
        "training_ran": result.training_ran,
        "runtime_action_selection_changed": result.runtime_action_selection_changed,
        "promotion_authorized": result.promotion_authorized,
        "exact_digest": result.exact_digest,
    }
    if (
        result.execution_contract_version
        == RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION
    ):
        payload["execution_contract_version"] = result.execution_contract_version
        payload["execution_compute"] = copy.deepcopy(result.execution_compute)
    return payload


def _collection_contract_version(
    config: RecurrentCounterfactualCollectionConfig,
) -> str:
    if not isinstance(config, RecurrentCounterfactualCollectionConfig):
        raise RecurrentCounterfactualCollectionError("collection config type drifted")
    return (
        RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION
        if config.multi_tape_enabled
        else RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION
    )


def recurrent_counterfactual_collection_config_payload(
    config: RecurrentCounterfactualCollectionConfig,
) -> dict[str, object]:
    """Serialize config without retroactively changing the legacy v1 shape."""

    if not isinstance(config, RecurrentCounterfactualCollectionConfig):
        raise RecurrentCounterfactualCollectionError(
            "collection config payload requires the exact config type"
        )
    payload: dict[str, object] = {
        "horizons": config.horizons,
        "gamma": config.gamma,
    }
    if config.multi_tape_enabled:
        payload.update(
            {
                "continuation_tape_count": config.continuation_tape_count,
                "terminal_target_world_tick": config.terminal_target_world_tick,
                "uncertainty_penalty": config.uncertainty_penalty,
            }
        )
    return payload


def recurrent_counterfactual_collection_task_payload(
    task: RecurrentCounterfactualCollectionTask,
) -> dict[str, object]:
    """Serialize a task while preserving the legacy task payload shape."""

    if not isinstance(task, RecurrentCounterfactualCollectionTask):
        raise RecurrentCounterfactualCollectionError(
            "collection task payload requires the exact task type"
        )
    payload = asdict(task)
    if task.branch_tick_stratum_index is None:
        payload.pop("branch_tick_stratum_index")
    return payload


def _collect_one(
    model: PublicRecurrentActorCritic,
    task: RecurrentCounterfactualCollectionTask,
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualCollectionConfig,
) -> dict[str, object]:
    try:
        return build_recurrent_counterfactual_nested_horizon_materialization(
            model,
            artifact_digest=artifact_digest,
            seed_role=task.seed_role,
            environment_seed=task.environment_seed,
            scenario=task.scenario,
            branch_tick_candidates=task.branch_tick_candidates,
            horizons=config.horizons,
            source_policy_sampling_seed=_sampling_seed(
                task.source_policy_sampling_seed,
                field="source policy sampling seed",
            ),
            branch_selection_seed=_sampling_seed(
                task.branch_selection_seed,
                field="branch selection seed",
            ),
            branch_tick_stratum_index=task.branch_tick_stratum_index,
            gamma=config.gamma,
            continuation_tape_count=config.continuation_tape_count,
            continuation_tape_identity=(
                f"{task.task_id}:continuation-tapes"
                if config.multi_tape_enabled
                else None
            ),
            terminal_target_world_tick=config.terminal_target_world_tick,
            uncertainty_penalty=config.uncertainty_penalty,
        )
    except RecurrentCounterfactualBranchError as error:
        raise RecurrentCounterfactualCollectionError(
            f"counterfactual task {task.task_id!r} failed exact collection"
        ) from error


def _collect_parallel(
    model: PublicRecurrentActorCritic,
    tasks: tuple[RecurrentCounterfactualCollectionTask, ...],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualCollectionConfig,
    workers: int,
) -> tuple[dict[str, object], ...]:
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    context = mp.get_context(RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD)
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(model.config, model_state, artifact_digest, config),
        ) as executor:
            results = tuple(executor.map(_collect_worker, tasks))
    except Exception as error:
        raise RecurrentCounterfactualCollectionError(
            "process-parallel counterfactual collection failed closed"
        ) from error
    if len(results) != len(tasks):
        raise RecurrentCounterfactualCollectionError(
            "parallel counterfactual result count drifted"
        )
    return results


def _collect_parallel_tape_chunks(
    model: PublicRecurrentActorCritic,
    tasks: tuple[RecurrentCounterfactualCollectionTask, ...],
    *,
    artifact_digest: str,
    config: RecurrentCounterfactualCollectionConfig,
    workers: int,
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    if not config.multi_tape_enabled or config.continuation_tape_count <= 1:
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel collection requires multiple continuation tapes"
        )
    work_items = _tape_chunk_work_items(
        tasks,
        tape_count=config.continuation_tape_count,
        workers=workers,
    )
    if len(work_items) <= len(tasks):
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel collection requires at least one split bundle"
        )
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    context = mp.get_context(RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD)
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(work_items)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(model.config, model_state, artifact_digest, config),
        ) as executor:
            chunk_materializations = tuple(
                executor.map(_collect_tape_chunk_worker, work_items)
            )
    except Exception as error:
        raise RecurrentCounterfactualCollectionError(
            "process-parallel counterfactual tape collection failed closed"
        ) from error
    if len(chunk_materializations) != len(work_items):
        raise RecurrentCounterfactualCollectionError(
            "parallel counterfactual tape result count drifted"
        )
    grouped: list[list[Mapping[str, object]]] = [[] for _ in tasks]
    for work_item, materialization in zip(
        work_items,
        chunk_materializations,
        strict=True,
    ):
        grouped[work_item.task_index].append(materialization)
    execution_compute = _tape_parallel_execution_compute(grouped)
    merged: list[dict[str, object]] = []
    for task, materializations in zip(tasks, grouped, strict=True):
        try:
            merged.append(
                merge_recurrent_counterfactual_nested_horizon_tape_chunks(
                    materializations,
                    continuation_tape_count=config.continuation_tape_count,
                    continuation_tape_identity=f"{task.task_id}:continuation-tapes",
                    terminal_target_world_tick=config.terminal_target_world_tick,
                    uncertainty_penalty=config.uncertainty_penalty,
                )
            )
        except RecurrentCounterfactualBranchError as error:
            raise RecurrentCounterfactualCollectionError(
                f"counterfactual task {task.task_id!r} failed ordered tape merge"
            ) from error
    return tuple(merged), execution_compute


def _tape_chunk_work_items(
    tasks: tuple[RecurrentCounterfactualCollectionTask, ...],
    *,
    tape_count: int,
    workers: int,
) -> tuple[_RecurrentCounterfactualTapeChunkTask, ...]:
    resolved_tape_count = _positive_int(tape_count, field="tape_count")
    resolved_workers = _worker_count(workers)
    maximum_chunks_per_task = _maximum_tape_chunks_per_task(
        resolved_tape_count
    )
    target_chunk_count = min(
        resolved_workers,
        len(tasks) * maximum_chunks_per_task,
    )
    chunk_counts = [1 for _ in tasks]
    extra_chunks = max(0, target_chunk_count - len(tasks))
    while extra_chunks > 0:
        progressed = False
        for task_index in range(len(tasks)):
            if chunk_counts[task_index] >= maximum_chunks_per_task:
                continue
            chunk_counts[task_index] += 1
            extra_chunks -= 1
            progressed = True
            if extra_chunks == 0:
                break
        if not progressed:
            raise AssertionError("tape chunk allocation could not make progress")
    work_items: list[_RecurrentCounterfactualTapeChunkTask] = []
    for task_index, (task, chunk_count) in enumerate(
        zip(tasks, chunk_counts, strict=True)
    ):
        base_size, remainder = divmod(resolved_tape_count, chunk_count)
        start = 0
        for chunk_index in range(chunk_count):
            size = base_size + int(chunk_index < remainder)
            stop = start + size
            work_items.append(
                _RecurrentCounterfactualTapeChunkTask(
                    task_index=task_index,
                    task=task,
                    tape_indices=tuple(range(start, stop)),
                )
            )
            start = stop
        if start != resolved_tape_count:
            raise AssertionError("tape chunk allocation did not cover all tapes")
        if chunk_count > 1 and any(
            len(work_item.tape_indices)
            < MIN_RECURRENT_COUNTERFACTUAL_TAPES_PER_CHUNK
            for work_item in work_items
            if work_item.task_index == task_index
        ):
            raise AssertionError("tape chunk allocation created an undersized chunk")
    return tuple(work_items)


def _maximum_tape_chunks_per_task(tape_count: int) -> int:
    resolved_tape_count = _positive_int(tape_count, field="tape_count")
    return max(
        1,
        resolved_tape_count // MIN_RECURRENT_COUNTERFACTUAL_TAPES_PER_CHUNK,
    )


def _tape_parallelism_used(
    *,
    workers: int,
    task_count: int,
    config: RecurrentCounterfactualCollectionConfig,
) -> bool:
    resolved_workers = _worker_count(workers)
    resolved_task_count = _positive_int(task_count, field="task_count")
    return (
        config.multi_tape_enabled
        and config.continuation_tape_count > 1
        and resolved_workers > resolved_task_count
        and _maximum_tape_chunks_per_task(config.continuation_tape_count) > 1
    )


def _tape_parallel_execution_compute(
    grouped_materializations: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, int]:
    if not grouped_materializations or any(
        not materializations for materializations in grouped_materializations
    ):
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel execution compute requires every bundle chunk group"
        )
    chunk_count = sum(
        len(materializations) for materializations in grouped_materializations
    )
    split_bundle_count = sum(
        int(len(materializations) > 1)
        for materializations in grouped_materializations
    )
    maximum_chunks_per_bundle = max(
        len(materializations) for materializations in grouped_materializations
    )
    source_tick_executions = 0
    canonical_source_tick_executions = 0
    legacy_actual_continuation_tick_count = 0
    canonical_legacy_actual_continuation_tick_count = 0
    multi_tape_actual_continuation_tick_count = 0
    for materializations in grouped_materializations:
        for materialization_index, materialization in enumerate(materializations):
            compute = _mapping(
                materialization.get("compute"),
                field="tape chunk source compute",
            )
            source_ticks = _nonnegative_int(
                compute.get("source_tick_executions"),
                field="tape chunk source ticks",
            )
            legacy_ticks = _nonnegative_int(
                compute.get("actual_continuation_tick_count"),
                field="tape chunk legacy continuation ticks",
            )
            chunk = _mapping(
                materialization.get("multi_tape_chunk"),
                field="tape chunk evidence",
            )
            tape_ticks = _nonnegative_int(
                chunk.get("actual_continuation_tick_count"),
                field="tape chunk continuation ticks",
            )
            source_tick_executions += source_ticks
            legacy_actual_continuation_tick_count += legacy_ticks
            multi_tape_actual_continuation_tick_count += tape_ticks
            if materialization_index == 0:
                canonical_source_tick_executions += source_ticks
                canonical_legacy_actual_continuation_tick_count += legacy_ticks
    duplicated_source_tick_executions = (
        source_tick_executions - canonical_source_tick_executions
    )
    duplicated_legacy_actual_continuation_tick_count = (
        legacy_actual_continuation_tick_count
        - canonical_legacy_actual_continuation_tick_count
    )
    return {
        "tape_chunk_work_item_count": chunk_count,
        "split_bundle_count": split_bundle_count,
        "maximum_chunks_per_bundle": maximum_chunks_per_bundle,
        "source_tick_executions": source_tick_executions,
        "canonical_source_tick_executions": canonical_source_tick_executions,
        "duplicated_source_tick_executions": duplicated_source_tick_executions,
        "legacy_actual_continuation_tick_count": (
            legacy_actual_continuation_tick_count
        ),
        "canonical_legacy_actual_continuation_tick_count": (
            canonical_legacy_actual_continuation_tick_count
        ),
        "duplicated_legacy_actual_continuation_tick_count": (
            duplicated_legacy_actual_continuation_tick_count
        ),
        "multi_tape_actual_continuation_tick_count": (
            multi_tape_actual_continuation_tick_count
        ),
        "total_actual_simulator_tick_executions": (
            source_tick_executions
            + legacy_actual_continuation_tick_count
            + multi_tape_actual_continuation_tick_count
        ),
    }


def _initialize_worker(
    model_config: RecurrentActorCriticConfig,
    model_state: dict[str, torch.Tensor],
    artifact_digest: str,
    config: RecurrentCounterfactualCollectionConfig,
) -> None:
    global _WORKER_MODEL
    global _WORKER_ARTIFACT_DIGEST
    global _WORKER_CONFIG
    torch.set_num_threads(RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS)
    torch.set_num_interop_threads(RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS)
    model = PublicRecurrentActorCritic(
        model_config,
        initialization_seed=1,
    ).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _WORKER_MODEL = model
    _WORKER_ARTIFACT_DIGEST = artifact_digest
    _WORKER_CONFIG = config


def _collect_worker(
    task: RecurrentCounterfactualCollectionTask,
) -> dict[str, object]:
    if (
        _WORKER_MODEL is None
        or _WORKER_ARTIFACT_DIGEST is None
        or _WORKER_CONFIG is None
    ):
        raise RecurrentCounterfactualCollectionError(
            "counterfactual worker was not initialized"
        )
    return _collect_one(
        _WORKER_MODEL,
        task,
        artifact_digest=_WORKER_ARTIFACT_DIGEST,
        config=_WORKER_CONFIG,
    )


def _collect_tape_chunk_worker(
    work_item: _RecurrentCounterfactualTapeChunkTask,
) -> dict[str, object]:
    if (
        _WORKER_MODEL is None
        or _WORKER_ARTIFACT_DIGEST is None
        or _WORKER_CONFIG is None
    ):
        raise RecurrentCounterfactualCollectionError(
            "counterfactual tape worker was not initialized"
        )
    task = work_item.task
    config = _WORKER_CONFIG
    try:
        return (
            build_recurrent_counterfactual_nested_horizon_tape_chunk_materialization(
                _WORKER_MODEL,
                artifact_digest=_WORKER_ARTIFACT_DIGEST,
                seed_role=task.seed_role,
                environment_seed=task.environment_seed,
                scenario=task.scenario,
                branch_tick_candidates=task.branch_tick_candidates,
                horizons=config.horizons,
                source_policy_sampling_seed=_sampling_seed(
                    task.source_policy_sampling_seed,
                    field="source policy sampling seed",
                ),
                branch_selection_seed=_sampling_seed(
                    task.branch_selection_seed,
                    field="branch selection seed",
                ),
                branch_tick_stratum_index=task.branch_tick_stratum_index,
                gamma=config.gamma,
                continuation_tape_count=config.continuation_tape_count,
                continuation_tape_identity=(
                    f"{task.task_id}:continuation-tapes"
                ),
                continuation_tape_indices=work_item.tape_indices,
                terminal_target_world_tick=config.terminal_target_world_tick,
                uncertainty_penalty=config.uncertainty_penalty,
            )
        )
    except RecurrentCounterfactualBranchError as error:
        raise RecurrentCounterfactualCollectionError(
            f"counterfactual task {task.task_id!r} tape chunk "
            f"{work_item.tape_indices!r} failed exact collection"
        ) from error


def _bundle_from_materialization(
    task: RecurrentCounterfactualCollectionTask,
    value: Mapping[str, object],
) -> RecurrentCounterfactualCollectionBundle:
    rows = value.get("rows")
    valid_actions = value.get("valid_actions")
    horizons = value.get("horizons")
    aggregate_rows = value.get("aggregate_rows")
    terminal_target = value.get("terminal_target")
    multi_tape_compute = value.get("multi_tape_compute")
    if (
        not isinstance(rows, tuple)
        or not isinstance(valid_actions, tuple)
        or not isinstance(
            horizons,
            tuple,
        )
    ):
        raise RecurrentCounterfactualCollectionError(
            "branch materialization tuple fields drifted"
        )
    fields = {
        "task": recurrent_counterfactual_collection_task_payload(task),
        "rows": list(rows),
        "selection": dict(_mapping(value.get("selection"), field="selection")),
        "prefix_proof": dict(_mapping(value.get("prefix_proof"), field="prefix_proof")),
        "compute": dict(_mapping(value.get("compute"), field="compute")),
        "source_model_state_sha256": value.get("source_model_state_sha256"),
        "source_artifact_digest": value.get("source_artifact_digest"),
        "valid_actions": list(valid_actions),
        "horizons": list(horizons),
    }
    if aggregate_rows is not None:
        if not isinstance(aggregate_rows, tuple):
            raise RecurrentCounterfactualCollectionError(
                "aggregate rows must be a tuple when present"
            )
        fields["aggregate_rows"] = copy.deepcopy(list(aggregate_rows))
        fields["terminal_target"] = (
            None
            if terminal_target is None
            else copy.deepcopy(dict(_mapping(terminal_target, field="terminal target")))
        )
        fields["multi_tape_compute"] = copy.deepcopy(
            dict(_mapping(multi_tape_compute, field="multi-tape compute"))
        )
    elif terminal_target is not None or multi_tape_compute is not None:
        raise RecurrentCounterfactualCollectionError(
            "terminal or compute evidence cannot exist without aggregate rows"
        )
    return RecurrentCounterfactualCollectionBundle(
        task=task,
        rows=tuple(copy.deepcopy(rows)),
        selection=copy.deepcopy(fields["selection"]),
        prefix_proof=copy.deepcopy(fields["prefix_proof"]),
        compute=copy.deepcopy(fields["compute"]),
        source_model_state_sha256=_sha256(
            fields["source_model_state_sha256"],
            field="source model digest",
        ),
        source_artifact_digest=_sha256(
            fields["source_artifact_digest"],
            field="source artifact digest",
        ),
        valid_actions=tuple(valid_actions),
        horizons=tuple(horizons),
        aggregate_rows=(
            None if aggregate_rows is None else tuple(copy.deepcopy(aggregate_rows))
        ),
        terminal_target=(
            None if terminal_target is None else copy.deepcopy(dict(terminal_target))
        ),
        multi_tape_compute=(
            None
            if multi_tape_compute is None
            else copy.deepcopy(dict(multi_tape_compute))
        ),
        exact_digest=stable_payload_digest(fields),
    )


def _validate_bundle(
    bundle: RecurrentCounterfactualCollectionBundle,
    *,
    config: RecurrentCounterfactualCollectionConfig,
    source_model_digest: str,
    artifact_digest: str,
) -> None:
    if not isinstance(bundle, RecurrentCounterfactualCollectionBundle):
        raise RecurrentCounterfactualCollectionError(
            "bundle must be a RecurrentCounterfactualCollectionBundle"
        )
    task = bundle.task
    source_identity = _nonempty_trimmed_string(
        task.source_policy_sampling_identity,
        field="source_policy_sampling_identity",
    )
    selection_identity = _nonempty_trimmed_string(
        task.branch_selection_identity,
        field="branch_selection_identity",
    )
    if source_identity == selection_identity:
        raise RecurrentCounterfactualCollectionError(
            "source-policy and branch-selection identities must differ"
        )
    expected_source_seed = derive_recurrent_counterfactual_collection_seed(
        namespace=RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE,
        identity=source_identity,
    )
    expected_selection_seed = derive_recurrent_counterfactual_collection_seed(
        namespace=RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE,
        identity=selection_identity,
    )
    if (
        _sampling_seed(
            task.source_policy_sampling_seed,
            field="source_policy_sampling_seed",
        )
        != expected_source_seed
        or _sampling_seed(
            task.branch_selection_seed,
            field="branch_selection_seed",
        )
        != expected_selection_seed
    ):
        raise RecurrentCounterfactualCollectionError(
            "collection task sampling seeds do not match their namespaced identities"
        )
    if bundle.source_model_state_sha256 != source_model_digest:
        raise RecurrentCounterfactualCollectionError("bundle model digest drifted")
    if bundle.source_artifact_digest != artifact_digest:
        raise RecurrentCounterfactualCollectionError("bundle artifact digest drifted")
    if not isinstance(bundle.horizons, tuple):
        raise RecurrentCounterfactualCollectionError(
            "bundle horizons must be a tuple"
        )
    parsed_bundle_horizons = tuple(
        _positive_int(horizon, field="bundle horizon")
        for horizon in bundle.horizons
    )
    if (
        parsed_bundle_horizons != config.horizons
        or len(bundle.rows) != len(config.horizons)
    ):
        raise RecurrentCounterfactualCollectionError("bundle horizons drifted")
    if tuple(action for action in ACTION_NAMES if action in bundle.valid_actions) != (
        bundle.valid_actions
    ):
        raise RecurrentCounterfactualCollectionError(
            "bundle valid actions are not in stable action order"
        )
    context_digest: str | None = None
    source_row_metadata: dict[str, object] | None = None
    source_row_optimizer: dict[str, object] | None = None
    source_row_behavior: dict[str, object] | None = None
    immutable_source_metadata_fields = (
        "seed_role",
        "environment_seed",
        "scenario",
        "branch_tick",
        "focal_agent_id",
        "source_artifact_digest",
        "source_model_state_sha256",
        "policy_sampling_seed",
        "source_action_selection",
        "source_decision_index",
        "source_record_digest",
        "source_decision_diagnostics_digest",
        "source_observation_digest",
        "source_action_mask_digest",
        "source_sampling_state_sha256",
        "source_public_history_prefix_sha256",
        "source_pre_boundary_environment_rng_state_sha256",
        "source_pre_boundary_policy_sampling_state_sha256",
        "source_boundary_identity_sha256",
    )
    for horizon, row in zip(bundle.horizons, bundle.rows, strict=True):
        try:
            validate_recurrent_counterfactual_branch_row(row)
        except RecurrentCounterfactualBranchError as error:
            raise RecurrentCounterfactualCollectionError(
                "bundle contains an invalid branch row"
            ) from error
        metadata = _mapping(row.get("metadata"), field="row metadata")
        optimizer = _mapping(row.get("optimizer_context"), field="optimizer context")
        labels = _mapping(row.get("labels"), field="row labels")
        row_behavior = {
            "source_requested_action": labels.get("source_requested_action"),
            "source_behavior_distribution": copy.deepcopy(
                labels.get("source_behavior_distribution")
            ),
        }
        expected_row_schema = (
            RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION
            if config.multi_tape_enabled
            else RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION
        )
        if row.get("schema_version") != expected_row_schema:
            raise RecurrentCounterfactualCollectionError(
                "bundle source branch-row schema drifted from collection mode"
            )
        if metadata.get("horizon_ticks") != horizon:
            raise RecurrentCounterfactualCollectionError("row horizon drifted")
        if metadata.get("gamma") != config.gamma:
            raise RecurrentCounterfactualCollectionError(
                "row return discount gamma drifted from collection config"
            )
        if (
            metadata.get("seed_role") != bundle.task.seed_role
            or metadata.get("environment_seed") != bundle.task.environment_seed
            or metadata.get("scenario") != bundle.task.scenario
            or metadata.get("policy_sampling_seed")
            != bundle.task.source_policy_sampling_seed
            or metadata.get("branch_tick")
            != bundle.selection.get("selected_branch_tick")
        ):
            raise RecurrentCounterfactualCollectionError(
                "row source identity drifted from its collection task"
            )
        if (
            optimizer.get("source_model_state_sha256") != source_model_digest
            or optimizer.get("source_artifact_digest") != artifact_digest
        ):
            raise RecurrentCounterfactualCollectionError("row source identity drifted")
        observed_actions = tuple(
            _mapping(outcome, field="action outcome").get("action")
            for outcome in labels.get("action_outcomes", [])
        )
        if observed_actions != bundle.valid_actions:
            raise RecurrentCounterfactualCollectionError(
                "row all-valid-action coverage drifted"
            )
        observed_context_digest = stable_payload_digest(
            row.get("trainable_public_context")
        )
        if context_digest is None:
            context_digest = observed_context_digest
            source_row_metadata = dict(metadata)
            source_row_optimizer = dict(optimizer)
            source_row_behavior = row_behavior
        elif observed_context_digest != context_digest:
            raise RecurrentCounterfactualCollectionError(
                "bundle horizons do not share one trainable public context"
            )
        elif (
            source_row_metadata is None
            or source_row_optimizer is None
            or source_row_behavior is None
            or any(
                metadata.get(field) != source_row_metadata.get(field)
                for field in immutable_source_metadata_fields
            )
            or dict(optimizer) != source_row_optimizer
            or row_behavior != source_row_behavior
        ):
            raise RecurrentCounterfactualCollectionError(
                "bundle horizons do not share one exact source identity"
            )
    if config.multi_tape_enabled:
        if (
            bundle.aggregate_rows is None
            or len(bundle.aggregate_rows) != len(bundle.horizons)
            or bundle.multi_tape_compute is None
        ):
            raise RecurrentCounterfactualCollectionError(
                "multi-tape bundle is missing aggregate evidence"
            )
        source_tape_projection: tuple[dict[str, object], ...] | None = None
        for horizon, aggregate_row in zip(
            bundle.horizons,
            bundle.aggregate_rows,
            strict=True,
        ):
            try:
                validate_recurrent_counterfactual_aggregate_row(aggregate_row)
            except RecurrentCounterfactualBranchError as error:
                raise RecurrentCounterfactualCollectionError(
                    "bundle contains an invalid aggregate row"
                ) from error
            tape_projection = _validate_aggregate_source_linkage(
                aggregate_row,
                bundle=bundle,
                source_model_digest=source_model_digest,
                artifact_digest=artifact_digest,
            )
            if source_tape_projection is None:
                source_tape_projection = tape_projection
            elif tape_projection != source_tape_projection:
                raise RecurrentCounterfactualCollectionError(
                    "aggregate horizons do not share one continuation-tape identity"
                )
            target = _mapping(
                aggregate_row.get("target"),
                field="aggregate row target",
            )
            if (
                target.get("kind") != "relative_horizon"
                or target.get("horizon_ticks") != horizon
                or target.get("branch_tick")
                != bundle.selection.get("selected_branch_tick")
                or target.get("absolute_terminal_target_world_tick")
                != config.terminal_target_world_tick
            ):
                raise RecurrentCounterfactualCollectionError(
                    "aggregate relative horizon target drifted"
                )
            tape_contract = _mapping(
                aggregate_row.get("tape_contract"),
                field="aggregate tape contract",
            )
            source_identity = _mapping(
                aggregate_row.get("source_identity"),
                field="aggregate source identity",
            )
            aggregate = _mapping(
                aggregate_row.get("aggregate"),
                field="aggregate statistics",
            )
            if (
                tape_contract.get("tape_count") != config.continuation_tape_count
                or aggregate.get("uncertainty_penalty") != config.uncertainty_penalty
            ):
                raise RecurrentCounterfactualCollectionError(
                    "aggregate collection config drifted"
                )
            if (
                source_identity.get("source_branch_row_exact_digest")
                != bundle.rows[0].get("exact_digest")
                or source_identity.get("source_environment_seed")
                != bundle.task.environment_seed
                or source_identity.get("source_policy_sampling_seed")
                != bundle.task.source_policy_sampling_seed
                or source_identity.get("source_branch_selection_seed")
                != bundle.task.branch_selection_seed
                or source_identity.get("source_model_state_sha256")
                != source_model_digest
                or source_identity.get("source_artifact_digest") != artifact_digest
            ):
                raise RecurrentCounterfactualCollectionError(
                    "aggregate source identity drifted from its source row or task"
                )
            if stable_payload_digest(
                aggregate_row.get("trainable_public_context")
            ) != stable_payload_digest(bundle.rows[0]["trainable_public_context"]):
                raise RecurrentCounterfactualCollectionError(
                    "aggregate public branch context drifted"
                )
        if config.terminal_target_world_tick is None:
            if bundle.terminal_target is not None:
                raise RecurrentCounterfactualCollectionError(
                    "bundle has an unrequested absolute terminal target"
                )
        else:
            if bundle.terminal_target is None:
                raise RecurrentCounterfactualCollectionError(
                    "bundle is missing its absolute terminal target"
                )
            try:
                validate_recurrent_counterfactual_aggregate_row(bundle.terminal_target)
            except RecurrentCounterfactualBranchError as error:
                raise RecurrentCounterfactualCollectionError(
                    "bundle terminal aggregate is invalid"
                ) from error
            terminal_tape_projection = _validate_aggregate_source_linkage(
                bundle.terminal_target,
                bundle=bundle,
                source_model_digest=source_model_digest,
                artifact_digest=artifact_digest,
            )
            if (
                source_tape_projection is None
                or terminal_tape_projection != source_tape_projection
            ):
                raise RecurrentCounterfactualCollectionError(
                    "terminal aggregate continuation-tape identity drifted"
                )
            terminal_target = _mapping(
                bundle.terminal_target.get("target"),
                field="terminal aggregate target",
            )
            terminal_source_identity = _mapping(
                bundle.terminal_target.get("source_identity"),
                field="terminal aggregate source identity",
            )
            if (
                terminal_target.get("kind") != "absolute_terminal_world_tick"
                or terminal_target.get("target_world_tick")
                != config.terminal_target_world_tick
                or terminal_target.get("branch_tick")
                != bundle.selection.get("selected_branch_tick")
            ):
                raise RecurrentCounterfactualCollectionError(
                    "bundle absolute terminal target drifted"
                )
            if (
                terminal_source_identity.get("source_branch_row_exact_digest")
                != bundle.rows[0].get("exact_digest")
                or terminal_source_identity.get("source_environment_seed")
                != bundle.task.environment_seed
                or terminal_source_identity.get("source_policy_sampling_seed")
                != bundle.task.source_policy_sampling_seed
                or terminal_source_identity.get("source_branch_selection_seed")
                != bundle.task.branch_selection_seed
                or terminal_source_identity.get("source_model_state_sha256")
                != source_model_digest
                or terminal_source_identity.get("source_artifact_digest")
                != artifact_digest
            ):
                raise RecurrentCounterfactualCollectionError(
                    "terminal aggregate source identity drifted from its source "
                    "row or task"
                )
    elif (
        bundle.aggregate_rows is not None
        or bundle.terminal_target is not None
        or bundle.multi_tape_compute is not None
    ):
        raise RecurrentCounterfactualCollectionError(
            "legacy bundle cannot carry multi-tape evidence"
        )
    _validate_selection(bundle)
    _validate_prefix_proof(bundle)
    _validate_compute(bundle)
    fields = _bundle_payload(bundle)
    observed_digest = fields.pop("exact_digest")
    if observed_digest != stable_payload_digest(fields):
        raise RecurrentCounterfactualCollectionError("bundle exact digest mismatch")


def _validate_aggregate_source_linkage(
    aggregate_row: Mapping[str, object],
    *,
    bundle: RecurrentCounterfactualCollectionBundle,
    source_model_digest: str,
    artifact_digest: str,
) -> tuple[dict[str, object], ...]:
    source_row = bundle.rows[0]
    source_metadata = _mapping(
        source_row.get("metadata"),
        field="source row metadata",
    )
    source_context = _mapping(
        source_row.get("trainable_public_context"),
        field="source row public context",
    )
    source_optimizer = _mapping(
        source_row.get("optimizer_context"),
        field="source row optimizer context",
    )
    source_labels = _mapping(source_row.get("labels"), field="source row labels")
    aggregate_context = _mapping(
        aggregate_row.get("trainable_public_context"),
        field="aggregate public context",
    )
    aggregate_optimizer = _mapping(
        aggregate_row.get("optimizer_context"),
        field="aggregate optimizer context",
    )
    source_identity = _mapping(
        aggregate_row.get("source_identity"),
        field="aggregate source identity",
    )
    source_behavior = _mapping(
        aggregate_row.get("source_behavior"),
        field="aggregate source behavior",
    )
    tape_contract = _mapping(
        aggregate_row.get("tape_contract"),
        field="aggregate tape contract",
    )
    expected_tape_identity = f"{bundle.task.task_id}:continuation-tapes"
    if tape_contract.get("tape_identity") != expected_tape_identity:
        raise RecurrentCounterfactualCollectionError(
            "aggregate tape identity drifted from its collection task id"
        )
    if dict(aggregate_context) != dict(source_context):
        raise RecurrentCounterfactualCollectionError(
            "aggregate public branch context drifted from its exact source row"
        )
    if dict(aggregate_optimizer) != dict(source_optimizer):
        raise RecurrentCounterfactualCollectionError(
            "aggregate optimizer context drifted from its exact source row"
        )
    expected_source_behavior = {
        "source_requested_action": source_labels.get("source_requested_action"),
        "current_public_action_mask": copy.deepcopy(
            source_context.get("current_public_action_mask")
        ),
        "source_behavior_distribution": copy.deepcopy(
            source_labels.get("source_behavior_distribution")
        ),
    }
    if dict(source_behavior) != expected_source_behavior:
        raise RecurrentCounterfactualCollectionError(
            "aggregate source behavior drifted from its exact source row"
        )
    expected_checkpoint_identity = {
        "source_branch_row_exact_digest": source_row.get("exact_digest"),
        "seed_role": source_metadata.get("seed_role"),
        "environment_seed": source_metadata.get("environment_seed"),
        "branch_selection_seed": bundle.task.branch_selection_seed,
        "scenario": source_metadata.get("scenario"),
        "branch_tick": source_metadata.get("branch_tick"),
        "focal_agent_id": source_metadata.get("focal_agent_id"),
        "source_record_digest": source_metadata.get("source_record_digest"),
        "source_observation_digest": source_metadata.get(
            "source_observation_digest"
        ),
        "source_action_mask_digest": source_metadata.get(
            "source_action_mask_digest"
        ),
        "source_public_history_prefix_sha256": source_metadata.get(
            "source_public_history_prefix_sha256"
        ),
        "source_pre_boundary_environment_rng_state_sha256": source_metadata.get(
            "source_pre_boundary_environment_rng_state_sha256"
        ),
        "source_pre_boundary_policy_sampling_state_sha256": source_metadata.get(
            "source_pre_boundary_policy_sampling_state_sha256"
        ),
        "source_boundary_identity_sha256": source_metadata.get(
            "source_boundary_identity_sha256"
        ),
    }
    if (
        source_identity.get("source_branch_row_exact_digest")
        != source_row.get("exact_digest")
        or source_identity.get("source_environment_seed")
        != bundle.task.environment_seed
        or source_identity.get("source_policy_sampling_seed")
        != bundle.task.source_policy_sampling_seed
        or source_identity.get("source_branch_selection_seed")
        != bundle.task.branch_selection_seed
        or source_identity.get("source_model_state_sha256") != source_model_digest
        or source_identity.get("source_artifact_digest") != artifact_digest
        or source_identity.get("source_checkpoint_identity_sha256")
        != stable_payload_digest(expected_checkpoint_identity)
        or source_identity.get(
            "source_pre_boundary_environment_rng_state_sha256"
        )
        != source_metadata.get(
            "source_pre_boundary_environment_rng_state_sha256"
        )
        or source_identity.get("source_pre_boundary_policy_sampling_state_sha256")
        != source_metadata.get(
            "source_pre_boundary_policy_sampling_state_sha256"
        )
        or source_identity.get("source_boundary_identity_sha256")
        != source_metadata.get("source_boundary_identity_sha256")
    ):
        raise RecurrentCounterfactualCollectionError(
            "aggregate source identity drifted from its boundary-bound source row "
            "or collection task"
        )
    raw_tapes = aggregate_row.get("tape_provenance")
    if not isinstance(raw_tapes, list):
        raise RecurrentCounterfactualCollectionError(
            "aggregate tape provenance must be a list"
        )
    projection_fields = (
        "tape_index",
        "environment_sampling_identity",
        "environment_sampling_seed",
        "policy_sampling_identity",
        "policy_sampling_seed",
        "continuation_rng_retape_boundary",
        "pre_boundary_environment_rng_state_sha256",
        "pre_boundary_policy_sampling_state_sha256",
        "post_boundary_environment_rng_state_sha256",
        "post_boundary_policy_sampling_state_sha256",
        "boundary_reached",
        "source_natural_action_match",
        "fixed_source_prefix_verified",
    )
    return tuple(
        {
            field: _mapping(tape, field="aggregate tape provenance").get(field)
            for field in projection_fields
        }
        for tape in raw_tapes
    )


def _validate_selection(bundle: RecurrentCounterfactualCollectionBundle) -> None:
    selection = bundle.selection
    expected_fields = set(_SELECTION_BASE_FIELDS)
    if bundle.aggregate_rows is not None:
        expected_fields.update(_SELECTION_MULTI_TAPE_FIELDS)
    if bundle.task.branch_tick_stratum_index is not None:
        expected_fields.update(_SELECTION_ROTATION_FIELDS)
    if set(selection) != expected_fields:
        raise RecurrentCounterfactualCollectionError(
            "bundle selection field set drifted"
        )
    requested_ticks = selection.get("requested_branch_ticks")
    if (
        not isinstance(requested_ticks, list)
        or tuple(
            _nonnegative_int(tick, field="requested branch tick")
            for tick in requested_ticks
        )
        != bundle.task.branch_tick_candidates
    ):
        raise RecurrentCounterfactualCollectionError(
            "bundle branch tick request drifted"
        )
    selected_tick = _nonnegative_int(
        selection.get("selected_branch_tick"),
        field="selected branch tick",
    )
    if selected_tick not in bundle.task.branch_tick_candidates:
        raise RecurrentCounterfactualCollectionError(
            "selected branch tick is outside the task candidates"
        )
    fallback_index = selection.get("fallback_index")
    if (
        isinstance(fallback_index, bool)
        or not isinstance(fallback_index, int)
        or fallback_index < 0
        or fallback_index >= len(bundle.task.branch_tick_candidates)
        or bundle.task.branch_tick_candidates[fallback_index] != selected_tick
    ):
        raise RecurrentCounterfactualCollectionError("fallback index drifted")
    candidate_count = _positive_int(
        selection.get("eligible_candidate_count"),
        field="eligible candidate count",
    )
    selected_index = _nonnegative_int(
        selection.get("selected_candidate_index"),
        field="selected candidate index",
    )
    if selected_index >= candidate_count:
        raise RecurrentCounterfactualCollectionError(
            "selected candidate index is out of bounds"
        )
    selection_seed = _sampling_seed(
        bundle.task.branch_selection_seed,
        field="branch selection seed",
    )
    selection_rng = random.Random(selection_seed)
    if bundle.aggregate_rows is None:
        if selection.get("selection_policy") != (
            "uniform_seeded_over_current_tick_multi_action_learner_decisions"
        ):
            raise RecurrentCounterfactualCollectionError(
                "legacy selection policy drifted"
            )
        unexpected_scale_fields = {
            "eligible_branch_ticks",
            "eligible_branch_tick_count",
            "selected_eligible_tick_index",
            "requested_branch_tick_stratum_index",
            "branch_tick_strata_traversal",
            "selected_branch_tick_stratum_index",
        } & set(selection)
        if unexpected_scale_fields:
            raise RecurrentCounterfactualCollectionError(
                "legacy selection contains scale-only tick strata"
            )
    else:
        raw_eligible_ticks = selection.get("eligible_branch_ticks")
        if (
            not isinstance(raw_eligible_ticks, list)
            or not raw_eligible_ticks
        ):
            raise RecurrentCounterfactualCollectionError(
                "eligible branch tick strata drifted"
            )
        eligible_ticks = [
            _nonnegative_int(tick, field="eligible branch tick")
            for tick in raw_eligible_ticks
        ]
        eligible_tick_count = _positive_int(
            selection.get("eligible_branch_tick_count"),
            field="eligible branch tick count",
        )
        selected_eligible_tick_index = _nonnegative_int(
            selection.get("selected_eligible_tick_index"),
            field="selected eligible tick index",
        )
        if (
            any(
                tick not in bundle.task.branch_tick_candidates
                for tick in eligible_ticks
            )
            or len(set(eligible_ticks)) != len(eligible_ticks)
            or eligible_tick_count != len(eligible_ticks)
            or selected_eligible_tick_index >= eligible_tick_count
        ):
            raise RecurrentCounterfactualCollectionError(
                "eligible branch tick strata drifted"
            )
        stratum_index = bundle.task.branch_tick_stratum_index
        if stratum_index is None:
            if selection.get("selection_policy") != (
                "stratified_uniform_seeded_tick_then_uniform_current_tick_"
                "learner_decision"
            ):
                raise RecurrentCounterfactualCollectionError(
                    "scale selection policy drifted"
                )
            expected_tick_index = selection_rng.randrange(len(eligible_ticks))
            if (
                selected_eligible_tick_index != expected_tick_index
                or eligible_ticks[expected_tick_index] != selected_tick
            ):
                raise RecurrentCounterfactualCollectionError(
                    "selected branch tick is not the seeded stratified draw"
                )
        else:
            if selection.get("selection_policy") != (
                "deterministic_stratified_tick_rotation_with_eligible_fallback_"
                "then_uniform_current_tick_learner_decision"
            ):
                raise RecurrentCounterfactualCollectionError(
                    "deterministic stratum selection policy drifted"
                )
            candidates = bundle.task.branch_tick_candidates
            traversal = tuple(
                candidates[(stratum_index + offset) % len(candidates)]
                for offset in range(len(candidates))
            )
            expected_tick = next(tick for tick in traversal if tick in eligible_ticks)
            requested_stratum_index = _nonnegative_int(
                selection.get("requested_branch_tick_stratum_index"),
                field="requested branch tick stratum index",
            )
            raw_traversal = selection.get("branch_tick_strata_traversal")
            if not isinstance(raw_traversal, list):
                raise RecurrentCounterfactualCollectionError(
                    "deterministic branch-tick stratum rotation drifted"
                )
            parsed_traversal = tuple(
                _nonnegative_int(tick, field="branch tick strata traversal")
                for tick in raw_traversal
            )
            selected_stratum_index = _nonnegative_int(
                selection.get("selected_branch_tick_stratum_index"),
                field="selected branch tick stratum index",
            )
            if (
                requested_stratum_index != stratum_index
                or parsed_traversal != traversal
                or selected_stratum_index != candidates.index(expected_tick)
                or selected_eligible_tick_index != eligible_ticks.index(expected_tick)
                or selected_tick != expected_tick
            ):
                raise RecurrentCounterfactualCollectionError(
                    "deterministic branch-tick stratum rotation drifted"
                )
    expected_selected_index = selection_rng.randrange(candidate_count)
    if selected_index != expected_selected_index:
        raise RecurrentCounterfactualCollectionError(
            "selected candidate index is not the seeded uniform draw"
        )
    _sha256(selection.get("candidate_list_digest"), field="candidate list digest")
    if selection.get("outcome_or_future_data_used") is not False:
        raise RecurrentCounterfactualCollectionError(
            "branch selection used forbidden outcome/future data"
        )
    first_metadata = _mapping(bundle.rows[0].get("metadata"), field="row metadata")
    selected_focal_agent_id = _positive_int(
        selection.get("selected_focal_agent_id"),
        field="selected focal agent id",
    )
    if selected_focal_agent_id != first_metadata.get(
        "focal_agent_id"
    ) or selected_tick != first_metadata.get("branch_tick"):
        raise RecurrentCounterfactualCollectionError(
            "selection and row focal state identities disagree"
        )


def _validate_prefix_proof(bundle: RecurrentCounterfactualCollectionBundle) -> None:
    proof = bundle.prefix_proof
    if set(proof) != _PREFIX_PROOF_FIELDS:
        raise RecurrentCounterfactualCollectionError(
            "nested prefix proof field set drifted"
        )
    raw_horizons = proof.get("horizons")
    if not isinstance(raw_horizons, list):
        raise RecurrentCounterfactualCollectionError(
            "nested prefix proof horizons drifted"
        )
    parsed_horizons = tuple(
        _positive_int(horizon, field="prefix proof horizon")
        for horizon in raw_horizons
    )
    stream_count = _positive_int(
        proof.get("stream_count"),
        field="prefix proof stream count",
    )
    if (
        proof.get("schema_version") != "mind_v3_recurrent_nested_prefix_proof_v1"
        or parsed_horizons != bundle.horizons
        or proof.get("all_shorter_streams_are_exact_max_stream_prefixes") is not True
        or proof.get("raw_private_world_state_serialized") is not False
    ):
        raise RecurrentCounterfactualCollectionError(
            "nested prefix proof contract drifted"
        )
    streams = _mapping(proof.get("streams"), field="prefix proof streams")
    expected_streams = {"baseline"} | {
        f"{kind}:{action}"
        for kind in ("action", "replay")
        for action in bundle.valid_actions
    }
    if set(streams) != expected_streams or stream_count != len(streams):
        raise RecurrentCounterfactualCollectionError(
            "nested prefix proof stream set drifted"
        )
    for stream_name, raw_stream in streams.items():
        stream = _mapping(raw_stream, field="prefix stream")
        if set(stream) != _PREFIX_STREAM_FIELDS:
            raise RecurrentCounterfactualCollectionError(
                "prefix stream field set drifted"
            )
        if stream.get("exact_prefix") is not True:
            raise RecurrentCounterfactualCollectionError(
                "prefix stream lacks exact-prefix proof"
            )
        by_horizon = _mapping(stream.get("horizons"), field="stream horizons")
        if set(by_horizon) != {str(horizon) for horizon in bundle.horizons}:
            raise RecurrentCounterfactualCollectionError(
                "prefix stream horizon set drifted"
            )
        maximum = _mapping(
            by_horizon[str(bundle.horizons[-1])],
            field="maximum prefix proof",
        )
        previous_executed_tick_count = 0
        previous_sequence_length = 0
        for horizon_index, horizon in enumerate(bundle.horizons):
            current = _mapping(
                by_horizon[str(horizon)],
                field="horizon prefix proof",
            )
            if set(current) != _PREFIX_HORIZON_FIELDS:
                raise RecurrentCounterfactualCollectionError(
                    "horizon prefix proof field set drifted"
                )
            _sha256(
                current.get("behavior_digest"),
                field="prefix behavior digest",
            )
            _sha256(
                current.get("evidence_digest"),
                field="prefix evidence digest",
            )
            executed_tick_count = _positive_int(
                current.get("executed_tick_count"),
                field="prefix executed tick count",
            )
            if (
                executed_tick_count > horizon
                or executed_tick_count < previous_executed_tick_count
            ):
                raise RecurrentCounterfactualCollectionError(
                    "prefix executed tick count drifted from its bounded "
                    "nondecreasing horizon contract"
                )
            previous_executed_tick_count = executed_tick_count
            for sequence_field in (
                "causal_record_digest_sequence",
                "policy_record_digest_sequence",
                "diagnostics_digest_sequence",
            ):
                sequence = current.get(sequence_field)
                maximum_sequence = maximum.get(sequence_field)
                if not isinstance(sequence, list) or not isinstance(
                    maximum_sequence,
                    list,
                ):
                    raise RecurrentCounterfactualCollectionError(
                        "prefix digest sequences must be lists"
                    )
                if any(
                    _sha256(item, field="prefix record digest") != item
                    for item in sequence
                ):
                    raise RecurrentCounterfactualCollectionError(
                        "prefix record digest is malformed"
                    )
                if sequence != maximum_sequence[: len(sequence)]:
                    raise RecurrentCounterfactualCollectionError(
                        "shorter horizon is not an exact maximum prefix"
                    )
            sequence_lengths = {
                len(current[sequence_field])
                for sequence_field in (
                    "causal_record_digest_sequence",
                    "policy_record_digest_sequence",
                    "diagnostics_digest_sequence",
                )
            }
            if len(sequence_lengths) != 1:
                raise RecurrentCounterfactualCollectionError(
                    "causal, policy, and diagnostic prefix lengths disagree"
                )
            sequence_length = next(iter(sequence_lengths))
            if (
                sequence_length <= 0
                or sequence_length < executed_tick_count
                or sequence_length < previous_sequence_length
            ):
                raise RecurrentCounterfactualCollectionError(
                    "prefix record sequence length violates its positive "
                    "nondecreasing executed-tick contract"
                )
            previous_sequence_length = sequence_length
            row = bundle.rows[horizon_index]
            labels = _mapping(row.get("labels"), field="row labels")
            if stream_name == "baseline":
                expected = _mapping(labels.get("baseline"), field="baseline")
                if current.get("behavior_digest") != expected.get(
                    "behavior_digest"
                ) or current.get("evidence_digest") != expected.get("evidence_digest"):
                    raise RecurrentCounterfactualCollectionError(
                        "baseline prefix proof digest disagrees with row"
                    )
            elif stream_name.startswith("action:"):
                action = stream_name.split(":", 1)[1]
                outcome = next(
                    _mapping(item, field="action outcome")
                    for item in labels.get("action_outcomes", [])
                    if _mapping(item, field="action outcome").get("action") == action
                )
                if current.get("behavior_digest") != outcome.get(
                    "behavior_digest"
                ) or current.get("evidence_digest") != outcome.get("evidence_digest"):
                    raise RecurrentCounterfactualCollectionError(
                        "action prefix proof digest disagrees with row"
                    )
            elif stream_name.startswith("replay:"):
                action = stream_name.split(":", 1)[1]
                outcome = next(
                    _mapping(item, field="action outcome")
                    for item in labels.get("action_outcomes", [])
                    if _mapping(item, field="action outcome").get("action") == action
                )
                if current.get("evidence_digest") != outcome.get(
                    "replay_evidence_digest"
                ):
                    raise RecurrentCounterfactualCollectionError(
                        "replay prefix proof digest disagrees with row"
                    )
                action_stream = _mapping(
                    streams[f"action:{action}"],
                    field="paired action prefix stream",
                )
                action_horizons = _mapping(
                    action_stream.get("horizons"),
                    field="paired action prefix horizons",
                )
                paired_action = _mapping(
                    action_horizons[str(horizon)],
                    field="paired action horizon proof",
                )
                if current.get("behavior_digest") != paired_action.get(
                    "behavior_digest"
                ):
                    raise RecurrentCounterfactualCollectionError(
                        "replay behavior digest disagrees with action behavior"
                    )


def _validate_compute(bundle: RecurrentCounterfactualCollectionBundle) -> None:
    compute = bundle.compute
    if set(compute) != _COMPUTE_FIELDS:
        raise RecurrentCounterfactualCollectionError(
            "compute diagnostic field set drifted"
        )
    action_count = len(bundle.valid_actions)
    horizon_count = len(bundle.horizons)
    continuation_count = 1 + 2 * action_count
    maximum_horizon = bundle.horizons[-1]
    independent_budget = continuation_count * sum(bundle.horizons)
    maximum_budget = continuation_count * maximum_horizon
    expected = {
        "selected_continuation_checkpoint_count": 1,
        "valid_action_count": action_count,
        "horizon_count": horizon_count,
        "action_horizon_target_count": action_count * horizon_count,
        "max_horizon_continuation_count": continuation_count,
        "exact_replay_continuation_count": action_count,
        "maximum_horizon_ticks": maximum_horizon,
        "maximum_continuation_tick_budget": maximum_budget,
        "independent_horizon_tick_budget": independent_budget,
        "nested_tick_budget_saved": independent_budget - maximum_budget,
    }
    for field, expected_value in expected.items():
        actual_value = _nonnegative_int(
            compute.get(field),
            field=f"compute diagnostic {field}",
        )
        if actual_value != expected_value:
            raise RecurrentCounterfactualCollectionError(
                f"compute diagnostic {field!r} drifted"
            )
    actual = _nonnegative_int(
        compute.get("actual_continuation_tick_count"),
        field="actual continuation tick count",
    )
    proof = _mapping(bundle.prefix_proof, field="compute prefix proof")
    streams = _mapping(proof.get("streams"), field="compute prefix streams")
    expected_actual = 0
    for raw_stream in streams.values():
        stream = _mapping(raw_stream, field="compute prefix stream")
        by_horizon = _mapping(
            stream.get("horizons"),
            field="compute prefix horizons",
        )
        maximum = _mapping(
            by_horizon.get(str(maximum_horizon)),
            field="compute maximum-horizon prefix",
        )
        expected_actual += _positive_int(
            maximum.get("executed_tick_count"),
            field="compute maximum-horizon executed tick count",
        )
    if actual != expected_actual:
        raise RecurrentCounterfactualCollectionError(
            "actual continuation ticks drifted from prefix evidence"
        )
    source_ticks = _positive_int(
        compute.get("source_tick_executions"),
        field="source ticks",
    )
    selected_tick = _nonnegative_int(
        bundle.selection.get("selected_branch_tick"),
        field="selected branch tick",
    )
    maximum_candidate_tick = max(bundle.task.branch_tick_candidates)
    if not selected_tick + 1 <= source_ticks <= maximum_candidate_tick + 1:
        raise RecurrentCounterfactualCollectionError(
            "source ticks drifted from the selected/candidate branch interval"
        )
    if bundle.multi_tape_compute is not None:
        if set(bundle.multi_tape_compute) != _MULTI_TAPE_COMPUTE_FIELDS:
            raise RecurrentCounterfactualCollectionError(
                "multi-tape compute diagnostic field set drifted"
            )
        if bundle.aggregate_rows is None:
            raise RecurrentCounterfactualCollectionError(
                "multi-tape compute lacks aggregate rows"
            )
        tape_contract = _mapping(
            bundle.aggregate_rows[0].get("tape_contract"),
            field="aggregate tape contract",
        )
        tape_count = _positive_int(
            tape_contract.get("tape_count"),
            field="continuation tape count",
        )
        continuation_count_per_tape = 2 + 2 * action_count
        target_horizons = list(bundle.horizons)
        if bundle.terminal_target is not None:
            terminal = _mapping(
                bundle.terminal_target.get("target"),
                field="terminal target",
            )
            target_horizons.append(
                _positive_int(
                    terminal.get("horizon_ticks"),
                    field="terminal horizon",
                )
            )
        multi_maximum_horizon = max(target_horizons)
        multi_expected = {
            "continuation_tape_count": tape_count,
            "continuation_count_per_tape": continuation_count_per_tape,
            "max_horizon_continuation_count": (
                tape_count * continuation_count_per_tape
            ),
            "exact_replay_continuation_count": (tape_count * (1 + action_count)),
            "maximum_horizon_ticks": multi_maximum_horizon,
            "maximum_continuation_tick_budget": (
                tape_count * continuation_count_per_tape * multi_maximum_horizon
            ),
        }
        for field, expected_value in multi_expected.items():
            actual_value = _nonnegative_int(
                bundle.multi_tape_compute.get(field),
                field=f"multi-tape compute diagnostic {field}",
            )
            if actual_value != expected_value:
                raise RecurrentCounterfactualCollectionError(
                    f"multi-tape compute diagnostic {field!r} drifted"
                )
        multi_actual = _positive_int(
            bundle.multi_tape_compute.get("actual_continuation_tick_count"),
            field="multi-tape actual continuation ticks",
        )
        if (
            multi_actual
            < multi_expected["max_horizon_continuation_count"]
            or multi_actual > multi_expected["maximum_continuation_tick_budget"]
        ):
            raise RecurrentCounterfactualCollectionError(
                "multi-tape actual continuation ticks are outside executable bounds"
            )


def _bundle_payload(
    bundle: RecurrentCounterfactualCollectionBundle,
) -> dict[str, object]:
    payload = {
        "task": recurrent_counterfactual_collection_task_payload(bundle.task),
        "rows": copy.deepcopy(list(bundle.rows)),
        "selection": copy.deepcopy(bundle.selection),
        "prefix_proof": copy.deepcopy(bundle.prefix_proof),
        "compute": copy.deepcopy(bundle.compute),
        "source_model_state_sha256": bundle.source_model_state_sha256,
        "source_artifact_digest": bundle.source_artifact_digest,
        "valid_actions": list(bundle.valid_actions),
        "horizons": list(bundle.horizons),
        "exact_digest": bundle.exact_digest,
    }
    if bundle.aggregate_rows is not None:
        payload.update(
            {
                "aggregate_rows": copy.deepcopy(list(bundle.aggregate_rows)),
                "terminal_target": copy.deepcopy(bundle.terminal_target),
                "multi_tape_compute": copy.deepcopy(bundle.multi_tape_compute),
            }
        )
    return payload


def _aggregate_compute(
    bundles: Sequence[RecurrentCounterfactualCollectionBundle],
) -> dict[str, int]:
    fields = (
        "source_tick_executions",
        "selected_continuation_checkpoint_count",
        "action_horizon_target_count",
        "max_horizon_continuation_count",
        "exact_replay_continuation_count",
        "actual_continuation_tick_count",
        "maximum_continuation_tick_budget",
        "independent_horizon_tick_budget",
        "nested_tick_budget_saved",
    )
    aggregate = {
        "bundle_count": len(bundles),
        **{
            field: sum(int(bundle.compute[field]) for bundle in bundles)
            for field in fields
        },
    }
    multi_tape_bundles = [
        bundle for bundle in bundles if bundle.multi_tape_compute is not None
    ]
    if multi_tape_bundles:
        if len(multi_tape_bundles) != len(bundles):
            raise RecurrentCounterfactualCollectionError(
                "collection cannot mix legacy and multi-tape bundles"
            )
        aggregate.update(
            {
                "multi_tape_continuation_tape_count": sum(
                    int(bundle.multi_tape_compute["continuation_tape_count"])
                    for bundle in multi_tape_bundles
                ),
                "multi_tape_max_horizon_continuation_count": sum(
                    int(bundle.multi_tape_compute["max_horizon_continuation_count"])
                    for bundle in multi_tape_bundles
                ),
                "multi_tape_exact_replay_continuation_count": sum(
                    int(bundle.multi_tape_compute["exact_replay_continuation_count"])
                    for bundle in multi_tape_bundles
                ),
                "multi_tape_actual_continuation_tick_count": sum(
                    int(bundle.multi_tape_compute["actual_continuation_tick_count"])
                    for bundle in multi_tape_bundles
                ),
                "multi_tape_maximum_continuation_tick_budget": sum(
                    int(bundle.multi_tape_compute["maximum_continuation_tick_budget"])
                    for bundle in multi_tape_bundles
                ),
            }
        )
    return aggregate


def _validated_tasks(
    tasks: Sequence[RecurrentCounterfactualCollectionTask],
) -> tuple[RecurrentCounterfactualCollectionTask, ...]:
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence) or not tasks:
        raise RecurrentCounterfactualCollectionError(
            "tasks must be a non-empty ordered sequence"
        )
    parsed = tuple(tasks)
    if any(
        not isinstance(task, RecurrentCounterfactualCollectionTask) for task in parsed
    ):
        raise RecurrentCounterfactualCollectionError(
            "every task must be a RecurrentCounterfactualCollectionTask"
        )
    if len({task.task_id for task in parsed}) != len(parsed):
        raise RecurrentCounterfactualCollectionError("task ids must be unique")
    source_seeds = [task.source_policy_sampling_seed for task in parsed]
    selection_seeds = [task.branch_selection_seed for task in parsed]
    if len(set(source_seeds)) != len(source_seeds):
        raise RecurrentCounterfactualCollectionError(
            "source policy sampling seeds must be unique across tasks"
        )
    if len(set(selection_seeds)) != len(selection_seeds):
        raise RecurrentCounterfactualCollectionError(
            "branch selection seeds must be unique across tasks"
        )
    if set(source_seeds) & set(selection_seeds):
        raise RecurrentCounterfactualCollectionError(
            "source and branch-selection seed sets must be disjoint"
        )
    return parsed


def _validate_task_selection_modes(
    tasks: Sequence[RecurrentCounterfactualCollectionTask],
    *,
    config: RecurrentCounterfactualCollectionConfig,
) -> None:
    for task in tasks:
        stratum_index = task.branch_tick_stratum_index
        if stratum_index is not None and not config.multi_tape_enabled:
            raise RecurrentCounterfactualCollectionError(
                "explicit branch-tick strata require the versioned multi-tape "
                "collection contract"
            )
        if (
            config.multi_tape_enabled
            and task.seed_role in RECURRENT_COUNTERFACTUAL_SCALE_SEED_ROLES
            and len(task.branch_tick_candidates) > 1
            and stratum_index is None
        ):
            raise RecurrentCounterfactualCollectionError(
                "scale multi-tape tasks with multiple branch ticks require an "
                "explicit deterministic stratum index"
            )


def _branch_tick_candidates(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise RecurrentCounterfactualCollectionError(
            "branch_tick_candidates must be a non-empty tuple"
        )
    parsed = tuple(
        _nonnegative_int(item, field="branch tick candidate") for item in value
    )
    if len(set(parsed)) != len(parsed):
        raise RecurrentCounterfactualCollectionError(
            "branch tick candidates must be unique"
        )
    return parsed


def _nested_horizons(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise RecurrentCounterfactualCollectionError("horizons must be a tuple")
    parsed = tuple(_positive_int(item, field="horizon") for item in value)
    if len(parsed) < 2 or tuple(sorted(set(parsed))) != parsed:
        raise RecurrentCounterfactualCollectionError(
            "horizons must contain at least two unique increasing values"
        )
    return parsed


def _worker_count(value: object) -> int:
    parsed = _positive_int(value, field="workers")
    if parsed > MAX_RECURRENT_COUNTERFACTUAL_COLLECTION_WORKERS:
        raise RecurrentCounterfactualCollectionError(
            "workers exceeds the collection worker cap"
        )
    return parsed


def _resolved_worker_count(
    *,
    workers: int,
    task_count: int,
    config: RecurrentCounterfactualCollectionConfig,
) -> int:
    resolved_workers = _worker_count(workers)
    resolved_task_count = _positive_int(task_count, field="task_count")
    work_item_count = resolved_task_count
    if _tape_parallelism_used(
        workers=resolved_workers,
        task_count=resolved_task_count,
        config=config,
    ):
        work_item_count *= _maximum_tape_chunks_per_task(
            config.continuation_tape_count
        )
    return min(resolved_workers, work_item_count)


def _validate_tape_parallel_execution_compute(
    value: object,
    *,
    tasks: tuple[RecurrentCounterfactualCollectionTask, ...],
    config: RecurrentCounterfactualCollectionConfig,
    bundles: tuple[RecurrentCounterfactualCollectionBundle, ...],
    workers: int,
) -> None:
    compute = _mapping(value, field="tape-parallel execution compute")
    if set(compute) != _TAPE_PARALLEL_EXECUTION_COMPUTE_FIELDS:
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel execution compute field set drifted"
        )
    parsed = {
        field: _nonnegative_int(
            compute.get(field),
            field=f"tape-parallel execution compute {field}",
        )
        for field in _TAPE_PARALLEL_EXECUTION_COMPUTE_FIELDS
    }
    work_items = _tape_chunk_work_items(
        tasks,
        tape_count=config.continuation_tape_count,
        workers=workers,
    )
    chunks_per_bundle = [
        sum(int(work_item.task_index == task_index) for work_item in work_items)
        for task_index in range(len(tasks))
    ]
    expected_canonical_source_ticks = sum(
        int(bundle.compute["source_tick_executions"]) for bundle in bundles
    )
    expected_canonical_legacy_ticks = sum(
        int(bundle.compute["actual_continuation_tick_count"])
        for bundle in bundles
    )
    expected_multi_tape_ticks = sum(
        int(
            _mapping(
                bundle.multi_tape_compute,
                field="bundle multi-tape compute",
            )["actual_continuation_tick_count"]
        )
        for bundle in bundles
    )
    expected = {
        "tape_chunk_work_item_count": len(work_items),
        "split_bundle_count": sum(int(count > 1) for count in chunks_per_bundle),
        "maximum_chunks_per_bundle": max(chunks_per_bundle),
        "canonical_source_tick_executions": expected_canonical_source_ticks,
        "canonical_legacy_actual_continuation_tick_count": (
            expected_canonical_legacy_ticks
        ),
        "multi_tape_actual_continuation_tick_count": expected_multi_tape_ticks,
    }
    if any(
        parsed[field] != expected_value
        for field, expected_value in expected.items()
    ):
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel execution compute does not match canonical evidence"
        )
    if (
        parsed["source_tick_executions"]
        - parsed["canonical_source_tick_executions"]
        != parsed["duplicated_source_tick_executions"]
        or parsed["legacy_actual_continuation_tick_count"]
        - parsed["canonical_legacy_actual_continuation_tick_count"]
        != parsed["duplicated_legacy_actual_continuation_tick_count"]
        or parsed["total_actual_simulator_tick_executions"]
        != (
            parsed["source_tick_executions"]
            + parsed["legacy_actual_continuation_tick_count"]
            + parsed["multi_tape_actual_continuation_tick_count"]
        )
    ):
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel execution compute arithmetic drifted"
        )
    if (
        parsed["duplicated_source_tick_executions"] <= 0
        or parsed["duplicated_legacy_actual_continuation_tick_count"] <= 0
    ):
        raise RecurrentCounterfactualCollectionError(
            "tape-parallel execution must disclose duplicated source and legacy work"
        )


def _positive_seed(value: object, *, field: str) -> int:
    parsed = _sampling_seed(value, field=field)
    if parsed <= 0:
        raise RecurrentCounterfactualCollectionError(f"{field} must be positive")
    return parsed


def _sampling_seed(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_RECURRENT_COUNTERFACTUAL_SEED
    ):
        raise RecurrentCounterfactualCollectionError(
            f"{field} must be an integer in [0, 2**63 - 1]"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentCounterfactualCollectionError(
            f"{field} must be a positive integer"
        )
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentCounterfactualCollectionError(
            f"{field} must be a non-negative integer"
        )
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentCounterfactualCollectionError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentCounterfactualCollectionError(f"{field} must be finite")
    return parsed


def _nonempty_trimmed_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentCounterfactualCollectionError(
            f"{field} must be non-empty and trimmed"
        )
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentCounterfactualCollectionError(
            f"{field} must be a lowercase SHA256 digest"
        )
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentCounterfactualCollectionError(f"{field} must be a mapping")
    return value


__all__ = [
    "MAX_RECURRENT_COUNTERFACTUAL_COLLECTION_WORKERS",
    "MIN_RECURRENT_COUNTERFACTUAL_TAPES_PER_CHUNK",
    "RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE",
    "RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION",
    "RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD",
    "RECURRENT_COUNTERFACTUAL_LEGACY_EXECUTION_CONTRACT_VERSION",
    "RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION",
    "RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE",
    "RECURRENT_COUNTERFACTUAL_TAPE_PARALLEL_EXECUTION_CONTRACT_VERSION",
    "RecurrentCounterfactualCollectionBundle",
    "RecurrentCounterfactualCollectionConfig",
    "RecurrentCounterfactualCollectionError",
    "RecurrentCounterfactualCollectionResult",
    "RecurrentCounterfactualCollectionTask",
    "collect_recurrent_counterfactual_bundles",
    "derive_recurrent_counterfactual_collection_seed",
    "recurrent_counterfactual_collection_config_payload",
    "recurrent_counterfactual_collection_result_payload",
    "recurrent_counterfactual_collection_task_payload",
    "validate_recurrent_counterfactual_collection_result",
]
