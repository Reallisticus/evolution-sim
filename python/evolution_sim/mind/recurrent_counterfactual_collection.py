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
    RECURRENT_COUNTERFACTUAL_BROAD_SCENARIO,
    RECURRENT_COUNTERFACTUAL_SCENARIOS,
    RECURRENT_COUNTERFACTUAL_SEED_ROLES,
    RecurrentCounterfactualBranchError,
    build_recurrent_counterfactual_nested_horizon_materialization,
    validate_recurrent_counterfactual_branch_row,
)
from evolution_sim.mind.recurrent_policy import (
    frozen_cpu_model_copy,
    recurrent_model_state_sha256,
)
from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY


RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION = (
    "mind_v3_recurrent_nested_counterfactual_collection_v1"
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
MAX_RECURRENT_COUNTERFACTUAL_SEED = 2**63 - 1


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

    def __post_init__(self) -> None:
        task_id = _nonempty_trimmed_string(self.task_id, field="task_id")
        if self.seed_role not in RECURRENT_COUNTERFACTUAL_SEED_ROLES:
            raise RecurrentCounterfactualCollectionError(
                "seed_role must be train or curriculum"
            )
        if self.scenario not in RECURRENT_COUNTERFACTUAL_SCENARIOS:
            raise RecurrentCounterfactualCollectionError("scenario is unsupported")
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
        if environment_seed not in RECURRENT_SEED_REGISTRY[self.seed_role]:
            raise RecurrentCounterfactualCollectionError(
                "environment_seed is not registered for its training role"
            )
        branch_ticks = _branch_tick_candidates(self.branch_tick_candidates)
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


@dataclass(frozen=True, slots=True)
class RecurrentCounterfactualCollectionConfig:
    horizons: tuple[int, ...] = (8, 32, 64)
    gamma: float = 0.99

    def __post_init__(self) -> None:
        horizons = _nested_horizons(self.horizons)
        gamma = _finite_number(self.gamma, field="gamma")
        if not 0.0 < gamma <= 1.0:
            raise RecurrentCounterfactualCollectionError("gamma must be in (0, 1]")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "gamma", gamma)


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
    worker_count = _worker_count(workers)
    source_model_digest = recurrent_model_state_sha256(model)
    behavior_model = frozen_cpu_model_copy(model)
    if recurrent_model_state_sha256(behavior_model) != source_model_digest:
        raise RecurrentCounterfactualCollectionError(
            "frozen CPU source model digest drifted"
        )

    if worker_count == 1 or len(task_tuple) == 1:
        materializations = tuple(
            _collect_one(
                behavior_model,
                task,
                artifact_digest=resolved_artifact,
                config=resolved_config,
            )
            for task in task_tuple
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
    result_without_digest = {
        "contract_version": RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION,
        "source_model_state_sha256": source_model_digest,
        "source_artifact_digest": resolved_artifact,
        "config": asdict(resolved_config),
        "bundles": [_bundle_payload(bundle) for bundle in bundles],
        "workers_requested": worker_count,
        "workers_resolved": min(worker_count, len(task_tuple)),
        "worker_start_method": RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD,
        "torch_threads_per_worker": RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS,
        "ordered_merge": True,
        "aggregate_compute": aggregate_compute,
        "training_ran": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    result = RecurrentCounterfactualCollectionResult(
        contract_version=RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION,
        source_model_state_sha256=source_model_digest,
        source_artifact_digest=resolved_artifact,
        config=resolved_config,
        bundles=bundles,
        workers_requested=worker_count,
        workers_resolved=min(worker_count, len(task_tuple)),
        worker_start_method=RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD,
        torch_threads_per_worker=RECURRENT_COUNTERFACTUAL_COLLECTION_TORCH_THREADS,
        ordered_merge=True,
        aggregate_compute=aggregate_compute,
        training_ran=False,
        runtime_action_selection_changed=False,
        promotion_authorized=False,
        exact_digest=stable_payload_digest(result_without_digest),
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
        "config": asdict(result.config),
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
    if result.contract_version != RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION:
        raise RecurrentCounterfactualCollectionError(
            "collection contract version drifted"
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
    if result.aggregate_compute != _aggregate_compute(result.bundles):
        raise RecurrentCounterfactualCollectionError(
            "aggregate compute diagnostics drifted"
        )
    if result.workers_requested != _worker_count(result.workers_requested):
        raise RecurrentCounterfactualCollectionError("workers_requested drifted")
    if result.workers_resolved != min(
        result.workers_requested,
        len(result.bundles),
    ):
        raise RecurrentCounterfactualCollectionError("workers_resolved drifted")
    if (
        result.worker_start_method != RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD
        or result.torch_threads_per_worker
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
    return {
        "contract_version": result.contract_version,
        "source_model_state_sha256": result.source_model_state_sha256,
        "source_artifact_digest": result.source_artifact_digest,
        "config": asdict(result.config),
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
            gamma=config.gamma,
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


def _bundle_from_materialization(
    task: RecurrentCounterfactualCollectionTask,
    value: Mapping[str, object],
) -> RecurrentCounterfactualCollectionBundle:
    rows = value.get("rows")
    valid_actions = value.get("valid_actions")
    horizons = value.get("horizons")
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
        "task": asdict(task),
        "rows": list(rows),
        "selection": dict(_mapping(value.get("selection"), field="selection")),
        "prefix_proof": dict(_mapping(value.get("prefix_proof"), field="prefix_proof")),
        "compute": dict(_mapping(value.get("compute"), field="compute")),
        "source_model_state_sha256": value.get("source_model_state_sha256"),
        "source_artifact_digest": value.get("source_artifact_digest"),
        "valid_actions": list(valid_actions),
        "horizons": list(horizons),
    }
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
    if bundle.source_model_state_sha256 != source_model_digest:
        raise RecurrentCounterfactualCollectionError("bundle model digest drifted")
    if bundle.source_artifact_digest != artifact_digest:
        raise RecurrentCounterfactualCollectionError("bundle artifact digest drifted")
    if bundle.horizons != config.horizons or len(bundle.rows) != len(config.horizons):
        raise RecurrentCounterfactualCollectionError("bundle horizons drifted")
    if tuple(action for action in ACTION_NAMES if action in bundle.valid_actions) != (
        bundle.valid_actions
    ):
        raise RecurrentCounterfactualCollectionError(
            "bundle valid actions are not in stable action order"
        )
    context_digest: str | None = None
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
        if metadata.get("horizon_ticks") != horizon:
            raise RecurrentCounterfactualCollectionError("row horizon drifted")
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
        elif observed_context_digest != context_digest:
            raise RecurrentCounterfactualCollectionError(
                "bundle horizons do not share one trainable public context"
            )
    _validate_selection(bundle)
    _validate_prefix_proof(bundle)
    _validate_compute(bundle)
    fields = _bundle_payload(bundle)
    observed_digest = fields.pop("exact_digest")
    if observed_digest != stable_payload_digest(fields):
        raise RecurrentCounterfactualCollectionError("bundle exact digest mismatch")


def _validate_selection(bundle: RecurrentCounterfactualCollectionBundle) -> None:
    selection = bundle.selection
    if selection.get("requested_branch_ticks") != list(
        bundle.task.branch_tick_candidates
    ):
        raise RecurrentCounterfactualCollectionError(
            "bundle branch tick request drifted"
        )
    selected_tick = selection.get("selected_branch_tick")
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
    expected_selected_index = random.Random(selection_seed).randrange(candidate_count)
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
    if selection.get("selected_focal_agent_id") != first_metadata.get(
        "focal_agent_id"
    ) or selected_tick != first_metadata.get("branch_tick"):
        raise RecurrentCounterfactualCollectionError(
            "selection and row focal state identities disagree"
        )


def _validate_prefix_proof(bundle: RecurrentCounterfactualCollectionBundle) -> None:
    proof = bundle.prefix_proof
    if (
        proof.get("schema_version") != "mind_v3_recurrent_nested_prefix_proof_v1"
        or proof.get("horizons") != list(bundle.horizons)
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
    if set(streams) != expected_streams or proof.get("stream_count") != len(streams):
        raise RecurrentCounterfactualCollectionError(
            "nested prefix proof stream set drifted"
        )
    for stream_name, raw_stream in streams.items():
        stream = _mapping(raw_stream, field="prefix stream")
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
        for horizon_index, horizon in enumerate(bundle.horizons):
            current = _mapping(
                by_horizon[str(horizon)],
                field="horizon prefix proof",
            )
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
        if compute.get(field) != expected_value:
            raise RecurrentCounterfactualCollectionError(
                f"compute diagnostic {field!r} drifted"
            )
    actual = _nonnegative_int(
        compute.get("actual_continuation_tick_count"),
        field="actual continuation tick count",
    )
    if actual > maximum_budget:
        raise RecurrentCounterfactualCollectionError(
            "actual continuation ticks exceed the exact maximum budget"
        )
    _positive_int(compute.get("source_tick_executions"), field="source ticks")


def _bundle_payload(
    bundle: RecurrentCounterfactualCollectionBundle,
) -> dict[str, object]:
    return {
        "task": asdict(bundle.task),
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
    return {
        "bundle_count": len(bundles),
        **{
            field: sum(int(bundle.compute[field]) for bundle in bundles)
            for field in fields
        },
    }


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
    "RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE",
    "RECURRENT_COUNTERFACTUAL_COLLECTION_CONTRACT_VERSION",
    "RECURRENT_COUNTERFACTUAL_COLLECTION_START_METHOD",
    "RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE",
    "RecurrentCounterfactualCollectionBundle",
    "RecurrentCounterfactualCollectionConfig",
    "RecurrentCounterfactualCollectionError",
    "RecurrentCounterfactualCollectionResult",
    "RecurrentCounterfactualCollectionTask",
    "collect_recurrent_counterfactual_bundles",
    "derive_recurrent_counterfactual_collection_seed",
    "recurrent_counterfactual_collection_result_payload",
    "validate_recurrent_counterfactual_collection_result",
]
