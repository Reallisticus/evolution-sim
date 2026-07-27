from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import heapq
import json
import math
import multiprocessing
from pathlib import Path
from statistics import median
import subprocess
from typing import Any

import torch

from evolution_sim.config import WorldConfig
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.runtime.trajectory import (
    REWARD_COMPONENT_BOUNDS,
    REWARD_TOTAL_BOUNDS,
)
from evolution_sim.env.runtime.observations import encode_observation_input
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
)
from evolution_sim.mind.policy_inputs import (
    TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ecological_policy_input_values,
    ecological_policy_input_vector_size,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    CRITIC_GENOME_CONDITIONING_FILM_V1,
    CRITIC_GENOME_CONDITIONING_NONE,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    PreviousPublicFeedbackInput,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
    VALUE_SHARED_TRUNK_GRADIENT_SHARED,
    VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
)
from evolution_sim.mind.recurrent_artifact import (
    FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
    RECURRENT_ARTIFACT_SCHEMA_VERSION,
    LoadedFrozenRecurrentPolicyArtifact,
    LoadedRecurrentArtifact,
    RecurrentArtifactError,
    load_frozen_recurrent_policy_artifact,
    load_recurrent_artifact,
)
from evolution_sim.mind.recurrent_experiment import (
    OPEN_ECOLOGY_MAX_AGENTS,
    OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE,
    OPEN_ECOLOGY_PHASE_B_DENSITY_CYCLE,
    OpenEcologyBroadWorldTreatment,
)
from evolution_sim.mind.recurrent_genome import (
    RECURRENT_CONTROLLER_GENOME_MAX,
    RECURRENT_CONTROLLER_GENOME_MIN,
    RECURRENT_CONTROLLER_GENOME_SIZE,
    RecurrentControllerGenome,
    zero_recurrent_genome,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationMode,
    recurrent_genome_stream_binding_sha256,
)
from evolution_sim.mind.recurrent_policy import (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_POLICY_ID,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
    RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION,
    DeterministicPublicRecurrentPolicy,
    RecurrentPolicyAdapterError,
    frozen_cpu_model_copy,
    recurrent_model_state_sha256,
)
from evolution_sim.mind.recurrent_rollout import (
    RECURRENT_ROLLOUT_ACTION_SOURCE,
    RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE,
    derive_recurrent_policy_sampling_seed,
)
from evolution_sim.mind.recurrent_scale_campaign import source_file_hash_manifest


OPEN_ECOLOGY_SELECTION_EVIDENCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_selection_evidence_v1"
)
OPEN_ECOLOGY_SELECTION_RUN_SCHEMA_VERSION = (
    "mind_v3_open_ecology_selection_run_evidence_v1"
)
OPEN_ECOLOGY_SELECTION_ENVIRONMENT_SCHEMA_VERSION = (
    "mind_v3_open_ecology_selection_environment_evidence_v1"
)
OPEN_ECOLOGY_SELECTION_CAUSAL_GENOME_SCHEMA_VERSION = (
    "mind_v3_open_ecology_causal_genome_battery_v1"
)
OPEN_ECOLOGY_SELECTION_ARTIFACT_BINDING_SCHEMA_VERSION = (
    "mind_v3_open_ecology_selection_artifact_binding_v2"
)
OPEN_ECOLOGY_PHASE_A_LEARNER_EVIDENCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_phase_a_learner_selection_evidence_v2"
)
OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION = (
    "mind_v3_open_ecology_terminal_selection_authority_v1"
)
OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SCHEMA_VERSION = (
    "mind_v3_open_ecology_capture_noninterference_proof_v1"
)
OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_CASE_TICKS = 128
OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_ROLE = "open_ecology_proof"
OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_INDICES = tuple(range(12))
OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_DENSITIES = (32, 64, 128)
OPEN_ECOLOGY_SELECTION_POLICY_SAMPLING_IDENTITY_VERSION = (
    "mind_v3_open_ecology_selection_policy_sampling_identity_v1"
)
OPEN_ECOLOGY_SELECTION_WORLD_IDENTITY_VERSION = (
    "mind_v3_open_ecology_selection_genome_world_identity_v1"
)
OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_SAMPLING_VERSION = (
    "mind_v3_open_ecology_min_hash_predecision_state_sample_v1"
)
OPEN_ECOLOGY_SELECTION_CAUSAL_INTERVENTION_VERSION = (
    "mind_v3_open_ecology_genome_zero_donor_locus_intervention_v1"
)
OPEN_ECOLOGY_SELECTION_METRIC_CONTRACT_VERSION = (
    "mind_v3_open_ecology_selection_metrics_v1"
)
OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_VERSION = (
    "mind_v3_open_ecology_action_collapse_nonoverlap_windows_v1"
)

OPEN_ECOLOGY_SELECTION_SEED_ROLE = "open_ecology_selection"
OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT = 32
OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT = 4
OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT = 64
OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_COUNT = (
    OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT
    * OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
)
OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW = 1_000
OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_CONSECUTIVE_WINDOWS = 3
OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_ACTION_SHARE = 0.80
OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_LINEAGE_SHARE = 0.90
OPEN_ECOLOGY_SELECTION_DONOR_JS_STATE_THRESHOLD = 0.01
OPEN_ECOLOGY_SELECTION_DONOR_JS_SHARE_THRESHOLD = 0.10
OPEN_ECOLOGY_SELECTION_DONOR_JS_MEAN_THRESHOLD = 0.002
OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_DELTA = 0.05
OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_TV_P99_MAX = 0.25
OPEN_ECOLOGY_SELECTION_DISCOUNT = 0.997

_PHASE_A_SELECTION_TICKS = 512
_PHASE_B_SELECTION_TICKS = 2_000
_PHASE_A_SELECTION_OFFSET = 0
_PHASE_B_SELECTION_OFFSET = 32
_PHASE_A_LEARNER_COUNT = 4
_PHASE_B_LEARNER_COUNT = 8
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_REWARD_NORMALIZATION_SCALE = max(abs(value) for value in REWARD_TOTAL_BOUNDS)
_PHASE_A_CELL_MODEL_CONTRACTS = {
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

_PARALLEL_PRIMARY_MODEL: PublicRecurrentActorCritic | None = None
_PARALLEL_REPLAY_MODEL: PublicRecurrentActorCritic | None = None
_PARALLEL_ARTIFACT_BINDING: dict[str, object] | None = None
_PARALLEL_SOURCE_REQUEST: OpenEcologySelectionRequest | None = None


class OpenEcologySelectionError(ValueError):
    """Raised when selection evidence differs from the sealed development plan."""


class OpenEcologySelectionPhase(StrEnum):
    PHASE_A = "phase_a"
    PHASE_B = "phase_b"


@dataclass(frozen=True, slots=True)
class OpenEcologySelectionPlan:
    phase: OpenEcologySelectionPhase
    environment_seed_indices: tuple[int, ...]
    environment_seeds: tuple[int, ...]
    ticks_per_execution: int
    stochastic_tape_identities: tuple[str, ...]
    density_cycle: tuple[int, ...]

    @classmethod
    def for_phase(
        cls,
        phase: OpenEcologySelectionPhase | str,
    ) -> OpenEcologySelectionPlan:
        resolved = _selection_phase(phase)
        values = _selection_plan_values(resolved)
        return cls(
            phase=resolved,
            environment_seed_indices=values["environment_seed_indices"],
            environment_seeds=values["environment_seeds"],
            ticks_per_execution=values["ticks_per_execution"],
            stochastic_tape_identities=values["stochastic_tape_identities"],
            density_cycle=values["density_cycle"],
        )

    def __post_init__(self) -> None:
        resolved = _selection_phase(self.phase)
        expected = _selection_plan_values(resolved)
        if (
            self.phase is not resolved
            or self.environment_seed_indices != expected["environment_seed_indices"]
            or self.environment_seeds != expected["environment_seeds"]
            or self.ticks_per_execution != expected["ticks_per_execution"]
            or self.stochastic_tape_identities != expected["stochastic_tape_identities"]
            or self.density_cycle != expected["density_cycle"]
        ):
            raise OpenEcologySelectionError(
                "open-ecology selection plan differs from its exact phase contract"
            )

    def initial_agents(self, local_environment_index: int) -> int:
        local = _bounded_index(
            local_environment_index,
            field="local_environment_index",
            upper=OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT,
        )
        return self.density_cycle[local % len(self.density_cycle)]


@dataclass(frozen=True, slots=True)
class OpenEcologySelectionRequest:
    artifact_path: Path
    run_contract_path: Path
    expected_artifact_sha256: str
    expected_source_commit: str
    expected_source_manifest_sha256: str
    campaign_digest: str
    run_contract_digest: str
    phase: OpenEcologySelectionPhase
    cell_id: str
    learner_index: int
    learner_seed: int
    evaluation_workers: int = 1
    artifact_logical_name: str = "terminal-training-artifact.json"
    run_contract_logical_name: str = "run-contract.json"
    training_authority: Mapping[str, object] | None = None
    source_repository_root: Path | None = None

    def __post_init__(self) -> None:
        path = Path(self.artifact_path)
        if not path.is_file() or path.is_symlink():
            raise OpenEcologySelectionError(
                "selection artifact path must be a regular non-symlink file"
            )
        object.__setattr__(self, "artifact_path", path)
        run_contract_path = Path(self.run_contract_path)
        if not run_contract_path.is_file() or run_contract_path.is_symlink():
            raise OpenEcologySelectionError(
                "run contract path must be a regular non-symlink file"
            )
        object.__setattr__(self, "run_contract_path", run_contract_path)
        for field in ("artifact_logical_name", "run_contract_logical_name"):
            logical_name = getattr(self, field)
            if (
                not isinstance(logical_name, str)
                or not logical_name
                or logical_name != logical_name.strip()
                or Path(logical_name).name != logical_name
            ):
                raise OpenEcologySelectionError(
                    f"{field} must be one portable logical file name"
                )
        for field in (
            "expected_artifact_sha256",
            "expected_source_manifest_sha256",
            "campaign_digest",
            "run_contract_digest",
        ):
            _sha256(getattr(self, field), field=field)
        _source_commit(self.expected_source_commit)
        source_root = self.source_repository_root
        if source_root is not None:
            source_root = Path(source_root)
            if (
                not source_root.is_absolute()
                or not source_root.is_dir()
                or source_root.is_symlink()
            ):
                raise OpenEcologySelectionError(
                    "source_repository_root must be one absolute regular directory"
                )
            object.__setattr__(self, "source_repository_root", source_root)
        if self.training_authority is not None:
            authority = dict(
                _mapping(
                    self.training_authority,
                    field="training_authority",
                )
            )
            _validate_training_authority_binding(
                authority,
                expected_campaign_digest=self.campaign_digest,
                expected_run_contract_digest=self.run_contract_digest,
                expected_artifact_sha256=self.expected_artifact_sha256,
                expected_artifact_file_sha256=_file_sha256(path),
                expected_source_commit=self.expected_source_commit,
                expected_source_manifest_sha256=(self.expected_source_manifest_sha256),
                expected_cell_id=self.cell_id,
                expected_learner_index=self.learner_index,
                expected_learner_seed=self.learner_seed,
                expected_run_contract_file_sha256=_file_sha256(run_contract_path),
            )
            object.__setattr__(self, "training_authority", authority)
        resolved_phase = _selection_phase(self.phase)
        object.__setattr__(self, "phase", resolved_phase)
        if self.cell_id not in _PHASE_A_CELL_MODEL_CONTRACTS:
            raise OpenEcologySelectionError(
                "cell_id must be one of the preregistered A0-A3 cells"
            )
        learner_count = (
            _PHASE_A_LEARNER_COUNT
            if resolved_phase is OpenEcologySelectionPhase.PHASE_A
            else _PHASE_B_LEARNER_COUNT
        )
        learner_index = _bounded_index(
            self.learner_index,
            field="learner_index",
            upper=learner_count,
        )
        expected_seed = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
            learner_index
        ]
        if self.learner_seed != expected_seed:
            raise OpenEcologySelectionError(
                "learner_seed differs from its canonical registry index"
            )
        _worker_count(self.evaluation_workers)


@dataclass(frozen=True, slots=True)
class _EnvironmentTask:
    phase: str
    cell_id: str
    learner_index: int
    artifact_sha256: str
    local_environment_index: int
    environment_seed_index: int
    environment_seed: int
    ticks: int
    initial_agents: int
    stochastic_tape_identities: tuple[str, ...]
    genome_stream_seed: int


@dataclass(frozen=True, slots=True)
class _CausalPolicyState:
    priority: int
    environment_seed: int
    decision_index: int
    agent_id: int
    observation: tuple[float, ...]
    action_mask: tuple[bool, ...]
    previous_feedback: tuple[float, ...]
    recurrent_state: tuple[float, ...]
    genome_values: tuple[float, ...]
    genome_sha256: str
    runtime_probabilities: tuple[float, ...]

    @property
    def identity_sha256(self) -> str:
        return stable_payload_digest(
            {
                "sampling_contract": (
                    OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_SAMPLING_VERSION
                ),
                "priority": f"{self.priority:064x}",
                "environment_seed": self.environment_seed,
                "decision_index": self.decision_index,
                "agent_id": self.agent_id,
                "genome_sha256": self.genome_sha256,
                "observation_sha256": stable_payload_digest(self.observation),
                "action_mask_sha256": stable_payload_digest(self.action_mask),
                "previous_feedback_sha256": stable_payload_digest(
                    self.previous_feedback
                ),
                "recurrent_state_sha256": stable_payload_digest(self.recurrent_state),
            }
        )


class _CausalStateCapturePolicy(DeterministicPublicRecurrentPolicy):
    """Read-only adapter retaining real pre-decision state with min-hash sampling.

    The base policy does not yet expose a public conditioned-state snapshot.
    This adapter deliberately reads its exact state before delegating to the
    unchanged decision implementation. It does not write policy state, consume
    RNG, alter action choice, or expose the captured values to the simulator.
    Exact full-world replay against the ordinary base policy is mandatory.
    """

    def __init__(
        self,
        model: PublicRecurrentActorCritic,
        *,
        artifact_digest: str,
        environment_seed: int,
    ) -> None:
        super().__init__(
            model,
            artifact_digest=artifact_digest,
            copy_to_cpu=False,
            reset_recurrent_state_each_decision=False,
            sampling_seed=None,
        )
        self._capture_environment_seed = environment_seed
        self._capture_seen = 0
        self._capture_heap: list[tuple[int, int, _CausalPolicyState]] = []

    @property
    def captured_states(self) -> tuple[_CausalPolicyState, ...]:
        return tuple(
            sorted(
                (entry[2] for entry in self._capture_heap),
                key=lambda state: (state.priority, state.decision_index),
            )
        )

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> Any:
        decision_index = self._capture_seen
        candidate = self._capture_candidate(
            observation,
            action_mask,
            decision_index=decision_index,
        )
        decision = super().decide(observation, action_mask)
        diagnostics = decision.diagnostics
        if not isinstance(diagnostics, Mapping):
            raise OpenEcologySelectionError(
                "causal capture decision lacks recurrent diagnostics"
            )
        distribution = _mapping(
            diagnostics.get("learned_masked_distribution"),
            field="learned_masked_distribution",
        )
        probabilities = _mapping(
            distribution.get("probabilities"),
            field="learned_masked_distribution.probabilities",
        )
        runtime_probabilities = tuple(
            _finite(probabilities.get(action), field=f"probabilities.{action}")
            for action in ACTION_NAMES
        )
        state = _CausalPolicyState(
            **candidate,
            runtime_probabilities=runtime_probabilities,
        )
        self._capture_seen += 1
        entry = (-state.priority, -state.decision_index, state)
        if len(self._capture_heap) < (
            OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
        ):
            heapq.heappush(self._capture_heap, entry)
        elif entry > self._capture_heap[0]:
            heapq.heapreplace(self._capture_heap, entry)
        return decision

    def _capture_candidate(
        self,
        observation: Mapping[str, object],
        action_mask: Mapping[str, object],
        *,
        decision_index: int,
    ) -> dict[str, object]:
        metadata = _mapping(observation.get("metadata"), field="observation.metadata")
        agent_id = _positive_int(
            metadata.get("agent_id"),
            field="observation.metadata.agent_id",
        )
        priority = int.from_bytes(
            hashlib.sha256(
                (
                    f"{OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_SAMPLING_VERSION}|"
                    f"environment={self._capture_environment_seed}|"
                    f"decision={decision_index}|agent={agent_id}"
                ).encode("ascii")
            ).digest(),
            "big",
        )
        encoded = encode_observation_input(dict(observation))
        public_values = tuple(ecological_policy_input_values(encoded))
        mask = tuple(
            _exact_bool(action_mask.get(action), field=action)
            for action in ACTION_NAMES
        )
        if not any(mask):
            raise OpenEcologySelectionError("causal state action mask is empty")
        feedback = self._feedback_by_agent.get(  # type: ignore[attr-defined]
            agent_id,
            PreviousPublicFeedbackInput.zero(),
        )
        state = self._state_by_agent.get(agent_id)  # type: ignore[attr-defined]
        if state is None:
            state = self.model.initial_state(1)
        manager = self._require_genome_population_manager()  # type: ignore[attr-defined]
        binding = manager.genome_binding_for_agent(agent_id)
        return {
            "priority": priority,
            "environment_seed": self._capture_environment_seed,
            "decision_index": decision_index,
            "agent_id": agent_id,
            "observation": public_values,
            "action_mask": mask,
            "previous_feedback": tuple(feedback.values()),
            "recurrent_state": tuple(
                float(value) for value in state.detach().cpu().reshape(-1).tolist()
            ),
            "genome_values": binding.genome.values,
            "genome_sha256": binding.genome_sha256,
        }


def evaluate_open_ecology_selection_artifact(
    request: OpenEcologySelectionRequest,
) -> dict[str, object]:
    """Run one exact Phase-A/B selection matrix under one CPU thread contract."""

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _evaluate_open_ecology_selection_artifact_single_thread(request)
    finally:
        torch.set_num_threads(previous_threads)


def _evaluate_open_ecology_selection_artifact_single_thread(
    request: OpenEcologySelectionRequest,
) -> dict[str, object]:
    """Implementation entered only after the parent CPU thread cap is fixed."""

    if not isinstance(request, OpenEcologySelectionRequest):
        raise TypeError("request must be an OpenEcologySelectionRequest")
    _require_live_selection_source(request)
    plan = OpenEcologySelectionPlan.for_phase(request.phase)
    primary, _replay, artifact_binding = _load_bound_artifact(request)
    tasks = tuple(
        _environment_task(
            request=request,
            plan=plan,
            local_environment_index=local_index,
        )
        for local_index in range(OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT)
    )
    workers = min(request.evaluation_workers, len(tasks))
    if workers == 1:
        environments = tuple(
            _evaluate_environment_task(
                task,
                primary_model=primary,
                replay_model=None,
            )
            for task in tasks
        )
    else:
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_selection_worker,
                initargs=(request,),
            ) as executor:
                environments = tuple(
                    executor.map(
                        _evaluate_environment_task_in_worker,
                        tasks,
                        chunksize=1,
                    )
                )
        except Exception as exc:
            raise OpenEcologySelectionError(
                "parallel open-ecology selection failed without sequential fallback"
            ) from exc
    if tuple(
        int(environment["local_environment_index"]) for environment in environments
    ) != tuple(range(OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT)):
        raise OpenEcologySelectionError(
            "selection environments returned out of canonical order"
        )
    _require_live_selection_source(request)
    causal = _aggregate_causal_genome_evidence(environments)
    metrics = _aggregate_learner_metrics(environments)
    gates = _learner_gates(environments=environments, causal=causal)
    report: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_SELECTION_EVIDENCE_SCHEMA_VERSION,
        "phase": request.phase.value,
        "campaign_digest": request.campaign_digest,
        "run_contract_digest": request.run_contract_digest,
        "cell_id": request.cell_id,
        "learner_index": request.learner_index,
        "learner_seed": request.learner_seed,
        "authority": {
            "authoritative": False,
            "state": "provisional_primary_execution_only",
            "authorization_requires": (
                "independent_full_artifact_reexecution_and_exact_report_match"
            ),
        },
        "artifact": artifact_binding,
        "contract": {
            "metric_contract_version": (OPEN_ECOLOGY_SELECTION_METRIC_CONTRACT_VERSION),
            "seed_registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
            "environment_seed_role": OPEN_ECOLOGY_SELECTION_SEED_ROLE,
            "environment_seed_indices": list(plan.environment_seed_indices),
            "environment_seeds": list(plan.environment_seeds),
            "ticks_per_execution": plan.ticks_per_execution,
            "stochastic_tape_identities": list(plan.stochastic_tape_identities),
            "argmax_diagnostic": True,
            "primary_metric_tapes": "four_stochastic_tapes_only",
            "aggregation_order": (
                "runs_within_environment_then_equal_weight_environments_within_learner"
            ),
            "density_cycle": list(plan.density_cycle),
            "max_agents": OPEN_ECOLOGY_MAX_AGENTS,
            "fixture_names": [],
            "carrion_objective": False,
            "genome_population_mode": (RecurrentGenomePopulationMode.HERITABLE.value),
            "genome_stream_seed_index": request.learner_index,
            "genome_stream_seed": OPEN_ECOLOGY_SEED_REGISTRY[
                "open_ecology_genome_stream"
            ][request.learner_index],
            "policy_sampling_seed_namespace": (
                RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
            ),
            "exact_cpu_replay": (
                "independent_full_artifact_reexecution_required_for_authority"
            ),
            "action_collapse_contract": {
                "schema_version": (OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_VERSION),
                "window_decisions": (OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW),
                "consecutive_windows": (
                    OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_CONSECUTIVE_WINDOWS
                ),
                "requested_action_share_threshold": (
                    OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_ACTION_SHARE
                ),
                "represented_lineage_share_threshold": (
                    OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_LINEAGE_SHARE
                ),
            },
            "causal_genome_contract": _causal_genome_contract(),
        },
        "execution": _selection_execution_accounting(
            environments,
            workers_requested=request.evaluation_workers,
            workers_used=workers,
        ),
        "environments": list(environments),
        "causal_genome": causal,
        "gates": gates,
        "metrics": metrics,
        "lifecycle": {
            "development_only": True,
            "training_performed": False,
            "validation_available": False,
            "validation_accessed": False,
            "lockbox_available": False,
            "lockbox_accessed": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
            "society_claim_authorized": False,
        },
    }
    report["exact_digest"] = stable_payload_digest(report)
    validate_open_ecology_selection_report(
        report,
        artifact_path=request.artifact_path,
        run_contract_path=request.run_contract_path,
    )
    return report


def _verify_open_ecology_selection_report_by_reexecution(
    report: Mapping[str, object],
    *,
    request: OpenEcologySelectionRequest,
) -> dict[str, object]:
    """Reexecute one request already rebuilt from a canonical terminal bundle.

    This is deliberately private. ``OpenEcologySelectionRequest`` and its
    unkeyed training-authority mapping are portable integrity data, not an
    authority capability. The public authority boundary lives in
    ``authorize_phase_a_terminal_selection_report`` and reconstructs the
    terminal, run contract, complete update prefix, checkpoint, and artifact
    before entering this helper.
    """

    if request.training_authority is None:
        raise OpenEcologySelectionError(
            "authoritative selection requires a verified terminal training chain"
        )
    validate_open_ecology_selection_report(
        report,
        artifact_path=request.artifact_path,
        run_contract_path=request.run_contract_path,
    )
    artifact = _mapping(report.get("artifact"), field="artifact")
    training_authority = _mapping(
        artifact.get("training_authority"),
        field="artifact.training_authority",
    )
    _validate_training_authority_binding(
        training_authority,
        expected_campaign_digest=str(report["campaign_digest"]),
        expected_run_contract_digest=str(report["run_contract_digest"]),
        expected_artifact_sha256=str(artifact["artifact_sha256"]),
        expected_artifact_file_sha256=str(artifact["file_sha256"]),
        expected_source_commit=str(artifact["source_commit"]),
        expected_source_manifest_sha256=str(artifact["source_manifest_sha256"]),
        expected_cell_id=report["cell_id"],
        expected_learner_index=report["learner_index"],
        expected_learner_seed=report["learner_seed"],
        expected_run_contract_file_sha256=str(artifact["run_contract_file_sha256"]),
        expected_model_state_sha256=str(artifact["model_state_sha256"]),
    )
    if (
        report.get("campaign_digest") != request.campaign_digest
        or report.get("run_contract_digest") != request.run_contract_digest
        or report.get("cell_id") != request.cell_id
        or report.get("learner_index") != request.learner_index
        or report.get("learner_seed") != request.learner_seed
        or artifact.get("artifact_sha256") != request.expected_artifact_sha256
        or artifact.get("source_commit") != request.expected_source_commit
        or artifact.get("source_manifest_sha256")
        != request.expected_source_manifest_sha256
    ):
        raise OpenEcologySelectionError(
            "selection reexecution request differs from the report"
        )
    reproduced = evaluate_open_ecology_selection_artifact(request)
    if reproduced.get("exact_digest") != report.get("exact_digest"):
        raise OpenEcologySelectionError(
            "authoritative selection reexecution changed the report"
        )
    return reproduced


def _phase_a_learner_evidence_from_verified_selection_report(
    report: Mapping[str, object],
    *,
    request: OpenEcologySelectionRequest,
) -> dict[str, object]:
    """Build selector evidence from a request rebuilt by the terminal verifier."""

    verification = _verify_open_ecology_selection_report_by_reexecution(
        report,
        request=request,
    )
    if report.get("phase") != OpenEcologySelectionPhase.PHASE_A.value:
        raise OpenEcologySelectionError(
            "Phase-A learner evidence requires a Phase-A selection report"
        )
    causal = _mapping(report.get("causal_genome"), field="causal_genome")
    if causal.get("state_count") != OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_COUNT:
        raise OpenEcologySelectionError(
            "Phase-A causal battery is incomplete; cell selection remains blocked"
        )
    artifact = _mapping(report.get("artifact"), field="artifact")
    training_authority = _mapping(
        artifact.get("training_authority"),
        field="artifact.training_authority",
    )
    contract = _mapping(report.get("contract"), field="contract")
    execution = _mapping(report.get("execution"), field="execution")
    authorization_accounting = _authorization_execution_accounting(execution)
    gates = _mapping(report.get("gates"), field="gates")
    metrics = _mapping(report.get("metrics"), field="metrics")
    authoritative_gates = dict(gates)
    authoritative_gates["exact_same_contract_replay"] = True
    authoritative_gates["eligible"] = all(
        value is True
        for name, value in authoritative_gates.items()
        if name != "eligible"
    )
    learner: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_PHASE_A_LEARNER_EVIDENCE_SCHEMA_VERSION,
        "campaign_digest": report["campaign_digest"],
        "run_contract_digest": report["run_contract_digest"],
        "cell_id": report["cell_id"],
        "learner_index": report["learner_index"],
        "learner_seed": report["learner_seed"],
        "artifact_sha256": artifact["artifact_sha256"],
        "terminal_authority": dict(training_authority),
        "evaluation": {
            "environment_seed_role": OPEN_ECOLOGY_SELECTION_SEED_ROLE,
            "environment_seed_indices": contract["environment_seed_indices"],
            "environment_seeds": contract["environment_seeds"],
            "ticks_per_execution": contract["ticks_per_execution"],
            "stochastic_tape_count": (OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT),
            "argmax_diagnostic": True,
            "fixture_names": [],
            "exact_cpu_artifact_replay": True,
            "selection_report_schema_version": report["schema_version"],
            "selection_report_exact_digest": report["exact_digest"],
            "artifact_file_sha256": artifact["file_sha256"],
            "per_environment_then_per_learner_aggregation": True,
            "authoritative_full_artifact_reexecution_verified": True,
            "authorization_verification_exact_digest": verification["exact_digest"],
            **authorization_accounting,
        },
        "causal_genome": {
            "schema_version": causal["schema_version"],
            "state_count": causal["state_count"],
            "state_manifest_sha256": causal["state_manifest_sha256"],
            "original_vs_donor_js_share_at_least_0.01": causal[
                "original_vs_donor_js_share_at_least_0.01"
            ],
            "mean_original_vs_donor_js": causal["mean_original_vs_donor_js"],
            "single_locus_total_variation_p99": causal[
                "single_locus_total_variation_p99"
            ],
        },
        "gates": authoritative_gates,
        "metrics": {
            "median_per_decision_normalized_individual_return": metrics[
                "normalized_return"
            ],
            "heldout_value_rmse": metrics["heldout_value_rmse"],
            "advantage_variance": metrics["advantage_variance"],
        },
        "lifecycle": {
            "development_only": True,
            "validation_accessed": False,
            "lockbox_accessed": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
        },
    }
    learner["exact_digest"] = stable_payload_digest(learner)
    return learner


def _selection_plan_values(
    phase: OpenEcologySelectionPhase,
) -> dict[str, Any]:
    if phase is OpenEcologySelectionPhase.PHASE_A:
        offset = _PHASE_A_SELECTION_OFFSET
        ticks = _PHASE_A_SELECTION_TICKS
        density_cycle = OPEN_ECOLOGY_PHASE_A_DENSITY_CYCLE
        tape_prefix = "phase-a-selection-tape"
    else:
        offset = _PHASE_B_SELECTION_OFFSET
        ticks = _PHASE_B_SELECTION_TICKS
        density_cycle = OPEN_ECOLOGY_PHASE_B_DENSITY_CYCLE
        tape_prefix = "phase-b-selection-tape"
    indices = tuple(range(offset, offset + OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT))
    seeds = OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_SELECTION_SEED_ROLE]
    if len(seeds) < indices[-1] + 1:
        raise OpenEcologySelectionError(
            "open-ecology selection registry is shorter than the sealed plan"
        )
    return {
        "environment_seed_indices": indices,
        "environment_seeds": tuple(seeds[index] for index in indices),
        "ticks_per_execution": ticks,
        "stochastic_tape_identities": tuple(
            f"{tape_prefix}-{index:02d}"
            for index in range(OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT)
        ),
        "density_cycle": tuple(density_cycle),
    }


def _selection_phase(
    value: OpenEcologySelectionPhase | str,
) -> OpenEcologySelectionPhase:
    try:
        return (
            value
            if isinstance(value, OpenEcologySelectionPhase)
            else OpenEcologySelectionPhase(value)
        )
    except (TypeError, ValueError) as exc:
        raise OpenEcologySelectionError(
            "selection phase must be phase_a or phase_b"
        ) from exc


def _bounded_index(value: object, *, field: str, upper: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= upper
    ):
        raise OpenEcologySelectionError(f"{field} must be an integer in [0, {upper})")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenEcologySelectionError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise OpenEcologySelectionError(f"{field} must be positive")
    return parsed


def _worker_count(value: object) -> int:
    parsed = _positive_int(value, field="evaluation_workers")
    if parsed > OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT:
        raise OpenEcologySelectionError(
            "evaluation_workers exceeds the environment count"
        )
    return parsed


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenEcologySelectionError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise OpenEcologySelectionError(f"{field} must be finite")
    return parsed


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise OpenEcologySelectionError(f"{field} must be an exact boolean")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpenEcologySelectionError(f"{field} must be a mapping")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise OpenEcologySelectionError(f"{field} must be a lowercase SHA256 digest")
    return value


def _source_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise OpenEcologySelectionError(
            "expected_source_commit must be a 40-character lowercase Git SHA"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise OpenEcologySelectionError(
            f"{field} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OpenEcologySelectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json_mapping(path: Path, *, field: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    OpenEcologySelectionError(
                        f"{field} contains non-finite JSON constant {value}"
                    )
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenEcologySelectionError(f"failed to read {field}") from exc
    if not isinstance(payload, dict):
        raise OpenEcologySelectionError(f"{field} root must be a mapping")
    return payload


def _load_bound_run_contract(
    request: OpenEcologySelectionRequest,
) -> tuple[dict[str, object], str]:
    contract = _load_strict_json_mapping(
        request.run_contract_path,
        field="run contract",
    )
    unsigned = dict(contract)
    exact_digest = _sha256(
        unsigned.pop("exact_digest", None),
        field="run contract exact_digest",
    )
    if stable_payload_digest(unsigned) != exact_digest:
        raise OpenEcologySelectionError("run contract exact digest changed")
    if exact_digest != request.run_contract_digest:
        raise OpenEcologySelectionError(
            "run contract bytes differ from the externally pinned digest"
        )
    source = _mapping(contract.get("source"), field="run contract source")
    model = _mapping(contract.get("model"), field="run contract model")
    terminal = _mapping(
        contract.get("terminal_policy"),
        field="run contract terminal_policy",
    )
    if (
        contract.get("campaign_digest") != request.campaign_digest
        or contract.get("cell_id") != request.cell_id
        or contract.get("learner_index") != request.learner_index
        or contract.get("learner_seed") != request.learner_seed
        or source.get("commit") != request.expected_source_commit
        or source.get("manifest_sha256") != request.expected_source_manifest_sha256
        or terminal.get("selection_evaluation_pending") is not True
        or terminal.get("runtime_integration_authorized") is not False
        or terminal.get("promotion_authorized") is not False
    ):
        raise OpenEcologySelectionError(
            "run contract bytes differ from the requested pending cell"
        )
    expected_critic, expected_gradient = _PHASE_A_CELL_MODEL_CONTRACTS[request.cell_id]
    if (
        model.get("critic_genome_conditioning") != expected_critic
        or model.get("value_shared_trunk_gradient") != expected_gradient
    ):
        raise OpenEcologySelectionError(
            "run contract model differs from its preregistered cell"
        )
    return contract, _file_sha256(request.run_contract_path)


def _validate_training_authority_binding(
    authority: Mapping[str, object],
    *,
    expected_campaign_digest: str,
    expected_run_contract_digest: str,
    expected_artifact_sha256: str,
    expected_artifact_file_sha256: str,
    expected_source_commit: str,
    expected_source_manifest_sha256: str,
    expected_cell_id: object,
    expected_learner_index: object,
    expected_learner_seed: object,
    expected_run_contract_file_sha256: str | None = None,
    expected_model_state_sha256: str | None = None,
) -> None:
    _require_exact_keys(
        authority,
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
        field="training authority",
    )
    unsigned = dict(authority)
    supplied_digest = _sha256(
        unsigned.pop("exact_digest", None),
        field="training authority exact_digest",
    )
    if stable_payload_digest(unsigned) != supplied_digest:
        raise OpenEcologySelectionError("training authority exact digest changed")
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
        _sha256(authority.get(field), field=f"training authority {field}")
    if (
        authority.get("schema_version")
        != OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION
        or authority.get("campaign_digest") != expected_campaign_digest
        or authority.get("source_commit") != expected_source_commit
        or authority.get("source_manifest_sha256") != expected_source_manifest_sha256
        or authority.get("cell_id") != expected_cell_id
        or authority.get("learner_index") != expected_learner_index
        or authority.get("learner_seed") != expected_learner_seed
        or authority.get("run_contract_exact_digest") != expected_run_contract_digest
        or authority.get("artifact_sha256") != expected_artifact_sha256
        or authority.get("artifact_file_sha256") != expected_artifact_file_sha256
        or authority.get("terminal_logical_name") != "terminal.json"
        or authority.get("run_contract_logical_name") != "run-contract.json"
        or authority.get("artifact_logical_name") != "terminal-training-artifact.json"
        or authority.get("selection_binding_required") is not True
    ):
        raise OpenEcologySelectionError(
            "training authority differs from its terminal selection request"
        )
    if (
        expected_run_contract_file_sha256 is not None
        and authority.get("run_contract_file_sha256")
        != expected_run_contract_file_sha256
    ):
        raise OpenEcologySelectionError("training authority run-contract bytes changed")
    if (
        expected_model_state_sha256 is not None
        and authority.get("terminal_checkpoint_model_state_sha256")
        != expected_model_state_sha256
    ):
        raise OpenEcologySelectionError(
            "training authority model differs from the loaded artifact"
        )
    run_id = authority.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
        raise OpenEcologySelectionError(
            "training authority run_id must be a non-empty canonical string"
        )


def _load_bound_artifact(
    request: OpenEcologySelectionRequest,
) -> tuple[
    PublicRecurrentActorCritic,
    PublicRecurrentActorCritic,
    dict[str, object],
]:
    run_contract, run_contract_file_sha256 = _load_bound_run_contract(request)
    try:
        with request.artifact_path.open("r", encoding="utf-8") as handle:
            header = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenEcologySelectionError(
            "failed to inspect selection artifact schema"
        ) from exc
    if not isinstance(header, Mapping):
        raise OpenEcologySelectionError("selection artifact root must be a mapping")
    schema = header.get("schema_version")
    try:
        if schema == FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION:
            loaded_one: (
                LoadedRecurrentArtifact | LoadedFrozenRecurrentPolicyArtifact
            ) = load_frozen_recurrent_policy_artifact(request.artifact_path)
            loaded_two: (
                LoadedRecurrentArtifact | LoadedFrozenRecurrentPolicyArtifact
            ) = load_frozen_recurrent_policy_artifact(request.artifact_path)
        elif schema == RECURRENT_ARTIFACT_SCHEMA_VERSION:
            loaded_one = load_recurrent_artifact(request.artifact_path)
            loaded_two = load_recurrent_artifact(request.artifact_path)
        else:
            raise OpenEcologySelectionError(
                "selection artifact has an unsupported schema"
            )
    except RecurrentArtifactError as exc:
        raise OpenEcologySelectionError(
            "selection artifact failed strict reconstruction"
        ) from exc
    artifact = loaded_one.artifact
    artifact_sha = _sha256(
        artifact.get("artifact_sha256"),
        field="artifact.artifact_sha256",
    )
    if artifact_sha != request.expected_artifact_sha256:
        raise OpenEcologySelectionError(
            "selection artifact SHA differs from the externally pinned SHA"
        )
    provenance = _mapping(artifact.get("provenance"), field="artifact.provenance")
    if provenance.get("seed_registry_digest") != OPEN_ECOLOGY_CANONICAL_SHA256:
        raise OpenEcologySelectionError(
            "selection artifact is not bound to the open-ecology registry"
        )
    if provenance.get("source_commit") != request.expected_source_commit:
        raise OpenEcologySelectionError(
            "selection artifact source commit differs from the external pin"
        )
    if provenance.get("learner_seed") != request.learner_seed:
        raise OpenEcologySelectionError(
            "selection artifact learner seed differs from the registry"
        )
    data = _mapping(
        provenance.get("data_metadata"),
        field="artifact.provenance.data_metadata",
    )
    run = _mapping(
        provenance.get("run_metadata"),
        field="artifact.provenance.run_metadata",
    )
    if (
        data.get("campaign_digest") != request.campaign_digest
        or data.get("run_contract_digest") != request.run_contract_digest
        or data.get("validation_accessed") is not False
        or data.get("lockbox_accessed") is not False
        or data.get("fixture_names") != []
    ):
        raise OpenEcologySelectionError(
            "selection artifact training-data boundary differs from the request"
        )
    if (
        run.get("cell_id") != request.cell_id
        or run.get("learner_index") != request.learner_index
        or run.get("selection_evaluation_pending") is not True
        or run.get("runtime_integration_authorized") is not False
        or run.get("promotion_authorized") is not False
    ):
        raise OpenEcologySelectionError(
            "selection artifact run metadata differs from its pending cell"
        )
    source_manifest = (
        provenance.get("source_manifest_sha256")
        if schema == FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION
        else run.get("source_manifest_sha256")
    )
    if source_manifest != request.expected_source_manifest_sha256:
        raise OpenEcologySelectionError(
            "selection artifact source manifest differs from the external pin"
        )
    config = loaded_one.model.config
    run_contract_model = _mapping(
        run_contract.get("model"),
        field="run contract model",
    )
    expected_critic, expected_gradient = _PHASE_A_CELL_MODEL_CONTRACTS[request.cell_id]
    signal_config = OpenEcologyBroadWorldTreatment(
        initial_agents=64
    ).signals.as_signal_config()
    if (
        config.encoder_size != 256
        or config.hidden_size != 256
        or config.recurrent_layers != 1
        or config.public_input_schema_version
        != TOKENIZED_ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION
        or config.public_input_size
        != ecological_policy_input_vector_size(signal_config)
        or config.genome_conditioning_mode != GENOME_CONDITIONING_ACTOR_FILM_V1
        or config.critic_genome_conditioning != expected_critic
        or config.value_shared_trunk_gradient != expected_gradient
        or dict(run_contract_model)
        != {
            "encoder_size": config.encoder_size,
            "hidden_size": config.hidden_size,
            "recurrent_layers": config.recurrent_layers,
            "public_input_schema_version": config.public_input_schema_version,
            "public_input_size": config.public_input_size,
            "genome_conditioning_mode": config.genome_conditioning_mode,
            "critic_genome_conditioning": config.critic_genome_conditioning,
            "value_shared_trunk_gradient": config.value_shared_trunk_gradient,
        }
    ):
        raise OpenEcologySelectionError(
            "selection artifact is not the preregistered tokenized actor_film_v1 model"
        )
    primary_sha = recurrent_model_state_sha256(loaded_one.model)
    replay_sha = recurrent_model_state_sha256(loaded_two.model)
    if primary_sha != replay_sha:
        raise OpenEcologySelectionError(
            "independent artifact reconstructions changed model state"
        )
    file_sha = _file_sha256(request.artifact_path)
    if request.training_authority is not None:
        _validate_training_authority_binding(
            request.training_authority,
            expected_campaign_digest=request.campaign_digest,
            expected_run_contract_digest=request.run_contract_digest,
            expected_artifact_sha256=request.expected_artifact_sha256,
            expected_artifact_file_sha256=file_sha,
            expected_source_commit=request.expected_source_commit,
            expected_source_manifest_sha256=(request.expected_source_manifest_sha256),
            expected_cell_id=request.cell_id,
            expected_learner_index=request.learner_index,
            expected_learner_seed=request.learner_seed,
            expected_run_contract_file_sha256=run_contract_file_sha256,
            expected_model_state_sha256=primary_sha,
        )
    binding: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_SELECTION_ARTIFACT_BINDING_SCHEMA_VERSION,
        "logical_name": request.artifact_logical_name,
        "file_sha256": file_sha,
        "artifact_schema_version": schema,
        "artifact_sha256": artifact_sha,
        "model_state_sha256": primary_sha,
        "source_commit": request.expected_source_commit,
        "source_manifest_sha256": request.expected_source_manifest_sha256,
        "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        "campaign_digest": request.campaign_digest,
        "run_contract_digest": request.run_contract_digest,
        "run_contract_logical_name": request.run_contract_logical_name,
        "run_contract_file_sha256": run_contract_file_sha256,
        "cell_id": request.cell_id,
        "learner_index": request.learner_index,
        "learner_seed": request.learner_seed,
        "training_authority": (
            dict(request.training_authority)
            if request.training_authority is not None
            else None
        ),
        "model_contract": {
            "encoder_size": config.encoder_size,
            "hidden_size": config.hidden_size,
            "recurrent_layers": config.recurrent_layers,
            "public_input_schema_version": config.public_input_schema_version,
            "public_input_size": config.public_input_size,
            "genome_conditioning_mode": config.genome_conditioning_mode,
            "critic_genome_conditioning": config.critic_genome_conditioning,
            "value_shared_trunk_gradient": config.value_shared_trunk_gradient,
        },
    }
    binding["exact_digest"] = stable_payload_digest(binding)
    return (
        frozen_cpu_model_copy(loaded_one.model),
        frozen_cpu_model_copy(loaded_two.model),
        binding,
    )


def _environment_task(
    *,
    request: OpenEcologySelectionRequest,
    plan: OpenEcologySelectionPlan,
    local_environment_index: int,
) -> _EnvironmentTask:
    local = _bounded_index(
        local_environment_index,
        field="local_environment_index",
        upper=OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT,
    )
    return _EnvironmentTask(
        phase=request.phase.value,
        cell_id=request.cell_id,
        learner_index=request.learner_index,
        artifact_sha256=request.expected_artifact_sha256,
        local_environment_index=local,
        environment_seed_index=plan.environment_seed_indices[local],
        environment_seed=plan.environment_seeds[local],
        ticks=plan.ticks_per_execution,
        initial_agents=plan.initial_agents(local),
        stochastic_tape_identities=plan.stochastic_tape_identities,
        genome_stream_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
            request.learner_index
        ],
    )


def _initialize_selection_worker(request: OpenEcologySelectionRequest) -> None:
    global _PARALLEL_ARTIFACT_BINDING
    global _PARALLEL_PRIMARY_MODEL
    global _PARALLEL_REPLAY_MODEL
    global _PARALLEL_SOURCE_REQUEST
    torch.set_num_threads(1)
    _require_live_selection_source(request)
    primary, replay, binding = _load_bound_artifact(request)
    _PARALLEL_PRIMARY_MODEL = primary
    _PARALLEL_REPLAY_MODEL = replay
    _PARALLEL_ARTIFACT_BINDING = binding
    _PARALLEL_SOURCE_REQUEST = request


def _require_live_selection_source(request: OpenEcologySelectionRequest) -> None:
    """Bind authoritative parent and spawned workers to one immutable checkout."""

    root = request.source_repository_root
    if root is None:
        return
    module_root = Path(__file__).resolve().parents[3]
    if module_root != root.resolve():
        raise OpenEcologySelectionError(
            "selection implementation was imported from a different checkout"
        )
    try:
        commit = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise OpenEcologySelectionError(
            "failed to inspect authoritative selection source"
        ) from error
    if commit != request.expected_source_commit or status.strip():
        raise OpenEcologySelectionError(
            "authoritative selection requires the exact clean source checkout"
        )
    try:
        manifest = source_file_hash_manifest(root)
    except (OSError, ValueError) as error:
        raise OpenEcologySelectionError(
            "failed to hash authoritative selection source"
        ) from error
    if manifest.get("aggregate_sha256") != request.expected_source_manifest_sha256:
        raise OpenEcologySelectionError(
            "authoritative selection source manifest drifted"
        )


def _evaluate_environment_task_in_worker(
    task: _EnvironmentTask,
) -> dict[str, object]:
    if (
        _PARALLEL_PRIMARY_MODEL is None
        or _PARALLEL_REPLAY_MODEL is None
        or _PARALLEL_ARTIFACT_BINDING is None
        or _PARALLEL_SOURCE_REQUEST is None
    ):
        raise OpenEcologySelectionError("selection worker was not initialized")
    if _PARALLEL_ARTIFACT_BINDING.get("artifact_sha256") != task.artifact_sha256:
        raise OpenEcologySelectionError(
            "selection worker artifact differs from its task"
        )
    _require_live_selection_source(_PARALLEL_SOURCE_REQUEST)
    result = _evaluate_environment_task(
        task,
        primary_model=_PARALLEL_PRIMARY_MODEL,
        replay_model=None,
    )
    _require_live_selection_source(_PARALLEL_SOURCE_REQUEST)
    return result


def _evaluate_environment_task(
    task: _EnvironmentTask,
    *,
    primary_model: PublicRecurrentActorCritic,
    replay_model: PublicRecurrentActorCritic | None,
) -> dict[str, object]:
    executions: list[dict[str, object]] = []
    for tape_identity in (*task.stochastic_tape_identities, "argmax"):
        sampling_seed = (
            None
            if tape_identity == "argmax"
            else _policy_sampling_seed(task, tape_identity=tape_identity)
        )
        primary, captured_states = _run_selection_world(
            model=primary_model,
            task=task,
            tape_identity=tape_identity,
            sampling_seed=sampling_seed,
            capture_causal_states=tape_identity == "argmax",
        )
        execution = dict(primary)
        if replay_model is None:
            execution["exact_replay"] = {
                "verified": False,
                "independent_artifact_reload": False,
                "replay_full_behavior_sha256": None,
                "replay_run_evidence_sha256": None,
            }
        else:
            replay, replay_captures = _run_selection_world(
                model=replay_model,
                task=task,
                tape_identity=tape_identity,
                sampling_seed=sampling_seed,
                capture_causal_states=False,
            )
            if replay_captures:
                raise OpenEcologySelectionError(
                    "ordinary replay unexpectedly captured causal states"
                )
            if primary["full_behavior_sha256"] != replay["full_behavior_sha256"]:
                raise OpenEcologySelectionError(
                    "independent exact CPU replay changed full-world behavior"
                )
            execution["exact_replay"] = {
                "verified": True,
                "independent_artifact_reload": True,
                "replay_full_behavior_sha256": replay["full_behavior_sha256"],
                "replay_run_evidence_sha256": replay["run_evidence_sha256"],
            }
        execution["exact_digest"] = stable_payload_digest(execution)
        executions.append(execution)
        if tape_identity == "argmax":
            causal_states = captured_states
    if "causal_states" not in locals():
        raise AssertionError("argmax causal capture was not run")
    stochastic = tuple(executions[:OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT])
    causal = _evaluate_causal_genome_states(
        primary_model,
        causal_states,
        environment_seed=task.environment_seed,
    )
    environment: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_SELECTION_ENVIRONMENT_SCHEMA_VERSION,
        "phase": task.phase,
        "cell_id": task.cell_id,
        "learner_index": task.learner_index,
        "artifact_sha256": task.artifact_sha256,
        "local_environment_index": task.local_environment_index,
        "environment_seed_index": task.environment_seed_index,
        "environment_seed": task.environment_seed,
        "ticks_per_execution": task.ticks,
        "initial_agents": task.initial_agents,
        "max_agents": OPEN_ECOLOGY_MAX_AGENTS,
        "genome_stream_seed": task.genome_stream_seed,
        "execution_count": len(executions),
        "exact_replay_execution_count": sum(
            int(
                _mapping(
                    execution["exact_replay"],
                    field="execution.exact_replay",
                )["verified"]
                is True
            )
            for execution in executions
        ),
        "executions": executions,
        "stochastic_tape_aggregate": _aggregate_stochastic_runs(stochastic),
        "causal_genome": causal,
        "gates": {
            "finite_outputs": all(
                _mapping(execution["gates"], field="execution.gates")["finite_outputs"]
                is True
                for execution in executions
            ),
            "exact_action_mask_legality": all(
                _mapping(execution["gates"], field="execution.gates")[
                    "exact_action_mask_legality"
                ]
                is True
                for execution in executions
            ),
            "exact_same_contract_replay": all(
                _mapping(execution["exact_replay"], field="exact_replay")["verified"]
                is True
                for execution in executions
            ),
            "no_heuristic_action_source": all(
                _mapping(execution["gates"], field="execution.gates")[
                    "no_heuristic_action_source"
                ]
                is True
                for execution in executions
            ),
            "no_global_action_collapse": all(
                _mapping(execution["gates"], field="execution.gates")[
                    "no_global_action_collapse"
                ]
                is True
                for execution in executions
            ),
        },
    }
    environment["exact_digest"] = stable_payload_digest(environment)
    _validate_environment_evidence(environment)
    return environment


def _policy_sampling_seed(
    task: _EnvironmentTask,
    *,
    tape_identity: str,
) -> int:
    identity = (
        f"{OPEN_ECOLOGY_SELECTION_POLICY_SAMPLING_IDENTITY_VERSION}|"
        f"phase={task.phase}|environment={task.environment_seed_index}|"
        f"tape={tape_identity}"
    )
    return derive_recurrent_policy_sampling_seed(task_identity=identity)


def _world_identity(task: _EnvironmentTask) -> str:
    return (
        f"{OPEN_ECOLOGY_SELECTION_WORLD_IDENTITY_VERSION}|"
        f"phase={task.phase}|learner={task.learner_index}|"
        f"environment={task.environment_seed_index}|seed={task.environment_seed}|"
        f"density={task.initial_agents}"
    )


def _run_selection_world(
    *,
    model: PublicRecurrentActorCritic,
    task: _EnvironmentTask,
    tape_identity: str,
    sampling_seed: int | None,
    capture_causal_states: bool,
) -> tuple[dict[str, object], tuple[_CausalPolicyState, ...]]:
    if capture_causal_states:
        if sampling_seed is not None or tape_identity != "argmax":
            raise OpenEcologySelectionError(
                "causal capture is restricted to the argmax diagnostic"
            )
        policy: DeterministicPublicRecurrentPolicy = _CausalStateCapturePolicy(
            model,
            artifact_digest=task.artifact_sha256,
            environment_seed=task.environment_seed,
        )
    else:
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest=task.artifact_sha256,
            copy_to_cpu=False,
            reset_recurrent_state_each_decision=False,
            sampling_seed=sampling_seed,
        )
    world_identity = _world_identity(task)
    try:
        policy.start_world(
            world_identity=world_identity,
            genome_stream_seed=task.genome_stream_seed,
            genome_population_mode=RecurrentGenomePopulationMode.HERITABLE,
        )
        treatment = OpenEcologyBroadWorldTreatment(initial_agents=task.initial_agents)
        world = SimulationWorld(
            WorldConfig(
                seed=task.environment_seed,
                max_ticks=task.ticks,
                initial_agents=treatment.initial_agents,
                max_agents=treatment.max_agents,
                signals=treatment.signals.as_signal_config(),
            ),
            policy=policy,
        )
        result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        captured = (
            policy.captured_states
            if isinstance(policy, _CausalStateCapturePolicy)
            else ()
        )
        policy.reconcile_live_agent_ids(
            live_agent_ids=tuple(
                sorted(agent.agent_id for agent in world.alive_agents())
            )
        )
        policy.reset_world()
    except (RecurrentPolicyAdapterError, RuntimeError, ValueError) as exc:
        raise OpenEcologySelectionError("open-ecology selection world failed") from exc
    provenance = policy.last_world_genome_provenance
    if provenance is None:
        raise OpenEcologySelectionError(
            "selection world lacks finalized genome population provenance"
        )
    _validate_genome_world_provenance(
        provenance,
        expected_world_identity=world_identity,
        expected_genome_stream_seed=task.genome_stream_seed,
        expected_action_selection=(
            PUBLIC_RECURRENT_ARGMAX_SELECTION
            if sampling_seed is None
            else PUBLIC_RECURRENT_SAMPLED_SELECTION
        ),
        expected_policy_sampling_seed=sampling_seed,
    )
    records = world.trajectory_records
    diagnostics = world.policy_decision_diagnostics_records
    if len(records) != len(diagnostics):
        raise OpenEcologySelectionError(
            "trajectory and decision diagnostics are not aligned"
        )
    evidence = _build_selection_run_evidence(
        task=task,
        tape_identity=tape_identity,
        sampling_seed=sampling_seed,
        summary=result.summary,
        records=records,
        diagnostics=diagnostics,
        genome_provenance=provenance,
    )
    return evidence, captured


def _capture_noninterference_cases() -> tuple[dict[str, object], ...]:
    cases: list[dict[str, object]] = []
    proof_seeds = OPEN_ECOLOGY_SEED_REGISTRY[
        OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_ROLE
    ]
    learner_seeds = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"]
    genome_seeds = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"]
    for cell_index, cell_id in enumerate(_PHASE_A_CELL_MODEL_CONTRACTS):
        for initial_agents in OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_DENSITIES:
            case_index = len(cases)
            environment_seed_index = OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_INDICES[
                case_index
            ]
            cases.append(
                {
                    "case_index": case_index,
                    "cell_id": cell_id,
                    "learner_index": cell_index,
                    "learner_seed": learner_seeds[cell_index],
                    "environment_seed_role": (
                        OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_ROLE
                    ),
                    "environment_seed_index": environment_seed_index,
                    "environment_seed": proof_seeds[environment_seed_index],
                    "initial_agents": initial_agents,
                    "ticks": OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_CASE_TICKS,
                    "genome_stream_seed": genome_seeds[cell_index],
                }
            )
    return tuple(cases)


def build_open_ecology_capture_noninterference_proof(
    *,
    source_commit: str,
    source_manifest_sha256: str,
) -> dict[str, object]:
    """Prove that protected-state capture cannot alter ordinary world behavior."""

    commit = _source_commit(source_commit)
    manifest = _sha256(
        source_manifest_sha256,
        field="source_manifest_sha256",
    )
    rows: list[dict[str, object]] = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for case in _capture_noninterference_cases():
            cell_id = str(case["cell_id"])
            critic, gradient = _PHASE_A_CELL_MODEL_CONTRACTS[cell_id]
            signal_config = OpenEcologyBroadWorldTreatment(
                initial_agents=int(case["initial_agents"])
            ).signals.as_signal_config()
            config = RecurrentActorCriticConfig.for_signal_config(
                signal_config,
                encoder_size=256,
                hidden_size=256,
                recurrent_layers=1,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
                critic_genome_conditioning=critic,
                value_shared_trunk_gradient=gradient,
            )
            base_model = PublicRecurrentActorCritic(
                config,
                initialization_seed=int(case["learner_seed"]),
            ).to(device="cpu", dtype=torch.float32)
            capture_model = frozen_cpu_model_copy(base_model)
            ordinary_model = frozen_cpu_model_copy(base_model)
            initial_model_sha256 = recurrent_model_state_sha256(base_model)
            task = _EnvironmentTask(
                phase="capture_noninterference",
                cell_id=cell_id,
                learner_index=int(case["learner_index"]),
                artifact_sha256=initial_model_sha256,
                local_environment_index=int(case["case_index"]),
                environment_seed_index=int(case["environment_seed_index"]),
                environment_seed=int(case["environment_seed"]),
                ticks=int(case["ticks"]),
                initial_agents=int(case["initial_agents"]),
                stochastic_tape_identities=(),
                genome_stream_seed=int(case["genome_stream_seed"]),
            )
            capture, captured_states = _run_selection_world(
                model=capture_model,
                task=task,
                tape_identity="argmax",
                sampling_seed=None,
                capture_causal_states=True,
            )
            ordinary, ordinary_captures = _run_selection_world(
                model=ordinary_model,
                task=task,
                tape_identity="argmax",
                sampling_seed=None,
                capture_causal_states=False,
            )
            if ordinary_captures:
                raise OpenEcologySelectionError(
                    "ordinary noninterference policy captured protected state"
                )
            terminal = _mapping(
                capture.get("terminal"),
                field="capture terminal",
            )
            row: dict[str, object] = {
                **case,
                "initial_model_state_sha256": initial_model_sha256,
                "capture_final_model_state_sha256": (
                    recurrent_model_state_sha256(capture_model)
                ),
                "ordinary_final_model_state_sha256": (
                    recurrent_model_state_sha256(ordinary_model)
                ),
                "captured_state_count": len(captured_states),
                "capture_full_behavior_sha256": capture["full_behavior_sha256"],
                "ordinary_full_behavior_sha256": ordinary["full_behavior_sha256"],
                "capture_run_evidence_sha256": capture["run_evidence_sha256"],
                "ordinary_run_evidence_sha256": ordinary["run_evidence_sha256"],
                "full_run_evidence_equal": capture == ordinary,
                "model_state_unchanged": (
                    recurrent_model_state_sha256(capture_model)
                    == initial_model_sha256
                    == recurrent_model_state_sha256(ordinary_model)
                ),
                "births": terminal["births"],
                "deaths": terminal["deaths"],
                "passed": (
                    capture == ordinary
                    and len(captured_states)
                    == OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
                    and recurrent_model_state_sha256(capture_model)
                    == initial_model_sha256
                    == recurrent_model_state_sha256(ordinary_model)
                ),
            }
            row["exact_digest"] = stable_payload_digest(row)
            rows.append(row)
    finally:
        torch.set_num_threads(previous_threads)
    proof: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SCHEMA_VERSION,
        "source": {
            "commit": commit,
            "manifest_sha256": manifest,
        },
        "contract": {
            "adapter": "read_only_protected_state_capture_v1",
            "ordinary_policy": PUBLIC_RECURRENT_POLICY_ID,
            "device": "cpu",
            "parameter_dtype": "torch.float32",
            "torch_num_threads": 1,
            "cell_order": list(_PHASE_A_CELL_MODEL_CONTRACTS),
            "case_matrix": "four_cells_by_three_densities_cartesian_v1",
            "density_levels": list(OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_DENSITIES),
            "environment_seed_role": (OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_ROLE),
            "environment_seed_indices": list(
                OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_INDICES
            ),
            "scientific_selection_seed_accessed": False,
            "case_count": len(OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_INDICES),
            "ticks_per_case": (OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_CASE_TICKS),
            "comparison": (
                "full_summary_trajectory_decision_diagnostics_rng_and_genome_provenance"
            ),
        },
        "cases": rows,
        "dynamics": {
            "births_observed": any(int(row["births"]) > 0 for row in rows),
            "deaths_observed": any(int(row["deaths"]) > 0 for row in rows),
            "total_births": sum(int(row["births"]) for row in rows),
            "total_deaths": sum(int(row["deaths"]) for row in rows),
        },
        "passed": (
            all(row["passed"] is True for row in rows)
            and any(int(row["births"]) > 0 for row in rows)
            and any(int(row["deaths"]) > 0 for row in rows)
        ),
    }
    proof["exact_digest"] = stable_payload_digest(proof)
    validate_open_ecology_capture_noninterference_proof(
        proof,
        expected_source_commit=commit,
        expected_source_manifest_sha256=manifest,
    )
    return proof


def validate_open_ecology_capture_noninterference_proof(
    proof: Mapping[str, object],
    *,
    expected_source_commit: str,
    expected_source_manifest_sha256: str,
) -> None:
    _require_exact_keys(
        proof,
        {
            "schema_version",
            "source",
            "contract",
            "cases",
            "dynamics",
            "passed",
            "exact_digest",
        },
        field="capture noninterference proof",
    )
    unsigned = dict(proof)
    supplied_digest = _sha256(
        unsigned.pop("exact_digest", None),
        field="capture noninterference exact_digest",
    )
    if stable_payload_digest(unsigned) != supplied_digest:
        raise OpenEcologySelectionError("capture noninterference proof digest changed")
    source = _mapping(proof.get("source"), field="capture proof source")
    contract = _mapping(
        proof.get("contract"),
        field="capture proof contract",
    )
    _require_exact_keys(
        source,
        {"commit", "manifest_sha256"},
        field="capture proof source",
    )
    if (
        proof.get("schema_version")
        != OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SCHEMA_VERSION
        or source.get("commit") != _source_commit(expected_source_commit)
        or source.get("manifest_sha256")
        != _sha256(
            expected_source_manifest_sha256,
            field="expected_source_manifest_sha256",
        )
        or dict(contract)
        != {
            "adapter": "read_only_protected_state_capture_v1",
            "ordinary_policy": PUBLIC_RECURRENT_POLICY_ID,
            "device": "cpu",
            "parameter_dtype": "torch.float32",
            "torch_num_threads": 1,
            "cell_order": list(_PHASE_A_CELL_MODEL_CONTRACTS),
            "case_matrix": "four_cells_by_three_densities_cartesian_v1",
            "density_levels": list(OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_DENSITIES),
            "environment_seed_role": (OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_ROLE),
            "environment_seed_indices": list(
                OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_INDICES
            ),
            "scientific_selection_seed_accessed": False,
            "case_count": len(OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SEED_INDICES),
            "ticks_per_case": (OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_CASE_TICKS),
            "comparison": (
                "full_summary_trajectory_decision_diagnostics_rng_and_genome_provenance"
            ),
        }
    ):
        raise OpenEcologySelectionError(
            "capture noninterference proof contract changed"
        )
    case_values = proof.get("cases")
    if not isinstance(case_values, Sequence) or isinstance(
        case_values,
        (str, bytes),
    ):
        raise OpenEcologySelectionError(
            "capture noninterference cases must be a sequence"
        )
    cases = tuple(_mapping(value, field="capture proof case") for value in case_values)
    expected_cases = _capture_noninterference_cases()
    if len(cases) != len(expected_cases) or len(cases) != contract.get("case_count"):
        raise OpenEcologySelectionError(
            "capture noninterference case matrix is incomplete"
        )
    total_births = 0
    total_deaths = 0
    for row, expected in zip(cases, expected_cases, strict=True):
        _require_exact_keys(
            row,
            {
                *expected,
                "initial_model_state_sha256",
                "capture_final_model_state_sha256",
                "ordinary_final_model_state_sha256",
                "captured_state_count",
                "capture_full_behavior_sha256",
                "ordinary_full_behavior_sha256",
                "capture_run_evidence_sha256",
                "ordinary_run_evidence_sha256",
                "full_run_evidence_equal",
                "model_state_unchanged",
                "births",
                "deaths",
                "passed",
                "exact_digest",
            },
            field="capture proof case",
        )
        row_unsigned = dict(row)
        row_digest = _sha256(
            row_unsigned.pop("exact_digest", None),
            field="capture proof case exact_digest",
        )
        if stable_payload_digest(row_unsigned) != row_digest or any(
            row.get(key) != value for key, value in expected.items()
        ):
            raise OpenEcologySelectionError(
                "capture noninterference case identity changed"
            )
        initial_sha = _sha256(
            row.get("initial_model_state_sha256"),
            field="capture initial model",
        )
        capture_sha = _sha256(
            row.get("capture_final_model_state_sha256"),
            field="capture final model",
        )
        ordinary_sha = _sha256(
            row.get("ordinary_final_model_state_sha256"),
            field="ordinary final model",
        )
        capture_behavior = _sha256(
            row.get("capture_full_behavior_sha256"),
            field="capture behavior",
        )
        ordinary_behavior = _sha256(
            row.get("ordinary_full_behavior_sha256"),
            field="ordinary behavior",
        )
        capture_evidence = _sha256(
            row.get("capture_run_evidence_sha256"),
            field="capture evidence",
        )
        ordinary_evidence = _sha256(
            row.get("ordinary_run_evidence_sha256"),
            field="ordinary evidence",
        )
        births = _nonnegative_int(row.get("births"), field="capture births")
        deaths = _nonnegative_int(row.get("deaths"), field="capture deaths")
        total_births += births
        total_deaths += deaths
        if (
            row.get("captured_state_count")
            != OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
            or capture_behavior != ordinary_behavior
            or capture_evidence != ordinary_evidence
            or initial_sha != capture_sha
            or initial_sha != ordinary_sha
            or row.get("full_run_evidence_equal") is not True
            or row.get("model_state_unchanged") is not True
            or row.get("passed") is not True
        ):
            raise OpenEcologySelectionError(
                "capture adapter changed ordinary policy behavior"
            )
    dynamics = _mapping(
        proof.get("dynamics"),
        field="capture proof dynamics",
    )
    if dict(dynamics) != {
        "births_observed": total_births > 0,
        "deaths_observed": total_deaths > 0,
        "total_births": total_births,
        "total_deaths": total_deaths,
    } or proof.get("passed") is not (total_births > 0 and total_deaths > 0):
        raise OpenEcologySelectionError(
            "capture noninterference dynamics coverage is incomplete"
        )


def verify_open_ecology_capture_noninterference_proof_by_reexecution(
    proof: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    """Reexecute only after binding this process to the clean preregistered source."""

    from evolution_sim.mind.open_ecology_phase_a import (
        _require_live_source,
        validate_open_ecology_phase_a_preregistration,
    )

    validate_open_ecology_phase_a_preregistration(preregistration)
    _require_live_source(preregistration)
    source = _mapping(preregistration.get("source"), field="preregistration.source")
    expected_source_commit = _source_commit(source.get("commit"))
    expected_source_manifest_sha256 = _sha256(
        source.get("manifest_sha256"),
        field="preregistration.source.manifest_sha256",
    )

    validate_open_ecology_capture_noninterference_proof(
        proof,
        expected_source_commit=expected_source_commit,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    reproduced = build_open_ecology_capture_noninterference_proof(
        source_commit=expected_source_commit,
        source_manifest_sha256=expected_source_manifest_sha256,
    )
    if reproduced.get("exact_digest") != proof.get("exact_digest"):
        raise OpenEcologySelectionError(
            "capture noninterference proof changed under exact reexecution"
        )
    return reproduced


def _validate_genome_world_provenance(
    provenance: Mapping[str, object],
    *,
    expected_world_identity: str,
    expected_genome_stream_seed: int,
    expected_action_selection: str,
    expected_policy_sampling_seed: int | None,
) -> None:
    _require_exact_keys(
        provenance,
        {
            "schema_version",
            "genome_conditioning_mode",
            "genome_population_mode",
            "action_selection",
            "policy_sampling_seed",
            "world_identity",
            "genome_stream_seed",
            "genome_population_binding_sha256",
            "genome_population_pre_founder_state_sha256",
            "genome_population_final_state_sha256",
            "genome_population_reset_state_sha256",
            "provenance_sha256",
        },
        field="genome world provenance",
    )
    unsigned = dict(provenance)
    supplied = _sha256(
        unsigned.pop("provenance_sha256"),
        field="genome provenance SHA256",
    )
    if stable_payload_digest(unsigned) != supplied:
        raise OpenEcologySelectionError("genome provenance digest changed")
    if (
        provenance.get("schema_version")
        != RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION
        or provenance.get("genome_conditioning_mode")
        != GENOME_CONDITIONING_ACTOR_FILM_V1
        or provenance.get("genome_population_mode")
        != RecurrentGenomePopulationMode.HERITABLE.value
        or provenance.get("action_selection") != expected_action_selection
        or provenance.get("policy_sampling_seed") != expected_policy_sampling_seed
        or provenance.get("world_identity") != expected_world_identity
        or provenance.get("genome_stream_seed") != expected_genome_stream_seed
    ):
        raise OpenEcologySelectionError(
            "genome provenance differs from the selection execution"
        )
    for field in (
        "genome_population_binding_sha256",
        "genome_population_pre_founder_state_sha256",
        "genome_population_final_state_sha256",
        "genome_population_reset_state_sha256",
    ):
        _sha256(provenance.get(field), field=f"genome provenance {field}")
    if provenance.get(
        "genome_population_binding_sha256"
    ) != recurrent_genome_stream_binding_sha256(
        genome_stream_seed=expected_genome_stream_seed,
        world_identity=expected_world_identity,
    ):
        raise OpenEcologySelectionError(
            "genome population binding digest differs from seed and world"
        )
    if provenance.get("genome_population_pre_founder_state_sha256") != provenance.get(
        "genome_population_reset_state_sha256"
    ):
        raise OpenEcologySelectionError(
            "genome population did not reset to its pre-founder state"
        )


def _build_selection_run_evidence(
    *,
    task: _EnvironmentTask,
    tape_identity: str,
    sampling_seed: int | None,
    summary: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object] | None],
    genome_provenance: Mapping[str, object],
) -> dict[str, object]:
    requested_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    policy_id_counts: Counter[str] = Counter()
    active_rows: list[tuple[str, int]] = []
    rewards_by_agent: dict[int, list[float]] = {}
    values_by_agent: dict[int, list[float]] = {}
    passive_rewards: dict[int, float] = {}
    reward_total = 0.0
    unsupported = 0
    finite_outputs = True
    for record, decision_diagnostics in zip(records, diagnostics, strict=True):
        requested = record.get("requested_action")
        if not isinstance(requested, str) or requested not in ACTION_NAMES:
            raise OpenEcologySelectionError(
                "selection trajectory requested action is invalid"
            )
        reward = _mapping(record.get("reward"), field="record.reward")
        reward_value = _finite(reward.get("total"), field="record.reward.total")
        reward_total += reward_value
        components = _mapping(
            reward.get("components"),
            field="record.reward.components",
        )
        if set(components) != set(REWARD_COMPONENT_BOUNDS):
            raise OpenEcologySelectionError(
                "selection reward components differ from the Foundation contract"
            )
        for name in REWARD_COMPONENT_BOUNDS:
            _finite(components.get(name), field=f"reward.components.{name}")
        agent_id = _positive_int(record.get("agent_id"), field="record.agent_id")
        source = str(record.get("action_source", "unknown"))
        action_source_counts[source] += 1
        policy_id_counts[str(record.get("policy_id", "unknown"))] += 1
        if source == "passive":
            passive_rewards[agent_id] = (
                passive_rewards.get(agent_id, 0.0) + reward_value
            )
            continue
        if decision_diagnostics is None:
            raise OpenEcologySelectionError(
                "active recurrent decision lacks diagnostics"
            )
        if source != RECURRENT_ROLLOUT_ACTION_SOURCE:
            finite_outputs = False
        if record.get("policy_id") != PUBLIC_RECURRENT_POLICY_ID:
            finite_outputs = False
        if record.get("action_valid") is not True:
            unsupported += 1
        action_mask = _mapping(
            record.get("action_mask"),
            field="record.action_mask",
        )
        if tuple(action_mask) != tuple(ACTION_NAMES):
            raise OpenEcologySelectionError(
                "selection action mask order differs from the public contract"
            )
        parsed_mask = {
            action: _exact_bool(
                action_mask.get(action),
                field=f"record.action_mask.{action}",
            )
            for action in ACTION_NAMES
        }
        if parsed_mask[requested] is not (record.get("action_valid") is True):
            raise OpenEcologySelectionError(
                "selection action_valid differs from the tick-start action mask"
            )
        distribution = _mapping(
            decision_diagnostics.get("learned_masked_distribution"),
            field="decision.learned_masked_distribution",
        )
        probabilities = _mapping(
            distribution.get("probabilities"),
            field="decision.probabilities",
        )
        probability_sum = 0.0
        for action in ACTION_NAMES:
            probability = _finite(
                probabilities.get(action),
                field=f"decision.probabilities.{action}",
            )
            if probability < 0.0 or probability > 1.0:
                finite_outputs = False
            if not parsed_mask[action] and probability != 0.0:
                finite_outputs = False
            probability_sum += probability
        if abs(probability_sum - 1.0) > 1.0e-8:
            finite_outputs = False
        value = _finite(decision_diagnostics.get("value"), field="decision.value")
        if (
            decision_diagnostics.get("genome_conditioning_mode")
            != GENOME_CONDITIONING_ACTOR_FILM_V1
            or decision_diagnostics.get("genome_population_mode")
            != RecurrentGenomePopulationMode.HERITABLE.value
            or decision_diagnostics.get("genome_stream_seed") != task.genome_stream_seed
        ):
            raise OpenEcologySelectionError(
                "selection decision lacks the exact heritable actor-FiLM binding"
            )
        _sha256(
            decision_diagnostics.get("genome_sha256"),
            field="decision.genome_sha256",
        )
        requested_counts[requested] += 1
        lineage_id = _positive_int(
            record.get("lineage_id"),
            field="record.lineage_id",
        )
        active_rows.append((requested, lineage_id))
        rewards_by_agent.setdefault(agent_id, []).append(reward_value)
        values_by_agent.setdefault(agent_id, []).append(value)
    if set(rewards_by_agent) != set(values_by_agent):
        raise OpenEcologySelectionError(
            "selection value targets differ from active agent histories"
        )
    for agent_id, passive_reward in passive_rewards.items():
        if agent_id in rewards_by_agent and passive_reward:
            rewards_by_agent[agent_id][-1] += passive_reward
    normalized_return, value_rmse, advantage_variance = _selection_metrics(
        rewards_by_agent=rewards_by_agent,
        values_by_agent=values_by_agent,
    )
    collapse = _action_collapse_evidence(active_rows)
    active_count = len(active_rows)
    dominant_action, dominant_count = _dominant_count(requested_counts)
    compact: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_SELECTION_RUN_SCHEMA_VERSION,
        "phase": task.phase,
        "environment_seed_index": task.environment_seed_index,
        "environment_seed": task.environment_seed,
        "initial_agents": task.initial_agents,
        "ticks_per_execution": task.ticks,
        "tape_identity": tape_identity,
        "action_selection": (
            PUBLIC_RECURRENT_ARGMAX_SELECTION
            if sampling_seed is None
            else PUBLIC_RECURRENT_SAMPLED_SELECTION
        ),
        "policy_sampling_seed": sampling_seed,
        "terminal": {
            "ticks_executed": _nonnegative_int(
                summary.get("ticks_executed"),
                field="summary.ticks_executed",
            ),
            "alive_agents": _nonnegative_int(
                summary.get("alive_agents"),
                field="summary.alive_agents",
            ),
            "births": _nonnegative_int(
                summary.get("births"),
                field="summary.births",
            ),
            "deaths": _nonnegative_int(
                summary.get("deaths"),
                field="summary.deaths",
            ),
            "extinct": _exact_bool(summary.get("extinct"), field="summary.extinct"),
        },
        "trajectory_record_count": len(records),
        "policy_decision_count": active_count,
        "passive_record_count": len(records) - active_count,
        "reward_total": round(reward_total, 12),
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_count": dominant_count,
        "dominant_requested_action_share": (
            round(dominant_count / active_count, 12) if active_count else 0.0
        ),
        "unsupported_requested_action_count": unsupported,
        "heuristic_action_source_count": sum(
            count
            for source, count in action_source_counts.items()
            if "heuristic" in source
        ),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
        "metrics": {
            "normalized_return": normalized_return,
            "heldout_value_rmse": value_rmse,
            "advantage_variance": advantage_variance,
            "target_contract": (
                "discounted_gamma_0.997_finite_world_zero_bootstrap_per_agent_v1"
            ),
        },
        "action_collapse": collapse,
        "genome_population_provenance": dict(genome_provenance),
        "gates": {
            "finite_outputs": finite_outputs,
            "exact_action_mask_legality": unsupported == 0,
            "no_heuristic_action_source": (
                all(
                    source in {RECURRENT_ROLLOUT_ACTION_SOURCE, "passive"}
                    for source in action_source_counts
                )
                and action_source_counts.get(
                    RECURRENT_ROLLOUT_ACTION_SOURCE,
                    0,
                )
                == active_count
            ),
            "no_global_action_collapse": collapse["collapsed"] is False,
        },
    }
    behavior_payload = {
        "task": {
            "phase": task.phase,
            "environment_seed_index": task.environment_seed_index,
            "environment_seed": task.environment_seed,
            "ticks": task.ticks,
            "initial_agents": task.initial_agents,
            "tape_identity": tape_identity,
            "sampling_seed": sampling_seed,
        },
        "summary": dict(summary),
        "trajectory_records": list(records),
        "policy_decision_diagnostics": list(diagnostics),
        "genome_population_provenance": dict(genome_provenance),
    }
    compact["full_behavior_sha256"] = stable_payload_digest(behavior_payload)
    compact["run_evidence_sha256"] = stable_payload_digest(compact)
    return compact


def _selection_metrics(
    *,
    rewards_by_agent: Mapping[int, Sequence[float]],
    values_by_agent: Mapping[int, Sequence[float]],
) -> tuple[float, float, float]:
    returns: list[float] = []
    values: list[float] = []
    normalized_individual_returns: list[float] = []
    for agent_id in sorted(rewards_by_agent):
        rewards = rewards_by_agent[agent_id]
        agent_values = values_by_agent[agent_id]
        if len(rewards) != len(agent_values):
            raise OpenEcologySelectionError(
                "selection rewards and values have different lengths"
            )
        normalized_individual_returns.append(
            sum(float(reward) for reward in rewards)
            / (len(rewards) * max(_REWARD_NORMALIZATION_SCALE, 1.0))
        )
        running = 0.0
        reversed_returns: list[float] = []
        for reward in reversed(rewards):
            running = float(reward) + OPEN_ECOLOGY_SELECTION_DISCOUNT * running
            reversed_returns.append(running)
        returns.extend(reversed(reversed_returns))
        values.extend(float(value) for value in agent_values)
    if not returns:
        return 0.0, 0.0, 0.0
    errors = [target - value for target, value in zip(returns, values, strict=True)]
    error_mean = sum(errors) / len(errors)
    return (
        round(median(normalized_individual_returns), 12),
        round(math.sqrt(sum(error * error for error in errors) / len(errors)), 12),
        round(
            sum((error - error_mean) ** 2 for error in errors) / len(errors),
            12,
        ),
    )


def _dominant_count(counts: Mapping[str, int]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    return min(
        ((action, -int(counts.get(action, 0))) for action in ACTION_NAMES),
        key=lambda item: (item[1], ACTION_NAMES.index(item[0])),
    )[0], max(int(counts.get(action, 0)) for action in ACTION_NAMES)


def _action_collapse_evidence(
    active_rows: Sequence[tuple[str, int]],
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    longest = 0
    current_action: str | None = None
    current_length = 0
    collapsed = False
    for start in range(
        0,
        len(active_rows) - OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW + 1,
        OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW,
    ):
        window = active_rows[
            start : start + OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW
        ]
        counts = Counter(action for action, _ in window)
        dominant, dominant_count = _dominant_count(counts)
        lineage_counts: dict[int, Counter[str]] = {}
        for action, lineage_id in window:
            lineage_counts.setdefault(lineage_id, Counter())[action] += 1
        represented = len(lineage_counts)
        lineage_dominant_count = sum(
            int(_dominant_count(lineage_action_counts)[0] == dominant)
            for lineage_action_counts in lineage_counts.values()
        )
        action_share = dominant_count / len(window)
        lineage_share = lineage_dominant_count / represented if represented else 0.0
        qualifies = (
            action_share >= OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_ACTION_SHARE
            and lineage_share >= OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_LINEAGE_SHARE
        )
        if qualifies and dominant == current_action:
            current_length += 1
        elif qualifies:
            current_action = dominant
            current_length = 1
        else:
            current_action = None
            current_length = 0
        longest = max(longest, current_length)
        collapsed = collapsed or (
            current_length >= OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_CONSECUTIVE_WINDOWS
        )
        summaries.append(
            {
                "window_index": len(summaries),
                "dominant_action": dominant,
                "dominant_action_share": round(action_share, 12),
                "represented_lineage_count": represented,
                "dominant_across_lineage_share": round(lineage_share, 12),
                "qualifies": qualifies,
            }
        )
    return {
        "schema_version": OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_VERSION,
        "window_decisions": OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW,
        "complete_window_count": len(summaries),
        "trailing_decision_count": (
            len(active_rows)
            - len(summaries) * OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW
        ),
        "longest_consecutive_qualifying_windows_same_action": longest,
        "collapsed": collapsed,
        "window_summaries_sha256": stable_payload_digest(summaries),
    }


def _causal_genome_contract() -> dict[str, object]:
    return {
        "schema_version": OPEN_ECOLOGY_SELECTION_CAUSAL_GENOME_SCHEMA_VERSION,
        "state_sampling_version": (
            OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_SAMPLING_VERSION
        ),
        "intervention_version": (OPEN_ECOLOGY_SELECTION_CAUSAL_INTERVENTION_VERSION),
        "states_per_environment": (
            OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
        ),
        "state_count_per_artifact": OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_COUNT,
        "state_source": "real_argmax_predecision_policy_states",
        "frozen_factors": [
            "observation",
            "action_mask",
            "previous_public_feedback",
            "recurrent_state",
            "model_parameters",
        ],
        "interventions": [
            "original_genome",
            "zero_genome",
            "deterministic_donor_genome",
            "single_locus_plus_or_minus_0.05",
        ],
        "donor_js_state_threshold": OPEN_ECOLOGY_SELECTION_DONOR_JS_STATE_THRESHOLD,
        "donor_js_share_threshold": OPEN_ECOLOGY_SELECTION_DONOR_JS_SHARE_THRESHOLD,
        "donor_js_mean_threshold": OPEN_ECOLOGY_SELECTION_DONOR_JS_MEAN_THRESHOLD,
        "single_locus_delta": OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_DELTA,
        "single_locus_tv_p99_max": (OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_TV_P99_MAX),
        "capture_adapter": {
            "public_snapshot_available": False,
            "read_only_protected_state_adapter": True,
            "ordinary_policy_source_bound_noninterference_proof_required": True,
            "noninterference_proof_schema_version": (
                OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SCHEMA_VERSION
            ),
            "per_artifact_third_argmax_world_required": False,
        },
    }


def _evaluate_causal_genome_states(
    model: PublicRecurrentActorCritic,
    states: Sequence[_CausalPolicyState],
    *,
    environment_seed: int,
) -> dict[str, object]:
    ordered = tuple(sorted(states, key=lambda state: state.identity_sha256))
    if len(ordered) != OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT:
        return {
            "schema_version": OPEN_ECOLOGY_SELECTION_CAUSAL_GENOME_SCHEMA_VERSION,
            "environment_seed": environment_seed,
            "state_count": len(ordered),
            "expected_state_count": (
                OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
            ),
            "complete": False,
            "state_rows": [],
            "state_manifest_sha256": stable_payload_digest([]),
        }
    donors: list[_CausalPolicyState] = []
    for index, state in enumerate(ordered):
        donor = next(
            (
                ordered[(index + offset) % len(ordered)]
                for offset in range(1, len(ordered))
                if ordered[(index + offset) % len(ordered)].genome_sha256
                != state.genome_sha256
            ),
            None,
        )
        if donor is None:
            raise OpenEcologySelectionError(
                "causal genome battery has no different donor genome"
            )
        donors.append(donor)
    perturbed_values: list[tuple[float, ...]] = []
    loci: list[int] = []
    deltas: list[float] = []
    for state in ordered:
        material = bytes.fromhex(state.identity_sha256)
        locus = int.from_bytes(material[:2], "big") % RECURRENT_CONTROLLER_GENOME_SIZE
        preferred = (
            OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_DELTA
            if material[2] % 2 == 0
            else -OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_DELTA
        )
        value = state.genome_values[locus]
        delta = preferred
        if not (
            RECURRENT_CONTROLLER_GENOME_MIN
            <= value + delta
            <= RECURRENT_CONTROLLER_GENOME_MAX
        ):
            delta = -preferred
        changed = list(state.genome_values)
        changed[locus] = round(value + delta, 8)
        genome = RecurrentControllerGenome(tuple(changed))
        perturbed_values.append(genome.values)
        loci.append(locus)
        deltas.append(delta)
    original = _causal_probabilities(
        model,
        ordered,
        genomes=tuple(state.genome_values for state in ordered),
    )
    zero_values = zero_recurrent_genome().values
    zero = _causal_probabilities(
        model,
        ordered,
        genomes=tuple(zero_values for _ in ordered),
    )
    donor = _causal_probabilities(
        model,
        ordered,
        genomes=tuple(state.genome_values for state in donors),
    )
    perturbed = _causal_probabilities(
        model,
        ordered,
        genomes=tuple(perturbed_values),
    )
    rows: list[dict[str, object]] = []
    for index, state in enumerate(ordered):
        if (
            max(
                abs(left - right)
                for left, right in zip(
                    original[index],
                    state.runtime_probabilities,
                    strict=True,
                )
            )
            > 2.0e-6
        ):
            raise OpenEcologySelectionError(
                "frozen causal state does not reproduce runtime probabilities"
            )
        rows.append(
            {
                "state_sha256": state.identity_sha256,
                "genome_sha256": state.genome_sha256,
                "donor_genome_sha256": donors[index].genome_sha256,
                "perturbed_locus": loci[index],
                "perturbed_delta": deltas[index],
                "original_vs_zero_js": round(
                    _jensen_shannon(original[index], zero[index]),
                    12,
                ),
                "original_vs_donor_js": round(
                    _jensen_shannon(original[index], donor[index]),
                    12,
                ),
                "single_locus_total_variation": round(
                    _total_variation(original[index], perturbed[index]),
                    12,
                ),
            }
        )
    return {
        "schema_version": OPEN_ECOLOGY_SELECTION_CAUSAL_GENOME_SCHEMA_VERSION,
        "environment_seed": environment_seed,
        "state_count": len(rows),
        "expected_state_count": (OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT),
        "complete": True,
        "state_rows": rows,
        "state_manifest_sha256": stable_payload_digest(rows),
    }


def _causal_probabilities(
    model: PublicRecurrentActorCritic,
    states: Sequence[_CausalPolicyState],
    *,
    genomes: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if len(states) != len(genomes) or not states:
        raise OpenEcologySelectionError(
            "causal intervention batch has invalid cardinality"
        )
    reference = next(model.parameters())
    observations = torch.tensor(
        [state.observation for state in states],
        device=reference.device,
        dtype=reference.dtype,
    ).unsqueeze(0)
    masks = torch.tensor(
        [state.action_mask for state in states],
        device=reference.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    feedback = torch.tensor(
        [state.previous_feedback for state in states],
        device=reference.device,
        dtype=reference.dtype,
    ).unsqueeze(0)
    genome_tensor = torch.tensor(
        genomes,
        device=reference.device,
        dtype=reference.dtype,
    ).unsqueeze(0)
    initial = torch.stack(
        [
            torch.tensor(
                state.recurrent_state,
                device=reference.device,
                dtype=reference.dtype,
            ).reshape(model.config.recurrent_layers, model.config.hidden_size)
            for state in states
        ],
        dim=1,
    )
    with torch.no_grad():
        output = model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=genome_tensor,
            initial_state=initial,
        )
        probabilities = torch.softmax(output.masked_logits[0], dim=-1)
    result: list[tuple[float, ...]] = []
    for row in probabilities.detach().cpu().tolist():
        parsed = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in parsed):
            raise OpenEcologySelectionError(
                "causal intervention produced non-finite probabilities"
            )
        result.append(parsed)
    return tuple(result)


def _jensen_shannon(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    midpoint = tuple((a + b) / 2.0 for a, b in zip(left, right, strict=True))

    def divergence(values: Sequence[float]) -> float:
        total = 0.0
        for value, middle in zip(values, midpoint, strict=True):
            if value > 0.0:
                total += value * math.log(value / middle)
        return total

    return 0.5 * (divergence(left) + divergence(right))


def _total_variation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise OpenEcologySelectionError("percentile requires non-empty values")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _aggregate_causal_genome_evidence(
    environments: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows: list[Mapping[str, object]] = []
    manifests: list[str] = []
    for environment in environments:
        causal = _mapping(
            environment.get("causal_genome"),
            field="environment.causal_genome",
        )
        manifests.append(
            _sha256(
                causal.get("state_manifest_sha256"),
                field="environment.causal_genome.state_manifest_sha256",
            )
        )
        raw_rows = causal.get("state_rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise OpenEcologySelectionError(
                "environment causal state rows must be a sequence"
            )
        rows.extend(_mapping(row, field="causal state row") for row in raw_rows)
    donor_js = [
        _finite(row.get("original_vs_donor_js"), field="original_vs_donor_js")
        for row in rows
    ]
    zero_js = [
        _finite(row.get("original_vs_zero_js"), field="original_vs_zero_js")
        for row in rows
    ]
    locus_tv = [
        _finite(
            row.get("single_locus_total_variation"),
            field="single_locus_total_variation",
        )
        for row in rows
    ]
    state_count = len(rows)
    donor_share = (
        sum(
            value >= OPEN_ECOLOGY_SELECTION_DONOR_JS_STATE_THRESHOLD
            for value in donor_js
        )
        / state_count
        if state_count
        else 0.0
    )
    donor_mean = sum(donor_js) / state_count if state_count else 0.0
    p99 = _nearest_rank(locus_tv, 0.99) if locus_tv else 0.0
    complete = state_count == OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_COUNT
    evidence: dict[str, object] = {
        "schema_version": OPEN_ECOLOGY_SELECTION_CAUSAL_GENOME_SCHEMA_VERSION,
        "state_count": state_count,
        "expected_state_count": OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_COUNT,
        "environment_count": len(environments),
        "complete": complete,
        "state_manifest_sha256": stable_payload_digest(
            {
                "environment_manifests": manifests,
                "state_rows": rows,
            }
        ),
        "original_vs_donor_js_share_at_least_0.01": round(donor_share, 12),
        "mean_original_vs_donor_js": round(donor_mean, 12),
        "mean_original_vs_zero_js": (
            round(sum(zero_js) / state_count, 12) if state_count else 0.0
        ),
        "single_locus_total_variation_p99": round(p99, 12),
        "gates": {
            "causal_genome_use": (
                complete
                and donor_share >= OPEN_ECOLOGY_SELECTION_DONOR_JS_SHARE_THRESHOLD
                and donor_mean >= OPEN_ECOLOGY_SELECTION_DONOR_JS_MEAN_THRESHOLD
            ),
            "bounded_single_locus_perturbation": (
                complete and p99 <= OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_TV_P99_MAX
            ),
        },
    }
    evidence["exact_digest"] = stable_payload_digest(evidence)
    return evidence


def _aggregate_stochastic_runs(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(runs) != OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT:
        raise OpenEcologySelectionError(
            "stochastic aggregate requires exactly four policy tapes"
        )
    metric_rows = [_mapping(run.get("metrics"), field="run.metrics") for run in runs]
    terminal_rows = [
        _mapping(run.get("terminal"), field="run.terminal") for run in runs
    ]
    aggregate: dict[str, object] = {
        "aggregation": "equal_weight_median_across_four_stochastic_tapes",
        "run_count": len(runs),
        "normalized_return": round(
            median(
                _finite(row.get("normalized_return"), field="normalized_return")
                for row in metric_rows
            ),
            12,
        ),
        "heldout_value_rmse": round(
            median(
                _finite(row.get("heldout_value_rmse"), field="heldout_value_rmse")
                for row in metric_rows
            ),
            12,
        ),
        "advantage_variance": round(
            median(
                _finite(
                    row.get("advantage_variance"),
                    field="advantage_variance",
                )
                for row in metric_rows
            ),
            12,
        ),
        "terminal_alive_median": round(
            median(
                _nonnegative_int(row.get("alive_agents"), field="alive_agents")
                for row in terminal_rows
            ),
            12,
        ),
        "births_median": round(
            median(
                _nonnegative_int(row.get("births"), field="births")
                for row in terminal_rows
            ),
            12,
        ),
        "run_evidence_sha256": stable_payload_digest(
            [run.get("run_evidence_sha256") for run in runs]
        ),
    }
    aggregate["exact_digest"] = stable_payload_digest(aggregate)
    return aggregate


def _selection_execution_accounting(
    environments: Sequence[Mapping[str, object]],
    *,
    workers_requested: int,
    workers_used: int,
) -> dict[str, object]:
    primary_count = sum(
        _nonnegative_int(
            environment.get("execution_count"),
            field="environment.execution_count",
        )
        for environment in environments
    )
    replay_count = sum(
        _nonnegative_int(
            environment.get("exact_replay_execution_count"),
            field="environment.exact_replay_execution_count",
        )
        for environment in environments
    )
    return {
        "environment_count": len(environments),
        "primary_evaluation_count": primary_count,
        "replay_evaluation_count": replay_count,
        "physical_world_run_count": primary_count + replay_count,
        "evaluation_workers_requested": _worker_count(workers_requested),
        "evaluation_workers_used": _worker_count(workers_used),
        "process_parallel": workers_used > 1,
        "ordered_collection": "increasing_local_environment_index",
        "numerical_runtime_contract": {
            "device": "cpu",
            "parameter_dtype": "torch.float32",
            "torch_num_threads_per_process": 1,
            "parent_torch_num_threads_during_evaluation": 1,
            "deterministic_full_behavior_pair_required": True,
        },
        "training_or_optimizer_steps": 0,
    }


def _authorization_execution_accounting(
    producer_execution: Mapping[str, object],
) -> dict[str, int]:
    primary = _nonnegative_int(
        producer_execution.get("primary_evaluation_count"),
        field="producer primary_evaluation_count",
    )
    replay = _nonnegative_int(
        producer_execution.get("replay_evaluation_count"),
        field="producer replay_evaluation_count",
    )
    physical = _nonnegative_int(
        producer_execution.get("physical_world_run_count"),
        field="producer physical_world_run_count",
    )
    expected = OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT * (
        OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT + 1
    )
    if primary != expected or replay != 0 or physical != expected:
        raise OpenEcologySelectionError(
            "authorization requires one provisional primary pass only"
        )
    return {
        "producer_primary_world_run_count": expected,
        "authorization_reexecution_world_run_count": expected,
        "total_physical_world_run_count_through_authorization": expected * 2,
    }


def _aggregate_learner_metrics(
    environments: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(environments) != OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT:
        raise OpenEcologySelectionError(
            "learner metrics require exactly 32 environment aggregates"
        )
    rows = [
        _mapping(
            environment.get("stochastic_tape_aggregate"),
            field="environment.stochastic_tape_aggregate",
        )
        for environment in environments
    ]
    return {
        "normalized_return": round(
            median(
                _finite(row.get("normalized_return"), field="normalized_return")
                for row in rows
            ),
            12,
        ),
        "heldout_value_rmse": round(
            median(
                _finite(row.get("heldout_value_rmse"), field="heldout_value_rmse")
                for row in rows
            ),
            12,
        ),
        "advantage_variance": round(
            median(
                _finite(
                    row.get("advantage_variance"),
                    field="advantage_variance",
                )
                for row in rows
            ),
            12,
        ),
    }


def _learner_gates(
    *,
    environments: Sequence[Mapping[str, object]],
    causal: Mapping[str, object],
) -> dict[str, object]:
    environment_gates = [
        _mapping(environment.get("gates"), field="environment.gates")
        for environment in environments
    ]
    causal_gates = _mapping(causal.get("gates"), field="causal.gates")
    gates: dict[str, object] = {
        name: all(gate.get(name) is True for gate in environment_gates)
        for name in (
            "finite_outputs",
            "exact_action_mask_legality",
            "exact_same_contract_replay",
            "no_heuristic_action_source",
            "no_global_action_collapse",
        )
    }
    gates["causal_genome_use"] = causal_gates.get("causal_genome_use") is True
    gates["bounded_single_locus_perturbation"] = (
        causal_gates.get("bounded_single_locus_perturbation") is True
    )
    gates["eligible"] = all(value is True for value in gates.values())
    return gates


def validate_open_ecology_selection_report(
    report: Mapping[str, object],
    *,
    artifact_path: str | Path,
    run_contract_path: str | Path,
) -> None:
    """Validate structure and byte bindings, never authorization by itself.

    Stable hashes are unkeyed integrity checks. Call
    Call
    :func:`evolution_sim.mind.open_ecology_phase_a.authorize_phase_a_terminal_selection_report`
    with the canonical terminal bundle before deriving learner evidence. A
    request or self-digested authority mapping is never sufficient.
    """

    _require_exact_keys(
        report,
        {
            "schema_version",
            "phase",
            "campaign_digest",
            "run_contract_digest",
            "cell_id",
            "learner_index",
            "learner_seed",
            "authority",
            "artifact",
            "contract",
            "execution",
            "environments",
            "causal_genome",
            "gates",
            "metrics",
            "lifecycle",
            "exact_digest",
        },
        field="selection report",
    )
    unsigned = dict(report)
    supplied_digest = _sha256(
        unsigned.pop("exact_digest"),
        field="selection report exact_digest",
    )
    if stable_payload_digest(unsigned) != supplied_digest:
        raise OpenEcologySelectionError(
            "selection report exact digest does not match its payload"
        )
    if report.get("schema_version") != OPEN_ECOLOGY_SELECTION_EVIDENCE_SCHEMA_VERSION:
        raise OpenEcologySelectionError("selection report schema differs")
    if dict(_mapping(report.get("authority"), field="authority")) != {
        "authoritative": False,
        "state": "provisional_primary_execution_only",
        "authorization_requires": (
            "independent_full_artifact_reexecution_and_exact_report_match"
        ),
    }:
        raise OpenEcologySelectionError(
            "selection producer report must remain explicitly provisional"
        )
    plan = OpenEcologySelectionPlan.for_phase(
        _selection_phase(report.get("phase"))  # type: ignore[arg-type]
    )
    campaign_digest = _sha256(
        report.get("campaign_digest"),
        field="campaign_digest",
    )
    run_contract_digest = _sha256(
        report.get("run_contract_digest"),
        field="run_contract_digest",
    )
    learner_index = _bounded_index(
        report.get("learner_index"),
        field="learner_index",
        upper=(
            _PHASE_A_LEARNER_COUNT
            if plan.phase is OpenEcologySelectionPhase.PHASE_A
            else _PHASE_B_LEARNER_COUNT
        ),
    )
    if (
        report.get("learner_seed")
        != OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][learner_index]
    ):
        raise OpenEcologySelectionError(
            "selection report learner seed differs from the registry"
        )
    artifact = _mapping(report.get("artifact"), field="artifact")
    _validate_artifact_binding(
        artifact,
        artifact_path=Path(artifact_path),
        run_contract_path=Path(run_contract_path),
        expected_campaign_digest=campaign_digest,
        expected_run_contract_digest=run_contract_digest,
        expected_cell_id=report.get("cell_id"),
        expected_learner_index=learner_index,
        expected_learner_seed=report.get("learner_seed"),
    )
    contract = _mapping(report.get("contract"), field="contract")
    _require_exact_keys(
        contract,
        {
            "metric_contract_version",
            "seed_registry_version",
            "seed_registry_sha256",
            "environment_seed_role",
            "environment_seed_indices",
            "environment_seeds",
            "ticks_per_execution",
            "stochastic_tape_identities",
            "argmax_diagnostic",
            "primary_metric_tapes",
            "aggregation_order",
            "density_cycle",
            "max_agents",
            "fixture_names",
            "carrion_objective",
            "genome_population_mode",
            "genome_stream_seed_index",
            "genome_stream_seed",
            "policy_sampling_seed_namespace",
            "exact_cpu_replay",
            "action_collapse_contract",
            "causal_genome_contract",
        },
        field="selection contract",
    )
    if (
        contract.get("metric_contract_version")
        != OPEN_ECOLOGY_SELECTION_METRIC_CONTRACT_VERSION
        or contract.get("seed_registry_version") != OPEN_ECOLOGY_SEED_REGISTRY_VERSION
        or contract.get("seed_registry_sha256") != OPEN_ECOLOGY_CANONICAL_SHA256
        or contract.get("environment_seed_role") != OPEN_ECOLOGY_SELECTION_SEED_ROLE
        or contract.get("environment_seed_indices")
        != list(plan.environment_seed_indices)
        or contract.get("environment_seeds") != list(plan.environment_seeds)
        or contract.get("ticks_per_execution") != plan.ticks_per_execution
        or contract.get("stochastic_tape_identities")
        != list(plan.stochastic_tape_identities)
        or contract.get("argmax_diagnostic") is not True
        or contract.get("primary_metric_tapes") != "four_stochastic_tapes_only"
        or contract.get("aggregation_order")
        != ("runs_within_environment_then_equal_weight_environments_within_learner")
        or contract.get("density_cycle") != list(plan.density_cycle)
        or contract.get("max_agents") != OPEN_ECOLOGY_MAX_AGENTS
        or contract.get("fixture_names") != []
        or contract.get("carrion_objective") is not False
        or contract.get("genome_population_mode")
        != RecurrentGenomePopulationMode.HERITABLE.value
        or contract.get("genome_stream_seed_index") != learner_index
        or contract.get("genome_stream_seed")
        != OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][learner_index]
        or contract.get("policy_sampling_seed_namespace")
        != RECURRENT_POLICY_SAMPLING_SEED_NAMESPACE
        or contract.get("exact_cpu_replay")
        != "independent_full_artifact_reexecution_required_for_authority"
        or dict(
            _mapping(
                contract.get("action_collapse_contract"),
                field="action_collapse_contract",
            )
        )
        != {
            "schema_version": (OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_VERSION),
            "window_decisions": (OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_WINDOW),
            "consecutive_windows": (
                OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_CONSECUTIVE_WINDOWS
            ),
            "requested_action_share_threshold": (
                OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_ACTION_SHARE
            ),
            "represented_lineage_share_threshold": (
                OPEN_ECOLOGY_SELECTION_ACTION_COLLAPSE_LINEAGE_SHARE
            ),
        }
        or dict(
            _mapping(
                contract.get("causal_genome_contract"),
                field="causal_genome_contract",
            )
        )
        != _causal_genome_contract()
    ):
        raise OpenEcologySelectionError(
            "selection report differs from its preregistered contract"
        )
    environments_value = report.get("environments")
    if not isinstance(environments_value, Sequence) or isinstance(
        environments_value,
        (str, bytes),
    ):
        raise OpenEcologySelectionError("selection environments must be a sequence")
    environments = tuple(
        _mapping(environment, field="environment") for environment in environments_value
    )
    if len(environments) != OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT:
        raise OpenEcologySelectionError(
            "selection report does not contain exactly 32 environments"
        )
    for local, environment in enumerate(environments):
        _validate_environment_evidence(environment)
        if (
            environment.get("local_environment_index") != local
            or environment.get("environment_seed_index")
            != plan.environment_seed_indices[local]
            or environment.get("environment_seed") != plan.environment_seeds[local]
            or environment.get("ticks_per_execution") != plan.ticks_per_execution
            or environment.get("initial_agents") != plan.initial_agents(local)
            or environment.get("exact_replay_execution_count") != 0
            or environment.get("phase") != plan.phase.value
            or environment.get("cell_id") != report.get("cell_id")
            or environment.get("learner_index") != learner_index
            or environment.get("artifact_sha256") != artifact.get("artifact_sha256")
        ):
            raise OpenEcologySelectionError(
                "selection environment differs from its canonical role"
            )
    expected_causal = _aggregate_causal_genome_evidence(environments)
    if (
        dict(_mapping(report.get("causal_genome"), field="causal_genome"))
        != expected_causal
    ):
        raise OpenEcologySelectionError(
            "selection causal aggregate does not match environment evidence"
        )
    expected_metrics = _aggregate_learner_metrics(environments)
    if dict(_mapping(report.get("metrics"), field="metrics")) != expected_metrics:
        raise OpenEcologySelectionError(
            "selection metrics do not match per-environment aggregation"
        )
    expected_gates = _learner_gates(
        environments=environments,
        causal=expected_causal,
    )
    if dict(_mapping(report.get("gates"), field="gates")) != expected_gates:
        raise OpenEcologySelectionError(
            "selection gates do not match underlying evidence"
        )
    execution = _mapping(report.get("execution"), field="execution")
    if (
        execution.get("environment_count") != OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT
        or execution.get("primary_evaluation_count")
        != OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT
        * (OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT + 1)
        or execution.get("replay_evaluation_count") != 0
        or execution.get("physical_world_run_count")
        != OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT
        * (OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT + 1)
        or dict(
            _mapping(
                execution.get("numerical_runtime_contract"),
                field="execution.numerical_runtime_contract",
            )
        )
        != {
            "device": "cpu",
            "parameter_dtype": "torch.float32",
            "torch_num_threads_per_process": 1,
            "parent_torch_num_threads_during_evaluation": 1,
            "deterministic_full_behavior_pair_required": True,
        }
        or execution.get("training_or_optimizer_steps") != 0
    ):
        raise OpenEcologySelectionError("selection execution accounting is incomplete")
    lifecycle = _mapping(report.get("lifecycle"), field="lifecycle")
    if dict(lifecycle) != {
        "development_only": True,
        "training_performed": False,
        "validation_available": False,
        "validation_accessed": False,
        "lockbox_available": False,
        "lockbox_accessed": False,
        "runtime_integration_authorized": False,
        "promotion_authorized": False,
        "society_claim_authorized": False,
    }:
        raise OpenEcologySelectionError("selection report lifecycle boundary differs")


def _validate_artifact_binding(
    binding: Mapping[str, object],
    *,
    artifact_path: Path,
    run_contract_path: Path,
    expected_campaign_digest: str,
    expected_run_contract_digest: str,
    expected_cell_id: object,
    expected_learner_index: int,
    expected_learner_seed: object,
) -> None:
    _require_exact_keys(
        binding,
        {
            "schema_version",
            "logical_name",
            "file_sha256",
            "artifact_schema_version",
            "artifact_sha256",
            "model_state_sha256",
            "source_commit",
            "source_manifest_sha256",
            "seed_registry_sha256",
            "campaign_digest",
            "run_contract_digest",
            "run_contract_logical_name",
            "run_contract_file_sha256",
            "cell_id",
            "learner_index",
            "learner_seed",
            "training_authority",
            "model_contract",
            "exact_digest",
        },
        field="artifact binding",
    )
    unsigned = dict(binding)
    binding_digest = _sha256(
        unsigned.pop("exact_digest", None),
        field="artifact binding exact_digest",
    )
    if stable_payload_digest(unsigned) != binding_digest:
        raise OpenEcologySelectionError("artifact binding digest changed")
    if (
        binding.get("schema_version")
        != OPEN_ECOLOGY_SELECTION_ARTIFACT_BINDING_SCHEMA_VERSION
        or binding.get("campaign_digest") != expected_campaign_digest
        or binding.get("run_contract_digest") != expected_run_contract_digest
        or binding.get("cell_id") != expected_cell_id
        or binding.get("learner_index") != expected_learner_index
        or binding.get("learner_seed") != expected_learner_seed
        or binding.get("seed_registry_sha256") != OPEN_ECOLOGY_CANONICAL_SHA256
    ):
        raise OpenEcologySelectionError("artifact binding context changed")
    if (
        not run_contract_path.is_file()
        or run_contract_path.is_symlink()
        or binding.get("run_contract_file_sha256") != _file_sha256(run_contract_path)
    ):
        raise OpenEcologySelectionError(
            "bound run-contract bytes changed after evaluation"
        )
    run_contract = _load_strict_json_mapping(
        run_contract_path,
        field="bound run contract",
    )
    run_contract_unsigned = dict(run_contract)
    if (
        run_contract_unsigned.pop("exact_digest", None) != expected_run_contract_digest
        or stable_payload_digest(run_contract_unsigned) != expected_run_contract_digest
    ):
        raise OpenEcologySelectionError(
            "bound run-contract digest changed after evaluation"
        )
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise OpenEcologySelectionError(
            "bound artifact path is not a regular non-symlink file"
        )
    if binding.get("file_sha256") != _file_sha256(artifact_path):
        raise OpenEcologySelectionError(
            "selection artifact bytes changed after evaluation"
        )
    training_authority = binding.get("training_authority")
    if training_authority is not None:
        _validate_training_authority_binding(
            _mapping(
                training_authority,
                field="artifact training_authority",
            ),
            expected_campaign_digest=expected_campaign_digest,
            expected_run_contract_digest=expected_run_contract_digest,
            expected_artifact_sha256=_sha256(
                binding.get("artifact_sha256"),
                field="artifact_sha256",
            ),
            expected_artifact_file_sha256=_file_sha256(artifact_path),
            expected_source_commit=_source_commit(binding.get("source_commit")),
            expected_source_manifest_sha256=_sha256(
                binding.get("source_manifest_sha256"),
                field="source_manifest_sha256",
            ),
            expected_cell_id=expected_cell_id,
            expected_learner_index=expected_learner_index,
            expected_learner_seed=expected_learner_seed,
        )
    schema = binding.get("artifact_schema_version")
    try:
        loaded = (
            load_frozen_recurrent_policy_artifact(artifact_path)
            if schema == FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION
            else load_recurrent_artifact(artifact_path)
            if schema == RECURRENT_ARTIFACT_SCHEMA_VERSION
            else None
        )
    except RecurrentArtifactError as exc:
        raise OpenEcologySelectionError(
            "bound selection artifact no longer reconstructs"
        ) from exc
    if loaded is None:
        raise OpenEcologySelectionError("artifact binding schema is unsupported")
    if loaded.artifact.get("artifact_sha256") != binding.get(
        "artifact_sha256"
    ) or recurrent_model_state_sha256(loaded.model) != binding.get(
        "model_state_sha256"
    ):
        raise OpenEcologySelectionError("bound selection artifact identity changed")
    config = loaded.model.config
    if dict(
        _mapping(binding.get("model_contract"), field="artifact model_contract")
    ) != {
        "encoder_size": config.encoder_size,
        "hidden_size": config.hidden_size,
        "recurrent_layers": config.recurrent_layers,
        "public_input_schema_version": config.public_input_schema_version,
        "public_input_size": config.public_input_size,
        "genome_conditioning_mode": config.genome_conditioning_mode,
        "critic_genome_conditioning": config.critic_genome_conditioning,
        "value_shared_trunk_gradient": config.value_shared_trunk_gradient,
    }:
        raise OpenEcologySelectionError("artifact model contract binding changed")
    provenance = _mapping(
        loaded.artifact.get("provenance"),
        field="bound artifact provenance",
    )
    run = _mapping(
        provenance.get("run_metadata"),
        field="bound artifact run metadata",
    )
    source_manifest = (
        provenance.get("source_manifest_sha256")
        if schema == FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION
        else run.get("source_manifest_sha256")
    )
    if provenance.get("source_commit") != binding.get(
        "source_commit"
    ) or source_manifest != binding.get("source_manifest_sha256"):
        raise OpenEcologySelectionError("bound selection artifact source pins changed")


def _validate_environment_evidence(
    environment: Mapping[str, object],
) -> None:
    _require_exact_keys(
        environment,
        {
            "schema_version",
            "phase",
            "cell_id",
            "learner_index",
            "artifact_sha256",
            "local_environment_index",
            "environment_seed_index",
            "environment_seed",
            "ticks_per_execution",
            "initial_agents",
            "max_agents",
            "genome_stream_seed",
            "execution_count",
            "exact_replay_execution_count",
            "executions",
            "stochastic_tape_aggregate",
            "causal_genome",
            "gates",
            "exact_digest",
        },
        field="environment",
    )
    unsigned = dict(environment)
    supplied = _sha256(
        unsigned.pop("exact_digest", None),
        field="environment exact_digest",
    )
    if stable_payload_digest(unsigned) != supplied:
        raise OpenEcologySelectionError("environment exact digest changed")
    if (
        environment.get("schema_version")
        != OPEN_ECOLOGY_SELECTION_ENVIRONMENT_SCHEMA_VERSION
        or environment.get("execution_count")
        != OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT + 1
        or environment.get("exact_replay_execution_count")
        not in {0, OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT + 1}
        or environment.get("max_agents") != OPEN_ECOLOGY_MAX_AGENTS
    ):
        raise OpenEcologySelectionError("environment execution evidence is incomplete")
    executions_value = environment.get("executions")
    if not isinstance(executions_value, Sequence) or isinstance(
        executions_value,
        (str, bytes),
    ):
        raise OpenEcologySelectionError("environment executions must be a sequence")
    executions = tuple(
        _mapping(execution, field="execution") for execution in executions_value
    )
    if len(executions) != OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT + 1:
        raise OpenEcologySelectionError(
            "environment does not contain four tapes plus argmax"
        )
    expected_tapes = (
        tuple(
            f"phase-a-selection-tape-{index:02d}"
            for index in range(OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT)
        )
        if environment.get("phase") == OpenEcologySelectionPhase.PHASE_A.value
        else tuple(
            f"phase-b-selection-tape-{index:02d}"
            for index in range(OPEN_ECOLOGY_SELECTION_STOCHASTIC_TAPE_COUNT)
        )
    ) + ("argmax",)
    for execution, expected_tape in zip(executions, expected_tapes, strict=True):
        _require_exact_keys(
            execution,
            {
                "schema_version",
                "phase",
                "environment_seed_index",
                "environment_seed",
                "initial_agents",
                "ticks_per_execution",
                "tape_identity",
                "action_selection",
                "policy_sampling_seed",
                "terminal",
                "trajectory_record_count",
                "policy_decision_count",
                "passive_record_count",
                "reward_total",
                "requested_action_counts",
                "dominant_requested_action",
                "dominant_requested_action_count",
                "dominant_requested_action_share",
                "unsupported_requested_action_count",
                "heuristic_action_source_count",
                "action_source_counts",
                "policy_id_counts",
                "metrics",
                "action_collapse",
                "genome_population_provenance",
                "gates",
                "full_behavior_sha256",
                "run_evidence_sha256",
                "exact_replay",
                "exact_digest",
            },
            field="execution",
        )
        supplied_execution = _sha256(
            execution.get("exact_digest"),
            field="execution exact_digest",
        )
        execution_unsigned = dict(execution)
        execution_unsigned.pop("exact_digest")
        if stable_payload_digest(execution_unsigned) != supplied_execution:
            raise OpenEcologySelectionError("execution exact digest changed")
        if execution.get("tape_identity") != expected_tape:
            raise OpenEcologySelectionError("execution tape identity changed")
        if (
            execution.get("schema_version") != OPEN_ECOLOGY_SELECTION_RUN_SCHEMA_VERSION
            or execution.get("phase") != environment.get("phase")
            or execution.get("environment_seed_index")
            != environment.get("environment_seed_index")
            or execution.get("environment_seed") != environment.get("environment_seed")
            or execution.get("initial_agents") != environment.get("initial_agents")
            or execution.get("ticks_per_execution")
            != environment.get("ticks_per_execution")
        ):
            raise OpenEcologySelectionError(
                "execution context differs from its environment"
            )
        expected_selection = (
            PUBLIC_RECURRENT_ARGMAX_SELECTION
            if expected_tape == "argmax"
            else PUBLIC_RECURRENT_SAMPLED_SELECTION
        )
        if execution.get("action_selection") != expected_selection:
            raise OpenEcologySelectionError(
                "execution action selection differs from its tape"
            )
        if expected_tape == "argmax":
            if execution.get("policy_sampling_seed") is not None:
                raise OpenEcologySelectionError(
                    "argmax execution cannot consume a policy RNG tape"
                )
        else:
            expected_sampling_seed = _policy_sampling_seed(
                _EnvironmentTask(
                    phase=str(environment["phase"]),
                    cell_id=str(environment["cell_id"]),
                    learner_index=int(environment["learner_index"]),
                    artifact_sha256=str(environment["artifact_sha256"]),
                    local_environment_index=int(environment["local_environment_index"]),
                    environment_seed_index=int(environment["environment_seed_index"]),
                    environment_seed=int(environment["environment_seed"]),
                    ticks=int(environment["ticks_per_execution"]),
                    initial_agents=int(environment["initial_agents"]),
                    stochastic_tape_identities=expected_tapes[:-1],
                    genome_stream_seed=0,
                ),
                tape_identity=expected_tape,
            )
            if execution.get("policy_sampling_seed") != expected_sampling_seed:
                raise OpenEcologySelectionError(
                    "execution sampling seed differs from its exact tape identity"
                )
        run_evidence_unsigned = dict(execution)
        run_evidence_sha = _sha256(
            run_evidence_unsigned.pop("run_evidence_sha256"),
            field="run_evidence_sha256",
        )
        run_evidence_unsigned.pop("exact_replay")
        run_evidence_unsigned.pop("exact_digest")
        if stable_payload_digest(run_evidence_unsigned) != run_evidence_sha:
            raise OpenEcologySelectionError("run evidence digest changed")
        replay = _mapping(execution.get("exact_replay"), field="exact_replay")
        _require_exact_keys(
            replay,
            {
                "verified",
                "independent_artifact_reload",
                "replay_full_behavior_sha256",
                "replay_run_evidence_sha256",
            },
            field="exact_replay",
        )
        replay_verified = _exact_bool(
            replay.get("verified"),
            field="exact_replay.verified",
        )
        if replay_verified:
            if (
                replay.get("independent_artifact_reload") is not True
                or replay.get("replay_full_behavior_sha256")
                != execution.get("full_behavior_sha256")
                or replay.get("replay_run_evidence_sha256")
                != execution.get("run_evidence_sha256")
            ):
                raise OpenEcologySelectionError(
                    "execution exact replay evidence does not match"
                )
        elif (
            replay.get("independent_artifact_reload") is not False
            or replay.get("replay_full_behavior_sha256") is not None
            or replay.get("replay_run_evidence_sha256") is not None
        ):
            raise OpenEcologySelectionError(
                "provisional execution cannot claim replay evidence"
            )
        run_gates = _mapping(execution.get("gates"), field="execution.gates")
        _require_exact_keys(
            run_gates,
            {
                "finite_outputs",
                "exact_action_mask_legality",
                "no_heuristic_action_source",
                "no_global_action_collapse",
            },
            field="execution.gates",
        )
        collapse = _mapping(
            execution.get("action_collapse"),
            field="execution.action_collapse",
        )
        policy_decision_count = _nonnegative_int(
            execution.get("policy_decision_count"),
            field="execution.policy_decision_count",
        )
        passive_record_count = _nonnegative_int(
            execution.get("passive_record_count"),
            field="execution.passive_record_count",
        )
        if (
            execution.get("trajectory_record_count")
            != policy_decision_count + passive_record_count
        ):
            raise OpenEcologySelectionError("execution trajectory accounting changed")
        action_source_counts = _mapping(
            execution.get("action_source_counts"),
            field="execution.action_source_counts",
        )
        parsed_source_counts = {
            str(source): _nonnegative_int(
                count,
                field=f"execution.action_source_counts.{source}",
            )
            for source, count in action_source_counts.items()
        }
        if (
            set(parsed_source_counts) - {RECURRENT_ROLLOUT_ACTION_SOURCE, "passive"}
            or parsed_source_counts.get(RECURRENT_ROLLOUT_ACTION_SOURCE, 0)
            != policy_decision_count
            or parsed_source_counts.get("passive", 0) != passive_record_count
            or sum(parsed_source_counts.values())
            != policy_decision_count + passive_record_count
        ):
            raise OpenEcologySelectionError(
                "execution action sources are not exact recurrent/passive evidence"
            )
        requested_counts = _mapping(
            execution.get("requested_action_counts"),
            field="execution.requested_action_counts",
        )
        if set(requested_counts) - set(ACTION_NAMES):
            raise OpenEcologySelectionError(
                "execution requested-action counts contain unknown actions"
            )
        parsed_requested_counts = {
            action: _nonnegative_int(
                requested_counts.get(action, 0),
                field=f"execution.requested_action_counts.{action}",
            )
            for action in ACTION_NAMES
        }
        dominant_action, dominant_count = _dominant_count(parsed_requested_counts)
        if (
            sum(parsed_requested_counts.values()) != policy_decision_count
            or execution.get("dominant_requested_action") != dominant_action
            or execution.get("dominant_requested_action_count") != dominant_count
            or execution.get("dominant_requested_action_share")
            != (
                round(dominant_count / policy_decision_count, 12)
                if policy_decision_count
                else 0.0
            )
        ):
            raise OpenEcologySelectionError(
                "execution requested-action summary changed"
            )
        if (
            run_gates.get("exact_action_mask_legality")
            is not (execution.get("unsupported_requested_action_count") == 0)
            or run_gates.get("no_heuristic_action_source")
            is not (execution.get("heuristic_action_source_count") == 0)
            or run_gates.get("no_global_action_collapse")
            is not (collapse.get("collapsed") is False)
        ):
            raise OpenEcologySelectionError(
                "execution gates differ from their evidence"
            )
        reconstructed_task = _EnvironmentTask(
            phase=str(environment["phase"]),
            cell_id=str(environment["cell_id"]),
            learner_index=int(environment["learner_index"]),
            artifact_sha256=str(environment["artifact_sha256"]),
            local_environment_index=int(environment["local_environment_index"]),
            environment_seed_index=int(environment["environment_seed_index"]),
            environment_seed=int(environment["environment_seed"]),
            ticks=int(environment["ticks_per_execution"]),
            initial_agents=int(environment["initial_agents"]),
            stochastic_tape_identities=expected_tapes[:-1],
            genome_stream_seed=int(environment["genome_stream_seed"]),
        )
        _validate_genome_world_provenance(
            _mapping(
                execution.get("genome_population_provenance"),
                field="execution.genome_population_provenance",
            ),
            expected_world_identity=_world_identity(reconstructed_task),
            expected_genome_stream_seed=reconstructed_task.genome_stream_seed,
            expected_action_selection=expected_selection,
            expected_policy_sampling_seed=execution.get("policy_sampling_seed"),  # type: ignore[arg-type]
        )
    if environment.get("exact_replay_execution_count") != sum(
        int(
            _mapping(execution["exact_replay"], field="exact_replay")["verified"]
            is True
        )
        for execution in executions
    ):
        raise OpenEcologySelectionError(
            "environment replay accounting differs from executions"
        )
    expected_aggregate = _aggregate_stochastic_runs(executions[:4])
    if (
        dict(
            _mapping(
                environment.get("stochastic_tape_aggregate"),
                field="stochastic_tape_aggregate",
            )
        )
        != expected_aggregate
    ):
        raise OpenEcologySelectionError(
            "environment stochastic aggregate does not match its tapes"
        )
    causal = _mapping(
        environment.get("causal_genome"),
        field="environment.causal_genome",
    )
    _validate_environment_causal_evidence(causal)
    expected_environment_gates = {
        "finite_outputs": all(
            _mapping(execution["gates"], field="execution.gates")["finite_outputs"]
            is True
            for execution in executions
        ),
        "exact_action_mask_legality": all(
            _mapping(execution["gates"], field="execution.gates")[
                "exact_action_mask_legality"
            ]
            is True
            for execution in executions
        ),
        "exact_same_contract_replay": all(
            _mapping(execution["exact_replay"], field="exact_replay")["verified"]
            is True
            for execution in executions
        ),
        "no_heuristic_action_source": all(
            _mapping(execution["gates"], field="execution.gates")[
                "no_heuristic_action_source"
            ]
            is True
            for execution in executions
        ),
        "no_global_action_collapse": all(
            _mapping(execution["gates"], field="execution.gates")[
                "no_global_action_collapse"
            ]
            is True
            for execution in executions
        ),
    }
    if (
        dict(_mapping(environment.get("gates"), field="environment.gates"))
        != expected_environment_gates
    ):
        raise OpenEcologySelectionError(
            "environment gates differ from execution evidence"
        )


def _validate_environment_causal_evidence(
    causal: Mapping[str, object],
) -> None:
    _require_exact_keys(
        causal,
        {
            "schema_version",
            "environment_seed",
            "state_count",
            "expected_state_count",
            "complete",
            "state_rows",
            "state_manifest_sha256",
        },
        field="environment causal evidence",
    )
    if (
        causal.get("schema_version")
        != OPEN_ECOLOGY_SELECTION_CAUSAL_GENOME_SCHEMA_VERSION
        or causal.get("expected_state_count")
        != OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
    ):
        raise OpenEcologySelectionError("environment causal evidence contract changed")
    rows_value = causal.get("state_rows")
    if not isinstance(rows_value, Sequence) or isinstance(
        rows_value,
        (str, bytes),
    ):
        raise OpenEcologySelectionError("causal state rows must be a sequence")
    rows = tuple(_mapping(row, field="causal state row") for row in rows_value)
    if causal.get("state_count") != len(rows):
        raise OpenEcologySelectionError("causal state count differs from its rows")
    if causal.get("complete") is not (
        len(rows) == OPEN_ECOLOGY_SELECTION_CAUSAL_STATES_PER_ENVIRONMENT
    ):
        raise OpenEcologySelectionError(
            "causal completeness differs from its state count"
        )
    for row in rows:
        _require_exact_keys(
            row,
            {
                "state_sha256",
                "genome_sha256",
                "donor_genome_sha256",
                "perturbed_locus",
                "perturbed_delta",
                "original_vs_zero_js",
                "original_vs_donor_js",
                "single_locus_total_variation",
            },
            field="causal state row",
        )
        _sha256(row.get("state_sha256"), field="causal state_sha256")
        genome_sha = _sha256(
            row.get("genome_sha256"),
            field="causal genome_sha256",
        )
        donor_sha = _sha256(
            row.get("donor_genome_sha256"),
            field="causal donor_genome_sha256",
        )
        if genome_sha == donor_sha:
            raise OpenEcologySelectionError(
                "causal donor intervention did not change genome"
            )
        _bounded_index(
            row.get("perturbed_locus"),
            field="causal perturbed_locus",
            upper=RECURRENT_CONTROLLER_GENOME_SIZE,
        )
        if (
            abs(_finite(row.get("perturbed_delta"), field="causal perturbed_delta"))
            != OPEN_ECOLOGY_SELECTION_SINGLE_LOCUS_DELTA
        ):
            raise OpenEcologySelectionError(
                "causal single-locus perturbation magnitude changed"
            )
        for field in (
            "original_vs_zero_js",
            "original_vs_donor_js",
            "single_locus_total_variation",
        ):
            value = _finite(row.get(field), field=f"causal {field}")
            if value < 0.0 or value > 1.0:
                raise OpenEcologySelectionError(f"causal {field} lies outside [0, 1]")
    if causal.get("state_manifest_sha256") != stable_payload_digest(rows):
        raise OpenEcologySelectionError("causal state manifest does not match its rows")
