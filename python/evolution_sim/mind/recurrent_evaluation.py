from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import multiprocessing
from pathlib import Path
import platform
from random import Random
from threading import Lock
from typing import Iterator

import torch

from evolution_sim.config import WorldConfig
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.runtime.trajectory import REWARD_COMPONENT_BOUNDS
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.evaluation_harness import CONTROLLED_FIXTURE_NAMES
from evolution_sim.mind.evaluation_helpers import dominant_action_summary, round_float
from evolution_sim.mind.recurrent_evaluation_contract import (
    RECURRENT_EVALUATION_SCHEMA_VERSION,
    RecurrentEvaluationError,
)
from evolution_sim.mind.recurrent_artifact import (
    LoadedFrozenRecurrentPolicyArtifact,
    LoadedRecurrentArtifact,
    load_frozen_recurrent_policy_artifact,
    load_recurrent_artifact,
)
from evolution_sim.mind.recurrent_actor_critic import PublicRecurrentActorCritic
from evolution_sim.mind.recurrent_policy import (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION,
    PUBLIC_RECURRENT_POLICY_ID,
    PUBLIC_RECURRENT_POLICY_VERSION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
    RECURRENT_COUNTERFACTUAL_ACTION_SOURCE,
    DeterministicPublicRecurrentPolicy,
    frozen_cpu_model_copy,
)
from evolution_sim.mind.recurrent_seed_registry import (
    CANONICAL_SEED_REGISTRY_SHA256,
    RECURRENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


RECURRENT_EVALUATION_EXECUTION_SCHEMA_VERSION = (
    "mind_public_recurrent_evaluation_execution_v2"
)
RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION = (
    "mind_public_recurrent_evaluation_runtime_reproducibility_v1"
)
RECURRENT_EVALUATION_TICKS = 120
MASKED_RANDOM_POLICY_ID = "mind_v3_masked_random_control"
MASKED_RANDOM_POLICY_VERSION = "mind_v3_masked_random_control_v1"
MASKED_RANDOM_ACTION_SOURCE = "masked_random_control"
_MASKED_RANDOM_SEED_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-evaluation|masked-random-v1"
)
_PUBLIC_RECURRENT_SAMPLING_SEED_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-evaluation|artifact-sampling-v1"
)
UNPINNED_NONCANDIDATE_DIGEST_PREFIX = "unpinned-noncandidate-development-canary:"
RECURRENT_EVALUATION_DEVELOPMENT_SEED_ROLE = "development"
RECURRENT_EVALUATION_SELECTION_SEED_ROLE = "selection"
RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE = "scale_selection"
RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE = "scale_v2_selection"
RECURRENT_EVALUATION_CANDIDATE_SEED_ROLE = "candidate"
RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE = "lockbox"
_EVALUATION_SEED_ROLE_TO_REGISTRY_ROLE = {
    RECURRENT_EVALUATION_SELECTION_SEED_ROLE: "selection",
    RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE: "scale_selection",
    RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE: "scale_v2_selection",
    RECURRENT_EVALUATION_CANDIDATE_SEED_ROLE: "validation",
    RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE: "lockbox",
}
_CANONICAL_TRAINING_SEED_ROLES = ("train", "curriculum")
_CANONICAL_TRAINING_SEEDS = frozenset(
    seed
    for role in _CANONICAL_TRAINING_SEED_ROLES
    for seed in RECURRENT_SEED_REGISTRY[role]
)
_SCALE_TRAINING_SEED_ROLES = ("scale_train", "scale_curriculum")
_SCALE_TRAINING_SEEDS = frozenset(
    seed
    for role in _SCALE_TRAINING_SEED_ROLES
    for seed in SCALE_DEVELOPMENT_SEED_REGISTRY[role]
)
_SCALE_V2_TRAINING_SEED_ROLES = ("scale_v2_train", "scale_v2_curriculum")
_SCALE_V2_TRAINING_SEEDS = frozenset(
    seed
    for role in _SCALE_V2_TRAINING_SEED_ROLES
    for seed in SCALE_DEVELOPMENT_V2_SEED_REGISTRY[role]
)
_POLICY_KEYS = ("public_recurrent", "mind_v3_linear", "masked_random")
_DISTRIBUTION_METRICS = (
    "entropy",
    "normalized_entropy",
    "selected_action_probability",
    "top_action_probability",
    "top_two_probability_margin",
    "eat_probability",
)
_RUN_OUTCOME_EVIDENCE_SCHEMA_VERSION = (
    "mind_public_recurrent_evaluation_run_outcome_evidence_v1"
)
_RUN_FIELDS = {
    "context",
    "seed",
    "policy_sampling_seed",
    "horizon_ticks",
    "ticks_executed",
    "terminal_alive",
    "births",
    "deaths",
    "reward_total",
    "reward_component_totals",
    "trajectory_record_count",
    "policy_decision_record_count",
    "passive_trajectory_record_count",
    "requested_action_counts",
    "dominant_requested_action",
    "dominant_requested_action_count",
    "dominant_requested_action_share",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
    "action_source_counts",
    "policy_id_counts",
    "eat_requested_count",
    "eat_without_positive_resource_gain_count",
    "eat_without_positive_resource_gain_share",
    "learned_masked_distribution",
    "behavior_digest",
    "replay_digest",
    "outcome_evidence_sha256",
}
_RUN_DIGEST_FIELDS = {
    "behavior_digest",
    "replay_digest",
    "outcome_evidence_sha256",
}
_LEARNED_DISTRIBUTION_SUMMARY_FIELDS = {
    "decision_count",
    "metric_observation_counts",
    "metric_means",
    "metric_minima",
    "metric_maxima",
    "mean_action_probabilities",
}
_AMBIGUOUS_V3_TERMINAL_FIELDS = frozenset(
    {
        "carrion_fixture_terminal_survivor_count",
        "terminal_survivor_run_count",
        "terminal_survival_probability",
        "terminal_survival_probability_wilson_95",
    }
)
_TORCH_THREAD_SCOPE_LOCK = Lock()
_PARALLEL_EVALUATION_MODEL: PublicRecurrentActorCritic | None = None


@dataclass(frozen=True, slots=True)
class _EvaluationEnvironmentTask:
    """One independently replay-verifiable environment evaluation unit."""

    seed: int
    fixture_name: str | None
    policy_digest_label: str
    feed_forward_history_ablation: bool
    candidate_action_selection: str
    sampling_seeds: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class RecurrentEvaluationSeedPlan:
    """Evaluation seeds plus an explicit lifecycle role.

    ``excluded_training_seeds`` is retained as a caller assertion for development
    diagnostics. It is never evidence for artifact or promotion eligibility.
    Role-bound plans must use one canonical ordered role subset. Lockbox access
    remains closed because this tree has no external one-use authorization ledger.
    """

    broad_seeds: Sequence[int]
    fixture_seeds: Sequence[int]
    excluded_training_seeds: Sequence[int]
    environment_seed_role: str = RECURRENT_EVALUATION_DEVELOPMENT_SEED_ROLE

    def __post_init__(self) -> None:
        broad = _validated_seed_sequence(self.broad_seeds, field="broad_seeds")
        fixture = _validated_seed_sequence(self.fixture_seeds, field="fixture_seeds")
        excluded = _validated_seed_sequence(
            self.excluded_training_seeds,
            field="excluded_training_seeds",
        )
        object.__setattr__(self, "broad_seeds", broad)
        object.__setattr__(self, "fixture_seeds", fixture)
        object.__setattr__(self, "excluded_training_seeds", excluded)
        role = _validated_evaluation_seed_role(self.environment_seed_role)
        object.__setattr__(self, "environment_seed_role", role)
        holdout = set(broad) | set(fixture)
        overlap = sorted(holdout & set(excluded))
        if overlap:
            raise RecurrentEvaluationError(
                "evaluation seeds overlap excluded training seeds: "
                + ", ".join(str(seed) for seed in overlap)
            )
        registry_role = _EVALUATION_SEED_ROLE_TO_REGISTRY_ROLE.get(role)
        if registry_role is None:
            reserved_seed_sets = (
                RECURRENT_SEED_REGISTRY["selection"],
                RECURRENT_SEED_REGISTRY["validation"],
                RECURRENT_SEED_REGISTRY["lockbox"],
                SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"],
                SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_selection"],
            )
            reserved_overlap = sorted(
                holdout
                & {
                    seed
                    for reserved_seeds in reserved_seed_sets
                    for seed in reserved_seeds
                }
            )
            if reserved_overlap:
                raise RecurrentEvaluationError(
                    "development evaluation cannot consume reserved selection, "
                    "validation, or lockbox seeds; declare the canonical role"
                )
        else:
            if role == RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE:
                expected = SCALE_DEVELOPMENT_SEED_REGISTRY[registry_role]
            elif role == RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE:
                expected = SCALE_DEVELOPMENT_V2_SEED_REGISTRY[registry_role]
            else:
                expected = RECURRENT_SEED_REGISTRY[registry_role]
            if (
                broad != fixture
                or not _is_canonical_ordered_subset(broad, expected)
                or not _is_canonical_ordered_subset(fixture, expected)
            ):
                raise RecurrentEvaluationError(
                    f"{role} evaluation seeds must be the same non-empty canonical "
                    f"ordered subset of the {registry_role} role for broad and "
                    "fixture contexts"
                )

    @property
    def canonical_registry_role(self) -> str | None:
        return _EVALUATION_SEED_ROLE_TO_REGISTRY_ROLE.get(self.environment_seed_role)

    @property
    def full_canonical_lockbox_plan(self) -> bool:
        return (
            self.environment_seed_role == RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE
            and self.broad_seeds == RECURRENT_SEED_REGISTRY["lockbox"]
            and self.fixture_seeds == RECURRENT_SEED_REGISTRY["lockbox"]
        )

    @property
    def contains_lockbox_seed(self) -> bool:
        return bool(
            (set(self.broad_seeds) | set(self.fixture_seeds))
            & set(RECURRENT_SEED_REGISTRY["lockbox"])
        )

    @property
    def contains_validation_seed(self) -> bool:
        return bool(
            (set(self.broad_seeds) | set(self.fixture_seeds))
            & set(RECURRENT_SEED_REGISTRY["validation"])
        )

    @property
    def digest(self) -> str:
        payload = {
            "broad_seeds": list(self.broad_seeds),
            "fixture_seeds": list(self.fixture_seeds),
            "excluded_training_seeds": list(self.excluded_training_seeds),
            "environment_seed_role": self.environment_seed_role,
        }
        return _canonical_sha256(payload)


class MaskedRandomPolicy:
    """No-heuristic random control constrained by the exact current action mask."""

    policy_id = MASKED_RANDOM_POLICY_ID
    policy_version = MASKED_RANDOM_POLICY_VERSION

    def __init__(self, *, world_seed: int) -> None:
        seed = _validated_seed(world_seed, field="world_seed")
        digest = hashlib.sha256(
            f"{_MASKED_RANDOM_SEED_NAMESPACE}|{seed}".encode("ascii")
        ).digest()
        self._rng = Random(int.from_bytes(digest[:8], byteorder="big"))
        self._decision_index = 0

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        del observation
        valid_actions = _strict_valid_actions(action_mask)
        requested_action = self._rng.choice(valid_actions)
        decision_index = self._decision_index
        self._decision_index += 1
        return ActionDecision(
            requested_action=requested_action,
            source=MASKED_RANDOM_ACTION_SOURCE,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics={
                "schema_version": "mind_v3_masked_random_decision_v1",
                "decision_index": decision_index,
                "valid_action_count": len(valid_actions),
            },
        )


def evaluate_recurrent_artifact(
    artifact_path: str | Path,
    *,
    seed_plan: RecurrentEvaluationSeedPlan,
    expected_source_commit: str | None = None,
    fixture_names: Sequence[str] = CONTROLLED_FIXTURE_NAMES,
    candidate_action_selection: str = PUBLIC_RECURRENT_ARGMAX_SELECTION,
    candidate_sampling_seed_count: int = 1,
    candidate_sampling_stream_id: str | None = None,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    """Evaluate one verified frozen artifact on broad and controlled holdouts.

    Every candidate run is replayed from a new policy and world instance. Exact
    replay mismatch, invalid requested action, heuristic fallback, or stale
    artifact aborts evaluation. Artifact-bound train/evaluation overlap or
    incomplete seed provenance makes the report promotion-ineligible while
    retaining it as a diagnostic.
    """

    resolved_workers = _validated_evaluation_workers(evaluation_workers)
    if not isinstance(seed_plan, RecurrentEvaluationSeedPlan):
        raise TypeError("seed_plan must be a RecurrentEvaluationSeedPlan")
    _reject_unavailable_restricted_evaluation_access(seed_plan)
    selected_fixtures = _validated_fixture_names(fixture_names)
    resolved_expected_source_commit = _validated_expected_source_commit(
        expected_source_commit
    )
    loaded = load_recurrent_artifact(artifact_path)
    artifact_digest = _artifact_digest(loaded)
    provenance = _required_mapping(
        loaded.artifact.get("provenance"),
        field="artifact.provenance",
    )
    seed_registry_digest = provenance.get("seed_registry_digest")
    if seed_registry_digest != CANONICAL_SEED_REGISTRY_SHA256:
        raise RecurrentEvaluationError(
            "artifact seed registry does not match the canonical registry"
        )
    artifact_source_commit = provenance.get("source_commit")
    if artifact_source_commit != resolved_expected_source_commit:
        raise RecurrentEvaluationError(
            "artifact source commit does not match the external expected pin"
        )
    selection = _validated_candidate_action_selection(candidate_action_selection)
    training_seed_evidence = _artifact_training_seed_evidence(
        provenance,
        seed_plan=seed_plan,
    )
    promotion_eligible = False
    feed_forward_history_ablation = _artifact_feed_forward_history_ablation(loaded)
    return _evaluate_frozen_model(
        candidate_model=loaded.model,
        policy_digest_label=artifact_digest,
        seed_plan=seed_plan,
        selected_fixtures=selected_fixtures,
        artifact_evidence={
            "path": str(Path(artifact_path)),
            "artifact_sha256": artifact_digest,
            "model_contract_version": _artifact_model_contract(loaded),
            "seed_registry_digest": seed_registry_digest,
            "source_commit": artifact_source_commit,
            "expected_source_commit": resolved_expected_source_commit,
            "source_commit_match": True,
            "training_seed_evidence": training_seed_evidence,
        },
        candidate_provenance={
            "mode": "verified_artifact_path",
            "source_pinned": True,
            "synthetic_digest_label": False,
            "noncandidate_development_canary": False,
            "promotion_evidence_eligible_from_provenance": promotion_eligible,
            "promotion_eligibility_requires_external_one_use_lockbox_authorization": (
                True
            ),
            "external_one_use_lockbox_authorization_available": False,
            "research_candidate_evidence_requires_external_validation_authorization": (
                True
            ),
            "external_validation_authorization_available": False,
            "full_canonical_lockbox_plan": seed_plan.full_canonical_lockbox_plan,
            "contains_canonical_validation_seed": seed_plan.contains_validation_seed,
            "contains_canonical_lockbox_seed": seed_plan.contains_lockbox_seed,
            "artifact_training_seed_evidence": training_seed_evidence,
            "caller_excluded_training_seeds_used_as_proof": False,
            "candidate_action_selection_promotion_eligible": (
                selection == PUBLIC_RECURRENT_ARGMAX_SELECTION
            ),
            "sampled_candidate_diagnostic": (
                selection == PUBLIC_RECURRENT_SAMPLED_SELECTION
            ),
            "feed_forward_history_ablation": feed_forward_history_ablation,
            "pin_verification": (
                "canonical_seed_registry_and_external_expected_source_commit"
            ),
        },
        feed_forward_history_ablation=feed_forward_history_ablation,
        candidate_action_selection=selection,
        candidate_sampling_seed_count=candidate_sampling_seed_count,
        candidate_sampling_stream_id=candidate_sampling_stream_id,
        evaluation_workers=resolved_workers,
    )


def evaluate_frozen_recurrent_policy_artifact(
    artifact_path: str | Path,
    *,
    seed_plan: RecurrentEvaluationSeedPlan,
    expected_source_commit: str,
    expected_source_manifest_sha256: str,
    expected_seed_registry_digest: str,
    fixture_names: Sequence[str] = CONTROLLED_FIXTURE_NAMES,
    candidate_action_selection: str = PUBLIC_RECURRENT_ARGMAX_SELECTION,
    candidate_sampling_seed_count: int = 1,
    candidate_sampling_stream_id: str | None = None,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    """Evaluate a v2 frozen policy after strict source and registry pin checks."""

    resolved_workers = _validated_evaluation_workers(evaluation_workers)
    if not isinstance(seed_plan, RecurrentEvaluationSeedPlan):
        raise TypeError("seed_plan must be a RecurrentEvaluationSeedPlan")
    _reject_unavailable_restricted_evaluation_access(seed_plan)
    selected_fixtures = _validated_fixture_names(fixture_names)
    resolved_source_commit = _validated_expected_source_commit(expected_source_commit)
    resolved_source_manifest = _validated_sha256(
        expected_source_manifest_sha256,
        field="expected source manifest SHA256",
    )
    resolved_registry_digest = _validated_sha256(
        expected_seed_registry_digest,
        field="expected seed registry digest",
    )
    loaded = load_frozen_recurrent_policy_artifact(artifact_path)
    artifact_digest = _artifact_digest(loaded)
    provenance = _required_mapping(
        loaded.artifact.get("provenance"),
        field="artifact.provenance",
    )
    integrity = _required_mapping(
        loaded.artifact.get("integrity"),
        field="artifact.integrity",
    )
    artifact_registry_digest = provenance.get("seed_registry_digest")
    if (
        artifact_registry_digest != resolved_registry_digest
        or integrity.get("seed_registry_digest") != resolved_registry_digest
    ):
        raise RecurrentEvaluationError(
            "frozen artifact seed registry does not match the expected pin"
        )
    if resolved_registry_digest not in {
        CANONICAL_SEED_REGISTRY_SHA256,
        SCALE_DEVELOPMENT_CANONICAL_SHA256,
        SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
    }:
        raise RecurrentEvaluationError("unsupported frozen artifact seed registry")
    if (
        seed_plan.environment_seed_role
        == RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE
        and resolved_registry_digest != SCALE_DEVELOPMENT_CANONICAL_SHA256
    ):
        raise RecurrentEvaluationError(
            "scale selection evaluation requires the scale seed registry"
        )
    if (
        seed_plan.environment_seed_role
        == RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE
        and resolved_registry_digest != SCALE_DEVELOPMENT_V2_CANONICAL_SHA256
    ):
        raise RecurrentEvaluationError(
            "scale-v2 selection evaluation requires the scale-v2 seed registry"
        )
    artifact_source_commit = provenance.get("source_commit")
    artifact_source_manifest = provenance.get("source_manifest_sha256")
    if artifact_source_commit != resolved_source_commit:
        raise RecurrentEvaluationError(
            "frozen artifact source commit does not match the expected pin"
        )
    if (
        artifact_source_manifest != resolved_source_manifest
        or integrity.get("source_manifest_sha256") != resolved_source_manifest
    ):
        raise RecurrentEvaluationError(
            "frozen artifact source manifest does not match the expected pin"
        )
    selection = _validated_candidate_action_selection(candidate_action_selection)
    training_seed_evidence = _artifact_training_seed_evidence(
        provenance,
        seed_plan=seed_plan,
    )
    feed_forward_history_ablation = _artifact_feed_forward_history_ablation(loaded)
    return _evaluate_frozen_model(
        candidate_model=loaded.model,
        policy_digest_label=artifact_digest,
        seed_plan=seed_plan,
        selected_fixtures=selected_fixtures,
        artifact_evidence={
            "path": str(Path(artifact_path)),
            "artifact_sha256": artifact_digest,
            "artifact_kind": loaded.artifact.get("artifact_kind"),
            "model_contract_version": _artifact_model_contract(loaded),
            "seed_registry_digest": artifact_registry_digest,
            "source_commit": artifact_source_commit,
            "expected_source_commit": resolved_source_commit,
            "source_commit_match": True,
            "source_manifest_sha256": artifact_source_manifest,
            "expected_source_manifest_sha256": resolved_source_manifest,
            "source_manifest_match": True,
            "training_seed_evidence": training_seed_evidence,
        },
        candidate_provenance={
            "mode": "verified_frozen_policy_artifact_v2",
            "source_pinned": True,
            "synthetic_digest_label": False,
            "noncandidate_development_canary": False,
            "promotion_evidence_eligible_from_provenance": False,
            "promotion_eligibility_requires_external_one_use_lockbox_authorization": (
                True
            ),
            "external_one_use_lockbox_authorization_available": False,
            "research_candidate_evidence_requires_external_validation_authorization": (
                True
            ),
            "external_validation_authorization_available": False,
            "full_canonical_lockbox_plan": seed_plan.full_canonical_lockbox_plan,
            "contains_canonical_validation_seed": seed_plan.contains_validation_seed,
            "contains_canonical_lockbox_seed": seed_plan.contains_lockbox_seed,
            "artifact_training_seed_evidence": training_seed_evidence,
            "caller_excluded_training_seeds_used_as_proof": False,
            "candidate_action_selection_promotion_eligible": (
                selection == PUBLIC_RECURRENT_ARGMAX_SELECTION
            ),
            "sampled_candidate_diagnostic": (
                selection == PUBLIC_RECURRENT_SAMPLED_SELECTION
            ),
            "feed_forward_history_ablation": feed_forward_history_ablation,
            "pin_verification": (
                "frozen_v2_registry_source_commit_source_manifest_and_cpu_probe"
            ),
        },
        feed_forward_history_ablation=feed_forward_history_ablation,
        candidate_action_selection=selection,
        candidate_sampling_seed_count=candidate_sampling_seed_count,
        candidate_sampling_stream_id=candidate_sampling_stream_id,
        evaluation_workers=resolved_workers,
    )


def evaluate_recurrent_model(
    model: PublicRecurrentActorCritic,
    *,
    synthetic_noncandidate_digest_label: str,
    seed_plan: RecurrentEvaluationSeedPlan,
    fixture_names: Sequence[str] = CONTROLLED_FIXTURE_NAMES,
    feed_forward_history_ablation: bool = False,
    candidate_action_selection: str = PUBLIC_RECURRENT_ARGMAX_SELECTION,
    candidate_sampling_seed_count: int = 1,
    candidate_sampling_stream_id: str | None = None,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    """Evaluate an in-memory development model without claiming source pinning.

    This path exists for dirty-tree canaries between optimizer updates and must
    never be treated as candidate, promotion, runtime-integration, or durable
    artifact evidence. The caller-supplied label is explicitly namespaced as an
    unpinned noncandidate and is used only to identify deterministic policy logs.
    """

    resolved_workers = _validated_evaluation_workers(evaluation_workers)
    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    if not isinstance(seed_plan, RecurrentEvaluationSeedPlan):
        raise TypeError("seed_plan must be a RecurrentEvaluationSeedPlan")
    _reject_unavailable_restricted_evaluation_access(seed_plan)
    if type(feed_forward_history_ablation) is not bool:
        raise TypeError("feed_forward_history_ablation must be an exact boolean")
    selection = _validated_candidate_action_selection(candidate_action_selection)
    selected_fixtures = _validated_fixture_names(fixture_names)
    digest_label = _validated_unpinned_digest_label(synthetic_noncandidate_digest_label)
    return _evaluate_frozen_model(
        candidate_model=model,
        policy_digest_label=digest_label,
        seed_plan=seed_plan,
        selected_fixtures=selected_fixtures,
        artifact_evidence=None,
        candidate_provenance={
            "mode": "in_memory_unpinned_development_canary",
            "source_pinned": False,
            "synthetic_digest_label": True,
            "synthetic_noncandidate_digest_label": digest_label,
            "noncandidate_development_canary": True,
            "promotion_evidence_eligible_from_provenance": False,
            "promotion_eligibility_requires_external_one_use_lockbox_authorization": (
                True
            ),
            "external_one_use_lockbox_authorization_available": False,
            "research_candidate_evidence_requires_external_validation_authorization": (
                True
            ),
            "external_validation_authorization_available": False,
            "full_canonical_lockbox_plan": seed_plan.full_canonical_lockbox_plan,
            "contains_canonical_validation_seed": seed_plan.contains_validation_seed,
            "contains_canonical_lockbox_seed": seed_plan.contains_lockbox_seed,
            "artifact_training_seed_evidence": None,
            "caller_excluded_training_seeds_used_as_proof": False,
            "candidate_action_selection_promotion_eligible": False,
            "sampled_candidate_diagnostic": (
                selection == PUBLIC_RECURRENT_SAMPLED_SELECTION
            ),
            "feed_forward_history_ablation": feed_forward_history_ablation,
            "runtime_integration_authorized": False,
        },
        feed_forward_history_ablation=feed_forward_history_ablation,
        candidate_action_selection=selection,
        candidate_sampling_seed_count=candidate_sampling_seed_count,
        candidate_sampling_stream_id=candidate_sampling_stream_id,
        evaluation_workers=resolved_workers,
    )


def _evaluate_frozen_model(
    *,
    candidate_model: PublicRecurrentActorCritic,
    policy_digest_label: str,
    seed_plan: RecurrentEvaluationSeedPlan,
    selected_fixtures: Sequence[str],
    artifact_evidence: Mapping[str, object] | None,
    candidate_provenance: Mapping[str, object],
    feed_forward_history_ablation: bool,
    candidate_action_selection: str,
    candidate_sampling_seed_count: int,
    candidate_sampling_stream_id: str | None,
    evaluation_workers: int = 1,
) -> dict[str, object]:
    resolved_workers = _validated_evaluation_workers(evaluation_workers)
    with _single_torch_thread_scope():
        return _evaluate_frozen_model_impl(
            candidate_model=candidate_model,
            policy_digest_label=policy_digest_label,
            seed_plan=seed_plan,
            selected_fixtures=selected_fixtures,
            artifact_evidence=artifact_evidence,
            candidate_provenance=candidate_provenance,
            feed_forward_history_ablation=feed_forward_history_ablation,
            candidate_action_selection=candidate_action_selection,
            candidate_sampling_seed_count=candidate_sampling_seed_count,
            candidate_sampling_stream_id=candidate_sampling_stream_id,
            evaluation_workers=resolved_workers,
        )


def _evaluate_frozen_model_impl(
    *,
    candidate_model: PublicRecurrentActorCritic,
    policy_digest_label: str,
    seed_plan: RecurrentEvaluationSeedPlan,
    selected_fixtures: Sequence[str],
    artifact_evidence: Mapping[str, object] | None,
    candidate_provenance: Mapping[str, object],
    feed_forward_history_ablation: bool,
    candidate_action_selection: str,
    candidate_sampling_seed_count: int,
    candidate_sampling_stream_id: str | None,
    evaluation_workers: int,
) -> dict[str, object]:
    frozen_model = frozen_cpu_model_copy(candidate_model)
    resolved_sampling_stream_id = (
        policy_digest_label
        if candidate_sampling_stream_id is None
        else _validated_sampling_stream_id(candidate_sampling_stream_id)
    )
    sampling_seeds = _candidate_sampling_seeds(
        resolved_sampling_stream_id,
        candidate_action_selection=candidate_action_selection,
        count=candidate_sampling_seed_count,
    )
    replay_checks: list[dict[str, object]] = []

    context_specs = (
        (None, seed_plan.broad_seeds),
        *(
            (fixture_name, seed_plan.fixture_seeds)
            for fixture_name in selected_fixtures
        ),
    )
    all_tasks = tuple(
        _EvaluationEnvironmentTask(
            seed=seed,
            fixture_name=fixture_name,
            policy_digest_label=policy_digest_label,
            feed_forward_history_ablation=feed_forward_history_ablation,
            candidate_action_selection=candidate_action_selection,
            sampling_seeds=tuple(sampling_seeds),
        )
        for fixture_name, seeds in context_specs
        for seed in seeds
    )
    parallel_results_by_context: dict[str | None, tuple[Mapping[str, object], ...]] = {}
    evaluation_workers_used = 1
    if evaluation_workers > 1:
        evaluation_workers_used = min(evaluation_workers, len(all_tasks))
        all_results = _run_parallel_environment_tasks(
            all_tasks,
            candidate_model=frozen_model,
            evaluation_workers=evaluation_workers_used,
        )
        cursor = 0
        for fixture_name, seeds in context_specs:
            next_cursor = cursor + len(seeds)
            parallel_results_by_context[fixture_name] = all_results[cursor:next_cursor]
            cursor = next_cursor
        if cursor != len(all_results):
            raise RecurrentEvaluationError(
                "parallel evaluation result partition is inconsistent"
            )

    broad = _evaluate_context(
        candidate_model=frozen_model,
        policy_digest_label=policy_digest_label,
        seeds=seed_plan.broad_seeds,
        fixture_name=None,
        replay_checks=replay_checks,
        feed_forward_history_ablation=feed_forward_history_ablation,
        candidate_action_selection=candidate_action_selection,
        sampling_seeds=sampling_seeds,
        environment_results=(
            parallel_results_by_context[None] if evaluation_workers > 1 else None
        ),
    )
    fixtures = [
        {
            "fixture": fixture_name,
            **_evaluate_context(
                candidate_model=frozen_model,
                policy_digest_label=policy_digest_label,
                seeds=seed_plan.fixture_seeds,
                fixture_name=fixture_name,
                replay_checks=replay_checks,
                feed_forward_history_ablation=feed_forward_history_ablation,
                candidate_action_selection=candidate_action_selection,
                sampling_seeds=sampling_seeds,
                environment_results=(
                    parallel_results_by_context[fixture_name]
                    if evaluation_workers > 1
                    else None
                ),
            ),
        }
        for fixture_name in selected_fixtures
    ]
    carrion = next(
        (fixture for fixture in fixtures if fixture["fixture"] == "carrion_only"),
        None,
    )
    carrion_terminal_nonextinct_run_count = 0
    carrion_terminal_alive_agent_total = 0
    if isinstance(carrion, Mapping):
        policies = _required_mapping(carrion.get("policies"), field="policies")
        candidate = _required_mapping(
            policies.get("public_recurrent"),
            field="policies.public_recurrent",
        )
        aggregate = _required_mapping(
            candidate.get("aggregate"),
            field="policies.public_recurrent.aggregate",
        )
        carrion_terminal_nonextinct_run_count = int(
            aggregate.get("terminal_nonextinct_run_count", 0)
        )
        carrion_terminal_alive_agent_total = int(
            aggregate.get("terminal_alive_agent_total", 0)
        )

    if not replay_checks or not all(
        check.get("passed") is True for check in replay_checks
    ):
        raise RecurrentEvaluationError("candidate replay verification is incomplete")

    report = {
        "schema_version": RECURRENT_EVALUATION_SCHEMA_VERSION,
        "artifact": dict(artifact_evidence) if artifact_evidence is not None else None,
        "candidate_provenance": dict(candidate_provenance),
        "execution_provenance": {
            "schema_version": RECURRENT_EVALUATION_EXECUTION_SCHEMA_VERSION,
            "evaluation_workers_requested": evaluation_workers,
            "evaluation_workers_used": evaluation_workers_used,
            "process_parallel": evaluation_workers > 1,
            "process_start_method": ("spawn" if evaluation_workers > 1 else None),
            "torch_threads_per_worker": 1,
            "environment_task_count": len(all_tasks),
            "task_unit": (
                "one_environment_controls_once_plus_candidate_streams_and_exact_replays"
            ),
            "ordered_collection": "context_then_seed_input_order",
            "failure_policy": "raise_without_sequential_fallback",
            "runtime_reproducibility": _runtime_reproducibility_provenance(
                source_model=candidate_model,
                evaluation_model=frozen_model,
            ),
        },
        "evaluation_contract": {
            "ticks": RECURRENT_EVALUATION_TICKS,
            "world_horizon_policy": "exact_configured_120_tick_horizon",
            "candidate_action_selection": candidate_action_selection,
            "candidate_sampling_seeds": [
                seed for seed in sampling_seeds if seed is not None
            ],
            "candidate_sampling_seed_count": len(sampling_seeds),
            "candidate_sampling_stream_id": resolved_sampling_stream_id,
            "candidate_sampling_stream_id_source": (
                "candidate_policy_digest_default"
                if candidate_sampling_stream_id is None
                else "caller_supplied_arm_independent_identifier"
            ),
            "candidate_sampling_seed_contract": (
                "none_argmax"
                if sampling_seeds == (None,)
                else (
                    "sha256_namespace_sampling_stream_id_and_replicate_index_"
                    "first_63_bits"
                )
            ),
            "candidate_recurrent_state": (
                "zero_before_every_decision"
                if feed_forward_history_ablation
                else "one_hidden_state_per_agent"
            ),
            "candidate_policy_inputs": [
                "current_public_ecological_observation",
                "current_public_action_mask",
                "same_agent_previous_public_outcome",
                "same_agent_recurrent_state",
            ],
            "candidate_forbidden_inputs": [
                "world_seed",
                "fixture_name",
                "private_world_state",
                "evaluation_split_role",
            ],
            "candidate_factory_receives_seed_or_fixture": False,
            "candidate_replay_verification": "every_run_exact_digest_repeat",
            "strict_zero_unsupported_requested_actions": True,
            "strict_zero_heuristic_candidate_actions": True,
        },
        "seed_plan": {
            "source": "caller_supplied_evaluation_plan",
            "digest": seed_plan.digest,
            "environment_seed_role": seed_plan.environment_seed_role,
            "canonical_registry_role": seed_plan.canonical_registry_role,
            "canonical_role_exact": seed_plan.canonical_registry_role is not None,
            "full_canonical_lockbox_plan": seed_plan.full_canonical_lockbox_plan,
            "contains_canonical_validation_seed": seed_plan.contains_validation_seed,
            "contains_canonical_lockbox_seed": seed_plan.contains_lockbox_seed,
            "broad_seeds": list(seed_plan.broad_seeds),
            "fixture_seeds": list(seed_plan.fixture_seeds),
            "caller_excluded_training_seeds": list(seed_plan.excluded_training_seeds),
            "excluded_training_seed_count": len(seed_plan.excluded_training_seeds),
            "caller_excluded_training_seeds_used_as_proof": False,
            "canonical_training_seed_count": len(
                _training_seed_set_for_plan(seed_plan)
            ),
            "train_evaluation_overlap_count": len(
                (set(seed_plan.broad_seeds) | set(seed_plan.fixture_seeds))
                & _training_seed_set_for_plan(seed_plan)
            ),
        },
        "broad": broad,
        "fixtures": fixtures,
        "carrion_fixture_terminal_nonextinct_run_count": (
            carrion_terminal_nonextinct_run_count
        ),
        "carrion_fixture_terminal_alive_agent_total": (
            carrion_terminal_alive_agent_total
        ),
        "replay_verification": {
            "all_passed": True,
            "checked_run_count": len(replay_checks),
            "checks": replay_checks,
        },
    }
    _validate_report(report)
    return report


def _evaluate_context(
    *,
    candidate_model: PublicRecurrentActorCritic,
    policy_digest_label: str,
    seeds: Sequence[int],
    fixture_name: str | None,
    replay_checks: list[dict[str, object]],
    feed_forward_history_ablation: bool,
    candidate_action_selection: str,
    sampling_seeds: Sequence[int | None],
    environment_results: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    runs_by_policy: dict[str, list[dict[str, object]]] = {
        key: [] for key in _POLICY_KEYS
    }
    paired_control_runs: dict[str, list[dict[str, object]]] = {
        "mind_v3_linear": [],
        "masked_random": [],
    }
    tasks = tuple(
        _EvaluationEnvironmentTask(
            seed=seed,
            fixture_name=fixture_name,
            policy_digest_label=policy_digest_label,
            feed_forward_history_ablation=feed_forward_history_ablation,
            candidate_action_selection=candidate_action_selection,
            sampling_seeds=tuple(sampling_seeds),
        )
        for seed in seeds
    )
    resolved_results = (
        tuple(
            _evaluate_environment_task(task, candidate_model=candidate_model)
            for task in tasks
        )
        if environment_results is None
        else tuple(environment_results)
    )
    if len(resolved_results) != len(tasks):
        raise RecurrentEvaluationError(
            "evaluation environment result count differs from the task count"
        )

    for task, result in zip(tasks, resolved_results, strict=True):
        if (
            result.get("seed") != task.seed
            or result.get("fixture_name") != task.fixture_name
        ):
            raise RecurrentEvaluationError(
                "evaluation environment result identity or order drifted"
            )
        linear = dict(
            _required_mapping(result.get("mind_v3_linear"), field="mind_v3_linear")
        )
        masked_random = dict(
            _required_mapping(result.get("masked_random"), field="masked_random")
        )
        runs_by_policy["mind_v3_linear"].append(linear)
        runs_by_policy["masked_random"].append(masked_random)
        candidates = result.get("public_recurrent")
        checks = result.get("replay_checks")
        if (
            isinstance(candidates, (str, bytes))
            or not isinstance(candidates, Sequence)
            or isinstance(checks, (str, bytes))
            or not isinstance(checks, Sequence)
            or len(candidates) != len(task.sampling_seeds)
            or len(checks) != len(task.sampling_seeds)
        ):
            raise RecurrentEvaluationError(
                "candidate or replay result count differs from sampling streams"
            )
        for sampling_seed, candidate_value, check_value in zip(
            task.sampling_seeds,
            candidates,
            checks,
            strict=True,
        ):
            candidate = dict(
                _required_mapping(candidate_value, field="public_recurrent run")
            )
            check = dict(_required_mapping(check_value, field="replay check"))
            if (
                candidate.get("policy_sampling_seed") != sampling_seed
                or check.get("policy_sampling_seed") != sampling_seed
                or check.get("seed") != task.seed
                or check.get("context") != _context_label(task.fixture_name)
                or check.get("passed") is not True
            ):
                raise RecurrentEvaluationError(
                    "candidate or replay result provenance drifted"
                )
            replay_checks.append(check)
            runs_by_policy["public_recurrent"].append(candidate)
            paired_control_runs["mind_v3_linear"].append(linear)
            paired_control_runs["masked_random"].append(masked_random)

    policies = {
        key: {
            "runs": runs,
            "aggregate": _aggregate_runs(runs),
        }
        for key, runs in runs_by_policy.items()
    }
    candidate_runs = runs_by_policy["public_recurrent"]
    return {
        "seeds": list(seeds),
        "candidate_sampling_seeds": [
            seed for seed in sampling_seeds if seed is not None
        ],
        "candidate_run_count_per_environment": len(sampling_seeds),
        "controls_run_once_per_environment": True,
        "candidate_sampling_analysis": _candidate_sampling_analysis(candidate_runs),
        "policies": policies,
        "paired_deltas": {
            "candidate_minus_mind_v3_linear": _paired_deltas(
                candidate_runs,
                paired_control_runs["mind_v3_linear"],
            ),
            "candidate_minus_masked_random": _paired_deltas(
                candidate_runs,
                paired_control_runs["masked_random"],
            ),
        },
    }


def _evaluate_environment_task(
    task: _EvaluationEnvironmentTask,
    *,
    candidate_model: PublicRecurrentActorCritic,
) -> dict[str, object]:
    linear = _run_policy_world(
        seed=task.seed,
        fixture_name=task.fixture_name,
        policy=MindV3EvolutionPolicy(seed=task.seed),
    )
    masked_random = _run_policy_world(
        seed=task.seed,
        fixture_name=task.fixture_name,
        policy=MaskedRandomPolicy(world_seed=task.seed),
    )
    candidate_runs: list[dict[str, object]] = []
    replay_checks: list[dict[str, object]] = []
    for sampling_seed in task.sampling_seeds:
        candidate = _run_policy_world(
            seed=task.seed,
            fixture_name=task.fixture_name,
            policy=_new_candidate_policy(
                candidate_model,
                task.policy_digest_label,
                feed_forward_history_ablation=(task.feed_forward_history_ablation),
                candidate_action_selection=task.candidate_action_selection,
                sampling_seed=sampling_seed,
            ),
            policy_sampling_seed=sampling_seed,
        )
        repeated = _run_policy_world(
            seed=task.seed,
            fixture_name=task.fixture_name,
            policy=_new_candidate_policy(
                candidate_model,
                task.policy_digest_label,
                feed_forward_history_ablation=(task.feed_forward_history_ablation),
                candidate_action_selection=task.candidate_action_selection,
                sampling_seed=sampling_seed,
            ),
            policy_sampling_seed=sampling_seed,
        )
        if candidate != repeated:
            raise RecurrentEvaluationError(
                "candidate replay mismatch for "
                f"{_context_label(task.fixture_name)} seed {task.seed} "
                f"policy sampling seed {sampling_seed}"
            )
        candidate_runs.append(candidate)
        replay_checks.append(
            {
                "context": _context_label(task.fixture_name),
                "seed": task.seed,
                "policy_sampling_seed": sampling_seed,
                "digest": candidate["replay_digest"],
                "outcome_evidence_sha256": candidate[
                    "outcome_evidence_sha256"
                ],
                "passed": True,
            }
        )
    return {
        "seed": task.seed,
        "fixture_name": task.fixture_name,
        "mind_v3_linear": linear,
        "masked_random": masked_random,
        "public_recurrent": candidate_runs,
        "replay_checks": replay_checks,
    }


@contextmanager
def _single_torch_thread_scope() -> Iterator[None]:
    """Make sequential and spawned CPU inference numerically identical."""

    with _TORCH_THREAD_SCOPE_LOCK:
        previous_threads = torch.get_num_threads()
        if previous_threads != 1:
            torch.set_num_threads(1)
        try:
            if torch.get_num_threads() != 1:
                raise RecurrentEvaluationError(
                    "evaluation could not enforce one Torch thread"
                )
            yield
        finally:
            if previous_threads != 1:
                torch.set_num_threads(previous_threads)


def _runtime_reproducibility_provenance(
    *,
    source_model: PublicRecurrentActorCritic,
    evaluation_model: PublicRecurrentActorCritic,
) -> dict[str, object]:
    """Record execution-only runtime facts without changing scientific results."""

    resolved_device = _model_device(evaluation_model, field="evaluation_model")
    if resolved_device != "cpu":
        raise RecurrentEvaluationError(
            "recurrent evaluation model must resolve to the CPU device"
        )
    return {
        "schema_version": RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "numpy_version": _installed_package_version("numpy"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform_string": platform.platform(),
        },
        "source_model_device_observed": _model_device(
            source_model,
            field="source_model",
        ),
        "requested_device": "cpu",
        "resolved_device": resolved_device,
        "requested_device_source": "frozen_cpu_evaluation_contract",
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def _model_device(
    model: PublicRecurrentActorCritic,
    *,
    field: str,
) -> str:
    devices = {
        str(tensor.device)
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if not devices:
        raise RecurrentEvaluationError(f"{field} has no parameters or buffers")
    if len(devices) != 1:
        raise RecurrentEvaluationError(
            f"{field} tensors span multiple devices: {sorted(devices)!r}"
        )
    return next(iter(devices))


def _installed_package_version(distribution_name: str) -> str | None:
    try:
        return package_version(distribution_name)
    except PackageNotFoundError:
        return None


def _initialize_parallel_evaluation_worker(
    candidate_model: PublicRecurrentActorCritic,
) -> None:
    global _PARALLEL_EVALUATION_MODEL
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RecurrentEvaluationError(
            "parallel evaluation worker could not enforce one Torch thread"
        )
    _PARALLEL_EVALUATION_MODEL = frozen_cpu_model_copy(candidate_model)
    if any(
        parameter.device.type != "cpu" or parameter.requires_grad
        for parameter in _PARALLEL_EVALUATION_MODEL.parameters()
    ):
        raise RecurrentEvaluationError(
            "parallel evaluation worker model is not frozen on CPU"
        )


def _evaluate_environment_task_in_worker(
    task: _EvaluationEnvironmentTask,
) -> dict[str, object]:
    candidate_model = _PARALLEL_EVALUATION_MODEL
    if candidate_model is None:
        raise RecurrentEvaluationError(
            "parallel evaluation worker lacks its frozen CPU model"
        )
    return _evaluate_environment_task(task, candidate_model=candidate_model)


def _run_parallel_environment_tasks(
    tasks: Sequence[_EvaluationEnvironmentTask],
    *,
    candidate_model: PublicRecurrentActorCritic,
    evaluation_workers: int,
) -> tuple[Mapping[str, object], ...]:
    if not tasks:
        raise RecurrentEvaluationError("parallel evaluation needs environment tasks")
    if evaluation_workers < 2 or evaluation_workers > len(tasks):
        raise RecurrentEvaluationError(
            "parallel evaluation worker count is inconsistent with its tasks"
        )
    spawn_context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=evaluation_workers,
            mp_context=spawn_context,
            initializer=_initialize_parallel_evaluation_worker,
            initargs=(candidate_model,),
        ) as executor:
            results = tuple(
                executor.map(
                    _evaluate_environment_task_in_worker,
                    tasks,
                    chunksize=1,
                )
            )
    except Exception as exc:
        raise RecurrentEvaluationError(
            "spawn-process evaluation failed; no sequential fallback was run"
        ) from exc
    if len(results) != len(tasks):
        raise RecurrentEvaluationError(
            "parallel evaluation returned an incomplete ordered result set"
        )
    return results


def _new_candidate_policy(
    model: PublicRecurrentActorCritic,
    policy_digest_label: str,
    *,
    feed_forward_history_ablation: bool,
    candidate_action_selection: str,
    sampling_seed: int | None,
) -> DeterministicPublicRecurrentPolicy:
    # Deliberately receives neither world seed nor fixture identity.
    return DeterministicPublicRecurrentPolicy(
        model,
        artifact_digest=policy_digest_label,
        reset_recurrent_state_each_decision=feed_forward_history_ablation,
        sampling_seed=(
            sampling_seed
            if candidate_action_selection == PUBLIC_RECURRENT_SAMPLED_SELECTION
            else None
        ),
    )


def _run_policy_world(
    *,
    seed: int,
    fixture_name: str | None,
    policy: object,
    policy_sampling_seed: int | None = None,
) -> dict[str, object]:
    world = (
        SimulationWorld(
            WorldConfig(seed=seed, max_ticks=RECURRENT_EVALUATION_TICKS),
            policy=policy,
        )
        if fixture_name is None
        else _build_controlled_fixture_world_adapter(
            fixture_name=fixture_name,
            seed=seed,
            policy=policy,
        )
    )
    result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    records = world.trajectory_records
    decision_diagnostics_records = world.policy_decision_diagnostics_records
    if len(decision_diagnostics_records) != len(records):
        raise RecurrentEvaluationError(
            "trajectory and policy-decision diagnostics are not aligned"
        )
    requested_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    policy_id_counts: Counter[str] = Counter()
    reward_total = 0.0
    reward_component_totals = {component: 0.0 for component in REWARD_COMPONENT_BOUNDS}
    unsupported_requested_action_count = 0
    recurrent_policy_record_count = 0
    policy_decision_record_count = 0
    passive_trajectory_record_count = 0
    learned_distribution_rows: list[dict[str, object]] = []
    eat_requested_count = 0
    eat_without_positive_resource_gain_count = 0
    for record, decision_diagnostics in zip(
        records,
        decision_diagnostics_records,
        strict=True,
    ):
        if not isinstance(record, Mapping):
            raise RecurrentEvaluationError("trajectory record must be a mapping")
        requested = record.get("requested_action")
        if not isinstance(requested, str) or requested not in ACTION_NAMES:
            raise RecurrentEvaluationError("trajectory requested action is invalid")
        action_source = str(record.get("action_source", "unknown"))
        action_source_counts[action_source] += 1
        policy_id_counts[str(record.get("policy_id", "unknown"))] += 1
        reward = _required_mapping(record.get("reward"), field="record.reward")
        reward_total += _finite_number(reward.get("total"), field="reward.total")
        components = _required_mapping(
            reward.get("components"),
            field="reward.components",
        )
        if set(components) != set(REWARD_COMPONENT_BOUNDS):
            raise RecurrentEvaluationError(
                "reward component keys differ from the canonical contract"
            )
        for component in REWARD_COMPONENT_BOUNDS:
            reward_component_totals[component] += _finite_number(
                components.get(component),
                field=f"reward.components.{component}",
            )

        # Runtime trajectory completeness includes agents that die before their
        # turn. Those passive records carry a synthetic, always-valid `stay`
        # solely to satisfy the trajectory schema; they are not policy choices
        # and must not enter learned action-distribution diagnostics.
        if action_source == "passive":
            passive_trajectory_record_count += 1
            continue

        policy_decision_record_count += 1
        requested_counts[requested] += 1
        if record.get("policy_id") == PUBLIC_RECURRENT_POLICY_ID:
            recurrent_policy_record_count += 1
        if record.get("action_valid") is not True:
            unsupported_requested_action_count += 1
        if requested == "eat":
            eat_requested_count += 1
            if float(components["resource_acquisition"]) <= 0.0:
                eat_without_positive_resource_gain_count += 1
        learned_distribution = _learned_distribution_from_record(
            record,
            requested_action=requested,
            decision_diagnostics=decision_diagnostics,
        )
        if learned_distribution is not None:
            learned_distribution_rows.append(learned_distribution)

    if (
        recurrent_policy_record_count
        and len(learned_distribution_rows) != recurrent_policy_record_count
    ):
        raise RecurrentEvaluationError(
            "public recurrent decisions lack complete learned-distribution diagnostics"
        )

    dominant = dominant_action_summary(requested_counts)
    summary = result.summary
    run = {
        "context": _context_label(fixture_name),
        "seed": seed,
        "policy_sampling_seed": policy_sampling_seed,
        "horizon_ticks": RECURRENT_EVALUATION_TICKS,
        "ticks_executed": _nonnegative_int(
            summary.get("ticks_executed"),
            field="summary.ticks_executed",
        ),
        "terminal_alive": _nonnegative_int(
            summary.get("alive_agents"),
            field="summary.alive_agents",
        ),
        "births": _nonnegative_int(summary.get("births"), field="summary.births"),
        "deaths": _nonnegative_int(summary.get("deaths"), field="summary.deaths"),
        "reward_total": round(reward_total, 12),
        "reward_component_totals": {
            component: round(total, 12)
            for component, total in reward_component_totals.items()
        },
        "trajectory_record_count": len(records),
        "policy_decision_record_count": policy_decision_record_count,
        "passive_trajectory_record_count": passive_trajectory_record_count,
        "requested_action_counts": dict(sorted(requested_counts.items())),
        "dominant_requested_action": dominant["action"],
        "dominant_requested_action_count": dominant["count"],
        "dominant_requested_action_share": dominant["share"],
        "unsupported_requested_action_count": unsupported_requested_action_count,
        "heuristic_action_source_count": sum(
            count
            for source, count in action_source_counts.items()
            if "heuristic" in source
        ),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
        "eat_requested_count": eat_requested_count,
        "eat_without_positive_resource_gain_count": (
            eat_without_positive_resource_gain_count
        ),
        "eat_without_positive_resource_gain_share": (
            round_float(eat_without_positive_resource_gain_count / eat_requested_count)
            if eat_requested_count
            else None
        ),
        "learned_masked_distribution": _aggregate_learned_distributions(
            learned_distribution_rows
        ),
    }
    full_behavior_payload = {
        "run": run,
        "summary": summary,
        "trajectory_records": records,
        "policy_decision_diagnostics": decision_diagnostics_records,
    }
    run["behavior_digest"] = _canonical_sha256(
        _artifact_independent_behavior_payload(full_behavior_payload)
    )
    run["replay_digest"] = _canonical_sha256(full_behavior_payload)
    run["outcome_evidence_sha256"] = _run_outcome_evidence_sha256(run)
    _validate_run(run)
    return run


def _artifact_independent_behavior_payload(value: object) -> object:
    """Remove only serialized artifact labels from full-world replay evidence.

    The ordinary replay digest remains provenance-bound.  This second view is
    deliberately narrow: it retains every observation, action, mask, outcome,
    reward, state transition, policy identifier, and controller version while
    replacing the frozen-artifact label embedded in recurrent diagnostics.
    That makes an in-memory policy and its serialized CPU reload comparable at
    full-trajectory granularity without weakening either artifact's own replay
    identity.
    """

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key == "artifact_digest":
                normalized[key] = "<artifact-digest>"
            elif (
                key == "policy_version"
                and isinstance(raw_value, str)
                and raw_value.startswith(f"{PUBLIC_RECURRENT_POLICY_VERSION}+")
            ):
                prefix, separator, selection = raw_value.rpartition("+")
                if not separator or "+" not in prefix:
                    raise RecurrentEvaluationError(
                        "public recurrent policy version cannot be normalized"
                    )
                normalized[key] = (
                    f"{PUBLIC_RECURRENT_POLICY_VERSION}+<artifact-digest-prefix>+"
                    f"{selection}"
                )
            else:
                normalized[key] = _artifact_independent_behavior_payload(raw_value)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_artifact_independent_behavior_payload(item) for item in value]
    return value


def _build_controlled_fixture_world_adapter(
    *,
    fixture_name: str,
    seed: int,
    policy: object,
) -> SimulationWorld:
    """Narrow adapter around the existing canonical private fixture builder."""

    # The canonical builder is private today. Keeping this dependency in one
    # named adapter avoids copying or silently drifting the controlled ecology.
    from evolution_sim.mind.evaluation_harness import _fixture_world

    return _fixture_world(
        fixture_name=fixture_name,
        seed=seed,
        ticks=RECURRENT_EVALUATION_TICKS,
        policy=policy,
    )


def _aggregate_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not runs:
        raise RecurrentEvaluationError("cannot aggregate an empty run set")
    action_counts: Counter[str] = Counter()
    for run in runs:
        counts = _required_mapping(
            run.get("requested_action_counts"),
            field="requested_action_counts",
        )
        action_counts.update({str(key): int(value) for key, value in counts.items()})
    dominant = dominant_action_summary(action_counts)
    count = len(runs)
    reward_component_totals = {
        component: round(
            math.fsum(
                _finite_number(
                    _required_mapping(
                        run.get("reward_component_totals"),
                        field="reward_component_totals",
                    ).get(component),
                    field=f"reward_component_totals.{component}",
                )
                for run in runs
            ),
            12,
        )
        for component in REWARD_COMPONENT_BOUNDS
    }
    eat_requested_count = sum(int(run["eat_requested_count"]) for run in runs)
    eat_without_gain_count = sum(
        int(run["eat_without_positive_resource_gain_count"]) for run in runs
    )
    return {
        "run_count": count,
        "terminal_alive_agent_total": sum(int(run["terminal_alive"]) for run in runs),
        "terminal_alive_mean": round_float(
            sum(int(run["terminal_alive"]) for run in runs) / count
        ),
        "births_mean": round_float(sum(int(run["births"]) for run in runs) / count),
        "reward_total_mean": round_float(
            sum(float(run["reward_total"]) for run in runs) / count
        ),
        "reward_component_totals": reward_component_totals,
        "reward_component_means_per_run": {
            component: round_float(total / count)
            for component, total in reward_component_totals.items()
        },
        "terminal_nonextinct_run_count": sum(
            1 for run in runs if int(run["terminal_alive"]) > 0
        ),
        "trajectory_record_count": sum(
            int(run["trajectory_record_count"]) for run in runs
        ),
        "policy_decision_record_count": sum(
            int(run["policy_decision_record_count"]) for run in runs
        ),
        "passive_trajectory_record_count": sum(
            int(run["passive_trajectory_record_count"]) for run in runs
        ),
        "requested_action_counts": dict(sorted(action_counts.items())),
        "dominant_requested_action": dominant["action"],
        "dominant_requested_action_count": dominant["count"],
        "dominant_requested_action_share": dominant["share"],
        "unsupported_requested_action_count": sum(
            int(run["unsupported_requested_action_count"]) for run in runs
        ),
        "heuristic_action_source_count": sum(
            int(run["heuristic_action_source_count"]) for run in runs
        ),
        "eat_requested_count": eat_requested_count,
        "eat_without_positive_resource_gain_count": eat_without_gain_count,
        "eat_without_positive_resource_gain_share": (
            round_float(eat_without_gain_count / eat_requested_count)
            if eat_requested_count
            else None
        ),
        "learned_masked_distribution": _aggregate_run_distributions(runs),
    }


def _learned_distribution_from_record(
    record: Mapping[str, object],
    *,
    requested_action: str,
    decision_diagnostics: object,
) -> dict[str, object] | None:
    decision = decision_diagnostics
    if decision is None:
        return None
    if not isinstance(decision, Mapping):
        raise RecurrentEvaluationError(
            "policy decision diagnostics must be a mapping when present"
        )
    payload = decision.get("learned_masked_distribution")
    if payload is None:
        return None
    distribution = _required_mapping(
        payload,
        field="learned_masked_distribution",
    )
    if (
        distribution.get("schema_version")
        != PUBLIC_RECURRENT_DISTRIBUTION_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise RecurrentEvaluationError(
            "learned masked distribution schema version drifted"
        )
    action_mask = _required_mapping(record.get("action_mask"), field="action_mask")
    valid_actions = _strict_valid_actions(action_mask)
    if distribution.get("valid_action_count") != len(valid_actions):
        raise RecurrentEvaluationError(
            "learned distribution valid-action count differs from the trajectory mask"
        )
    selected_action = distribution.get("selected_action")
    top_action = distribution.get("top_action")
    if selected_action not in valid_actions or top_action not in valid_actions:
        raise RecurrentEvaluationError(
            "learned distribution selected or top action is not mask-valid"
        )
    if (
        record.get("action_source") != RECURRENT_COUNTERFACTUAL_ACTION_SOURCE
        and selected_action != requested_action
    ):
        raise RecurrentEvaluationError(
            "ordinary recurrent requested action differs from its learned selection"
        )
    probabilities = _required_mapping(
        distribution.get("probabilities"),
        field="learned_masked_distribution.probabilities",
    )
    masked_logits = _required_mapping(
        distribution.get("masked_logits"),
        field="learned_masked_distribution.masked_logits",
    )
    if set(probabilities) != set(ACTION_NAMES) or set(masked_logits) != set(
        ACTION_NAMES
    ):
        raise RecurrentEvaluationError(
            "learned distribution action mapping differs from the stable vocabulary"
        )
    parsed_probabilities: dict[str, float] = {}
    for action in ACTION_NAMES:
        probability = _finite_number(
            probabilities.get(action),
            field=f"learned probability {action}",
        )
        if not 0.0 <= probability <= 1.0:
            raise RecurrentEvaluationError(
                "learned distribution probability is outside [0, 1]"
            )
        if action not in valid_actions and probability != 0.0:
            raise RecurrentEvaluationError(
                "mask-invalid action has nonzero learned probability"
            )
        logit = masked_logits.get(action)
        if action in valid_actions:
            _finite_number(logit, field=f"learned masked logit {action}")
        elif logit is not None:
            raise RecurrentEvaluationError(
                "mask-invalid action must serialize a null masked logit"
            )
        parsed_probabilities[action] = probability
    if not math.isclose(
        math.fsum(parsed_probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-7,
    ):
        raise RecurrentEvaluationError("learned masked probabilities do not sum to one")

    parsed: dict[str, object] = {}
    for metric in _DISTRIBUTION_METRICS:
        value = distribution.get(metric)
        if value is None and metric in {
            "normalized_entropy",
            "top_two_probability_margin",
        }:
            parsed[metric] = None
        else:
            parsed[metric] = _finite_number(
                value,
                field=f"learned_masked_distribution.{metric}",
            )
    if parsed["normalized_entropy"] is not None:
        normalized_upper = 1.0 + 1.0e-6 / math.log(len(valid_actions))
        if (
            not 0.0
            <= float(parsed["normalized_entropy"])
            <= (normalized_upper + 2.0e-9)
        ):
            raise RecurrentEvaluationError(
                "normalized entropy is outside its numerical mask bounds"
            )
    for metric in (
        "selected_action_probability",
        "top_action_probability",
        "top_two_probability_margin",
        "eat_probability",
    ):
        if parsed[metric] is not None and not 0.0 <= float(parsed[metric]) <= 1.0:
            raise RecurrentEvaluationError(
                f"learned distribution {metric} is outside [0, 1]"
            )
    parsed["probabilities"] = parsed_probabilities
    return parsed


def _aggregate_learned_distributions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(rows)
    if count == 0:
        return {
            "decision_count": 0,
            "metric_observation_counts": {
                metric: 0 for metric in _DISTRIBUTION_METRICS
            },
            "metric_means": {metric: None for metric in _DISTRIBUTION_METRICS},
            "metric_minima": {metric: None for metric in _DISTRIBUTION_METRICS},
            "metric_maxima": {metric: None for metric in _DISTRIBUTION_METRICS},
            "mean_action_probabilities": {action: None for action in ACTION_NAMES},
        }
    observed_by_metric = {
        metric: [float(row[metric]) for row in rows if row[metric] is not None]
        for metric in _DISTRIBUTION_METRICS
    }
    return {
        "decision_count": count,
        "metric_observation_counts": {
            metric: len(values) for metric, values in observed_by_metric.items()
        },
        "metric_means": {
            metric: (round_float(math.fsum(values) / len(values)) if values else None)
            for metric, values in observed_by_metric.items()
        },
        "metric_minima": {
            metric: round_float(min(values)) if values else None
            for metric, values in observed_by_metric.items()
        },
        "metric_maxima": {
            metric: round_float(max(values)) if values else None
            for metric, values in observed_by_metric.items()
        },
        "mean_action_probabilities": {
            action: round_float(
                math.fsum(
                    float(
                        _required_mapping(
                            row.get("probabilities"),
                            field="learned probabilities",
                        )[action]
                    )
                    for row in rows
                )
                / count
            )
            for action in ACTION_NAMES
        },
    }


def _aggregate_run_distributions(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summaries = [
        _required_mapping(
            run.get("learned_masked_distribution"),
            field="learned_masked_distribution",
        )
        for run in runs
    ]
    total = sum(int(summary.get("decision_count", 0)) for summary in summaries)
    if total == 0:
        return _aggregate_learned_distributions(())
    metric_means: dict[str, float | None] = {}
    metric_minima: dict[str, float | None] = {}
    metric_maxima: dict[str, float | None] = {}
    metric_observation_counts: dict[str, int] = {}
    for metric in _DISTRIBUTION_METRICS:
        weighted_values: list[float] = []
        minima: list[float] = []
        maxima: list[float] = []
        for summary in summaries:
            observations = _required_mapping(
                summary.get("metric_observation_counts"),
                field="metric_observation_counts",
            )
            observation_count = int(observations.get(metric, 0))
            if observation_count == 0:
                continue
            means = _required_mapping(summary.get("metric_means"), field="metric_means")
            mins = _required_mapping(
                summary.get("metric_minima"), field="metric_minima"
            )
            maxes = _required_mapping(
                summary.get("metric_maxima"), field="metric_maxima"
            )
            weighted_values.append(float(means[metric]) * observation_count)
            minima.append(float(mins[metric]))
            maxima.append(float(maxes[metric]))
        metric_count = sum(
            int(
                _required_mapping(
                    summary.get("metric_observation_counts"),
                    field="metric_observation_counts",
                ).get(metric, 0)
            )
            for summary in summaries
        )
        metric_observation_counts[metric] = metric_count
        metric_means[metric] = (
            round_float(math.fsum(weighted_values) / metric_count)
            if metric_count
            else None
        )
        metric_minima[metric] = round_float(min(minima)) if minima else None
        metric_maxima[metric] = round_float(max(maxima)) if maxima else None
    action_means: dict[str, float] = {}
    for action in ACTION_NAMES:
        weighted = 0.0
        for summary in summaries:
            decision_count = int(summary.get("decision_count", 0))
            if decision_count == 0:
                continue
            action_probabilities = _required_mapping(
                summary.get("mean_action_probabilities"),
                field="mean_action_probabilities",
            )
            weighted += float(action_probabilities[action]) * decision_count
        action_means[action] = round_float(weighted / total)
    return {
        "decision_count": total,
        "metric_observation_counts": metric_observation_counts,
        "metric_means": metric_means,
        "metric_minima": metric_minima,
        "metric_maxima": metric_maxima,
        "mean_action_probabilities": action_means,
    }


def _candidate_sampling_analysis(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not runs:
        raise RecurrentEvaluationError("candidate sampling analysis needs runs")
    by_environment: dict[int, list[Mapping[str, object]]] = {}
    by_sampling_stream: dict[int | None, list[Mapping[str, object]]] = {}
    for run in runs:
        seed = _validated_seed(run.get("seed"), field="candidate run seed")
        by_environment.setdefault(seed, []).append(run)
        sampling_seed = run.get("policy_sampling_seed")
        if sampling_seed is not None:
            sampling_seed = _validated_sampling_seed(
                sampling_seed,
                field="candidate policy sampling seed",
            )
        by_sampling_stream.setdefault(sampling_seed, []).append(run)
    replicate_counts = {len(group) for group in by_environment.values()}
    if len(replicate_counts) != 1:
        raise RecurrentEvaluationError(
            "candidate sampling replicate count differs across environments"
        )
    expected_streams = set(by_sampling_stream)
    for seed, group in by_environment.items():
        observed_streams = [run.get("policy_sampling_seed") for run in group]
        if len(observed_streams) != len(set(observed_streams)):
            raise RecurrentEvaluationError(
                f"candidate sampling stream is duplicated for environment {seed}"
            )
        if set(observed_streams) != expected_streams:
            raise RecurrentEvaluationError(
                "candidate sampling stream set differs across environments"
            )
    expected_environments = set(by_environment)
    for sampling_seed, group in by_sampling_stream.items():
        observed_environments = [int(run["seed"]) for run in group]
        if (
            len(observed_environments) != len(set(observed_environments))
            or set(observed_environments) != expected_environments
        ):
            raise RecurrentEvaluationError(
                "candidate environment set differs across sampling streams"
            )

    nonextinct_run_count = sum(1 for run in runs if int(run["terminal_alive"]) > 0)
    terminal_alive_agent_total = sum(int(run["terminal_alive"]) for run in runs)
    lower, upper = _wilson_interval(
        successes=nonextinct_run_count,
        trials=len(runs),
    )
    environment_rows = []
    environments_with_any_nonextinct_stream_count = 0
    for seed, group in sorted(by_environment.items()):
        count = len(group)
        nonextinct_runs = sum(1 for run in group if int(run["terminal_alive"]) > 0)
        if nonextinct_runs:
            environments_with_any_nonextinct_stream_count += 1
        environment_rows.append(
            {
                "seed": seed,
                "sampling_run_count": count,
                "terminal_nonextinct_run_count": nonextinct_runs,
                "terminal_nonextinct_run_fraction": round_float(
                    nonextinct_runs / count
                ),
                "terminal_alive_agent_total": sum(
                    int(run["terminal_alive"]) for run in group
                ),
                "terminal_alive_mean": round_float(
                    math.fsum(float(run["terminal_alive"]) for run in group) / count
                ),
                "births_mean": round_float(
                    math.fsum(float(run["births"]) for run in group) / count
                ),
                "reward_total_mean": round_float(
                    math.fsum(float(run["reward_total"]) for run in group) / count
                ),
            }
        )
    sampling_stream_rows = []
    for sampling_seed, group in sorted(
        by_sampling_stream.items(),
        key=lambda item: -1 if item[0] is None else item[0],
    ):
        stream_nonextinct_run_count = sum(
            1 for run in group if int(run["terminal_alive"]) > 0
        )
        sampling_stream_rows.append(
            {
                "policy_sampling_seed": sampling_seed,
                "environment_run_count": len(group),
                "terminal_nonextinct_run_count": (stream_nonextinct_run_count),
                "terminal_nonextinct_run_fraction": round_float(
                    stream_nonextinct_run_count / len(group)
                ),
                "terminal_alive_agent_total": sum(
                    int(run["terminal_alive"]) for run in group
                ),
                "terminal_alive_mean": round_float(
                    math.fsum(float(run["terminal_alive"]) for run in group)
                    / len(group)
                ),
            }
        )
    return {
        "environment_count": len(by_environment),
        "policy_sampling_stream_count": len(by_sampling_stream),
        "sampling_runs_per_environment": next(iter(replicate_counts)),
        "candidate_run_count": len(runs),
        "terminal_nonextinct_run_count": nonextinct_run_count,
        "terminal_alive_agent_total": terminal_alive_agent_total,
        "observed_terminal_nonextinct_run_fraction": round_float(
            nonextinct_run_count / len(runs)
        ),
        "environments_with_any_nonextinct_stream_count": (
            environments_with_any_nonextinct_stream_count
        ),
        "environments_with_all_streams_extinct_count": (
            len(by_environment) - environments_with_any_nonextinct_stream_count
        ),
        "descriptive_run_cell_wilson_95": {
            "lower": round_float(lower),
            "upper": round_float(upper),
            "calculation_unit": "environment_policy_sampling_stream_run_cell",
            "descriptive_only": True,
            "independence_assumption_satisfied": False,
            "correlation_structure": [
                "multiple_policy_streams_share_each_environment",
                "the_same_policy_stream_is_reused_across_environments",
            ],
            "warning": (
                "This run-cell Wilson interval is descriptive only. Its cells "
                "are crossed repeated measurements sharing environment seeds "
                "and policy sampling streams, not independent trials."
            ),
        },
        "environment_stratified": environment_rows,
        "policy_sampling_stream_stratified": sampling_stream_rows,
    }


def _wilson_interval(*, successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise RecurrentEvaluationError("Wilson interval counts are invalid")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_deltas(
    candidate_runs: Sequence[Mapping[str, object]],
    baseline_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(candidate_runs) != len(baseline_runs) or not candidate_runs:
        raise RecurrentEvaluationError("paired runs must be non-empty and aligned")
    deltas: list[dict[str, object]] = []
    for candidate, baseline in zip(candidate_runs, baseline_runs, strict=True):
        if candidate.get("seed") != baseline.get("seed") or candidate.get(
            "context"
        ) != baseline.get("context"):
            raise RecurrentEvaluationError("paired run identity mismatch")
        deltas.append(
            {
                "context": candidate["context"],
                "seed": candidate["seed"],
                "terminal_alive_delta": int(candidate["terminal_alive"])
                - int(baseline["terminal_alive"]),
                "births_delta": int(candidate["births"]) - int(baseline["births"]),
                "reward_total_delta": round(
                    float(candidate["reward_total"]) - float(baseline["reward_total"]),
                    12,
                ),
                "dominant_requested_action_share_delta": round_float(
                    float(candidate["dominant_requested_action_share"])
                    - float(baseline["dominant_requested_action_share"])
                ),
                "unsupported_requested_action_count_delta": int(
                    candidate["unsupported_requested_action_count"]
                )
                - int(baseline["unsupported_requested_action_count"]),
                "heuristic_action_source_count_delta": int(
                    candidate["heuristic_action_source_count"]
                )
                - int(baseline["heuristic_action_source_count"]),
            }
        )
    count = len(deltas)
    return {
        "runs": deltas,
        "mean_terminal_alive_delta": round_float(
            sum(int(item["terminal_alive_delta"]) for item in deltas) / count
        ),
        "mean_births_delta": round_float(
            sum(int(item["births_delta"]) for item in deltas) / count
        ),
        "mean_reward_total_delta": round_float(
            sum(float(item["reward_total_delta"]) for item in deltas) / count
        ),
    }


def _run_outcome_evidence_sha256(run: Mapping[str, object]) -> str:
    """Bind every persisted run outcome to its full-behavior replay digests."""

    behavior_digest = _validated_sha256(
        run.get("behavior_digest"),
        field="run.behavior_digest",
    )
    replay_digest = _validated_sha256(
        run.get("replay_digest"),
        field="run.replay_digest",
    )
    outcome_fields = _RUN_FIELDS - _RUN_DIGEST_FIELDS
    missing = sorted(field for field in outcome_fields if field not in run)
    if missing:
        raise RecurrentEvaluationError(
            f"run outcome evidence fields are missing: {missing!r}"
        )
    projection = {
        field: run[field]
        for field in sorted(outcome_fields)
    }
    return _canonical_sha256(
        {
            "schema_version": _RUN_OUTCOME_EVIDENCE_SCHEMA_VERSION,
            "behavior_digest": behavior_digest,
            "replay_digest": replay_digest,
            "outcome": projection,
        }
    )


def _validated_named_counts(
    value: Mapping[str, object],
    *,
    field: str,
) -> dict[str, int]:
    raw_names = tuple(value)
    if any(
        not isinstance(raw_name, str)
        or not raw_name
        or raw_name != raw_name.strip()
        for raw_name in raw_names
    ):
        raise RecurrentEvaluationError(
            f"{field} keys must be non-empty trimmed strings"
        )
    if raw_names != tuple(sorted(raw_names)):
        raise RecurrentEvaluationError(f"{field} keys are not in canonical order")
    parsed: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        parsed[raw_name] = _nonnegative_int(
            raw_count,
            field=f"{field}.{raw_name}",
        )
    return parsed


def _validate_learned_distribution_summary(
    distribution: Mapping[str, object],
) -> None:
    _require_exact_fields(
        distribution,
        _LEARNED_DISTRIBUTION_SUMMARY_FIELDS,
        field="learned_masked_distribution",
    )
    decision_count = _nonnegative_int(
        distribution.get("decision_count"),
        field="learned_masked_distribution.decision_count",
    )
    observation_counts = _required_mapping(
        distribution.get("metric_observation_counts"),
        field="learned_masked_distribution.metric_observation_counts",
    )
    means = _required_mapping(
        distribution.get("metric_means"),
        field="learned_masked_distribution.metric_means",
    )
    minima = _required_mapping(
        distribution.get("metric_minima"),
        field="learned_masked_distribution.metric_minima",
    )
    maxima = _required_mapping(
        distribution.get("metric_maxima"),
        field="learned_masked_distribution.metric_maxima",
    )
    expected_metrics = set(_DISTRIBUTION_METRICS)
    for name, values in (
        ("metric_observation_counts", observation_counts),
        ("metric_means", means),
        ("metric_minima", minima),
        ("metric_maxima", maxima),
    ):
        _require_exact_fields(
            values,
            expected_metrics,
            field=f"learned_masked_distribution.{name}",
        )
    for metric in _DISTRIBUTION_METRICS:
        count = _nonnegative_int(
            observation_counts.get(metric),
            field=f"learned_masked_distribution.metric_observation_counts.{metric}",
        )
        if count > decision_count:
            raise RecurrentEvaluationError(
                "learned-distribution metric count exceeds decision count"
            )
        metric_values = (
            means.get(metric),
            minima.get(metric),
            maxima.get(metric),
        )
        if count == 0:
            if metric_values != (None, None, None):
                raise RecurrentEvaluationError(
                    "unobserved learned-distribution metric must serialize null "
                    "summary values"
                )
            continue
        parsed_mean, parsed_minimum, parsed_maximum = (
            _finite_number(
                value,
                field=f"learned_masked_distribution.{name}.{metric}",
            )
            for name, value in zip(
                ("metric_means", "metric_minima", "metric_maxima"),
                metric_values,
                strict=True,
            )
        )
        if not parsed_minimum <= parsed_mean <= parsed_maximum:
            raise RecurrentEvaluationError(
                "learned-distribution metric summary bounds are inconsistent"
            )
    mean_action_probabilities = _required_mapping(
        distribution.get("mean_action_probabilities"),
        field="learned_masked_distribution.mean_action_probabilities",
    )
    _require_exact_fields(
        mean_action_probabilities,
        set(ACTION_NAMES),
        field="learned_masked_distribution.mean_action_probabilities",
    )
    if decision_count == 0:
        if any(
            mean_action_probabilities.get(action) is not None
            for action in ACTION_NAMES
        ):
            raise RecurrentEvaluationError(
                "zero-decision learned distribution must have null action means"
            )
        return
    action_means = tuple(
        _finite_number(
            mean_action_probabilities.get(action),
            field=(
                "learned_masked_distribution.mean_action_probabilities."
                f"{action}"
            ),
        )
        for action in ACTION_NAMES
    )
    if any(not 0.0 <= value <= 1.0 for value in action_means) or not math.isclose(
        math.fsum(action_means),
        1.0,
        rel_tol=0.0,
        # Per-action means are intentionally serialized at four decimal
        # places, so the aggregate normalization tolerance must include the
        # worst-case sum of those independent rounding errors.
        abs_tol=(len(ACTION_NAMES) * 5.0e-5) + 1.0e-9,
    ):
        raise RecurrentEvaluationError(
            "learned-distribution mean action probabilities are invalid"
        )


def _validate_run(run: Mapping[str, object]) -> None:
    _require_exact_fields(run, _RUN_FIELDS, field="run")
    behavior_digest = _validated_sha256(
        run.get("behavior_digest"),
        field="run.behavior_digest",
    )
    replay_digest = _validated_sha256(
        run.get("replay_digest"),
        field="run.replay_digest",
    )
    outcome_evidence_sha256 = _validated_sha256(
        run.get("outcome_evidence_sha256"),
        field="run.outcome_evidence_sha256",
    )
    context = run.get("context")
    if not isinstance(context, str) or not context or context != context.strip():
        raise RecurrentEvaluationError("run.context must be a non-empty trimmed string")
    _validated_seed(run.get("seed"), field="run.seed")
    _validated_optional_sampling_seed(
        run.get("policy_sampling_seed"),
        field="run.policy_sampling_seed",
    )
    if run.get("horizon_ticks") != RECURRENT_EVALUATION_TICKS:
        raise RecurrentEvaluationError("run horizon is not the exact 120-tick contract")
    ticks_executed = _nonnegative_int(
        run.get("ticks_executed"),
        field="ticks_executed",
    )
    if not 1 <= ticks_executed <= RECURRENT_EVALUATION_TICKS:
        raise RecurrentEvaluationError("ticks_executed is outside the world horizon")
    _nonnegative_int(run.get("terminal_alive"), field="terminal_alive")
    _nonnegative_int(run.get("births"), field="births")
    _nonnegative_int(run.get("deaths"), field="deaths")
    _finite_number(run.get("reward_total"), field="reward_total")
    counts = _required_mapping(
        run.get("requested_action_counts"),
        field="requested_action_counts",
    )
    if any(not isinstance(action, str) or action not in ACTION_NAMES for action in counts):
        raise RecurrentEvaluationError("run contains an action outside the contract")
    if tuple(counts) != tuple(sorted(counts)):
        raise RecurrentEvaluationError(
            "run requested-action counts are not in canonical key order"
        )
    parsed_counts = {
        str(action): _nonnegative_int(
            count,
            field=f"requested_action_counts.{action}",
        )
        for action, count in counts.items()
    }
    trajectory_record_count = _nonnegative_int(
        run.get("trajectory_record_count"),
        field="trajectory_record_count",
    )
    policy_decision_record_count = _nonnegative_int(
        run.get("policy_decision_record_count"),
        field="policy_decision_record_count",
    )
    passive_trajectory_record_count = _nonnegative_int(
        run.get("passive_trajectory_record_count"),
        field="passive_trajectory_record_count",
    )
    if (
        policy_decision_record_count + passive_trajectory_record_count
        != trajectory_record_count
    ):
        raise RecurrentEvaluationError(
            "policy-decision and passive counts do not cover trajectory"
        )
    if sum(parsed_counts.values()) != policy_decision_record_count:
        raise RecurrentEvaluationError(
            "action distribution does not cover policy decisions"
        )
    expected_dominant = dominant_action_summary(parsed_counts)
    dominant_action = run.get("dominant_requested_action")
    if dominant_action is not None and (
        not isinstance(dominant_action, str) or dominant_action not in ACTION_NAMES
    ):
        raise RecurrentEvaluationError(
            "run dominant requested action is outside the contract"
        )
    dominant_count = _nonnegative_int(
        run.get("dominant_requested_action_count"),
        field="dominant_requested_action_count",
    )
    dominant_share = _finite_number(
        run.get("dominant_requested_action_share"),
        field="dominant_requested_action_share",
    )
    if (
        dominant_action != expected_dominant["action"]
        or dominant_count != expected_dominant["count"]
        or dominant_share != expected_dominant["share"]
    ):
        raise RecurrentEvaluationError(
            "run dominant requested-action summary differs from action counts"
        )
    action_source_counts = _required_mapping(
        run.get("action_source_counts"),
        field="action_source_counts",
    )
    parsed_action_source_counts = _validated_named_counts(
        action_source_counts,
        field="action_source_counts",
    )
    if (
        sum(parsed_action_source_counts.values()) != trajectory_record_count
    ):
        raise RecurrentEvaluationError(
            "action-source distribution does not cover trajectory"
        )
    if (
        parsed_action_source_counts.get("passive", 0)
        != passive_trajectory_record_count
    ):
        raise RecurrentEvaluationError(
            "passive action-source count differs from passive trajectory count"
        )
    unsupported_requested_action_count = _nonnegative_int(
        run.get("unsupported_requested_action_count"),
        field="unsupported_requested_action_count",
    )
    if unsupported_requested_action_count != 0:
        raise RecurrentEvaluationError("evaluation requested an unsupported action")
    heuristic_action_source_count = _nonnegative_int(
        run.get("heuristic_action_source_count"),
        field="heuristic_action_source_count",
    )
    if heuristic_action_source_count != 0:
        raise RecurrentEvaluationError("evaluation used a heuristic action source")
    expected_heuristic_count = sum(
        count
        for source, count in parsed_action_source_counts.items()
        if "heuristic" in source
    )
    if heuristic_action_source_count != expected_heuristic_count:
        raise RecurrentEvaluationError(
            "heuristic action-source count differs from source distribution"
        )
    policy_id_counts = _required_mapping(
        run.get("policy_id_counts"),
        field="policy_id_counts",
    )
    parsed_policy_id_counts = _validated_named_counts(
        policy_id_counts,
        field="policy_id_counts",
    )
    if sum(parsed_policy_id_counts.values()) != trajectory_record_count:
        raise RecurrentEvaluationError(
            "policy-id distribution does not cover trajectory"
        )
    reward_components = _required_mapping(
        run.get("reward_component_totals"),
        field="reward_component_totals",
    )
    if set(reward_components) != set(REWARD_COMPONENT_BOUNDS):
        raise RecurrentEvaluationError("run reward component totals drifted")
    for component in REWARD_COMPONENT_BOUNDS:
        _finite_number(
            reward_components.get(component),
            field=f"reward_component_totals.{component}",
        )
    eat_requested_count = _nonnegative_int(
        run.get("eat_requested_count"),
        field="eat_requested_count",
    )
    if eat_requested_count != parsed_counts.get("eat", 0):
        raise RecurrentEvaluationError("run eat-request count drifted")
    eat_without_gain = _nonnegative_int(
        run.get("eat_without_positive_resource_gain_count"),
        field="eat_without_positive_resource_gain_count",
    )
    if eat_without_gain > eat_requested_count:
        raise RecurrentEvaluationError(
            "run eat-without-resource-gain count is inconsistent"
        )
    expected_eat_without_gain_share = (
        round_float(eat_without_gain / eat_requested_count)
        if eat_requested_count
        else None
    )
    observed_eat_without_gain_share = run.get(
        "eat_without_positive_resource_gain_share"
    )
    if observed_eat_without_gain_share is not None:
        observed_eat_without_gain_share = _finite_number(
            observed_eat_without_gain_share,
            field="eat_without_positive_resource_gain_share",
        )
    if (
        observed_eat_without_gain_share
        != expected_eat_without_gain_share
    ):
        raise RecurrentEvaluationError(
            "run eat-without-resource-gain share differs from its counts"
        )
    distribution = _required_mapping(
        run.get("learned_masked_distribution"),
        field="learned_masked_distribution",
    )
    _validate_learned_distribution_summary(distribution)
    recurrent_records = parsed_policy_id_counts.get(
        PUBLIC_RECURRENT_POLICY_ID,
        0,
    )
    if distribution.get("decision_count") != recurrent_records:
        raise RecurrentEvaluationError(
            "learned-distribution count differs from recurrent policy records"
        )
    if _run_outcome_evidence_sha256(
        {
            **dict(run),
            "behavior_digest": behavior_digest,
            "replay_digest": replay_digest,
        }
    ) != outcome_evidence_sha256:
        raise RecurrentEvaluationError(
            "run outcome fields are detached from behavior/replay evidence"
        )


def _validate_report(report: Mapping[str, object]) -> None:
    _reject_ambiguous_v3_terminal_fields(report, field="report")
    if report.get("schema_version") != RECURRENT_EVALUATION_SCHEMA_VERSION:
        raise RecurrentEvaluationError("evaluation report schema drifted")
    execution = _required_mapping(
        report.get("execution_provenance"),
        field="execution_provenance",
    )
    if execution.get("schema_version") != RECURRENT_EVALUATION_EXECUTION_SCHEMA_VERSION:
        raise RecurrentEvaluationError("evaluation execution schema drifted")
    _validate_runtime_reproducibility_provenance(
        _required_mapping(
            execution.get("runtime_reproducibility"),
            field="execution_provenance.runtime_reproducibility",
        )
    )
    requested_workers = _validated_evaluation_workers(
        execution.get("evaluation_workers_requested")
    )
    used_workers = _validated_evaluation_workers(
        execution.get("evaluation_workers_used")
    )
    process_parallel = execution.get("process_parallel")
    if type(process_parallel) is not bool:
        raise RecurrentEvaluationError("process_parallel must be an exact boolean")
    if used_workers > requested_workers:
        raise RecurrentEvaluationError(
            "evaluation workers used exceeds the requested count"
        )
    if process_parallel:
        if (
            requested_workers <= 1
            or used_workers <= 1
            or execution.get("process_start_method") != "spawn"
        ):
            raise RecurrentEvaluationError(
                "parallel evaluation execution provenance is inconsistent"
            )
    elif (
        requested_workers != 1
        or used_workers != 1
        or execution.get("process_start_method") is not None
    ):
        raise RecurrentEvaluationError(
            "sequential evaluation execution provenance is inconsistent"
        )
    if execution.get("torch_threads_per_worker") != 1:
        raise RecurrentEvaluationError(
            "evaluation worker Torch thread contract drifted"
        )
    replay = _required_mapping(
        report.get("replay_verification"),
        field="replay_verification",
    )
    if replay.get("all_passed") is not True:
        raise RecurrentEvaluationError("candidate replay verification failed")
    replay_checks = _required_sequence(
        replay.get("checks"),
        field="replay_verification.checks",
    )
    if any(
        _required_mapping(check, field="replay verification check").get("passed")
        is not True
        for check in replay_checks
    ):
        raise RecurrentEvaluationError("candidate replay check failed")
    if replay.get("checked_run_count") != len(replay_checks):
        raise RecurrentEvaluationError("candidate replay check count drifted")
    seed_plan = _validate_seed_plan_report(
        _required_mapping(report.get("seed_plan"), field="seed_plan")
    )
    candidate_action_selection, expected_sampling_seeds = _validate_evaluation_contract(
        _required_mapping(
            report.get("evaluation_contract"), field="evaluation_contract"
        )
    )
    provenance = _required_mapping(
        report.get("candidate_provenance"),
        field="candidate_provenance",
    )
    source_pinned = provenance.get("source_pinned")
    if type(source_pinned) is not bool:
        raise RecurrentEvaluationError("candidate source_pinned must be boolean")
    common_provenance_keys = {
        "mode",
        "source_pinned",
        "synthetic_digest_label",
        "noncandidate_development_canary",
        "promotion_evidence_eligible_from_provenance",
        "promotion_eligibility_requires_external_one_use_lockbox_authorization",
        "external_one_use_lockbox_authorization_available",
        "research_candidate_evidence_requires_external_validation_authorization",
        "external_validation_authorization_available",
        "full_canonical_lockbox_plan",
        "contains_canonical_validation_seed",
        "contains_canonical_lockbox_seed",
        "artifact_training_seed_evidence",
        "caller_excluded_training_seeds_used_as_proof",
        "candidate_action_selection_promotion_eligible",
        "sampled_candidate_diagnostic",
        "feed_forward_history_ablation",
    }
    expected_provenance_keys = common_provenance_keys | (
        {"pin_verification"}
        if source_pinned
        else {
            "synthetic_noncandidate_digest_label",
            "runtime_integration_authorized",
        }
    )
    if set(provenance) != expected_provenance_keys:
        raise RecurrentEvaluationError("candidate provenance keys drifted")
    if source_pinned:
        if not isinstance(report.get("artifact"), Mapping):
            raise RecurrentEvaluationError("source-pinned evaluation lacks artifact")
        artifact = _required_mapping(report.get("artifact"), field="artifact")
        artifact_training_evidence = _required_mapping(
            artifact.get("training_seed_evidence"),
            field="artifact.training_seed_evidence",
        )
        provenance_training_evidence = _required_mapping(
            provenance.get("artifact_training_seed_evidence"),
            field="candidate_provenance.artifact_training_seed_evidence",
        )
        if artifact_training_evidence != provenance_training_evidence:
            raise RecurrentEvaluationError(
                "artifact training-seed evidence differs from candidate provenance"
            )
        _validate_artifact_training_seed_evidence(
            provenance_training_evidence,
            seed_plan=seed_plan,
        )
        expected_promotion_eligible = False
        if (
            provenance.get("promotion_evidence_eligible_from_provenance")
            is not expected_promotion_eligible
        ):
            raise RecurrentEvaluationError(
                "promotion eligibility differs from lockbox and artifact training "
                "seed evidence"
            )
    else:
        if report.get("artifact") is not None:
            raise RecurrentEvaluationError(
                "unpinned evaluation must not claim artifact"
            )
        if provenance.get("noncandidate_development_canary") is not True:
            raise RecurrentEvaluationError("unpinned evaluation is not noncandidate")
        if provenance.get("promotion_evidence_eligible_from_provenance") is not False:
            raise RecurrentEvaluationError(
                "unpinned evaluation must be ineligible for promotion evidence"
            )
        if provenance.get("artifact_training_seed_evidence") is not None:
            raise RecurrentEvaluationError(
                "unpinned evaluation must not claim artifact training-seed evidence"
            )
    evaluation_contract = _required_mapping(
        report.get("evaluation_contract"), field="evaluation_contract"
    )
    if evaluation_contract.get("candidate_sampling_stream_id_source") == (
        "candidate_policy_digest_default"
    ):
        expected_stream_id = (
            _required_mapping(report.get("artifact"), field="artifact").get(
                "artifact_sha256"
            )
            if source_pinned
            else provenance.get("synthetic_noncandidate_digest_label")
        )
        if evaluation_contract.get("candidate_sampling_stream_id") != (
            expected_stream_id
        ):
            raise RecurrentEvaluationError(
                "default candidate sampling stream is not bound to candidate identity"
            )
    if (
        provenance.get(
            "promotion_eligibility_requires_external_one_use_lockbox_authorization"
        )
        is not True
        or provenance.get("external_one_use_lockbox_authorization_available")
        is not False
        or provenance.get(
            "research_candidate_evidence_requires_external_validation_authorization"
        )
        is not True
        or provenance.get("external_validation_authorization_available") is not False
        or provenance.get("full_canonical_lockbox_plan")
        is not seed_plan.full_canonical_lockbox_plan
        or provenance.get("contains_canonical_validation_seed")
        is not seed_plan.contains_validation_seed
        or provenance.get("contains_canonical_lockbox_seed")
        is not seed_plan.contains_lockbox_seed
        or provenance.get("caller_excluded_training_seeds_used_as_proof") is not False
    ):
        raise RecurrentEvaluationError("candidate seed-integrity provenance drifted")
    expected_selection_eligible = bool(
        source_pinned
        and candidate_action_selection == PUBLIC_RECURRENT_ARGMAX_SELECTION
    )
    if provenance.get(
        "candidate_action_selection_promotion_eligible"
    ) is not expected_selection_eligible or provenance.get(
        "sampled_candidate_diagnostic"
    ) is not (candidate_action_selection == PUBLIC_RECURRENT_SAMPLED_SELECTION):
        raise RecurrentEvaluationError(
            "candidate action-selection eligibility provenance drifted"
        )
    feed_forward_history_ablation = provenance.get("feed_forward_history_ablation")
    if type(feed_forward_history_ablation) is not bool:
        raise RecurrentEvaluationError(
            "candidate feed-forward history-ablation provenance must be boolean"
        )
    expected_recurrent_state = (
        "zero_before_every_decision"
        if feed_forward_history_ablation
        else "one_hidden_state_per_agent"
    )
    if evaluation_contract.get("candidate_recurrent_state") != (
        expected_recurrent_state
    ):
        raise RecurrentEvaluationError(
            "candidate recurrent-state contract differs from provenance"
        )

    candidate_run_count = _validate_context_report(
        _required_mapping(report.get("broad"), field="broad"),
        field="broad",
        expected_context="broad_default",
        expected_sampling_seeds=expected_sampling_seeds,
    )
    broad_context = _required_mapping(report.get("broad"), field="broad")
    if tuple(broad_context.get("seeds", ())) != seed_plan.broad_seeds:
        raise RecurrentEvaluationError("broad context seeds differ from seed plan")
    expected_replay_checks = list(
        _expected_replay_checks_from_context(broad_context, field="broad")
    )
    fixture_values = _required_sequence(report.get("fixtures"), field="fixtures")
    fixture_names: set[str] = set()
    carrion_aggregate: Mapping[str, object] | None = None
    for index, fixture_value in enumerate(fixture_values):
        fixture = _required_mapping(fixture_value, field=f"fixtures[{index}]")
        fixture_name = fixture.get("fixture")
        if (
            not isinstance(fixture_name, str)
            or not fixture_name
            or fixture_name in fixture_names
        ):
            raise RecurrentEvaluationError(
                "evaluation fixture names must be non-empty and unique"
            )
        fixture_names.add(fixture_name)
        if tuple(fixture.get("seeds", ())) != seed_plan.fixture_seeds:
            raise RecurrentEvaluationError(
                f"fixture:{fixture_name} seeds differ from seed plan"
            )
        candidate_run_count += _validate_context_report(
            fixture,
            field=f"fixture:{fixture_name}",
            expected_context=f"fixture:{fixture_name}",
            expected_sampling_seeds=expected_sampling_seeds,
        )
        expected_replay_checks.extend(
            _expected_replay_checks_from_context(
                fixture,
                field=f"fixture:{fixture_name}",
            )
        )
        if fixture_name == "carrion_only":
            policies = _required_mapping(
                fixture.get("policies"),
                field="fixture:carrion_only.policies",
            )
            candidate = _required_mapping(
                policies.get("public_recurrent"),
                field="fixture:carrion_only.policies.public_recurrent",
            )
            carrion_aggregate = _required_mapping(
                candidate.get("aggregate"),
                field=("fixture:carrion_only.policies.public_recurrent.aggregate"),
            )
    if candidate_run_count != len(replay_checks):
        raise RecurrentEvaluationError(
            "candidate replay checks do not cover every candidate run"
        )
    observed_replay_checks = tuple(
        dict(_required_mapping(check, field="replay verification check"))
        for check in replay_checks
    )
    if observed_replay_checks != tuple(expected_replay_checks):
        raise RecurrentEvaluationError(
            "candidate replay identities or digests differ from raw candidate runs"
        )

    expected_nonextinct_run_count = (
        int(carrion_aggregate["terminal_nonextinct_run_count"])
        if carrion_aggregate is not None
        else 0
    )
    expected_alive_agent_total = (
        int(carrion_aggregate["terminal_alive_agent_total"])
        if carrion_aggregate is not None
        else 0
    )
    observed_nonextinct_run_count = _nonnegative_int(
        report.get("carrion_fixture_terminal_nonextinct_run_count"),
        field="carrion_fixture_terminal_nonextinct_run_count",
    )
    if observed_nonextinct_run_count != expected_nonextinct_run_count:
        raise RecurrentEvaluationError(
            "carrion terminal nonextinct-run count differs from raw runs"
        )
    observed_alive_agent_total = _nonnegative_int(
        report.get("carrion_fixture_terminal_alive_agent_total"),
        field="carrion_fixture_terminal_alive_agent_total",
    )
    if observed_alive_agent_total != expected_alive_agent_total:
        raise RecurrentEvaluationError(
            "carrion terminal alive-agent total differs from raw runs"
        )


def validate_recurrent_evaluation_report(report: Mapping[str, object]) -> None:
    """Public fail-closed validator for persisted or nested v4 evaluations."""

    _validate_report(report)


def _training_seed_set_for_plan(
    seed_plan: RecurrentEvaluationSeedPlan,
) -> frozenset[int]:
    if (
        seed_plan.environment_seed_role
        == RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE
    ):
        return _SCALE_TRAINING_SEEDS
    if (
        seed_plan.environment_seed_role
        == RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE
    ):
        return _SCALE_V2_TRAINING_SEEDS
    return _CANONICAL_TRAINING_SEEDS


def _artifact_training_seed_contract(
    artifact_provenance: Mapping[str, object],
    *,
    seed_plan: RecurrentEvaluationSeedPlan,
) -> tuple[tuple[str, str], Mapping[str, Sequence[int]], str]:
    registry_digest = artifact_provenance.get("seed_registry_digest")
    if registry_digest == SCALE_DEVELOPMENT_CANONICAL_SHA256:
        if (
            seed_plan.environment_seed_role
            != RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE
        ):
            raise RecurrentEvaluationError(
                "scale artifact requires an explicit scale selection seed plan"
            )
        return (
            _SCALE_TRAINING_SEED_ROLES,
            SCALE_DEVELOPMENT_SEED_REGISTRY,
            "mind_public_recurrent_training_seed_evidence_v2",
        )
    if registry_digest == SCALE_DEVELOPMENT_V2_CANONICAL_SHA256:
        if (
            seed_plan.environment_seed_role
            != RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE
        ):
            raise RecurrentEvaluationError(
                "scale-v2 artifact requires an explicit scale-v2 selection seed plan"
            )
        return (
            _SCALE_V2_TRAINING_SEED_ROLES,
            SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
            "mind_public_recurrent_training_seed_evidence_v3",
        )
    if registry_digest != CANONICAL_SEED_REGISTRY_SHA256:
        raise RecurrentEvaluationError("artifact training seed registry is unsupported")
    return (
        _CANONICAL_TRAINING_SEED_ROLES,
        RECURRENT_SEED_REGISTRY,
        "mind_public_recurrent_training_seed_evidence_v1",
    )


def _artifact_training_seed_evidence(
    artifact_provenance: Mapping[str, object],
    *,
    seed_plan: RecurrentEvaluationSeedPlan,
) -> dict[str, object]:
    """Derive train/evaluation separation from artifact-bound provenance.

    The legacy caller exclusion list is intentionally not an input. An artifact
    is promotion-eligible only when it binds every observed environment seed to
    the canonical ``train`` or ``curriculum`` role.
    """

    data_metadata_value = artifact_provenance.get("data_metadata")
    data_metadata = (
        data_metadata_value if isinstance(data_metadata_value, Mapping) else {}
    )
    declared_roles_value = data_metadata.get("environment_seed_roles")
    declared_roles = (
        tuple(declared_roles_value)
        if isinstance(declared_roles_value, Sequence)
        and not isinstance(declared_roles_value, (str, bytes))
        and all(isinstance(role, str) for role in declared_roles_value)
        else ()
    )
    required_roles, training_registry, evidence_schema = (
        _artifact_training_seed_contract(
            artifact_provenance,
            seed_plan=seed_plan,
        )
    )
    roles_exact = declared_roles == required_roles

    role_seed_map_value = data_metadata.get("environment_seeds_by_role")
    role_seed_map = (
        role_seed_map_value if isinstance(role_seed_map_value, Mapping) else None
    )
    role_seed_counts: dict[str, int] = {}
    role_seed_values: dict[str, list[int]] = {}
    observed_training_seeds: set[int] = set()
    membership_valid = role_seed_map is not None and set(role_seed_map) == set(
        required_roles
    )
    if membership_valid:
        assert role_seed_map is not None
        for role in required_roles:
            parsed = _optional_validated_seed_sequence(role_seed_map.get(role))
            canonical = training_registry[role]
            if parsed is None or not _is_canonical_ordered_subset(parsed, canonical):
                membership_valid = False
                break
            role_seed_counts[role] = len(parsed)
            role_seed_values[role] = list(parsed)
            observed_training_seeds.update(parsed)

    legacy_training_seeds = _optional_validated_seed_sequence(
        data_metadata.get("training_seeds")
    )
    role_bound_training_seeds = set(observed_training_seeds)
    legacy_consistent = legacy_training_seeds is None or bool(
        membership_valid and set(legacy_training_seeds) == role_bound_training_seeds
    )
    if legacy_training_seeds is not None:
        observed_training_seeds.update(legacy_training_seeds)

    evaluation_seeds = set(seed_plan.broad_seeds) | set(seed_plan.fixture_seeds)
    overlap = sorted(evaluation_seeds & observed_training_seeds)
    complete = bool(
        roles_exact
        and membership_valid
        and legacy_consistent
        and observed_training_seeds
    )
    reasons: list[str] = []
    if not roles_exact:
        reasons.append("canonical_training_roles_missing_or_not_exact")
    if role_seed_map is None:
        reasons.append("artifact_role_bound_training_seed_map_missing")
    elif not membership_valid:
        reasons.append("artifact_training_seeds_not_canonical_for_declared_roles")
    if not legacy_consistent:
        reasons.append("legacy_training_seeds_differ_from_role_bound_seed_map")
    if not observed_training_seeds:
        reasons.append("artifact_observed_training_seeds_missing")
    if overlap:
        reasons.append("artifact_training_and_evaluation_seeds_overlap")

    return {
        "schema_version": evidence_schema,
        "evidence_source": "artifact.provenance.data_metadata",
        "declared_environment_seed_roles": list(declared_roles),
        "required_environment_seed_roles": list(required_roles),
        "canonical_training_roles_exact": roles_exact,
        "canonical_training_membership_valid": membership_valid,
        "legacy_training_seed_evidence_consistent": legacy_consistent,
        "role_bound_provenance_complete": complete,
        "role_seed_counts": role_seed_counts,
        "environment_seeds_by_role": role_seed_values,
        "legacy_training_seeds": (
            list(legacy_training_seeds) if legacy_training_seeds is not None else None
        ),
        "observed_training_seeds": sorted(observed_training_seeds),
        "observed_training_seed_count": len(observed_training_seeds),
        "training_evaluation_overlap_count": len(overlap),
        "training_evaluation_overlap_seeds": overlap,
        "legacy_caller_exclusions_consulted": False,
        "failure_reasons": reasons,
    }


def _validate_seed_plan_report(
    payload: Mapping[str, object],
) -> RecurrentEvaluationSeedPlan:
    expected_keys = {
        "source",
        "digest",
        "environment_seed_role",
        "canonical_registry_role",
        "canonical_role_exact",
        "full_canonical_lockbox_plan",
        "contains_canonical_validation_seed",
        "contains_canonical_lockbox_seed",
        "broad_seeds",
        "fixture_seeds",
        "caller_excluded_training_seeds",
        "excluded_training_seed_count",
        "caller_excluded_training_seeds_used_as_proof",
        "canonical_training_seed_count",
        "train_evaluation_overlap_count",
    }
    if set(payload) != expected_keys:
        raise RecurrentEvaluationError("evaluation seed-plan keys drifted")
    if payload.get("source") != "caller_supplied_evaluation_plan":
        raise RecurrentEvaluationError("evaluation seed-plan source drifted")
    plan = RecurrentEvaluationSeedPlan(
        broad_seeds=_required_sequence(
            payload.get("broad_seeds"), field="seed_plan.broad_seeds"
        ),
        fixture_seeds=_required_sequence(
            payload.get("fixture_seeds"), field="seed_plan.fixture_seeds"
        ),
        excluded_training_seeds=_required_sequence(
            payload.get("caller_excluded_training_seeds"),
            field="seed_plan.caller_excluded_training_seeds",
        ),
        environment_seed_role=payload.get("environment_seed_role"),  # type: ignore[arg-type]
    )
    _reject_unavailable_restricted_evaluation_access(plan)
    if payload.get("digest") != plan.digest:
        raise RecurrentEvaluationError("evaluation seed-plan digest drifted")
    if payload.get("canonical_registry_role") != plan.canonical_registry_role:
        raise RecurrentEvaluationError(
            "evaluation seed-plan canonical registry role drifted"
        )
    if payload.get("canonical_role_exact") is not (
        plan.canonical_registry_role is not None
    ):
        raise RecurrentEvaluationError(
            "evaluation seed-plan canonical-role evidence drifted"
        )
    if (
        payload.get("full_canonical_lockbox_plan")
        is not plan.full_canonical_lockbox_plan
    ):
        raise RecurrentEvaluationError(
            "evaluation seed-plan full-lockbox evidence drifted"
        )
    if (
        payload.get("contains_canonical_validation_seed")
        is not plan.contains_validation_seed
        or payload.get("contains_canonical_lockbox_seed")
        is not plan.contains_lockbox_seed
    ):
        raise RecurrentEvaluationError(
            "evaluation seed-plan restricted membership evidence drifted"
        )
    if payload.get("caller_excluded_training_seeds_used_as_proof") is not False:
        raise RecurrentEvaluationError(
            "caller exclusions must not be treated as training provenance"
        )
    if payload.get("excluded_training_seed_count") != len(plan.excluded_training_seeds):
        raise RecurrentEvaluationError(
            "evaluation seed-plan caller exclusion count drifted"
        )
    expected_training_seeds = _training_seed_set_for_plan(plan)
    if payload.get("canonical_training_seed_count") != len(expected_training_seeds):
        raise RecurrentEvaluationError(
            "evaluation seed-plan canonical training count drifted"
        )
    expected_overlap = len(
        (set(plan.broad_seeds) | set(plan.fixture_seeds)) & expected_training_seeds
    )
    if payload.get("train_evaluation_overlap_count") != expected_overlap:
        raise RecurrentEvaluationError(
            "evaluation seed-plan canonical training overlap drifted"
        )
    return plan


def _validate_artifact_training_seed_evidence(
    evidence: Mapping[str, object],
    *,
    seed_plan: RecurrentEvaluationSeedPlan,
) -> None:
    scale_v1_contract = (
        seed_plan.environment_seed_role
        == RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE
    )
    scale_v2_contract = (
        seed_plan.environment_seed_role
        == RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE
    )
    if scale_v2_contract:
        expected_schema = "mind_public_recurrent_training_seed_evidence_v3"
        expected_roles = _SCALE_V2_TRAINING_SEED_ROLES
        expected_registry = SCALE_DEVELOPMENT_V2_SEED_REGISTRY
    elif scale_v1_contract:
        expected_schema = "mind_public_recurrent_training_seed_evidence_v2"
        expected_roles = _SCALE_TRAINING_SEED_ROLES
        expected_registry = SCALE_DEVELOPMENT_SEED_REGISTRY
    else:
        expected_schema = "mind_public_recurrent_training_seed_evidence_v1"
        expected_roles = _CANONICAL_TRAINING_SEED_ROLES
        expected_registry = RECURRENT_SEED_REGISTRY
    if (
        evidence.get("schema_version") != expected_schema
        or evidence.get("evidence_source") != "artifact.provenance.data_metadata"
        or evidence.get("legacy_caller_exclusions_consulted") is not False
    ):
        raise RecurrentEvaluationError("artifact training-seed evidence drifted")
    required_roles = tuple(
        _required_sequence(
            evidence.get("required_environment_seed_roles"),
            field="artifact training required roles",
        )
    )
    if required_roles != expected_roles:
        raise RecurrentEvaluationError("artifact training required roles drifted")
    declared_roles_value = _required_sequence(
        evidence.get("declared_environment_seed_roles"),
        field="artifact training declared roles",
    )
    if not all(isinstance(role, str) for role in declared_roles_value):
        raise RecurrentEvaluationError(
            "artifact training declared roles must be strings"
        )
    declared_roles = tuple(declared_roles_value)
    expected_roles_exact = declared_roles == expected_roles

    role_seed_map = _required_mapping(
        evidence.get("environment_seeds_by_role"),
        field="artifact training environment_seeds_by_role",
    )
    expected_membership_valid = set(role_seed_map) == set(expected_roles)
    observed_from_roles: set[int] = set()
    expected_role_counts: dict[str, int] = {}
    if expected_membership_valid:
        for role in expected_roles:
            parsed = _optional_validated_seed_sequence(role_seed_map.get(role))
            if parsed is None or not _is_canonical_ordered_subset(
                parsed, expected_registry[role]
            ):
                expected_membership_valid = False
                break
            observed_from_roles.update(parsed)
            expected_role_counts[role] = len(parsed)

    legacy = _optional_validated_seed_sequence(evidence.get("legacy_training_seeds"))
    role_bound_observed = set(observed_from_roles)
    expected_legacy_consistent = legacy is None or bool(
        expected_membership_valid and set(legacy) == role_bound_observed
    )
    observed = set(observed_from_roles)
    if legacy is not None:
        observed.update(legacy)
    observed_payload_value = _required_sequence(
        evidence.get("observed_training_seeds"),
        field="artifact observed training seeds",
    )
    observed_payload = tuple(
        _validated_seed(seed, field="artifact observed training seeds")
        for seed in observed_payload_value
    )
    if (
        len(observed_payload) != len(set(observed_payload))
        or list(observed_payload) != sorted(observed_payload)
        or set(observed_payload) != observed
    ):
        raise RecurrentEvaluationError(
            "artifact observed training seeds differ from role evidence"
        )
    expected_complete = bool(
        expected_roles_exact
        and expected_membership_valid
        and expected_legacy_consistent
        and observed
    )
    overlap = sorted(
        observed & (set(seed_plan.broad_seeds) | set(seed_plan.fixture_seeds))
    )
    if (
        evidence.get("canonical_training_roles_exact") is not expected_roles_exact
        or evidence.get("canonical_training_membership_valid")
        is not expected_membership_valid
        or evidence.get("legacy_training_seed_evidence_consistent")
        is not expected_legacy_consistent
        or evidence.get("role_bound_provenance_complete") is not expected_complete
        or evidence.get("role_seed_counts") != expected_role_counts
        or evidence.get("observed_training_seed_count") != len(observed)
        or evidence.get("training_evaluation_overlap_count") != len(overlap)
        or evidence.get("training_evaluation_overlap_seeds") != overlap
    ):
        raise RecurrentEvaluationError(
            "artifact training-seed evidence does not rederive"
        )
    reasons = _required_sequence(
        evidence.get("failure_reasons"), field="artifact training failure reasons"
    )
    if not all(isinstance(reason, str) and reason for reason in reasons):
        raise RecurrentEvaluationError(
            "artifact training failure reasons must be strings"
        )


def _validate_runtime_reproducibility_provenance(
    runtime: Mapping[str, object],
) -> None:
    if runtime.get("schema_version") != RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION:
        raise RecurrentEvaluationError(
            "evaluation runtime reproducibility schema drifted"
        )
    for field in (
        "python_version",
        "python_implementation",
        "torch_version",
        "source_model_device_observed",
    ):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RecurrentEvaluationError(
                f"evaluation runtime {field} must be a non-empty string"
            )
    numpy_version = runtime.get("numpy_version")
    if numpy_version is not None and (
        not isinstance(numpy_version, str) or not numpy_version
    ):
        raise RecurrentEvaluationError(
            "evaluation runtime numpy_version must be null or a non-empty string"
        )
    platform_evidence = _required_mapping(
        runtime.get("platform"),
        field="execution_provenance.runtime_reproducibility.platform",
    )
    for field in ("system", "release", "machine", "platform_string"):
        if not isinstance(platform_evidence.get(field), str):
            raise RecurrentEvaluationError(
                f"evaluation runtime platform {field} must be a string"
            )
    if (
        runtime.get("requested_device") != "cpu"
        or runtime.get("resolved_device") != "cpu"
        or runtime.get("requested_device_source") != "frozen_cpu_evaluation_contract"
    ):
        raise RecurrentEvaluationError(
            "evaluation runtime requested/resolved device contract drifted"
        )
    if type(runtime.get("torch_deterministic_algorithms_enabled")) is not bool:
        raise RecurrentEvaluationError(
            "evaluation runtime deterministic-algorithms flag must be boolean"
        )


def _validate_evaluation_contract(
    contract: Mapping[str, object],
) -> tuple[str, tuple[int | None, ...]]:
    expected_keys = {
        "ticks",
        "world_horizon_policy",
        "candidate_action_selection",
        "candidate_sampling_seeds",
        "candidate_sampling_seed_count",
        "candidate_sampling_stream_id",
        "candidate_sampling_stream_id_source",
        "candidate_sampling_seed_contract",
        "candidate_recurrent_state",
        "candidate_policy_inputs",
        "candidate_forbidden_inputs",
        "candidate_factory_receives_seed_or_fixture",
        "candidate_replay_verification",
        "strict_zero_unsupported_requested_actions",
        "strict_zero_heuristic_candidate_actions",
    }
    if set(contract) != expected_keys:
        raise RecurrentEvaluationError("evaluation canonical contract keys drifted")
    if (
        contract.get("ticks") != RECURRENT_EVALUATION_TICKS
        or contract.get("world_horizon_policy") != "exact_configured_120_tick_horizon"
        or contract.get("candidate_factory_receives_seed_or_fixture") is not False
        or contract.get("candidate_replay_verification")
        != "every_run_exact_digest_repeat"
        or contract.get("strict_zero_unsupported_requested_actions") is not True
        or contract.get("strict_zero_heuristic_candidate_actions") is not True
    ):
        raise RecurrentEvaluationError("evaluation canonical contract drifted")
    if contract.get("candidate_policy_inputs") != [
        "current_public_ecological_observation",
        "current_public_action_mask",
        "same_agent_previous_public_outcome",
        "same_agent_recurrent_state",
    ]:
        raise RecurrentEvaluationError("evaluation candidate input contract drifted")
    if contract.get("candidate_forbidden_inputs") != [
        "world_seed",
        "fixture_name",
        "private_world_state",
        "evaluation_split_role",
    ]:
        raise RecurrentEvaluationError(
            "evaluation candidate forbidden-input contract drifted"
        )
    if contract.get("candidate_recurrent_state") not in {
        "zero_before_every_decision",
        "one_hidden_state_per_agent",
    }:
        raise RecurrentEvaluationError(
            "evaluation candidate recurrent-state contract drifted"
        )
    selection = _validated_candidate_action_selection(
        contract.get("candidate_action_selection")
    )
    stream_id = _validated_sampling_stream_id(
        contract.get("candidate_sampling_stream_id")
    )
    stream_source = contract.get("candidate_sampling_stream_id_source")
    if stream_source not in {
        "candidate_policy_digest_default",
        "caller_supplied_arm_independent_identifier",
    }:
        raise RecurrentEvaluationError(
            "evaluation candidate sampling-stream source drifted"
        )
    count = contract.get("candidate_sampling_seed_count")
    expected_sampling_seeds = _candidate_sampling_seeds(
        stream_id,
        candidate_action_selection=selection,
        count=count,  # type: ignore[arg-type]
    )
    if count != len(expected_sampling_seeds):
        raise RecurrentEvaluationError(
            "evaluation candidate sampling-seed count drifted"
        )
    reported_values = _required_sequence(
        contract.get("candidate_sampling_seeds"),
        field="evaluation_contract.candidate_sampling_seeds",
    )
    reported_sampling_seeds = tuple(
        _validated_sampling_seed(
            seed, field="evaluation_contract.candidate_sampling_seeds"
        )
        for seed in reported_values
    )
    expected_reported_seeds = tuple(
        seed for seed in expected_sampling_seeds if seed is not None
    )
    if reported_sampling_seeds != expected_reported_seeds:
        raise RecurrentEvaluationError(
            "evaluation candidate sampling seeds do not rederive"
        )
    expected_seed_contract = (
        "none_argmax"
        if expected_sampling_seeds == (None,)
        else ("sha256_namespace_sampling_stream_id_and_replicate_index_first_63_bits")
    )
    if contract.get("candidate_sampling_seed_contract") != expected_seed_contract:
        raise RecurrentEvaluationError(
            "evaluation candidate sampling-seed contract drifted"
        )
    return selection, expected_sampling_seeds


def _expected_replay_checks_from_context(
    context: Mapping[str, object],
    *,
    field: str,
) -> tuple[dict[str, object], ...]:
    policies = _required_mapping(context.get("policies"), field=f"{field}.policies")
    candidate = _required_mapping(
        policies.get("public_recurrent"),
        field=f"{field}.policies.public_recurrent",
    )
    runs = _required_sequence(
        candidate.get("runs"), field=f"{field}.policies.public_recurrent.runs"
    )
    return tuple(
        {
            "context": run["context"],
            "seed": run["seed"],
            "policy_sampling_seed": run["policy_sampling_seed"],
            "digest": run["replay_digest"],
            "outcome_evidence_sha256": run["outcome_evidence_sha256"],
            "passed": True,
        }
        for value in runs
        for run in (_required_mapping(value, field=f"{field} candidate run"),)
    )


def _validate_context_report(
    context: Mapping[str, object],
    *,
    field: str,
    expected_context: str,
    expected_sampling_seeds: Sequence[int | None],
) -> int:
    seeds = _validated_seed_sequence(context.get("seeds"), field=f"{field}.seeds")
    policies = _required_mapping(context.get("policies"), field=f"{field}.policies")
    if set(policies) != set(_POLICY_KEYS):
        raise RecurrentEvaluationError(f"{field} policy set drifted")

    runs_by_policy: dict[str, tuple[Mapping[str, object], ...]] = {}
    for policy_key in _POLICY_KEYS:
        policy = _required_mapping(
            policies.get(policy_key),
            field=f"{field}.policies.{policy_key}",
        )
        run_values = _required_sequence(
            policy.get("runs"),
            field=f"{field}.policies.{policy_key}.runs",
        )
        runs = tuple(
            _required_mapping(run, field=f"{field}.{policy_key}.run")
            for run in run_values
        )
        if not runs:
            raise RecurrentEvaluationError(f"{field} policy run set is empty")
        for run in runs:
            _validate_run(run)
            if run.get("context") != expected_context or run.get("seed") not in seeds:
                raise RecurrentEvaluationError(f"{field} policy run identity drifted")
        expected_aggregate = _aggregate_runs(runs)
        aggregate = _required_mapping(
            policy.get("aggregate"),
            field=f"{field}.policies.{policy_key}.aggregate",
        )
        _nonnegative_int(
            aggregate.get("terminal_nonextinct_run_count"),
            field=f"{field}.{policy_key}.terminal_nonextinct_run_count",
        )
        _nonnegative_int(
            aggregate.get("terminal_alive_agent_total"),
            field=f"{field}.{policy_key}.terminal_alive_agent_total",
        )
        if aggregate != expected_aggregate:
            raise RecurrentEvaluationError(
                f"{field} {policy_key} aggregate differs from raw runs"
            )
        runs_by_policy[policy_key] = runs

    candidate_runs = runs_by_policy["public_recurrent"]
    expected_candidate_grid = tuple(
        (seed, sampling_seed)
        for seed in seeds
        for sampling_seed in expected_sampling_seeds
    )
    observed_candidate_grid = tuple(
        (
            _validated_seed(run.get("seed"), field=f"{field} candidate seed"),
            _validated_optional_sampling_seed(
                run.get("policy_sampling_seed"),
                field=f"{field} candidate policy sampling seed",
            ),
        )
        for run in candidate_runs
    )
    if observed_candidate_grid != expected_candidate_grid:
        raise RecurrentEvaluationError(
            f"{field} candidate identity grid differs from seed and sampling plan"
        )
    expected_control_grid = tuple((seed, None) for seed in seeds)
    for control_key in ("mind_v3_linear", "masked_random"):
        observed_control_grid = tuple(
            (
                _validated_seed(run.get("seed"), field=f"{field} {control_key} seed"),
                _validated_optional_sampling_seed(
                    run.get("policy_sampling_seed"),
                    field=f"{field} {control_key} policy sampling seed",
                ),
            )
            for run in runs_by_policy[control_key]
        )
        if observed_control_grid != expected_control_grid:
            raise RecurrentEvaluationError(
                f"{field} {control_key} identity grid differs from seed plan"
            )
    reported_sampling_seeds = tuple(
        _validated_sampling_seed(seed, field=f"{field}.candidate_sampling_seeds")
        for seed in _required_sequence(
            context.get("candidate_sampling_seeds"),
            field=f"{field}.candidate_sampling_seeds",
        )
    )
    if reported_sampling_seeds != tuple(
        seed for seed in expected_sampling_seeds if seed is not None
    ):
        raise RecurrentEvaluationError(
            f"{field} candidate sampling seeds differ from evaluation contract"
        )
    candidate_runs_per_environment = context.get("candidate_run_count_per_environment")
    if (
        isinstance(candidate_runs_per_environment, bool)
        or not isinstance(candidate_runs_per_environment, int)
        or candidate_runs_per_environment <= 0
        or len(candidate_runs) != len(seeds) * candidate_runs_per_environment
    ):
        raise RecurrentEvaluationError(
            f"{field} candidate run multiplicity differs from environments"
        )
    if context.get("controls_run_once_per_environment") is not True:
        raise RecurrentEvaluationError(
            f"{field} controls-once-per-environment contract drifted"
        )
    for control_key in ("mind_v3_linear", "masked_random"):
        if len(runs_by_policy[control_key]) != len(seeds):
            raise RecurrentEvaluationError(
                f"{field} {control_key} run count differs from environments"
            )

    control_by_seed = {
        control_key: {int(run["seed"]): run for run in runs_by_policy[control_key]}
        for control_key in ("mind_v3_linear", "masked_random")
    }
    paired = _required_mapping(
        context.get("paired_deltas"), field=f"{field}.paired_deltas"
    )
    expected_paired = {
        "candidate_minus_mind_v3_linear": _paired_deltas(
            candidate_runs,
            tuple(
                control_by_seed["mind_v3_linear"][int(candidate["seed"])]
                for candidate in candidate_runs
            ),
        ),
        "candidate_minus_masked_random": _paired_deltas(
            candidate_runs,
            tuple(
                control_by_seed["masked_random"][int(candidate["seed"])]
                for candidate in candidate_runs
            ),
        ),
    }
    if paired != expected_paired:
        raise RecurrentEvaluationError(
            f"{field} paired deltas differ from raw candidate/control runs"
        )

    expected_sampling_analysis = _candidate_sampling_analysis(candidate_runs)
    sampling_analysis = _required_mapping(
        context.get("candidate_sampling_analysis"),
        field=f"{field}.candidate_sampling_analysis",
    )
    if sampling_analysis != expected_sampling_analysis:
        raise RecurrentEvaluationError(
            f"{field} candidate sampling analysis differs from raw runs"
        )
    if (
        sampling_analysis.get("sampling_runs_per_environment")
        != candidate_runs_per_environment
    ):
        raise RecurrentEvaluationError(
            f"{field} candidate sampling multiplicity drifted"
        )
    return len(candidate_runs)


def _reject_ambiguous_v3_terminal_fields(value: object, *, field: str) -> None:
    if isinstance(value, Mapping):
        ambiguous = sorted(set(value) & _AMBIGUOUS_V3_TERMINAL_FIELDS)
        if ambiguous:
            raise RecurrentEvaluationError(
                f"{field} contains ambiguous v3 terminal fields: {ambiguous!r}"
            )
        for key, nested in value.items():
            _reject_ambiguous_v3_terminal_fields(
                nested,
                field=f"{field}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_ambiguous_v3_terminal_fields(
                nested,
                field=f"{field}[{index}]",
            )


def _validated_unpinned_digest_label(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(
        UNPINNED_NONCANDIDATE_DIGEST_PREFIX
    ):
        raise RecurrentEvaluationError(
            "synthetic noncandidate digest label must start with "
            f"{UNPINNED_NONCANDIDATE_DIGEST_PREFIX!r}"
        )
    digest = value.removeprefix(UNPINNED_NONCANDIDATE_DIGEST_PREFIX)
    if len(digest) != 64 or digest.lower() != digest:
        raise RecurrentEvaluationError(
            "synthetic noncandidate digest label must end in lowercase sha256"
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise RecurrentEvaluationError(
            "synthetic noncandidate digest label must end in lowercase sha256"
        ) from exc
    return value


def _validated_expected_source_commit(value: object) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        raise RecurrentEvaluationError(
            "expected_source_commit must be an external 40- or 64-character Git pin"
        )
    if value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RecurrentEvaluationError(
            "expected_source_commit must be a lowercase hexadecimal Git pin"
        )
    return value


def _validated_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentEvaluationError(
            f"{field} must be a lowercase hexadecimal SHA256"
        )
    return value


def _validated_candidate_action_selection(value: object) -> str:
    allowed = {
        PUBLIC_RECURRENT_ARGMAX_SELECTION,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
    }
    if not isinstance(value, str) or value not in allowed:
        raise RecurrentEvaluationError(
            f"candidate_action_selection must be one of {sorted(allowed)!r}"
        )
    return value


def _candidate_sampling_seeds(
    sampling_stream_id: str,
    *,
    candidate_action_selection: str,
    count: int,
) -> tuple[int | None, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
        raise RecurrentEvaluationError(
            "candidate_sampling_seed_count must be an integer in [1, 32]"
        )
    if candidate_action_selection == PUBLIC_RECURRENT_ARGMAX_SELECTION:
        return (None,)
    if candidate_action_selection != PUBLIC_RECURRENT_SAMPLED_SELECTION:
        raise RecurrentEvaluationError("candidate action selection drifted")
    resolved_stream_id = _validated_sampling_stream_id(sampling_stream_id)

    seeds = tuple(
        int.from_bytes(
            hashlib.sha256(
                (
                    f"{_PUBLIC_RECURRENT_SAMPLING_SEED_NAMESPACE}|"
                    f"{resolved_stream_id}|replicate-{index:04d}"
                ).encode("ascii")
            ).digest()[:8],
            byteorder="big",
        )
        & (2**63 - 1)
        for index in range(count)
    )
    if len(set(seeds)) != len(seeds):
        raise RecurrentEvaluationError("candidate sampling seeds collided")
    return seeds


def _validated_sampling_stream_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
    ):
        raise RecurrentEvaluationError(
            "candidate_sampling_stream_id must be a non-empty trimmed string "
            "of at most 512 characters"
        )
    return value


def _validated_evaluation_workers(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
        raise RecurrentEvaluationError(
            "evaluation_workers must be an integer in [1, 64]"
        )
    return value


def _artifact_digest(
    loaded: LoadedRecurrentArtifact | LoadedFrozenRecurrentPolicyArtifact,
) -> str:
    digest = loaded.artifact.get("artifact_sha256")
    return _validated_sha256(digest, field="loaded artifact digest")


def _artifact_model_contract(
    loaded: LoadedRecurrentArtifact | LoadedFrozenRecurrentPolicyArtifact,
) -> str:
    model = _required_mapping(loaded.artifact.get("model"), field="artifact.model")
    contract = model.get("contract_version")
    if not isinstance(contract, str) or not contract:
        raise RecurrentEvaluationError("artifact model contract is invalid")
    return contract


def _artifact_feed_forward_history_ablation(
    loaded: LoadedRecurrentArtifact | LoadedFrozenRecurrentPolicyArtifact,
) -> bool:
    provenance = _required_mapping(
        loaded.artifact.get("provenance"),
        field="artifact.provenance",
    )
    training_config = _required_mapping(
        provenance.get("training_config"),
        field="artifact.provenance.training_config",
    )
    value = training_config.get("feed_forward_history_ablation", False)
    if type(value) is not bool:
        raise RecurrentEvaluationError(
            "artifact feed_forward_history_ablation must be an exact boolean"
        )
    return value


def _validated_fixture_names(fixture_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fixture_names, (str, bytes)):
        raise RecurrentEvaluationError("fixture_names must be a sequence of names")
    selected = tuple(fixture_names)
    if not selected:
        raise RecurrentEvaluationError("fixture_names must not be empty")
    if any(not isinstance(name, str) or not name for name in selected):
        raise RecurrentEvaluationError("fixture_names must contain non-empty strings")
    if len(set(selected)) != len(selected):
        raise RecurrentEvaluationError("fixture_names must be unique")
    unsupported = [name for name in selected if name not in CONTROLLED_FIXTURE_NAMES]
    if unsupported:
        raise RecurrentEvaluationError(
            "unsupported fixture names: " + ", ".join(sorted(unsupported))
        )
    return selected


def _strict_valid_actions(action_mask: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(action_mask, Mapping):
        raise RecurrentEvaluationError("action mask must be a mapping")
    expected = set(ACTION_NAMES)
    observed = set(action_mask)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RecurrentEvaluationError(
            f"action mask keys differ; missing={missing!r}, extra={extra!r}"
        )
    if any(type(action_mask[action]) is not bool for action in ACTION_NAMES):
        raise RecurrentEvaluationError("action mask values must be exact booleans")
    valid = tuple(action for action in ACTION_NAMES if action_mask[action] is True)
    if not valid:
        raise RecurrentEvaluationError("action mask has no valid action")
    return valid


def _validated_seed_sequence(values: object, *, field: str) -> tuple[int, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise RecurrentEvaluationError(f"{field} must be a non-empty sequence")
    parsed = tuple(_validated_seed(value, field=field) for value in values)
    if len(set(parsed)) != len(parsed):
        raise RecurrentEvaluationError(f"{field} must contain unique seeds")
    return parsed


def _optional_validated_seed_sequence(values: object) -> tuple[int, ...] | None:
    try:
        return _validated_seed_sequence(values, field="artifact training seeds")
    except RecurrentEvaluationError:
        return None


def _is_canonical_ordered_subset(
    values: Sequence[int],
    canonical: Sequence[int],
) -> bool:
    canonical_positions = {seed: index for index, seed in enumerate(canonical)}
    try:
        positions = tuple(canonical_positions[seed] for seed in values)
    except KeyError:
        return False
    return bool(positions) and all(
        left < right for left, right in zip(positions, positions[1:])
    )


def _validated_evaluation_seed_role(value: object) -> str:
    allowed = {
        RECURRENT_EVALUATION_DEVELOPMENT_SEED_ROLE,
        *_EVALUATION_SEED_ROLE_TO_REGISTRY_ROLE,
    }
    if not isinstance(value, str) or value not in allowed:
        raise RecurrentEvaluationError(
            "environment_seed_role must be one of " + ", ".join(sorted(allowed))
        )
    return value


def _reject_unavailable_restricted_evaluation_access(
    seed_plan: RecurrentEvaluationSeedPlan,
) -> None:
    if seed_plan.contains_validation_seed:
        raise RecurrentEvaluationError(
            "validation evaluation requires external candidate-freeze and "
            "limited-access authorization; none is available in this repository"
        )
    if seed_plan.contains_lockbox_seed:
        raise RecurrentEvaluationError(
            "lockbox evaluation requires an external one-use authorization and "
            "consumption ledger; none is available in this repository"
        )


def _validated_seed(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentEvaluationError(f"{field} must contain positive integers")
    if value > 2_147_483_647:
        raise RecurrentEvaluationError(f"{field} seed exceeds the supported range")
    return value


def _validated_sampling_seed(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise RecurrentEvaluationError(f"{field} must be an integer in [0, 2**63)")
    return value


def _validated_optional_sampling_seed(
    value: object,
    *,
    field: str,
) -> int | None:
    return None if value is None else _validated_sampling_seed(value, field=field)


def _required_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentEvaluationError(f"{field} must be a mapping")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise RecurrentEvaluationError(
            f"{field} field set drifted: expected={sorted(expected)!r} "
            f"observed={sorted(observed)!r}"
        )


def _required_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecurrentEvaluationError(f"{field} must be a sequence")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentEvaluationError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentEvaluationError(f"{field} must be finite")
    return parsed


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentEvaluationError(f"{field} must be a nonnegative integer")
    return value


def _context_label(fixture_name: str | None) -> str:
    return "broad_default" if fixture_name is None else f"fixture:{fixture_name}"


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecurrentEvaluationError(
            f"evaluation evidence is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "MASKED_RANDOM_ACTION_SOURCE",
    "MASKED_RANDOM_POLICY_ID",
    "MASKED_RANDOM_POLICY_VERSION",
    "RECURRENT_EVALUATION_EXECUTION_SCHEMA_VERSION",
    "RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION",
    "RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE",
    "RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE",
    "RECURRENT_EVALUATION_SCHEMA_VERSION",
    "RECURRENT_EVALUATION_TICKS",
    "UNPINNED_NONCANDIDATE_DIGEST_PREFIX",
    "MaskedRandomPolicy",
    "RecurrentEvaluationError",
    "RecurrentEvaluationSeedPlan",
    "evaluate_recurrent_artifact",
    "evaluate_recurrent_model",
    "validate_recurrent_evaluation_report",
]
