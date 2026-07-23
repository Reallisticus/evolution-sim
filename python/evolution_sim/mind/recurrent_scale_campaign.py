from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_counterfactual_comparison import (
    BASE_ARM,
    EXACT_ARM,
    SHUFFLED_ARM,
)
from evolution_sim.mind.recurrent_counterfactual_branch import (
    RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE,
    RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE,
    RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY,
)
from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
    RECURRENT_COUNTERFACTUAL_AGGREGATE_AUXILIARY_SCHEMA_VERSION,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION,
)
from evolution_sim.mind.recurrent_evaluation import (
    RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE,
    RecurrentEvaluationSeedPlan,
)
from evolution_sim.mind.recurrent_experiment import RECURRENT_TRAINING_SCENARIOS
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
)


RECURRENT_SCALE_CAMPAIGN_PREREGISTRATION_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_scale_campaign_preregistration_v2"
)
RECURRENT_SCALE_ARM_REPORT_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_scale_arm_report_v2"
)
RECURRENT_SCALE_CAMPAIGN_ANALYSIS_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_scale_campaign_analysis_v2"
)
RECURRENT_SCALE_CAMPAIGN_POLICY = (
    "public_recurrent_ippo_multi_tape_terminal_counterfactual_scale_v2"
)
RECURRENT_SCALE_ARMS = (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
RECURRENT_SCALE_UPDATE_COUNT = 16
RECURRENT_SCALE_WORLDS_PER_UPDATE = 16
RECURRENT_SCALE_ROLLOUT_TICKS = 120
RECURRENT_SCALE_SELECTION_SEED_COUNT = 8
RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT = 4
RECURRENT_SCALE_CONCURRENT_CUDA_ARM_PROCESSES = 3
RECURRENT_SCALE_TOTAL_TRAINING_WORLDS = (
    len(RECURRENT_SCALE_ARMS)
    * len(SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"])
    * RECURRENT_SCALE_UPDATE_COUNT
    * RECURRENT_SCALE_WORLDS_PER_UPDATE
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RecurrentScaleCampaignError(ValueError):
    """Raised when a scale campaign stops matching its preregistration."""


def build_recurrent_scale_campaign_preregistration(
    *,
    source_commit: str,
    source_manifest_sha256: str,
) -> dict[str, object]:
    preregistration = _build_recurrent_scale_campaign_preregistration(
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
    )
    validate_recurrent_scale_campaign_preregistration(preregistration)
    return preregistration


def _build_recurrent_scale_campaign_preregistration(
    *,
    source_commit: str,
    source_manifest_sha256: str,
) -> dict[str, object]:
    """Build the immutable multi-learner, three-arm development protocol."""

    _source_commit(source_commit)
    _sha256(source_manifest_sha256, field="source_manifest_sha256")
    learner_seeds = list(SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"])
    selection_seeds = list(
        SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_selection"][
            :RECURRENT_SCALE_SELECTION_SEED_COUNT
        ]
    )
    preregistration: dict[str, object] = {
        "schema_version": RECURRENT_SCALE_CAMPAIGN_PREREGISTRATION_SCHEMA_VERSION,
        "policy": RECURRENT_SCALE_CAMPAIGN_POLICY,
        "source": {
            "commit": source_commit,
            "manifest_sha256": source_manifest_sha256,
            "clean_tree_required_for_every_arm": True,
        },
        "seed_contract": {
            "registry_version": SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
            "registry_sha256": SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
            "learner_role": "scale_v2_learner",
            "learner_seeds": learner_seeds,
            "learner_specific_policy_and_counterfactual_rng_namespaces": True,
            "arm_paired_within_learner": True,
            "optimization_roles": ["scale_v2_train", "scale_v2_curriculum"],
            "selection_role": "scale_v2_selection",
            "selection_seeds": selection_seeds,
            "validation_available": False,
            "validation_accessed": False,
            "lockbox_available": False,
            "lockbox_accessed": False,
        },
        "arms": {
            "order": list(RECURRENT_SCALE_ARMS),
            "base": {
                "name": BASE_ARM,
                "counterfactual_collection": False,
                "counterfactual_auxiliary": False,
            },
            "exact": {
                "name": EXACT_ARM,
                "counterfactual_collection": True,
                "counterfactual_auxiliary": "exact_aggregate_targets",
            },
            "shuffled": {
                "name": SHUFFLED_ARM,
                "counterfactual_collection": True,
                "counterfactual_auxiliary": (
                    "deterministic_valid_action_label_shuffled_control"
                ),
                "permutation_seed_derivation": (
                    "sha256(campaign_digest,learner_seed,public_context)"
                ),
            },
        },
        "training": {
            "updates_per_arm_learner": RECURRENT_SCALE_UPDATE_COUNT,
            "worlds_per_update": RECURRENT_SCALE_WORLDS_PER_UPDATE,
            "worlds_per_arm_learner": (
                RECURRENT_SCALE_UPDATE_COUNT * RECURRENT_SCALE_WORLDS_PER_UPDATE
            ),
            "total_worlds_all_arms_learners": (RECURRENT_SCALE_TOTAL_TRAINING_WORLDS),
            "rollout_ticks": RECURRENT_SCALE_ROLLOUT_TICKS,
            "scenarios": list(RECURRENT_TRAINING_SCENARIOS),
            "model": {
                "encoder_size": 192,
                "hidden_size": 192,
                "recurrent_layers": 1,
                "input_contract": "public_policy_input_only",
            },
            "ppo": {
                "learning_rate": 0.0002,
                "adam_epsilon": 1.0e-8,
                "gamma": 0.997,
                "gae_lambda": 0.97,
                "policy_clip_range": 0.15,
                "value_clip_range": 0.2,
                "value_loss_coefficient": 0.5,
                "entropy_coefficient": 0.02,
                "update_epochs": 4,
                "sequence_minibatch_size": 16,
                "tbptt_steps": RECURRENT_SCALE_ROLLOUT_TICKS,
                "burn_in_steps": 16,
                "max_gradient_norm": 0.5,
                "normalize_advantages": True,
                "advantage_epsilon": 1.0e-8,
                "target_kl": 0.02,
                "post_step_kl_gate": True,
                "post_step_kl_gate_scope": (
                    "unchanged_optimizer_minibatch_vs_frozen_update_start_policy"
                ),
                "world_balanced_loss": True,
                "feed_forward_history_ablation": False,
            },
        },
        "counterfactual": {
            "collection_contract_version": (
                RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION
            ),
            "source_branch_row_schema_version": (
                RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION
            ),
            "aggregate_row_schema_version": (
                RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION
            ),
            "aggregate_auxiliary_schema_version": (
                RECURRENT_COUNTERFACTUAL_AGGREGATE_AUXILIARY_SCHEMA_VERSION
            ),
            "bundles_per_update": 2,
            "branch_tick_candidates": [16, 40, 64, 72],
            "branch_selection": (
                "deterministic_stratified_rotation_with_cyclic_policy_visible_"
                "fallback_without_future_outcomes"
            ),
            "relative_horizons": [16, 48],
            "relative_horizon_weights": [0.25, 0.25],
            "absolute_terminal_target_tick": RECURRENT_SCALE_ROLLOUT_TICKS,
            "absolute_terminal_weight": 0.5,
            "independent_rng_tapes_per_branch": 8,
            "same_checkpoint_and_rng_tape_across_forced_actions": True,
            "continuation_rng_retape_boundary": (
                RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY
            ),
            "continuation_environment_tape_seed_namespace": (
                RECURRENT_COUNTERFACTUAL_CONTINUATION_ENVIRONMENT_TAPE_SEED_NAMESPACE
            ),
            "continuation_policy_tape_seed_namespace": (
                RECURRENT_COUNTERFACTUAL_CONTINUATION_POLICY_TAPE_SEED_NAMESPACE
            ),
            "fixed_source_prefix_through_focal_natural_draw": True,
            "horizon_includes_branch_tick": True,
            "same_tick_post_focal_consequences_governed_by_tape": True,
            "uncertainty": {
                "estimator": "paired_action_delta_mean_variance_standard_error",
                "conservative_score": "mean_delta_minus_lambda_times_standard_error",
                "lambda": 0.5,
                "survival_probability_reported": True,
            },
            "outcome_scalarization": {
                "focal_discounted_return_weight": 1.0,
                "focal_terminal_alive_weight": 1.0,
                "population_alive_weight": 0.05,
                "births_during_horizon_weight": 0.02,
                "deaths_during_horizon_weight": -0.02,
                "implicit_weight_normalization": False,
            },
            "auxiliary": {
                "temperature": 0.5,
                "advantage_clip": 2.0,
                "policy_improvement_coefficient": 2.0,
                "behavior_kl_coefficient": 1.0,
                "value_loss_coefficient": 0.0,
                "value_target_mode": "disabled",
                "critic_disabled_reason": (
                    "fixed_tick_survivors_are_truncated_and_no_frozen_bootstrap_"
                    "target_is_serialized"
                ),
                "learning_rate_multiplier": 0.25,
                "max_gradient_norm": 0.5,
                "mean_behavior_kl_limit": 0.003,
                "max_state_behavior_kl_limit": 0.015,
                "one_transactional_step_per_ppo_update": True,
                "retry_after_rejection": False,
            },
        },
        "evaluation": {
            "role": RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE,
            "seeds": selection_seeds,
            "fixtures": ["carrion_only"],
            "horizon_ticks": RECURRENT_SCALE_ROLLOUT_TICKS,
            "selection_modes": [
                "deterministic_masked_argmax",
                "replay_deterministic_masked_sampling",
            ],
            "sampled_policy_streams_per_environment": (
                RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
            ),
            "same_environment_and_policy_sampling_streams_across_arms": True,
            "artifacts_reloaded_on_cpu_before_evaluation": True,
        },
        "execution": {
            "training_device_type": "cuda",
            "concurrent_cuda_arm_processes": (
                RECURRENT_SCALE_CONCURRENT_CUDA_ARM_PROCESSES
            ),
            "runtime_provenance_required": True,
            "same_runtime_provenance_all_24_cells": True,
            "dependency_name_version_freeze_required": True,
            "worker_topology_pinned_before_first_cell": True,
            "checkpoint_binds_runtime_provenance_digest": True,
            "cumulative_elapsed_time_across_training_resumes": True,
        },
        "acceptance": {
            "primary_estimand": (
                "learner_paired_exact_minus_max_base_shuffled_carrion_terminal_alive"
            ),
            "exact_beats_both_controls_learner_count_min": 7,
            "exact_nonzero_carrion_survival_learner_count_min": 6,
            "exact_beats_shuffled_learner_count_min": 7,
            "argmax_carrion_improves_learner_count_min": 4,
            "dominant_requested_action_share_max": 0.5,
            "dominant_requested_action_share_scope": (
                "maximum_over_learner_selection_mode_and_context_after_"
                "aggregating_requested_action_counts_across_paired_runs"
            ),
            "heuristic_action_source_count_max": 0,
            "unsupported_requested_action_count_max": 0,
            "broad_alive_and_birth_strict_per_seed_nonregression": True,
            "broad_nonregression_comparators": [
                "fixed_mind_v3_linear_control",
                "base_recurrent_ppo",
                "counterfactual_label_shuffled_control",
            ],
            "paired_unit": "learner_seed",
            "sampled_primary_statistic": (
                "sum_terminal_alive_agents_across_8_environment_seeds_and_4_"
                "arm_paired_policy_sampling_streams"
            ),
            "argmax_statistic": (
                "sum_terminal_alive_agents_across_8_environment_seeds"
            ),
            "control_comparison": "strict_greater_than_max_base_and_shuffled",
            "ties_pass": False,
            "minimum_terminal_alive_agent_difference": 1,
            "accepted_auxiliary_updates_per_treatment_run_min": 12,
            "treatment_parameter_delta_l2_sum_min_exclusive": 0.0,
            "every_accepted_auxiliary_transaction_within_preregistered_kl_bounds": (
                True
            ),
            "validation_and_lockbox_must_remain_sealed": True,
            "all_learner_runs_required": True,
            "gate_relaxation_authorized": False,
        },
        "durability": {
            "atomic_update_checkpoints": True,
            "checkpoint_contains_optimizer_and_rng_state": True,
            "checkpoint_runtime_policy_eligible": False,
            "frozen_policy_artifact_per_arm_learner": True,
            "exact_cpu_replay_probe_required": True,
            "full_world_replay_manifest_required": True,
        },
        "research_basis": {
            "frozen_context_leave_one_out_credit": "arxiv:2603.06859",
            "common_random_numbers_and_terminal_causal_credit": ("arxiv:2607.16999v1"),
            "uncertainty_pessimism": "arxiv:2404.06188",
            "joint_trust_region_context": "arxiv:2508.10340",
            "paper_results_treated_as_hypotheses_not_repo_evidence": True,
        },
        "lifecycle": {
            "development_only": True,
            "campaign_slice_consumed": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
            "validation_seeds_accessed": False,
            "lockbox_seeds_accessed": False,
        },
    }
    preregistration["exact_digest"] = stable_payload_digest(preregistration)
    return preregistration


def validate_recurrent_scale_campaign_preregistration(
    preregistration: Mapping[str, object],
) -> None:
    if not isinstance(preregistration, Mapping):
        raise RecurrentScaleCampaignError("preregistration must be a mapping")
    source = _mapping(preregistration.get("source"), field="source")
    expected = _build_recurrent_scale_campaign_preregistration(
        source_commit=_source_commit(source.get("commit")),
        source_manifest_sha256=_sha256(
            source.get("manifest_sha256"), field="source.manifest_sha256"
        ),
    )
    payload = copy.deepcopy(dict(preregistration))
    supplied_digest = _sha256(payload.pop("exact_digest", None), field="exact_digest")
    if stable_payload_digest(payload) != supplied_digest:
        raise RecurrentScaleCampaignError("preregistration exact digest mismatched")
    if dict(preregistration) != expected:
        raise RecurrentScaleCampaignError("preregistration contract drifted")
    if preregistration.get("schema_version") != (
        RECURRENT_SCALE_CAMPAIGN_PREREGISTRATION_SCHEMA_VERSION
    ):
        raise RecurrentScaleCampaignError("preregistration schema drifted")
    training = _mapping(preregistration.get("training"), field="training")
    if training.get("total_worlds_all_arms_learners") != (
        RECURRENT_SCALE_TOTAL_TRAINING_WORLDS
    ):
        raise RecurrentScaleCampaignError("total training-world budget drifted")
    seed_contract = _mapping(
        preregistration.get("seed_contract"), field="seed_contract"
    )
    if (
        seed_contract.get("registry_sha256")
        != SCALE_DEVELOPMENT_V2_CANONICAL_SHA256
        or seed_contract.get("validation_accessed") is not False
        or seed_contract.get("lockbox_accessed") is not False
    ):
        raise RecurrentScaleCampaignError("scale seed contract drifted")


def recurrent_scale_selection_seed_plan() -> RecurrentEvaluationSeedPlan:
    selection = SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_selection"][
        :RECURRENT_SCALE_SELECTION_SEED_COUNT
    ]
    excluded = (
        *SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_train"],
        *SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_curriculum"],
    )
    return RecurrentEvaluationSeedPlan(
        broad_seeds=selection,
        fixture_seeds=selection,
        excluded_training_seeds=excluded,
        environment_seed_role=RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE,
    )


def recurrent_scale_arm_run_id(*, learner_seed: int, arm: str) -> str:
    if learner_seed not in SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"]:
        raise RecurrentScaleCampaignError("learner seed is not in scale_v2_learner")
    if arm not in RECURRENT_SCALE_ARMS:
        raise RecurrentScaleCampaignError(f"unsupported scale arm: {arm!r}")
    return f"scale-v2-learner-{learner_seed}-{arm}"


def write_atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
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
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_strict_json(path: str | Path) -> dict[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrentScaleCampaignError(f"failed to load JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecurrentScaleCampaignError("JSON root must be an object")
    return payload


def source_file_hash_manifest(repository_root: str | Path) -> dict[str, object]:
    root = Path(repository_root).resolve()
    source_root = root / "python" / "evolution_sim"
    paths = sorted(source_root.rglob("*.py"))
    for repository_file in ("package.json", "requirements-mind-ml.txt"):
        candidate = root / repository_file
        if candidate.is_file():
            paths.append(candidate)
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    if not files:
        raise RecurrentScaleCampaignError("source manifest found no files")
    return {
        "hash_algorithm": "sha256",
        "path_contract": (
            "repository_relative_sorted_runtime_python_package_and_mind_requirements"
        ),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": stable_payload_digest(files),
    }


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentScaleCampaignError(f"{field} must be a mapping")
    return value


def _source_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise RecurrentScaleCampaignError("source commit must be a full Git hash")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecurrentScaleCampaignError(f"{field} must be lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecurrentScaleCampaignError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise RecurrentScaleCampaignError(f"non-finite JSON constant: {value}")


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentScaleCampaignError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentScaleCampaignError(f"{field} must be finite")
    return parsed
