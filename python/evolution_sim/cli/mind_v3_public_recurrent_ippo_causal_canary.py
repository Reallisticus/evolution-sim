from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from evolution_sim.mind.candidate_campaign import write_json
from evolution_sim.mind.evaluation_harness import CONTROLLED_FIXTURE_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED,
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS,
    CounterfactualHorizonScalarization,
    RecurrentCounterfactualAuxiliaryConfig,
    RecurrentCounterfactualAuxiliaryStepConfig,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RecurrentCounterfactualCollectionConfig,
    recurrent_counterfactual_collection_config_payload,
)
from evolution_sim.mind.recurrent_evaluation import (
    RECURRENT_EVALUATION_SELECTION_SEED_ROLE,
    UNPINNED_NONCANDIDATE_DIGEST_PREFIX,
    RecurrentEvaluationSeedPlan,
    evaluate_recurrent_model,
)
from evolution_sim.mind.recurrent_experiment import (
    MAX_RECURRENT_ROLLOUT_WORKERS,
    RECURRENT_TRAINING_SCENARIOS,
    RecurrentCounterfactualExperimentConfig,
    RecurrentExperimentRunner,
    build_recurrent_training_schedule,
    recurrent_training_run_payload,
)
from evolution_sim.mind.recurrent_policy import (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
    frozen_cpu_model_copy,
)
from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig
from evolution_sim.mind.recurrent_seed_registry import (
    CANONICAL_SEED_REGISTRY_SHA256,
    RECURRENT_SEED_REGISTRY,
    SEED_ROLE_POLICIES,
)


CAUSAL_CANARY_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_development_causal_canary_v4"
)
CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION = (
    "mind_v3_recurrent_causal_canary_preregistration_v4"
)
CAUSAL_CANARY_POLICY_VERSION = "public_recurrent_ippo_in_memory_causal_canary_v4"
CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_causal_canary_runtime_reproducibility_v1"
)
DEFAULT_REPORT_PATH = Path(
    "output/mind/mind-v3-public-recurrent-ippo-development-causal-canary.json"
)
ACTION_SELECTION_MODES: tuple[str, str] = (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_LIFECYCLE_FLAGS: tuple[str, ...] = (
    "campaign_slice_requested",
    "campaign_training_slice_consumed",
    "campaign_candidate_registered",
    "campaign_candidate_selected",
    "training_artifact_created",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "runtime_integration_authorized",
    "runtime_integrated",
    "promotion_evidence_eligible",
    "promotion_authorized",
    "promotion_attempted",
    "promoted",
    "validation_seeds_accessed",
    "validation_slice_consumed",
    "validation_run",
    "validation_authorized",
    "lockbox_seeds_accessed",
    "lockbox_slice_consumed",
    "lockbox_opened",
    "lockbox_run",
    "lockbox_authorized",
)
_CAUSAL_METRICS: tuple[str, ...] = (
    "terminal_alive",
    "births",
    "deaths",
    "reward_total",
    "dominant_requested_action_share",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
)
_MAX_EVALUATION_WORKERS = 64


class CausalCanaryError(ValueError):
    """Raised when the development canary contract fails closed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible, unpinned before/train/after causal canary for "
            "public recurrent IPPO. This command cannot create an artifact, "
            "select a campaign candidate, or access validation/lockbox seeds."
        )
    )
    parser.add_argument("--development-run", action="store_true")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--worlds-per-update", type=int, default=16)
    parser.add_argument("--rollout-ticks", type=int, default=64)
    parser.add_argument(
        "--scenarios",
        default=",".join(RECURRENT_TRAINING_SCENARIOS),
    )
    parser.add_argument(
        "--evaluation-fixtures",
        default="carrion_only",
    )
    parser.add_argument("--selection-seed-count", type=int, default=3)
    parser.add_argument("--selection-seed-offset", type=int, default=0)
    parser.add_argument(
        "--candidate-sampling-seed-count",
        type=int,
        default=1,
        help=(
            "Independent replay-deterministic policy-sampling streams per "
            "environment for sampled evaluation; argmax still runs once."
        ),
    )
    parser.add_argument(
        "--learner-seed",
        type=int,
        default=RECURRENT_SEED_REGISTRY["learner_development"][0],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help=(
            "Spawned simulator workers for ordered frozen-policy rollout "
            "collection; 1 uses the sequential reference path."
        ),
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=1,
        help=(
            "Spawned workers for ordered environment-level evaluation; "
            "1 uses the sequential reference path."
        ),
    )

    parser.add_argument("--encoder-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--policy-clip-range", type=float, default=0.2)
    parser.add_argument("--value-clip-range", type=float, default=0.2)
    parser.add_argument("--value-loss-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--sequence-minibatch-size", type=int, default=8)
    parser.add_argument("--tbptt-steps", type=int, default=32)
    parser.add_argument(
        "--full-sequence-tbptt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--advantage-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument(
        "--world-balanced-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--feed-forward-history-ablation", action="store_true")

    counterfactual_mode = parser.add_mutually_exclusive_group()
    counterfactual_mode.add_argument(
        "--counterfactual-exact-auxiliary",
        "--enable-counterfactual-auxiliary",
        dest="counterfactual_exact_auxiliary",
        action="store_true",
        help=(
            "Opt in to one exact-branch transactional auxiliary step after "
            "each PPO update."
        ),
    )
    counterfactual_mode.add_argument(
        "--counterfactual-label-shuffled-control",
        action="store_true",
        help=(
            "Run the same auxiliary path after deterministically permuting "
            "scalarized labels among currently valid actions."
        ),
    )
    parser.add_argument(
        "--counterfactual-permutation-seed",
        type=int,
        help=(
            "Required only for --counterfactual-label-shuffled-control; no "
            "implicit seed is permitted."
        ),
    )
    parser.add_argument("--counterfactual-horizons", default="8,32,64")
    parser.add_argument(
        "--counterfactual-horizon-weights",
        default="0.25,0.5,0.25",
    )
    parser.add_argument(
        "--counterfactual-branch-ticks",
        default="16,32,48,64",
    )
    parser.add_argument("--counterfactual-bundles-per-update", type=int, default=2)
    parser.add_argument("--counterfactual-temperature", type=float, default=1.0)
    parser.add_argument("--counterfactual-advantage-clip", type=float, default=2.0)
    parser.add_argument(
        "--counterfactual-policy-coefficient",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--counterfactual-behavior-kl-coefficient",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--counterfactual-step-learning-rate-multiplier",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--counterfactual-step-max-gradient-norm",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--counterfactual-step-mean-kl-limit",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--counterfactual-step-max-state-kl-limit",
        type=float,
        default=0.010,
    )
    parser.add_argument("--counterfactual-workers", type=int, default=1)

    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_entrypoint_args(args)
    report = run_development_causal_canary(args)
    _validate_closed_report(report)
    report["exact_digest"] = stable_payload_digest(report)
    write_json(args.report, report)
    training = _required_mapping(report.get("training"), field="training")
    print(
        "public_recurrent_ippo_causal_canary_complete "
        f"worlds={training.get('total_worlds')} "
        f"transitions={training.get('total_transitions')} "
        f"report={args.report}"
    )
    return 0


def run_development_causal_canary(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Execute one explicitly noncandidate before/train/after experiment."""

    _validate_entrypoint_args(args)
    total_started = time.perf_counter()
    source_manifest = _source_file_hash_manifest()
    source_state = _git_source_state()
    device = _resolve_device(args.device)
    scenarios = _parse_csv_choice(
        args.scenarios,
        allowed=RECURRENT_TRAINING_SCENARIOS,
        field="scenarios",
    )
    fixtures = _parse_csv_choice(
        args.evaluation_fixtures,
        allowed=CONTROLLED_FIXTURE_NAMES,
        field="evaluation_fixtures",
    )
    model_config, ppo_config, tbptt_contract = _resolved_configs(args)
    counterfactual_config, counterfactual_preregistration = (
        _resolved_counterfactual_config(args)
    )
    runtime_reproducibility = _runtime_reproducibility_provenance(
        requested_device=args.device,
        resolved_device=device,
    )
    schedule = build_recurrent_training_schedule(
        update_count=args.updates,
        worlds_per_update=args.worlds_per_update,
        rollout_ticks=args.rollout_ticks,
        scenarios=scenarios,
    )
    schedule_payload = [
        [asdict(task) for task in update_tasks] for update_tasks in schedule
    ]
    seed_plan, seed_roles = _selection_seed_plan(
        count=args.selection_seed_count,
        offset=args.selection_seed_offset,
        learner_seed=args.learner_seed,
    )
    evaluation_sampling_stream = _evaluation_sampling_stream(
        seed_plan=seed_plan,
        fixtures=fixtures,
        candidate_sampling_seed_count=args.candidate_sampling_seed_count,
    )
    run_config: dict[str, object] = {
        "updates": args.updates,
        "worlds_per_update": args.worlds_per_update,
        "rollout_ticks": args.rollout_ticks,
        "training_scenarios": list(scenarios),
        "evaluation_fixtures": list(fixtures),
        "evaluation_horizon_ticks": 120,
        "candidate_sampling_seed_count": args.candidate_sampling_seed_count,
        "evaluation_sampling_stream": evaluation_sampling_stream,
        "requested_device": args.device,
        "resolved_device": str(device),
        "rollout_workers": args.rollout_workers,
        "evaluation_workers": args.evaluation_workers,
        "model": asdict(model_config),
        "ppo": asdict(ppo_config),
        "tbptt_contract": tbptt_contract,
        "feed_forward_history_ablation": (ppo_config.feed_forward_history_ablation),
        "counterfactual_auxiliary": counterfactual_preregistration,
        "runtime_reproducibility": runtime_reproducibility,
        "training_schedule": schedule_payload,
        "training_schedule_sha256": stable_payload_digest(schedule_payload),
    }
    preregistration: dict[str, object] = {
        "schema_version": CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION,
        "created_before_model_initialization": True,
        "action_selection_modes_in_order": list(ACTION_SELECTION_MODES),
        "before_after_sampling_stream_identical": True,
        "candidate_controls": ["mind_v3_linear", "masked_random"],
        "run_config": run_config,
        "seed_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
        "seed_roles": seed_roles,
        "source_manifest_aggregate_sha256": source_manifest["aggregate_sha256"],
    }
    preregistration_digest = stable_payload_digest(preregistration)
    evaluation_label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + preregistration_digest

    runner = RecurrentExperimentRunner(
        learner_seed=args.learner_seed,
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        rollout_workers=args.rollout_workers,
        counterfactual_config=counterfactual_config,
    )
    before_model = frozen_cpu_model_copy(runner.model)
    before_fingerprint = _model_state_fingerprint(before_model)
    live_initial_fingerprint = _model_state_fingerprint(runner.model)
    if before_fingerprint["state_sha256"] != live_initial_fingerprint["state_sha256"]:
        raise CausalCanaryError(
            "the frozen before-model state differs from the live initial model"
        )

    timings: dict[str, float] = {}
    before_evaluations: dict[str, object] = {}
    for selection_mode in ACTION_SELECTION_MODES:
        phase = f"before_evaluation.{selection_mode}"
        before_evaluations[selection_mode], timings[phase] = _timed_evaluation(
            before_model,
            label=evaluation_label,
            seed_plan=seed_plan,
            fixtures=fixtures,
            feed_forward_history_ablation=(ppo_config.feed_forward_history_ablation),
            selection_mode=selection_mode,
            candidate_sampling_seed_count=(args.candidate_sampling_seed_count),
            candidate_sampling_stream_id=evaluation_sampling_stream["id"],
            evaluation_workers=args.evaluation_workers,
        )

    pretraining_source_manifest = _source_file_hash_manifest()
    if pretraining_source_manifest != source_manifest:
        raise CausalCanaryError(
            "runtime source changed during the causal canary; report withheld"
        )

    training_started = time.perf_counter()
    training_result = runner.run(schedule)
    timings["training"] = _elapsed_seconds(training_started)
    after_fingerprint = _model_state_fingerprint(runner.model)
    if after_fingerprint["state_sha256"] == before_fingerprint["state_sha256"]:
        raise CausalCanaryError("training did not change the model state")

    after_evaluations: dict[str, object] = {}
    for selection_mode in ACTION_SELECTION_MODES:
        phase = f"after_evaluation.{selection_mode}"
        after_evaluations[selection_mode], timings[phase] = _timed_evaluation(
            runner.model,
            label=evaluation_label,
            seed_plan=seed_plan,
            fixtures=fixtures,
            feed_forward_history_ablation=(ppo_config.feed_forward_history_ablation),
            selection_mode=selection_mode,
            candidate_sampling_seed_count=(args.candidate_sampling_seed_count),
            candidate_sampling_stream_id=evaluation_sampling_stream["id"],
            evaluation_workers=args.evaluation_workers,
        )

    final_source_manifest = _source_file_hash_manifest()
    if final_source_manifest != source_manifest:
        raise CausalCanaryError(
            "runtime source changed during the causal canary; report withheld"
        )

    paired_deltas = {
        selection_mode: _paired_causal_deltas(
            _required_mapping(
                before_evaluations[selection_mode],
                field=f"before_evaluations.{selection_mode}",
            ),
            _required_mapping(
                after_evaluations[selection_mode],
                field=f"after_evaluations.{selection_mode}",
            ),
        )
        for selection_mode in ACTION_SELECTION_MODES
    }
    timings["total"] = _elapsed_seconds(total_started)
    lifecycle = {flag: False for flag in _LIFECYCLE_FLAGS}
    training_payload = recurrent_training_run_payload(training_result)
    report: dict[str, object] = {
        "schema_version": CAUSAL_CANARY_SCHEMA_VERSION,
        "policy": CAUSAL_CANARY_POLICY_VERSION,
        "development_run": True,
        "development_experiment": True,
        "noncandidate_development_canary": True,
        "source_dirty": source_state["source_tree_dirty"],
        "source_unpinned": True,
        "source_pinned": False,
        "artifact_output_refused_by_contract": True,
        "artifact_output_requested": False,
        "artifact_path": None,
        "source": {
            **source_state,
            "source_pinned": False,
            "source_commit_explicitly_pinned": False,
            "unpinned": True,
            "noncandidate": True,
            "source_file_hash_manifest": source_manifest,
            "source_stable_during_run": True,
        },
        "runtime_reproducibility": {
            **runtime_reproducibility,
            "torch_deterministic_algorithms_enabled_after_runner": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "seed_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
        "seed_roles": seed_roles,
        "configuration": run_config,
        "evaluation_preregistration": {
            **preregistration,
            "exact_digest": preregistration_digest,
            "synthetic_noncandidate_digest_label": evaluation_label,
        },
        "model_states": {
            "before": before_fingerprint,
            "after": after_fingerprint,
            "changed": True,
            "same_before_snapshot_used_for_both_selection_modes": True,
            "same_after_model_used_for_both_selection_modes": True,
        },
        "elapsed_seconds": timings,
        "training": training_payload,
        "evaluations": {
            "before": before_evaluations,
            "after": after_evaluations,
        },
        "paired_outcome_deltas": paired_deltas,
        "lifecycle": lifecycle,
        "campaign_training_slice_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
        "non_promoted": True,
    }
    _validate_closed_report(report)
    return report


def _resolved_configs(
    args: argparse.Namespace,
) -> tuple[RecurrentActorCriticConfig, RecurrentPPOConfig, dict[str, object]]:
    model_config = RecurrentActorCriticConfig(
        encoder_size=args.encoder_size,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
    )
    resolved_tbptt_steps = (
        args.rollout_ticks if args.full_sequence_tbptt else args.tbptt_steps
    )
    ppo_config = RecurrentPPOConfig(
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip_range=args.policy_clip_range,
        value_clip_range=args.value_clip_range,
        value_loss_coefficient=args.value_loss_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        update_epochs=args.update_epochs,
        sequence_minibatch_size=args.sequence_minibatch_size,
        tbptt_steps=resolved_tbptt_steps,
        burn_in_steps=args.burn_in_steps,
        max_gradient_norm=args.max_gradient_norm,
        normalize_advantages=args.normalize_advantages,
        advantage_epsilon=args.advantage_epsilon,
        target_kl=args.target_kl,
        learner_seed=args.learner_seed,
        feed_forward_history_ablation=args.feed_forward_history_ablation,
        world_balanced_loss=args.world_balanced_loss,
    )
    return (
        model_config,
        ppo_config,
        {
            "full_sequence_tbptt": args.full_sequence_tbptt,
            "requested_tbptt_steps": args.tbptt_steps,
            "resolved_tbptt_steps": resolved_tbptt_steps,
            "maximum_agent_sequence_length": args.rollout_ticks,
            "full_sequence_invariant_satisfied": (
                not args.full_sequence_tbptt
                or resolved_tbptt_steps >= args.rollout_ticks
            ),
        },
    )


def _resolved_counterfactual_config(
    args: argparse.Namespace,
) -> tuple[RecurrentCounterfactualExperimentConfig | None, dict[str, object]]:
    """Resolve the dormant or active auxiliary contract before model creation."""

    exact_enabled = args.counterfactual_exact_auxiliary is True
    shuffled_enabled = args.counterfactual_label_shuffled_control is True
    if exact_enabled and shuffled_enabled:
        raise CausalCanaryError(
            "exact and label-shuffled counterfactual modes are mutually exclusive"
        )
    enabled = exact_enabled or shuffled_enabled
    if shuffled_enabled and args.counterfactual_permutation_seed is None:
        raise CausalCanaryError(
            "--counterfactual-label-shuffled-control requires an explicit "
            "--counterfactual-permutation-seed"
        )
    if not shuffled_enabled and args.counterfactual_permutation_seed is not None:
        raise CausalCanaryError(
            "--counterfactual-permutation-seed is only valid with "
            "--counterfactual-label-shuffled-control"
        )
    if enabled and args.feed_forward_history_ablation:
        raise CausalCanaryError(
            "counterfactual recurrent-history auxiliary cannot be combined "
            "with --feed-forward-history-ablation"
        )
    if enabled and args.counterfactual_bundles_per_update > args.worlds_per_update:
        raise CausalCanaryError(
            "--counterfactual-bundles-per-update cannot exceed --worlds-per-update"
        )

    horizons = _parse_integer_csv(
        args.counterfactual_horizons,
        field="counterfactual horizons",
        minimum=1,
    )
    horizon_weights = _parse_float_csv(
        args.counterfactual_horizon_weights,
        field="counterfactual horizon weights",
    )
    if len(horizons) != len(horizon_weights):
        raise CausalCanaryError(
            "counterfactual horizons and horizon weights must have equal length"
        )
    branch_ticks = _parse_integer_csv(
        args.counterfactual_branch_ticks,
        field="counterfactual branch ticks",
        minimum=0,
    )
    target_permutation_mode = (
        RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
        if shuffled_enabled
        else RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
    )
    declared_config = RecurrentCounterfactualExperimentConfig(
        collection=RecurrentCounterfactualCollectionConfig(
            horizons=horizons,
            gamma=args.gamma,
        ),
        auxiliary=RecurrentCounterfactualAuxiliaryConfig(
            scalarization=CounterfactualHorizonScalarization(
                horizon_weights=tuple(zip(horizons, horizon_weights, strict=True)),
                focal_discounted_return_weight=1.0,
                focal_terminal_alive_weight=0.0,
                population_alive_weight=0.0,
                births_during_horizon_weight=0.0,
                deaths_during_horizon_weight=0.0,
            ),
            temperature=args.counterfactual_temperature,
            advantage_clip=args.counterfactual_advantage_clip,
            policy_improvement_coefficient=(args.counterfactual_policy_coefficient),
            behavior_kl_coefficient=(args.counterfactual_behavior_kl_coefficient),
            value_loss_coefficient=0.0,
            target_permutation_mode=target_permutation_mode,
            target_permutation_seed=(
                args.counterfactual_permutation_seed if shuffled_enabled else None
            ),
        ),
        step=RecurrentCounterfactualAuxiliaryStepConfig(
            learning_rate_multiplier=(
                args.counterfactual_step_learning_rate_multiplier
            ),
            max_gradient_norm=args.counterfactual_step_max_gradient_norm,
            mean_behavior_kl_limit=args.counterfactual_step_mean_kl_limit,
            max_state_behavior_kl_limit=(args.counterfactual_step_max_state_kl_limit),
        ),
        bundles_per_update=args.counterfactual_bundles_per_update,
        branch_tick_candidates=branch_ticks,
        workers=args.counterfactual_workers,
    )
    mode = (
        "exact_counterfactual_labels"
        if exact_enabled
        else (
            "deterministic_valid_action_label_shuffled_control"
            if shuffled_enabled
            else "disabled"
        )
    )
    resolved_contract = {
        "collection": recurrent_counterfactual_collection_config_payload(
            declared_config.collection
        ),
        "auxiliary": declared_config.auxiliary.as_contract(),
        "step": declared_config.step.as_contract(),
        "bundles_per_update": declared_config.bundles_per_update,
        "branch_tick_candidates": list(declared_config.branch_tick_candidates),
        "workers": declared_config.workers,
    }
    preregistration = {
        "enabled": enabled,
        "mode": mode,
        "default_off": True,
        "configuration_resolved_before_model_initialization": True,
        "resolved_configuration": resolved_contract,
        "one_auxiliary_step_per_ppo_update": enabled,
        "exact_branch_labels_consumed": exact_enabled,
        "scientific_negative_control": {
            "enabled": shuffled_enabled,
            "mode": target_permutation_mode,
            "permutation_seed_explicit": (
                args.counterfactual_permutation_seed is not None
            ),
            "permutation_seed": args.counterfactual_permutation_seed,
            "permutation_domain": "currently_valid_actions_only",
        },
        "feed_forward_history_ablation_compatible": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }
    return (declared_config if enabled else None), preregistration


def _selection_seed_plan(
    *,
    count: int,
    offset: int,
    learner_seed: int,
) -> tuple[RecurrentEvaluationSeedPlan, dict[str, object]]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise CausalCanaryError("selection_seed_count must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise CausalCanaryError("selection_seed_offset must be a nonnegative integer")
    selection_registry = RECURRENT_SEED_REGISTRY["selection"]
    stop = offset + count
    if stop > len(selection_registry):
        raise CausalCanaryError(
            "selection seed slice exceeds the canonical selection registry"
        )
    selected = selection_registry[offset:stop]
    excluded_training = (
        *RECURRENT_SEED_REGISTRY["train"],
        *RECURRENT_SEED_REGISTRY["curriculum"],
    )
    plan = RecurrentEvaluationSeedPlan(
        broad_seeds=selected,
        fixture_seeds=selected,
        excluded_training_seeds=excluded_training,
        environment_seed_role=RECURRENT_EVALUATION_SELECTION_SEED_ROLE,
    )
    learner_role = "learner_development"
    if learner_seed not in RECURRENT_SEED_REGISTRY[learner_role]:
        raise CausalCanaryError(
            "learner_seed must come from the learner_development role"
        )
    return plan, {
        "learner": {
            "role": learner_role,
            "seed": learner_seed,
            "policy": asdict(SEED_ROLE_POLICIES[learner_role]),
        },
        "optimization_environment": {
            "roles": ["train", "curriculum"],
            "policies": {
                role: asdict(SEED_ROLE_POLICIES[role])
                for role in ("train", "curriculum")
            },
            "excluded_from_evaluation_seed_count": len(excluded_training),
        },
        "development_selection": {
            "role": "selection",
            "offset": offset,
            "count": count,
            "seeds": list(selected),
            "seed_plan_digest": plan.digest,
            "used_for_broad_and_fixture_evaluation": True,
            "policy": asdict(SEED_ROLE_POLICIES["selection"]),
        },
        "validation": {
            "role": "validation",
            "seeds_materialized_by_canary": False,
            "accessed": False,
        },
        "lockbox": {
            "role": "lockbox",
            "seeds_materialized_by_canary": False,
            "accessed": False,
        },
    }


def _timed_evaluation(
    model: PublicRecurrentActorCritic,
    *,
    label: str,
    seed_plan: RecurrentEvaluationSeedPlan,
    fixtures: Sequence[str],
    feed_forward_history_ablation: bool,
    selection_mode: str,
    candidate_sampling_seed_count: int,
    candidate_sampling_stream_id: str,
    evaluation_workers: int,
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    report = evaluate_recurrent_model(
        model,
        synthetic_noncandidate_digest_label=label,
        seed_plan=seed_plan,
        fixture_names=fixtures,
        feed_forward_history_ablation=feed_forward_history_ablation,
        candidate_action_selection=selection_mode,
        candidate_sampling_seed_count=candidate_sampling_seed_count,
        candidate_sampling_stream_id=candidate_sampling_stream_id,
        evaluation_workers=evaluation_workers,
    )
    return report, _elapsed_seconds(started)


def _evaluation_sampling_stream(
    *,
    seed_plan: RecurrentEvaluationSeedPlan,
    fixtures: Sequence[str],
    candidate_sampling_seed_count: int,
) -> dict[str, object]:
    """Return an arm-independent stream identity for matched causal controls.

    The identifier deliberately excludes model architecture, learner seed,
    parameters, and the feed-forward ablation flag. GRU and feed-forward arms
    using the same held-out environment plan therefore receive exactly the
    same sampled-action streams, while model provenance remains separately
    pinned by the preregistration and source/model fingerprints.
    """

    payload = {
        "schema_version": ("mind_v3_recurrent_matched_evaluation_sampling_stream_v1"),
        "seed_plan_digest": seed_plan.digest,
        "fixtures": list(fixtures),
        "candidate_sampling_seed_count": candidate_sampling_seed_count,
        "excluded_fields": [
            "model_architecture",
            "model_parameters",
            "learner_seed",
            "feed_forward_history_ablation",
            "candidate_policy_digest",
        ],
    }
    return {
        "id": "matched-recurrent-evaluation:" + stable_payload_digest(payload),
        "contract": payload,
        "arm_independent": True,
    }


def _paired_causal_deltas(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    before_contract = _required_mapping(
        before.get("evaluation_contract"),
        field="before.evaluation_contract",
    )
    after_contract = _required_mapping(
        after.get("evaluation_contract"),
        field="after.evaluation_contract",
    )
    sampling_fields = (
        "candidate_action_selection",
        "candidate_sampling_stream_id",
        "candidate_sampling_seeds",
        "candidate_sampling_seed_count",
    )
    if any(
        before_contract.get(field) != after_contract.get(field)
        for field in sampling_fields
    ):
        raise CausalCanaryError(
            "before/after candidate evaluation sampling contract drifted"
        )
    before_contexts = _evaluation_contexts(before)
    after_contexts = _evaluation_contexts(after)
    if tuple(before_contexts) != tuple(after_contexts):
        raise CausalCanaryError("before/after evaluation contexts differ")

    runs: list[dict[str, object]] = []
    candidate_vs_controls: dict[str, object] = {}
    for context, before_context in before_contexts.items():
        after_context = after_contexts[context]
        before_policies = _required_mapping(
            before_context.get("policies"),
            field=f"before.{context}.policies",
        )
        after_policies = _required_mapping(
            after_context.get("policies"),
            field=f"after.{context}.policies",
        )
        for control in ("mind_v3_linear", "masked_random"):
            if before_policies.get(control) != after_policies.get(control):
                raise CausalCanaryError(
                    f"before/after {control} control drifted in {context}"
                )
        before_candidate = _policy_runs(
            before_policies,
            policy="public_recurrent",
            field=f"before.{context}",
        )
        after_candidate = _policy_runs(
            after_policies,
            policy="public_recurrent",
            field=f"after.{context}",
        )
        if len(before_candidate) != len(after_candidate) or not before_candidate:
            raise CausalCanaryError(
                f"before/after candidate run counts differ in {context}"
            )
        for before_run, after_run in zip(
            before_candidate,
            after_candidate,
            strict=True,
        ):
            if (
                before_run.get("context") != after_run.get("context")
                or before_run.get("seed") != after_run.get("seed")
                or before_run.get("policy_sampling_seed")
                != after_run.get("policy_sampling_seed")
            ):
                raise CausalCanaryError(
                    f"before/after candidate run identity differs in {context}"
                )
            delta: dict[str, object] = {
                "context": before_run["context"],
                "seed": before_run["seed"],
                "policy_sampling_seed": before_run.get("policy_sampling_seed"),
            }
            for metric in _CAUSAL_METRICS:
                before_value = _finite_number(
                    before_run.get(metric),
                    field=f"before.{context}.{metric}",
                )
                after_value = _finite_number(
                    after_run.get(metric),
                    field=f"after.{context}.{metric}",
                )
                difference = after_value - before_value
                delta[f"{metric}_delta"] = (
                    int(difference)
                    if metric
                    in {
                        "terminal_alive",
                        "births",
                        "deaths",
                        "unsupported_requested_action_count",
                        "heuristic_action_source_count",
                    }
                    else round(difference, 12)
                )
            runs.append(delta)
        candidate_vs_controls[context] = {
            "before": before_context.get("paired_deltas"),
            "after": after_context.get("paired_deltas"),
        }

    return {
        "run_count": len(runs),
        "runs": runs,
        "mean_deltas": {
            f"{metric}_delta": round(
                math.fsum(float(run[f"{metric}_delta"]) for run in runs) / len(runs),
                12,
            )
            for metric in _CAUSAL_METRICS
        },
        "candidate_minus_controls": candidate_vs_controls,
        "controls_exactly_stable_before_after": True,
        "same_selection_seed_stream_before_after": True,
        "candidate_sampling_stream_id": before_contract.get(
            "candidate_sampling_stream_id"
        ),
        "candidate_sampling_seeds": before_contract.get("candidate_sampling_seeds"),
    }


def _evaluation_contexts(
    report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    contexts: dict[str, Mapping[str, object]] = {
        "broad": _required_mapping(report.get("broad"), field="broad")
    }
    fixtures = report.get("fixtures")
    if not isinstance(fixtures, list):
        raise CausalCanaryError("evaluation fixtures must be a list")
    for fixture in fixtures:
        fixture_report = _required_mapping(fixture, field="fixture")
        name = fixture_report.get("fixture")
        if not isinstance(name, str) or not name:
            raise CausalCanaryError("evaluation fixture name is invalid")
        key = f"fixture:{name}"
        if key in contexts:
            raise CausalCanaryError("evaluation fixture names are duplicated")
        contexts[key] = fixture_report
    return contexts


def _policy_runs(
    policies: Mapping[str, object],
    *,
    policy: str,
    field: str,
) -> tuple[Mapping[str, object], ...]:
    payload = _required_mapping(policies.get(policy), field=f"{field}.{policy}")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise CausalCanaryError(f"{field}.{policy}.runs must be a list")
    return tuple(_required_mapping(run, field=f"{field}.{policy}.runs") for run in runs)


def _model_state_fingerprint(
    model: PublicRecurrentActorCritic,
) -> dict[str, object]:
    if not isinstance(model, PublicRecurrentActorCritic):
        raise TypeError("model must be a PublicRecurrentActorCritic")
    digest = hashlib.sha256()
    state = model.state_dict()
    dtype_counts: dict[str, int] = {}
    tensor_value_count = 0
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, byteorder="big"))
        digest.update(metadata)
        digest.update(tensor.numpy().tobytes(order="C"))
        dtype = str(tensor.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + tensor.numel()
        tensor_value_count += tensor.numel()
    return {
        "state_sha256": digest.hexdigest(),
        "state_tensor_count": len(state),
        "state_value_count": tensor_value_count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "dtype_value_counts": dict(sorted(dtype_counts.items())),
        "hash_contract": "sorted_state_dict_name_dtype_shape_and_raw_cpu_bytes_v1",
    }


def _source_file_hash_manifest() -> dict[str, object]:
    source_root = _REPOSITORY_ROOT / "python" / "evolution_sim"
    paths = sorted(source_root.rglob("*.py"))
    package_path = _REPOSITORY_ROOT / "package.json"
    if package_path.is_file():
        paths.append(package_path)
    if not paths:
        raise CausalCanaryError("source file manifest found no source files")
    files = {
        path.relative_to(_REPOSITORY_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
        if path.is_file()
    }
    return {
        "hash_algorithm": "sha256",
        "path_contract": "repository_relative_sorted_runtime_python_plus_package",
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": stable_payload_digest(files),
    }


def _git_source_state() -> dict[str, object]:
    try:
        head = subprocess.run(
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
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_head_observed": None,
            "git_state_available": False,
            "source_tree_dirty": True,
            "git_status_porcelain_sha256": None,
            "git_status_entry_count": None,
        }
    status_lines = tuple(line for line in status.splitlines() if line)
    return {
        "git_head_observed": head or None,
        "git_state_available": True,
        "source_tree_dirty": bool(status_lines),
        "git_status_porcelain_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "git_status_entry_count": len(status_lines),
    }


def _resolve_device(label: str) -> torch.device:
    if not isinstance(label, str) or not label.strip():
        raise CausalCanaryError("device must be a nonempty string")
    normalized = label.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CausalCanaryError("requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise CausalCanaryError("requested MPS device is unavailable")
    return device


def _parse_csv_choice(
    value: object,
    *,
    allowed: Sequence[str],
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise CausalCanaryError(f"{field} must be a comma-separated string")
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selected:
        raise CausalCanaryError(f"{field} cannot be empty")
    if len(set(selected)) != len(selected):
        raise CausalCanaryError(f"{field} cannot contain duplicates")
    unsupported = [item for item in selected if item not in allowed]
    if unsupported:
        raise CausalCanaryError(f"{field} contains unsupported values: {unsupported!r}")
    return selected


def _parse_integer_csv(
    value: object,
    *,
    field: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise CausalCanaryError(f"{field} must be a comma-separated string")
    raw_values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not raw_values:
        raise CausalCanaryError(f"{field} cannot be empty")
    try:
        parsed = tuple(int(part) for part in raw_values)
    except ValueError as error:
        raise CausalCanaryError(f"{field} must contain only integers") from error
    if any(item < minimum for item in parsed):
        raise CausalCanaryError(
            f"{field} values must be greater than or equal to {minimum}"
        )
    return parsed


def _parse_float_csv(
    value: object,
    *,
    field: str,
) -> tuple[float, ...]:
    if not isinstance(value, str):
        raise CausalCanaryError(f"{field} must be a comma-separated string")
    raw_values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not raw_values:
        raise CausalCanaryError(f"{field} cannot be empty")
    try:
        parsed = tuple(float(part) for part in raw_values)
    except ValueError as error:
        raise CausalCanaryError(f"{field} must contain only numbers") from error
    if any(not math.isfinite(item) for item in parsed):
        raise CausalCanaryError(f"{field} values must be finite")
    return parsed


def _runtime_reproducibility_provenance(
    *,
    requested_device: str,
    resolved_device: torch.device,
) -> dict[str, object]:
    """Capture execution facts before model initialization and training."""

    try:
        numpy_version: str | None = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    device_name: str
    if resolved_device.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if resolved_device.index is None
            else resolved_device.index
        )
        device_name = torch.cuda.get_device_name(device_index)
    elif resolved_device.type == "mps":
        device_name = "Apple Metal Performance Shaders"
    else:
        device_name = platform.processor() or "cpu"
    return {
        "schema_version": CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION,
        "captured_before_model_initialization": True,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "numpy_version": numpy_version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform_string": platform.platform(),
        },
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "resolved_device_type": resolved_device.type,
        "resolved_device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_deterministic_algorithms_enabled_before_runner": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def _validate_entrypoint_args(args: argparse.Namespace) -> None:
    if not args.development_run:
        raise SystemExit(
            "refusing to run without --development-run; this is an unpinned, "
            "noncandidate development canary"
        )
    if args.artifact is not None:
        raise SystemExit(
            "artifact output is forbidden for the development causal canary"
        )
    if args.report.suffix.lower() != ".json":
        raise SystemExit("--report must name one canonical .json report")
    if (
        isinstance(args.candidate_sampling_seed_count, bool)
        or not isinstance(args.candidate_sampling_seed_count, int)
        or not 1 <= args.candidate_sampling_seed_count <= 32
    ):
        raise SystemExit("--candidate-sampling-seed-count must be in [1, 32]")
    if (
        isinstance(args.rollout_workers, bool)
        or not isinstance(args.rollout_workers, int)
        or not 1 <= args.rollout_workers <= MAX_RECURRENT_ROLLOUT_WORKERS
    ):
        raise SystemExit(
            f"--rollout-workers must be in [1, {MAX_RECURRENT_ROLLOUT_WORKERS}]"
        )
    if (
        isinstance(args.evaluation_workers, bool)
        or not isinstance(args.evaluation_workers, int)
        or not 1 <= args.evaluation_workers <= _MAX_EVALUATION_WORKERS
    ):
        raise SystemExit(
            f"--evaluation-workers must be in [1, {_MAX_EVALUATION_WORKERS}]"
        )
    try:
        _resolved_counterfactual_config(args)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid counterfactual configuration: {error}") from error


def _validate_closed_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != CAUSAL_CANARY_SCHEMA_VERSION:
        raise CausalCanaryError("causal canary schema version drifted")
    if report.get("policy") != CAUSAL_CANARY_POLICY_VERSION:
        raise CausalCanaryError("causal canary policy version drifted")
    if report.get("development_run") is not True:
        raise CausalCanaryError("development_run must be true")
    if report.get("source_unpinned") is not True:
        raise CausalCanaryError("source_unpinned must be true")
    if report.get("source_pinned") is not False:
        raise CausalCanaryError("source_pinned must be false")
    if report.get("noncandidate_development_canary") is not True:
        raise CausalCanaryError("noncandidate_development_canary must be true")
    if report.get("artifact_output_refused_by_contract") is not True:
        raise CausalCanaryError("artifact output must be refused by contract")
    if report.get("artifact_path") is not None:
        raise CausalCanaryError("development causal canary cannot name an artifact")
    source = _required_mapping(report.get("source"), field="source")
    if (
        source.get("source_pinned") is not False
        or source.get("unpinned") is not True
        or source.get("noncandidate") is not True
        or source.get("source_stable_during_run") is not True
    ):
        raise CausalCanaryError("source provenance is not closed and unpinned")
    source_manifest = _required_mapping(
        source.get("source_file_hash_manifest"),
        field="source.source_file_hash_manifest",
    )
    if not isinstance(source_manifest.get("aggregate_sha256"), str):
        raise CausalCanaryError("source manifest digest is missing")
    runtime = _required_mapping(
        report.get("runtime_reproducibility"),
        field="runtime_reproducibility",
    )
    if runtime.get("schema_version") != CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION:
        raise CausalCanaryError("runtime reproducibility schema drifted")
    if runtime.get("captured_before_model_initialization") is not True:
        raise CausalCanaryError(
            "runtime reproducibility must be captured before model initialization"
        )
    if (
        type(runtime.get("torch_deterministic_algorithms_enabled_after_runner"))
        is not bool
    ):
        raise CausalCanaryError(
            "post-run deterministic-algorithms provenance must be boolean"
        )
    preregistration = _required_mapping(
        report.get("evaluation_preregistration"),
        field="evaluation_preregistration",
    )
    if (
        preregistration.get("schema_version")
        != CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION
        or preregistration.get("created_before_model_initialization") is not True
    ):
        raise CausalCanaryError("causal canary preregistration schema drifted")
    observed_preregistration_digest = preregistration.get("exact_digest")
    preregistration_payload = dict(preregistration)
    preregistration_payload.pop("exact_digest", None)
    preregistration_payload.pop("synthetic_noncandidate_digest_label", None)
    expected_preregistration_digest = stable_payload_digest(preregistration_payload)
    if observed_preregistration_digest != expected_preregistration_digest:
        raise CausalCanaryError("causal canary preregistration digest drifted")
    if (
        preregistration.get("synthetic_noncandidate_digest_label")
        != UNPINNED_NONCANDIDATE_DIGEST_PREFIX + expected_preregistration_digest
    ):
        raise CausalCanaryError(
            "causal canary synthetic noncandidate digest label drifted"
        )
    if preregistration.get("source_manifest_aggregate_sha256") != source_manifest.get(
        "aggregate_sha256"
    ):
        raise CausalCanaryError(
            "preregistered source manifest differs from report provenance"
        )
    run_config = _required_mapping(
        preregistration.get("run_config"),
        field="evaluation_preregistration.run_config",
    )
    counterfactual = _required_mapping(
        run_config.get("counterfactual_auxiliary"),
        field="evaluation_preregistration.run_config.counterfactual_auxiliary",
    )
    if (
        type(counterfactual.get("enabled")) is not bool
        or counterfactual.get("configuration_resolved_before_model_initialization")
        is not True
        or counterfactual.get("runtime_artifact_created") is not False
        or counterfactual.get("runtime_action_selection_changed") is not False
        or counterfactual.get("promotion_authorized") is not False
    ):
        raise CausalCanaryError(
            "counterfactual preregistration does not preserve the closed contract"
        )
    training = _required_mapping(report.get("training"), field="training")
    training_counterfactual = training.get("counterfactual_experiment")
    if counterfactual["enabled"]:
        if not isinstance(training_counterfactual, Mapping):
            raise CausalCanaryError(
                "enabled counterfactual preregistration is missing training evidence"
            )
    elif training_counterfactual is not None:
        raise CausalCanaryError(
            "disabled counterfactual preregistration produced training evidence"
        )
    lifecycle = _required_mapping(report.get("lifecycle"), field="lifecycle")
    if set(lifecycle) != set(_LIFECYCLE_FLAGS):
        raise CausalCanaryError("lifecycle flags do not match the closed contract")
    open_flags = sorted(flag for flag, value in lifecycle.items() if value is not False)
    if open_flags:
        raise CausalCanaryError(f"lifecycle flags are not closed: {open_flags!r}")
    for flag in (
        "campaign_training_slice_consumed",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
        "validation_seeds_accessed",
        "lockbox_seeds_accessed",
    ):
        if report.get(flag) is not False:
            raise CausalCanaryError(f"top-level {flag} must be false")


def _elapsed_seconds(started: float) -> float:
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise CausalCanaryError("elapsed time became invalid")
    return round(elapsed, 6)


def _required_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CausalCanaryError(f"{field} must be a mapping")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalCanaryError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CausalCanaryError(f"{field} must be finite")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
