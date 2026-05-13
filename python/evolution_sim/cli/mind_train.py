from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution_sim.mind.artifacts import write_model_artifact
from evolution_sim.mind.baseline import (
    CONTEXTUAL_PRIOR_TRAINER,
    TRAINER_CHOICES,
    train_baseline_with_trainer,
)
from evolution_sim.mind.dataset import (
    combined_dataset_provenance,
    load_trajectory_jsonl,
    records_with_trajectory_context,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a deterministic Mind v1 offline BC baseline artifact.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        action="append",
        help="Trajectory JSONL or JSONL.GZ input.",
    )
    parser.add_argument(
        "--calibration-trajectory",
        type=Path,
        action="append",
        default=[],
        help=(
            "Separate trajectory JSONL or JSONL.GZ bank for calibrated "
            "supported actor extraction. These records are not added to the "
            "training bank."
        ),
    )
    parser.add_argument(
        "--calibration-validation-trajectory",
        type=Path,
        action="append",
        default=[],
        help=(
            "Separate trajectory JSONL or JSONL.GZ bank for validating "
            "calibrated supported actor extraction. These records are not "
            "added to training or calibration fitting."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/bc-baseline-artifact.json"),
        help="Model artifact destination.",
    )
    parser.add_argument(
        "--trainer",
        choices=TRAINER_CHOICES,
        default=CONTEXTUAL_PRIOR_TRAINER,
        help="Offline baseline trainer to use.",
    )
    parser.add_argument(
        "--torch-device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help=(
            "Device for optional PyTorch trainers. 'auto' resolves to CUDA "
            "when available, then MPS, then CPU. Runtime artifacts are still "
            "serialized as CPU JSON weights."
        ),
    )
    parser.add_argument(
        "--torch-iql-detach-viability-heads",
        action="store_true",
        help=(
            "Opt in to detached auxiliary viability heads for "
            "torch-discrete-iql experiments. The shared representation remains "
            "the default."
        ),
    )
    parser.add_argument(
        "--torch-iql-behavior-margin-anchor",
        action="store_true",
        help=(
            "Opt in to the viability-safe logged-action margin anchor for "
            "torch-discrete-iql actor-confidence experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-calibrated-actor-extraction",
        action="store_true",
        help=(
            "Opt in to behavior-anchored batch-standardized IQL advantage "
            "weighting for torch-discrete-iql actor extraction experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-return-calibration",
        action="store_true",
        help=(
            "Opt in to a discounted-return Q/V auxiliary loss for "
            "torch-discrete-iql critic-scale experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-suppression-critic-calibration",
        action="store_true",
        help=(
            "Opt in to Q/V margin calibration on runtime-suppressed learned "
            "actions for torch-discrete-iql critic experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-action-distribution-regularization",
        action="store_true",
        help=(
            "Opt in to actor marginal-action distribution regularization for "
            "torch-discrete-iql behavior-preservation experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-contextual-behavior-prior-loss-weight",
        type=float,
        help=(
            "Override the contextual behavior-prior actor loss weight for "
            "torch-discrete-iql experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-action-distribution-loss-weight",
        type=float,
        help=(
            "Override the actor marginal-action distribution loss weight for "
            "torch-discrete-iql experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-action-distribution-temperature",
        type=float,
        help=(
            "Override the actor marginal-action distribution softmax "
            "temperature for torch-discrete-iql experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-rollout-state-action-calibration",
        action="store_true",
        help=(
            "Opt in to post-training actor-bias calibration that caps "
            "deterministic top-1 action share on the calibration trajectory "
            "bank for torch-discrete-iql experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-rollout-state-action-max-share",
        type=float,
        help=(
            "Override the rollout-state top-1 action share cap for "
            "torch-discrete-iql actor-bias calibration."
        ),
    )
    parser.add_argument(
        "--torch-iql-rollout-state-action-bias-step",
        type=float,
        help=(
            "Override the per-iteration actor-bias decrement for "
            "rollout-state top-1 action calibration."
        ),
    )
    parser.add_argument(
        "--torch-iql-rollout-state-action-max-bias-delta",
        type=float,
        help=(
            "Override the maximum absolute actor-bias decrement applied by "
            "rollout-state top-1 action calibration."
        ),
    )
    parser.add_argument(
        "--torch-iql-constraint-aware-actor-extraction",
        action="store_true",
        help=(
            "Opt in to observed viability-risk actor weight filtering for "
            "torch-discrete-iql constraint-aware actor extraction experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-risk-adjusted-actor-extraction",
        action="store_true",
        help=(
            "Opt in to detached Q-minus-action-risk actor distillation for "
            "torch-discrete-iql action-conditioned actor extraction experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-calibrated-supported-actor-extraction",
        action="store_true",
        help=(
            "Opt in to calibration-bank action-risk calibration, observed "
            "support constraints, and positive-advantage actor target "
            "extraction for torch-discrete-iql experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-contextual-behavior-supported-actor-extraction",
        action="store_true",
        help=(
            "Opt in to context/action support and logged-behavior proximity "
            "constraints on calibrated actor target extraction for "
            "torch-discrete-iql experiments."
        ),
    )
    parser.add_argument(
        "--torch-iql-contextual-behavior-prior-regularization",
        action="store_true",
        help=(
            "Opt in to contextual behavior-prior cross-entropy regularization "
            "during torch-discrete-iql actor finetuning."
        ),
    )
    parser.add_argument(
        "--torch-iql-neural-actor-prior-blend-weight",
        type=float,
        help=(
            "Override the contextual-prior blend weight for torch-discrete-iql "
            "experiments. Lower values give the neural actor more runtime "
            "authority; the default remains the artifact contract value."
        ),
    )
    parser.add_argument(
        "--torch-iql-counterfactual-labels",
        type=Path,
        help=(
            "Opt in to Mind v3 carrion counterfactual label supervision for "
            "torch-discrete-iql. The report must align with one or more "
            "training trajectories by path and record index."
        ),
    )
    parser.add_argument(
        "--torch-iql-counterfactual-label-weight-scale",
        type=float,
        help=(
            "Non-negative row-weight multiplier scale for counterfactual "
            "label supervision. Defaults to the torch-discrete-iql contract "
            "value."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = [load_trajectory_jsonl(path) for path in args.trajectory]
    calibration_datasets = [
        load_trajectory_jsonl(path)
        for path in args.calibration_trajectory
    ]
    calibration_validation_datasets = [
        load_trajectory_jsonl(path)
        for path in args.calibration_validation_trajectory
    ]
    records = records_with_trajectory_context(datasets)
    calibration_records = (
        records_with_trajectory_context(calibration_datasets)
        if calibration_datasets
        else ()
    )
    calibration_validation_records = (
        records_with_trajectory_context(calibration_validation_datasets)
        if calibration_validation_datasets
        else ()
    )
    counterfactual_label_report = (
        _load_json_report(args.torch_iql_counterfactual_labels)
        if args.torch_iql_counterfactual_labels is not None
        else None
    )
    baseline = train_baseline_with_trainer(
        records,
        provenance=combined_dataset_provenance(datasets),
        trainer=args.trainer,
        torch_iql_detach_viability_heads=args.torch_iql_detach_viability_heads,
        torch_iql_behavior_margin_anchor=args.torch_iql_behavior_margin_anchor,
        torch_iql_calibrated_actor_extraction=(
            args.torch_iql_calibrated_actor_extraction
        ),
        torch_iql_constraint_aware_actor_extraction=(
            args.torch_iql_constraint_aware_actor_extraction
        ),
        torch_iql_risk_adjusted_actor_extraction=(
            args.torch_iql_risk_adjusted_actor_extraction
        ),
        torch_iql_calibrated_supported_actor_extraction=(
            args.torch_iql_calibrated_supported_actor_extraction
        ),
        torch_iql_actor_calibration_records=calibration_records,
        torch_iql_contextual_behavior_supported_actor_extraction=(
            args.torch_iql_contextual_behavior_supported_actor_extraction
        ),
        torch_iql_contextual_behavior_prior_regularization=(
            args.torch_iql_contextual_behavior_prior_regularization
        ),
        torch_iql_actor_calibration_validation_records=(
            calibration_validation_records
        ),
        torch_iql_return_calibration=args.torch_iql_return_calibration,
        torch_iql_suppression_critic_calibration=(
            args.torch_iql_suppression_critic_calibration
        ),
        torch_iql_action_distribution_regularization=(
            args.torch_iql_action_distribution_regularization
        ),
        torch_iql_contextual_behavior_prior_loss_weight=(
            args.torch_iql_contextual_behavior_prior_loss_weight
        ),
        torch_iql_action_distribution_loss_weight=(
            args.torch_iql_action_distribution_loss_weight
        ),
        torch_iql_action_distribution_temperature=(
            args.torch_iql_action_distribution_temperature
        ),
        torch_iql_rollout_state_action_calibration=(
            args.torch_iql_rollout_state_action_calibration
        ),
        torch_iql_rollout_state_action_max_share=(
            args.torch_iql_rollout_state_action_max_share
        ),
        torch_iql_rollout_state_action_bias_step=(
            args.torch_iql_rollout_state_action_bias_step
        ),
        torch_iql_rollout_state_action_max_bias_delta=(
            args.torch_iql_rollout_state_action_max_bias_delta
        ),
        torch_iql_neural_actor_prior_blend_weight=(
            args.torch_iql_neural_actor_prior_blend_weight
        ),
        torch_iql_counterfactual_label_report=counterfactual_label_report,
        torch_iql_counterfactual_label_weight_scale=(
            args.torch_iql_counterfactual_label_weight_scale
        ),
        torch_device=args.torch_device,
    )
    artifact = baseline.to_artifact()
    write_model_artifact(args.output, artifact)
    print(f"artifact={args.output}")
    print(f"trainer={args.trainer}")
    print(f"model_type={artifact['manifest']['model_type']}")
    print(f"sample_weight_policy={artifact['model']['sample_weight_policy']}")
    blend_weight = artifact["model"].get("reward_advantage_blend_weight")
    if blend_weight is not None:
        print(
            "reward_advantage_blend_policy="
            f"{artifact['model'].get('reward_advantage_blend_policy')}"
        )
        print(f"reward_advantage_blend_weight={blend_weight}")
    value_blend_weight = artifact["model"].get("value_score_blend_weight")
    if value_blend_weight is not None:
        print(
            "value_estimation_policy="
            f"{artifact['model'].get('value_estimation_policy')}"
        )
        print(
            "value_score_blend_policy="
            f"{artifact['model'].get('value_score_blend_policy')}"
        )
        print(f"value_score_blend_weight={value_blend_weight}")
        print(
            "value_supported_deviation_policy="
            f"{artifact['model'].get('value_supported_deviation_policy')}"
        )
        print(
            "value_supported_deviation_min_support="
            f"{artifact['model'].get('value_supported_deviation_min_support')}"
        )
    neural_backend = artifact["model"].get("neural_backend")
    if neural_backend is not None:
        print(f"neural_backend={neural_backend}")
        print(f"neural_architecture={artifact['model'].get('neural_architecture')}")
        print(
            "neural_training_policy="
            f"{artifact['model'].get('neural_training_policy')}"
        )
        print(f"neural_hidden_units={artifact['model'].get('neural_hidden_units')}")
        print(
            "neural_actor_prior_blend_weight="
            f"{artifact['model'].get('neural_actor_prior_blend_weight')}"
        )
        torch_device_metadata = artifact["model"].get("torch_device_metadata")
        if isinstance(torch_device_metadata, dict):
            print(
                "torch_requested_device="
                f"{torch_device_metadata.get('requested_device')}"
            )
            print(
                "torch_resolved_device="
                f"{torch_device_metadata.get('resolved_device')}"
            )
    print(f"trained_record_count={baseline.record_count}")
    print(f"source_records={sum(dataset.record_count for dataset in datasets)}")
    print(f"source_trajectories={len(datasets)}")
    if calibration_datasets:
        print(
            "calibration_records="
            f"{sum(dataset.record_count for dataset in calibration_datasets)}"
        )
        print(f"calibration_trajectories={len(calibration_datasets)}")
    if calibration_validation_datasets:
        print(
            "calibration_validation_records="
            f"{sum(dataset.record_count for dataset in calibration_validation_datasets)}"
        )
        print(
            "calibration_validation_trajectories="
            f"{len(calibration_validation_datasets)}"
        )
    if args.torch_iql_counterfactual_labels is not None:
        print(f"torch_iql_counterfactual_labels={args.torch_iql_counterfactual_labels}")


def _load_json_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must be an object: {path}")
    return payload


if __name__ == "__main__":
    main()
