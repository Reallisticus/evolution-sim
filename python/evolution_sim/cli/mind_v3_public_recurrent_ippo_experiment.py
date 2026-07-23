from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from evolution_sim.mind.candidate_campaign import write_json
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import RecurrentActorCriticConfig
from evolution_sim.mind.recurrent_artifact import save_recurrent_artifact
from evolution_sim.mind.recurrent_experiment import (
    MAX_RECURRENT_ROLLOUT_WORKERS,
    RECURRENT_TRAINING_SCENARIOS,
    RecurrentExperimentRunner,
    build_recurrent_training_schedule,
    recurrent_training_run_payload,
)
from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig
from evolution_sim.mind.recurrent_seed_registry import (
    CANONICAL_SEED_REGISTRY_SHA256,
    RECURRENT_SEED_REGISTRY,
)


DEFAULT_REPORT_PATH = Path(
    "output/mind/mind-v3-public-recurrent-ippo-development-experiment.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train-only/public-input recurrent IPPO development experiment. "
            "This command does not consume or authorize a Mind campaign slice."
        )
    )
    parser.add_argument("--development-run", action="store_true")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--worlds-per-update", type=int, default=8)
    parser.add_argument("--rollout-ticks", type=int, default=32)
    parser.add_argument(
        "--scenarios",
        default=",".join(RECURRENT_TRAINING_SCENARIOS),
    )
    parser.add_argument(
        "--learner-seed",
        type=int,
        default=RECURRENT_SEED_REGISTRY["learner_development"][0],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--encoder-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--sequence-minibatch-size", type=int, default=8)
    parser.add_argument("--tbptt-steps", type=int, default=32)
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--feed-forward-history-ablation", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--source-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.development_run:
        raise SystemExit(
            "refusing to train without --development-run; this command does not "
            "consume or authorize a campaign slice"
        )
    if args.artifact is not None and args.source_commit is None:
        raise SystemExit(
            "--artifact requires an explicit --source-commit for provenance"
        )
    if not 1 <= args.rollout_workers <= MAX_RECURRENT_ROLLOUT_WORKERS:
        raise SystemExit(
            f"--rollout-workers must be in [1, {MAX_RECURRENT_ROLLOUT_WORKERS}]"
        )
    if args.artifact is not None:
        head, clean = _git_source_state()
        if head is None or args.source_commit != head:
            raise SystemExit(
                "--source-commit must exactly match the checked-out Git HEAD"
            )
        if not clean:
            raise SystemExit(
                "refusing to serialize a durable artifact from a dirty source tree"
            )
    scenarios = tuple(
        part.strip() for part in args.scenarios.split(",") if part.strip()
    )
    device = _resolve_device(args.device)
    model_config = RecurrentActorCriticConfig(
        encoder_size=args.encoder_size,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
    )
    ppo_config = RecurrentPPOConfig(
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        sequence_minibatch_size=args.sequence_minibatch_size,
        tbptt_steps=args.tbptt_steps,
        burn_in_steps=args.burn_in_steps,
        entropy_coefficient=args.entropy_coefficient,
        target_kl=args.target_kl,
        learner_seed=args.learner_seed,
        feed_forward_history_ablation=args.feed_forward_history_ablation,
    )
    schedule = build_recurrent_training_schedule(
        update_count=args.updates,
        worlds_per_update=args.worlds_per_update,
        rollout_ticks=args.rollout_ticks,
        scenarios=scenarios,
    )
    runner = RecurrentExperimentRunner(
        learner_seed=args.learner_seed,
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        rollout_workers=args.rollout_workers,
    )
    result = runner.run(schedule)
    source_commit = args.source_commit or _git_head()
    artifact_sha256: str | None = None
    if args.artifact is not None:
        assert args.source_commit is not None
        artifact = save_recurrent_artifact(
            args.artifact,
            runner.model.to(device="cpu", dtype=torch.float32),
            training_config=asdict(ppo_config),
            seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
            source_commit=args.source_commit,
            data_metadata={
                "policy_induced": True,
                "training_seed_provenance_schema_version": (
                    result.environment_seed_provenance["schema_version"]
                ),
                "environment_seed_roles": result.environment_seed_provenance[
                    "environment_seed_roles"
                ],
                "environment_seeds_by_role": result.environment_seed_provenance[
                    "environment_seeds_by_role"
                ],
                "worlds": result.total_worlds,
                "agent_transitions": result.total_transitions,
                "scenarios": list(result.training_scenarios),
            },
            run_metadata={
                "purpose": "development_experiment",
                "campaign_slice_consumed": False,
                "report_path": str(args.report),
            },
            learner_seed=args.learner_seed,
            learner_device=str(device),
        )
        artifact_sha256 = str(artifact["artifact_sha256"])

    report: dict[str, object] = {
        "schema_version": "mind_v3_public_recurrent_ippo_development_experiment_v1",
        "policy": "public_recurrent_ippo_train_only_development_v1",
        "source_commit": source_commit,
        "source_commit_explicitly_pinned": args.source_commit is not None,
        "seed_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
        "training": recurrent_training_run_payload(result),
        "artifact_path": str(args.artifact) if args.artifact is not None else None,
        "artifact_sha256": artifact_sha256,
        "development_experiment": True,
        "campaign_training_slice_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "non_promoted": True,
    }
    report["exact_digest"] = stable_payload_digest(report)
    write_json(args.report, report)
    print(
        "recurrent_ippo_development_complete "
        f"worlds={result.total_worlds} transitions={result.total_transitions} "
        f"device={device} report={args.report}"
    )
    return 0


def _resolve_device(label: str) -> torch.device:
    normalized = label.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(label)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("requested MPS device is unavailable")
    return device


def _git_head() -> str | None:
    head, _clean = _git_source_state()
    return head


def _git_source_state() -> tuple[str | None, bool]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return head, not bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, False


if __name__ == "__main__":
    raise SystemExit(main())
