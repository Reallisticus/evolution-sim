from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import torch

from evolution_sim.mind.recurrent_scale_campaign import (
    build_recurrent_scale_campaign_preregistration,
    load_strict_json,
    source_file_hash_manifest,
    validate_recurrent_scale_campaign_preregistration,
    write_atomic_json,
)
from evolution_sim.mind.recurrent_scale_execution import (
    analyze_recurrent_scale_campaign,
    load_scale_arm_reports,
    run_recurrent_scale_arm,
    validate_recurrent_scale_campaign_analysis,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preregister, execute, and reconcile the pinned multi-learner "
            "three-arm recurrent-IPPO scale-v2 campaign."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--source-commit", required=True)
    preregister.add_argument("--output", type=Path, required=True)

    run_arm = subparsers.add_parser("run-arm")
    run_arm.add_argument("--preregistration", type=Path, required=True)
    run_arm.add_argument("--expected-preregistration-digest", required=True)
    run_arm.add_argument("--learner-seed", type=int, required=True)
    run_arm.add_argument("--arm", required=True)
    run_arm.add_argument("--output-root", type=Path, required=True)
    run_arm.add_argument("--runtime-provenance", type=Path, required=True)
    run_arm.add_argument("--device", default="auto")
    run_arm.add_argument("--rollout-workers", type=int, default=16)
    run_arm.add_argument("--counterfactual-workers", type=int, default=16)
    run_arm.add_argument("--evaluation-workers", type=int, default=16)
    run_arm.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--preregistration", type=Path, required=True)
    aggregate.add_argument("--reports-root", type=Path, required=True)
    aggregate.add_argument("--runtime-provenance", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preregister":
        head, clean = _git_source_state()
        if head != args.source_commit or not clean:
            raise SystemExit(
                "preregistration requires --source-commit to match a clean Git HEAD"
            )
        manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
        preregistration = build_recurrent_scale_campaign_preregistration(
            source_commit=args.source_commit,
            source_manifest_sha256=str(manifest["aggregate_sha256"]),
        )
        write_atomic_json(args.output, preregistration)
        print(
            "recurrent_scale_v2_preregistered "
            f"digest={preregistration['exact_digest']} output={args.output}"
        )
        return 0
    if args.command == "run-arm":
        preregistration = load_strict_json(args.preregistration)
        validate_recurrent_scale_campaign_preregistration(preregistration)
        report = run_recurrent_scale_arm(
            preregistration,
            expected_preregistration_digest=args.expected_preregistration_digest,
            learner_seed=args.learner_seed,
            arm=args.arm,
            output_root=args.output_root,
            device=_resolve_device(args.device),
            rollout_workers=args.rollout_workers,
            counterfactual_workers=args.counterfactual_workers,
            evaluation_workers=args.evaluation_workers,
            runtime_provenance_path=args.runtime_provenance,
            resume=args.resume,
        )
        print(
            "recurrent_scale_v2_arm_complete "
            f"run_id={report['run_id']} digest={report['exact_digest']}"
        )
        return 0
    if args.command == "aggregate":
        preregistration = load_strict_json(args.preregistration)
        reports = load_scale_arm_reports(
            preregistration,
            output_root=args.reports_root,
            runtime_provenance_path=args.runtime_provenance,
        )
        analysis = analyze_recurrent_scale_campaign(preregistration, reports)
        validate_recurrent_scale_campaign_analysis(
            analysis,
            preregistration=preregistration,
            reports=reports,
        )
        write_atomic_json(args.output, analysis)
        persisted_analysis = load_strict_json(args.output)
        validate_recurrent_scale_campaign_analysis(
            persisted_analysis,
            preregistration=preregistration,
            reports=reports,
        )
        print(
            "recurrent_scale_v2_campaign_analyzed "
            f"accepted={persisted_analysis['accepted']} "
            f"digest={persisted_analysis['exact_digest']}"
        )
        return 0
    raise AssertionError("argparse accepted an unknown command")


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


def _git_source_state() -> tuple[str | None, bool]:
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
        return None, False
    return head, not bool(status.strip())


if __name__ == "__main__":
    raise SystemExit(main())
