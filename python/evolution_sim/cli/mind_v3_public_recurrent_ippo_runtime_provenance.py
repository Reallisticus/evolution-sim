from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import torch

from evolution_sim.mind.recurrent_runtime_provenance import (
    RecurrentRuntimeProvenanceError,
    assert_recurrent_scale_runtime_provenance_match,
    build_recurrent_scale_runtime_provenance,
    validate_recurrent_scale_runtime_provenance,
)
from evolution_sim.mind.recurrent_experiment import (
    configure_recurrent_training_determinism,
)
from evolution_sim.mind.recurrent_scale_campaign import (
    load_strict_json,
    source_file_hash_manifest,
    validate_recurrent_scale_campaign_preregistration,
    write_atomic_json,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture or verify the non-secret content-addressed runtime "
            "contract for the recurrent-IPPO scale campaign."
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-workers", type=int, default=16)
    parser.add_argument("--counterfactual-workers", type=int, default=16)
    parser.add_argument("--evaluation-workers", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = load_strict_json(args.preregistration)
    validate_recurrent_scale_campaign_preregistration(preregistration)
    source = preregistration["source"]
    if not isinstance(source, dict):
        raise SystemExit("preregistration source must be an object")

    head, clean = _git_source_state()
    manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
    if head != source.get("commit") or not clean:
        raise SystemExit("runtime provenance requires the preregistered clean Git HEAD")
    if manifest["aggregate_sha256"] != source.get("manifest_sha256"):
        raise SystemExit(
            "runtime provenance source manifest differs from preregistration"
        )

    seed_contract = preregistration.get("seed_contract")
    if not isinstance(seed_contract, dict):
        raise SystemExit("preregistration seed contract must be an object")
    learner_seeds = seed_contract.get("learner_seeds")
    if not isinstance(learner_seeds, list) or not learner_seeds:
        raise SystemExit("preregistration learner seeds must be a non-empty list")
    configure_recurrent_training_determinism(
        learner_seed=int(learner_seeds[0]),
        device=torch.device(args.device),
    )

    try:
        observed = build_recurrent_scale_runtime_provenance(
            source_commit=head,
            source_manifest_sha256=str(manifest["aggregate_sha256"]),
            preregistration_digest=str(preregistration["exact_digest"]),
            repository_clean=True,
            device=torch.device(args.device),
            rollout_workers=args.rollout_workers,
            counterfactual_workers=args.counterfactual_workers,
            evaluation_workers=args.evaluation_workers,
        )
        if args.output.exists():
            expected = load_strict_json(args.output)
            validate_recurrent_scale_runtime_provenance(expected)
            assert_recurrent_scale_runtime_provenance_match(expected, observed)
            operation = "verified"
        else:
            write_atomic_json(args.output, observed)
            operation = "captured"
    except RecurrentRuntimeProvenanceError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "recurrent_scale_runtime_provenance_"
        f"{operation} digest={observed['exact_digest']} output={args.output}"
    )
    return 0


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
