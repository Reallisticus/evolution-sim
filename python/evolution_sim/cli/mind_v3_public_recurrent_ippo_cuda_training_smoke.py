from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evolution_sim.mind.recurrent_cuda_training_smoke import (
    run_recurrent_cuda_training_smoke,
)
from evolution_sim.mind.recurrent_scale_campaign import load_strict_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated one-update CUDA recurrent-IPPO, exact-branch, "
            "and crash-resume parity launch gate."
        )
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-digest", required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-workers", type=int, required=True)
    parser.add_argument("--counterfactual-workers", type=int, required=True)
    parser.add_argument("--evaluation-workers", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA training smoke requires an available CUDA device")
    preregistration = load_strict_json(args.preregistration)
    report = run_recurrent_cuda_training_smoke(
        preregistration,
        expected_preregistration_digest=args.expected_preregistration_digest,
        runtime_provenance_path=args.runtime_provenance,
        output_root=args.output_root,
        device=device,
        rollout_workers=args.rollout_workers,
        counterfactual_workers=args.counterfactual_workers,
        evaluation_workers=args.evaluation_workers,
    )
    print(
        "recurrent_cuda_training_smoke_complete "
        f"digest={report['exact_digest']} output={args.output_root / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
