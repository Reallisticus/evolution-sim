from __future__ import annotations

import argparse
from pathlib import Path
import sys

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.learned_policy import (
    MIND_RUNTIME_MODE_GUARDED,
    MIND_RUNTIME_MODES,
    load_learned_policy,
)
from evolution_sim.mind.evolution import load_mind_v3_founder_template
from evolution_sim.mind.v3_neural import load_mind_v3_neural_artifact
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect trajectory records without building full replay surfaces."
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed.")
    parser.add_argument("--ticks", type=int, default=400, help="Maximum ticks to simulate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/trajectories/latest-trajectory.jsonl.gz"),
        help="Trajectory JSONL destination. Use a .gz suffix for gzip compression.",
    )
    parser.add_argument(
        "--split-id",
        default="unsplit",
        help="Dataset split identifier to record in trajectory provenance.",
    )
    parser.add_argument(
        "--mind-artifact",
        type=Path,
        help="Optional Mind model artifact to use as the trajectory policy.",
    )
    parser.add_argument(
        "--enable-mind",
        action="store_true",
        help="Required with --mind-artifact to enable learned-policy inference.",
    )
    parser.add_argument(
        "--include-policy-diagnostics",
        action="store_true",
        help=(
            "Persist learned policy decision diagnostics in each trajectory "
            "record when the active policy emits them."
        ),
    )
    parser.add_argument(
        "--include-policy-update-trace",
        action="store_true",
        help=(
            "Persist replayable online policy update traces when the active "
            "policy emits them."
        ),
    )
    parser.add_argument(
        "--mind-runtime-mode",
        choices=sorted(MIND_RUNTIME_MODES),
        default=MIND_RUNTIME_MODE_GUARDED,
        help=(
            "Mind runtime mode. The default guarded mode keeps heuristic safety "
            "fallback active; autonomous modes are legacy learned-artifact "
            "experiment surfaces."
        ),
    )
    parser.add_argument(
        "--mind-v3-autonomous-evolution",
        action="store_true",
        help=(
            "Use the feature-gated Mind v3 inherited autonomous controller "
            "without heuristic fallback."
        ),
    )
    parser.add_argument(
        "--mind-v3-founder-template",
        type=Path,
        help=(
            "Optional raw Mind v3 controller metadata or evolution-search "
            "report used by --mind-v3-autonomous-evolution."
        ),
    )
    parser.add_argument(
        "--mind-v3-neural-artifact",
        type=Path,
        help=(
            "Optional frozen Mind v3 neural policy artifact used by "
            "--mind-v3-autonomous-evolution."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mind_v3_founder_template is not None and not args.mind_v3_autonomous_evolution:
        raise SystemExit(
            "--mind-v3-founder-template requires --mind-v3-autonomous-evolution"
        )
    if args.mind_v3_neural_artifact is not None and not args.mind_v3_autonomous_evolution:
        raise SystemExit(
            "--mind-v3-neural-artifact requires --mind-v3-autonomous-evolution"
        )
    if args.mind_v3_autonomous_evolution and (
        args.mind_artifact is not None or args.enable_mind
    ):
        raise SystemExit(
            "--mind-v3-autonomous-evolution cannot be combined with "
            "--mind-artifact or --enable-mind"
        )
    if args.enable_mind and args.mind_artifact is None:
        raise SystemExit("--enable-mind requires --mind-artifact")
    if args.mind_artifact is not None and not args.enable_mind:
        raise SystemExit("--mind-artifact requires --enable-mind")
    if args.mind_v3_autonomous_evolution:
        founder_template = (
            load_mind_v3_founder_template(args.mind_v3_founder_template)
            if args.mind_v3_founder_template is not None
            else None
        )
        neural_artifact = (
            load_mind_v3_neural_artifact(args.mind_v3_neural_artifact)
            if args.mind_v3_neural_artifact is not None
            else None
        )
        policy = MindV3EvolutionPolicy(
            seed=args.seed,
            founder_template_metadata=founder_template,
            neural_artifact=neural_artifact,
        )
        mind_runtime_mode = "mind-v3-autonomous-evolution"
    else:
        policy = (
            load_learned_policy(
                args.mind_artifact,
                enable_mind=args.enable_mind,
                runtime_mode=args.mind_runtime_mode,
            )
            if args.mind_artifact is not None
            else None
        )
        mind_runtime_mode = args.mind_runtime_mode
    config = WorldConfig(seed=args.seed, max_ticks=args.ticks)
    writer = JsonlTrajectoryWriter(
        args.output,
        source_seeds=[args.seed],
        split_id=args.split_id,
        include_policy_decision_diagnostics=args.include_policy_diagnostics,
        include_policy_update_trace=args.include_policy_update_trace,
    )
    if args.output.exists():
        print(f"warning: overwriting existing trajectory {args.output}", file=sys.stderr)
    try:
        result = SimulationWorld(config, policy=policy).run(
            mode=RunMode.SUMMARY_ONLY,
            trajectory_sink=writer,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"failed to write trajectory {args.output}: {exc}") from exc

    summary = result.summary
    print(f"trajectory={args.output}")
    print(f"run_id={summary['run_id']}")
    print(f"ticks_executed={summary['ticks_executed']}")
    print(f"alive_agents={summary['alive_agents']}")
    print(f"births={summary['births']}")
    print(f"deaths={summary['deaths']}")
    if policy is not None:
        print(f"mind_policy={policy.policy_id}")
        print(f"mind_policy_version={policy.policy_version}")
        print(f"mind_runtime_mode={mind_runtime_mode}")
        if args.mind_v3_founder_template is not None:
            print(f"mind_v3_founder_template={args.mind_v3_founder_template}")
        if args.mind_v3_neural_artifact is not None:
            print(f"mind_v3_neural_artifact={args.mind_v3_neural_artifact}")
    print(
        "policy_decision_diagnostics="
        f"{'included' if args.include_policy_diagnostics else 'omitted'}"
    )
    print(
        "policy_update_trace="
        f"{'included' if args.include_policy_update_trace else 'omitted'}"
    )
    print(f"trajectory_records={writer.record_count}")
    print(f"mean_reward={writer.trajectory_summary['mean_reward']}")


if __name__ == "__main__":
    main()
