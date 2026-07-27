from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch

from evolution_sim.mind.open_ecology_phase_a import (
    OPEN_ECOLOGY_PHASE_A_CELL_ORDER,
    build_open_ecology_phase_a_evidence_index,
    build_open_ecology_phase_a_launch_authorization,
    build_open_ecology_phase_a_preregistration,
    build_open_ecology_phase_a_runtime_contract,
    build_open_ecology_phase_a_throughput_gate,
    configure_open_ecology_phase_a_determinism,
    open_ecology_phase_a_launch_readiness,
    run_open_ecology_phase_a_cell,
    validate_open_ecology_phase_a_launch_authorization,
    validate_open_ecology_phase_a_preregistration,
)
from evolution_sim.mind.open_ecology_phase_a_readiness import (
    DEPENDENCY_EVIDENCE_KINDS,
    OPERATIONAL_EVIDENCE_KINDS,
    produce_capture_noninterference_reexecution_report,
)
from evolution_sim.mind.recurrent_scale_campaign import source_file_hash_manifest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal and execute the fixed Phase-A open-ecology 2x2 causal "
            "critic/gradient campaign. Execution remains blocked without a "
            "complete machine-readable launch authorization."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--source-commit", required=True)
    preregister.add_argument(
        "--heritable-benchmark",
        type=Path,
        required=True,
    )
    preregister.add_argument(
        "--zero-all-benchmark",
        type=Path,
        required=True,
    )
    preregister.add_argument(
        "--resource-envelope",
        type=Path,
        required=True,
    )
    preregister.add_argument("--device", default="cuda")
    preregister.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--preregistration", type=Path, required=True)
    validate.add_argument("--launch-authorization", type=Path)

    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--preregistration", type=Path)
    readiness.add_argument("--evidence-index", type=Path)
    readiness.add_argument("--output", type=Path)

    build_index = subparsers.add_parser("build-evidence-index")
    build_index.add_argument("--preregistration", type=Path, required=True)
    build_index.add_argument("--evidence-root", type=Path, required=True)
    build_index.add_argument(
        "--dependency-report",
        action="append",
        default=[],
        metavar="DEPENDENCY/KIND=PATH",
    )
    build_index.add_argument(
        "--gate-report",
        action="append",
        default=[],
        metavar="GATE=PATH",
    )
    build_index.add_argument("--output", type=Path, required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--preregistration", type=Path, required=True)
    authorize.add_argument("--evidence-index", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)

    capture_reexecution = subparsers.add_parser("prove-capture-reexecution")
    capture_reexecution.add_argument(
        "--preregistration",
        type=Path,
        required=True,
    )
    capture_reexecution.add_argument(
        "--primary-proof",
        type=Path,
        required=True,
    )
    capture_reexecution.add_argument("--output", type=Path, required=True)

    run_cell = subparsers.add_parser("run-cell")
    run_cell.add_argument("--preregistration", type=Path, required=True)
    run_cell.add_argument("--expected-preregistration-digest", required=True)
    run_cell.add_argument(
        "--launch-authorization",
        type=Path,
        required=True,
    )
    run_cell.add_argument(
        "--cell",
        choices=OPEN_ECOLOGY_PHASE_A_CELL_ORDER,
        required=True,
    )
    run_cell.add_argument(
        "--learner-index",
        type=int,
        choices=range(4),
        required=True,
    )
    run_cell.add_argument("--output-root", type=Path, required=True)
    run_cell.add_argument("--device", default="cuda")
    run_cell.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preregister":
        head, clean = _git_source_state()
        if head != args.source_commit or not clean:
            raise SystemExit(
                "Phase A preregistration requires --source-commit to match "
                "a clean Git HEAD"
            )
        configure_open_ecology_phase_a_determinism()
        heritable = _load_strict_json(args.heritable_benchmark)
        zero_all = _load_strict_json(args.zero_all_benchmark)
        resource_envelope = _load_strict_json(args.resource_envelope)
        throughput_gate = build_open_ecology_phase_a_throughput_gate(
            source_commit=head,
            heritable_report=heritable,
            zero_all_report=zero_all,
            resource_envelope=resource_envelope,
        )
        runtime = build_open_ecology_phase_a_runtime_contract(
            device=torch.device(args.device),
            rollout_workers=int(throughput_gate["selected_rollout_workers"]),
        )
        manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
        preregistration = build_open_ecology_phase_a_preregistration(
            source_commit=head,
            source_manifest_sha256=str(manifest["aggregate_sha256"]),
            runtime_contract=runtime,
            throughput_gate=throughput_gate,
        )
        _write_atomic_json(args.output, preregistration)
        print(
            "open_ecology_phase_a_preregistered "
            f"digest={preregistration['exact_digest']} "
            f"workers={throughput_gate['selected_rollout_workers']} "
            f"output={args.output}"
        )
        return 0
    if args.command == "validate":
        preregistration = _load_strict_json(args.preregistration)
        validate_open_ecology_phase_a_preregistration(preregistration)
        if args.launch_authorization is not None:
            authorization = _load_strict_json(args.launch_authorization)
            validate_open_ecology_phase_a_launch_authorization(
                authorization,
                preregistration=preregistration,
                authorization_path=args.launch_authorization,
            )
        print(
            "open_ecology_phase_a_contract_valid "
            f"digest={preregistration['exact_digest']} "
            f"launch_authorized={args.launch_authorization is not None}"
        )
        return 0
    if args.command == "readiness":
        if (args.preregistration is None) != (args.evidence_index is None):
            raise SystemExit(
                "--preregistration and --evidence-index must be supplied together"
            )
        if args.preregistration is None:
            report = open_ecology_phase_a_launch_readiness()
        else:
            report = open_ecology_phase_a_launch_readiness(
                preregistration=_load_strict_json(args.preregistration),
                evidence_index=_load_strict_json(args.evidence_index),
                evidence_index_path=args.evidence_index,
            )
        if args.output is not None:
            _write_atomic_json(args.output, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0 if report["phase_a_training_authorized"] is True else 2
    if args.command == "build-evidence-index":
        evidence_root = args.evidence_root.resolve()
        if (
            args.evidence_root.is_symlink()
            or args.output.is_symlink()
            or args.output.resolve().parent != evidence_root
        ):
            raise SystemExit(
                "--output must be a non-symlink file directly inside --evidence-root"
            )
        preregistration = _load_strict_json(args.preregistration)
        dependency_reports = _parse_dependency_reports(args.dependency_report)
        gate_reports = _parse_gate_reports(args.gate_report)
        evidence_index = build_open_ecology_phase_a_evidence_index(
            preregistration,
            evidence_root=evidence_root,
            dependency_reports=dependency_reports,
            operational_reports=gate_reports,
        )
        _write_atomic_json(args.output, evidence_index)
        print(
            "open_ecology_phase_a_evidence_index_built "
            f"digest={evidence_index['exact_digest']} output={args.output}"
        )
        return 0
    if args.command == "authorize":
        preregistration = _load_strict_json(args.preregistration)
        evidence_index = _load_strict_json(args.evidence_index)
        authorization = build_open_ecology_phase_a_launch_authorization(
            preregistration,
            evidence_index=evidence_index,
            evidence_index_path=args.evidence_index,
            authorization_path=args.output,
        )
        _write_atomic_json(args.output, authorization)
        validate_open_ecology_phase_a_launch_authorization(
            authorization,
            preregistration=preregistration,
            authorization_path=args.output,
        )
        print(
            "open_ecology_phase_a_launch_authorized "
            f"digest={authorization['exact_digest']} output={args.output}"
        )
        return 0
    if args.command == "prove-capture-reexecution":
        report = produce_capture_noninterference_reexecution_report(
            _load_strict_json(args.preregistration),
            primary_proof_path=args.primary_proof,
        )
        _write_atomic_json(args.output, report)
        print(
            "open_ecology_capture_noninterference_reexecuted "
            f"digest={report['exact_digest']} output={args.output}"
        )
        return 0
    if args.command == "run-cell":
        preregistration = _load_strict_json(args.preregistration)
        report = run_open_ecology_phase_a_cell(
            preregistration,
            expected_preregistration_digest=(args.expected_preregistration_digest),
            launch_authorization_path=args.launch_authorization,
            cell_id=args.cell,
            learner_index=args.learner_index,
            output_root=args.output_root,
            device=torch.device(args.device),
            resume=args.resume,
        )
        print(
            "open_ecology_phase_a_cell_complete "
            f"run_id={report['run_id']} digest={report['exact_digest']}"
        )
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


def _git_source_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, not bool(status)


def _parse_dependency_reports(
    values: list[str],
) -> dict[str, dict[str, Path]]:
    reports: dict[str, dict[str, Path]] = {}
    for value in values:
        identity, separator, raw_path = value.partition("=")
        dependency_id, slash, evidence_kind = identity.partition("/")
        if not separator or not slash or not raw_path:
            raise SystemExit("--dependency-report must be DEPENDENCY/KIND=PATH")
        if (
            dependency_id not in DEPENDENCY_EVIDENCE_KINDS
            or evidence_kind not in DEPENDENCY_EVIDENCE_KINDS[dependency_id]
        ):
            raise SystemExit(f"unknown dependency evidence identity {identity!r}")
        dependency = reports.setdefault(dependency_id, {})
        if evidence_kind in dependency:
            raise SystemExit(f"duplicate dependency evidence {identity!r}")
        dependency[evidence_kind] = Path(raw_path)
    return reports


def _parse_gate_reports(values: list[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for value in values:
        gate_name, separator, raw_path = value.partition("=")
        if not separator or not raw_path or gate_name not in OPERATIONAL_EVIDENCE_KINDS:
            raise SystemExit("--gate-report must be a known GATE=PATH")
        if gate_name in reports:
            raise SystemExit(f"duplicate gate report {gate_name!r}")
        reports[gate_name] = Path(raw_path)
    return reports


def _load_strict_json(path: Path) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
