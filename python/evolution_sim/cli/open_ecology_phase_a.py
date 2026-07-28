from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from evolution_sim.cli.open_ecology_health import collect_host_observations
from evolution_sim.io.open_ecology_git_authority import (
    OpenEcologyGitAuthorityError,
    PinnedGitExecutable,
    discover_pinned_git_executable,
    run_pinned_git,
)
from evolution_sim.mind.open_ecology_phase_a import (
    OPEN_ECOLOGY_PHASE_A_CELL_ORDER,
    OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH,
    OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256,
    build_open_ecology_phase_a_evidence_index,
    build_open_ecology_phase_a_launch_authorization,
    build_open_ecology_phase_a_preregistration,
    build_open_ecology_phase_a_resource_envelope,
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
    _rename_path_no_replace,
    produce_campaign_storage_capacity_report,
    produce_capture_noninterference_reexecution_report,
    produce_cross_surface_and_self_echo_report,
    produce_critic_gradient_and_density_schedule_report,
    produce_exact_sha_phase_a_training_and_torch_ci_report,
    produce_fixed_batch_equivalence_and_speed_report,
    produce_output_lock_contention_report,
    produce_phase_a_training_throughput_report,
    produce_preregistration_roundtrip_fail_closed_report,
    produce_runtime_genome_and_action_source_report,
)
from evolution_sim.mind.open_ecology_phase_a_qualification import (
    ephemeral_github_credential,
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

    resource_envelope = subparsers.add_parser("build-resource-envelope")
    resource_envelope.add_argument("--source-commit", required=True)
    resource_envelope.add_argument("--output", type=Path, required=True)

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
    preregister.add_argument(
        "--expected-archive-tool-authority-sha256",
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

    preregistration_roundtrip = subparsers.add_parser("prove-preregistration-roundtrip")
    preregistration_roundtrip.add_argument(
        "--preregistration",
        type=Path,
        required=True,
    )
    preregistration_roundtrip.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    cross_surface = subparsers.add_parser("prove-cross-surface-and-self-echo")
    cross_surface.add_argument("--preregistration", type=Path, required=True)
    cross_surface.add_argument("--output-directory", type=Path, required=True)

    runtime_genome = subparsers.add_parser("prove-runtime-genome-and-action-source")
    runtime_genome.add_argument("--preregistration", type=Path, required=True)
    runtime_genome.add_argument("--output-directory", type=Path, required=True)

    critic_gradient = subparsers.add_parser(
        "prove-critic-gradient-and-density-schedule"
    )
    critic_gradient.add_argument("--preregistration", type=Path, required=True)
    critic_gradient.add_argument("--output-directory", type=Path, required=True)

    fixed_batch = subparsers.add_parser("prove-fixed-batch-equivalence-and-speed")
    fixed_batch.add_argument("--preregistration", type=Path, required=True)
    fixed_batch.add_argument("--device", default="cuda")
    fixed_batch.add_argument("--output-directory", type=Path, required=True)

    exact_source = subparsers.add_parser("prove-exact-source-training")
    exact_source.add_argument("--preregistration", type=Path, required=True)
    exact_source.add_argument("--host-class", required=True)
    exact_source.add_argument(
        "--heritable-benchmark",
        type=Path,
        required=True,
    )
    exact_source.add_argument(
        "--zero-all-benchmark",
        type=Path,
        required=True,
    )
    exact_source.add_argument(
        "--github-token-stdin",
        action="store_true",
        help=(
            "read one bounded GitHub token from stdin and retain it only in "
            "memory for the exact-SHA HTTPS request"
        ),
    )
    exact_source.add_argument("--output-directory", type=Path, required=True)

    training_throughput = subparsers.add_parser("prove-training-throughput")
    training_throughput.add_argument(
        "--preregistration",
        type=Path,
        required=True,
    )
    training_throughput.add_argument("--host-class", required=True)
    training_throughput.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    storage = subparsers.add_parser("prove-storage-capacity")
    storage.add_argument("--preregistration", type=Path, required=True)
    storage.add_argument("--target-filesystem", type=Path, required=True)
    storage.add_argument(
        "--archive-tool-authority",
        type=Path,
        required=True,
    )
    storage.add_argument(
        "--expected-archive-tool-authority-sha256",
        required=True,
    )
    storage.add_argument("--drive-remote", default="gdrive:")
    storage.add_argument("--output-directory", type=Path, required=True)

    output_lock = subparsers.add_parser("prove-output-lock")
    output_lock.add_argument("--preregistration", type=Path, required=True)
    output_lock.add_argument("--output-directory", type=Path, required=True)

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
    if args.command == "build-resource-envelope":
        try:
            git_authority = discover_pinned_git_executable()
            head, clean = _git_source_state(git_authority=git_authority)
        except (OpenEcologyGitAuthorityError, RuntimeError) as error:
            raise SystemExit(
                "Phase A resource-envelope Git inspection failed closed"
            ) from error
        if head != args.source_commit or not clean:
            raise SystemExit(
                "Phase A resource-envelope construction requires "
                "--source-commit to match a clean Git HEAD"
            )
        initial_manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
        initial_document_sha256 = _preregistration_document_sha256()
        if initial_document_sha256 != OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256:
            raise SystemExit(
                "Phase A resource-envelope evidence differs from the sealed "
                "preregistration document"
            )
        envelope = build_open_ecology_phase_a_resource_envelope(
            source_commit=head,
        )
        final_manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
        final_document_sha256 = _preregistration_document_sha256()
        try:
            final_head, final_clean = _git_source_state(
                git_authority=git_authority,
            )
        except (OpenEcologyGitAuthorityError, RuntimeError) as error:
            raise SystemExit(
                "Phase A resource-envelope Git reinspection failed closed"
            ) from error
        if (
            final_head != head
            or not final_clean
            or final_manifest != initial_manifest
            or final_document_sha256 != initial_document_sha256
        ):
            raise SystemExit(
                "Phase A source changed while the resource envelope was being assembled"
            )
        _write_atomic_json(args.output, envelope)
        print(
            "open_ecology_phase_a_resource_envelope_built "
            f"digest={envelope['exact_digest']} output={args.output}"
        )
        return 0
    if args.command == "preregister":
        try:
            git_authority = discover_pinned_git_executable()
            head, clean = _git_source_state(git_authority=git_authority)
        except (OpenEcologyGitAuthorityError, RuntimeError) as error:
            raise SystemExit("Phase A Git source inspection failed closed") from error
        if head != args.source_commit or not clean:
            raise SystemExit(
                "Phase A preregistration requires --source-commit to match "
                "a clean Git HEAD"
            )
        initial_manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
        initial_document_sha256 = _preregistration_document_sha256()
        if initial_document_sha256 != OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_SHA256:
            raise SystemExit(
                "Phase A preregistration document differs from its sealed hash"
            )
        configure_open_ecology_phase_a_determinism()
        heritable = _load_strict_json(args.heritable_benchmark)
        zero_all = _load_strict_json(args.zero_all_benchmark)
        resource_envelope = _load_canonical_resource_envelope(
            args.resource_envelope,
            source_commit=head,
        )
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
        final_manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
        final_document_sha256 = _preregistration_document_sha256()
        try:
            final_head, final_clean = _git_source_state(
                git_authority=git_authority,
            )
        except (OpenEcologyGitAuthorityError, RuntimeError) as error:
            raise SystemExit("Phase A Git source reinspection failed closed") from error
        if (
            final_head != head
            or not final_clean
            or final_manifest != initial_manifest
            or final_document_sha256 != initial_document_sha256
        ):
            raise SystemExit(
                "Phase A source changed while preregistration was being assembled"
            )
        _load_canonical_resource_envelope(
            args.resource_envelope,
            source_commit=head,
        )
        preregistration = build_open_ecology_phase_a_preregistration(
            source_commit=head,
            source_manifest_sha256=str(initial_manifest["aggregate_sha256"]),
            archive_tool_authority_sha256=(args.expected_archive_tool_authority_sha256),
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
    if args.command == "prove-preregistration-roundtrip":
        report = produce_preregistration_roundtrip_fail_closed_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
        )
        print(
            "open_ecology_preregistration_roundtrip_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-cross-surface-and-self-echo":
        report = produce_cross_surface_and_self_echo_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
        )
        print(
            "open_ecology_cross_surface_and_self_echo_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-runtime-genome-and-action-source":
        report = produce_runtime_genome_and_action_source_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
        )
        print(
            "open_ecology_runtime_genome_and_action_source_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-critic-gradient-and-density-schedule":
        report = produce_critic_gradient_and_density_schedule_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
        )
        print(
            "open_ecology_critic_gradient_and_density_schedule_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-fixed-batch-equivalence-and-speed":
        report = produce_fixed_batch_equivalence_and_speed_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
            device=args.device,
        )
        print(
            "open_ecology_fixed_batch_equivalence_and_speed_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-exact-source-training":
        preregistration = _load_strict_json(args.preregistration)
        if args.github_token_stdin:
            github_token = _read_secret_token_from_stdin()
            with ephemeral_github_credential(github_token):
                report = produce_exact_sha_phase_a_training_and_torch_ci_report(
                    preregistration,
                    output_directory=args.output_directory,
                    host_class=args.host_class,
                    heritable_benchmark_path=args.heritable_benchmark,
                    zero_all_benchmark_path=args.zero_all_benchmark,
                )
            if any(github_token):
                raise SystemExit("GitHub token was not zeroed after D10 production")
        else:
            report = produce_exact_sha_phase_a_training_and_torch_ci_report(
                preregistration,
                output_directory=args.output_directory,
                host_class=args.host_class,
                heritable_benchmark_path=args.heritable_benchmark,
                zero_all_benchmark_path=args.zero_all_benchmark,
            )
        print(
            "open_ecology_exact_source_training_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-training-throughput":
        report = produce_phase_a_training_throughput_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
            host_class=args.host_class,
            host_observer=collect_host_observations,
        )
        print(
            "open_ecology_training_throughput_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-storage-capacity":
        report = produce_campaign_storage_capacity_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
            target_filesystem=args.target_filesystem,
            archive_tool_authority_path=args.archive_tool_authority,
            expected_archive_tool_authority_sha256=(
                args.expected_archive_tool_authority_sha256
            ),
            drive_remote=args.drive_remote,
        )
        print(
            "open_ecology_storage_capacity_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
        )
        return 0
    if args.command == "prove-output-lock":
        report = produce_output_lock_contention_report(
            _load_strict_json(args.preregistration),
            output_directory=args.output_directory,
        )
        print(
            "open_ecology_output_lock_proved "
            f"digest={report['exact_digest']} "
            f"output={args.output_directory / 'report.json'}"
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


def _git_source_state(
    *,
    git_authority: PinnedGitExecutable | None = None,
) -> tuple[str, bool]:
    try:
        authority = git_authority or discover_pinned_git_executable()
        head = run_pinned_git(
            authority,
            repository_root=_REPOSITORY_ROOT,
            arguments=("rev-parse", "HEAD"),
        )
        status = run_pinned_git(
            authority,
            repository_root=_REPOSITORY_ROOT,
            arguments=("status", "--porcelain=v1", "--untracked-files=normal"),
        )
    except OpenEcologyGitAuthorityError as error:
        raise RuntimeError("Phase A Git source inspection failed closed") from error
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


def _read_secret_token_from_stdin() -> bytearray:
    payload = os.read(0, 4097)
    if (
        not payload
        or len(payload) > 4096
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
    ):
        raise SystemExit(
            "--github-token-stdin requires one bounded newline-terminated token"
        )
    token = payload[:-1]
    if len(token) < 20 or any(byte < 0x21 or byte > 0x7E for byte in token):
        raise SystemExit("GitHub token input is malformed")
    return bytearray(token)


def _load_canonical_resource_envelope(
    path: Path,
    *,
    source_commit: str,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "Phase A resource envelope must be one regular non-symlink file"
        )
    envelope = _load_strict_json(path)
    expected = build_open_ecology_phase_a_resource_envelope(
        source_commit=source_commit,
    )
    if envelope != expected or path.read_bytes() != _canonical_json_bytes(expected):
        raise ValueError(
            "Phase A preregistration requires the byte-canonical resource envelope"
        )
    return envelope


def _preregistration_document_sha256() -> str:
    path = _REPOSITORY_ROOT / OPEN_ECOLOGY_PHASE_A_PREREGISTRATION_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "sealed Phase A preregistration document must be one regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    raw_parent = path.parent
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"authority output already exists: {path}")
    if raw_parent.is_symlink():
        raise ValueError("authority output parent must be one real directory")
    parent = raw_parent.resolve()
    if not parent.is_dir():
        raise ValueError("authority output parent must be one real directory")
    destination = parent / path.name
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        # The platform-exclusive rename is atomic and create-if-absent. It
        # neither overwrites prior authority nor leaves a second hard link
        # during a crash window.
        _rename_path_no_replace(temporary, destination)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
