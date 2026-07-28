"""CLI for the live two-party Phase-A training guardian."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evolution_sim.cli.open_ecology_health import collect_host_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase-A only while live Mac storage authority and an "
            "exact-source GPU guardian share one authenticated SSH channel."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    coordinate = subparsers.add_parser("coordinate")
    coordinate.add_argument("--local-preregistration", type=Path, required=True)
    coordinate.add_argument(
        "--local-launch-authorization",
        type=Path,
        required=True,
    )
    coordinate.add_argument("--remote-preregistration", type=Path, required=True)
    coordinate.add_argument(
        "--remote-launch-authorization",
        type=Path,
        required=True,
    )
    coordinate.add_argument("--remote-output-root", type=Path, required=True)
    coordinate.add_argument("--archive-tool-authority", type=Path, required=True)
    coordinate.add_argument(
        "--expected-archive-tool-authority-sha256",
        required=True,
    )
    coordinate.add_argument(
        "--remote-runtime-venv-authority",
        type=Path,
        required=True,
    )
    coordinate.add_argument(
        "--expected-remote-runtime-venv-authority-sha256",
        required=True,
    )
    coordinate.add_argument(
        "--github-token-stdin",
        action="store_true",
        required=True,
    )
    coordinate.add_argument("--transcript", type=Path, required=True)
    coordinate.add_argument("--device", default="cuda")
    coordinate.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=14 * 60 * 60.0,
    )
    coordinate.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=24 * 60 * 60.0,
    )

    serve = subparsers.add_parser("serve")
    serve.add_argument("--preregistration", type=Path, required=True)
    serve.add_argument("--launch-authorization", type=Path, required=True)
    serve.add_argument("--output-root", type=Path, required=True)
    serve.add_argument("--source-git-sha", required=True)
    serve.add_argument("--source-manifest-sha256", required=True)
    serve.add_argument("--preregistration-digest", required=True)
    serve.add_argument("--evidence-index-digest", required=True)
    serve.add_argument("--launch-authorization-digest", required=True)
    serve.add_argument("--runtime-venv-authority", type=Path, required=True)
    serve.add_argument("--runtime-venv-authority-sha256", required=True)
    serve.add_argument("--archive-authority-sha256", required=True)
    serve.add_argument("--ssh-connection-sha256", required=True)
    serve.add_argument("--git-executable", type=Path, required=True)
    serve.add_argument("--git-executable-sha256", required=True)
    serve.add_argument("--ssh-target", required=True)
    serve.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "coordinate":
        from evolution_sim.mind.open_ecology_phase_a_guardian import (
            run_mac_coordinator,
        )

        github_token = _read_github_token_from_stdin()
        transcript = run_mac_coordinator(
            local_preregistration_path=args.local_preregistration,
            local_launch_authorization_path=args.local_launch_authorization,
            remote_preregistration_path=args.remote_preregistration,
            remote_launch_authorization_path=args.remote_launch_authorization,
            remote_output_root=args.remote_output_root,
            archive_authority_path=args.archive_tool_authority,
            expected_archive_authority_sha256=(
                args.expected_archive_tool_authority_sha256
            ),
            remote_runtime_venv_authority_path=(args.remote_runtime_venv_authority),
            expected_remote_runtime_venv_authority_sha256=(
                args.expected_remote_runtime_venv_authority_sha256
            ),
            transcript_path=args.transcript,
            github_token=github_token,
            device=args.device,
            ready_timeout_seconds=args.ready_timeout_seconds,
            cell_timeout_seconds=args.cell_timeout_seconds,
        )
        print(json.dumps(transcript, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "serve":
        from evolution_sim.io.open_ecology_runtime_venv_authority import (
            revalidate_running_guardian_runtime,
        )

        revalidate_running_guardian_runtime(
            authority_path=args.runtime_venv_authority,
            expected_authority_sha256=args.runtime_venv_authority_sha256,
            expected_source_git_sha=args.source_git_sha,
            expected_source_manifest_sha256=args.source_manifest_sha256,
            expected_archive_authority_sha256=args.archive_authority_sha256,
            expected_ssh_target=args.ssh_target,
            expected_ssh_connection_sha256=args.ssh_connection_sha256,
            git_executable=args.git_executable,
            git_executable_sha256=args.git_executable_sha256,
            observed_entrypoint_path=Path(__file__),
        )
        from evolution_sim.mind.open_ecology_phase_a_guardian import (
            PhaseAAuthorityBindings,
            install_guardian_process_safety,
            run_remote_guardian_session,
        )

        liveness = install_guardian_process_safety()
        result = run_remote_guardian_session(
            stdin=sys.stdin.buffer,
            stdout=sys.stdout.buffer,
            preregistration_path=args.preregistration,
            launch_authorization_path=args.launch_authorization,
            output_root=args.output_root,
            expected_bindings=PhaseAAuthorityBindings(
                source_git_sha=args.source_git_sha,
                source_manifest_sha256=args.source_manifest_sha256,
                preregistration_digest=args.preregistration_digest,
                evidence_index_digest=args.evidence_index_digest,
                launch_authorization_digest=args.launch_authorization_digest,
                runtime_venv_authority_sha256=(args.runtime_venv_authority_sha256),
                ssh_target=args.ssh_target,
            ),
            channel=liveness,
            host_observer=collect_host_observations,
            device=args.device,
        )
        if len(result["completed"]) != 16:
            raise RuntimeError("remote guardian did not complete the fixed matrix")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


def _read_github_token_from_stdin() -> bytearray:
    payload = sys.stdin.buffer.readline(4097)
    if (
        not payload
        or len(payload) > 4096
        or not payload.endswith(b"\n")
        or b"\x00" in payload
    ):
        raise RuntimeError(
            "--github-token-stdin requires one bounded newline-terminated token"
        )
    token = payload[:-1]
    if len(token) < 20 or any(byte < 0x21 or byte > 0x7E for byte in token):
        raise RuntimeError("GitHub token input is malformed")
    return bytearray(token)


if __name__ == "__main__":
    raise SystemExit(main())
