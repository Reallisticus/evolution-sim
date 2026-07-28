"""Close and archive one complete, guardian-authorized Phase-A matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evolution_sim.mind.open_ecology_phase_a_archive import (
    archive_phase_a_terminal_matrix,
    close_phase_a_terminal_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all 16 Phase-A terminal chains, materialize an immutable "
            "closed bundle outside the active tree, and archive it through the "
            "sealed Drive uploader."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    close = subparsers.add_parser("close")
    close.add_argument("--preregistration", type=Path, required=True)
    close.add_argument("--launch-authorization", type=Path, required=True)
    close.add_argument("--guardian-transcript", type=Path, required=True)
    close.add_argument("--runtime-venv-authority", type=Path, required=True)
    close.add_argument("--active-output-root", type=Path, required=True)
    close.add_argument("--authority-root", type=Path, required=True)
    close.add_argument("--closed-bundle-parent", type=Path, required=True)
    close.add_argument("--bundle-id", required=True)
    close.add_argument("--closure-receipt", type=Path, required=True)

    archive = subparsers.add_parser("archive")
    archive.add_argument("--closure-receipt", type=Path, required=True)
    archive.add_argument("--expected-closure-receipt-sha256", required=True)
    archive.add_argument("--archive-tool-authority", type=Path, required=True)
    archive.add_argument(
        "--expected-archive-tool-authority-sha256",
        required=True,
    )
    archive.add_argument(
        "--remote-staging-directory",
        type=Path,
        required=True,
    )
    archive.add_argument("--drive-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "close":
        result = close_phase_a_terminal_matrix(
            preregistration_path=arguments.preregistration,
            launch_authorization_path=arguments.launch_authorization,
            guardian_transcript_path=arguments.guardian_transcript,
            runtime_venv_authority_path=arguments.runtime_venv_authority,
            active_output_root=arguments.active_output_root,
            authority_root=arguments.authority_root,
            closed_bundle_parent=arguments.closed_bundle_parent,
            bundle_id=arguments.bundle_id,
            closure_receipt_path=arguments.closure_receipt,
        )
    elif arguments.command == "archive":
        result = archive_phase_a_terminal_matrix(
            closure_receipt_path=arguments.closure_receipt,
            expected_closure_receipt_sha256=(arguments.expected_closure_receipt_sha256),
            archive_tool_authority_path=arguments.archive_tool_authority,
            expected_archive_tool_authority_sha256=(
                arguments.expected_archive_tool_authority_sha256
            ),
            remote_staging_directory=arguments.remote_staging_directory,
            drive_receipt_path=arguments.drive_receipt,
        )
    else:  # pragma: no cover - argparse enforces the closed command set.
        raise AssertionError(f"unhandled command {arguments.command!r}")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
