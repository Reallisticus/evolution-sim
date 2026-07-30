#!/usr/bin/env python3
"""Fail-closed compatibility CLI for the retired remote archive streamer.

The historical two-process SSH-to-rclone transport did not have the bounded
process-group liveness contract required by the open-ecology campaign. Git
retains that implementation as provenance; no executable copy remains here.
Use ``archive_open_ecology_campaign.py`` under its sealed authority instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
import sys
from typing import Any


DEFAULT_RCLONE_REMOTE = "gdrive:evolution-sim-backups"
DEFAULT_RCLONE_SUBDIR = "archives"
_SSH_TARGET_PATTERN = re.compile(r"[A-Za-z0-9_.@-]+")
_ARCHIVE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.zst")


class RemoteArchiveError(RuntimeError):
    """The retired remote archive transport was requested."""


@dataclass(frozen=True, slots=True)
class RemoteArchiveOptions:
    ssh_target: str
    remote_repository_root: str
    remote_input_dir: str
    remote_staging_dir: str
    archive_name: str
    rclone_remote: str = DEFAULT_RCLONE_REMOTE
    rclone_subdir: str = DEFAULT_RCLONE_SUBDIR

    def validate(self) -> None:
        if (
            not self.ssh_target
            or self.ssh_target.startswith("-")
            or _SSH_TARGET_PATTERN.fullmatch(self.ssh_target) is None
        ):
            raise RemoteArchiveError("ssh target contains unsupported characters")
        repository_root = _absolute_remote_path(
            self.remote_repository_root,
            field="remote repository root",
        )
        input_dir = _absolute_remote_path(
            self.remote_input_dir,
            field="remote input directory",
        )
        staging_dir = _absolute_remote_path(
            self.remote_staging_dir,
            field="remote staging directory",
        )
        if input_dir == PurePosixPath("/"):
            raise RemoteArchiveError("refusing to archive the remote filesystem root")
        if _is_relative_to(staging_dir, input_dir):
            raise RemoteArchiveError(
                "remote staging directory may not be inside the input directory"
            )
        if repository_root == PurePosixPath("/"):
            raise RemoteArchiveError(
                "remote repository root must identify an exact checkout"
            )
        if _ARCHIVE_NAME_PATTERN.fullmatch(self.archive_name) is None:
            raise RemoteArchiveError(
                "archive name must be one conservative filename ending in .tar.zst"
            )
        _validate_rclone_directory(self.rclone_remote, self.rclone_subdir)


def execute(options: RemoteArchiveOptions) -> dict[str, object]:
    """Reject every programmatic use before starting a transport."""

    options.validate()
    raise RemoteArchiveError(
        "legacy remote archive streaming is deauthorized; use "
        "scripts/archive_open_ecology_campaign.py"
    )


def _absolute_remote_path(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise RemoteArchiveError(f"{field} must be a non-empty safe path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise RemoteArchiveError(f"{field} must be an absolute normalized path")
    return path


def _is_relative_to(path: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_rclone_directory(remote: str, subdir: str) -> None:
    if (
        not isinstance(remote, str)
        or ":" not in remote
        or remote.startswith("-")
        or any(character in remote for character in ("\x00", "\n", "\r"))
    ):
        raise RemoteArchiveError("rclone remote must be one safe remote path")
    if (
        not isinstance(subdir, str)
        or not subdir
        or subdir.startswith("/")
        or ".." in PurePosixPath(subdir).parts
        or any(character in subdir for character in ("\x00", "\n", "\r"))
    ):
        raise RemoteArchiveError("rclone subdirectory must be one safe relative path")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retired compatibility command. All valid invocations fail closed "
            "and direct operators to the sealed open-ecology archiver."
        )
    )
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--remote-repository-root", required=True)
    parser.add_argument("--remote-input-dir", required=True)
    parser.add_argument("--remote-staging-dir", required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--rclone-remote", default=DEFAULT_RCLONE_REMOTE)
    parser.add_argument("--rclone-subdir", default=DEFAULT_RCLONE_SUBDIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    options = RemoteArchiveOptions(
        ssh_target=parsed.ssh_target,
        remote_repository_root=parsed.remote_repository_root,
        remote_input_dir=parsed.remote_input_dir,
        remote_staging_dir=parsed.remote_staging_dir,
        archive_name=parsed.archive_name,
        rclone_remote=parsed.rclone_remote,
        rclone_subdir=parsed.rclone_subdir,
    )
    try:
        execute(options)
    except RemoteArchiveError as error:
        print(f"remote archive failed closed: {error}", file=sys.stderr)
        return 2
    raise AssertionError("retired remote archive transport unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
