#!/usr/bin/env python3
"""Produce the exact-source causal-capture noninterference proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import evolution_sim.mind.open_ecology_selection as open_ecology_selection
import evolution_sim.mind.recurrent_scale_campaign as recurrent_scale_campaign


def _loaded_repository_roots() -> frozenset[Path]:
    selection_path = Path(open_ecology_selection.__file__).resolve()
    campaign_path = Path(recurrent_scale_campaign.__file__).resolve()
    script_path = Path(__file__).resolve()
    return frozenset(
        {
            selection_path.parents[3],
            campaign_path.parents[3],
            script_path.parents[1],
        }
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or "git command failed without stderr"
        )
    return completed.stdout.strip()


def _write_atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.pending-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
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
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def execute(*, repository_root: Path, output_path: Path) -> dict[str, object]:
    root = repository_root.resolve()
    loaded_roots = _loaded_repository_roots()
    if loaded_roots != frozenset({root}):
        raise RuntimeError(
            "capture proof source imports do not match --repository-root"
        )
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError(
            "capture noninterference proof requires a clean exact checkout"
        )
    commit = _git(root, "rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("git did not return one full source commit")
    manifest = recurrent_scale_campaign.source_file_hash_manifest(root)
    proof = open_ecology_selection.build_open_ecology_capture_noninterference_proof(
        source_commit=commit,
        source_manifest_sha256=str(manifest["aggregate_sha256"]),
    )
    _write_atomic_json(output_path.resolve(), proof)
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        proof = execute(
            repository_root=arguments.repository_root,
            output_path=arguments.output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"capture noninterference proof failed closed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": proof["schema_version"],
                "exact_digest": proof["exact_digest"],
                "case_count": len(proof["cases"]),
                "passed": proof["passed"],
                "output_path": str(arguments.output.resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
