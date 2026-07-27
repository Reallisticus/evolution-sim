"""Torch-free source-manifest construction for exact-runtime binding."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evolution_sim.mind.provenance import stable_payload_digest


class SourceManifestError(ValueError):
    """The repository did not contain the sealed runtime source set."""


def source_file_hash_manifest(repository_root: str | Path) -> dict[str, object]:
    """Hash the runtime package and dependency contract in canonical order."""

    root = Path(repository_root).resolve()
    source_root = root / "python" / "evolution_sim"
    paths = sorted(source_root.rglob("*.py"))
    for repository_file in ("package.json", "requirements-mind-ml.txt"):
        candidate = root / repository_file
        if candidate.is_file():
            paths.append(candidate)
    paths.sort()
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    if not files:
        raise SourceManifestError("source manifest found no files")
    return {
        "hash_algorithm": "sha256",
        "path_contract": (
            "repository_relative_sorted_runtime_python_package_and_mind_requirements"
        ),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": stable_payload_digest(files),
    }


__all__ = ["SourceManifestError", "source_file_hash_manifest"]
