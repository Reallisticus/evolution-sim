from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SPLIT_ID = "unsplit"


def stable_payload_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dataset_provenance(
    *,
    config: dict[str, object],
    contract: dict[str, object],
    trajectory_paths: Sequence[str | Path],
    source_seeds: Sequence[int] | None = None,
    split_id: str = DEFAULT_SPLIT_ID,
    record_count: int | None = None,
) -> dict[str, object]:
    return {
        "source_seeds": [int(seed) for seed in (source_seeds or [])],
        "config_digest": stable_payload_digest(config),
        "contract_digest": stable_payload_digest(contract),
        "split_id": split_id,
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "record_count": record_count,
    }


def validate_dataset_provenance(provenance: object) -> dict[str, object]:
    if not isinstance(provenance, dict):
        raise ValueError("dataset provenance must be an object")
    source_seeds = provenance.get("source_seeds")
    trajectory_paths = provenance.get("trajectory_paths")
    if not isinstance(source_seeds, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool)
        for seed in source_seeds
    ):
        raise ValueError("dataset provenance source_seeds must be integer list")
    for field in ("config_digest", "contract_digest", "split_id"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"dataset provenance {field} must be a non-empty string")
    if not isinstance(trajectory_paths, list) or not all(
        isinstance(path, str) and path for path in trajectory_paths
    ):
        raise ValueError("dataset provenance trajectory_paths must be string list")
    record_count = provenance.get("record_count")
    if record_count is not None and (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
    ):
        raise ValueError("dataset provenance record_count must be a non-negative integer")
    return provenance
