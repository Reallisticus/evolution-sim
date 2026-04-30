from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from evolution_sim.env.contracts import REPLAY_TOP_LEVEL_KEYS
from evolution_sim.env import SimulationWorldResult
from evolution_sim.env.runtime.state import RunMode

REPLAY_JSON_SEPARATORS = (",", ":")


def build_replay_payload(result: SimulationWorldResult) -> dict[str, object]:
    if result.mode != RunMode.FULL_REPLAY or result.events is None or result.viewer is None:
        raise ValueError("build_replay_payload only supports full replay results.")
    replay_parts = {
        "run_id": result.run_id,
        "config": result.config,
        "summary": result.summary,
        "events": result.events,
        "viewer": result.viewer,
    }
    return {key: replay_parts[key] for key in REPLAY_TOP_LEVEL_KEYS}


def replay_payload_size_bytes(payload: dict[str, object]) -> int:
    return len(
        json.dumps(payload, separators=REPLAY_JSON_SEPARATORS).encode("utf-8")
    )


def write_json_replay(result: SimulationWorldResult, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_replay_payload(result)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, separators=REPLAY_JSON_SEPARATORS)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination
