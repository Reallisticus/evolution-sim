from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from evolution_sim.env.contracts import REPLAY_TOP_LEVEL_KEYS
from evolution_sim.env import SimulationWorldResult
from evolution_sim.env.runtime.state import RunMode


def write_json_replay(result: SimulationWorldResult, output_path: str | Path) -> Path:
    if result.mode != RunMode.FULL_REPLAY or result.events is None or result.viewer is None:
        raise ValueError("write_json_replay only supports full replay results.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay_parts = {
        "run_id": result.run_id,
        "config": result.config,
        "summary": result.summary,
        "events": result.events,
        "viewer": result.viewer,
    }
    payload = {key: replay_parts[key] for key in REPLAY_TOP_LEVEL_KEYS}
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
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination
