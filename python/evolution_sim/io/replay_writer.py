from __future__ import annotations

import json
from pathlib import Path

from evolution_sim.env import SimulationWorldResult


def write_json_replay(result: SimulationWorldResult, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.run_id,
        "config": result.config,
        "summary": result.summary,
        "events": result.events,
        "viewer": result.viewer,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
