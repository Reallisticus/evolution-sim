from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.io import write_json_replay


GOLDEN_SPECIATION_SEED = 3
REQUIRED_PYTHON_VERSION = "3.14.3"
REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "python" / "tests" / "goldens"
MANIFEST_PATH = GOLDEN_DIR / "replay_manifest.json"


@dataclass(frozen=True, slots=True)
class GoldenScenario:
    name: str
    seed: int
    ticks: int
    requires_speciation: bool = False

    @property
    def fixture_path(self) -> Path:
        return GOLDEN_DIR / f"{self.name}.json.gz"


SCENARIOS: tuple[GoldenScenario, ...] = (
    GoldenScenario(name="seed7_ticks20", seed=7, ticks=20),
    GoldenScenario(name="seed7_ticks100", seed=7, ticks=100),
    GoldenScenario(
        name="speciation_seed_ticks320",
        seed=GOLDEN_SPECIATION_SEED,
        ticks=320,
        requires_speciation=True,
    ),
    GoldenScenario(name="seed5_ticks120_provenance", seed=5, ticks=120),
)


def _payload_bytes(scenario: GoldenScenario) -> bytes:
    result = SimulationWorld(WorldConfig(seed=scenario.seed, max_ticks=scenario.ticks)).run()
    if scenario.requires_speciation and int(result.summary.get("speciation_events", 0)) <= 0:
        raise SystemExit(
            f"Scenario {scenario.name} expected speciation events for seed={scenario.seed}."
        )
    with TemporaryDirectory() as tmpdir:
        replay_path = write_json_replay(result, Path(tmpdir) / f"{scenario.name}.json")
        return replay_path.read_bytes()


def _write_manifest(entries: list[dict[str, object]]) -> None:
    payload = {
        "python_version": REQUIRED_PYTHON_VERSION,
        "scenarios": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assert_python_runtime() -> None:
    actual = platform.python_version()
    if actual != REQUIRED_PYTHON_VERSION:
        raise SystemExit(
            f"Golden runtime pin mismatch: expected Python {REQUIRED_PYTHON_VERSION}, got {actual}."
        )


def _manifest_entry_for(scenario: GoldenScenario) -> dict[str, object]:
    payload_bytes = _payload_bytes(scenario)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    with gzip.open(scenario.fixture_path, "wb") as handle:
        handle.write(payload_bytes)
    print(f"updated {scenario.name} sha256={digest} bytes={len(payload_bytes)}", flush=True)
    return {
        **asdict(scenario),
        "fixture": scenario.fixture_path.name,
        "sha256": digest,
        "bytes": len(payload_bytes),
    }


def update_goldens(scenarios: tuple[GoldenScenario, ...]) -> None:
    _assert_python_runtime()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    selected_names = {scenario.name for scenario in scenarios}
    existing_entries: dict[str, dict[str, object]] = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        existing_entries = {entry["name"]: entry for entry in manifest["scenarios"]}

    manifest_entries: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        if scenario.name in selected_names:
            manifest_entries.append(_manifest_entry_for(scenario))
        elif scenario.name in existing_entries:
            manifest_entries.append(existing_entries[scenario.name])
        else:
            raise SystemExit(
                f"Cannot preserve unselected golden {scenario.name}; run --update --all first."
            )
    _write_manifest(manifest_entries)


def verify_goldens(scenarios: tuple[GoldenScenario, ...]) -> None:
    _assert_python_runtime()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("python_version") != REQUIRED_PYTHON_VERSION:
        raise SystemExit("Golden manifest runtime pin mismatch.")
    scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}
    selected_names = {scenario.name for scenario in scenarios}
    for entry in manifest["scenarios"]:
        if entry["name"] not in selected_names:
            continue
        scenario = scenarios_by_name[entry["name"]]
        payload_bytes = _payload_bytes(scenario)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(
                f"Golden hash mismatch for {scenario.name}: {digest} != {entry['sha256']}"
            )
        with gzip.open(GOLDEN_DIR / entry["fixture"], "rb") as handle:
            fixture_bytes = handle.read()
        if json.loads(payload_bytes) != json.loads(fixture_bytes):
            raise SystemExit(f"Golden payload structure mismatch for {scenario.name}.")
        print(f"verified {scenario.name} sha256={digest} bytes={len(payload_bytes)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or update deterministic replay goldens.")
    parser.add_argument("--update", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Allow --update to regenerate every golden fixture.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Limit verification or update to one scenario. Can be repeated.",
    )
    args = parser.parse_args()
    if args.all and args.scenario:
        raise SystemExit("--all cannot be combined with --scenario.")
    if args.update and not args.scenario and not args.all:
        raise SystemExit("--update without --scenario requires --all.")
    selected_scenarios = (
        tuple(scenario for scenario in SCENARIOS if scenario.name in set(args.scenario))
        if args.scenario
        else SCENARIOS
    )

    if args.update or not MANIFEST_PATH.exists():
        update_goldens(selected_scenarios)
    else:
        verify_goldens(selected_scenarios)


if __name__ == "__main__":
    main()
