from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import build_replay_payload, replay_payload_size_bytes
from evolution_sim.env.world import (
    ANIMAL_RESOURCE_KINDS,
    ANIMAL_RESOURCE_POLICY_BLOCKERS,
)
from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    validate_observation_input_payload,
)
from evolution_sim.env.runtime.mating import (
    ASEXUAL_REPRODUCTION_MODE,
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    REPRODUCTIVE_STAGE_ORDER,
    SEXUAL_EXPRESSION,
    SEXUAL_REPRODUCTION_MODE,
    STAGE2_PROTO_ROLES,
    STAGE3_X_Y_Z,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
    stage_rank,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTION_EVENT_SCHEMA_VERSION,
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
)
from evolution_sim.env.runtime.signals import SIGNAL_CONTRACT_VERSION
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_RECORD_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
)
from evolution_sim.genome.recombination import GENOME_RECOMBINATION_CONTRACT_VERSION

from .evaluate import (
    build_evaluation_report_from_runs,
    parse_seed_selection,
    run_evaluation,
)
from .golden_harness import GOLDEN_SPECIATION_SEED


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class FullReplayProbe:
    name: str
    seed: int
    ticks: int
    min_alive_species: int = 1
    min_species_created: int = 1
    require_speciation: bool = False
    max_replay_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class GateProfile:
    name: str
    summary_seeds: tuple[int, ...]
    summary_ticks: int
    min_alive_agents: int
    min_births: int
    min_last_birth_tick: int
    min_trophic_roles: int
    min_meat_modes: int
    min_hazardous_tiles: int
    min_ecology_pressure_tiles: int
    full_replay_probes: tuple[FullReplayProbe, ...]
    min_animal_energy_share: float = 0.0
    min_carrion_consumed_deposited_ratio: float = 0.0
    min_animal_resource_consumption_run_share_by_mode: float = 0.0
    use_late_window_population_floor: bool = False
    min_aggregate_meat_modes: int | None = None
    dominance_warning_share: float = 0.75
    required_terminal_meat_mode_alternatives_by_seed: Mapping[
        int,
        tuple[str, ...],
    ] | None = None


QUICK_PROFILE = GateProfile(
    name="quick",
    summary_seeds=(7, 8),
    summary_ticks=40,
    min_alive_agents=1,
    min_births=1,
    min_last_birth_tick=1,
    min_trophic_roles=2,
    min_meat_modes=2,
    min_hazardous_tiles=1,
    min_ecology_pressure_tiles=1,
    full_replay_probes=(
        FullReplayProbe(
            name="compact_species_surfaces",
            seed=7,
            ticks=40,
            min_alive_species=1,
            min_species_created=1,
            max_replay_size_bytes=30_000_000,
        ),
    ),
)

ECOLOGY_PROFILE = GateProfile(
    name="ecology",
    summary_seeds=tuple(range(1, 21)),
    summary_ticks=120,
    min_alive_agents=1,
    min_births=1,
    min_last_birth_tick=1,
    min_trophic_roles=2,
    min_meat_modes=1,
    min_hazardous_tiles=1,
    min_ecology_pressure_tiles=1,
    full_replay_probes=(),
    use_late_window_population_floor=True,
    min_aggregate_meat_modes=2,
    min_animal_resource_consumption_run_share_by_mode=0.5,
)

RELEASE_PROFILE = GateProfile(
    name="release",
    summary_seeds=(3, 7, 11, 17, 29),
    summary_ticks=800,
    min_alive_agents=1,
    min_births=WorldConfig().initial_agents + 1,
    min_last_birth_tick=320,
    min_trophic_roles=2,
    min_meat_modes=2,
    min_hazardous_tiles=1,
    min_ecology_pressure_tiles=1,
    min_animal_energy_share=0.01,
    min_carrion_consumed_deposited_ratio=0.01,
    min_animal_resource_consumption_run_share_by_mode=0.75,
    full_replay_probes=(
        FullReplayProbe(
            name="compact_species_surfaces",
            seed=7,
            ticks=120,
            min_alive_species=2,
            min_species_created=2,
            max_replay_size_bytes=75_000_000,
        ),
        FullReplayProbe(
            name="speciation_taxonomy_surfaces",
            seed=GOLDEN_SPECIATION_SEED,
            ticks=320,
            min_alive_species=1,
            min_species_created=2,
            require_speciation=True,
            max_replay_size_bytes=260_000_000,
        ),
    ),
    dominance_warning_share=0.85,
    required_terminal_meat_mode_alternatives_by_seed={
        3: ("hunter", "mixed"),
        11: ("hunter", "mixed"),
    },
)

PROFILES: dict[str, GateProfile] = {
    QUICK_PROFILE.name: QUICK_PROFILE,
    ECOLOGY_PROFILE.name: ECOLOGY_PROFILE,
    RELEASE_PROFILE.name: RELEASE_PROFILE,
}
REPRODUCTIVE_EXPRESSION_VALUES = {
    ASEXUAL_REPRODUCTION_MODE,
    SEXUAL_EXPRESSION,
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
}


def _flag(severity: str, scope: str, field: str, message: str) -> dict[str, object]:
    return {
        "severity": severity,
        "scope": scope,
        "field": field,
        "message": message,
    }


def _round_seconds(seconds: float) -> float:
    return round(seconds, 4)


def _normalized_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    return timeout_seconds


def _emit_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _write_gate_report(output_path: Path | None, report: dict[str, object]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _worker_result(
    *,
    process: mp.Process,
    result_queue: mp.Queue,
    timeout_seconds: float | None,
    timeout_message: str,
) -> dict[str, object]:
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "error": "TimeoutError", "message": timeout_message}
    try:
        result = result_queue.get(timeout=1.0)
    except queue.Empty:
        result = {
            "ok": False,
            "error": "RuntimeError",
            "message": "Scenario worker exited without returning a result.",
        }
    if process.exitcode not in (0, None) and bool(result.get("ok", False)):
        return {
            "ok": False,
            "error": "RuntimeError",
            "message": f"Scenario worker exited with code {process.exitcode}.",
        }
    return result


def _summary_seed_worker(
    seed: int,
    ticks: int,
    mode_value: str,
    result_queue: mp.Queue,
) -> None:
    try:
        result_queue.put(
            {
                "ok": True,
                "result": run_evaluation(seed=seed, ticks=ticks, mode=RunMode(mode_value)),
            }
        )
    except BaseException as exc:  # pragma: no cover - exercised through parent reports.
        result_queue.put(
            {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _full_replay_probe_worker(
    probe: FullReplayProbe,
    result_queue: mp.Queue,
) -> None:
    try:
        result_queue.put({"ok": True, "result": _run_full_replay_probe(probe)})
    except BaseException as exc:  # pragma: no cover - exercised through parent reports.
        result_queue.put(
            {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _positive_keys(
    counts: Mapping[str, object],
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = exclude or set()
    return sorted(
        key
        for key, value in counts.items()
        if key not in excluded and isinstance(value, int) and value > 0
    )


def _late_window_presence_counts(
    run: Mapping[str, object],
    field: str,
) -> Mapping[str, object] | None:
    lifecycle = run.get("trophic_lifecycle")
    if not isinstance(lifecycle, Mapping):
        return None
    late_window = lifecycle.get("late_window")
    if not isinstance(late_window, Mapping):
        return None
    counts = late_window.get(field)
    return counts if isinstance(counts, Mapping) else None


def _terminal_biological_blockers_by_live_meat_mode(
    reproduction_aggregate: Mapping[str, object],
) -> dict[str, object]:
    reproduction_by_mode = reproduction_aggregate.get("reproduction_by_meat_mode_at_end")
    blockers_by_mode = reproduction_aggregate.get(
        "reproduction_biological_blockers_by_meat_mode_at_end"
    )
    if not isinstance(reproduction_by_mode, Mapping) or not isinstance(
        blockers_by_mode,
        Mapping,
    ):
        return {}
    live_blockers: dict[str, object] = {}
    for mode, raw_readiness in reproduction_by_mode.items():
        if mode == "none" or not isinstance(raw_readiness, Mapping):
            continue
        totals = raw_readiness.get("total")
        if not isinstance(totals, Mapping) or int(totals.get("alive_agents", 0)) <= 0:
            continue
        raw_blockers = blockers_by_mode.get(mode)
        if isinstance(raw_blockers, Mapping):
            live_blockers[str(mode)] = raw_blockers
    return live_blockers


def _ecology_failure_rollup(
    evaluation: dict[str, object],
    profile: GateProfile,
) -> dict[str, object]:
    herbivore_only_seeds: list[int] = []
    role_shortfall_seeds: list[int] = []
    meat_mode_shortfall_seeds: list[int] = []
    no_terminal_animal_mode_seeds: list[int] = []
    no_animal_consumption_seeds: list[int] = []
    no_animal_consumption_seeds_by_meat_mode: dict[str, list[int]] = {}
    animal_resource_absent_seeds_by_meat_mode: dict[str, list[int]] = {}
    animal_resource_present_unconsumed_seeds_by_meat_mode: dict[str, list[int]] = {}
    animal_resource_present_unreachable_seeds_by_meat_mode: dict[str, list[int]] = {}
    animal_resource_reachable_unconsumed_seeds_by_meat_mode: dict[str, list[int]] = {}
    animal_resource_reachable_policy_blocked_seeds_by_meat_mode: dict[str, list[int]] = {}
    animal_resource_policy_blocker_seeds_by_meat_mode: dict[
        str,
        dict[str, dict[str, list[int]]],
    ] = {}
    animal_resource_opportunity_by_seed: dict[str, dict[str, dict[str, int | float]]] = {}
    terminal_roles_by_seed: dict[str, list[str]] = {}
    terminal_animal_modes_by_seed: dict[str, list[str]] = {}
    late_window_roles_by_seed: dict[str, dict[str, int]] = {}
    late_window_modes_by_seed: dict[str, dict[str, int]] = {}
    meat_mode_persistence_by_seed: dict[str, dict[str, object]] = {}
    terminal_role_presence_runs: dict[str, int] = {}
    terminal_meat_mode_presence_runs: dict[str, int] = {}
    aggregate_lifecycle = {}
    reproduction_aggregate = {}
    aggregate = evaluation.get("aggregate")
    if isinstance(aggregate, dict):
        lifecycle = aggregate.get("trophic_lifecycle")
        if isinstance(lifecycle, dict):
            aggregate_lifecycle = lifecycle
        reproduction_aggregate = {
            "diet_by_trophic_role_at_end": aggregate.get(
                "diet_by_trophic_role_at_end",
                {},
            ),
            "diet_by_meat_mode_at_end": aggregate.get(
                "diet_by_meat_mode_at_end",
                {},
            ),
            "animal_resource_opportunity_by_meat_mode_at_end": aggregate.get(
                "animal_resource_opportunity_by_meat_mode_at_end",
                {},
            ),
            "animal_resource_opportunity_run_counts_by_meat_mode": aggregate.get(
                "animal_resource_opportunity_run_counts_by_meat_mode",
                {},
            ),
            "reproduction_by_trophic_role_at_end": aggregate.get(
                "reproduction_by_trophic_role_at_end",
                {},
            ),
            "reproduction_by_meat_mode_at_end": aggregate.get(
                "reproduction_by_meat_mode_at_end",
                {},
            ),
            "reproduction_biological_blockers_at_end": aggregate.get(
                "reproduction_biological_blockers_at_end",
                {},
            ),
            "reproduction_biological_blockers_by_trophic_role_at_end": (
                aggregate.get(
                    "reproduction_biological_blockers_by_trophic_role_at_end",
                    {},
                )
            ),
            "reproduction_biological_blockers_by_meat_mode_at_end": (
                aggregate.get(
                    "reproduction_biological_blockers_by_meat_mode_at_end",
                    {},
                )
            ),
            "reproduction_energy_readiness_by_trophic_role_at_end": aggregate.get(
                "reproduction_energy_readiness_by_trophic_role_at_end",
                {},
            ),
            "reproduction_energy_readiness_by_meat_mode_at_end": aggregate.get(
                "reproduction_energy_readiness_by_meat_mode_at_end",
                {},
            ),
            "reproduction_blocked_run_counts_by_trophic_role": aggregate.get(
                "reproduction_blocked_run_counts_by_trophic_role",
                {},
            ),
            "reproduction_blocked_run_counts_by_meat_mode": aggregate.get(
                "reproduction_blocked_run_counts_by_meat_mode",
                {},
            ),
        }

    for run in evaluation["runs"]:
        seed = int(run["seed"])
        trophic = run["trophic"]
        role_counts = trophic["role_counts"]
        role_keys = _positive_keys(role_counts)
        animal_mode_counts = trophic["meat_mode_counts"]
        animal_modes = _positive_keys(animal_mode_counts, exclude={"none"})
        terminal_roles_by_seed[str(seed)] = role_keys
        terminal_animal_modes_by_seed[str(seed)] = animal_modes

        if role_keys == ["herbivore"]:
            herbivore_only_seeds.append(seed)
        if len(role_keys) < profile.min_trophic_roles:
            role_shortfall_seeds.append(seed)
        if len(animal_modes) < profile.min_meat_modes:
            meat_mode_shortfall_seeds.append(seed)
        if not animal_modes:
            no_terminal_animal_mode_seeds.append(seed)
        for role in role_counts:
            terminal_role_presence_runs.setdefault(role, 0)
            if int(role_counts[role]) > 0:
                terminal_role_presence_runs[role] += 1
        for mode in animal_mode_counts:
            terminal_meat_mode_presence_runs.setdefault(mode, 0)
            if int(animal_mode_counts[mode]) > 0:
                terminal_meat_mode_presence_runs[mode] += 1

        carrion_consumed = float(run["carrion"].get("energy_consumed", 0.0))
        fresh_kill_consumed = float(run["fresh_kill"].get("energy_consumed", 0.0))
        if carrion_consumed <= 0 and fresh_kill_consumed <= 0:
            no_animal_consumption_seeds.append(seed)

        opportunities = trophic.get("animal_resource_opportunity_by_meat_mode")
        if isinstance(opportunities, dict):
            animal_resource_opportunity_by_seed[str(seed)] = {}
            for mode, raw_counts in opportunities.items():
                if mode == "none":
                    continue
                if not isinstance(raw_counts, dict):
                    continue
                animal_resource_opportunity_by_seed[str(seed)][str(mode)] = {
                    str(key): value
                    for key, value in raw_counts.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                alive_ticks = int(raw_counts.get("alive_ticks", 0))
                if alive_ticks <= 0:
                    continue
                mode_key = str(mode)
                consumption_events = int(
                    raw_counts.get("animal_resource_consumption_events", 0)
                )
                present_ticks = int(raw_counts.get("animal_resource_present_ticks", 0))
                reachable_ticks = int(
                    raw_counts.get("animal_resource_reachable_ticks", 0)
                )
                reachable_policy_blocked_ticks = int(
                    raw_counts.get(
                        "animal_resource_reachable_policy_blocked_ticks",
                        0,
                    )
                )
                if consumption_events <= 0:
                    no_animal_consumption_seeds_by_meat_mode.setdefault(
                        mode_key,
                        [],
                    ).append(seed)
                if present_ticks <= 0:
                    animal_resource_absent_seeds_by_meat_mode.setdefault(
                        mode_key,
                        [],
                    ).append(seed)
                if present_ticks > 0 and consumption_events <= 0:
                    animal_resource_present_unconsumed_seeds_by_meat_mode.setdefault(
                        mode_key,
                        [],
                    ).append(seed)
                    if reachable_ticks <= 0:
                        animal_resource_present_unreachable_seeds_by_meat_mode.setdefault(
                            mode_key,
                            [],
                        ).append(seed)
                    else:
                        animal_resource_reachable_unconsumed_seeds_by_meat_mode.setdefault(
                            mode_key,
                            [],
                        ).append(seed)
                if reachable_policy_blocked_ticks > 0:
                    animal_resource_reachable_policy_blocked_seeds_by_meat_mode.setdefault(
                        mode_key,
                        [],
                    ).append(seed)
                for resource in ("animal_resource", *ANIMAL_RESOURCE_KINDS):
                    for blocker in ANIMAL_RESOURCE_POLICY_BLOCKERS:
                        blocked_ticks = int(
                            raw_counts.get(
                                f"{resource}_policy_blocked_by_{blocker}_ticks",
                                0,
                            )
                        )
                        if blocked_ticks <= 0:
                            continue
                        animal_resource_policy_blocker_seeds_by_meat_mode.setdefault(
                            mode_key,
                            {},
                        ).setdefault(resource, {}).setdefault(blocker, []).append(seed)

        lifecycle = run.get("trophic_lifecycle")
        if isinstance(lifecycle, dict):
            persistence = lifecycle.get("meat_mode_persistence")
            if isinstance(persistence, dict):
                last_alive = persistence.get("last_alive_tick_by_meat_mode")
                deaths_by_band = persistence.get("deaths_by_meat_mode_by_tick_band")
                death_causes_by_band = persistence.get(
                    "death_causes_by_meat_mode_by_tick_band"
                )
                parent_births_by_band = persistence.get(
                    "births_by_parent_meat_mode_by_tick_band"
                )
                child_births_by_band = persistence.get(
                    "births_by_child_meat_mode_by_tick_band"
                )
                meat_mode_persistence_by_seed[str(seed)] = {
                    "last_alive_tick_by_meat_mode": (
                        {
                            str(mode): tick
                            for mode, tick in last_alive.items()
                            if mode != "none"
                        }
                        if isinstance(last_alive, dict)
                        else {}
                    ),
                    "deaths_by_meat_mode_by_tick_band": (
                        deaths_by_band if isinstance(deaths_by_band, dict) else {}
                    ),
                    "death_causes_by_meat_mode_by_tick_band": (
                        death_causes_by_band
                        if isinstance(death_causes_by_band, dict)
                        else {}
                    ),
                    "births_by_parent_meat_mode_by_tick_band": (
                        parent_births_by_band
                        if isinstance(parent_births_by_band, dict)
                        else {}
                    ),
                    "births_by_child_meat_mode_by_tick_band": (
                        child_births_by_band
                        if isinstance(child_births_by_band, dict)
                        else {}
                    ),
                }
            late_window = lifecycle.get("late_window")
            if isinstance(late_window, dict):
                role_presence = late_window.get("presence_ticks_by_trophic_role")
                if isinstance(role_presence, dict):
                    late_window_roles_by_seed[str(seed)] = {
                        str(role): int(count)
                        for role, count in role_presence.items()
                    }
                mode_presence = late_window.get("presence_ticks_by_meat_mode")
                if isinstance(mode_presence, dict):
                    late_window_modes_by_seed[str(seed)] = {
                        str(mode): int(count)
                        for mode, count in mode_presence.items()
                    }

    return {
        "run_count": len(evaluation["runs"]),
        "role_shortfall_seeds": role_shortfall_seeds,
        "meat_mode_shortfall_seeds": meat_mode_shortfall_seeds,
        "herbivore_only_seeds": herbivore_only_seeds,
        "no_terminal_animal_mode_seeds": no_terminal_animal_mode_seeds,
        "no_animal_consumption_seeds": no_animal_consumption_seeds,
        "no_animal_consumption_seeds_by_meat_mode": {
            key: no_animal_consumption_seeds_by_meat_mode[key]
            for key in sorted(no_animal_consumption_seeds_by_meat_mode)
        },
        "animal_resource_absent_seeds_by_meat_mode": {
            key: animal_resource_absent_seeds_by_meat_mode[key]
            for key in sorted(animal_resource_absent_seeds_by_meat_mode)
        },
        "animal_resource_present_unconsumed_seeds_by_meat_mode": {
            key: animal_resource_present_unconsumed_seeds_by_meat_mode[key]
            for key in sorted(animal_resource_present_unconsumed_seeds_by_meat_mode)
        },
        "animal_resource_present_unreachable_seeds_by_meat_mode": {
            key: animal_resource_present_unreachable_seeds_by_meat_mode[key]
            for key in sorted(animal_resource_present_unreachable_seeds_by_meat_mode)
        },
        "animal_resource_reachable_unconsumed_seeds_by_meat_mode": {
            key: animal_resource_reachable_unconsumed_seeds_by_meat_mode[key]
            for key in sorted(animal_resource_reachable_unconsumed_seeds_by_meat_mode)
        },
        "animal_resource_reachable_policy_blocked_seeds_by_meat_mode": {
            key: animal_resource_reachable_policy_blocked_seeds_by_meat_mode[key]
            for key in sorted(
                animal_resource_reachable_policy_blocked_seeds_by_meat_mode
            )
        },
        "animal_resource_policy_blocker_seeds_by_meat_mode": {
            mode: {
                resource: {
                    blocker: blockers[blocker]
                    for blocker in sorted(blockers)
                }
                for resource, blockers in sorted(resources.items())
            }
            for mode, resources in sorted(
                animal_resource_policy_blocker_seeds_by_meat_mode.items()
            )
        },
        "animal_resource_opportunity_by_seed": animal_resource_opportunity_by_seed,
        "terminal_role_presence_runs": {
            key: terminal_role_presence_runs[key]
            for key in sorted(terminal_role_presence_runs)
        },
        "terminal_meat_mode_presence_runs": {
            key: terminal_meat_mode_presence_runs[key]
            for key in sorted(terminal_meat_mode_presence_runs)
        },
        "terminal_roles_by_seed": terminal_roles_by_seed,
        "terminal_animal_modes_by_seed": terminal_animal_modes_by_seed,
        "late_window_roles_by_seed": late_window_roles_by_seed,
        "late_window_modes_by_seed": late_window_modes_by_seed,
        "meat_mode_persistence_by_seed": meat_mode_persistence_by_seed,
        "death_causes": aggregate_lifecycle.get("death_causes", {}),
        "death_causes_by_trophic_role": aggregate_lifecycle.get(
            "death_causes_by_trophic_role",
            {},
        ),
        "death_causes_by_meat_mode": aggregate_lifecycle.get(
            "death_causes_by_meat_mode",
            {},
        ),
        **reproduction_aggregate,
        "meat_mode_persistence": aggregate_lifecycle.get(
            "meat_mode_persistence",
            {},
        ),
        "terminal_biological_blockers_by_live_meat_mode": (
            _terminal_biological_blockers_by_live_meat_mode(reproduction_aggregate)
        ),
    }


def _summary_gate_flags(
    evaluation: dict[str, object],
    profile: GateProfile,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for evaluator_flag in evaluation["flags"]:
        severity = str(evaluator_flag["severity"])
        flags.append(
            _flag(
                severity,
                f"summary_seed_{evaluator_flag['seed']}",
                str(evaluator_flag["field"]),
                str(evaluator_flag["message"]),
            )
        )

    for run in evaluation["runs"]:
        seed = int(run["seed"])
        flags.extend(
            _reproductive_role_readiness_flags(
                scope=f"summary_seed_{seed}",
                reproduction=run.get("reproduction"),
            )
        )
        if run["last_birth_tick"] is None or int(run["last_birth_tick"]) < profile.min_last_birth_tick:
            flags.append(
                _flag(
                    "error",
                    f"summary_seed_{seed}",
                    "last_birth_tick",
                    (
                        "Run did not reproduce late enough for this gate "
                        f"(minimum tick {profile.min_last_birth_tick})."
                    ),
                )
            )
        trophic = run["trophic"]
        role_counts = trophic["role_counts"]
        role_floor_counts = (
            _late_window_presence_counts(
                run,
                "presence_ticks_by_trophic_role",
            )
            if profile.use_late_window_population_floor
            else None
        ) or role_counts
        run_trophic_roles = _positive_keys(role_floor_counts)
        if len(run_trophic_roles) < profile.min_trophic_roles:
            role_field = (
                "trophic_lifecycle.late_window.presence_ticks_by_trophic_role"
                if profile.use_late_window_population_floor
                else "trophic.role_counts"
            )
            flags.append(
                _flag(
                    "error",
                    f"summary_seed_{seed}",
                    role_field,
                    (
                        "Run stayed below the per-run trophic-role floor of "
                        f"{profile.min_trophic_roles}: {run_trophic_roles}."
                    ),
                )
            )
        meat_mode_counts = trophic["meat_mode_counts"]
        mode_floor_counts = (
            _late_window_presence_counts(
                run,
                "presence_ticks_by_meat_mode",
            )
            if profile.use_late_window_population_floor
            else None
        ) or meat_mode_counts
        run_meat_modes = _positive_keys(mode_floor_counts, exclude={"none"})
        animal_mode_floor = profile.min_meat_modes
        if len(run_meat_modes) < animal_mode_floor:
            mode_field = (
                "trophic_lifecycle.late_window.presence_ticks_by_meat_mode"
                if profile.use_late_window_population_floor
                else "trophic.meat_mode_counts"
            )
            flags.append(
                _flag(
                    "error",
                    f"summary_seed_{seed}",
                    mode_field,
                    (
                        "Run stayed below the per-run animal-resource strategy floor "
                        f"of {animal_mode_floor}: {run_meat_modes}."
                    ),
                )
            )
        terminal_alternatives = (
            profile.required_terminal_meat_mode_alternatives_by_seed or {}
        ).get(seed)
        if terminal_alternatives:
            lifecycle = run.get("trophic_lifecycle")
            persistence = (
                lifecycle.get("meat_mode_persistence")
                if isinstance(lifecycle, Mapping)
                else None
            )
            last_alive = (
                persistence.get("last_alive_tick_by_meat_mode")
                if isinstance(persistence, Mapping)
                else None
            )
            terminal_tick = max(
                0,
                int(run.get("ticks_executed", profile.summary_ticks)) - 1,
            )
            observed_last_alive: dict[str, int | None] = {}
            terminal_modes: list[str] = []
            if isinstance(last_alive, Mapping):
                for mode in terminal_alternatives:
                    tick = _as_optional_int(last_alive.get(mode))
                    observed_last_alive[mode] = tick
                    if tick is not None and tick >= terminal_tick:
                        terminal_modes.append(mode)
            if not terminal_modes:
                flags.append(
                    _flag(
                        "error",
                        f"summary_seed_{seed}",
                        (
                            "trophic_lifecycle.meat_mode_persistence."
                            "last_alive_tick_by_meat_mode"
                        ),
                        (
                            "Required at least one terminal hunter/mixed timeline "
                            f"for seed {seed}; observed {observed_last_alive}."
                        ),
                    )
                )
        if profile.min_animal_energy_share > 0:
            animal_share = float(trophic["diet"].get("animal_energy_share", 0.0))
            if animal_share < profile.min_animal_energy_share:
                flags.append(
                    _flag(
                        "error",
                        f"summary_seed_{seed}",
                        "trophic.diet.animal_energy_share",
                        (
                            "Run ended below the animal-energy share floor of "
                            f"{profile.min_animal_energy_share:.4f}: {animal_share:.4f}."
                        ),
                    )
                )
        if profile.min_carrion_consumed_deposited_ratio > 0:
            carrion = run["carrion"]
            deposited = float(carrion.get("energy_deposited", 0.0))
            consumed = float(carrion.get("energy_consumed", 0.0))
            if deposited <= 0:
                flags.append(
                    _flag(
                        "error",
                        f"summary_seed_{seed}",
                        "carrion.energy_deposited",
                        "Run did not create any carrion pressure to test.",
                    )
                )
            else:
                consumed_ratio = consumed / deposited
                if consumed_ratio < profile.min_carrion_consumed_deposited_ratio:
                    flags.append(
                        _flag(
                            "error",
                            f"summary_seed_{seed}",
                            "carrion.energy_consumed_ratio",
                            (
                                "Run ended below the carrion consumption/deposition "
                                "floor of "
                                f"{profile.min_carrion_consumed_deposited_ratio:.4f}: "
                                f"{consumed_ratio:.4f}."
                            ),
                        )
                    )

    aggregate = evaluation["aggregate"]
    hazardous_min_raw = aggregate["hazardous_tiles"]["min"]
    hazardous_min = int(hazardous_min_raw) if hazardous_min_raw is not None else 0
    if hazardous_min < profile.min_hazardous_tiles:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "hazardous_tiles",
                "At least one run ended without visible hazard pressure.",
            )
        )

    aggregate_lifecycle = aggregate.get("trophic_lifecycle", {})
    trophic_counts = (
        aggregate_lifecycle.get("late_window_trophic_role_presence_runs", {})
        if profile.use_late_window_population_floor
        and isinstance(aggregate_lifecycle, Mapping)
        else aggregate["trophic_role_counts_at_end"]["total"]
    )
    trophic_roles = _positive_keys(trophic_counts)
    if len(trophic_roles) < profile.min_trophic_roles:
        trophic_field = (
            "trophic_lifecycle.late_window_trophic_role_presence_runs"
            if profile.use_late_window_population_floor
            else "trophic_role_counts_at_end"
        )
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                trophic_field,
                (
                    "Summary populations do not yet show enough trophic-role "
                    f"diversity: {trophic_roles}."
                ),
            )
        )

    meat_mode_counts = (
        aggregate_lifecycle.get("late_window_meat_mode_presence_runs", {})
        if profile.use_late_window_population_floor
        and isinstance(aggregate_lifecycle, Mapping)
        else aggregate["meat_mode_counts_at_end"]["total"]
    )
    meat_modes = _positive_keys(meat_mode_counts, exclude={"none"})
    animal_mode_floor = profile.min_aggregate_meat_modes or profile.min_meat_modes
    if len(meat_modes) < animal_mode_floor:
        meat_mode_field = (
            "trophic_lifecycle.late_window_meat_mode_presence_runs"
            if profile.use_late_window_population_floor
            else "meat_mode_counts_at_end"
        )
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                meat_mode_field,
                (
                    "Summary populations do not yet show enough animal-resource "
                    f"strategy diversity: {meat_modes}."
                ),
            )
        )

    ecology_counts = aggregate["ecology_state_counts_at_end"]["total"]
    pressure_tiles = int(ecology_counts.get("recovering", 0)) + int(
        ecology_counts.get("depleted", 0)
    )
    if pressure_tiles < profile.min_ecology_pressure_tiles:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "ecology_state_counts_at_end",
                "Ending ecology lacks visible recovery or depletion pressure.",
            )
        )

    carrion_consumed_raw = aggregate["carrion_energy_consumed"]["max"]
    fresh_kill_consumed_raw = aggregate["fresh_kill_energy_consumed"]["max"]
    carrion_consumed_max = (
        float(carrion_consumed_raw) if carrion_consumed_raw is not None else 0.0
    )
    fresh_kill_consumed_max = (
        float(fresh_kill_consumed_raw) if fresh_kill_consumed_raw is not None else 0.0
    )
    if carrion_consumed_max <= 0 and fresh_kill_consumed_max <= 0:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "animal_resource_consumption",
                "No animal-resource consumption was observed across the sweep.",
            )
        )
    if profile.min_animal_resource_consumption_run_share_by_mode > 0:
        opportunity_counts = aggregate.get(
            "animal_resource_opportunity_run_counts_by_meat_mode",
            {},
        )
        if isinstance(opportunity_counts, Mapping):
            for mode in ("hunter", "mixed", "scavenger"):
                mode_counts = opportunity_counts.get(mode)
                if not isinstance(mode_counts, Mapping):
                    continue
                alive_runs = int(mode_counts.get("alive_runs", 0))
                if alive_runs <= 0:
                    continue
                no_consumption_runs = int(
                    mode_counts.get("no_animal_consumption_runs", 0)
                )
                consuming_runs = alive_runs - no_consumption_runs
                consuming_run_share = consuming_runs / max(alive_runs, 1)
                floor = profile.min_animal_resource_consumption_run_share_by_mode
                if consuming_run_share < floor:
                    flags.append(
                        _flag(
                            "error",
                            "summary_sweep",
                            (
                                "animal_resource_opportunity_run_counts_by_meat_mode."
                                f"{mode}.consuming_run_share"
                            ),
                            (
                                f"{mode} consuming-run share is below the floor of "
                                f"{floor:.4f}: {consuming_run_share:.4f} "
                                f"({consuming_runs}/{alive_runs})."
                            ),
                        )
                    )
    return flags


def _replay_size_bytes(result_payload: dict[str, object]) -> int:
    return replay_payload_size_bytes(result_payload)


def _as_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_int_count_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        parsed_count = _as_optional_int(raw_count)
        if parsed_count is not None:
            counts[str(key)] = parsed_count
    return counts


def _reproductive_role_readiness_flags(
    *,
    scope: str,
    reproduction: object,
) -> list[dict[str, object]]:
    if not isinstance(reproduction, Mapping):
        return []
    stage_counts = _as_int_count_mapping(reproduction.get("reproductive_stage_counts"))
    expression_counts = _as_int_count_mapping(
        reproduction.get("reproductive_expression_counts")
    )
    capability_counts = _as_int_count_mapping(
        reproduction.get("reproductive_capability_counts")
    )
    role_stage_total = stage_counts.get(STAGE2_PROTO_ROLES, 0) + stage_counts.get(
        STAGE3_X_Y_Z,
        0,
    )
    if (
        role_stage_total <= 0
        and capability_counts.get("proto_role_differentiation", 0) <= 0
        and capability_counts.get("xyz_expression", 0) <= 0
    ):
        return []

    flags: list[dict[str, object]] = []
    proto_x_count = expression_counts.get(PROTO_X_EXPRESSION, 0)
    proto_y_count = expression_counts.get(PROTO_Y_EXPRESSION, 0)
    proto_z_count = expression_counts.get(PROTO_Z_EXPRESSION, 0)
    x_count = expression_counts.get(X_EXPRESSION, 0)
    y_count = expression_counts.get(Y_EXPRESSION, 0)
    z_count = expression_counts.get(Z_EXPRESSION, 0)
    has_x_like = proto_x_count + x_count > 0
    has_y_like = proto_y_count + y_count > 0
    plastic_count = proto_z_count + z_count
    if role_stage_total > 0 and not ((has_x_like and has_y_like) or plastic_count > 0):
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.reproductive_expression_counts",
                (
                    "Role-stage reproduction is present without complementary "
                    "X/Y-like or Z-plastic expression coverage."
                ),
            )
        )
    if stage_counts.get(STAGE3_X_Y_Z, 0) > 0 and z_count <= 0:
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.reproductive_expression_counts.z_plastic",
                "Stage 3 X/Y/Z reproduction is present without alive Z-plastic expression.",
            )
        )

    ready_stage_counts = _as_int_count_mapping(
        reproduction.get("ready_by_reproductive_stage")
    )
    ready_role_stage_total = ready_stage_counts.get(
        STAGE2_PROTO_ROLES,
        0,
    ) + ready_stage_counts.get(STAGE3_X_Y_Z, 0)
    if role_stage_total > 0 and ready_role_stage_total <= 0:
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.ready_by_reproductive_stage",
                "Role-stage agents are alive but none are currently reproduction-ready.",
            )
        )

    mate_search_counts = _as_int_count_mapping(reproduction.get("mate_search_run_counts"))
    sexual_searches = mate_search_counts.get("sexual_searches", 0)
    sexual_successes = mate_search_counts.get("sexual_successes", 0)
    if sexual_searches > 0 and sexual_successes <= 0:
        fallback_counts = {
            key: mate_search_counts.get(key, 0)
            for key in (
                "fallback_expression_incompatible",
                "fallback_no_compatible_partner",
                "fallback_no_same_group_partner",
            )
        }
        if any(count > 0 for count in fallback_counts.values()):
            flags.append(
                _flag(
                    "warning",
                    scope,
                    "reproduction.mate_search_run_counts",
                    (
                        "Role-stage sexual searches occurred without successes; "
                        f"fallbacks={fallback_counts}."
                    ),
                )
            )
    if mate_search_counts.get("constraint_expression_incompatible", 0) > 0:
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.mate_search_run_counts.constraint_expression_incompatible",
                "Mate-search diagnostics observed role expression incompatibility.",
            )
        )
    return flags


def _mind_contract_flags(
    *,
    scope: str,
    summary: dict[str, object],
    viewer: dict[str, object],
    events: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    contracts = summary.get("mind_contracts")
    if not isinstance(contracts, dict):
        flags.append(
            _flag(
                "error",
                scope,
                "summary.mind_contracts",
                "Full replay summary is missing Mind contract metadata.",
            )
        )
    else:
        expected_versions = {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "policy_interface_version": POLICY_INTERFACE_VERSION,
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
            "genome_recombination_contract_version": (
                GENOME_RECOMBINATION_CONTRACT_VERSION
            ),
        }
        for field, expected in expected_versions.items():
            if contracts.get(field) != expected:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        f"summary.mind_contracts.{field}",
                        f"Expected {expected}, found {contracts.get(field)!r}.",
                    )
                )
        if int(contracts.get("record_count", 0)) <= 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "summary.mind_contracts.record_count",
                    "Full replay did not record any per-agent trajectory rows.",
                )
            )

    trajectory = viewer.get("trajectory")
    if not isinstance(trajectory, dict):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory",
                "Full replay viewer is missing trajectory payload.",
            )
        )
        return flags
    expected_trajectory_versions = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "genome_recombination_contract_version": GENOME_RECOMBINATION_CONTRACT_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
    }
    for field, expected in expected_trajectory_versions.items():
        if trajectory.get(field) != expected:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"viewer.trajectory.{field}",
                    f"Expected {expected}, found {trajectory.get(field)!r}.",
                )
            )
    action_contract = trajectory.get("action_contract")
    if (
        not isinstance(action_contract, dict)
        or action_contract.get("schema_version") != ACTION_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.action_contract.schema_version",
                "Action contract metadata is missing or stale.",
            )
        )
    reproductive_group_contract = trajectory.get("reproductive_group_contract")
    if (
        not isinstance(reproductive_group_contract, dict)
        or reproductive_group_contract.get("schema_version")
        != REPRODUCTIVE_GROUP_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.reproductive_group_contract.schema_version",
                "Reproductive group contract metadata is missing or stale.",
            )
        )
    genome_recombination_contract = trajectory.get("genome_recombination_contract")
    if (
        not isinstance(genome_recombination_contract, dict)
        or genome_recombination_contract.get("schema_version")
        != GENOME_RECOMBINATION_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.genome_recombination_contract.schema_version",
                "Genome recombination contract metadata is missing or stale.",
            )
        )
    reproductive_group_catalog = viewer.get("reproductive_group_catalog")
    if (
        not isinstance(reproductive_group_catalog, dict)
        or reproductive_group_catalog.get("schema_version")
        != REPRODUCTIVE_GROUP_CONTRACT_VERSION
        or not isinstance(reproductive_group_catalog.get("groups"), dict)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.reproductive_group_catalog",
                "Viewer reproductive group catalog is missing or stale.",
            )
        )
    else:
        flags.extend(
            _reproductive_group_catalog_flags(
                scope=scope,
                summary=summary,
                viewer=viewer,
                reproductive_group_catalog=reproductive_group_catalog,
                events=events,
            )
        )
    observation_contract = trajectory.get("observation_contract")
    if (
        not isinstance(observation_contract, dict)
        or observation_contract.get("schema_version") != OBSERVATION_SCHEMA_VERSION
        or observation_contract.get("privileged_world_state") is not False
        or observation_contract.get("metadata_policy_excluded") is not True
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract",
                "Observation contract metadata is missing or permits privileged world state.",
            )
        )
    else:
        policy_input = observation_contract.get("policy_input")
        if (
            not isinstance(policy_input, dict)
            or policy_input.get("encoder_version") != OBSERVATION_ENCODER_VERSION
            or policy_input.get("shape") != [OBSERVATION_INPUT_VECTOR_SIZE]
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.policy_input",
                    "Observation contract does not declare the canonical policy input encoder.",
                )
            )
        embedded_action_contract = observation_contract.get("action_contract")
        if (
            not isinstance(embedded_action_contract, dict)
            or embedded_action_contract.get("schema_version")
            != ACTION_CONTRACT_VERSION
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.action_contract.schema_version",
                    "Observation contract action metadata is missing or stale.",
                )
            )
        signal_contract = observation_contract.get("signal_contract")
        signal_metadata_fields = (
            signal_contract.get("emission_debug_metadata_fields")
            if isinstance(signal_contract, dict)
            else None
        )
        required_signal_metadata_fields = {
            "profile_id",
            "source_agent_id",
            "token_id",
            "profile_index",
            "radius",
            "decay_rate",
            "energy_cost",
        }
        if (
            not isinstance(signal_contract, dict)
            or signal_contract.get("schema_version") != SIGNAL_CONTRACT_VERSION
            or signal_contract.get("profile_metadata_policy_visible") is not False
            or not isinstance(signal_metadata_fields, list)
            or not required_signal_metadata_fields.issubset(signal_metadata_fields)
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.signal_contract",
                    "Signal contract does not declare debug-only profile/provenance metadata.",
                )
            )
        elif isinstance(observation_contract, dict):
            flags.extend(
                _signal_action_capacity_flags(
                    scope=scope,
                    trajectory=trajectory,
                    observation_contract=observation_contract,
                    signal_contract=signal_contract,
                )
            )
    frames = viewer.get("frames")
    last_frame = frames[-1] if isinstance(frames, list) and frames else None
    signal_emissions = (
        last_frame.get("signal_emissions")
        if isinstance(last_frame, dict)
        else None
    )
    if (
        not isinstance(signal_emissions, dict)
        or signal_emissions.get("schema_version") != SIGNAL_CONTRACT_VERSION
        or signal_emissions.get("policy_visible") is not False
        or not isinstance(signal_emissions.get("events"), list)
        or not isinstance(signal_emissions.get("active_counts"), dict)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.frames.signal_emissions",
                "Full replay frames are missing versioned signal emission debug metadata.",
            )
        )
    records = trajectory.get("records")
    reward_contract = trajectory.get("reward_contract")
    if (
        not isinstance(reward_contract, dict)
        or reward_contract.get("schema_version") != REWARD_SCHEMA_VERSION
        or not isinstance(reward_contract.get("component_bounds"), dict)
        or not isinstance(reward_contract.get("total_bounds"), list)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.reward_contract",
                "Trajectory payload does not declare versioned reward component bounds.",
            )
        )
    if not isinstance(records, list) or not records:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records",
                "Trajectory payload has no decision records.",
            )
        )
        return flags
    for index, record in enumerate(records):
        flags.extend(
            _trajectory_record_contract_flags(
                scope=scope,
                record=record,
                index=index,
            )
        )
    return flags


def _trajectory_record_contract_flags(
    *,
    scope: str,
    record: object,
    index: int,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    required_record_fields = set(TRAJECTORY_RECORD_FIELDS)
    if not isinstance(record, dict) or not required_record_fields.issubset(record):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records",
                (
                    f"Trajectory record {index} does not include the canonical "
                    "contract fields."
                ),
            )
        )
        return flags

    if record.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.observation_schema",
                (
                    f"Trajectory record {index} does not declare the current "
                    "observation schema."
                ),
            )
        )
    observation_metadata = record.get("observation_metadata")
    if (
        not isinstance(observation_metadata, dict)
        or observation_metadata.get("agent_id") != record.get("agent_id")
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.observation_metadata",
                (
                    f"Trajectory record {index} does not include policy-excluded "
                    "observation metadata."
                ),
            )
        )
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.observation_input",
                f"Trajectory record {index} does not include encoded observation inputs.",
            )
        )
    else:
        validation_errors = validate_observation_input_payload(observation_input)
        if validation_errors:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.records.observation_input",
                    f"Trajectory record {index}: {validation_errors[0]}",
                )
            )
    reward = record.get("reward")
    if (
        not isinstance(reward, dict)
        or reward.get("schema_version") != REWARD_SCHEMA_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.reward",
                f"Trajectory record {index} does not include versioned reward components.",
            )
        )
    outcome = record.get("outcome")
    if (
        not isinstance(outcome, dict)
        or outcome.get("schema_version") != ACTION_OUTCOME_SCHEMA_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.outcome",
                f"Trajectory record {index} does not include a versioned action outcome.",
            )
        )
    if isinstance(outcome, dict):
        signal_outcome = outcome.get("signal")
        required_signal_outcome_fields = {
            "emitted",
            "token_id",
            "profile_index",
            "intensity",
            "radius",
            "duration_ticks",
            "decay_rate",
            "energy_cost",
            "invalid_reason",
        }
        if (
            not isinstance(signal_outcome, dict)
            or not isinstance(signal_outcome.get("emitted"), bool)
            or not required_signal_outcome_fields.issubset(signal_outcome)
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.records.outcome.signal",
                    (
                        f"Trajectory record {index} action outcome must include "
                        "stable signal outcome metadata."
                    ),
                )
            )
    return flags


def _reproductive_group_catalog_flags(
    *,
    scope: str,
    summary: dict[str, object],
    viewer: dict[str, object],
    reproductive_group_catalog: dict[str, object],
    events: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    groups = reproductive_group_catalog.get("groups")
    agent_catalog = viewer.get("agent_catalog")
    if not isinstance(groups, dict):
        return [
            _flag(
                "error",
                scope,
                "viewer.reproductive_group_catalog.groups",
                "Viewer reproductive group catalog does not declare groups.",
            )
        ]
    if not isinstance(agent_catalog, dict):
        return [
            _flag(
                "error",
                scope,
                "viewer.agent_catalog",
                "Viewer agent catalog is missing while validating reproductive groups.",
            )
        ]

    event_birth_counts, event_flags = _reproductive_group_birth_counts_from_events(
        scope=scope,
        events=events,
        groups=groups,
    )
    flags.extend(event_flags)

    member_counts: dict[str, int] = {}
    alive_member_counts: dict[str, int] = {}
    alive_stage_counts: dict[str, dict[str, int]] = {}
    alive_expression_counts: dict[str, dict[str, int]] = {}
    for raw_agent_id, agent_payload in agent_catalog.items():
        if not isinstance(agent_payload, dict):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog",
                    f"Agent catalog entry {raw_agent_id} is not an object.",
                )
            )
            continue
        group_id = agent_payload.get("reproductive_group_id")
        reproductive_stage = agent_payload.get("reproductive_stage")
        reproductive_expression = agent_payload.get("reproductive_expression")
        if "death_tick" not in agent_payload:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.death_tick",
                    f"Agent catalog entry {raw_agent_id} is missing death_tick.",
                )
            )
        elif (
            agent_payload.get("death_tick") is not None
            and (
                _as_optional_int(agent_payload.get("death_tick")) is None
                or int(agent_payload["death_tick"]) < 0
            )
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.death_tick",
                    (
                        f"Agent catalog entry {raw_agent_id} death_tick must "
                        "be a nonnegative integer or null."
                    ),
                )
            )
        if reproductive_stage not in REPRODUCTIVE_STAGE_ORDER:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_stage",
                    (
                        f"Agent catalog entry {raw_agent_id} has unknown "
                        "reproductive stage."
                    ),
                )
            )
        if reproductive_expression not in REPRODUCTIVE_EXPRESSION_VALUES:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_expression",
                    (
                        f"Agent catalog entry {raw_agent_id} has unknown "
                        "reproductive expression."
                    ),
                )
            )
        if group_id is None:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_group_id",
                    f"Agent catalog entry {raw_agent_id} is missing a reproductive group.",
                )
            )
            continue
        parsed_group_id = _as_optional_int(group_id)
        if parsed_group_id is None or parsed_group_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_group_id",
                    (
                        f"Agent catalog entry {raw_agent_id} reproductive_group_id "
                        "must be a nonnegative integer."
                    ),
                )
            )
            continue
        group_key = str(group_id)
        if group_key not in groups:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_group_id",
                    (
                        f"Agent catalog entry {raw_agent_id} references missing "
                        f"reproductive group {group_key}."
                    ),
                )
            )
            continue
        member_counts[group_key] = member_counts.get(group_key, 0) + 1
        if agent_payload.get("death_tick") is None:
            alive_member_counts[group_key] = alive_member_counts.get(group_key, 0) + 1
            _increment_nested_count(
                alive_stage_counts,
                group_key,
                str(reproductive_stage),
            )
            _increment_nested_count(
                alive_expression_counts,
                group_key,
                str(reproductive_expression),
            )

    for group_key, group_payload in groups.items():
        if not isinstance(group_payload, dict):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups",
                    f"Reproductive group {group_key} is not an object.",
                )
            )
            continue
        parsed_payload_group_id = _as_optional_int(group_payload.get("group_id"))
        if parsed_payload_group_id is None or parsed_payload_group_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.group_id",
                    f"Reproductive group {group_key} group_id must be a nonnegative integer.",
                )
            )
        elif str(parsed_payload_group_id) != str(group_key):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.group_id",
                    f"Reproductive group {group_key} group_id does not match its key.",
                )
            )
        expected_member_count = member_counts.get(str(group_key), 0)
        expected_alive_member_count = alive_member_counts.get(str(group_key), 0)
        member_count = _as_optional_int(group_payload.get("member_count"))
        alive_member_count = _as_optional_int(group_payload.get("alive_member_count"))
        actual_stage_counts = _as_count_dict(group_payload.get("alive_stage_counts"))
        actual_expression_counts = _as_count_dict(
            group_payload.get("alive_expression_counts")
        )
        if member_count != expected_member_count:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.member_count",
                    (
                        f"Reproductive group {group_key} member_count does not "
                        "match the agent catalog."
                    ),
                )
            )
        if alive_member_count != expected_alive_member_count:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.alive_member_count",
                    (
                        f"Reproductive group {group_key} alive_member_count does "
                        "not match the agent catalog."
                    ),
                )
            )
        expected_stage_counts = alive_stage_counts.get(str(group_key), {})
        expected_expression_counts = alive_expression_counts.get(str(group_key), {})
        if actual_stage_counts != expected_stage_counts:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.alive_stage_counts",
                    (
                        f"Reproductive group {group_key} alive_stage_counts does "
                        "not match the agent catalog."
                    ),
                )
            )
        if actual_expression_counts != expected_expression_counts:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.alive_expression_counts",
                    (
                        f"Reproductive group {group_key} alive_expression_counts "
                        "does not match the agent catalog."
                    ),
                )
            )
        birth_counts = {
            field: _as_optional_int(group_payload.get(field))
            for field in ("asexual_births", "sexual_births", "hybrid_births")
        }
        group_stage = group_payload.get("stage")
        if group_stage not in REPRODUCTIVE_STAGE_ORDER:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.stage",
                    f"Reproductive group {group_key} has unknown stage.",
                )
            )
        for field, count in birth_counts.items():
            if count is None or count < 0:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        f"viewer.reproductive_group_catalog.groups.{field}",
                        (
                            f"Reproductive group {group_key} {field} must be a "
                            "nonnegative integer."
                        ),
                    )
                )
        if event_birth_counts is not None:
            expected_event_births = event_birth_counts.get(str(group_key), {})
            for field, count in birth_counts.items():
                if count is None or count < 0:
                    continue
                expected_count = expected_event_births.get(field, 0)
                if count != expected_count:
                    flags.append(
                        _flag(
                            "error",
                            scope,
                            f"viewer.reproductive_group_catalog.groups.{field}",
                            (
                                f"Reproductive group {group_key} {field} does "
                                "not match agent_reproduced events."
                            ),
                        )
                    )
        sexual_births = birth_counts.get("sexual_births")
        hybrid_births = birth_counts.get("hybrid_births")
        if (
            sexual_births is not None
            and hybrid_births is not None
            and hybrid_births > sexual_births
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.hybrid_births",
                    (
                        f"Reproductive group {group_key} hybrid_births exceeds "
                        "sexual_births."
                    ),
                )
            )
        highest_alive_stage = _highest_counted_stage(expected_stage_counts)
        if (
            highest_alive_stage is not None
            and isinstance(group_stage, str)
            and stage_rank(group_stage) < stage_rank(highest_alive_stage)
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.stage",
                    (
                        f"Reproductive group {group_key} stage is below an "
                        "alive member stage."
                    ),
                )
            )
    if event_birth_counts is not None:
        flags.extend(
            _reproductive_summary_birth_flags(
                scope=scope,
                summary=summary,
                event_birth_counts=event_birth_counts,
            )
        )
    flags.extend(
        _reproductive_summary_catalog_flags(
            scope=scope,
            summary=summary,
            groups=groups,
            alive_member_counts=alive_member_counts,
            alive_stage_counts=alive_stage_counts,
            alive_expression_counts=alive_expression_counts,
        )
    )
    return flags


def _reproductive_summary_birth_flags(
    *,
    scope: str,
    summary: dict[str, object],
    event_birth_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    event_totals = {
        field: sum(
            int(group_counts.get(field, 0))
            for group_counts in event_birth_counts.values()
        )
        for field in ("asexual_births", "sexual_births", "hybrid_births")
    }
    event_birth_total = event_totals["asexual_births"] + event_totals["sexual_births"]
    summary_births = _as_optional_int(summary.get("births"))
    if summary_births is None or summary_births != event_birth_total:
        flags.append(
            _flag(
                "error",
                scope,
                "summary.births",
                "Summary births does not match agent_reproduced events.",
            )
        )
    reproductive_groups_end = summary.get("reproductive_groups_end")
    if not isinstance(reproductive_groups_end, Mapping):
        flags.append(
            _flag(
                "error",
                scope,
                "summary.reproductive_groups_end",
                "Full replay summary is missing reproductive group totals.",
            )
        )
        return flags
    if (
        reproductive_groups_end.get("schema_version")
        != REPRODUCTIVE_GROUP_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "summary.reproductive_groups_end.schema_version",
                "Summary reproductive group totals are missing or stale.",
            )
        )
    for field, expected_count in event_totals.items():
        summary_count = _as_optional_int(reproductive_groups_end.get(field))
        if summary_count is None or summary_count != expected_count:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"summary.reproductive_groups_end.{field}",
                    (
                        f"Summary reproductive group {field} does not match "
                        "agent_reproduced events."
                    ),
                )
            )
    return flags


def _reproductive_summary_catalog_flags(
    *,
    scope: str,
    summary: dict[str, object],
    groups: Mapping[object, object],
    alive_member_counts: Mapping[str, int],
    alive_stage_counts: Mapping[str, Mapping[str, int]],
    alive_expression_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    reproductive_groups_end = summary.get("reproductive_groups_end")
    if not isinstance(reproductive_groups_end, Mapping):
        return []

    stage_counts: dict[str, int] = {}
    for group_payload in groups.values():
        if not isinstance(group_payload, Mapping):
            continue
        stage = group_payload.get("stage")
        if isinstance(stage, str):
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    catalog_totals = {
        "group_count": len(groups),
        "alive_group_count": sum(
            1 for count in alive_member_counts.values() if count > 0
        ),
        "stage_counts": stage_counts,
        "alive_stage_counts": _sum_grouped_counts(alive_stage_counts),
        "alive_expression_counts": _sum_grouped_counts(alive_expression_counts),
    }
    flags: list[dict[str, object]] = []
    for field in ("group_count", "alive_group_count"):
        summary_count = _as_optional_int(reproductive_groups_end.get(field))
        if summary_count is None or summary_count != catalog_totals[field]:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"summary.reproductive_groups_end.{field}",
                    (
                        f"Summary reproductive group {field} does not match "
                        "the viewer catalog."
                    ),
                )
            )
    for field in ("stage_counts", "alive_stage_counts", "alive_expression_counts"):
        summary_counts = _as_count_dict(reproductive_groups_end.get(field))
        if summary_counts != catalog_totals[field]:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"summary.reproductive_groups_end.{field}",
                    (
                        f"Summary reproductive group {field} does not match "
                        "the viewer catalog."
                    ),
                )
            )
    return flags


def _reproductive_group_birth_counts_from_events(
    *,
    scope: str,
    events: Sequence[object] | None,
    groups: Mapping[object, object],
) -> tuple[dict[str, dict[str, int]] | None, list[dict[str, object]]]:
    flags: list[dict[str, object]] = []
    if events is None:
        return None, flags
    if isinstance(events, (str, bytes, bytearray)):
        return None, [
            _flag(
                "error",
                scope,
                "events",
                "Full replay events must be a sequence of event objects.",
            )
        ]

    group_keys = {str(group_key) for group_key in groups}
    counts: dict[str, dict[str, int]] = {
        group_key: {
            "asexual_births": 0,
            "sexual_births": 0,
            "hybrid_births": 0,
        }
        for group_key in group_keys
    }
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events",
                    f"Full replay event {index} is not an object.",
                )
            )
            continue
        if event.get("type") != "agent_reproduced":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.data",
                    f"Reproduction event {index} is missing a data object.",
                )
            )
            continue
        if data.get("schema_version") != REPRODUCTION_EVENT_SCHEMA_VERSION:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.schema_version",
                    (
                        f"Reproduction event {index} must declare "
                        f"{REPRODUCTION_EVENT_SCHEMA_VERSION}."
                    ),
                )
            )
        raw_group_id = data.get("child_reproductive_group_id")
        group_id = _as_optional_int(raw_group_id)
        if group_id is None or group_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.child_reproductive_group_id",
                    (
                        f"Reproduction event {index} child_reproductive_group_id "
                        "must be a nonnegative integer."
                    ),
                )
            )
            continue
        group_key = str(group_id)
        if group_key not in group_keys:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.child_reproductive_group_id",
                    (
                        f"Reproduction event {index} references missing "
                        f"reproductive group {group_key}."
                    ),
                )
            )
            continue
        reproduction_mode = data.get("reproduction_mode")
        hybrid = data.get("hybrid")
        if not isinstance(hybrid, bool):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.hybrid",
                    f"Reproduction event {index} hybrid must be boolean.",
                )
            )
            hybrid = False
        if reproduction_mode == ASEXUAL_REPRODUCTION_MODE:
            counts[group_key]["asexual_births"] += 1
            if hybrid:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        "events.agent_reproduced.hybrid",
                        (
                            f"Reproduction event {index} cannot mark an "
                            "asexual birth as hybrid."
                        ),
                    )
                )
        elif reproduction_mode == SEXUAL_REPRODUCTION_MODE:
            counts[group_key]["sexual_births"] += 1
            if hybrid:
                counts[group_key]["hybrid_births"] += 1
        else:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.reproduction_mode",
                    (
                        f"Reproduction event {index} has unsupported "
                        f"reproduction_mode {reproduction_mode!r}."
                    ),
                )
            )
    return counts, flags


def _increment_nested_count(
    counts: dict[str, dict[str, int]],
    group_key: str,
    field_value: str,
) -> None:
    group_counts = counts.setdefault(group_key, {})
    group_counts[field_value] = group_counts.get(field_value, 0) + 1


def _sum_grouped_counts(
    grouped_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counts in grouped_counts.values():
        for key, count in counts.items():
            totals[key] = totals.get(key, 0) + int(count)
    return totals


def _highest_counted_stage(stage_counts: dict[str, int]) -> str | None:
    stages = [stage for stage, count in stage_counts.items() if count > 0]
    if not stages:
        return None
    return max(stages, key=stage_rank)


def _as_count_dict(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    counts: dict[str, int] = {}
    for key, count in value.items():
        parsed_count = _as_optional_int(count)
        if parsed_count is None or parsed_count < 0:
            return None
        counts[str(key)] = parsed_count
    return counts


def _signal_action_capacity_flags(
    *,
    scope: str,
    trajectory: dict[str, object],
    observation_contract: dict[str, object],
    signal_contract: dict[str, object],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    token_count = _as_optional_int(signal_contract.get("communication_token_count"))
    profiles_per_token = _as_optional_int(
        signal_contract.get("communication_profiles_per_token")
    )
    communication_enabled = signal_contract.get(
        "communication_signal_emission_enabled"
    )
    reserved_profiles = signal_contract.get("reserved_profiles")
    if (
        token_count is None
        or profiles_per_token is None
        or not isinstance(communication_enabled, bool)
    ):
        return [
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.signal_contract",
                "Signal contract does not declare communication capacity and enablement.",
            )
        ]

    expected_actions = [
        f"signal_{token_index}_profile_{profile_index}"
        for token_index in range(token_count)
        for profile_index in range(profiles_per_token)
    ]
    if (
        not isinstance(reserved_profiles, list)
        or len(reserved_profiles) != len(expected_actions)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.signal_contract.reserved_profiles",
                "Signal contract reserved profile count does not match communication capacity.",
            )
        )

    for path, action_contract in (
        ("viewer.trajectory.action_contract", trajectory.get("action_contract")),
        (
            "viewer.trajectory.observation_contract.action_contract",
            observation_contract.get("action_contract"),
        ),
    ):
        if not _action_contract_matches_signal_capacity(
            action_contract,
            token_count=token_count,
            profiles_per_token=profiles_per_token,
            expected_actions=expected_actions,
            communication_enabled=communication_enabled,
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    path,
                    "Action contract communication slots do not match signal contract capacity.",
                )
            )

    action_names = observation_contract.get("action_names")
    if not isinstance(action_names, list) or not set(expected_actions).issubset(
        set(action_names)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.action_names",
                "Observation action names do not include every reserved communication slot.",
            )
        )
    return flags


def _action_contract_matches_signal_capacity(
    action_contract: object,
    *,
    token_count: int,
    profiles_per_token: int,
    expected_actions: list[str],
    communication_enabled: bool,
) -> bool:
    if not isinstance(action_contract, dict):
        return False
    communication = action_contract.get("communication")
    if not isinstance(communication, dict):
        return False
    if communication.get("emission_enabled") is not communication_enabled:
        return False
    if communication.get("token_count") != token_count:
        return False
    if communication.get("profiles_per_token") != profiles_per_token:
        return False
    if communication.get("action_keys") != expected_actions:
        return False
    reserved_actions = action_contract.get("reserved_action_keys")
    if not isinstance(reserved_actions, list) or not set(expected_actions).issubset(
        set(reserved_actions)
    ):
        return False
    active_actions = action_contract.get("active_action_keys")
    if not isinstance(active_actions, list):
        return False
    active_set = set(active_actions)
    expected_set = set(expected_actions)
    if communication_enabled:
        return expected_set.issubset(active_set)
    return active_set.isdisjoint(expected_set)


def _run_full_replay_probe(probe: FullReplayProbe) -> dict[str, object]:
    result = SimulationWorld(WorldConfig(seed=probe.seed, max_ticks=probe.ticks)).run(
        mode=RunMode.FULL_REPLAY
    )
    summary = result.summary
    replay_size_bytes = 0
    flags: list[dict[str, object]] = []

    if result.events is None or result.viewer is None:
        flags.append(
            _flag(
                "error",
                probe.name,
                "result",
                "Full replay probe did not return events and viewer payloads.",
            )
        )
    else:
        replay_size_bytes = _replay_size_bytes(build_replay_payload(result))
        frame_count = len(result.viewer["frames"])
        if frame_count != int(summary["ticks_executed"]):
            flags.append(
                _flag(
                    "error",
                    probe.name,
                    "viewer.frames",
                    "Viewer frame count does not match executed ticks.",
                )
            )
        if "taxonomy" not in result.viewer:
            flags.append(
                _flag(
                    "error",
                    probe.name,
                    "viewer.taxonomy",
                    "Full replay probe did not include taxonomy payload.",
                )
            )
        if "species_metrics" not in result.viewer["frames"][-1]:
            flags.append(
                _flag(
                    "error",
                    probe.name,
                    "viewer.frames.species_metrics",
                    "Full replay probe did not include species metrics.",
                )
            )
        flags.extend(
            _mind_contract_flags(
                scope=probe.name,
                summary=summary,
                viewer=result.viewer,
                events=result.events,
            )
        )
        flags.extend(
            _reproductive_role_readiness_flags(
                scope=probe.name,
                reproduction=summary.get("reproduction_end"),
            )
        )

    if int(summary["alive_species_count"]) < probe.min_alive_species:
        flags.append(
            _flag(
                "error",
                probe.name,
                "alive_species_count",
                (
                    "Full replay probe ended below the alive-species floor of "
                    f"{probe.min_alive_species}."
                ),
            )
        )
    if int(summary["species_created"]) < probe.min_species_created:
        flags.append(
            _flag(
                "error",
                probe.name,
                "species_created",
                (
                    "Full replay probe created fewer species than required "
                    f"({probe.min_species_created})."
                ),
            )
        )
    if probe.require_speciation and int(summary["speciation_events"]) <= 0:
        flags.append(
            _flag(
                "error",
                probe.name,
                "speciation_events",
                "Expected at least one replay-taxonomy speciation event.",
            )
        )
    if (
        probe.max_replay_size_bytes is not None
        and replay_size_bytes > probe.max_replay_size_bytes
    ):
        flags.append(
            _flag(
                "warning",
                probe.name,
                "replay_size_bytes",
                (
                    f"Replay payload size {replay_size_bytes} exceeds budget "
                    f"{probe.max_replay_size_bytes}."
                ),
            )
        )

    return {
        "name": probe.name,
        "seed": probe.seed,
        "ticks": probe.ticks,
        "run_id": summary["run_id"],
        "ticks_executed": summary["ticks_executed"],
        "alive_agents": summary["alive_agents"],
        "max_agents": summary["max_agents"],
        "max_agent_saturation_at_end": summary["max_agent_saturation_at_end"],
        "peak_max_agent_saturation": summary["peak_max_agent_saturation"],
        "births": summary["births"],
        "deaths": summary["deaths"],
        "reproduction": summary["reproduction_end"],
        "species_created": summary["species_created"],
        "alive_species_count": summary["alive_species_count"],
        "speciation_events": summary["speciation_events"],
        "taxonomy_mode": summary["taxonomy_mode"],
        "replay_size_bytes": replay_size_bytes,
        "max_replay_size_bytes": probe.max_replay_size_bytes,
        "flags": flags,
    }


def _run_summary_seed(
    *,
    seed: int,
    ticks: int,
    timeout_seconds: float | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, float]:
    started = time.perf_counter()
    if timeout_seconds is None:
        try:
            record = run_evaluation(seed=seed, ticks=ticks, mode=RunMode.SUMMARY_ONLY)
        except Exception as exc:
            wall_seconds = _round_seconds(time.perf_counter() - started)
            return (
                None,
                {
                    "severity": "error",
                    "seed": seed,
                    "field": "scenario_exception",
                    "message": f"Summary seed raised {type(exc).__name__}: {exc}",
                },
                wall_seconds,
            )
        wall_seconds = _round_seconds(time.perf_counter() - started)
        record["wall_seconds"] = wall_seconds
        return record, None, wall_seconds

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(
        target=_summary_seed_worker,
        args=(seed, ticks, RunMode.SUMMARY_ONLY.value, result_queue),
    )
    result = _worker_result(
        process=process,
        result_queue=result_queue,
        timeout_seconds=timeout_seconds,
        timeout_message=(
            f"Summary seed {seed} exceeded scenario timeout of "
            f"{timeout_seconds:.2f}s."
        ),
    )
    wall_seconds = _round_seconds(time.perf_counter() - started)
    if bool(result.get("ok", False)):
        record = dict(result["result"])
        record["wall_seconds"] = wall_seconds
        return record, None, wall_seconds
    error = str(result.get("error", "RuntimeError"))
    message = str(result.get("message", "Summary seed failed."))
    field = "scenario_timeout" if error == "TimeoutError" else "scenario_exception"
    return (
        None,
        {
            "severity": "error",
            "seed": seed,
            "field": field,
            "message": message,
        },
        wall_seconds,
    )


def _full_replay_probe_error_report(
    *,
    probe: FullReplayProbe,
    field: str,
    message: str,
    wall_seconds: float,
) -> dict[str, object]:
    return {
        "name": probe.name,
        "seed": probe.seed,
        "ticks": probe.ticks,
        "run_id": None,
        "ticks_executed": None,
        "alive_agents": None,
        "max_agents": None,
        "max_agent_saturation_at_end": None,
        "peak_max_agent_saturation": None,
        "births": None,
        "deaths": None,
        "reproduction": None,
        "species_created": None,
        "alive_species_count": None,
        "speciation_events": None,
        "taxonomy_mode": None,
        "replay_size_bytes": None,
        "max_replay_size_bytes": probe.max_replay_size_bytes,
        "wall_seconds": wall_seconds,
        "flags": [_flag("error", probe.name, field, message)],
    }


def _run_full_replay_probe_with_timeout(
    probe: FullReplayProbe,
    *,
    timeout_seconds: float | None,
) -> dict[str, object]:
    started = time.perf_counter()
    if timeout_seconds is None:
        try:
            report = _run_full_replay_probe(probe)
        except Exception as exc:
            wall_seconds = _round_seconds(time.perf_counter() - started)
            return _full_replay_probe_error_report(
                probe=probe,
                field="scenario_exception",
                message=f"Full replay probe raised {type(exc).__name__}: {exc}",
                wall_seconds=wall_seconds,
            )
        report["wall_seconds"] = _round_seconds(time.perf_counter() - started)
        return report

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(
        target=_full_replay_probe_worker,
        args=(probe, result_queue),
    )
    result = _worker_result(
        process=process,
        result_queue=result_queue,
        timeout_seconds=timeout_seconds,
        timeout_message=(
            f"Full replay probe {probe.name!r} exceeded scenario timeout of "
            f"{timeout_seconds:.2f}s."
        ),
    )
    wall_seconds = _round_seconds(time.perf_counter() - started)
    if bool(result.get("ok", False)):
        report = dict(result["result"])
        report["wall_seconds"] = wall_seconds
        return report
    error = str(result.get("error", "RuntimeError"))
    message = str(result.get("message", "Full replay probe failed."))
    field = "scenario_timeout" if error == "TimeoutError" else "scenario_exception"
    return _full_replay_probe_error_report(
        probe=probe,
        field=field,
        message=message,
        wall_seconds=wall_seconds,
    )


def _readiness(
    *,
    summary_flags: Sequence[dict[str, object]],
    full_replay_probes: Sequence[dict[str, object]],
) -> dict[str, object]:
    all_flags = list(summary_flags)
    for probe in full_replay_probes:
        all_flags.extend(probe["flags"])
    blockers = [flag for flag in all_flags if flag["severity"] == "error"]
    warnings = [flag for flag in all_flags if flag["severity"] == "warning"]
    if blockers:
        status = "fail"
        recommendation = "Fix Foundation blockers before starting Mind v1."
    elif warnings:
        status = "review"
        recommendation = "Review warnings before treating Foundation as closed."
    else:
        status = "pass"
        recommendation = "Foundation gate checks passed for this profile."
    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "recommendation": recommendation,
    }


def build_foundation_gate_report(
    profile: GateProfile,
    *,
    summary_seeds: Sequence[int] | None = None,
    summary_ticks: int | None = None,
    progress: ProgressCallback | None = None,
    incremental_output_path: Path | None = None,
    scenario_timeout_seconds: float | None = None,
) -> dict[str, object]:
    seeds = tuple(summary_seeds or profile.summary_seeds)
    ticks = summary_ticks or profile.summary_ticks
    timeout_seconds = _normalized_timeout(scenario_timeout_seconds)
    total_started = time.perf_counter()
    protocol = {
        "protocol": {
            "profile": profile.name,
            "summary_sweep": {
                "seeds": list(seeds),
                "ticks": ticks,
                "mode": RunMode.SUMMARY_ONLY.value,
            },
            "scenario_timeout_seconds": timeout_seconds,
            "criteria": {
                "min_alive_agents": profile.min_alive_agents,
                "min_births": profile.min_births,
                "min_last_birth_tick": profile.min_last_birth_tick,
                "min_trophic_roles": profile.min_trophic_roles,
                "min_meat_modes": profile.min_meat_modes,
                "min_aggregate_meat_modes": (
                    profile.min_aggregate_meat_modes or profile.min_meat_modes
                ),
                "min_hazardous_tiles": profile.min_hazardous_tiles,
                "min_ecology_pressure_tiles": profile.min_ecology_pressure_tiles,
                "min_animal_energy_share": profile.min_animal_energy_share,
                "min_carrion_consumed_deposited_ratio": (
                    profile.min_carrion_consumed_deposited_ratio
                ),
                "min_animal_resource_consumption_run_share_by_mode": (
                    profile.min_animal_resource_consumption_run_share_by_mode
                ),
                "use_late_window_population_floor": profile.use_late_window_population_floor,
                "dominance_warning_share": profile.dominance_warning_share,
                "required_terminal_meat_mode_alternatives_by_seed": {
                    str(seed): list(modes)
                    for seed, modes in (
                        profile.required_terminal_meat_mode_alternatives_by_seed
                        or {}
                    ).items()
                },
            },
            "full_replay_probes": [asdict(probe) for probe in profile.full_replay_probes],
        }
    }
    timings: dict[str, object] = {
        "scenario_timeout_seconds": timeout_seconds,
        "summary_sweep_wall_seconds": None,
        "summary_seed_wall_seconds": [],
        "full_replay_probe_wall_seconds": [],
        "total_wall_seconds": None,
    }
    runs: list[dict[str, object]] = []
    run_errors: list[dict[str, object]] = []
    full_replay_probes: list[dict[str, object]] = []

    def build_evaluation() -> dict[str, object]:
        return build_evaluation_report_from_runs(
            seeds=seeds,
            ticks=ticks,
            mode=RunMode.SUMMARY_ONLY,
            min_alive_agents=profile.min_alive_agents,
            min_births=profile.min_births,
            dominance_warning_share=profile.dominance_warning_share,
            runs=runs,
            run_errors=run_errors,
        )

    def build_report(*, complete: bool) -> dict[str, object]:
        evaluation = build_evaluation()
        summary_flags = _summary_gate_flags(evaluation, profile)
        report = {
            **protocol,
            "complete": complete,
            "summary_evaluation": evaluation,
            "summary_gate_flags": summary_flags,
            "ecology_failure_rollup": _ecology_failure_rollup(evaluation, profile),
            "full_replay_probes": full_replay_probes,
            "timings": timings,
            "readiness": _readiness(
                summary_flags=summary_flags,
                full_replay_probes=full_replay_probes,
            ),
        }
        if not complete:
            report["readiness"] = {
                "status": "running",
                "blockers": [],
                "warnings": [],
                "recommendation": "Foundation gate is still running.",
            }
        return report

    _emit_progress(
        progress,
        f"summary sweep start seeds={list(seeds)} ticks={ticks} profile={profile.name}",
    )
    summary_started = time.perf_counter()
    for index, seed in enumerate(seeds, start=1):
        _emit_progress(progress, f"summary seed {seed} start ({index}/{len(seeds)})")
        record, error, wall_seconds = _run_summary_seed(
            seed=seed,
            ticks=ticks,
            timeout_seconds=timeout_seconds,
        )
        if record is not None:
            runs.append(record)
        if error is not None:
            run_errors.append(error)
        timings["summary_seed_wall_seconds"].append(
            {"seed": seed, "wall_seconds": wall_seconds}
        )
        _write_gate_report(incremental_output_path, build_report(complete=False))
        _emit_progress(
            progress,
            f"summary seed {seed} complete ({index}/{len(seeds)}) wall_seconds={wall_seconds}",
        )

    timings["summary_sweep_wall_seconds"] = _round_seconds(
        time.perf_counter() - summary_started
    )
    _write_gate_report(incremental_output_path, build_report(complete=False))
    _emit_progress(
        progress,
        f"summary sweep complete wall_seconds={timings['summary_sweep_wall_seconds']}",
    )

    for index, probe in enumerate(profile.full_replay_probes, start=1):
        _emit_progress(
            progress,
            f"full replay probe {probe.name} start ({index}/{len(profile.full_replay_probes)})",
        )
        probe_report = _run_full_replay_probe_with_timeout(
            probe,
            timeout_seconds=timeout_seconds,
        )
        full_replay_probes.append(probe_report)
        timings["full_replay_probe_wall_seconds"].append(
            {
                "name": probe.name,
                "seed": probe.seed,
                "ticks": probe.ticks,
                "wall_seconds": probe_report["wall_seconds"],
            }
        )
        _write_gate_report(incremental_output_path, build_report(complete=False))
        _emit_progress(
            progress,
            (
                f"full replay probe {probe.name} complete "
                f"({index}/{len(profile.full_replay_probes)}) "
                f"wall_seconds={probe_report['wall_seconds']}"
            ),
        )

    timings["total_wall_seconds"] = _round_seconds(time.perf_counter() - total_started)
    final_report = build_report(complete=True)
    _write_gate_report(incremental_output_path, final_report)
    _emit_progress(progress, f"gate complete wall_seconds={timings['total_wall_seconds']}")
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Foundation readiness checks before starting Mind v1."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=QUICK_PROFILE.name,
        help="quick is local/CI-safe; release includes long opt-in gate probes.",
    )
    parser.add_argument(
        "--seeds",
        help="Override the profile's summary-only seed list with comma-separated seeds.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Add one summary-only seed override. May be supplied more than once.",
    )
    parser.add_argument("--ticks", type=int, help="Override summary-only sweep ticks.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--scenario-timeout-seconds",
        type=float,
        default=600.0,
        help=(
            "Per summary seed and full-replay probe timeout. "
            "Use 0 to disable timeout isolation."
        ),
    )
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit non-zero when the gate status is fail.",
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Exit non-zero when the gate status is fail or review.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = PROFILES[args.profile]
    summary_seeds = None
    if args.seeds or args.seed:
        summary_seeds = parse_seed_selection(args.seed, args.seeds)
    report = build_foundation_gate_report(
        profile,
        summary_seeds=summary_seeds,
        summary_ticks=args.ticks,
        progress=lambda message: print(
            f"[foundation-gate] {message}",
            file=sys.stderr,
            flush=True,
        ),
        incremental_output_path=args.output,
        scenario_timeout_seconds=args.scenario_timeout_seconds,
    )
    payload = json.dumps(report, indent=2)
    print(payload)

    status = report["readiness"]["status"]
    if args.fail_on_review and status in {"fail", "review"}:
        raise SystemExit(1)
    if args.fail_on_blockers and status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
