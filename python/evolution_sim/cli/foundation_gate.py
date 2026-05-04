from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.contracts import SUMMARY_SCHEMA_VERSION
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
from .foundation_gate_validators.common import (
    _as_int_count_mapping,
    _as_optional_int,
)
from .foundation_gate_validators.reproduction import (
    _reproductive_group_catalog_flags,
    _reproductive_role_readiness_flags,
)
from .foundation_gate_validators.summary import (
    _ecology_failure_rollup,
    _summary_gate_flags,
)
from .foundation_gate_validators.trajectory import _mind_contract_flags


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
    max_at_cap_tick_share_warning: float | None = None
    max_at_cap_tick_share_error: float | None = None
    min_plant_energy_available_per_land_tile_warning: float | None = None
    min_terminal_selection_abs_mean_delta_warning: float | None = None
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
    max_at_cap_tick_share_warning=0.35,
    max_at_cap_tick_share_error=0.6,
    min_plant_energy_available_per_land_tile_warning=0.05,
    min_terminal_selection_abs_mean_delta_warning=0.001,
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




def _replay_size_bytes(result_payload: dict[str, object]) -> int:
    return replay_payload_size_bytes(result_payload)












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
        "carrying_capacity": summary["carrying_capacity"],
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
        "carrying_capacity": None,
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
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
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
                "max_at_cap_tick_share_warning": profile.max_at_cap_tick_share_warning,
                "max_at_cap_tick_share_error": profile.max_at_cap_tick_share_error,
                "min_plant_energy_available_per_land_tile_warning": (
                    profile.min_plant_energy_available_per_land_tile_warning
                ),
                "min_terminal_selection_abs_mean_delta_warning": (
                    profile.min_terminal_selection_abs_mean_delta_warning
                ),
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
