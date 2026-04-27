from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.observations import OBSERVATION_SCHEMA_VERSION
from evolution_sim.env.runtime.trajectory import REWARD_SCHEMA_VERSION, TRAJECTORY_SCHEMA_VERSION

from .evaluate import build_evaluation_report, parse_seed_selection
from .golden_harness import GOLDEN_SPECIATION_SEED


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
)

PROFILES: dict[str, GateProfile] = {
    QUICK_PROFILE.name: QUICK_PROFILE,
    RELEASE_PROFILE.name: RELEASE_PROFILE,
}


def _flag(severity: str, scope: str, field: str, message: str) -> dict[str, object]:
    return {
        "severity": severity,
        "scope": scope,
        "field": field,
        "message": message,
    }


def _positive_keys(counts: dict[str, int], *, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return sorted(key for key, value in counts.items() if key not in excluded and value > 0)


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
        run_trophic_roles = _positive_keys(role_counts)
        if len(run_trophic_roles) < profile.min_trophic_roles:
            flags.append(
                _flag(
                    "error",
                    f"summary_seed_{seed}",
                    "trophic.role_counts",
                    (
                        "Run ended below the per-run trophic-role floor of "
                        f"{profile.min_trophic_roles}: {run_trophic_roles}."
                    ),
                )
            )
        meat_mode_counts = trophic["meat_mode_counts"]
        run_meat_modes = _positive_keys(meat_mode_counts, exclude={"none"})
        animal_mode_floor = profile.min_meat_modes
        if len(run_meat_modes) < animal_mode_floor:
            flags.append(
                _flag(
                    "error",
                    f"summary_seed_{seed}",
                    "trophic.meat_mode_counts",
                    (
                        "Run ended below the per-run animal-resource strategy floor "
                        f"of {animal_mode_floor}: {run_meat_modes}."
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
    hazardous_min = int(aggregate["hazardous_tiles"]["min"])
    if hazardous_min < profile.min_hazardous_tiles:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "hazardous_tiles",
                "At least one run ended without visible hazard pressure.",
            )
        )

    trophic_counts = aggregate["trophic_role_counts_at_end"]["total"]
    trophic_roles = _positive_keys(trophic_counts)
    if len(trophic_roles) < profile.min_trophic_roles:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "trophic_role_counts_at_end",
                (
                    "Ending populations do not yet show enough trophic-role "
                    f"diversity: {trophic_roles}."
                ),
            )
        )

    meat_mode_counts = aggregate["meat_mode_counts_at_end"]["total"]
    meat_modes = _positive_keys(meat_mode_counts, exclude={"none"})
    animal_mode_floor = profile.min_meat_modes
    if len(meat_modes) < animal_mode_floor:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "meat_mode_counts_at_end",
                (
                    "Ending populations do not yet show enough animal-resource "
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

    carrion_consumed_max = float(aggregate["carrion_energy_consumed"]["max"])
    fresh_kill_consumed_max = float(aggregate["fresh_kill_energy_consumed"]["max"])
    if carrion_consumed_max <= 0 and fresh_kill_consumed_max <= 0:
        flags.append(
            _flag(
                "error",
                "summary_sweep",
                "animal_resource_consumption",
                "No animal-resource consumption was observed across the sweep.",
            )
        )
    return flags


def _replay_size_bytes(result_payload: dict[str, object]) -> int:
    return len(json.dumps(result_payload, indent=2).encode("utf-8"))


def _mind_contract_flags(
    *,
    scope: str,
    summary: dict[str, object],
    viewer: dict[str, object],
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
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
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
    if trajectory.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.schema_version",
                "Trajectory schema version is missing or stale.",
            )
        )
    observation_contract = trajectory.get("observation_contract")
    if (
        not isinstance(observation_contract, dict)
        or observation_contract.get("schema_version") != OBSERVATION_SCHEMA_VERSION
        or observation_contract.get("privileged_world_state") is not False
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract",
                "Observation contract metadata is missing or permits privileged world state.",
            )
        )
    records = trajectory.get("records")
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
    first_record = records[0]
    required_record_fields = {
        "observation_digest",
        "action_mask",
        "requested_action",
        "action_valid",
        "resolved_action",
        "outcome",
        "reward",
    }
    if not isinstance(first_record, dict) or not required_record_fields.issubset(first_record):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records",
                "Trajectory records do not include action, outcome, mask, and reward fields.",
            )
        )
    elif not isinstance(first_record.get("reward"), dict) or first_record["reward"].get(
        "schema_version"
    ) != REWARD_SCHEMA_VERSION:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.reward",
                "Trajectory records do not include versioned reward components.",
            )
        )
    return flags


def _run_full_replay_probe(probe: FullReplayProbe) -> dict[str, object]:
    result = SimulationWorld(WorldConfig(seed=probe.seed, max_ticks=probe.ticks)).run(
        mode=RunMode.FULL_REPLAY
    )
    summary = result.summary
    payload = {
        "run_id": result.run_id,
        "config": result.config,
        "summary": result.summary,
        "events": result.events,
        "viewer": result.viewer,
    }
    replay_size_bytes = _replay_size_bytes(payload)
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
        flags.extend(_mind_contract_flags(scope=probe.name, summary=summary, viewer=result.viewer))

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
        "births": summary["births"],
        "deaths": summary["deaths"],
        "species_created": summary["species_created"],
        "alive_species_count": summary["alive_species_count"],
        "speciation_events": summary["speciation_events"],
        "taxonomy_mode": summary["taxonomy_mode"],
        "replay_size_bytes": replay_size_bytes,
        "max_replay_size_bytes": probe.max_replay_size_bytes,
        "flags": flags,
    }


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
) -> dict[str, object]:
    seeds = tuple(summary_seeds or profile.summary_seeds)
    ticks = summary_ticks or profile.summary_ticks
    evaluation = build_evaluation_report(
        seeds=seeds,
        ticks=ticks,
        mode=RunMode.SUMMARY_ONLY,
        min_alive_agents=profile.min_alive_agents,
        min_births=profile.min_births,
    )
    summary_flags = _summary_gate_flags(evaluation, profile)
    full_replay_probes = [
        _run_full_replay_probe(probe) for probe in profile.full_replay_probes
    ]
    return {
        "protocol": {
            "profile": profile.name,
            "summary_sweep": {
                "seeds": list(seeds),
                "ticks": ticks,
                "mode": RunMode.SUMMARY_ONLY.value,
            },
            "criteria": {
                "min_alive_agents": profile.min_alive_agents,
                "min_births": profile.min_births,
                "min_last_birth_tick": profile.min_last_birth_tick,
                "min_trophic_roles": profile.min_trophic_roles,
                "min_meat_modes": profile.min_meat_modes,
                "min_hazardous_tiles": profile.min_hazardous_tiles,
                "min_ecology_pressure_tiles": profile.min_ecology_pressure_tiles,
                "min_animal_energy_share": profile.min_animal_energy_share,
                "min_carrion_consumed_deposited_ratio": (
                    profile.min_carrion_consumed_deposited_ratio
                ),
            },
            "full_replay_probes": [asdict(probe) for probe in profile.full_replay_probes],
        },
        "summary_evaluation": evaluation,
        "summary_gate_flags": summary_flags,
        "full_replay_probes": full_replay_probes,
        "readiness": _readiness(
            summary_flags=summary_flags,
            full_replay_probes=full_replay_probes,
        ),
    }


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
    )
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    status = report["readiness"]["status"]
    if args.fail_on_review and status in {"fail", "review"}:
        raise SystemExit(1)
    if args.fail_on_blockers and status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
