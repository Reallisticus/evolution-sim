from __future__ import annotations

from collections.abc import Mapping

from evolution_sim.env.world import (
    ANIMAL_RESOURCE_KINDS,
    ANIMAL_RESOURCE_POLICY_BLOCKERS,
)
from evolution_sim.cli.foundation_gate_validators.common import (
    _as_optional_int,
    _flag,
)
from evolution_sim.cli.foundation_gate_validators.reproduction import (
    _reproductive_role_readiness_flags,
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
        flags.extend(
            _carrying_capacity_saturation_flags(
                scope=f"summary_seed_{seed}",
                carrying_capacity=run.get("carrying_capacity"),
                profile=profile,
            )
        )
        flags.extend(
            _resource_pressure_budget_flags(
                scope=f"summary_seed_{seed}",
                run=run,
                profile=profile,
            )
        )
        flags.extend(
            _terminal_selection_signal_flags(
                scope=f"summary_seed_{seed}",
                selection=run.get("selection_heredity"),
                profile=profile,
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
                opportunity_runs = int(
                    mode_counts.get("animal_resource_opportunity_runs", alive_runs)
                )
                if opportunity_runs <= 0:
                    continue
                no_consumption_runs = int(
                    mode_counts.get("no_animal_consumption_runs", 0)
                )
                consuming_runs = int(
                    mode_counts.get(
                        "animal_resource_consuming_runs",
                        alive_runs - no_consumption_runs,
                    )
                )
                consuming_run_share = consuming_runs / max(opportunity_runs, 1)
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
                                f"({consuming_runs}/{opportunity_runs})."
                            ),
                        )
                    )
    return flags


def _carrying_capacity_saturation_flags(
    *,
    scope: str,
    carrying_capacity: object,
    profile: GateProfile,
) -> list[dict[str, object]]:
    if not isinstance(carrying_capacity, Mapping):
        return []
    at_cap_share_raw = carrying_capacity.get("at_cap_tick_share")
    if isinstance(at_cap_share_raw, bool) or not isinstance(
        at_cap_share_raw,
        (int, float),
    ):
        return []
    at_cap_share = float(at_cap_share_raw)
    if (
        profile.max_at_cap_tick_share_error is not None
        and at_cap_share > profile.max_at_cap_tick_share_error
    ):
        return [
            _flag(
                "error",
                scope,
                "carrying_capacity.at_cap_tick_share",
                (
                    "Run spent too much of the scenario at max-agent saturation "
                    f"({at_cap_share:.4f} > "
                    f"{profile.max_at_cap_tick_share_error:.4f})."
                ),
            )
        ]
    if (
        profile.max_at_cap_tick_share_warning is not None
        and at_cap_share > profile.max_at_cap_tick_share_warning
    ):
        return [
            _flag(
                "warning",
                scope,
                "carrying_capacity.at_cap_tick_share",
                (
                    "Run spent a sustained share of the scenario at max-agent "
                    f"saturation ({at_cap_share:.4f} > "
                    f"{profile.max_at_cap_tick_share_warning:.4f})."
                ),
            )
        ]
    return []


def _resource_pressure_budget_flags(
    *,
    scope: str,
    run: Mapping[str, object],
    profile: GateProfile,
) -> list[dict[str, object]]:
    threshold = profile.min_plant_energy_available_per_land_tile_warning
    if threshold is None:
        return []
    resource_pressure = run.get("resource_pressure")
    if not isinstance(resource_pressure, Mapping):
        return [
            _flag(
                "warning",
                scope,
                "resource_pressure",
                "Run is missing resource-pressure analytics for plant-budget review.",
            )
        ]
    plant_budget = resource_pressure.get("plant_budget")
    if not isinstance(plant_budget, Mapping):
        return [
            _flag(
                "warning",
                scope,
                "resource_pressure.plant_budget",
                "Run is missing plant-budget analytics for resource-pressure review.",
            )
        ]
    available_raw = plant_budget.get("energy_available_at_end")
    land_tiles_raw = run.get("land_tile_count")
    if (
        isinstance(available_raw, bool)
        or not isinstance(available_raw, (int, float))
        or isinstance(land_tiles_raw, bool)
        or not isinstance(land_tiles_raw, int)
        or land_tiles_raw <= 0
    ):
        return [
            _flag(
                "warning",
                scope,
                "resource_pressure.plant_budget.energy_available_at_end",
                "Run has incomplete plant-budget analytics for resource-pressure review.",
            )
        ]
    available_per_land_tile = float(available_raw) / land_tiles_raw
    if available_per_land_tile >= threshold:
        return []
    return [
        _flag(
            "warning",
            scope,
            "resource_pressure.plant_budget.energy_available_per_land_tile",
            (
                "Ending plant budget is below the release review floor "
                f"({available_per_land_tile:.4f} < {threshold:.4f})."
            ),
        )
    ]


def _terminal_selection_signal_flags(
    *,
    scope: str,
    selection: object,
    profile: GateProfile,
) -> list[dict[str, object]]:
    threshold = profile.min_terminal_selection_abs_mean_delta_warning
    if threshold is None:
        return []
    if not isinstance(selection, Mapping):
        return [
            _flag(
                "warning",
                scope,
                "selection_heredity",
                "Run is missing terminal selection/heredity analytics.",
            )
        ]
    deltas = selection.get("terminal_minus_initial_mean")
    if not isinstance(deltas, Mapping):
        return [
            _flag(
                "warning",
                scope,
                "selection_heredity.terminal_minus_initial_mean",
                "Run is missing terminal-minus-initial trait deltas.",
            )
        ]
    numeric_deltas = [
        abs(float(value))
        for value in deltas.values()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    ]
    if not numeric_deltas:
        return [
            _flag(
                "warning",
                scope,
                "selection_heredity.terminal_minus_initial_mean",
                "Run did not produce a numeric terminal selection signal.",
            )
        ]
    max_abs_delta = max(numeric_deltas)
    if max_abs_delta >= threshold:
        return []
    return [
        _flag(
            "warning",
            scope,
            "selection_heredity.terminal_minus_initial_mean",
            (
                "Terminal trait distribution is effectively unchanged from the "
                f"initial population (max_abs_delta={max_abs_delta:.4f} < "
                f"{threshold:.4f})."
            ),
        )
    ]
