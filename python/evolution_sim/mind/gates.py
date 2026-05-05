from __future__ import annotations

from collections.abc import Mapping, Sequence


MIND_V1_GATE_CRITERIA_DEFAULTS: dict[str, float] = {
    "max_alive_agents_mean_regression": 0.0,
    "max_alive_agents_per_seed_regression": 2.0,
    "max_births_mean_regression": 0.0,
    "max_births_per_seed_regression": 1.0,
    "max_invalid_action_rate": 0.02,
    "max_guard_intervention_rate": 0.45,
    "max_guard_intervention_rate_by_group": 0.5,
    "min_guard_intervention_rate_reduction": 0.0,
    "min_viable_run_share": 1.0,
    "min_births_per_run_mean": 0.0,
    "min_plant_energy_available_per_land_tile": 0.02,
}


def normalize_mind_v1_gate_criteria(
    overrides: Mapping[str, object] | None = None,
) -> dict[str, float]:
    criteria = dict(MIND_V1_GATE_CRITERIA_DEFAULTS)
    if overrides is None:
        return criteria
    for key, value in overrides.items():
        if key not in criteria:
            raise ValueError(f"unknown Mind v1 gate criterion: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Mind v1 gate criterion {key} must be numeric")
        criteria[key] = float(value)
    return criteria


def build_mind_v1_gate_report(
    policy_report: Mapping[str, object],
    *,
    baseline_report: Mapping[str, object] | None = None,
    max_alive_agents_mean_regression: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_alive_agents_mean_regression"
    ],
    max_alive_agents_per_seed_regression: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_alive_agents_per_seed_regression"
    ],
    max_births_mean_regression: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_births_mean_regression"
    ],
    max_births_per_seed_regression: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_births_per_seed_regression"
    ],
    max_invalid_action_rate: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_invalid_action_rate"
    ],
    max_guard_intervention_rate: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_guard_intervention_rate"
    ],
    max_guard_intervention_rate_by_group: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "max_guard_intervention_rate_by_group"
    ],
    min_guard_intervention_rate_reduction: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "min_guard_intervention_rate_reduction"
    ],
    reference_guard_intervention_rate: float | None = None,
    min_viable_run_share: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "min_viable_run_share"
    ],
    min_births_per_run_mean: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "min_births_per_run_mean"
    ],
    min_plant_energy_available_per_land_tile: float = MIND_V1_GATE_CRITERIA_DEFAULTS[
        "min_plant_energy_available_per_land_tile"
    ],
) -> dict[str, object]:
    flags: list[dict[str, object]] = []
    runs = policy_report.get("runs")
    if not isinstance(runs, Sequence):
        runs = []
    aggregate = policy_report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    trajectory = aggregate.get("trajectory")
    if not isinstance(trajectory, Mapping):
        flags.append(_flag("error", "policy", "trajectory", "Missing trajectory aggregate."))
    else:
        invalid_rate = float(
            trajectory.get(
                "invalid_observation_action_rate",
                trajectory.get("invalid_action_rate", 1.0),
            )
        )
        if invalid_rate > max_invalid_action_rate:
            flags.append(
                _flag(
                    "error",
                    "policy",
                    "trajectory.invalid_observation_action_rate",
                    (
                        f"Invalid observation-action rate {invalid_rate:.4f} exceeds "
                        f"{max_invalid_action_rate:.4f}."
                    ),
                )
            )

    _append_guard_intervention_flags(
        flags,
        aggregate=aggregate,
        max_guard_intervention_rate=max_guard_intervention_rate,
        max_guard_intervention_rate_by_group=max_guard_intervention_rate_by_group,
        min_guard_intervention_rate_reduction=min_guard_intervention_rate_reduction,
        reference_guard_intervention_rate=reference_guard_intervention_rate,
    )

    run_count = max(len(runs), 1)
    viable_runs = sum(
        1
        for run in runs
        if isinstance(run, Mapping) and int(run.get("alive_agents", 0)) > 0
    )
    viable_share = viable_runs / run_count
    if viable_share < min_viable_run_share:
        flags.append(
            _flag(
                "error",
                "policy",
                "alive_agents",
                (
                    f"Viable-run share {viable_share:.4f} is below "
                    f"{min_viable_run_share:.4f}."
                ),
            )
        )

    births_mean = _aggregate_stat_mean(aggregate, "births")
    if births_mean is None or births_mean < min_births_per_run_mean:
        flags.append(
            _flag(
                "warning",
                "policy",
                "births.mean",
                "Policy did not meet the Mind v1 reproduction review floor.",
            )
        )

    if baseline_report is not None:
        baseline_aggregate = baseline_report.get("aggregate")
        if isinstance(baseline_aggregate, Mapping):
            alive_delta = _aggregate_mean_delta(
                aggregate,
                baseline_aggregate,
                "alive_agents",
            )
            if (
                alive_delta is not None
                and alive_delta < -max_alive_agents_mean_regression
            ):
                flags.append(
                    _flag(
                        "error",
                        "policy_vs_heuristic",
                        "alive_agents.mean_delta",
                        (
                            "Policy regressed terminal alive agents versus the "
                            f"heuristic baseline ({alive_delta:.4f})."
                        ),
                    )
                )
            births_delta = _aggregate_mean_delta(aggregate, baseline_aggregate, "births")
            if (
                births_delta is not None
                and births_delta < -max_births_mean_regression
            ):
                flags.append(
                    _flag(
                        "warning",
                        "policy_vs_heuristic",
                        "births.mean_delta",
                        (
                            "Policy regressed births versus the heuristic baseline "
                            f"({births_delta:.4f})."
                        ),
                    )
                )
            baseline_runs = baseline_report.get("runs")
            if isinstance(baseline_runs, Sequence):
                _append_per_seed_regression_flags(
                    flags,
                    policy_runs=runs,
                    baseline_runs=baseline_runs,
                    max_alive_agents_per_seed_regression=(
                        max_alive_agents_per_seed_regression
                    ),
                    max_births_per_seed_regression=max_births_per_seed_regression,
                )

    min_plant_available = _min_plant_available_per_land_tile(runs)
    if min_plant_available is None:
        flags.append(
            _flag(
                "warning",
                "policy",
                "resource_pressure.plant_budget",
                "Policy report is missing resource-pressure plant-budget analytics.",
            )
        )
    elif min_plant_available < min_plant_energy_available_per_land_tile:
        flags.append(
            _flag(
                "warning",
                "policy",
                "resource_pressure.plant_budget.energy_available_per_land_tile",
                (
                    "Plant budget fell below the Mind v1 review floor "
                    f"({min_plant_available:.4f} < "
                    f"{min_plant_energy_available_per_land_tile:.4f})."
                ),
            )
        )

    blockers = [flag for flag in flags if flag["severity"] == "error"]
    warnings = [flag for flag in flags if flag["severity"] == "warning"]
    return {
        "status": "fail" if blockers else ("review" if warnings else "pass"),
        "blockers": blockers,
        "warnings": warnings,
    }


def _aggregate_stat_mean(
    aggregate: Mapping[str, object],
    field: str,
) -> float | None:
    stats = aggregate.get(field)
    if not isinstance(stats, Mapping):
        return None
    mean = stats.get("mean")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)):
        return None
    return float(mean)


def _aggregate_mean_delta(
    left: Mapping[str, object],
    right: Mapping[str, object],
    field: str,
) -> float | None:
    left_mean = _aggregate_stat_mean(left, field)
    right_mean = _aggregate_stat_mean(right, field)
    if left_mean is None or right_mean is None:
        return None
    return round(left_mean - right_mean, 4)


def _min_plant_available_per_land_tile(
    runs: Sequence[object],
) -> float | None:
    values: list[float] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        resource_pressure = run.get("resource_pressure")
        if not isinstance(resource_pressure, Mapping):
            continue
        plant_budget = resource_pressure.get("plant_budget")
        if not isinstance(plant_budget, Mapping):
            continue
        available = plant_budget.get("energy_available_at_end")
        land_tile_count = run.get("land_tile_count")
        if (
            isinstance(available, bool)
            or not isinstance(available, (int, float))
            or isinstance(land_tile_count, bool)
            or not isinstance(land_tile_count, int)
            or land_tile_count <= 0
        ):
            continue
        values.append(float(available) / land_tile_count)
    return min(values) if values else None


def _append_guard_intervention_flags(
    flags: list[dict[str, object]],
    *,
    aggregate: Mapping[str, object],
    max_guard_intervention_rate: float,
    max_guard_intervention_rate_by_group: float,
    min_guard_intervention_rate_reduction: float,
    reference_guard_intervention_rate: float | None,
) -> None:
    diagnostics = aggregate.get("policy_diagnostics")
    if not isinstance(diagnostics, Mapping):
        flags.append(
            _flag(
                "error",
                "policy",
                "policy_diagnostics",
                "Policy report is missing Mind policy diagnostics.",
            )
        )
        return
    guard_rate = _mapping_number(diagnostics, "guard_intervention_rate")
    if guard_rate is None:
        flags.append(
            _flag(
                "error",
                "policy",
                "policy_diagnostics.guard_intervention_rate",
                "Policy diagnostics are missing guard intervention rate.",
            )
        )
        return
    if guard_rate > max_guard_intervention_rate:
        flags.append(
            _flag(
                "error",
                "policy",
                "policy_diagnostics.guard_intervention_rate",
                (
                    f"Guard intervention rate {guard_rate:.4f} exceeds "
                    f"{max_guard_intervention_rate:.4f}."
                ),
            )
        )
    if min_guard_intervention_rate_reduction > 0.0:
        if reference_guard_intervention_rate is None:
            flags.append(
                _flag(
                    "warning",
                    "policy",
                    "policy_diagnostics.guard_intervention_rate_reduction",
                    (
                        "Guard intervention reduction criterion is configured, "
                        "but no reference guard intervention rate was supplied."
                    ),
                )
            )
        else:
            reduction = reference_guard_intervention_rate - guard_rate
            if reduction < min_guard_intervention_rate_reduction:
                flags.append(
                    _flag(
                        "error",
                        "policy",
                        "policy_diagnostics.guard_intervention_rate_reduction",
                        (
                            f"Guard intervention rate reduction {reduction:.4f} "
                            f"is below {min_guard_intervention_rate_reduction:.4f}."
                        ),
                    )
                )
    for group_field in ("by_trophic_role", "by_meat_mode"):
        groups = diagnostics.get(group_field)
        if not isinstance(groups, Mapping):
            flags.append(
                _flag(
                    "error",
                    "policy",
                    f"policy_diagnostics.{group_field}",
                    f"Policy diagnostics are missing {group_field} guard rates.",
                )
            )
            continue
        for label, group in groups.items():
            if not isinstance(group, Mapping):
                continue
            group_guard_rate = _mapping_number(group, "guard_intervention_rate")
            if group_guard_rate is None:
                flags.append(
                    _flag(
                        "error",
                        "policy",
                        (
                            f"policy_diagnostics.{group_field}."
                            f"{label}.guard_intervention_rate"
                        ),
                        (
                            f"Policy diagnostics for {group_field} {label!r} "
                            "are missing guard intervention rate."
                        ),
                    )
                )
                continue
            if group_guard_rate > max_guard_intervention_rate_by_group:
                flags.append(
                    _flag(
                        "error",
                        "policy",
                        (
                            f"policy_diagnostics.{group_field}."
                            f"{label}.guard_intervention_rate"
                        ),
                        (
                            f"Guard intervention rate for {group_field} {label!r} "
                            f"{group_guard_rate:.4f} exceeds "
                            f"{max_guard_intervention_rate_by_group:.4f}."
                        ),
                    )
                )


def _mapping_number(
    payload: Mapping[str, object],
    field: str,
) -> float | None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _append_per_seed_regression_flags(
    flags: list[dict[str, object]],
    *,
    policy_runs: Sequence[object],
    baseline_runs: Sequence[object],
    max_alive_agents_per_seed_regression: float,
    max_births_per_seed_regression: float,
) -> None:
    policy_by_seed = _runs_by_seed(policy_runs)
    baseline_by_seed = _runs_by_seed(baseline_runs)
    for seed in sorted(set(policy_by_seed) & set(baseline_by_seed)):
        policy_run = policy_by_seed[seed]
        baseline_run = baseline_by_seed[seed]
        alive_delta = _run_value_delta(policy_run, baseline_run, "alive_agents")
        if (
            alive_delta is not None
            and alive_delta < -max_alive_agents_per_seed_regression
        ):
            flags.append(
                _flag(
                    "error",
                    "policy_vs_heuristic",
                    "alive_agents.per_seed_delta",
                    (
                        "Policy regressed terminal alive agents for validation "
                        f"seed {seed} versus the heuristic baseline "
                        f"({alive_delta:.4f})."
                    ),
                )
            )
        births_delta = _run_value_delta(policy_run, baseline_run, "births")
        if (
            births_delta is not None
            and births_delta < -max_births_per_seed_regression
        ):
            flags.append(
                _flag(
                    "warning",
                    "policy_vs_heuristic",
                    "births.per_seed_delta",
                    (
                        "Policy regressed births for validation seed "
                        f"{seed} versus the heuristic baseline "
                        f"({births_delta:.4f})."
                    ),
                )
            )


def _runs_by_seed(runs: Sequence[object]) -> dict[int, Mapping[str, object]]:
    by_seed: dict[int, Mapping[str, object]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        by_seed[seed] = run
    return by_seed


def _run_value_delta(
    left: Mapping[str, object],
    right: Mapping[str, object],
    field: str,
) -> float | None:
    left_value = _run_numeric_value(left, field)
    right_value = _run_numeric_value(right, field)
    if left_value is None or right_value is None:
        return None
    return round(left_value - right_value, 4)


def _run_numeric_value(
    run: Mapping[str, object],
    field: str,
) -> float | None:
    value = run.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _flag(
    severity: str,
    scope: str,
    field: str,
    message: str,
) -> dict[str, object]:
    return {
        "severity": severity,
        "scope": scope,
        "field": field,
        "message": message,
    }
