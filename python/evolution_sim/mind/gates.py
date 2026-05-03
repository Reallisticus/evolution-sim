from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_mind_v1_gate_report(
    policy_report: Mapping[str, object],
    *,
    max_invalid_action_rate: float = 0.02,
    min_viable_run_share: float = 1.0,
    min_births_per_run_mean: float = 0.0,
    min_plant_energy_available_per_land_tile: float = 0.02,
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
        invalid_rate = float(trajectory.get("invalid_action_rate", 1.0))
        if invalid_rate > max_invalid_action_rate:
            flags.append(
                _flag(
                    "error",
                    "policy",
                    "trajectory.invalid_action_rate",
                    (
                        f"Invalid-action rate {invalid_rate:.4f} exceeds "
                        f"{max_invalid_action_rate:.4f}."
                    ),
                )
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
