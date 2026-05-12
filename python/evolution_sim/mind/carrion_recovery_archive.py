from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.cli.mind_v3_evaluate import _round
from evolution_sim.mind.carrion_branch_explore import (
    DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    build_carrion_branch_explore_report,
)
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
)
from evolution_sim.mind.outcome_metrics import aggregate_run_outcome_metrics
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION = (
    "mind_v3_carrion_recovery_archive_v1"
)
MIND_V3_CARRION_RECOVERY_ARCHIVE_POLICY = (
    "quality_diverse_post_contact_recovery_archive_v1"
)
MIND_V3_CARRION_RECOVERY_DATASET_RECORD_SCHEMA_VERSION = (
    "mind_v3_carrion_recovery_dataset_record_v1"
)
DEFAULT_RECOVERY_ARCHIVE_MAX_DATASET_RECORDS_PER_CLASS = 8
DEFAULT_RECOVERY_ARCHIVE_MIN_SURVIVOR_CELLS = 2
DEFAULT_RECOVERY_ARCHIVE_MIN_FAILURE_CELLS = 1


class CarrionRecoveryArchiveError(ValueError):
    pass


def load_carrion_recovery_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    with _open_input(resolved) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CarrionRecoveryArchiveError(f"report must be a JSON object: {resolved}")
    return payload


def build_carrion_recovery_archive_report(
    *,
    branch_report: Mapping[str, object] | None = None,
    branch_report_path: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    ticks: int = DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    base_script: str = DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    continuation_scripts: Sequence[str] = DEFAULT_COUNTERFACTUAL_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = 0,
    trajectory_output_dir: str | Path | None = None,
    verify_replay: bool = True,
    dataset_output_path: str | Path | None = None,
    max_dataset_records_per_class: int = (
        DEFAULT_RECOVERY_ARCHIVE_MAX_DATASET_RECORDS_PER_CLASS
    ),
    min_survivor_cells: int = DEFAULT_RECOVERY_ARCHIVE_MIN_SURVIVOR_CELLS,
    min_failure_cells: int = DEFAULT_RECOVERY_ARCHIVE_MIN_FAILURE_CELLS,
) -> dict[str, object]:
    if branch_report is None:
        if branch_report_path is not None:
            branch_report = load_carrion_recovery_json_report(branch_report_path)
        else:
            branch_report = build_carrion_branch_explore_report(
                seeds=seeds,
                ticks=ticks,
                base_script=base_script,
                continuation_scripts=continuation_scripts,
                max_branch_points_per_seed=max_branch_points_per_seed,
                min_branch_tick=min_branch_tick,
                trajectory_output_dir=trajectory_output_dir,
                verify_replay=verify_replay,
            )
    _validate_branch_report(branch_report)
    max_records = _positive_int(
        max_dataset_records_per_class,
        field="max_dataset_records_per_class",
    )
    min_survivors = _nonnegative_int(
        min_survivor_cells,
        field="min_survivor_cells",
    )
    min_failures = _nonnegative_int(
        min_failure_cells,
        field="min_failure_cells",
    )
    branch_runs = _branch_runs(branch_report)
    branch_digest = stable_payload_digest(
        {
            "schema_version": branch_report.get("schema_version"),
            "contract": branch_report.get("contract"),
            "branch_points": branch_report.get("branch_points"),
            "branch_runs": branch_runs,
        }
    )
    contract = _archive_contract(
        branch_report=branch_report,
        max_dataset_records_per_class=max_records,
        min_survivor_cells=min_survivors,
        min_failure_cells=min_failures,
    )
    cells = _archive_cells(branch_runs)
    dataset_records = _balanced_dataset_records(
        cells,
        max_dataset_records_per_class=max_records,
    )
    if dataset_output_path is not None:
        write_carrion_recovery_dataset_records(dataset_records, dataset_output_path)
    aggregate = _archive_aggregate(cells, branch_runs, dataset_records)
    acceptance = _archive_acceptance(
        aggregate,
        branch_report=branch_report,
        min_survivor_cells=min_survivors,
        min_failure_cells=min_failures,
    )
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        "archive_policy": MIND_V3_CARRION_RECOVERY_ARCHIVE_POLICY,
        "archive_contract": contract,
        "provenance": {
            "archive_contract_digest": stable_payload_digest(contract),
            "source_branch_report_digest": branch_digest,
        },
        "source": {
            "branch_report_path": (
                str(branch_report_path) if branch_report_path is not None else None
            ),
            "branch_schema_version": branch_report.get("schema_version"),
            "branch_policy": branch_report.get("branch_policy"),
            "branch_acceptance": branch_report.get("acceptance"),
        },
        "archive": {
            "cell_count": len(cells),
            "cells": cells,
        },
        "dataset": {
            "schema_version": MIND_V3_CARRION_RECOVERY_DATASET_RECORD_SCHEMA_VERSION,
            "output_path": (
                str(dataset_output_path) if dataset_output_path is not None else None
            ),
            "record_count": len(dataset_records),
            "survivor_count": sum(
                1 for record in dataset_records if record["label"]["terminal_survivor"]
            ),
            "failure_count": sum(
                1
                for record in dataset_records
                if not record["label"]["terminal_survivor"]
            ),
            "records": dataset_records,
        },
        "aggregate": aggregate,
        "acceptance": acceptance,
    }


def write_carrion_recovery_archive_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_carrion_recovery_dataset_records(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _archive_contract(
    *,
    branch_report: Mapping[str, object],
    max_dataset_records_per_class: int,
    min_survivor_cells: int,
    min_failure_cells: int,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_RECOVERY_ARCHIVE_POLICY,
        "source_branch_schema_version": branch_report.get("schema_version"),
        "descriptor_policy": (
            "branch_tick_band + contact_energy_bin + contact_hydration_bin + "
            "resource_gain_bin + terminal_alive_bin + births_bin + "
            "continuation_script + dominant_requested_action"
        ),
        "quality_score_policy": (
            "terminal_alive*100 + births*10 + unique_actions*2 - "
            "dominant_action_share*5 - heuristic_action_count*1000"
        ),
        "dataset_selection_policy": (
            "best_elite_per_descriptor_cell_balanced_by_survivor_failure_class_v1"
        ),
        "max_dataset_records_per_class": int(max_dataset_records_per_class),
        "min_survivor_cells": int(min_survivor_cells),
        "min_failure_cells": int(min_failure_cells),
    }


def _validate_branch_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION:
        raise CarrionRecoveryArchiveError("branch report has stale schema_version")
    runs = report.get("branch_runs")
    if not isinstance(runs, list) or not runs:
        raise CarrionRecoveryArchiveError("branch report must include branch_runs")


def _branch_runs(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        run
        for run in list(report.get("branch_runs", []))
        if isinstance(run, Mapping)
    ]


def _archive_cells(
    branch_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for run in branch_runs:
        descriptor = _descriptor(run)
        key = _descriptor_key(descriptor)
        grouped.setdefault(key, []).append(dict(run))
        descriptors[key] = descriptor
    cells = []
    for key, runs in sorted(grouped.items()):
        ranked = sorted(runs, key=_run_quality_sort_key)
        elite = ranked[0]
        cells.append(
            {
                "cell_key": key,
                "descriptor": descriptors[key],
                "run_count": len(runs),
                "outcome_class": _outcome_class(elite),
                "quality_score": _quality_score(elite),
                "elite": _elite_payload(elite),
            }
        )
    return cells


def _balanced_dataset_records(
    cells: Sequence[Mapping[str, object]],
    *,
    max_dataset_records_per_class: int,
) -> list[dict[str, object]]:
    survivors = [
        cell for cell in cells if str(cell.get("outcome_class")) == "survivor"
    ]
    failures = [
        cell for cell in cells if str(cell.get("outcome_class")) == "failure"
    ]
    selected = _select_dataset_cells(
        survivors,
        limit=max_dataset_records_per_class,
    ) + _select_dataset_cells(
        failures,
        limit=max_dataset_records_per_class,
    )
    selected.sort(
        key=lambda cell: (
            str(cell.get("outcome_class")),
            str(cell.get("cell_key")),
        )
    )
    records = []
    for cell in selected:
        elite = cell.get("elite")
        if not isinstance(elite, Mapping):
            continue
        records.append(_dataset_record(cell, elite))
    return records


def _select_dataset_cells(
    cells: Sequence[Mapping[str, object]],
    *,
    limit: int,
) -> list[Mapping[str, object]]:
    ranked = sorted(
        cells,
        key=lambda cell: (
            -float(cell.get("quality_score", 0.0)),
            str(cell.get("cell_key")),
        ),
    )
    return ranked[:limit]


def _dataset_record(
    cell: Mapping[str, object],
    elite: Mapping[str, object],
) -> dict[str, object]:
    terminal_survivor = str(cell.get("outcome_class")) == "survivor"
    payload = {
        "schema_version": MIND_V3_CARRION_RECOVERY_DATASET_RECORD_SCHEMA_VERSION,
        "source": {
            "branch_id": elite.get("branch_id"),
            "seed": elite.get("seed"),
            "fixture": elite.get("fixture"),
            "branch_tick": elite.get("branch_tick"),
            "base_script": elite.get("base_script"),
            "continuation_script": elite.get("continuation_script"),
            "trajectory_path": elite.get("trajectory_path"),
        },
        "descriptor": cell.get("descriptor"),
        "label": {
            "outcome_class": cell.get("outcome_class"),
            "terminal_survivor": terminal_survivor,
            "alive_agents": elite.get("alive_agents"),
            "births": elite.get("births"),
            "zero_heuristic_runtime_actions": elite.get(
                "zero_heuristic_runtime_actions"
            ),
        },
        "metrics": {
            "quality_score": cell.get("quality_score"),
            "dominant_requested_action": elite.get("dominant_requested_action"),
            "dominant_requested_action_share": elite.get(
                "dominant_requested_action_share"
            ),
            "unique_requested_actions": elite.get("unique_requested_actions"),
            "heuristic_action_source_count": elite.get(
                "heuristic_action_source_count"
            ),
        },
        "outcome_metrics": elite.get("outcome_metrics"),
    }
    payload["record_id"] = stable_payload_digest(payload)
    return payload


def _archive_aggregate(
    cells: Sequence[Mapping[str, object]],
    branch_runs: Sequence[Mapping[str, object]],
    dataset_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    survivor_cells = [
        cell for cell in cells if str(cell.get("outcome_class")) == "survivor"
    ]
    failure_cells = [
        cell for cell in cells if str(cell.get("outcome_class")) == "failure"
    ]
    seeds_with_survivor = sorted(
        {
            int(cell["elite"]["seed"])
            for cell in survivor_cells
            if isinstance(cell.get("elite"), Mapping)
        }
    )
    descriptor_counts = Counter(
        str(cell.get("outcome_class", "unknown")) for cell in cells
    )
    elite_runs = [
        dict(cell["elite"])
        for cell in cells
        if isinstance(cell.get("elite"), Mapping)
    ]
    return {
        "source_branch_run_count": len(branch_runs),
        "cell_count": len(cells),
        "survivor_cell_count": len(survivor_cells),
        "failure_cell_count": len(failure_cells),
        "seeds_with_survivor_cells": seeds_with_survivor,
        "outcome_class_counts": dict(sorted(descriptor_counts.items())),
        "dataset_record_count": len(dataset_records),
        "dataset_survivor_count": sum(
            1
            for record in dataset_records
            if bool(dict(record.get("label", {})).get("terminal_survivor", False))
        ),
        "dataset_failure_count": sum(
            1
            for record in dataset_records
            if not bool(dict(record.get("label", {})).get("terminal_survivor", False))
        ),
        "source_outcome_metrics": aggregate_run_outcome_metrics(branch_runs),
        "outcome_metrics": aggregate_run_outcome_metrics(elite_runs),
        "best_survivor_elite": _best_elite(survivor_cells),
        "best_failure_elite": _best_elite(failure_cells),
    }


def _archive_acceptance(
    aggregate: Mapping[str, object],
    *,
    branch_report: Mapping[str, object],
    min_survivor_cells: int,
    min_failure_cells: int,
) -> dict[str, object]:
    branch_acceptance = branch_report.get("acceptance")
    branch_acceptance_payload = (
        branch_acceptance if isinstance(branch_acceptance, Mapping) else {}
    )
    branch_passed = bool(
        branch_acceptance_payload.get("diagnostic_acceptance_passed", False)
    )
    branch_aggregate = branch_report.get("aggregate")
    branch_aggregate_payload = (
        branch_aggregate if isinstance(branch_aggregate, Mapping) else {}
    )
    replay_verified = bool(branch_aggregate_payload.get("replay_verified"))
    survivor_cells = int(aggregate.get("survivor_cell_count", 0))
    failure_cells = int(aggregate.get("failure_cell_count", 0))
    dataset_survivors = int(aggregate.get("dataset_survivor_count", 0))
    dataset_failures = int(aggregate.get("dataset_failure_count", 0))
    blockers = []
    if not branch_passed:
        blockers.append("source_branch_report_not_accepted")
    if not replay_verified:
        blockers.append("source_branch_replay_not_verified")
    if survivor_cells < min_survivor_cells:
        blockers.append("insufficient_survivor_descriptor_cells")
    if failure_cells < min_failure_cells:
        blockers.append("insufficient_failure_descriptor_cells")
    if dataset_survivors <= 0:
        blockers.append("dataset_missing_survivor_records")
    if min_failure_cells > 0 and dataset_failures <= 0:
        blockers.append("dataset_missing_failure_records")
    return {
        "archive_acceptance_passed": not blockers,
        "blockers": blockers,
        "source_branch_acceptance_passed": branch_passed,
        "source_branch_replay_verified": replay_verified,
        "min_survivor_cells": min_survivor_cells,
        "min_failure_cells": min_failure_cells,
        "survivor_cell_count": survivor_cells,
        "failure_cell_count": failure_cells,
        "dataset_survivor_count": dataset_survivors,
        "dataset_failure_count": dataset_failures,
    }


def _descriptor(run: Mapping[str, object]) -> dict[str, object]:
    contact = run.get("contact")
    contact_payload = contact if isinstance(contact, Mapping) else {}
    after = contact_payload.get("after")
    after_payload = after if isinstance(after, Mapping) else {}
    gained_energy = _finite_float(contact_payload.get("gained_energy"))
    return {
        "branch_tick_band": _tick_band(_int_value(run.get("branch_tick"))),
        "contact_energy_bin": _ratio_bin(
            _finite_float(after_payload.get("energy_ratio")),
            low=0.35,
            high=0.70,
        ),
        "contact_hydration_bin": _ratio_bin(
            _finite_float(after_payload.get("hydration_ratio")),
            low=0.35,
            high=0.75,
        ),
        "resource_gain_bin": _resource_gain_bin(gained_energy),
        "terminal_alive_bin": _alive_bin(_int_value(run.get("alive_agents"))),
        "births_bin": _births_bin(_int_value(run.get("births"))),
        "continuation_script": str(run.get("continuation_script", "unknown")),
        "dominant_requested_action": str(
            run.get("dominant_requested_action", "unknown")
        ),
    }


def _descriptor_key(descriptor: Mapping[str, object]) -> str:
    fields = (
        "branch_tick_band",
        "contact_energy_bin",
        "contact_hydration_bin",
        "resource_gain_bin",
        "terminal_alive_bin",
        "births_bin",
        "continuation_script",
        "dominant_requested_action",
    )
    return "|".join(str(descriptor[field]) for field in fields)


def _quality_score(run: Mapping[str, object]) -> float:
    alive = _int_value(run.get("alive_agents"))
    births = _int_value(run.get("births"))
    unique_actions = _int_value(run.get("unique_requested_actions"))
    dominant_share = _finite_float(run.get("dominant_requested_action_share"))
    heuristic_count = _int_value(run.get("heuristic_action_source_count"))
    return _round(
        alive * 100.0
        + births * 10.0
        + unique_actions * 2.0
        - dominant_share * 5.0
        - heuristic_count * 1000.0
    )


def _run_quality_sort_key(run: Mapping[str, object]) -> tuple[float, str]:
    return (-_quality_score(run), _run_id(run))


def _elite_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "run_id": _run_id(run),
        "branch_id": run.get("branch_id"),
        "seed": _int_value(run.get("seed")),
        "fixture": run.get("fixture"),
        "branch_tick": _int_value(run.get("branch_tick")),
        "base_script": run.get("base_script"),
        "continuation_script": run.get("continuation_script"),
        "alive_agents": _int_value(run.get("alive_agents")),
        "births": _int_value(run.get("births")),
        "deaths": _int_value(run.get("deaths")),
        "dominant_requested_action": run.get("dominant_requested_action"),
        "dominant_requested_action_share": _round(
            _finite_float(run.get("dominant_requested_action_share"))
        ),
        "unique_requested_actions": _int_value(run.get("unique_requested_actions")),
        "heuristic_action_source_count": _int_value(
            run.get("heuristic_action_source_count")
        ),
        "zero_heuristic_runtime_actions": bool(
            run.get("zero_heuristic_runtime_actions", False)
        ),
        "trajectory_path": run.get("trajectory_path"),
        "outcome_metrics": run.get("outcome_metrics"),
        "quality_score": _quality_score(run),
    }


def _best_elite(cells: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if not cells:
        return None
    selected = sorted(
        cells,
        key=lambda cell: (
            -float(cell.get("quality_score", 0.0)),
            str(cell.get("cell_key")),
        ),
    )[0]
    elite = selected.get("elite")
    return dict(elite) if isinstance(elite, Mapping) else None


def _outcome_class(run: Mapping[str, object]) -> str:
    if _int_value(run.get("alive_agents")) > 0 and _int_value(
        run.get("heuristic_action_source_count")
    ) == 0:
        return "survivor"
    return "failure"


def _run_id(run: Mapping[str, object]) -> str:
    return f"{run.get('branch_id')}::{run.get('continuation_script')}"


def _tick_band(tick: int) -> str:
    if tick < 20:
        return "early"
    if tick < 80:
        return "middle"
    return "late"


def _ratio_bin(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "mid"
    return "high"


def _resource_gain_bin(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value < 0.05:
        return "trace"
    if value < 0.2:
        return "small"
    if value < 0.5:
        return "medium"
    return "large"


def _alive_bin(alive: int) -> str:
    if alive <= 0:
        return "dead"
    if alive == 1:
        return "one"
    if alive <= 4:
        return "few"
    return "many"


def _births_bin(births: int) -> str:
    if births <= 0:
        return "none"
    if births <= 2:
        return "low"
    if births <= 8:
        return "medium"
    return "high"


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return 0.0
    return parsed


def _positive_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise CarrionRecoveryArchiveError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise CarrionRecoveryArchiveError(f"{field} must be nonnegative")
    return parsed


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
