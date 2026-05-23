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
    MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
)
from evolution_sim.mind.outcome_metrics import aggregate_run_outcome_metrics
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION = (
    "mind_v3_carrion_recovery_archive_v1"
)
MIND_V3_CARRION_RECOVERY_ARCHIVE_POLICY = (
    "quality_diverse_post_contact_recovery_archive_v1"
)
MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_BRANCH = "branch-continuation"
MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE = (
    "fixture-rerank-recovery-probe"
)
MIND_V3_CARRION_RERANK_RECOVERY_ARCHIVE_AUDIT_POLICY = (
    "fixture_rerank_recovery_probe_archive_audit_v1"
)
MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY = (
    "gate_aligned_carrion_recovery_archive_v1"
)
MIND_V3_CARRION_RECOVERY_DATASET_RECORD_SCHEMA_VERSION = (
    "mind_v3_carrion_recovery_dataset_record_v1"
)
MIND_V3_RERANK_RECOVERY_DOMINANT_ACTION_CAP = 0.5
DEFAULT_RECOVERY_ARCHIVE_MAX_DATASET_RECORDS_PER_CLASS = 8
DEFAULT_RECOVERY_ARCHIVE_MIN_SURVIVOR_CELLS = 2
DEFAULT_RECOVERY_ARCHIVE_MIN_FAILURE_CELLS = 1

_RERANK_RECOVERY_REQUIRED_PROBE_FIELDS = (
    "fixture_blocker_count",
    "carrion_only_blocker_count",
    "post_contact_survival_rate",
    "drink_after_carrion_rate",
    "mean_hydration_delta_after_carrion",
    "mean_water_distance",
    "unsupported_resolved_action_count",
    "dominant_requested_action_share",
)


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
    counterfactual_report: Mapping[str, object] | None = None,
    counterfactual_report_path: str | Path | None = None,
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
    min_counterfactual_survivor_seeds: int = 0,
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
    min_counterfactual_seed_count = _nonnegative_int(
        min_counterfactual_survivor_seeds,
        field="min_counterfactual_survivor_seeds",
    )
    if counterfactual_report is None and counterfactual_report_path is not None:
        counterfactual_report = load_carrion_recovery_json_report(
            counterfactual_report_path
        )
    if counterfactual_report is not None:
        _validate_counterfactual_report(counterfactual_report)
    branch_runs = _branch_runs(branch_report)
    counterfactual_runs = (
        _counterfactual_runs(counterfactual_report)
        if counterfactual_report is not None
        else []
    )
    branch_digest = stable_payload_digest(
        {
            "schema_version": branch_report.get("schema_version"),
            "contract": branch_report.get("contract"),
            "branch_points": branch_report.get("branch_points"),
            "branch_runs": branch_runs,
        }
    )
    counterfactual_digest = (
        stable_payload_digest(
            {
                "schema_version": counterfactual_report.get("schema_version"),
                "counterfactual_contract": counterfactual_report.get(
                    "counterfactual_contract"
                ),
                "runs": counterfactual_runs,
            }
        )
        if counterfactual_report is not None
        else None
    )
    contract = _archive_contract(
        branch_report=branch_report,
        counterfactual_report=counterfactual_report,
        max_dataset_records_per_class=max_records,
        min_survivor_cells=min_survivors,
        min_failure_cells=min_failures,
        min_counterfactual_survivor_seeds=min_counterfactual_seed_count,
    )
    cells = _archive_cells(branch_runs)
    branch_dataset_records = _balanced_dataset_records(
        cells,
        max_dataset_records_per_class=max_records,
    )
    counterfactual_dataset_records = _counterfactual_dataset_records(
        counterfactual_runs
    )
    dataset_records = branch_dataset_records + counterfactual_dataset_records
    if dataset_output_path is not None:
        write_carrion_recovery_dataset_records(dataset_records, dataset_output_path)
    aggregate = _archive_aggregate(
        cells,
        branch_runs,
        dataset_records,
        counterfactual_runs=counterfactual_runs,
        counterfactual_dataset_records=counterfactual_dataset_records,
    )
    acceptance = _archive_acceptance(
        aggregate,
        branch_report=branch_report,
        min_survivor_cells=min_survivors,
        min_failure_cells=min_failures,
        min_counterfactual_survivor_seeds=min_counterfactual_seed_count,
    )
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        "archive_policy": MIND_V3_CARRION_RECOVERY_ARCHIVE_POLICY,
        "archive_contract": contract,
        "provenance": {
            "archive_contract_digest": stable_payload_digest(contract),
            "source_branch_report_digest": branch_digest,
            "source_counterfactual_report_digest": counterfactual_digest,
        },
        "source": {
            "branch_report_path": (
                str(branch_report_path) if branch_report_path is not None else None
            ),
            "branch_schema_version": branch_report.get("schema_version"),
            "branch_policy": branch_report.get("branch_policy"),
            "branch_acceptance": branch_report.get("acceptance"),
            "counterfactual_report_path": (
                str(counterfactual_report_path)
                if counterfactual_report_path is not None
                else None
            ),
            "counterfactual_schema_version": (
                counterfactual_report.get("schema_version")
                if counterfactual_report is not None
                else None
            ),
            "counterfactual_policy": (
                counterfactual_report.get("counterfactual_policy")
                if counterfactual_report is not None
                else None
            ),
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


def build_fixture_rerank_recovery_probe_archive_report(
    *,
    search_reports: Sequence[
        tuple[str, Mapping[str, object], str | Path | None]
    ],
) -> dict[str, object]:
    if not search_reports:
        raise CarrionRecoveryArchiveError(
            "fixture rerank recovery probe archive requires at least one search report"
        )
    source_reports = []
    complete_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    for label, report, path in search_reports:
        path_text = str(path) if path is not None else None
        digest = stable_payload_digest(report)
        source_reports.append(
            {
                "label": str(label),
                "path": path_text,
                "digest": digest,
                "schema_version": report.get("schema_version"),
                "selected_candidate_id": _selected_rerank_candidate_id(report),
            }
        )
        rows, missing = _fixture_rerank_recovery_probe_rows(
            label=str(label),
            report=report,
            path=path_text,
        )
        complete_rows.extend(rows)
        missing_rows.extend(missing)
        selected_ids.update(
            str(row["candidate_id"]) for row in rows if bool(row.get("selected"))
        )
        selected_ids.update(
            str(row["candidate_id"])
            for row in missing
            if bool(row.get("selected"))
        )
    _normalize_rerank_probe_descriptors(complete_rows)
    cells = _fixture_rerank_recovery_probe_archive_cells(complete_rows)
    selected_memberships = _selected_rerank_probe_cell_memberships(
        complete_rows,
        cells,
    )
    selected_rows = [row for row in complete_rows if bool(row.get("selected"))]
    recovery_better = _non_selected_recovery_better_rows(
        complete_rows,
        selected_rows,
    )
    selected_dominates = bool(selected_rows) and not recovery_better
    retained_candidate_ids = _retained_rerank_probe_candidate_ids(
        cells,
        selected_ids=selected_ids,
    )
    important_cells = _fixture_rerank_recovery_important_cells(complete_rows)
    input_summary = _fixture_rerank_recovery_input_summary(
        complete_rows=complete_rows,
        missing_rows=missing_rows,
        selected_ids=selected_ids,
    )
    contract = _fixture_rerank_recovery_archive_contract()
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        "archive_policy": MIND_V3_CARRION_RERANK_RECOVERY_ARCHIVE_AUDIT_POLICY,
        "source_mode": MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
        "report_only": True,
        "changes_runtime_policy": False,
        "changes_evolve_selection": False,
        "archive_contract": contract,
        "provenance": {
            "archive_contract_digest": stable_payload_digest(contract),
            "source_search_report_count": len(source_reports),
            "source_search_reports": source_reports,
            "combined_source_digest": stable_payload_digest(source_reports),
        },
        "input_summary": input_summary,
        "archive": {
            "cell_count": len(cells),
            "cells": cells,
            "empty_important_cells": [
                cell for cell in important_cells if bool(cell.get("empty"))
            ],
            "important_cells": important_cells,
            "elite_candidate_ids": _elite_candidate_ids(cells),
        },
        "selected": {
            "candidate_ids": sorted(selected_ids),
            "candidate_cell_memberships": selected_memberships,
            "dominates_all_non_selected_recovery_cells": selected_dominates,
        },
        "retention": {
            "report_only": True,
            "would_add_new_parent_candidates": bool(retained_candidate_ids),
            "retained_candidate_ids_that_would_be_added": retained_candidate_ids,
            "non_selected_recovery_better_than_selected": bool(recovery_better),
            "recovery_better_candidate_ids": [
                str(row["candidate_id"]) for row in recovery_better
            ],
            "non_selected_cell_retention_justified_by_recovery_better": (
                bool(recovery_better)
            ),
            "non_selected_cell_retention_justified_by_diversity": (
                bool(retained_candidate_ids)
            ),
        },
        "candidates": {
            "complete": complete_rows,
            "missing": missing_rows,
        },
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


def _fixture_rerank_recovery_archive_contract() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_RERANK_RECOVERY_ARCHIVE_AUDIT_POLICY,
        "source_mode": MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
        "source_fields": [
            "fixture_rerank.candidates[*].carrion_recovery_probe",
            "fixture_rerank.candidates[*].fixture_repair",
            "fixture_rerank.selected_candidate_id",
        ],
        "report_only": True,
        "runtime_policy_effect": "none",
        "evolve_selection_effect": "none",
        "descriptor_policy": (
            "post_contact_survival_niche + drink_after_carrion_niche + "
            "hydration_recovery_niche + water_distance_niche + "
            "unsupported_resolution_niche + dominant_action_cap + "
            "carrion_only_blocker_niche + fewer_blocker_lane + source_type"
        ),
        "elite_selection_policy": (
            "lexicographic recovery quality: fewer fixture blockers, fewer "
            "carrion_only blockers, higher post-contact survival, higher "
            "drink-after-carrion rate, higher hydration delta, shorter water "
            "distance, fewer unsupported resolved actions, dominant action cap "
            "pass, lower dominant share, existing search score as final context"
        ),
        "dominant_action_cap": MIND_V3_RERANK_RECOVERY_DOMINANT_ACTION_CAP,
        "retention_policy": (
            "report-only cell elite retention by default; active evolve "
            f"parent/archive retention requires explicit opt-in to "
            f"{MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY}"
        ),
    }


def _selected_rerank_candidate_id(report: Mapping[str, object]) -> str | None:
    rerank = _mapping(report.get("fixture_rerank"))
    candidate_id = rerank.get("selected_candidate_id")
    return str(candidate_id) if isinstance(candidate_id, str) and candidate_id else None


def _fixture_rerank_recovery_probe_rows(
    *,
    label: str,
    report: Mapping[str, object],
    path: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rerank = _mapping(report.get("fixture_rerank"))
    selected_candidate_id = str(rerank.get("selected_candidate_id") or "")
    candidate_items = rerank.get("candidates")
    candidates = candidate_items if isinstance(candidate_items, list) else []
    complete: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for index, raw_candidate in enumerate(candidates):
        if not isinstance(raw_candidate, Mapping):
            missing.append(
                {
                    "label": label,
                    "report_path": path,
                    "candidate_index": index,
                    "candidate_id": "",
                    "selected": False,
                    "missing_reason": "candidate_not_object",
                    "missing_fields": ["candidate"],
                }
            )
            continue
        candidate = raw_candidate
        candidate_id = str(candidate.get("candidate_id") or "")
        selected = bool(candidate_id and candidate_id == selected_candidate_id)
        probe = _candidate_recovery_probe(candidate)
        if not probe:
            missing.append(
                _missing_rerank_probe_row(
                    label=label,
                    path=path,
                    candidate=candidate,
                    index=index,
                    selected=selected,
                    reason="probe_missing",
                    missing_fields=["carrion_recovery_probe"],
                )
            )
            continue
        missing_fields = _missing_rerank_probe_fields(probe)
        available = probe.get("available")
        if available is False:
            reason = probe.get("missing_reason")
            missing.append(
                _missing_rerank_probe_row(
                    label=label,
                    path=path,
                    candidate=candidate,
                    index=index,
                    selected=selected,
                    reason=(
                        str(reason)
                        if isinstance(reason, str) and reason
                        else "probe_unavailable"
                    ),
                    missing_fields=missing_fields or ["available"],
                )
            )
            continue
        if missing_fields:
            missing.append(
                _missing_rerank_probe_row(
                    label=label,
                    path=path,
                    candidate=candidate,
                    index=index,
                    selected=selected,
                    reason="probe_missing_required_fields",
                    missing_fields=missing_fields,
                )
            )
            continue
        complete.append(
            _complete_rerank_probe_row(
                label=label,
                path=path,
                candidate=candidate,
                probe=probe,
                index=index,
                selected=selected,
            )
        )
    complete.sort(
        key=lambda row: (
            str(row.get("report_label")),
            int(row.get("candidate_index", 0)),
            str(row.get("candidate_id")),
        )
    )
    missing.sort(
        key=lambda row: (
            str(row.get("report_label")),
            int(row.get("candidate_index", 0)),
            str(row.get("candidate_id")),
            str(row.get("missing_reason")),
        )
    )
    return complete, missing


def _candidate_recovery_probe(
    candidate: Mapping[str, object],
) -> Mapping[str, object]:
    for key in (
        "carrion_recovery_probe",
        "gate_aligned_carrion_recovery_probe_v1",
    ):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _missing_rerank_probe_fields(probe: Mapping[str, object]) -> list[str]:
    missing = [
        field
        for field in _RERANK_RECOVERY_REQUIRED_PROBE_FIELDS
        if _optional_number(probe.get(field)) is None
    ]
    return missing


def _missing_rerank_probe_row(
    *,
    label: str,
    path: str | None,
    candidate: Mapping[str, object],
    index: int,
    selected: bool,
    reason: str,
    missing_fields: Sequence[str],
) -> dict[str, object]:
    return {
        "report_label": label,
        "report_path": path,
        "candidate_index": int(index),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "selected": selected,
        "source_type": _fixture_rerank_probe_source_type(
            candidate,
            selected=selected,
        ),
        "missing_reason": reason,
        "missing_fields": sorted(str(field) for field in missing_fields),
    }


def _complete_rerank_probe_row(
    *,
    label: str,
    path: str | None,
    candidate: Mapping[str, object],
    probe: Mapping[str, object],
    index: int,
    selected: bool,
) -> dict[str, object]:
    source_type = _fixture_rerank_probe_source_type(candidate, selected=selected)
    origin_source_type = _fixture_rerank_probe_origin_source_type(candidate)
    fixture_blockers = int(_optional_number(probe.get("fixture_blocker_count")) or 0)
    carrion_blockers = int(
        _optional_number(probe.get("carrion_only_blocker_count")) or 0
    )
    survival = float(_optional_number(probe.get("post_contact_survival_rate")) or 0.0)
    drink = float(_optional_number(probe.get("drink_after_carrion_rate")) or 0.0)
    hydration = float(
        _optional_number(probe.get("mean_hydration_delta_after_carrion")) or 0.0
    )
    water = float(_optional_number(probe.get("mean_water_distance")) or 0.0)
    unsupported_requested = int(
        _optional_number(probe.get("unsupported_requested_action_count")) or 0
    )
    unsupported_resolved = int(
        _optional_number(probe.get("unsupported_resolved_action_count")) or 0
    )
    dominant_share = float(
        _optional_number(probe.get("dominant_requested_action_share")) or 0.0
    )
    descriptor = _fixture_rerank_probe_descriptor(
        source_type=source_type,
        fixture_blocker_count=fixture_blockers,
        carrion_only_blocker_count=carrion_blockers,
        selected_carrion_only_blocker_count=carrion_blockers,
        survival=survival,
        drink=drink,
        hydration=hydration,
        water_distance=water,
        unsupported_resolved=unsupported_resolved,
        dominant_share=dominant_share,
    )
    row = {
        "report_label": label,
        "report_path": path,
        "candidate_index": int(index),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "selected": selected,
        "source_type": source_type,
        "origin_source_type": origin_source_type,
        "prefilter_rank": _optional_int(candidate.get("prefilter_rank")),
        "search_score": _optional_number(candidate.get("search_score")),
        "fixture_blocker_count": fixture_blockers,
        "carrion_only_blocker_count": carrion_blockers,
        "post_contact_survival_rate": _round(survival),
        "drink_after_carrion_rate": _round(drink),
        "mean_hydration_delta_after_carrion": _round(hydration),
        "mean_water_distance": _round(water),
        "unsupported_requested_action_count": unsupported_requested,
        "unsupported_resolved_action_count": unsupported_resolved,
        "dominant_requested_action_share": _round(dominant_share),
        "descriptor": descriptor,
        "cell_key": _fixture_rerank_probe_descriptor_key(descriptor),
        "facet_cells": _fixture_rerank_probe_facet_cells(descriptor),
    }
    return row


def _fixture_rerank_probe_source_type(
    candidate: Mapping[str, object],
    *,
    selected: bool,
) -> str:
    if selected:
        return "selected"
    return _fixture_rerank_probe_origin_source_type(candidate)


def _fixture_rerank_probe_origin_source_type(
    candidate: Mapping[str, object],
) -> str:
    repair = candidate.get("fixture_repair")
    if isinstance(repair, Mapping):
        if repair.get("donor_selection_reason") == "promotion_safe_bridge":
            return "bridge-repair"
        return "repair"
    return "initial"


def _fixture_rerank_probe_descriptor(
    *,
    source_type: str,
    fixture_blocker_count: int,
    carrion_only_blocker_count: int,
    selected_carrion_only_blocker_count: int,
    survival: float,
    drink: float,
    hydration: float,
    water_distance: float,
    unsupported_resolved: int,
    dominant_share: float,
) -> dict[str, object]:
    return {
        "post_contact_survival_niche": (
            "nonzero" if survival > 0.0 else "zero"
        ),
        "drink_after_carrion_niche": _drink_after_carrion_niche(drink),
        "hydration_recovery_niche": _hydration_recovery_niche(hydration),
        "water_distance_niche": _water_distance_niche(water_distance),
        "unsupported_resolution_niche": (
            "none" if unsupported_resolved <= 0 else "present"
        ),
        "dominant_action_cap": (
            "pass"
            if dominant_share <= MIND_V3_RERANK_RECOVERY_DOMINANT_ACTION_CAP
            else "fail"
        ),
        "fixture_blocker_count": int(fixture_blocker_count),
        "carrion_only_blocker_count": int(carrion_only_blocker_count),
        "carrion_only_blocker_niche": f"count_{int(carrion_only_blocker_count)}",
        "fewer_blocker_lane": _fewer_blocker_lane(
            carrion_only_blocker_count,
            selected_carrion_only_blocker_count,
        ),
        "source_type": source_type,
    }


def _fixture_rerank_probe_descriptor_key(
    descriptor: Mapping[str, object],
) -> str:
    fields = (
        "post_contact_survival_niche",
        "drink_after_carrion_niche",
        "hydration_recovery_niche",
        "water_distance_niche",
        "unsupported_resolution_niche",
        "dominant_action_cap",
        "carrion_only_blocker_niche",
        "fewer_blocker_lane",
        "source_type",
    )
    return "|".join(str(descriptor[field]) for field in fields)


def _fixture_rerank_probe_facet_cells(
    descriptor: Mapping[str, object],
) -> list[str]:
    return [
        f"{field}:{descriptor[field]}"
        for field in (
            "post_contact_survival_niche",
            "drink_after_carrion_niche",
            "hydration_recovery_niche",
            "water_distance_niche",
            "unsupported_resolution_niche",
            "dominant_action_cap",
            "carrion_only_blocker_niche",
            "fewer_blocker_lane",
            "source_type",
        )
    ]


def _fixture_rerank_recovery_probe_archive_cells(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    selected_carrion_blockers = _selected_carrion_blocker_count(rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for row in rows:
        candidate = dict(row)
        descriptor = _fixture_rerank_probe_descriptor(
            source_type=str(candidate.get("source_type", "initial")),
            fixture_blocker_count=int(candidate.get("fixture_blocker_count", 0)),
            carrion_only_blocker_count=int(
                candidate.get("carrion_only_blocker_count", 0)
            ),
            selected_carrion_only_blocker_count=selected_carrion_blockers,
            survival=float(candidate.get("post_contact_survival_rate", 0.0)),
            drink=float(candidate.get("drink_after_carrion_rate", 0.0)),
            hydration=float(
                candidate.get("mean_hydration_delta_after_carrion", 0.0)
            ),
            water_distance=float(candidate.get("mean_water_distance", 0.0)),
            unsupported_resolved=int(
                candidate.get("unsupported_resolved_action_count", 0)
            ),
            dominant_share=float(
                candidate.get("dominant_requested_action_share", 0.0)
            ),
        )
        key = _fixture_rerank_probe_descriptor_key(descriptor)
        candidate["descriptor"] = descriptor
        candidate["cell_key"] = key
        candidate["facet_cells"] = _fixture_rerank_probe_facet_cells(descriptor)
        grouped.setdefault(key, []).append(candidate)
        descriptors[key] = descriptor
    cells = []
    for key, candidates in sorted(grouped.items()):
        elite = sorted(candidates, key=_rerank_recovery_quality_sort_key)[0]
        cells.append(
            {
                "cell_key": key,
                "descriptor": descriptors[key],
                "candidate_count": len(candidates),
                "elite": _rerank_recovery_elite_payload(elite),
                "elite_selection_key": list(
                    _rerank_recovery_quality_sort_key(elite)
                ),
            }
        )
    return cells


def _normalize_rerank_probe_descriptors(rows: list[dict[str, object]]) -> None:
    selected_carrion_blockers = _selected_carrion_blocker_count(rows)
    for row in rows:
        descriptor = _fixture_rerank_probe_descriptor(
            source_type=str(row.get("source_type", "initial")),
            fixture_blocker_count=int(row.get("fixture_blocker_count", 0)),
            carrion_only_blocker_count=int(row.get("carrion_only_blocker_count", 0)),
            selected_carrion_only_blocker_count=selected_carrion_blockers,
            survival=float(row.get("post_contact_survival_rate", 0.0)),
            drink=float(row.get("drink_after_carrion_rate", 0.0)),
            hydration=float(row.get("mean_hydration_delta_after_carrion", 0.0)),
            water_distance=float(row.get("mean_water_distance", 0.0)),
            unsupported_resolved=int(row.get("unsupported_resolved_action_count", 0)),
            dominant_share=float(row.get("dominant_requested_action_share", 0.0)),
        )
        row["descriptor"] = descriptor
        row["cell_key"] = _fixture_rerank_probe_descriptor_key(descriptor)
        row["facet_cells"] = _fixture_rerank_probe_facet_cells(descriptor)


def _selected_carrion_blocker_count(
    rows: Sequence[Mapping[str, object]],
) -> int:
    selected = [
        int(row.get("carrion_only_blocker_count", 0))
        for row in rows
        if bool(row.get("selected"))
    ]
    if selected:
        return min(selected)
    values = [int(row.get("carrion_only_blocker_count", 0)) for row in rows]
    return min(values) if values else 0


def _selected_rerank_probe_cell_memberships(
    rows: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    cell_keys = {str(cell.get("cell_key")) for cell in cells}
    memberships = []
    selected_carrion_blockers = _selected_carrion_blocker_count(rows)
    for row in rows:
        if not bool(row.get("selected")):
            continue
        descriptor = _fixture_rerank_probe_descriptor(
            source_type=str(row.get("source_type", "selected")),
            fixture_blocker_count=int(row.get("fixture_blocker_count", 0)),
            carrion_only_blocker_count=int(row.get("carrion_only_blocker_count", 0)),
            selected_carrion_only_blocker_count=selected_carrion_blockers,
            survival=float(row.get("post_contact_survival_rate", 0.0)),
            drink=float(row.get("drink_after_carrion_rate", 0.0)),
            hydration=float(row.get("mean_hydration_delta_after_carrion", 0.0)),
            water_distance=float(row.get("mean_water_distance", 0.0)),
            unsupported_resolved=int(row.get("unsupported_resolved_action_count", 0)),
            dominant_share=float(row.get("dominant_requested_action_share", 0.0)),
        )
        cell_key = _fixture_rerank_probe_descriptor_key(descriptor)
        memberships.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "cell_key": cell_key,
                "cell_present": cell_key in cell_keys,
                "descriptor": descriptor,
                "facet_cells": _fixture_rerank_probe_facet_cells(descriptor),
            }
        )
    return memberships


def _non_selected_recovery_better_rows(
    rows: Sequence[Mapping[str, object]],
    selected_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not selected_rows:
        return []
    selected_key = min(_rerank_recovery_quality_sort_key(row) for row in selected_rows)
    better = [
        dict(row)
        for row in rows
        if not bool(row.get("selected"))
        and _rerank_recovery_quality_sort_key(row) < selected_key
    ]
    better.sort(key=_rerank_recovery_quality_sort_key)
    return [_rerank_recovery_elite_payload(row) for row in better]


def _retained_rerank_probe_candidate_ids(
    cells: Sequence[Mapping[str, object]],
    *,
    selected_ids: set[str],
) -> list[str]:
    ids = {
        str(dict(cell.get("elite", {})).get("candidate_id", ""))
        for cell in cells
        if isinstance(cell.get("elite"), Mapping)
    }
    return sorted(candidate_id for candidate_id in ids if candidate_id and candidate_id not in selected_ids)


def _fixture_rerank_recovery_important_cells(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    checks = (
        (
            "nonzero_post_contact_survival",
            lambda row: float(row.get("post_contact_survival_rate", 0.0)) > 0.0,
        ),
        (
            "nonzero_post_contact_survival_repair",
            lambda row: float(row.get("post_contact_survival_rate", 0.0)) > 0.0
            and str(row.get("origin_source_type")) == "repair",
        ),
        (
            "nonzero_post_contact_survival_selected",
            lambda row: float(row.get("post_contact_survival_rate", 0.0)) > 0.0
            and bool(row.get("selected")),
        ),
        (
            "high_drink_nonnegative_hydration",
            lambda row: str(
                dict(row.get("descriptor", {})).get("drink_after_carrion_niche")
            )
            == "high"
            and float(row.get("mean_hydration_delta_after_carrion", 0.0)) >= 0.0,
        ),
        (
            "close_water_nonzero_survival",
            lambda row: float(row.get("post_contact_survival_rate", 0.0)) > 0.0
            and str(dict(row.get("descriptor", {})).get("water_distance_niche"))
            == "close",
        ),
        (
            "fewer_carrion_blockers_than_selected",
            lambda row: str(dict(row.get("descriptor", {})).get("fewer_blocker_lane"))
            == "fewer_than_selected",
        ),
        (
            "dominant_cap_pass_nonzero_survival",
            lambda row: float(row.get("post_contact_survival_rate", 0.0)) > 0.0
            and str(dict(row.get("descriptor", {})).get("dominant_action_cap"))
            == "pass",
        ),
    )
    result = []
    for name, predicate in checks:
        candidate_ids = sorted(
            str(row.get("candidate_id", ""))
            for row in rows
            if predicate(row)
        )
        result.append(
            {
                "cell": name,
                "empty": not candidate_ids,
                "candidate_ids": candidate_ids,
            }
        )
    return result


def _fixture_rerank_recovery_input_summary(
    *,
    complete_rows: Sequence[Mapping[str, object]],
    missing_rows: Sequence[Mapping[str, object]],
    selected_ids: set[str],
) -> dict[str, object]:
    missing_reasons = Counter(str(row.get("missing_reason", "unknown")) for row in missing_rows)
    missing_fields = Counter(
        str(field)
        for row in missing_rows
        for field in list(row.get("missing_fields", []))
    )
    source_type_counts = Counter(str(row.get("source_type", "unknown")) for row in complete_rows)
    return {
        "candidate_count": len(complete_rows) + len(missing_rows),
        "complete_probe_count": len(complete_rows),
        "missing_probe_count": len(missing_rows),
        "missing_probe_count_by_reason": dict(sorted(missing_reasons.items())),
        "missing_probe_count_by_field": dict(sorted(missing_fields.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "selected_candidate_ids": sorted(selected_ids),
        "selected_candidate_probe_completed": any(
            bool(row.get("selected")) for row in complete_rows
        ),
        "selected_candidate_probe_missing": any(
            bool(row.get("selected")) for row in missing_rows
        ),
    }


def _elite_candidate_ids(cells: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(dict(cell.get("elite", {})).get("candidate_id", ""))
            for cell in cells
            if isinstance(cell.get("elite"), Mapping)
            and str(dict(cell.get("elite", {})).get("candidate_id", ""))
        }
    )


def _rerank_recovery_quality_sort_key(row: Mapping[str, object]) -> tuple:
    dominant_share = float(row.get("dominant_requested_action_share", 1.0))
    return (
        int(row.get("fixture_blocker_count", 999_999)),
        int(row.get("carrion_only_blocker_count", 999_999)),
        -float(row.get("post_contact_survival_rate", 0.0)),
        -float(row.get("drink_after_carrion_rate", 0.0)),
        -float(row.get("mean_hydration_delta_after_carrion", 0.0)),
        float(row.get("mean_water_distance", 999_999.0)),
        int(row.get("unsupported_resolved_action_count", 999_999)),
        0
        if dominant_share <= MIND_V3_RERANK_RECOVERY_DOMINANT_ACTION_CAP
        else 1,
        dominant_share,
        -float(row.get("search_score") or 0.0),
        int(row.get("prefilter_rank") or 999_999),
        str(row.get("candidate_id", "")),
    )


def _rerank_recovery_elite_payload(
    row: Mapping[str, object],
) -> dict[str, object]:
    return {
        "candidate_id": row.get("candidate_id"),
        "report_label": row.get("report_label"),
        "source_type": row.get("source_type"),
        "origin_source_type": row.get("origin_source_type"),
        "selected": bool(row.get("selected")),
        "prefilter_rank": row.get("prefilter_rank"),
        "search_score": row.get("search_score"),
        "fixture_blocker_count": row.get("fixture_blocker_count"),
        "carrion_only_blocker_count": row.get("carrion_only_blocker_count"),
        "post_contact_survival_rate": row.get("post_contact_survival_rate"),
        "drink_after_carrion_rate": row.get("drink_after_carrion_rate"),
        "mean_hydration_delta_after_carrion": row.get(
            "mean_hydration_delta_after_carrion"
        ),
        "mean_water_distance": row.get("mean_water_distance"),
        "unsupported_requested_action_count": row.get(
            "unsupported_requested_action_count"
        ),
        "unsupported_resolved_action_count": row.get(
            "unsupported_resolved_action_count"
        ),
        "dominant_requested_action_share": row.get(
            "dominant_requested_action_share"
        ),
        "descriptor": row.get("descriptor"),
        "cell_key": row.get("cell_key"),
        "facet_cells": row.get("facet_cells"),
    }


def _drink_after_carrion_niche(value: float) -> str:
    if value <= 0.0:
        return "zero"
    if value < (1.0 / 3.0):
        return "low"
    if value < (2.0 / 3.0):
        return "medium"
    return "high"


def _hydration_recovery_niche(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value >= -0.05:
        return "near_flat"
    return "negative"


def _water_distance_niche(value: float) -> str:
    if value <= 2.5:
        return "close"
    if value <= 4.0:
        return "mid"
    return "far"


def _fewer_blocker_lane(
    carrion_only_blocker_count: int,
    selected_carrion_only_blocker_count: int,
) -> str:
    if carrion_only_blocker_count < selected_carrion_only_blocker_count:
        return "fewer_than_selected"
    if carrion_only_blocker_count > selected_carrion_only_blocker_count:
        return "more_than_selected"
    return "same_as_selected"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _optional_int(value: object) -> int | None:
    number = _optional_number(value)
    return int(number) if number is not None else None


def _archive_contract(
    *,
    branch_report: Mapping[str, object],
    counterfactual_report: Mapping[str, object] | None,
    max_dataset_records_per_class: int,
    min_survivor_cells: int,
    min_failure_cells: int,
    min_counterfactual_survivor_seeds: int,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_RECOVERY_ARCHIVE_POLICY,
        "source_branch_schema_version": branch_report.get("schema_version"),
        "source_counterfactual_schema_version": (
            counterfactual_report.get("schema_version")
            if counterfactual_report is not None
            else None
        ),
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
        "min_counterfactual_survivor_seeds": int(
            min_counterfactual_survivor_seeds
        ),
    }


def _validate_branch_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION:
        raise CarrionRecoveryArchiveError("branch report has stale schema_version")
    runs = report.get("branch_runs")
    if not isinstance(runs, list) or not runs:
        raise CarrionRecoveryArchiveError("branch report must include branch_runs")


def _validate_counterfactual_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION:
        raise CarrionRecoveryArchiveError(
            "counterfactual report has stale schema_version"
        )
    scripts = report.get("scripts")
    if not isinstance(scripts, list) or not scripts:
        raise CarrionRecoveryArchiveError(
            "counterfactual report must include scripts"
        )


def _branch_runs(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        run
        for run in list(report.get("branch_runs", []))
        if isinstance(run, Mapping)
    ]


def _counterfactual_runs(
    report: Mapping[str, object] | None,
) -> list[Mapping[str, object]]:
    if report is None:
        return []
    runs = []
    scripts = report.get("scripts")
    script_items = scripts if isinstance(scripts, list) else []
    for script in script_items:
        if not isinstance(script, Mapping):
            continue
        script_runs = script.get("runs")
        run_items = script_runs if isinstance(script_runs, list) else []
        for run in run_items:
            if isinstance(run, Mapping):
                runs.append(run)
    return runs


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


def _counterfactual_dataset_records(
    runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records = []
    for run in runs:
        if not _has_trajectory_path(run):
            continue
        records.append(_counterfactual_dataset_record(run))
    records.sort(
        key=lambda record: (
            not bool(dict(record.get("label", {})).get("terminal_survivor")),
            int(dict(record.get("source", {})).get("seed", 0)),
            str(dict(record.get("source", {})).get("continuation_script", "")),
        )
    )
    return records


def _counterfactual_dataset_record(
    run: Mapping[str, object],
) -> dict[str, object]:
    alive = _int_value(run.get("alive_agents"))
    births = _int_value(run.get("births"))
    heuristic_count = _int_value(run.get("heuristic_action_source_count"))
    terminal_survivor = alive > 0 and heuristic_count == 0
    script = str(run.get("counterfactual_script", "unknown"))
    seed = _int_value(run.get("seed"))
    payload = {
        "schema_version": MIND_V3_CARRION_RECOVERY_DATASET_RECORD_SCHEMA_VERSION,
        "source": {
            "source_type": "counterfactual_fixture_rollout",
            "branch_id": f"counterfactual-{script}-seed-{seed}",
            "seed": seed,
            "fixture": run.get("fixture"),
            "branch_tick": None,
            "base_script": None,
            "continuation_script": script,
            "trajectory_path": run.get("trajectory_path"),
        },
        "descriptor": {
            "source_type": "counterfactual_fixture_rollout",
            "fixture": run.get("fixture"),
            "terminal_alive_bin": _alive_bin(alive),
            "births_bin": _births_bin(births),
            "continuation_script": script,
            "dominant_requested_action": run.get("dominant_requested_action"),
        },
        "label": {
            "outcome_class": "survivor" if terminal_survivor else "failure",
            "terminal_survivor": terminal_survivor,
            "alive_agents": alive,
            "births": births,
            "zero_heuristic_runtime_actions": heuristic_count == 0,
        },
        "metrics": {
            "quality_score": _quality_score(run),
            "dominant_requested_action": run.get("dominant_requested_action"),
            "dominant_requested_action_share": run.get(
                "dominant_requested_action_share"
            ),
            "unique_requested_actions": run.get("unique_requested_actions"),
            "heuristic_action_source_count": heuristic_count,
        },
        "outcome_metrics": run.get("outcome_metrics"),
    }
    payload["record_id"] = stable_payload_digest(payload)
    return payload


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
    *,
    counterfactual_runs: Sequence[Mapping[str, object]] = (),
    counterfactual_dataset_records: Sequence[Mapping[str, object]] = (),
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
    counterfactual_survivor_seeds = sorted(
        {
            _int_value(run.get("seed"))
            for run in counterfactual_runs
            if _has_trajectory_path(run)
            and _int_value(run.get("alive_agents")) > 0
            and _int_value(run.get("heuristic_action_source_count")) == 0
        }
    )
    counterfactual_report_survivor_seeds = sorted(
        {
            _int_value(run.get("seed"))
            for run in counterfactual_runs
            if _int_value(run.get("alive_agents")) > 0
            and _int_value(run.get("heuristic_action_source_count")) == 0
        }
    )
    return {
        "source_branch_run_count": len(branch_runs),
        "cell_count": len(cells),
        "survivor_cell_count": len(survivor_cells),
        "failure_cell_count": len(failure_cells),
        "seeds_with_survivor_cells": seeds_with_survivor,
        "outcome_class_counts": dict(sorted(descriptor_counts.items())),
        "dataset_record_count": len(dataset_records),
        "branch_dataset_record_count": (
            len(dataset_records) - len(counterfactual_dataset_records)
        ),
        "counterfactual_dataset_record_count": len(counterfactual_dataset_records),
        "counterfactual_source_run_count": len(counterfactual_runs),
        "counterfactual_survivor_seed_count": len(counterfactual_survivor_seeds),
        "counterfactual_survivor_seeds": counterfactual_survivor_seeds,
        "counterfactual_report_survivor_seed_count": (
            len(counterfactual_report_survivor_seeds)
        ),
        "counterfactual_report_survivor_seeds": counterfactual_report_survivor_seeds,
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
        "counterfactual_outcome_metrics": aggregate_run_outcome_metrics(
            counterfactual_runs
        )
        if counterfactual_runs
        else {},
        "outcome_metrics": aggregate_run_outcome_metrics(elite_runs),
        "best_survivor_elite": _best_elite(survivor_cells),
        "best_failure_elite": _best_elite(failure_cells),
    }


def _has_trajectory_path(run: Mapping[str, object]) -> bool:
    trajectory_path = run.get("trajectory_path")
    return isinstance(trajectory_path, str) and bool(trajectory_path)


def _archive_acceptance(
    aggregate: Mapping[str, object],
    *,
    branch_report: Mapping[str, object],
    min_survivor_cells: int,
    min_failure_cells: int,
    min_counterfactual_survivor_seeds: int,
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
    counterfactual_survivor_seeds = int(
        aggregate.get("counterfactual_survivor_seed_count", 0)
    )
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
    if counterfactual_survivor_seeds < min_counterfactual_survivor_seeds:
        blockers.append("insufficient_counterfactual_survivor_seeds")
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
        "min_counterfactual_survivor_seeds": min_counterfactual_survivor_seeds,
        "counterfactual_survivor_seed_count": counterfactual_survivor_seeds,
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
