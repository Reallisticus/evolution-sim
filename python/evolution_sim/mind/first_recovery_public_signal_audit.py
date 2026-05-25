from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    OBSERVATION_INPUT_VECTOR_SIZE,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
    observation_contract,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
    MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
    feature_name_leakage,
    group_archive_rows_by_branch_target,
    load_first_recovery_archive_rows,
    _load_join_records,
    _matching_record,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_public_signal_audit_v1"
)
MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_public_state_action_alias_audit_v1"
)

DEFAULT_ARCHIVE_REPORT_PATH = Path(
    "output/mind/mind-v3-v109-first-recovery-branch-archive.json"
)
DEFAULT_ARCHIVE_ROWS_PATH = Path(
    "output/mind/mind-v3-v109-first-recovery-branch-archive.jsonl.gz"
)
DEFAULT_SHADOW_RANKER_REPORT_PATH = Path(
    "output/mind/mind-v3-v111-first-recovery-shadow-ranker-joined-public.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v112-first-recovery-public-signal-audit.json"
)
DEFAULT_HISTORY_WINDOW = 8
MAX_EXAMPLES = 12
ALIAS_QUANTIZATION_DIGITS = 2
NEAR_CONSTANT_RANGE_EPSILON = 1e-6

SIGNAL_FAMILY_ORDER: tuple[str, ...] = (
    "carrion_freshness_depletion",
    "patch_residence_diminishing_returns",
    "recent_failed_intake",
    "reproduction_readiness_debt",
    "water_route_quality",
    "contestedness_competitor_pressure",
)

FORBIDDEN_SIGNAL_FIELD_TOKENS = frozenset(
    {
        "seed",
        "source",
        "source_path",
        "source_kind",
        "fixture",
        "fixture_name",
        "branch",
        "branch_id",
        "record_index",
        "agent_id",
        "tick",
        "logged_action",
        "private",
        "private_world_state",
        "world",
        "simulation_world",
        "provenance",
    }
)


class FirstRecoveryPublicSignalAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublicSignalAuditBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    group_key: str
    rows: tuple[dict[str, object], ...]
    oracle_actions: tuple[str, ...]
    provenance: Mapping[str, object]
    record: Mapping[str, object]
    decoded: Mapping[str, object]
    history: tuple[Mapping[str, object], ...]
    family_values: Mapping[str, Mapping[str, float]]


def build_first_recovery_public_signal_audit(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    shadow_ranker_report: Mapping[str, object] | None = None,
    shadow_ranker_report_path: str | Path | None = DEFAULT_SHADOW_RANKER_REPORT_PATH,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> PublicSignalAuditBuild:
    history_limit = _positive_int(history_window, field="history_window")
    contract = _contract(history_window=history_limit)
    archive_payload, archive_report_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    shadow_payload, shadow_report_evidence = _resolve_json_report(
        shadow_ranker_report,
        shadow_ranker_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    groups = group_archive_rows_by_branch_target(rows)
    joined = _load_join_records(trajectory_paths, trajectory_glob_patterns)
    targets, target_join = _target_evidence(
        groups,
        joined,
        history_window=history_limit,
    )
    source = _source_section(
        archive_report=archive_payload,
        shadow_ranker_report=shadow_payload,
        rows=rows,
        groups=groups,
        archive_report_evidence=archive_report_evidence,
        archive_rows_evidence=rows_evidence,
        shadow_report_evidence=shadow_report_evidence,
    )
    join_evidence = _join_evidence_section(joined, target_join)
    family_sections = {
        family: _signal_family_section(family, targets)
        for family in SIGNAL_FAMILY_ORDER
    }
    state_action_aliasing = _state_action_aliasing(targets)
    observability_summary = _observability_summary(
        signal_families=family_sections,
        state_action_aliasing=state_action_aliasing,
        source=source,
        join_evidence=join_evidence,
    )
    leakage = _leakage_guard(family_sections)
    classification = _classification(
        source=source,
        join_evidence=join_evidence,
        observability_summary=observability_summary,
        leakage=leakage,
    )
    recommendation = _recommendation(
        classification=classification,
        observability_summary=observability_summary,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_POLICY,
        "contract": contract,
        "source": source,
        "join_evidence": join_evidence,
        "signal_families": family_sections,
        "state_action_aliasing": state_action_aliasing,
        "observability_summary": observability_summary,
        "leakage_guard": leakage,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return PublicSignalAuditBuild(report=report)


def write_first_recovery_public_signal_audit_report(
    build: PublicSignalAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def signal_field_leakage(field_names: Sequence[str]) -> tuple[str, ...]:
    leaks = set(feature_name_leakage(field_names))
    for name in field_names:
        tokens = tuple(name.replace("=", ".").replace("[", ".").replace("]", "").split("."))
        if any(token in FORBIDDEN_SIGNAL_FIELD_TOKENS for token in tokens):
            leaks.add(name)
    return tuple(sorted(leaks))


def _contract(*, history_window: int) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "private_world_state_serialized": False,
        "private_world_state_input": False,
        "fixture_identity_signal": False,
        "seed_identity_signal": False,
        "source_identity_signal": False,
        "branch_identity_signal": False,
        "logged_action_signal": False,
        "observation_field_change": False,
        "trajectory_reader_policy": LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
        "public_observation_contract": {
            "schema_version": observation_contract().get("schema_version"),
            "self_input_fields": [
                field for field in SELF_INPUT_FIELDS if field != "mind_inheritance_available"
            ],
            "patch_input_fields": list(PATCH_INPUT_FIELDS),
            "navigation_targets": list(NAVIGATION_TARGETS),
            "navigation_input_fields": list(NAVIGATION_INPUT_FIELDS),
            "excluded_controller_private_fields": ["self.mind_inheritance_available"],
        },
        "history_window": int(history_window),
        "history_policy": "same_agent_records_strictly_before_branch_target_only",
        "branch_outcomes_policy": "offline_labels_only_not_runtime_inputs",
        "contract_digest": stable_payload_digest(
            {
                "schema_version": MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
                "history_window": int(history_window),
                "trajectory_reader_policy": LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
            }
        ),
    }


def _resolve_json_report(
    payload: Mapping[str, object] | None,
    path: str | Path | None,
    *,
    expected_schema: str,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    if payload is not None:
        report = dict(payload)
        return report, {
            "path": str(path) if path is not None else None,
            "loaded": True,
            "schema_version": report.get("schema_version"),
            "schema_matches": report.get("schema_version") == expected_schema,
            "in_memory": True,
        }
    if path is None:
        return None, {
            "path": None,
            "loaded": False,
            "schema_matches": False,
            "error": "missing_report_path",
        }
    try:
        with _open_input(Path(path)) as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, {
            "path": str(path),
            "loaded": False,
            "schema_matches": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    if not isinstance(report, dict):
        return None, {
            "path": str(path),
            "loaded": False,
            "schema_matches": False,
            "error": "report_not_object",
        }
    return report, {
        "path": str(path),
        "loaded": True,
        "schema_version": report.get("schema_version"),
        "schema_matches": report.get("schema_version") == expected_schema,
        "file_sha256": _file_sha256(path),
    }


def _resolve_archive_rows(
    rows: Sequence[Mapping[str, object]] | None,
    path: str | Path | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if rows is not None:
        parsed = tuple(dict(row) for row in rows)
        return parsed, {
            "path": str(path) if path is not None else None,
            "loaded": True,
            "row_count": len(parsed),
            "in_memory": True,
        }
    if path is None:
        return (), {
            "path": None,
            "loaded": False,
            "row_count": 0,
            "error": "missing_archive_rows_path",
        }
    try:
        parsed = load_first_recovery_archive_rows(path)
    except (OSError, ValueError) as exc:
        return (), {
            "path": str(path),
            "loaded": False,
            "row_count": 0,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return parsed, {
        "path": str(path),
        "loaded": True,
        "row_count": len(parsed),
        "file_sha256": _file_sha256(path),
    }


def _source_section(
    *,
    archive_report: Mapping[str, object] | None,
    shadow_ranker_report: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
    groups: Sequence[object],
    archive_report_evidence: Mapping[str, object],
    archive_rows_evidence: Mapping[str, object],
    shadow_report_evidence: Mapping[str, object],
) -> dict[str, object]:
    by_seed = Counter()
    by_source = Counter()
    for group in groups:
        seed = getattr(group, "seed", None)
        source = getattr(group, "source_kind", None)
        if seed is not None:
            by_seed[str(seed)] += 1
        if source:
            by_source[str(source)] += 1
    return {
        "archive_report": archive_report_evidence,
        "archive_rows": archive_rows_evidence,
        "shadow_ranker_report": shadow_report_evidence,
        "archive_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "shadow_ranker_schema_version": _mapping(shadow_ranker_report or {}).get(
            "schema_version"
        ),
        "archive_row_count": len(rows),
        "target_group_count": len(groups),
        "groups_by_seed": _counter_to_dict(by_seed),
        "groups_by_source_kind": _counter_to_dict(by_source),
        "shadow_ranker_primary": _mapping(
            _mapping(shadow_ranker_report or {}).get("classification")
        ).get("primary"),
        "shadow_ranker_missing_evidence": _list(
            _mapping(_mapping(shadow_ranker_report or {}).get("classification")).get(
                "missing_evidence"
            )
        ),
    }


def _join_evidence_section(
    joined: Mapping[str, object],
    target_join: Mapping[str, object],
) -> dict[str, object]:
    evidence = _mapping(joined.get("evidence"))
    return {
        "reader_policy": LOCAL_LENIENT_TRAJECTORY_READER_POLICY,
        "trajectory_paths": _list(evidence.get("trajectory_paths")),
        "trajectory_globs": _list(evidence.get("trajectory_globs")),
        "loaded_path_count": _int(evidence.get("loaded_path_count")),
        "loaded_paths": _list(evidence.get("loaded_paths")),
        "load_failure_count": _int(evidence.get("load_failure_count")),
        "load_failures": _list(evidence.get("load_failures")),
        "malformed_record_count": _int(evidence.get("malformed_record_count")),
        "malformed_records": _list(evidence.get("malformed_records"))[:MAX_EXAMPLES],
        "trajectory_record_count": _int(evidence.get("record_count")),
        "matched_archive_row_count": _int(target_join.get("matched_archive_row_count")),
        "missing_archive_row_count": _int(target_join.get("missing_archive_row_count")),
        "matched_target_group_count": _int(target_join.get("matched_target_group_count")),
        "missing_target_group_count": _int(target_join.get("missing_target_group_count")),
        "missing_examples": _list(target_join.get("missing_examples"))[:MAX_EXAMPLES],
        "missing_evidence": sorted(
            set(
                str(item)
                for item in (
                    _list(joined.get("missing_evidence"))
                    + _list(target_join.get("missing_evidence"))
                )
            )
        ),
    }


def _target_evidence(
    groups: Sequence[object],
    joined: Mapping[str, object],
    *,
    history_window: int,
) -> tuple[tuple[TargetEvidence, ...], dict[str, object]]:
    record_by_key = _mapping(joined.get("record_by_key"))
    records_by_path = _mapping(joined.get("records_by_path"))
    targets: list[TargetEvidence] = []
    missing_rows = 0
    missing_groups = 0
    missing_examples: list[dict[str, object]] = []
    for group in groups:
        rows = tuple(dict(row) for row in getattr(group, "rows", ()))
        if not rows:
            continue
        first = rows[0]
        record = _matching_record(first, record_by_key)
        if record is None:
            missing_groups += 1
            missing_rows += len(rows)
            if len(missing_examples) < MAX_EXAMPLES:
                missing_examples.append(_join_missing_example(first))
            continue
        try:
            decoded = _decode_observation(_mapping(record.get("observation_input")))
        except (ValueError, TypeError):
            missing_groups += 1
            missing_rows += len(rows)
            if len(missing_examples) < MAX_EXAMPLES:
                example = _join_missing_example(first)
                example["reason"] = "observation_input_decode_failed"
                missing_examples.append(example)
            continue
        provenance = _mapping(first.get("provenance"))
        source_path = _optional_string(provenance.get("source_path"))
        record_index = _int_or_none(provenance.get("record_index"))
        agent_id = _int_or_none(provenance.get("agent_id"))
        history: tuple[Mapping[str, object], ...] = ()
        if source_path is not None and record_index is not None and agent_id is not None:
            history = tuple(
                record
                for index, record in _list(records_by_path.get(source_path))
                if isinstance(record, Mapping)
                and _int(record.get("record_index")) < record_index
                and _int_or_none(record.get("agent_id")) == agent_id
            )[-history_window:]
        targets.append(
            TargetEvidence(
                group_key=str(getattr(group, "group_key", "")),
                rows=rows,
                oracle_actions=_oracle_actions(rows),
                provenance=provenance,
                record=record,
                decoded=decoded,
                history=history,
                family_values=_family_values(decoded, history, record),
            )
        )
    matched_rows = sum(len(target.rows) for target in targets)
    missing: list[str] = []
    if missing_rows:
        missing.append("archive_trajectory_join")
    return tuple(targets), {
        "matched_archive_row_count": matched_rows,
        "missing_archive_row_count": missing_rows,
        "matched_target_group_count": len(targets),
        "missing_target_group_count": missing_groups,
        "missing_examples": missing_examples,
        "missing_evidence": missing,
    }


def _decode_observation(observation_input: Mapping[str, object]) -> dict[str, object]:
    values = decode_observation_input(dict(observation_input))
    if len(values) != OBSERVATION_INPUT_VECTOR_SIZE:
        raise ValueError("decoded observation has unexpected vector size")
    cursor = 0
    self_values = {
        field: _round(values[cursor + index])
        for index, field in enumerate(SELF_INPUT_FIELDS)
        if field != "mind_inheritance_available"
    }
    cursor += len(SELF_INPUT_FIELDS)
    patch: dict[tuple[int, int], dict[str, float]] = {}
    for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1):
        for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1):
            patch[(dx, dy)] = {
                field: _round(values[cursor + index])
                for index, field in enumerate(PATCH_INPUT_FIELDS)
            }
            cursor += len(PATCH_INPUT_FIELDS)
    navigation: dict[str, dict[str, float]] = {}
    for target in NAVIGATION_TARGETS:
        navigation[target] = {
            field: _round(values[cursor + index])
            for index, field in enumerate(NAVIGATION_INPUT_FIELDS)
        }
        cursor += len(NAVIGATION_INPUT_FIELDS)
    return {
        "self": self_values,
        "patch": patch,
        "navigation": navigation,
    }


def _family_values(
    decoded: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    record: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    self_values = _mapping(decoded.get("self"))
    patch = _mapping(decoded.get("patch"))
    navigation = _mapping(decoded.get("navigation"))
    center = _mapping(patch.get((0, 0)))
    patch_cells = [_mapping(cell) for cell in patch.values()]
    recent_outcomes = [_mapping(item.get("outcome")) for item in history]
    recent_requested = [str(item.get("requested_action", "")) for item in history]
    failed_moves = sum(
        1
        for item in history
        if str(item.get("requested_action", "")).startswith("move_")
        and item.get("moved") is not True
    )
    action_mask = _mapping(record.get("action_mask"))
    return {
        "carrion_freshness_depletion": {
            "patch.max_fresh_kill_energy": _max_cell_value(patch_cells, "fresh_kill_energy"),
            "patch.max_carcass_energy": _max_cell_value(patch_cells, "carcass_energy"),
            "patch.max_carrion_signal": _max_cell_value(patch_cells, "carrion_signal"),
            "navigation.carrion.distance": _number(
                _mapping(navigation.get("carrion")).get("distance")
            ),
            "navigation.carrion.strength": _number(
                _mapping(navigation.get("carrion")).get("strength")
            ),
            "action_mask.eat": 1.0 if action_mask.get("eat") is True else 0.0,
        },
        "patch_residence_diminishing_returns": {
            "center.food": _number(center.get("food")),
            "center.vegetation": _number(center.get("vegetation")),
            "center.recovery_debt": _number(center.get("recovery_debt")),
            "center.ecology_state_code": _number(center.get("ecology_state_code")),
            "history.resource_gain_mean": _mean(
                _number_or_none(outcome.get("resource_gain"))
                for outcome in recent_outcomes
            ),
            "history.resource_gain_zero_rate": _share(
                sum(1 for outcome in recent_outcomes if _number(outcome.get("resource_gain")) <= 0.0),
                len(recent_outcomes),
            ),
        },
        "recent_failed_intake": {
            "history.eat_count": float(sum(1 for action in recent_requested if action == "eat")),
            "history.drink_count": float(sum(1 for action in recent_requested if action == "drink")),
            "history.eat_no_gain_count": float(
                sum(
                    1
                    for action, outcome in zip(recent_requested, recent_outcomes, strict=False)
                    if action == "eat" and _number(outcome.get("resource_gain")) <= 0.0
                )
            ),
            "history.drink_no_gain_count": float(
                sum(
                    1
                    for action, outcome in zip(recent_requested, recent_outcomes, strict=False)
                    if action == "drink" and _number(outcome.get("resource_gain")) <= 0.0
                )
            ),
            "history.failed_movement_rate": _share(failed_moves, len(history)),
        },
        "reproduction_readiness_debt": {
            "self.reproduction_ready": _number(self_values.get("reproduction_ready")),
            "self.reproductive_stage_code": _number(
                self_values.get("reproductive_stage_code")
            ),
            "self.reproductive_expression_code": _number(
                self_values.get("reproductive_expression_code")
            ),
            "self.sexual_reproduction_unlocked": _number(
                self_values.get("sexual_reproduction_unlocked")
            ),
            "self.reproductive_signal": _number(self_values.get("reproductive_signal")),
            "self.energy_debt": _round(1.0 - _number(self_values.get("energy_ratio"))),
            "self.hydration_debt": _round(
                1.0 - _number(self_values.get("hydration_ratio"))
            ),
            "self.health_debt": _round(1.0 - _number(self_values.get("health_ratio"))),
            "action_mask.mate": 1.0 if action_mask.get("mate") is True else 0.0,
        },
        "water_route_quality": {
            "self.water_access_reason_code": _number(
                self_values.get("water_access_reason_code")
            ),
            "self.hydrology_support_adjacent_to_water": _number(
                self_values.get("hydrology_support_adjacent_to_water")
            ),
            "self.hydrology_support_wetland": _number(
                self_values.get("hydrology_support_wetland")
            ),
            "navigation.water.distance": _number(
                _mapping(navigation.get("water")).get("distance")
            ),
            "navigation.water.strength": _number(
                _mapping(navigation.get("water")).get("strength")
            ),
            "action_mask.drink": 1.0 if action_mask.get("drink") is True else 0.0,
        },
        "contestedness_competitor_pressure": {
            "patch.occupant_agent_count": float(
                sum(1 for cell in patch_cells if _number(cell.get("occupant_code")) > 0.5)
            ),
            "patch.same_lineage_count": float(
                sum(1 for cell in patch_cells if _number(cell.get("same_lineage")) > 0.5)
            ),
            "patch.max_predator_risk": _max_cell_value(patch_cells, "predator_risk"),
            "patch.max_prey_biomass": _max_cell_value(patch_cells, "prey_biomass"),
            "center.occupant_code": _number(center.get("occupant_code")),
        },
    }


def _signal_family_section(
    family: str,
    targets: Sequence[TargetEvidence],
) -> dict[str, object]:
    values_by_feature: dict[str, list[float]] = defaultdict(list)
    support_seed = Counter()
    support_source = Counter()
    for target in targets:
        family_values = _mapping(target.family_values.get(family))
        if family_values:
            seed = target.provenance.get("seed")
            source = target.provenance.get("source_kind")
            if seed is not None:
                support_seed[str(seed)] += 1
            if source is not None:
                support_source[str(source)] += 1
        for name, value in family_values.items():
            values_by_feature[name].append(_number(value))
    present = tuple(sorted(values_by_feature))
    constants = _constant_fields(values_by_feature)
    alias_groups = _family_alias_groups(family, targets)
    separability = _family_separability(family, targets)
    missing_fields = _family_missing_fields(family, present)
    return {
        "public_field_presence": {
            "present_fields": list(present),
            "missing_public_fields": missing_fields,
            "field_count": len(present),
        },
        "missing_public_fields": missing_fields,
        "constant_near_constant_fields": constants,
        "alias_groups": alias_groups,
        "alias_group_count": len(alias_groups),
        "support_counts": {
            "target_group_count": len(targets),
            "supported_target_group_count": sum(
                1 for target in targets if target.family_values.get(family)
            ),
            "candidate_row_count": sum(len(target.rows) for target in targets),
        },
        "per_seed_support": _counter_to_dict(support_seed),
        "fixture_open_support": _counter_to_dict(support_source),
        "oracle_vs_nonoracle_separability": separability,
        "examples": _family_examples(family, targets),
    }


def _family_missing_fields(family: str, present: Sequence[str]) -> list[str]:
    present_set = set(present)
    expected = {
        "carrion_freshness_depletion": (
            "patch.max_fresh_kill_energy",
            "patch.max_carcass_energy",
            "patch.max_carrion_signal",
            "navigation.carrion.distance",
            "navigation.carrion.strength",
            "public.carcass_age",
            "public.recent_carcass_depletion_by_others",
        ),
        "patch_residence_diminishing_returns": (
            "center.food",
            "center.vegetation",
            "center.recovery_debt",
            "history.resource_gain_mean",
            "public.same_patch_residence_duration",
        ),
        "recent_failed_intake": (
            "history.eat_no_gain_count",
            "history.drink_no_gain_count",
            "history.failed_movement_rate",
            "public.failed_intake_reason",
        ),
        "reproduction_readiness_debt": (
            "self.reproduction_ready",
            "self.reproductive_stage_code",
            "self.reproductive_signal",
            "public.nearby_compatible_mate_count",
        ),
        "water_route_quality": (
            "self.water_access_reason_code",
            "navigation.water.distance",
            "navigation.water.strength",
            "public.water_route_blocker_count",
        ),
        "contestedness_competitor_pressure": (
            "patch.occupant_agent_count",
            "patch.max_predator_risk",
            "patch.max_prey_biomass",
            "public.competitor_intake_pressure",
        ),
    }
    return [field for field in expected.get(family, ()) if field not in present_set]


def _constant_fields(values_by_feature: Mapping[str, Sequence[float]]) -> list[dict[str, object]]:
    fields: list[dict[str, object]] = []
    for name, values in sorted(values_by_feature.items()):
        parsed = [_number(value) for value in values]
        if not parsed:
            continue
        unique = sorted({_round(value) for value in parsed})
        value_range = _round(max(parsed) - min(parsed))
        if len(unique) <= 1 or value_range <= NEAR_CONSTANT_RANGE_EPSILON:
            fields.append(
                {
                    "field": name,
                    "unique_value_count": len(unique),
                    "min": _round(min(parsed)),
                    "max": _round(max(parsed)),
                }
            )
    return fields


def _family_alias_groups(
    family: str,
    targets: Sequence[TargetEvidence],
) -> list[dict[str, object]]:
    buckets: dict[tuple[tuple[str, float], ...], list[TargetEvidence]] = defaultdict(list)
    for target in targets:
        signature = _signature(_mapping(target.family_values.get(family)))
        buckets[signature].append(target)
    aliases: list[dict[str, object]] = []
    for signature, members in sorted(buckets.items(), key=lambda item: str(item[0])):
        oracle_actions = sorted(
            {action for member in members for action in member.oracle_actions}
        )
        if len(members) < 2 or len(oracle_actions) < 2:
            continue
        aliases.append(
            {
                "signature": [[name, value] for name, value in signature],
                "target_group_count": len(members),
                "oracle_actions": oracle_actions,
                "examples": [_target_example(member) for member in members[:MAX_EXAMPLES]],
            }
        )
    return aliases[:MAX_EXAMPLES]


def _family_separability(
    family: str,
    targets: Sequence[TargetEvidence],
) -> dict[str, object]:
    oracle_values: dict[str, list[float]] = defaultdict(list)
    non_oracle_values: dict[str, list[float]] = defaultdict(list)
    oracle_unique_groups = 0
    compared_groups = 0
    for target in targets:
        oracle_actions = set(target.oracle_actions)
        if not oracle_actions:
            continue
        vectors: dict[str, tuple[tuple[str, float], ...]] = {}
        for row in target.rows:
            action = _optional_string(row.get("candidate_action"))
            if action is None:
                continue
            features = _candidate_family_features(family, target, row)
            signature = _signature(features)
            vectors[action] = signature
            sink = oracle_values if action in oracle_actions else non_oracle_values
            for name, value in features.items():
                sink[name].append(value)
        oracle_signatures = {
            signature for action, signature in vectors.items() if action in oracle_actions
        }
        non_oracle_signatures = {
            signature for action, signature in vectors.items() if action not in oracle_actions
        }
        if oracle_signatures:
            compared_groups += 1
            if oracle_signatures.isdisjoint(non_oracle_signatures):
                oracle_unique_groups += 1
    contrasts: list[dict[str, object]] = []
    for name in sorted(set(oracle_values) | set(non_oracle_values)):
        left = oracle_values.get(name, ())
        right = non_oracle_values.get(name, ())
        if not left or not right:
            continue
        oracle_mean = _round(_mean(left))
        non_oracle_mean = _round(_mean(right))
        contrasts.append(
            {
                "field": name,
                "oracle_mean": oracle_mean,
                "non_oracle_mean": non_oracle_mean,
                "absolute_delta": _round(abs(oracle_mean - non_oracle_mean)),
            }
        )
    contrasts.sort(key=lambda item: (-_number(item["absolute_delta"]), str(item["field"])))
    return {
        "compared_group_count": compared_groups,
        "oracle_candidate_unique_signature_count": oracle_unique_groups,
        "oracle_candidate_unique_signature_share": _share(
            oracle_unique_groups,
            compared_groups,
        ),
        "top_feature_contrasts": contrasts[:MAX_EXAMPLES],
    }


def _candidate_family_features(
    family: str,
    target: TargetEvidence,
    row: Mapping[str, object],
) -> dict[str, float]:
    values = dict(_mapping(target.family_values.get(family)))
    action = _optional_string(row.get("candidate_action")) or ""
    action_flags = {
        "candidate_action.is_eat": 1.0 if action == "eat" else 0.0,
        "candidate_action.is_drink": 1.0 if action == "drink" else 0.0,
        "candidate_action.is_mate": 1.0 if action == "mate" else 0.0,
        "candidate_action.is_stay": 1.0 if action == "stay" else 0.0,
        "candidate_action.is_move": 1.0 if action in MOVEMENT_ACTIONS else 0.0,
        "candidate_action.is_attack": 1.0 if action.startswith("attack_") else 0.0,
    }
    for name, value in action_flags.items():
        values[name] = value
    if family == "carrion_freshness_depletion":
        values["state_action.eat_x_carrion_signal"] = (
            action_flags["candidate_action.is_eat"]
            * _number(values.get("patch.max_carrion_signal"))
        )
        values["state_action.eat_x_carcass_energy"] = (
            action_flags["candidate_action.is_eat"]
            * _number(values.get("patch.max_carcass_energy"))
        )
    elif family == "water_route_quality":
        values["state_action.drink_x_water_strength"] = (
            action_flags["candidate_action.is_drink"]
            * _number(values.get("navigation.water.strength"))
        )
    elif family == "reproduction_readiness_debt":
        values["state_action.mate_x_reproduction_ready"] = (
            action_flags["candidate_action.is_mate"]
            * _number(values.get("self.reproduction_ready"))
        )
    elif family == "recent_failed_intake":
        values["state_action.eat_x_recent_eat_failures"] = (
            action_flags["candidate_action.is_eat"]
            * _number(values.get("history.eat_no_gain_count"))
        )
        values["state_action.drink_x_recent_drink_failures"] = (
            action_flags["candidate_action.is_drink"]
            * _number(values.get("history.drink_no_gain_count"))
        )
    return {name: _round(value) for name, value in values.items()}


def _family_examples(
    family: str,
    targets: Sequence[TargetEvidence],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for target in targets[:MAX_EXAMPLES]:
        examples.append(
            {
                **_target_example(target),
                "family_values": {
                    key: value
                    for key, value in sorted(_mapping(target.family_values.get(family)).items())
                },
            }
        )
    return examples


def _state_action_aliasing(targets: Sequence[TargetEvidence]) -> dict[str, object]:
    exact_buckets: dict[tuple[tuple[str, float], ...], list[TargetEvidence]] = defaultdict(list)
    near_buckets: dict[tuple[tuple[str, float], ...], list[TargetEvidence]] = defaultdict(list)
    separable_groups = 0
    compared_groups = 0
    for target in targets:
        state_values = _all_state_values(target)
        exact_buckets[_signature(state_values, digits=6)].append(target)
        near_buckets[_signature(state_values, digits=ALIAS_QUANTIZATION_DIGITS)].append(target)
        oracle_actions = set(target.oracle_actions)
        if oracle_actions:
            compared_groups += 1
            oracle_vectors = {
                _signature(_candidate_all_features(target, row))
                for row in target.rows
                if row.get("candidate_action") in oracle_actions
            }
            non_oracle_vectors = {
                _signature(_candidate_all_features(target, row))
                for row in target.rows
                if row.get("candidate_action") not in oracle_actions
            }
            if oracle_vectors and oracle_vectors.isdisjoint(non_oracle_vectors):
                separable_groups += 1
    exact_aliases = _alias_bucket_examples(exact_buckets)
    near_aliases = _alias_bucket_examples(near_buckets)
    return {
        "state_signature_policy": "decoded_public_observation_plus_same_agent_public_history_no_provenance_v1",
        "candidate_action_semantics_policy": "candidate_action_family_flags_and_state_action_cross_features_v1",
        "exact_public_state_alias_group_count": len(exact_aliases),
        "near_public_state_alias_group_count": len(near_aliases),
        "groups_with_same_or_near_same_public_state_but_different_oracle_actions": near_aliases,
        "candidate_action_separability": {
            "compared_group_count": compared_groups,
            "oracle_candidate_unique_signature_count": separable_groups,
            "oracle_candidate_unique_signature_share": _share(
                separable_groups,
                compared_groups,
            ),
        },
    }


def _all_state_values(target: TargetEvidence) -> dict[str, float]:
    values: dict[str, float] = {}
    for family in SIGNAL_FAMILY_ORDER:
        for name, value in _mapping(target.family_values.get(family)).items():
            values[f"{family}.{name}"] = _number(value)
    return values


def _candidate_all_features(
    target: TargetEvidence,
    row: Mapping[str, object],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for family in SIGNAL_FAMILY_ORDER:
        for name, value in _candidate_family_features(family, target, row).items():
            values[f"{family}.{name}"] = value
    return values


def _alias_bucket_examples(
    buckets: Mapping[tuple[tuple[str, float], ...], Sequence[TargetEvidence]],
) -> list[dict[str, object]]:
    aliases: list[dict[str, object]] = []
    for signature, members in sorted(buckets.items(), key=lambda item: str(item[0])):
        oracle_actions = sorted(
            {action for member in members for action in member.oracle_actions}
        )
        if len(members) < 2 or len(oracle_actions) < 2:
            continue
        aliases.append(
            {
                "signature_digest": _signature_digest(signature),
                "target_group_count": len(members),
                "oracle_actions": oracle_actions,
                "examples": [_target_example(member) for member in members[:MAX_EXAMPLES]],
            }
        )
    return aliases[:MAX_EXAMPLES]


def _observability_summary(
    *,
    signal_families: Mapping[str, Mapping[str, object]],
    state_action_aliasing: Mapping[str, object],
    source: Mapping[str, object],
    join_evidence: Mapping[str, object],
) -> dict[str, object]:
    missing_fields = sorted(
        {
            str(field)
            for section in signal_families.values()
            for field in _list(section.get("missing_public_fields"))
        }
    )
    family_alias_counts = {
        family: _int(section.get("alias_group_count"))
        for family, section in signal_families.items()
    }
    alias_count = _int(
        state_action_aliasing.get("near_public_state_alias_group_count")
    )
    sources = set(_mapping(source.get("groups_by_source_kind")))
    seed_count = len(_mapping(source.get("groups_by_seed")))
    non_sparse_aliasing = alias_count > 0 and (seed_count >= 2 or len(sources) >= 2)
    separability = _mapping(state_action_aliasing.get("candidate_action_separability"))
    separable_share = _number(separability.get("oracle_candidate_unique_signature_share"))
    return {
        "missing_public_field_count": len(missing_fields),
        "missing_public_fields": missing_fields,
        "signal_family_alias_group_counts": family_alias_counts,
        "near_public_state_alias_group_count": alias_count,
        "non_sparse_aliasing_support": non_sparse_aliasing,
        "candidate_action_unique_signature_share": separable_share,
        "existing_public_signals_state_action_separable": separable_share >= 0.9,
        "trajectory_join_complete": (
            _int(join_evidence.get("matched_archive_row_count"))
            == _int(source.get("archive_row_count"))
            and _int(join_evidence.get("missing_archive_row_count")) == 0
        ),
        "loaded_path_count": _int(join_evidence.get("loaded_path_count")),
        "malformed_record_count": _int(join_evidence.get("malformed_record_count")),
    }


def _leakage_guard(
    signal_families: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    names: list[str] = []
    for section in signal_families.values():
        presence = _mapping(section.get("public_field_presence"))
        names.extend(str(name) for name in _list(presence.get("present_fields")))
        for contrast in _list(_mapping(section.get("oracle_vs_nonoracle_separability")).get("top_feature_contrasts")):
            if isinstance(contrast, Mapping):
                names.append(str(contrast.get("field")))
    leaks = signal_field_leakage(names)
    return {
        "answer": "leakage_detected" if leaks else "leakage_free",
        "leakage_count": len(leaks),
        "leaking_signal_fields": list(leaks),
        "forbidden_signal_field_tokens": sorted(FORBIDDEN_SIGNAL_FIELD_TOKENS),
        "provenance_used_for_join_and_examples_only": True,
    }


def _classification(
    *,
    source: Mapping[str, object],
    join_evidence: Mapping[str, object],
    observability_summary: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    missing = []
    if _mapping(source.get("archive_report")).get("loaded") is not True:
        missing.append("archive_report")
    if _mapping(source.get("archive_rows")).get("loaded") is not True:
        missing.append("archive_rows")
    if _mapping(source.get("shadow_ranker_report")).get("loaded") is not True:
        missing.append("shadow_ranker_report")
    missing.extend(str(item) for item in _list(join_evidence.get("missing_evidence")))
    labels: list[str] = []
    if leakage.get("leakage_count"):
        labels.append("public_signal_leakage_detected")
    if missing:
        labels.append("missing_evidence_inconclusive")
    elif observability_summary.get("non_sparse_aliasing_support") is True:
        labels.append("public_recovery_signal_aliasing_detected")
    elif observability_summary.get("existing_public_signals_state_action_separable") is True:
        labels.append("existing_public_state_action_signal_separable")
    else:
        labels.append("public_recovery_signal_support_sparse_or_constant")
    labels.append("diagnostics_only_no_runtime_promotion")
    primary = labels[0]
    return {
        "primary": primary,
        "labels": labels,
        "missing_evidence": sorted(set(missing)),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    observability_summary: Mapping[str, object],
) -> dict[str, object]:
    if "missing_evidence_inconclusive" in set(_list(classification.get("labels"))):
        next_step = "reader_or_join_repair"
    elif observability_summary.get("non_sparse_aliasing_support") is True:
        next_step = "planner_may_prepare_observation_field_proposal"
    elif observability_summary.get("existing_public_signals_state_action_separable") is True:
        next_step = "state_action_interaction_diagnostic"
    else:
        next_step = "stop_current_ranker_path_and_plan_biology_observability_diagnostics"
    return {
        "next_step": next_step,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "summary": (
            "v112 is diagnostics-only; it does not provide runtime promotion evidence."
        ),
    }


def _oracle_actions(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if not rows:
        return ()
    best = min(_int(row.get("oracle_rank"), default=999) for row in rows)
    return tuple(
        sorted(
            {
                str(row.get("candidate_action"))
                for row in rows
                if _int(row.get("oracle_rank"), default=999) == best
                and _optional_string(row.get("candidate_action")) is not None
            },
            key=_action_sort_key,
        )
    )


def _target_example(target: TargetEvidence) -> dict[str, object]:
    return {
        "group_key": target.group_key,
        "oracle_actions": list(target.oracle_actions),
        "seed": target.provenance.get("seed"),
        "source_kind": target.provenance.get("source_kind"),
        "source_path": target.provenance.get("source_path"),
        "record_index": target.provenance.get("record_index"),
        "agent_id": target.provenance.get("agent_id"),
    }


def _join_missing_example(row: Mapping[str, object]) -> dict[str, object]:
    provenance = _mapping(row.get("provenance"))
    return {
        "archive_row_id": row.get("archive_row_id"),
        "source_path": provenance.get("source_path"),
        "record_index": provenance.get("record_index"),
        "agent_id": provenance.get("agent_id"),
        "tick": row.get("tick"),
        "observation_digest": row.get("observation_digest"),
    }


def _signature(
    values: Mapping[str, object],
    *,
    digits: int = ALIAS_QUANTIZATION_DIGITS,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (str(name), round(_number(value), digits))
        for name, value in sorted(values.items())
    )


def _signature_digest(signature: Sequence[Sequence[object]]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _max_cell_value(cells: Sequence[Mapping[str, object]], field: str) -> float:
    return _round(max((_number(cell.get(field)) for cell in cells), default=0.0))


def _open_input(path: str | Path) -> TextIO:
    resolved = Path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt", encoding="utf-8")
    return resolved.open("r", encoding="utf-8")


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _int_or_none(value: object) -> int | None:
    parsed = _int(value, default=-10**12)
    return None if parsed == -10**12 else parsed


def _positive_int(value: object, *, field: str) -> int:
    parsed = _int(value, default=-1)
    if parsed <= 0:
        raise FirstRecoveryPublicSignalAuditError(f"{field} must be positive")
    return parsed


def _number(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _round(value: float) -> float:
    return round(float(value), 6)


def _mean(values: Sequence[float] | object) -> float:
    parsed = [float(value) for value in values if value is not None and math.isfinite(float(value))]  # type: ignore[arg-type]
    return _round(sum(parsed) / float(len(parsed))) if parsed else 0.0


def _share(numerator: int, denominator: int) -> float:
    return _round(float(numerator) / float(denominator)) if denominator else 0.0


def _counter_to_dict(counter: Mapping[str, int] | Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _action_sort_key(action: str) -> int:
    try:
        return ACTION_NAMES.index(action)
    except ValueError:
        return len(ACTION_NAMES) + 1
