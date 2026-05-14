from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, deque
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    load_mind_v3_planner_distilled_artifact,
    planner_distilled_runtime_row,
    score_distilled_planner_artifact,
)

MIND_V3_V98_BROAD_TRANSFER_RESIDUAL_AUDIT_SCHEMA_VERSION = (
    "mind_v3_v98_broad_transfer_residual_audit_v1"
)
MIND_V3_V98_BROAD_SUPPORT_ARCHIVE_SCHEMA_VERSION = (
    "mind_v3_v98_broad_transfer_support_archive_v1"
)
MIND_V3_V98_SUPPORT_GATED_RESIDUAL_POLICY = (
    "v98_support_gated_planner_distilled_residual_feasibility_v1"
)
V98_STRICT_EXCLUDED_SEEDS: tuple[int, ...] = (5, 13, 19, 29, 37, 41, 43)
V98_MIN_SUPPORT_ROWS = 240
V98_MIN_SOURCE_SEEDS = 8
V98_MIN_REPOSITION_SHARE = 0.25
V98_MIN_MODE_COUNT = 4
V98_MAX_DOMINANT_SUPPORT_TEACHER_ACTION_SHARE = 0.50
V98_MAX_OVERRIDE_ACTION_SHARE = 0.50
V98_MAX_ABSTENTION_RATE = 0.90
V98_SUPPORT_DISTANCE_QUANTILE = 0.80
V98_MIN_PLANNER_SCORE_MARGIN = 0.05
V98_SUPPORT_TARGET_ROW_COUNT = 960
V98_LOCAL_ANTI_COLLAPSE_WINDOW = 48
V98_LOCAL_ANTI_COLLAPSE_WARMUP = 8
_TRAJECTORY_SEED_RE = re.compile(r"-(\d+)-(\d+)\.jsonl\.gz$")
_REQUIRED_SUPPORT_CATEGORIES = (
    "plant_food",
    "hydration",
    "movement",
    "reproduction_readiness",
    "pre_death",
    "recovery",
    "animal_resource",
)


class BroadTransferResidualAuditError(ValueError):
    pass


def load_json_mapping(source: str | Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BroadTransferResidualAuditError(f"failed to read JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BroadTransferResidualAuditError(
            f"invalid JSON in {path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BroadTransferResidualAuditError(f"JSON payload must be an object: {path}")
    return payload


def build_broad_transfer_residual_audit_report(
    *,
    v97_report: Mapping[str, object],
    v97_trajectory_dir: str | Path,
    planner_distilled_artifact: str | Path | Mapping[str, object],
    support_trajectory_dir: str | Path,
    support_target_row_count: int = V98_SUPPORT_TARGET_ROW_COUNT,
    min_support_rows: int = V98_MIN_SUPPORT_ROWS,
    min_source_seeds: int = V98_MIN_SOURCE_SEEDS,
    min_reposition_share: float = V98_MIN_REPOSITION_SHARE,
    max_dominant_support_teacher_action_share: float = (
        V98_MAX_DOMINANT_SUPPORT_TEACHER_ACTION_SHARE
    ),
    max_override_action_share: float = V98_MAX_OVERRIDE_ACTION_SHARE,
    max_abstention_rate: float = V98_MAX_ABSTENTION_RATE,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = load_mind_v3_planner_distilled_artifact(planner_distilled_artifact)
    v97_rows = load_v97_planner_broad_rows(v97_trajectory_dir)
    support_rows_all = load_linear_support_rows(support_trajectory_dir)
    support_leaked_seeds = sorted(
        {int(row["source_seed"]) for row in support_rows_all}
        & set(V98_STRICT_EXCLUDED_SEEDS)
    )
    successful_support_rows = [
        row
        for row in support_rows_all
        if int(row.get("terminal_alive_count", 0)) > 0
    ]
    selected_support_rows = select_support_archive_rows(
        successful_support_rows,
        target_count=support_target_row_count,
        max_dominant_action_share=max_dominant_support_teacher_action_share,
    )
    support_candidate_examples = build_support_candidate_examples(selected_support_rows)
    support_distance_threshold = _support_distance_threshold(
        selected_support_rows,
        support_candidate_examples,
    )
    support_archive = _support_archive(
        selected_support_rows,
        support_candidate_examples,
        support_distance_threshold=support_distance_threshold,
    )
    v97_collapse = _v97_collapse_analysis(
        v97_report=v97_report,
        v97_rows=v97_rows,
        artifact=artifact,
        support_candidate_examples=support_candidate_examples,
    )
    residual = _residual_gate_report(
        support_rows=selected_support_rows,
        support_candidate_examples=support_candidate_examples,
        artifact=artifact,
        support_distance_threshold=support_distance_threshold,
        planner_score_margin_threshold=V98_MIN_PLANNER_SCORE_MARGIN,
        max_override_action_share=max_override_action_share,
    )
    support_coverage = _support_coverage(
        rows=selected_support_rows,
        all_rows=successful_support_rows,
        leaked_seeds=support_leaked_seeds,
    )
    floors = {
        "no_strict_seed_leakage": True,
        "unsupported_proposed_action_count": 0,
        "broad_support_rows": int(min_support_rows),
        "non_strict_source_seeds": int(min_source_seeds),
        "movement_reposition_share": _round(min_reposition_share),
        "represented_mode_count": V98_MIN_MODE_COUNT,
        "dominant_support_teacher_action_share": _round(
            max_dominant_support_teacher_action_share
        ),
        "residual_override_action_share": _round(max_override_action_share),
        "residual_max_abstention_rate": _round(max_abstention_rate),
        "first_concrete_v97_collapse_pattern_required": True,
    }
    acceptance = _acceptance(
        coverage=support_coverage,
        residual=residual,
        collapse=v97_collapse,
        floors=floors,
    )
    support_probe = {
        "policy": MIND_V3_V98_SUPPORT_GATED_RESIDUAL_POLICY,
        "support_accuracy_floor": 1.0,
        "accuracy": 1.0
        if acceptance["v98_broad_transfer_residual_diagnostic_accepted"]
        else 0.0,
        "materially_supports_v99_residual": bool(
            acceptance["v98_broad_transfer_residual_diagnostic_accepted"]
        ),
        "runtime_policy_status": (
            "v99_allowed"
            if acceptance["v99_support_gated_residual_runtime_allowed"]
            else "rejected_no_runtime"
        ),
        "support_row_count": support_coverage["support_row_count"],
        "source_seed_count": support_coverage["source_seed_count"],
        "residual_abstention_rate": residual["residual_abstention_rate"],
        "residual_override_action_share": residual[
            "dominant_override_action_share"
        ],
        "blocker_count": acceptance["blocker_count"],
    }
    contract = {
        "schema_version": MIND_V3_V98_BROAD_TRANSFER_RESIDUAL_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_V98_SUPPORT_GATED_RESIDUAL_POLICY,
        "diagnostic_only": True,
        "runtime_policy_trained": False,
        "promotion_run_executed": False,
        "full_replacement_policy_allowed": False,
        "linear_default_residual_design": True,
        "strict_excluded_seeds": list(V98_STRICT_EXCLUDED_SEEDS),
        "support_split_policy": (
            "non_strict_successful_linear_broad_trajectories_only_v1"
        ),
        "residual_gate_contract": {
            "default_action": "existing_linear_mind_v3",
            "candidate_action": "v96_planner_distilled_scorer",
            "nearest_support_distance_quantile": V98_SUPPORT_DISTANCE_QUANTILE,
            "support_distance_threshold": support_distance_threshold,
            "planner_score_margin_threshold": V98_MIN_PLANNER_SCORE_MARGIN,
            "local_anti_collapse_window": V98_LOCAL_ANTI_COLLAPSE_WINDOW,
            "local_anti_collapse_warmup": V98_LOCAL_ANTI_COLLAPSE_WARMUP,
            "max_override_action_share": max_override_action_share,
            "uses_seed_id_as_runtime_feature": False,
            "uses_fixture_identity": False,
            "uses_branch_id": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_private_simulator_state": False,
            "uses_global_batch_quota": False,
            "uses_heuristic_fallback": False,
        },
        "acceptance_floors": floors,
    }
    report = {
        **contract,
        "source_v97_policy": _mapping(v97_report.get("policy")),
        "source_v97_promotion": _mapping(
            v97_report.get("v97_planner_distilled_promotion")
        ),
        "support_archive_summary": support_archive["summary"],
        "v97_collapse_analysis": v97_collapse,
        "support_coverage": support_coverage,
        "residual_gate_diagnostic": residual,
        "broad_transfer_residual_support_probe": support_probe,
        "acceptance": acceptance,
        "v98_broad_transfer_residual_diagnostic_accepted": bool(
            acceptance["v98_broad_transfer_residual_diagnostic_accepted"]
        ),
        "v99_support_gated_residual_runtime_allowed": bool(
            acceptance["v99_support_gated_residual_runtime_allowed"]
        ),
        "blocker_count": int(acceptance["blocker_count"]),
        "provenance": {
            "v97_trajectory_dir": str(v97_trajectory_dir),
            "support_trajectory_dir": str(support_trajectory_dir),
            "planner_artifact_schema_version": artifact.get("schema_version"),
            "support_archive_digest": stable_payload_digest(
                support_archive["summary"]
            ),
            "contract_digest": stable_payload_digest(contract),
        },
    }
    return report, support_archive


def write_broad_transfer_residual_audit_report(
    report: Mapping[str, object],
    *,
    output_path: str | Path,
    support_archive: Mapping[str, object] | None = None,
    support_archive_output_path: str | Path | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if support_archive is not None and support_archive_output_path is not None:
        archive_path = Path(support_archive_output_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("w", encoding="utf-8") as handle:
            json.dump(
                support_archive,
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")


def load_v97_planner_broad_rows(trajectory_dir: str | Path) -> list[dict[str, object]]:
    paths = _trajectory_paths(
        trajectory_dir,
        prefix="open-mind-v3",
        exclude_prefixes=("open-mind-v3-linear",),
    )
    return _load_trajectory_rows(paths, row_kind="v97_planner_broad")


def load_linear_support_rows(trajectory_dir: str | Path) -> list[dict[str, object]]:
    paths = _trajectory_paths(trajectory_dir, prefix="open-mind-v3-linear")
    return _load_trajectory_rows(paths, row_kind="linear_support")


def select_support_archive_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    target_count: int,
    max_dominant_action_share: float,
) -> list[dict[str, object]]:
    sorted_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row.get("source_seed", 0)),
            int(row.get("tick", 0)),
            int(row.get("agent_id", 0)),
            str(row.get("requested_action", "")),
        ),
    )
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, int, int]] = set()
    action_counts: Counter[str] = Counter()
    action_cap = max(1, int(math.floor(float(target_count) * max_dominant_action_share)))

    def try_add(row: Mapping[str, object]) -> bool:
        key = (
            int(row.get("source_seed", 0)),
            int(row.get("tick", 0)),
            int(row.get("agent_id", 0)),
        )
        if key in seen:
            return False
        action = str(row.get("requested_action", ""))
        if action_counts[action] >= action_cap:
            return False
        selected.append(dict(row))
        seen.add(key)
        action_counts.update([action])
        return True

    for category in _REQUIRED_SUPPORT_CATEGORIES:
        _round_robin_add(
            [row for row in sorted_rows if category in _string_list(row.get("categories"))],
            selected=selected,
            target_total=min(target_count, len(selected) + 48),
            try_add=try_add,
        )
    movement_target = int(math.ceil(float(target_count) * V98_MIN_REPOSITION_SHARE))
    if _mode_count(selected, "reposition") < movement_target:
        _round_robin_add(
            [row for row in sorted_rows if row.get("mode") == "reposition"],
            selected=selected,
            target_total=min(
                target_count,
                len(selected)
                + max(0, movement_target - _mode_count(selected, "reposition")),
            ),
            try_add=try_add,
        )
    for mode in ("conserve", "recover_hydration", "exploit_resource", "reposition"):
        _round_robin_add(
            [row for row in sorted_rows if row.get("mode") == mode],
            selected=selected,
            target_total=min(target_count, len(selected) + 160),
            try_add=try_add,
        )
    _round_robin_add(
        sorted_rows,
        selected=selected,
        target_total=target_count,
        try_add=try_add,
    )
    return selected


def build_support_candidate_examples(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        for action in _legal_actions(row):
            features = candidate_feature_vector(row, action)
            if not features:
                continue
            examples.append(
                {
                    "example_index": len(examples),
                    "support_row_index": row_index,
                    "source_seed": int(row.get("source_seed", 0)),
                    "tick": int(row.get("tick", 0)),
                    "agent_id": int(row.get("agent_id", 0)),
                    "action": action,
                    "mode": _action_option_mode(action),
                    "features": list(features),
                }
            )
    return examples


def _load_trajectory_rows(
    paths: Sequence[Path],
    *,
    row_kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(_load_single_trajectory_rows(path, row_kind=row_kind))
    terminal_alive_by_path = _terminal_alive_by_path(rows)
    for row in rows:
        row["terminal_alive_count"] = terminal_alive_by_path.get(
            str(row.get("trajectory_path")),
            0,
        )
    return rows


def _load_single_trajectory_rows(path: Path, *, row_kind: str) -> list[dict[str, object]]:
    seed = _seed_from_path(path)
    history_by_agent: dict[int, list[dict[str, object]]] = {}
    ticks_since_resource: dict[int, int | None] = {}
    ticks_since_drink: dict[int, int | None] = {}
    record_index = 0
    rows: list[dict[str, object]] = []
    with _open_trajectory(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                continue
            if payload.get("type") == "header":
                seed = _seed_from_header(payload, fallback=seed)
                continue
            if payload.get("type") != "record":
                continue
            record = _mapping(payload.get("record"))
            agent_id = _int(record.get("agent_id"))
            tick = _int(record.get("tick"))
            history = _project_public_history(
                history_by_agent.get(agent_id, []),
                current_tick=tick,
                current_record_index=record_index,
            )
            observation_input = _mapping(record.get("observation_input"))
            action_mask = _complete_action_mask(_mapping(record.get("action_mask")))
            runtime_row = planner_distilled_runtime_row(
                observation_input=observation_input,
                action_mask=action_mask,
                public_history_trace=history,
            )
            requested_action = _record_requested_action(record)
            resolved_action = _record_resolved_action(record)
            categories = _support_categories(record, runtime_row)
            before = _mapping(record.get("before"))
            after = _mapping(record.get("after"))
            row = {
                **runtime_row,
                "row_kind": row_kind,
                "source_seed": seed,
                "trajectory_path": str(path),
                "tick": tick,
                "agent_id": agent_id,
                "requested_action": requested_action,
                "resolved_action": resolved_action,
                "mode": _action_option_mode(requested_action),
                "categories": categories,
                "action_valid": bool(record.get("action_valid", False)),
                "resolution_action_valid": bool(
                    record.get("resolution_action_valid", False)
                ),
                "before": _vitals_snapshot(before),
                "after": _vitals_snapshot(after),
                "outcome": _outcome_summary(record),
                "policy_decision_diagnostics": dict(
                    _mapping(record.get("policy_decision_diagnostics"))
                ),
            }
            rows.append(row)
            item = _public_history_item_from_record(
                record,
                record_index=record_index,
                ticks_since_animal_resource_gain=ticks_since_resource.get(agent_id),
                ticks_since_drink=ticks_since_drink.get(agent_id),
            )
            agent_history = history_by_agent.setdefault(agent_id, [])
            agent_history.append(item)
            if len(agent_history) > 8:
                del agent_history[0 : len(agent_history) - 8]
            if _record_consumed_animal_resource(record):
                ticks_since_resource[agent_id] = 0
            else:
                previous = ticks_since_resource.get(agent_id)
                ticks_since_resource[agent_id] = (
                    None if previous is None else previous + 1
                )
            if _record_drank(record):
                ticks_since_drink[agent_id] = 0
            else:
                previous_drink = ticks_since_drink.get(agent_id)
                ticks_since_drink[agent_id] = (
                    None if previous_drink is None else previous_drink + 1
                )
            record_index += 1
    return rows


def _v97_collapse_analysis(
    *,
    v97_report: Mapping[str, object],
    v97_rows: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
    support_candidate_examples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_by_seed_tick = _action_distribution_by_seed_tick_band(v97_rows)
    first_deaths = _first_death_windows(v97_rows)
    unsupported = _unsupported_contexts(v97_rows)
    high_margin_stay = _high_margin_stay_decisions(v97_rows)
    component_breakdown = _scorer_component_breakdown(v97_rows)
    v96_examples = _list_of_mappings(artifact.get("sequence_support_examples"))
    nearest_v96 = _nearest_support_summary(v97_rows, v96_examples)
    nearest_broad = _nearest_support_summary(v97_rows, support_candidate_examples)
    first_collapse = high_margin_stay[0] if high_margin_stay else None
    promotion = _mapping(v97_report.get("v97_planner_distilled_promotion"))
    return {
        "v97_promotion_blocker_count": len(_list_of_mappings(promotion.get("blockers"))),
        "action_distribution_by_seed_tick_band": action_by_seed_tick,
        "first_death_windows": first_deaths,
        "unsupported_action_contexts": unsupported,
        "high_margin_stay_decision_count": len(high_margin_stay),
        "high_margin_stay_examples": high_margin_stay[:24],
        "first_concrete_collapse_pattern": first_collapse,
        "scorer_component_breakdown": component_breakdown,
        "nearest_v96_support_distance": nearest_v96,
        "nearest_broad_support_distance": nearest_broad,
    }


def _residual_gate_report(
    *,
    support_rows: Sequence[Mapping[str, object]],
    support_candidate_examples: Sequence[Mapping[str, object]],
    artifact: Mapping[str, object],
    support_distance_threshold: float,
    planner_score_margin_threshold: float,
    max_override_action_share: float,
) -> dict[str, object]:
    override_window: deque[str] = deque(maxlen=V98_LOCAL_ANTI_COLLAPSE_WINDOW)
    override_action_counts: Counter[str] = Counter()
    final_action_counts: Counter[str] = Counter()
    abstain_reasons: Counter[str] = Counter()
    unsupported_proposed = 0
    identity_count = 0
    applied_count = 0
    abstained_count = 0
    examples: list[dict[str, object]] = []
    rows = sorted(
        support_rows,
        key=lambda row: (
            int(row.get("source_seed", 0)),
            int(row.get("tick", 0)),
            int(row.get("agent_id", 0)),
        ),
    )
    for row in rows:
        linear_action = str(row.get("requested_action", "stay"))
        scored = score_distilled_planner_artifact(row=row, artifact=artifact)
        candidate = scored.get("selected_action")
        candidate_action = str(candidate) if isinstance(candidate, str) else ""
        candidate_scores = _list_of_mappings(scored.get("candidate_scores"))
        margin = _score_margin(candidate_scores)
        legal = candidate_action in _legal_actions(row)
        if not legal:
            unsupported_proposed += 1
        if not candidate_action or candidate_action == linear_action:
            identity_count += 1
            final_action_counts.update([linear_action])
            continue
        distance = _nearest_support_distance(
            candidate_feature_vector(row, candidate_action),
            support_candidate_examples,
            action=candidate_action,
            exclude_source_seed=int(row.get("source_seed", 0)),
        )
        allowed, reason = _residual_gate_allows(
            candidate_action=candidate_action,
            legal=legal,
            distance=distance,
            support_distance_threshold=support_distance_threshold,
            margin=margin,
            planner_score_margin_threshold=planner_score_margin_threshold,
            override_window=override_window,
            max_override_action_share=max_override_action_share,
        )
        if allowed:
            applied_count += 1
            override_window.append(candidate_action)
            override_action_counts.update([candidate_action])
            final_action_counts.update([candidate_action])
        else:
            abstained_count += 1
            abstain_reasons.update([reason])
            final_action_counts.update([linear_action])
        if len(examples) < 24 and (allowed or reason != "below_margin"):
            examples.append(
                {
                    "source_seed": int(row.get("source_seed", 0)),
                    "tick": int(row.get("tick", 0)),
                    "agent_id": int(row.get("agent_id", 0)),
                    "linear_action": linear_action,
                    "candidate_action": candidate_action,
                    "allowed": allowed,
                    "reason": reason,
                    "support_distance": _round(distance),
                    "score_margin": _round(margin),
                    "top_scores": [
                        _score_summary(item)
                        for item in sorted(
                            candidate_scores,
                            key=lambda item: _float(item.get("final_score")),
                            reverse=True,
                        )[:3]
                    ],
                }
            )
    opportunity_count = applied_count + abstained_count
    residual_abstention_rate = (
        _round(abstained_count / float(opportunity_count))
        if opportunity_count
        else 0.0
    )
    dominant_override = _dominant_count_share(override_action_counts)
    dominant_final = _dominant_count_share(final_action_counts)
    return {
        "policy": MIND_V3_V98_SUPPORT_GATED_RESIDUAL_POLICY,
        "evaluated_support_row_count": len(rows),
        "identity_with_linear_count": identity_count,
        "non_identity_opportunity_count": opportunity_count,
        "override_applied_count": applied_count,
        "override_abstained_count": abstained_count,
        "residual_abstention_rate": residual_abstention_rate,
        "mostly_abstains": bool(
            opportunity_count >= 40 and residual_abstention_rate > V98_MAX_ABSTENTION_RATE
        ),
        "unsupported_proposed_action_count": unsupported_proposed,
        "override_action_counts": dict(sorted(override_action_counts.items())),
        "dominant_override_action": dominant_override["action"],
        "dominant_override_action_count": dominant_override["count"],
        "dominant_override_action_share": dominant_override["share"],
        "final_action_counts_if_applied_to_support": dict(
            sorted(final_action_counts.items())
        ),
        "dominant_final_action": dominant_final["action"],
        "dominant_final_action_share": dominant_final["share"],
        "abstain_reason_counts": dict(sorted(abstain_reasons.items())),
        "gate_examples": examples,
    }


def _residual_gate_allows(
    *,
    candidate_action: str,
    legal: bool,
    distance: float,
    support_distance_threshold: float,
    margin: float,
    planner_score_margin_threshold: float,
    override_window: deque[str],
    max_override_action_share: float,
) -> tuple[bool, str]:
    if not legal:
        return False, "unsupported_action"
    if not math.isfinite(distance) or distance > support_distance_threshold:
        return False, "outside_support"
    if margin < planner_score_margin_threshold:
        return False, "below_margin"
    projected = list(override_window) + [candidate_action]
    if len(projected) >= V98_LOCAL_ANTI_COLLAPSE_WARMUP:
        counts = Counter(projected)
        if max(counts.values()) / float(len(projected)) > max_override_action_share:
            return False, "local_action_collapse_cap"
    return True, "allowed"


def _support_archive(
    rows: Sequence[Mapping[str, object]],
    examples: Sequence[Mapping[str, object]],
    *,
    support_distance_threshold: float,
) -> dict[str, object]:
    coverage = _support_coverage(rows=rows, all_rows=rows, leaked_seeds=[])
    summary = {
        "schema_version": MIND_V3_V98_BROAD_SUPPORT_ARCHIVE_SCHEMA_VERSION,
        "support_row_count": len(rows),
        "support_candidate_example_count": len(examples),
        "source_seeds": sorted({int(row.get("source_seed", 0)) for row in rows}),
        "support_distance_threshold": support_distance_threshold,
        "coverage": coverage,
    }
    archive_rows = [
        {
            "support_row_index": index,
            "source_seed": int(row.get("source_seed", 0)),
            "tick": int(row.get("tick", 0)),
            "agent_id": int(row.get("agent_id", 0)),
            "teacher_action": str(row.get("requested_action", "")),
            "teacher_mode": str(row.get("mode", "")),
            "categories": _string_list(row.get("categories")),
            "feature_vector": list(
                candidate_feature_vector(row, str(row.get("requested_action", "")))
            ),
        }
        for index, row in enumerate(rows)
    ]
    return {
        "schema_version": MIND_V3_V98_BROAD_SUPPORT_ARCHIVE_SCHEMA_VERSION,
        "summary": summary,
        "rows": archive_rows,
        "candidate_examples": list(examples),
    }


def _support_coverage(
    *,
    rows: Sequence[Mapping[str, object]],
    all_rows: Sequence[Mapping[str, object]],
    leaked_seeds: Sequence[int],
) -> dict[str, object]:
    action_counts = Counter(str(row.get("requested_action", "")) for row in rows)
    mode_counts = Counter(str(row.get("mode", "")) for row in rows)
    category_counts: Counter[str] = Counter()
    for row in rows:
        category_counts.update(_string_list(row.get("categories")))
    dominant = _dominant_count_share(action_counts)
    source_seeds = sorted({int(row.get("source_seed", 0)) for row in rows})
    return {
        "source_policy": "linear_mind_v3",
        "all_successful_linear_row_count": len(all_rows),
        "support_row_count": len(rows),
        "source_seeds": source_seeds,
        "source_seed_count": len(source_seeds),
        "strict_excluded_seeds": list(V98_STRICT_EXCLUDED_SEEDS),
        "strict_seed_leakage_seeds": list(leaked_seeds),
        "strict_seed_leakage": bool(leaked_seeds),
        "teacher_action_counts": dict(sorted(action_counts.items())),
        "dominant_support_teacher_action": dominant["action"],
        "dominant_support_teacher_action_share": dominant["share"],
        "teacher_mode_counts": dict(sorted(mode_counts.items())),
        "represented_mode_count": len([mode for mode, count in mode_counts.items() if count]),
        "movement_reposition_row_count": int(mode_counts.get("reposition", 0)),
        "movement_reposition_row_share": _round(
            float(mode_counts.get("reposition", 0)) / float(len(rows))
        )
        if rows
        else 0.0,
        "category_counts": dict(sorted(category_counts.items())),
        "branch_outcomes_generated": False,
        "replay_verified_branch_outcomes_not_required": True,
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    residual: Mapping[str, object],
    collapse: Mapping[str, object],
    floors: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []

    def block(reason: str, metric: str, value: object, floor: object, comparator: str) -> None:
        blockers.append(
            {
                "reason": reason,
                "metric": metric,
                "value": value,
                "floor": floor,
                "comparator": comparator,
            }
        )

    if coverage.get("strict_seed_leakage") is True:
        block(
            "strict_seed_leakage",
            "strict_seed_leakage",
            coverage.get("strict_seed_leakage_seeds"),
            [],
            "eq",
        )
    if int(residual.get("unsupported_proposed_action_count", 0)) != 0:
        block(
            "unsupported_proposed_actions",
            "unsupported_proposed_action_count",
            residual.get("unsupported_proposed_action_count"),
            0,
            "eq",
        )
    if int(coverage.get("support_row_count", 0)) < int(floors["broad_support_rows"]):
        block(
            "insufficient_broad_support_rows",
            "support_row_count",
            coverage.get("support_row_count"),
            floors["broad_support_rows"],
            "ge",
        )
    if int(coverage.get("source_seed_count", 0)) < int(floors["non_strict_source_seeds"]):
        block(
            "insufficient_non_strict_source_seeds",
            "source_seed_count",
            coverage.get("source_seed_count"),
            floors["non_strict_source_seeds"],
            "ge",
        )
    if _float(coverage.get("movement_reposition_row_share")) < _float(
        floors["movement_reposition_share"]
    ):
        block(
            "insufficient_movement_reposition_support",
            "movement_reposition_row_share",
            coverage.get("movement_reposition_row_share"),
            floors["movement_reposition_share"],
            "ge",
        )
    if int(coverage.get("represented_mode_count", 0)) < int(
        floors["represented_mode_count"]
    ):
        block(
            "insufficient_mode_coverage",
            "represented_mode_count",
            coverage.get("represented_mode_count"),
            floors["represented_mode_count"],
            "ge",
        )
    if _float(coverage.get("dominant_support_teacher_action_share")) > _float(
        floors["dominant_support_teacher_action_share"]
    ):
        block(
            "support_teacher_action_collapse",
            "dominant_support_teacher_action_share",
            coverage.get("dominant_support_teacher_action_share"),
            floors["dominant_support_teacher_action_share"],
            "le",
        )
    if _float(residual.get("dominant_override_action_share")) > _float(
        floors["residual_override_action_share"]
    ):
        block(
            "residual_override_action_collapse",
            "dominant_override_action_share",
            residual.get("dominant_override_action_share"),
            floors["residual_override_action_share"],
            "le",
        )
    if residual.get("mostly_abstains") is True:
        block(
            "residual_gate_mostly_abstains",
            "residual_abstention_rate",
            residual.get("residual_abstention_rate"),
            floors["residual_max_abstention_rate"],
            "le",
        )
    if collapse.get("first_concrete_collapse_pattern") is None:
        block(
            "v97_collapse_pattern_not_identified",
            "first_concrete_collapse_pattern",
            None,
            True,
            "present",
        )
    accepted = not blockers
    return {
        "v98_broad_transfer_residual_diagnostic_accepted": accepted,
        "v99_support_gated_residual_runtime_allowed": accepted,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "v99_recommendation": (
            "Implement opt-in support-gated planner-distilled residual runtime: "
            "linear Mind v3 remains default; planner override only inside "
            "non-strict broad support with margin/legal/anti-collapse gates."
            if accepted
            else "Do not implement v99 runtime from this diagnostic result."
        ),
    }


def _support_distance_threshold(
    rows: Sequence[Mapping[str, object]],
    examples: Sequence[Mapping[str, object]],
) -> float:
    distances: list[float] = []
    for row in rows:
        action = str(row.get("requested_action", ""))
        features = candidate_feature_vector(row, action)
        distance = _nearest_support_distance(
            features,
            examples,
            action=action,
            exclude_source_seed=int(row.get("source_seed", 0)),
        )
        if math.isfinite(distance):
            distances.append(distance)
    return _round(_quantile(distances, V98_SUPPORT_DISTANCE_QUANTILE))


def _nearest_support_distance(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    action: str,
    exclude_source_seed: int | None = None,
) -> float:
    if not features:
        return float("inf")
    best = float("inf")
    for example in examples:
        if str(example.get("action", "")) != action:
            continue
        if exclude_source_seed is not None and int(example.get("source_seed", -1)) == exclude_source_seed:
            continue
        distance = _squared_distance(features, _float_tuple(example.get("features")))
        if distance < best:
            best = distance
    return best


def _nearest_support_summary(
    rows: Sequence[Mapping[str, object]],
    examples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    distances: list[float] = []
    by_action: dict[str, list[float]] = {}
    high_margin_stay_distances: list[float] = []
    for row in rows:
        action = str(row.get("requested_action", ""))
        features = candidate_feature_vector(row, action)
        distance = _nearest_support_distance(features, examples, action=action)
        if not math.isfinite(distance):
            continue
        distances.append(distance)
        by_action.setdefault(action, []).append(distance)
        if action == "stay" and _planner_margin(row) >= V98_MIN_PLANNER_SCORE_MARGIN:
            high_margin_stay_distances.append(distance)
    return {
        "decision_count_with_distance": len(distances),
        "overall": _distance_summary(distances),
        "by_action": {
            action: _distance_summary(values)
            for action, values in sorted(by_action.items())
        },
        "high_margin_stay": _distance_summary(high_margin_stay_distances),
    }


def _action_distribution_by_seed_tick_band(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], Counter[str]] = {}
    for row in rows:
        seed = int(row.get("source_seed", 0))
        band = _tick_band(int(row.get("tick", 0)))
        grouped.setdefault((seed, band), Counter()).update(
            [str(row.get("requested_action", ""))]
        )
    return [
        {
            "seed": seed,
            "tick_band": band,
            "requested_action_counts": dict(sorted(counter.items())),
            "dominant_action": _dominant_count_share(counter)["action"],
            "dominant_action_share": _dominant_count_share(counter)["share"],
        }
        for (seed, band), counter in sorted(grouped.items())
    ]


def _first_death_windows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    previous_by_seed_agent: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    first_by_seed: dict[int, dict[str, object]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("source_seed", 0)),
            int(item.get("tick", 0)),
            int(item.get("agent_id", 0)),
        ),
    ):
        seed = int(row.get("source_seed", 0))
        agent = int(row.get("agent_id", 0))
        key = (seed, agent)
        outcome = _mapping(row.get("outcome"))
        after = _mapping(row.get("after"))
        died = outcome.get("died") is True or after.get("alive") is False
        if died and seed not in first_by_seed:
            window = list(previous_by_seed_agent.get(key, []))[-5:] + [row]
            first_by_seed[seed] = {
                "seed": seed,
                "tick": int(row.get("tick", 0)),
                "agent_id": agent,
                "death_record": _row_context(row),
                "pre_death_window": [_row_context(item) for item in window],
            }
        history = previous_by_seed_agent.setdefault(key, [])
        history.append(row)
        if len(history) > 8:
            del history[0 : len(history) - 8]
    return [first_by_seed[seed] for seed in sorted(first_by_seed)]


def _unsupported_contexts(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    contexts = []
    for row in rows:
        if row.get("action_valid") is True and row.get("resolution_action_valid") is True:
            continue
        contexts.append(
            {
                **_row_context(row),
                "action_valid": bool(row.get("action_valid", False)),
                "resolution_action_valid": bool(
                    row.get("resolution_action_valid", False)
                ),
                "legal_actions": _legal_actions(row),
                "outcome": row.get("outcome"),
            }
        )
    return contexts[:24]


def _high_margin_stay_decisions(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    examples = []
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("source_seed", 0)),
            int(item.get("tick", 0)),
            int(item.get("agent_id", 0)),
        ),
    ):
        if row.get("requested_action") != "stay":
            continue
        margin = _planner_margin(row)
        if margin < V98_MIN_PLANNER_SCORE_MARGIN:
            continue
        legal = _legal_actions(row)
        if not any(action in legal for action in ("eat", "drink")) and not any(
            action.startswith("move_") for action in legal
        ):
            continue
        examples.append(
            {
                **_row_context(row),
                "score_margin": margin,
                "legal_actions": legal,
                "top_scores": _top_score_summaries(row),
            }
        )
    return examples


def _scorer_component_breakdown(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_action: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        top = _top_score_summaries(row)
        if not top:
            continue
        action = str(top[0].get("action", ""))
        by_action.setdefault(action, []).append(top[0])
    return {
        action: {
            "count": len(items),
            "final_score_mean": _round(_mean([_float(item.get("final_score")) for item in items])),
            "sequence_cvar_score_mean": _round(
                _mean([_float(item.get("sequence_cvar_score")) for item in items])
            ),
            "teacher_margin_mean": _round(
                _mean([
                    _float(item.get("utility_weighted_teacher_margin"))
                    for item in items
                ])
            ),
            "learned_action_penalty_mean": _round(
                _mean([_float(item.get("learned_action_penalty")) for item in items])
            ),
        }
        for action, items in sorted(by_action.items())
    }


def _top_score_summaries(row: Mapping[str, object]) -> list[dict[str, object]]:
    diagnostics = _mapping(row.get("policy_decision_diagnostics"))
    scores = _list_of_mappings(diagnostics.get("planner_distilled_candidate_scores_top"))
    return [_score_summary(item) for item in scores]


def _score_summary(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "action": str(item.get("action", "")),
        "mode": str(item.get("mode", "")),
        "final_score": _round(_float(item.get("final_score"))),
        "sequence_cvar_score": _round(_float(item.get("sequence_cvar_score"))),
        "utility_weighted_teacher_margin": _round(
            _float(item.get("utility_weighted_teacher_margin"))
        ),
        "learned_action_penalty": _round(_float(item.get("learned_action_penalty"))),
    }


def _row_context(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "seed": int(row.get("source_seed", 0)),
        "tick": int(row.get("tick", 0)),
        "agent_id": int(row.get("agent_id", 0)),
        "requested_action": str(row.get("requested_action", "")),
        "resolved_action": str(row.get("resolved_action", "")),
        "before": row.get("before"),
        "after": row.get("after"),
    }


def _trajectory_paths(
    trajectory_dir: str | Path,
    *,
    prefix: str,
    exclude_prefixes: Sequence[str] = (),
) -> list[Path]:
    root = Path(trajectory_dir)
    if not root.exists():
        raise BroadTransferResidualAuditError(
            f"trajectory directory does not exist: {root}"
        )
    paths = []
    for path in sorted(root.glob(f"{prefix}-*.jsonl.gz")):
        if any(path.name.startswith(excluded) for excluded in exclude_prefixes):
            continue
        paths.append(path)
    if not paths:
        raise BroadTransferResidualAuditError(
            f"no trajectory files found for prefix {prefix} in {root}"
        )
    return paths


def _seed_from_path(path: Path) -> int:
    match = _TRAJECTORY_SEED_RE.search(path.name)
    if match is None:
        return 0
    return int(match.group(1))


def _seed_from_header(header: Mapping[str, object], *, fallback: int) -> int:
    config = _mapping(header.get("config"))
    value = config.get("seed")
    return _int(value) if value is not None else fallback


def _terminal_alive_by_path(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    state: dict[str, dict[int, bool]] = {}
    for row in rows:
        path = str(row.get("trajectory_path", ""))
        agent = int(row.get("agent_id", 0))
        after = _mapping(row.get("after"))
        alive = after.get("alive")
        if isinstance(alive, bool):
            state.setdefault(path, {})[agent] = alive
    return {
        path: sum(1 for alive in agents.values() if alive)
        for path, agents in state.items()
    }


def _project_public_history(
    history: Sequence[Mapping[str, object]],
    *,
    current_tick: int,
    current_record_index: int,
) -> list[dict[str, object]]:
    projected = []
    for item in history[-8:]:
        copied = dict(item)
        copied["tick_delta"] = current_tick - int(item.get("tick", 0))
        copied["record_index_delta"] = current_record_index - int(
            item.get("record_index", 0)
        )
        projected.append(copied)
    return projected


def _public_history_item_from_record(
    record: Mapping[str, object],
    *,
    record_index: int,
    ticks_since_animal_resource_gain: int | None,
    ticks_since_drink: int | None,
) -> dict[str, object]:
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    passive = _mapping(outcome.get("passive"))
    return {
        "tick": _int(record.get("tick")),
        "tick_delta": 0,
        "record_index": int(record_index),
        "record_index_delta": 0,
        "requested_action": _record_requested_action(record),
        "resolved_action": _record_resolved_action(record),
        "action_valid": bool(record.get("action_valid", False)),
        "resolution_action_valid": bool(record.get("resolution_action_valid", False)),
        "moved": bool(record.get("moved", False)),
        "x_delta": _int(after.get("x")) - _int(before.get("x")),
        "y_delta": _int(after.get("y")) - _int(before.get("y")),
        "energy_ratio_before": _optional_float(before.get("energy_ratio")),
        "energy_ratio_after": _optional_float(after.get("energy_ratio")),
        "energy_ratio_delta": _optional_delta(after, before, "energy_ratio"),
        "hydration_ratio_before": _optional_float(before.get("hydration_ratio")),
        "hydration_ratio_after": _optional_float(after.get("hydration_ratio")),
        "hydration_ratio_delta": _optional_delta(after, before, "hydration_ratio"),
        "health_ratio_before": _optional_float(before.get("health_ratio")),
        "health_ratio_after": _optional_float(after.get("health_ratio")),
        "health_ratio_delta": _optional_delta(after, before, "health_ratio"),
        "resource_gain": _optional_float(outcome.get("resource_gain")),
        "drank": bool(drinking.get("drank", False)),
        "ate": bool(feeding.get("ate", False)),
        "died": bool(outcome.get("died", False)),
        "death_cause": str(passive.get("death_cause"))
        if isinstance(passive.get("death_cause"), str)
        else None,
        "died_after_action": bool(passive.get("died_after_action", False)),
        "post_carrion_contact": _record_consumed_animal_resource(record),
        "ticks_since_animal_resource_gain": ticks_since_animal_resource_gain,
        "ticks_since_drink": ticks_since_drink,
    }


def _support_categories(
    record: Mapping[str, object],
    runtime_row: Mapping[str, object],
) -> list[str]:
    action = _record_requested_action(record)
    action_mask = _mapping(record.get("action_mask"))
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    compact = _mapping(runtime_row.get("compact_state"))
    center = _mapping(compact.get("center"))
    local = _mapping(compact.get("local"))
    categories: set[str] = set()
    if action == "eat" or bool(action_mask.get("eat", False)) or _record_ate(record):
        categories.add("plant_food")
    if (
        action == "drink"
        or bool(action_mask.get("drink", False))
        or _float(before.get("hydration_ratio")) < 0.5
        or _record_drank(record)
    ):
        categories.add("hydration")
    if action.startswith("move_") or any(
        bool(action_mask.get(move, False))
        for move in ("move_north", "move_south", "move_east", "move_west")
    ):
        categories.add("movement")
    reward = _mapping(record.get("reward"))
    reward_components = _mapping(reward.get("components"))
    if (
        bool(outcome.get("reproduced", False))
        or bool(outcome.get("reproduction_ready_after", False))
        or abs(_float(reward_components.get("reproduction_readiness"))) > 0.0
        or (
            _float(before.get("energy_ratio")) >= 0.65
            and _float(before.get("hydration_ratio")) >= 0.65
            and _float(before.get("health_ratio")) >= 0.70
        )
    ):
        categories.add("reproduction_readiness")
    if (
        after.get("alive") is False
        or min(
            _float(before.get("energy_ratio")),
            _float(before.get("hydration_ratio")),
            _float(before.get("health_ratio")),
        )
        < 0.28
    ):
        categories.add("pre_death")
    if (
        _float(outcome.get("resource_gain")) > 0.0
        or _optional_delta(after, before, "energy_ratio") is not None
        and (_optional_delta(after, before, "energy_ratio") or 0.0) > 0.0
        or _optional_delta(after, before, "hydration_ratio") is not None
        and (_optional_delta(after, before, "hydration_ratio") or 0.0) > 0.0
    ):
        categories.add("recovery")
    if (
        _record_consumed_animal_resource(record)
        or _cell_carrion(center) > 0.0
        or _float(local.get("radius1_carrion")) > 0.0
        or _float(local.get("radius2_carrion")) > 0.0
    ):
        categories.add("animal_resource")
    return sorted(categories)


def _record_ate(record: Mapping[str, object]) -> bool:
    return bool(_mapping(_mapping(record.get("outcome")).get("feeding")).get("ate", False))


def _record_drank(record: Mapping[str, object]) -> bool:
    return bool(
        _mapping(_mapping(record.get("outcome")).get("drinking")).get("drank", False)
    )


def _record_consumed_animal_resource(record: Mapping[str, object]) -> bool:
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    if feeding.get("food_source") not in {"carcass", "fresh_kill"}:
        return False
    return bool(feeding.get("ate", False)) or _float(outcome.get("resource_gain")) > 0.0


def _record_requested_action(record: Mapping[str, object]) -> str:
    action = record.get("requested_action")
    return str(action) if isinstance(action, str) and action else "stay"


def _record_resolved_action(record: Mapping[str, object]) -> str:
    action = record.get("resolved_action", record.get("requested_action"))
    return str(action) if isinstance(action, str) and action else "stay"


def _outcome_summary(record: Mapping[str, object]) -> dict[str, object]:
    outcome = _mapping(record.get("outcome"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    passive = _mapping(outcome.get("passive"))
    return {
        "died": bool(outcome.get("died", False)),
        "died_after_action": bool(passive.get("died_after_action", False)),
        "death_cause": passive.get("death_cause"),
        "resource_gain": _round(_float(outcome.get("resource_gain"))),
        "ate": bool(feeding.get("ate", False)),
        "food_source": feeding.get("food_source"),
        "drank": bool(drinking.get("drank", False)),
        "reproduced": bool(outcome.get("reproduced", False)),
        "reproduction_ready_after": bool(
            outcome.get("reproduction_ready_after", False)
        ),
    }


def _vitals_snapshot(state: Mapping[str, object]) -> dict[str, object]:
    return {
        "alive": state.get("alive"),
        "x": _int(state.get("x")),
        "y": _int(state.get("y")),
        "energy_ratio": _round(_float(state.get("energy_ratio"))),
        "hydration_ratio": _round(_float(state.get("hydration_ratio"))),
        "health_ratio": _round(_float(state.get("health_ratio"))),
    }


def _complete_action_mask(raw: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(raw.get(action, False)) for action in ACTION_NAMES}


def _legal_actions(row: Mapping[str, object]) -> list[str]:
    action_mask = _mapping(row.get("action_mask"))
    return [action for action in ACTION_NAMES if bool(action_mask.get(action, False))]


def _action_option_mode(action: str) -> str:
    if action == "drink":
        return "recover_hydration"
    if action == "eat":
        return "exploit_resource"
    if action == "stay":
        return "conserve"
    if action.startswith("move_"):
        return "reposition"
    return "other"


def _cell_carrion(cell: Mapping[str, object]) -> float:
    return max(
        _float(cell.get("fresh_kill")),
        _float(cell.get("carcass")),
        _float(cell.get("carrion_signal")),
    )


def _round_robin_add(
    candidates: Sequence[Mapping[str, object]],
    *,
    selected: Sequence[Mapping[str, object]],
    target_total: int,
    try_add,
) -> None:
    if len(selected) >= target_total:
        return
    by_seed: dict[int, list[Mapping[str, object]]] = {}
    for row in candidates:
        by_seed.setdefault(int(row.get("source_seed", 0)), []).append(row)
    positions = {seed: 0 for seed in by_seed}
    while len(selected) < target_total:
        added = False
        for seed in sorted(by_seed):
            rows = by_seed[seed]
            while positions[seed] < len(rows):
                row = rows[positions[seed]]
                positions[seed] += 1
                if try_add(row):
                    added = True
                    break
            if len(selected) >= target_total:
                break
        if not added:
            break


def _mode_count(rows: Sequence[Mapping[str, object]], mode: str) -> int:
    return sum(1 for row in rows if row.get("mode") == mode)


def _planner_margin(row: Mapping[str, object]) -> float:
    diagnostics = _mapping(row.get("policy_decision_diagnostics"))
    return _round(_float(diagnostics.get("planner_distilled_score_margin")))


def _score_margin(candidate_scores: Sequence[Mapping[str, object]]) -> float:
    sorted_scores = sorted(
        candidate_scores,
        key=lambda item: _float(item.get("final_score")),
        reverse=True,
    )
    if len(sorted_scores) < 2:
        return 0.0
    return _round(
        _float(sorted_scores[0].get("final_score"))
        - _float(sorted_scores[1].get("final_score"))
    )


def _distance_summary(values: Sequence[float]) -> dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(finite),
        "mean": _round(_mean(finite)),
        "p50": _round(_quantile(finite, 0.50)),
        "p90": _round(_quantile(finite, 0.90)),
        "max": _round(max(finite) if finite else 0.0),
    }


def _dominant_count_share(counter: Counter[str]) -> dict[str, object]:
    total = sum(counter.values())
    if total <= 0:
        return {"action": None, "count": 0, "share": 0.0}
    action, count = max(counter.items(), key=lambda item: (item[1], item[0]))
    return {"action": action, "count": int(count), "share": _round(count / total)}


def _tick_band(tick: int) -> str:
    start = (int(tick) // 30) * 30
    return f"{start:03d}-{start + 29:03d}"


def _optional_delta(
    after: Mapping[str, object],
    before: Mapping[str, object],
    key: str,
) -> float | None:
    left = _optional_float(after.get(key))
    right = _optional_float(before.get(key))
    if left is None or right is None:
        return None
    return left - right


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(float(item) for item in value)


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _quantile(values: Sequence[float], fraction: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    index = min(
        len(finite) - 1,
        max(0, int(math.ceil(float(len(finite)) * fraction)) - 1),
    )
    return finite[index]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_trajectory(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8")
