from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v161_shadow_eval import (
    DEFAULT_TRAJECTORY_GLOB,
    _action_or_empty,
    load_shadow_evidence,
)
from evolution_sim.mind.carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy import (
    shadow_tie_collapse_autopsy_records,
)
from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V163_REPORT_PATH,
    DEFAULT_TICKS,
    M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
    PREFERRED_NEAREST_ROWS,
    PREFERRED_TIED_SET_KEYS,
    STRICT_BROAD_SEEDS,
    V163SelectedBranchPoint,
    _bool_action_mask,
    _branch_run_digest_payload,
    _candidate_set_key,
    _record_materialization_payload,
    _source_v162_digest_validation,
    _v162_lifecycle_validation,
    evaluate_materialized_branch_points,
    materialize_selected_branch_points,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_preterminal_tied_set_branch_target_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_preterminal_tied_set_branch_target_expansion_v1"
)
EXPECTED_V163_CLASSIFICATION = "branch_target_support_insufficient_no_live_ab"
EXPECTED_V163_EXACT_DIGEST = (
    "8e3298dc7b64a81bee56ebceba8a6375b090ee1f186fe1a7515cf8cc0b9e0074"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v164-carrion-survivor-continuation-preterminal-tied-set-branch-target-expansion.json"
)
DEFAULT_MAX_BRANCH_TICK = 100
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = 2
DEFAULT_MAX_DOMINANT_OUTCOME_SUPPORT_ACTION_SHARE = 0.75
DEFAULT_TICK_BUCKET_SIZE = 20


class CarrionSurvivorContinuationV164PreterminalTiedSetBranchTargetExpansionError(
    ValueError
):
    pass


def run_carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion(
    *,
    v163_report_path: str | Path = DEFAULT_V163_REPORT_PATH,
    trajectory_glob: str = DEFAULT_TRAJECTORY_GLOB,
    trajectory_paths: Sequence[str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    strict_broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_tick: int = DEFAULT_MAX_BRANCH_TICK,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    verify_replay: bool = True,
    attempt_branch_replay: bool = True,
    expected_v163_exact_digest: str | None = EXPECTED_V163_EXACT_DIGEST,
    max_dominant_outcome_support_action_share: float = (
        DEFAULT_MAX_DOMINANT_OUTCOME_SUPPORT_ACTION_SHARE
    ),
) -> dict[str, object]:
    v163_report = load_json_report(v163_report_path)
    source_validation = validate_v164_sources(
        v163_report,
        expected_v163_exact_digest=expected_v163_exact_digest,
    )
    v162_report: dict[str, object] = {}
    artifact_load: dict[str, object] = {
        "passed": False,
        "reason": "source_validation_failed",
    }
    selected_paths = _selected_trajectory_paths(
        v163_report=v163_report,
        v162_report=v162_report,
        trajectory_paths=trajectory_paths,
    )
    selected_glob = _selected_trajectory_glob(
        v163_report=v163_report,
        v162_report=v162_report,
        trajectory_glob=trajectory_glob,
    )
    if source_validation.get("passed") is True:
        v162_load = _load_v162_source(v163_report)
        if v162_load.get("passed") is True:
            v162_report = dict(_mapping(v162_load.get("report")))
        else:
            source_validation = _with_source_failure(
                source_validation,
                str(v162_load.get("reason") or "v162_source_load_failed"),
            )
        selected_paths = _selected_trajectory_paths(
            v163_report=v163_report,
            v162_report=v162_report,
            trajectory_paths=trajectory_paths,
        )
        selected_glob = _selected_trajectory_glob(
            v163_report=v163_report,
            v162_report=v162_report,
            trajectory_glob=trajectory_glob,
        )
        artifact_load = _load_v160_artifact_source(v163_report)
        if artifact_load.get("passed") is not True:
            source_validation = _with_source_failure(
                source_validation,
                str(artifact_load.get("reason") or "v160_artifact_load_failed"),
            )
    evidence_report: dict[str, object] = _empty_evidence_report(
        trajectory_glob=selected_glob,
        trajectory_paths=selected_paths,
    )
    available_candidates = _empty_available_candidate_summary(
        strict_broad_seeds=strict_broad_seeds,
        ticks=ticks,
        max_branch_tick=max_branch_tick,
    )
    selection = _empty_selection_report(
        strict_broad_seeds=strict_broad_seeds,
        ticks=ticks,
        max_branch_tick=max_branch_tick,
        max_branch_points_per_seed=max_branch_points_per_seed,
        available_candidates=available_candidates,
    )
    materialization = _empty_materialization_report(
        reason="source_validation_failed_or_no_selection",
    )
    branch_results: list[dict[str, object]] = []
    outcome_support = summarize_preterminal_outcome_support(
        branch_results=[],
        ticks=ticks,
        max_branch_tick=max_branch_tick,
        max_dominant_outcome_support_action_share=(
            max_dominant_outcome_support_action_share
        ),
    )
    if source_validation.get("passed") is True:
        evidence = load_shadow_evidence(
            trajectory_glob=selected_glob,
            trajectory_paths=selected_paths,
        )
        evidence_report = _evidence_report(evidence)
        artifact = _mapping(artifact_load.get("artifact"))
        evidence_records = _list_of_mappings(evidence.get("records"))
        predictions = shadow_tie_collapse_autopsy_records(
            artifact=artifact,
            records=evidence_records,
        )
        available_candidates = summarize_available_preterminal_candidates(
            predictions=predictions,
            evidence_records=evidence_records,
            strict_broad_seeds=strict_broad_seeds,
            ticks=ticks,
            max_branch_tick=max_branch_tick,
        )
        selected = select_preterminal_tied_branch_points(
            predictions=predictions,
            evidence_records=evidence_records,
            strict_broad_seeds=strict_broad_seeds,
            ticks=ticks,
            max_branch_tick=max_branch_tick,
            max_branch_points_per_seed=max_branch_points_per_seed,
        )
        selection = _selection_report(
            selected,
            strict_broad_seeds=strict_broad_seeds,
            ticks=ticks,
            max_branch_tick=max_branch_tick,
            max_branch_points_per_seed=max_branch_points_per_seed,
            available_candidates=available_candidates,
        )
        if selected and attempt_branch_replay:
            materialized, materialization = materialize_selected_branch_points(
                selected,
                ticks=int(ticks),
            )
            materialization = _v164_materialization_report(materialization)
            if materialization.get("passed") is True:
                raw_results = evaluate_materialized_branch_points(
                    materialized,
                    verify_replay=bool(verify_replay),
                )
                branch_results = _annotated_branch_results(raw_results, ticks=ticks)
        elif selected:
            materialization = _empty_materialization_report(
                reason="branch_replay_disabled_plan_only",
                selected_branch_point_count=len(selected),
            )
        outcome_support = summarize_preterminal_outcome_support(
            branch_results=branch_results,
            ticks=ticks,
            max_branch_tick=max_branch_tick,
            max_dominant_outcome_support_action_share=(
                max_dominant_outcome_support_action_share
            ),
        )
    classification = _classification(
        source_validation=source_validation,
        materialization=materialization,
        branch_results=branch_results,
        outcome_support=outcome_support,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v163_report": str(v163_report_path),
            "expected_v163_exact_digest": expected_v163_exact_digest,
            "v162_report": str(_mapping(v163_report.get("inputs")).get("v162_report") or ""),
            "v160_artifact": str(_mapping(v163_report.get("inputs")).get("v160_artifact") or ""),
            "trajectory_glob": selected_glob,
            "trajectory_paths": [str(path) for path in selected_paths or []],
            "strict_broad_seeds": [int(seed) for seed in strict_broad_seeds],
            "ticks": int(ticks),
            "max_branch_tick": int(max_branch_tick),
            "min_remaining_horizon": max(0, int(ticks) - int(max_branch_tick)),
            "max_branch_points_per_seed": int(max_branch_points_per_seed),
            "verify_replay": bool(verify_replay),
            "attempt_branch_replay": bool(attempt_branch_replay),
            "max_dominant_outcome_support_action_share": _round(
                max_dominant_outcome_support_action_share
            ),
        },
        "source_validation": source_validation,
        "source_v163_digest_validation": _source_v163_digest_validation(
            source_validation
        ),
        "source_v162_digest_validation": _source_v162_digest_validation(
            _mapping(v163_report.get("source_validation"))
        ),
        "artifact_load": _artifact_load_report(artifact_load),
        "evidence": evidence_report,
        "available_preterminal_candidate_summary": available_candidates,
        "selection_plan": selection,
        "branch_materialization": materialization,
        "branch_results": branch_results,
        "outcome_support": outcome_support,
        "action_value_target_expansion_feasible": (
            classification
            == "preterminal_tied_set_target_support_ready_no_training"
        ),
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _route_recommendation(classification),
        "lifecycle_proof": _lifecycle_proof(),
        "diagnostics_only": True,
        "training_ran": False,
        "artifact_created": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "live_ab_ran": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
        "runtime_override_path_created": False,
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v164_sources(
    v163_report: Mapping[str, object],
    *,
    expected_v163_exact_digest: str | None = EXPECTED_V163_EXACT_DIGEST,
) -> dict[str, object]:
    failures: list[str] = []
    if (
        v163_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION
    ):
        failures.append("v163_schema_version_mismatch")
    if (
        v163_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY
    ):
        failures.append("v163_policy_mismatch")
    observed_classification = _mapping(v163_report.get("classification")).get(
        "primary"
    )
    if observed_classification != EXPECTED_V163_CLASSIFICATION:
        failures.append("v163_unexpected_classification")
    digest_validation = exact_digest_validation_report(v163_report)
    if digest_validation.get("passed") is not True:
        failures.append("v163_exact_digest_mismatch")
    observed_digest = str(v163_report.get("exact_digest") or "")
    if expected_v163_exact_digest and observed_digest != expected_v163_exact_digest:
        failures.append("v163_unexpected_exact_digest")
    lifecycle = _v163_lifecycle_validation(v163_report)
    if lifecycle.get("passed") is not True:
        failures.append("v163_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v164_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v163_classification": EXPECTED_V163_CLASSIFICATION,
        "observed_v163_classification": observed_classification,
        "expected_v163_exact_digest": expected_v163_exact_digest,
        "observed_v163_exact_digest": observed_digest,
        "v163_exact_digest_validation": digest_validation,
        "v163_lifecycle_validation": lifecycle,
    }


def summarize_available_preterminal_candidates(
    *,
    predictions: Sequence[Mapping[str, object]],
    evidence_records: Sequence[Mapping[str, object]],
    strict_broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_tick: int = DEFAULT_MAX_BRANCH_TICK,
    tick_bucket_size: int = DEFAULT_TICK_BUCKET_SIZE,
) -> dict[str, object]:
    strict = {int(seed) for seed in strict_broad_seeds}
    all_bucket: Counter[str] = Counter()
    eligible_bucket: Counter[str] = Counter()
    preferred_bucket: Counter[str] = Counter()
    eligible_by_seed: Counter[int] = Counter()
    eligible_by_row: Counter[str] = Counter()
    eligible_by_set: Counter[str] = Counter()
    eligible_by_tick: Counter[int] = Counter()
    final_or_late_count = 0
    preferred_count = 0
    all_tied_count = 0
    for prediction, evidence in zip(predictions, evidence_records, strict=True):
        record = _mapping(evidence.get("record"))
        seed = _int(prediction.get("source_seed"))
        if seed not in strict:
            continue
        top_set = _top_value_candidate_set(prediction)
        if len(top_set) <= 1:
            continue
        all_tied_count += 1
        tick = _int(record.get("tick"))
        bucket = _tick_bucket(tick, ticks=ticks, bucket_size=tick_bucket_size)
        all_bucket.update([bucket])
        if tick > int(max_branch_tick):
            final_or_late_count += 1
            continue
        eligible_bucket.update([bucket])
        eligible_by_seed.update([seed])
        eligible_by_tick.update([tick])
        set_key = _candidate_set_key(top_set)
        row_key = str(_int(prediction.get("nearest_neighbor_row_index")))
        eligible_by_row.update([row_key])
        eligible_by_set.update([set_key])
        if set_key in PREFERRED_TIED_SET_KEYS:
            preferred_count += 1
            preferred_bucket.update([bucket])
    return {
        "policy": "m3_carrion_survivor_continuation_v164_available_preterminal_candidate_summary_v1",
        "strict_broad_seeds": [int(seed) for seed in strict_broad_seeds],
        "ticks": int(ticks),
        "max_branch_tick": int(max_branch_tick),
        "min_remaining_horizon": max(0, int(ticks) - int(max_branch_tick)),
        "tick_bucket_size": int(tick_bucket_size),
        "preferred_tied_set_keys": list(PREFERRED_TIED_SET_KEYS),
        "preferred_nearest_rows": [int(row) for row in PREFERRED_NEAREST_ROWS],
        "all_strict_broad_tied_candidate_count": all_tied_count,
        "all_strict_broad_tied_candidate_count_by_tick_bucket": dict(
            sorted(all_bucket.items())
        ),
        "eligible_preterminal_tied_candidate_count": sum(eligible_bucket.values()),
        "eligible_preterminal_tied_candidate_count_by_tick_bucket": dict(
            sorted(eligible_bucket.items())
        ),
        "eligible_preterminal_preferred_candidate_count": preferred_count,
        "eligible_preterminal_preferred_candidate_count_by_tick_bucket": dict(
            sorted(preferred_bucket.items())
        ),
        "eligible_preterminal_candidate_count_by_seed": dict(
            sorted(eligible_by_seed.items())
        ),
        "eligible_preterminal_candidate_count_by_nearest_row": dict(
            sorted(eligible_by_row.items())
        ),
        "eligible_preterminal_candidate_count_by_tied_set": dict(
            sorted(eligible_by_set.items())
        ),
        "eligible_preterminal_candidate_count_by_tick": dict(
            sorted(eligible_by_tick.items())
        ),
        "excluded_after_max_branch_tick_count": final_or_late_count,
    }


def select_preterminal_tied_branch_points(
    *,
    predictions: Sequence[Mapping[str, object]],
    evidence_records: Sequence[Mapping[str, object]],
    strict_broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_tick: int = DEFAULT_MAX_BRANCH_TICK,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
) -> list[V163SelectedBranchPoint]:
    candidates_by_seed: defaultdict[int, list[tuple[Mapping[str, object], Mapping[str, object]]]]
    candidates_by_seed = defaultdict(list)
    strict = {int(seed) for seed in strict_broad_seeds}
    for prediction, evidence in zip(predictions, evidence_records, strict=True):
        record = _mapping(evidence.get("record"))
        seed = _int(prediction.get("source_seed"))
        tick = _int(record.get("tick"))
        top_set = _top_value_candidate_set(prediction)
        if seed not in strict or len(top_set) <= 1 or tick > int(max_branch_tick):
            continue
        candidates_by_seed[seed].append((prediction, evidence))
    selected: list[V163SelectedBranchPoint] = []
    for seed in strict_broad_seeds:
        seed_items = candidates_by_seed.get(int(seed), [])
        seed_preferred = [
            item
            for item in seed_items
            if _candidate_set_key(_top_value_candidate_set(item[0]))
            in PREFERRED_TIED_SET_KEYS
        ]
        pool = seed_preferred or seed_items
        chosen = _choose_seed_preterminal_candidates(
            pool,
            max_count=max(0, int(max_branch_points_per_seed)),
        )
        for branch_index, (prediction, evidence) in enumerate(chosen):
            selected.append(
                _selected_branch_point_from_prediction(
                    prediction=prediction,
                    evidence=evidence,
                    seed=int(seed),
                    ticks=int(ticks),
                    max_branch_tick=int(max_branch_tick),
                    branch_index=branch_index,
                )
            )
    return selected


def summarize_preterminal_outcome_support(
    *,
    branch_results: Sequence[Mapping[str, object]],
    ticks: int = DEFAULT_TICKS,
    max_branch_tick: int = DEFAULT_MAX_BRANCH_TICK,
    max_dominant_outcome_support_action_share: float = (
        DEFAULT_MAX_DOMINANT_OUTCOME_SUPPORT_ACTION_SHARE
    ),
) -> dict[str, object]:
    terminal = _support_summary_for_view(
        branch_results,
        view_name="terminal_population_outcome",
        key_fn=_terminal_population_key,
        max_dominant_outcome_support_action_share=(
            max_dominant_outcome_support_action_share
        ),
    )
    target = _support_summary_for_view(
        branch_results,
        view_name="target_local_continuation_outcome",
        key_fn=_target_local_continuation_key,
        max_dominant_outcome_support_action_share=(
            max_dominant_outcome_support_action_share
        ),
    )
    replay_count = 0
    replay_verified_count = 0
    candidate_run_count = 0
    remaining_horizons: list[int] = []
    for result in branch_results:
        remaining_horizons.append(max(0, int(ticks) - _int(result.get("branch_tick"))))
        for run in _list_of_mappings(result.get("candidate_runs")):
            candidate_run_count += 1
            replay = _mapping(run.get("replay_verification"))
            if replay:
                replay_count += 1
                replay_verified_count += int(replay.get("verified") is True)
    min_remaining = min(remaining_horizons) if remaining_horizons else None
    horizon_sensitive = (
        bool(remaining_horizons)
        and min_remaining is not None
        and min_remaining >= max(0, int(ticks) - int(max_branch_tick))
        and all(_int(result.get("branch_tick")) <= int(max_branch_tick) for result in branch_results)
    )
    genuinely_noncollapsed = (
        horizon_sensitive
        and terminal.get("outcome_support_noncollapsed") is True
        and target.get("outcome_support_noncollapsed") is True
    )
    has_meaningful_signal = (
        _int(terminal.get("informative_branch_count")) > 0
        or _int(target.get("informative_branch_count")) > 0
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v164_preterminal_outcome_support_summary_v1",
        "branch_result_count": len(branch_results),
        "candidate_run_count": candidate_run_count,
        "replay_verification_count": replay_count,
        "replay_verified_count": replay_verified_count,
        "all_replays_verified": replay_count > 0 and replay_count == replay_verified_count,
        "ticks": int(ticks),
        "max_branch_tick": int(max_branch_tick),
        "remaining_horizon_by_branch": {
            str(result.get("branch_id")): max(0, int(ticks) - _int(result.get("branch_tick")))
            for result in branch_results
        },
        "min_remaining_horizon": min_remaining,
        "meaningful_horizon_required": max(0, int(ticks) - int(max_branch_tick)),
        "horizon_sensitive": horizon_sensitive,
        "has_meaningful_outcome_signal": has_meaningful_signal,
        "outcome_support_by_scoring_view": {
            "terminal_population_outcome": terminal,
            "target_local_continuation_outcome": target,
        },
        "genuinely_noncollapsed_under_meaningful_horizon": genuinely_noncollapsed,
        "preterminal_target_support_noncollapsed": genuinely_noncollapsed,
    }


def _choose_seed_preterminal_candidates(
    items: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    max_count: int,
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    if max_count <= 0:
        return []
    ordered = sorted(items, key=_preterminal_selection_sort_key)
    chosen: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    used_ticks: set[int] = set()
    for row in PREFERRED_NEAREST_ROWS:
        if len(chosen) >= max_count:
            break
        row_items = [
            item
            for item in ordered
            if _int(item[0].get("nearest_neighbor_row_index")) == int(row)
            and item not in chosen
        ]
        pick = _first_avoiding_ticks(row_items, used_ticks) or (
            row_items[0] if row_items else None
        )
        if pick is not None:
            chosen.append(pick)
            used_ticks.add(_int(_mapping(pick[1].get("record")).get("tick")))
    for item in ordered:
        if len(chosen) >= max_count:
            break
        if item in chosen:
            continue
        pick = item
        if _int(_mapping(item[1].get("record")).get("tick")) in used_ticks:
            alternate = _first_avoiding_ticks(
                [candidate for candidate in ordered if candidate not in chosen],
                used_ticks,
            )
            if alternate is not None:
                pick = alternate
        if pick in chosen:
            continue
        chosen.append(pick)
        used_ticks.add(_int(_mapping(pick[1].get("record")).get("tick")))
    return chosen


def _first_avoiding_ticks(
    items: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    used_ticks: set[int],
) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    for item in items:
        tick = _int(_mapping(item[1].get("record")).get("tick"))
        if tick not in used_ticks:
            return item
    return None


def _preterminal_selection_sort_key(
    item: tuple[Mapping[str, object], Mapping[str, object]],
) -> tuple[int, int, int, int, int]:
    prediction, evidence = item
    record = _mapping(evidence.get("record"))
    set_key = _candidate_set_key(_top_value_candidate_set(prediction))
    try:
        set_rank = PREFERRED_TIED_SET_KEYS.index(set_key)
    except ValueError:
        set_rank = 2 if len(set_key.split("|")) >= 3 else 3
    row = _int(prediction.get("nearest_neighbor_row_index"))
    try:
        row_rank = PREFERRED_NEAREST_ROWS.index(row)
    except ValueError:
        row_rank = len(PREFERRED_NEAREST_ROWS)
    return (
        set_rank,
        row_rank,
        -_int(record.get("tick")),
        -_int(evidence.get("line_number")),
        _int(record.get("agent_id")),
    )


def _selected_branch_point_from_prediction(
    *,
    prediction: Mapping[str, object],
    evidence: Mapping[str, object],
    seed: int,
    ticks: int,
    max_branch_tick: int,
    branch_index: int,
) -> V163SelectedBranchPoint:
    record = _mapping(evidence.get("record"))
    tick = _int(record.get("tick"))
    agent_id = _int(record.get("agent_id"))
    top_set = _top_value_candidate_set(prediction)
    action_mask = _bool_action_mask(
        record.get("public_action_mask") or record.get("action_mask")
    )
    branch_id = (
        f"v164-broad-seed-{int(seed)}-branch-{int(branch_index)}-"
        f"tick-{tick}-agent-{agent_id}"
    )
    set_key = _candidate_set_key(top_set)
    row = _int(prediction.get("nearest_neighbor_row_index"))
    return V163SelectedBranchPoint(
        branch_id=branch_id,
        seed=int(seed),
        ticks=int(ticks),
        branch_tick=tick,
        record_index=_int(prediction.get("record_index")),
        branch_index=int(branch_index),
        agent_id=agent_id,
        source_path=str(evidence.get("source_path", "")),
        line_number=_int(evidence.get("line_number")),
        runtime_requested_action=_action_or_empty(record.get("requested_action")),
        runtime_resolved_action=_action_or_empty(record.get("resolved_action")),
        predicted_action=_action_or_empty(prediction.get("predicted_action")),
        nearest_neighbor_row_index=row,
        top_value_candidate_set=top_set,
        action_mask=action_mask,
        observation_input=dict(_mapping(record.get("observation_input"))),
        observation_schema=_optional_string(record.get("observation_schema")),
        observation_digest=_optional_string(record.get("observation_digest")),
        source_record_digest=stable_payload_digest(
            _record_materialization_payload(record)
        ),
        selection_rationale={
            "preterminal_selector": True,
            "branch_tick_max": int(max_branch_tick),
            "branch_tick": tick,
            "remaining_horizon": max(0, int(ticks) - tick),
            "excluded_final_tick": tick < int(ticks) - 1,
            "preferred_tied_set": set_key in PREFERRED_TIED_SET_KEYS,
            "preferred_nearest_row": row in PREFERRED_NEAREST_ROWS,
            "top_value_candidate_set_key": set_key,
            "top_value_candidate_set_size": len(top_set),
            "nearest_neighbor_row_index": row,
            "runtime_requested_action_used_as_scorer_input": False,
            "seed_tick_agent_path_digest_for_materialization_only": True,
        },
    )


def _annotated_branch_results(
    branch_results: Sequence[Mapping[str, object]],
    *,
    ticks: int,
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for result in branch_results:
        payload = dict(result)
        remaining_horizon = max(0, int(ticks) - _int(result.get("branch_tick")))
        candidate_runs = [
            _annotated_candidate_run(run)
            for run in _list_of_mappings(result.get("candidate_runs"))
        ]
        payload["remaining_horizon"] = remaining_horizon
        payload["candidate_runs"] = candidate_runs
        payload["first_action_outcome_summaries"] = [
            _first_action_outcome_summary(run) for run in candidate_runs
        ]
        payload["scoring_views"] = {
            "terminal_population_outcome": _branch_view_result(
                candidate_runs,
                key_fn=_terminal_population_key,
            ),
            "target_local_continuation_outcome": _branch_view_result(
                candidate_runs,
                key_fn=_target_local_continuation_key,
            ),
        }
        annotated.append(payload)
    return annotated


def _annotated_candidate_run(run: Mapping[str, object]) -> dict[str, object]:
    payload = dict(run)
    reference_delta = _mapping(payload.get("deltas_vs_recorded_reference_runtime"))
    payload["terminal_population_delta_vs_reference"] = {
        "alive_agents": reference_delta.get("alive_agents"),
        "births": reference_delta.get("births"),
        "deaths": reference_delta.get("deaths"),
        "unsupported_requested_action_count": reference_delta.get(
            "unsupported_requested_action_count"
        ),
    }
    payload["target_local_delta_vs_reference"] = {
        "target_alive": reference_delta.get("target_alive"),
        "target_energy_ratio": reference_delta.get("target_energy_ratio"),
        "target_hydration_ratio": reference_delta.get("target_hydration_ratio"),
        "target_health_ratio": reference_delta.get("target_health_ratio"),
    }
    payload["first_action_outcome_summary"] = _first_action_outcome_summary(payload)
    payload["branch_run_digest_payload_digest"] = stable_payload_digest(
        _branch_run_digest_payload(payload)
    )
    return payload


def _branch_view_result(
    candidate_runs: Sequence[Mapping[str, object]],
    *,
    key_fn: object,
) -> dict[str, object]:
    if not callable(key_fn):
        raise CarrionSurvivorContinuationV164PreterminalTiedSetBranchTargetExpansionError(
            "v164 scoring view key function is not callable"
        )
    keyed = [(run, key_fn(run)) for run in candidate_runs]
    if not keyed:
        return {"best_actions": [], "unique_outcome_count": 0, "all_candidates_tied": False}
    best_key = max(key for _, key in keyed)
    best_actions = sorted(
        [
            str(run.get("forced_action"))
            for run, key in keyed
            if key == best_key and str(run.get("forced_action")) in ACTION_NAMES
        ],
        key=_action_order,
    )
    unique_keys = {key for _, key in keyed}
    return {
        "best_actions": best_actions,
        "best_action_count": len(best_actions),
        "unique_outcome_count": len(unique_keys),
        "all_candidates_tied": len(unique_keys) == 1 and len(candidate_runs) > 1,
        "best_key_digest": stable_payload_digest(best_key),
    }


def _support_summary_for_view(
    branch_results: Sequence[Mapping[str, object]],
    *,
    view_name: str,
    key_fn: object,
    max_dominant_outcome_support_action_share: float,
) -> dict[str, object]:
    support_counts: Counter[str] = Counter()
    informative_branch_count = 0
    all_candidates_tied_count = 0
    branches_with_best_action = 0
    for result in branch_results:
        candidate_runs = _list_of_mappings(result.get("candidate_runs"))
        branch_view = _branch_view_result(candidate_runs, key_fn=key_fn)
        if _int(branch_view.get("unique_outcome_count")) > 1:
            informative_branch_count += 1
        if branch_view.get("all_candidates_tied") is True:
            all_candidates_tied_count += 1
        best_actions = [
            str(action)
            for action in branch_view.get("best_actions", [])
            if str(action) in ACTION_NAMES
        ]
        if best_actions:
            branches_with_best_action += 1
            support_counts.update(best_actions)
    dominant = _dominant_count_share(support_counts)
    noncollapsed = (
        informative_branch_count > 0
        and sum(support_counts.values()) > 0
        and len(support_counts) >= 2
        and float(dominant.get("share") or 0.0)
        <= float(max_dominant_outcome_support_action_share)
    )
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v164_"
            f"{view_name}_support_summary_v1"
        ),
        "scoring_view": view_name,
        "branch_result_count": len(branch_results),
        "branches_with_best_action": branches_with_best_action,
        "informative_branch_count": informative_branch_count,
        "all_candidates_tied_branch_count": all_candidates_tied_count,
        "per_action_outcome_support_counts": dict(sorted(support_counts.items())),
        "dominant_outcome_support_action": dominant.get("key"),
        "dominant_outcome_support_action_count": dominant.get("count"),
        "dominant_outcome_support_action_share": dominant.get("share"),
        "max_dominant_outcome_support_action_share": _round(
            max_dominant_outcome_support_action_share
        ),
        "outcome_support_noncollapsed": noncollapsed,
    }


def _terminal_population_key(run: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _int(run.get("alive_agents")),
        _int(run.get("births")),
        -_int(run.get("deaths")),
        -_int(run.get("unsupported_requested_action_count")),
    )


def _target_local_continuation_key(run: Mapping[str, object]) -> tuple[object, ...]:
    target = _mapping(run.get("target_terminal"))
    first = _mapping(run.get("first_action_outcome"))
    outcome = _mapping(first.get("outcome"))
    return (
        _bool_int(target.get("alive")),
        _finite_or_negative(target.get("energy_ratio")),
        _finite_or_negative(target.get("hydration_ratio")),
        _finite_or_negative(target.get("health_ratio")),
        _finite_or_negative(outcome.get("resource_gain")),
        -_int(run.get("unsupported_requested_action_count")),
    )


def _first_action_outcome_summary(run: Mapping[str, object]) -> dict[str, object]:
    first = _mapping(run.get("first_action_outcome"))
    outcome = _mapping(first.get("outcome"))
    movement = _mapping(outcome.get("movement"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    passive = _mapping(outcome.get("passive"))
    return {
        "forced_action": run.get("forced_action"),
        "requested_action": first.get("requested_action"),
        "resolved_action": first.get("resolved_action"),
        "matches_forced_action": first.get("matches_forced_action"),
        "action_valid": first.get("action_valid"),
        "resolution_action_valid": first.get("resolution_action_valid"),
        "moved": movement.get("moved"),
        "ate": feeding.get("ate"),
        "drank": drinking.get("drank"),
        "resource_gain": outcome.get("resource_gain"),
        "died": outcome.get("died"),
        "damage_taken": passive.get("damage_taken"),
        "death_cause": passive.get("death_cause"),
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    materialization: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    outcome_support: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return "source_invalid_closed_no_live_ab"
    if materialization.get("passed") is not True:
        return "preterminal_branch_materialization_blocked_no_training"
    if outcome_support.get("all_replays_verified") is not True:
        return "preterminal_branch_replay_not_deterministic_closed_no_training"
    if not branch_results or outcome_support.get("horizon_sensitive") is not True:
        return "preterminal_target_support_insufficient_no_live_ab"
    if (
        outcome_support.get("genuinely_noncollapsed_under_meaningful_horizon")
        is True
    ):
        return "preterminal_tied_set_target_support_ready_no_training"
    views = _mapping(outcome_support.get("outcome_support_by_scoring_view"))
    if outcome_support.get("has_meaningful_outcome_signal") is True and any(
        _view_is_collapsed(_mapping(view)) for view in views.values()
    ):
        return "preterminal_target_support_collapsed_no_live_ab"
    return "preterminal_target_support_insufficient_no_live_ab"


def _view_is_collapsed(view: Mapping[str, object]) -> bool:
    counts = _mapping(view.get("per_action_outcome_support_counts"))
    dominant_share = view.get("dominant_outcome_support_action_share")
    return bool(counts) and (
        len(counts) <= 1
        or (
            isinstance(dominant_share, (int, float))
            and not isinstance(dominant_share, bool)
            and float(dominant_share)
            > float(view.get("max_dominant_outcome_support_action_share") or 1.0)
        )
    )


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification == "preterminal_tied_set_target_support_ready_no_training"
    collapsed = classification == "preterminal_target_support_collapsed_no_live_ab"
    return {
        "policy": "m3_carrion_survivor_continuation_v164_route_recommendation_v1",
        "recommended_next_route": (
            "v165_target_dataset_expansion_from_preterminal_tied_set_branch_evidence"
            if ready
            else (
                "close_scorer_tie_set_branch_broader_archive_world_model_route"
                if collapsed
                else "branch_replay_contract_work_before_target_expansion"
            )
        ),
        "v165_target_dataset_expansion_recommended": ready,
        "close_scorer_tie_set_branch_recommended": collapsed,
        "broader_archive_world_model_route_recommended": collapsed or not ready,
        "branch_replay_contract_work_recommended": (
            classification
            in {
                "preterminal_branch_materialization_blocked_no_training",
                "preterminal_branch_replay_not_deterministic_closed_no_training",
            }
        ),
        "threshold_tuning_recommended": False,
        "live_ab_allowed": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
    }


def _selection_report(
    selected: Sequence[V163SelectedBranchPoint],
    *,
    strict_broad_seeds: Sequence[int],
    ticks: int,
    max_branch_tick: int,
    max_branch_points_per_seed: int,
    available_candidates: Mapping[str, object],
) -> dict[str, object]:
    by_seed = Counter(point.seed for point in selected)
    by_set = Counter(_candidate_set_key(point.top_value_candidate_set) for point in selected)
    by_row = Counter(str(point.nearest_neighbor_row_index) for point in selected)
    by_tick = Counter(str(point.branch_tick) for point in selected)
    by_remaining = Counter(str(max(0, int(ticks) - point.branch_tick)) for point in selected)
    duplicate_tick_seed_count = 0
    for seed in strict_broad_seeds:
        ticks_for_seed = [
            point.branch_tick for point in selected if point.seed == int(seed)
        ]
        duplicate_tick_seed_count += int(len(ticks_for_seed) != len(set(ticks_for_seed)))
    return {
        "policy": "m3_carrion_survivor_continuation_v164_preterminal_selection_plan_v1",
        "strict_broad_seeds": [int(seed) for seed in strict_broad_seeds],
        "ticks": int(ticks),
        "max_branch_tick": int(max_branch_tick),
        "final_tick_excluded": True,
        "min_remaining_horizon": max(0, int(ticks) - int(max_branch_tick)),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "preferred_tied_set_keys": list(PREFERRED_TIED_SET_KEYS),
        "preferred_nearest_rows": [int(row) for row in PREFERRED_NEAREST_ROWS],
        "selected_branch_point_count": len(selected),
        "selected_branch_points_by_seed": dict(sorted(by_seed.items())),
        "selected_tied_set_counts": dict(sorted(by_set.items())),
        "selected_nearest_row_counts": dict(sorted(by_row.items())),
        "selected_tick_distribution": dict(sorted(by_tick.items())),
        "selected_remaining_horizon_distribution": dict(sorted(by_remaining.items())),
        "duplicate_tick_seed_count": duplicate_tick_seed_count,
        "available_candidate_count_by_tick_bucket_before_selection": dict(
            _mapping(
                available_candidates.get(
                    "eligible_preterminal_preferred_candidate_count_by_tick_bucket"
                )
            )
        ),
        "selected_branch_points": [_selected_payload(point, ticks=ticks) for point in selected],
    }


def _empty_selection_report(
    *,
    strict_broad_seeds: Sequence[int],
    ticks: int,
    max_branch_tick: int,
    max_branch_points_per_seed: int,
    available_candidates: Mapping[str, object],
) -> dict[str, object]:
    return _selection_report(
        [],
        strict_broad_seeds=strict_broad_seeds,
        ticks=ticks,
        max_branch_tick=max_branch_tick,
        max_branch_points_per_seed=max_branch_points_per_seed,
        available_candidates=available_candidates,
    )


def _selected_payload(point: V163SelectedBranchPoint, *, ticks: int) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": "broad",
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "remaining_horizon": max(0, int(ticks) - point.branch_tick),
        "record_index": point.record_index,
        "branch_index": point.branch_index,
        "agent_id": point.agent_id,
        "source_path": point.source_path,
        "line_number": point.line_number,
        "runtime_requested_action": point.runtime_requested_action,
        "runtime_resolved_action": point.runtime_resolved_action,
        "predicted_action": point.predicted_action,
        "nearest_neighbor_row_index": point.nearest_neighbor_row_index,
        "top_value_candidate_set": list(point.top_value_candidate_set),
        "candidate_action_count": len(point.top_value_candidate_set),
        "observation_schema": point.observation_schema,
        "observation_digest": point.observation_digest,
        "source_record_digest": point.source_record_digest,
        "selection_rationale": dict(point.selection_rationale),
        "seed_tick_agent_path_digest_for_materialization_only": True,
        "private_world_state_serialized": False,
    }


def _empty_available_candidate_summary(
    *,
    strict_broad_seeds: Sequence[int],
    ticks: int,
    max_branch_tick: int,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v164_available_preterminal_candidate_summary_v1",
        "strict_broad_seeds": [int(seed) for seed in strict_broad_seeds],
        "ticks": int(ticks),
        "max_branch_tick": int(max_branch_tick),
        "min_remaining_horizon": max(0, int(ticks) - int(max_branch_tick)),
        "all_strict_broad_tied_candidate_count": 0,
        "all_strict_broad_tied_candidate_count_by_tick_bucket": {},
        "eligible_preterminal_tied_candidate_count": 0,
        "eligible_preterminal_tied_candidate_count_by_tick_bucket": {},
        "eligible_preterminal_preferred_candidate_count": 0,
        "eligible_preterminal_preferred_candidate_count_by_tick_bucket": {},
    }


def _empty_evidence_report(
    *,
    trajectory_glob: str,
    trajectory_paths: Sequence[Path] | None,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v164_shadow_evidence_not_loaded_v1",
        "trajectory_glob": trajectory_glob,
        "trajectory_paths": [str(path) for path in trajectory_paths or []],
        "record_count": 0,
        "decision_record_count": 0,
        "not_loaded_reason": "source_validation_failed",
    }


def _empty_materialization_report(
    *,
    reason: str,
    selected_branch_point_count: int = 0,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v164_preterminal_exact_branch_materialization_v1",
        "selected_branch_point_count": int(selected_branch_point_count),
        "materialized_branch_point_count": 0,
        "materialization_failure_count": int(selected_branch_point_count > 0),
        "materialization_failures": (
            [{"reason": reason}] if selected_branch_point_count > 0 else []
        ),
        "reference_runs": [],
        "exact_materialization_proven": False,
        "passed": False,
    }


def _v164_materialization_report(report: Mapping[str, object]) -> dict[str, object]:
    payload = dict(report)
    payload["policy"] = (
        "m3_carrion_survivor_continuation_v164_preterminal_exact_branch_materialization_v1"
    )
    payload["preterminal_exact_materialization_proven"] = payload.get(
        "exact_materialization_proven"
    )
    return payload


def _load_v162_source(v163_report: Mapping[str, object]) -> dict[str, object]:
    path = str(_mapping(v163_report.get("inputs")).get("v162_report") or "")
    if not path:
        return {"passed": False, "reason": "missing_v162_report_path"}
    try:
        return {"passed": True, "path": path, "report": load_json_report(path)}
    except (OSError, ValueError) as exc:
        return {
            "passed": False,
            "reason": "v162_source_load_failed",
            "path": path,
            "error": str(exc),
        }


def _load_v160_artifact_source(v163_report: Mapping[str, object]) -> dict[str, object]:
    path = str(_mapping(v163_report.get("inputs")).get("v160_artifact") or "")
    if not path:
        return {"passed": False, "reason": "missing_v160_artifact_path"}
    try:
        artifact = load_json_report(path)
    except (OSError, ValueError) as exc:
        return {
            "passed": False,
            "reason": "v160_artifact_load_failed",
            "path": path,
            "error": str(exc),
        }
    return {
        "passed": True,
        "path": path,
        "artifact_digest": stable_payload_digest(artifact),
        "artifact": artifact,
    }


def _artifact_load_report(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "artifact"}


def _with_source_failure(
    source_validation: Mapping[str, object],
    failure: str,
) -> dict[str, object]:
    failures = sorted(
        set([str(item) for item in source_validation.get("failures", [])] + [failure])
    )
    payload = dict(source_validation)
    payload["passed"] = False
    payload["failures"] = failures
    return payload


def _selected_trajectory_paths(
    *,
    v163_report: Mapping[str, object],
    v162_report: Mapping[str, object],
    trajectory_paths: Sequence[str | Path] | None,
) -> list[Path] | None:
    if trajectory_paths:
        return [Path(path) for path in trajectory_paths]
    v163_paths = _mapping(v163_report.get("inputs")).get("trajectory_paths")
    if isinstance(v163_paths, Sequence) and not isinstance(v163_paths, (str, bytes)):
        paths = [Path(str(path)) for path in v163_paths]
        if paths:
            return paths
    v162_paths = _mapping(v162_report.get("inputs")).get("trajectory_paths")
    if isinstance(v162_paths, Sequence) and not isinstance(v162_paths, (str, bytes)):
        paths = [Path(str(path)) for path in v162_paths]
        if paths:
            return paths
    return None


def _selected_trajectory_glob(
    *,
    v163_report: Mapping[str, object],
    v162_report: Mapping[str, object],
    trajectory_glob: str,
) -> str:
    return str(
        _mapping(v163_report.get("inputs")).get("trajectory_glob")
        or _mapping(v162_report.get("inputs")).get("trajectory_glob")
        or trajectory_glob
    )


def _evidence_report(evidence: Mapping[str, object]) -> dict[str, object]:
    payload = dict(evidence)
    payload.pop("records", None)
    return payload


def _source_v163_digest_validation(
    source_validation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "expected_v163_classification": source_validation.get(
            "expected_v163_classification"
        ),
        "observed_v163_classification": source_validation.get(
            "observed_v163_classification"
        ),
        "expected_v163_exact_digest": source_validation.get(
            "expected_v163_exact_digest"
        ),
        "observed_v163_exact_digest": source_validation.get(
            "observed_v163_exact_digest"
        ),
        "v163_exact_digest_validation": source_validation.get(
            "v163_exact_digest_validation"
        ),
    }


def _v163_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_allowed",
        "promotion_authorized",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    lifecycle = _mapping(report.get("lifecycle_proof"))
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_allowed",
        "promotion_authorized",
    ):
        if field in lifecycle and lifecycle.get(field) is not False:
            failures.append(
                {"field": f"lifecycle_proof.{field}", "observed": lifecycle.get(field)}
            )
    v162 = _v162_lifecycle_validation(report)
    if v162.get("passed") is not True:
        failures.extend(_list_of_mappings(v162.get("failures")))
    return {
        "policy": "m3_carrion_survivor_continuation_v164_v163_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "preterminal_branch_target_expansion_diagnostic_only": True,
        "validates_v163_exact_digest_before_analysis": True,
        "requires_v163_insufficient_classification": True,
        "requires_preterminal_branch_tick": True,
        "uses_v160_artifact_for_tied_set_recomputation": True,
        "uses_seed_tick_agent_path_digest_for_branch_materialization_only": True,
        "uses_seed_fixture_branch_tick_agent_path_digest_provenance_as_trainable_input": False,
        "uses_runtime_requested_actions_for_comparison_only": True,
        "uses_runtime_requested_actions_as_scorer_input": False,
        "training_authorized": False,
        "training_ran": False,
        "serialized_scorer_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_policy_integration_allowed": False,
        "live_ab_allowed": False,
        "live_override_allowed": False,
        "runtime_override_path_created": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }


def _lifecycle_proof() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v164_lifecycle_proof_v1",
        "diagnostics_only": True,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "threshold_tuning_recommended": False,
        "runtime_override_path_created": False,
    }


def _top_value_candidate_set(prediction: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(action)
        for action in prediction.get("top_value_candidate_set", [])
        if str(action) in ACTION_NAMES
    )


def _tick_bucket(tick: int, *, ticks: int, bucket_size: int) -> str:
    size = max(1, int(bucket_size))
    start = (max(0, int(tick)) // size) * size
    end = min(max(0, int(ticks) - 1), start + size - 1)
    return f"{start:03d}-{end:03d}"


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _bool_int(value: object) -> int:
    return 1 if value is True else 0


def _finite_or_negative(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return -1.0


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    json_payload = json.loads(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
    return stable_payload_digest(json_payload)
