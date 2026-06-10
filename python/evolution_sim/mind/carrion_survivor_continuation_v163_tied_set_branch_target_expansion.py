from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind import evaluation_harness as evaluate_cli
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.broad_regression_branch_intervention import (
    _configure_manual_summary_run,
    _normal_mind_v3_delegate,
    _target_terminal,
)
from evolution_sim.mind.candidate_campaign import (
    _dominant_count_share,
    _int,
    _list_of_mappings,
    _mapping,
    _round,
    write_json,
)
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
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
    DEFAULT_OUTPUT_PATH as DEFAULT_V162_REPORT_PATH,
    M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_SCHEMA_VERSION,
    shadow_tie_collapse_autopsy_records,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_tied_set_branch_target_expansion_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_tied_set_branch_target_expansion_v1"
)
EXPECTED_V162_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_"
    "tie_collapse_archive_feature_support_blocked_no_live_ab"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v163-carrion-survivor-continuation-tied-set-branch-target-expansion.json"
)
STRICT_BROAD_SEEDS = (5, 13, 19, 29, 37, 41)
DEFAULT_TICKS = 120
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = 1
DEFAULT_MAX_DOMINANT_OUTCOME_SUPPORT_ACTION_SHARE = 0.75
PREFERRED_TIED_SET_KEYS = (
    "stay|eat|move_north|move_south",
    "stay|eat|move_north",
)
PREFERRED_NEAREST_ROWS = (7, 4)


class CarrionSurvivorContinuationV163TiedSetBranchTargetExpansionError(
    ValueError
):
    pass


@dataclass(frozen=True, slots=True)
class V163SelectedBranchPoint:
    branch_id: str
    seed: int
    ticks: int
    branch_tick: int
    record_index: int
    branch_index: int
    agent_id: int
    source_path: str
    line_number: int
    runtime_requested_action: str
    runtime_resolved_action: str
    predicted_action: str
    nearest_neighbor_row_index: int
    top_value_candidate_set: tuple[str, ...]
    action_mask: dict[str, bool]
    observation_input: dict[str, object]
    observation_schema: str | None
    observation_digest: str | None
    source_record_digest: str
    selection_rationale: dict[str, object]


@dataclass(frozen=True, slots=True)
class V163MaterializedBranchPoint:
    selected: V163SelectedBranchPoint
    branch_state_digest: str
    world: SimulationWorld
    materialized_record_digest: str


class _ForcedTiedCandidateThenMindV3Policy:
    policy_id = "mind_v3_v163_tied_set_branch_target_expansion_force"
    policy_version = (
        "mind_v3_v163_tied_set_branch_target_expansion_force_v1"
    )

    def __init__(
        self,
        *,
        target_agent_id: int,
        forced_action: str,
        delegate: object,
    ) -> None:
        self.target_agent_id = int(target_agent_id)
        self.forced_action = str(forced_action)
        self.delegate = delegate
        self.used = False

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        metadata = observation.get("metadata")
        agent_id = metadata.get("agent_id") if isinstance(metadata, Mapping) else None
        if (
            not self.used
            and agent_id is not None
            and int(agent_id) == self.target_agent_id
            and self.forced_action in ACTION_NAMES
            and bool(action_mask.get(self.forced_action, False))
        ):
            self.used = True
            return ActionDecision(
                requested_action=self.forced_action,
                source=f"tied_set_branch_target_expansion_force:{self.forced_action}",
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                diagnostics={
                    "diagnostics_only": True,
                    "heuristic_free": True,
                    "target_agent_id": self.target_agent_id,
                    "forced_action": self.forced_action,
                },
            )
        decide = getattr(self.delegate, "decide", None)
        if not callable(decide):
            raise CarrionSurvivorContinuationV163TiedSetBranchTargetExpansionError(
                "v163 branch delegate policy does not implement decide"
            )
        return decide(observation, action_mask)

    def observe_transition(self, record: dict[str, object]) -> dict[str, object] | None:
        observe = getattr(self.delegate, "observe_transition", None)
        if not callable(observe):
            return None
        feedback = dict(record)
        if (
            _int(feedback.get("agent_id"), default=-1) == self.target_agent_id
            and str(feedback.get("action_source", "")).startswith(
                "tied_set_branch_target_expansion_force:"
            )
        ):
            feedback["policy_id"] = getattr(self.delegate, "policy_id", None)
            feedback["policy_version"] = getattr(self.delegate, "policy_version", None)
        return observe(feedback)


def run_carrion_survivor_continuation_v163_tied_set_branch_target_expansion(
    *,
    v162_report_path: str | Path = DEFAULT_V162_REPORT_PATH,
    v160_artifact_path: str | Path | None = None,
    trajectory_glob: str = DEFAULT_TRAJECTORY_GLOB,
    trajectory_paths: Sequence[str | Path] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    strict_broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    verify_replay: bool = True,
    attempt_branch_replay: bool = True,
    max_dominant_outcome_support_action_share: float = (
        DEFAULT_MAX_DOMINANT_OUTCOME_SUPPORT_ACTION_SHARE
    ),
) -> dict[str, object]:
    v162_report = load_json_report(v162_report_path)
    source_validation = validate_v163_sources(v162_report)
    selected_paths = _selected_trajectory_paths(
        v162_report=v162_report,
        trajectory_paths=trajectory_paths,
    )
    selected_glob = str(
        _mapping(v162_report.get("inputs")).get("trajectory_glob") or trajectory_glob
    )
    artifact_path = Path(
        v160_artifact_path
        or str(_mapping(v162_report.get("inputs")).get("v160_artifact") or "")
    )
    artifact_load = _load_artifact_for_v163(artifact_path)
    if artifact_load.get("passed") is not True:
        source_validation = _with_source_failure(
            source_validation,
            "v160_artifact_load_failed",
        )
    evidence_report: dict[str, object] = _empty_evidence_report(
        trajectory_glob=selected_glob,
        trajectory_paths=selected_paths,
    )
    selection = _empty_selection_report(
        strict_broad_seeds=strict_broad_seeds,
        max_branch_points_per_seed=max_branch_points_per_seed,
    )
    materialization = _empty_materialization_report(
        reason="source_validation_failed_or_no_selection"
    )
    branch_results: list[dict[str, object]] = []
    outcome_support = _outcome_support_summary(
        branch_results=[],
        max_dominant_outcome_support_action_share=(
            max_dominant_outcome_support_action_share
        ),
    )
    if source_validation.get("passed") is True:
        artifact = _mapping(artifact_load.get("artifact"))
        evidence = load_shadow_evidence(
            trajectory_glob=selected_glob,
            trajectory_paths=selected_paths,
        )
        evidence_report = _evidence_report(evidence)
        predictions = shadow_tie_collapse_autopsy_records(
            artifact=artifact,
            records=_list_of_mappings(evidence.get("records")),
        )
        selected = select_representative_tied_branch_points(
            predictions=predictions,
            evidence_records=_list_of_mappings(evidence.get("records")),
            strict_broad_seeds=strict_broad_seeds,
            ticks=int(ticks),
            max_branch_points_per_seed=int(max_branch_points_per_seed),
        )
        selection = _selection_report(
            selected,
            strict_broad_seeds=strict_broad_seeds,
            max_branch_points_per_seed=max_branch_points_per_seed,
        )
        if selected and attempt_branch_replay:
            materialized, materialization = materialize_selected_branch_points(
                selected,
                ticks=int(ticks),
            )
            if materialization.get("passed") is True:
                branch_results = evaluate_materialized_branch_points(
                    materialized,
                    verify_replay=bool(verify_replay),
                )
        elif selected:
            materialization = _empty_materialization_report(
                reason="branch_replay_disabled_plan_only",
                selected_branch_point_count=len(selected),
            )
        outcome_support = _outcome_support_summary(
            branch_results=branch_results,
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
            M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V163_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
        "contract": _diagnostics_only_contract(),
        "inputs": {
            "v162_report": str(v162_report_path),
            "v160_artifact": str(artifact_path),
            "trajectory_glob": selected_glob,
            "trajectory_paths": [str(path) for path in selected_paths or []],
            "strict_broad_seeds": [int(seed) for seed in strict_broad_seeds],
            "ticks": int(ticks),
            "max_branch_points_per_seed": int(max_branch_points_per_seed),
            "verify_replay": bool(verify_replay),
            "attempt_branch_replay": bool(attempt_branch_replay),
            "max_dominant_outcome_support_action_share": _round(
                max_dominant_outcome_support_action_share
            ),
        },
        "source_validation": source_validation,
        "source_v162_digest_validation": _source_v162_digest_validation(
            source_validation
        ),
        "artifact_load": _artifact_load_report(artifact_load),
        "evidence": evidence_report,
        "selection_plan": selection,
        "branch_materialization": materialization,
        "branch_results": branch_results,
        "outcome_support": outcome_support,
        "action_value_target_expansion_feasible": (
            classification
            == "tied_set_branch_target_expansion_support_ready_no_training"
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
        "non_promoted": True,
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def validate_v163_sources(v162_report: Mapping[str, object]) -> dict[str, object]:
    failures: list[str] = []
    if (
        v162_report.get("schema_version")
        != M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_SCHEMA_VERSION
    ):
        failures.append("v162_schema_version_mismatch")
    if (
        v162_report.get("policy")
        != M3_CARRION_SURVIVOR_CONTINUATION_V162_SHADOW_TIE_COLLAPSE_AUTOPSY_POLICY
    ):
        failures.append("v162_policy_mismatch")
    observed_classification = _mapping(v162_report.get("classification")).get(
        "primary"
    )
    if observed_classification != EXPECTED_V162_CLASSIFICATION:
        failures.append("v162_unexpected_classification")
    digest_validation = exact_digest_validation_report(v162_report)
    if digest_validation.get("passed") is not True:
        failures.append("v162_exact_digest_mismatch")
    lifecycle = _v162_lifecycle_validation(v162_report)
    if lifecycle.get("passed") is not True:
        failures.append("v162_lifecycle_not_diagnostics_only")
    return {
        "policy": "m3_carrion_survivor_continuation_v163_source_validation_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "expected_v162_classification": EXPECTED_V162_CLASSIFICATION,
        "observed_v162_classification": observed_classification,
        "v162_exact_digest": v162_report.get("exact_digest"),
        "v162_exact_digest_validation": digest_validation,
        "v162_lifecycle_validation": lifecycle,
    }


def select_representative_tied_branch_points(
    *,
    predictions: Sequence[Mapping[str, object]],
    evidence_records: Sequence[Mapping[str, object]],
    strict_broad_seeds: Sequence[int] = STRICT_BROAD_SEEDS,
    ticks: int = DEFAULT_TICKS,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
) -> list[V163SelectedBranchPoint]:
    by_seed: defaultdict[int, list[tuple[Mapping[str, object], Mapping[str, object]]]]
    by_seed = defaultdict(list)
    for prediction, evidence in zip(predictions, evidence_records, strict=True):
        seed = _int(prediction.get("source_seed"))
        if seed not in {int(item) for item in strict_broad_seeds}:
            continue
        top_set = tuple(
            str(action)
            for action in prediction.get("top_value_candidate_set", [])
            if str(action) in ACTION_NAMES
        )
        if len(top_set) <= 1:
            continue
        by_seed[seed].append((prediction, evidence))
    selected: list[V163SelectedBranchPoint] = []
    for seed in strict_broad_seeds:
        seed_items = sorted(
            by_seed.get(int(seed), []),
            key=_selection_sort_key,
        )
        for branch_index, (prediction, evidence) in enumerate(
            seed_items[: max(0, int(max_branch_points_per_seed))]
        ):
            record = _mapping(evidence.get("record"))
            top_set = tuple(
                str(action)
                for action in prediction.get("top_value_candidate_set", [])
                if str(action) in ACTION_NAMES
            )
            action_mask = _bool_action_mask(
                record.get("public_action_mask") or record.get("action_mask")
            )
            source_path = str(evidence.get("source_path", ""))
            line_number = _int(evidence.get("line_number"))
            tick = _int(record.get("tick"))
            agent_id = _int(record.get("agent_id"))
            branch_id = (
                f"v163-broad-seed-{int(seed)}-branch-{branch_index}-"
                f"tick-{tick}-agent-{agent_id}"
            )
            selected.append(
                V163SelectedBranchPoint(
                    branch_id=branch_id,
                    seed=int(seed),
                    ticks=int(ticks),
                    branch_tick=tick,
                    record_index=_int(prediction.get("record_index")),
                    branch_index=branch_index,
                    agent_id=agent_id,
                    source_path=source_path,
                    line_number=line_number,
                    runtime_requested_action=_action_or_empty(
                        record.get("requested_action")
                    ),
                    runtime_resolved_action=_action_or_empty(
                        record.get("resolved_action")
                    ),
                    predicted_action=_action_or_empty(
                        prediction.get("predicted_action")
                    ),
                    nearest_neighbor_row_index=_int(
                        prediction.get("nearest_neighbor_row_index")
                    ),
                    top_value_candidate_set=top_set,
                    action_mask=action_mask,
                    observation_input=dict(_mapping(record.get("observation_input"))),
                    observation_schema=_optional_string(
                        record.get("observation_schema")
                    ),
                    observation_digest=_optional_string(
                        record.get("observation_digest")
                    ),
                    source_record_digest=stable_payload_digest(
                        _record_materialization_payload(record)
                    ),
                    selection_rationale={
                        "preferred_tied_set": (
                            _candidate_set_key(top_set) in PREFERRED_TIED_SET_KEYS
                        ),
                        "preferred_nearest_row": _int(
                            prediction.get("nearest_neighbor_row_index")
                        )
                        in PREFERRED_NEAREST_ROWS,
                        "top_value_candidate_set_key": _candidate_set_key(top_set),
                        "top_value_candidate_set_size": len(top_set),
                        "nearest_neighbor_row_index": _int(
                            prediction.get("nearest_neighbor_row_index")
                        ),
                        "runtime_requested_action_used_as_scorer_input": False,
                        "seed_tick_agent_path_digest_for_materialization_only": True,
                    },
                )
            )
    return selected


def materialize_selected_branch_points(
    selected: Sequence[V163SelectedBranchPoint],
    *,
    ticks: int = DEFAULT_TICKS,
) -> tuple[list[V163MaterializedBranchPoint], dict[str, object]]:
    by_seed: defaultdict[int, list[V163SelectedBranchPoint]] = defaultdict(list)
    for point in selected:
        by_seed[int(point.seed)].append(point)
    materialized: list[V163MaterializedBranchPoint] = []
    failures: list[dict[str, object]] = []
    reference_runs: list[dict[str, object]] = []
    for seed, seed_points in sorted(by_seed.items()):
        world = SimulationWorld(
            WorldConfig(seed=int(seed), max_ticks=int(ticks)),
            policy=evaluate_cli._mind_v3_policy(seed=int(seed), founder_template=None),
        )
        _configure_manual_summary_run(world)
        points_by_tick: defaultdict[int, list[V163SelectedBranchPoint]]
        points_by_tick = defaultdict(list)
        for point in seed_points:
            points_by_tick[int(point.branch_tick)].append(point)
        pending_ids = {point.branch_id for point in seed_points}
        for tick in range(int(ticks)):
            world.tick = tick
            snapshot = deepcopy(world) if tick in points_by_tick else None
            world._run_tick()
            if tick in points_by_tick and snapshot is not None:
                for point in points_by_tick[tick]:
                    match = _matching_tick_record(
                        world.tick_trajectory_records,
                        point=point,
                    )
                    if match is None:
                        failures.append(
                            _materialization_failure(
                                point,
                                reason="selected_record_not_materialized",
                            )
                        )
                        continue
                    mismatches = _record_mismatches(point, match)
                    if mismatches:
                        failures.append(
                            _materialization_failure(
                                point,
                                reason="selected_record_mismatch",
                                mismatches=mismatches,
                                materialized_record_digest=stable_payload_digest(
                                    _record_materialization_payload(match)
                                ),
                            )
                        )
                        continue
                    branch_state = deepcopy(snapshot)
                    branch_state_digest = _branch_state_digest(
                        branch_state,
                        branch_id=point.branch_id,
                        branch_tick=point.branch_tick,
                    )
                    materialized.append(
                        V163MaterializedBranchPoint(
                            selected=point,
                            branch_state_digest=branch_state_digest,
                            world=branch_state,
                            materialized_record_digest=stable_payload_digest(
                                _record_materialization_payload(match)
                            ),
                        )
                    )
                    pending_ids.discard(point.branch_id)
            if not world.alive_agents():
                break
        for branch_id in sorted(pending_ids):
            point = next(item for item in seed_points if item.branch_id == branch_id)
            failures.append(
                _materialization_failure(
                    point,
                    reason="selected_tick_not_reached_before_terminal_run",
                )
            )
        reference_runs.append(_reference_run_from_world(world, seed=seed, ticks=ticks))
    return (
        materialized,
        {
            "policy": "m3_carrion_survivor_continuation_v163_exact_branch_materialization_v1",
            "selected_branch_point_count": len(selected),
            "materialized_branch_point_count": len(materialized),
            "materialization_failure_count": len(failures),
            "materialization_failures": failures[:24],
            "reference_runs": reference_runs,
            "exact_materialization_proven": not failures
            and len(materialized) == len(selected)
            and bool(selected),
            "passed": not failures and len(materialized) == len(selected) and bool(selected),
        },
    )


def evaluate_materialized_branch_points(
    materialized: Sequence[V163MaterializedBranchPoint],
    *,
    verify_replay: bool = True,
) -> list[dict[str, object]]:
    return [
        _evaluate_materialized_branch_point(point, verify_replay=bool(verify_replay))
        for point in materialized
    ]


def _evaluate_materialized_branch_point(
    point: V163MaterializedBranchPoint,
    *,
    verify_replay: bool,
) -> dict[str, object]:
    selected = point.selected
    reference_run = _execute_reference_once(point)
    candidate_runs = [
        _execute_candidate_action_branch(
            point,
            forced_action=action,
            reference_run=reference_run,
            verify_replay=bool(verify_replay),
        )
        for action in selected.top_value_candidate_set
        if selected.action_mask.get(action) is True
    ]
    best_actions = _best_outcome_actions(candidate_runs)
    return {
        "branch_id": selected.branch_id,
        "seed": selected.seed,
        "fixture": "broad",
        "ticks": selected.ticks,
        "branch_tick": selected.branch_tick,
        "record_index": selected.record_index,
        "branch_index": selected.branch_index,
        "agent_id": selected.agent_id,
        "source_path": selected.source_path,
        "line_number": selected.line_number,
        "runtime_requested_action": selected.runtime_requested_action,
        "runtime_resolved_action": selected.runtime_resolved_action,
        "predicted_action": selected.predicted_action,
        "nearest_neighbor_row_index": selected.nearest_neighbor_row_index,
        "top_value_candidate_set": list(selected.top_value_candidate_set),
        "branch_state_digest": point.branch_state_digest,
        "source_record_digest": selected.source_record_digest,
        "materialized_record_digest": point.materialized_record_digest,
        "reference_runtime": reference_run,
        "candidate_runs": candidate_runs,
        "candidate_run_count": len(candidate_runs),
        "best_outcome_actions": best_actions,
        "best_outcome_action_count": len(best_actions),
        "replay_verification_passed": all(
            _mapping(run.get("replay_verification")).get("verified") is True
            for run in candidate_runs
        )
        if candidate_runs
        else False,
        "selection_rationale": dict(selected.selection_rationale),
        "diagnostics_only": True,
    }


def _execute_reference_once(
    point: V163MaterializedBranchPoint,
) -> dict[str, object]:
    world = deepcopy(point.world)
    _configure_manual_summary_run(world)
    for tick in range(point.selected.branch_tick, point.selected.ticks):
        world.tick = tick
        world._run_tick()
        if not world.alive_agents():
            break
    return _summary_from_world(
        world,
        selected=point.selected,
        forced_action=None,
        forced_used=False,
        reference_run=None,
        branch_state_digest=point.branch_state_digest,
    )


def _execute_candidate_action_branch(
    point: V163MaterializedBranchPoint,
    *,
    forced_action: str,
    reference_run: Mapping[str, object],
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_candidate_once(
        point,
        forced_action=forced_action,
        reference_run=reference_run,
    )
    verification = None
    if verify_replay:
        replay, replay_digest = _execute_candidate_once(
            point,
            forced_action=forced_action,
            reference_run=reference_run,
        )
        verification = {
            "verified": replay_digest == digest,
            "expected_digest": digest,
            "actual_digest": replay_digest,
            "replay_alive_agents": replay.get("alive_agents"),
            "replay_births": replay.get("births"),
            "replay_deaths": replay.get("deaths"),
        }
    run["replay_digest"] = digest
    run["replay_verification"] = verification
    return run


def _execute_candidate_once(
    point: V163MaterializedBranchPoint,
    *,
    forced_action: str,
    reference_run: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = _normal_mind_v3_delegate(world.policy)
    world.policy = _ForcedTiedCandidateThenMindV3Policy(
        target_agent_id=point.selected.agent_id,
        forced_action=forced_action,
        delegate=delegate,
    )
    _configure_manual_summary_run(world)
    for tick in range(point.selected.branch_tick, point.selected.ticks):
        world.tick = tick
        world._run_tick()
        if not world.alive_agents():
            break
    forced_used = bool(getattr(world.policy, "used", False))
    run = _summary_from_world(
        world,
        selected=point.selected,
        forced_action=forced_action,
        forced_used=forced_used,
        reference_run=reference_run,
        branch_state_digest=point.branch_state_digest,
    )
    return run, stable_payload_digest(_branch_run_digest_payload(run))


def _summary_from_world(
    world: SimulationWorld,
    *,
    selected: V163SelectedBranchPoint,
    forced_action: str | None,
    forced_used: bool,
    reference_run: Mapping[str, object] | None,
    branch_state_digest: str,
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    requested_action_counts = Counter(
        str(record.get("requested_action"))
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    resolved_action_counts = Counter(
        str(record.get("resolved_action"))
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
    )
    unsupported_requested_action_count = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
        and record.get("action_valid") is False
    )
    target = _target_terminal(world, selected.agent_id)
    run = {
        "branch_id": selected.branch_id,
        "seed": selected.seed,
        "fixture": "broad",
        "ticks": selected.ticks,
        "branch_tick": selected.branch_tick,
        "record_index": selected.record_index,
        "agent_id": selected.agent_id,
        "runtime_requested_action": selected.runtime_requested_action,
        "predicted_action": selected.predicted_action,
        "forced_action": forced_action,
        "forced_action_used": bool(forced_used),
        "forced_action_supported": (
            forced_action in selected.top_value_candidate_set
            and selected.action_mask.get(str(forced_action), False)
            if forced_action is not None
            else None
        ),
        "branch_state_digest": branch_state_digest,
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "target_terminal": target,
        "target_alive_at_end": target["alive"],
        "target_energy_ratio_at_end": target["energy_ratio"],
        "target_hydration_ratio_at_end": target["hydration_ratio"],
        "target_health_ratio_at_end": target["health_ratio"],
        "first_action_outcome": _first_target_action_outcome(
            world.trajectory_records,
            selected=selected,
            forced_action=forced_action,
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": evaluate_cli._heuristic_action_source_count(
            action_source_counts
        ),
        "diagnostic_forced_action_source_count": sum(
            count
            for source, count in action_source_counts.items()
            if source.startswith("tied_set_branch_target_expansion_force:")
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "action_source_counts": dict(sorted(action_source_counts.items())),
    }
    if reference_run is not None:
        run["deltas_vs_recorded_reference_runtime"] = _run_deltas_vs_reference(
            run,
            reference_run,
        )
    return run


def _outcome_support_summary(
    *,
    branch_results: Sequence[Mapping[str, object]],
    max_dominant_outcome_support_action_share: float,
) -> dict[str, object]:
    support_counts: Counter[str] = Counter()
    replay_count = 0
    replay_verified_count = 0
    candidate_run_count = 0
    for result in branch_results:
        for action in result.get("best_outcome_actions", []):
            if str(action) in ACTION_NAMES:
                support_counts.update([str(action)])
        for run in _list_of_mappings(result.get("candidate_runs")):
            candidate_run_count += 1
            replay = _mapping(run.get("replay_verification"))
            if replay:
                replay_count += 1
                replay_verified_count += int(replay.get("verified") is True)
    dominant = _dominant_count_share(support_counts)
    noncollapsed = (
        sum(support_counts.values()) > 0
        and len(support_counts) >= 2
        and float(dominant.get("share") or 0.0)
        <= float(max_dominant_outcome_support_action_share)
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v163_outcome_support_summary_v1",
        "branch_result_count": len(branch_results),
        "candidate_run_count": candidate_run_count,
        "replay_verification_count": replay_count,
        "replay_verified_count": replay_verified_count,
        "all_replays_verified": replay_count > 0 and replay_count == replay_verified_count,
        "per_action_outcome_support_counts": dict(sorted(support_counts.items())),
        "dominant_outcome_support_action": dominant.get("key"),
        "dominant_outcome_support_action_count": dominant.get("count"),
        "dominant_outcome_support_action_share": dominant.get("share"),
        "max_dominant_outcome_support_action_share": _round(
            max_dominant_outcome_support_action_share
        ),
        "outcome_support_noncollapsed": noncollapsed,
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
        return "exact_branch_materialization_blocked_no_training"
    if outcome_support.get("all_replays_verified") is not True:
        return "branch_replay_not_deterministic_closed_no_training"
    if (
        branch_results
        and outcome_support.get("outcome_support_noncollapsed") is True
    ):
        return "tied_set_branch_target_expansion_support_ready_no_training"
    return "branch_target_support_insufficient_no_live_ab"


def _route_recommendation(classification: str) -> dict[str, object]:
    ready = classification == "tied_set_branch_target_expansion_support_ready_no_training"
    return {
        "policy": "m3_carrion_survivor_continuation_v163_route_recommendation_v1",
        "recommended_next_route": (
            "v164_target_dataset_expansion_from_exact_tied_set_branch_evidence"
            if ready
            else "branch_replay_contract_work_before_target_expansion"
        ),
        "v164_target_dataset_expansion_recommended": ready,
        "branch_replay_contract_work_recommended": not ready,
        "threshold_tuning_recommended": False,
        "future_shadow_evaluation_recommended": False,
        "future_separate_opt_in_live_ab_diagnostic_recommended": False,
        "live_ab_allowed": False,
        "runtime_policy_integration_allowed": False,
        "promotion_authorized": False,
        "runtime_action_selection_changed": False,
    }


def _load_artifact_for_v163(path: Path) -> dict[str, object]:
    if not path or not str(path):
        return {"passed": False, "reason": "missing_v160_artifact_path"}
    try:
        artifact = load_json_report(path)
    except OSError as exc:
        return {"passed": False, "reason": "v160_artifact_load_error", "error": str(exc)}
    return {
        "passed": True,
        "path": str(path),
        "artifact_digest": stable_payload_digest(artifact),
        "artifact": artifact,
    }


def _artifact_load_report(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "artifact"}


def _with_source_failure(
    source_validation: Mapping[str, object],
    failure: str,
) -> dict[str, object]:
    failures = sorted(set([str(item) for item in source_validation.get("failures", [])] + [failure]))
    payload = dict(source_validation)
    payload["passed"] = False
    payload["failures"] = failures
    return payload


def _selected_trajectory_paths(
    *,
    v162_report: Mapping[str, object],
    trajectory_paths: Sequence[str | Path] | None,
) -> list[Path] | None:
    if trajectory_paths:
        return [Path(path) for path in trajectory_paths]
    report_paths = _mapping(v162_report.get("inputs")).get("trajectory_paths")
    if isinstance(report_paths, Sequence) and not isinstance(report_paths, (str, bytes)):
        paths = [Path(str(path)) for path in report_paths]
        if paths:
            return paths
    return None


def _evidence_report(evidence: Mapping[str, object]) -> dict[str, object]:
    payload = dict(evidence)
    payload.pop("records", None)
    return payload


def _selection_sort_key(
    item: tuple[Mapping[str, object], Mapping[str, object]],
) -> tuple[int, int, int, int, int]:
    prediction, evidence = item
    record = _mapping(evidence.get("record"))
    set_key = _candidate_set_key(
        [str(action) for action in prediction.get("top_value_candidate_set", [])]
    )
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


def _selection_report(
    selected: Sequence[V163SelectedBranchPoint],
    *,
    strict_broad_seeds: Sequence[int],
    max_branch_points_per_seed: int,
) -> dict[str, object]:
    by_seed = Counter(point.seed for point in selected)
    by_set = Counter(_candidate_set_key(point.top_value_candidate_set) for point in selected)
    by_row = Counter(str(point.nearest_neighbor_row_index) for point in selected)
    return {
        "policy": "m3_carrion_survivor_continuation_v163_representative_selection_plan_v1",
        "strict_broad_seeds": [int(seed) for seed in strict_broad_seeds],
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "preferred_tied_set_keys": list(PREFERRED_TIED_SET_KEYS),
        "preferred_nearest_rows": [int(row) for row in PREFERRED_NEAREST_ROWS],
        "selected_branch_point_count": len(selected),
        "selected_branch_points_by_seed": dict(sorted(by_seed.items())),
        "selected_tied_set_counts": dict(sorted(by_set.items())),
        "selected_nearest_row_counts": dict(sorted(by_row.items())),
        "selected_branch_points": [_selected_payload(point) for point in selected],
    }


def _empty_selection_report(
    *,
    strict_broad_seeds: Sequence[int],
    max_branch_points_per_seed: int,
) -> dict[str, object]:
    return _selection_report(
        [],
        strict_broad_seeds=strict_broad_seeds,
        max_branch_points_per_seed=max_branch_points_per_seed,
    )


def _selected_payload(point: V163SelectedBranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": "broad",
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
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


def _empty_evidence_report(
    *,
    trajectory_glob: str,
    trajectory_paths: Sequence[Path] | None,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v163_shadow_evidence_not_loaded_v1",
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
        "policy": "m3_carrion_survivor_continuation_v163_exact_branch_materialization_v1",
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


def _matching_tick_record(
    records: Sequence[Mapping[str, object]],
    *,
    point: V163SelectedBranchPoint,
) -> Mapping[str, object] | None:
    matches = [
        record
        for record in records
        if _int(record.get("tick")) == point.branch_tick
        and _int(record.get("agent_id")) == point.agent_id
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _record_mismatches(
    point: V163SelectedBranchPoint,
    record: Mapping[str, object],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    comparisons = {
        "requested_action": (
            point.runtime_requested_action,
            _action_or_empty(record.get("requested_action")),
        ),
        "resolved_action": (
            point.runtime_resolved_action,
            _action_or_empty(record.get("resolved_action")),
        ),
        "observation_digest": (point.observation_digest, record.get("observation_digest")),
        "action_mask_digest": (
            stable_payload_digest(point.action_mask),
            stable_payload_digest(
                _bool_action_mask(record.get("public_action_mask") or record.get("action_mask"))
            ),
        ),
        "record_materialization_digest": (
            point.source_record_digest,
            stable_payload_digest(_record_materialization_payload(record)),
        ),
    }
    for field, (expected, observed) in comparisons.items():
        if expected != observed:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                }
            )
    return mismatches


def _materialization_failure(
    point: V163SelectedBranchPoint,
    *,
    reason: str,
    mismatches: Sequence[Mapping[str, object]] = (),
    materialized_record_digest: str | None = None,
) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "source_path": point.source_path,
        "line_number": point.line_number,
        "reason": reason,
        "mismatches": [dict(item) for item in mismatches],
        "source_record_digest": point.source_record_digest,
        "materialized_record_digest": materialized_record_digest,
    }


def _reference_run_from_world(
    world: SimulationWorld,
    *,
    seed: int,
    ticks: int,
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    return {
        "seed": int(seed),
        "ticks": int(ticks),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": evaluate_cli._heuristic_action_source_count(
            action_source_counts
        ),
    }


def _first_target_action_outcome(
    records: Sequence[Mapping[str, object]],
    *,
    selected: V163SelectedBranchPoint,
    forced_action: str | None,
) -> dict[str, object] | None:
    for record in records:
        if (
            _int(record.get("tick")) == selected.branch_tick
            and _int(record.get("agent_id")) == selected.agent_id
        ):
            return {
                "requested_action": record.get("requested_action"),
                "resolved_action": record.get("resolved_action"),
                "action_valid": record.get("action_valid"),
                "resolution_action_valid": record.get("resolution_action_valid"),
                "matches_forced_action": (
                    None
                    if forced_action is None
                    else record.get("requested_action") == forced_action
                ),
                "outcome": evaluate_cli._json_ready(record.get("outcome", {})),
            }
    return None


def _run_deltas_vs_reference(
    run: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, object]:
    target = _mapping(run.get("target_terminal"))
    reference_target = _mapping(reference.get("target_terminal"))
    return {
        "alive_agents": _int(run.get("alive_agents")) - _int(reference.get("alive_agents")),
        "births": _int(run.get("births")) - _int(reference.get("births")),
        "deaths": _int(run.get("deaths")) - _int(reference.get("deaths")),
        "target_alive": _bool_int(target.get("alive")) - _bool_int(reference_target.get("alive")),
        "target_energy_ratio": _finite_delta(
            target.get("energy_ratio"),
            reference_target.get("energy_ratio"),
        ),
        "target_hydration_ratio": _finite_delta(
            target.get("hydration_ratio"),
            reference_target.get("hydration_ratio"),
        ),
        "target_health_ratio": _finite_delta(
            target.get("health_ratio"),
            reference_target.get("health_ratio"),
        ),
        "unsupported_requested_action_count": _int(
            run.get("unsupported_requested_action_count")
        )
        - _int(reference.get("unsupported_requested_action_count")),
    }


def _best_outcome_actions(
    candidate_runs: Sequence[Mapping[str, object]],
) -> list[str]:
    if not candidate_runs:
        return []
    keyed = [(run, _outcome_key(run)) for run in candidate_runs]
    best_key = max(key for _, key in keyed)
    return sorted(
        [
            str(run.get("forced_action"))
            for run, key in keyed
            if key == best_key and str(run.get("forced_action")) in ACTION_NAMES
        ],
        key=_action_order,
    )


def _outcome_key(run: Mapping[str, object]) -> tuple[object, ...]:
    target = _mapping(run.get("target_terminal"))
    return (
        _int(run.get("alive_agents")),
        _int(run.get("births")),
        -_int(run.get("deaths")),
        _bool_int(target.get("alive")),
        _finite_or_negative(target.get("energy_ratio")),
        _finite_or_negative(target.get("hydration_ratio")),
        _finite_or_negative(target.get("health_ratio")),
        -_int(run.get("unsupported_requested_action_count")),
    )


def _branch_run_digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "record_index": run.get("record_index"),
        "agent_id": run.get("agent_id"),
        "runtime_requested_action": run.get("runtime_requested_action"),
        "predicted_action": run.get("predicted_action"),
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "branch_state_digest": run.get("branch_state_digest"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_terminal": run.get("target_terminal"),
        "requested_action_counts": run.get("requested_action_counts"),
        "resolved_action_counts": run.get("resolved_action_counts"),
        "unsupported_requested_action_count": run.get(
            "unsupported_requested_action_count"
        ),
        "action_source_counts": run.get("action_source_counts"),
        "deltas_vs_recorded_reference_runtime": run.get(
            "deltas_vs_recorded_reference_runtime"
        ),
    }


def _record_materialization_payload(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "tick": record.get("tick"),
        "agent_id": record.get("agent_id"),
        "requested_action": record.get("requested_action"),
        "resolved_action": record.get("resolved_action"),
        "action_mask": _bool_action_mask(
            record.get("public_action_mask") or record.get("action_mask")
        ),
        "observation_digest": record.get("observation_digest"),
        "before": record.get("before"),
    }


def _candidate_set_key(actions: Sequence[str]) -> str:
    valid = [str(action) for action in actions if str(action) in ACTION_NAMES]
    if not valid:
        return "<none>"
    return "|".join(sorted(valid, key=_action_order))


def _bool_action_mask(value: object) -> dict[str, bool]:
    payload = _mapping(value)
    return {
        action: payload.get(action) is True
        for action in ACTION_NAMES
    }


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _bool_int(value: object) -> int:
    return 1 if value is True else 0


def _finite_delta(left: object, right: object) -> float | None:
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return None
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return None
    return _round(float(left) - float(right))


def _finite_or_negative(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return -1.0


def _v162_lifecycle_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_allowed",
        "promotion_authorized",
        "runtime_promotion_allowed",
    ):
        if field in report and report.get(field) is not False:
            failures.append({"field": field, "observed": report.get(field)})
    contract = _mapping(report.get("contract"))
    for field in (
        "runtime_action_selection_changed",
        "runtime_artifact_created",
        "live_ab_allowed",
        "promotion_authorized",
        "runtime_promotion_allowed",
    ):
        if field in contract and contract.get(field) is not False:
            failures.append(
                {"field": f"contract.{field}", "observed": contract.get(field)}
            )
    return {
        "policy": "m3_carrion_survivor_continuation_v163_v162_lifecycle_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _source_v162_digest_validation(
    source_validation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "v162_exact_digest": source_validation.get("v162_exact_digest"),
        "v162_exact_digest_validation": source_validation.get(
            "v162_exact_digest_validation"
        ),
        "expected_v162_classification": source_validation.get(
            "expected_v162_classification"
        ),
        "observed_v162_classification": source_validation.get(
            "observed_v162_classification"
        ),
    }


def _diagnostics_only_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "branch_target_expansion_diagnostic_only": True,
        "validates_v162_exact_digest_before_analysis": True,
        "requires_v162_blocked_classification": True,
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
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }


def _lifecycle_proof() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v163_lifecycle_proof_v1",
        "diagnostics_only": True,
        "runtime_action_selection_changed": False,
        "runtime_artifact_created": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "threshold_tuning_recommended": False,
    }


def _json_round_trip_digest(payload: Mapping[str, object]) -> str:
    json_payload = json.loads(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
    return stable_payload_digest(json_payload)
