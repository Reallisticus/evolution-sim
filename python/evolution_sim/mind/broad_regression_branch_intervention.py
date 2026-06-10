from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.mind import evaluation_harness as evaluate_cli
from evolution_sim.mind import transition_value_live_ab as v142_live_ab
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.carrion_branch_explore import _branch_state_digest
from evolution_sim.mind.dataset import load_trajectory_jsonl
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION = (
    "mind_v3_v143_broad_regression_branch_intervention_v1"
)
MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION = (
    "mind_v3_v143_broad_regression_branch_intervention_dataset_row_v1"
)
MIND_V3_V143_BRANCH_INTERVENTION_POLICY = (
    "diagnostics_only_v143_broad_regression_branch_intervention_archive_v1"
)
DEFAULT_SCORER_PATH = Path(
    "output/mind/mind-v3-v142-transition-value-scorer.json"
)
DEFAULT_LIVE_REPORT_PATH = Path(
    "output/mind/mind-v3-v142-transition-value-live-ab.json"
)
DEFAULT_REPORT_PATH = Path(
    "output/mind/mind-v3-v143-broad-regression-branch-intervention-report.json"
)
DEFAULT_DATASET_PATH = Path(
    "output/mind/mind-v3-v143-broad-regression-branch-intervention-dataset.jsonl"
)
DEFAULT_V142_TRAJECTORY_DIR = Path(
    "output/mind/v143-broad-regression-branch-intervention/v142-trajectories"
)
DEFAULT_MAX_BRANCH_POINTS_PER_SEED = 1
DEFAULT_MAX_CANDIDATE_ACTIONS = 4
MAX_DOMINANT_LABEL_ACTION_SHARE = 0.50
FORBIDDEN_TRAINABLE_TOKENS = (
    "seed",
    "fixture",
    "branch",
    "path",
    "digest",
    "provenance",
    "record_index",
    "tick",
    "agent_id",
    "source",
    "trajectory",
)


class BroadRegressionBranchInterventionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BroadRegressionBranchPoint:
    branch_id: str
    seed: int
    fixture: str
    ticks: int
    branch_tick: int
    record_index: int
    branch_index: int
    agent_id: int
    baseline_action: str
    v142_requested_action: str
    v142_resolved_action: str
    action_mask: dict[str, bool]
    observation_input: dict[str, object]
    observation_schema: str | None
    observation_digest: str | None
    source_trajectory_path: str | None
    branch_state_digest: str
    world: SimulationWorld


class _ForcedFirstActionThenMindV3Policy:
    policy_id = "mind_v3_v143_broad_regression_branch_intervention_force"
    policy_version = (
        "mind_v3_v143_broad_regression_branch_intervention_force_v1"
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
            and int(agent_id) == self.target_agent_id
            and self.forced_action in ACTION_NAMES
            and bool(action_mask.get(self.forced_action, False))
        ):
            self.used = True
            return ActionDecision(
                requested_action=self.forced_action,
                source=f"branch_intervention_force:{self.forced_action}",
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                diagnostics={
                    "forced_first_action": self.forced_action,
                    "target_agent_id": self.target_agent_id,
                    "diagnostics_only": True,
                    "heuristic_free": True,
                },
            )
        decide = getattr(self.delegate, "decide", None)
        if not callable(decide):
            raise BroadRegressionBranchInterventionError(
                "branch delegate policy does not implement decide"
            )
        return decide(observation, action_mask)

    def observe_transition(self, record: dict[str, object]) -> dict[str, object] | None:
        observe = getattr(self.delegate, "observe_transition", None)
        if not callable(observe):
            return None
        feedback = dict(record)
        if (
            _int(feedback.get("agent_id"), default=-1) == self.target_agent_id
            and feedback.get("action_source")
            == f"branch_intervention_force:{self.forced_action}"
        ):
            feedback["policy_id"] = getattr(self.delegate, "policy_id", None)
            feedback["policy_version"] = getattr(self.delegate, "policy_version", None)
        return observe(feedback)


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    with _open_input(resolved) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BroadRegressionBranchInterventionError(
            f"report must be a JSON object: {resolved}"
        )
    return payload


def write_broad_regression_branch_intervention_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        handle.write(_canonical_json(report, indent=2))
        handle.write("\n")


def write_broad_regression_branch_intervention_dataset(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        for row in rows:
            handle.write(_canonical_json(row, indent=None))
            handle.write("\n")


def build_broad_regression_branch_intervention_archive(
    *,
    scorer_report_path: str | Path = DEFAULT_SCORER_PATH,
    live_report_path: str | Path = DEFAULT_LIVE_REPORT_PATH,
    v142_trajectory_output_dir: str | Path = DEFAULT_V142_TRAJECTORY_DIR,
    max_branch_points_per_seed: int = DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    max_candidate_actions: int = DEFAULT_MAX_CANDIDATE_ACTIONS,
    regenerate_v142_trajectories: bool = True,
    verify_replay: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    scorer_path = Path(scorer_report_path)
    live_path = Path(live_report_path)
    trajectory_dir = Path(v142_trajectory_output_dir)
    if regenerate_v142_trajectories:
        regeneration = _ensure_v142_trajectory_inputs(
            scorer_report_path=scorer_path,
            live_report_path=live_path,
            trajectory_output_dir=trajectory_dir,
        )
    else:
        regeneration = {
            "policy": "v142_trajectory_regeneration_skipped_v1",
            "trajectory_output_dir": str(trajectory_dir),
            "missing_before_regeneration": [],
            "regenerated": False,
            "expected_paths": [
                str(path)
                for path in _expected_v142_trajectory_paths(trajectory_dir)
            ],
        }
    scorer_report = load_json_report(scorer_path)
    live_report = load_json_report(live_path)
    source_precheck = _source_precheck(scorer_report, live_report)
    scorer = evaluate_cli._load_transition_value_scorer_artifact(scorer_report)
    branch_limit = _positive_int(
        max_branch_points_per_seed,
        field="max_branch_points_per_seed",
    )
    candidate_limit = _nonnegative_int(
        max_candidate_actions,
        field="max_candidate_actions",
    )
    regression_seeds = _ordered_regression_seeds(live_report)
    reference_runs, branch_points, materialization = _materialize_branch_points(
        scorer=scorer,
        live_report=live_report,
        regression_seeds=regression_seeds,
        max_branch_points_per_seed=branch_limit,
        v142_trajectory_output_dir=trajectory_dir,
    )
    branch_results = [
        _evaluate_branch_point(
            point,
            reference_runs=reference_runs,
            max_candidate_actions=candidate_limit,
            verify_replay=verify_replay,
        )
        for point in branch_points
    ]
    dataset_rows = _dataset_rows_from_branch_results(
        branch_results=branch_results,
        source_report_path=live_path,
        scorer_report_path=scorer_path,
    )
    dataset_scan = _dataset_leakage_scan(dataset_rows)
    acceptance = _acceptance(
        source_precheck=source_precheck,
        branch_results=branch_results,
        dataset_rows=dataset_rows,
        regression_seeds=regression_seeds,
        dataset_scan=dataset_scan,
    )
    source_integrity = _source_integrity(
        source_precheck=source_precheck,
        branch_results=branch_results,
        materialization=materialization,
        dataset_scan=dataset_scan,
    )
    classification = _classification(source_integrity, acceptance)
    contract = {
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "trainer_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "trajectory_schema_changed": False,
        "runtime_promotion_authorized": False,
        "training_authorized": False,
        "diagnostics_only": True,
        "branch_replay_policy": (
            "deterministic_in_process_v142_override_state_snapshot_forced_"
            "first_action_then_normal_mind_v3_v1"
        ),
        "candidate_action_policy": (
            "baseline_mind_v3_action_then_v142_requested_action_then_all_valid_"
            "public_action_mask_actions_in_action_contract_order"
        ),
        "bounded_branch_points_per_seed": branch_limit,
        "max_candidate_actions": (
            None if candidate_limit == 0 else int(candidate_limit)
        ),
    }
    report = {
        "schema_version": MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
        "policy": MIND_V3_V143_BRANCH_INTERVENTION_POLICY,
        "contract": contract,
        "inputs": {
            "v142_scorer_report": str(scorer_path),
            "v142_live_report": str(live_path),
            "v142_trajectory_output_dir": str(trajectory_dir),
        },
        "v142_trajectory_regeneration": regeneration,
        "source_precheck": source_precheck,
        "source_integrity": source_integrity,
        "matrix": {
            "broad_seeds": list(v142_live_ab.STRICT_BROAD_SEEDS),
            "ticks": v142_live_ab.STRICT_TICKS,
            "fixture": "carrion_only",
            "fixture_seeds": list(v142_live_ab.STRICT_CARRION_FIXTURE_SEEDS),
            "fixture_ticks": v142_live_ab.STRICT_TICKS,
            "regression_seeds": list(regression_seeds),
        },
        "materialization": materialization,
        "reference_runs": _reference_report_section(reference_runs),
        "branch_points": [_branch_point_payload(point) for point in branch_points],
        "branch_results": branch_results,
        "dataset": {
            "row_schema_version": (
                MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION
            ),
            "row_count": len(dataset_rows),
            "leakage_scan": dataset_scan,
            "dataset_digest": stable_payload_digest(dataset_rows),
        },
        "acceptance": acceptance,
        "classification": {
            "primary": classification,
            "labels": [classification],
        },
        "non_default_runtime": True,
        "non_promoted": True,
    }
    return report, dataset_rows


def _ensure_v142_trajectory_inputs(
    *,
    scorer_report_path: Path,
    live_report_path: Path,
    trajectory_output_dir: Path,
) -> dict[str, object]:
    expected = _expected_v142_trajectory_paths(trajectory_output_dir)
    missing = [path for path in expected if not path.exists()]
    if missing:
        regenerated = v142_live_ab.build_transition_value_live_ab_report(
            transition_value_artifact_path=scorer_report_path,
            broad_seeds=v142_live_ab.STRICT_BROAD_SEEDS,
            ticks=v142_live_ab.STRICT_TICKS,
            fixture_seeds=v142_live_ab.STRICT_CARRION_FIXTURE_SEEDS,
            fixture_ticks=v142_live_ab.STRICT_TICKS,
            trajectory_output_dir=trajectory_output_dir,
        )
        live_report_path.parent.mkdir(parents=True, exist_ok=True)
        live_report_path.write_text(
            _canonical_json(regenerated, indent=2) + "\n",
            encoding="utf-8",
        )
    remaining = [path for path in expected if not path.exists()]
    return {
        "policy": "v143_requires_v142_trajectory_inputs_v1",
        "trajectory_output_dir": str(trajectory_output_dir),
        "expected_paths": [str(path) for path in expected],
        "missing_before_regeneration": [str(path) for path in missing],
        "regenerated": bool(missing),
        "missing_after_regeneration": [str(path) for path in remaining],
        "passed": not remaining,
    }


def _expected_v142_trajectory_paths(output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for seed in v142_live_ab.STRICT_BROAD_SEEDS:
        paths.append(
            evaluate_cli._trajectory_output_path(
                output_dir,
                "v142",
                "broad",
                "baseline",
                seed,
                v142_live_ab.STRICT_TICKS,
            )
        )
        paths.append(
            evaluate_cli._trajectory_output_path(
                output_dir,
                "v142",
                "broad",
                "override",
                seed,
                v142_live_ab.STRICT_TICKS,
            )
        )
    for prefix in ("v142-carrion-baseline", "v142-carrion-override"):
        for seed in v142_live_ab.STRICT_CARRION_FIXTURE_SEEDS:
            paths.append(
                evaluate_cli._trajectory_output_path(
                    output_dir,
                    prefix,
                    "carrion_only",
                    "mind_v3",
                    seed,
                    v142_live_ab.STRICT_TICKS,
                )
            )
            paths.append(
                evaluate_cli._trajectory_output_path(
                    output_dir,
                    prefix,
                    "carrion_only",
                    "heuristic",
                    seed,
                    v142_live_ab.STRICT_TICKS,
                )
            )
    return [path for path in paths if path is not None]


def _source_precheck(
    scorer_report: Mapping[str, object],
    live_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    scorer_integrity = evaluate_cli._transition_value_scorer_source_integrity(
        scorer_report
    )
    scorer_classification = _mapping(scorer_report.get("classification")).get(
        "primary"
    )
    live_classification = _mapping(live_report.get("classification")).get("primary")
    live_acceptance = _mapping(live_report.get("acceptance"))
    live_matrix = _mapping(live_report.get("matrix"))
    if scorer_integrity.get("passed") is not True:
        failures.append("v142_scorer_source_integrity_failed")
    if scorer_classification != evaluate_cli.MIND_V3_TRANSITION_VALUE_READY_CLASSIFICATION:
        failures.append("v142_scorer_not_ready")
    if live_report.get("schema_version") != v142_live_ab.MIND_V3_TRANSITION_VALUE_LIVE_AB_SCHEMA_VERSION:
        failures.append("v142_live_report_schema_mismatch")
    if live_classification != "transition_value_live_override_blocked_non_promotable":
        failures.append("v142_live_report_not_blocked_non_promotable")
    first_floor = str(live_acceptance.get("first_failed_floor", ""))
    first_fixture = str(live_acceptance.get("first_failing_fixture", ""))
    if not first_floor.startswith("broad_seed_") or first_fixture != "broad":
        failures.append("v142_live_report_not_blocked_on_broad_regression")
    if tuple(_int(seed) for seed in _list(live_matrix.get("broad_seeds"))) != v142_live_ab.STRICT_BROAD_SEEDS:
        failures.append("v142_live_report_broad_seed_matrix_mismatch")
    if _int(live_matrix.get("ticks")) != v142_live_ab.STRICT_TICKS:
        failures.append("v142_live_report_broad_ticks_mismatch")
    if tuple(_int(seed) for seed in _list(live_matrix.get("fixture_seeds"))) != v142_live_ab.STRICT_CARRION_FIXTURE_SEEDS:
        failures.append("v142_live_report_fixture_seed_matrix_mismatch")
    if _int(live_matrix.get("fixture_ticks")) != v142_live_ab.STRICT_TICKS:
        failures.append("v142_live_report_fixture_ticks_mismatch")
    regression_seeds = _ordered_regression_seeds(live_report)
    if not regression_seeds:
        failures.append("v142_live_report_has_no_broad_regression_seeds")
    return {
        "policy": "v143_v142_source_precheck_v1",
        "passed": not failures,
        "failures": failures,
        "v142_scorer_source_integrity_passed": scorer_integrity.get("passed") is True,
        "v142_scorer_classification": scorer_classification,
        "v142_live_classification": live_classification,
        "v142_first_failed_floor": live_acceptance.get("first_failed_floor"),
        "v142_first_failing_seed": live_acceptance.get("first_failing_seed"),
        "v142_first_failing_fixture": live_acceptance.get("first_failing_fixture"),
        "broad_regression_seeds": list(regression_seeds),
    }


def _ordered_regression_seeds(live_report: Mapping[str, object]) -> tuple[int, ...]:
    broad = _mapping(live_report.get("broad"))
    rows = _list_of_mappings(broad.get("per_seed_delta"))
    regression = [
        _int(row.get("seed"))
        for row in rows
        if _int(row.get("alive_delta")) < 0 or _int(row.get("births_delta")) < 0
    ]
    first = _int_or_none(_mapping(live_report.get("acceptance")).get("first_failing_seed"))
    ordered: list[int] = []
    if first is not None and first in regression:
        ordered.append(first)
    for seed in sorted(regression):
        if seed not in ordered:
            ordered.append(seed)
    return tuple(ordered)


def _materialize_branch_points(
    *,
    scorer: object,
    live_report: Mapping[str, object],
    regression_seeds: Sequence[int],
    max_branch_points_per_seed: int,
    v142_trajectory_output_dir: Path,
) -> tuple[
    dict[int, dict[str, Mapping[str, object]]],
    list[BroadRegressionBranchPoint],
    dict[str, object],
]:
    reference_runs: dict[int, dict[str, Mapping[str, object]]] = {}
    points: list[BroadRegressionBranchPoint] = []
    failures: list[dict[str, object]] = []
    outcome_mismatches: list[dict[str, object]] = []
    live_by_seed = _live_broad_runs_by_seed(live_report)
    for seed in regression_seeds:
        baseline_path = evaluate_cli._trajectory_output_path(
            v142_trajectory_output_dir,
            "v142",
            "broad",
            "baseline",
            seed,
            v142_live_ab.STRICT_TICKS,
        )
        override_path = evaluate_cli._trajectory_output_path(
            v142_trajectory_output_dir,
            "v142",
            "broad",
            "override",
            seed,
            v142_live_ab.STRICT_TICKS,
        )
        baseline_ref = _reference_from_live_run_and_trajectory(
            _mapping(_mapping(live_by_seed.get(int(seed))).get("baseline")),
            trajectory_path=baseline_path,
            seed=int(seed),
            ticks=v142_live_ab.STRICT_TICKS,
            runtime="baseline_mind_v3",
        )
        override_ref = _reference_from_live_run_and_trajectory(
            _mapping(_mapping(live_by_seed.get(int(seed))).get("override")),
            trajectory_path=override_path,
            seed=int(seed),
            ticks=v142_live_ab.STRICT_TICKS,
            runtime="v142_transition_value_override",
        )
        override_world = SimulationWorld(
            WorldConfig(seed=int(seed), max_ticks=v142_live_ab.STRICT_TICKS),
            policy=evaluate_cli._mind_v3_policy(
                seed=int(seed),
                founder_template=None,
                transition_value_scorer=scorer,
                transition_value_action_override=True,
                transition_value_action_override_source_integrity_passed=True,
            ),
        )
        _configure_manual_summary_run(override_world)
        seed_points, seed_failures = _run_override_seed_and_select_points(
            override_world,
            seed=int(seed),
            max_branch_points=max_branch_points_per_seed,
            source_trajectory_path=str(override_path),
        )
        failures.extend(seed_failures)
        reference_runs[int(seed)] = {
            "baseline": baseline_ref,
            "v142_override": override_ref,
        }
        points.extend(seed_points)
        outcome_mismatches.extend(
            _reference_mismatches(
                seed=int(seed),
                live_runs=live_by_seed,
                baseline_ref=baseline_ref,
                override_ref=override_ref,
            )
        )
    return (
        reference_runs,
        points,
        {
            "policy": "v143_materialize_v142_override_applied_branch_points_v1",
            "regression_seeds": [int(seed) for seed in regression_seeds],
            "max_branch_points_per_seed": int(max_branch_points_per_seed),
            "branch_point_count": len(points),
            "branch_points_by_seed": dict(
                sorted(Counter(point.seed for point in points).items())
            ),
            "materialization_failure_count": len(failures),
            "materialization_failures": failures[:24],
            "v142_reference_outcome_mismatch_count": len(outcome_mismatches),
            "v142_reference_outcome_mismatches": outcome_mismatches[:24],
            "passed": not failures and not outcome_mismatches,
        },
    )


def _run_override_seed_and_select_points(
    world: SimulationWorld,
    *,
    seed: int,
    max_branch_points: int,
    source_trajectory_path: str,
) -> tuple[list[BroadRegressionBranchPoint], list[dict[str, object]]]:
    points: list[BroadRegressionBranchPoint] = []
    failures: list[dict[str, object]] = []
    record_index = 0
    for tick in range(v142_live_ab.STRICT_TICKS):
        world.tick = tick
        diagnostics_start = len(world.policy_decision_diagnostics_records)
        snapshot = deepcopy(world)
        world._run_tick()
        tick_records = list(world.tick_trajectory_records)
        tick_diagnostics = list(world.policy_decision_diagnostics_records)[
            diagnostics_start:
        ]
        if len(tick_records) != len(tick_diagnostics):
            failures.append(
                {
                    "seed": seed,
                    "tick": tick,
                    "reason": "trajectory_diagnostic_count_mismatch",
                    "record_count": len(tick_records),
                    "diagnostic_count": len(tick_diagnostics),
                }
            )
        for offset, record in enumerate(tick_records):
            diagnostic = (
                tick_diagnostics[offset] if offset < len(tick_diagnostics) else {}
            )
            tv = _mapping(_mapping(diagnostic).get("transition_value_scorer"))
            if (
                len(points) < max_branch_points
                and tv.get("override_applied") is True
            ):
                original = str(tv.get("original_mind_v3_requested_action", ""))
                requested = str(record.get("requested_action", ""))
                action_mask = _bool_mapping(record.get("action_mask"))
                if not original or original not in ACTION_NAMES:
                    failures.append(
                        {
                            "seed": seed,
                            "tick": tick,
                            "record_index": record_index,
                            "reason": "missing_baseline_action",
                        }
                    )
                elif requested not in ACTION_NAMES:
                    failures.append(
                        {
                            "seed": seed,
                            "tick": tick,
                            "record_index": record_index,
                            "reason": "missing_v142_requested_action",
                        }
                    )
                else:
                    branch_id = (
                        f"v143-broad-seed-{seed}-branch-{len(points)}-"
                        f"tick-{tick}-agent-{_int(record.get('agent_id'))}"
                    )
                    branch_snapshot = deepcopy(snapshot)
                    _strip_non_continuation_artifacts(branch_snapshot.policy)
                    points.append(
                        BroadRegressionBranchPoint(
                            branch_id=branch_id,
                            seed=seed,
                            fixture="broad",
                            ticks=v142_live_ab.STRICT_TICKS,
                            branch_tick=tick,
                            record_index=record_index,
                            branch_index=len(points),
                            agent_id=_int(record.get("agent_id")),
                            baseline_action=original,
                            v142_requested_action=requested,
                            v142_resolved_action=str(record.get("resolved_action", "")),
                            action_mask=action_mask,
                            observation_input=dict(
                                _mapping(record.get("observation_input"))
                            ),
                            observation_schema=_optional_string(
                                record.get("observation_schema")
                            ),
                            observation_digest=_optional_string(
                                record.get("observation_digest")
                            ),
                            source_trajectory_path=source_trajectory_path,
                            branch_state_digest=_branch_state_digest(
                                snapshot,
                                branch_id=branch_id,
                                branch_tick=tick,
                            ),
                            world=branch_snapshot,
                        )
                    )
            record_index += 1
        if len(points) >= max_branch_points:
            break
        if not world.alive_agents():
            break
    return points, failures


def _evaluate_branch_point(
    point: BroadRegressionBranchPoint,
    *,
    reference_runs: Mapping[int, Mapping[str, Mapping[str, object]]],
    max_candidate_actions: int,
    verify_replay: bool,
) -> dict[str, object]:
    refs = _mapping(reference_runs.get(point.seed))
    baseline_ref = _mapping(refs.get("baseline"))
    override_ref = _mapping(refs.get("v142_override"))
    actions = _candidate_actions(point, max_candidate_actions=max_candidate_actions)
    action_runs = [
        _execute_action_branch(
            point,
            forced_action=action,
            baseline_ref=baseline_ref,
            override_ref=override_ref,
            verify_replay=verify_replay,
        )
        for action in actions
    ]
    supported = [
        run
        for run in action_runs
        if _intervention_run_supported(run)
    ]
    best_supported = _best_intervention_run(supported)
    best_overall = _best_intervention_run(action_runs)
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "branch_index": point.branch_index,
        "agent_id": point.agent_id,
        "baseline_action": point.baseline_action,
        "v142_requested_action": point.v142_requested_action,
        "v142_resolved_action": point.v142_resolved_action,
        "candidate_actions": actions,
        "candidate_action_count": len(actions),
        "branch_state_digest": point.branch_state_digest,
        "source_trajectory_path": point.source_trajectory_path,
        "public_features": {
            "observation_input": point.observation_input,
            "action_mask": dict(sorted(point.action_mask.items())),
        },
        "action_runs": action_runs,
        "supported_intervention_count": len(supported),
        "intervention_supported": best_supported is not None,
        "intervention_label_action": (
            None if best_supported is None else best_supported.get("forced_action")
        ),
        "best_supported_action_run": _action_run_excerpt(best_supported),
        "best_overall_action_run": _action_run_excerpt(best_overall),
        "diagnostics_only": True,
    }


def _candidate_actions(
    point: BroadRegressionBranchPoint,
    *,
    max_candidate_actions: int,
) -> list[str]:
    ordered: list[str] = []

    def add(action: object) -> None:
        if not isinstance(action, str):
            return
        if action not in ACTION_NAMES:
            return
        if not point.action_mask.get(action, False):
            return
        if action not in ordered:
            ordered.append(action)

    add(point.baseline_action)
    add(point.v142_requested_action)
    for action in ACTION_NAMES:
        add(action)
    if max_candidate_actions > 0:
        return ordered[:max_candidate_actions]
    return ordered


def _execute_action_branch(
    point: BroadRegressionBranchPoint,
    *,
    forced_action: str,
    baseline_ref: Mapping[str, object],
    override_ref: Mapping[str, object],
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_once(
        point,
        forced_action=forced_action,
        baseline_ref=baseline_ref,
        override_ref=override_ref,
    )
    verification = None
    if verify_replay:
        replay, replay_digest = _execute_once(
            point,
            forced_action=forced_action,
            baseline_ref=baseline_ref,
            override_ref=override_ref,
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


def _execute_once(
    point: BroadRegressionBranchPoint,
    *,
    forced_action: str,
    baseline_ref: Mapping[str, object],
    override_ref: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    world = deepcopy(point.world)
    delegate = _normal_mind_v3_delegate(world.policy)
    world.policy = _ForcedFirstActionThenMindV3Policy(
        target_agent_id=point.agent_id,
        forced_action=forced_action,
        delegate=delegate,
    )
    _configure_manual_summary_run(world)
    for tick in range(point.branch_tick, point.ticks):
        world.tick = tick
        world._run_tick()
        if not world.alive_agents():
            break
    forced_used = bool(getattr(world.policy, "used", False))
    run = _summarize_branch_world(
        world,
        point=point,
        forced_action=forced_action,
        forced_used=forced_used,
        baseline_ref=baseline_ref,
        override_ref=override_ref,
    )
    return run, stable_payload_digest(_branch_run_digest_payload(run))


def _normal_mind_v3_delegate(policy: object) -> object:
    delegate = deepcopy(policy)
    _strip_non_continuation_artifacts(delegate)
    return delegate


def _strip_non_continuation_artifacts(policy: object) -> None:
    for field, value in (
        ("_transition_value_scorer", None),
        ("_transition_value_action_override", False),
        ("_transition_value_action_override_source_integrity_passed", False),
        ("_transition_value_states", {}),
        ("_sequence_history_shadow_scorer", None),
        ("_sequence_history_action_override", False),
        ("_sequence_history_action_override_source_integrity_passed", False),
        ("_sequence_history_shadow_states", {}),
    ):
        if hasattr(policy, field):
            setattr(policy, field, value)


def _summarize_branch_world(
    world: SimulationWorld,
    *,
    point: BroadRegressionBranchPoint,
    forced_action: str,
    forced_used: bool,
    baseline_ref: Mapping[str, object],
    override_ref: Mapping[str, object],
) -> dict[str, object]:
    summary = world._build_summary(mode=RunMode.SUMMARY_ONLY)
    action_source_counts = Counter(
        str(record.get("action_source", "unknown"))
        for record in world.trajectory_records
    )
    policy_id_counts = Counter(
        str(record.get("policy_id", "unknown"))
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
    unsupported_resolved_action_count = sum(
        1
        for record in world.trajectory_records
        if isinstance(record.get("resolved_action"), str)
        and record.get("resolution_action_valid") is False
    )
    target = _target_terminal(world, point.agent_id)
    run = {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "agent_id": point.agent_id,
        "baseline_action": point.baseline_action,
        "v142_requested_action": point.v142_requested_action,
        "forced_action": forced_action,
        "forced_action_used": bool(forced_used),
        "forced_action_supported": bool(point.action_mask.get(forced_action, False)),
        "branch_state_digest": point.branch_state_digest,
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
            point=point,
            forced_action=forced_action,
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": evaluate_cli._heuristic_action_source_count(
            action_source_counts
        ),
        "diagnostic_forced_action_source_count": sum(
            count
            for source, count in action_source_counts.items()
            if source.startswith("branch_intervention_force:")
        ),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved_action_count),
        "dominant_requested_action": evaluate_cli._dominant_action_summary(
            requested_action_counts
        )["action"],
        "dominant_requested_action_share": evaluate_cli._dominant_action_summary(
            requested_action_counts
        )["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }
    run["deltas_vs_v142_override"] = _run_deltas(
        run,
        override_ref,
        target_agent_id=point.agent_id,
    )
    run["deltas_vs_baseline"] = _run_deltas(
        run,
        baseline_ref,
        target_agent_id=point.agent_id,
    )
    return run


def _reference_from_live_run_and_trajectory(
    live_run: Mapping[str, object],
    *,
    trajectory_path: Path | None,
    seed: int,
    ticks: int,
    runtime: str,
) -> dict[str, object]:
    records: Sequence[Mapping[str, object]] = ()
    if trajectory_path is not None and trajectory_path.exists():
        records = load_trajectory_jsonl(trajectory_path).records
    return {
        "seed": int(seed),
        "ticks": int(ticks),
        "runtime": runtime,
        "source_trajectory_path": str(trajectory_path) if trajectory_path else None,
        "ticks_executed": _int(live_run.get("ticks_executed")),
        "alive_agents": _int(live_run.get("alive_agents")),
        "births": _int(live_run.get("births")),
        "deaths": _int(live_run.get("deaths")),
        "trajectory_record_count": _int(live_run.get("trajectory_record_count")),
        "heuristic_action_source_count": _int(
            live_run.get("heuristic_action_source_count")
        ),
        "requested_action_counts": dict(
            sorted(_mapping(live_run.get("requested_action_counts")).items())
        ),
        "resolved_action_counts": dict(
            sorted(_mapping(live_run.get("resolved_action_counts")).items())
        ),
        "unsupported_requested_action_count": int(
            _int(live_run.get("unsupported_requested_action_count"))
        ),
        "unsupported_resolved_action_count": int(
            _int(live_run.get("unsupported_resolved_action_count"))
        ),
        "target_terminal_by_agent": _target_terminal_by_agent_from_records(records),
    }


def _run_deltas(
    run: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    target_agent_id: int,
) -> dict[str, object]:
    ref_targets = _mapping(reference.get("target_terminal_by_agent"))
    ref_target = _mapping(ref_targets.get(str(target_agent_id)))
    target = _mapping(run.get("target_terminal"))
    return {
        "alive_agents": _int(run.get("alive_agents")) - _int(reference.get("alive_agents")),
        "births": _int(run.get("births")) - _int(reference.get("births")),
        "deaths": _int(run.get("deaths")) - _int(reference.get("deaths")),
        "target_alive": _bool_int(target.get("alive")) - _bool_int(ref_target.get("alive")),
        "target_energy_ratio": _finite_delta(
            target.get("energy_ratio"),
            ref_target.get("energy_ratio"),
        ),
        "target_hydration_ratio": _finite_delta(
            target.get("hydration_ratio"),
            ref_target.get("hydration_ratio"),
        ),
        "target_health_ratio": _finite_delta(
            target.get("health_ratio"),
            ref_target.get("health_ratio"),
        ),
        "requested_action_counts": _count_delta(
            reference.get("requested_action_counts"),
            run.get("requested_action_counts"),
        ),
        "resolved_action_counts": _count_delta(
            reference.get("resolved_action_counts"),
            run.get("resolved_action_counts"),
        ),
        "unsupported_requested_action_count": _int(
            run.get("unsupported_requested_action_count")
        )
        - _int(reference.get("unsupported_requested_action_count")),
        "unsupported_resolved_action_count": _int(
            run.get("unsupported_resolved_action_count")
        )
        - _int(reference.get("unsupported_resolved_action_count")),
    }


def _intervention_run_supported(run: Mapping[str, object]) -> bool:
    delta = _mapping(run.get("deltas_vs_v142_override"))
    verification = _mapping(run.get("replay_verification"))
    return (
        run.get("forced_action_used") is True
        and run.get("forced_action_supported") is True
        and _int(run.get("unsupported_requested_action_count")) == 0
        and _int(run.get("heuristic_action_source_count")) == 0
        and (not verification or verification.get("verified") is True)
        and (_int(delta.get("alive_agents")) > 0 or _int(delta.get("births")) > 0)
    )


def _best_intervention_run(
    runs: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    if not runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[object, ...]:
        delta = _mapping(run.get("deltas_vs_v142_override"))
        target = _mapping(run.get("target_terminal"))
        return (
            _int(delta.get("alive_agents")),
            _int(delta.get("births")),
            -_int(delta.get("deaths")),
            _int(delta.get("target_alive")),
            _float_or_negative(target.get("energy_ratio")),
            _float_or_negative(target.get("hydration_ratio")),
            _float_or_negative(target.get("health_ratio")),
            -_int(run.get("unsupported_requested_action_count")),
            str(run.get("forced_action", "")),
        )

    return max(runs, key=key)


def _dataset_rows_from_branch_results(
    *,
    branch_results: Sequence[Mapping[str, object]],
    source_report_path: Path,
    scorer_report_path: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in branch_results:
        if result.get("intervention_supported") is not True:
            continue
        best = _mapping(result.get("best_supported_action_run"))
        if not best:
            continue
        point_features = _mapping(result.get("public_features"))
        # Older synthetic unit-test payloads may not include public_features.
        features = point_features if point_features else {}
        row = {
            "schema_version": MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
            "trainable": {
                "feature_policy": "public_observation_input_and_public_action_mask_v1",
                "features": {
                    "observation_input": features.get("observation_input"),
                    "action_mask": features.get("action_mask"),
                },
                "label": {
                    "action": result.get("intervention_label_action"),
                    "label_policy": (
                        "best_supported_terminal_alive_or_birth_improvement_v1"
                    ),
                },
            },
            "metadata": {
                "seed": result.get("seed"),
                "fixture": result.get("fixture"),
                "branch_id": result.get("branch_id"),
                "branch_tick": result.get("branch_tick"),
                "record_index": result.get("record_index"),
                "agent_id": result.get("agent_id"),
                "source_trajectory_path": result.get("source_trajectory_path"),
                "branch_state_digest": result.get("branch_state_digest"),
                "replay_digest": best.get("replay_digest"),
                "v142_scorer_report": str(scorer_report_path),
                "v142_live_report": str(source_report_path),
                "outcome_evidence": {
                    "deltas_vs_v142_override": best.get("deltas_vs_v142_override"),
                    "deltas_vs_baseline": best.get("deltas_vs_baseline"),
                },
                "provenance": {
                    "policy": MIND_V3_V143_BRANCH_INTERVENTION_POLICY,
                    "source_report_digest": _file_sha256(source_report_path),
                    "scorer_report_digest": _file_sha256(scorer_report_path),
                },
            },
        }
        rows.append(row)
    return rows


def _dataset_leakage_scan(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    finite_failures: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        trainable = _mapping(row.get("trainable"))
        flattened = _flatten(trainable)
        for path, value in flattened:
            lower_path = path.lower()
            if any(token in lower_path for token in FORBIDDEN_TRAINABLE_TOKENS):
                failures.append(
                    {
                        "row_index": index,
                        "path": path,
                        "reason": "forbidden_trainable_path_token",
                    }
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    finite_failures.append(
                        {
                            "row_index": index,
                            "path": path,
                            "reason": "non_finite_numeric_value",
                        }
                    )
    return {
        "policy": "v143_trainable_dataset_leakage_scan_v1",
        "passed": not failures and not finite_failures,
        "row_count": len(rows),
        "forbidden_tokens": list(FORBIDDEN_TRAINABLE_TOKENS),
        "forbidden_path_failure_count": len(failures),
        "finite_value_failure_count": len(finite_failures),
        "failures": failures[:24] + finite_failures[:24],
    }


def _acceptance(
    *,
    source_precheck: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
    regression_seeds: Sequence[int],
    dataset_scan: Mapping[str, object],
) -> dict[str, object]:
    action_runs = [
        run
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
    ]
    replay_items = [
        _mapping(run.get("replay_verification"))
        for run in action_runs
        if run.get("replay_verification") is not None
    ]
    label_counts = Counter(
        str(_mapping(_mapping(row.get("trainable")).get("label")).get("action"))
        for row in dataset_rows
    )
    label_counts.pop("None", None)
    dominant = _dominant_count_share(label_counts)
    support_by_seed = {
        int(seed): any(
            _int(result.get("seed")) == int(seed)
            and result.get("intervention_supported") is True
            for result in branch_results
        )
        for seed in regression_seeds
    }
    heuristic_count = sum(_int(run.get("heuristic_action_source_count")) for run in action_runs)
    unsupported_supported_runs = sum(
        1
        for result in branch_results
        if result.get("intervention_supported") is True
        for run in [_mapping(result.get("best_supported_action_run"))]
        if _int(run.get("unsupported_requested_action_count")) != 0
    )
    floors = [
        _floor(
            "source_precheck_passed",
            source_precheck.get("passed") is True,
            observed=source_precheck.get("failures"),
            required=[],
        ),
        _floor(
            "branch_replay_deterministic",
            bool(replay_items)
            and all(item.get("verified") is True for item in replay_items),
            observed=sum(1 for item in replay_items if item.get("verified") is not True),
            required=0,
        ),
        _floor(
            "one_supported_intervention_per_broad_regression_seed",
            bool(support_by_seed) and all(support_by_seed.values()),
            observed=support_by_seed,
            required={int(seed): True for seed in regression_seeds},
        ),
        _floor(
            "oracle_intervention_label_dominant_action_share_lte_0_50",
            _float(dominant.get("share")) <= MAX_DOMINANT_LABEL_ACTION_SHARE,
            observed=dominant.get("share"),
            required=MAX_DOMINANT_LABEL_ACTION_SHARE,
        ),
        _floor(
            "zero_heuristic_action_source_count",
            heuristic_count == 0,
            observed=heuristic_count,
            required=0,
        ),
        _floor(
            "supported_interventions_have_zero_unsupported_actions",
            unsupported_supported_runs == 0,
            observed=unsupported_supported_runs,
            required=0,
        ),
        _floor(
            "dataset_trainable_rows_leakage_safe",
            dataset_scan.get("passed") is True,
            observed=dataset_scan.get("failures"),
            required=[],
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "v143_broad_regression_branch_intervention_acceptance_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "regression_seeds": [int(seed) for seed in regression_seeds],
        "support_by_seed": support_by_seed,
        "supported_seed_count": sum(1 for supported in support_by_seed.values() if supported),
        "dataset_row_count": len(dataset_rows),
        "label_action_counts": dict(sorted(label_counts.items())),
        "dominant_label_action": dominant["key"],
        "dominant_label_action_count": dominant["count"],
        "dominant_label_action_share": dominant["share"],
        "heuristic_action_source_count": heuristic_count,
        "action_run_count": len(action_runs),
        "replay_verification_count": len(replay_items),
        "floors": floors,
    }


def _source_integrity(
    *,
    source_precheck: Mapping[str, object],
    branch_results: Sequence[Mapping[str, object]],
    materialization: Mapping[str, object],
    dataset_scan: Mapping[str, object],
) -> dict[str, object]:
    failures = list(str(item) for item in _list(source_precheck.get("failures")))
    if materialization.get("passed") is not True:
        failures.append("branch_point_materialization_failed")
    action_runs = [
        run
        for result in branch_results
        for run in _list_of_mappings(result.get("action_runs"))
    ]
    replay_items = [
        _mapping(run.get("replay_verification"))
        for run in action_runs
        if run.get("replay_verification") is not None
    ]
    if not replay_items or any(item.get("verified") is not True for item in replay_items):
        failures.append("branch_replay_not_deterministic")
    if dataset_scan.get("passed") is not True:
        failures.append("dataset_trainable_leakage_scan_failed")
    return {
        "policy": "v143_branch_intervention_source_integrity_v1",
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v142_source_precheck_passed": source_precheck.get("passed") is True,
        "materialization_passed": materialization.get("passed") is True,
        "branch_replay_deterministic": bool(replay_items)
        and all(item.get("verified") is True for item in replay_items),
        "replay_verification_count": len(replay_items),
        "dataset_leakage_scan_passed": dataset_scan.get("passed") is True,
    }


def _classification(
    source_integrity: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> str:
    if source_integrity.get("passed") is not True:
        return "broad_regression_branch_intervention_source_integrity_failed"
    if acceptance.get("passed") is True:
        return "broad_regression_branch_intervention_supported_for_v144_training"
    support = _mapping(acceptance.get("support_by_seed"))
    if support and not all(bool(value) for value in support.values()):
        return (
            "broad_regression_branch_intervention_unsupported_"
            "close_transition_world_model_route"
        )
    if (
        acceptance.get("first_failed_floor")
        == "oracle_intervention_label_dominant_action_share_lte_0_50"
    ):
        return "broad_regression_branch_intervention_blocked_label_collapse"
    return "broad_regression_branch_intervention_blocked_non_promotable"


def _branch_point_payload(point: BroadRegressionBranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture,
        "ticks": point.ticks,
        "branch_tick": point.branch_tick,
        "record_index": point.record_index,
        "branch_index": point.branch_index,
        "agent_id": point.agent_id,
        "baseline_action": point.baseline_action,
        "v142_requested_action": point.v142_requested_action,
        "v142_resolved_action": point.v142_resolved_action,
        "valid_action_count": sum(1 for value in point.action_mask.values() if value),
        "action_mask": dict(sorted(point.action_mask.items())),
        "observation_schema": point.observation_schema,
        "observation_digest": point.observation_digest,
        "source_trajectory_path": point.source_trajectory_path,
        "branch_state_digest": point.branch_state_digest,
        "public_feature_digest": stable_payload_digest(
            {
                "observation_input": point.observation_input,
                "action_mask": point.action_mask,
            }
        ),
        "private_world_state_serialized": False,
    }


def _reference_report_section(
    reference_runs: Mapping[int, Mapping[str, Mapping[str, object]]]
) -> dict[str, object]:
    rows = []
    for seed, refs in sorted(reference_runs.items()):
        rows.append(
            {
                "seed": int(seed),
                "baseline": _reference_excerpt(_mapping(refs.get("baseline"))),
                "v142_override": _reference_excerpt(
                    _mapping(refs.get("v142_override"))
                ),
            }
        )
    return {
        "policy": "v143_reference_seed_replay_v1",
        "runs": rows,
    }


def _reference_excerpt(ref: Mapping[str, object]) -> dict[str, object]:
    return {
        "runtime": ref.get("runtime"),
        "source_trajectory_path": ref.get("source_trajectory_path"),
        "alive_agents": ref.get("alive_agents"),
        "births": ref.get("births"),
        "deaths": ref.get("deaths"),
        "trajectory_record_count": ref.get("trajectory_record_count"),
        "heuristic_action_source_count": ref.get("heuristic_action_source_count"),
        "unsupported_requested_action_count": ref.get(
            "unsupported_requested_action_count"
        ),
    }


def _action_run_excerpt(run: Mapping[str, object] | None) -> dict[str, object] | None:
    if run is None:
        return None
    return {
        "forced_action": run.get("forced_action"),
        "forced_action_used": run.get("forced_action_used"),
        "forced_action_supported": run.get("forced_action_supported"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "target_alive_at_end": run.get("target_alive_at_end"),
        "target_energy_ratio_at_end": run.get("target_energy_ratio_at_end"),
        "target_hydration_ratio_at_end": run.get("target_hydration_ratio_at_end"),
        "target_health_ratio_at_end": run.get("target_health_ratio_at_end"),
        "unsupported_requested_action_count": run.get(
            "unsupported_requested_action_count"
        ),
        "heuristic_action_source_count": run.get("heuristic_action_source_count"),
        "deltas_vs_v142_override": run.get("deltas_vs_v142_override"),
        "deltas_vs_baseline": run.get("deltas_vs_baseline"),
        "replay_digest": run.get("replay_digest"),
    }


def attach_public_features_to_branch_results(
    branch_results: Sequence[Mapping[str, object]],
    branch_points: Sequence[BroadRegressionBranchPoint],
) -> list[dict[str, object]]:
    by_id = {point.branch_id: point for point in branch_points}
    rows: list[dict[str, object]] = []
    for result in branch_results:
        row = dict(result)
        point = by_id.get(str(result.get("branch_id")))
        if point is not None:
            row["public_features"] = {
                "observation_input": point.observation_input,
                "action_mask": dict(sorted(point.action_mask.items())),
            }
        rows.append(row)
    return rows


def _target_terminal(world: SimulationWorld, agent_id: int) -> dict[str, object]:
    agent = world.agents.get(int(agent_id))
    alive = bool(agent is not None and agent.alive)
    return {
        "alive": alive,
        "energy_ratio": evaluate_cli._round(world._energy_ratio(agent)) if alive else None,
        "hydration_ratio": evaluate_cli._round(world._hydration_ratio(agent)) if alive else None,
        "health_ratio": evaluate_cli._round(world._health_ratio(agent)) if alive else None,
    }


def _target_terminal_by_agent_from_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    terminal: dict[str, dict[str, object]] = {}
    for record in records:
        agent_id = _int_or_none(record.get("agent_id"))
        if agent_id is None:
            continue
        after = _mapping(record.get("after"))
        alive = after.get("alive")
        terminal[str(agent_id)] = {
            "alive": alive is True,
            "energy_ratio": _finite_or_none(after.get("energy_ratio")),
            "hydration_ratio": _finite_or_none(after.get("hydration_ratio")),
            "health_ratio": _finite_or_none(after.get("health_ratio")),
        }
    return terminal


def _first_target_action_outcome(
    records: Sequence[Mapping[str, object]],
    *,
    point: BroadRegressionBranchPoint,
    forced_action: str,
) -> dict[str, object] | None:
    for record in records:
        if (
            _int(record.get("tick")) == point.branch_tick
            and _int(record.get("agent_id")) == point.agent_id
        ):
            return {
                "requested_action": record.get("requested_action"),
                "resolved_action": record.get("resolved_action"),
                "action_valid": record.get("action_valid"),
                "resolution_action_valid": record.get("resolution_action_valid"),
                "matches_forced_action": record.get("requested_action") == forced_action,
                "outcome": evaluate_cli._json_ready(record.get("outcome", {})),
            }
    return None


def _configure_manual_summary_run(world: SimulationWorld) -> None:
    world.mode = RunMode.SUMMARY_ONLY
    world.record_trajectory = True
    world.retain_trajectory_records = True
    world.trajectory_sink = None


def _live_broad_runs_by_seed(
    live_report: Mapping[str, object]
) -> dict[int, dict[str, Mapping[str, object]]]:
    broad = _mapping(live_report.get("broad"))
    result: dict[int, dict[str, Mapping[str, object]]] = {}
    for key in ("baseline", "override"):
        section = _mapping(broad.get(key))
        for run in _list_of_mappings(section.get("runs")):
            seed = _int(run.get("seed"))
            result.setdefault(seed, {})[key] = run
    return result


def _reference_mismatches(
    *,
    seed: int,
    live_runs: Mapping[int, Mapping[str, Mapping[str, object]]],
    baseline_ref: Mapping[str, object],
    override_ref: Mapping[str, object],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    live_seed = _mapping(live_runs.get(seed))
    for runtime, ref in (("baseline", baseline_ref), ("override", override_ref)):
        live = _mapping(live_seed.get(runtime))
        for field in ("alive_agents", "births", "deaths"):
            if _int(live.get(field)) != _int(ref.get(field)):
                mismatches.append(
                    {
                        "seed": seed,
                        "runtime": runtime,
                        "field": field,
                        "live_report": live.get(field),
                        "replayed": ref.get(field),
                    }
                )
    return mismatches


def _branch_run_digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "record_index": run.get("record_index"),
        "agent_id": run.get("agent_id"),
        "baseline_action": run.get("baseline_action"),
        "v142_requested_action": run.get("v142_requested_action"),
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
        "unsupported_resolved_action_count": run.get(
            "unsupported_resolved_action_count"
        ),
        "action_source_counts": run.get("action_source_counts"),
        "policy_id_counts": run.get("policy_id_counts"),
        "deltas_vs_v142_override": run.get("deltas_vs_v142_override"),
        "deltas_vs_baseline": run.get("deltas_vs_baseline"),
    }


def _floor(
    name: str,
    passed: bool,
    *,
    observed: object,
    required: object,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _canonical_json(payload: object, *, indent: int | None) -> str:
    if indent is None:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _file_sha256(path: str | Path) -> str:
    h = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    return [dict(item) for item in _list(value) if isinstance(item, Mapping)]


def _bool_mapping(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): bool(item)
        for key, item in value.items()
        if str(key) in ACTION_NAMES
    }


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return default
        return parsed if math.isfinite(parsed) else default
    return default


def _float_or_negative(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return -1.0


def _bool_int(value: object) -> int:
    return 1 if value is True else 0


def _finite_delta(value: object, reference: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and math.isfinite(float(value))
        and math.isfinite(float(reference))
    ):
        return evaluate_cli._round(float(value) - float(reference))
    return None


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed):
            return evaluate_cli._round(parsed)
    return None


def _count_delta(before: object, after: object) -> dict[str, int]:
    before_counts = Counter(
        {str(key): _int(value) for key, value in _mapping(before).items()}
    )
    after_counts = Counter(
        {str(key): _int(value) for key, value in _mapping(after).items()}
    )
    keys = sorted(set(before_counts) | set(after_counts))
    return {
        key: int(after_counts.get(key, 0) - before_counts.get(key, 0))
        for key in keys
        if int(after_counts.get(key, 0) - before_counts.get(key, 0)) != 0
    }


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(
        sorted(counts.items()),
        key=lambda item: (int(item[1]), str(item[0])),
    )
    return {
        "key": str(key),
        "count": int(count),
        "share": evaluate_cli._round(int(count) / float(total)),
    }


def _positive_int(value: object, *, field: str) -> int:
    parsed = _int(value, default=-1)
    if parsed <= 0:
        raise BroadRegressionBranchInterventionError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: object, *, field: str) -> int:
    parsed = _int(value, default=-1)
    if parsed < 0:
        raise BroadRegressionBranchInterventionError(f"{field} must be non-negative")
    return parsed


def _flatten(value: object, *, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(value[key], prefix=path))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten(item, prefix=path))
        return rows
    return [(prefix, value)]
