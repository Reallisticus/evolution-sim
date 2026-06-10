from __future__ import annotations

import gzip
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.mind.evaluation_harness import (
    _aggregate_runs,
    _fixture_world,
    _neural_anchor_diagnostics,
    _reproduction_failure_attribution,
    _mind_v3_policy,
    _temporal_readiness_attribution,
    _write_trajectory_records,
)
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
    CarrionCounterfactualPolicy,
)
from evolution_sim.mind.evaluation_helpers import (
    dominant_action_summary as _dominant_action_summary,
    heuristic_action_source_count as _heuristic_action_source_count,
    json_ready as _json_ready,
    round_float as _round,
    safe_path_part as _safe_path_part,
)
from evolution_sim.mind.evolution import _validated_metadata
from evolution_sim.mind.horizon_labels import ANIMAL_RESOURCE_FOOD_SOURCES
from evolution_sim.mind.outcome_metrics import (
    aggregate_run_outcome_metrics,
    build_run_outcome_metrics,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION = (
    "mind_v3_carrion_branch_explore_v1"
)
MIND_V3_CARRION_BRANCH_EXPLORE_POLICY = (
    "deterministic_post_contact_branch_explore_v1"
)
MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE = (
    "mind_v3_search_report_candidate_current_policy_v1"
)
DEFAULT_CARRION_BRANCH_BASE_SCRIPT = "hydration_safe_carrion_cycle"
DEFAULT_CARRION_BRANCH_POINTS_PER_SEED = 1


class CarrionBranchExploreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _BranchPoint:
    branch_id: str
    seed: int
    fixture_name: str
    branch_tick: int
    branch_index: int
    base_script: str
    source_policy_kind: str
    source_candidate_id: str | None
    contact: dict[str, object]
    public_observation_bucket: dict[str, object]
    alive_agents_at_branch: int
    births_at_branch: int
    deaths_at_branch: int
    trajectory_record_count_at_branch: int
    branch_state_digest: str
    world: SimulationWorld


def build_carrion_branch_explore_report(
    *,
    seeds: Sequence[int] = DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    ticks: int = DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    fixture_name: str = "carrion_only",
    base_script: str = DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    continuation_scripts: Sequence[str] = DEFAULT_COUNTERFACTUAL_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = 0,
    trajectory_output_dir: str | Path | None = None,
    verify_replay: bool = True,
) -> dict[str, object]:
    if fixture_name != "carrion_only":
        raise CarrionBranchExploreError(
            "carrion branch explore currently supports carrion_only only"
        )
    seed_values = _validated_seeds(seeds)
    tick_count = _positive_int(ticks, field="ticks")
    branch_limit = _positive_int(
        max_branch_points_per_seed,
        field="max_branch_points_per_seed",
    )
    min_tick = _nonnegative_int(min_branch_tick, field="min_branch_tick")
    script_values = _validated_scripts(continuation_scripts)
    _validated_scripts((base_script,))
    output_dir = Path(trajectory_output_dir) if trajectory_output_dir else None

    contract = _branch_contract(
        seeds=seed_values,
        ticks=tick_count,
        fixture_name=fixture_name,
        base_script=base_script,
        continuation_scripts=script_values,
        max_branch_points_per_seed=branch_limit,
        min_branch_tick=min_tick,
        verify_replay=verify_replay,
    )
    branch_points: list[_BranchPoint] = []
    branch_runs: list[dict[str, object]] = []
    discovery_reports: list[dict[str, object]] = []
    for seed in seed_values:
        discovered = _discover_branch_points(
            seed=seed,
            ticks=tick_count,
            fixture_name=fixture_name,
            base_script=base_script,
            max_branch_points=branch_limit,
            min_branch_tick=min_tick,
        )
        branch_points.extend(discovered["branch_points"])
        discovery_reports.append(discovered["report"])
        for branch_point in discovered["branch_points"]:
            for continuation_script in script_values:
                branch_runs.append(
                    _run_branch_continuation(
                        branch_point,
                        continuation_script=continuation_script,
                        ticks=tick_count,
                        trajectory_output_dir=output_dir,
                        verify_replay=verify_replay,
                    )
                )

    aggregate = _aggregate_branch_runs(
        branch_points=branch_points,
        branch_runs=branch_runs,
        seeds=seed_values,
    )
    acceptance = _acceptance(
        aggregate,
        seeds=seed_values,
        replay_verification_required=verify_replay,
    )
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "contract": contract,
        "provenance": {
            "branch_contract_digest": stable_payload_digest(contract),
        },
        "scope": {
            "fixture_name": fixture_name,
            "state_restore_available": True,
            "state_restore_policy": (
                "in_process_deepcopy_of_exact_simulator_state_after_contact_tick_v1"
            ),
            "branch_replay_policy": (
                "deterministic_in_process_copied_branch_state_continuation_"
                "replay_v1"
            ),
            "policy_input_policy": (
                "base and continuation scripts use policy-visible observation "
                "self, local_patch, navigation, and action_mask only"
            ),
        },
        "aggregate": aggregate,
        "acceptance": acceptance,
        "discovery": discovery_reports,
        "branch_points": [_branch_point_payload(point) for point in branch_points],
        "branch_runs": branch_runs,
    }


def build_current_policy_carrion_branch_explore_report(
    *,
    source_search_report_path: str | Path,
    source_candidate_id: str,
    seeds: Sequence[int] = DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    ticks: int = DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    fixture_name: str = "carrion_only",
    continuation_scripts: Sequence[str] = DEFAULT_COUNTERFACTUAL_SCRIPTS,
    max_branch_points_per_seed: int = DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
    min_branch_tick: int = 0,
    trajectory_output_dir: str | Path | None = None,
    verify_replay: bool = True,
) -> dict[str, object]:
    if fixture_name != "carrion_only":
        raise CarrionBranchExploreError(
            "current-policy carrion branch explore currently supports carrion_only only"
        )
    seed_values = _validated_seeds(seeds)
    tick_count = _positive_int(ticks, field="ticks")
    branch_limit = _positive_int(
        max_branch_points_per_seed,
        field="max_branch_points_per_seed",
    )
    min_tick = _nonnegative_int(min_branch_tick, field="min_branch_tick")
    script_values = _validated_scripts(continuation_scripts)
    candidate_id = str(source_candidate_id).strip()
    if not candidate_id:
        raise CarrionBranchExploreError("source_candidate_id must be non-empty")
    output_dir = Path(trajectory_output_dir) if trajectory_output_dir else None

    source_path = Path(source_search_report_path)
    source_report = _load_json_report(source_path)
    candidate = _source_candidate_from_search_report(
        source_report,
        candidate_id=candidate_id,
    )
    source_metadata = _source_candidate_metadata(candidate)
    source_architecture = str(source_metadata.get("architecture", ""))
    source_report_digest = stable_payload_digest(source_report)

    contract = _current_policy_branch_contract(
        source_search_report_path=source_path,
        source_search_report_digest=source_report_digest,
        source_candidate_id=candidate_id,
        source_architecture=source_architecture,
        seeds=seed_values,
        ticks=tick_count,
        fixture_name=fixture_name,
        continuation_scripts=script_values,
        max_branch_points_per_seed=branch_limit,
        min_branch_tick=min_tick,
        verify_replay=verify_replay,
    )
    branch_points: list[_BranchPoint] = []
    branch_runs: list[dict[str, object]] = []
    discovery_reports: list[dict[str, object]] = []
    for seed in seed_values:
        discovered = _discover_current_policy_branch_points(
            seed=seed,
            ticks=tick_count,
            fixture_name=fixture_name,
            source_candidate_id=candidate_id,
            source_candidate_metadata=source_metadata,
            max_branch_points=branch_limit,
            min_branch_tick=min_tick,
        )
        branch_points.extend(discovered["branch_points"])
        discovery_reports.append(discovered["report"])
        for branch_point in discovered["branch_points"]:
            for continuation_script in script_values:
                branch_runs.append(
                    _run_branch_continuation(
                        branch_point,
                        continuation_script=continuation_script,
                        ticks=tick_count,
                        trajectory_output_dir=output_dir,
                        verify_replay=verify_replay,
                    )
                )

    aggregate = _aggregate_branch_runs(
        branch_points=branch_points,
        branch_runs=branch_runs,
        seeds=seed_values,
    )
    source_replay_summary = _source_replay_summary(discovery_reports)
    split_metadata = _train_heldout_split_metadata(
        source_report=source_report,
        seeds=seed_values,
        branch_points=branch_points,
    )
    support = _branch_support_summary(
        branch_points=branch_points,
        branch_runs=branch_runs,
    )
    aggregate["source_replay_action_source_counts"] = source_replay_summary[
        "action_source_counts"
    ]
    aggregate["source_replay_heuristic_action_source_count"] = source_replay_summary[
        "heuristic_action_source_count"
    ]
    aggregate["source_replay_unsupported_requested_action_count"] = (
        source_replay_summary["unsupported_requested_action_count"]
    )
    aggregate["source_replay_unsupported_resolved_action_count"] = (
        source_replay_summary["unsupported_resolved_action_count"]
    )
    aggregate["terminal_survivor_count_by_seed"] = support[
        "terminal_survivor_count_by_seed"
    ]
    aggregate["post_contact_survival_rate_by_seed_and_script"] = support[
        "post_contact_survival_rate_by_seed_and_script"
    ]
    aggregate["unrecoverable_state_summary"] = support[
        "unrecoverable_state_summary"
    ]
    aggregate["trajectory_paths"] = _branch_run_trajectory_paths(branch_runs)
    aggregate["unsupported_requested_action_count"] = sum(
        int(run.get("unsupported_requested_action_count", 0))
        for run in branch_runs
    )
    aggregate["unsupported_resolved_action_count"] = sum(
        int(run.get("unsupported_resolved_action_count", 0))
        for run in branch_runs
    )
    acceptance = _acceptance(
        aggregate,
        seeds=seed_values,
        replay_verification_required=verify_replay,
    )
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "source_mode": MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE,
        "contract": contract,
        "provenance": {
            "branch_contract_digest": stable_payload_digest(contract),
            "source_search_report_digest": source_report_digest,
        },
        "source": {
            "source_mode": MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE,
            "source_search_report_path": str(source_path),
            "source_search_report_digest": source_report_digest,
            "source_candidate_id": candidate_id,
            "source_candidate_architecture": source_architecture,
            "source_candidate_template_mode": "controller_metadata_exact_candidate",
            "source_replay_policy": (
                "autonomous_mind_v3_source_candidate_replay_v1"
            ),
            "source_replay_heuristic_free_required": True,
            "source_replay_action_source_counts": source_replay_summary[
                "action_source_counts"
            ],
            "source_replay_policy_id_counts": source_replay_summary[
                "policy_id_counts"
            ],
            "source_replay_heuristic_action_source_count": (
                source_replay_summary["heuristic_action_source_count"]
            ),
            "source_replay_unsupported_requested_action_count": (
                source_replay_summary["unsupported_requested_action_count"]
            ),
            "source_replay_unsupported_resolved_action_count": (
                source_replay_summary["unsupported_resolved_action_count"]
            ),
        },
        "scope": {
            "fixture_name": fixture_name,
            "state_restore_available": True,
            "state_restore_policy": (
                "in_process_deepcopy_of_exact_simulator_state_after_current_"
                "policy_contact_tick_v1"
            ),
            "branch_replay_policy": (
                "deterministic_in_process_copied_branch_state_continuation_"
                "replay_v1"
            ),
            "source_action_semantics": (
                "source replay is autonomous Mind v3 with no heuristic fallback"
            ),
            "continuation_action_semantics": (
                "continuation scripts are offline diagnostic continuations, "
                "not runtime policy or promotion evidence"
            ),
            "continuation_scripts": list(script_values),
        },
        "aggregate": aggregate,
        "acceptance": acceptance,
        "train_heldout_split_metadata": split_metadata,
        "discovery": discovery_reports,
        "branch_points": [_branch_point_payload(point) for point in branch_points],
        "branch_runs": branch_runs,
    }


def write_carrion_branch_explore_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _discover_branch_points(
    *,
    seed: int,
    ticks: int,
    fixture_name: str,
    base_script: str,
    max_branch_points: int,
    min_branch_tick: int,
) -> dict[str, object]:
    world = _fixture_world(
        fixture_name=fixture_name,
        seed=seed,
        ticks=ticks,
        policy=CarrionCounterfactualPolicy(base_script),
    )
    _configure_manual_summary_run(world)
    branch_points: list[_BranchPoint] = []
    contact_count = 0
    ticks_executed = 0
    for tick in range(ticks):
        world.tick = tick
        world._run_tick()
        ticks_executed = tick + 1
        contacts = _animal_resource_contact_records(world.tick_trajectory_records)
        contact_count += len(contacts)
        if tick >= min_branch_tick:
            for contact in contacts:
                if len(branch_points) >= max_branch_points:
                    break
                snapshot = deepcopy(world)
                branch_index = len(branch_points)
                branch_id = _branch_id(
                    fixture_name=fixture_name,
                    seed=seed,
                    branch_index=branch_index,
                    tick=tick,
                    contact=contact,
                )
                branch_points.append(
                    _BranchPoint(
                        branch_id=branch_id,
                        seed=seed,
                        fixture_name=fixture_name,
                        branch_tick=tick,
                        branch_index=branch_index,
                        base_script=base_script,
                        source_policy_kind="scripted_base_policy",
                        source_candidate_id=None,
                        contact=contact,
                        public_observation_bucket=_public_observation_bucket(
                            contact
                        ),
                        alive_agents_at_branch=len(snapshot.alive_agents()),
                        births_at_branch=int(snapshot.births),
                        deaths_at_branch=int(snapshot.deaths),
                        trajectory_record_count_at_branch=len(
                            snapshot.trajectory_records
                        ),
                        branch_state_digest=_branch_state_digest(
                            snapshot,
                            branch_id=branch_id,
                            branch_tick=tick,
                        ),
                        world=snapshot,
                    )
                )
            if len(branch_points) >= max_branch_points:
                break
        if not world.alive_agents():
            break
    return {
        "branch_points": branch_points,
        "report": {
            "seed": int(seed),
            "fixture": fixture_name,
            "base_script": base_script,
            "ticks_requested": int(ticks),
            "ticks_executed_until_discovery_stop": int(ticks_executed),
            "animal_resource_contact_count_seen": int(contact_count),
            "branch_point_count": len(branch_points),
            "branch_ids": [point.branch_id for point in branch_points],
            "terminal_alive_before_branch_stop": len(world.alive_agents()),
            "births_before_branch_stop": int(world.births),
            "deaths_before_branch_stop": int(world.deaths),
        },
    }


def _discover_current_policy_branch_points(
    *,
    seed: int,
    ticks: int,
    fixture_name: str,
    source_candidate_id: str,
    source_candidate_metadata: Mapping[str, object],
    max_branch_points: int,
    min_branch_tick: int,
) -> dict[str, object]:
    policy = _mind_v3_policy(
        seed=seed,
        founder_template=dict(source_candidate_metadata),
    )
    world = _fixture_world(
        fixture_name=fixture_name,
        seed=seed,
        ticks=ticks,
        policy=policy,
    )
    _configure_manual_summary_run(world)
    branch_points: list[_BranchPoint] = []
    contact_count = 0
    ticks_executed = 0
    source_action_source_counts: Counter[str] = Counter()
    source_policy_id_counts: Counter[str] = Counter()
    unsupported_requested_action_count = 0
    unsupported_resolved_action_count = 0
    for tick in range(ticks):
        world.tick = tick
        world._run_tick()
        ticks_executed = tick + 1
        tick_records = list(world.tick_trajectory_records)
        source_action_source_counts.update(
            str(record.get("action_source", "unknown")) for record in tick_records
        )
        source_policy_id_counts.update(
            str(record.get("policy_id", "unknown")) for record in tick_records
        )
        unsupported_requested_action_count += sum(
            1
            for record in tick_records
            if isinstance(record.get("requested_action"), str)
            and record.get("action_valid") is False
        )
        unsupported_resolved_action_count += sum(
            1
            for record in tick_records
            if isinstance(record.get("resolved_action"), str)
            and record.get("resolution_action_valid") is False
        )
        contacts = _animal_resource_contact_records(tick_records)
        contact_count += len(contacts)
        if tick >= min_branch_tick:
            for contact in contacts:
                if len(branch_points) >= max_branch_points:
                    break
                snapshot = deepcopy(world)
                branch_index = len(branch_points)
                branch_id = _branch_id(
                    fixture_name=fixture_name,
                    seed=seed,
                    branch_index=branch_index,
                    tick=tick,
                    contact=contact,
                )
                branch_points.append(
                    _BranchPoint(
                        branch_id=branch_id,
                        seed=seed,
                        fixture_name=fixture_name,
                        branch_tick=tick,
                        branch_index=branch_index,
                        base_script=f"current_policy:{source_candidate_id}",
                        source_policy_kind="current_mind_v3_candidate_policy",
                        source_candidate_id=source_candidate_id,
                        contact=contact,
                        public_observation_bucket=_public_observation_bucket(
                            contact
                        ),
                        alive_agents_at_branch=len(snapshot.alive_agents()),
                        births_at_branch=int(snapshot.births),
                        deaths_at_branch=int(snapshot.deaths),
                        trajectory_record_count_at_branch=len(
                            snapshot.trajectory_records
                        ),
                        branch_state_digest=_branch_state_digest(
                            snapshot,
                            branch_id=branch_id,
                            branch_tick=tick,
                        ),
                        world=snapshot,
                    )
                )
            if len(branch_points) >= max_branch_points:
                break
        if not world.alive_agents():
            break
    return {
        "branch_points": branch_points,
        "report": {
            "seed": int(seed),
            "fixture": fixture_name,
            "source_mode": MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE,
            "source_candidate_id": source_candidate_id,
            "base_script": None,
            "ticks_requested": int(ticks),
            "ticks_executed_until_discovery_stop": int(ticks_executed),
            "animal_resource_contact_count_seen": int(contact_count),
            "branch_point_count": len(branch_points),
            "branch_ids": [point.branch_id for point in branch_points],
            "branch_state_digests": [
                point.branch_state_digest for point in branch_points
            ],
            "public_observation_buckets": [
                dict(point.public_observation_bucket) for point in branch_points
            ],
            "source_action_source_counts": dict(
                sorted(source_action_source_counts.items())
            ),
            "source_policy_id_counts": dict(sorted(source_policy_id_counts.items())),
            "source_heuristic_action_source_count": _heuristic_action_source_count(
                source_action_source_counts
            ),
            "source_unsupported_requested_action_count": int(
                unsupported_requested_action_count
            ),
            "source_unsupported_resolved_action_count": int(
                unsupported_resolved_action_count
            ),
            "terminal_alive_before_branch_stop": len(world.alive_agents()),
            "births_before_branch_stop": int(world.births),
            "deaths_before_branch_stop": int(world.deaths),
        },
    }


def _run_branch_continuation(
    branch_point: _BranchPoint,
    *,
    continuation_script: str,
    ticks: int,
    trajectory_output_dir: Path | None,
    verify_replay: bool,
) -> dict[str, object]:
    run, digest = _execute_branch_continuation(
        branch_point,
        continuation_script=continuation_script,
        ticks=ticks,
        trajectory_output_dir=trajectory_output_dir,
    )
    verification = None
    if verify_replay:
        replay_run, replay_digest = _execute_branch_continuation(
            branch_point,
            continuation_script=continuation_script,
            ticks=ticks,
            trajectory_output_dir=None,
        )
        verification = {
            "verified": replay_digest == digest,
            "expected_digest": digest,
            "actual_digest": replay_digest,
            "replay_alive_agents": int(replay_run["alive_agents"]),
            "replay_births": int(replay_run["births"]),
            "replay_deaths": int(replay_run["deaths"]),
        }
    run["replay_verification"] = verification
    return run


def _execute_branch_continuation(
    branch_point: _BranchPoint,
    *,
    continuation_script: str,
    ticks: int,
    trajectory_output_dir: Path | None,
) -> tuple[dict[str, object], str]:
    world = deepcopy(branch_point.world)
    world.policy = CarrionCounterfactualPolicy(continuation_script)
    _configure_manual_summary_run(world)
    for tick in range(branch_point.branch_tick + 1, ticks):
        world.tick = tick
        world._run_tick()
        if not world.alive_agents():
            break
    trajectory_path = _branch_trajectory_path(
        trajectory_output_dir,
        branch_id=branch_point.branch_id,
        continuation_script=continuation_script,
        ticks=ticks,
    )
    run = _summarize_branch_world(
        world,
        branch_point=branch_point,
        continuation_script=continuation_script,
        ticks=ticks,
        trajectory_output_path=trajectory_path,
    )
    digest = stable_payload_digest(_replay_digest_payload(run))
    return run, digest


def _summarize_branch_world(
    world: SimulationWorld,
    *,
    branch_point: _BranchPoint,
    continuation_script: str,
    ticks: int,
    trajectory_output_path: Path | None,
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
        str(record["requested_action"])
        for record in world.trajectory_records
        if isinstance(record.get("requested_action"), str)
    )
    resolved_action_counts = Counter(
        str(record["resolved_action"])
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
    dominant_action = _dominant_action_summary(requested_action_counts)
    if trajectory_output_path is not None:
        _write_trajectory_records(
            world=world,
            summary=summary,
            output_path=trajectory_output_path,
            seed=branch_point.seed,
            split_id=(
                "mind_v3_carrion_branch_explore_"
                f"{branch_point.branch_id}_{continuation_script}"
            ),
        )
    heuristic_action_count = _heuristic_action_source_count(action_source_counts)
    run = {
        "branch_id": branch_point.branch_id,
        "seed": branch_point.seed,
        "fixture": branch_point.fixture_name,
        "ticks": int(ticks),
        "branch_tick": branch_point.branch_tick,
        "branch_index": branch_point.branch_index,
        "base_script": branch_point.base_script,
        "continuation_script": continuation_script,
        "branch_state_digest": branch_point.branch_state_digest,
        "contact": dict(branch_point.contact),
        "ticks_executed": int(summary["ticks_executed"]),
        "alive_agents": int(summary["alive_agents"]),
        "births": int(summary["births"]),
        "deaths": int(summary["deaths"]),
        "trophic_role_counts_at_end": _json_ready(
            summary.get("trophic_role_counts_at_end", {})
        ),
        "meat_mode_counts_at_end": _json_ready(
            summary.get("meat_mode_counts_at_end", {})
        ),
        "animal_resource_opportunity_by_meat_mode_end": _json_ready(
            summary.get("animal_resource_opportunity_by_meat_mode_end", {})
        ),
        "diet_by_trophic_role_end": _json_ready(
            summary.get("diet_by_trophic_role_end", {})
        ),
        "diet_by_meat_mode_end": _json_ready(summary.get("diet_by_meat_mode_end", {})),
        "combat_end": _json_ready(summary.get("combat_end", {})),
        "fresh_kill_end": _json_ready(summary.get("fresh_kill_end", {})),
        "carcass_end": _json_ready(summary.get("carcass_end", {})),
        "outcome_metrics": build_run_outcome_metrics(
            summary=summary,
            trajectory_records=world.trajectory_records,
        ),
        "reproduction_failure_attribution": _reproduction_failure_attribution(
            summary
        ),
        "temporal_readiness_attribution": _temporal_readiness_attribution(
            world.trajectory_records
        ),
        "neural_anchor_diagnostics": _neural_anchor_diagnostics(
            world.policy_decision_diagnostics_records
        ),
        "trajectory_record_count": len(world.trajectory_records),
        "heuristic_action_source_count": heuristic_action_count,
        "zero_heuristic_runtime_actions": heuristic_action_count == 0,
        "unique_requested_actions": len(requested_action_counts),
        "requested_action_counts": dict(sorted(requested_action_counts.items())),
        "resolved_action_counts": dict(sorted(resolved_action_counts.items())),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved_action_count),
        "dominant_requested_action": dominant_action["action"],
        "dominant_requested_action_count": dominant_action["count"],
        "dominant_requested_action_share": dominant_action["share"],
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "policy_id_counts": dict(sorted(policy_id_counts.items())),
    }
    if trajectory_output_path is not None:
        run["trajectory_path"] = str(trajectory_output_path)
    return run


def _aggregate_branch_runs(
    *,
    branch_points: Sequence[_BranchPoint],
    branch_runs: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    successful_runs = [
        run
        for run in branch_runs
        if int(run.get("alive_agents", 0)) > 0
        and int(run.get("heuristic_action_source_count", 1)) == 0
    ]
    seeds_with_branch = sorted({int(point.seed) for point in branch_points})
    seeds_with_success = sorted({int(run["seed"]) for run in successful_runs})
    replay_items = [
        run.get("replay_verification")
        for run in branch_runs
        if run.get("replay_verification") is not None
    ]
    replay_verified = bool(replay_items) and all(
        bool(item.get("verified", False))
        for item in replay_items
        if isinstance(item, Mapping)
    )
    unsupported_requested_action_count = sum(
        int(run.get("unsupported_requested_action_count", 0))
        for run in branch_runs
    )
    unsupported_resolved_action_count = sum(
        int(run.get("unsupported_resolved_action_count", 0))
        for run in branch_runs
    )
    support = _branch_support_summary(
        branch_points=branch_points,
        branch_runs=branch_runs,
    )
    return {
        "seed_count": len(seeds),
        "branch_point_count": len(branch_points),
        "branch_run_count": len(branch_runs),
        "seeds_with_branch": seeds_with_branch,
        "seeds_with_branch_count": len(seeds_with_branch),
        "seeds_with_successful_branch": seeds_with_success,
        "positive_seed_count": len(seeds_with_success),
        "successful_branch_run_count": len(successful_runs),
        "terminal_alive_agent_total": sum(
            int(run.get("alive_agents", 0)) for run in successful_runs
        ),
        "max_alive_agents": max(
            (int(run.get("alive_agents", 0)) for run in branch_runs),
            default=0,
        ),
        "max_births": max(
            (int(run.get("births", 0)) for run in branch_runs),
            default=0,
        ),
        "replay_verification_count": len(replay_items),
        "replay_verified": replay_verified,
        "best_branch_run": _best_branch_run(branch_runs),
        "by_continuation_script": _continuation_script_summaries(branch_runs),
        "terminal_survivor_count_by_seed": support[
            "terminal_survivor_count_by_seed"
        ],
        "post_contact_survival_rate_by_seed_and_script": support[
            "post_contact_survival_rate_by_seed_and_script"
        ],
        "unrecoverable_state_summary": support["unrecoverable_state_summary"],
        "trajectory_paths": _branch_run_trajectory_paths(branch_runs),
        "unsupported_requested_action_count": int(
            unsupported_requested_action_count
        ),
        "unsupported_resolved_action_count": int(unsupported_resolved_action_count),
        "combined": _aggregate_runs([dict(run) for run in branch_runs])
        if branch_runs
        else {},
        "outcome_metrics": aggregate_run_outcome_metrics(branch_runs),
    }


def _acceptance(
    aggregate: Mapping[str, object],
    *,
    seeds: Sequence[int],
    replay_verification_required: bool,
) -> dict[str, object]:
    positive_seed_count = int(aggregate.get("positive_seed_count", 0))
    branch_point_count = int(aggregate.get("branch_point_count", 0))
    successful_branch_run_count = int(aggregate.get("successful_branch_run_count", 0))
    replay_verified = bool(aggregate.get("replay_verified", False))
    blockers = []
    if branch_point_count <= 0:
        blockers.append("no_branch_points_found")
    if successful_branch_run_count <= 0:
        blockers.append("no_successful_terminal_survivor_branch")
    if positive_seed_count < len(seeds):
        blockers.append("not_all_target_seeds_have_terminal_survivor_branch")
    if replay_verification_required and not replay_verified:
        blockers.append("branch_replay_not_verified")
    return {
        "diagnostic_acceptance_passed": not blockers,
        "blockers": blockers,
        "requires_positive_terminal_survivor_per_seed": True,
        "requires_zero_heuristic_runtime_actions": True,
        "requires_replay_verification": replay_verification_required,
        "positive_seed_count": positive_seed_count,
        "target_seed_count": len(seeds),
        "successful_branch_run_count": successful_branch_run_count,
    }


def _continuation_script_summaries(
    branch_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in branch_runs:
        grouped.setdefault(str(run.get("continuation_script", "unknown")), []).append(
            dict(run)
        )
    return {
        script: {
            "run_count": len(runs),
            "successful_branch_run_count": sum(
                1
                for run in runs
                if int(run.get("alive_agents", 0)) > 0
                and int(run.get("heuristic_action_source_count", 1)) == 0
            ),
            "aggregate": _aggregate_runs(runs),
            "outcome_metrics": aggregate_run_outcome_metrics(runs),
        }
        for script, runs in sorted(grouped.items())
    }


def _best_branch_run(
    branch_runs: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not branch_runs:
        return None

    def key(run: Mapping[str, object]) -> tuple[int, int, float, str]:
        return (
            int(run.get("alive_agents", 0)),
            int(run.get("births", 0)),
            -float(run.get("dominant_requested_action_share", 1.0)),
            str(run.get("branch_id", "")),
        )

    selected = max(branch_runs, key=key)
    return {
        "branch_id": str(selected.get("branch_id", "")),
        "seed": int(selected.get("seed", 0)),
        "branch_tick": int(selected.get("branch_tick", 0)),
        "base_script": str(selected.get("base_script", "")),
        "continuation_script": str(selected.get("continuation_script", "")),
        "alive_agents": int(selected.get("alive_agents", 0)),
        "births": int(selected.get("births", 0)),
        "dominant_requested_action": selected.get("dominant_requested_action"),
        "dominant_requested_action_share": selected.get(
            "dominant_requested_action_share",
            0.0,
        ),
        "trajectory_path": selected.get("trajectory_path"),
    }


def _animal_resource_contact_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    contacts = []
    for record in records:
        outcome = record.get("outcome")
        outcome_payload = outcome if isinstance(outcome, Mapping) else {}
        feeding = outcome_payload.get("feeding")
        feeding_payload = feeding if isinstance(feeding, Mapping) else {}
        food_source = feeding_payload.get("food_source")
        if food_source not in ANIMAL_RESOURCE_FOOD_SOURCES:
            continue
        if feeding_payload.get("ate") is False:
            continue
        contacts.append(
            {
                "tick": int(record.get("tick", 0)),
                "agent_id": int(record.get("agent_id", 0)),
                "requested_action": str(record.get("requested_action", "")),
                "resolved_action": str(record.get("resolved_action", "")),
                "food_source": str(food_source),
                "gained_energy": _round(
                    _finite_float(feeding_payload.get("gained_energy"))
                ),
                "consumed": _round(_finite_float(feeding_payload.get("consumed"))),
                "before": _state_excerpt(record.get("before")),
                "after": _state_excerpt(record.get("after")),
                "action_mask_availability": _action_mask_availability(
                    record.get("action_mask")
                ),
                "resolution_action_mask_availability": _action_mask_availability(
                    record.get("resolution_action_mask")
                ),
                "public_observation_bucket": _public_observation_bucket_from_record(
                    record
                ),
            }
        )
    return contacts


def _branch_point_payload(point: _BranchPoint) -> dict[str, object]:
    return {
        "branch_id": point.branch_id,
        "seed": point.seed,
        "fixture": point.fixture_name,
        "branch_tick": point.branch_tick,
        "branch_index": point.branch_index,
        "base_script": point.base_script,
        "source_policy_kind": point.source_policy_kind,
        "source_candidate_id": point.source_candidate_id,
        "contact": dict(point.contact),
        "public_observation_bucket": dict(point.public_observation_bucket),
        "alive_agents_at_branch": point.alive_agents_at_branch,
        "births_at_branch": point.births_at_branch,
        "deaths_at_branch": point.deaths_at_branch,
        "trajectory_record_count_at_branch": point.trajectory_record_count_at_branch,
        "branch_state_digest": point.branch_state_digest,
    }


def _load_json_report(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CarrionBranchExploreError(
            f"failed to read source search report {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CarrionBranchExploreError(
            f"source search report {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CarrionBranchExploreError("source search report must contain an object")
    return payload


def _source_candidate_from_search_report(
    report: Mapping[str, object],
    *,
    candidate_id: str,
) -> dict[str, object]:
    candidate_id = str(candidate_id)
    candidate_without_metadata = False
    for candidate in _candidate_search_locations(report):
        if str(candidate.get("candidate_id", "")) != candidate_id:
            continue
        if isinstance(candidate.get("controller_metadata"), Mapping):
            return dict(candidate)
        candidate_without_metadata = True
    if candidate_without_metadata:
        raise CarrionBranchExploreError(
            "source candidate id "
            f"{candidate_id!r} was found but has no controller_metadata"
        )
    raise CarrionBranchExploreError(
        f"source candidate id {candidate_id!r} was not found in search report"
    )


def _candidate_search_locations(
    report: Mapping[str, object],
) -> list[Mapping[str, object]]:
    candidates: list[Mapping[str, object]] = []
    best = report.get("best_candidate")
    if isinstance(best, Mapping):
        candidates.append(best)
    for generation in _list_of_mappings(report.get("generations")):
        candidates.extend(_list_of_mappings(generation.get("candidates")))
        fixture_selection = generation.get("fixture_selection")
        if isinstance(fixture_selection, Mapping):
            candidates.extend(_list_of_mappings(fixture_selection.get("candidates")))
    fixture_rerank = report.get("fixture_rerank")
    if isinstance(fixture_rerank, Mapping):
        candidates.extend(_list_of_mappings(fixture_rerank.get("candidates")))
    return candidates


def _source_candidate_metadata(candidate: Mapping[str, object]) -> dict[str, object]:
    metadata = candidate.get("controller_metadata")
    if not isinstance(metadata, Mapping):
        raise CarrionBranchExploreError(
            "source candidate does not include controller_metadata"
        )
    try:
        return _validated_metadata(dict(metadata))
    except ValueError as exc:
        raise CarrionBranchExploreError(
            f"source candidate controller_metadata is invalid: {exc}"
        ) from exc


def _current_policy_branch_contract(
    *,
    source_search_report_path: Path,
    source_search_report_digest: str,
    source_candidate_id: str,
    source_architecture: str,
    seeds: Sequence[int],
    ticks: int,
    fixture_name: str,
    continuation_scripts: Sequence[str],
    max_branch_points_per_seed: int,
    min_branch_tick: int,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "source_mode": MIND_V3_CARRION_BRANCH_EXPLORE_CURRENT_POLICY_SOURCE,
        "source_search_report_path": str(source_search_report_path),
        "source_search_report_digest": source_search_report_digest,
        "source_candidate_id": source_candidate_id,
        "source_candidate_architecture": source_architecture,
        "fixture_name": fixture_name,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "continuation_scripts": list(continuation_scripts),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "branch_trigger": (
            "current-policy trajectory record at or after min_branch_tick whose "
            "outcome feeding.food_source is carcass or fresh_kill"
        ),
        "branch_timing": "post_tick_after_current_policy_contact_record_v1",
        "source_replay_policy": "autonomous_mind_v3_source_candidate_replay_v1",
        "continuation_policy": (
            "offline_scripted_public_policy_continuation_diagnostic_v1"
        ),
        "continuation_policy_is_runtime_evidence": False,
        "verify_replay": bool(verify_replay),
    }


def _branch_contract(
    *,
    seeds: Sequence[int],
    ticks: int,
    fixture_name: str,
    base_script: str,
    continuation_scripts: Sequence[str],
    max_branch_points_per_seed: int,
    min_branch_tick: int,
    verify_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "fixture_name": fixture_name,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "base_script": base_script,
        "continuation_scripts": list(continuation_scripts),
        "max_branch_points_per_seed": int(max_branch_points_per_seed),
        "min_branch_tick": int(min_branch_tick),
        "branch_trigger": (
            "first trajectory record at or after min_branch_tick whose outcome "
            "feeding.food_source is carcass or fresh_kill"
        ),
        "branch_timing": "post_tick_after_contact_record_v1",
        "verify_replay": bool(verify_replay),
    }


def _branch_id(
    *,
    fixture_name: str,
    seed: int,
    branch_index: int,
    tick: int,
    contact: Mapping[str, object],
) -> str:
    agent_id = int(contact.get("agent_id", 0))
    return (
        f"{_safe_path_part(fixture_name)}-seed-{int(seed)}-"
        f"branch-{int(branch_index)}-tick-{int(tick)}-agent-{agent_id}"
    )


def _branch_state_digest(
    world: SimulationWorld,
    *,
    branch_id: str,
    branch_tick: int,
) -> str:
    alive = [
        {
            "agent_id": int(agent.agent_id),
            "lineage_id": int(agent.lineage_id),
            "x": int(agent.x),
            "y": int(agent.y),
            "age": int(agent.age),
            "energy": _round(float(agent.energy)),
            "hydration": _round(float(agent.hydration)),
            "health": _round(float(agent.health)),
            "alive": bool(agent.alive),
        }
        for agent in sorted(world.alive_agents(), key=lambda item: item.agent_id)
    ]
    carcass_cells = []
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if tile.carcass_deposits:
                carcass_cells.append(
                    {
                        "x": x,
                        "y": y,
                        "energy": _round(
                            sum(
                                float(deposit.energy_remaining)
                                for deposit in tile.carcass_deposits
                            )
                        ),
                        "count": len(tile.carcass_deposits),
                    }
                )
    return stable_payload_digest(
        {
            "branch_id": branch_id,
            "seed": int(world.config.seed),
            "tick": int(branch_tick),
            "births": int(world.births),
            "deaths": int(world.deaths),
            "next_agent_id": int(world.next_agent_id),
            "alive_agents": alive,
            "carcass_cells": carcass_cells,
        }
    )


def _replay_digest_payload(run: Mapping[str, object]) -> dict[str, object]:
    return {
        "branch_id": run.get("branch_id"),
        "seed": run.get("seed"),
        "fixture": run.get("fixture"),
        "ticks": run.get("ticks"),
        "branch_tick": run.get("branch_tick"),
        "base_script": run.get("base_script"),
        "continuation_script": run.get("continuation_script"),
        "branch_state_digest": run.get("branch_state_digest"),
        "ticks_executed": run.get("ticks_executed"),
        "alive_agents": run.get("alive_agents"),
        "births": run.get("births"),
        "deaths": run.get("deaths"),
        "outcome_metrics": run.get("outcome_metrics"),
        "requested_action_counts": run.get("requested_action_counts"),
        "resolved_action_counts": run.get("resolved_action_counts"),
        "action_source_counts": run.get("action_source_counts"),
        "policy_id_counts": run.get("policy_id_counts"),
    }


def _branch_trajectory_path(
    output_dir: Path | None,
    *,
    branch_id: str,
    continuation_script: str,
    ticks: int,
) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / (
        f"branch-{_safe_path_part(branch_id)}-"
        f"{_safe_path_part(continuation_script)}-{int(ticks)}.jsonl.gz"
    )


def _configure_manual_summary_run(world: SimulationWorld) -> None:
    world.record_events = False
    world.record_tick_details = True
    world.record_trajectory = True
    world.retain_trajectory_records = True
    world.trajectory_sink = None


def _state_excerpt(payload: object) -> dict[str, object]:
    state = payload if isinstance(payload, Mapping) else {}
    return {
        "x": int(state.get("x", 0)) if isinstance(state.get("x", 0), int) else 0,
        "y": int(state.get("y", 0)) if isinstance(state.get("y", 0), int) else 0,
        "energy_ratio": _round(_finite_float(state.get("energy_ratio"))),
        "hydration_ratio": _round(_finite_float(state.get("hydration_ratio"))),
        "health_ratio": _round(_finite_float(state.get("health_ratio"))),
        "alive": bool(state.get("alive", False)),
    }


def _source_replay_summary(
    discovery_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_sources: Counter[str] = Counter()
    policy_ids: Counter[str] = Counter()
    unsupported_requested = 0
    unsupported_resolved = 0
    for report in discovery_reports:
        action_sources.update(_int_counter(report.get("source_action_source_counts")))
        policy_ids.update(_int_counter(report.get("source_policy_id_counts")))
        unsupported_requested += int(
            report.get("source_unsupported_requested_action_count", 0)
        )
        unsupported_resolved += int(
            report.get("source_unsupported_resolved_action_count", 0)
        )
    return {
        "action_source_counts": dict(sorted(action_sources.items())),
        "policy_id_counts": dict(sorted(policy_ids.items())),
        "heuristic_action_source_count": _heuristic_action_source_count(
            action_sources
        ),
        "unsupported_requested_action_count": int(unsupported_requested),
        "unsupported_resolved_action_count": int(unsupported_resolved),
    }


def _train_heldout_split_metadata(
    *,
    source_report: Mapping[str, object],
    seeds: Sequence[int],
    branch_points: Sequence[_BranchPoint],
) -> dict[str, object]:
    search = source_report.get("search")
    search_payload = search if isinstance(search, Mapping) else {}
    train_seeds = _int_set(search_payload.get("seeds"))
    holdout_seeds = _int_set(search_payload.get("holdout_seeds"))
    by_seed = {
        str(int(seed)): _seed_split(
            int(seed),
            train_seeds=train_seeds,
            holdout_seeds=holdout_seeds,
        )
        for seed in seeds
    }
    by_branch_state_digest = [
        {
            "branch_state_digest": point.branch_state_digest,
            "branch_id": point.branch_id,
            "seed": int(point.seed),
            "split": by_seed.get(str(int(point.seed)), "unclassified"),
        }
        for point in branch_points
    ]
    return {
        "policy": "source_search_report_seed_split_by_branch_state_digest_v1",
        "train_seeds": sorted(train_seeds),
        "heldout_seeds": sorted(holdout_seeds),
        "fixture_probe_seed_count": len(seeds),
        "by_seed": by_seed,
        "by_branch_state_digest": by_branch_state_digest,
    }


def _branch_support_summary(
    *,
    branch_points: Sequence[_BranchPoint],
    branch_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_seed_script: dict[str, dict[str, dict[str, object]]] = {}
    survivor_count_by_seed: Counter[str] = Counter()
    runs_by_branch: dict[str, list[Mapping[str, object]]] = {}
    for run in branch_runs:
        seed_key = str(int(run.get("seed", 0)))
        script = str(run.get("continuation_script", "unknown"))
        branch_id = str(run.get("branch_id", ""))
        runs_by_branch.setdefault(branch_id, []).append(run)
        script_summary = by_seed_script.setdefault(seed_key, {}).setdefault(
            script,
            {
                "branch_run_count": 0,
                "terminal_survivor_run_count": 0,
                "post_contact_survival_rate": 0.0,
            },
        )
        script_summary["branch_run_count"] = int(
            script_summary["branch_run_count"]
        ) + 1
        if int(run.get("alive_agents", 0)) > 0:
            script_summary["terminal_survivor_run_count"] = int(
                script_summary["terminal_survivor_run_count"]
            ) + 1
            survivor_count_by_seed[seed_key] += 1
    for seed_payload in by_seed_script.values():
        for script_payload in seed_payload.values():
            count = int(script_payload["branch_run_count"])
            survivors = int(script_payload["terminal_survivor_run_count"])
            script_payload["post_contact_survival_rate"] = (
                _round(survivors / count) if count else 0.0
            )
    unrecoverable = []
    for point in branch_points:
        runs = runs_by_branch.get(point.branch_id, [])
        if not any(int(run.get("alive_agents", 0)) > 0 for run in runs):
            unrecoverable.append(
                {
                    "branch_id": point.branch_id,
                    "seed": int(point.seed),
                    "branch_tick": int(point.branch_tick),
                    "branch_state_digest": point.branch_state_digest,
                    "public_observation_bucket": dict(
                        point.public_observation_bucket
                    ),
                }
            )
    by_seed_unrecoverable = Counter(
        str(item["seed"]) for item in unrecoverable
    )
    return {
        "terminal_survivor_count_by_seed": dict(
            sorted(survivor_count_by_seed.items())
        ),
        "post_contact_survival_rate_by_seed_and_script": {
            seed: dict(sorted(script_payload.items()))
            for seed, script_payload in sorted(by_seed_script.items())
        },
        "unrecoverable_state_summary": {
            "unrecoverable_branch_state_count": len(unrecoverable),
            "branch_state_count": len(branch_points),
            "unrecoverable_branch_state_share": (
                _round(len(unrecoverable) / len(branch_points))
                if branch_points
                else 0.0
            ),
            "by_seed": dict(sorted(by_seed_unrecoverable.items())),
            "branch_states": unrecoverable,
        },
    }


def _branch_run_trajectory_paths(
    branch_runs: Sequence[Mapping[str, object]],
) -> list[str]:
    return sorted(
        str(run["trajectory_path"])
        for run in branch_runs
        if isinstance(run.get("trajectory_path"), str)
        and str(run.get("trajectory_path"))
    )


def _public_observation_bucket(contact: Mapping[str, object]) -> dict[str, object]:
    embedded = contact.get("public_observation_bucket")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    after = contact.get("after")
    after_payload = after if isinstance(after, Mapping) else {}
    return {
        "policy": "post_contact_public_observation_bucket_v1",
        "branch_tick": int(contact.get("tick", 0)),
        "branch_tick_bin": _tick_bin(int(contact.get("tick", 0))),
        "energy_bin": _ratio_bin(_finite_float(after_payload.get("energy_ratio"))),
        "hydration_bin": _ratio_bin(
            _finite_float(after_payload.get("hydration_ratio"))
        ),
        "health_bin": _ratio_bin(_finite_float(after_payload.get("health_ratio"))),
        "water_distance": None,
        "water_distance_bin": "unknown",
        "drink_available": False,
        "eat_available": False,
        "movement_available": False,
        "observation_source": "contact_record_after_state_only",
    }


def _public_observation_bucket_from_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    after = record.get("after")
    after_payload = after if isinstance(after, Mapping) else {}
    mask = _action_mask_availability(record.get("action_mask"))
    values = _decoded_observation_values(record.get("observation_input"))
    water_distance = _navigation_feature(values, "water", "distance")
    water_strength = _navigation_feature(values, "water", "strength")
    return {
        "policy": "post_contact_public_observation_bucket_v1",
        "branch_tick": int(record.get("tick", 0)),
        "branch_tick_bin": _tick_bin(int(record.get("tick", 0))),
        "energy_bin": _ratio_bin(_finite_float(after_payload.get("energy_ratio"))),
        "hydration_bin": _ratio_bin(
            _finite_float(after_payload.get("hydration_ratio"))
        ),
        "health_bin": _ratio_bin(_finite_float(after_payload.get("health_ratio"))),
        "water_distance": (
            _round(water_distance) if water_distance is not None else None
        ),
        "water_distance_bin": _distance_bin(water_distance),
        "water_strength": (
            _round(water_strength) if water_strength is not None else None
        ),
        "drink_available": bool(mask["drink_available"]),
        "eat_available": bool(mask["eat_available"]),
        "movement_available": bool(mask["movement_available"]),
        "observation_source": "contact_decision_row_public_observation_input",
    }


def _action_mask_availability(mask: object) -> dict[str, bool]:
    payload = mask if isinstance(mask, Mapping) else {}
    return {
        "eat_available": bool(payload.get("eat", False)),
        "drink_available": bool(payload.get("drink", False)),
        "movement_available": any(
            bool(payload.get(action, False))
            for action in ("move_north", "move_south", "move_east", "move_west")
        ),
    }


def _decoded_observation_values(payload: object) -> list[float]:
    if not isinstance(payload, Mapping):
        return []
    raw_values = payload.get("values")
    if isinstance(raw_values, Sequence) and not isinstance(
        raw_values,
        (str, bytes),
    ):
        values: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return []
            values.append(_finite_float(value))
        return values
    try:
        return decode_observation_input(dict(payload))
    except (TypeError, ValueError):
        return []


def _navigation_feature(
    values: Sequence[float],
    target: str,
    field: str,
) -> float | None:
    if not values:
        return None
    try:
        target_index = NAVIGATION_TARGETS.index(target)
        field_index = NAVIGATION_INPUT_FIELDS.index(field)
    except ValueError:
        return None
    offset = (
        len(SELF_INPUT_FIELDS)
        + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        + target_index * len(NAVIGATION_INPUT_FIELDS)
        + field_index
    )
    if offset >= len(values):
        return None
    return _finite_float(values[offset])


def _ratio_bin(value: float) -> str:
    if value <= 0.2:
        return "critical"
    if value <= 0.4:
        return "low"
    if value <= 0.7:
        return "medium"
    return "high"


def _distance_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 0.2:
        return "close"
    if value <= 0.5:
        return "mid"
    return "far"


def _tick_bin(tick: int) -> str:
    if tick < 40:
        return "early"
    if tick < 80:
        return "mid"
    return "late"


def _int_counter(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            continue
        counter[str(key)] += int(count)
    return counter


def _int_set(value: object) -> set[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    parsed: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        parsed.add(int(item))
    return parsed


def _seed_split(
    seed: int,
    *,
    train_seeds: set[int],
    holdout_seeds: set[int],
) -> str:
    if seed in train_seeds:
        return "source_search_train"
    if seed in holdout_seeds:
        return "source_search_holdout"
    return "fixture_probe_only"


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return 0.0
    return parsed


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise CarrionBranchExploreError("at least one seed is required")
    return values


def _validated_scripts(scripts: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(script) for script in scripts if str(script)))
    if not values:
        raise CarrionBranchExploreError("at least one script is required")
    unsupported = sorted(
        script for script in values if script not in DEFAULT_COUNTERFACTUAL_SCRIPTS
    )
    if unsupported:
        raise CarrionBranchExploreError(
            "unsupported branch script(s): " + ", ".join(unsupported)
        )
    return values


def _positive_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise CarrionBranchExploreError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise CarrionBranchExploreError(f"{field} must be nonnegative")
    return parsed


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
