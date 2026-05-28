from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from random import Random

from evolution_sim.cli.mind_v3_evaluate import _fixture_world
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.evolution import founder_mind_v3_metadata
from evolution_sim.mind.first_recovery_branch_oracle_audit import (
    ALLOWED_CLASSIFICATION_LABELS,
    FIXTURE_SOURCE_KIND,
    OPEN_SOURCE_KIND,
    MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION,
    _ForcedFirstActionThenDelegatePolicy,
    _action_support_legality,
    _best_oracle_run,
    _materialize_target_group,
    _outcome_oracle,
    _resolution_drift_root_cause,
    _seed29_attribution,
    _source_kind,
    _target_matches_record,
    build_first_recovery_branch_oracle_audit_report,
    write_first_recovery_branch_oracle_audit_report,
)
from evolution_sim.mind.carrion_branch_explore import _configure_manual_summary_run
from evolution_sim.mind.dataset import TRAJECTORY_EPISODE_ID_FIELD, TrajectoryJsonlDataset
from evolution_sim.mind.transition_aligned_recovery_audit import (
    reconstruct_transition_aligned_first_recovery_rows,
)


class MindV3FirstRecoveryBranchOracleAuditTests(unittest.TestCase):
    def test_first_recovery_branch_oracle_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        self.assertEqual(
            package["scripts"]["sim:mind:v3:first-recovery-branch-oracle-audit"],
            "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
            "evolution_sim.cli.mind_v3_first_recovery_branch_oracle_audit",
        )

    def test_v107_row_reconstruction_matches_expected_count(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(tick=1, agent_id=7, requested_action="move_east"),
            ]
        )
        reconstructed = reconstruct_transition_aligned_first_recovery_rows(
            trajectory_datasets=(dataset,)
        )
        self.assertEqual(
            reconstructed["section"]["constructible_first_recovery_row_count"],  # type: ignore[index]
            1,
        )
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report=_v107_report(expected_count=1),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
            max_targets_per_seed_source=0,
        )
        self.assertTrue(report["v107_row_alignment"]["row_count_matches_v107"])  # type: ignore[index]
        self.assertEqual(
            report["v107_row_alignment"]["reconstructed_first_recovery_row_count"],  # type: ignore[index]
            1,
        )

    def test_row_count_mismatch_is_inconclusive(self) -> None:
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report=_v107_report(expected_count=2),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(
                _trajectory_dataset(
                    [
                        _record(
                            tick=0,
                            agent_id=7,
                            requested_action="eat",
                            food_source="carcass",
                            resource_gain=0.2,
                        ),
                        _record(tick=1, agent_id=7, requested_action="stay"),
                    ]
                ),
            ),
        )
        self.assertFalse(report["v107_row_alignment"]["row_count_matches_v107"])  # type: ignore[index]
        self.assertEqual(report["classification"]["primary"], "missing_evidence_inconclusive")  # type: ignore[index]

    def test_branch_target_matching_rejects_wrong_tick_agent_action_or_digest(self) -> None:
        row = _target_row(
            tick=3,
            record_index=4,
            agent_id=9,
            action="eat",
            observation_digest="expected-digest",
        )
        matched, mismatches = _target_matches_record(
            row,
            _record(
                tick=4,
                agent_id=10,
                requested_action="drink",
                observation_digest="actual-digest",
            ),
            record_index=5,
            source_path=str(row["path"]),
            source_kind=FIXTURE_SOURCE_KIND,
        )
        self.assertFalse(matched)
        self.assertEqual(
            {item["field"] for item in mismatches},
            {"tick", "agent_id", "requested_action", "record_index", "observation_digest"},
        )

    def test_fixture_v5_policy_replay_materializes_exact_target_row(self) -> None:
        template = founder_mind_v3_metadata(agent_id=0, rng=Random(13))
        policy = __import__(
            "evolution_sim.mind.v3_policy",
            fromlist=["MindV3EvolutionPolicy"],
        ).MindV3EvolutionPolicy(seed=13, founder_template_metadata=template)
        world = _fixture_world(
            fixture_name="carrion_only",
            seed=13,
            ticks=1,
            policy=policy,
        )
        _configure_manual_summary_run(world)
        world.tick = 0
        world._run_tick()
        record = dict(world.tick_trajectory_records[0])
        row = _target_row(
            tick=int(record["tick"]),
            record_index=0,
            agent_id=int(record["agent_id"]),
            action=str(record["requested_action"]),
            observation_digest=str(record["observation_digest"]),
            action_mask=dict(record["action_mask"]),
            resolution_action_mask=dict(record["resolution_action_mask"]),
        )
        points, failures = _materialize_target_group(
            [row],
            source_path=str(row["path"]),
            source_kind=FIXTURE_SOURCE_KIND,
            seed=13,
            ticks=1,
            founder_template=template,
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].record_index, 0)
        self.assertEqual(points[0].logged_action, record["requested_action"])

    def test_open_world_source_kind_is_skipped_by_default(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(tick=1, agent_id=7, requested_action="stay"),
            ],
            path=Path("open-mind-v3-13-120.jsonl.gz"),
        )
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report=_v107_report(expected_count=1),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
        )
        self.assertEqual(_source_kind(str(dataset.path)), OPEN_SOURCE_KIND)
        self.assertEqual(report["branch_target_selection"]["selected_target_count"], 0)  # type: ignore[index]
        self.assertEqual(
            report["branch_target_selection"]["skip_reason_counts"],  # type: ignore[index]
            {"open_source_skipped_by_default": 1},
        )

    def test_forced_first_action_wrapper_delegates_continuation(self) -> None:
        delegate = _DelegatePolicy()
        policy = _ForcedFirstActionThenDelegatePolicy(
            target_agent_id=3,
            forced_action="eat",
            delegate=delegate,
        )
        observation = {"metadata": {"agent_id": 3}}
        action_mask = {action: action == "eat" for action in ACTION_NAMES}
        first = policy.decide(observation, action_mask)
        second = policy.decide(observation, action_mask)
        self.assertEqual(first.requested_action, "eat")
        self.assertEqual(first.source, "branch_oracle_force:eat")
        self.assertEqual(second.requested_action, "stay")
        policy.observe_transition(
            {
                "agent_id": 3,
                "action_source": "branch_oracle_force:eat",
                "policy_id": policy.policy_id,
                "policy_version": policy.policy_version,
            }
        )
        self.assertEqual(delegate.observed[0]["policy_id"], delegate.policy_id)

    def test_forced_branch_counts_zero_heuristic_except_diagnostic_force(self) -> None:
        oracle = _outcome_oracle(
            [
                _branch_result(
                    oracle_best_action="eat",
                    action_runs=[
                        _action_run(
                            "eat",
                            action_source_counts={
                                "branch_oracle_force:eat": 1,
                                "mind_v3_autonomous_evolution_policy_v1": 4,
                            },
                        )
                    ],
                )
            ]
        )
        self.assertEqual(oracle["heuristic_action_source_count"], 0)
        self.assertTrue(oracle["zero_heuristic_runtime_actions_except_diagnostic_force"])

    def test_oracle_ranking_is_deterministic(self) -> None:
        runs = [
            _action_run("drink", alive=2, births=0, target_score=0.5),
            _action_run("eat", alive=2, births=0, target_score=0.5),
            _action_run("stay", alive=1, births=10, target_score=1.0),
        ]
        self.assertEqual(_best_oracle_run(runs, runs[0])["forced_action"], "eat")  # type: ignore[index]

    def test_action_legality_counts_observation_legal_and_resolution_invalid(self) -> None:
        section = _action_support_legality(
            [
                _branch_result(
                    oracle_best_action="move_east",
                    oracle_run=_action_run(
                        "move_east",
                        first_action_outcome={
                            "observation_legal": True,
                            "resolution_legal": False,
                        },
                    ),
                )
            ]
        )
        self.assertEqual(section["answer"], "oracle_supported_actions_observation_legal")
        self.assertEqual(
            section["resolution_answer"],
            "oracle_supported_actions_resolution_invalid",
        )

    def test_resolution_root_causes_classify_occupancy_and_depleted_resource(self) -> None:
        occupancy = _resolution_drift_root_cause(
            [
                _branch_result(
                    oracle_best_action="move_east",
                    oracle_run=_action_run(
                        "move_east",
                        first_action_outcome={
                            "observation_legal": True,
                            "resolution_legal": False,
                        },
                    ),
                )
            ]
        )
        depleted = _resolution_drift_root_cause(
            [
                _branch_result(
                    oracle_best_action="eat",
                    oracle_run=_action_run(
                        "eat",
                        first_action_outcome={
                            "observation_legal": True,
                            "resolution_legal": False,
                        },
                    ),
                )
            ]
        )
        self.assertEqual(occupancy["answer"], "resolution_invalid_occupancy_race")
        self.assertEqual(depleted["answer"], "resolution_invalid_depleted_resource")

    def test_seed29_attribution_from_synthetic_branch_evidence(self) -> None:
        section = _seed29_attribution(
            [
                _branch_result(
                    seed=29,
                    logged_action="eat",
                    oracle_best_action="drink",
                    oracle_run=_action_run(
                        "drink",
                        first_action_outcome={
                            "observation_legal": True,
                            "resolution_legal": True,
                            "drank": True,
                            "hydration_ratio_delta": 0.2,
                        },
                    ),
                )
            ],
            rollout_context_report=_rollout_context_report(seed29_births=1),
            baseline_report=_baseline_report(seed29_births=3),
        )
        self.assertEqual(section["answer"], "seed29_public_attribution_missed_drink")

    def test_missing_evidence_classifies_inconclusive_not_crash(self) -> None:
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report_path=Path("missing-v107.json"),
            rollout_context_report_path=Path("missing-rollout.json"),
            baseline_report_path=Path("missing-baseline.json"),
            trajectory_paths=(),
        )
        self.assertEqual(report["classification"]["primary"], "missing_evidence_inconclusive")  # type: ignore[index]

    def test_cli_writes_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v107 = root / "v107.json"
            rollout = root / "rollout.json"
            baseline = root / "baseline.json"
            trajectory = root / "empty.jsonl"
            out_a = root / "a.json"
            out_b = root / "b.json"
            _write_json(v107, _v107_report(expected_count=0))
            _write_json(rollout, _rollout_context_report())
            _write_json(baseline, _baseline_report())
            _write_trajectory(trajectory, _trajectory_dataset([], path=trajectory))
            base_cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_branch_oracle_audit",
                "--v107-report",
                str(v107),
                "--rollout-context-report",
                str(rollout),
                "--baseline-report",
                str(baseline),
                "--trajectory",
                str(trajectory),
            ]
            first = subprocess.run(
                [*base_cmd, "--output", str(out_a)],
                check=True,
                cwd=Path.cwd(),
                env={**os.environ, "PYTHONPATH": "python", "PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [*base_cmd, "--output", str(out_b)],
                check=True,
                cwd=Path.cwd(),
                env={**os.environ, "PYTHONPATH": "python", "PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
            )
            self.assertIn("first_recovery_branch_oracle_audit=", first.stdout)
            self.assertEqual(json.loads(out_a.read_text()), json.loads(out_b.read_text()))

    def test_contract_declares_no_runtime_artifact_gate_replay_or_summary_effects(self) -> None:
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report=_v107_report(expected_count=0),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(_trajectory_dataset([]),),
        )
        contract = report["contract"]  # type: ignore[index]
        self.assertTrue(contract["diagnostics_only"])  # type: ignore[index]
        for key in (
            "runtime_policy_effect",
            "trained_artifact_effect",
            "gate_effect",
            "replay_golden_effect",
            "summary_only_effect",
        ):
            self.assertEqual(contract[key], "none")  # type: ignore[index]
        self.assertFalse(contract["private_world_state_serialized"])  # type: ignore[index]
        self.assertTrue(report["non_promoted"])  # type: ignore[index]

    def test_taxonomy_sanity_check(self) -> None:
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report=_v107_report(expected_count=1),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(
                _trajectory_dataset(
                    [
                        _record(
                            tick=0,
                            agent_id=7,
                            requested_action="eat",
                            food_source="carcass",
                            resource_gain=0.2,
                        ),
                        _record(tick=1, agent_id=7, requested_action="stay"),
                    ]
                ),
            ),
            max_targets_per_seed_source=0,
        )
        self.assertEqual(_taxonomy_violations(report), [])


class _DelegatePolicy:
    policy_id = "delegate-policy"
    policy_version = "delegate-policy-v1"

    def __init__(self) -> None:
        self.observed: list[dict[str, object]] = []

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        return ActionDecision(
            requested_action="stay",
            source="delegate",
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )

    def observe_transition(self, record: dict[str, object]) -> dict[str, object]:
        self.observed.append(dict(record))
        return {"observed": True}


def _trajectory_dataset(
    records: list[dict[str, object]],
    *,
    seed: int = 13,
    path: Path | None = None,
) -> TrajectoryJsonlDataset:
    resolved_path = path or Path(f"fixture-carrion-only-mind-v3-{seed}-120.jsonl.gz")
    return TrajectoryJsonlDataset(
        path=resolved_path,
        header={"config": {"seed": seed, "max_ticks": 120}, "format": "test"},
        records=tuple(records),
        footer={
            "summary": {
                "seed": seed,
                "alive_agents": 1,
                "births": 1,
                "deaths": 0,
                "ticks_executed": 120,
            }
        },
    )


def _record(
    *,
    tick: int,
    agent_id: int,
    requested_action: str,
    action_mask: dict[str, bool] | None = None,
    resolution_action_mask: dict[str, bool] | None = None,
    food_source: str | None = None,
    resource_gain: float = 0.0,
    observation_digest: str = "observation-digest",
) -> dict[str, object]:
    resolved_mask = action_mask or _action_mask(requested_action, "stay")
    resolution_mask = resolution_action_mask or dict(resolved_mask)
    resolution_valid = bool(resolution_mask.get(requested_action, False))
    outcome = {
        "schema_version": "mind_action_outcome_v2",
        "requested_action": requested_action,
        "resolved_action": requested_action if resolution_valid else "stay",
        "observation_action_valid": bool(resolved_mask.get(requested_action, False)),
        "resolution_action_valid": resolution_valid,
        "invalid_reason": None if resolution_valid else "not_in_resolution_action_mask",
        "feeding": {
            "ate": food_source is not None and resource_gain > 0.0,
            "food_source": food_source,
            "gained_energy": resource_gain,
        },
        "drinking": {"drank": requested_action == "drink" and resolution_valid},
        "movement": {
            "moved": requested_action.startswith("move_") and resolution_valid
        },
        "resource_gain": resource_gain,
        "reproduced": False,
        "reproduction_ready_after": False,
        "died": False,
    }
    return {
        "tick": tick,
        "agent_id": agent_id,
        "observation_schema": "mind_observation_v3",
        "observation_input": {},
        "observation_digest": observation_digest,
        "action_mask": resolved_mask,
        "resolution_action_mask": resolution_mask,
        "requested_action": requested_action,
        "action_source": "mind_v3_autonomous_evolution_policy_v1",
        "policy_id": "mind-v3-test",
        "policy_version": "test",
        "action_valid": bool(resolved_mask.get(requested_action, False)),
        "resolution_action_valid": resolution_valid,
        "resolved_action": requested_action if resolution_valid else "stay",
        "moved": requested_action.startswith("move_") and resolution_valid,
        "before": {
            "x": 3,
            "y": 4,
            "alive": True,
            "energy_ratio": 0.5,
            "hydration_ratio": 0.45,
            "health_ratio": 1.0,
            "age": 10,
        },
        "after": {
            "x": 4 if requested_action == "move_east" and resolution_valid else 3,
            "y": 4,
            "alive": True,
            "energy_ratio": 0.5 + resource_gain,
            "hydration_ratio": 0.9 if requested_action == "drink" and resolution_valid else 0.45,
            "health_ratio": 1.0,
            "age": 11,
        },
        "outcome": outcome,
        "reward": {"schema_version": "mind_reward_v1", "total": 0.0},
        TRAJECTORY_EPISODE_ID_FIELD: "episode-0",
    }


def _target_row(
    *,
    tick: int,
    record_index: int,
    agent_id: int,
    action: str,
    observation_digest: str,
    action_mask: dict[str, bool] | None = None,
    resolution_action_mask: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "path": "fixture-carrion-only-mind-v3-13-120.jsonl.gz",
        "source_kind": FIXTURE_SOURCE_KIND,
        "seed": 13,
        "ticks": 120,
        "episode_id": "episode-0",
        "agent_id": agent_id,
        "gain_record_index": 0,
        "gain_tick": max(0, tick - 1),
        "recovery_record_index": record_index,
        "recovery_tick": tick,
        "requested_action": action,
        "resolved_action": action,
        "action_valid": True,
        "resolution_action_valid": True,
        "observation_digest": observation_digest,
        "action_mask": action_mask or _action_mask(action, "stay"),
        "resolution_action_mask": resolution_action_mask or _action_mask(action, "stay"),
        "before": {"alive": True, "energy_ratio": 0.5, "hydration_ratio": 0.5, "health_ratio": 1.0},
    }


def _action_mask(*enabled: str) -> dict[str, bool]:
    return {action: action in set(enabled) for action in ACTION_NAMES}


def _action_run(
    action: str,
    *,
    alive: int = 1,
    births: int = 0,
    deaths: int = 0,
    target_alive: bool = True,
    target_score: float = 0.5,
    first_action_outcome: dict[str, object] | None = None,
    action_source_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "forced_action": action,
        "forced_action_used": True,
        "forced_action_supported": True,
        "alive_agents": alive,
        "terminal_alive_agents": alive,
        "births": births,
        "deaths": deaths,
        "target_alive_at_end": target_alive,
        "target_recovery_score_at_end": target_score,
        "first_action_outcome": first_action_outcome
        or {"observation_legal": True, "resolution_legal": True},
        "heuristic_action_source_count": sum(
            count
            for source, count in (action_source_counts or {}).items()
            if "heuristic" in source
        ),
        "diagnostic_forced_action_source_count": sum(
            count
            for source, count in (action_source_counts or {}).items()
            if source.startswith("branch_oracle_force:")
        ),
        "action_source_counts": action_source_counts or {},
    }


def _branch_result(
    *,
    seed: int = 13,
    logged_action: str = "eat",
    oracle_best_action: str = "eat",
    oracle_run: dict[str, object] | None = None,
    action_runs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    run = oracle_run or _action_run(oracle_best_action)
    return {
        "branch_id": f"branch-{seed}",
        "seed": seed,
        "source_kind": FIXTURE_SOURCE_KIND,
        "branch_tick": 1,
        "agent_id": 7,
        "logged_action": logged_action,
        "oracle_best_action": oracle_best_action,
        "oracle_best_action_run": run,
        "action_runs": action_runs or [run],
        "oracle_changed_action": oracle_best_action != logged_action,
        "oracle_deltas_vs_logged": {
            "birth_delta": 0,
            "target_recovery_score_delta": 0.0,
        },
        "material_oracle_gain": False,
    }


def _v107_report(*, expected_count: int) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_transition_aligned_recovery_audit_v1",
        "transition_aligned_first_recovery": {
            "constructible_first_recovery_row_count": expected_count
        },
    }


def _rollout_context_report(*, seed29_births: int = 1) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "holdout_evaluation": {"runs": [{"seed": 29, "births": seed29_births}]},
    }


def _baseline_report(*, seed29_births: int = 3) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "holdout_evaluation": {"runs": [{"seed": 29, "births": seed29_births}]},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_trajectory(path: Path, dataset: TrajectoryJsonlDataset) -> None:
    payloads = [
        {"format": "test", "config": {"seed": 13, "max_ticks": 120}},
        *({"record": record} for record in dataset.records),
        dataset.footer,
    ]
    path.write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads),
        encoding="utf-8",
    )


def _taxonomy_violations(report: dict[str, object]) -> list[tuple[str, str]]:
    allowed = set(ALLOWED_CLASSIFICATION_LABELS)
    keys = {
        "primary",
        "labels",
        "answer",
        "replay_answer",
        "distribution_answer",
        "resolution_answer",
        "allowed_answers",
        "category_scores",
        "diagnostic_answers",
    }
    violations: list[tuple[str, str]] = []

    def values_for_check(value: object) -> list[object]:
        if isinstance(value, dict):
            if all(isinstance(item, (int, float)) for item in value.values()):
                return list(value.keys())
            return list(value.values())
        if isinstance(value, list):
            return list(value)
        return [value]

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key in keys:
                    for item in values_for_check(child):
                        if item is None:
                            continue
                        if isinstance(item, str) and item not in allowed:
                            violations.append((child_path, item))
                walk(child, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(report, "$")
    return violations


if __name__ == "__main__":
    unittest.main()
