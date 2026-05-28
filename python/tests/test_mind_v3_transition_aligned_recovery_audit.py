from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_transition_aligned_recovery_audit
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import TRAJECTORY_EPISODE_ID_FIELD, TrajectoryJsonlDataset
from evolution_sim.mind.transition_aligned_recovery_audit import (
    ALLOWED_CLASSIFICATION_LABELS,
    MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
    build_transition_aligned_recovery_audit_report,
)


class MindV3TransitionAlignedRecoveryAuditTests(unittest.TestCase):
    def test_transition_aligned_recovery_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:transition-aligned-recovery-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_transition_aligned_recovery_audit"
            ),
        )

    def test_synthetic_trajectory_constructs_first_same_agent_decision_after_gain(
        self,
    ) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(tick=1, agent_id=8, requested_action="stay"),
                _record(tick=2, agent_id=7, requested_action="drink"),
            ]
        )

        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([]),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
        )

        recovery = report["transition_aligned_first_recovery"]
        self.assertEqual(
            recovery["answer"],
            "transition_aligned_first_recovery_constructible",
        )
        self.assertEqual(recovery["constructible_first_recovery_row_count"], 1)
        example = recovery["examples"][0]
        self.assertEqual(example["agent_id"], 7)
        self.assertEqual(example["gain_tick"], 0)
        self.assertEqual(example["recovery_tick"], 2)
        self.assertEqual(example["requested_action"], "drink")

    def test_construction_does_not_cross_agent_or_episode_boundaries(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                    episode_id="episode-a",
                ),
                _record(tick=1, agent_id=8, requested_action="drink", episode_id="episode-a"),
                _record(tick=0, agent_id=7, requested_action="drink", episode_id="episode-b"),
            ]
        )

        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([]),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
        )

        recovery = report["transition_aligned_first_recovery"]
        self.assertEqual(recovery["constructible_first_recovery_row_count"], 0)
        self.assertEqual(recovery["missing_next_same_agent_decision_count"], 1)
        self.assertEqual(
            recovery["answer"],
            "transition_aligned_first_recovery_inconclusive",
        )

    def test_missing_reports_or_missing_next_decision_classify_inconclusive(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                )
            ]
        )

        report = build_transition_aligned_recovery_audit_report(
            trajectory_datasets=(dataset,),
        )

        self.assertEqual(
            report["transition_aligned_first_recovery"]["answer"],
            "transition_aligned_first_recovery_inconclusive",
        )
        self.assertEqual(
            report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )
        self.assertIn("v106_report", report["classification"]["missing_evidence"])

    def test_branch_oracle_overlap_joins_by_exact_state_and_available_digest(
        self,
    ) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(tick=1, agent_id=7, requested_action="drink"),
                _record(
                    tick=2,
                    agent_id=8,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(tick=3, agent_id=8, requested_action="drink"),
                _record(
                    tick=4,
                    agent_id=9,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(
                    tick=5,
                    agent_id=9,
                    requested_action="drink",
                    branch_state_digest="digest-c",
                ),
            ]
        )
        branch_results = [
            _branch_result(seed=13, tick=1, agent_id=7, logged_action="drink"),
            _branch_result(seed=13, tick=3, agent_id=8, logged_action="eat"),
            _branch_result(
                seed=13,
                tick=99,
                agent_id=9,
                logged_action="drink",
                branch_state_digest="digest-c",
            ),
        ]

        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit(branch_results),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
        )

        overlap = report["branch_oracle_overlap"]
        self.assertEqual(overlap["matched_branch_result_count"], 3)
        self.assertEqual(overlap["exact_join_branch_result_count"], 1)
        self.assertGreaterEqual(overlap["state_join_branch_result_count"], 2)
        self.assertEqual(overlap["digest_join_branch_result_count"], 1)
        self.assertEqual(
            overlap["answer"],
            "branch_oracle_first_recovery_overlap_present",
        )

    def test_oracle_recovery_action_legality_counts_resolution_invalid(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(
                    tick=1,
                    agent_id=7,
                    requested_action="move_east",
                    action_mask=_action_mask("move_east", "stay"),
                    resolution_action_mask=_action_mask("stay"),
                    resolution_action_valid=False,
                ),
            ]
        )
        branch = _branch_result(
            seed=13,
            tick=1,
            agent_id=7,
            logged_action="move_east",
            oracle_best_action="move_east",
        )

        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([branch]),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
        )

        legality = report["action_support_legality"]
        self.assertEqual(
            legality["answer"],
            "supported_recovery_actions_observation_legal_resolution_invalid",
        )
        self.assertEqual(
            legality["category_counts"][
                "supported_recovery_actions_observation_legal_resolution_invalid"
            ],
            1,
        )

    def test_movement_drift_public_cause_classifies_occupancy_race(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(
                    tick=1,
                    agent_id=7,
                    requested_action="move_north",
                    action_mask=_action_mask("move_north", "stay"),
                    resolution_action_mask=_action_mask("stay"),
                    resolution_action_valid=False,
                ),
            ]
        )

        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([]),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(dataset,),
        )

        drift = report["mask_drift_root_cause"]
        self.assertEqual(drift["answer"], "movement_mask_drift_occupancy_race")
        self.assertEqual(drift["category_counts"]["movement_mask_drift_occupancy_race"], 1)

    def test_seed29_attribution_from_synthetic_evidence(self) -> None:
        dataset = _trajectory_dataset(
            [
                _record(
                    tick=0,
                    agent_id=7,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(
                    tick=1,
                    agent_id=7,
                    requested_action="move_east",
                    action_mask=_action_mask("move_east", "stay"),
                    resolution_action_mask=_action_mask("stay"),
                    resolution_action_valid=False,
                ),
            ],
            seed=29,
        )

        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([]),
            rollout_context_report=_rollout_context_report(seed29_births=1),
            baseline_report=_baseline_report(seed29_births=3),
            trajectory_datasets=(dataset,),
        )

        self.assertEqual(
            report["seed29_birth_regression"]["answer"],
            "seed29_birth_regression_movement_failure_after_carrion",
        )

    def test_cli_writes_deterministic_json_and_prints_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "fixture-carrion-only-mind-v3-13-120.jsonl"
            output_path = tmp_path / "v107.json"
            v106_path = tmp_path / "v106.json"
            recovery_path = tmp_path / "recovery.json"
            branch_path = tmp_path / "branch.json"
            rollout_path = tmp_path / "rollout.json"
            baseline_path = tmp_path / "baseline.json"
            _write_trajectory(
                trajectory_path,
                _trajectory_dataset(
                    [
                        _record(
                            tick=0,
                            agent_id=7,
                            requested_action="eat",
                            food_source="carcass",
                            resource_gain=0.2,
                        ),
                        _record(tick=1, agent_id=7, requested_action="drink"),
                    ]
                ),
            )
            _write_json(v106_path, _v106_report())
            _write_json(recovery_path, _recovery_audit())
            _write_json(branch_path, _branch_oracle_audit([]))
            _write_json(rollout_path, _rollout_context_report())
            _write_json(baseline_path, _baseline_report())
            stdout = io.StringIO()

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_transition_aligned_recovery_audit",
                        "--v106-report",
                        str(v106_path),
                        "--recovery-action-target-audit",
                        str(recovery_path),
                        "--branch-oracle-audit",
                        str(branch_path),
                        "--rollout-context-report",
                        str(rollout_path),
                        "--baseline-report",
                        str(baseline_path),
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(output_path),
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_v3_transition_aligned_recovery_audit.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))
            first_digest = json.dumps(report, sort_keys=True)

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_transition_aligned_recovery_audit",
                        "--v106-report",
                        str(v106_path),
                        "--recovery-action-target-audit",
                        str(recovery_path),
                        "--branch-oracle-audit",
                        str(branch_path),
                        "--rollout-context-report",
                        str(rollout_path),
                        "--baseline-report",
                        str(baseline_path),
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(output_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_v3_transition_aligned_recovery_audit.main()

            second = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("transition_aligned_recovery_audit=", stdout.getvalue())
        self.assertIn("first_recovery_row_count=1", stdout.getvalue())
        self.assertEqual(
            report["schema_version"],
            MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(first_digest, json.dumps(second, sort_keys=True))

    def test_contract_declares_no_runtime_or_gate_effects(self) -> None:
        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([]),
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            trajectory_datasets=(_trajectory_dataset([]),),
        )

        contract = report["contract"]
        self.assertTrue(contract["diagnostics_only"])
        for key in (
            "runtime_policy_effect",
            "trained_artifact_effect",
            "summary_only_effect",
            "replay_golden_effect",
            "gate_effect",
        ):
            self.assertEqual(contract[key], "none")

    def test_taxonomy_sanity_check(self) -> None:
        report = build_transition_aligned_recovery_audit_report(
            v106_report=_v106_report(),
            recovery_action_target_audit=_recovery_audit(),
            branch_oracle_audit=_branch_oracle_audit([]),
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
                        _record(tick=1, agent_id=7, requested_action="drink"),
                    ]
                ),
            ),
        )

        violations = _taxonomy_violations(report)
        self.assertEqual(violations, [])


def _trajectory_dataset(
    records: list[dict[str, object]],
    *,
    seed: int = 13,
) -> TrajectoryJsonlDataset:
    return TrajectoryJsonlDataset(
        path=Path(f"fixture-carrion-only-mind-v3-{seed}-120.jsonl.gz"),
        header={"config": {"seed": seed}, "format": "test"},
        records=tuple(records),
        footer={
            "summary": {
                "seed": seed,
                "alive_agents": 1,
                "births": 1,
                "deaths": 0,
                "ticks_executed": 120,
            },
            "trajectory_summary": {
                "record_count": len(records),
                "invalid_resolution_action_count": sum(
                    1 for record in records if record.get("resolution_action_valid") is False
                ),
            },
        },
    )


def _record(
    *,
    tick: int,
    agent_id: int,
    requested_action: str,
    action_mask: dict[str, bool] | None = None,
    resolution_action_mask: dict[str, bool] | None = None,
    action_valid: bool | None = None,
    resolution_action_valid: bool | None = None,
    food_source: str | None = None,
    resource_gain: float = 0.0,
    branch_state_digest: str | None = None,
    observation_digest: str = "observation-digest",
    episode_id: str | None = None,
) -> dict[str, object]:
    resolved_mask = action_mask or _action_mask(requested_action, "stay")
    resolution_mask = resolution_action_mask or dict(resolved_mask)
    observed_valid = (
        bool(resolved_mask.get(requested_action, False))
        if action_valid is None
        else action_valid
    )
    resolution_valid = (
        bool(resolution_mask.get(requested_action, False))
        if resolution_action_valid is None
        else resolution_action_valid
    )
    resolved_action = requested_action if resolution_valid else "stay"
    ate = food_source is not None and resource_gain > 0.0
    outcome = {
        "schema_version": "mind_action_outcome_v2",
        "requested_action": requested_action,
        "resolved_action": resolved_action,
        "observation_action_valid": observed_valid,
        "resolution_action_valid": resolution_valid,
        "invalid_reason": None if resolution_valid else "not_in_resolution_action_mask",
        "feeding": {
            "ate": ate,
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
    record: dict[str, object] = {
        "tick": tick,
        "agent_id": agent_id,
        "lineage_id": agent_id,
        "runtime_species_id": None,
        "runtime_ecotype_id": None,
        "observation_schema": "mind_observation_v3",
        "observation_metadata": {},
        "observation_input": {},
        "observation_digest": observation_digest,
        "action_mask": resolved_mask,
        "resolution_action_mask": resolution_mask,
        "requested_action": requested_action,
        "action_source": "mind_v3_autonomous_evolution_policy_v1",
        "policy_id": "mind-v3-test",
        "policy_version": "test",
        "action_valid": observed_valid,
        "resolution_action_valid": resolution_valid,
        "resolved_action": resolved_action,
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
    }
    if branch_state_digest is not None:
        record["branch_state_digest"] = branch_state_digest
    if episode_id is not None:
        record[TRAJECTORY_EPISODE_ID_FIELD] = episode_id
    return record


def _action_mask(*enabled: str) -> dict[str, bool]:
    return {action: action in set(enabled) for action in ACTION_NAMES}


def _branch_result(
    *,
    seed: int,
    tick: int,
    agent_id: int,
    logged_action: str,
    oracle_best_action: str = "drink",
    branch_state_digest: str = "branch-digest",
) -> dict[str, object]:
    return {
        "branch_id": f"branch-{seed}-{tick}-{agent_id}-{logged_action}",
        "seed": seed,
        "branch_tick": tick,
        "agent_id": agent_id,
        "logged_action": logged_action,
        "oracle_best_action": oracle_best_action,
        "branch_state_digest": branch_state_digest,
    }


def _branch_oracle_audit(
    branch_results: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_branch_action_oracle_audit_v1",
        "branch_results": branch_results,
        "aggregate": {"branch_point_count": len(branch_results)},
    }


def _v106_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_post_carrion_rollout_context_coverage_audit_v1",
        "classification": {
            "primary": "post_carrion_context_adequate_but_unhelpful",
            "labels": ["post_carrion_context_adequate_but_unhelpful"],
        },
    }


def _recovery_audit() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_recovery_action_target_alignment_audit_v1",
        "classification": {
            "primary": "first_record_sampling_gap",
            "labels": ["first_record_sampling_gap"],
        },
    }


def _rollout_context_report(*, seed29_births: int = 1) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "holdout_evaluation": {
            "runs": [
                {
                    "seed": 29,
                    "alive_agents": 10,
                    "births": seed29_births,
                    "movement_event_rate": 0.2,
                    "unsupported_resolved_action_count": 3,
                }
            ]
        },
    }


def _baseline_report(*, seed29_births: int = 3) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "holdout_evaluation": {
            "runs": [
                {
                    "seed": 29,
                    "alive_agents": 10,
                    "births": seed29_births,
                    "movement_event_rate": 0.25,
                    "unsupported_resolved_action_count": 1,
                }
            ]
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_trajectory(path: Path, dataset: TrajectoryJsonlDataset) -> None:
    payloads = [
        {"format": "test", "config": {"seed": dataset.footer["summary"]["seed"]}},
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
        "allowed_answers",
        "category_scores",
        "evidence",
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
