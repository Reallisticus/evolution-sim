from __future__ import annotations

import base64
import json
import unittest
import zlib

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    _pack_quantized_values,
)
from evolution_sim.mind.broad_branch_residual_constrained_audit import (
    MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.expanded_broad_residual_training import (
    build_expanded_broad_residual_training_report,
)


class MindV3ExpandedBroadResidualTrainingTests(unittest.TestCase):
    def test_expanded_report_accepts_balanced_replay_positive_training_set(self) -> None:
        v99 = _v99_report()
        report, artifact = build_expanded_broad_residual_training_report(
            v99_expanded_oracle_report=v99,
            v100_constrained_report=_v100_report(),
        )

        self.assertTrue(report["v102_expanded_broad_residual_training_accepted"])
        self.assertTrue(report["v103_support_gated_residual_runtime_allowed"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertEqual(report["coverage"]["training_row_count"], 80)
        self.assertEqual(report["coverage"]["source_seed_count"], 10)
        self.assertEqual(report["coverage"]["strict_seed_leak_count"], 0)
        self.assertEqual(report["coverage"]["represented_teacher_mode_count"], 4)
        self.assertEqual(report["coverage"]["movement_reposition_label_share"], 0.25)
        self.assertEqual(report["coverage"]["dominant_teacher_action_share"], 0.5)
        self.assertEqual(report["coverage"]["target_alive_delta_negative_count"], 0)
        self.assertGreater(
            report["leave_one_source_seed_out_evaluation"][
                "target_local_score_delta_mean"
            ],
            0.0,
        )
        self.assertLessEqual(
            report["leave_one_source_seed_out_evaluation"][
                "dominant_predicted_action_share"
            ],
            0.5,
        )
        self.assertTrue(
            report["reload_evaluation"]["artifact_reload_identical_choices"]
        )
        self.assertEqual(artifact["training_row_count"], 80)

    def test_strict_seed_leakage_blocks_v103(self) -> None:
        seeds = (2, 3, 5, 11, 17, 23, 31, 47, 53, 59)
        report, _artifact = build_expanded_broad_residual_training_report(
            v99_expanded_oracle_report=_v99_report(seeds=seeds),
            v100_constrained_report=_v100_report(seeds=seeds),
        )

        self.assertFalse(report["v102_expanded_broad_residual_training_accepted"])
        self.assertFalse(report["v103_support_gated_residual_runtime_allowed"])
        self.assertIn(
            "strict_seed_leakage",
            {blocker["reason"] for blocker in report["acceptance"]["blockers"]},
        )

    def test_loo_collapse_blocks_v103_even_when_teacher_is_balanced(self) -> None:
        v99 = _v99_report()
        report, _artifact = build_expanded_broad_residual_training_report(
            v99_expanded_oracle_report=v99,
            v100_constrained_report=_v100_report(
                action_pattern=("eat", "eat", "eat", "eat", "stay", "stay", "move_east", "drink")
            ),
        )

        self.assertFalse(report["v102_expanded_broad_residual_training_accepted"])
        self.assertIn(
            "insufficient_reposition_label_share",
            {blocker["reason"] for blocker in report["acceptance"]["blockers"]},
        )


def _v99_report(
    *,
    seeds: tuple[int, ...] = (2, 3, 7, 11, 17, 23, 31, 47, 53, 59),
) -> dict[str, object]:
    branch_points = []
    branch_results = []
    for seed_index, seed in enumerate(seeds):
        for slot in range(8):
            index = seed_index * 8 + slot
            branch_id = f"branch-{seed}-{slot}"
            before = {
                "alive": True,
                "energy_ratio": 0.45,
                "hydration_ratio": 0.45,
                "health_ratio": 0.8,
            }
            branch_points.append(
                {
                    "branch_id": branch_id,
                    "seed": seed,
                    "branch_tick": 20 + slot,
                    "agent_id": 100 + index,
                    "logged_action": "move_west",
                    "before": before,
                    "action_mask": _action_mask(),
                    "observation_input": _observation(slot),
                    "public_history_trace": [],
                    "categories": [
                        "movement",
                        "plant_food",
                        "hydration",
                        "recovery",
                        "reproduction_readiness",
                        "pre_death",
                        "animal_resource",
                    ],
                }
            )
            branch_results.append(
                {
                    "branch_id": branch_id,
                    "seed": seed,
                    "branch_tick": 20 + slot,
                    "agent_id": 100 + index,
                    "logged_action": "move_west",
                    "target_local_oracle_action": _teacher_action(slot),
                    "action_runs": [
                        _run("move_west", energy=0.45, hydration=0.45, resource=0.0),
                        _run("eat", energy=0.68 if _teacher_action(slot) == "eat" else 0.50, hydration=0.50, resource=0.4),
                        _run("drink", energy=0.50, hydration=0.68 if _teacher_action(slot) == "drink" else 0.50, resource=0.3),
                        _run("move_east", energy=0.62 if _teacher_action(slot) == "move_east" else 0.50, hydration=0.55, resource=0.1),
                        _run("stay", energy=0.60 if _teacher_action(slot) == "stay" else 0.50, hydration=0.52, resource=0.0),
                    ],
                }
            )
    return {
        "schema_version": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
        "aggregate": {
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "unsupported_candidate_action_count": 0,
        },
        "branch_points": branch_points,
        "branch_results": branch_results,
    }


def _v100_report(
    *,
    seeds: tuple[int, ...] = (2, 3, 7, 11, 17, 23, 31, 47, 53, 59),
    action_pattern: tuple[str, ...] = (
        "eat",
        "eat",
        "eat",
        "eat",
        "move_east",
        "move_east",
        "stay",
        "drink",
    ),
) -> dict[str, object]:
    assignment = []
    for seed_index, seed in enumerate(seeds):
        for slot, action in enumerate(action_pattern):
            index = seed_index * len(action_pattern) + slot
            assignment.append(
                {
                    "branch_id": f"branch-{seed}-{slot}",
                    "seed": seed,
                    "branch_tick": 20 + slot,
                    "agent_id": 100 + index,
                    "logged_action": "move_west",
                    "predicted_action": action,
                    "safe_non_logged_override": True,
                    "target_local_score_delta": 10.0 + float(slot),
                    "terminal_alive_delta": 0.0,
                    "birth_delta": 0.0,
                    "target_alive_delta": 0.0,
                }
            )
    return {
        "schema_version": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
        "v100_broad_branch_residual_constrained_diagnostic_accepted": True,
        "acceptance": {
            "accepted_rules": ["greedy_diversity_constrained_broad_residual_v1"]
        },
        "rule_reports": [
            {
                "rule": "greedy_diversity_constrained_broad_residual_v1",
                "assignment": assignment,
            }
        ],
    }


def _teacher_action(slot: int) -> str:
    return ("eat", "eat", "eat", "eat", "move_east", "move_east", "stay", "drink")[slot]


def _run(action: str, *, energy: float, hydration: float, resource: float) -> dict[str, object]:
    return {
        "forced_action": action,
        "forced_action_supported": True,
        "forced_action_used": True,
        "terminal_alive_agents": 10,
        "births": 0,
        "deaths": 0,
        "dominant_requested_action_share": 0.3,
        "first_action_outcome": {"resource_gain": resource},
        "target_horizon_trace": [
            {
                "horizon_tick_delta": 1,
                "record_found": True,
                "alive_after": True,
                "energy_ratio_after": energy,
                "hydration_ratio_after": hydration,
                "health_ratio_after": 0.8,
                "resource_gain": resource,
            }
        ],
        "population_horizon_trace": [
            {
                "horizon_tick_delta": 1,
                "target_alive": True,
                "target_energy_ratio": energy,
                "target_hydration_ratio": hydration,
                "target_health_ratio": 0.8,
            }
        ],
    }


def _action_mask() -> dict[str, bool]:
    return {
        action: action in {"eat", "drink", "move_east", "move_west", "stay"}
        for action in ACTION_NAMES
    }


def _observation(slot: int) -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[0] = round(0.1 + float(slot) / 20.0, 6)
    values[1] = round(0.9 - float(slot) / 30.0, 6)
    data = base64.b64encode(
        zlib.compress(_pack_quantized_values(values), level=6)
    ).decode("ascii")
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "encoder_version": OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": data,
    }


if __name__ == "__main__":
    unittest.main()
