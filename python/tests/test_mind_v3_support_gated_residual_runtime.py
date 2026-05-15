from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_INPUT_VECTOR_SIZE
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_SCHEMA_VERSION,
    mind_v3_parameter_count,
)
from evolution_sim.mind.expanded_broad_residual_training import (
    build_expanded_broad_residual_training_report,
)
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V103_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
    SupportGatedResidualError,
    build_branch_replay_feasibility_report,
    build_support_gated_residual_runtime_artifact,
    load_support_gated_residual_artifact,
    score_support_gated_residual_artifact,
    validate_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy
from python.tests.test_mind_v3_expanded_broad_residual_training import (
    _v100_report,
    _v99_report,
)


class MindV3SupportGatedResidualRuntimeTests(unittest.TestCase):
    def test_loader_round_trips_serialized_runtime_artifact(self) -> None:
        artifact = _minimal_runtime_artifact()

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            loaded = load_support_gated_residual_artifact(path)

        self.assertEqual(
            loaded["schema_version"],
            MIND_V3_V103_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertTrue(loaded["runtime_ready"])
        self.assertFalse(loaded["promotion_ready"])
        self.assertFalse(loaded["runtime_promotion_allowed"])

    def test_loader_rejects_forbidden_runtime_inputs(self) -> None:
        artifact = _minimal_runtime_artifact()
        artifact["support_examples"][0]["source_seed"] = 7

        with self.assertRaisesRegex(SupportGatedResidualError, "forbidden"):
            validate_support_gated_residual_artifact(artifact)

    def test_scorer_abstains_when_margin_gate_fails(self) -> None:
        artifact = _minimal_runtime_artifact(margin_threshold=10.0)
        scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_observation_input(),
            action_mask=_action_mask(),
            public_history_trace=[],
            linear_action="stay",
        )

        self.assertEqual(scored["selected_action"], "eat")
        self.assertTrue(scored["override_proposed"])
        self.assertFalse(scored["override_allowed"])
        self.assertEqual(scored["abstention_reason"], "margin_below_threshold")

    def test_branch_replay_safety_counts_abstentions_and_blocks_collapse(self) -> None:
        v99 = _v99_report()
        v100 = _v100_report()
        v102_report, v102_artifact = build_expanded_broad_residual_training_report(
            v99_expanded_oracle_report=v99,
            v100_constrained_report=v100,
        )
        runtime_artifact, _thresholds = build_support_gated_residual_runtime_artifact(
            v102_report=v102_report,
            v102_artifact=v102_artifact,
            v99_report=v99,
            v100_report=v100,
        )

        report = build_branch_replay_feasibility_report(
            runtime_artifact=runtime_artifact,
            v99_report=v99,
            v100_report=v100,
        )

        self.assertFalse(report["v103_branch_replay_feasibility_passed"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertGreater(report["summary"]["applied_override_count"], 0)
        self.assertGreater(report["summary"]["abstention_count"], 0)
        self.assertEqual(report["summary"]["target_alive_delta_negative_count"], 0)
        self.assertIn(
            "dominant_applied_override_action_share_above_cap",
            {blocker["reason"] for blocker in report["safety_gate"]["blockers"]},
        )
        for seed_report in report["summary"]["per_source_seed"]:
            self.assertGreaterEqual(
                seed_report["target_local_score_delta_mean"],
                0.0,
            )

    def test_shadow_mode_keeps_linear_action_and_logs_override(self) -> None:
        policy = MindV3EvolutionPolicy(
            seed=7,
            support_residual_artifact=_minimal_runtime_artifact(),
            support_residual_runtime_mode="shadow",
        )
        policy.register_agent_mind(agent_id=3, metadata=_linear_stay_metadata())

        decision = policy.decide(_observation(), _action_mask())

        self.assertEqual(decision.requested_action, "stay")
        self.assertEqual(
            decision.diagnostics["controller_backend"],
            "support_gated_residual_runtime_shadow",
        )
        self.assertEqual(decision.diagnostics["support_residual_proposed_action"], "eat")
        self.assertTrue(decision.diagnostics["support_residual_override_allowed"])
        self.assertTrue(decision.diagnostics["support_residual_shadowed"])
        self.assertFalse(decision.diagnostics["support_residual_override_applied"])

    def test_live_runtime_records_override_before_next_history_update(self) -> None:
        policy = MindV3EvolutionPolicy(
            seed=7,
            support_residual_artifact=_minimal_runtime_artifact(),
            support_residual_runtime_mode="live",
        )
        policy.register_agent_mind(agent_id=3, metadata=_linear_stay_metadata())

        first = policy.decide(_observation(), _action_mask())
        trace = policy.observe_transition(
            {
                "tick": 0,
                "agent_id": 3,
                "policy_id": policy.policy_id,
                "observation_input": _observation_input(),
                "action_mask": _action_mask(),
                "requested_action": first.requested_action,
                "resolved_action": first.requested_action,
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {"x": 1, "y": 1, "energy_ratio": 0.4, "hydration_ratio": 0.6, "health_ratio": 0.9},
                "after": {"x": 1, "y": 1, "energy_ratio": 0.5, "hydration_ratio": 0.6, "health_ratio": 0.9},
                "outcome": {"resource_gain": 0.1, "feeding": {"ate": True}},
                "reward": {"total": 0.5},
            }
        )
        second = policy.decide(_observation(), _action_mask())

        self.assertEqual(first.requested_action, "eat")
        self.assertEqual(trace["action"], "eat")
        self.assertTrue(trace["support_residual_artifact_frozen"])
        self.assertEqual(
            second.diagnostics["support_residual_public_history_steps"],
            1,
        )


def _minimal_runtime_artifact(
    *,
    distance_threshold: float = 0.05,
    margin_threshold: float = 0.0001,
) -> dict[str, object]:
    row = planner_distilled_runtime_row(
        observation_input=_observation_input(),
        action_mask=_action_mask(),
        public_history_trace=[],
    )
    eat_vector = list(candidate_feature_vector(row, "eat"))
    stay_vector = list(candidate_feature_vector(row, "stay"))
    artifact = {
        "schema_version": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        "policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": 0.0,
        "support_gate": {
            "policy": "unit_test_thresholds",
            "threshold_source": "unit_test",
            "legal_action_required": True,
            "action_support_required": True,
            "distance_threshold_inclusive": True,
            "margin_threshold_inclusive": True,
            "nearest_support_distance_threshold": distance_threshold,
            "residual_score_margin_threshold": margin_threshold,
        },
        "inference_contract": {
            "one_row_one_agent_local_decision": True,
            "requires_action_mask": True,
            "requires_policy_visible_features_only": True,
            "requires_linear_default_action": True,
            "requires_planner_outcome_tables": False,
            "requires_global_batch_assignment": False,
            "uses_heuristic_fallback": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_fixture_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_private_simulator_state": False,
        },
        "training_row_count": 2,
        "support_action_counts": {"eat": 1, "stay": 1},
        "teacher_action_counts": {"eat": 1, "stay": 1},
        "support_examples": [
            {
                "example_index": 0,
                "action": "eat",
                "mode": "exploit_resource",
                "feature_vector": eat_vector,
                "weight": 20.0,
            },
            {
                "example_index": 1,
                "action": "stay",
                "mode": "conserve",
                "feature_vector": stay_vector,
                "weight": 1.0,
            },
        ],
    }
    validate_support_gated_residual_artifact(deepcopy(artifact))
    return artifact


def _linear_stay_metadata() -> dict[str, object]:
    weights = {action: [0.0] * 8 for action in ACTION_NAMES}
    bias = {action: 0.0 for action in ACTION_NAMES}
    bias["stay"] = 1.0
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "state_size": mind_v3_parameter_count(
            architecture="homeostatic_feature_projection_linear_action_head_v2"
        ),
        "architecture": "homeostatic_feature_projection_linear_action_head_v2",
        "action_head_weights": weights,
        "action_head_bias": bias,
    }


def _observation() -> dict[str, object]:
    return {
        "metadata": {"agent_id": 3},
        "self": {"trophic_role": "herbivore", "meat_mode": "none"},
        "observation_input": _observation_input(),
    }


def _observation_input() -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[0] = 0.4
    values[1] = 0.6
    values[2] = 0.9
    return {"values": values}


def _action_mask() -> dict[str, bool]:
    return {action: action in {"eat", "stay"} for action in ACTION_NAMES}


if __name__ == "__main__":
    unittest.main()
