from __future__ import annotations

import base64
import json
import struct
import unittest
import zlib
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_QUANTIZATION_SCALE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    SELF_INPUT_FIELDS,
)
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
    MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
    SupportGatedResidualError,
    build_action_conditioned_support_gated_residual_runtime_artifact,
    build_branch_replay_feasibility_report,
    build_support_gated_residual_runtime_artifact,
    build_v104_branch_replay_feasibility_report,
    load_support_gated_residual_artifact,
    score_support_gated_residual_artifact,
    validate_support_gated_residual_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy
from evolution_sim.cli.mind_v3_action_conditioned_support_gated_residual_runtime import (
    _v104_live_feasibility_gate,
    build_shadow_failure_audit_from_decisions,
)
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

    def test_v104_action_conditioned_artifact_serializes_thresholds(self) -> None:
        v99 = _v99_report()
        v100 = _v100_report()
        v102_report, v102_artifact = build_expanded_broad_residual_training_report(
            v99_expanded_oracle_report=v99,
            v100_constrained_report=v100,
        )

        artifact, threshold_report = (
            build_action_conditioned_support_gated_residual_runtime_artifact(
                v102_report=v102_report,
                v102_artifact=v102_artifact,
                v99_report=v99,
                v100_report=v100,
            )
        )
        loaded = load_support_gated_residual_artifact(artifact)
        branch_report = build_v104_branch_replay_feasibility_report(
            runtime_artifact=artifact,
            v99_report=v99,
            v100_report=v100,
        )

        self.assertEqual(
            loaded["schema_version"],
            MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(loaded["policy"], MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY)
        self.assertIn("action_thresholds", loaded["support_gate"])
        self.assertNotIn("nearest_support_distance_threshold", loaded["support_gate"])
        self.assertIn("drink", threshold_report["action_thresholds"])
        self.assertIn("v104_branch_replay_feasibility_passed", branch_report)
        self.assertNotIn("v103_branch_replay_feasibility_passed", branch_report)

    def test_v104_scorer_uses_selected_action_threshold(self) -> None:
        artifact = _minimal_v104_runtime_artifact(eat_margin_threshold=10.0)

        scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_observation_input(),
            action_mask=_action_mask(),
            public_history_trace=[],
            linear_action="stay",
        )

        self.assertEqual(scored["selected_action"], "eat")
        self.assertEqual(scored["threshold_scope"], "selected_action")
        self.assertEqual(scored["margin_threshold"], 10.0)
        self.assertTrue(scored["override_proposed"])
        self.assertFalse(scored["override_allowed"])
        self.assertEqual(scored["abstention_reason"], "margin_below_threshold")

    def test_support_gated_residual_scores_are_invariant_to_mind_inheritance_bit(
        self,
    ) -> None:
        artifact = _minimal_runtime_artifact()
        unavailable = _observation_values()
        available = list(unavailable)
        available[SELF_INPUT_FIELDS.index("mind_inheritance_available")] = 1.0

        unavailable_scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_encoded_observation_values(unavailable),
            action_mask=_action_mask(),
            public_history_trace=[],
            linear_action="stay",
        )
        available_scored = score_support_gated_residual_artifact(
            artifact=artifact,
            observation_input=_encoded_observation_values(available),
            action_mask=_action_mask(),
            public_history_trace=[],
            linear_action="stay",
        )

        self.assertEqual(unavailable_scored, available_scored)

    def test_support_candidate_feature_vector_rejects_nonnumeric_values(
        self,
    ) -> None:
        malformed_values = (0.0,) * (OBSERVATION_INPUT_VECTOR_SIZE - 1) + ("bad",)

        vector = candidate_feature_vector(
            {
                "policy_observation_values": malformed_values,
                "action_mask": _action_mask(),
                "public_history_trace": [],
            },
            "eat",
        )

        self.assertEqual(vector, ())

    def test_v104_shadow_failure_audit_reports_drink_concentration(self) -> None:
        vector = _candidate_vector_for(action="drink", action_mask=_drink_action_mask())
        decisions = [
            {
                "seed": 5,
                "tick": 10,
                "agent_id": 1,
                "linear_action": "eat",
                "proposed_action": "drink",
                "override_proposed": True,
                "override_allowed": True,
                "nearest_support_distance": 0.25,
                "score_margin": 1.2,
                "selected_score": 2.0,
                "candidate_scores_top": [{"action": "drink", "score": 2.0}],
                "policy_visible_features": {"self_hydration_ratio": 0.25},
                "feature_vector": vector,
            },
            {
                "seed": 5,
                "tick": 11,
                "agent_id": 1,
                "linear_action": "eat",
                "proposed_action": "drink",
                "override_proposed": True,
                "override_allowed": True,
                "nearest_support_distance": 0.5,
                "score_margin": 1.1,
                "selected_score": 1.8,
                "candidate_scores_top": [{"action": "drink", "score": 1.8}],
                "policy_visible_features": {"self_hydration_ratio": 0.2},
                "feature_vector": vector,
            },
            {
                "seed": 13,
                "tick": 12,
                "agent_id": 2,
                "linear_action": "stay",
                "proposed_action": "move_north",
                "override_proposed": True,
                "override_allowed": True,
                "nearest_support_distance": 0.75,
                "score_margin": 0.9,
                "selected_score": 1.1,
                "candidate_scores_top": [{"action": "move_north", "score": 1.1}],
                "policy_visible_features": {"self_hydration_ratio": 0.6},
                "feature_vector": _candidate_vector_for(
                    action="move_north",
                    action_mask=_drink_action_mask(),
                ),
            },
        ]
        v102_artifact = {
            "support_examples": [
                {
                    "example_index": 4,
                    "action": "drink",
                    "mode": "recover_hydration",
                    "feature_vector": vector,
                    "weight": 1.0,
                }
            ]
        }

        report = build_shadow_failure_audit_from_decisions(
            decisions=decisions,
            v102_artifact=v102_artifact,
            seeds=[5, 13],
            ticks=120,
        )

        self.assertEqual(
            report["schema_version"],
            "mind_v3_v104_shadow_failure_audit_v1",
        )
        self.assertEqual(
            report["gate_accepted_override_action_counts"]["drink"],
            2,
        )
        self.assertEqual(
            report["per_strict_seed_gate_accepted_action_counts"]["5"]["drink"],
            2,
        )
        self.assertEqual(report["accepted_drink_repetition"]["max_seed_agent"], "5:1")
        self.assertEqual(
            report["strict_accepted_drink_vs_v102_support"]["v102_drink_support_count"],
            1,
        )
        self.assertEqual(report["top_accepted_drink_examples"][0]["linear_action"], "eat")

    def test_v104_live_gate_blocks_dominant_override_share(self) -> None:
        gate = _v104_live_feasibility_gate(
            linear_runs=[{"seed": 2, "alive_agents": 10, "births": 1}],
            residual_runs=[
                {
                    "seed": 2,
                    "alive_agents": 10,
                    "births": 1,
                    "unsupported_requested_action_count": 0,
                    "unsupported_resolved_action_count": 0,
                    "support_residual_diagnostics": {
                        "unsupported_proposed_action_count": 0,
                        "dominant_applied_override_action": "drink",
                        "dominant_applied_override_action_share": 0.75,
                        "applied_override_action_counts": {"drink": 3, "eat": 1},
                    },
                }
            ],
            residual_aggregate={
                "unsupported_requested_action_count": 0,
                "unsupported_resolved_action_count": 0,
                "requested_action_counts": {"drink": 3, "eat": 3},
                "support_residual_diagnostics": {
                    "unsupported_proposed_action_count": 0,
                    "dominant_applied_override_action": "drink",
                    "dominant_applied_override_action_share": 0.75,
                },
            },
            delta={"alive_agents_mean": 0.0, "births_mean": 0.0},
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "dominant_applied_override_action_share_above_cap",
            {blocker["reason"] for blocker in gate["blockers"]},
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


def _minimal_v104_runtime_artifact(
    *,
    eat_margin_threshold: float = 0.0001,
) -> dict[str, object]:
    artifact = _minimal_runtime_artifact()
    artifact["schema_version"] = (
        MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION
    )
    artifact["policy"] = MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY
    artifact["support_gate"] = {
        "policy": "unit_test_action_conditioned_thresholds",
        "threshold_source": "unit_test_non_strict_calibration",
        "legal_action_required": True,
        "action_support_required": True,
        "distance_threshold_inclusive": True,
        "margin_threshold_inclusive": True,
        "threshold_scope": "selected_action",
        "action_thresholds": {
            "eat": {
                "nearest_support_distance_threshold": 0.05,
                "residual_score_margin_threshold": eat_margin_threshold,
            },
            "stay": {
                "nearest_support_distance_threshold": 0.05,
                "residual_score_margin_threshold": 0.0001,
            },
        },
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
    return _encoded_observation_values(_observation_values())


def _observation_values() -> list[float]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.4
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.6
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
    return values


def _encoded_observation_values(values: list[float]) -> dict[str, object]:
    quantized = [
        int(round(max(-1.0, min(1.0, value)) * OBSERVATION_QUANTIZATION_SCALE))
        for value in values
    ]
    packed = struct.pack(f"<{len(quantized)}h", *quantized)
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "encoder_version": OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": base64.b64encode(zlib.compress(packed, level=6)).decode("ascii"),
    }


def _action_mask() -> dict[str, bool]:
    return {action: action in {"eat", "stay"} for action in ACTION_NAMES}


def _drink_action_mask() -> dict[str, bool]:
    return {action: action in {"drink", "eat", "stay", "move_north"} for action in ACTION_NAMES}


def _candidate_vector_for(
    *,
    action: str,
    action_mask: dict[str, bool],
) -> list[float]:
    row = planner_distilled_runtime_row(
        observation_input=_observation_input(),
        action_mask=action_mask,
        public_history_trace=[],
    )
    return list(candidate_feature_vector(row, action))


if __name__ == "__main__":
    unittest.main()
