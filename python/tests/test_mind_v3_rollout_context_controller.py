from __future__ import annotations

import json
import unittest
from random import Random
from tempfile import TemporaryDirectory
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    SELF_INPUT_FIELDS,
)
from evolution_sim.cli import mind_v3_evolve as evolve_cli
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_ARCHITECTURE,
    MIND_V3_CONTROLLER_SCHEMA_VERSION,
    MIND_V3_HIDDEN_UNITS,
    MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
    MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
    founder_mind_v3_metadata,
    inherit_mind_v3_metadata,
    load_mind_v3_controller_metadata,
    load_mind_v3_founder_template,
    mind_v3_controller_architecture_status,
    mind_v3_parameter_count,
    require_mind_v3_founder_template_promotion_eligible,
    score_mind_v3_metadata,
    adapt_mind_v3_metadata,
)
from evolution_sim.mind.contracts import mind_v3_autonomous_evolution_contract
from evolution_sim.mind.rollout_context import (
    MIND_V3_ROLLOUT_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION,
    RolloutContextState,
    rollout_context_vector_fields,
    rollout_context_vector_size,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


class MindV3RolloutContextControllerTests(unittest.TestCase):
    def test_v5_founder_context_unit_weights_are_exactly_zero(self) -> None:
        profile = "forager"
        v4 = founder_mind_v3_metadata(
            agent_id=7,
            rng=Random(123),
            specialization_profile=profile,
        )
        v5 = founder_mind_v3_metadata(
            agent_id=7,
            rng=Random(123),
            specialization_profile=profile,
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )

        self.assertEqual(
            v5["founder_prior_policy"],
            "diverse_need_gated_navigation_action_prior_v3_plus_zero_initialized_rollout_context_units_v1",
        )
        self.assertEqual(v5["action_head_bias"], v4["action_head_bias"])
        for action in ACTION_NAMES:
            self.assertEqual(
                v5["action_head_weights"][action][:MIND_V3_HIDDEN_UNITS],
                v4["action_head_weights"][action],
            )
            self.assertEqual(
                v5["action_head_weights"][action][MIND_V3_HIDDEN_UNITS:],
                [0.0] * rollout_context_vector_size(),
            )

    def test_v5_metadata_round_trips_through_save_load_validation(self) -> None:
        metadata = founder_mind_v3_metadata(
            agent_id=7,
            rng=Random(123),
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )

        self.assertEqual(
            metadata["architecture"],
            MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        self.assertEqual(
            metadata["state_size"],
            mind_v3_parameter_count(
                architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE
            ),
        )
        self.assertEqual(
            len(metadata["action_head_weights"]["eat"]),
            MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
        )
        self.assertEqual(
            mind_v3_controller_architecture_status(
                MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE
            ),
            {
                "architecture": MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
                "loadable": True,
                "policy_input_safe": True,
                "promotion_eligible": True,
                "reason": None,
            },
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v5-controller.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            loaded = load_mind_v3_controller_metadata(path)
            founder_template = load_mind_v3_founder_template(path)

        self.assertEqual(loaded, metadata)
        self.assertEqual(founder_template, metadata)
        require_mind_v3_founder_template_promotion_eligible(founder_template)

    def test_v5_parameter_count_and_inherited_child_metadata_shape(self) -> None:
        parent = founder_mind_v3_metadata(
            agent_id=3,
            rng=Random(9),
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        child = inherit_mind_v3_metadata(
            primary_parent_metadata=parent,
            secondary_parent_metadata=None,
            child_agent_id=4,
            rng=Random(11),
        )

        self.assertEqual(
            child["architecture"],
            MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        self.assertEqual(
            child["state_size"],
            mind_v3_parameter_count(
                architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE
            ),
        )
        self.assertTrue(
            all(
                len(weights) == MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS
                for weights in child["action_head_weights"].values()
            )
        )

    def test_v5_descendants_can_mutate_context_unit_weights(self) -> None:
        parent = founder_mind_v3_metadata(
            agent_id=3,
            rng=Random(9),
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        child = inherit_mind_v3_metadata(
            primary_parent_metadata=parent,
            secondary_parent_metadata=None,
            child_agent_id=4,
            rng=Random(11),
        )

        context_weights = [
            value
            for weights in child["action_head_weights"].values()
            for value in weights[MIND_V3_HIDDEN_UNITS:]
        ]
        self.assertTrue(any(value != 0.0 for value in context_weights))

    def test_v5_scoring_is_invariant_to_mind_inheritance_bit(self) -> None:
        metadata = _linear_metadata(
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
        )
        unavailable = _observation_values()
        available = list(unavailable)
        available[SELF_INPUT_FIELDS.index("mind_inheritance_available")] = 1.0
        context_values = tuple(0.25 for _ in range(rollout_context_vector_size()))

        self.assertEqual(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=unavailable,
                action_mask=_action_mask(),
                rollout_context_values=context_values,
            ),
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=available,
                action_mask=_action_mask(),
                rollout_context_values=context_values,
            ),
        )

    def test_prior_public_rollout_context_can_change_v5_scores(self) -> None:
        context_unit = MIND_V3_HIDDEN_UNITS + rollout_context_vector_fields().index(
            "recent_resolved_count:eat"
        )
        metadata = _zero_metadata(
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
        )
        metadata["action_head_weights"]["eat"][context_unit] = 1.0
        context = RolloutContextState()
        context.update_from_record(_transition_record("eat"))

        without_context = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=_observation_values(),
            action_mask=_action_mask(),
            rollout_context_values=tuple(0.0 for _ in range(rollout_context_vector_size())),
        )
        with_context = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=_observation_values(),
            action_mask=_action_mask(),
            rollout_context_values=tuple(context.values()),
        )

        self.assertNotEqual(without_context, with_context)
        self.assertGreater(with_context["eat"], without_context["eat"])

    def test_context_is_not_updated_before_current_decision_finalizes(self) -> None:
        policy = MindV3EvolutionPolicy(
            seed=7,
            founder_template_metadata=founder_mind_v3_metadata(
                agent_id=0,
                rng=Random(1),
                architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            ),
        )
        observation = _observation(agent_id=3)
        action_mask = _action_mask()

        first = policy.decide(observation, action_mask)
        second = policy.decide(observation, action_mask)

        self.assertFalse(first.diagnostics["rollout_context_non_empty"])
        self.assertFalse(second.diagnostics["rollout_context_non_empty"])

    def test_observe_transition_returns_rollout_context_update_trace(self) -> None:
        policy = MindV3EvolutionPolicy(
            seed=7,
            founder_template_metadata=founder_mind_v3_metadata(
                agent_id=0,
                rng=Random(1),
                architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            ),
        )
        observation = _observation(agent_id=3)
        action_mask = _action_mask()
        decision = policy.decide(observation, action_mask)

        trace = policy.observe_transition(
            _transition_record(
                decision.requested_action,
                agent_id=3,
                observation_input=observation["observation_input"],
                action_mask=action_mask,
            )
        )

        self.assertIsNotNone(trace)
        context_trace = trace["rollout_context_update_trace"]
        self.assertEqual(
            context_trace["schema_version"],
            MIND_V3_ROLLOUT_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION,
        )
        self.assertEqual(context_trace["agent_id"], 3)
        self.assertEqual(
            context_trace["resolved_action"],
            decision.requested_action,
        )
        self.assertEqual(
            context_trace["previous_context"]["recent_resolved_actions"],
            [],
        )
        self.assertEqual(
            context_trace["updated_context"]["recent_resolved_actions"],
            [decision.requested_action],
        )
        json.dumps(trace)

    def test_current_transition_outcome_does_not_leak_into_same_adaptation_context(
        self,
    ) -> None:
        metadata = _zero_metadata(
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
        )
        policy = MindV3EvolutionPolicy(
            seed=7,
            founder_template_metadata=None,
        )
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        observation = _observation(agent_id=3)
        action_mask = _only_action_mask("eat")

        decision = policy.decide(observation, action_mask)
        self.assertEqual(decision.requested_action, "eat")
        self.assertFalse(decision.diagnostics["rollout_context_non_empty"])
        self.assertFalse(
            decision.diagnostics["rollout_context_post_carrion_context"]
        )

        trace = policy.observe_transition(
            _transition_record(
                "eat",
                agent_id=3,
                observation_input=observation["observation_input"],
                action_mask=action_mask,
                food_source="carcass",
                resource_gain=0.25,
            )
        )

        self.assertIsNotNone(trace)
        context_trace = trace["rollout_context_update_trace"]
        self.assertFalse(context_trace["previous_context"]["post_carrion_contact"])
        self.assertTrue(context_trace["updated_context"]["post_carrion_contact"])
        updated = policy._agent_metadata[3]
        post_only_fields = [
            "recent_requested_count:eat",
            "recent_resolved_count:eat",
            "recent_ate_rate",
            "recent_resource_gain_sum",
            "post_carrion_contact",
        ]
        for field in post_only_fields:
            unit = MIND_V3_HIDDEN_UNITS + rollout_context_vector_fields().index(field)
            self.assertEqual(updated["action_head_weights"]["eat"][unit], 0.0)

        next_decision = policy.decide(observation, action_mask)
        self.assertTrue(next_decision.diagnostics["rollout_context_non_empty"])
        self.assertTrue(
            next_decision.diagnostics["rollout_context_post_carrion_context"]
        )

    def test_v4_score_and_adapt_ignore_rollout_context_parameter(self) -> None:
        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        self.assertEqual(metadata["architecture"], MIND_V3_CONTROLLER_ARCHITECTURE)
        context_values = tuple(0.75 for _ in range(rollout_context_vector_size()))

        self.assertEqual(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
            ),
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
                rollout_context_values=context_values,
            ),
        )
        self.assertEqual(
            adapt_mind_v3_metadata(
                metadata=metadata,
                observation_input=_observation_values(),
                action="eat",
                reward_signal=0.5,
            ),
            adapt_mind_v3_metadata(
                metadata=metadata,
                observation_input=_observation_values(),
                action="eat",
                reward_signal=0.5,
                rollout_context_values=context_values,
            ),
        )

    def test_rollout_context_diagnostics_aggregate_from_decision_records(self) -> None:
        diagnostics = [
            {
                "rollout_context_schema_version": "mind_v3_rollout_context_v1",
                "rollout_context_non_empty": False,
                "rollout_context_post_carrion_context": False,
                "rollout_context_selected_score_delta": 0.0,
            },
            {
                "rollout_context_schema_version": "mind_v3_rollout_context_v1",
                "rollout_context_non_empty": True,
                "rollout_context_post_carrion_context": True,
                "rollout_context_selected_score_delta": -0.25,
            },
            {"controller_architecture": MIND_V3_CONTROLLER_ARCHITECTURE},
        ]
        records = [
            {"requested_action": "eat"},
            {"requested_action": "drink"},
            {"requested_action": "stay"},
        ]

        summary = evolve_cli._rollout_context_decision_summary(
            diagnostics_records=diagnostics,
            trajectory_records=records,
        )
        aggregate = evolve_cli._aggregate_rollout_context_diagnostics(
            [summary, summary]
        )

        self.assertEqual(summary["rollout_context_decision_count"], 2)
        self.assertEqual(summary["rollout_context_non_empty_count"], 1)
        self.assertEqual(summary["rollout_context_non_empty_share"], 0.5)
        self.assertEqual(
            summary["rollout_context_selected_score_delta_nonzero_count"],
            1,
        )
        self.assertEqual(
            summary["rollout_context_selected_score_delta_abs_mean"],
            0.125,
        )
        self.assertEqual(
            summary["rollout_context_selected_score_delta_abs_max"],
            0.25,
        )
        self.assertEqual(summary["rollout_context_post_carrion_context_count"], 1)
        self.assertEqual(
            summary["rollout_context_selected_score_delta_by_requested_action"][
                "drink"
            ]["abs_max"],
            0.25,
        )
        self.assertEqual(aggregate["rollout_context_decision_count"], 4)
        self.assertEqual(
            aggregate["rollout_context_selected_score_delta_abs_mean"],
            0.125,
        )

    def test_unsupported_resolution_breakdown_aggregates_by_seed_action_and_reason(
        self,
    ) -> None:
        records = [
            {
                "requested_action": "eat",
                "resolved_action": "stay",
                "resolution_action_valid": False,
                "outcome": {"invalid_reason": "not_in_resolution_action_mask"},
            },
            {
                "requested_action": "drink",
                "resolved_action": "drink",
                "resolution_action_valid": False,
                "outcome": {},
            },
        ]

        run_breakdown = evolve_cli._unsupported_action_breakdown(
            seed=5,
            records=records,
            validity_key="resolution_action_valid",
        )
        aggregate = evolve_cli._aggregate_unsupported_action_breakdowns(
            [{"seed": 5, "unsupported_resolved_action_breakdown": run_breakdown}],
            key="unsupported_resolved_action_breakdown",
        )

        self.assertEqual(run_breakdown["by_requested_action"]["eat"], 1)
        self.assertEqual(run_breakdown["by_resolved_action"]["drink"], 1)
        self.assertEqual(
            run_breakdown["by_invalid_reason"][
                "not_in_resolution_action_mask"
            ],
            1,
        )
        self.assertEqual(run_breakdown["by_invalid_reason"]["unknown"], 1)
        self.assertIn("5", aggregate["by_seed"])
        self.assertEqual(aggregate["by_requested_action"]["drink"], 1)
        self.assertEqual(aggregate["by_invalid_reason"]["unknown"], 1)

    def test_baseline_comparison_reports_matched_holdout_seed_deltas(self) -> None:
        current = {
            "holdout_evaluation": {
                "runs": [
                    {
                        "seed": 19,
                        "alive_agents": 13,
                        "births": 6,
                        "deaths": 3,
                        "unsupported_requested_action_count": 0,
                        "unsupported_resolved_action_count": 2,
                        "requested_action_counts": {"eat": 3, "drink": 1},
                    }
                ]
            }
        }
        baseline = {
            "holdout_evaluation": {
                "runs": [
                    {
                        "seed": 19,
                        "alive_agents": 10,
                        "births": 4,
                        "deaths": 5,
                        "unsupported_requested_action_count": 1,
                        "unsupported_resolved_action_count": 0,
                        "requested_action_counts": {"stay": 2, "eat": 1},
                    }
                ]
            }
        }
        with TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            comparison = evolve_cli._comparison_baseline_report(
                current,
                baseline_path,
            )

        self.assertEqual(comparison["matched_seed_count"], 1)
        delta = comparison["matched_holdout_seed_deltas"][0]
        self.assertEqual(delta["alive_agents_delta"], 3.0)
        self.assertEqual(delta["births_delta"], 2.0)
        self.assertEqual(delta["deaths_delta"], -2.0)
        self.assertEqual(
            delta["unsupported_requested_action_count_delta"],
            -1.0,
        )
        self.assertEqual(delta["unsupported_resolved_action_count_delta"], 2.0)
        self.assertEqual(delta["current_dominant_requested_action"], "eat")
        self.assertEqual(delta["baseline_dominant_requested_action"], "stay")
        self.assertTrue(delta["dominant_requested_action_changed"])

    def test_short_v5_smoke_run_is_heuristic_free_and_supported(self) -> None:
        policy = MindV3EvolutionPolicy(
            seed=5,
            founder_template_metadata=founder_mind_v3_metadata(
                agent_id=0,
                rng=Random(5),
                architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
            ),
        )
        world = SimulationWorld(
            WorldConfig(seed=5, max_ticks=3),
            policy=policy,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        diagnostics = [
            diagnostics
            for diagnostics in world.policy_decision_diagnostics_records
            if isinstance(diagnostics, dict)
        ]
        self.assertTrue(diagnostics)
        self.assertTrue(all(item["heuristic_free"] for item in diagnostics))
        self.assertFalse(
            any(
                "heuristic" in str(record.get("action_source", ""))
                for record in world.trajectory_records
            )
        )
        self.assertEqual(
            sum(
                1
                for record in world.trajectory_records
                if record.get("action_valid") is False
            ),
            0,
        )
        self.assertEqual(
            sum(
                1
                for record in world.trajectory_records
                if record.get("resolution_action_valid") is False
            ),
            0,
        )

    def test_contract_declares_v5_rollout_context_boundary(self) -> None:
        contract = mind_v3_autonomous_evolution_contract()
        opt_in = {
            item["architecture"]: item
            for item in contract["controller"]["opt_in_architectures"]
        }
        v5 = opt_in[MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE]

        self.assertEqual(v5["base_architecture"], MIND_V3_CONTROLLER_ARCHITECTURE)
        self.assertEqual(
            v5["founder_initialization"],
            "v4_base_prior_plus_exact_zero_rollout_context_units",
        )
        self.assertFalse(v5["current_or_future_outcome_input"])
        self.assertTrue(v5["controller_private_diagnostics_excluded"])
        self.assertEqual(
            v5["rollout_context_feature_contract"]["row_scope"],
            "per_agent_previous_rows_before_current_decision",
        )
        self.assertIn(
            "future trajectory rows",
            v5["rollout_context_feature_contract"]["excluded_runtime_inputs"],
        )


def _linear_metadata(*, architecture: str, hidden_units: int) -> dict[str, object]:
    weights = {
        action: [0.01 * float(index + 1) for index in range(hidden_units)]
        for action in ACTION_NAMES
    }
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "state_size": mind_v3_parameter_count(architecture=architecture),
        "architecture": architecture,
        "action_head_weights": weights,
        "action_head_bias": {action: 0.0 for action in ACTION_NAMES},
    }


def _zero_metadata(*, architecture: str, hidden_units: int) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "state_size": mind_v3_parameter_count(architecture=architecture),
        "architecture": architecture,
        "action_head_weights": {
            action: [0.0] * hidden_units for action in ACTION_NAMES
        },
        "action_head_bias": {action: 0.0 for action in ACTION_NAMES},
    }


def _observation(*, agent_id: int) -> dict[str, object]:
    return {
        "metadata": {"agent_id": agent_id},
        "self": {"trophic_role": "omnivore", "meat_mode": "mixed"},
        "observation_input": {"values": _observation_values()},
    }


def _observation_values() -> list[float]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.42
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.58
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.86
    values[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.51
    return values


def _action_mask() -> dict[str, bool]:
    return {action: action in {"stay", "eat", "drink"} for action in ACTION_NAMES}


def _only_action_mask(enabled_action: str) -> dict[str, bool]:
    return {action: action == enabled_action for action in ACTION_NAMES}


def _transition_record(
    action: str,
    *,
    agent_id: int = 3,
    observation_input: dict[str, object] | None = None,
    action_mask: dict[str, bool] | None = None,
    food_source: str | None = None,
    resource_gain: float | None = None,
) -> dict[str, object]:
    resolved_food_source = (
        food_source if food_source is not None else ("plant" if action == "eat" else None)
    )
    resolved_resource_gain = (
        float(resource_gain)
        if resource_gain is not None
        else (0.1 if action == "eat" else 0.0)
    )
    return {
        "tick": 0,
        "agent_id": agent_id,
        "policy_id": "mind_v3_autonomous_evolution_policy",
        "policy_version": "mind_v3_autonomous_evolution_policy_v1",
        "action_source": "mind_v3_autonomous_evolution_policy_v1",
        "observation_input": observation_input or {"values": _observation_values()},
        "action_mask": action_mask or _action_mask(),
        "requested_action": action,
        "resolved_action": action,
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": action.startswith("move_"),
        "before": {
            "x": 1,
            "y": 1,
            "energy_ratio": 0.45,
            "hydration_ratio": 0.55,
            "health_ratio": 0.9,
            "alive": True,
        },
        "after": {
            "x": 1,
            "y": 1,
            "energy_ratio": 0.5 if action == "eat" else 0.44,
            "hydration_ratio": 0.7 if action == "drink" else 0.54,
            "health_ratio": 0.9,
            "alive": True,
        },
        "outcome": {
            "feeding": {
                "ate": action == "eat",
                "food_source": resolved_food_source,
            },
            "drinking": {"drank": action == "drink"},
            "passive": {"died_after_action": False},
            "resource_gain": resolved_resource_gain,
            "died": False,
            "reproduced": False,
            "reproduction_ready_after": False,
        },
        "reward": {"total": 0.2},
    }


if __name__ == "__main__":
    unittest.main()
