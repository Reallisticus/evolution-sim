from __future__ import annotations

import json
import unittest
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_evolve as evolve_cli
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    SELF_INPUT_FIELDS,
    encode_observation_input,
)
from evolution_sim.mind.contracts import mind_v3_autonomous_evolution_contract
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_ARCHITECTURE,
    MIND_V3_CONTROLLER_SCHEMA_VERSION,
    MIND_V3_HIDDEN_UNITS,
    MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
    MIND_V3_RECOVERY_CONTEXT_HIDDEN_UNITS,
    MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
    MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
    adapt_mind_v3_metadata,
    founder_mind_v3_metadata,
    inherit_mind_v3_metadata,
    load_mind_v3_controller_metadata,
    mind_v3_parameter_count,
    score_mind_v3_metadata,
)
from evolution_sim.mind.recovery_context import (
    MIND_V3_RECOVERY_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION,
    RecoveryContextState,
    recovery_context_decoded_observation_values,
    recovery_context_feature_contract,
    recovery_context_values,
    recovery_context_vector_fields,
    recovery_context_vector_size,
)
from evolution_sim.mind.rollout_context import (
    RolloutContextState,
    rollout_context_vector_size,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


class MindV3RecoveryContextControllerTests(unittest.TestCase):
    def test_v6_metadata_save_load_inherit_and_adapt_round_trips(self) -> None:
        parent = founder_mind_v3_metadata(
            agent_id=7,
            rng=Random(123),
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        child = inherit_mind_v3_metadata(
            primary_parent_metadata=parent,
            secondary_parent_metadata=None,
            child_agent_id=9,
            rng=Random(456),
        )
        updated = adapt_mind_v3_metadata(
            metadata=child,
            observation_input=_observation_values(),
            action="drink",
            reward_signal=0.5,
            rollout_context_values=tuple(0.2 for _ in range(rollout_context_vector_size())),
            recovery_context_values=tuple(
                0.3 for _ in range(recovery_context_vector_size())
            ),
        )

        self.assertEqual(
            parent["architecture"],
            MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        self.assertEqual(
            child["state_size"],
            mind_v3_parameter_count(
                architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE
            ),
        )
        self.assertEqual(
            updated["architecture"],
            MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        context_weights = [
            value
            for weights in child["action_head_weights"].values()
            for value in weights[MIND_V3_HIDDEN_UNITS:]
        ]
        self.assertTrue(any(value != 0.0 for value in context_weights))
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v6-controller.json"
            path.write_text(json.dumps(updated), encoding="utf-8")
            loaded = load_mind_v3_controller_metadata(path)

        self.assertEqual(loaded, updated)

    def test_v6_founder_context_and_recovery_units_are_exactly_zero(self) -> None:
        profile = "scavenger"
        v4 = founder_mind_v3_metadata(
            agent_id=5,
            rng=Random(22),
            specialization_profile=profile,
        )
        v6 = founder_mind_v3_metadata(
            agent_id=5,
            rng=Random(22),
            specialization_profile=profile,
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        )

        self.assertEqual(v6["action_head_bias"], v4["action_head_bias"])
        self.assertEqual(
            v6["founder_prior_policy"],
            (
                "diverse_need_gated_navigation_action_prior_v3_plus_zero_initialized_"
                "rollout_and_recovery_context_units_v1"
            ),
        )
        for action in ACTION_NAMES:
            self.assertEqual(
                v6["action_head_weights"][action][:MIND_V3_HIDDEN_UNITS],
                v4["action_head_weights"][action],
            )
            self.assertEqual(
                v6["action_head_weights"][action][MIND_V3_HIDDEN_UNITS:],
                [0.0]
                * (rollout_context_vector_size() + recovery_context_vector_size()),
            )

    def test_v4_default_and_v5_behavior_remain_unchanged(self) -> None:
        v4 = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        v5 = founder_mind_v3_metadata(
            agent_id=3,
            rng=Random(9),
            architecture=MIND_V3_ROLLOUT_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        recovery_values = tuple(0.9 for _ in range(recovery_context_vector_size()))
        rollout_values = tuple(0.8 for _ in range(rollout_context_vector_size()))

        self.assertEqual(v4["architecture"], MIND_V3_CONTROLLER_ARCHITECTURE)
        self.assertEqual(
            len(v5["action_head_weights"]["eat"]),
            MIND_V3_ROLLOUT_CONTEXT_HIDDEN_UNITS,
        )
        self.assertEqual(
            score_mind_v3_metadata(
                metadata=v4,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
            ),
            score_mind_v3_metadata(
                metadata=v4,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
                rollout_context_values=rollout_values,
                recovery_context_values=recovery_values,
            ),
        )
        self.assertEqual(
            score_mind_v3_metadata(
                metadata=v5,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
                rollout_context_values=rollout_values,
            ),
            score_mind_v3_metadata(
                metadata=v5,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
                rollout_context_values=rollout_values,
                recovery_context_values=recovery_values,
            ),
        )

    def test_evolve_cli_parser_accepts_v6_architecture(self) -> None:
        args = evolve_cli.build_parser().parse_args(
            [
                "--controller-architecture",
                MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
            ]
        )

        self.assertEqual(
            args.controller_architecture,
            MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        )

    def test_v6_scoring_is_invariant_to_mind_inheritance_available(self) -> None:
        metadata = _linear_metadata(
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_RECOVERY_CONTEXT_HIDDEN_UNITS,
        )
        unavailable = _observation_values()
        available = list(unavailable)
        available[SELF_INPUT_FIELDS.index("mind_inheritance_available")] = 1.0

        self.assertEqual(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=unavailable,
                action_mask=_action_mask(),
                rollout_context_values=tuple(0.2 for _ in range(rollout_context_vector_size())),
                recovery_context_values=tuple(
                    0.4 for _ in range(recovery_context_vector_size())
                ),
            ),
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=available,
                action_mask=_action_mask(),
                rollout_context_values=tuple(0.2 for _ in range(rollout_context_vector_size())),
                recovery_context_values=tuple(
                    0.4 for _ in range(recovery_context_vector_size())
                ),
            ),
        )

    def test_same_row_outcome_cannot_affect_current_adaptation_context(self) -> None:
        metadata = _zero_metadata(
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_RECOVERY_CONTEXT_HIDDEN_UNITS,
        )
        recovery_unit = _recovery_unit(
            "post_carrion_contact_x_hydration_debt"
        )
        policy = MindV3EvolutionPolicy(seed=7)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        observation = _observation(agent_id=3, hydration=0.35)
        action_mask = _only_action_mask("eat")

        decision = policy.decide(observation, action_mask)
        trace = policy.observe_transition(
            _transition_record(
                "eat",
                agent_id=3,
                observation_input=observation["observation_input"],
                action_mask=action_mask,
                food_source="carcass",
                resource_gain=0.3,
            )
        )

        self.assertEqual(decision.requested_action, "eat")
        self.assertFalse(decision.diagnostics["recovery_context_non_empty"])
        self.assertIsNotNone(trace)
        self.assertEqual(
            trace["recovery_context_update_trace"]["schema_version"],
            MIND_V3_RECOVERY_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION,
        )
        self.assertEqual(
            policy._agent_metadata[3]["action_head_weights"]["eat"][recovery_unit],
            0.0,
        )

        next_decision = policy.decide(observation, action_mask)
        self.assertTrue(next_decision.diagnostics["recovery_context_non_empty"])
        self.assertTrue(
            next_decision.diagnostics["recovery_context_post_carrion_contact"]
        )

    def test_prior_navigation_progress_uses_previous_public_observation(self) -> None:
        metadata = _zero_metadata(
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_RECOVERY_CONTEXT_HIDDEN_UNITS,
        )
        progress_unit = _recovery_unit(
            "post_carrion_contact_x_water_approach_progress"
        )
        metadata["action_head_weights"]["drink"][progress_unit] = 1.0
        policy = MindV3EvolutionPolicy(seed=7)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        first = _observation(agent_id=3, water_distance=8.0)
        second = _observation(agent_id=3, water_distance=3.0)

        policy.decide(first, _only_action_mask("eat"))
        policy.observe_transition(
            _transition_record(
                "eat",
                agent_id=3,
                observation_input=first["observation_input"],
                action_mask=_only_action_mask("eat"),
                food_source="carcass",
                resource_gain=0.3,
            )
        )
        decision = policy.decide(second, _action_mask())

        self.assertGreater(
            decision.diagnostics["recovery_context_score_delta_by_action"]["drink"],
            0.0,
        )
        self.assertEqual(
            decision.diagnostics["recovery_context_water_distance_bin"],
            "low",
        )

    def test_missing_prior_navigation_and_malformed_vectors_zero_consistently(
        self,
    ) -> None:
        state = RecoveryContextState()
        rollout = RolloutContextState()
        rollout.update_from_record(_transition_record("eat", food_source="carcass"))
        current = _observation(agent_id=3, water_distance=3.0)
        values = state.values(
            rollout_context_snapshot=rollout.snapshot(),
            current_observation_input=current["observation_input"],
            action_mask=_action_mask(),
        )
        self.assertEqual(
            values[
                recovery_context_vector_fields().index(
                    "post_carrion_contact_x_water_approach_progress"
                )
            ],
            0.0,
        )

        metadata = _zero_metadata(
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
            hidden_units=MIND_V3_RECOVERY_CONTEXT_HIDDEN_UNITS,
        )
        metadata["action_head_weights"]["drink"][
            _recovery_unit("post_carrion_contact_x_hydration_debt")
        ] = 1.0
        zero_scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=_observation_values(),
            action_mask=_action_mask(),
            rollout_context_values=tuple(0.0 for _ in range(rollout_context_vector_size())),
            recovery_context_values=tuple(
                0.0 for _ in range(recovery_context_vector_size())
            ),
        )
        self.assertEqual(
            zero_scores,
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
                rollout_context_values=tuple(0.0 for _ in range(rollout_context_vector_size())),
                recovery_context_values=(1.0, "bad"),
            ),
        )
        self.assertGreater(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=_observation_values(),
                action_mask=_action_mask(),
                rollout_context_values=tuple(0.0 for _ in range(rollout_context_vector_size())),
                recovery_context_values=tuple(
                    1.0 for _ in range(recovery_context_vector_size())
                ),
            )["drink"],
            zero_scores["drink"],
        )

    def test_optimized_recovery_context_values_match_raw_decode_path(self) -> None:
        rollout = RolloutContextState()
        rollout.update_from_record(_transition_record("eat", food_source="carcass"))
        current = _observation_input(
            energy=0.24,
            hydration=0.31,
            water_distance=2.0,
        )
        previous = _observation_input(
            energy=0.7,
            hydration=0.8,
            water_distance=7.0,
        )

        decoded_current = recovery_context_decoded_observation_values(current)
        decoded_previous = recovery_context_decoded_observation_values(previous)
        legacy_encoded = recovery_context_values(
            rollout_context_snapshot=rollout.snapshot(),
            current_observation_input=current,
            previous_observation_input=previous,
            action_mask=_action_mask(),
        )
        state = RecoveryContextState()
        state.update_from_record(
            {"agent_id": 3, "tick": 1, "observation_input": previous},
            observation_values=decoded_previous,
        )
        optimized_encoded = state.values(
            rollout_context_snapshot=rollout.snapshot(),
            current_observation_values=decoded_current,
            action_mask=_action_mask(),
        )

        self.assertEqual(optimized_encoded, legacy_encoded)

        legacy_raw_values = recovery_context_values(
            rollout_context_snapshot=rollout.snapshot(),
            current_observation_input={"values": list(decoded_current)},
            previous_observation_input={"values": list(decoded_previous)},
            action_mask=_action_mask(),
        )
        raw_state = RecoveryContextState()
        raw_state.update_from_record(
            {"agent_id": 3, "tick": 1, "observation_input": previous},
            observation_values=decoded_previous,
        )
        optimized_raw_values = raw_state.values(
            rollout_context_snapshot=rollout.snapshot(),
            current_observation_values=decoded_current,
            action_mask=_action_mask(),
        )

        self.assertEqual(optimized_raw_values, legacy_raw_values)

    def test_recovery_context_reuses_decoded_values_without_decoding(self) -> None:
        rollout = RolloutContextState()
        rollout.update_from_record(_transition_record("eat", food_source="carcass"))
        current = _observation_input(
            energy=0.24,
            hydration=0.31,
            water_distance=2.0,
        )
        previous = _observation_input(
            energy=0.7,
            hydration=0.8,
            water_distance=7.0,
        )
        decoded_current = recovery_context_decoded_observation_values(current)
        decoded_previous = recovery_context_decoded_observation_values(previous)
        state = RecoveryContextState()
        state.update_from_record(
            {"agent_id": 3, "tick": 1, "observation_input": previous},
            observation_values=decoded_previous,
        )

        with patch(
            "evolution_sim.mind.recovery_context.decode_observation_input",
            side_effect=AssertionError("decode should not be called"),
        ):
            values = state.values(
                rollout_context_snapshot=rollout.snapshot(),
                current_observation_values=decoded_current,
                action_mask=_action_mask(),
            )

        self.assertTrue(any(value != 0.0 for value in values))

    def test_previous_observation_storage_is_public_compact_summary(self) -> None:
        state = RecoveryContextState()
        previous = _observation_input(
            energy=0.7,
            hydration=0.8,
            water_distance=7.0,
        )
        decoded_previous = recovery_context_decoded_observation_values(previous)
        trace = state.update_from_record(
            {
                "agent_id": 3,
                "tick": 1,
                "observation_input": previous,
                "fixture_name": "carrion_only",
            },
            observation_values=decoded_previous,
        )

        self.assertFalse(hasattr(state, "previous_observation_input"))
        self.assertIn("previous_public_observation_summary", state.__dict__)
        state_payload = json.dumps(state.__dict__, sort_keys=True).lower()
        trace_payload = json.dumps(trace, sort_keys=True).lower()
        self.assertIn("water_distance", state_payload)
        self.assertNotIn("observation_input", state_payload)
        self.assertNotIn("mind_inheritance_available", state_payload)
        self.assertNotIn("fixture", state_payload)
        self.assertNotIn("mind_inheritance_available", trace_payload)
        self.assertNotIn("fixture", trace_payload)

    def test_no_fixture_identity_in_v6_metadata_contract_or_diagnostics(self) -> None:
        metadata = founder_mind_v3_metadata(
            agent_id=1,
            rng=Random(2),
            architecture=MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE,
        )
        contract = mind_v3_autonomous_evolution_contract()
        opt_in = {
            item["architecture"]: item
            for item in contract["controller"]["opt_in_architectures"]
        }
        v6_contract = opt_in[MIND_V3_RECOVERY_CONTEXT_CONTROLLER_ARCHITECTURE]
        feature_contract = recovery_context_feature_contract()
        policy = MindV3EvolutionPolicy(seed=7)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        diagnostics = policy.decide(_observation(agent_id=3), _action_mask()).diagnostics

        for payload in (metadata, v6_contract, feature_contract, diagnostics):
            self.assertNotIn("fixture", json.dumps(payload, sort_keys=True).lower())

    def test_recovery_context_contract_declares_previous_public_outcomes(
        self,
    ) -> None:
        feature_contract = recovery_context_feature_contract()

        self.assertEqual(
            feature_contract["outcome_timing_contract"],
            "previous_finalized_public_rows_only_not_current_or_future_outcomes",
        )
        self.assertIn(
            "outcome.resource_gain",
            feature_contract["previous_finalized_public_outcome_dependencies"],
        )
        self.assertIn(
            "outcome.feeding.food_source",
            feature_contract["previous_finalized_public_outcome_dependencies"],
        )
        self.assertIn(
            "post_carrion_contact",
            feature_contract["derived_rollout_snapshot_dependencies"],
        )
        self.assertIn(
            "ticks_since_drink",
            feature_contract["derived_rollout_snapshot_dependencies"],
        )
        self.assertIn(
            "no_gain_eat_streak",
            feature_contract["derived_rollout_snapshot_dependencies"],
        )


def _recovery_unit(field: str) -> int:
    return (
        MIND_V3_HIDDEN_UNITS
        + rollout_context_vector_size()
        + recovery_context_vector_fields().index(field)
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


def _observation(
    *,
    agent_id: int,
    energy: float = 0.42,
    hydration: float = 0.58,
    water_distance: float = 4.0,
) -> dict[str, object]:
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "metadata": {"agent_id": agent_id},
        "self": {"trophic_role": "omnivore", "meat_mode": "mixed"},
        "observation_input": _observation_input(
            energy=energy,
            hydration=hydration,
            water_distance=water_distance,
        ),
    }


def _observation_values() -> list[float]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.42
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.58
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.86
    values[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.51
    return values


def _observation_input(
    *,
    energy: float,
    hydration: float,
    water_distance: float,
) -> dict[str, object]:
    patch = []
    for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1):
        for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1):
            patch.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "same_lineage": False,
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_type": "none",
                    "hazard_level": 0.0,
                    "ecology_state": "stable",
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                    "reproductive_signal": 0.0,
                    "communication_signal": 0.0,
                }
            )
    return encode_observation_input(
        {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "metadata": {"agent_id": 3},
            "self": {
                "energy_ratio": energy,
                "hydration_ratio": hydration,
                "health_ratio": 0.9,
                "injury_load": 0.0,
                "age_norm": 0.1,
                "reproduction_ready": False,
                "matched_diet_ratio": 1.0,
                "trophic_role": "carnivore",
                "meat_mode": "scavenger",
                "season": "wet",
                "water_access_reason": "none",
                "hydrology_support_code": 1,
                "refuge_score": 0.0,
                "hazard_type": "none",
                "hazard_level": 0.0,
                "tile_vegetation": 0.0,
                "tile_recovery_debt": 0.0,
                "reproductive_stage": "stage0_asexual",
                "reproductive_expression": "asexual",
                "sexual_reproduction_unlocked": False,
                "reproductive_signal": 0.0,
                "communication_signal": 0.0,
                "mind_inheritance_available": False,
            },
            "local_patch": patch,
            "navigation": {
                "water": {
                    "dx": 0,
                    "dy": -int(water_distance),
                    "distance": water_distance,
                    "strength": 1.0,
                },
                "plant": {"dx": 0, "dy": 0, "distance": 0.0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0.0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0.0, "strength": 0.0},
            },
            "action_mask": _action_mask(),
        }
    )


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
        "observation_input": observation_input
        or _observation(agent_id=agent_id)["observation_input"],
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
                "food_source": food_source if food_source is not None else "plant",
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
