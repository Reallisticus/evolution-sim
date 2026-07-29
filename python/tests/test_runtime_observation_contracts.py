from __future__ import annotations

from python.tests.runtime_test_helpers import *
from evolution_sim.mind.feature_policy import feature_keys_from_observation
from evolution_sim.env.runtime.observations import (
    SELF_INPUT_FIELDS,
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    quantized_observation_input_values,
)
from evolution_sim.mind.policy_inputs import (
    ecological_policy_input_values,
    ecological_policy_values_from_observation,
)


class RuntimeObservationContractTests(RuntimeContractTestHelpers):
    def test_direct_quantized_observation_values_match_storage_round_trip(
        self,
    ) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        encoded = encode_observation_input(observation)

        self.assertEqual(
            quantized_observation_input_values(observation),
            decode_observation_input(encoded),
        )
        self.assertEqual(
            ecological_policy_values_from_observation(observation),
            ecological_policy_input_values(encoded),
        )
        diagnostic_index = SELF_INPUT_FIELDS.index("mind_inheritance_available")
        decoded = decode_observation_input(encoded)
        self.assertEqual(
            ecological_policy_values_from_observation(observation),
            tuple(decoded[:diagnostic_index] + decoded[diagnostic_index + 1 :]),
        )

    def test_direct_quantized_projection_rejects_invalid_excluded_indices(
        self,
    ) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )

        for index in (-1, OBSERVATION_INPUT_VECTOR_SIZE, True):
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                    ValueError,
                    "excluded observation input index is invalid",
                ):
                    quantized_observation_input_values(
                        observation,
                        excluded_indices=frozenset({index}),
                    )

    def test_direct_observation_values_fail_closed_like_encoder(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        nonfinite_value = copy.deepcopy(observation)
        nonfinite_value["self"]["energy_ratio"] = float("nan")
        unknown_enum = copy.deepcopy(observation)
        unknown_enum["local_patch"][0]["terrain"] = "hostile_terrain"
        hostile_observations = {
            "stale schema": {
                **copy.deepcopy(observation),
                "schema_version": "mind_observation_v2",
            },
            "token schema without channels": {
                **copy.deepcopy(observation),
                "schema_version": TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
            },
            "nonfinite value": nonfinite_value,
            "unknown enum": unknown_enum,
        }

        for label, hostile in hostile_observations.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as encoded_error:
                    encode_observation_input(hostile)
                with self.assertRaises(ValueError) as direct_error:
                    quantized_observation_input_values(hostile)
                with self.assertRaises(ValueError) as policy_error:
                    ecological_policy_values_from_observation(hostile)
                self.assertEqual(
                    str(direct_error.exception), str(encoded_error.exception)
                )
                self.assertEqual(
                    str(policy_error.exception), str(encoded_error.exception)
                )

    def test_observation_contract_is_serializable_and_unprivileged(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]

        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        digest = observation_digest(observation)
        encoded = encode_observation_input(observation)
        decoded = decode_observation_input(encoded)
        contract = observation_contract()

        self.assertEqual(observation["schema_version"], OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(
            set(observation),
            {
                "schema_version",
                "metadata",
                "self",
                "local_patch",
                "navigation",
                "action_mask",
            },
        )
        self.assertNotIn("agent_id", observation)
        self.assertEqual(observation["metadata"], {"agent_id": agent.agent_id})
        self.assertEqual(len(observation["local_patch"]), PATCH_CELL_COUNT)
        self.assertNotIn("world", observation)
        self.assertNotIn("grid", observation)
        self.assertNotIn("agents", observation)
        self.assertTrue(contract["metadata_policy_excluded"])
        self.assertEqual(
            contract["policy_input"]["encoder_version"],
            OBSERVATION_ENCODER_VERSION,
        )
        self.assertEqual(
            contract["action_contract"]["schema_version"],
            ACTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            contract["signal_contract"]["schema_version"],
            SIGNAL_CONTRACT_VERSION,
        )
        self.assertIn("reproductive_expression", contract["enum_vocabs"])
        self.assertIn(
            "reproductive_expression_code",
            contract["policy_input"]["self_input_fields"],
        )
        self.assertFalse(contract["mind_inheritance_placeholder"]["policy_visible"])
        self.assertEqual(
            contract["policy_input"]["shape"],
            [OBSERVATION_INPUT_VECTOR_SIZE],
        )
        self.assertEqual(
            contract["policy_input"]["semantic_role"],
            "raw_encoded_observation_tensor",
        )
        self.assertFalse(
            contract["policy_input"]["promotion_eligible_direct_policy_input"]
        )
        self.assertEqual(encoded["decoded_dtype"], OBSERVATION_INPUT_DTYPE)
        self.assertEqual(encoded["shape"], [OBSERVATION_INPUT_VECTOR_SIZE])
        self.assertEqual(len(decoded), OBSERVATION_INPUT_VECTOR_SIZE)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in decoded))
        self.assertTrue(all(isinstance(value, float) for value in decoded))
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        json.dumps(observation)
        json.dumps(encoded)

    def test_raw_observation_contract_keeps_mind_diagnostic_as_compatibility_field(
        self,
    ) -> None:
        contract = observation_contract()

        self.assertIn(
            "mind_inheritance_available",
            contract["policy_input"]["self_input_fields"],
        )
        self.assertIn("mind_inheritance_available", SELF_INPUT_FIELDS)
        self.assertEqual(
            contract["policy_input"]["shape"],
            [OBSERVATION_INPUT_VECTOR_SIZE],
        )
        self.assertEqual(
            contract["policy_input"]["compatibility_role"],
            "historical_policy_input_key_compatibility",
        )
        self.assertTrue(
            contract["policy_input"]["contains_controller_private_diagnostics"]
        )
        self.assertEqual(
            contract["policy_input"]["controller_private_diagnostic_fields"],
            ["self.mind_inheritance_available"],
        )
        self.assertFalse(
            contract["policy_input"]["promotion_eligible_direct_policy_input"]
        )
        self.assertTrue(
            contract["policy_input"][
                "safe_projection_required_for_mind_v3_promotion"
            ]
        )
        self.assertIn(
            "mind_ecological_policy_input_v1",
            contract["policy_input"]["promotion_safe_projection_examples"],
        )
        self.assertIn(
            "architecture-specific safe feature selection",
            contract["policy_input"]["promotion_policy_input_guidance"],
        )

    def test_runtime_observation_contract_has_no_mind_dependency(self) -> None:
        source = Path("python/evolution_sim/env/runtime/observations.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("evolution_sim.mind", source)

    def test_observation_builder_uses_explicit_context_for_policy_inputs(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        context = world._observation_context(agent)
        expected = build_observation(world, agent, observation_context=context)

        private_reads = (
            "_trophic_profile",
            "_hazard_at",
            "_current_biotic_state",
            "_current_signal_state",
            "_energy_ratio",
            "_hydration_ratio",
            "_health_ratio",
            "_is_reproduction_ready",
            "_matched_diet_ratio",
            "_water_access_reason",
            "_hydrology_support_code",
            "_refuge_score",
            "_ecology_state_at",
            "_in_bounds",
            "_movement_actions",
            "_prey_vulnerability",
        )
        with ExitStack() as stack:
            for name in private_reads:
                stack.enter_context(
                    patch.object(world, name, side_effect=AssertionError(name))
                )
            actual = build_observation(world, agent, observation_context=context)

        self.assertEqual(observation_digest(actual), observation_digest(expected))
        self.assertEqual(
            decode_observation_input(encode_observation_input(actual)),
            decode_observation_input(encode_observation_input(expected)),
        )

    def test_observation_input_encoder_excludes_privileged_payloads(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        baseline = encode_observation_input(observation)

        mutated = copy.deepcopy(observation)
        mutated["metadata"] = {"agent_id": agent.agent_id + 10_000}
        mutated["action_mask"] = {
            str(action): not bool(enabled)
            for action, enabled in dict(observation["action_mask"]).items()
        }
        mutated["world"] = {"width": world.config.width, "height": world.config.height}
        mutated["grid"] = [["privileged"]]
        mutated["agents"] = [agent.agent_id]
        mutated["privileged_world_state"] = True

        self.assertEqual(encode_observation_input(mutated), baseline)
        self.assertEqual(
            decode_observation_input(encode_observation_input(mutated)),
            decode_observation_input(baseline),
        )

    def test_policy_feature_keys_ignore_observation_metadata_payloads(self) -> None:
        world = SimulationWorld(WorldConfig(seed=7, max_ticks=1))
        agent = world.alive_agents()[0]
        observation = build_observation(
            world,
            agent,
            observation_context=world._observation_context(agent),
        )
        action_mask = dict(observation["action_mask"])
        baseline_encoded = encode_observation_input(observation)
        baseline_features = feature_keys_from_observation(observation, action_mask)

        mutated = copy.deepcopy(observation)
        mutated["metadata"] = {
            "agent_id": agent.agent_id + 1000,
            "x": agent.x,
            "y": agent.y,
            "energy": agent.energy,
        }
        mutated["action_mask"] = {
            action: not bool(available)
            for action, available in action_mask.items()
        }
        mutated["world"] = world.config.to_dict()
        mutated["agents"] = list(world.agents)

        self.assertEqual(encode_observation_input(mutated), baseline_encoded)
        self.assertEqual(
            feature_keys_from_observation(mutated, action_mask),
            baseline_features,
        )
