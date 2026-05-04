from __future__ import annotations

from python.tests.runtime_test_helpers import *


class RuntimeObservationContractTests(RuntimeContractTestHelpers):
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
        self.assertEqual(contract["policy_input"]["shape"], [OBSERVATION_INPUT_VECTOR_SIZE])
        self.assertEqual(encoded["decoded_dtype"], OBSERVATION_INPUT_DTYPE)
        self.assertEqual(encoded["shape"], [OBSERVATION_INPUT_VECTOR_SIZE])
        self.assertEqual(len(decoded), OBSERVATION_INPUT_VECTOR_SIZE)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in decoded))
        self.assertTrue(all(isinstance(value, float) for value in decoded))
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        json.dumps(observation)
        json.dumps(encoded)

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
