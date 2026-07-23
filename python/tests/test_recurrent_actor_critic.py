from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.env.runtime.observations import (
        OBSERVATION_INPUT_VECTOR_SIZE,
        SELF_INPUT_FIELDS,
    )
    from evolution_sim.mind.recurrent_actor_critic import (
        ACTION_COUNT,
        BackendStableLayerNorm,
        LEARNED_ENCODER_INPUT_SIZE,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        PUBLIC_INPUT_SIZE,
        ActionMaskError,
        PerAgentRecurrentStateStore,
        PreviousPublicFeedbackInput,
        PublicInputError,
        PublicRecurrentActorCritic,
        RecurrentContextError,
        RecurrentPolicyContractError,
        RecurrentStateError,
        public_policy_tensor_from_decoded,
        previous_public_feedback_tensor,
        recurrent_actor_critic_contract,
        seeded_torch_generator,
        strict_action_mask_tensor,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class PublicRecurrentActorCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        self.model = PublicRecurrentActorCritic(initialization_seed=1729)

    def test_contract_pins_public_input_and_stable_action_space(self) -> None:
        contract = recurrent_actor_critic_contract(self.model.config)

        self.assertEqual(PUBLIC_INPUT_SIZE, 541)
        self.assertEqual(ACTION_COUNT, 20)
        self.assertEqual(PREVIOUS_PUBLIC_FEEDBACK_SIZE, 43)
        self.assertEqual(LEARNED_ENCODER_INPUT_SIZE, 604)
        self.assertEqual(contract["public_input_size"], 541)
        self.assertEqual(contract["action_names"], list(ACTION_NAMES))
        self.assertEqual(contract["architecture"]["actor"], "linear_20_logits")
        self.assertEqual(
            contract["architecture"]["input_normalization"],
            "explicit_layer_norm",
        )
        self.assertEqual(
            contract["architecture"]["learned_encoder_input_size"],
            604,
        )
        self.assertFalse(contract["runtime_integrated"])
        self.assertFalse(contract["heuristic_action_source"])

    def test_backend_stable_layer_norm_matches_torch_layer_norm(self) -> None:
        inputs = torch.linspace(-2.0, 2.0, 4 * 3 * 604).reshape(4, 3, 604)
        reference = torch.nn.LayerNorm(604)
        stable = BackendStableLayerNorm(604)
        stable.load_state_dict(reference.state_dict())

        reference_output = reference(inputs)
        stable_output = stable(inputs)
        torch.testing.assert_close(stable_output, reference_output)

        reference_output.square().mean().backward()
        stable_output.square().mean().backward()
        torch.testing.assert_close(stable.weight.grad, reference.weight.grad)
        torch.testing.assert_close(stable.bias.grad, reference.bias.grad)

    def test_public_projection_excludes_controller_diagnostic(self) -> None:
        unavailable = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        unavailable[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.4
        available = list(unavailable)
        available[SELF_INPUT_FIELDS.index("mind_inheritance_available")] = 1.0

        unavailable_tensor = public_policy_tensor_from_decoded(unavailable)
        available_tensor = public_policy_tensor_from_decoded(available)

        self.assertEqual(tuple(unavailable_tensor.shape), (PUBLIC_INPUT_SIZE,))
        torch.testing.assert_close(unavailable_tensor, available_tensor)

    def test_public_input_validation_rejects_wrong_shape_dtype_and_values(self) -> None:
        masks = self._masks(time_steps=1, batch_size=1)
        feedback = self._feedback(time_steps=1, batch_size=1)
        cases = (
            torch.zeros(1, 1, PUBLIC_INPUT_SIZE + 1),
            torch.zeros(1, 1, PUBLIC_INPUT_SIZE, dtype=torch.int64),
            torch.full((1, 1, PUBLIC_INPUT_SIZE), float("nan")),
            torch.full((1, 1, PUBLIC_INPUT_SIZE), 1.01),
        )

        for observations in cases:
            with self.subTest(
                shape=tuple(observations.shape), dtype=observations.dtype
            ):
                with self.assertRaises(PublicInputError):
                    self.model.forward_sequence(observations, masks, feedback)

    def test_mapping_mask_requires_exact_keys_exact_bools_and_nonempty(self) -> None:
        legal = {action: action in {"eat", "move_east"} for action in ACTION_NAMES}
        encoded = strict_action_mask_tensor(legal)

        self.assertEqual(encoded.dtype, torch.bool)
        self.assertEqual(tuple(encoded.shape), (ACTION_COUNT,))
        self.assertEqual(int(encoded.sum().item()), 2)

        missing = dict(legal)
        missing.pop("stay")
        non_boolean = dict(legal)
        non_boolean["eat"] = 1
        empty = {action: False for action in ACTION_NAMES}
        for invalid in (missing, non_boolean, empty):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ActionMaskError):
                    strict_action_mask_tensor(invalid)

    def test_tensor_mask_rejects_non_bool_wrong_shape_and_empty_rows(self) -> None:
        observations = self._observations(time_steps=2, batch_size=2)
        feedback = self._feedback(time_steps=2, batch_size=2)
        invalid_masks = (
            torch.ones(2, 2, ACTION_COUNT, dtype=torch.float32),
            torch.ones(2, 2, ACTION_COUNT - 1, dtype=torch.bool),
            torch.zeros(2, 2, ACTION_COUNT, dtype=torch.bool),
        )

        for masks in invalid_masks:
            with self.subTest(shape=tuple(masks.shape), dtype=masks.dtype):
                with self.assertRaises(ActionMaskError):
                    self.model.forward_sequence(observations, masks, feedback)

    def test_previous_feedback_is_typed_bounded_and_zero_at_birth(self) -> None:
        zero = previous_public_feedback_tensor(PreviousPublicFeedbackInput.zero())
        feedback = previous_public_feedback_tensor(
            PreviousPublicFeedbackInput(
                requested_action_id=ACTION_NAMES.index("move_east"),
                resolved_action_id=ACTION_NAMES.index("stay"),
                resolution_action_valid=False,
                moved=False,
                reward_total=-1.0,
            )
        )

        self.assertEqual(tuple(zero.shape), (PREVIOUS_PUBLIC_FEEDBACK_SIZE,))
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        self.assertEqual(
            feedback[ACTION_NAMES.index("move_east")].item(),
            1.0,
        )
        self.assertEqual(
            feedback[ACTION_COUNT + ACTION_NAMES.index("stay")].item(),
            1.0,
        )
        self.assertGreaterEqual(feedback[-1].item(), -1.0)
        self.assertLessEqual(feedback[-1].item(), 1.0)

        with self.assertRaises(RecurrentContextError):
            PreviousPublicFeedbackInput(
                requested_action_id=None,
                resolved_action_id=None,
                resolution_action_valid=True,
                moved=False,
                reward_total=0.0,
            )

    def test_feedback_tensor_rejects_soft_ids_unpaired_ids_and_invalid_movement(
        self,
    ) -> None:
        observations = self._observations(time_steps=1, batch_size=1)
        masks = self._masks(time_steps=1, batch_size=1)
        cases: list[torch.Tensor] = []

        soft = self._feedback(time_steps=1, batch_size=1)
        soft[0, 0, 0] = 0.5
        cases.append(soft)
        unpaired = self._feedback(time_steps=1, batch_size=1)
        unpaired[0, 0, 0] = 1.0
        cases.append(unpaired)
        invalid_movement = self._feedback(time_steps=1, batch_size=1)
        invalid_movement[0, 0, 0] = 1.0
        invalid_movement[0, 0, ACTION_COUNT] = 1.0
        invalid_movement[0, 0, -2] = 1.0
        cases.append(invalid_movement)
        valid_mismatch = self._feedback(time_steps=1, batch_size=1)
        valid_mismatch[0, 0, ACTION_NAMES.index("stay")] = 1.0
        valid_mismatch[
            0,
            0,
            ACTION_COUNT + ACTION_NAMES.index("eat"),
        ] = 1.0
        valid_mismatch[0, 0, -3] = 1.0
        cases.append(valid_mismatch)
        invalid_nonstay = self._feedback(time_steps=1, batch_size=1)
        invalid_nonstay[0, 0, ACTION_NAMES.index("eat")] = 1.0
        invalid_nonstay[
            0,
            0,
            ACTION_COUNT + ACTION_NAMES.index("eat"),
        ] = 1.0
        cases.append(invalid_nonstay)
        valid_move_not_moved = self._feedback(time_steps=1, batch_size=1)
        move_index = ACTION_NAMES.index("move_north")
        valid_move_not_moved[0, 0, move_index] = 1.0
        valid_move_not_moved[0, 0, ACTION_COUNT + move_index] = 1.0
        valid_move_not_moved[0, 0, -3] = 1.0
        cases.append(valid_move_not_moved)

        for feedback in cases:
            with self.subTest(feedback=feedback):
                with self.assertRaises(RecurrentContextError):
                    self.model.forward_sequence(observations, masks, feedback)

    def test_episode_start_requires_zero_previous_feedback(self) -> None:
        observations = self._observations(time_steps=2, batch_size=1)
        masks = self._masks(time_steps=2, batch_size=1)
        feedback = self._feedback(time_steps=2, batch_size=1)
        feedback[1, 0, 0] = 1.0
        feedback[1, 0, ACTION_COUNT] = 1.0
        starts = torch.tensor([[False], [True]], dtype=torch.bool)

        with self.assertRaisesRegex(RecurrentContextError, "episode-start"):
            self.model.forward_sequence(
                observations,
                masks,
                feedback,
                episode_starts=starts,
            )

    def test_encoder_consumes_mask_and_previous_feedback_without_changing_541_payload(
        self,
    ) -> None:
        observations = self._observations(time_steps=1, batch_size=1)
        masks = self._masks(time_steps=1, batch_size=1)
        zero_feedback = self._feedback(time_steps=1, batch_size=1)
        prior_feedback = previous_public_feedback_tensor(
            PreviousPublicFeedbackInput(
                requested_action_id=ACTION_NAMES.index("move_east"),
                resolved_action_id=ACTION_NAMES.index("stay"),
                resolution_action_valid=False,
                moved=False,
                reward_total=-1.0,
            )
        ).reshape(1, 1, -1)

        zero_output = self.model.forward_sequence(
            observations,
            masks,
            zero_feedback,
        )
        prior_output = self.model.forward_sequence(
            observations,
            masks,
            prior_feedback,
        )

        self.assertEqual(self.model.encoder[0].in_features, 604)
        self.assertEqual(observations.shape[-1], 541)
        self.assertFalse(torch.equal(zero_output.raw_logits, prior_output.raw_logits))

    def test_forward_has_shared_recurrent_actor_and_value_shapes(self) -> None:
        observations = self._observations(time_steps=5, batch_size=3)
        masks = self._masks(time_steps=5, batch_size=3)
        feedback = self._feedback(time_steps=5, batch_size=3)

        output = self.model.forward_sequence(observations, masks, feedback)

        self.assertEqual(tuple(output.raw_logits.shape), (5, 3, ACTION_COUNT))
        self.assertEqual(tuple(output.values.shape), (5, 3))
        self.assertEqual(
            tuple(output.final_state.shape),
            (self.model.config.recurrent_layers, 3, self.model.config.hidden_size),
        )
        self.assertTrue(torch.isneginf(output.masked_logits[~masks]).all())
        self.assertTrue(torch.isfinite(output.masked_logits[masks]).all())

    def test_episode_start_reset_matches_fresh_segment_evaluation(self) -> None:
        observations = self._observations(time_steps=6, batch_size=1)
        masks = self._masks(time_steps=6, batch_size=1)
        feedback = self._feedback(time_steps=6, batch_size=1)
        starts = torch.zeros(6, 1, dtype=torch.bool)
        starts[3, 0] = True

        joined = self.model.forward_sequence(
            observations,
            masks,
            feedback,
            initial_state=torch.ones_like(self.model.initial_state(1)),
            episode_starts=starts,
        )
        fresh = self.model.forward_sequence(
            observations[3:],
            masks[3:],
            feedback[3:],
        )

        torch.testing.assert_close(joined.raw_logits[3:], fresh.raw_logits)
        torch.testing.assert_close(joined.values[3:], fresh.values)
        torch.testing.assert_close(joined.final_state, fresh.final_state)

    def test_sequence_evaluation_is_differentiable_and_rejects_illegal_labels(
        self,
    ) -> None:
        observations = self._observations(time_steps=4, batch_size=2)
        masks = self._masks(time_steps=4, batch_size=2)
        feedback = self._feedback(time_steps=4, batch_size=2)
        actions = torch.full((4, 2), ACTION_NAMES.index("eat"), dtype=torch.long)

        evaluation = self.model.evaluate_sequence(
            observations,
            masks,
            feedback,
            actions,
        )
        loss = -evaluation.log_probs.mean() - 0.01 * evaluation.entropy.mean()
        loss = loss + evaluation.values.square().mean()
        loss.backward()

        self.assertEqual(tuple(evaluation.log_probs.shape), (4, 2))
        self.assertEqual(tuple(evaluation.entropy.shape), (4, 2))
        self.assertIsNotNone(self.model.actor.weight.grad)
        self.assertIsNotNone(self.model.recurrent.weight_hh_l0.grad)

        illegal_actions = actions.clone()
        illegal_actions[0, 0] = ACTION_NAMES.index("attack_north")
        with self.assertRaises(ActionMaskError):
            self.model.evaluate_sequence(
                observations,
                masks,
                feedback,
                illegal_actions,
            )

    def test_seeded_sampling_is_repeatable_and_never_escapes_mask(self) -> None:
        observations = self._observations(time_steps=1, batch_size=64).squeeze(0)
        masks = self._masks(time_steps=1, batch_size=64).squeeze(0)
        feedback = self._feedback(time_steps=1, batch_size=64).squeeze(0)

        first = self.model.act(
            observations,
            masks,
            feedback,
            deterministic=False,
            generator=seeded_torch_generator(90210),
        )
        second = self.model.act(
            observations,
            masks,
            feedback,
            deterministic=False,
            generator=seeded_torch_generator(90210),
        )

        torch.testing.assert_close(first.actions, second.actions)
        self.assertTrue(masks.gather(-1, first.actions.unsqueeze(-1)).all())
        with self.assertRaises(RecurrentPolicyContractError):
            self.model.act(observations, masks, feedback, deterministic=False)

    def test_deterministic_evaluation_uses_masked_argmax(self) -> None:
        observations = self._observations(time_steps=1, batch_size=3).squeeze(0)
        only_drink = torch.zeros(3, ACTION_COUNT, dtype=torch.bool)
        only_drink[:, ACTION_NAMES.index("drink")] = True
        feedback = self._feedback(time_steps=1, batch_size=3).squeeze(0)

        selection = self.model.act(
            observations,
            only_drink,
            feedback,
            deterministic=True,
        )

        self.assertEqual(
            selection.actions.tolist(),
            [ACTION_NAMES.index("drink")] * 3,
        )

    def test_initialization_seed_is_repeatable_without_global_rng_side_effect(
        self,
    ) -> None:
        torch.manual_seed(88)
        expected_next = torch.rand(3)
        torch.manual_seed(88)
        first = PublicRecurrentActorCritic(initialization_seed=11)
        actual_next = torch.rand(3)
        second = PublicRecurrentActorCritic(initialization_seed=11)

        torch.testing.assert_close(actual_next, expected_next)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            torch.testing.assert_close(first_parameter, second_parameter)

    def test_per_agent_store_zeroes_birth_and_discards_death_or_world_state(
        self,
    ) -> None:
        store = PerAgentRecurrentStateStore(self.model)
        birth_state = store.state_for(7)
        self.assertTrue(torch.equal(birth_state, torch.zeros_like(birth_state)))

        learned_state = torch.ones_like(birth_state, requires_grad=True)
        store.update(7, learned_state)
        store.update(8, learned_state * 2.0)
        self.assertEqual(set(store.tracked_agent_ids), {7, 8})
        self.assertFalse(store.state_for(7).requires_grad)
        self.assertTrue(store.reset_agent(7))
        self.assertTrue(torch.equal(store.state_for(7), torch.zeros_like(birth_state)))
        self.assertEqual(store.retain_agents([9]), 1)
        self.assertEqual(len(store), 0)

        store.update(10, learned_state)
        store.update(11, learned_state)
        self.assertEqual(store.reset_world(), 2)
        self.assertEqual(len(store), 0)

    def test_recurrent_state_shape_dtype_and_episode_start_contract_fail_closed(
        self,
    ) -> None:
        observations = self._observations(time_steps=2, batch_size=1)
        masks = self._masks(time_steps=2, batch_size=1)
        feedback = self._feedback(time_steps=2, batch_size=1)
        bad_state = torch.zeros(1, 2, self.model.config.hidden_size)
        bad_starts = torch.zeros(2, 1, dtype=torch.float32)

        with self.assertRaises(RecurrentStateError):
            self.model.forward_sequence(
                observations,
                masks,
                feedback,
                initial_state=bad_state,
            )
        with self.assertRaises(RecurrentStateError):
            self.model.forward_sequence(
                observations,
                masks,
                feedback,
                episode_starts=bad_starts,
            )

    @staticmethod
    def _observations(*, time_steps: int, batch_size: int) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, PUBLIC_INPUT_SIZE)
        return values.repeat(time_steps, batch_size, 1)

    @staticmethod
    def _masks(*, time_steps: int, batch_size: int) -> torch.Tensor:
        mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
        for action in ("stay", "eat", "drink", "move_east"):
            mask[ACTION_NAMES.index(action)] = True
        return mask.repeat(time_steps, batch_size, 1)

    @staticmethod
    def _feedback(*, time_steps: int, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            time_steps,
            batch_size,
            PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        )


if __name__ == "__main__":
    unittest.main()
