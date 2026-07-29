from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.config.schema import SignalConfig
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.env.runtime.observations import (
        OBSERVATION_INPUT_VECTOR_SIZE,
        SELF_INPUT_FIELDS,
    )
    from evolution_sim.mind.recurrent_actor_critic import (
        ACTION_COUNT,
        BackendStableGRU,
        BackendStableLayerNorm,
        BackendStableLinear,
        CRITIC_GENOME_CONDITIONING_FILM_V1,
        CRITIC_GENOME_CONDITIONING_NONE,
        GENOME_CONDITIONING_ACTOR_FILM_V1,
        GENOME_CONDITIONING_DISABLED,
        LEARNED_ENCODER_INPUT_SIZE,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        PUBLIC_INPUT_SIZE,
        VALUE_SHARED_TRUNK_GRADIENT_SHARED,
        VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
        ActionMaskError,
        GenomeConditioningError,
        PerAgentRecurrentStateStore,
        PreviousPublicFeedbackInput,
        PublicInputError,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
        RecurrentContextError,
        RecurrentPolicyContractError,
        RecurrentStateError,
        public_policy_tensor_from_decoded,
        previous_public_feedback_tensor,
        recurrent_actor_critic_contract,
        seeded_torch_generator,
        strict_action_mask_tensor,
        _backend_stable_linear,
        _per_row_bmm_linear_forward,
    )
    from evolution_sim.mind.recurrent_genome import (
        RECURRENT_CONTROLLER_GENOME_SIZE,
        RECURRENT_CONTROLLER_VECTORIZED_DEVELOPMENT_ABSOLUTE_TOLERANCE,
        develop_recurrent_genome,
        founder_recurrent_genome,
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
        self.assertEqual(
            contract["architecture"]["numeric_kernel"],
            (
                "batch_size_tolerance_stable_per_row_bmm_forward_"
                "native_gemm_backward_v1"
            ),
        )
        self.assertEqual(
            contract["architecture"]["actor"],
            "backend_stable_linear_20_logits",
        )
        self.assertEqual(
            contract["architecture"]["input_normalization"],
            "explicit_layer_norm",
        )
        self.assertEqual(
            contract["architecture"]["learned_encoder_input_size"],
            604,
        )
        self.assertEqual(
            self.model.config.genome_conditioning_mode,
            GENOME_CONDITIONING_DISABLED,
        )
        self.assertEqual(
            self.model.config.critic_genome_conditioning,
            CRITIC_GENOME_CONDITIONING_NONE,
        )
        self.assertEqual(
            self.model.config.value_shared_trunk_gradient,
            VALUE_SHARED_TRUNK_GRADIENT_SHARED,
        )
        self.assertEqual(
            contract["architecture"]["genome_conditioning"]["required_input"],
            "forbidden",
        )
        self.assertIsNone(
            contract["architecture"]["genome_conditioning"]["development"]
        )
        self.assertFalse(contract["runtime_integrated"])
        self.assertFalse(contract["heuristic_action_source"])

    def test_genome_and_critic_ablation_config_modes_fail_closed(self) -> None:
        enabled = RecurrentActorCriticConfig(
            encoder_size=8,
            hidden_size=8,
            genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_FILM_V1,
            value_shared_trunk_gradient=VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
        )
        contract = recurrent_actor_critic_contract(enabled)

        self.assertEqual(
            contract["architecture"]["genome_conditioning"]["mode"],
            GENOME_CONDITIONING_ACTOR_FILM_V1,
        )
        self.assertEqual(
            contract["architecture"]["genome_conditioning"][
                "critic_genome_conditioning"
            ],
            CRITIC_GENOME_CONDITIONING_FILM_V1,
        )
        self.assertEqual(
            contract["architecture"]["genome_conditioning"][
                "value_shared_trunk_gradient"
            ],
            VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
        )
        self.assertEqual(
            contract["architecture"]["genome_conditioning"]["development"][
                "hidden_dimension"
            ],
            8,
        )

        invalid_cases = (
            {"genome_conditioning_mode": "film"},
            {"critic_genome_conditioning": "actor_film_v1"},
            {"value_shared_trunk_gradient": "separate_critic"},
            {
                "genome_conditioning_mode": GENOME_CONDITIONING_DISABLED,
                "critic_genome_conditioning": (CRITIC_GENOME_CONDITIONING_FILM_V1),
            },
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RecurrentActorCriticConfig(**kwargs)

    def test_token_aware_contract_binds_expanded_public_input_shape(self) -> None:
        config = RecurrentActorCriticConfig.for_signal_config(
            SignalConfig(communication_signal_emission_enabled=True),
            encoder_size=16,
            hidden_size=16,
        )
        model = PublicRecurrentActorCritic(config, initialization_seed=1729)
        contract = recurrent_actor_critic_contract(model.config)

        self.assertEqual(
            config.public_input_schema_version, "mind_ecological_policy_input_v3"
        )
        self.assertEqual(config.public_input_size, 645)
        self.assertEqual(config.learned_encoder_input_size, 708)
        self.assertEqual(
            contract["public_input_schema_version"], "mind_ecological_policy_input_v3"
        )
        self.assertEqual(contract["public_input_size"], 645)
        self.assertEqual(
            contract["architecture"]["learned_encoder_input_size"],
            708,
        )
        self.assertEqual(model.input_norm.normalized_size, 708)
        self.assertEqual(model.encoder[0].in_features, 708)

        decoded = [0.0] * 646
        decoded[len(SELF_INPUT_FIELDS)] = 0.75
        projected = public_policy_tensor_from_decoded(
            decoded,
            expected_schema_version="mind_ecological_policy_input_v3",
            expected_size=645,
        )
        self.assertEqual(tuple(projected.shape), (645,))
        self.assertEqual(
            float(projected[SELF_INPUT_FIELDS.index("mind_inheritance_available")]),
            0.75,
        )

        with self.assertRaisesRegex(ValueError, "unsupported"):
            RecurrentActorCriticConfig(
                public_input_schema_version="mind_ecological_policy_input_v2",
                public_input_size=645,
            )

        with self.assertRaisesRegex(ValueError, "action ordering"):
            RecurrentActorCriticConfig.for_signal_config(
                SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=2,
                )
            )

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

    def test_backend_stable_model_is_tolerance_stable_across_batch_size(
        self,
    ) -> None:
        config = RecurrentActorCriticConfig(
            encoder_size=16,
            hidden_size=16,
            recurrent_layers=2,
            genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        )
        model = PublicRecurrentActorCritic(config, initialization_seed=20260729)
        time_steps = 4
        batch_size = 3
        observations = self._observations(
            time_steps=time_steps,
            batch_size=batch_size,
        )
        observations[:, 1] = torch.roll(observations[:, 1], shifts=17, dims=-1)
        observations[:, 2] = torch.roll(observations[:, 2], shifts=31, dims=-1)
        masks = self._masks(time_steps=time_steps, batch_size=batch_size)
        feedback = self._feedback(time_steps=time_steps, batch_size=batch_size)
        genomes = torch.stack(
            tuple(
                self._founder_genome_tensor(seed=20260729 + row)
                for row in range(batch_size)
            ),
            dim=0,
        ).unsqueeze(0).repeat(time_steps, 1, 1)
        initial_state = torch.linspace(
            -0.5,
            0.5,
            config.recurrent_layers * batch_size * config.hidden_size,
        ).reshape(config.recurrent_layers, batch_size, config.hidden_size)

        batched = model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=genomes,
            initial_state=initial_state,
        )
        scalar = tuple(
            model.forward_sequence(
                observations[:, row : row + 1],
                masks[:, row : row + 1],
                feedback[:, row : row + 1],
                genome_values=genomes[:, row : row + 1],
                initial_state=initial_state[:, row : row + 1],
            )
            for row in range(batch_size)
        )

        torch.testing.assert_close(
            batched.raw_logits,
            torch.cat(tuple(output.raw_logits for output in scalar), dim=1),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            batched.values,
            torch.cat(tuple(output.values for output in scalar), dim=1),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            batched.final_state,
            torch.cat(tuple(output.final_state for output in scalar), dim=1),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_backend_stable_linear_preserves_forward_and_native_gradients(
        self,
    ) -> None:
        torch.manual_seed(20260729)
        for bias in (False, True):
            with self.subTest(bias=bias):
                reference = torch.nn.Linear(13, 7, bias=bias)
                stable = BackendStableLinear(13, 7, bias=bias)
                stable.load_state_dict(reference.state_dict(), strict=True)
                self.assertEqual(
                    tuple(stable.state_dict()),
                    tuple(reference.state_dict()),
                )

                base_inputs = torch.randn(3, 4, 13)
                reference_inputs = base_inputs.clone().requires_grad_(True)
                stable_inputs = base_inputs.clone().requires_grad_(True)
                hook_events: list[str] = []
                pre_handle = stable.register_forward_pre_hook(
                    lambda _module, _inputs: hook_events.append("pre")
                )
                post_handle = stable.register_forward_hook(
                    lambda _module, _inputs, _output: hook_events.append("post")
                )
                try:
                    stable_output = stable(stable_inputs)
                finally:
                    pre_handle.remove()
                    post_handle.remove()
                reference_output = reference(reference_inputs)

                scalar_output = torch.cat(
                    tuple(
                        stable(stable_inputs[:, row : row + 1])
                        for row in range(stable_inputs.shape[1])
                    ),
                    dim=1,
                )
                self.assertTrue(torch.equal(stable_output, scalar_output))
                self.assertEqual(hook_events, ["pre", "post"])

                loss_weights = torch.linspace(
                    -0.75,
                    1.25,
                    stable_output.numel(),
                ).reshape_as(stable_output)
                (stable_output.square() * loss_weights).sum().backward()
                (reference_output.square() * loss_weights).sum().backward()

                torch.testing.assert_close(
                    stable_output,
                    reference_output,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(
                    stable_inputs.grad,
                    reference_inputs.grad,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(
                    stable.weight.grad,
                    reference.weight.grad,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                self.assertTrue(torch.isfinite(stable.weight.grad).all())
                if bias:
                    torch.testing.assert_close(
                        stable.bias.grad,
                        reference.bias.grad,
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    )
                    self.assertTrue(torch.isfinite(stable.bias.grad).all())

                learning_rate = 1.0e-3
                with torch.no_grad():
                    for stable_parameter, reference_parameter in zip(
                        stable.parameters(),
                        reference.parameters(),
                        strict=True,
                    ):
                        stable_parameter.add_(
                            stable_parameter.grad,
                            alpha=-learning_rate,
                        )
                        reference_parameter.add_(
                            reference_parameter.grad,
                            alpha=-learning_rate,
                        )
                        self.assertTrue(torch.isfinite(stable_parameter).all())
                        torch.testing.assert_close(
                            stable_parameter,
                            reference_parameter,
                            rtol=1.0e-5,
                            atol=1.0e-6,
                        )

    def test_backend_stable_linear_has_exact_d04_bucket_row_parity(self) -> None:
        active_row_counts = (1, 2, 3, 5, 9, 17, 33, 64, 65, 129, 257, 319, 320)
        torch.manual_seed(20260729)
        inputs = torch.randn(max(active_row_counts), 31)
        weight = torch.randn(19, 31)
        bias = torch.randn(19)

        for active_rows in active_row_counts:
            with self.subTest(active_rows=active_rows):
                active_inputs = inputs[:active_rows]
                batched = _backend_stable_linear(active_inputs, weight, bias)
                scalar = torch.cat(
                    tuple(
                        _backend_stable_linear(
                            active_inputs[row : row + 1],
                            weight,
                            bias,
                        )
                        for row in range(active_rows)
                    ),
                    dim=0,
                )
                self.assertTrue(torch.equal(batched, scalar))
                torch.testing.assert_close(
                    batched,
                    _per_row_bmm_linear_forward(active_inputs, weight, bias),
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

    def test_backend_stable_linear_keeps_compact_dense_backward_exact(self) -> None:
        torch.manual_seed(20260729)
        inputs = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
        grad_output = torch.linspace(
            -0.75,
            1.25,
            3 * 7,
            dtype=torch.float64,
        ).reshape(3, 7)

        output = _backend_stable_linear(inputs, weight, bias)
        grad_inputs, grad_weight, grad_bias = torch.autograd.grad(
            output,
            (inputs, weight, bias),
            grad_outputs=grad_output,
        )

        self.assertTrue(torch.equal(grad_inputs, grad_output @ weight))
        self.assertTrue(torch.equal(grad_weight, grad_output.T @ inputs))
        self.assertTrue(torch.equal(grad_bias, grad_output.sum(dim=0)))
        for gradient in (grad_inputs, grad_weight, grad_bias):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_backend_stable_linear_rejects_malformed_ranks_and_shapes(self) -> None:
        cases = (
            (torch.tensor(1.0), torch.zeros(2, 1), None),
            (torch.zeros(2, 3), torch.zeros(3), None),
            (torch.zeros(2, 3), torch.zeros(2, 4), None),
            (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(1, 2)),
            (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3)),
        )

        for inputs, weight, bias in cases:
            with self.subTest(
                input_shape=tuple(inputs.shape),
                weight_shape=tuple(weight.shape),
                bias_shape=None if bias is None else tuple(bias.shape),
            ):
                with self.assertRaises(PublicInputError):
                    _backend_stable_linear(inputs, weight, bias)

    def test_backend_stable_linear_keeps_empty_row_shape_and_gradients(self) -> None:
        inputs = torch.empty(0, 3, requires_grad=True)
        weight = torch.randn(2, 3, requires_grad=True)
        bias = torch.randn(2, requires_grad=True)

        output = _backend_stable_linear(inputs, weight, bias)
        self.assertEqual(tuple(output.shape), (0, 2))
        output.sum().backward()

        self.assertEqual(tuple(inputs.grad.shape), (0, 3))
        self.assertTrue(torch.equal(weight.grad, torch.zeros_like(weight)))
        self.assertTrue(torch.equal(bias.grad, torch.zeros_like(bias)))

    def test_backend_stable_two_layer_gru_matches_native_gradients(self) -> None:
        torch.manual_seed(20260729)
        for bias in (False, True):
            with self.subTest(bias=bias):
                reference = torch.nn.GRU(11, 9, num_layers=2, bias=bias)
                stable = BackendStableGRU(11, 9, num_layers=2, bias=bias)
                stable.load_state_dict(reference.state_dict(), strict=True)
                self.assertEqual(
                    tuple(stable.state_dict()),
                    tuple(reference.state_dict()),
                )

                base_inputs = torch.randn(5, 4, 11)
                base_hidden = torch.randn(2, 4, 9)
                reference_inputs = base_inputs.clone().requires_grad_(True)
                stable_inputs = base_inputs.clone().requires_grad_(True)
                reference_hidden = base_hidden.clone().requires_grad_(True)
                stable_hidden = base_hidden.clone().requires_grad_(True)
                hook_events: list[str] = []
                pre_handle = stable.register_forward_pre_hook(
                    lambda _module, _inputs: hook_events.append("pre")
                )
                post_handle = stable.register_forward_hook(
                    lambda _module, _inputs, _output: hook_events.append("post")
                )
                try:
                    stable_output, stable_final = stable(
                        stable_inputs,
                        stable_hidden,
                    )
                finally:
                    pre_handle.remove()
                    post_handle.remove()
                reference_output, reference_final = reference(
                    reference_inputs,
                    reference_hidden,
                )
                self.assertEqual(hook_events, ["pre", "post"])

                output_weights = torch.linspace(
                    -0.5,
                    1.0,
                    stable_output.numel(),
                ).reshape_as(stable_output)
                final_weights = torch.linspace(
                    0.75,
                    -0.25,
                    stable_final.numel(),
                ).reshape_as(stable_final)
                stable_loss = (
                    (stable_output.square() * output_weights).sum()
                    + (stable_final.square() * final_weights).sum()
                )
                reference_loss = (
                    (reference_output.square() * output_weights).sum()
                    + (reference_final.square() * final_weights).sum()
                )
                stable_loss.backward()
                reference_loss.backward()

                for actual, expected in (
                    (stable_output, reference_output),
                    (stable_final, reference_final),
                    (stable_inputs.grad, reference_inputs.grad),
                    (stable_hidden.grad, reference_hidden.grad),
                ):
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    )
                    self.assertTrue(torch.isfinite(actual).all())
                for name, parameter in stable.named_parameters():
                    reference_parameter = dict(reference.named_parameters())[name]
                    torch.testing.assert_close(
                        parameter.grad,
                        reference_parameter.grad,
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    )
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_backend_stable_linear_supports_higher_order_gradients(self) -> None:
        torch.manual_seed(20260729)
        for bias in (False, True):
            with self.subTest(bias=bias):
                inputs = torch.randn(
                    2,
                    3,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                weight = torch.randn(
                    2,
                    3,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                if bias:
                    bias_value = torch.randn(
                        2,
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    function = _backend_stable_linear
                    arguments = (inputs, weight, bias_value)
                else:
                    def function(
                        value: torch.Tensor,
                        projection: torch.Tensor,
                    ) -> torch.Tensor:
                        return _backend_stable_linear(value, projection)

                    arguments = (inputs, weight)
                self.assertTrue(
                    torch.autograd.gradgradcheck(
                        function,
                        arguments,
                        rtol=1.0e-5,
                        atol=1.0e-7,
                    )
                )

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

    def test_actor_film_requires_exact_explicit_genome_tensors(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            ),
            initialization_seed=17,
        )
        observations = self._observations(time_steps=2, batch_size=3)
        masks = self._masks(time_steps=2, batch_size=3)
        feedback = self._feedback(time_steps=2, batch_size=3)

        with self.assertRaisesRegex(GenomeConditioningError, "requires explicit"):
            model.forward_sequence(observations, masks, feedback)
        with self.assertRaisesRegex(GenomeConditioningError, "requires explicit"):
            model.evaluate_sequence(
                observations,
                masks,
                feedback,
                torch.full(
                    (2, 3),
                    ACTION_NAMES.index("eat"),
                    dtype=torch.long,
                ),
            )

        invalid_cases: tuple[tuple[str, object], ...] = (
            ("shape", torch.zeros(2, 3, RECURRENT_CONTROLLER_GENOME_SIZE - 1)),
            ("shape", torch.zeros(3, RECURRENT_CONTROLLER_GENOME_SIZE)),
            (
                "dtype",
                torch.zeros(
                    2,
                    3,
                    RECURRENT_CONTROLLER_GENOME_SIZE,
                    dtype=torch.float64,
                ),
            ),
            (
                "finite",
                torch.full(
                    (2, 3, RECURRENT_CONTROLLER_GENOME_SIZE),
                    float("nan"),
                ),
            ),
            (
                "finite",
                torch.full(
                    (2, 3, RECURRENT_CONTROLLER_GENOME_SIZE),
                    float("inf"),
                ),
            ),
            (
                "bounds",
                torch.full(
                    (2, 3, RECURRENT_CONTROLLER_GENOME_SIZE),
                    1.0001,
                ),
            ),
            (
                "bounds",
                torch.full(
                    (2, 3, RECURRENT_CONTROLLER_GENOME_SIZE),
                    -1.0001,
                ),
            ),
            (
                "device",
                torch.zeros(
                    2,
                    3,
                    RECURRENT_CONTROLLER_GENOME_SIZE,
                    device="meta",
                ),
            ),
            ("torch.Tensor", object()),
        )
        for expected, genome_values in invalid_cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(GenomeConditioningError, expected):
                    model.forward_sequence(
                        observations,
                        masks,
                        feedback,
                        genome_values=genome_values,  # type: ignore[arg-type]
                    )

        single_observation = observations[0, 0]
        single_mask = masks[0, 0]
        single_feedback = feedback[0, 0]
        with self.assertRaisesRegex(GenomeConditioningError, "requires explicit"):
            model.act(
                single_observation,
                single_mask,
                single_feedback,
                deterministic=True,
            )
        with self.assertRaisesRegex(GenomeConditioningError, "act genome"):
            model.act(
                single_observation,
                single_mask,
                single_feedback,
                genome_values=torch.zeros(1, RECURRENT_CONTROLLER_GENOME_SIZE),
                deterministic=True,
            )
        selection = model.act(
            single_observation,
            single_mask,
            single_feedback,
            genome_values=self._founder_genome_tensor(seed=17),
            deterministic=True,
        )
        self.assertEqual(tuple(selection.raw_logits.shape), (1, ACTION_COUNT))

        batched_selection = model.act(
            observations[0],
            masks[0],
            feedback[0],
            genome_values=self._genomes(time_steps=1, batch_size=3).squeeze(0),
            deterministic=True,
        )
        self.assertEqual(tuple(batched_selection.raw_logits.shape), (3, ACTION_COUNT))
        evaluation = model.evaluate_sequence(
            observations,
            masks,
            feedback,
            torch.full(
                (2, 3),
                ACTION_NAMES.index("eat"),
                dtype=torch.long,
            ),
            genome_values=self._genomes(time_steps=2, batch_size=3, seed=18),
        )
        self.assertEqual(tuple(evaluation.log_probs.shape), (2, 3))

    def test_zero_genome_is_neutral_and_swap_changes_actor_only(self) -> None:
        disabled = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=16, hidden_size=16),
            initialization_seed=701,
        )
        conditioned = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=16,
                hidden_size=16,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
                critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
            ),
            initialization_seed=701,
        )
        observations = self._observations(time_steps=3, batch_size=2)
        masks = self._masks(time_steps=3, batch_size=2)
        feedback = self._feedback(time_steps=3, batch_size=2)
        zero_genomes = self._genomes(time_steps=3, batch_size=2)
        nonzero_genomes = self._genomes(
            time_steps=3,
            batch_size=2,
            seed=702,
        )

        for left, right in zip(
            disabled.parameters(),
            conditioned.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(
            any("genome" in name for name, _ in conditioned.named_parameters())
        )
        self.assertEqual(
            {name for name, _ in conditioned.named_buffers() if "genome" in name},
            {
                "_genome_film_scale_coefficients",
                "_genome_film_bias_coefficients",
            },
        )

        baseline = disabled.forward_sequence(observations, masks, feedback)
        neutral = conditioned.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=zero_genomes,
        )
        swapped = conditioned.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=nonzero_genomes,
        )

        self.assertTrue(torch.equal(neutral.raw_logits, baseline.raw_logits))
        self.assertTrue(torch.equal(neutral.values, baseline.values))
        self.assertTrue(torch.equal(neutral.final_state, baseline.final_state))
        self.assertFalse(torch.equal(swapped.raw_logits, neutral.raw_logits))
        self.assertTrue(torch.equal(swapped.values, neutral.values))
        self.assertTrue(torch.equal(swapped.final_state, neutral.final_state))

    def test_vectorized_film_matches_scalar_sha_development(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=7,
                hidden_size=7,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            ),
            initialization_seed=31,
        ).double()
        scalar_genomes = (
            founder_recurrent_genome(seed=31),
            founder_recurrent_genome(seed=32),
        )
        genomes = torch.tensor(
            [genome.values for genome in scalar_genomes],
            dtype=torch.float64,
        ).reshape(1, 2, RECURRENT_CONTROLLER_GENOME_SIZE)

        scale, bias = model._develop_genome_film(genomes)

        for row, genome in enumerate(scalar_genomes):
            developed = develop_recurrent_genome(genome, hidden_dimension=7)
            torch.testing.assert_close(
                scale[0, row],
                torch.tensor(developed.scale, dtype=torch.float64),
                rtol=0.0,
                atol=(RECURRENT_CONTROLLER_VECTORIZED_DEVELOPMENT_ABSOLUTE_TOLERANCE),
            )
            torch.testing.assert_close(
                bias[0, row],
                torch.tensor(developed.bias, dtype=torch.float64),
                rtol=0.0,
                atol=(RECURRENT_CONTROLLER_VECTORIZED_DEVELOPMENT_ABSOLUTE_TOLERANCE),
            )

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

    def test_actor_and_unconditioned_critic_gradient_routes_are_distinct(
        self,
    ) -> None:
        config = RecurrentActorCriticConfig(
            encoder_size=12,
            hidden_size=12,
            genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
        )
        observations = self._observations(time_steps=3, batch_size=2)
        masks = self._masks(time_steps=3, batch_size=2)
        feedback = self._feedback(time_steps=3, batch_size=2)

        actor_model = PublicRecurrentActorCritic(
            config,
            initialization_seed=810,
        )
        actor_genomes = self._genomes(
            time_steps=3,
            batch_size=2,
            seed=811,
        ).requires_grad_()
        actor_output = actor_model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=actor_genomes,
        )
        actor_output.raw_logits[..., ACTION_NAMES.index("eat")].sum().backward()

        self.assertIsNone(actor_genomes.grad)
        self.assertIsNotNone(actor_model.recurrent.weight_ih_l0.grad)
        self.assertGreater(
            float(actor_model.recurrent.weight_ih_l0.grad.abs().sum().item()),
            0.0,
        )
        self.assertIsNone(actor_model.value.weight.grad)

        value_model = PublicRecurrentActorCritic(
            config,
            initialization_seed=810,
        )
        value_genomes = self._genomes(
            time_steps=3,
            batch_size=2,
            seed=811,
        ).requires_grad_()
        value_output = value_model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=value_genomes,
        )
        value_output.values.sum().backward()

        self.assertIsNone(value_genomes.grad)
        self.assertIsNotNone(value_model.value.weight.grad)
        self.assertGreater(
            float(value_model.value.weight.grad.abs().sum().item()),
            0.0,
        )
        self.assertIsNotNone(value_model.recurrent.weight_ih_l0.grad)
        self.assertGreater(
            float(value_model.recurrent.weight_ih_l0.grad.abs().sum().item()),
            0.0,
        )
        self.assertIsNone(value_model.actor.weight.grad)

    def test_critic_film_and_stop_gradient_route_value_loss_exactly(self) -> None:
        observations = self._observations(time_steps=3, batch_size=2)
        masks = self._masks(time_steps=3, batch_size=2)
        feedback = self._feedback(time_steps=3, batch_size=2)

        shared_model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=12,
                hidden_size=12,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
                critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_FILM_V1,
                value_shared_trunk_gradient=VALUE_SHARED_TRUNK_GRADIENT_SHARED,
            ),
            initialization_seed=820,
        )
        shared_genomes = self._genomes(
            time_steps=3,
            batch_size=2,
            seed=821,
        ).requires_grad_()
        shared_output = shared_model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=shared_genomes,
        )
        shared_output.values.sum().backward()

        self.assertIsNone(shared_genomes.grad)
        self.assertIsNotNone(shared_model.recurrent.weight_ih_l0.grad)
        self.assertGreater(
            float(shared_model.recurrent.weight_ih_l0.grad.abs().sum().item()),
            0.0,
        )
        self.assertIsNotNone(shared_model.value.weight.grad)
        self.assertIsNotNone(shared_model.critic_genome_film_scale_coefficients.grad)
        self.assertGreater(
            float(
                shared_model.critic_genome_film_scale_coefficients.grad.abs()
                .sum()
                .item()
            ),
            0.0,
        )
        self.assertIsNotNone(shared_model.critic_genome_film_bias_coefficients.grad)
        self.assertGreater(
            float(
                shared_model.critic_genome_film_bias_coefficients.grad.abs()
                .sum()
                .item()
            ),
            0.0,
        )

        stopped_model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=12,
                hidden_size=12,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
                critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_FILM_V1,
                value_shared_trunk_gradient=(VALUE_SHARED_TRUNK_GRADIENT_STOP_V1),
            ),
            initialization_seed=820,
        )
        stopped_genomes = self._genomes(
            time_steps=3,
            batch_size=2,
            seed=821,
        ).requires_grad_()
        stopped_output = stopped_model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=stopped_genomes,
        )
        stopped_output.values.sum().backward()

        self.assertIsNone(stopped_genomes.grad)
        self.assertIsNone(stopped_model.encoder[0].weight.grad)
        self.assertIsNone(stopped_model.recurrent.weight_ih_l0.grad)
        self.assertIsNotNone(stopped_model.value.weight.grad)
        self.assertGreater(
            float(stopped_model.value.weight.grad.abs().sum().item()),
            0.0,
        )
        self.assertIsNotNone(stopped_model.critic_genome_film_scale_coefficients.grad)
        self.assertGreater(
            float(
                stopped_model.critic_genome_film_scale_coefficients.grad.abs()
                .sum()
                .item()
            ),
            0.0,
        )
        self.assertIsNotNone(stopped_model.critic_genome_film_bias_coefficients.grad)
        self.assertGreater(
            float(
                stopped_model.critic_genome_film_bias_coefficients.grad.abs()
                .sum()
                .item()
            ),
            0.0,
        )

        stopped_model.zero_grad(set_to_none=True)
        stopped_genomes.grad = None
        stopped_actor_output = stopped_model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=stopped_genomes,
        )
        stopped_actor_output.raw_logits[..., ACTION_NAMES.index("eat")].sum().backward()
        self.assertIsNone(stopped_genomes.grad)
        self.assertIsNotNone(stopped_model.recurrent.weight_ih_l0.grad)
        self.assertGreater(
            float(stopped_model.recurrent.weight_ih_l0.grad.abs().sum().item()),
            0.0,
        )
        self.assertIsNone(stopped_model.value.weight.grad)
        self.assertIsNone(stopped_model.critic_genome_film_scale_coefficients.grad)
        self.assertIsNone(stopped_model.critic_genome_film_bias_coefficients.grad)

        disabled_stopped_model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=12,
                hidden_size=12,
                value_shared_trunk_gradient=(VALUE_SHARED_TRUNK_GRADIENT_STOP_V1),
            ),
            initialization_seed=820,
        )
        disabled_stopped_output = disabled_stopped_model.forward_sequence(
            observations,
            masks,
            feedback,
        )
        disabled_stopped_output.values.sum().backward()
        self.assertIsNone(disabled_stopped_model.recurrent.weight_ih_l0.grad)
        self.assertIsNotNone(disabled_stopped_model.value.weight.grad)

    def test_critic_film_is_orthogonal_to_actor_film(self) -> None:
        actor_only = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=12,
                hidden_size=12,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
                critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
            ),
            initialization_seed=830,
        )
        actor_and_critic = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=12,
                hidden_size=12,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
                critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_FILM_V1,
            ),
            initialization_seed=830,
        )
        observations = self._observations(time_steps=3, batch_size=2)
        masks = self._masks(time_steps=3, batch_size=2)
        feedback = self._feedback(time_steps=3, batch_size=2)
        genomes = self._genomes(time_steps=3, batch_size=2, seed=831)
        zero_genomes = self._genomes(time_steps=3, batch_size=2)

        actor_only_output = actor_only.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=genomes,
        )
        actor_and_critic_output = actor_and_critic.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=genomes,
        )
        actor_only_zero = actor_only.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=zero_genomes,
        )
        actor_and_critic_zero = actor_and_critic.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=zero_genomes,
        )

        self.assertTrue(
            torch.equal(
                actor_only_output.raw_logits,
                actor_and_critic_output.raw_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                actor_only_output.final_state,
                actor_and_critic_output.final_state,
            )
        )
        self.assertFalse(
            torch.equal(actor_only_output.values, actor_and_critic_output.values)
        )
        self.assertTrue(
            torch.equal(actor_only_zero.values, actor_and_critic_zero.values)
        )

    def test_phase_a_cells_share_actor_backbone_and_critic_substream_tensors(
        self,
    ) -> None:
        common = {
            "encoder_size": 12,
            "hidden_size": 12,
            "genome_conditioning_mode": GENOME_CONDITIONING_ACTOR_FILM_V1,
        }
        cells = (
            PublicRecurrentActorCritic(
                RecurrentActorCriticConfig(
                    **common,
                    critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
                    value_shared_trunk_gradient=(VALUE_SHARED_TRUNK_GRADIENT_SHARED),
                ),
                initialization_seed=832,
            ),
            PublicRecurrentActorCritic(
                RecurrentActorCriticConfig(
                    **common,
                    critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
                    value_shared_trunk_gradient=(VALUE_SHARED_TRUNK_GRADIENT_STOP_V1),
                ),
                initialization_seed=832,
            ),
            PublicRecurrentActorCritic(
                RecurrentActorCriticConfig(
                    **common,
                    critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_FILM_V1,
                    value_shared_trunk_gradient=(VALUE_SHARED_TRUNK_GRADIENT_SHARED),
                ),
                initialization_seed=832,
            ),
            PublicRecurrentActorCritic(
                RecurrentActorCriticConfig(
                    **common,
                    critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_FILM_V1,
                    value_shared_trunk_gradient=(VALUE_SHARED_TRUNK_GRADIENT_STOP_V1),
                ),
                initialization_seed=832,
            ),
        )
        critic_parameter_names = {
            "critic_genome_film_scale_coefficients",
            "critic_genome_film_bias_coefficients",
        }
        shared_names = set(cells[0].state_dict())
        self.assertEqual(shared_names, set(cells[1].state_dict()))
        self.assertEqual(
            shared_names,
            set(cells[2].state_dict()) - critic_parameter_names,
        )
        self.assertEqual(
            shared_names,
            set(cells[3].state_dict()) - critic_parameter_names,
        )
        for name in shared_names:
            baseline = cells[0].state_dict()[name]
            self.assertTrue(
                all(
                    torch.equal(baseline, cell.state_dict()[name]) for cell in cells[1:]
                )
            )
        for name in critic_parameter_names:
            self.assertTrue(
                torch.equal(cells[2].state_dict()[name], cells[3].state_dict()[name])
            )
            self.assertFalse(
                torch.equal(
                    cells[2].state_dict()[name],
                    torch.zeros_like(cells[2].state_dict()[name]),
                )
            )
        self.assertFalse(
            any(
                name.startswith("critic_genome_film")
                for name, _ in cells[0].named_parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in cells[2].named_parameters()
                if name in critic_parameter_names
            )
        )

    def test_token_aware_actor_film_sequence_and_act_are_compatible(self) -> None:
        config = RecurrentActorCriticConfig.for_signal_config(
            SignalConfig(communication_signal_emission_enabled=True),
            encoder_size=12,
            hidden_size=12,
            genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        )
        model = PublicRecurrentActorCritic(config, initialization_seed=840)
        observations = torch.linspace(
            -1.0,
            1.0,
            config.public_input_size,
        ).repeat(2, 2, 1)
        masks = self._masks(time_steps=2, batch_size=2)
        feedback = self._feedback(time_steps=2, batch_size=2)
        genomes = self._genomes(time_steps=2, batch_size=2, seed=841)

        output = model.forward_sequence(
            observations,
            masks,
            feedback,
            genome_values=genomes,
        )
        selection = model.act(
            observations[0],
            masks[0],
            feedback[0],
            genome_values=genomes[0],
            deterministic=True,
        )

        self.assertEqual(config.public_input_size, 645)
        self.assertEqual(tuple(output.raw_logits.shape), (2, 2, ACTION_COUNT))
        self.assertEqual(tuple(output.values.shape), (2, 2))
        self.assertEqual(tuple(selection.actions.shape), (2,))

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

    def test_disabled_conditioning_is_rng_neutral_and_bit_identical(self) -> None:
        default_model = PublicRecurrentActorCritic(initialization_seed=606)
        explicit_disabled = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                genome_conditioning_mode=GENOME_CONDITIONING_DISABLED,
                critic_genome_conditioning=CRITIC_GENOME_CONDITIONING_NONE,
                value_shared_trunk_gradient=VALUE_SHARED_TRUNK_GRADIENT_SHARED,
            ),
            initialization_seed=606,
        )
        observations = self._observations(time_steps=3, batch_size=2)
        masks = self._masks(time_steps=3, batch_size=2)
        feedback = self._feedback(time_steps=3, batch_size=2)

        self.assertEqual(
            tuple(default_model.state_dict()),
            tuple(explicit_disabled.state_dict()),
        )
        self.assertFalse(
            any("genome" in name for name, _ in default_model.named_parameters())
        )
        self.assertFalse(
            any("genome" in name for name, _ in default_model.named_buffers())
        )
        for left, right in zip(
            default_model.parameters(),
            explicit_disabled.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(left, right))

        default_output = default_model.forward_sequence(
            observations,
            masks,
            feedback,
        )
        explicit_output = explicit_disabled.forward_sequence(
            observations,
            masks,
            feedback,
        )
        self.assertTrue(
            torch.equal(default_output.raw_logits, explicit_output.raw_logits)
        )
        self.assertTrue(torch.equal(default_output.values, explicit_output.values))
        self.assertTrue(
            torch.equal(default_output.final_state, explicit_output.final_state)
        )
        with self.assertRaisesRegex(GenomeConditioningError, "forbidden"):
            default_model.forward_sequence(
                observations,
                masks,
                feedback,
                genome_values=self._genomes(time_steps=3, batch_size=2),
            )
        with self.assertRaisesRegex(GenomeConditioningError, "forbidden"):
            default_model.act(
                observations[0],
                masks[0],
                feedback[0],
                genome_values=self._genomes(
                    time_steps=1,
                    batch_size=2,
                ).squeeze(0),
                deterministic=True,
            )

        torch.manual_seed(911)
        PublicRecurrentActorCritic()
        default_next = torch.rand(4)
        torch.manual_seed(911)
        PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                genome_conditioning_mode=GENOME_CONDITIONING_DISABLED
            )
        )
        explicit_next = torch.rand(4)
        self.assertTrue(torch.equal(default_next, explicit_next))

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
    def _founder_genome_tensor(*, seed: int) -> torch.Tensor:
        return torch.tensor(
            founder_recurrent_genome(seed=seed).values,
            dtype=torch.float32,
        )

    @classmethod
    def _genomes(
        cls,
        *,
        time_steps: int,
        batch_size: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        values = (
            torch.zeros(RECURRENT_CONTROLLER_GENOME_SIZE)
            if seed is None
            else cls._founder_genome_tensor(seed=seed)
        )
        return values.repeat(time_steps, batch_size, 1)

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
