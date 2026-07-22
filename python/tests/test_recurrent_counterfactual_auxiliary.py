from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_actor_critic import (
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
        CounterfactualHorizonScalarization,
        RECURRENT_COUNTERFACTUAL_TERMINAL_VALUE_TARGET,
        RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS,
        RecurrentCounterfactualAuxiliaryConfig,
        RecurrentCounterfactualAuxiliaryError,
        RecurrentCounterfactualAuxiliaryStepConfig,
        build_recurrent_counterfactual_auxiliary_target,
        recurrent_counterfactual_auxiliary_batch_loss,
        recurrent_counterfactual_auxiliary_loss,
    )
    from evolution_sim.mind.recurrent_counterfactual_branch import (
        build_recurrent_counterfactual_branch_row,
    )
    from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
    from evolution_sim.mind.recurrent_ppo import RecurrentPPOTrainer
    from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentCounterfactualAuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        cls.artifact_digest = "c" * 64
        cls.model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=16,
                hidden_size=16,
                recurrent_layers=1,
            ),
            initialization_seed=71,
        )
        common = {
            "artifact_digest": cls.artifact_digest,
            "seed_role": "curriculum",
            "environment_seed": RECURRENT_SEED_REGISTRY["curriculum"][0],
            "scenario": "carrion_only",
            "branch_tick": 2,
            "policy_sampling_seed": 827,
            "gamma": 0.99,
            "verify_replay": True,
        }
        cls.real_horizon_1 = build_recurrent_counterfactual_branch_row(
            cls.model,
            horizon_ticks=1,
            **common,
        )
        cls.real_horizon_2 = build_recurrent_counterfactual_branch_row(
            cls.model,
            horizon_ticks=2,
            **common,
        )
        alternate_common = dict(common)
        alternate_common["branch_tick"] = 1
        cls.alternate_horizon_2 = build_recurrent_counterfactual_branch_row(
            cls.model,
            horizon_ticks=2,
            **alternate_common,
        )

    def test_one_real_exact_branch_row_builds_finite_masked_soft_target(self) -> None:
        config = _config_for_horizons((2,))
        target = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [self.real_horizon_2],
            artifact_digest=self.artifact_digest,
            config=config,
        )

        self.assertEqual(
            target.source_model_state_sha256,
            recurrent_model_state_sha256(self.model),
        )
        self.assertEqual(target.horizons, (2,))
        self.assertEqual(
            target.row_exact_digests,
            (self.real_horizon_2["exact_digest"],),
        )
        self.assertTrue(torch.isfinite(target.scalarized_action_values).all())
        self.assertTrue(torch.isfinite(target.behavior_centered_advantages).all())
        self.assertTrue(torch.isfinite(target.soft_policy_target).all())
        self.assertAlmostEqual(float(target.soft_policy_target.sum()), 1.0, places=6)
        self.assertTrue(
            torch.equal(
                target.soft_policy_target[~target.action_mask],
                torch.zeros_like(target.soft_policy_target[~target.action_mask]),
            )
        )
        self.assertFalse(target.contract["ppo_ratio_data_use"])
        self.assertEqual(
            target.contract["branch_label_statistical_semantics"],
            "single_exact_rollout_target_not_expected_causal_effect",
        )
        self.assertFalse(target.contract["branch_outcome_uncertainty_estimated"])
        self.assertFalse(target.contract["optimizer_step_performed"])
        self.assertFalse(target.contract["runtime_integrated"])

    def test_synthetic_horizon_vector_matches_exact_soft_improvement_math(self) -> None:
        first = _with_synthetic_returns(self.real_horizon_1, multiplier=1.0)
        second = _with_synthetic_returns(self.real_horizon_2, multiplier=2.0)
        config = RecurrentCounterfactualAuxiliaryConfig(
            scalarization=CounterfactualHorizonScalarization(
                horizon_weights=((1, 0.25), (2, 0.75)),
            ),
            temperature=0.7,
            advantage_clip=100.0,
            behavior_kl_coefficient=0.05,
        )
        parameter_copies = [
            parameter.detach().clone() for parameter in self.model.parameters()
        ]
        parameter_versions = tuple(
            parameter._version for parameter in self.model.parameters()
        )
        gradients_before = [parameter.grad for parameter in self.model.parameters()]

        target = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [second, first],
            artifact_digest=self.artifact_digest,
            config=config,
        )

        valid_indices = torch.nonzero(target.action_mask, as_tuple=False).flatten()
        expected_values = torch.zeros_like(target.scalarized_action_values)
        for ordinal, action_index in enumerate(valid_indices.tolist()):
            expected_values[action_index] = 0.25 * ordinal + 0.75 * (2.0 * ordinal)
        torch.testing.assert_close(
            target.scalarized_action_values,
            expected_values,
            rtol=0.0,
            atol=1.0e-6,
        )
        baseline = torch.sum(target.behavior_probabilities * expected_values)
        expected_advantages = torch.where(
            target.action_mask,
            expected_values - baseline,
            torch.zeros_like(expected_values),
        )
        torch.testing.assert_close(
            target.behavior_centered_advantages,
            expected_advantages,
            rtol=0.0,
            atol=1.0e-6,
        )
        expected_unnormalized = target.behavior_probabilities * torch.exp(
            expected_advantages / 0.7
        )
        expected_target = expected_unnormalized / expected_unnormalized.sum()
        torch.testing.assert_close(
            target.soft_policy_target,
            expected_target,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        self.assertGreater(
            int((target.soft_policy_target > 0.0).sum().item()),
            1,
        )
        for observed, expected in zip(
            self.model.parameters(),
            parameter_copies,
            strict=True,
        ):
            torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        self.assertEqual(
            tuple(parameter._version for parameter in self.model.parameters()),
            parameter_versions,
        )
        self.assertEqual(
            [parameter.grad for parameter in self.model.parameters()],
            gradients_before,
        )

    def test_auxiliary_loss_is_finite_and_reaches_actor_encoder_and_recurrent(
        self,
    ) -> None:
        synthetic = _with_synthetic_returns(
            self.real_horizon_2,
            multiplier=1.0,
        )
        model = deepcopy(self.model)
        model.zero_grad(set_to_none=True)
        result = recurrent_counterfactual_auxiliary_loss(
            model,
            [synthetic],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreater(float(result.policy_improvement_kl.detach()), 0.0)
        self.assertAlmostEqual(float(result.behavior_kl.detach()), 0.0, places=6)
        self.assertEqual(float(result.value_loss.detach()), 0.0)
        result.loss.backward()

        self.assertGreater(float(model.actor.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.recurrent.weight_ih_l0.grad.abs().sum()), 0.0)
        self.assertIsNone(model.value.weight.grad)

    def test_nonempty_public_prefix_remains_in_the_differentiable_graph(self) -> None:
        synthetic = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        prefix_length = synthetic["trainable_public_context"]["public_history_prefix"][
            "record_count"
        ]
        self.assertGreater(prefix_length, 0)
        model = deepcopy(self.model)
        differentiable_encoder_outputs: list[torch.Tensor] = []

        def capture_encoder_output(_module, _inputs, output) -> None:
            if output.requires_grad:
                output.retain_grad()
                differentiable_encoder_outputs.append(output)

        handle = model.encoder[0].register_forward_hook(capture_encoder_output)
        try:
            result = recurrent_counterfactual_auxiliary_loss(
                model,
                [synthetic],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((2,)),
            )
            result.loss.backward()
        finally:
            handle.remove()

        self.assertEqual(
            len(differentiable_encoder_outputs),
            prefix_length + 1,
        )
        self.assertIsNotNone(differentiable_encoder_outputs[0].grad)
        self.assertGreater(
            float(differentiable_encoder_outputs[0].grad.abs().sum()),
            0.0,
        )

    def test_variable_prefix_batch_matches_independent_scalar_targets_and_losses(
        self,
    ) -> None:
        first = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        second = _with_synthetic_returns(
            self.alternate_horizon_2,
            multiplier=-0.5,
        )
        config = _config_for_horizons((2,))
        model = deepcopy(self.model)
        first_scalar = recurrent_counterfactual_auxiliary_loss(
            model,
            [first],
            artifact_digest=self.artifact_digest,
            config=config,
        )
        second_scalar = recurrent_counterfactual_auxiliary_loss(
            model,
            [second],
            artifact_digest=self.artifact_digest,
            config=config,
        )
        batch = recurrent_counterfactual_auxiliary_batch_loss(
            model,
            [[first], [second]],
            artifact_digest=self.artifact_digest,
            config=config,
        )

        self.assertNotEqual(
            first["trainable_public_context"]["public_history_prefix"]["record_count"],
            second["trainable_public_context"]["public_history_prefix"]["record_count"],
        )
        torch.testing.assert_close(
            batch.targets[0].soft_policy_target,
            first_scalar.target.soft_policy_target,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            batch.targets[1].soft_policy_target,
            second_scalar.target.soft_policy_target,
            rtol=0.0,
            atol=0.0,
        )
        expected_loss = torch.stack([first_scalar.loss, second_scalar.loss]).mean()
        torch.testing.assert_close(batch.loss, expected_loss, rtol=0.0, atol=0.0)

    def test_deterministic_valid_action_target_permutation_is_explicit_control(
        self,
    ) -> None:
        row = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        row_before = deepcopy(row)
        normal = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [row],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
        )
        control_config = RecurrentCounterfactualAuxiliaryConfig(
            scalarization=CounterfactualHorizonScalarization(
                horizon_weights=((2, 1.0),),
            ),
            advantage_clip=100.0,
            target_permutation_mode=(
                RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
            ),
            target_permutation_seed=991,
        )
        first = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [row],
            artifact_digest=self.artifact_digest,
            config=control_config,
        )
        second = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [row],
            artifact_digest=self.artifact_digest,
            config=control_config,
        )

        self.assertEqual(row, row_before)
        torch.testing.assert_close(
            first.scalarized_action_values,
            second.scalarized_action_values,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            torch.equal(
                first.scalarized_action_values,
                normal.scalarized_action_values,
            )
        )
        torch.testing.assert_close(
            torch.sort(first.scalarized_action_values[first.action_mask]).values,
            torch.sort(normal.scalarized_action_values[normal.action_mask]).values,
            rtol=0.0,
            atol=0.0,
        )
        control = first.contract["scientific_negative_control"]
        self.assertTrue(control["enabled"])
        self.assertFalse(control["branch_rows_mutated"])
        self.assertFalse(control["exact_branch_labels_claimed_after_permutation"])

    def test_negative_control_permutation_is_context_keyed_not_positional_code(
        self,
    ) -> None:
        first_row = _with_synthetic_returns(
            self.real_horizon_2,
            multiplier=1.0,
        )
        second_row = _with_synthetic_returns(
            self.alternate_horizon_2,
            multiplier=1.0,
        )
        config = RecurrentCounterfactualAuxiliaryConfig(
            scalarization=CounterfactualHorizonScalarization(
                horizon_weights=((2, 1.0),),
            ),
            advantage_clip=100.0,
            target_permutation_mode=(
                RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
            ),
            target_permutation_seed=991,
        )

        first = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [first_row],
            artifact_digest=self.artifact_digest,
            config=config,
        )
        second = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            [second_row],
            artifact_digest=self.artifact_digest,
            config=config,
        )

        self.assertTrue(torch.equal(first.action_mask, second.action_mask))
        first_control = first.contract["scientific_negative_control"]
        second_control = second.contract["scientific_negative_control"]
        self.assertNotEqual(
            first_control["group_permutation_identity_sha256"],
            second_control["group_permutation_identity_sha256"],
        )
        self.assertFalse(
            first_control["same_positional_permutation_reused_for_every_group"]
        )
        self.assertFalse(
            torch.equal(
                first.scalarized_action_values[first.action_mask],
                second.scalarized_action_values[second.action_mask],
            )
        )

    def test_transaction_accepts_once_and_accepted_rows_are_stale(self) -> None:
        row = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        trainer = RecurrentPPOTrainer(deepcopy(self.model))
        base_learning_rates = tuple(
            group["lr"] for group in trainer.optimizer.param_groups
        )

        diagnostics = trainer.counterfactual_auxiliary_update(
            [[row]],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
        )

        self.assertTrue(diagnostics.accepted)
        self.assertEqual(diagnostics.optimizer_step_count, 1)
        self.assertEqual(diagnostics.backward_pass_count, 1)
        self.assertEqual(diagnostics.auxiliary_update_count, 1)
        self.assertEqual(trainer.counterfactual_auxiliary_update_count, 1)
        self.assertNotEqual(
            diagnostics.pre_model_state_sha256,
            diagnostics.final_model_state_sha256,
        )
        self.assertTrue(diagnostics.rows_stale_after_step)
        self.assertFalse(diagnostics.rows_exact_model_valid_after_step)
        self.assertFalse(diagnostics.retry_authorized)
        self.assertLessEqual(
            diagnostics.mean_behavior_kl_old_to_post,
            diagnostics.mean_behavior_kl_limit,
        )
        self.assertLessEqual(
            diagnostics.max_state_behavior_kl_old_to_post,
            diagnostics.max_state_behavior_kl_limit,
        )
        self.assertEqual(
            tuple(group["lr"] for group in trainer.optimizer.param_groups),
            base_learning_rates,
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "already attempted",
        ):
            trainer.counterfactual_auxiliary_update(
                [[row]],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((2,)),
            )
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "stale for the current model",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                trainer.model,
                [row],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((2,)),
            )

    def test_forced_kl_rejection_restores_model_adam_lrs_and_counters_exactly(
        self,
    ) -> None:
        row = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        trainer = RecurrentPPOTrainer(deepcopy(self.model))
        trainer.optimizer.zero_grad(set_to_none=True)
        zero_loss = sum(
            parameter.square().sum() * 0.0 for parameter in trainer.model.parameters()
        )
        zero_loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        model_before = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        optimizer_before = deepcopy(trainer.optimizer.state_dict())
        learning_rates_before = tuple(
            deepcopy(group["lr"]) for group in trainer.optimizer.param_groups
        )

        diagnostics = trainer.counterfactual_auxiliary_update(
            [[row]],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
            step_config=RecurrentCounterfactualAuxiliaryStepConfig(
                learning_rate_multiplier=100.0,
                mean_behavior_kl_limit=0.0,
                max_state_behavior_kl_limit=0.0,
            ),
        )

        self.assertFalse(diagnostics.accepted)
        self.assertTrue(diagnostics.rollback_performed)
        self.assertTrue(diagnostics.optimizer_state_restored)
        self.assertTrue(diagnostics.parameter_group_learning_rates_restored)
        self.assertTrue(diagnostics.update_counters_restored)
        self.assertTrue(diagnostics.rows_exact_model_valid_after_step)
        self.assertFalse(diagnostics.retry_authorized)
        self.assertEqual(trainer.update_index, 0)
        self.assertEqual(trainer.counterfactual_auxiliary_update_count, 0)
        self.assertTrue(
            all(
                torch.equal(trainer.model.state_dict()[key], expected)
                for key, expected in model_before.items()
            )
        )
        self.assertTrue(_nested_equal(trainer.optimizer.state_dict(), optimizer_before))
        self.assertEqual(
            tuple(group["lr"] for group in trainer.optimizer.param_groups),
            learning_rates_before,
        )
        build_recurrent_counterfactual_auxiliary_target(
            trainer.model,
            [row],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "retry is forbidden",
        ):
            trainer.counterfactual_auxiliary_update(
                [[row]],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((2,)),
            )

    def test_optimizer_error_restores_exact_state_and_forbids_retry(self) -> None:
        row = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        trainer = RecurrentPPOTrainer(deepcopy(self.model))
        trainer.optimizer.zero_grad(set_to_none=True)
        zero_loss = sum(
            parameter.square().sum() * 0.0 for parameter in trainer.model.parameters()
        )
        zero_loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        model_before = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        optimizer_before = deepcopy(trainer.optimizer.state_dict())
        real_step = trainer.optimizer.step

        def mutate_then_fail(*args, **kwargs):
            real_step(*args, **kwargs)
            raise RuntimeError("forced optimizer failure")

        with patch.object(trainer.optimizer, "step", side_effect=mutate_then_fail):
            with self.assertRaisesRegex(
                RecurrentCounterfactualAuxiliaryError,
                "failed closed and restored exact state",
            ):
                trainer.counterfactual_auxiliary_update(
                    [[row]],
                    artifact_digest=self.artifact_digest,
                    config=_config_for_horizons((2,)),
                )

        self.assertEqual(trainer.update_index, 0)
        self.assertEqual(trainer.counterfactual_auxiliary_update_count, 0)
        self.assertTrue(
            all(
                torch.equal(trainer.model.state_dict()[key], expected)
                for key, expected in model_before.items()
            )
        )
        self.assertTrue(_nested_equal(trainer.optimizer.state_dict(), optimizer_before))
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "retry is forbidden",
        ):
            trainer.counterfactual_auxiliary_update(
                [[row]],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((2,)),
            )

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(),
        "MPS is unavailable",
    )
    def test_transaction_is_finite_on_mps(self) -> None:
        row = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        model = deepcopy(self.model).to(device="mps", dtype=torch.float32)
        trainer = RecurrentPPOTrainer(model)
        diagnostics = trainer.counterfactual_auxiliary_update(
            [[row]],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
        )
        torch.mps.synchronize()

        self.assertTrue(diagnostics.accepted)
        for value in (
            diagnostics.total_loss,
            diagnostics.gradient_norm_before_clip,
            diagnostics.gradient_norm_after_clip,
            diagnostics.mean_behavior_kl_old_to_post,
            diagnostics.max_state_behavior_kl_old_to_post,
        ):
            self.assertTrue(torch.isfinite(torch.tensor(value)).item())

    def test_terminal_only_value_target_is_available_but_truncation_fails_closed(
        self,
    ) -> None:
        alive = _with_synthetic_terminal_state(
            self.real_horizon_2,
            alive=True,
        )
        config = RecurrentCounterfactualAuxiliaryConfig(
            scalarization=CounterfactualHorizonScalarization(
                horizon_weights=((2, 1.0),),
            ),
            value_target_mode=RECURRENT_COUNTERFACTUAL_TERMINAL_VALUE_TARGET,
            value_loss_coefficient=0.5,
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "every branch return to be terminal",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                self.model,
                [alive],
                artifact_digest=self.artifact_digest,
                config=config,
            )

        terminal = _with_synthetic_terminal_state(
            self.real_horizon_2,
            alive=False,
        )
        terminal_model = deepcopy(self.model)
        result = recurrent_counterfactual_auxiliary_loss(
            terminal_model,
            [terminal],
            artifact_digest=self.artifact_digest,
            config=config,
        )
        self.assertIsNotNone(result.target.value_target)
        self.assertTrue(torch.isfinite(result.value_loss))
        result.loss.backward()
        self.assertIsNotNone(terminal_model.value.weight.grad)

    def test_tamper_leakage_mask_artifact_and_horizon_fail_closed(self) -> None:
        config = _config_for_horizons((2,))
        tampered = deepcopy(self.real_horizon_2)
        tampered["labels"]["action_outcomes"][0]["focal_discounted_return"] += 1.0
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "failed its exact branch contract",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                self.model,
                [tampered],
                artifact_digest=self.artifact_digest,
                config=config,
            )

        leaked = deepcopy(self.real_horizon_2)
        leaked["trainable_public_context"]["environment_seed"] = 17
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "failed its exact branch contract",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                self.model,
                [leaked],
                artifact_digest=self.artifact_digest,
                config=config,
            )

        mask_tampered = deepcopy(self.real_horizon_2)
        mask = mask_tampered["trainable_public_context"]["current_public_action_mask"]
        action = next(action for action in ACTION_NAMES if mask[action])
        mask[action] = False
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "failed its exact branch contract",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                self.model,
                [mask_tampered],
                artifact_digest=self.artifact_digest,
                config=config,
            )

        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "artifact",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                self.model,
                [self.real_horizon_2],
                artifact_digest="d" * 64,
                config=config,
            )
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "horizons do not exactly match",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                self.model,
                [self.real_horizon_2],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((1,)),
            )

    def test_model_update_makes_exact_rows_stale(self) -> None:
        synthetic = _with_synthetic_returns(self.real_horizon_2, multiplier=1.0)
        model = deepcopy(self.model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        result = recurrent_counterfactual_auxiliary_loss(
            model,
            [synthetic],
            artifact_digest=self.artifact_digest,
            config=_config_for_horizons((2,)),
        )
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        optimizer.step()
        self.assertNotEqual(
            recurrent_model_state_sha256(model),
            self.real_horizon_2["optimizer_context"]["source_model_state_sha256"],
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualAuxiliaryError,
            "stale for the current model",
        ):
            build_recurrent_counterfactual_auxiliary_target(
                model,
                [synthetic],
                artifact_digest=self.artifact_digest,
                config=_config_for_horizons((2,)),
            )


def _config_for_horizons(
    horizons: tuple[int, ...],
) -> RecurrentCounterfactualAuxiliaryConfig:
    weight = 1.0 / len(horizons)
    return RecurrentCounterfactualAuxiliaryConfig(
        scalarization=CounterfactualHorizonScalarization(
            horizon_weights=tuple((horizon, weight) for horizon in horizons),
        ),
        temperature=1.0,
        advantage_clip=100.0,
        behavior_kl_coefficient=0.1,
    )


def _with_synthetic_returns(
    source: dict[str, object],
    *,
    multiplier: float,
) -> dict[str, object]:
    row = deepcopy(source)
    outcomes = row["labels"]["action_outcomes"]
    for ordinal, outcome in enumerate(outcomes):
        outcome["focal_discounted_return"] = float(ordinal) * multiplier
    _refresh_label_and_row_digests(row)
    return row


def _with_synthetic_terminal_state(
    source: dict[str, object],
    *,
    alive: bool,
) -> dict[str, object]:
    row = deepcopy(source)
    outcomes = row["labels"]["action_outcomes"]
    for outcome in outcomes:
        outcome["focal_terminal"]["alive"] = alive
    _refresh_label_and_row_digests(row)
    return row


def _refresh_label_and_row_digests(row: dict[str, object]) -> None:
    row["component_digests"]["labels"] = stable_payload_digest(row["labels"])
    without_exact = dict(row)
    without_exact.pop("exact_digest", None)
    row["exact_digest"] = stable_payload_digest(without_exact)


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _nested_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return left == right


if __name__ == "__main__":
    unittest.main()
