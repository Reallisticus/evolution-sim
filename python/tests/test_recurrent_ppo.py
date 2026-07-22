from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.mind.recurrent_actor_critic import (
        ACTION_COUNT,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        PUBLIC_INPUT_SIZE,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_ppo import (
        PPOTrainingSequence,
        RECURRENT_PPO_CONTRACT_VERSION,
        RecurrentPPOConfig,
        RecurrentPPOError,
        RecurrentPPOTrainer,
        _ChunkReference,
        _EvaluatedRows,
        _WorldLossWeighting,
        _chunk_references,
        _loss_terms,
        _world_loss_weighting,
        recurrent_ppo_contract,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentPPOTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_config_and_contract_pin_real_sequence_ppo_boundaries(self) -> None:
        config = RecurrentPPOConfig()
        contract = recurrent_ppo_contract(config)

        self.assertEqual(contract["schema_version"], RECURRENT_PPO_CONTRACT_VERSION)
        self.assertEqual(
            contract["algorithm"],
            "parameter_shared_recurrent_ippo_clipped_ppo",
        )
        self.assertFalse(contract["sequence_batching"]["row_shuffle"])
        self.assertEqual(
            contract["sequence_batching"]["burn_in_steps"],
            config.burn_in_steps,
        )
        self.assertEqual(
            contract["likelihood_contract"]["action_masks"],
            "stored_observation_time_masks_only",
        )
        self.assertTrue(config.normalize_advantages)
        self.assertFalse(config.world_balanced_loss)
        self.assertFalse(contract["world_balancing"]["enabled"])
        self.assertFalse(contract["world_balancing"]["default_enabled"])

        with self.assertRaises(RecurrentPPOError):
            RecurrentPPOConfig(gamma=1.01)
        with self.assertRaises(RecurrentPPOError):
            RecurrentPPOConfig(tbptt_steps=0)
        with self.assertRaises(RecurrentPPOError):
            RecurrentPPOConfig(burn_in_steps=-1)
        with self.assertRaises(RecurrentPPOError):
            RecurrentPPOConfig(normalize_advantages=1)  # type: ignore[arg-type]
        with self.assertRaises(RecurrentPPOError):
            RecurrentPPOConfig(world_balanced_loss=1)  # type: ignore[arg-type]

    def test_counterfactual_opt_out_preserves_base_ppo_bit_identity(self) -> None:
        model_config = RecurrentActorCriticConfig(
            encoder_size=16,
            hidden_size=16,
        )
        first_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=771,
        )
        second_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=771,
        )
        sequences = _behavior_sequences(first_model, lengths=(5, 7))
        config = RecurrentPPOConfig(
            learning_rate=1.0e-3,
            update_epochs=2,
            sequence_minibatch_size=2,
            tbptt_steps=4,
            burn_in_steps=2,
            learner_seed=9917,
        )
        first = RecurrentPPOTrainer(first_model, config)
        second = RecurrentPPOTrainer(second_model, config)
        self.assertEqual(second.counterfactual_auxiliary_update_count, 0)

        first_diagnostics = first.update(sequences)
        second_diagnostics = second.update(sequences)

        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(first.update_index, second.update_index)
        self.assertEqual(first.counterfactual_auxiliary_update_count, 0)
        self.assertEqual(second.counterfactual_auxiliary_update_count, 0)
        self.assertTrue(
            all(
                torch.equal(first_model.state_dict()[key], value)
                for key, value in second_model.state_dict().items()
            )
        )
        self.assertTrue(
            _nested_state_equal(
                first.optimizer.state_dict(),
                second.optimizer.state_dict(),
            )
        )

    def test_real_ppo_improves_legal_targets_and_shuffled_labels_are_worse(
        self,
    ) -> None:
        model_config = RecurrentActorCriticConfig(
            encoder_size=24,
            hidden_size=24,
        )
        correct_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=733,
        )
        shuffled_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=733,
        )
        correct_sequences = _behavior_sequences(correct_model)
        shuffled_sequences = _behavior_sequences(
            shuffled_model,
            shuffled_labels=True,
        )
        old_log_probs = tuple(
            sequence.old_log_probs.clone() for sequence in correct_sequences
        )
        old_values = tuple(
            sequence.old_values.clone() for sequence in correct_sequences
        )
        before_correct = _mean_action_log_prob(correct_model, correct_sequences)

        ppo_config = RecurrentPPOConfig(
            learning_rate=8.0e-3,
            policy_clip_range=0.4,
            value_clip_range=0.4,
            value_loss_coefficient=0.0,
            entropy_coefficient=0.0,
            update_epochs=10,
            sequence_minibatch_size=16,
            tbptt_steps=3,
            burn_in_steps=2,
            max_gradient_norm=5.0,
            normalize_advantages=False,
            learner_seed=404_337_389,
        )
        correct_diagnostics = RecurrentPPOTrainer(
            correct_model,
            ppo_config,
        ).update(correct_sequences)
        shuffled_diagnostics = RecurrentPPOTrainer(
            shuffled_model,
            ppo_config,
        ).update(shuffled_sequences)

        after_correct = _mean_action_log_prob(correct_model, correct_sequences)
        after_shuffled_on_correct = _mean_action_log_prob(
            shuffled_model,
            correct_sequences,
        )
        self.assertGreater(correct_diagnostics.policy_objective_delta, 0.05)
        self.assertGreater(correct_diagnostics.parameter_delta_l2, 0.0)
        self.assertGreater(after_correct, before_correct + 0.05)
        self.assertGreater(after_correct, after_shuffled_on_correct + 0.10)
        self.assertEqual(correct_diagnostics.transition_count, 21)
        self.assertEqual(correct_diagnostics.chunk_count, 8)
        self.assertGreater(correct_diagnostics.burn_in_transition_count, 0)
        self.assertEqual(
            correct_diagnostics.minibatch_order_sha256,
            shuffled_diagnostics.minibatch_order_sha256,
        )
        self.assertTrue(correct_diagnostics.old_statistics_frozen)
        self.assertTrue(
            all(
                torch.equal(sequence.old_log_probs, expected)
                for sequence, expected in zip(
                    correct_sequences, old_log_probs, strict=True
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(sequence.old_values, expected)
                for sequence, expected in zip(
                    correct_sequences, old_values, strict=True
                )
            )
        )
        for value in (
            correct_diagnostics.approximate_kl,
            correct_diagnostics.policy_clip_fraction,
            correct_diagnostics.value_clip_fraction,
            correct_diagnostics.explained_variance,
            correct_diagnostics.gradient_norm_mean,
            correct_diagnostics.gradient_norm_max,
        ):
            self.assertTrue(math.isfinite(value))

    def test_burn_in_reconstructs_current_hidden_without_adding_loss_rows(self) -> None:
        model_config = RecurrentActorCriticConfig(
            encoder_size=16,
            hidden_size=16,
        )
        behavior_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=91,
        )
        sequences = _behavior_sequences(behavior_model, lengths=(9,))
        no_burn_model = PublicRecurrentActorCritic(model_config)
        burn_model = PublicRecurrentActorCritic(model_config)
        no_burn_model.load_state_dict(behavior_model.state_dict())
        burn_model.load_state_dict(behavior_model.state_dict())
        with torch.no_grad():
            no_burn_model.recurrent.weight_hh_l0.add_(0.08)
            burn_model.recurrent.weight_hh_l0.add_(0.08)

        common = dict(
            learning_rate=1.0e-8,
            value_loss_coefficient=0.0,
            entropy_coefficient=0.0,
            update_epochs=1,
            sequence_minibatch_size=8,
            tbptt_steps=3,
            max_gradient_norm=5.0,
            normalize_advantages=False,
            learner_seed=709_266_037,
        )
        no_burn = RecurrentPPOTrainer(
            no_burn_model,
            RecurrentPPOConfig(burn_in_steps=0, **common),
        ).update(sequences)
        with_burn = RecurrentPPOTrainer(
            burn_model,
            RecurrentPPOConfig(burn_in_steps=2, **common),
        ).update(sequences)

        self.assertEqual(no_burn.transition_count, with_burn.transition_count)
        self.assertEqual(no_burn.transition_count, 9)
        self.assertEqual(no_burn.burn_in_transition_count, 0)
        self.assertEqual(with_burn.burn_in_transition_count, 4)
        self.assertEqual(with_burn.burn_in_reconstructed_chunk_count, 2)
        self.assertGreater(with_burn.burn_in_state_delta_mean, 0.0)
        self.assertNotAlmostEqual(
            no_burn.initial_policy_objective,
            with_burn.initial_policy_objective,
            places=7,
        )

    def test_feed_forward_ablation_resets_memory_with_public_feedback_present(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=12, hidden_size=12),
            initialization_seed=17,
        )
        sequences = _behavior_sequences(
            model,
            lengths=(6,),
            include_previous_feedback=True,
        )
        diagnostics = RecurrentPPOTrainer(
            model,
            RecurrentPPOConfig(
                learning_rate=1.0e-3,
                update_epochs=1,
                sequence_minibatch_size=2,
                tbptt_steps=3,
                burn_in_steps=2,
                feed_forward_history_ablation=True,
                learner_seed=103_406_2175,
            ),
        ).update(sequences)

        self.assertTrue(diagnostics.feed_forward_history_ablation)
        self.assertEqual(diagnostics.transition_count, 6)
        self.assertEqual(diagnostics.burn_in_transition_count, 0)

    def test_feed_forward_chunks_batch_independent_one_step_rows_equivalently(
        self,
    ) -> None:
        model_config = RecurrentActorCriticConfig(
            encoder_size=16,
            hidden_size=16,
        )
        vector_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=131,
        )
        scalar_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=131,
        )
        sequences = _behavior_sequences(
            scalar_model,
            lengths=(5, 8),
            include_previous_feedback=True,
            world_ids=("short-world", "long-world"),
        )
        varied_sequences: list[PPOTrainingSequence] = []
        for sequence_index, sequence in enumerate(sequences):
            masks = sequence.action_masks.clone()
            for step in range(sequence.length):
                masks[step, 2] = (step + sequence_index) % 2 == 0
                masks[step, 3] = (step + sequence_index) % 3 == 0
            varied_sequences.append(replace(sequence, action_masks=masks))
        sequences = tuple(varied_sequences)
        config = RecurrentPPOConfig(
            learning_rate=1.0e-3,
            update_epochs=1,
            sequence_minibatch_size=8,
            tbptt_steps=3,
            burn_in_steps=0,
            normalize_advantages=False,
            world_balanced_loss=True,
            feed_forward_history_ablation=True,
            learner_seed=103_406_2175,
        )
        chunks = _chunk_references(sequences, tbptt_steps=config.tbptt_steps)
        weighting = _world_loss_weighting(sequences, enabled=True)
        trainer = RecurrentPPOTrainer(vector_model, config)

        with patch.object(
            vector_model,
            "evaluate_sequence",
            wraps=vector_model.evaluate_sequence,
        ) as evaluate:
            vector_rows = trainer._evaluate_chunks(
                chunks,
                sequences,
                world_weighting=weighting,
            )
        scalar_rows = _scalar_feed_forward_rows(
            scalar_model,
            chunks,
            sequences,
            weighting,
        )

        self.assertEqual(evaluate.call_count, len(chunks))
        self.assertLess(
            evaluate.call_count, sum(sequence.length for sequence in sequences)
        )
        self.assertEqual(
            [
                bool(sequences[chunk.sequence_index].episode_starts[chunk.start])
                for chunk in chunks
            ],
            [True, False, True, False, False],
        )
        for field_name in (
            "new_log_probs",
            "new_values",
            "entropy",
            "old_log_probs",
            "old_values",
            "advantages",
            "return_targets",
            "loss_weights",
        ):
            torch.testing.assert_close(
                getattr(vector_rows, field_name),
                getattr(scalar_rows, field_name),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        vector_terms = _loss_terms(vector_rows, config=config)
        scalar_terms = _loss_terms(scalar_rows, config=config)
        for field_name in vector_terms.__dataclass_fields__:
            torch.testing.assert_close(
                getattr(vector_terms, field_name),
                getattr(scalar_terms, field_name),
                rtol=1.0e-6,
                atol=1.0e-7,
            )

        vector_terms.total_loss.backward()
        scalar_terms.total_loss.backward()
        vector_gradients = dict(vector_model.named_parameters())
        scalar_gradients = dict(scalar_model.named_parameters())
        self.assertEqual(set(vector_gradients), set(scalar_gradients))
        for name in vector_gradients:
            vector_gradient = vector_gradients[name].grad
            scalar_gradient = scalar_gradients[name].grad
            self.assertIsNotNone(vector_gradient, name)
            self.assertIsNotNone(scalar_gradient, name)
            torch.testing.assert_close(
                vector_gradient,
                scalar_gradient,
                rtol=2.0e-5,
                atol=2.0e-7,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(),
        "MPS is unavailable",
    )
    def test_vectorized_feed_forward_update_is_finite_on_mps(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=16, hidden_size=16),
            initialization_seed=151,
        ).to(device="mps", dtype=torch.float32)
        sequences = _behavior_sequences(
            model,
            lengths=(7, 9),
            include_previous_feedback=True,
            world_ids=("mps-short", "mps-long"),
        )
        diagnostics = RecurrentPPOTrainer(
            model,
            RecurrentPPOConfig(
                learning_rate=1.0e-3,
                update_epochs=1,
                sequence_minibatch_size=16,
                tbptt_steps=4,
                burn_in_steps=0,
                world_balanced_loss=True,
                feed_forward_history_ablation=True,
                learner_seed=103_406_2175,
            ),
        ).update(sequences)
        torch.mps.synchronize()

        self.assertEqual(diagnostics.transition_count, 16)
        self.assertTrue(diagnostics.feed_forward_history_ablation)
        self.assertTrue(
            all(
                bool(torch.isfinite(value).all().item())
                for value in model.state_dict().values()
                if value.is_floating_point()
            )
        )
        for value in (
            diagnostics.final_total_loss,
            diagnostics.final_policy_loss,
            diagnostics.final_value_loss,
            diagnostics.final_entropy,
            diagnostics.approximate_kl,
            diagnostics.gradient_norm_mean,
            diagnostics.parameter_delta_l2,
        ):
            self.assertTrue(math.isfinite(value))

    def test_world_balanced_loss_equalizes_world_mass_and_weighted_statistics(
        self,
    ) -> None:
        model_config = RecurrentActorCriticConfig(
            encoder_size=12,
            hidden_size=12,
        )
        behavior_model = PublicRecurrentActorCritic(
            model_config,
            initialization_seed=29,
        )
        sequences = _behavior_sequences(
            behavior_model,
            lengths=(2, 3, 10),
            world_ids=("short", "short", "long"),
        )
        sequences = tuple(
            replace(
                sequence,
                advantages=torch.full(
                    (sequence.length,),
                    10.0 if sequence.world_id == "short" else -2.0,
                ),
                return_targets=(
                    sequence.old_values
                    + (10.0 if sequence.world_id == "short" else 0.0)
                ),
            )
            for sequence in sequences
        )
        default_model = PublicRecurrentActorCritic(model_config)
        balanced_model = PublicRecurrentActorCritic(model_config)
        normalized_model = PublicRecurrentActorCritic(model_config)
        default_model.load_state_dict(behavior_model.state_dict())
        balanced_model.load_state_dict(behavior_model.state_dict())
        normalized_model.load_state_dict(behavior_model.state_dict())
        common = dict(
            learning_rate=1.0e-12,
            value_loss_coefficient=1.0,
            entropy_coefficient=0.0,
            update_epochs=1,
            sequence_minibatch_size=32,
            tbptt_steps=32,
            burn_in_steps=0,
            max_gradient_norm=100.0,
            normalize_advantages=False,
            learner_seed=167_443_951,
        )

        default = RecurrentPPOTrainer(
            default_model,
            RecurrentPPOConfig(world_balanced_loss=False, **common),
        ).update(sequences)
        balanced = RecurrentPPOTrainer(
            balanced_model,
            RecurrentPPOConfig(world_balanced_loss=True, **common),
        ).update(sequences)
        normalized = RecurrentPPOTrainer(
            normalized_model,
            replace(
                RecurrentPPOConfig(world_balanced_loss=True, **common),
                normalize_advantages=True,
            ),
        ).update(sequences)

        self.assertFalse(default.world_balanced_loss)
        self.assertTrue(balanced.world_balanced_loss)
        self.assertEqual(balanced.world_count, 2)
        self.assertEqual(balanced.min_world_transition_count, 5)
        self.assertEqual(balanced.max_world_transition_count, 10)
        self.assertAlmostEqual(default.advantage_mean, 2.0, places=6)
        self.assertAlmostEqual(balanced.advantage_mean, 4.0, places=6)
        self.assertAlmostEqual(balanced.advantage_std, 6.0, places=6)
        self.assertAlmostEqual(normalized.advantage_mean, 4.0, places=6)
        self.assertAlmostEqual(normalized.advantage_std, 6.0, places=6)
        self.assertAlmostEqual(normalized.initial_policy_objective, 0.0, places=6)
        self.assertAlmostEqual(default.initial_policy_objective, 2.0, places=5)
        self.assertAlmostEqual(balanced.initial_policy_objective, 4.0, places=5)
        self.assertAlmostEqual(default.final_value_loss, 50.0 / 3.0, places=4)
        self.assertAlmostEqual(balanced.final_value_loss, 25.0, places=4)
        self.assertAlmostEqual(
            default.min_effective_world_total_weight,
            5.0,
            places=6,
        )
        self.assertAlmostEqual(
            default.max_effective_world_total_weight,
            10.0,
            places=6,
        )
        self.assertAlmostEqual(
            balanced.min_effective_world_total_weight,
            7.5,
            places=6,
        )
        self.assertAlmostEqual(
            balanced.max_effective_world_total_weight,
            7.5,
            places=6,
        )
        self.assertAlmostEqual(balanced.mean_transition_loss_weight, 1.0)

    def test_illegal_stored_action_fails_closed_without_parameter_change(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=12, hidden_size=12),
            initialization_seed=23,
        )
        sequence = _behavior_sequences(model, lengths=(5,))[0]
        invalid_masks = sequence.action_masks.clone()
        invalid_masks[0, sequence.actions[0]] = False
        invalid = replace(sequence, action_masks=invalid_masks)
        before = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        trainer = RecurrentPPOTrainer(model, RecurrentPPOConfig(update_epochs=1))

        with self.assertRaisesRegex(
            RecurrentPPOError, "behavior actions must be legal"
        ):
            trainer.update((invalid,))

        self.assertEqual(trainer.update_index, 0)
        self.assertTrue(
            all(
                torch.equal(model.state_dict()[key], value)
                for key, value in before.items()
            )
        )


def _scalar_feed_forward_rows(
    model: PublicRecurrentActorCritic,
    chunks: tuple[_ChunkReference, ...],
    sequences: tuple[PPOTrainingSequence, ...],
    world_weighting: _WorldLossWeighting,
) -> _EvaluatedRows:
    """Retained test-only reference for the original scalar ablation path."""

    new_log_probs: list[torch.Tensor] = []
    new_values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    return_targets: list[torch.Tensor] = []
    loss_weights: list[torch.Tensor] = []
    for chunk in chunks:
        sequence = sequences[chunk.sequence_index]
        selected = slice(chunk.start, chunk.stop)
        chunk_log_probs: list[torch.Tensor] = []
        chunk_values: list[torch.Tensor] = []
        chunk_entropies: list[torch.Tensor] = []
        for step in range(chunk.start, chunk.stop):
            evaluation = model.evaluate_sequence(
                sequence.observations[step : step + 1].unsqueeze(1),
                sequence.action_masks[step : step + 1].unsqueeze(1),
                sequence.previous_feedback[step : step + 1].unsqueeze(1),
                sequence.actions[step : step + 1].unsqueeze(1),
                initial_state=model.initial_state(1),
                episode_starts=torch.zeros(
                    (1, 1),
                    dtype=torch.bool,
                    device=sequence.observations.device,
                ),
            )
            chunk_log_probs.append(evaluation.log_probs.squeeze(1))
            chunk_values.append(evaluation.values.squeeze(1))
            chunk_entropies.append(evaluation.entropy.squeeze(1))
        new_log_probs.append(torch.cat(chunk_log_probs))
        new_values.append(torch.cat(chunk_values))
        entropies.append(torch.cat(chunk_entropies))
        old_log_probs.append(sequence.old_log_probs[selected])
        old_values.append(sequence.old_values[selected])
        advantages.append(sequence.advantages[selected])
        return_targets.append(sequence.return_targets[selected])
        loss_weights.append(
            torch.full(
                (chunk.stop - chunk.start,),
                world_weighting.transition_weights[sequence.world_id],
                dtype=sequence.observations.dtype,
                device=sequence.observations.device,
            )
        )
    return _EvaluatedRows(
        new_log_probs=torch.cat(new_log_probs),
        new_values=torch.cat(new_values),
        entropy=torch.cat(entropies),
        old_log_probs=torch.cat(old_log_probs),
        old_values=torch.cat(old_values),
        advantages=torch.cat(advantages),
        return_targets=torch.cat(return_targets),
        loss_weights=torch.cat(loss_weights),
        burn_in_transition_count=0,
        burn_in_reconstructed_chunk_count=0,
        burn_in_state_delta_sum=0.0,
    )


def _behavior_sequences(
    model: PublicRecurrentActorCritic,
    *,
    lengths: tuple[int, ...] = (5, 7, 9),
    shuffled_labels: bool = False,
    include_previous_feedback: bool = False,
    world_ids: tuple[str, ...] | None = None,
) -> tuple[PPOTrainingSequence, ...]:
    if world_ids is None:
        world_ids = tuple(f"world-{index}" for index in range(len(lengths)))
    if len(world_ids) != len(lengths):
        raise ValueError("world_ids must match lengths")
    sequences: list[PPOTrainingSequence] = []
    model.eval()
    reference = next(model.parameters())
    for sequence_index, length in enumerate(lengths):
        observations = torch.zeros(
            length,
            PUBLIC_INPUT_SIZE,
            dtype=reference.dtype,
            device=reference.device,
        )
        for step in range(length):
            positive = (step + sequence_index) % 4 < 2
            observations[step, 0] = 0.85 if positive else -0.85
            observations[step, 1] = (sequence_index + 1) / 10.0
        target_actions = (observations[:, 0] < 0.0).to(torch.long)
        actions = (
            torch.roll(target_actions, shifts=1)
            if shuffled_labels
            else target_actions.clone()
        )
        action_masks = torch.zeros(
            length,
            ACTION_COUNT,
            dtype=torch.bool,
            device=reference.device,
        )
        action_masks[:, 0] = True
        action_masks[:, 1] = True
        previous_feedback = torch.zeros(
            length,
            PREVIOUS_PUBLIC_FEEDBACK_SIZE,
            dtype=reference.dtype,
            device=reference.device,
        )
        if include_previous_feedback:
            for step in range(1, length):
                previous_action = int(actions[step - 1].item())
                previous_feedback[step, previous_action] = 1.0
                previous_feedback[step, ACTION_COUNT + previous_action] = 1.0
                previous_feedback[step, -3] = 1.0
        episode_starts = torch.zeros(
            length,
            dtype=torch.bool,
            device=reference.device,
        )
        episode_starts[0] = True

        state = model.initial_state(1)
        recurrent_states: list[torch.Tensor] = []
        old_log_probs: list[torch.Tensor] = []
        old_values: list[torch.Tensor] = []
        with torch.no_grad():
            for step in range(length):
                recurrent_states.append(state[:, 0].clone())
                output = model.forward_sequence(
                    observations[step : step + 1].unsqueeze(1),
                    action_masks[step : step + 1].unsqueeze(1),
                    previous_feedback[step : step + 1].unsqueeze(1),
                    initial_state=state,
                    episode_starts=episode_starts[step : step + 1].unsqueeze(1),
                )
                distribution = torch.distributions.Categorical(
                    logits=output.masked_logits[0, 0]
                )
                old_log_probs.append(distribution.log_prob(actions[step]))
                old_values.append(output.values[0, 0])
                state = output.final_state

        old_value_tensor = torch.stack(old_values)
        sequences.append(
            PPOTrainingSequence(
                world_id=world_ids[sequence_index],
                observations=observations,
                action_masks=action_masks,
                previous_feedback=previous_feedback,
                recurrent_states=torch.stack(recurrent_states),
                actions=actions,
                old_log_probs=torch.stack(old_log_probs),
                old_values=old_value_tensor,
                advantages=torch.ones(
                    length,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                return_targets=old_value_tensor
                + torch.linspace(
                    0.2,
                    0.8,
                    length,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                episode_starts=episode_starts,
            )
        )
    return tuple(sequences)


def _mean_action_log_prob(
    model: PublicRecurrentActorCritic,
    sequences: tuple[PPOTrainingSequence, ...],
) -> float:
    values: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            evaluation = model.evaluate_sequence(
                sequence.observations.unsqueeze(1),
                sequence.action_masks.unsqueeze(1),
                sequence.previous_feedback.unsqueeze(1),
                sequence.actions.unsqueeze(1),
                initial_state=model.initial_state(1),
                episode_starts=sequence.episode_starts.unsqueeze(1),
            )
            values.append(evaluation.log_probs.squeeze(1))
    return float(torch.cat(values).mean().item())


def _nested_state_equal(left: object, right: object) -> bool:
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
            and all(_nested_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _nested_state_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return left == right


if __name__ == "__main__":
    unittest.main()
