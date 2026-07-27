from __future__ import annotations

import copy
import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from evolution_sim.config.schema import SignalConfig, WorldConfig
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.runtime.trajectory import (
    REWARD_COMPONENT_BOUNDS,
    REWARD_SCHEMA_VERSION,
)
from evolution_sim.env.world import SimulationWorld
from evolution_sim.mind.policy_inputs import ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE
from evolution_sim.mind.recurrent_rollout import (
    RECURRENT_ROLLOUT_ACTION_SOURCE,
    RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE,
    PreviousPublicFeedback,
    RecurrentCoreOutput,
    RecurrentOnPolicyCollector,
    RecurrentRolloutBuffer,
    RecurrentRolloutError,
    RecurrentRolloutStep,
    TorchRecurrentPolicyCore,
    derive_recurrent_policy_sampling_seed,
)

if torch is not None:
    from evolution_sim.mind.recurrent_actor_critic import (
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )


class _RecordingCore:
    hidden_size = 3

    def __init__(self) -> None:
        self.input_sizes: list[int] = []

    def initial_hidden(self) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0)

    def forward_step(
        self,
        observation: tuple[float, ...],
        current_action_mask: tuple[bool, ...],
        previous_feedback: PreviousPublicFeedback,
        hidden: tuple[float, ...],
    ) -> RecurrentCoreOutput:
        self.input_sizes.append(len(observation))
        observation_signal = (
            sum(observation[:8])
            + sum(current_action_mask)
            + sum(previous_feedback.vector())
        ) * 0.001
        return RecurrentCoreOutput(
            logits=tuple(
                observation_signal + action_index * 0.01
                for action_index in range(len(ACTION_NAMES))
            ),
            value=sum(observation[:3]) * 0.1 + sum(hidden) * 0.01,
            next_hidden=tuple(value + 1.0 for value in hidden),
        )


class RecurrentRolloutTests(unittest.TestCase):
    def test_real_world_rollout_aligns_public_policy_steps_and_bootstraps(self) -> None:
        core = _RecordingCore()
        collector = RecurrentOnPolicyCollector(core)
        collector.start_world(world_id="seed-7-a", seed=7, rollout_ticks=2)
        world = SimulationWorld(
            WorldConfig(seed=7, max_ticks=3),
            policy=collector,
        )

        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()

        steps = collector.buffer.steps
        self.assertGreater(len(steps), 0)
        self.assertTrue(all(step.tick < 2 for step in steps))
        self.assertTrue(all(len(step.observation) == 541 for step in steps))
        self.assertEqual(
            set(core.input_sizes),
            {ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE},
        )
        self.assertTrue(any(step.truncated for step in steps))

        records_by_tick_agent = {
            (int(record["tick"]), int(record["agent_id"])): record
            for record in world.trajectory_records
            if record["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
            and int(record["tick"]) < 2
        }
        self.assertEqual(len(records_by_tick_agent), len(steps))
        for step in steps:
            record = records_by_tick_agent[(step.tick, step.agent_id)]
            self.assertEqual(step.requested_action, record["requested_action"])
            self.assertEqual(step.resolved_action, record["resolved_action"])
            self.assertEqual(step.reward, record["reward"]["total"])
            self.assertEqual(
                step.reward_components,
                record["reward"]["components"],
            )
            self.assertEqual(
                set(step.reward_components),
                set(REWARD_COMPONENT_BOUNDS),
            )
            self.assertEqual(step.outcome, record["outcome"])
            self.assertEqual(step.environment_seed, 7)
            self.assertIsInstance(step.policy_sampling_seed, int)
            self.assertTrue(step.action_mask[step.action_index])
            self.assertTrue(math.isfinite(step.logprob))
            self.assertTrue(math.isfinite(step.value))

        first_by_agent: dict[int, RecurrentRolloutStep] = {}
        for step in steps:
            first_by_agent.setdefault(step.agent_id, step)
        self.assertTrue(first_by_agent)
        self.assertTrue(
            all(step.hidden == (0.0, 0.0, 0.0) for step in first_by_agent.values())
        )
        self.assertTrue(
            all(
                step.previous_feedback == PreviousPublicFeedback.zero()
                for step in first_by_agent.values()
            )
        )
        for sequence in collector.buffer.sequences():
            for previous, current in zip(sequence, sequence[1:]):
                self.assertEqual(
                    current.previous_feedback.requested_action_index,
                    previous.action_index,
                )
                self.assertEqual(
                    current.previous_feedback.resolved_action_index,
                    previous.resolved_action_index,
                )
                self.assertEqual(
                    current.previous_feedback.resolution_action_valid,
                    previous.resolution_action_valid,
                )
                self.assertEqual(
                    current.previous_feedback.moved,
                    previous.moved,
                )
                self.assertEqual(
                    current.previous_feedback.reward_total,
                    previous.reward,
                )

    def test_seeded_sampling_and_hidden_state_reset_across_worlds(self) -> None:
        core = _RecordingCore()
        collector = RecurrentOnPolicyCollector(core)
        policy_sampling_seed = derive_recurrent_policy_sampling_seed(
            task_identity="test-seeded-sampling-reset"
        )

        collected: list[tuple[tuple[int, int, str, float], ...]] = []
        for world_id in ("seed-11-first", "seed-11-second"):
            collector.start_world(
                world_id=world_id,
                environment_seed=11,
                policy_sampling_seed=policy_sampling_seed,
                rollout_ticks=2,
            )
            SimulationWorld(
                WorldConfig(seed=11, max_ticks=3),
                policy=collector,
            ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
            collector.finish_world()
            world_steps = tuple(
                step for step in collector.buffer.steps if step.world_id == world_id
            )
            collected.append(
                tuple(
                    (
                        step.tick,
                        step.agent_id,
                        step.requested_action,
                        step.logprob,
                    )
                    for step in world_steps
                )
            )
            first_by_agent: dict[int, RecurrentRolloutStep] = {}
            for step in world_steps:
                first_by_agent.setdefault(step.agent_id, step)
            self.assertTrue(
                all(step.hidden == (0.0, 0.0, 0.0) for step in first_by_agent.values())
            )
            self.assertTrue(
                all(
                    step.previous_feedback == PreviousPublicFeedback.zero()
                    for step in first_by_agent.values()
                )
            )

        self.assertEqual(collected[0], collected[1])

    def test_policy_sampling_seed_can_change_actions_without_changing_initial_state(
        self,
    ) -> None:
        def collect(policy_sampling_seed: int) -> tuple[RecurrentRolloutStep, ...]:
            collector = RecurrentOnPolicyCollector(_RecordingCore())
            collector.start_world(
                world_id=f"same-environment-policy-{policy_sampling_seed}",
                environment_seed=11,
                policy_sampling_seed=policy_sampling_seed,
                rollout_ticks=1,
            )
            SimulationWorld(
                WorldConfig(seed=11, max_ticks=2),
                policy=collector,
            ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
            collector.finish_world()
            return collector.buffer.steps

        first = collect(1)
        second = collect(2)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(first[0].environment_seed, 11)
        self.assertEqual(second[0].environment_seed, 11)
        self.assertEqual(first[0].agent_id, second[0].agent_id)
        self.assertEqual(first[0].observation, second[0].observation)
        self.assertEqual(first[0].action_mask, second[0].action_mask)
        self.assertEqual(first[0].policy_sampling_seed, 1)
        self.assertEqual(second[0].policy_sampling_seed, 2)
        self.assertNotEqual(first[0].requested_action, second[0].requested_action)

    def test_feed_forward_ablation_resets_hidden_for_every_decision(self) -> None:
        collector = RecurrentOnPolicyCollector(
            _RecordingCore(),
            reset_recurrent_state_each_decision=True,
        )
        collector.start_world(world_id="feed-forward", seed=11, rollout_ticks=2)
        SimulationWorld(
            WorldConfig(seed=11, max_ticks=3),
            policy=collector,
        ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()

        self.assertTrue(collector.buffer.steps)
        self.assertTrue(
            all(step.hidden == (0.0, 0.0, 0.0) for step in collector.buffer.steps)
        )
        self.assertTrue(
            any(
                step.previous_feedback.available
                for step in collector.buffer.steps
                if step.tick > 0
            )
        )

    def test_finish_fails_closed_without_the_public_bootstrap_tick(self) -> None:
        collector = RecurrentOnPolicyCollector(_RecordingCore())
        collector.start_world(world_id="missing-bootstrap", seed=3, rollout_ticks=2)
        SimulationWorld(
            WorldConfig(seed=3, max_ticks=2),
            policy=collector,
        ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)

        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "run one bootstrap tick beyond rollout_ticks",
        ):
            collector.finish_world()

    def test_passive_death_is_a_boundary_reward_not_a_fabricated_action(self) -> None:
        buffer = RecurrentRolloutBuffer()
        buffer.register_world("passive-death")
        buffer.append(_step(world_id="passive-death", reward=0.2, value=0.1))

        marked = buffer.mark_passive_terminal(
            world_id="passive-death",
            agent_id=4,
            tick=1,
            reward=-1.0,
            reward_components=_reward_components(-1.0),
        )
        rows = buffer.compute_gae(gamma=0.9, gae_lambda=1.0)

        self.assertTrue(marked)
        self.assertEqual(len(buffer.steps), 1)
        self.assertEqual(buffer.steps[0].requested_action, "eat")
        self.assertEqual(buffer.steps[0].reward, 0.2)
        self.assertEqual(buffer.steps[0].passive_terminal_reward, -1.0)
        self.assertEqual(
            buffer.steps[0].passive_terminal_reward_components,
            _reward_components(-1.0),
        )
        self.assertTrue(buffer.steps[0].terminated)
        self.assertAlmostEqual(rows[0].advantage, -0.8)
        self.assertAlmostEqual(rows[0].return_target, -0.7)

        gap_buffer = RecurrentRolloutBuffer()
        gap_buffer.register_world("passive-gap")
        gap_buffer.append(_step(world_id="passive-gap", reward=0.2, value=0.1))
        with self.assertRaisesRegex(
            RecurrentRolloutError,
            "must immediately follow",
        ):
            gap_buffer.mark_passive_terminal(
                world_id="passive-gap",
                agent_id=4,
                tick=2,
                reward=-1.0,
                reward_components=_reward_components(-1.0),
            )

    def test_passive_death_on_bootstrap_tick_is_terminal_not_truncated(self) -> None:
        collector = RecurrentOnPolicyCollector(_RecordingCore())
        collector.start_world(
            world_id="bootstrap-passive-death",
            environment_seed=7,
            policy_sampling_seed=17,
            rollout_ticks=1,
        )
        collector.buffer.append(
            _step(
                world_id="bootstrap-passive-death",
                tick=0,
                reward=0.2,
                value=0.1,
                environment_seed=7,
                policy_sampling_seed=17,
            )
        )

        collector.observe_transition(
            {
                "tick": 1,
                "agent_id": 4,
                "action_source": "passive",
                "after": {"alive": False},
                "outcome": {"died": True},
                "reward": _reward_payload(-1.0),
            }
        )
        collector.finish_world()

        final = collector.buffer.steps[0]
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertIsNone(final.bootstrap_value)
        self.assertEqual(final.passive_terminal_tick, 1)
        self.assertEqual(final.passive_terminal_reward, -1.0)
        self.assertEqual(
            final.passive_terminal_reward_components,
            _reward_components(-1.0),
        )

    def test_reward_components_fail_closed_on_schema_key_bound_and_total_drift(
        self,
    ) -> None:
        collector = RecurrentOnPolicyCollector(_RecordingCore())
        collector.start_world(world_id="strict-reward", seed=7, rollout_ticks=1)
        world = SimulationWorld(
            WorldConfig(seed=7, max_ticks=2),
            policy=collector,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()
        record = next(
            item
            for item in world.trajectory_records
            if item["action_source"] == RECURRENT_ROLLOUT_ACTION_SOURCE
        )

        invalid_payloads: list[dict[str, object]] = []
        wrong_schema = copy.deepcopy(record)
        wrong_schema["reward"]["schema_version"] = "wrong"
        invalid_payloads.append(wrong_schema)
        missing_component = copy.deepcopy(record)
        del missing_component["reward"]["components"]["energy_stability"]
        invalid_payloads.append(missing_component)
        out_of_bounds = copy.deepcopy(record)
        out_of_bounds["reward"]["components"]["resource_acquisition"] = 2.0
        invalid_payloads.append(out_of_bounds)
        mismatched_total = copy.deepcopy(record)
        mismatched_total["reward"]["total"] = 0.1234
        invalid_payloads.append(mismatched_total)

        for payload in invalid_payloads:
            with self.subTest(reward=payload["reward"]):
                with self.assertRaises(RecurrentRolloutError):
                    PreviousPublicFeedback.from_record(payload)

    def test_gae_distinguishes_time_limit_bootstrap_from_termination(self) -> None:
        buffer = RecurrentRolloutBuffer()
        buffer.register_world("truncated")
        buffer.append(
            _step(
                world_id="truncated",
                tick=0,
                decision_index=0,
                reward=1.0,
                value=0.5,
            )
        )
        buffer.append(
            _step(
                world_id="truncated",
                tick=1,
                decision_index=1,
                reward=2.0,
                value=0.25,
            )
        )
        buffer.mark_truncated(
            world_id="truncated",
            agent_id=4,
            bootstrap_value=0.4,
        )

        rows = buffer.compute_gae(gamma=0.9, gae_lambda=1.0)

        self.assertAlmostEqual(rows[1].advantage, 2.11)
        self.assertAlmostEqual(rows[1].return_target, 2.36)
        self.assertAlmostEqual(rows[0].advantage, 2.624)
        self.assertAlmostEqual(rows[0].return_target, 3.124)
        self.assertFalse(rows[1].step.terminated)
        self.assertTrue(rows[1].step.truncated)

    def test_previous_feedback_vector_is_public_typed_and_zero_at_birth(self) -> None:
        zero = PreviousPublicFeedback.zero()
        feedback = PreviousPublicFeedback(
            requested_action_index=ACTION_NAMES.index("move_east"),
            resolved_action_index=ACTION_NAMES.index("stay"),
            resolution_action_valid=False,
            moved=False,
            reward_total=-1.0,
        )

        self.assertEqual(
            zero.vector(),
            tuple(0.0 for _ in range(RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE)),
        )
        self.assertEqual(len(feedback.vector()), 43)
        self.assertEqual(
            feedback.vector()[ACTION_NAMES.index("move_east")],
            1.0,
        )
        self.assertEqual(
            feedback.vector()[len(ACTION_NAMES) + ACTION_NAMES.index("stay")],
            1.0,
        )
        self.assertGreaterEqual(feedback.vector()[-1], -1.0)
        self.assertLessEqual(feedback.vector()[-1], 1.0)

    @unittest.skipIf(
        torch is None, "optional Mind ML dependency torch is not installed"
    )
    def test_torch_model_adapter_recomputes_collected_value_and_logprob(self) -> None:
        assert torch is not None
        model = PublicRecurrentActorCritic(initialization_seed=73)
        collector = RecurrentOnPolicyCollector(TorchRecurrentPolicyCore(model))
        collector.start_world(world_id="torch-adapter", seed=23, rollout_ticks=1)
        SimulationWorld(
            WorldConfig(seed=23, max_ticks=2),
            policy=collector,
        ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()
        step = collector.buffer.steps[0]

        observations = torch.tensor(step.observation).reshape(1, 1, -1)
        masks = torch.tensor(step.action_mask).reshape(1, 1, -1)
        feedback = torch.tensor(step.previous_feedback.vector()).reshape(1, 1, -1)
        state = torch.tensor(step.hidden).reshape(
            model.config.recurrent_layers,
            1,
            model.config.hidden_size,
        )
        with torch.no_grad():
            output = model.forward_sequence(
                observations,
                masks,
                feedback,
                initial_state=state,
            )
            distribution = torch.distributions.Categorical(
                logits=output.masked_logits[0, 0]
            )
            expected_logprob = distribution.log_prob(torch.tensor(step.action_index))

        self.assertAlmostEqual(step.value, float(output.values[0, 0]), places=6)
        self.assertAlmostEqual(step.logprob, float(expected_logprob), places=6)

    @unittest.skipIf(
        torch is None, "optional Mind ML dependency torch is not installed"
    )
    def test_torch_model_adapter_collects_token_aware_world_without_shape_loss(
        self,
    ) -> None:
        assert torch is not None
        signals = SignalConfig(communication_signal_emission_enabled=True)
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig.for_signal_config(
                signals,
                encoder_size=8,
                hidden_size=8,
            ),
            initialization_seed=73,
        )
        core = TorchRecurrentPolicyCore(model)
        collector = RecurrentOnPolicyCollector(core)
        collector.start_world(
            world_id="torch-token-adapter",
            seed=23,
            rollout_ticks=1,
        )

        SimulationWorld(
            WorldConfig(seed=23, max_ticks=2, signals=signals),
            policy=collector,
        ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        collector.finish_world()

        self.assertEqual(
            core.public_input_schema_version, "mind_ecological_policy_input_v2"
        )
        self.assertEqual(core.public_input_size, 645)
        self.assertEqual(core.learned_input_size, 708)
        self.assertTrue(collector.buffer.steps)
        self.assertTrue(
            all(len(step.observation) == 645 for step in collector.buffer.steps)
        )


def _step(
    *,
    world_id: str,
    tick: int = 0,
    decision_index: int = 0,
    reward: float,
    value: float,
    environment_seed: int = 7,
    policy_sampling_seed: int | None = None,
) -> RecurrentRolloutStep:
    action_mask = tuple(True for _ in ACTION_NAMES)
    return RecurrentRolloutStep(
        world_id=world_id,
        world_seed=environment_seed,
        tick=tick,
        agent_id=4,
        decision_index=decision_index,
        observation=tuple(0.0 for _ in range(ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE)),
        previous_feedback=PreviousPublicFeedback.zero(),
        action_mask=action_mask,
        hidden=(0.0, 0.0, 0.0),
        action_index=ACTION_NAMES.index("eat"),
        requested_action="eat",
        logprob=-1.0,
        entropy=1.0,
        value=value,
        reward=reward,
        reward_components=_reward_components(reward),
        resolved_action_index=ACTION_NAMES.index("eat"),
        resolved_action="eat",
        resolution_action_mask=action_mask,
        action_valid=True,
        resolution_action_valid=True,
        moved=False,
        outcome={"died": False},
        environment_seed=environment_seed,
        policy_sampling_seed=policy_sampling_seed,
    )


def _reward_payload(total: float) -> dict[str, object]:
    return {
        "schema_version": REWARD_SCHEMA_VERSION,
        "components": _reward_components(total),
        "total": total,
    }


def _reward_components(total: float) -> dict[str, float]:
    remaining = round(float(total), 4)
    components = {name: 0.0 for name in REWARD_COMPONENT_BOUNDS}
    for name, (lower, upper) in REWARD_COMPONENT_BOUNDS.items():
        if remaining > 0.0 and upper > 0.0:
            value = min(remaining, upper)
        elif remaining < 0.0 and lower < 0.0:
            value = max(remaining, lower)
        else:
            continue
        components[name] = round(value, 4)
        remaining = round(remaining - value, 4)
        if remaining == 0.0:
            break
    if remaining != 0.0:
        raise AssertionError(
            f"synthetic test reward is outside component bounds: {total}"
        )
    return components


if __name__ == "__main__":
    unittest.main()
