from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    SELF_INPUT_FIELDS,
)
from evolution_sim.mind.dataset import load_trajectory_jsonl
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_ARCHITECTURE,
    MIND_V3_CONTROLLER_SCHEMA_VERSION,
    MIND_V3_HIDDEN_UNITS,
    mind_v3_parameter_count,
)
from evolution_sim.mind.rollout_context import RolloutContextState
from evolution_sim.mind.rollout_sequence_support_audit import _sequence_keys
from evolution_sim.mind.sequence_history_shadow_scorer import (
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_POLICY,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
    SequenceHistoryShadowScorer,
    sequence_history_shadow_sequence_keys,
)
from evolution_sim.mind import evaluation_harness as evaluate_harness
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

ROOT = Path(__file__).resolve().parents[2]


class MindV3SequenceHistoryShadowRuntimeTests(unittest.TestCase):
    def test_shadow_scorer_does_not_change_requested_action(self) -> None:
        baseline = _policy()
        with_shadow = _policy(sequence_history_shadow_scorer=_eat_shadow_scorer())
        observation = _observation(agent_id=3)
        action_mask = _action_mask("drink", "eat")

        baseline_decision = baseline.decide(dict(observation), dict(action_mask))
        shadow_decision = with_shadow.decide(dict(observation), dict(action_mask))

        self.assertEqual(baseline_decision.requested_action, "drink")
        self.assertEqual(shadow_decision.requested_action, "drink")
        shadow = shadow_decision.diagnostics["sequence_history_shadow_scorer"]
        self.assertEqual(shadow["predicted_action"], "eat")
        self.assertTrue(shadow["would_change_action"])
        self.assertFalse(shadow["runtime_action_selection_changed"])
        self.assertEqual(shadow_decision.requested_action, baseline_decision.requested_action)

    def test_policy_keeps_independent_shadow_rollout_history(self) -> None:
        policy = _policy(sequence_history_shadow_scorer=_empty_shadow_scorer())
        observation = _observation(agent_id=3)
        action_mask = _action_mask("drink", "eat")

        first = policy.decide(dict(observation), dict(action_mask))
        first_shadow = first.diagnostics["sequence_history_shadow_scorer"]
        self.assertIn("history_any=False", first_shadow["sequence_keys"][-1])

        policy.observe_transition(
            _transition_record(
                first.requested_action,
                agent_id=3,
                observation_input=observation["observation_input"],
                action_mask=action_mask,
            )
        )
        second = policy.decide(dict(observation), dict(action_mask))
        second_shadow = second.diagnostics["sequence_history_shadow_scorer"]

        self.assertIn("history_any=True", second_shadow["sequence_keys"][-1])
        self.assertEqual(second.requested_action, "drink")

    def test_passive_transition_does_not_update_shadow_history(self) -> None:
        policy = _policy(sequence_history_shadow_scorer=_empty_shadow_scorer())
        observation = _observation(agent_id=3)
        action_mask = _action_mask("drink", "eat")

        first = policy.decide(dict(observation), dict(action_mask))
        passive = _transition_record(
            first.requested_action,
            agent_id=3,
            observation_input=observation["observation_input"],
            action_mask=action_mask,
        )
        passive["action_source"] = "passive"
        passive["requested_action"] = "stay"
        passive["resolved_action"] = "stay"
        passive["outcome"]["passive"] = {"died_after_action": True}
        passive["outcome"]["died"] = True
        passive["after"]["alive"] = False

        policy.observe_transition(passive)
        second = policy.decide(dict(observation), dict(action_mask))
        second_shadow = second.diagnostics["sequence_history_shadow_scorer"]

        self.assertIn("history_any=False", second_shadow["sequence_keys"][-1])

    def test_public_sequence_key_helper_matches_v137_order(self) -> None:
        state = RolloutContextState()
        state.update_from_record(_transition_record("drink"))
        action_mask = _action_mask("drink", "eat")

        self.assertEqual(
            sequence_history_shadow_sequence_keys(
                state=state,
                valid_action_mask=action_mask,
            ),
            _sequence_keys(
                state=state,
                snapshot=state.snapshot(),
                valid_actions=("drink", "eat"),
            ),
        )

    def test_runtime_diagnostics_aggregate_shadow_counts(self) -> None:
        diagnostics = evaluate_harness._sequence_history_shadow_scorer_diagnostics(
            [
                {
                    "sequence_history_shadow_scorer": {
                        "predicted_action": "eat",
                        "supported_prediction": True,
                        "would_change_action": True,
                        "score_source": "sequence_key",
                    }
                },
                {
                    "sequence_history_shadow_scorer": {
                        "predicted_action": None,
                        "supported_prediction": False,
                        "would_change_action": False,
                        "score_source": "no_data_supported_action",
                    }
                },
                {"score": 0.0},
            ]
        )

        self.assertEqual(diagnostics["total_decision_count"], 3)
        self.assertEqual(diagnostics["decision_count"], 2)
        self.assertEqual(diagnostics["supported_count"], 1)
        self.assertEqual(diagnostics["would_change_count"], 1)
        self.assertEqual(diagnostics["unsupported_prediction_count"], 1)
        self.assertEqual(diagnostics["no_prediction_count"], 1)
        self.assertEqual(diagnostics["predicted_action_counts"], {"eat": 1})
        self.assertEqual(
            diagnostics["score_source_counts"],
            {"no_data_supported_action": 1, "sequence_key": 1},
        )
        json.dumps(diagnostics)

    def test_shadow_trajectory_output_is_dataset_loadable_and_scalar_safe(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scorer_path = tmp / "shadow-scorer.json"
            shadow_output = tmp / "shadow-report.json"
            baseline_output = tmp / "baseline-report.json"
            trajectory_dir = tmp / "trajectories"
            scorer_path.write_text(
                json.dumps({"artifact": _eat_shadow_scorer().artifact}),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolution_sim.cli.mind_v3_evaluate",
                    "--seeds",
                    "5",
                    "--ticks",
                    "2",
                    "--output",
                    str(shadow_output),
                    "--trajectory-output-dir",
                    str(trajectory_dir),
                    "--sequence-history-shadow-scorer",
                    str(scorer_path),
                ],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolution_sim.cli.mind_v3_evaluate",
                    "--seeds",
                    "5",
                    "--ticks",
                    "2",
                    "--output",
                    str(baseline_output),
                ],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            trajectory_paths = sorted(
                trajectory_dir.glob("open-mind-v3-*-2.jsonl.gz")
            )
            self.assertEqual(len(trajectory_paths), 1)
            dataset = load_trajectory_jsonl(trajectory_paths[0])
            self.assertGreater(dataset.record_count, 0)
            diagnostic_records = [
                record.get("policy_decision_diagnostics")
                for record in dataset.records
                if isinstance(record.get("policy_decision_diagnostics"), dict)
            ]
            self.assertTrue(diagnostic_records)
            for diagnostics in diagnostic_records:
                self.assertNotIn("sequence_history_shadow_scorer", diagnostics)
                self.assertIn(
                    "sequence_history_shadow_predicted_action",
                    diagnostics,
                )
                self.assertIn("sequence_history_shadow_score_source", diagnostics)
                self.assertTrue(
                    all(_scalar_safe(value) for value in diagnostics.values())
                )

            shadow_report = json.loads(shadow_output.read_text(encoding="utf-8"))
            baseline_report = json.loads(baseline_output.read_text(encoding="utf-8"))
            shadow_aggregate = shadow_report["comparison"]["mind_v3"]["aggregate"]
            baseline_aggregate = baseline_report["comparison"]["mind_v3"][
                "aggregate"
            ]
            shadow_diagnostics = shadow_aggregate[
                "sequence_history_shadow_scorer_diagnostics"
            ]
            self.assertGreater(shadow_diagnostics["decision_count"], 0)
            self.assertIn("supported_count", shadow_diagnostics)
            self.assertIn("would_change_count", shadow_diagnostics)
            self.assertIn("score_source_counts", shadow_diagnostics)
            self.assertEqual(
                shadow_aggregate["requested_action_counts"],
                baseline_aggregate["requested_action_counts"],
            )
            self.assertEqual(
                shadow_aggregate["resolved_action_counts"],
                baseline_aggregate["resolved_action_counts"],
            )


def _policy(
    *,
    sequence_history_shadow_scorer: SequenceHistoryShadowScorer | None = None,
) -> MindV3EvolutionPolicy:
    policy = MindV3EvolutionPolicy(
        seed=7,
        sequence_history_shadow_scorer=sequence_history_shadow_scorer,
    )
    metadata = _metadata()
    metadata["action_head_bias"]["drink"] = 1.0
    policy.register_agent_mind(agent_id=3, metadata=metadata)
    return policy


def _metadata() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "state_size": mind_v3_parameter_count(
            architecture=MIND_V3_CONTROLLER_ARCHITECTURE
        ),
        "architecture": MIND_V3_CONTROLLER_ARCHITECTURE,
        "action_head_weights": {
            action: [0.0] * MIND_V3_HIDDEN_UNITS for action in ACTION_NAMES
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


def _action_mask(*enabled_actions: str) -> dict[str, bool]:
    enabled = set(enabled_actions)
    return {action: action in enabled for action in ACTION_NAMES}


def _transition_record(
    action: str,
    *,
    agent_id: int = 3,
    observation_input: dict[str, object] | None = None,
    action_mask: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "tick": 0,
        "agent_id": agent_id,
        "policy_id": "mind_v3_autonomous_evolution_policy",
        "policy_version": "mind_v3_autonomous_evolution_policy_v1",
        "action_source": "mind_v3_autonomous_evolution_policy_v1",
        "observation_input": observation_input or {"values": _observation_values()},
        "action_mask": action_mask or _action_mask("drink", "eat"),
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
                "food_source": "plant" if action == "eat" else None,
            },
            "drinking": {"drank": action == "drink"},
            "passive": {"died_after_action": False},
            "resource_gain": 0.1 if action == "eat" else 0.0,
            "died": False,
            "reproduced": False,
            "reproduction_ready_after": False,
        },
        "reward": {"total": 0.2},
    }


def _eat_shadow_scorer() -> SequenceHistoryShadowScorer:
    mask_token = "".join("1" if action in {"drink", "eat"} else "0" for action in ACTION_NAMES)
    artifact = _artifact()
    artifact["backoff"]["mask_action_counts"][mask_token] = _counts(eat=7, drink=1)
    return SequenceHistoryShadowScorer(artifact=artifact)


def _empty_shadow_scorer() -> SequenceHistoryShadowScorer:
    return SequenceHistoryShadowScorer(artifact=_artifact())


def _artifact() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
        "artifact_policy": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_POLICY,
        "sequence_lookup": {"sequence_counts": {}},
        "backoff": {
            "mask_action_counts": {},
            "action_counts": _counts(),
        },
    }


def _counts(**counts: int) -> dict[str, int]:
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


def _scalar_safe(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return False


if __name__ == "__main__":
    unittest.main()
