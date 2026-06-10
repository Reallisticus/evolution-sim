from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from evolution_sim.cli import mind_v3_evaluate as evaluate_cli
from evolution_sim.cli import mind_v3_transition_value_live_ab as live_ab_cli
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    SELF_INPUT_FIELDS,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_ARCHITECTURE,
    MIND_V3_CONTROLLER_SCHEMA_VERSION,
    MIND_V3_HIDDEN_UNITS,
    mind_v3_parameter_count,
)
from evolution_sim.mind.rollout_context import RolloutContextState
from evolution_sim.mind.transition_value_scorer import (
    MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
    build_transition_value_scorer_report,
    load_transition_value_scorer_artifact,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

ROOT = Path(__file__).resolve().parents[2]


class MindV3TransitionValueScorerTests(unittest.TestCase):
    def test_transition_value_scorer_has_npm_entrypoints(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        scripts = package["scripts"]

        self.assertIn("sim:mind:v3:transition-value-scorer", scripts)
        self.assertIn("sim:mind:v3:transition-value-live-ab", scripts)
        self.assertIn(
            "evolution_sim.cli.mind_v3_transition_value_scorer",
            scripts["sim:mind:v3:transition-value-scorer"],
        )
        self.assertIn(
            "evolution_sim.cli.mind_v3_transition_value_live_ab",
            scripts["sim:mind:v3:transition-value-live-ab"],
        )

    def test_utility_scorer_prefers_high_outcome_action_not_majority_count(self) -> None:
        train = _dataset(
            seed=1,
            records=[
                _record("eat", agent_id=1, energy_delta=-0.2),
                _record("eat", agent_id=2, energy_delta=-0.2),
                _record("eat", agent_id=3, energy_delta=-0.2),
                _record("stay", agent_id=4),
                _record("drink", agent_id=5, hydration_delta=0.6),
            ],
        )
        heldout = _dataset(seed=5, records=[_record("stay", agent_id=6)])

        build = build_transition_value_scorer_report(
            train_trajectory_datasets=[train],
            strict_heldout_trajectory_datasets=[heldout],
            strict_seed_values=(5,),
        )
        report = build.report
        scorer = load_transition_value_scorer_artifact(report)
        score = scorer.score(
            observation_input={"values": _observation_values()},
            valid_action_mask=_action_mask("stay", "eat", "drink"),
            state=RolloutContextState(),
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["support_floors"]["passed"])
        self.assertTrue(report["contract"]["explicit_opt_in_live_ab_eligible"])
        self.assertTrue(
            report["contract"]["runtime_policy_change_requires_explicit_flag"]
        )
        self.assertFalse(report["contract"]["promotion_authorized"])
        self.assertTrue(report["artifact"]["explicit_opt_in_live_ab_eligible"])
        self.assertTrue(
            report["artifact"]["runtime_policy_change_requires_explicit_flag"]
        )
        self.assertFalse(report["artifact"]["runtime_action_selection_authorized"])
        self.assertFalse(report["artifact"]["promotion_authorized"])
        self.assertTrue(
            report["artifact_roundtrip"][
                "loaded_artifact_scores_match_pre_serialization"
            ]
        )
        self.assertEqual(score["predicted_action"], "drink")
        self.assertGreater(
            score["valid_action_support_counts"]["eat"],
            score["valid_action_support_counts"]["drink"],
        )

    def test_policy_override_logs_required_transition_value_fields(self) -> None:
        build = build_transition_value_scorer_report(
            train_trajectory_datasets=[
                _dataset(
                    seed=1,
                    records=[
                        _record("eat", agent_id=1, energy_delta=-0.2),
                        _record("stay", agent_id=2),
                        _record("drink", agent_id=3, hydration_delta=0.6),
                    ],
                )
            ],
            strict_heldout_trajectory_datasets=[
                _dataset(seed=5, records=[_record("stay", agent_id=4)])
            ],
            strict_seed_values=(5,),
        )
        scorer = load_transition_value_scorer_artifact(build.report)
        policy = MindV3EvolutionPolicy(
            seed=7,
            transition_value_scorer=scorer,
            transition_value_action_override=True,
            transition_value_action_override_source_integrity_passed=True,
        )
        metadata = _metadata()
        metadata["action_head_bias"]["eat"] = 1.0
        policy.register_agent_mind(agent_id=3, metadata=metadata)

        decision = policy.decide(
            _observation(agent_id=3),
            _action_mask("stay", "eat", "drink"),
        )
        diagnostics = decision.diagnostics["transition_value_scorer"]
        flattened = evaluate_cli._trajectory_safe_policy_decision_diagnostics(
            decision.diagnostics
        )

        self.assertEqual(decision.requested_action, "drink")
        self.assertEqual(diagnostics["original_mind_v3_requested_action"], "eat")
        self.assertEqual(diagnostics["predicted_action"], "drink")
        self.assertEqual(diagnostics["final_requested_action"], "drink")
        self.assertTrue(diagnostics["override_applied"])
        self.assertIsNone(diagnostics["override_rejected_reason"])
        self.assertGreater(diagnostics["utility_margin"], 0.0)
        self.assertIn("transition_value_utility_margin", flattened)
        self.assertNotIn("transition_value_score", flattened)
        self.assertTrue(all(_scalar_safe(value) for value in flattened.values()))

    def test_evaluator_rejects_non_ready_transition_value_report(self) -> None:
        integrity = evaluate_cli._transition_value_scorer_source_integrity(
            {
                "schema_version": MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
                "artifact": {
                    "schema_version": MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION
                },
                "classification": {"primary": "transition_value_scorer_blocked"},
                "source_integrity": {"passed": True},
                "support_floors": {"passed": False},
                "artifact_roundtrip": {
                    "loaded_artifact_scores_match_pre_serialization": True
                },
                "artifact_feature_leakage_scan": {"passed": True},
                "action_support": {"heuristic_action_source_count": 0},
            }
        )

        self.assertFalse(integrity["passed"])
        self.assertIn("v142_support_floors_not_passed", integrity["failures"])
        self.assertIn("v142_classification_not_ready", integrity["failures"])

    def test_live_ab_records_broad_fixture_for_first_broad_regression(self) -> None:
        artifact_report = {
            "artifact_roundtrip": {
                "loaded_artifact_scores_match_pre_serialization": True,
                "mismatch_count": 0,
            },
        }
        broad = {
            "override": {
                "aggregate": {
                    "heuristic_action_source_count": 0,
                    "unsupported_requested_action_count": 0,
                    "dominant_requested_action_share": 0.25,
                    "requested_action_counts": {"eat": 1},
                    "transition_value_scorer_diagnostics": {
                        "override_applied_count": 1,
                        "decision_count": 2,
                    },
                }
            },
            "per_seed_delta": [
                {"seed": 5, "alive_delta": -1, "births_delta": 0}
            ],
        }
        carrion = {
            "override": {
                "aggregate": {
                    "heuristic_action_source_count": 0,
                    "unsupported_requested_action_count": 0,
                    "dominant_requested_action_share": 0.2,
                    "requested_action_counts": {"stay": 1},
                    "transition_value_scorer_diagnostics": {
                        "override_applied_count": 0,
                        "decision_count": 1,
                    },
                }
            },
            "baseline_fixture_gate": {"blockers": [{"id": "baseline"}]},
            "override_fixture_gate": {"blockers": [{"id": "override"}]},
            "aggregate_delta": {"alive_agents_mean": 0.0},
        }

        acceptance = live_ab_cli._acceptance(
            broad=broad,
            carrion=carrion,
            artifact_report=artifact_report,
            broad_seeds=live_ab_cli.STRICT_BROAD_SEEDS,
            fixture_seeds=live_ab_cli.STRICT_CARRION_FIXTURE_SEEDS,
            ticks=live_ab_cli.STRICT_TICKS,
            fixture_ticks=live_ab_cli.STRICT_TICKS,
        )

        self.assertFalse(acceptance["passed"])
        self.assertEqual(
            acceptance["first_failed_floor"],
            "broad_seed_5_alive_no_regression",
        )
        self.assertEqual(acceptance["first_failing_seed"], 5)
        self.assertEqual(acceptance["first_failing_fixture"], "broad")


def _dataset(*, seed: int, records: list[dict[str, object]]) -> TrajectoryJsonlDataset:
    return TrajectoryJsonlDataset(
        path=Path(
            f"output/mind/test-trajectories/open-mind-v3-{seed}-120.jsonl.gz"
        ),
        header={"run_id": f"seed-{seed}"},
        records=tuple(records),
        footer={"provenance": {"source_seeds": [seed], "split_id": "test"}},
    )


def _record(
    action: str,
    *,
    agent_id: int,
    energy_delta: float = 0.0,
    hydration_delta: float = 0.0,
    health_delta: float = 0.0,
    reproduced: bool = False,
) -> dict[str, object]:
    before = {
        "x": 1,
        "y": 1,
        "energy_ratio": 0.5,
        "hydration_ratio": 0.5,
        "health_ratio": 0.8,
        "alive": True,
    }
    after = {
        "x": 1,
        "y": 1,
        "energy_ratio": 0.5 + energy_delta,
        "hydration_ratio": 0.5 + hydration_delta,
        "health_ratio": 0.8 + health_delta,
        "alive": True,
    }
    return {
        "tick": 0,
        "agent_id": agent_id,
        "policy_id": "mind_v3_autonomous_evolution_policy",
        "policy_version": "mind_v3_autonomous_evolution_policy_v1",
        "action_source": "mind_v3_autonomous_evolution_policy_v1",
        "observation_input": {"values": _observation_values()},
        "action_mask": _action_mask("stay", "eat", "drink"),
        "requested_action": action,
        "resolved_action": action,
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": action.startswith("move_"),
        "before": before,
        "after": after,
        "outcome": {
            "feeding": {"ate": action == "eat"},
            "drinking": {"drank": action == "drink"},
            "passive": {"died_after_action": False},
            "resource_gain": max(0.0, energy_delta),
            "died": False,
            "reproduced": reproduced,
            "reproduction_ready_after": False,
        },
        "reward": {"total": 0.0},
    }


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


def _scalar_safe(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return False


if __name__ == "__main__":
    unittest.main()
