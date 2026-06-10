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
from evolution_sim.mind.sequence_history_shadow_scorer import (
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_POLICY,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
    SequenceHistoryShadowScorer,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

ROOT = Path(__file__).resolve().parents[2]
READY_V139_CLASSIFICATION = (
    "sequence_history_shadow_scorer_passed_diagnostics_only_ready_for_shadow_runtime_logging"
)


class MindV3SequenceHistoryLiveAbTests(unittest.TestCase):
    def test_live_ab_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:sequence-history-live-ab"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_sequence_history_live_ab"
            ),
        )

    def test_override_disabled_preserves_no_shadow_action(self) -> None:
        baseline = _policy()
        disabled = _policy(sequence_history_shadow_scorer=_eat_scorer())
        observation = _observation(agent_id=3)
        action_mask = _action_mask("drink", "eat")

        baseline_decision = baseline.decide(dict(observation), dict(action_mask))
        disabled_decision = disabled.decide(dict(observation), dict(action_mask))

        self.assertEqual(baseline_decision.requested_action, "drink")
        self.assertEqual(disabled_decision.requested_action, "drink")
        shadow = disabled_decision.diagnostics["sequence_history_shadow_scorer"]
        self.assertFalse(shadow["override_applied"])
        self.assertEqual(shadow["override_rejected_reason"], "override_disabled")
        self.assertFalse(shadow["runtime_action_selection_changed"])

    def test_override_enabled_changes_only_supported_valid_prediction(self) -> None:
        policy = _policy(
            sequence_history_shadow_scorer=_eat_scorer(),
            sequence_history_action_override=True,
            sequence_history_action_override_source_integrity_passed=True,
        )
        decision = policy.decide(
            _observation(agent_id=3),
            _action_mask("drink", "eat"),
        )

        self.assertEqual(decision.requested_action, "eat")
        shadow = decision.diagnostics["sequence_history_shadow_scorer"]
        self.assertEqual(shadow["original_mind_v3_requested_action"], "drink")
        self.assertEqual(shadow["predicted_action"], "eat")
        self.assertTrue(shadow["override_applied"])
        self.assertIsNone(shadow["override_rejected_reason"])
        self.assertTrue(shadow["runtime_action_selection_changed"])

    def test_invalid_unsupported_and_no_data_predictions_never_override(self) -> None:
        cases = [
            (
                _fake_scorer(
                    predicted_action="eat",
                    supported_prediction=True,
                    score_source="unit_invalid",
                ),
                _action_mask("drink"),
                "invalid_prediction",
            ),
            (
                _fake_scorer(
                    predicted_action="eat",
                    supported_prediction=False,
                    score_source="unit_unsupported",
                ),
                _action_mask("drink", "eat"),
                "unsupported_prediction",
            ),
            (
                _fake_scorer(
                    predicted_action=None,
                    supported_prediction=False,
                    score_source="no_data_supported_action",
                ),
                _action_mask("drink", "eat"),
                "no_prediction",
            ),
        ]
        for scorer, action_mask, reason in cases:
            with self.subTest(reason=reason):
                policy = _policy(
                    sequence_history_shadow_scorer=scorer,
                    sequence_history_action_override=True,
                    sequence_history_action_override_source_integrity_passed=True,
                )
                decision = policy.decide(_observation(agent_id=3), action_mask)
                shadow = decision.diagnostics["sequence_history_shadow_scorer"]

                self.assertEqual(decision.requested_action, "drink")
                self.assertFalse(shadow["override_applied"])
                self.assertEqual(shadow["override_rejected_reason"], reason)
                self.assertFalse(shadow["runtime_action_selection_changed"])

    def test_override_requires_source_integrity_authorization(self) -> None:
        policy = _policy(
            sequence_history_shadow_scorer=_eat_scorer(),
            sequence_history_action_override=True,
            sequence_history_action_override_source_integrity_passed=False,
        )

        decision = policy.decide(
            _observation(agent_id=3),
            _action_mask("drink", "eat"),
        )

        self.assertEqual(decision.requested_action, "drink")
        shadow = decision.diagnostics["sequence_history_shadow_scorer"]
        self.assertFalse(shadow["override_applied"])
        self.assertEqual(shadow["override_rejected_reason"], "source_integrity_failed")

    def test_override_trajectory_output_reloads_and_flattens_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scorer_path = tmp / "v139-report.json"
            output = tmp / "eval.json"
            trajectory_dir = tmp / "trajectories"
            scorer_path.write_text(
                json.dumps(_v139_report(_eat_scorer())),
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
                    str(output),
                    "--trajectory-output-dir",
                    str(trajectory_dir),
                    "--sequence-history-shadow-scorer",
                    str(scorer_path),
                    "--sequence-history-action-override",
                ],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            aggregate = report["comparison"]["mind_v3"]["aggregate"]
            shadow = aggregate["sequence_history_shadow_scorer_diagnostics"]
            self.assertGreater(shadow["override_applied_count"], 0)
            trajectory_paths = sorted(
                trajectory_dir.glob("open-mind-v3-*-2.jsonl.gz")
            )
            self.assertEqual(len(trajectory_paths), 1)
            dataset = load_trajectory_jsonl(trajectory_paths[0])
            diagnostics = [
                record.get("policy_decision_diagnostics")
                for record in dataset.records
                if isinstance(record.get("policy_decision_diagnostics"), dict)
            ]
            self.assertTrue(diagnostics)
            for item in diagnostics:
                self.assertNotIn("sequence_history_shadow_scorer", item)
                self.assertIn("sequence_history_shadow_override_applied", item)
                self.assertTrue(all(_scalar_safe(value) for value in item.values()))


def _policy(
    *,
    sequence_history_shadow_scorer: object | None = None,
    sequence_history_action_override: bool = False,
    sequence_history_action_override_source_integrity_passed: bool = False,
) -> MindV3EvolutionPolicy:
    policy = MindV3EvolutionPolicy(
        seed=7,
        sequence_history_shadow_scorer=sequence_history_shadow_scorer,
        sequence_history_action_override=sequence_history_action_override,
        sequence_history_action_override_source_integrity_passed=(
            sequence_history_action_override_source_integrity_passed
        ),
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


def _eat_scorer() -> SequenceHistoryShadowScorer:
    artifact = _artifact()
    mask_token = "".join(
        "1" if action in {"drink", "eat"} else "0" for action in ACTION_NAMES
    )
    artifact["backoff"]["mask_action_counts"][mask_token] = _counts(eat=7, drink=1)
    artifact["backoff"]["action_counts"] = _counts(eat=7, drink=1)
    return SequenceHistoryShadowScorer(artifact=artifact)


def _fake_scorer(
    *,
    predicted_action: str | None,
    supported_prediction: bool,
    score_source: str,
) -> object:
    class FakeScorer:
        def score(self, *, sequence_keys: object, valid_actions: object) -> dict[str, object]:
            return {
                "predicted_action": predicted_action,
                "score_source": score_source,
                "supported_prediction": supported_prediction,
                "scores": {},
            }

    return FakeScorer()


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


def _v139_report(scorer: SequenceHistoryShadowScorer) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
        "source_integrity": {"passed": True},
        "support_floors": {"passed": True},
        "classification": {"primary": READY_V139_CLASSIFICATION},
        "artifact_roundtrip": {
            "loaded_artifact_scores_match_pre_serialization": True
        },
        "artifact_feature_leakage_scan": {"passed": True},
        "artifact": dict(scorer.artifact),
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
