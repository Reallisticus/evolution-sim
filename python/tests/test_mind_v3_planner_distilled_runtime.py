from __future__ import annotations

import copy
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_evaluate
from evolution_sim.mind.branch_planner_distillation_audit import (
    build_branch_planner_distillation_audit_report,
    score_distilled_planner_artifact,
)
from evolution_sim.mind.branch_sequence_continuation_scorer import (
    build_branch_sequence_continuation_scorer_report,
)
from evolution_sim.mind.branch_utility_risk_audit import _utility_rows
from evolution_sim.mind.evolution import MIND_V3_POLICY_ID, MIND_V3_POLICY_VERSION
from evolution_sim.mind.v3_planner_distilled import (
    MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
    MindV3PlannerDistilledArtifactError,
    load_mind_v3_planner_distilled_artifact,
    score_mind_v3_planner_distilled_runtime,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy
from python.tests.test_mind_v3_branch_sequence_continuation_scorer import (
    _labels_with_trace_targets,
)


class MindV3PlannerDistilledRuntimeTests(unittest.TestCase):
    def test_loader_accepts_raw_artifact_and_v96_report_payload(self) -> None:
        report = _planner_distillation_report()
        artifact = report["distilled_artifact"]

        with TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "artifact.json"
            report_path = Path(tmpdir) / "report.json"
            raw_path.write_text(json.dumps(artifact), encoding="utf-8")
            report_path.write_text(json.dumps(report), encoding="utf-8")

            raw_loaded = load_mind_v3_planner_distilled_artifact(raw_path)
            report_loaded = load_mind_v3_planner_distilled_artifact(report_path)

        self.assertEqual(
            raw_loaded["schema_version"],
            MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(raw_loaded, report_loaded)

    def test_loader_rejects_forbidden_runtime_example_keys(self) -> None:
        report = _planner_distillation_report()
        artifact = copy.deepcopy(report["distilled_artifact"])
        artifact["sequence_support_examples"][0]["branch_id"] = "leak"

        with self.assertRaises(MindV3PlannerDistilledArtifactError):
            load_mind_v3_planner_distilled_artifact(artifact)

    def test_online_candidate_scores_match_offline_branch_row(self) -> None:
        report = _planner_distillation_report()
        artifact = report["distilled_artifact"]
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        row = _utility_rows(strict["labels"])[0]
        policy_state = row["policy_state"]

        offline = score_distilled_planner_artifact(row=row, artifact=artifact)
        online = score_mind_v3_planner_distilled_runtime(
            artifact=artifact,
            observation_input=policy_state["observation_input"],
            action_mask=policy_state["action_mask"],
            public_history_trace=policy_state.get("public_history_trace", []),
        )

        self.assertEqual(online["selected_action"], offline["selected_action"])
        self.assertEqual(online["candidate_scores"], offline["candidate_scores"])

    def test_policy_updates_public_history_after_transition(self) -> None:
        report = _planner_distillation_report()
        artifact = report["distilled_artifact"]
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        row = _utility_rows(strict["labels"])[0]
        policy_state = row["policy_state"]
        action_mask = dict(policy_state["action_mask"])
        policy = MindV3EvolutionPolicy(seed=13, planner_distilled_artifact=artifact)
        observation = {
            "metadata": {"agent_id": 101},
            "observation_input": policy_state["observation_input"],
            "self": {"trophic_role": "omnivore", "meat_mode": "scavenger"},
        }

        first = policy.decide(observation, action_mask)
        self.assertEqual(
            first.diagnostics["planner_distilled_public_history_steps"],
            0,
        )
        policy.observe_transition(
            {
                "tick": 0,
                "agent_id": 101,
                "policy_id": MIND_V3_POLICY_ID,
                "policy_version": MIND_V3_POLICY_VERSION,
                "action_source": MIND_V3_POLICY_VERSION,
                "requested_action": first.requested_action,
                "resolved_action": first.requested_action,
                "action_valid": True,
                "resolution_action_valid": True,
                "moved": first.requested_action.startswith("move_"),
                "action_mask": action_mask,
                "before": {
                    "x": 3,
                    "y": 3,
                    "energy_ratio": 0.5,
                    "hydration_ratio": 0.5,
                    "health_ratio": 1.0,
                    "alive": True,
                },
                "after": {
                    "x": 3,
                    "y": 3,
                    "energy_ratio": 0.49,
                    "hydration_ratio": 0.49,
                    "health_ratio": 1.0,
                    "alive": True,
                },
                "observation_input": policy_state["observation_input"],
                "outcome": {
                    "feeding": {"ate": False},
                    "drinking": {"drank": False},
                    "passive": {"died_after_action": False},
                    "resource_gain": 0.0,
                    "died": False,
                    "reproduced": False,
                    "reproduction_ready_after": False,
                },
                "reward": {"total": 0.0},
            }
        )

        second = policy.decide(observation, action_mask)
        self.assertEqual(
            second.diagnostics["planner_distilled_public_history_steps"],
            1,
        )

    def test_evaluate_cli_accepts_planner_distilled_artifact(self) -> None:
        report = _planner_distillation_report()

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_path = tmp / "v96-report.json"
            output_path = tmp / "eval.json"
            artifact_path.write_text(json.dumps(report), encoding="utf-8")
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evaluate",
                        "--seeds",
                        "5",
                        "--ticks",
                        "1",
                        "--planner-distilled-artifact",
                        str(artifact_path),
                        "--compare-linear-baseline",
                        "--fixture-suite",
                        "basic",
                        "--fixture-names",
                        "carrion_only",
                        "--fixture-seeds",
                        "13",
                        "--fixture-ticks",
                        "1",
                        "--output",
                        str(output_path),
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evaluate.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("mind_v3_v97_promotion_candidate_passed=False", stdout.getvalue())
        self.assertEqual(
            payload["policy"]["planner_distilled_artifact_schema_version"],
            MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertIn("planner_vs_linear_delta", payload["comparison"])
        self.assertFalse(
            payload["v97_planner_distilled_promotion"][
                "promotion_candidate_passed"
            ]
        )
        self.assertFalse(payload["promotion_candidate_passed"])
        self.assertGreater(payload["promotion_blocker_count"], 0)
        self.assertEqual(
            payload["comparison"]["mind_v3"]["aggregate"][
                "heuristic_action_source_count"
            ],
            0,
        )


def _planner_distillation_report() -> dict[str, object]:
    support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
    strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
    sequence = build_branch_sequence_continuation_scorer_report(
        support_branch_action_oracle_labels=support,
        strict_branch_action_oracle_labels=strict,
    )
    return build_branch_planner_distillation_audit_report(
        support_branch_action_oracle_labels=support,
        strict_branch_action_oracle_labels=strict,
        branch_sequence_continuation_scorer_report=sequence,
    )


if __name__ == "__main__":
    unittest.main()
