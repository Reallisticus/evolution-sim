from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_sequence_continuation_scorer
from evolution_sim.mind.branch_action_oracle_labels import (
    build_branch_action_oracle_label_report,
)
from evolution_sim.mind.branch_sequence_continuation_scorer import (
    MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
    _sequence_target_metrics,
    build_branch_sequence_continuation_scorer_report,
)
from python.tests.test_mind_v3_branch_action_oracle_labels import (
    _synthetic_audit_report,
)


class MindV3BranchSequenceContinuationScorerTests(unittest.TestCase):
    def test_branch_sequence_continuation_scorer_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-sequence-continuation-scorer"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_sequence_continuation_scorer"
            ),
        )

    def test_sequence_continuation_scorer_reports_schema_and_contract(self) -> None:
        support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")

        report = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
        )
        self.assertFalse(report["contract"]["runtime_policy_trained"])
        feature_contract = report["contract"]["feature_contract"]
        self.assertFalse(feature_contract["uses_private_world_state"])
        self.assertFalse(feature_contract["uses_fixture_identity"])
        self.assertFalse(feature_contract["uses_seed_id_as_runtime_feature"])
        self.assertFalse(feature_contract["uses_logged_action_as_runtime_fallback"])
        target_contract = report["contract"]["target_contract"]
        self.assertTrue(target_contract["uses_target_horizon_trace_as_target"])
        self.assertTrue(target_contract["models_after_first_action_deltas"])
        self.assertFalse(target_contract["first_action_imitation_target"])
        self.assertEqual(report["coverage"]["support_strict_seed_overlap"], [])
        self.assertEqual(report["coverage"]["strict_seed_training_leak_count"], 0)
        self.assertIn("decision_rule_reports", report)
        self.assertFalse(
            report["acceptance"]["v94_sequence_continuation_scorer_accepted"]
        )

    def test_sequence_target_metrics_model_after_first_action_deltas(self) -> None:
        row = {"before": {"energy_ratio": 0.2, "hydration_ratio": 0.5, "health_ratio": 0.9}}
        candidate = {
            "action": "move_south",
            "terminal_alive_agents": 0,
            "births": 0,
            "deaths": 3,
            "first_action_outcome": {
                "record_found": True,
                "alive_after": True,
                "died": False,
                "energy_ratio_after": 0.18,
                "hydration_ratio_after": 0.48,
                "health_ratio_after": 0.9,
                "resource_gain": 0.0,
            },
            "target_horizon_trace": [
                {
                    "record_found": True,
                    "horizon_tick_delta": 1,
                    "alive_after": True,
                    "energy_ratio_after": 0.24,
                    "hydration_ratio_after": 0.46,
                    "health_ratio_after": 0.9,
                    "resource_gain": 0.12,
                },
                {
                    "record_found": True,
                    "horizon_tick_delta": 2,
                    "alive_after": True,
                    "energy_ratio_after": 0.28,
                    "hydration_ratio_after": 0.44,
                    "health_ratio_after": 0.9,
                    "resource_gain": 0.18,
                },
            ],
            "population_horizon_trace": [
                {
                    "horizon_tick_delta": 2,
                    "alive_agents": 0,
                    "births": 0,
                    "deaths": 3,
                    "target_alive": True,
                    "target_energy_ratio": 0.28,
                    "target_hydration_ratio": 0.44,
                    "target_health_ratio": 0.9,
                }
            ],
        }

        metrics = _sequence_target_metrics(row, candidate)

        self.assertEqual(metrics["first_action_resource_gain"], 0.0)
        self.assertEqual(metrics["target_resource_gain_after_first"], 0.3)
        self.assertEqual(metrics["target_survival_area_after_first"], 1.0)
        self.assertEqual(metrics["terminal_target_alive"], 1.0)
        self.assertEqual(metrics["terminal_alive_agents"], 0)
        self.assertEqual(metrics["births"], 0)

    def test_sequence_continuation_scorer_json_is_deterministic(self) -> None:
        support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")

        first = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
        )
        second = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
        )

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_branch_sequence_continuation_scorer_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            support_path = tmp / "support.json"
            strict_path = tmp / "strict.json"
            output_path = tmp / "sequence.json"
            support_path.write_text(
                json.dumps(_labels_with_trace_targets(seed_start=101, branch_prefix="support")),
                encoding="utf-8",
            )
            strict_path.write_text(
                json.dumps(_labels_with_trace_targets(seed_start=13, branch_prefix="strict")),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_sequence_continuation_scorer",
                    "--support-branch-action-oracle-labels",
                    str(support_path),
                    "--strict-branch-action-oracle-labels",
                    str(strict_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_sequence_continuation_scorer.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["support_label_count"], 3)


def _labels_with_trace_targets(*, seed_start: int, branch_prefix: str) -> dict[str, object]:
    labels = build_branch_action_oracle_label_report(_synthetic_audit_report())
    labels = copy.deepcopy(labels)
    for index, label in enumerate(labels["labels"]):  # type: ignore[index]
        branch_id = f"{branch_prefix}-{index}"
        label["branch_id"] = branch_id
        label["source"]["seed"] = seed_start + index
        label["source"]["branch_id"] = branch_id
        for action in label["action_value_targets"]["actions"]:
            action_name = str(action["action"])
            action["first_action_outcome"] = _first_action(action_name)
            action["target_horizon_trace"] = _target_trace(action_name)
            action["population_horizon_trace"] = _population_trace(action_name)
    return labels


def _first_action(action: str) -> dict[str, object]:
    return {
        "record_found": True,
        "alive_after": action != "drink",
        "died": action == "drink",
        "energy_ratio_after": 0.42 if action == "eat" else 0.36,
        "hydration_ratio_after": 0.74 if action == "drink" else 0.62,
        "health_ratio_after": 0.9,
        "resource_gain": 0.2 if action == "eat" else 0.0,
    }


def _target_trace(action: str) -> list[dict[str, object]]:
    alive = action != "drink"
    return [
        {
            "record_found": True,
            "horizon_tick_delta": 1,
            "alive_after": alive,
            "energy_ratio_after": 0.45 if alive else 0.0,
            "hydration_ratio_after": 0.6 if alive else 0.0,
            "health_ratio_after": 0.88 if alive else 0.0,
            "resource_gain": 0.1 if action == "eat" else 0.0,
        },
        {
            "record_found": alive,
            "horizon_tick_delta": 2,
            "alive_after": alive,
            "energy_ratio_after": 0.48 if alive else 0.0,
            "hydration_ratio_after": 0.58 if alive else 0.0,
            "health_ratio_after": 0.86 if alive else 0.0,
            "resource_gain": 0.12 if action == "eat" else 0.0,
        },
    ]


def _population_trace(action: str) -> list[dict[str, object]]:
    alive = action != "drink"
    return [
        {
            "horizon_tick_delta": 2,
            "alive_agents": 3 if alive else 2,
            "births": 1 if action == "eat" else 0,
            "deaths": 0 if alive else 1,
            "target_alive": alive,
            "target_energy_ratio": 0.48 if alive else 0.0,
            "target_hydration_ratio": 0.58 if alive else 0.0,
            "target_health_ratio": 0.86 if alive else 0.0,
        }
    ]


if __name__ == "__main__":
    unittest.main()
