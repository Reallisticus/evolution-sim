from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_action_oracle_labels
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_action_oracle_audit import (
    DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS,
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.branch_action_oracle_labels import (
    MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
    _public_history_feature_vector,
    build_branch_action_oracle_label_report,
)


class MindV3BranchActionOracleLabelTests(unittest.TestCase):
    def test_branch_action_oracle_labels_have_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-action-oracle-labels"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_action_oracle_labels"
            ),
        )

    def test_label_report_preserves_policy_visible_oracle_targets(self) -> None:
        report = build_branch_action_oracle_label_report(_synthetic_audit_report())

        self.assertEqual(
            report["schema_version"],
            MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
        )
        self.assertTrue(report["acceptance"]["label_archive_acceptance_passed"])
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["label_count"], 3)
        self.assertEqual(aggregate["oracle_action_counts"], {"drink": 1, "eat": 1, "stay": 1})
        self.assertEqual(aggregate["material_oracle_gain_label_count"], 2)
        self.assertEqual(aggregate["terminal_alive_gain_total_vs_logged"], 1)
        self.assertEqual(aggregate["birth_gain_total_vs_logged"], 1)
        self.assertEqual(aggregate["unsupported_oracle_action_count"], 0)
        self.assertEqual(aggregate["conflicting_observation_digest_count"], 0)
        self.assertEqual(report["support_probe"]["eligible_label_count"], 3)
        self.assertFalse(
            report["support_probe"]["materially_supports_runtime_classifier"]
        )
        self.assertIn("action_conditioned_value_ranker", report["support_probes"])
        self.assertFalse(
            report["support_probes"]["action_conditioned_value_ranker"][
                "materially_supports_action_value_model"
            ]
        )
        self.assertIn("compact_outcome_world_model", report["support_probes"])
        self.assertFalse(
            report["support_probes"]["compact_outcome_world_model"][
                "materially_supports_compact_world_model"
            ]
        )
        self.assertIn("compact_first_step_world_model", report["support_probes"])
        self.assertFalse(
            report["support_probes"]["compact_first_step_world_model"][
                "materially_supports_first_step_world_model_policy"
            ]
        )
        self.assertIn(
            "first_step_augmented_terminal_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "first_step_augmented_terminal_world_model"
            ]["materially_supports_first_step_augmented_terminal_model"]
        )
        self.assertIn("compact_option_mode_world_model", report["support_probes"])
        self.assertFalse(
            report["support_probes"]["compact_option_mode_world_model"][
                "materially_supports_option_mode_model"
            ]
        )
        self.assertIn(
            "compact_reposition_direction_world_model",
            report["support_probes"],
        )
        self.assertIn(
            "short_horizon_trace_terminal_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"]["short_horizon_trace_terminal_world_model"][
                "materially_supports_short_horizon_trace_model"
            ]
        )
        self.assertIn("actual_horizon_trace_alignment", report["support_probes"])
        self.assertFalse(
            report["support_probes"]["actual_horizon_trace_alignment"][
                "materially_supports_short_horizon_trace_signal"
            ]
        )
        self.assertIn("actual_population_horizon_alignment", report["support_probes"])
        self.assertFalse(
            report["support_probes"]["actual_population_horizon_alignment"][
                "materially_supports_population_horizon_trace_signal"
            ]
        )
        self.assertIn(
            "material_only_actual_population_horizon_alignment",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "material_only_actual_population_horizon_alignment"
            ]["materially_supports_population_horizon_trace_signal"]
        )
        self.assertIn(
            "compact_population_horizon_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"]["compact_population_horizon_world_model"][
                "materially_supports_population_horizon_model"
            ]
        )
        self.assertIn(
            "material_only_compact_population_horizon_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "material_only_compact_population_horizon_world_model"
            ]["materially_supports_population_horizon_model"]
        )
        self.assertIn(
            "policy_observation_population_horizon_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "policy_observation_population_horizon_world_model"
            ]["materially_supports_population_horizon_model"]
        )
        self.assertIn(
            "material_only_policy_observation_population_horizon_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "material_only_policy_observation_population_horizon_world_model"
            ]["materially_supports_population_horizon_model"]
        )
        self.assertIn(
            "policy_observation_history_population_horizon_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "policy_observation_history_population_horizon_world_model"
            ]["materially_supports_population_horizon_model"]
        )
        self.assertIn(
            "material_only_policy_observation_history_population_horizon_world_model",
            report["support_probes"],
        )
        self.assertFalse(
            report["support_probes"][
                "material_only_policy_observation_history_population_horizon_world_model"
            ]["materially_supports_population_horizon_model"]
        )

        first = report["labels"][0]
        self.assertEqual(first["oracle_label"]["action"], "eat")
        self.assertEqual(first["oracle_label"]["logged_action"], "drink")
        self.assertEqual(
            first["policy_state"]["observation_input"]["schema_version"],
            "mind_observation_v3",
        )
        self.assertEqual(set(first["policy_state"]["action_mask"]), set(ACTION_NAMES))
        self.assertEqual(len(first["policy_state"]["public_history_trace"]), 1)
        self.assertEqual(
            first["action_value_targets"]["objective"],
            "lexicographic_terminal_alive_birth_target_alive_deaths_diversity_v1",
        )

    def test_label_report_blocks_dominant_oracle_action_collapse(self) -> None:
        audit = _synthetic_audit_report()
        for result in audit["branch_results"]:
            result["oracle_best_action"] = "eat"

        report = build_branch_action_oracle_label_report(audit)

        self.assertFalse(report["acceptance"]["label_archive_acceptance_passed"])
        self.assertEqual(
            report["acceptance"]["blockers"][0]["reason"],
            "dominant_oracle_action_share_above_cap",
        )

    def test_branch_action_oracle_labels_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audit_path = tmp_path / "audit.json"
            output_path = tmp_path / "labels.json"
            audit_path.write_text(
                json.dumps(_synthetic_audit_report()),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_action_oracle_labels",
                    "--branch-action-oracle-audit",
                    str(audit_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_action_oracle_labels.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
        )
        self.assertEqual(payload["aggregate"]["label_count"], 3)
        self.assertTrue(payload["acceptance"]["label_archive_acceptance_passed"])

    def test_public_history_feature_padding_marks_absent_slots_empty(self) -> None:
        features = _public_history_feature_vector([_history_item()])
        slot_size = len(features) // DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS
        padded_prefix = features[
            : slot_size * (DEFAULT_BRANCH_ACTION_ORACLE_HISTORY_STEPS - 1)
        ]

        self.assertTrue(padded_prefix)
        self.assertTrue(all(value == 0.0 for value in padded_prefix))
        self.assertEqual(features[-slot_size], 1.0)


def _synthetic_audit_report() -> dict[str, object]:
    branch_specs = (
        ("branch-1", "drink", "eat", 1, 0, True),
        ("branch-2", "eat", "drink", 0, 1, True),
        ("branch-3", "drink", "stay", 0, 0, False),
    )
    branch_points = []
    branch_results = []
    for index, (
        branch_id,
        logged_action,
        oracle_action,
        alive_delta,
        birth_delta,
        material,
    ) in enumerate(branch_specs):
        branch_points.append(
            {
                "branch_id": branch_id,
                "seed": 37 + index,
                "fixture": "carrion_only",
                "branch_tick": 16 + index,
                "branch_index": index,
                "record_index": 100 + index,
                "agent_id": 10 + index,
                "logged_action": logged_action,
                "branch_state_digest": f"state-digest-{index}",
                "policy_state": {
                    "observation_input": {
                        "schema_version": "mind_observation_v3",
                        "shape": [1],
                        "values": [index / 10.0],
                    },
                    "observation_digest": f"observation-digest-{index}",
                    "observation_schema": "mind_observation_v3",
                    "action_mask": _action_mask(),
                    "public_history_trace": [_history_item()],
                },
                "valid_actions": ["stay", "eat", "drink"],
            }
        )
        branch_results.append(
            {
                "branch_id": branch_id,
                "seed": 37 + index,
                "fixture": "carrion_only",
                "branch_tick": 16 + index,
                "branch_index": index,
                "record_index": 100 + index,
                "agent_id": 10 + index,
                "base_script": "carrion_then_water",
                "continuation_script": "carrion_then_water",
                "logged_action": logged_action,
                "before": {"alive": True, "energy_ratio": 0.4},
                "context": {"post_carrion_contact": True},
                "oracle_best_action": oracle_action,
                "oracle_changed_action": oracle_action != logged_action,
                "material_oracle_gain": material,
                "oracle_alive_delta_vs_logged": alive_delta,
                "oracle_birth_delta_vs_logged": birth_delta,
                "oracle_target_alive_delta_vs_logged": 0,
                "branch_state_digest": f"state-digest-{index}",
                "action_runs": _action_runs(
                    logged_action=logged_action,
                    oracle_action=oracle_action,
                    alive_delta=alive_delta,
                    birth_delta=birth_delta,
                ),
            }
        )
    return {
        "schema_version": MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
        "provenance": {"contract_digest": "synthetic-contract-digest"},
        "contract": {"ticks": 120},
        "aggregate": {
            "branch_point_count": len(branch_points),
            "oracle_changed_action_count": len(branch_points),
            "terminal_alive_gain_total_vs_logged": 1,
            "birth_gain_total_vs_logged": 1,
            "heuristic_action_source_count": 0,
            "replay_verified": True,
        },
        "acceptance": {"diagnostic_acceptance_passed": True, "blockers": []},
        "branch_points": branch_points,
        "branch_results": branch_results,
    }


def _action_mask() -> dict[str, bool]:
    return {action: action in ("stay", "eat", "drink") for action in ACTION_NAMES}


def _history_item() -> dict[str, object]:
    return {
        "tick": 10,
        "tick_delta": 2,
        "record_index": 90,
        "record_index_delta": 10,
        "requested_action": "eat",
        "resolved_action": "eat",
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": False,
        "x_delta": 0,
        "y_delta": 0,
        "energy_ratio_before": 0.25,
        "energy_ratio_after": 0.45,
        "energy_ratio_delta": 0.2,
        "hydration_ratio_before": 0.8,
        "hydration_ratio_after": 0.78,
        "hydration_ratio_delta": -0.02,
        "health_ratio_before": 0.9,
        "health_ratio_after": 0.9,
        "health_ratio_delta": 0.0,
        "resource_gain": 0.2,
        "drank": False,
        "ate": True,
        "died": False,
        "death_cause": None,
        "died_after_action": False,
        "post_carrion_contact": True,
        "ticks_since_animal_resource_gain": 0,
        "ticks_since_drink": 3,
    }


def _action_runs(
    *,
    logged_action: str,
    oracle_action: str,
    alive_delta: int,
    birth_delta: int,
) -> list[dict[str, object]]:
    base_alive = 2
    base_births = 1
    runs = []
    for action in ("drink", "eat", "stay"):
        alive = base_alive
        births = base_births
        if action == oracle_action:
            alive += alive_delta
            births += birth_delta
        if action == logged_action:
            alive = base_alive
            births = base_births
        runs.append(
            {
                "forced_action": action,
                "forced_action_used": True,
                "alive_agents": alive,
                "births": births,
                "deaths": max(0, 10 - alive),
                "target_alive_at_end": False,
                "dominant_requested_action_share": 0.25,
                "heuristic_action_source_count": 0,
                "replay_verification": {"verified": True},
            }
        )
    return copy.deepcopy(runs)


if __name__ == "__main__":
    unittest.main()
