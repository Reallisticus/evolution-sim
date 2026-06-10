from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_sim.cli import mind_v3_carrion_archive_override_autopsy as cli
from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_OUTPUT_PATH,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
    build_carrion_archive_override_autopsy_report,
    trainable_input_leakage_scan,
    write_carrion_archive_override_autopsy_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionArchiveOverrideAutopsyTests(unittest.TestCase):
    def test_report_joins_overrides_to_support_rows_and_branch_evidence(self) -> None:
        archive_report, rows, train_eval, artifact, rerun_cases = _autopsy_inputs()

        report = build_carrion_archive_override_autopsy_report(
            archive_report=archive_report,
            dataset_rows=rows,
            train_eval_report=train_eval,
            artifact=artifact,
            rerun_cases=rerun_cases,
            target_seeds=(13, 41),
        )

        self.assertEqual(
            report["schema_version"],
            M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertEqual(
            report["classification"]["primary"],
            "m3_carrion_archive_override_autopsy_complete_no_training",
        )
        self.assertEqual(
            report["aggregates"]["observed_live_carrion_override_count"],
            2,
        )
        self.assertTrue(report["aggregates"]["all_live_overrides_directly_legal"])
        self.assertEqual(report["aggregates"]["direct_invalid_override_count"], 0)
        modes = report["failure_mode_classification"]["modes"]
        self.assertTrue(modes["stale_one_step_aliasing"]["present"])
        self.assertTrue(modes["movement_near_carrion_side_effects"]["present"])
        self.assertTrue(modes["missing_hydration_reproduction_context"]["present"])
        self.assertTrue(modes["resolution_conflict"]["present"])
        self.assertEqual(
            report["failure_mode_classification"]["primary"],
            "missing_hydration_reproduction_context",
        )
        by_action = {
            row["label_action"]: row
            for row in report["support_label_action_failure_map"]
        }
        self.assertEqual(
            by_action["move_north"]["birth_regression_override_count"],
            1,
        )
        self.assertEqual(
            by_action["move_north"]["resolved_invalid_increase_override_count"],
            1,
        )
        self.assertEqual(
            by_action["move_east"]["birth_regression_override_count"],
            1,
        )
        self.assertEqual(
            by_action["move_north"]["source_branch_reason"],
            "carrion_contact",
        )
        self.assertEqual(
            by_action["move_east"]["source_branch_reason"],
            "movement_stall",
        )
        self.assertTrue(report["inputs"]["input_validation"]["passed"])
        self.assertEqual(len(str(report["exact_digest"])), 64)

    def test_rejects_promotional_train_eval_report(self) -> None:
        archive_report, rows, train_eval, artifact, rerun_cases = _autopsy_inputs()
        train_eval["promotion_authorized"] = True

        with self.assertRaisesRegex(ValueError, "train_eval_promotion_authorized"):
            build_carrion_archive_override_autopsy_report(
                archive_report=archive_report,
                dataset_rows=rows,
                train_eval_report=train_eval,
                artifact=artifact,
                rerun_cases=rerun_cases,
                target_seeds=(13, 41),
            )

    def test_rejects_archive_source_integrity_failure(self) -> None:
        archive_report, rows, train_eval, artifact, rerun_cases = _autopsy_inputs()
        archive_report["source_integrity"] = {
            "passed": False,
            "failures": ["test_failure"],
        }

        with self.assertRaisesRegex(
            ValueError,
            "archive_report_source_integrity_not_passed",
        ):
            build_carrion_archive_override_autopsy_report(
                archive_report=archive_report,
                dataset_rows=rows,
                train_eval_report=train_eval,
                artifact=artifact,
                rerun_cases=rerun_cases,
                target_seeds=(13, 41),
            )

    def test_rejects_archive_dataset_digest_mismatch(self) -> None:
        archive_report, rows, train_eval, artifact, rerun_cases = _autopsy_inputs()
        archive_report["dataset"] = {
            **archive_report["dataset"],
            "dataset_digest": "wrong-digest",
        }

        with self.assertRaisesRegex(
            ValueError,
            "archive_report_dataset_digest_mismatch",
        ):
            build_carrion_archive_override_autopsy_report(
                archive_report=archive_report,
                dataset_rows=rows,
                train_eval_report=train_eval,
                artifact=artifact,
                rerun_cases=rerun_cases,
                target_seeds=(13, 41),
            )

    def test_trainable_leakage_scan_rejects_fixture_private_fields(self) -> None:
        leaking_rows = [
            {
                "trainable": {
                    "features": {
                        "fixture": "carrion_only",
                        "private_world_state": {"x": 1},
                    },
                    "label": {"action": "move_north"},
                }
            }
        ]

        scan = trainable_input_leakage_scan(leaking_rows)

        self.assertFalse(scan["passed"])
        reasons = {failure["reason"] for failure in scan["failures"]}
        self.assertIn("forbidden_trainable_path_token", reasons)

    def test_cli_and_npm_entrypoint_exist(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertIn(
            "evolution_sim.cli.mind_v3_carrion_archive_override_autopsy",
            package["scripts"]["sim:mind:v3:carrion-archive-override-autopsy"],
        )
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.output, DEFAULT_OUTPUT_PATH)

    def test_report_writer_outputs_json_object(self) -> None:
        archive_report, rows, train_eval, artifact, rerun_cases = _autopsy_inputs()
        report = build_carrion_archive_override_autopsy_report(
            archive_report=archive_report,
            dataset_rows=rows,
            train_eval_report=train_eval,
            artifact=artifact,
            rerun_cases=rerun_cases,
            target_seeds=(13, 41),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "report.json"
            write_carrion_archive_override_autopsy_report(report, output)
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(written["exact_digest"], report["exact_digest"])


def _autopsy_inputs() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    rows = [_dataset_row(0, "move_north", "branch-carrion"), _dataset_row(1, "move_east", "branch-stall")]
    artifact = _artifact()
    dataset_digest = stable_payload_digest(rows)
    branch_digest = "branch-digest"
    archive_report = {
        "schema_version": "m3_carrion_specific_archive_expansion_report_v1",
        "policy": "diagnostics_only_m3_carrion_specific_archive_expansion_v1",
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "branch_evidence_digest": branch_digest,
        "dataset": {"dataset_digest": dataset_digest, "safe_label_count": len(rows)},
        "source_integrity": {"passed": True, "failures": []},
        "generation_status": {"state": "complete", "partial": False},
        "branch_results": [
            _branch_result(
                branch_id="branch-carrion",
                reason="carrion_contact",
                action="move_north",
                births_delta=0,
                resolved_invalid_delta=0,
                resolution_valid=True,
            ),
            _branch_result(
                branch_id="branch-stall",
                reason="movement_stall",
                action="move_east",
                births_delta=2,
                resolved_invalid_delta=-1,
                resolution_valid=True,
            ),
        ],
    }
    archive_report["branch_result_count"] = len(archive_report["branch_results"])
    archive_report["branch_evidence_digest"] = stable_payload_digest(
        archive_report["branch_results"]
    )
    branch_digest = str(archive_report["branch_evidence_digest"])
    train_eval = {
        "schema_version": "m3_carrion_specific_archive_train_eval_report_v1",
        "policy": (
            "diagnostics_only_m3_carrion_specific_archive_"
            "support_gated_train_eval_v1"
        ),
        "classification": {
            "primary": "m3_safe_archive_diagnostic_failed_non_promotional"
        },
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "inputs": {
            "expected_dataset_digest": dataset_digest,
            "expected_branch_evidence_digest": branch_digest,
        },
        "validation": {
            "passed": True,
            "failures": [],
            "dataset_digest": dataset_digest,
            "branch_evidence_digest": branch_digest,
        },
        "artifact": {
            "digest": stable_payload_digest(artifact),
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
        },
        "acceptance": {
            "passed": False,
            "blockers": [
                {
                    "fixture": "carrion_only",
                    "seed": 13,
                    "reason": "carrion_seed_birth_regression",
                    "observed": -1,
                },
                {
                    "fixture": "carrion_only",
                    "seed": 13,
                    "reason": "carrion_seed_resolved_invalid_increase",
                    "observed": 1,
                },
                {
                    "fixture": "carrion_only",
                    "seed": 41,
                    "reason": "carrion_seed_birth_regression",
                    "observed": -1,
                },
            ],
            "metrics": {
                "carrion_applied_override_count": 2,
                "dominant_requested_action_share": 0.42,
                "heuristic_action_source_count": 0,
                "carrion_per_seed_delta": [
                    {
                        "seed": 13,
                        "alive_delta": 0,
                        "births_delta": -1,
                        "resolved_invalid_action_count_delta": 1,
                        "applied_override_count": 1,
                        "applied_override_action_counts": {"move_north": 1},
                    },
                    {
                        "seed": 41,
                        "alive_delta": 0,
                        "births_delta": -1,
                        "resolved_invalid_action_count_delta": -1,
                        "applied_override_count": 1,
                        "applied_override_action_counts": {"move_east": 1},
                    },
                ],
            },
        },
    }
    rerun_cases = [
        _rerun_case(seed=13, support_index=0, action="move_north", invalid_delta=1),
        _rerun_case(seed=41, support_index=1, action="move_east", invalid_delta=-1),
    ]
    return archive_report, rows, train_eval, artifact, rerun_cases


def _artifact() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_v144_branch_intervention_residual_artifact_v1",
        "policy": "mind_v3_v144_branch_intervention_residual_runtime_v1",
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "default_runtime_behavior_changed": False,
        "training_policy": "test",
        "action_prior_penalty_scale": 2.0,
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "support_gate": {
            "nearest_support_distance_threshold": 0.0,
            "residual_score_margin_threshold": 0.0,
        },
        "inference_contract": {
            "one_row_one_agent_local_decision": True,
            "requires_action_mask": True,
            "requires_policy_visible_features_only": True,
            "requires_linear_default_action": True,
            "requires_planner_outcome_tables": False,
            "requires_global_batch_assignment": False,
            "uses_heuristic_fallback": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_fixture_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_private_simulator_state": False,
        },
        "training_row_count": 2,
        "support_action_counts": {"move_east": 1, "move_north": 1},
        "teacher_action_counts": {"move_east": 1, "move_north": 1},
        "support_examples": [
            {
                "example_index": 0,
                "action": "move_north",
                "mode": "test",
                "weight": 1.0,
                "feature_vector": [0.0, 1.0],
            },
            {
                "example_index": 1,
                "action": "move_east",
                "mode": "test",
                "weight": 1.0,
                "feature_vector": [1.0, 0.0],
            },
        ],
    }


def _dataset_row(index: int, action: str, branch_id: str) -> dict[str, object]:
    return {
        "schema_version": "m3_carrion_broad_safe_archive_expansion_dataset_row_v1",
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {
                "observation_input": {"values": [0.1, 0.2]},
                "action_mask": {action: True},
            },
            "label": {"action": action, "label_policy": "test"},
        },
        "metadata": {
            "row_index": int(index),
            "fixture": "carrion_only",
            "seed": 29 if action == "move_north" else 41,
            "branch_id": branch_id,
            "branch_tick": 0,
            "agent_id": 9 + index,
        },
    }


def _branch_result(
    *,
    branch_id: str,
    reason: str,
    action: str,
    births_delta: int,
    resolved_invalid_delta: int,
    resolution_valid: bool,
) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "seed": 29,
        "fixture": "carrion_only",
        "branch_tick": 0,
        "branch_index": 0,
        "candidate_actions": [action],
        "public_features": {"action_mask": {action: True}},
        "carrion_archive_context": {
            "branch_reason": reason,
            "reason_rank": 0,
            "reason_evidence": {
                "branch_reason": reason,
                "requested_action": "eat" if reason == "carrion_contact" else action,
                "resolved_action": "eat" if reason == "carrion_contact" else "stay",
                "food_source": "carcass" if reason == "carrion_contact" else None,
                "hydration_ratio_before": 0.9,
                "hydration_ratio_after": 0.88,
            },
        },
        "action_runs": [
            {
                "forced_action": action,
                "replay_verification": {"verified": True},
                "deltas_vs_baseline": {
                    "births": births_delta,
                    "alive_agents": 0,
                    "unsupported_resolved_action_count": resolved_invalid_delta,
                },
                "deltas_vs_v142_override": {
                    "births": births_delta,
                    "alive_agents": 0,
                    "unsupported_resolved_action_count": resolved_invalid_delta,
                },
                "target_terminal": {"alive": False},
                "first_action_outcome": {
                    "requested_action": action,
                    "resolved_action": action if resolution_valid else "stay",
                    "action_valid": True,
                    "resolution_action_valid": bool(resolution_valid),
                    "outcome": {
                        "requested_action": action,
                        "resolved_action": action if resolution_valid else "stay",
                        "observation_action_valid": True,
                        "resolution_action_valid": bool(resolution_valid),
                        "invalid_reason": None
                        if resolution_valid
                        else "not_in_resolution_action_mask",
                        "reproduced": False,
                        "reproduction_ready_after": False,
                        "movement": {"moved": bool(resolution_valid)},
                        "feeding": {"ate": False},
                    },
                },
            }
        ],
    }


def _rerun_case(
    *,
    seed: int,
    support_index: int,
    action: str,
    invalid_delta: int,
) -> dict[str, object]:
    return {
        "fixture": "carrion_only",
        "seed": int(seed),
        "ticks": 120,
        "seed_delta": {
            "alive_delta": 0,
            "births_delta": -1,
            "resolved_invalid_action_count_delta": int(invalid_delta),
        },
        "override_traces": [
            {
                "fixture": "carrion_only",
                "seed": int(seed),
                "tick": 0,
                "agent_id": 9,
                "record_index": 0,
                "linear_action": "eat",
                "selected_support_action": action,
                "final_requested_action": action,
                "final_resolved_action": action,
                "override_applied": True,
                "nearest_support_distance": 0.0,
                "score_margin": 0.1,
                "support_example_index": int(support_index),
                "action_mask_legality": {
                    "selected_support_action_legal": True,
                    "selected_support_action_resolution_legal": True,
                    "final_requested_action_valid": True,
                    "final_resolved_action_valid": True,
                    "invalid_reason": None,
                },
                "live_public_outcome_summary": {
                    "requested_action": action,
                    "resolved_action": action,
                    "resolution_action_valid": True,
                    "reproduced": False,
                    "reproduction_ready_after": False,
                },
                "resolved_invalid_contribution": 0,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
