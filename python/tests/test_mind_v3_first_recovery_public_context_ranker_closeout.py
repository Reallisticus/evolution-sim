from __future__ import annotations

import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V134_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_context_ranker_closeout import (
    CLASSIFICATION,
    FINAL_RECOMMENDATION,
    build_first_recovery_public_context_ranker_closeout,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V132_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryPublicContextRankerCloseoutTests(unittest.TestCase):
    def test_public_context_ranker_closeout_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-public-context-ranker-closeout"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_first_recovery_public_context_ranker_closeout"
            ),
        )

    def test_synthetic_closeout_emits_terminal_decision(self) -> None:
        build = _build(_synthetic_reports())
        report = build.report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(report["classification"]["primary"], CLASSIFICATION)
        self.assertEqual(report["recommendation"]["next_step"], FINAL_RECOMMENDATION)
        self.assertTrue(report["path_closeout_decision"]["closed"])
        self.assertFalse(report["path_closeout_decision"]["promotable"])
        self.assertFalse(
            report["recommendation"]["first_recovery_public_context_ranker_path_promotable"]
        )
        self.assertTrue(report["recommendation"]["no_v136_first_recovery_feature_probe"])
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(report["authorization_block"]["shadow_scorer_execution_authorized"])
        self.assertFalse(report["authorization_block"]["downstream_shadow_scorer_allowed"])
        self.assertFalse(report["authorization_block"]["v113_readiness_rerun_allowed"])
        self.assertFalse(report["authorization_block"]["runtime_policy_change_recommended"])
        self.assertFalse(report["authorization_block"]["gate_change_authorized"])
        self.assertFalse(report["authorization_block"]["viewer_change_authorized"])
        self.assertFalse(report["authorization_block"]["replay_golden_change_authorized"])
        self.assertFalse(report["authorization_block"]["foundation_change_authorized"])
        self.assertEqual(
            [step["version"] for step in report["decision_chain"]],
            ["v131", "v132", "v133", "v134"],
        )

    def test_v132_source_integrity_failure_is_recorded(self) -> None:
        payloads = _synthetic_reports()
        payloads["v132_report"]["source_integrity"]["passed"] = False

        report = _build(payloads).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v132_source_integrity_not_passed",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(report["classification"]["primary"], CLASSIFICATION)
        self.assertFalse(report["authorization_block"]["training_authorized"])

    def test_v133_recommendation_contract_mismatch_is_recorded(self) -> None:
        payloads = _synthetic_reports()
        payloads["v133_report"]["recommendation"]["next_step"] = (
            "collect_missing_public_trajectory_history"
        )

        report = _build(payloads).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v133_recommendation_contract_mismatch",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(report["recommendation"]["next_step"], FINAL_RECOMMENDATION)

    def test_v134_not_blocked_by_signal_is_recorded(self) -> None:
        payloads = _synthetic_reports()
        payloads["v134_report"]["classification"]["primary"] = (
            "history_refreshed_surface_ready_for_shadow_proposal"
        )

        report = _build(payloads).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn("v134_not_blocked_by_signal", report["source_integrity"]["failures"])
        self.assertFalse(
            report["authorization_block"]["shadow_scorer_execution_authorized"]
        )

    def test_v134_leakage_nonzero_is_recorded(self) -> None:
        payloads = _synthetic_reports()
        payloads["v134_report"]["leakage_audit"]["leakage_count"] = 1

        report = _build(payloads).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn("v134_leakage_nonzero", report["source_integrity"]["failures"])
        self.assertFalse(report["authorization_block"]["runtime_policy_change_authorized"])

    def test_real_artifact_current_v135_closeout_facts_when_outputs_exist(self) -> None:
        required = [
            DEFAULT_V131_REPORT_PATH,
            DEFAULT_V132_REPORT_PATH,
            DEFAULT_V133_REPORT_PATH,
            DEFAULT_V134_REPORT_PATH,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("local v131-v134 reports are not present")

        report = build_first_recovery_public_context_ranker_closeout().report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(report["classification"]["primary"], CLASSIFICATION)
        self.assertEqual(report["recommendation"]["next_step"], FINAL_RECOMMENDATION)
        self.assertEqual(
            report["terminal_signal_summary"]["v134_classification"],
            "history_refreshed_surface_blocked_by_signal",
        )
        self.assertEqual(report["terminal_signal_summary"]["heldout_accuracy"], 2 / 9)
        self.assertEqual(
            report["terminal_signal_summary"]["heldout_action_only_accuracy"],
            2 / 9,
        )
        self.assertEqual(
            report["terminal_signal_summary"]["heldout_delta_vs_action_only"],
            0.0,
        )
        self.assertEqual(
            report["terminal_signal_summary"][
                "fixture_open_dominant_predicted_action"
            ],
            {"action": "move_north", "count": 5, "share": 0.5, "total": 10},
        )
        self.assertFalse(report["terminal_signal_summary"]["seed29_passed"])
        self.assertEqual(report["terminal_signal_summary"]["seed29_accuracy"], 0.0)
        self.assertEqual(
            report["terminal_signal_summary"]["overall_dominant_predicted_action"],
            {"action": "move_north", "count": 14, "share": 0.56, "total": 25},
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(report["authorization_block"]["downstream_shadow_scorer_allowed"])
        self.assertFalse(report["authorization_block"]["v113_readiness_rerun_allowed"])
        self.assertFalse(report["authorization_block"]["runtime_policy_change_recommended"])


def _build(payloads: dict[str, dict[str, object]]):
    return build_first_recovery_public_context_ranker_closeout(
        v131_report=payloads["v131_report"],
        v131_report_path=None,
        v132_report=payloads["v132_report"],
        v132_report_path=None,
        v133_report=payloads["v133_report"],
        v133_report_path=None,
        v134_report=payloads["v134_report"],
        v134_report_path=None,
    )


def _synthetic_reports() -> dict[str, dict[str, object]]:
    return {
        "v131_report": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
            ),
            "source_integrity": {"passed": True},
            "classification": {
                "primary": "refreshed_candidate_public_feature_surface_blocked_by_signal",
            },
            "metric_gate": {
                "failures": [
                    "fixture_open_failed",
                    "heldout_signal_not_above_action_only_baseline",
                ],
            },
            "probe_comparison": {
                "heldout_accuracy": 0.5,
                "heldout_action_only_accuracy": 0.5,
                "heldout_action_order_accuracy": 7 / 18,
            },
            "fixture_open_evaluation": {
                "groups": {
                    "open_mind_v3": {
                        "dominant_predicted_action": {
                            "action": "move_north",
                            "count": 6,
                            "share": 0.6,
                            "total": 10,
                        },
                    },
                },
            },
        },
        "v132_report": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION
            ),
            "source_integrity": {"passed": True},
            "classification": {
                "primary": "refreshed_surface_blocker_slice_audit_completed",
            },
            "recommendation": {"next_step": "add_public_rollout_history_context"},
            "blocker_class_aggregate": {
                "counts_by_failed_blocker_class": {
                    "fixture_open_action_collapse": 6,
                    "train_support_hole": 2,
                    "action_prior_tie_or_fallback": 2,
                    "insufficient_public_context_signal": 2,
                    "material_candidate_not_selected": 1,
                },
            },
        },
        "v133_report": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION
            ),
            "source_integrity": {"passed": True},
            "classification": {
                "primary": "public_rollout_history_context_ready_for_refreshed_surface",
            },
            "recommendation": {
                "next_step": "public_rollout_history_context_ready_for_refreshed_surface",
            },
            "leakage_audit": {"leakage_count": 0},
            "public_history_availability": {
                "history_row_count": 25,
                "history_available_row_count": 25,
                "missing_public_history_field_count": 0,
            },
            "open_fixture_collapse_analysis": {
                "fixture_open_collapse_row_count": 6,
                "collapse_history_available_row_count": 6,
                "collapse_variance": {"varying_feature_path_count": 27},
            },
            "heldout_failure_analysis": {
                "failed_heldout_variance": {"varying_feature_path_count": 35},
            },
        },
        "v134_report": {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION
            ),
            "source_integrity": {"passed": True},
            "classification": {
                "primary": "history_refreshed_surface_blocked_by_signal",
            },
            "leakage_audit": {"leakage_count": 0},
            "metric_gate": {"failures": ["heldout_signal_not_above_action_only_baseline"]},
            "probe_comparison": {
                "heldout_accuracy": 2 / 9,
                "heldout_action_only_accuracy": 2 / 9,
                "heldout_delta_vs_action_only": 0.0,
                "heldout_action_order_accuracy": 7 / 18,
                "heldout_delta_vs_action_order": -1 / 6,
            },
            "fixture_open_evaluation": {
                "groups": {
                    "open_mind_v3": {
                        "dominant_predicted_action": {
                            "action": "move_north",
                            "count": 5,
                            "share": 0.5,
                            "total": 10,
                        },
                    },
                },
            },
            "seed29_evaluation": {
                "passed": False,
                "metrics": {"accuracy": 0.0},
            },
            "action_distribution": {
                "dominant_predicted_action": {
                    "action": "move_north",
                    "count": 14,
                    "share": 0.56,
                    "total": 25,
                },
            },
            "unsupported_action_audit": {
                "unsupported_action_count": 0,
                "unsupported_action_rate": 0.0,
            },
            "material_exact_match_recall": {
                "repaired_label_material_gain_exact_match_recall": 0.47058823529411764,
            },
        },
    }
