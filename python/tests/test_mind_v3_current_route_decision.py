from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.mind.current_route_decision import (
    CLOSED_FAMILIES,
    CURRENT_CLOSED_PATH,
    DEFAULT_OUTPUT_PATH,
    LATEST_EVIDENCE_VERSION,
    NEXT_ALLOWED_RESEARCH_DIRECTION,
    NEXT_DISALLOWED_ACTIONS,
    build_mind_v3_current_route_decision_report,
    write_mind_v3_current_route_decision_report,
)
from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V134_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_HISTORY_REFRESHED_SURFACE_PROBE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_context_ranker_closeout import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V135_REPORT_PATH,
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


class MindV3CurrentRouteDecisionTests(unittest.TestCase):
    def test_current_route_decision_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:current-route-decision"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_current_route_decision"
            ),
        )

    def test_synthetic_current_route_decision_closes_current_path(self) -> None:
        report = _build_current_route(_synthetic_report_bundle()).report

        self.assertEqual(report["latest_evidence_version"], LATEST_EVIDENCE_VERSION)
        self.assertEqual(report["current_closed_path"], CURRENT_CLOSED_PATH)
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(report["closed_families"], CLOSED_FAMILIES)
        self.assertEqual(
            report["next_allowed_research_direction"],
            NEXT_ALLOWED_RESEARCH_DIRECTION,
        )
        self.assertEqual(report["next_disallowed_actions"], NEXT_DISALLOWED_ACTIONS)
        self.assertTrue(report["decision"]["closed"])
        self.assertFalse(report["decision"]["promotable"])
        self.assertFalse(
            report["decision"]["more_first_recovery_ranker_or_feature_probes_authorized"]
        )
        self.assertFalse(
            report["decision"]["first_recovery_public_context_ranker_path_open"]
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )
        self.assertFalse(
            report["authorization_block"]["shadow_scorer_execution_authorized"]
        )
        self.assertFalse(
            report["authorization_block"]["v136_first_recovery_feature_probe_authorized"]
        )
        self.assertFalse(report["authorization_block"]["runtime_promotion_authorized"])

    def test_synthetic_current_route_decision_carries_required_v135_metrics(self) -> None:
        metrics = _build_current_route(_synthetic_report_bundle()).report[
            "v135_source_integrity_and_terminal_metrics"
        ]

        self.assertTrue(metrics["source_integrity_passed"])
        self.assertTrue(metrics["matches_expected"])
        self.assertEqual(metrics["heldout_accuracy"], 0.2222222222222222)
        self.assertEqual(metrics["heldout_action_only_accuracy"], 0.2222222222222222)
        self.assertEqual(metrics["heldout_action_order_accuracy"], 0.3888888888888889)
        self.assertEqual(metrics["seed29_accuracy"], 0.0)
        self.assertEqual(metrics["dominant_predicted_action_name"], "move_north")
        self.assertEqual(metrics["dominant_predicted_action_share"], 0.56)
        self.assertEqual(metrics["leakage_count"], 0)
        self.assertEqual(metrics["unsupported_action_count"], 0)

    def test_v135_terminal_metric_drift_is_recorded_without_authorizing_training(
        self,
    ) -> None:
        bundle = _synthetic_report_bundle()
        terminal = bundle["v135_report"]["terminal_signal_summary"]
        terminal["heldout_accuracy"] = 0.5

        report = _build_current_route(bundle).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v135_terminal_metrics_mismatch",
            report["source_integrity"]["failures"],
        )
        self.assertFalse(
            report["v135_source_integrity_and_terminal_metrics"]["matches_expected"]
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )

    def test_v135_v136_feature_probe_authorization_drift_is_recorded(self) -> None:
        bundle = _synthetic_report_bundle()
        bundle["v135_report"]["recommendation"][
            "no_v136_first_recovery_feature_probe"
        ] = False

        report = _build_current_route(bundle).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v135_does_not_block_v136_feature_probe",
            report["source_integrity"]["failures"],
        )
        self.assertFalse(
            report["authorization_block"][
                "v136_first_recovery_feature_probe_authorized"
            ]
        )

    def test_write_current_route_decision_report(self) -> None:
        build = _build_current_route(_synthetic_report_bundle())
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "decision.json"
            write_mind_v3_current_route_decision_report(build, output_path=output)
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(written["current_closed_path"], CURRENT_CLOSED_PATH)
        self.assertEqual(
            written["next_allowed_research_direction"],
            NEXT_ALLOWED_RESEARCH_DIRECTION,
        )

    def test_real_artifact_current_route_facts_when_outputs_exist(self) -> None:
        required = [
            DEFAULT_V131_REPORT_PATH,
            DEFAULT_V132_REPORT_PATH,
            DEFAULT_V133_REPORT_PATH,
            DEFAULT_V134_REPORT_PATH,
            DEFAULT_V135_REPORT_PATH,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("local v131-v135 reports are not present")

        report = build_mind_v3_current_route_decision_report().report
        metrics = report["v135_source_integrity_and_terminal_metrics"]

        self.assertEqual(
            DEFAULT_OUTPUT_PATH.name,
            "mind-v3-v136-current-route-decision.json",
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(report["latest_evidence_version"], 135)
        self.assertEqual(
            report["current_closed_path"],
            "first_recovery_public_context_ranker_path_closed_not_promotable",
        )
        self.assertTrue(metrics["matches_expected"])
        self.assertEqual(metrics["heldout_accuracy"], 0.2222222222222222)
        self.assertEqual(metrics["heldout_action_only_accuracy"], 0.2222222222222222)
        self.assertEqual(metrics["heldout_action_order_accuracy"], 0.3888888888888889)
        self.assertEqual(metrics["seed29_accuracy"], 0.0)
        self.assertEqual(
            metrics["dominant_predicted_action"],
            {"action": "move_north", "count": 14, "share": 0.56, "total": 25},
        )
        self.assertEqual(metrics["leakage_count"], 0)
        self.assertEqual(metrics["unsupported_action_count"], 0)
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )
        self.assertFalse(
            report["authorization_block"]["shadow_scorer_execution_authorized"]
        )


def _build_current_route(payloads: dict[str, dict[str, object]]):
    return build_mind_v3_current_route_decision_report(
        v131_report=payloads["v131_report"],
        v131_report_path=None,
        v132_report=payloads["v132_report"],
        v132_report_path=None,
        v133_report=payloads["v133_report"],
        v133_report_path=None,
        v134_report=payloads["v134_report"],
        v134_report_path=None,
        v135_report=payloads["v135_report"],
        v135_report_path=None,
    )


def _synthetic_report_bundle() -> dict[str, dict[str, object]]:
    payloads = _synthetic_v131_to_v134_reports()
    v135 = build_first_recovery_public_context_ranker_closeout(
        v131_report=payloads["v131_report"],
        v131_report_path=None,
        v132_report=payloads["v132_report"],
        v132_report_path=None,
        v133_report=payloads["v133_report"],
        v133_report_path=None,
        v134_report=payloads["v134_report"],
        v134_report_path=None,
    ).report
    return payloads | {"v135_report": v135}


def _synthetic_v131_to_v134_reports() -> dict[str, dict[str, object]]:
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
            "metric_gate": {
                "failures": ["heldout_signal_not_above_action_only_baseline"]
            },
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
