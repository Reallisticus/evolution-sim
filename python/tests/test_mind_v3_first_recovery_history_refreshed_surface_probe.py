from __future__ import annotations

import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    REQUIRED_V133_RECOMMENDATION,
    build_first_recovery_history_refreshed_surface_probe,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_HISTORY_ROWS_OUTPUT_PATH as DEFAULT_V133_HISTORY_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V131_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryHistoryRefreshedSurfaceProbeTests(unittest.TestCase):
    def test_history_refreshed_surface_probe_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-history-refreshed-surface-probe"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_first_recovery_history_refreshed_surface_probe"
            ),
        )

    def test_synthetic_history_surface_can_be_ready(self) -> None:
        build = _build(_synthetic_payloads())
        report = build.report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "history_refreshed_surface_ready_for_shadow_proposal",
        )
        self.assertGreater(
            report["probe_comparison"]["heldout_accuracy"],
            report["probe_comparison"]["heldout_action_only_accuracy"],
        )
        self.assertLessEqual(
            report["metric_gate"]["fixture_open_dominant_predicted_action"][
                "share"
            ],
            0.5,
        )
        self.assertEqual(
            report["unsupported_action_audit"]["unsupported_action_count"],
            0,
        )
        self.assertEqual(report["leakage_audit"]["leakage_count"], 0)
        self.assertFalse(
            report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(report["authorization_block"]["training_executed"])

    def test_v131_source_integrity_failure_blocks_probe(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v131_report"]["source_integrity"]["passed"] = False

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v131_source_integrity_not_passed",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "history_refreshed_surface_source_integrity_failed",
        )

    def test_v133_not_ready_blocks_probe(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v133_report"]["recommendation"]["next_step"] = (
            "collect_missing_public_trajectory_history"
        )

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v133_not_ready_for_history_refreshed_surface",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "history_refreshed_surface_source_integrity_failed",
        )

    def test_v133_leakage_blocks_probe(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v133_report"]["leakage_audit"]["leakage_count"] = 1

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn("v133_leakage_nonzero", build.report["source_integrity"]["failures"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "history_refreshed_surface_source_integrity_failed",
        )

    def test_trainable_history_metadata_leakage_fails_closed(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v133_history_rows"][0]["history_features"]["source_path"] = (
            "not_trainable"
        )

        build = _build(payloads)

        self.assertFalse(build.report["leakage_audit"]["passed"])
        self.assertEqual(build.report["leakage_audit"]["leakage_count"], 1)
        self.assertEqual(
            build.report["classification"]["primary"],
            "history_refreshed_surface_source_integrity_failed",
        )

    def test_real_artifact_current_v134_facts_when_outputs_exist(self) -> None:
        required = [
            DEFAULT_V131_REPORT_PATH,
            DEFAULT_V131_FEATURE_ROWS_PATH,
            DEFAULT_V133_REPORT_PATH,
            DEFAULT_V133_HISTORY_ROWS_PATH,
            DEFAULT_V124_MANIFEST_PATH,
            DEFAULT_V115_ARCHIVE_ROWS_PATH,
            DEFAULT_V123_ARCHIVE_ROWS_PATH,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("local v131/v133/v124/v115/v123 outputs are not present")

        build = build_first_recovery_history_refreshed_surface_probe()
        report = build.report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(
            report["source_integrity"]["v133_recommendation"],
            REQUIRED_V133_RECOMMENDATION,
        )
        self.assertEqual(report["feature_surface_summary"]["candidate_feature_row_count"], 142)
        self.assertEqual(report["feature_surface_summary"]["candidate_branch_count"], 25)
        self.assertEqual(report["feature_surface_summary"]["history_feature_path_count"], 79)
        self.assertEqual(report["leakage_audit"]["leakage_count"], 0)
        self.assertEqual(
            report["classification"]["primary"],
            "history_refreshed_surface_blocked_by_signal",
        )
        self.assertEqual(
            report["metric_gate"]["failures"],
            ["heldout_signal_not_above_action_only_baseline"],
        )
        self.assertEqual(report["probe_comparison"]["heldout_accuracy"], 2 / 9)
        self.assertEqual(
            report["probe_comparison"]["heldout_action_only_accuracy"],
            2 / 9,
        )
        self.assertEqual(
            report["metric_gate"]["fixture_open_dominant_predicted_action"]["share"],
            0.5,
        )
        self.assertEqual(
            report["unsupported_action_audit"]["unsupported_action_count"],
            0,
        )
        self.assertFalse(
            report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(report["authorization_block"]["training_executed"])


def _build(payloads: dict[str, object]):
    return build_first_recovery_history_refreshed_surface_probe(
        v131_report=payloads["v131_report"],
        v131_report_path=None,
        v131_feature_rows=payloads["v131_feature_rows"],
        v131_feature_rows_path=None,
        v133_report=payloads["v133_report"],
        v133_report_path=None,
        v133_history_rows=payloads["v133_history_rows"],
        v133_history_rows_path=None,
        v124_manifest_rows=payloads["manifest_rows"],
        v124_manifest_path=None,
        v115_archive_rows=payloads["candidate_rows"],
        v115_archive_rows_path=None,
        v123_archive_rows=[],
        v123_archive_rows_path=None,
    )


def _synthetic_payloads() -> dict[str, object]:
    branch_specs = [
        ("b00_train_good", "train", "open_mind_v3", "move_north", "h_good"),
        ("b01_train_bad", "train", "open_mind_v3", "eat", "h_bad"),
        ("b02_test_good", "test", "open_mind_v3", "move_north", "h_good"),
        ("b03_test_bad", "test", "open_mind_v3", "eat", "h_bad"),
    ]
    branch_ordinals = {
        spec[0]: index for index, spec in enumerate(sorted(branch_specs))
    }
    manifest_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    v131_rows: list[dict[str, object]] = []
    v133_rows: list[dict[str, object]] = []
    for branch_id, split, fixture, positive_action, history_bucket in sorted(branch_specs):
        branch_ordinal = branch_ordinals[branch_id]
        candidates = [
            ("eat", positive_action == "eat", "bad" if history_bucket == "h_good" else "good"),
            ("move_north", positive_action == "move_north", "good" if history_bucket == "h_good" else "bad"),
        ]
        sorted_candidates = sorted(
            enumerate(candidates),
            key=lambda item: (
                item[1][0],
                _archive_row_id(branch_id, item[0], item[1][1]),
            ),
        )
        repaired_archive_id = next(
            _archive_row_id(branch_id, original_index, positive)
            for original_index, (_, positive, _) in sorted_candidates
            if positive
        )
        manifest_rows.append(
            {
                "branch_id": branch_id,
                "repaired_action": positive_action,
                "repaired_archive_row_id": repaired_archive_id,
                "non_trainable_audit_metadata": {
                    "seed": 29,
                    "source_kind": fixture,
                },
            }
        )
        v133_rows.append(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_ROW_SCHEMA_VERSION
                ),
                "candidate_set_ordinal": branch_ordinal,
                "public_history_available": True,
                "history_features": {
                    "rollout_context.post_carrion_contact": 1.0,
                    "rollout_context.recent_resolved_count:eat": (
                        1.0 if history_bucket == "h_bad" else 0.0
                    ),
                    "rollout_context.recent_resolved_count:move_north": (
                        1.0 if history_bucket == "h_good" else 0.0
                    ),
                },
            }
        )
        for candidate_ordinal, (original_index, candidate) in enumerate(sorted_candidates):
            action, positive, target_bucket = candidate
            archive_id = _archive_row_id(branch_id, original_index, positive)
            candidate_rows.append(
                {
                    "archive_row_id": archive_id,
                    "candidate_action": action,
                    "material_gain_label": positive,
                    "provenance": {"branch_id": branch_id},
                    "trainable_public_input": {
                        "action_mask": {
                            "eat": True,
                            "move_north": True,
                            "stay": True,
                        }
                    },
                }
            )
            v131_rows.append(
                {
                    "schema_version": (
                        MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
                    ),
                    "candidate_public_features": {
                        "candidate_action": action,
                        "candidate_action_family": (
                            "movement" if action.startswith("move_") else "resource"
                        ),
                        "candidate_action_observation_legal": True,
                        "candidate_target_context.history_match_bucket": target_bucket,
                    },
                    "extractor_source_paths": [
                        "v129_feature_row.candidate_public_features.candidate_action",
                        "v130_context_row.candidate_observation_context.candidate_target_context.history_match_bucket",
                    ],
                    "non_feature_metadata": {
                        "candidate_set_ordinal": branch_ordinal,
                        "candidate_ordinal_within_set": candidate_ordinal,
                        "split": split,
                    },
                }
            )
    return {
        "v131_report": _v131_report(v131_rows),
        "v131_feature_rows": v131_rows,
        "v133_report": _v133_report(v133_rows),
        "v133_history_rows": v133_rows,
        "manifest_rows": manifest_rows,
        "candidate_rows": candidate_rows,
    }


def _v131_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
        "source_integrity": {"passed": True, "failures": []},
        "feature_surface_summary": {"candidate_feature_row_count": len(rows)},
    }


def _v133_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_PUBLIC_ROLLOUT_HISTORY_CONTEXT_AUDIT_SCHEMA_VERSION
        ),
        "source_integrity": {"passed": True, "failures": []},
        "recommendation": {"next_step": REQUIRED_V133_RECOMMENDATION},
        "leakage_audit": {"leakage_count": 0},
        "history_rows_summary": {"row_count": len(rows)},
    }


def _archive_row_id(branch_id: str, original_index: int, positive: bool) -> str:
    suffix = "1_positive" if positive else "0_negative"
    return f"{branch_id}_{original_index}_{suffix}"
