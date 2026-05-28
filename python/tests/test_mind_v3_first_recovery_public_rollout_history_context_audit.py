from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    REQUIRED_V132_RECOMMENDATION,
    build_first_recovery_public_rollout_history_context_audit,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_DETAIL_ROWS_OUTPUT_PATH as DEFAULT_V132_DETAIL_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V132_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryPublicRolloutHistoryContextAuditTests(unittest.TestCase):
    def test_public_rollout_history_context_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-public-rollout-history-context-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_first_recovery_public_rollout_history_context_audit"
            ),
        )

    def test_synthetic_public_history_context_can_be_ready(self) -> None:
        payloads = _synthetic_payloads()

        build = _build(payloads)
        report = build.report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["public_history_availability"]["complete"])
        self.assertEqual(
            report["classification"]["primary"],
            "public_rollout_history_context_ready_for_refreshed_surface",
        )
        self.assertEqual(
            report["recommendation"]["next_step"],
            "public_rollout_history_context_ready_for_refreshed_surface",
        )
        self.assertGreater(
            report["within_branch_history_variance"]["fixture_open_action_collapse_rows"][
                "varying_feature_path_count"
            ],
            0,
        )
        self.assertGreater(
            report["open_fixture_collapse_analysis"][
                "collapse_vs_nonfailed_open_contrast"
            ]["contrasting_feature_path_count"],
            0,
        )
        self.assertFalse(
            report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(report["authorization_block"]["claim_causality"])

    def test_v132_source_integrity_failure_blocks_v133(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v132_report"]["source_integrity"]["passed"] = False

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v132_source_integrity_not_passed",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "public_rollout_history_context_source_integrity_failed",
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "history_context_leakage_or_integrity_failed",
        )

    def test_non_rollout_history_v132_recommendation_blocks_v133(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v132_report"]["recommendation"]["next_step"] = (
            "collect_targeted_public_context_rows"
        )

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v132_recommendation_not_public_rollout_history_context",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "public_rollout_history_context_source_integrity_failed",
        )

    def test_missing_public_trajectory_history_fails_closed(self) -> None:
        payloads = _synthetic_payloads()
        payloads["trajectory_records_by_path"] = {}

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertFalse(build.report["public_history_availability"]["complete"])
        self.assertGreater(
            build.report["public_history_availability"][
                "missing_public_history_field_count"
            ],
            0,
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "public_rollout_history_context_public_inputs_missing",
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "collect_missing_public_trajectory_history",
        )

    def test_trainable_history_feature_leakage_fails_closed(self) -> None:
        payloads = _synthetic_payloads()
        patch_target = (
            "evolution_sim.mind."
            "first_recovery_public_rollout_history_context_audit."
            "_history_features_for_record"
        )
        with mock.patch(
            patch_target,
            return_value=({"source_path": "not_trainable"}, ["current_record.action_mask"]),
        ):
            build = _build(payloads)

        self.assertFalse(build.report["leakage_audit"]["passed"])
        self.assertEqual(build.report["leakage_audit"]["leakage_count"], 1)
        self.assertEqual(
            build.report["classification"]["primary"],
            "public_rollout_history_context_source_integrity_failed",
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "history_context_leakage_or_integrity_failed",
        )

    def test_real_artifact_current_v133_facts_when_outputs_exist(self) -> None:
        required = [
            DEFAULT_V132_REPORT_PATH,
            DEFAULT_V132_DETAIL_ROWS_PATH,
            DEFAULT_V124_MANIFEST_PATH,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("local v132/v124 outputs are not present")

        build = build_first_recovery_public_rollout_history_context_audit()
        report = build.report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(
            report["source_integrity"]["v132_recommendation"],
            REQUIRED_V132_RECOMMENDATION,
        )
        self.assertEqual(
            report["classification"]["primary"],
            "public_rollout_history_context_ready_for_refreshed_surface",
        )
        self.assertEqual(
            report["recommendation"]["next_step"],
            "public_rollout_history_context_ready_for_refreshed_surface",
        )
        self.assertEqual(report["public_history_availability"]["branch_count"], 25)
        self.assertEqual(
            report["public_history_availability"]["history_available_row_count"],
            25,
        )
        self.assertEqual(
            report["public_history_availability"]["history_missing_row_count"],
            0,
        )
        self.assertEqual(
            report["extracted_history_feature_surface"]["history_feature_path_count"],
            79,
        )
        self.assertEqual(
            report["within_branch_history_variance"]["all_blocker_rows"][
                "varying_feature_path_count"
            ],
            54,
        )
        self.assertEqual(
            report["within_branch_history_variance"][
                "fixture_open_action_collapse_rows"
            ]["varying_feature_path_count"],
            27,
        )
        self.assertEqual(
            report["failed_vs_correct_history_contrast"][
                "contrasting_feature_path_count"
            ],
            41,
        )
        self.assertEqual(
            report["open_fixture_collapse_analysis"][
                "collapse_vs_nonfailed_open_contrast"
            ]["contrasting_feature_path_count"],
            34,
        )
        self.assertFalse(
            report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(report["authorization_block"]["training_executed"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_recommended"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_public_rollout_history_context_audit(
        v132_report=payloads["v132_report"],
        v132_report_path=None,
        v132_detail_rows=payloads["v132_detail_rows"],
        v132_detail_rows_path=None,
        v124_manifest_rows=payloads["manifest_rows"],
        v124_manifest_path=None,
        trajectory_records_by_path=payloads["trajectory_records_by_path"],
    )


def _synthetic_payloads() -> dict[str, object]:
    source_path = "synthetic://public-history"
    branch_specs = [
        ("b00_open_failed", "train", "open_mind_v3", "fixture_open_action_collapse", True, 1, 1),
        ("b01_open_correct", "train", "open_mind_v3", "inconclusive", False, 3, 2),
        ("b02_heldout_failed", "validation", "fixture_carrion_only", "train_support_hole", True, 5, 3),
        ("b03_heldout_correct", "test", "fixture_carrion_only", "inconclusive", False, 7, 4),
        ("b04_open_failed", "train", "open_mind_v3", "fixture_open_action_collapse", True, 9, 5),
    ]
    manifest_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for ordinal, spec in enumerate(branch_specs):
        branch_id, split, fixture, blocker_class, failed, record_index, agent_id = spec
        manifest_rows.append(
            {
                "branch_id": branch_id,
                "selected_observation_digest": f"digest-{ordinal}",
                "non_trainable_audit_metadata": {
                    "source_path": source_path,
                    "record_index": record_index,
                    "agent_id": agent_id,
                    "source_kind": fixture,
                },
            }
        )
        detail_rows.append(
            {
                "schema_version": (
                    "mind_v3_first_recovery_refreshed_surface_blocker_slice_row_v1"
                ),
                "candidate_set_ordinal": ordinal,
                "slice_memberships": (
                    ["fixture_open_mind_v3"]
                    if fixture == "open_mind_v3"
                    else ["heldout_validation_test"]
                ),
                "split": split,
                "fixture_group": fixture,
                "blocker_class": blocker_class,
                "is_failed_branch": failed,
                "prediction_correct": not failed,
                "repaired_action": "move_south",
                "predicted_action": "move_north" if failed else "move_south",
            }
        )
    records = [
        _record(agent_id=1, tick=0, action="eat", moved=False, digest="prior-0"),
        _record(agent_id=1, tick=1, action="move_north", moved=True, digest="digest-0"),
        _record(agent_id=2, tick=0, action="drink", moved=False, digest="prior-1", drank=True),
        _record(agent_id=2, tick=1, action="move_south", moved=True, digest="digest-1"),
        _record(agent_id=3, tick=0, action="stay", moved=False, digest="prior-2"),
        _record(agent_id=3, tick=1, action="eat", moved=False, digest="digest-2"),
        _record(agent_id=4, tick=0, action="move_east", moved=True, digest="prior-3"),
        _record(agent_id=4, tick=1, action="move_west", moved=True, digest="digest-3"),
        _record(agent_id=5, tick=0, action="stay", moved=False, digest="prior-4"),
        _record(agent_id=5, tick=1, action="move_north", moved=True, digest="digest-4"),
    ]
    return {
        "v132_report": _v132_report(detail_rows),
        "v132_detail_rows": detail_rows,
        "manifest_rows": manifest_rows,
        "trajectory_records_by_path": {source_path: records},
    }


def _v132_report(detail_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_SURFACE_BLOCKER_SLICE_AUDIT_SCHEMA_VERSION
        ),
        "source_integrity": {"passed": True, "failures": []},
        "recommendation": {"next_step": REQUIRED_V132_RECOMMENDATION},
        "detail_rows_summary": {"row_count": len(detail_rows)},
        "blocker_class_aggregate": {
            "counts_by_blocker_class": {
                "fixture_open_action_collapse": 2,
                "train_support_hole": 1,
                "inconclusive": 2,
            },
            "fixture_open_failures": {
                "dominant_predicted_action": {
                    "action": "move_north",
                    "count": 2,
                    "share": 0.6666666666666666,
                    "total": 3,
                    "failed": True,
                }
            },
        },
    }


def _record(
    *,
    agent_id: int,
    tick: int,
    action: str,
    moved: bool,
    digest: str,
    drank: bool = False,
) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": action,
        "resolved_action": action,
        "moved": moved,
        "observation_digest": digest,
        "action_mask": _action_mask(drink=drank, move_north=True, move_south=True),
        "observation_input": {"values": [0.5] * 542},
        "before": {
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 0.9,
        },
        "after": {
            "energy_ratio": 0.6 if action == "eat" else 0.48,
            "hydration_ratio": 0.7 if drank else 0.48,
            "health_ratio": 0.9,
        },
        "outcome": {
            "resource_gain": 0.1 if action == "eat" else 0.0,
            "feeding": {
                "ate": action == "eat",
                "food_source": "carcass" if action == "eat" else None,
            },
            "drinking": {"drank": drank},
            "movement": {"moved": moved},
        },
    }


def _action_mask(**overrides: bool) -> dict[str, bool]:
    mask = {name: False for name in ("eat", "drink", "move_north", "move_south", "stay")}
    mask.update(overrides)
    mask["eat"] = True
    mask["stay"] = True
    return mask
