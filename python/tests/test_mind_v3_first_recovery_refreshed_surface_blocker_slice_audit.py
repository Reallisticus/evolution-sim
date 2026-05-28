from __future__ import annotations

import json
import unittest
from pathlib import Path

from evolution_sim.cli.mind_v3_first_recovery_refreshed_surface_blocker_slice_audit import (
    build_parser,
)
from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V131_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION,
    MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    EXPECTED_V131_FAILURES,
    build_first_recovery_refreshed_surface_blocker_slice_audit,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryRefreshedSurfaceBlockerSliceAuditTests(
    unittest.TestCase
):
    def test_blocker_slice_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-refreshed-surface-blocker-slice-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_first_recovery_refreshed_surface_blocker_slice_audit"
            ),
        )

    def test_empty_detail_rows_output_argument_stays_empty(self) -> None:
        args = build_parser().parse_args(["--detail-rows-output", ""])

        self.assertEqual(args.detail_rows_output, "")

    def test_synthetic_clean_source_path_completes_diagnostics(self) -> None:
        build = _build(_synthetic_payloads())

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_surface_blocker_slice_audit_completed",
        )
        self.assertEqual(len(build.detail_rows), 5)
        self.assertFalse(
            build.report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["authorization_block"]["training_executed"])
        self.assertFalse(build.report["authorization_block"]["claim_causality"])

    def test_v131_source_integrity_failure_blocks_v132(self) -> None:
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
            "refreshed_surface_blocker_slice_audit_source_integrity_failed",
        )

    def test_missing_required_v131_failure_labels_blocks_v132(self) -> None:
        payloads = _synthetic_payloads()
        payloads["v131_report"]["metric_gate"]["failures"] = [
            "fixture_open_failed"
        ]

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v131_required_failure_labels_mismatch",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_surface_blocker_slice_audit_source_integrity_failed",
        )

    def test_heldout_branch_support_hole_classification(self) -> None:
        build = _build(_synthetic_payloads())

        self.assertTrue(
            any(
                row["blocker_class"] == "train_support_hole"
                and row["split"] == "validation"
                and row["repaired_action"] == "move_east"
                and row["predicted_action"] == "eat"
                for row in build.detail_rows
            )
        )

    def test_feature_alias_collision_classification(self) -> None:
        build = _build(_synthetic_payloads())

        self.assertTrue(
            any(
                row["blocker_class"] == "feature_alias_collision"
                and row["split"] == "test"
                and row["repaired_action"] == "move_east"
                and row["predicted_action"] == "move_east"
                for row in build.detail_rows
            )
        )

    def test_fixture_open_action_collapse_classification(self) -> None:
        build = _build(_synthetic_payloads())
        aggregate = build.report["blocker_class_aggregate"]

        self.assertEqual(
            aggregate["counts_by_blocker_class"]["fixture_open_action_collapse"],
            3,
        )
        self.assertEqual(
            aggregate["counts_by_failed_blocker_class"][
                "fixture_open_action_collapse"
            ],
            3,
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "add_public_rollout_history_context",
        )
        self.assertEqual(
            aggregate["fixture_open_failures"]["dominant_predicted_action"][
                "action"
            ],
            "move_north",
        )
        self.assertGreater(
            aggregate["fixture_open_failures"]["dominant_predicted_action"][
                "share"
            ],
            0.5,
        )

    def test_trainable_leakage_source_metadata_in_feature_rows_fails_closed(
        self,
    ) -> None:
        payloads = _synthetic_payloads()
        payloads["v131_feature_rows"][0]["candidate_public_features"][
            "source_path"
        ] = "private/fixture.json"
        payloads["v131_feature_rows"][1]["candidate_public_features"][
            "candidate_context_join_digest"
        ] = "not_trainable"

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v131_feature_rows_forbidden_or_leaky",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_surface_blocker_slice_audit_source_integrity_failed",
        )
        paths = {
            item["path"]
            for item in build.report["source_integrity"][
                "feature_row_forbidden_scan"
            ]["forbidden_feature_paths"]
        }
        self.assertIn("source_path", paths)
        self.assertIn("candidate_context_join_digest", paths)

    def test_real_artifact_current_v131_facts_when_outputs_exist(self) -> None:
        required = [
            DEFAULT_V131_REPORT_PATH,
            DEFAULT_V131_FEATURE_ROWS_PATH,
            DEFAULT_V124_MANIFEST_PATH,
            DEFAULT_V115_ARCHIVE_ROWS_PATH,
            DEFAULT_V123_ARCHIVE_ROWS_PATH,
        ]
        if not all(path.exists() for path in required):
            self.skipTest("local v131/v124/v115/v123 outputs are not present")

        build = build_first_recovery_refreshed_surface_blocker_slice_audit()
        report = build.report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(
            report["source_integrity"]["v131_classification"],
            "refreshed_candidate_public_feature_surface_blocked_by_signal",
        )
        self.assertEqual(
            report["source_integrity"]["v131_metric_failures"],
            sorted(EXPECTED_V131_FAILURES),
        )
        self.assertEqual(report["source_integrity"]["v131_feature_row_count"], 542)
        self.assertEqual(
            report["source_integrity"]["feature_row_forbidden_scan"][
                "forbidden_feature_path_count"
            ],
            0,
        )
        self.assertEqual(
            report["slice_extraction"]["heldout_validation_test_branch_count"],
            18,
        )
        self.assertEqual(
            report["slice_extraction"]["fixture_open_mind_v3_branch_count"],
            10,
        )
        self.assertEqual(
            report["blocker_class_aggregate"]["fixture_open_failures"][
                "dominant_predicted_action"
            ],
            {
                "action": "move_north",
                "count": 6,
                "failed": True,
                "share": 0.6,
                "total": 10,
            },
        )
        self.assertEqual(
            report["classification"]["primary"],
            "refreshed_surface_blocker_slice_audit_completed",
        )
        self.assertFalse(
            report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(report["authorization_block"]["v113_readiness_rerun_allowed"])
        self.assertFalse(report["authorization_block"]["training_executed"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_recommended"]
        )
        self.assertEqual(
            report["recommendation"]["next_step"],
            "add_public_rollout_history_context",
        )
        self.assertEqual(
            report["recommendation"]["selection_policy"],
            "largest_failed_blocker_class_before_nonfailed_support_notes",
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_refreshed_surface_blocker_slice_audit(
        v131_report=payloads["v131_report"],
        v131_report_path=None,
        v131_feature_rows=payloads["v131_feature_rows"],
        v131_feature_rows_path=None,
        v124_manifest_rows=payloads["manifest_rows"],
        v124_manifest_path=None,
        v115_archive_rows=payloads["candidate_rows"],
        v115_archive_rows_path=None,
        v123_archive_rows=[],
        v123_archive_rows_path=None,
    )


def _synthetic_payloads() -> dict[str, object]:
    branch_specs = [
        {
            "branch_id": "b00_train_eat",
            "split": "train",
            "fixture_group": "fixture_carrion_only",
            "repaired_action": "eat",
            "context": "train_eat",
            "candidates": [
                ("eat", True, True),
                ("move_north", False, False),
            ],
        },
        {
            "branch_id": "b01_train_north",
            "split": "train",
            "fixture_group": "fixture_carrion_only",
            "repaired_action": "move_north",
            "context": "train_north",
            "candidates": [
                ("move_north", True, True),
                ("move_south", False, False),
            ],
        },
        {
            "branch_id": "b10_holdout_support_hole",
            "split": "validation",
            "fixture_group": "fixture_carrion_only",
            "repaired_action": "move_east",
            "context": "support_hole",
            "candidates": [
                ("eat", False, False),
                ("move_east", True, True),
            ],
        },
        {
            "branch_id": "b11_holdout_alias",
            "split": "test",
            "fixture_group": "fixture_carrion_only",
            "repaired_action": "move_east",
            "context": "alias_collision",
            "alias_features": True,
            "candidates": [
                ("move_east", False, False),
                ("move_east", True, True),
            ],
        },
        {
            "branch_id": "b20_open_a",
            "split": "test",
            "fixture_group": "open_mind_v3",
            "repaired_action": "move_south",
            "context": "open_a",
            "candidates": [
                ("move_north", False, False),
                ("move_south", True, True),
            ],
        },
        {
            "branch_id": "b21_open_b",
            "split": "test",
            "fixture_group": "open_mind_v3",
            "repaired_action": "move_south",
            "context": "open_b",
            "candidates": [
                ("move_north", False, False),
                ("move_south", True, True),
            ],
        },
        {
            "branch_id": "b22_open_c",
            "split": "test",
            "fixture_group": "open_mind_v3",
            "repaired_action": "move_south",
            "context": "open_c",
            "candidates": [
                ("move_north", False, False),
                ("move_south", True, True),
            ],
        },
    ]
    manifest_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    branch_ordinals = {
        spec["branch_id"]: index
        for index, spec in enumerate(sorted(branch_specs, key=lambda item: item["branch_id"]))
    }
    for spec in sorted(branch_specs, key=lambda item: item["branch_id"]):
        branch_id = str(spec["branch_id"])
        sorted_candidates = sorted(
            enumerate(spec["candidates"]),
            key=lambda item: (
                str(item[1][0]),
                _archive_row_id(branch_id, item[0], item[1][1]),
            ),
        )
        repaired_archive_id = next(
            _archive_row_id(branch_id, original_index, is_positive)
            for original_index, (_, is_positive, _) in sorted_candidates
            if is_positive
        )
        manifest_rows.append(
            {
                "branch_id": branch_id,
                "repaired_action": spec["repaired_action"],
                "repaired_archive_row_id": repaired_archive_id,
                "non_trainable_audit_metadata": {
                    "source": spec["fixture_group"],
                },
            }
        )
        for candidate_ordinal, (original_index, candidate) in enumerate(
            sorted_candidates
        ):
            action, is_positive, material = candidate
            archive_id = _archive_row_id(branch_id, original_index, is_positive)
            candidate_rows.append(
                {
                    "archive_row_id": archive_id,
                    "candidate_action": action,
                    "material_gain_label": material,
                    "provenance": {"branch_id": branch_id},
                }
            )
            feature_context = (
                str(spec["context"])
                if spec.get("alias_features") is True
                else f"{spec['context']}_{candidate_ordinal}"
            )
            features = _features(action, feature_context)
            feature_rows.append(
                {
                    "schema_version": (
                        MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
                    ),
                    "candidate_public_features": features,
                    "extractor_source_paths": sorted(features),
                    "non_feature_metadata": {
                        "candidate_set_ordinal": branch_ordinals[branch_id],
                        "candidate_ordinal_within_set": candidate_ordinal,
                        "split": spec["split"],
                    },
                }
            )
    report = _v131_report(
        feature_rows=feature_rows,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
    )
    return {
        "v131_report": report,
        "v131_feature_rows": feature_rows,
        "manifest_rows": manifest_rows,
        "candidate_rows": candidate_rows,
    }


def _v131_report(
    *,
    feature_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    feature_rows_digest = stable_payload_digest(feature_rows)
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REFRESHED_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
        "classification": {
            "primary": "refreshed_candidate_public_feature_surface_blocked_by_signal",
        },
        "source_integrity": {
            "passed": True,
            "failures": [],
            "candidate_row_count": len(candidate_rows),
            "manifest_row_count": len(manifest_rows),
            "source_row_digests": {
                "v129_feature_rows": {
                    "matches_report_digest": True,
                    "computed_digest": "synthetic_v129_rows",
                    "reported_digest": "synthetic_v129_rows",
                },
                "v130_context_rows": {
                    "matches_report_digest": True,
                    "computed_digest": "synthetic_v130_rows",
                    "reported_digest": "synthetic_v130_rows",
                },
            },
        },
        "feature_surface_summary": {
            "candidate_feature_row_count": len(feature_rows),
            "feature_rows_digest": feature_rows_digest,
        },
        "forbidden_feature_scan": {"forbidden_feature_path_count": 0},
        "leakage_audit": {"leakage_count": 0},
        "metric_gate": {"failures": list(EXPECTED_V131_FAILURES)},
        "probe_comparison": {
            "heldout_accuracy": 0.5,
            "heldout_action_only_accuracy": 0.5,
            "heldout_action_order_accuracy": 0.25,
        },
        "fixture_open_evaluation": {
            "groups": {
                "open_mind_v3": {
                    "dominant_predicted_action": {
                        "action": "move_north",
                        "count": 3,
                        "share": 1.0,
                        "total": 3,
                        "failed": True,
                    }
                }
            }
        },
    }


def _features(action: str, context: str) -> dict[str, object]:
    return {
        "candidate_action": action,
        "candidate_action_observation_legal": True,
        "candidate_action_family": (
            "movement" if action.startswith("move_") else "resource_use"
        ),
        "candidate_action_direction": (
            action.removeprefix("move_") if action.startswith("move_") else "none"
        ),
        "candidate_target_context.availability_bucket": context,
        "candidate_resource_context.affordance_bucket": context,
        "candidate_neighborhood_context.summary_bucket": context,
    }


def _archive_row_id(branch_id: str, original_index: int, positive: bool) -> str:
    suffix = "1_positive" if positive else "0_negative"
    return f"{branch_id}_{original_index}_{suffix}"
