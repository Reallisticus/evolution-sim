from __future__ import annotations

import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_inspector_bundle import (
    MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION,
    build_first_recovery_inspector_bundle,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryInspectorBundleTests(unittest.TestCase):
    def test_inspector_bundle_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"]["sim:mind:v3:first-recovery-inspector-bundle"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_inspector_bundle"
            ),
        )

    def test_contract_and_boundary_labels_are_diagnostics_only(self) -> None:
        build = _build(_manifest_rows())
        contract = build.report["contract"]
        labels = build.report["boundary_labels"]
        recommendation = build.report["recommendation"]

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION,
        )
        self.assertTrue(contract["diagnostics_only"])
        self.assertFalse(contract["training_executed"])
        self.assertFalse(contract["runtime_policy_implemented"])
        self.assertFalse(contract["shadow_scorer_implemented"])
        self.assertFalse(contract["bundle_is_replay_contract"])
        self.assertFalse(contract["bundle_is_training_manifest"])
        self.assertFalse(labels["private_world_state_exposed"])
        self.assertFalse(labels["trainable_public_input_contents_exposed"])
        self.assertFalse(recommendation["v113_readiness_rerun_allowed"])
        self.assertFalse(recommendation["downstream_shadow_scorer_allowed"])
        self.assertFalse(recommendation["claim_causality"])

    def test_branch_rows_hide_trainable_payload_and_mark_audit_metadata(self) -> None:
        build = _build(_manifest_rows())
        branch = build.report["branch_rows"]["b001"]

        self.assertEqual(branch["audit_metadata"]["audit_only"], True)
        self.assertEqual(branch["audit_metadata"]["audit_metadata_trainable"], False)
        self.assertEqual(branch["audit_metadata"]["seed"], 13)
        self.assertEqual(branch["audit_metadata"]["agent_id"], 9)
        self.assertNotIn("target_public_state_before", branch["trainable_public_input"])
        self.assertFalse(branch["trainable_public_input"]["contents_exposed"])
        self.assertTrue(branch["trainable_public_input"]["present"])
        self.assertTrue(branch["trainable_public_input"]["clean"])
        self.assertNotIn("trainable_public_input", json.dumps(branch["audit_metadata"]))

    def test_classification_chain_and_blocker_summary_are_visible(self) -> None:
        build = _build(_manifest_rows())
        chain = build.report["classification_chain"]
        summary = build.report["blocker_summary"]

        self.assertEqual(chain["v116"]["primary"], "stay_oracle_dominance_detected")
        self.assertEqual(chain["v117"]["primary"], "tie_break_artifact_likely")
        self.assertEqual(
            chain["v118"]["primary"],
            "tie_aware_repair_clears_action_collapse",
        )
        self.assertEqual(
            chain["v119"]["primary"],
            "repaired_label_contract_support_limited",
        )
        self.assertEqual(
            chain["v120"]["primary"],
            "split_support_feasibility_limited_by_rare_actions",
        )
        self.assertEqual(
            chain["v121"]["primary"],
            "rare_action_coverage_not_available_in_existing_archive",
        )
        self.assertTrue(summary["v115_archive_verified"]["verified"])
        self.assertEqual(
            summary["v116_stay_dominance"]["dominant_oracle_action"],
            "stay",
        )
        self.assertEqual(
            summary["v121_no_recoverable_rare_attack_candidates"][
                "valid_candidate_counts"
            ],
            {"attack_east": 0, "attack_west": 0},
        )

    def test_manifest_digest_mismatch_fails_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        reports["v119"]["manifest"]["manifest_digest"] = "0" * 64
        build = build_first_recovery_inspector_bundle(
            archive_report=reports["v115"],
            archive_rows=[],
            v116_report=reports["v116"],
            v117_report=reports["v117"],
            v118_report=reports["v118"],
            v119_report=reports["v119"],
            manifest_rows=rows,
            v120_report=reports["v120"],
            v121_report=reports["v121"],
        )

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v119_manifest_digest_mismatch",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "first_recovery_inspector_source_integrity_failed",
        )

    def test_empty_archive_rows_fail_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        build = build_first_recovery_inspector_bundle(
            archive_report=reports["v115"],
            archive_rows=[],
            v116_report=reports["v116"],
            v117_report=reports["v117"],
            v118_report=reports["v118"],
            v119_report=reports["v119"],
            manifest_rows=rows,
            v120_report=reports["v120"],
            v121_report=reports["v121"],
        )

        failures = build.report["source_integrity"]["failures"]
        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn("v115_archive_rows_empty", failures)
        self.assertIn("v115_archive_row_count_mismatch", failures)
        self.assertEqual(
            build.report["classification"]["primary"],
            "first_recovery_inspector_source_integrity_failed",
        )

    def test_truncated_archive_rows_fail_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        build = build_first_recovery_inspector_bundle(
            archive_report=reports["v115"],
            archive_rows=_archive_rows(rows)[:1],
            v116_report=reports["v116"],
            v117_report=reports["v117"],
            v118_report=reports["v118"],
            v119_report=reports["v119"],
            manifest_rows=rows,
            v120_report=reports["v120"],
            v121_report=reports["v121"],
        )

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v115_archive_row_count_mismatch",
            build.report["source_integrity"]["failures"],
        )

    def test_branch_result_count_mismatch_fails_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        reports["v116"]["source_archive_verification"]["branch_result_count"] = 1

        build = _build_with_reports(rows, reports)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v115_branch_result_count_mismatch",
            build.report["source_integrity"]["failures"],
        )

    def test_replay_not_verified_fails_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        reports["v116"]["source_archive_verification"]["replay_verified"] = False

        build = _build_with_reports(rows, reports)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v115_replay_not_verified",
            build.report["source_integrity"]["failures"],
        )

    def test_missing_digest_fields_fail_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        del reports["v119"]["manifest"]["manifest_digest"]
        del reports["v120"]["source_integrity"]["reported_manifest_digest"]
        reports["v120"]["source_integrity"]["computed_manifest_digest"] = "not-a-digest"
        del reports["v121"]["source_integrity"]["manifest_digest"]

        build = _build_with_reports(rows, reports)
        failures = build.report["source_integrity"]["failures"]

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn("v119_manifest_digest_missing_or_malformed", failures)
        self.assertIn(
            "v120_reported_manifest_digest_missing_or_malformed",
            failures,
        )
        self.assertIn(
            "v120_computed_manifest_digest_missing_or_malformed",
            failures,
        )
        self.assertIn("v121_manifest_digest_missing_or_malformed", failures)

    def test_upstream_authorization_flags_fail_source_integrity(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        reports["v118"]["recommendation"]["runtime_policy_change_recommended"] = True
        del reports["v119"]["recommendation"]["trained_artifact_change_recommended"]
        reports["v120"]["recommendation"]["downstream_shadow_scorer_allowed"] = True
        reports["v121"]["recommendation"]["replay_golden_change_recommended"] = True

        build = _build_with_reports(rows, reports)
        failures = build.report["source_integrity"]["failures"]

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn("v118_runtime_policy_change_recommended_not_false", failures)
        self.assertIn("v119_trained_artifact_change_recommended_not_false", failures)
        self.assertIn("v120_downstream_shadow_scorer_allowed_not_false", failures)
        self.assertIn("v121_replay_golden_change_recommended_not_false", failures)

    def test_v121_expected_conclusion_is_pinned(self) -> None:
        rows = _manifest_rows()
        reports = _reports(rows)
        reports["v121"]["classification"]["primary"] = (
            "rare_action_coverage_candidate_available"
        )
        reports["v121"]["source_integrity"]["failures"] = ["tampered"]
        reports["v121"]["candidate_search"]["per_action"]["attack_east"][
            "valid_candidate_count"
        ] = 1

        build = _build_with_reports(rows, reports)
        failures = build.report["source_integrity"]["failures"]

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn("v121_classification_unexpected", failures)
        self.assertIn("v121_source_failures_not_empty", failures)
        self.assertIn("v121_candidate_counts_unexpected", failures)

    def test_missing_sources_do_not_authorize_any_downstream_path(self) -> None:
        build = build_first_recovery_inspector_bundle(
            archive_report_path=None,
            archive_rows_path=None,
            v116_report_path=None,
            v117_report_path=None,
            v118_report_path=None,
            v119_report_path=None,
            manifest_path=None,
            v120_report_path=None,
            v121_report_path=None,
        )

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertFalse(build.report["recommendation"]["v113_readiness_rerun_allowed"])
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_real_v115_v121_bundle_matches_expected_read_only_result(self) -> None:
        paths = [
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT
            / "output/mind/mind-v3-v116-first-recovery-archive-blocker-diagnostic.json",
            ROOT
            / "output/mind/mind-v3-v117-first-recovery-oracle-tie-break-audit.json",
            ROOT
            / "output/mind/mind-v3-v118-first-recovery-tie-aware-label-repair.json",
            ROOT
            / "output/mind/mind-v3-v119-first-recovery-repaired-label-contract-audit.json",
            ROOT
            / "output/mind/mind-v3-v119-first-recovery-repaired-label-manifest.jsonl",
            ROOT
            / "output/mind/mind-v3-v120-first-recovery-repaired-label-split-support-feasibility.json",
            ROOT
            / "output/mind/mind-v3-v121-first-recovery-rare-action-coverage-targeting.json",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115-v121 artifacts are not present")

        build = build_first_recovery_inspector_bundle(
            archive_report_path=paths[0],
            archive_rows_path=paths[1],
            v116_report_path=paths[2],
            v117_report_path=paths[3],
            v118_report_path=paths[4],
            v119_report_path=paths[5],
            manifest_path=paths[6],
            v120_report_path=paths[7],
            v121_report_path=paths[8],
        )

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(build.report["branch_count"], 106)
        self.assertEqual(build.report["source_integrity"]["archive_row_count"], 530)
        self.assertEqual(build.report["source_integrity"]["manifest_row_count"], 106)
        self.assertEqual(
            build.report["source_integrity"]["v121_candidate_counts"],
            {"attack_east": 0, "attack_west": 0},
        )
        self.assertEqual(build.report["source_integrity"]["failures"], [])
        self.assertEqual(
            build.report["classification"]["primary"],
            "first_recovery_inspector_bundle_ready",
        )
        self.assertEqual(
            build.report["classification_chain"]["v121"]["primary"],
            "rare_action_coverage_not_available_in_existing_archive",
        )
        self.assertEqual(
            build.report["blocker_summary"]["v121_no_recoverable_rare_attack_candidates"][
                "valid_candidate_counts"
            ],
            {"attack_east": 0, "attack_west": 0},
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )


def _build(rows: list[dict[str, object]]):
    reports = _reports(rows)
    return _build_with_reports(rows, reports)


def _build_with_reports(
    rows: list[dict[str, object]],
    reports: dict[str, dict[str, object]],
):
    return build_first_recovery_inspector_bundle(
        archive_report=reports["v115"],
        archive_rows=_archive_rows(rows),
        v116_report=reports["v116"],
        v117_report=reports["v117"],
        v118_report=reports["v118"],
        v119_report=reports["v119"],
        manifest_rows=rows,
        v120_report=reports["v120"],
        v121_report=reports["v121"],
    )


def _manifest_rows() -> list[dict[str, object]]:
    return [
        _manifest_row("b001", current="stay", repaired="move_east", changed=True),
        _manifest_row("b002", current="eat", repaired="eat", changed=False),
    ]


def _archive_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "archive_row_id": row["repaired_archive_row_id"],
            "candidate_action": row["repaired_action"],
            "provenance": {"branch_id": row["branch_id"]},
        }
        for row in rows
    ]


def _reports(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    digest = stable_payload_digest(list(rows))
    return {
        "v115": {
            "schema_version": "mind_v3_first_recovery_branch_archive_v1",
            "classification": {"primary": "branch_archive_replay_partial", "labels": []},
            "branch_archive_summary": {
                "replay_verified": True,
                "archive_row_count": 2,
            },
            "oracle_label_summary": {
                "dominant_oracle_action": "stay",
                "dominant_oracle_action_count": 1,
                "dominant_oracle_action_share": 0.5,
            },
            "legality_summary": {"resolution_invalid_count": 0},
            "research_recommendation": {"recommendation": "keep_diagnostics_only"},
        },
        "v116": {
            "schema_version": "mind_v3_first_recovery_archive_blocker_diagnostic_v1",
            "classification": {
                "primary": "stay_oracle_dominance_detected",
                "labels": ["diagnostics_only_no_runtime_promotion"],
            },
            "source_archive_verification": {
                "verification_passed": True,
                "replay_verified": True,
                "selected_target_count": 2,
                "skipped_target_count": 0,
                "branch_result_count": 2,
                "archive_row_count_report": 2,
                "expected_v115_facts": {"archive_row_count": 2},
                "heuristic_action_source_count": 0,
            },
            "stay_dominance_analysis": {
                "dominant_oracle_action": "stay",
                "dominant_oracle_action_count": 1,
                "dominant_oracle_action_share": 0.5,
                "objective_scoring_artifact_risk": True,
                "branch_outcome_signal_present": True,
            },
            "recommendation": {
                "next_step": "diagnose_stay_oracle_objective_tie_break",
                "v113_readiness_rerun_allowed": False,
                "downstream_shadow_scorer_allowed": False,
                "claim_causality": False,
            },
        },
        "v117": {
            "schema_version": "mind_v3_first_recovery_oracle_tie_break_audit_v1",
            "classification": {
                "primary": "tie_break_artifact_likely",
                "labels": ["diagnostics_only_no_runtime_promotion"],
            },
            "objective_tie_break_analysis": {
                "current_serialized_oracle_stay_count": 1,
                "unique_objective_best_action_counts": {"stay": 0},
                "tie_neutral_stay_counts": {"action_name": 0},
            },
            "recommendation": {
                "next_step": "treat_stay_dominance_as_tie_break_artifact",
                "v113_readiness_rerun_allowed": False,
                "downstream_shadow_scorer_allowed": False,
                "claim_causality": False,
            },
        },
        "v118": {
            "schema_version": "mind_v3_first_recovery_tie_aware_label_repair_v1",
            "classification": {
                "primary": "tie_aware_repair_clears_action_collapse",
                "labels": ["diagnostics_only_no_runtime_promotion"],
            },
            "tie_aware_label_repair": {
                "best_valid_policy": "action_balance_resolution_legal",
                "policies": {
                    "action_balance_resolution_legal": {
                        "repaired_action_counts": {"eat": 1, "move_east": 1},
                        "dominant_action": "eat",
                        "dominant_action_share": 0.5,
                        "changed_branch_count": 1,
                        "unique_best_changed_count": 0,
                        "resolution_invalid_selected_count": 0,
                        "objective_equivalence_violation_count": 0,
                    }
                },
            },
            "recommendation": _blocked_recommendation(
                "record_tie_aware_label_repair_as_diagnostic_only"
            ),
        },
        "v119": {
            "schema_version": "mind_v3_first_recovery_repaired_label_contract_audit_v1",
            "classification": {
                "primary": "repaired_label_contract_support_limited",
                "labels": ["diagnostics_only_no_runtime_promotion"],
            },
            "manifest": {"manifest_digest": digest, "manifest_row_count": len(rows)},
            "contract_checks": {
                "passed": True,
                "total_violation_count": 0,
                "repaired_action_counts": {"eat": 1, "move_east": 1},
            },
            "split_support": {"support_adequate": False, "warnings": ["thin_split"]},
            "recommendation": _blocked_recommendation(
                "keep_repaired_labels_diagnostics_only"
            ),
        },
        "v120": {
            "schema_version": "mind_v3_first_recovery_repaired_label_split_support_feasibility_v1",
            "classification": {
                "primary": "split_support_feasibility_limited_by_rare_actions",
                "labels": ["diagnostics_only_no_runtime_promotion"],
            },
            "source_integrity": {
                "passed": True,
                "failures": [],
                "reported_manifest_digest": digest,
                "computed_manifest_digest": digest,
            },
            "scarcity_analysis": {
                "all_splits_can_contain_every_action_class_by_total_support": True,
                "train2_validation1_test1_feasible_by_total_support": False,
                "rare_action_additional_needed_for_train2_validation1_test1": {
                    "attack_east": 1,
                    "attack_west": 1,
                },
            },
            "recommendation": _blocked_recommendation(
                "run_diagnostics_only_archive_coverage_slice_for_rare_repaired_actions"
            ),
        },
        "v121": {
            "schema_version": "mind_v3_first_recovery_rare_action_coverage_targeting_v1",
            "classification": {
                "primary": "rare_action_coverage_not_available_in_existing_archive",
                "labels": ["diagnostics_only_no_runtime_promotion"],
            },
            "source_integrity": {
                "passed": True,
                "failures": [],
                "manifest_digest": digest,
            },
            "current_rare_action_support": {
                "repaired_action_counts": {"attack_east": 0, "attack_west": 0},
            },
            "candidate_search": {
                "any_candidate_available": False,
                "all_required_actions_have_candidate": False,
                "per_action": {
                    "attack_east": {"valid_candidate_count": 0},
                    "attack_west": {"valid_candidate_count": 0},
                },
            },
            "recommendation": _blocked_recommendation(
                "collect_additional_first_recovery_attack_coverage_diagnostics"
            ),
        },
    }


def _blocked_recommendation(next_step: str) -> dict[str, object]:
    return {
        "next_step": next_step,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
    }


def _manifest_row(
    branch_id: str,
    *,
    current: str,
    repaired: str,
    changed: bool,
) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_repaired_label_contract_audit_v1",
        "branch_id": branch_id,
        "current_oracle_action": current,
        "repaired_action": repaired,
        "current_archive_row_id": f"{branch_id}::action::{current}",
        "repaired_archive_row_id": f"{branch_id}::action::{repaired}",
        "changed": changed,
        "unique_objective_best": not changed,
        "objective_equivalence_verified": True,
        "selected_resolution_legal": True,
        "selected_observation_digest": f"digest-{branch_id}",
        "legal_tied_candidate_actions": sorted({current, repaired}),
        "serialized_objective_key": [0, 0.0, 0.0, 0.0, 0.0],
        "trainable_public_input": {
            "candidate_action": repaired,
            "target_public_state_before": {"alive": True},
        },
        "non_trainable_audit_metadata": {
            "non_trainable": True,
            "purpose": "audit_only_not_trainable",
            "source_kind": "fixture_carrion_only",
            "source_path": f"output/test/{branch_id}.jsonl.gz",
            "seed": 13,
            "tick": 1,
            "agent_id": 9,
            "record_index": 2,
            "provenance": {
                "source_kind": "fixture_carrion_only",
                "source_path": f"output/test/{branch_id}.jsonl.gz",
                "seed": 13,
                "agent_id": 9,
                "record_index": 2,
                "logged_action": current,
                "branch_state_digest": f"branch-digest-{branch_id}",
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
