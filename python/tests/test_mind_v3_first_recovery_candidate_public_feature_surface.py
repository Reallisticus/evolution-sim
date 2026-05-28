from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_candidate_public_feature_surface import (
    PRIMARY_PROBE,
    build_first_recovery_candidate_public_feature_surface,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    build_first_recovery_candidate_ranker_capacity_audit,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_first_recovery_candidate_ranker_capacity_audit import (
    _payloads as _v128_payloads,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryCandidatePublicFeatureSurfaceTests(unittest.TestCase):
    def test_candidate_public_feature_surface_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-candidate-public-feature-surface"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_candidate_public_feature_surface"
            ),
        )

    def test_missing_candidate_specific_target_signal_is_reported(self) -> None:
        payloads = _payloads_for_v129()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertTrue(build.report["feature_allowlist_audit"]["passed"])
        self.assertTrue(build.report["forbidden_feature_scan"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_missing_candidate_specific_signal",
        )
        self.assertFalse(
            build.report["candidate_specific_public_signal_inventory"][
                "candidate_specific_target_resource_neighborhood_fields_present"
            ]
        )
        self.assertIn(
            "target_resource_neighborhood_features_do_not_vary",
            build.report["metric_gate"]["failures"],
        )
        variance = build.report["within_branch_feature_variance"]
        self.assertTrue(
            variance["v128_baseline"]["only_candidate_action_and_index_vary"]
        )
        self.assertGreater(
            variance["candidate_public_features"][
                "action_semantics_variance"
            ]["varying_path_count"],
            0,
        )
        self.assertEqual(
            variance["candidate_public_features"][
                "target_resource_neighborhood_variance"
            ]["varying_path_count"],
            0,
        )
        self.assertNotIn(
            "non_action_candidate_specific_varying_path_count",
            variance["candidate_public_features"],
        )
        self.assertTrue(
            variance["candidate_public_features"][
                "deprecated_non_action_candidate_specific_variance"
            ]["deprecated"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )

    def test_action_derived_semantic_variation_alone_still_blocks(self) -> None:
        payloads = _payloads_for_v129()

        build = _build(payloads)

        candidate_variance = build.report["within_branch_feature_variance"][
            "candidate_public_features"
        ]
        self.assertGreater(
            candidate_variance["action_semantics_variance"]["varying_path_count"],
            0,
        )
        self.assertEqual(
            candidate_variance["target_resource_neighborhood_variance"][
                "varying_path_count"
            ],
            0,
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_missing_candidate_specific_signal",
        )
        self.assertIn(
            "target_resource_neighborhood_features_do_not_vary",
            build.report["metric_gate"]["failures"],
        )

    def test_optional_branch_constant_target_signal_still_blocks(self) -> None:
        payloads = _payloads_for_v129()
        _add_branch_constant_candidate_target_public_signal(payloads)

        build = _build(payloads)

        self.assertTrue(
            build.report["candidate_specific_public_signal_inventory"][
                "candidate_specific_target_resource_neighborhood_fields_present"
            ]
        )
        self.assertFalse(
            build.report["candidate_specific_public_signal_inventory"][
                "candidate_specific_target_resource_neighborhood_fields_vary"
            ]
        )
        self.assertEqual(
            build.report["within_branch_feature_variance"][
                "candidate_public_features"
            ]["target_resource_neighborhood_variance"]["varying_path_count"],
            0,
        )
        self.assertIn(
            "target_resource_neighborhood_features_do_not_vary",
            build.report["metric_gate"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_missing_candidate_specific_signal",
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_varying_candidate_target_public_signal_can_be_ready(self) -> None:
        payloads = _payloads_for_v129()
        _add_varying_candidate_target_public_signal_and_relabel(payloads)

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_ready_for_ranker_probe",
        )
        self.assertTrue(build.report["metric_gate"]["passed"])
        self.assertTrue(
            build.report["probe_comparisons"][
                "heldout_signal_beats_action_only_baseline"
            ]
        )
        self.assertTrue(
            build.report["candidate_specific_public_signal_inventory"][
                "candidate_specific_target_resource_neighborhood_fields_present"
            ]
        )
        self.assertTrue(
            build.report["candidate_specific_public_signal_inventory"][
                "candidate_specific_target_resource_neighborhood_fields_vary"
            ]
        )
        self.assertGreater(
            build.report["within_branch_feature_variance"][
                "candidate_public_features"
            ]["target_resource_neighborhood_variance"]["varying_path_count"],
            0,
        )
        self.assertIn(
            PRIMARY_PROBE,
            build.report["probe_comparisons"]["per_probe_vs_action_only"],
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])

    def test_forbidden_candidate_public_feature_path_blocks_surface(self) -> None:
        payloads = _payloads_for_v129()
        _add_forbidden_candidate_public_feature(payloads)

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertFalse(build.report["forbidden_feature_scan"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_source_integrity_failed",
        )
        forbidden_paths = {
            item["path"]
            for item in build.report["forbidden_feature_scan"][
                "forbidden_feature_paths"
            ]
        }
        self.assertIn("candidate_public_features.repaired_action", forbidden_paths)

    def test_v128_baseline_variance_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads_for_v129()
        payloads["v128_report"]["feature_variance"]["varying_paths"][
            "target_public_state_before.energy_ratio"
        ] = {"branch_count": 1, "example_branches": [], "field_family": "non_action"}

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v128_baseline_variance_not_action_only",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_source_integrity_failed",
        )

    def test_candidate_trainable_leakage_fails_source_integrity(self) -> None:
        payloads = _payloads_for_v129()
        payloads["v115_rows"][0]["trainable_public_input"]["seed"] = 29

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "candidate_trainable_leakage_detected",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_source_integrity_failed",
        )

    def test_real_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v128-first-recovery-candidate-ranker-capacity-audit.json",
            ROOT / "output/mind/mind-v3-v127-first-recovery-candidate-set-shadow-execution.json",
            ROOT / "output/mind/mind-v3-v127-first-recovery-candidate-set-predictions.jsonl",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115/v123/v124/v127/v128 artifacts are not present")

        build = build_first_recovery_candidate_public_feature_surface()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(build.report["source_integrity"]["candidate_row_count"], 542)
        self.assertTrue(build.report["feature_allowlist_audit"]["passed"])
        self.assertFalse(
            build.report["feature_allowlist_audit"]["candidate_action_index_used"]
        )
        self.assertTrue(build.report["forbidden_feature_scan"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_public_feature_surface_missing_candidate_specific_signal",
        )
        self.assertTrue(
            build.report["within_branch_feature_variance"]["v128_baseline"][
                "only_candidate_action_and_index_vary"
            ]
        )
        self.assertFalse(
            build.report["candidate_specific_public_signal_inventory"][
                "candidate_specific_target_resource_neighborhood_fields_present"
            ]
        )
        self.assertEqual(
            build.report["within_branch_feature_variance"][
                "candidate_public_features"
            ]["target_resource_neighborhood_variance"]["varying_path_count"],
            0,
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["runtime_policy_change_recommended"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_candidate_public_feature_surface(
        v128_report=copy.deepcopy(payloads["v128_report"]),
        v127_report=copy.deepcopy(payloads["v127_report"]),
        v127_prediction_rows=copy.deepcopy(payloads["v127_prediction_rows"]),
        v124_report=copy.deepcopy(payloads["v124_report"]),
        v124_manifest_rows=copy.deepcopy(payloads["manifest_rows"]),
        v115_report=copy.deepcopy(payloads["v115_report"]),
        v115_archive_rows=copy.deepcopy(payloads["v115_rows"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
    )


def _payloads_for_v129() -> dict[str, object]:
    payloads = _v128_payloads(context_mode="by_action")
    _refresh_v124_and_v128_reports(payloads)
    return payloads


def _refresh_v124_and_v128_reports(payloads: dict[str, object]) -> None:
    payloads["v128_report"] = build_first_recovery_candidate_ranker_capacity_audit(
        v127_report=copy.deepcopy(payloads["v127_report"]),
        v127_prediction_rows=copy.deepcopy(payloads["v127_prediction_rows"]),
        v124_manifest_rows=copy.deepcopy(payloads["manifest_rows"]),
        v115_report=copy.deepcopy(payloads["v115_report"]),
        v115_archive_rows=copy.deepcopy(payloads["v115_rows"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
    ).report
    payloads["v124_report"] = _v124_report(payloads["manifest_rows"])


def _v124_report(manifest_rows: list[dict[str, object]]) -> dict[str, object]:
    digest = stable_payload_digest(manifest_rows)
    return {
        "schema_version": "mind_v3_first_recovery_accepted_rare_attack_contract_v1",
        "classification": {
            "primary": "accepted_rare_attack_contract_ready_for_shadow_proposal"
        },
        "source_integrity": {"passed": True, "failures": []},
        "contract_checks": {"passed": True, "failures": []},
        "manifest": {
            "manifest_digest": digest,
            "manifest_row_count": len(manifest_rows),
        },
        "recommendation": _no_authorization(),
    }


def _add_branch_constant_candidate_target_public_signal(
    payloads: dict[str, object],
) -> None:
    for row in [*payloads["v115_rows"], *payloads["v123_rows"]]:
        row["trainable_public_input"]["candidate_target_public_state"] = {
            "public_affordance_bucket": "branch_constant"
        }


def _add_varying_candidate_target_public_signal_and_relabel(
    payloads: dict[str, object],
) -> None:
    candidates_by_branch: dict[str, list[dict[str, object]]] = {}
    for row in [*payloads["v115_rows"], *payloads["v123_rows"]]:
        candidates_by_branch.setdefault(
            str(row["provenance"]["branch_id"]),
            [],
        ).append(row)
    selected_by_branch: dict[str, dict[str, object]] = {}
    for branch_id, rows in candidates_by_branch.items():
        selected = max(
            rows,
            key=lambda row: _synthetic_public_affordance_score(
                branch_id,
                str(row["candidate_action"]),
            ),
        )
        selected_by_branch[branch_id] = selected
        for row in rows:
            is_selected = row is selected
            row["material_gain_label"] = is_selected
            row["trainable_public_input"]["candidate_target_public_state"] = {
                "public_affordance_bucket": "high" if is_selected else "low"
            }
    for manifest in payloads["manifest_rows"]:
        branch_id = str(manifest["branch_id"])
        selected = selected_by_branch[branch_id]
        manifest["repaired_action"] = selected["candidate_action"]
        manifest["repaired_archive_row_id"] = selected["archive_row_id"]
    payloads["v124_report"] = _v124_report(payloads["manifest_rows"])


def _synthetic_public_affordance_score(branch_id: str, action: str) -> int:
    digest = stable_payload_digest(
        {
            "public_feature": "candidate_target_public_state.public_affordance_bucket",
            "branch_id": branch_id,
            "candidate_action": action,
        }
    )
    return int(digest[:12], 16)


def _add_forbidden_candidate_public_feature(payloads: dict[str, object]) -> None:
    repaired_by_branch = {
        str(row["branch_id"]): str(row["repaired_action"])
        for row in payloads["manifest_rows"]
    }
    for row in [*payloads["v115_rows"], *payloads["v123_rows"]]:
        branch_id = str(row["provenance"]["branch_id"])
        row["trainable_public_input"]["candidate_public_features"] = {
            "repaired_action": repaired_by_branch[branch_id]
        }


def _no_authorization() -> dict[str, object]:
    return {
        "downstream_shadow_scorer_allowed": False,
        "training_executed": False,
        "trained_artifact_change_recommended": False,
        "model_artifact_created": False,
        "runtime_policy_change_recommended": False,
        "v113_readiness_rerun_allowed": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "foundation_change_recommended": False,
        "claim_causality": False,
    }


if __name__ == "__main__":
    unittest.main()
