from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_shadow_scorer_execution import (
    build_first_recovery_shadow_scorer_execution,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    EXPECTED_REPAIRED_ACTION_COUNTS,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryShadowScorerExecutionTests(unittest.TestCase):
    def test_shadow_scorer_execution_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-shadow-scorer-execution"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_shadow_scorer_execution"
            ),
        )

    def test_positive_only_manifest_blocks_report_only_execution(self) -> None:
        payloads = _payloads()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "shadow_scorer_execution_blocked_by_metrics",
        )
        self.assertEqual(len(build.prediction_rows), 108)
        self.assertEqual(
            build.report["prediction_summary"]["predicted_action_counts"],
            EXPECTED_REPAIRED_ACTION_COUNTS,
        )
        audit = build.report["candidate_set_audit"]
        self.assertEqual(audit["manifest_row_count"], 108)
        self.assertEqual(audit["prediction_row_count"], 108)
        self.assertEqual(audit["predicted_equals_candidate_action_count"], 108)
        self.assertEqual(audit["candidate_action_equals_repaired_action_count"], 108)
        self.assertEqual(audit["predicted_equals_repaired_action_count"], 108)
        self.assertEqual(audit["non_selected_candidate_row_count"], 0)
        self.assertEqual(audit["branches_with_multiple_candidate_rows"], 0)
        self.assertTrue(audit["positive_only_manifest_detected"])
        self.assertTrue(audit["candidate_action_label_echo_detected"])
        self.assertFalse(build.report["heldout_current_row_signal"]["passed"])
        self.assertEqual(
            build.report["heldout_current_row_signal"]["label"],
            "heldout_row_group_label_echo_metrics_only",
        )
        self.assertFalse(build.report["seed29_evaluation"]["passed"])
        self.assertNotEqual(
            build.report["seed29_evaluation"]["label"],
            "shadow_ranker_seed29_passes",
        )
        self.assertFalse(build.report["fixture_open_evaluation"]["passed"])
        self.assertNotEqual(
            build.report["fixture_open_evaluation"]["label"],
            "shadow_ranker_fixture_open_generalizes",
        )
        self.assertEqual(build.report["material_gain_recall"]["recall"], 1.0)
        self.assertFalse(build.report["material_gain_recall"]["passed"])
        self.assertFalse(
            build.report["material_gain_recall"][
                "material_gain_floor_satisfied_by_exact_label"
            ]
        )
        self.assertFalse(
            build.report["material_gain_recall"]["claim_exact_material_gain_label"]
        )
        for failure in (
            "positive_only_manifest_no_candidate_ranking_evidence",
            "candidate_action_label_echo_detected",
            "material_gain_exact_label_missing",
        ):
            self.assertIn(failure, build.report["metric_gate"]["failures"])
        self.assertEqual(
            build.report["action_distribution"][
                "dominant_selected_action_share_max"
            ],
            0.5,
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])
        first_prediction = build.prediction_rows[0]
        self.assertFalse(first_prediction["scorer_input"]["contents_exposed"])
        self.assertIn("non_trainable_audit_metadata", first_prediction)
        self.assertEqual(
            first_prediction["split_assignment_role"],
            "evaluation_group_only_not_scorer_input",
        )

    def test_v125_source_integrity_failure_blocks_execution(self) -> None:
        payloads = _payloads()
        payloads["v125_report"]["source_integrity"]["passed"] = False
        payloads["v125_report"]["source_integrity"]["failures"] = ["bad"]

        build = _build(payloads)

        self._assert_source_failure(build, "v125_source_integrity_not_passed")
        self._assert_source_failure(
            build,
            "v125_source_failures_not_empty_or_malformed",
        )
        self.assertEqual(build.prediction_rows, ())

    def test_v125_required_metric_contract_tampering_fails_source_integrity(
        self,
    ) -> None:
        cases = {
            "seed29_missing": (
                lambda report: report["proposal"][
                    "planned_scorer_acceptance_metrics"
                ]["seed29_evaluation"].pop("required"),
                "v125_seed29_requirement_missing",
            ),
            "dominant_cap_wrong": (
                lambda report: report["proposal"][
                    "planned_scorer_acceptance_metrics"
                ]["action_distribution"].__setitem__(
                    "dominant_selected_action_share_max",
                    0.9,
                ),
                "v125_dominant_action_share_cap_unexpected",
            ),
            "material_floor_wrong": (
                lambda report: report["proposal"][
                    "planned_scorer_acceptance_metrics"
                ]["material_gain_recall"].__setitem__("minimum_recall", 0.1),
                "v125_material_gain_recall_floor_unexpected",
            ),
            "heldout_output_missing": (
                lambda report: report["proposal"][
                    "planned_scorer_acceptance_metrics"
                ]["required_execution_outputs"].remove("heldout_signal"),
                "v125_required_output_heldout_signal_missing",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                payloads = _payloads()
                mutate(payloads["v125_report"])

                build = _build(payloads)

                self._assert_source_failure(build, failure)

    def test_v125_authorization_tampering_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["v125_report"]["recommendation"][
            "downstream_shadow_scorer_allowed"
        ] = True

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "v125_downstream_shadow_scorer_allowed_not_false",
        )

    def test_v124_manifest_digest_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["v124_report"]["manifest"]["manifest_digest"] = "bad"

        build = _build(payloads)

        self._assert_source_failure(build, "v124_manifest_digest_mismatch")

    def test_metric_failure_blocks_after_clean_source(self) -> None:
        payloads = _payloads()
        for row in payloads["rows"]:
            if row["repaired_action"] == "eat":
                row["trainable_public_input"]["candidate_action"] = "stay"
                row["trainable_public_input"]["action_mask"]["stay"] = True
        _refresh_digests(payloads)

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "shadow_scorer_execution_blocked_by_metrics",
        )
        self.assertIn(
            "material_gain_exact_label_missing",
            build.report["metric_gate"]["failures"],
        )
        self.assertIn(
            "unsupported_selection_detected",
            build.report["metric_gate"]["failures"],
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_real_v124_v125_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v125-first-recovery-shadow-scorer-proposal.json",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v124/v125 artifacts are not present")

        build = build_first_recovery_shadow_scorer_execution()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "shadow_scorer_execution_blocked_by_metrics",
        )
        self.assertEqual(len(build.prediction_rows), 108)
        self.assertEqual(
            build.report["candidate_set_audit"][
                "candidate_action_equals_repaired_action_count"
            ],
            108,
        )
        self.assertEqual(
            build.report["candidate_set_audit"]["non_selected_candidate_row_count"],
            0,
        )
        self.assertTrue(
            build.report["candidate_set_audit"]["positive_only_manifest_detected"]
        )
        self.assertTrue(
            build.report["candidate_set_audit"][
                "candidate_action_label_echo_detected"
            ]
        )
        self.assertEqual(
            build.report["prediction_summary"]["predicted_action_counts"],
            EXPECTED_REPAIRED_ACTION_COUNTS,
        )
        self.assertEqual(build.report["unsupported_action_audit"]["unsupported_action_count"], 0)
        self.assertEqual(build.report["leakage_audit"]["trainable_leakage_count"], 0)
        for failure in (
            "positive_only_manifest_no_candidate_ranking_evidence",
            "candidate_action_label_echo_detected",
            "material_gain_exact_label_missing",
        ):
            self.assertIn(failure, build.report["metric_gate"]["failures"])
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )

    def _assert_source_failure(self, build, failure: str) -> None:
        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(failure, build.report["source_integrity"]["failures"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "shadow_scorer_execution_source_integrity_failed",
        )
        self.assertEqual(build.prediction_rows, ())
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_shadow_scorer_execution(
        v125_report=copy.deepcopy(payloads["v125_report"]),
        v124_report=copy.deepcopy(payloads["v124_report"]),
        v124_manifest_rows=copy.deepcopy(payloads["rows"]),
    )


def _payloads() -> dict[str, object]:
    rows = _manifest_rows()
    digest = stable_payload_digest(rows)
    v124_report = _v124_report(rows, digest)
    v125_report = _v125_report(rows, digest)
    return {"rows": rows, "v124_report": v124_report, "v125_report": v125_report}


def _refresh_digests(payloads: dict[str, object]) -> None:
    digest = stable_payload_digest(payloads["rows"])
    payloads["v124_report"]["manifest"]["manifest_digest"] = digest
    payloads["v125_report"]["source_integrity"]["manifest_digest"] = digest


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_cycle = [29, 13, 37, 41, 43, 19]
    for action, count in EXPECTED_REPAIRED_ACTION_COUNTS.items():
        for index in range(count):
            branch_id = f"v124-{action}-{index}"
            seed = seed_cycle[(len(rows) + index) % len(seed_cycle)]
            rows.append(
                {
                    "schema_version": (
                        "mind_v3_first_recovery_accepted_rare_attack_contract_v1"
                    ),
                    "branch_id": branch_id,
                    "current_oracle_action": "stay",
                    "repaired_action": action,
                    "current_archive_row_id": f"{branch_id}::action::stay",
                    "repaired_archive_row_id": f"{branch_id}::action::{action}",
                    "changed": action != "stay",
                    "unique_objective_best": False,
                    "objective_equivalence_verified": True,
                    "selected_resolution_legal": True,
                    "legal_tied_candidate_actions": [action, "stay"],
                    "serialized_objective_key": [0, 0.0, 0.0, 0.0, 0.0],
                    "selected_observation_digest": f"digest-{branch_id}",
                    "trainable_public_input": {
                        "schema_version": (
                            "mind_v3_first_recovery_branch_archive_trainable_public_input_v1"
                        ),
                        "candidate_action": action,
                        "candidate_action_index": index,
                        "action_mask": {action: True, "stay": True},
                        "post_carrion_first_recovery": True,
                        "public_transition_context": {
                            "records_after_animal_resource_gain": 1,
                            "ticks_after_animal_resource_gain": None,
                        },
                        "target_public_state_before": {
                            "age": 10,
                            "alive": True,
                            "energy_ratio": 0.9,
                            "health_ratio": 0.9,
                            "hydration_ratio": 0.9,
                        },
                    },
                    "non_trainable_audit_metadata": {
                        "non_trainable": True,
                        "purpose": "audit_only_not_trainable",
                        "seed": seed,
                        "source": "synthetic_fixture_open_source",
                    },
                    "diagnostics_only": True,
                    "selection_authorized": False,
                    "training_authorized": False,
                    "runtime_policy_authorized": False,
                }
            )
    return rows


def _v124_report(rows: list[dict[str, object]], digest: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_accepted_rare_attack_contract_v1",
        "classification": {
            "primary": "accepted_rare_attack_contract_ready_for_shadow_proposal"
        },
        "source_integrity": {
            "passed": True,
            "failures": [],
            "accepted_candidate_join_validation": {
                "passed": True,
                "mismatch_count": 0,
                "failure_labels": [],
            },
            "accepted_candidate_row_replay_validation": {
                "passed": True,
                "failure_count": 0,
                "failure_labels": [],
            },
        },
        "manifest": {
            "manifest_row_count": len(rows),
            "manifest_digest": digest,
        },
        "contract_checks": {
            "passed": True,
            "failures": [],
            "manifest_row_count": len(rows),
            "unique_branch_count": len(rows),
            "repaired_action_counts": dict(EXPECTED_REPAIRED_ACTION_COUNTS),
            "trainable_leakage": {
                "split_key_leak_count": 0,
                "forbidden_metadata_key_count": 0,
            },
            "branch_archive_trainable_leakage": {"leak_count": 0},
        },
        "split_support": {
            "strict_train_validation_test_support_met": True,
            "all_splits_contain_every_action_class": True,
            "minimum_per_action_support_by_split": {
                "train": 2,
                "validation": 1,
                "test": 1,
            },
        },
        "recommendation": {
            "training_executed": False,
            "trained_artifact_change_recommended": False,
            "v113_readiness_rerun_allowed": False,
            "downstream_shadow_scorer_allowed": False,
            "runtime_policy_change_recommended": False,
            "gate_change_recommended": False,
            "viewer_change_recommended": False,
            "replay_golden_change_recommended": False,
            "claim_causality": False,
        },
    }


def _v125_report(rows: list[dict[str, object]], digest: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_shadow_scorer_proposal_v1",
        "classification": {"primary": "shadow_scorer_proposal_ready_for_review"},
        "source_integrity": {
            "passed": True,
            "failures": [],
            "manifest_digest": digest,
            "manifest_row_count": len(rows),
            "unique_branch_count": len(rows),
            "v124_contract_boundary_checks": {"passed": True, "failures": []},
        },
        "proposal": {
            "planned_trainable_input_allowlist": {
                "top_level_keys_observed": [
                    "action_mask",
                    "candidate_action",
                    "candidate_action_index",
                    "post_carrion_first_recovery",
                    "public_transition_context",
                    "schema_version",
                    "target_public_state_before",
                ],
                "required_top_level_keys": [
                    "action_mask",
                    "candidate_action",
                    "candidate_action_index",
                    "post_carrion_first_recovery",
                    "public_transition_context",
                    "schema_version",
                    "target_public_state_before",
                ],
            },
            "planned_scorer_acceptance_metrics": {
                "heldout_current_row_signal": {"required": True},
                "seed29_evaluation": {"required": True},
                "fixture_open_generalization": {"required": True},
                "action_distribution": {
                    "dominant_selected_action_share_max": 0.5,
                },
                "unsupported_action_audit": {
                    "unsupported_action_rate_required": 0.0,
                },
                "leakage_audit": {"trainable_leakage_required": 0},
                "material_gain_recall": {"minimum_recall": 0.2},
                "required_execution_outputs": [
                    "per_branch_prediction_rows",
                    "heldout_signal",
                    "seed29_evaluation",
                    "fixture_open_evaluation",
                    "material_gain_recall",
                    "action_distribution",
                    "unsupported_action_audit",
                    "leakage_audit",
                ],
            },
            "planned_blocker_taxonomy": [
                {"label": "source_integrity_failed"},
                {"label": "heldout_signal_missing_or_weak"},
                {"label": "seed29_failed"},
                {"label": "fixture_open_failed"},
                {"label": "material_gain_recall_below_floor"},
            ],
        },
        "recommendation": {
            "shadow_scorer_executed": False,
            "shadow_scorer_execution_allowed": False,
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
        },
    }


if __name__ == "__main__":
    unittest.main()
