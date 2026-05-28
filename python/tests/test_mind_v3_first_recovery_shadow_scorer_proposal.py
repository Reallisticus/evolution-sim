from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    EXPECTED_REPAIRED_ACTION_COUNTS,
    build_first_recovery_shadow_scorer_proposal,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryShadowScorerProposalTests(unittest.TestCase):
    def test_shadow_scorer_proposal_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-shadow-scorer-proposal"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_shadow_scorer_proposal"
            ),
        )

    def test_clean_v124_contract_is_ready_for_review(self) -> None:
        payloads = _payloads()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "shadow_scorer_proposal_ready_for_review",
        )
        self.assertTrue(build.report["proposal"]["proposal_only"])
        self.assertTrue(
            build.report["source_integrity"]["v124_contract_boundary_checks"]["passed"]
        )
        self.assertFalse(build.report["proposal"]["shadow_scorer_execution_authorized"])
        self.assertFalse(
            build.report["recommendation"]["shadow_scorer_execution_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        forbidden = build.report["proposal"]["forbidden_input_list"][
            "explicit_forbidden_keys"
        ]
        for key in ("seed", "branch_id", "archive_row_id", "logged_action"):
            self.assertIn(key, forbidden)
        self.assertIn(
            "output/mind/mind-v3-v126-first-recovery-shadow-scorer-execution.json",
            build.report["proposal"]["planned_report_paths"]["execution_report"],
        )
        metrics = build.report["proposal"]["planned_scorer_acceptance_metrics"]
        self.assertIn("seed29_evaluation", metrics)
        self.assertIn("material_gain_recall", metrics)
        self.assertEqual(
            metrics["material_gain_recall"]["minimum_recall"],
            0.2,
        )
        self.assertEqual(
            metrics["action_distribution"]["dominant_selected_action_share_max"],
            0.5,
        )
        self.assertEqual(
            metrics["unsupported_action_audit"]["unsupported_action_rate_required"],
            0.0,
        )
        self.assertEqual(metrics["leakage_audit"]["trainable_leakage_required"], 0)
        for section in (
            "heldout_signal",
            "seed29_evaluation",
            "fixture_open_evaluation",
            "material_gain_recall",
            "action_distribution",
            "unsupported_action_audit",
            "leakage_audit",
        ):
            self.assertIn(section, metrics["required_execution_outputs"])
        blocker_labels = {
            item["label"] for item in build.report["proposal"]["planned_blocker_taxonomy"]
        }
        for label in (
            "heldout_signal_missing_or_weak",
            "seed29_failed",
            "fixture_open_failed",
            "material_gain_recall_below_floor",
        ):
            self.assertIn(label, blocker_labels)

    def test_v124_classification_tampering_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["report"]["classification"]["primary"] = "not_ready"

        build = _build(payloads)

        self._assert_source_failure(build, "v124_classification_unexpected")

    def test_v124_source_failures_fail_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["report"]["source_integrity"]["failures"] = ["bad"]

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "v124_source_failures_not_empty_or_malformed",
        )

    def test_manifest_row_count_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["rows"] = payloads["rows"][:-1]

        build = _build(payloads)

        self._assert_source_failure(build, "v124_manifest_row_count_unexpected")
        self._assert_source_failure(build, "v124_manifest_digest_mismatch")

    def test_action_count_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["report"]["contract_checks"]["repaired_action_counts"][
            "attack_east"
        ] = 3

        build = _build(payloads)

        self._assert_source_failure(build, "v124_repaired_action_counts_unexpected")

    def test_duplicate_manifest_branch_id_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["rows"][1]["branch_id"] = payloads["rows"][0]["branch_id"]

        build = _build(payloads)

        self._assert_source_failure(build, "v124_manifest_branch_ids_not_unique")

    def test_join_validation_failure_blocks_proposal(self) -> None:
        payloads = _payloads()
        payloads["report"]["source_integrity"][
            "accepted_candidate_join_validation"
        ]["passed"] = False
        payloads["report"]["source_integrity"][
            "accepted_candidate_join_validation"
        ]["mismatch_count"] = 1
        payloads["report"]["source_integrity"][
            "accepted_candidate_join_validation"
        ]["failure_labels"] = ["accepted_candidate_selected_action_mismatch"]

        build = _build(payloads)

        self._assert_source_failure(build, "v124_join_validation_not_passed")
        self._assert_source_failure(build, "v124_join_validation_mismatch_nonzero")

    def test_row_replay_validation_failure_blocks_proposal(self) -> None:
        payloads = _payloads()
        payloads["report"]["source_integrity"][
            "accepted_candidate_row_replay_validation"
        ]["passed"] = False
        payloads["report"]["source_integrity"][
            "accepted_candidate_row_replay_validation"
        ]["failure_count"] = 1
        payloads["report"]["source_integrity"][
            "accepted_candidate_row_replay_validation"
        ]["failure_labels"] = ["accepted_candidate_row_replay_result_not_true"]

        build = _build(payloads)

        self._assert_source_failure(build, "v124_row_replay_validation_not_passed")
        self._assert_source_failure(
            build,
            "v124_row_replay_validation_failures_nonzero",
        )

    def test_trainable_leakage_blocks_proposal(self) -> None:
        payloads = _payloads()
        payloads["rows"][0]["trainable_public_input"]["seed"] = 13
        payloads["report"]["contract_checks"]["trainable_leakage"][
            "forbidden_metadata_key_count"
        ] = 1

        build = _build(payloads)

        self._assert_source_failure(build, "v124_trainable_metadata_leakage_nonzero")
        self._assert_source_failure(
            build,
            "manifest_trainable_metadata_leakage_detected",
        )

    def test_authorization_flag_true_blocks_proposal(self) -> None:
        payloads = _payloads()
        payloads["report"]["recommendation"]["downstream_shadow_scorer_allowed"] = True

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "v124_downstream_shadow_scorer_allowed_not_false_or_none",
        )
        self.assertFalse(
            build.report["recommendation"]["shadow_scorer_execution_allowed"]
        )

    def test_v124_contract_boundary_tampering_fails_source_integrity(self) -> None:
        cases = {
            "diagnostics_only": (
                "diagnostics_only",
                False,
                "v124_contract_diagnostics_only_not_true",
            ),
            "training_executed": (
                "training_executed",
                True,
                "v124_contract_training_executed_not_false",
            ),
            "trained_artifact_effect": (
                "trained_artifact_effect",
                "writes_artifact",
                "v124_contract_trained_artifact_effect_not_none",
            ),
            "runtime_policy_effect": (
                "runtime_policy_effect",
                "changes_policy",
                "v124_contract_runtime_policy_effect_not_none",
            ),
            "gate_effect": (
                "gate_effect",
                "changes_gate",
                "v124_contract_gate_effect_not_none",
            ),
            "viewer_effect": (
                "viewer_effect",
                "changes_viewer",
                "v124_contract_viewer_effect_not_none",
            ),
            "replay_golden_effect": (
                "replay_golden_effect",
                "changes_replay",
                "v124_contract_replay_golden_effect_not_none",
            ),
            "readiness_rerun_executed": (
                "readiness_rerun_executed",
                True,
                "v124_contract_readiness_rerun_executed_not_false",
            ),
            "v113_readiness_rerun_allowed": (
                "v113_readiness_rerun_allowed",
                True,
                "v124_contract_v113_readiness_rerun_allowed_not_false",
            ),
            "downstream_shadow_scorer_allowed": (
                "downstream_shadow_scorer_allowed",
                True,
                "v124_contract_downstream_shadow_scorer_allowed_not_false",
            ),
            "claim_causality": (
                "claim_causality",
                True,
                "v124_contract_claim_causality_not_false",
            ),
            "shadow_scorer_implemented": (
                "shadow_scorer_implemented",
                True,
                "v124_contract_shadow_scorer_implemented_not_false",
            ),
            "source_seed_provenance_trainable": (
                "source_seed_provenance_trainable",
                True,
                "v124_contract_source_seed_provenance_trainable_not_false",
            ),
            "strict_heldout_generalization_claimed": (
                "strict_heldout_generalization_claimed",
                True,
                "v124_contract_strict_heldout_generalization_claimed_not_false",
            ),
            "synthetic_labels_created": (
                "synthetic_labels_created",
                True,
                "v124_contract_synthetic_labels_created_not_false",
            ),
            "objective_values_changed": (
                "objective_values_changed",
                True,
                "v124_contract_objective_values_changed_not_false",
            ),
        }
        for name, (field, value, failure) in cases.items():
            with self.subTest(name=name):
                payloads = _payloads()
                payloads["report"]["contract"][field] = value

                build = _build(payloads)

                self._assert_source_failure(build, failure)

    def test_strict_split_support_false_blocks_proposal(self) -> None:
        payloads = _payloads()
        payloads["report"]["split_support"][
            "strict_train_validation_test_support_met"
        ] = False

        build = _build(payloads)

        self._assert_source_failure(build, "v124_strict_split_support_not_true")

    def test_unsupported_selection_blocks_proposal(self) -> None:
        payloads = _payloads()
        payloads["rows"][0]["selected_resolution_legal"] = False
        payloads["report"]["manifest"]["manifest_digest"] = stable_payload_digest(
            payloads["rows"]
        )

        build = _build(payloads)

        self._assert_source_failure(build, "manifest_unsupported_selection_detected")

    def test_real_v124_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v124 artifacts are not present")

        build = build_first_recovery_shadow_scorer_proposal()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "shadow_scorer_proposal_ready_for_review",
        )
        self.assertEqual(
            build.report["source_integrity"]["repaired_action_counts"],
            EXPECTED_REPAIRED_ACTION_COUNTS,
        )
        self.assertTrue(
            build.report["source_integrity"]["v124_contract_boundary_checks"]["passed"]
        )
        self.assertFalse(
            build.report["recommendation"]["shadow_scorer_execution_allowed"]
        )
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
            "shadow_scorer_proposal_source_integrity_failed",
        )
        self.assertFalse(
            build.report["recommendation"]["shadow_scorer_execution_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_shadow_scorer_proposal(
        v124_report=copy.deepcopy(payloads["report"]),
        v124_manifest_rows=copy.deepcopy(payloads["rows"]),
    )


def _payloads() -> dict[str, object]:
    rows = _manifest_rows()
    digest = stable_payload_digest(rows)
    return {"rows": rows, "report": _v124_report(rows, digest)}


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for action, count in EXPECTED_REPAIRED_ACTION_COUNTS.items():
        for index in range(count):
            branch_id = f"v124-{action}-{index}"
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
        "contract": {
            "schema_version": "mind_v3_first_recovery_accepted_rare_attack_contract_v1",
            "diagnostics_only": True,
            "runtime_policy_effect": "none",
            "trained_artifact_effect": "none",
            "gate_effect": "none",
            "viewer_effect": "none",
            "replay_golden_effect": "none",
            "objective_values_changed": False,
            "synthetic_labels_created": False,
            "training_executed": False,
            "readiness_rerun_executed": False,
            "shadow_scorer_implemented": False,
            "source_seed_provenance_trainable": False,
            "strict_heldout_generalization_claimed": False,
            "v113_readiness_rerun_allowed": False,
            "downstream_shadow_scorer_allowed": False,
            "claim_causality": False,
        },
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
            "policy": {
                "method": "deterministic_stratified_by_repaired_action",
                "split_assignment_inputs": ["branch_id", "repaired_action"],
                "excluded_from_trainable_public_input": True,
            },
            "splits": {},
            "strict_targets": {"train_min": 2, "validation_min": 1, "test_min": 1},
        },
        "source_seed_policy": {
            "source_seed_in_trainable_input": False,
            "source_seed_trainable_leak_count": 0,
            "strict_heldout_generalization_claimed": False,
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


if __name__ == "__main__":
    unittest.main()
