from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION,
    MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION,
    build_first_recovery_rare_action_coverage_targeting,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    EXPECTED_REPAIRED_ACTION_COUNTS,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryRareActionCoverageTargetingTests(unittest.TestCase):
    def test_rare_action_coverage_targeting_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-rare-action-coverage-targeting"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_rare_action_coverage_targeting"
            ),
        )

    def test_source_integrity_failure_blocks_analysis_and_recommendation(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        v120 = _v120_report(rows)
        v120["source_integrity"]["passed"] = False

        build = _build(manifest_rows=rows, v120_report=v120)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v120_source_integrity_not_passed",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_action_coverage_source_integrity_failed",
        )
        self.assertTrue(build.report["candidate_search"]["analysis_blocked"])
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "source_integrity_must_pass_before_rare_action_coverage_targeting",
        )

    def test_empty_archive_rows_fail_source_integrity(self) -> None:
        build = _build(archive_rows=[])

        self._assert_source_failure(build, "v115_archive_rows_empty")
        self.assertIn(
            "v115_archive_row_count_mismatch",
            build.report["source_integrity"]["failures"],
        )
        self.assertTrue(build.report["candidate_search"]["analysis_blocked"])

    def test_partial_archive_rows_fail_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(rows)[:100]

        build = _build(manifest_rows=rows, archive_rows=archive_rows)

        self._assert_source_failure(build, "v115_archive_row_count_mismatch")
        self.assertIn(
            "v115_archive_branch_count_mismatch",
            build.report["source_integrity"]["failures"],
        )

    def test_archive_manifest_branch_mismatch_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(rows)
        archive_rows[0]["provenance"]["branch_id"] = "not-in-manifest"

        build = _build(manifest_rows=rows, archive_rows=archive_rows)

        self._assert_source_failure(build, "v115_archive_manifest_branch_mismatch")

    def test_branch_integrity_failure_is_source_integrity_failure(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(rows)
        branch_id = rows[0]["branch_id"]
        for row in archive_rows:
            if row["provenance"]["branch_id"] == branch_id and row["oracle_rank"] == 2:
                row["oracle_rank"] = 1
                break

        build = _build(manifest_rows=rows, archive_rows=archive_rows)

        self._assert_source_failure(build, "v115_archive_branch_integrity_failed")

    def test_missing_provenance_branch_id_on_non_rank1_row_fails(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(rows)
        for row in archive_rows:
            if row["oracle_rank"] != 1:
                row["provenance"].pop("branch_id")
                break

        build = _build(manifest_rows=rows, archive_rows=archive_rows)

        self._assert_source_failure(
            build,
            "v115_archive_branch_id_missing_or_malformed",
        )
        self.assertEqual(
            build.report["source_integrity"]["v115_archive_missing_branch_id_row_count"],
            1,
        )

    def test_same_count_provenance_corruption_fails_branch_shape(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(rows)
        from_branch = str(rows[0]["branch_id"])
        to_branch = str(rows[1]["branch_id"])
        for row in archive_rows:
            if row["provenance"]["branch_id"] == from_branch and row["oracle_rank"] != 1:
                row["provenance"]["branch_id"] = to_branch
                break

        build = _build(manifest_rows=rows, archive_rows=archive_rows)

        self._assert_source_failure(
            build,
            "v115_archive_branch_size_distribution_mismatch",
        )
        self.assertEqual(build.report["source_integrity"]["v115_archive_row_count"], 530)
        self.assertEqual(
            build.report["source_integrity"]["v115_archive_unique_branch_count"],
            106,
        )
        self.assertTrue(
            build.report["source_integrity"]["v115_archive_branch_sets_match_manifest"]
        )

    def test_symmetric_same_size_provenance_swap_fails_row_identity(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(rows)
        first_branch = str(rows[0]["branch_id"])
        second_branch = str(rows[1]["branch_id"])
        first_row = next(
            row
            for row in archive_rows
            if row["provenance"]["branch_id"] == first_branch
            and row["oracle_rank"] != 1
        )
        second_row = next(
            row
            for row in archive_rows
            if row["provenance"]["branch_id"] == second_branch
            and row["oracle_rank"] != 1
        )
        first_row["provenance"]["branch_id"] = second_branch
        second_row["provenance"]["branch_id"] = first_branch

        build = _build(manifest_rows=rows, archive_rows=archive_rows)

        self._assert_source_failure(build, "v115_archive_row_identity_mismatch")
        self.assertEqual(
            build.report["source_integrity"]["v115_archive_row_identity_mismatch_count"],
            2,
        )
        self.assertEqual(build.report["source_integrity"]["v115_archive_row_count"], 530)
        self.assertEqual(
            build.report["source_integrity"]["v115_archive_unique_branch_count"],
            106,
        )
        self.assertTrue(
            build.report["source_integrity"]["v115_archive_branch_sets_match_manifest"]
        )

    def test_missing_inputs_are_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = build_first_recovery_rare_action_coverage_targeting(
                archive_report_path=root / "missing-v115.json",
                archive_rows_path=root / "missing-v115.jsonl.gz",
                v118_report_path=root / "missing-v118.json",
                v119_report_path=root / "missing-v119.json",
                manifest_path=root / "missing-manifest.jsonl",
                v120_report_path=root / "missing-v120.json",
            )

        self.assertEqual(
            build.report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )

    def test_clean_synthetic_v120_and_v119_inputs_pass_source_checks(self) -> None:
        build = _build()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(build.report["source_integrity"]["failures"], [])
        self.assertTrue(
            build.report["source_integrity"]["manifest_digest_matches_v119"]
        )
        self.assertTrue(
            build.report["source_integrity"]["manifest_digest_matches_v120"]
        )

    def test_existing_rare_action_counts_are_reported_exactly(self) -> None:
        build = _build()
        support = build.report["current_rare_action_support"]

        self.assertEqual(
            support["repaired_action_counts"],
            {"attack_east": 3, "attack_west": 3},
        )
        self.assertEqual(
            support["per_action"]["attack_east"]["additional_needed_for_train2_validation1_test1"],
            1,
        )
        self.assertEqual(
            support["per_action"]["attack_west"]["additional_needed_for_train2_validation1_test1"],
            1,
        )

    def test_manifest_trainable_public_input_presence_is_required(self) -> None:
        cases = {
            "missing": lambda row: row.pop("trainable_public_input"),
            "non_mapping": lambda row: row.__setitem__(
                "trainable_public_input",
                ["not", "mapping"],
            ),
            "empty_mapping": lambda row: row.__setitem__(
                "trainable_public_input",
                {},
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                rows = _expected_manifest_rows()
                mutate(rows[0])
                build = _build(manifest_rows=rows)
                self._assert_source_failure(
                    build,
                    "manifest_trainable_public_input_missing_or_malformed",
                )

    def test_candidate_detection_finds_objective_equivalent_legal_attack_near_misses(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows("eat-000", target_action="attack_east"),
                "drink-000": _branch_rows("drink-000", target_action="attack_west"),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        search = build.report["candidate_search"]

        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_action_coverage_candidate_available",
        )
        self.assertTrue(search["all_required_actions_have_candidate"])
        self.assertEqual(
            search["per_action"]["attack_east"]["valid_candidate_count"],
            1,
        )
        self.assertEqual(
            search["per_action"]["attack_west"]["valid_candidate_count"],
            1,
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "propose_diagnostics_only_repair_policy_for_identified_rare_action_candidates",
        )

    def test_candidate_detection_rejects_resolution_invalid_candidates(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows(
                    "eat-000",
                    target_action="attack_east",
                    target_resolution_legal=False,
                ),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        attack = build.report["candidate_search"]["per_action"]["attack_east"]

        self.assertEqual(attack["valid_candidate_count"], 0)
        self.assertEqual(attack["rejection_counts"]["resolution_invalid"], 1)

    def test_candidate_detection_rejects_unique_objective_best_changes(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows(
                    "eat-000",
                    target_action="attack_east",
                    target_objective_delta=-1.0,
                ),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        attack = build.report["candidate_search"]["per_action"]["attack_east"]

        self.assertEqual(attack["valid_candidate_count"], 0)
        self.assertEqual(
            attack["rejection_counts"]["unique_objective_best_change_rejected"],
            1,
        )

    def test_candidate_detection_rejects_trainable_leakage(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows(
                    "eat-000",
                    target_action="attack_east",
                    target_trainable_extra={"seed_id": 13},
                ),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        attack = build.report["candidate_search"]["per_action"]["attack_east"]

        self.assertEqual(attack["valid_candidate_count"], 0)
        self.assertEqual(attack["rejection_counts"]["trainable_leakage"], 1)

    def test_candidate_missing_trainable_public_input_is_rejected(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows(
                    "eat-000",
                    target_action="attack_east",
                    target_trainable_value=None,
                ),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        attack = build.report["candidate_search"]["per_action"]["attack_east"]

        self.assertEqual(attack["valid_candidate_count"], 0)
        self.assertEqual(
            attack["rejection_counts"]["trainable_public_input_missing_or_malformed"],
            1,
        )

    def test_candidate_non_mapping_trainable_public_input_is_rejected(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows(
                    "eat-000",
                    target_action="attack_east",
                    target_trainable_value=["not", "mapping"],
                ),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        attack = build.report["candidate_search"]["per_action"]["attack_east"]

        self.assertEqual(attack["valid_candidate_count"], 0)
        self.assertEqual(
            attack["rejection_counts"]["trainable_public_input_missing_or_malformed"],
            1,
        )

    def test_candidate_empty_trainable_public_input_is_rejected(self) -> None:
        rows = _expected_manifest_rows()
        archive_rows = _expected_archive_rows(
            rows,
            branch_overrides={
                "eat-000": _branch_rows(
                    "eat-000",
                    target_action="attack_east",
                    target_trainable_value={},
                ),
            },
        )

        build = _build(manifest_rows=rows, archive_rows=archive_rows)
        attack = build.report["candidate_search"]["per_action"]["attack_east"]

        self.assertEqual(attack["valid_candidate_count"], 0)
        self.assertEqual(
            attack["rejection_counts"]["trainable_public_input_missing_or_malformed"],
            1,
        )

    def test_v120_source_failures_status_is_required_and_empty(self) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "missing": (
                lambda report: report["source_integrity"].pop("failures"),
                "v120_source_failures_missing",
            ),
            "non_empty": (
                lambda report: report["source_integrity"].__setitem__(
                    "failures",
                    ["bad"],
                ),
                "v120_source_failures_not_empty_or_malformed",
            ),
            "malformed": (
                lambda report: report["source_integrity"].__setitem__(
                    "failures",
                    "bad",
                ),
                "v120_source_failures_not_empty_or_malformed",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                v120 = _v120_report(rows)
                mutate(v120)
                self._assert_source_failure(
                    _build(manifest_rows=rows, v120_report=v120),
                    failure,
                )

    def test_v120_trainable_leakage_status_is_required_and_explicit(self) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "missing": (
                lambda report: report["source_integrity"].pop("trainable_leakage"),
                "v120_trainable_leakage_missing_or_malformed",
            ),
            "malformed": (
                lambda report: report["source_integrity"].__setitem__(
                    "trainable_leakage",
                    "bad",
                ),
                "v120_trainable_leakage_missing_or_malformed",
            ),
            "split_missing": (
                lambda report: report["source_integrity"]["trainable_leakage"].pop(
                    "split_key_leak_count"
                ),
                "v120_split_trainable_leakage_missing_or_malformed",
            ),
            "metadata_malformed": (
                lambda report: report["source_integrity"]["trainable_leakage"].__setitem__(
                    "forbidden_metadata_key_count",
                    "0",
                ),
                "v120_metadata_trainable_leakage_missing_or_malformed",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                v120 = _v120_report(rows)
                mutate(v120)
                self._assert_source_failure(
                    _build(manifest_rows=rows, v120_report=v120),
                    failure,
                )

    def test_v120_blocker_recommendation_flags_are_required_false(self) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "readiness_true": (
                "v113_readiness_rerun_allowed",
                True,
                "v120_v113_readiness_not_blocked",
            ),
            "shadow_missing": (
                "downstream_shadow_scorer_allowed",
                None,
                "v120_shadow_scorer_not_blocked",
            ),
            "runtime_true": (
                "runtime_policy_change_recommended",
                True,
                "v120_runtime_policy_change_recommended",
            ),
            "training_missing": (
                "trained_artifact_change_recommended",
                None,
                "v120_trained_artifact_change_recommended",
            ),
            "gate_true": (
                "gate_change_recommended",
                True,
                "v120_gate_change_recommended",
            ),
            "observation_missing": (
                "observation_field_change_recommended",
                None,
                "v120_observation_field_change_recommended",
            ),
            "viewer_true": (
                "viewer_change_recommended",
                True,
                "v120_viewer_change_recommended",
            ),
            "causality_missing": (
                "claim_causality",
                None,
                "v120_claim_causality_not_false",
            ),
        }
        for name, (field, value, failure) in cases.items():
            with self.subTest(name=name):
                v120 = _v120_report(rows)
                if value is None:
                    v120["recommendation"].pop(field)
                else:
                    v120["recommendation"][field] = value
                self._assert_source_failure(
                    _build(manifest_rows=rows, v120_report=v120),
                    failure,
                )

    def test_v119_classification_contract_and_recommendation_are_required(self) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "classification": (
                lambda report: report["classification"].__setitem__(
                    "primary",
                    "repaired_label_contract_ready_for_shadow_scorer_proposal",
                ),
                "v119_classification_not_support_limited",
            ),
            "contract_passed": (
                lambda report: report["contract_checks"].__setitem__("passed", False),
                "v119_contract_checks_not_passed",
            ),
            "violation_count": (
                lambda report: report["contract_checks"].__setitem__(
                    "total_violation_count",
                    1,
                ),
                "v119_contract_total_violation_count_not_zero",
            ),
            "integrity_failures": (
                lambda report: report["contract_checks"].__setitem__(
                    "integrity_failures",
                    ["bad"],
                ),
                "v119_contract_integrity_failures_not_empty_or_malformed",
            ),
            "readiness": (
                lambda report: report["recommendation"].__setitem__(
                    "v113_readiness_rerun_allowed",
                    True,
                ),
                "v119_v113_readiness_rerun_allowed_not_false",
            ),
            "shadow": (
                lambda report: report["recommendation"].__setitem__(
                    "downstream_shadow_scorer_allowed",
                    True,
                ),
                "v119_downstream_shadow_scorer_allowed_not_false",
            ),
            "causality": (
                lambda report: report["recommendation"].__setitem__(
                    "claim_causality",
                    True,
                ),
                "v119_claim_causality_not_false",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                v119 = _v119_report(rows)
                mutate(v119)
                build = _build(manifest_rows=rows, v119_report=v119)
                self._assert_source_failure(build, failure)

    def test_v118_classification_and_recommendation_are_required(self) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "classification": (
                lambda report: report["classification"].__setitem__(
                    "primary",
                    "tie_aware_repair_inconclusive",
                ),
                "v118_classification_not_clearing",
            ),
            "diagnostics_label_missing": (
                lambda report: report["classification"].__setitem__(
                    "labels",
                    ["tie_aware_repair_clears_action_collapse", "readiness_rerun_blocked"],
                ),
                "v118_classification_missing_diagnostics_only_no_runtime_promotion",
            ),
            "readiness_label_missing": (
                lambda report: report["classification"].__setitem__(
                    "labels",
                    [
                        "tie_aware_repair_clears_action_collapse",
                        "diagnostics_only_no_runtime_promotion",
                    ],
                ),
                "v118_classification_missing_readiness_rerun_blocked",
            ),
            "readiness": (
                lambda report: report["recommendation"].__setitem__(
                    "v113_readiness_rerun_allowed",
                    True,
                ),
                "v118_v113_readiness_rerun_allowed_not_false",
            ),
            "readiness_missing": (
                lambda report: report["recommendation"].pop(
                    "v113_readiness_rerun_allowed"
                ),
                "v118_v113_readiness_rerun_allowed_not_false",
            ),
            "readiness_malformed": (
                lambda report: report["recommendation"].__setitem__(
                    "v113_readiness_rerun_allowed",
                    "false",
                ),
                "v118_v113_readiness_rerun_allowed_not_false",
            ),
            "shadow": (
                lambda report: report["recommendation"].__setitem__(
                    "downstream_shadow_scorer_allowed",
                    True,
                ),
                "v118_downstream_shadow_scorer_allowed_not_false",
            ),
            "shadow_missing": (
                lambda report: report["recommendation"].pop(
                    "downstream_shadow_scorer_allowed"
                ),
                "v118_downstream_shadow_scorer_allowed_not_false",
            ),
            "causality": (
                lambda report: report["recommendation"].__setitem__(
                    "claim_causality",
                    True,
                ),
                "v118_claim_causality_not_false",
            ),
            "causality_malformed": (
                lambda report: report["recommendation"].__setitem__(
                    "claim_causality",
                    0,
                ),
                "v118_claim_causality_not_false",
            ),
            "runtime_true": (
                lambda report: report["recommendation"].__setitem__(
                    "runtime_policy_change_recommended",
                    True,
                ),
                "v118_runtime_policy_change_recommended_not_false",
            ),
            "training_missing": (
                lambda report: report["recommendation"].pop(
                    "trained_artifact_change_recommended"
                ),
                "v118_trained_artifact_change_recommended_not_false",
            ),
            "gate_malformed": (
                lambda report: report["recommendation"].__setitem__(
                    "gate_change_recommended",
                    "false",
                ),
                "v118_gate_change_recommended_not_false",
            ),
            "observation_true": (
                lambda report: report["recommendation"].__setitem__(
                    "observation_field_change_recommended",
                    True,
                ),
                "v118_observation_field_change_recommended_not_false",
            ),
            "viewer_missing": (
                lambda report: report["recommendation"].pop(
                    "viewer_change_recommended"
                ),
                "v118_viewer_change_recommended_not_false",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                v118 = _v118_report()
                mutate(v118)
                build = _build(manifest_rows=rows, v118_report=v118)
                self._assert_source_failure(build, failure)

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        build = _build()
        contract = build.report["contract"]

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION,
        )
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertFalse(contract["observation_field_change"])
        self.assertEqual(contract["viewer_effect"], "none")
        self.assertFalse(contract["v113_readiness_rerun_allowed"])
        self.assertFalse(contract["downstream_shadow_scorer_allowed"])
        self.assertFalse(contract["claim_causality"])
        self.assertTrue(build.report["non_promoted"])

    def test_readiness_shadow_and_causality_are_always_false(self) -> None:
        for build in (
            _build(),
            _build(
                archive_rows=_expected_archive_rows(
                    _expected_manifest_rows(),
                    branch_overrides={
                        "eat-000": _branch_rows(
                            "eat-000",
                            target_action="attack_east",
                        ),
                        "drink-000": _branch_rows(
                            "drink-000",
                            target_action="attack_west",
                        ),
                    },
                )
            ),
        ):
            with self.subTest(
                classification=build.report["classification"]["primary"]
            ):
                recommendation = build.report["recommendation"]
                self.assertFalse(recommendation["v113_readiness_rerun_allowed"])
                self.assertFalse(recommendation["downstream_shadow_scorer_allowed"])
                self.assertFalse(recommendation["claim_causality"])

    def test_real_artifact_path_pins_not_available_result(self) -> None:
        paths = [
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT / "output/mind/mind-v3-v118-first-recovery-tie-aware-label-repair.json",
            ROOT
            / "output/mind/mind-v3-v119-first-recovery-repaired-label-contract-audit.json",
            ROOT
            / "output/mind/mind-v3-v119-first-recovery-repaired-label-manifest.jsonl",
            ROOT
            / "output/mind/mind-v3-v120-first-recovery-repaired-label-split-support-feasibility.json",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115/v118/v119/v120 artifacts are not present")

        build = build_first_recovery_rare_action_coverage_targeting(
            archive_report_path=paths[0],
            archive_rows_path=paths[1],
            v118_report_path=paths[2],
            v119_report_path=paths[3],
            manifest_path=paths[4],
            v120_report_path=paths[5],
        )
        search = build.report["candidate_search"]
        source = build.report["source_integrity"]

        self.assertTrue(source["passed"])
        self.assertEqual(source["v115_archive_row_count"], 530)
        self.assertEqual(source["v115_archive_unique_branch_count"], 106)
        self.assertEqual(
            source["v115_archive_branch_size_distribution"],
            EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION,
        )
        self.assertTrue(source["manifest_digest_matches_v119"])
        self.assertTrue(source["manifest_digest_matches_v120"])
        self.assertEqual(source["v120_source_failures"], [])
        self.assertEqual(
            source["v120_trainable_leakage"],
            {"split_key_leak_count": 0, "forbidden_metadata_key_count": 0},
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_action_coverage_not_available_in_existing_archive",
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "collect_additional_first_recovery_attack_coverage_diagnostics",
        )
        self.assertEqual(
            build.report["current_rare_action_support"]["repaired_action_counts"],
            {"attack_east": 3, "attack_west": 3},
        )
        self.assertEqual(
            search["per_action"]["attack_east"]["valid_candidate_count"],
            0,
        )
        self.assertEqual(
            search["per_action"]["attack_west"]["valid_candidate_count"],
            0,
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["claim_causality"])

    def _assert_source_failure(
        self,
        build,
        failure: str,
    ) -> None:
        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(failure, build.report["source_integrity"]["failures"])
        self.assertTrue(build.report["candidate_search"]["analysis_blocked"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_action_coverage_source_integrity_failed",
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "source_integrity_must_pass_before_rare_action_coverage_targeting",
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["claim_causality"])


def _build(
    *,
    manifest_rows: list[dict[str, object]] | None = None,
    archive_rows: list[dict[str, object]] | None = None,
    v118_report: dict[str, object] | None = None,
    v119_report: dict[str, object] | None = None,
    v120_report: dict[str, object] | None = None,
):
    rows = copy.deepcopy(manifest_rows or _expected_manifest_rows())
    return build_first_recovery_rare_action_coverage_targeting(
        archive_report=_archive_report(),
        archive_rows=copy.deepcopy(
            archive_rows if archive_rows is not None else _expected_archive_rows(rows)
        ),
        v118_report=copy.deepcopy(v118_report or _v118_report()),
        v119_report=copy.deepcopy(v119_report or _v119_report(rows)),
        manifest_rows=copy.deepcopy(rows),
        v120_report=copy.deepcopy(v120_report or _v120_report(rows)),
    )


def _v119_report(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = _counts(rows)
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "classification": {
            "primary": "repaired_label_contract_support_limited",
        },
        "manifest": {
            "manifest_row_count": len(rows),
            "manifest_digest": stable_payload_digest(rows),
        },
        "contract_checks": {
            "passed": True,
            "manifest_row_count": len(rows),
            "branch_count": len({row["branch_id"] for row in rows}),
            "repaired_action_counts": counts,
            "total_violation_count": 0,
            "integrity_failures": [],
        },
        "recommendation": {
            "v113_readiness_rerun_allowed": False,
            "downstream_shadow_scorer_allowed": False,
            "claim_causality": False,
        },
    }


def _v120_report(rows: list[dict[str, object]]) -> dict[str, object]:
    digest = stable_payload_digest(rows)
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION
        ),
        "classification": {
            "primary": "split_support_feasibility_limited_by_rare_actions",
        },
        "source_integrity": {
            "passed": True,
            "failures": [],
            "manifest_digest_matches": True,
            "reported_manifest_digest": digest,
            "computed_manifest_digest": digest,
            "manifest_repaired_action_counts": _counts(rows),
            "trainable_leakage": {
                "split_key_leak_count": 0,
                "forbidden_metadata_key_count": 0,
            },
        },
        "scarcity_analysis": {
            "rare_action_additional_needed_for_train2_validation1_test1": {
                "attack_east": 1,
                "attack_west": 1,
            },
        },
        "recommendation": {
            "v113_readiness_rerun_allowed": False,
            "downstream_shadow_scorer_allowed": False,
            "claim_causality": False,
            "runtime_policy_change_recommended": False,
            "trained_artifact_change_recommended": False,
            "gate_change_recommended": False,
            "observation_field_change_recommended": False,
            "viewer_change_recommended": False,
        },
    }


def _v118_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
        "classification": {
            "primary": "tie_aware_repair_clears_action_collapse",
            "labels": [
                "tie_aware_repair_clears_action_collapse",
                "diagnostics_only_no_runtime_promotion",
                "readiness_rerun_blocked",
            ],
        },
        "recommendation": {
            "v113_readiness_rerun_allowed": False,
            "downstream_shadow_scorer_allowed": False,
            "claim_causality": False,
            "runtime_policy_change_recommended": False,
            "trained_artifact_change_recommended": False,
            "gate_change_recommended": False,
            "observation_field_change_recommended": False,
            "viewer_change_recommended": False,
        },
    }


def _archive_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    }


def _expected_manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for action, count in EXPECTED_REPAIRED_ACTION_COUNTS.items():
        for index in range(count):
            branch_id = f"{action}-{index:03d}"
            rows.append(
                {
                    "schema_version": (
                        MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION
                    ),
                    "branch_id": branch_id,
                    "current_oracle_action": "stay",
                    "repaired_action": action,
                    "current_archive_row_id": f"{branch_id}::action::stay",
                    "repaired_archive_row_id": f"{branch_id}::action::{action}",
                    "changed": action != "stay",
                    "unique_objective_best": False,
                    "serialized_objective_key": [0, 0.0, 0.0, 0.0, 0.0],
                    "legal_tied_candidate_actions": ["stay", action],
                    "trainable_public_input": {
                        "candidate_action": action,
                        "target_public_state_before": {"alive": True},
                    },
                    "selected_observation_digest": f"digest-{branch_id}",
                    "selected_resolution_legal": True,
                    "objective_equivalence_verified": True,
                    "non_trainable_audit_metadata": {
                        "non_trainable": True,
                    },
                }
            )
    return rows


def _branch_rows(
    branch_id: str,
    *,
    target_action: str,
    size: int = 5,
    target_resolution_legal: bool = True,
    target_objective_delta: float = 0.0,
    target_trainable_extra: dict[str, object] | None = None,
    target_trainable_value: object = "__default__",
) -> list[dict[str, object]]:
    filler_actions = [
        action
        for action in (
            "eat",
            "drink",
            "move_east",
            "move_north",
            "move_south",
            "move_west",
            "attack_north",
            "attack_south",
        )
        if action not in {"stay", target_action}
    ][: max(0, size - 2)]
    rows = [
        _archive_row(branch_id, "stay", oracle_rank=1),
        _archive_row(
            branch_id,
            target_action,
            oracle_rank=2,
            resolution_legal=target_resolution_legal,
            terminal_alive_delta=target_objective_delta,
            trainable_extra=target_trainable_extra,
            trainable_value=target_trainable_value,
        ),
    ] + [
        _archive_row(
            branch_id,
            action,
            oracle_rank=index + 3,
            terminal_alive_delta=target_objective_delta,
        )
        for index, action in enumerate(filler_actions)
    ]
    return rows[:size]


def _expected_archive_rows(
    manifest_rows: list[dict[str, object]],
    *,
    branch_overrides: dict[str, list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    overrides = branch_overrides or {}
    rows: list[dict[str, object]] = []
    for index, manifest_row in enumerate(manifest_rows):
        branch_id = str(manifest_row["branch_id"])
        expected_size = _expected_branch_size(index)
        if branch_id in overrides:
            branch_rows = _resize_branch_rows(
                branch_id,
                copy.deepcopy(overrides[branch_id]),
                expected_size,
            )
        else:
            repaired_action = str(manifest_row["repaired_action"])
            branch_rows = _branch_rows(
                branch_id,
                target_action=(
                    repaired_action if repaired_action != "stay" else "move_east"
                ),
                size=expected_size,
            )
        if len(branch_rows) != expected_size:
            raise AssertionError(
                f"test archive branch {branch_id} must have {expected_size} rows"
            )
        rows.extend(branch_rows)
    return rows


def _expected_branch_size(index: int) -> int:
    cursor = 0
    for size_text, count in EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION.items():
        next_cursor = cursor + count
        if cursor <= index < next_cursor:
            return int(size_text)
        cursor = next_cursor
    raise AssertionError(f"unexpected branch index {index}")


def _resize_branch_rows(
    branch_id: str,
    rows: list[dict[str, object]],
    expected_size: int,
) -> list[dict[str, object]]:
    if len(rows) > expected_size:
        return rows[:expected_size]
    used_actions = {str(row.get("candidate_action")) for row in rows}
    next_rank = max(int(row.get("oracle_rank", 0)) for row in rows) + 1
    for action in (
        "eat",
        "drink",
        "move_east",
        "move_north",
        "move_south",
        "move_west",
        "attack_north",
        "attack_south",
    ):
        if len(rows) >= expected_size:
            break
        if action in used_actions:
            continue
        rows.append(_archive_row(branch_id, action, oracle_rank=next_rank))
        used_actions.add(action)
        next_rank += 1
    if len(rows) != expected_size:
        raise AssertionError(f"could not resize branch {branch_id}")
    return rows


def _archive_row(
    branch_id: str,
    action: str,
    *,
    oracle_rank: int,
    resolution_legal: bool = True,
    terminal_alive_delta: float = 0.0,
    trainable_extra: dict[str, object] | None = None,
    trainable_value: object = "__default__",
) -> dict[str, object]:
    trainable = {
        "candidate_action": action,
        "target_public_state_before": {"alive": True},
    }
    if trainable_extra:
        trainable.update(trainable_extra)
    row = {
        "archive_row_id": f"{branch_id}::action::{action}",
        "branch_id": branch_id,
        "provenance": {"branch_id": branch_id},
        "candidate_action": action,
        "oracle_rank": oracle_rank,
        "resolution_legal": resolution_legal,
        "biology_homeostasis_labels": {"target_alive": False},
        "terminal_alive_delta": terminal_alive_delta,
        "birth_delta": 0.0,
        "recovery_vitals_deltas": {"target_recovery_score_delta": 0.0},
        "death_reduction_delta": 0.0,
    }
    if trainable_value == "__default__":
        row["trainable_public_input"] = trainable
    elif trainable_value is not None:
        row["trainable_public_input"] = trainable_value
    return row


def _counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        action = str(row["repaired_action"])
        counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    unittest.main()
