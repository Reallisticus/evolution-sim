from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    EXPECTED_REPAIRED_ACTION_COUNTS,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
    build_first_recovery_repaired_label_split_support_feasibility,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryRepairedLabelSplitSupportFeasibilityTests(unittest.TestCase):
    def test_split_support_feasibility_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-repaired-label-split-support-feasibility"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_repaired_label_split_support_feasibility"
            ),
        )

    def test_source_integrity_accepts_clean_v119_contract(self) -> None:
        rows = _expected_manifest_rows()
        build = _build(rows)
        source = build.report["source_integrity"]

        self.assertTrue(source["passed"])
        self.assertEqual(source["manifest_row_count"], 106)
        self.assertEqual(source["unique_branch_id_count"], 106)
        self.assertEqual(
            source["manifest_repaired_action_counts"],
            EXPECTED_REPAIRED_ACTION_COUNTS,
        )
        self.assertTrue(source["manifest_digest_matches"])

    def test_tampered_manifest_payload_fails_manifest_digest_mismatch(self) -> None:
        rows = _expected_manifest_rows()
        tampered = copy.deepcopy(rows)
        for row in tampered:
            row["branch_id"] = f"tampered-{row['branch_id']}"

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=_v119_report(rows),
            manifest_rows=tampered,
        )

        self._assert_source_failure(build, "manifest_digest_mismatch")

    def test_malformed_v119_manifest_digest_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        report["manifest"]["manifest_digest"] = "not-a-sha256-digest"

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self._assert_source_failure(build, "manifest_digest_malformed")

    def test_missing_v119_manifest_digest_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        del report["manifest"]["manifest_digest"]

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self._assert_source_failure(build, "manifest_digest_missing")

    def test_missing_repaired_action_counts_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        del report["contract_checks"]["repaired_action_counts"]

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self._assert_source_failure(build, "contract_repaired_action_counts_missing")

    def test_missing_or_mismatched_manifest_row_count_fails_source_integrity(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "missing": lambda report: report["manifest"].pop("manifest_row_count"),
            "mismatched": lambda report: report["manifest"].__setitem__(
                "manifest_row_count",
                len(rows) - 1,
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                report = _v119_report(rows)
                mutate(report)
                build = build_first_recovery_repaired_label_split_support_feasibility(
                    v119_report=report,
                    manifest_rows=rows,
                )
                self._assert_source_failure(
                    build,
                    "v119_manifest_row_count_mismatch",
                )

    def test_mismatched_contract_manifest_row_count_fails_source_integrity(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        report["contract_checks"]["manifest_row_count"] = len(rows) - 1

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self._assert_source_failure(build, "contract_manifest_row_count_mismatch")

    def test_mismatched_contract_branch_count_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        report["contract_checks"]["branch_count"] = len(rows) - 1

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self._assert_source_failure(build, "contract_branch_count_mismatch")

    def test_duplicate_branch_id_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        rows[1]["branch_id"] = rows[0]["branch_id"]

        build = _build(rows)

        self._assert_source_failure(build, "manifest_branch_ids_not_unique")

    def test_manifest_report_action_count_mismatch_fails_source_integrity(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        report["contract_checks"]["repaired_action_counts"]["stay"] -= 1

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self._assert_source_failure(build, "manifest_counts_do_not_match_v119_report")

    def test_missing_empty_or_non_string_branch_id_fails_source_integrity(self) -> None:
        cases = {
            "missing": lambda row: row.pop("branch_id"),
            "empty": lambda row: row.__setitem__("branch_id", ""),
            "non_string": lambda row: row.__setitem__("branch_id", 17),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                rows = _expected_manifest_rows()
                mutate(rows[0])
                build = build_first_recovery_repaired_label_split_support_feasibility(
                    v119_report=_v119_report(_expected_manifest_rows()),
                    manifest_rows=rows,
                )
                self._assert_source_failure(build, "manifest_branch_id_invalid")

    def test_contract_checks_passed_status_is_required_and_exactly_true(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "missing": (
                lambda report: report["contract_checks"].pop("passed"),
                "contract_checks_passed_missing",
            ),
            "false": (
                lambda report: report["contract_checks"].__setitem__(
                    "passed",
                    False,
                ),
                "contract_checks_passed_not_true",
            ),
            "malformed": (
                lambda report: report["contract_checks"].__setitem__(
                    "passed",
                    "true",
                ),
                "contract_checks_passed_not_true",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                report = _v119_report(rows)
                mutate(report)
                build = build_first_recovery_repaired_label_split_support_feasibility(
                    v119_report=report,
                    manifest_rows=rows,
                )
                self._assert_source_failure(build, failure)

    def test_contract_integrity_failures_status_is_required_and_empty(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        cases = {
            "missing": (
                lambda report: report["contract_checks"].pop("integrity_failures"),
                "contract_integrity_failures_missing",
            ),
            "non_empty": (
                lambda report: report["contract_checks"].__setitem__(
                    "integrity_failures",
                    ["bad"],
                ),
                "contract_integrity_failures_not_empty_or_malformed",
            ),
            "malformed": (
                lambda report: report["contract_checks"].__setitem__(
                    "integrity_failures",
                    "bad",
                ),
                "contract_integrity_failures_not_empty_or_malformed",
            ),
            "optional_failures_malformed": (
                lambda report: report["contract_checks"].__setitem__(
                    "failures",
                    "bad",
                ),
                "contract_failures_not_empty_or_malformed",
            ),
            "optional_source_failures_non_empty": (
                lambda report: report["contract_checks"].__setitem__(
                    "source_integrity_failures",
                    ["bad"],
                ),
                "contract_source_integrity_failures_not_empty_or_malformed",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                report = _v119_report(rows)
                mutate(report)
                build = build_first_recovery_repaired_label_split_support_feasibility(
                    v119_report=report,
                    manifest_rows=rows,
                )
                self._assert_source_failure(build, failure)

    def test_deterministic_stratified_split_covers_all_actions_but_strict_target_fails(
        self,
    ) -> None:
        build = _build(_expected_manifest_rows())
        policies = build.report["split_policy_evaluations"]
        stratified = policies["stratified_by_repaired_action_min_one_each"]
        scarcity = build.report["scarcity_analysis"]

        self.assertTrue(stratified["all_splits_contain_every_action_class"])
        self.assertFalse(stratified["train2_validation1_test1_target_met"])
        self.assertFalse(
            scarcity["train2_validation1_test1_feasible_by_total_support"]
        )
        self.assertEqual(
            scarcity[
                "rare_action_additional_needed_for_train2_validation1_test1"
            ],
            {"attack_east": 1, "attack_west": 1},
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "split_support_feasibility_limited_by_rare_actions",
        )

    def test_stratified_kfold_feasibility_reports_rare_action_limit(self) -> None:
        build = _build(_expected_manifest_rows())
        kfold = build.report["stratified_kfold_feasibility"]

        self.assertEqual(kfold["maximum_complete_action_class_k"], 3)
        self.assertTrue(kfold["k3_every_fold_has_every_action_class"])
        self.assertFalse(kfold["k4_every_fold_has_every_action_class"])

    def test_source_integrity_rejects_authorizing_v119_report(self) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        report["recommendation"]["downstream_shadow_scorer_allowed"] = True

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "shadow_scorer_not_blocked",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "repaired_label_source_integrity_failed",
        )

    def test_split_assignment_is_reported_as_excluded_from_trainable_input(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        clean = _build(rows).report["split_policy_evaluations"][
            "stratified_by_repaired_action_min_one_each"
        ]
        dirty_rows = copy.deepcopy(rows)
        dirty_rows[0]["trainable_public_input"]["split"] = "train"
        dirty_build = _build(dirty_rows)
        dirty = dirty_build.report["split_policy_evaluations"][
            "stratified_by_repaired_action_min_one_each"
        ]

        self.assertTrue(clean["policy"]["uses_only_allowed_audit_metadata"])
        self.assertTrue(clean["policy"]["excluded_from_trainable_public_input"])
        self.assertFalse(dirty["policy"]["excluded_from_trainable_public_input"])
        self._assert_source_failure(dirty_build, "trainable_split_assignment_leakage")

    def test_nested_split_assignment_key_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        rows[0]["trainable_public_input"]["nested"] = [
            {"safe": True},
            {"fold": 0},
        ]

        build = _build(rows)

        self._assert_source_failure(build, "trainable_split_assignment_leakage")

    def test_forbidden_trainable_metadata_aliases_fail_source_integrity(self) -> None:
        aliases = (
            "source_metadata",
            "audit_trail",
            "private_state",
            "seed_id",
            "digest_hex",
            "path_name",
            "logged_action",
            "world",
            "simulation_world",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                rows = _expected_manifest_rows()
                rows[0]["trainable_public_input"][alias] = "leak"
                build = _build(rows)
                self._assert_source_failure(build, "trainable_metadata_leakage")

    def test_nested_forbidden_trainable_metadata_fails_source_integrity(self) -> None:
        rows = _expected_manifest_rows()
        rows[0]["trainable_public_input"]["nested"] = [
            {"safe": True},
            {"audit_trail": ["not", "trainable"]},
        ]

        build = _build(rows)

        self._assert_source_failure(build, "trainable_metadata_leakage")

    def test_allowed_trainable_public_keys_do_not_trigger_metadata_leakage(
        self,
    ) -> None:
        rows = _expected_manifest_rows()
        rows[0]["trainable_public_input"].update(
            {
                "candidate_action_index": 0,
                "public_transition_context": {"candidate_action": "eat"},
                "target_public_state_before": {"alive": True},
            }
        )

        build = _build(rows)
        leakage = build.report["source_integrity"]["trainable_leakage"]

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(leakage["split_key_leak_count"], 0)
        self.assertEqual(leakage["forbidden_metadata_key_count"], 0)

    def test_trainable_metadata_leakage_fails_source_integrity(self) -> None:
        cases = {
            "seed": lambda trainable: trainable.__setitem__("seed", 13),
            "provenance": lambda trainable: trainable.__setitem__(
                "provenance",
                {"branch_id": "b0"},
            ),
            "audit": lambda trainable: trainable.__setitem__(
                "non_trainable_audit_metadata",
                {"non_trainable": True},
            ),
            "digest": lambda trainable: trainable.__setitem__(
                "observation_digest",
                "abc",
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                rows = _expected_manifest_rows()
                mutate(rows[0]["trainable_public_input"])
                build = _build(rows)
                self._assert_source_failure(build, "trainable_metadata_leakage")

    def test_source_integrity_failure_uses_fail_closed_recommendation(self) -> None:
        rows = _expected_manifest_rows()
        report = _v119_report(rows)
        del report["manifest"]["manifest_digest"]

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report=report,
            manifest_rows=rows,
        )

        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "source_integrity_must_pass_before_split_support_feasibility",
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_missing_inputs_are_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = build_first_recovery_repaired_label_split_support_feasibility(
                v119_report_path=root / "missing.json",
                manifest_path=root / "missing.jsonl",
            )

        self.assertEqual(
            build.report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        build = _build(_expected_manifest_rows())
        contract = build.report["contract"]

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
        )
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertFalse(contract["observation_field_change"])
        self.assertEqual(contract["viewer_effect"], "none")
        self.assertFalse(contract["v113_readiness_rerun_allowed"])
        self.assertFalse(contract["downstream_shadow_scorer_allowed"])
        self.assertTrue(build.report["non_promoted"])

    def test_real_v119_path_preserves_support_limited_result(self) -> None:
        report_path = (
            ROOT
            / "output/mind/mind-v3-v119-first-recovery-repaired-label-contract-audit.json"
        )
        manifest_path = (
            ROOT
            / "output/mind/mind-v3-v119-first-recovery-repaired-label-manifest.jsonl"
        )
        if not report_path.exists() or not manifest_path.exists():
            self.skipTest("real v119 report/manifest artifacts are not present")

        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report_path=report_path,
            manifest_path=manifest_path,
        )
        recommendation = build.report["recommendation"]

        self.assertEqual(
            build.report["classification"]["primary"],
            "split_support_feasibility_limited_by_rare_actions",
        )
        self.assertEqual(
            build.report["total_repaired_label_support"]["repaired_action_counts"],
            EXPECTED_REPAIRED_ACTION_COUNTS,
        )
        self.assertTrue(
            build.report["split_policy_evaluations"][
                "stratified_by_repaired_action_min_one_each"
            ]["all_splits_contain_every_action_class"]
        )
        self.assertEqual(
            build.report["scarcity_analysis"][
                "rare_action_additional_needed_for_train2_validation1_test1"
            ],
            {"attack_east": 1, "attack_west": 1},
        )
        self.assertFalse(recommendation["v113_readiness_rerun_allowed"])
        self.assertFalse(recommendation["downstream_shadow_scorer_allowed"])
        self.assertFalse(recommendation["claim_causality"])

    def _assert_source_failure(
        self,
        build,
        failure: str,
    ) -> None:
        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(failure, build.report["source_integrity"]["failures"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "repaired_label_source_integrity_failed",
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "source_integrity_must_pass_before_split_support_feasibility",
        )


def _build(rows: list[dict[str, object]]):
    return build_first_recovery_repaired_label_split_support_feasibility(
        v119_report=_v119_report(rows),
        manifest_rows=copy.deepcopy(rows),
    )


def _v119_report(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {}
    for row in rows:
        action = row["repaired_action"]
        counts[action] = counts.get(action, 0) + 1
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
            "repaired_action_counts": dict(sorted(counts.items())),
            "total_violation_count": 0,
            "integrity_failures": [],
        },
        "recommendation": {
            "v113_readiness_rerun_allowed": False,
            "downstream_shadow_scorer_allowed": False,
            "claim_causality": False,
        },
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
                    "changed": action != "stay",
                    "trainable_public_input": {
                        "candidate_action": action,
                        "target_public_state_before": {"alive": True},
                    },
                    "non_trainable_audit_metadata": {
                        "non_trainable": True,
                    },
                }
            )
    return rows


if __name__ == "__main__":
    unittest.main()
