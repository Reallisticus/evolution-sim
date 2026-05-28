from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    build_first_recovery_candidate_set_shadow_execution,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    EXPECTED_REPAIRED_ACTION_COUNTS,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryCandidateSetShadowExecutionTests(unittest.TestCase):
    def test_candidate_set_shadow_execution_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-candidate-set-shadow-execution"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_candidate_set_shadow_execution"
            ),
        )

    def test_complete_candidate_set_can_be_ready_for_review(self) -> None:
        payloads = _payloads()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertTrue(
            build.report["candidate_set_audit"][
                "candidate_ranking_evidence_available"
            ]
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_set_shadow_execution_ready_for_review",
        )
        self.assertEqual(len(build.prediction_rows), 108)
        self.assertEqual(
            build.report["prediction_summary"]["predicted_action_counts"],
            EXPECTED_REPAIRED_ACTION_COUNTS,
        )
        self.assertEqual(build.report["unsupported_action_audit"]["unsupported_action_count"], 0)
        self.assertEqual(build.report["leakage_audit"]["trainable_leakage_count"], 0)
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )

    def test_v126_must_show_positive_only_blocker(self) -> None:
        payloads = _payloads()
        payloads["v126_report"]["metric_gate"]["failures"] = [
            "material_gain_exact_label_missing"
        ]

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "v126_positive_only_metric_failures_missing",
        )

    def test_missing_candidate_rows_blocks_as_missing_candidate_sets(self) -> None:
        payloads = _payloads()
        missing_branch = payloads["manifest_rows"][0]["branch_id"]
        payloads["v115_rows"] = [
            row
            for row in payloads["v115_rows"]
            if row["provenance"]["branch_id"] != missing_branch
        ]
        payloads["v115_report"]["branch_archive_summary"]["archive_row_count"] = len(
            payloads["v115_rows"]
        )
        payloads["v115_report"]["branch_archive_summary"]["branch_result_count"] = len(
            {row["provenance"]["branch_id"] for row in payloads["v115_rows"]}
        )

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "candidate_set_branches_missing",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_set_shadow_execution_blocked_by_missing_candidate_sets",
        )
        self.assertEqual(build.prediction_rows, ())

    def test_repaired_label_missing_from_candidate_set_fails_source_integrity(
        self,
    ) -> None:
        payloads = _payloads()
        target = payloads["manifest_rows"][0]
        for row in payloads["v115_rows"]:
            if (
                row["provenance"]["branch_id"] == target["branch_id"]
                and row["candidate_action"] == target["repaired_action"]
            ):
                row["candidate_action"] = "zzz_other"
                row["archive_row_id"] = (
                    f"{target['branch_id']}::action::zzz_other"
                )

        build = _build(payloads)

        self._assert_source_failure(build, "repaired_action_missing_from_candidate_set")

    def test_repaired_archive_row_id_missing_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["manifest_rows"][0]["repaired_archive_row_id"] = ""
        _refresh_manifest_digests(payloads)

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "repaired_archive_row_id_missing_or_malformed",
        )

    def test_repaired_archive_row_id_not_found_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["manifest_rows"][0][
            "repaired_archive_row_id"
        ] = "missing-branch::action::eat"
        _refresh_manifest_digests(payloads)

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "repaired_archive_row_id_not_in_candidate_set",
        )

    def test_repaired_archive_row_id_branch_mismatch_fails_source_integrity(
        self,
    ) -> None:
        payloads = _payloads()
        target_id = payloads["manifest_rows"][0]["repaired_archive_row_id"]
        for row in payloads["v115_rows"]:
            if row["archive_row_id"] == target_id:
                row["provenance"]["branch_id"] = "different-branch"
        _refresh_archive_reports(payloads)

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "repaired_archive_row_id_branch_mismatch",
        )

    def test_repaired_archive_row_id_action_mismatch_fails_source_integrity(
        self,
    ) -> None:
        payloads = _payloads()
        target_id = payloads["manifest_rows"][0]["repaired_archive_row_id"]
        for row in payloads["v115_rows"]:
            if row["archive_row_id"] == target_id:
                row["candidate_action"] = "different_action"

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "repaired_archive_row_id_action_mismatch",
        )

    def test_stale_manifest_digest_in_v125_or_v126_fails_source_integrity(
        self,
    ) -> None:
        cases = {
            "v125": (
                lambda payloads: payloads["v125_report"]["source_integrity"].__setitem__(
                    "manifest_digest",
                    "bad",
                ),
                "v125_manifest_digest_mismatch",
            ),
            "v126": (
                lambda payloads: payloads["v126_report"]["source_integrity"].__setitem__(
                    "manifest_digest",
                    "bad",
                ),
                "v126_manifest_digest_mismatch",
            ),
            "v126_bool": (
                lambda payloads: payloads["v126_report"]["source_integrity"].__setitem__(
                    "manifest_digest_matches_v124",
                    False,
                ),
                "v126_manifest_digest_matches_v124_not_true",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                payloads = _payloads()
                mutate(payloads)

                build = _build(payloads)

                self._assert_source_failure(build, failure)

    def test_stale_archive_row_digest_fails_source_integrity(self) -> None:
        cases = {
            "v115": ("v115_report", "v115_archive_rows_digest_mismatch"),
            "v123": ("v123_report", "v123_archive_rows_digest_mismatch"),
        }
        for name, (report_key, failure) in cases.items():
            with self.subTest(name=name):
                payloads = _payloads()
                payloads[report_key]["branch_archive_summary"][
                    "archive_rows_sha256"
                ] = "0" * 64

                build = _build(payloads)

                self._assert_source_failure(build, failure)

    def test_missing_authorization_fields_are_listed_not_null(self) -> None:
        payloads = _payloads()
        payloads["v125_report"]["recommendation"].pop("foundation_change_recommended")

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        summary = build.report["source_integrity"]["authorization_summary"]["v125"]
        self.assertIn(
            "foundation_change_recommended",
            summary["legacy_missing_authorization_fields"],
        )
        self.assertNotIn(
            "foundation_change_recommended",
            summary["present_false_or_none_authorization_fields"],
        )

    def test_candidate_trainable_leakage_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["v115_rows"][0]["trainable_public_input"]["seed"] = 13

        build = _build(payloads)

        self._assert_source_failure(build, "candidate_trainable_leakage_detected")

    def test_metrics_block_when_action_order_baseline_collapses(self) -> None:
        payloads = _payloads(negative_action="aaa_non_selected")

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_set_shadow_execution_blocked_by_metrics",
        )
        self.assertIn("action_collapse_detected", build.report["metric_gate"]["failures"])
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_real_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v126-first-recovery-shadow-scorer-execution.json",
            ROOT / "output/mind/mind-v3-v125-first-recovery-shadow-scorer-proposal.json",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115/v123/v124/v125/v126 artifacts are not present")

        build = build_first_recovery_candidate_set_shadow_execution()
        audit = build.report["candidate_set_audit"]

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_set_shadow_execution_blocked_by_metrics",
        )
        self.assertEqual(audit["branch_count"], 108)
        self.assertEqual(audit["total_candidate_rows"], 542)
        self.assertEqual(
            audit["candidate_rows_per_branch_distribution"],
            {"3": 10, "4": 24, "5": 32, "6": 38, "7": 4},
        )
        self.assertEqual(audit["positive_rows_count"], 108)
        self.assertEqual(audit["negative_rows_count"], 434)
        self.assertEqual(audit["missing_candidate_set_branch_count"], 0)
        self.assertEqual(audit["unsupported_repaired_labels_count"], 0)
        self.assertTrue(audit["candidate_ranking_evidence_available"])
        self.assertIn("action_collapse_detected", build.report["metric_gate"]["failures"])
        self.assertIn(
            "material_gain_recall_below_floor",
            build.report["metric_gate"]["failures"],
        )
        self.assertIn("fixture_open_failed", build.report["metric_gate"]["failures"])
        self.assertIn("seed29_failed", build.report["metric_gate"]["failures"])
        self.assertEqual(
            audit["repaired_archive_row_join"]["exact_join_count"],
            108,
        )
        self.assertEqual(build.report["unsupported_action_audit"]["unsupported_action_count"], 0)
        self.assertEqual(build.report["leakage_audit"]["trainable_leakage_count"], 0)
        self.assertFalse(build.report["seed29_evaluation"]["passed"])
        self.assertFalse(build.report["fixture_open_evaluation"]["passed"])
        self.assertEqual(
            build.report["material_gain_recall"]["metric_name"],
            "repaired_label_material_gain_exact_match_recall",
        )
        self.assertFalse(
            build.report["material_gain_recall"]["generic_material_gain_recall_claimed"]
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
            "candidate_set_shadow_execution_source_integrity_failed",
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_candidate_set_shadow_execution(
        v126_report=copy.deepcopy(payloads["v126_report"]),
        v125_report=copy.deepcopy(payloads["v125_report"]),
        v124_report=copy.deepcopy(payloads["v124_report"]),
        v124_manifest_rows=copy.deepcopy(payloads["manifest_rows"]),
        v115_report=copy.deepcopy(payloads["v115_report"]),
        v115_archive_rows=copy.deepcopy(payloads["v115_rows"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
    )


def _payloads(*, negative_action: str = "zzz_non_selected") -> dict[str, object]:
    manifest_rows = _manifest_rows()
    v115_branches = {row["branch_id"] for row in manifest_rows[:106]}
    v115_rows: list[dict[str, object]] = []
    v123_rows: list[dict[str, object]] = []
    for manifest in manifest_rows:
        target = v115_rows if manifest["branch_id"] in v115_branches else v123_rows
        target.extend(_candidate_rows(manifest, negative_action=negative_action))
    digest = stable_payload_digest(manifest_rows)
    return {
        "manifest_rows": manifest_rows,
        "v115_rows": v115_rows,
        "v123_rows": v123_rows,
        "v126_report": _v126_report(digest),
        "v125_report": _v125_report(digest),
        "v124_report": _v124_report(manifest_rows, digest),
        "v115_report": _archive_report(v115_rows),
        "v123_report": _archive_report(v123_rows, active=True),
    }


def _refresh_manifest_digests(payloads: dict[str, object]) -> None:
    digest = stable_payload_digest(payloads["manifest_rows"])
    payloads["v124_report"]["manifest"]["manifest_digest"] = digest
    payloads["v125_report"]["source_integrity"]["manifest_digest"] = digest
    payloads["v126_report"]["source_integrity"]["manifest_digest"] = digest


def _refresh_archive_reports(payloads: dict[str, object]) -> None:
    for rows_key, report_key in (
        ("v115_rows", "v115_report"),
        ("v123_rows", "v123_report"),
    ):
        rows = payloads[rows_key]
        payloads[report_key]["branch_archive_summary"]["archive_row_count"] = len(rows)
        payloads[report_key]["branch_archive_summary"]["branch_result_count"] = len(
            {row["provenance"]["branch_id"] for row in rows}
        )
        payloads[report_key]["branch_archive_summary"]["archive_rows_sha256"] = (
            stable_payload_digest(rows)
        )


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_cycle = [29, 13, 37, 41, 43, 19]
    for action, count in EXPECTED_REPAIRED_ACTION_COUNTS.items():
        for index in range(count):
            branch_id = f"v127-{action}-{index}"
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
                    "selected_observation_digest": f"digest-{branch_id}",
                    "trainable_public_input": _trainable(action, seed=seed),
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


def _candidate_rows(
    manifest: dict[str, object],
    *,
    negative_action: str,
) -> list[dict[str, object]]:
    branch_id = str(manifest["branch_id"])
    seed = int(manifest["non_trainable_audit_metadata"]["seed"])
    repaired = str(manifest["repaired_action"])
    return [
        _candidate_row(branch_id, repaired, seed=seed, material_gain=repaired == "eat"),
        _candidate_row(branch_id, negative_action, seed=seed, material_gain=False),
    ]


def _candidate_row(
    branch_id: str,
    action: str,
    *,
    seed: int,
    material_gain: bool,
) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_row_v1",
        "archive_row_id": f"{branch_id}::action::{action}",
        "candidate_action": action,
        "oracle_rank": 1 if material_gain else 2,
        "resolution_legal": True,
        "observation_legal": True,
        "material_gain_label": material_gain,
        "trainable_public_input": _trainable(action, seed=seed),
        "provenance": {
            "branch_id": branch_id,
            "seed": seed,
            "source_kind": "synthetic_fixture_open_source",
            "source_path": "synthetic.jsonl.gz",
            "diagnostics_only": True,
        },
        "replay_verification_result": True,
        "replay_verification_digest": "a" * 64,
    }


def _trainable(action: str, *, seed: int) -> dict[str, object]:
    del seed
    return {
        "schema_version": (
            "mind_v3_first_recovery_branch_archive_trainable_public_input_v1"
        ),
        "candidate_action": action,
        "candidate_action_index": 1,
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
    }


def _v126_report(digest: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_shadow_scorer_execution_v1",
        "classification": {"primary": "shadow_scorer_execution_blocked_by_metrics"},
        "source_integrity": {
            "passed": True,
            "failures": [],
            "manifest_digest": digest,
            "manifest_digest_matches_v124": True,
            "manifest_digest_matches_v125": True,
        },
        "metric_gate": {
            "failures": [
                "positive_only_manifest_no_candidate_ranking_evidence",
                "candidate_action_label_echo_detected",
                "material_gain_exact_label_missing",
            ]
        },
        "recommendation": _recommendation(),
    }


def _v125_report(digest: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_shadow_scorer_proposal_v1",
        "classification": {"primary": "shadow_scorer_proposal_ready_for_review"},
        "source_integrity": {
            "passed": True,
            "failures": [],
            "manifest_digest": digest,
        },
        "recommendation": _recommendation(),
    }


def _v124_report(rows: list[dict[str, object]], digest: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_accepted_rare_attack_contract_v1",
        "classification": {
            "primary": "accepted_rare_attack_contract_ready_for_shadow_proposal"
        },
        "source_integrity": {"passed": True, "failures": []},
        "manifest": {"manifest_row_count": len(rows), "manifest_digest": digest},
        "contract_checks": {
            "passed": True,
            "failures": [],
            "repaired_action_counts": dict(EXPECTED_REPAIRED_ACTION_COUNTS),
        },
        "recommendation": _recommendation(),
    }


def _archive_report(rows: list[dict[str, object]], *, active: bool = False) -> dict[str, object]:
    report = {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "classification": {"primary": "active_coverage_rare_attacks_found" if active else "branch_archive_replay_verified"},
        "branch_archive_summary": {
            "archive_row_count": len(rows),
            "branch_result_count": len({row["provenance"]["branch_id"] for row in rows}),
            "heuristic_action_source_count": 0,
            "replay_verified": True,
            "archive_rows_sha256": stable_payload_digest(rows),
        },
        "recommendation": _recommendation(),
    }
    if active:
        report.update(
            {
                "replay_verified": True,
                "replay_verification": {
                    "replay_verified": True,
                    "missing_replay_verification_count": 0,
                    "replay_verification_skipped_count": 0,
                    "replay_verification_failure_count": 0,
                },
                "source_integrity": {"passed": True, "failures": []},
            }
        )
    return report


def _recommendation() -> dict[str, object]:
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
