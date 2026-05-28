from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_rare_attack_coverage_collection import (
    MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
    build_first_recovery_rare_attack_coverage_collection,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryRareAttackCoverageCollectionTests(unittest.TestCase):
    def test_rare_attack_coverage_collection_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-rare-attack-coverage-collection"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_rare_attack_coverage_collection"
            ),
        )

    def test_source_integrity_passes_for_clean_prerequisites(self) -> None:
        build = _build(candidate_rows=[])

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(build.report["source_integrity"]["failures"], [])
        self.assertEqual(
            build.report["source_integrity"]["current_rare_support"],
            {"attack_east": 3, "attack_west": 3},
        )
        self.assertEqual(
            build.report["source_integrity"]["v121_valid_candidate_counts"],
            {"attack_east": 0, "attack_west": 0},
        )

    def test_candidate_source_domain_counts_manifested_and_unmanifested_branches(
        self,
    ) -> None:
        rows = _candidate_branch("existing-east-0", "attack_east")
        rows += _candidate_branch("new-west", "attack_west")

        build = _build(candidate_rows=rows)
        domain = build.report["candidate_collection"]["candidate_source_domain"]

        self.assertEqual(domain["candidate_branch_count"], 2)
        self.assertEqual(domain["manifest_branch_count"], 6)
        self.assertEqual(domain["already_manifested_branch_count"], 1)
        self.assertEqual(domain["unmanifested_candidate_branch_count"], 1)
        self.assertEqual(
            domain["answer"],
            "candidate_source_contains_unmanifested_branches",
        )

    def test_candidate_source_with_no_unmanifested_branches_is_labeled(self) -> None:
        rows = (
            _candidate_branch("existing-east-0", "attack_east")
            + _candidate_branch("existing-west-0", "attack_west")
        )

        build = _build(candidate_rows=rows)
        domain = build.report["candidate_collection"]["candidate_source_domain"]

        self.assertEqual(domain["candidate_branch_count"], 2)
        self.assertEqual(domain["already_manifested_branch_count"], 2)
        self.assertEqual(domain["unmanifested_candidate_branch_count"], 0)
        self.assertIn(
            "candidate_source_contains_no_unmanifested_branches",
            build.report["classification"]["labels"],
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "use_active_coverage_archive_with_unmanifested_branches",
        )

    def test_malformed_prerequisite_reports_fail_source_integrity(self) -> None:
        reports = _reports(_manifest_rows())
        reports["v119"]["classification"]["primary"] = "bad"
        reports["v120"]["source_integrity"]["failures"] = ["tampered"]
        reports["v121"]["recommendation"]["claim_causality"] = True

        build = _build(reports=reports, candidate_rows=[])
        failures = build.report["source_integrity"]["failures"]

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn("v119_classification_unexpected", failures)
        self.assertIn("v120_source_failures_not_empty_or_malformed", failures)
        self.assertIn("v121_claim_causality_not_false", failures)
        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_attack_coverage_source_integrity_failed",
        )

    def test_candidate_acceptance_for_both_rare_attacks(self) -> None:
        candidate_rows = (
            _candidate_branch("new-east", "attack_east")
            + _candidate_branch("new-west", "attack_west")
        )

        build = _build(candidate_rows=candidate_rows)

        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_attack_coverage_candidates_found",
        )
        self.assertEqual(
            build.report["found_candidate_counts"],
            {"attack_east": 1, "attack_west": 1},
        )
        self.assertTrue(
            build.report["stricter_split_support_if_accepted"][
                "train2_validation1_test1_would_be_feasible_for_targets"
            ]
        )
        self.assertEqual(len(build.candidate_rows), 2)
        self.assertFalse(
            build.candidate_rows[0]["trainable_public_input_contents_exposed"]
        )

    def test_resolution_illegal_candidate_is_rejected(self) -> None:
        rows = _candidate_branch("new-east", "attack_east", resolution_legal=False)

        build = _build(candidate_rows=rows)

        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_attack_coverage_not_found_within_budget",
        )
        self.assertEqual(build.report["found_candidate_counts"]["attack_east"], 0)
        self.assertEqual(
            build.report["rejection_counts"]["attack_east"]["resolution_illegal"],
            1,
        )

    def test_objective_equivalence_violation_is_rejected(self) -> None:
        rows = _candidate_branch(
            "new-east",
            "attack_east",
            target_terminal_delta=1,
        )

        build = _build(candidate_rows=rows)

        self.assertEqual(build.report["found_candidate_counts"]["attack_east"], 0)
        self.assertEqual(
            build.report["rejection_counts"]["attack_east"][
                "objective_equivalence_violation"
            ],
            1,
        )

    def test_trainable_metadata_leakage_is_rejected(self) -> None:
        rows = _candidate_branch("new-east", "attack_east")
        for row in rows:
            if row["candidate_action"] == "attack_east":
                row["trainable_public_input"]["seed"] = 13

        build = _build(candidate_rows=rows)

        self.assertEqual(build.report["found_candidate_counts"]["attack_east"], 0)
        self.assertEqual(
            build.report["rejection_counts"]["attack_east"][
                "trainable_metadata_leakage"
            ],
            1,
        )

    def test_missing_or_malformed_branch_identity_is_rejected(self) -> None:
        rows = _candidate_branch("new-east", "attack_east")
        for row in rows:
            if row["candidate_action"] == "attack_east":
                row["archive_row_id"] = "wrong::action::attack_east"

        build = _build(candidate_rows=rows)

        self.assertEqual(build.report["found_candidate_counts"]["attack_east"], 0)
        self.assertEqual(
            build.report["rejection_counts"]["attack_east"][
                "branch_identity_missing_or_malformed"
            ],
            1,
        )

    def test_budget_exhaustion_classifies_not_found(self) -> None:
        rows = _candidate_branch("new-east", "attack_east")

        build = _build(candidate_rows=rows, max_branches=0)

        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_attack_coverage_not_found_within_budget",
        )
        self.assertTrue(
            build.report["search_budget_used"]["budget_exhausted"]
        )
        self.assertEqual(build.report["search_budget_used"]["branches_evaluated"], 0)

    def test_partial_coverage_classification(self) -> None:
        rows = _candidate_branch("new-east", "attack_east")

        build = _build(candidate_rows=rows)

        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_attack_coverage_partial",
        )
        self.assertEqual(
            build.report["found_candidate_counts"],
            {"attack_east": 1, "attack_west": 0},
        )

    def test_real_reports_default_to_no_candidates_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v119-first-recovery-repaired-label-contract-audit.json",
            ROOT / "output/mind/mind-v3-v119-first-recovery-repaired-label-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v120-first-recovery-repaired-label-split-support-feasibility.json",
            ROOT / "output/mind/mind-v3-v121-first-recovery-rare-action-coverage-targeting.json",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115/v119/v120/v121 artifacts are not present")

        build = build_first_recovery_rare_attack_coverage_collection(
            v119_report_path=paths[0],
            manifest_path=paths[1],
            v120_report_path=paths[2],
            v121_report_path=paths[3],
            candidate_archive_report_path=paths[4],
            candidate_archive_rows_path=paths[5],
        )

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
        )
        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "rare_attack_coverage_not_found_within_budget",
        )
        self.assertEqual(
            build.report["found_candidate_counts"],
            {"attack_east": 0, "attack_west": 0},
        )
        domain = build.report["candidate_collection"]["candidate_source_domain"]
        self.assertEqual(domain["candidate_branch_count"], 106)
        self.assertEqual(domain["manifest_branch_count"], 106)
        self.assertEqual(domain["already_manifested_branch_count"], 106)
        self.assertEqual(domain["unmanifested_candidate_branch_count"], 0)
        self.assertIn(
            "candidate_source_contains_no_unmanifested_branches",
            build.report["classification"]["labels"],
        )
        self.assertEqual(
            build.report["recommendation"]["next_step"],
            "use_active_coverage_archive_with_unmanifested_branches",
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )


def _build(
    *,
    reports: dict[str, dict[str, object]] | None = None,
    candidate_rows: list[dict[str, object]],
    max_branches: int = 106,
):
    manifest = _manifest_rows()
    payloads = reports if reports is not None else _reports(manifest)
    return build_first_recovery_rare_attack_coverage_collection(
        v119_report=payloads["v119"],
        manifest_rows=copy.deepcopy(manifest),
        v120_report=payloads["v120"],
        v121_report=payloads["v121"],
        candidate_archive_report={
            "schema_version": "mind_v3_first_recovery_branch_archive_v1",
            "branch_archive_summary": {"archive_row_count": len(candidate_rows)},
        },
        candidate_archive_rows=copy.deepcopy(candidate_rows),
        max_branches=max_branches,
    )


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.append(_manifest_row(f"existing-east-{index}", "attack_east"))
        rows.append(_manifest_row(f"existing-west-{index}", "attack_west"))
    return rows


def _manifest_row(branch_id: str, action: str) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "repaired_action": action,
        "trainable_public_input": {
            "candidate_action": action,
            "target_public_state_before": {"alive": True},
        },
    }


def _reports(manifest_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    digest = stable_payload_digest(list(manifest_rows))
    return {
        "v119": {
            "schema_version": "mind_v3_first_recovery_repaired_label_contract_audit_v1",
            "classification": {"primary": "repaired_label_contract_support_limited"},
            "manifest": {"manifest_digest": digest},
            "contract_checks": {"passed": True, "total_violation_count": 0},
            "recommendation": _blocked_recommendation(),
        },
        "v120": {
            "schema_version": "mind_v3_first_recovery_repaired_label_split_support_feasibility_v1",
            "classification": {
                "primary": "split_support_feasibility_limited_by_rare_actions"
            },
            "source_integrity": {
                "passed": True,
                "failures": [],
                "reported_manifest_digest": digest,
                "computed_manifest_digest": digest,
            },
            "scarcity_analysis": {
                "rare_action_additional_needed_for_train2_validation1_test1": {
                    "attack_east": 1,
                    "attack_west": 1,
                }
            },
            "recommendation": _blocked_recommendation(),
        },
        "v121": {
            "schema_version": "mind_v3_first_recovery_rare_action_coverage_targeting_v1",
            "classification": {
                "primary": "rare_action_coverage_not_available_in_existing_archive"
            },
            "source_integrity": {
                "passed": True,
                "failures": [],
                "manifest_digest": digest,
            },
            "current_rare_action_support": {
                "repaired_action_counts": {
                    "attack_east": 3,
                    "attack_west": 3,
                }
            },
            "candidate_search": {
                "per_action": {
                    "attack_east": {"valid_candidate_count": 0},
                    "attack_west": {"valid_candidate_count": 0},
                }
            },
            "recommendation": _blocked_recommendation(),
        },
    }


def _blocked_recommendation() -> dict[str, object]:
    return {
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
    }


def _candidate_branch(
    branch_id: str,
    target_action: str,
    *,
    resolution_legal: bool = True,
    target_terminal_delta: int = 0,
) -> list[dict[str, object]]:
    return [
        _archive_row(branch_id, "stay", rank=1, terminal_delta=0),
        _archive_row(
            branch_id,
            target_action,
            rank=2,
            terminal_delta=target_terminal_delta,
            resolution_legal=resolution_legal,
        ),
    ]


def _archive_row(
    branch_id: str,
    action: str,
    *,
    rank: int,
    terminal_delta: int,
    resolution_legal: bool = True,
) -> dict[str, object]:
    return {
        "archive_row_id": f"{branch_id}::action::{action}",
        "candidate_action": action,
        "oracle_rank": rank,
        "terminal_alive_delta": terminal_delta,
        "birth_delta": 0,
        "death_reduction_delta": 0,
        "recovery_vitals_deltas": {
            "target_recovery_score_delta": 0.0,
        },
        "biology_homeostasis_labels": {
            "target_alive": False,
        },
        "resolution_legal": resolution_legal,
        "trainable_public_input": {
            "candidate_action": action,
            "target_public_state_before": {"alive": True},
        },
        "provenance": {
            "branch_id": branch_id,
            "seed": 999,
        },
    }

if __name__ == "__main__":
    unittest.main()
