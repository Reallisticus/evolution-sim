from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
    build_first_recovery_repaired_label_contract_audit,
    _branch_split,
    _contract_checks,
)
from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    build_first_recovery_tie_aware_label_repair,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryRepairedLabelContractAuditTests(unittest.TestCase):
    def test_repaired_label_contract_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-repaired-label-contract-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_repaired_label_contract_audit"
            ),
        )

    def test_manifest_has_exactly_one_row_per_branch(self) -> None:
        rows = _tied_rows(6)
        build = _build(rows)

        self.assertEqual(len(build.manifest_rows), 6)
        self.assertEqual(
            len({row["branch_id"] for row in build.manifest_rows}),
            6,
        )
        self.assertTrue(
            build.report["contract_checks"]["manifest_row_count_matches_branch_count"]
        )

    def test_unique_best_branch_cannot_be_changed(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=1),
            _row("b1", "eat", rank=2, terminal=0),
        ]
        build = _build(rows)
        row = build.manifest_rows[0]

        self.assertTrue(row["unique_objective_best"])
        self.assertFalse(row["changed"])
        self.assertEqual(row["current_oracle_action"], row["repaired_action"])
        self.assertEqual(build.report["contract_checks"]["unique_best_changed_count"], 0)

    def test_resolution_invalid_selected_row_fails_contract(self) -> None:
        build = _build(_tied_rows(1))
        manifest = [dict(build.manifest_rows[0], selected_resolution_legal=False)]

        checks = _contract_checks(
            manifest_rows=manifest,
            computed_policy_report=_computed_policy(build),
            v118_policy_report=_v118_policy(build),
            strict_source_verification=False,
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(checks["violation_counts"]["resolution_invalid_selected"], 1)

    def test_objective_equivalence_violation_fails_contract(self) -> None:
        build = _build(_tied_rows(1))
        manifest = [
            dict(build.manifest_rows[0], objective_equivalence_verified=False)
        ]

        checks = _contract_checks(
            manifest_rows=manifest,
            computed_policy_report=_computed_policy(build),
            v118_policy_report=_v118_policy(build),
            strict_source_verification=False,
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(checks["violation_counts"]["objective_equivalence_violation"], 1)

    def test_missing_trainable_public_input_fails_contract(self) -> None:
        build = _build(_tied_rows(1))
        row = dict(build.manifest_rows[0])
        row["trainable_public_input"] = {}

        checks = _contract_checks(
            manifest_rows=[row],
            computed_policy_report=_computed_policy(build),
            v118_policy_report=_v118_policy(build),
            strict_source_verification=False,
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(checks["trainable_public_input_missing_count"], 1)

    def test_forbidden_private_key_inside_trainable_input_fails_contract(self) -> None:
        build = _build(_tied_rows(1))
        row = copy.deepcopy(build.manifest_rows[0])
        row["trainable_public_input"]["branch_id"] = "b1"

        checks = _contract_checks(
            manifest_rows=[row],
            computed_policy_report=_computed_policy(build),
            v118_policy_report=_v118_policy(build),
            strict_source_verification=False,
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(checks["forbidden_trainable_key_count"], 1)

    def test_audit_metadata_is_separated_from_trainable_fields(self) -> None:
        build = _build(_tied_rows(1))
        row = build.manifest_rows[0]

        self.assertIn("non_trainable_audit_metadata", row)
        self.assertTrue(row["non_trainable_audit_metadata"]["non_trainable"])
        self.assertNotIn("non_trainable_audit_metadata", row["trainable_public_input"])
        self.assertEqual(
            build.report["contract_checks"][
                "audit_metadata_separation_violation_count"
            ],
            0,
        )

    def test_deterministic_split_is_stable(self) -> None:
        branch_ids = [f"branch-{index}" for index in range(32)]
        first = [_branch_split(branch_id) for branch_id in branch_ids]
        second = [_branch_split(branch_id) for branch_id in branch_ids]

        self.assertEqual(first, second)
        self.assertEqual(_branch_split("b000"), "train")
        self.assertEqual(_branch_split("b001"), "validation")
        self.assertEqual(_branch_split("b002"), "test")
        self.assertEqual(_branch_split("split-golden-2"), "train")
        self.assertEqual(_branch_split("split-golden-13"), "validation")
        self.assertEqual(_branch_split("split-golden-66"), "test")

    def test_split_support_classification_ready_and_limited(self) -> None:
        ready = _build(_tied_rows(120))
        limited = _build(_tied_rows(2))

        self.assertEqual(
            ready.report["classification"]["primary"],
            "repaired_label_contract_ready_for_shadow_scorer_proposal",
        )
        self.assertTrue(ready.report["split_support"]["support_adequate"])
        self.assertEqual(
            limited.report["classification"]["primary"],
            "repaired_label_contract_support_limited",
        )
        self.assertFalse(limited.report["split_support"]["support_adequate"])

    def test_real_v115_v118_path_matches_expected_counts_and_blocks_readiness(
        self,
    ) -> None:
        archive_report = (
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json"
        )
        archive_rows = (
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz"
        )
        v118_report = (
            ROOT
            / "output/mind/mind-v3-v118-first-recovery-tie-aware-label-repair.json"
        )
        if not archive_report.exists() or not archive_rows.exists() or not v118_report.exists():
            self.skipTest("real v115/v118 artifacts are not present")

        build = build_first_recovery_repaired_label_contract_audit(
            archive_report_path=archive_report,
            archive_rows_path=archive_rows,
            v118_report_path=v118_report,
        )
        checks = build.report["contract_checks"]
        split_support = build.report["split_support"]
        splits = split_support["splits"]
        recommendation = build.report["recommendation"]

        self.assertEqual(
            build.report["classification"]["primary"],
            "repaired_label_contract_support_limited",
        )
        self.assertEqual(len(build.manifest_rows), 106)
        self.assertEqual(
            checks["repaired_action_counts"],
            {
                "attack_east": 3,
                "attack_west": 3,
                "drink": 7,
                "eat": 17,
                "move_east": 16,
                "move_north": 15,
                "move_south": 15,
                "move_west": 14,
                "stay": 16,
            },
        )
        self.assertEqual(checks["total_violation_count"], 0)
        self.assertFalse(recommendation["downstream_shadow_scorer_allowed"])
        self.assertFalse(recommendation["v113_readiness_rerun_allowed"])
        self.assertFalse(recommendation["claim_causality"])
        self.assertEqual(
            split_support["warnings"],
            [
                "test_split_missing_action_classes",
                "test_split_too_small",
                "validation_split_missing_action_classes",
            ],
        )
        self.assertEqual(
            split_support["split_policy"]["method"],
            "sha256_branch_id_mod_100",
        )
        self.assertEqual(split_support["split_policy"]["train"], "bucket < 70")
        self.assertEqual(
            split_support["split_policy"]["validation"],
            "70 <= bucket < 85",
        )
        self.assertEqual(split_support["split_policy"]["test"], "bucket >= 85")
        self.assertEqual(splits["train"]["branch_count"], 86)
        self.assertTrue(splits["train"]["support_adequate"])
        self.assertEqual(splits["train"]["missing_action_classes"], [])
        self.assertEqual(splits["train"]["minimum_per_action_support"], 2)
        self.assertEqual(splits["validation"]["branch_count"], 12)
        self.assertFalse(splits["validation"]["support_adequate"])
        self.assertEqual(
            splits["validation"]["missing_action_classes"],
            ["attack_west", "move_west"],
        )
        self.assertEqual(splits["validation"]["minimum_per_action_support"], 0)
        self.assertEqual(splits["test"]["branch_count"], 8)
        self.assertFalse(splits["test"]["support_adequate"])
        self.assertEqual(
            splits["test"]["missing_action_classes"],
            ["attack_east", "drink"],
        )
        self.assertEqual(splits["test"]["minimum_per_action_support"], 0)
        self.assertTrue(splits["test"]["too_small_for_shadow_scorer_evaluation"])


def _build(rows: list[dict[str, object]]):
    v118 = build_first_recovery_tie_aware_label_repair(
        archive_report=_archive_report(rows),
        archive_rows=copy.deepcopy(rows),
        strict_source_verification=False,
    ).report
    return build_first_recovery_repaired_label_contract_audit(
        archive_report=_archive_report(rows),
        archive_rows=copy.deepcopy(rows),
        v118_report=v118,
        strict_source_verification=False,
    )


def _computed_policy(build) -> dict[str, object]:
    return build.report["reconstruction"]["computed_policy_report"]


def _v118_policy(build) -> dict[str, object]:
    return build.report["v118_reference"] | {
        "repaired_action_counts": build.report["v118_reference"][
            "policy_repaired_action_counts"
        ]
    }


def _tied_rows(branch_count: int) -> list[dict[str, object]]:
    rows = []
    for index in range(branch_count):
        branch_id = f"b{index:03d}"
        rows.append(_row(branch_id, "stay", rank=1, terminal=0))
        rows.append(_row(branch_id, "eat", rank=2, terminal=0))
    return rows


def _archive_report(rows: list[dict[str, object]]) -> dict[str, object]:
    branch_ids = {
        row["provenance"]["branch_id"]
        for row in rows
        if row.get("oracle_rank") == 1
    }
    action_counts = Counter(
        str(row.get("candidate_action"))
        for row in rows
        if row.get("oracle_rank") == 1
    )
    dominant_action, dominant_count = (
        sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if action_counts
        else (None, 0)
    )
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "row_reconstruction": {
            "reconstructed_first_recovery_row_count": len(branch_ids),
            "expected_constructible_first_recovery_row_count": len(branch_ids),
            "row_count_matches_v107": True,
            "row_count_matches_v108": True,
        },
        "target_selection": {
            "selected_target_count": len(branch_ids),
            "skipped_target_count": 0,
        },
        "branch_archive_summary": {
            "selected_row_count": len(branch_ids),
            "materialized_branch_point_count": len(branch_ids),
            "materialization_failure_count": 0,
            "branch_result_count": len(branch_ids),
            "action_run_count": len(rows),
            "archive_row_count": len(rows),
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "full_reconstruction_row_count": len(branch_ids),
        },
        "oracle_label_summary": {
            "oracle_action_counts": dict(sorted(action_counts.items())),
            "dominant_oracle_action": dominant_action,
            "dominant_oracle_action_count": dominant_count,
            "dominant_oracle_action_share": (
                round(dominant_count / len(branch_ids), 6) if branch_ids else 0
            ),
        },
        "legality_summary": {
            "resolution_invalid_count": sum(
                1 for row in rows if row.get("resolution_legal") is False
            )
        },
        "research_recommendation": {
            "recommendation": (
                "do_not_start_v110_until_archive_support_blockers_are_resolved"
            )
        },
    }


def _row(
    branch_id: str,
    action: str,
    *,
    rank: int,
    terminal: int,
    birth: int = 0,
    recovery: float = 0.0,
    death_reduction: int = 0,
    resolution_legal: bool = True,
    material: bool = False,
) -> dict[str, object]:
    return {
        "archive_row_id": f"{branch_id}::action::{action}",
        "candidate_action": action,
        "oracle_rank": rank,
        "terminal_alive_delta": terminal,
        "birth_delta": birth,
        "death_reduction_delta": death_reduction,
        "death_delta": -death_reduction,
        "recovery_vitals_deltas": {
            "target_recovery_score_delta": recovery,
        },
        "biology_homeostasis_labels": {
            "target_alive": terminal > 0,
        },
        "resolution_legal": resolution_legal,
        "material_gain_label": material,
        "observation_digest": f"digest-{branch_id}-{action}",
        "source_kind": "fixture_carrion_only",
        "source_path": f"/tmp/{branch_id}.jsonl.gz",
        "seed": 13,
        "tick": 1,
        "agent_id": 9,
        "record_index": 2,
        "trainable_public_input": {
            "candidate_action": action,
            "target_public_state_before": {"alive": True},
        },
        "provenance": {
            "branch_id": branch_id,
            "logged_action": "stay",
            "source_kind": "fixture_carrion_only",
            "seed": 13,
        },
    }


if __name__ == "__main__":
    unittest.main()
