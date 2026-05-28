from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
    build_first_recovery_tie_aware_label_repair,
    _branch_repair_contexts,
    _summarize_policy_report,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryTieAwareLabelRepairTests(unittest.TestCase):
    def test_tie_aware_label_repair_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-tie-aware-label-repair"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_tie_aware_label_repair"
            ),
        )

    def test_unique_objective_best_branch_is_never_changed(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=1),
            _row("b1", "eat", rank=2, terminal=0),
        ]
        build = _build(rows)
        policies = build.report["tie_aware_label_repair"]["policies"]

        for report in policies.values():
            self.assertEqual(report["unique_best_branches_changed_count"], 0)
            self.assertEqual(report["changed_branch_count"], 0)
            self.assertEqual(report["repaired_action_counts"], {"stay": 1})

    def test_tied_branch_can_repair_to_legal_non_stay_candidate(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0),
        ]
        build = _build(rows)
        report = build.report["tie_aware_label_repair"]["policies"][
            "prefer_non_stay_resolution_legal"
        ]

        self.assertTrue(report["policy_valid"])
        self.assertEqual(report["repaired_action_counts"], {"eat": 1})
        self.assertEqual(report["changed_branch_count"], 1)
        self.assertEqual(report["tied_branches_changed_count"], 1)
        self.assertEqual(report["resolution_invalid_selected_count"], 0)
        self.assertEqual(report["objective_equivalence_violation_count"], 0)

    def test_resolution_invalid_tied_candidate_is_never_selected(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0, resolution_legal=True),
            _row("b1", "eat", rank=2, terminal=0, resolution_legal=False),
        ]
        build = _build(rows)
        policies = build.report["tie_aware_label_repair"]["policies"]

        for report in policies.values():
            self.assertTrue(report["policy_valid"])
            self.assertEqual(report["resolution_invalid_selected_count"], 0)
            self.assertEqual(report["repaired_action_counts"], {"stay": 1})

    def test_objective_equivalence_violation_is_detected(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0),
            _row("b1", "drink", rank=3, terminal=1),
        ]
        contexts, failures, _examples = _branch_repair_contexts(rows)
        self.assertEqual(failures, [])
        drink = next(row for row in rows if row["candidate_action"] == "drink")

        report = _summarize_policy_report(
            "test_policy",
            contexts,
            {"b1": drink},
        )

        self.assertFalse(report["policy_valid"])
        self.assertEqual(report["objective_equivalence_violation_count"], 1)

    def test_action_balance_policy_is_deterministic(self) -> None:
        rows = _four_tied_stay_branches()
        first = _build(rows).report["tie_aware_label_repair"]["policies"][
            "action_balance_resolution_legal"
        ]
        second = _build(rows).report["tie_aware_label_repair"]["policies"][
            "action_balance_resolution_legal"
        ]

        self.assertEqual(first, second)

    def test_classification_clears_collapse_when_valid_repair_distribution_clears(
        self,
    ) -> None:
        build = _build(_four_tied_stay_branches())
        repair = build.report["tie_aware_label_repair"]

        self.assertEqual(
            build.report["classification"]["primary"],
            "tie_aware_repair_clears_action_collapse",
        )
        self.assertEqual(
            repair["best_clearing_policy"],
            "action_balance_resolution_legal",
        )
        best = repair["policies"][repair["best_clearing_policy"]]
        self.assertLessEqual(best["dominant_action_share"], 0.5)
        self.assertEqual(best["objective_equivalence_violation_count"], 0)
        self.assertEqual(best["resolution_invalid_selected_count"], 0)
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_classification_remains_collapsed_when_all_valid_policies_are_dominated(
        self,
    ) -> None:
        rows = []
        for index in range(4):
            branch_id = f"b{index}"
            rows.append(_row(branch_id, "stay", rank=1, terminal=1))
            rows.append(_row(branch_id, "eat", rank=2, terminal=0))

        build = _build(rows)

        self.assertEqual(
            build.report["classification"]["primary"],
            "tie_aware_repair_still_action_collapsed",
        )
        self.assertEqual(build.report["tie_aware_label_repair"]["clearing_policy_names"], [])

    def test_malformed_objective_fields_block_through_v117_integrity(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0),
        ]
        del rows[0]["terminal_alive_delta"]

        build = _build(rows)

        self.assertFalse(build.report["objective_input_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "tie_aware_repair_inconclusive",
        )
        self.assertIn(
            "objective_input_field_integrity",
            build.report["classification"]["labels"],
        )
        self.assertTrue(build.report["tie_aware_label_repair"]["analysis_blocked"])

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        build = _build(_four_tied_stay_branches())
        contract = build.report["contract"]

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
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


def _build(rows: list[dict[str, object]]):
    return build_first_recovery_tie_aware_label_repair(
        archive_report=_archive_report(rows),
        archive_rows=copy.deepcopy(rows),
        strict_source_verification=False,
    )


def _four_tied_stay_branches() -> list[dict[str, object]]:
    rows = []
    for index in range(4):
        branch_id = f"b{index}"
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
