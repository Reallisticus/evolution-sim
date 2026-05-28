from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
    build_first_recovery_oracle_tie_break_audit,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryOracleTieBreakAuditTests(unittest.TestCase):
    def test_oracle_tie_break_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-oracle-tie-break-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_oracle_tie_break_audit"
            ),
        )

    def test_unique_best_candidate_is_counted(self) -> None:
        rows = [
            _row("b1", "stay", rank=2, terminal=0),
            _row("b1", "eat", rank=1, terminal=1, material=True),
        ]
        report = _archive_report(rows)
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=report,
            archive_rows=rows,
            strict_source_verification=False,
        )
        analysis = build.report["objective_tie_break_analysis"]

        self.assertEqual(analysis["unique_objective_best_branch_count"], 1)
        self.assertEqual(analysis["multiple_objective_best_branch_count"], 0)
        self.assertEqual(
            analysis["unique_objective_best_action_counts"],
            {"eat": 1},
        )

    def test_tied_objective_candidates_are_tie_neutralized(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0, material=True),
        ]
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        )
        analysis = build.report["objective_tie_break_analysis"]

        self.assertEqual(analysis["multiple_objective_best_branch_count"], 1)
        self.assertEqual(analysis["rank1_tie_size_distribution"], {"2": 1})
        self.assertEqual(
            analysis["tie_neutral_alternatives"][
                "prefer_non_stay_resolution_legal"
            ]["action_counts"],
            {"eat": 1},
        )
        self.assertEqual(
            analysis["tie_neutral_alternatives"][
                "prefer_material_gain_resolution_legal"
            ]["action_counts"],
            {"eat": 1},
        )

    def test_resolution_invalid_candidates_are_not_executable_recommendations(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0, resolution_legal=True),
            _row("b1", "move_east", rank=2, terminal=0, resolution_legal=False),
        ]
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        )
        analysis = build.report["objective_tie_break_analysis"]

        self.assertEqual(
            analysis["resolution_separation"][
                "rank1_equivalent_candidate_resolution_counts"
            ],
            {"resolution_invalid": 1, "resolution_legal": 1},
        )
        self.assertEqual(
            analysis["tie_neutral_alternatives"][
                "prefer_non_stay_resolution_legal"
            ]["action_counts"],
            {"stay": 1},
        )
        self.assertEqual(
            analysis["tie_neutral_alternatives"]["action_name"][
                "selected_resolution_invalid_count"
            ],
            1,
        )

    def test_deterministic_hash_tie_break_is_stable(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0),
            _row("b1", "drink", rank=3, terminal=0),
        ]
        first = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        ).report
        second = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        ).report

        self.assertEqual(
            first["objective_tie_break_analysis"]["tie_neutral_alternatives"][
                "deterministic_hash"
            ],
            second["objective_tie_break_analysis"]["tie_neutral_alternatives"][
                "deterministic_hash"
            ],
        )

    def test_classifies_tie_break_artifact(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0),
            _row("b2", "stay", rank=1, terminal=0),
            _row("b2", "drink", rank=2, terminal=0),
        ]
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        )

        self.assertEqual(
            build.report["classification"]["primary"],
            "tie_break_artifact_likely",
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_classifies_objective_support_stay_dominance(self) -> None:
        rows = [
            _row("b1", "stay", rank=1, terminal=1),
            _row("b1", "eat", rank=2, terminal=0),
            _row("b2", "stay", rank=1, terminal=1),
            _row("b2", "drink", rank=2, terminal=0),
            _row("b3", "stay", rank=1, terminal=1),
            _row("b3", "move_east", rank=2, terminal=0),
            _row("b4", "eat", rank=1, terminal=1),
            _row("b4", "stay", rank=2, terminal=0),
        ]
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        )

        self.assertEqual(
            build.report["classification"]["primary"],
            "objective_support_stay_dominance",
        )

    def test_missing_or_malformed_inputs_are_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = build_first_recovery_oracle_tie_break_audit(
                archive_report_path=root / "missing.json",
                archive_rows_path=root / "missing.jsonl.gz",
            ).report

        self.assertEqual(
            missing["classification"]["primary"],
            "missing_evidence_inconclusive",
        )
        malformed = build_first_recovery_oracle_tie_break_audit(
            archive_report={"schema_version": "wrong"},
            archive_rows=[],
            strict_source_verification=False,
        ).report
        self.assertEqual(
            malformed["classification"]["primary"],
            "missing_evidence_inconclusive",
        )

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        rows = [_row("b1", "stay", rank=1, terminal=0)]
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        )

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
        )
        contract = build.report["contract"]
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertFalse(contract["observation_field_change"])
        self.assertEqual(contract["viewer_effect"], "none")
        self.assertFalse(contract["v113_readiness_rerun_allowed"])
        self.assertTrue(build.report["non_promoted"])

    def test_missing_terminal_alive_delta_fails_objective_input_integrity(self) -> None:
        rows = _objective_integrity_rows()
        del rows[0]["terminal_alive_delta"]

        self._assert_objective_input_integrity_failure(
            rows,
            count_field="missing_field_count",
        )

    def test_non_numeric_terminal_alive_delta_fails_objective_input_integrity(
        self,
    ) -> None:
        rows = _objective_integrity_rows()
        rows[0]["terminal_alive_delta"] = "0"

        self._assert_objective_input_integrity_failure(
            rows,
            count_field="malformed_field_count",
        )

    def test_missing_recovery_score_delta_fails_objective_input_integrity(self) -> None:
        rows = _objective_integrity_rows()
        del rows[0]["recovery_vitals_deltas"]["target_recovery_score_delta"]

        self._assert_objective_input_integrity_failure(
            rows,
            count_field="missing_field_count",
        )

    def test_non_finite_objective_value_fails_objective_input_integrity(self) -> None:
        rows = _objective_integrity_rows()
        rows[0]["death_reduction_delta"] = float("inf")

        self._assert_objective_input_integrity_failure(
            rows,
            count_field="non_finite_numeric_value_count",
        )

    def test_missing_or_non_bool_target_alive_fails_objective_input_integrity(
        self,
    ) -> None:
        cases = {
            "missing": lambda row: row["biology_homeostasis_labels"].pop(
                "target_alive"
            ),
            "non_bool": lambda row: row["biology_homeostasis_labels"].__setitem__(
                "target_alive",
                1,
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                rows = _objective_integrity_rows()
                mutate(rows[0])
                self._assert_objective_input_integrity_failure(
                    rows,
                    count_field=(
                        "missing_field_count"
                        if name == "missing"
                        else "malformed_field_count"
                    ),
                )

    def test_real_v115_objective_input_integrity_preserves_v117_conclusion(
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
        if not archive_report.exists() or not archive_rows.exists():
            self.skipTest("real v115 archive artifacts are not present")

        build = build_first_recovery_oracle_tie_break_audit(
            archive_report_path=archive_report,
            archive_rows_path=archive_rows,
        )
        integrity = build.report["objective_input_integrity"]
        analysis = build.report["objective_tie_break_analysis"]

        self.assertTrue(integrity["passed"])
        self.assertEqual(integrity["total_failure_count"], 0)
        self.assertEqual(integrity["failed_row_count"], 0)
        self.assertEqual(
            build.report["classification"]["primary"],
            "tie_break_artifact_likely",
        )
        self.assertEqual(analysis["current_serialized_oracle_stay_count"], 64)
        self.assertEqual(
            analysis["tie_neutral_stay_counts"],
            {
                "action_name": 7,
                "prefer_non_stay_resolution_legal": 7,
                "prefer_material_gain_resolution_legal": 7,
                "deterministic_hash": 21,
            },
        )

    def _assert_objective_input_integrity_failure(
        self,
        rows: list[dict[str, object]],
        *,
        count_field: str,
    ) -> None:
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report=_archive_report(rows),
            archive_rows=rows,
            strict_source_verification=False,
        )
        integrity = build.report["objective_input_integrity"]

        self.assertFalse(integrity["passed"])
        self.assertGreater(integrity[count_field], 0)
        self.assertIn(
            "objective_input_field_integrity",
            integrity["integrity_failures"],
        )
        self.assertIn(
            "objective_input_field_integrity",
            build.report["classification"]["labels"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "objective_tie_break_inconclusive",
        )
        self.assertTrue(
            build.report["objective_tie_break_analysis"]["analysis_blocked"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )


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


def _objective_integrity_rows() -> list[dict[str, object]]:
    return copy.deepcopy(
        [
            _row("b1", "stay", rank=1, terminal=0),
            _row("b1", "eat", rank=2, terminal=0),
        ]
    )


if __name__ == "__main__":
    unittest.main()
