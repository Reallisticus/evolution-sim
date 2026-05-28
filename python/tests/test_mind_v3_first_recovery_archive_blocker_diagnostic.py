from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION,
    build_first_recovery_archive_blocker_diagnostic,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH_SIZE_DISTRIBUTION = {"3": 10, "4": 24, "5": 32, "6": 36, "7": 4}


class MindV3FirstRecoveryArchiveBlockerDiagnosticTests(unittest.TestCase):
    def test_archive_blocker_diagnostic_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-archive-blocker-diagnostic"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_archive_blocker_diagnostic"
            ),
        )

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        archive_report, rows = _archive_inputs()
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION,
        )
        self.assertEqual(
            list(build.report.keys()),
            [
                "schema_version",
                "audit_policy",
                "contract",
                "source_reports",
                "source_archive_verification",
                "stay_dominance_analysis",
                "resolution_invalid_analysis",
                "classification",
                "recommendation",
                "non_promoted",
            ],
        )
        contract = build.report["contract"]
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertEqual(contract["replay_golden_effect"], "none")
        self.assertEqual(contract["summary_only_effect"], "none")
        self.assertFalse(contract["observation_field_change"])
        self.assertEqual(contract["viewer_effect"], "none")
        self.assertTrue(build.report["non_promoted"])

    def test_stay_dominance_blocks_downstream_use_with_synthetic_archive(self) -> None:
        archive_report, rows = _archive_inputs(stay_best_count=64)
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        stay = build.report["stay_dominance_analysis"]
        self.assertTrue(stay["stay_dominance_detected"])
        self.assertEqual(stay["stay_branch_count"], 64)
        self.assertEqual(stay["dominant_oracle_action"], "stay")
        self.assertEqual(stay["dominant_oracle_action_share"], 0.603774)
        self.assertTrue(stay["data_support_artifact_not_primary"])
        self.assertIn(
            "stay_oracle_dominance_detected",
            build.report["classification"]["labels"],
        )
        self.assertFalse(
            build.report["recommendation"]["v115_usable_for_downstream_shadow_scorer"]
        )
        self.assertFalse(build.report["recommendation"]["v113_readiness_rerun_allowed"])

    def test_non_dominant_synthetic_archive_does_not_claim_stay_collapse(self) -> None:
        archive_report, rows = _archive_inputs(stay_best_count=50)
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        stay = build.report["stay_dominance_analysis"]
        self.assertFalse(stay["stay_dominance_detected"])
        self.assertNotIn(
            "stay_oracle_dominance_detected",
            build.report["classification"]["labels"],
        )

    def test_clear_synthetic_archive_allows_shadow_but_not_v113_readiness(self) -> None:
        archive_report, rows = _archive_inputs(
            stay_best_count=50,
            invalid_mode="none",
        )
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        labels = build.report["classification"]["labels"]
        self.assertTrue(build.report["source_archive_verification"]["verification_passed"])
        self.assertIn("downstream_shadow_scorer_allowed", labels)
        self.assertIn("readiness_rerun_blocked", labels)
        self.assertNotIn("readiness_rerun_allowed", labels)
        self.assertTrue(
            build.report["recommendation"]["v115_usable_for_downstream_shadow_scorer"]
        )
        self.assertFalse(build.report["recommendation"]["v113_readiness_rerun_allowed"])

    def test_resolution_invalid_rows_are_classified_expected_or_unexplained(self) -> None:
        archive_report, rows = _archive_inputs(invalid_mode="mixed")
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        invalid = build.report["resolution_invalid_analysis"]
        self.assertEqual(invalid["invalid_count"], 4)
        self.assertEqual(invalid["expected_public_drift_count"], 1)
        self.assertEqual(invalid["unexplained_count"], 3)
        self.assertEqual(invalid["answer"], "resolution_invalid_unexplained")
        self.assertIn(
            "resolution_invalid_unexplained",
            build.report["classification"]["labels"],
        )

    def test_leakage_guard_blocks_identity_trainable_fields(self) -> None:
        archive_report, rows = _archive_inputs()
        rows[0]["trainable_public_input"]["seed"] = 13
        rows[0]["trainable_public_input"]["private_world_state"] = {"x": 1}
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        self.assertTrue(verification["trainable_leakage_detected"])
        self.assertGreater(verification["trainable_leak_count"], 0)
        self.assertEqual(
            build.report["classification"]["primary"],
            "trainable_signal_leakage_detected",
        )
        self.assertFalse(
            build.report["recommendation"]["v115_usable_for_downstream_shadow_scorer"]
        )

    def test_duplicate_candidate_or_missing_logged_action_fails_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        rows[0]["candidate_action"] = "move_north"
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        branch_shape = verification["branch_shape"]
        self.assertFalse(verification["verification_passed"])
        self.assertGreater(branch_shape["duplicate_candidate_action_branch_count"], 0)
        self.assertGreater(branch_shape["missing_logged_action_branch_count"], 0)
        self.assertIn(
            "no_duplicate_candidate_actions_within_branch",
            verification["integrity_failures"],
        )
        self.assertIn(
            "logged_action_present_in_each_branch",
            verification["integrity_failures"],
        )

    def test_wrong_branch_size_distribution_fails_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        first_branch = rows[0]["provenance"]["branch_id"]
        destination = next(
            row["provenance"]["branch_id"]
            for row in rows
            if row["provenance"]["branch_id"] != first_branch
            and row["candidate_action"] not in {"stay", "eat", "move_north"}
        )
        rows[0]["provenance"]["branch_id"] = destination
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        self.assertFalse(verification["verification_passed"])
        self.assertFalse(
            verification["branch_shape"]["branch_size_distribution_matches_expected"]
        )
        self.assertIn(
            "expected_branch_size_distribution",
            verification["integrity_failures"],
        )

    def test_missing_branch_id_row_fails_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        rows[0]["provenance"].pop("branch_id")
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        self.assertFalse(verification["verification_passed"])
        self.assertEqual(
            verification["branch_shape"]["missing_branch_id_row_count"],
            1,
        )
        self.assertIn("no_missing_branch_id_rows", verification["integrity_failures"])

    def test_zero_rank1_row_in_branch_fails_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        branch_id = rows[0]["provenance"]["branch_id"]
        for row in rows:
            if row["provenance"]["branch_id"] == branch_id and row["oracle_rank"] == 1:
                row["oracle_rank"] = 2
                break
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        self.assertFalse(verification["verification_passed"])
        self.assertEqual(verification["branch_shape"]["no_rank1_branch_count"], 1)
        self.assertIn(
            "exactly_one_rank1_row_per_branch",
            verification["integrity_failures"],
        )

    def test_multiple_rank1_rows_in_branch_fails_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        branch_id = rows[0]["provenance"]["branch_id"]
        for row in rows:
            if row["provenance"]["branch_id"] == branch_id and row["oracle_rank"] != 1:
                row["oracle_rank"] = 1
                break
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        self.assertFalse(verification["verification_passed"])
        self.assertEqual(
            verification["branch_shape"]["multiple_rank1_branch_count"],
            1,
        )
        self.assertIn(
            "exactly_one_rank1_row_per_branch",
            verification["integrity_failures"],
        )

    def test_report_row_branch_best_oracle_count_mismatch_fails_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        oracle = archive_report["oracle_label_summary"]["oracle_action_counts"]
        oracle["stay"] -= 1
        oracle["eat"] = oracle.get("eat", 0) + 1
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        self.assertFalse(verification["verification_passed"])
        self.assertFalse(
            verification["branch_shape"][
                "branch_best_oracle_action_counts_match_report"
            ]
        )
        self.assertIn(
            "branch_best_oracle_action_counts_match_report",
            verification["integrity_failures"],
        )

    def test_mixed_serialized_oracle_best_action_values_fail_verification(self) -> None:
        archive_report, rows = _archive_inputs()
        branch_id = rows[0]["provenance"]["branch_id"]
        branch_rows = [
            row for row in rows if row["provenance"]["branch_id"] == branch_id
        ]
        rank1_action = next(
            row["candidate_action"] for row in branch_rows if row["oracle_rank"] == 1
        )
        for row in branch_rows:
            row["oracle_best_action"] = rank1_action
        wrong_row = next(row for row in branch_rows if row["oracle_rank"] != 1)
        wrong_row["oracle_best_action"] = wrong_row["candidate_action"]

        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        verification = build.report["source_archive_verification"]
        branch_shape = verification["branch_shape"]
        self.assertFalse(verification["verification_passed"])
        self.assertEqual(
            branch_shape["serialized_oracle_best_action_mismatch_count"],
            1,
        )
        self.assertIn(
            "serialized_oracle_best_action_consistency",
            verification["integrity_failures"],
        )
        self.assertEqual(
            branch_shape["serialized_oracle_best_action_mismatch_examples"][0][
                "rank1_candidate_action"
            ],
            rank1_action,
        )
        self.assertIn(
            wrong_row["candidate_action"],
            branch_shape["serialized_oracle_best_action_mismatch_examples"][0][
                "serialized_oracle_best_actions_present"
            ],
        )

    def test_fact_and_integrity_failures_are_separate(self) -> None:
        archive_report, rows = _archive_inputs()
        archive_report["target_selection"]["selected_target_count"] = 105
        rows[1]["candidate_action"] = "move_north"
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
            enforce_expected_v115_facts=True,
        )

        verification = build.report["source_archive_verification"]
        self.assertFalse(verification["verification_passed"])
        self.assertIn(
            "selected_target_count_report",
            verification["expected_v115_fact_failures"],
        )
        self.assertNotIn(
            "selected_target_count_report",
            verification["integrity_failures"],
        )
        self.assertIn(
            "no_duplicate_candidate_actions_within_branch",
            verification["integrity_failures"],
        )
        self.assertNotIn(
            "no_duplicate_candidate_actions_within_branch",
            verification["expected_v115_fact_failures"],
        )

    def test_missing_report_and_rows_are_inconclusive_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = build_first_recovery_archive_blocker_diagnostic(
                archive_report_path=root / "missing-report.json",
                archive_rows_path=root / "missing-rows.jsonl.gz",
            )

        self.assertEqual(
            build.report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )
        self.assertIn("archive_report", build.report["classification"]["missing_evidence"])
        self.assertIn("archive_rows", build.report["classification"]["missing_evidence"])

    def test_schema_mismatch_is_inconclusive(self) -> None:
        archive_report, rows = _archive_inputs()
        archive_report["schema_version"] = "wrong_schema_v1"
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
        )

        self.assertEqual(
            build.report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )
        self.assertFalse(
            build.report["source_reports"]["archive_report"]["schema_matches"]
        )
        self.assertIn(
            "archive_report_schema",
            build.report["classification"]["missing_evidence"],
        )

    def test_cli_smoke_with_temp_json_inputs(self) -> None:
        archive_report, rows = _archive_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_report_path = root / "archive.json"
            archive_rows_path = root / "archive.jsonl.gz"
            first_output = root / "blocker-first.json"
            second_output = root / "blocker-second.json"
            archive_report_path.write_text(
                json.dumps(archive_report, sort_keys=True),
                encoding="utf-8",
            )
            _write_rows(archive_rows_path, rows)
            cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_archive_blocker_diagnostic",
                "--archive-report",
                str(archive_report_path),
                "--archive-rows",
                str(archive_rows_path),
            ]
            env = {**os.environ, "PYTHONPATH": "python", "PYTHONHASHSEED": "0"}
            first = subprocess.run(
                [*cmd, "--output", str(first_output)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                [*cmd, "--output", str(second_output)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            first_bytes = first_output.read_bytes()
            second_bytes = second_output.read_bytes()
            report = json.loads(first_output.read_text())

        self.assertIn("first_recovery_archive_blocker_diagnostic=", first.stdout)
        self.assertIn("stay_oracle_count=64", first.stdout)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            report["schema_version"],
            MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION,
        )


def _archive_inputs(
    *,
    stay_best_count: int = 64,
    invalid_mode: str = "expected",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    non_stay_best_actions = ("eat", "drink", "move_east", "move_north")
    best_actions = [
        "stay" if index < stay_best_count else non_stay_best_actions[index % 4]
        for index in range(106)
    ]
    rows: list[dict[str, object]] = []
    for branch_index, best_action in enumerate(best_actions):
        logged = _logged_action(branch_index)
        actions = _candidate_actions_for_branch(
            branch_index=branch_index,
            best_action=best_action,
            logged_action=logged,
        )
        for rank_index, action in enumerate(actions, start=1):
            rank = 1 if action == best_action else rank_index + 1
            if action != best_action and rank == 1:
                rank = 2
            material = action == best_action and action != "stay"
            terminal_delta = 1 if material else 0
            birth_delta = 1 if material and branch_index % 2 == 0 else 0
            death_reduction_delta = 1 if material and branch_index % 3 == 0 else 0
            recovery_delta = 0.4 if material else 0.0
            invalid = _invalid_case(
                branch_index=branch_index,
                action=action,
                invalid_mode=invalid_mode,
            )
            rows.append(
                _row(
                    branch_index=branch_index,
                    action=action,
                    rank=rank,
                    material=material,
                    terminal_delta=terminal_delta,
                    birth_delta=birth_delta,
                    death_reduction_delta=death_reduction_delta,
                    recovery_delta=recovery_delta,
                    resolution_legal=invalid is None,
                    public_cause=invalid,
                    logged_action=logged,
                )
            )
    return _archive_report(rows, best_actions), rows


def _candidate_actions_for_branch(
    *,
    branch_index: int,
    best_action: str,
    logged_action: str,
) -> tuple[str, ...]:
    size = _branch_size(branch_index)
    preferred = [
        logged_action,
        best_action,
        "move_north",
        "stay",
        "eat",
        "drink",
        "move_east",
        "move_south",
        "move_west",
    ]
    actions: list[str] = []
    for action in preferred:
        if action is None or action in actions:
            continue
        actions.append(action)
        if len(actions) == size:
            return tuple(actions)
    return tuple(actions)


def _branch_size(branch_index: int) -> int:
    if branch_index < 10:
        return 3
    if branch_index < 34:
        return 4
    if branch_index < 66:
        return 5
    if branch_index < 102:
        return 6
    return 7


def _logged_action(branch_index: int) -> str:
    return ("eat", "move_east", "stay", "drink")[branch_index % 4]


def _invalid_case(
    *,
    branch_index: int,
    action: str,
    invalid_mode: str,
) -> str | None:
    if invalid_mode == "none":
        return None
    if branch_index >= 4 or action != "move_north":
        return None
    if invalid_mode == "mixed" and branch_index > 0:
        return "unexpected_private_failure"
    return "resolution_invalid_occupancy_race"


def _archive_report(
    rows: list[dict[str, object]],
    best_actions: list[str],
) -> dict[str, object]:
    action_counts = Counter(best_actions)
    dominant_action, dominant_count = sorted(
        action_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    invalid_count = sum(1 for row in rows if row["resolution_legal"] is False)
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "row_reconstruction": {
            "reconstructed_first_recovery_row_count": 106,
            "expected_constructible_first_recovery_row_count": 106,
            "row_count_matches_v107": True,
            "row_count_matches_v108": True,
        },
        "target_selection": {
            "selected_target_count": 106,
            "skipped_target_count": 0,
        },
        "branch_archive_summary": {
            "answer": "branch_archive_replay_verified",
            "selected_row_count": 106,
            "materialized_branch_point_count": 106,
            "materialization_failure_count": 0,
            "branch_result_count": 106,
            "action_run_count": len(rows),
            "archive_row_count": len(rows),
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "full_reconstruction_row_count": 106,
        },
        "oracle_label_summary": {
            "oracle_action_counts": dict(sorted(action_counts.items())),
            "dominant_oracle_action": dominant_action,
            "dominant_oracle_action_count": dominant_count,
            "dominant_oracle_action_share": round(dominant_count / 106, 6),
        },
        "legality_summary": {
            "resolution_invalid_count": invalid_count,
            "resolution_category_counts": {
                "oracle_actions_resolution_invalid": invalid_count,
                "oracle_actions_resolution_legal": len(rows) - invalid_count,
            },
        },
        "research_recommendation": {
            "recommendation": (
                "do_not_start_v110_until_archive_support_blockers_are_resolved"
            ),
        },
    }


def _row(
    *,
    branch_index: int,
    action: str,
    rank: int,
    material: bool,
    terminal_delta: int,
    birth_delta: int,
    death_reduction_delta: int,
    recovery_delta: float,
    resolution_legal: bool,
    public_cause: str | None,
    logged_action: str,
) -> dict[str, object]:
    seeds = (13, 19, 29, 37, 41, 43)
    seed = seeds[branch_index % len(seeds)]
    source = "open_mind_v3" if branch_index % 11 == 0 else "fixture_carrion_only"
    branch_id = (
        f"first-recovery-{source}-seed-{seed}-branch-{branch_index}-"
        f"tick-{branch_index % 120}-agent-{branch_index % 17}-logged-{logged_action}"
    )
    first = {
        "record_found": True,
        "forced_action": action,
        "observation_legal": True,
        "resolution_legal": resolution_legal,
        "action_valid": True,
        "resolution_action_valid": resolution_legal,
        "invalid_reason": None if resolution_legal else "not_in_resolution_action_mask",
        "resolved_action": action if resolution_legal else "stay",
        "moved": action.startswith("move_") and resolution_legal,
        "ate": action == "eat" and resolution_legal,
        "drank": action == "drink" and resolution_legal,
        "energy_ratio_delta": 0.2 if action == "eat" and resolution_legal else -0.01,
        "hydration_ratio_delta": (
            0.2 if action == "drink" and resolution_legal else -0.01
        ),
        "health_ratio_delta": 0.0,
    }
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "archive_row_id": f"{branch_id}::action::{action}",
        "source_kind": source,
        "seed": seed,
        "tick": branch_index % 120,
        "agent_id": branch_index % 17,
        "record_index": branch_index,
        "candidate_action": action,
        "replay_verification_result": True,
        "first_action_outcome": first,
        "recovery_vitals_deltas": {
            "energy_ratio_delta": first["energy_ratio_delta"],
            "hydration_ratio_delta": first["hydration_ratio_delta"],
            "health_ratio_delta": first["health_ratio_delta"],
            "target_recovery_score_delta": recovery_delta,
        },
        "terminal_alive_delta": terminal_delta,
        "birth_delta": birth_delta,
        "death_delta": -death_reduction_delta,
        "death_reduction_delta": death_reduction_delta,
        "material_gain_label": material,
        "oracle_rank": rank,
        "oracle_best_action": action if rank == 1 else None,
        "observation_legal": True,
        "resolution_legal": resolution_legal,
        "resolution_invalid_public_cause": public_cause,
        "trainable_public_input": {
            "schema_version": (
                "mind_v3_first_recovery_branch_archive_trainable_public_input_v1"
            ),
            "candidate_action": action,
            "candidate_action_index": 0,
            "action_mask": {
                "stay": True,
                "eat": True,
                "drink": True,
                "move_east": True,
                "move_south": True,
                "move_north": True,
                "move_west": True,
            },
            "target_public_state_before": {
                "alive": True,
                "energy_ratio": round((branch_index % 4 + 1) / 5, 4),
                "hydration_ratio": round((branch_index % 5 + 1) / 6, 4),
                "health_ratio": round((branch_index % 3 + 2) / 4, 4),
                "age": 20 + branch_index,
            },
            "public_transition_context": {
                "ticks_after_animal_resource_gain": branch_index % 9,
                "records_after_animal_resource_gain": branch_index % 7,
            },
        },
        "provenance": {
            "branch_id": branch_id,
            "source_kind": source,
            "seed": seed,
            "logged_action": logged_action,
            "diagnostics_only": True,
        },
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            with io.TextIOWrapper(gzip_file, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
                    handle.write("\n")


if __name__ == "__main__":
    unittest.main()
