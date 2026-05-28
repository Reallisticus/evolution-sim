from __future__ import annotations

import copy
import gzip
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution_sim.mind.first_recovery_active_coverage_archive import (
    _GenerationResult,
    build_first_recovery_active_coverage_archive,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryActiveCoverageArchiveTests(unittest.TestCase):
    def test_active_coverage_archive_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-active-coverage-archive"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_active_coverage_archive"
            ),
        )

    def test_source_integrity_failure_blocks_generation(self) -> None:
        manifest = _manifest_rows()
        reports = _reports(manifest)
        reports["v120"]["source_integrity"]["passed"] = False

        with patch(
            "evolution_sim.mind.first_recovery_active_coverage_archive."
            "_generate_active_archive_rows"
        ) as generator:
            build = _build(manifest=manifest, reports=reports)

        generator.assert_not_called()
        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "active_coverage_source_integrity_failed",
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )

    def test_complete_coverage_found_with_clean_generated_rows(self) -> None:
        rows = _candidate_branch("v123-east", "attack_east")
        rows += _candidate_branch("v123-west", "attack_west")

        build = _build_with_generation(rows)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "active_coverage_rare_attacks_found",
        )
        self.assertEqual(
            build.report["accepted_by_v122_candidate_counts"],
            {"attack_east": 1, "attack_west": 1},
        )
        self.assertTrue(
            build.report["recommendation"][
                "would_clear_v120_rare_action_limitation_if_accepted"
            ]
        )
        self.assertEqual(build.report["strict_seed_leakage_count"], 0)

    def test_partial_coverage_classification(self) -> None:
        rows = _candidate_branch("v123-east", "attack_east")

        build = _build_with_generation(rows)

        self.assertEqual(
            build.report["classification"]["primary"],
            "active_coverage_partial_rare_attack_found",
        )
        self.assertEqual(
            build.report["accepted_by_v122_candidate_counts"],
            {"attack_east": 1, "attack_west": 0},
        )

    def test_no_rare_attack_found_classification(self) -> None:
        build = _build_with_generation([])

        self.assertEqual(
            build.report["classification"]["primary"],
            "active_coverage_source_integrity_failed",
        )
        self.assertIn(
            "replay_verification_failed",
            build.report["source_integrity"]["failures"],
        )

    def test_replay_failure_fails_source_integrity(self) -> None:
        rows = _candidate_branch("v123-east", "attack_east")

        build = _build_with_generation(rows, replay_verified=False)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "replay_verification_failed",
            build.report["source_integrity"]["failures"],
        )

    def test_missing_replay_verification_fails_source_integrity(self) -> None:
        rows = _candidate_branch("v123-east", "attack_east")

        build = _build_with_generation(rows, replay_verified=None)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertFalse(build.report["replay_verified"])
        self.assertEqual(
            build.report["replay_verification"]["missing_replay_verification_count"],
            1,
        )
        self.assertIn(
            "replay_verification_missing",
            build.report["source_integrity"]["failures"],
        )

    def test_skipped_replay_verification_fails_source_integrity(self) -> None:
        rows = _candidate_branch("v123-east", "attack_east")

        build = _build_with_generation(rows, replay_verified="skipped")

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertFalse(build.report["replay_verified"])
        self.assertEqual(
            build.report["replay_verification"]["replay_verification_skipped_count"],
            1,
        )
        self.assertIn(
            "replay_verification_skipped",
            build.report["source_integrity"]["failures"],
        )

    def test_trainable_leakage_fails_source_integrity(self) -> None:
        rows = _candidate_branch("v123-east", "attack_east")
        rows[1]["trainable_public_input"]["seed"] = 13

        build = _build_with_generation(rows)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "trainable_public_input_leakage",
            build.report["source_integrity"]["failures"],
        )
        self.assertIn(
            "strict_seed_leakage",
            build.report["source_integrity"]["failures"],
        )

    def test_v122_rejection_keeps_no_found_classification_when_replay_clean(self) -> None:
        rows = _candidate_branch(
            "v123-east",
            "attack_east",
            target_terminal_delta=1,
        )

        build = _build_with_generation(rows)

        self.assertEqual(
            build.report["classification"]["primary"],
            "active_coverage_no_rare_attack_found",
        )
        self.assertEqual(
            build.report["accepted_by_v122_candidate_counts"],
            {"attack_east": 0, "attack_west": 0},
        )

    def test_real_v123_artifacts_when_local_data_exists(self) -> None:
        report_path = ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json"
        rows_path = ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz"
        candidates_path = ROOT / "output/mind/mind-v3-v122-over-v123-first-recovery-rare-attack-candidates.jsonl"
        if not all(path.exists() for path in (report_path, rows_path, candidates_path)):
            self.skipTest("real v123 artifacts are not present")

        report = json.loads(report_path.read_text())
        rows: list[dict[str, object]] = []
        with gzip.open(rows_path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        candidate_rows = [
            json.loads(line)
            for line in candidates_path.read_text().splitlines()
            if line.strip()
        ]

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            len({row["provenance"]["branch_id"] for row in rows}),
            2,
        )
        rare_rows = [
            row for row in rows
            if row["candidate_action"] in {"attack_east", "attack_west"}
        ]
        self.assertEqual(len(rare_rows), 2)
        for row in rare_rows:
            self.assertTrue(row["observation_legal"])
            self.assertTrue(row["resolution_legal"])
        self.assertEqual(len(candidate_rows), 2)


def _build_with_generation(
    rows: list[dict[str, object]],
    *,
    replay_verified: bool = True,
):
    generation = _generation(rows, replay_verified=replay_verified)
    with patch(
        "evolution_sim.mind.first_recovery_active_coverage_archive."
        "_generate_active_archive_rows",
        return_value=generation,
    ):
        return _build()


def _build(
    *,
    manifest: list[dict[str, object]] | None = None,
    reports: dict[str, dict[str, object]] | None = None,
):
    manifest_rows = manifest if manifest is not None else _manifest_rows()
    payloads = reports if reports is not None else _reports(manifest_rows)
    return build_first_recovery_active_coverage_archive(
        v119_report=payloads["v119"],
        manifest_rows=copy.deepcopy(manifest_rows),
        v120_report=payloads["v120"],
        v121_report=payloads["v121"],
        rollout_context_report_path=ROOT / "package.json",
        trajectory_paths=(Path(__file__),),
        trajectory_glob_patterns=(),
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


def _generation(
    rows: list[dict[str, object]],
    *,
    replay_verified: bool | None | str,
) -> _GenerationResult:
    branch_ids = sorted(
        {
            str(row["provenance"]["branch_id"])
            for row in rows
            if isinstance(row.get("provenance"), dict)
        }
    )
    if replay_verified == "skipped":
        run = {
            "forced_action": "stay",
            "replay_verification": None,
            "heuristic_action_source_count": 0,
        }
    elif replay_verified is None:
        run = {
            "forced_action": "stay",
            "heuristic_action_source_count": 0,
        }
    else:
        run = {
            "forced_action": "stay",
            "replay_verification": {"verified": replay_verified},
            "heuristic_action_source_count": 0,
        }
    return _GenerationResult(
        archive_rows=tuple(copy.deepcopy(rows)),
        branch_results=tuple(
            {
                "branch_id": branch_id,
                "action_runs": [dict(run)],
            }
            for branch_id in branch_ids
        ),
        accepted_branch_ids=tuple(branch_ids),
        search_budget_used={
            "trajectory_count": 1,
            "source_record_counts": {"attack_east": 1, "attack_west": 1},
            "selected_source_record_count": len(branch_ids),
            "evaluated_branch_count": len(branch_ids),
            "max_evaluated_branches": 12,
            "accepted_branch_count": len(branch_ids),
            "budget_exhausted": False,
        },
        source_record_counts={"attack_east": 1, "attack_west": 1},
        rejection_counts={"attack_east": {}, "attack_west": {}},
        rejection_examples={"attack_east": {}, "attack_west": {}},
        materialization_failures=(),
    )


def _candidate_branch(
    branch_id: str,
    target_action: str,
    *,
    target_terminal_delta: int = 0,
) -> list[dict[str, object]]:
    return [
        _archive_row(branch_id, "stay", rank=1, terminal_delta=0),
        _archive_row(
            branch_id,
            target_action,
            rank=2,
            terminal_delta=target_terminal_delta,
        ),
    ]


def _archive_row(
    branch_id: str,
    action: str,
    *,
    rank: int,
    terminal_delta: int,
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
        "resolution_legal": True,
        "trainable_public_input": {
            "candidate_action": action,
            "target_public_state_before": {"alive": True},
        },
        "provenance": {
            "branch_id": branch_id,
        },
    }


if __name__ == "__main__":
    unittest.main()
