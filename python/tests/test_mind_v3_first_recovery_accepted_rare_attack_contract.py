from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    build_first_recovery_accepted_rare_attack_contract,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = (
    "attack_east",
    "attack_west",
    "drink",
    "eat",
    "move_east",
    "move_north",
    "move_south",
    "move_west",
    "stay",
)


class MindV3FirstRecoveryAcceptedRareAttackContractTests(unittest.TestCase):
    def test_accepted_rare_attack_contract_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-accepted-rare-attack-contract"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_accepted_rare_attack_contract"
            ),
        )

    def test_clean_sources_merge_two_candidates_and_clear_split_support(self) -> None:
        payloads = _payloads()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertTrue(build.report["contract_checks"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "accepted_rare_attack_contract_ready_for_shadow_proposal",
        )
        self.assertEqual(len(build.manifest_rows), len(payloads["v119_rows"]) + 2)
        counts = build.report["contract_checks"]["repaired_action_counts"]
        self.assertEqual(counts["attack_east"], 4)
        self.assertEqual(counts["attack_west"], 4)
        self.assertTrue(
            build.report["split_support"][
                "strict_train_validation_test_support_met"
            ]
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["claim_causality"])

    def test_missing_v123_replay_verification_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["v123_report"]["replay_verified"] = False
        payloads["v123_report"]["replay_verification"][
            "missing_replay_verification_count"
        ] = 1
        payloads["v123_report"]["replay_verification"]["replay_verified"] = False

        build = _build(payloads)

        self._assert_source_failure(build, "v123_replay_not_verified")
        self._assert_source_failure(build, "v123_replay_verification_missing")

    def test_resolution_illegal_candidate_fails_contract(self) -> None:
        payloads = _payloads()
        payloads["candidate_rows"][0]["selected_resolution_legal"] = False

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_resolution_not_legal",
        )

    def test_objective_equivalence_violation_fails_contract(self) -> None:
        payloads = _payloads()
        payloads["candidate_rows"][0]["objective_equivalence_verified"] = False

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_objective_equivalence_not_verified",
        )

    def test_trainable_leakage_in_selected_v123_row_fails_contract(self) -> None:
        payloads = _payloads()
        selected_id = payloads["candidate_rows"][0]["candidate_archive_row_id"]
        for row in payloads["v123_rows"]:
            if row["archive_row_id"] == selected_id:
                row["trainable_public_input"]["seed"] = 13

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_trainable_input_leakage",
        )

    def test_duplicate_candidate_branch_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["candidate_rows"][1]["branch_id"] = payloads["candidate_rows"][0][
            "branch_id"
        ]

        build = _build(payloads)

        self._assert_source_failure(build, "accepted_candidate_branch_ids_not_unique")

    def test_candidate_selected_provenance_branch_mismatch_fails(self) -> None:
        payloads = _payloads()
        selected_id = payloads["candidate_rows"][0]["candidate_archive_row_id"]
        for row in payloads["v123_rows"]:
            if row["archive_row_id"] == selected_id:
                row["provenance"]["branch_id"] = "different-branch"

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_selected_provenance_branch_mismatch",
        )

    def test_candidate_archive_row_id_branch_prefix_mismatch_fails(self) -> None:
        payloads = _payloads()
        payloads["candidate_rows"][0][
            "candidate_archive_row_id"
        ] = "different-branch::action::attack_east"

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_archive_row_id_branch_prefix_mismatch",
        )

    def test_current_archive_row_id_branch_prefix_mismatch_fails(self) -> None:
        payloads = _payloads()
        payloads["candidate_rows"][0][
            "current_archive_row_id"
        ] = "different-branch::action::stay"

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_current_archive_row_id_branch_prefix_mismatch",
        )

    def test_candidate_archive_row_id_action_suffix_mismatch_fails(self) -> None:
        payloads = _payloads()
        payloads["candidate_rows"][0][
            "candidate_archive_row_id"
        ] = "v123-east::action::attack_west"

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "accepted_candidate_archive_row_id_action_suffix_mismatch",
        )

    def test_selected_v123_candidate_action_mismatch_fails(self) -> None:
        payloads = _payloads()
        selected_id = payloads["candidate_rows"][0]["candidate_archive_row_id"]
        for row in payloads["v123_rows"]:
            if row["archive_row_id"] == selected_id:
                row["candidate_action"] = "stay"

        build = _build(payloads)

        self._assert_source_failure(build, "accepted_candidate_selected_action_mismatch")

    def test_selected_v123_row_replay_evidence_is_required(self) -> None:
        cases = {
            "missing_result": (
                lambda row: row.pop("replay_verification_result"),
                "accepted_candidate_row_replay_result_not_true",
            ),
            "false_result": (
                lambda row: row.__setitem__("replay_verification_result", False),
                "accepted_candidate_row_replay_result_not_true",
            ),
            "malformed_result": (
                lambda row: row.__setitem__("replay_verification_result", "true"),
                "accepted_candidate_row_replay_result_not_true",
            ),
            "missing_digest": (
                lambda row: row.pop("replay_verification_digest"),
                "accepted_candidate_row_replay_digest_missing_or_malformed",
            ),
            "malformed_digest": (
                lambda row: row.__setitem__("replay_verification_digest", "bad"),
                "accepted_candidate_row_replay_digest_missing_or_malformed",
            ),
        }
        for name, (mutate, failure) in cases.items():
            with self.subTest(name=name):
                payloads = _payloads()
                selected_id = payloads["candidate_rows"][0]["candidate_archive_row_id"]
                for row in payloads["v123_rows"]:
                    if row["archive_row_id"] == selected_id:
                        mutate(row)

                build = _build(payloads)

                self._assert_source_failure(build, failure)

    def test_split_support_failed_when_other_action_support_is_too_thin(self) -> None:
        payloads = _payloads(other_action_count=3)

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertTrue(build.report["contract_checks"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "accepted_rare_attack_contract_split_support_failed",
        )
        self.assertFalse(
            build.report["split_support"][
                "strict_train_validation_test_support_met"
            ]
        )

    def test_blocker_flag_tampering_fails_source_integrity(self) -> None:
        payloads = _payloads()
        payloads["v122_over_report"]["recommendation"][
            "downstream_shadow_scorer_allowed"
        ] = True

        build = _build(payloads)

        self._assert_source_failure(
            build,
            "v122_over_v123_downstream_shadow_scorer_allowed_not_false",
        )

    def test_real_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v119-first-recovery-repaired-label-contract-audit.json",
            ROOT / "output/mind/mind-v3-v119-first-recovery-repaired-label-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v120-first-recovery-repaired-label-split-support-feasibility.json",
            ROOT / "output/mind/mind-v3-v122-first-recovery-rare-attack-coverage-collection.json",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz",
            ROOT / "output/mind/mind-v3-v122-over-v123-first-recovery-rare-attack-coverage-collection.json",
            ROOT / "output/mind/mind-v3-v122-over-v123-first-recovery-rare-attack-candidates.jsonl",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v119-v123 artifacts are not present")

        build = build_first_recovery_accepted_rare_attack_contract()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "accepted_rare_attack_contract_ready_for_shadow_proposal",
        )
        self.assertEqual(build.report["contract_checks"]["manifest_row_count"], 108)
        self.assertEqual(build.report["contract_checks"]["unique_branch_count"], 108)
        counts = build.report["contract_checks"]["repaired_action_counts"]
        self.assertEqual(counts["attack_east"], 4)
        self.assertEqual(counts["attack_west"], 4)
        self.assertTrue(
            build.report["split_support"][
                "strict_train_validation_test_support_met"
            ]
        )
        self.assertTrue(
            build.report["source_integrity"][
                "accepted_candidate_join_validation"
            ]["passed"]
        )
        self.assertTrue(
            build.report["source_integrity"][
                "accepted_candidate_row_replay_validation"
            ]["passed"]
        )
        self.assertEqual(
            build.report["source_integrity"][
                "accepted_candidate_row_replay_validation"
            ]["failure_count"],
            0,
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
            "accepted_rare_attack_contract_source_integrity_failed",
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_accepted_rare_attack_contract(
        v119_report=copy.deepcopy(payloads["v119_report"]),
        v119_manifest_rows=copy.deepcopy(payloads["v119_rows"]),
        v120_report=copy.deepcopy(payloads["v120_report"]),
        v122_default_report=copy.deepcopy(payloads["v122_default_report"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
        v122_over_v123_report=copy.deepcopy(payloads["v122_over_report"]),
        v122_over_v123_candidate_rows=copy.deepcopy(payloads["candidate_rows"]),
    )


def _payloads(*, other_action_count: int = 4) -> dict[str, object]:
    v119_rows = _v119_rows(other_action_count=other_action_count)
    digest = stable_payload_digest(list(v119_rows))
    v123_rows = _v123_rows()
    candidate_rows = _candidate_rows()
    return {
        "v119_rows": v119_rows,
        "v119_report": _v119_report(v119_rows, digest),
        "v120_report": _v120_report(digest),
        "v122_default_report": _v122_default_report(),
        "v123_report": _v123_report(v123_rows),
        "v123_rows": v123_rows,
        "v122_over_report": _v122_over_report(),
        "candidate_rows": candidate_rows,
    }


def _v119_rows(*, other_action_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.append(_manifest_row(f"v119-attack-east-{index}", "attack_east"))
        rows.append(_manifest_row(f"v119-attack-west-{index}", "attack_west"))
    for action in ACTIONS:
        if action in {"attack_east", "attack_west"}:
            continue
        for index in range(other_action_count):
            rows.append(_manifest_row(f"v119-{action}-{index}", action))
    return rows


def _manifest_row(branch_id: str, action: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_repaired_label_contract_audit_v1",
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
            "candidate_action": action,
            "target_public_state_before": {"alive": True},
        },
        "non_trainable_audit_metadata": {
            "non_trainable": True,
            "purpose": "audit_only_not_trainable",
            "seed": 1,
        },
    }


def _v119_report(rows: list[dict[str, object]], digest: str) -> dict[str, object]:
    counts = _counts(rows)
    return {
        "schema_version": "mind_v3_first_recovery_repaired_label_contract_audit_v1",
        "classification": {"primary": "repaired_label_contract_support_limited"},
        "manifest": {
            "manifest_digest": digest,
            "manifest_row_count": len(rows),
        },
        "contract_checks": {
            "passed": True,
            "total_violation_count": 0,
            "integrity_failures": [],
            "manifest_row_count": len(rows),
            "branch_count": len(rows),
            "repaired_action_counts": counts,
        },
        "recommendation": _blocked_recommendation(include_replay_golden=False),
    }


def _v120_report(digest: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_repaired_label_split_support_feasibility_v1",
        "classification": {
            "primary": "split_support_feasibility_limited_by_rare_actions"
        },
        "source_integrity": {
            "passed": True,
            "failures": [],
            "reported_manifest_digest": digest,
            "computed_manifest_digest": digest,
            "manifest_digest_matches": True,
        },
        "scarcity_analysis": {
            "rare_action_additional_needed_for_train2_validation1_test1": {
                "attack_east": 1,
                "attack_west": 1,
            }
        },
        "recommendation": _blocked_recommendation(include_replay_golden=False),
    }


def _v122_default_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_rare_attack_coverage_collection_v1",
        "classification": {
            "primary": "rare_attack_coverage_not_found_within_budget",
            "labels": [
                "rare_attack_coverage_not_found_within_budget",
                "diagnostics_only_no_runtime_promotion",
                "readiness_rerun_blocked",
                "candidate_source_contains_no_unmanifested_branches",
            ],
        },
        "source_integrity": {"passed": True, "failures": []},
        "found_candidate_counts": {"attack_east": 0, "attack_west": 0},
        "candidate_collection": {
            "candidate_source_domain": {
                "unmanifested_candidate_branch_count": 0,
            }
        },
        "recommendation": _blocked_recommendation(include_replay_golden=True),
    }


def _v123_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "active_coverage_schema_version": (
            "mind_v3_first_recovery_active_coverage_archive_v1"
        ),
        "classification": {"primary": "active_coverage_rare_attacks_found"},
        "source_integrity": {"passed": True, "failures": []},
        "generated_archive_row_count": len(rows),
        "generated_branch_count": 2,
        "accepted_by_v122_candidate_counts": {
            "attack_east": 1,
            "attack_west": 1,
        },
        "replay_verified": True,
        "replay_verification": {
            "replay_verified": True,
            "missing_replay_verification_count": 0,
            "replay_verification_skipped_count": 0,
            "replay_verification_failure_count": 0,
        },
        "heuristic_action_source_count": 0,
        "strict_seed_leakage_count": 0,
        "trainable_leakage": {
            "branch_archive_leakage": {"leakage_count": 0},
            "split_key_leak_count": 0,
            "forbidden_metadata_key_count": 0,
        },
        "recommendation": _blocked_recommendation(include_replay_golden=True),
    }


def _v122_over_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_rare_attack_coverage_collection_v1",
        "classification": {"primary": "rare_attack_coverage_candidates_found"},
        "source_integrity": {"passed": True, "failures": []},
        "found_candidate_counts": {"attack_east": 1, "attack_west": 1},
        "recommendation": {
            **_blocked_recommendation(include_replay_golden=True),
            "would_clear_v120_rare_action_limitation_if_accepted": True,
        },
    }


def _candidate_rows() -> list[dict[str, object]]:
    return [
        _candidate_row("v123-east", "attack_east"),
        _candidate_row("v123-west", "attack_west"),
    ]


def _candidate_row(branch_id: str, action: str) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_rare_attack_coverage_collection_v1",
        "branch_id": branch_id,
        "target_action": action,
        "current_oracle_action": "stay",
        "repaired_action": action,
        "current_archive_row_id": f"{branch_id}::action::stay",
        "candidate_archive_row_id": f"{branch_id}::action::{action}",
        "changed": True,
        "unique_objective_best": False,
        "objective_equivalence_verified": True,
        "selected_resolution_legal": True,
        "trainable_public_input_present": True,
        "trainable_public_input_clean": True,
        "trainable_public_input_contents_exposed": False,
        "legal_tied_candidate_actions": [action, "stay"],
        "serialized_objective_key": [0, 0.0, 1.0, 0.0, -1.0],
        "diagnostics_only": True,
        "selection_authorized": False,
    }


def _v123_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for branch_id, action in (("v123-east", "attack_east"), ("v123-west", "attack_west")):
        rows.append(_archive_row(branch_id, "stay", rank=1))
        rows.append(_archive_row(branch_id, action, rank=2))
    return rows


def _archive_row(branch_id: str, action: str, *, rank: int) -> dict[str, object]:
    return {
        "archive_row_id": f"{branch_id}::action::{action}",
        "candidate_action": action,
        "oracle_rank": rank,
        "resolution_legal": True,
        "observation_legal": True,
        "trainable_public_input": {
            "candidate_action": action,
            "target_public_state_before": {"alive": True},
        },
        "observation_digest": f"observation-{branch_id}-{action}",
        "replay_verification_result": True,
        "replay_verification_digest": stable_payload_digest(
            {
                "branch_id": branch_id,
                "candidate_action": action,
                "verified": True,
            }
        ),
        "provenance": {
            "branch_id": branch_id,
            "seed": 13,
            "source_kind": "fixture_carrion_only",
            "source_path": "synthetic.jsonl.gz",
            "record_index": 1,
            "agent_id": 2,
        },
    }


def _blocked_recommendation(*, include_replay_golden: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
    }
    if include_replay_golden:
        payload["replay_golden_change_recommended"] = False
    return payload


def _counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {}
    for row in rows:
        action = str(row["repaired_action"])
        counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    unittest.main()
