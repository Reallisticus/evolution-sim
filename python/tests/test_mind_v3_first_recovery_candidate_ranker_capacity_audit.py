from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    ACTION_ONLY_BASELINE,
    ACTION_ORDER_BASELINE,
    EXPECTED_CANDIDATE_ROW_COUNT,
    EXPECTED_NEGATIVE_ROW_COUNT,
    EXPECTED_POSITIVE_ROW_COUNT,
    EXPECTED_BRANCH_COUNT,
    PRIMARY_PROBE,
    build_first_recovery_candidate_ranker_capacity_audit,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    EXPECTED_REPAIRED_ACTION_COUNTS,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryCandidateRankerCapacityAuditTests(unittest.TestCase):
    def test_candidate_ranker_capacity_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-candidate-ranker-capacity-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_candidate_ranker_capacity_audit"
            ),
        )

    def test_public_input_interaction_probe_can_be_ready_for_review(self) -> None:
        payloads = _payloads(context_mode="by_action")

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_ranker_capacity_ready_for_review",
        )
        self.assertTrue(
            build.report["probe_comparisons"][
                "heldout_signal_beats_action_only_baseline"
            ]
        )
        self.assertTrue(build.report["metric_gate"]["passed"])
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )

    def test_no_interaction_gain_blocks_by_signal(self) -> None:
        payloads = _payloads(context_mode="constant")

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_ranker_capacity_blocked_by_signal",
        )
        self.assertIn(
            "heldout_signal_not_above_action_only_baseline",
            build.report["metric_gate"]["failures"],
        )
        self.assertIn(
            PRIMARY_PROBE,
            build.report["probe_comparisons"]["per_probe_vs_action_only"],
        )
        self.assertIn(
            ACTION_ORDER_BASELINE,
            build.report["probe_comparisons"]["per_probe_vs_action_only"],
        )
        self.assertIn(
            ACTION_ONLY_BASELINE,
            build.report["probe_comparisons"]["per_probe_vs_action_only"],
        )

    def test_feature_variance_reports_only_action_fields_when_context_is_branch_constant(
        self,
    ) -> None:
        payloads = _payloads(context_mode="by_action")

        build = _build(payloads)
        variance = build.report["feature_variance"]

        self.assertTrue(variance["non_action_public_fields_branch_constant"])
        self.assertEqual(variance["candidate_action_varies_by_branch_count"], 108)
        self.assertEqual(
            variance["candidate_action_index_varies_by_branch_count"],
            108,
        )
        self.assertEqual(
            build.report["input_allowlist_audit"]["forbidden_feature_path_count"],
            0,
        )
        self.assertFalse(
            build.report["input_allowlist_audit"]["candidate_action_index_used"]
        )

    def test_v127_count_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads(context_mode="by_action")
        payloads["v127_report"]["candidate_set_audit"][
            "total_candidate_rows"
        ] = EXPECTED_CANDIDATE_ROW_COUNT - 1

        build = _build(payloads)

        self._assert_source_failure(build, "v127_candidate_row_count_unexpected")

    def test_v127_prediction_count_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads(context_mode="by_action")
        payloads["v127_prediction_rows"] = payloads["v127_prediction_rows"][:-1]

        build = _build(payloads)

        self._assert_source_failure(build, "v127_prediction_row_count_unexpected")

    def test_candidate_trainable_leakage_fails_source_integrity(self) -> None:
        payloads = _payloads(context_mode="by_action")
        payloads["v115_rows"][0]["trainable_public_input"]["seed"] = 29

        build = _build(payloads)

        self._assert_source_failure(build, "candidate_trainable_leakage_detected")

    def test_real_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v127-first-recovery-candidate-set-shadow-execution.json",
            ROOT / "output/mind/mind-v3-v127-first-recovery-candidate-set-predictions.jsonl",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115/v123/v124/v127 artifacts are not present")

        build = build_first_recovery_candidate_ranker_capacity_audit()
        source = build.report["source_integrity"]
        variance = build.report["feature_variance"]

        self.assertTrue(source["passed"])
        self.assertEqual(source["branch_count"], 108)
        self.assertEqual(source["candidate_row_count"], 542)
        self.assertEqual(source["positive_row_count"], 108)
        self.assertEqual(source["negative_row_count"], 434)
        self.assertEqual(source["exact_repaired_archive_row_joins"], 108)
        self.assertEqual(source["trainable_leakage_count"], 0)
        self.assertEqual(source["unsupported_repaired_labels_count"], 0)
        self.assertEqual(
            build.report["classification"]["primary"],
            "candidate_ranker_capacity_blocked_by_signal",
        )
        self.assertTrue(variance["non_action_public_fields_branch_constant"])
        self.assertEqual(variance["candidate_action_varies_by_branch_count"], 108)
        self.assertIn(
            "heldout_signal_not_above_action_only_baseline",
            build.report["metric_gate"]["failures"],
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
            "candidate_ranker_capacity_source_integrity_failed",
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_candidate_ranker_capacity_audit(
        v127_report=copy.deepcopy(payloads["v127_report"]),
        v127_prediction_rows=copy.deepcopy(payloads["v127_prediction_rows"]),
        v124_manifest_rows=copy.deepcopy(payloads["manifest_rows"]),
        v115_report=copy.deepcopy(payloads["v115_report"]),
        v115_archive_rows=copy.deepcopy(payloads["v115_rows"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
    )


def _payloads(*, context_mode: str) -> dict[str, object]:
    manifest_rows = _manifest_rows()
    branch_sizes = _branch_sizes()
    v115_branch_ids = {row["branch_id"] for row in manifest_rows[:106]}
    v115_rows: list[dict[str, object]] = []
    v123_rows: list[dict[str, object]] = []
    for manifest, size in zip(manifest_rows, branch_sizes, strict=True):
        target = v115_rows if manifest["branch_id"] in v115_branch_ids else v123_rows
        target.extend(
            _candidate_rows(
                manifest,
                size=size,
                context_mode=context_mode,
            )
        )
    v127_predictions = [
        {
            "schema_version": "mind_v3_first_recovery_candidate_set_prediction_v1",
            "branch_id": row["branch_id"],
            "predicted_action": "eat",
        }
        for row in manifest_rows
    ]
    return {
        "manifest_rows": manifest_rows,
        "v115_rows": v115_rows,
        "v123_rows": v123_rows,
        "v127_prediction_rows": v127_predictions,
        "v127_report": _v127_report(v127_predictions),
        "v115_report": _archive_report(v115_rows),
        "v123_report": _archive_report(v123_rows, active=True),
    }


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_cycle = [29, 13, 37, 41, 43, 19]
    for action, count in EXPECTED_REPAIRED_ACTION_COUNTS.items():
        for index in range(count):
            branch_id = f"v128-{action}-{index}"
            seed = seed_cycle[len(rows) % len(seed_cycle)]
            rows.append(
                {
                    "schema_version": (
                        "mind_v3_first_recovery_accepted_rare_attack_contract_v1"
                    ),
                    "branch_id": branch_id,
                    "repaired_action": action,
                    "repaired_archive_row_id": f"{branch_id}::action::{action}",
                    "trainable_public_input": _trainable(
                        action,
                        branch_actions=[action, "stay"],
                        context_action=action,
                    ),
                    "non_trainable_audit_metadata": {
                        "non_trainable": True,
                        "purpose": "audit_only_not_trainable",
                        "seed": seed,
                        "source": "synthetic_fixture_open_source",
                    },
                }
            )
    return rows


def _branch_sizes() -> list[int]:
    sizes = ([3] * 10) + ([4] * 24) + ([5] * 32) + ([6] * 38) + ([7] * 4)
    self_check = sum(sizes)
    if len(sizes) != EXPECTED_BRANCH_COUNT or self_check != EXPECTED_CANDIDATE_ROW_COUNT:
        raise AssertionError("synthetic branch size distribution is invalid")
    return sizes


def _candidate_rows(
    manifest: dict[str, object],
    *,
    size: int,
    context_mode: str,
) -> list[dict[str, object]]:
    branch_id = str(manifest["branch_id"])
    repaired = str(manifest["repaired_action"])
    actions = _branch_actions(repaired, size)
    context_action = repaired if context_mode == "by_action" else "constant"
    return [
        _candidate_row(
            branch_id,
            action,
            branch_actions=actions,
            context_action=context_action,
            material_gain=(action == repaired and repaired in {"attack_east", "attack_west", "eat"}),
        )
        for action in actions
    ]


def _branch_actions(repaired: str, size: int) -> list[str]:
    action_pool = list(EXPECTED_REPAIRED_ACTION_COUNTS)
    actions = [repaired]
    start = action_pool.index(repaired)
    for offset in range(1, len(action_pool) + 1):
        action = action_pool[(start + offset) % len(action_pool)]
        if action not in actions:
            actions.append(action)
        if len(actions) == size:
            break
    return actions


def _candidate_row(
    branch_id: str,
    action: str,
    *,
    branch_actions: list[str],
    context_action: str,
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
        "trainable_public_input": _trainable(
            action,
            branch_actions=branch_actions,
            context_action=context_action,
        ),
        "provenance": {
            "branch_id": branch_id,
            "source_kind": "synthetic_fixture_open_source",
            "source_path": "synthetic.jsonl.gz",
            "diagnostics_only": True,
        },
        "replay_verification_result": True,
        "replay_verification_digest": "a" * 64,
    }


def _trainable(
    action: str,
    *,
    branch_actions: list[str],
    context_action: str,
) -> dict[str, object]:
    context = _context_values(context_action)
    return {
        "schema_version": (
            "mind_v3_first_recovery_branch_archive_trainable_public_input_v1"
        ),
        "candidate_action": action,
        "candidate_action_index": list(EXPECTED_REPAIRED_ACTION_COUNTS).index(action),
        "action_mask": {
            candidate: candidate in branch_actions
            for candidate in EXPECTED_REPAIRED_ACTION_COUNTS
        },
        "post_carrion_first_recovery": context["post_carrion"],
        "public_transition_context": {
            "records_after_animal_resource_gain": context["records"],
            "ticks_after_animal_resource_gain": context["ticks"],
        },
        "target_public_state_before": {
            "age": context["age"],
            "alive": True,
            "energy_ratio": context["energy"],
            "health_ratio": context["health"],
            "hydration_ratio": context["hydration"],
        },
    }


def _context_values(action: str) -> dict[str, object]:
    action_contexts = {
        "attack_east": (0.35, 0.35, 0.35, 8, 0, 0, True),
        "attack_west": (0.35, 0.35, 0.9, 30, 2, 2, True),
        "drink": (0.35, 0.65, 0.35, 60, 8, 8, True),
        "eat": (0.35, 0.65, 0.9, 8, 20, 20, True),
        "move_east": (0.65, 0.35, 0.35, 30, 0, 8, True),
        "move_north": (0.65, 0.35, 0.9, 60, 2, 20, False),
        "move_south": (0.65, 0.9, 0.35, 8, 8, 0, False),
        "move_west": (0.9, 0.35, 0.35, 30, 20, 2, False),
        "stay": (0.9, 0.9, 0.9, 60, 0, 20, False),
        "constant": (0.9, 0.9, 0.9, 60, 0, 20, False),
    }
    energy, hydration, health, age, records, ticks, post_carrion = action_contexts[action]
    return {
        "energy": energy,
        "hydration": hydration,
        "health": health,
        "age": age,
        "records": records,
        "ticks": ticks,
        "post_carrion": post_carrion,
    }


def _v127_report(prediction_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_candidate_set_shadow_execution_v1",
        "classification": {
            "primary": "candidate_set_shadow_execution_blocked_by_metrics"
        },
        "source_integrity": {"passed": True, "failures": []},
        "candidate_set_audit": {
            "branch_count": EXPECTED_BRANCH_COUNT,
            "total_candidate_rows": EXPECTED_CANDIDATE_ROW_COUNT,
            "positive_rows_count": EXPECTED_POSITIVE_ROW_COUNT,
            "negative_rows_count": EXPECTED_NEGATIVE_ROW_COUNT,
            "unsupported_repaired_labels_count": 0,
            "repaired_archive_row_join": {
                "passed": True,
                "exact_join_count": EXPECTED_BRANCH_COUNT,
            },
        },
        "leakage_audit": {"trainable_leakage_count": 0},
        "unsupported_action_audit": {"unsupported_action_count": 0},
        "prediction_summary": {
            "prediction_row_count": len(prediction_rows),
            "prediction_rows_digest": stable_payload_digest(prediction_rows),
        },
        "recommendation": _recommendation(),
    }


def _archive_report(rows: list[dict[str, object]], *, active: bool = False) -> dict[str, object]:
    del active
    branch_ids = {
        row["provenance"]["branch_id"]
        for row in rows
    }
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "branch_archive_summary": {
            "archive_row_count": len(rows),
            "branch_result_count": len(branch_ids),
            "heuristic_action_source_count": 0,
            "replay_verified": True,
            "archive_rows_sha256": stable_payload_digest(rows),
        },
        "recommendation": _recommendation(),
    }


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
