from __future__ import annotations

from copy import deepcopy
import json
import unittest
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_archive_override_autopsy import (
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
    V149_CARRION_TRAIN_EVAL_POLICY,
    V149_CARRION_TRAIN_EVAL_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_sequence_context_archive import (
    DEFAULT_HARMFUL_SUPPORT_SOURCES,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
    CarrionSequenceContextArchiveError,
    build_carrion_sequence_context_archive_report,
    carrion_sequence_context_trainable_leakage_scan,
    merge_carrion_sequence_context_archive_shards,
)
from evolution_sim.mind.carrion_specific_archive_expansion import (
    M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY,
    M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION,
    TARGET_CARRION_SEEDS,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSequenceContextArchiveTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-sequence-context-archive"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_sequence_context_archive"
            ),
        )

    def test_report_separates_harmful_sources_with_prior_public_context(self) -> None:
        inputs = _synthetic_inputs()

        report, rows = build_carrion_sequence_context_archive_report(
            **inputs,
            prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
        )

        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertTrue(report["replay_verification"]["complete"])
        self.assertEqual(report["coverage"]["harmful_source_row_count"], 3)
        self.assertEqual(report["coverage"]["safe_comparator_row_count"], 3)
        self.assertEqual(
            report["sequence_context_separation"][
                "sequence_separated_harmful_source_count"
            ],
            3,
        )
        self.assertTrue(
            report["sequence_context_separation"][
                "all_harmful_sources_sequence_separated"
            ]
        )
        self.assertEqual(
            report["classification"]["primary"],
            "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training",
        )
        self.assertEqual(report["dataset"]["row_count"], len(rows))
        self.assertEqual(report["dataset"]["dataset_digest"], stable_payload_digest(rows))
        feature_text = json.dumps(
            [row["trainable"]["features"] for row in rows],
            sort_keys=True,
        )
        for forbidden in (
            "seed",
            "fixture",
            "branch",
            "agent",
            "tick",
            "digest",
            "path",
            "private",
            "future",
            "label",
            "source",
        ):
            self.assertNotIn(forbidden, feature_text)

    def test_trainable_input_leakage_rejects_identity_digest_and_label_features(
        self,
    ) -> None:
        scan = carrion_sequence_context_trainable_leakage_scan(
            [
                {
                    "trainable": {
                        "features": {
                            "seed": 13,
                            "branch_id": "bad",
                            "agent_id": 7,
                            "digest": "a" * 64,
                            "label": "move_north",
                        },
                    }
                }
            ]
        )

        self.assertFalse(scan["passed"])
        reasons = {failure["reason"] for failure in scan["failures"]}
        self.assertIn("forbidden_trainable_input_path_token", reasons)
        self.assertIn("forbidden_trainable_input_digest_value", reasons)
        self.assertIn("label_present_in_trainable_input_features", reasons)

    def test_rejects_v148_archive_branch_result_count_mismatch(self) -> None:
        inputs = _synthetic_inputs()
        archive_report = deepcopy(inputs["archive_report"])
        archive_report["branch_result_count"] = 999
        inputs["archive_report"] = archive_report

        with self.assertRaisesRegex(
            CarrionSequenceContextArchiveError,
            "v148_archive_branch_result_count_mismatch",
        ):
            build_carrion_sequence_context_archive_report(
                **inputs,
                prior_context_by_branch_id=_prior_contexts(archive_report),
            )

    def test_rejects_v150_autopsy_input_validation_failure(self) -> None:
        inputs = _synthetic_inputs()
        autopsy_report = deepcopy(inputs["autopsy_report"])
        autopsy_report["inputs"]["input_validation"] = {
            "passed": False,
            "failures": ["unit_failure"],
        }
        inputs["autopsy_report"] = autopsy_report

        with self.assertRaisesRegex(
            CarrionSequenceContextArchiveError,
            "v150_autopsy_input_validation_not_passed",
        ):
            build_carrion_sequence_context_archive_report(
                **inputs,
                prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
            )

    def test_synthetic_two_shard_merge_succeeds(self) -> None:
        inputs = _synthetic_inputs()
        full_report, _rows = build_carrion_sequence_context_archive_report(
            **inputs,
            prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
        )
        shard_a = _shard_report(
            inputs,
            full_report["branch_results"][:3],
            shard_id="shard-a",
        )
        shard_b = _shard_report(
            inputs,
            full_report["branch_results"][3:],
            shard_id="shard-b",
        )

        merged, rows = merge_carrion_sequence_context_archive_shards(
            **inputs,
            shard_reports=[shard_a, shard_b],
        )

        self.assertTrue(merged["source_integrity"]["passed"])
        self.assertEqual(merged["coverage"]["harmful_source_row_count"], 3)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(merged["shard_merge"]["sources"]), 2)
        self.assertEqual(
            {
                source["shard_id"]
                for source in merged["shard_merge"]["sources"]
                if source.get("shard_id")
            },
            {"shard-a", "shard-b"},
        )

    def test_reversed_shard_order_yields_same_exact_and_dataset_digest(self) -> None:
        inputs = _synthetic_inputs()
        full_report, _rows = build_carrion_sequence_context_archive_report(
            **inputs,
            prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
        )
        shard_a = _shard_report(
            inputs,
            full_report["branch_results"][:3],
            shard_id="shard-a",
        )
        shard_b = _shard_report(
            inputs,
            full_report["branch_results"][3:],
            shard_id="shard-b",
        )

        merged_ab, rows_ab = merge_carrion_sequence_context_archive_shards(
            **inputs,
            shard_reports=[shard_a, shard_b],
        )
        merged_ba, rows_ba = merge_carrion_sequence_context_archive_shards(
            **inputs,
            shard_reports=[shard_b, shard_a],
        )

        self.assertEqual(merged_ab["exact_digest"], merged_ba["exact_digest"])
        self.assertEqual(stable_payload_digest(rows_ab), stable_payload_digest(rows_ba))
        self.assertEqual(
            merged_ab["dataset"]["dataset_digest"],
            merged_ba["dataset"]["dataset_digest"],
        )

    def test_conflicting_duplicate_branch_id_fails(self) -> None:
        inputs = _synthetic_inputs()
        full_report, _rows = build_carrion_sequence_context_archive_report(
            **inputs,
            prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
        )
        original = full_report["branch_results"][0]
        conflicting = deepcopy(original)
        conflicting["source_rows"][0]["trainable"]["label"]["action"] = "stay"
        shard_a = _shard_report(inputs, [original], shard_id="shard-a")
        shard_b = _shard_report(inputs, [conflicting], shard_id="shard-b")

        with self.assertRaisesRegex(
            CarrionSequenceContextArchiveError,
            "duplicate branch_id with different digest",
        ):
            merge_carrion_sequence_context_archive_shards(
                **inputs,
                shard_reports=[shard_a, shard_b],
            )

    def test_partial_shard_rejected_by_default(self) -> None:
        inputs = _synthetic_inputs()
        full_report, _rows = build_carrion_sequence_context_archive_report(
            **inputs,
            prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
        )
        shard = _shard_report(
            inputs,
            full_report["branch_results"][:1],
            shard_id="partial",
            partial=True,
        )

        with self.assertRaisesRegex(
            CarrionSequenceContextArchiveError,
            "partial shard evidence requires explicit partial merge",
        ):
            merge_carrion_sequence_context_archive_shards(
                **inputs,
                shard_reports=[shard],
            )

    def test_shard_id_appears_in_generation_evidence_and_status(self) -> None:
        inputs = _synthetic_inputs()

        report, _rows = build_carrion_sequence_context_archive_report(
            **inputs,
            prior_context_by_branch_id=_prior_contexts(inputs["archive_report"]),
            seed_include=(29,),
            branch_index_include=(0,),
            shard_id="unit-shard",
        )

        self.assertEqual(report["inputs"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_status"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_evidence"]["shard_id"], "unit-shard")


def _synthetic_inputs() -> dict[str, object]:
    branch_results = [
        _archive_branch_result(
            source["source_branch_id"],
            seed=int(source["source_seed"]),
            branch_reason=str(source["source_branch_reason"]),
            action=str(source["label_action"]),
            harmful=True,
        )
        for source in DEFAULT_HARMFUL_SUPPORT_SOURCES
    ]
    branch_results.extend(
        [
            _archive_branch_result(
                "m3-carrion-specific-archive-seed-13-post-carrion-hydration-risk-branch-1-tick-11-agent-14",
                seed=13,
                branch_reason="post_carrion_hydration_risk",
                action="move_north",
            ),
            _archive_branch_result(
                "m3-carrion-specific-archive-seed-37-post-carrion-hydration-risk-branch-1-tick-15-agent-13",
                seed=37,
                branch_reason="post_carrion_hydration_risk",
                action="move_east",
            ),
            _archive_branch_result(
                "m3-carrion-specific-archive-seed-43-post-carrion-hydration-risk-branch-1-tick-11-agent-14",
                seed=43,
                branch_reason="post_carrion_hydration_risk",
                action="move_south",
            ),
        ]
    )
    dataset_rows = [
        _dataset_row(index, branch_result)
        for index, branch_result in enumerate(branch_results)
    ]
    archive_report = _archive_report(branch_results, dataset_rows)
    train_eval_report = _train_eval_report(archive_report, dataset_rows)
    autopsy_report = _autopsy_report(archive_report, dataset_rows, train_eval_report)
    return {
        "archive_report": archive_report,
        "dataset_rows": dataset_rows,
        "train_eval_report": train_eval_report,
        "autopsy_report": autopsy_report,
    }


def _archive_report(
    branch_results: list[dict[str, object]],
    dataset_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION,
        "policy": M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_POLICY,
        "diagnostics_only": True,
        "source_integrity": {"passed": True, "failures": []},
        "generation_status": {"state": "complete", "partial": False},
        "dataset": {
            "dataset_digest": stable_payload_digest(dataset_rows),
            "safe_label_count": len(dataset_rows),
        },
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
    }


def _train_eval_report(
    archive_report: dict[str, object],
    dataset_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": V149_CARRION_TRAIN_EVAL_SCHEMA_VERSION,
        "policy": V149_CARRION_TRAIN_EVAL_POLICY,
        "diagnostics_only": True,
        "validation": {
            "passed": True,
            "failures": [],
            "dataset_digest": stable_payload_digest(dataset_rows),
            "branch_evidence_digest": archive_report["branch_evidence_digest"],
        },
        "classification": {
            "primary": "m3_safe_archive_diagnostic_failed_non_promotional",
            "labels": ["m3_safe_archive_diagnostic_failed_non_promotional"],
        },
        "acceptance": {
            "passed": False,
            "blockers": [
                {
                    "reason": "unit_test_non_promotional_blocker",
                    "fixture": "carrion_only",
                    "seed": 13,
                }
            ],
        },
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }


def _autopsy_report(
    archive_report: dict[str, object],
    dataset_rows: list[dict[str, object]],
    train_eval_report: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
        "policy": M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
        "diagnostics_only": True,
        "classification": {
            "primary": "m3_carrion_archive_override_autopsy_complete_no_training",
            "labels": ["m3_carrion_archive_override_autopsy_complete_no_training"],
        },
        "inputs": {
            "dataset_digest": stable_payload_digest(dataset_rows),
            "branch_evidence_digest": archive_report["branch_evidence_digest"],
            "train_eval_report_digest": stable_payload_digest(train_eval_report),
            "input_validation": {
                "passed": True,
                "failures": [],
                "policy": "unit_v150_input_validation",
            },
        },
        "failure_mode_classification": {
            "primary": "missing_hydration_reproduction_context",
            "labels": [
                "missing_hydration_reproduction_context",
                "stale_one_step_aliasing",
            ],
        },
        "recommended_next_route": (
            "build_carrion_hydration_reproduction_sequence_context_archive"
        ),
        "support_label_action_failure_map": [
            {
                **dict(source),
                "autopsy_tags": [
                    "birth_regression_associated",
                    "missing_hydration_reproduction_context",
                    "stale_one_step_aliasing",
                ],
            }
            for source in DEFAULT_HARMFUL_SUPPORT_SOURCES
        ],
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }


def _archive_branch_result(
    branch_id: str,
    *,
    seed: int,
    branch_reason: str,
    action: str,
    harmful: bool = False,
) -> dict[str, object]:
    branch_index = 0 if branch_reason == "carrion_contact" else 2
    branch_tick = 0 if harmful else 11 + (seed % 5)
    observation = {"schema_version": "unit", "data": [seed / 100.0, branch_tick / 100.0]}
    return {
        "branch_id": branch_id,
        "fixture": "carrion_only",
        "seed": int(seed),
        "ticks": 120,
        "branch_index": int(branch_index),
        "branch_tick": int(branch_tick),
        "record_index": int(branch_tick + seed),
        "agent_id": int(seed % 17),
        "public_features": {
            "observation_input": observation,
            "action_mask": _action_mask(action),
        },
        "carrion_archive_context": {
            "branch_reason": branch_reason,
            "fixture": "carrion_only",
            "policy": "carrion_contact_hydration_stall_terminal_branch_selection_v1",
        },
    }


def _dataset_row(index: int, branch_result: dict[str, object]) -> dict[str, object]:
    action = next(
        action
        for action, allowed in branch_result["public_features"]["action_mask"].items()
        if allowed
    )
    replay = {
        "verified": True,
        "expected_digest": f"{index:064x}",
        "actual_digest": f"{index:064x}",
    }
    return {
        "schema_version": "m3_safe_archive_expansion_dataset_row_v1",
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {
                "observation_input": branch_result["public_features"][
                    "observation_input"
                ],
                "action_mask": branch_result["public_features"]["action_mask"],
            },
            "label": {"action": action},
        },
        "metadata": {
            "row_index": int(index),
            "source_branch_result_index": int(index),
            "seed": branch_result["seed"],
            "fixture": "carrion_only",
            "branch_id": branch_result["branch_id"],
            "branch_tick": branch_result["branch_tick"],
            "record_index": branch_result["record_index"],
            "agent_id": branch_result["agent_id"],
            "safety_vet": {"passed": True, "failures": []},
            "outcome_evidence": {"replay_verification": replay},
        },
    }


def _prior_contexts(archive_report: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    contexts = {}
    for result in archive_report["branch_results"]:
        branch_id = str(result["branch_id"])
        if int(result["branch_tick"]) == 0:
            contexts[branch_id] = []
        else:
            contexts[branch_id] = [
                {
                    "requested_action": "drink",
                    "resolved_action": "drink",
                    "action_valid": True,
                    "resolution_action_valid": True,
                    "moved": False,
                    "drank": True,
                    "ate": False,
                    "carcass_food": False,
                    "fresh_kill_food": False,
                    "animal_food": False,
                    "plant_food": False,
                    "gain_present": False,
                    "reproduced": False,
                    "reproduction_ready_after": True,
                    "died_after_action": False,
                }
            ]
    return contexts


def _shard_report(
    inputs: dict[str, object],
    branch_results: list[dict[str, object]],
    *,
    shard_id: str,
    partial: bool = False,
) -> dict[str, object]:
    report, _rows = build_carrion_sequence_context_archive_report(
        **inputs,
        branch_results=branch_results,
        shard_id=shard_id,
        generation_status={
            "policy": "test_v151_shard_generation_status_v1",
            "state": "partial" if partial else "complete",
            "partial": bool(partial),
            "stop_reason": "unit_test_partial" if partial else None,
            "target_fixture": "carrion_only",
            "target_carrion_seeds": [int(seed) for seed in TARGET_CARRION_SEEDS],
            "ticks": 120,
            "history_window": 8,
            "min_prior_public_steps": 0,
            "branch_result_count": len(branch_results),
            "shard_id": shard_id,
        },
        generation_evidence={
            "policy": "test_v151_shard_generation_evidence_v1",
            "shard_id": shard_id,
        },
    )
    return report


def _action_mask(action: str) -> dict[str, bool]:
    return {name: name == action for name in ACTION_NAMES}


if __name__ == "__main__":
    unittest.main()
