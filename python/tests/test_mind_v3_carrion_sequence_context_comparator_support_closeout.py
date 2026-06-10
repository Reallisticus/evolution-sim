from __future__ import annotations

from copy import deepcopy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.carrion_sequence_context_ablation_shadow_audit import (
    M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_sequence_context_archive import (
    DEFAULT_HARMFUL_SUPPORT_SOURCES,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
    CarrionSequenceContextComparatorSupportCloseoutError,
    build_carrion_sequence_context_comparator_support_closeout_report,
    merge_carrion_sequence_context_comparator_support_closeout_shards,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSequenceContextComparatorSupportCloseoutTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-sequence-context-comparator-support-closeout"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_sequence_context_comparator_support_closeout"
            ),
        )

    def test_report_closes_sequence_context_route_when_strong_comparators_missing(
        self,
    ) -> None:
        inputs = _synthetic_inputs()

        report = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            comparator_chunk_dir=None,
        )

        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertEqual(report["coverage"]["comparator_evidence_row_count"], 27)
        support = report["comparator_support"]
        self.assertEqual(support["exact_one_step_safe_comparator_total"], 0)
        self.assertEqual(support["zero_prior_safe_comparator_total"], 0)
        self.assertEqual(support["nonzero_prior_harmful_comparator_total"], 0)
        self.assertGreater(
            support["same_label_action_exact_mask_safe_comparator_total"],
            0,
        )
        self.assertEqual(
            report["classification"]["primary"],
            "m3_carrion_sequence_context_comparator_support_limited_closed_no_training",
        )
        self.assertTrue(report["recommendation"]["route_closed"])
        self.assertEqual(
            report["recommendation"]["recommended_next_route"],
            "return_to_carrion_recovery_archive_or_go_explore_survivor_continuation_evidence",
        )
        feature_text = json.dumps(
            [row["trainable"]["features"] for row in report["comparator_evidence_rows"]],
            sort_keys=True,
        )
        for forbidden in ("seed", "fixture", "branch", "path", "digest", "private"):
            self.assertNotIn(forbidden, feature_text)

    def test_rejects_unexpected_v152_classification(self) -> None:
        inputs = _synthetic_inputs()
        v152 = deepcopy(inputs["v152_report"])
        v152["classification"]["primary"] = "unexpected"
        inputs["v152_report"] = v152

        with self.assertRaisesRegex(
            CarrionSequenceContextComparatorSupportCloseoutError,
            "v152_unexpected_classification",
        ):
            build_carrion_sequence_context_comparator_support_closeout_report(
                **inputs,
                comparator_chunk_dir=None,
            )

    def test_shard_id_appears_in_generation_evidence_status_and_inputs(self) -> None:
        inputs = _synthetic_inputs()

        report = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_include=(0, 1),
            shard_id="unit-shard",
            comparator_chunk_dir=None,
        )

        self.assertEqual(report["inputs"]["shard"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_status"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_evidence"]["shard_id"], "unit-shard")
        self.assertTrue(report["source_integrity"]["passed"])

    def test_synthetic_two_shard_merge_succeeds(self) -> None:
        inputs = _synthetic_inputs()
        shard_a = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_start=0,
            row_index_count=5,
            shard_id="shard-a",
            comparator_chunk_dir=None,
        )
        shard_b = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_start=5,
            row_index_count=4,
            shard_id="shard-b",
            comparator_chunk_dir=None,
        )

        merged = merge_carrion_sequence_context_comparator_support_closeout_shards(
            **inputs,
            shard_reports=[shard_a, shard_b],
        )

        self.assertTrue(merged["source_integrity"]["passed"])
        self.assertEqual(merged["coverage"]["comparator_evidence_row_count"], 27)
        self.assertEqual(len(merged["shard_merge"]["sources"]), 2)
        self.assertEqual(
            {
                source["shard_id"]
                for source in merged["shard_merge"]["sources"]
                if source.get("shard_id")
            },
            {"shard-a", "shard-b"},
        )

    def test_reversed_shard_order_yields_same_exact_digest(self) -> None:
        inputs = _synthetic_inputs()
        shard_a = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_start=0,
            row_index_count=5,
            shard_id="shard-a",
            comparator_chunk_dir=None,
        )
        shard_b = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_start=5,
            row_index_count=4,
            shard_id="shard-b",
            comparator_chunk_dir=None,
        )

        merged_ab = merge_carrion_sequence_context_comparator_support_closeout_shards(
            **inputs,
            shard_reports=[shard_a, shard_b],
        )
        merged_ba = merge_carrion_sequence_context_comparator_support_closeout_shards(
            **inputs,
            shard_reports=[shard_b, shard_a],
        )

        self.assertEqual(merged_ab["exact_digest"], merged_ba["exact_digest"])
        self.assertEqual(
            merged_ab["comparator_evidence_digest"],
            merged_ba["comparator_evidence_digest"],
        )

    def test_conflicting_duplicate_evidence_row_fails(self) -> None:
        inputs = _synthetic_inputs()
        shard_a = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_include=(0,),
            shard_id="shard-a",
            comparator_chunk_dir=None,
        )
        shard_b = deepcopy(shard_a)
        shard_b["inputs"]["shard"]["shard_id"] = "shard-b"
        shard_b["comparator_evidence_rows"][0]["trainable"]["features"][
            "observation_input"
        ] = {"schema_version": "unit", "data": [999.0]}
        shard_b["comparator_evidence_digest"] = stable_payload_digest(
            shard_b["comparator_evidence_rows"]
        )

        with self.assertRaisesRegex(
            CarrionSequenceContextComparatorSupportCloseoutError,
            "duplicate comparator evidence with different digest",
        ):
            merge_carrion_sequence_context_comparator_support_closeout_shards(
                **inputs,
                shard_reports=[shard_a, shard_b],
            )

    def test_partial_merge_rejected_by_default(self) -> None:
        inputs = _synthetic_inputs()
        shard = build_carrion_sequence_context_comparator_support_closeout_report(
            **inputs,
            row_index_start=0,
            row_index_count=2,
            shard_id="partial",
            comparator_chunk_dir=None,
        )

        with self.assertRaisesRegex(
            CarrionSequenceContextComparatorSupportCloseoutError,
            "partial shard evidence requires explicit partial merge",
        ):
            merge_carrion_sequence_context_comparator_support_closeout_shards(
                **inputs,
                shard_reports=[shard],
            )


def _synthetic_inputs() -> dict[str, object]:
    rows = [
        _row(
            0,
            action="move_south",
            seed=19,
            branch_reason="movement_stall",
            branch_id=DEFAULT_HARMFUL_SUPPORT_SOURCES[2]["source_branch_id"],
            harmful_source=DEFAULT_HARMFUL_SUPPORT_SOURCES[2],
            prior=(),
        ),
        _row(
            1,
            action="move_north",
            seed=29,
            branch_reason="carrion_contact",
            branch_id=DEFAULT_HARMFUL_SUPPORT_SOURCES[0]["source_branch_id"],
            harmful_source=DEFAULT_HARMFUL_SUPPORT_SOURCES[0],
            prior=(),
        ),
        _row(
            2,
            action="move_east",
            seed=41,
            branch_reason="movement_stall",
            branch_id=DEFAULT_HARMFUL_SUPPORT_SOURCES[1]["source_branch_id"],
            harmful_source=DEFAULT_HARMFUL_SUPPORT_SOURCES[1],
            prior=(),
        ),
        _row(
            3,
            action="move_south",
            seed=13,
            branch_reason="movement_stall",
            branch_id="safe-move-south-same-reason",
            prior=(_movement_only_step(),),
        ),
        _row(
            4,
            action="move_north",
            seed=37,
            branch_reason="terminal_extinction",
            branch_id="safe-move-north-different-reason",
            prior=(_movement_only_step(),),
        ),
        _row(
            5,
            action="move_east",
            seed=43,
            branch_reason="movement_stall",
            branch_id="safe-move-east-same-reason",
            prior=(_movement_only_step(),),
        ),
        _row(
            6,
            action="move_north",
            seed=13,
            branch_reason="post_carrion_hydration_risk",
            branch_id="safe-move-north-extra",
            prior=(_hydration_step(),),
        ),
        _row(
            7,
            action="move_east",
            seed=37,
            branch_reason="terminal_extinction",
            branch_id="safe-move-east-extra",
            prior=(_hydration_step(),),
        ),
        _row(
            8,
            action="move_south",
            seed=43,
            branch_reason="terminal_extinction",
            branch_id="safe-move-south-extra",
            prior=(_hydration_step(),),
        ),
    ]
    branch_results = [
        {
            "branch_id": row["metadata"]["branch_id"],
            "seed": row["metadata"]["seed"],
            "branch_index": row["metadata"]["branch_index"],
            "branch_reason": row["metadata"]["branch_reason"],
        }
        for row in rows
    ]
    v151_report = {
        "schema_version": M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
        "policy": M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY,
        "diagnostics_only": True,
        "inputs": {"history_window": 8, "min_prior_public_steps": 0},
        "generation_status": {"state": "complete", "partial": False},
        "source_integrity": {"passed": True, "failures": []},
        "leakage_scan": {"passed": True, "failures": []},
        "replay_verification": {"complete": True},
        "classification": {
            "primary": "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training",
            "labels": [
                "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training"
            ],
        },
        "dataset": {
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
        },
        "branch_results": branch_results,
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "exact_digest": "unit-v151-exact-digest",
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }
    v152_report = {
        "schema_version": (
            M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_POLICY,
        "diagnostics_only": True,
        "generation_status": {"state": "complete", "partial": False},
        "source_integrity": {"passed": True, "failures": []},
        "classification": {
            "primary": (
                "m3_carrion_sequence_context_ablation_one_step_and_prior_presence_support_limited_no_training"
            ),
            "labels": [
                "m3_carrion_sequence_context_ablation_one_step_and_prior_presence_support_limited_no_training"
            ],
        },
        "audit_row_count": len(rows),
        "audit_row_digest": stable_payload_digest(rows),
        "exact_digest": "unit-v152-exact-digest",
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
    }
    return {
        "v151_report": v151_report,
        "v151_rows": rows,
        "v152_report": v152_report,
    }


def _row(
    index: int,
    *,
    action: str,
    seed: int,
    branch_reason: str,
    branch_id: object,
    prior: tuple[dict[str, object], ...],
    harmful_source: object | None = None,
) -> dict[str, object]:
    row = {
        "schema_version": "m3_carrion_sequence_context_archive_dataset_row_v1",
        "trainable": {
            "feature_policy": (
                "public_observation_action_mask_with_prior_public_hydration_"
                "reproduction_sequence_context_v1"
            ),
            "features": {
                "observation_input": {
                    "schema_version": "unit",
                    "data": [float(seed) / 100.0, float(index) / 100.0],
                },
                "action_mask": _action_mask(action),
                "prior_public_sequence_context": list(prior),
            },
            "label": {"action": action},
        },
        "metadata": {
            "row_index": index,
            "source_dataset_row_index": index,
            "seed": seed,
            "fixture": "carrion_only",
            "branch_id": branch_id,
            "branch_index": 0 if branch_reason == "carrion_contact" else 2,
            "branch_reason": branch_reason,
            "label_action": action,
            "harmful_source": harmful_source is not None,
            "harmful_source_evidence": harmful_source,
            "hydration_reproduction_markers": _markers(prior),
        },
    }
    row["stable_source_row_digest"] = stable_payload_digest(row)
    return row


def _movement_only_step() -> dict[str, object]:
    return {
        "relative_step": -1,
        "requested_action": "move_west",
        "resolved_action": "move_west",
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": True,
        "drank": False,
        "ate": False,
        "carcass_food": False,
        "fresh_kill_food": False,
        "animal_food": False,
        "plant_food": False,
        "gain_present": False,
        "reproduced": False,
        "reproduction_ready_after": False,
        "died_after_action": False,
    }


def _hydration_step() -> dict[str, object]:
    step = _movement_only_step()
    step.update({"requested_action": "drink", "resolved_action": "drink", "drank": True})
    return step


def _markers(prior: tuple[dict[str, object], ...]) -> dict[str, int]:
    return {
        "prior_step_count": len(prior),
        "movement_count": sum(1 for step in prior if step.get("moved")),
        "drank_count": sum(1 for step in prior if step.get("drank")),
        "ate_count": sum(1 for step in prior if step.get("ate")),
        "animal_food_count": sum(1 for step in prior if step.get("animal_food")),
        "reproduced_count": sum(1 for step in prior if step.get("reproduced")),
        "reproduction_ready_after_count": sum(
            1 for step in prior if step.get("reproduction_ready_after")
        ),
    }


def _action_mask(action: str) -> dict[str, bool]:
    return {
        "move_north": action == "move_north",
        "move_south": action == "move_south",
        "move_east": action == "move_east",
        "move_west": action == "move_west",
        "eat": False,
        "drink": False,
        "reproduce": False,
        "attack": False,
        "share": False,
        "rest": False,
    }


if __name__ == "__main__":
    unittest.main()
