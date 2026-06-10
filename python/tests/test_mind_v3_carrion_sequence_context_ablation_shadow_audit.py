from __future__ import annotations

from copy import deepcopy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.carrion_sequence_context_ablation_shadow_audit import (
    M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION,
    CarrionSequenceContextAblationShadowAuditError,
    build_carrion_sequence_context_ablation_shadow_audit_report,
    merge_carrion_sequence_context_ablation_shadow_audit_shards,
)
from evolution_sim.mind.carrion_sequence_context_archive import (
    DEFAULT_HARMFUL_SUPPORT_SOURCES,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSequenceContextAblationShadowAuditTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-sequence-context-ablation-shadow-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_sequence_context_ablation_shadow_audit"
            ),
        )

    def test_classifies_support_limited_when_one_step_and_length_separate_markers_do_not(
        self,
    ) -> None:
        inputs = _synthetic_v151_inputs()

        report = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
        )

        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SEQUENCE_CONTEXT_ABLATION_SHADOW_AUDIT_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["source_integrity"]["leakage_scan_passed"])
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_sequence_context_ablation_"
                "one_step_and_prior_presence_support_limited_no_training"
            ),
        )
        comparator = report["comparator_availability"]
        self.assertEqual(comparator["same_label_action_available_count"], 3)
        self.assertEqual(comparator["same_action_exact_mask_available_count"], 3)
        self.assertEqual(
            comparator[
                "same_action_same_harmful_source_branch_reason_available_count"
            ],
            2,
        )
        self.assertEqual(comparator["zero_prior_safe_available_count"], 0)
        self.assertEqual(comparator["nonzero_prior_harmful_available_count"], 0)

        modes = report["ablations"]["modes"]
        self.assertTrue(
            modes["one_step_only_features"]["all_harmful_sources_separated"]
        )
        self.assertTrue(
            modes["prior_length_only_features"][
                "all_harmful_sources_separated"
            ]
        )
        self.assertFalse(
            modes["public_hydration_reproduction_marker_features"][
                "all_harmful_sources_separated"
            ]
        )
        self.assertTrue(
            modes["full_public_prior_sequence_features"][
                "all_harmful_sources_separated"
            ]
        )
        self.assertTrue(
            report["ablations"][
                "coarse_prior_presence_explains_full_sequence_separation"
            ]
        )
        self.assertTrue(
            report["ablations"][
                "one_step_support_limited_explains_full_sequence_separation"
            ]
        )
        self.assertEqual(report["shadow_live_probe"]["status"], "not_run")

    def test_rejects_v151_classification_mismatch(self) -> None:
        inputs = _synthetic_v151_inputs()
        report = deepcopy(inputs["v151_report"])
        report["classification"] = {
            "primary": "m3_carrion_sequence_context_archive_no_sequence_separation_no_training",
            "labels": [
                "m3_carrion_sequence_context_archive_no_sequence_separation_no_training"
            ],
        }
        inputs["v151_report"] = report

        with self.assertRaisesRegex(
            CarrionSequenceContextAblationShadowAuditError,
            "v151_classification_not_sequence_separated",
        ):
            build_carrion_sequence_context_ablation_shadow_audit_report(**inputs)

    def test_rejects_v151_partial_generation(self) -> None:
        inputs = _synthetic_v151_inputs()
        report = deepcopy(inputs["v151_report"])
        report["generation_status"] = {
            **report["generation_status"],
            "state": "partial",
            "partial": True,
        }
        inputs["v151_report"] = report

        with self.assertRaisesRegex(
            CarrionSequenceContextAblationShadowAuditError,
            "v151_generation_not_complete",
        ):
            build_carrion_sequence_context_ablation_shadow_audit_report(**inputs)

    def test_shard_id_appears_in_generation_evidence_status_and_inputs(self) -> None:
        inputs = _synthetic_v151_inputs()

        report = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_include=(0, 1),
            shard_id="unit-shard",
        )

        self.assertEqual(report["inputs"]["shard"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_status"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_evidence"]["shard_id"], "unit-shard")
        self.assertTrue(report["source_integrity"]["passed"])

    def test_synthetic_two_shard_merge_succeeds(self) -> None:
        inputs = _synthetic_v151_inputs()
        shard_a = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_start=0,
            row_index_count=5,
            shard_id="shard-a",
        )
        shard_b = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_start=5,
            row_index_count=4,
            shard_id="shard-b",
        )

        merged = merge_carrion_sequence_context_ablation_shadow_audit_shards(
            **inputs,
            shard_reports=[shard_a, shard_b],
        )

        self.assertTrue(merged["source_integrity"]["passed"])
        self.assertEqual(merged["audit_row_count"], 9)
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
        inputs = _synthetic_v151_inputs()
        shard_a = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_start=0,
            row_index_count=5,
            shard_id="shard-a",
        )
        shard_b = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_start=5,
            row_index_count=4,
            shard_id="shard-b",
        )

        merged_ab = merge_carrion_sequence_context_ablation_shadow_audit_shards(
            **inputs,
            shard_reports=[shard_a, shard_b],
        )
        merged_ba = merge_carrion_sequence_context_ablation_shadow_audit_shards(
            **inputs,
            shard_reports=[shard_b, shard_a],
        )

        self.assertEqual(merged_ab["exact_digest"], merged_ba["exact_digest"])
        self.assertEqual(
            merged_ab["audit_row_digest"],
            merged_ba["audit_row_digest"],
        )

    def test_conflicting_duplicate_audit_row_fails(self) -> None:
        inputs = _synthetic_v151_inputs()
        shard_a = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_include=(0,),
            shard_id="shard-a",
        )
        shard_b = deepcopy(shard_a)
        shard_b["inputs"]["shard"]["shard_id"] = "shard-b"
        shard_b["audit_rows"][0]["trainable"]["features"]["observation_input"] = {
            "schema_version": "unit",
            "data": [999.0],
        }
        shard_b["audit_row_digest"] = stable_payload_digest(shard_b["audit_rows"])

        with self.assertRaisesRegex(
            CarrionSequenceContextAblationShadowAuditError,
            "duplicate audit row with different digest",
        ):
            merge_carrion_sequence_context_ablation_shadow_audit_shards(
                **inputs,
                shard_reports=[shard_a, shard_b],
            )

    def test_partial_merge_rejected_by_default(self) -> None:
        inputs = _synthetic_v151_inputs()
        shard = build_carrion_sequence_context_ablation_shadow_audit_report(
            **inputs,
            row_index_start=0,
            row_index_count=2,
            shard_id="partial",
        )

        with self.assertRaisesRegex(
            CarrionSequenceContextAblationShadowAuditError,
            "partial shard evidence requires explicit partial merge",
        ):
            merge_carrion_sequence_context_ablation_shadow_audit_shards(
                **inputs,
                shard_reports=[shard],
            )


def _synthetic_v151_inputs() -> dict[str, object]:
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
        "inputs": {
            "target_carrion_seeds": [13, 19, 29, 37, 41, 43],
            "ticks": 120,
            "history_window": 8,
            "min_prior_public_steps": 0,
        },
        "source_integrity": {"passed": True, "failures": []},
        "generation_status": {"state": "complete", "partial": False},
        "classification": {
            "primary": "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training",
            "labels": [
                "m3_carrion_sequence_context_archive_separates_harmful_sources_no_training"
            ],
        },
        "leakage_scan": {"passed": True, "failures": []},
        "replay_verification": {"complete": True},
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
    return {"v151_report": v151_report, "v151_rows": rows}


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
