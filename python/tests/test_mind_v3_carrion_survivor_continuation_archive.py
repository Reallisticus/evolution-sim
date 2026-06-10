from __future__ import annotations

from copy import deepcopy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.carrion_archive_override_autopsy import (
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
    M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
    M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
    CarrionSurvivorContinuationArchiveError,
    build_carrion_survivor_continuation_archive,
    merge_carrion_survivor_continuation_archive_shards,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationArchiveTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-archive"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_archive"
            ),
        )

    def test_report_accepts_replay_verified_survivor_continuation_labels(self) -> None:
        inputs = _synthetic_inputs()

        report, rows = build_carrion_survivor_continuation_archive(
            **inputs,
            min_label_count=4,
            branch_result_chunk_dir=None,
        )

        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["replay_verification"]["complete"])
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertEqual(report["label_count"], 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            report["action_distribution"]["dominant_label_action_share"],
            0.5,
        )
        self.assertEqual(
            report["classification"]["primary"],
            "m3_carrion_survivor_continuation_archive_support_ready_no_training",
        )
        feature_text = json.dumps(
            [row["trainable"]["features"] for row in rows],
            sort_keys=True,
        )
        for forbidden in ("seed", "fixture", "branch", "tick", "agent", "digest"):
            self.assertNotIn(forbidden, feature_text)

    def test_rejects_v153_classification_mismatch(self) -> None:
        inputs = _synthetic_inputs()
        v153 = deepcopy(inputs["v153_report"])
        v153["classification"]["primary"] = "unexpected"
        inputs["v153_report"] = v153

        with self.assertRaisesRegex(
            CarrionSurvivorContinuationArchiveError,
            "v153_unexpected_classification",
        ):
            build_carrion_survivor_continuation_archive(
                **inputs,
                min_label_count=4,
                branch_result_chunk_dir=None,
            )

    def test_blacklist_excludes_invalid_resolution_risk_label(self) -> None:
        inputs = _synthetic_inputs()
        inputs["v145_report"] = {
            "route_decision": {
                "label_blacklist": [
                    {
                        "seed": 13,
                        "agent_id": 1,
                        "residual_action": "eat",
                        "branch_id": "branch-13-0",
                    }
                ]
            }
        }

        report, rows = build_carrion_survivor_continuation_archive(
            **inputs,
            min_label_count=3,
            branch_result_chunk_dir=None,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(report["blacklist"]["blacklist_hit_count"], 1)
        self.assertIn(
            "invalid_resolution_risk_blacklist",
            {row["excluded_reason"] for row in report["excluded_rows"]},
        )

    def test_shard_id_appears_in_generation_evidence_status_and_inputs(self) -> None:
        inputs = {
            **_base_sources(),
            "continuation_branch_results": _branch_results(seeds=(13,)),
        }

        report, _ = build_carrion_survivor_continuation_archive(
            **inputs,
            target_seeds=(13, 19),
            seed_include=(13,),
            shard_id="unit-shard",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )

        self.assertEqual(report["inputs"]["shard"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_status"]["shard_id"], "unit-shard")
        self.assertEqual(report["generation_evidence"]["shard_id"], "unit-shard")
        self.assertTrue(report["source_integrity"]["passed"])

    def test_synthetic_two_shard_merge_succeeds(self) -> None:
        base = _base_sources()
        shard_a, _ = build_carrion_survivor_continuation_archive(
            **base,
            continuation_branch_results=_branch_results(seeds=(13,)),
            target_seeds=(13, 19),
            seed_include=(13,),
            shard_id="shard-a",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )
        shard_b, _ = build_carrion_survivor_continuation_archive(
            **base,
            continuation_branch_results=_branch_results(seeds=(19,)),
            target_seeds=(13, 19),
            seed_include=(19,),
            shard_id="shard-b",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )

        merged, rows = merge_carrion_survivor_continuation_archive_shards(
            **base,
            shard_reports=[shard_a, shard_b],
            target_seeds=(13, 19),
        )

        self.assertEqual(len(rows), 4)
        self.assertTrue(merged["source_integrity"]["passed"])
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
        base = _base_sources()
        shard_a, _ = build_carrion_survivor_continuation_archive(
            **base,
            continuation_branch_results=_branch_results(seeds=(13,)),
            target_seeds=(13, 19),
            seed_include=(13,),
            shard_id="shard-a",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )
        shard_b, _ = build_carrion_survivor_continuation_archive(
            **base,
            continuation_branch_results=_branch_results(seeds=(19,)),
            target_seeds=(13, 19),
            seed_include=(19,),
            shard_id="shard-b",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )

        merged_ab, _ = merge_carrion_survivor_continuation_archive_shards(
            **base,
            shard_reports=[shard_a, shard_b],
            target_seeds=(13, 19),
        )
        merged_ba, _ = merge_carrion_survivor_continuation_archive_shards(
            **base,
            shard_reports=[shard_b, shard_a],
            target_seeds=(13, 19),
        )

        self.assertEqual(merged_ab["exact_digest"], merged_ba["exact_digest"])
        self.assertEqual(
            merged_ab["dataset"]["dataset_digest"],
            merged_ba["dataset"]["dataset_digest"],
        )

    def test_conflicting_duplicate_branch_id_fails(self) -> None:
        base = _base_sources()
        shard_a, _ = build_carrion_survivor_continuation_archive(
            **base,
            continuation_branch_results=_branch_results(seeds=(13,)),
            target_seeds=(13,),
            seed_include=(13,),
            shard_id="shard-a",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )
        shard_b = deepcopy(shard_a)
        shard_b["inputs"]["shard"]["shard_id"] = "shard-b"
        shard_b["continuation_branch_results"][0]["continuation_runs"][0][
            "alive_agents"
        ] = 99
        shard_b["continuation_branch_evidence_digest"] = stable_payload_digest(
            shard_b["continuation_branch_results"]
        )

        with self.assertRaisesRegex(
            CarrionSurvivorContinuationArchiveError,
            "duplicate branch_id with different digest",
        ):
            merge_carrion_survivor_continuation_archive_shards(
                **base,
                shard_reports=[shard_a, shard_b],
                target_seeds=(13,),
            )

    def test_partial_shard_rejected_by_default(self) -> None:
        base = _base_sources()
        shard, _ = build_carrion_survivor_continuation_archive(
            **base,
            continuation_branch_results=_branch_results(seeds=(13,)),
            target_seeds=(13, 19),
            seed_include=(13,),
            shard_id="partial",
            min_label_count=1,
            branch_result_chunk_dir=None,
        )

        with self.assertRaisesRegex(
            CarrionSurvivorContinuationArchiveError,
            "partial shard evidence requires explicit partial merge",
        ):
            merge_carrion_survivor_continuation_archive_shards(
                **base,
                shard_reports=[shard],
                target_seeds=(13, 19),
            )


def _synthetic_inputs(seeds: tuple[int, ...] = (13, 19)) -> dict[str, object]:
    return {
        **_base_sources(),
        "continuation_branch_results": _branch_results(seeds=seeds),
        "target_seeds": seeds,
    }


def _base_sources() -> dict[str, object]:
    return {
        "autopsy_report": _v150_report(),
        "v153_report": _v153_report(),
        "v145_report": {"route_decision": {"label_blacklist": []}},
    }


def _v150_report() -> dict[str, object]:
    return {
        "schema_version": M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_SCHEMA_VERSION,
        "policy": M3_CARRION_ARCHIVE_OVERRIDE_AUTOPSY_POLICY,
        "classification": {
            "primary": "m3_carrion_archive_override_autopsy_complete_no_training"
        },
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "exact_digest": "v150-unit",
    }


def _v153_report() -> dict[str, object]:
    return {
        "schema_version": (
            M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SEQUENCE_CONTEXT_COMPARATOR_SUPPORT_CLOSEOUT_POLICY,
        "classification": {
            "primary": (
                "m3_carrion_sequence_context_comparator_support_limited_closed_no_training"
            )
        },
        "source_integrity": {"passed": True, "failures": []},
        "recommendation": {"route_closed": True},
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "runtime_action_selection_changed": False,
        "exact_digest": "v153-unit",
    }


def _branch_results(*, seeds: tuple[int, ...]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for seed in seeds:
        if seed == 13:
            results.append(_branch(seed=13, branch_index=0, agent_id=1, action="eat"))
            results.append(_branch(seed=13, branch_index=1, agent_id=2, action="drink"))
        elif seed == 19:
            results.append(_branch(seed=19, branch_index=0, agent_id=3, action="eat"))
            results.append(_branch(seed=19, branch_index=1, agent_id=4, action="drink"))
    return results


def _branch(*, seed: int, branch_index: int, agent_id: int, action: str) -> dict[str, object]:
    branch_id = f"branch-{seed}-{branch_index}"
    run = _run(branch_id=branch_id, seed=seed, agent_id=agent_id, action=action)
    return {
        "branch_id": branch_id,
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "record_index": branch_index,
        "branch_index": branch_index,
        "agent_id": agent_id,
        "baseline_action": "stay",
        "v142_requested_action": "stay",
        "v142_resolved_action": "stay",
        "candidate_actions": [action],
        "candidate_action_count": 1,
        "continuation_indexes": [0],
        "continuation_count": 1,
        "branch_state_digest": f"digest-{branch_id}",
        "source_trajectory_path": None,
        "public_features": {
            "observation_input": {
                "schema_version": "unit_public_observation_v1",
                "values": [0.1, 0.2, 0.3],
            },
            "action_mask": {action: True},
        },
        "carrion_archive_context": {
            "policy": "unit",
            "fixture": "carrion_only",
            "branch_reason": (
                "carrion_contact" if branch_index == 0 else "movement_stall"
            ),
            "reason_rank": branch_index,
            "reason_evidence": {},
        },
        "continuation_runs": [run],
        "diagnostics_only": True,
    }


def _run(*, branch_id: str, seed: int, agent_id: int, action: str) -> dict[str, object]:
    payload = {
        "branch_id": branch_id,
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "record_index": 0,
        "agent_id": agent_id,
        "baseline_action": "stay",
        "v142_requested_action": "stay",
        "forced_action": action,
        "forced_action_used": True,
        "forced_action_supported": True,
        "continuation_index": 0,
        "continuation_name": "delegate_after_label",
        "continuation_tail_actions": [],
        "planned_action_sequence": [action],
        "forced_action_sequence_used": [
            {"sequence_index": 0, "action": action, "public_mask_legal": True}
        ],
        "skipped_continuation_actions": [],
        "forced_sequence_action_count": 1,
        "skipped_sequence_action_count": 0,
        "continuation_forced_action_source_count": 1,
        "ticks_executed": 120,
        "alive_agents": 2,
        "births": 0,
        "deaths": 0,
        "target_terminal": {
            "alive": True,
            "energy_ratio": 0.8,
            "hydration_ratio": 0.7,
            "health_ratio": 0.9,
        },
        "target_alive_at_end": True,
        "target_energy_ratio_at_end": 0.8,
        "target_hydration_ratio_at_end": 0.7,
        "target_health_ratio_at_end": 0.9,
        "first_action_outcome": {
            "requested_action": action,
            "resolved_action": action,
            "action_valid": True,
            "resolution_action_valid": True,
            "matches_forced_action": True,
            "outcome": {
                "reproduced": False,
                "reproduction_ready_after": False,
            },
        },
        "trajectory_record_count": 1,
        "heuristic_action_source_count": 0,
        "diagnostic_forced_action_source_count": 0,
        "requested_action_counts": {action: 1},
        "resolved_action_counts": {action: 1},
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "dominant_requested_action": action,
        "dominant_requested_action_share": 1.0,
        "action_source_counts": {
            f"carrion_survivor_continuation_force:0:0:{action}": 1
        },
        "policy_id_counts": {"mind_v3_v154_carrion_survivor_continuation_force": 1},
        "deltas_vs_baseline": {
            "alive_agents": 1,
            "births": 0,
            "deaths": -1,
            "target_alive": 1,
            "target_energy_ratio": 0.2,
            "target_hydration_ratio": 0.2,
            "target_health_ratio": 0.1,
            "requested_action_counts": {},
            "resolved_action_counts": {},
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
        },
        "deltas_vs_v142_override": {
            "alive_agents": 1,
            "births": 0,
            "deaths": -1,
            "target_alive": 1,
            "target_energy_ratio": 0.2,
            "target_hydration_ratio": 0.2,
            "target_health_ratio": 0.1,
            "requested_action_counts": {},
            "resolved_action_counts": {},
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
        },
        "outcome_improvement": {
            "policy": "m3_carrion_survivor_continuation_outcome_improvement_v1",
            "improved": True,
            "labels": ["survival", "hydration_recovery", "fewer_blockers"],
            "survival_improved": True,
            "hydration_recovery": True,
            "reproduction_readiness": False,
            "fewer_blockers": True,
            "no_resolved_invalid_increase": True,
        },
    }
    digest = stable_payload_digest(payload)
    payload["replay_digest"] = digest
    payload["replay_verification"] = {
        "verified": True,
        "expected_digest": digest,
        "actual_digest": digest,
    }
    payload["continuation_run_id"] = f"{branch_id}::action={action}::continuation=0"
    return payload


if __name__ == "__main__":
    unittest.main()
