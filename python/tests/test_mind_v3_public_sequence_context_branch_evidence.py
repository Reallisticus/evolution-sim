from __future__ import annotations

from copy import deepcopy
import json
import unittest
from pathlib import Path

from evolution_sim.config import WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.candidate_campaign import (
    M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
    M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
    M3_SAFE_ARCHIVE_EXPANSION_REPORT_SCHEMA_VERSION,
)
from evolution_sim.mind.public_sequence_context_branch_evidence import (
    M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_SCHEMA_VERSION,
    PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY,
    PublicSequenceContextBranchEvidenceError,
    _point_from_record,
    build_public_sequence_context_branch_evidence_report,
    merge_public_sequence_context_branch_evidence_reports,
    public_sequence_context_for_branch_result_matches,
    public_sequence_context_trainable_leakage_scan,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

ROOT = Path(__file__).resolve().parents[2]


class MindV3PublicSequenceContextBranchEvidenceTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:public-sequence-context-branch-evidence"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_public_sequence_context_branch_evidence"
            ),
        )

    def test_synthetic_bp3_one_step_alias_is_sequence_separated(self) -> None:
        dataset_rows = [_bp3_dataset_row()]
        branch_results = [_branch_result()]

        report = build_public_sequence_context_branch_evidence_report(
            bp3_safe_archive_report=_bp3_report(dataset_rows),
            bp3_dataset_rows=dataset_rows,
            bp3_branch_evidence=_bp3_branch_evidence(),
            branch_results=branch_results,
            generation_status=_complete_status(branch_results),
        )

        self.assertEqual(
            report["schema_version"],
            M3_PUBLIC_SEQUENCE_CONTEXT_BRANCH_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertTrue(report["contract"]["diagnostics_only"])
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["source_integrity"]["passed"])
        aliasing = report["bp3_action_only_aliasing"]
        self.assertEqual(aliasing["action_only_alias_count"], 1)
        self.assertEqual(aliasing["sequence_separated_alias_count"], 1)
        self.assertEqual(aliasing["safe_action_only_alias_count"], 1)
        self.assertEqual(aliasing["safe_sequence_separated_alias_count"], 1)
        self.assertEqual(
            report["classification"]["primary"],
            "m3_public_sequence_context_branch_evidence_separates_bp3_aliases_no_training",
        )
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["authorization_block"]["training_authorized"])

    def test_missing_replay_verification_fails_closed(self) -> None:
        dataset_rows = [_bp3_dataset_row()]
        branch_result = _branch_result()
        branch_result["action_runs"][0]["replay_verification"] = None

        report = build_public_sequence_context_branch_evidence_report(
            bp3_safe_archive_report=_bp3_report(dataset_rows),
            bp3_dataset_rows=dataset_rows,
            bp3_branch_evidence=_bp3_branch_evidence(),
            branch_results=[branch_result],
            generation_status=_complete_status([branch_result]),
        )

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "missing_or_failed_branch_replay_verification",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "m3_public_sequence_context_branch_evidence_source_integrity_failed",
        )
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])

    def test_trainable_leakage_scan_rejects_identity_and_digest_fields(self) -> None:
        allowed = public_sequence_context_trainable_leakage_scan(
            [
                {
                    "feature_policy": PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY,
                    "features": {
                        "observation_input": _observation_input(),
                        "action_mask": _action_mask(),
                        "prior_public_transition_summaries": [_prior_summary()],
                    },
                }
            ]
        )
        leaking = public_sequence_context_trainable_leakage_scan(
            [
                {
                    "features": {
                        "seed": 13,
                        "branch_id": "branch-1",
                        "tick": 7,
                        "agent_id": 2,
                        "digest": "a" * 64,
                    }
                }
            ]
        )

        self.assertTrue(allowed["passed"])
        self.assertFalse(leaking["passed"])
        self.assertGreaterEqual(leaking["forbidden_failure_count"], 5)

    def test_resume_context_match_rejects_stale_public_sequence_context(self) -> None:
        branch_result = _branch_result()
        context = deepcopy(branch_result["public_sequence_context"]["trainable"])
        stale = deepcopy(context)
        stale["features"]["prior_public_transition_summaries"][0][
            "requested_action"
        ] = "eat"

        self.assertTrue(
            public_sequence_context_for_branch_result_matches(branch_result, context)
        )
        self.assertFalse(
            public_sequence_context_for_branch_result_matches(branch_result, stale)
        )

    def test_point_from_record_preserves_requested_branch_index(self) -> None:
        world = SimulationWorld(
            WorldConfig(
                seed=5,
                width=8,
                height=8,
                max_ticks=3,
                initial_agents=2,
                max_agents=8,
            ),
            policy=MindV3EvolutionPolicy(seed=5),
        )
        record = {
            "agent_id": 1,
            "requested_action": "stay",
            "resolved_action": "stay",
            "action_mask": _action_mask(),
            "observation_input": _observation_input(),
            "observation_schema": "unit",
            "observation_digest": "d" * 64,
        }

        point = _point_from_record(
            fixture="broad",
            seed=5,
            ticks=3,
            tick=1,
            record_index=0,
            branch_index=1,
            record=record,
            snapshot=world,
            prior_history=[_prior_summary()],
            history_window=3,
            min_prior_public_steps=1,
        )

        self.assertFalse(isinstance(point, dict))
        self.assertIsNotNone(point)
        self.assertEqual(point.point.branch_index, 1)
        self.assertIn("-branch-1-", point.point.branch_id)

    def test_merge_rejects_shard_without_top_level_diagnostics_contract(self) -> None:
        dataset_rows = [_bp3_dataset_row()]
        branch_results = [_branch_result()]
        shard = build_public_sequence_context_branch_evidence_report(
            bp3_safe_archive_report=_bp3_report(dataset_rows),
            bp3_dataset_rows=dataset_rows,
            bp3_branch_evidence=_bp3_branch_evidence(),
            branch_results=branch_results,
            generation_status=_complete_status(branch_results),
        )
        merged = merge_public_sequence_context_branch_evidence_reports(
            bp3_safe_archive_report=_bp3_report(dataset_rows),
            bp3_dataset_rows=dataset_rows,
            bp3_branch_evidence=_bp3_branch_evidence(),
            shard_reports=[deepcopy(shard)],
        )
        self.assertTrue(merged["diagnostics_only"])

        shard.pop("diagnostics_only")

        with self.assertRaises(PublicSequenceContextBranchEvidenceError):
            merge_public_sequence_context_branch_evidence_reports(
                bp3_safe_archive_report=_bp3_report(dataset_rows),
                bp3_dataset_rows=dataset_rows,
                bp3_branch_evidence=_bp3_branch_evidence(),
                shard_reports=[shard],
            )


def _bp3_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_REPORT_SCHEMA_VERSION,
        "policy": "diagnostics_only_m3_carrion_broad_safe_archive_expansion_001",
        "classification": {
            "primary": "m3_safe_archive_expansion_support_ready_no_training_run",
            "labels": ["m3_safe_archive_expansion_support_ready_no_training_run"],
        },
        "source_integrity": {"passed": True, "failures": []},
        "dataset": {
            "dataset_digest": stable_payload_digest(rows),
            "safe_label_count": len(rows),
            "leakage_scan": {"passed": True, "failures": []},
        },
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "non_promoted": True,
        "contract": {
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
        },
    }


def _bp3_branch_evidence() -> dict[str, object]:
    branch_results: list[dict[str, object]] = []
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
        "policy": "diagnostics_only_m3_carrion_broad_safe_archive_expansion_001_branch_evidence_v1",
        "generation_status": {"state": "complete", "partial": False},
        "source_integrity": {"passed": True, "failures": []},
        "branch_results": branch_results,
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "non_promoted": True,
        "contract": {
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
        },
    }


def _bp3_dataset_row() -> dict[str, object]:
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {
                "observation_input": _observation_input(),
                "action_mask": _action_mask(),
            },
            "label": {
                "action": "stay",
                "label_policy": "unit_bp3_label",
            },
        },
        "metadata": {"row_index": 0, "seed": 5, "fixture": "broad"},
    }


def _branch_result() -> dict[str, object]:
    trainable_context = {
        "feature_policy": PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY,
        "features": {
            "observation_input": _observation_input(),
            "action_mask": _action_mask(),
            "prior_public_transition_summaries": [_prior_summary()],
        },
    }
    return {
        "branch_id": "m3-public-sequence-context-broad-seed-5-branch-0",
        "seed": 5,
        "fixture": "broad",
        "ticks": 120,
        "branch_tick": 3,
        "record_index": 10,
        "branch_index": 0,
        "agent_id": 4,
        "baseline_action": "stay",
        "v142_requested_action": "stay",
        "v142_resolved_action": "stay",
        "candidate_actions": ["stay"],
        "branch_state_digest": "b" * 64,
        "public_features": {
            "observation_input": _observation_input(),
            "action_mask": _action_mask(),
        },
        "public_sequence_context": {
            "feature_policy": PUBLIC_SEQUENCE_CONTEXT_FEATURE_POLICY,
            "trainable": trainable_context,
            "history_step_count": 1,
            "trainable_context_digest": stable_payload_digest(trainable_context),
        },
        "action_runs": [_action_run()],
    }


def _action_run() -> dict[str, object]:
    return {
        "forced_action": "stay",
        "forced_action_used": True,
        "forced_action_supported": True,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "deltas_vs_baseline": {
            "alive_agents": 0,
            "births": 0,
            "deaths": 0,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
        },
        "deltas_vs_v142_override": {
            "alive_agents": 0,
            "births": 0,
            "deaths": 0,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
        },
        "replay_verification": {
            "verified": True,
            "expected_digest": "c" * 64,
            "actual_digest": "c" * 64,
        },
    }


def _complete_status(branch_results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "state": "complete",
        "partial": False,
        "branch_point_count": len(branch_results),
        "branch_result_count": len(branch_results),
    }


def _observation_input() -> dict[str, object]:
    return {"values": [0.1, 0.2, 0.3]}


def _action_mask() -> dict[str, bool]:
    return {action: action == "stay" for action in ACTION_NAMES}


def _prior_summary() -> dict[str, object]:
    return {
        "requested_action": "drink",
        "resolved_action": "drink",
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": False,
        "x_delta": 0,
        "y_delta": 0,
        "energy_ratio_delta": -0.02,
        "hydration_ratio_delta": 0.1,
        "health_ratio_delta": 0.0,
        "resource_gain": 0.0,
        "drank": True,
        "ate": False,
        "animal_resource_gain": False,
        "died": False,
        "died_after_action": False,
    }


if __name__ == "__main__":
    unittest.main()
