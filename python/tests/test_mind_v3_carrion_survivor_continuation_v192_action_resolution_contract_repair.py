from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion as v190,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v191_legal_support_repair_architecture_review as v191,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v192_action_resolution_contract_repair as v192,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV192ActionResolutionContractRepairTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v192-action-resolution-contract-repair"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v192_action_resolution_contract_repair"
            ),
        )

    def test_support_contract_repair_serializes_target_and_routes_to_fresh_support(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "branch.jsonl.gz"
            _write_trajectory(trajectory, requested_action="move_north")
            v190_report = _v190_report(
                trajectory_path=trajectory,
                requested_action="move_north",
            )
            v190_digest = _attach_digest(v190_report)
            v191_report = _v191_report(
                v190_digest=v190_digest,
                counts_by_seed={"29": 1},
                counts_by_action={"move_north": 1},
            )
            v191_digest = _attach_digest(v191_report)

            report = (
                v192.run_carrion_survivor_continuation_v192_action_resolution_contract_repair(
                    v191_report_override=v191_report,
                    v190_report_override=v190_report,
                    output_path=root / "v192.json",
                    expected_v191_report_exact_digest=v191_digest,
                    expected_v190_report_exact_digest=v190_digest,
                    expected_unsupported_resolved_action_count=1,
                    expected_counts_by_seed={"29": 1},
                    expected_counts_by_action={"move_north": 1},
                    expected_counts_by_reason={"not_in_resolution_action_mask": 1},
                    target_seeds=(29,),
                )
            )

        self.assertTrue(report["source_validation"]["passed"])
        movement = report["movement_target_blocker_audit"]
        self.assertTrue(movement["contract_repair_sufficient"])
        self.assertEqual(
            movement["blocker_classification_counts"],
            {"resolution_invalid_same_tick_occupancy_race": 1},
        )
        event = movement["movement_target_blocker_events"][0]
        self.assertTrue(event["requested_action_valid_at_observation_time"])
        self.assertFalse(event["requested_action_valid_at_resolution_time"])
        self.assertEqual(event["from_x"], 5)
        self.assertEqual(event["from_y"], 5)
        self.assertEqual(event["target_x"], 5)
        self.assertEqual(event["target_y"], 4)
        self.assertTrue(event["target_in_bounds"])
        self.assertTrue(event["resolution_invalid_due_to_same_tick_occupancy_race"])
        self.assertEqual(
            report["repair_scope_decision"]["selected_repair_type"],
            "support_evidence_contract_repair",
        )
        self.assertFalse(
            report["repair_scope_decision"]["narrow_simulator_replay_contract_repair"][
                "selected"
            ]
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v192.NEXT_ROUTE_AFTER_REPAIR,
        )
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_non_movement_resolution_invalid_routes_to_architecture_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "branch.jsonl.gz"
            _write_trajectory(trajectory, requested_action="eat")
            v190_report = _v190_report(
                trajectory_path=trajectory,
                requested_action="eat",
            )
            v190_digest = _attach_digest(v190_report)
            v191_report = _v191_report(
                v190_digest=v190_digest,
                counts_by_seed={"29": 1},
                counts_by_action={"eat": 1},
            )
            v191_digest = _attach_digest(v191_report)

            report = (
                v192.run_carrion_survivor_continuation_v192_action_resolution_contract_repair(
                    v191_report_override=v191_report,
                    v190_report_override=v190_report,
                    output_path=root / "v192.json",
                    expected_v191_report_exact_digest=v191_digest,
                    expected_v190_report_exact_digest=v190_digest,
                    expected_unsupported_resolved_action_count=1,
                    expected_counts_by_seed={"29": 1},
                    expected_counts_by_action={"eat": 1},
                    expected_counts_by_reason={"not_in_resolution_action_mask": 1},
                    target_seeds=(29,),
                )
            )

        movement = report["movement_target_blocker_audit"]
        self.assertFalse(movement["contract_repair_sufficient"])
        self.assertEqual(
            movement["blocker_classification_counts"],
            {"depleted_resource_blocker": 1},
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v192.NEXT_ROUTE_ARCHITECTURE_REVIEW,
        )
        self.assertFalse(report["training_ran"])

    def test_v191_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "branch.jsonl.gz"
            _write_trajectory(trajectory, requested_action="move_north")
            v190_report = _v190_report(
                trajectory_path=trajectory,
                requested_action="move_north",
            )
            v190_digest = _attach_digest(v190_report)
            v191_report = _v191_report(
                v190_digest=v190_digest,
                counts_by_seed={"29": 1},
                counts_by_action={"move_north": 1},
            )
            _attach_digest(v191_report)

            report = (
                v192.run_carrion_survivor_continuation_v192_action_resolution_contract_repair(
                    v191_report_override=v191_report,
                    v190_report_override=v190_report,
                    output_path=root / "v192.json",
                    expected_v191_report_exact_digest="wrong",
                    expected_v190_report_exact_digest=v190_digest,
                    expected_unsupported_resolved_action_count=1,
                    expected_counts_by_seed={"29": 1},
                    expected_counts_by_action={"move_north": 1},
                    expected_counts_by_reason={"not_in_resolution_action_mask": 1},
                    target_seeds=(29,),
                )
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v191_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["recommended_next_route"], v192.STOP_ROUTE)
        self.assertFalse(report["support_generation_ran"])
        self.assertFalse(report["promotion_authorized"])


def _v190_report(
    *,
    trajectory_path: Path,
    requested_action: str,
) -> dict[str, object]:
    run = {
        "seed": 29,
        "branch_id": "carrion-only-seed-29-branch-0-tick-0-agent-9",
        "branch_tick": 0,
        "continuation_script": "conserve_after_carrion",
        "alive_agents": 3,
        "births": 14,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 1,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.42,
        "trajectory_path": str(trajectory_path),
    }
    return {
        "schema_version": (
            v190.M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_SCHEMA_VERSION
        ),
        "policy": (
            v190.M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_POLICY
        ),
        "source_validation": {"passed": True},
        "target_manifest": {
            "targeted_expansion_seeds": [29],
            "target_seeds": [29],
        },
        "expansion_search": {
            "ran": True,
            "branch_point_count": 1,
            "branch_run_count": 1,
            "replay_verified": True,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 1,
            "branch_report": {
                "aggregate": {
                    "branch_point_count": 1,
                    "branch_run_count": 1,
                    "replay_verified": True,
                },
                "branch_runs": [run],
            },
        },
        "legal_support_audit": {
            "clean_legal_support_seed_count": 0,
            "target_seed_count": 1,
        },
        "route_decision": {"recommended_next_route": v190.BLOCKED_ROUTE},
        "diagnostics_only": True,
    }


def _v191_report(
    *,
    v190_digest: str,
    counts_by_seed: dict[str, int],
    counts_by_action: dict[str, int],
) -> dict[str, object]:
    return {
        "schema_version": (
            v191.M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_SCHEMA_VERSION
        ),
        "policy": (
            v191.M3_CARRION_SURVIVOR_CONTINUATION_V191_LEGAL_SUPPORT_REPAIR_ARCHITECTURE_REVIEW_POLICY
        ),
        "source_validation": {
            "passed": True,
            "observed_v190_report_exact_digest": v190_digest,
        },
        "unsupported_resolved_action_classification": {
            "v190_reported_unsupported_requested_action_count": 0,
            "v190_reported_unsupported_resolved_action_count": 1,
            "observed_unsupported_resolved_action_count": 1,
            "counts_by_seed": counts_by_seed,
            "counts_by_requested_action": counts_by_action,
            "counts_by_legality_reason": {"not_in_resolution_action_mask": 1},
            "counts_by_resolved_action": {"stay": 1},
            "all_invalid_resolution_records_resolve_to_stay": True,
            "all_invalid_resolution_records_are_observation_valid": True,
            "all_invalid_resolution_records_flip_observation_true_to_resolution_false": True,
            "all_invalid_resolution_requested_actions_are_movement": True,
            "root_cause_assessment": {
                "primary": v192.EXPECTED_ROOT_CAUSE,
            },
        },
        "route_decision": {
            "recommended_next_route": v191.ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE,
            "exactly_one_next_route_recommended": True,
        },
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
    }


def _write_trajectory(path: Path, *, requested_action: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    action_mask = {
        "stay": True,
        "eat": requested_action == "eat",
        "drink": False,
        "move_north": requested_action == "move_north",
        "move_south": True,
        "move_east": True,
        "move_west": True,
    }
    resolution_mask = dict(action_mask)
    resolution_mask[requested_action] = False
    records = [
        {
            "type": "header",
            "config": {"width": 10, "height": 10, "seed": 29, "max_ticks": 120},
        },
        {
            "type": "record",
            "record": {
                "tick": 3,
                "agent_id": 12,
                "requested_action": requested_action,
                "resolved_action": "stay",
                "action_valid": True,
                "resolution_action_valid": False,
                "action_mask": action_mask,
                "resolution_action_mask": resolution_mask,
                "before": {
                    "x": 5,
                    "y": 5,
                    "alive": True,
                    "energy": 1.0,
                    "hydration": 1.0,
                    "health": 1.0,
                },
                "after": {
                    "x": 5,
                    "y": 5,
                    "alive": True,
                    "energy": 1.0,
                    "hydration": 1.0,
                    "health": 1.0,
                },
                "moved": False,
                "outcome": {
                    "invalid_reason": "not_in_resolution_action_mask",
                },
            },
        },
        {
            "type": "footer",
            "summary": {"alive_agents": 3, "births": 14},
            "trajectory_summary": {
                "record_count": 1,
                "invalid_action_count": 0,
                "invalid_observation_action_count": 0,
                "invalid_resolution_action_count": 1,
            },
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _attach_digest(report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(payload)
    return str(report["exact_digest"])


if __name__ == "__main__":
    unittest.main()
