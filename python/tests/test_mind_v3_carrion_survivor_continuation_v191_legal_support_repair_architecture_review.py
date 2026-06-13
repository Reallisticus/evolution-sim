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
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV191LegalSupportRepairTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v191-legal-support-repair-architecture-review"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v191_legal_support_repair_architecture_review"
            ),
        )

    def test_classifies_resolution_mask_flip_and_routes_to_contract_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "branch.jsonl.gz"
            _write_trajectory(trajectory, include_resolution_flip=True)
            v190_report = _v190_report(
                trajectory_path=trajectory,
                unsupported_resolved_action_count=1,
                target_seeds=(29,),
            )
            v190_digest = _attach_digest(v190_report)
            report = (
                v191.run_carrion_survivor_continuation_v191_legal_support_repair_architecture_review(
                    v190_report_override=v190_report,
                    output_path=root / "v191.json",
                    expected_v190_report_exact_digest=v190_digest,
                    target_seeds=(29,),
                    expected_branch_point_count=1,
                    expected_branch_run_count=1,
                    expected_unsupported_resolved_action_count=1,
                )
            )

        classification = report["unsupported_resolved_action_classification"]
        root_cause = classification["root_cause_assessment"]
        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(classification["observed_unsupported_resolved_action_count"], 1)
        self.assertTrue(classification["event_count_matches_v190_report"])
        self.assertTrue(
            classification[
                "all_invalid_resolution_records_flip_observation_true_to_resolution_false"
            ]
        )
        self.assertEqual(
            root_cause["primary"],
            "action_mask_timing_mismatch_same_tick_movement_occupancy_race",
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v191.ACTION_RESOLUTION_CONTRACT_REPAIR_ROUTE,
        )
        self.assertFalse(report["support_generation_ran"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_seed_29_zero_unsupported_routes_to_action_diversity_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "branch.jsonl.gz"
            _write_trajectory(trajectory, include_resolution_flip=False)
            v190_report = _v190_report(
                trajectory_path=trajectory,
                unsupported_resolved_action_count=0,
                target_seeds=(29,),
            )
            v190_digest = _attach_digest(v190_report)
            report = (
                v191.run_carrion_survivor_continuation_v191_legal_support_repair_architecture_review(
                    v190_report_override=v190_report,
                    output_path=root / "v191.json",
                    expected_v190_report_exact_digest=v190_digest,
                    target_seeds=(29,),
                    expected_branch_point_count=1,
                    expected_branch_run_count=1,
                    expected_unsupported_resolved_action_count=0,
                )
            )

        seed_29 = report["near_clean_seed_analysis"]["per_seed"]["29"]
        self.assertEqual(
            seed_29["repair_recommendation"]["recommendation"],
            "action_diversity_cap_repair_only",
        )
        self.assertTrue(seed_29["repair_recommendation"]["needs_action_share_cap_repair"])
        self.assertFalse(report["slice_3_training_consumed"])

    def test_v190_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "branch.jsonl.gz"
            _write_trajectory(trajectory, include_resolution_flip=True)
            v190_report = _v190_report(
                trajectory_path=trajectory,
                unsupported_resolved_action_count=1,
                target_seeds=(29,),
            )
            _attach_digest(v190_report)
            report = (
                v191.run_carrion_survivor_continuation_v191_legal_support_repair_architecture_review(
                    v190_report_override=v190_report,
                    output_path=root / "v191.json",
                    expected_v190_report_exact_digest="wrong",
                    target_seeds=(29,),
                    expected_branch_point_count=1,
                    expected_branch_run_count=1,
                    expected_unsupported_resolved_action_count=1,
                )
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v190_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["recommended_next_route"], v191.STOP_ROUTE)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["support_generation_ran"])


def _v190_report(
    *,
    trajectory_path: Path,
    unsupported_resolved_action_count: int,
    target_seeds: tuple[int, ...],
) -> dict[str, object]:
    run = {
        "seed": target_seeds[0],
        "branch_id": f"carrion-only-seed-{target_seeds[0]}-branch-0-tick-0-agent-9",
        "branch_tick": 0,
        "continuation_script": "conserve_after_carrion",
        "alive_agents": 3,
        "births": 14,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": unsupported_resolved_action_count,
        "heuristic_action_source_count": 0,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.5154,
        "trajectory_path": str(trajectory_path),
    }
    target_seed_strings = [str(seed) for seed in target_seeds]
    return {
        "schema_version": (
            v190.M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_SCHEMA_VERSION
        ),
        "policy": (
            v190.M3_CARRION_SURVIVOR_CONTINUATION_V190_TARGETED_LEGAL_TERMINAL_SURVIVAL_SUPPORT_EXPANSION_POLICY
        ),
        "source_validation": {"passed": True},
        "target_manifest": {
            "targeted_expansion_seeds": list(target_seeds),
            "target_seeds": list(target_seeds),
        },
        "expansion_search": {
            "ran": True,
            "branch_point_count": 1,
            "branch_run_count": 1,
            "replay_verified": True,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": unsupported_resolved_action_count,
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
            "target_seed_count": len(target_seeds),
            "per_seed": {
                seed: {
                    "best_attempt": {
                        "branch_id": run["branch_id"],
                        "continuation_script": run["continuation_script"],
                        "alive_agents": run["alive_agents"],
                        "births": run["births"],
                        "unsupported_requested_action_count": 0,
                        "unsupported_resolved_action_count": (
                            unsupported_resolved_action_count
                        ),
                        "dominant_requested_action": "stay",
                        "dominant_requested_action_share": 0.5154,
                    }
                }
                for seed in target_seed_strings
            },
        },
        "route_decision": {"recommended_next_route": v190.BLOCKED_ROUTE},
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_expansion_ran": True,
        "diagnostics_only": True,
    }


def _write_trajectory(path: Path, *, include_resolution_flip: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = [{"type": "header", "trajectory_contract": {}}]
    action_mask = {
        "stay": True,
        "eat": True,
        "drink": False,
        "move_north": True,
        "move_south": True,
        "move_east": True,
        "move_west": True,
    }
    resolution_mask = dict(action_mask)
    if include_resolution_flip:
        resolution_mask["move_north"] = False
    records.append(
        {
            "type": "record",
            "record": {
                "tick": 7,
                "agent_id": 12,
                "requested_action": "move_north",
                "resolved_action": "stay" if include_resolution_flip else "move_north",
                "action_valid": True,
                "resolution_action_valid": not include_resolution_flip,
                "action_mask": action_mask,
                "resolution_action_mask": resolution_mask,
                "outcome": {
                    "invalid_reason": (
                        "not_in_resolution_action_mask"
                        if include_resolution_flip
                        else None
                    )
                },
            },
        }
    )
    records.append(
        {
            "type": "footer",
            "summary": {"alive_agents": 3, "births": 14},
            "trajectory_summary": {
                "record_count": 1,
                "invalid_action_count": 0,
                "invalid_observation_action_count": 0,
                "invalid_resolution_action_count": 1 if include_resolution_flip else 0,
            },
        }
    )
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
