from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v192_action_resolution_contract_repair as v192,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair as v193,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV193FreshTargetedSupportTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v193-fresh-targeted-legal-support-expansion-after-contract-repair"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair"
            ),
        )

    def test_expected_occupancy_drift_counts_as_repaired_contract_support(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "support.jsonl.gz"
            _write_trajectory(
                trajectory,
                records=[
                    _record("move_north", resolution_valid=False),
                    _record("eat"),
                ],
                alive_agents=2,
                births=5,
                deaths=1,
            )
            v192_report = _v192_report(v191_digest="v191", v190_digest="v190")
            v192_digest = _attach_digest(v192_report)
            report = (
                v193.run_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair(
                    v192_report_override=v192_report,
                    branch_report_override=_branch_report(
                        [
                            _support_run(
                                seed=29,
                                trajectory_path=trajectory,
                                unsupported_resolved=1,
                            )
                        ]
                    ),
                    output_path=root / "v193.json",
                    expected_v192_report_exact_digest=v192_digest,
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    seeds=(29,),
                )
            )

        self.assertTrue(report["source_validation"]["passed"])
        support = report["repaired_contract_support_audit"]
        self.assertTrue(support["passed"])
        self.assertEqual(support["clean_support_seed_count"], 1)
        self.assertEqual(support["aggregate_expected_same_tick_occupancy_drift_count"], 1)
        self.assertEqual(support["aggregate_unexpected_resolution_invalid_count"], 0)
        best = support["per_seed"]["29"]["best_repaired_contract_attempt"]
        self.assertTrue(best["counts_as_repaired_contract_support"])
        self.assertEqual(best["unsupported_requested_action_count"], 0)
        self.assertEqual(best["expected_same_tick_occupancy_drift_count"], 1)
        self.assertEqual(best["unexpected_resolution_invalid_count"], 0)
        self.assertTrue(best["replay_verified"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v193.SUCCESS_ROUTE,
        )
        self.assertTrue(
            report["route_decision"][
                "fresh_dataset_audit_required_before_slice_3_training"
            ]
        )
        self.assertFalse(report["route_decision"]["slice_3_training_authorized"])
        self.assertTrue(report["support_expansion_ran"])
        _assert_closed_hard_stops(self, report)
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_unexpected_resolution_invalid_blocks_support_repair_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "depleted.jsonl.gz"
            _write_trajectory(
                trajectory,
                records=[
                    _record("eat", resolution_valid=False),
                    _record("drink"),
                ],
                alive_agents=2,
                births=5,
                deaths=1,
            )
            v192_report = _v192_report(v191_digest="v191", v190_digest="v190")
            v192_digest = _attach_digest(v192_report)
            report = (
                v193.run_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair(
                    v192_report_override=v192_report,
                    branch_report_override=_branch_report(
                        [
                            _support_run(
                                seed=29,
                                trajectory_path=trajectory,
                                unsupported_resolved=1,
                            )
                        ]
                    ),
                    output_path=root / "v193.json",
                    expected_v192_report_exact_digest=v192_digest,
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    seeds=(29,),
                )
            )

        support = report["repaired_contract_support_audit"]
        self.assertFalse(support["passed"])
        self.assertEqual(support["aggregate_expected_same_tick_occupancy_drift_count"], 0)
        self.assertEqual(support["aggregate_unexpected_resolution_invalid_count"], 1)
        best = support["per_seed"]["29"]["best_repaired_contract_attempt"]
        self.assertFalse(best["counts_as_repaired_contract_support"])
        self.assertIn("unexpected_resolution_invalid_count_zero", best["failures"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v193.BLOCKED_ROUTE,
        )
        _assert_closed_hard_stops(self, report)

    def test_action_share_cap_blocks_even_when_actions_are_legal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "collapsed.jsonl.gz"
            _write_trajectory(
                trajectory,
                records=[_record("stay"), _record("stay"), _record("eat")],
                alive_agents=2,
                births=5,
                deaths=1,
            )
            v192_report = _v192_report(v191_digest="v191", v190_digest="v190")
            v192_digest = _attach_digest(v192_report)
            report = (
                v193.run_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair(
                    v192_report_override=v192_report,
                    branch_report_override=_branch_report(
                        [_support_run(seed=29, trajectory_path=trajectory)]
                    ),
                    output_path=root / "v193.json",
                    expected_v192_report_exact_digest=v192_digest,
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    seeds=(29,),
                )
            )

        best = report["repaired_contract_support_audit"]["per_seed"]["29"][
            "best_repaired_contract_attempt"
        ]
        self.assertFalse(best["counts_as_repaired_contract_support"])
        self.assertIn("dominant_requested_action_share_within_cap", best["failures"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v193.BLOCKED_ROUTE,
        )

    def test_v192_digest_mismatch_fails_closed_without_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v192_report = _v192_report(v191_digest="v191", v190_digest="v190")
            _attach_digest(v192_report)
            report = (
                v193.run_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair(
                    v192_report_override=v192_report,
                    output_path=root / "v193.json",
                    expected_v192_report_exact_digest="wrong",
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    seeds=(29,),
                )
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v192_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["fresh_expansion_search"]["ran"])
        self.assertEqual(report["route_decision"]["recommended_next_route"], v193.STOP_ROUTE)
        self.assertFalse(report["support_expansion_ran"])
        _assert_closed_hard_stops(self, report)


def _v192_report(*, v191_digest: str, v190_digest: str) -> dict[str, object]:
    return {
        "schema_version": (
            v192.M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_SCHEMA_VERSION
        ),
        "policy": (
            v192.M3_CARRION_SURVIVOR_CONTINUATION_V192_ACTION_RESOLUTION_CONTRACT_REPAIR_POLICY
        ),
        "source_validation": {
            "passed": True,
            "observed_v191_report_exact_digest": v191_digest,
            "observed_v190_report_exact_digest": v190_digest,
        },
        "repair_scope_decision": {
            "selected_repair_type": "support_evidence_contract_repair",
            "contract_repair_sufficient": True,
        },
        "movement_target_blocker_audit": {
            "blocker_classification_counts": {
                "resolution_invalid_same_tick_occupancy_race": (
                    v192.EXPECTED_UNSUPPORTED_RESOLVED_ACTION_COUNT
                )
            },
            "bounds_blocker_count": 0,
            "water_blocker_count": 0,
            "hazard_blocker_count": 0,
            "depleted_resource_blocker_count": 0,
            "stale_or_illegal_script_mask_count": 0,
            "all_resolution_invalid_events_are_same_tick_occupancy_races": True,
        },
        "route_decision": {
            "recommended_next_route": v193.REQUIRED_V192_ROUTE,
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


def _branch_report(runs: list[dict[str, object]]) -> dict[str, object]:
    terminal_by_seed = {
        str(run["seed"]): sum(
            1
            for item in runs
            if int(item["seed"]) == int(run["seed"])
            and int(item.get("alive_agents", 0)) > 0
        )
        for run in runs
    }
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "aggregate": {
            "branch_point_count": len(runs),
            "branch_run_count": len(runs),
            "successful_branch_run_count": len(runs),
            "positive_seed_count": len(terminal_by_seed),
            "replay_verified": True,
            "unsupported_requested_action_count": sum(
                int(run.get("unsupported_requested_action_count", 0)) for run in runs
            ),
            "unsupported_resolved_action_count": sum(
                int(run.get("unsupported_resolved_action_count", 0)) for run in runs
            ),
            "terminal_survivor_count_by_seed": terminal_by_seed,
            "trajectory_paths": [run["trajectory_path"] for run in runs],
        },
        "branch_runs": runs,
    }


def _support_run(
    *,
    seed: int,
    trajectory_path: Path,
    unsupported_resolved: int = 0,
) -> dict[str, object]:
    requested_counts = _requested_counts_from_trajectory(trajectory_path)
    dominant_action, dominant_count, dominant_share = _dominant(requested_counts)
    return {
        "branch_id": f"carrion-only-seed-{seed}-branch-0-tick-0-agent-9",
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "base_script": "hydration_safe_carrion_cycle",
        "continuation_script": "balanced_legal_probe",
        "branch_state_digest": f"branch-{seed}",
        "alive_agents": 2,
        "births": 5,
        "deaths": 1,
        "trajectory_path": str(trajectory_path),
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": unsupported_resolved,
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_count": dominant_count,
        "dominant_requested_action_share": dominant_share,
        "requested_action_counts": requested_counts,
        "replay_digest": f"replay-{seed}",
        "replay_verified": True,
        "replay_verification": {
            "verified": True,
            "expected_digest": f"replay-{seed}",
            "actual_digest": f"replay-{seed}",
            "replay_alive_agents": 2,
            "replay_births": 5,
            "replay_deaths": 1,
        },
    }


def _record(action: str, *, resolution_valid: bool = True) -> dict[str, object]:
    mask = {
        "stay": True,
        "eat": True,
        "drink": True,
        "move_north": True,
        "move_south": True,
        "move_east": True,
        "move_west": True,
    }
    resolution_mask = dict(mask)
    if not resolution_valid:
        resolution_mask[action] = False
    return {
        "tick": 0,
        "agent_id": 9,
        "requested_action": action,
        "resolved_action": action if resolution_valid else "stay",
        "action_valid": True,
        "resolution_action_valid": resolution_valid,
        "action_mask": mask,
        "resolution_action_mask": resolution_mask,
        "action_source": "counterfactual_script:balanced_legal_probe",
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
            "y": 5 if not resolution_valid else 4 if action == "move_north" else 5,
            "alive": True,
            "energy": 1.0,
            "hydration": 1.0,
            "health": 1.0,
        },
        "moved": resolution_valid and action == "move_north",
        "outcome": {
            "invalid_reason": (
                "not_in_resolution_action_mask" if not resolution_valid else None
            )
        },
        "observation_input": {
            "schema_version": "mind_observation_v3",
            "encoder_version": "mind_observation_encoder_v2",
            "shape": [1],
            "data": "AA==",
        },
    }


def _write_trajectory(
    path: Path,
    *,
    records: list[dict[str, object]],
    alive_agents: int,
    births: int,
    deaths: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [
        {
            "type": "header",
            "trajectory_contract": {},
            "config": {"width": 10, "height": 10, "seed": 29, "max_ticks": 120},
        }
    ]
    for index, record in enumerate(records):
        payload = dict(record)
        payload["tick"] = index
        rows.append({"type": "record", "record": payload})
    rows.append(
        {
            "type": "footer",
            "summary": {
                "alive_agents": alive_agents,
                "births": births,
                "deaths": deaths,
            },
            "trajectory_summary": {
                "record_count": len(records),
                "invalid_action_count": 0,
                "invalid_observation_action_count": 0,
                "invalid_resolution_action_count": sum(
                    1
                    for record in records
                    if record.get("resolution_action_valid") is False
                ),
            },
        }
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _requested_counts_from_trajectory(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") != "record":
                continue
            action = str(row["record"]["requested_action"])
            counts[action] = counts.get(action, 0) + 1
    return counts


def _dominant(counts: dict[str, int]) -> tuple[str | None, int, float]:
    if not counts:
        return None, 0, 0.0
    action, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    total = sum(counts.values())
    return action, count, round(count / total, 6)


def _attach_digest(report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(payload)
    return str(report["exact_digest"])


def _assert_closed_hard_stops(
    test_case: unittest.TestCase,
    report: dict[str, object],
) -> None:
    for key in (
        "training_ran",
        "training_artifact_created",
        "slice_3_training_consumed",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "promotion_authorized",
        "gate_relaxation_allowed",
        "gate_relaxation_ran",
        "v180_rerun",
        "v186_rerun",
        "v190_rerun",
        "v191_rerun",
        "v192_rerun",
    ):
        test_case.assertFalse(report[key], key)


if __name__ == "__main__":
    unittest.main()
