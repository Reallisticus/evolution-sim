from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair as v193,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit as v194,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV194DatasetAuditTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v194-repaired-contract-terminal-survival-support-dataset-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit"
            ),
        )

    def test_selected_support_dataset_passes_and_routes_to_slice_3_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "seed13.jsonl.gz"
            second = root / "seed19.jsonl.gz"
            _write_trajectory(
                first,
                records=[
                    _record("move_north", resolution_valid=False),
                    _record("eat"),
                    _record("stay"),
                ],
                alive_agents=2,
                births=4,
                deaths=1,
            )
            _write_trajectory(
                second,
                records=[_record("drink"), _record("eat"), _record("stay")],
                alive_agents=3,
                births=5,
                deaths=1,
            )
            runs = [
                _run_audit(seed=13, trajectory_path=first, alive=2, births=4, deaths=1),
                _run_audit(seed=19, trajectory_path=second, alive=3, births=5, deaths=1),
            ]
            report = _v193_report(runs)
            report_digest = _attach_digest(report)
            expected_by_seed = {
                "13": {
                    "alive_agents": 2,
                    "births": 4,
                    "dominant_requested_action": "stay",
                    "dominant_requested_action_share": 0.333333,
                    "expected_same_tick_occupancy_drift_count": 1,
                },
                "19": {
                    "alive_agents": 3,
                    "births": 5,
                    "dominant_requested_action": "stay",
                    "dominant_requested_action_share": 0.333333,
                    "expected_same_tick_occupancy_drift_count": 0,
                },
            }
            expected_aggregate = _expected_aggregate(runs)

            audit = (
                v194.run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit(
                    v193_report_override=report,
                    output_path=root / "v194.json",
                    compact_support_dataset_output_path=root / "dataset.jsonl",
                    expected_v193_report_exact_digest=report_digest,
                    expected_v192_report_exact_digest="v192",
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    expected_selected_support_by_seed=expected_by_seed,
                    expected_aggregate_support=expected_aggregate,
                    backup_metadata_override={"passed": True},
                )
            )

            dataset_rows = _read_jsonl(root / "dataset.jsonl")

        self.assertTrue(audit["source_validation"]["passed"])
        self.assertTrue(audit["selected_support_facts_audit"]["passed"])
        self.assertTrue(audit["dataset_audit"]["passed"])
        self.assertTrue(audit["compact_support_dataset_created"])
        self.assertEqual(audit["compact_support_dataset"]["row_count"], 6)
        self.assertEqual(
            audit["compact_support_dataset"]["dataset_digest"],
            stable_payload_digest(dataset_rows),
        )
        self.assertEqual(
            audit["route_decision"]["recommended_next_route"],
            v194.SUCCESS_ROUTE,
        )
        self.assertTrue(
            audit["route_decision"][
                "future_explicit_slice_3_training_route_authorized"
            ]
        )
        self.assertEqual(
            audit["dataset_audit"]["repaired_resolution_check"][
                "expected_same_tick_occupancy_drift_count"
            ],
            1,
        )
        self.assertEqual(
            audit["dataset_audit"]["repaired_resolution_check"][
                "expected_same_tick_occupancy_drift_counted_as_successful_move_count"
            ],
            0,
        )
        _assert_closed_hard_stops(self, audit)
        self.assertTrue(exact_digest_validation_report(audit)["passed"])

    def test_v193_digest_mismatch_fails_closed_without_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "seed13.jsonl.gz"
            _write_trajectory(
                trajectory,
                records=[_record("eat"), _record("stay")],
                alive_agents=2,
                births=4,
                deaths=1,
            )
            runs = [
                _run_audit(seed=13, trajectory_path=trajectory, alive=2, births=4, deaths=1)
            ]
            report = _v193_report(runs)
            _attach_digest(report)
            audit = (
                v194.run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit(
                    v193_report_override=report,
                    output_path=root / "v194.json",
                    compact_support_dataset_output_path=root / "dataset.jsonl",
                    expected_v193_report_exact_digest="wrong",
                    expected_v192_report_exact_digest="v192",
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    expected_selected_support_by_seed={
                        "13": {
                            "alive_agents": 2,
                            "births": 4,
                            "dominant_requested_action": "stay",
                            "dominant_requested_action_share": 0.5,
                            "expected_same_tick_occupancy_drift_count": 0,
                        }
                    },
                    expected_aggregate_support=_expected_aggregate(runs),
                    backup_metadata_override={"passed": True},
                )
            )

        self.assertFalse(audit["source_validation"]["passed"])
        self.assertIn(
            "v193_exact_digest_matches_expected",
            audit["source_validation"]["failures"],
        )
        self.assertFalse(audit["compact_support_dataset_created"])
        self.assertFalse(audit["dataset_audit"]["passed"])
        self.assertIn(
            "v193_source_validation_failed",
            audit["route_decision"]["recommended_next_route"],
        )
        _assert_closed_hard_stops(self, audit)

    def test_private_future_key_in_observation_blocks_training_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "leaky.jsonl.gz"
            leaky_record = _record("eat")
            leaky_record["observation_input"] = {
                **dict(leaky_record["observation_input"]),
                "future_outcome": "not allowed in trainable payload",
            }
            _write_trajectory(
                trajectory,
                records=[leaky_record, _record("stay")],
                alive_agents=2,
                births=4,
                deaths=1,
            )
            runs = [
                _run_audit(seed=13, trajectory_path=trajectory, alive=2, births=4, deaths=1)
            ]
            report = _v193_report(runs)
            report_digest = _attach_digest(report)
            audit = (
                v194.run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit(
                    v193_report_override=report,
                    output_path=root / "v194.json",
                    compact_support_dataset_output_path=root / "dataset.jsonl",
                    expected_v193_report_exact_digest=report_digest,
                    expected_v192_report_exact_digest="v192",
                    expected_v191_report_exact_digest="v191",
                    expected_v190_report_exact_digest="v190",
                    expected_selected_support_by_seed={
                        "13": {
                            "alive_agents": 2,
                            "births": 4,
                            "dominant_requested_action": "stay",
                            "dominant_requested_action_share": 0.5,
                            "expected_same_tick_occupancy_drift_count": 0,
                        }
                    },
                    expected_aggregate_support=_expected_aggregate(runs),
                    backup_metadata_override={"passed": True},
                )
            )

        self.assertTrue(audit["source_validation"]["passed"])
        self.assertFalse(audit["selected_support_facts_audit"]["passed"])
        self.assertFalse(audit["compact_support_dataset_created"])
        self.assertIn(
            "selected_support_facts_audit_failed",
            audit["route_decision"]["recommended_next_route"],
        )
        _assert_closed_hard_stops(self, audit)


def _run_audit(
    *,
    seed: int,
    trajectory_path: Path,
    alive: int,
    births: int,
    deaths: int,
) -> dict[str, object]:
    requested_counts = _requested_counts_from_trajectory(trajectory_path)
    dominant_action, dominant_count, dominant_share = _dominant(requested_counts)
    run = {
        "branch_id": f"carrion-only-seed-{seed}-branch-0-tick-0-agent-9",
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "base_script": "hydration_safe_carrion_cycle",
        "continuation_script": "balanced_legal_probe",
        "branch_state_digest": f"branch-{seed}",
        "alive_agents": alive,
        "births": births,
        "deaths": deaths,
        "trajectory_path": str(trajectory_path),
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": sum(
            1
            for record in _trajectory_records(trajectory_path)
            if record.get("resolution_action_valid") is False
        ),
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
            "replay_alive_agents": alive,
            "replay_births": births,
            "replay_deaths": deaths,
        },
    }
    return v193.repaired_contract_run_audit(
        run,
        max_dominant_requested_action_share=0.50,
        replay_verification_required=True,
    )


def _v193_report(runs: list[dict[str, object]]) -> dict[str, object]:
    diversity = v193.selected_support_set_action_diversity(runs)
    per_seed = {
        str(run["seed"]): {
            "seed": int(run["seed"]),
            "branch_run_count": 1,
            "terminal_survivor_attempt_count": 1,
            "within_action_share_cap_attempt_count": 1,
            "clean_support_run_count": 1,
            "expected_same_tick_occupancy_drift_count": int(
                run["expected_same_tick_occupancy_drift_count"]
            ),
            "unexpected_resolution_invalid_count": int(
                run["unexpected_resolution_invalid_count"]
            ),
            "blockers": [],
            "best_repaired_contract_attempt": dict(run),
        }
        for run in runs
    }
    return {
        "schema_version": (
            v193.M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_SCHEMA_VERSION
        ),
        "policy": (
            v193.M3_CARRION_SURVIVOR_CONTINUATION_V193_FRESH_TARGETED_LEGAL_SUPPORT_EXPANSION_AFTER_CONTRACT_REPAIR_POLICY
        ),
        "source_validation": {
            "passed": True,
            "observed_v192_report_exact_digest": "v192",
            "observed_v191_report_exact_digest": "v191",
            "observed_v190_report_exact_digest": "v190",
        },
        "repaired_contract_support_audit": {
            "passed": True,
            "blockers": [],
            "target_seed_count": len(runs),
            "target_seeds": [int(run["seed"]) for run in runs],
            "clean_support_seed_count": len(runs),
            "missing_clean_support_seeds": [],
            "run_audit_count": len(runs),
            "clean_run_count": len(runs),
            "terminal_survivor_attempt_count": len(runs),
            "aggregate_unsupported_requested_action_count": 0,
            "aggregate_expected_same_tick_occupancy_drift_count": sum(
                int(run["expected_same_tick_occupancy_drift_count"]) for run in runs
            ),
            "aggregate_unexpected_resolution_invalid_count": 0,
            "aggregate_expected_occupancy_drift_counted_as_successful_move": False,
            "selected_support_run_count": len(runs),
            "selected_support_runs": runs,
            "selected_expected_same_tick_occupancy_drift_count": sum(
                int(run["expected_same_tick_occupancy_drift_count"]) for run in runs
            ),
            "selected_support_set_action_diversity": diversity,
            "dominant_requested_action": diversity["dominant_requested_action"],
            "dominant_requested_action_count": diversity[
                "dominant_requested_action_count"
            ],
            "dominant_requested_action_share": diversity[
                "dominant_requested_action_share"
            ],
            "per_seed": per_seed,
            "run_audits": runs,
            "future_trainable_payload_policy": (
                "current_public_observation_and_current_public_action_mask_only"
            ),
            "trainable_dataset_created": False,
        },
        "route_decision": {"recommended_next_route": v193.SUCCESS_ROUTE},
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": True,
        "support_expansion_ran": True,
    }


def _expected_aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    diversity = v193.selected_support_set_action_diversity(runs)
    return {
        "clean_support_seed_count": len(runs),
        "target_seed_count": len(runs),
        "dominant_requested_action": diversity["dominant_requested_action"],
        "dominant_requested_action_share": diversity["dominant_requested_action_share"],
        "aggregate_unsupported_requested_action_count": 0,
        "aggregate_expected_same_tick_occupancy_drift_count": sum(
            int(run["expected_same_tick_occupancy_drift_count"]) for run in runs
        ),
        "aggregate_unexpected_resolution_invalid_count": 0,
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
    for record in _trajectory_records(path):
        action = str(record["requested_action"])
        counts[action] = counts.get(action, 0) + 1
    return counts


def _trajectory_records(path: Path) -> list[dict[str, object]]:
    return [
        dict(row["record"])
        for row in _read_jsonl(path)
        if row.get("type") == "record"
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        "support_generation_ran",
        "support_expansion_ran",
        "v180_rerun",
        "v186_rerun",
        "v190_rerun",
        "v191_rerun",
        "v192_rerun",
        "v193_rerun",
    ):
        test_case.assertFalse(report[key], key)


if __name__ == "__main__":
    unittest.main()
