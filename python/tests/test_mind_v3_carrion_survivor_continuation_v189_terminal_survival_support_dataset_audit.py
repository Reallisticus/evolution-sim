from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v188_terminal_carrion_survival_support as v188,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit as v189,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV189TerminalSurvivalSupportDatasetAuditTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v189-terminal-survival-support-dataset-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit"
            ),
        )

    def test_one_of_six_legal_support_and_dominant_share_block_slice_3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "selected.jsonl.gz"
            _write_trajectory(
                trajectory,
                requested_actions=("stay", "stay", "stay", "eat", "drink"),
                alive_agents=1,
                births=5,
                deaths=16,
            )
            v188_report = _v188_report(
                [13, 19, 29, 37, 41, 43],
                support_runs=[
                    _support_run(
                        seed=13,
                        trajectory_path=trajectory,
                        dominant_requested_action="stay",
                        dominant_requested_action_share=0.6,
                    )
                ],
                terminal_by_seed={
                    "13": 1,
                    "19": 0,
                    "29": 0,
                    "37": 0,
                    "41": 0,
                    "43": 0,
                },
                attempted_terminal_by_seed={
                    "13": 3,
                    "19": 3,
                    "29": 4,
                    "37": 3,
                    "41": 5,
                    "43": 4,
                },
                aggregate_unsupported_resolved=285,
            )
            v188_digest = _attach_digest(v188_report)
            report = v189.run_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit(
                v188_report_override=v188_report,
                output_path=root / "v189.json",
                expected_v188_report_exact_digest=v188_digest,
                expected_selected_support=None,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["selected_support_trajectory_audit"]["passed"])
        self.assertTrue(report["trainable_leakage_audit"]["passed"])
        self.assertFalse(report["support_coverage_audit"]["passed"])
        self.assertEqual(
            report["support_coverage_audit"]["legal_positive_target_seed_count"],
            1,
        )
        self.assertEqual(report["support_coverage_audit"]["target_seed_count"], 6)
        self.assertTrue(
            report["aggregate_attempted_continuation_audit"][
                "aggregate_attempted_continuations_rejected_as_support"
            ]
        )
        self.assertIn(
            "legal_terminal_support_target_seed_coverage_insufficient",
            report["route_decision"]["blockers"],
        )
        self.assertIn(
            "selected_support_dominant_requested_action_share_above_cap",
            report["route_decision"]["blockers"],
        )
        self.assertIn(
            "aggregate_attempted_continuations_rejected_as_support",
            report["route_decision"]["blockers"],
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v189.TARGETED_SUPPORT_EXPANSION_ROUTE,
        )
        self.assertFalse(report["route_decision"]["slice_3_training_authorized"])
        _assert_closed_lifecycle(self, report)
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v188_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "selected.jsonl.gz"
            _write_trajectory(trajectory, requested_actions=("stay", "eat"))
            v188_report = _v188_report(
                [13],
                support_runs=[_support_run(seed=13, trajectory_path=trajectory)],
                terminal_by_seed={"13": 1},
                attempted_terminal_by_seed={"13": 1},
                aggregate_unsupported_resolved=0,
            )
            _attach_digest(v188_report)
            report = v189.run_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit(
                v188_report_override=v188_report,
                output_path=root / "v189.json",
                expected_v188_report_exact_digest="wrong",
                expected_selected_support=None,
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v188_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertEqual(report["route_decision"]["recommended_next_route"], "stop")
        self.assertFalse(report["route_decision"]["slice_3_training_authorized"])
        _assert_closed_lifecycle(self, report)

    def test_all_seed_legal_support_with_diverse_actions_authorizes_only_future_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seeds = [13, 19, 29, 37, 41, 43]
            support_runs = []
            for seed in seeds:
                trajectory = root / f"selected-{seed}.jsonl.gz"
                _write_trajectory(
                    trajectory,
                    requested_actions=("stay", "eat"),
                    alive_agents=1,
                    births=5,
                    deaths=16,
                )
                support_runs.append(_support_run(seed=seed, trajectory_path=trajectory))
            terminal_by_seed = {str(seed): 1 for seed in seeds}
            v188_report = _v188_report(
                seeds,
                support_runs=support_runs,
                terminal_by_seed=terminal_by_seed,
                attempted_terminal_by_seed=terminal_by_seed,
                aggregate_unsupported_resolved=0,
            )
            v188_digest = _attach_digest(v188_report)
            report = v189.run_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit(
                v188_report_override=v188_report,
                output_path=root / "v189.json",
                expected_v188_report_exact_digest=v188_digest,
                expected_selected_support=None,
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["support_coverage_audit"]["passed"])
        self.assertTrue(report["selected_support_trajectory_audit"]["passed"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v189.SLICE_3_TRAINING_ROUTE,
        )
        self.assertTrue(report["route_decision"]["slice_3_training_authorized"])
        self.assertFalse(report["route_decision"]["slice_3_training_allowed_for_this_command"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertFalse(report["training_ran"])

    def test_cli_writes_report_and_prints_blocked_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "selected.jsonl.gz"
            _write_trajectory(
                trajectory,
                requested_actions=("stay", "stay", "eat"),
                alive_agents=1,
                births=5,
                deaths=16,
            )
            v188_report = _v188_report(
                [13, 19],
                support_runs=[_support_run(seed=13, trajectory_path=trajectory)],
                terminal_by_seed={"13": 1, "19": 0},
                attempted_terminal_by_seed={"13": 1, "19": 1},
                aggregate_unsupported_resolved=7,
            )
            v188_digest = _attach_digest(v188_report)
            v188_path = root / "v188.json"
            v189_path = root / "v189.json"
            _write_json(v188_path, v188_report)

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit",
                    "--v188-report",
                    str(v188_path),
                    "--output",
                    str(v189_path),
                    "--expected-v188-report-exact-digest",
                    v188_digest,
                    "--skip-canonical-selected-support-check",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(v189_path.read_text(encoding="utf-8"))

        self.assertIn("source_validation_passed=True", result.stdout)
        self.assertIn("slice_3_training_authorized=False", result.stdout)
        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_3_training_consumed=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v189.TARGETED_SUPPORT_EXPANSION_ROUTE,
        )
        _assert_closed_lifecycle(self, report)


def _v188_report(
    seeds: list[int],
    *,
    support_runs: list[dict[str, object]],
    terminal_by_seed: dict[str, int],
    attempted_terminal_by_seed: dict[str, int],
    aggregate_unsupported_resolved: int,
) -> dict[str, object]:
    return {
        "schema_version": (
            v188.M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_SCHEMA_VERSION
        ),
        "policy": (
            v188.M3_CARRION_SURVIVOR_CONTINUATION_V188_TERMINAL_CARRION_SURVIVAL_SUPPORT_POLICY
        ),
        "source_validation": {
            "passed": True,
            "observed_v187_report_exact_digest": (
                v189.EXPECTED_V187_REPORT_EXACT_DIGEST
            ),
        },
        "support_search": {
            "ran": True,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": aggregate_unsupported_resolved,
        },
        "support_result": {
            "positive_support_found": bool(support_runs),
            "target_seed_count": len(seeds),
            "positive_seed_count": sum(1 for value in terminal_by_seed.values() if value),
            "terminal_survivors_by_seed": terminal_by_seed,
            "attempted_terminal_survivors_by_seed": attempted_terminal_by_seed,
            "all_target_seeds_have_positive_support": all(
                terminal_by_seed.get(str(seed), 0) > 0 for seed in seeds
            ),
            "attempted_successful_branch_run_count": sum(
                1 for value in attempted_terminal_by_seed.values() if value
            ),
            "attempted_positive_seed_count": sum(
                1 for value in attempted_terminal_by_seed.values() if value
            ),
            "support_runs": support_runs,
            "support_trajectory_manifest": [
                {
                    "path": run["trajectory_path"],
                    "seed": run["seed"],
                    "branch_id": run["branch_id"],
                    "continuation_script": run["continuation_script"],
                    "logical_replay_digest": run["replay_digest"],
                    "terminal_alive_agents": run["alive_agents"],
                    "births": run["births"],
                }
                for run in support_runs
            ],
        },
        "route_decision": {
            "recommended_next_route": v189.EXPECTED_V188_REQUIRED_ROUTE,
        },
        "classification": {
            "primary": (
                "m3_carrion_survivor_continuation_v188_terminal_carrion_"
                "survival_support_positive_legal_branch_support_found_no_training"
            )
        },
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
    }


def _support_run(
    *,
    seed: int,
    trajectory_path: Path,
    dominant_requested_action: str = "stay",
    dominant_requested_action_share: float = 0.5,
) -> dict[str, object]:
    return {
        "branch_id": f"carrion-only-seed-{seed}-branch-0-tick-0-agent-9",
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "base_script": "hydration_safe_carrion_cycle",
        "continuation_script": "conserve_after_carrion",
        "branch_state_digest": f"branch-{seed}",
        "alive_agents": 1,
        "births": 5,
        "deaths": 16,
        "trajectory_path": str(trajectory_path),
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "dominant_requested_action": dominant_requested_action,
        "dominant_requested_action_share": dominant_requested_action_share,
        "replay_verified": True,
        "replay_digest": f"replay-{seed}",
        "replay_verification": {
            "verified": True,
            "expected_digest": f"replay-{seed}",
            "actual_digest": f"replay-{seed}",
            "replay_alive_agents": 1,
            "replay_births": 5,
            "replay_deaths": 16,
        },
    }


def _write_trajectory(
    path: Path,
    *,
    requested_actions: tuple[str, ...],
    alive_agents: int = 1,
    births: int = 5,
    deaths: int = 16,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [{"type": "header", "trajectory_contract": {}}]
    for tick, action in enumerate(requested_actions):
        mask = {
            "stay": True,
            "eat": True,
            "drink": True,
            "move_north": True,
            "move_south": True,
            "move_east": True,
            "move_west": True,
        }
        rows.append(
            {
                "type": "record",
                "record": {
                    "tick": tick,
                    "requested_action": action,
                    "resolved_action": action,
                    "action_valid": True,
                    "resolution_action_valid": True,
                    "action_mask": mask,
                    "resolution_action_mask": mask,
                    "action_source": "counterfactual_script:conserve_after_carrion",
                    "observation_input": {
                        "schema_version": "mind_observation_v3",
                        "encoder_version": "mind_observation_encoder_v2",
                        "shape": [1],
                        "data": "AA==",
                    },
                },
            }
        )
    rows.append(
        {
            "type": "footer",
            "summary": {
                "alive_agents": alive_agents,
                "births": births,
                "deaths": deaths,
            },
            "trajectory_summary": {
                "record_count": len(requested_actions),
                "invalid_action_count": 0,
                "invalid_observation_action_count": 0,
                "invalid_resolution_action_count": 0,
            },
        }
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _attach_digest(report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(payload)
    return str(report["exact_digest"])


def _write_json(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_closed_lifecycle(
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
        "generic_carrion_autopsy_rerun",
        "v180_rerun",
        "v186_rerun",
        "v188_rerun",
    ):
        test_case.assertFalse(report[key], key)


if __name__ == "__main__":
    unittest.main()
