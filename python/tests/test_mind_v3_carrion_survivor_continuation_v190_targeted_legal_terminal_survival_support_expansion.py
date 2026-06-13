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
    carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit as v189,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion as v190,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV190TargetedLegalSupportExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v190-targeted-legal-terminal-survival-support-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion"
            ),
        )

    def test_v189_blockers_mine_target_manifest_and_route_to_repair_when_empty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v189_report = _v189_report()
            v189_digest = _attach_digest(v189_report)
            report = (
                v190.run_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion(
                    v189_report_override=v189_report,
                    branch_report_override=_branch_report([]),
                    output_path=root / "v190.json",
                    expected_v189_report_exact_digest=v189_digest,
                )
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertEqual(
            report["target_manifest"]["missing_legal_support_seeds"],
            [19, 29, 37, 41, 43],
        )
        self.assertEqual(report["target_manifest"]["action_diversity_repair_seeds"], [13])
        self.assertEqual(
            report["target_manifest"]["targeted_expansion_seeds"],
            [13, 19, 29, 37, 41, 43],
        )
        self.assertFalse(report["legal_support_audit"]["passed"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v190.BLOCKED_ROUTE,
        )
        self.assertFalse(report["route_decision"]["slice_3_training_authorized"])
        self.assertTrue(report["support_expansion_ran"])
        _assert_closed_lifecycle(self, report)
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_all_seed_clean_legal_support_routes_only_to_v191_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seeds = [13, 19, 29, 37, 41, 43]
            runs = []
            for seed in seeds:
                trajectory = root / f"support-{seed}.jsonl.gz"
                _write_trajectory(
                    trajectory,
                    requested_actions=("eat", "stay"),
                    alive_agents=1,
                    births=5,
                    deaths=16,
                )
                runs.append(_support_run(seed=seed, trajectory_path=trajectory))
            v189_report = _v189_report()
            v189_digest = _attach_digest(v189_report)
            report = (
                v190.run_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion(
                    v189_report_override=v189_report,
                    branch_report_override=_branch_report(runs),
                    output_path=root / "v190.json",
                    expected_v189_report_exact_digest=v189_digest,
                )
            )

        self.assertTrue(report["legal_support_audit"]["passed"])
        self.assertEqual(report["legal_support_audit"]["clean_legal_support_seed_count"], 6)
        self.assertEqual(report["legal_support_audit"]["support_run_count"], 6)
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v190.SUCCESS_ROUTE,
        )
        self.assertTrue(
            report["route_decision"][
                "fresh_v191_dataset_audit_required_before_slice_3_training"
            ]
        )
        self.assertFalse(report["route_decision"]["slice_3_training_authorized"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v189_digest_mismatch_fails_closed_without_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v189_report = _v189_report()
            _attach_digest(v189_report)
            report = (
                v190.run_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion(
                    v189_report_override=v189_report,
                    output_path=root / "v190.json",
                    expected_v189_report_exact_digest="wrong",
                )
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v189_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["expansion_search"]["ran"])
        self.assertEqual(report["route_decision"]["recommended_next_route"], v190.STOP_ROUTE)
        self.assertFalse(report["support_expansion_ran"])
        _assert_closed_lifecycle(self, report)


def _v189_report() -> dict[str, object]:
    return {
        "schema_version": (
            v189.M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy": (
            v189.M3_CARRION_SURVIVOR_CONTINUATION_V189_TERMINAL_SURVIVAL_SUPPORT_DATASET_AUDIT_POLICY
        ),
        "route_decision": {
            "recommended_next_route": v190.EXPECTED_V189_REQUIRED_ROUTE,
            "blockers": list(v190.EXPECTED_V189_BLOCKERS),
        },
        "support_coverage_audit": {
            "legal_positive_target_seed_count": 1,
            "target_seed_count": 6,
            "terminal_survivors_by_seed": {
                "13": 1,
                "19": 0,
                "29": 0,
                "37": 0,
                "41": 0,
                "43": 0,
            },
            "attempted_terminal_survivors_by_seed": {
                "13": 3,
                "19": 3,
                "29": 4,
                "37": 3,
                "41": 5,
                "43": 4,
            },
        },
        "aggregate_attempted_continuation_audit": {
            "aggregate_attempted_continuations_rejected_as_support": True,
            "attempted_positive_seed_count": 6,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 285,
            "attempted_terminal_survivors_by_seed": {
                "13": 3,
                "19": 3,
                "29": 4,
                "37": 3,
                "41": 5,
                "43": 4,
            },
        },
        "selected_support_trajectory_audit": {
            "dominant_requested_action": "stay",
            "dominant_requested_action_share": 0.539244,
            "run_audits": [
                {
                    "seed": 13,
                    "branch_id": "carrion-only-seed-13-branch-0-tick-0-agent-9",
                    "continuation_script": "conserve_after_carrion",
                    "alive_agents": 1,
                    "births": 5,
                }
            ],
        },
        "historical_dedupe": {
            "passed": True,
            "rerun_old_feasibility_counterfactual_iql_or_scorer_loops": False,
        },
        "training_ran": False,
        "training_artifact_created": False,
        "slice_3_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "support_expansion_ran": False,
    }


def _branch_report(runs: list[dict[str, object]]) -> dict[str, object]:
    seeds = [13, 19, 29, 37, 41, 43]
    terminal_by_seed = {
        str(seed): sum(1 for run in runs if int(run["seed"]) == seed) for seed in seeds
    }
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "aggregate": {
            "branch_point_count": len(seeds),
            "branch_run_count": len(runs),
            "successful_branch_run_count": len(runs),
            "positive_seed_count": sum(1 for value in terminal_by_seed.values() if value),
            "replay_verified": True,
            "unsupported_requested_action_count": 0,
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
) -> dict[str, object]:
    return {
        "branch_id": f"carrion-only-seed-{seed}-branch-0-tick-0-agent-9",
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "base_script": "hydration_safe_carrion_cycle",
        "continuation_script": "balanced_legal_probe",
        "branch_state_digest": f"branch-{seed}",
        "alive_agents": 1,
        "births": 5,
        "deaths": 16,
        "trajectory_path": str(trajectory_path),
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.5,
        "replay_digest": f"replay-{seed}",
        "replay_verified": True,
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
    alive_agents: int,
    births: int,
    deaths: int,
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
                    "action_source": "counterfactual_script:balanced_legal_probe",
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
        "generic_carrion_autopsy_rerun",
        "v180_rerun",
        "v186_rerun",
        "v188_rerun",
        "v189_rerun",
    ):
        test_case.assertFalse(report[key], key)


if __name__ == "__main__":
    unittest.main()
