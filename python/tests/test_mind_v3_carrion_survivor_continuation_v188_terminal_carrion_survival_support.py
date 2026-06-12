from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v187_v186_delta_blocker_review as v187,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v188_terminal_carrion_survival_support as v188,
)
from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV188TerminalCarrionSurvivalSupportTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v188-terminal-carrion-survival-support"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v188_terminal_carrion_survival_support"
            ),
        )

    def test_positive_legal_branch_support_routes_to_fresh_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v187_digest = _write_valid_v187_report(tmpdir)
            report = v188.run_carrion_survivor_continuation_v188_terminal_carrion_survival_support(
                v187_report_path=paths["v187_report"],
                output_path=paths["v188_report"],
                support_trajectory_dir=paths["trajectory_dir"],
                expected_v187_report_exact_digest=v187_digest,
                branch_report_override=_positive_branch_report(),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["support_search"]["ran"])
        self.assertTrue(report["legality_action_mask_validation"]["passed"])
        self.assertTrue(report["support_result"]["positive_support_found"])
        self.assertEqual(
            report["support_result"]["evidence_type"],
            "legal_deterministic_branch_continuation",
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v188.POSITIVE_SUPPORT_ROUTE,
        )
        self.assertTrue(
            report["support_result"]["fresh_audit_required_before_slice_3_training"]
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v188_terminal_carrion_"
                "survival_support_positive_legal_branch_support_found_no_training"
            ),
        )
        _assert_closed_lifecycle(self, report)
        self.assertTrue(report["support_generation_ran"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v187_digest_mismatch_fails_closed_without_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _v187_digest = _write_valid_v187_report(tmpdir)
            report = v188.run_carrion_survivor_continuation_v188_terminal_carrion_survival_support(
                v187_report_path=paths["v187_report"],
                output_path=paths["v188_report"],
                support_trajectory_dir=paths["trajectory_dir"],
                expected_v187_report_exact_digest="wrong",
                branch_report_override=_positive_branch_report(),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v187_exact_digest_matches_expected",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["support_search"]["ran"])
        self.assertFalse(report["support_result"]["positive_support_found"])
        self.assertEqual(report["route_decision"]["recommended_next_route"], "stop")
        self.assertFalse(report["support_generation_ran"])
        _assert_closed_lifecycle(self, report)

    def test_v187_wrong_route_fails_closed_without_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _v187_digest = _write_valid_v187_report(
                tmpdir,
                route="wrong_route",
            )
            expected_digest = _write_report(
                paths["v187_report"],
                json.loads(paths["v187_report"].read_text(encoding="utf-8")),
            )
            report = v188.run_carrion_survivor_continuation_v188_terminal_carrion_survival_support(
                v187_report_path=paths["v187_report"],
                output_path=paths["v188_report"],
                support_trajectory_dir=paths["trajectory_dir"],
                expected_v187_report_exact_digest=expected_digest,
                branch_report_override=_positive_branch_report(),
            )

        self.assertFalse(report["source_validation"]["passed"])
        self.assertIn(
            "v187_route_matches_required",
            report["source_validation"]["failures"],
        )
        self.assertFalse(report["support_search"]["ran"])
        self.assertEqual(report["route_decision"]["recommended_next_route"], "stop")
        _assert_closed_lifecycle(self, report)

    def test_no_positive_support_records_bounded_infeasibility_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v187_digest = _write_valid_v187_report(tmpdir)
            report = v188.run_carrion_survivor_continuation_v188_terminal_carrion_survival_support(
                v187_report_path=paths["v187_report"],
                output_path=paths["v188_report"],
                support_trajectory_dir=paths["trajectory_dir"],
                expected_v187_report_exact_digest=v187_digest,
                branch_report_override=_no_positive_branch_report(),
            )

        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["support_search"]["ran"])
        self.assertTrue(report["legality_action_mask_validation"]["passed"])
        self.assertFalse(report["support_result"]["positive_support_found"])
        self.assertEqual(
            report["support_result"]["evidence_type"],
            "bounded_policy_visible_script_space_infeasibility_proof",
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v188.NO_SUPPORT_ROUTE,
        )
        self.assertIsNotNone(report["support_result"]["infeasibility_scope"])
        self.assertTrue(report["support_generation_ran"])
        _assert_closed_lifecycle(self, report)

    def test_cli_writes_report_and_prints_lifecycle_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, v187_digest = _write_valid_v187_report(tmpdir)

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v188_terminal_carrion_survival_support",
                    "--v187-report",
                    str(paths["v187_report"]),
                    "--output",
                    str(paths["v188_report"]),
                    "--support-trajectory-dir",
                    str(paths["trajectory_dir"]),
                    "--expected-v187-report-exact-digest",
                    v187_digest,
                    "--seeds",
                    "13",
                    "--ticks",
                    "1",
                    "--max-branch-points-per-seed",
                    "1",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v188_report"].read_text(encoding="utf-8"))

        self.assertIn("source_validation_passed=True", result.stdout)
        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_3_training_consumed=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertIn("gate_relaxation_allowed=False", result.stdout)
        self.assertTrue(report["source_validation"]["passed"])
        _assert_closed_lifecycle(self, report)
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_v187_report(
    tmpdir: str,
    *,
    route: str = v187.RECOMMENDED_NEXT_ROUTE,
) -> tuple[dict[str, Path], str]:
    root = Path(tmpdir)
    paths = {
        "v187_report": root / "v187-report.json",
        "v188_report": root / "v188-report.json",
        "trajectory_dir": root / "v188-trajectories",
    }
    report = {
        "schema_version": (
            v187.M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_SCHEMA_VERSION
        ),
        "policy": v187.M3_CARRION_SURVIVOR_CONTINUATION_V187_V186_DELTA_BLOCKER_REVIEW_POLICY,
        "route_decision": {
            "recommended_next_route": route,
            "selected_route": route,
            "slice_3_training_allowed": False,
            "runtime_integration_allowed": False,
            "promotion_authorized": False,
        },
        "classification": {
            "primary": (
                "m3_carrion_survivor_continuation_v187_v186_delta_blocker_"
                "review_terminal_survival_support_generation_before_slice_3_"
                "no_training"
            ),
            "labels": ["diagnostics_only"],
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
    return paths, _write_report(paths["v187_report"], report)


def _positive_branch_report() -> dict[str, object]:
    run_13 = _branch_run(seed=13, alive_agents=1, births=3)
    run_19 = _branch_run(seed=19, alive_agents=2, births=4)
    return _branch_report([run_13, run_19], terminal_by_seed={"13": 1, "19": 2})


def _no_positive_branch_report() -> dict[str, object]:
    run_13 = _branch_run(seed=13, alive_agents=0, births=0)
    run_19 = _branch_run(seed=19, alive_agents=0, births=0)
    return _branch_report([run_13, run_19], terminal_by_seed={"13": 0, "19": 0})


def _branch_report(
    branch_runs: list[dict[str, object]],
    *,
    terminal_by_seed: dict[str, int],
) -> dict[str, object]:
    successful = [
        run for run in branch_runs if int(run.get("alive_agents", 0)) > 0
    ]
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": MIND_V3_CARRION_BRANCH_EXPLORE_POLICY,
        "contract": {
            "fixture_name": "carrion_only",
            "ticks": 120,
            "base_script": v188.DEFAULT_BASE_SCRIPT,
            "continuation_scripts": list(v188.DEFAULT_CONTINUATION_SCRIPTS),
            "max_branch_points_per_seed": 1,
        },
        "aggregate": {
            "branch_point_count": 2,
            "branch_run_count": len(branch_runs),
            "successful_branch_run_count": len(successful),
            "positive_seed_count": sum(1 for count in terminal_by_seed.values() if count),
            "replay_verified": True,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
            "terminal_survivor_count_by_seed": terminal_by_seed,
            "trajectory_paths": [
                str(run["trajectory_path"]) for run in branch_runs
            ],
            "terminal_alive_agent_total": sum(
                int(run.get("alive_agents", 0)) for run in branch_runs
            ),
            "best_branch_run": max(
                branch_runs,
                key=lambda run: (
                    int(run.get("alive_agents", 0)),
                    int(run.get("births", 0)),
                ),
            ),
            "unrecoverable_state_summary": {
                "terminal_extinct_runs": len(branch_runs) - len(successful),
            },
        },
        "branch_runs": branch_runs,
    }


def _branch_run(*, seed: int, alive_agents: int, births: int) -> dict[str, object]:
    return {
        "branch_id": f"carrion-only-seed-{seed}-branch-0-tick-0-agent-1",
        "seed": seed,
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": 0,
        "base_script": v188.DEFAULT_BASE_SCRIPT,
        "continuation_script": "hydration_safe_carrion_cycle",
        "branch_state_digest": f"branch-state-{seed}",
        "alive_agents": alive_agents,
        "births": births,
        "deaths": 0,
        "trajectory_path": f"output/mind/test-v188-seed-{seed}.jsonl.gz",
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": 0,
        "dominant_requested_action": "eat",
        "dominant_requested_action_share": 0.4,
        "replay_verification": {
            "verified": True,
            "expected_digest": f"replay-{seed}",
        },
    }


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
        "v186_rerun",
        "v187_rerun",
    ):
        test_case.assertFalse(report[key], key)


def _write_report(path: Path, report: dict[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(payload)
    _write_json(path, report)
    return str(report["exact_digest"])


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
