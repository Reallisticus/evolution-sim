from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_post_carrion_rollout_context_audit
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.post_carrion_rollout_context_audit import (
    MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
    build_post_carrion_rollout_context_audit_report,
)


class MindV3PostCarrionRolloutContextAuditTests(unittest.TestCase):
    def test_post_carrion_rollout_context_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:post-carrion-rollout-context-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_post_carrion_rollout_context_audit"
            ),
        )

    def test_build_report_restores_planner_sections_and_taxonomy(
        self,
    ) -> None:
        dataset = _trajectory_dataset(
            Path("fixture-carrion-only-mind-v3-13-120.jsonl.gz"),
            records=[
                _record(
                    tick=0,
                    requested_action="eat",
                    food_source="carcass",
                    resource_gain=0.2,
                    post_carrion_context=False,
                    non_empty=False,
                ),
                _record(
                    tick=1,
                    requested_action="drink",
                    post_carrion_context=True,
                    non_empty=True,
                    score_delta=0.4,
                ),
            ],
            alive_agents=0,
            births=1,
            deaths=3,
        )

        report = build_post_carrion_rollout_context_audit_report(
            rollout_context_report=_rollout_context_report(),
            baseline_report=_baseline_report(),
            v64_rollout_context_audit=_v64_audit(),
            v69_recovery_action_target_audit=_recovery_action_target_audit(),
            branch_oracle_audit=_branch_oracle_audit(),
            trajectory_datasets=(dataset,),
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
        )
        self.assertTrue(report["contract"]["diagnostics_only"])
        self.assertEqual(report["contract"]["runtime_policy_effect"], "none")
        self.assertEqual(report["contract"]["trained_artifact_effect"], "none")
        self.assertTrue(report["non_promoted"])
        for section in (
            "config",
            "coverage",
            "temporal_alignment",
            "fixture_action_support",
            "policy_scoring_pressure",
            "unsupported_resolution",
            "seed29_birth_regression",
            "research_recommendation",
        ):
            self.assertIn(section, report)
        self.assertEqual(
            report["config"]["rollout_context_report_option"],
            "--rollout-context-report",
        )

        coverage = report["trajectory_rollout_context_coverage"]["aggregate"]
        self.assertEqual(coverage["decision_row_count"], 2)
        self.assertEqual(coverage["rollout_context_post_carrion_context_count"], 1)
        self.assertEqual(coverage["animal_resource_gain_record_count"], 1)
        self.assertEqual(coverage["post_carrion_requested_action_counts"]["drink"], 1)

        fixture = report["fixture_failure_audit"]
        carrion_120 = fixture["fixture_gate_comparison"]["carrion_only_120"]
        self.assertEqual(carrion_120["metric_deltas"]["alive_agents_mean"], -1.0)
        self.assertEqual(
            fixture["branch_oracle_overlap"][
                "exact_action_matched_branch_result_count"
            ],
            1,
        )
        self.assertEqual(
            fixture["branch_oracle_overlap"][
                "exact_action_matched_post_carrion_context_count"
            ],
            1,
        )
        self.assertIn(
            "post_carrion_context_adequate_but_unhelpful",
            report["classification"]["labels"],
        )
        self.assertNotIn(
            "fixture_failure_persists",
            report["classification"]["labels"],
        )
        self.assertEqual(
            report["coverage"]["answer"],
            "post_carrion_context_adequate_but_unhelpful",
        )
        self.assertEqual(
            report["coverage"]["broad_holdout"]["answer"],
            "post_carrion_context_too_rare",
        )
        self.assertEqual(
            report["temporal_alignment"]["answer"],
            "first_recovery_decision_sampling_gap",
        )
        self.assertEqual(
            report["fixture_action_support"]["answer"],
            "heldout_fixture_recovery_support_adequate",
        )
        self.assertEqual(
            report["policy_scoring_pressure"]["answer"],
            "policy_scoring_suppresses_supported_recovery_action",
        )
        self.assertEqual(
            report["unsupported_resolution"]["answer"],
            "unsupported_resolution_movement_mask_drift",
        )
        self.assertEqual(
            report["seed29_birth_regression"]["answer"],
            "seed29_birth_regression_movement_failure",
        )
        self.assertFalse(report["research_recommendation"]["promote_v105"])

    def test_missing_evidence_is_inconclusive_not_an_exception(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.json"
            missing_trajectory = Path(tmpdir) / "missing.jsonl.gz"

            report = build_post_carrion_rollout_context_audit_report(
                rollout_context_report_path=missing,
                baseline_report_path=missing,
                v64_rollout_context_audit_path=missing,
                v69_recovery_action_target_audit_path=missing,
                branch_oracle_audit_path=missing,
                trajectory_paths=(missing_trajectory,),
            )

        self.assertEqual(
            report["classification"]["primary"],
            None,
        )
        self.assertIn("rollout_context_report", report["classification"]["missing_evidence"])
        self.assertIn("trajectory_load_failures", report["classification"]["missing_evidence"])
        self.assertEqual(
            report["evidence"]["trajectories"]["load_failure_count"],
            1,
        )
        for section in (
            "config",
            "coverage",
            "temporal_alignment",
            "fixture_action_support",
            "policy_scoring_pressure",
            "unsupported_resolution",
            "seed29_birth_regression",
            "research_recommendation",
        ):
            self.assertIn(section, report)

    def test_lenient_trajectory_reader_reports_malformed_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "fixture-carrion-only-mind-v3-13-120.jsonl"
            payloads = [
                {"format": "test", "config": {"seed": 13}},
                {"type": "metadata", "note": "not a trajectory record"},
                {
                    "record": _record(
                        tick=1,
                        requested_action="drink",
                        post_carrion_context=True,
                        non_empty=True,
                    )
                },
                {
                    "summary": {
                        "seed": 13,
                        "alive_agents": 1,
                        "births": 0,
                        "deaths": 0,
                        "ticks_executed": 120,
                    },
                    "trajectory_summary": {
                        "record_count": 2,
                        "invalid_resolution_action_count": 0,
                    },
                },
            ]
            trajectory_path.write_text(
                "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads),
                encoding="utf-8",
            )

            report = build_post_carrion_rollout_context_audit_report(
                rollout_context_report=_rollout_context_report(),
                baseline_report=_baseline_report(),
                v64_rollout_context_audit=_v64_audit(),
                v69_recovery_action_target_audit=_recovery_action_target_audit(),
                branch_oracle_audit=_branch_oracle_audit(),
                trajectory_paths=(trajectory_path,),
            )

        self.assertEqual(
            report["evidence"]["trajectories"]["loaded_path_count"],
            1,
        )
        self.assertEqual(
            report["evidence"]["trajectories"]["malformed_record_count"],
            1,
        )
        self.assertEqual(
            report["trajectory_rollout_context_coverage"]["aggregate"][
                "decision_row_count"
            ],
            1,
        )

    def test_cli_writes_report_with_rollout_context_report_option(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rollout_path = tmp_path / "rollout.json"
            baseline_path = tmp_path / "baseline.json"
            v64_path = tmp_path / "v64.json"
            recovery_path = tmp_path / "recovery.json"
            oracle_path = tmp_path / "oracle.json"
            output_path = tmp_path / "v106.json"
            _write_json(rollout_path, _rollout_context_report())
            _write_json(baseline_path, _baseline_report())
            _write_json(v64_path, _v64_audit())
            _write_json(recovery_path, _recovery_action_target_audit())
            _write_json(oracle_path, _branch_oracle_audit())
            stdout = io.StringIO()

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_post_carrion_rollout_context_audit",
                        "--rollout-context-report",
                        str(rollout_path),
                        "--baseline-report",
                        str(baseline_path),
                        "--v64-rollout-context-audit",
                        str(v64_path),
                        "--v69-recovery-action-target-audit",
                        str(recovery_path),
                        "--branch-oracle-audit",
                        str(oracle_path),
                        "--output",
                        str(output_path),
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_v3_post_carrion_rollout_context_audit.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("post_carrion_rollout_context_audit=", stdout.getvalue())
        self.assertEqual(
            report["schema_version"],
            MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["classification"]["primary"],
            None,
        )
        self.assertIn("trajectories", report["classification"]["missing_evidence"])


def _trajectory_dataset(
    path: Path,
    *,
    records: list[dict[str, object]],
    alive_agents: int,
    births: int,
    deaths: int,
) -> TrajectoryJsonlDataset:
    return TrajectoryJsonlDataset(
        path=path,
        header={"config": {"seed": 13}, "format": "test"},
        records=tuple(records),
        footer={
            "summary": {
                "seed": 13,
                "alive_agents": alive_agents,
                "births": births,
                "deaths": deaths,
                "ticks_executed": 120,
            },
            "trajectory_summary": {
                "record_count": len(records),
                "invalid_resolution_action_count": 0,
            },
        },
    )


def _record(
    *,
    tick: int,
    requested_action: str,
    food_source: str | None = None,
    resource_gain: float = 0.0,
    post_carrion_context: bool,
    non_empty: bool,
    score_delta: float = 0.0,
) -> dict[str, object]:
    ate = food_source is not None and resource_gain > 0.0
    return {
        "tick": tick,
        "agent_id": 7,
        "action_source": "mind_v3_autonomous_evolution_policy_v1",
        "requested_action": requested_action,
        "resolved_action": requested_action,
        "action_valid": True,
        "resolution_action_valid": True,
        "action_mask": {action: True for action in ACTION_NAMES},
        "before": {
            "alive": True,
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 1.0,
        },
        "after": {
            "alive": True,
            "energy_ratio": 0.5 + resource_gain,
            "hydration_ratio": 0.8 if requested_action == "drink" else 0.5,
            "health_ratio": 1.0,
        },
        "outcome": {
            "feeding": {
                "ate": ate,
                "food_source": food_source,
                "gained_energy": resource_gain,
            },
            "drinking": {"drank": requested_action == "drink"},
            "movement": {"moved": requested_action.startswith("move_")},
            "resource_gain": resource_gain,
            "died": False,
            "reproduced": False,
        },
        "policy_decision_diagnostics": {
            "rollout_context_non_empty": non_empty,
            "rollout_context_post_carrion_context": post_carrion_context,
            "rollout_context_selected_score_delta": score_delta,
        },
        "policy_update_trace": {
            "rollout_context_update_trace": {
                "previous_context": {
                    "post_carrion_contact": post_carrion_context,
                    "recent_resolved_actions": ["eat"] if non_empty else [],
                    "ticks_since_animal_resource_gain": 1
                    if post_carrion_context
                    else None,
                    "ticks_since_drink": None,
                    "recovery_phase_remaining": 11
                    if post_carrion_context
                    else 0,
                },
                "updated_context": {
                    "post_carrion_contact": post_carrion_context or ate,
                },
            }
        },
    }


def _rollout_context_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "best_candidate": {
            "candidate_id": "g2-c19",
            "generation_index": 2,
            "score": 166.8,
            "controller_metadata": {
                "architecture": (
                    "rollout_context_need_gated_local_navigation_feature_projection_"
                    "linear_action_head_v5"
                ),
                "schema_version": "mind_v3_controller_metadata_v1",
            },
        },
        "holdout_evaluation": {
            "aggregate": {
                "alive_agents_mean": 20.0,
                "births_mean": 10.0,
                "deaths_mean": 8.0,
                "movement_event_rate": 0.3,
                "dominant_requested_action_share": 0.49,
                "heuristic_action_source_count": 0,
                "unsupported_resolved_action_count": 4,
                "unsupported_resolved_action_breakdown": {
                    "by_invalid_reason": {"not_in_resolution_action_mask": 4},
                    "by_requested_action": {"move_east": 4},
                },
                "rollout_context_decision_count": 100,
                "rollout_context_non_empty_count": 90,
                "rollout_context_non_empty_share": 0.9,
                "rollout_context_post_carrion_context_count": 1,
                "rollout_context_post_carrion_context_share": 0.01,
            },
            "runs": [
                {
                    "seed": 29,
                    "alive_agents": 20,
                    "births": 10,
                    "deaths": 8,
                    "movement_event_rate": 0.3,
                    "unsupported_resolved_action_count": 4,
                    "rollout_context_post_carrion_context_count": 1,
                }
            ],
        },
        "fixture_gate": {
            "passed": False,
            "fixture_names": ["carrion_only"],
            "horizon_ticks": [120],
            "blockers": [
                {
                    "fixture": "carrion_only",
                    "ticks": 120,
                    "metric": "alive_agents_mean",
                    "reason": "fixture_alive_floor",
                    "floor": 1.0,
                    "value": 0.0,
                }
            ],
            "per_horizon": {
                "120": {
                    "per_fixture": {
                        "carrion_only": {
                            "passed": False,
                            "metrics": {
                                "alive_agents_mean": 0.0,
                                "births_mean": 1.0,
                                "energy_viability_share_mean": 0.0,
                            },
                            "blockers": [
                                {
                                    "fixture": "carrion_only",
                                    "metric": "alive_agents_mean",
                                    "reason": "fixture_alive_floor",
                                    "floor": 1.0,
                                    "value": 0.0,
                                }
                            ],
                        }
                    }
                }
            },
        },
    }


def _baseline_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "best_candidate": {
            "candidate_id": "g1-c23",
            "generation_index": 1,
            "score": 157.4,
            "controller_metadata": {
                "architecture": (
                    "need_gated_local_navigation_feature_projection_linear_action_"
                    "head_v4"
                ),
                "schema_version": "mind_v3_controller_metadata_v1",
            },
        },
        "holdout_evaluation": {
            "aggregate": {
                "alive_agents_mean": 19.0,
                "births_mean": 9.0,
                "deaths_mean": 9.0,
                "movement_event_rate": 0.35,
                "dominant_requested_action_share": 0.44,
                "heuristic_action_source_count": 0,
                "unsupported_resolved_action_count": 2,
            },
            "runs": [
                {
                    "seed": 29,
                    "alive_agents": 21,
                    "births": 13,
                    "deaths": 12,
                    "movement_event_rate": 0.35,
                    "unsupported_resolved_action_count": 2,
                }
            ],
        },
        "fixture_gate": {
            "passed": True,
            "fixture_names": ["carrion_only"],
            "horizon_ticks": [120],
            "blockers": [],
            "per_horizon": {
                "120": {
                    "per_fixture": {
                        "carrion_only": {
                            "passed": True,
                            "metrics": {
                                "alive_agents_mean": 1.0,
                                "births_mean": 1.0,
                                "energy_viability_share_mean": 0.25,
                            },
                            "blockers": [],
                        }
                    }
                }
            },
        },
    }


def _v64_audit() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_rollout_context_audit_v1",
        "failure_mode_assessment": {
            "materially_improves_v62_failure_mode": False,
            "movement_drink_stay_absolute_rate_reduction": 0.01,
            "post_carrion_absolute_rate_reduction": 0.0,
            "blockers": [{"reason": "below_floor"}],
        },
        "train_heldout_split": {
            "train_record_count": 10,
            "heldout_record_count": 4,
        },
    }


def _recovery_action_target_audit() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_recovery_action_target_alignment_audit_v1",
        "classification": {
            "primary": "first_record_sampling_gap",
            "labels": ["first_record_sampling_gap"],
        },
        "heldout_scoring": {"decision_row_count": 12},
        "oracle_comparison": {
            "loaded": True,
            "matched_row_count": 0,
            "unmatched_row_count": 12,
        },
    }


def _branch_oracle_audit() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_branch_action_oracle_audit_v1",
        "aggregate": {
            "branch_point_count": 1,
            "oracle_changed_action_count": 1,
            "material_oracle_gain_count": 1,
            "terminal_alive_gain_total_vs_logged": 4,
            "birth_gain_total_vs_logged": 2,
            "zero_heuristic_runtime_actions": True,
            "oracle_best_action_counts": {"eat": 1},
        },
        "acceptance": {
            "diagnostic_acceptance_passed": True,
            "materially_supports_branch_action_oracle": True,
            "blockers": [],
        },
        "branch_results": [
            {
                "branch_id": "branch-1",
                "seed": 13,
                "branch_tick": 1,
                "agent_id": 7,
                "logged_action": "drink",
                "oracle_best_action": "eat",
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
