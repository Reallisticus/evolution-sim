from __future__ import annotations

import json
import gzip
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import (
    mind_v3_carrion_recovery_distill,
    mind_v3_carrion_recovery_residual_audit,
)
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    NAVIGATION_TARGETS,
    OBSERVATION_SCHEMA_VERSION,
    encode_observation_input,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED,
    MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
    build_carrion_recovery_archive_report,
    write_carrion_recovery_archive_report,
)
from evolution_sim.mind.carrion_recovery_distill import (
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_MAX_LINEAR_OVERRIDE_MARGIN,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE,
    MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
    _acceptance,
    _branch_action_log_odds_bias,
    _evaluation_action_balance,
    _trajectory_action_balance,
    build_carrion_recovery_distillation_report,
)
from evolution_sim.mind.carrion_recovery_residual_audit import (
    MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
    _calibration_classification,
    _counter_reconciliation,
    _failure_classification,
    _fixture_diagnostic_summary,
    _load_trajectory_records,
    _shadow_calibration_for_snapshots,
)
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY,
    MindV3NeuralArtifactError,
    score_mind_v3_neural_artifact,
    validate_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy


class MindV3CarrionRecoveryDistillTests(unittest.TestCase):
    def test_recovery_distill_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-recovery-distill"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_recovery_distill"
            ),
        )
        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-recovery-residual-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_recovery_residual_audit"
            ),
        )

    def test_recovery_residual_audit_cli_writes_diagnostics_only_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.json"
            report = {
                "schema_version": (
                    MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION
                ),
                "contract": {
                    "diagnostics_only": True,
                    "runtime_policy_effect": "none",
                    "artifact_promotion_effect": "none",
                },
                "feature_contract_checks": {"passed": True},
                "offline_score_summaries": {
                    "heldout": {
                        "configured_would_change_count": 0,
                        "shadow_forced_gate_would_change_count": 0,
                        "shadow_margin_ignored_would_change_count": 3,
                    }
                },
                "fixture_replay_diagnostics": {
                    "context_gate_pass_count": 4,
                    "residual_applied_count": 0,
                    "actual_changed_linear_count": 0,
                },
                "failure_classification": {"primary": "margin domination"},
            }
            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_residual_audit",
                    "--artifact",
                    "artifact.json",
                    "--distill-report",
                    "distill.json",
                    "--archive-report",
                    "archive.json",
                    "--split-report",
                    "split.json",
                    "--fixture-seeds",
                    "13,19",
                    "--fixture-ticks",
                    "120",
                    "--evaluation-report",
                    "evaluation.json",
                    "--activation-audit",
                    "activation.json",
                    "--margin-sweep",
                    "0,0.008",
                    "--scale-sweep",
                    "0.01,0.03",
                    "--output",
                    str(output_path),
                ],
            ), patch(
                (
                    "evolution_sim.cli.mind_v3_carrion_recovery_residual_audit."
                    "build_carrion_recovery_residual_audit_report"
                ),
                return_value=report,
            ) as build_report, redirect_stdout(stdout):
                mind_v3_carrion_recovery_residual_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
        )
        self.assertTrue(payload["contract"]["diagnostics_only"])
        self.assertEqual(payload["contract"]["runtime_policy_effect"], "none")
        build_report.assert_called_once()
        self.assertEqual(
            build_report.call_args.kwargs["fixture_seeds"],
            (13, 19),
        )
        self.assertEqual(
            build_report.call_args.kwargs["margin_sweep"],
            (0.0, 0.008),
        )
        self.assertEqual(
            build_report.call_args.kwargs["scale_sweep"],
            (0.01, 0.03),
        )
        self.assertIn("classification=margin domination", stdout.getvalue())
        self.assertIn("fixture_residual_applied_count=0", stdout.getvalue())

    def test_recovery_residual_counter_reconciliation_flags_reporting_bug(
        self,
    ) -> None:
        report = _counter_reconciliation(
            evaluation_report={
                "action_balance_diagnostics": {
                    "open": {
                        "mind_v3_recovery_distilled": {
                            "record_count": 10,
                            "requested_action_counts": {"eat": 10},
                            "changed_linear_decision_count": 0,
                            "residual_application_count": 0,
                        }
                    }
                },
                "open": {
                    "comparison": {
                        "mind_v3_recovery_distilled": {
                            "aggregate": {
                                "trajectory_record_count": 10,
                                "requested_action_counts": {"eat": 10},
                                "neural_anchor_diagnostics": {
                                    "decision_count": 10,
                                    "changed_linear_action_count": 2,
                                    "residual_applied_count": 2,
                                    "shadow_reason_counts": {
                                        "context_gate:test": 3,
                                        "none": 2,
                                    },
                                    "linear_anchor_action_counts": {"eat": 10},
                                },
                            }
                        }
                    }
                },
            },
            distill_report={
                "heldout_branch_state_evaluation": {
                    "aggregate_action_balance": {
                        "record_count": 4,
                        "missing_decision_diagnostics_count": 0,
                        "requested_action_counts": {"stay": 4},
                        "changed_linear_decision_count": 0,
                        "residual_application_count": 0,
                    }
                }
            },
            activation_audit={
                "fixture_replay_diagnostics": {
                    "decision_count": 6,
                    "context_gate_pass_count": 3,
                    "context_gate_fail_count": 3,
                    "residual_applied_count": 1,
                    "actual_changed_linear_count": 1,
                    "configured_residual_would_change_count": 1,
                    "safety_guard_shadow_reason_counts": {"none": 1},
                }
            },
            heldout_artifact_replay={
                "decision_count": 4,
                "residual_applied_count": 1,
                "actual_changed_linear_count": 1,
                "configured_would_change_count": 1,
                "shadow_reason_counts": {"none": 1},
            },
            current_fixture_replay={},
        )

        self.assertEqual(report["classification"]["primary"], "reporting bug")
        broad = report["surfaces"][0]
        self.assertEqual(broad["surface"], "broad_open_evaluation")
        self.assertEqual(broad["actual_changed_linear_count"], 0)
        self.assertEqual(broad["configured_would_change_count"], 2)
        self.assertTrue(broad["counter_mismatch"])

    def test_recovery_residual_shadow_calibration_reports_margin_sweep(
        self,
    ) -> None:
        snapshots = (
            {
                "seed": 13,
                "linear_action": "eat",
                "neural_action": "drink",
                "linear_margin": 0.01,
                "gate": "visible_carrion_or_recovery_phase_v1",
                "gate_passed": True,
                "gate_reason": "recovery_phase",
                "safety_guard_reason": "none",
                "action_mask": {"drink": True, "eat": True, "stay": True},
                "linear_scores": {"drink": 0.99, "eat": 1.0, "stay": 0.0},
                "neural_scores": {"drink": 1.0, "eat": 0.0, "stay": 0.0},
            },
            {
                "seed": 19,
                "linear_action": "eat",
                "neural_action": "drink",
                "linear_margin": 0.01,
                "gate": "visible_carrion_or_recovery_phase_v1",
                "gate_passed": True,
                "gate_reason": "recovery_phase",
                "safety_guard_reason": "none",
                "action_mask": {"drink": True, "eat": True, "stay": True},
                "linear_scores": {"drink": 0.99, "eat": 1.0, "stay": 0.0},
                "neural_scores": {"drink": 1.0, "eat": 0.0, "stay": 0.0},
            },
        )

        calibration = _shadow_calibration_for_snapshots(
            snapshots,
            configured_margin=0.0,
            configured_scale=0.05,
            margin_sweep=(0.0, 0.02),
            scale_sweep=(0.01, 0.05),
        )
        classification = _calibration_classification(
            feature_checks={"passed": True},
            failure_classification={"primary": "margin domination"},
            shadow_calibration={
                "heldout_branch_states": calibration,
                "carrion_fixture_rows": calibration,
            },
        )

        self.assertEqual(calibration["configured"]["would_change_count"], 0)
        self.assertGreater(
            calibration["margin_ignored"]["would_change_count"],
            calibration["configured"]["would_change_count"],
        )
        self.assertEqual(
            calibration["margin_threshold_sweep"][1]["would_change_by_seed"],
            {"13": 1, "19": 1},
        )
        self.assertEqual(classification["primary"], "margin domination")

    def test_recovery_residual_shadow_calibration_marks_branch_state_sampling(
        self,
    ) -> None:
        from evolution_sim.mind.carrion_recovery_residual_audit import (
            _shadow_calibration,
        )

        calibration = _shadow_calibration(
            artifact={
                "neural_residual_max_linear_override_margin": 0.008,
                "neural_residual_scale": 0.03,
            },
            heldout_rows=(),
            fixture_records=(),
            margin_sweep=(0.0,),
            scale_sweep=(0.01,),
        )

        self.assertEqual(
            calibration["heldout_branch_state_sampling_policy"],
            "first_record_per_split_trajectory_v1",
        )
        self.assertEqual(
            calibration["carrion_fixture_row_sampling_policy"],
            "all_fixture_decision_rows_v1",
        )

    def test_recovery_residual_audit_classifies_margin_domination(self) -> None:
        classification = _failure_classification(
            feature_checks={"passed": True},
            offline={
                "train": {"neural_score_variance_abs_max": 0.03},
                "heldout": {
                    "neural_score_variance_abs_max": 0.04,
                    "configured_would_change_count": 0,
                    "shadow_forced_gate_would_change_count": 0,
                    "shadow_margin_ignored_would_change_count": 5,
                },
            },
            fixture_replay={
                "decision_count": 12,
                "context_gate_pass_count": 6,
                "safety_guard_shadow_reason_counts": {
                    "linear_margin_guard": 4,
                },
            },
            distill_report={"acceptance": {"promotion_candidate_passed": False}},
        )

        self.assertEqual(classification["primary"], "margin domination")
        self.assertIn(
            "linear_margin_guard_shadowed_fixture_decisions",
            classification["evidence"],
        )
        self.assertIn(
            "margin_ignored_shadow_changes_more_than_configured",
            classification["evidence"],
        )

    def test_recovery_residual_fixture_summary_counts_activation_bins(self) -> None:
        summary = _fixture_diagnostic_summary(
            [
                {
                    "neural_linear_anchor_policy": "linear_anchor",
                    "neural_residual_context_gate_passed": True,
                    "neural_residual_context_gate_reason": "recovery_phase",
                    "neural_residual_shadow_reason": "linear_margin_guard",
                    "neural_residual_effective_scale": 0.0,
                    "linear_anchor_score_margin": 0.25,
                    "neural_residual_recovery_phase_remaining": 3,
                    "neural_top_action": "drink",
                    "linear_anchor_action": "eat",
                    "anchored_action": "eat",
                    "neural_residual_changed_linear_action": False,
                    "neural_residual_applied": False,
                },
                {
                    "neural_linear_anchor_policy": "linear_anchor",
                    "neural_residual_context_gate_passed": False,
                    "neural_residual_context_gate_reason": "no_visible_carrion",
                    "neural_residual_shadow_reason": (
                        "context_gate:visible_carrion_or_recovery_phase_v1"
                    ),
                    "neural_residual_effective_scale": 0.0,
                    "linear_anchor_score_margin": 0.01,
                    "neural_residual_recovery_phase_remaining": 0,
                    "neural_top_action": "eat",
                    "linear_anchor_action": "eat",
                    "anchored_action": "eat",
                    "neural_residual_changed_linear_action": False,
                    "neural_residual_applied": False,
                },
            ],
            suite={
                "fixtures": [
                    {
                        "comparison": {
                            "mind_v3": {
                                "aggregate": {
                                    "alive_agents_mean": 0.0,
                                    "births_mean": 0.0,
                                    "heuristic_action_source_count": 0,
                                }
                            }
                        }
                    }
                ]
            },
            seeds=(13,),
            ticks=120,
        )

        self.assertEqual(summary["decision_count"], 2)
        self.assertEqual(summary["context_gate_pass_count"], 1)
        self.assertEqual(summary["context_gate_fail_count"], 1)
        self.assertEqual(
            summary["safety_guard_shadow_reason_counts"]["linear_margin_guard"],
            1,
        )
        self.assertEqual(summary["recovery_phase_remaining_buckets"]["3_8"], 1)
        self.assertEqual(summary["linear_margin_buckets"]["gt_0.15"], 1)
        self.assertEqual(summary["actual_changed_linear_count"], 0)

    def test_recovery_residual_audit_loads_existing_nonscalar_diagnostics(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"format": "test"}) + "\n")
                handle.write(
                    json.dumps(
                        {
                            "record": {
                                "tick": 0,
                                "agent_id": 7,
                                "requested_action": "eat",
                                "action_mask": _basic_action_mask(),
                                "observation_input": _gated_observation_input(
                                    meat_mode="scavenger",
                                    carcass_energy=1.0,
                                )["observation_input"],
                                "policy_decision_diagnostics": {
                                    "neural_head_predictions": {
                                        "survival": 0.0,
                                    }
                                },
                            }
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.write(json.dumps({"footer": "test"}) + "\n")

            records = _load_trajectory_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["requested_action"], "eat")

    def test_recovery_distill_trains_artifact_and_reports_outcomes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            horizon_path = tmp_path / "horizon-labels.json"
            artifact_path = tmp_path / "artifact.json"
            evaluation_path = tmp_path / "evaluation.json"

            report = build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                horizon_output_path=horizon_path,
                artifact_output_path=artifact_path,
                evaluation_output_path=evaluation_path,
            )

            self.assertEqual(
                report["schema_version"],
                MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
            )
            self.assertTrue(report["acceptance"]["data_path_acceptance_passed"])
            self.assertGreater(report["training"]["selected_trajectory_count"], 0)
            self.assertGreater(report["training"]["trained_record_count"], 0)
            self.assertEqual(
                report["training"]["neural_residual_scale"],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE,
            )
            self.assertEqual(
                report["training"][
                    "neural_residual_max_linear_override_margin"
                ],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_MAX_LINEAR_OVERRIDE_MARGIN,
            )
            self.assertEqual(
                report["training"]["neural_residual_context_gate"],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE,
            )
            self.assertEqual(
                report["training"]["neural_residual_recovery_phase_ticks"],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS,
            )
            recovery_bias = report["training"]["recovery_phase_action_bias"]
            self.assertEqual(
                recovery_bias["policy"],
                MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY,
            )
            self.assertIn("action_bias", recovery_bias)
            self.assertTrue(horizon_path.exists())
            self.assertTrue(artifact_path.exists())
            self.assertTrue(evaluation_path.exists())
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["neural_residual_scale"],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE,
            )
            self.assertEqual(
                artifact["neural_residual_context_gate"],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE,
            )
            self.assertEqual(
                artifact["neural_residual_recovery_phase_ticks"],
                DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS,
            )
            self.assertEqual(
                artifact["recovery_phase_action_bias"]["policy"],
                MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY,
            )
            candidate = report["evaluation"]["open"]["comparison"][
                "mind_v3_recovery_distilled"
            ]["aggregate"]
            self.assertEqual(candidate["heuristic_action_source_count"], 0)
            per_seed = report["evaluation"]["open"]["comparison"][
                "candidate_vs_linear_per_seed"
            ]
            self.assertEqual(len(per_seed), 1)
            self.assertEqual(per_seed[0]["seed"], 29)
            self.assertIn("births_delta", per_seed[0])
            self.assertIn("outcome_metrics", candidate)
            self.assertIn("total_births", candidate["outcome_metrics"])
            self.assertIn(
                "total_scavenger_carcass_events",
                candidate["outcome_metrics"],
            )

    def test_recovery_distill_cli_writes_report_and_prints_counters(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            archive_path = tmp_path / "archive.json"
            output_path = tmp_path / "distill.json"
            horizon_path = tmp_path / "horizon-labels.json"
            artifact_path = tmp_path / "artifact.json"
            evaluation_path = tmp_path / "evaluation.json"
            write_carrion_recovery_archive_report(archive, archive_path)

            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_distill",
                    "--archive-report",
                    str(archive_path),
                    "--horizons",
                    "1",
                    "--hidden-units",
                    "4",
                    "--eval-seeds",
                    "29",
                    "--eval-ticks",
                    "8",
                    "--fixture-names",
                    "carrion_only",
                    "--fixture-seeds",
                    "29",
                    "--fixture-ticks",
                    "8",
                    "--horizon-output",
                    str(horizon_path),
                    "--artifact-output",
                    str(artifact_path),
                    "--evaluation-output",
                    str(evaluation_path),
                    "--output",
                    str(output_path),
                ],
            ), redirect_stdout(stdout):
                mind_v3_carrion_recovery_distill.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
        )
        self.assertTrue(payload["acceptance"]["data_path_acceptance_passed"])
        self.assertIn("open_candidate_total_births=", stdout.getvalue())
        self.assertIn(
            "neural_residual_scale="
            f"{DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE}",
            stdout.getvalue(),
        )
        self.assertIn(
            "neural_residual_context_gate="
            f"{DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE}",
            stdout.getvalue(),
        )
        self.assertIn(
            "neural_residual_recovery_phase_ticks="
            f"{DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS}",
            stdout.getvalue(),
        )
        self.assertIn(
            "recovery_phase_action_bias_policy="
            f"{MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY}",
            stdout.getvalue(),
        )
        self.assertIn(
            "open_candidate_vs_linear_alive_agents_mean_delta=",
            stdout.getvalue(),
        )
        self.assertIn(
            "open_candidate_vs_linear_min_seed_births_delta=",
            stdout.getvalue(),
        )
        self.assertIn(
            "fixture_candidate_carrion_only_total_scavenger_carcass_events=",
            stdout.getvalue(),
        )

    def test_recovery_distill_consumes_train_records_from_split_manifest(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            first_record = dict(archive["dataset"]["records"][0])
            heldout_record = json.loads(json.dumps(first_record))
            heldout_record["record_id"] = "heldout-record-not-used-for-training"
            heldout_record["source"]["branch_id"] = "heldout-branch"
            heldout_record["source"]["trajectory_path"] = str(
                tmp_path / "missing-heldout.jsonl.gz"
            )
            archive["dataset"]["records"] = [first_record, heldout_record]
            archive["dataset"]["record_count"] = 2
            split_report = _split_report(
                train_record=first_record,
                heldout_record=heldout_record,
            )

            report = build_carrion_recovery_distillation_report(
                archive_report=archive,
                archive_split_report=split_report,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
            )

        self.assertTrue(report["training"]["archive_split_consumed"])
        self.assertEqual(report["training"]["selected_trajectory_count"], 1)
        self.assertEqual(
            report["training"]["selected_trajectories"][0]["record_id"],
            first_record["record_id"],
        )
        heldout = report["heldout_branch_state_evaluation"]
        self.assertTrue(heldout["enabled"])
        self.assertEqual(heldout["heldout_record_count"], 1)
        self.assertEqual(heldout["load_failure_count"], 1)
        self.assertFalse(report["promotion"]["promoted"])
        action_balance = report["evaluation"]["action_balance_diagnostics"]
        self.assertIn("candidate_vs_linear_delta", action_balance["open"])

    def test_live_evaluation_action_balance_uses_neural_anchor_counters(
        self,
    ) -> None:
        evaluation = {
            "open": {
                "comparison": {
                    "mind_v3_linear": {
                        "runs": [
                            {
                                "requested_action_counts": {"eat": 10},
                                "resolved_action_counts": {"eat": 10},
                            }
                        ]
                    },
                    "mind_v3_recovery_distilled": {
                        "runs": [
                            {
                                "requested_action_counts": {"eat": 8, "drink": 2},
                                "resolved_action_counts": {"eat": 8, "drink": 2},
                                "neural_anchor_diagnostics": {
                                    "changed_linear_action_count": 2,
                                    "residual_applied_count": 3,
                                },
                            }
                        ]
                    },
                }
            },
            "fixture": {
                "linear_baseline_suite": {
                    "evaluated_policy_key": "mind_v3",
                    "fixtures": [
                        {
                            "fixture": "carrion_only",
                            "comparison": {
                                "mind_v3": {
                                    "aggregate": {
                                        "requested_action_counts": {"eat": 10},
                                        "dominant_requested_action": "eat",
                                        "dominant_requested_action_share": 1.0,
                                    }
                                }
                            },
                        }
                    ],
                },
                "candidate_suite": {
                    "evaluated_policy_key": "mind_v3",
                    "fixtures": [
                        {
                            "fixture": "carrion_only",
                            "comparison": {
                                "mind_v3": {
                                    "aggregate": {
                                        "requested_action_counts": {
                                            "eat": 7,
                                            "drink": 3,
                                        },
                                        "dominant_requested_action": "eat",
                                        "dominant_requested_action_share": 0.7,
                                        "neural_anchor_diagnostics": {
                                            "changed_linear_action_count": 4,
                                            "residual_applied_count": 5,
                                        },
                                    }
                                }
                            },
                        }
                    ],
                },
            },
        }

        balance = _evaluation_action_balance(evaluation)

        open_candidate = balance["open"]["mind_v3_recovery_distilled"]
        self.assertEqual(open_candidate["changed_linear_decision_count"], 2)
        self.assertEqual(open_candidate["residual_application_count"], 3)
        self.assertEqual(
            balance["open"]["candidate_vs_linear_delta"][
                "changed_linear_decision_count_delta"
            ],
            2,
        )
        fixture_candidate = balance["fixture"]["mind_v3_recovery_distilled"][
            "carrion_only"
        ]
        self.assertEqual(fixture_candidate["changed_linear_decision_count"], 4)
        self.assertEqual(fixture_candidate["residual_application_count"], 5)

    def test_logged_trajectory_action_balance_does_not_invent_live_residuals(
        self,
    ) -> None:
        balance = _trajectory_action_balance(
            [
                {
                    "requested_action": "eat",
                    "resolved_action": "eat",
                    "action_source": "counterfactual_script:water_rescue",
                }
            ]
        )

        self.assertEqual(balance["changed_linear_decision_count"], 0)
        self.assertEqual(balance["residual_application_count"], 0)
        self.assertEqual(balance["missing_decision_diagnostics_count"], 1)

    def test_recovery_distill_blocks_per_seed_open_regression(self) -> None:
        report = {
            "training": {
                "selected_trajectory_count": 1,
                "trained_record_count": 1,
            },
            "evaluation": {
                "open": {
                    "comparison": {
                        "mind_v3_recovery_distilled": {
                            "aggregate": {
                                "alive_agents_mean": 10.0,
                                "births_mean": 5.0,
                                "heuristic_action_source_count": 0,
                            }
                        },
                        "mind_v3_linear": {
                            "aggregate": {
                                "alive_agents_mean": 10.0,
                                "births_mean": 5.0,
                            }
                        },
                        "candidate_vs_linear_per_seed": [
                            {
                                "seed": 13,
                                "alive_agents_delta": 0,
                                "births_delta": -1,
                            }
                        ],
                    }
                },
                "fixture": {"candidate_vs_linear_summary": {}},
            },
        }

        acceptance = _acceptance(report)

        self.assertFalse(acceptance["promotion_candidate_passed"])
        self.assertIn(
            "open_seed_13_birth_regression_vs_linear",
            acceptance["promotion_blockers"],
        )
        self.assertEqual(
            acceptance["open_per_seed_regression_summary"]["min_births_delta"],
            -1.0,
        )

    def test_recovery_distill_blocks_missing_per_seed_open_delta_rows(self) -> None:
        report = {
            "training": {
                "selected_trajectory_count": 1,
                "trained_record_count": 1,
            },
            "evaluation": {
                "open": {
                    "seeds": [5, 13],
                    "comparison": {
                        "mind_v3_recovery_distilled": {
                            "aggregate": {
                                "alive_agents_mean": 10.0,
                                "births_mean": 5.0,
                                "heuristic_action_source_count": 0,
                            }
                        },
                        "mind_v3_linear": {
                            "aggregate": {
                                "alive_agents_mean": 10.0,
                                "births_mean": 5.0,
                            }
                        },
                        "candidate_vs_linear_per_seed": [
                            {
                                "seed": 5,
                                "alive_agents_delta": 0,
                                "births_delta": 0,
                            }
                        ],
                    },
                },
                "fixture": {"candidate_vs_linear_summary": {}},
            },
        }

        acceptance = _acceptance(report)

        self.assertFalse(acceptance["promotion_candidate_passed"])
        self.assertIn(
            "open_per_seed_delta_coverage_mismatch",
            acceptance["promotion_blockers"],
        )
        self.assertEqual(
            acceptance["open_per_seed_regression_summary"]["expected_seed_count"],
            2,
        )
        self.assertEqual(
            acceptance["open_per_seed_regression_summary"]["missing_seeds"],
            [13],
        )

    def test_recovery_distill_rejects_unknown_residual_context_gate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            artifact_path = tmp_path / "artifact.json"
            build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                artifact_output_path=artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        artifact["neural_residual_context_gate"] = "typo_gate_v1"
        with self.assertRaises(MindV3NeuralArtifactError):
            validate_mind_v3_neural_artifact(artifact)

    def test_recovery_distill_rejects_unknown_recovery_action_bias_policy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            artifact_path = tmp_path / "artifact.json"
            build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                artifact_output_path=artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        artifact["recovery_phase_action_bias"]["policy"] = "typo_bias_v1"
        with self.assertRaises(MindV3NeuralArtifactError):
            validate_mind_v3_neural_artifact(artifact)

    def test_recovery_action_bias_changes_scores_only_during_recovery(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            artifact_path = tmp_path / "artifact.json"
            build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                artifact_output_path=artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        action_bias = {action: 0.0 for action in ACTION_NAMES}
        action_bias["drink"] = 1.0
        artifact["recovery_phase_action_bias"] = {
            "policy": MIND_V3_NEURAL_RECOVERY_PHASE_ACTION_BIAS_POLICY,
            "scale": 1.0,
            "max_abs_bias": 1.0,
            "action_bias": action_bias,
            "survivor_action_weight": {action: 0.0 for action in ACTION_NAMES},
            "failure_action_weight": {action: 0.0 for action in ACTION_NAMES},
            "survivor_total_weight": 0.0,
            "failure_total_weight": 0.0,
            "record_count": 0,
        }
        validate_mind_v3_neural_artifact(artifact)

        observation = _gated_observation_input(
            meat_mode="scavenger",
            carcass_energy=0.0,
        )
        base = score_mind_v3_neural_artifact(
            artifact=artifact,
            observation_input=observation["observation_input"],
            action_mask=_basic_action_mask(),
        )
        recovered = score_mind_v3_neural_artifact(
            artifact=artifact,
            observation_input=observation["observation_input"],
            action_mask=_basic_action_mask(),
            recovery_phase_remaining=1,
        )

        self.assertAlmostEqual(recovered["drink"] - base["drink"], 1.0)
        for action in _basic_action_mask():
            if action == "drink" or action not in base:
                continue
            self.assertEqual(recovered[action], base[action])

    def test_recovery_action_bias_ignores_unobserved_actions_when_centering(
        self,
    ) -> None:
        survivor_counts = {action: 0.0 for action in ACTION_NAMES}
        failure_counts = {action: 0.0 for action in ACTION_NAMES}
        survivor_counts.update({"drink": 1.0, "stay": 9.0})
        failure_counts.update({"drink": 9.0, "stay": 1.0})

        bias = _branch_action_log_odds_bias(
            survivor_counts=survivor_counts,
            failure_counts=failure_counts,
            scale=1.0,
            max_abs_bias=1.0,
        )

        self.assertLess(bias["drink"], 0.0)
        self.assertGreater(bias["stay"], 0.0)
        self.assertEqual(bias["attack_north"], 0.0)

    def test_recovery_distill_residual_is_context_gated(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            artifact_path = tmp_path / "artifact.json"
            report = build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                artifact_output_path=artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["training"]["neural_residual_context_gate"],
            DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE,
        )
        policy = MindV3EvolutionPolicy(seed=29, neural_artifact=artifact)
        action_mask = _basic_action_mask()

        no_context = policy.decide(
            _gated_observation_input(meat_mode="none", carcass_energy=0.0),
            action_mask,
        )
        self.assertFalse(
            no_context.diagnostics["neural_residual_context_gate_passed"]
        )
        self.assertEqual(
            no_context.diagnostics["neural_residual_effective_scale"],
            0.0,
        )
        self.assertEqual(
            no_context.diagnostics["neural_residual_shadow_reason"],
            f"context_gate:{DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE}",
        )

        carrion_context = policy.decide(
            _gated_observation_input(meat_mode="scavenger", carcass_energy=1.0),
            action_mask,
        )
        self.assertTrue(
            carrion_context.diagnostics["neural_residual_context_gate_passed"]
        )
        self.assertEqual(
            carrion_context.diagnostics["neural_residual_context_gate_reason"],
            "visible_carrion_scavenger",
        )

    def test_recovery_residual_preserves_linear_eat_on_local_animal_resource(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            artifact_path = tmp_path / "artifact.json"
            build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                artifact_output_path=artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        artifact["neural_residual_scale"] = 1.0
        artifact["neural_residual_max_linear_override_margin"] = 1.0
        artifact["action_output_bias"] = {action: 0.0 for action in ACTION_NAMES}
        artifact["action_output_bias"]["move_east"] = 100.0
        validate_mind_v3_neural_artifact(artifact)

        policy = MindV3EvolutionPolicy(seed=29, neural_artifact=artifact)
        decision = policy.decide(
            _gated_observation_input(meat_mode="scavenger", carcass_energy=1.0),
            _basic_action_mask(),
        )

        self.assertEqual(decision.diagnostics["linear_anchor_action"], "eat")
        self.assertEqual(decision.diagnostics["neural_top_action"], "move_east")
        self.assertEqual(
            decision.diagnostics["neural_residual_shadow_reason"],
            "linear_local_animal_resource_eat_guard",
        )
        self.assertEqual(
            decision.diagnostics["neural_residual_effective_scale"],
            0.0,
        )
        self.assertEqual(decision.requested_action, "eat")

    def test_recovery_distill_residual_gate_carries_after_carrion_contact(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive = _build_small_recovery_archive(tmp_path)
            artifact_path = tmp_path / "artifact.json"
            build_carrion_recovery_distillation_report(
                archive_report=archive,
                horizons=(1,),
                hidden_units=4,
                eval_seeds=(29,),
                eval_ticks=8,
                fixture_names=("carrion_only",),
                fixture_seeds=(29,),
                fixture_ticks=8,
                neural_residual_recovery_phase_ticks=1,
                artifact_output_path=artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        policy = MindV3EvolutionPolicy(seed=29, neural_artifact=artifact)
        action_mask = _basic_action_mask()
        scavenger_no_carrion = _gated_observation_input(
            meat_mode="scavenger",
            carcass_energy=0.0,
        )
        initial = policy.decide(scavenger_no_carrion, action_mask)
        self.assertFalse(initial.diagnostics["neural_residual_context_gate_passed"])

        update = policy.observe_transition(
            _animal_resource_contact_record(
                agent_id=7,
                observation=scavenger_no_carrion,
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
            )
        )

        self.assertIsNotNone(update)
        assert update is not None
        self.assertTrue(update["neural_residual_recovery_phase_activated"])
        self.assertEqual(update["neural_residual_recovery_phase_remaining"], 1)

        carried = policy.decide(scavenger_no_carrion, action_mask)
        self.assertTrue(carried.diagnostics["neural_residual_context_gate_passed"])
        self.assertEqual(
            carried.diagnostics["neural_residual_context_gate_reason"],
            "recovery_phase",
        )
        self.assertEqual(
            carried.diagnostics["neural_residual_recovery_phase_remaining"],
            1,
        )

        policy.observe_transition(
            _non_contact_record(
                agent_id=7,
                observation=scavenger_no_carrion,
                policy_id=policy.policy_id,
                policy_version=policy.policy_version,
            )
        )
        expired = policy.decide(scavenger_no_carrion, action_mask)
        self.assertFalse(expired.diagnostics["neural_residual_context_gate_passed"])


def _build_small_recovery_archive(tmp_path: Path) -> dict[str, object]:
    report = build_carrion_recovery_archive_report(
        seeds=(29,),
        ticks=8,
        continuation_scripts=("hydration_safe_carrion_cycle",),
        trajectory_output_dir=tmp_path / "trajectories",
        max_dataset_records_per_class=1,
        min_survivor_cells=1,
        min_failure_cells=0,
    )
    if not report["acceptance"]["archive_acceptance_passed"]:
        raise AssertionError(report["acceptance"])
    return report


def _split_report(
    *,
    train_record: dict[str, object],
    heldout_record: dict[str, object],
) -> dict[str, object]:
    train_ref = _split_record_ref(train_record, split_key="train-branch")
    heldout_ref = _split_record_ref(heldout_record, split_key="heldout-branch")
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
        "split_policy": (
            MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED
        ),
        "records": {
            "train": [train_ref],
            "heldout": [heldout_ref],
        },
        "record_ids": {
            "train": [train_record["record_id"]],
            "heldout": [heldout_record["record_id"]],
        },
        "branch_state_keys": {
            "train": ["train-branch"],
            "heldout": ["heldout-branch"],
        },
        "aggregate": {
            "train_record_count": 1,
            "heldout_record_count": 1,
            "train_survivor_count": 1,
            "train_failure_count": 0,
            "heldout_survivor_count": 1,
            "heldout_failure_count": 0,
        },
        "leakage_check": {
            "passed": True,
            "overlapping_branch_state_keys": [],
        },
        "acceptance": {
            "validation_passed": True,
            "training_blocked": False,
            "blockers": [],
        },
    }


def _split_record_ref(
    record: dict[str, object],
    *,
    split_key: str,
) -> dict[str, object]:
    source = record["source"]
    label = record["label"]
    return {
        "record_id": record["record_id"],
        "dataset_record_index": 0,
        "seed": source["seed"],
        "branch_id": source["branch_id"],
        "branch_state_digest": split_key,
        "branch_state_key": split_key,
        "branch_state_key_type": "branch_state_digest",
        "branch_tick": source["branch_tick"],
        "continuation_script": source["continuation_script"],
        "trajectory_path": source["trajectory_path"],
        "terminal_survivor": label["terminal_survivor"],
        "outcome_class": label["outcome_class"],
        "trajectory_record_count": 1,
    }


def _basic_action_mask() -> dict[str, bool]:
    legal = {"drink", "eat", "move_east", "move_north", "move_south", "move_west", "stay"}
    return {action: action in legal for action in ACTION_NAMES}


def _gated_observation_input(
    *,
    meat_mode: str,
    carcass_energy: float,
) -> dict[str, object]:
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "metadata": {"agent_id": 7},
        "self": {
            "energy_ratio": 0.72,
            "hydration_ratio": 0.7,
            "health_ratio": 1.0,
            "injury_load": 0.0,
            "age_norm": 0.2,
            "reproduction_ready": False,
            "matched_diet_ratio": 1.0,
            "trophic_role": "carnivore" if meat_mode == "scavenger" else "herbivore",
            "meat_mode": meat_mode,
            "season": "wet",
            "water_access_reason": "none",
            "hydrology_support_code": 1,
            "refuge_score": 0.0,
            "hazard_type": "none",
            "hazard_level": 0.0,
            "tile_vegetation": 0.0,
            "tile_recovery_debt": 0.0,
            "reproductive_stage": "stage0_asexual",
            "reproductive_expression": "asexual",
            "sexual_reproduction_unlocked": False,
            "reproductive_signal": 0.0,
            "communication_signal": 0.0,
            "mind_inheritance_available": False,
        },
        "local_patch": [
            _patch_cell(dx, dy, carcass_energy=carcass_energy if dx == 0 and dy == 0 else 0.0)
            for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
            for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
        ],
        "navigation": {
            target: {
                "dx": 0,
                "dy": 0,
                "distance": 0,
                "strength": carcass_energy if target == "carrion" else 0.0,
            }
            for target in NAVIGATION_TARGETS
        },
    }
    return {
        "metadata": {"agent_id": 7},
        "self": observation["self"],
        "observation_input": encode_observation_input(observation),
    }


def _animal_resource_contact_record(
    *,
    agent_id: int,
    observation: dict[str, object],
    policy_id: str,
    policy_version: str,
) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "tick": 1,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "requested_action": "eat",
        "resolved_action": "eat",
        "action_valid": True,
        "resolution_action_valid": True,
        "observation_input": observation["observation_input"],
        "before": {
            "alive": True,
            "energy_ratio": 0.35,
            "hydration_ratio": 0.55,
            "health_ratio": 0.9,
        },
        "after": {
            "alive": True,
            "energy_ratio": 0.62,
            "hydration_ratio": 0.58,
            "health_ratio": 0.91,
        },
        "outcome": {
            "feeding": {
                "ate": True,
                "food_source": "carcass",
                "gained_energy": 0.27,
            },
            "drinking": {"drank": False},
            "movement": {"moved": False},
            "passive": {},
            "resource_gain": 0.27,
            "reproduced": False,
            "died": False,
            "reproduction_ready_after": False,
        },
    }


def _non_contact_record(
    *,
    agent_id: int,
    observation: dict[str, object],
    policy_id: str,
    policy_version: str,
) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "tick": 2,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "requested_action": "stay",
        "resolved_action": "stay",
        "action_valid": True,
        "resolution_action_valid": True,
        "observation_input": observation["observation_input"],
        "before": {
            "alive": True,
            "energy_ratio": 0.62,
            "hydration_ratio": 0.58,
            "health_ratio": 0.91,
        },
        "after": {
            "alive": True,
            "energy_ratio": 0.6,
            "hydration_ratio": 0.56,
            "health_ratio": 0.9,
        },
        "outcome": {
            "feeding": {"ate": False},
            "drinking": {"drank": False},
            "movement": {"moved": False},
            "passive": {},
            "resource_gain": 0.0,
            "reproduced": False,
            "died": False,
            "reproduction_ready_after": False,
        },
    }


def _patch_cell(dx: int, dy: int, *, carcass_energy: float) -> dict[str, object]:
    return {
        "dx": dx,
        "dy": dy,
        "in_bounds": True,
        "terrain": "plain",
        "occupant": "self" if dx == 0 and dy == 0 else "none",
        "same_lineage": False,
        "water_access_reason": "none",
        "food": 0.0,
        "vegetation": 0.0,
        "recovery_debt": 0.0,
        "fresh_kill_energy": 0.0,
        "carcass_energy": carcass_energy,
        "hazard_type": "none",
        "hazard_level": 0.0,
        "ecology_state": "stable",
        "prey_biomass": 0.0,
        "carrion_signal": carcass_energy,
        "predator_risk": 0.0,
        "reproductive_signal": 0.0,
        "communication_signal": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
