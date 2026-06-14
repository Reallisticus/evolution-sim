from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training as v195,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response as v196,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV196FailureResponseTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v196-repaired-contract-slice-3-failure-response"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response"
            ),
        )

    def test_valid_v195_failure_classifies_coverage_collapse_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            report = v196.run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response(
                v195_report_path=paths["v195_report"],
                v195_artifact_path=paths["v195_artifact"],
                v186_report_path=paths["v186_report"],
                output_path=paths["v196_report"],
                expected_v195_report_exact_digest=report_digest,
                expected_v195_artifact_digest=artifact_digest,
                backup_metadata_override={"passed": True},
                run_diagnostic_replay=False,
                lookup_diagnostics_override=_coverage_collapse_lookup_diagnostics(),
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertTrue(report["failure_fact_validation"]["passed"])
        self.assertEqual(
            report["mechanism_analysis"]["primary_mechanism"],
            v196.PRIMARY_COVERAGE_COLLAPSE_MECHANISM,
        )
        self.assertTrue(report["mechanism_analysis"]["confirmed"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v196.COVERAGE_OR_ABSTENTION_REPAIR_ROUTE,
        )
        self.assertTrue(
            report["action_distribution_comparison"][
                "label_imbalance_alone_ruled_out"
            ]
        )
        self.assertFalse(
            report["lookup_coverage_diagnostics"]["combined"][
                "missing_states_defaulted_to_stay"
            ]
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["gate_relaxation_allowed"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v195_report_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            report = v196.run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response(
                v195_report_path=paths["v195_report"],
                v195_artifact_path=paths["v195_artifact"],
                v186_report_path=paths["v186_report"],
                output_path=paths["v196_report"],
                expected_v195_report_exact_digest="wrong",
                expected_v195_artifact_digest=artifact_digest,
                backup_metadata_override={"passed": True},
                run_diagnostic_replay=False,
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn(
            "v195_report_exact_digest_matches_expected",
            report["source_pin_validation"]["failures"],
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v196.STOP_ROUTE,
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])

    def test_unexpected_failure_facts_route_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            payload = json.loads(paths["v195_report"].read_text(encoding="utf-8"))
            payload["acceptance"] = dict(payload["acceptance"])
            payload["acceptance"]["dominant_requested_action_share"] = 0.42
            report_digest = _write_report(paths["v195_report"], payload)

            report = v196.run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response(
                v195_report_path=paths["v195_report"],
                v195_artifact_path=paths["v195_artifact"],
                v186_report_path=paths["v186_report"],
                output_path=paths["v196_report"],
                expected_v195_report_exact_digest=report_digest,
                expected_v195_artifact_digest=artifact_digest,
                backup_metadata_override={"passed": True},
                run_diagnostic_replay=False,
            )

        self.assertTrue(report["source_pin_validation"]["passed"])
        self.assertFalse(report["failure_fact_validation"]["passed"])
        self.assertIn(
            "dominant_requested_action_share_matches_expected",
            report["failure_fact_validation"]["failures"],
        )
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v196.STOP_ROUTE,
        )

    def test_cli_writes_report_without_training_when_replay_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, artifact_digest = _write_valid_inputs(tmpdir)
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response",
                    "--v195-report",
                    str(paths["v195_report"]),
                    "--v195-artifact",
                    str(paths["v195_artifact"]),
                    "--v186-report",
                    str(paths["v186_report"]),
                    "--output",
                    str(paths["v196_report"]),
                    "--expected-v195-report-exact-digest",
                    report_digest,
                    "--expected-v195-artifact-digest",
                    artifact_digest,
                    "--backup-doc",
                    str(paths["backup_doc"]),
                    "--skip-diagnostic-replay",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(paths["v196_report"].read_text(encoding="utf-8"))

        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("slice_4_training_consumed=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["slice_4_training_consumed"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            v196.INSTRUMENTATION_ROUTE,
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])


def _write_valid_inputs(tmpdir: str) -> tuple[dict[str, Path], str, str]:
    root = Path(tmpdir)
    paths = {
        "v195_report": root / "v195-report.json",
        "v195_artifact": root / "v195-artifact.json",
        "v186_report": root / "v186-report.json",
        "v196_report": root / "v196-report.json",
        "backup_doc": root / "backup.md",
    }
    artifact = _valid_v195_artifact()
    artifact_digest = stable_payload_digest(artifact)
    _write_json(paths["v195_artifact"], artifact)
    report = _valid_v195_report(artifact_digest)
    report_digest = _write_report(paths["v195_report"], report)
    _write_report(paths["v186_report"], _valid_v186_report())
    paths["backup_doc"].write_text(
        "\n".join(
            [
                v196.EXPECTED_V195_BACKUP,
                v196.EXPECTED_V195_BACKUP_SHA256,
                "rclone check verification of `0` differences and `1` matching file.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths, report_digest, artifact_digest


def _valid_v195_artifact() -> dict[str, object]:
    return {
        "schema_version": v195.MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        "artifact_policy": v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_ARTIFACT_POLICY,
        "model_id": v195.MIND_V3_TRANSITION_VALUE_MODEL_ID,
        "runtime_action_selection_authorized": False,
        "promotion_authorized": False,
        "built_from": {
            "source": "v194_repaired_contract_terminal_survival_support_compact_dataset",
            "training_row_count": 3599,
            "dataset_digest": v196.EXPECTED_V194_COMPACT_DATASET_DIGEST,
            "authorization_report_exact_digest": (
                v196.EXPECTED_V194_REPORT_EXACT_DIGEST
            ),
            "authorization_route": v195.EXPECTED_V194_ROUTE,
            "feature_key_limit": 8,
            "global_feature_key_excluded": True,
            "training_slice_index": 3,
            "previous_training_slices_consumed": 2,
            "campaign_slice_cap": 10,
            "slice_3_opt_in_training_slice": True,
            "runtime_action_selection_authorized": False,
            "promotion_authorized": False,
        },
        "utility_tables": {
            "policy": "v195_test_feature_action_utility",
            "feature_action_utility": {
                "mask=11000000000000000000": {
                    "eat": {"count": 2, "utility_mean": 6.7, "component_means": {}},
                    "stay": {"count": 2, "utility_mean": 6.1, "component_means": {}},
                }
            },
        },
        "action_support_counts": {
            "drink": 494,
            "eat": 611,
            "move_east": 478,
            "move_north": 673,
            "move_south": 317,
            "move_west": 279,
            "stay": 747,
        },
    }


def _valid_v195_report(artifact_digest: str) -> dict[str, object]:
    regressions = [
        {
            "suite": "broad",
            "seed": seed,
            "baseline_alive_agents": 20,
            "candidate_alive_agents": 20 + deltas["alive_agents_delta"],
            "alive_agents_delta": deltas["alive_agents_delta"],
            "baseline_births": 20,
            "candidate_births": 20 + deltas["births_delta"],
            "births_delta": deltas["births_delta"],
        }
        for seed, deltas in sorted(v196.EXPECTED_BROAD_REGRESSIONS.items())
    ]
    return {
        "schema_version": (
            v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_SCHEMA_VERSION
        ),
        "policy": (
            v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_REPAIRED_CONTRACT_TERMINAL_SURVIVAL_SUPPORT_TRAINING_POLICY
        ),
        "artifact": {"created": True, "digest": artifact_digest},
        "authorization_report_validation": {
            "passed": True,
            "observed_exact_digest": v196.EXPECTED_V194_REPORT_EXACT_DIGEST,
            "observed_dataset_digest": v196.EXPECTED_V194_COMPACT_DATASET_DIGEST,
            "observed_training_route": v195.EXPECTED_V194_ROUTE,
        },
        "dataset": {
            "dataset_digest": v196.EXPECTED_V194_COMPACT_DATASET_DIGEST,
            "row_count": 3599,
        },
        "source_validation": {"passed": True},
        "training": {
            "ran": True,
            "passed": True,
            "training_slice_index": 3,
            "action_support_counts": {
                "drink": 494,
                "eat": 611,
                "move_east": 478,
                "move_north": 673,
                "move_south": 317,
                "move_west": 279,
                "stay": 747,
            },
            "dominant_training_action": "stay",
            "dominant_training_action_share": 0.207558,
        },
        "training_slice_budget": {
            "this_slice_consumed": True,
            "current_slices_consumed": 3,
        },
        "evaluation": {
            "broad": {
                "candidate": {
                    "aggregate": {
                        "requested_action_counts": {"eat": 7234, "stay": 1030},
                        "dominant_requested_action": "eat",
                        "dominant_requested_action_share": 0.7632,
                    }
                }
            },
            "controlled_fixture": {
                "candidate": {
                    "aggregate": {
                        "requested_action_counts": {"eat": 4891, "stay": 26},
                        "dominant_requested_action": "eat",
                        "dominant_requested_action_share": 0.9422,
                    }
                }
            },
        },
        "acceptance": {
            "passed": False,
            "controlled_fixture": {"total_terminal_alive_agents": 0},
            "dominant_requested_action_share": 0.9422,
            "heuristic_action_source_count": 0,
            "per_seed_alive_birth_regressions": regressions,
            "blockers": [
                {"reason": "carrion_only_terminal_survivors_zero"},
                {"reason": "dominant_requested_action_share_above_limit"},
                {"reason": "per_seed_alive_or_birth_regression"},
            ],
        },
        "route_decision": {
            "recommended_next_route": v196.EXPECTED_V195_ROUTE,
            "selected_route": v196.EXPECTED_V195_ROUTE,
        },
        "classification": {
            "primary": (
                "m3_carrion_survivor_continuation_v195_repaired_contract_terminal_"
                "survival_support_training_slice_3_shadow_acceptance_failed_routes_"
                "to_failure_response"
            )
        },
        "training_ran": True,
        "training_artifact_created": True,
        "slice_3_training_consumed": True,
        "shadow_eval_ran": True,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
    }


def _valid_v186_report() -> dict[str, object]:
    report = {
        "schema_version": "v186",
        "acceptance": {
            "passed": False,
            "dominant_requested_action_share": 0.4179,
            "per_seed_alive_birth_regressions": [],
            "controlled_fixture": {"total_terminal_alive_agents": 0},
        },
    }
    report["exact_digest"] = v196.digest_without_exact(report)
    return report


def _coverage_collapse_lookup_diagnostics() -> dict[str, object]:
    broad = _lookup_payload(
        decision_count=100,
        override_applied_count=70,
        mask_only_count=68,
        miss_count=30,
        applied_eat_count=65,
        final_eat_count=75,
        missing_final={"eat": 12, "stay": 5, "move_north": 13},
    )
    carrion = _lookup_payload(
        decision_count=100,
        override_applied_count=96,
        mask_only_count=95,
        miss_count=2,
        applied_eat_count=94,
        final_eat_count=94,
        missing_final={"eat": 1, "stay": 1},
    )
    combined = v196.combine_lookup_scopes(broad, carrion)
    return {
        "policy": "test_lookup_diagnostics",
        "ran": True,
        "broad": broad,
        "carrion_only": carrion,
        "combined": combined,
    }


def _lookup_payload(
    *,
    decision_count: int,
    override_applied_count: int,
    mask_only_count: int,
    miss_count: int,
    applied_eat_count: int,
    final_eat_count: int,
    missing_final: dict[str, int],
) -> dict[str, object]:
    stats = v196._empty_lookup_stats()
    stats["decision_count"] = decision_count
    stats["supported_score_count"] = decision_count - miss_count
    stats["observed_support_floor_satisfied_count"] = override_applied_count
    stats["clear_best_count"] = decision_count - miss_count
    stats["override_applied_count"] = override_applied_count
    stats["runtime_action_selection_changed_count"] = override_applied_count
    stats["missing_supported_score_count"] = miss_count
    stats["source_key_category_counts"] = {
        "mask_only_hit": mask_only_count,
        "miss": miss_count,
        "coarse_feature_hit": decision_count - mask_only_count - miss_count,
    }
    stats["applied_override_action_counts"] = {
        "eat": applied_eat_count,
        "stay": override_applied_count - applied_eat_count,
    }
    stats["predicted_action_counts"] = dict(stats["applied_override_action_counts"])
    stats["final_requested_action_counts"] = {
        "eat": final_eat_count,
        "stay": decision_count - final_eat_count,
    }
    stats["override_rejected_reason_counts"] = {
        "missing_supported_scores_for_valid_actions": miss_count,
        "low_observed_support_for_valid_actions": decision_count
        - miss_count
        - override_applied_count,
    }
    stats["final_requested_action_counts_by_rejected_reason"] = {
        "missing_supported_scores_for_valid_actions": missing_final
    }
    return v196._finalize_lookup_stats(stats)


def _write_report(path: Path, payload: dict[str, object]) -> str:
    report = dict(payload)
    report["exact_digest"] = v196.digest_without_exact(report)
    _write_json(path, report)
    return str(report["exact_digest"])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
