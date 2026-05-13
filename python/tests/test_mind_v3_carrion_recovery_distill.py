from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_carrion_recovery_distill
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    NAVIGATION_TARGETS,
    OBSERVATION_SCHEMA_VERSION,
    encode_observation_input,
)
from evolution_sim.mind.carrion_recovery_archive import (
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
    build_carrion_recovery_distillation_report,
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
