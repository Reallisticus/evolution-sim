from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    NAVIGATION_TARGETS,
    OBSERVATION_SCHEMA_VERSION,
    decode_observation_input,
    encode_observation_input,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED,
    MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_recovery_distill import (
    MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_recovery_residual_audit import (
    MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.policy_inputs import (
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    ecological_policy_input_contract,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recovery_action_target_alignment import (
    MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_SCHEMA_VERSION,
    _oracle_comparison,
    _window_memberships,
    _window_summaries,
    build_recovery_action_target_alignment_report,
)
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
    MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_NEURAL_BACKEND,
    MIND_V3_NEURAL_INPUT_POLICY,
    MIND_V3_NEURAL_MODEL_TYPE,
    MIND_V3_NEURAL_TRAINER,
)
from evolution_sim.mind.v3_policy import (
    MIND_V3_POLICY_ID,
    MIND_V3_POLICY_VERSION,
)


class MindV3RecoveryActionTargetAlignmentTests(unittest.TestCase):
    def test_recovery_action_target_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:recovery-action-target-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_recovery_action_target_audit"
            ),
        )

    def test_build_report_scores_all_heldout_records_and_is_diagnostics_only(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "heldout.jsonl.gz"
            records = [
                _record(
                    tick=0,
                    requested_action="eat",
                    observation_input=_observation_input(
                        hydration=0.8,
                        water_dx=1,
                        water_dy=0,
                        carcass_energy=1.0,
                    ),
                    action_mask=_action_mask("eat", "stay"),
                    food_source="carcass",
                    resource_gain=0.2,
                ),
                _record(
                    tick=1,
                    requested_action="move_east",
                    observation_input=_observation_input(
                        hydration=0.52,
                        water_dx=1,
                        water_dy=0,
                    ),
                    action_mask=_action_mask("move_east", "stay", "eat"),
                ),
                _record(
                    tick=2,
                    requested_action="drink",
                    observation_input=_observation_input(
                        hydration=0.45,
                        water_dx=0,
                        water_dy=0,
                    ),
                    action_mask=_action_mask("drink", "stay", "eat"),
                ),
            ]
            _write_trajectory(trajectory_path, records)
            artifact = _minimal_artifact(neural_top_action="drink")
            before_digest = stable_payload_digest(artifact)

            report = build_recovery_action_target_alignment_report(
                artifact=artifact,
                distill_report=_distill_report(),
                evaluation_report={"evaluation": "present"},
                split_report=_split_report(trajectory_path),
                residual_audit=_residual_audit(),
            )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_RECOVERY_ACTION_TARGET_ALIGNMENT_SCHEMA_VERSION,
        )
        self.assertTrue(report["contract"]["diagnostics_only"])
        self.assertEqual(report["contract"]["runtime_policy_effect"], "none")
        self.assertEqual(report["contract"]["trained_artifact_effect"], "none")
        self.assertTrue(report["non_promoted"])
        self.assertEqual(stable_payload_digest(artifact), before_digest)
        self.assertEqual(report["heldout_scoring"]["decision_row_count"], 3)
        all_records = report["windows"]["all_records"]["by_outcome_class"]["all"]
        first_record = report["windows"]["first_record"]["by_outcome_class"]["all"]
        self.assertEqual(all_records["row_count"], 3)
        self.assertEqual(first_record["row_count"], 1)
        self.assertEqual(report["heldout_scoring"]["split_rows"][0]["scored_decision_count"], 3)

    def test_windows_classify_recovery_rows_deterministically(self) -> None:
        values = decode_observation_input(
            _observation_input(hydration=0.42, water_dx=1, water_dy=0)
        )

        move_windows = _window_memberships(
            trajectory_record_position=4,
            context_snapshot={"post_carrion_contact": True},
            recovery_phase_remaining=3,
            action_mask=_action_mask("drink", "eat", "move_east", "stay"),
            logged_action="move_east",
            observation_values=values,
        )
        stay_windows = _window_memberships(
            trajectory_record_position=5,
            context_snapshot={"post_carrion_contact": True},
            recovery_phase_remaining=0,
            action_mask=_action_mask("eat", "stay"),
            logged_action="stay",
            observation_values=decode_observation_input(
                _observation_input(hydration=0.5, water_dx=0, water_dy=0)
            ),
        )

        self.assertIn("post_carrion", move_windows)
        self.assertIn("recovery_phase", move_windows)
        self.assertIn("drink_available", move_windows)
        self.assertIn("low_hydration_after_carrion", move_windows)
        self.assertIn("water_directed_move", move_windows)
        self.assertIn("eat_available_non_eat_logged", move_windows)
        self.assertIn("conserve_or_stay_candidate", stay_windows)

    def test_survivor_and_failure_rank_distributions_are_emitted(self) -> None:
        summary = _window_summaries(
            [
                _scored_row(
                    terminal_survivor=True,
                    logged_action_rank=1,
                    logged_action="drink",
                    neural_top_action="drink",
                ),
                _scored_row(
                    terminal_survivor=False,
                    logged_action_rank=3,
                    logged_action="stay",
                    neural_top_action="eat",
                ),
            ]
        )

        all_records = summary["all_records"]["by_outcome_class"]["all"]
        self.assertEqual(all_records["logged_survivor_action_rank_counts"], {"1": 1})
        self.assertEqual(all_records["logged_failure_action_rank_counts"], {"3": 1})
        self.assertEqual(all_records["survivor_minus_failure_rank_delta"], -2.0)

    def test_extra_eat_pressure_counts_shadow_changes_that_end_in_eat(self) -> None:
        summary = _window_summaries(
            [
                _scored_row(
                    logged_action="stay",
                    linear_top_action="stay",
                    configured_blended_action="eat",
                    margin_ignored_action="eat",
                    configured_would_change=True,
                    margin_ignored_would_change=True,
                    eat_available_non_eat_logged=True,
                )
            ]
        )

        all_records = summary["all_records"]["by_outcome_class"]["all"]
        self.assertEqual(
            all_records["extra_eat_pressure_counts"],
            {"configured": 1, "margin_ignored": 1},
        )
        self.assertEqual(
            all_records["would_change_counts_by_final_action"]["configured"],
            {"eat": 1},
        )
        self.assertEqual(
            all_records["would_change_counts_by_seed"]["margin_ignored"],
            {"13": 1},
        )

    def test_oracle_join_reports_matched_and_unmatched_without_coverage_requirement(
        self,
    ) -> None:
        rows = [
            _scored_row(
                seed=13,
                branch_id="branch-a",
                branch_state_digest="digest-a",
                branch_tick=2,
                tick=2,
                agent_id=7,
                logged_action="drink",
                neural_top_action="drink",
                configured_blended_action="drink",
                margin_ignored_action="drink",
            ),
            _scored_row(
                seed=19,
                branch_id="branch-b",
                branch_state_digest="digest-b",
                branch_tick=4,
                tick=4,
                agent_id=8,
                logged_action="stay",
            ),
        ]

        comparison = _oracle_comparison(
            rows,
            oracle_audit={
                "schema_version": "mind_v3_branch_action_oracle_audit_v1",
                "branch_results": [
                    {
                        "branch_id": "branch-a",
                        "branch_state_digest": "digest-a",
                        "seed": 13,
                        "branch_tick": 2,
                        "agent_id": 7,
                        "logged_action": "drink",
                        "oracle_best_action": "drink",
                    }
                ],
            },
            oracle_audit_path=Path("oracle.json"),
            oracle_load_error=None,
        )

        self.assertTrue(comparison["loaded"])
        self.assertEqual(comparison["matched_row_count"], 1)
        self.assertEqual(comparison["unmatched_row_count"], 1)
        self.assertEqual(
            comparison["action_match_counts"]["margin_ignored"]["match"],
            1,
        )


def _minimal_artifact(*, neural_top_action: str) -> dict[str, object]:
    hidden_units = 1
    zero_action_vector = {action: 0.0 for action in ACTION_NAMES}
    action_bias = {action: -1.0 for action in ACTION_NAMES}
    action_bias[neural_top_action] = 2.0
    action_matrix = {action: [0.0] for action in ACTION_NAMES}
    head = {
        "1": {
            "positive_count": 0,
            "negative_count": 0,
            "weights": [0.0],
            "bias": 0.0,
        }
    }
    return {
        "schema_version": MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
        "artifact_mode": MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
        "model_type": MIND_V3_NEURAL_MODEL_TYPE,
        "trainer": MIND_V3_NEURAL_TRAINER,
        "backend": MIND_V3_NEURAL_BACKEND,
        "architecture": "fixed_projection_mlp_action_horizon_heads_v1",
        "input_policy": MIND_V3_NEURAL_INPUT_POLICY,
        "input_contract": ecological_policy_input_contract(),
        "hidden_units": hidden_units,
        "trained_record_count": 1,
        "horizon_ticks": [1],
        "neural_residual_scale": 0.03,
        "neural_residual_max_linear_override_margin": 0.008,
        "neural_residual_context_gate": "visible_carrion_or_recovery_phase_v1",
        "neural_residual_recovery_phase_ticks": 2,
        "hidden_weights": [[0.0] * ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE],
        "hidden_bias": [0.0],
        "action_output_weights": action_matrix,
        "action_output_bias": action_bias,
        "action_value_weights": action_matrix,
        "action_value_bias": zero_action_vector,
        "fixture_action_bias_delta": zero_action_vector,
        "survival_heads": head,
        "reproduction_heads": head,
        "provenance": {},
    }


def _split_report(trajectory_path: Path) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
        "split_policy": (
            MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED
        ),
        "records": {
            "train": [],
            "heldout": [
                {
                    "record_id": "heldout-1",
                    "dataset_record_index": 0,
                    "seed": 13,
                    "branch_id": "branch-1",
                    "branch_state_digest": "digest-1",
                    "branch_state_key": "digest-1",
                    "branch_state_key_type": "branch_state_digest",
                    "branch_tick": 0,
                    "continuation_script": "carrion_then_water",
                    "trajectory_path": str(trajectory_path),
                    "terminal_survivor": True,
                    "outcome_class": "survivor",
                    "trajectory_record_count": 3,
                }
            ],
        },
        "acceptance": {"validation_passed": True, "training_blocked": False},
    }


def _distill_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
        "acceptance": {"promotion_candidate_passed": False},
    }


def _residual_audit() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_RECOVERY_RESIDUAL_AUDIT_SCHEMA_VERSION,
        "failure_classification": {"primary": "margin domination"},
        "calibration_classification": {"primary": "margin domination"},
    }


def _write_trajectory(path: Path, records: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"format": "test"}) + "\n")
        for record in records:
            handle.write(json.dumps({"record": record}, sort_keys=True) + "\n")
        handle.write(json.dumps({"footer": "test"}) + "\n")


def _record(
    *,
    tick: int,
    requested_action: str,
    observation_input: dict[str, object],
    action_mask: dict[str, bool],
    food_source: str | None = None,
    resource_gain: float = 0.0,
) -> dict[str, object]:
    ate = food_source is not None and resource_gain > 0.0
    drank = requested_action == "drink"
    return {
        "tick": tick,
        "agent_id": 7,
        "policy_id": MIND_V3_POLICY_ID,
        "policy_version": MIND_V3_POLICY_VERSION,
        "action_source": MIND_V3_POLICY_VERSION,
        "requested_action": requested_action,
        "resolved_action": requested_action,
        "action_valid": True,
        "resolution_action_valid": True,
        "action_mask": action_mask,
        "observation_metadata": {"agent_id": 7},
        "observation_input": observation_input,
        "before": {
            "alive": True,
            "energy_ratio": 0.6,
            "hydration_ratio": 0.6,
            "health_ratio": 1.0,
            "x": 2,
            "y": 2,
        },
        "after": {
            "alive": True,
            "energy_ratio": 0.6 + resource_gain,
            "hydration_ratio": 0.75 if drank else 0.6,
            "health_ratio": 1.0,
            "x": 3 if requested_action == "move_east" else 2,
            "y": 2,
        },
        "outcome": {
            "feeding": {
                "ate": ate,
                "food_source": food_source,
                "gained_energy": resource_gain,
            },
            "drinking": {"drank": drank},
            "movement": {"moved": requested_action.startswith("move_")},
            "resource_gain": resource_gain,
            "died": False,
            "reproduced": False,
        },
        "reward": {"total": 0.0},
    }


def _action_mask(*legal: str) -> dict[str, bool]:
    allowed = set(legal)
    return {action: action in allowed for action in ACTION_NAMES}


def _observation_input(
    *,
    hydration: float,
    water_dx: int,
    water_dy: int,
    carcass_energy: float = 0.0,
) -> dict[str, object]:
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "metadata": {"agent_id": 7},
        "self": {
            "energy_ratio": 0.7,
            "hydration_ratio": hydration,
            "health_ratio": 1.0,
            "injury_load": 0.0,
            "age_norm": 0.2,
            "reproduction_ready": False,
            "matched_diet_ratio": 1.0,
            "trophic_role": "carnivore",
            "meat_mode": "scavenger",
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
            _patch_cell(
                dx,
                dy,
                carcass_energy=carcass_energy if dx == 0 and dy == 0 else 0.0,
            )
            for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
            for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
        ],
        "navigation": {
            target: {
                "dx": water_dx if target == "water" else 0,
                "dy": water_dy if target == "water" else 0,
                "distance": abs(water_dx) + abs(water_dy) if target == "water" else 0,
                "strength": 1.0 if target == "water" and (water_dx or water_dy) else 0.0,
            }
            for target in NAVIGATION_TARGETS
        },
    }
    return encode_observation_input(observation)


def _patch_cell(
    dx: int,
    dy: int,
    *,
    carcass_energy: float,
) -> dict[str, object]:
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


def _scored_row(
    *,
    terminal_survivor: bool = True,
    logged_action_rank: int = 1,
    logged_action: str = "drink",
    neural_top_action: str = "drink",
    linear_top_action: str = "stay",
    configured_blended_action: str = "stay",
    margin_ignored_action: str = "stay",
    forced_gate_action: str = "stay",
    configured_would_change: bool = False,
    margin_ignored_would_change: bool = False,
    forced_gate_would_change: bool = False,
    eat_available_non_eat_logged: bool = False,
    seed: int = 13,
    branch_id: str = "branch",
    branch_state_digest: str = "digest",
    branch_tick: int = 0,
    tick: int = 0,
    agent_id: int = 7,
) -> dict[str, object]:
    outcome_class = "survivor" if terminal_survivor else "failure"
    return {
        "windows": ["all_records"],
        "terminal_survivor": terminal_survivor,
        "outcome_class": outcome_class,
        "logged_action_rank": logged_action_rank,
        "logged_action": logged_action,
        "neural_top_action": neural_top_action,
        "linear_top_action": linear_top_action,
        "configured_blended_action": configured_blended_action,
        "margin_ignored_action": margin_ignored_action,
        "forced_gate_action": forced_gate_action,
        "configured_would_change": configured_would_change,
        "margin_ignored_would_change": margin_ignored_would_change,
        "forced_gate_would_change": forced_gate_would_change,
        "eat_available_non_eat_logged": eat_available_non_eat_logged,
        "drink_available": logged_action == "drink",
        "water_directed_actions": ["move_east"],
        "seed": seed,
        "branch_id": branch_id,
        "branch_state_digest": branch_state_digest,
        "branch_tick": branch_tick,
        "tick": tick,
        "agent_id": agent_id,
    }


if __name__ == "__main__":
    unittest.main()
