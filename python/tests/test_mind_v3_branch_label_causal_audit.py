from __future__ import annotations

import base64
import json
import struct
import unittest
import zlib
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_QUANTIZATION_SCALE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    SELF_INPUT_FIELDS,
)
from evolution_sim.mind.branch_label_causal_audit import (
    LABEL_RESOLUTION_INVALID_RISK,
    ROUTE_LABEL_BLACKLIST,
    build_branch_label_causal_audit_report,
)
from evolution_sim.mind.branch_intervention_residual import (
    MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_REPORT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_regression_branch_intervention import (
    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
    MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
    V103_ACTION_PRIOR_BALANCE_PENALTY,
)
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3BranchLabelCausalAuditTests(unittest.TestCase):
    def test_applied_override_maps_to_support_row_and_attributes_invalids(self) -> None:
        rows = [_dataset_row(seed=5, label="stay")]
        artifact = _artifact(rows)
        report = build_branch_label_causal_audit_report(
            v143_report=_v143_report(rows),
            dataset_rows=rows,
            v144_report=_v144_report(artifact),
            artifact=artifact,
            trace_runs={5: _trace_run(rows[0])},
        )

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(report["applied_label_count"], 1)
        label = report["applied_labels"][0]
        self.assertEqual(label["seed"], 5)
        self.assertEqual(label["tick"], 0)
        self.assertEqual(label["agent_id"], 6)
        self.assertEqual(label["linear_action"], "eat")
        self.assertEqual(label["residual_action"], "stay")
        self.assertTrue(label["support_example"]["matched"])
        self.assertEqual(label["label_source_row"]["row_index"], 0)
        attribution = label["downstream_invalid_resolution_attribution"]
        self.assertEqual(attribution["invalid_after_override_delta_vs_linear"], 2)
        self.assertEqual(attribution["category_counts"]["occupancy_conflicts"], 1)
        self.assertEqual(attribution["category_counts"]["depleted_resources"], 1)
        self.assertEqual(label["label_classification"], LABEL_RESOLUTION_INVALID_RISK)

    def test_route_recommends_blacklist_for_causally_bad_label(self) -> None:
        rows = [_dataset_row(seed=5, label="stay")]
        artifact = _artifact(rows)
        report = build_branch_label_causal_audit_report(
            v143_report=_v143_report(rows),
            dataset_rows=rows,
            v144_report=_v144_report(artifact),
            artifact=artifact,
            trace_runs={5: _trace_run(rows[0])},
        )

        route = report["route_decision"]
        self.assertEqual(route["route"], ROUTE_LABEL_BLACKLIST)
        self.assertEqual(len(route["label_blacklist"]), 1)
        self.assertEqual(route["label_blacklist"][0]["seed"], 5)
        self.assertEqual(route["label_blacklist"][0]["residual_action"], "stay")
        self.assertIn(
            "Seed 5 applied the v143 branch label",
            report["seed_5_birth_regression_explanation"]["explanation"],
        )

    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:v3:branch-label-causal-audit"]

        self.assertIn(
            "evolution_sim.cli.mind_v3_branch_label_causal_audit",
            script,
        )


def _dataset_row(*, seed: int, label: str) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {
                "observation_input": _observation_input(energy=0.42),
                "action_mask": _action_mask(),
            },
            "label": {
                "action": label,
                "label_policy": "best_supported_terminal_alive_or_birth_improvement_v1",
            },
        },
        "metadata": {
            "seed": seed,
            "fixture": "broad",
            "branch_id": f"v143-broad-seed-{seed}-branch-0-tick-0-agent-6",
            "branch_tick": 0,
            "record_index": 0,
            "agent_id": 6,
            "branch_state_digest": "digest-branch",
            "replay_digest": "digest-replay",
            "outcome_evidence": {
                "deltas_vs_baseline": {
                    "alive_agents": 7,
                    "births": 4,
                    "deaths": -3,
                    "unsupported_resolved_action_count": 0,
                },
                "deltas_vs_v142_override": {
                    "alive_agents": 9,
                    "births": 4,
                    "deaths": -5,
                    "target_alive": 1,
                    "unsupported_resolved_action_count": 3,
                },
            },
        },
    }


def _v143_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
        "classification": {
            "primary": "broad_regression_branch_intervention_supported_for_v144_training"
        },
        "dataset": {
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
        },
    }


def _v144_report(artifact: dict[str, object]) -> dict[str, object]:
    per_seed = [
        {
            "seed": 5,
            "alive_delta": 1,
            "births_delta": -1,
            "resolved_invalid_action_count_delta": 1,
            "unsupported_requested_action_count_delta": 0,
            "unsupported_proposed_action_count": 0,
            "applied_override_count": 1,
            "applied_override_action_counts": {"stay": 1},
        }
    ]
    return {
        "schema_version": MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_REPORT_SCHEMA_VERSION,
        "classification": {
            "primary": "branch_intervention_residual_blocked_non_promotable"
        },
        "artifact_output": "output/mind/mind-v3-v144-branch-intervention-residual-artifact.json",
        "artifact_digest": stable_payload_digest(artifact),
        "acceptance": {
            "passed": False,
            "runtime_promotion_allowed": False,
            "training_promotion_allowed": False,
        },
        "live_strict_broad": {
            "linear": {"aggregate": {"resolved_invalid_action_count": 32}},
            "residual": {"aggregate": {"resolved_invalid_action_count": 36}},
            "per_seed_delta": per_seed,
        },
    }


def _artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    row = rows[0]
    features = row["trainable"]["features"]
    action = row["trainable"]["label"]["action"]
    runtime_row = planner_distilled_runtime_row(
        observation_input=features["observation_input"],
        action_mask=features["action_mask"],
        public_history_trace=[],
    )
    vector = candidate_feature_vector(runtime_row, action)
    return {
        "schema_version": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        "policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
        "support_gate": {
            "nearest_support_distance_threshold": 0.0,
            "residual_score_margin_threshold": 0.0,
        },
        "inference_contract": {
            "one_row_one_agent_local_decision": True,
            "requires_action_mask": True,
            "requires_policy_visible_features_only": True,
            "requires_linear_default_action": True,
            "requires_planner_outcome_tables": False,
            "requires_global_batch_assignment": False,
            "uses_heuristic_fallback": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_fixture_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_private_simulator_state": False,
        },
        "training_row_count": 1,
        "support_action_counts": {action: 1},
        "teacher_action_counts": {action: 1},
        "support_examples": [
            {
                "example_index": 0,
                "action": action,
                "mode": "other",
                "feature_vector": [round(float(value), 6) for value in vector],
                "weight": 1.0,
            }
        ],
    }


def _trace_run(row: dict[str, object]) -> dict[str, object]:
    features = row["trainable"]["features"]
    override_record = _record(
        tick=0,
        agent_id=6,
        requested_action="stay",
        resolved_action="stay",
        action_mask=features["action_mask"],
        resolution_action_mask=features["action_mask"],
        observation_input=features["observation_input"],
        x=4,
        y=4,
        valid=True,
    )
    invalid_move = _record(
        tick=0,
        agent_id=7,
        requested_action="move_north",
        resolved_action="stay",
        action_mask={**features["action_mask"], "move_north": True},
        resolution_action_mask={**features["action_mask"], "move_north": False},
        observation_input=features["observation_input"],
        x=4,
        y=5,
        valid=False,
    )
    invalid_eat = _record(
        tick=3,
        agent_id=9,
        requested_action="eat",
        resolved_action="stay",
        action_mask={**features["action_mask"], "eat": True},
        resolution_action_mask={**features["action_mask"], "eat": False},
        observation_input=features["observation_input"],
        x=9,
        y=9,
        valid=False,
    )
    linear_record = _record(
        tick=0,
        agent_id=6,
        requested_action="eat",
        resolved_action="eat",
        action_mask=features["action_mask"],
        resolution_action_mask=features["action_mask"],
        observation_input=features["observation_input"],
        x=4,
        y=4,
        valid=True,
    )
    return {
        "linear": {"records": [linear_record], "diagnostics": [None]},
        "residual": {
            "records": [override_record, invalid_move, invalid_eat],
            "diagnostics": [
                {
                    "support_residual_policy": MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
                    "support_residual_linear_action": "eat",
                    "support_residual_proposed_action": "stay",
                    "support_residual_final_action": "stay",
                    "support_residual_override_proposed": True,
                    "support_residual_override_allowed": True,
                    "support_residual_override_applied": True,
                    "support_residual_support_gate_passed": True,
                    "support_residual_nearest_support_distance": 0.0,
                    "support_residual_score_margin": 0.3,
                },
                None,
                None,
            ],
        },
    }


def _record(
    *,
    tick: int,
    agent_id: int,
    requested_action: str,
    resolved_action: str,
    action_mask: dict[str, bool],
    resolution_action_mask: dict[str, bool],
    observation_input: dict[str, object],
    x: int,
    y: int,
    valid: bool,
) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": agent_id,
        "observation_input": observation_input,
        "action_mask": action_mask,
        "resolution_action_mask": resolution_action_mask,
        "requested_action": requested_action,
        "resolved_action": resolved_action,
        "action_valid": bool(action_mask.get(requested_action)),
        "resolution_action_valid": valid,
        "before": {
            "x": x,
            "y": y,
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 0.9,
            "alive": True,
        },
        "after": {
            "x": x,
            "y": y,
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 0.9,
            "alive": True,
        },
        "outcome": {
            "invalid_reason": None if valid else "not_in_resolution_action_mask"
        },
    }


def _observation_input(*, energy: float) -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = energy
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.6
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
    quantized = [
        int(round(max(-1.0, min(1.0, value)) * OBSERVATION_QUANTIZATION_SCALE))
        for value in values
    ]
    packed = struct.pack(f"<{len(quantized)}h", *quantized)
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "encoder_version": OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": base64.b64encode(zlib.compress(packed, level=6)).decode("ascii"),
    }


def _action_mask() -> dict[str, bool]:
    return {action: action in {"stay", "eat"} for action in ACTION_NAMES}


if __name__ == "__main__":
    unittest.main()
