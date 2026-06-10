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
from evolution_sim.mind.branch_intervention_residual import (
    MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_TRAINING_POLICY,
    REQUIRED_V143_REGRESSION_SEEDS,
    STRICT_BROAD_SEEDS,
    STRICT_CARRION_FIXTURE_SEEDS,
    STRICT_TICKS,
    _acceptance,
    build_branch_intervention_residual_artifact,
    validate_v143_branch_intervention_source,
)
from evolution_sim.mind.broad_regression_branch_intervention import (
    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
    MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.support_gated_residual import (
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY,
    load_support_gated_residual_artifact,
    score_support_gated_residual_artifact,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3BranchInterventionResidualTests(unittest.TestCase):
    def test_v143_source_integrity_accepts_seed_41_as_non_regression(self) -> None:
        rows = _dataset_rows()
        report = _v143_report(rows)

        integrity = validate_v143_branch_intervention_source(
            v143_report=report,
            dataset_rows=rows,
        )

        self.assertTrue(integrity["passed"])
        self.assertEqual(integrity["missing_regression_seed_support"], [])
        self.assertTrue(
            integrity["seed_41_non_regression"]["missing_support_is_expected"]
        )
        self.assertEqual(integrity["dominant_label_action_share"], 0.4)

    def test_artifact_roundtrips_and_scores_training_rows_equivalently(self) -> None:
        rows = _dataset_rows()
        artifact, training = build_branch_intervention_residual_artifact(
            v143_report=_v143_report(rows),
            dataset_rows=rows,
        )
        loaded = load_support_gated_residual_artifact(artifact)
        first = rows[0]["trainable"]
        features = first["features"]
        label = first["label"]["action"]

        scored = score_support_gated_residual_artifact(
            artifact=loaded,
            observation_input=features["observation_input"],
            action_mask=features["action_mask"],
            public_history_trace=[],
            linear_action=label,
        )

        self.assertEqual(
            artifact["schema_version"],
            MIND_V3_V144_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(artifact["policy"], MIND_V3_V144_SUPPORT_GATED_RESIDUAL_POLICY)
        self.assertEqual(
            training["policy"],
            MIND_V3_V144_BRANCH_INTERVENTION_RESIDUAL_TRAINING_POLICY,
        )
        self.assertTrue(
            training["artifact_roundtrip"][
                "loaded_artifact_scores_match_pre_serialization"
            ]
        )
        self.assertEqual(scored["selected_action"], label)

    def test_acceptance_blocks_resolved_invalid_increase_separately(self) -> None:
        broad = _comparison_report(
            fixture="broad",
            seeds=STRICT_BROAD_SEEDS,
            residual_resolved_invalid=1,
            linear_resolved_invalid=0,
        )
        carrion = _comparison_report(
            fixture="carrion_only",
            seeds=STRICT_CARRION_FIXTURE_SEEDS,
            residual_resolved_invalid=0,
            linear_resolved_invalid=0,
        )
        carrion["linear_fixture_gate"] = {"blockers": [{"reason": "min_alive"}]}
        carrion["residual_fixture_gate"] = {"blockers": []}

        acceptance = _acceptance(
            training_report={
                "source_integrity": {"passed": True, "failures": []},
                "artifact_roundtrip": {
                    "loaded_artifact_scores_match_pre_serialization": True,
                    "mismatch_count": 0,
                },
            },
            shadow={"shadow_gate": {"passed": True, "blockers": []}},
            broad=broad,
            carrion=carrion,
            broad_seeds=STRICT_BROAD_SEEDS,
            ticks=STRICT_TICKS,
            fixture_seeds=STRICT_CARRION_FIXTURE_SEEDS,
            fixture_ticks=STRICT_TICKS,
        )

        self.assertFalse(acceptance["passed"])
        self.assertEqual(
            acceptance["first_failed_floor"],
            "broad_resolved_invalid_not_increased",
        )
        self.assertIn(
            "zero_unsupported_requested_action_count",
            {floor["name"] for floor in acceptance["floors"] if floor["passed"]},
        )

    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        script = package["scripts"][
            "sim:mind:v3:branch-intervention-residual-live-ab"
        ]

        self.assertIn(
            "evolution_sim.cli.mind_v3_branch_intervention_residual_live_ab",
            script,
        )


def _v143_report(rows: list[dict[str, object]]) -> dict[str, object]:
    support_by_seed = {str(seed): True for seed in REQUIRED_V143_REGRESSION_SEEDS}
    return {
        "schema_version": MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
        "classification": {
            "primary": "broad_regression_branch_intervention_supported_for_v144_training"
        },
        "source_precheck": {"passed": True},
        "source_integrity": {"passed": True},
        "matrix": {
            "broad_seeds": list(STRICT_BROAD_SEEDS),
            "regression_seeds": list(REQUIRED_V143_REGRESSION_SEEDS),
            "ticks": STRICT_TICKS,
        },
        "dataset": {
            "row_count": len(rows),
            "dataset_digest": stable_payload_digest(rows),
        },
        "acceptance": {
            "passed": True,
            "dominant_label_action_share": 0.4,
            "regression_seeds": list(REQUIRED_V143_REGRESSION_SEEDS),
            "support_by_seed": support_by_seed,
        },
    }


def _dataset_rows() -> list[dict[str, object]]:
    labels = ["stay", "drink", "drink", "eat", "move_north"]
    rows = []
    for index, (seed, label) in enumerate(
        zip(REQUIRED_V143_REGRESSION_SEEDS, labels)
    ):
        rows.append(
            {
                "schema_version": (
                    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION
                ),
                "trainable": {
                    "feature_policy": (
                        "public_observation_input_and_public_action_mask_v1"
                    ),
                    "features": {
                        "observation_input": _observation_input(
                            energy=0.4 + 0.01 * index
                        ),
                        "action_mask": _action_mask(),
                    },
                    "label": {
                        "action": label,
                        "label_policy": (
                            "best_supported_terminal_alive_or_birth_improvement_v1"
                        ),
                    },
                },
                "metadata": {"seed": seed, "fixture": "broad"},
            }
        )
    return rows


def _comparison_report(
    *,
    fixture: str,
    seeds: tuple[int, ...],
    residual_resolved_invalid: int,
    linear_resolved_invalid: int,
) -> dict[str, object]:
    linear_runs = [
        _run(seed=seed, resolved_invalid=linear_resolved_invalid if index == 0 else 0)
        for index, seed in enumerate(seeds)
    ]
    residual_runs = [
        _run(
            seed=seed,
            resolved_invalid=residual_resolved_invalid if index == 0 else 0,
            applied_override_count=1 if index == 0 else 0,
        )
        for index, seed in enumerate(seeds)
    ]
    return {
        "fixture": fixture,
        "linear": {"runs": linear_runs, "aggregate": _aggregate(linear_runs)},
        "residual": {"runs": residual_runs, "aggregate": _aggregate(residual_runs)},
        "aggregate_delta": {"alive_agents_mean": 0.0},
        "per_seed_delta": [
            {
                "seed": seed,
                "alive_delta": 0,
                "births_delta": 0,
                "resolved_invalid_action_count_delta": (
                    residual_runs[index]["resolved_invalid_action_count"]
                    - linear_runs[index]["resolved_invalid_action_count"]
                ),
            }
            for index, seed in enumerate(seeds)
        ],
    }


def _run(
    *,
    seed: int,
    resolved_invalid: int,
    applied_override_count: int = 0,
) -> dict[str, object]:
    return {
        "seed": seed,
        "alive_agents": 10,
        "births": 1,
        "deaths": 0,
        "trajectory_record_count": 10,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": resolved_invalid,
        "resolved_invalid_action_count": resolved_invalid,
        "requested_action_counts": {"eat": 5, "drink": 5},
        "resolved_action_counts": {"eat": 5, "drink": 5},
        "dominant_requested_action_share": 0.5,
        "support_residual_diagnostics": {
            "unsupported_proposed_action_count": 0,
            "applied_override_count": applied_override_count,
            "applied_override_share": 0.1 if applied_override_count else 0.0,
            "dominant_applied_override_action_share": (
                0.5 if applied_override_count else 0.0
            ),
            "applied_override_action_counts": (
                {"eat": applied_override_count} if applied_override_count else {}
            ),
        },
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    resolved_invalid = sum(run["resolved_invalid_action_count"] for run in runs)
    applied = sum(
        run["support_residual_diagnostics"]["applied_override_count"]
        for run in runs
    )
    return {
        "alive_agents_mean": 10.0,
        "births_mean": 1.0,
        "deaths_mean": 0.0,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "unsupported_resolved_action_count": resolved_invalid,
        "resolved_invalid_action_count": resolved_invalid,
        "requested_action_counts": {"eat": 5 * len(runs), "drink": 5 * len(runs)},
        "resolved_action_counts": {"eat": 5 * len(runs), "drink": 5 * len(runs)},
        "dominant_requested_action_share": 0.5,
        "support_residual_diagnostics": {
            "unsupported_proposed_action_count": 0,
            "applied_override_count": applied,
            "applied_override_share": 0.1 if applied else 0.0,
            "dominant_applied_override_action_share": 0.5 if applied else 0.0,
            "applied_override_action_counts": {"eat": applied} if applied else {},
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
    return {
        action: action in {"stay", "drink", "eat", "move_north"}
        for action in ACTION_NAMES
    }


if __name__ == "__main__":
    unittest.main()
