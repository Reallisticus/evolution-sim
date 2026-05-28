from __future__ import annotations

import base64
import json
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    _pack_quantized_values,
)
from evolution_sim.mind.broad_branch_residual_constrained_audit import (
    MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_branch_residual_distillation_example import (
    BroadBranchResidualDistillationExampleError,
    build_broad_branch_residual_distillation_example_report,
    load_broad_residual_distillation_example_artifact,
    score_broad_residual_distillation_example_artifact,
    validate_broad_residual_distillation_example_artifact,
    write_broad_branch_residual_distillation_example_report,
)
from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
)


class MindV3BroadBranchResidualDistillationExampleTests(unittest.TestCase):
    def test_build_report_writes_training_ready_reloadable_artifact(self) -> None:
        report, artifact = build_broad_branch_residual_distillation_example_report(
            v99_broad_branch_residual_oracle_report=_v99_report(),
            v100_broad_branch_residual_constrained_report=_v100_report(),
        )

        self.assertTrue(
            report["v101_broad_residual_distillation_example_accepted"]
        )
        self.assertTrue(report["v102_expanded_training_allowed"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertEqual(report["coverage"]["training_row_count"], 10)
        self.assertEqual(report["coverage"]["source_seed_count"], 10)
        self.assertEqual(report["coverage"]["strict_seed_leak_count"], 0)
        self.assertEqual(report["coverage"]["teacher_action_counts"]["eat"], 5)
        self.assertLessEqual(
            report["coverage"]["dominant_teacher_action_share"],
            0.5,
        )
        self.assertEqual(report["training_evaluation"]["training_accuracy"], 1.0)
        self.assertTrue(
            report["reload_evaluation"]["artifact_reload_identical_choices"]
        )
        validate_broad_residual_distillation_example_artifact(artifact)

        reloaded = json.loads(json.dumps(artifact, sort_keys=True))
        choices = []
        for row in report["training_rows"]:
            state = row["policy_state"]
            scored = score_broad_residual_distillation_example_artifact(
                artifact=reloaded,
                observation_input=state["observation_input"],
                action_mask=state["action_mask"],
                public_history_trace=state["public_history_trace"],
            )
            choices.append(scored["selected_action"])
        self.assertEqual(
            choices,
            [row["teacher_action"] for row in report["training_rows"]],
        )

    def test_write_and_load_artifact(self) -> None:
        report, artifact = build_broad_branch_residual_distillation_example_report(
            v99_broad_branch_residual_oracle_report=_v99_report(),
            v100_broad_branch_residual_constrained_report=_v100_report(),
        )
        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            artifact_path = Path(tmpdir) / "artifact.json"
            write_broad_branch_residual_distillation_example_report(
                report,
                output_path=report_path,
                artifact=artifact,
                artifact_output_path=artifact_path,
            )
            loaded = load_broad_residual_distillation_example_artifact(
                artifact_path
            )
            loaded_from_report = load_broad_residual_distillation_example_artifact(
                report_path
            )

        self.assertEqual(loaded["schema_version"], artifact["schema_version"])
        self.assertEqual(
            loaded_from_report["schema_version"],
            artifact["schema_version"],
        )
        self.assertEqual(
            loaded["teacher_action_counts"],
            {"eat": 5, "move_east": 2, "stay": 3},
        )

    def test_strict_seed_leakage_is_rejected(self) -> None:
        v99 = _v99_report(seeds=(2, 3, 5, 11, 17, 23, 31, 47, 53, 59))
        v100 = _v100_report(seeds=(2, 3, 5, 11, 17, 23, 31, 47, 53, 59))

        report, _artifact = build_broad_branch_residual_distillation_example_report(
            v99_broad_branch_residual_oracle_report=v99,
            v100_broad_branch_residual_constrained_report=v100,
        )

        self.assertFalse(
            report["v101_broad_residual_distillation_example_accepted"]
        )
        self.assertIn(
            "strict_seed_leakage",
            {blocker["reason"] for blocker in report["acceptance"]["blockers"]},
        )

    def test_missing_policy_visible_observation_requires_regenerating_v99(self) -> None:
        v99 = _v99_report()
        del v99["branch_points"][0]["observation_input"]

        with self.assertRaisesRegex(
            BroadBranchResidualDistillationExampleError,
            "missing observation_input",
        ):
            build_broad_branch_residual_distillation_example_report(
                v99_broad_branch_residual_oracle_report=v99,
                v100_broad_branch_residual_constrained_report=_v100_report(),
            )

    def test_artifact_forbidden_runtime_key_guard(self) -> None:
        _report, artifact = build_broad_branch_residual_distillation_example_report(
            v99_broad_branch_residual_oracle_report=_v99_report(),
            v100_broad_branch_residual_constrained_report=_v100_report(),
        )
        artifact = json.loads(json.dumps(artifact))
        artifact["support_examples"][0]["source_seed"] = 2

        with self.assertRaisesRegex(
            BroadBranchResidualDistillationExampleError,
            "forbidden runtime",
        ):
            validate_broad_residual_distillation_example_artifact(artifact)


def _v99_report(
    *,
    seeds: tuple[int, ...] = (2, 3, 7, 11, 17, 23, 31, 47, 53, 59),
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
        "aggregate": {
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "unsupported_candidate_action_count": 0,
        },
        "branch_points": [
            {
                "branch_id": f"branch-{index}",
                "seed": seed,
                "branch_tick": 20 + index,
                "agent_id": 100 + index,
                "logged_action": "move_west",
                "action_mask": _action_mask(),
                "observation_input": _observation(index),
                "public_history_trace": [],
            }
            for index, seed in enumerate(seeds)
        ],
    }


def _v100_report(
    *,
    seeds: tuple[int, ...] = (2, 3, 7, 11, 17, 23, 31, 47, 53, 59),
) -> dict[str, object]:
    actions = (
        "eat",
        "eat",
        "eat",
        "eat",
        "eat",
        "move_east",
        "move_east",
        "stay",
        "stay",
        "stay",
    )
    assignment = [
        {
            "branch_id": f"branch-{index}",
            "seed": seed,
            "branch_tick": 20 + index,
            "agent_id": 100 + index,
            "logged_action": "move_west",
            "predicted_action": action,
            "safe_non_logged_override": True,
            "target_local_score_delta": 10.0 + index,
            "terminal_alive_delta": 0.0,
            "birth_delta": 0.0,
            "target_alive_delta": 0.0,
        }
        for index, (seed, action) in enumerate(zip(seeds, actions))
    ]
    return {
        "schema_version": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
        "v100_broad_branch_residual_constrained_diagnostic_accepted": True,
        "acceptance": {
            "accepted_rules": ["greedy_diversity_constrained_broad_residual_v1"]
        },
        "rule_reports": [
            {
                "rule": "greedy_diversity_constrained_broad_residual_v1",
                "assignment": assignment,
            }
        ],
    }


def _action_mask() -> dict[str, bool]:
    return {
        action: action in {"eat", "move_east", "stay"} for action in ACTION_NAMES
    }


def _observation(index: int) -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[0] = round(0.1 + float(index) / 20.0, 6)
    values[1] = round(0.9 - float(index) / 30.0, 6)
    data = base64.b64encode(
        zlib.compress(_pack_quantized_values(values), level=6)
    ).decode("ascii")
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "encoder_version": OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": data,
    }


if __name__ == "__main__":
    unittest.main()
