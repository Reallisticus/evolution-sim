from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_branch_planner_distillation_audit
from evolution_sim.mind.branch_constrained_planning_audit import (
    MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.branch_planner_distillation_audit import (
    MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION,
    MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
    BranchPlannerDistillationAuditError,
    build_branch_planner_distillation_audit_report,
    score_distilled_planner_artifact,
)
from evolution_sim.mind.branch_sequence_continuation_scorer import (
    build_branch_sequence_continuation_scorer_report,
)
from evolution_sim.mind.branch_utility_risk_audit import _utility_rows
from evolution_sim.mind.branch_mode_objective_audit import _list_of_mappings
from python.tests.test_mind_v3_branch_sequence_continuation_scorer import (
    _labels_with_trace_targets,
)


class MindV3BranchPlannerDistillationAuditTests(unittest.TestCase):
    def test_branch_planner_distillation_audit_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:branch-planner-distillation-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_branch_planner_distillation_audit"
            ),
        )

    def test_distillation_report_serializes_runtime_feasible_artifact(self) -> None:
        support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        sequence = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
        )

        report = build_branch_planner_distillation_audit_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION,
        )
        artifact = report["distilled_artifact"]
        self.assertEqual(
            artifact["schema_version"],
            MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertFalse(artifact["promotion_ready"])
        self.assertFalse(
            artifact["inference_contract"]["requires_planner_outcome_tables"]
        )
        self.assertFalse(
            artifact["inference_contract"]["requires_global_batch_assignment"]
        )
        self.assertTrue(report["runtime_reload_check"]["actions_match"])
        self.assertIn(
            "planner_distillation_runtime_feasibility_support_probe",
            report,
        )

    def test_strict_seed_training_leakage_is_rejected(self) -> None:
        support = _labels_with_trace_targets(seed_start=13, branch_prefix="support")
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        sequence = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=_labels_with_trace_targets(
                seed_start=101,
                branch_prefix="safe-support",
            ),
            strict_branch_action_oracle_labels=strict,
        )

        with self.assertRaises(BranchPlannerDistillationAuditError):
            build_branch_planner_distillation_audit_report(
                support_branch_action_oracle_labels=support,
                strict_branch_action_oracle_labels=strict,
                branch_sequence_continuation_scorer_report=sequence,
            )

    def test_runtime_inference_does_not_need_seed_branch_or_planner_outcomes(self) -> None:
        support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        sequence = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
        )
        report = build_branch_planner_distillation_audit_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )
        artifact = report["distilled_artifact"]
        rows = _utility_rows(_list_of_mappings(strict["labels"], "labels"))
        row = copy.deepcopy(rows[0])
        row.pop("branch_id", None)
        row.pop("seed", None)
        row.pop("source", None)

        scored = score_distilled_planner_artifact(row=row, artifact=artifact)

        self.assertIn(scored["selected_action"], row["action_mask"])
        for section in ("sequence_support_examples", "teacher_imitation_examples"):
            for example in artifact[section]:
                self.assertNotIn("seed", example)
                self.assertNotIn("branch_id", example)
                self.assertNotIn("fixture", example)
                self.assertNotIn("logged_action", example)
                self.assertNotIn("planner_candidate_outcome_table", example)

    def test_distillation_report_json_is_deterministic(self) -> None:
        support = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
        strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
        sequence = build_branch_sequence_continuation_scorer_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
        )

        first = build_branch_planner_distillation_audit_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )
        second = build_branch_planner_distillation_audit_report(
            support_branch_action_oracle_labels=support,
            strict_branch_action_oracle_labels=strict,
            branch_sequence_continuation_scorer_report=sequence,
        )

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_branch_planner_distillation_audit_cli_writes_json_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            support = _labels_with_trace_targets(
                seed_start=101,
                branch_prefix="support",
            )
            strict = _labels_with_trace_targets(seed_start=13, branch_prefix="strict")
            sequence = build_branch_sequence_continuation_scorer_report(
                support_branch_action_oracle_labels=support,
                strict_branch_action_oracle_labels=strict,
            )
            support_path = tmp / "support.json"
            strict_path = tmp / "strict.json"
            sequence_path = tmp / "sequence.json"
            constrained_path = tmp / "constrained.json"
            output_path = tmp / "distill.json"
            support_path.write_text(json.dumps(support), encoding="utf-8")
            strict_path.write_text(json.dumps(strict), encoding="utf-8")
            sequence_path.write_text(json.dumps(sequence), encoding="utf-8")
            constrained_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION
                        ),
                        "acceptance": {"best_rule_for_diagnostics": {}},
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_branch_planner_distillation_audit",
                    "--support-branch-action-oracle-labels",
                    str(support_path),
                    "--strict-branch-action-oracle-labels",
                    str(strict_path),
                    "--branch-sequence-continuation-scorer",
                    str(sequence_path),
                    "--branch-constrained-planning-audit",
                    str(constrained_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_branch_planner_distillation_audit.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["coverage"]["strict_eval_label_count"], 3)


if __name__ == "__main__":
    unittest.main()
