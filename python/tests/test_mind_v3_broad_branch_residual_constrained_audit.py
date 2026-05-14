from __future__ import annotations

import unittest

from evolution_sim.mind.broad_branch_residual_constrained_audit import (
    _acceptance,
    _assignment_report,
    _greedy_constrained_assignment,
)


class MindV3BroadBranchResidualConstrainedAuditTests(unittest.TestCase):
    def test_greedy_assignment_repairs_dominant_action_cap(self) -> None:
        rows = [_row(index, baseline_action="eat") for index in range(8)]
        rows.extend(_row(index, baseline_action="stay") for index in range(8, 10))

        assignment = _greedy_constrained_assignment(rows, max_action_count=5)
        report = _assignment_report(
            "greedy_diversity_constrained_broad_residual_v1",
            assignment,
            acceptance_candidate=True,
        )

        self.assertEqual(report["dominant_predicted_action"], "stay")
        self.assertEqual(report["dominant_predicted_action_share"], 0.5)
        self.assertGreater(report["target_local_score_delta_summary"]["mean"], 0.0)

    def test_acceptance_allows_v101_only_when_constrained_rule_passes(self) -> None:
        rows = [_row(index, baseline_action="eat") for index in range(8)]
        rows.extend(_row(index, baseline_action="stay") for index in range(8, 10))
        assignment = _greedy_constrained_assignment(rows, max_action_count=5)
        rule = _assignment_report(
            "greedy_diversity_constrained_broad_residual_v1",
            assignment,
            acceptance_candidate=True,
        )
        accepted = _acceptance(
            coverage={
                "source_replay_verified": True,
                "source_heuristic_action_source_count": 0,
                "source_unsupported_candidate_action_count": 0,
                "strict_seed_leak_count": 0,
                "source_seed_count": 8,
            },
            rule_reports=[rule],
            floors={
                "comparison_count": 10,
                "source_seed_count": 8,
                "dominant_predicted_action_share_max": 0.5,
                "safe_non_logged_override_count": 8,
                "mean_target_local_score_delta_gt": 0.0,
                "mean_terminal_alive_delta_min": 0.0,
                "mean_birth_delta_min": 0.0,
            },
        )
        rejected = _acceptance(
            coverage={
                "source_replay_verified": True,
                "source_heuristic_action_source_count": 0,
                "source_unsupported_candidate_action_count": 1,
                "strict_seed_leak_count": 0,
                "source_seed_count": 8,
            },
            rule_reports=[rule],
            floors={
                "comparison_count": 10,
                "source_seed_count": 8,
                "dominant_predicted_action_share_max": 0.5,
                "safe_non_logged_override_count": 8,
                "mean_target_local_score_delta_gt": 0.0,
                "mean_terminal_alive_delta_min": 0.0,
                "mean_birth_delta_min": 0.0,
            },
        )

        self.assertTrue(accepted["v101_residual_distillation_allowed"])
        self.assertFalse(rejected["v101_residual_distillation_allowed"])
        self.assertIn(
            "source_unsupported_candidate_action_count_nonzero",
            {blocker["reason"] for blocker in rejected["blockers"]},
        )


def _row(index: int, *, baseline_action: str) -> dict[str, object]:
    return {
        "branch_id": f"branch-{index}",
        "seed": 100 + index,
        "branch_tick": index,
        "agent_id": 200 + index,
        "logged_action": "move_east",
        "baseline_action": baseline_action,
        "candidates": [
            _candidate("eat", 10.0 - index * 0.01),
            _candidate("stay", 9.0 - index * 0.01),
            _candidate("move_east", 0.0),
        ],
    }


def _candidate(action: str, target_delta: float) -> dict[str, object]:
    return {
        "action": action,
        "target_local_score_delta": target_delta,
        "terminal_alive_delta": 0.0,
        "birth_delta": 0.0,
        "target_alive_delta": 0.0,
        "forced_action_supported": True,
        "forced_action_used": True,
    }


if __name__ == "__main__":
    unittest.main()
