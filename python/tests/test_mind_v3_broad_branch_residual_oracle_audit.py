from __future__ import annotations

import unittest

from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY,
    _BroadBranchCandidate,
    _ForcedFirstActionThenDelegatePolicy,
    _acceptance,
    _validated_non_strict_seeds,
    select_broad_branch_candidates,
)


class MindV3BroadBranchResidualOracleAuditTests(unittest.TestCase):
    def test_strict_seed_leakage_is_rejected_before_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclude strict seeds"):
            _validated_non_strict_seeds((2, 5, 11))

    def test_selection_balances_categories_before_priority_fill(self) -> None:
        candidates = [
            _candidate(index=0, category="energy", priority=100.0),
            _candidate(index=1, category="pre_death", priority=10.0),
            _candidate(index=2, category="movement", priority=9.0),
            _candidate(index=3, category="hydration", priority=8.0),
            _candidate(index=4, category="animal_resource", priority=7.0),
            _candidate(index=5, category="recovery", priority=6.0),
        ]

        selected = select_broad_branch_candidates(candidates, max_branch_points=4)

        categories = [item.categories[0] for item in selected]
        self.assertEqual(
            categories,
            ["pre_death", "recovery", "movement", "hydration"],
        )

    def test_forced_policy_updates_delegate_history_for_forced_action(self) -> None:
        delegate = _FakeDelegate()
        policy = _ForcedFirstActionThenDelegatePolicy(
            target_agent_id=7,
            forced_action="eat",
            delegate=delegate,
        )

        decision = policy.decide(
            {"metadata": {"agent_id": 7}},
            {"eat": True, "stay": True},
        )
        update = policy.observe_transition(
            {
                "agent_id": 7,
                "action_source": "branch_oracle_force:eat",
                "policy_id": policy.policy_id,
                "policy_version": policy.policy_version,
                "requested_action": "eat",
            }
        )

        self.assertEqual(decision.requested_action, "eat")
        self.assertTrue(update["observed"])
        self.assertEqual(delegate.observed[0]["policy_id"], delegate.policy_id)

    def test_acceptance_reports_support_floor_pass_and_blockers(self) -> None:
        floors = {
            "branch_point_count": 40,
            "source_seed_count": 8,
            "safe_non_logged_override_count": 8,
            "safe_non_logged_override_share": 0.2,
            "dominant_oracle_action_share_max": 0.5,
        }
        accepted = _acceptance(
            aggregate={
                "strict_seed_leak_count": 0,
                "replay_verified": True,
                "heuristic_action_source_count": 0,
                "unsupported_candidate_action_count": 0,
                "branch_point_count": 40,
                "source_seed_count": 8,
                "safe_non_logged_override_count": 8,
                "safe_non_logged_override_share": 0.2,
                "dominant_oracle_action_share": 0.5,
                "target_alive_delta_negative_count": 0,
                "target_local_score_delta_summary": {"mean": 0.1},
                "terminal_alive_delta_summary": {"mean": 0.0},
                "birth_delta_summary": {"mean": 0.0},
            },
            floors=floors,
        )
        rejected = _acceptance(
            aggregate={
                "strict_seed_leak_count": 0,
                "replay_verified": True,
                "heuristic_action_source_count": 0,
                "unsupported_candidate_action_count": 0,
                "branch_point_count": 40,
                "source_seed_count": 8,
                "safe_non_logged_override_count": 0,
                "safe_non_logged_override_share": 0.0,
                "dominant_oracle_action_share": 0.5,
                "target_alive_delta_negative_count": 0,
                "target_local_score_delta_summary": {"mean": 0.1},
                "terminal_alive_delta_summary": {"mean": 0.0},
                "birth_delta_summary": {"mean": 0.0},
            },
            floors=floors,
        )

        self.assertTrue(accepted["v100_residual_distillation_allowed"])
        self.assertFalse(rejected["v100_residual_distillation_allowed"])
        self.assertIn(
            "insufficient_safe_non_logged_overrides",
            {blocker["reason"] for blocker in rejected["blockers"]},
        )

    def test_support_probe_policy_name_is_stable(self) -> None:
        self.assertEqual(
            MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_POLICY,
            "v99_broad_linear_residual_branch_oracle_v1",
        )


def _candidate(
    *,
    index: int,
    category: str,
    priority: float,
) -> _BroadBranchCandidate:
    return _BroadBranchCandidate(
        seed=2,
        tick=index,
        record_index=index,
        agent_id=100 + index,
        logged_action="stay",
        before={
            "alive": True,
            "x": 0,
            "y": 0,
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 0.8,
        },
        action_mask={"stay": True, "eat": True},
        observation_input={},
        observation_digest=None,
        observation_schema=None,
        public_history_trace=(),
        compact_state={},
        legal_actions=("stay", "eat"),
        categories=(category,),
        priority=priority,
    )


class _FakeDelegate:
    policy_id = "fake_linear"
    policy_version = "fake_linear_v1"

    def __init__(self) -> None:
        self.observed: list[dict[str, object]] = []

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        return ActionDecision(
            requested_action="stay",
            source=self.policy_version,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics={"heuristic_free": True},
        )

    def observe_transition(self, record: dict[str, object]) -> dict[str, object]:
        self.observed.append(dict(record))
        return {"observed": True}


if __name__ == "__main__":
    unittest.main()
