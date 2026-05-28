from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.mind.broad_transfer_residual_audit import (
    build_broad_transfer_residual_audit_report,
    load_linear_support_rows,
    load_v97_planner_broad_rows,
    select_support_archive_rows,
    _residual_gate_allows,
)
from evolution_sim.mind.branch_utility_risk_audit import _utility_rows
from python.tests.test_mind_v3_branch_sequence_continuation_scorer import (
    _labels_with_trace_targets,
)
from python.tests.test_mind_v3_planner_distilled_runtime import (
    _planner_distillation_report,
)


class MindV3BroadTransferResidualAuditTests(unittest.TestCase):
    def test_v97_parser_identifies_high_margin_stay_collapse(self) -> None:
        report = _planner_distillation_report()
        state = _policy_state()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v97_dir = root / "v97"
            support_dir = root / "support"
            v97_dir.mkdir()
            support_dir.mkdir()
            _write_trajectory(
                v97_dir / "open-mind-v3-5-120.jsonl.gz",
                seed=5,
                records=[
                    _record(
                        state=state,
                        tick=0,
                        agent_id=1,
                        action="stay",
                        planner=True,
                    )
                ],
            )
            _write_support_trajectories(support_dir, state=state, seeds=(2, 3))

            rows = load_v97_planner_broad_rows(v97_dir)
            audit, _archive = build_broad_transfer_residual_audit_report(
                v97_report={},
                v97_trajectory_dir=v97_dir,
                planner_distilled_artifact=report,
                support_trajectory_dir=support_dir,
                support_target_row_count=8,
                min_support_rows=8,
                min_source_seeds=2,
                min_reposition_share=0.25,
                max_dominant_support_teacher_action_share=0.5,
                max_override_action_share=1.0,
                max_abstention_rate=1.0,
            )

        self.assertEqual(len(rows), 1)
        collapse = audit["v97_collapse_analysis"][
            "first_concrete_collapse_pattern"
        ]
        self.assertEqual(collapse["seed"], 5)
        self.assertEqual(collapse["requested_action"], "stay")
        self.assertIn("eat", collapse["legal_actions"])

    def test_support_selection_reports_balanced_non_strict_coverage(self) -> None:
        state = _policy_state()
        with TemporaryDirectory() as tmpdir:
            support_dir = Path(tmpdir)
            _write_support_trajectories(support_dir, state=state, seeds=(2, 3))

            rows = load_linear_support_rows(support_dir)
            selected = select_support_archive_rows(
                rows,
                target_count=8,
                max_dominant_action_share=0.5,
            )

        self.assertEqual({row["source_seed"] for row in selected}, {2, 3})
        self.assertEqual(len(selected), 8)
        modes = {row["mode"] for row in selected}
        self.assertTrue(
            {"conserve", "recover_hydration", "exploit_resource", "reposition"}
            <= modes
        )

    def test_strict_seed_support_leakage_is_a_blocker(self) -> None:
        report = _planner_distillation_report()
        state = _policy_state()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v97_dir = root / "v97"
            support_dir = root / "support"
            v97_dir.mkdir()
            support_dir.mkdir()
            _write_trajectory(
                v97_dir / "open-mind-v3-5-120.jsonl.gz",
                seed=5,
                records=[
                    _record(
                        state=state,
                        tick=0,
                        agent_id=1,
                        action="stay",
                        planner=True,
                    )
                ],
            )
            _write_support_trajectories(support_dir, state=state, seeds=(5,))

            audit, _archive = build_broad_transfer_residual_audit_report(
                v97_report={},
                v97_trajectory_dir=v97_dir,
                planner_distilled_artifact=report,
                support_trajectory_dir=support_dir,
                support_target_row_count=4,
                min_support_rows=4,
                min_source_seeds=1,
                min_reposition_share=0.0,
                max_dominant_support_teacher_action_share=1.0,
                max_override_action_share=1.0,
                max_abstention_rate=1.0,
            )

        reasons = {
            blocker["reason"] for blocker in audit["acceptance"]["blockers"]
        }
        self.assertIn("strict_seed_leakage", reasons)
        self.assertFalse(audit["v99_support_gated_residual_runtime_allowed"])

    def test_residual_gate_blocks_unsupported_distance_and_local_collapse(self) -> None:
        allowed, reason = _residual_gate_allows(
            candidate_action="eat",
            legal=False,
            distance=0.0,
            support_distance_threshold=1.0,
            margin=1.0,
            planner_score_margin_threshold=0.05,
            override_window=[],
            max_override_action_share=0.5,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "unsupported_action")

        allowed, reason = _residual_gate_allows(
            candidate_action="eat",
            legal=True,
            distance=2.0,
            support_distance_threshold=1.0,
            margin=1.0,
            planner_score_margin_threshold=0.05,
            override_window=[],
            max_override_action_share=0.5,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "outside_support")

        allowed, reason = _residual_gate_allows(
            candidate_action="eat",
            legal=True,
            distance=0.0,
            support_distance_threshold=1.0,
            margin=1.0,
            planner_score_margin_threshold=0.05,
            override_window=["eat"] * 7,
            max_override_action_share=0.5,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "local_action_collapse_cap")


def _policy_state() -> dict[str, object]:
    labels = _labels_with_trace_targets(seed_start=101, branch_prefix="support")
    row = _utility_rows(labels["labels"])[0]
    return dict(row["policy_state"])


def _write_support_trajectories(
    root: Path,
    *,
    state: dict[str, object],
    seeds: tuple[int, ...],
) -> None:
    actions = ("stay", "drink", "eat", "move_east")
    for seed in seeds:
        records = [
            _record(
                state=state,
                tick=index,
                agent_id=seed * 10 + index,
                action=action,
                planner=False,
            )
            for index, action in enumerate(actions)
        ]
        _write_trajectory(
            root / f"open-mind-v3-linear-{seed}-120.jsonl.gz",
            seed=seed,
            records=records,
        )


def _write_trajectory(path: Path, *, seed: int, records: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"type": "header", "config": {"seed": seed}})
            + "\n"
        )
        for record in records:
            handle.write(json.dumps({"type": "record", "record": record}) + "\n")


def _record(
    *,
    state: dict[str, object],
    tick: int,
    agent_id: int,
    action: str,
    planner: bool,
) -> dict[str, object]:
    action_mask = dict(state["action_mask"])
    for required in ("stay", "drink", "eat", "move_east"):
        action_mask[required] = True
    before = {
        "alive": True,
        "x": agent_id % 5,
        "y": agent_id % 7,
        "energy_ratio": 0.7,
        "hydration_ratio": 0.7,
        "health_ratio": 0.9,
    }
    after = dict(before)
    after["energy_ratio"] = 0.68
    diagnostics = {"runtime_mode": "linear", "heuristic_free": True}
    if planner:
        diagnostics = {
            "planner_distilled_score_margin": 0.25,
            "planner_distilled_candidate_scores_top": [
                {
                    "action": "stay",
                    "mode": "conserve",
                    "final_score": 10.0,
                    "sequence_cvar_score": 10.0,
                    "utility_weighted_teacher_margin": 0.0,
                    "learned_action_penalty": 0.0,
                },
                {
                    "action": "eat",
                    "mode": "exploit_resource",
                    "final_score": 9.75,
                    "sequence_cvar_score": 9.75,
                    "utility_weighted_teacher_margin": 0.0,
                    "learned_action_penalty": 0.0,
                },
            ],
            "heuristic_free": True,
        }
    return {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": action,
        "resolved_action": action,
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": action.startswith("move_"),
        "action_mask": action_mask,
        "observation_input": state["observation_input"],
        "before": before,
        "after": after,
        "outcome": {
            "died": False,
            "resource_gain": 0.1 if action in {"eat", "drink"} else 0.0,
            "feeding": {
                "ate": action == "eat",
                "food_source": "plant" if action == "eat" else None,
            },
            "drinking": {"drank": action == "drink"},
            "passive": {"died_after_action": False, "death_cause": None},
            "reproduced": False,
            "reproduction_ready_after": action == "stay",
        },
        "reward": {
            "total": 0.0,
            "components": {
                "reproduction_readiness": 0.1 if action == "stay" else 0.0,
            },
        },
        "policy_decision_diagnostics": diagnostics,
    }


if __name__ == "__main__":
    unittest.main()
