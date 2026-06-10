from __future__ import annotations

import base64
from copy import deepcopy
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
)
from evolution_sim.mind.broad_regression_branch_intervention import (
    MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
    MIND_V3_V143_BRANCH_INTERVENTION_SCHEMA_VERSION,
)
from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
    M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
    M3_SAFE_ARCHIVE_EXPANSION_POLICY,
    CandidateCampaignError,
    build_safe_archive_expansion_report,
    build_safe_archive_train_eval_acceptance,
    build_safe_archive_train_eval_preflight,
    build_candidate_specs,
    build_safe_branch_label_archive,
    load_safe_archive_expansion_branch_result_chunks,
    merge_safe_archive_expansion_branch_evidence,
    rank_candidate_results,
    run_safe_archive_train_eval,
    safe_archive_expansion_leakage_scan,
    validate_safe_archive_train_eval_inputs,
    _safe_archive_expansion_branch_result_matches_point,
    _safe_archive_expansion_selected_branch_indexes,
    _safe_archive_expansion_carrion_branch_evidence,
    write_campaign_ledger,
    write_safe_archive_expansion_branch_evidence,
    write_safe_archive_expansion_dataset,
    write_safe_archive_expansion_branch_result_chunk,
    write_safe_archive_expansion_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.safe_archive_failure_autopsy import (
    build_safe_archive_failure_autopsy_report,
)
from evolution_sim.mind.safe_archive_sequence_context_audit import (
    build_safe_archive_sequence_context_audit_report,
    sequence_context_trainable_leakage_scan,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CandidateCampaignTests(unittest.TestCase):
    def test_safe_archive_excludes_v145_blacklist_and_blocks_small_support(self) -> None:
        rows = [
            _dataset_row(seed=5, agent_id=6, action="stay", unsupported_resolved=0),
            _dataset_row(seed=37, agent_id=3, action="move_north", unsupported_resolved=-4),
        ]
        archive = build_safe_branch_label_archive(
            v143_report=_v143_report(rows),
            dataset_rows=rows,
            v145_report=_v145_report(),
            min_safe_label_count=DEFAULT_MIN_SAFE_LABEL_COUNT,
        )

        self.assertEqual(archive["source_dataset_row_count"], 2)
        self.assertEqual(archive["v145_blacklist_count"], 1)
        self.assertEqual(archive["safe_label_count"], 1)
        self.assertFalse(archive["archive_support_sufficient"])
        self.assertEqual(archive["safe_action_counts"], {"move_north": 1})
        self.assertEqual(
            archive["expansion_plan"]["blocked_reason"],
            "safe_label_count_below_minimum_before_training",
        )

    def test_candidate_specs_mark_support_arms_skipped_when_safe_count_small(self) -> None:
        rows = [_dataset_row(seed=37, agent_id=3, action="move_north", unsupported_resolved=-4)]
        archive = build_safe_branch_label_archive(
            v143_report=_v143_report(rows),
            dataset_rows=rows,
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=DEFAULT_MIN_SAFE_LABEL_COUNT,
        )

        specs = build_candidate_specs(
            archive=archive,
            dataset_rows=rows,
            mode="smoke",
            candidate_limit=None,
            v144_report=None,
            v144_artifact_path=ROOT / "does-not-exist.json",
            output_dir=Path("output/mind/v146-test"),
            keep_trajectories=False,
        )

        by_id = {spec["candidate_id"]: spec for spec in specs}
        self.assertEqual(by_id["linear_control"]["skip_reason"], None)
        self.assertEqual(
            by_id["v146_blacklist_safe_branch_residual"]["skip_reason"],
            "archive_support_insufficient_safe_label_count_lt_20",
        )
        self.assertEqual(
            by_id["neural_offline"]["skip_reason"],
            "safe_label_count_too_small_for_neural_offline_arm",
        )

    def test_leaderboard_order_is_deterministic(self) -> None:
        results = [
            _result("slow", accepted=False, failed=2, resolved=4, order=2),
            _result("accepted", accepted=True, failed=0, resolved=0, order=1),
            _result("less_bad", accepted=False, failed=1, resolved=1, order=0),
        ]

        first = rank_candidate_results(results)
        second = rank_candidate_results(list(reversed(results)))

        self.assertEqual(
            [row["candidate_id"] for row in first],
            ["accepted", "less_bad", "slow"],
        )
        self.assertEqual(first, second)

    def test_linear_control_is_excluded_from_candidate_leaderboard(self) -> None:
        results = [
            _result("linear_control", accepted=False, failed=0, resolved=0, order=0),
            _result("candidate", accepted=False, failed=1, resolved=2, order=1),
        ]
        results[0]["is_control"] = True
        results[0]["candidate_evaluation_role"] = "control"
        results[0]["control_passed"] = True

        leaderboard = rank_candidate_results(results)

        self.assertEqual([row["candidate_id"] for row in leaderboard], ["candidate"])
        self.assertEqual(leaderboard[0]["candidate_evaluation_role"], "candidate")

    def test_ledger_writes_jsonl_rows(self) -> None:
        report = {
            "schema_version": "mind_v3_v146_candidate_campaign_report_v1",
            "controls": [
                {
                    **_result(
                        "linear_control",
                        accepted=False,
                        failed=0,
                        resolved=0,
                        order=0,
                    ),
                    "is_control": True,
                    "candidate_evaluation_role": "control",
                    "control_passed": True,
                }
            ],
            "candidates": [
                _result("candidate", accepted=False, failed=1, resolved=2, order=1)
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ledger.jsonl"
            write_campaign_ledger(path, report)
            lines = path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), 2)
        control = json.loads(lines[0])
        candidate = json.loads(lines[1])
        self.assertEqual(control["candidate_id"], "linear_control")
        self.assertTrue(control["is_control"])
        self.assertEqual(control["candidate_evaluation_role"], "control")
        self.assertTrue(control["control_passed"])
        self.assertFalse(control["accepted"])
        self.assertEqual(candidate["candidate_id"], "candidate")
        self.assertFalse(candidate["promotion_authorized"])

    def test_safe_archive_expansion_leakage_guard_blocks_trainable_identity(self) -> None:
        rows = [
            {
                "schema_version": M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
                "trainable": {
                    "features": {
                        "observation_input": {},
                        "action_mask": {},
                        "seed": 5,
                    },
                    "label": {"action": "stay"},
                },
                "metadata": {},
            }
        ]

        scan = safe_archive_expansion_leakage_scan(
            rows,
            strict_heldout_seeds=(5,),
        )

        self.assertFalse(scan["passed"])
        self.assertGreater(scan["forbidden_path_failure_count"], 0)
        self.assertGreater(scan["strict_heldout_seed_trainable_leakage_count"], 0)

    def test_safe_archive_expansion_json_and_jsonl_are_deterministic(self) -> None:
        report = {"schema_version": "test", "z": 2, "a": {"b": 1}}
        rows = [_safe_expansion_branch_result(seed=13, fixture="carrion_only", action="drink")]
        expansion_report, dataset_rows = build_safe_archive_expansion_report(
            branch_results=rows,
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            a = Path(tmpdir) / "a.json"
            b = Path(tmpdir) / "b.json"
            aj = Path(tmpdir) / "a.jsonl"
            bj = Path(tmpdir) / "b.jsonl"
            write_safe_archive_expansion_report({**report, **expansion_report}, a)
            write_safe_archive_expansion_report({**report, **expansion_report}, b)
            write_safe_archive_expansion_dataset(dataset_rows, aj)
            write_safe_archive_expansion_dataset(dataset_rows, bj)

            self.assertEqual(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"))
            self.assertEqual(aj.read_text(encoding="utf-8"), bj.read_text(encoding="utf-8"))

    def test_safe_archive_expansion_branch_result_chunks_resume_deterministically(
        self,
    ) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_dir = Path(tmpdir) / "chunks"
            first_path = write_safe_archive_expansion_branch_result_chunk(
                branch_result,
                chunk_dir,
            )
            first_text = first_path.read_text(encoding="utf-8")
            second_path = write_safe_archive_expansion_branch_result_chunk(
                branch_result,
                chunk_dir,
            )
            second_text = second_path.read_text(encoding="utf-8")
            loaded_once = load_safe_archive_expansion_branch_result_chunks(chunk_dir)
            loaded_twice = load_safe_archive_expansion_branch_result_chunks(chunk_dir)

        self.assertEqual(first_path, second_path)
        self.assertEqual(first_text, second_text)
        self.assertEqual(loaded_once, loaded_twice)
        self.assertEqual(
            loaded_once[branch_result["branch_id"]]["branch_id"],
            branch_result["branch_id"],
        )

    def test_safe_archive_expansion_shard_branch_index_selection(self) -> None:
        self.assertEqual(
            _safe_archive_expansion_selected_branch_indexes(
                max_branch_points_per_seed=4,
                branch_index_start=1,
                branch_index_count=2,
            ),
            (1, 2),
        )
        self.assertEqual(
            _safe_archive_expansion_selected_branch_indexes(
                max_branch_points_per_seed=4,
                branch_index_include=(3, 1),
            ),
            (1, 3),
        )
        with self.assertRaises(CandidateCampaignError):
            _safe_archive_expansion_selected_branch_indexes(
                max_branch_points_per_seed=2,
                branch_index_include=(5,),
            )

    def test_safe_archive_expansion_shard_merge_is_deterministic(self) -> None:
        broad = _safe_expansion_branch_result(
            seed=5,
            fixture="broad",
            action="stay",
        )
        carrion = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        carrion = _rewrite_branch_identity(
            carrion,
            branch_id="m3-carrion-seed-13-branch-1-tick-0-agent-4",
            branch_index=1,
        )

        first = merge_safe_archive_expansion_branch_evidence(
            shard_evidence_reports=[
                _safe_archive_shard_evidence([carrion], shard_id="carrion-13-1"),
                _safe_archive_shard_evidence([broad], shard_id="broad-5-0"),
            ]
        )
        second = merge_safe_archive_expansion_branch_evidence(
            shard_evidence_reports=[
                _safe_archive_shard_evidence([broad], shard_id="broad-5-0"),
                _safe_archive_shard_evidence([carrion], shard_id="carrion-13-1"),
            ]
        )

        self.assertEqual(first["branch_results"], second["branch_results"])
        self.assertEqual(first["branch_evidence_digest"], second["branch_evidence_digest"])
        self.assertEqual(
            [row["branch_id"] for row in first["branch_results"]],
            [broad["branch_id"], carrion["branch_id"]],
        )

    def test_safe_archive_expansion_shard_merge_rejects_duplicate_conflict(
        self,
    ) -> None:
        first = _safe_expansion_branch_result(
            seed=5,
            fixture="broad",
            action="stay",
        )
        conflicting = deepcopy(first)
        conflicting["action_runs"][0]["alive_agents"] += 1

        with self.assertRaises(CandidateCampaignError):
            merge_safe_archive_expansion_branch_evidence(
                shard_evidence_reports=[
                    _safe_archive_shard_evidence([first], shard_id="a"),
                    _safe_archive_shard_evidence([conflicting], shard_id="b"),
                ]
            )

    def test_safe_archive_expansion_shard_merge_rejects_source_integrity_failure(
        self,
    ) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=5,
            fixture="broad",
            action="stay",
        )

        with self.assertRaises(CandidateCampaignError):
            merge_safe_archive_expansion_branch_evidence(
                shard_evidence_reports=[
                    _safe_archive_shard_evidence(
                        [branch_result],
                        shard_id="failed-source",
                        source_integrity_failures=(
                            "broad_branch_point_materialization_failed",
                        ),
                    )
                ]
            )

    def test_safe_archive_expansion_shard_merge_rejects_unidentified_chunk_dir(
        self,
    ) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=5,
            fixture="broad",
            action="stay",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_dir = Path(tmpdir) / "chunks"
            write_safe_archive_expansion_branch_result_chunk(branch_result, chunk_dir)

            with self.assertRaises(CandidateCampaignError):
                merge_safe_archive_expansion_branch_evidence(
                    shard_chunk_dirs=[chunk_dir],
                )

    def test_safe_archive_expansion_shard_merge_rejects_unmatched_chunk_result(
        self,
    ) -> None:
        evidence_result = _safe_expansion_branch_result(
            seed=5,
            fixture="broad",
            action="stay",
        )
        chunk_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_dir = Path(tmpdir) / "chunks"
            write_safe_archive_expansion_branch_result_chunk(chunk_result, chunk_dir)

            with self.assertRaises(CandidateCampaignError):
                merge_safe_archive_expansion_branch_evidence(
                    shard_evidence_reports=[
                        _safe_archive_shard_evidence(
                            [evidence_result],
                            shard_id="evidence",
                        )
                    ],
                    shard_chunk_dirs=[chunk_dir],
                )

    def test_safe_archive_expansion_shard_merge_partial_fails_closed(self) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        partial = _safe_archive_shard_evidence(
            [branch_result],
            shard_id="partial",
            partial=True,
        )

        with self.assertRaises(CandidateCampaignError):
            merge_safe_archive_expansion_branch_evidence(
                shard_evidence_reports=[partial],
            )

        merged = merge_safe_archive_expansion_branch_evidence(
            shard_evidence_reports=[partial],
            allow_partial_shard_evidence=True,
        )
        report, rows = build_safe_archive_expansion_report(
            branch_results=merged["branch_results"],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
            branch_evidence_status=merged["generation_status"],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(merged["generation_status"]["state"], "partial")
        self.assertFalse(report["source_integrity"]["passed"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["contract"]["promotion_authorized"])
        self.assertEqual(
            report["classification"]["primary"],
            "m3_safe_archive_expansion_partial_branch_evidence_no_training",
        )

    def test_safe_archive_expansion_resume_chunk_must_match_current_branch_point(
        self,
    ) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        point = _branch_point_for_safe_expansion_result(branch_result)

        self.assertTrue(
            _safe_archive_expansion_branch_result_matches_point(
                branch_result,
                point,
                max_candidate_actions=0,
                verify_replay=True,
            )
        )

        stale_state = deepcopy(branch_result)
        stale_state["branch_state_digest"] = "old-branch-state"
        self.assertFalse(
            _safe_archive_expansion_branch_result_matches_point(
                stale_state,
                point,
                max_candidate_actions=0,
                verify_replay=True,
            )
        )

        stale_action_limit = deepcopy(branch_result)
        stale_action_limit["candidate_actions"] = ["stay"]
        stale_action_limit["action_runs"] = stale_action_limit["action_runs"][:1]
        self.assertFalse(
            _safe_archive_expansion_branch_result_matches_point(
                stale_action_limit,
                point,
                max_candidate_actions=0,
                verify_replay=True,
            )
        )

        stale_replay_policy = deepcopy(branch_result)
        stale_replay_policy["action_runs"][0]["replay_verification"] = None
        self.assertFalse(
            _safe_archive_expansion_branch_result_matches_point(
                stale_replay_policy,
                point,
                max_candidate_actions=0,
                verify_replay=True,
            )
        )

    def test_safe_archive_expansion_excludes_v145_blacklist(self) -> None:
        branch_results = [
            _safe_expansion_branch_result(
                seed=5,
                fixture="broad",
                action="stay",
                branch_id="v143-broad-seed-5-branch-0-tick-0-agent-6",
                agent_id=6,
            ),
            _safe_expansion_branch_result(
                seed=37,
                fixture="broad",
                action="drink",
                branch_id="m3-broad-seed-37-branch-0-tick-0-agent-7",
                agent_id=7,
            ),
        ]

        report, rows = build_safe_archive_expansion_report(
            branch_results=branch_results,
            v145_report=_v145_report(),
            min_safe_label_count=1,
        )

        self.assertEqual(report["blacklist"]["blacklist_count"], 1)
        self.assertEqual(report["dataset"]["safe_label_count"], 1)
        self.assertEqual(rows[0]["trainable"]["label"]["action"], "drink")
        self.assertEqual(report["excluded_rows"][0]["excluded_reason"], "v145_causal_blacklist")

    def test_safe_archive_expansion_does_not_reuse_v143_row_index_blacklist(self) -> None:
        report, rows = build_safe_archive_expansion_report(
            branch_results=[
                _safe_expansion_branch_result(
                    seed=5,
                    fixture="broad",
                    action="eat",
                    branch_id=(
                        "m3-safe-archive-broad-regression-seed-5-branch-0-"
                        "tick-0-agent-4"
                    ),
                    agent_id=4,
                )
            ],
            v145_report=_v145_report(),
            min_safe_label_count=1,
        )

        self.assertEqual(report["dataset"]["safe_label_count"], 1)
        self.assertEqual(rows[0]["trainable"]["label"]["action"], "eat")
        self.assertEqual(report["excluded_row_count"], 0)

    def test_safe_archive_expansion_records_carrion_row_coverage(self) -> None:
        branch_results = [
            _safe_expansion_branch_result(seed=5, fixture="broad", action="stay"),
            _safe_expansion_branch_result(seed=13, fixture="carrion_only", action="drink"),
        ]

        report, rows = build_safe_archive_expansion_report(
            branch_results=branch_results,
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(report["dataset"]["broad_safe_label_count"], 1)
        self.assertEqual(report["dataset"]["carrion_safe_label_count"], 1)
        self.assertEqual(report["coverage"]["branch_point_counts"]["carrion_only"], 1)

    def test_safe_archive_carrion_evidence_evaluates_all_public_mask_actions(self) -> None:
        evidence = _safe_archive_expansion_carrion_branch_evidence(
            seeds=(13,),
            ticks=2,
            max_branch_points_per_seed=1,
            max_candidate_actions=0,
            verify_replay=True,
        )

        self.assertTrue(evidence["source_integrity"]["passed"])
        result = evidence["branch_results"][0]
        public_mask = result["public_features"]["action_mask"]
        valid_actions = {
            action for action in ACTION_NAMES if public_mask.get(action) is True
        }
        action_runs = result["action_runs"]
        represented_actions = {run["forced_action"] for run in action_runs}

        self.assertGreaterEqual(len(valid_actions), 1)
        self.assertEqual(represented_actions, valid_actions)
        for run in action_runs:
            self.assertTrue(run["replay_verification"]["verified"])
            baseline = run["deltas_vs_baseline"]
            for key in (
                "alive_agents",
                "births",
                "deaths",
                "target_alive",
                "unsupported_resolved_action_count",
            ):
                self.assertIn(key, baseline)

    def test_safe_archive_expansion_source_integrity_fails_when_action_missing(self) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        branch_result["public_features"]["action_mask"]["move_north"] = True

        report, rows = build_safe_archive_expansion_report(
            branch_results=[branch_result],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "not_all_valid_candidate_actions_evaluated",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(
            report["source_integrity"]["first_missing_valid_action"][
                "missing_action"
            ],
            "move_north",
        )
        self.assertEqual(len(rows), 1)

    def test_safe_archive_expansion_diagnostics_aggregate_safety_failures(self) -> None:
        carrion = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        for action_run in carrion["action_runs"]:
            action_run["deltas_vs_baseline"]["alive_agents"] = -1
            action_run["deltas_vs_baseline"]["births"] = -1
        carrion["action_runs"][0]["deltas_vs_baseline"][
            "unsupported_resolved_action_count"
        ] = 2
        carrion["action_runs"][1]["replay_verification"]["verified"] = False
        carrion["action_runs"][2]["heuristic_action_source_count"] = 1

        report, rows = build_safe_archive_expansion_report(
            branch_results=[carrion],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        self.assertEqual(rows, [])
        diagnostics = report["diagnostics"]
        failures = diagnostics["safety_vet_failures"]
        self.assertEqual(failures["action_run_count"], 3)
        self.assertEqual(failures["safe_action_run_count"], 0)
        self.assertEqual(
            failures["failed_floor_counts_by_fixture"]["carrion_only"][
                "no_alive_regression"
            ],
            3,
        )
        self.assertEqual(
            failures["failed_floor_counts_by_action"]["stay"][
                "no_resolved_invalid_increase"
            ],
            1,
        )
        self.assertEqual(
            failures["failed_floor_counts_by_fixture_action"]["carrion_only:eat"][
                "branch_replay_deterministic"
            ],
            1,
        )
        excluded = diagnostics["excluded"]
        self.assertEqual(
            excluded["excluded_reason_counts"],
            {"no_safety_vetted_label_action": 1},
        )

    def test_safe_archive_expansion_carrion_failure_summary_is_actionable(self) -> None:
        carrion = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        for action_run in carrion["action_runs"]:
            action_run["deltas_vs_baseline"]["alive_agents"] = -1
            action_run["deltas_vs_baseline"]["births"] = -1
        carrion["action_runs"][0]["deltas_vs_baseline"][
            "unsupported_resolved_action_count"
        ] = 2
        carrion["action_runs"][1]["replay_verification"]["verified"] = False
        carrion["action_runs"][2]["heuristic_action_source_count"] = 1

        report, _rows = build_safe_archive_expansion_report(
            branch_results=[carrion],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        carrion_diag = report["diagnostics"]["carrion"]
        self.assertEqual(carrion_diag["action_run_count"], 3)
        self.assertEqual(carrion_diag["safe_action_run_count"], 0)
        self.assertEqual(
            carrion_diag["unsupported_resolved_action_count"]["failed_run_count"],
            1,
        )
        self.assertEqual(carrion_diag["alive_regression"]["failed_run_count"], 3)
        self.assertEqual(carrion_diag["birth_regression"]["failed_run_count"], 3)
        self.assertEqual(carrion_diag["replay_failure"]["failed_run_count"], 1)
        self.assertEqual(
            carrion_diag["heuristic_action_source_count"]["failed_run_count"],
            1,
        )
        self.assertEqual(
            report["diagnostics"]["support_limitation_assessment"]["primary"],
            "source_integrity_limited",
        )

    def test_safe_archive_expansion_blocks_training_when_support_floor_fails(self) -> None:
        report, rows = build_safe_archive_expansion_report(
            branch_results=[
                _safe_expansion_branch_result(seed=13, fixture="carrion_only", action="drink")
            ],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=DEFAULT_MIN_SAFE_LABEL_COUNT,
        )

        self.assertEqual(len(rows), 1)
        self.assertFalse(report["archive_support_passed"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(
            report["classification"]["primary"],
            "m3_safe_archive_expansion_blocked_no_training",
        )
        self.assertEqual(
            report["support_floors"]["first_failed_floor"],
            "safe_label_count_gte_minimum",
        )

    def test_safe_archive_expansion_partial_evidence_fails_closed(self) -> None:
        report, rows = build_safe_archive_expansion_report(
            branch_results=[
                _safe_expansion_branch_result(seed=13, fixture="carrion_only", action="drink")
            ],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
            branch_evidence_status={
                "state": "partial",
                "partial": True,
                "stop_reason": "max_wall_seconds_elapsed_before_branch_point",
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn("partial_branch_evidence", report["source_integrity"]["failures"])
        self.assertFalse(report["archive_support_passed"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["contract"]["promotion_authorized"])
        self.assertEqual(
            report["classification"]["primary"],
            "m3_safe_archive_expansion_partial_branch_evidence_no_training",
        )

    def test_safe_archive_expansion_requires_baseline_delta_evidence(self) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        for action_run in branch_result["action_runs"]:
            action_run.pop("deltas_vs_baseline")

        report, rows = build_safe_archive_expansion_report(
            branch_results=[branch_result],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        self.assertEqual(rows, [])
        self.assertEqual(report["dataset"]["safe_label_count"], 0)
        self.assertEqual(
            report["excluded_rows"][0]["excluded_reason"],
            "no_safety_vetted_label_action",
        )

    def test_safe_archive_expansion_trainable_row_is_public_surface_only(self) -> None:
        report, rows = build_safe_archive_expansion_report(
            branch_results=[
                _safe_expansion_branch_result(seed=13, fixture="carrion_only", action="drink")
            ],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=1,
        )

        self.assertEqual(report["dataset"]["safe_label_count"], 1)
        trainable = rows[0]["trainable"]
        self.assertEqual(
            set(trainable),
            {"feature_policy", "features", "label"},
        )
        self.assertEqual(
            set(trainable["features"]),
            {"observation_input", "action_mask"},
        )
        self.assertEqual(set(trainable["label"]), {"action", "label_policy"})
        trainable_text = json.dumps(trainable, sort_keys=True)
        for forbidden in (
            "seed",
            "fixture",
            "branch_id",
            "agent_id",
            "path",
            "digest",
        ):
            self.assertNotIn(forbidden, trainable_text)

    def test_safe_archive_train_eval_rejects_blocked_default_archive(self) -> None:
        branch_result = _safe_expansion_branch_result(
            seed=13,
            fixture="carrion_only",
            action="drink",
        )
        report, rows = build_safe_archive_expansion_report(
            branch_results=[branch_result],
            v145_report={"route_decision": {"label_blacklist": []}},
            min_safe_label_count=DEFAULT_MIN_SAFE_LABEL_COUNT,
        )
        evidence = _safe_archive_shard_evidence([branch_result], shard_id="old")

        with self.assertRaises(CandidateCampaignError) as context:
            validate_safe_archive_train_eval_inputs(
                safe_archive_report=report,
                safe_archive_dataset_rows=rows,
                branch_evidence_report=evidence,
            )

        self.assertIn("safe_archive_report_not_support_ready", str(context.exception))

    def test_safe_archive_train_eval_rejects_wrong_dataset_digest(self) -> None:
        report, rows, evidence = _safe_train_eval_fixture()

        with self.assertRaises(CandidateCampaignError) as context:
            validate_safe_archive_train_eval_inputs(
                safe_archive_report=report,
                safe_archive_dataset_rows=rows,
                branch_evidence_report=evidence,
                expected_safe_label_count=2,
                expected_min_safe_label_count=2,
                expected_dataset_digest="wrong-digest",
                expected_branch_evidence_digest=str(evidence["branch_evidence_digest"]),
            )

        self.assertIn("safe_archive_dataset_digest_mismatch", str(context.exception))

    def test_safe_archive_train_eval_rejects_promotion_authorized_true(self) -> None:
        report, rows, evidence = _safe_train_eval_fixture()
        report["promotion_authorized"] = True

        with self.assertRaises(CandidateCampaignError) as context:
            validate_safe_archive_train_eval_inputs(
                safe_archive_report=report,
                safe_archive_dataset_rows=rows,
                branch_evidence_report=evidence,
                expected_safe_label_count=2,
                expected_min_safe_label_count=2,
                expected_dataset_digest=stable_payload_digest(rows),
                expected_branch_evidence_digest=str(evidence["branch_evidence_digest"]),
            )

        self.assertIn(
            "safe_archive_report_promotion_authorized_not_false",
            str(context.exception),
        )

    def test_safe_archive_train_eval_rejects_trainable_leakage(self) -> None:
        report, rows, evidence = _safe_train_eval_fixture()
        rows = deepcopy(rows)
        rows[0]["trainable"]["features"]["seed"] = 5
        report = deepcopy(report)
        report["dataset"]["dataset_digest"] = stable_payload_digest(rows)

        with self.assertRaises(CandidateCampaignError) as context:
            validate_safe_archive_train_eval_inputs(
                safe_archive_report=report,
                safe_archive_dataset_rows=rows,
                branch_evidence_report=evidence,
                expected_safe_label_count=2,
                expected_min_safe_label_count=2,
                expected_dataset_digest=stable_payload_digest(rows),
                expected_branch_evidence_digest=str(evidence["branch_evidence_digest"]),
            )

        self.assertIn("safe_archive_trainable_leakage_scan_failed", str(context.exception))

    def test_safe_archive_train_eval_preflight_reports_zero_support_seeds(self) -> None:
        _report, rows, evidence = _safe_train_eval_fixture()

        preflight = build_safe_archive_train_eval_preflight(
            safe_archive_dataset_rows=rows,
            branch_evidence_report=evidence,
        )

        self.assertTrue(preflight["broad_seed_41_absent_from_archive_support"])
        self.assertTrue(preflight["carrion_seed_13_safe_label_absent"])
        self.assertIn(
            {"fixture": "broad", "seed": 41},
            preflight["zero_support_strict_seeds"],
        )
        self.assertTrue(preflight["candidate_action_coverage"]["complete"])
        self.assertTrue(preflight["replay_verification"]["complete"])

    def test_safe_archive_train_eval_writes_non_promoted_artifact_and_report(self) -> None:
        report, rows, evidence = _safe_train_eval_fixture()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "safe-report.json"
            dataset_path = root / "safe-dataset.jsonl"
            evidence_path = root / "safe-evidence.json"
            artifact_path = root / "artifact.json"
            output_path = root / "train-eval-report.json"
            write_safe_archive_expansion_report(report, report_path)
            write_safe_archive_expansion_dataset(rows, dataset_path)
            write_safe_archive_expansion_branch_evidence(evidence, evidence_path)

            result = run_safe_archive_train_eval(
                safe_archive_report_path=report_path,
                safe_archive_dataset_path=dataset_path,
                branch_evidence_report_path=evidence_path,
                artifact_output_path=artifact_path,
                output_path=output_path,
                run_evaluation=False,
                expected_safe_label_count=2,
                expected_min_safe_label_count=2,
                expected_dataset_digest=stable_payload_digest(rows),
                expected_branch_evidence_digest=str(evidence["branch_evidence_digest"]),
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertTrue(artifact["diagnostics_only"])
        self.assertFalse(artifact["promotion_authorized"])
        self.assertFalse(artifact["runtime_promotion_allowed"])
        self.assertTrue(result["artifact"]["roundtrip_load_passed"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(written["classification"], result["classification"])
        self.assertEqual(
            result["classification"]["primary"],
            "m3_safe_archive_diagnostic_failed_non_promotional",
        )

    def test_carrion_specific_archive_train_eval_writes_route_metadata(self) -> None:
        report, rows, evidence = _safe_train_eval_fixture()
        report = deepcopy(report)
        report["classification"] = {
            "primary": "carrion_specific_archive_support_ready_no_training",
            "labels": ["carrion_specific_archive_support_ready_no_training"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "carrion-report.json"
            dataset_path = root / "carrion-dataset.jsonl"
            evidence_path = root / "carrion-evidence.json"
            artifact_path = root / "carrion-artifact.json"
            output_path = root / "carrion-train-eval-report.json"
            write_safe_archive_expansion_report(report, report_path)
            write_safe_archive_expansion_dataset(rows, dataset_path)
            write_safe_archive_expansion_branch_evidence(evidence, evidence_path)

            result = run_safe_archive_train_eval(
                safe_archive_report_path=report_path,
                safe_archive_dataset_path=dataset_path,
                branch_evidence_report_path=evidence_path,
                artifact_output_path=artifact_path,
                output_path=output_path,
                run_evaluation=False,
                expected_report_classification=(
                    "carrion_specific_archive_support_ready_no_training"
                ),
                leakage_strict_heldout_seeds=(13, 19, 29, 37, 41, 43),
                expected_safe_label_count=2,
                expected_min_safe_label_count=2,
                expected_dataset_digest=stable_payload_digest(rows),
                expected_branch_evidence_digest=str(evidence["branch_evidence_digest"]),
                candidate_id="carrion_specific_archive_support_gated_residual",
                support_mode="carrion_specific_archive_support",
                training_source_label="carrion_specific_archive",
                report_schema_version=(
                    "m3_carrion_specific_archive_train_eval_report_v1"
                ),
                report_policy=(
                    "diagnostics_only_m3_carrion_specific_archive_"
                    "support_gated_train_eval_v1"
                ),
                require_live_carrion_override=True,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(
            result["schema_version"],
            "m3_carrion_specific_archive_train_eval_report_v1",
        )
        self.assertEqual(
            result["policy"],
            "diagnostics_only_m3_carrion_specific_archive_"
            "support_gated_train_eval_v1",
        )
        self.assertEqual(
            artifact["training_policy"],
            "carrion_specific_archive_support_gated_residual_"
            "diagnostic_training_from_carrion_specific_archive_v1",
        )
        self.assertEqual(
            {example["mode"] for example in artifact["support_examples"]},
            {"carrion_specific_archive_support"},
        )
        self.assertTrue(result["inputs"]["require_live_carrion_override"])
        self.assertEqual(
            result["validation"]["trainable_leakage_scan"]["strict_heldout_seeds"],
            [13, 19, 29, 37, 41, 43],
        )
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["promotion_authorized"])

    def test_safe_archive_train_eval_diagnostic_failure_remains_non_promotional(
        self,
    ) -> None:
        acceptance = build_safe_archive_train_eval_acceptance(
            broad=_diagnostic_comparison(
                fixture="broad",
                seed=5,
                alive_delta=-1,
                births_delta=0,
                dominant_share=1.0,
                resolved_invalid_delta=1,
            ),
            carrion=_diagnostic_comparison(
                fixture="carrion_only",
                seed=13,
                alive_delta=0,
                births_delta=0,
                dominant_share=1.0,
                fixture_gate_passed=False,
                unsupported_requested_delta=1,
            ),
            preflight={
                "broad_seed_41_absent_from_archive_support": True,
                "carrion_seed_13_safe_label_absent": True,
            },
        )

        self.assertFalse(acceptance["passed"])
        self.assertEqual(acceptance["first_failing_seed"], 5)
        self.assertEqual(acceptance["first_failing_fixture"], "broad")
        self.assertIn("broad_seed_alive_regression", [
            blocker["reason"] for blocker in acceptance["blockers"]
        ])
        self.assertIn("broad_seed_resolved_invalid_increase", [
            blocker["reason"] for blocker in acceptance["blockers"]
        ])
        self.assertIn("carrion_seed_unsupported_requested_increase", [
            blocker["reason"] for blocker in acceptance["blockers"]
        ])

    def test_safe_archive_train_eval_can_require_live_carrion_override(self) -> None:
        acceptance = build_safe_archive_train_eval_acceptance(
            broad=_diagnostic_comparison(
                fixture="broad",
                seed=5,
                alive_delta=0,
                births_delta=0,
                dominant_share=0.4,
            ),
            carrion=_diagnostic_comparison(
                fixture="carrion_only",
                seed=13,
                alive_delta=0,
                births_delta=0,
                dominant_share=0.4,
                fixture_gate_passed=True,
                applied_override_count=0,
            ),
            preflight={
                "broad_seed_41_absent_from_archive_support": True,
                "carrion_seed_13_safe_label_absent": False,
            },
            require_live_carrion_override=True,
        )

        self.assertFalse(acceptance["passed"])
        self.assertIn("carrion_live_applied_override_count_zero", [
            blocker["reason"] for blocker in acceptance["blockers"]
        ])
        self.assertEqual(
            acceptance["metrics"]["carrion_applied_override_count"],
            0,
        )
        self.assertTrue(acceptance["metrics"]["require_live_carrion_override"])

    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:v3:candidate-campaign"]
        expansion_script = package["scripts"]["sim:mind:v3:safe-archive-expansion"]
        train_eval_script = package["scripts"]["sim:mind:v3:safe-archive-train-eval"]
        autopsy_script = package["scripts"][
            "sim:mind:v3:safe-archive-failure-autopsy"
        ]
        sequence_context_script = package["scripts"][
            "sim:mind:v3:safe-archive-sequence-context-audit"
        ]
        carrion_specific_train_eval_script = package["scripts"][
            "sim:mind:v3:carrion-specific-archive-train-eval"
        ]

        self.assertIn("evolution_sim.cli.mind_v3_candidate_campaign", script)
        self.assertIn(
            "evolution_sim.cli.mind_v3_safe_archive_expansion",
            expansion_script,
        )
        self.assertIn(
            "evolution_sim.cli.mind_v3_safe_archive_train_eval",
            train_eval_script,
        )
        self.assertIn(
            "evolution_sim.cli.mind_v3_safe_archive_failure_autopsy",
            autopsy_script,
        )
        self.assertIn(
            "evolution_sim.cli.mind_v3_safe_archive_sequence_context_audit",
            sequence_context_script,
        )
        self.assertIn(
            "evolution_sim.cli.mind_v3_carrion_specific_archive_train_eval",
            carrion_specific_train_eval_script,
        )

    def test_safe_archive_failure_autopsy_contract_is_no_training(self) -> None:
        artifact = _autopsy_artifact()
        train_eval_report = _autopsy_train_eval_report()
        dataset_rows = [_safe_expansion_dataset_row_for_autopsy()]
        rerun_cases = [
            {
                "fixture": "broad",
                "seed": 19,
                "ticks": 120,
                "seed_delta": {
                    "alive_delta": -3,
                    "births_delta": -4,
                    "resolved_invalid_action_count_delta": -2,
                },
                "applied_override_count": 1,
                "applied_override_action_counts": {"move_east": 1},
                "override_traces": [
                    {
                        "fixture": "broad",
                        "seed": 19,
                        "tick": 0,
                        "agent_id": 3,
                        "linear_action": "move_west",
                        "selected_support_action": "move_east",
                        "final_requested_action": "move_east",
                        "final_resolved_action": "move_east",
                        "override_applied": True,
                        "nearest_support_distance": 0.0,
                        "score_margin": 0.5,
                        "support_example_index": 0,
                        "joined_support_row_metadata": {
                            "source_fixture": "broad",
                            "source_seed": 5,
                            "branch_id": "branch-0",
                            "label_action": "move_east",
                        },
                        "action_mask_legality": {
                            "selected_support_action_legal": True,
                            "final_requested_action_valid": True,
                            "final_resolved_action_valid": True,
                        },
                        "resolved_invalid_contribution": 0,
                        "seed_resolved_invalid_delta": -2,
                        "failed_seed_fixture": True,
                    }
                ],
            }
        ]

        report = build_safe_archive_failure_autopsy_report(
            artifact=artifact,
            train_eval_report=train_eval_report,
            dataset_rows=dataset_rows,
            branch_evidence={"branch_evidence_digest": "digest"},
            failure_cases=(("broad", 19),),
            rerun_cases=rerun_cases,
        )

        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertEqual(
            report["classification"]["primary"],
            "m3_safe_archive_failure_autopsy_complete_no_training",
        )
        self.assertEqual(
            report["recommended_next_route"],
            "build_sequence_or_rollout_context_archive",
        )
        self.assertTrue(
            report["failure_cause_assessment"]["runtime_interaction_effects"][
                "present"
            ]
        )
        self.assertTrue(report["inputs"]["input_validation"]["passed"])

    def test_safe_archive_failure_autopsy_rejects_promotional_train_eval_report(
        self,
    ) -> None:
        train_eval_report = _autopsy_train_eval_report()
        train_eval_report["promotion_authorized"] = True

        with self.assertRaises(ValueError) as context:
            build_safe_archive_failure_autopsy_report(
                artifact=_autopsy_artifact(),
                train_eval_report=train_eval_report,
                dataset_rows=[_safe_expansion_dataset_row_for_autopsy()],
                branch_evidence={"branch_evidence_digest": "digest"},
                failure_cases=(("broad", 19),),
                rerun_cases=[],
            )

        self.assertIn(
            "train_eval_report_promotion_authorized_not_false",
            str(context.exception),
        )

    def test_safe_archive_failure_autopsy_rejects_digest_mismatch(self) -> None:
        train_eval_report = _autopsy_train_eval_report()
        train_eval_report["validation"]["dataset_digest"] = "wrong-digest"

        with self.assertRaises(ValueError) as context:
            build_safe_archive_failure_autopsy_report(
                artifact=_autopsy_artifact(),
                train_eval_report=train_eval_report,
                dataset_rows=[_safe_expansion_dataset_row_for_autopsy()],
                branch_evidence={"branch_evidence_digest": "digest"},
                failure_cases=(("broad", 19),),
                rerun_cases=[],
            )

        self.assertIn("dataset_digest_mismatch", str(context.exception))

    def test_safe_archive_sequence_context_audit_contract_is_no_training(
        self,
    ) -> None:
        autopsy, train_eval_report, dataset_rows, branch_evidence = (
            _sequence_context_audit_inputs()
        )

        report = build_safe_archive_sequence_context_audit_report(
            autopsy_report=autopsy,
            train_eval_report=train_eval_report,
            dataset_rows=dataset_rows,
            branch_evidence=branch_evidence,
            failure_cases=(("broad", 19),),
            input_paths={
                "autopsy_report": "custom/autopsy.json",
                "train_eval_report": "custom/train-eval.json",
                "dataset": "custom/dataset.jsonl",
                "branch_evidence": "custom/branch-evidence.json",
            },
        )

        self.assertTrue(report["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertEqual(
            report["classification"]["primary"],
            "m3_safe_archive_sequence_context_audit_complete_no_training",
        )
        self.assertTrue(report["input_validation"]["passed"])
        self.assertTrue(report["leakage_scan"]["passed"])
        comparison = report["action_only_vs_sequence_support_comparison"]
        self.assertEqual(comparison["action_only"]["zero_distance_alias_count"], 1)
        self.assertEqual(
            comparison["sequence_context"][
                "current_override_previous_context_available_count"
            ],
            0,
        )
        self.assertEqual(
            comparison["sequence_context"]["primary_limitation"],
            "no_finalized_prior_public_sequence_context_available",
        )
        self.assertEqual(
            report["recommended_next_route"],
            "stop_bp3_action_only_support_gated_residual_family",
        )
        self.assertEqual(
            report["route_decision"]["secondary"],
            "collect_public_sequence_context_branch_evidence",
        )
        self.assertEqual(report["inputs"]["autopsy_report"], "custom/autopsy.json")

    def test_safe_archive_sequence_context_audit_rejects_promotional_inputs(
        self,
    ) -> None:
        autopsy, train_eval_report, dataset_rows, branch_evidence = (
            _sequence_context_audit_inputs()
        )
        train_eval_report["promotion_authorized"] = True

        with self.assertRaises(ValueError) as context:
            build_safe_archive_sequence_context_audit_report(
                autopsy_report=autopsy,
                train_eval_report=train_eval_report,
                dataset_rows=dataset_rows,
                branch_evidence=branch_evidence,
                failure_cases=(("broad", 19),),
            )

        self.assertIn(
            "train_eval_report_promotion_authorized_not_false",
            str(context.exception),
        )

    def test_safe_archive_sequence_context_audit_rejects_digest_mismatch(
        self,
    ) -> None:
        autopsy, train_eval_report, dataset_rows, branch_evidence = (
            _sequence_context_audit_inputs()
        )
        autopsy["inputs"]["dataset_digest"] = "wrong-digest"

        with self.assertRaises(ValueError) as context:
            build_safe_archive_sequence_context_audit_report(
                autopsy_report=autopsy,
                train_eval_report=train_eval_report,
                dataset_rows=dataset_rows,
                branch_evidence=branch_evidence,
                failure_cases=(("broad", 19),),
            )

        self.assertIn("autopsy_dataset_digest_mismatch", str(context.exception))

    def test_safe_archive_sequence_context_audit_rejects_missing_branch_status(
        self,
    ) -> None:
        autopsy, train_eval_report, dataset_rows, branch_evidence = (
            _sequence_context_audit_inputs()
        )
        branch_evidence.pop("generation_status")

        with self.assertRaises(ValueError) as context:
            build_safe_archive_sequence_context_audit_report(
                autopsy_report=autopsy,
                train_eval_report=train_eval_report,
                dataset_rows=dataset_rows,
                branch_evidence=branch_evidence,
                failure_cases=(("broad", 19),),
            )

        self.assertIn("branch_evidence_not_complete", str(context.exception))

    def test_safe_archive_sequence_context_leakage_scan_rejects_private_fields(
        self,
    ) -> None:
        allowed = sequence_context_trainable_leakage_scan(
            [
                {
                    "features": {
                        "previous_public_transition_summaries": [
                            {
                                "previous_public_action": "stay",
                                "public_outcome_summary": {
                                    "resolved_action": "stay",
                                    "action_valid": True,
                                },
                            }
                        ]
                    }
                }
            ]
        )
        leaking = sequence_context_trainable_leakage_scan(
            [
                {
                    "features": {
                        "seed": 13,
                        "future_outcome": {"alive_agents": 4},
                        "branch_id": "m3-safe-archive-branch-0",
                    }
                }
            ]
        )

        self.assertTrue(allowed["passed"])
        self.assertFalse(leaking["passed"])
        self.assertGreaterEqual(leaking["forbidden_failure_count"], 3)

    def test_safe_archive_sequence_context_audit_reports_sequence_separation(
        self,
    ) -> None:
        support_history = [
            {
                "previous_public_action": "stay",
                "public_outcome_summary": {
                    "resolved_action": "stay",
                    "action_valid": True,
                },
            }
        ]
        override_history = [
            {
                "previous_public_action": "eat",
                "public_outcome_summary": {
                    "resolved_action": "eat",
                    "action_valid": True,
                },
            }
        ]
        autopsy, train_eval_report, dataset_rows, branch_evidence = (
            _sequence_context_audit_inputs(
                support_previous=support_history,
                override_previous=override_history,
            )
        )

        report = build_safe_archive_sequence_context_audit_report(
            autopsy_report=autopsy,
            train_eval_report=train_eval_report,
            dataset_rows=dataset_rows,
            branch_evidence=branch_evidence,
            failure_cases=(("broad", 19),),
        )

        comparison = report["action_only_vs_sequence_support_comparison"]
        self.assertEqual(
            comparison["sequence_context"]["separated_alias_count"],
            1,
        )
        self.assertTrue(
            comparison["sequence_context"]["all_action_only_aliases_separated"]
        )
        self.assertEqual(
            report["recommended_next_route"],
            "build_sequence_or_rollout_context_archive",
        )


def _safe_train_eval_fixture() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    branch_results = [
        _safe_expansion_branch_result(seed=5, fixture="broad", action="stay"),
        _safe_expansion_branch_result(seed=19, fixture="carrion_only", action="drink"),
    ]
    evidence = _safe_archive_shard_evidence(branch_results, shard_id="bp3-test")
    report, rows = build_safe_archive_expansion_report(
        branch_results=branch_results,
        v145_report={"route_decision": {"label_blacklist": []}},
        min_safe_label_count=2,
        branch_evidence_status=evidence["generation_status"],
    )
    return report, rows, evidence


def _autopsy_artifact() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_v144_branch_intervention_residual_artifact_v1",
        "policy": "mind_v3_v144_branch_intervention_residual_runtime_v1",
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "training_policy": "test",
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "default_runtime_behavior_changed": False,
        "action_prior_penalty_scale": 2.0,
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
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
        "support_action_counts": {"move_east": 1},
        "teacher_action_counts": {"move_east": 1},
        "support_examples": [
            {
                "example_index": 0,
                "action": "move_east",
                "mode": "test",
                "weight": 1.0,
                "feature_vector": [0.0, 1.0],
            }
        ],
    }


def _autopsy_train_eval_report() -> dict[str, object]:
    artifact = _autopsy_artifact()
    rows = [_safe_expansion_dataset_row_for_autopsy()]
    return {
        "classification": {
            "primary": "m3_safe_archive_diagnostic_failed_non_promotional"
        },
        "diagnostics_only": True,
        "training_authorized": False,
        "promotion_authorized": False,
        "runtime_promotion_allowed": False,
        "default_runtime_behavior_changed": False,
        "non_promoted": True,
        "inputs": {
            "expected_dataset_digest": stable_payload_digest(rows),
            "expected_branch_evidence_digest": "digest",
        },
        "validation": {
            "passed": True,
            "failures": [],
            "dataset_digest": stable_payload_digest(rows),
            "branch_evidence_digest": "digest",
        },
        "artifact": {
            "digest": stable_payload_digest(artifact),
            "training_authorized": False,
            "promotion_authorized": False,
            "runtime_promotion_allowed": False,
        },
        "acceptance": {
            "passed": False,
            "blockers": [
                {
                    "reason": "broad_seed_alive_regression",
                    "fixture": "broad",
                    "seed": 19,
                    "observed": -3,
                }
            ],
            "metrics": {
                "broad_per_seed_delta": [
                    {
                        "seed": 19,
                        "alive_delta": -3,
                        "births_delta": -4,
                        "resolved_invalid_action_count_delta": -2,
                    }
                ],
                "carrion_per_seed_delta": [],
            },
        },
        "preflight": {"zero_support_strict_seeds": []},
    }


def _safe_expansion_dataset_row_for_autopsy() -> dict[str, object]:
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_DATASET_ROW_SCHEMA_VERSION,
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {"observation_input": {}, "action_mask": {}},
            "label": {
                "action": "move_east",
                "label_policy": "test",
            },
        },
        "metadata": {
            "row_index": 0,
            "fixture": "broad",
            "seed": 5,
            "branch_id": "branch-0",
        },
    }


def _sequence_context_audit_inputs(
    *,
    support_previous: list[dict[str, object]] | None = None,
    override_previous: list[dict[str, object]] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    dataset_row = _safe_expansion_dataset_row_for_autopsy()
    if support_previous is not None:
        dataset_row["trainable"]["features"][
            "previous_public_transition_summaries"
        ] = support_previous
    dataset_rows = [dataset_row]
    train_eval_report = _autopsy_train_eval_report()
    dataset_digest = stable_payload_digest(dataset_rows)
    train_eval_report["inputs"]["expected_dataset_digest"] = dataset_digest
    train_eval_report["validation"]["dataset_digest"] = dataset_digest
    branch_evidence = _sequence_context_branch_evidence()
    trace = {
        "fixture": "broad",
        "seed": 19,
        "tick": 0,
        "agent_id": 3,
        "linear_action": "move_west",
        "selected_support_action": "move_east",
        "final_requested_action": "move_east",
        "final_resolved_action": "move_east",
        "override_applied": True,
        "nearest_support_distance": 0.0,
        "score_margin": 0.5,
        "support_example_index": 0,
        "joined_support_row_metadata": {
            "source_fixture": "broad",
            "source_seed": 5,
            "branch_id": "branch-0",
            "label_action": "move_east",
        },
        "action_mask_legality": {
            "selected_support_action_legal": True,
            "final_requested_action_valid": True,
            "final_resolved_action_valid": True,
        },
        "resolved_invalid_contribution": 0,
        "seed_resolved_invalid_delta": -2,
        "failed_seed_fixture": True,
    }
    if override_previous is not None:
        trace["current_public_sequence_context"] = {
            "previous_public_transition_summaries": override_previous,
        }
    autopsy = build_safe_archive_failure_autopsy_report(
        artifact=_autopsy_artifact(),
        train_eval_report=train_eval_report,
        dataset_rows=dataset_rows,
        branch_evidence=branch_evidence,
        failure_cases=(("broad", 19),),
        rerun_cases=[
            {
                "fixture": "broad",
                "seed": 19,
                "ticks": 120,
                "seed_delta": {
                    "alive_delta": -3,
                    "births_delta": -4,
                    "resolved_invalid_action_count_delta": -2,
                },
                "applied_override_count": 1,
                "applied_override_action_counts": {"move_east": 1},
                "override_traces": [trace],
            }
        ],
    )
    return autopsy, train_eval_report, dataset_rows, branch_evidence


def _sequence_context_branch_evidence() -> dict[str, object]:
    return {
        "branch_evidence_digest": "digest",
        "training_authorized": False,
        "promotion_authorized": False,
        "non_promoted": True,
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "gate_relaxation": False,
        },
        "generation_status": {
            "state": "complete",
            "partial": False,
        },
        "source_integrity": {
            "passed": True,
            "failures": [],
        },
        "branch_results": [],
    }


def _diagnostic_comparison(
    *,
    fixture: str,
    seed: int,
    alive_delta: int,
    births_delta: int,
    dominant_share: float,
    fixture_gate_passed: bool = True,
    resolved_invalid_delta: int = 0,
    unsupported_requested_delta: int = 0,
    applied_override_count: int = 0,
) -> dict[str, object]:
    support_diag = {
        "applied_override_count": applied_override_count,
        "applied_override_action_counts": (
            {"stay": applied_override_count} if applied_override_count else {}
        ),
    }
    candidate_aggregate = {
        "alive_agents_mean": 10 + alive_delta,
        "births_mean": 2 + births_delta,
        "requested_action_counts": {"stay": 10},
        "action_source_counts": {"mind_v3_autonomous": 10},
        "dominant_requested_action_share": dominant_share,
        "heuristic_action_source_count": 0,
        "support_residual_diagnostics": support_diag,
    }
    baseline_aggregate = {
        "alive_agents_mean": 10,
        "births_mean": 2,
        "requested_action_counts": {"stay": 10},
        "action_source_counts": {"mind_v3_autonomous": 10},
        "dominant_requested_action_share": dominant_share,
        "heuristic_action_source_count": 0,
        "support_residual_diagnostics": {},
    }
    payload = {
        "fixture": fixture,
        "baseline": {"aggregate": baseline_aggregate, "runs": []},
        "candidate": {"aggregate": candidate_aggregate, "runs": []},
        "aggregate_delta": {
            "alive_agents_mean": alive_delta,
            "births_mean": births_delta,
            "resolved_invalid_action_count_delta": resolved_invalid_delta,
            "unsupported_requested_action_count_delta": unsupported_requested_delta,
        },
        "per_seed_delta": [
            {
                "seed": seed,
                "alive_delta": alive_delta,
                "births_delta": births_delta,
                "resolved_invalid_action_count_delta": resolved_invalid_delta,
                "unsupported_requested_action_count_delta": unsupported_requested_delta,
            }
        ],
    }
    if fixture == "carrion_only":
        payload["candidate_fixture_gate"] = {
            "passed": fixture_gate_passed,
            "blockers": [] if fixture_gate_passed else [{"reason": "blocked"}],
        }
    return payload


def _dataset_row(
    *,
    seed: int,
    agent_id: int,
    action: str,
    unsupported_resolved: int,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_V143_BRANCH_INTERVENTION_DATASET_ROW_SCHEMA_VERSION,
        "trainable": {
            "feature_policy": "public_observation_input_and_public_action_mask_v1",
            "features": {"observation_input": {}, "action_mask": {}},
            "label": {
                "action": action,
                "label_policy": "best_supported_terminal_alive_or_birth_improvement_v1",
            },
        },
        "metadata": {
            "seed": seed,
            "fixture": "broad",
            "branch_id": f"v143-broad-seed-{seed}-branch-0-tick-0-agent-{agent_id}",
            "branch_tick": 0,
            "agent_id": agent_id,
            "outcome_evidence": {
                "deltas_vs_baseline": {
                    "alive_agents": 1,
                    "births": 1,
                    "unsupported_requested_action_count": 0,
                    "unsupported_resolved_action_count": unsupported_resolved,
                }
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


def _v145_report() -> dict[str, object]:
    return {
        "route_decision": {
            "label_blacklist": [
                {
                    "seed": 5,
                    "agent_id": 6,
                    "tick": 0,
                    "residual_action": "stay",
                    "branch_id": "v143-broad-seed-5-branch-0-tick-0-agent-6",
                    "label_source_row_index": 0,
                }
            ]
        }
    }


def _safe_expansion_branch_result(
    *,
    seed: int,
    fixture: str,
    action: str,
    branch_id: str | None = None,
    agent_id: int = 4,
) -> dict[str, object]:
    branch = branch_id or f"m3-{fixture}-seed-{seed}-branch-0-tick-0-agent-{agent_id}"
    action_mask = {"stay": True, "eat": True, "drink": True}
    action_runs = [
        _safe_expansion_action_run(
            branch_id=branch,
            seed=seed,
            fixture=fixture,
            action=candidate,
            alive_delta=1 if candidate == action else 0,
            births_delta=1 if candidate == action else 0,
        )
        for candidate in ("stay", "eat", "drink")
    ]
    return {
        "branch_id": branch,
        "seed": seed,
        "fixture": fixture,
        "ticks": 120,
        "branch_tick": 0,
        "record_index": 0,
        "branch_index": 0,
        "agent_id": agent_id,
        "baseline_action": "stay",
        "v142_requested_action": "stay",
        "v142_resolved_action": "stay",
        "source_trajectory_path": f"private/{fixture}-{seed}.jsonl.gz",
        "branch_state_digest": f"digest-{fixture}-{seed}",
        "candidate_actions": ["stay", "eat", "drink"],
        "candidate_action_count": 3,
        "public_features": {
            "observation_input": _encoded_zero_observation_input(),
            "action_mask": action_mask,
        },
        "action_runs": action_runs,
    }


def _encoded_zero_observation_input() -> dict[str, object]:
    packed = struct.pack(
        f"<{OBSERVATION_INPUT_VECTOR_SIZE}h",
        *([0] * OBSERVATION_INPUT_VECTOR_SIZE),
    )
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


def _rewrite_branch_identity(
    result: dict[str, object],
    *,
    branch_id: str,
    branch_index: int,
) -> dict[str, object]:
    updated = deepcopy(result)
    updated["branch_id"] = branch_id
    updated["branch_index"] = branch_index
    for action_run in updated["action_runs"]:
        action_run["branch_id"] = branch_id
        action = action_run["forced_action"]
        action_run["replay_digest"] = f"replay-{branch_id}-{action}"
        action_run["replay_verification"] = {
            "verified": True,
            "expected_digest": f"replay-{branch_id}-{action}",
            "actual_digest": f"replay-{branch_id}-{action}",
        }
    return updated


def _safe_archive_shard_evidence(
    branch_results: list[dict[str, object]],
    *,
    shard_id: str,
    partial: bool = False,
    source_integrity_failures: tuple[str, ...] = (),
) -> dict[str, object]:
    failures = list(source_integrity_failures)
    if partial:
        failures.append("partial_branch_evidence")
    return {
        "schema_version": M3_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_SCHEMA_VERSION,
        "policy": f"{M3_SAFE_ARCHIVE_EXPANSION_POLICY}_branch_evidence_v1",
        "contract": {
            "diagnostics_only": True,
            "training_authorized": False,
            "promotion_authorized": False,
            "default_runtime_behavior_changed": False,
            "runtime_action_selection_changed": False,
            "heuristic_fallback_added": False,
            "gate_relaxation": False,
            "branch_replay_policy": (
                "deterministic_public_mask_valid_action_forced_first_action_v1"
            ),
            "candidate_action_policy": (
                "all_currently_valid_public_action_mask_actions_when_"
                "max_candidate_actions_is_zero"
            ),
            "max_candidate_actions": None,
        },
        "inputs": {
            "v142_scorer_report": "v142-scorer.json",
            "v142_live_report": "v142-live.json",
            "v142_trajectory_output_dir": "v142-trajectories",
            "carrion_ticks": 120,
            "max_branch_points_per_seed": 2,
            "verify_replay": True,
            "shard_id": shard_id,
        },
        "generation_status": {
            "policy": "m3_safe_archive_expansion_branch_evidence_generation_status_v1",
            "state": "partial" if partial else "complete",
            "partial": partial,
            "stop_reason": "test_partial" if partial else None,
        },
        "source_integrity": {
            "policy": "m3_safe_archive_expansion_branch_evidence_source_integrity_v1",
            "passed": not failures,
            "failures": failures,
            "first_missing_valid_action": None,
        },
        "branch_results": branch_results,
        "branch_result_count": len(branch_results),
        "branch_evidence_digest": stable_payload_digest(branch_results),
        "training_authorized": False,
        "promotion_authorized": False,
        "non_default_runtime": True,
        "non_promoted": True,
    }


def _branch_point_for_safe_expansion_result(
    result: dict[str, object],
) -> SimpleNamespace:
    public_features = result["public_features"]
    assert isinstance(public_features, dict)
    return SimpleNamespace(
        branch_id=result["branch_id"],
        seed=result["seed"],
        fixture=result["fixture"],
        ticks=result["ticks"],
        branch_tick=result["branch_tick"],
        record_index=result["record_index"],
        branch_index=result["branch_index"],
        agent_id=result["agent_id"],
        baseline_action=result["baseline_action"],
        v142_requested_action=result["v142_requested_action"],
        v142_resolved_action=result["v142_resolved_action"],
        action_mask=public_features["action_mask"],
        observation_input=public_features["observation_input"],
        branch_state_digest=result["branch_state_digest"],
    )


def _safe_expansion_action_run(
    *,
    branch_id: str,
    seed: int,
    fixture: str,
    action: str,
    alive_delta: int,
    births_delta: int,
) -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "seed": seed,
        "fixture": fixture,
        "forced_action": action,
        "forced_action_used": True,
        "forced_action_supported": True,
        "alive_agents": 10 + alive_delta,
        "births": 2 + births_delta,
        "deaths": 0,
        "target_terminal": {
            "alive": True,
            "energy_ratio": 0.5,
            "hydration_ratio": 0.6,
            "health_ratio": 0.8,
        },
        "unsupported_requested_action_count": 0,
        "heuristic_action_source_count": 0,
        "deltas_vs_baseline": {
            "alive_agents": alive_delta,
            "births": births_delta,
            "deaths": 0,
            "target_alive": 1,
            "unsupported_resolved_action_count": 0,
        },
        "deltas_vs_v142_override": {
            "alive_agents": alive_delta,
            "births": births_delta,
            "deaths": 0,
            "target_alive": 1,
            "unsupported_resolved_action_count": 0,
        },
        "replay_digest": f"replay-{branch_id}-{action}",
        "replay_verification": {
            "verified": True,
            "expected_digest": f"replay-{branch_id}-{action}",
            "actual_digest": f"replay-{branch_id}-{action}",
        },
    }


def _result(
    candidate_id: str,
    *,
    accepted: bool,
    failed: int,
    resolved: int,
    order: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "accepted": accepted,
        "skipped": False,
        "failed_floor_count": failed,
        "first_failed_floor": None if accepted else "floor",
        "first_failing_seed": None,
        "first_failing_fixture": None,
        "order": order,
        "ranking_metrics": {
            "accepted_rank_value": 1 if accepted else 0,
            "failed_floor_count": failed,
            "broad_per_seed_regression_count": 0,
            "carrion_improved": False,
            "resolved_invalid_delta": resolved,
            "dominant_requested_action_share": 0.4,
        },
    }


if __name__ == "__main__":
    unittest.main()
