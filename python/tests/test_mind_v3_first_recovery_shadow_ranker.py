from __future__ import annotations

import base64
import gzip
import json
import os
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

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
from evolution_sim.mind.first_recovery_shadow_ranker import (
    ALLOWED_CLASSIFICATION_LABELS,
    MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
    build_first_recovery_shadow_ranker,
    build_pairwise_examples,
    current_row_feature_names,
    feature_name_leakage,
    group_archive_rows_by_branch_target,
    history_usefulness_answer,
    load_first_recovery_archive_rows,
    model_scores_for_group,
    selected_action_distribution_answer,
    train_pairwise_linear_ranker,
    write_first_recovery_shadow_ranker_report,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryShadowRankerTests(unittest.TestCase):
    def test_first_recovery_shadow_ranker_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"]["sim:mind:v3:first-recovery-shadow-ranker"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_shadow_ranker"
            ),
        )

    def test_archive_jsonl_loading_is_deterministic(self) -> None:
        rows = _archive_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl.gz"
            _write_rows(path, rows)
            first = load_first_recovery_archive_rows(path)
            second = load_first_recovery_archive_rows(path)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(rows))

    def test_branch_target_grouping_is_by_provenance_branch_id(self) -> None:
        groups = group_archive_rows_by_branch_target(_archive_rows())

        self.assertEqual(len(groups), 4)
        self.assertTrue(all(group.group_key.startswith("branch-") for group in groups))
        self.assertEqual({len(group.rows) for group in groups}, {3})

    def test_pairwise_examples_are_same_target_only_and_rank_one_is_best(self) -> None:
        groups = group_archive_rows_by_branch_target(_archive_rows())
        examples = build_pairwise_examples(groups)

        self.assertTrue(examples)
        self.assertTrue(
            all(example["better_oracle_rank"] < example["worse_oracle_rank"] for example in examples)
        )
        self.assertTrue(
            all(str(example["better_archive_row_id"]).split("::")[0] == example["target_key"] for example in examples)
        )
        self.assertTrue(
            all(str(example["worse_archive_row_id"]).split("::")[0] == example["target_key"] for example in examples)
        )

    def test_linear_scorer_output_is_deterministic(self) -> None:
        groups = group_archive_rows_by_branch_target(_archive_rows(best_action="drink"))
        first = train_pairwise_linear_ranker(groups)
        second = train_pairwise_linear_ranker(groups)
        ranking = model_scores_for_group(first, groups[0])

        self.assertEqual(first.weights, second.weights)
        self.assertEqual(ranking[0]["candidate_action"], "drink")

    def test_no_provenance_private_or_logged_fields_enter_features(self) -> None:
        forbidden = {
            "seed",
            "source_path",
            "source_kind",
            "fixture",
            "branch_id",
            "agent_id",
            "record_index",
            "logged_action",
            "private_world_state",
        }
        tokens = {
            token
            for name in current_row_feature_names()
            for token in name.replace("=", ".").split(".")
        }
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
        )

        self.assertTrue(tokens.isdisjoint(forbidden))
        self.assertEqual(build.report["leakage_audit"]["answer"], "shadow_ranker_leakage_free")
        self.assertFalse(build.report["contract"]["branch_identity_input"])
        self.assertFalse(build.report["contract"]["logged_action_fallback_input"])

    def test_provenance_is_grouping_only_not_feature_input(self) -> None:
        rows = _archive_rows()
        self.assertIn("seed", rows[0]["provenance"])
        self.assertIn("branch_id", rows[0]["provenance"])
        groups = group_archive_rows_by_branch_target(rows)
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(rows),
            archive_rows=rows,
        )

        self.assertEqual(len(groups), 4)
        self.assertTrue(build.report["leakage_audit"]["branch_id_used_for_grouping_only"])
        self.assertFalse(build.report["leakage_audit"]["provenance_used_for_training_features"])

    def test_unsupported_action_rate_remains_zero_when_invalid_candidate_is_best(self) -> None:
        rows = _archive_rows(best_action="move_north", invalid_best=True)
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(rows),
            archive_rows=rows,
        )

        self.assertEqual(
            build.report["unsupported_action_audit"]["answer"],
            "shadow_ranker_unsupported_action_free",
        )
        self.assertEqual(
            build.report["current_row_linear_ranker"]["all_data_descriptive"]["metrics"][
                "unsupported_action_rate"
            ],
            0.0,
        )

    def test_logged_and_stay_baselines_are_reported(self) -> None:
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
        )
        baselines = build.report["baselines"]["all_data"]

        self.assertIn("logged_action", baselines)
        self.assertIn("stay", baselines)
        self.assertIn("immediate_homeostatic_delta", baselines)
        self.assertIn("weak_best_action_frequency", baselines)

    def test_holdout_evaluations_are_emitted(self) -> None:
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
        )

        self.assertGreater(build.report["leave_one_seed_evaluation"]["split_count"], 0)
        self.assertGreater(build.report["leave_one_source_evaluation"]["split_count"], 0)
        self.assertGreater(build.report["fixture_open_evaluation"]["split_count"], 0)
        self.assertEqual(build.report["seed29_evaluation"]["holdout_seed"], 29)

    def test_selected_action_share_above_half_classifies_collapsed(self) -> None:
        answer = selected_action_distribution_answer({"stay": 3, "drink": 1})

        self.assertEqual(
            answer["answer"],
            "shadow_ranker_action_distribution_collapsed",
        )

    def test_history_cannot_be_useful_unless_it_beats_current_row(self) -> None:
        answer = history_usefulness_answer(
            current_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.5,
            },
            history_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.75,
            },
        )

        self.assertEqual(answer, "shadow_ranker_history_no_improvement")

        split_only_answer = history_usefulness_answer(
            current_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.5,
            },
            history_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.7,
                "top1_oracle_match_rate": 0.5,
            },
            current_fixture_open_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.5,
            },
            history_fixture_open_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.75,
            },
        )
        both_answer = history_usefulness_answer(
            current_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.5,
            },
            history_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.7,
                "top1_oracle_match_rate": 0.5,
            },
            current_fixture_open_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.6,
                "top1_oracle_match_rate": 0.5,
            },
            history_fixture_open_metrics={
                "evaluable_group_count": 1,
                "mrr": 0.7,
                "top1_oracle_match_rate": 0.5,
            },
        )

        self.assertEqual(split_only_answer, "shadow_ranker_history_no_improvement")
        self.assertEqual(both_answer, "shadow_ranker_history_improves")

    def test_optional_joined_features_without_trajectories_are_inconclusive(self) -> None:
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
            enable_observation_input=True,
            enable_public_history=True,
        )
        labels = set(build.report["classification"]["labels"])

        self.assertEqual(
            build.report["observation_input_ranker"]["answer"],
            "missing_evidence_inconclusive",
        )
        self.assertEqual(
            build.report["public_history_ranker"]["answer"],
            "missing_evidence_inconclusive",
        )
        self.assertIn("trajectory_paths", build.report["classification"]["missing_evidence"])
        self.assertNotIn("shadow_ranker_observation_input_improves", labels)
        self.assertNotIn("shadow_ranker_history_improves", labels)

    def test_lenient_join_reader_ignores_nested_rollout_context_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.jsonl.gz"
            rows = _join_archive_rows(trajectory_path, record_index=3)
            records = [
                _trajectory_record(agent_id=7, tick=1, digest="history-0", observation_index=0),
                _trajectory_record(agent_id=8, tick=2, digest="other-agent", observation_index=1),
                _trajectory_record(agent_id=7, tick=3, digest="history-2", observation_index=2),
                _trajectory_record(
                    agent_id=7,
                    tick=4,
                    digest=str(rows[0]["observation_digest"]),
                    observation_index=3,
                    include_nested_diagnostics=True,
                ),
            ]
            _write_trajectory(trajectory_path, records)

            build = build_first_recovery_shadow_ranker(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                trajectory_paths=(trajectory_path,),
                enable_observation_input=True,
            )

        join = build.report["join_evidence"]
        observation_join = build.report["observation_input_ranker"]["join_evidence"]
        self.assertEqual(join["reader_policy"], "local_lenient_public_trajectory_join_reader_v1")
        self.assertEqual(join["loaded_path_count"], 1)
        self.assertEqual(join["load_failure_count"], 0)
        self.assertEqual(join["malformed_record_count"], 0)
        self.assertEqual(observation_join["matched_archive_row_count"], len(rows))
        self.assertGreater(
            join["ignored_optional_field_occurrences"]["policy_decision_diagnostics"],
            0,
        )
        self.assertNotIn(
            "trajectory_load_failures",
            build.report["classification"]["missing_evidence"],
        )

    def test_joined_observation_matches_exact_public_target_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.jsonl.gz"
            rows = _join_archive_rows(trajectory_path, record_index=2)
            records = [
                _trajectory_record(
                    agent_id=7,
                    tick=4,
                    digest=str(rows[0]["observation_digest"]),
                    observation_index=0,
                ),
                _trajectory_record(agent_id=8, tick=4, digest="wrong-agent", observation_index=1),
                _trajectory_record(
                    agent_id=7,
                    tick=4,
                    digest=str(rows[0]["observation_digest"]),
                    observation_index=2,
                ),
            ]
            _write_trajectory(trajectory_path, records)

            build = build_first_recovery_shadow_ranker(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                trajectory_paths=(trajectory_path,),
                enable_observation_input=True,
            )

        example = build.report["observation_input_ranker"]["join_evidence"]["match_examples"][0]
        self.assertEqual(example["source_path"], str(trajectory_path))
        self.assertEqual(example["record_index"], 2)
        self.assertEqual(example["agent_id"], 7)
        self.assertEqual(example["tick"], 4)
        self.assertEqual(example["observation_digest"], rows[0]["observation_digest"])

    def test_public_history_uses_only_same_agent_prior_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.jsonl.gz"
            rows = _join_archive_rows(trajectory_path, record_index=3)
            records = [
                _trajectory_record(agent_id=7, tick=1, digest="history-0", observation_index=0),
                _trajectory_record(agent_id=8, tick=2, digest="other-agent", observation_index=1),
                _trajectory_record(agent_id=7, tick=3, digest="history-2", observation_index=2),
                _trajectory_record(
                    agent_id=7,
                    tick=4,
                    digest=str(rows[0]["observation_digest"]),
                    observation_index=3,
                ),
                _trajectory_record(agent_id=7, tick=5, digest="future", observation_index=4),
            ]
            _write_trajectory(trajectory_path, records)

            build = build_first_recovery_shadow_ranker(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                trajectory_paths=(trajectory_path,),
                enable_public_history=True,
            )

        example = build.report["public_history_ranker"]["join_evidence"][
            "history_prefix_examples"
        ][0]
        self.assertEqual(example["prefix_record_indices"], [0, 2])
        self.assertEqual(example["prefix_agent_ids"], [7, 7])

    def test_malformed_trajectory_rows_are_inconclusive_not_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "malformed.jsonl"
            rows = _join_archive_rows(trajectory_path, record_index=0)
            trajectory_path.write_text("{not-json}\n", encoding="utf-8")

            build = build_first_recovery_shadow_ranker(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                trajectory_paths=(trajectory_path,),
                enable_observation_input=True,
            )

        self.assertEqual(
            build.report["observation_input_ranker"]["answer"],
            "missing_evidence_inconclusive",
        )
        self.assertIn(
            "trajectory_malformed_records",
            build.report["classification"]["missing_evidence"],
        )
        self.assertEqual(build.report["join_evidence"]["malformed_record_count"], 1)

    def test_joined_feature_name_leakage_rejects_provenance_tokens(self) -> None:
        safe_names = (
            "observation_input.ecological_value_0",
            "public_history.record_count",
            "public_history.failed_movement_rate",
        )
        forbidden_names = (
            "source_path",
            "record_index",
            "agent_id",
            "tick",
            "seed",
            "source_kind",
            "fixture.identity",
            "logged_action",
            "branch_id",
            "private_world_state",
        )

        self.assertEqual(feature_name_leakage(safe_names), ())
        self.assertEqual(set(feature_name_leakage(forbidden_names)), set(forbidden_names))

    def test_json_report_is_byte_stable(self) -> None:
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "shadow.json"
            write_first_recovery_shadow_ranker_report(build, output_path=output)
            first = output.read_bytes()
            write_first_recovery_shadow_ranker_report(build, output_path=output)
            second = output.read_bytes()

        self.assertEqual(first, second)

    def test_cli_writes_deterministic_json_and_prints_summary(self) -> None:
        rows = _archive_rows()
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "archive.json"
            rows_path = Path(tmp) / "archive.jsonl.gz"
            output = Path(tmp) / "shadow.json"
            report.write_text(json.dumps(_archive_report(rows), sort_keys=True), encoding="utf-8")
            _write_rows(rows_path, rows)
            cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_shadow_ranker",
                "--archive-report",
                str(report),
                "--archive-rows",
                str(rows_path),
                "--output",
                str(output),
            ]
            env = {**os.environ, "PYTHONPATH": "python", "PYTHONHASHSEED": "0"}
            first = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            first_bytes = output.read_bytes()
            second = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            second_bytes = output.read_bytes()

        self.assertIn("first_recovery_shadow_ranker=", first.stdout)
        self.assertIn("current_row_answer=", second.stdout)
        self.assertEqual(first_bytes, second_bytes)

    def test_contract_declares_no_runtime_or_artifact_effects(self) -> None:
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
        )
        contract = build.report["contract"]

        self.assertTrue(contract["shadow_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertEqual(contract["replay_golden_effect"], "none")
        self.assertEqual(contract["summary_only_effect"], "none")
        self.assertFalse(contract["private_world_state_input"])
        self.assertFalse(contract["fixture_identity_input"])
        self.assertFalse(contract["seed_identity_input"])
        self.assertFalse(contract["source_identity_input"])
        self.assertFalse(contract["branch_identity_input"])

    def test_taxonomy_sanity_check(self) -> None:
        build = build_first_recovery_shadow_ranker(
            archive_report=_archive_report(_archive_rows()),
            archive_rows=_archive_rows(),
        )
        allowed = set(ALLOWED_CLASSIFICATION_LABELS)
        labels = set(build.report["classification"]["labels"])
        answers = set(build.report["classification"]["diagnostic_answers"].values())

        self.assertTrue(labels.issubset(allowed))
        self.assertTrue(answers.issubset(allowed))


def _archive_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "branch_archive_summary": {
            "archive_row_count": len(rows),
            "selected_row_count": len({row["provenance"]["branch_id"] for row in rows}),
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "zero_heuristic_runtime_actions_except_diagnostic_force": True,
        },
        "row_reconstruction": {"reconstructed_first_recovery_row_count": 106},
        "legality_summary": {
            "answer": "oracle_actions_observation_legal",
            "resolution_answer": "oracle_actions_resolution_legal",
        },
        "learnability_readiness": {
            "trainable_public_input_leakage": {"leak_count": 0}
        },
    }


def _archive_rows(
    *,
    best_action: str = "drink",
    invalid_best: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = [
        ("branch-13-fixture", 13, "fixture_carrion_only", "stay"),
        ("branch-29-fixture", 29, "fixture_carrion_only", "stay"),
        ("branch-41-open", 41, "open_mind_v3", "stay"),
        ("branch-43-open", 43, "open_mind_v3", "stay"),
    ]
    for index, (branch_id, seed, source_kind, logged_action) in enumerate(specs):
        for action, rank in (
            (best_action, 1),
            ("eat", 2 if best_action != "eat" else 1),
            ("stay", 3 if best_action != "stay" else 1),
        ):
            observation_legal = not (invalid_best and action == best_action)
            rows.append(
                _row(
                    branch_id=branch_id,
                    seed=seed,
                    source_kind=source_kind,
                    record_index=10 + index,
                    action=action,
                    oracle_rank=rank,
                    logged_action=logged_action,
                    observation_legal=observation_legal,
                    material_gain=rank == 1 and observation_legal,
                )
            )
    return rows


def _row(
    *,
    branch_id: str,
    seed: int,
    source_kind: str,
    record_index: int,
    action: str,
    oracle_rank: int,
    logged_action: str,
    observation_legal: bool = True,
    material_gain: bool = False,
) -> dict[str, object]:
    action_mask = {name: name in {"stay", "eat", "drink", "move_north"} for name in ACTION_NAMES}
    if not observation_legal:
        action_mask[action] = False
    before = {
        "alive": True,
        "energy_ratio": 0.4,
        "hydration_ratio": 0.2,
        "health_ratio": 0.8,
        "age": 12,
    }
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "archive_row_id": f"{branch_id}::{action}",
        "candidate_action": action,
        "oracle_rank": oracle_rank,
        "oracle_best_action": "drink",
        "observation_legal": observation_legal,
        "resolution_legal": observation_legal,
        "action_mask": dict(action_mask),
        "resolution_action_mask": dict(action_mask),
        "material_gain_label": material_gain,
        "tick": 4,
        "record_index": record_index,
        "observation_digest": f"digest-{branch_id}",
        "recovery_vitals_deltas": {
            "energy_ratio_delta": 0.2 if action == "eat" else 0.0,
            "hydration_ratio_delta": 0.3 if action == "drink" else 0.0,
            "health_ratio_delta": 0.0,
            "target_recovery_score_delta": 0.5 if material_gain else 0.0,
        },
        "trainable_public_input": {
            "schema_version": "mind_v3_first_recovery_branch_archive_trainable_public_input_v1",
            "post_carrion_first_recovery": True,
            "candidate_action": action,
            "candidate_action_index": ACTION_NAMES.index(action),
            "action_mask": dict(action_mask),
            "target_public_state_before": dict(before),
            "public_transition_context": {
                "ticks_after_animal_resource_gain": 1,
                "records_after_animal_resource_gain": 3,
            },
        },
        "provenance": {
            "branch_id": branch_id,
            "seed": seed,
            "source_kind": source_kind,
            "source_path": f"output/mind/synthetic-{source_kind}-{seed}.jsonl.gz",
            "agent_id": 7,
            "record_index": record_index,
            "logged_action": logged_action,
        },
    }


def _join_archive_rows(
    source_path: Path,
    *,
    record_index: int,
) -> list[dict[str, object]]:
    rows = json.loads(json.dumps(_archive_rows()[:3]))
    for row in rows:
        row["tick"] = 4
        row["record_index"] = record_index
        row["observation_digest"] = "joined-digest"
        provenance = row["provenance"]
        provenance["source_path"] = str(source_path)
        provenance["record_index"] = record_index
        provenance["agent_id"] = 7
    return rows


def _trajectory_record(
    *,
    agent_id: int,
    tick: int,
    digest: str,
    observation_index: int,
    include_nested_diagnostics: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {
        "tick": tick,
        "agent_id": agent_id,
        "observation_digest": digest,
        "observation_input": _observation(observation_index),
        "action_mask": {
            action: action in {"stay", "eat", "drink"} for action in ACTION_NAMES
        },
        "requested_action": "stay",
        "resolved_action": "stay",
        "before": {
            "alive": True,
            "energy_ratio": round(0.2 + 0.01 * observation_index, 6),
            "hydration_ratio": round(0.4 + 0.01 * observation_index, 6),
            "health_ratio": round(0.8 - 0.01 * observation_index, 6),
        },
        "after": {
            "alive": True,
            "energy_ratio": round(0.19 + 0.01 * observation_index, 6),
            "hydration_ratio": round(0.39 + 0.01 * observation_index, 6),
            "health_ratio": round(0.8 - 0.01 * observation_index, 6),
        },
        "outcome": {"resource_gain": 0.0},
        "moved": False,
    }
    if include_nested_diagnostics:
        record["policy_decision_diagnostics"] = {
            "rollout_context_score_delta_by_action": {
                "eat": {"nested": "ignored_by_lenient_join_reader"}
            }
        }
        record["policy_update_trace"] = {"nested": ["ignored"]}
    return record


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


def _write_trajectory(path: Path, records: list[dict[str, object]]) -> None:
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
                _write_trajectory_lines(gzip_file, records)
    else:
        with path.open("w", encoding="utf-8") as handle:
            for line in _trajectory_lines(records):
                handle.write(line)
                handle.write("\n")


def _write_trajectory_lines(handle: gzip.GzipFile, records: list[dict[str, object]]) -> None:
    for line in _trajectory_lines(records):
        handle.write(line.encode("utf-8"))
        handle.write(b"\n")


def _trajectory_lines(records: list[dict[str, object]]) -> list[str]:
    payloads = [{"type": "header"}]
    payloads.extend({"type": "record", "record": record} for record in records)
    payloads.append({"type": "footer"})
    return [json.dumps(payload, sort_keys=True) for payload in payloads]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            for row in rows:
                gzip_file.write(json.dumps(row, sort_keys=True).encode("utf-8"))
                gzip_file.write(b"\n")


if __name__ == "__main__":
    unittest.main()
