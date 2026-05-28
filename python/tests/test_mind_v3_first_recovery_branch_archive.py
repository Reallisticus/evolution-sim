from __future__ import annotations

import gzip
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.first_recovery_branch_archive import (
    ALLOWED_CLASSIFICATION_LABELS,
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    build_first_recovery_archive_rows,
    build_first_recovery_branch_archive,
    default_archive_rows_path,
    select_first_recovery_archive_targets,
    trainable_public_input_leakage,
    write_first_recovery_branch_archive_outputs,
)
from evolution_sim.mind.first_recovery_branch_oracle_audit import (
    FIXTURE_SOURCE_KIND,
    OPEN_SOURCE_KIND,
    _source_kind,
)
from evolution_sim.mind.transition_aligned_recovery_audit import (
    reconstruct_transition_aligned_first_recovery_rows,
)

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_GLOB = (
    "output/mind/mind-v3-v5-carrion-only-120-trajectories/*mind-v3*.jsonl.gz"
)
V108_REPORT = Path("output/mind/mind-v3-v108-first-recovery-branch-oracle-audit.json")
V107_REPORT = Path("output/mind/mind-v3-v107-transition-aligned-recovery-audit.json")
ROLLOUT_CONTEXT_REPORT = Path(
    "output/mind/mind-v3-v5-rollout-context-search-80-120-diagnostic.json"
)
BASELINE_REPORT = Path("output/mind/mind-v3-v4-baseline-search-80-120.json")


class MindV3FirstRecoveryBranchArchiveTests(unittest.TestCase):
    def test_first_recovery_branch_archive_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"]["sim:mind:v3:first-recovery-branch-archive"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_branch_archive"
            ),
        )

    def test_reconstructs_106_first_recovery_rows(self) -> None:
        _require_real_first_recovery_artifacts(self)
        paths = _trajectory_paths()
        reconstructed = reconstruct_transition_aligned_first_recovery_rows(
            trajectory_paths=paths,
            trajectory_glob_patterns=(TRAJECTORY_GLOB,),
        )
        self.assertEqual(len(reconstructed["rows"]), 106)

    def test_stratified_selection_is_deterministic_and_includes_open_rows(self) -> None:
        _require_real_first_recovery_artifacts(self)
        reconstructed = _real_reconstruction()
        rows = reconstructed["rows"]
        path_metadata = reconstructed["path_metadata"]

        first = select_first_recovery_archive_targets(rows, path_metadata=path_metadata)
        second = select_first_recovery_archive_targets(rows, path_metadata=path_metadata)

        self.assertEqual(first["selected_targets"], second["selected_targets"])
        self.assertEqual(first["open_candidate_row_count"], 10)
        self.assertEqual(first["open_selected_row_count"], 10)
        self.assertTrue(first["all_open_rows_selected"])
        self.assertEqual(first["fixture_seed_coverage"], [13, 19, 29, 37, 41, 43])
        self.assertTrue(first["seed29_selected"])
        selected_actions = set(first["selected_by_logged_action"])
        all_actions = {str(row["requested_action"]) for row in rows}
        self.assertTrue(all_actions.issubset(selected_actions))

    def test_open_rows_can_be_explicitly_classified_unselected(self) -> None:
        _require_real_first_recovery_artifacts(self)
        reconstructed = _real_reconstruction()
        selection = select_first_recovery_archive_targets(
            reconstructed["rows"],
            path_metadata=reconstructed["path_metadata"],
            include_open=False,
        )
        self.assertEqual(selection["open_candidate_row_count"], 10)
        self.assertEqual(selection["open_selected_row_count"], 0)
        self.assertEqual(
            selection["skip_reason_counts"]["open_rows_skipped_by_option"],
            10,
        )

    def test_archive_rows_record_replay_digest_and_legality(self) -> None:
        rows = build_first_recovery_archive_rows([_synthetic_branch_result()])
        drink = next(row for row in rows if row["candidate_action"] == "drink")

        self.assertEqual(drink["replay_verification_digest"], "digest-drink")
        self.assertTrue(drink["replay_verification_result"])
        self.assertTrue(drink["observation_legal"])
        self.assertTrue(drink["resolution_legal"])
        self.assertEqual(drink["oracle_rank"], 1)
        self.assertTrue(drink["material_gain_label"])

    def test_json_and_jsonl_outputs_are_byte_stable(self) -> None:
        build = _synthetic_build()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "archive.json"
            rows = default_archive_rows_path(output)
            write_first_recovery_branch_archive_outputs(
                build,
                output_path=output,
                archive_rows_path=rows,
            )
            first_json = output.read_bytes()
            first_rows = rows.read_bytes()
            write_first_recovery_branch_archive_outputs(
                build,
                output_path=output,
                archive_rows_path=rows,
            )
            self.assertEqual(first_json, output.read_bytes())
            self.assertEqual(first_rows, rows.read_bytes())
            with gzip.open(rows, "rt", encoding="utf-8") as handle:
                self.assertEqual(len([json.loads(line) for line in handle]), 2)

    def test_trainable_public_input_excludes_provenance_and_private_fields(self) -> None:
        rows = build_first_recovery_archive_rows([_synthetic_branch_result()])
        self.assertFalse(trainable_public_input_leakage(rows)["leakage_detected"])
        trainable = rows[0]["trainable_public_input"]
        self.assertNotIn("seed", trainable)
        self.assertNotIn("source_path", trainable)
        self.assertNotIn("branch_id", trainable)
        self.assertNotIn("logged_action", trainable)
        self.assertIn("seed", rows[0]["provenance"])

        leaked = [dict(rows[0])]
        leaked[0]["trainable_public_input"] = {
            **dict(trainable),
            "seed": 29,
        }
        self.assertTrue(trainable_public_input_leakage(leaked)["leakage_detected"])

    def test_heuristic_count_remains_zero_except_diagnostic_force(self) -> None:
        build = _real_report_build_with_precomputed([_synthetic_branch_result()])
        summary = build.report["branch_archive_summary"]
        self.assertEqual(summary["heuristic_action_source_count"], 0)
        self.assertGreater(summary["diagnostic_forced_action_source_count"], 0)
        self.assertTrue(
            summary["zero_heuristic_runtime_actions_except_diagnostic_force"]
        )

    def test_dominant_oracle_action_share_above_half_classifies_collapsed(self) -> None:
        results = [
            _synthetic_branch_result(branch_id=f"b{i}", seed=seed, oracle_action="drink")
            for i, seed in enumerate((13, 29, 41), start=1)
        ]
        build = _synthetic_report_build_with_precomputed(results)
        self.assertIn(
            "oracle_action_distribution_collapsed",
            build.report["classification"]["labels"],
        )

    def test_leave_one_seed_and_source_splits_are_emitted(self) -> None:
        results = [
            _synthetic_branch_result(branch_id="fixture", seed=29),
            _synthetic_branch_result(
                branch_id="open",
                seed=13,
                source_kind=OPEN_SOURCE_KIND,
                source_path="output/mind/open-mind-v3-seed-13.jsonl.gz",
            ),
        ]
        build = _real_report_build_with_precomputed(results)
        split = build.report["split_metadata"]
        self.assertTrue(split["leave_one_seed_viable"])
        self.assertTrue(split["leave_one_source_viable"])
        self.assertEqual(split["source_count"], 2)

    def test_seed29_rows_are_represented(self) -> None:
        build = _real_report_build_with_precomputed([_synthetic_branch_result(seed=29)])
        self.assertEqual(
            build.report["seed29_summary"]["answer"],
            "seed29_archive_support_present",
        )

    def test_missing_evidence_classifies_inconclusive_not_crash(self) -> None:
        build = build_first_recovery_branch_archive(
            v108_report_path=Path("missing-v108.json"),
            v107_report_path=Path("missing-v107.json"),
            rollout_context_report_path=Path("missing-rollout.json"),
            baseline_report_path=Path("missing-baseline.json"),
            trajectory_paths=(),
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )

    def test_cli_writes_deterministic_json_and_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "archive.json"
            rows = Path(tmp) / "archive.jsonl.gz"
            cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_branch_archive",
                "--trajectory-glob",
                "no-such-v109-trajectories/*.jsonl.gz",
                "--output",
                str(output),
                "--archive-rows-output",
                str(rows),
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
            json_bytes = output.read_bytes()
            rows_bytes = rows.read_bytes()
            second = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("first_recovery_branch_archive=", first.stdout)
            self.assertIn("learnability_readiness=", second.stdout)
            self.assertEqual(json_bytes, output.read_bytes())
            self.assertEqual(rows_bytes, rows.read_bytes())

    def test_contract_declares_no_runtime_or_artifact_effects(self) -> None:
        build = _synthetic_build()
        contract = build.report["contract"]
        self.assertTrue(contract["diagnostics_and_training_data_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertEqual(contract["replay_golden_effect"], "none")
        self.assertEqual(contract["summary_only_effect"], "none")
        self.assertFalse(contract["private_world_state_serialized"])
        self.assertFalse(contract["fixture_identity_policy_input"])
        self.assertFalse(contract["seed_identity_policy_input"])
        self.assertFalse(contract["branch_identity_policy_input"])
        self.assertFalse(contract["source_identity_policy_input"])
        self.assertFalse(contract["logged_action_fallback_policy_input"])
        self.assertEqual(contract["heuristic_action_selection_effect"], "none")

    def test_taxonomy_sanity_check(self) -> None:
        build = _real_report_build_with_precomputed([_synthetic_branch_result()])
        allowed = set(ALLOWED_CLASSIFICATION_LABELS)
        labels = set(build.report["classification"]["labels"])
        answers = set(build.report["classification"]["diagnostic_answers"].values())
        self.assertTrue(labels.issubset(allowed))
        self.assertTrue(answers.issubset(allowed))


def _real_reconstruction() -> Mapping[str, object]:
    return reconstruct_transition_aligned_first_recovery_rows(
        trajectory_paths=_trajectory_paths(),
        trajectory_glob_patterns=(TRAJECTORY_GLOB,),
    )


def _trajectory_paths() -> tuple[Path, ...]:
    return tuple(sorted((ROOT / "output/mind/mind-v3-v5-carrion-only-120-trajectories").glob("*mind-v3*.jsonl.gz")))


def _real_first_recovery_artifact_paths() -> tuple[Path, ...]:
    return (
        ROOT / V108_REPORT,
        ROOT / V107_REPORT,
        ROOT / ROLLOUT_CONTEXT_REPORT,
        ROOT / BASELINE_REPORT,
    )


def _real_first_recovery_artifacts_available() -> bool:
    return bool(_trajectory_paths()) and all(
        path.exists() for path in _real_first_recovery_artifact_paths()
    )


def _require_real_first_recovery_artifacts(
    testcase: unittest.TestCase,
) -> None:
    if not _real_first_recovery_artifacts_available():
        testcase.skipTest("real v5/v107/v108 first-recovery artifacts are not present")


def _real_report_build_with_precomputed(
    branch_results: list[Mapping[str, object]],
):
    return build_first_recovery_branch_archive(
        v108_report_path=ROOT / V108_REPORT,
        v107_report_path=ROOT / V107_REPORT,
        rollout_context_report_path=ROOT / ROLLOUT_CONTEXT_REPORT,
        baseline_report_path=ROOT / BASELINE_REPORT,
        trajectory_paths=_trajectory_paths(),
        trajectory_glob_patterns=(TRAJECTORY_GLOB,),
        precomputed_branch_results=branch_results,
    )


def _synthetic_report_build_with_precomputed(
    branch_results: list[Mapping[str, object]],
):
    return build_first_recovery_branch_archive(
        v108_report={
            "schema_version": "mind_v3_first_recovery_branch_oracle_audit_v1",
            "v107_row_alignment": {
                "reconstructed_first_recovery_row_count": 0,
                "row_count_matches_v107": True,
            },
        },
        v107_report={
            "schema_version": "mind_v3_transition_aligned_recovery_audit_v1",
            "transition_aligned_first_recovery": {
                "constructible_first_recovery_row_count": 0,
            },
        },
        rollout_context_report={"schema_version": "mind_v3_evolution_search_v1"},
        baseline_report={"schema_version": "mind_v3_evolution_search_v1"},
        trajectory_datasets=[_empty_synthetic_trajectory_dataset()],
        precomputed_branch_results=branch_results,
    )


def _synthetic_build():
    return build_first_recovery_branch_archive(
        v108_report={
            "schema_version": "mind_v3_first_recovery_branch_oracle_audit_v1",
            "v107_row_alignment": {
                "reconstructed_first_recovery_row_count": 0,
                "row_count_matches_v107": True,
            },
        },
        v107_report={
            "schema_version": "mind_v3_transition_aligned_recovery_audit_v1",
            "transition_aligned_first_recovery": {
                "constructible_first_recovery_row_count": 0,
            },
        },
        rollout_context_report={"schema_version": "mind_v3_evolution_search_v1"},
        baseline_report={"schema_version": "mind_v3_evolution_search_v1"},
        trajectory_datasets=[],
        precomputed_branch_results=[_synthetic_branch_result()],
        archive_rows_path=Path("archive.jsonl.gz"),
    )


def _empty_synthetic_trajectory_dataset() -> TrajectoryJsonlDataset:
    return TrajectoryJsonlDataset(
        path=Path("synthetic-mind-v3-seed-29.jsonl.gz"),
        header={"config": {"seed": 29, "max_ticks": 120}},
        records=(),
        footer={"summary": {"seed": 29, "ticks_executed": 120}},
    )


def _synthetic_branch_result(
    *,
    branch_id: str = "first-recovery-fixture-seed-29",
    seed: int = 29,
    source_kind: str = FIXTURE_SOURCE_KIND,
    source_path: str = "output/mind/fixture-carrion-only-mind-v3-seed-29.jsonl.gz",
    oracle_action: str = "drink",
) -> dict[str, object]:
    action_mask = {
        "stay": True,
        "move_north": True,
        "move_south": False,
        "move_west": False,
        "move_east": False,
        "eat": True,
        "drink": True,
        "attack_north": False,
        "attack_south": False,
        "attack_west": False,
        "attack_east": False,
    }
    resolution_mask = dict(action_mask)
    stay = _run(
        "stay",
        alive=8,
        births=0,
        deaths=2,
        recovery=0.3,
        digest="digest-stay",
        hydration_delta=-0.05,
        drank=False,
    )
    drink = _run(
        "drink",
        alive=9,
        births=1,
        deaths=1,
        recovery=0.8,
        digest="digest-drink",
        hydration_delta=0.2,
        drank=True,
    )
    runs = [stay, drink]
    best = drink if oracle_action == "drink" else stay
    return {
        "branch_id": branch_id,
        "source_path": source_path,
        "source_kind": source_kind,
        "seed": seed,
        "ticks": 120,
        "branch_tick": 7,
        "record_index": 42,
        "agent_id": 9,
        "logged_action": "stay",
        "gain_tick": 6,
        "gain_record_index": 39,
        "observation_digest": "obs-digest",
        "branch_state_digest": "branch-digest",
        "action_mask": action_mask,
        "resolution_action_mask": resolution_mask,
        "before": {
            "alive": True,
            "energy_ratio": 0.4,
            "hydration_ratio": 0.2,
            "health_ratio": 0.7,
            "age": 31,
        },
        "candidate_action_count": len(runs),
        "action_runs": runs,
        "logged_action_run": stay,
        "oracle_best_action_run": best,
        "oracle_best_action": oracle_action,
        "oracle_changed_action": oracle_action != "stay",
        "oracle_deltas_vs_logged": {
            "target_alive_delta": 0,
            "terminal_alive_delta": 1 if oracle_action == "drink" else 0,
            "birth_delta": 1 if oracle_action == "drink" else 0,
            "target_recovery_score_delta": 0.5 if oracle_action == "drink" else 0,
            "death_reduction_delta": 1 if oracle_action == "drink" else 0,
        },
        "material_oracle_gain": oracle_action == "drink",
    }


def _run(
    action: str,
    *,
    alive: int,
    births: int,
    deaths: int,
    recovery: float,
    digest: str,
    hydration_delta: float,
    drank: bool,
) -> dict[str, object]:
    return {
        "forced_action": action,
        "forced_action_used": True,
        "forced_action_supported": True,
        "alive_agents": alive,
        "births": births,
        "deaths": deaths,
        "terminal_alive_agents": alive,
        "target_alive_at_end": True,
        "target_recovery_score_at_end": recovery,
        "first_action_outcome": {
            "record_found": True,
            "forced_action": action,
            "observation_legal": True,
            "resolution_legal": True,
            "hydration_ratio_delta": hydration_delta,
            "energy_ratio_delta": -0.05,
            "health_ratio_delta": 0.01,
            "moved": False,
            "drank": drank,
            "ate": False,
        },
        "heuristic_action_source_count": 0,
        "diagnostic_forced_action_source_count": 1,
        "requested_action_counts": {action: 1},
        "replay_verification": {
            "verified": True,
            "expected_digest": digest,
            "actual_digest": digest,
        },
    }


if __name__ == "__main__":
    unittest.main()
