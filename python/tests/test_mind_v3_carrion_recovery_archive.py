from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import (
    mind_v3_carrion_recovery_archive,
    mind_v3_carrion_recovery_archive_validate,
)
from evolution_sim.mind.carrion_branch_explore import (
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_counterfactual import (
    MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_ARCHIVE_VALIDATION_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
    MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
    MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY,
    build_fixture_rerank_recovery_probe_archive_report,
    build_carrion_recovery_archive_report,
    build_carrion_recovery_archive_validation_report,
    write_carrion_recovery_archive_report,
    write_carrion_recovery_dataset_records,
)


class MindV3CarrionRecoveryArchiveTests(unittest.TestCase):
    def test_recovery_archive_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-recovery-archive"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_recovery_archive"
            ),
        )
        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-recovery-archive-validate"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_recovery_archive_validate"
            ),
        )

    def test_archive_keeps_survivor_and_failure_elites_and_exports_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "recovery-dataset.jsonl"

            report = build_carrion_recovery_archive_report(
                branch_report=_synthetic_branch_report(),
                dataset_output_path=dataset_path,
                min_survivor_cells=2,
                min_failure_cells=1,
            )

            lines = dataset_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertTrue(report["acceptance"]["archive_acceptance_passed"])
        self.assertGreaterEqual(report["aggregate"]["survivor_cell_count"], 2)
        self.assertGreaterEqual(report["aggregate"]["failure_cell_count"], 1)
        self.assertGreaterEqual(report["dataset"]["survivor_count"], 2)
        self.assertGreaterEqual(report["dataset"]["failure_count"], 1)
        outcome = report["aggregate"]["outcome_metrics"]
        self.assertEqual(outcome["terminal_survivor_run_count"], 3)
        self.assertEqual(outcome["total_births"], 36)
        self.assertIn("total_scavenger_animal_resource_events", outcome)
        self.assertEqual(len(lines), report["dataset"]["record_count"])
        first_record = json.loads(lines[0])
        self.assertIn("record_id", first_record)
        self.assertIn("outcome_metrics", first_record)
        self.assertEqual(
            first_record["schema_version"],
            "mind_v3_carrion_recovery_dataset_record_v1",
        )

    def test_recovery_archive_cli_writes_report_and_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "archive.json"
            dataset_path = Path(tmpdir) / "dataset.jsonl"

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_archive",
                    "--seeds",
                    "29",
                    "--ticks",
                    "8",
                    "--continuation-script",
                    "hydration_safe_carrion_cycle",
                    "--min-survivor-cells",
                    "1",
                    "--min-failure-cells",
                    "0",
                    "--dataset-output",
                    str(dataset_path),
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_recovery_archive.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            dataset_lines = dataset_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            payload["schema_version"],
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertTrue(payload["acceptance"]["archive_acceptance_passed"])
        self.assertEqual(len(dataset_lines), payload["dataset"]["record_count"])

    def test_archive_can_append_counterfactual_fixture_survivor_records(
        self,
    ) -> None:
        report = build_carrion_recovery_archive_report(
            branch_report=_synthetic_branch_report(),
            counterfactual_report=_synthetic_counterfactual_report(),
            min_survivor_cells=2,
            min_failure_cells=1,
            min_counterfactual_survivor_seeds=2,
        )

        counterfactual_records = [
            record
            for record in report["dataset"]["records"]
            if record["source"].get("source_type")
            == "counterfactual_fixture_rollout"
        ]

        self.assertTrue(report["acceptance"]["archive_acceptance_passed"])
        self.assertEqual(
            report["aggregate"]["counterfactual_survivor_seed_count"],
            2,
        )
        self.assertEqual(
            report["aggregate"]["counterfactual_survivor_seeds"],
            [13, 29],
        )
        self.assertEqual(len(counterfactual_records), 3)
        self.assertGreaterEqual(report["dataset"]["survivor_count"], 4)

    def test_archive_counts_only_trainable_counterfactual_survivor_seeds(
        self,
    ) -> None:
        counterfactual_report = _synthetic_counterfactual_report()
        counterfactual_report["scripts"][0]["runs"][1].pop("trajectory_path")

        report = build_carrion_recovery_archive_report(
            branch_report=_synthetic_branch_report(),
            counterfactual_report=counterfactual_report,
            min_survivor_cells=2,
            min_failure_cells=1,
            min_counterfactual_survivor_seeds=2,
        )

        self.assertFalse(report["acceptance"]["archive_acceptance_passed"])
        self.assertIn(
            "insufficient_counterfactual_survivor_seeds",
            report["acceptance"]["blockers"],
        )
        self.assertEqual(
            report["aggregate"]["counterfactual_report_survivor_seeds"],
            [13, 29],
        )
        self.assertEqual(
            report["aggregate"]["counterfactual_survivor_seeds"],
            [13],
        )

    def test_rerank_probe_archive_reads_completed_fixture_rerank_candidates(
        self,
    ) -> None:
        report = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[
                (
                    "probe",
                    _synthetic_rerank_probe_search_report(),
                    "search.json",
                )
            ]
        )

        self.assertEqual(
            report["source_mode"],
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
        )
        self.assertTrue(report["report_only"])
        self.assertFalse(report["changes_evolve_selection"])
        self.assertEqual(report["input_summary"]["candidate_count"], 4)
        self.assertEqual(report["input_summary"]["complete_probe_count"], 4)
        self.assertEqual(report["input_summary"]["missing_probe_count"], 0)
        self.assertGreaterEqual(report["archive"]["cell_count"], 3)
        self.assertIn(
            "selected",
            report["input_summary"]["source_type_counts"],
        )
        selected_memberships = report["selected"]["candidate_cell_memberships"]
        self.assertEqual(selected_memberships[0]["candidate_id"], "selected")
        self.assertIn(
            "source_type:selected",
            selected_memberships[0]["facet_cells"],
        )

    def test_rerank_probe_archive_counts_missing_probe_fields(self) -> None:
        search_report = _synthetic_rerank_probe_search_report()
        candidates = search_report["fixture_rerank"]["candidates"]
        candidates[1]["carrion_recovery_probe"].pop("mean_water_distance")
        candidates[2].pop("carrion_recovery_probe")

        report = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[("probe", search_report, "search.json")]
        )

        self.assertEqual(report["input_summary"]["candidate_count"], 4)
        self.assertEqual(report["input_summary"]["complete_probe_count"], 2)
        self.assertEqual(report["input_summary"]["missing_probe_count"], 2)
        self.assertEqual(
            report["input_summary"]["missing_probe_count_by_reason"],
            {
                "probe_missing": 1,
                "probe_missing_required_fields": 1,
            },
        )
        self.assertEqual(
            report["input_summary"]["missing_probe_count_by_field"][
                "mean_water_distance"
            ],
            1,
        )

    def test_rerank_probe_archive_cells_are_deterministic(self) -> None:
        search_report = _synthetic_rerank_probe_search_report()
        first = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[("probe", search_report, "search.json")]
        )
        second = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[("probe", search_report, "search.json")]
        )

        self.assertEqual(first["archive"]["cells"], second["archive"]["cells"])
        self.assertEqual(
            first["provenance"]["combined_source_digest"],
            second["provenance"]["combined_source_digest"],
        )

    def test_rerank_probe_archive_reports_selected_dominance(self) -> None:
        report = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[
                (
                    "probe",
                    _synthetic_rerank_probe_search_report(),
                    "search.json",
                )
            ]
        )

        self.assertTrue(
            report["selected"]["dominates_all_non_selected_recovery_cells"]
        )
        self.assertFalse(
            report["retention"]["non_selected_recovery_better_than_selected"]
        )
        self.assertEqual(
            report["retention"]["recovery_better_candidate_ids"],
            [],
        )
        self.assertTrue(
            report["retention"]["would_add_new_parent_candidates"]
        )
        self.assertIn(
            "hydration",
            report["retention"]["retained_candidate_ids_that_would_be_added"],
        )

    def test_rerank_probe_archive_reports_non_selected_recovery_better(
        self,
    ) -> None:
        search_report = _synthetic_rerank_probe_search_report()
        candidate = search_report["fixture_rerank"]["candidates"][1]
        candidate["carrion_recovery_probe"]["post_contact_survival_rate"] = 0.5
        candidate["carrion_recovery_probe"]["drink_after_carrion_rate"] = 0.95

        report = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[("probe", search_report, "search.json")]
        )

        self.assertFalse(
            report["selected"]["dominates_all_non_selected_recovery_cells"]
        )
        self.assertTrue(
            report["retention"]["non_selected_recovery_better_than_selected"]
        )
        self.assertEqual(
            report["retention"]["recovery_better_candidate_ids"],
            ["hydration"],
        )

    def test_rerank_probe_archive_cli_writes_report_only_audit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            search_path = root / "search.json"
            output_path = root / "archive.json"
            search_path.write_text(
                json.dumps(_synthetic_rerank_probe_search_report()),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_archive",
                    "--source",
                    "fixture-rerank-recovery-probe",
                    "--search-report",
                    f"probe={search_path}",
                    "--output",
                    str(output_path),
                ],
            ):
                mind_v3_carrion_recovery_archive.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["source_mode"],
            "fixture-rerank-recovery-probe",
        )
        self.assertTrue(payload["report_only"])
        self.assertFalse(payload["changes_evolve_selection"])

    def test_rerank_probe_archive_contract_names_opt_in_retention_policy(
        self,
    ) -> None:
        report = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=[
                (
                    "probe",
                    _synthetic_rerank_probe_search_report(),
                    "search.json",
                )
            ]
        )

        self.assertIn(
            MIND_V3_GATE_ALIGNED_CARRION_RECOVERY_ARCHIVE_RETENTION_POLICY,
            report["archive_contract"]["retention_policy"],
        )

    def test_archive_validate_builds_leakage_safe_split_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive, branch_report, dataset_path = _validation_archive_fixture(root)

            with patch(
                "evolution_sim.mind.carrion_recovery_archive.load_trajectory_jsonl",
                return_value=SimpleNamespace(
                    record_count=2,
                    records=(
                        {"action_source": "mind_v3_autonomous_evolution_policy_v1"},
                        {"action_source": "counterfactual_script:water_first_recovery"},
                    ),
                ),
            ):
                report = build_carrion_recovery_archive_validation_report(
                    archive_report=archive,
                    dataset_path=dataset_path,
                    branch_report=branch_report,
                )

        split = report["split"]
        train_keys = set(split["branch_state_keys"]["train"])
        heldout_keys = set(split["branch_state_keys"]["heldout"])
        self.assertEqual(
            report["schema_version"],
            MIND_V3_CARRION_RECOVERY_ARCHIVE_VALIDATION_SCHEMA_VERSION,
        )
        self.assertTrue(report["acceptance"]["validation_passed"])
        self.assertFalse(report["acceptance"]["training_blocked"])
        self.assertEqual(
            split["schema_version"],
            MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
        )
        self.assertFalse(train_keys & heldout_keys)
        self.assertGreater(split["aggregate"]["train_record_count"], 0)
        self.assertGreater(split["aggregate"]["heldout_record_count"], 0)
        self.assertGreater(split["aggregate"]["heldout_survivor_count"], 0)
        self.assertGreater(split["aggregate"]["heldout_failure_count"], 0)
        self.assertTrue(split["leakage_check"]["passed"])

    def test_archive_validate_blocks_duplicate_record_ids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive, branch_report, dataset_path = _validation_archive_fixture(root)
            records = list(archive["dataset"]["records"])
            records[1]["record_id"] = records[0]["record_id"]
            write_carrion_recovery_dataset_records(records, dataset_path)

            with patch(
                "evolution_sim.mind.carrion_recovery_archive.load_trajectory_jsonl",
                return_value=SimpleNamespace(record_count=1, records=()),
            ):
                report = build_carrion_recovery_archive_validation_report(
                    archive_report=archive,
                    dataset_path=dataset_path,
                    branch_report=branch_report,
                )

        self.assertFalse(report["acceptance"]["validation_passed"])
        self.assertIn("duplicate_record_ids", report["acceptance"]["blockers"])
        self.assertTrue(report["acceptance"]["training_blocked"])

    def test_archive_validate_cli_writes_validation_and_split(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive, branch_report, dataset_path = _validation_archive_fixture(root)
            archive_path = root / "archive.json"
            branch_path = root / "branch.json"
            validation_path = root / "validation.json"
            split_path = root / "split.json"
            archive["source"]["branch_report_path"] = str(branch_path)
            write_carrion_recovery_archive_report(archive, archive_path)
            branch_path.write_text(json.dumps(branch_report), encoding="utf-8")

            with patch(
                "evolution_sim.mind.carrion_recovery_archive.load_trajectory_jsonl",
                return_value=SimpleNamespace(record_count=1, records=()),
            ), patch(
                "sys.argv",
                [
                    "mind_v3_carrion_recovery_archive_validate",
                    "--archive-report",
                    str(archive_path),
                    "--dataset",
                    str(dataset_path),
                    "--split-output",
                    str(split_path),
                    "--output",
                    str(validation_path),
                ],
            ):
                mind_v3_carrion_recovery_archive_validate.main()

            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            split = json.loads(split_path.read_text(encoding="utf-8"))

        self.assertTrue(validation["acceptance"]["validation_passed"])
        self.assertEqual(
            split["schema_version"],
            MIND_V3_CARRION_RECOVERY_SPLIT_SCHEMA_VERSION,
        )


def _synthetic_branch_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
        "branch_policy": "deterministic_post_contact_branch_explore_v1",
        "contract": {"ticks": 120, "seeds": [29, 37]},
        "aggregate": {
            "replay_verified": True,
            "branch_run_count": 4,
        },
        "acceptance": {
            "diagnostic_acceptance_passed": True,
        },
        "branch_points": [],
        "branch_runs": [
            _branch_run(
                seed=29,
                script="hydration_safe_carrion_cycle",
                alive=3,
                births=12,
                dominant_action="stay",
                dominant_share=0.24,
            ),
            _branch_run(
                seed=29,
                script="water_first_recovery",
                alive=1,
                births=6,
                dominant_action="drink",
                dominant_share=0.25,
            ),
            _branch_run(
                seed=37,
                script="conserve_after_carrion",
                alive=1,
                births=5,
                dominant_action="stay",
                dominant_share=0.54,
            ),
            _branch_run(
                seed=37,
                script="carrion_then_water",
                alive=0,
                births=13,
                dominant_action="move_north",
                dominant_share=0.20,
            ),
        ],
    }


def _synthetic_counterfactual_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
        "counterfactual_policy": "scripted_policy_visible_carrion_water_recovery_v1",
        "counterfactual_contract": {
            "seeds": [13, 29, 37],
            "ticks": 120,
            "fixture_name": "carrion_only",
        },
        "scripts": [
            {
                "script_name": "hydration_safe_carrion_cycle",
                "runs": [
                    _counterfactual_run(seed=13, alive=2, births=9),
                    _counterfactual_run(seed=29, alive=3, births=12),
                    _counterfactual_run(seed=37, alive=0, births=4),
                ],
            }
        ],
    }


def _synthetic_rerank_probe_search_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_evolution_search_v1",
        "fixture_rerank": {
            "selected_candidate_id": "selected",
            "candidates": [
                _rerank_candidate(
                    candidate_id="selected",
                    selected=True,
                    source="initial",
                    survival=0.2,
                    drink=0.8,
                    hydration=0.02,
                    water=2.0,
                    unsupported=4,
                    dominant=0.42,
                    score=20.0,
                ),
                _rerank_candidate(
                    candidate_id="hydration",
                    source="initial",
                    survival=0.0,
                    drink=0.9,
                    hydration=0.2,
                    water=1.5,
                    unsupported=2,
                    dominant=0.4,
                    score=30.0,
                ),
                _rerank_candidate(
                    candidate_id="repair",
                    source="repair",
                    survival=0.1,
                    drink=0.4,
                    hydration=-0.1,
                    water=3.0,
                    unsupported=3,
                    dominant=0.45,
                    score=15.0,
                ),
                _rerank_candidate(
                    candidate_id="bridge",
                    source="bridge-repair",
                    survival=0.0,
                    drink=0.0,
                    hydration=-0.3,
                    water=5.0,
                    unsupported=9,
                    dominant=0.6,
                    score=10.0,
                ),
            ],
        },
    }


def _validation_archive_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    branch_report = _synthetic_branch_report()
    branch_report["branch_points"] = [
        {
            "branch_id": "branch-29",
            "branch_state_digest": "digest-branch-29",
            "seed": 29,
            "branch_tick": 0,
        },
        {
            "branch_id": "branch-37",
            "branch_state_digest": "digest-branch-37",
            "seed": 37,
            "branch_tick": 0,
        },
    ]
    branch_report["train_heldout_split_metadata"] = {
        "by_branch_state_digest": [
            {
                "branch_id": "branch-29",
                "branch_state_digest": "digest-branch-29",
                "seed": 29,
                "split": "source_search_train",
            },
            {
                "branch_id": "branch-37",
                "branch_state_digest": "digest-branch-37",
                "seed": 37,
                "split": "source_search_holdout",
            },
        ],
    }
    archive = build_carrion_recovery_archive_report(
        branch_report=branch_report,
        min_survivor_cells=2,
        min_failure_cells=1,
    )
    records = archive["dataset"]["records"]
    for index, record in enumerate(records):
        path = root / f"trajectory-{index}.jsonl.gz"
        path.write_text("", encoding="utf-8")
        record["source"]["trajectory_path"] = str(path)
    dataset_path = root / "dataset.jsonl"
    write_carrion_recovery_dataset_records(records, dataset_path)
    return archive, branch_report, dataset_path


def _rerank_candidate(
    *,
    candidate_id: str,
    survival: float,
    drink: float,
    hydration: float,
    water: float,
    unsupported: int,
    dominant: float,
    score: float,
    source: str = "initial",
    selected: bool = False,
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "candidate_id": candidate_id,
        "prefilter_rank": 0 if selected else 1,
        "search_score": score,
        "carrion_recovery_probe": {
            "available": True,
            "fixture_blocker_count": 5,
            "carrion_only_blocker_count": 5,
            "post_contact_survival_rate": survival,
            "drink_after_carrion_rate": drink,
            "mean_hydration_delta_after_carrion": hydration,
            "mean_water_distance": water,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": unsupported,
            "dominant_requested_action_share": dominant,
            "missing_fields": [],
        },
    }
    if source == "repair":
        candidate["fixture_repair"] = {
            "donor_selection_reason": "scavenger_lane",
        }
    elif source == "bridge-repair":
        candidate["fixture_repair"] = {
            "donor_selection_reason": "promotion_safe_bridge",
        }
    return candidate


def _branch_run(
    *,
    seed: int,
    script: str,
    alive: int,
    births: int,
    dominant_action: str,
    dominant_share: float,
) -> dict[str, object]:
    return {
        "branch_id": f"branch-{seed}",
        "seed": seed,
        "fixture": "carrion_only",
        "branch_tick": 0,
        "base_script": "hydration_safe_carrion_cycle",
        "continuation_script": script,
        "alive_agents": alive,
        "births": births,
        "deaths": 12 + births - alive,
        "contact": {
            "food_source": "carcass",
            "gained_energy": 0.1017,
            "after": {
                "energy_ratio": 0.5609,
                "hydration_ratio": 0.9423,
                "health_ratio": 1.0,
            },
        },
        "dominant_requested_action": dominant_action,
        "dominant_requested_action_share": dominant_share,
        "unique_requested_actions": 7,
        "heuristic_action_source_count": 0,
        "zero_heuristic_runtime_actions": True,
        "trajectory_path": f"output/branch-{seed}-{script}.jsonl.gz",
    }


def _counterfactual_run(
    *,
    seed: int,
    alive: int,
    births: int,
) -> dict[str, object]:
    return {
        "seed": seed,
        "fixture": "carrion_only",
        "counterfactual_script": "hydration_safe_carrion_cycle",
        "alive_agents": alive,
        "births": births,
        "deaths": 12 + births - alive,
        "dominant_requested_action": "stay",
        "dominant_requested_action_share": 0.24,
        "unique_requested_actions": 7,
        "heuristic_action_source_count": 0,
        "trajectory_path": (
            "output/mind/counterfactual-"
            f"hydration-safe-carrion-cycle-seed-{seed}.jsonl.gz"
        ),
    }


if __name__ == "__main__":
    unittest.main()
