from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from evolution_sim.cli import mind_v3_carrion_specific_archive_expansion as cli
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.carrion_specific_archive_expansion import (
    CARRION_BRANCH_REASONS,
    M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION,
    TARGET_CARRION_SEEDS,
    build_carrion_specific_archive_expansion_report,
    carrion_specific_archive_leakage_scan,
    generate_carrion_specific_archive_branch_results,
    merge_carrion_specific_archive_expansion_shards,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSpecificArchiveExpansionTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:carrion-specific-archive-expansion"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_carrion_specific_archive_expansion"
            ),
        )

    def test_report_accepts_balanced_carrion_specific_safe_archive(self) -> None:
        branch_results = _balanced_branch_results()

        report, rows = build_carrion_specific_archive_expansion_report(
            branch_results=branch_results,
            v145_report=_empty_v145_report(),
        )

        self.assertEqual(
            report["schema_version"],
            M3_CARRION_SPECIFIC_ARCHIVE_EXPANSION_SCHEMA_VERSION,
        )
        self.assertTrue(report["diagnostics_only"])
        self.assertTrue(report["contract"]["diagnostics_only"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["runtime_promotion_allowed"])
        self.assertFalse(report["default_runtime_behavior_changed"])
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(report["source_integrity"]["heuristic_action_source_count"], 0)
        self.assertTrue(report["replay_verification"]["complete"])
        self.assertEqual(report["replay_verification"]["verified_count"], 24)
        self.assertTrue(report["action_coverage"]["complete"])
        self.assertTrue(report["leakage_scan"]["passed"])
        self.assertEqual(report["safe_label_count"], 24)
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            report["action_distribution"]["safe_label_action_counts"],
            {"drink": 12, "stay": 12},
        )
        self.assertEqual(
            report["action_distribution"]["dominant_safe_label_action_share"],
            0.5,
        )
        self.assertEqual(
            report["per_seed_support"]["missing_safe_label_seeds"],
            [],
        )
        self.assertEqual(
            report["branch_reason_support"]["missing_branch_reasons"],
            [],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "carrion_specific_archive_support_ready_no_training",
        )
        self.assertEqual(len(str(report["exact_digest"])), 64)
        trainable_text = json.dumps([row["trainable"] for row in rows], sort_keys=True)
        for forbidden in (
            "seed",
            "fixture",
            "branch_id",
            "tick",
            "agent_id",
            "digest",
            "path",
            "private",
            "future",
        ):
            self.assertNotIn(forbidden, trainable_text)

    def test_safe_label_count_below_floor_classifies_archive_support_insufficient(
        self,
    ) -> None:
        branch_results = _balanced_branch_results()[:19]

        report, rows = build_carrion_specific_archive_expansion_report(
            branch_results=branch_results,
            v145_report=_empty_v145_report(),
        )

        self.assertEqual(len(rows), 19)
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "archive_support_insufficient",
        )
        self.assertEqual(
            report["support_floors"]["first_failed_floor"],
            "safe_label_count_gte_minimum",
        )
        self.assertFalse(report["training_authorized"])

    def test_missing_carrion_seed_support_is_explicit_blocker(self) -> None:
        branch_results = [
            _branch_result(
                seed=seed,
                reason=reason,
                action="stay" if index % 2 == 0 else "drink",
                branch_index=index % len(CARRION_BRANCH_REASONS),
            )
            for index, (seed, reason) in enumerate(
                (item for item in _seed_reason_pairs() if item[0] != 43)
            )
        ]
        while len(branch_results) < 20:
            branch_results.append(
                _branch_result(
                    seed=13,
                    reason=CARRION_BRANCH_REASONS[len(branch_results) % 4],
                    action="drink" if len(branch_results) % 2 else "stay",
                    branch_index=len(branch_results) % 4,
                    suffix=f"extra-{len(branch_results)}",
                )
            )

        report, _rows = build_carrion_specific_archive_expansion_report(
            branch_results=branch_results,
            v145_report=_empty_v145_report(),
        )

        self.assertIn(43, report["per_seed_support"]["missing_branch_result_seeds"])
        self.assertIn(43, report["per_seed_support"]["missing_safe_label_seeds"])
        self.assertEqual(
            report["classification"]["primary"],
            "carrion_specific_archive_missing_seed_support_no_training",
        )
        floor_names = {
            floor["name"]: floor["passed"]
            for floor in report["support_floors"]["floors"]
        }
        self.assertFalse(floor_names["all_target_seeds_have_safe_labels"])

    def test_v145_invalid_resolution_risk_blacklist_excludes_label_source(self) -> None:
        branch_result = _branch_result(
            seed=13,
            reason="carrion_contact",
            action="drink",
            branch_index=0,
            agent_id=4,
        )

        report, rows = build_carrion_specific_archive_expansion_report(
            branch_results=[branch_result],
            v145_report={
                "route_decision": {
                    "label_blacklist": [
                        {
                            "seed": 13,
                            "agent_id": 4,
                            "tick": 0,
                            "residual_action": "drink",
                            "branch_id": "other-branch",
                            "label_source_row_index": 0,
                        }
                    ]
                }
            },
            min_safe_label_count=1,
        )

        self.assertEqual(rows, [])
        self.assertEqual(report["safe_label_count"], 0)
        self.assertEqual(report["blacklist"]["blacklist_count"], 1)
        self.assertEqual(
            report["excluded_rows"][0]["excluded_reason"],
            "invalid_resolution_risk_blacklist",
        )
        self.assertEqual(
            report["classification"]["primary"],
            "archive_support_insufficient",
        )

    def test_carrion_specific_leakage_scan_rejects_future_and_digest_values(
        self,
    ) -> None:
        row = {
            "trainable": {
                "features": {
                    "observation_input": {
                        "future_outcome": 1,
                        "opaque": "a" * 64,
                    },
                    "action_mask": _action_mask("stay"),
                },
                "label": {"action": "stay"},
            }
        }

        scan = carrion_specific_archive_leakage_scan([row])

        self.assertFalse(scan["passed"])
        reasons = {failure["reason"] for failure in scan["failures"]}
        self.assertIn(
            "carrion_specific_forbidden_trainable_path_token",
            reasons,
        )
        self.assertIn("carrion_specific_forbidden_digest_value", reasons)

    def test_cli_writes_report_and_dataset_from_existing_branch_results(self) -> None:
        branch_results = _balanced_branch_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            branch_results_path = root / "branch-results.json"
            v145_path = root / "v145.json"
            output = root / "report.json"
            dataset = root / "dataset.jsonl"
            branch_results_path.write_text(
                json.dumps(branch_results, sort_keys=True),
                encoding="utf-8",
            )
            v145_path.write_text(
                json.dumps(_empty_v145_report(), sort_keys=True),
                encoding="utf-8",
            )

            cli.main(
                [
                    "--use-existing-branch-results",
                    "--branch-results",
                    str(branch_results_path),
                    "--v145-report",
                    str(v145_path),
                    "--output",
                    str(output),
                    "--dataset-output",
                    str(dataset),
                ]
            )

            written = json.loads(output.read_text(encoding="utf-8"))
            dataset_rows = [
                json.loads(line)
                for line in dataset.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(
            written["classification"]["primary"],
            "carrion_specific_archive_support_ready_no_training",
        )
        self.assertEqual(len(dataset_rows), 24)
        self.assertEqual(
            written["dataset"]["dataset_digest"],
            stable_payload_digest(dataset_rows),
        )
        self.assertFalse(written["runtime_promotion_allowed"])

    def test_synthetic_two_shard_merge_succeeds(self) -> None:
        branch_results = _balanced_branch_results()
        shard_a = _shard_report(branch_results[:12], shard_id="shard-a")
        shard_b = _shard_report(branch_results[12:], shard_id="shard-b")

        report, rows = merge_carrion_specific_archive_expansion_shards(
            shard_reports=[shard_a, shard_b],
            v145_report=_empty_v145_report(),
        )

        self.assertEqual(report["branch_result_count"], 24)
        self.assertEqual(len(rows), 24)
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "carrion_specific_archive_support_ready_no_training",
        )
        self.assertEqual(len(report["shard_merge"]["sources"]), 2)
        self.assertEqual(
            {
                source["shard_id"]
                for source in report["shard_merge"]["sources"]
                if source.get("shard_id")
            },
            {"shard-a", "shard-b"},
        )

    def test_seed_subset_shards_merge_into_requested_target_seed_bank(self) -> None:
        branch_results = _balanced_branch_results()
        seed_13_results = [
            result for result in branch_results if result.get("seed") == 13
        ]
        seed_19_results = [
            result for result in branch_results if result.get("seed") == 19
        ]
        shard_13 = _shard_report(
            seed_13_results,
            shard_id="seed-13",
            target_seeds=(13,),
        )
        shard_19 = _shard_report(
            seed_19_results,
            shard_id="seed-19",
            target_seeds=(19,),
        )

        report, rows = merge_carrion_specific_archive_expansion_shards(
            shard_reports=[shard_13, shard_19],
            v145_report=_empty_v145_report(),
            target_seeds=(13, 19, 29, 37, 41, 43),
        )

        self.assertEqual(report["branch_result_count"], 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            report["generation_status"]["target_carrion_seeds"],
            [13, 19, 29, 37, 41, 43],
        )
        self.assertEqual(
            report["per_seed_support"]["missing_safe_label_seeds"],
            [29, 37, 41, 43],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "archive_support_insufficient",
        )

    def test_reversed_shard_order_yields_same_exact_and_dataset_digest(self) -> None:
        branch_results = _balanced_branch_results()
        shard_a = _shard_report(branch_results[:12], shard_id="shard-a")
        shard_b = _shard_report(branch_results[12:], shard_id="shard-b")

        report_ab, _rows_ab = merge_carrion_specific_archive_expansion_shards(
            shard_reports=[shard_a, shard_b],
            v145_report=_empty_v145_report(),
        )
        report_ba, _rows_ba = merge_carrion_specific_archive_expansion_shards(
            shard_reports=[shard_b, shard_a],
            v145_report=_empty_v145_report(),
        )

        self.assertEqual(report_ab["exact_digest"], report_ba["exact_digest"])
        self.assertEqual(
            report_ab["dataset"]["dataset_digest"],
            report_ba["dataset"]["dataset_digest"],
        )

    def test_conflicting_duplicate_branch_id_fails(self) -> None:
        branch_results = _balanced_branch_results()
        conflicting = deepcopy(branch_results[0])
        conflicting["action_runs"][0]["target_terminal"]["energy_ratio"] = 0.75
        shard_a = _shard_report([branch_results[0]], shard_id="shard-a")
        shard_b = _shard_report([conflicting], shard_id="shard-b")

        with self.assertRaisesRegex(
            ValueError,
            "duplicate branch_id with different digest",
        ):
            merge_carrion_specific_archive_expansion_shards(
                shard_reports=[shard_a, shard_b],
                v145_report=_empty_v145_report(),
            )

    def test_partial_shard_rejected_by_default(self) -> None:
        shard = _shard_report(
            _balanced_branch_results()[:4],
            shard_id="partial-shard",
            partial=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "partial shard evidence requires explicit partial merge",
        ):
            merge_carrion_specific_archive_expansion_shards(
                shard_reports=[shard],
                v145_report=_empty_v145_report(),
            )

    def test_shard_id_appears_in_generation_evidence_status(self) -> None:
        generated = generate_carrion_specific_archive_branch_results(
            seeds=(13,),
            ticks=1,
            max_branch_points_per_seed=1,
            max_candidate_actions=1,
            shard_id="unit-shard",
            branch_index_include=(0,),
        )

        self.assertEqual(generated["shard_id"], "unit-shard")
        self.assertEqual(
            generated["generation_status"]["shard_id"],
            "unit-shard",
        )
        self.assertEqual(generated["branch_index_include"], [0])
        self.assertEqual(
            generated["generation_status"]["branch_index_include"],
            [0],
        )


def _balanced_branch_results() -> list[dict[str, object]]:
    results = []
    for index, (seed, reason) in enumerate(_seed_reason_pairs()):
        results.append(
            _branch_result(
                seed=seed,
                reason=reason,
                action="stay" if index % 2 == 0 else "drink",
                branch_index=index % len(CARRION_BRANCH_REASONS),
            )
        )
    return results


def _shard_report(
    branch_results: list[dict[str, object]],
    *,
    shard_id: str,
    partial: bool = False,
    target_seeds: tuple[int, ...] = TARGET_CARRION_SEEDS,
) -> dict[str, object]:
    report, _rows = build_carrion_specific_archive_expansion_report(
        branch_results=branch_results,
        v145_report=_empty_v145_report(),
        target_seeds=target_seeds,
        generation_status={
            "policy": "test_carrion_specific_archive_shard_status_v1",
            "state": "partial" if partial else "complete",
            "partial": bool(partial),
            "stop_reason": "unit_test_partial" if partial else None,
            "target_fixture": "carrion_only",
            "target_carrion_seeds": [int(seed) for seed in target_seeds],
            "ticks": 120,
            "branch_point_count": len(branch_results) + (1 if partial else 0),
            "branch_result_count": len(branch_results),
            "shard_id": shard_id,
        },
        generation_evidence={
            "policy": "test_carrion_specific_archive_shard_evidence_v1",
            "shard_id": shard_id,
        },
    )
    return report


def _seed_reason_pairs() -> list[tuple[int, str]]:
    return [
        (seed, reason)
        for seed in TARGET_CARRION_SEEDS
        for reason in CARRION_BRANCH_REASONS
    ]


def _branch_result(
    *,
    seed: int,
    reason: str,
    action: str,
    branch_index: int,
    agent_id: int = 9,
    suffix: str | None = None,
) -> dict[str, object]:
    slug = reason.replace("_", "-")
    suffix_text = "" if suffix is None else f"-{suffix}"
    branch_id = (
        f"m3-carrion-specific-archive-seed-{seed}-{slug}-branch-"
        f"{branch_index}-tick-{branch_index}-agent-{agent_id}{suffix_text}"
    )
    observation = {"values": [float(seed) / 100.0, float(branch_index), 0.25]}
    mask = _action_mask(action)
    return {
        "branch_id": branch_id,
        "seed": int(seed),
        "fixture": "carrion_only",
        "ticks": 120,
        "branch_tick": int(branch_index),
        "record_index": int(branch_index),
        "branch_index": int(branch_index),
        "agent_id": int(agent_id),
        "baseline_action": action,
        "v142_requested_action": action,
        "v142_resolved_action": action,
        "candidate_actions": [action],
        "candidate_action_count": 1,
        "branch_state_digest": "b" * 64,
        "source_trajectory_path": None,
        "public_features": {
            "observation_input": observation,
            "action_mask": mask,
        },
        "carrion_archive_context": {
            "policy": "carrion_contact_hydration_stall_terminal_branch_selection_v1",
            "fixture": "carrion_only",
            "branch_reason": reason,
            "reason_rank": int(branch_index),
            "reason_evidence": {"branch_reason": reason},
        },
        "action_runs": [_action_run(action)],
        "diagnostics_only": True,
    }


def _action_run(action: str) -> dict[str, object]:
    return {
        "forced_action": action,
        "forced_action_used": True,
        "forced_action_supported": True,
        "heuristic_action_source_count": 0,
        "unsupported_requested_action_count": 0,
        "deltas_vs_baseline": {
            "alive_agents": 0,
            "births": 0,
            "deaths": 0,
            "target_alive": 0,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
        },
        "deltas_vs_v142_override": {
            "alive_agents": 0,
            "births": 0,
            "deaths": 0,
            "target_alive": 0,
            "unsupported_requested_action_count": 0,
            "unsupported_resolved_action_count": 0,
        },
        "target_terminal": {
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 1.0,
        },
        "replay_digest": "c" * 64,
        "replay_verification": {
            "verified": True,
            "expected_digest": "c" * 64,
            "actual_digest": "c" * 64,
        },
    }


def _action_mask(action: str) -> dict[str, bool]:
    return {name: name == action for name in ACTION_NAMES}


def _empty_v145_report() -> dict[str, object]:
    return {"route_decision": {"label_blacklist": []}}


if __name__ == "__main__":
    unittest.main()
