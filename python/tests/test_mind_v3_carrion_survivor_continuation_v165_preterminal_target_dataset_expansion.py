from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    _record_materialization_payload,
)
from evolution_sim.mind.carrion_survivor_continuation_v164_preterminal_tied_set_branch_target_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    EXPECTED_V164_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion,
)
from evolution_sim.mind.provenance import stable_payload_digest

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_v159_scorer_readiness import (
        _write_report_with_exact_digest,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV165PreterminalTargetDatasetExpansionTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v165-preterminal-target-dataset-expansion"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion"
            ),
        )

    def test_source_invalid_v164_classification_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v164_path, base_path, _ = _write_ready_inputs(tmpdir)
            report = json.loads(v164_path.read_text(encoding="utf-8"))
            report["classification"]["primary"] = "unexpected"
            _write_report_with_exact_digest(v164_path, report)
            written = json.loads(v164_path.read_text(encoding="utf-8"))

            result = run_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion(
                v164_report_path=v164_path,
                base_dataset_path=base_path,
                output_path=Path(tmpdir) / "v165.json",
                target_dataset_output_path=Path(tmpdir) / "v165.jsonl",
                expected_v164_exact_digest=written["exact_digest"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v164_unexpected_classification",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            "source_invalid_closed_no_training",
        )
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_builds_combined_dataset_with_metadata_only_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v164_path, base_path, source_records = _write_ready_inputs(tmpdir)
            written = json.loads(v164_path.read_text(encoding="utf-8"))
            output_path = Path(tmpdir) / "v165.json"
            dataset_path = Path(tmpdir) / "v165.jsonl"

            result = run_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion(
                v164_report_path=v164_path,
                base_dataset_path=base_path,
                output_path=output_path,
                target_dataset_output_path=dataset_path,
                expected_v164_exact_digest=written["exact_digest"],
            )
            rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(
            result["classification"]["primary"],
            "expanded_target_dataset_support_ready_no_training",
        )
        self.assertTrue(result["leakage_scan"]["passed"])
        self.assertEqual(result["dataset"]["base_row_count"], 1)
        self.assertEqual(result["dataset"]["preterminal_row_count"], 2)
        self.assertEqual(result["dataset"]["combined_row_count"], 3)
        self.assertEqual(len(rows), 3)
        preterminal = [row for row in rows if row.get("row_origin") == "v165_preterminal_branch_target"]
        self.assertEqual(len(preterminal), 2)
        self.assertEqual(
            result["preterminal_rows"]["per_action_support_counts"],
            {"eat": 1, "move_north": 1},
        )
        for row, source_record in zip(preterminal, source_records, strict=True):
            self.assertIn("metadata", row)
            self.assertEqual(row["metadata"]["source_record_digest"], stable_payload_digest(_record_materialization_payload(source_record)))
            trainable_text = json.dumps(row["trainable_public_features"], sort_keys=True)
            self.assertNotIn("branch_id", trainable_text)
            self.assertNotIn("source_path", trainable_text)
            self.assertNotIn("replay_digest", trainable_text)
        self.assertTrue(
            result["future_evaluation_policy"][
                "leave_source_seed_out_diagnostics_required"
            ]
        )
        self.assertTrue(
            result["source_seed_overlap"][
                "v164_strict_broad_seeds_become_support_provenance_seeds"
            ]
        )

    def test_cli_writes_source_invalid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v164_path = Path(tmpdir) / "invalid-v164.json"
            base_path = Path(tmpdir) / "base.jsonl"
            output_path = Path(tmpdir) / "v165.json"
            dataset_path = Path(tmpdir) / "v165.jsonl"
            base_path.write_text("", encoding="utf-8")
            _write_report_with_exact_digest(
                v164_path,
                {
                    "schema_version": "wrong",
                    "policy": "wrong",
                    "classification": {"primary": "wrong", "labels": ["wrong"]},
                    "diagnostics_only": True,
                    "training_ran": False,
                    "artifact_created": False,
                    "runtime_artifact_created": False,
                    "live_ab_allowed": False,
                    "promotion_authorized": False,
                    "runtime_action_selection_changed": False,
                },
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion",
                    "--v164-report",
                    str(v164_path),
                    "--base-dataset",
                    str(base_path),
                    "--output",
                    str(output_path),
                    "--target-dataset-output",
                    str(dataset_path),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v165_preterminal_target_dataset_expansion_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_SCHEMA_VERSION,
        )
        self.assertEqual(
            written["classification"]["primary"],
            "source_invalid_closed_no_training",
        )
        self.assertIn("exact_digest", written)


def _write_ready_inputs(tmpdir: str) -> tuple[Path, Path, list[dict[str, object]]]:
    root = Path(tmpdir)
    source_path = root / "source.jsonl"
    base_path = root / "base.jsonl"
    v164_path = root / "v164.json"
    source_records = [
        _source_record(tick=10, agent_id=2),
        _source_record(tick=11, agent_id=3),
    ]
    source_path.write_text(
        "\n".join(
            json.dumps({"type": "decision", "record": record}, sort_keys=True)
            for record in source_records
        )
        + "\n",
        encoding="utf-8",
    )
    base_path.write_text(
        json.dumps(
            {
                "schema_version": "m3_carrion_survivor_continuation_set_valued_action_value_target_dataset_row_v1",
                "feature_policy_id": "fixture_public",
                "trainable_public_features": {
                    "public_observation": {"data": "base"},
                    "action_mask": {"stay": True, "eat": True},
                },
                "public_action_mask": {"stay": True, "eat": True},
                "action_value_targets": [],
                "safe_action_set": ["stay"],
                "target_classification": "unique_robust_winner",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    branches = [
        _branch_result(
            source_path=source_path,
            line_number=1,
            seed=5,
            agent_id=2,
            tick=10,
            source_record=source_records[0],
            best_action="eat",
        ),
        _branch_result(
            source_path=source_path,
            line_number=2,
            seed=13,
            agent_id=3,
            tick=11,
            source_record=source_records[1],
            best_action="move_north",
        ),
    ]
    _write_report_with_exact_digest(
        v164_path,
        {
            "schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_SCHEMA_VERSION
            ),
            "policy": M3_CARRION_SURVIVOR_CONTINUATION_V164_PRETERMINAL_TIED_SET_BRANCH_TARGET_EXPANSION_POLICY,
            "classification": {
                "primary": EXPECTED_V164_CLASSIFICATION,
                "labels": [EXPECTED_V164_CLASSIFICATION],
            },
            "branch_materialization": {
                "passed": True,
                "exact_materialization_proven": True,
                "materialized_branch_point_count": 2,
                "materialization_failure_count": 0,
            },
            "outcome_support": {
                "all_replays_verified": True,
                "candidate_run_count": 4,
                "preterminal_target_support_noncollapsed": True,
            },
            "branch_results": branches,
            "diagnostics_only": True,
            "training_ran": False,
            "artifact_created": False,
            "runtime_artifact_created": False,
            "live_ab_allowed": False,
            "promotion_authorized": False,
            "runtime_action_selection_changed": False,
            "runtime_override_path_created": False,
            "threshold_tuning_recommended": False,
            "lifecycle_proof": {
                "runtime_action_selection_changed": False,
                "runtime_artifact_created": False,
                "live_ab_allowed": False,
                "promotion_authorized": False,
                "runtime_override_path_created": False,
                "threshold_tuning_recommended": False,
            },
        },
    )
    return v164_path, base_path, source_records


def _source_record(*, tick: int, agent_id: int) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": "stay",
        "resolved_action": "stay",
        "action_mask": {
            "stay": True,
            "eat": True,
            "move_north": True,
            "move_south": True,
        },
        "observation_input": {
            "schema_version": "mind_observation_v3",
            "storage_encoding": "fixture_public_vector",
            "data": [0.1, 0.2, 0.3],
        },
        "observation_digest": f"obs-{tick}-{agent_id}",
        "before": {"x": 1, "y": 2},
    }


def _branch_result(
    *,
    source_path: Path,
    line_number: int,
    seed: int,
    agent_id: int,
    tick: int,
    source_record: dict[str, object],
    best_action: str,
) -> dict[str, object]:
    source_digest = stable_payload_digest(_record_materialization_payload(source_record))
    branch_id = f"branch-{seed}-{agent_id}"
    stay_run = _candidate_run(branch_id, seed=seed, agent_id=agent_id, tick=tick, action="stay", score=0.2)
    best_run = _candidate_run(branch_id, seed=seed, agent_id=agent_id, tick=tick, action=best_action, score=0.8)
    return {
        "branch_id": branch_id,
        "seed": seed,
        "fixture": "broad",
        "ticks": 120,
        "branch_tick": tick,
        "remaining_horizon": 120 - tick,
        "record_index": line_number - 1,
        "agent_id": agent_id,
        "source_path": str(source_path),
        "line_number": line_number,
        "runtime_requested_action": "stay",
        "runtime_resolved_action": "stay",
        "predicted_action": "stay",
        "nearest_neighbor_row_index": 7,
        "top_value_candidate_set": ["stay", best_action],
        "source_record_digest": source_digest,
        "materialized_record_digest": source_digest,
        "branch_state_digest": f"state-{branch_id}",
        "replay_verification_passed": True,
        "candidate_runs": [stay_run, best_run],
    }


def _candidate_run(
    branch_id: str,
    *,
    seed: int,
    agent_id: int,
    tick: int,
    action: str,
    score: float,
) -> dict[str, object]:
    digest = f"digest-{branch_id}-{action}"
    return {
        "branch_id": branch_id,
        "seed": seed,
        "agent_id": agent_id,
        "branch_tick": tick,
        "forced_action": action,
        "forced_action_used": True,
        "forced_action_supported": True,
        "target_terminal": {
            "alive": True,
            "energy_ratio": score,
            "hydration_ratio": score,
            "health_ratio": score,
        },
        "target_alive_at_end": True,
        "target_energy_ratio_at_end": score,
        "target_hydration_ratio_at_end": score,
        "target_health_ratio_at_end": score,
        "alive_agents": 5,
        "births": 1,
        "deaths": 0,
        "unsupported_requested_action_count": 0,
        "terminal_population_delta_vs_reference": {
            "alive_agents": 0,
            "births": 0,
            "deaths": 0,
            "unsupported_requested_action_count": 0,
        },
        "target_local_delta_vs_reference": {
            "target_alive": 0,
            "target_energy_ratio": score,
            "target_hydration_ratio": score,
            "target_health_ratio": score,
        },
        "first_action_outcome_summary": {
            "forced_action": action,
            "requested_action": action,
            "resolved_action": action,
            "matches_forced_action": True,
            "action_valid": True,
            "resolution_action_valid": True,
            "resource_gain": 0.0,
        },
        "replay_digest": digest,
        "replay_verification": {
            "verified": True,
            "expected_digest": digest,
            "actual_digest": digest,
        },
    }


if __name__ == "__main__":
    unittest.main()
