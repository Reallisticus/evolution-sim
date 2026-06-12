from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v184_v183_transition_row_dataset_audit as v184,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v185_v183_target_resolution_repair as v185,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    _report_exact_digest,
    _transition_row,
)
from python.tests.test_mind_v3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit import (
    _patched_v183_digests,
    _write_v183_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV185V183TargetResolutionRepairTests(
    unittest.TestCase
):
    def test_npm_entrypoints_exist(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v185-v183-target-resolution-repair"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v185_v183_target_resolution_repair"
            ),
        )
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v185-repaired-transition-row-dataset-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit"
            ),
        )

    def test_strict_filter_repairs_v183_target_resolution_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, dataset_digest = _write_v184_failure_inputs(tmpdir)
            report = v185.run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
                v184_report_path=paths["v184_report"],
                v183_transition_dataset_path=paths["dataset"],
                output_path=paths["v185_report"],
                repaired_transition_dataset_output_path=paths["repaired_dataset"],
                expected_v184_report_exact_digest=report_digest,
                expected_v183_report_exact_digest=_report_exact_digest(
                    paths["v183_report"]
                ),
                expected_v183_dataset_digest=dataset_digest,
            )
            repaired_rows = _read_jsonl(paths["repaired_dataset"])

        self.assertEqual(
            report["classification"]["primary"],
            v185.V185_REPAIR_READY_CLASSIFICATION,
        )
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["repair_validation"]["passed"])
        self.assertEqual(report["repair_validation"]["input_row_count"], 144)
        self.assertEqual(report["repair_validation"]["invalid_input_row_count"], 7)
        self.assertEqual(
            report["repair_validation"]["removed_or_replaced_invalid_row_count"],
            7,
        )
        self.assertEqual(report["repair_validation"]["repaired_row_count"], 137)
        self.assertFalse(report["repair_validation"]["backfill_used"])
        self.assertEqual(
            report["repair_validation"]["pre_repair_target_audit"]["failure_count"],
            14,
        )
        self.assertTrue(
            report["repair_validation"]["repaired_target_audit"]["passed"]
        )
        self.assertTrue(report["support_summary"]["passed"])
        self.assertEqual(report["support_summary"]["observed"]["row_count"], 137)
        self.assertEqual(
            report["route_recommendation"]["recommended_next_route"],
            "v185_repaired_transition_row_dataset_audit_before_slice_2_training",
        )
        self.assertEqual(report["dataset"]["dataset_digest"], stable_payload_digest(repaired_rows))
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_2_training_consumed"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        for row in repaired_rows:
            summary = row["short_horizon_public_outcome_summary"]
            self.assertEqual(summary["current_resolved_action"], row["forced_action"])
            self.assertTrue(summary["current_resolution_action_valid"])
            self.assertNotIn(
                "current_resolution_action_valid",
                row["trainable_public_features"],
            )
            self.assertIn(
                "current_public_action_mask",
                row["trainable_public_features"],
            )
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_v178_accepts_pinned_v185_source_and_routes_to_v186_slice_2(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, dataset_digest = _write_v184_failure_inputs(tmpdir)
            repair = v185.run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
                v184_report_path=paths["v184_report"],
                v183_transition_dataset_path=paths["dataset"],
                output_path=paths["v185_report"],
                repaired_transition_dataset_output_path=paths["repaired_dataset"],
                expected_v184_report_exact_digest=report_digest,
                expected_v183_report_exact_digest=_report_exact_digest(
                    paths["v183_report"]
                ),
                expected_v183_dataset_digest=dataset_digest,
            )
            repaired_digest = str(repair["dataset"]["dataset_digest"])
            audit = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
                transition_dataset_path=paths["repaired_dataset"],
                v177_report_path=paths["v185_report"],
                output_path=paths["v178_audit"],
                expected_v177_report_exact_digest=str(repair["exact_digest"]),
                expected_dataset_digest=repaired_digest,
            )

        self.assertTrue(audit["source_validation"]["passed"])
        self.assertEqual(
            audit["source_validation"]["source_producer"],
            v178.V185_SOURCE_PRODUCER,
        )
        self.assertTrue(
            audit["source_validation"]["v185_repair_validation"]["passed"]
        )
        self.assertTrue(audit["target_audit"]["passed"])
        self.assertTrue(audit["training_authorization"]["authorized"])
        self.assertEqual(
            audit["route_recommendation"]["recommended_next_route"],
            v178.V186_SLICE_2_TRAINING_ROUTE,
        )
        self.assertTrue(
            audit["route_recommendation"]["slice_2_opt_in_training_route_authorized"]
        )
        self.assertFalse(
            audit["route_recommendation"]["first_opt_in_training_slice_authorized"]
        )
        self.assertFalse(audit["training_authorized"])
        self.assertFalse(audit["training_ran"])
        self.assertFalse(audit["runtime_artifact_created"])
        self.assertFalse(audit["runtime_action_selection_changed"])
        self.assertFalse(audit["promotion_authorized"])

    def test_v185_repaired_audit_authorizes_only_future_route_without_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, dataset_digest = _write_v184_failure_inputs(tmpdir)
            repair = v185.run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
                v184_report_path=paths["v184_report"],
                v183_transition_dataset_path=paths["dataset"],
                output_path=paths["v185_report"],
                repaired_transition_dataset_output_path=paths["repaired_dataset"],
                expected_v184_report_exact_digest=report_digest,
                expected_v183_report_exact_digest=_report_exact_digest(
                    paths["v183_report"]
                ),
                expected_v183_dataset_digest=dataset_digest,
            )
            audit = v185.run_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit(
                v185_report_path=paths["v185_report"],
                transition_dataset_path=paths["repaired_dataset"],
                output_path=paths["v185_audit"],
                expected_v185_report_exact_digest=str(repair["exact_digest"]),
                expected_dataset_digest=str(repair["dataset"]["dataset_digest"]),
            )

        self.assertEqual(
            audit["schema_version"],
            v185.M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(
            audit["classification"]["primary"],
            v185.V185_AUDIT_AUTHORIZED_CLASSIFICATION,
        )
        self.assertEqual(
            audit["source_validation"]["source_producer"],
            v178.V185_SOURCE_PRODUCER,
        )
        self.assertTrue(audit["target_audit"]["passed"])
        self.assertEqual(audit["target_audit"]["failure_count"], 0)
        self.assertEqual(
            audit["route_recommendation"]["recommended_next_route"],
            v178.V186_SLICE_2_TRAINING_ROUTE,
        )
        self.assertTrue(
            audit["route_recommendation"]["slice_2_opt_in_training_route_authorized"]
        )
        self.assertFalse(
            audit["route_recommendation"]["first_opt_in_training_slice_authorized"]
        )
        self.assertFalse(audit["training_authorized"])
        self.assertFalse(audit["training_ran"])
        self.assertFalse(audit["training_artifact_created"])
        self.assertFalse(audit["slice_2_training_consumed"])
        self.assertFalse(audit["runtime_artifact_created"])
        self.assertFalse(audit["runtime_action_selection_changed"])
        self.assertFalse(audit["promotion_authorized"])
        self.assertTrue(exact_digest_validation_report(audit)["passed"])

    def test_v185_repaired_audit_requires_explicit_digest_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, dataset_digest = _write_v184_failure_inputs(tmpdir)
            v185.run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
                v184_report_path=paths["v184_report"],
                v183_transition_dataset_path=paths["dataset"],
                output_path=paths["v185_report"],
                repaired_transition_dataset_output_path=paths["repaired_dataset"],
                expected_v184_report_exact_digest=report_digest,
                expected_v183_report_exact_digest=_report_exact_digest(
                    paths["v183_report"]
                ),
                expected_v183_dataset_digest=dataset_digest,
            )
            audit = v185.run_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit(
                v185_report_path=paths["v185_report"],
                transition_dataset_path=paths["repaired_dataset"],
                output_path=paths["v185_audit"],
            )

        self.assertTrue(audit["source_validation"]["passed"])
        self.assertTrue(audit["target_audit"]["passed"])
        self.assertFalse(audit["training_authorization"]["authorized"])
        self.assertEqual(
            audit["classification"]["primary"],
            v185.V185_AUDIT_DIGEST_PINS_REQUIRED_CLASSIFICATION,
        )
        self.assertFalse(
            audit["route_recommendation"]["slice_2_opt_in_training_route_authorized"]
        )
        self.assertFalse(audit["training_ran"])

    def test_v185_cli_writes_repair_and_audit_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, report_digest, dataset_digest = _write_v184_failure_inputs(tmpdir)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"
            repair_cli = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v185_v183_target_resolution_repair",
                    "--v184-report",
                    str(paths["v184_report"]),
                    "--v183-transition-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(paths["v185_report"]),
                    "--repaired-transition-dataset-output",
                    str(paths["repaired_dataset"]),
                    "--expected-v184-report-exact-digest",
                    report_digest,
                    "--expected-v183-report-exact-digest",
                    _report_exact_digest(paths["v183_report"]),
                    "--expected-v183-dataset-digest",
                    dataset_digest,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            repair = json.loads(paths["v185_report"].read_text(encoding="utf-8"))
            audit_cli = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit",
                    "--v185-report",
                    str(paths["v185_report"]),
                    "--transition-dataset",
                    str(paths["repaired_dataset"]),
                    "--output",
                    str(paths["v185_audit"]),
                    "--expected-v185-report-exact-digest",
                    str(repair["exact_digest"]),
                    "--expected-dataset-digest",
                    str(repair["dataset"]["dataset_digest"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            audit = json.loads(paths["v185_audit"].read_text(encoding="utf-8"))

        self.assertIn("invalid_input_row_count=7", repair_cli.stdout)
        self.assertIn("repaired_row_count=137", repair_cli.stdout)
        self.assertIn("slice_2_training_authorized=False", repair_cli.stdout)
        self.assertIn(
            f"recommended_next_route={v178.V186_SLICE_2_TRAINING_ROUTE}",
            audit_cli.stdout,
        )
        self.assertIn("target_audit_failure_count=0", audit_cli.stdout)
        self.assertFalse(audit["training_ran"])
        self.assertFalse(audit["slice_2_training_consumed"])


def _write_v184_failure_inputs(
    tmpdir: str,
) -> tuple[dict[str, Path], str, str]:
    rows = _v183_rows_with_target_resolution_failures()
    paths, rows = _write_v183_inputs(tmpdir, rows)
    paths.update(
        {
            "v185_report": Path(tmpdir) / "v185-repair.json",
            "repaired_dataset": Path(tmpdir) / "v185-repaired.jsonl",
            "v185_audit": Path(tmpdir) / "v185-audit.json",
            "v178_audit": Path(tmpdir) / "v178-audit.json",
        }
    )
    v183_report_digest = _report_exact_digest(paths["v183_report"])
    dataset_digest = stable_payload_digest(rows)
    with _patched_v183_digests(v183_report_digest, dataset_digest):
        v184_report = (
            v184.run_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit(
                v183_report_path=paths["v183_report"],
                transition_dataset_path=paths["dataset"],
                output_path=paths["v184_report"],
                expected_v183_report_exact_digest=v183_report_digest,
                expected_dataset_digest=dataset_digest,
            )
        )
    return paths, str(v184_report["exact_digest"]), dataset_digest


def _v183_rows_with_target_resolution_failures() -> list[dict[str, object]]:
    seeds = [101, 103, 107, 109, 113, 127]
    actions = list(ACTION_NAMES[:9])
    rows = [
        _transition_row(
            branch_id=f"branch-{index:03d}",
            seed=seeds[index % len(seeds)],
            forced_action=actions[index % len(actions)],
            value_offset=0.01 + index * 0.0001,
        )
        for index in range(144)
    ]
    bad_actions = [
        "attack_east",
        "attack_east",
        "attack_west",
        "attack_west",
        "attack_west",
        "attack_west",
        "attack_east",
    ]
    for index, action in enumerate(bad_actions):
        row = rows[index]
        row["forced_action"] = action
        row["trainable_public_features"]["forced_action"] = action
        summary = row["short_horizon_public_outcome_summary"]
        summary["current_requested_action"] = action
        summary["current_resolved_action"] = "stay"
        summary["current_action_valid"] = True
        summary["current_resolution_action_valid"] = False
        summary["forced_action_used"] = True
        summary["current_moved"] = False
    return rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
