from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind import (
    carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit as v194,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training as v195,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit import (
    _attach_digest,
    _expected_aggregate,
    _read_jsonl,
    _record,
    _run_audit,
    _v193_report,
    _write_trajectory,
)
from python.tests.test_mind_v3_carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    _encoded_observation,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV195TrainingTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v195-repaired-contract-terminal-survival-support-training"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training"
            ),
        )

    def test_valid_pinned_v194_dataset_trains_slice_3_artifact_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, audit, rows = _authorized_v194_inputs(tmpdir)
            artifact_path = paths["root"] / "v195-artifact.json"
            report = v195.run_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training(
                authorization_report_path=paths["v194_report"],
                compact_support_dataset_path=paths["compact_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v195-report.json",
                expected_authorization_report_exact_digest=str(audit["exact_digest"]),
                expected_dataset_digest=stable_payload_digest(rows),
                expected_v193_report_exact_digest=audit["source_validation"][
                    "observed_v193_report_exact_digest"
                ],
                expected_v192_report_exact_digest="v192",
                expected_dataset_row_count=len(rows),
                expected_selected_same_tick_occupancy_drift_count=1,
                backup_metadata_override={"passed": True},
                run_evaluation=False,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertTrue(report["authorization_report_validation"]["passed"])
        self.assertTrue(report["source_validation"]["passed"])
        self.assertTrue(report["training"]["ran"])
        self.assertTrue(report["training"]["passed"])
        self.assertTrue(report["training_ran"])
        self.assertTrue(report["training_artifact_created"])
        self.assertTrue(report["slice_3_training_consumed"])
        self.assertEqual(report["training_slice_budget"]["current_slices_consumed"], 3)
        self.assertEqual(report["artifact"]["digest"], stable_payload_digest(artifact))
        self.assertEqual(
            artifact["artifact_policy"],
            v195.M3_CARRION_SURVIVOR_CONTINUATION_V195_ARTIFACT_POLICY,
        )
        self.assertEqual(
            artifact["built_from"]["source"],
            "v194_repaired_contract_terminal_survival_support_compact_dataset",
        )
        self.assertEqual(
            artifact["built_from"]["authorization_route"],
            v194.SUCCESS_ROUTE,
        )
        self.assertEqual(artifact["built_from"]["training_slice_index"], 3)
        self.assertTrue(artifact["built_from"]["slice_3_opt_in_training_slice"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["gate_relaxation_allowed"])
        self.assertFalse(report["shadow_eval_ran"])
        self.assertEqual(
            report["route_decision"]["recommended_next_route"],
            (
                "v196_repaired_contract_slice_3_shadow_evaluation_not_run_repair_no_training"
            ),
        )
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_authorization_report_digest_mismatch_fails_closed_without_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, audit, rows = _authorized_v194_inputs(tmpdir)
            artifact_path = paths["root"] / "v195-artifact.json"
            report = v195.run_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training(
                authorization_report_path=paths["v194_report"],
                compact_support_dataset_path=paths["compact_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v195-report.json",
                expected_authorization_report_exact_digest="wrong",
                expected_dataset_digest=stable_payload_digest(rows),
                expected_v193_report_exact_digest=audit["source_validation"][
                    "observed_v193_report_exact_digest"
                ],
                expected_v192_report_exact_digest="v192",
                expected_dataset_row_count=len(rows),
                expected_selected_same_tick_occupancy_drift_count=1,
                backup_metadata_override={"passed": True},
                run_evaluation=False,
            )

        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "expected_exact_digest_matches",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["source_validation"]["passed"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertFalse(report["artifact"]["created"])
        self.assertFalse(artifact_path.exists())

    def test_route_mismatch_fails_closed_even_with_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, audit, rows = _authorized_v194_inputs(tmpdir)
            bad_audit = dict(audit)
            bad_audit["route_decision"] = dict(audit["route_decision"])
            bad_audit["route_decision"]["recommended_next_route"] = (
                "v195_repaired_contract_terminal_survival_support_dataset_repair_no_training"
            )
            bad_audit["exact_digest"] = v194.digest_without_exact(bad_audit)
            bad_path = paths["root"] / "v194-wrong-route.json"
            bad_path.write_text(json.dumps(bad_audit, sort_keys=True, indent=2) + "\n")
            artifact_path = paths["root"] / "v195-artifact.json"
            report = v195.run_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training(
                authorization_report_path=bad_path,
                compact_support_dataset_path=paths["compact_dataset"],
                artifact_output_path=artifact_path,
                output_path=paths["root"] / "v195-report.json",
                expected_authorization_report_exact_digest=str(bad_audit["exact_digest"]),
                expected_dataset_digest=stable_payload_digest(rows),
                expected_v193_report_exact_digest=bad_audit["source_validation"][
                    "observed_v193_report_exact_digest"
                ],
                expected_v192_report_exact_digest="v192",
                expected_dataset_row_count=len(rows),
                expected_selected_same_tick_occupancy_drift_count=1,
                backup_metadata_override={"passed": True},
                run_evaluation=False,
            )

        self.assertTrue(report["authorization_report_validation"]["exact_digest_valid"])
        self.assertFalse(report["authorization_report_validation"]["passed"])
        self.assertIn(
            "route_matches_v195_slice_3_opt_in",
            report["authorization_report_validation"]["failures"],
        )
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["slice_3_training_consumed"])
        self.assertFalse(artifact_path.exists())

    def test_cli_writes_parseable_report_and_prints_slice_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, audit, rows = _authorized_v194_inputs(tmpdir)
            artifact_path = paths["root"] / "v195-artifact.json"
            output_path = paths["root"] / "v195-report.json"

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training",
                    "--authorization-report",
                    str(paths["v194_report"]),
                    "--compact-support-dataset",
                    str(paths["compact_dataset"]),
                    "--artifact-output",
                    str(artifact_path),
                    "--output",
                    str(output_path),
                    "--expected-authorization-report-exact-digest",
                    str(audit["exact_digest"]),
                    "--expected-dataset-digest",
                    stable_payload_digest(rows),
                    "--expected-v193-report-exact-digest",
                    str(audit["source_validation"]["observed_v193_report_exact_digest"]),
                    "--expected-v192-report-exact-digest",
                    "v192",
                    "--expected-dataset-row-count",
                    str(len(rows)),
                    "--expected-selected-same-tick-occupancy-drift-count",
                    "1",
                    "--skip-evaluation",
                    "--backup-doc",
                    str(paths["backup_doc"]),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("training_ran=True", result.stdout)
        self.assertIn("training_artifact_created=True", result.stdout)
        self.assertIn("slice_3_training_consumed=True", result.stdout)
        self.assertIn("current_slices_consumed=3", result.stdout)
        self.assertIn(f"required_route={v194.SUCCESS_ROUTE}", result.stdout)
        self.assertTrue(report["training_ran"])
        self.assertTrue(report["slice_3_training_consumed"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["promotion_authorized"])


def _authorized_v194_inputs(
    tmpdir: str,
) -> tuple[dict[str, Path], dict[str, object], list[dict[str, object]]]:
    root = Path(tmpdir)
    first = root / "seed13.jsonl.gz"
    second = root / "seed19.jsonl.gz"
    _write_trajectory(
        first,
        records=[
            _valid_record("move_north", resolution_valid=False, offset=0.01),
            _valid_record("eat", offset=0.02),
            _valid_record("stay", offset=0.03),
        ],
        alive_agents=2,
        births=4,
        deaths=1,
    )
    _write_trajectory(
        second,
        records=[
            _valid_record("drink", offset=0.04),
            _valid_record("eat", offset=0.05),
            _valid_record("stay", offset=0.06),
        ],
        alive_agents=3,
        births=5,
        deaths=1,
    )
    runs = [
        _run_audit(seed=13, trajectory_path=first, alive=2, births=4, deaths=1),
        _run_audit(seed=19, trajectory_path=second, alive=3, births=5, deaths=1),
    ]
    v193_report = _v193_report(runs)
    v193_digest = _attach_digest(v193_report)
    expected_by_seed = {
        "13": {
            "alive_agents": 2,
            "births": 4,
            "dominant_requested_action": "stay",
            "dominant_requested_action_share": 0.333333,
            "expected_same_tick_occupancy_drift_count": 1,
        },
        "19": {
            "alive_agents": 3,
            "births": 5,
            "dominant_requested_action": "stay",
            "dominant_requested_action_share": 0.333333,
            "expected_same_tick_occupancy_drift_count": 0,
        },
    }
    backup_doc = root / "backup.md"
    backup_doc.write_text(
        "\n".join(
            [
                v195.EXPECTED_V194_BACKUP,
                v195.EXPECTED_V194_BACKUP_SHA256,
                "rclone check reported 0 differences and 1 matching file",
            ]
        ),
        encoding="utf-8",
    )
    v194_report_path = root / "v194-report.json"
    compact_dataset_path = root / "dataset.jsonl"
    audit = (
        v194.run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit(
            v193_report_override=v193_report,
            output_path=v194_report_path,
            compact_support_dataset_output_path=compact_dataset_path,
            expected_v193_report_exact_digest=v193_digest,
            expected_v192_report_exact_digest="v192",
            expected_v191_report_exact_digest="v191",
            expected_v190_report_exact_digest="v190",
            expected_selected_support_by_seed=expected_by_seed,
            expected_aggregate_support=_expected_aggregate(runs),
            backup_metadata_override={"passed": True},
        )
    )
    v194_report_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    rows = _read_jsonl(compact_dataset_path)
    return (
        {
            "root": root,
            "v194_report": v194_report_path,
            "compact_dataset": compact_dataset_path,
            "backup_doc": backup_doc,
        },
        audit,
        rows,
    )


def _valid_record(
    action: str,
    *,
    resolution_valid: bool = True,
    offset: float = 0.0,
) -> dict[str, object]:
    record = _record(action, resolution_valid=resolution_valid)
    record["observation_input"] = _encoded_observation(offset)
    return record


if __name__ == "__main__":
    unittest.main()
