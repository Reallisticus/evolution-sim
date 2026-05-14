from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import mind_v3_progress_ledger
from evolution_sim.mind.v3_progress_ledger import (
    MIND_V3_PROGRESS_RULE_VERSION,
    MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION,
    build_standard_progress_ledger_report,
)


class MindV3ProgressLedgerTests(unittest.TestCase):
    def test_progress_ledger_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:progress-ledger"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_progress_ledger"
            ),
        )

    def test_progress_ledger_writes_one_row_per_slice(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "mind-v3.md"
            legacy = tmp / "ledger.jsonl"
            output_dir = tmp / "output"
            output_dir.mkdir()
            docs.write_text(
                "\n".join(
                    [
                        "v1 documented only.",
                        "v2 clears a support floor.",
                        "Mind v3 should not be counted as slice evidence.",
                    ]
                ),
                encoding="utf-8",
            )
            legacy.write_text(
                json.dumps(
                    {
                        "schema_version": "legacy",
                        "candidate_artifact": "output/mind/mind-v3-v1-x.json",
                        "promotion_candidate_passed": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "mind-v3-v2-diagnostic.json").write_text(
                json.dumps(
                    {
                        "support_probes": {
                            "unit_probe": {
                                "policy": "unit",
                                "best_accuracy": 0.6,
                                "material_support_accuracy_floor": 0.55,
                                "materially_supports_unit_probe": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_standard_progress_ledger_report(
                docs_path=docs,
                legacy_ledger_path=legacy,
                output_dir=output_dir,
                start_version=1,
                through_version=3,
            )

        self.assertEqual(
            report["schema_version"],
            MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["contract"]["progress_rule_version"],  # type: ignore[index]
            MIND_V3_PROGRESS_RULE_VERSION,
        )
        rows = report["rows"]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["version"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["status"], "strict_policy_fail")
        self.assertEqual(rows[1]["status"], "diagnostic_support_floor_pass")
        self.assertTrue(rows[1]["progress_passed"])
        self.assertEqual(rows[2]["status"], "missing_evidence")

    def test_progress_ledger_prefers_authoritative_version_probe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "docs.md"
            legacy = tmp / "ledger.jsonl"
            output_dir = tmp / "output"
            output_dir.mkdir()
            docs.write_text("v93 documented.\n", encoding="utf-8")
            legacy.write_text("", encoding="utf-8")
            (output_dir / "mind-v3-v93-labels.json").write_text(
                json.dumps(
                    {
                        "support_probes": {
                            "generic_label_probe": {
                                "policy": "generic",
                                "accuracy": 0.8,
                                "support_accuracy_floor": 0.5,
                                "materially_supports_generic_label_probe": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "mind-v3-v93-depleted-resource-trap-audit.json").write_text(
                json.dumps(
                    {
                        "depleted_resource_trap_support_probe": {
                            "policy": "v93_depleted_resource_trap_support_v1",
                            "accuracy": 0.0,
                            "support_accuracy_floor": 1.0,
                            "materially_supports_depleted_resource_trap": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_standard_progress_ledger_report(
                docs_path=docs,
                legacy_ledger_path=legacy,
                output_dir=output_dir,
                start_version=93,
                through_version=93,
            )

        row = report["rows"][0]
        self.assertEqual(row["status"], "diagnostic_support_floor_fail")
        self.assertFalse(row["progress_passed"])
        self.assertEqual(
            row["best_support_probe"]["path"],
            "$.depleted_resource_trap_support_probe",
        )

    def test_progress_ledger_treats_v94_sequence_probe_as_authoritative(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "docs.md"
            legacy = tmp / "ledger.jsonl"
            output_dir = tmp / "output"
            output_dir.mkdir()
            docs.write_text("v94 documented.\n", encoding="utf-8")
            legacy.write_text("", encoding="utf-8")
            (output_dir / "mind-v3-v94-generic.json").write_text(
                json.dumps(
                    {
                        "support_probes": {
                            "generic_probe": {
                                "policy": "generic",
                                "accuracy": 0.9,
                                "support_accuracy_floor": 0.5,
                                "materially_supports_generic_probe": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "mind-v3-v94-branch-sequence-continuation-scorer.json").write_text(
                json.dumps(
                    {
                        "sequence_continuation_support_probe": {
                            "policy": "v94_sequence_world_model_branch_continuation_v1",
                            "accuracy": 0.0,
                            "support_accuracy_floor": 1.0,
                            "materially_supports_sequence_continuation_scorer": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_standard_progress_ledger_report(
                docs_path=docs,
                legacy_ledger_path=legacy,
                output_dir=output_dir,
                start_version=94,
                through_version=94,
            )

        row = report["rows"][0]
        self.assertEqual(row["status"], "diagnostic_support_floor_fail")
        self.assertFalse(row["progress_passed"])
        self.assertEqual(
            row["best_support_probe"]["path"],
            "$.sequence_continuation_support_probe",
        )

    def test_progress_ledger_treats_v95_planning_probe_as_authoritative(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "docs.md"
            legacy = tmp / "ledger.jsonl"
            output_dir = tmp / "output"
            output_dir.mkdir()
            docs.write_text("v95 documented.\n", encoding="utf-8")
            legacy.write_text("", encoding="utf-8")
            (output_dir / "mind-v3-v95-generic.json").write_text(
                json.dumps(
                    {
                        "support_probes": {
                            "generic_probe": {
                                "policy": "generic",
                                "accuracy": 0.0,
                                "support_accuracy_floor": 1.0,
                                "materially_supports_generic_probe": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "mind-v3-v95-branch-constrained-planning-audit.json").write_text(
                json.dumps(
                    {
                        "constrained_planning_support_probe": {
                            "policy": "v95_simulator_in_loop_constrained_planning_v1",
                            "accuracy": 1.0,
                            "support_accuracy_floor": 1.0,
                            "materially_supports_constrained_planning": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_standard_progress_ledger_report(
                docs_path=docs,
                legacy_ledger_path=legacy,
                output_dir=output_dir,
                start_version=95,
                through_version=95,
            )

        row = report["rows"][0]
        self.assertEqual(row["status"], "diagnostic_support_floor_pass")
        self.assertTrue(row["progress_passed"])
        self.assertEqual(
            row["best_support_probe"]["path"],
            "$.constrained_planning_support_probe",
        )

    def test_progress_ledger_treats_v96_distillation_probe_as_authoritative(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "docs.md"
            legacy = tmp / "ledger.jsonl"
            output_dir = tmp / "output"
            output_dir.mkdir()
            docs.write_text("v96 documented.\n", encoding="utf-8")
            legacy.write_text("", encoding="utf-8")
            (output_dir / "mind-v3-v96-generic.json").write_text(
                json.dumps(
                    {
                        "support_probes": {
                            "generic_probe": {
                                "policy": "generic",
                                "accuracy": 0.0,
                                "support_accuracy_floor": 1.0,
                                "materially_supports_generic_probe": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (
                output_dir
                / "mind-v3-v96-planner-distillation-runtime-feasibility.json"
            ).write_text(
                json.dumps(
                    {
                        "planner_distillation_runtime_feasibility_support_probe": {
                            "policy": (
                                "v96_planner_distillation_runtime_feasibility_v1"
                            ),
                            "accuracy": 1.0,
                            "support_accuracy_floor": 1.0,
                            "materially_supports_runtime_feasible_planner_distillation": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_standard_progress_ledger_report(
                docs_path=docs,
                legacy_ledger_path=legacy,
                output_dir=output_dir,
                start_version=96,
                through_version=96,
            )

        row = report["rows"][0]
        self.assertEqual(row["status"], "diagnostic_support_floor_pass")
        self.assertTrue(row["progress_passed"])
        self.assertEqual(
            row["best_support_probe"]["path"],
            "$.planner_distillation_runtime_feasibility_support_probe",
        )

    def test_progress_ledger_treats_v98_residual_probe_as_authoritative(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "docs.md"
            legacy = tmp / "ledger.jsonl"
            output_dir = tmp / "output"
            output_dir.mkdir()
            docs.write_text("v98 documented.\n", encoding="utf-8")
            legacy.write_text("", encoding="utf-8")
            (output_dir / "mind-v3-v98-broad-transfer-residual-audit.json").write_text(
                json.dumps(
                    {
                        "broad_transfer_residual_support_probe": {
                            "policy": "v98_support_gated_residual",
                            "accuracy": 1.0,
                            "support_accuracy_floor": 1.0,
                            "materially_supports_v99_residual": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_standard_progress_ledger_report(
                docs_path=docs,
                legacy_ledger_path=legacy,
                output_dir=output_dir,
                start_version=98,
                through_version=98,
            )

        row = report["rows"][0]
        self.assertEqual(row["status"], "diagnostic_support_floor_pass")
        self.assertTrue(row["progress_passed"])
        self.assertEqual(
            row["best_support_probe"]["path"],
            "$.broad_transfer_residual_support_probe",
        )

    def test_progress_ledger_cli_writes_jsonl_and_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            docs = tmp / "docs.md"
            legacy = tmp / "legacy.jsonl"
            output_dir = tmp / "output"
            jsonl = tmp / "progress.jsonl"
            summary = tmp / "summary.json"
            output_dir.mkdir()
            docs.write_text("v1 documented.\n", encoding="utf-8")
            legacy.write_text("", encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "mind_v3_progress_ledger",
                    "--docs",
                    str(docs),
                    "--legacy-ledger",
                    str(legacy),
                    "--output-dir",
                    str(output_dir),
                    "--through-version",
                    "1",
                    "--output",
                    str(jsonl),
                    "--summary-output",
                    str(summary),
                ],
            ):
                mind_v3_progress_ledger.main()

            rows = [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
            ]
            payload = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["version"], 1)
        self.assertEqual(payload["summary"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
