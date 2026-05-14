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
