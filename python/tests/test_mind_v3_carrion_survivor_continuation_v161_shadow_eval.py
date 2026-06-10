from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    run_carrion_survivor_continuation_v160_action_value_scorer,
)
from evolution_sim.mind.carrion_survivor_continuation_v161_shadow_eval import (
    M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_SCHEMA_VERSION,
    run_carrion_survivor_continuation_v161_shadow_eval,
    summarize_shadow_predictions,
)

try:
    from python.tests.test_mind_v3_carrion_survivor_continuation_v160_action_value_scorer import (
        _load_jsonl,
        _write_ready_v159_inputs,
    )
except ModuleNotFoundError:  # pragma: no cover - unittest discovery fallback.
    from test_mind_v3_carrion_survivor_continuation_v160_action_value_scorer import (
        _load_jsonl,
        _write_ready_v159_inputs,
    )

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV161ShadowEvalTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v161-shadow-eval"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v161_shadow_eval"
            ),
        )

    def test_digest_and_contract_validation_failure_blocks_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v160_inputs(tmpdir)
            artifact = json.loads(paths["v160_artifact"].read_text(encoding="utf-8"))
            artifact["contract"]["runtime_artifact"] = True
            paths["v160_artifact"].write_text(
                json.dumps(artifact, sort_keys=True),
                encoding="utf-8",
            )
            evidence_path = _write_shadow_evidence(tmpdir, paths)

            result = run_carrion_survivor_continuation_v161_shadow_eval(
                v160_report_path=paths["v160_report"],
                v160_artifact_path=paths["v160_artifact"],
                trajectory_paths=[evidence_path],
                output_path=paths["v161_report"],
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v160_artifact_contract_invalid",
            result["source_validation"]["failures"],
        )
        self.assertIn(
            "v160_artifact_exact_digest_mismatch",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v161_shadow_eval_"
                "source_invalid_closed_no_live_ab"
            ),
        )
        self.assertFalse(result["shadow_eval_ran"])
        self.assertEqual(result["shadow_evaluation"]["prediction_count"], 0)
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_lifecycle_fields_remain_diagnostics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v160_inputs(tmpdir)
            evidence_path = _write_shadow_evidence(tmpdir, paths)

            result = run_carrion_survivor_continuation_v161_shadow_eval(
                v160_report_path=paths["v160_report"],
                v160_artifact_path=paths["v160_artifact"],
                trajectory_paths=[evidence_path],
                output_path=paths["v161_report"],
            )

        contract = result["contract"]
        self.assertTrue(contract["diagnostics_only"])
        self.assertTrue(contract["shadow_eval_only"])
        self.assertFalse(contract["runtime_policy_integration_allowed"])
        self.assertFalse(contract["live_runtime_override_allowed"])
        self.assertFalse(contract["promotion_authorized"])
        self.assertFalse(contract["runtime_action_selection_changed"])
        self.assertFalse(result["runtime_artifact_created"])
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["runtime_promotion_allowed"])
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_unsupported_prediction_and_action_distribution_summary(self) -> None:
        summary = summarize_shadow_predictions(
            [
                {
                    "source_seed": 5,
                    "predicted_action": "eat",
                    "supported_prediction": False,
                    "unsupported_prediction": True,
                    "actual_runtime_requested_action_available": True,
                    "actual_runtime_requested_action": "stay",
                    "would_change_action": True,
                    "score_source": "forced",
                    "feature_source": "test",
                },
                {
                    "source_seed": 5,
                    "predicted_action": "stay",
                    "supported_prediction": True,
                    "unsupported_prediction": False,
                    "actual_runtime_requested_action_available": True,
                    "actual_runtime_requested_action": "stay",
                    "would_change_action": False,
                    "score_source": "forced",
                    "feature_source": "test",
                },
            ],
            decision_record_count=2,
        )

        self.assertEqual(summary["unsupported_shadow_prediction_count"], 1)
        self.assertEqual(summary["predicted_action_counts"], {"eat": 1, "stay": 1})
        self.assertEqual(summary["dominant_predicted_action_share"], 0.5)
        self.assertTrue(summary["predicted_action_distribution_noncollapsed"])
        self.assertEqual(summary["would_change_count"], 1)
        self.assertEqual(summary["would_change_share"], 0.5)

    def test_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v160_inputs(tmpdir)
            evidence_path = _write_shadow_evidence(tmpdir, paths)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v161_shadow_eval",
                    "--v160-report",
                    str(paths["v160_report"]),
                    "--v160-artifact",
                    str(paths["v160_artifact"]),
                    "--trajectory",
                    str(evidence_path),
                    "--output",
                    str(paths["v161_report"]),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v161_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v161_shadow_eval_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V161_SHADOW_EVAL_SCHEMA_VERSION,
        )
        self.assertEqual(written["source_validation"]["passed"], True)
        self.assertEqual(
            written["shadow_evaluation"]["unsupported_shadow_prediction_count"],
            0,
        )
        self.assertIn("exact_digest", written)

    def test_no_runtime_action_selection_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_ready_v160_inputs(tmpdir)
            evidence_path = _write_shadow_evidence(tmpdir, paths)
            before = evidence_path.read_text(encoding="utf-8")

            result = run_carrion_survivor_continuation_v161_shadow_eval(
                v160_report_path=paths["v160_report"],
                v160_artifact_path=paths["v160_artifact"],
                trajectory_paths=[evidence_path],
                output_path=paths["v161_report"],
            )
            after = evidence_path.read_text(encoding="utf-8")

        shadow = result["shadow_evaluation"]
        self.assertEqual(before, after)
        self.assertFalse(result["runtime_action_selection_changed"])
        self.assertFalse(shadow["runtime_action_selection_changed"])
        self.assertEqual(
            shadow["runtime_requested_action_sequence_digest_before_shadow"],
            shadow["runtime_requested_action_sequence_digest_after_shadow"],
        )
        self.assertTrue(shadow["runtime_requested_action_sequence_digest_preserved"])


def _write_ready_v160_inputs(tmpdir: str) -> dict[str, Path]:
    paths = _write_ready_v159_inputs(tmpdir, mode="passing_loo")
    paths["v161_report"] = Path(tmpdir) / "v161-report.json"
    run_carrion_survivor_continuation_v160_action_value_scorer(
        v158_dataset_path=paths["v158_dataset"],
        v159_report_path=paths["v159_report"],
        output_path=paths["v160_report"],
        artifact_output_path=paths["v160_artifact"],
    )
    return paths


def _write_shadow_evidence(tmpdir: str, paths: Mapping[str, Path]) -> Path:
    rows = _load_jsonl(paths["v158_dataset"])
    evidence_path = Path(tmpdir) / "shadow-evidence.jsonl"
    payloads = []
    requested_cycle = ("stay", "eat")
    for index, row in enumerate(rows):
        record = {
            "source_seed": 5 if index % 2 == 0 else 13,
            "tick": index,
            "agent_id": index + 1,
            "trainable_public_features": deepcopy(row["trainable_public_features"]),
            "public_action_mask": deepcopy(row["public_action_mask"]),
            "action_mask": deepcopy(row["public_action_mask"]),
            "requested_action": requested_cycle[index % len(requested_cycle)],
            "action_source": "mind_v3_autonomous_evolution_policy_v1",
        }
        payloads.append({"record": record})
    evidence_path.write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + "\n",
        encoding="utf-8",
    )
    return evidence_path


if __name__ == "__main__":
    unittest.main()
