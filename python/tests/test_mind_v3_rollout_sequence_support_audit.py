from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.current_route_decision import (
    CURRENT_CLOSED_PATH,
    MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
    NEXT_ALLOWED_RESEARCH_DIRECTION,
)
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.rollout_sequence_support_audit import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
    build_rollout_sequence_support_audit_report,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3RolloutSequenceSupportAuditTests(unittest.TestCase):
    def test_rollout_sequence_support_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"]["sim:mind:v3:rollout-sequence-support-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_rollout_sequence_support_audit"
            ),
        )

    def test_synthetic_sequence_history_support_clears_floors(self) -> None:
        report = build_rollout_sequence_support_audit_report(
            v136_report=_v136_report(),
            v136_report_path=None,
            trajectory_datasets=(
                _dataset("synthetic-seed2.jsonl", ["eat", "drink"] * 4),
                _dataset("synthetic-seed5.jsonl", ["eat", "drink"] * 4),
            ),
        ).report

        self.assertEqual(
            report["schema_version"],
            MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["support_floors"]["passed"])
        self.assertIsNone(report["support_floors"]["first_failed_floor"])
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_support_ready_for_future_sequence_world_model_scorer",
        )
        comparisons = report["baseline_comparisons"]
        self.assertTrue(comparisons["beats_action_only_baseline"])
        self.assertTrue(comparisons["beats_action_order_baseline"])
        sequence = report["model_summaries"]["public_rollout_sequence_history_lookup"]
        self.assertEqual(sequence["unsupported_action_count"], 0)
        self.assertLessEqual(sequence["dominant_predicted_action_share"], 0.5)
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )
        self.assertFalse(
            report["authorization_block"]["first_recovery_public_context_ranker_reopened"]
        )

    def test_blocked_when_sequence_does_not_beat_action_only_baseline(self) -> None:
        report = build_rollout_sequence_support_audit_report(
            v136_report=_v136_report(),
            v136_report_path=None,
            trajectory_datasets=(
                _dataset("synthetic-seed2.jsonl", ["eat"] * 5),
                _dataset("synthetic-seed5.jsonl", ["eat"] * 5),
            ),
        ).report

        self.assertTrue(report["source_integrity"]["passed"])
        self.assertFalse(report["support_floors"]["passed"])
        self.assertEqual(
            report["support_floors"]["first_failed_floor"],
            "beats_action_only_baseline",
        )
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_support_blocked",
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])

    def test_v136_mismatch_fails_closed_without_authorizing_work(self) -> None:
        v136 = _v136_report()
        v136["next_allowed_research_direction"] = "first_recovery_feature_probe"

        report = build_rollout_sequence_support_audit_report(
            v136_report=v136,
            v136_report_path=None,
            trajectory_datasets=(
                _dataset("synthetic-seed2.jsonl", ["eat", "drink"] * 3),
                _dataset("synthetic-seed5.jsonl", ["eat", "drink"] * 3),
            ),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertIn(
            "v136_next_allowed_research_direction_mismatch",
            report["source_integrity"]["failures"],
        )
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_support_audit_source_integrity_failed",
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )

    def test_leakage_future_row_and_provenance_fields_are_rejected(self) -> None:
        train = _dataset("synthetic-seed2.jsonl", ["eat", "drink"] * 3)
        heldout_records = [
            _record(0, "eat") | {"future_rows": [{"requested_action": "drink"}]},
            _record(1, "drink") | {"provenance": {"seed": 5}},
        ]
        heldout = _dataset_from_records("synthetic-seed5.jsonl", heldout_records)

        report = build_rollout_sequence_support_audit_report(
            v136_report=_v136_report(),
            v136_report_path=None,
            trajectory_datasets=(train, heldout),
        ).report

        self.assertFalse(report["source_integrity"]["passed"])
        self.assertFalse(report["leakage_scan"]["passed"])
        self.assertEqual(report["leakage_scan"]["leakage_count"], 2)
        fields = {leak["field"] for leak in report["leakage_scan"]["leaks"]}
        self.assertIn("future_rows", fields)
        self.assertIn("provenance", fields)
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_support_audit_source_integrity_failed",
        )

    def test_report_path_inputs_extract_and_load_trajectory_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            first = tmp / "synthetic-seed2.jsonl"
            second = tmp / "synthetic-seed5.jsonl"
            _write_jsonl_trajectory(
                first,
                [_record(i, action) for i, action in enumerate(["eat", "drink"] * 3)],
            )
            _write_jsonl_trajectory(
                second,
                [_record(i, action) for i, action in enumerate(["eat", "drink"] * 3)],
            )
            report_path = tmp / "source-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "unit_report_v1",
                        "source_trajectories": {
                            "trajectory_paths": [str(first), str(second)]
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            report = build_rollout_sequence_support_audit_report(
                v136_report=_v136_report(),
                v136_report_path=None,
                report_paths=(report_path,),
            ).report

        evidence = report["source_evidence"]["trajectories"]
        self.assertEqual(evidence["loaded_path_count"], 2)
        self.assertEqual(
            evidence["report_path_evidence"]["extracted_trajectory_path_count"],
            2,
        )
        self.assertTrue(report["source_integrity"]["passed"])

    def test_opaque_path_footer_provenance_seed_is_held_out(self) -> None:
        report = build_rollout_sequence_support_audit_report(
            v136_report=_v136_report(),
            v136_report_path=None,
            trajectory_datasets=(
                _dataset("opaque-train.jsonl", ["eat", "drink"] * 3, seed=2),
                _dataset("opaque-heldout.jsonl", ["eat", "drink"] * 3, seed=5),
            ),
            heldout_fraction=0.0,
        ).report

        split = report["train_heldout_split"]
        self.assertIn("opaque-heldout.jsonl", split["heldout_sources_sample"])
        self.assertIn("opaque-train.jsonl", split["train_sources_sample"])
        self.assertEqual(split["configured_heldout_seeds"], [5, 13, 19, 29, 37, 41])
        self.assertEqual(split["configured_heldout_seed_leakage_count"], 0)
        self.assertEqual(split["strict_seed_leakage_count"], 0)
        self.assertTrue(report["source_integrity"]["passed"])

    def test_custom_heldout_seed_values_are_reported_and_used_for_leakage(
        self,
    ) -> None:
        report = build_rollout_sequence_support_audit_report(
            v136_report=_v136_report(),
            v136_report_path=None,
            trajectory_datasets=(
                _dataset("opaque-seed-two.jsonl", ["eat", "drink"] * 3, seed=2),
                _dataset("opaque-seed-five.jsonl", ["eat", "drink"] * 3, seed=5),
            ),
            heldout_seed_values=(2,),
            heldout_fraction=0.0,
        ).report

        split = report["train_heldout_split"]
        self.assertEqual(split["configured_heldout_seeds"], [2])
        self.assertEqual(split["heldout_seed_values"], [2])
        self.assertIn("opaque-seed-two.jsonl", split["heldout_sources_sample"])
        self.assertIn("opaque-seed-five.jsonl", split["train_sources_sample"])
        self.assertEqual(split["configured_heldout_seed_leakage_count"], 0)
        leakage_floor = [
            floor
            for floor in report["support_floors"]["floors"]
            if floor["name"] == "configured_heldout_seed_leakage_count_eq_0"
        ][0]
        self.assertTrue(leakage_floor["passed"])
        self.assertEqual(leakage_floor["observed"], 0)

    def test_no_input_cli_exits_nonzero_without_clobbering_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "v137.json"
            output.write_text("keep-me\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolution_sim.cli.mind_v3_rollout_sequence_support_audit",
                    "--output",
                    str(output),
                ],
                check=False,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least one --trajectory", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep-me\n")

    def test_trainable_feature_seed_or_future_tokens_are_rejected(self) -> None:
        train = _dataset(
            "synthetic-seed2.jsonl",
            ["eat", "drink"] * 3,
            extra_by_index={0: {"trainable_features": {"seed_identity": 2}}},
        )
        heldout = _dataset("synthetic-seed5.jsonl", ["eat", "drink"] * 3)

        report = build_rollout_sequence_support_audit_report(
            v136_report=_v136_report(),
            v136_report_path=None,
            trajectory_datasets=(train, heldout),
        ).report

        self.assertFalse(report["leakage_scan"]["passed"])
        self.assertEqual(
            report["leakage_scan"]["leaks"][0]["reason"],
            "forbidden_trainable_feature_token",
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])

    def test_real_artifact_numeric_v98_bank_metrics_when_outputs_exist(self) -> None:
        v136 = Path("output/mind/mind-v3-v136-current-route-decision.json")
        trajectories = sorted(
            Path("output/mind/v98-broad-support-trajectories").glob(
                "open-mind-v3-[0-9]*-120.jsonl.gz"
            )
        )
        if not v136.exists() or len(trajectories) != 10:
            self.skipTest("local v136 report and v98 trajectory artifacts are not present")

        report = build_rollout_sequence_support_audit_report(
            v136_report_path=v136,
            trajectory_paths=trajectories,
        ).report

        self.assertEqual(len(trajectories), 10)
        self.assertEqual(
            DEFAULT_OUTPUT_PATH.name,
            "mind-v3-v137-rollout-sequence-support-audit.json",
        )
        self.assertEqual(
            report["schema_version"],
            MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
        )
        self.assertTrue(report["source_integrity"]["passed"])
        self.assertTrue(report["support_floors"]["passed"])
        self.assertEqual(
            report["classification"]["primary"],
            "rollout_sequence_support_ready_for_future_sequence_world_model_scorer",
        )
        self.assertEqual(report["source_integrity"]["trajectory_loaded_path_count"], 10)
        self.assertEqual(report["source_integrity"]["trajectory_record_count"], 9912)
        split = report["train_heldout_split"]
        self.assertEqual(split["train_record_count"], 6847)
        self.assertEqual(split["heldout_record_count"], 3061)
        self.assertEqual(split["train_source_count"], 7)
        self.assertEqual(split["heldout_source_count"], 3)
        self.assertEqual(split["configured_heldout_seed_leakage_count"], 0)
        sequence = report["model_summaries"]["public_rollout_sequence_history_lookup"]
        self.assertEqual(sequence["accuracy"], 0.805619)
        self.assertEqual(sequence["dominant_predicted_action"]["action"], "stay")
        self.assertEqual(sequence["dominant_predicted_action_share"], 0.421431)
        self.assertEqual(sequence["unsupported_action_count"], 0)
        self.assertEqual(
            report["baseline_comparisons"]["heldout_accuracy_delta_vs_action_only"],
            0.443972,
        )
        self.assertEqual(
            report["baseline_comparisons"]["heldout_accuracy_delta_vs_action_order"],
            0.443972,
        )
        self.assertFalse(report["authorization_block"]["training_authorized"])
        self.assertFalse(
            report["authorization_block"]["runtime_policy_change_authorized"]
        )


def _v136_report() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
        "source_integrity": {"passed": True},
        "next_allowed_research_direction": NEXT_ALLOWED_RESEARCH_DIRECTION,
        "current_closed_path": CURRENT_CLOSED_PATH,
    }


def _dataset(
    path: str,
    actions: list[str],
    *,
    extra_by_index: dict[int, dict[str, object]] | None = None,
    seed: int | None = None,
) -> TrajectoryJsonlDataset:
    records = []
    for index, action in enumerate(actions):
        extra = (extra_by_index or {}).get(index, {})
        records.append(_record(index, action) | extra)
    return _dataset_from_records(path, records, seed=seed)


def _dataset_from_records(
    path: str,
    records: list[dict[str, object]],
    *,
    seed: int | None = None,
) -> TrajectoryJsonlDataset:
    resolved_seed = seed
    if resolved_seed is None:
        resolved_seed = 0
        for token in ("seed", "mind-v3-"):
            if token in path:
                digits = "".join(ch for ch in path.split(token, 1)[1] if ch.isdigit())
                if digits:
                    resolved_seed = int(digits)
                break
    return TrajectoryJsonlDataset(
        path=Path(path),
        header={},
        records=tuple(records),
        footer={
            "provenance": {
                "source_seeds": [resolved_seed],
                "config_digest": "unit-config",
                "contract_digest": "unit-contract",
                "split_id": "unit",
                "trajectory_paths": [path],
                "record_count": len(records),
            }
        },
    )


def _record(tick: int, action: str, *, agent_id: int = 1) -> dict[str, object]:
    mask = {name: False for name in ACTION_NAMES}
    mask["eat"] = True
    mask["drink"] = True
    return {
        "tick": tick,
        "agent_id": agent_id,
        "requested_action": action,
        "resolved_action": action,
        "action_source": "mind_v3",
        "resolution_action_valid": True,
        "action_mask": mask,
        "moved": False,
        "before": {
            "energy_ratio": 0.5,
            "hydration_ratio": 0.5,
            "health_ratio": 1.0,
        },
        "after": {
            "energy_ratio": 0.6 if action == "eat" else 0.5,
            "hydration_ratio": 0.6 if action == "drink" else 0.5,
            "health_ratio": 1.0,
        },
        "outcome": {
            "resource_gain": 0.1 if action == "eat" else 0.0,
            "feeding": {
                "ate": action == "eat",
                "food_source": "plant" if action == "eat" else None,
            },
            "drinking": {"drank": action == "drink"},
        },
    }


def _write_jsonl_trajectory(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    payloads = [{"type": "header"}]
    payloads.extend({"type": "record", "record": record} for record in records)
    payloads.append({"type": "footer"})
    path.write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads),
        encoding="utf-8",
    )
