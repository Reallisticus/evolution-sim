from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_candidate_public_feature_surface import (
    build_first_recovery_candidate_public_feature_surface,
)
from evolution_sim.mind.first_recovery_observation_candidate_context import (
    build_first_recovery_observation_candidate_context,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_first_recovery_candidate_ranker_capacity_audit import (
    _archive_report,
)
from python.tests.test_mind_v3_first_recovery_candidate_public_feature_surface import (
    _payloads_for_v129,
    _refresh_v124_and_v128_reports,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryObservationCandidateContextTests(unittest.TestCase):
    def test_observation_candidate_context_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-observation-candidate-context"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_observation_candidate_context"
            ),
        )

    def test_synthetic_public_observation_context_is_ready(self) -> None:
        payloads = _payloads()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "observation_candidate_context_ready_for_v129_surface_refresh",
        )
        self.assertEqual(
            build.report["candidate_context_availability"][
                "missing_public_observation_fields"
            ],
            [],
        )
        self.assertGreater(
            build.report["within_branch_variance"][
                "target_resource_neighborhood_variance"
            ]["varying_path_count"],
            0,
        )
        self.assertEqual(
            build.report["forbidden_field_scan"]["forbidden_feature_path_count"],
            0,
        )
        self.assertEqual(len(build.context_rows), 542)
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])
        self.assertEqual(
            build.report["authorization_block"]["foundation_effect"],
            "none",
        )
        self.assertFalse(
            build.report["authorization_block"]["runtime_policy_change_authorized"]
        )

    def test_missing_public_observation_fields_fail_closed(self) -> None:
        payloads = _payloads()
        for record in payloads["trajectory_records"]:
            record.pop("decoded_observation", None)
            record.pop("observation_input", None)

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "observation_candidate_context_public_fields_missing",
        )
        missing = {
            item["field"]
            for item in build.report["candidate_context_availability"][
                "missing_public_observation_fields"
            ]
        }
        self.assertIn("trajectory_record.observation_input", missing)
        self.assertIn(
            "public_observation_fields_missing",
            build.report["metric_gate"]["failures"],
        )
        self.assertFalse(
            build.report["recommendation"]["v113_readiness_rerun_allowed"]
        )

    def test_source_integrity_fails_when_v129_rows_do_not_match_candidates(self) -> None:
        payloads = _payloads()
        payloads["v129_feature_rows"] = payloads["v129_feature_rows"][:-1]

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v129_feature_row_count_mismatch",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "observation_candidate_context_source_integrity_failed",
        )

    def test_real_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT / "output/mind/mind-v3-v129-first-recovery-candidate-public-feature-surface.json",
            ROOT / "output/mind/mind-v3-v129-first-recovery-candidate-public-feature-rows.jsonl",
            ROOT / "output/mind/mind-v3-v128-first-recovery-candidate-ranker-capacity-audit.json",
            ROOT / "output/mind/mind-v3-v127-first-recovery-candidate-set-shadow-execution.json",
            ROOT / "output/mind/mind-v3-v127-first-recovery-candidate-set-predictions.jsonl",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json",
            ROOT / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json",
            ROOT / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115-v129 artifacts are not present")

        build = build_first_recovery_observation_candidate_context()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["candidate_context_availability"]["candidate_row_count"],
            542,
        )
        self.assertEqual(
            build.report["forbidden_field_scan"]["forbidden_feature_path_count"],
            0,
        )
        if (
            build.report["classification"]["primary"]
            == "observation_candidate_context_ready_for_v129_surface_refresh"
        ):
            self.assertGreater(
                build.report["within_branch_variance"][
                    "target_resource_neighborhood_variance"
                ]["varying_path_count"],
                0,
            )
        else:
            self.assertEqual(
                build.report["classification"]["primary"],
                "observation_candidate_context_public_fields_missing",
            )
            self.assertTrue(
                build.report["candidate_context_availability"][
                    "missing_public_observation_fields"
                ]
            )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])
        self.assertFalse(
            build.report["authorization_block"]["downstream_shadow_scorer_allowed"]
        )


def _build(payloads: dict[str, object]):
    return build_first_recovery_observation_candidate_context(
        v129_report=copy.deepcopy(payloads["v129_report"]),
        v129_feature_rows=copy.deepcopy(payloads["v129_feature_rows"]),
        v128_report=copy.deepcopy(payloads["v128_report"]),
        v127_report=copy.deepcopy(payloads["v127_report"]),
        v127_prediction_rows=copy.deepcopy(payloads["v127_prediction_rows"]),
        v124_report=copy.deepcopy(payloads["v124_report"]),
        v124_manifest_rows=copy.deepcopy(payloads["manifest_rows"]),
        v115_report=copy.deepcopy(payloads["v115_report"]),
        v115_archive_rows=copy.deepcopy(payloads["v115_rows"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
        trajectory_records=copy.deepcopy(payloads["trajectory_records"]),
    )


def _payloads() -> dict[str, object]:
    payloads = _payloads_for_v129()
    _add_synthetic_record_identity(payloads)
    payloads["v115_report"] = _archive_report(payloads["v115_rows"])
    payloads["v123_report"] = _archive_report(payloads["v123_rows"], active=True)
    _refresh_v124_and_v128_reports(payloads)
    v129 = build_first_recovery_candidate_public_feature_surface(
        v128_report=copy.deepcopy(payloads["v128_report"]),
        v127_report=copy.deepcopy(payloads["v127_report"]),
        v127_prediction_rows=copy.deepcopy(payloads["v127_prediction_rows"]),
        v124_report=copy.deepcopy(payloads["v124_report"]),
        v124_manifest_rows=copy.deepcopy(payloads["manifest_rows"]),
        v115_report=copy.deepcopy(payloads["v115_report"]),
        v115_archive_rows=copy.deepcopy(payloads["v115_rows"]),
        v123_report=copy.deepcopy(payloads["v123_report"]),
        v123_archive_rows=copy.deepcopy(payloads["v123_rows"]),
    )
    payloads["v129_report"] = v129.report
    payloads["v129_feature_rows"] = list(v129.feature_rows)
    payloads["trajectory_records"] = _trajectory_records(payloads)
    return payloads


def _add_synthetic_record_identity(payloads: dict[str, object]) -> None:
    rows_by_branch: dict[str, list[dict[str, object]]] = {}
    for row in [*payloads["v115_rows"], *payloads["v123_rows"]]:
        rows_by_branch.setdefault(str(row["provenance"]["branch_id"]), []).append(row)
    for index, (branch_id, rows) in enumerate(rows_by_branch.items()):
        agent_id = 1000 + index
        observation_digest = stable_payload_digest(
            {
                "schema_version": "synthetic_v130_public_observation_digest_v1",
                "branch_id": branch_id,
            }
        )
        for row in rows:
            row["record_index"] = index
            row["agent_id"] = agent_id
            row["tick"] = 1
            row["observation_digest"] = observation_digest
            row["action_mask"] = row["trainable_public_input"]["action_mask"]
            row["provenance"]["record_index"] = index
            row["provenance"]["agent_id"] = agent_id


def _trajectory_records(payloads: dict[str, object]) -> list[dict[str, object]]:
    rows_by_branch: dict[str, dict[str, object]] = {}
    for row in [*payloads["v115_rows"], *payloads["v123_rows"]]:
        rows_by_branch.setdefault(str(row["provenance"]["branch_id"]), row)
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows_by_branch.values()):
        provenance = row["provenance"]
        records.append(
            {
                "source_path": provenance["source_path"],
                "record_index": row["record_index"],
                "agent_id": provenance["agent_id"],
                "tick": row["tick"],
                "observation_digest": row["observation_digest"],
                "action_mask": row["trainable_public_input"]["action_mask"],
                "decoded_observation": _decoded_observation(index),
            }
        )
    return records


def _decoded_observation(index: int) -> dict[str, object]:
    patch: dict[tuple[int, int], dict[str, object]] = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            signal = ((index + (dx * 3) + (dy * 5)) % 9) / 8
            patch[(dx, dy)] = {
                "in_bounds": 1.0,
                "terrain_code": signal,
                "occupant_code": 1.0 if (dx, dy) == (1, 0) else 0.0,
                "same_lineage": 0.0,
                "water_access_reason_code": 1.0 if dy == -1 else 0.0,
                "food": signal if (dx, dy) == (0, 0) else 0.0,
                "vegetation": signal,
                "fresh_kill_energy": 0.75 if (dx, dy) == (-1, 0) else 0.0,
                "carcass_energy": 0.5 if (dx, dy) == (0, 1) else 0.0,
                "hazard_level": 0.25 if dx == 0 else 0.0,
                "ecology_state_code": signal,
                "prey_biomass": 0.8 if (dx, dy) == (1, 0) else 0.0,
                "carrion_signal": 0.7 if dx < 0 else 0.0,
                "predator_risk": 0.6 if dy > 0 else 0.0,
            }
    return {
        "patch": patch,
        "navigation": {
            "water": {"strength": 0.8, "distance": 0.2},
            "plant": {"strength": 0.7, "distance": 0.3},
            "carrion": {"strength": 0.6, "distance": 0.4},
            "prey": {"strength": 0.5, "distance": 0.5},
        },
    }


if __name__ == "__main__":
    unittest.main()
