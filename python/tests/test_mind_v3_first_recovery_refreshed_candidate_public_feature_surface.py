from __future__ import annotations

import copy
import json
import unittest
from collections import defaultdict
from pathlib import Path

from evolution_sim.mind.first_recovery_candidate_public_feature_surface import (
    build_first_recovery_candidate_public_feature_surface,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    _candidate_groups,
)
from evolution_sim.mind.first_recovery_observation_candidate_context import (
    build_first_recovery_observation_candidate_context,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    _candidate_context_join_digest,
    build_first_recovery_refreshed_candidate_public_feature_surface,
)
from evolution_sim.mind.provenance import stable_payload_digest
from python.tests.test_mind_v3_first_recovery_candidate_public_feature_surface import (
    _refresh_v124_and_v128_reports,
)
from python.tests.test_mind_v3_first_recovery_candidate_ranker_capacity_audit import (
    _archive_report,
)
from python.tests.test_mind_v3_first_recovery_observation_candidate_context import (
    _build as _build_v130,
    _payloads as _v130_payloads,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryRefreshedCandidatePublicFeatureSurfaceTests(
    unittest.TestCase
):
    def test_refreshed_surface_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-refreshed-candidate-public-feature-surface"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_first_recovery_refreshed_candidate_public_feature_surface"
            ),
        )

    def test_synthetic_refreshed_context_can_be_ready(self) -> None:
        payloads = _payloads_for_ready_v131()

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_candidate_public_feature_surface_ready_for_ranker_review",
        )
        self.assertTrue(build.report["metric_gate"]["passed"])
        self.assertGreater(
            build.report["within_branch_variance"][
                "target_resource_neighborhood_variance"
            ]["varying_path_count"],
            0,
        )
        self.assertTrue(
            build.report["probe_comparison"][
                "heldout_signal_beats_action_only_baseline"
            ]
        )
        self.assertTrue(
            build.report["probe_comparison"][
                "heldout_signal_beats_action_order_baseline"
            ]
        )
        self.assertEqual(
            build.report["forbidden_feature_scan"]["forbidden_feature_path_count"],
            0,
        )
        self.assertEqual(build.report["leakage_audit"]["leakage_count"], 0)
        self.assertEqual(len(build.feature_rows), 542)
        self.assertIn(
            "heldout",
            build.report["heldout_material_exact_match_recall"],
        )
        self.assertIn(
            "candidate_context_join_digest",
            build.feature_rows[0]["non_feature_metadata"],
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])

    def test_v130_not_ready_fails_source_integrity(self) -> None:
        payloads = _payloads_for_ready_v131()
        payloads["v130_report"]["classification"]["primary"] = (
            "observation_candidate_context_public_fields_missing"
        )

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v130_classification_not_ready_for_surface_refresh",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_candidate_public_feature_surface_source_integrity_failed",
        )

    def test_context_row_ordinal_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads_for_ready_v131()
        payloads["v130_context_rows"][0]["non_feature_metadata"][
            "candidate_ordinal_within_set"
        ] = 999

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "candidate_context_row_alignment_failed",
            build.report["source_integrity"]["failures"],
        )
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_candidate_public_feature_surface_source_integrity_failed",
        )

    def test_branch_constant_context_blocks_by_signal(self) -> None:
        payloads = _payloads_for_ready_v131()
        for row in payloads["v130_context_rows"]:
            context = row["candidate_observation_context"]
            for key in list(context):
                if key.startswith(
                    (
                        "candidate_target_context.",
                        "candidate_resource_context.",
                        "candidate_neighborhood_context.",
                    )
                ):
                    context[key] = "branch_constant"
        _refresh_v130_context_rows_digest(payloads)

        build = _build(payloads)

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_candidate_public_feature_surface_blocked_by_signal",
        )
        self.assertIn(
            "target_resource_neighborhood_context_does_not_vary",
            build.report["metric_gate"]["failures"],
        )
        self.assertEqual(
            build.report["within_branch_variance"][
                "target_resource_neighborhood_variance"
            ]["varying_path_count"],
            0,
        )

    def test_v130_private_world_oracle_context_paths_fail_closed(self) -> None:
        payloads = _payloads_for_ready_v131()
        forbidden_paths = [
            "candidate_target_context.private_world_state",
            "candidate_resource_context.oracle_score",
            "candidate_neighborhood_context.world_state",
        ]
        for index, path in enumerate(forbidden_paths):
            payloads["v130_context_rows"][index]["candidate_observation_context"][
                path
            ] = "leak"
        _refresh_v130_context_rows_digest(payloads)

        build = _build(payloads)

        self.assertFalse(build.report["forbidden_feature_scan"]["passed"])
        observed_paths = {
            item["path"]
            for item in build.report["forbidden_feature_scan"][
                "forbidden_feature_paths"
            ]
        }
        for path in forbidden_paths:
            self.assertIn(path, observed_paths)
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_candidate_public_feature_surface_source_integrity_failed",
        )

    def test_v129_feature_row_digest_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads_for_ready_v131()
        payloads["v129_feature_rows"][0]["candidate_public_features"][
            "candidate_action_is_signal"
        ] = True

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v129_feature_rows_digest_mismatch",
            build.report["source_integrity"]["failures"],
        )
        digest_report = build.report["source_integrity"]["source_row_digests"][
            "v129_feature_rows"
        ]
        self.assertTrue(digest_report["report_digest_present"])
        self.assertFalse(digest_report["matches_report_digest"])

    def test_v129_feature_row_digest_missing_fails_source_integrity(self) -> None:
        payloads = _payloads_for_ready_v131()
        del payloads["v129_report"]["feature_surface_summary"]["feature_rows_digest"]

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v129_feature_rows_digest_missing",
            build.report["source_integrity"]["failures"],
        )

    def test_v130_context_row_digest_mismatch_fails_source_integrity(self) -> None:
        payloads = _payloads_for_ready_v131()
        payloads["v130_context_rows"][0]["candidate_observation_context"][
            "candidate_target_context.synthetic_public_signal_bucket"
        ] = "mutated"

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v130_context_rows_digest_mismatch",
            build.report["source_integrity"]["failures"],
        )
        digest_report = build.report["source_integrity"]["source_row_digests"][
            "v130_context_rows"
        ]
        self.assertTrue(digest_report["report_digest_present"])
        self.assertFalse(digest_report["matches_report_digest"])

    def test_v130_context_row_digest_missing_fails_source_integrity(self) -> None:
        payloads = _payloads_for_ready_v131()
        del payloads["v130_report"]["context_rows_summary"]["context_rows_digest"]

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "v130_context_rows_digest_missing",
            build.report["source_integrity"]["failures"],
        )

    def test_same_action_cross_branch_context_swap_fails_join_proof(self) -> None:
        payloads = _payloads_for_ready_v131()
        _add_v130_join_digests(payloads)
        first, second = _same_action_cross_branch_pair(payloads["v130_context_rows"])
        rows = payloads["v130_context_rows"]
        rows[first]["candidate_observation_context"], rows[second][
            "candidate_observation_context"
        ] = (
            rows[second]["candidate_observation_context"],
            rows[first]["candidate_observation_context"],
        )

        build = _build(payloads)

        self.assertFalse(build.report["source_integrity"]["passed"])
        self.assertIn(
            "candidate_context_row_alignment_failed",
            build.report["source_integrity"]["failures"],
        )
        alignment = build.report["source_integrity"][
            "candidate_context_row_alignment"
        ]
        self.assertIn(
            "v130_candidate_context_join_digest_mismatch",
            alignment["failures"],
        )
        self.assertTrue(alignment["v130_join_digest_evidence_present"])

    def test_real_artifacts_when_local_data_exists(self) -> None:
        paths = [
            ROOT
            / "output/mind/mind-v3-v130-first-recovery-observation-candidate-context.json",
            ROOT
            / "output/mind/mind-v3-v130-first-recovery-observation-candidate-context-rows.jsonl",
            ROOT
            / "output/mind/mind-v3-v129-first-recovery-candidate-public-feature-surface.json",
            ROOT
            / "output/mind/mind-v3-v129-first-recovery-candidate-public-feature-rows.jsonl",
            ROOT
            / "output/mind/mind-v3-v128-first-recovery-candidate-ranker-capacity-audit.json",
            ROOT
            / "output/mind/mind-v3-v127-first-recovery-candidate-set-shadow-execution.json",
            ROOT
            / "output/mind/mind-v3-v127-first-recovery-candidate-set-predictions.jsonl",
            ROOT
            / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-contract.json",
            ROOT
            / "output/mind/mind-v3-v124-first-recovery-accepted-rare-attack-manifest.jsonl",
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json",
            ROOT
            / "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz",
            ROOT
            / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.json",
            ROOT
            / "output/mind/mind-v3-v123-first-recovery-active-coverage-archive.jsonl.gz",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("real v115-v130 artifacts are not present")

        build = build_first_recovery_refreshed_candidate_public_feature_surface()

        self.assertTrue(build.report["source_integrity"]["passed"])
        self.assertEqual(build.report["source_integrity"]["candidate_row_count"], 542)
        self.assertEqual(len(build.feature_rows), 542)
        self.assertEqual(
            build.report["forbidden_feature_scan"]["forbidden_feature_path_count"],
            0,
        )
        self.assertEqual(build.report["leakage_audit"]["leakage_count"], 0)
        self.assertEqual(
            build.report["classification"]["primary"],
            "refreshed_candidate_public_feature_surface_blocked_by_signal",
        )
        self.assertFalse(build.report["metric_gate"]["passed"])
        self.assertIn(
            "fixture_open_failed",
            build.report["metric_gate"]["failures"],
        )
        self.assertIn(
            "heldout_signal_not_above_action_only_baseline",
            build.report["metric_gate"]["failures"],
        )
        self.assertEqual(
            build.report["within_branch_variance"][
                "target_resource_neighborhood_variance"
            ]["varying_path_count"],
            26,
        )
        self.assertIn(
            "source_row_digests",
            build.report["source_integrity"],
        )
        self.assertIn(
            "heldout",
            build.report["heldout_material_exact_match_recall"],
        )
        self.assertFalse(
            build.report["recommendation"]["downstream_shadow_scorer_allowed"]
        )
        self.assertFalse(build.report["recommendation"]["training_executed"])


def _build(payloads: dict[str, object]):
    return build_first_recovery_refreshed_candidate_public_feature_surface(
        v130_report=copy.deepcopy(payloads["v130_report"]),
        v130_context_rows=copy.deepcopy(payloads["v130_context_rows"]),
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
    )


def _payloads_for_ready_v131() -> dict[str, object]:
    payloads = _v130_payloads()
    initial_v130 = _build_v130(payloads)
    selected_by_ordinal = _selected_ordinals_from_public_context(
        initial_v130.context_rows
    )
    _relabel_from_selected_ordinals(payloads, selected_by_ordinal)
    _refresh_reports(payloads)
    final_v130 = _build_v130(payloads)
    payloads["v130_report"] = final_v130.report
    payloads["v130_context_rows"] = [
        _with_synthetic_context_signal(row, selected_by_ordinal)
        for row in final_v130.context_rows
    ]
    _refresh_v130_context_rows_digest(payloads)
    return payloads


def _refresh_v130_context_rows_digest(payloads: dict[str, object]) -> None:
    payloads["v130_report"]["context_rows_summary"]["context_rows_digest"] = (
        stable_payload_digest(payloads["v130_context_rows"])
    )


def _add_v130_join_digests(payloads: dict[str, object]) -> None:
    branch_ordinals = {
        str(manifest["branch_id"]): index
        for index, manifest in enumerate(
            sorted(payloads["manifest_rows"], key=lambda row: str(row["branch_id"]))
        )
    }
    candidate_groups = _candidate_groups([*payloads["v115_rows"], *payloads["v123_rows"]])
    candidates_by_ordinal: dict[tuple[int, int], dict[str, object]] = {}
    for branch_id, branch_ordinal in branch_ordinals.items():
        for candidate_ordinal, candidate in enumerate(candidate_groups[branch_id]):
            candidates_by_ordinal[(branch_ordinal, candidate_ordinal)] = candidate
    for row in payloads["v130_context_rows"]:
        metadata = row["non_feature_metadata"]
        ordinal = (
            int(metadata["candidate_set_ordinal"]),
            int(metadata["candidate_ordinal_within_set"]),
        )
        metadata["candidate_context_join_digest"] = _candidate_context_join_digest(
            ordinal=ordinal,
            candidate=candidates_by_ordinal[ordinal],
            context_row=row,
        )


def _same_action_cross_branch_pair(rows: list[dict[str, object]]) -> tuple[int, int]:
    by_action: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        action = str(row["candidate_observation_context"]["candidate_action"])
        by_action[action].append(index)
    for indexes in by_action.values():
        for first in indexes:
            first_ordinal = rows[first]["non_feature_metadata"][
                "candidate_set_ordinal"
            ]
            for second in indexes:
                if first == second:
                    continue
                second_ordinal = rows[second]["non_feature_metadata"][
                    "candidate_set_ordinal"
                ]
                if first_ordinal != second_ordinal:
                    return first, second
    raise AssertionError("synthetic payload has no same-action cross-branch pair")


def _refresh_reports(payloads: dict[str, object]) -> None:
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


def _selected_ordinals_from_public_context(
    rows: tuple[dict[str, object], ...],
) -> set[tuple[int, int]]:
    grouped: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        meta = row["non_feature_metadata"]
        grouped[int(meta["candidate_set_ordinal"])].append(row)
    selected: set[tuple[int, int]] = set()
    for set_ordinal, candidates in grouped.items():
        winner = max(
            candidates,
            key=lambda row: stable_payload_digest(
                {
                    "public_candidate_context": row["candidate_observation_context"],
                    "candidate_action": row["candidate_observation_context"][
                        "candidate_action"
                    ],
                }
            ),
        )
        winner_meta = winner["non_feature_metadata"]
        selected.add(
            (
                set_ordinal,
                int(winner_meta["candidate_ordinal_within_set"]),
            )
        )
    return selected


def _relabel_from_selected_ordinals(
    payloads: dict[str, object],
    selected_ordinals: set[tuple[int, int]],
) -> None:
    manifest_by_ordinal = {
        index: row
        for index, row in enumerate(
            sorted(payloads["manifest_rows"], key=lambda row: str(row["branch_id"]))
        )
    }
    candidate_groups = _candidate_groups([*payloads["v115_rows"], *payloads["v123_rows"]])
    selected_by_branch: dict[str, dict[str, object]] = {}
    for set_ordinal, manifest in manifest_by_ordinal.items():
        branch_id = str(manifest["branch_id"])
        candidates = list(candidate_groups[branch_id])
        selected_index = next(
            ordinal[1] for ordinal in selected_ordinals if ordinal[0] == set_ordinal
        )
        selected = candidates[selected_index]
        selected_by_branch[branch_id] = dict(selected)
        for index, candidate in enumerate(candidates):
            candidate["material_gain_label"] = index == selected_index
    for manifest in payloads["manifest_rows"]:
        branch_id = str(manifest["branch_id"])
        selected = selected_by_branch[branch_id]
        manifest["repaired_action"] = selected["candidate_action"]
        manifest["repaired_archive_row_id"] = selected["archive_row_id"]


def _with_synthetic_context_signal(
    row: dict[str, object],
    selected_ordinals: set[tuple[int, int]],
) -> dict[str, object]:
    cloned = copy.deepcopy(row)
    meta = cloned["non_feature_metadata"]
    ordinal = (
        int(meta["candidate_set_ordinal"]),
        int(meta["candidate_ordinal_within_set"]),
    )
    bucket = "synthetic_high" if ordinal in selected_ordinals else "synthetic_low"
    context = cloned["candidate_observation_context"]
    for key in list(context):
        if key.startswith(
            (
                "candidate_target_context.",
                "candidate_resource_context.",
                "candidate_neighborhood_context.",
            )
        ):
            context[key] = bucket
    context["candidate_target_context.synthetic_public_signal_bucket"] = bucket
    return cloned


if __name__ == "__main__":
    unittest.main()
