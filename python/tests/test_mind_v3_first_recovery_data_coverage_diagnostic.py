from __future__ import annotations

import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from evolution_sim.mind.first_recovery_data_coverage_diagnostic import (
    MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
    build_first_recovery_data_coverage_diagnostic,
    first_recovery_data_coverage_trainable_field_leakage,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryDataCoverageDiagnosticTests(unittest.TestCase):
    def test_data_coverage_diagnostic_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-data-coverage-diagnostic"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_data_coverage_diagnostic"
            ),
        )

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        archive_report, rows, public_signal, readiness = _biased_inputs()
        build = build_first_recovery_data_coverage_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
            public_signal_audit=public_signal,
            state_action_readiness_audit=readiness,
        )

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
        )
        self.assertEqual(
            list(build.report.keys()),
            [
                "schema_version",
                "audit_policy",
                "contract",
                "source_reports",
                "reconstruction_coverage",
                "selected_vs_reconstructed_coverage",
                "skipped_target_coverage",
                "v112_alias_family_coverage",
                "v113_blocker_review",
                "expansion_budget",
                "leakage_guard",
                "classification",
                "recommendation",
                "non_promoted",
            ],
        )
        contract = build.report["contract"]
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertEqual(contract["replay_golden_effect"], "none")
        self.assertEqual(contract["summary_only_effect"], "none")
        self.assertFalse(contract["observation_field_change"])
        self.assertFalse(contract["expanded_archive_replay_executed"])
        self.assertFalse(contract["readiness_rerun_executed"])
        self.assertTrue(build.report["non_promoted"])

    def test_selected_vs_reconstructed_bias_detection_with_synthetic_data(self) -> None:
        archive_report, rows, public_signal, readiness = _biased_inputs()
        build = build_first_recovery_data_coverage_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
            public_signal_audit=public_signal,
            state_action_readiness_audit=readiness,
        )

        coverage = build.report["selected_vs_reconstructed_coverage"]
        self.assertTrue(coverage["bias_detected"])
        self.assertIn("source", coverage["biased_dimensions"])
        self.assertIn("logged_action", coverage["biased_dimensions"])
        self.assertIn(
            "v109_selection_bias_detected",
            build.report["classification"]["labels"],
        )
        self.assertIn(
            "expanded_archive_required",
            build.report["classification"]["labels"],
        )
        self.assertTrue(build.report["expansion_budget"]["recommend_expanded_archive"])
        self.assertTrue(build.report["recommendation"]["expanded_archive_allowed"])
        self.assertTrue(
            build.report["v113_blocker_review"][
                "coverage_plausibly_confounds_v113_readiness"
            ]
        )
        self.assertNotIn(
            "coverage_plausibly_explains_v113_blockers",
            build.report["v113_blocker_review"],
        )
        self.assertNotIn("explains", json.dumps(build.report, sort_keys=True))

    def test_representative_synthetic_data_does_not_claim_bias(self) -> None:
        archive_report, rows, public_signal, readiness = _representative_inputs()
        build = build_first_recovery_data_coverage_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
            public_signal_audit=public_signal,
            state_action_readiness_audit=readiness,
        )

        self.assertFalse(build.report["selected_vs_reconstructed_coverage"]["bias_detected"])
        self.assertTrue(build.report["selected_vs_reconstructed_coverage"]["representative"])
        self.assertIn(
            "v109_selection_representative",
            build.report["classification"]["labels"],
        )
        self.assertNotIn(
            "v109_selection_bias_detected",
            build.report["classification"]["labels"],
        )
        self.assertIn(
            "expanded_archive_not_justified",
            build.report["classification"]["labels"],
        )
        self.assertFalse(build.report["expansion_budget"]["recommend_expanded_archive"])

    def test_missing_input_evidence_is_inconclusive_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            build = build_first_recovery_data_coverage_diagnostic(
                archive_report_path=missing,
                archive_rows_path=Path(tmp) / "missing.jsonl.gz",
                public_signal_audit_path=missing,
                state_action_readiness_audit_path=missing,
            )

        self.assertEqual(
            build.report["classification"]["primary"],
            "missing_evidence_inconclusive",
        )
        self.assertIn(
            "archive_report",
            build.report["classification"]["missing_evidence"],
        )
        self.assertFalse(
            build.report["source_reports"]["archive_report"]["loaded"],
        )

    def test_leakage_guard_rejects_identity_and_private_world_trainable_fields(self) -> None:
        forbidden = (
            "seed",
            "source_kind",
            "fixture.identity",
            "branch_id",
            "private_world_state",
            "simulation_world.snapshot",
        )
        safe = (
            "candidate_action.eat",
            "target_public_state_before.energy_ratio",
            "public_transition_context.ticks_after_animal_resource_gain",
        )
        self.assertEqual(first_recovery_data_coverage_trainable_field_leakage(safe), ())
        self.assertEqual(
            set(first_recovery_data_coverage_trainable_field_leakage(forbidden)),
            set(forbidden),
        )

        archive_report, rows, public_signal, readiness = _representative_inputs()
        rows[0]["trainable_public_input"]["seed"] = 13
        rows[0]["trainable_public_input"]["private_world_state"] = {"x": 1}
        build = build_first_recovery_data_coverage_diagnostic(
            archive_report=archive_report,
            archive_rows=rows,
            public_signal_audit=public_signal,
            state_action_readiness_audit=readiness,
        )

        self.assertEqual(
            build.report["classification"]["primary"],
            "trainable_signal_leakage_detected",
        )
        self.assertGreater(build.report["leakage_guard"]["leakage_count"], 0)
        self.assertFalse(build.report["recommendation"]["expanded_archive_allowed"])

    def test_cli_smoke_with_temp_json_inputs(self) -> None:
        archive_report, rows, public_signal, readiness = _biased_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_report_path = root / "archive.json"
            archive_rows_path = root / "archive.jsonl.gz"
            public_signal_path = root / "public-signal.json"
            readiness_path = root / "readiness.json"
            first_output = root / "coverage-first.json"
            second_output = root / "coverage-second.json"
            archive_report_path.write_text(
                json.dumps(archive_report, sort_keys=True),
                encoding="utf-8",
            )
            public_signal_path.write_text(
                json.dumps(public_signal, sort_keys=True),
                encoding="utf-8",
            )
            readiness_path.write_text(
                json.dumps(readiness, sort_keys=True),
                encoding="utf-8",
            )
            _write_rows(archive_rows_path, rows)
            cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_data_coverage_diagnostic",
                "--archive-report",
                str(archive_report_path),
                "--archive-rows",
                str(archive_rows_path),
                "--public-signal-audit",
                str(public_signal_path),
                "--state-action-readiness-audit",
                str(readiness_path),
            ]
            env = {**os.environ, "PYTHONPATH": "python", "PYTHONHASHSEED": "0"}
            first = subprocess.run(
                [*cmd, "--output", str(first_output)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                [*cmd, "--output", str(second_output)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            first_bytes = first_output.read_bytes()
            second_bytes = second_output.read_bytes()
            report = json.loads(first_output.read_text())

        self.assertIn("first_recovery_data_coverage_diagnostic=", first.stdout)
        self.assertIn("selection_bias_detected=True", first.stdout)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            report["schema_version"],
            MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
        )


def _biased_inputs() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    reconstructed_by_source = {"fixture_carrion_only": 80, "open_mind_v3": 20}
    selected_by_source = {"fixture_carrion_only": 5, "open_mind_v3": 20}
    reconstructed_by_seed = {"13": 50, "29": 50}
    selected_by_seed = {"13": 12, "29": 13}
    reconstructed_by_action = {"eat": 50, "move_east": 50}
    selected_by_action = {"eat": 5, "move_east": 20}
    selected_targets = _selected_targets(
        selected_by_source=selected_by_source,
        selected_by_seed=selected_by_seed,
        selected_by_action=selected_by_action,
    )
    archive_report = _archive_report(
        reconstructed_by_source=reconstructed_by_source,
        selected_by_source=selected_by_source,
        reconstructed_by_seed=reconstructed_by_seed,
        selected_by_seed=selected_by_seed,
        reconstructed_by_action=reconstructed_by_action,
        selected_by_action=selected_by_action,
        selected_targets=selected_targets,
    )
    rows = _archive_rows(selected_targets)
    public_signal = _public_signal_audit(
        selected_by_seed=selected_by_seed,
        selected_by_source=selected_by_source,
    )
    return archive_report, rows, public_signal, _readiness_audit()


def _representative_inputs() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    reconstructed_by_source = {"fixture_carrion_only": 10, "open_mind_v3": 10}
    selected_by_source = {"fixture_carrion_only": 5, "open_mind_v3": 5}
    reconstructed_by_seed = {"13": 10, "29": 10}
    selected_by_seed = {"13": 5, "29": 5}
    reconstructed_by_action = {"eat": 10, "move_east": 10}
    selected_by_action = {"eat": 5, "move_east": 5}
    selected_targets = _selected_targets(
        selected_by_source=selected_by_source,
        selected_by_seed=selected_by_seed,
        selected_by_action=selected_by_action,
    )
    archive_report = _archive_report(
        reconstructed_by_source=reconstructed_by_source,
        selected_by_source=selected_by_source,
        reconstructed_by_seed=reconstructed_by_seed,
        selected_by_seed=selected_by_seed,
        reconstructed_by_action=reconstructed_by_action,
        selected_by_action=selected_by_action,
        selected_targets=selected_targets,
    )
    rows = _archive_rows(selected_targets)
    public_signal = _public_signal_audit(
        selected_by_seed=selected_by_seed,
        selected_by_source=selected_by_source,
    )
    return archive_report, rows, public_signal, _readiness_audit()


def _archive_report(
    *,
    reconstructed_by_source: dict[str, int],
    selected_by_source: dict[str, int],
    reconstructed_by_seed: dict[str, int],
    selected_by_seed: dict[str, int],
    reconstructed_by_action: dict[str, int],
    selected_by_action: dict[str, int],
    selected_targets: list[dict[str, object]],
) -> dict[str, object]:
    reconstructed_count = sum(reconstructed_by_source.values())
    selected_count = sum(selected_by_source.values())
    skipped_count = reconstructed_count - selected_count
    reconstructed_by_seed_source = _seed_source_counts(
        source_counts=reconstructed_by_source,
        seed_counts=reconstructed_by_seed,
        total=reconstructed_count,
    )
    selected_by_seed_source = _seed_source_counts(
        source_counts=selected_by_source,
        seed_counts=selected_by_seed,
        total=selected_count,
    )
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "row_reconstruction": {
            "answer": "branch_archive_replay_partial",
            "reconstructed_first_recovery_row_count": reconstructed_count,
            "expected_constructible_first_recovery_row_count": reconstructed_count,
            "expected_v108_reconstructed_first_recovery_row_count": reconstructed_count,
            "row_count_matches_v107": True,
            "row_count_matches_v108": True,
            "by_source": reconstructed_by_source,
            "by_seed": reconstructed_by_seed,
            "by_seed_source": reconstructed_by_seed_source,
            "by_logged_action": reconstructed_by_action,
        },
        "target_selection": {
            "answer": "branch_archive_replay_partial",
            "selection_policy": "synthetic_selection_policy",
            "candidate_row_count": reconstructed_count,
            "selected_target_count": selected_count,
            "skipped_target_count": skipped_count,
            "selected_by_source": selected_by_source,
            "selected_by_seed": selected_by_seed,
            "selected_by_seed_source": selected_by_seed_source,
            "selected_by_logged_action": selected_by_action,
            "selection_reason_counts": {"synthetic": selected_count},
            "skip_reason_counts": {"synthetic_skipped": skipped_count},
            "selected_targets": selected_targets,
            "skipped_examples": [
                {
                    "seed": 13,
                    "source_kind": "fixture_carrion_only",
                    "requested_action": "eat",
                    "recovery_tick": 9,
                    "reason": "synthetic_skipped",
                }
            ],
        },
        "branch_archive_summary": {
            "selected_row_count": selected_count,
            "full_reconstruction_row_count": reconstructed_count,
            "archive_row_count": selected_count,
            "heuristic_action_source_count": 0,
            "zero_heuristic_runtime_actions_except_diagnostic_force": True,
        },
    }


def _selected_targets(
    *,
    selected_by_source: dict[str, int],
    selected_by_seed: dict[str, int],
    selected_by_action: dict[str, int],
) -> list[dict[str, object]]:
    sources = _expand_counts(selected_by_source)
    seeds = _expand_counts(selected_by_seed)
    actions = _expand_counts(selected_by_action)
    targets: list[dict[str, object]] = []
    for index, (source, seed, action) in enumerate(
        zip(sources, seeds, actions, strict=True)
    ):
        targets.append(
            {
                "source_kind": source,
                "seed": int(seed),
                "requested_action": action,
                "recovery_tick": index + 1,
                "recovery_record_index": index + 10,
                "agent_id": index + 100,
                "observation_digest": f"digest-{index}",
                "selection_reason": "synthetic",
            }
        )
    return targets


def _archive_rows(selected_targets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, target in enumerate(selected_targets):
        branch_id = f"branch-{index}"
        rows.append(
            {
                "schema_version": "mind_v3_first_recovery_branch_archive_v1",
                "archive_row_id": f"{branch_id}::action::{target['requested_action']}",
                "source_kind": target["source_kind"],
                "seed": target["seed"],
                "tick": target["recovery_tick"],
                "candidate_action": target["requested_action"],
                "oracle_rank": 1,
                "material_gain_label": True,
                "trainable_public_input": {
                    "schema_version": (
                        "mind_v3_first_recovery_branch_archive_trainable_public_input_v1"
                    ),
                    "post_carrion_first_recovery": True,
                    "candidate_action": target["requested_action"],
                    "action_mask": {"eat": True, "move_east": True, "stay": True},
                    "target_public_state_before": {
                        "alive": True,
                        "energy_ratio": 0.4,
                        "hydration_ratio": 0.5,
                        "health_ratio": 0.9,
                    },
                },
                "provenance": {
                    "branch_id": branch_id,
                    "source_kind": target["source_kind"],
                    "seed": target["seed"],
                    "logged_action": target["requested_action"],
                    "record_index": target["recovery_record_index"],
                    "diagnostics_only": True,
                },
            }
        )
    return rows


def _public_signal_audit(
    *,
    selected_by_seed: dict[str, int],
    selected_by_source: dict[str, int],
) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_public_signal_audit_v1",
        "classification": {
            "primary": "public_recovery_signal_aliasing_detected",
            "labels": [
                "public_recovery_signal_aliasing_detected",
                "diagnostics_only_no_runtime_promotion",
            ],
            "missing_evidence": [],
        },
        "signal_families": {
            family: {
                "alias_group_count": 1,
                "missing_public_fields": [f"public.{family}.synthetic_missing_field"],
                "per_seed_support": selected_by_seed,
                "fixture_open_support": selected_by_source,
            }
            for family in (
                "water_route_quality",
                "recent_failed_intake",
                "patch_residence_diminishing_returns",
            )
        },
    }


def _readiness_audit() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_state_action_readiness_audit_v1",
        "classification": {
            "primary": "state_action_observation_field_readiness_blocked",
            "labels": [
                "diagnostics_only_no_runtime_promotion",
                "state_action_interaction_not_better_than_action_only",
                "alias_resolution_unstable",
                "shuffled_control_artifact_detected",
                "state_action_observation_field_readiness_blocked",
            ],
            "missing_evidence": [],
        },
        "recommendation": {
            "next_step": "readiness_blocked_repair_audit_or_run_broader_data_diagnostics",
            "blockers": [
                "shuffled_control_artifact_detected",
                "state_action_interaction_does_not_beat_action_only_heldout",
                "alias_resolution_not_stable_under_quantization",
            ],
            "runtime_policy_change_recommended": False,
            "trained_artifact_change_recommended": False,
            "gate_change_recommended": False,
        },
    }


def _seed_source_counts(
    *,
    source_counts: dict[str, int],
    seed_counts: dict[str, int],
    total: int,
) -> dict[str, int]:
    if total <= 0:
        return {}
    counts: dict[str, int] = {}
    remaining = total
    pairs = [
        (source, seed)
        for source in sorted(source_counts)
        for seed in sorted(seed_counts)
    ]
    for index, (source, seed) in enumerate(pairs):
        if index == len(pairs) - 1:
            value = remaining
        else:
            value = min(source_counts[source], seed_counts[seed], max(0, remaining))
            value = int(round(value / max(1, len(pairs) - index)))
        if value > 0:
            counts[f"{source}:{seed}"] = value
            remaining -= value
    if sum(counts.values()) != total:
        first = f"{sorted(source_counts)[0]}:{sorted(seed_counts)[0]}"
        counts[first] = counts.get(first, 0) + total - sum(counts.values())
    return {key: value for key, value in counts.items() if value > 0}


def _expand_counts(counts: dict[str, int]) -> list[str]:
    values: list[str] = []
    for key, count in sorted(counts.items()):
        values.extend([str(key)] * int(count))
    return values


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            for row in rows:
                gzip_file.write(json.dumps(row, sort_keys=True).encode("utf-8"))
                gzip_file.write(b"\n")


if __name__ == "__main__":
    unittest.main()
