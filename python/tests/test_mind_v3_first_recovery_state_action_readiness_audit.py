from __future__ import annotations

import base64
import gzip
import json
import os
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    _pack_quantized_values,
)
from evolution_sim.mind.first_recovery_state_action_readiness_audit import (
    CANDIDATE_FIELD_DEFINITIONS,
    MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
    build_first_recovery_state_action_readiness_audit,
    state_action_signal_field_leakage,
    write_first_recovery_state_action_readiness_audit_report,
)
import evolution_sim.mind.first_recovery_state_action_readiness_audit as audit_module

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryStateActionReadinessAuditTests(unittest.TestCase):
    def test_state_action_readiness_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:first-recovery-state-action-readiness-audit"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_state_action_readiness_audit"
            ),
        )

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
        )
        contract = build.report["contract"]
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertEqual(contract["replay_golden_effect"], "none")
        self.assertEqual(contract["summary_only_effect"], "none")
        self.assertFalse(contract["observation_field_change"])
        self.assertFalse(contract["private_world_state_input"])
        self.assertFalse(contract["seed_identity_signal"])
        self.assertTrue(build.report["non_promoted"])

    def test_baselines_include_action_state_interaction_and_shuffled_control(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        baselines = build.report["baseline_evaluation"]
        self.assertIn("action_only", baselines)
        self.assertIn("state_only", baselines)
        self.assertIn("state_action_interaction", baselines)
        self.assertIn("shuffled_action_control", baselines)
        self.assertIn("leave_one_seed", baselines["state_action_interaction"])
        self.assertIn("fixture_open", baselines["state_action_interaction"])

    def test_state_action_interaction_can_beat_action_only_on_heldout_splits(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        heldout = build.report["baseline_evaluation"]["heldout_summary"]
        self.assertTrue(heldout["go_criterion_state_action_beats_action_only"])
        interaction = build.report["baseline_evaluation"]["state_action_interaction"]
        self.assertGreater(
            interaction["leave_one_seed"]["summary"]["top1_oracle_match_rate"],
            build.report["baseline_evaluation"]["action_only"]["leave_one_seed"][
                "summary"
            ]["top1_oracle_match_rate"],
        )

    def test_alias_quantization_exact_1_2_3_digit_is_reported(self) -> None:
        rows, records = _synthetic_inputs(include_alias=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        policies = build.report["alias_quantization_stability"][
            "quantization_policies"
        ]
        self.assertEqual(
            [policy["quantization"] for policy in policies],
            ["exact", "1_digit", "2_digit", "3_digit"],
        )
        self.assertGreater(policies[0]["alias_group_count"], 0)

    def test_candidate_field_readiness_covers_required_fields(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        expected = {str(definition["field"]) for definition in CANDIDATE_FIELD_DEFINITIONS}
        actual = {
            item["field"]
            for item in build.report["candidate_field_readiness"]["candidate_fields"]
        }
        readiness = build.report["candidate_field_readiness"]
        self.assertEqual(actual, expected)
        self.assertGreater(
            readiness["contractable_missing_field_candidate_count"],
            0,
        )
        self.assertNotIn("ready_candidate_field_count", readiness)
        self.assertNotIn("ready_candidate_fields", readiness)

    def test_top_level_leakage_catches_mocked_state_action_feature_names(self) -> None:
        rows, records = _synthetic_inputs()
        original = audit_module._candidate_all_features

        def leaking_features(_target: object, _row: object) -> dict[str, float]:
            return {"seed": 1.0, "candidate_action.eat": 1.0}

        audit_module._candidate_all_features = leaking_features
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trajectory.jsonl.gz"
                _write_trajectory(path, records)
                rows = _with_source_path(rows, path)
                build = build_first_recovery_state_action_readiness_audit(
                    public_signal_audit=_public_signal_audit(),
                    archive_report=_archive_report(rows),
                    archive_rows=rows,
                    shadow_ranker_report=_shadow_ranker_report(),
                    trajectory_paths=(path,),
                )
        finally:
            audit_module._candidate_all_features = original

        leakage = build.report["leakage_guard"]
        self.assertEqual(leakage["direct_field_leakage_count"], 0)
        self.assertGreater(leakage["baseline_feature_name_leakage_count"], 0)
        self.assertIn("seed", leakage["baseline_leaking_feature_names"])
        self.assertIn(
            "state_action_signal_leakage_detected",
            build.report["classification"]["labels"],
        )

    def test_shuffled_control_beating_action_only_creates_blocker(self) -> None:
        rows, records = _synthetic_inputs()
        original = audit_module._rotated_oracle_actions
        audit_module._rotated_oracle_actions = lambda target: target.oracle_actions
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trajectory.jsonl.gz"
                _write_trajectory(path, records)
                rows = _with_source_path(rows, path)
                build = build_first_recovery_state_action_readiness_audit(
                    public_signal_audit=_public_signal_audit(),
                    archive_report=_archive_report(rows),
                    archive_rows=rows,
                    shadow_ranker_report=_shadow_ranker_report(),
                    trajectory_paths=(path,),
                )
        finally:
            audit_module._rotated_oracle_actions = original

        self.assertIn(
            "shuffled_control_artifact_detected",
            build.report["classification"]["labels"],
        )
        self.assertIn(
            "shuffled_control_artifact_detected",
            build.report["recommendation"]["blockers"],
        )

    def test_fallback_metrics_exist_on_heldout_summaries(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        summary = build.report["baseline_evaluation"]["action_only"]["leave_one_seed"][
            "summary"
        ]
        self.assertIn("candidate_row_score_source_counts", summary)
        self.assertIn("candidate_row_score_source_rates", summary)
        self.assertIn("selected_score_source_distribution", summary)
        self.assertIsNotNone(summary["signature_hit_count"])
        self.assertIsNotNone(summary["action_prior_fallback_count"])
        self.assertIsNotNone(summary["zero_score_fallback_count"])

    def test_quantization_no_alias_precision_is_not_instability(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_state_action_readiness_audit(
                public_signal_audit=_public_signal_audit(),
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        quantization = build.report["alias_quantization_stability"]
        self.assertTrue(quantization["alias_precision_without_aliases_is_not_instability"])
        self.assertEqual(quantization["v112_reference_quantization_digits"], 2)
        self.assertTrue(
            all(
                policy["resolution_status"] == "no_aliases_at_precision"
                for policy in quantization["quantization_policies"]
            )
        )

    def test_leakage_guard_rejects_identity_and_provenance_tokens(self) -> None:
        forbidden = (
            "seed",
            "source_path",
            "source_kind",
            "fixture.identity",
            "branch_id",
            "record_index",
            "agent_id",
            "tick",
            "logged_action",
            "private_world_state",
        )
        safe = (
            "state_action.eat_x_carrion_signal",
            "candidate_action.eat",
            "water_route_blocker_count",
        )

        self.assertEqual(state_action_signal_field_leakage(safe), ())
        self.assertEqual(set(state_action_signal_field_leakage(forbidden)), set(forbidden))

    def test_cli_writes_deterministic_json(self) -> None:
        rows, records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "trajectory.jsonl.gz"
            rows = _with_source_path(rows, trajectory)
            public_signal = root / "public-signal.json"
            archive_report = root / "archive.json"
            archive_rows = root / "archive.jsonl.gz"
            shadow_report = root / "shadow.json"
            first_output = root / "audit-first.json"
            second_output = root / "audit-second.json"
            public_signal.write_text(
                json.dumps(_public_signal_audit(), sort_keys=True),
                encoding="utf-8",
            )
            archive_report.write_text(
                json.dumps(_archive_report(rows), sort_keys=True),
                encoding="utf-8",
            )
            shadow_report.write_text(
                json.dumps(_shadow_ranker_report(), sort_keys=True),
                encoding="utf-8",
            )
            _write_rows(archive_rows, rows)
            _write_trajectory(trajectory, records)
            cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_state_action_readiness_audit",
                "--public-signal-audit",
                str(public_signal),
                "--archive-report",
                str(archive_report),
                "--archive-rows",
                str(archive_rows),
                "--shadow-ranker-report",
                str(shadow_report),
                "--trajectory",
                str(trajectory),
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

        self.assertIn("first_recovery_state_action_readiness_audit=", first.stdout)
        self.assertIn("contractable_missing_field_candidate_count=", first.stdout)
        self.assertNotIn("ready_candidate_field_count=", first.stdout)
        self.assertEqual(first_bytes, second_bytes)


def _synthetic_inputs(
    *,
    include_alias: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    specs = [
        ("branch-eat-fixture", 13, "fixture_carrion_only", 0, "eat", 0.9, 0.1),
        ("branch-drink-fixture", 19, "fixture_carrion_only", 1, "drink", 0.1, 0.9),
        ("branch-eat-open", 29, "open_mind_v3", 2, "eat", 0.9, 0.1),
        ("branch-drink-open", 37, "open_mind_v3", 3, "drink", 0.1, 0.9),
    ]
    if include_alias:
        specs.append(("branch-alias", 41, "open_mind_v3", 4, "drink", 0.9, 0.1))
    for branch_id, seed, source_kind, record_index, best, carrion, water in specs:
        digest = f"digest-{branch_id}"
        rows.extend(
            _group_rows(
                branch_id=branch_id,
                seed=seed,
                source_kind=source_kind,
                source_path=Path("synthetic.jsonl.gz"),
                record_index=record_index,
                digest=digest,
                best_action=best,
            )
        )
        records.append(
            _trajectory_record(
                agent_id=7,
                tick=4,
                digest=digest,
                requested_action="stay",
                resource_gain=0.0,
                observation_input=_observation(carrion_signal=carrion, water_strength=water),
            )
        )
    return rows, records


def _group_rows(
    *,
    branch_id: str,
    seed: int,
    source_kind: str,
    source_path: Path,
    record_index: int,
    digest: str,
    best_action: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for action in ("eat", "drink", "stay"):
        rows.append(
            {
                "schema_version": "mind_v3_first_recovery_branch_archive_v1",
                "archive_row_id": f"{branch_id}::{action}",
                "candidate_action": action,
                "oracle_rank": 1 if action == best_action else 2,
                "observation_legal": True,
                "action_mask": {
                    name: name in {"eat", "drink", "stay"} for name in ACTION_NAMES
                },
                "tick": 4,
                "record_index": record_index,
                "observation_digest": digest,
                "material_gain_label": action == best_action,
                "provenance": {
                    "branch_id": branch_id,
                    "seed": seed,
                    "source_kind": source_kind,
                    "source_path": str(source_path),
                    "record_index": record_index,
                    "agent_id": 7,
                    "logged_action": "stay",
                },
                "trainable_public_input": {
                    "post_carrion_first_recovery": True,
                    "candidate_action": action,
                    "action_mask": {
                        name: name in {"eat", "drink", "stay"} for name in ACTION_NAMES
                    },
                },
            }
        )
    return rows


def _with_source_path(
    rows: list[dict[str, object]],
    source_path: Path,
) -> list[dict[str, object]]:
    copied = json.loads(json.dumps(rows))
    for row in copied:
        row["provenance"]["source_path"] = str(source_path)
    return copied


def _trajectory_record(
    *,
    agent_id: int,
    tick: int,
    digest: str,
    requested_action: str,
    resource_gain: float,
    observation_input: dict[str, object],
) -> dict[str, object]:
    return {
        "tick": tick,
        "agent_id": agent_id,
        "observation_digest": digest,
        "observation_input": observation_input,
        "action_mask": {name: name in {"eat", "drink", "stay"} for name in ACTION_NAMES},
        "requested_action": requested_action,
        "resolved_action": requested_action,
        "before": {
            "alive": True,
            "energy_ratio": 0.4,
            "hydration_ratio": 0.4,
            "health_ratio": 0.9,
        },
        "after": {
            "alive": True,
            "energy_ratio": 0.4 + resource_gain,
            "hydration_ratio": 0.4,
            "health_ratio": 0.9,
        },
        "outcome": {"resource_gain": resource_gain},
        "moved": False,
    }


def _observation(
    *,
    carrion_signal: float,
    water_strength: float,
) -> dict[str, object]:
    values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
    values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.4
    values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.4
    values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
    values[SELF_INPUT_FIELDS.index("reproduction_ready")] = 0.0
    center_index = PATCH_CELL_COUNT // 2
    patch_start = len(SELF_INPUT_FIELDS) + center_index * len(PATCH_INPUT_FIELDS)
    values[patch_start + PATCH_INPUT_FIELDS.index("food")] = 0.2
    values[patch_start + PATCH_INPUT_FIELDS.index("vegetation")] = 0.1
    values[patch_start + PATCH_INPUT_FIELDS.index("recovery_debt")] = 0.3
    values[patch_start + PATCH_INPUT_FIELDS.index("carcass_energy")] = carrion_signal
    values[patch_start + PATCH_INPUT_FIELDS.index("carrion_signal")] = carrion_signal
    nav_start = len(SELF_INPUT_FIELDS) + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    water_start = nav_start + NAVIGATION_TARGETS.index("water") * len(NAVIGATION_INPUT_FIELDS)
    carrion_start = nav_start + NAVIGATION_TARGETS.index("carrion") * len(NAVIGATION_INPUT_FIELDS)
    values[water_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 1.0 - water_strength
    values[water_start + NAVIGATION_INPUT_FIELDS.index("strength")] = water_strength
    values[carrion_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 1.0 - carrion_signal
    values[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = carrion_signal
    data = base64.b64encode(
        zlib.compress(_pack_quantized_values(values), level=6)
    ).decode("ascii")
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "encoder_version": OBSERVATION_ENCODER_VERSION,
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": data,
    }


def _archive_report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_branch_archive_v1",
        "branch_archive_summary": {"archive_row_count": len(rows), "replay_verified": True},
    }


def _public_signal_audit() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_public_signal_audit_v1",
        "classification": {
            "primary": "public_recovery_signal_aliasing_detected",
            "missing_evidence": [],
        },
        "observability_summary": {
            "near_public_state_alias_group_count": 1,
            "non_sparse_aliasing_support": True,
        },
        "signal_families": {
            "water_route_quality": _family_section("public.water_route_blocker_count"),
            "recent_failed_intake": _family_section("public.failed_intake_reason"),
            "patch_residence_diminishing_returns": _family_section(
                "public.same_patch_residence_duration"
            ),
            "contestedness_competitor_pressure": _family_section(
                "public.competitor_intake_pressure"
            ),
            "carrion_freshness_depletion": _family_section(
                "public.recent_carcass_depletion_by_others",
                "public.carcass_age",
            ),
            "reproduction_readiness_debt": _family_section(
                "public.nearby_compatible_mate_count"
            ),
        },
    }


def _family_section(*missing_fields: str) -> dict[str, object]:
    return {
        "public_field_presence": {"present_fields": [], "field_count": 0},
        "missing_public_fields": list(missing_fields),
        "alias_group_count": 1,
        "per_seed_support": {"13": 1, "29": 1},
        "fixture_open_support": {"fixture_carrion_only": 1, "open_mind_v3": 1},
    }


def _shadow_ranker_report() -> dict[str, object]:
    return {
        "schema_version": "mind_v3_first_recovery_shadow_ranker_v1",
        "classification": {
            "primary": "shadow_ranker_action_distribution_collapsed",
            "missing_evidence": [],
        },
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            for row in rows:
                gzip_file.write(json.dumps(row, sort_keys=True).encode("utf-8"))
                gzip_file.write(b"\n")


def _write_trajectory(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzip_file:
            gzip_file.write(json.dumps({"type": "header"}, sort_keys=True).encode("utf-8"))
            gzip_file.write(b"\n")
            for record in records:
                gzip_file.write(
                    json.dumps({"type": "record", "record": record}, sort_keys=True).encode(
                        "utf-8"
                    )
                )
                gzip_file.write(b"\n")
            gzip_file.write(json.dumps({"type": "footer"}, sort_keys=True).encode("utf-8"))
            gzip_file.write(b"\n")


if __name__ == "__main__":
    unittest.main()
