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
from evolution_sim.mind.first_recovery_public_signal_audit import (
    MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
    build_first_recovery_public_signal_audit,
    signal_field_leakage,
    write_first_recovery_public_signal_audit_report,
)

ROOT = Path(__file__).resolve().parents[2]


class MindV3FirstRecoveryPublicSignalAuditTests(unittest.TestCase):
    def test_public_signal_audit_has_npm_entrypoint(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(
            package["scripts"]["sim:mind:v3:first-recovery-public-signal-audit"],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli.mind_v3_first_recovery_public_signal_audit"
            ),
        )

    def test_schema_and_contract_are_diagnostics_only(self) -> None:
        rows, trajectory_path, trajectory_records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, trajectory_records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_public_signal_audit(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        self.assertEqual(
            build.report["schema_version"],
            MIND_V3_FIRST_RECOVERY_PUBLIC_SIGNAL_AUDIT_SCHEMA_VERSION,
        )
        contract = build.report["contract"]
        self.assertTrue(contract["diagnostics_only"])
        self.assertEqual(contract["runtime_policy_effect"], "none")
        self.assertEqual(contract["trained_artifact_effect"], "none")
        self.assertEqual(contract["gate_effect"], "none")
        self.assertEqual(contract["replay_golden_effect"], "none")
        self.assertEqual(contract["summary_only_effect"], "none")
        self.assertFalse(contract["private_world_state_serialized"])
        self.assertFalse(contract["fixture_identity_signal"])
        self.assertFalse(contract["seed_identity_signal"])
        self.assertTrue(build.report["non_promoted"])

    def test_synthetic_archive_row_joins_by_public_provenance_and_digest(self) -> None:
        rows, _trajectory_path, trajectory_records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, trajectory_records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_public_signal_audit(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        join = build.report["join_evidence"]
        self.assertEqual(join["loaded_path_count"], 1)
        self.assertEqual(join["malformed_record_count"], 0)
        self.assertEqual(join["matched_archive_row_count"], len(rows))
        self.assertEqual(join["missing_archive_row_count"], 0)

    def test_same_agent_history_excludes_current_future_and_other_agent_rows(self) -> None:
        rows, _trajectory_path, trajectory_records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, trajectory_records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_public_signal_audit(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        examples = build.report["signal_families"]["recent_failed_intake"]["examples"]
        values = examples[0]["family_values"]
        self.assertEqual(values["history.eat_count"], 1.0)
        self.assertEqual(values["history.eat_no_gain_count"], 1.0)

    def test_alias_detection_catches_same_public_state_with_different_oracles(self) -> None:
        rows, _trajectory_path, trajectory_records = _synthetic_inputs(
            include_alias_group=True
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, trajectory_records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_public_signal_audit(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        aliasing = build.report["state_action_aliasing"]
        self.assertGreater(aliasing["near_public_state_alias_group_count"], 0)
        self.assertIn(
            "public_recovery_signal_aliasing_detected",
            build.report["classification"]["labels"],
        )

    def test_constant_and_missing_field_detection_works(self) -> None:
        rows, _trajectory_path, trajectory_records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl.gz"
            _write_trajectory(path, trajectory_records)
            rows = _with_source_path(rows, path)
            build = build_first_recovery_public_signal_audit(
                archive_report=_archive_report(rows),
                archive_rows=rows,
                shadow_ranker_report=_shadow_ranker_report(),
                trajectory_paths=(path,),
            )

        carrion = build.report["signal_families"]["carrion_freshness_depletion"]
        constant_fields = {
            item["field"] for item in carrion["constant_near_constant_fields"]
        }
        self.assertIn("patch.max_carcass_energy", constant_fields)
        self.assertIn("public.carcass_age", carrion["missing_public_fields"])

    def test_signal_field_leakage_guard_rejects_provenance_tokens(self) -> None:
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
        )
        safe = (
            "carrion_freshness_depletion.patch.max_carcass_energy",
            "water_route_quality.navigation.water.distance",
        )

        self.assertEqual(signal_field_leakage(safe), ())
        self.assertEqual(set(signal_field_leakage(forbidden)), set(forbidden))

    def test_cli_writes_deterministic_json(self) -> None:
        rows, _trajectory_path, trajectory_records = _synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "trajectory.jsonl.gz"
            rows = _with_source_path(rows, trajectory)
            archive_report = root / "archive.json"
            archive_rows = root / "archive.jsonl.gz"
            shadow_report = root / "shadow.json"
            first_output = root / "audit-first.json"
            second_output = root / "audit-second.json"
            archive_report.write_text(
                json.dumps(_archive_report(rows), sort_keys=True),
                encoding="utf-8",
            )
            shadow_report.write_text(
                json.dumps(_shadow_ranker_report(), sort_keys=True),
                encoding="utf-8",
            )
            _write_rows(archive_rows, rows)
            _write_trajectory(trajectory, trajectory_records)
            cmd = [
                "python3",
                "-m",
                "evolution_sim.cli.mind_v3_first_recovery_public_signal_audit",
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

        self.assertIn("first_recovery_public_signal_audit=", first.stdout)
        self.assertEqual(first_bytes, second_bytes)


def _synthetic_inputs(
    *,
    include_alias_group: bool = False,
) -> tuple[list[dict[str, object]], Path, list[dict[str, object]]]:
    trajectory = Path("synthetic.jsonl.gz")
    rows = _group_rows(
        branch_id="branch-a",
        seed=13,
        source_kind="fixture_carrion_only",
        source_path=trajectory,
        record_index=2,
        digest="digest-a",
        best_action="eat",
    )
    records = [
        _trajectory_record(
            agent_id=7,
            tick=1,
            digest="history-a",
            requested_action="eat",
            resource_gain=0.0,
            observation_input=_observation(),
        ),
        _trajectory_record(
            agent_id=8,
            tick=2,
            digest="other-agent",
            requested_action="eat",
            resource_gain=0.0,
            observation_input=_observation(),
        ),
        _trajectory_record(
            agent_id=7,
            tick=4,
            digest="digest-a",
            requested_action="stay",
            resource_gain=0.0,
            observation_input=_observation(),
        ),
        _trajectory_record(
            agent_id=7,
            tick=5,
            digest="future-a",
            requested_action="eat",
            resource_gain=0.0,
            observation_input=_observation(),
        ),
    ]
    if include_alias_group:
        rows.extend(
            _group_rows(
                branch_id="branch-b",
                seed=29,
                source_kind="open_mind_v3",
                source_path=trajectory,
                record_index=6,
                digest="digest-b",
                best_action="drink",
                agent_id=9,
            )
        )
        records.extend(
            [
                _trajectory_record(
                    agent_id=9,
                    tick=6,
                    digest="history-b",
                    requested_action="eat",
                    resource_gain=0.0,
                    observation_input=_observation(),
                ),
                _trajectory_record(
                    agent_id=8,
                    tick=7,
                    digest="other-agent-b",
                    requested_action="eat",
                    resource_gain=0.0,
                    observation_input=_observation(),
                ),
                _trajectory_record(
                    agent_id=9,
                    tick=4,
                    digest="digest-b",
                    requested_action="stay",
                    resource_gain=0.0,
                    observation_input=_observation(),
                ),
            ]
        )
    return rows, trajectory, records


def _group_rows(
    *,
    branch_id: str,
    seed: int,
    source_kind: str,
    source_path: Path,
    record_index: int,
    digest: str,
    best_action: str,
    agent_id: int = 7,
) -> list[dict[str, object]]:
    actions = ("eat", "drink", "stay")
    rows: list[dict[str, object]] = []
    for action in actions:
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
                    "agent_id": agent_id,
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


def _observation() -> dict[str, object]:
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
    values[patch_start + PATCH_INPUT_FIELDS.index("carcass_energy")] = 0.6
    values[patch_start + PATCH_INPUT_FIELDS.index("carrion_signal")] = 0.7
    nav_start = len(SELF_INPUT_FIELDS) + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    water_start = nav_start + NAVIGATION_TARGETS.index("water") * len(NAVIGATION_INPUT_FIELDS)
    carrion_start = nav_start + NAVIGATION_TARGETS.index("carrion") * len(NAVIGATION_INPUT_FIELDS)
    values[water_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.4
    values[water_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 0.6
    values[carrion_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.1
    values[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 0.8
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
