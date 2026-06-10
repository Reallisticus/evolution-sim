from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    OBSERVATION_SCHEMA_VERSION,
    PATCH_CELL_COUNT,
    encode_observation_input,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
    candidate_feature_leakage_scan,
)
from evolution_sim.mind.carrion_survivor_continuation_v169_public_temporal_context_probe import (
    EXPECTED_V168_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION,
    _classification,
    build_candidate_feature_families,
    load_public_temporal_contexts,
    run_carrion_survivor_continuation_v169_public_temporal_context_probe,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV169PublicTemporalContextProbeTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v169-public-temporal-context-probe"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v169_public_temporal_context_probe"
            ),
        )

    def test_v168_digest_or_classification_mismatch_closes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, v168_report, _v167_report = _write_inputs(tmpdir)

            result = run_carrion_survivor_continuation_v169_public_temporal_context_probe(
                v168_report_path=paths["v168_report"],
                v167_report_path=paths["v167_report"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v169_report"],
                expected_v168_exact_digest="wrong",
                expected_v167_exact_digest=str(_v167_report["exact_digest"]),
                expected_v165_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v168_unexpected_exact_digest",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v169_public_temporal_context_probe_"
                "closed_invalid"
            ),
        )
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_prior_context_uses_strictly_previous_public_same_actor_records_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _paths, rows, _v168_report, _v167_report = _write_inputs(tmpdir)
            contexts = load_public_temporal_contexts(rows)
            families = build_candidate_feature_families(
                rows=rows,
                temporal_contexts_by_row=contexts["contexts_by_row"],
            )

        row_context = contexts["contexts_by_row"][0]
        self.assertEqual(row_context["strict_prior_public_record_count"], 2)
        payload = families[
            "decoded_public_numeric_plus_prior_observation_delta_window_5"
        ]["payloads"][0]
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertNotIn("requested_action", payload_text)
        self.assertNotIn("resolved_action", payload_text)
        self.assertNotIn("runtime_action", payload_text)
        self.assertNotIn("future", payload_text)
        self.assertNotIn("outcome", payload_text)
        self.assertNotIn("agent", payload_text)
        self.assertNotIn("tick", payload_text)
        self.assertNotIn("path", payload_text)
        self.assertTrue(candidate_feature_leakage_scan([payload])["passed"])

    def test_candidate_payloads_pass_leakage_scan_with_temporal_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _paths, rows, _v168_report, _v167_report = _write_inputs(tmpdir)
            contexts = load_public_temporal_contexts(rows)
            families = build_candidate_feature_families(
                rows=rows,
                temporal_contexts_by_row=contexts["contexts_by_row"],
            )

        for family in families.values():
            scan = candidate_feature_leakage_scan(family["payloads"])
            self.assertTrue(scan["passed"], family["feature_family"])

    def test_classification_branches_cover_no_net_partial_and_ready(self) -> None:
        source_validation = {"passed": True}
        baseline = {
            "zero_safe_hit_preterminal_source_seeds": [5, 13],
            "safe_hit_rate": 0.20,
        }

        no_net = _classification(
            source_validation=source_validation,
            leakage_passed=True,
            baseline=baseline,
            candidate_reports=[
                {
                    "feature_family": "decoded_public_numeric_plus_prior_observation_delta_window_1",
                    "zero_safe_hit_preterminal_source_seeds": [5, 19],
                }
            ],
        )
        partial = _classification(
            source_validation=source_validation,
            leakage_passed=True,
            baseline=baseline,
            candidate_reports=[
                {
                    "feature_family": "decoded_public_numeric_plus_prior_observation_delta_window_3",
                    "zero_safe_hit_preterminal_source_seeds": [5],
                    "unsupported_prediction_count": 0,
                    "dominant_predicted_action_share": 0.40,
                    "safe_hit_margin_over_best_trivial": 0.01,
                    "safe_hit_rate": 0.25,
                }
            ],
        )
        ready = _classification(
            source_validation=source_validation,
            leakage_passed=True,
            baseline=baseline,
            candidate_reports=[
                {
                    "feature_family": "decoded_public_numeric_plus_action_mask_transition_window_5",
                    "zero_safe_hit_preterminal_source_seeds": [],
                    "unsupported_prediction_count": 0,
                    "dominant_predicted_action_share": 0.50,
                    "safe_hit_margin_over_best_trivial": 0.05,
                    "safe_hit_rate": 0.30,
                }
            ],
        )

        self.assertTrue(no_net.endswith("public_temporal_context_probe_no_net_viable_projection_closed"))
        self.assertTrue(partial.endswith("public_temporal_context_partial_support_closed_more_context_or_source"))
        self.assertTrue(ready.endswith("public_temporal_context_candidate_ready_for_dataset_expansion_design"))

    def test_cli_writes_report_with_exact_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, rows, v168_report, v167_report = _write_inputs(tmpdir)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v169_public_temporal_context_probe",
                    "--v168-report",
                    str(paths["v168_report"]),
                    "--v167-report",
                    str(paths["v167_report"]),
                    "--v165-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(paths["v169_report"]),
                    "--expected-v168-exact-digest",
                    str(v168_report["exact_digest"]),
                    "--expected-v167-exact-digest",
                    str(v167_report["exact_digest"]),
                    "--expected-v165-dataset-digest",
                    stable_payload_digest(rows),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            written = json.loads(paths["v169_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v169_public_temporal_context_probe_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V169_PUBLIC_TEMPORAL_CONTEXT_PROBE_SCHEMA_VERSION,
        )
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))
        self.assertIn("per_window_ablation_table", written)
        self.assertFalse(written["training_ran"])
        self.assertFalse(written["runtime_artifact_created"])
        self.assertFalse(written["runtime_action_selection_changed"])


def _write_inputs(
    tmpdir: str,
) -> tuple[dict[str, Path], list[dict[str, object]], dict[str, object], dict[str, object]]:
    root = Path(tmpdir)
    source_path = root / "source.jsonl"
    mask = {action: action in {"stay", "eat", "move_north"} for action in ACTION_NAMES}
    _write_source_records(source_path, mask)
    rows = [
        _row(
            seed=5,
            safe_action="stay",
            top_action="stay",
            observation_input=_encoded_observation_input(mask, energy=0.7),
            source_path=source_path,
            line_number=3,
        ),
        _row(
            seed=13,
            safe_action="eat",
            top_action="eat",
            observation_input=_encoded_observation_input(mask, energy=0.5),
            source_path=source_path,
            line_number=6,
        ),
        _row(
            schema="base",
            safe_action="stay",
            top_action="stay",
            observation_input={"bucket": 10.0},
        ),
        _row(
            schema="base",
            safe_action="eat",
            top_action="eat",
            observation_input={"bucket": 10.1},
        ),
    ]
    paths = {
        "dataset": root / "v165.jsonl",
        "v167_report": root / "v167.json",
        "v168_report": root / "v168.json",
        "v169_report": root / "v169.json",
    }
    paths["dataset"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    v167_report = _v167_report(rows)
    v168_report = _v168_report(rows, v167_digest=str(v167_report["exact_digest"]))
    paths["v167_report"].write_text(
        json.dumps(v167_report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["v168_report"].write_text(
        json.dumps(v168_report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths, rows, v168_report, v167_report


def _write_source_records(path: Path, mask: dict[str, bool]) -> None:
    records = [
        _source_record(actor_id=101, tick=1, mask=mask, energy=0.4),
        _source_record(actor_id=101, tick=2, mask=mask, energy=0.5),
        _source_record(actor_id=101, tick=3, mask=mask, energy=0.7),
        _source_record(actor_id=101, tick=4, mask=mask, energy=0.9),
        _source_record(actor_id=202, tick=1, mask=mask, energy=0.3),
        _source_record(actor_id=202, tick=2, mask=mask, energy=0.5),
    ]
    path.write_text(
        "\n".join(json.dumps({"type": "record", "record": record}, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )


def _source_record(
    *,
    actor_id: int,
    tick: int,
    mask: dict[str, bool],
    energy: float,
) -> dict[str, object]:
    return {
        "agent_id": actor_id,
        "tick": tick,
        "observation_input": _encoded_observation_input(mask, energy=energy),
        "action_mask": mask,
        "requested_action": "move_north",
        "resolved_action": "move_north",
        "runtime_action": "move_north",
        "outcome": {"future_marker": True},
        "private_state": {"hidden": True},
    }


def _row(
    *,
    safe_action: str,
    top_action: str,
    observation_input: Mapping[str, object],
    schema: str = "v165",
    seed: int | None = None,
    source_path: Path | None = None,
    line_number: int | None = None,
) -> dict[str, object]:
    mask = {action: action in {"stay", "eat", "move_north"} for action in ACTION_NAMES}
    targets = []
    for action in ACTION_NAMES:
        available = action in {"stay", "eat", "move_north"}
        value = 10.0 if action == top_action else 1.0
        targets.append(
            {
                "action": action,
                "public_mask": bool(mask[action]),
                "target_available": available,
                "safe_target": action == safe_action,
                "robust_safe_action": action == safe_action,
                "value_target": value if available else None,
                "score_target": value if available else None,
            }
        )
    row: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
            if schema == "v165"
            else M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
        ),
        "feature_policy_id": "fixture_public_features",
        "trainable_public_features": {
            "public_observation": dict(observation_input),
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": targets,
        "safe_action_set": [safe_action],
        "target_classification": "unique_robust_winner",
        "robust_winner_action": safe_action,
    }
    if schema == "v165":
        if seed is None or source_path is None or line_number is None:
            raise AssertionError("v165 rows require source metadata")
        row["row_origin"] = "v165_preterminal_branch_target"
        row["candidate_action_value_targets"] = [
            target for target in targets if target["target_available"] is True
        ]
        row["target_local_primary_support_actions"] = [safe_action]
        row["terminal_population_guard_support_actions"] = [safe_action]
        row["metadata"] = {
            "seed": seed,
            "fixture": "broad",
            "branch_tick": 10,
            "agent_id": seed,
            "branch_id": f"branch-{seed}",
            "source_path": str(source_path),
            "line_number": line_number,
            "source_record_digest": f"digest-{seed}",
            "source_seed_is_support_provenance_not_future_promotion_holdout": True,
            "runtime_requested_action_used_as_scorer_input": False,
            "future_outcomes_used_as_trainable_input": False,
        }
    return row


def _encoded_observation_input(
    mask: dict[str, bool],
    *,
    energy: float,
) -> dict[str, object]:
    patch = [_patch_cell(index) for index in range(PATCH_CELL_COUNT)]
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "metadata": {"agent_id": 1},
        "self": {
            "energy_ratio": energy,
            "hydration_ratio": 0.7,
            "health_ratio": 0.9,
            "injury_load": 0.0,
            "age_norm": 0.2,
            "reproduction_ready": False,
            "matched_diet_ratio": 1.0,
            "trophic_role": "carnivore",
            "meat_mode": "scavenger",
            "season": "wet",
            "water_access_reason": "adjacent_water",
            "hydrology_support_code": 1,
            "refuge_score": 0.1,
            "hazard_type": "none",
            "hazard_level": 0.0,
            "tile_vegetation": 0.4,
            "tile_recovery_debt": 0.0,
            "reproductive_stage": "stage0_asexual",
            "reproductive_expression": "asexual",
            "sexual_reproduction_unlocked": False,
            "reproductive_signal": 0.0,
            "communication_signal": 0.0,
            "mind_inheritance_available": True,
        },
        "local_patch": patch,
        "navigation": {
            "water": {"dx": 1, "dy": 0, "distance": 1, "strength": 1.0},
            "plant": {"dx": 0, "dy": 1, "distance": 1, "strength": 0.4},
            "carrion": {"dx": -1, "dy": 0, "distance": 1, "strength": 0.6},
            "prey": {"dx": 0, "dy": -1, "distance": 1, "strength": 0.2},
        },
        "action_mask": mask,
    }
    return encode_observation_input(observation)


def _patch_cell(index: int) -> dict[str, object]:
    center = index == PATCH_CELL_COUNT // 2
    return {
        "dx": 0,
        "dy": 0,
        "in_bounds": True,
        "terrain": "plain",
        "occupant": "self" if center else "none",
        "same_lineage": center,
        "water_access_reason": "wetland" if center else "none",
        "food": 0.2 if center else 0.0,
        "vegetation": 0.4,
        "recovery_debt": 0.0,
        "fresh_kill_energy": 0.0,
        "carcass_energy": 0.5 if center else 0.0,
        "hazard_type": "none",
        "hazard_level": 0.0,
        "ecology_state": "stable",
        "prey_biomass": 0.0,
        "carrion_signal": 0.5 if center else 0.0,
        "predator_risk": 0.0,
        "reproductive_signal": 0.0,
        "communication_signal": 0.0,
    }


def _v167_report(rows: list[dict[str, object]]) -> dict[str, object]:
    report = {
        "schema_version": "synthetic_v167_report",
        "policy": "synthetic_v167_policy",
        "dataset_digest": stable_payload_digest(rows),
        "preterminal_row_autopsy": [
            {
                "row_index": 0,
                "source_seed": 5,
                "safe_hit": False,
                "rank_of_first_safe_support_neighbor": 2,
            },
            {
                "row_index": 1,
                "source_seed": 13,
                "safe_hit": False,
                "rank_of_first_safe_support_neighbor": 2,
            },
        ],
        "diagnostics_only": True,
    }
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _v168_report(
    rows: list[dict[str, object]],
    *,
    v167_digest: str,
) -> dict[str, object]:
    report = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_POLICY,
        "classification": {
            "primary": EXPECTED_V168_CLASSIFICATION,
            "labels": [EXPECTED_V168_CLASSIFICATION],
        },
        "source_validation": {
            "passed": True,
            "observed_v167_exact_digest": v167_digest,
        },
        "dataset_digest": stable_payload_digest(rows),
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "replay_viewer_schema_changed": False,
        "threshold_tuning_ran": False,
        "k_tuning_ran": False,
        "training_ran": False,
        "shadow_eval_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "diagnostics_only": True,
    }
    report["exact_digest"] = _digest_without_exact(report)
    return report


def _digest_without_exact(payload: dict[str, object]) -> str:
    without_digest = dict(payload)
    without_digest.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(without_digest, sort_keys=True, allow_nan=False))
    )


if __name__ == "__main__":
    unittest.main()
