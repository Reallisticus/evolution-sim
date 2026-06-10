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
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    EXPECTED_V166_CLASSIFICATION,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
    _classification,
    _decoded_public_payload,
    build_candidate_feature_families,
    candidate_feature_leakage_scan,
    run_carrion_survivor_continuation_v168_public_feature_contract_probe,
)
from evolution_sim.mind.provenance import stable_payload_digest

ROOT = Path(__file__).resolve().parents[2]


class MindV3CarrionSurvivorContinuationV168PublicFeatureContractProbeTests(
    unittest.TestCase
):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v168-public-feature-contract-probe"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v168_public_feature_contract_probe"
            ),
        )

    def test_v167_digest_or_classification_mismatch_closes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _aliasing_rows()
            paths, v167_report = _write_inputs(tmpdir, rows=rows)

            result = run_carrion_survivor_continuation_v168_public_feature_contract_probe(
                v167_report_path=paths["v167_report"],
                v165_dataset_path=paths["dataset"],
                output_path=paths["v168_report"],
                expected_v167_exact_digest="wrong",
                expected_v166_exact_digest=str(
                    v167_report["source_validation"]["observed_v166_exact_digest"]
                ),
                expected_v165_dataset_digest=stable_payload_digest(rows),
            )

        self.assertFalse(result["source_validation"]["passed"])
        self.assertIn(
            "v167_unexpected_exact_digest",
            result["source_validation"]["failures"],
        )
        self.assertEqual(
            result["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v168_public_feature_contract_probe_"
                "closed_invalid"
            ),
        )
        self.assertFalse(result["runtime_action_selection_changed"])

    def test_candidate_trainable_payloads_exclude_provenance_private_and_future_keys(
        self,
    ) -> None:
        rows = _aliasing_rows()
        families = build_candidate_feature_families(
            rows=rows,
            source_records_by_row={},
        )

        for family in families.values():
            scan = candidate_feature_leakage_scan(family["payloads"])
            self.assertTrue(scan["passed"], family["feature_family"])

    def test_decoded_projection_sanitizes_public_reason_code_key(self) -> None:
        row = _row(
            seed=5,
            safe_action="stay",
            top_action="stay",
            bucket=0.0,
            encoded_observation=True,
        )

        payload = _decoded_public_payload(row)
        text = json.dumps(payload, sort_keys=True)

        self.assertIn("water_access_code", text)
        self.assertNotIn("reason", text)
        self.assertTrue(candidate_feature_leakage_scan([payload])["passed"])

    def test_classification_branches_cover_no_viable_ready_and_partial(self) -> None:
        source_validation = {"passed": True}
        baseline = {"zero_safe_hit_preterminal_source_seeds": [5, 13]}

        no_viable = _classification(
            source_validation=source_validation,
            leakage_passed=True,
            baseline=baseline,
            candidate_reports=[
                {
                    "feature_family": "decoded_public_numeric_projection",
                    "zero_safe_hit_preterminal_source_seeds": [5, 13],
                }
            ],
        )
        ready = _classification(
            source_validation=source_validation,
            leakage_passed=True,
            baseline=baseline,
            candidate_reports=[
                {
                    "feature_family": "decoded_public_numeric_projection",
                    "zero_safe_hit_preterminal_source_seeds": [],
                    "dominant_predicted_action_share": 0.50,
                    "unsupported_prediction_count": 0,
                    "safe_hit_margin_over_best_trivial": 0.05,
                }
            ],
        )
        partial = _classification(
            source_validation=source_validation,
            leakage_passed=True,
            baseline=baseline,
            candidate_reports=[
                {
                    "feature_family": "decoded_public_numeric_projection",
                    "zero_safe_hit_preterminal_source_seeds": [5, 19],
                    "dominant_predicted_action_share": 0.40,
                    "unsupported_prediction_count": 0,
                    "safe_hit_margin_over_best_trivial": 0.01,
                }
            ],
        )

        self.assertTrue(no_viable.endswith("public_feature_contract_probe_no_viable_projection_closed"))
        self.assertTrue(ready.endswith("public_feature_contract_candidate_ready_for_dataset_expansion_design"))
        self.assertTrue(partial.endswith("partial_feature_contract_support_recommend_another_feature_probe"))

    def test_cli_writes_report_with_exact_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _aliasing_rows()
            paths, v167_report = _write_inputs(tmpdir, rows=rows)
            env = dict(os.environ)
            env["PYTHONPATH"] = "python"
            env["PYTHONHASHSEED"] = "0"

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v168_public_feature_contract_probe",
                    "--v167-report",
                    str(paths["v167_report"]),
                    "--v165-dataset",
                    str(paths["dataset"]),
                    "--output",
                    str(paths["v168_report"]),
                    "--expected-v167-exact-digest",
                    str(v167_report["exact_digest"]),
                    "--expected-v166-exact-digest",
                    str(v167_report["source_validation"]["observed_v166_exact_digest"]),
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
            written = json.loads(paths["v168_report"].read_text(encoding="utf-8"))

        self.assertIn(
            "carrion_survivor_continuation_v168_public_feature_contract_probe_report=",
            completed.stdout,
        )
        self.assertEqual(
            written["schema_version"],
            M3_CARRION_SURVIVOR_CONTINUATION_V168_PUBLIC_FEATURE_CONTRACT_PROBE_SCHEMA_VERSION,
        )
        self.assertEqual(written["exact_digest"], _digest_without_exact(written))
        self.assertFalse(written["training_ran"])
        self.assertFalse(written["runtime_artifact_created"])
        self.assertFalse(written["runtime_action_selection_changed"])


def _aliasing_rows() -> list[dict[str, object]]:
    return [
        _row(seed=5, safe_action="stay", top_action="stay", bucket=0.0),
        _row(seed=13, safe_action="eat", top_action="eat", bucket=0.1),
        _row(schema="base", safe_action="stay", top_action="stay", bucket=10.0),
        _row(schema="base", safe_action="eat", top_action="eat", bucket=10.1),
    ]


def _row(
    *,
    safe_action: str,
    top_action: str,
    bucket: float,
    schema: str = "v165",
    seed: int | None = None,
    encoded_observation: bool = False,
) -> dict[str, object]:
    mask = {action: action in {"stay", "eat"} for action in ACTION_NAMES}
    targets = []
    for action in ACTION_NAMES:
        available = action in {"stay", "eat"}
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
    public_observation = (
        _encoded_observation_input(mask)
        if encoded_observation
        else {"bucket": bucket}
    )
    row: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V165_PRETERMINAL_TARGET_DATASET_EXPANSION_ROW_SCHEMA_VERSION
            if schema == "v165"
            else M3_CARRION_SURVIVOR_CONTINUATION_ACTION_VALUE_TARGET_DATASET_ROW_SCHEMA_VERSION
        ),
        "feature_policy_id": "fixture_public_features",
        "trainable_public_features": {
            "public_observation": public_observation,
            "action_mask": mask,
        },
        "public_action_mask": mask,
        "action_value_targets": targets,
        "safe_action_set": [safe_action],
        "target_classification": "unique_robust_winner",
        "robust_winner_action": safe_action,
    }
    if schema == "v165":
        if seed is None:
            raise AssertionError("v165 rows require a seed")
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
            "source_path": "source.jsonl",
            "line_number": seed,
            "source_record_digest": f"digest-{seed}",
            "source_seed_is_support_provenance_not_future_promotion_holdout": True,
            "runtime_requested_action_used_as_scorer_input": False,
            "future_outcomes_used_as_trainable_input": False,
        }
    return row


def _encoded_observation_input(mask: dict[str, bool]) -> dict[str, object]:
    patch = [_patch_cell(index) for index in range(PATCH_CELL_COUNT)]
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "metadata": {"agent_id": 1},
        "self": {
            "energy_ratio": 0.8,
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


def _write_inputs(
    tmpdir: str,
    *,
    rows: list[dict[str, object]],
) -> tuple[dict[str, Path], dict[str, object]]:
    root = Path(tmpdir)
    paths = {
        "dataset": root / "v165.jsonl",
        "v167_report": root / "v167.json",
        "v168_report": root / "v168.json",
    }
    paths["dataset"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = _v167_report(rows)
    paths["v167_report"].write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths, report


def _v167_report(rows: list[dict[str, object]]) -> dict[str, object]:
    digest = stable_payload_digest(rows)
    report = {
        "schema_version": M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_SCHEMA_VERSION,
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V167_SOURCE_SPLIT_FAILURE_AUTOPSY_POLICY,
        "classification": {
            "primary": (
                "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
                "nearest_neighbor_feature_aliasing_closed_feature_contract_expansion"
            ),
            "labels": [
                "m3_carrion_survivor_continuation_v167_source_split_failure_autopsy_"
                "nearest_neighbor_feature_aliasing_closed_feature_contract_expansion"
            ],
        },
        "source_validation": {
            "observed_v166_exact_digest": "synthetic-v166-digest",
            "observed_v166_classification": EXPECTED_V166_CLASSIFICATION,
        },
        "failure_mode_summary": {
            "primary_failure_mode": "nearest_neighbor_feature_aliasing",
            "zero_safe_hit_source_seeds": [5, 13],
        },
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
        "dataset_digest": digest,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
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
