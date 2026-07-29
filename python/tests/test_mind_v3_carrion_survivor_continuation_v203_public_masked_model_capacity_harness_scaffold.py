from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from evolution_sim.mind import (
    carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract as v202,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold as v203,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.provenance import stable_payload_digest


ROOT = Path(__file__).resolve().parents[2]


class MindV3V203ScaffoldEntrypointTests(unittest.TestCase):
    def test_npm_entrypoint_exists(self) -> None:
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["scripts"][
                "sim:mind:v3:carrion-survivor-continuation-v203-public-masked-model-capacity-harness-scaffold"
            ],
            (
                "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m "
                "evolution_sim.cli."
                "mind_v3_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold"
            ),
        )

    def test_route_decision_fails_closed_without_good_lineage_or_scaffold(self) -> None:
        lineage = v203.route_decision_for_v203(
            source_validation={"passed": False},
            scaffold_surface_audit={"passed": True},
        )
        scaffold = v203.route_decision_for_v203(
            source_validation={"passed": True},
            scaffold_surface_audit={"passed": False},
        )

        self.assertEqual(lineage["selected_route"], v203.LINEAGE_REPAIR_ROUTE)
        self.assertEqual(scaffold["selected_route"], v203.SCAFFOLD_REPAIR_ROUTE)
        self.assertFalse(lineage["training_allowed_by_v203"])
        self.assertFalse(scaffold["promotion_authorized"])


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class MindV3V203ScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_valid_v202_lineage_writes_closed_executable_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v202_path, digest = _write_valid_v202_report(tmpdir)
            output = Path(tmpdir) / "v203-report.json"
            report = v203.run_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold(
                v202_report_path=v202_path,
                output_path=output,
                expected_v202_report_exact_digest=digest,
            )
            on_disk = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(report["source_pin_validation"]["passed"])
        surface = report["public_recurrent_ippo_scaffold_surface_audit"]
        self.assertTrue(surface["passed"])
        self.assertEqual(surface["failures"], [])
        self.assertGreaterEqual(len(surface["checks"]), 31)
        self.assertEqual(
            surface["contract_summary"]["learned_encoder_input_size"],
            604,
        )
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v203.FUTURE_RECURRENT_IPPO_TRAINING_ROUTE,
        )
        self.assertEqual(
            report["classification"]["primary"],
            (
                "m3_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold_ready_for_future_explicit_recurrent_ippo_training_no_training"
            ),
        )
        self.assertTrue(report["scaffold_created"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["training_ran"])
        self.assertFalse(report["training_slice_4_consumed"])
        self.assertFalse(report["training_artifact_created"])
        self.assertFalse(report["runtime_artifact_created"])
        self.assertFalse(report["runtime_action_selection_changed"])
        self.assertFalse(report["dataset_created"])
        self.assertFalse(report["dataset_mutated"])
        self.assertFalse(report["gate_relaxed"])
        self.assertFalse(report["promotion_authorized"])
        self.assertTrue(report["non_promoted"])
        self.assertEqual(on_disk, report)
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_surface_audit_proves_mask_state_alignment_gae_artifact_and_splits(
        self,
    ) -> None:
        audit = v203.public_recurrent_ippo_scaffold_surface_audit()
        checks = audit["checks"]

        self.assertEqual(
            audit["policy"],
            (
                "m3_carrion_survivor_continuation_v203_"
                "public_recurrent_ippo_scaffold_surface_audit_v2"
            ),
        )
        self.assertTrue(checks["learned_input_parity_is_604"])
        self.assertTrue(checks["hard_mask_selected_only_legal_action"])
        self.assertTrue(checks["empty_action_mask_fails_closed"])
        self.assertTrue(checks["recurrent_state_shape_and_update_work"])
        self.assertTrue(checks["finalized_transition_alignment_fields_present"])
        self.assertTrue(
            checks["real_world_finalized_transition_alignment_probe_passed"]
        )
        self.assertTrue(checks["terminated_gae_zero_bootstrap_probe_passed"])
        self.assertTrue(checks["truncated_gae_frozen_bootstrap_probe_passed"])
        self.assertTrue(checks["ppo_ordered_sequence_contract_complete"])
        self.assertTrue(checks["artifact_roundtrip_replay_is_exact"])
        self.assertTrue(checks["artifact_tamper_fails_closed"])
        self.assertTrue(checks["fresh_seed_roles_are_globally_disjoint"])
        self.assertTrue(checks["legacy_diagnostics_are_excluded_from_fresh_roles"])
        self.assertFalse(audit["training_ran"])
        self.assertFalse(audit["optimizer_update_ran"])
        self.assertTrue(audit["simulator_contract_probe_ran"])
        self.assertFalse(audit["simulator_training_rollout_ran"])
        self.assertEqual(audit["simulator_contract_probe_world_count"], 1)
        self.assertFalse(audit["artifact_persisted"])

    def test_bad_v202_digest_routes_to_lineage_repair_without_opening_lifecycle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v202_path, _ = _write_valid_v202_report(tmpdir)
            report = v203.run_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold(
                v202_report_path=v202_path,
                output_path=Path(tmpdir) / "v203-report.json",
                expected_v202_report_exact_digest="bad-v202-digest",
            )

        self.assertFalse(report["source_pin_validation"]["passed"])
        self.assertIn(
            "v202_exact_digest_matches_expected",
            report["source_pin_validation"]["failures"],
        )
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v203.LINEAGE_REPAIR_ROUTE,
        )
        self.assertFalse(report["training_started"])
        self.assertFalse(report["promotion_authorized"])
        self.assertTrue(exact_digest_validation_report(report)["passed"])

    def test_controls_keep_baselines_separate_from_learned_action_authority(
        self,
    ) -> None:
        plan = v203.controls_and_future_training_plan()
        controls = {control["name"]: control for control in plan["required_controls"]}

        self.assertFalse(plan["candidate"]["scripted_or_hardcoded_learner_actions"])
        self.assertFalse(plan["candidate"]["heuristic_fallback_in_candidate"])
        self.assertFalse(controls["masked_random"]["candidate_or_action_authority"])
        self.assertFalse(
            controls["current_linear_mind_v3"]["candidate_or_action_authority"]
        )
        self.assertTrue(
            controls["public_recurrent_ippo"]["candidate_or_action_authority"]
        )
        self.assertEqual(
            plan["future_explicit_route_required"],
            v203.FUTURE_RECURRENT_IPPO_TRAINING_ROUTE,
        )
        self.assertFalse(plan["training_authorized_by_v203"])

    def test_cli_writes_report_and_prints_closed_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            v202_path, digest = _write_valid_v202_report(tmpdir)
            output = Path(tmpdir) / "v203-report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evolution_sim.cli."
                    "mind_v3_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold",
                    "--v202-report",
                    str(v202_path),
                    "--output",
                    str(output),
                    "--expected-v202-report-exact-digest",
                    digest,
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": "python"},
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertIn("public_recurrent_ippo_scaffold_passed=True", result.stdout)
        self.assertIn("training_started=False", result.stdout)
        self.assertIn("training_ran=False", result.stdout)
        self.assertIn("training_slice_4_consumed=False", result.stdout)
        self.assertIn("runtime_action_selection_changed=False", result.stdout)
        self.assertIn("promotion_authorized=False", result.stdout)
        self.assertEqual(
            report["route_decision"]["selected_route"],
            v203.FUTURE_RECURRENT_IPPO_TRAINING_ROUTE,
        )


def _write_valid_v202_report(tmpdir: str) -> tuple[Path, str]:
    report: dict[str, object] = {
        "schema_version": (
            v202.M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_SCHEMA_VERSION
        ),
        "policy": (
            v202.M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_POLICY
        ),
        "source_pin_validation": {"passed": True},
        "v201_fact_assessment": {"passed": True},
        "public_masked_model_capacity_harness_contract": (
            v202.public_masked_model_capacity_harness_contract()
        ),
        "budget_state": {
            "training_slices_consumed": 3,
            "training_slice_budget": 10,
        },
        "route_decision": {
            "selected_route": v203.EXPECTED_V202_ROUTE,
            "recommended_next_route": v203.EXPECTED_V202_ROUTE,
        },
        "classification": {"primary": v203.EXPECTED_V202_CLASSIFICATION},
        **v202.lifecycle_flags(),
    }
    report["exact_digest"] = stable_payload_digest(report)
    path = Path(tmpdir) / "v202-report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, str(report["exact_digest"])


if __name__ == "__main__":
    unittest.main()
