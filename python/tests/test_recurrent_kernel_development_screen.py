from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from torch.nn import functional as F

    from evolution_sim.cli import recurrent_kernel_development_screen as cli
    from evolution_sim.mind import recurrent_actor_critic
    from evolution_sim.mind import recurrent_kernel_development_screen as screen
    from evolution_sim.mind import recurrent_rollout
    from evolution_sim.mind.recurrent_kernel_development_screen import (
        RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES,
        _candidate_integrity_gate,
        _development_screen_contract,
        _fixed_row_tile_4_linear,
        recurrent_kernel_candidate,
        select_development_candidate,
        validate_development_screen_report,
        write_development_screen_report,
    )
    from evolution_sim.mind.provenance import stable_payload_digest


def _fixed_batch_runtime_binding(
    candidate: str,
    *,
    legacy: bool = False,
) -> dict[str, object]:
    implementation = screen._CANDIDATE_IMPLEMENTATION[
        (
            "per_row_bmm_legacy_adapter_baseline_v2"
            if legacy
            else candidate
        )
    ]
    contract = recurrent_rollout.RecurrentFixedBatchRuntimeContract.open_ecology()
    payload: dict[str, object] = {
        "schema_version": (
            recurrent_rollout.RECURRENT_FIXED_BATCH_RUNTIME_SCHEMA_VERSION
        ),
        "batch_capacity": contract.batch_capacity,
        "execution_batch_buckets": list(
            recurrent_rollout.RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS
        ),
        "contract": contract.as_contract(),
        "observed_runtime": {
            "implementation": (
                "PublicRecurrentActorCritic.forward_sequence_time1_"
                "bounded_bucket_batch_v3"
            ),
            "tensor_bridge_dispatch_version": implementation[
                "tensor_bridge_dispatch"
            ],
            "tensor_bridge_actual_branch": implementation[
                "tensor_bridge_actual_branch"
            ],
            "tensor_bridge_probe_branches": {
                "float32": implementation["tensor_bridge_actual_branch"],
                "bool": implementation["tensor_bridge_actual_branch"],
            },
            "inference_context_version": implementation["inference_context"],
            "output_materialization_version": implementation[
                "output_materialization"
            ],
            "observation_projection_version": implementation[
                "observation_projection"
            ],
            "torch_version": "test",
            "numpy_version": "test",
            "native_byte_order": "little",
            "cpu_capability": "test",
            "device_type": "cpu",
            "device_index": None,
            "dtype": "torch.float32",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        },
    }
    payload["exact_digest"] = recurrent_rollout._stable_payload_sha256(payload)
    return payload


def _completed_result(
    candidate: str,
    *,
    scalar_ns: int,
    batched_ns: int,
    rss: int = 100,
    max_ratio: float = 0.25,
    wins: int = 5,
) -> dict[str, object]:
    semantic = "a" * 64
    nonce = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:32]
    return {
        "candidate": candidate,
        "status": "completed",
        "child_nonce": nonce,
        "source_state": {
            "expected_source_sha": "b" * 40,
            "observed_source_sha": "b" * 40,
            "source_clean_including_untracked": True,
        },
        "runtime": {
            "python_version": "3.11.0",
            "torch_version": "test",
            "platform": "test",
            "machine": "test",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "pythonhashseed": "0",
        },
        "fixed_batch_runtime_binding": _fixed_batch_runtime_binding(candidate),
        "legacy_fixed_batch_runtime_binding": _fixed_batch_runtime_binding(
            candidate,
            legacy=True,
        ),
        "peak_rss_bytes": rss,
        "equivalence": {
            "action_mismatch_count": 0,
            "max_hidden_tolerance_ratio": max_ratio,
            "scalar_semantic_sha256": semantic,
            "batched_semantic_sha256": semantic,
            "reference_semantic_sha256": semantic,
            "bucket_matrix": {
                "action_mismatch_count": 0,
                "max_hidden_tolerance_ratio": max_ratio,
                "scalar_semantic_sha256": semantic,
                "batched_semantic_sha256": semantic,
                "reference_semantic_sha256": semantic,
            },
        },
        "collector_equivalence": {
            "semantic_mismatch_count": 0,
            "identity_mismatch_count": 0,
            "hidden_shape_mismatch_count": 0,
            "bootstrap_none_mismatch_count": 0,
            "max_input_hidden_tolerance_ratio": max_ratio,
        },
        "timing": {
            "paired_batched_wins": wins,
            "scalar": {
                "median_elapsed_ns": scalar_ns,
                "semantic_sha256": [semantic] * 5,
            },
            "batched": {
                "median_elapsed_ns": batched_ns,
                "semantic_sha256": [semantic] * 5,
            },
        },
    }


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentKernelDevelopmentScreenTests(unittest.TestCase):
    def test_legacy_projection_executes_the_frozen_two_pass_pipeline(self) -> None:
        observation: dict[str, object] = {"schema_version": "test"}
        source_values = [0.25, -0.5, 1.0]
        quantized = [8192, -16384, 32767]
        decoded = [0.250008, -0.500015, 1.0]
        expected = (0.250008, 1.0)
        with (
            patch.object(
                screen.observations,
                "_validated_observation_input_values",
                return_value=("schema", "encoder", 3, source_values),
            ) as validate,
            patch.object(
                screen.observations,
                "_quantize_observation_input_values",
                return_value=quantized,
            ) as quantize,
            patch.object(
                screen.observations,
                "_dequantize_observation_input_values",
                return_value=decoded,
            ) as dequantize,
            patch.object(
                screen,
                "ecological_policy_values_from_decoded",
                return_value=expected,
            ) as project,
        ):
            observed = screen._legacy_observation_projection(observation)

        self.assertEqual(observed, expected)
        validate.assert_called_once_with(observation)
        quantize.assert_called_once_with(source_values)
        dequantize.assert_called_once_with(quantized)
        project.assert_called_once_with(decoded, source_vector_size=3)

    def test_candidate_context_restores_production_functions(self) -> None:
        original_linear_forward = (
            recurrent_actor_critic._backend_stable_linear_forward
        )
        original_projection = (
            recurrent_rollout.ecological_policy_values_from_observation
        )
        original_projection_version = (
            recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION
        )
        original_bridge_version = (
            recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION
        )
        original_inference_context = recurrent_rollout._recurrent_inference_context
        original_initial_hidden = (
            recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden
        )
        original_forward_step = recurrent_rollout.TorchRecurrentPolicyCore.forward_step
        original_forward_fixed_batch = (
            recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch
        )
        with recurrent_kernel_candidate(
            "per_row_bmm_legacy_adapter_baseline_v2"
        ):
            self.assertIs(
                recurrent_actor_critic._backend_stable_linear_forward,
                original_linear_forward,
            )
            self.assertIsNot(
                recurrent_rollout.ecological_policy_values_from_observation,
                original_projection,
            )
            self.assertEqual(
                recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION,
                "quantize_dequantize_then_diagnostic_filter_v1",
            )
            self.assertEqual(
                recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION,
                "torch_tensor_device_dtype_layout_legacy_v0",
            )
            self.assertIsNot(
                recurrent_rollout._recurrent_inference_context,
                original_inference_context,
            )
            self.assertIs(
                recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden,
                screen._legacy_initial_hidden,
            )
            self.assertIs(
                recurrent_rollout.TorchRecurrentPolicyCore.forward_step,
                screen._legacy_forward_step,
            )
            self.assertIs(
                recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch,
                screen._legacy_forward_fixed_batch,
            )
        self.assertIs(
            recurrent_actor_critic._backend_stable_linear_forward,
            original_linear_forward,
        )
        self.assertIs(
            recurrent_rollout.ecological_policy_values_from_observation,
            original_projection,
        )
        self.assertEqual(
            recurrent_rollout.ECOLOGICAL_POLICY_OBSERVATION_PROJECTION_VERSION,
            original_projection_version,
        )
        self.assertEqual(
            recurrent_rollout.RECURRENT_TENSOR_BRIDGE_DISPATCH_VERSION,
            original_bridge_version,
        )
        self.assertIs(
            recurrent_rollout._recurrent_inference_context,
            original_inference_context,
        )
        self.assertIs(
            recurrent_rollout.TorchRecurrentPolicyCore.initial_hidden,
            original_initial_hidden,
        )
        self.assertIs(
            recurrent_rollout.TorchRecurrentPolicyCore.forward_step,
            original_forward_step,
        )
        self.assertIs(
            recurrent_rollout.TorchRecurrentPolicyCore.forward_fixed_batch,
            original_forward_fixed_batch,
        )

        with recurrent_kernel_candidate(
            "per_row_bmm_fast_adapter_candidate_v2"
        ):
            self.assertIs(
                recurrent_actor_critic._backend_stable_linear_forward,
                original_linear_forward,
            )
            self.assertIs(
                recurrent_rollout.ecological_policy_values_from_observation,
                original_projection,
            )
        with recurrent_kernel_candidate(
            "fixed_row_tile_4_fast_adapter_control_v2"
        ):
            self.assertIs(
                recurrent_actor_critic._backend_stable_linear_forward,
                recurrent_actor_critic._fixed_row_tile_4_linear_forward,
            )

    def test_fixed_tile_matches_one_invariant_dense_tile(self) -> None:
        inputs = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7.0
        bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
        observed = _fixed_row_tile_4_linear(inputs, weight, bias)
        expected = torch.cat(
            (
                F.linear(inputs[:4], weight, bias),
                F.linear(
                    torch.cat((inputs[4:], torch.zeros(3, 3)), dim=0),
                    weight,
                    bias,
                )[:1],
            )
        )
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_adapter_runtime_binding_rejects_non_cpu_screen_identity(self) -> None:
        candidate = "per_row_bmm_fast_adapter_candidate_v2"
        binding = _fixed_batch_runtime_binding(candidate)
        binding["observed_runtime"]["device_type"] = "cuda"
        binding["observed_runtime"]["device_index"] = 0
        binding["exact_digest"] = recurrent_rollout._stable_payload_sha256(
            {
                key: value
                for key, value in binding.items()
                if key != "exact_digest"
            }
        )

        with self.assertRaisesRegex(ValueError, "adapter provenance"):
            screen._validated_adapter_runtime_binding(
                binding,
                implementation=screen._CANDIDATE_IMPLEMENTATION[candidate],
                field="test binding",
            )

    def test_selection_requires_integrity_and_absolute_speed(self) -> None:
        baseline = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
            scalar_ns=1000,
            batched_ns=950,
        )
        native = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[1],
            scalar_ns=900,
            batched_ns=800,
        )
        fused = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[2],
            scalar_ns=800,
            batched_ns=850,
        )
        tiled = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[3],
            scalar_ns=1000,
            batched_ns=875,
        )

        def integrity(
            result: dict[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "passed": True,
                "reasons": [],
                "max_numeric_tolerance_ratio": 0.25,
            }

        confirmation = copy.deepcopy(baseline)
        confirmation["child_nonce"] = "f" * 32
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            selection = select_development_candidate(
                [baseline, native, fused, tiled],
                baseline_confirmation=confirmation,
            )
        self.assertEqual(
            selection["selected_candidate"],
            "per_row_bmm_fast_adapter_candidate_v2",
        )
        self.assertTrue(selection["selection_authorized"])
        gates = selection["candidate_gates"]
        self.assertTrue(
            gates["fixed_row_tile_4_fast_adapter_control_v2"]["evidence_valid"]
        )
        self.assertTrue(
            gates["fixed_row_tile_4_legacy_adapter_control_v2"]["evidence_valid"]
        )

        runtime_drift = copy.deepcopy(native)
        runtime_drift["fixed_batch_runtime_binding"]["observed_runtime"][
            "cpu_capability"
        ] = "drifted"
        runtime_drift["fixed_batch_runtime_binding"]["exact_digest"] = (
            recurrent_rollout._stable_payload_sha256(
                {
                    key: value
                    for key, value in runtime_drift[
                        "fixed_batch_runtime_binding"
                    ].items()
                    if key != "exact_digest"
                }
            )
        )
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            rejected_runtime_drift = select_development_candidate(
                [baseline, runtime_drift, fused, tiled],
                baseline_confirmation=confirmation,
            )
        self.assertFalse(rejected_runtime_drift["selection_authorized"])
        self.assertIn("runtime isolation", rejected_runtime_drift["reason"])

        native["equivalence"]["scalar_semantic_sha256"] = "b" * 64
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            drifted = select_development_candidate(
                [baseline, native, fused, tiled],
                baseline_confirmation=confirmation,
            )
        self.assertIsNone(drifted["selected_candidate"])
        self.assertIn(
            "behavior semantics differ",
            " ".join(
                drifted["candidate_gates"][
                    "per_row_bmm_fast_adapter_candidate_v2"
                ]["reasons"]
            ),
        )
        native["equivalence"]["scalar_semantic_sha256"] = "a" * 64

        def failed_control_integrity(
            result: dict[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            if result["candidate"] == "fixed_row_tile_4_fast_adapter_control_v2":
                return {
                    "passed": False,
                    "reasons": ["control evidence failed"],
                    "max_numeric_tolerance_ratio": 0.8,
                }
            return integrity(result)

        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=failed_control_integrity,
        ):
            missing_control = select_development_candidate(
                [baseline, native, fused, tiled],
                baseline_confirmation=confirmation,
            )
        self.assertEqual(
            missing_control["selected_candidate"],
            "per_row_bmm_fast_adapter_candidate_v2",
        )
        self.assertFalse(
            missing_control["candidate_gates"][
                "fixed_row_tile_4_fast_adapter_control_v2"
            ]["evidence_valid"]
        )

    def test_missing_evidence_and_duplicate_candidates_fail_closed(self) -> None:
        incomplete = _candidate_integrity_gate(
            {
                "candidate": "per_row_bmm_fast_adapter_candidate_v2",
                "status": "completed",
            }
        )
        self.assertFalse(incomplete["passed"])
        self.assertTrue(incomplete["reasons"])

        baseline = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
            scalar_ns=1000,
            batched_ns=1100,
        )
        duplicate = select_development_candidate(
            [baseline, baseline, baseline, baseline],
            baseline_confirmation=copy.deepcopy(baseline),
        )
        self.assertFalse(duplicate["selection_authorized"])
        self.assertIn("order", duplicate["reason"])

    def test_report_requires_development_namespace_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            development_root = root / "development-screens"
            with patch.dict(
                os.environ,
                {
                    "EVOLUTION_SIM_OPEN_ECOLOGY_DEVELOPMENT_SCREEN_ROOT": str(
                        development_root
                    )
                },
            ), patch.object(
                screen,
                "validate_development_screen_report",
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "approved",
                ):
                    write_development_screen_report(
                        root / "authority" / "development-screens" / "report.json",
                        {"development_only": True},
                    )
                report = development_root / "report.json"
                write_development_screen_report(
                    report,
                    {"development_only": True},
                )
                self.assertTrue(report.is_file())
                with self.assertRaisesRegex(ValueError, "clobbering"):
                    write_development_screen_report(
                        report,
                        {"development_only": True},
                    )

    def test_report_validation_reconstructs_selection_contract_and_digest(
        self,
    ) -> None:
        selection = {
            "selected_candidate": None,
            "selection_authorized": False,
            "reason": "test negative",
        }
        report: dict[str, object] = {
            "schema_version": (
                screen.RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION
            ),
            "development_only": True,
            "launch_authorized": False,
            "authority_evidence_eligible": False,
            "scientific_result": False,
            "training_run": False,
            "training_artifact_created": False,
            "runtime_action_selection_changed": False,
            "promotion_authorized": False,
            "source_state": {
                "repository_root": str(screen._REPOSITORY_ROOT),
                "expected_source_sha": "b" * 40,
                "observed_source_sha": "b" * 40,
                "source_clean_including_untracked": True,
                "imported_module_paths": dict(screen._SOURCE_BOUND_MODULE_PATHS),
            },
            "screen_contract": _development_screen_contract(),
            "candidate_results": [
                {"candidate": name}
                for name in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES
            ],
            "baseline_confirmation": {
                "candidate": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
            },
            "selection": selection,
        }
        report["exact_digest"] = stable_payload_digest(report)
        with patch.object(
            screen,
            "select_development_candidate",
            return_value=selection,
        ):
            validate_development_screen_report(report)
            drifted = copy.deepcopy(report)
            drifted["screen_contract"]["numeric_headroom"] = 0.9
            with self.assertRaisesRegex(ValueError, "declared contract"):
                validate_development_screen_report(drifted)
            drifted = copy.deepcopy(report)
            drifted["exact_digest"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_development_screen_report(drifted)
            drifted = copy.deepcopy(report)
            drifted["source_state"]["imported_module_paths"]["observations"] = (
                "../arbitrary.py"
            )
            drifted["exact_digest"] = stable_payload_digest(
                {
                    key: value
                    for key, value in drifted.items()
                    if key != "exact_digest"
                }
            )
            with self.assertRaisesRegex(ValueError, "exact-source binding"):
                validate_development_screen_report(drifted)

    def test_cli_is_explicitly_development_only(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--development-run"):
            cli.main(["--expected-source-sha", "a" * 40])
        with self.assertRaisesRegex(SystemExit, "canonical .json"):
            cli.main(
                [
                    "--development-run",
                    "--expected-source-sha",
                    "a" * 40,
                    "--report",
                    "screen.pt",
                ]
            )
        with self.assertRaisesRegex(SystemExit, "--child-nonce"):
            cli.main(
                [
                    "--development-run",
                    "--expected-source-sha",
                    "a" * 40,
                    "--child-candidate",
                    "per_row_bmm_fast_adapter_candidate_v2",
                ]
            )
        closed_report = {
            "selection": {"selected_candidate": None},
            "development_only": True,
            "launch_authorized": False,
        }
        with (
            patch.object(cli, "validate_development_screen_report_path"),
            patch.object(cli, "run_development_screen", return_value=closed_report),
            patch.object(cli, "write_development_screen_report") as write_report,
        ):
            observed = cli.main(
                [
                    "--development-run",
                    "--expected-source-sha",
                    "a" * 40,
                ]
            )
        self.assertEqual(observed, 0)
        write_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
