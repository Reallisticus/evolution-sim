from __future__ import annotations

import copy
import hashlib
import json
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
    from evolution_sim.mind import recurrent_accuracy_screen as accuracy
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
    nonce = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
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
            "torch_cuda_available": False,
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

    @staticmethod
    def _spawn_protocol_test_child(
        source: str,
    ) -> object:
        return screen.subprocess.Popen(
            [screen.sys.executable, "-c", source],
            stdin=screen.subprocess.PIPE,
            stdout=screen.subprocess.PIPE,
            stderr=screen.subprocess.DEVNULL,
            text=True,
        )

    @staticmethod
    def _close_protocol_test_child(process: object) -> None:
        if process.poll() is None:
            process.kill()
        process.wait()
        for stream_name in ("stdin", "stdout"):
            stream = getattr(process, stream_name, None)
            if stream is not None and not stream.closed:
                stream.close()

    def test_interactive_frame_partial_write_obeys_deadline(self) -> None:
        process = self._spawn_protocol_test_child(
            "import os,time;os.write(1,b'{');time.sleep(5)"
        )
        started = screen.time.monotonic()
        try:
            with self.assertRaisesRegex(ValueError, "frame timed out"):
                screen._read_interactive_child_frame(
                    process,
                    timeout_seconds=0.1,
                )
            self.assertLess(screen.time.monotonic() - started, 1.5)
        finally:
            self._close_protocol_test_child(process)

    def test_interactive_frame_rejects_oversize_and_extra_bytes(self) -> None:
        eof_terminated = self._spawn_protocol_test_child(
            "import os;os.write(1,b'{}')"
        )
        try:
            self.assertEqual(
                screen._read_interactive_child_frame(
                    eof_terminated,
                    timeout_seconds=1,
                ),
                {},
            )
        finally:
            self._close_protocol_test_child(eof_terminated)

        oversize = self._spawn_protocol_test_child(
            "import os;os.write(1,b'x'*33)"
        )
        try:
            with (
                patch.object(screen, "_INTERACTIVE_FRAME_MAX_BYTES", 32),
                self.assertRaisesRegex(ValueError, "maximum byte count"),
            ):
                screen._read_interactive_child_frame(
                    oversize,
                    timeout_seconds=1,
                )
        finally:
            self._close_protocol_test_child(oversize)

        extra = self._spawn_protocol_test_child(
            "import os;os.write(1,b'{}\\nextra')"
        )
        try:
            with self.assertRaisesRegex(ValueError, "extra bytes"):
                screen._read_interactive_child_frame(
                    extra,
                    timeout_seconds=1,
                )
        finally:
            self._close_protocol_test_child(extra)

    def test_interactive_exit_drain_caps_trailing_stdout_flood(self) -> None:
        process = self._spawn_protocol_test_child(
            "import os,time;"
            "os.write(1,b'{}\\n');"
            "time.sleep(0.1);"
            "os.write(1,b'x'*(1024*1024))"
        )
        try:
            self.assertEqual(
                screen._read_interactive_child_frame(
                    process,
                    timeout_seconds=1,
                ),
                {},
            )
            started = screen.time.monotonic()
            with (
                patch.object(
                    screen,
                    "_INTERACTIVE_TRAILING_STDOUT_MAX_BYTES",
                    1024,
                ),
                self.assertRaisesRegex(ValueError, "maximum byte count"),
            ):
                screen._wait_for_interactive_child_exit(
                    process,
                    timeout_seconds=2,
                )
            self.assertLess(screen.time.monotonic() - started, 1.5)
        finally:
            self._close_protocol_test_child(process)

    def test_process_start_identity_uses_the_accuracy_authority(self) -> None:
        self.assertIs(
            screen._process_start_identity,
            accuracy.process_start_identity,
        )
        self.assertEqual(
            screen._process_start_identity(os.getpid()),
            accuracy.process_start_identity(os.getpid()),
        )

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
            "per_row_bmm_fast_adapter_control_v3"
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
            "fixed_row_tile_4_fast_adapter_candidate_v3"
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

    def test_accuracy_validation_uses_production_autograd_dispatch(self) -> None:
        public_forward = {
            "aggregate": {"forward_gate_passed": True}
        }
        gradients = {
            "aggregate": {"gradient_gate_passed": True}
        }
        evidence: dict[str, object] = {
            "applicable": True,
            "candidate_locked_before_holdout": True,
            "holdout_access_count": 0,
            "holdout_consumed": False,
            "public_forward": public_forward,
            "gradients": gradients,
            "holdout_forward": None,
            "holdout_receipt": None,
            "preregistration_digest": "a" * 64,
        }
        evidence["preholdout_candidate_exact_digest"] = (
            stable_payload_digest(
                screen._preholdout_accuracy_payload(evidence)
            )
        )
        candidate = {
            "accuracy_preregistration_digest": "a" * 64,
            "accuracy_evidence": evidence,
        }
        with (
            patch.object(
                accuracy,
                "validate_public_forward_report",
            ) as validate_public,
            patch.object(
                accuracy,
                "validate_gradient_probe_report",
            ) as validate_gradients,
        ):
            screen._validate_selectable_preholdout_accuracy(
                candidate,
                reexecute_public_evidence=True,
                expect_holdout_consumed=False,
            )
        for observed in (validate_public, validate_gradients):
            forwarded = observed.call_args.kwargs["candidate_forward"]
            self.assertIs(
                forwarded,
                recurrent_actor_critic._backend_stable_linear,
            )
            self.assertIsNot(
                forwarded,
                recurrent_actor_critic._fixed_row_tile_4_linear_forward,
            )
            self.assertIsNot(forwarded, screen._fixed_row_tile_4_linear)
        drifted = copy.deepcopy(candidate)
        drifted["accuracy_evidence"]["holdout_access_count"] = 1
        drifted["accuracy_evidence"]["holdout_consumed"] = True
        with self.assertRaisesRegex(ValueError, "preholdout|schema|binding"):
            screen._validate_selectable_preholdout_accuracy(
                drifted,
                reexecute_public_evidence=False,
                expect_holdout_consumed=False,
            )
        boolean_preholdout = copy.deepcopy(candidate)
        boolean_preholdout["accuracy_evidence"][
            "holdout_access_count"
        ] = False
        with self.assertRaisesRegex(ValueError, "binding"):
            screen._validate_selectable_preholdout_accuracy(
                boolean_preholdout,
                reexecute_public_evidence=False,
                expect_holdout_consumed=False,
            )
        boolean_completed = copy.deepcopy(candidate)
        boolean_completed_evidence = boolean_completed[
            "accuracy_evidence"
        ]
        boolean_completed_evidence["holdout_access_count"] = True
        boolean_completed_evidence["holdout_consumed"] = True
        boolean_completed_evidence["preholdout_evidence_digest"] = "b" * 64
        boolean_completed_evidence["retirement_marker_path"] = (
            "/tmp/development-screens/"
            "recurrent-kernel-development-screen.holdout-retired.json"
        )
        boolean_completed_evidence["retirement_marker_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "binding"):
            screen._validate_selectable_preholdout_accuracy(
                boolean_completed,
                reexecute_public_evidence=False,
                expect_holdout_consumed=True,
            )

    def test_adapter_runtime_binding_rejects_non_cpu_screen_identity(self) -> None:
        candidate = "per_row_bmm_fast_adapter_control_v3"
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
        boolean_binding = _fixed_batch_runtime_binding(candidate)
        boolean_binding["observed_runtime"]["torch_num_threads"] = True
        boolean_binding["exact_digest"] = (
            recurrent_rollout._stable_payload_sha256(
                {
                    key: value
                    for key, value in boolean_binding.items()
                    if key != "exact_digest"
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "adapter provenance"):
            screen._validated_adapter_runtime_binding(
                boolean_binding,
                implementation=screen._CANDIDATE_IMPLEMENTATION[candidate],
                field="test binding",
            )
        candidate_binding = _fixed_batch_runtime_binding(candidate)
        legacy_binding = _fixed_batch_runtime_binding(
            candidate,
            legacy=True,
        )
        runtime = {
            "python_version": "3.12.0",
            "torch_version": "test",
            "torch_cuda_available": False,
            "platform": "test-platform",
            "machine": "test-machine",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "pythonhashseed": "0",
        }
        screen._validated_child_runtime(
            runtime,
            candidate_runtime_binding=candidate_binding,
            legacy_runtime_binding=legacy_binding,
        )
        runtime["torch_cuda_available"] = True
        with self.assertRaisesRegex(ValueError, "runtime binding"):
            screen._validated_child_runtime(
                runtime,
                candidate_runtime_binding=candidate_binding,
                legacy_runtime_binding=legacy_binding,
            )
        runtime["torch_cuda_available"] = False
        runtime["torch_num_threads"] = True
        with self.assertRaisesRegex(ValueError, "runtime binding"):
            screen._validated_child_runtime(
                runtime,
                candidate_runtime_binding=candidate_binding,
                legacy_runtime_binding=legacy_binding,
            )

    def test_collector_bootstrap_counts_reject_boolean_aliases(self) -> None:
        semantic = "d" * 64
        runtime_digest = "a" * 64
        legacy_runtime_digest = "b" * 64
        transition_count = 3
        collector = {
            "transition_count": transition_count,
            "batched_transition_count": transition_count,
            "paired_transition_count": transition_count,
            "numeric_transition_comparison_count": transition_count,
            "hidden_component_comparison_count": (
                transition_count * screen._HIDDEN_SIZE
            ),
            "bootstrap_value_comparison_count": True,
            "semantic_mismatch_count": 0,
            "identity_mismatch_count": 0,
            "hidden_shape_mismatch_count": 0,
            "bootstrap_none_mismatch_count": 0,
            "max_abs_input_hidden_error": 0.0,
            "max_abs_logprob_error": 0.0,
            "max_abs_entropy_error": 0.0,
            "max_abs_value_error": 0.0,
            "max_abs_bootstrap_value_error": 0.0,
            "max_input_hidden_tolerance_ratio": 0.0,
            "max_logprob_tolerance_ratio": 0.0,
            "max_entropy_tolerance_ratio": 0.0,
            "max_value_tolerance_ratio": 0.0,
            "max_bootstrap_value_tolerance_ratio": 0.0,
            "ordered_merge_semantic_sha256": semantic,
        }
        with self.assertRaisesRegex(
            ValueError,
            "collector bootstrap comparison count",
        ):
            screen._validate_collector_and_timing(
                {
                    "collector_equivalence": collector,
                    "timing": {},
                    "fixed_batch_runtime_binding": {
                        "exact_digest": runtime_digest
                    },
                }
            )

        valid_collector = copy.deepcopy(collector)
        valid_collector["bootstrap_value_comparison_count"] = 1
        timing_order = [
            ["scalar", "batched"]
            if repeat % 2 == 0
            else ["batched", "scalar"]
            for repeat in range(
                screen.RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS
            )
        ]
        timing = {
            "warmup_pairs": (
                screen.RECURRENT_KERNEL_DEVELOPMENT_SCREEN_WARMUP_PAIRS
            ),
            "repeat_count": (
                screen.RECURRENT_KERNEL_DEVELOPMENT_SCREEN_TIMING_REPEATS
            ),
            "timing_order": timing_order,
            "paired_batched_wins": 5,
            "own_path_speedup": 1.0 - (90 / 100),
            "scalar": {
                "elapsed_ns": [100] * 5,
                "median_elapsed_ns": 100,
                "semantic_sha256": [semantic] * 5,
                "samples": [],
            },
            "batched": {
                "elapsed_ns": [90] * 5,
                "median_elapsed_ns": 90,
                "semantic_sha256": [semantic] * 5,
                "samples": [],
            },
        }

        def timed_samples(
            _samples: object,
            *,
            mode: str,
            **_kwargs: object,
        ) -> list[dict[str, object]]:
            runtime_digests = [] if mode == "scalar" else [runtime_digest]
            path = {
                "bootstrap_value_count": 1,
                "fixed_batch_runtime_sha256": runtime_digests,
                "semantic_sha256": semantic,
            }
            return [{"collector_path": dict(path)} for _ in range(5)]

        def legacy_path(
            _path: object,
            *,
            mode: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "bootstrap_value_count": 1,
                "fixed_batch_runtime_sha256": (
                    [] if mode == "scalar" else [legacy_runtime_digest]
                ),
                "semantic_sha256": semantic,
            }

        def comparison(bootstrap_count: object) -> dict[str, object]:
            return {
                "transition_count": transition_count,
                "batched_transition_count": transition_count,
                "paired_transition_count": transition_count,
                "numeric_transition_comparison_count": transition_count,
                "hidden_component_comparison_count": (
                    transition_count * screen._HIDDEN_SIZE
                ),
                "bootstrap_value_comparison_count": bootstrap_count,
                "semantic_mismatch_count": 0,
                "identity_mismatch_count": 0,
                "hidden_shape_mismatch_count": 0,
                "bootstrap_none_mismatch_count": 0,
                "max_abs_input_hidden_error": 0.0,
                "max_abs_logprob_error": 0.0,
                "max_abs_entropy_error": 0.0,
                "max_abs_value_error": 0.0,
                "max_abs_bootstrap_value_error": 0.0,
                "max_input_hidden_tolerance_ratio": 0.0,
                "max_logprob_tolerance_ratio": 0.0,
                "max_entropy_tolerance_ratio": 0.0,
                "max_value_tolerance_ratio": 0.0,
                "max_bootstrap_value_tolerance_ratio": 0.0,
                "ordered_merge_semantic_sha256": semantic,
            }

        for boolean_mode in ("scalar", "batched"):
            with self.subTest(boolean_mode=boolean_mode):
                candidate = {
                    "collector_equivalence": valid_collector,
                    "timing": timing,
                    "fixed_batch_runtime_binding": {
                        "exact_digest": runtime_digest
                    },
                    "legacy_fixed_batch_runtime_binding": {
                        "exact_digest": legacy_runtime_digest
                    },
                    "legacy_runtime_paths": {
                        "scalar": {},
                        "batched": {},
                    },
                    "legacy_runtime_equivalence": {
                        mode: comparison(
                            True if mode == boolean_mode else 1
                        )
                        for mode in ("scalar", "batched")
                    },
                }
                with (
                    patch.object(
                        screen,
                        "_validated_d4_timed_samples",
                        side_effect=timed_samples,
                    ),
                    patch.object(
                        screen,
                        "_validated_d4_collector_path",
                        side_effect=legacy_path,
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        (
                            "legacy_runtime."
                            f"{boolean_mode}."
                            "bootstrap_value_comparison_count"
                        ),
                    ),
                ):
                    screen._validate_collector_and_timing(candidate)

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
        tiled_legacy = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[2],
            scalar_ns=1000,
            batched_ns=875,
        )
        fixed_candidate = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[3],
            scalar_ns=850,
            batched_ns=700,
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
        confirmation["child_nonce"] = "f" * 64
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            selection = select_development_candidate(
                [baseline, native, tiled_legacy, fixed_candidate],
                baseline_confirmation=confirmation,
            )
        self.assertEqual(
            selection["selected_candidate"],
            "fixed_row_tile_4_fast_adapter_candidate_v3",
        )
        self.assertTrue(selection["selection_authorized"])
        gates = selection["candidate_gates"]
        self.assertTrue(
            gates["fixed_row_tile_4_fast_adapter_candidate_v3"]["evidence_valid"]
        )
        self.assertTrue(
            gates["fixed_row_tile_4_legacy_adapter_control_v3"]["evidence_valid"]
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
                [baseline, runtime_drift, tiled_legacy, fixed_candidate],
                baseline_confirmation=confirmation,
            )
        self.assertFalse(rejected_runtime_drift["selection_authorized"])
        self.assertIn("runtime isolation", rejected_runtime_drift["reason"])

        fixed_candidate["equivalence"]["scalar_semantic_sha256"] = "b" * 64
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            drifted = select_development_candidate(
                [baseline, native, tiled_legacy, fixed_candidate],
                baseline_confirmation=confirmation,
            )
        self.assertIsNone(drifted["selected_candidate"])
        self.assertIn(
            "behavior semantics differ",
            " ".join(
                drifted["candidate_gates"][
                    "fixed_row_tile_4_fast_adapter_candidate_v3"
                ]["reasons"]
            ),
        )
        fixed_candidate["equivalence"]["scalar_semantic_sha256"] = "a" * 64

        def failed_control_integrity(
            result: dict[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            if result["candidate"] == "per_row_bmm_fast_adapter_control_v3":
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
                [baseline, native, tiled_legacy, fixed_candidate],
                baseline_confirmation=confirmation,
            )
        self.assertIsNone(missing_control["selected_candidate"])
        self.assertFalse(
            missing_control["candidate_gates"][
                "per_row_bmm_fast_adapter_control_v3"
            ]["evidence_valid"]
        )

    def test_missing_evidence_and_duplicate_candidates_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            screen.run_candidate_screen(
                candidate=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
                expected_source_sha="a" * 40,
                child_nonce="b" * 64,
                report_path=(
                    "/tmp/development-screens/"
                    "recurrent-kernel-development-screen.json"
                ),
                accuracy_preregistration_digest="c" * 64,
                parent_process_id=123,
                parent_process_start_identity="synthetic-parent-start",
                timing_repeats=1,
            )
        incomplete = _candidate_integrity_gate(
            {
                "candidate": "per_row_bmm_fast_adapter_control_v3",
                "status": "completed",
            }
        )
        self.assertFalse(incomplete["passed"])
        self.assertTrue(incomplete["reasons"])
        with self.assertRaisesRegex(ValueError, "envelope schema"):
            screen._validate_candidate_identity(
                {
                    "candidate": (
                        "per_row_bmm_fast_adapter_control_v3"
                    ),
                    "unexpected": "field",
                }
            )

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
                        root
                        / "authority"
                        / "development-screens"
                        / "recurrent-kernel-development-screen.json",
                        {"development_only": True},
                    )
                report = (
                    development_root / "recurrent-kernel-development-screen.json"
                )
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

    def test_retirement_markers_precede_late_capability_issue(self) -> None:
        preimage = "1" * 64
        capability_commitment = accuracy.holdout_capability_commitment(
            preimage
        )
        source_state = {
            "observed_source_sha": "a" * 40,
            "accuracy_method_document_sha256": "b" * 64,
        }
        process_binding = {
            "parent_process_id": 123,
            "parent_process_start_identity": "synthetic-parent-start",
            "child_process_id": 456,
            "child_process_start_identity": "synthetic-child-start",
            "child_nonce": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as raw:
            development_root = Path(raw) / "development-screens"
            report_path = (
                development_root
                / "recurrent-kernel-development-screen.json"
            )
            with patch.dict(
                os.environ,
                {
                    "EVOLUTION_SIM_OPEN_ECOLOGY_DEVELOPMENT_SCREEN_ROOT": str(
                        development_root
                    )
                },
            ):
                (
                    source_marker_path,
                    source_marker,
                    source_marker_sha256,
                ) = screen._write_source_retirement_marker(
                    report_path=report_path,
                    source_state=source_state,
                    accuracy_preregistration_digest="d" * 64,
                    parent_process_id=123,
                    parent_process_start_identity=(
                        "synthetic-parent-start"
                    ),
                    capability_commitment_sha256=capability_commitment,
                )
                self.assertEqual(
                    source_marker_path.name,
                    screen._CANONICAL_SOURCE_RETIREMENT_MARKER_NAME,
                )
                self.assertTrue(source_marker_path.is_file())
                self.assertEqual(
                    hashlib.sha256(
                        source_marker_path.read_bytes()
                    ).hexdigest(),
                    source_marker_sha256,
                )
                accuracy._validate_retirement_marker_file(
                    path=source_marker_path,
                    file_sha256=source_marker_sha256,
                    expected_marker=source_marker,
                    label="source",
                )
                self.assertNotIn(
                    preimage,
                    source_marker_path.read_text(encoding="utf-8"),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "validated closing baseline",
                ):
                    accuracy.issue_holdout_capability(
                        preimage=preimage,
                        capability_commitment_sha256=(
                            capability_commitment
                        ),
                        preholdout_evidence_digest="e" * 64,
                        source_retirement_marker_sha256=(
                            source_marker_sha256
                        ),
                        holdout_retirement_marker_sha256="f" * 64,
                        candidate_identity=(
                            accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
                        ),
                        closing_baseline_validated=False,
                        source_retirement_marker_exists=True,
                        holdout_retirement_marker_exists=True,
                    )
                (
                    holdout_marker_path,
                    holdout_marker,
                    holdout_marker_sha256,
                ) = screen._write_holdout_retirement_marker(
                    report_path=(
                        source_marker_path.parent
                        / "recurrent-kernel-development-screen.json"
                    ),
                    preholdout_evidence_digest="e" * 64,
                    candidate_process_binding=process_binding,
                    source_retirement_marker_path=source_marker_path,
                    source_retirement_marker_sha256=(
                        source_marker_sha256
                    ),
                    capability_commitment_sha256=capability_commitment,
                    baseline_confirmation_digest="f" * 64,
                )
                self.assertEqual(
                    holdout_marker_path.name,
                    screen._CANONICAL_HOLDOUT_RETIREMENT_MARKER_NAME,
                )
                self.assertTrue(holdout_marker_path.is_file())
                self.assertEqual(
                    hashlib.sha256(
                        holdout_marker_path.read_bytes()
                    ).hexdigest(),
                    holdout_marker_sha256,
                )
                accuracy._validate_retirement_marker_file(
                    path=holdout_marker_path,
                    file_sha256=holdout_marker_sha256,
                    expected_marker=holdout_marker,
                    label="holdout",
                )
                self.assertEqual(
                    source_marker["exact_digest"],
                    holdout_marker[
                        "source_retirement_marker_exact_digest"
                    ],
                )
                self.assertNotIn(
                    preimage,
                    holdout_marker_path.read_text(encoding="utf-8"),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "does not match its commitment",
                ):
                    accuracy.issue_holdout_capability(
                        preimage="2" * 64,
                        capability_commitment_sha256=(
                            capability_commitment
                        ),
                        preholdout_evidence_digest="e" * 64,
                        source_retirement_marker_sha256=(
                            source_marker_sha256
                        ),
                        holdout_retirement_marker_sha256=(
                            holdout_marker_sha256
                        ),
                        candidate_identity=(
                            accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
                        ),
                        closing_baseline_validated=True,
                        source_retirement_marker_exists=True,
                        holdout_retirement_marker_exists=True,
                    )
                capability = accuracy.issue_holdout_capability(
                    preimage=preimage,
                    capability_commitment_sha256=capability_commitment,
                    preholdout_evidence_digest="e" * 64,
                    source_retirement_marker_sha256=(
                        source_marker_sha256
                    ),
                    holdout_retirement_marker_sha256=(
                        holdout_marker_sha256
                    ),
                    candidate_identity=(
                        accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
                    ),
                    closing_baseline_validated=True,
                    source_retirement_marker_exists=True,
                    holdout_retirement_marker_exists=True,
                )
                self.assertEqual(capability.token, preimage)

    def test_nonselectable_accuracy_sentinel_rejects_tamper(self) -> None:
        preregistration_digest = "a" * 64
        sentinel = screen._nonselectable_accuracy_sentinel(
            preregistration_digest=preregistration_digest
        )
        screen._validate_nonselectable_accuracy_sentinel(
            sentinel,
            preregistration_digest=preregistration_digest,
        )
        drifted = copy.deepcopy(sentinel)
        drifted["holdout_access_count"] = 1
        drifted["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in drifted.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(ValueError, "sentinel"):
            screen._validate_nonselectable_accuracy_sentinel(
                drifted,
                preregistration_digest=preregistration_digest,
            )

    def test_report_validation_reconstructs_selection_contract_and_digest(
        self,
    ) -> None:
        selection = {
            "selected_candidate": None,
            "selection_authorized": False,
            "reason": "test negative",
        }
        source_state = {
            "repository_root": str(screen._REPOSITORY_ROOT),
            "expected_source_sha": "b" * 40,
            "observed_source_sha": "b" * 40,
            "source_clean_including_untracked": True,
            "detached_head": True,
            "imported_module_paths": dict(screen._SOURCE_BOUND_MODULE_PATHS),
            "imported_module_sha256": screen._source_bound_module_sha256(),
            "module_byte_sha256": screen._source_bound_path_sha256(),
            "source_module_bundle_sha256": stable_payload_digest(
                screen._source_bound_path_sha256()
            ),
            "accuracy_method_document_path": (
                screen._ACCURACY_METHOD_DOCUMENT_PATH
            ),
            "accuracy_method_document_sha256": screen._source_artifact_sha256(
                screen._ACCURACY_METHOD_DOCUMENT_PATH
            ),
        }
        preregistration = {
            "exact_digest": "c" * 64,
            "corpus_contract": {"schema_version": "synthetic_contract_v1"},
        }
        preholdout = {
            "schema_version": screen._PREHOLDOUT_EVIDENCE_VERSION,
            "admission_authorized": False,
            "reasons": ["test negative"],
            "exact_digest": "d" * 64,
        }
        candidate_preholdout = {
            "candidate": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[-1],
            "status": "preholdout_completed",
            "child_nonce": "9" * 64,
            "accuracy_evidence": {
                "public_forward": {"synthetic": "public"},
                "gradients": {"synthetic": "gradients"},
                "holdout_forward": None,
                "holdout_receipt": None,
            },
        }
        candidate_result = {
            "candidate": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[-1],
            "status": "failed",
            "child_nonce": "9" * 64,
            "failure": "holdout_admission_denied",
            "reason": "test negative",
            "preholdout_result": candidate_preholdout,
            "returncode": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_tail": "",
        }
        candidate_result["child_exact_digest"] = stable_payload_digest(
            candidate_result
        )
        children = [
            {
                "candidate": name,
                "status": "completed",
                "source_state": source_state,
            }
            for name in RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[:3]
        ]
        children.append(candidate_result)
        baseline_confirmation = {
            "candidate": RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
            "status": "completed",
            "source_state": source_state,
        }
        capability_commitment = "e" * 64
        with tempfile.TemporaryDirectory() as raw:
            development_root = Path(raw) / "development-screens"
            development_root.mkdir(parents=True)
            report_path = (
                development_root
                / "recurrent-kernel-development-screen.json"
            )
            source_marker_path = (
                development_root
                / "recurrent-kernel-development-screen.source-retired.json"
            )
            source_marker = accuracy.source_retirement_marker(
                source_sha="b" * 40,
                source_module_digest=stable_payload_digest(
                    screen._source_bound_path_sha256()
                ),
                method_document_digest=str(
                    source_state["accuracy_method_document_sha256"]
                ),
                preregistration_digest=str(
                    preregistration["exact_digest"]
                ),
                report_path=str(report_path),
                parent_process_id=123,
                parent_process_start_identity="synthetic-parent-start",
                capability_commitment_sha256=capability_commitment,
            )
            source_marker_bytes = (
                json.dumps(
                    source_marker,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            source_marker_path.write_bytes(source_marker_bytes)
            report: dict[str, object] = {
                "schema_version": (
                    screen.RECURRENT_KERNEL_DEVELOPMENT_SCREEN_SCHEMA_VERSION
                ),
                "status": "completed",
                "development_only": True,
                "launch_authorized": False,
                "authority_evidence_eligible": False,
                "scientific_result": False,
                "source": source_state,
                "contract": {
                    "screen": _development_screen_contract(),
                    "accuracy_preregistration": preregistration,
                    "preholdout_evidence": preholdout,
                    "source_retirement_marker": source_marker,
                    "source_retirement_marker_path": str(source_marker_path),
                    "source_retirement_marker_sha256": hashlib.sha256(
                        source_marker_bytes
                    ).hexdigest(),
                    "holdout_retirement_marker": None,
                    "holdout_retirement_marker_path": None,
                    "holdout_retirement_marker_sha256": None,
                    "capability_commitment_sha256": capability_commitment,
                },
                "runtime": {
                    "common_child_runtime": None,
                    "child_count": 5,
                    "isolated_children": True,
                    "candidate_interactive_holdout_admission": True,
                },
                "children": children,
                "baseline_confirmation": baseline_confirmation,
                "corpus": preregistration["corpus_contract"],
                "accuracy": {
                    "public_forward": None,
                    "holdout_forward": None,
                },
                "gradients": None,
                "holdout_receipt": None,
                "selection": selection,
                "lifecycle": {
                    "qualification_run": False,
                    "training_run": False,
                    "training_artifact_created": False,
                    "runtime_artifact_created": False,
                    "runtime_action_selection_changed": False,
                    "promotion_authorized": False,
                    "phase_a_output_created": False,
                    "source_attempt_retired": True,
                    "holdout_attempt_retired": False,
                },
            }
            report["exact_digest"] = stable_payload_digest(report)
            with (
                patch.object(
                    accuracy,
                    "validate_accuracy_preregistration",
                ),
                patch.object(
                    screen,
                    "_accuracy_preregistration_for_source",
                    return_value=preregistration,
                ),
                patch.object(
                    screen,
                    "_preholdout_admission_bundle",
                    return_value=(
                        preholdout,
                        False,
                        ["test negative"],
                    ),
                ),
                patch.object(
                    screen,
                    "select_development_candidate",
                    return_value=selection,
                ),
                patch.object(
                    screen,
                    "_validate_candidate_identity",
                    side_effect=lambda child, **_kwargs: str(
                        child["candidate"]
                    ),
                ),
            ):
                validate_development_screen_report(report)
                admitted_preholdout = copy.deepcopy(preholdout)
                admitted_preholdout["admission_authorized"] = True
                admitted_preholdout["reasons"] = []
                admitted_preholdout["exact_digest"] = stable_payload_digest(
                    {
                        key: value
                        for key, value in admitted_preholdout.items()
                        if key != "exact_digest"
                    }
                )
                forged_admitted_failure = copy.deepcopy(report)
                forged_admitted_failure["contract"][
                    "preholdout_evidence"
                ] = admitted_preholdout
                forged_admitted_failure["exact_digest"] = (
                    stable_payload_digest(
                        {
                            key: value
                            for key, value in forged_admitted_failure.items()
                            if key != "exact_digest"
                        }
                    )
                )
                with (
                    patch.object(
                        screen,
                        "_preholdout_admission_bundle",
                        return_value=(
                            admitted_preholdout,
                            True,
                            [],
                        ),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "completed candidate terminal state",
                    ),
                ):
                    validate_development_screen_report(
                        forged_admitted_failure
                    )
                drifted = copy.deepcopy(report)
                integer_flag_marker = copy.deepcopy(source_marker)
                integer_flag_marker["source_retired"] = 1
                integer_flag_marker_bytes = (
                    json.dumps(
                        integer_flag_marker,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                source_marker_path.write_bytes(integer_flag_marker_bytes)
                drifted["contract"]["source_retirement_marker"] = (
                    integer_flag_marker
                )
                drifted["contract"][
                    "source_retirement_marker_sha256"
                ] = hashlib.sha256(integer_flag_marker_bytes).hexdigest()
                drifted["exact_digest"] = stable_payload_digest(
                    {
                        key: value
                        for key, value in drifted.items()
                        if key != "exact_digest"
                    }
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "process or report binding",
                ):
                    validate_development_screen_report(drifted)
                source_marker_path.write_bytes(source_marker_bytes)
                drifted = copy.deepcopy(report)
                drifted["contract"]["screen"]["d04_numeric_contract"][
                    "maximum_tolerance_ratio"
                ] = 0.9
                with self.assertRaisesRegex(
                    ValueError,
                    "declared contract",
                ):
                    validate_development_screen_report(drifted)
                drifted = copy.deepcopy(report)
                drifted["exact_digest"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "digest"):
                    validate_development_screen_report(drifted)
                drifted = copy.deepcopy(report)
                drifted["source"]["module_byte_sha256"][
                    screen._SOURCE_BOUND_MODULE_PATHS["observations"]
                ] = "0" * 64
                drifted["exact_digest"] = stable_payload_digest(
                    {
                        key: value
                        for key, value in drifted.items()
                        if key != "exact_digest"
                    }
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "exact-source binding",
                ):
                    validate_development_screen_report(drifted)
                source_marker_path.write_bytes(b"{}\n")
                with self.assertRaisesRegex(
                    ValueError,
                    "source retirement marker",
                ):
                    validate_development_screen_report(report)

    def test_cli_is_explicitly_development_only(self) -> None:
        self.assertEqual(
            cli._child_command(
                "per_row_bmm_fast_adapter_control_v3",
                "a" * 40,
                "b" * 64,
                (
                    "/tmp/development-screens/"
                    "recurrent-kernel-development-screen.json"
                ),
                "c" * 64,
                123,
                "synthetic-parent-start",
            ),
            (
                cli.sys.executable,
                "-m",
                "evolution_sim.cli.recurrent_kernel_development_screen",
                "--development-run",
                "--expected-source-sha",
                "a" * 40,
                "--child-candidate",
                "per_row_bmm_fast_adapter_control_v3",
                "--child-nonce",
                "b" * 64,
                "--report",
                (
                    "/tmp/development-screens/"
                    "recurrent-kernel-development-screen.json"
                ),
                "--accuracy-preregistration-digest",
                "c" * 64,
                "--parent-process-id",
                "123",
                "--parent-process-start-identity",
                "synthetic-parent-start",
            ),
        )
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
                    "per_row_bmm_fast_adapter_control_v3",
                ]
            )
        closed_report = {
            "selection": {"selected_candidate": None},
            "development_only": True,
            "launch_authorized": False,
        }
        with (
            patch.object(cli, "validate_development_screen_report_path"),
            patch.object(
                cli,
                "run_development_screen",
                return_value=closed_report,
            ) as run_screen,
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
        run_screen.assert_called_once()
        run_kwargs = run_screen.call_args.kwargs
        self.assertEqual(run_kwargs["expected_source_sha"], "a" * 40)
        self.assertEqual(
            run_kwargs["report_path"],
            Path(
                "output/open-ecology/development-screens/"
                "recurrent-kernel-development-screen.json"
            ),
        )
        self.assertIs(run_kwargs["child_command_factory"], cli._child_command)
        self.assertTrue(callable(run_kwargs["progress"]))
        write_report.assert_called_once()

    def test_successful_child_with_stderr_fails_closed(self) -> None:
        completed = screen.subprocess.CompletedProcess(
            args=("screen-child",),
            returncode=0,
            stdout="{}",
            stderr="unexpected warning\n",
        )
        with patch.object(screen.subprocess, "run", return_value=completed):
            observed = screen._run_child_candidate(
                candidate=RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
                expected_source_sha="a" * 40,
                report_path=(
                    "/tmp/development-screens/"
                    "recurrent-kernel-development-screen.json"
                ),
                accuracy_preregistration_digest="c" * 64,
                parent_process_id=123,
                parent_process_start_identity="synthetic-parent-start",
                child_command_factory=lambda *_args: ("screen-child",),
            )
        self.assertEqual(observed["status"], "failed")
        self.assertEqual(observed["failure"], "child_nonempty_stderr")
        self.assertEqual(observed["returncode"], 0)
        self.assertEqual(observed["stderr_tail"], "unexpected warning\n")
        screen._validate_failed_child_envelope(
            observed,
            expected_candidate=(
                RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0]
            ),
            allow_preholdout_abort=False,
        )


if __name__ == "__main__":
    unittest.main()
