from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
import secrets
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.mind import recurrent_accuracy_screen as accuracy
    from evolution_sim.mind import recurrent_actor_critic
    from evolution_sim.mind.recurrent_kernel_development_screen import (
        recurrent_kernel_candidate,
    )

    RecurrentKernelDevelopmentScreenError = (
        accuracy.RecurrentKernelDevelopmentScreenError
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentAccuracyScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        # Deliberately synthetic and unrelated to the sealed source contract.
        self.seeds = accuracy.AccuracySeedContract(
            ordinary=101,
            holdout=202,
            cancellation_underflow=303,
            gradient=404,
            synthetic_test_contract=True,
        )
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

    def test_independent_oracle_reachable_source_forbids_kernels(self) -> None:
        module_tree = ast.parse(inspect.getsource(accuracy))
        function_nodes = {
            node.name: node
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        root_name = "_independent_float64_linear_oracle"
        expected_reachable = {
            root_name,
            "_float32_tensor_exact_digest",
            "_independent_oracle_provenance",
            "_independent_oracle_raw_values",
            "_validated_oracle_inputs",
            "_validated_oracle_tensor",
        }

        def call_name(node: ast.Call) -> str | None:
            parts: list[str] = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name):
                return None
            parts.append(current.id)
            return ".".join(reversed(parts))

        reachable: set[str] = set()
        pending = [root_name]
        calls_by_function: dict[str, tuple[str, ...]] = {}
        while pending:
            function_name = pending.pop()
            if function_name in reachable:
                continue
            function_node = function_nodes[function_name]
            reachable.add(function_name)
            calls = tuple(
                target
                for node in ast.walk(function_node)
                if isinstance(node, ast.Call)
                and (target := call_name(node)) is not None
            )
            calls_by_function[function_name] = calls
            pending.extend(
                target
                for target in calls
                if target in function_nodes and target not in reachable
            )

        self.assertEqual(reachable, expected_reachable)
        forbidden_exact = {
            "F.linear",
            "torch.addbmm",
            "torch.addmm",
            "torch.baddbmm",
            "torch.bmm",
            "torch.einsum",
            "torch.matmul",
            "torch.mm",
            "torch.nn.functional.linear",
        }
        forbidden_calls = {
            f"{function_name}:{target}"
            for function_name, calls in calls_by_function.items()
            for target in calls
            if target in forbidden_exact
            or target.startswith("torch.ops.aten.")
            or target.startswith("torch._C._nn.")
            or "candidate" in target.lower()
        }
        self.assertEqual(forbidden_calls, set())

    def test_linux_proc_start_parser_handles_parentheses_and_spaces(self) -> None:
        suffix_fields = [
            "S",
            *(str(index) for index in range(1, 19)),
            "987654321",
            "ignored-after-start",
        ]
        payload = (
            "4242 (worker name with ) embedded parenthesis) "
            + " ".join(suffix_fields)
        )
        self.assertEqual(
            accuracy._linux_proc_stat_start_ticks(payload),
            "987654321",
        )
        with self.assertRaises(RecurrentKernelDevelopmentScreenError):
            accuracy._linux_proc_stat_start_ticks(
                "4242 (unterminated worker S 1 2"
            )

    def _synthetic_source_tree(
        self,
    ) -> tuple[Path, Path, dict[str, str], str]:
        root = (
            Path(self.temporary_directory.name)
            / f"attempt-{secrets.token_hex(4)}"
        )
        checkout = root / "checkout"
        source_digests: dict[str, str] = {}
        for relative_path in (
            accuracy.RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS
        ):
            source = checkout / relative_path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"synthetic:{relative_path}\n".encode())
            source_digests[relative_path] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        method = checkout / accuracy.RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH
        method.parent.mkdir(parents=True, exist_ok=True)
        method.write_bytes(
            (
                "The machine-readable canonical selection manifest is\n"
                f"`{accuracy.RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION}`."
                "\nIts canonical `stable_payload_digest` is\n"
                f"`{accuracy.RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST}`."
                "\n"
            ).encode()
        )
        return (
            root,
            checkout,
            source_digests,
            hashlib.sha256(method.read_bytes()).hexdigest(),
        )

    def _holdout_authority(
        self,
        *,
        seeds: accuracy.AccuracySeedContract | None = None,
        candidate_identity: str | None = None,
    ) -> tuple[accuracy.HoldoutSeedAuthority, accuracy.HoldoutCapability]:
        root, checkout, source_digests, method_digest = (
            self._synthetic_source_tree()
        )
        source_module_digest = accuracy.stable_payload_digest(source_digests)
        child_nonce = secrets.token_hex(32)
        child_start = accuracy.process_start_identity(os.getpid())
        parent_start = accuracy.process_start_identity(os.getppid())
        report_path = root / "recurrent-kernel-development-screen.json"
        preimage = accuracy.generate_holdout_capability_preimage()
        commitment = accuracy.holdout_capability_commitment(preimage)
        source_marker = accuracy.source_retirement_marker(
            source_sha="a" * 40,
            source_module_digest=source_module_digest,
            method_document_digest=method_digest,
            preregistration_digest="c" * 64,
            report_path=str(report_path),
            parent_process_id=max(1, os.getppid()),
            parent_process_start_identity=parent_start,
            capability_commitment_sha256=commitment,
        )
        source_marker_path = root / (
            accuracy.RECURRENT_ACCURACY_SOURCE_RETIREMENT_MARKER_NAME
        )
        source_marker_bytes = json.dumps(
            source_marker,
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n"
        source_marker_path.write_bytes(source_marker_bytes)
        source_marker_sha256 = hashlib.sha256(
            source_marker_bytes
        ).hexdigest()
        holdout_marker = accuracy.holdout_retirement_marker(
            source_retirement_marker_sha256=source_marker_sha256,
            source_retirement_marker_exact_digest=source_marker[
                "exact_digest"
            ],
            preholdout_evidence_digest="d" * 64,
            closing_baseline_evidence_digest="9" * 64,
            parent_process_id=max(1, os.getppid()),
            parent_process_start_identity=parent_start,
            child_process_id=os.getpid(),
            child_process_start_identity=child_start,
            child_nonce=child_nonce,
            capability_commitment_sha256=commitment,
        )
        holdout_marker_path = root / (
            accuracy.RECURRENT_ACCURACY_HOLDOUT_RETIREMENT_MARKER_NAME
        )
        holdout_marker_bytes = json.dumps(
            holdout_marker,
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n"
        holdout_marker_path.write_bytes(holdout_marker_bytes)
        holdout_marker_sha256 = hashlib.sha256(
            holdout_marker_bytes
        ).hexdigest()
        capability = accuracy.issue_holdout_capability(
            preimage=preimage,
            capability_commitment_sha256=commitment,
            preholdout_evidence_digest="d" * 64,
            source_retirement_marker_sha256=source_marker_sha256,
            holdout_retirement_marker_sha256=holdout_marker_sha256,
            candidate_identity=(
                accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
            ),
            closing_baseline_validated=True,
            source_retirement_marker_exists=True,
            holdout_retirement_marker_exists=True,
        )
        authority = accuracy.HoldoutSeedAuthority(
            seeds=self.seeds if seeds is None else seeds,
            capability=capability,
            source_sha="a" * 40,
            source_clean_detached=True,
            checkout_root=str(checkout),
            required_source_paths=tuple(source_digests),
            source_module_digests=source_digests,
            method_document_path=(
                accuracy.RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH
            ),
            method_document_digest=method_digest,
            source_retirement_marker_path=str(source_marker_path),
            source_retirement_marker_sha256=source_marker_sha256,
            holdout_retirement_marker_path=str(holdout_marker_path),
            holdout_retirement_marker_sha256=holdout_marker_sha256,
            capability_commitment_sha256=commitment,
            parent_process_id=max(1, os.getppid()),
            parent_process_start_identity=parent_start,
            child_process_id=os.getpid(),
            child_process_start_identity=child_start,
            child_nonce=child_nonce,
            report_path=str(report_path),
            preregistration_digest="c" * 64,
            candidate_identity=(
                accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
                if candidate_identity is None
                else candidate_identity
            ),
            preholdout_evidence_digest="d" * 64,
            closing_baseline_evidence_digest="9" * 64,
        )
        return authority, capability

    def _evaluate_synthetic_holdout(
        self,
        authority: accuracy.HoldoutSeedAuthority,
        capability: accuracy.HoldoutCapability,
    ) -> dict[str, object]:
        with recurrent_kernel_candidate(
            accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
        ):
            return accuracy.evaluate_holdout_forward_corpus(
                recurrent_actor_critic._backend_stable_linear,
                authority=authority,
                capability=capability,
            )

    def test_factorized_tables_have_frozen_counts_and_indices(self) -> None:
        operational = accuracy.operational_probe_cases(self.seeds.ordinary)
        self.assertEqual(len(operational), 23)
        self.assertEqual(sum(case.dot_count for case in operational), 425)
        self.assertEqual(
            sum(case.product_count for case in operational),
            109_568,
        )
        self.assertEqual(
            [case.active_rows for case in operational[7:20]],
            [1, 2, 3, 5, 9, 17, 33, 64, 65, 129, 257, 319, 320],
        )
        self.assertEqual(
            [case.physical_rows for case in operational[7:20]],
            [1, 2, 4, 8, 16, 32, 64, 64, 128, 256, 320, 320, 320],
        )
        for case in operational:
            self.assertEqual(
                tuple(sorted(set(case.selected_rows))),
                case.selected_rows,
            )
            self.assertEqual(
                tuple(sorted(set(case.selected_outputs))),
                case.selected_outputs,
            )

        cancellation = accuracy.cancellation_probe_cases(
            self.seeds.cancellation_underflow
        )
        underflow = accuracy.underflow_probe_cases(
            self.seeds.cancellation_underflow,
            family="underflow_u1",
        )
        self.assertEqual(sum(case.dot_count for case in cancellation), 116)
        self.assertEqual(
            sum(case.product_count for case in cancellation),
            27_584,
        )
        self.assertEqual(sum(case.dot_count for case in underflow), 29)
        self.assertEqual(
            sum(case.product_count for case in underflow),
            6_896,
        )

    def test_sha256_selection_is_deterministic_and_seed_bound(self) -> None:
        first = accuracy.operational_probe_cases(self.seeds.ordinary)
        second = accuracy.operational_probe_cases(self.seeds.ordinary)
        changed = accuracy.operational_probe_cases(self.seeds.ordinary + 1)
        self.assertEqual(first, second)
        self.assertNotEqual(
            [case.selected_rows for case in first],
            [case.selected_rows for case in changed],
        )
        gru = first[1]
        self.assertEqual(gru.selected_outputs, (0, 255, 256, 511, 512, 767))
        encoder = first[0]
        self.assertIn(0, encoder.selected_rows)
        self.assertIn(encoder.active_rows - 1, encoder.selected_rows)
        self.assertIn(0, encoder.selected_outputs)
        self.assertIn(255, encoder.selected_outputs)

    def test_synthetic_rng_and_selection_golden_vectors(self) -> None:
        self.assertEqual(
            accuracy._splitmix64(0),
            0xE220A8397B1DCDAF,
        )
        self.assertEqual(
            accuracy._splitmix64((1 << 64) - 1),
            0xE4D971771B652C20,
        )
        self.assertEqual(
            accuracy._sha256_rank(
                seed=17,
                case_id="synthetic.case",
                axis="row",
                index=3,
            ).hex(),
            "6eb86be269677f65e40ed81cefc2179e39be152a87a3ffb96832600c6420624b",
        )
        self.assertEqual(
            accuracy._selected_indices(
                size=11,
                count=4,
                seed=17,
                case_id="synthetic.case",
                axis="row",
                pinned=(0, 10),
            ),
            (0, 5, 7, 10),
        )
        self.assertEqual(
            tuple(
                accuracy._value_word(
                    seed=33,
                    family="balanced",
                    case_id="synthetic.case",
                    tensor_name="input",
                    flat_index=5,
                    draw=draw,
                )
                for draw in range(4)
            ),
            (
                0xD7CA4ED341AF5B95,
                0xC42CD171B491C034,
                0x046C37CE99AED167,
                0xD68EE87806905B19,
            ),
        )
        value = accuracy._normal_value(
            seed=33,
            case_id="synthetic.case",
            tensor="input",
            index=(0, 5),
            flat_index=5,
            family="balanced",
        )
        self.assertEqual(struct.pack(">f", value).hex(), "3e658000")

    def test_contract_reconstructs_and_detects_tampering(self) -> None:
        contract = accuracy.accuracy_corpus_contract(seeds=self.seeds)
        self.assertIs(
            accuracy.validate_accuracy_corpus_contract(
                contract,
                seeds=self.seeds,
            ),
            contract,
        )
        counts = contract["counts"]
        self.assertEqual(counts["forward_dot_count"], 1_874)
        self.assertEqual(counts["forward_product_count"], 479_648)
        self.assertEqual(
            counts["gradient_and_hvp_component_count"],
            1_144_066,
        )
        tampered = copy.deepcopy(contract)
        tampered["counts"]["forward_dot_count"] = 1
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "reconstruction failed",
        ):
            accuracy.validate_accuracy_corpus_contract(
                tampered,
                seeds=self.seeds,
            )
    def test_normal_generator_is_dyadic_finite_and_same_sign_per_dot(self) -> None:
        probe = accuracy.operational_probe_cases(self.seeds.ordinary)[0]
        for family in ("balanced", "alternating", "mixed"):
            generated = accuracy.generate_normal_case(
                probe,
                seed=self.seeds.ordinary,
                family=family,
            )
            self.assertTrue(torch.isfinite(generated.inputs).all())
            selected_inputs, selected_weight, selected_bias = (
                accuracy._selected_components(generated)
            )
            for row in selected_inputs:
                for output_index, weight in enumerate(selected_weight):
                    terms = [
                        float(row[index]) * float(weight[index])
                        for index in range(probe.input_features)
                    ]
                    terms.append(float(selected_bias[output_index]))
                    signs = {term > 0.0 for term in terms if term != 0.0}
                    if family == "alternating":
                        self.assertEqual(len(signs), 1)
                    self.assertTrue(
                        all(
                            abs(term) >= 2.0**-126
                            for term in terms
                            if term != 0.0
                        )
                    )

    def test_cancellation_and_underflow_classification(self) -> None:
        cancellation_probe = accuracy.cancellation_probe_cases(
            self.seeds.cancellation_underflow
        )[-2]
        cancellation = accuracy.generate_cancellation_case(
            cancellation_probe,
            seed=self.seeds.cancellation_underflow,
        )
        candidate = torch.nn.functional.linear(
            cancellation.inputs,
            cancellation.weight,
            cancellation.bias,
        )
        selected = accuracy._selected_candidate_output(
            candidate,
            cancellation_probe,
        )
        components = accuracy._selected_components(cancellation)
        classification = accuracy._per_dot_classification(
            probe=cancellation_probe,
            inputs=components[0],
            weight=components[1],
            bias=components[2],
            selected_candidate_output=selected,
        )
        self.assertEqual(
            classification["standard_bound_applicable_count"],
            0,
        )
        self.assertEqual(
            classification["comparison_count"],
            cancellation_probe.dot_count,
        )
        self.assertTrue(
            all(
                record["condition_ratio_kind"] == "exact_zero"
                or math.isfinite(record["condition_ratio"])
                for record in classification["records"]
            )
        )

        for family in ("underflow_u1", "underflow_u2"):
            underflow_probe = accuracy.underflow_probe_cases(
                self.seeds.cancellation_underflow,
                family=family,
            )[-1]
            generated = accuracy.generate_underflow_case(
                underflow_probe,
                seed=self.seeds.cancellation_underflow,
                family=family,
            )
            output = torch.nn.functional.linear(
                generated.inputs,
                generated.weight,
                generated.bias,
            )
            inputs, weight, bias = accuracy._selected_components(generated)
            classification = accuracy._per_dot_classification(
                probe=underflow_probe,
                inputs=inputs,
                weight=weight,
                bias=bias,
                selected_candidate_output=accuracy._selected_candidate_output(
                    output, underflow_probe
                ),
            )
            self.assertEqual(
                classification["standard_bound_applicable_count"],
                0,
            )
            self.assertEqual(
                classification["candidate_zero_result_count"]
                + classification["candidate_nonzero_result_count"],
                underflow_probe.dot_count,
            )
            self.assertEqual(
                classification["oracle_zero_result_count"]
                + classification["oracle_nonzero_result_count"],
                underflow_probe.dot_count,
            )

    def test_forward_aggregate_enforces_term_classification_gates(self) -> None:
        def case(
            family: str,
            *,
            dot_count: int,
            subnormal_term_dot_count: int,
            cancellation: bool = False,
        ) -> dict[str, object]:
            return {
                "case": {
                    "dot_count": dot_count,
                    "product_count": dot_count,
                },
                "family": family,
                "metrics": {},
                "per_dot_classification": {
                    "comparison_count": dot_count,
                    "standard_bound_applicable_count": (
                        0 if subnormal_term_dot_count else dot_count
                    ),
                    "mixed_sign_dot_count": int(cancellation),
                    "subnormal_term_dot_count": subnormal_term_dot_count,
                    "d04_violation_count": 0,
                    "applicable_standard_bound_violation_count": 0,
                    "condition_ratio_gate_passed": True,
                    "candidate_zero_result_count": dot_count,
                    "candidate_nonzero_result_count": 0,
                    "oracle_zero_result_count": dot_count,
                    "oracle_nonzero_result_count": 0,
                },
            }

        ordinary = {
            "ordinary": [
                case(
                    "synthetic_normal",
                    dot_count=1,
                    subnormal_term_dot_count=0,
                )
            ]
        }
        self.assertTrue(
            accuracy._aggregate_forward_evidence(ordinary)[
                "forward_gate_passed"
            ]
        )
        ordinary_drift = copy.deepcopy(ordinary)
        ordinary_drift["ordinary"][0]["per_dot_classification"][  # type: ignore[index]
            "subnormal_term_dot_count"
        ] = 1
        normal_aggregate = accuracy._aggregate_forward_evidence(
            ordinary_drift
        )
        self.assertEqual(
            normal_aggregate["normal_subnormal_term_dot_count"],
            1,
        )
        self.assertFalse(normal_aggregate["forward_gate_passed"])

        cancellation = {
            "cancellation": [
                case(
                    "cancellation",
                    dot_count=1,
                    subnormal_term_dot_count=1,
                    cancellation=True,
                )
            ]
        }
        cancellation_aggregate = accuracy._aggregate_forward_evidence(
            cancellation
        )
        self.assertEqual(
            cancellation_aggregate[
                "cancellation_subnormal_term_dot_count"
            ],
            1,
        )
        self.assertFalse(cancellation_aggregate["forward_gate_passed"])

        underflow = {
            family: [
                case(
                    family,
                    dot_count=29,
                    subnormal_term_dot_count=29,
                )
            ]
            for family in ("underflow_u1", "underflow_u2")
        }
        underflow_aggregate = accuracy._aggregate_forward_evidence(underflow)
        self.assertEqual(
            underflow_aggregate[
                "underflow_subnormal_term_dot_count_by_family"
            ],
            {"underflow_u1": 29, "underflow_u2": 29},
        )
        self.assertTrue(
            underflow_aggregate["underflow_subnormal_coverage_passed"]
        )
        self.assertEqual(
            underflow_aggregate["underflow_result_counts_by_family"],
            {
                "underflow_u1": {
                    "candidate_zero_result_count": 29,
                    "candidate_nonzero_result_count": 0,
                    "oracle_zero_result_count": 29,
                    "oracle_nonzero_result_count": 0,
                },
                "underflow_u2": {
                    "candidate_zero_result_count": 29,
                    "candidate_nonzero_result_count": 0,
                    "oracle_zero_result_count": 29,
                    "oracle_nonzero_result_count": 0,
                },
            },
        )
        self.assertTrue(
            underflow_aggregate["underflow_result_count_coverage_passed"]
        )
        self.assertTrue(underflow_aggregate["forward_gate_passed"])
        underflow_drift = copy.deepcopy(underflow)
        underflow_drift["underflow_u2"][0][  # type: ignore[index]
            "per_dot_classification"
        ]["subnormal_term_dot_count"] = 28  # type: ignore[index]
        drift_aggregate = accuracy._aggregate_forward_evidence(
            underflow_drift
        )
        self.assertFalse(
            drift_aggregate["underflow_subnormal_coverage_passed"]
        )
        self.assertFalse(drift_aggregate["forward_gate_passed"])
        result_drift = copy.deepcopy(underflow)
        result_drift["underflow_u2"][0][  # type: ignore[index]
            "per_dot_classification"
        ]["candidate_zero_result_count"] = 28  # type: ignore[index]
        result_aggregate = accuracy._aggregate_forward_evidence(result_drift)
        self.assertFalse(
            result_aggregate["underflow_result_count_coverage_passed"]
        )
        self.assertFalse(result_aggregate["forward_gate_passed"])

    def test_exact_zero_condition_is_json_safe_and_inapplicable(self) -> None:
        probe = accuracy.LinearProbeCase(
            case_id="synthetic.exact_zero",
            table="synthetic",
            role="synthetic",
            input_features=2,
            output_features=1,
            active_rows=1,
            physical_rows=1,
            selected_rows=(0,),
            selected_outputs=(0,),
        )
        inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        weight = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        bias = torch.zeros(1, dtype=torch.float32)
        output = torch.nn.functional.linear(inputs, weight, bias)
        classification = accuracy._per_dot_classification(
            probe=probe,
            inputs=inputs,
            weight=weight,
            bias=bias,
            selected_candidate_output=output,
        )
        record = classification["records"][0]
        self.assertEqual(record["condition_ratio_kind"], "exact_zero")
        self.assertIsNone(record["condition_ratio"])
        self.assertFalse(
            record["standard_forward_error_bound_applicable"]
        )
        self.assertTrue(classification["condition_ratio_gate_passed"])
        json.dumps(classification, allow_nan=False)

    def test_gradient_cases_bind_full_component_and_loss_counts(self) -> None:
        cases = accuracy.gradient_probe_cases(self.seeds.gradient)
        self.assertEqual(sum(case.dot_count for case in cases), 116)
        component_count = sum(
            case.physical_rows * case.input_features
            + case.output_features * case.input_features
            + case.output_features
            for case in cases
        )
        self.assertEqual(component_count, 572_033)

    def test_gradient_and_hvp_probe_passes_fixed_tile_candidate(self) -> None:
        with recurrent_kernel_candidate(
            accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
        ):
            report = accuracy.evaluate_gradient_probe(
                recurrent_actor_critic._backend_stable_linear,
                seeds=self.seeds,
            )
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["loss_term_count"], 11_625)
        self.assertEqual(
            aggregate["diagnostic_spot_selection_count"],
            116,
        )
        self.assertEqual(aggregate["gradient_component_count"], 572_033)
        self.assertEqual(aggregate["hvp_component_count"], 572_033)
        self.assertEqual(
            aggregate["gradient_and_hvp_component_count"],
            1_144_066,
        )
        self.assertTrue(
            aggregate["gradient_active_reference_coverage_passed"]
        )
        self.assertTrue(aggregate["hvp_active_reference_coverage_passed"])
        self.assertGreaterEqual(
            aggregate[
                "gradient_active_reference_nonzero_component_share"
            ],
            0.95,
        )
        self.assertGreaterEqual(
            aggregate["hvp_active_reference_nonzero_component_share"],
            0.95,
        )
        self.assertTrue(aggregate["gradient_gate_passed"])

    def test_gradient_probe_rejects_nonproduction_callable(self) -> None:
        with recurrent_kernel_candidate(
            accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
        ):
            with self.assertRaisesRegex(
                RecurrentKernelDevelopmentScreenError,
                "production backend-stable callable",
            ):
                accuracy.evaluate_gradient_probe(
                    recurrent_actor_critic._fixed_row_tile_4_linear_forward,
                    seeds=self.seeds,
                )

    def test_holdout_capability_is_process_bound_and_exactly_once(self) -> None:
        authority, capability = self._holdout_authority()
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "production backend-stable callable",
        ):
            accuracy.evaluate_holdout_forward_corpus(
                recurrent_actor_critic._fixed_row_tile_4_linear_forward,
                authority=authority,
                capability=capability,
            )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "not in fixed-row context",
        ):
            accuracy.evaluate_holdout_forward_corpus(
                recurrent_actor_critic._backend_stable_linear,
                authority=authority,
                capability=capability,
            )
        report = self._evaluate_synthetic_holdout(authority, capability)
        receipt = report["holdout_receipt"]
        self.assertEqual(receipt["access_ordinal"], 1)
        self.assertTrue(receipt["test_mode"])
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "exactly once",
        ):
            self._evaluate_synthetic_holdout(authority, capability)

    def test_holdout_capability_rejects_wrong_preimage_and_early_issue(
        self,
    ) -> None:
        preimage = "01" * 32
        commitment = accuracy.holdout_capability_commitment(preimage)
        common = {
            "preimage": preimage,
            "capability_commitment_sha256": commitment,
            "preholdout_evidence_digest": "2" * 64,
            "source_retirement_marker_sha256": "3" * 64,
            "holdout_retirement_marker_sha256": "4" * 64,
            "candidate_identity": (
                accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
            ),
            "closing_baseline_validated": True,
            "source_retirement_marker_exists": True,
            "holdout_retirement_marker_exists": True,
        }
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "preimage does not match",
        ):
            accuracy.issue_holdout_capability(
                **{
                    **common,
                    "capability_commitment_sha256": (
                        accuracy.holdout_capability_commitment("02" * 32)
                    ),
                }
            )
        for missing in (
            "source_retirement_marker_exists",
            "holdout_retirement_marker_exists",
        ):
            with self.subTest(missing=missing), self.assertRaisesRegex(
                RecurrentKernelDevelopmentScreenError,
                "both durable retirement markers",
            ):
                accuracy.issue_holdout_capability(
                    **{**common, missing: False}
                )

    def test_holdout_authority_detects_marker_and_source_drift(self) -> None:
        authority, capability = self._holdout_authority()
        authority._holdout_retirement_marker_path.write_bytes(b"{}\n")
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "marker file digest is invalid",
        ):
            self._evaluate_synthetic_holdout(authority, capability)

        authority, capability = self._holdout_authority()
        relative = accuracy.RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS[0]
        (authority._checkout_root / relative).write_bytes(b"drift\n")
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "source-module byte digest drifted",
        ):
            self._evaluate_synthetic_holdout(authority, capability)

    def test_retirement_marker_rejects_integer_boolean_alias(self) -> None:
        root = Path(self.temporary_directory.name)
        marker_path = root / "retirement.json"
        expected = {"source_retired": True, "exact_digest": "a" * 64}
        aliased = {"source_retired": 1, "exact_digest": "a" * 64}
        raw = json.dumps(
            aliased,
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n"
        marker_path.write_bytes(raw)
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "reconstruction failed",
        ):
            accuracy._validate_retirement_marker_file(
                path=marker_path,
                file_sha256=hashlib.sha256(raw).hexdigest(),
                expected_marker=expected,
                label="synthetic",
            )

    def test_composite_generation_has_no_public_bypass(self) -> None:
        probe = accuracy.operational_probe_cases(self.seeds.holdout)[0]
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "consumed private permit",
        ):
            accuracy.generate_normal_case(
                probe,
                seed=self.seeds.holdout,
                family="composite",
            )

    def test_deciding_validator_rejects_synthetic_holdout(self) -> None:
        authority, capability = self._holdout_authority()
        report = self._evaluate_synthetic_holdout(authority, capability)
        expected_cases = [
            case.as_contract()
            for case in accuracy.operational_probe_cases(self.seeds.holdout)
        ]
        self.assertIs(
            accuracy.validate_holdout_report_offline(
                report,
                expected_seed_contract=self.seeds,
                expected_case_contracts=expected_cases,
                allow_synthetic_test=True,
            ),
            report,
        )

        tampered_seed_digest = copy.deepcopy(report)
        seed_receipt = tampered_seed_digest["holdout_receipt"]
        seed_receipt["seed_digest"] = "0" * 64
        seed_receipt["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in seed_receipt.items()
                if key != "exact_digest"
            }
        )
        tampered_seed_digest["exact_digest"] = (
            accuracy.stable_payload_digest(
                {
                    key: value
                    for key, value in tampered_seed_digest.items()
                    if key != "exact_digest"
                }
            )
        )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "seed digest differs",
        ):
            accuracy.validate_holdout_report_offline(
                tampered_seed_digest,
                expected_seed_contract=self.seeds,
                expected_case_contracts=expected_cases,
                allow_synthetic_test=True,
            )

        alternate_corpus = copy.deepcopy(report)
        first_case = alternate_corpus["groups"]["holdout"][0]
        first_probe = accuracy._probe_from_contract(first_case["case"])
        alternate_generated = accuracy._materialize_normal_case(
            first_probe,
            seed=909,
            family="composite",
        )
        with recurrent_kernel_candidate(
            accuracy.RECURRENT_ACCURACY_SCREEN_CANDIDATE
        ):
            alternate_case = accuracy._evaluate_generated_case(
                recurrent_actor_critic._backend_stable_linear,
                alternate_generated,
                serialize_selected_values=True,
            )
        alternate_corpus["groups"]["holdout"][0] = alternate_case
        alternate_corpus["aggregate"] = accuracy._aggregate_forward_evidence(
            {"holdout": alternate_corpus["groups"]["holdout"]}
        )
        alternate_corpus["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in alternate_corpus.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "seed-bound component reconstruction failed",
        ):
            accuracy.validate_holdout_report_offline(
                alternate_corpus,
                expected_seed_contract=self.seeds,
                expected_case_contracts=expected_cases,
                allow_synthetic_test=True,
            )

        tampered_full_digest = copy.deepcopy(report)
        full_case = tampered_full_digest["groups"]["holdout"][0]
        full_case["full_component_exact_digests"][
            "candidate_output"
        ] = "0" * 64
        full_case["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in full_case.items()
                if key != "exact_digest"
            }
        )
        tampered_full_digest["exact_digest"] = (
            accuracy.stable_payload_digest(
                {
                    key: value
                    for key, value in tampered_full_digest.items()
                    if key != "exact_digest"
                }
            )
        )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "full component digest is invalid",
        ):
            accuracy.validate_holdout_report_offline(
                tampered_full_digest,
                expected_seed_contract=self.seeds,
                expected_case_contracts=expected_cases,
                allow_synthetic_test=True,
            )

        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "cannot support a deciding report",
        ):
            accuracy.validate_deciding_holdout_report(
                report,
                expected_seed_contract=self.seeds,
            )

        tampered_family = copy.deepcopy(report)
        family_case = tampered_family["groups"]["holdout"][0]
        family_case["family"] = "balanced"
        family_case["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in family_case.items()
                if key != "exact_digest"
            }
        )
        tampered_family["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in tampered_family.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "family is not composite",
        ):
            accuracy.validate_holdout_report_offline(
                tampered_family,
                expected_seed_contract=self.seeds,
                expected_case_contracts=expected_cases,
                allow_synthetic_test=True,
            )

        tampered_tensor = copy.deepcopy(report)
        tensor_case = tampered_tensor["groups"]["holdout"][0]
        raw_input = tensor_case["serialized_selected_values"]["input"]
        raw_input["unexpected"] = "drift"
        raw_input["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in raw_input.items()
                if key != "exact_digest"
            }
        )
        tensor_case["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in tensor_case.items()
                if key != "exact_digest"
            }
        )
        tampered_tensor["exact_digest"] = accuracy.stable_payload_digest(
            {
                key: value
                for key, value in tampered_tensor.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "tensor schema is malformed",
        ):
            accuracy.validate_holdout_report_offline(
                tampered_tensor,
                expected_seed_contract=self.seeds,
                expected_case_contracts=expected_cases,
                allow_synthetic_test=True,
            )

    def test_serialized_probe_rejects_boolean_integer_alias(self) -> None:
        contract = accuracy.operational_probe_cases(self.seeds.holdout)[0]
        malformed = contract.as_contract()
        malformed["active_rows"] = True
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "integer is malformed",
        ):
            accuracy._probe_from_contract(malformed)

    def test_sealed_holdout_refuses_pytest_environment(self) -> None:
        # Patch the factory itself so this test never reads the sealed literals.
        synthetic_non_test = accuracy.AccuracySeedContract(
            ordinary=501,
            holdout=502,
            cancellation_underflow=503,
            gradient=504,
            synthetic_test_contract=False,
        )
        with patch.object(
            accuracy,
            "sealed_accuracy_seed_contract",
            return_value=synthetic_non_test,
        ):
            authority, capability = self._holdout_authority(
                seeds=synthetic_non_test,
            )
            with patch.dict(
                os.environ,
                {"PYTEST_CURRENT_TEST": "synthetic"},
            ):
                with self.assertRaisesRegex(
                    RecurrentKernelDevelopmentScreenError,
                    "cannot be consumed under a test runner",
                ):
                    self._evaluate_synthetic_holdout(authority, capability)

    def test_preregistration_binds_source_modules_and_reconstructs(self) -> None:
        _, checkout, digests, method_digest = self._synthetic_source_tree()
        report = accuracy.build_accuracy_preregistration(
            source_sha="a" * 40,
            checkout_root=str(checkout),
            source_module_digests=digests,
            required_source_paths=(
                accuracy.RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS
            ),
            method_document_path=(
                accuracy.RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH
            ),
            method_document_sha256=method_digest,
            seeds=self.seeds,
        )
        self.assertFalse(report["holdout_consumed"])
        self.assertFalse(report["candidate_selected"])
        self.assertFalse(report["launch_authorized"])
        self.assertIs(
            accuracy.validate_accuracy_preregistration(
                report,
                seeds=self.seeds,
            ),
            report,
        )
        tampered = copy.deepcopy(report)
        tampered["candidate_count"] = 2
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "reconstruction failed",
        ):
            accuracy.validate_accuracy_preregistration(
                tampered,
                seeds=self.seeds,
            )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "frozen order",
        ):
            accuracy.build_accuracy_preregistration(
                source_sha="a" * 40,
                checkout_root=str(checkout),
                source_module_digests=digests,
                required_source_paths=tuple(reversed(tuple(digests))),
                method_document_path=(
                    accuracy.RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH
                ),
                method_document_sha256=method_digest,
                seeds=self.seeds,
            )
        method = (
            checkout / accuracy.RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH
        )
        method.write_bytes(b"drift\n")
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "byte digest drifted",
        ):
            accuracy.validate_accuracy_preregistration(
                report,
                seeds=self.seeds,
            )

    def test_source_file_digest_is_exact_and_missing_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "module.py"
            source.write_bytes(b"one\n")
            first = accuracy.source_file_sha256(source)
            source.write_bytes(b"two\n")
            second = accuracy.source_file_sha256(source)
            self.assertNotEqual(first, second)
            with self.assertRaisesRegex(
                RecurrentKernelDevelopmentScreenError,
                "not a file",
            ):
                accuracy.source_file_sha256(Path(directory) / "missing.py")

    def test_test_sources_do_not_embed_real_seed_literals(self) -> None:
        def folded_literal_string(node: ast.AST) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left = folded_literal_string(node.left)
                right = folded_literal_string(node.right)
                return None if left is None or right is None else left + right
            if isinstance(node, ast.JoinedStr):
                parts = [
                    folded_literal_string(value)
                    for value in node.values
                ]
                return None if any(part is None for part in parts) else "".join(
                    part for part in parts if part is not None
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"
                and not node.keywords
                and len(node.args) == 1
                and isinstance(node.args[0], (ast.List, ast.Tuple))
            ):
                separator = folded_literal_string(node.func.value)
                parts = [
                    folded_literal_string(element)
                    for element in node.args[0].elts
                ]
                if separator is not None and all(
                    part is not None for part in parts
                ):
                    return separator.join(
                        part for part in parts if part is not None
                    )
            return None

        test_root = Path(__file__).resolve().parent
        for source in test_root.rglob("*.py"):
            raw = source.read_text(encoding="utf-8")
            tree = ast.parse(raw, filename=str(source))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, int)
                    and not isinstance(node.value, bool)
                    and len(str(abs(node.value))) == 19
                ):
                    self.fail(
                        f"{source}:{node.lineno} has a 19-digit integer"
                    )
                folded = folded_literal_string(node)
                if (
                    folded is not None
                    and len(folded) == 19
                    and folded.isdigit()
                ):
                    self.fail(
                        f"{source}:{node.lineno} folds to a 19-digit string"
                    )

    def test_wrong_candidate_identity_and_seed_alias_fail_closed(self) -> None:
        for marker in (1, "true"):
            with self.assertRaisesRegex(
                RecurrentKernelDevelopmentScreenError,
                "marker must be boolean",
            ):
                accuracy.accuracy_corpus_contract(
                    seeds=accuracy.AccuracySeedContract(
                        ordinary=1,
                        holdout=2,
                        cancellation_underflow=3,
                        gradient=4,
                        synthetic_test_contract=marker,  # type: ignore[arg-type]
                    )
                )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "pairwise distinct",
        ):
            accuracy.accuracy_corpus_contract(
                seeds=accuracy.AccuracySeedContract(
                    ordinary=1,
                    holdout=1,
                    cancellation_underflow=2,
                    gradient=3,
                    synthetic_test_contract=True,
                )
            )
        with self.assertRaisesRegex(
            RecurrentKernelDevelopmentScreenError,
            "sole frozen candidate",
        ):
            self._holdout_authority(candidate_identity="another-candidate")
