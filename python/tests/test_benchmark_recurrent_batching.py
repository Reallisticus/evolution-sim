from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from scripts import benchmark_recurrent_batching as benchmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentBatchingBenchmarkTests(unittest.TestCase):
    def test_base_cpu_report_is_scoped_and_complete(self) -> None:
        protocol = benchmark.BenchmarkProtocol(
            widths=(4,),
            batch_sizes=(1, 2),
            device="cpu",
            dtype="float32",
            warmup=0,
            repeats=2,
            input_contract="base",
            genome_conditioning="disabled",
        )

        report = benchmark.run_benchmark(
            protocol,
            repository_root=REPOSITORY_ROOT,
        )

        self.assertEqual(
            report["schema_version"],
            benchmark.REPORT_SCHEMA_VERSION,
        )
        scope = report["scope"]
        self.assertEqual(
            scope["measured_operation"],
            "PublicRecurrentActorCritic.forward_sequence",
        )
        self.assertTrue(scope["model_forward_only"])
        self.assertFalse(scope["end_to_end_simulator_throughput_measured"])
        self.assertFalse(scope["end_to_end_simulator_claim_authorized"])
        self.assertIn("world_step", scope["excluded_work"])
        self.assertEqual(report["protocol"]["batch_sizes"], [1, 2])
        self.assertEqual(len(report["measurements"]), 2)

        for measurement in report["measurements"]:
            self.assertEqual(measurement["encoder_size"], 4)
            self.assertEqual(measurement["hidden_size"], 4)
            self.assertEqual(measurement["time_steps"], 1)
            self.assertEqual(measurement["public_input_size"], 541)
            self.assertFalse(measurement["genome_values_supplied"])
            self.assertIsNone(measurement["genome_size"])
            self.assertGreater(measurement["parameter_count"], 0)
            self.assertEqual(len(measurement["latency_ns"]["samples"]), 2)
            self.assertGreater(measurement["latency_ns"]["minimum"], 0)
            self.assertGreater(measurement["rows_per_second"]["median"], 0.0)
            self.assertFalse(measurement["peak_memory"]["available"])
            self.assertIsNone(measurement["peak_memory"]["peak_vram_bytes"])

        source = report["source"]
        self.assertRegex(source["commit_sha"], re.compile(r"[0-9a-f]{40}"))
        self.assertIs(type(source["dirty"]), bool)
        runtime = report["runtime"]
        self.assertEqual(runtime["device"]["resolved"], "cpu")
        self.assertEqual(runtime["device"]["dtype"], "float32")
        self.assertIn("version", runtime["torch"])
        self.assertIn("torch_build_version", runtime["cuda"])
        self.assertIn("version", runtime["cudnn"])
        self.assertIn(
            "deterministic_algorithms_enabled",
            runtime["determinism"],
        )
        self.assertIn("cuda_matmul_allow_tf32", runtime["tf32"])

    def test_tokenized_actor_film_supplies_explicit_genomes(self) -> None:
        protocol = benchmark.BenchmarkProtocol(
            widths=(4,),
            batch_sizes=(2,),
            device="cpu",
            dtype="float32",
            warmup=0,
            repeats=1,
            input_contract="tokenized",
            genome_conditioning="actor_film_v1",
            initialization_seed=7,
        )

        report = benchmark.run_benchmark(
            protocol,
            repository_root=REPOSITORY_ROOT,
        )

        measurement = report["measurements"][0]
        self.assertGreater(measurement["public_input_size"], 541)
        self.assertIn(
            "mind_ecological_policy_input_v3",
            measurement["public_input_schema_version"],
        )
        self.assertEqual(
            measurement["genome_conditioning_mode"],
            "actor_film_v1",
        )
        self.assertTrue(measurement["genome_values_supplied"])
        self.assertEqual(measurement["genome_size"], 16)

    def test_protocol_rejects_duplicates_and_non_positive_counts(self) -> None:
        with self.assertRaisesRegex(
            benchmark.BenchmarkConfigurationError,
            "widths cannot contain duplicates",
        ):
            benchmark.BenchmarkProtocol(
                widths=(4, 4),
                batch_sizes=(1,),
                device="cpu",
                dtype="float32",
                warmup=0,
                repeats=1,
                input_contract="base",
                genome_conditioning="disabled",
            ).validate()

        with self.assertRaisesRegex(
            benchmark.BenchmarkConfigurationError,
            "repeats must be a positive integer",
        ):
            benchmark.BenchmarkProtocol(
                widths=(4,),
                batch_sizes=(1,),
                device="cpu",
                dtype="float32",
                warmup=0,
                repeats=0,
                input_contract="base",
                genome_conditioning="disabled",
            ).validate()

    def test_unavailable_device_fails_closed(self) -> None:
        protocol = benchmark.BenchmarkProtocol(
            widths=(4,),
            batch_sizes=(1,),
            device="not-a-device",
            dtype="float32",
            warmup=0,
            repeats=1,
            input_contract="base",
            genome_conditioning="disabled",
        )

        with self.assertRaisesRegex(
            benchmark.BenchmarkConfigurationError,
            "invalid torch device",
        ):
            benchmark.run_benchmark(
                protocol,
                repository_root=REPOSITORY_ROOT,
            )

    def test_invalid_cli_emits_no_partial_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = benchmark.main(
                [
                    "--widths",
                    "0",
                    "--batch-sizes",
                    "1",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--warmup",
                    "0",
                    "--repeats",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("configuration rejected", stderr.getvalue())

    def test_canonical_json_is_compact_sorted_and_rejects_nan(self) -> None:
        payload = benchmark.canonical_json({"z": 1, "a": {"b": 2}})

        self.assertEqual(payload, '{"a":{"b":2},"z":1}')
        self.assertEqual(json.loads(payload), {"a": {"b": 2}, "z": 1})
        with self.assertRaises(ValueError):
            benchmark.canonical_json({"not_finite": float("nan")})


if __name__ == "__main__":
    unittest.main()
