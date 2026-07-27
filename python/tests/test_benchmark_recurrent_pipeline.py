from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import dataclasses
import io
import json
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from scripts import benchmark_recurrent_pipeline as benchmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentPipelineBenchmarkTests(unittest.TestCase):
    def _protocol(self, **overrides: object) -> benchmark.PipelineBenchmarkProtocol:
        values: dict[str, object] = {
            "worker_counts": (1,),
            "repeats": 1,
            "updates": 1,
            "worlds_per_update": 1,
            "rollout_ticks": 1,
            "scenarios": ("broad",),
            "device": "cpu",
            "input_contract": "base",
            "genome_conditioning": "disabled",
            "genome_population_mode": "disabled",
            "genome_stream_seed": None,
            "encoder_size": 4,
            "hidden_size": 4,
            "recurrent_layers": 1,
            "learner_seed": 7,
            "update_epochs": 1,
            "sequence_minibatch_size": 128,
            "tbptt_steps": 2,
            "burn_in_steps": 0,
        }
        values.update(overrides)
        return benchmark.PipelineBenchmarkProtocol(**values)  # type: ignore[arg-type]

    def test_cpu_report_measures_full_pipeline_and_pins_determinism(self) -> None:
        report = benchmark.run_benchmark(
            self._protocol(),
            repository_root=REPOSITORY_ROOT,
        )

        self.assertEqual(report["schema_version"], benchmark.REPORT_SCHEMA_VERSION)
        scope = report["scope"]
        self.assertTrue(scope["world_construction_included"])
        self.assertTrue(scope["world_ticks_included"])
        self.assertTrue(scope["ppo_update_included"])
        self.assertFalse(scope["artifact_serialization_included"])
        self.assertFalse(scope["policy_promotion_authorized"])
        self.assertEqual(report["protocol"]["scheduled_worlds"], 1)
        self.assertEqual(report["protocol"]["scheduled_world_ticks"], 2)
        self.assertEqual(report["protocol"]["timing_boundary"], "runner.run_only")
        self.assertTrue(report["determinism"]["cross_worker_model_state_match"])
        self.assertTrue(report["determinism"]["cross_worker_semantic_evidence_match"])
        self.assertEqual(len(report["cases"]), 1)
        case = report["cases"][0]
        self.assertEqual(case["rollout_workers"], 1)
        self.assertEqual(case["speedup_vs_first_case"], 1.0)
        self.assertEqual(case["parallel_efficiency_vs_first_case"], 1.0)
        self.assertEqual(len(case["samples"]), 1)
        sample = case["samples"][0]
        self.assertEqual(sample["total_worlds"], 1)
        self.assertGreater(sample["total_transitions"], 0)
        self.assertGreater(sample["worlds_per_second"], 0.0)
        self.assertGreater(sample["transitions_per_second"], 0.0)
        self.assertEqual(
            sample["final_model_state_sha256"],
            report["determinism"]["final_model_state_sha256"],
        )

    def test_conditioning_contract_rejects_implicit_or_mixed_genomes(self) -> None:
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "disabled conditioning",
        ):
            self._protocol(
                genome_population_mode="heritable",
                genome_stream_seed=1,
            ).validate()
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "requires heritable or zero_all",
        ):
            self._protocol(
                genome_conditioning="actor_film_v1",
            ).validate()
        self._protocol(
            genome_conditioning="actor_film_v1",
            genome_population_mode="zero_all",
            genome_stream_seed=2**64 - 1,
        ).validate()

    def test_tokenized_workload_is_exact_registered_open_ecology(self) -> None:
        learner_seed = benchmark.OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0]
        genome_stream_seed = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            "open_ecology_genome_stream"
        ][0]
        protocol = self._protocol(
            input_contract="tokenized",
            genome_conditioning="actor_film_v1",
            genome_population_mode="heritable",
            genome_stream_seed=genome_stream_seed,
            learner_seed=learner_seed,
        )

        protocol.validate()
        for overrides, message in (
            ({"scenarios": ("broad", "carrion_only")}, "broad-only"),
            ({"learner_seed": 7}, "learner seed"),
            ({"genome_stream_seed": 7}, "genome stream seed"),
            (
                {
                    "updates": 1,
                    "worlds_per_update": (
                        len(benchmark.OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_train"])
                        + 1
                    ),
                },
                "non-reused",
            ),
        ):
            with self.subTest(overrides=overrides):
                values = {
                    "input_contract": "tokenized",
                    "genome_conditioning": "actor_film_v1",
                    "genome_population_mode": "heritable",
                    "genome_stream_seed": genome_stream_seed,
                    "learner_seed": learner_seed,
                    **overrides,
                }
                with self.assertRaisesRegex(
                    benchmark.PipelineBenchmarkConfigurationError,
                    message,
                ):
                    self._protocol(**values).validate()

    def test_tiny_tokenized_pipeline_executes_phase_a_contract(self) -> None:
        learner_seed = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            "open_ecology_learner"
        ][0]
        genome_stream_seed = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            "open_ecology_genome_stream"
        ][0]

        report = benchmark.run_benchmark(
            self._protocol(
                input_contract="tokenized",
                genome_conditioning="actor_film_v1",
                genome_population_mode="heritable",
                genome_stream_seed=genome_stream_seed,
                learner_seed=learner_seed,
            ),
            repository_root=REPOSITORY_ROOT,
        )

        self.assertEqual(
            report["protocol"]["schedule_contract"],
            "canonical_open_ecology_phase_a_broad_treatment_v1",
        )
        self.assertEqual(report["protocol"]["training_phase"], "phase_a")
        self.assertEqual(
            report["protocol"]["ppo_config"]["learning_rate"],
            0.0002,
        )
        self.assertTrue(report["protocol"]["ppo_config"]["world_balanced_loss"])
        self.assertFalse(report["preregistered_gate"]["member_shape_valid"])
        self.assertFalse(report["preregistered_gate"]["complete_gate_claimed"])

    def test_preregistered_gate_member_requires_exact_full_shape(self) -> None:
        learner_seed = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            "open_ecology_learner"
        ][0]
        genome_stream_seed = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            "open_ecology_genome_stream"
        ][0]
        exact = self._protocol(
            worker_counts=(1, 2, 4, 8, 16),
            updates=1,
            worlds_per_update=16,
            rollout_ticks=128,
            device="cuda",
            input_contract="tokenized",
            genome_conditioning="actor_film_v1",
            genome_population_mode="heritable",
            genome_stream_seed=genome_stream_seed,
            learner_seed=learner_seed,
            encoder_size=256,
            hidden_size=256,
            update_epochs=4,
            sequence_minibatch_size=16,
            tbptt_steps=128,
            burn_in_steps=16,
            preregistered_gate_member=True,
        )

        exact.validate()
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "worker_counts",
        ):
            dataclasses.replace(exact, worker_counts=(1, 2, 4)).validate()

    def test_protocol_rejects_duplicate_or_excessive_workers(self) -> None:
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "non-empty and unique",
        ):
            self._protocol(worker_counts=(1, 1)).validate()
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "must not exceed",
        ):
            self._protocol(
                worker_counts=(benchmark.MAX_RECURRENT_ROLLOUT_WORKERS + 1,)
            ).validate()

    def test_invalid_cli_emits_no_partial_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = benchmark.main(
                [
                    "--worker-counts",
                    "0",
                    "--repeats",
                    "1",
                    "--updates",
                    "1",
                    "--worlds-per-update",
                    "1",
                    "--rollout-ticks",
                    "1",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("configuration rejected", stderr.getvalue())

    def test_canonical_json_is_compact_and_rejects_nan(self) -> None:
        payload = benchmark.canonical_json({"z": 1, "a": {"b": 2}})

        self.assertEqual(payload, '{"a":{"b":2},"z":1}')
        self.assertEqual(json.loads(payload), {"a": {"b": 2}, "z": 1})
        with self.assertRaises(ValueError):
            benchmark.canonical_json({"not_finite": float("nan")})


if __name__ == "__main__":
    unittest.main()
