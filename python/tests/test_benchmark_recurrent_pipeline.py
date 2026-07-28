from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import dataclasses
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from scripts import benchmark_recurrent_pipeline as benchmark
    import evolution_sim.mind.recurrent_experiment as recurrent_experiment
    from evolution_sim.mind.recurrent_rollout import (
        RecurrentFixedBatchRuntimeContract,
    )


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
            "fixed_batch_capacity": None,
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
        benchmark_seeds = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            benchmark.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
        ]
        learner_seed = benchmark_seeds[
            benchmark.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
        ]
        genome_stream_seed = benchmark_seeds[
            benchmark.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
        ]
        protocol = self._protocol(
            input_contract="tokenized",
            genome_conditioning="actor_film_v1",
            genome_population_mode="heritable",
            genome_stream_seed=genome_stream_seed,
            learner_seed=learner_seed,
            fixed_batch_capacity=(
                benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
            ),
        )

        protocol.validate()
        for overrides, message in (
            ({"scenarios": ("broad", "carrion_only")}, "broad-only"),
            ({"learner_seed": 7}, "model seed"),
            ({"genome_stream_seed": 7}, "genome seed"),
            (
                {
                    "updates": 1,
                    "worlds_per_update": (
                        benchmark.OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT + 1
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
                    "fixed_batch_capacity": (
                        benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
                    ),
                    **overrides,
                }
                with self.assertRaisesRegex(
                    benchmark.PipelineBenchmarkConfigurationError,
                    message,
                ):
                    self._protocol(**values).validate()

    def test_tiny_tokenized_pipeline_executes_phase_a_contract(self) -> None:
        benchmark_seeds = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            benchmark.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
        ]
        learner_seed = benchmark_seeds[
            benchmark.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
        ]
        genome_stream_seed = benchmark_seeds[
            benchmark.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
        ]

        report = benchmark.run_benchmark(
            self._protocol(
                input_contract="tokenized",
                genome_conditioning="actor_film_v1",
                genome_population_mode="heritable",
                genome_stream_seed=genome_stream_seed,
                learner_seed=learner_seed,
                fixed_batch_capacity=(
                    benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
                ),
            ),
            repository_root=REPOSITORY_ROOT,
        )

        self.assertEqual(
            report["protocol"]["schedule_contract"],
            "canonical_open_ecology_operational_benchmark_phase_a_v1",
        )
        self.assertEqual(report["protocol"]["training_phase"], "phase_a")
        self.assertEqual(
            report["protocol"]["ppo_config"]["learning_rate"],
            0.0002,
        )
        self.assertTrue(report["protocol"]["ppo_config"]["world_balanced_loss"])
        self.assertFalse(report["preregistered_gate"]["member_shape_valid"])
        self.assertFalse(report["preregistered_gate"]["complete_gate_claimed"])
        self.assertEqual(
            report["seed_access"]["environment_seed_roles"],
            [benchmark.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE],
        )
        self.assertFalse(report["seed_access"]["training_seeds_accessed"])
        self.assertTrue(report["collector"]["fixed_batch_enabled"])
        self.assertEqual(
            report["collector"]["fixed_batch_capacity"],
            benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        )

    def test_collector_contract_is_derived_from_executed_world_provenance(
        self,
    ) -> None:
        declared = RecurrentFixedBatchRuntimeContract(
            batch_capacity=benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
        ).as_contract()
        observed = RecurrentFixedBatchRuntimeContract(
            batch_capacity=benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY - 1
        ).as_contract()
        result = SimpleNamespace(
            contract_version=(
                benchmark.RECURRENT_FIXED_BATCH_EXPERIMENT_CONTRACT_VERSION
            ),
            rollout_execution={
                "open_ecology": {
                    "fixed_batch_enabled": True,
                    "fixed_batch_runtime_contract": declared,
                }
            },
            updates=(
                SimpleNamespace(
                    tasks=(SimpleNamespace(task_id="world-0"),),
                    rollout=SimpleNamespace(
                        world_seed_provenance={
                            "world-0": {
                                "fixed_batch_runtime": {
                                    "batch_capacity": observed["batch_capacity"],
                                    "contract": observed,
                                }
                            }
                        }
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkExecutionError,
            "differs from executed worlds",
        ):
            benchmark._observed_collector_contract(result)

    def test_benchmark_seed_provenance_uses_actual_nonuniform_update_groups(
        self,
    ) -> None:
        flat_tasks = list(
            recurrent_experiment.build_open_ecology_benchmark_schedule(
                update_count=1,
                worlds_per_update=3,
                rollout_ticks=1,
                genome_population_mode="heritable",
            )[0]
        )

        def rebind_update(task: object, update_index: int) -> object:
            world_index = task.open_ecology_world_index  # type: ignore[attr-defined]
            environment_index = task.open_ecology_environment_seed_index  # type: ignore[attr-defined]
            initial_agents = task.open_ecology_treatment.initial_agents  # type: ignore[attr-defined]
            population_mode = task.genome_population_mode  # type: ignore[attr-defined]
            identity = (
                recurrent_experiment._open_ecology_benchmark_policy_sampling_identity(
                    genome_population_mode=population_mode,
                    update_index=update_index,
                    world_index=world_index,
                    environment_seed_index=environment_index,
                    initial_agents=initial_agents,
                )
            )
            return dataclasses.replace(
                task,
                task_id=recurrent_experiment._open_ecology_benchmark_task_id(
                    genome_population_mode=population_mode,
                    update_index=update_index,
                    world_index=world_index,
                    environment_seed_index=environment_index,
                    environment_seed=task.environment_seed,  # type: ignore[attr-defined]
                    initial_agents=initial_agents,
                ),
                policy_sampling_identity=identity,
                policy_sampling_seed=(
                    recurrent_experiment.derive_recurrent_policy_sampling_seed(
                        task_identity=identity
                    )
                ),
                open_ecology_update_index=update_index,
            )

        second = rebind_update(flat_tasks[1], 1)
        third = rebind_update(flat_tasks[2], 1)
        provenance = recurrent_experiment._open_ecology_benchmark_seed_provenance(
            (
                SimpleNamespace(tasks=(flat_tasks[0],)),
                SimpleNamespace(tasks=(second, third)),
            )
        )

        self.assertEqual(provenance["environment_seed_indices"], [0, 1, 2])

    def test_old_scientific_seed_and_scalar_protocols_fail_closed(self) -> None:
        common = {
            "input_contract": "tokenized",
            "genome_conditioning": "actor_film_v1",
            "genome_population_mode": "heritable",
            "fixed_batch_capacity": (
                benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
            ),
        }
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "dedicated operational seed",
        ):
            self._protocol(
                **common,
                learner_seed=benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
                    "open_ecology_learner"
                ][0],
                genome_stream_seed=benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
                    "open_ecology_genome_stream"
                ][0],
            ).validate()

        benchmark_seeds = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            benchmark.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
        ]
        with self.assertRaisesRegex(
            benchmark.PipelineBenchmarkConfigurationError,
            "canonical fixed-batch capacity",
        ):
            self._protocol(
                input_contract="tokenized",
                genome_conditioning="actor_film_v1",
                genome_population_mode="heritable",
                learner_seed=benchmark_seeds[
                    benchmark.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
                ],
                genome_stream_seed=benchmark_seeds[
                    benchmark.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
                ],
                fixed_batch_capacity=None,
            ).validate()

    def test_preregistered_gate_member_requires_exact_full_shape(self) -> None:
        benchmark_seeds = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            benchmark.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
        ]
        learner_seed = benchmark_seeds[
            benchmark.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX
        ]
        genome_stream_seed = benchmark_seeds[
            benchmark.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
        ]
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
            fixed_batch_capacity=(
                benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
            ),
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

    def test_cli_seed_defaults_preserve_each_input_contract(self) -> None:
        required = [
            "--worker-counts",
            "1",
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
        legacy = benchmark.protocol_from_args(
            benchmark.build_parser().parse_args(required)
        )
        self.assertEqual(
            legacy.learner_seed,
            benchmark.OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
        )

        operational_seeds = benchmark.OPEN_ECOLOGY_SEED_REGISTRY[
            benchmark.OPEN_ECOLOGY_BENCHMARK_SEED_ROLE
        ]
        tokenized = benchmark.protocol_from_args(
            benchmark.build_parser().parse_args(
                [
                    *required,
                    "--input-contract",
                    "tokenized",
                    "--genome-conditioning",
                    "actor_film_v1",
                    "--genome-population-mode",
                    "heritable",
                    "--genome-stream-seed",
                    str(
                        operational_seeds[
                            benchmark.OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX
                        ]
                    ),
                    "--fixed-batch-capacity",
                    str(benchmark.OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY),
                ]
            )
        )
        self.assertEqual(
            tokenized.learner_seed,
            operational_seeds[benchmark.OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX],
        )

    def test_canonical_json_is_compact_and_rejects_nan(self) -> None:
        payload = benchmark.canonical_json({"z": 1, "a": {"b": 2}})

        self.assertEqual(payload, '{"a":{"b":2},"z":1}')
        self.assertEqual(json.loads(payload), {"a": {"b": 2}, "z": 1})
        with self.assertRaises(ValueError):
            benchmark.canonical_json({"not_finite": float("nan")})


if __name__ == "__main__":
    unittest.main()
