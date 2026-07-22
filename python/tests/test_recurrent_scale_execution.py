from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from evolution_sim.mind.provenance import stable_payload_digest

if torch is not None:
    from evolution_sim.mind.recurrent_actor_critic import PublicRecurrentActorCritic
    from evolution_sim.mind.recurrent_artifact import (
        FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
        load_recurrent_training_crash_checkpoint,
        save_recurrent_training_crash_checkpoint,
    )
    from evolution_sim.mind.recurrent_counterfactual_comparison import (
        BASE_ARM,
        EXACT_ARM,
        SHUFFLED_ARM,
    )
    from evolution_sim.mind.recurrent_policy import (
        PUBLIC_RECURRENT_ARGMAX_SELECTION,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
    )
    from evolution_sim.mind.recurrent_runtime_provenance import (
        build_recurrent_scale_runtime_provenance,
    )
    from evolution_sim.mind.recurrent_scale_campaign import (
        RECURRENT_SCALE_ARM_REPORT_SCHEMA_VERSION,
        RECURRENT_SCALE_ARMS,
        RECURRENT_SCALE_TOTAL_TRAINING_WORLDS,
        build_recurrent_scale_campaign_preregistration,
        load_strict_json,
        recurrent_scale_arm_run_id,
        recurrent_scale_selection_seed_plan,
        write_atomic_json,
    )
    from evolution_sim.mind.recurrent_scale_execution import (
        RECURRENT_SCALE_FULL_WORLD_VERIFICATION_RUNNER,
        _scale_policy_sampling_seeds,
        _scale_run_contract,
        _scale_sampling_stream_id,
        _scale_training_seed_provenance,
        _verify_completed_evaluation_evidence,
        analyze_recurrent_scale_campaign,
        build_scale_run_components,
        load_scale_arm_reports,
        validate_recurrent_scale_arm_report,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        SCALE_DEVELOPMENT_SEED_REGISTRY,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentScaleExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preregistration = build_recurrent_scale_campaign_preregistration(
            source_commit="a" * 40,
            source_manifest_sha256="b" * 64,
        )
        self.runtime_provenance = build_recurrent_scale_runtime_provenance(
            source_commit="a" * 40,
            source_manifest_sha256="b" * 64,
            preregistration_digest=str(self.preregistration["exact_digest"]),
            repository_clean=True,
            device="cpu",
            rollout_workers=1,
            counterfactual_workers=1,
            evaluation_workers=1,
        )
        self.runtime_provenance["device"] = {
            "type": "cuda",
            "index": 0,
            "name": "synthetic-cuda",
            "compute_capability": [8, 9],
            "total_memory_bytes": 12 * 1024**3,
            "multiprocessor_count": 56,
        }
        self.runtime_provenance["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in self.runtime_provenance.items()
                if key != "exact_digest"
            }
        )
        self.runtime_provenance_reference = {
            "path": "/synthetic/runtime-provenance.json",
            "sha256": stable_payload_digest(self.runtime_provenance),
            "byte_length": 4096,
        }

    def test_base_components_use_full_scale_schedule_and_roles(self) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        model, ppo, counterfactual, schedule = build_scale_run_components(
            self.preregistration,
            learner_seed=learner_seed,
            arm=BASE_ARM,
            counterfactual_workers=2,
        )
        self.assertEqual(model.hidden_size, 192)
        self.assertEqual(ppo.learner_seed, learner_seed)
        self.assertIsNone(counterfactual)
        self.assertEqual(len(schedule), 16)
        self.assertEqual(sum(len(update) for update in schedule), 256)
        self.assertEqual(
            {task.seed_role for update in schedule for task in update},
            {"scale_train", "scale_curriculum"},
        )

    def test_scale_components_pair_arms_within_learner_not_across_learners(
        self,
    ) -> None:
        first_learner, second_learner = SCALE_DEVELOPMENT_SEED_REGISTRY[
            "scale_learner"
        ][:2]
        schedules = []
        for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM):
            _model, _ppo, _counterfactual, schedule = build_scale_run_components(
                self.preregistration,
                learner_seed=first_learner,
                arm=arm,
                counterfactual_workers=1,
            )
            schedules.append(schedule)
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(schedules[0], schedules[2])

        _model, _ppo, _counterfactual, second_schedule = build_scale_run_components(
            self.preregistration,
            learner_seed=second_learner,
            arm=BASE_ARM,
            counterfactual_workers=1,
        )
        first_tasks = tuple(task for update in schedules[0] for task in update)
        second_tasks = tuple(task for update in second_schedule for task in update)
        self.assertEqual(
            tuple((task.scenario, task.environment_seed) for task in first_tasks),
            tuple((task.scenario, task.environment_seed) for task in second_tasks),
        )
        self.assertTrue(
            all(
                first.task_id != second.task_id
                and first.policy_sampling_identity != second.policy_sampling_identity
                and first.policy_sampling_seed != second.policy_sampling_seed
                for first, second in zip(first_tasks, second_tasks, strict=True)
            )
        )

    def test_exact_components_bind_multi_tape_terminal_aggregate_treatment(
        self,
    ) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        _model, _ppo, counterfactual, schedule = build_scale_run_components(
            self.preregistration,
            learner_seed=learner_seed,
            arm=EXACT_ARM,
            counterfactual_workers=3,
        )
        assert counterfactual is not None
        self.assertEqual(counterfactual.collection.continuation_tape_count, 8)
        self.assertEqual(counterfactual.collection.terminal_target_world_tick, 120)
        self.assertEqual(counterfactual.auxiliary.terminal_target_weight, 0.5)
        self.assertEqual(counterfactual.auxiliary.value_loss_coefficient, 0.0)
        self.assertEqual(counterfactual.branch_tick_candidates, (16, 40, 64, 72))
        self.assertEqual(len(schedule), 16)

    def test_all_scale_run_contracts_are_exact_json_round_trips(self) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM):
            with self.subTest(arm=arm):
                model, ppo, counterfactual, schedule = build_scale_run_components(
                    self.preregistration,
                    learner_seed=learner_seed,
                    arm=arm,
                    counterfactual_workers=2,
                )
                contract = _scale_run_contract(
                    preregistration=self.preregistration,
                    learner_seed=learner_seed,
                    arm=arm,
                    model_config=model,
                    ppo_config=ppo,
                    counterfactual_config=counterfactual,
                    schedule=schedule,
                )
                reloaded = json.loads(
                    json.dumps(contract, sort_keys=True, allow_nan=False)
                )

                self.assertEqual(reloaded, contract)
                if arm != BASE_ARM:
                    self.assertIsInstance(
                        contract["counterfactual"]["collection"]["horizons"],  # type: ignore[index]
                        list,
                    )

    def test_treatment_checkpoint_training_config_survives_write_load_for_resume(
        self,
    ) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        with tempfile.TemporaryDirectory() as tmpdir:
            for arm in (EXACT_ARM, SHUFFLED_ARM):
                with self.subTest(arm=arm):
                    model_config, ppo, counterfactual, schedule = (
                        build_scale_run_components(
                            self.preregistration,
                            learner_seed=learner_seed,
                            arm=arm,
                            counterfactual_workers=2,
                        )
                    )
                    contract = _scale_run_contract(
                        preregistration=self.preregistration,
                        learner_seed=learner_seed,
                        arm=arm,
                        model_config=model_config,
                        ppo_config=ppo,
                        counterfactual_config=counterfactual,
                        schedule=schedule,
                    )
                    run_id = recurrent_scale_arm_run_id(
                        learner_seed=learner_seed,
                        arm=arm,
                    )
                    checkpoint_path = Path(tmpdir) / arm / "checkpoint.json"
                    save_recurrent_training_crash_checkpoint(
                        checkpoint_path,
                        PublicRecurrentActorCritic(
                            model_config,
                            initialization_seed=learner_seed,
                        ),
                        optimizer_state={},
                        rng_state={},
                        optimizer_type="torch.optim.Adam",
                        training_config=contract,
                        seed_registry_digest=self.preregistration["seed_contract"][  # type: ignore[index]
                            "registry_sha256"
                        ],
                        source_commit=self.preregistration["source"]["commit"],  # type: ignore[index]
                        source_manifest_sha256=self.preregistration["source"][  # type: ignore[index]
                            "manifest_sha256"
                        ],
                        learner_seed=learner_seed,
                        completed_updates=1,
                        run_id=run_id,
                    )

                    loaded = load_recurrent_training_crash_checkpoint(checkpoint_path)
                    loaded_configuration = loaded.checkpoint["configuration"]
                    self.assertEqual(
                        loaded_configuration["training_config"],  # type: ignore[index]
                        contract,
                    )

    def test_written_treatment_reports_reload_and_validate(self) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        with tempfile.TemporaryDirectory() as tmpdir:
            for arm in (EXACT_ARM, SHUFFLED_ARM):
                with self.subTest(arm=arm):
                    report = self._report(learner_seed=learner_seed, arm=arm)
                    report_path = Path(tmpdir) / arm / "report.json"
                    write_atomic_json(report_path, report)
                    reloaded = load_strict_json(report_path)

                    self.assertEqual(reloaded, report)
                    validate_recurrent_scale_arm_report(
                        reloaded,
                        preregistration=self.preregistration,
                    )

    def test_complete_paired_matrix_passes_declared_synthetic_outcomes(self) -> None:
        reports = [
            self._report(learner_seed=learner_seed, arm=arm)
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
            for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
        ]
        analysis = analyze_recurrent_scale_campaign(
            self.preregistration,
            reports,
        )
        self.assertEqual(
            analysis["total_training_worlds"],
            RECURRENT_SCALE_TOTAL_TRAINING_WORLDS,
        )
        self.assertEqual(len(analysis["input_reports"]), 24)
        for run_id, evidence in analysis["input_reports"].items():
            self.assertEqual(len(run_id), len(str(run_id)))
            unsigned = dict(evidence)
            exact_digest = unsigned.pop("exact_digest")
            self.assertEqual(stable_payload_digest(unsigned), exact_digest)
        self.assertTrue(analysis["accepted"])
        self.assertTrue(all(analysis["gates"].values()))

    def test_analysis_digest_binds_report_artifact_and_evaluation_evidence(
        self,
    ) -> None:
        reports = [
            self._report(learner_seed=learner_seed, arm=arm)
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
            for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
        ]
        original = analyze_recurrent_scale_campaign(self.preregistration, reports)
        tampered = copy.deepcopy(reports)
        target = tampered[0]
        target["artifact"]["artifact_sha256"] = "e" * 64  # type: ignore[index]
        target["evaluations"]["modes"][PUBLIC_RECURRENT_ARGMAX_SELECTION][  # type: ignore[index]
            "artifact_cpu_report"
        ]["sha256"] = "f" * 64
        self._resign(target)

        changed = analyze_recurrent_scale_campaign(self.preregistration, tampered)

        run_id = target["run_id"]
        self.assertNotEqual(original["exact_digest"], changed["exact_digest"])
        self.assertNotEqual(
            original["input_reports"][run_id],  # type: ignore[index]
            changed["input_reports"][run_id],  # type: ignore[index]
        )

    def test_campaign_rejects_mixed_runtime_provenance(self) -> None:
        reports = [
            self._report(learner_seed=learner_seed, arm=arm)
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
            for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
        ]
        target = reports[0]
        runtime = target["runtime_provenance"]  # type: ignore[index]
        payload = runtime["payload"]
        payload["workers"]["rollout_workers"] = 2
        payload["exact_digest"] = stable_payload_digest(
            {key: value for key, value in payload.items() if key != "exact_digest"}
        )
        runtime["file"]["sha256"] = stable_payload_digest(payload)
        target["artifact"]["runtime_provenance_digest"] = payload["exact_digest"]  # type: ignore[index]
        self._resign(target)

        with self.assertRaisesRegex(ValueError, "one pinned runtime"):
            analyze_recurrent_scale_campaign(self.preregistration, reports)

    def test_report_contract_tampering_fails_closed(self) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        mutations = {
            "source": lambda report: report["source"].__setitem__(  # type: ignore[union-attr]
                "manifest_sha256", "c" * 64
            ),
            "configuration": lambda report: report["configuration"].__setitem__(  # type: ignore[union-attr]
                "schedule_sha256", "d" * 64
            ),
            "selection seed plan": lambda report: report["evaluations"].__setitem__(  # type: ignore[union-attr]
                "selection_seed_plan_sha256", "e" * 64
            ),
            "checkpoint evidence": lambda report: report["training"][  # type: ignore[index]
                "checkpoint"
            ].__setitem__("byte_length", 0),
            "update evidence": lambda report: report["training"].__setitem__(  # type: ignore[index]
                "update_record_digests",
                ["f" * 64] * 16,
            ),
            "evaluation evidence": lambda report: report["evaluations"][  # type: ignore[index]
                "modes"
            ][
                PUBLIC_RECURRENT_SAMPLED_SELECTION
            ]["in_memory_report"].__setitem__("sha256", "not-a-sha"),
            "artifact manifest": lambda report: report["artifact"][  # type: ignore[index]
                "full_world_replay_manifest"
            ].__setitem__("world_count", 79),
            "behavior digest": lambda report: report["evaluations"]["modes"][  # type: ignore[index]
                PUBLIC_RECURRENT_ARGMAX_SELECTION
            ][
                "candidate_outcome_summary"
            ]["contexts"][0]["runs"][0].pop("behavior_digest"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                report = self._report(learner_seed=learner_seed, arm=EXACT_ARM)
                mutate(report)
                self._resign(report)
                with self.assertRaises(ValueError):
                    validate_recurrent_scale_arm_report(
                        report,
                        preregistration=self.preregistration,
                    )

    def test_paired_fixture_identity_tampering_fails_closed(self) -> None:
        def duplicate(runs: list[dict[str, object]]) -> None:
            runs[0]["policy_sampling_seed"] = runs[1]["policy_sampling_seed"]

        def missing(runs: list[dict[str, object]]) -> None:
            runs.pop()

        def mismatch(runs: list[dict[str, object]]) -> None:
            runs[0]["policy_sampling_seed"] = 0

        for name, mutate in {
            "duplicate": duplicate,
            "missing": missing,
            "mismatch": mismatch,
        }.items():
            with self.subTest(name=name):
                reports = [
                    self._report(learner_seed=learner_seed, arm=arm)
                    for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
                    for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
                ]
                exact_report = reports[1]
                sampled_fixture = exact_report["evaluations"]["modes"][  # type: ignore[index]
                    PUBLIC_RECURRENT_SAMPLED_SELECTION
                ]["candidate_outcome_summary"]["contexts"][1]
                mutate(sampled_fixture["runs"])
                self._resign(exact_report)

                with self.assertRaisesRegex(ValueError, "duplicate|identit"):
                    analyze_recurrent_scale_campaign(self.preregistration, reports)

    def test_duplicate_cell_fails_closed(self) -> None:
        reports = [
            self._report(learner_seed=learner_seed, arm=arm)
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
            for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
        ]
        reports[-1] = copy.deepcopy(reports[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_recurrent_scale_campaign(self.preregistration, reports)

    def test_exact_regression_against_fixed_linear_control_fails_gate(self) -> None:
        reports = [
            self._report(learner_seed=learner_seed, arm=arm)
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]
            for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
        ]
        for report in reports:
            modes = report["evaluations"]["modes"]  # type: ignore[index]
            for mode in modes.values():  # type: ignore[union-attr]
                summary = mode["candidate_outcome_summary"]
                broad = summary["contexts"][0]
                for run in broad["linear_runs"]:
                    run["terminal_alive"] = 2
                    run["births"] = 3
            report.pop("exact_digest")
            report["exact_digest"] = stable_payload_digest(report)

        analysis = analyze_recurrent_scale_campaign(self.preregistration, reports)

        self.assertFalse(analysis["accepted"])
        self.assertFalse(
            analysis["gates"][  # type: ignore[index]
                "broad_alive_birth_strict_per_seed_nonregression"
            ]
        )

    def test_aggregate_loader_rejects_self_consistent_reports_with_missing_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_path = root / "runtime-provenance.json"
            write_atomic_json(runtime_path, self.runtime_provenance)
            runtime_reference = self._actual_file_reference(runtime_path)
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]:
                for arm in (BASE_ARM, EXACT_ARM, SHUFFLED_ARM):
                    report = self._report(learner_seed=learner_seed, arm=arm)
                    report["runtime_provenance"]["file"] = runtime_reference  # type: ignore[index]
                    self._resign(report)
                    write_atomic_json(
                        root / str(report["run_id"]) / "report.json",
                        report,
                    )

            with self.assertRaisesRegex(ValueError, "evidence file is missing"):
                load_scale_arm_reports(
                    self.preregistration,
                    output_root=root,
                    runtime_provenance_path=runtime_path,
                )

    def test_aggregate_loader_rejects_report_summary_detached_from_evaluation_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_path = root / "runtime-provenance.json"
            write_atomic_json(runtime_path, self.runtime_provenance)
            runtime_reference = self._actual_file_reference(runtime_path)
            first_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
            first_arm = RECURRENT_SCALE_ARMS[0]
            first_report: dict[str, object] | None = None
            first_run_directory: Path | None = None
            for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]:
                for arm in RECURRENT_SCALE_ARMS:
                    report = self._report(learner_seed=learner_seed, arm=arm)
                    report["runtime_provenance"]["file"] = runtime_reference  # type: ignore[index]
                    run_directory = root / str(report["run_id"])
                    if learner_seed == first_seed and arm == first_arm:
                        first_report = report
                        first_run_directory = run_directory
                        checkpoint_path = run_directory / "training-checkpoint.json"
                        artifact_path = run_directory / "frozen-policy.json"
                        write_atomic_json(checkpoint_path, {"synthetic": "checkpoint"})
                        write_atomic_json(artifact_path, {"synthetic": "artifact"})
                        report["training"]["checkpoint"] = (  # type: ignore[index]
                            self._actual_file_reference(checkpoint_path)
                        )
                        report["artifact"]["path"] = str(artifact_path)  # type: ignore[index]
                        report["artifact"]["file"] = self._actual_file_reference(  # type: ignore[index]
                            artifact_path
                        )
                        modes = report["evaluations"]["modes"]  # type: ignore[index]
                        for mode, evidence in modes.items():  # type: ignore[union-attr]
                            evaluation = self._evaluation_from_summary(
                                evidence["candidate_outcome_summary"]
                            )
                            for label in ("in-memory", "artifact-cpu"):
                                path = (
                                    run_directory
                                    / "evaluations"
                                    / f"{label}-{mode}.json"
                                )
                                persisted_evaluation = copy.deepcopy(evaluation)
                                if label == "artifact-cpu":
                                    persisted_evaluation["artifact"] = {
                                        "path": str(artifact_path),
                                        "artifact_sha256": report["artifact"][  # type: ignore[index]
                                            "artifact_sha256"
                                        ],
                                    }
                                write_atomic_json(path, persisted_evaluation)
                                evidence[f"{label.replace('-', '_')}_report"] = (
                                    self._actual_file_reference(path)
                                )
                        modes[PUBLIC_RECURRENT_ARGMAX_SELECTION][  # type: ignore[index]
                            "candidate_outcome_summary"
                        ]["contexts"][1]["runs"][0]["terminal_alive"] += 1
                    self._resign(report)
                    write_atomic_json(run_directory / "report.json", report)

            assert first_report is not None
            assert first_run_directory is not None
            checkpoint = SimpleNamespace(
                checkpoint={
                    "progress": {
                        "completed_updates": 16,
                        "run_id": first_report["run_id"],
                        "learner_seed": first_seed,
                    },
                    "source": {
                        "source_commit": first_report["source"]["commit"],  # type: ignore[index]
                        "source_manifest_sha256": first_report["source"][  # type: ignore[index]
                            "manifest_sha256"
                        ],
                        "seed_registry_digest": self.preregistration["seed_contract"][  # type: ignore[index]
                            "registry_sha256"
                        ],
                    },
                    "configuration": {
                        "training_config": first_report["configuration"],
                        "model_config": first_report["configuration"]["model"],  # type: ignore[index]
                    },
                    "parameters_sha256": "c" * 64,
                },
                rng_state={
                    "scale_runtime_provenance_digest": self.runtime_provenance[
                        "exact_digest"
                    ]
                },
            )
            artifact = SimpleNamespace(
                artifact={
                    "artifact_sha256": first_report["artifact"][  # type: ignore[index]
                        "artifact_sha256"
                    ],
                    "provenance": {
                        "run_metadata": {
                            "runtime_provenance_digest": self.runtime_provenance[
                                "exact_digest"
                            ],
                            "run_id": first_report["run_id"],
                            "arm": first_arm,
                            "preregistration_digest": self.preregistration[
                                "exact_digest"
                            ],
                            "runtime_integration_authorized": False,
                            "promotion_authorized": False,
                        },
                        "data_metadata": {
                            "runtime_provenance_digest": self.runtime_provenance[
                                "exact_digest"
                            ],
                            "policy_induced": True,
                        },
                        "source_commit": first_report["source"]["commit"],  # type: ignore[index]
                        "source_manifest_sha256": first_report["source"][  # type: ignore[index]
                            "manifest_sha256"
                        ],
                        "experiment_config": first_report["configuration"],
                        "training_config": first_report["configuration"]["ppo"],  # type: ignore[index]
                        "seed_registry_digest": self.preregistration["seed_contract"][  # type: ignore[index]
                            "registry_sha256"
                        ],
                        "learner_seed": first_seed,
                    },
                    "integrity": {"parameters_sha256": "c" * 64},
                }
            )
            with (
                patch(
                    "evolution_sim.mind.recurrent_scale_execution."
                    "load_recurrent_training_crash_checkpoint",
                    return_value=checkpoint,
                ),
                patch(
                    "evolution_sim.mind.recurrent_scale_execution."
                    "load_frozen_recurrent_policy_artifact",
                    return_value=artifact,
                ),
                patch(
                    "evolution_sim.mind.recurrent_scale_execution."
                    "validate_recurrent_evaluation_report"
                ),
                self.assertRaisesRegex(ValueError, "summary detached"),
            ):
                load_scale_arm_reports(
                    self.preregistration,
                    output_root=root,
                    runtime_provenance_path=runtime_path,
                )

    def test_completed_evaluation_evidence_rejects_byte_tamper_and_detached_summary(
        self,
    ) -> None:
        learner_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        for failure_mode in (
            "tampered bytes",
            "detached summary",
            "wrong artifact identity",
        ):
            with (
                self.subTest(failure_mode=failure_mode),
                tempfile.TemporaryDirectory() as temporary,
            ):
                run_directory = Path(temporary)
                report = self._report(learner_seed=learner_seed, arm=BASE_ARM)
                modes = report["evaluations"]["modes"]  # type: ignore[index]
                for mode, evidence in modes.items():  # type: ignore[union-attr]
                    evaluation = self._evaluation_from_summary(
                        evidence["candidate_outcome_summary"]
                    )
                    in_memory_path = (
                        run_directory / "evaluations" / f"in-memory-{mode}.json"
                    )
                    artifact_path = (
                        run_directory / "evaluations" / f"artifact-cpu-{mode}.json"
                    )
                    write_atomic_json(in_memory_path, evaluation)
                    artifact_evaluation = copy.deepcopy(evaluation)
                    artifact_evaluation["artifact"] = {
                        "path": str(run_directory / "frozen-policy.json"),
                        "artifact_sha256": report["artifact"][  # type: ignore[index]
                            "artifact_sha256"
                        ],
                    }
                    write_atomic_json(artifact_path, artifact_evaluation)
                    evidence["in_memory_report"] = self._actual_file_reference(
                        in_memory_path
                    )
                    evidence["artifact_cpu_report"] = self._actual_file_reference(
                        artifact_path
                    )

                if failure_mode == "tampered bytes":
                    path = (
                        run_directory
                        / "evaluations"
                        / (f"artifact-cpu-{PUBLIC_RECURRENT_ARGMAX_SELECTION}.json")
                    )
                    path.write_bytes(path.read_bytes() + b"\n")
                    expected_error = "evidence file drifted"
                else:
                    if failure_mode == "detached summary":
                        modes[PUBLIC_RECURRENT_ARGMAX_SELECTION][  # type: ignore[index]
                            "candidate_outcome_summary"
                        ]["contexts"][1]["runs"][0]["terminal_alive"] += 1
                        expected_error = "summary detached"
                    else:
                        mode = PUBLIC_RECURRENT_ARGMAX_SELECTION
                        path = (
                            run_directory / "evaluations" / f"artifact-cpu-{mode}.json"
                        )
                        evaluation = load_strict_json(path)
                        evaluation["artifact"]["artifact_sha256"] = "d" * 64  # type: ignore[index]
                        write_atomic_json(path, evaluation)
                        modes[mode]["artifact_cpu_report"] = (  # type: ignore[index]
                            self._actual_file_reference(path)
                        )
                        expected_error = "different frozen policy"
                self._resign(report)
                validate_recurrent_scale_arm_report(
                    report,
                    preregistration=self.preregistration,
                )

                with (
                    patch(
                        "evolution_sim.mind.recurrent_scale_execution."
                        "validate_recurrent_evaluation_report"
                    ),
                    self.assertRaisesRegex(ValueError, expected_error),
                ):
                    _verify_completed_evaluation_evidence(
                        report,
                        run_directory=run_directory,
                    )

    def _report(self, *, learner_seed: int, arm: str) -> dict[str, object]:
        run_id = recurrent_scale_arm_run_id(learner_seed=learner_seed, arm=arm)
        alive = 1 if arm == EXACT_ARM else 0
        births = 2 if arm == EXACT_ARM else 1
        model, ppo, counterfactual, schedule = build_scale_run_components(
            self.preregistration,
            learner_seed=learner_seed,
            arm=arm,
            counterfactual_workers=1,
        )
        configuration = _scale_run_contract(
            preregistration=self.preregistration,
            learner_seed=learner_seed,
            arm=arm,
            model_config=model,
            ppo_config=ppo,
            counterfactual_config=counterfactual,
            schedule=schedule,
        )
        sampling_stream_id = _scale_sampling_stream_id(
            self.preregistration["exact_digest"],
            learner_seed=learner_seed,
        )
        mode_payloads = {
            mode: {
                "exact_cpu_outcome_replay_match": True,
                "candidate_outcome_summary": self._summary(
                    mode=mode,
                    alive=alive,
                    births=births,
                    sampling_stream_id=sampling_stream_id,
                ),
                "in_memory_report": self._file_reference(
                    run_id=run_id,
                    label=f"in-memory-{mode}",
                ),
                "artifact_cpu_report": self._file_reference(
                    run_id=run_id,
                    label=f"artifact-cpu-{mode}",
                ),
            }
            for mode in (
                PUBLIC_RECURRENT_ARGMAX_SELECTION,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            )
        }
        report: dict[str, object] = {
            "schema_version": RECURRENT_SCALE_ARM_REPORT_SCHEMA_VERSION,
            "run_id": run_id,
            "preregistration_digest": self.preregistration["exact_digest"],
            "source": {
                "commit": self.preregistration["source"]["commit"],  # type: ignore[index]
                "manifest_sha256": self.preregistration["source"][  # type: ignore[index]
                    "manifest_sha256"
                ],
                "clean_tree_verified_before_training": True,
                "source_stable_through_completion": True,
            },
            "runtime_provenance": {
                "payload": copy.deepcopy(self.runtime_provenance),
                "file": dict(self.runtime_provenance_reference),
            },
            "learner_seed": learner_seed,
            "learner_seed_role": "scale_learner",
            "arm": arm,
            "configuration": configuration,
            "training": {
                "completed_updates": 16,
                "worlds": 256,
                "agent_transitions": 1024,
                "update_record_digests": [
                    stable_payload_digest({"run_id": run_id, "update": index})
                    for index in range(16)
                ],
                "update_journals": [
                    self._file_reference(
                        run_id=run_id,
                        label=f"update-{index:04d}",
                    )
                    for index in range(16)
                ],
                "environment_seed_provenance": _scale_training_seed_provenance(
                    schedule
                ),
                "treatment_delivery": {
                    "treatment_expected": arm != BASE_ARM,
                    "attempted_update_count": 0 if arm == BASE_ARM else 16,
                    "accepted_update_count": 0 if arm == BASE_ARM else 16,
                    "parameter_delta_l2_sum": 0.0 if arm == BASE_ARM else 1.0,
                    "all_transactions_within_kl_bounds": True,
                    "meets_preregistered_delivery_floor": True,
                },
                "checkpoint": self._file_reference(
                    run_id=run_id,
                    label="training-checkpoint",
                ),
            },
            "artifact": {
                "path": f"/synthetic/{run_id}/frozen-policy.json",
                "artifact_sha256": stable_payload_digest(
                    {"run_id": run_id, "artifact": "frozen-policy"}
                ),
                "file": self._file_reference(
                    run_id=run_id,
                    label="frozen-policy",
                ),
                "runtime_provenance_digest": self.runtime_provenance["exact_digest"],
                "runtime_policy_eligible": False,
                "full_world_replay_manifest": {
                    "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
                    "manifest_sha256": stable_payload_digest(
                        {"run_id": run_id, "manifest": "full-world"}
                    ),
                    "replay_engine_contract_sha256": stable_payload_digest(
                        {"contract": "synthetic-scale-evaluation"}
                    ),
                    "environment_seed_registry_sha256": self.preregistration[
                        "seed_contract"
                    ]["registry_sha256"],  # type: ignore[index]
                    "environment_seed_roles": ["scale_selection"],
                    "scenario_names": ["broad", "carrion_only"],
                    "tick_horizons": [120],
                    "world_count": 80,
                    "replay_verified_world_count": 80,
                    "policy_sampling_stream_count": 5,
                    "all_replays_exact": True,
                    "verification_runner": (
                        RECURRENT_SCALE_FULL_WORLD_VERIFICATION_RUNNER
                    ),
                    "verification_runner_sha256": "c" * 64,
                },
            },
            "evaluations": {
                "sampling_stream_id": sampling_stream_id,
                "selection_seed_plan_sha256": (
                    recurrent_scale_selection_seed_plan().digest
                ),
                "modes": mode_payloads,
            },
            "elapsed_seconds": 1.0,
            "lifecycle": {
                "development_only": True,
                "campaign_slice_consumed": True,
                "runtime_artifact_created": False,
                "runtime_action_selection_changed": False,
                "runtime_integration_authorized": False,
                "promotion_authorized": False,
                "validation_seeds_accessed": False,
                "lockbox_seeds_accessed": False,
                "gate_relaxation_authorized": False,
            },
        }
        report["exact_digest"] = stable_payload_digest(report)
        return report

    def _summary(
        self,
        *,
        mode: str,
        alive: int,
        births: int,
        sampling_stream_id: str,
    ) -> dict[str, object]:
        stream_seeds = (
            (None,)
            if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
            else _scale_policy_sampling_seeds(sampling_stream_id)
        )
        broad_runs = []
        carrion_runs = []
        selection_seeds = tuple(recurrent_scale_selection_seed_plan().broad_seeds)
        for seed in selection_seeds:
            for stream_seed in stream_seeds:
                common = {
                    "seed": seed,
                    "policy_sampling_seed": stream_seed,
                    "terminal_alive": alive,
                    "births": births,
                    "requested_action_counts": {
                        "stay": 2,
                        "move_n": 2,
                        "move_s": 2,
                        "eat": 2,
                        "drink": 2,
                    },
                    "dominant_requested_action_share": 0.2,
                    "heuristic_action_source_count": 0,
                    "unsupported_requested_action_count": 0,
                }
                broad_runs.append(
                    {
                        "context": "broad_default",
                        "behavior_digest": stable_payload_digest(
                            {
                                "context": "broad_default",
                                **common,
                            }
                        ),
                        **common,
                    }
                )
                carrion_runs.append(
                    {
                        "context": "fixture:carrion_only",
                        "behavior_digest": stable_payload_digest(
                            {
                                "context": "fixture:carrion_only",
                                **common,
                            }
                        ),
                        **common,
                    }
                )
        broad_linear_runs = [
            {
                "context": "broad_default",
                "seed": seed,
                "policy_sampling_seed": None,
                "terminal_alive": 0,
                "births": 1,
                "behavior_digest": stable_payload_digest(
                    {"context": "broad_default", "linear_seed": seed}
                ),
            }
            for seed in selection_seeds
        ]
        carrion_linear_runs = [
            {
                "context": "fixture:carrion_only",
                "seed": seed,
                "policy_sampling_seed": None,
                "terminal_alive": 0,
                "births": 1,
                "behavior_digest": stable_payload_digest(
                    {"context": "fixture:carrion_only", "linear_seed": seed}
                ),
            }
            for seed in selection_seeds
        ]
        return {
            "evaluation_contract": {
                "ticks": 120,
                "world_horizon_policy": "exact_configured_120_tick_horizon",
                "candidate_action_selection": mode,
                "candidate_sampling_seeds": [
                    seed for seed in stream_seeds if seed is not None
                ],
                "candidate_sampling_seed_count": len(stream_seeds),
                "candidate_sampling_stream_id": sampling_stream_id,
                "candidate_sampling_stream_id_source": (
                    "caller_supplied_arm_independent_identifier"
                ),
                "candidate_sampling_seed_contract": (
                    "none_argmax"
                    if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
                    else (
                        "sha256_namespace_sampling_stream_id_and_replicate_index_"
                        "first_63_bits"
                    )
                ),
                "candidate_recurrent_state": "one_hidden_state_per_agent",
                "candidate_policy_inputs": [
                    "current_public_ecological_observation",
                    "current_public_action_mask",
                    "same_agent_previous_public_outcome",
                    "same_agent_recurrent_state",
                ],
                "candidate_forbidden_inputs": [
                    "world_seed",
                    "fixture_name",
                    "private_world_state",
                    "evaluation_split_role",
                ],
                "candidate_factory_receives_seed_or_fixture": False,
                "candidate_replay_verification": "every_run_exact_digest_repeat",
                "strict_zero_unsupported_requested_actions": True,
                "strict_zero_heuristic_candidate_actions": True,
            },
            "seed_plan_digest": recurrent_scale_selection_seed_plan().digest,
            "contexts": [
                {
                    "context": "broad_default",
                    "runs": broad_runs,
                    "linear_runs": broad_linear_runs,
                },
                {
                    "context": "fixture:carrion_only",
                    "runs": carrion_runs,
                    "linear_runs": carrion_linear_runs,
                },
            ],
        }

    @staticmethod
    def _evaluation_from_summary(
        summary: dict[str, object],
    ) -> dict[str, object]:
        contexts = summary["contexts"]
        assert isinstance(contexts, list)

        def expand(context: object) -> dict[str, object]:
            assert isinstance(context, dict)
            return {
                "policies": {
                    "public_recurrent": {
                        "runs": copy.deepcopy(context["runs"]),
                    },
                    "mind_v3_linear": {
                        "runs": copy.deepcopy(context["linear_runs"]),
                    },
                }
            }

        return {
            "evaluation_contract": copy.deepcopy(summary["evaluation_contract"]),
            "seed_plan": {"digest": summary["seed_plan_digest"]},
            "broad": expand(contexts[0]),
            "fixtures": [expand(context) for context in contexts[1:]],
            "replay_verification": {
                "all_passed": True,
                "checks": [{"synthetic": True}],
            },
        }

    @staticmethod
    def _actual_file_reference(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_length": len(payload),
        }

    @staticmethod
    def _file_reference(*, run_id: str, label: str) -> dict[str, object]:
        return {
            "path": f"/synthetic/{run_id}/{label}.json",
            "sha256": stable_payload_digest({"run_id": run_id, "label": label}),
            "byte_length": 1024,
        }

    @staticmethod
    def _resign(report: dict[str, object]) -> None:
        report.pop("exact_digest", None)
        report["exact_digest"] = stable_payload_digest(report)


if __name__ == "__main__":
    unittest.main()
