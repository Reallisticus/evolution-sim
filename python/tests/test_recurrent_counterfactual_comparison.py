from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    import evolution_sim.mind.recurrent_counterfactual_comparison as comparison_module
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.env.runtime.trajectory import REWARD_COMPONENT_BOUNDS
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_actor_critic import (
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
        RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED,
        RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS,
        CounterfactualHorizonScalarization,
        RecurrentCounterfactualAuxiliaryConfig,
        RecurrentCounterfactualAuxiliaryStepConfig,
        recurrent_counterfactual_auxiliary_bundle_digest,
    )
    from evolution_sim.mind.recurrent_counterfactual_collection import (
        RecurrentCounterfactualCollectionConfig,
        RecurrentCounterfactualCollectionTask,
        collect_recurrent_counterfactual_bundles,
    )
    from evolution_sim.mind.recurrent_counterfactual_comparison import (
        BASE_ARM,
        CAUSAL_CANARY_POLICY_VERSION,
        CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION,
        CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION,
        CAUSAL_CANARY_SCHEMA_VERSION,
        EXACT_ARM,
        RECURRENT_COUNTERFACTUAL_ANALYZER_PROVENANCE_SCHEMA_VERSION,
        RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION,
        SHUFFLED_ARM,
        UNPINNED_LABEL_PREFIX,
        RecurrentCounterfactualComparisonError,
        _LIFECYCLE_FLAGS,
        _ANALYZER_RUNTIME_PATH,
        _causal_delta_payload,
        _current_runtime_source_manifest,
        analyze_recurrent_counterfactual_three_arm,
        main as comparison_main,
    )
    from evolution_sim.mind.recurrent_evaluation import (
        RECURRENT_EVALUATION_EXECUTION_SCHEMA_VERSION,
        RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION,
        RECURRENT_EVALUATION_SCHEMA_VERSION,
        RecurrentEvaluationSeedPlan,
        _aggregate_runs,
        _candidate_sampling_seeds,
        _candidate_sampling_analysis,
        _paired_deltas,
    )
    from evolution_sim.mind.recurrent_experiment import (
        RECURRENT_EXPERIMENT_CONTRACT_VERSION,
    )
    from evolution_sim.mind.recurrent_policy import (
        PUBLIC_RECURRENT_ARGMAX_SELECTION,
        PUBLIC_RECURRENT_POLICY_ID,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
    )
    from evolution_sim.mind.recurrent_rollout import (
        derive_recurrent_policy_sampling_seed,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        RECURRENT_SEED_REGISTRY,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentCounterfactualComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=4, hidden_size=4),
            initialization_seed=47,
        )
        task = RecurrentCounterfactualCollectionTask(
            task_id="comparison-test-counterfactual",
            seed_role="curriculum",
            scenario="carrion_only",
            environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
            branch_tick_candidates=(0,),
            source_policy_sampling_identity="comparison-test-source-policy",
            branch_selection_identity="comparison-test-branch-selection",
        )
        cls.collection = collect_recurrent_counterfactual_bundles(
            model,
            (task,),
            artifact_digest="d" * 64,
            config=RecurrentCounterfactualCollectionConfig(
                horizons=(1, 2),
                gamma=0.99,
            ),
        )

    def setUp(self) -> None:
        self.base = _causal_report(BASE_ARM, collection=self.collection)
        self.exact = _causal_report(EXACT_ARM, collection=self.collection)
        self.shuffled = _causal_report(SHUFFLED_ARM, collection=self.collection)

    def test_valid_three_arm_analysis_is_stable_and_stratified(self) -> None:
        first = analyze_recurrent_counterfactual_three_arm(
            self.base,
            self.exact,
            self.shuffled,
        )
        second = analyze_recurrent_counterfactual_three_arm(
            self.base,
            self.exact,
            self.shuffled,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["schema_version"],
            RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION,
        )
        observed_digest = first["exact_digest"]
        digest_payload = copy.deepcopy(first)
        digest_payload.pop("exact_digest")
        self.assertEqual(observed_digest, stable_payload_digest(digest_payload))
        self.assertFalse(first["comparison_contract"]["inferential_claim_authorized"])
        self.assertFalse(first["comparison_contract"]["p_values_emitted"])

        sampled = first["selection_modes"][PUBLIC_RECURRENT_SAMPLED_SELECTION]
        self.assertEqual(sampled["run_identity_count"], 8)
        exact_minus_base = sampled["exact_minus_base"]
        self.assertEqual(exact_minus_base["raw_run_cell_count"], 8)
        self.assertEqual(len(exact_minus_base["environment_stratified"]), 4)
        self.assertEqual(
            len(exact_minus_base["policy_sampling_stream_stratified"]),
            4,
        )
        self.assertEqual(
            exact_minus_base["overall_mean_contrast"]["terminal_alive"],
            2.0,
        )
        self.assertEqual(
            sampled["exact_minus_shuffled"]["overall_mean_contrast"]["terminal_alive"],
            1.0,
        )
        self.assertTrue(
            first["evidence"]["counterfactual_bundle_one_use_semantics_verified"]
        )
        provenance = first["analyzer_provenance"]
        self.assertEqual(
            provenance["schema_version"],
            RECURRENT_COUNTERFACTUAL_ANALYZER_PROVENANCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            provenance["comparison_schema_version"],
            RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION,
        )
        self.assertEqual(provenance["module_path"], _ANALYZER_RUNTIME_PATH)
        self.assertEqual(
            provenance["runtime_source_changed_paths"],
            [_ANALYZER_RUNTIME_PATH],
        )
        self.assertTrue(provenance["only_analyzer_module_changed_since_training"])
        self.assertTrue(provenance["post_training_analyzer_repair_transparent"])
        self.assertFalse(provenance["non_analyzer_runtime_source_drift_detected"])
        digest_contracts = first["comparison_contract"]["end_state_digest_contracts"]
        self.assertFalse(digest_contracts["digest_values_directly_comparable"])
        self.assertFalse(
            digest_contracts["same_state_cross_digest_link_present_in_input_reports"]
        )
        self.assertNotEqual(
            self.base["configuration"]["counterfactual_auxiliary"][
                "resolved_configuration"
            ]["branch_tick_candidates"],
            self.exact["configuration"]["counterfactual_auxiliary"][
                "resolved_configuration"
            ]["branch_tick_candidates"],
        )
        for report in (self.exact, self.shuffled):
            self.assertNotEqual(
                report["training"]["updates"][-1]["counterfactual_auxiliary"][
                    "final_model_state_sha256"
                ],
                report["model_states"]["after"]["state_sha256"],
            )

    def test_cli_broad_container_key_is_distinct_from_raw_run_context(self) -> None:
        for report in (self.base, self.exact, self.shuffled):
            for mode in (
                PUBLIC_RECURRENT_ARGMAX_SELECTION,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            ):
                controls = report["paired_outcome_deltas"][mode][
                    "candidate_minus_controls"
                ]
                self.assertIn("broad", controls)
                self.assertNotIn("broad_default", controls)
                self.assertEqual(
                    {
                        run["context"]
                        for run in report["paired_outcome_deltas"][mode]["runs"]
                    },
                    {"broad_default", "fixture:carrion_only"},
                )

        observed = analyze_recurrent_counterfactual_three_arm(
            self.base,
            self.exact,
            self.shuffled,
        )
        self.assertEqual(
            observed["selection_modes"][PUBLIC_RECURRENT_ARGMAX_SELECTION][
                "run_identity_count"
            ],
            4,
        )

    def test_non_analyzer_current_runtime_source_drift_is_rejected(self) -> None:
        current_manifest = _current_runtime_source_manifest()
        drifted_files = copy.deepcopy(current_manifest["files"])
        original_package_digest = drifted_files["package.json"]
        drifted_files["package.json"] = (
            "0" * 64 if original_package_digest != "0" * 64 else "1" * 64
        )
        drifted_manifest = {
            **current_manifest,
            "files": drifted_files,
            "aggregate_sha256": stable_payload_digest(drifted_files),
        }

        with mock.patch.object(
            comparison_module,
            "_current_runtime_source_manifest",
            return_value=drifted_manifest,
        ):
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "beyond the comparison analyzer",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.base,
                    self.exact,
                    self.shuffled,
                )

    def test_exact_training_and_analyzer_source_match_is_also_supported(self) -> None:
        training_manifest = copy.deepcopy(
            self.base["source"]["source_file_hash_manifest"]
        )
        with mock.patch.object(
            comparison_module,
            "_current_runtime_source_manifest",
            return_value=training_manifest,
        ):
            observed = analyze_recurrent_counterfactual_three_arm(
                self.base,
                self.exact,
                self.shuffled,
            )

        provenance = observed["analyzer_provenance"]
        self.assertTrue(provenance["runtime_source_exactly_matches_training"])
        self.assertFalse(provenance["only_analyzer_module_changed_since_training"])
        self.assertFalse(provenance["post_training_analyzer_repair_transparent"])
        self.assertEqual(provenance["runtime_source_changed_paths"], [])

    def test_float32_kl_roundoff_is_tolerated_but_substantive_negative_is_not(
        self,
    ) -> None:
        roundoff = copy.deepcopy(self.exact)
        roundoff["training"]["updates"][0]["counterfactual_auxiliary"][
            "behavior_kl"
        ] = -3.6e-8
        _reseal(roundoff)
        analyze_recurrent_counterfactual_three_arm(
            self.base,
            roundoff,
            self.shuffled,
        )

        invalid = copy.deepcopy(self.exact)
        invalid["training"]["updates"][0]["counterfactual_auxiliary"][
            "behavior_kl"
        ] = -1.0e-4
        _reseal(invalid)
        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "below its numerical non-negative tolerance",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                invalid,
                self.shuffled,
            )

    def test_mapping_and_json_path_inputs_produce_the_same_analysis(self) -> None:
        expected = analyze_recurrent_counterfactual_three_arm(
            self.base,
            self.exact,
            self.shuffled,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, report in (
                ("base", self.base),
                ("exact", self.exact),
                ("shuffled", self.shuffled),
            ):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(report, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                paths.append(path)
            observed = analyze_recurrent_counterfactual_three_arm(*paths)
        self.assertEqual(observed, expected)

    def test_cli_writes_an_exclusively_created_sealed_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = []
            for option, name, report in (
                ("--base-report", "base", self.base),
                ("--exact-report", "exact", self.exact),
                ("--shuffled-report", "shuffled", self.shuffled),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                arguments.extend((option, str(path)))
            output = root / "comparison.json"
            arguments.extend(("--output", str(output)))

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = comparison_main(arguments)

            self.assertEqual(exit_code, 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            claimed = written.pop("exact_digest")
            self.assertEqual(claimed, stable_payload_digest(written))
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["exact_digest"], claimed)
            self.assertEqual(summary["output"], str(output))

    def test_arm_schema_and_evaluation_mode_mismatches_are_rejected(self) -> None:
        with self.subTest("arm"):
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "arm role",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.exact,
                    self.base,
                    self.shuffled,
                )

        with self.subTest("schema"):
            tampered = copy.deepcopy(self.exact)
            tampered["schema_version"] = (
                "mind_v3_public_recurrent_ippo_development_causal_canary_v3"
            )
            _reseal(tampered)
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "schema must be v4",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.base,
                    tampered,
                    self.shuffled,
                )

        with self.subTest("evaluation mode"):
            tampered = copy.deepcopy(self.exact)
            evaluation = tampered["evaluations"]["before"][
                PUBLIC_RECURRENT_ARGMAX_SELECTION
            ]
            evaluation["evaluation_contract"]["candidate_action_selection"] = (
                PUBLIC_RECURRENT_SAMPLED_SELECTION
            )
            _reseal(tampered)
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "public v4 evaluator contract",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.base,
                    tampered,
                    self.shuffled,
                )

    def test_unsealed_tamper_is_rejected_by_top_exact_digest(self) -> None:
        tampered = copy.deepcopy(self.exact)
        tampered["training"]["total_transitions"] += 1

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "exact digest",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                tampered,
                self.shuffled,
            )

    def test_resealed_source_head_tamper_is_rejected_by_three_arm_match(self) -> None:
        tampered = copy.deepcopy(self.exact)
        tampered["source"]["git_head_observed"] = "e" * 40
        _reseal(tampered)

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "source head",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                tampered,
                self.shuffled,
            )

    def test_resealed_schedule_and_stream_mismatches_are_rejected(self) -> None:
        with self.subTest("active counterfactual settings"):
            tampered = copy.deepcopy(self.shuffled)
            tampered["configuration"]["counterfactual_auxiliary"][
                "resolved_configuration"
            ]["bundles_per_update"] = 2
            tampered["evaluation_preregistration"]["run_config"] = copy.deepcopy(
                tampered["configuration"]
            )
            _reseal_preregistration_and_labels(tampered)
            _reseal(tampered)
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "bundles_per_update|active counterfactual configurations differ",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.base,
                    self.exact,
                    tampered,
                )

        with self.subTest("schedule"):
            tampered = copy.deepcopy(self.exact)
            schedule = tampered["configuration"]["training_schedule"]
            schedule[0][0]["rollout_ticks"] = 9
            tampered["configuration"]["training_schedule_sha256"] = (
                stable_payload_digest(schedule)
            )
            tampered["training"]["updates"][0]["tasks"] = copy.deepcopy(schedule[0])
            tampered["evaluation_preregistration"]["run_config"] = copy.deepcopy(
                tampered["configuration"]
            )
            _reseal_preregistration_and_labels(tampered)
            _reseal(tampered)
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "configuration differs|task/sampling schedule differs",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.base,
                    tampered,
                    self.shuffled,
                )

        with self.subTest("evaluation stream"):
            tampered = copy.deepcopy(self.exact)
            tampered["configuration"]["evaluation_sampling_stream"]["id"] = (
                "matched-recurrent-evaluation:wrong-stream"
            )
            tampered["evaluation_preregistration"]["run_config"] = copy.deepcopy(
                tampered["configuration"]
            )
            _reseal_preregistration_and_labels(tampered)
            _reseal(tampered)
            with self.assertRaisesRegex(
                RecurrentCounterfactualComparisonError,
                "configuration differs|evaluation stream",
            ):
                analyze_recurrent_counterfactual_three_arm(
                    self.base,
                    tampered,
                    self.shuffled,
                )

    def test_resealed_auxiliary_retry_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.exact)
        tampered["training"]["updates"][0]["counterfactual_auxiliary"][
            "retry_authorized"
        ] = True
        _reseal(tampered)

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "one-use",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                tampered,
                self.shuffled,
            )

    def test_resealed_nested_evaluation_tamper_uses_public_v4_validator(self) -> None:
        tampered = copy.deepcopy(self.exact)
        candidate = tampered["evaluations"]["after"][
            PUBLIC_RECURRENT_SAMPLED_SELECTION
        ]["broad"]["policies"]["public_recurrent"]
        candidate["aggregate"]["terminal_alive_agent_total"] += 1
        _reseal(tampered)

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "public v4 evaluator contract",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                tampered,
                self.shuffled,
            )

    def test_duplicate_candidate_run_identity_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.exact)
        evaluation = tampered["evaluations"]["after"][
            PUBLIC_RECURRENT_SAMPLED_SELECTION
        ]
        runs = evaluation["broad"]["policies"]["public_recurrent"]["runs"]
        runs[1] = copy.deepcopy(runs[0])
        _reseal(tampered)

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "public v4 evaluator contract|duplicated",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                tampered,
                self.shuffled,
            )

    def test_nested_valid_control_drift_is_rejected_before_comparison(self) -> None:
        tampered = copy.deepcopy(self.exact)
        evaluation = tampered["evaluations"]["after"][
            PUBLIC_RECURRENT_SAMPLED_SELECTION
        ]
        context = evaluation["broad"]
        policies = context["policies"]
        candidate_runs = policies["public_recurrent"]["runs"]
        control_runs = policies["mind_v3_linear"]["runs"]
        control_runs[0]["terminal_alive"] += 1
        policies["mind_v3_linear"]["aggregate"] = _aggregate_runs(control_runs)
        by_seed = {run["seed"]: run for run in control_runs}
        context["paired_deltas"]["candidate_minus_mind_v3_linear"] = _paired_deltas(
            candidate_runs,
            [by_seed[run["seed"]] for run in candidate_runs],
        )
        _reseal(tampered)

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "mind_v3_linear drifted",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                tampered,
                self.shuffled,
            )

    def test_resealed_label_shuffle_role_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.shuffled)
        counterfactual = tampered["configuration"]["counterfactual_auxiliary"]
        counterfactual["exact_branch_labels_consumed"] = True
        prereg = tampered["evaluation_preregistration"]
        prereg["run_config"] = copy.deepcopy(tampered["configuration"])
        _reseal_preregistration_and_labels(tampered)
        _reseal(tampered)

        with self.assertRaisesRegex(
            RecurrentCounterfactualComparisonError,
            "arm role",
        ):
            analyze_recurrent_counterfactual_three_arm(
                self.base,
                self.exact,
                tampered,
            )


def _cli_structural_causal_delta_payload(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    """Independently preserve the producer's broad-container field name."""

    payload = _causal_delta_payload(before, after)
    controls = payload["candidate_minus_controls"]
    if "broad_default" in controls:
        controls["broad"] = controls.pop("broad_default")
    return payload


def _causal_report(
    arm: str,
    *,
    collection: object,
) -> dict[str, object]:
    learner_seed = 1777057840
    evaluation_seed_plan = RecurrentEvaluationSeedPlan(
        broad_seeds=(101, 103),
        fixture_seeds=(101, 103),
        excluded_training_seeds=(1,),
    )
    task_identity = (
        "mind_public_recurrent_ippo_training_schedule_task_v1|"
        "update=0000|world=000000|scenario=broad"
    )
    schedule = [
        [
            {
                "task_id": "update-0000-world-000000-broad-seed-1",
                "scenario": "broad",
                "environment_seed": 1,
                "rollout_ticks": 8,
                "policy_sampling_identity": task_identity,
                "policy_sampling_seed": derive_recurrent_policy_sampling_seed(
                    task_identity=task_identity
                ),
            }
        ]
    ]
    counterfactual = _counterfactual_preregistration(arm)
    runtime = {
        "schema_version": CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION,
        "captured_before_model_initialization": True,
        "python_version": "3.14.3",
        "python_implementation": "CPython",
        "torch_version": "2.11.0",
        "numpy_version": "2.4.4",
        "platform": {
            "system": "Darwin",
            "release": "test",
            "machine": "arm64",
            "platform_string": "test-platform",
        },
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "resolved_device_type": "cpu",
        "resolved_device_name": "cpu",
        "cuda_available": False,
        "cuda_runtime_version": None,
        "mps_available": True,
        "torch_default_dtype": "torch.float32",
        "torch_deterministic_algorithms_enabled_before_runner": True,
        "torch_deterministic_algorithms_enabled_after_runner": True,
    }
    model_config = {
        "encoder_size": 4,
        "hidden_size": 4,
        "recurrent_layers": 1,
    }
    ppo_config = {
        "learning_rate": 0.0003,
        "learner_seed": learner_seed,
        "feed_forward_history_ablation": False,
        "world_balanced_loss": True,
    }
    stream = {
        "id": "matched-recurrent-evaluation:test-three-arm",
        "contract": {
            "schema_version": "mind_v3_recurrent_matched_evaluation_sampling_stream_v1",
            "seed_plan_digest": evaluation_seed_plan.digest,
            "fixtures": ["carrion_only"],
            "candidate_sampling_seed_count": 2,
            "excluded_fields": [
                "model_architecture",
                "model_parameters",
                "learner_seed",
                "feed_forward_history_ablation",
                "candidate_policy_digest",
            ],
        },
        "arm_independent": True,
    }
    configuration = {
        "updates": 1,
        "worlds_per_update": 1,
        "rollout_ticks": 8,
        "training_scenarios": ["broad"],
        "evaluation_fixtures": ["carrion_only"],
        "evaluation_horizon_ticks": 120,
        "candidate_sampling_seed_count": 2,
        "evaluation_sampling_stream": stream,
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "rollout_workers": 1,
        "evaluation_workers": 1,
        "model": model_config,
        "ppo": ppo_config,
        "tbptt_contract": {
            "full_sequence_tbptt": True,
            "requested_tbptt_steps": 8,
            "resolved_tbptt_steps": 8,
            "maximum_agent_sequence_length": 8,
            "full_sequence_invariant_satisfied": True,
        },
        "feed_forward_history_ablation": False,
        "counterfactual_auxiliary": counterfactual,
        "runtime_reproducibility": runtime,
        "training_schedule": schedule,
        "training_schedule_sha256": stable_payload_digest(schedule),
    }
    seed_roles = {
        "learner": {"seed": learner_seed, "role": "learner_development"},
        "development_selection": {
            "seeds": [101, 103],
            "offset": 0,
            "accessed": True,
        },
        "validation": {"accessed": False, "seeds_materialized_by_canary": False},
        "lockbox": {"accessed": False, "seeds_materialized_by_canary": False},
    }
    current_source_manifest = _current_runtime_source_manifest()
    files = copy.deepcopy(current_source_manifest["files"])
    current_analyzer_digest = files[_ANALYZER_RUNTIME_PATH]
    files[_ANALYZER_RUNTIME_PATH] = (
        "0" * 64 if current_analyzer_digest != "0" * 64 else "1" * 64
    )
    source_manifest = {
        "hash_algorithm": "sha256",
        "path_contract": "repository_relative_sorted_runtime_python_plus_package",
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": stable_payload_digest(files),
    }
    preregistration: dict[str, object] = {
        "schema_version": CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION,
        "created_before_model_initialization": True,
        "action_selection_modes_in_order": [
            PUBLIC_RECURRENT_ARGMAX_SELECTION,
            PUBLIC_RECURRENT_SAMPLED_SELECTION,
        ],
        "before_after_sampling_stream_identical": True,
        "candidate_controls": ["mind_v3_linear", "masked_random"],
        "run_config": configuration,
        "seed_registry_sha256": "c" * 64,
        "seed_roles": seed_roles,
        "source_manifest_aggregate_sha256": source_manifest["aggregate_sha256"],
    }
    preregistration_digest = stable_payload_digest(preregistration)
    label = UNPINNED_LABEL_PREFIX + preregistration_digest
    preregistration["exact_digest"] = preregistration_digest
    preregistration["synthetic_noncandidate_digest_label"] = label

    evaluations = {
        "before": {
            mode: _evaluation_report(
                arm=arm,
                phase="before",
                mode=mode,
                label=label,
                stream_id=stream["id"],
                seed_plan=evaluation_seed_plan,
            )
            for mode in (
                PUBLIC_RECURRENT_ARGMAX_SELECTION,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            )
        },
        "after": {
            mode: _evaluation_report(
                arm=arm,
                phase="after",
                mode=mode,
                label=label,
                stream_id=stream["id"],
                seed_plan=evaluation_seed_plan,
            )
            for mode in (
                PUBLIC_RECURRENT_ARGMAX_SELECTION,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            )
        },
    }
    training, after_state = _training_payload(
        arm,
        schedule=schedule,
        learner_seed=learner_seed,
        model_config=model_config,
        ppo_config=ppo_config,
        counterfactual=counterfactual,
        collection=collection,
    )
    report: dict[str, object] = {
        "schema_version": CAUSAL_CANARY_SCHEMA_VERSION,
        "policy": CAUSAL_CANARY_POLICY_VERSION,
        "development_run": True,
        "development_experiment": True,
        "noncandidate_development_canary": True,
        "source_dirty": True,
        "source_unpinned": True,
        "source_pinned": False,
        "artifact_output_refused_by_contract": True,
        "artifact_output_requested": False,
        "artifact_path": None,
        "source": {
            "git_head_observed": "b" * 40,
            "git_state_available": True,
            "source_tree_dirty": True,
            "git_status_porcelain_sha256": "7" * 64,
            "git_status_entry_count": 3,
            "source_pinned": False,
            "source_commit_explicitly_pinned": False,
            "unpinned": True,
            "noncandidate": True,
            "source_file_hash_manifest": source_manifest,
            "source_stable_during_run": True,
        },
        "runtime_reproducibility": runtime,
        "seed_registry_sha256": "c" * 64,
        "seed_roles": seed_roles,
        "configuration": configuration,
        "evaluation_preregistration": preregistration,
        "model_states": {
            "before": _state("1" * 64),
            "after": _state(after_state),
            "changed": True,
            "same_before_snapshot_used_for_both_selection_modes": True,
            "same_after_model_used_for_both_selection_modes": True,
        },
        "elapsed_seconds": {"training": 1.0, "total": 2.0},
        "training": training,
        "evaluations": evaluations,
        "paired_outcome_deltas": {
            mode: _cli_structural_causal_delta_payload(
                evaluations["before"][mode],
                evaluations["after"][mode],
            )
            for mode in (
                PUBLIC_RECURRENT_ARGMAX_SELECTION,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            )
        },
        "lifecycle": {flag: False for flag in _LIFECYCLE_FLAGS},
        "campaign_training_slice_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
        "non_promoted": True,
    }
    _reseal(report)
    return report


def _counterfactual_preregistration(arm: str) -> dict[str, object]:
    shuffled = arm == SHUFFLED_ARM
    exact = arm == EXACT_ARM
    scalarization = CounterfactualHorizonScalarization(
        horizon_weights=((1, 0.5), (2, 0.5)),
    )
    auxiliary = RecurrentCounterfactualAuxiliaryConfig(
        scalarization=scalarization,
        temperature=1.0,
        advantage_clip=2.0,
        policy_improvement_coefficient=1.0,
        behavior_kl_coefficient=1.0,
        value_loss_coefficient=0.0,
        target_permutation_mode=(
            RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
            if shuffled
            else RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
        ),
        target_permutation_seed=991 if shuffled else None,
    )
    step = RecurrentCounterfactualAuxiliaryStepConfig(
        learning_rate_multiplier=0.1,
        max_gradient_norm=0.5,
        mean_behavior_kl_limit=0.002,
        max_state_behavior_kl_limit=0.01,
    )
    enabled = exact or shuffled
    mode = (
        "exact_counterfactual_labels"
        if exact
        else (
            "deterministic_valid_action_label_shuffled_control"
            if shuffled
            else "disabled"
        )
    )
    return {
        "enabled": enabled,
        "mode": mode,
        "default_off": True,
        "configuration_resolved_before_model_initialization": True,
        "resolved_configuration": {
            "collection": {"horizons": (1, 2), "gamma": 0.99},
            "auxiliary": auxiliary.as_contract(),
            "step": step.as_contract(),
            "bundles_per_update": 2 if not enabled else 1,
            "branch_tick_candidates": [0, 2] if not enabled else [0],
            "workers": 2 if not enabled else 1,
        },
        "one_auxiliary_step_per_ppo_update": enabled,
        "exact_branch_labels_consumed": exact,
        "scientific_negative_control": {
            "enabled": shuffled,
            "mode": (
                RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
                if shuffled
                else RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
            ),
            "permutation_seed_explicit": shuffled,
            "permutation_seed": 991 if shuffled else None,
            "permutation_domain": "currently_valid_actions_only",
        },
        "feed_forward_history_ablation_compatible": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
    }


def _training_payload(
    arm: str,
    *,
    schedule: list[object],
    learner_seed: int,
    model_config: Mapping[str, object],
    ppo_config: Mapping[str, object],
    counterfactual: Mapping[str, object],
    collection: object,
) -> tuple[dict[str, object], str]:
    update: dict[str, object] = {
        "update_index": 0,
        "tasks": copy.deepcopy(schedule[0]),
        "rollout": {"world_count": 1, "transition_count": 10},
        "optimizer": {"update_index": 0},
        "rollout_workers_requested": 1,
        "rollout_workers_resolved": 1,
        "counterfactual_collection": None,
        "counterfactual_auxiliary": None,
    }
    experiment = None
    if arm != BASE_ARM:
        collection_payload = asdict(collection)
        row_groups = tuple(bundle.rows for bundle in collection.bundles)
        bundle_digest = recurrent_counterfactual_auxiliary_bundle_digest(
            row_groups,
            artifact_digest=collection.source_artifact_digest,
        )
        final_digest = "2" * 64 if arm == EXACT_ARM else "3" * 64
        after_fingerprint_digest = "5" * 64 if arm == EXACT_ARM else "6" * 64
        update["counterfactual_collection"] = collection_payload
        update["counterfactual_auxiliary"] = {
            "schema_version": (
                "mind_v3_recurrent_counterfactual_transactional_auxiliary_step_v1"
            ),
            "accepted": True,
            "rejection_reason": None,
            "retry_authorized": False,
            "bundle_digest": bundle_digest,
            "pre_model_state_sha256": collection.source_model_state_sha256,
            "attempted_post_model_state_sha256": final_digest,
            "final_model_state_sha256": final_digest,
            "ppo_update_index": 1,
            "auxiliary_update_count": 1,
            "group_count": len(collection.bundles),
            "row_count": sum(len(bundle.rows) for bundle in collection.bundles),
            "min_public_prefix_length": 0,
            "max_public_prefix_length": 0,
            "optimizer_step_count": 1,
            "backward_pass_count": 1,
            "learning_rate_multiplier": 0.1,
            "parameter_group_learning_rates": (0.0003,),
            "total_loss": 0.25,
            "policy_improvement_kl": 0.2,
            "behavior_kl": 0.05,
            "value_loss": 0.0,
            "gradient_norm_before_clip": 0.4,
            "gradient_norm_after_clip": 0.4,
            "parameter_delta_l2": 0.01,
            "mean_behavior_kl_old_to_post": 0.001,
            "max_state_behavior_kl_old_to_post": 0.005,
            "mean_behavior_kl_limit": 0.002,
            "max_state_behavior_kl_limit": 0.01,
            "rollback_performed": False,
            "optimizer_state_restored": False,
            "parameter_group_learning_rates_restored": True,
            "update_counters_restored": False,
            "rows_stale_after_step": True,
            "rows_exact_model_valid_after_step": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "promotion_authorized": False,
        }
        resolved = counterfactual["resolved_configuration"]
        experiment = {
            "enabled": True,
            "collection": copy.deepcopy(resolved["collection"]),
            "auxiliary": copy.deepcopy(resolved["auxiliary"]),
            "step": copy.deepcopy(resolved["step"]),
            "bundles_per_update": resolved["bundles_per_update"],
            "branch_tick_candidates": copy.deepcopy(resolved["branch_tick_candidates"]),
            "workers": resolved["workers"],
            "source_artifact_policy": (
                "ephemeral_in_memory_post_ppo_model_identity_only"
            ),
            "durable_artifact_created": False,
            "runtime_integrated": False,
            "promotion_authorized": False,
        }
    else:
        after_fingerprint_digest = "4" * 64
    return (
        {
            "contract_version": RECURRENT_EXPERIMENT_CONTRACT_VERSION,
            "learner_seed": learner_seed,
            "device": "cpu",
            "deterministic_algorithms_enabled": True,
            "model_config": copy.deepcopy(model_config),
            "ppo_config": copy.deepcopy(ppo_config),
            "rollout_execution": {
                "requested_workers": 1,
                "start_method": "spawn",
                "torch_threads_per_worker": 1,
                "ordered_merge": True,
                "sequential_default": True,
            },
            "training_scenarios": ["broad"],
            "updates": [update],
            "total_worlds": 1,
            "total_transitions": 10,
            "counterfactual_experiment": experiment,
        },
        after_fingerprint_digest,
    )


def _evaluation_report(
    *,
    arm: str,
    phase: str,
    mode: str,
    label: str,
    stream_id: str,
    seed_plan: RecurrentEvaluationSeedPlan,
) -> dict[str, object]:
    sampling_seeds = _candidate_sampling_seeds(
        stream_id,
        candidate_action_selection=mode,
        count=2,
    )
    replay_checks: list[dict[str, object]] = []
    broad = _evaluation_context(
        arm=arm,
        phase=phase,
        context="broad_default",
        seeds=(101, 103),
        sampling_seeds=sampling_seeds,
        label=label,
        replay_checks=replay_checks,
    )
    carrion = _evaluation_context(
        arm=arm,
        phase=phase,
        context="fixture:carrion_only",
        seeds=(101, 103),
        sampling_seeds=sampling_seeds,
        label=label,
        replay_checks=replay_checks,
    )
    carrion_candidate = carrion["policies"]["public_recurrent"]["aggregate"]
    return {
        "schema_version": RECURRENT_EVALUATION_SCHEMA_VERSION,
        "artifact": None,
        "candidate_provenance": {
            "mode": "in_memory_unpinned_development_canary",
            "source_pinned": False,
            "synthetic_digest_label": True,
            "synthetic_noncandidate_digest_label": label,
            "noncandidate_development_canary": True,
            "promotion_evidence_eligible_from_provenance": False,
            "promotion_eligibility_requires_external_one_use_lockbox_authorization": (
                True
            ),
            "external_one_use_lockbox_authorization_available": False,
            "research_candidate_evidence_requires_external_validation_authorization": (
                True
            ),
            "external_validation_authorization_available": False,
            "full_canonical_lockbox_plan": seed_plan.full_canonical_lockbox_plan,
            "contains_canonical_validation_seed": (seed_plan.contains_validation_seed),
            "contains_canonical_lockbox_seed": seed_plan.contains_lockbox_seed,
            "artifact_training_seed_evidence": None,
            "caller_excluded_training_seeds_used_as_proof": False,
            "candidate_action_selection_promotion_eligible": False,
            "sampled_candidate_diagnostic": (
                mode == PUBLIC_RECURRENT_SAMPLED_SELECTION
            ),
            "feed_forward_history_ablation": False,
            "runtime_integration_authorized": False,
        },
        "execution_provenance": {
            "schema_version": RECURRENT_EVALUATION_EXECUTION_SCHEMA_VERSION,
            "evaluation_workers_requested": 1,
            "evaluation_workers_used": 1,
            "process_parallel": False,
            "process_start_method": None,
            "torch_threads_per_worker": 1,
            "environment_task_count": 4,
            "task_unit": (
                "one_environment_controls_once_plus_candidate_streams_and_exact_replays"
            ),
            "ordered_collection": "context_then_seed_input_order",
            "failure_policy": "raise_without_sequential_fallback",
            "runtime_reproducibility": {
                "schema_version": RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION,
                "python_version": "3.14.3",
                "python_implementation": "CPython",
                "torch_version": "2.11.0",
                "numpy_version": "2.4.4",
                "platform": {
                    "system": "Darwin",
                    "release": "test",
                    "machine": "arm64",
                    "platform_string": "test-platform",
                },
                "source_model_device_observed": "cpu",
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "requested_device_source": "frozen_cpu_evaluation_contract",
                "torch_deterministic_algorithms_enabled": True,
            },
        },
        "evaluation_contract": {
            "ticks": 120,
            "world_horizon_policy": "exact_configured_120_tick_horizon",
            "candidate_action_selection": mode,
            "candidate_sampling_seeds": [
                seed for seed in sampling_seeds if seed is not None
            ],
            "candidate_sampling_seed_count": len(sampling_seeds),
            "candidate_sampling_stream_id": stream_id,
            "candidate_sampling_stream_id_source": (
                "caller_supplied_arm_independent_identifier"
            ),
            "candidate_sampling_seed_contract": (
                "none_argmax"
                if sampling_seeds == (None,)
                else "sha256_namespace_sampling_stream_id_and_replicate_index_first_63_bits"
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
        "seed_plan": {
            "source": "caller_supplied_evaluation_plan",
            "digest": seed_plan.digest,
            "environment_seed_role": seed_plan.environment_seed_role,
            "canonical_registry_role": seed_plan.canonical_registry_role,
            "canonical_role_exact": seed_plan.canonical_registry_role is not None,
            "full_canonical_lockbox_plan": seed_plan.full_canonical_lockbox_plan,
            "contains_canonical_validation_seed": (seed_plan.contains_validation_seed),
            "contains_canonical_lockbox_seed": seed_plan.contains_lockbox_seed,
            "broad_seeds": list(seed_plan.broad_seeds),
            "fixture_seeds": list(seed_plan.fixture_seeds),
            "caller_excluded_training_seeds": list(seed_plan.excluded_training_seeds),
            "excluded_training_seed_count": len(seed_plan.excluded_training_seeds),
            "caller_excluded_training_seeds_used_as_proof": False,
            "canonical_training_seed_count": len(
                set(RECURRENT_SEED_REGISTRY["train"])
                | set(RECURRENT_SEED_REGISTRY["curriculum"])
            ),
            "train_evaluation_overlap_count": len(
                (set(seed_plan.broad_seeds) | set(seed_plan.fixture_seeds))
                & (
                    set(RECURRENT_SEED_REGISTRY["train"])
                    | set(RECURRENT_SEED_REGISTRY["curriculum"])
                )
            ),
        },
        "broad": broad,
        "fixtures": [{"fixture": "carrion_only", **carrion}],
        "carrion_fixture_terminal_nonextinct_run_count": carrion_candidate[
            "terminal_nonextinct_run_count"
        ],
        "carrion_fixture_terminal_alive_agent_total": carrion_candidate[
            "terminal_alive_agent_total"
        ],
        "replay_verification": {
            "all_passed": True,
            "checked_run_count": len(replay_checks),
            "checks": replay_checks,
        },
    }


def _evaluation_context(
    *,
    arm: str,
    phase: str,
    context: str,
    seeds: tuple[int, ...],
    sampling_seeds: tuple[int | None, ...],
    label: str,
    replay_checks: list[dict[str, object]],
) -> dict[str, object]:
    candidate_runs = []
    linear_runs = []
    random_runs = []
    paired_linear = []
    paired_random = []
    for seed in seeds:
        linear = _run(
            context=context,
            seed=seed,
            sampling_seed=None,
            terminal_alive=2,
            births=3,
            deaths=1,
            reward_total=3.0,
            candidate=False,
            label="linear",
        )
        random = _run(
            context=context,
            seed=seed,
            sampling_seed=None,
            terminal_alive=1,
            births=2,
            deaths=2,
            reward_total=1.0,
            candidate=False,
            label="random",
        )
        linear_runs.append(linear)
        random_runs.append(random)
        for sampling_seed in sampling_seeds:
            before = 1
            after_offset = {
                BASE_ARM: 1,
                EXACT_ARM: 3,
                SHUFFLED_ARM: 2,
            }[arm]
            terminal_alive = before if phase == "before" else before + after_offset
            candidate = _run(
                context=context,
                seed=seed,
                sampling_seed=sampling_seed,
                terminal_alive=terminal_alive,
                births=(2 if phase == "before" else 2 + after_offset),
                deaths=1,
                reward_total=(2.0 if phase == "before" else 2.0 + after_offset),
                candidate=True,
                label=label,
            )
            candidate_runs.append(candidate)
            paired_linear.append(linear)
            paired_random.append(random)
            replay_checks.append(
                {
                    "context": context,
                    "seed": seed,
                    "policy_sampling_seed": sampling_seed,
                    "digest": candidate["replay_digest"],
                    "passed": True,
                }
            )
    policies = {
        "public_recurrent": {
            "runs": candidate_runs,
            "aggregate": _aggregate_runs(candidate_runs),
        },
        "mind_v3_linear": {
            "runs": linear_runs,
            "aggregate": _aggregate_runs(linear_runs),
        },
        "masked_random": {
            "runs": random_runs,
            "aggregate": _aggregate_runs(random_runs),
        },
    }
    return {
        "seeds": list(seeds),
        "candidate_sampling_seeds": [
            seed for seed in sampling_seeds if seed is not None
        ],
        "candidate_run_count_per_environment": len(sampling_seeds),
        "controls_run_once_per_environment": True,
        "candidate_sampling_analysis": _candidate_sampling_analysis(candidate_runs),
        "policies": policies,
        "paired_deltas": {
            "candidate_minus_mind_v3_linear": _paired_deltas(
                candidate_runs,
                paired_linear,
            ),
            "candidate_minus_masked_random": _paired_deltas(
                candidate_runs,
                paired_random,
            ),
        },
    }


def _run(
    *,
    context: str,
    seed: int,
    sampling_seed: int | None,
    terminal_alive: int,
    births: int,
    deaths: int,
    reward_total: float,
    candidate: bool,
    label: str,
) -> dict[str, object]:
    components = {component: 0.0 for component in REWARD_COMPONENT_BOUNDS}
    first_component = next(iter(components))
    components[first_component] = reward_total
    requested_counts = {"eat": 3, "stay": 1}
    if candidate:
        distribution = _learned_distribution()
        policy_id_counts = {PUBLIC_RECURRENT_POLICY_ID: 4}
        source = "public_recurrent_rollout"
    else:
        distribution = _empty_distribution()
        policy_id_counts = {label: 4}
        source = f"{label}_control"
    run: dict[str, object] = {
        "context": context,
        "seed": seed,
        "policy_sampling_seed": sampling_seed,
        "horizon_ticks": 120,
        "ticks_executed": 120,
        "terminal_alive": terminal_alive,
        "births": births,
        "deaths": deaths,
        "reward_total": reward_total,
        "reward_component_totals": components,
        "trajectory_record_count": 4,
        "policy_decision_record_count": 4,
        "passive_trajectory_record_count": 0,
        "requested_action_counts": requested_counts,
        "dominant_requested_action": "eat",
        "dominant_requested_action_count": 3,
        "dominant_requested_action_share": 0.75,
        "unsupported_requested_action_count": 0,
        "heuristic_action_source_count": 0,
        "action_source_counts": {source: 4},
        "policy_id_counts": policy_id_counts,
        "eat_requested_count": 3,
        "eat_without_positive_resource_gain_count": 1,
        "eat_without_positive_resource_gain_share": 1.0 / 3.0,
        "learned_masked_distribution": distribution,
    }
    run["replay_digest"] = stable_payload_digest(
        {"run": run, "arm_specific_policy_label": label}
    )
    return run


def _learned_distribution() -> dict[str, object]:
    metrics = (
        "entropy",
        "normalized_entropy",
        "selected_action_probability",
        "top_action_probability",
        "top_two_probability_margin",
        "eat_probability",
    )
    means = {
        "entropy": 0.5,
        "normalized_entropy": 0.4,
        "selected_action_probability": 0.75,
        "top_action_probability": 0.75,
        "top_two_probability_margin": 0.5,
        "eat_probability": 0.75,
    }
    probabilities = {action: 0.0 for action in ACTION_NAMES}
    probabilities["eat"] = 0.75
    probabilities["stay"] = 0.25
    return {
        "decision_count": 4,
        "metric_observation_counts": {metric: 4 for metric in metrics},
        "metric_means": means,
        "metric_minima": dict(means),
        "metric_maxima": dict(means),
        "mean_action_probabilities": probabilities,
    }


def _empty_distribution() -> dict[str, object]:
    metrics = (
        "entropy",
        "normalized_entropy",
        "selected_action_probability",
        "top_action_probability",
        "top_two_probability_margin",
        "eat_probability",
    )
    return {
        "decision_count": 0,
        "metric_observation_counts": {metric: 0 for metric in metrics},
        "metric_means": {metric: None for metric in metrics},
        "metric_minima": {metric: None for metric in metrics},
        "metric_maxima": {metric: None for metric in metrics},
        "mean_action_probabilities": {action: None for action in ACTION_NAMES},
    }


def _state(digest: str) -> dict[str, object]:
    return {
        "state_sha256": digest,
        "state_tensor_count": 10,
        "state_value_count": 100,
        "parameter_count": 100,
        "trainable_parameter_count": 100,
        "dtype_value_counts": {"torch.float32": 100},
        "hash_contract": "sorted_state_dict_name_dtype_shape_and_raw_cpu_bytes_v1",
    }


def _reseal(report: dict[str, object]) -> None:
    report.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(report)


def _reseal_preregistration_and_labels(report: dict[str, object]) -> None:
    prereg = report["evaluation_preregistration"]
    prereg.pop("exact_digest", None)
    prereg.pop("synthetic_noncandidate_digest_label", None)
    digest = stable_payload_digest(prereg)
    label = UNPINNED_LABEL_PREFIX + digest
    prereg["exact_digest"] = digest
    prereg["synthetic_noncandidate_digest_label"] = label
    for phase in ("before", "after"):
        for mode in (
            PUBLIC_RECURRENT_ARGMAX_SELECTION,
            PUBLIC_RECURRENT_SAMPLED_SELECTION,
        ):
            report["evaluations"][phase][mode]["candidate_provenance"][
                "synthetic_noncandidate_digest_label"
            ] = label


if __name__ == "__main__":
    unittest.main()
