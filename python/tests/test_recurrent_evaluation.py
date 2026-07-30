from __future__ import annotations

from collections.abc import Mapping
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.env.runtime.trajectory import REWARD_COMPONENT_BOUNDS
    from evolution_sim.mind.recurrent_actor_critic import (
        GENOME_CONDITIONING_ACTOR_FILM_V1,
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_artifact import (
        FROZEN_RECURRENT_POLICY_ARTIFACT_KIND,
        FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
        FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
        RECURRENT_ARTIFACT_SCHEMA_VERSION,
        RECURRENT_REPLAY_PROBE_CONTRACT_VERSION,
        save_frozen_recurrent_policy_artifact,
        save_recurrent_artifact,
    )
    from evolution_sim.mind.recurrent_evaluation import (
        RECURRENT_EVALUATION_CANDIDATE_SEED_ROLE,
        RECURRENT_EVALUATION_GENOME_POPULATION_SCHEMA_VERSION,
        LEGACY_RECURRENT_EVALUATION_SCHEMA_VERSION,
        RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE,
        RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE,
        RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE,
        RECURRENT_EVALUATION_SELECTION_SEED_ROLE,
        RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION,
        RECURRENT_EVALUATION_SCHEMA_VERSION,
        RECURRENT_EVALUATION_TICKS,
        UNPINNED_NONCANDIDATE_DIGEST_PREFIX,
        VERIFIED_FROZEN_RECURRENT_POLICY_ARTIFACT_MODE,
        MaskedRandomPolicy,
        RecurrentEvaluationError,
        RecurrentEvaluationSeedPlan,
        _EvaluationEnvironmentTask,
        _aggregate_runs,
        _artifact_independent_behavior_payload,
        _canonical_sha256,
        _candidate_sampling_seeds,
        _evaluate_context,
        _run_parallel_environment_tasks,
        _run_outcome_evidence_sha256,
        _run_policy_world,
        _validated_evaluation_genome_population_configuration,
        _source_pinned_artifact_evidence_sha256,
        _validate_report,
        _validate_run,
        _validate_source_pinned_artifact_identity,
        evaluate_frozen_recurrent_policy_artifact,
        evaluate_recurrent_artifact,
        evaluate_recurrent_model,
    )
    from evolution_sim.mind.recurrent_policy import (
        PUBLIC_RECURRENT_ARGMAX_SELECTION,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
        RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION,
        DeterministicPublicRecurrentPolicy,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        CANONICAL_SEED_REGISTRY_SHA256,
        RECURRENT_SEED_REGISTRY,
        SCALE_DEVELOPMENT_CANONICAL_SHA256,
        SCALE_DEVELOPMENT_SEED_REGISTRY,
        SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
        SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class MaskedRandomPolicyTests(unittest.TestCase):
    def test_seeded_control_is_repeatable_and_uses_only_current_valid_actions(
        self,
    ) -> None:
        action_mask = {
            action: action in {"stay", "move_east"} for action in ACTION_NAMES
        }
        first = MaskedRandomPolicy(world_seed=813)
        second = MaskedRandomPolicy(world_seed=813)

        first_actions = [
            first.decide({}, dict(action_mask)).requested_action for _ in range(40)
        ]
        second_actions = [
            second.decide({}, dict(action_mask)).requested_action for _ in range(40)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertTrue(set(first_actions).issubset({"stay", "move_east"}))
        self.assertEqual(set(first_actions), {"stay", "move_east"})

    def test_masked_random_fails_closed_on_mask_drift(self) -> None:
        policy = MaskedRandomPolicy(world_seed=17)
        valid = {action: action == "stay" for action in ACTION_NAMES}

        missing = dict(valid)
        missing.pop(ACTION_NAMES[-1])
        with self.assertRaisesRegex(RecurrentEvaluationError, "keys differ"):
            policy.decide({}, missing)

        non_boolean = dict(valid)
        non_boolean["stay"] = 1  # type: ignore[assignment]
        with self.assertRaisesRegex(RecurrentEvaluationError, "exact booleans"):
            policy.decide({}, non_boolean)

        empty = {action: False for action in ACTION_NAMES}
        with self.assertRaisesRegex(RecurrentEvaluationError, "no valid action"):
            policy.decide({}, empty)


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentEvaluationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_full_behavior_digest_normalizes_only_recurrent_artifact_labels(
        self,
    ) -> None:
        first = {
            "summary": {"alive_agents": 2},
            "trajectory_records": [
                {
                    "policy_id": "mind_v3_public_recurrent_actor_critic",
                    "policy_version": (
                        "mind_v3_public_recurrent_actor_critic_v1+aaaaaaaaaaaaaaaa+"
                        "deterministic_masked_argmax"
                    ),
                    "requested_action": "eat",
                }
            ],
            "policy_decision_diagnostics": [
                {"artifact_digest": "a" * 64, "selected_action": "eat"}
            ],
        }
        second = copy.deepcopy(first)
        second["trajectory_records"][0]["policy_version"] = (  # type: ignore[index]
            "mind_v3_public_recurrent_actor_critic_v1+bbbbbbbbbbbbbbbb+"
            "deterministic_masked_argmax"
        )
        second["policy_decision_diagnostics"][0]["artifact_digest"] = (  # type: ignore[index]
            "b" * 64
        )

        self.assertNotEqual(_canonical_sha256(first), _canonical_sha256(second))
        self.assertEqual(
            _canonical_sha256(_artifact_independent_behavior_payload(first)),
            _canonical_sha256(_artifact_independent_behavior_payload(second)),
        )
        second["trajectory_records"][0]["requested_action"] = "stay"  # type: ignore[index]
        self.assertNotEqual(
            _canonical_sha256(_artifact_independent_behavior_payload(first)),
            _canonical_sha256(_artifact_independent_behavior_payload(second)),
        )

    def test_seed_plan_requires_unique_holdouts_excluded_from_training(self) -> None:
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=[101, 103],
            fixture_seeds=[107],
            excluded_training_seeds=[109, 113],
        )
        self.assertEqual(plan.broad_seeds, (101, 103))
        self.assertEqual(len(plan.digest), 64)

        with self.assertRaisesRegex(RecurrentEvaluationError, "overlap"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=(101,),
                fixture_seeds=(103,),
                excluded_training_seeds=(101, 109),
            )
        with self.assertRaisesRegex(RecurrentEvaluationError, "unique"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=(101, 101),
                fixture_seeds=(103,),
                excluded_training_seeds=(109,),
            )
        with self.assertRaisesRegex(RecurrentEvaluationError, "non-empty sequence"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=(101,),
                fixture_seeds=(),
                excluded_training_seeds=(109,),
            )
        with self.assertRaisesRegex(RecurrentEvaluationError, "non-empty sequence"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=(101,),
                fixture_seeds=(103,),
                excluded_training_seeds=(),
            )
        with self.assertRaises(TypeError):
            RecurrentEvaluationSeedPlan(  # type: ignore[call-arg]
                broad_seeds=(101,),
                fixture_seeds=(103,),
            )

    def test_scale_selection_plan_is_canonical_and_keeps_old_sealed_roles_closed(
        self,
    ) -> None:
        selection = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"][:8]
        excluded = (
            *SCALE_DEVELOPMENT_SEED_REGISTRY["scale_train"],
            *SCALE_DEVELOPMENT_SEED_REGISTRY["scale_curriculum"],
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=selection,
            fixture_seeds=selection,
            excluded_training_seeds=excluded,
            environment_seed_role=(RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE),
        )

        self.assertEqual(plan.canonical_registry_role, "scale_selection")
        self.assertFalse(plan.contains_validation_seed)
        self.assertFalse(plan.contains_lockbox_seed)
        self.assertTrue(set(selection).isdisjoint(excluded))

        with self.assertRaisesRegex(RecurrentEvaluationError, "canonical"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=tuple(reversed(selection)),
                fixture_seeds=tuple(reversed(selection)),
                excluded_training_seeds=excluded,
                environment_seed_role=(RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE),
            )

        with self.assertRaisesRegex(RecurrentEvaluationError, "declare"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=selection,
                fixture_seeds=selection,
                excluded_training_seeds=excluded,
            )

    def test_scale_v2_selection_plan_is_canonical_and_registry_separated(
        self,
    ) -> None:
        selection = SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_selection"][:8]
        excluded = (
            *SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_train"],
            *SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_curriculum"],
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=selection,
            fixture_seeds=selection,
            excluded_training_seeds=excluded,
            environment_seed_role=(RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE),
        )

        self.assertEqual(plan.canonical_registry_role, "scale_v2_selection")
        self.assertFalse(plan.contains_validation_seed)
        self.assertFalse(plan.contains_lockbox_seed)
        self.assertTrue(set(selection).isdisjoint(excluded))
        with self.assertRaisesRegex(RecurrentEvaluationError, "canonical"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"][:8],
                fixture_seeds=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"][:8],
                excluded_training_seeds=excluded,
                environment_seed_role=(
                    RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE
                ),
            )
        with self.assertRaisesRegex(RecurrentEvaluationError, "declare"):
            RecurrentEvaluationSeedPlan(
                broad_seeds=selection,
                fixture_seeds=selection,
                excluded_training_seeds=excluded,
            )

    def test_frozen_scale_v1_artifact_replay_contract_remains_supported(
        self,
    ) -> None:
        source_commit = "a" * 40
        source_manifest = "b" * 64
        train_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_train"][0]
        curriculum_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_curriculum"][0]
        selection_seed = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"][0]
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(selection_seed,),
            fixture_seeds=(selection_seed,),
            excluded_training_seeds=(train_seed, curriculum_seed),
            environment_seed_role=(RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE),
        )
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=16, hidden_size=16),
            initialization_seed=7,
        )
        replay_manifest = {
            "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": "c" * 64,
            "replay_engine_contract_sha256": "d" * 64,
            "environment_seed_registry_sha256": (SCALE_DEVELOPMENT_CANONICAL_SHA256),
            "environment_seed_roles": ["scale_selection"],
            "scenario_names": ["broad", "carrion_only"],
            "tick_horizons": [120],
            "world_count": 2,
            "replay_verified_world_count": 2,
            "policy_sampling_stream_count": 1,
            "all_replays_exact": True,
            "verification_runner": "unit-test",
            "verification_runner_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "scale-policy.json"
            save_frozen_recurrent_policy_artifact(
                artifact_path,
                model,
                training_config={"feed_forward_history_ablation": False},
                experiment_config={"arm": "base"},
                seed_registry_digest=SCALE_DEVELOPMENT_CANONICAL_SHA256,
                source_commit=source_commit,
                source_manifest_sha256=source_manifest,
                data_metadata={
                    "environment_seed_roles": [
                        "scale_train",
                        "scale_curriculum",
                    ],
                    "environment_seeds_by_role": {
                        "scale_train": [train_seed],
                        "scale_curriculum": [curriculum_seed],
                    },
                },
                run_metadata={"development_only": True},
                full_world_replay_manifest=replay_manifest,
                learner_seed=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0],
                learner_device="cpu",
            )
            with patch(
                "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
                return_value={"verified": True},
            ) as evaluator:
                report = evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=plan,
                    expected_source_commit=source_commit,
                    expected_source_manifest_sha256=source_manifest,
                    expected_seed_registry_digest=(SCALE_DEVELOPMENT_CANONICAL_SHA256),
                    fixture_names=("carrion_only",),
                )
            self.assertEqual(report, {"verified": True})
            artifact_evidence = evaluator.call_args.kwargs["artifact_evidence"]
            self.assertEqual(
                artifact_evidence["schema_version"],
                FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
            )
            self.assertEqual(
                artifact_evidence["artifact_kind"],
                FROZEN_RECURRENT_POLICY_ARTIFACT_KIND,
            )
            self.assertEqual(
                artifact_evidence["replay_probe_contract_version"],
                RECURRENT_REPLAY_PROBE_CONTRACT_VERSION,
            )
            self.assertEqual(
                artifact_evidence["replay_probe_verified_on_device"],
                "cpu",
            )
            self.assertEqual(
                artifact_evidence["artifact_evidence_sha256"],
                _source_pinned_artifact_evidence_sha256(artifact_evidence),
            )
            candidate_provenance = evaluator.call_args.kwargs["candidate_provenance"]
            self.assertEqual(
                candidate_provenance["mode"],
                VERIFIED_FROZEN_RECURRENT_POLICY_ARTIFACT_MODE,
            )
            self.assertEqual(
                candidate_provenance["pin_verification"],
                "frozen_policy_artifact_v4_registry_source_commit_"
                "source_manifest_and_cpu_probe_v3",
            )
            _validate_source_pinned_artifact_identity(
                artifact_evidence,
                candidate_provenance,
                seed_plan=plan,
                trusted_artifact_path=artifact_path,
            )
            evidence = artifact_evidence["training_seed_evidence"]
            self.assertEqual(
                evidence["required_environment_seed_roles"],
                ["scale_train", "scale_curriculum"],
            )
            self.assertTrue(evidence["role_bound_provenance_complete"])

            with self.assertRaisesRegex(RecurrentEvaluationError, "manifest"):
                evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=plan,
                    expected_source_commit=source_commit,
                    expected_source_manifest_sha256="f" * 64,
                    expected_seed_registry_digest=(SCALE_DEVELOPMENT_CANONICAL_SHA256),
                    fixture_names=("carrion_only",),
                )

            scale_v2_selection_seed = SCALE_DEVELOPMENT_V2_SEED_REGISTRY[
                "scale_v2_selection"
            ][0]
            scale_v2_plan = RecurrentEvaluationSeedPlan(
                broad_seeds=(scale_v2_selection_seed,),
                fixture_seeds=(scale_v2_selection_seed,),
                excluded_training_seeds=(
                    SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_train"][0],
                    SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_curriculum"][0],
                ),
                environment_seed_role=(
                    RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE
                ),
            )
            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "scale-v2 selection evaluation requires",
            ):
                evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=scale_v2_plan,
                    expected_source_commit=source_commit,
                    expected_source_manifest_sha256=source_manifest,
                    expected_seed_registry_digest=(SCALE_DEVELOPMENT_CANONICAL_SHA256),
                    fixture_names=("carrion_only",),
                )

    def test_frozen_scale_v2_artifact_requires_v2_plan_and_registry_digest(
        self,
    ) -> None:
        source_commit = "1" * 40
        source_manifest = "2" * 64
        train_seed = SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_train"][0]
        curriculum_seed = SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_curriculum"][0]
        selection_seed = SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_selection"][0]
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(selection_seed,),
            fixture_seeds=(selection_seed,),
            excluded_training_seeds=(train_seed, curriculum_seed),
            environment_seed_role=(RECURRENT_EVALUATION_SCALE_V2_SELECTION_SEED_ROLE),
        )
        v1_plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"][0],),
            fixture_seeds=(SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"][0],),
            excluded_training_seeds=(
                SCALE_DEVELOPMENT_SEED_REGISTRY["scale_train"][0],
                SCALE_DEVELOPMENT_SEED_REGISTRY["scale_curriculum"][0],
            ),
            environment_seed_role=(RECURRENT_EVALUATION_SCALE_SELECTION_SEED_ROLE),
        )
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=16, hidden_size=16),
            initialization_seed=11,
        )
        replay_manifest = {
            "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": "3" * 64,
            "replay_engine_contract_sha256": "4" * 64,
            "environment_seed_registry_sha256": (SCALE_DEVELOPMENT_V2_CANONICAL_SHA256),
            "environment_seed_roles": ["scale_v2_selection"],
            "scenario_names": ["broad", "carrion_only"],
            "tick_horizons": [120],
            "world_count": 2,
            "replay_verified_world_count": 2,
            "policy_sampling_stream_count": 1,
            "all_replays_exact": True,
            "verification_runner": "unit-test",
            "verification_runner_sha256": "5" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "scale-v2-policy.json"
            save_frozen_recurrent_policy_artifact(
                artifact_path,
                model,
                training_config={"feed_forward_history_ablation": False},
                experiment_config={"arm": "base"},
                seed_registry_digest=SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
                source_commit=source_commit,
                source_manifest_sha256=source_manifest,
                data_metadata={
                    "environment_seed_roles": [
                        "scale_v2_train",
                        "scale_v2_curriculum",
                    ],
                    "environment_seeds_by_role": {
                        "scale_v2_train": [train_seed],
                        "scale_v2_curriculum": [curriculum_seed],
                    },
                },
                run_metadata={"development_only": True},
                full_world_replay_manifest=replay_manifest,
                learner_seed=SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"][0],
                learner_device="cpu",
            )
            with patch(
                "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
                return_value={"verified": "scale-v2"},
            ) as evaluator:
                report = evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=plan,
                    expected_source_commit=source_commit,
                    expected_source_manifest_sha256=source_manifest,
                    expected_seed_registry_digest=(
                        SCALE_DEVELOPMENT_V2_CANONICAL_SHA256
                    ),
                    fixture_names=("carrion_only",),
                )
            self.assertEqual(report, {"verified": "scale-v2"})
            evidence = evaluator.call_args.kwargs["artifact_evidence"][
                "training_seed_evidence"
            ]
            self.assertEqual(
                evidence["schema_version"],
                "mind_public_recurrent_training_seed_evidence_v3",
            )
            self.assertEqual(
                evidence["required_environment_seed_roles"],
                ["scale_v2_train", "scale_v2_curriculum"],
            )
            self.assertTrue(evidence["role_bound_provenance_complete"])

            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "scale selection evaluation requires",
            ):
                evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=v1_plan,
                    expected_source_commit=source_commit,
                    expected_source_manifest_sha256=source_manifest,
                    expected_seed_registry_digest=(
                        SCALE_DEVELOPMENT_V2_CANONICAL_SHA256
                    ),
                    fixture_names=("carrion_only",),
                )
            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "seed registry does not match",
            ):
                evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=plan,
                    expected_source_commit=source_commit,
                    expected_source_manifest_sha256=source_manifest,
                    expected_seed_registry_digest=(SCALE_DEVELOPMENT_CANONICAL_SHA256),
                    fixture_names=("carrion_only",),
                )

    def test_role_bound_seed_plans_require_exact_canonical_membership(
        self,
    ) -> None:
        selection = RECURRENT_SEED_REGISTRY["selection"]
        validation = RECURRENT_SEED_REGISTRY["validation"]
        lockbox = RECURRENT_SEED_REGISTRY["lockbox"]
        excluded = (
            *RECURRENT_SEED_REGISTRY["train"],
            *RECURRENT_SEED_REGISTRY["curriculum"],
        )

        plans = (
            (
                RECURRENT_EVALUATION_SELECTION_SEED_ROLE,
                selection,
                "selection",
                False,
            ),
            (
                RECURRENT_EVALUATION_CANDIDATE_SEED_ROLE,
                validation,
                "validation",
                False,
            ),
            (
                RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE,
                lockbox,
                "lockbox",
                True,
            ),
        )
        for role, seeds, registry_role, authorized in plans:
            with self.subTest(role=role):
                plan = RecurrentEvaluationSeedPlan(
                    broad_seeds=seeds,
                    fixture_seeds=seeds,
                    excluded_training_seeds=excluded,
                    environment_seed_role=role,
                )
                self.assertEqual(plan.canonical_registry_role, registry_role)
                self.assertIs(plan.full_canonical_lockbox_plan, authorized)
                subset = RecurrentEvaluationSeedPlan(
                    broad_seeds=seeds[:2],
                    fixture_seeds=seeds[:2],
                    excluded_training_seeds=excluded,
                    environment_seed_role=role,
                )
                self.assertEqual(subset.canonical_registry_role, registry_role)
                self.assertFalse(subset.full_canonical_lockbox_plan)
                with self.assertRaisesRegex(
                    RecurrentEvaluationError,
                    "canonical ordered subset",
                ):
                    RecurrentEvaluationSeedPlan(
                        broad_seeds=tuple(reversed(seeds[:2])),
                        fixture_seeds=tuple(reversed(seeds[:2])),
                        excluded_training_seeds=excluded,
                        environment_seed_role=role,
                    )

        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "canonical ordered subset",
        ):
            RecurrentEvaluationSeedPlan(
                broad_seeds=selection[:2],
                fixture_seeds=selection[:2],
                excluded_training_seeds=excluded,
                environment_seed_role=RECURRENT_EVALUATION_CANDIDATE_SEED_ROLE,
            )

    def test_development_label_cannot_hide_reserved_seed_membership(self) -> None:
        excluded = RECURRENT_SEED_REGISTRY["train"][:1]
        lockbox = RECURRENT_SEED_REGISTRY["lockbox"]
        for reserved_seeds in (
            RECURRENT_SEED_REGISTRY["selection"][:1],
            RECURRENT_SEED_REGISTRY["validation"][:1],
            lockbox[:1],
            lockbox,
        ):
            with self.subTest(reserved_count=len(reserved_seeds)):
                with self.assertRaisesRegex(
                    RecurrentEvaluationError,
                    "development evaluation cannot consume reserved",
                ):
                    RecurrentEvaluationSeedPlan(
                        broad_seeds=reserved_seeds,
                        fixture_seeds=reserved_seeds,
                        excluded_training_seeds=excluded,
                    )

    def test_fixture_names_fail_closed_before_artifact_io(self) -> None:
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(101,),
            fixture_seeds=(103,),
            excluded_training_seeds=(109,),
        )
        with self.assertRaisesRegex(RecurrentEvaluationError, "unsupported fixture"):
            evaluate_recurrent_artifact(
                Path("does-not-exist.json"),
                seed_plan=plan,
                fixture_names=("invented_fixture",),
            )

    def test_in_memory_entrypoint_is_explicitly_unpinned_and_noncandidate(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=29,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=[101],
            fixture_seeds=[103],
            excluded_training_seeds=[107],
        )
        label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + ("c" * 64)

        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
            return_value={"result": "development_canary"},
        ) as evaluate:
            result = evaluate_recurrent_model(
                model,
                synthetic_noncandidate_digest_label=label,
                seed_plan=plan,
                fixture_names=("carrion_only",),
            )

        self.assertEqual(result, {"result": "development_canary"})
        call = evaluate.call_args.kwargs
        self.assertIs(call["candidate_model"], model)
        self.assertEqual(call["policy_digest_label"], label)
        self.assertIsNone(call["artifact_evidence"])
        provenance = call["candidate_provenance"]
        self.assertFalse(provenance["source_pinned"])
        self.assertTrue(provenance["noncandidate_development_canary"])
        self.assertFalse(provenance["promotion_evidence_eligible_from_provenance"])
        self.assertFalse(provenance["runtime_integration_authorized"])
        self.assertFalse(call["feed_forward_history_ablation"])
        self.assertEqual(
            call["candidate_action_selection"],
            PUBLIC_RECURRENT_ARGMAX_SELECTION,
        )
        self.assertEqual(call["candidate_sampling_seed_count"], 1)
        self.assertIsNone(call["candidate_sampling_stream_id"])
        self.assertEqual(call["evaluation_workers"], 1)

        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
            return_value={"result": "feed_forward_canary"},
        ) as feed_forward_evaluate:
            result = evaluate_recurrent_model(
                model,
                synthetic_noncandidate_digest_label=label,
                seed_plan=plan,
                fixture_names=("carrion_only",),
                feed_forward_history_ablation=True,
            )
        self.assertEqual(result, {"result": "feed_forward_canary"})
        self.assertTrue(
            feed_forward_evaluate.call_args.kwargs["feed_forward_history_ablation"]
        )

        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
            return_value={"result": "sampled_canary"},
        ) as sampled_evaluate:
            result = evaluate_recurrent_model(
                model,
                synthetic_noncandidate_digest_label=label,
                seed_plan=plan,
                fixture_names=("carrion_only",),
                candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
            )
        self.assertEqual(result, {"result": "sampled_canary"})
        self.assertEqual(
            sampled_evaluate.call_args.kwargs["candidate_action_selection"],
            PUBLIC_RECURRENT_SAMPLED_SELECTION,
        )

        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
            return_value={"result": "multi_sampler_canary"},
        ) as multi_sampler_evaluate:
            result = evaluate_recurrent_model(
                model,
                synthetic_noncandidate_digest_label=label,
                seed_plan=plan,
                fixture_names=("carrion_only",),
                candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
                candidate_sampling_seed_count=4,
                candidate_sampling_stream_id="matched-arm-stream:test",
                evaluation_workers=3,
            )
        self.assertEqual(result, {"result": "multi_sampler_canary"})
        self.assertEqual(
            multi_sampler_evaluate.call_args.kwargs["candidate_sampling_seed_count"],
            4,
        )
        self.assertEqual(
            multi_sampler_evaluate.call_args.kwargs["candidate_sampling_stream_id"],
            "matched-arm-stream:test",
        )
        self.assertEqual(
            multi_sampler_evaluate.call_args.kwargs["evaluation_workers"],
            3,
        )

        with self.assertRaisesRegex(RecurrentEvaluationError, "must start"):
            evaluate_recurrent_model(
                model,
                synthetic_noncandidate_digest_label="c" * 64,
                seed_plan=plan,
                fixture_names=("carrion_only",),
            )

    def test_evaluation_workers_fail_closed_before_artifact_io_or_model_work(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=43,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(101,),
            fixture_seeds=(103,),
            excluded_training_seeds=(107,),
        )
        label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + ("e" * 64)

        for invalid_workers in (0, 65, True, 1.5):
            with self.subTest(invalid_workers=invalid_workers):
                with self.assertRaisesRegex(
                    RecurrentEvaluationError,
                    r"evaluation_workers.*\[1, 64\]",
                ):
                    evaluate_recurrent_artifact(
                        Path("does-not-exist.json"),
                        seed_plan=plan,
                        fixture_names=("carrion_only",),
                        evaluation_workers=invalid_workers,  # type: ignore[arg-type]
                    )
                with (
                    patch(
                        "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model"
                    ) as frozen_evaluate,
                    self.assertRaisesRegex(
                        RecurrentEvaluationError,
                        r"evaluation_workers.*\[1, 64\]",
                    ),
                ):
                    evaluate_recurrent_model(
                        model,
                        synthetic_noncandidate_digest_label=label,
                        seed_plan=plan,
                        fixture_names=("carrion_only",),
                        evaluation_workers=invalid_workers,  # type: ignore[arg-type]
                    )
                frozen_evaluate.assert_not_called()

    def test_spawn_worker_failure_has_no_sequential_fallback(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=47,
        )
        tasks = tuple(
            _EvaluationEnvironmentTask(
                seed=seed,
                fixture_name=None,
                policy_digest_label="test-label",
                feed_forward_history_ablation=False,
                candidate_action_selection=PUBLIC_RECURRENT_ARGMAX_SELECTION,
                sampling_seeds=(None,),
            )
            for seed in (101, 103)
        )
        with (
            patch(
                "evolution_sim.mind.recurrent_evaluation.ProcessPoolExecutor",
                side_effect=RuntimeError("spawn unavailable"),
            ),
            patch(
                "evolution_sim.mind.recurrent_evaluation._evaluate_environment_task"
            ) as sequential_task,
            self.assertRaisesRegex(
                RecurrentEvaluationError,
                "no sequential fallback",
            ),
        ):
            _run_parallel_environment_tasks(
                tasks,
                candidate_model=model,
                evaluation_workers=2,
            )
        sequential_task.assert_not_called()

    def test_candidate_sampling_seeds_are_namespaced_unique_and_fail_closed(
        self,
    ) -> None:
        label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + ("d" * 64)
        first = _candidate_sampling_seeds(
            label,
            candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
            count=8,
        )
        second = _candidate_sampling_seeds(
            label,
            candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
            count=8,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)
        self.assertTrue(all(seed is not None and 0 <= seed < 2**63 for seed in first))
        self.assertEqual(
            _candidate_sampling_seeds(
                label,
                candidate_action_selection=PUBLIC_RECURRENT_ARGMAX_SELECTION,
                count=8,
            ),
            (None,),
        )
        for invalid_count in (0, 33, True, 1.5):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaisesRegex(RecurrentEvaluationError, r"\[1, 32\]"):
                    _candidate_sampling_seeds(
                        label,
                        candidate_action_selection=(PUBLIC_RECURRENT_SAMPLED_SELECTION),
                        count=invalid_count,  # type: ignore[arg-type]
                    )

    def test_shared_sampling_stream_is_independent_of_candidate_digest(self) -> None:
        stream_id = "matched-arm-stream:test"
        first = _candidate_sampling_seeds(
            stream_id,
            candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
            count=4,
        )
        second = _candidate_sampling_seeds(
            stream_id,
            candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
            count=4,
        )

        self.assertEqual(first, second)

    def test_multi_sampler_context_reuses_each_environment_control_once(
        self,
    ) -> None:
        calls: list[tuple[int, object, int | None]] = []

        def fake_candidate_policy(
            _model: object,
            _label: str,
            *,
            feed_forward_history_ablation: bool,
            candidate_action_selection: str,
            sampling_seed: int | None,
        ) -> tuple[str, int | None]:
            self.assertFalse(feed_forward_history_ablation)
            self.assertEqual(
                candidate_action_selection,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            )
            return ("candidate", sampling_seed)

        def fake_run(
            *,
            seed: int,
            fixture_name: str | None,
            policy: object,
            policy_sampling_seed: int | None = None,
            genome_population_mode: str = "disabled",
            genome_stream_seed: int | None = None,
        ) -> dict[str, object]:
            self.assertEqual(genome_population_mode, "disabled")
            self.assertIsNone(genome_stream_seed)
            calls.append((seed, policy, policy_sampling_seed))
            is_candidate = isinstance(policy, tuple) and policy[0] == "candidate"
            empty_distribution = {
                "decision_count": 0,
                "metric_observation_counts": {
                    metric: 0
                    for metric in (
                        "entropy",
                        "normalized_entropy",
                        "selected_action_probability",
                        "top_action_probability",
                        "top_two_probability_margin",
                        "eat_probability",
                    )
                },
                "metric_means": {
                    metric: None
                    for metric in (
                        "entropy",
                        "normalized_entropy",
                        "selected_action_probability",
                        "top_action_probability",
                        "top_two_probability_margin",
                        "eat_probability",
                    )
                },
                "metric_minima": {
                    metric: None
                    for metric in (
                        "entropy",
                        "normalized_entropy",
                        "selected_action_probability",
                        "top_action_probability",
                        "top_two_probability_margin",
                        "eat_probability",
                    )
                },
                "metric_maxima": {
                    metric: None
                    for metric in (
                        "entropy",
                        "normalized_entropy",
                        "selected_action_probability",
                        "top_action_probability",
                        "top_two_probability_margin",
                        "eat_probability",
                    )
                },
                "mean_action_probabilities": {action: None for action in ACTION_NAMES},
            }
            run = {
                "context": (
                    "broad_default"
                    if fixture_name is None
                    else f"fixture:{fixture_name}"
                ),
                "seed": seed,
                "policy_sampling_seed": (
                    policy_sampling_seed if is_candidate else None
                ),
                "horizon_ticks": 120,
                "ticks_executed": 120,
                "terminal_alive": int(is_candidate),
                "births": int(is_candidate),
                "deaths": 0,
                "reward_total": float(is_candidate),
                "reward_component_totals": {
                    component: 0.0 for component in REWARD_COMPONENT_BOUNDS
                },
                "trajectory_record_count": 1,
                "policy_decision_record_count": 1,
                "passive_trajectory_record_count": 0,
                "requested_action_counts": {"stay": 1},
                "dominant_requested_action": "stay",
                "dominant_requested_action_count": 1,
                "dominant_requested_action_share": 1.0,
                "unsupported_requested_action_count": 0,
                "heuristic_action_source_count": 0,
                "action_source_counts": {"test": 1},
                "policy_id_counts": {"test": 1},
                "eat_requested_count": 0,
                "eat_without_positive_resource_gain_count": 0,
                "eat_without_positive_resource_gain_share": None,
                "learned_masked_distribution": empty_distribution,
            }
            run["behavior_digest"] = _canonical_sha256({"fake_behavior": run})
            run["replay_digest"] = _canonical_sha256({"fake_replay": run})
            run["outcome_evidence_sha256"] = _run_outcome_evidence_sha256(run)
            return run

        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=41,
        )
        replay_checks: list[dict[str, object]] = []
        with (
            patch(
                "evolution_sim.mind.recurrent_evaluation._new_candidate_policy",
                side_effect=fake_candidate_policy,
            ),
            patch(
                "evolution_sim.mind.recurrent_evaluation._run_policy_world",
                side_effect=fake_run,
            ),
        ):
            result = _evaluate_context(
                candidate_model=model,
                policy_digest_label="test-label",
                seeds=(101, 103),
                fixture_name="carrion_only",
                replay_checks=replay_checks,
                feed_forward_history_ablation=False,
                candidate_action_selection=PUBLIC_RECURRENT_SAMPLED_SELECTION,
                sampling_seeds=(11, 22),
            )

        self.assertEqual(len(calls), 12)
        self.assertEqual(result["candidate_run_count_per_environment"], 2)
        self.assertTrue(result["controls_run_once_per_environment"])
        policies = result["policies"]
        self.assertEqual(len(policies["public_recurrent"]["runs"]), 4)
        self.assertEqual(len(policies["mind_v3_linear"]["runs"]), 2)
        self.assertEqual(len(policies["masked_random"]["runs"]), 2)
        self.assertEqual(len(replay_checks), 4)
        self.assertTrue(
            all(
                check["outcome_evidence_sha256"] == run["outcome_evidence_sha256"]
                for check, run in zip(
                    replay_checks,
                    policies["public_recurrent"]["runs"],
                    strict=True,
                )
            )
        )
        for comparison in result["paired_deltas"].values():
            self.assertEqual(len(comparison["runs"]), 4)
        sampling = result["candidate_sampling_analysis"]
        self.assertEqual(sampling["environment_count"], 2)
        self.assertEqual(sampling["policy_sampling_stream_count"], 2)
        self.assertEqual(sampling["terminal_nonextinct_run_count"], 4)
        self.assertEqual(sampling["terminal_alive_agent_total"], 4)
        self.assertEqual(
            sampling["environments_with_any_nonextinct_stream_count"],
            2,
        )
        self.assertEqual(
            sampling["environments_with_all_streams_extinct_count"],
            0,
        )
        self.assertEqual(
            [
                row["terminal_nonextinct_run_count"]
                for row in sampling["policy_sampling_stream_stratified"]
            ],
            [2, 2],
        )
        wilson = sampling["descriptive_run_cell_wilson_95"]
        self.assertTrue(wilson["descriptive_only"])
        self.assertFalse(wilson["independence_assumption_satisfied"])
        self.assertIn("crossed", wilson["warning"])
        self.assertNotIn("terminal_survival_probability", sampling)
        self.assertNotIn("terminal_survival_probability_wilson_95", sampling)

        valid_run = policies["public_recurrent"]["runs"][0]
        _validate_run(valid_run)

        private_extension = copy.deepcopy(valid_run)
        private_extension["private_world_state"] = {"forged": True}
        with self.assertRaisesRegex(RecurrentEvaluationError, "field set drifted"):
            _validate_run(private_extension)

        boolean_ticks = copy.deepcopy(valid_run)
        boolean_ticks["ticks_executed"] = True
        with self.assertRaisesRegex(RecurrentEvaluationError, "nonnegative integer"):
            _validate_run(boolean_ticks)

        detached_outcome = copy.deepcopy(valid_run)
        detached_outcome["terminal_alive"] = 999
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "detached from behavior/replay evidence",
        ):
            _validate_run(detached_outcome)

    def test_artifact_evaluation_requires_external_commit_and_canonical_registry(
        self,
    ) -> None:
        assert torch is not None
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=20260720,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(101,),
            fixture_seeds=(103,),
            excluded_training_seeds=(107,),
        )
        with tempfile.TemporaryDirectory() as directory:
            canonical_path = Path(directory) / "canonical.json"
            save_recurrent_artifact(
                canonical_path,
                model,
                training_config={"algorithm": "test"},
                seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
                source_commit="a" * 40,
                data_metadata={"training_seeds": [107]},
                run_metadata={"purpose": "pin_contract_test"},
                learner_seed=20260720,
                learner_device="cpu",
            )
            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "expected_source_commit",
            ):
                evaluate_recurrent_artifact(canonical_path, seed_plan=plan)
            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "external expected pin",
            ):
                evaluate_recurrent_artifact(
                    canonical_path,
                    seed_plan=plan,
                    expected_source_commit="b" * 40,
                )

            noncanonical_path = Path(directory) / "noncanonical.json"
            save_recurrent_artifact(
                noncanonical_path,
                model,
                training_config={"algorithm": "test"},
                seed_registry_digest="b" * 64,
                source_commit="a" * 40,
                data_metadata={"training_seeds": [107]},
                run_metadata={"purpose": "pin_contract_test"},
                learner_seed=20260720,
                learner_device="cpu",
            )
            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "canonical registry",
            ):
                evaluate_recurrent_artifact(
                    noncanonical_path,
                    seed_plan=plan,
                    expected_source_commit="a" * 40,
                )

    def test_lockbox_plan_cannot_self_authorize_and_forged_exclusions_stay_ineligible(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=20260722,
        )
        train_seed = RECURRENT_SEED_REGISTRY["train"][0]
        curriculum_seed = RECURRENT_SEED_REGISTRY["curriculum"][0]
        excluded = (
            *RECURRENT_SEED_REGISTRY["train"],
            *RECURRENT_SEED_REGISTRY["curriculum"],
        )
        complete_training_provenance = {
            "environment_seed_roles": ["train", "curriculum"],
            "environment_seeds_by_role": {
                "train": [train_seed],
                "curriculum": [curriculum_seed],
            },
        }
        lockbox_plan = RecurrentEvaluationSeedPlan(
            broad_seeds=RECURRENT_SEED_REGISTRY["lockbox"],
            fixture_seeds=RECURRENT_SEED_REGISTRY["lockbox"],
            excluded_training_seeds=excluded,
            environment_seed_role=RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE,
        )
        partial_lockbox_plan = RecurrentEvaluationSeedPlan(
            broad_seeds=RECURRENT_SEED_REGISTRY["lockbox"][:2],
            fixture_seeds=RECURRENT_SEED_REGISTRY["lockbox"][:2],
            excluded_training_seeds=excluded,
            environment_seed_role=RECURRENT_EVALUATION_LOCKBOX_SEED_ROLE,
        )
        candidate_validation_plan = RecurrentEvaluationSeedPlan(
            broad_seeds=RECURRENT_SEED_REGISTRY["validation"][:2],
            fixture_seeds=RECURRENT_SEED_REGISTRY["validation"][:2],
            excluded_training_seeds=excluded,
            environment_seed_role=RECURRENT_EVALUATION_CANDIDATE_SEED_ROLE,
        )
        development_plan_with_forged_exclusion = RecurrentEvaluationSeedPlan(
            broad_seeds=(train_seed,),
            fixture_seeds=(train_seed,),
            excluded_training_seeds=(curriculum_seed,),
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "candidate.json"
            save_recurrent_artifact(
                artifact_path,
                model,
                training_config={"algorithm": "test"},
                seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
                source_commit="a" * 40,
                data_metadata=complete_training_provenance,
                run_metadata={"purpose": "seed_integrity_test"},
                learner_seed=20260722,
                learner_device="cpu",
            )

            with (
                patch(
                    "evolution_sim.mind.recurrent_evaluation.load_recurrent_artifact"
                ) as load_artifact,
                patch(
                    "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model"
                ) as lockbox_evaluate,
            ):
                blocked_cases = (
                    (lockbox_plan, "external one-use authorization"),
                    (partial_lockbox_plan, "external one-use authorization"),
                    (candidate_validation_plan, "candidate-freeze"),
                )
                for blocked_plan, error_pattern in blocked_cases:
                    for _attempt in range(2):
                        with self.assertRaisesRegex(
                            RecurrentEvaluationError,
                            error_pattern,
                        ):
                            evaluate_recurrent_artifact(
                                artifact_path,
                                seed_plan=blocked_plan,
                                expected_source_commit="a" * 40,
                            )
            load_artifact.assert_not_called()
            lockbox_evaluate.assert_not_called()

            label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + ("9" * 64)
            with patch(
                "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model"
            ) as in_memory_evaluate:
                with self.assertRaisesRegex(
                    RecurrentEvaluationError,
                    "candidate-freeze",
                ):
                    evaluate_recurrent_model(
                        model,
                        synthetic_noncandidate_digest_label=label,
                        seed_plan=candidate_validation_plan,
                    )
            in_memory_evaluate.assert_not_called()

            with patch(
                "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
                return_value={"result": "development"},
            ) as development_evaluate:
                evaluate_recurrent_artifact(
                    artifact_path,
                    seed_plan=development_plan_with_forged_exclusion,
                    expected_source_commit="a" * 40,
                )
            development_provenance = development_evaluate.call_args.kwargs[
                "candidate_provenance"
            ]
            training_evidence = development_provenance[
                "artifact_training_seed_evidence"
            ]
            self.assertFalse(
                development_provenance["promotion_evidence_eligible_from_provenance"]
            )
            self.assertFalse(
                development_provenance["caller_excluded_training_seeds_used_as_proof"]
            )
            self.assertEqual(
                training_evidence["training_evaluation_overlap_seeds"],
                [train_seed],
            )

    def test_selection_role_with_legacy_training_provenance_is_ineligible(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=20260723,
        )
        train_seed = RECURRENT_SEED_REGISTRY["train"][0]
        selection_plan = RecurrentEvaluationSeedPlan(
            broad_seeds=RECURRENT_SEED_REGISTRY["selection"][:2],
            fixture_seeds=RECURRENT_SEED_REGISTRY["selection"][:2],
            excluded_training_seeds=(train_seed,),
            environment_seed_role=RECURRENT_EVALUATION_SELECTION_SEED_ROLE,
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        artifact_path = Path(temporary.name) / "legacy.json"
        save_recurrent_artifact(
            artifact_path,
            model,
            training_config={"algorithm": "test"},
            seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
            source_commit="a" * 40,
            data_metadata={"training_seeds": [train_seed]},
            run_metadata={"purpose": "seed_integrity_test"},
            learner_seed=20260723,
            learner_device="cpu",
        )
        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
            return_value={"result": "legacy"},
        ) as evaluate:
            evaluate_recurrent_artifact(
                artifact_path,
                seed_plan=selection_plan,
                expected_source_commit="a" * 40,
            )
        provenance = evaluate.call_args.kwargs["candidate_provenance"]
        artifact_evidence = evaluate.call_args.kwargs["artifact_evidence"]
        _validate_source_pinned_artifact_identity(
            artifact_evidence,
            provenance,
            seed_plan=selection_plan,
            trusted_artifact_path=artifact_path,
        )

        self.assertFalse(provenance["promotion_evidence_eligible_from_provenance"])
        self.assertFalse(provenance["external_validation_authorization_available"])
        self.assertEqual(
            artifact_evidence["schema_version"],
            RECURRENT_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(
            artifact_evidence["artifact_evidence_sha256"],
            _source_pinned_artifact_evidence_sha256(artifact_evidence),
        )
        evidence = provenance["artifact_training_seed_evidence"]
        self.assertFalse(evidence["role_bound_provenance_complete"])
        self.assertIn(
            "artifact_role_bound_training_seed_map_missing",
            evidence["failure_reasons"],
        )

        converted_evidence = copy.deepcopy(artifact_evidence)
        converted_evidence.update(
            {
                "schema_version": FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
                "artifact_kind": FROZEN_RECURRENT_POLICY_ARTIFACT_KIND,
                "source_manifest_sha256": "b" * 64,
                "expected_source_manifest_sha256": "b" * 64,
                "source_manifest_match": True,
                "replay_probe_contract_version": (
                    RECURRENT_REPLAY_PROBE_CONTRACT_VERSION
                ),
                "replay_probe_verified_on_device": "cpu",
            }
        )
        converted_provenance = copy.deepcopy(provenance)
        converted_provenance["mode"] = VERIFIED_FROZEN_RECURRENT_POLICY_ARTIFACT_MODE
        converted_provenance["pin_verification"] = (
            "frozen_policy_artifact_v4_registry_source_commit_"
            "source_manifest_and_cpu_probe_v3"
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "artifact evidence SHA256 mismatch",
        ):
            _validate_source_pinned_artifact_identity(
                converted_evidence,
                converted_provenance,
                seed_plan=selection_plan,
                trusted_artifact_path=artifact_path,
            )
        converted_evidence["artifact_evidence_sha256"] = (
            _source_pinned_artifact_evidence_sha256(converted_evidence)
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "artifact bytes are missing, invalid, or stale",
        ):
            _validate_source_pinned_artifact_identity(
                converted_evidence,
                converted_provenance,
                seed_plan=selection_plan,
                trusted_artifact_path=artifact_path,
            )

    def test_verified_frozen_artifact_runs_real_broad_and_carrion_120_tick_controls(
        self,
    ) -> None:
        assert torch is not None
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=8,
                hidden_size=8,
                recurrent_layers=1,
            ),
            initialization_seed=20260721,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(101,),
            fixture_seeds=(103,),
            excluded_training_seeds=(107, 109, 113),
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        artifact_path = Path(temporary.name) / "frozen-candidate.json"
        replay_manifest = {
            "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": "b" * 64,
            "replay_engine_contract_sha256": "c" * 64,
            "environment_seed_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
            "environment_seed_roles": ["development"],
            "scenario_names": ["broad", "carrion_only"],
            "tick_horizons": [120],
            "world_count": 2,
            "replay_verified_world_count": 2,
            "policy_sampling_stream_count": 1,
            "all_replays_exact": True,
            "verification_runner": "unit-test",
            "verification_runner_sha256": "d" * 64,
        }
        artifact = save_frozen_recurrent_policy_artifact(
            artifact_path,
            model,
            training_config={
                "algorithm": "test_untrained_canary",
                "feed_forward_history_ablation": False,
            },
            experiment_config={"arm": "frozen-contract-test"},
            seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
            source_commit="a" * 40,
            source_manifest_sha256="e" * 64,
            data_metadata={"training_seeds": [107, 109, 113]},
            run_metadata={"purpose": "evaluation_contract_test"},
            full_world_replay_manifest=replay_manifest,
            learner_seed=20260721,
            learner_device="cpu",
        )
        report = evaluate_frozen_recurrent_policy_artifact(
            artifact_path,
            seed_plan=plan,
            expected_source_commit="a" * 40,
            expected_source_manifest_sha256="e" * 64,
            expected_seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
            fixture_names=("carrion_only",),
        )

        self.assertEqual(
            report["artifact"]["artifact_sha256"], artifact["artifact_sha256"]
        )
        self.assertEqual(
            report["artifact"]["schema_version"],
            FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["artifact"]["artifact_evidence_sha256"],
            _source_pinned_artifact_evidence_sha256(report["artifact"]),
        )
        self.assertEqual(report["schema_version"], RECURRENT_EVALUATION_SCHEMA_VERSION)
        self.assertEqual(
            report["evaluation_contract"]["ticks"], RECURRENT_EVALUATION_TICKS
        )
        self.assertFalse(
            report["evaluation_contract"]["candidate_factory_receives_seed_or_fixture"]
        )
        self.assertEqual(report["seed_plan"]["train_evaluation_overlap_count"], 0)
        self.assertEqual(report["seed_plan"]["environment_seed_role"], "development")
        self.assertFalse(report["seed_plan"]["full_canonical_lockbox_plan"])
        self.assertFalse(
            report["seed_plan"]["caller_excluded_training_seeds_used_as_proof"]
        )
        self.assertFalse(
            report["candidate_provenance"][
                "promotion_evidence_eligible_from_provenance"
            ]
        )
        self.assertFalse(
            report["candidate_provenance"][
                "external_validation_authorization_available"
            ]
        )
        self.assertTrue(report["replay_verification"]["all_passed"])
        self.assertEqual(report["replay_verification"]["checked_run_count"], 2)

        broad = report["broad"]
        carrion = report["fixtures"][0]
        self.assertEqual(carrion["fixture"], "carrion_only")
        for context in (broad, carrion):
            self.assertEqual(
                set(context["policies"]),
                {"public_recurrent", "mind_v3_linear", "masked_random"},
            )
            for policy in context["policies"].values():
                run = policy["runs"][0]
                aggregate = policy["aggregate"]
                self.assertEqual(run["horizon_ticks"], 120)
                self.assertLessEqual(run["ticks_executed"], 120)
                self.assertEqual(run["unsupported_requested_action_count"], 0)
                self.assertEqual(run["heuristic_action_source_count"], 0)
                self.assertEqual(
                    sum(run["requested_action_counts"].values()),
                    run["policy_decision_record_count"],
                )
                self.assertEqual(
                    run["policy_decision_record_count"]
                    + run["passive_trajectory_record_count"],
                    run["trajectory_record_count"],
                )
                self.assertIn("reward_total", run)
                self.assertIn("dominant_requested_action_share", run)
                self.assertEqual(aggregate["run_count"], 1)
                self.assertEqual(
                    aggregate["terminal_alive_agent_total"],
                    sum(item["terminal_alive"] for item in policy["runs"]),
                )
                self.assertEqual(
                    aggregate["terminal_nonextinct_run_count"],
                    sum(item["terminal_alive"] > 0 for item in policy["runs"]),
                )
                self.assertNotIn("terminal_survivor_run_count", aggregate)
            self.assertEqual(
                len(context["paired_deltas"]["candidate_minus_mind_v3_linear"]["runs"]),
                1,
            )
            self.assertEqual(
                set(
                    context["policies"]["public_recurrent"]["aggregate"][
                        "reward_component_totals"
                    ]
                ),
                set(REWARD_COMPONENT_BOUNDS),
            )
            distribution = context["policies"]["public_recurrent"]["aggregate"][
                "learned_masked_distribution"
            ]
            self.assertGreater(distribution["decision_count"], 0)
            self.assertIsNotNone(distribution["metric_means"]["normalized_entropy"])
            self.assertEqual(
                context["candidate_sampling_analysis"]["environment_count"],
                1,
            )
            self.assertEqual(
                len(context["paired_deltas"]["candidate_minus_masked_random"]["runs"]),
                1,
            )

        candidate_nonextinct_runs = carrion["policies"]["public_recurrent"][
            "aggregate"
        ]["terminal_nonextinct_run_count"]
        candidate_alive_agents = carrion["policies"]["public_recurrent"]["aggregate"][
            "terminal_alive_agent_total"
        ]
        self.assertEqual(
            report["carrion_fixture_terminal_nonextinct_run_count"],
            candidate_nonextinct_runs,
        )
        self.assertEqual(
            report["carrion_fixture_terminal_alive_agent_total"],
            candidate_alive_agents,
        )
        self.assertNotIn("carrion_fixture_terminal_survivor_count", report)
        runtime = report["execution_provenance"]["runtime_reproducibility"]
        self.assertEqual(
            runtime["schema_version"],
            RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION,
        )
        self.assertTrue(runtime["python_version"])
        self.assertTrue(runtime["torch_version"])
        self.assertTrue(runtime["numpy_version"] is None or runtime["numpy_version"])
        self.assertEqual(runtime["requested_device"], "cpu")
        self.assertEqual(runtime["resolved_device"], "cpu")
        self.assertTrue(runtime["platform"]["platform_string"])

        def validate_source_pinned(candidate: Mapping[str, object]) -> None:
            _validate_report(candidate, trusted_artifact_path=artifact_path)

        frozen_v4_report = copy.deepcopy(report)
        validate_source_pinned(frozen_v4_report)
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "explicit trusted artifact path",
        ):
            _validate_report(frozen_v4_report)

        stale_v4_evaluation = copy.deepcopy(frozen_v4_report)
        stale_v4_evaluation["schema_version"] = (
            LEGACY_RECURRENT_EVALUATION_SCHEMA_VERSION
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "schema v4 is stale",
        ):
            validate_source_pinned(stale_v4_evaluation)

        untrusted_embedded_path = copy.deepcopy(frozen_v4_report)
        untrusted_embedded_path["artifact"]["path"] = str(
            Path(temporary.name) / "missing-frozen-candidate.json"
        )
        untrusted_embedded_path["artifact"]["artifact_evidence_sha256"] = (
            _source_pinned_artifact_evidence_sha256(untrusted_embedded_path["artifact"])
        )
        validate_source_pinned(untrusted_embedded_path)
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "artifact bytes are missing, invalid, or stale",
        ):
            _validate_report(
                frozen_v4_report,
                trusted_artifact_path=(
                    Path(temporary.name) / "missing-frozen-candidate.json"
                ),
            )

        unbound_frozen_identity = copy.deepcopy(frozen_v4_report)
        unbound_frozen_identity["artifact"]["replay_probe_verified_on_device"] = "cuda"
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "artifact evidence SHA256 mismatch",
        ):
            validate_source_pinned(unbound_frozen_identity)

        stale_frozen_mode = copy.deepcopy(frozen_v4_report)
        stale_frozen_mode["candidate_provenance"]["mode"] = (
            VERIFIED_FROZEN_RECURRENT_POLICY_ARTIFACT_MODE.replace("_v4", "_v3")
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "mode is missing, stale, or unsupported",
        ):
            validate_source_pinned(stale_frozen_mode)

        stale_frozen_schema = copy.deepcopy(frozen_v4_report)
        stale_frozen_schema["artifact"]["schema_version"] = (
            FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION.replace("_v5", "_v4")
        )
        stale_frozen_schema["artifact"]["artifact_evidence_sha256"] = (
            _source_pinned_artifact_evidence_sha256(stale_frozen_schema["artifact"])
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "identity or schema is missing or stale",
        ):
            validate_source_pinned(stale_frozen_schema)

        stale_cpu_probe = copy.deepcopy(frozen_v4_report)
        stale_cpu_probe["artifact"]["replay_probe_contract_version"] = (
            RECURRENT_REPLAY_PROBE_CONTRACT_VERSION.replace("_v3", "_v2")
        )
        stale_cpu_probe["artifact"]["artifact_evidence_sha256"] = (
            _source_pinned_artifact_evidence_sha256(stale_cpu_probe["artifact"])
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "CPU replay-probe evidence is missing or stale",
        ):
            validate_source_pinned(stale_cpu_probe)

        stale_v3_report = copy.deepcopy(report)
        stale_v3_report["carrion_fixture_terminal_survivor_count"] = (
            candidate_nonextinct_runs
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "ambiguous v3 terminal fields",
        ):
            validate_source_pinned(stale_v3_report)

        forged_eligibility = copy.deepcopy(report)
        forged_eligibility["candidate_provenance"][
            "promotion_evidence_eligible_from_provenance"
        ] = True
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "promotion eligibility differs",
        ):
            validate_source_pinned(forged_eligibility)

        contract_ticks = copy.deepcopy(report)
        contract_ticks["evaluation_contract"]["ticks"] = 999
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "canonical contract drifted",
        ):
            validate_source_pinned(contract_ticks)

        factory_seed_leak = copy.deepcopy(report)
        factory_seed_leak["evaluation_contract"][
            "candidate_factory_receives_seed_or_fixture"
        ] = True
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "canonical contract drifted",
        ):
            validate_source_pinned(factory_seed_leak)

        erased_paired_deltas = copy.deepcopy(report)
        erased_paired_deltas["broad"]["paired_deltas"] = {}
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "paired deltas differ",
        ):
            validate_source_pinned(erased_paired_deltas)

        forged_candidate_grid = copy.deepcopy(report)
        forged_grid_run = forged_candidate_grid["broad"]["policies"][
            "public_recurrent"
        ]["runs"][0]
        forged_grid_run["policy_sampling_seed"] = 7
        forged_grid_run["outcome_evidence_sha256"] = _run_outcome_evidence_sha256(
            forged_grid_run
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "candidate identity grid differs",
        ):
            validate_source_pinned(forged_candidate_grid)

        forged_replay_digest = copy.deepcopy(report)
        forged_replay_digest["replay_verification"]["checks"][0]["digest"] = "0" * 64
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "replay identities or digests differ",
        ):
            validate_source_pinned(forged_replay_digest)

        coherently_rehashed_outcome = copy.deepcopy(report)
        rehashed_run = coherently_rehashed_outcome["broad"]["policies"][
            "public_recurrent"
        ]["runs"][0]
        rehashed_run["deaths"] += 1
        rehashed_run["outcome_evidence_sha256"] = _run_outcome_evidence_sha256(
            rehashed_run
        )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "replay identities or digests differ",
        ):
            validate_source_pinned(coherently_rehashed_outcome)

    def test_passive_terminal_stay_is_not_counted_as_a_policy_action(self) -> None:
        reward_components = {component: 0.0 for component in REWARD_COMPONENT_BOUNDS}

        class FakeWorld:
            trajectory_records = [
                {
                    "requested_action": "eat",
                    "action_source": "test_policy",
                    "policy_id": "test_policy",
                    "action_valid": True,
                    "reward": {
                        "total": 0.25,
                        "components": dict(reward_components),
                    },
                },
                {
                    "requested_action": "stay",
                    "action_source": "passive",
                    "policy_id": None,
                    "action_valid": True,
                    "reward": {
                        "total": -1.0,
                        "components": dict(reward_components),
                    },
                },
            ]
            policy_decision_diagnostics_records = [None, None]

            def run(self, **_: object) -> object:
                class Result:
                    summary = {
                        "ticks_executed": 1,
                        "alive_agents": 0,
                        "births": 0,
                        "deaths": 1,
                    }

                return Result()

        with patch(
            "evolution_sim.mind.recurrent_evaluation.SimulationWorld",
            return_value=FakeWorld(),
        ):
            run = _run_policy_world(
                seed=2,
                fixture_name=None,
                policy=object(),
            )

        self.assertEqual(run["trajectory_record_count"], 2)
        self.assertEqual(run["policy_decision_record_count"], 1)
        self.assertEqual(run["passive_trajectory_record_count"], 1)
        self.assertEqual(run["requested_action_counts"], {"eat": 1})
        self.assertEqual(run["dominant_requested_action"], "eat")
        self.assertEqual(run["dominant_requested_action_share"], 1.0)
        self.assertEqual(run["eat_requested_count"], 1)
        self.assertEqual(run["reward_total"], -0.75)

        nonextinct_run = dict(run)
        nonextinct_run["terminal_alive"] = 3
        aggregate = _aggregate_runs((nonextinct_run, run))
        self.assertEqual(aggregate["run_count"], 2)
        self.assertEqual(aggregate["terminal_nonextinct_run_count"], 1)
        self.assertEqual(aggregate["terminal_alive_agent_total"], 3)
        self.assertEqual(aggregate["terminal_alive_mean"], 1.5)
        self.assertNotIn("terminal_survivor_run_count", aggregate)

    def test_conditioned_evaluation_binding_fails_before_world_execution(
        self,
    ) -> None:
        conditioned = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=4,
                hidden_size=4,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            ),
            initialization_seed=20260727,
        )
        disabled = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=4, hidden_size=4),
            initialization_seed=20260727,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(151,),
            fixture_seeds=(157,),
            excluded_training_seeds=(163,),
        )
        label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + ("1" * 64)

        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model"
        ) as evaluator:
            with self.assertRaisesRegex(
                RecurrentEvaluationError,
                "requires an explicit heritable or zero_all",
            ):
                evaluate_recurrent_model(
                    conditioned,
                    synthetic_noncandidate_digest_label=label,
                    seed_plan=plan,
                    fixture_names=("carrion_only",),
                )
            evaluator.assert_not_called()

        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "unsigned 64-bit",
        ):
            _validated_evaluation_genome_population_configuration(
                conditioned,
                genome_population_mode="heritable",
                genome_stream_seed=None,
            )
        with self.assertRaisesRegex(
            RecurrentEvaluationError,
            "forbids evaluation genome binding",
        ):
            _validated_evaluation_genome_population_configuration(
                disabled,
                genome_population_mode="zero_all",
                genome_stream_seed=17,
            )

        with patch(
            "evolution_sim.mind.recurrent_evaluation._evaluate_frozen_model",
            return_value={"bound": True},
        ) as evaluator:
            report = evaluate_recurrent_model(
                conditioned,
                synthetic_noncandidate_digest_label=label,
                seed_plan=plan,
                fixture_names=("carrion_only",),
                genome_population_mode="heritable",
                genome_stream_seed=2**64 - 1,
            )
        self.assertEqual(report, {"bound": True})
        call = evaluator.call_args.kwargs
        self.assertEqual(call["genome_population_mode"], "heritable")
        self.assertEqual(call["genome_stream_seed"], 2**64 - 1)
        genome_evaluation = call["candidate_provenance"]["genome_evaluation"]
        self.assertEqual(
            genome_evaluation["schema_version"],
            RECURRENT_EVALUATION_GENOME_POPULATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            genome_evaluation["genome_conditioning_mode"],
            GENOME_CONDITIONING_ACTOR_FILM_V1,
        )

    def test_disabled_evaluation_default_and_explicit_binding_are_exactly_equal(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=4, hidden_size=4),
            initialization_seed=919,
        )

        def run(*, explicit: bool) -> dict[str, object]:
            policy = DeterministicPublicRecurrentPolicy(
                model,
                artifact_digest="disabled-evaluation-compatibility",
            )
            kwargs: dict[str, object] = {}
            if explicit:
                kwargs = {
                    "genome_population_mode": "disabled",
                    "genome_stream_seed": None,
                }
            with patch(
                "evolution_sim.mind.recurrent_evaluation.RECURRENT_EVALUATION_TICKS",
                2,
            ):
                return _run_policy_world(
                    seed=151,
                    fixture_name=None,
                    policy=policy,
                    **kwargs,
                )

        implicit = run(explicit=False)
        explicit = run(explicit=True)
        self.assertEqual(implicit, explicit)
        self.assertNotIn("genome_population_provenance", implicit)
        self.assertEqual(
            set(implicit),
            {
                "context",
                "seed",
                "policy_sampling_seed",
                "horizon_ticks",
                "ticks_executed",
                "terminal_alive",
                "births",
                "deaths",
                "reward_total",
                "reward_component_totals",
                "trajectory_record_count",
                "policy_decision_record_count",
                "passive_trajectory_record_count",
                "requested_action_counts",
                "dominant_requested_action",
                "dominant_requested_action_count",
                "dominant_requested_action_share",
                "unsupported_requested_action_count",
                "heuristic_action_source_count",
                "action_source_counts",
                "policy_id_counts",
                "eat_requested_count",
                "eat_without_positive_resource_gain_count",
                "eat_without_positive_resource_gain_share",
                "learned_masked_distribution",
                "behavior_digest",
                "replay_digest",
                "outcome_evidence_sha256",
            },
        )

    def test_actor_film_frozen_artifact_replays_heritable_and_zero_all_exactly(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=4,
                hidden_size=4,
                genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
            ),
            initialization_seed=20260728,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(151,),
            fixture_seeds=(157,),
            excluded_training_seeds=(163, 167),
        )
        replay_manifest = {
            "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": "2" * 64,
            "replay_engine_contract_sha256": "3" * 64,
            "environment_seed_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
            "environment_seed_roles": ["development"],
            "scenario_names": ["broad", "carrion_only"],
            "tick_horizons": [120],
            "world_count": 2,
            "replay_verified_world_count": 2,
            "policy_sampling_stream_count": 1,
            "all_replays_exact": True,
            "verification_runner": "conditioned-unit-test",
            "verification_runner_sha256": "4" * 64,
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        artifact_path = Path(temporary.name) / "conditioned-policy.json"
        artifact = save_frozen_recurrent_policy_artifact(
            artifact_path,
            model,
            training_config={
                "algorithm": "conditioned_evaluation_test",
                "feed_forward_history_ablation": False,
            },
            experiment_config={"arm": "conditioned-contract-test"},
            seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
            source_commit="5" * 40,
            source_manifest_sha256="6" * 64,
            data_metadata={"training_seeds": [163, 167]},
            run_metadata={"development_only": True},
            full_world_replay_manifest=replay_manifest,
            learner_seed=20260728,
            learner_device="cpu",
        )

        def evaluate(mode: str) -> dict[str, object]:
            with patch(
                "evolution_sim.mind.recurrent_evaluation.RECURRENT_EVALUATION_TICKS",
                2,
            ):
                return evaluate_frozen_recurrent_policy_artifact(
                    artifact_path,
                    seed_plan=plan,
                    expected_source_commit="5" * 40,
                    expected_source_manifest_sha256="6" * 64,
                    expected_seed_registry_digest=(CANONICAL_SEED_REGISTRY_SHA256),
                    fixture_names=("carrion_only",),
                    genome_population_mode=mode,
                    genome_stream_seed=1776,
                )

        heritable = evaluate("heritable")
        repeated_heritable = evaluate("heritable")
        zero_all = evaluate("zero_all")
        repeated_zero_all = evaluate("zero_all")

        self.assertEqual(heritable, repeated_heritable)
        self.assertEqual(zero_all, repeated_zero_all)
        self.assertEqual(
            heritable["artifact"]["artifact_sha256"],
            artifact["artifact_sha256"],
        )
        self.assertEqual(heritable["replay_verification"]["checked_run_count"], 2)
        self.assertTrue(heritable["replay_verification"]["all_passed"])
        self.assertIn(
            "inherited_controller_genome",
            heritable["evaluation_contract"]["candidate_policy_inputs"],
        )
        heritable_runs = [
            heritable["broad"]["policies"]["public_recurrent"]["runs"][0],
            heritable["fixtures"][0]["policies"]["public_recurrent"]["runs"][0],
        ]
        zero_runs = [
            zero_all["broad"]["policies"]["public_recurrent"]["runs"][0],
            zero_all["fixtures"][0]["policies"]["public_recurrent"]["runs"][0],
        ]
        for mode, runs in (("heritable", heritable_runs), ("zero_all", zero_runs)):
            for run in runs:
                provenance = run["genome_population_provenance"]
                self.assertEqual(
                    provenance["schema_version"],
                    RECURRENT_GENOME_WORLD_PROVENANCE_SCHEMA_VERSION,
                )
                self.assertEqual(provenance["genome_population_mode"], mode)
                self.assertEqual(provenance["genome_stream_seed"], 1776)
                self.assertEqual(
                    provenance["genome_population_pre_founder_state_sha256"],
                    provenance["genome_population_reset_state_sha256"],
                )
        self.assertNotEqual(
            heritable_runs[0]["replay_digest"],
            zero_runs[0]["replay_digest"],
        )
        self.assertNotEqual(
            heritable_runs[0]["outcome_evidence_sha256"],
            zero_runs[0]["outcome_evidence_sha256"],
        )

        tampered = copy.deepcopy(heritable)
        tampered_run = tampered["broad"]["policies"]["public_recurrent"]["runs"][0]
        tampered_provenance = tampered_run["genome_population_provenance"]
        tampered_provenance["genome_population_mode"] = "zero_all"
        unsigned = dict(tampered_provenance)
        unsigned.pop("provenance_sha256")
        tampered_provenance["provenance_sha256"] = _canonical_sha256(unsigned)
        tampered_run["outcome_evidence_sha256"] = _run_outcome_evidence_sha256(
            tampered_run
        )
        with (
            patch(
                "evolution_sim.mind.recurrent_evaluation.RECURRENT_EVALUATION_TICKS",
                2,
            ),
            self.assertRaisesRegex(
                RecurrentEvaluationError,
                "binding differs from its task",
            ),
        ):
            _validate_report(tampered, trusted_artifact_path=artifact_path)

    def test_spawn_parallel_evaluation_is_exactly_sequential_equivalent(
        self,
    ) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=4,
                hidden_size=4,
                recurrent_layers=1,
            ),
            initialization_seed=20260722,
        )
        plan = RecurrentEvaluationSeedPlan(
            broad_seeds=(127,),
            fixture_seeds=(131,),
            excluded_training_seeds=(137, 139),
        )
        label = UNPINNED_NONCANDIDATE_DIGEST_PREFIX + ("f" * 64)
        common = {
            "synthetic_noncandidate_digest_label": label,
            "seed_plan": plan,
            "fixture_names": ("carrion_only",),
            "candidate_action_selection": PUBLIC_RECURRENT_SAMPLED_SELECTION,
            "candidate_sampling_seed_count": 2,
            "candidate_sampling_stream_id": "parallel-equivalence:test",
        }

        sequential = evaluate_recurrent_model(
            model,
            **common,
            evaluation_workers=1,
        )
        parallel = evaluate_recurrent_model(
            model,
            **common,
            evaluation_workers=2,
        )

        sequential_execution = sequential.pop("execution_provenance")
        parallel_execution = parallel.pop("execution_provenance")
        self.assertEqual(sequential, parallel)
        self.assertFalse(sequential_execution["process_parallel"])
        self.assertEqual(sequential_execution["evaluation_workers_used"], 1)
        self.assertTrue(parallel_execution["process_parallel"])
        self.assertEqual(parallel_execution["evaluation_workers_requested"], 2)
        self.assertEqual(parallel_execution["evaluation_workers_used"], 2)
        self.assertEqual(parallel_execution["process_start_method"], "spawn")
        self.assertEqual(parallel_execution["torch_threads_per_worker"], 1)
        for execution in (sequential_execution, parallel_execution):
            runtime = execution["runtime_reproducibility"]
            self.assertEqual(
                runtime["schema_version"],
                RECURRENT_EVALUATION_RUNTIME_SCHEMA_VERSION,
            )
            self.assertEqual(runtime["requested_device"], "cpu")
            self.assertEqual(runtime["resolved_device"], "cpu")
            self.assertTrue(runtime["python_version"])
            self.assertTrue(runtime["torch_version"])
            self.assertIn("system", runtime["platform"])


if __name__ == "__main__":
    unittest.main()
