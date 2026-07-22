from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    import evolution_sim.mind.recurrent_counterfactual_branch as branch_module
    from evolution_sim.config import WorldConfig
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.env.runtime.state import RunMode
    from evolution_sim.env.world import SimulationWorld
    from evolution_sim.mind.recurrent_actor_critic import (
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
        CounterfactualHorizonScalarization,
        RecurrentCounterfactualAuxiliaryConfig,
        build_recurrent_counterfactual_auxiliary_target,
    )
    from evolution_sim.mind.recurrent_counterfactual_collection import (
        RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE,
        RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE,
        RecurrentCounterfactualCollectionConfig,
        RecurrentCounterfactualCollectionError,
        RecurrentCounterfactualCollectionTask,
        collect_recurrent_counterfactual_bundles,
        derive_recurrent_counterfactual_collection_seed,
        recurrent_counterfactual_collection_result_payload,
        validate_recurrent_counterfactual_collection_result,
    )
    from evolution_sim.mind.recurrent_counterfactual_branch import (
        build_recurrent_counterfactual_branch_row,
        validate_recurrent_counterfactual_branch_row,
    )
    from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
    from evolution_sim.mind.recurrent_policy import (
        DeterministicPublicRecurrentPolicy,
    )
    from evolution_sim.mind.recurrent_seed_registry import RECURRENT_SEED_REGISTRY


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentCounterfactualCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        cls.artifact_digest = "6" * 64
        cls.model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(
                encoder_size=12,
                hidden_size=12,
                recurrent_layers=1,
            ),
            initialization_seed=83,
        )
        cls.config = RecurrentCounterfactualCollectionConfig(
            horizons=(1, 2),
            gamma=0.99,
        )
        curriculum_seed = RECURRENT_SEED_REGISTRY["curriculum"][0]
        cls.tasks = (
            RecurrentCounterfactualCollectionTask(
                task_id="nested-collection-a",
                seed_role="curriculum",
                scenario="carrion_only",
                environment_seed=curriculum_seed,
                branch_tick_candidates=(0,),
                source_policy_sampling_identity="nested-collection-a:source",
                branch_selection_identity="nested-collection-a:selection",
            ),
            RecurrentCounterfactualCollectionTask(
                task_id="nested-collection-b",
                seed_role="curriculum",
                scenario="carrion_only",
                environment_seed=curriculum_seed,
                branch_tick_candidates=(0,),
                source_policy_sampling_identity="nested-collection-b:source",
                branch_selection_identity="nested-collection-b:selection",
            ),
        )
        cls.parameter_copies = [
            parameter.detach().clone() for parameter in cls.model.parameters()
        ]
        cls.sequential = collect_recurrent_counterfactual_bundles(
            cls.model,
            cls.tasks,
            artifact_digest=cls.artifact_digest,
            config=cls.config,
            workers=1,
        )

    def test_fresh_frozen_source_rows_cover_all_valid_actions_and_horizons(
        self,
    ) -> None:
        result = self.sequential
        validate_recurrent_counterfactual_collection_result(
            result,
            model=self.model,
            artifact_digest=self.artifact_digest,
        )
        self.assertEqual(
            result.source_model_state_sha256,
            recurrent_model_state_sha256(self.model),
        )
        self.assertEqual(result.config.horizons, (1, 2))
        for observed, expected in zip(
            self.model.parameters(),
            self.parameter_copies,
            strict=True,
        ):
            torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        for bundle in result.bundles:
            self.assertEqual(bundle.horizons, (1, 2))
            self.assertEqual(len(bundle.rows), 2)
            contexts = []
            for horizon, row in zip(bundle.horizons, bundle.rows, strict=True):
                validate_recurrent_counterfactual_branch_row(row)
                self.assertEqual(row["metadata"]["horizon_ticks"], horizon)
                self.assertEqual(
                    row["optimizer_context"]["source_model_state_sha256"],
                    result.source_model_state_sha256,
                )
                self.assertEqual(
                    row["optimizer_context"]["source_artifact_digest"],
                    self.artifact_digest,
                )
                contexts.append(row["trainable_public_context"])
                mask = row["trainable_public_context"]["current_public_action_mask"]
                expected_actions = tuple(
                    action for action in ACTION_NAMES if mask[action]
                )
                self.assertEqual(expected_actions, bundle.valid_actions)
                outcomes = row["labels"]["action_outcomes"]
                self.assertEqual(
                    tuple(outcome["action"] for outcome in outcomes),
                    expected_actions,
                )
                for outcome in outcomes:
                    self.assertTrue(outcome["replay_verified"])
                    self.assertEqual(
                        outcome["evidence_digest"],
                        outcome["replay_evidence_digest"],
                    )
            self.assertEqual(contexts[0], contexts[1])
            self.assertFalse(bundle.selection["outcome_or_future_data_used"])
            self.assertEqual(
                bundle.compute["selected_continuation_checkpoint_count"],
                1,
            )
        self.assertFalse(result.training_ran)
        self.assertFalse(result.runtime_action_selection_changed)
        self.assertFalse(result.promotion_authorized)

    def test_prefix_proof_is_exact_for_baseline_actions_and_replays(self) -> None:
        for bundle in self.sequential.bundles:
            proof = bundle.prefix_proof
            expected_streams = 1 + 2 * len(bundle.valid_actions)
            self.assertEqual(proof["stream_count"], expected_streams)
            self.assertTrue(proof["all_shorter_streams_are_exact_max_stream_prefixes"])
            for stream in proof["streams"].values():
                short = stream["horizons"]["1"]
                maximum = stream["horizons"]["2"]
                for field in (
                    "causal_record_digest_sequence",
                    "policy_record_digest_sequence",
                    "diagnostics_digest_sequence",
                ):
                    self.assertEqual(
                        short[field],
                        maximum[field][: len(short[field])],
                    )

    def test_nested_rows_are_directly_consumable_as_one_auxiliary_group(self) -> None:
        bundle = self.sequential.bundles[0]
        config = RecurrentCounterfactualAuxiliaryConfig(
            scalarization=CounterfactualHorizonScalarization(
                horizon_weights=((1, 0.5), (2, 0.5)),
            ),
        )
        target = build_recurrent_counterfactual_auxiliary_target(
            self.model,
            bundle.rows,
            artifact_digest=self.artifact_digest,
            config=config,
        )
        self.assertEqual(target.horizons, (1, 2))
        self.assertTrue(torch.isfinite(target.soft_policy_target).all())
        self.assertAlmostEqual(float(target.soft_policy_target.sum()), 1.0, places=6)

    def test_nested_rows_match_independent_single_horizon_v3_queries(self) -> None:
        bundle = self.sequential.bundles[0]
        for nested_row in bundle.rows:
            metadata = nested_row["metadata"]
            independent = build_recurrent_counterfactual_branch_row(
                self.model,
                artifact_digest=self.artifact_digest,
                seed_role=bundle.task.seed_role,
                environment_seed=bundle.task.environment_seed,
                scenario=bundle.task.scenario,
                branch_tick=metadata["branch_tick"],
                horizon_ticks=metadata["horizon_ticks"],
                focal_agent_id=metadata["focal_agent_id"],
                policy_sampling_seed=bundle.task.source_policy_sampling_seed,
                gamma=self.config.gamma,
                verify_replay=True,
            )
            self.assertEqual(nested_row, independent)

    def test_sequential_and_spawn_parallel_bundles_are_exactly_equal(self) -> None:
        parallel = collect_recurrent_counterfactual_bundles(
            self.model,
            self.tasks,
            artifact_digest=self.artifact_digest,
            config=self.config,
            workers=2,
        )
        self.assertEqual(
            [bundle.task.task_id for bundle in parallel.bundles],
            [task.task_id for task in self.tasks],
        )
        self.assertEqual(
            [bundle.exact_digest for bundle in parallel.bundles],
            [bundle.exact_digest for bundle in self.sequential.bundles],
        )
        self.assertEqual(
            [bundle.rows for bundle in parallel.bundles],
            [bundle.rows for bundle in self.sequential.bundles],
        )
        self.assertEqual(parallel.aggregate_compute, self.sequential.aggregate_compute)
        self.assertEqual(parallel.workers_resolved, 2)
        self.assertEqual(self.sequential.workers_resolved, 1)

    def test_live_mps_model_freezes_to_exact_cpu_source(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        model = deepcopy(self.model).to("mps")
        result = collect_recurrent_counterfactual_bundles(
            model,
            [self.tasks[0]],
            artifact_digest=self.artifact_digest,
            config=self.config,
            workers=1,
        )
        self.assertEqual(
            result.source_model_state_sha256,
            recurrent_model_state_sha256(model),
        )
        validate_recurrent_counterfactual_collection_result(result, model=model)

    def test_environment_source_and_selection_seed_contracts_are_separate(self) -> None:
        for task in self.tasks:
            expected_source = derive_recurrent_counterfactual_collection_seed(
                namespace=RECURRENT_COUNTERFACTUAL_SOURCE_SAMPLING_SEED_NAMESPACE,
                identity=task.source_policy_sampling_identity,
            )
            expected_selection = derive_recurrent_counterfactual_collection_seed(
                namespace=RECURRENT_COUNTERFACTUAL_BRANCH_SELECTION_SEED_NAMESPACE,
                identity=task.branch_selection_identity,
            )
            self.assertEqual(task.source_policy_sampling_seed, expected_source)
            self.assertEqual(task.branch_selection_seed, expected_selection)
            self.assertEqual(
                len(
                    {
                        task.environment_seed,
                        task.source_policy_sampling_seed,
                        task.branch_selection_seed,
                    }
                ),
                3,
            )
            bundle = next(
                bundle
                for bundle in self.sequential.bundles
                if bundle.task.task_id == task.task_id
            )
            row = bundle.rows[0]
            self.assertEqual(row["metadata"]["environment_seed"], task.environment_seed)
            self.assertEqual(
                row["metadata"]["policy_sampling_seed"],
                task.source_policy_sampling_seed,
            )
            self.assertNotIn(
                "branch_selection_seed",
                row["trainable_public_context"],
            )

    def test_compute_budget_is_exact_and_strictly_better_than_independent_horizons(
        self,
    ) -> None:
        for bundle in self.sequential.bundles:
            action_count = len(bundle.valid_actions)
            continuation_count = 1 + 2 * action_count
            compute = bundle.compute
            self.assertEqual(
                compute["max_horizon_continuation_count"],
                continuation_count,
            )
            self.assertEqual(
                compute["action_horizon_target_count"],
                action_count * len(bundle.horizons),
            )
            self.assertEqual(
                compute["maximum_continuation_tick_budget"],
                continuation_count * max(bundle.horizons),
            )
            self.assertEqual(
                compute["independent_horizon_tick_budget"],
                continuation_count * sum(bundle.horizons),
            )
            self.assertGreater(compute["nested_tick_budget_saved"], 0)
            self.assertLessEqual(
                compute["actual_continuation_tick_count"],
                compute["maximum_continuation_tick_budget"],
            )

    def test_uniform_selection_uses_deterministic_candidate_tick_fallback(self) -> None:
        original = branch_module._eligible_focal_source_records
        calls = 0

        def empty_first_candidate(records, diagnostics):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ()
            return original(records, diagnostics)

        task = RecurrentCounterfactualCollectionTask(
            task_id="fallback",
            seed_role="curriculum",
            scenario="carrion_only",
            environment_seed=RECURRENT_SEED_REGISTRY["curriculum"][0],
            branch_tick_candidates=(0, 1),
            source_policy_sampling_identity="fallback:source",
            branch_selection_identity="fallback:selection",
        )
        with patch.object(
            branch_module,
            "_eligible_focal_source_records",
            side_effect=empty_first_candidate,
        ):
            result = collect_recurrent_counterfactual_bundles(
                self.model,
                [task],
                artifact_digest=self.artifact_digest,
                config=self.config,
                workers=1,
            )
        selection = result.bundles[0].selection
        self.assertEqual(selection["selected_branch_tick"], 1)
        self.assertEqual(selection["fallback_index"], 1)
        self.assertGreater(selection["eligible_candidate_count"], 0)

    def test_nested_materialization_accepts_aligned_passive_none_diagnostics(
        self,
    ) -> None:
        action_mask = {action: action in {"stay", "eat"} for action in ACTION_NAMES}
        records = (
            {
                "agent_id": 3,
                "action_source": "passive",
                "action_mask": action_mask,
            },
            {
                "agent_id": 4,
                "action_source": branch_module.RECURRENT_ROLLOUT_ACTION_SOURCE,
                "action_mask": action_mask,
            },
        )
        recurrent_diagnostics = {"agent_id": 4, "decision_index": 7}

        copied = branch_module._copy_aligned_source_decision_diagnostics(
            records,
            (None, recurrent_diagnostics),
        )
        eligible = branch_module._eligible_focal_source_records(records, copied)

        self.assertEqual(copied, (None, recurrent_diagnostics))
        self.assertEqual(eligible, ((records[1], recurrent_diagnostics),))

    def test_nested_materialization_rejects_recurrent_none_diagnostics(self) -> None:
        action_mask = {action: action in {"stay", "eat"} for action in ACTION_NAMES}
        records = (
            {
                "agent_id": 4,
                "action_source": branch_module.RECURRENT_ROLLOUT_ACTION_SOURCE,
                "action_mask": action_mask,
            },
        )

        with self.assertRaisesRegex(
            branch_module.RecurrentCounterfactualBranchError,
            "source decision diagnostics must be a mapping",
        ):
            branch_module._copy_aligned_source_decision_diagnostics(
                records,
                (None,),
            )

    def test_nested_materialization_rejects_diagnostics_on_passive_rows(self) -> None:
        records = (
            {
                "agent_id": 3,
                "action_source": "passive",
                "action_mask": {action: action == "stay" for action in ACTION_NAMES},
            },
        )

        with self.assertRaisesRegex(
            branch_module.RecurrentCounterfactualBranchError,
            "passive source decision diagnostics must be absent",
        ):
            branch_module._copy_aligned_source_decision_diagnostics(
                records,
                ({"agent_id": 3},),
            )

    def test_source_materialization_rejects_unknown_action_sources(self) -> None:
        action_mask = {action: action in {"stay", "eat"} for action in ACTION_NAMES}
        unknown_record = {
            "agent_id": 3,
            "action_source": "unknown_source",
            "action_mask": action_mask,
        }
        diagnostic = {"agent_id": 3, "decision_index": 7}

        for operation in (
            lambda: branch_module._copy_aligned_source_decision_diagnostics(
                (unknown_record,),
                (diagnostic,),
            ),
            lambda: branch_module._eligible_focal_source_records(
                (unknown_record,),
                ("malformed",),
            ),
            lambda: branch_module._select_focal_source_record(
                (unknown_record,),
                (diagnostic,),
                focal_agent_id=None,
            ),
        ):
            with self.assertRaisesRegex(
                branch_module.RecurrentCounterfactualBranchError,
                "not canonical recurrent or passive",
            ):
                operation()

    def test_single_horizon_selection_rejects_diagnostics_on_passive_rows(
        self,
    ) -> None:
        passive_record = {
            "agent_id": 3,
            "action_source": "passive",
            "action_mask": {action: action == "stay" for action in ACTION_NAMES},
        }

        with self.assertRaisesRegex(
            branch_module.RecurrentCounterfactualBranchError,
            "passive source decision diagnostics must be absent",
        ):
            branch_module._select_focal_source_record(
                (passive_record,),
                ({"agent_id": 3},),
                focal_agent_id=None,
            )

    def test_real_passive_tick_materializes_through_public_nested_query(self) -> None:
        environment_seed = RECURRENT_SEED_REGISTRY["train"][1]
        source_sampling_seed = 5_000_000_004
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=12, hidden_size=12),
            initialization_seed=3,
        )
        policy = DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest=self.artifact_digest,
            sampling_seed=source_sampling_seed,
            capture_public_history=True,
        )
        world = SimulationWorld(
            WorldConfig(seed=environment_seed, max_ticks=10),
            policy=policy,
        )
        world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        tick_records = [
            (record, diagnostic)
            for record, diagnostic in zip(
                world.trajectory_records,
                world.policy_decision_diagnostics_records,
                strict=True,
            )
            if record["tick"] == 9
        ]
        passive_records = [
            (record, diagnostic)
            for record, diagnostic in tick_records
            if record["action_source"] == "passive"
        ]
        self.assertEqual(len(passive_records), 1)
        self.assertIsNone(passive_records[0][1])
        self.assertTrue(
            passive_records[0][0]["outcome"]["passive"]["died_before_action"]
        )

        materialized = (
            branch_module.build_recurrent_counterfactual_nested_horizon_materialization(
                model,
                artifact_digest=self.artifact_digest,
                seed_role="train",
                environment_seed=environment_seed,
                scenario="broad",
                branch_tick_candidates=(9,),
                horizons=(1, 2),
                source_policy_sampling_seed=source_sampling_seed,
                branch_selection_seed=5_000_000_100,
                gamma=0.99,
            )
        )
        self.assertEqual(materialized["selection"]["selected_branch_tick"], 9)
        self.assertTrue(
            materialized["prefix_proof"][
                "all_shorter_streams_are_exact_max_stream_prefixes"
            ]
        )
        for row in materialized["rows"]:
            branch_module.validate_recurrent_counterfactual_branch_row(row)

    def test_tamper_stale_model_and_invalid_seed_provenance_fail_closed(self) -> None:
        tampered = deepcopy(self.sequential)
        stream = next(iter(tampered.bundles[0].prefix_proof["streams"].values()))
        stream["horizons"]["1"]["causal_record_digest_sequence"].append("0" * 64)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "exact maximum prefix|exact digest",
        ):
            validate_recurrent_counterfactual_collection_result(tampered)

        stale_model = deepcopy(self.model)
        with torch.no_grad():
            stale_model.actor.bias.add_(0.01)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "stale",
        ):
            validate_recurrent_counterfactual_collection_result(
                self.sequential,
                model=stale_model,
            )

        task = self.tasks[0]
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "does not match its namespaced identity",
        ):
            RecurrentCounterfactualCollectionTask(
                task_id="bad-seed",
                seed_role=task.seed_role,
                scenario=task.scenario,
                environment_seed=task.environment_seed,
                branch_tick_candidates=(0,),
                source_policy_sampling_identity="bad-seed:source",
                branch_selection_identity="bad-seed:selection",
                source_policy_sampling_seed=7,
            )

    def test_payload_is_closed_diagnostics_only_and_digest_stable(self) -> None:
        first = recurrent_counterfactual_collection_result_payload(self.sequential)
        second = recurrent_counterfactual_collection_result_payload(self.sequential)
        self.assertEqual(first, second)
        self.assertEqual(first["exact_digest"], self.sequential.exact_digest)
        self.assertFalse(first["training_ran"])
        self.assertFalse(first["runtime_action_selection_changed"])
        self.assertFalse(first["promotion_authorized"])
        for bundle in first["bundles"]:
            for row in bundle["rows"]:
                self.assertFalse(row["training_ran"])
                self.assertFalse(row["runtime_action_selection_changed"])
                self.assertFalse(row["metadata"]["private_checkpoint_serialized"])


if __name__ == "__main__":
    unittest.main()
