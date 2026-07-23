from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    import evolution_sim.mind.recurrent_counterfactual_collection as collection_module
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
        RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION,
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
        RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION,
        build_recurrent_counterfactual_branch_row,
        validate_recurrent_counterfactual_aggregate_row,
        validate_recurrent_counterfactual_branch_row,
    )
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
    from evolution_sim.mind.recurrent_policy import (
        DeterministicPublicRecurrentPolicy,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        RECURRENT_SEED_REGISTRY,
        SCALE_DEVELOPMENT_SEED_REGISTRY,
    )


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

    @staticmethod
    def _rehash_aggregate(row: dict[str, object]) -> None:
        row["component_digests"] = {
            field: stable_payload_digest(row[field])
            for field in (
                "target",
                "trainable_public_context",
                "optimizer_context",
                "source_identity",
                "source_behavior",
                "tape_contract",
                "tape_provenance",
                "aggregate",
            )
        }
        row.pop("exact_digest", None)
        row["exact_digest"] = stable_payload_digest(row)

    @staticmethod
    def _rehash_bundle_and_result(
        result: object,
        *,
        bundle_index: int = 0,
    ) -> None:
        bundle = result.bundles[bundle_index]  # type: ignore[attr-defined]
        bundle_payload = collection_module._bundle_payload(bundle)
        bundle_payload.pop("exact_digest")
        object.__setattr__(
            bundle,
            "exact_digest",
            stable_payload_digest(bundle_payload),
        )
        result_payload = collection_module._result_payload_without_validation(result)
        result_payload.pop("exact_digest")
        object.__setattr__(
            result,
            "exact_digest",
            stable_payload_digest(result_payload),
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

    def test_legacy_one_tape_payload_omits_multi_tape_evidence(self) -> None:
        payload = recurrent_counterfactual_collection_result_payload(self.sequential)

        self.assertEqual(
            payload["config"],
            {"horizons": (1, 2), "gamma": 0.99},
        )
        for bundle in payload["bundles"]:
            self.assertNotIn("branch_tick_stratum_index", bundle["task"])
            self.assertNotIn("aggregate_rows", bundle)
            self.assertNotIn("terminal_target", bundle)
            self.assertNotIn("multi_tape_compute", bundle)

    def test_nested_collection_payloads_have_exact_schemas_and_types(self) -> None:
        mutations = (
            (
                "selection extra field",
                lambda bundle: bundle.selection.__setitem__(
                    "undeclared_private_state",
                    {"forbidden": True},
                ),
                "selection field set",
            ),
            (
                "selection bool tick",
                lambda bundle: bundle.selection.__setitem__(
                    "selected_branch_tick",
                    False,
                ),
                "non-negative integer",
            ),
            (
                "prefix proof extra field",
                lambda bundle: bundle.prefix_proof.__setitem__(
                    "undeclared",
                    True,
                ),
                "prefix proof field set",
            ),
            (
                "prefix stream extra field",
                lambda bundle: bundle.prefix_proof["streams"][
                    "baseline"
                ].__setitem__("undeclared", True),
                "prefix stream field set",
            ),
            (
                "prefix horizon extra field",
                lambda bundle: bundle.prefix_proof["streams"]["baseline"][
                    "horizons"
                ]["1"].__setitem__("undeclared", True),
                "horizon prefix proof field set",
            ),
            (
                "prefix horizon bool count",
                lambda bundle: bundle.prefix_proof["streams"]["baseline"][
                    "horizons"
                ]["1"].__setitem__("executed_tick_count", True),
                "positive integer",
            ),
            (
                "compute extra field",
                lambda bundle: bundle.compute.__setitem__("undeclared", 0),
                "compute diagnostic field set",
            ),
            (
                "compute bool count",
                lambda bundle: bundle.compute.__setitem__(
                    "selected_continuation_checkpoint_count",
                    True,
                ),
                "non-negative integer",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name):
                tampered = deepcopy(self.sequential)
                mutate(tampered.bundles[0])
                self._rehash_bundle_and_result(tampered)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualCollectionError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_collection_result(tampered)

        aggregate_compute_tampered = deepcopy(self.sequential)
        aggregate_compute_tampered.aggregate_compute[
            "selected_continuation_checkpoint_count"
        ] = False
        payload = collection_module._result_payload_without_validation(
            aggregate_compute_tampered
        )
        payload.pop("exact_digest")
        object.__setattr__(
            aggregate_compute_tampered,
            "exact_digest",
            stable_payload_digest(payload),
        )
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "non-negative integer",
        ):
            validate_recurrent_counterfactual_collection_result(
                aggregate_compute_tampered
            )

        for field in ("workers_resolved", "torch_threads_per_worker"):
            with self.subTest(result_integer_field=field):
                result_tampered = deepcopy(self.sequential)
                object.__setattr__(result_tampered, field, True)
                payload = collection_module._result_payload_without_validation(
                    result_tampered
                )
                payload.pop("exact_digest")
                object.__setattr__(
                    result_tampered,
                    "exact_digest",
                    stable_payload_digest(payload),
                )
                with self.assertRaisesRegex(
                    RecurrentCounterfactualCollectionError,
                    "positive integer",
                ):
                    validate_recurrent_counterfactual_collection_result(
                        result_tampered
                    )

        horizon_alias = deepcopy(self.sequential)
        alias_bundle = horizon_alias.bundles[0]
        object.__setattr__(alias_bundle, "horizons", (True, 2))
        for stream in alias_bundle.prefix_proof["streams"].values():
            stream["horizons"]["True"] = stream["horizons"].pop("1")
        self._rehash_bundle_and_result(horizon_alias)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "positive integer",
        ):
            validate_recurrent_counterfactual_collection_result(horizon_alias)

        executed_ticks_tampered = deepcopy(self.sequential)
        executed_ticks_tampered.bundles[0].prefix_proof["streams"]["baseline"][
            "horizons"
        ]["1"]["executed_tick_count"] = 999
        self._rehash_bundle_and_result(executed_ticks_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "tick count drifted",
        ):
            validate_recurrent_counterfactual_collection_result(
                executed_ticks_tampered
            )

        for field, value, expected_error in (
            ("source_tick_executions", 999, "source ticks drifted"),
            (
                "actual_continuation_tick_count",
                0,
                "actual continuation ticks drifted",
            ),
        ):
            with self.subTest(compute_field=field):
                compute_tampered = deepcopy(self.sequential)
                compute_tampered.bundles[0].compute[field] = value
                object.__setattr__(
                    compute_tampered,
                    "aggregate_compute",
                    collection_module._aggregate_compute(
                        compute_tampered.bundles
                    ),
                )
                self._rehash_bundle_and_result(compute_tampered)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualCollectionError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_collection_result(
                        compute_tampered
                    )

    def test_prefix_sequence_lengths_cannot_shrink_across_horizons(self) -> None:
        three_horizon = collect_recurrent_counterfactual_bundles(
            self.model,
            [self.tasks[0]],
            artifact_digest=self.artifact_digest,
            config=RecurrentCounterfactualCollectionConfig(
                horizons=(1, 2, 3),
                gamma=0.99,
            ),
            workers=1,
        )
        bundle = three_horizon.bundles[0]
        middle = bundle.prefix_proof["streams"]["baseline"]["horizons"]["2"]
        maximum = bundle.prefix_proof["streams"]["baseline"]["horizons"]["3"]
        for field in (
            "causal_record_digest_sequence",
            "policy_record_digest_sequence",
            "diagnostics_digest_sequence",
        ):
            middle[field] = maximum[field][:1]
        self._rehash_bundle_and_result(three_horizon)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "sequence length",
        ):
            validate_recurrent_counterfactual_collection_result(three_horizon)

    def test_multi_tape_collection_is_deterministic_and_strictly_validated(
        self,
    ) -> None:
        scale_task = RecurrentCounterfactualCollectionTask(
            task_id="scale-multi-tape",
            seed_role="scale_curriculum",
            scenario="carrion_only",
            environment_seed=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_curriculum"][0],
            branch_tick_candidates=(0, 1),
            source_policy_sampling_identity="scale-multi-tape:source",
            branch_selection_identity="scale-multi-tape:selection",
            branch_tick_stratum_index=1,
        )
        config = RecurrentCounterfactualCollectionConfig(
            horizons=(1, 2),
            gamma=0.99,
            continuation_tape_count=2,
            terminal_target_world_tick=3,
            uncertainty_penalty=0.5,
        )
        first = collect_recurrent_counterfactual_bundles(
            self.model,
            [scale_task],
            artifact_digest=self.artifact_digest,
            config=config,
            workers=1,
        )
        second = collect_recurrent_counterfactual_bundles(
            self.model,
            [scale_task],
            artifact_digest=self.artifact_digest,
            config=config,
            workers=1,
        )

        self.assertEqual(
            first.contract_version,
            RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION,
        )
        self.assertEqual(first.config.continuation_tape_count, 2)
        self.assertEqual(first.config.terminal_target_world_tick, 3)
        self.assertEqual(first.config.uncertainty_penalty, 0.5)
        self.assertEqual(
            recurrent_counterfactual_collection_result_payload(first)["config"],
            {
                "horizons": (1, 2),
                "gamma": 0.99,
                "continuation_tape_count": 2,
                "terminal_target_world_tick": 3,
                "uncertainty_penalty": 0.5,
            },
        )
        self.assertEqual(first.exact_digest, second.exact_digest)
        self.assertEqual(first.bundles[0].exact_digest, second.bundles[0].exact_digest)
        bundle = first.bundles[0]
        for row in bundle.rows:
            self.assertEqual(
                row["schema_version"],
                RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION,
            )
            metadata = row["metadata"]
            self.assertEqual(
                metadata[
                    "source_pre_boundary_environment_rng_state_sha256"
                ],
                bundle.aggregate_rows[0]["source_identity"][
                    "source_pre_boundary_environment_rng_state_sha256"
                ],
            )
            self.assertEqual(
                metadata["source_pre_boundary_policy_sampling_state_sha256"],
                bundle.aggregate_rows[0]["source_identity"][
                    "source_pre_boundary_policy_sampling_state_sha256"
                ],
            )
        self.assertEqual(bundle.selection["selected_branch_tick"], 1)
        self.assertEqual(
            bundle.selection["requested_branch_tick_stratum_index"],
            1,
        )
        self.assertIsNotNone(bundle.aggregate_rows)
        self.assertEqual(len(bundle.aggregate_rows), 2)
        self.assertIsNotNone(bundle.terminal_target)
        self.assertEqual(bundle.terminal_target["target"]["target_world_tick"], 3)
        self.assertEqual(bundle.multi_tape_compute["continuation_tape_count"], 2)
        self.assertEqual(
            bundle.aggregate_rows[0]["source_behavior"]["current_public_action_mask"],
            bundle.rows[0]["trainable_public_context"]["current_public_action_mask"],
        )
        self.assertEqual(
            bundle.aggregate_rows[0]["source_behavior"]["source_behavior_distribution"],
            bundle.rows[0]["labels"]["source_behavior_distribution"],
        )
        validate_recurrent_counterfactual_collection_result(
            first,
            model=self.model,
            artifact_digest=self.artifact_digest,
        )

        multi_schema_mutations = (
            (
                "multi-tape selection extra field",
                lambda bundle: bundle.selection.__setitem__(
                    "undeclared",
                    True,
                ),
                "selection field set",
            ),
            (
                "rotation bool index",
                lambda bundle: bundle.selection.__setitem__(
                    "requested_branch_tick_stratum_index",
                    True,
                ),
                "non-negative integer",
            ),
            (
                "multi-tape compute extra field",
                lambda bundle: bundle.multi_tape_compute.__setitem__(
                    "undeclared",
                    0,
                ),
                "multi-tape compute diagnostic field set",
            ),
            (
                "multi-tape compute bool count",
                lambda bundle: bundle.multi_tape_compute.__setitem__(
                    "continuation_tape_count",
                    True,
                ),
                "non-negative integer",
            ),
        )
        for name, mutate, expected_error in multi_schema_mutations:
            with self.subTest(name=name):
                schema_tampered = deepcopy(first)
                mutate(schema_tampered.bundles[0])
                self._rehash_bundle_and_result(schema_tampered)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualCollectionError,
                    expected_error,
                ):
                    validate_recurrent_counterfactual_collection_result(
                        schema_tampered
                    )

        multi_actual_tampered = deepcopy(first)
        multi_actual_tampered.bundles[0].multi_tape_compute[
            "actual_continuation_tick_count"
        ] = 0
        object.__setattr__(
            multi_actual_tampered,
            "aggregate_compute",
            collection_module._aggregate_compute(
                multi_actual_tampered.bundles
            ),
        )
        self._rehash_bundle_and_result(multi_actual_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "positive integer",
        ):
            validate_recurrent_counterfactual_collection_result(
                multi_actual_tampered
            )

        tampered = deepcopy(first)
        tampered.bundles[0].aggregate_rows[0]["tape_provenance"][0][
            "environment_sampling_seed"
        ] += 1
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "aggregate|seed|exact digest",
        ):
            validate_recurrent_counterfactual_collection_result(tampered)

        scenario_tampered = deepcopy(first)
        object.__setattr__(
            scenario_tampered.bundles[0],
            "task",
            replace(scenario_tampered.bundles[0].task, scenario="plant_only"),
        )
        self._rehash_bundle_and_result(scenario_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "row source identity",
        ):
            validate_recurrent_counterfactual_collection_result(scenario_tampered)

        task_id_tampered = deepcopy(first)
        object.__setattr__(
            task_id_tampered.bundles[0],
            "task",
            replace(task_id_tampered.bundles[0].task, task_id="relabelled-task"),
        )
        self._rehash_bundle_and_result(task_id_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "tape identity",
        ):
            validate_recurrent_counterfactual_collection_result(task_id_tampered)

        for identity_field in (
            "source_policy_sampling_identity",
            "branch_selection_identity",
        ):
            with self.subTest(identity_field=identity_field):
                identity_tampered = deepcopy(first)
                object.__setattr__(
                    identity_tampered.bundles[0].task,
                    identity_field,
                    f"relabelled:{identity_field}",
                )
                self._rehash_bundle_and_result(identity_tampered)
                with self.assertRaisesRegex(
                    RecurrentCounterfactualCollectionError,
                    "namespaced identit",
                ):
                    validate_recurrent_counterfactual_collection_result(
                        identity_tampered
                    )

        gamma_tampered = deepcopy(first)
        object.__setattr__(gamma_tampered.config, "gamma", 0.5)
        self._rehash_bundle_and_result(gamma_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "gamma",
        ):
            validate_recurrent_counterfactual_collection_result(gamma_tampered)

        duplicated_evidence = deepcopy(first)
        duplicate_bundle = deepcopy(duplicated_evidence.bundles[0])
        object.__setattr__(
            duplicate_bundle,
            "task",
            replace(
                duplicate_bundle.task,
                task_id="unique-id-with-duplicated-seed-evidence",
            ),
        )
        object.__setattr__(
            duplicated_evidence,
            "bundles",
            (duplicated_evidence.bundles[0], duplicate_bundle),
        )
        object.__setattr__(
            duplicated_evidence,
            "aggregate_compute",
            collection_module._aggregate_compute(duplicated_evidence.bundles),
        )
        self._rehash_bundle_and_result(duplicated_evidence, bundle_index=1)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "sampling seeds must be unique",
        ):
            validate_recurrent_counterfactual_collection_result(
                duplicated_evidence
            )

        optimizer_tampered = deepcopy(first)
        optimizer_aggregate = optimizer_tampered.bundles[0].aggregate_rows[0]
        optimizer_state = optimizer_aggregate["optimizer_context"][
            "source_recurrent_state"
        ]
        optimizer_state[0][0][0] += 0.125
        optimizer_tensor = torch.tensor(optimizer_state, dtype=torch.float32)
        optimizer_aggregate["optimizer_context"][
            "source_recurrent_state_sha256"
        ] = hashlib.sha256(
            bytes(
                optimizer_tensor.contiguous()
                .view(torch.uint8)
                .reshape(-1)
                .tolist()
            )
        ).hexdigest()
        self._rehash_aggregate(optimizer_aggregate)
        validate_recurrent_counterfactual_aggregate_row(optimizer_aggregate)
        self._rehash_bundle_and_result(optimizer_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "optimizer context",
        ):
            validate_recurrent_counterfactual_collection_result(optimizer_tampered)

        behavior_tampered = deepcopy(first)
        behavior_aggregate = behavior_tampered.bundles[0].aggregate_rows[0]
        logits = behavior_aggregate["source_behavior"][
            "source_behavior_distribution"
        ]["masked_logits"]
        for action, value in logits.items():
            if value is not None:
                logits[action] = value + 1.0
        self._rehash_aggregate(behavior_aggregate)
        validate_recurrent_counterfactual_aggregate_row(behavior_aggregate)
        self._rehash_bundle_and_result(behavior_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "source behavior",
        ):
            validate_recurrent_counterfactual_collection_result(behavior_tampered)

        terminal_behavior_tampered = deepcopy(first)
        terminal_aggregate = terminal_behavior_tampered.bundles[0].terminal_target
        terminal_logits = terminal_aggregate["source_behavior"][
            "source_behavior_distribution"
        ]["masked_logits"]
        for action, value in terminal_logits.items():
            if value is not None:
                terminal_logits[action] = value + 1.0
        self._rehash_aggregate(terminal_aggregate)
        validate_recurrent_counterfactual_aggregate_row(terminal_aggregate)
        self._rehash_bundle_and_result(terminal_behavior_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "source behavior",
        ):
            validate_recurrent_counterfactual_collection_result(
                terminal_behavior_tampered
            )

        checkpoint_tampered = deepcopy(first)
        checkpoint_aggregate = checkpoint_tampered.bundles[0].aggregate_rows[0]
        checkpoint_aggregate["source_identity"][
            "source_checkpoint_identity_sha256"
        ] = "f" * 64
        self._rehash_aggregate(checkpoint_aggregate)
        validate_recurrent_counterfactual_aggregate_row(checkpoint_aggregate)
        self._rehash_bundle_and_result(checkpoint_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "source identity",
        ):
            validate_recurrent_counterfactual_collection_result(checkpoint_tampered)

        pre_boundary_tampered = deepcopy(first)
        pre_boundary_aggregate = pre_boundary_tampered.bundles[0].aggregate_rows[0]
        pre_environment = "e" * 64
        pre_policy = "d" * 64
        pre_boundary_aggregate["source_identity"][
            "source_pre_boundary_environment_rng_state_sha256"
        ] = pre_environment
        pre_boundary_aggregate["source_identity"][
            "source_pre_boundary_policy_sampling_state_sha256"
        ] = pre_policy
        pre_boundary_aggregate["source_identity"][
            "source_checkpoint_identity_sha256"
        ] = "c" * 64
        for tape in pre_boundary_aggregate["tape_provenance"]:
            tape["pre_boundary_environment_rng_state_sha256"] = pre_environment
            tape["pre_boundary_policy_sampling_state_sha256"] = pre_policy
            for outcome in (tape["baseline"], *tape["action_outcomes"]):
                outcome[
                    "pre_boundary_environment_rng_state_sha256"
                ] = pre_environment
                outcome["pre_boundary_policy_sampling_state_sha256"] = pre_policy
        self._rehash_aggregate(pre_boundary_aggregate)
        validate_recurrent_counterfactual_aggregate_row(pre_boundary_aggregate)
        self._rehash_bundle_and_result(pre_boundary_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "source identity",
        ):
            validate_recurrent_counterfactual_collection_result(
                pre_boundary_tampered
            )

        v4_source_tampered = deepcopy(first)
        v4_source_row = v4_source_tampered.bundles[0].rows[0]
        v4_source_row["schema_version"] = (
            branch_module.RECURRENT_COUNTERFACTUAL_BRANCH_SCHEMA_VERSION
        )
        for field in (
            "source_pre_boundary_environment_rng_state_sha256",
            "source_pre_boundary_policy_sampling_state_sha256",
            "source_boundary_identity_sha256",
        ):
            v4_source_row["metadata"].pop(field)
        v4_source_row["component_digests"]["metadata"] = stable_payload_digest(
            v4_source_row["metadata"]
        )
        v4_source_row.pop("exact_digest")
        v4_source_row["exact_digest"] = stable_payload_digest(v4_source_row)
        self._rehash_bundle_and_result(v4_source_tampered)
        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "source branch-row schema",
        ):
            validate_recurrent_counterfactual_collection_result(v4_source_tampered)

    def test_scale_multi_tape_collection_rejects_implicit_tick_strata(self) -> None:
        task = RecurrentCounterfactualCollectionTask(
            task_id="scale-unstratified",
            seed_role="scale_curriculum",
            scenario="carrion_only",
            environment_seed=SCALE_DEVELOPMENT_SEED_REGISTRY["scale_curriculum"][0],
            branch_tick_candidates=(0, 1),
            source_policy_sampling_identity="scale-unstratified:source",
            branch_selection_identity="scale-unstratified:selection",
        )
        config = RecurrentCounterfactualCollectionConfig(
            horizons=(1, 2),
            continuation_tape_count=2,
        )

        with self.assertRaisesRegex(
            RecurrentCounterfactualCollectionError,
            "explicit deterministic stratum index",
        ):
            collect_recurrent_counterfactual_bundles(
                self.model,
                [task],
                artifact_digest=self.artifact_digest,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
