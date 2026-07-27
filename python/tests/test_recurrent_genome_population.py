from __future__ import annotations

import copy
import json
import random
import unittest

from evolution_sim.config import (
    CombatConfig,
    DietMatchingConfig,
    ReproductionConfig,
    WorldConfig,
)
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.policy import ActionDecision
import evolution_sim.env.runtime.reproduction as runtime_reproduction
from evolution_sim.env.runtime.state import empty_mind_inheritance_metadata
from evolution_sim.mind.recurrent_genome import (
    RecurrentGenomeMutationConfig,
    recurrent_genome_artifact,
    zero_recurrent_genome,
)
from evolution_sim.mind.recurrent_genome_population import (
    RECURRENT_GENOME_EVENT_SEED_MAX,
    RECURRENT_GENOME_EVENT_SEED_MIN,
    RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION,
    RECURRENT_GENOME_POPULATION_SNAPSHOT_SCHEMA_VERSION,
    RECURRENT_GENOME_STREAM_NAMESPACE_VERSION,
    RecurrentGenomePopulationError,
    RecurrentGenomePopulationManager,
    RecurrentGenomePopulationMode,
    derive_recurrent_genome_event_seed,
    recurrent_genome_population_contract,
    recurrent_genome_stream_binding_sha256,
)


def _canonical_snapshot_sha256_payload(
    snapshot: dict[str, object],
) -> dict[str, object]:
    payload = copy.deepcopy(snapshot)
    payload.pop("snapshot_sha256")
    return payload


def _logical_sha256(payload: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


class _PopulationPolicy:
    policy_id = "recurrent_genome_population_test_policy"
    policy_version = "recurrent_genome_population_test_policy_v1"

    def __init__(self, manager: RecurrentGenomePopulationManager) -> None:
        self.manager = manager
        self.founder_calls: list[int] = []
        self.child_calls: list[tuple[int, int, int | None]] = []

    def founder_metadata(self, *, agent_id: int) -> dict[str, object]:
        self.founder_calls.append(agent_id)
        return self.manager.founder_metadata(agent_id=agent_id)

    def child_metadata(
        self,
        *,
        child_agent_id: int,
        primary_parent_id: int,
        secondary_parent_id: int | None,
    ) -> dict[str, object]:
        self.child_calls.append(
            (child_agent_id, primary_parent_id, secondary_parent_id)
        )
        return self.manager.child_metadata(
            child_agent_id=child_agent_id,
            primary_parent_id=primary_parent_id,
            secondary_parent_id=secondary_parent_id,
        )

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        del observation, action_mask
        return ActionDecision(
            requested_action="stay",
            source=self.policy_version,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )


class RecurrentGenomePopulationTests(unittest.TestCase):
    def test_disabled_mode_is_exact_empty_metadata_and_stores_no_genome(
        self,
    ) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=0,
            world_identity="disabled-world",
        )
        global_rng_state = random.getstate()

        founder = manager.founder_metadata(agent_id=1)
        child = manager.child_metadata(
            child_agent_id=2,
            primary_parent_id=1,
            secondary_parent_id=None,
        )

        self.assertIs(manager.mode, RecurrentGenomePopulationMode.DISABLED)
        self.assertEqual(founder, empty_mind_inheritance_metadata())
        self.assertEqual(child, empty_mind_inheritance_metadata())
        self.assertEqual(manager.population_size, 0)
        self.assertEqual(manager.snapshot_artifact()["agents"], [])
        self.assertEqual(random.getstate(), global_rng_state)
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "no registered recurrent genome",
        ):
            manager.genome_for_agent(1)

    def test_seed_namespace_is_bound_disjoint_and_rng_free(self) -> None:
        contract = recurrent_genome_population_contract()
        stream_contract = contract["genome_stream"]
        self.assertEqual(
            stream_contract["namespace_version"],
            RECURRENT_GENOME_STREAM_NAMESPACE_VERSION,
        )
        self.assertEqual(
            set(stream_contract["independent_from"]),
            {
                "simulation_environment_rng",
                "policy_action_sampling_rng",
                "learner_rng",
                "torch_rng",
                "python_global_rng",
            },
        )
        random.seed(991)
        global_rng_state = random.getstate()

        first = derive_recurrent_genome_event_seed(
            genome_stream_seed=42,
            world_identity="world-a",
            event_kind="founder",
            agent_id=1,
        )
        repeated = derive_recurrent_genome_event_seed(
            genome_stream_seed=42,
            world_identity="world-a",
            event_kind="founder",
            agent_id=1,
        )
        other_stream = derive_recurrent_genome_event_seed(
            genome_stream_seed=43,
            world_identity="world-a",
            event_kind="founder",
            agent_id=1,
        )
        other_world = derive_recurrent_genome_event_seed(
            genome_stream_seed=42,
            world_identity="world-b",
            event_kind="founder",
            agent_id=1,
        )

        self.assertEqual(first, repeated)
        self.assertGreaterEqual(first, RECURRENT_GENOME_EVENT_SEED_MIN)
        self.assertLessEqual(first, RECURRENT_GENOME_EVENT_SEED_MAX)
        self.assertNotEqual(first, 42)
        self.assertNotEqual(first, other_stream)
        self.assertNotEqual(first, other_world)
        self.assertEqual(random.getstate(), global_rng_state)
        self.assertEqual(first, 15734785943120834184)
        self.assertEqual(
            recurrent_genome_stream_binding_sha256(
                genome_stream_seed=42,
                world_identity="world-a",
            ),
            "99ff7ce9282f22d6615b467590653492ad2596eeb327458b15f24e7ca2f365f9",
        )

    def test_founders_are_identity_derived_and_sequential_order_independent(
        self,
    ) -> None:
        first = RecurrentGenomePopulationManager(
            genome_stream_seed=20260727,
            world_identity="ordering-world",
            mode="heritable",
        )
        second = RecurrentGenomePopulationManager(
            genome_stream_seed=20260727,
            world_identity="ordering-world",
            mode=RecurrentGenomePopulationMode.HERITABLE,
        )

        first_metadata = {
            agent_id: first.founder_metadata(agent_id=agent_id)
            for agent_id in (1, 2, 3)
        }
        second_metadata = {
            agent_id: second.founder_metadata(agent_id=agent_id)
            for agent_id in (3, 1, 2)
        }
        first_children = {
            child_id: first.child_metadata(
                child_agent_id=child_id,
                primary_parent_id=1,
                secondary_parent_id=2,
            )
            for child_id in (4, 5)
        }
        second_children = {
            child_id: second.child_metadata(
                child_agent_id=child_id,
                primary_parent_id=1,
                secondary_parent_id=2,
            )
            for child_id in (5, 4)
        }

        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_children, second_children)
        self.assertEqual(first.snapshot_artifact(), second.snapshot_artifact())
        metadata = first_metadata[1]
        self.assertEqual(
            metadata["schema_version"],
            RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION,
        )
        self.assertTrue(metadata["inherited_state"])
        self.assertEqual(metadata["state_size"], 16)
        self.assertEqual(metadata["inheritance_kind"], "founder")
        self.assertEqual(metadata["parent_genome_sha256s"], [])
        self.assertEqual(
            set(metadata),
            {
                "schema_version",
                "inherited_state",
                "state_size",
                "population_mode",
                "population_binding_sha256",
                "inheritance_kind",
                "genome_sha256",
                "parent_genome_sha256s",
            },
        )
        serialized_metadata = json.dumps(metadata, sort_keys=True)
        for forbidden in (
            "genome_stream_seed",
            "world_identity",
            "optimizer",
            "hidden_state",
            "model_weights",
            '"values"',
        ):
            self.assertNotIn(forbidden, serialized_metadata)

    def test_two_parent_inheritance_is_parent_order_commutative(self) -> None:
        left_first = RecurrentGenomePopulationManager(
            genome_stream_seed=111,
            world_identity="commutative-world",
            mode="heritable",
            mutation=RecurrentGenomeMutationConfig(rate=1.0, max_step=0.04),
        )
        right_first = RecurrentGenomePopulationManager(
            genome_stream_seed=111,
            world_identity="commutative-world",
            mode="heritable",
            mutation=RecurrentGenomeMutationConfig(rate=1.0, max_step=0.04),
        )
        for manager in (left_first, right_first):
            manager.founder_metadata(agent_id=1)
            manager.founder_metadata(agent_id=2)

        left_metadata = left_first.child_metadata(
            child_agent_id=3,
            primary_parent_id=1,
            secondary_parent_id=2,
        )
        right_metadata = right_first.child_metadata(
            child_agent_id=3,
            primary_parent_id=2,
            secondary_parent_id=1,
        )

        self.assertEqual(left_metadata, right_metadata)
        self.assertEqual(
            left_first.genome_for_agent(3),
            right_first.genome_for_agent(3),
        )
        self.assertEqual(
            left_metadata["parent_genome_sha256s"],
            sorted(left_metadata["parent_genome_sha256s"]),
        )

    def test_zero_all_mode_keeps_founders_and_children_neutral(self) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=99,
            world_identity="zero-control-world",
            mode="zero_all",
        )

        founder_one = manager.founder_metadata(agent_id=1)
        founder_two = manager.founder_metadata(agent_id=2)
        child = manager.child_metadata(
            child_agent_id=3,
            primary_parent_id=2,
            secondary_parent_id=1,
        )

        zero = zero_recurrent_genome()
        self.assertEqual(manager.genome_for_agent(1), zero)
        self.assertEqual(manager.genome_for_agent(2), zero)
        self.assertEqual(manager.genome_for_agent(3), zero)
        self.assertEqual(founder_one["population_mode"], "zero_all")
        self.assertEqual(founder_two["population_mode"], "zero_all")
        self.assertEqual(child["population_mode"], "zero_all")
        self.assertEqual(child["inheritance_kind"], "two_parent")
        self.assertEqual(
            child["parent_genome_sha256s"],
            [zero.sha256, zero.sha256],
        )
        self.assertTrue(child["inherited_state"])

    def test_duplicate_missing_and_invalid_parent_paths_fail_transactionally(
        self,
    ) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=3,
            world_identity="strict-world",
            mode="heritable",
        )
        manager.founder_metadata(agent_id=1)
        initial_snapshot = manager.snapshot_artifact()

        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "already has",
        ):
            manager.founder_metadata(agent_id=1)
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "agent 99 has no registered",
        ):
            manager.child_metadata(
                child_agent_id=2,
                primary_parent_id=99,
                secondary_parent_id=None,
            )
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "distinct parent ids",
        ):
            manager.child_metadata(
                child_agent_id=2,
                primary_parent_id=1,
                secondary_parent_id=1,
            )
        self.assertEqual(manager.snapshot_artifact(), initial_snapshot)

        manager.child_metadata(
            child_agent_id=2,
            primary_parent_id=1,
            secondary_parent_id=None,
        )
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "already has",
        ):
            manager.child_metadata(
                child_agent_id=2,
                primary_parent_id=1,
                secondary_parent_id=None,
            )

    def test_discard_and_reset_remove_genomes_without_changing_binding(self) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=7,
            world_identity="lifecycle-world",
            mode="heritable",
        )
        original_metadata = manager.founder_metadata(agent_id=1)
        original_binding = manager.binding_sha256

        self.assertTrue(manager.discard_agent(1))
        self.assertFalse(manager.discard_agent(1))
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "no registered",
        ):
            manager.genome_for_agent(1)
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "no registered",
        ):
            manager.child_metadata(
                child_agent_id=2,
                primary_parent_id=1,
                secondary_parent_id=None,
            )

        manager.founder_metadata(agent_id=3)
        manager.reset()
        self.assertEqual(manager.population_size, 0)
        self.assertEqual(manager.binding_sha256, original_binding)
        repeated_metadata = manager.founder_metadata(agent_id=1)
        self.assertEqual(repeated_metadata, original_metadata)

    def test_live_agent_reconciliation_is_strict_atomic_and_dead_only(self) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=8,
            world_identity="live-reconciliation",
            mode="heritable",
        )
        manager.founder_metadata(agent_id=1)
        manager.founder_metadata(agent_id=2)
        before = manager.snapshot_artifact()

        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "live agents have no registered",
        ):
            manager.reconcile_live_agent_ids((1, 3))
        self.assertEqual(manager.snapshot_artifact(), before)
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "strictly increasing and unique",
        ):
            manager.reconcile_live_agent_ids((2, 1))
        self.assertEqual(manager.snapshot_artifact(), before)

        planned = manager.reconciliation_dead_agent_ids((1,))
        self.assertEqual(planned, (2,))
        self.assertEqual(manager.snapshot_artifact(), before)
        discarded = manager.reconcile_live_agent_ids((1,))

        self.assertEqual(discarded, (2,))
        self.assertEqual(manager.population_size, 1)
        self.assertEqual(
            manager.genome_sha256_for_agent(1),
            before["agents"][0]["genome"]["genome_sha256"],
        )
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "agent 2 has no registered",
        ):
            manager.genome_for_agent(2)

        disabled = RecurrentGenomePopulationManager(
            genome_stream_seed=8,
            world_identity="disabled-live-reconciliation",
            mode="disabled",
        )
        self.assertEqual(disabled.reconcile_live_agent_ids((1, 2)), ())

    def test_immutable_binding_and_state_digest_caches_track_public_mutations(
        self,
    ) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=9,
            world_identity="binding-cache",
            mode="heritable",
        )
        empty_digest = manager.empty_state_sha256
        self.assertEqual(manager.state_sha256, empty_digest)
        manager.founder_metadata(agent_id=1)
        first_binding = manager.genome_binding_for_agent(1)
        first_live_digest = manager.state_sha256

        self.assertIs(first_binding, manager.genome_binding_for_agent(1))
        self.assertEqual(
            first_binding.genome_sha256,
            first_binding.genome.sha256,
        )
        self.assertEqual(manager.state_sha256, first_live_digest)
        self.assertNotEqual(first_live_digest, empty_digest)

        manager.reset()
        self.assertEqual(manager.state_sha256, empty_digest)

    def test_snapshot_round_trip_digest_pin_and_strict_serialization(self) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=2**64 - 1,
            world_identity="snapshot-world",
            mode="heritable",
            mutation=RecurrentGenomeMutationConfig(rate=0.75, max_step=0.03),
        )
        manager.founder_metadata(agent_id=2)
        manager.founder_metadata(agent_id=1)
        manager.child_metadata(
            child_agent_id=3,
            primary_parent_id=1,
            secondary_parent_id=2,
        )
        artifact = manager.snapshot_artifact()
        digest = artifact["snapshot_sha256"]

        restored = RecurrentGenomePopulationManager.from_snapshot_artifact(
            copy.deepcopy(artifact),
            expected_snapshot_sha256=digest,
        )
        serialized = manager.serialize_snapshot()
        restored_serialized = RecurrentGenomePopulationManager.from_serialized_snapshot(
            serialized,
            expected_snapshot_sha256=digest,
        )

        self.assertEqual(
            artifact["schema_version"],
            RECURRENT_GENOME_POPULATION_SNAPSHOT_SCHEMA_VERSION,
        )
        self.assertEqual(restored.snapshot_artifact(), artifact)
        self.assertEqual(restored_serialized.snapshot_artifact(), artifact)
        self.assertEqual(restored.state_sha256, digest)
        self.assertEqual(
            serialized,
            json.dumps(
                artifact,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii"),
        )
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "does not match expected",
        ):
            RecurrentGenomePopulationManager.from_snapshot_artifact(
                artifact,
                expected_snapshot_sha256="0" * 64,
            )
        pretty = json.dumps(artifact, indent=2, sort_keys=True)
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "not canonical",
        ):
            RecurrentGenomePopulationManager.from_serialized_snapshot(pretty)
        self.assertEqual(
            RecurrentGenomePopulationManager.from_serialized_snapshot(
                pretty,
                require_canonical=False,
            ).snapshot_artifact(),
            artifact,
        )

    def test_snapshot_tampering_ordering_duplicates_and_zero_mode_fail_closed(
        self,
    ) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=8,
            world_identity="tamper-world",
            mode="heritable",
        )
        manager.founder_metadata(agent_id=1)
        manager.founder_metadata(agent_id=2)
        artifact = manager.snapshot_artifact()

        value_tamper = copy.deepcopy(artifact)
        value_tamper["agents"][0]["genome"]["values"][0] = 0.999
        with self.assertRaises(RecurrentGenomePopulationError):
            RecurrentGenomePopulationManager.from_snapshot_artifact(value_tamper)

        binding_tamper = copy.deepcopy(artifact)
        binding_tamper["genome_stream"]["binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "binding SHA256 mismatch",
        ):
            RecurrentGenomePopulationManager.from_snapshot_artifact(binding_tamper)

        ordering_tamper = copy.deepcopy(artifact)
        ordering_tamper["agents"].reverse()
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "strictly ordered",
        ):
            RecurrentGenomePopulationManager.from_snapshot_artifact(ordering_tamper)

        extra_key = copy.deepcopy(artifact)
        extra_key["private_state"] = {}
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "keys mismatch",
        ):
            RecurrentGenomePopulationManager.from_snapshot_artifact(extra_key)

        duplicate_json = '{"schema_version":"first","schema_version":"second"}'
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "duplicate key",
        ):
            RecurrentGenomePopulationManager.from_serialized_snapshot(duplicate_json)

        zero_tamper = copy.deepcopy(artifact)
        zero_tamper["mode"] = "zero_all"
        zero_payload = _canonical_snapshot_sha256_payload(zero_tamper)
        zero_tamper["snapshot_sha256"] = _logical_sha256(zero_payload)
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "nonzero genome",
        ):
            RecurrentGenomePopulationManager.from_snapshot_artifact(zero_tamper)

    def test_invalid_constructor_and_event_material_fail_closed(self) -> None:
        invalid_manager_args = (
            {"genome_stream_seed": -1, "world_identity": "world"},
            {"genome_stream_seed": 2**64, "world_identity": "world"},
            {"genome_stream_seed": True, "world_identity": "world"},
            {"genome_stream_seed": 1, "world_identity": ""},
            {"genome_stream_seed": 1, "world_identity": " world"},
            {"genome_stream_seed": 1, "world_identity": "world\x00x"},
            {
                "genome_stream_seed": 1,
                "world_identity": "world",
                "mode": "HERITABLE",
            },
        )
        for kwargs in invalid_manager_args:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(RecurrentGenomePopulationError):
                    RecurrentGenomePopulationManager(**kwargs)

        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "require 1 parent digests",
        ):
            derive_recurrent_genome_event_seed(
                genome_stream_seed=1,
                world_identity="world",
                event_kind="asexual",
                agent_id=1,
            )
        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "positive integer",
        ):
            derive_recurrent_genome_event_seed(
                genome_stream_seed=1,
                world_identity="world",
                event_kind="founder",
                agent_id=0,
            )

    def test_real_world_founder_and_asexual_child_hooks_assign_genomes(
        self,
    ) -> None:
        manager = RecurrentGenomePopulationManager(
            genome_stream_seed=1776,
            world_identity="real-world-hook-flow",
            mode="heritable",
            mutation=RecurrentGenomeMutationConfig(rate=1.0, max_step=0.02),
        )
        policy = _PopulationPolicy(manager)
        world = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=1,
                width=5,
                height=5,
                initial_agents=1,
                max_agents=20,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
                base_energy_drain=0.0,
                base_hydration_drain=0.0,
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=0,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                ),
                diet_matching=DietMatchingConfig(
                    specialist_threshold=0.0,
                    omnivore_threshold=0.0,
                ),
                combat=CombatConfig(
                    min_attack_health_ratio=1.0,
                    min_attack_energy_ratio=1.0,
                    min_attack_hydration_ratio=1.0,
                ),
            ),
            policy=policy,
        )
        parent = world.alive_agents()[0]
        parent.age = 10
        parent.energy = parent.genome.max_energy * 1.25
        parent.hydration = parent.genome.max_hydration
        parent.health = parent.max_health
        parent.last_reproduction_tick = -10_000

        births = runtime_reproduction.run_reproduction_phase(
            world,
            context=world._reproduction_context(),
        )
        child = next(
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
        )

        self.assertEqual(births, 1)
        self.assertEqual(policy.founder_calls, [parent.agent_id])
        self.assertEqual(
            policy.child_calls,
            [(child.agent_id, parent.agent_id, None)],
        )
        self.assertEqual(
            parent.mind_inheritance_metadata["genome_sha256"],
            manager.genome_sha256_for_agent(parent.agent_id),
        )
        self.assertEqual(
            child.mind_inheritance_metadata["genome_sha256"],
            manager.genome_sha256_for_agent(child.agent_id),
        )
        self.assertEqual(
            child.mind_inheritance_metadata["inheritance_kind"],
            "asexual",
        )
        self.assertTrue(child.mind_inheritance_metadata["inherited_state"])

        reference = RecurrentGenomePopulationManager(
            genome_stream_seed=1776,
            world_identity="real-world-hook-flow",
            mode="heritable",
            mutation=RecurrentGenomeMutationConfig(rate=1.0, max_step=0.02),
        )
        reference.founder_metadata(agent_id=parent.agent_id)
        reference.child_metadata(
            child_agent_id=child.agent_id,
            primary_parent_id=parent.agent_id,
            secondary_parent_id=None,
        )
        self.assertEqual(
            manager.genome_for_agent(child.agent_id),
            reference.genome_for_agent(child.agent_id),
        )

    def test_disabled_snapshot_cannot_hide_genomes(self) -> None:
        disabled = RecurrentGenomePopulationManager(
            genome_stream_seed=0,
            world_identity="disabled-snapshot",
        )
        artifact = disabled.snapshot_artifact()
        artifact["agents"] = [
            {
                "agent_id": 1,
                "genome": recurrent_genome_artifact(zero_recurrent_genome()),
            }
        ]
        artifact["snapshot_sha256"] = _logical_sha256(
            _canonical_snapshot_sha256_payload(artifact)
        )

        with self.assertRaisesRegex(
            RecurrentGenomePopulationError,
            "must contain no genomes",
        ):
            RecurrentGenomePopulationManager.from_snapshot_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
