from __future__ import annotations

import hashlib
import json
import unittest

from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_EXPECTED_FIRST_GENOME_STREAM_SEEDS,
    OPEN_ECOLOGY_EXPECTED_LEARNER_SEEDS,
    OPEN_ECOLOGY_GENERATED_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
    OPEN_ECOLOGY_SEED_ROLE_COUNTS,
    OPEN_ECOLOGY_SEED_ROLE_POLICIES,
    OPEN_ECOLOGY_UNAVAILABLE_ROLES,
    OpenEcologySeedRegistryError,
    build_open_ecology_seed_registry,
    open_ecology_seed_registry_contract,
    open_ecology_seed_registry_json,
    open_ecology_seed_registry_payload,
    open_ecology_seeds_for_role,
    validate_open_ecology_seed_registry,
)
from evolution_sim.mind.recurrent_seed_registry import (
    build_recurrent_seed_registry,
    build_scale_development_seed_registry,
    build_scale_development_v2_seed_registry,
)


class OpenEcologySeedRegistryTests(unittest.TestCase):
    def test_registry_is_pinned_fresh_and_has_declared_axes(self) -> None:
        self.assertEqual(
            OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
            "mind_v3_open_ecology_development_seed_registry_v2",
        )
        self.assertEqual(
            OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE,
            "evolution-sim|mind-v3-open-ecology|"
            "development-seed-registry-v1|2026-07-27",
        )
        self.assertEqual(
            hashlib.sha256(open_ecology_seed_registry_json()).hexdigest(),
            OPEN_ECOLOGY_CANONICAL_SHA256,
        )
        self.assertEqual(
            OPEN_ECOLOGY_GENERATED_SHA256,
            "ed4ba14609fe2e98795cc4352ec50ab1e172c3c758b1a268f6481ac024e348e0",
        )
        self.assertEqual(
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"],
            OPEN_ECOLOGY_EXPECTED_LEARNER_SEEDS,
        )
        self.assertEqual(
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][:8],
            OPEN_ECOLOGY_EXPECTED_FIRST_GENOME_STREAM_SEEDS,
        )
        self.assertEqual(
            {role: len(values) for role, values in OPEN_ECOLOGY_SEED_REGISTRY.items()},
            dict(OPEN_ECOLOGY_SEED_ROLE_COUNTS),
        )

    def test_public_axes_are_globally_disjoint_and_genome_axis_is_high_u64(
        self,
    ) -> None:
        current = open_ecology_seed_registry_payload()
        public_values = [
            seed
            for role, values in current.items()
            if role != "open_ecology_genome_stream"
            for seed in values
        ]
        genome_values = current["open_ecology_genome_stream"]
        prior = {
            seed
            for registry in (
                build_recurrent_seed_registry(),
                build_scale_development_seed_registry(),
                build_scale_development_v2_seed_registry(),
            )
            for values in registry.values()
            for seed in values
        }

        self.assertEqual(len(public_values), len(set(public_values)))
        self.assertTrue(set(public_values).isdisjoint(prior))
        self.assertTrue(all(1 <= seed <= 2_147_483_647 for seed in public_values))
        self.assertEqual(len(genome_values), len(set(genome_values)))
        self.assertTrue(all(2**63 <= seed <= 2**64 - 1 for seed in genome_values))

    def test_generation_returns_copies_and_validator_rejects_drift(self) -> None:
        first = build_open_ecology_seed_registry()
        second = build_open_ecology_seed_registry()
        self.assertEqual(first, second)
        first["open_ecology_train"][0] = -1
        self.assertNotEqual(first, second)
        self.assertEqual(second, open_ecology_seed_registry_payload())
        validate_open_ecology_seed_registry(second)

        tampered = build_open_ecology_seed_registry()
        tampered["open_ecology_selection"][0] = tampered["open_ecology_train"][0]
        with self.assertRaisesRegex(OpenEcologySeedRegistryError, "not canonical"):
            validate_open_ecology_seed_registry(tampered)

        exposed = build_open_ecology_seed_registry()
        exposed["validation"] = [1]
        with self.assertRaisesRegex(OpenEcologySeedRegistryError, "roles"):
            validate_open_ecology_seed_registry(exposed)

    def test_role_contract_keeps_validation_and_lockbox_unavailable(self) -> None:
        contract = open_ecology_seed_registry_contract()
        self.assertEqual(
            set(contract["unavailable_roles"]),
            set(OPEN_ECOLOGY_UNAVAILABLE_ROLES),
        )
        self.assertEqual(
            set(contract["policies"]),
            set(contract["seeds"]),
        )
        self.assertEqual(
            OPEN_ECOLOGY_SEED_ROLE_POLICIES["open_ecology_genome_stream"].axis,
            "controller_genome",
        )
        self.assertFalse(
            OPEN_ECOLOGY_SEED_ROLE_POLICIES["open_ecology_selection"].promotion_evidence
        )
        benchmark_policy = OPEN_ECOLOGY_SEED_ROLE_POLICIES["open_ecology_benchmark"]
        self.assertFalse(benchmark_policy.may_tune_configuration)
        self.assertFalse(benchmark_policy.promotion_evidence)
        self.assertEqual(
            benchmark_policy.lifecycle,
            "open_ecology_operational_resource_benchmark",
        )
        proof_policy = OPEN_ECOLOGY_SEED_ROLE_POLICIES["open_ecology_proof"]
        self.assertFalse(proof_policy.may_tune_configuration)
        self.assertFalse(proof_policy.promotion_evidence)
        self.assertEqual(
            proof_policy.lifecycle,
            "open_ecology_prelaunch_engineering_proof",
        )
        self.assertTrue(
            set(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"]).isdisjoint(
                OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"]
            )
        )
        for role in OPEN_ECOLOGY_UNAVAILABLE_ROLES:
            self.assertIs(contract["unavailable_roles"][role]["available"], False)
            self.assertIs(
                contract["unavailable_roles"][role]["seed_values_included"],
                False,
            )
            with self.assertRaisesRegex(OpenEcologySeedRegistryError, "forbidden"):
                open_ecology_seeds_for_role(role)

        self.assertEqual(
            open_ecology_seeds_for_role("open_ecology_train"),
            OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_train"],
        )
        with self.assertRaisesRegex(OpenEcologySeedRegistryError, "unknown"):
            open_ecology_seeds_for_role("train")
        json.dumps(contract, allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
