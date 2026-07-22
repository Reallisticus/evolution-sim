from __future__ import annotations

import hashlib
import json
import unittest

from evolution_sim.mind.recurrent_seed_registry import (
    CANONICAL_SEED_REGISTRY_SHA256,
    EXPECTED_LEARNER_CONFIRMATION_SEEDS,
    EXPECTED_LEARNER_DEVELOPMENT_SEEDS,
    GENERATED_SEED_REGISTRY_SHA256,
    LEGACY_DIAGNOSTIC_SEEDS,
    RECURRENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_EXPECTED_LEARNER_SEEDS,
    SCALE_DEVELOPMENT_GENERATED_SHA256,
    SCALE_DEVELOPMENT_SEED_REGISTRY,
    SCALE_DEVELOPMENT_SEED_REGISTRY_NAMESPACE,
    SCALE_DEVELOPMENT_SEED_REGISTRY_VERSION,
    SCALE_DEVELOPMENT_SEED_ROLE_COUNTS,
    SCALE_DEVELOPMENT_SEED_ROLE_POLICIES,
    SCALE_DEVELOPMENT_UNAVAILABLE_ROLES,
    SEED_REGISTRY_NAMESPACE,
    SEED_ROLE_COUNTS,
    SEED_ROLE_POLICIES,
    SeedRegistryError,
    build_recurrent_seed_registry,
    build_scale_development_seed_registry,
    canonical_seed_registry_json,
    canonical_seed_registry_payload,
    recurrent_seed_registry_contract,
    scale_development_seed_registry_contract,
    scale_development_seed_registry_json,
    scale_development_seed_registry_payload,
    scale_development_seeds_for_role,
    validate_scale_development_seed_registry,
    validate_recurrent_seed_registry,
)


class RecurrentSeedRegistryTests(unittest.TestCase):
    def test_registry_matches_audited_canonical_digest(self) -> None:
        self.assertEqual(
            SEED_REGISTRY_NAMESPACE,
            "evolution-sim|mind-v3-public-recurrent-ppo|seed-registry-v1|2026-07-21",
        )
        self.assertEqual(
            hashlib.sha256(canonical_seed_registry_json()).hexdigest(),
            CANONICAL_SEED_REGISTRY_SHA256,
        )
        self.assertEqual(
            GENERATED_SEED_REGISTRY_SHA256,
            "3e37cdfaf5e65c48f5c81a4c344041a7642f630f057b77c008ba98df1970daaf",
        )

    def test_roles_have_exact_counts_and_no_overlap(self) -> None:
        payload = canonical_seed_registry_payload()
        self.assertEqual(
            payload["diagnostic_red_team"],
            list(LEGACY_DIAGNOSTIC_SEEDS),
        )
        for role, expected_count in SEED_ROLE_COUNTS:
            self.assertEqual(len(payload[role]), expected_count)

        all_seeds = [seed for seeds in payload.values() for seed in seeds]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        fresh_seeds = set(all_seeds) - set(LEGACY_DIAGNOSTIC_SEEDS)
        self.assertTrue(fresh_seeds.isdisjoint(LEGACY_DIAGNOSTIC_SEEDS))
        self.assertTrue(all(1 <= seed <= 2_147_483_647 for seed in all_seeds))

    def test_learner_seed_axes_have_exact_audited_values(self) -> None:
        self.assertEqual(
            RECURRENT_SEED_REGISTRY["learner_development"],
            EXPECTED_LEARNER_DEVELOPMENT_SEEDS,
        )
        self.assertEqual(
            RECURRENT_SEED_REGISTRY["learner_confirmation"],
            EXPECTED_LEARNER_CONFIRMATION_SEEDS,
        )

    def test_generation_is_deterministic_and_callers_receive_copies(self) -> None:
        first = build_recurrent_seed_registry()
        second = build_recurrent_seed_registry()
        self.assertEqual(first, second)
        first["train"][0] = -1
        self.assertNotEqual(first, second)
        self.assertEqual(
            canonical_seed_registry_payload(),
            build_recurrent_seed_registry(),
        )

    def test_validator_rejects_noncanonical_or_overlapping_registry(self) -> None:
        canonical = build_recurrent_seed_registry()
        validate_recurrent_seed_registry(canonical)

        changed = build_recurrent_seed_registry()
        changed["selection"][0] = changed["train"][0]
        with self.assertRaisesRegex(SeedRegistryError, "not canonical"):
            validate_recurrent_seed_registry(changed)

    def test_lifecycle_policy_preserves_split_boundaries(self) -> None:
        diagnostic = SEED_ROLE_POLICIES["diagnostic_red_team"]
        self.assertEqual(diagnostic.lifecycle, "legacy_diagnostic_only")
        self.assertFalse(diagnostic.may_tune_configuration)
        self.assertFalse(diagnostic.promotion_evidence)

        selection = SEED_ROLE_POLICIES["selection"]
        self.assertTrue(selection.may_tune_configuration)

        validation = SEED_ROLE_POLICIES["validation"]
        self.assertEqual(validation.lifecycle, "limited_validation")
        self.assertFalse(validation.may_tune_configuration)

        lockbox = SEED_ROLE_POLICIES["lockbox"]
        self.assertEqual(lockbox.lifecycle, "sealed_promotion_lockbox")
        self.assertEqual(
            lockbox.access_policy,
            "one_time_after_candidate_and_configuration_freeze",
        )
        self.assertTrue(lockbox.promotion_evidence)

    def test_contract_is_json_serializable_and_names_every_role(self) -> None:
        contract = recurrent_seed_registry_contract()
        self.assertEqual(
            contract["canonical_sha256"],
            CANONICAL_SEED_REGISTRY_SHA256,
        )
        self.assertEqual(
            contract["role_order"],
            ["diagnostic_red_team", *dict(SEED_ROLE_COUNTS)],
        )
        self.assertEqual(set(contract["policies"]), set(contract["seeds"]))
        json.dumps(contract, sort_keys=True, allow_nan=False)

    def test_scale_registry_is_versioned_pinned_and_has_eight_learners(self) -> None:
        self.assertEqual(
            SCALE_DEVELOPMENT_SEED_REGISTRY_VERSION,
            "mind_v3_public_recurrent_scale_development_seed_registry_v1",
        )
        self.assertEqual(
            SCALE_DEVELOPMENT_SEED_REGISTRY_NAMESPACE,
            "evolution-sim|mind-v3-public-recurrent-ppo|"
            "scale-development-seed-registry-v1|2026-07-22",
        )
        self.assertEqual(
            hashlib.sha256(scale_development_seed_registry_json()).hexdigest(),
            SCALE_DEVELOPMENT_CANONICAL_SHA256,
        )
        self.assertEqual(
            SCALE_DEVELOPMENT_GENERATED_SHA256,
            "42d6b8aa68340379cef16ac48278a579643f15a28497ea4b07e28c90bc6b8557",
        )
        self.assertEqual(
            SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"],
            SCALE_DEVELOPMENT_EXPECTED_LEARNER_SEEDS,
        )
        self.assertEqual(len(SCALE_DEVELOPMENT_EXPECTED_LEARNER_SEEDS), 8)

    def test_scale_environment_and_learner_roles_are_globally_disjoint(self) -> None:
        scale = scale_development_seed_registry_payload()
        old = canonical_seed_registry_payload()
        self.assertEqual(
            {role: len(seeds) for role, seeds in scale.items()},
            dict(SCALE_DEVELOPMENT_SEED_ROLE_COUNTS),
        )
        scale_flat = [seed for seeds in scale.values() for seed in seeds]
        old_flat = [seed for seeds in old.values() for seed in seeds]
        self.assertEqual(len(scale_flat), len(set(scale_flat)))
        self.assertTrue(set(scale_flat).isdisjoint(old_flat))
        self.assertTrue(all(1 <= seed <= 2_147_483_647 for seed in scale_flat))

        for role in ("scale_curriculum", "scale_train", "scale_selection"):
            self.assertEqual(
                SCALE_DEVELOPMENT_SEED_ROLE_POLICIES[role].axis,
                "environment",
            )
        self.assertEqual(
            SCALE_DEVELOPMENT_SEED_ROLE_POLICIES["scale_learner"].axis,
            "learner",
        )

    def test_scale_validation_and_lockbox_are_unavailable_not_empty_roles(self) -> None:
        contract = scale_development_seed_registry_contract()
        self.assertEqual(
            set(contract["unavailable_roles"]),
            set(SCALE_DEVELOPMENT_UNAVAILABLE_ROLES),
        )
        self.assertTrue(
            set(contract["seeds"]).isdisjoint(SCALE_DEVELOPMENT_UNAVAILABLE_ROLES)
        )
        for role in SCALE_DEVELOPMENT_UNAVAILABLE_ROLES:
            unavailable = contract["unavailable_roles"][role]
            self.assertIs(unavailable["available"], False)
            self.assertIs(unavailable["seed_values_included"], False)
            with self.assertRaisesRegex(SeedRegistryError, "forbidden"):
                scale_development_seeds_for_role(role)

        self.assertEqual(
            scale_development_seeds_for_role("scale_train"),
            SCALE_DEVELOPMENT_SEED_REGISTRY["scale_train"],
        )
        with self.assertRaisesRegex(SeedRegistryError, "unknown"):
            scale_development_seeds_for_role("train")

    def test_scale_validator_rejects_cross_role_and_v1_seed_substitution(self) -> None:
        canonical = build_scale_development_seed_registry()
        validate_scale_development_seed_registry(canonical)

        cross_role = build_scale_development_seed_registry()
        cross_role["scale_selection"][0] = cross_role["scale_train"][0]
        with self.assertRaisesRegex(SeedRegistryError, "not canonical"):
            validate_scale_development_seed_registry(cross_role)

        v1_substitution = build_scale_development_seed_registry()
        v1_substitution["scale_train"][0] = RECURRENT_SEED_REGISTRY["train"][0]
        with self.assertRaisesRegex(SeedRegistryError, "not canonical"):
            validate_scale_development_seed_registry(v1_substitution)

        exposed = build_scale_development_seed_registry()
        exposed["validation"] = [RECURRENT_SEED_REGISTRY["validation"][0]]
        with self.assertRaisesRegex(SeedRegistryError, "roles"):
            validate_scale_development_seed_registry(exposed)

    def test_scale_generation_is_deterministic_and_returns_copies(self) -> None:
        first = build_scale_development_seed_registry()
        second = build_scale_development_seed_registry()
        self.assertEqual(first, second)
        first["scale_train"][0] = -1
        self.assertNotEqual(first, second)
        self.assertEqual(second, scale_development_seed_registry_payload())
        json.dumps(
            scale_development_seed_registry_contract(),
            sort_keys=True,
            allow_nan=False,
        )


if __name__ == "__main__":
    unittest.main()
