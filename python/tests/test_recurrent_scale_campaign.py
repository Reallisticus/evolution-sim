from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.mind.recurrent_counterfactual_comparison import (
        BASE_ARM,
        EXACT_ARM,
        SHUFFLED_ARM,
    )
    from evolution_sim.mind.recurrent_scale_campaign import (
        RECURRENT_SCALE_TOTAL_TRAINING_WORLDS,
        RecurrentScaleCampaignError,
        build_recurrent_scale_campaign_preregistration,
        load_strict_json,
        recurrent_scale_arm_run_id,
        recurrent_scale_selection_seed_plan,
        validate_recurrent_scale_campaign_preregistration,
        write_atomic_json,
    )
    from evolution_sim.mind.recurrent_seed_registry import (
        SCALE_DEVELOPMENT_SEED_REGISTRY,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentScaleCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_commit = "a" * 40
        self.source_manifest_sha256 = "b" * 64

    def test_preregistration_pins_three_arms_eight_learners_and_6144_worlds(
        self,
    ) -> None:
        preregistration = build_recurrent_scale_campaign_preregistration(
            source_commit=self.source_commit,
            source_manifest_sha256=self.source_manifest_sha256,
        )

        validate_recurrent_scale_campaign_preregistration(preregistration)
        self.assertEqual(RECURRENT_SCALE_TOTAL_TRAINING_WORLDS, 6144)
        self.assertEqual(
            preregistration["arms"]["order"],  # type: ignore[index]
            [BASE_ARM, EXACT_ARM, SHUFFLED_ARM],
        )
        self.assertEqual(
            len(preregistration["seed_contract"]["learner_seeds"]),  # type: ignore[index]
            8,
        )
        self.assertTrue(  # type: ignore[index]
            preregistration["seed_contract"][
                "learner_specific_policy_and_counterfactual_rng_namespaces"
            ]
        )
        self.assertTrue(  # type: ignore[index]
            preregistration["seed_contract"]["arm_paired_within_learner"]
        )
        counterfactual = preregistration["counterfactual"]
        self.assertEqual(  # type: ignore[index]
            counterfactual["independent_rng_tapes_per_branch"],
            8,
        )
        self.assertEqual(  # type: ignore[index]
            counterfactual["absolute_terminal_target_tick"],
            120,
        )
        self.assertFalse(  # type: ignore[index]
            preregistration["lifecycle"]["validation_seeds_accessed"]
        )
        self.assertFalse(  # type: ignore[index]
            preregistration["lifecycle"]["lockbox_seeds_accessed"]
        )

    def test_preregistration_rejects_digest_rewrite_and_contract_rewrite(self) -> None:
        preregistration = build_recurrent_scale_campaign_preregistration(
            source_commit=self.source_commit,
            source_manifest_sha256=self.source_manifest_sha256,
        )
        tampered = copy.deepcopy(preregistration)
        tampered["training"]["updates_per_arm_learner"] = 15  # type: ignore[index]
        with self.assertRaisesRegex(RecurrentScaleCampaignError, "digest"):
            validate_recurrent_scale_campaign_preregistration(tampered)

        tampered["exact_digest"] = "c" * 64
        with self.assertRaisesRegex(RecurrentScaleCampaignError, "digest"):
            validate_recurrent_scale_campaign_preregistration(tampered)

    def test_selection_plan_uses_only_fresh_scale_selection_seeds(self) -> None:
        plan = recurrent_scale_selection_seed_plan()
        selected = set(plan.broad_seeds)
        optimized = set(plan.excluded_training_seeds)

        self.assertEqual(plan.broad_seeds, plan.fixture_seeds)
        self.assertEqual(plan.canonical_registry_role, "scale_selection")
        self.assertEqual(len(selected), 8)
        self.assertTrue(selected.isdisjoint(optimized))
        self.assertTrue(
            selected.issubset(SCALE_DEVELOPMENT_SEED_REGISTRY["scale_selection"])
        )

    def test_arm_run_identity_rejects_noncanonical_inputs(self) -> None:
        learner = SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]
        self.assertEqual(
            recurrent_scale_arm_run_id(learner_seed=learner, arm=BASE_ARM),
            f"scale-v1-learner-{learner}-{BASE_ARM}",
        )
        with self.assertRaisesRegex(RecurrentScaleCampaignError, "learner"):
            recurrent_scale_arm_run_id(learner_seed=1, arm=BASE_ARM)
        with self.assertRaisesRegex(RecurrentScaleCampaignError, "arm"):
            recurrent_scale_arm_run_id(learner_seed=learner, arm="unknown")

    def test_atomic_json_round_trip_and_strict_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "preregistration.json"
            payload = build_recurrent_scale_campaign_preregistration(
                source_commit=self.source_commit,
                source_manifest_sha256=self.source_manifest_sha256,
            )
            write_atomic_json(path, payload)
            self.assertEqual(load_strict_json(path), payload)

            duplicate = Path(temporary) / "duplicate.json"
            duplicate.write_text('{"key":1,"key":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RecurrentScaleCampaignError, "duplicate"):
                load_strict_json(duplicate)

            json.dumps(payload, allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
