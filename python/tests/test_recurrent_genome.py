from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest

from evolution_sim.mind.recurrent_genome import (
    RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION,
    RECURRENT_CONTROLLER_FILM_BIAS_LIMIT,
    RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT,
    RECURRENT_CONTROLLER_GENOME_MAX,
    RECURRENT_CONTROLLER_GENOME_MIN,
    RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION,
    RECURRENT_CONTROLLER_GENOME_SIZE,
    FilmDevelopment,
    RecurrentControllerGenome,
    RecurrentGenomeError,
    RecurrentGenomeMutationConfig,
    apply_film,
    bounded_recurrent_genome_perturbation,
    deserialize_recurrent_genome,
    develop_recurrent_genome,
    evaluate_recurrent_genome_swap,
    founder_recurrent_genome,
    inherit_asexual_recurrent_genome,
    recombine_recurrent_genomes,
    recurrent_genome_artifact,
    recurrent_genome_contract,
    recurrent_genome_development_contract,
    recurrent_genome_mean_absolute_distance,
    serialize_recurrent_genome,
    validate_recurrent_genome_artifact,
    zero_recurrent_genome,
)


class RecurrentGenomeTests(unittest.TestCase):
    def test_seeded_founders_are_fixed_bounded_and_identity_free(self) -> None:
        first = founder_recurrent_genome(seed=20260727)
        repeat = founder_recurrent_genome(seed=20260727)
        other = founder_recurrent_genome(seed=20260728)

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first.values), RECURRENT_CONTROLLER_GENOME_SIZE)
        self.assertTrue(
            all(
                RECURRENT_CONTROLLER_GENOME_MIN
                <= value
                <= RECURRENT_CONTROLLER_GENOME_MAX
                for value in first.values
            )
        )
        artifact_text = serialize_recurrent_genome(first).decode("ascii")
        for forbidden in (
            "seed",
            "fixture",
            "private",
            "action",
            "optimizer",
            "hidden_state",
        ):
            self.assertNotIn(forbidden, artifact_text)

    def test_exact_canonical_serialization_digest_and_round_trip(self) -> None:
        genome = founder_recurrent_genome(seed=42)
        artifact = recurrent_genome_artifact(genome)
        serialized = serialize_recurrent_genome(genome)
        without_digest = {
            key: value for key, value in artifact.items() if key != "genome_sha256"
        }
        expected_digest = hashlib.sha256(
            json.dumps(
                without_digest,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()

        self.assertEqual(
            artifact["schema_version"],
            RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION,
        )
        self.assertEqual(artifact["contract"], recurrent_genome_contract())
        self.assertEqual(artifact["genome_sha256"], expected_digest)
        self.assertEqual(genome.sha256, expected_digest)
        self.assertEqual(
            expected_digest,
            "229e666724b3dc5d4cffbca13a1694abc349a37c633f0ab334cd581527a0d18f",
        )
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
        self.assertEqual(deserialize_recurrent_genome(serialized), genome)

    def test_noncanonical_and_duplicate_json_fail_closed(self) -> None:
        genome = founder_recurrent_genome(seed=9)
        artifact = recurrent_genome_artifact(genome)
        pretty = json.dumps(artifact, indent=2, sort_keys=True)

        with self.assertRaisesRegex(RecurrentGenomeError, "not canonical"):
            deserialize_recurrent_genome(pretty)
        self.assertEqual(
            deserialize_recurrent_genome(pretty, require_canonical=False),
            genome,
        )
        with self.assertRaisesRegex(RecurrentGenomeError, "duplicate"):
            deserialize_recurrent_genome(
                '{"schema_version":"first","schema_version":"second"}'
            )

    def test_schema_contract_value_and_digest_tampering_fail_closed(self) -> None:
        artifact = recurrent_genome_artifact(founder_recurrent_genome(seed=123))
        cases: list[tuple[str, dict[str, object]]] = []

        extra = copy.deepcopy(artifact)
        extra["parent_id"] = 17
        cases.append(("keys mismatch", extra))

        stale = copy.deepcopy(artifact)
        stale["schema_version"] = "stale"
        cases.append(("schema_version", stale))

        contract = copy.deepcopy(artifact)
        contract["contract"]["genome_size"] = RECURRENT_CONTROLLER_GENOME_SIZE + 1
        cases.append(("contract", contract))

        value = copy.deepcopy(artifact)
        value["values"][0] = 1.1
        cases.append(("out of bounds", value))

        precision = copy.deepcopy(artifact)
        precision["values"][0] = 0.123456789
        cases.append(("canonical float precision", precision))

        digest = copy.deepcopy(artifact)
        digest["values"][0] = round(-digest["values"][0], 8)
        cases.append(("SHA256 mismatch", digest))

        for expected, tampered in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RecurrentGenomeError, expected):
                    validate_recurrent_genome_artifact(tampered)

    def test_nonfinite_nonfloat_and_negative_zero_values_fail_closed(self) -> None:
        base = [0.0] * RECURRENT_CONTROLLER_GENOME_SIZE
        invalid_values = (
            [*base[:-1], float("nan")],
            [*base[:-1], float("inf")],
            [*base[:-1], True],
            [*base[:-1], 0],
            [*base[:-1], -0.0],
        )
        for values in invalid_values:
            with self.subTest(value=values[-1]):
                with self.assertRaises(RecurrentGenomeError):
                    RecurrentControllerGenome(values=tuple(values))

    def test_asexual_inheritance_is_replayable_and_mutation_bounded(self) -> None:
        parent = founder_recurrent_genome(seed=1)
        mutation = RecurrentGenomeMutationConfig(rate=1.0, max_step=0.05)
        first = inherit_asexual_recurrent_genome(
            parent,
            seed=99,
            mutation=mutation,
        )
        repeat = inherit_asexual_recurrent_genome(
            parent,
            seed=99,
            mutation=mutation,
        )
        other = inherit_asexual_recurrent_genome(
            parent,
            seed=100,
            mutation=mutation,
        )

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, parent)
        self.assertNotEqual(first, other)
        self.assertTrue(
            all(
                abs(child - source) <= mutation.max_step + 1e-8
                for child, source in zip(
                    first.values,
                    parent.values,
                    strict=True,
                )
            )
        )

    def test_two_parent_recombination_is_commutative_and_replayable(self) -> None:
        left = founder_recurrent_genome(seed=10)
        right = founder_recurrent_genome(seed=20)
        no_mutation = RecurrentGenomeMutationConfig(rate=0.0, max_step=0.0)

        child = recombine_recurrent_genomes(
            left,
            right,
            seed=30,
            mutation=no_mutation,
        )
        swapped = recombine_recurrent_genomes(
            right,
            left,
            seed=30,
            mutation=no_mutation,
        )
        repeated = recombine_recurrent_genomes(
            left,
            right,
            seed=30,
            mutation=no_mutation,
        )

        self.assertEqual(child, swapped)
        self.assertEqual(child, repeated)
        self.assertTrue(
            all(
                allele in (left.values[index], right.values[index])
                for index, allele in enumerate(child.values)
            )
        )
        self.assertTrue(
            any(
                allele == left.values[index]
                for index, allele in enumerate(child.values)
            )
        )
        self.assertTrue(
            any(
                allele == right.values[index]
                for index, allele in enumerate(child.values)
            )
        )

    def test_two_parent_post_crossover_mutation_is_bounded(self) -> None:
        left = founder_recurrent_genome(seed=10)
        right = founder_recurrent_genome(seed=20)
        no_mutation = RecurrentGenomeMutationConfig(rate=0.0, max_step=0.0)
        base = recombine_recurrent_genomes(
            left,
            right,
            seed=31,
            mutation=no_mutation,
        )
        mutation = RecurrentGenomeMutationConfig(rate=1.0, max_step=0.025)
        mutated = recombine_recurrent_genomes(
            left,
            right,
            seed=31,
            mutation=mutation,
        )

        self.assertNotEqual(base, mutated)
        self.assertTrue(
            all(
                abs(changed - source) <= mutation.max_step + 1e-8
                for changed, source in zip(
                    mutated.values,
                    base.values,
                    strict=True,
                )
            )
        )

    def test_zero_genome_develops_to_neutral_film_at_requested_shape(self) -> None:
        genome = zero_recurrent_genome()
        first = develop_recurrent_genome(genome, hidden_dimension=128)
        repeat = develop_recurrent_genome(genome, hidden_dimension=128)
        contract = recurrent_genome_development_contract(hidden_dimension=128)

        self.assertEqual(first, repeat)
        self.assertEqual(
            first.contract_version,
            RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION,
        )
        self.assertEqual(first.hidden_dimension, 128)
        self.assertEqual(first.scale, (1.0,) * 128)
        self.assertEqual(first.bias, (0.0,) * 128)
        self.assertEqual(contract["hidden_dimension"], 128)
        self.assertEqual(
            contract["birth_state_policy"],
            "external_zero_initialization_required",
        )

    def test_nonzero_genome_development_is_bounded_and_shape_generic(self) -> None:
        genome = founder_recurrent_genome(seed=777)

        for hidden_dimension in (1, 7, 128, 257):
            with self.subTest(hidden_dimension=hidden_dimension):
                film = develop_recurrent_genome(
                    genome,
                    hidden_dimension=hidden_dimension,
                )
                self.assertEqual(len(film.scale), hidden_dimension)
                self.assertEqual(len(film.bias), hidden_dimension)
                self.assertTrue(
                    all(
                        1.0 - RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT
                        <= value
                        <= 1.0 + RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT
                        for value in film.scale
                    )
                )
                self.assertTrue(
                    all(
                        -RECURRENT_CONTROLLER_FILM_BIAS_LIMIT
                        <= value
                        <= RECURRENT_CONTROLLER_FILM_BIAS_LIMIT
                        for value in film.bias
                    )
                )
        seed_42_film = develop_recurrent_genome(
            founder_recurrent_genome(seed=42),
            hidden_dimension=4,
        )
        self.assertEqual(
            seed_42_film.scale,
            (1.04957323, 1.00488434, 1.02357246, 1.01363024),
        )
        self.assertEqual(
            seed_42_film.bias,
            (0.01870591, -0.03845304, 0.02578422, -0.00340162),
        )

    def test_bounded_perturbation_and_swap_hold_activations_fixed(self) -> None:
        original = founder_recurrent_genome(seed=55)
        perturbed = bounded_recurrent_genome_perturbation(
            original,
            locus=3,
            delta=0.04,
            max_absolute_delta=0.05,
        )
        activations = tuple((index - 4) / 5.0 for index in range(9))
        effect = evaluate_recurrent_genome_swap(
            activations,
            left=original,
            right=perturbed,
        )
        reversed_effect = evaluate_recurrent_genome_swap(
            activations,
            left=perturbed,
            right=original,
        )

        self.assertEqual(
            recurrent_genome_mean_absolute_distance(original, perturbed),
            round(0.04 / RECURRENT_CONTROLLER_GENOME_SIZE, 8),
        )
        self.assertTrue(effect.changed)
        self.assertGreater(effect.mean_absolute_difference, 0.0)
        self.assertEqual(effect.left_output, reversed_effect.right_output)
        self.assertEqual(effect.right_output, reversed_effect.left_output)
        self.assertEqual(
            effect.max_absolute_difference,
            reversed_effect.max_absolute_difference,
        )

    def test_apply_film_and_invalid_contract_inputs_fail_closed(self) -> None:
        genome = founder_recurrent_genome(seed=8)
        film = develop_recurrent_genome(genome, hidden_dimension=3)
        output = apply_film((0.25, -0.5, 1.0), film)

        self.assertEqual(len(output), 3)
        self.assertTrue(all(math.isfinite(value) for value in output))
        with self.assertRaises(RecurrentGenomeError):
            apply_film((0.25, -0.5), film)
        with self.assertRaises(RecurrentGenomeError):
            apply_film((0.25, float("nan"), 1.0), film)
        with self.assertRaises(RecurrentGenomeError):
            founder_recurrent_genome(seed=True)
        with self.assertRaises(RecurrentGenomeError):
            develop_recurrent_genome(genome, hidden_dimension=0)
        with self.assertRaises(RecurrentGenomeError):
            bounded_recurrent_genome_perturbation(
                genome,
                locus=0,
                delta=0.2,
                max_absolute_delta=0.1,
            )
        with self.assertRaises(RecurrentGenomeError):
            FilmDevelopment(
                contract_version="stale",
                genome_sha256=genome.sha256,
                hidden_dimension=1,
                scale=(1.0,),
                bias=(0.0,),
            )


if __name__ == "__main__":
    unittest.main()
