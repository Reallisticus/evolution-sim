from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType

from evolution_sim.mind.recurrent_seed_registry import (
    MAX_SEED,
    build_recurrent_seed_registry,
    build_scale_development_seed_registry,
    build_scale_development_v2_seed_registry,
)


OPEN_ECOLOGY_SEED_REGISTRY_VERSION = "mind_v3_open_ecology_development_seed_registry_v4"
OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE = (
    "evolution-sim|mind-v3-open-ecology|development-seed-registry-v1|2026-07-27"
)
OPEN_ECOLOGY_SEED_ROLE_COUNTS: tuple[tuple[str, int], ...] = (
    ("open_ecology_train", 512),
    ("open_ecology_selection", 128),
    ("open_ecology_island", 32),
    ("open_ecology_learner", 8),
    ("open_ecology_genome_stream", 32),
    ("open_ecology_benchmark", 64),
    ("open_ecology_proof", 16),
)
OPEN_ECOLOGY_UNAVAILABLE_ROLES: tuple[str, ...] = ("validation", "lockbox")
OPEN_ECOLOGY_BENCHMARK_SEED_ROLE = "open_ecology_benchmark"
OPEN_ECOLOGY_PHASE_A_BENCHMARK_ENVIRONMENT_SEED_INDICES: tuple[int, ...] = tuple(
    range(0, 16)
)
OPEN_ECOLOGY_PHASE_A_BENCHMARK_MODEL_SEED_INDEX = 16
OPEN_ECOLOGY_PHASE_A_BENCHMARK_GENOME_STREAM_SEED_INDEX = 17
OPEN_ECOLOGY_PHASE_B_BENCHMARK_ENVIRONMENT_SEED_INDICES: tuple[int, ...] = tuple(
    range(18, 34)
)
OPEN_ECOLOGY_PHASE_B_BENCHMARK_MODEL_SEED_INDEX = 34
OPEN_ECOLOGY_PHASE_B_BENCHMARK_GENOME_STREAM_SEED_INDEX = 35
OPEN_ECOLOGY_SELECTION_BENCHMARK_ENVIRONMENT_SEED_INDICES: tuple[int, ...] = tuple(
    range(36, 40)
)
OPEN_ECOLOGY_SELECTION_BENCHMARK_MODEL_SEED_INDEX = 40
OPEN_ECOLOGY_SELECTION_BENCHMARK_GENOME_STREAM_SEED_INDEX = 41
OPEN_ECOLOGY_PHASE_D_BENCHMARK_ENVIRONMENT_SEED_INDEX = 42
OPEN_ECOLOGY_PHASE_D_BENCHMARK_MODEL_SEED_INDEX = 43
OPEN_ECOLOGY_PHASE_D_BENCHMARK_GENOME_STREAM_SEED_INDEX = 44
OPEN_ECOLOGY_BENCHMARK_RESERVED_SEED_INDICES: tuple[int, ...] = tuple(range(45, 64))

# Compatibility names remain Phase-A-specific. New code should use the explicit
# Phase-A constants above rather than treating this role as a single shared axis.
OPEN_ECOLOGY_BENCHMARK_ENVIRONMENT_SEED_COUNT = len(
    OPEN_ECOLOGY_PHASE_A_BENCHMARK_ENVIRONMENT_SEED_INDICES
)
OPEN_ECOLOGY_BENCHMARK_LEARNER_SEED_INDEX = (
    OPEN_ECOLOGY_PHASE_A_BENCHMARK_MODEL_SEED_INDEX
)
OPEN_ECOLOGY_BENCHMARK_GENOME_STREAM_SEED_INDEX = (
    OPEN_ECOLOGY_PHASE_A_BENCHMARK_GENOME_STREAM_SEED_INDEX
)
OPEN_ECOLOGY_CANONICAL_SHA256 = (
    "ed4ba14609fe2e98795cc4352ec50ab1e172c3c758b1a268f6481ac024e348e0"
)
OPEN_ECOLOGY_EXPECTED_LEARNER_SEEDS: tuple[int, ...] = (
    1_570_849_880,
    1_323_833_341,
    67_094_806,
    1_871_495_966,
    411_483_968,
    1_589_768_480,
    1_483_917_974,
    372_226_408,
)
OPEN_ECOLOGY_EXPECTED_FIRST_GENOME_STREAM_SEEDS: tuple[int, ...] = (
    16_663_129_824_008_534_063,
    16_925_559_243_916_883_243,
    10_876_559_716_473_268_626,
    16_263_353_148_861_378_742,
    9_505_534_202_986_513_683,
    16_174_670_609_993_915_595,
    15_696_593_396_493_072_015,
    13_080_471_060_878_181_776,
)


class OpenEcologySeedRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OpenEcologySeedRolePolicy:
    axis: str
    lifecycle: str
    may_tune_configuration: bool
    promotion_evidence: bool
    access_policy: str


OPEN_ECOLOGY_SEED_ROLE_POLICIES: Mapping[str, OpenEcologySeedRolePolicy] = (
    MappingProxyType(
        {
            "open_ecology_train": OpenEcologySeedRolePolicy(
                axis="environment",
                lifecycle="open_ecology_optimization_training",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_open_ecology_training_only",
            ),
            "open_ecology_selection": OpenEcologySeedRolePolicy(
                axis="environment",
                lifecycle="open_ecology_development_selection",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_preregistered_development_selection",
            ),
            "open_ecology_island": OpenEcologySeedRolePolicy(
                axis="environment",
                lifecycle="open_ecology_persistent_development_island",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_long_development_islands",
            ),
            "open_ecology_learner": OpenEcologySeedRolePolicy(
                axis="learner",
                lifecycle="open_ecology_learner_development",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_independent_learner_replication",
            ),
            "open_ecology_genome_stream": OpenEcologySeedRolePolicy(
                axis="controller_genome",
                lifecycle="open_ecology_inheritance_development",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="paired_across_preregistered_treatment_arms",
            ),
            "open_ecology_benchmark": OpenEcologySeedRolePolicy(
                axis="operational_benchmark_environment_model_and_genome",
                lifecycle="open_ecology_operational_resource_benchmark",
                may_tune_configuration=False,
                promotion_evidence=False,
                access_policy=(
                    "phase_a_environment_0_15_model_16_genome_17_phase_b_"
                    "environment_18_33_model_34_genome_35_selection_environment_"
                    "36_39_model_40_genome_41_phase_d_paired_environment_42_"
                    "model_43_genome_44_reserved_45_63_operational_only"
                ),
            ),
            "open_ecology_proof": OpenEcologySeedRolePolicy(
                axis="environment",
                lifecycle="open_ecology_prelaunch_engineering_proof",
                may_tune_configuration=False,
                promotion_evidence=False,
                access_policy=(
                    "source_bound_noninterference_and_continuation_proofs_only"
                ),
            ),
        }
    )
)


def build_open_ecology_seed_registry() -> dict[str, list[int]]:
    """Build fresh development-only environment, learner, and genome axes."""

    prior = {
        seed
        for registry in (
            build_recurrent_seed_registry(),
            build_scale_development_seed_registry(),
            build_scale_development_v2_seed_registry(),
        )
        for seeds in registry.values()
        for seed in seeds
    }
    seen = set(prior)
    registry: dict[str, list[int]] = {}
    for role, count in OPEN_ECOLOGY_SEED_ROLE_COUNTS:
        if role == "open_ecology_genome_stream":
            registry[role] = _generate_genome_stream_seeds(role=role, count=count)
            continue
        values: list[int] = []
        index = 0
        while len(values) < count:
            digest = hashlib.sha256(
                f"{OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE}|{role}|{index:04d}".encode(
                    "utf-8"
                )
            ).digest()
            index += 1
            candidate = 1 + (int.from_bytes(digest[:8], "big") % (MAX_SEED - 1))
            if candidate in seen:
                continue
            seen.add(candidate)
            values.append(candidate)
        registry[role] = values
    return registry


def validate_open_ecology_seed_registry(
    registry: Mapping[str, Sequence[int]],
) -> None:
    expected = build_open_ecology_seed_registry()
    if set(registry) != set(expected):
        raise OpenEcologySeedRegistryError(
            "open-ecology seed registry roles do not match the contract"
        )
    if set(registry).intersection(OPEN_ECOLOGY_UNAVAILABLE_ROLES):
        raise OpenEcologySeedRegistryError(
            "open-ecology development cannot expose validation or lockbox seeds"
        )

    flattened_public_axes: list[int] = []
    for role, expected_values in expected.items():
        actual_values = list(registry[role])
        if actual_values != expected_values:
            raise OpenEcologySeedRegistryError(
                f"open-ecology seed role {role!r} is not canonical"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in actual_values
        ):
            raise OpenEcologySeedRegistryError(
                f"open-ecology seed role {role!r} contains non-integers"
            )
        if role == "open_ecology_genome_stream":
            if any(not 2**63 <= seed <= 2**64 - 1 for seed in actual_values):
                raise OpenEcologySeedRegistryError(
                    "genome-stream seeds must occupy the unsigned high-64-bit range"
                )
            if len(actual_values) != len(set(actual_values)):
                raise OpenEcologySeedRegistryError(
                    "open-ecology genome-stream seeds overlap"
                )
        else:
            if any(not 1 <= seed <= MAX_SEED for seed in actual_values):
                raise OpenEcologySeedRegistryError(
                    f"open-ecology seed role {role!r} is outside int32 range"
                )
            flattened_public_axes.extend(actual_values)

    if len(flattened_public_axes) != len(set(flattened_public_axes)):
        raise OpenEcologySeedRegistryError(
            "open-ecology environment and learner roles overlap"
        )
    prior = {
        seed
        for prior_registry in (
            build_recurrent_seed_registry(),
            build_scale_development_seed_registry(),
            build_scale_development_v2_seed_registry(),
        )
        for values in prior_registry.values()
        for seed in values
    }
    if not set(flattened_public_axes).isdisjoint(prior):
        raise OpenEcologySeedRegistryError(
            "open-ecology seeds overlap a prior recurrent registry"
        )


def open_ecology_seed_registry_payload() -> dict[str, list[int]]:
    return {role: list(values) for role, values in OPEN_ECOLOGY_SEED_REGISTRY.items()}


def open_ecology_seed_registry_json() -> bytes:
    return json.dumps(
        open_ecology_seed_registry_payload(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def open_ecology_seeds_for_role(role: str) -> tuple[int, ...]:
    if not isinstance(role, str) or not role or role != role.strip():
        raise OpenEcologySeedRegistryError(
            "open-ecology seed role must be non-empty trimmed text"
        )
    if role in OPEN_ECOLOGY_UNAVAILABLE_ROLES:
        raise OpenEcologySeedRegistryError(
            f"open-ecology access to sealed role {role!r} is forbidden"
        )
    try:
        return OPEN_ECOLOGY_SEED_REGISTRY[role]
    except KeyError as error:
        raise OpenEcologySeedRegistryError(
            f"unknown open-ecology seed role: {role!r}"
        ) from error


def open_ecology_seed_registry_contract() -> dict[str, object]:
    return {
        "schema_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
        "namespace": OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE,
        "generator": {
            "environment_and_learner": (
                "sha256_first_8_bytes_mod_int32_disjoint_from_prior_registries"
            ),
            "controller_genome": (
                "sha256_first_8_bytes_mapped_to_unsigned_high_64_bit_range"
            ),
        },
        "canonical_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        "role_order": list(dict(OPEN_ECOLOGY_SEED_ROLE_COUNTS)),
        "role_counts": dict(OPEN_ECOLOGY_SEED_ROLE_COUNTS),
        "policies": {
            role: asdict(policy)
            for role, policy in OPEN_ECOLOGY_SEED_ROLE_POLICIES.items()
        },
        "operational_benchmark_allocation": (
            open_ecology_operational_benchmark_seed_contract()
        ),
        "seeds": open_ecology_seed_registry_payload(),
        "unavailable_roles": {
            role: {
                "available": False,
                "seed_values_included": False,
                "reason": "separately_sealed_not_available_to_open_ecology_development",
            }
            for role in OPEN_ECOLOGY_UNAVAILABLE_ROLES
        },
    }


def open_ecology_operational_benchmark_seed_contract() -> dict[str, object]:
    """Return the disjoint, non-scientific allocation within the benchmark role."""

    return {
        "seed_role": OPEN_ECOLOGY_BENCHMARK_SEED_ROLE,
        "phase_a_training": {
            "environment_seed_indices": list(
                OPEN_ECOLOGY_PHASE_A_BENCHMARK_ENVIRONMENT_SEED_INDICES
            ),
            "model_initialization_seed_index": (
                OPEN_ECOLOGY_PHASE_A_BENCHMARK_MODEL_SEED_INDEX
            ),
            "genome_stream_seed_index": (
                OPEN_ECOLOGY_PHASE_A_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ),
        },
        "phase_b_training": {
            "environment_seed_indices": list(
                OPEN_ECOLOGY_PHASE_B_BENCHMARK_ENVIRONMENT_SEED_INDICES
            ),
            "model_initialization_seed_index": (
                OPEN_ECOLOGY_PHASE_B_BENCHMARK_MODEL_SEED_INDEX
            ),
            "genome_stream_seed_index": (
                OPEN_ECOLOGY_PHASE_B_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ),
        },
        "selection": {
            "environment_seed_indices": list(
                OPEN_ECOLOGY_SELECTION_BENCHMARK_ENVIRONMENT_SEED_INDICES
            ),
            "model_initialization_seed_index": (
                OPEN_ECOLOGY_SELECTION_BENCHMARK_MODEL_SEED_INDEX
            ),
            "genome_stream_seed_index": (
                OPEN_ECOLOGY_SELECTION_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ),
        },
        "phase_d_throughput": {
            "environment_seed_indices": [
                OPEN_ECOLOGY_PHASE_D_BENCHMARK_ENVIRONMENT_SEED_INDEX
            ],
            "environment_pairing": "same_environment_root_across_h_z_r",
            "model_initialization_seed_index": (
                OPEN_ECOLOGY_PHASE_D_BENCHMARK_MODEL_SEED_INDEX
            ),
            "genome_stream_seed_index": (
                OPEN_ECOLOGY_PHASE_D_BENCHMARK_GENOME_STREAM_SEED_INDEX
            ),
        },
        "reserved_seed_indices": list(OPEN_ECOLOGY_BENCHMARK_RESERVED_SEED_INDICES),
        "scientific_seed_roles_accessed": [],
    }


def _validate_operational_benchmark_seed_contract() -> None:
    assigned = (
        *OPEN_ECOLOGY_PHASE_A_BENCHMARK_ENVIRONMENT_SEED_INDICES,
        OPEN_ECOLOGY_PHASE_A_BENCHMARK_MODEL_SEED_INDEX,
        OPEN_ECOLOGY_PHASE_A_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        *OPEN_ECOLOGY_PHASE_B_BENCHMARK_ENVIRONMENT_SEED_INDICES,
        OPEN_ECOLOGY_PHASE_B_BENCHMARK_MODEL_SEED_INDEX,
        OPEN_ECOLOGY_PHASE_B_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        *OPEN_ECOLOGY_SELECTION_BENCHMARK_ENVIRONMENT_SEED_INDICES,
        OPEN_ECOLOGY_SELECTION_BENCHMARK_MODEL_SEED_INDEX,
        OPEN_ECOLOGY_SELECTION_BENCHMARK_GENOME_STREAM_SEED_INDEX,
        OPEN_ECOLOGY_PHASE_D_BENCHMARK_ENVIRONMENT_SEED_INDEX,
        OPEN_ECOLOGY_PHASE_D_BENCHMARK_MODEL_SEED_INDEX,
        OPEN_ECOLOGY_PHASE_D_BENCHMARK_GENOME_STREAM_SEED_INDEX,
    )
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("operational benchmark seed subaxes overlap")
    if set(assigned).intersection(OPEN_ECOLOGY_BENCHMARK_RESERVED_SEED_INDICES):
        raise RuntimeError("operational benchmark allocation consumes reserved seeds")
    if set(assigned).union(OPEN_ECOLOGY_BENCHMARK_RESERVED_SEED_INDICES) != set(
        range(dict(OPEN_ECOLOGY_SEED_ROLE_COUNTS)[OPEN_ECOLOGY_BENCHMARK_SEED_ROLE])
    ):
        raise RuntimeError("operational benchmark allocation is incomplete")


def _generate_genome_stream_seeds(*, role: str, count: int) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    index = 0
    while len(values) < count:
        digest = hashlib.sha256(
            f"{OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE}|{role}|{index:04d}".encode("utf-8")
        ).digest()
        index += 1
        candidate = 2**63 + (int.from_bytes(digest[:8], "big") % 2**63)
        if candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)
    return values


_validate_operational_benchmark_seed_contract()
_BUILT_OPEN_ECOLOGY_REGISTRY = build_open_ecology_seed_registry()
validate_open_ecology_seed_registry(_BUILT_OPEN_ECOLOGY_REGISTRY)
OPEN_ECOLOGY_SEED_REGISTRY: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {role: tuple(values) for role, values in _BUILT_OPEN_ECOLOGY_REGISTRY.items()}
)
OPEN_ECOLOGY_GENERATED_SHA256 = hashlib.sha256(
    json.dumps(
        _BUILT_OPEN_ECOLOGY_REGISTRY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

if OPEN_ECOLOGY_GENERATED_SHA256 != OPEN_ECOLOGY_CANONICAL_SHA256:
    raise RuntimeError("generated open-ecology seed registry digest changed")
if OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"] != (
    OPEN_ECOLOGY_EXPECTED_LEARNER_SEEDS
):
    raise RuntimeError("open-ecology learner seed values changed")
if OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][:8] != (
    OPEN_ECOLOGY_EXPECTED_FIRST_GENOME_STREAM_SEEDS
):
    raise RuntimeError("open-ecology genome-stream seed values changed")
