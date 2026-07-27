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


OPEN_ECOLOGY_SEED_REGISTRY_VERSION = "mind_v3_open_ecology_development_seed_registry_v1"
OPEN_ECOLOGY_SEED_REGISTRY_NAMESPACE = (
    "evolution-sim|mind-v3-open-ecology|development-seed-registry-v1|2026-07-27"
)
OPEN_ECOLOGY_SEED_ROLE_COUNTS: tuple[tuple[str, int], ...] = (
    ("open_ecology_train", 512),
    ("open_ecology_selection", 128),
    ("open_ecology_island", 32),
    ("open_ecology_learner", 8),
    ("open_ecology_genome_stream", 32),
)
OPEN_ECOLOGY_UNAVAILABLE_ROLES: tuple[str, ...] = ("validation", "lockbox")
OPEN_ECOLOGY_CANONICAL_SHA256 = (
    "3614d3e2ff0feba461a968e8128d84e0087beda842153e9f251abca482ab8e08"
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
