from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType


SEED_REGISTRY_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ppo|seed-registry-v1|2026-07-21"
)
MAX_SEED = 2_147_483_647
LEGACY_DIAGNOSTIC_SEEDS: tuple[int, ...] = (5, 13, 19, 29, 37, 41, 43)
SEED_ROLE_COUNTS: tuple[tuple[str, int], ...] = (
    ("curriculum", 32),
    ("train", 128),
    ("selection", 32),
    ("validation", 32),
    ("lockbox", 64),
    ("learner_development", 5),
    ("learner_confirmation", 5),
)
EXPECTED_LEARNER_DEVELOPMENT_SEEDS: tuple[int, ...] = (
    404_337_389,
    709_266_037,
    1_034_062_175,
    1_777_057_840,
    176_114_824,
)
EXPECTED_LEARNER_CONFIRMATION_SEEDS: tuple[int, ...] = (
    2_005_892_729,
    1_255_249_095,
    1_646_721_240,
    1_956_183_753,
    2_051_309_122,
)
CANONICAL_SEED_REGISTRY_SHA256 = (
    "3e37cdfaf5e65c48f5c81a4c344041a7642f630f057b77c008ba98df1970daaf"
)

# Scale-development seeds intentionally live in a second versioned namespace.
# They are not an extension of the v1 mapping: consumers must opt in to this
# registry and therefore cannot silently consume the older development,
# validation, or promotion-lockbox roles.
SCALE_DEVELOPMENT_SEED_REGISTRY_VERSION = (
    "mind_v3_public_recurrent_scale_development_seed_registry_v1"
)
SCALE_DEVELOPMENT_SEED_REGISTRY_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ppo|"
    "scale-development-seed-registry-v1|2026-07-22"
)
SCALE_DEVELOPMENT_SEED_ROLE_COUNTS: tuple[tuple[str, int], ...] = (
    ("scale_curriculum", 128),
    ("scale_train", 512),
    ("scale_selection", 128),
    ("scale_learner", 8),
)
SCALE_DEVELOPMENT_UNAVAILABLE_ROLES: tuple[str, ...] = ("validation", "lockbox")
SCALE_DEVELOPMENT_CANONICAL_SHA256 = (
    "42d6b8aa68340379cef16ac48278a579643f15a28497ea4b07e28c90bc6b8557"
)
SCALE_DEVELOPMENT_EXPECTED_LEARNER_SEEDS: tuple[int, ...] = (
    1_171_366_450,
    816_635_699,
    583_118_350,
    681_152_893,
    843_242_413,
    2_113_615_877,
    456_099_216,
    1_782_198_429,
)

# The first scale campaign consumed every v1 development role.  Replacement
# campaigns therefore use a third namespace with distinct role names so v1
# reports and artifacts remain replay-valid while new code cannot silently
# reuse any learner, optimization, or selection seed.
SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION = (
    "mind_v3_public_recurrent_scale_development_seed_registry_v2"
)
SCALE_DEVELOPMENT_V2_SEED_REGISTRY_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ppo|"
    "scale-development-seed-registry-v2|2026-07-23"
)
SCALE_DEVELOPMENT_V2_SEED_ROLE_COUNTS: tuple[tuple[str, int], ...] = (
    ("scale_v2_curriculum", 128),
    ("scale_v2_train", 512),
    ("scale_v2_selection", 128),
    ("scale_v2_learner", 8),
)
SCALE_DEVELOPMENT_V2_UNAVAILABLE_ROLES: tuple[str, ...] = (
    "validation",
    "lockbox",
)
SCALE_DEVELOPMENT_V2_CANONICAL_SHA256 = (
    "1531a9e782dc414ae43c37346a45d581cb00a29a8348e0bd902c106166ba5ee7"
)
SCALE_DEVELOPMENT_V2_EXPECTED_LEARNER_SEEDS: tuple[int, ...] = (
    1_505_354_251,
    1_916_254_251,
    870_080_605,
    699_185_506,
    58_022_765,
    2_067_239_703,
    1_485_353_601,
    2_083_034_500,
)


class SeedRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class SeedRolePolicy:
    axis: str
    lifecycle: str
    may_tune_configuration: bool
    promotion_evidence: bool
    access_policy: str


SEED_ROLE_POLICIES: Mapping[str, SeedRolePolicy] = MappingProxyType(
    {
        "diagnostic_red_team": SeedRolePolicy(
            axis="environment",
            lifecycle="legacy_diagnostic_only",
            may_tune_configuration=False,
            promotion_evidence=False,
            access_policy="repeatable_but_never_clean_heldout",
        ),
        "curriculum": SeedRolePolicy(
            axis="environment",
            lifecycle="optimization_curriculum",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_training",
        ),
        "train": SeedRolePolicy(
            axis="environment",
            lifecycle="optimization_training",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_training",
        ),
        "selection": SeedRolePolicy(
            axis="environment",
            lifecycle="development_selection",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_pre_registered_candidate_selection",
        ),
        "validation": SeedRolePolicy(
            axis="environment",
            lifecycle="limited_validation",
            may_tune_configuration=False,
            promotion_evidence=False,
            access_policy="limited_access_after_candidate_selection",
        ),
        "lockbox": SeedRolePolicy(
            axis="environment",
            lifecycle="sealed_promotion_lockbox",
            may_tune_configuration=False,
            promotion_evidence=True,
            access_policy="one_time_after_candidate_and_configuration_freeze",
        ),
        "learner_development": SeedRolePolicy(
            axis="learner",
            lifecycle="learner_development",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_training_variance_and_selection",
        ),
        "learner_confirmation": SeedRolePolicy(
            axis="learner",
            lifecycle="learner_confirmation",
            may_tune_configuration=False,
            promotion_evidence=False,
            access_policy="after_candidate_and_configuration_freeze",
        ),
    }
)

SCALE_DEVELOPMENT_SEED_ROLE_POLICIES: Mapping[str, SeedRolePolicy] = MappingProxyType(
    {
        "scale_curriculum": SeedRolePolicy(
            axis="environment",
            lifecycle="scale_optimization_curriculum",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_scale_training_only",
        ),
        "scale_train": SeedRolePolicy(
            axis="environment",
            lifecycle="scale_optimization_training",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_scale_training_only",
        ),
        "scale_selection": SeedRolePolicy(
            axis="environment",
            lifecycle="scale_development_selection",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_preregistered_scale_selection",
        ),
        "scale_learner": SeedRolePolicy(
            axis="learner",
            lifecycle="scale_learner_development",
            may_tune_configuration=True,
            promotion_evidence=False,
            access_policy="reusable_for_independent_scale_replication",
        ),
    }
)

SCALE_DEVELOPMENT_V2_SEED_ROLE_POLICIES: Mapping[str, SeedRolePolicy] = (
    MappingProxyType(
        {
            "scale_v2_curriculum": SeedRolePolicy(
                axis="environment",
                lifecycle="scale_v2_optimization_curriculum",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_scale_v2_training_only",
            ),
            "scale_v2_train": SeedRolePolicy(
                axis="environment",
                lifecycle="scale_v2_optimization_training",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_scale_v2_training_only",
            ),
            "scale_v2_selection": SeedRolePolicy(
                axis="environment",
                lifecycle="scale_v2_development_selection",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_preregistered_scale_v2_selection",
            ),
            "scale_v2_learner": SeedRolePolicy(
                axis="learner",
                lifecycle="scale_v2_learner_development",
                may_tune_configuration=True,
                promotion_evidence=False,
                access_policy="reusable_for_independent_scale_v2_replication",
            ),
        }
    )
)


def build_recurrent_seed_registry() -> dict[str, list[int]]:
    """Build the deterministic, globally disjoint Mind v3 seed registry."""

    seen = set(LEGACY_DIAGNOSTIC_SEEDS)
    registry = {"diagnostic_red_team": list(LEGACY_DIAGNOSTIC_SEEDS)}

    for role, count in SEED_ROLE_COUNTS:
        seeds: list[int] = []
        index = 0
        while len(seeds) < count:
            material = f"{SEED_REGISTRY_NAMESPACE}|{role}|{index:04d}".encode("utf-8")
            digest = hashlib.sha256(material).digest()
            candidate = 1 + (int.from_bytes(digest[:8], "big") % (MAX_SEED - 1))
            index += 1
            if candidate in seen:
                continue
            seen.add(candidate)
            seeds.append(candidate)
        registry[role] = seeds

    return registry


def validate_recurrent_seed_registry(
    registry: Mapping[str, Sequence[int]],
) -> None:
    expected = build_recurrent_seed_registry()
    if set(registry) != set(expected):
        raise SeedRegistryError("seed registry roles do not match the contract")

    flattened: list[int] = []
    for role, expected_seeds in expected.items():
        actual_seeds = list(registry[role])
        if actual_seeds != expected_seeds:
            raise SeedRegistryError(f"seed registry role {role!r} is not canonical")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in actual_seeds
        ):
            raise SeedRegistryError(
                f"seed registry role {role!r} contains non-integers"
            )
        if any(seed < 1 or seed > MAX_SEED for seed in actual_seeds):
            raise SeedRegistryError(f"seed registry role {role!r} is out of range")
        flattened.extend(actual_seeds)

    if len(flattened) != len(set(flattened)):
        raise SeedRegistryError("seed registry roles overlap")


def canonical_seed_registry_payload() -> dict[str, list[int]]:
    return {role: list(seeds) for role, seeds in RECURRENT_SEED_REGISTRY.items()}


def canonical_seed_registry_json() -> bytes:
    return json.dumps(
        canonical_seed_registry_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def recurrent_seed_registry_contract() -> dict[str, object]:
    return {
        "namespace": SEED_REGISTRY_NAMESPACE,
        "generator": "sha256_first_8_bytes_mod_int32_v1",
        "canonical_sha256": CANONICAL_SEED_REGISTRY_SHA256,
        "role_order": ["diagnostic_red_team", *dict(SEED_ROLE_COUNTS)],
        "role_counts": {
            "diagnostic_red_team": len(LEGACY_DIAGNOSTIC_SEEDS),
            **dict(SEED_ROLE_COUNTS),
        },
        "policies": {
            role: asdict(policy) for role, policy in SEED_ROLE_POLICIES.items()
        },
        "seeds": canonical_seed_registry_payload(),
    }


def build_scale_development_seed_registry() -> dict[str, list[int]]:
    """Build fresh scale-only seeds disjoint from every v1 registry role."""

    existing = build_recurrent_seed_registry()
    seen = {seed for seeds in existing.values() for seed in seeds}
    registry: dict[str, list[int]] = {}
    for role, count in SCALE_DEVELOPMENT_SEED_ROLE_COUNTS:
        seeds: list[int] = []
        index = 0
        while len(seeds) < count:
            material = (
                f"{SCALE_DEVELOPMENT_SEED_REGISTRY_NAMESPACE}|{role}|{index:04d}"
            ).encode("utf-8")
            digest = hashlib.sha256(material).digest()
            candidate = 1 + (int.from_bytes(digest[:8], "big") % (MAX_SEED - 1))
            index += 1
            if candidate in seen:
                continue
            seen.add(candidate)
            seeds.append(candidate)
        registry[role] = seeds
    return registry


def validate_scale_development_seed_registry(
    registry: Mapping[str, Sequence[int]],
) -> None:
    """Fail closed on drift, overlap, or sealed-role exposure."""

    expected = build_scale_development_seed_registry()
    if set(registry) != set(expected):
        raise SeedRegistryError(
            "scale-development seed registry roles do not match the contract"
        )
    if set(registry).intersection(SCALE_DEVELOPMENT_UNAVAILABLE_ROLES):
        raise SeedRegistryError(
            "scale-development registry must not expose validation or lockbox seeds"
        )

    flattened: list[int] = []
    for role, expected_seeds in expected.items():
        actual_seeds = list(registry[role])
        if actual_seeds != expected_seeds:
            raise SeedRegistryError(
                f"scale-development seed registry role {role!r} is not canonical"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in actual_seeds
        ):
            raise SeedRegistryError(
                f"scale-development seed registry role {role!r} contains non-integers"
            )
        if any(seed < 1 or seed > MAX_SEED for seed in actual_seeds):
            raise SeedRegistryError(
                f"scale-development seed registry role {role!r} is out of range"
            )
        flattened.extend(actual_seeds)

    if len(flattened) != len(set(flattened)):
        raise SeedRegistryError("scale-development seed registry roles overlap")
    existing = {
        seed for seeds in build_recurrent_seed_registry().values() for seed in seeds
    }
    if not set(flattened).isdisjoint(existing):
        raise SeedRegistryError(
            "scale-development seeds overlap the existing recurrent registry"
        )


def build_scale_development_v2_seed_registry() -> dict[str, list[int]]:
    """Build fresh scale-v2 seeds disjoint from base and scale-v1 roles."""

    existing = build_recurrent_seed_registry()
    scale_v1 = build_scale_development_seed_registry()
    seen = {
        seed
        for registry in (existing, scale_v1)
        for seeds in registry.values()
        for seed in seeds
    }
    registry: dict[str, list[int]] = {}
    for role, count in SCALE_DEVELOPMENT_V2_SEED_ROLE_COUNTS:
        seeds: list[int] = []
        index = 0
        while len(seeds) < count:
            material = (
                f"{SCALE_DEVELOPMENT_V2_SEED_REGISTRY_NAMESPACE}|"
                f"{role}|{index:04d}"
            ).encode("utf-8")
            digest = hashlib.sha256(material).digest()
            candidate = 1 + (int.from_bytes(digest[:8], "big") % (MAX_SEED - 1))
            index += 1
            if candidate in seen:
                continue
            seen.add(candidate)
            seeds.append(candidate)
        registry[role] = seeds
    return registry


def validate_scale_development_v2_seed_registry(
    registry: Mapping[str, Sequence[int]],
) -> None:
    """Fail closed on v2 drift, overlap, or sealed-role exposure."""

    expected = build_scale_development_v2_seed_registry()
    if set(registry) != set(expected):
        raise SeedRegistryError(
            "scale-v2 development seed registry roles do not match the contract"
        )
    if set(registry).intersection(SCALE_DEVELOPMENT_V2_UNAVAILABLE_ROLES):
        raise SeedRegistryError(
            "scale-v2 development registry must not expose validation or lockbox seeds"
        )

    flattened: list[int] = []
    for role, expected_seeds in expected.items():
        actual_seeds = list(registry[role])
        if actual_seeds != expected_seeds:
            raise SeedRegistryError(
                f"scale-v2 development seed registry role {role!r} is not canonical"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in actual_seeds
        ):
            raise SeedRegistryError(
                f"scale-v2 development seed registry role {role!r} "
                "contains non-integers"
            )
        if any(seed < 1 or seed > MAX_SEED for seed in actual_seeds):
            raise SeedRegistryError(
                f"scale-v2 development seed registry role {role!r} is out of range"
            )
        flattened.extend(actual_seeds)

    if len(flattened) != len(set(flattened)):
        raise SeedRegistryError("scale-v2 development seed registry roles overlap")
    prior = {
        seed
        for registry in (
            build_recurrent_seed_registry(),
            build_scale_development_seed_registry(),
        )
        for seeds in registry.values()
        for seed in seeds
    }
    if not set(flattened).isdisjoint(prior):
        raise SeedRegistryError(
            "scale-v2 development seeds overlap a prior recurrent registry"
        )


def scale_development_seed_registry_payload() -> dict[str, list[int]]:
    return {
        role: list(seeds) for role, seeds in SCALE_DEVELOPMENT_SEED_REGISTRY.items()
    }


def scale_development_seed_registry_json() -> bytes:
    return json.dumps(
        scale_development_seed_registry_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def scale_development_seeds_for_role(role: str) -> tuple[int, ...]:
    """Return one opt-in scale role while keeping held-out roles unavailable."""

    if not isinstance(role, str) or not role or role != role.strip():
        raise SeedRegistryError("scale-development seed role must be trimmed text")
    if role in SCALE_DEVELOPMENT_UNAVAILABLE_ROLES:
        raise SeedRegistryError(
            f"scale-development access to sealed role {role!r} is forbidden"
        )
    try:
        return SCALE_DEVELOPMENT_SEED_REGISTRY[role]
    except KeyError as exc:
        raise SeedRegistryError(
            f"unknown scale-development seed role: {role!r}"
        ) from exc


def scale_development_seed_registry_contract() -> dict[str, object]:
    """Describe scale roles without materializing validation/lockbox seeds."""

    return {
        "schema_version": SCALE_DEVELOPMENT_SEED_REGISTRY_VERSION,
        "namespace": SCALE_DEVELOPMENT_SEED_REGISTRY_NAMESPACE,
        "generator": "sha256_first_8_bytes_mod_int32_disjoint_from_v1",
        "canonical_sha256": SCALE_DEVELOPMENT_CANONICAL_SHA256,
        "disjoint_from_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
        "role_order": list(dict(SCALE_DEVELOPMENT_SEED_ROLE_COUNTS)),
        "role_counts": dict(SCALE_DEVELOPMENT_SEED_ROLE_COUNTS),
        "policies": {
            role: asdict(policy)
            for role, policy in SCALE_DEVELOPMENT_SEED_ROLE_POLICIES.items()
        },
        "seeds": scale_development_seed_registry_payload(),
        "unavailable_roles": {
            role: {
                "available": False,
                "seed_values_included": False,
                "reason": "separately_sealed_not_available_to_scale_development",
            }
            for role in SCALE_DEVELOPMENT_UNAVAILABLE_ROLES
        },
    }


def scale_development_v2_seed_registry_payload() -> dict[str, list[int]]:
    return {
        role: list(seeds)
        for role, seeds in SCALE_DEVELOPMENT_V2_SEED_REGISTRY.items()
    }


def scale_development_v2_seed_registry_json() -> bytes:
    return json.dumps(
        scale_development_v2_seed_registry_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def scale_development_v2_seeds_for_role(role: str) -> tuple[int, ...]:
    """Return one opt-in scale-v2 role; sealed roles remain unavailable."""

    if not isinstance(role, str) or not role or role != role.strip():
        raise SeedRegistryError("scale-v2 development seed role must be trimmed text")
    if role in SCALE_DEVELOPMENT_V2_UNAVAILABLE_ROLES:
        raise SeedRegistryError(
            f"scale-v2 development access to sealed role {role!r} is forbidden"
        )
    try:
        return SCALE_DEVELOPMENT_V2_SEED_REGISTRY[role]
    except KeyError as exc:
        raise SeedRegistryError(
            f"unknown scale-v2 development seed role: {role!r}"
        ) from exc


def scale_development_v2_seed_registry_contract() -> dict[str, object]:
    """Describe v2 roles without materializing validation/lockbox seeds."""

    return {
        "schema_version": SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
        "namespace": SCALE_DEVELOPMENT_V2_SEED_REGISTRY_NAMESPACE,
        "generator": (
            "sha256_first_8_bytes_mod_int32_disjoint_from_base_and_scale_v1"
        ),
        "canonical_sha256": SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
        "disjoint_from_registry_sha256": [
            CANONICAL_SEED_REGISTRY_SHA256,
            SCALE_DEVELOPMENT_CANONICAL_SHA256,
        ],
        "role_order": list(dict(SCALE_DEVELOPMENT_V2_SEED_ROLE_COUNTS)),
        "role_counts": dict(SCALE_DEVELOPMENT_V2_SEED_ROLE_COUNTS),
        "policies": {
            role: asdict(policy)
            for role, policy in SCALE_DEVELOPMENT_V2_SEED_ROLE_POLICIES.items()
        },
        "seeds": scale_development_v2_seed_registry_payload(),
        "unavailable_roles": {
            role: {
                "available": False,
                "seed_values_included": False,
                "reason": "separately_sealed_not_available_to_scale_v2_development",
            }
            for role in SCALE_DEVELOPMENT_V2_UNAVAILABLE_ROLES
        },
    }


_BUILT_REGISTRY = build_recurrent_seed_registry()
validate_recurrent_seed_registry(_BUILT_REGISTRY)
RECURRENT_SEED_REGISTRY: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {role: tuple(seeds) for role, seeds in _BUILT_REGISTRY.items()}
)
GENERATED_SEED_REGISTRY_SHA256 = hashlib.sha256(
    json.dumps(
        _BUILT_REGISTRY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

if GENERATED_SEED_REGISTRY_SHA256 != CANONICAL_SEED_REGISTRY_SHA256:
    raise RuntimeError("generated recurrent seed registry digest changed")
if RECURRENT_SEED_REGISTRY["learner_development"] != (
    EXPECTED_LEARNER_DEVELOPMENT_SEEDS
):
    raise RuntimeError("learner development seed values changed")
if RECURRENT_SEED_REGISTRY["learner_confirmation"] != (
    EXPECTED_LEARNER_CONFIRMATION_SEEDS
):
    raise RuntimeError("learner confirmation seed values changed")

_BUILT_SCALE_DEVELOPMENT_REGISTRY = build_scale_development_seed_registry()
validate_scale_development_seed_registry(_BUILT_SCALE_DEVELOPMENT_REGISTRY)
SCALE_DEVELOPMENT_SEED_REGISTRY: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {role: tuple(seeds) for role, seeds in _BUILT_SCALE_DEVELOPMENT_REGISTRY.items()}
)
SCALE_DEVELOPMENT_GENERATED_SHA256 = hashlib.sha256(
    json.dumps(
        _BUILT_SCALE_DEVELOPMENT_REGISTRY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

if SCALE_DEVELOPMENT_GENERATED_SHA256 != SCALE_DEVELOPMENT_CANONICAL_SHA256:
    raise RuntimeError("generated scale-development seed registry digest changed")
if SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"] != (
    SCALE_DEVELOPMENT_EXPECTED_LEARNER_SEEDS
):
    raise RuntimeError("scale-development learner seed values changed")

_BUILT_SCALE_DEVELOPMENT_V2_REGISTRY = (
    build_scale_development_v2_seed_registry()
)
validate_scale_development_v2_seed_registry(
    _BUILT_SCALE_DEVELOPMENT_V2_REGISTRY
)
SCALE_DEVELOPMENT_V2_SEED_REGISTRY: Mapping[str, tuple[int, ...]] = (
    MappingProxyType(
        {
            role: tuple(seeds)
            for role, seeds in _BUILT_SCALE_DEVELOPMENT_V2_REGISTRY.items()
        }
    )
)
SCALE_DEVELOPMENT_V2_GENERATED_SHA256 = hashlib.sha256(
    json.dumps(
        _BUILT_SCALE_DEVELOPMENT_V2_REGISTRY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

if (
    SCALE_DEVELOPMENT_V2_GENERATED_SHA256
    != SCALE_DEVELOPMENT_V2_CANONICAL_SHA256
):
    raise RuntimeError("generated scale-v2 development seed registry digest changed")
if SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_learner"] != (
    SCALE_DEVELOPMENT_V2_EXPECTED_LEARNER_SEEDS
):
    raise RuntimeError("scale-v2 development learner seed values changed")
