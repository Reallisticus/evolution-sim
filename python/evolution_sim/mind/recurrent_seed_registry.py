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
