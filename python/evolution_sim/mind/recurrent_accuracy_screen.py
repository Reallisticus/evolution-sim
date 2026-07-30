"""Prospective, bounded accuracy corpus for recurrent linear-kernel screening.

This module is deliberately development-only.  It does not select a candidate,
change runtime dispatch, alter an action, or create launch authority.  Its
purpose is to make the numerical corpus and the first-/second-order gradient
probe reconstructible before the sealed holdout can be consumed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F

from evolution_sim.mind.provenance import stable_payload_digest


RECURRENT_ACCURACY_SCREEN_SCHEMA_VERSION: Final = (
    "mind_v3_recurrent_adapter_accuracy_development_screen_v4"
)
RECURRENT_ACCURACY_CORPUS_VERSION: Final = (
    "open_ecology_recurrent_accuracy_corpus_v1"
)
RECURRENT_ACCURACY_SCREEN_PREREGISTRATION_VERSION: Final = (
    "mind_v3_recurrent_accuracy_preregistration_v1"
)
RECURRENT_ACCURACY_SCREEN_CANDIDATE: Final = (
    "fixed_row_tile_4_fast_adapter_candidate_v3"
)
RECURRENT_ACCURACY_SCREEN_SELECTION_VERSION: Final = (
    "sha256_seeded_pin_then_rank_v1"
)
RECURRENT_ACCURACY_SCREEN_RNG_VERSION: Final = (
    "domain_separated_sha256_splitmix64_dyadic_v1"
)
RECURRENT_ACCURACY_SCREEN_GRADIENT_VERSION: Final = (
    "fixed_loss_gradient_hvp_probe_v1"
)
RECURRENT_ACCURACY_SCREEN_HOLDOUT_VERSION: Final = (
    "process_bound_one_shot_holdout_v1"
)
RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION: Final = (
    "recurrent_accuracy_selection_manifest_v1"
)
RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST: Final = (
    "61e6cd3897a1dcf08cb777f09c3238efe0d167c5c313bd9077dc86c87ef42efa"
)
RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH: Final = (
    "docs/research/open-ecology-recurrent-accuracy-screen-method-v1.md"
)
RECURRENT_ACCURACY_SOURCE_RETIREMENT_MARKER_NAME: Final = (
    "recurrent-kernel-development-screen.source-retired.json"
)
RECURRENT_ACCURACY_HOLDOUT_RETIREMENT_MARKER_NAME: Final = (
    "recurrent-kernel-development-screen.holdout-retired.json"
)
RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS: Final = (
    "python/evolution_sim/env/runtime/observations.py",
    "python/evolution_sim/mind/policy_inputs.py",
    "python/evolution_sim/mind/recurrent_actor_critic.py",
    "python/evolution_sim/mind/recurrent_experiment.py",
    "python/evolution_sim/mind/recurrent_rollout.py",
    "python/evolution_sim/mind/recurrent_policy.py",
    "python/evolution_sim/mind/recurrent_ppo.py",
    "python/evolution_sim/mind/recurrent_accuracy_screen.py",
    "python/evolution_sim/mind/open_ecology_phase_a_behavioral_evidence.py",
    "python/evolution_sim/mind/recurrent_kernel_development_screen.py",
)

# These values are part of the prospective source contract.  Tests must inject
# a synthetic AccuracySeedContract and must never consume these sealed values.
_SEALED_ORDINARY_SEED: Final = 8206998037284159040
_SEALED_HOLDOUT_SEED: Final = 3411624975195142037
_SEALED_CANCELLATION_UNDERFLOW_SEED: Final = 6319666271073447219
_SEALED_GRADIENT_SEED: Final = 5016224177442350447

_UINT64_MASK = (1 << 64) - 1
_CORPUS_DOMAIN = "oe-accuracy-corpus-v1"
_VALUE_DOMAIN = "oe-accuracy-value-v1"
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_MAXIMUM_DOT_COUNT = 4_096
_MAXIMUM_PRODUCT_COUNT = 2_500_000
_EXPECTED_FORWARD_DOT_COUNT = 1_874
_EXPECTED_FORWARD_PRODUCT_COUNT = 479_648
_EXPECTED_ORDINARY_DOT_COUNT = 1_275
_EXPECTED_ORDINARY_PRODUCT_COUNT = 328_704
_EXPECTED_HOLDOUT_DOT_COUNT = 425
_EXPECTED_HOLDOUT_PRODUCT_COUNT = 109_568
_EXPECTED_CANCELLATION_DOT_COUNT = 116
_EXPECTED_CANCELLATION_PRODUCT_COUNT = 27_584
_EXPECTED_UNDERFLOW_DOT_COUNT = 29
_EXPECTED_UNDERFLOW_PRODUCT_COUNT = 6_896
_EXPECTED_OPERATIONAL_DOT_COUNT = 425
_EXPECTED_OPERATIONAL_PRODUCT_COUNT = 109_568
_EXPECTED_GRADIENT_DIAGNOSTIC_SPOT_COUNT = 116
_EXPECTED_GRADIENT_LOSS_TERM_COUNT = 11_625
_EXPECTED_GRADIENT_COMPONENT_COUNT = 572_033
_EXPECTED_GRADIENT_AND_HVP_COMPONENT_COUNT = 1_144_066
_GRADIENT_RELATIVE_TOLERANCE = 2e-5
_GRADIENT_ABSOLUTE_TOLERANCE = 2e-6
_HVP_RELATIVE_TOLERANCE = 5e-5
_HVP_ABSOLUTE_TOLERANCE = 5e-6
_D04_RELATIVE_TOLERANCE = 1e-5
_D04_ABSOLUTE_TOLERANCE = 1e-6
_FLOAT32_SMALLEST_NORMAL = 2.0**-126
_SOURCE_SHA_LENGTH = 40
_INDEPENDENT_LINEAR_ORACLE_VERSION = (
    "python_float64_products_math_fsum_single_float32_round_v2"
)
_INDEPENDENT_LINEAR_ORACLE_SOURCE_PATH = (
    "python/evolution_sim/mind/recurrent_accuracy_screen.py"
)
_INDEPENDENT_ORACLE_ERROR_METRICS_VERSION = (
    "authenticated_monotone_gamma_and_d04_float32_forward_error_metrics_v4"
)
_INDEPENDENT_LINEAR_ORACLE_MAXIMUM_TENSOR_ELEMENTS = 1_000_000
_INDEPENDENT_LINEAR_ORACLE_MAXIMUM_PRODUCTS = 1_000_000
_FLOAT32_UNIT_ROUNDOFF = 2.0**-24
_FLOAT32_SMALLEST_SUBNORMAL = 2.0**-149

LinearForward = Callable[[Tensor, Tensor, Tensor | None], Tensor]


class RecurrentKernelDevelopmentScreenError(ValueError):
    """Raised when the development-only accuracy contract fails closed."""


# Descriptive public alias; the legacy name remains the shared contract type
# consumed by the development-screen integration.
RecurrentAccuracyScreenError = RecurrentKernelDevelopmentScreenError


_HOLDOUT_REPORT_KEYS: Final = frozenset(
    {
        "schema_version",
        "development_only",
        "deciding",
        "candidate_identity",
        "holdout_receipt",
        "groups",
        "aggregate",
        "exact_digest",
    }
)
_HOLDOUT_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "source_sha",
        "source_clean_detached",
        "checkout_root",
        "required_source_paths",
        "source_module_digests",
        "source_module_digest",
        "method_document_path",
        "method_document_digest",
        "selection_manifest_schema_version",
        "selection_manifest_exact_digest",
        "parent_process_id",
        "parent_process_start_identity",
        "child_process_id",
        "child_process_start_identity",
        "child_nonce",
        "report_path",
        "screen_schema_version",
        "corpus_version",
        "preregistration_digest",
        "candidate_identity",
        "preholdout_evidence_digest",
        "access_ordinal",
        "phase",
        "test_mode",
        "pytest_current_test_absent",
        "holdout_access_count",
        "holdout_consumed",
        "candidate_locked_before_holdout",
        "retirement",
        "source_retirement_marker_path",
        "source_retirement_marker_sha256",
        "source_retirement_marker_exact_digest",
        "holdout_retirement_marker_path",
        "holdout_retirement_marker_sha256",
        "holdout_retirement_marker_exact_digest",
        "capability_commitment_sha256",
        "closing_baseline_evidence_digest",
        "seed_digest",
        "exact_digest",
    }
)
_HOLDOUT_CASE_KEYS: Final = frozenset(
    {
        "case",
        "family",
        "full_component_exact_digests",
        "selected_component_exact_digests",
        "metrics",
        "per_dot_classification",
        "serialized_selected_values",
        "exact_digest",
    }
)
_FULL_COMPONENT_DIGEST_KEYS: Final = frozenset(
    {"input", "weight", "bias", "candidate_output"}
)


@dataclass(frozen=True)
class AccuracySeedContract:
    """Four independent seeds used by the corpus and gradient probes."""

    ordinary: int
    holdout: int
    cancellation_underflow: int
    gradient: int
    synthetic_test_contract: bool = False


@dataclass(frozen=True)
class LinearRole:
    role: str
    input_features: int
    output_features: int


@dataclass(frozen=True)
class LinearProbeCase:
    """One factorized linear probe, including its operational shape."""

    case_id: str
    table: str
    role: str
    input_features: int
    output_features: int
    active_rows: int
    physical_rows: int
    selected_rows: tuple[int, ...]
    selected_outputs: tuple[int, ...]

    @property
    def dot_count(self) -> int:
        return len(self.selected_rows) * len(self.selected_outputs)

    @property
    def product_count(self) -> int:
        return self.dot_count * self.input_features

    def as_contract(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "table": self.table,
            "role": self.role,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "active_rows": self.active_rows,
            "physical_rows": self.physical_rows,
            "selected_rows": list(self.selected_rows),
            "selected_outputs": list(self.selected_outputs),
            "dot_count": self.dot_count,
            "product_count": self.product_count,
        }


@dataclass(frozen=True)
class GeneratedLinearCase:
    """Materialized full-shape case plus its bounded selected comparison."""

    probe: LinearProbeCase
    family: str
    inputs: Tensor
    weight: Tensor
    bias: Tensor


@dataclass(frozen=True)
class HoldoutCapability:
    """Opaque process-local capability consumed by HoldoutSeedAuthority."""

    token: str
    capability_commitment_sha256: str
    preholdout_evidence_digest: str
    source_retirement_marker_sha256: str
    holdout_retirement_marker_sha256: str
    candidate_identity: str


@dataclass(frozen=True)
class _HoldoutGenerationPermit:
    seed: int
    token: str


_HOLDOUT_GENERATION_REGISTRY: dict[str, tuple[int, set[str]]] = {}


def generate_holdout_capability_preimage() -> str:
    """Generate the private preimage before the first child starts."""

    return secrets.token_hex(32)


def holdout_capability_commitment(preimage: str) -> str:
    token = _validated_hex(
        preimage,
        length=64,
        field="holdout capability preimage",
    )
    return hashlib.sha256(bytes.fromhex(token)).hexdigest()


def issue_holdout_capability(
    *,
    preimage: str,
    capability_commitment_sha256: str,
    preholdout_evidence_digest: str,
    source_retirement_marker_sha256: str,
    holdout_retirement_marker_sha256: str,
    candidate_identity: str,
    closing_baseline_validated: bool,
    source_retirement_marker_exists: bool,
    holdout_retirement_marker_exists: bool,
) -> HoldoutCapability:
    """Create the private token in the parent after pre-holdout sealing."""

    if closing_baseline_validated is not True:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout capability requires a validated closing baseline"
        )
    if (
        source_retirement_marker_exists is not True
        or holdout_retirement_marker_exists is not True
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout capability requires both durable retirement markers"
        )
    if candidate_identity != RECURRENT_ACCURACY_SCREEN_CANDIDATE:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout capability candidate is not the sole frozen candidate"
        )
    commitment = _validated_hex(
        capability_commitment_sha256,
        length=64,
        field="holdout capability commitment",
    )
    if holdout_capability_commitment(preimage) != commitment:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout capability preimage does not match its commitment"
        )
    return HoldoutCapability(
        token=preimage,
        capability_commitment_sha256=commitment,
        preholdout_evidence_digest=_validated_hex(
            preholdout_evidence_digest,
            length=64,
            field="pre-holdout evidence digest",
        ),
        source_retirement_marker_sha256=_validated_hex(
            source_retirement_marker_sha256,
            length=64,
            field="source-retirement-marker SHA256",
        ),
        holdout_retirement_marker_sha256=_validated_hex(
            holdout_retirement_marker_sha256,
            length=64,
            field="holdout-retirement-marker SHA256",
        ),
        candidate_identity=candidate_identity,
    )


_LINEAR_ROLES: Final = (
    LinearRole("encoder", 604, 256),
    LinearRole("gru_input", 256, 768),
    LinearRole("gru_hidden", 256, 768),
    LinearRole("film_scale", 16, 256),
    LinearRole("film_bias", 16, 256),
    LinearRole("actor", 256, 20),
    LinearRole("value", 256, 1),
)
_ROLE_BY_NAME: Final = {role.role: role for role in _LINEAR_ROLES}
_AFFINE_ROWS: Final = {
    "encoder": 5,
    "gru_input": 9,
    "gru_hidden": 17,
    "film_scale": 3,
    "film_bias": 2,
    "actor": 33,
    "value": 1,
}
_AFFINE_ROW_SELECTIONS: Final = {
    "encoder": 4,
    "gru_input": 4,
    "gru_hidden": 4,
    "film_scale": 3,
    "film_bias": 2,
    "actor": 4,
    "value": 1,
}
_AFFINE_OUTPUT_SELECTIONS: Final = {
    "encoder": 4,
    "gru_input": 6,
    "gru_hidden": 6,
    "film_scale": 4,
    "film_bias": 4,
    "actor": 4,
    "value": 1,
}
_BUCKET_ACTIVE_ROWS: Final = (
    1,
    2,
    3,
    5,
    9,
    17,
    33,
    64,
    65,
    129,
    257,
    319,
    320,
)
_BUCKET_PHYSICAL_ROWS: Final = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    64,
    128,
    256,
    320,
    320,
    320,
)
_LEARNER_ROWS: Final = (896, 1920, 2048)
_CANCELLATION_OUTPUT_SELECTIONS: Final = {
    "encoder": 4,
    "gru_input": 6,
    "gru_hidden": 6,
    "film_scale": 4,
    "film_bias": 4,
    "actor": 4,
    "value": 1,
}
_ORDINARY_FAMILIES: Final = ("balanced", "alternating", "mixed")
_G_OUTPUTS: Final = (0, 255, 256, 511, 512, 767)

# Independently frozen from the method document.  This is intentionally
# literal rather than produced by the generator it audits.
_FROZEN_REAL_SELECTION_MANIFEST: Final = {
    "ordinary": (
        ("affine.encoder_r5", (0, 1, 2, 4), (0, 58, 93, 255)),
        ("affine.gru_input_r9", (0, 2, 4, 8), _G_OUTPUTS),
        ("affine.gru_hidden_r17", (0, 12, 15, 16), _G_OUTPUTS),
        ("affine.film_scale_r3", (0, 1, 2), (0, 25, 32, 255)),
        ("affine.film_bias_r2", (0, 1), (0, 211, 248, 255)),
        ("affine.actor_r33", (0, 19, 25, 32), (0, 7, 14, 19)),
        ("affine.value_r1", (0,), (0,)),
        ("bucket.gru_input.r1.p1", (0,), _G_OUTPUTS),
        ("bucket.gru_input.r2.p2", (0, 1), _G_OUTPUTS),
        ("bucket.gru_input.r3.p4", (0, 1, 2), _G_OUTPUTS),
        ("bucket.gru_input.r5.p8", (0, 1, 3, 4), _G_OUTPUTS),
        ("bucket.gru_input.r9.p16", (0, 2, 3, 8), _G_OUTPUTS),
        ("bucket.gru_input.r17.p32", (0, 2, 4, 16), _G_OUTPUTS),
        ("bucket.gru_input.r33.p64", (0, 13, 19, 32), _G_OUTPUTS),
        ("bucket.gru_input.r64.p64", (0, 1, 54, 63), _G_OUTPUTS),
        ("bucket.gru_input.r65.p128", (0, 23, 28, 64), _G_OUTPUTS),
        ("bucket.gru_input.r129.p256", (0, 69, 85, 128), _G_OUTPUTS),
        ("bucket.gru_input.r257.p320", (0, 51, 182, 256), _G_OUTPUTS),
        ("bucket.gru_input.r319.p320", (0, 53, 213, 318), _G_OUTPUTS),
        ("bucket.gru_input.r320.p320", (0, 33, 278, 319), _G_OUTPUTS),
        ("learner.actor.r896", (0, 302, 821, 895), (0, 6, 8, 19)),
        (
            "learner.actor.r1920",
            (0, 1114, 1616, 1919),
            (0, 9, 17, 19),
        ),
        (
            "learner.actor.r2048",
            (0, 147, 1445, 2047),
            (0, 8, 18, 19),
        ),
    ),
    "holdout": (
        ("affine.encoder_r5", (0, 1, 2, 4), (0, 29, 143, 255)),
        ("affine.gru_input_r9", (0, 1, 3, 8), _G_OUTPUTS),
        ("affine.gru_hidden_r17", (0, 1, 13, 16), _G_OUTPUTS),
        ("affine.film_scale_r3", (0, 1, 2), (0, 11, 218, 255)),
        ("affine.film_bias_r2", (0, 1), (0, 22, 118, 255)),
        ("affine.actor_r33", (0, 5, 26, 32), (0, 8, 13, 19)),
        ("affine.value_r1", (0,), (0,)),
        ("bucket.gru_input.r1.p1", (0,), _G_OUTPUTS),
        ("bucket.gru_input.r2.p2", (0, 1), _G_OUTPUTS),
        ("bucket.gru_input.r3.p4", (0, 1, 2), _G_OUTPUTS),
        ("bucket.gru_input.r5.p8", (0, 1, 2, 4), _G_OUTPUTS),
        ("bucket.gru_input.r9.p16", (0, 1, 5, 8), _G_OUTPUTS),
        ("bucket.gru_input.r17.p32", (0, 12, 14, 16), _G_OUTPUTS),
        ("bucket.gru_input.r33.p64", (0, 25, 28, 32), _G_OUTPUTS),
        ("bucket.gru_input.r64.p64", (0, 6, 7, 63), _G_OUTPUTS),
        ("bucket.gru_input.r65.p128", (0, 2, 60, 64), _G_OUTPUTS),
        ("bucket.gru_input.r129.p256", (0, 80, 109, 128), _G_OUTPUTS),
        ("bucket.gru_input.r257.p320", (0, 42, 92, 256), _G_OUTPUTS),
        ("bucket.gru_input.r319.p320", (0, 7, 239, 318), _G_OUTPUTS),
        ("bucket.gru_input.r320.p320", (0, 207, 307, 319), _G_OUTPUTS),
        ("learner.actor.r896", (0, 308, 493, 895), (0, 1, 17, 19)),
        (
            "learner.actor.r1920",
            (0, 1095, 1136, 1919),
            (0, 6, 15, 19),
        ),
        (
            "learner.actor.r2048",
            (0, 1298, 1756, 2047),
            (0, 6, 18, 19),
        ),
    ),
    "cancellation": (
        ("cancellation.encoder_r4", (0, 1, 2, 3), (0, 54, 102, 255)),
        ("cancellation.gru_input_r4", (0, 1, 2, 3), _G_OUTPUTS),
        ("cancellation.gru_hidden_r4", (0, 1, 2, 3), _G_OUTPUTS),
        (
            "cancellation.film_scale_r4",
            (0, 1, 2, 3),
            (0, 46, 78, 255),
        ),
        (
            "cancellation.film_bias_r4",
            (0, 1, 2, 3),
            (0, 127, 142, 255),
        ),
        ("cancellation.actor_r4", (0, 1, 2, 3), (0, 6, 17, 19)),
        ("cancellation.value_r4", (0, 1, 2, 3), (0,)),
    ),
    "underflow_u1": (
        ("underflow_u1.encoder_r1", (0,), (0, 165, 207, 255)),
        ("underflow_u1.gru_input_r1", (0,), _G_OUTPUTS),
        ("underflow_u1.gru_hidden_r1", (0,), _G_OUTPUTS),
        ("underflow_u1.film_scale_r1", (0,), (0, 133, 167, 255)),
        ("underflow_u1.film_bias_r1", (0,), (0, 16, 36, 255)),
        ("underflow_u1.actor_r1", (0,), (0, 12, 13, 19)),
        ("underflow_u1.value_r1", (0,), (0,)),
    ),
    "underflow_u2": (
        ("underflow_u2.encoder_r1", (0,), (0, 11, 164, 255)),
        ("underflow_u2.gru_input_r1", (0,), _G_OUTPUTS),
        ("underflow_u2.gru_hidden_r1", (0,), _G_OUTPUTS),
        ("underflow_u2.film_scale_r1", (0,), (0, 46, 251, 255)),
        ("underflow_u2.film_bias_r1", (0,), (0, 21, 175, 255)),
        ("underflow_u2.actor_r1", (0,), (0, 17, 18, 19)),
        ("underflow_u2.value_r1", (0,), (0,)),
    ),
    "gradient": (
        ("gradient.encoder_r5", (0, 2, 3, 4), (0, 116, 138, 255)),
        ("gradient.gru_input_r5", (0, 2, 3, 4), _G_OUTPUTS),
        ("gradient.gru_hidden_r5", (0, 1, 2, 4), _G_OUTPUTS),
        ("gradient.film_scale_r5", (0, 2, 3, 4), (0, 155, 240, 255)),
        ("gradient.film_bias_r5", (0, 1, 3, 4), (0, 153, 159, 255)),
        ("gradient.actor_r5", (0, 1, 3, 4), (0, 7, 17, 19)),
        ("gradient.value_r5", (0, 1, 3, 4), (0,)),
    ),
}


def sealed_accuracy_seed_contract() -> AccuracySeedContract:
    """Return the source-frozen seed contract for a non-test child process."""

    return AccuracySeedContract(
        ordinary=_SEALED_ORDINARY_SEED,
        holdout=_SEALED_HOLDOUT_SEED,
        cancellation_underflow=_SEALED_CANCELLATION_UNDERFLOW_SEED,
        gradient=_SEALED_GRADIENT_SEED,
        synthetic_test_contract=False,
    )


def frozen_real_selection_manifest() -> dict[str, object]:
    """Return the literal source/doc selection manifest without any RNG call."""

    manifest: dict[str, object] = {
        "schema_version": (
            RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION
        ),
        "groups": {
            group: [
                {
                    "case_id": case_id,
                    "selected_rows": list(rows),
                    "selected_outputs": list(outputs),
                }
                for case_id, rows, outputs in entries
            ]
            for group, entries in _FROZEN_REAL_SELECTION_MANIFEST.items()
        },
    }
    manifest["exact_digest"] = stable_payload_digest(manifest)
    if (
        manifest["exact_digest"]
        != RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "source-frozen selection manifest digest drifted"
        )
    return manifest


def frozen_real_selection_manifest_document_json() -> str:
    """Canonical one-line JSON for exact embedding in the method document."""

    return json.dumps(
        frozen_real_selection_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_method_document_selection_manifest_binding(
    raw_document: bytes,
) -> dict[str, str]:
    """Parse the one literal source-manifest binding from the method document."""

    if not isinstance(raw_document, bytes) or not raw_document:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy method document bytes are malformed"
        )
    try:
        text = raw_document.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy method document is not UTF-8"
        ) from exc
    matches = re.findall(
        (
            r"The machine-readable canonical selection manifest is\s+"
            r"`([a-z0-9_]+)`.*?"
            r"Its canonical `stable_payload_digest` is\s+"
            r"`([0-9a-f]{64})`\."
        ),
        text,
        flags=re.DOTALL,
    )
    expected = (
        RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION,
        RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST,
    )
    if matches != [expected]:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy method document selection-manifest binding is invalid"
        )
    # Recompute the source constant as a separate self-consistency check.  The
    # literal document match alone must not bless a drifted source manifest.
    manifest = frozen_real_selection_manifest()
    if (
        manifest["schema_version"] != expected[0]
        or manifest["exact_digest"] != expected[1]
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy source and method selection manifests differ"
        )
    return {
        "schema_version": expected[0],
        "exact_digest": expected[1],
    }


def _validate_cases_against_frozen_manifest(
    group: str,
    cases: Sequence[LinearProbeCase],
) -> None:
    expected = _FROZEN_REAL_SELECTION_MANIFEST[group]
    observed = tuple(
        (case.case_id, case.selected_rows, case.selected_outputs)
        for case in cases
    )
    if observed != expected:
        raise RecurrentKernelDevelopmentScreenError(
            f"sealed {group} selection differs from the frozen manifest"
        )


def _validated_seed_contract(
    seeds: AccuracySeedContract,
) -> AccuracySeedContract:
    if not isinstance(seeds, AccuracySeedContract):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy seed contract must use AccuracySeedContract"
        )
    if type(seeds.synthetic_test_contract) is not bool:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy synthetic-test marker must be boolean"
        )
    values = (
        seeds.ordinary,
        seeds.holdout,
        seeds.cancellation_underflow,
        seeds.gradient,
    )
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _UINT64_MASK
        for seed in values
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy seeds must be unsigned 64-bit integers"
        )
    if len(set(values)) != 4:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy seeds must be pairwise distinct"
        )
    if seeds.synthetic_test_contract:
        return seeds
    elif seeds != sealed_accuracy_seed_contract():
        raise RecurrentKernelDevelopmentScreenError(
            "non-test accuracy seed contract does not match sealed source"
        )
    return seeds


def _seed_digest(seed: int, purpose: str) -> str:
    return hashlib.sha256(
        f"{_CORPUS_DOMAIN}\0{purpose}\0{seed}".encode()
    ).hexdigest()


def _linux_proc_stat_start_ticks(payload: str) -> str:
    """Parse field 22 after the final parenthesized Linux comm value."""

    if not isinstance(payload, str) or not payload:
        raise RecurrentKernelDevelopmentScreenError(
            "Linux process start identity is malformed"
        )
    _prefix, separator, suffix = payload.rpartition(")")
    fields = suffix.strip().split() if separator else []
    if len(fields) <= 19 or not fields[19].isdigit():
        raise RecurrentKernelDevelopmentScreenError(
            "Linux process start identity is malformed"
        )
    return fields[19]


def process_start_identity(process_id: int) -> str:
    """Return a live OS-bound process-start identity without extra packages."""

    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "process identity PID must be positive"
    )
    proc_stat = Path(f"/proc/{process_id}/stat")
    if proc_stat.is_file():
        start_ticks = _linux_proc_stat_start_ticks(
            proc_stat.read_text(encoding="utf-8")
        )
        return f"linux_proc_stat_start_ticks:{start_ticks}"
    try:
        completed = subprocess.run(
            ("ps", "-o", "lstart=", "-p", str(process_id)),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "process start identity is unavailable"
        ) from exc
    observed = completed.stdout.strip()
    if not observed or "\n" in observed or completed.stderr.strip():
        raise RecurrentKernelDevelopmentScreenError(
            "process start identity is unavailable"
        )
    return f"ps_lstart:{observed}"


def _validate_source_files(
    *,
    checkout_root: Path,
    required_source_paths: Sequence[str],
    source_module_digests: Mapping[str, str],
) -> None:
    if not checkout_root.is_absolute() or not checkout_root.is_dir():
        raise RecurrentKernelDevelopmentScreenError(
            "holdout checkout root must be an absolute directory"
        )
    required = tuple(required_source_paths)
    if (
        required != RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS
        or set(required) != set(source_module_digests)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy required source paths differ from the frozen order"
        )
    resolved_root = checkout_root.resolve()
    for relative_path in required:
        candidate = checkout_root / relative_path
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or candidate.is_symlink()
            or not candidate.is_file()
            or not candidate.resolve().is_relative_to(resolved_root)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout source path is unsafe or missing"
            )
        if source_file_sha256(candidate) != source_module_digests[relative_path]:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout source-module byte digest drifted"
            )


def _validate_method_document_file(
    *,
    checkout_root: Path,
    method_document_path: str,
    method_document_digest: str,
) -> dict[str, str]:
    if method_document_path != RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy method-document path differs from the frozen path"
        )
    expected_digest = _validated_hex(
        method_document_digest,
        length=64,
        field="method-document digest",
    )
    resolved_root = checkout_root.resolve()
    candidate = checkout_root / method_document_path
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not candidate.resolve().is_relative_to(resolved_root)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy method document is unsafe or missing"
        )
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy method-document byte digest drifted"
        )
    return validate_method_document_selection_manifest_binding(raw)


def _validated_positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} must be a positive integer"
        )
    return value


def _validated_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} must be a non-empty string"
        )
    return value


def _validated_absolute_path(value: object, *, field: str) -> str:
    path = Path(_validated_nonempty_string(value, field=field))
    if not path.is_absolute():
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} must be an absolute path"
        )
    return str(path)


def source_retirement_marker(
    *,
    source_sha: str,
    source_module_digest: str,
    method_document_digest: str,
    preregistration_digest: str,
    report_path: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    capability_commitment_sha256: str,
) -> dict[str, object]:
    """Build the exact attempt marker persisted before the first child."""

    report_path = _validated_absolute_path(
        report_path,
        field="retirement report path",
    )
    parent_process_id = _validated_positive_integer(
        parent_process_id,
        field="retirement parent process ID",
    )
    parent_process_start_identity = _validated_nonempty_string(
        parent_process_start_identity,
        field="retirement parent process start identity",
    )
    marker: dict[str, object] = {
        "schema_version": "recurrent_accuracy_source_retirement_v1",
        "source_sha": _validated_hex(
            source_sha,
            length=40,
            field="retirement source SHA",
        ),
        "source_module_digest": _validated_hex(
            source_module_digest,
            length=64,
            field="retirement source-module digest",
        ),
        "method_document_digest": _validated_hex(
            method_document_digest,
            length=64,
            field="retirement method-document digest",
        ),
        "preregistration_digest": _validated_hex(
            preregistration_digest,
            length=64,
            field="retirement preregistration digest",
        ),
        "report_path": report_path,
        "parent_process_id": parent_process_id,
        "parent_process_start_identity": parent_process_start_identity,
        "capability_commitment_sha256": _validated_hex(
            capability_commitment_sha256,
            length=64,
            field="retirement capability commitment",
        ),
        "source_retired": True,
        "no_retry_for_exact_source": True,
    }
    marker["exact_digest"] = stable_payload_digest(marker)
    return marker


def holdout_retirement_marker(
    *,
    source_retirement_marker_sha256: str,
    source_retirement_marker_exact_digest: str,
    preholdout_evidence_digest: str,
    closing_baseline_evidence_digest: str,
    parent_process_id: int,
    parent_process_start_identity: str,
    child_process_id: int,
    child_process_start_identity: str,
    child_nonce: str,
    capability_commitment_sha256: str,
) -> dict[str, object]:
    """Build the admission marker persisted immediately before token send."""

    parent_process_id = _validated_positive_integer(
        parent_process_id,
        field="admission parent process ID",
    )
    child_process_id = _validated_positive_integer(
        child_process_id,
        field="admission child process ID",
    )
    parent_process_start_identity = _validated_nonempty_string(
        parent_process_start_identity,
        field="admission parent process start identity",
    )
    child_process_start_identity = _validated_nonempty_string(
        child_process_start_identity,
        field="admission child process start identity",
    )
    marker: dict[str, object] = {
        "schema_version": "recurrent_accuracy_holdout_retirement_v1",
        "source_retirement_marker_sha256": _validated_hex(
            source_retirement_marker_sha256,
            length=64,
            field="admission attempt-marker SHA256",
        ),
        "source_retirement_marker_exact_digest": _validated_hex(
            source_retirement_marker_exact_digest,
            length=64,
            field="admission attempt-marker exact digest",
        ),
        "preholdout_evidence_digest": _validated_hex(
            preholdout_evidence_digest,
            length=64,
            field="admission pre-holdout digest",
        ),
        "closing_baseline_evidence_digest": _validated_hex(
            closing_baseline_evidence_digest,
            length=64,
            field="admission closing-baseline digest",
        ),
        "parent_process_id": parent_process_id,
        "parent_process_start_identity": parent_process_start_identity,
        "child_process_id": child_process_id,
        "child_process_start_identity": child_process_start_identity,
        "child_nonce": _validated_hex(
            child_nonce,
            length=64,
            field="admission child nonce",
        ),
        "capability_commitment_sha256": _validated_hex(
            capability_commitment_sha256,
            length=64,
            field="admission capability commitment",
        ),
        "holdout_retired": True,
        "no_retry_for_exact_source": True,
    }
    marker["exact_digest"] = stable_payload_digest(marker)
    return marker


def _validate_retirement_marker_file(
    *,
    path: Path,
    file_sha256: str,
    expected_marker: Mapping[str, object],
    label: str,
) -> None:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RecurrentKernelDevelopmentScreenError(
            f"{label} retirement marker is missing or unsafe"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise RecurrentKernelDevelopmentScreenError(
            f"{label} retirement marker file digest is invalid"
        )
    try:
        observed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecurrentKernelDevelopmentScreenError(
            f"{label} retirement marker JSON is invalid"
        ) from exc
    expected_raw = json.dumps(
        expected_marker,
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    if raw != expected_raw or observed != expected_marker:
        raise RecurrentKernelDevelopmentScreenError(
            f"{label} retirement marker reconstruction failed"
        )


def _sha256_rank(
    *,
    seed: int,
    case_id: str,
    axis: str,
    index: int,
) -> bytes:
    return hashlib.sha256(
        (
            f"{_CORPUS_DOMAIN}\0{seed}\0{case_id}\0{axis}\0{index}"
        ).encode()
    ).digest()


def _selected_indices(
    *,
    size: int,
    count: int,
    seed: int,
    case_id: str,
    axis: str,
    pinned: Sequence[int],
) -> tuple[int, ...]:
    if (
        isinstance(size, bool)
        or isinstance(count, bool)
        or not isinstance(size, int)
        or not isinstance(count, int)
        or size <= 0
        or not 0 < count <= size
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy index selection shape is malformed"
        )
    selected = set()
    for index in pinned:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < size
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "accuracy pinned index is malformed"
            )
        selected.add(index)
    if len(selected) > count:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy pinned indices exceed selection count"
        )
    remaining = sorted(
        (index for index in range(size) if index not in selected),
        key=lambda index: (
            _sha256_rank(
                seed=seed,
                case_id=case_id,
                axis=axis,
                index=index,
            ),
            index,
        ),
    )
    selected.update(remaining[: count - len(selected)])
    result = tuple(sorted(selected))
    if len(result) != count:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy index selection did not reach exact count"
        )
    return result


def _row_indices(
    *,
    rows: int,
    count: int,
    seed: int,
    case_id: str,
) -> tuple[int, ...]:
    pins = (0,) if rows == 1 else (0, rows - 1)
    return _selected_indices(
        size=rows,
        count=count,
        seed=seed,
        case_id=case_id,
        axis="row",
        pinned=pins,
    )


def _output_indices(
    *,
    outputs: int,
    count: int,
    seed: int,
    case_id: str,
) -> tuple[int, ...]:
    if outputs == 768:
        pins = (0, 255, 256, 511, 512, 767)
    elif outputs == 256:
        pins = (0, 255)
    elif outputs == 20:
        pins = (0, 19)
    elif outputs == 1:
        pins = (0,)
    else:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy output width has no frozen selection rule"
        )
    return _selected_indices(
        size=outputs,
        count=count,
        seed=seed,
        case_id=case_id,
        axis="output",
        pinned=pins,
    )


def _make_probe(
    *,
    case_id: str,
    table: str,
    role: LinearRole,
    active_rows: int,
    physical_rows: int,
    selected_row_count: int,
    selected_output_count: int,
    seed: int,
) -> LinearProbeCase:
    return LinearProbeCase(
        case_id=case_id,
        table=table,
        role=role.role,
        input_features=role.input_features,
        output_features=role.output_features,
        active_rows=active_rows,
        physical_rows=physical_rows,
        selected_rows=_row_indices(
            rows=active_rows,
            count=selected_row_count,
            seed=seed,
            case_id=case_id,
        ),
        selected_outputs=_output_indices(
            outputs=role.output_features,
            count=selected_output_count,
            seed=seed,
            case_id=case_id,
        ),
    )


def operational_probe_cases(seed: int) -> tuple[LinearProbeCase, ...]:
    """Build the factorized affine, bucket, and learner case tables."""

    cases: list[LinearProbeCase] = []
    for role in _LINEAR_ROLES:
        rows = _AFFINE_ROWS[role.role]
        cases.append(
            _make_probe(
                case_id=f"affine.{role.role}_r{rows}",
                table="T_affine",
                role=role,
                active_rows=rows,
                physical_rows=rows,
                selected_row_count=_AFFINE_ROW_SELECTIONS[role.role],
                selected_output_count=_AFFINE_OUTPUT_SELECTIONS[role.role],
                seed=seed,
            )
        )
    gru_input = _ROLE_BY_NAME["gru_input"]
    for active_rows, physical_rows in zip(
        _BUCKET_ACTIVE_ROWS,
        _BUCKET_PHYSICAL_ROWS,
        strict=True,
    ):
        cases.append(
            _make_probe(
                case_id=(
                    f"bucket.gru_input.r{active_rows}.p{physical_rows}"
                ),
                table="T_bucket",
                role=gru_input,
                active_rows=active_rows,
                physical_rows=physical_rows,
                selected_row_count=min(active_rows, 4),
                selected_output_count=6,
                seed=seed,
            )
        )
    actor = _ROLE_BY_NAME["actor"]
    for rows in _LEARNER_ROWS:
        cases.append(
            _make_probe(
                case_id=f"learner.actor.r{rows}",
                table="T_learner",
                role=actor,
                active_rows=rows,
                physical_rows=rows,
                selected_row_count=4,
                selected_output_count=4,
                seed=seed,
            )
        )
    _validate_case_counts(
        cases,
        expected_dots=_EXPECTED_OPERATIONAL_DOT_COUNT,
        expected_products=_EXPECTED_OPERATIONAL_PRODUCT_COUNT,
        context="operational",
    )
    return tuple(cases)


def cancellation_probe_cases(seed: int) -> tuple[LinearProbeCase, ...]:
    cases = tuple(
            _make_probe(
                case_id=f"cancellation.{role.role}_r4",
                table="T_cancellation",
            role=role,
            active_rows=4,
            physical_rows=4,
            selected_row_count=4,
            selected_output_count=_CANCELLATION_OUTPUT_SELECTIONS[role.role],
            seed=seed,
            )
        for role in _LINEAR_ROLES
    )
    _validate_case_counts(
        cases,
        expected_dots=_EXPECTED_CANCELLATION_DOT_COUNT,
        expected_products=_EXPECTED_CANCELLATION_PRODUCT_COUNT,
        context="cancellation",
    )
    return cases


def underflow_probe_cases(
    seed: int,
    *,
    family: str,
) -> tuple[LinearProbeCase, ...]:
    if family not in {"underflow_u1", "underflow_u2"}:
        raise RecurrentKernelDevelopmentScreenError(
            "underflow probe family is unknown"
        )
    cases = tuple(
        _make_probe(
            case_id=f"{family}.{role.role}_r1",
            table="T_underflow",
            role=role,
            active_rows=1,
            physical_rows=1,
            selected_row_count=1,
            selected_output_count=_CANCELLATION_OUTPUT_SELECTIONS[role.role],
            seed=seed,
        )
        for role in _LINEAR_ROLES
    )
    _validate_case_counts(
        cases,
        expected_dots=_EXPECTED_UNDERFLOW_DOT_COUNT,
        expected_products=_EXPECTED_UNDERFLOW_PRODUCT_COUNT,
        context="underflow",
    )
    return cases


def gradient_probe_cases(seed: int) -> tuple[LinearProbeCase, ...]:
    cases = tuple(
            _make_probe(
                case_id=f"gradient.{role.role}_r5",
                table="T_gradient",
            role=role,
            active_rows=5,
            physical_rows=5,
            selected_row_count=4,
            selected_output_count=_CANCELLATION_OUTPUT_SELECTIONS[role.role],
            seed=seed,
        )
        for role in _LINEAR_ROLES
    )
    if (
        sum(case.dot_count for case in cases)
        != _EXPECTED_GRADIENT_DIAGNOSTIC_SPOT_COUNT
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "gradient diagnostic spot-selection count drifted"
        )
    component_count = sum(
        case.physical_rows * case.input_features
        + case.output_features * case.input_features
        + case.output_features
        for case in cases
    )
    if component_count != _EXPECTED_GRADIENT_COMPONENT_COUNT:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient component count drifted"
        )
    return cases


def _validate_case_counts(
    cases: Sequence[LinearProbeCase],
    *,
    expected_dots: int,
    expected_products: int,
    context: str,
) -> None:
    dots = sum(case.dot_count for case in cases)
    products = sum(case.product_count for case in cases)
    if dots != expected_dots or products != expected_products:
        raise RecurrentKernelDevelopmentScreenError(
            f"{context} probe counts drifted"
        )


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_INCREMENT) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _value_word(
    *,
    seed: int,
    family: str,
    case_id: str,
    tensor_name: str,
    flat_index: int,
    draw: int,
) -> int:
    if (
        isinstance(flat_index, bool)
        or not isinstance(flat_index, int)
        or flat_index < 0
        or isinstance(draw, bool)
        or not isinstance(draw, int)
        or not 0 <= draw <= 3
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy value draw coordinates are malformed"
        )
    encoded = "\0".join(
        (_VALUE_DOMAIN, family, case_id, tensor_name)
    )
    domain_word = int.from_bytes(
        hashlib.sha256(encoded.encode()).digest()[:8],
        "big",
    )
    state = (
        seed
        + domain_word
        + _SPLITMIX_INCREMENT * (1 + 4 * flat_index + draw)
    ) & _UINT64_MASK
    return _splitmix64(state)


def _sign_from_word(word: int) -> float:
    return -1.0 if word & 1 else 1.0


def _normal_value(
    *,
    seed: int,
    case_id: str,
    tensor: str,
    index: tuple[int, ...],
    flat_index: int,
    family: str,
) -> float:
    words = tuple(
        _value_word(
            seed=seed,
            family=family,
            case_id=case_id,
            tensor_name=tensor,
            flat_index=flat_index,
            draw=draw,
        )
        for draw in range(4)
    )
    component_family = family
    if family == "composite":
        component_family = ("balanced", "alternating", "mixed")[
            words[3] % 3
        ]
    if component_family not in {"balanced", "alternating", "mixed"}:
        raise RecurrentKernelDevelopmentScreenError(
            "normal accuracy family is unknown"
        )
    feature_index = index[-1] if tensor != "bias" else 0
    output_index = index[0] if tensor in {"weight", "bias"} else 0
    if component_family == "alternating":
        feature_sign = -1.0 if feature_index % 2 else 1.0
        output_sign = -1.0 if output_index % 2 else 1.0
        if tensor == "input":
            sign = feature_sign
        elif tensor in {"weight", "bias"}:
            sign = feature_sign * output_sign if tensor == "weight" else output_sign
        else:
            raise RecurrentKernelDevelopmentScreenError(
                "normal tensor kind is unknown"
            )
    else:
        sign = _sign_from_word(words[1])
    if component_family in {"balanced", "alternating"}:
        magnitude = (
            2.0**-14
            if tensor == "bias"
            else (1 + (words[0] & 1023)) * 2.0**-12
        )
    else:
        mantissa = 1024 + (words[0] & 1023)
        exponents = (-14, -10, -6) if tensor == "bias" else (-12, -8, -4)
        exponent = exponents[words[2] % 3]
        magnitude = mantissa * 2.0 ** (exponent - 10)
    return sign * magnitude


def _tensor_from_values(
    shape: tuple[int, ...],
    value: Callable[[tuple[int, ...]], float],
) -> Tensor:
    if len(shape) == 1:
        values = [value((index,)) for index in range(shape[0])]
    elif len(shape) == 2:
        values = [
            value((row, column))
            for row in range(shape[0])
            for column in range(shape[1])
        ]
    else:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy tensor shape rank is unsupported"
        )
    tensor = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(
        shape
    )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy tensor generator produced non-finite values"
        )
    return tensor


def generate_normal_case(
    probe: LinearProbeCase,
    *,
    seed: int,
    family: str,
    _holdout_permit: _HoldoutGenerationPermit | None = None,
) -> GeneratedLinearCase:
    if (
        family == "composite"
        and (
            not isinstance(_holdout_permit, _HoldoutGenerationPermit)
            or _holdout_permit.seed != seed
            or not _holdout_permit.token
        )
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "sealed holdout generation requires a consumed private permit"
        )
    if family == "composite":
        registered = _HOLDOUT_GENERATION_REGISTRY.get(
            _holdout_permit.token
        )
        if (
            registered is None
            or registered[0] != seed
            or probe.case_id not in registered[1]
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "sealed holdout generation permit is invalid or reused"
            )
        registered[1].remove(probe.case_id)
        if not registered[1]:
            del _HOLDOUT_GENERATION_REGISTRY[_holdout_permit.token]
    return _materialize_normal_case(
        probe,
        seed=seed,
        family=family,
    )


def _materialize_normal_case(
    probe: LinearProbeCase,
    *,
    seed: int,
    family: str,
) -> GeneratedLinearCase:
    """Reconstruct one formula-bound case after its caller proves authority."""

    inputs = _tensor_from_values(
        (probe.physical_rows, probe.input_features),
        lambda index: _normal_value(
            seed=seed,
            case_id=probe.case_id,
            tensor="input",
            index=index,
            flat_index=index[0] * probe.input_features + index[1],
            family=family,
        ),
    )
    # Padded physical bucket rows are explicit zeros, as in the runtime.
    if probe.physical_rows > probe.active_rows:
        inputs[probe.active_rows :] = 0.0
    weight = _tensor_from_values(
        (probe.output_features, probe.input_features),
        lambda index: _normal_value(
            seed=seed,
            case_id=probe.case_id,
            tensor="weight",
            index=index,
            flat_index=index[0] * probe.input_features + index[1],
            family=family,
        ),
    )
    bias = _tensor_from_values(
        (probe.output_features,),
        lambda index: _normal_value(
            seed=seed,
            case_id=probe.case_id,
            tensor="bias",
            index=index,
            flat_index=index[0],
            family=family,
        ),
    )
    return GeneratedLinearCase(probe, family, inputs, weight, bias)


def generate_cancellation_case(
    probe: LinearProbeCase,
    *,
    seed: int,
) -> GeneratedLinearCase:
    if probe.input_features % 2:
        raise RecurrentKernelDevelopmentScreenError(
            "cancellation probe width must be even"
        )
    half = probe.input_features // 2

    def mirrored_magnitude(
        tensor: str,
        row_or_output: int,
        feature: int,
    ) -> float:
        flat_index = row_or_output * half + feature % half
        word = _value_word(
            seed=seed,
            family="cancellation",
            case_id=probe.case_id,
            tensor_name=tensor,
            flat_index=flat_index,
            draw=0,
        )
        return (1024 + (word & 1023)) * 2.0**-11

    inputs = _tensor_from_values(
        (probe.physical_rows, probe.input_features),
        lambda index: mirrored_magnitude("input", index[0], index[1]),
    )
    weight = _tensor_from_values(
        (probe.output_features, probe.input_features),
        lambda index: (
            1.0 if index[1] < half else -1.0
        )
        * mirrored_magnitude("weight", index[0], index[1]),
    )
    bias = _tensor_from_values(
        (probe.output_features,),
        lambda index: _sign_from_word(
            _value_word(
                seed=seed,
                family="cancellation",
                case_id=probe.case_id,
                tensor_name="bias",
                flat_index=index[0],
                draw=1,
            )
        )
        * (
            1
            + (
                _value_word(
                    seed=seed,
                    family="cancellation",
                    case_id=probe.case_id,
                    tensor_name="bias",
                    flat_index=index[0],
                    draw=0,
                )
                & 7
            )
        )
        * 2.0**-22,
    )
    selected_inputs = inputs[list(probe.selected_rows)]
    selected_weight = weight[list(probe.selected_outputs)]
    selected_bias = bias[list(probe.selected_outputs)]
    for row in selected_inputs:
        for output_index, output_weight in enumerate(selected_weight):
            terms = [
                float(row[column]) * float(output_weight[column])
                for column in range(probe.input_features)
            ]
            bias_value = float(selected_bias[output_index])
            absolute_sum = math.fsum(
                (*[abs(term) for term in terms], abs(bias_value))
            )
            exact_sum = math.fsum((*terms, bias_value))
            condition_ratio = absolute_sum / abs(exact_sum)
            if not math.isfinite(condition_ratio) or condition_ratio < 2.0**18:
                raise RecurrentKernelDevelopmentScreenError(
                    "cancellation probe condition ratio is below contract"
                )
    return GeneratedLinearCase(
        probe,
        "cancellation_mirrored_halves",
        inputs,
        weight,
        bias,
    )


def generate_underflow_case(
    probe: LinearProbeCase,
    *,
    seed: int,
    family: str,
) -> GeneratedLinearCase:
    if family == "underflow_u1":
        input_value = weight_value = 2.0**-70
        bias_exponent = -120
    elif family == "underflow_u2":
        input_value = weight_value = 2.0**-75
        bias_exponent = None
    else:
        raise RecurrentKernelDevelopmentScreenError(
            "underflow family is unknown"
        )
    inputs = torch.full(
        (probe.physical_rows, probe.input_features),
        input_value,
        dtype=torch.float32,
    )
    weight = torch.full(
        (probe.output_features, probe.input_features),
        weight_value,
        dtype=torch.float32,
    )
    if bias_exponent is None:
        bias = torch.zeros(probe.output_features, dtype=torch.float32)
    else:
        bias = _tensor_from_values(
            (probe.output_features,),
            lambda index: _sign_from_word(
                _value_word(
                    seed=seed,
                    family=family,
                    case_id=probe.case_id,
                    tensor_name="bias",
                    flat_index=index[0],
                    draw=1,
                )
            )
            * 2.0**bias_exponent,
        )
    return GeneratedLinearCase(probe, family, inputs, weight, bias)


def _selected_candidate_output(
    full_output: Tensor,
    probe: LinearProbeCase,
) -> Tensor:
    if full_output.shape != (probe.physical_rows, probe.output_features):
        raise RecurrentKernelDevelopmentScreenError(
            "candidate accuracy output shape is malformed"
        )
    return full_output[list(probe.selected_rows)][:, list(probe.selected_outputs)]


def _selected_components(
    generated: GeneratedLinearCase,
) -> tuple[Tensor, Tensor, Tensor]:
    probe = generated.probe
    return (
        generated.inputs[list(probe.selected_rows)],
        generated.weight[list(probe.selected_outputs)],
        generated.bias[list(probe.selected_outputs)],
    )


def _validated_oracle_tensor(
    value: Tensor,
    *,
    name: str,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise RecurrentKernelDevelopmentScreenError(
            f"independent linear oracle {name} must be a tensor"
        )
    if value.device.type != "cpu" or value.device.index is not None:
        raise RecurrentKernelDevelopmentScreenError(
            f"independent linear oracle {name} must be on CPU"
        )
    if value.dtype != torch.float32:
        raise RecurrentKernelDevelopmentScreenError(
            f"independent linear oracle {name} must use torch.float32"
        )
    if value.layout != torch.strided:
        raise RecurrentKernelDevelopmentScreenError(
            f"independent linear oracle {name} must use strided layout"
        )
    detached = value.detach().contiguous()
    if not bool(torch.isfinite(detached).all().item()):
        raise RecurrentKernelDevelopmentScreenError(
            f"independent linear oracle {name} must contain finite values"
        )
    return detached


def _float32_tensor_exact_digest(value: Tensor | None) -> str | None:
    """Digest exact binary32 components without a numerical kernel."""

    if value is None:
        return None
    detached = _validated_oracle_tensor(value, name="digest tensor")
    payload = {
        "dtype": "torch.float32",
        "shape": list(detached.shape),
        "component_encoding": "ieee754_binary32_big_endian_hex_v1",
        "binary32_big_endian_hex": [
            struct.pack(">f", float(item)).hex()
            for item in detached.reshape(-1).tolist()
        ],
    }
    return stable_payload_digest(payload)


def _validated_oracle_inputs(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    detached_inputs = _validated_oracle_tensor(inputs, name="inputs")
    detached_weight = _validated_oracle_tensor(weight, name="weight")
    detached_bias = (
        None
        if bias is None
        else _validated_oracle_tensor(bias, name="bias")
    )
    if detached_inputs.ndim != 2 or detached_weight.ndim != 2:
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle inputs and weight must be rank two"
        )
    if detached_inputs.shape[1] != detached_weight.shape[1]:
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle input width does not match weight"
        )
    if detached_bias is not None and (
        detached_bias.ndim != 1
        or detached_bias.shape[0] != detached_weight.shape[0]
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle bias shape is malformed"
        )
    if (
        detached_inputs.numel() == 0
        or detached_weight.numel() == 0
        or (detached_bias is not None and detached_bias.numel() == 0)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle tensors must be non-empty"
        )
    return detached_inputs, detached_weight, detached_bias


def _independent_oracle_raw_values(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> tuple[
    Tensor,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...] | None,
    tuple[float, ...],
]:
    inputs, weight, bias = _validated_oracle_inputs(inputs, weight, bias)
    row_count = int(inputs.shape[0])
    input_width = int(weight.shape[1])
    output_width = int(weight.shape[0])
    output_element_count = row_count * output_width
    tensor_element_count = (
        int(inputs.numel())
        + int(weight.numel())
        + (0 if bias is None else int(bias.numel()))
        + output_element_count
    )
    product_count = row_count * output_width * input_width
    if (
        tensor_element_count
        > _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_TENSOR_ELEMENTS
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle tensor element bound exceeded"
        )
    if product_count > _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_PRODUCTS:
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle product bound exceeded"
        )

    input_values = tuple(float(value) for value in inputs.reshape(-1).tolist())
    weight_values = tuple(
        float(value) for value in weight.reshape(-1).tolist()
    )
    bias_values = (
        None
        if bias is None
        else tuple(float(value) for value in bias.reshape(-1).tolist())
    )
    accumulated_values: list[float] = []
    try:
        for row_index in range(row_count):
            input_offset = row_index * input_width
            for output_index in range(output_width):
                weight_offset = output_index * input_width
                products = [
                    input_values[input_offset + input_index]
                    * weight_values[weight_offset + input_index]
                    for input_index in range(input_width)
                ]
                if bias_values is not None:
                    products.append(bias_values[output_index])
                accumulated_values.append(math.fsum(products))
    except OverflowError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle float64 accumulation overflowed"
        ) from exc

    output = torch.tensor(
        accumulated_values,
        dtype=torch.float32,
        device="cpu",
    ).reshape(row_count, output_width)
    if not bool(torch.isfinite(output).all().item()):
        raise RecurrentKernelDevelopmentScreenError(
            "independent linear oracle float32 output overflowed"
        )
    output_values = tuple(float(value) for value in output.reshape(-1).tolist())
    return (
        output,
        input_values,
        weight_values,
        bias_values,
        output_values,
    )


def _independent_oracle_provenance(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    output: Tensor,
    *,
    input_values: Sequence[float],
    weight_values: Sequence[float],
    bias_values: Sequence[float] | None,
    output_values: Sequence[float],
) -> dict[str, object]:
    row_count = int(inputs.shape[0])
    input_width = int(weight.shape[1])
    output_width = int(weight.shape[0])
    output_element_count = row_count * output_width
    product_count = output_element_count * input_width
    contract = {
        "oracle_version": _INDEPENDENT_LINEAR_ORACLE_VERSION,
        "source_path": _INDEPENDENT_LINEAR_ORACLE_SOURCE_PATH,
        "product_precision": "python_binary64",
        "accumulator": "math.fsum",
        "bias_application": "one_binary64_term_per_output",
        "output_rounding": "one_torch_float32_construction",
        "exact_digest_encoding": "stable_json_binary64_hex_values_v1",
        "component_digest_encoding": (
            "stable_json_ieee754_binary32_big_endian_hex_v1"
        ),
    }
    component_exact_digests = {
        "input": _float32_tensor_exact_digest(inputs),
        "weight": _float32_tensor_exact_digest(weight),
        "bias": _float32_tensor_exact_digest(bias),
        "output": _float32_tensor_exact_digest(output),
    }
    digest_payload = {
        **contract,
        "input_shape": list(inputs.shape),
        "weight_shape": list(weight.shape),
        "bias_shape": None if bias is None else list(bias.shape),
        "output_shape": list(output.shape),
        "component_exact_digests": component_exact_digests,
        "input_values": [value.hex() for value in input_values],
        "weight_values": [value.hex() for value in weight_values],
        "bias_values": (
            None
            if bias_values is None
            else [value.hex() for value in bias_values]
        ),
        "output_values": [value.hex() for value in output_values],
    }
    return {
        **contract,
        "development_only": True,
        "trainable": False,
        "device": "cpu",
        "dtype": "torch.float32",
        "input_shape": list(inputs.shape),
        "weight_shape": list(weight.shape),
        "bias_shape": None if bias is None else list(bias.shape),
        "output_shape": list(output.shape),
        "row_count": row_count,
        "input_feature_count": input_width,
        "output_feature_count": output_width,
        "input_element_count": int(inputs.numel()),
        "weight_element_count": int(weight.numel()),
        "bias_element_count": 0 if bias is None else int(bias.numel()),
        "output_element_count": output_element_count,
        "float64_product_count": product_count,
        "float64_accumulation_count": output_element_count,
        "float64_bias_term_count": (
            0 if bias is None else output_element_count
        ),
        "float32_round_count": output_element_count,
        "maximum_tensor_elements": (
            _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_TENSOR_ELEMENTS
        ),
        "maximum_products": _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_PRODUCTS,
        "component_exact_digests": component_exact_digests,
        "input_exact_digest": component_exact_digests["input"],
        "weight_exact_digest": component_exact_digests["weight"],
        "bias_exact_digest": component_exact_digests["bias"],
        "output_exact_digest": component_exact_digests["output"],
        "exact_digest": stable_payload_digest(digest_payload),
    }


def _independent_float64_linear_oracle(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, dict[str, object]]:
    """Run the bounded Python-binary64 oracle without a Torch LA kernel."""

    inputs, weight, bias = _validated_oracle_inputs(inputs, weight, bias)
    (
        output,
        input_values,
        weight_values,
        bias_values,
        output_values,
    ) = _independent_oracle_raw_values(inputs, weight, bias)
    provenance = _independent_oracle_provenance(
        inputs,
        weight,
        bias,
        output,
        input_values=input_values,
        weight_values=weight_values,
        bias_values=bias_values,
        output_values=output_values,
    )
    return output, provenance


def _ordered_float32_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    magnitude = bits & 0x7FFFFFFF
    if bits & 0x80000000:
        return 0x80000000 - magnitude
    return 0x80000000 + magnitude


def _float32_ulp(value: float) -> float:
    absolute = abs(value)
    if absolute == 0.0:
        return 2.0**-149
    bits = struct.unpack(">I", struct.pack(">f", absolute))[0]
    next_value = struct.unpack(">f", struct.pack(">I", bits + 1))[0]
    spacing = next_value - absolute
    if not math.isfinite(spacing) or spacing <= 0.0:
        previous = struct.unpack(">f", struct.pack(">I", bits - 1))[0]
        spacing = absolute - previous
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy corpus float32 ULP is malformed"
        )
    return spacing


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 < probability <= 1.0:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy nearest-rank request is malformed"
        )
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(probability * len(ordered)) - 1)])


def _independent_oracle_error_metrics(
    candidate_output: Tensor,
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> dict[str, object]:
    """Recompute, authenticate, and compare with the bounded CPU oracle."""

    candidate_output = _validated_oracle_tensor(
        candidate_output,
        name="candidate output",
    )
    inputs, weight, bias = _validated_oracle_inputs(inputs, weight, bias)
    expected_output_shape = (int(inputs.shape[0]), int(weight.shape[0]))
    if tuple(candidate_output.shape) != expected_output_shape:
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics output shape is malformed"
        )
    if candidate_output.numel() == 0:
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics output must be non-empty"
        )

    oracle_output, oracle_provenance = _independent_float64_linear_oracle(
        inputs,
        weight,
        bias,
    )
    (
        authenticated_output,
        input_values,
        weight_values,
        bias_values,
        authenticated_output_values,
    ) = _independent_oracle_raw_values(inputs, weight, bias)
    authenticated_provenance = _independent_oracle_provenance(
        inputs,
        weight,
        bias,
        authenticated_output,
        input_values=input_values,
        weight_values=weight_values,
        bias_values=bias_values,
        output_values=authenticated_output_values,
    )
    if (
        not torch.equal(oracle_output, authenticated_output)
        or oracle_provenance != authenticated_provenance
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics oracle authentication failed"
        )

    row_count = int(inputs.shape[0])
    input_width = int(weight.shape[1])
    output_width = int(weight.shape[0])
    comparison_count = row_count * output_width
    product_count = comparison_count * input_width
    oracle_tensor_element_count = (
        int(inputs.numel())
        + int(weight.numel())
        + (0 if bias is None else int(bias.numel()))
        + comparison_count
    )
    metrics_tensor_element_count = (
        oracle_tensor_element_count + int(candidate_output.numel())
    )
    if (
        oracle_tensor_element_count
        > _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_TENSOR_ELEMENTS
        or metrics_tensor_element_count
        > _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_TENSOR_ELEMENTS
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics tensor element bound exceeded"
        )
    if product_count > _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_PRODUCTS:
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics product bound exceeded"
        )

    k = 2 * input_width + 1
    ku = k * _FLOAT32_UNIT_ROUNDOFF
    if not 0.0 < ku < 1.0:
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics gamma bound is undefined"
        )
    gamma = ku / (1.0 - ku)
    candidate_values = tuple(
        float(value) for value in candidate_output.reshape(-1).tolist()
    )
    oracle_values = tuple(
        float(value) for value in authenticated_output.reshape(-1).tolist()
    )

    absolute_errors: list[float] = []
    allowances: list[float] = []
    normalized_ratios: list[float] = []
    d04_tolerance_ratios: list[float] = []
    ulp_distances: list[int] = []
    applicable_rows: list[bool] = []
    subnormal_exact_term_count = 0
    subnormal_exact_term_comparison_count = 0
    mixed_sign_exact_term_comparison_count = 0
    pre_round_subnormal_comparison_count = 0
    applicable_violation_count = 0
    try:
        for row_index in range(row_count):
            input_offset = row_index * input_width
            for output_index in range(output_width):
                weight_offset = output_index * input_width
                output_offset = row_index * output_width + output_index
                products = [
                    input_values[input_offset + input_index]
                    * weight_values[weight_offset + input_index]
                    for input_index in range(input_width)
                ]
                exact_terms = list(products)
                if bias_values is not None:
                    exact_terms.append(bias_values[output_index])
                if not all(math.isfinite(term) for term in exact_terms):
                    raise RecurrentKernelDevelopmentScreenError(
                        "independent oracle metrics exact term is non-finite"
                    )
                comparison_subnormal_term_count = sum(
                    0.0 < abs(term) < _FLOAT32_SMALLEST_NORMAL
                    for term in exact_terms
                )
                signs = {
                    math.copysign(1.0, term)
                    for term in exact_terms
                    if term != 0.0
                }
                mixed_sign = len(signs) > 1
                pre_round = math.fsum(exact_terms)
                pre_round_normal_or_zero = (
                    pre_round == 0.0
                    or abs(pre_round) >= _FLOAT32_SMALLEST_NORMAL
                )
                applicable = (
                    comparison_subnormal_term_count == 0
                    and not mixed_sign
                    and pre_round_normal_or_zero
                )
                absolute_product_sum = math.fsum(
                    abs(term) for term in exact_terms
                )
                oracle_value = oracle_values[output_offset]
                candidate_value = candidate_values[output_offset]
                allowance = (
                    gamma * absolute_product_sum
                    + 0.5 * _float32_ulp(oracle_value)
                )
                absolute_error = abs(candidate_value - oracle_value)
                normalized_ratio = absolute_error / allowance
                d04_tolerance = (
                    _D04_ABSOLUTE_TOLERANCE
                    + _D04_RELATIVE_TOLERANCE * abs(oracle_value)
                )
                d04_tolerance_ratio = absolute_error / d04_tolerance
                if not all(
                    math.isfinite(value)
                    for value in (
                        absolute_product_sum,
                        allowance,
                        absolute_error,
                        normalized_ratio,
                        d04_tolerance_ratio,
                    )
                ) or allowance <= 0.0:
                    raise RecurrentKernelDevelopmentScreenError(
                        "independent oracle metrics produced a non-finite "
                        "comparison"
                    )
                absolute_errors.append(absolute_error)
                allowances.append(allowance)
                normalized_ratios.append(normalized_ratio)
                d04_tolerance_ratios.append(d04_tolerance_ratio)
                ulp_distances.append(
                    abs(
                        _ordered_float32_bits(candidate_value)
                        - _ordered_float32_bits(oracle_value)
                    )
                )
                applicable_rows.append(applicable)
                applicable_violation_count += int(
                    applicable and normalized_ratio > 1.0
                )
                subnormal_exact_term_count += (
                    comparison_subnormal_term_count
                )
                subnormal_exact_term_comparison_count += int(
                    comparison_subnormal_term_count > 0
                )
                mixed_sign_exact_term_comparison_count += int(mixed_sign)
                pre_round_subnormal_comparison_count += int(
                    not pre_round_normal_or_zero
                )
    except OverflowError as exc:
        raise RecurrentKernelDevelopmentScreenError(
            "independent oracle metrics accumulation overflowed"
        ) from exc

    applicable_count = sum(applicable_rows)
    d04_within_count = sum(
        ratio <= 1.0 for ratio in d04_tolerance_ratios
    )
    evidence: dict[str, object] = {
        "schema_version": _INDEPENDENT_ORACLE_ERROR_METRICS_VERSION,
        "oracle_version": _INDEPENDENT_LINEAR_ORACLE_VERSION,
        "oracle_provenance": authenticated_provenance,
        "forward_error_bound": {
            "unit_roundoff": _FLOAT32_UNIT_ROUNDOFF,
            "dot_product_length": input_width,
            "operation_count": k,
            "gamma": gamma,
            "absolute_product_and_bias_accumulator": "math.fsum_binary64",
            "allowance_formula": (
                "gamma_2n_plus_1_times_absolute_product_and_bias_sum"
                "_plus_half_oracle_float32_ulp"
            ),
            "standard_bound_precondition": (
                "all_exact_product_and_bias_terms_finite_and_every_nonzero_"
                "term_normal_in_binary32_and_all_nonzero_terms_share_one_sign_"
                "and_binary64_math_fsum_pre_round_is_zero_or_float32_normal"
            ),
            "pre_round_normal_or_exact_zero_required": True,
            "summation_order_reason": (
                "same_sign_nonzero_terms_make_every_order_monotone_in_magnitude"
            ),
            "smallest_float32_normal": _FLOAT32_SMALLEST_NORMAL,
            "smallest_float32_subnormal": _FLOAT32_SMALLEST_SUBNORMAL,
        },
        "d04_oracle_tolerance": {
            "relative_tolerance": _D04_RELATIVE_TOLERANCE,
            "absolute_tolerance": _D04_ABSOLUTE_TOLERANCE,
            "ratio_formula": (
                "absolute_error_divided_by_"
                "absolute_tolerance_plus_relative_tolerance_times_abs_oracle"
            ),
        },
        "input_shape": list(inputs.shape),
        "weight_shape": list(weight.shape),
        "bias_shape": None if bias is None else list(bias.shape),
        "output_shape": list(candidate_output.shape),
        "input_exact_digest": _float32_tensor_exact_digest(inputs),
        "weight_exact_digest": _float32_tensor_exact_digest(weight),
        "bias_exact_digest": _float32_tensor_exact_digest(bias),
        "candidate_output_exact_digest": _float32_tensor_exact_digest(
            candidate_output
        ),
        "oracle_output_exact_digest": _float32_tensor_exact_digest(
            authenticated_output
        ),
        "comparison_count": comparison_count,
        "finite_comparison_count": comparison_count,
        "oracle_tensor_element_count": oracle_tensor_element_count,
        "metrics_tensor_element_count": metrics_tensor_element_count,
        "maximum_tensor_elements": (
            _INDEPENDENT_LINEAR_ORACLE_MAXIMUM_TENSOR_ELEMENTS
        ),
        "float64_product_count": product_count,
        "standard_bound_applicable": applicable_count == comparison_count,
        "standard_bound_applicable_comparison_count": applicable_count,
        "standard_bound_inapplicable_comparison_count": (
            comparison_count - applicable_count
        ),
        "subnormal_exact_term_count": subnormal_exact_term_count,
        "subnormal_exact_term_comparison_count": (
            subnormal_exact_term_comparison_count
        ),
        "mixed_sign_exact_term_comparison_count": (
            mixed_sign_exact_term_comparison_count
        ),
        "pre_round_subnormal_comparison_count": (
            pre_round_subnormal_comparison_count
        ),
        "within_forward_error_bound_count": (
            applicable_count - applicable_violation_count
        ),
        "forward_error_bound_violation_count": applicable_violation_count,
        "maximum_absolute_error": max(absolute_errors),
        "p999_absolute_error": _nearest_rank(absolute_errors, 0.999),
        "rms_absolute_error": math.sqrt(
            math.fsum(error * error for error in absolute_errors)
            / comparison_count
        ),
        "minimum_forward_error_allowance": min(allowances),
        "maximum_forward_error_allowance": max(allowances),
        "maximum_normalized_forward_error_ratio": max(normalized_ratios),
        "p999_normalized_forward_error_ratio": _nearest_rank(
            normalized_ratios,
            0.999,
        ),
        "maximum_float32_ulp_distance": max(ulp_distances),
        "d04_oracle_tolerance_within_count": d04_within_count,
        "d04_oracle_tolerance_violation_count": (
            comparison_count - d04_within_count
        ),
        "maximum_d04_oracle_tolerance_ratio": max(d04_tolerance_ratios),
        "p999_d04_oracle_tolerance_ratio": _nearest_rank(
            d04_tolerance_ratios,
            0.999,
        ),
        "d04_oracle_tolerance_passed": (
            d04_within_count == comparison_count
        ),
        "standard_forward_error_bound_passed": (
            applicable_violation_count == 0
        ),
    }
    evidence["exact_digest"] = stable_payload_digest(evidence)
    return evidence


def _per_dot_classification(
    *,
    probe: LinearProbeCase,
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
    selected_candidate_output: Tensor,
) -> dict[str, object]:
    oracle, _ = _independent_float64_linear_oracle(inputs, weight, bias)
    candidate_values = selected_candidate_output.reshape(-1).tolist()
    oracle_values = oracle.reshape(-1).tolist()
    input_values = inputs.reshape(-1).tolist()
    weight_values = weight.reshape(-1).tolist()
    bias_values = bias.reshape(-1).tolist()
    input_width = probe.input_features
    output_count = len(probe.selected_outputs)
    k = 2 * input_width + 1
    ku = k * 2.0**-24
    gamma = ku / (1.0 - ku)
    records: list[dict[str, object]] = []
    applicable_violation_count = 0
    mixed_sign_count = 0
    subnormal_term_dot_count = 0
    candidate_zero_result_count = 0
    oracle_zero_result_count = 0
    for selected_row_offset, original_row in enumerate(
        probe.selected_rows
    ):
        input_offset = selected_row_offset * input_width
        for selected_output_offset, original_output in enumerate(
            probe.selected_outputs
        ):
            weight_offset = selected_output_offset * input_width
            output_offset = (
                selected_row_offset * output_count + selected_output_offset
            )
            products = [
                float(input_values[input_offset + feature])
                * float(weight_values[weight_offset + feature])
                for feature in range(input_width)
            ]
            bias_value = float(bias_values[selected_output_offset])
            exact_terms = (*products, bias_value)
            all_terms_finite = all(math.isfinite(term) for term in exact_terms)
            if not all_terms_finite:
                raise RecurrentKernelDevelopmentScreenError(
                    "accuracy per-dot exact term is non-finite"
                )
            nonzero_terms = tuple(term for term in exact_terms if term != 0.0)
            signs = {math.copysign(1.0, term) for term in nonzero_terms}
            mixed_sign = len(signs) > 1
            all_nonzero_terms_normal = all(
                abs(term) >= _FLOAT32_SMALLEST_NORMAL
                for term in nonzero_terms
            )
            pre_round = math.fsum(exact_terms)
            pre_round_normal_or_zero = (
                pre_round == 0.0
                or abs(pre_round) >= _FLOAT32_SMALLEST_NORMAL
            )
            applicable = (
                all_terms_finite
                and all_nonzero_terms_normal
                and not mixed_sign
                and pre_round_normal_or_zero
            )
            absolute_sum = math.fsum(abs(term) for term in exact_terms)
            oracle_value = float(oracle_values[output_offset])
            candidate_value = float(candidate_values[output_offset])
            absolute_error = abs(candidate_value - oracle_value)
            allowance = (
                gamma * absolute_sum + 0.5 * _float32_ulp(oracle_value)
            )
            standard_ratio = absolute_error / allowance
            d04_allowance = (
                _D04_ABSOLUTE_TOLERANCE
                + _D04_RELATIVE_TOLERANCE * abs(oracle_value)
            )
            d04_ratio = absolute_error / d04_allowance
            condition_ratio = (
                None
                if pre_round == 0.0
                else absolute_sum / abs(pre_round)
            )
            condition_ratio_kind = (
                "exact_zero" if condition_ratio is None else "finite"
            )
            if (
                not math.isfinite(oracle_value)
                or not math.isfinite(candidate_value)
                or not math.isfinite(absolute_error)
                or not math.isfinite(allowance)
                or allowance <= 0.0
                or not math.isfinite(standard_ratio)
                or not math.isfinite(d04_ratio)
                or (
                    condition_ratio is not None
                    and not math.isfinite(condition_ratio)
                )
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    "accuracy per-dot metric is non-finite"
                )
            applicable_violation_count += int(
                applicable and standard_ratio > 1.0
            )
            mixed_sign_count += int(mixed_sign)
            subnormal_term_dot_count += int(not all_nonzero_terms_normal)
            candidate_zero_result_count += int(candidate_value == 0.0)
            oracle_zero_result_count += int(oracle_value == 0.0)
            records.append(
                {
                    "selected_row_offset": selected_row_offset,
                    "selected_output_offset": selected_output_offset,
                    "original_row": original_row,
                    "original_output": original_output,
                    "term_class": (
                        "underflow_contaminated"
                        if not all_nonzero_terms_normal
                        else "mixed_sign"
                        if mixed_sign
                        else "same_sign_normal"
                    ),
                    "standard_forward_error_bound_applicable": applicable,
                    "standard_forward_error_bound_ratio": standard_ratio,
                    "standard_forward_error_bound_violation": (
                        applicable and standard_ratio > 1.0
                    ),
                    "d04_oracle_tolerance_ratio": d04_ratio,
                    "d04_oracle_tolerance_violation": d04_ratio > 1.0,
                    "absolute_error": absolute_error,
                    "float32_ulp_distance": abs(
                        _ordered_float32_bits(candidate_value)
                        - _ordered_float32_bits(oracle_value)
                    ),
                    "condition_ratio_kind": condition_ratio_kind,
                    "condition_ratio": condition_ratio,
                }
            )
    finite_condition_ratios = [
        float(record["condition_ratio"])
        for record in records
        if record["condition_ratio_kind"] == "finite"
    ]
    exact_zero_condition_count = sum(
        int(record["condition_ratio_kind"] == "exact_zero")
        for record in records
    )
    summary: dict[str, object] = {
        "comparison_count": len(records),
        "standard_bound_applicable_count": sum(
            int(record["standard_forward_error_bound_applicable"])
            for record in records
        ),
        "applicable_standard_bound_violation_count": (
            applicable_violation_count
        ),
        "mixed_sign_dot_count": mixed_sign_count,
        "subnormal_term_dot_count": subnormal_term_dot_count,
        "candidate_zero_result_count": candidate_zero_result_count,
        "candidate_nonzero_result_count": (
            len(records) - candidate_zero_result_count
        ),
        "oracle_zero_result_count": oracle_zero_result_count,
        "oracle_nonzero_result_count": (
            len(records) - oracle_zero_result_count
        ),
        "d04_violation_count": sum(
            int(record["d04_oracle_tolerance_violation"])
            for record in records
        ),
        "minimum_finite_condition_ratio": (
            min(finite_condition_ratios)
            if finite_condition_ratios
            else None
        ),
        "exact_zero_condition_count": exact_zero_condition_count,
        "condition_ratio_gate_passed": all(
            record["condition_ratio_kind"] == "exact_zero"
            or (
                isinstance(record["condition_ratio"], float)
                and float(record["condition_ratio"]) >= 2.0**18
            )
            for record in records
        ),
        "records": records,
    }
    summary["exact_digest"] = stable_payload_digest(summary)
    return summary


def _authenticated_oracle_metrics(
    candidate_output: Tensor,
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    classification: Mapping[str, object],
) -> dict[str, object]:
    """Bind shared oracle statistics to the stricter frozen applicability rule."""

    metrics = _independent_oracle_error_metrics(
        candidate_output,
        inputs,
        weight,
        bias,
    )
    raw = dict(metrics)
    raw_digest = raw.pop("exact_digest", None)
    if raw_digest != stable_payload_digest(raw):
        raise RecurrentKernelDevelopmentScreenError(
            "shared independent oracle metric digest is invalid"
        )
    comparison_count = classification.get("comparison_count")
    applicable_count = classification.get(
        "standard_bound_applicable_count"
    )
    violation_count = classification.get(
        "applicable_standard_bound_violation_count"
    )
    if (
        isinstance(comparison_count, bool)
        or not isinstance(comparison_count, int)
        or isinstance(applicable_count, bool)
        or not isinstance(applicable_count, int)
        or isinstance(violation_count, bool)
        or not isinstance(violation_count, int)
        or metrics.get("comparison_count") != comparison_count
        or not 0 <= violation_count <= applicable_count <= comparison_count
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "authenticated oracle classification counts are malformed"
        )
    forward_error_bound = metrics.get("forward_error_bound")
    if not isinstance(forward_error_bound, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "shared independent oracle precondition evidence is malformed"
        )
    authenticated = dict(metrics)
    authenticated.pop("exact_digest", None)
    authenticated["forward_error_bound"] = {
        **forward_error_bound,
        "standard_bound_precondition": (
            "all_exact_product_and_bias_terms_finite_and_every_nonzero_"
            "term_normal_in_binary32_and_all_nonzero_terms_share_one_sign_"
            "and_binary64_math_fsum_pre_round_is_zero_or_float32_normal"
        ),
        "pre_round_normal_or_exact_zero_required": True,
        "applicability_source": (
            "recurrent_accuracy_screen_per_dot_authenticated_v1"
        ),
    }
    authenticated["standard_bound_applicable"] = (
        applicable_count == comparison_count
    )
    authenticated["standard_bound_applicable_comparison_count"] = (
        applicable_count
    )
    authenticated["standard_bound_inapplicable_comparison_count"] = (
        comparison_count - applicable_count
    )
    authenticated["within_forward_error_bound_count"] = (
        applicable_count - violation_count
    )
    authenticated["forward_error_bound_violation_count"] = violation_count
    authenticated["standard_forward_error_bound_passed"] = (
        violation_count == 0
    )
    authenticated["exact_digest"] = stable_payload_digest(authenticated)
    return authenticated


def _serialize_float32_tensor(tensor: Tensor) -> dict[str, object]:
    detached = tensor.detach().contiguous()
    if (
        detached.device.type != "cpu"
        or detached.device.index is not None
        or detached.dtype != torch.float32
        or detached.layout != torch.strided
        or not bool(torch.isfinite(detached).all().item())
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized accuracy tensor must be finite CPU float32 strided"
        )
    payload: dict[str, object] = {
        "dtype": "torch.float32",
        "shape": [int(size) for size in detached.shape],
        "component_encoding": "ieee754_binary32_big_endian_hex_v1",
        "binary32_big_endian_hex": [
            struct.pack(">f", float(value)).hex()
            for value in detached.reshape(-1).tolist()
        ],
    }
    payload["exact_digest"] = stable_payload_digest(payload)
    return payload


def _deserialize_float32_tensor(payload: object) -> Tensor:
    if not isinstance(payload, dict) or set(payload) != {
        "dtype",
        "shape",
        "component_encoding",
        "binary32_big_endian_hex",
        "exact_digest",
    }:
        raise RecurrentKernelDevelopmentScreenError(
            "serialized accuracy tensor schema is malformed"
        )
    candidate = dict(payload)
    exact_digest = candidate.pop("exact_digest", None)
    if exact_digest != stable_payload_digest(candidate):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized accuracy tensor digest is invalid"
        )
    if (
        candidate.get("dtype") != "torch.float32"
        or candidate.get("component_encoding")
        != "ieee754_binary32_big_endian_hex_v1"
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized accuracy tensor encoding is invalid"
        )
    shape = candidate.get("shape")
    components = candidate.get("binary32_big_endian_hex")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            for size in shape
        )
        or not isinstance(components, list)
        or len(components) != math.prod(shape)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized accuracy tensor shape is malformed"
        )
    values: list[float] = []
    for component in components:
        if (
            not isinstance(component, str)
            or len(component) != 8
            or any(character not in "0123456789abcdef" for character in component)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "serialized accuracy tensor component is malformed"
            )
        values.append(struct.unpack(">f", bytes.fromhex(component))[0])
    tensor = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(
        tuple(shape)
    )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized accuracy tensor contains non-finite values"
        )
    return tensor


def _evaluate_generated_case(
    candidate_forward: LinearForward,
    generated: GeneratedLinearCase,
    *,
    serialize_selected_values: bool = False,
) -> dict[str, object]:
    with torch.inference_mode():
        full_output = candidate_forward(
            generated.inputs,
            generated.weight,
            generated.bias,
        )
        selected_output = _selected_candidate_output(
            full_output,
            generated.probe,
        )
    inputs, weight, bias = _selected_components(generated)
    classification = _per_dot_classification(
        probe=generated.probe,
        inputs=inputs,
        weight=weight,
        bias=bias,
        selected_candidate_output=selected_output,
    )
    metrics = _authenticated_oracle_metrics(
        selected_output,
        inputs,
        weight,
        bias,
        classification=classification,
    )
    evidence: dict[str, object] = {
        "case": generated.probe.as_contract(),
        "family": generated.family,
        "full_component_exact_digests": {
            "input": _float32_tensor_exact_digest(generated.inputs),
            "weight": _float32_tensor_exact_digest(generated.weight),
            "bias": _float32_tensor_exact_digest(generated.bias),
            "candidate_output": _float32_tensor_exact_digest(full_output),
        },
        "selected_component_exact_digests": {
            "input": _float32_tensor_exact_digest(inputs),
            "weight": _float32_tensor_exact_digest(weight),
            "bias": _float32_tensor_exact_digest(bias),
            "candidate_output": _float32_tensor_exact_digest(selected_output),
        },
        "metrics": metrics,
        "per_dot_classification": classification,
    }
    if serialize_selected_values:
        oracle_output, _ = _independent_float64_linear_oracle(
            inputs,
            weight,
            bias,
        )
        evidence["serialized_selected_values"] = {
            "input": _serialize_float32_tensor(inputs),
            "weight": _serialize_float32_tensor(weight),
            "bias": _serialize_float32_tensor(bias),
            "candidate_output": _serialize_float32_tensor(selected_output),
            "full_candidate_output": _serialize_float32_tensor(full_output),
            "oracle_output": _serialize_float32_tensor(oracle_output),
        }
    evidence["exact_digest"] = stable_payload_digest(evidence)
    return evidence


def _aggregate_forward_evidence(
    groups: Mapping[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    dots = 0
    products = 0
    normal_d04_violations = 0
    normal_standard_violations = 0
    normal_subnormal_term_dots = 0
    cancellation_d04_violations = 0
    cancellation_condition_violations = 0
    cancellation_subnormal_term_dots = 0
    underflow_standard_applicable = 0
    underflow_subnormal_term_dots: dict[str, int] = {}
    underflow_result_counts: dict[str, dict[str, int]] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for group_name, cases in groups.items():
        for case in cases:
            case_contract = case["case"]
            metrics = case["metrics"]
            classification = case["per_dot_classification"]
            if not isinstance(case_contract, dict) or not isinstance(
                metrics,
                dict,
            ) or not isinstance(
                classification,
                dict,
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    "forward evidence case is malformed"
                )
            dots += int(case_contract["dot_count"])
            products += int(case_contract["product_count"])
            if group_name in {"ordinary", "holdout"}:
                family = str(case["family"])
                family_summary = family_counts.setdefault(
                    family,
                    {
                        "comparison_count": 0,
                        "standard_bound_applicable_count": 0,
                        "mixed_sign_dot_count": 0,
                        "subnormal_term_dot_count": 0,
                    },
                )
                family_summary["comparison_count"] += int(
                    classification["comparison_count"]
                )
                family_summary["standard_bound_applicable_count"] += int(
                    classification["standard_bound_applicable_count"]
                )
                family_summary["mixed_sign_dot_count"] += int(
                    classification["mixed_sign_dot_count"]
                )
                family_summary["subnormal_term_dot_count"] += int(
                    classification["subnormal_term_dot_count"]
                )
                normal_subnormal_term_dots += int(
                    classification["subnormal_term_dot_count"]
                )
                normal_d04_violations += int(
                    classification["d04_violation_count"]
                )
                normal_standard_violations += int(
                    classification[
                        "applicable_standard_bound_violation_count"
                    ]
                )
            elif group_name == "cancellation":
                cancellation_d04_violations += int(
                    classification["d04_violation_count"]
                )
                cancellation_condition_violations += int(
                    not bool(classification["condition_ratio_gate_passed"])
                )
                cancellation_subnormal_term_dots += int(
                    classification["subnormal_term_dot_count"]
                )
            elif group_name.startswith("underflow"):
                underflow_standard_applicable += int(
                    classification["standard_bound_applicable_count"]
                )
                underflow_subnormal_term_dots[group_name] = (
                    underflow_subnormal_term_dots.get(group_name, 0)
                    + int(classification["subnormal_term_dot_count"])
                )
                result_summary = underflow_result_counts.setdefault(
                    group_name,
                    {
                        "candidate_zero_result_count": 0,
                        "candidate_nonzero_result_count": 0,
                        "oracle_zero_result_count": 0,
                        "oracle_nonzero_result_count": 0,
                    },
                )
                for field in result_summary:
                    result_summary[field] += int(classification[field])
    family_coverage_passed = all(
        (
            "alternating" not in family_counts
            or (
                family_counts["alternating"]["comparison_count"]
                == _EXPECTED_OPERATIONAL_DOT_COUNT
                and family_counts["alternating"][
                    "standard_bound_applicable_count"
                ]
                == _EXPECTED_OPERATIONAL_DOT_COUNT
            ),
            "balanced" not in family_counts
            or family_counts["balanced"]["mixed_sign_dot_count"] > 0,
            "mixed" not in family_counts
            or family_counts["mixed"]["mixed_sign_dot_count"] > 0,
            "composite" not in family_counts
            or family_counts["composite"]["mixed_sign_dot_count"] > 0,
        )
    )
    underflow_subnormal_coverage_passed = (
        not underflow_subnormal_term_dots
        or underflow_subnormal_term_dots
        == {
            "underflow_u1": _EXPECTED_UNDERFLOW_DOT_COUNT,
            "underflow_u2": _EXPECTED_UNDERFLOW_DOT_COUNT,
        }
    )
    underflow_result_count_coverage_passed = (
        not underflow_result_counts
        or (
            set(underflow_result_counts)
            == {"underflow_u1", "underflow_u2"}
            and all(
                summary["candidate_zero_result_count"]
                + summary["candidate_nonzero_result_count"]
                == _EXPECTED_UNDERFLOW_DOT_COUNT
                and summary["oracle_zero_result_count"]
                + summary["oracle_nonzero_result_count"]
                == _EXPECTED_UNDERFLOW_DOT_COUNT
                for summary in underflow_result_counts.values()
            )
        )
    )
    return {
        "dot_count": dots,
        "product_count": products,
        "maximum_dot_count": _MAXIMUM_DOT_COUNT,
        "maximum_product_count": _MAXIMUM_PRODUCT_COUNT,
        "normal_d04_violation_count": normal_d04_violations,
        "normal_standard_forward_error_violation_count": (
            normal_standard_violations
        ),
        "normal_subnormal_term_dot_count": normal_subnormal_term_dots,
        "cancellation_d04_violation_count": (
            cancellation_d04_violations
        ),
        "cancellation_condition_violation_count": (
            cancellation_condition_violations
        ),
        "cancellation_subnormal_term_dot_count": (
            cancellation_subnormal_term_dots
        ),
        "underflow_standard_bound_applicable_count": (
            underflow_standard_applicable
        ),
        "underflow_subnormal_term_dot_count_by_family": (
            underflow_subnormal_term_dots
        ),
        "underflow_subnormal_term_dot_count": sum(
            underflow_subnormal_term_dots.values()
        ),
        "underflow_subnormal_coverage_passed": (
            underflow_subnormal_coverage_passed
        ),
        "underflow_result_counts_by_family": underflow_result_counts,
        "underflow_result_count_coverage_passed": (
            underflow_result_count_coverage_passed
        ),
        "normal_family_classification": family_counts,
        "normal_family_coverage_passed": family_coverage_passed,
        "forward_gate_passed": (
            dots <= _MAXIMUM_DOT_COUNT
            and products <= _MAXIMUM_PRODUCT_COUNT
            and normal_d04_violations == 0
            and normal_standard_violations == 0
            and normal_subnormal_term_dots == 0
            and cancellation_d04_violations == 0
            and cancellation_condition_violations == 0
            and cancellation_subnormal_term_dots == 0
            and underflow_standard_applicable == 0
            and underflow_subnormal_coverage_passed
            and underflow_result_count_coverage_passed
            and family_coverage_passed
        ),
    }


def evaluate_public_forward_corpus(
    candidate_forward: LinearForward,
    *,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Evaluate ordinary, cancellation, and diagnostic underflow probes."""

    resolved = _validated_seed_contract(
        sealed_accuracy_seed_contract() if seeds is None else seeds
    )
    ordinary_cases = operational_probe_cases(resolved.ordinary)
    ordinary = [
        _evaluate_generated_case(
            candidate_forward,
            generate_normal_case(case, seed=resolved.ordinary, family=family),
        )
        for family in _ORDINARY_FAMILIES
        for case in ordinary_cases
    ]
    cancellation = [
        _evaluate_generated_case(
            candidate_forward,
            generate_cancellation_case(
                case,
                seed=resolved.cancellation_underflow,
            ),
        )
        for case in cancellation_probe_cases(
            resolved.cancellation_underflow
        )
    ]
    underflow_groups = {
        family: [
            _evaluate_generated_case(
                candidate_forward,
                generate_underflow_case(
                    case,
                    seed=resolved.cancellation_underflow,
                    family=family,
                ),
            )
            for case in underflow_probe_cases(
                resolved.cancellation_underflow,
                family=family,
            )
        ]
        for family in ("underflow_u1", "underflow_u2")
    }
    groups: dict[str, Sequence[dict[str, object]]] = {
        "ordinary": ordinary,
        "cancellation": cancellation,
        **underflow_groups,
    }
    aggregate = _aggregate_forward_evidence(groups)
    if (
        aggregate["dot_count"]
        != _EXPECTED_ORDINARY_DOT_COUNT
        + _EXPECTED_CANCELLATION_DOT_COUNT
        + 2 * _EXPECTED_UNDERFLOW_DOT_COUNT
        or aggregate["product_count"]
        != _EXPECTED_ORDINARY_PRODUCT_COUNT
        + _EXPECTED_CANCELLATION_PRODUCT_COUNT
        + 2 * _EXPECTED_UNDERFLOW_PRODUCT_COUNT
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "public forward corpus count reconstruction failed"
        )
    report: dict[str, object] = {
        "schema_version": RECURRENT_ACCURACY_CORPUS_VERSION,
        "development_only": True,
        "deciding": False,
        "candidate_identity": RECURRENT_ACCURACY_SCREEN_CANDIDATE,
        "seed_contract_digest": _seed_contract_digest(resolved),
        "groups": groups,
        "aggregate": aggregate,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def validate_public_forward_report(
    report: object,
    *,
    candidate_forward: LinearForward,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Re-execute only the public corpus and require exact reconstruction."""

    if not isinstance(report, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "public forward report must be an object"
        )
    expected = evaluate_public_forward_corpus(
        candidate_forward,
        seeds=seeds,
    )
    if report != expected:
        raise RecurrentKernelDevelopmentScreenError(
            "public forward report reconstruction failed"
        )
    return report


def _seed_contract_digest(seeds: AccuracySeedContract) -> str:
    return stable_payload_digest(
        {
            "ordinary": _seed_digest(seeds.ordinary, "ordinary"),
            "holdout": _seed_digest(seeds.holdout, "holdout"),
            "cancellation_underflow": _seed_digest(
                seeds.cancellation_underflow,
                "cancellation_underflow",
            ),
            "gradient": _seed_digest(seeds.gradient, "gradient"),
            "pairwise_distinct": True,
            "synthetic_test_contract": seeds.synthetic_test_contract,
        }
    )


class HoldoutSeedAuthority:
    """Process-bound, exactly-once access to one holdout corpus."""

    def __init__(
        self,
        *,
        seeds: AccuracySeedContract,
        capability: HoldoutCapability,
        source_sha: str,
        source_clean_detached: bool,
        checkout_root: str,
        required_source_paths: Sequence[str],
        source_module_digests: Mapping[str, str],
        method_document_path: str,
        method_document_digest: str,
        source_retirement_marker_path: str,
        source_retirement_marker_sha256: str,
        holdout_retirement_marker_path: str,
        holdout_retirement_marker_sha256: str,
        capability_commitment_sha256: str,
        parent_process_id: int,
        parent_process_start_identity: str,
        child_process_id: int,
        child_process_start_identity: str,
        child_nonce: str,
        report_path: str,
        preregistration_digest: str,
        candidate_identity: str,
        preholdout_evidence_digest: str,
        closing_baseline_evidence_digest: str,
    ) -> None:
        self._seeds = _validated_seed_contract(seeds)
        if not isinstance(capability, HoldoutCapability):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout parent capability is malformed"
            )
        _validated_hex(
            capability.token,
            length=64,
            field="holdout parent capability token",
        )
        self._source_sha = _validated_hex(
            source_sha,
            length=_SOURCE_SHA_LENGTH,
            field="source SHA",
        )
        if source_clean_detached is not True:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout source must be a clean detached checkout"
            )
        self._checkout_root = Path(checkout_root)
        self._required_source_paths = tuple(required_source_paths)
        if not isinstance(source_module_digests, Mapping) or not source_module_digests:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout source-module digests must be a non-empty mapping"
            )
        self._source_module_digests = {
            path: _validated_hex(
                digest,
                length=64,
                field=f"source-module digest {path}",
            )
            for path, digest in sorted(source_module_digests.items())
            if isinstance(path, str) and path
        }
        if len(self._source_module_digests) != len(source_module_digests):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout source-module digest path is malformed"
            )
        self._source_module_digest = stable_payload_digest(
            self._source_module_digests
        )
        self._method_document_digest = _validated_hex(
            method_document_digest,
            length=64,
            field="method-document digest",
        )
        if method_document_path != RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout method-document path differs from the frozen path"
            )
        self._method_document_path = method_document_path
        _validate_source_files(
            checkout_root=self._checkout_root,
            required_source_paths=self._required_source_paths,
            source_module_digests=self._source_module_digests,
        )
        _validate_method_document_file(
            checkout_root=self._checkout_root,
            method_document_path=self._method_document_path,
            method_document_digest=self._method_document_digest,
        )
        for field, value in (
            ("parent process ID", parent_process_id),
            ("child process ID", child_process_id),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    f"holdout {field} must be a positive integer"
                )
        if child_process_id != os.getpid():
            raise RecurrentKernelDevelopmentScreenError(
                "holdout child process ID is not the current process"
            )
        if parent_process_id != os.getppid():
            raise RecurrentKernelDevelopmentScreenError(
                "holdout parent process ID is not the live parent"
            )
        if not parent_process_start_identity:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout parent process start identity must be non-empty"
            )
        if parent_process_start_identity != process_start_identity(
            parent_process_id
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout parent process start identity is not live"
            )
        if not child_process_start_identity:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout child process start identity must be non-empty"
            )
        if child_process_start_identity != process_start_identity(
            child_process_id
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout child process start identity is not live"
            )
        child_nonce = _validated_hex(
            child_nonce,
            length=64,
            field="holdout child nonce",
        )
        if not report_path or not Path(report_path).is_absolute():
            raise RecurrentKernelDevelopmentScreenError(
                "holdout report path must be absolute"
            )
        if Path(report_path).exists():
            raise RecurrentKernelDevelopmentScreenError(
                "holdout report path must be absent before access"
            )
        self._parent_process_id = parent_process_id
        self._parent_process_start_identity = parent_process_start_identity
        self._child_process_id = child_process_id
        self._child_process_start_identity = child_process_start_identity
        self._child_nonce = child_nonce
        self._report_path = report_path
        self._preregistration_digest = _validated_hex(
            preregistration_digest,
            length=64,
            field="preregistration digest",
        )
        if candidate_identity != RECURRENT_ACCURACY_SCREEN_CANDIDATE:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout candidate identity is not the sole frozen candidate"
            )
        self._candidate_identity = candidate_identity
        self._preholdout_evidence_digest = _validated_hex(
            preholdout_evidence_digest,
            length=64,
            field="pre-holdout evidence digest",
        )
        self._closing_baseline_evidence_digest = _validated_hex(
            closing_baseline_evidence_digest,
            length=64,
            field="closing-baseline evidence digest",
        )
        self._capability_commitment_sha256 = _validated_hex(
            capability_commitment_sha256,
            length=64,
            field="capability commitment",
        )
        if (
            capability.preholdout_evidence_digest
            != self._preholdout_evidence_digest
            or capability.candidate_identity != self._candidate_identity
            or capability.capability_commitment_sha256
            != self._capability_commitment_sha256
            or holdout_capability_commitment(capability.token)
            != self._capability_commitment_sha256
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout capability binding does not match child evidence"
            )
        self._source_retirement_marker_path = Path(
            source_retirement_marker_path
        )
        self._source_retirement_marker_sha256 = _validated_hex(
            source_retirement_marker_sha256,
            length=64,
            field="source-retirement-marker SHA256",
        )
        self._holdout_retirement_marker_path = Path(
            holdout_retirement_marker_path
        )
        self._holdout_retirement_marker_sha256 = _validated_hex(
            holdout_retirement_marker_sha256,
            length=64,
            field="holdout-retirement-marker SHA256",
        )
        report_parent = Path(self._report_path).parent
        if (
            self._source_retirement_marker_path.parent != report_parent
            or self._source_retirement_marker_path.name
            != RECURRENT_ACCURACY_SOURCE_RETIREMENT_MARKER_NAME
            or self._holdout_retirement_marker_path.parent != report_parent
            or self._holdout_retirement_marker_path.name
            != RECURRENT_ACCURACY_HOLDOUT_RETIREMENT_MARKER_NAME
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout retirement markers are not canonical report siblings"
            )
        if (
            capability.source_retirement_marker_sha256
            != self._source_retirement_marker_sha256
            or capability.holdout_retirement_marker_sha256
            != self._holdout_retirement_marker_sha256
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout capability retirement marker binding is invalid"
            )
        expected_source_marker = source_retirement_marker(
            source_sha=self._source_sha,
            source_module_digest=self._source_module_digest,
            method_document_digest=self._method_document_digest,
            preregistration_digest=self._preregistration_digest,
            report_path=self._report_path,
            parent_process_id=self._parent_process_id,
            parent_process_start_identity=(
                self._parent_process_start_identity
            ),
            capability_commitment_sha256=(
                self._capability_commitment_sha256
            ),
        )
        self._validate_retirement_marker(
            path=self._source_retirement_marker_path,
            file_sha256=self._source_retirement_marker_sha256,
            expected_marker=expected_source_marker,
            label="source",
        )
        expected_holdout_marker = holdout_retirement_marker(
            source_retirement_marker_sha256=(
                self._source_retirement_marker_sha256
            ),
            source_retirement_marker_exact_digest=str(
                expected_source_marker["exact_digest"]
            ),
            preholdout_evidence_digest=self._preholdout_evidence_digest,
            closing_baseline_evidence_digest=(
                self._closing_baseline_evidence_digest
            ),
            parent_process_id=self._parent_process_id,
            parent_process_start_identity=(
                self._parent_process_start_identity
            ),
            child_process_id=self._child_process_id,
            child_process_start_identity=self._child_process_start_identity,
            child_nonce=self._child_nonce,
            capability_commitment_sha256=(
                self._capability_commitment_sha256
            ),
        )
        self._validate_retirement_marker(
            path=self._holdout_retirement_marker_path,
            file_sha256=self._holdout_retirement_marker_sha256,
            expected_marker=expected_holdout_marker,
            label="holdout",
        )
        self._expected_source_marker = expected_source_marker
        self._expected_holdout_marker = expected_holdout_marker
        self._token = capability.token
        self._capability = capability
        self._generation_token = secrets.token_hex(32)
        self._consumed = False

    def _validate_retirement_marker(
        self,
        *,
        path: Path,
        file_sha256: str,
        expected_marker: Mapping[str, object],
        label: str,
    ) -> None:
        _validate_retirement_marker_file(
            path=path,
            file_sha256=file_sha256,
            expected_marker=expected_marker,
            label=label,
        )

    def _consume(
        self,
        capability: HoldoutCapability,
    ) -> tuple[_HoldoutGenerationPermit, dict[str, object]]:
        if self._consumed:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout access is exactly once"
            )
        if (
            not isinstance(capability, HoldoutCapability)
            or capability != self._capability
            or not secrets.compare_digest(capability.token, self._token)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout capability is invalid"
            )
        if os.getpid() != self._child_process_id:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout capability moved to a different process"
            )
        if (
            os.getppid() != self._parent_process_id
            or process_start_identity(os.getppid())
            != self._parent_process_start_identity
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "holdout parent process identity drifted"
            )
        if process_start_identity(os.getpid()) != self._child_process_start_identity:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout child process start identity drifted"
            )
        if Path(self._report_path).exists():
            raise RecurrentKernelDevelopmentScreenError(
                "holdout report path was created before consumption"
            )
        _validate_source_files(
            checkout_root=self._checkout_root,
            required_source_paths=self._required_source_paths,
            source_module_digests=self._source_module_digests,
        )
        _validate_method_document_file(
            checkout_root=self._checkout_root,
            method_document_path=self._method_document_path,
            method_document_digest=self._method_document_digest,
        )
        self._validate_retirement_marker(
            path=self._source_retirement_marker_path,
            file_sha256=self._source_retirement_marker_sha256,
            expected_marker=self._expected_source_marker,
            label="source",
        )
        self._validate_retirement_marker(
            path=self._holdout_retirement_marker_path,
            file_sha256=self._holdout_retirement_marker_sha256,
            expected_marker=self._expected_holdout_marker,
            label="holdout",
        )
        if (
            not self._seeds.synthetic_test_contract
            and os.environ.get("PYTEST_CURRENT_TEST")
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "sealed holdout cannot be consumed under a test runner"
            )
        self._consumed = True
        pending_case_ids = {
            case.case_id
            for case in operational_probe_cases(self._seeds.holdout)
        }
        if self._generation_token in _HOLDOUT_GENERATION_REGISTRY:
            raise RecurrentKernelDevelopmentScreenError(
                "holdout generation token unexpectedly exists"
            )
        _HOLDOUT_GENERATION_REGISTRY[self._generation_token] = (
            self._seeds.holdout,
            pending_case_ids,
        )
        receipt: dict[str, object] = {
            "schema_version": RECURRENT_ACCURACY_SCREEN_HOLDOUT_VERSION,
            "source_sha": self._source_sha,
            "source_clean_detached": True,
            "checkout_root": str(self._checkout_root),
            "required_source_paths": list(self._required_source_paths),
            "source_module_digests": self._source_module_digests,
            "source_module_digest": self._source_module_digest,
            "method_document_path": self._method_document_path,
            "method_document_digest": self._method_document_digest,
            "selection_manifest_schema_version": (
                RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION
            ),
            "selection_manifest_exact_digest": (
                RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST
            ),
            "parent_process_id": self._parent_process_id,
            "parent_process_start_identity": (
                self._parent_process_start_identity
            ),
            "child_process_id": self._child_process_id,
            "child_process_start_identity": (
                self._child_process_start_identity
            ),
            "child_nonce": self._child_nonce,
            "report_path": self._report_path,
            "screen_schema_version": (
                RECURRENT_ACCURACY_SCREEN_SCHEMA_VERSION
            ),
            "corpus_version": RECURRENT_ACCURACY_CORPUS_VERSION,
            "preregistration_digest": self._preregistration_digest,
            "candidate_identity": self._candidate_identity,
            "preholdout_evidence_digest": (
                self._preholdout_evidence_digest
            ),
            "access_ordinal": 1,
            "phase": "candidate_locked_before_selection",
            "test_mode": self._seeds.synthetic_test_contract,
            "pytest_current_test_absent": not bool(
                os.environ.get("PYTEST_CURRENT_TEST")
            ),
            "holdout_access_count": 1,
            "holdout_consumed": True,
            "candidate_locked_before_holdout": True,
            "retirement": "consumed_for_exact_source_no_retry",
            "source_retirement_marker_path": str(
                self._source_retirement_marker_path
            ),
            "source_retirement_marker_sha256": (
                self._source_retirement_marker_sha256
            ),
            "source_retirement_marker_exact_digest": (
                self._expected_source_marker["exact_digest"]
            ),
            "holdout_retirement_marker_path": str(
                self._holdout_retirement_marker_path
            ),
            "holdout_retirement_marker_sha256": (
                self._holdout_retirement_marker_sha256
            ),
            "holdout_retirement_marker_exact_digest": (
                self._expected_holdout_marker["exact_digest"]
            ),
            "capability_commitment_sha256": (
                self._capability_commitment_sha256
            ),
            "closing_baseline_evidence_digest": (
                self._closing_baseline_evidence_digest
            ),
            "seed_digest": _seed_digest(
                self._seeds.holdout,
                "holdout",
            ),
        }
        receipt["exact_digest"] = stable_payload_digest(receipt)
        return (
            _HoldoutGenerationPermit(
                seed=self._seeds.holdout,
                token=self._generation_token,
            ),
            receipt,
        )


def evaluate_holdout_forward_corpus(
    candidate_forward: LinearForward,
    *,
    authority: HoldoutSeedAuthority,
    capability: HoldoutCapability,
) -> dict[str, object]:
    """Consume and evaluate the holdout exactly once."""

    from evolution_sim.mind import recurrent_actor_critic

    if candidate_forward is not recurrent_actor_critic._backend_stable_linear:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout must use the production backend-stable callable"
        )
    if (
        recurrent_actor_critic._backend_stable_linear_forward
        is not recurrent_actor_critic._fixed_row_tile_4_linear_forward
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout production callable is not in fixed-row context"
        )
    permit, receipt = authority._consume(capability)
    try:
        cases = [
            _evaluate_generated_case(
                candidate_forward,
                generate_normal_case(
                    case,
                    seed=permit.seed,
                    family="composite",
                    _holdout_permit=permit,
                ),
                serialize_selected_values=True,
            )
            for case in operational_probe_cases(permit.seed)
        ]
    except BaseException:
        _HOLDOUT_GENERATION_REGISTRY.pop(permit.token, None)
        raise
    if permit.token in _HOLDOUT_GENERATION_REGISTRY:
        _HOLDOUT_GENERATION_REGISTRY.pop(permit.token, None)
        raise RecurrentKernelDevelopmentScreenError(
            "holdout materialization did not consume every frozen case"
        )
    aggregate = _aggregate_forward_evidence({"holdout": cases})
    if (
        aggregate["dot_count"] != _EXPECTED_HOLDOUT_DOT_COUNT
        or aggregate["product_count"] != _EXPECTED_HOLDOUT_PRODUCT_COUNT
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout forward corpus count reconstruction failed"
        )
    report: dict[str, object] = {
        "schema_version": RECURRENT_ACCURACY_CORPUS_VERSION,
        "development_only": True,
        "deciding": receipt["test_mode"] is False,
        "candidate_identity": RECURRENT_ACCURACY_SCREEN_CANDIDATE,
        "holdout_receipt": receipt,
        "groups": {"holdout": cases},
        "aggregate": aggregate,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def _gradient_data(
    probe: LinearProbeCase,
    *,
    seed: int,
) -> GeneratedLinearCase:
    return generate_normal_case(
        probe,
        seed=seed,
        family="balanced",
    )


def _gradient_loss(
    output: Tensor,
    probe: LinearProbeCase,
    *,
    seed: int,
) -> Tensor:
    expected_shape = (probe.physical_rows, probe.output_features)
    if tuple(output.shape) != expected_shape:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient loss output shape is malformed"
        )
    lambdas = _tensor_from_values(
        expected_shape,
        lambda index: _sign_from_word(
            _value_word(
                seed=seed,
                family="gradient",
                case_id=probe.case_id,
                tensor_name="loss_coefficient",
                flat_index=index[0] * probe.output_features + index[1],
                draw=1,
            )
        )
        * (
            1
            + (
                _value_word(
                    seed=seed,
                    family="gradient",
                    case_id=probe.case_id,
                    tensor_name="loss_coefficient",
                    flat_index=index[0] * probe.output_features + index[1],
                    draw=0,
                )
                & 7
            )
        )
        * 2.0**-3,
    )
    return (
        lambdas * (output + output.square() * 2.0**-3)
    ).sum() * 2.0**-8


def _gradient_direction(
    tensor: Tensor,
    *,
    seed: int,
    case_id: str,
    name: str,
) -> Tensor:
    shape = tuple(int(size) for size in tensor.shape)
    width = shape[-1] if len(shape) == 2 else 1
    return _tensor_from_values(
        shape,
        lambda index: _sign_from_word(
            _value_word(
                seed=seed,
                family="gradient",
                case_id=case_id,
                tensor_name=f"hvp_direction_{name}",
                flat_index=(
                    index[0] * width + index[1]
                    if len(index) == 2
                    else index[0]
                ),
                draw=1,
            )
        )
        * (
            1
            + (
                _value_word(
                    seed=seed,
                    family="gradient",
                    case_id=case_id,
                    tensor_name=f"hvp_direction_{name}",
                    flat_index=(
                        index[0] * width + index[1]
                        if len(index) == 2
                        else index[0]
                    ),
                    draw=0,
                )
                & 15
            )
        )
        * 2.0**-8,
    )


def _gradient_lane(
    forward: LinearForward,
    generated: GeneratedLinearCase,
    *,
    seed: int,
) -> tuple[
    Tensor,
    Tensor,
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]:
    inputs = generated.inputs.detach().clone().requires_grad_(True)
    weight = generated.weight.detach().clone().requires_grad_(True)
    bias = generated.bias.detach().clone().requires_grad_(True)
    output = forward(inputs, weight, bias)
    if output.shape != (
        generated.probe.physical_rows,
        generated.probe.output_features,
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "gradient probe output shape is malformed"
        )
    loss = _gradient_loss(output, generated.probe, seed=seed)
    gradients = torch.autograd.grad(
        loss,
        (inputs, weight, bias),
        create_graph=True,
    )
    directions = tuple(
        _gradient_direction(
            tensor,
            seed=seed,
            case_id=generated.probe.case_id,
            name=name,
        )
        for tensor, name in zip(
            (inputs, weight, bias),
            ("input", "weight", "bias"),
            strict=True,
        )
    )
    directional_gradient = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(
            gradients,
            directions,
            strict=True,
        )
    )
    hvps = torch.autograd.grad(
        directional_gradient,
        (inputs, weight, bias),
    )
    return (
        output.detach(),
        loss.detach(),
        tuple(gradients),
        tuple(directions),
        tuple(hvps),
    )


def _named_tensor_digests(
    tensors: Sequence[Tensor],
) -> dict[str, str | None]:
    if len(tensors) != 3:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient tensor digest arity is malformed"
        )
    return {
        name: _float32_tensor_exact_digest(tensor)
        for name, tensor in zip(
            ("input", "weight", "bias"),
            tensors,
            strict=True,
        )
    }


def _comparison_summary(
    candidate: Sequence[Tensor],
    reference: Sequence[Tensor],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    if len(candidate) != len(reference):
        raise RecurrentKernelDevelopmentScreenError(
            "gradient comparison arity is malformed"
        )
    comparison_count = 0
    violation_count = 0
    maximum_ratio = 0.0
    shape_mismatch_count = 0
    nonfinite_count = 0
    for observed, expected in zip(candidate, reference, strict=True):
        if observed.shape != expected.shape:
            shape_mismatch_count += 1
            continue
        comparison_count += int(expected.numel())
        finite = torch.isfinite(observed) & torch.isfinite(expected)
        nonfinite_count += int((~finite).sum().item())
        allowance = absolute_tolerance + relative_tolerance * expected.abs()
        ratio = (observed - expected).abs() / allowance
        finite_ratio = ratio[finite]
        if finite_ratio.numel():
            maximum_ratio = max(
                maximum_ratio,
                float(finite_ratio.max().item()),
            )
            violation_count += int((finite_ratio > 1.0).sum().item())
    return {
        "comparison_count": comparison_count,
        "shape_mismatch_count": shape_mismatch_count,
        "nonfinite_count": nonfinite_count,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "maximum_tolerance_ratio": maximum_ratio,
        "violation_count": violation_count,
        "passed": (
            shape_mismatch_count == 0
            and nonfinite_count == 0
            and violation_count == 0
        ),
    }


def _active_reference_coverage(
    tensors: Sequence[Tensor],
) -> dict[str, object]:
    if len(tensors) != 3:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient active-reference coverage arity is malformed"
        )
    inputs, weight, bias = tensors
    if inputs.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient active-reference coverage shape is malformed"
        )
    nonzero_masks = tuple(tensor.detach().ne(0) for tensor in tensors)
    component_count = sum(int(mask.numel()) for mask in nonzero_masks)
    nonzero_component_count = sum(
        int(mask.sum().item()) for mask in nonzero_masks
    )
    input_row_count = int(inputs.shape[0])
    weight_output_row_count = int(weight.shape[0])
    bias_component_count = int(bias.shape[0])
    nonzero_input_row_count = int(
        nonzero_masks[0].any(dim=1).sum().item()
    )
    nonzero_weight_output_row_count = int(
        nonzero_masks[1].any(dim=1).sum().item()
    )
    nonzero_bias_component_count = int(nonzero_masks[2].sum().item())
    nonzero_share = nonzero_component_count / component_count
    return {
        "component_count": component_count,
        "nonzero_component_count": nonzero_component_count,
        "nonzero_component_share": nonzero_share,
        "input_row_count": input_row_count,
        "nonzero_input_row_count": nonzero_input_row_count,
        "weight_output_row_count": weight_output_row_count,
        "nonzero_weight_output_row_count": (
            nonzero_weight_output_row_count
        ),
        "bias_component_count": bias_component_count,
        "nonzero_bias_component_count": nonzero_bias_component_count,
        "all_input_rows_active": (
            nonzero_input_row_count == input_row_count
        ),
        "all_weight_output_rows_active": (
            nonzero_weight_output_row_count == weight_output_row_count
        ),
        "all_bias_components_active": (
            nonzero_bias_component_count == bias_component_count
        ),
    }


def evaluate_gradient_probe(
    candidate_forward: LinearForward,
    *,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Compare production-backward gradients and HVPs with ``F.linear``."""

    from evolution_sim.mind import recurrent_actor_critic

    if candidate_forward is not recurrent_actor_critic._backend_stable_linear:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient probe must use the production backend-stable callable"
        )
    if (
        recurrent_actor_critic._backend_stable_linear_forward
        is not recurrent_actor_critic._fixed_row_tile_4_linear_forward
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "gradient probe production callable is not in fixed-row context"
        )
    resolved = _validated_seed_contract(
        sealed_accuracy_seed_contract() if seeds is None else seeds
    )
    case_reports: list[dict[str, object]] = []
    gradient_components = 0
    hvp_components = 0
    forward_bitwise_mismatch_count = 0
    for probe in gradient_probe_cases(resolved.gradient):
        generated = _gradient_data(probe, seed=resolved.gradient)
        (
            candidate_output,
            candidate_loss,
            candidate_gradients,
            candidate_directions,
            candidate_hvps,
        ) = _gradient_lane(
            candidate_forward,
            generated,
            seed=resolved.gradient,
        )
        (
            reference_output,
            reference_loss,
            reference_gradients,
            reference_directions,
            reference_hvps,
        ) = _gradient_lane(F.linear, generated, seed=resolved.gradient)
        with torch.inference_mode():
            inference_output = candidate_forward(
                generated.inputs,
                generated.weight,
                generated.bias,
            )
        forward_bitwise_mismatch_count += int(
            not torch.equal(candidate_output, inference_output)
        )
        gradient_summary = _comparison_summary(
            candidate_gradients,
            reference_gradients,
            relative_tolerance=_GRADIENT_RELATIVE_TOLERANCE,
            absolute_tolerance=_GRADIENT_ABSOLUTE_TOLERANCE,
        )
        hvp_summary = _comparison_summary(
            candidate_hvps,
            reference_hvps,
            relative_tolerance=_HVP_RELATIVE_TOLERANCE,
            absolute_tolerance=_HVP_ABSOLUTE_TOLERANCE,
        )
        gradient_components += int(gradient_summary["comparison_count"])
        hvp_components += int(hvp_summary["comparison_count"])
        gradient_active_reference = _active_reference_coverage(
            reference_gradients
        )
        hvp_active_reference = _active_reference_coverage(reference_hvps)
        case_report: dict[str, object] = {
            "case": probe.as_contract(),
            "full_component_exact_digests": {
                "input": _float32_tensor_exact_digest(generated.inputs),
                "weight": _float32_tensor_exact_digest(generated.weight),
                "bias": _float32_tensor_exact_digest(generated.bias),
                "candidate_output": _float32_tensor_exact_digest(
                    candidate_output
                ),
                "reference_output": _float32_tensor_exact_digest(
                    reference_output
                ),
                "candidate_loss": _float32_tensor_exact_digest(
                    candidate_loss.reshape(1)
                ),
                "reference_loss": _float32_tensor_exact_digest(
                    reference_loss.reshape(1)
                ),
                "hvp_direction": _named_tensor_digests(
                    candidate_directions
                ),
                "reference_hvp_direction": _named_tensor_digests(
                    reference_directions
                ),
                "candidate_gradient": _named_tensor_digests(
                    candidate_gradients
                ),
                "reference_gradient": _named_tensor_digests(
                    reference_gradients
                ),
                "candidate_hvp": _named_tensor_digests(candidate_hvps),
                "reference_hvp": _named_tensor_digests(reference_hvps),
            },
            "gradient": gradient_summary,
            "hvp": hvp_summary,
            "gradient_active_reference_coverage": (
                gradient_active_reference
            ),
            "hvp_active_reference_coverage": hvp_active_reference,
            "hvp_direction_bitwise_match": all(
                torch.equal(candidate, reference)
                for candidate, reference in zip(
                    candidate_directions,
                    reference_directions,
                    strict=True,
                )
            ),
            "candidate_inference_matches_gradient_forward_bitwise": torch.equal(
                candidate_output,
                inference_output,
            ),
        }
        case_report["exact_digest"] = stable_payload_digest(case_report)
        case_reports.append(case_report)
    if gradient_components != _EXPECTED_GRADIENT_COMPONENT_COUNT:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient comparison count drifted"
        )
    if hvp_components != _EXPECTED_GRADIENT_COMPONENT_COUNT:
        raise RecurrentKernelDevelopmentScreenError(
            "HVP comparison count drifted"
        )
    aggregate = {
        "loss_term_count": _EXPECTED_GRADIENT_LOSS_TERM_COUNT,
        "diagnostic_spot_selection_count": (
            _EXPECTED_GRADIENT_DIAGNOSTIC_SPOT_COUNT
        ),
        "gradient_component_count": gradient_components,
        "hvp_component_count": hvp_components,
        "gradient_and_hvp_component_count": (
            gradient_components + hvp_components
        ),
        "expected_gradient_and_hvp_component_count": (
            _EXPECTED_GRADIENT_AND_HVP_COMPONENT_COUNT
        ),
        "gradient_violation_count": sum(
            int(case["gradient"]["violation_count"])  # type: ignore[index]
            for case in case_reports
        ),
        "hvp_violation_count": sum(
            int(case["hvp"]["violation_count"])  # type: ignore[index]
            for case in case_reports
        ),
        "shape_mismatch_count": sum(
            int(case["gradient"]["shape_mismatch_count"])  # type: ignore[index]
            + int(case["hvp"]["shape_mismatch_count"])  # type: ignore[index]
            for case in case_reports
        ),
        "nonfinite_count": sum(
            int(case["gradient"]["nonfinite_count"])  # type: ignore[index]
            + int(case["hvp"]["nonfinite_count"])  # type: ignore[index]
            for case in case_reports
        ),
        "candidate_inference_gradient_forward_bitwise_mismatch_count": (
            forward_bitwise_mismatch_count
        ),
        "hvp_direction_bitwise_mismatch_count": sum(
            int(not bool(case["hvp_direction_bitwise_match"]))
            for case in case_reports
        ),
    }
    for order in ("gradient", "hvp"):
        coverage_key = f"{order}_active_reference_coverage"
        component_count = sum(
            int(case[coverage_key]["component_count"])  # type: ignore[index]
            for case in case_reports
        )
        nonzero_component_count = sum(
            int(case[coverage_key]["nonzero_component_count"])  # type: ignore[index]
            for case in case_reports
        )
        aggregate[f"{order}_active_reference_component_count"] = (
            component_count
        )
        aggregate[f"{order}_active_reference_nonzero_component_count"] = (
            nonzero_component_count
        )
        aggregate[f"{order}_active_reference_nonzero_component_share"] = (
            nonzero_component_count / component_count
        )
        aggregate[f"{order}_all_input_rows_active"] = all(
            bool(case[coverage_key]["all_input_rows_active"])  # type: ignore[index]
            for case in case_reports
        )
        aggregate[f"{order}_all_weight_output_rows_active"] = all(
            bool(case[coverage_key]["all_weight_output_rows_active"])  # type: ignore[index]
            for case in case_reports
        )
        aggregate[f"{order}_all_bias_components_active"] = all(
            bool(case[coverage_key]["all_bias_components_active"])  # type: ignore[index]
            for case in case_reports
        )
        aggregate[f"{order}_active_reference_coverage_passed"] = all(
            (
                component_count == _EXPECTED_GRADIENT_COMPONENT_COUNT,
                nonzero_component_count / component_count >= 0.95,
                aggregate[f"{order}_all_input_rows_active"],
                aggregate[f"{order}_all_weight_output_rows_active"],
                aggregate[f"{order}_all_bias_components_active"],
            )
        )
    aggregate["gradient_gate_passed"] = all(
        (
            aggregate["gradient_and_hvp_component_count"]
            == _EXPECTED_GRADIENT_AND_HVP_COMPONENT_COUNT,
            aggregate["gradient_violation_count"] == 0,
            aggregate["hvp_violation_count"] == 0,
            aggregate["shape_mismatch_count"] == 0,
            aggregate["nonfinite_count"] == 0,
            forward_bitwise_mismatch_count == 0,
            aggregate["hvp_direction_bitwise_mismatch_count"] == 0,
            aggregate["gradient_active_reference_coverage_passed"],
            aggregate["hvp_active_reference_coverage_passed"],
        )
    )
    report: dict[str, object] = {
        "schema_version": RECURRENT_ACCURACY_SCREEN_GRADIENT_VERSION,
        "development_only": True,
        "deciding": False,
        "candidate_identity": RECURRENT_ACCURACY_SCREEN_CANDIDATE,
        "seed_contract_digest": _seed_contract_digest(resolved),
        "cases": case_reports,
        "aggregate": aggregate,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def validate_gradient_probe_report(
    report: object,
    *,
    candidate_forward: LinearForward,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Re-execute the public gradient/HVP probe and require exact evidence."""

    if not isinstance(report, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "gradient probe report must be an object"
        )
    expected = evaluate_gradient_probe(candidate_forward, seeds=seeds)
    if report != expected:
        raise RecurrentKernelDevelopmentScreenError(
            "gradient probe report reconstruction failed"
        )
    return report


def accuracy_corpus_contract(
    *,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Return a fully reconstructible prospective corpus contract."""

    resolved = _validated_seed_contract(
        sealed_accuracy_seed_contract() if seeds is None else seeds
    )
    ordinary_cases = operational_probe_cases(resolved.ordinary)
    holdout_cases = operational_probe_cases(resolved.holdout)
    cancellation_cases = cancellation_probe_cases(
        resolved.cancellation_underflow
    )
    underflow_u1_cases = underflow_probe_cases(
        resolved.cancellation_underflow,
        family="underflow_u1",
    )
    underflow_u2_cases = underflow_probe_cases(
        resolved.cancellation_underflow,
        family="underflow_u2",
    )
    gradient_cases = gradient_probe_cases(resolved.gradient)
    if not resolved.synthetic_test_contract:
        for group, cases in (
            ("ordinary", ordinary_cases),
            ("holdout", holdout_cases),
            ("cancellation", cancellation_cases),
            ("underflow_u1", underflow_u1_cases),
            ("underflow_u2", underflow_u2_cases),
            ("gradient", gradient_cases),
        ):
            _validate_cases_against_frozen_manifest(group, cases)
    contract: dict[str, object] = {
        "schema_version": RECURRENT_ACCURACY_CORPUS_VERSION,
        "development_only": True,
        "candidate_identity": RECURRENT_ACCURACY_SCREEN_CANDIDATE,
        "candidate_count": 1,
        "selection_manifest_schema_version": (
            RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION
        ),
        "selection_manifest_exact_digest": (
            RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST
        ),
        "frozen_real_selection_manifest": (
            None
            if resolved.synthetic_test_contract
            else frozen_real_selection_manifest()
        ),
        "selection_version": RECURRENT_ACCURACY_SCREEN_SELECTION_VERSION,
        "rng_version": RECURRENT_ACCURACY_SCREEN_RNG_VERSION,
        "seed_contract": {
            "ordinary_seed": resolved.ordinary,
            "holdout_seed": resolved.holdout,
            "cancellation_underflow_seed": (
                resolved.cancellation_underflow
            ),
            "gradient_seed": resolved.gradient,
            "pairwise_distinct": True,
            "synthetic_test_contract": resolved.synthetic_test_contract,
            "exact_digest": _seed_contract_digest(resolved),
        },
        "selection_algorithm": {
            "score": (
                "SHA256(UTF8('oe-accuracy-corpus-v1\\0'+decimal_seed+'\\0'+"
                "case_id+'\\0'+axis+'\\0'+decimal_index))"
            ),
            "ordering": "ascending_score_bytes_then_integer_index",
            "fill": "pinned_indices_then_ranked_fill_then_numeric_sort",
            "row_pins": "zero_and_last_distinct",
            "output_pins": {
                "768": [0, 255, 256, 511, 512, 767],
                "256": [0, 255],
                "20": [0, 19],
                "1": [0],
            },
        },
        "rng_algorithm": {
            "domain_hash": (
                "first_8_bytes_big_endian_sha256("
                "oe-accuracy-value-v1_nul_family_nul_case_nul_tensor)"
            ),
            "state": (
                "seed_plus_domain_plus_C_times_"
                "(1_plus_4_times_flat_index_plus_draw)_mod_2^64"
            ),
            "generator": "splitmix64_reference_constants",
            "value_encoding": "exact_binary_dyadic_then_float32",
        },
        "value_formulas": {
            "balanced_input_weight": (
                "(-1)^(h1&1)*(1+(h0&1023))*2^-12"
            ),
            "balanced_bias": "(-1)^(h1&1)*2^-14",
            "alternating": (
                "balanced_magnitude_with_coordinate_parity_signs"
            ),
            "mixed_input_weight": (
                "(-1)^(h1&1)*(1024+(h0&1023))*"
                "2^([-12,-8,-4][h2%3]-10)"
            ),
            "mixed_bias": (
                "(-1)^(h1&1)*(1024+(h0&1023))*"
                "2^([-14,-10,-6][h2%3]-10)"
            ),
            "composite": (
                "h3%3_selects_balanced_alternating_or_mixed_"
                "within_the_composite_domain"
            ),
            "cancellation": (
                "mirrored_halves_(1024+(h0&1023))*2^-11_"
                "with_opposite_weight_halves"
            ),
            "underflow_u1": "input_weight_2^-70_bias_signed_2^-120",
            "underflow_u2": "input_weight_2^-75_bias_zero",
        },
        "tables": {
            "ordinary": [case.as_contract() for case in ordinary_cases],
            "holdout": [case.as_contract() for case in holdout_cases],
            "cancellation": [
                case.as_contract() for case in cancellation_cases
            ],
            "underflow_u1": [
                case.as_contract() for case in underflow_u1_cases
            ],
            "underflow_u2": [
                case.as_contract() for case in underflow_u2_cases
            ],
            "gradient": [case.as_contract() for case in gradient_cases],
        },
        "families": {
            "ordinary": list(_ORDINARY_FAMILIES),
            "holdout": ["composite"],
            "cancellation": ["cancellation_mirrored_halves"],
            "underflow": ["underflow_u1", "underflow_u2"],
            "gradient": ["balanced"],
        },
        "counts": {
            "operational_case_count": len(ordinary_cases),
            "ordinary_dot_count": _EXPECTED_ORDINARY_DOT_COUNT,
            "ordinary_product_count": _EXPECTED_ORDINARY_PRODUCT_COUNT,
            "holdout_dot_count": _EXPECTED_HOLDOUT_DOT_COUNT,
            "holdout_product_count": _EXPECTED_HOLDOUT_PRODUCT_COUNT,
            "cancellation_dot_count": _EXPECTED_CANCELLATION_DOT_COUNT,
            "cancellation_product_count": (
                _EXPECTED_CANCELLATION_PRODUCT_COUNT
            ),
            "underflow_u1_dot_count": _EXPECTED_UNDERFLOW_DOT_COUNT,
            "underflow_u1_product_count": (
                _EXPECTED_UNDERFLOW_PRODUCT_COUNT
            ),
            "underflow_u2_dot_count": _EXPECTED_UNDERFLOW_DOT_COUNT,
            "underflow_u2_product_count": (
                _EXPECTED_UNDERFLOW_PRODUCT_COUNT
            ),
            "forward_dot_count": _EXPECTED_FORWARD_DOT_COUNT,
            "forward_product_count": _EXPECTED_FORWARD_PRODUCT_COUNT,
            "maximum_dot_count": _MAXIMUM_DOT_COUNT,
            "maximum_product_count": _MAXIMUM_PRODUCT_COUNT,
            "gradient_loss_term_count": (
                _EXPECTED_GRADIENT_LOSS_TERM_COUNT
            ),
            "gradient_diagnostic_spot_selection_count": (
                _EXPECTED_GRADIENT_DIAGNOSTIC_SPOT_COUNT
            ),
            "gradient_component_count_per_order": (
                _EXPECTED_GRADIENT_COMPONENT_COUNT
            ),
            "gradient_and_hvp_component_count": (
                _EXPECTED_GRADIENT_AND_HVP_COMPONENT_COUNT
            ),
        },
        "forward_gates": {
            "ordinary_and_holdout": (
                "zero_standard_gamma_violations_and_zero_d04_violations"
            ),
            "cancellation": (
                "condition_ratio_at_least_2^18_and_zero_d04_violations"
            ),
            "underflow": (
                "diagnostic_only_standard_bound_inapplicable_"
                "cannot_veto_or_rescue"
            ),
            "d04_relative_tolerance": _D04_RELATIVE_TOLERANCE,
            "d04_absolute_tolerance": _D04_ABSOLUTE_TOLERANCE,
            "smallest_float32_normal": _FLOAT32_SMALLEST_NORMAL,
        },
        "gradient_gates": {
            "loss": (
                "full_5_by_output_width_2^-8_sum_lambda_q_times_"
                "z_q_plus_z_q_squared_over_8"
            ),
            "diagnostic_case_selections": (
                "frozen_diagnostic_metadata_only_not_loss_selection"
            ),
            "loss_flat_index": "q_equals_row_times_output_width_plus_output",
            "lambda": (
                "signed_integer_1_through_8_times_2^-3_from_splitmix64"
            ),
            "hvp_direction": (
                "signed_integer_1_through_16_times_2^-8_from_splitmix64"
            ),
            "gradient_relative_tolerance": (
                _GRADIENT_RELATIVE_TOLERANCE
            ),
            "gradient_absolute_tolerance": (
                _GRADIENT_ABSOLUTE_TOLERANCE
            ),
            "hvp_relative_tolerance": _HVP_RELATIVE_TOLERANCE,
            "hvp_absolute_tolerance": _HVP_ABSOLUTE_TOLERANCE,
            "candidate_inference_and_gradient_forward": "bitwise_equal",
            "production_callable": (
                "backend_stable_linear_under_fixed_row_tile_4_context"
            ),
            "active_reference_coverage": (
                "every_input_row_every_weight_output_row_and_every_bias_"
                "component_nonzero_with_aggregate_nonzero_share_at_least_0.95_"
                "for_gradient_and_hvp"
            ),
        },
        "holdout": {
            "access_version": RECURRENT_ACCURACY_SCREEN_HOLDOUT_VERSION,
            "access_ordinal": 1,
            "phase": "candidate_locked_before_selection",
            "tests_must_not_consume_sealed_holdout": True,
            "deciding_report_rejects_test_mode": True,
            "retirement": "negative_or_positive_screen_retires_exact_source",
        },
        "legacy_lane_role": (
            "semantic_and_timing_controls_not_rounding_targets"
        ),
        "forbidden_changes": [
            "inference_only_trainable_forward_shim",
            "hard_coded_action",
            "hidden_state_clamp_or_rounding",
            "public_observation_change",
            "learner_action_change",
        ],
    }
    contract["exact_digest"] = stable_payload_digest(contract)
    return contract


def validate_accuracy_corpus_contract(
    contract: object,
    *,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Reconstruct every table, index, count, and digest from source."""

    if not isinstance(contract, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy corpus contract must be an object"
        )
    expected = accuracy_corpus_contract(seeds=seeds)
    if contract != expected:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy corpus contract reconstruction failed"
        )
    return contract


def build_accuracy_preregistration(
    *,
    source_sha: str,
    checkout_root: str,
    source_module_digests: Mapping[str, str],
    method_document_path: str,
    method_document_sha256: str,
    required_source_paths: Sequence[str] | None = None,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    """Bind the exact corpus to source and the sole candidate identity."""

    source_sha = _validated_hex(
        source_sha,
        length=_SOURCE_SHA_LENGTH,
        field="source SHA",
    )
    checkout = Path(checkout_root)
    if not checkout.is_absolute() or not checkout.is_dir():
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration checkout root must be absolute"
        )
    required_paths = tuple(
        RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS
        if required_source_paths is None
        else required_source_paths
    )
    if required_paths != RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration source paths differ from the frozen order"
        )
    if (
        not isinstance(source_module_digests, Mapping)
        or any(not isinstance(path, str) for path in source_module_digests)
        or set(source_module_digests)
        != set(RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration source-module digests are incomplete"
        )
    validated_digests = {
        path: _validated_hex(
            digest,
            length=64,
            field=f"source-module digest {path}",
        )
        for path, digest in sorted(source_module_digests.items())
    }
    if method_document_path != RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration method-document path is not frozen"
        )
    method_document_sha256 = _validated_hex(
        method_document_sha256,
        length=64,
        field="method-document digest",
    )
    _validate_source_files(
        checkout_root=checkout,
        required_source_paths=required_paths,
        source_module_digests=validated_digests,
    )
    manifest_binding = _validate_method_document_file(
        checkout_root=checkout,
        method_document_path=method_document_path,
        method_document_digest=method_document_sha256,
    )
    contract = accuracy_corpus_contract(seeds=seeds)
    preregistration: dict[str, object] = {
        "schema_version": (
            RECURRENT_ACCURACY_SCREEN_PREREGISTRATION_VERSION
        ),
        "source_sha": source_sha,
        "checkout_root": str(checkout),
        "candidate_identity": RECURRENT_ACCURACY_SCREEN_CANDIDATE,
        "candidate_count": 1,
        "source_module_digests": validated_digests,
        "required_source_paths": list(required_paths),
        "source_module_digest": stable_payload_digest(validated_digests),
        "method_document_path": method_document_path,
        "method_document_sha256": method_document_sha256,
        "selection_manifest_schema_version": manifest_binding[
            "schema_version"
        ],
        "selection_manifest_exact_digest": manifest_binding["exact_digest"],
        "corpus_contract": contract,
        "corpus_contract_digest": contract["exact_digest"],
        "holdout_consumed": False,
        "candidate_selected": False,
        "launch_authorized": False,
        "training_authorized": False,
        "scientific_result": False,
    }
    preregistration["exact_digest"] = stable_payload_digest(preregistration)
    return preregistration


def validate_accuracy_preregistration(
    preregistration: object,
    *,
    seeds: AccuracySeedContract | None = None,
) -> dict[str, object]:
    if not isinstance(preregistration, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration must be an object"
        )
    source_digests = preregistration.get("source_module_digests")
    if not isinstance(source_digests, dict):
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration source digests are malformed"
        )
    expected = build_accuracy_preregistration(
        source_sha=str(preregistration.get("source_sha", "")),
        checkout_root=str(preregistration.get("checkout_root", "")),
        source_module_digests=source_digests,
        required_source_paths=preregistration.get(
            "required_source_paths",
            (),
        ),
        method_document_path=str(
            preregistration.get("method_document_path", "")
        ),
        method_document_sha256=str(
            preregistration.get("method_document_sha256", "")
        ),
        seeds=seeds,
    )
    if preregistration != expected:
        raise RecurrentKernelDevelopmentScreenError(
            "accuracy preregistration reconstruction failed"
        )
    return preregistration


def _validate_holdout_receipt(
    receipt: object,
    *,
    require_deciding: bool,
    expected_seed_contract: AccuracySeedContract,
) -> dict[str, object]:
    expected_seeds = _validated_seed_contract(expected_seed_contract)
    if not isinstance(receipt, dict) or set(receipt) != _HOLDOUT_RECEIPT_KEYS:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt schema is malformed"
        )
    receipt_without_digest = dict(receipt)
    receipt_digest = receipt_without_digest.pop("exact_digest", None)
    if receipt_digest != stable_payload_digest(receipt_without_digest):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt exact digest is invalid"
        )
    if (
        receipt.get("schema_version")
        != RECURRENT_ACCURACY_SCREEN_HOLDOUT_VERSION
        or receipt.get("screen_schema_version")
        != RECURRENT_ACCURACY_SCREEN_SCHEMA_VERSION
        or receipt.get("corpus_version") != RECURRENT_ACCURACY_CORPUS_VERSION
        or receipt.get("selection_manifest_schema_version")
        != RECURRENT_ACCURACY_SELECTION_MANIFEST_SCHEMA_VERSION
        or receipt.get("selection_manifest_exact_digest")
        != RECURRENT_ACCURACY_SELECTION_MANIFEST_EXACT_DIGEST
        or receipt.get("source_clean_detached") is not True
        or receipt.get("access_ordinal") != 1
        or isinstance(receipt.get("access_ordinal"), bool)
        or receipt.get("phase") != "candidate_locked_before_selection"
        or receipt.get("holdout_access_count") != 1
        or isinstance(receipt.get("holdout_access_count"), bool)
        or receipt.get("holdout_consumed") is not True
        or receipt.get("candidate_locked_before_holdout") is not True
        or receipt.get("candidate_identity")
        != RECURRENT_ACCURACY_SCREEN_CANDIDATE
        or receipt.get("retirement")
        != "consumed_for_exact_source_no_retry"
        or not isinstance(receipt.get("test_mode"), bool)
        or not isinstance(receipt.get("pytest_current_test_absent"), bool)
        or receipt.get("test_mode")
        is not expected_seeds.synthetic_test_contract
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt lifecycle or contract binding is invalid"
        )
    if require_deciding and (
        receipt["test_mode"] is not False
        or receipt["pytest_current_test_absent"] is not True
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt cannot support a deciding report"
        )
    source_sha = _validated_hex(
        receipt.get("source_sha"),
        length=40,
        field="receipt source SHA",
    )
    checkout_root = Path(
        _validated_absolute_path(
            receipt.get("checkout_root"),
            field="receipt checkout root",
        )
    )
    required_source_paths = receipt.get("required_source_paths")
    if (
        not isinstance(required_source_paths, list)
        or tuple(required_source_paths)
        != RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt source paths differ from the frozen order"
        )
    source_module_digests = receipt.get("source_module_digests")
    if (
        not isinstance(source_module_digests, dict)
        or any(not isinstance(path, str) for path in source_module_digests)
        or set(source_module_digests)
        != set(RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt source-module digests are malformed"
        )
    validated_source_digests = {
        path: _validated_hex(
            source_module_digests[path],
            length=64,
            field=f"receipt source-module digest {path}",
        )
        for path in RECURRENT_ACCURACY_REQUIRED_SOURCE_PATHS
    }
    source_module_digest = _validated_hex(
        receipt.get("source_module_digest"),
        length=64,
        field="receipt source-module aggregate digest",
    )
    if source_module_digest != stable_payload_digest(
        validated_source_digests
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt source-module aggregate digest is invalid"
        )
    method_document_path = receipt.get("method_document_path")
    if method_document_path != RECURRENT_ACCURACY_METHOD_DOCUMENT_PATH:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt method-document path is not frozen"
        )
    method_document_digest = _validated_hex(
        receipt.get("method_document_digest"),
        length=64,
        field="receipt method-document digest",
    )
    _validate_source_files(
        checkout_root=checkout_root,
        required_source_paths=required_source_paths,
        source_module_digests=validated_source_digests,
    )
    _validate_method_document_file(
        checkout_root=checkout_root,
        method_document_path=method_document_path,
        method_document_digest=method_document_digest,
    )
    parent_process_id = _validated_positive_integer(
        receipt.get("parent_process_id"),
        field="receipt parent process ID",
    )
    child_process_id = _validated_positive_integer(
        receipt.get("child_process_id"),
        field="receipt child process ID",
    )
    parent_process_start_identity = _validated_nonempty_string(
        receipt.get("parent_process_start_identity"),
        field="receipt parent process start identity",
    )
    child_process_start_identity = _validated_nonempty_string(
        receipt.get("child_process_start_identity"),
        field="receipt child process start identity",
    )
    child_nonce = _validated_hex(
        receipt.get("child_nonce"),
        length=64,
        field="receipt child nonce",
    )
    report_path = Path(
        _validated_absolute_path(
            receipt.get("report_path"),
            field="receipt report path",
        )
    )
    preregistration_digest = _validated_hex(
        receipt.get("preregistration_digest"),
        length=64,
        field="receipt preregistration digest",
    )
    preholdout_evidence_digest = _validated_hex(
        receipt.get("preholdout_evidence_digest"),
        length=64,
        field="receipt pre-holdout evidence digest",
    )
    closing_baseline_evidence_digest = _validated_hex(
        receipt.get("closing_baseline_evidence_digest"),
        length=64,
        field="receipt closing-baseline evidence digest",
    )
    capability_commitment_sha256 = _validated_hex(
        receipt.get("capability_commitment_sha256"),
        length=64,
        field="receipt capability commitment",
    )
    seed_digest = _validated_hex(
        receipt.get("seed_digest"),
        length=64,
        field="receipt seed digest",
    )
    if seed_digest != _seed_digest(expected_seeds.holdout, "holdout"):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt seed digest differs from expected contract"
        )
    source_marker_path = Path(
        _validated_absolute_path(
            receipt.get("source_retirement_marker_path"),
            field="receipt source-retirement-marker path",
        )
    )
    holdout_marker_path = Path(
        _validated_absolute_path(
            receipt.get("holdout_retirement_marker_path"),
            field="receipt holdout-retirement-marker path",
        )
    )
    if (
        source_marker_path.parent != report_path.parent
        or source_marker_path.name
        != RECURRENT_ACCURACY_SOURCE_RETIREMENT_MARKER_NAME
        or holdout_marker_path.parent != report_path.parent
        or holdout_marker_path.name
        != RECURRENT_ACCURACY_HOLDOUT_RETIREMENT_MARKER_NAME
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt retirement-marker paths are not canonical"
        )
    source_marker_sha256 = _validated_hex(
        receipt.get("source_retirement_marker_sha256"),
        length=64,
        field="receipt source-retirement-marker SHA256",
    )
    source_marker_exact_digest = _validated_hex(
        receipt.get("source_retirement_marker_exact_digest"),
        length=64,
        field="receipt source-retirement-marker exact digest",
    )
    expected_source_marker = source_retirement_marker(
        source_sha=source_sha,
        source_module_digest=source_module_digest,
        method_document_digest=method_document_digest,
        preregistration_digest=preregistration_digest,
        report_path=str(report_path),
        parent_process_id=parent_process_id,
        parent_process_start_identity=parent_process_start_identity,
        capability_commitment_sha256=capability_commitment_sha256,
    )
    if expected_source_marker["exact_digest"] != source_marker_exact_digest:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt source-retirement digest is invalid"
        )
    _validate_retirement_marker_file(
        path=source_marker_path,
        file_sha256=source_marker_sha256,
        expected_marker=expected_source_marker,
        label="source",
    )
    holdout_marker_sha256 = _validated_hex(
        receipt.get("holdout_retirement_marker_sha256"),
        length=64,
        field="receipt holdout-retirement-marker SHA256",
    )
    holdout_marker_exact_digest = _validated_hex(
        receipt.get("holdout_retirement_marker_exact_digest"),
        length=64,
        field="receipt holdout-retirement-marker exact digest",
    )
    expected_holdout_marker = holdout_retirement_marker(
        source_retirement_marker_sha256=source_marker_sha256,
        source_retirement_marker_exact_digest=source_marker_exact_digest,
        preholdout_evidence_digest=preholdout_evidence_digest,
        closing_baseline_evidence_digest=closing_baseline_evidence_digest,
        parent_process_id=parent_process_id,
        parent_process_start_identity=parent_process_start_identity,
        child_process_id=child_process_id,
        child_process_start_identity=child_process_start_identity,
        child_nonce=child_nonce,
        capability_commitment_sha256=capability_commitment_sha256,
    )
    if expected_holdout_marker["exact_digest"] != holdout_marker_exact_digest:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout receipt holdout-retirement digest is invalid"
        )
    _validate_retirement_marker_file(
        path=holdout_marker_path,
        file_sha256=holdout_marker_sha256,
        expected_marker=expected_holdout_marker,
        label="holdout",
    )
    return receipt


def validate_deciding_holdout_report(
    report: object,
    *,
    expected_seed_contract: AccuracySeedContract,
) -> dict[str, object]:
    """Fail closed on synthetic, repeated, or digest-tampered holdout reports."""

    if not isinstance(report, dict) or set(report) != _HOLDOUT_REPORT_KEYS:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout report schema is malformed"
        )
    candidate = dict(report)
    exact_digest = candidate.pop("exact_digest", None)
    if exact_digest != stable_payload_digest(candidate):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout report exact digest is invalid"
        )
    _validate_holdout_receipt(
        candidate.get("holdout_receipt"),
        require_deciding=True,
        expected_seed_contract=expected_seed_contract,
    )
    if (
        report.get("schema_version") != RECURRENT_ACCURACY_CORPUS_VERSION
        or report.get("development_only") is not True
        or report.get("deciding") is not True
        or report.get("candidate_identity")
        != RECURRENT_ACCURACY_SCREEN_CANDIDATE
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout report cannot support a deciding result"
        )
    groups = report.get("groups")
    if not isinstance(groups, dict) or set(groups) != {"holdout"}:
        raise RecurrentKernelDevelopmentScreenError(
            "holdout report group schema is malformed"
        )
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict) or (
        aggregate.get("dot_count") != _EXPECTED_HOLDOUT_DOT_COUNT
        or aggregate.get("product_count")
        != _EXPECTED_HOLDOUT_PRODUCT_COUNT
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "holdout report counts are malformed"
        )
    return report


def _probe_from_contract(contract: object) -> LinearProbeCase:
    expected_keys = {
        "case_id",
        "table",
        "role",
        "input_features",
        "output_features",
        "active_rows",
        "physical_rows",
        "selected_rows",
        "selected_outputs",
        "dot_count",
        "product_count",
    }
    if not isinstance(contract, dict) or set(contract) != expected_keys:
        raise RecurrentKernelDevelopmentScreenError(
            "serialized holdout case contract is malformed"
        )
    for field in ("case_id", "table", "role"):
        _validated_nonempty_string(contract[field], field=f"case {field}")
    integer_fields = (
        "input_features",
        "output_features",
        "active_rows",
        "physical_rows",
        "dot_count",
        "product_count",
    )
    if any(
        isinstance(contract[field], bool)
        or not isinstance(contract[field], int)
        or contract[field] <= 0
        for field in integer_fields
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized holdout case integer is malformed"
        )
    selected_rows = contract["selected_rows"]
    selected_outputs = contract["selected_outputs"]
    if (
        not isinstance(selected_rows, list)
        or not selected_rows
        or not isinstance(selected_outputs, list)
        or not selected_outputs
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (*selected_rows, *selected_outputs)
        )
        or selected_rows != sorted(set(selected_rows))
        or selected_outputs != sorted(set(selected_outputs))
        or selected_rows[-1] >= contract["active_rows"]
        or selected_outputs[-1] >= contract["output_features"]
        or contract["physical_rows"] < contract["active_rows"]
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "serialized holdout case selection is malformed"
        )
    probe = LinearProbeCase(
        case_id=contract["case_id"],
        table=contract["table"],
        role=contract["role"],
        input_features=contract["input_features"],
        output_features=contract["output_features"],
        active_rows=contract["active_rows"],
        physical_rows=contract["physical_rows"],
        selected_rows=tuple(selected_rows),
        selected_outputs=tuple(selected_outputs),
    )
    if probe.as_contract() != contract:
        raise RecurrentKernelDevelopmentScreenError(
            "serialized holdout case contract has unknown or derived drift"
        )
    return probe


def validate_holdout_report_offline(
    report: object,
    *,
    expected_seed_contract: AccuracySeedContract,
    expected_case_contracts: Sequence[Mapping[str, object]],
    allow_synthetic_test: bool = False,
) -> dict[str, object]:
    """Recompute holdout metrics from serialized values, never the candidate."""

    expected_seeds = _validated_seed_contract(expected_seed_contract)
    if type(allow_synthetic_test) is not bool:
        raise RecurrentKernelDevelopmentScreenError(
            "offline synthetic-test allowance must be boolean"
        )
    if not isinstance(report, dict) or set(report) != _HOLDOUT_REPORT_KEYS:
        raise RecurrentKernelDevelopmentScreenError(
            "offline holdout report schema is malformed"
        )
    report_without_digest = dict(report)
    exact_digest = report_without_digest.pop("exact_digest", None)
    if exact_digest != stable_payload_digest(report_without_digest):
        raise RecurrentKernelDevelopmentScreenError(
            "offline holdout report exact digest is invalid"
        )
    receipt = _validate_holdout_receipt(
        report.get("holdout_receipt"),
        require_deciding=not allow_synthetic_test,
        expected_seed_contract=expected_seeds,
    )
    if allow_synthetic_test:
        if receipt.get("test_mode") is not True:
            raise RecurrentKernelDevelopmentScreenError(
                "synthetic offline holdout must carry test mode"
            )
    else:
        validate_deciding_holdout_report(
            report,
            expected_seed_contract=expected_seeds,
        )
    if (
        report.get("schema_version") != RECURRENT_ACCURACY_CORPUS_VERSION
        or report.get("development_only") is not True
        or report.get("deciding") is not (
            receipt.get("test_mode") is False
        )
        or report.get("candidate_identity")
        != RECURRENT_ACCURACY_SCREEN_CANDIDATE
    ):
        raise RecurrentKernelDevelopmentScreenError(
            "offline holdout report lifecycle is invalid"
        )
    groups = report.get("groups")
    if not isinstance(groups, dict) or set(groups) != {"holdout"}:
        raise RecurrentKernelDevelopmentScreenError(
            "offline holdout group schema is malformed"
        )
    cases = groups["holdout"]
    expected = [dict(contract) for contract in expected_case_contracts]
    frozen_expected = [
        case.as_contract()
        for case in operational_probe_cases(expected_seeds.holdout)
    ]
    if expected != frozen_expected:
        raise RecurrentKernelDevelopmentScreenError(
            "offline expected case table differs from the seed-bound corpus"
        )
    if not isinstance(cases, list) or [
        case.get("case") if isinstance(case, dict) else None for case in cases
    ] != expected:
        raise RecurrentKernelDevelopmentScreenError(
            "offline holdout case table is invalid"
        )
    reconstructed_cases: list[dict[str, object]] = []
    for serialized_case in cases:
        if (
            not isinstance(serialized_case, dict)
            or set(serialized_case) != _HOLDOUT_CASE_KEYS
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout case schema is malformed"
            )
        case_without_digest = dict(serialized_case)
        case_digest = case_without_digest.pop("exact_digest", None)
        if case_digest != stable_payload_digest(case_without_digest):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout case digest is invalid"
            )
        probe = _probe_from_contract(serialized_case.get("case"))
        if serialized_case.get("family") != "composite":
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout case family is not composite"
            )
        expected_generated = _materialize_normal_case(
            probe,
            seed=expected_seeds.holdout,
            family="composite",
        )
        for digest_field in (
            "full_component_exact_digests",
            "selected_component_exact_digests",
        ):
            digests = serialized_case.get(digest_field)
            if (
                not isinstance(digests, dict)
                or set(digests) != _FULL_COMPONENT_DIGEST_KEYS
            ):
                raise RecurrentKernelDevelopmentScreenError(
                    f"offline holdout {digest_field} schema is malformed"
                )
            for name, digest in digests.items():
                _validated_hex(
                    digest,
                    length=64,
                    field=f"offline holdout {digest_field} {name}",
                )
        for evidence_field in ("metrics", "per_dot_classification"):
            nested = serialized_case.get(evidence_field)
            if not isinstance(nested, dict):
                raise RecurrentKernelDevelopmentScreenError(
                    f"offline holdout {evidence_field} is malformed"
                )
            nested_without_digest = dict(nested)
            nested_digest = nested_without_digest.pop("exact_digest", None)
            if nested_digest != stable_payload_digest(nested_without_digest):
                raise RecurrentKernelDevelopmentScreenError(
                    f"offline holdout {evidence_field} digest is invalid"
                )
        raw = serialized_case.get("serialized_selected_values")
        if not isinstance(raw, dict) or set(raw) != {
            "input",
            "weight",
            "bias",
            "candidate_output",
            "full_candidate_output",
            "oracle_output",
        }:
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout serialized values are incomplete"
            )
        inputs = _deserialize_float32_tensor(raw["input"])
        weight = _deserialize_float32_tensor(raw["weight"])
        bias = _deserialize_float32_tensor(raw["bias"])
        candidate_output = _deserialize_float32_tensor(
            raw["candidate_output"]
        )
        full_candidate_output = _deserialize_float32_tensor(
            raw["full_candidate_output"]
        )
        serialized_oracle = _deserialize_float32_tensor(raw["oracle_output"])
        expected_shapes = (
            (len(probe.selected_rows), probe.input_features),
            (len(probe.selected_outputs), probe.input_features),
            (len(probe.selected_outputs),),
            (len(probe.selected_rows), len(probe.selected_outputs)),
            (probe.physical_rows, probe.output_features),
        )
        if (
            tuple(inputs.shape),
            tuple(weight.shape),
            tuple(bias.shape),
            tuple(candidate_output.shape),
            tuple(full_candidate_output.shape),
        ) != expected_shapes:
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout serialized tensor shape is invalid"
            )
        expected_inputs, expected_weight, expected_bias = (
            _selected_components(expected_generated)
        )
        if (
            not torch.equal(inputs, expected_inputs)
            or not torch.equal(weight, expected_weight)
            or not torch.equal(bias, expected_bias)
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout seed-bound component reconstruction failed"
            )
        selected_full_candidate_output = _selected_candidate_output(
            full_candidate_output,
            probe,
        )
        if not torch.equal(
            candidate_output,
            selected_full_candidate_output,
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout full and selected candidate outputs differ"
            )
        oracle, _ = _independent_float64_linear_oracle(inputs, weight, bias)
        if not torch.equal(oracle, serialized_oracle):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout oracle reconstruction failed"
            )
        selected_digests = {
            "input": _float32_tensor_exact_digest(expected_inputs),
            "weight": _float32_tensor_exact_digest(expected_weight),
            "bias": _float32_tensor_exact_digest(expected_bias),
            "candidate_output": _float32_tensor_exact_digest(
                selected_full_candidate_output
            ),
        }
        if serialized_case.get("selected_component_exact_digests") != (
            selected_digests
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout selected component digest is invalid"
            )
        full_digests = {
            "input": _float32_tensor_exact_digest(expected_generated.inputs),
            "weight": _float32_tensor_exact_digest(expected_generated.weight),
            "bias": _float32_tensor_exact_digest(expected_generated.bias),
            "candidate_output": _float32_tensor_exact_digest(
                full_candidate_output
            ),
        }
        if serialized_case.get("full_component_exact_digests") != (
            full_digests
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout full component digest is invalid"
            )
        classification = _per_dot_classification(
            probe=probe,
            inputs=inputs,
            weight=weight,
            bias=bias,
            selected_candidate_output=candidate_output,
        )
        metrics = _authenticated_oracle_metrics(
            candidate_output,
            inputs,
            weight,
            bias,
            classification=classification,
        )
        if (
            serialized_case.get("metrics") != metrics
            or serialized_case.get("per_dot_classification")
            != classification
        ):
            raise RecurrentKernelDevelopmentScreenError(
                "offline holdout metric reconstruction failed"
            )
        reconstructed_cases.append(serialized_case)
    aggregate = _aggregate_forward_evidence(
        {"holdout": reconstructed_cases}
    )
    if report.get("aggregate") != aggregate:
        raise RecurrentKernelDevelopmentScreenError(
            "offline holdout aggregate reconstruction failed"
        )
    return report


def source_file_sha256(path: str | Path) -> str:
    source_path = Path(path)
    if not source_path.is_file():
        raise RecurrentKernelDevelopmentScreenError(
            "source module path is not a file"
        )
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _validated_hex(value: object, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentKernelDevelopmentScreenError(
            f"{field} must be lowercase hexadecimal"
        )
    return value
