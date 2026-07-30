from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from typing import Any


RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION = "mind_recurrent_controller_genome_v1"
RECURRENT_CONTROLLER_GENOME_DIGEST_POLICY = (
    "sha256_canonical_json_without_genome_sha256_v1"
)
RECURRENT_CONTROLLER_INHERITANCE_CONTRACT_VERSION = (
    "mind_recurrent_controller_inheritance_v1"
)
RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION = (
    "mind_recurrent_controller_film_development_v1"
)
RECURRENT_CONTROLLER_GENOME_SIZE = 16
RECURRENT_CONTROLLER_GENOME_MIN = -1.0
RECURRENT_CONTROLLER_GENOME_MAX = 1.0
RECURRENT_CONTROLLER_GENOME_FLOAT_DECIMALS = 8
RECURRENT_CONTROLLER_FOUNDER_ABS_LIMIT = 0.35
RECURRENT_CONTROLLER_DEFAULT_MUTATION_RATE = 0.20
RECURRENT_CONTROLLER_DEFAULT_MUTATION_MAX_STEP = 0.08
RECURRENT_CONTROLLER_MAX_MUTATION_STEP = 0.50
RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT = 0.25
RECURRENT_CONTROLLER_FILM_BIAS_LIMIT = 0.25
RECURRENT_CONTROLLER_VECTORIZED_DEVELOPMENT_ABSOLUTE_TOLERANCE = 1.0e-6
RECURRENT_CONTROLLER_MAX_HIDDEN_DIMENSION = 4096

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "values",
        "genome_sha256",
    }
)
_CONTRACT = {
    "digest_policy": RECURRENT_CONTROLLER_GENOME_DIGEST_POLICY,
    "inheritance_contract_version": (RECURRENT_CONTROLLER_INHERITANCE_CONTRACT_VERSION),
    "development_map_version": RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION,
    "genome_size": RECURRENT_CONTROLLER_GENOME_SIZE,
    "value_bounds": [
        RECURRENT_CONTROLLER_GENOME_MIN,
        RECURRENT_CONTROLLER_GENOME_MAX,
    ],
    "float_decimals": RECURRENT_CONTROLLER_GENOME_FLOAT_DECIMALS,
    "inherited_payload": "genome_values_only",
}


class RecurrentGenomeError(ValueError):
    """Raised when a recurrent controller genome fails closed."""


@dataclass(frozen=True, slots=True)
class RecurrentControllerGenome:
    """A fixed-size inherited controller code with no acquired runtime state."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_values(self.values)

    @property
    def sha256(self) -> str:
        return recurrent_genome_sha256(self)

    def to_artifact(self) -> dict[str, object]:
        return recurrent_genome_artifact(self)


@dataclass(frozen=True, slots=True)
class RecurrentGenomeMutationConfig:
    rate: float = RECURRENT_CONTROLLER_DEFAULT_MUTATION_RATE
    max_step: float = RECURRENT_CONTROLLER_DEFAULT_MUTATION_MAX_STEP

    def __post_init__(self) -> None:
        rate = _strict_finite_float(self.rate, field="mutation rate")
        max_step = _strict_finite_float(
            self.max_step,
            field="mutation max_step",
        )
        if not 0.0 <= rate <= 1.0:
            raise RecurrentGenomeError("mutation rate must be in [0, 1]")
        if not 0.0 <= max_step <= RECURRENT_CONTROLLER_MAX_MUTATION_STEP:
            raise RecurrentGenomeError(
                "mutation max_step must be in "
                f"[0, {RECURRENT_CONTROLLER_MAX_MUTATION_STEP}]"
            )
        if rate > 0.0 and max_step == 0.0:
            raise RecurrentGenomeError(
                "mutation max_step must be positive when mutation rate is positive"
            )


@dataclass(frozen=True, slots=True)
class FilmDevelopment:
    contract_version: str
    genome_sha256: str
    hidden_dimension: int
    scale: tuple[float, ...]
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.contract_version != RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION:
            raise RecurrentGenomeError("FiLM development contract is stale")
        _validate_sha256(self.genome_sha256, field="genome_sha256")
        _validate_hidden_dimension(self.hidden_dimension)
        if len(self.scale) != self.hidden_dimension:
            raise RecurrentGenomeError("FiLM scale has invalid shape")
        if len(self.bias) != self.hidden_dimension:
            raise RecurrentGenomeError("FiLM bias has invalid shape")
        for index, value in enumerate(self.scale):
            parsed = _strict_finite_float(value, field=f"scale[{index}]")
            if not (
                1.0 - RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT
                <= parsed
                <= 1.0 + RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT
            ):
                raise RecurrentGenomeError(f"scale[{index}] is out of bounds")
        for index, value in enumerate(self.bias):
            parsed = _strict_finite_float(value, field=f"bias[{index}]")
            if not (
                -RECURRENT_CONTROLLER_FILM_BIAS_LIMIT
                <= parsed
                <= RECURRENT_CONTROLLER_FILM_BIAS_LIMIT
            ):
                raise RecurrentGenomeError(f"bias[{index}] is out of bounds")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "genome_sha256": self.genome_sha256,
            "hidden_dimension": self.hidden_dimension,
            "scale": list(self.scale),
            "bias": list(self.bias),
        }


@dataclass(frozen=True, slots=True)
class FilmDevelopmentCoefficients:
    """Immutable SHA-derived dense projections shared by scalar and tensor paths."""

    contract_version: str
    hidden_dimension: int
    genome_size: int
    scale: tuple[tuple[float, ...], ...]
    bias: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.contract_version != RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION:
            raise RecurrentGenomeError("FiLM coefficient contract is stale")
        _validate_hidden_dimension(self.hidden_dimension)
        if self.genome_size != RECURRENT_CONTROLLER_GENOME_SIZE:
            raise RecurrentGenomeError("FiLM coefficient genome_size is stale")
        for channel_name, channel in (("scale", self.scale), ("bias", self.bias)):
            if len(channel) != self.hidden_dimension:
                raise RecurrentGenomeError(
                    f"FiLM {channel_name} coefficients have invalid hidden shape"
                )
            for hidden_index, row in enumerate(channel):
                if len(row) != RECURRENT_CONTROLLER_GENOME_SIZE:
                    raise RecurrentGenomeError(
                        f"FiLM {channel_name}[{hidden_index}] has invalid genome shape"
                    )
                for locus, raw_value in enumerate(row):
                    value = _strict_finite_float(
                        raw_value,
                        field=f"{channel_name}[{hidden_index}][{locus}]",
                    )
                    if not -1.0 <= value <= 1.0:
                        raise RecurrentGenomeError(
                            f"FiLM {channel_name}[{hidden_index}][{locus}] "
                            "is out of bounds"
                        )


@dataclass(frozen=True, slots=True)
class GenomeSwapEffect:
    left_genome_sha256: str
    right_genome_sha256: str
    left_output: tuple[float, ...]
    right_output: tuple[float, ...]
    max_absolute_difference: float
    mean_absolute_difference: float

    @property
    def changed(self) -> bool:
        return self.max_absolute_difference > 0.0


def recurrent_genome_contract() -> dict[str, object]:
    """Return the exact versioned genome and inheritance contract."""
    return _copy_json_value(_CONTRACT)


def recurrent_genome_development_contract(
    *,
    hidden_dimension: int,
) -> dict[str, object]:
    """Return the versioned, requested-shape FiLM development contract."""
    parsed_hidden_dimension = _validate_hidden_dimension(hidden_dimension)
    return {
        "schema_version": RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION,
        "source_genome_schema_version": (RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION),
        "genome_size": RECURRENT_CONTROLLER_GENOME_SIZE,
        "hidden_dimension": parsed_hidden_dimension,
        "projection": "sha256_dense_signed_projection_v1",
        "transform": "bounded_softsign_featurewise_affine_v1",
        "scale_bounds": [
            1.0 - RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT,
            1.0 + RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT,
        ],
        "bias_bounds": [
            -RECURRENT_CONTROLLER_FILM_BIAS_LIMIT,
            RECURRENT_CONTROLLER_FILM_BIAS_LIMIT,
        ],
        "birth_state_policy": "external_zero_initialization_required",
        "vectorized_development_absolute_tolerance": (
            RECURRENT_CONTROLLER_VECTORIZED_DEVELOPMENT_ABSOLUTE_TOLERANCE
        ),
    }


def recurrent_genome_development_coefficients(
    *,
    hidden_dimension: int,
) -> FilmDevelopmentCoefficients:
    """Return immutable fixed coefficients for scalar or vectorized FiLM.

    Coefficients depend only on the versioned development-map contract and
    requested hidden width. They may therefore be materialized once as model
    buffers; no genome-row identity or hashing is needed during inference.
    """

    parsed_hidden_dimension = _validate_hidden_dimension(hidden_dimension)
    return _cached_recurrent_genome_development_coefficients(parsed_hidden_dimension)


def founder_recurrent_genome(*, seed: int) -> RecurrentControllerGenome:
    """Create a deterministic, near-neutral founder without recording identity."""
    parsed_seed = _validate_seed(seed)
    values = tuple(
        _quantize(
            RECURRENT_CONTROLLER_FOUNDER_ABS_LIMIT
            * (2.0 * _stream_unit(parsed_seed, "founder", index, 0) - 1.0)
        )
        for index in range(RECURRENT_CONTROLLER_GENOME_SIZE)
    )
    return RecurrentControllerGenome(values=values)


def zero_recurrent_genome() -> RecurrentControllerGenome:
    """Return the preregisterable genotype-zeroed causal control."""
    return RecurrentControllerGenome(values=(0.0,) * RECURRENT_CONTROLLER_GENOME_SIZE)


def inherit_asexual_recurrent_genome(
    parent: RecurrentControllerGenome,
    *,
    seed: int,
    mutation: RecurrentGenomeMutationConfig | None = None,
) -> RecurrentControllerGenome:
    """Clone and deterministically mutate only the compact genome values."""
    parsed_parent = _require_genome(parent, field="parent")
    parsed_seed = _validate_seed(seed)
    resolved_mutation = mutation or RecurrentGenomeMutationConfig()
    return _mutate_values(
        parsed_parent.values,
        seed=parsed_seed,
        domain=f"asexual:{parsed_parent.sha256}",
        mutation=resolved_mutation,
    )


def recombine_recurrent_genomes(
    left: RecurrentControllerGenome,
    right: RecurrentControllerGenome,
    *,
    seed: int,
    mutation: RecurrentGenomeMutationConfig | None = None,
) -> RecurrentControllerGenome:
    """Uniformly recombine two genomes, then apply deterministic bounded mutation."""
    parsed_left = _require_genome(left, field="left")
    parsed_right = _require_genome(right, field="right")
    parsed_seed = _validate_seed(seed)
    resolved_mutation = mutation or RecurrentGenomeMutationConfig()
    ordered = sorted(
        (parsed_left, parsed_right),
        key=lambda genome: (genome.sha256, genome.values),
    )
    domain = f"two_parent:{ordered[0].sha256}:{ordered[1].sha256}"
    crossed = tuple(
        ordered[0 if _stream_unit(parsed_seed, domain, index, 0) < 0.5 else 1].values[
            index
        ]
        for index in range(RECURRENT_CONTROLLER_GENOME_SIZE)
    )
    return _mutate_values(
        crossed,
        seed=parsed_seed,
        domain=f"{domain}:mutation",
        mutation=resolved_mutation,
    )


def recurrent_genome_artifact(
    genome: RecurrentControllerGenome,
) -> dict[str, object]:
    parsed = _require_genome(genome, field="genome")
    payload = _genome_payload(parsed)
    return {
        **payload,
        "genome_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }


def recurrent_genome_sha256(genome: RecurrentControllerGenome) -> str:
    parsed = _require_genome(genome, field="genome")
    return hashlib.sha256(_canonical_json_bytes(_genome_payload(parsed))).hexdigest()


def serialize_recurrent_genome(genome: RecurrentControllerGenome) -> bytes:
    """Serialize the complete artifact as exact compact canonical JSON bytes."""
    return _canonical_json_bytes(recurrent_genome_artifact(genome))


def validate_recurrent_genome_artifact(
    artifact: Mapping[str, object],
) -> RecurrentControllerGenome:
    if not isinstance(artifact, Mapping):
        raise RecurrentGenomeError("recurrent genome artifact must be an object")
    _require_exact_keys(artifact, _TOP_LEVEL_KEYS, field="artifact")
    if artifact.get("schema_version") != RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION:
        raise RecurrentGenomeError("recurrent genome schema_version is stale")
    contract = artifact.get("contract")
    if contract != _CONTRACT:
        raise RecurrentGenomeError("recurrent genome contract is missing or stale")
    raw_values = artifact.get("values")
    if not isinstance(raw_values, list):
        raise RecurrentGenomeError("recurrent genome values must be a list")
    genome = RecurrentControllerGenome(values=tuple(raw_values))
    observed_sha256 = artifact.get("genome_sha256")
    _validate_sha256(observed_sha256, field="genome_sha256")
    if observed_sha256 != recurrent_genome_sha256(genome):
        raise RecurrentGenomeError("recurrent genome SHA256 mismatch")
    return genome


def deserialize_recurrent_genome(
    serialized: str | bytes,
    *,
    require_canonical: bool = True,
) -> RecurrentControllerGenome:
    if isinstance(serialized, bytes):
        try:
            text = serialized.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecurrentGenomeError(
                "recurrent genome serialization is not UTF-8"
            ) from exc
    elif isinstance(serialized, str):
        text = serialized
    else:
        raise RecurrentGenomeError(
            "recurrent genome serialization must be str or bytes"
        )
    try:
        artifact = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RecurrentGenomeError(
            "recurrent genome serialization is not valid JSON"
        ) from exc
    if not isinstance(artifact, dict):
        raise RecurrentGenomeError("recurrent genome artifact must be an object")
    genome = validate_recurrent_genome_artifact(artifact)
    if require_canonical and text.encode("utf-8") != _canonical_json_bytes(artifact):
        raise RecurrentGenomeError(
            "recurrent genome serialization is not canonical JSON"
        )
    return genome


def develop_recurrent_genome(
    genome: RecurrentControllerGenome,
    *,
    hidden_dimension: int,
) -> FilmDevelopment:
    """Develop a compact genome into bounded FiLM scale and bias vectors."""
    parsed = _require_genome(genome, field="genome")
    parsed_hidden_dimension = _validate_hidden_dimension(hidden_dimension)
    coefficients = recurrent_genome_development_coefficients(
        hidden_dimension=parsed_hidden_dimension
    )
    normalization = math.sqrt(RECURRENT_CONTROLLER_GENOME_SIZE)
    scale: list[float] = []
    bias: list[float] = []
    for hidden_index in range(parsed_hidden_dimension):
        scale_projection = (
            sum(
                coefficient * value
                for coefficient, value in zip(
                    coefficients.scale[hidden_index],
                    parsed.values,
                    strict=True,
                )
            )
            / normalization
        )
        bias_projection = (
            sum(
                coefficient * value
                for coefficient, value in zip(
                    coefficients.bias[hidden_index],
                    parsed.values,
                    strict=True,
                )
            )
            / normalization
        )
        scale.append(
            _quantize(
                1.0
                + RECURRENT_CONTROLLER_FILM_SCALE_DELTA_LIMIT
                * _softsign(scale_projection)
            )
        )
        bias.append(
            _quantize(RECURRENT_CONTROLLER_FILM_BIAS_LIMIT * _softsign(bias_projection))
        )
    return FilmDevelopment(
        contract_version=RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION,
        genome_sha256=parsed.sha256,
        hidden_dimension=parsed_hidden_dimension,
        scale=tuple(scale),
        bias=tuple(bias),
    )


def apply_film(
    activations: Sequence[object],
    development: FilmDevelopment,
) -> tuple[float, ...]:
    """Apply developed FiLM parameters to a fixed activation vector."""
    if not isinstance(development, FilmDevelopment):
        raise RecurrentGenomeError("development must be FilmDevelopment")
    if isinstance(activations, (str, bytes)) or not isinstance(activations, Sequence):
        raise RecurrentGenomeError("activations must be a numeric sequence")
    if len(activations) != development.hidden_dimension:
        raise RecurrentGenomeError(
            "activations length does not match FiLM hidden_dimension"
        )
    outputs: list[float] = []
    for index, raw_value in enumerate(activations):
        value = _strict_finite_float(raw_value, field=f"activations[{index}]")
        outputs.append(
            _quantize(development.scale[index] * value + development.bias[index])
        )
    return tuple(outputs)


def bounded_recurrent_genome_perturbation(
    genome: RecurrentControllerGenome,
    *,
    locus: int,
    delta: float,
    max_absolute_delta: float = RECURRENT_CONTROLLER_DEFAULT_MUTATION_MAX_STEP,
) -> RecurrentControllerGenome:
    """Change one locus under an explicit bound for causal perturbation gates."""
    parsed = _require_genome(genome, field="genome")
    if isinstance(locus, bool) or not isinstance(locus, int):
        raise RecurrentGenomeError("perturbation locus must be an integer")
    if not 0 <= locus < RECURRENT_CONTROLLER_GENOME_SIZE:
        raise RecurrentGenomeError("perturbation locus is out of range")
    parsed_delta = _strict_finite_float(delta, field="perturbation delta")
    parsed_limit = _strict_finite_float(
        max_absolute_delta,
        field="perturbation max_absolute_delta",
    )
    if not 0.0 < parsed_limit <= RECURRENT_CONTROLLER_MAX_MUTATION_STEP:
        raise RecurrentGenomeError(
            "perturbation max_absolute_delta must be positive and bounded"
        )
    if parsed_delta == 0.0 or abs(parsed_delta) > parsed_limit:
        raise RecurrentGenomeError(
            "perturbation delta must be non-zero and within max_absolute_delta"
        )
    values = list(parsed.values)
    values[locus] = _quantize(_clamp_genome_value(values[locus] + parsed_delta))
    if values[locus] == parsed.values[locus]:
        raise RecurrentGenomeError(
            "bounded perturbation was erased by clipping or quantization"
        )
    return RecurrentControllerGenome(values=tuple(values))


def evaluate_recurrent_genome_swap(
    activations: Sequence[object],
    *,
    left: RecurrentControllerGenome,
    right: RecurrentControllerGenome,
) -> GenomeSwapEffect:
    """Hold activations fixed and report the causal FiLM change from a genome swap."""
    if isinstance(activations, (str, bytes)) or not isinstance(activations, Sequence):
        raise RecurrentGenomeError("activations must be a numeric sequence")
    hidden_dimension = _validate_hidden_dimension(len(activations))
    left_development = develop_recurrent_genome(
        left,
        hidden_dimension=hidden_dimension,
    )
    right_development = develop_recurrent_genome(
        right,
        hidden_dimension=hidden_dimension,
    )
    left_output = apply_film(activations, left_development)
    right_output = apply_film(activations, right_development)
    differences = tuple(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            left_output,
            right_output,
            strict=True,
        )
    )
    return GenomeSwapEffect(
        left_genome_sha256=left_development.genome_sha256,
        right_genome_sha256=right_development.genome_sha256,
        left_output=left_output,
        right_output=right_output,
        max_absolute_difference=_quantize(max(differences)),
        mean_absolute_difference=_quantize(sum(differences) / len(differences)),
    )


def recurrent_genome_mean_absolute_distance(
    left: RecurrentControllerGenome,
    right: RecurrentControllerGenome,
) -> float:
    parsed_left = _require_genome(left, field="left")
    parsed_right = _require_genome(right, field="right")
    return _quantize(
        sum(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                parsed_left.values,
                parsed_right.values,
                strict=True,
            )
        )
        / RECURRENT_CONTROLLER_GENOME_SIZE
    )


def _genome_payload(
    genome: RecurrentControllerGenome,
) -> dict[str, object]:
    return {
        "schema_version": RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION,
        "contract": recurrent_genome_contract(),
        "values": list(genome.values),
    }


def _mutate_values(
    values: tuple[float, ...],
    *,
    seed: int,
    domain: str,
    mutation: RecurrentGenomeMutationConfig,
) -> RecurrentControllerGenome:
    if not isinstance(mutation, RecurrentGenomeMutationConfig):
        raise RecurrentGenomeError("mutation must be RecurrentGenomeMutationConfig")
    mutated: list[float] = []
    for index, value in enumerate(values):
        if _stream_unit(seed, domain, index, 1) >= mutation.rate:
            mutated.append(value)
            continue
        step = mutation.max_step * (2.0 * _stream_unit(seed, domain, index, 2) - 1.0)
        mutated.append(_quantize(_clamp_genome_value(value + step)))
    return RecurrentControllerGenome(values=tuple(mutated))


def _validate_values(values: object) -> None:
    if not isinstance(values, tuple):
        raise RecurrentGenomeError("recurrent genome values must be a tuple")
    if len(values) != RECURRENT_CONTROLLER_GENOME_SIZE:
        raise RecurrentGenomeError(
            "recurrent genome must have exactly "
            f"{RECURRENT_CONTROLLER_GENOME_SIZE} values"
        )
    for index, raw_value in enumerate(values):
        value = _strict_finite_float(raw_value, field=f"values[{index}]")
        if not (
            RECURRENT_CONTROLLER_GENOME_MIN <= value <= RECURRENT_CONTROLLER_GENOME_MAX
        ):
            raise RecurrentGenomeError(f"values[{index}] is out of bounds")
        if value != _quantize(value):
            raise RecurrentGenomeError(
                f"values[{index}] exceeds canonical float precision"
            )
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise RecurrentGenomeError(f"values[{index}] must not encode negative zero")


def _require_genome(
    genome: object,
    *,
    field: str,
) -> RecurrentControllerGenome:
    if not isinstance(genome, RecurrentControllerGenome):
        raise RecurrentGenomeError(f"{field} must be RecurrentControllerGenome")
    _validate_values(genome.values)
    return genome


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RecurrentGenomeError("seed must be an integer")
    if not 0 <= seed <= (2**64 - 1):
        raise RecurrentGenomeError("seed must be an unsigned 64-bit integer")
    return seed


def _validate_hidden_dimension(hidden_dimension: object) -> int:
    if isinstance(hidden_dimension, bool) or not isinstance(
        hidden_dimension,
        int,
    ):
        raise RecurrentGenomeError("hidden_dimension must be an integer")
    if not 1 <= hidden_dimension <= RECURRENT_CONTROLLER_MAX_HIDDEN_DIMENSION:
        raise RecurrentGenomeError(
            "hidden_dimension must be in "
            f"[1, {RECURRENT_CONTROLLER_MAX_HIDDEN_DIMENSION}]"
        )
    return hidden_dimension


def _stream_unit(seed: int, domain: str, index: int, draw: int) -> float:
    material = (
        f"{RECURRENT_CONTROLLER_INHERITANCE_CONTRACT_VERSION}\0"
        f"{seed}\0{domain}\0{index}\0{draw}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return integer / 2**64


def _development_coefficient(
    channel: str,
    hidden_index: int,
    locus: int,
) -> float:
    material = (
        f"{RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION}\0"
        f"{channel}\0{hidden_index}\0{locus}"
    ).encode("ascii")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return 2.0 * (integer / 2**64) - 1.0


@lru_cache(maxsize=None)
def _cached_recurrent_genome_development_coefficients(
    hidden_dimension: int,
) -> FilmDevelopmentCoefficients:
    return FilmDevelopmentCoefficients(
        contract_version=RECURRENT_CONTROLLER_DEVELOPMENT_MAP_VERSION,
        hidden_dimension=hidden_dimension,
        genome_size=RECURRENT_CONTROLLER_GENOME_SIZE,
        scale=tuple(
            tuple(
                _development_coefficient("scale", hidden_index, locus)
                for locus in range(RECURRENT_CONTROLLER_GENOME_SIZE)
            )
            for hidden_index in range(hidden_dimension)
        ),
        bias=tuple(
            tuple(
                _development_coefficient("bias", hidden_index, locus)
                for locus in range(RECURRENT_CONTROLLER_GENOME_SIZE)
            )
            for hidden_index in range(hidden_dimension)
        ),
    )


def _clamp_genome_value(value: float) -> float:
    return max(
        RECURRENT_CONTROLLER_GENOME_MIN,
        min(RECURRENT_CONTROLLER_GENOME_MAX, value),
    )


def _softsign(value: float) -> float:
    return value / (1.0 + abs(value))


def _quantize(value: float) -> float:
    rounded = round(float(value), RECURRENT_CONTROLLER_GENOME_FLOAT_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _strict_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise RecurrentGenomeError(f"{field} must be a float")
    if not math.isfinite(value):
        raise RecurrentGenomeError(f"{field} must be finite")
    return value


def _validate_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentGenomeError(f"{field} must be lowercase SHA256")
    return value


def _require_exact_keys(
    mapping: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    actual = frozenset(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RecurrentGenomeError(
            f"{field} keys mismatch; missing={missing}, extra={extra}"
        )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RecurrentGenomeError(
            "recurrent genome payload is not canonical JSON"
        ) from exc


def _unique_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise RecurrentGenomeError(
                f"recurrent genome JSON has duplicate key {key!r}"
            )
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> None:
    raise RecurrentGenomeError(
        f"recurrent genome JSON constant {value!r} is not finite"
    )


def _copy_json_value(value: object) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("ascii"))
