from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from typing import Any, Self

from evolution_sim.env.runtime.state import empty_mind_inheritance_metadata
from evolution_sim.mind.recurrent_genome import (
    RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION,
    RECURRENT_CONTROLLER_GENOME_SIZE,
    RECURRENT_CONTROLLER_INHERITANCE_CONTRACT_VERSION,
    RecurrentControllerGenome,
    RecurrentGenomeError,
    RecurrentGenomeMutationConfig,
    founder_recurrent_genome,
    inherit_asexual_recurrent_genome,
    recombine_recurrent_genomes,
    recurrent_genome_artifact,
    validate_recurrent_genome_artifact,
    zero_recurrent_genome,
)


RECURRENT_GENOME_POPULATION_CONTRACT_VERSION = (
    "mind_recurrent_genome_population_contract_v1"
)
RECURRENT_GENOME_STREAM_NAMESPACE_VERSION = (
    "mind_recurrent_genome_independent_sha256_stream_v1"
)
RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION = (
    "mind_recurrent_genome_inheritance_metadata_v1"
)
RECURRENT_GENOME_POPULATION_SNAPSHOT_SCHEMA_VERSION = (
    "mind_recurrent_genome_population_snapshot_v1"
)
RECURRENT_GENOME_POPULATION_SNAPSHOT_DIGEST_POLICY = (
    "sha256_canonical_json_without_snapshot_sha256_v1"
)
RECURRENT_GENOME_EVENT_SEED_MIN = 2**63
RECURRENT_GENOME_EVENT_SEED_MAX = 2**64 - 1

_EVENT_KINDS = frozenset({"founder", "asexual", "two_parent"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "mode",
        "genome_stream",
        "mutation",
        "agents",
        "snapshot_sha256",
    }
)
_GENOME_STREAM_KEYS = frozenset(
    {
        "namespace_version",
        "genome_stream_seed",
        "world_identity",
        "binding_sha256",
    }
)
_MUTATION_KEYS = frozenset({"rate", "max_step"})
_AGENT_ENTRY_KEYS = frozenset({"agent_id", "genome"})
_POPULATION_CONTRACT = {
    "schema_version": RECURRENT_GENOME_POPULATION_CONTRACT_VERSION,
    "controller_genome_schema_version": RECURRENT_CONTROLLER_GENOME_SCHEMA_VERSION,
    "controller_inheritance_contract_version": (
        RECURRENT_CONTROLLER_INHERITANCE_CONTRACT_VERSION
    ),
    "genome_stream": {
        "namespace_version": RECURRENT_GENOME_STREAM_NAMESPACE_VERSION,
        "root_seed_type": "unsigned_64_bit_integer",
        "world_identity_policy": "nonempty_exact_string_without_nul_or_edge_space",
        "event_seed_policy": (
            "sha256_canonical_event_material_first_64_bits_with_high_bit_set_v1"
        ),
        "event_seed_bounds": [
            RECURRENT_GENOME_EVENT_SEED_MIN,
            RECURRENT_GENOME_EVENT_SEED_MAX,
        ],
        "independent_from": [
            "simulation_environment_rng",
            "policy_action_sampling_rng",
            "learner_rng",
            "torch_rng",
            "python_global_rng",
        ],
    },
    "modes": ["disabled", "heritable", "zero_all"],
    "mind_metadata_schema_version": (RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION),
    "snapshot": {
        "schema_version": RECURRENT_GENOME_POPULATION_SNAPSHOT_SCHEMA_VERSION,
        "digest_policy": RECURRENT_GENOME_POPULATION_SNAPSHOT_DIGEST_POLICY,
        "agent_order": "strictly_increasing_agent_id",
    },
    "inherited_payload": "controller_genome_values_only",
    "never_inherited": [
        "recurrent_hidden_state",
        "previous_public_feedback",
        "optimizer_state",
        "learned_model_weights",
        "lifetime_acquired_state",
    ],
}


class RecurrentGenomePopulationError(RecurrentGenomeError):
    """Raised when runtime controller-genome population state fails closed."""


class RecurrentGenomePopulationMode(StrEnum):
    DISABLED = "disabled"
    HERITABLE = "heritable"
    ZERO_ALL = "zero_all"


@dataclass(frozen=True, slots=True)
class RecurrentGenomeBinding:
    """One immutable live-agent genome plus its validated cached digest."""

    genome: RecurrentControllerGenome
    genome_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.genome, RecurrentControllerGenome):
            raise RecurrentGenomePopulationError(
                "binding genome must be RecurrentControllerGenome"
            )
        object.__setattr__(self, "genome_sha256", self.genome.sha256)


def _bind_genome(genome: RecurrentControllerGenome) -> RecurrentGenomeBinding:
    return RecurrentGenomeBinding(genome=genome)


def recurrent_genome_population_contract() -> dict[str, object]:
    """Return the exact runtime population, seed, and snapshot contract."""
    return _copy_json_value(_POPULATION_CONTRACT)


def recurrent_genome_stream_binding_sha256(
    *,
    genome_stream_seed: int,
    world_identity: str,
) -> str:
    """Bind one independent genome stream to one exact world identity."""
    parsed_seed = _validate_unsigned_64_bit_seed(genome_stream_seed)
    parsed_world_identity = _validate_world_identity(world_identity)
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "namespace_version": RECURRENT_GENOME_STREAM_NAMESPACE_VERSION,
                "genome_stream_seed": parsed_seed,
                "world_identity": parsed_world_identity,
            }
        )
    ).hexdigest()


def derive_recurrent_genome_event_seed(
    *,
    genome_stream_seed: int,
    world_identity: str,
    event_kind: str,
    agent_id: int,
    parent_genome_sha256s: Sequence[str] = (),
) -> int:
    """Derive one order-independent event seed without touching any RNG."""
    parsed_seed = _validate_unsigned_64_bit_seed(genome_stream_seed)
    parsed_world_identity = _validate_world_identity(world_identity)
    parsed_agent_id = _validate_agent_id(agent_id)
    if not isinstance(event_kind, str) or event_kind not in _EVENT_KINDS:
        raise RecurrentGenomePopulationError(
            f"event_kind must be one of {sorted(_EVENT_KINDS)}"
        )
    if isinstance(parent_genome_sha256s, (str, bytes)) or not isinstance(
        parent_genome_sha256s,
        Sequence,
    ):
        raise RecurrentGenomePopulationError("parent_genome_sha256s must be a sequence")
    parsed_parent_digests = tuple(
        sorted(
            _validate_sha256(value, field=f"parent_genome_sha256s[{index}]")
            for index, value in enumerate(parent_genome_sha256s)
        )
    )
    expected_parent_count = {
        "founder": 0,
        "asexual": 1,
        "two_parent": 2,
    }[event_kind]
    if len(parsed_parent_digests) != expected_parent_count:
        raise RecurrentGenomePopulationError(
            f"{event_kind} events require {expected_parent_count} parent digests"
        )
    binding_sha256 = recurrent_genome_stream_binding_sha256(
        genome_stream_seed=parsed_seed,
        world_identity=parsed_world_identity,
    )
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "namespace_version": RECURRENT_GENOME_STREAM_NAMESPACE_VERSION,
                "population_binding_sha256": binding_sha256,
                "event_kind": event_kind,
                "agent_id": parsed_agent_id,
                "parent_genome_sha256s": list(parsed_parent_digests),
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") | RECURRENT_GENOME_EVENT_SEED_MIN


class RecurrentGenomePopulationManager:
    """Own deterministic inherited controller genomes for one exact world."""

    def __init__(
        self,
        *,
        genome_stream_seed: int,
        world_identity: str,
        mode: RecurrentGenomePopulationMode | str = (
            RecurrentGenomePopulationMode.DISABLED
        ),
        mutation: RecurrentGenomeMutationConfig | None = None,
    ) -> None:
        self._genome_stream_seed = _validate_unsigned_64_bit_seed(genome_stream_seed)
        self._world_identity = _validate_world_identity(world_identity)
        self._mode = _parse_mode(mode)
        if mutation is None:
            self._mutation = RecurrentGenomeMutationConfig()
        elif isinstance(mutation, RecurrentGenomeMutationConfig):
            self._mutation = mutation
        else:
            raise RecurrentGenomePopulationError(
                "mutation must be RecurrentGenomeMutationConfig"
            )
        self._binding_sha256 = recurrent_genome_stream_binding_sha256(
            genome_stream_seed=self._genome_stream_seed,
            world_identity=self._world_identity,
        )
        self._bindings: dict[int, RecurrentGenomeBinding] = {}
        self._state_sha256_cache: str | None = None
        self._empty_state_sha256 = self.state_sha256

    @property
    def mode(self) -> RecurrentGenomePopulationMode:
        return self._mode

    @property
    def genome_stream_seed(self) -> int:
        return self._genome_stream_seed

    @property
    def world_identity(self) -> str:
        return self._world_identity

    @property
    def binding_sha256(self) -> str:
        return self._binding_sha256

    @property
    def mutation(self) -> RecurrentGenomeMutationConfig:
        return self._mutation

    @property
    def population_size(self) -> int:
        return len(self._bindings)

    @property
    def registered_agent_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._bindings))

    @property
    def state_sha256(self) -> str:
        if self._state_sha256_cache is None:
            self._state_sha256_cache = str(self.snapshot_artifact()["snapshot_sha256"])
        return self._state_sha256_cache

    @property
    def empty_state_sha256(self) -> str:
        return self._empty_state_sha256

    def founder_metadata(self, *, agent_id: int) -> dict[str, object]:
        """Register one founder and return compact replay-visible metadata."""
        parsed_agent_id = _validate_agent_id(agent_id)
        if self._mode is RecurrentGenomePopulationMode.DISABLED:
            return empty_mind_inheritance_metadata()
        self._require_new_agent_id(parsed_agent_id)
        if self._mode is RecurrentGenomePopulationMode.ZERO_ALL:
            genome = zero_recurrent_genome()
        else:
            genome = founder_recurrent_genome(
                seed=self._event_seed(
                    event_kind="founder",
                    agent_id=parsed_agent_id,
                    parent_genome_sha256s=(),
                )
            )
        binding = _bind_genome(genome)
        self._bindings[parsed_agent_id] = binding
        self._invalidate_state_sha256()
        return self._mind_metadata(
            binding=binding,
            inheritance_kind="founder",
            parent_genome_sha256s=(),
        )

    def child_metadata(
        self,
        *,
        child_agent_id: int,
        primary_parent_id: int,
        secondary_parent_id: int | None,
    ) -> dict[str, object]:
        """Register an actual child from exact current parent genomes."""
        parsed_child_id = _validate_agent_id(child_agent_id)
        parsed_primary_parent_id = _validate_agent_id(primary_parent_id)
        parsed_secondary_parent_id = (
            None
            if secondary_parent_id is None
            else _validate_agent_id(secondary_parent_id)
        )
        if self._mode is RecurrentGenomePopulationMode.DISABLED:
            return empty_mind_inheritance_metadata()
        self._require_new_agent_id(parsed_child_id)
        if parsed_child_id == parsed_primary_parent_id or (
            parsed_secondary_parent_id is not None
            and parsed_child_id == parsed_secondary_parent_id
        ):
            raise RecurrentGenomePopulationError(
                "child agent id must differ from parent ids"
            )
        primary_binding = self.genome_binding_for_agent(parsed_primary_parent_id)
        primary = primary_binding.genome
        if parsed_secondary_parent_id is None:
            parents = (primary,)
            parent_genome_sha256s = (primary_binding.genome_sha256,)
            inheritance_kind = "asexual"
        else:
            if parsed_secondary_parent_id == parsed_primary_parent_id:
                raise RecurrentGenomePopulationError(
                    "two-parent inheritance requires distinct parent ids"
                )
            secondary_binding = self.genome_binding_for_agent(
                parsed_secondary_parent_id
            )
            secondary = secondary_binding.genome
            parents = (primary, secondary)
            parent_genome_sha256s = tuple(
                sorted(
                    (
                        primary_binding.genome_sha256,
                        secondary_binding.genome_sha256,
                    )
                )
            )
            inheritance_kind = "two_parent"
        if self._mode is RecurrentGenomePopulationMode.ZERO_ALL:
            genome = zero_recurrent_genome()
        elif inheritance_kind == "asexual":
            genome = inherit_asexual_recurrent_genome(
                primary,
                seed=self._event_seed(
                    event_kind=inheritance_kind,
                    agent_id=parsed_child_id,
                    parent_genome_sha256s=parent_genome_sha256s,
                ),
                mutation=self._mutation,
            )
        else:
            genome = recombine_recurrent_genomes(
                parents[0],
                parents[1],
                seed=self._event_seed(
                    event_kind=inheritance_kind,
                    agent_id=parsed_child_id,
                    parent_genome_sha256s=parent_genome_sha256s,
                ),
                mutation=self._mutation,
            )
        binding = _bind_genome(genome)
        self._bindings[parsed_child_id] = binding
        self._invalidate_state_sha256()
        return self._mind_metadata(
            binding=binding,
            inheritance_kind=inheritance_kind,
            parent_genome_sha256s=parent_genome_sha256s,
        )

    def genome_for_agent(self, agent_id: int) -> RecurrentControllerGenome:
        return self.genome_binding_for_agent(agent_id).genome

    def genome_binding_for_agent(self, agent_id: int) -> RecurrentGenomeBinding:
        parsed_agent_id = _validate_agent_id(agent_id)
        try:
            return self._bindings[parsed_agent_id]
        except KeyError as exc:
            raise RecurrentGenomePopulationError(
                f"agent {parsed_agent_id} has no registered recurrent genome"
            ) from exc

    def genome_sha256_for_agent(self, agent_id: int) -> str:
        return self.genome_binding_for_agent(agent_id).genome_sha256

    def genome_artifact_for_agent(self, agent_id: int) -> dict[str, object]:
        return recurrent_genome_artifact(self.genome_for_agent(agent_id))

    def discard_agent(self, agent_id: int) -> bool:
        """Discard acquired runtime ownership after death without using RNG."""
        parsed_agent_id = _validate_agent_id(agent_id)
        discarded = self._bindings.pop(parsed_agent_id, None) is not None
        if discarded:
            self._invalidate_state_sha256()
        return discarded

    def reconcile_live_agent_ids(
        self,
        live_agent_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Atomically validate every live owner, then discard registered dead."""
        discarded_agent_ids = self.reconciliation_dead_agent_ids(live_agent_ids)
        for agent_id in discarded_agent_ids:
            del self._bindings[agent_id]
        if discarded_agent_ids:
            self._invalidate_state_sha256()
        return discarded_agent_ids

    def reconciliation_dead_agent_ids(
        self,
        live_agent_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Return the exact dead-owner plan without mutating population state."""
        if isinstance(live_agent_ids, (str, bytes)) or not isinstance(
            live_agent_ids,
            Sequence,
        ):
            raise RecurrentGenomePopulationError(
                "live_agent_ids must be a strictly increasing sequence"
            )
        parsed_live_agent_ids = tuple(
            _validate_agent_id(agent_id) for agent_id in live_agent_ids
        )
        if parsed_live_agent_ids != tuple(sorted(set(parsed_live_agent_ids))):
            raise RecurrentGenomePopulationError(
                "live_agent_ids must be strictly increasing and unique"
            )
        if self._mode is RecurrentGenomePopulationMode.DISABLED:
            if self._bindings:
                raise RecurrentGenomePopulationError(
                    "disabled population cannot own recurrent genomes"
                )
            return ()
        missing_live_agent_ids = tuple(
            agent_id
            for agent_id in parsed_live_agent_ids
            if agent_id not in self._bindings
        )
        if missing_live_agent_ids:
            raise RecurrentGenomePopulationError(
                "live agents have no registered recurrent genome: "
                f"{list(missing_live_agent_ids)}"
            )
        live_agent_id_set = set(parsed_live_agent_ids)
        return tuple(
            agent_id
            for agent_id in self.registered_agent_ids
            if agent_id not in live_agent_id_set
        )

    def reset(self) -> None:
        """Clear all live genomes while preserving the exact stream binding."""
        if self._bindings:
            self._bindings.clear()
            self._invalidate_state_sha256()

    def snapshot_artifact(self) -> dict[str, object]:
        """Serialize the complete live population under an exact state digest."""
        payload: dict[str, object] = {
            "schema_version": RECURRENT_GENOME_POPULATION_SNAPSHOT_SCHEMA_VERSION,
            "contract": recurrent_genome_population_contract(),
            "mode": self._mode.value,
            "genome_stream": {
                "namespace_version": RECURRENT_GENOME_STREAM_NAMESPACE_VERSION,
                "genome_stream_seed": self._genome_stream_seed,
                "world_identity": self._world_identity,
                "binding_sha256": self._binding_sha256,
            },
            "mutation": {
                "rate": self._mutation.rate,
                "max_step": self._mutation.max_step,
            },
            "agents": [
                {
                    "agent_id": agent_id,
                    "genome": recurrent_genome_artifact(binding.genome),
                }
                for agent_id, binding in sorted(self._bindings.items())
            ],
        }
        snapshot_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        self._state_sha256_cache = snapshot_sha256
        return {
            **payload,
            "snapshot_sha256": snapshot_sha256,
        }

    def serialize_snapshot(self) -> bytes:
        return _canonical_json_bytes(self.snapshot_artifact())

    @classmethod
    def from_snapshot_artifact(
        cls,
        artifact: Mapping[str, object],
        *,
        expected_snapshot_sha256: str | None = None,
    ) -> Self:
        """Strictly restore a manager from one exact validated snapshot."""
        if not isinstance(artifact, Mapping):
            raise RecurrentGenomePopulationError(
                "population snapshot must be an object"
            )
        _require_exact_keys(artifact, _SNAPSHOT_KEYS, field="population snapshot")
        if (
            artifact.get("schema_version")
            != RECURRENT_GENOME_POPULATION_SNAPSHOT_SCHEMA_VERSION
        ):
            raise RecurrentGenomePopulationError(
                "population snapshot schema_version is stale"
            )
        if artifact.get("contract") != _POPULATION_CONTRACT:
            raise RecurrentGenomePopulationError(
                "population snapshot contract is missing or stale"
            )
        mode = _parse_mode(artifact.get("mode"))
        stream = artifact.get("genome_stream")
        if not isinstance(stream, Mapping):
            raise RecurrentGenomePopulationError(
                "population snapshot genome_stream must be an object"
            )
        _require_exact_keys(stream, _GENOME_STREAM_KEYS, field="genome_stream")
        if stream.get("namespace_version") != RECURRENT_GENOME_STREAM_NAMESPACE_VERSION:
            raise RecurrentGenomePopulationError(
                "population snapshot genome stream namespace is stale"
            )
        genome_stream_seed = _validate_unsigned_64_bit_seed(
            stream.get("genome_stream_seed")
        )
        world_identity = _validate_world_identity(stream.get("world_identity"))
        binding_sha256 = _validate_sha256(
            stream.get("binding_sha256"),
            field="genome_stream.binding_sha256",
        )
        expected_binding_sha256 = recurrent_genome_stream_binding_sha256(
            genome_stream_seed=genome_stream_seed,
            world_identity=world_identity,
        )
        if binding_sha256 != expected_binding_sha256:
            raise RecurrentGenomePopulationError(
                "population snapshot genome stream binding SHA256 mismatch"
            )
        raw_mutation = artifact.get("mutation")
        if not isinstance(raw_mutation, Mapping):
            raise RecurrentGenomePopulationError(
                "population snapshot mutation must be an object"
            )
        _require_exact_keys(raw_mutation, _MUTATION_KEYS, field="mutation")
        try:
            mutation = RecurrentGenomeMutationConfig(
                rate=raw_mutation.get("rate"),  # type: ignore[arg-type]
                max_step=raw_mutation.get("max_step"),  # type: ignore[arg-type]
            )
        except RecurrentGenomeError as exc:
            raise RecurrentGenomePopulationError(
                f"population snapshot mutation is invalid: {exc}"
            ) from exc
        raw_agents = artifact.get("agents")
        if not isinstance(raw_agents, list):
            raise RecurrentGenomePopulationError(
                "population snapshot agents must be a list"
            )
        if mode is RecurrentGenomePopulationMode.DISABLED and raw_agents:
            raise RecurrentGenomePopulationError(
                "disabled population snapshot must contain no genomes"
            )
        genomes: dict[int, RecurrentControllerGenome] = {}
        previous_agent_id = 0
        for index, raw_entry in enumerate(raw_agents):
            if not isinstance(raw_entry, Mapping):
                raise RecurrentGenomePopulationError(
                    f"agents[{index}] must be an object"
                )
            _require_exact_keys(
                raw_entry,
                _AGENT_ENTRY_KEYS,
                field=f"agents[{index}]",
            )
            agent_id = _validate_agent_id(raw_entry.get("agent_id"))
            if agent_id <= previous_agent_id:
                raise RecurrentGenomePopulationError(
                    "population snapshot agents must be strictly ordered "
                    "with unique ids"
                )
            previous_agent_id = agent_id
            raw_genome = raw_entry.get("genome")
            if not isinstance(raw_genome, Mapping):
                raise RecurrentGenomePopulationError(
                    f"agents[{index}].genome must be an object"
                )
            try:
                genome = validate_recurrent_genome_artifact(raw_genome)
            except RecurrentGenomeError as exc:
                raise RecurrentGenomePopulationError(
                    f"agents[{index}].genome is invalid: {exc}"
                ) from exc
            if (
                mode is RecurrentGenomePopulationMode.ZERO_ALL
                and genome != zero_recurrent_genome()
            ):
                raise RecurrentGenomePopulationError(
                    "zero_all population snapshot contains a nonzero genome"
                )
            genomes[agent_id] = genome
        observed_snapshot_sha256 = _validate_sha256(
            artifact.get("snapshot_sha256"),
            field="snapshot_sha256",
        )
        payload = {
            key: artifact[key] for key in _SNAPSHOT_KEYS if key != "snapshot_sha256"
        }
        computed_snapshot_sha256 = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
        if observed_snapshot_sha256 != computed_snapshot_sha256:
            raise RecurrentGenomePopulationError("population snapshot SHA256 mismatch")
        if expected_snapshot_sha256 is not None:
            parsed_expected = _validate_sha256(
                expected_snapshot_sha256,
                field="expected_snapshot_sha256",
            )
            if observed_snapshot_sha256 != parsed_expected:
                raise RecurrentGenomePopulationError(
                    "population snapshot does not match expected SHA256"
                )
        manager = cls(
            genome_stream_seed=genome_stream_seed,
            world_identity=world_identity,
            mode=mode,
            mutation=mutation,
        )
        manager._bindings.update(
            {agent_id: _bind_genome(genome) for agent_id, genome in genomes.items()}
        )
        manager._invalidate_state_sha256()
        if manager.snapshot_artifact() != artifact:
            raise RecurrentGenomePopulationError(
                "population snapshot is not canonical for restored state"
            )
        return manager

    @classmethod
    def from_serialized_snapshot(
        cls,
        serialized: str | bytes,
        *,
        expected_snapshot_sha256: str | None = None,
        require_canonical: bool = True,
    ) -> Self:
        if isinstance(serialized, bytes):
            try:
                text = serialized.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RecurrentGenomePopulationError(
                    "population snapshot serialization is not UTF-8"
                ) from exc
        elif isinstance(serialized, str):
            text = serialized
        else:
            raise RecurrentGenomePopulationError(
                "population snapshot serialization must be str or bytes"
            )
        try:
            artifact = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise RecurrentGenomePopulationError(
                "population snapshot serialization is not valid JSON"
            ) from exc
        if not isinstance(artifact, dict):
            raise RecurrentGenomePopulationError(
                "population snapshot must be an object"
            )
        manager = cls.from_snapshot_artifact(
            artifact,
            expected_snapshot_sha256=expected_snapshot_sha256,
        )
        if require_canonical and text.encode("utf-8") != _canonical_json_bytes(
            artifact
        ):
            raise RecurrentGenomePopulationError(
                "population snapshot serialization is not canonical JSON"
            )
        return manager

    def _require_new_agent_id(self, agent_id: int) -> None:
        if agent_id in self._bindings:
            raise RecurrentGenomePopulationError(
                f"agent {agent_id} already has a registered recurrent genome"
            )

    def _invalidate_state_sha256(self) -> None:
        self._state_sha256_cache = None

    def _event_seed(
        self,
        *,
        event_kind: str,
        agent_id: int,
        parent_genome_sha256s: Sequence[str],
    ) -> int:
        return derive_recurrent_genome_event_seed(
            genome_stream_seed=self._genome_stream_seed,
            world_identity=self._world_identity,
            event_kind=event_kind,
            agent_id=agent_id,
            parent_genome_sha256s=parent_genome_sha256s,
        )

    def _mind_metadata(
        self,
        *,
        binding: RecurrentGenomeBinding,
        inheritance_kind: str,
        parent_genome_sha256s: Sequence[str],
    ) -> dict[str, object]:
        return {
            "schema_version": RECURRENT_GENOME_MIND_METADATA_SCHEMA_VERSION,
            "inherited_state": True,
            "state_size": RECURRENT_CONTROLLER_GENOME_SIZE,
            "population_mode": self._mode.value,
            "population_binding_sha256": self._binding_sha256,
            "inheritance_kind": inheritance_kind,
            "genome_sha256": binding.genome_sha256,
            "parent_genome_sha256s": list(parent_genome_sha256s),
        }


def _parse_mode(value: object) -> RecurrentGenomePopulationMode:
    if isinstance(value, RecurrentGenomePopulationMode):
        return value
    if isinstance(value, str):
        try:
            return RecurrentGenomePopulationMode(value)
        except ValueError as exc:
            raise RecurrentGenomePopulationError(
                "mode must be exactly disabled, heritable, or zero_all"
            ) from exc
    raise RecurrentGenomePopulationError(
        "mode must be exactly disabled, heritable, or zero_all"
    )


def _validate_unsigned_64_bit_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrentGenomePopulationError(
            "genome_stream_seed must be an unsigned 64-bit integer"
        )
    if not 0 <= value <= RECURRENT_GENOME_EVENT_SEED_MAX:
        raise RecurrentGenomePopulationError(
            "genome_stream_seed must be an unsigned 64-bit integer"
        )
    return value


def _validate_world_identity(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrentGenomePopulationError("world_identity must be a nonempty string")
    if value != value.strip() or "\x00" in value:
        raise RecurrentGenomePopulationError(
            "world_identity must not contain edge whitespace or NUL"
        )
    return value


def _validate_agent_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentGenomePopulationError("agent_id must be a positive integer")
    return value


def _validate_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise RecurrentGenomePopulationError(f"{field} must be lowercase SHA256")
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
        raise RecurrentGenomePopulationError(
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
        raise RecurrentGenomePopulationError(
            "population payload is not canonical JSON"
        ) from exc


def _copy_json_value(value: object) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise RecurrentGenomePopulationError(
                f"population snapshot JSON has duplicate key {key!r}"
            )
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> None:
    raise RecurrentGenomePopulationError(
        f"population snapshot JSON constant {value!r} is not finite"
    )
