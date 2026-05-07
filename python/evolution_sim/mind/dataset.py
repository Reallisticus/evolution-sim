from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TextIO

from evolution_sim.env.contracts import SUMMARY_SCHEMA_VERSION
from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.observations import OBSERVATION_SCHEMA_VERSION
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reproduction import REPRODUCTIVE_GROUP_CONTRACT_VERSION
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_RECORD_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
    build_trajectory_summary_from_stats,
    empty_trajectory_stats,
    update_trajectory_stats,
)
from evolution_sim.genome.recombination import GENOME_RECOMBINATION_CONTRACT_VERSION
from evolution_sim.mind.provenance import (
    stable_payload_digest,
    validate_dataset_provenance,
)

TRAJECTORY_JSONL_FORMAT = "evolution_sim_trajectory_jsonl_v1"
TRAJECTORY_EPISODE_ID_FIELD = "__trajectory_episode_id"


class TrajectoryDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrajectoryJsonlDataset:
    path: Path
    header: dict[str, object]
    records: tuple[dict[str, object], ...]
    footer: dict[str, object]

    @property
    def record_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class TrajectoryTransition:
    episode_id: str
    tick: int
    agent_id: int
    observation_input: dict[str, object]
    action_mask: dict[str, bool]
    action: str
    reward_total: float
    next_observation_input: dict[str, object] | None
    next_action_mask: dict[str, bool] | None
    done: bool


def load_trajectory_jsonl(path: str | Path) -> TrajectoryJsonlDataset:
    resolved_path = Path(path)
    payloads = list(_read_jsonl_payloads(resolved_path))
    if len(payloads) < 2:
        raise TrajectoryDatasetError("trajectory JSONL must include header and footer")
    header = payloads[0]
    footer = payloads[-1]
    records = payloads[1:-1]
    _validate_header(header)
    parsed_records = [
        _validate_record_payload(payload, index)
        for index, payload in enumerate(records)
    ]
    _validate_footer(footer, parsed_records)
    return TrajectoryJsonlDataset(
        path=resolved_path,
        header=header,
        records=tuple(parsed_records),
        footer=footer,
    )


def build_trajectory_transitions(
    records: Sequence[dict[str, object]],
) -> tuple[TrajectoryTransition, ...]:
    transitions: list[TrajectoryTransition] = []
    for episode_index, segment in enumerate(_transition_episode_segments(records)):
        episode_id = _transition_episode_id(segment[0], episode_index=episode_index)
        next_record_by_agent: dict[int, dict[str, object]] = {}
        transitions_reversed: list[TrajectoryTransition] = []
        for record in reversed(segment):
            agent_id = _transition_agent_id(record)
            next_record = next_record_by_agent.get(agent_id)
            done = _transition_done(record, next_record=next_record)
            transitions_reversed.append(
                TrajectoryTransition(
                    episode_id=episode_id,
                    tick=_transition_tick(record),
                    agent_id=agent_id,
                    observation_input=_transition_observation_input(record),
                    action_mask=_transition_action_mask(record),
                    action=_transition_action(record),
                    reward_total=_transition_reward_total(record),
                    next_observation_input=(
                        None
                        if done or next_record is None
                        else _transition_observation_input(next_record)
                    ),
                    next_action_mask=(
                        None
                        if done or next_record is None
                        else _transition_action_mask(next_record)
                    ),
                    done=done,
                )
            )
            next_record_by_agent[agent_id] = record
        transitions.extend(reversed(transitions_reversed))
    return tuple(transitions)


def records_with_trajectory_context(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> Iterator[dict[str, object]]:
    for dataset_index, dataset in enumerate(datasets):
        episode_id = _dataset_episode_id(dataset, dataset_index=dataset_index)
        for record in dataset.records:
            contextual_record = dict(record)
            contextual_record[TRAJECTORY_EPISODE_ID_FIELD] = episode_id
            yield contextual_record


def _transition_episode_segments(
    records: Sequence[dict[str, object]],
) -> tuple[tuple[dict[str, object], ...], ...]:
    segments: list[tuple[dict[str, object], ...]] = []
    current: list[dict[str, object]] = []
    previous_tick: int | None = None
    previous_episode_id: str | None = None
    for record in records:
        tick = _transition_tick(record)
        episode_id = _explicit_transition_episode_id(record)
        if current and _starts_new_transition_episode(
            tick=tick,
            episode_id=episode_id,
            previous_tick=previous_tick,
            previous_episode_id=previous_episode_id,
        ):
            segments.append(tuple(current))
            current = []
        current.append(record)
        previous_tick = tick
        previous_episode_id = episode_id
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _starts_new_transition_episode(
    *,
    tick: int,
    episode_id: str | None,
    previous_tick: int | None,
    previous_episode_id: str | None,
) -> bool:
    if episode_id is not None or previous_episode_id is not None:
        return episode_id != previous_episode_id
    return previous_tick is not None and tick < previous_tick


def _transition_episode_id(
    record: dict[str, object],
    *,
    episode_index: int,
) -> str:
    return _explicit_transition_episode_id(record) or f"inferred-episode-{episode_index}"


def _explicit_transition_episode_id(record: dict[str, object]) -> str | None:
    episode_id = record.get(TRAJECTORY_EPISODE_ID_FIELD)
    if episode_id is None:
        return None
    if not isinstance(episode_id, str) or not episode_id:
        raise TrajectoryDatasetError(
            f"trajectory record {TRAJECTORY_EPISODE_ID_FIELD} must be a non-empty string"
        )
    return episode_id


def _dataset_episode_id(
    dataset: TrajectoryJsonlDataset,
    *,
    dataset_index: int,
) -> str:
    provenance = validate_dataset_provenance(dataset.footer.get("provenance"))
    source_seeds = ",".join(str(seed) for seed in provenance["source_seeds"])
    split_id = str(provenance["split_id"])
    return f"{dataset_index}:{split_id}:seed={source_seeds}:path={dataset.path}"


def _transition_tick(record: dict[str, object]) -> int:
    tick = record.get("tick")
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise TrajectoryDatasetError("trajectory record tick must be an integer")
    return tick


def _transition_agent_id(record: dict[str, object]) -> int:
    agent_id = record.get("agent_id")
    if isinstance(agent_id, bool) or not isinstance(agent_id, int):
        raise TrajectoryDatasetError("trajectory record agent_id must be an integer")
    return agent_id


def _transition_observation_input(record: dict[str, object]) -> dict[str, object]:
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        raise TrajectoryDatasetError(
            "trajectory record observation_input must be an object"
        )
    return dict(observation_input)


def _transition_action_mask(record: dict[str, object]) -> dict[str, bool]:
    action_mask = record.get("action_mask")
    if not isinstance(action_mask, dict):
        raise TrajectoryDatasetError("trajectory record action_mask must be an object")
    return {str(action): bool(available) for action, available in action_mask.items()}


def _transition_action(record: dict[str, object]) -> str:
    requested_action = str(record.get("requested_action"))
    resolved_action = str(record.get("resolved_action"))
    return (
        requested_action
        if bool(record.get("resolution_action_valid", False))
        else resolved_action
    )


def _transition_reward_total(record: dict[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return 0.0
    total = reward.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    parsed = float(total)
    return parsed if math.isfinite(parsed) else 0.0


def _transition_done(
    record: dict[str, object],
    *,
    next_record: dict[str, object] | None,
) -> bool:
    after = record.get("after")
    if isinstance(after, dict) and after.get("alive") is False:
        return True
    return next_record is None


def _read_jsonl_payloads(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        raise TrajectoryDatasetError(f"trajectory path does not exist: {path}")
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TrajectoryDatasetError(
                    f"line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise TrajectoryDatasetError(f"line {line_number} must be a JSON object")
            yield payload


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _validate_header(header: dict[str, object]) -> None:
    if header.get("type") != "header":
        raise TrajectoryDatasetError("first trajectory JSONL row must be a header")
    if header.get("format") != TRAJECTORY_JSONL_FORMAT:
        raise TrajectoryDatasetError("trajectory JSONL header format is unsupported")
    contract = header.get("trajectory_contract")
    if not isinstance(contract, dict):
        raise TrajectoryDatasetError("trajectory JSONL header is missing contract metadata")
    expected_versions = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "genome_recombination_contract_version": GENOME_RECOMBINATION_CONTRACT_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
    }
    for field, expected in expected_versions.items():
        if contract.get(field) != expected:
            raise TrajectoryDatasetError(
                f"trajectory contract {field} expected {expected}, found {contract.get(field)!r}"
            )
    if contract.get("record_fields") != list(TRAJECTORY_RECORD_FIELDS):
        raise TrajectoryDatasetError("trajectory contract record fields are stale")
    try:
        validate_dataset_provenance(header.get("provenance"))
    except ValueError as exc:
        raise TrajectoryDatasetError(
            f"trajectory header provenance is invalid: {exc}"
        ) from exc


def _validate_record_payload(
    payload: dict[str, object],
    index: int,
) -> dict[str, object]:
    if payload.get("type") != "record":
        raise TrajectoryDatasetError(f"trajectory row {index + 2} must be a record")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise TrajectoryDatasetError(f"trajectory record {index} must be an object")
    missing = [field for field in TRAJECTORY_RECORD_FIELDS if field not in record]
    if missing:
        raise TrajectoryDatasetError(
            f"trajectory record {index} is missing fields: {missing}"
        )
    if record.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
        raise TrajectoryDatasetError(
            f"trajectory record {index} has stale observation schema"
        )
    outcome = record.get("outcome")
    if (
        not isinstance(outcome, dict)
        or outcome.get("schema_version") != ACTION_OUTCOME_SCHEMA_VERSION
    ):
        raise TrajectoryDatasetError(
            f"trajectory record {index} has stale action outcome schema"
        )
    reward = record.get("reward")
    if not isinstance(reward, dict) or reward.get("schema_version") != REWARD_SCHEMA_VERSION:
        raise TrajectoryDatasetError(f"trajectory record {index} has stale reward schema")
    return record


def _validate_footer(
    footer: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    if footer.get("type") != "footer":
        raise TrajectoryDatasetError("last trajectory JSONL row must be a footer")
    summary = footer.get("summary")
    if not isinstance(summary, dict):
        raise TrajectoryDatasetError("trajectory footer is missing run summary")
    if summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        raise TrajectoryDatasetError("trajectory footer summary schema is stale")
    trajectory_summary = footer.get("trajectory_summary")
    if not isinstance(trajectory_summary, dict):
        raise TrajectoryDatasetError("trajectory footer is missing trajectory summary")
    try:
        footer_provenance = validate_dataset_provenance(footer.get("provenance"))
    except ValueError as exc:
        raise TrajectoryDatasetError(
            f"trajectory footer provenance is invalid: {exc}"
        ) from exc
    stats = empty_trajectory_stats()
    for record in records:
        update_trajectory_stats(stats, record)
    expected_summary = build_trajectory_summary_from_stats(stats)
    for field, expected in expected_summary.items():
        if trajectory_summary.get(field) != expected:
            raise TrajectoryDatasetError(
                (
                    f"trajectory footer {field} expected {expected}, "
                    f"found {trajectory_summary.get(field)!r}"
                )
            )
    if footer_provenance.get("record_count") != expected_summary["record_count"]:
        raise TrajectoryDatasetError(
            "trajectory footer provenance record_count is stale"
        )


def dataset_provenance(dataset: TrajectoryJsonlDataset) -> dict[str, object]:
    provenance = dict(dataset.footer["provenance"])
    provenance["trajectory_paths"] = [str(dataset.path)]
    provenance["record_count"] = dataset.record_count
    return provenance


def combined_dataset_provenance(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> dict[str, object]:
    if not datasets:
        raise TrajectoryDatasetError("at least one trajectory dataset is required")
    if len(datasets) == 1:
        return dataset_provenance(datasets[0])

    source_seeds: list[int] = []
    seen_seeds: set[int] = set()
    trajectory_paths: list[str] = []
    split_ids: list[str] = []
    contract_digests: set[str] = set()
    config_sources: list[dict[str, object]] = []
    record_count = 0

    for dataset in datasets:
        provenance = validate_dataset_provenance(dataset.footer.get("provenance"))
        trajectory_paths.append(str(dataset.path))
        record_count += dataset.record_count
        split_id = str(provenance["split_id"])
        if split_id not in split_ids:
            split_ids.append(split_id)
        contract_digests.add(str(provenance["contract_digest"]))
        for seed in provenance["source_seeds"]:
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            source_seeds.append(int(seed))
        config_sources.append(
            {
                "path": str(dataset.path),
                "source_seeds": list(provenance["source_seeds"]),
                "config_digest": str(provenance["config_digest"]),
                "record_count": dataset.record_count,
            }
        )

    if len(contract_digests) != 1:
        raise TrajectoryDatasetError(
            "cannot combine trajectory datasets with different contract digests"
        )

    return {
        "source_seeds": source_seeds,
        "config_digest": stable_payload_digest(
            {
                "combined_from": config_sources,
            }
        ),
        "contract_digest": next(iter(contract_digests)),
        "split_id": split_ids[0] if len(split_ids) == 1 else "+".join(split_ids),
        "trajectory_paths": trajectory_paths,
        "record_count": record_count,
        "source_dataset_count": len(datasets),
    }
