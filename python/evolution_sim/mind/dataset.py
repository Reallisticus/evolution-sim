from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

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
from evolution_sim.mind.provenance import validate_dataset_provenance

TRAJECTORY_JSONL_FORMAT = "evolution_sim_trajectory_jsonl_v1"


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
