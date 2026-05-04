from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

from evolution_sim.env.runtime.trajectory import (
    build_trajectory_summary_from_stats,
    empty_trajectory_stats,
    update_trajectory_stats,
)
from evolution_sim.mind.provenance import (
    DEFAULT_SPLIT_ID,
    build_dataset_provenance,
)


class JsonlTrajectoryWriter:
    """Stream trajectory records as JSON Lines without retaining replay payloads."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        source_seeds: list[int] | None = None,
        split_id: str = DEFAULT_SPLIT_ID,
    ):
        self.output_path = Path(output_path)
        self.source_seeds = tuple(source_seeds or ())
        self.split_id = split_id
        self._temp_path: Path | None = None
        self._handle: TextIO | None = None
        self._stats = empty_trajectory_stats()
        self._contract: dict[str, object] | None = None
        self._config: dict[str, object] | None = None
        self._finished = False

    @property
    def record_count(self) -> int:
        return int(self._stats["record_count"])

    @property
    def trajectory_summary(self) -> dict[str, object]:
        return self._summary_payload()

    def begin(
        self,
        *,
        run_id: str,
        config: dict[str, object],
        contract: dict[str, object],
    ) -> None:
        if self._handle is not None:
            raise RuntimeError("trajectory writer is already open")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "wb",
            dir=self.output_path.parent,
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            self._temp_path = Path(temp_file.name)
        if self.output_path.suffix == ".gz":
            self._handle = gzip.open(self._temp_path, "wt", encoding="utf-8")
        else:
            self._handle = self._temp_path.open("w", encoding="utf-8")
        self._contract = contract
        self._config = config
        self._write_line(
            {
                "type": "header",
                "format": "evolution_sim_trajectory_jsonl_v1",
                "compression": "gzip" if self.output_path.suffix == ".gz" else "none",
                "run_id": run_id,
                "config": config,
                "trajectory_contract": contract,
                "provenance": self._provenance_payload(record_count=None),
            }
        )

    def write_record(self, record: dict[str, object]) -> None:
        self._ensure_open()
        update_trajectory_stats(self._stats, record)
        self._write_line({"type": "record", "record": record})

    def finish(self, *, summary: dict[str, object]) -> None:
        if self._finished:
            return
        self._ensure_open()
        self._write_line(
            {
                "type": "footer",
                "summary": summary,
                "trajectory_summary": self._summary_payload(),
                "provenance": self._provenance_payload(
                    record_count=self.record_count,
                ),
            }
        )
        self._close_handle()
        if self._temp_path is None:
            raise RuntimeError("trajectory writer temp path is missing")
        os.replace(self._temp_path, self.output_path)
        self._temp_path = None
        self._finished = True

    def abort(self) -> None:
        self._close_handle()
        if self._temp_path is not None:
            self._temp_path.unlink(missing_ok=True)
            self._temp_path = None

    def __enter__(self) -> JsonlTrajectoryWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or not self._finished:
            self.abort()

    def _summary_payload(self) -> dict[str, object]:
        contract = self._contract or {}
        return {
            "schema_version": contract.get("schema_version"),
            "observation_schema_version": contract.get("observation_schema_version"),
            "policy_interface_version": contract.get("policy_interface_version"),
            "action_contract_version": contract.get("action_contract_version"),
            "reproductive_group_contract_version": contract.get(
                "reproductive_group_contract_version"
            ),
            "genome_recombination_contract_version": contract.get(
                "genome_recombination_contract_version"
            ),
            "reward_schema_version": contract.get("reward_schema_version"),
            "action_outcome_schema_version": contract.get("action_outcome_schema_version"),
            **build_trajectory_summary_from_stats(self._stats),
        }

    def _provenance_payload(self, *, record_count: int | None) -> dict[str, object]:
        return build_dataset_provenance(
            config=self._config or {},
            contract=self._contract or {},
            trajectory_paths=[self.output_path],
            source_seeds=self.source_seeds,
            split_id=self.split_id,
            record_count=record_count,
        )

    def _write_line(self, payload: dict[str, object]) -> None:
        self._ensure_open()
        assert self._handle is not None
        self._handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self._handle.write("\n")

    def _ensure_open(self) -> None:
        if self._handle is None:
            raise RuntimeError("trajectory writer is not open")

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
