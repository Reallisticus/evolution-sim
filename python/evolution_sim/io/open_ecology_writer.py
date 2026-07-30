from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Mapping, Sequence


OPEN_ECOLOGY_STREAM_FORMAT = "evolution_sim_open_ecology_stream_v1"
_REQUIRED_RECORD_FIELDS = frozenset(
    {
        "tick",
        "agent_id",
        "action_mask",
        "resolution_action_mask",
        "requested_action",
        "resolved_action",
        "action_valid",
        "resolution_action_valid",
        "action_source",
        "moved",
        "before",
        "after",
        "outcome",
        "reward",
    }
)


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    """Inclusive tick range for deliberately retained full trajectory records."""

    window_id: str
    start_tick: int
    end_tick: int

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
        }


@dataclass(slots=True)
class _BoundedCounter:
    max_keys: int
    values: dict[str, int] = field(default_factory=dict)
    overflow_count: int = 0

    def add(self, key: object, amount: int = 1) -> None:
        normalized = "unknown" if key is None else str(key)
        if normalized in self.values:
            self.values[normalized] += amount
        elif len(self.values) < self.max_keys:
            self.values[normalized] = amount
        else:
            self.overflow_count += amount

    def payload(self) -> dict[str, object]:
        return {
            "counts": {key: self.values[key] for key in sorted(self.values)},
            "overflow_count": self.overflow_count,
        }


@dataclass(slots=True)
class _EcologyStats:
    max_counter_keys: int
    first_tick: int | None = None
    last_tick: int | None = None
    record_count: int = 0
    observation_invalid_action_count: int = 0
    resolution_invalid_action_count: int = 0
    moved_count: int = 0
    alive_after_count: int = 0
    reproduction_event_record_count: int = 0
    death_event_record_count: int = 0
    reproduction_ready_after_count: int = 0
    mate_requested_count: int = 0
    mate_resolved_count: int = 0
    attack_attempt_count: int = 0
    attack_success_count: int = 0
    attack_kill_count: int = 0
    passive_attack_kill_count: int = 0
    feeding_event_count: int = 0
    drinking_event_count: int = 0
    signal_emission_count: int = 0
    reward_total: float = 0.0
    resource_gain_total: float = 0.0
    attack_damage_total: float = 0.0
    passive_attack_damage_total: float = 0.0
    signal_energy_cost_total: float = 0.0
    after_energy_ratio_total: float = 0.0
    after_energy_ratio_count: int = 0
    after_hydration_ratio_total: float = 0.0
    after_hydration_ratio_count: int = 0
    after_health_ratio_total: float = 0.0
    after_health_ratio_count: int = 0
    requested_actions: _BoundedCounter = field(init=False)
    resolved_actions: _BoundedCounter = field(init=False)
    action_sources: _BoundedCounter = field(init=False)
    runtime_species: _BoundedCounter = field(init=False)
    runtime_ecotypes: _BoundedCounter = field(init=False)
    food_sources: _BoundedCounter = field(init=False)
    signal_tokens: _BoundedCounter = field(init=False)
    death_causes: _BoundedCounter = field(init=False)

    def __post_init__(self) -> None:
        self.requested_actions = _BoundedCounter(self.max_counter_keys)
        self.resolved_actions = _BoundedCounter(self.max_counter_keys)
        self.action_sources = _BoundedCounter(self.max_counter_keys)
        self.runtime_species = _BoundedCounter(self.max_counter_keys)
        self.runtime_ecotypes = _BoundedCounter(self.max_counter_keys)
        self.food_sources = _BoundedCounter(self.max_counter_keys)
        self.signal_tokens = _BoundedCounter(self.max_counter_keys)
        self.death_causes = _BoundedCounter(self.max_counter_keys)

    @property
    def counter_key_count(self) -> int:
        return sum(
            len(counter.values)
            for counter in (
                self.requested_actions,
                self.resolved_actions,
                self.action_sources,
                self.runtime_species,
                self.runtime_ecotypes,
                self.food_sources,
                self.signal_tokens,
                self.death_causes,
            )
        )

    def update(self, record: Mapping[str, object]) -> None:
        tick = int(record["tick"])
        self.first_tick = (
            tick if self.first_tick is None else min(self.first_tick, tick)
        )
        self.last_tick = tick if self.last_tick is None else max(self.last_tick, tick)
        self.record_count += 1
        self.observation_invalid_action_count += int(not bool(record["action_valid"]))
        self.resolution_invalid_action_count += int(
            not bool(record["resolution_action_valid"])
        )
        self.moved_count += int(bool(record["moved"]))

        requested_action = str(record["requested_action"])
        resolved_action = str(record["resolved_action"])
        self.requested_actions.add(requested_action)
        self.resolved_actions.add(resolved_action)
        self.action_sources.add(record["action_source"])
        self.runtime_species.add(record.get("runtime_species_id"))
        self.runtime_ecotypes.add(record.get("runtime_ecotype_id"))
        self.mate_requested_count += int(requested_action == "mate")
        self.mate_resolved_count += int(resolved_action == "mate")

        after = _mapping(record["after"], "record.after")
        self.alive_after_count += int(bool(after["alive"]))
        self._add_ratio(after, "energy_ratio")
        self._add_ratio(after, "hydration_ratio")
        self._add_ratio(after, "health_ratio")

        reward = _mapping(record["reward"], "record.reward")
        self.reward_total += _finite_number(reward["total"], "record.reward.total")

        outcome = _mapping(record["outcome"], "record.outcome")
        self.reproduction_event_record_count += int(bool(outcome["reproduced"]))
        self.death_event_record_count += int(bool(outcome["died"]))
        self.reproduction_ready_after_count += int(
            bool(outcome["reproduction_ready_after"])
        )
        self.resource_gain_total += _optional_nonnegative_number(
            outcome,
            "resource_gain",
            "record.outcome.resource_gain",
        )

        attack = _mapping(outcome.get("attack", {}), "record.outcome.attack")
        attempted = _optional_bool(
            attack,
            "attempted",
            "record.outcome.attack.attempted",
        )
        succeeded = _optional_bool(
            attack,
            "success",
            "record.outcome.attack.success",
        )
        killed_target = _optional_bool(
            attack,
            "kill",
            "record.outcome.attack.kill",
        )
        if succeeded and not attempted:
            raise ValueError("attack success cannot be true without an attempt")
        if killed_target and not succeeded:
            raise ValueError("attack kill cannot be true without success")
        self.attack_attempt_count += int(attempted)
        self.attack_success_count += int(succeeded)
        self.attack_kill_count += int(killed_target)
        self.attack_damage_total += _optional_nonnegative_number(
            attack,
            "damage",
            "record.outcome.attack.damage",
        )

        feeding = _mapping(outcome.get("feeding", {}), "record.outcome.feeding")
        ate = _optional_bool(feeding, "ate", "record.outcome.feeding.ate")
        self.feeding_event_count += int(ate)
        if ate:
            food_source = feeding.get("food_source")
            if not isinstance(food_source, str) or not food_source:
                raise ValueError(
                    "record.outcome.feeding.food_source must identify an "
                    "observed source when ate is true"
                )
            self.food_sources.add(food_source)

        drinking = _mapping(outcome.get("drinking", {}), "record.outcome.drinking")
        self.drinking_event_count += int(
            _optional_bool(drinking, "drank", "record.outcome.drinking.drank")
        )

        signal = _mapping(outcome.get("signal", {}), "record.outcome.signal")
        emitted = _optional_bool(signal, "emitted", "record.outcome.signal.emitted")
        self.signal_emission_count += int(emitted)
        if emitted:
            token_id = _nonnegative_int(
                signal.get("token_id"),
                "record.outcome.signal.token_id",
            )
            self.signal_tokens.add(token_id)
        self.signal_energy_cost_total += _optional_nonnegative_number(
            signal,
            "energy_cost",
            "record.outcome.signal.energy_cost",
        )

        passive = _mapping(outcome.get("passive", {}), "record.outcome.passive")
        killed = _optional_bool(passive, "killed", "record.outcome.passive.killed")
        death_cause = passive.get("death_cause")
        if death_cause is not None and not isinstance(death_cause, str):
            raise ValueError(
                "record.outcome.passive.death_cause must be a string or null"
            )
        if bool(outcome["died"]) and death_cause:
            self.death_causes.add(death_cause)
        self.passive_attack_kill_count += int(killed and death_cause == "attack")
        self.passive_attack_damage_total += _optional_nonnegative_number(
            passive,
            "attack_damage_taken",
            "record.outcome.passive.attack_damage_taken",
        )

    def payload(self) -> dict[str, object]:
        return {
            "first_observed_tick": self.first_tick,
            "last_observed_tick": self.last_tick,
            "decision_record_count": self.record_count,
            "observation_invalid_action_count": self.observation_invalid_action_count,
            "resolution_invalid_action_count": self.resolution_invalid_action_count,
            "moved_decision_record_count": self.moved_count,
            "alive_after_decision_record_count": self.alive_after_count,
            "reward_total": _rounded(self.reward_total),
            "mean_reward": _mean(self.reward_total, self.record_count),
            "resource_gain_total": _rounded(self.resource_gain_total),
            "mean_after_energy_ratio": _mean(
                self.after_energy_ratio_total,
                self.after_energy_ratio_count,
            ),
            "mean_after_hydration_ratio": _mean(
                self.after_hydration_ratio_total,
                self.after_hydration_ratio_count,
            ),
            "mean_after_health_ratio": _mean(
                self.after_health_ratio_total,
                self.after_health_ratio_count,
            ),
            "ecology": {
                "reproduction_event_record_count": (
                    self.reproduction_event_record_count
                ),
                "death_event_record_count": self.death_event_record_count,
                "reproduction_ready_after_record_count": (
                    self.reproduction_ready_after_count
                ),
                "feeding_event_count": self.feeding_event_count,
                "drinking_event_count": self.drinking_event_count,
                "food_source_counts": self.food_sources.payload(),
                "death_cause_counts": self.death_causes.payload(),
                "decision_records_by_runtime_species": self.runtime_species.payload(),
                "decision_records_by_runtime_ecotype": self.runtime_ecotypes.payload(),
            },
            "actions": {
                "requested_action_counts": self.requested_actions.payload(),
                "resolved_action_counts": self.resolved_actions.payload(),
                "action_source_counts": self.action_sources.payload(),
            },
            "social": {
                "mate_requested_count": self.mate_requested_count,
                "mate_resolved_count": self.mate_resolved_count,
                "attack_attempt_count": self.attack_attempt_count,
                "attack_success_count": self.attack_success_count,
                "attack_kill_count": self.attack_kill_count,
                "attack_damage_total": _rounded(self.attack_damage_total),
                "passive_attack_kill_count": self.passive_attack_kill_count,
                "passive_attack_damage_total": _rounded(
                    self.passive_attack_damage_total
                ),
                "signal_emission_count": self.signal_emission_count,
                "signal_token_counts": self.signal_tokens.payload(),
                "signal_energy_cost_total": _rounded(self.signal_energy_cost_total),
            },
        }

    def _add_ratio(self, after: Mapping[str, object], key: str) -> None:
        if key not in after:
            return
        value = _finite_number(after[key], f"record.after.{key}")
        if key == "energy_ratio":
            self.after_energy_ratio_total += value
            self.after_energy_ratio_count += 1
        elif key == "hydration_ratio":
            self.after_hydration_ratio_total += value
            self.after_hydration_ratio_count += 1
        else:
            self.after_health_ratio_total += value
            self.after_health_ratio_count += 1


class BoundedOpenEcologyTrajectoryWriter:
    """Stream compact ecology summaries and only bounded, explicit replay windows."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        summary_interval_ticks: int = 100,
        replay_windows: Sequence[ReplayWindow] = (),
        max_replay_rows: int = 2_048,
        max_replay_bytes: int = 64 * 1024 * 1024,
        max_replay_window_ticks: int = 256,
        max_replay_windows: int = 64,
        max_counter_keys: int = 256,
        max_input_record_bytes: int = 4 * 1024 * 1024,
        max_metadata_bytes: int = 4 * 1024 * 1024,
    ):
        self.output_path = Path(output_path)
        self.summary_interval_ticks = _positive_int(
            summary_interval_ticks,
            "summary_interval_ticks",
        )
        self.max_replay_rows = _nonnegative_int(
            max_replay_rows,
            "max_replay_rows",
        )
        self.max_replay_bytes = _nonnegative_int(
            max_replay_bytes,
            "max_replay_bytes",
        )
        self.max_replay_window_ticks = _positive_int(
            max_replay_window_ticks,
            "max_replay_window_ticks",
        )
        self.max_replay_windows = _positive_int(
            max_replay_windows,
            "max_replay_windows",
        )
        self.max_counter_keys = _positive_int(
            max_counter_keys,
            "max_counter_keys",
        )
        self.max_input_record_bytes = _positive_int(
            max_input_record_bytes,
            "max_input_record_bytes",
        )
        self.max_metadata_bytes = _positive_int(
            max_metadata_bytes,
            "max_metadata_bytes",
        )
        self.replay_windows = self._validate_windows(replay_windows)

        self._temp_path: Path | None = None
        self._raw_handle: BinaryIO | None = None
        self._stream_handle: BinaryIO | None = None
        self._started = False
        self._finished = False
        self._aborted = False
        self._last_tick: int | None = None
        self._period_index: int | None = None
        self._period_stats = _EcologyStats(self.max_counter_keys)
        self._lifetime_stats = _EcologyStats(self.max_counter_keys)
        self._record_stream_hash = hashlib.sha256()
        self._content_hash = hashlib.sha256()
        self._logical_bytes_written = 0
        self._header_sha256: str | None = None
        self._config_sha256: str | None = None
        self._contract_sha256: str | None = None
        self._writer_config_sha256: str | None = None
        self._allowed_actions: frozenset[str] | None = None
        self._period_summary_count = 0
        self._replay_rows_written = 0
        self._replay_bytes_written = 0
        self._replay_rows_dropped = 0
        self._peak_counter_keys = 0
        self._abort_cleanup_error: str | None = None

    @property
    def record_count(self) -> int:
        return self._lifetime_stats.record_count

    @property
    def diagnostics(self) -> dict[str, object]:
        status = (
            "finished"
            if self._finished
            else "aborted"
            if self._aborted
            else "open"
            if self._started
            else "not_started"
        )
        return {
            "status": status,
            "input_record_count": self.record_count,
            "period_summary_count": self._period_summary_count,
            "replay_rows_written": self._replay_rows_written,
            "replay_bytes_written": self._replay_bytes_written,
            "replay_rows_dropped_by_hard_bounds": self._replay_rows_dropped,
            "max_replay_rows": self.max_replay_rows,
            "max_replay_bytes": self.max_replay_bytes,
            "max_input_record_bytes": self.max_input_record_bytes,
            "peak_counter_key_count": self._peak_counter_keys,
            "maximum_counter_key_count": self.max_counter_keys * 16,
            "in_memory_full_record_count": 0,
            "logical_output_bytes_written": self._logical_bytes_written,
            "abort_cleanup_error": self._abort_cleanup_error,
        }

    def begin(
        self,
        *,
        run_id: str,
        config: dict[str, object],
        contract: dict[str, object],
    ) -> None:
        if self._started or self._finished or self._aborted:
            raise RuntimeError("open ecology writer is one-shot and already used")
        if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
            raise ValueError("run_id must be a non-empty string of at most 256 chars")

        normalized_config, config_bytes = _canonical_clone(
            config,
            "config",
            max_bytes=self.max_metadata_bytes,
        )
        normalized_contract, contract_bytes = _canonical_clone(
            contract,
            "contract",
            max_bytes=self.max_metadata_bytes,
        )
        self._validate_contract(normalized_contract)
        self._config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self._contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
        writer_config = self._writer_config_payload()
        writer_config_bytes = _canonical_bytes(writer_config)
        self._writer_config_sha256 = hashlib.sha256(writer_config_bytes).hexdigest()

        header: dict[str, object] = {
            "type": "header",
            "format": OPEN_ECOLOGY_STREAM_FORMAT,
            "compression": "gzip" if self.output_path.suffix == ".gz" else "none",
            "run_id": run_id,
            "config": normalized_config,
            "trajectory_contract": normalized_contract,
            "writer_config": writer_config,
            "source_binding": {
                "config_sha256": self._config_sha256,
                "trajectory_contract_sha256": self._contract_sha256,
                "writer_config_sha256": self._writer_config_sha256,
            },
        }
        header_bytes = _canonical_bytes(header)
        self._header_sha256 = hashlib.sha256(header_bytes).hexdigest()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with NamedTemporaryFile(
                "wb",
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                self._temp_path = Path(temp_file.name)
            self._raw_handle = self._temp_path.open("wb")
            if self.output_path.suffix == ".gz":
                self._stream_handle = gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=self._raw_handle,
                    mtime=0,
                )
            else:
                self._stream_handle = self._raw_handle
            self._started = True
            self._write_payload(header)
        except Exception:
            self.abort()
            raise

    def write_record(self, record: dict[str, object]) -> None:
        self._ensure_open()
        try:
            record_bytes = _canonical_bytes(record)
            if len(record_bytes) > self.max_input_record_bytes:
                raise ValueError(
                    "trajectory record exceeds max_input_record_bytes "
                    f"({len(record_bytes)} > {self.max_input_record_bytes})"
                )
            self._validate_record(record)
            tick = int(record["tick"])
            if self._last_tick is not None and tick < self._last_tick:
                raise ValueError(
                    "trajectory record ticks must be monotonically nondecreasing"
                )

            period_index = tick // self.summary_interval_ticks
            if self._period_index is None:
                self._period_index = period_index
            elif period_index != self._period_index:
                self._flush_period()
                self._period_index = period_index

            self._record_stream_hash.update(len(record_bytes).to_bytes(8, "big"))
            self._record_stream_hash.update(record_bytes)
            self._period_stats.update(record)
            self._lifetime_stats.update(record)
            self._last_tick = tick
            self._peak_counter_keys = max(
                self._peak_counter_keys,
                self._period_stats.counter_key_count
                + self._lifetime_stats.counter_key_count,
            )
            self._write_replay_record_if_selected(record, tick)
        except Exception:
            self.abort()
            raise

    def finish(self, *, summary: dict[str, object]) -> None:
        if self._finished:
            return
        self._ensure_open()
        try:
            _, summary_bytes = _canonical_clone(
                summary,
                "summary",
                max_bytes=self.max_metadata_bytes,
            )
            self._flush_period()
            pre_footer_content_sha256 = self._content_hash.hexdigest()
            footer: dict[str, object] = {
                "type": "footer",
                "format": OPEN_ECOLOGY_STREAM_FORMAT,
                "lifetime_summary": self._lifetime_stats.payload(),
                "bounded_output_diagnostics": {
                    **self.diagnostics,
                    "status": "finishing",
                    "logical_output_bytes_before_footer": (self._logical_bytes_written),
                },
                "source_binding": {
                    "config_sha256": self._config_sha256,
                    "trajectory_contract_sha256": self._contract_sha256,
                    "writer_config_sha256": self._writer_config_sha256,
                    "header_sha256": self._header_sha256,
                    "canonical_source_record_stream_sha256": (
                        self._record_stream_hash.hexdigest()
                    ),
                    "run_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                    "pre_footer_content_sha256": pre_footer_content_sha256,
                    "content_digest_scope": (
                        "canonical JSONL bytes from header through final "
                        "period/replay record, excluding footer"
                    ),
                },
            }
            self._write_payload(footer, include_in_content_digest=False)
            self._close_handles(durable=True)
            if self._temp_path is None:
                raise RuntimeError("open ecology writer temp path is missing")
            os.replace(self._temp_path, self.output_path)
            self._temp_path = None
            _fsync_directory_best_effort(self.output_path.parent)
            self._finished = True
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        try:
            self._close_handles(durable=False)
        except Exception as error:  # Cleanup must not mask the triggering failure.
            self._abort_cleanup_error = f"{type(error).__name__}: {error}"
        if self._temp_path is not None:
            try:
                self._temp_path.unlink(missing_ok=True)
            except OSError as error:
                self._abort_cleanup_error = f"{type(error).__name__}: {error}"
            else:
                self._temp_path = None
        if not self._finished:
            self._aborted = True

    def __enter__(self) -> BoundedOpenEcologyTrajectoryWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None or not self._finished:
            self.abort()

    def _validate_windows(
        self,
        windows: Sequence[ReplayWindow],
    ) -> tuple[ReplayWindow, ...]:
        if len(windows) > self.max_replay_windows:
            raise ValueError(
                f"replay_windows exceeds max_replay_windows ({len(windows)} > "
                f"{self.max_replay_windows})"
            )
        validated: list[ReplayWindow] = []
        seen_ids: set[str] = set()
        for window in windows:
            if not isinstance(window, ReplayWindow):
                raise TypeError("replay_windows must contain ReplayWindow values")
            if (
                not window.window_id
                or len(window.window_id) > 128
                or window.window_id in seen_ids
            ):
                raise ValueError(
                    "replay window ids must be unique non-empty strings of at "
                    "most 128 chars"
                )
            start_tick = _nonnegative_int(window.start_tick, "window.start_tick")
            end_tick = _nonnegative_int(window.end_tick, "window.end_tick")
            if end_tick < start_tick:
                raise ValueError("replay window end_tick must be >= start_tick")
            if end_tick - start_tick + 1 > self.max_replay_window_ticks:
                raise ValueError(
                    "replay window exceeds max_replay_window_ticks "
                    f"({end_tick - start_tick + 1} > "
                    f"{self.max_replay_window_ticks})"
                )
            validated.append(window)
            seen_ids.add(window.window_id)
        return tuple(
            sorted(
                validated,
                key=lambda item: (item.start_tick, item.end_tick, item.window_id),
            )
        )

    def _validate_contract(self, contract: Mapping[str, object]) -> None:
        schema_version = contract.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version:
            raise ValueError("contract.schema_version must be a non-empty string")
        record_fields = contract.get("record_fields")
        if not isinstance(record_fields, list) or not all(
            isinstance(field_name, str) for field_name in record_fields
        ):
            raise ValueError("contract.record_fields must be a list of strings")
        missing = sorted(_REQUIRED_RECORD_FIELDS.difference(record_fields))
        if missing:
            raise ValueError(
                f"contract.record_fields is missing required fields: {missing}"
            )

        action_contract = contract.get("action_contract")
        if isinstance(action_contract, dict):
            actions = action_contract.get("actions")
            if isinstance(actions, list):
                keys = {
                    str(action["key"])
                    for action in actions
                    if isinstance(action, dict)
                    and isinstance(action.get("key"), str)
                    and action["key"]
                }
                if keys:
                    self._allowed_actions = frozenset(keys)

    def _validate_record(self, record: Mapping[str, object]) -> None:
        missing = sorted(_REQUIRED_RECORD_FIELDS.difference(record))
        if missing:
            raise ValueError(f"trajectory record is missing required fields: {missing}")
        _nonnegative_int(record["tick"], "record.tick")
        _nonnegative_int(record["agent_id"], "record.agent_id")

        requested_action = record["requested_action"]
        resolved_action = record["resolved_action"]
        action_source = record["action_source"]
        for value, label in (
            (requested_action, "record.requested_action"),
            (resolved_action, "record.resolved_action"),
            (action_source, "record.action_source"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if self._allowed_actions is not None:
            for value, label in (
                (requested_action, "record.requested_action"),
                (resolved_action, "record.resolved_action"),
            ):
                if value not in self._allowed_actions:
                    raise ValueError(f"{label} is not declared by the action contract")

        action_mask = _bool_mapping(record["action_mask"], "record.action_mask")
        resolution_action_mask = _bool_mapping(
            record["resolution_action_mask"],
            "record.resolution_action_mask",
        )
        if requested_action not in action_mask:
            raise ValueError("requested_action is absent from record.action_mask")
        if requested_action not in resolution_action_mask:
            raise ValueError(
                "requested_action is absent from record.resolution_action_mask"
            )
        for key in ("action_valid", "resolution_action_valid", "moved"):
            if type(record[key]) is not bool:
                raise ValueError(f"record.{key} must be a bool")
        if bool(record["action_valid"]) != action_mask[requested_action]:
            raise ValueError("record.action_valid does not match record.action_mask")
        if (
            bool(record["resolution_action_valid"])
            != resolution_action_mask[requested_action]
        ):
            raise ValueError(
                "record.resolution_action_valid does not match "
                "record.resolution_action_mask"
            )

        before = _mapping(record["before"], "record.before")
        after = _mapping(record["after"], "record.after")
        for state, label in ((before, "record.before"), (after, "record.after")):
            if "alive" not in state or type(state["alive"]) is not bool:
                raise ValueError(f"{label}.alive must be a bool")

        outcome = _mapping(record["outcome"], "record.outcome")
        for key in ("reproduced", "died", "reproduction_ready_after"):
            if key not in outcome or type(outcome[key]) is not bool:
                raise ValueError(f"record.outcome.{key} must be a bool")
        if bool(outcome["died"]) == bool(after["alive"]):
            raise ValueError(
                "record.outcome.died must be the inverse of record.after.alive"
            )
        if (
            "requested_action" in outcome
            and outcome["requested_action"] != requested_action
        ):
            raise ValueError(
                "record.outcome.requested_action does not match the record"
            )
        if (
            "resolved_action" in outcome
            and outcome["resolved_action"] != resolved_action
        ):
            raise ValueError("record.outcome.resolved_action does not match the record")
        reward = _mapping(record["reward"], "record.reward")
        if "total" not in reward:
            raise ValueError("record.reward.total is required")
        _finite_number(reward["total"], "record.reward.total")

    def _writer_config_payload(self) -> dict[str, object]:
        return {
            "summary_interval_ticks": self.summary_interval_ticks,
            "replay_windows": [window.to_dict() for window in self.replay_windows],
            "max_replay_rows": self.max_replay_rows,
            "max_replay_bytes": self.max_replay_bytes,
            "max_replay_window_ticks": self.max_replay_window_ticks,
            "max_replay_windows": self.max_replay_windows,
            "max_counter_keys": self.max_counter_keys,
            "max_input_record_bytes": self.max_input_record_bytes,
            "max_metadata_bytes": self.max_metadata_bytes,
            "replay_overflow_policy": "truncate_and_report",
            "record_retention_policy": (
                "full records only inside explicit replay windows"
            ),
        }

    def _write_replay_record_if_selected(
        self,
        record: Mapping[str, object],
        tick: int,
    ) -> None:
        window_ids = [
            window.window_id
            for window in self.replay_windows
            if window.start_tick <= tick <= window.end_tick
        ]
        if not window_ids:
            return
        payload: dict[str, object] = {
            "type": "replay_record",
            "window_ids": window_ids,
            "record": dict(record),
        }
        line_bytes = _canonical_line_bytes(payload)
        if (
            self._replay_rows_written >= self.max_replay_rows
            or self._replay_bytes_written + len(line_bytes) > self.max_replay_bytes
        ):
            self._replay_rows_dropped += 1
            return
        self._write_line_bytes(line_bytes)
        self._replay_rows_written += 1
        self._replay_bytes_written += len(line_bytes)

    def _flush_period(self) -> None:
        if self._period_stats.record_count == 0 or self._period_index is None:
            return
        self._write_payload(
            {
                "type": "period_summary",
                "period_index": self._period_index,
                "summary_interval_ticks": self.summary_interval_ticks,
                "metrics": self._period_stats.payload(),
            }
        )
        self._period_summary_count += 1
        self._period_stats = _EcologyStats(self.max_counter_keys)

    def _write_payload(
        self,
        payload: Mapping[str, object],
        *,
        include_in_content_digest: bool = True,
    ) -> None:
        self._write_line_bytes(
            _canonical_line_bytes(payload),
            include_in_content_digest=include_in_content_digest,
        )

    def _write_line_bytes(
        self,
        line_bytes: bytes,
        *,
        include_in_content_digest: bool = True,
    ) -> None:
        self._ensure_open()
        assert self._stream_handle is not None
        self._stream_handle.write(line_bytes)
        self._logical_bytes_written += len(line_bytes)
        if include_in_content_digest:
            self._content_hash.update(line_bytes)

    def _ensure_open(self) -> None:
        if (
            not self._started
            or self._finished
            or self._aborted
            or self._stream_handle is None
        ):
            raise RuntimeError("open ecology writer is not open")

    def _close_handles(self, *, durable: bool) -> None:
        stream_handle = self._stream_handle
        raw_handle = self._raw_handle
        self._stream_handle = None
        self._raw_handle = None
        if stream_handle is not None and stream_handle is not raw_handle:
            stream_handle.close()
        if raw_handle is not None:
            if durable and not raw_handle.closed:
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            raw_handle.close()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _bool_mapping(value: object, label: str) -> dict[str, bool]:
    mapping = _mapping(value, label)
    if not mapping or not all(type(item) is bool for item in mapping.values()):
        raise ValueError(f"{label} must be a non-empty object of bool values")
    return {key: bool(item) for key, item in mapping.items()}


def _optional_bool(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> bool:
    if key not in mapping:
        return False
    if type(mapping[key]) is not bool:
        raise ValueError(f"{label} must be a bool")
    return bool(mapping[key])


def _optional_finite_number(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float:
    if key not in mapping:
        return 0.0
    return _finite_number(mapping[key], label)


def _optional_nonnegative_number(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float:
    value = _optional_finite_number(mapping, key, label)
    if value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _canonical_clone(
    value: object,
    label: str,
    *,
    max_bytes: int,
) -> tuple[dict[str, object], bytes]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    encoded = _canonical_bytes(value)
    if len(encoded) > max_bytes:
        raise ValueError(
            f"{label} exceeds max_metadata_bytes ({len(encoded)} > {max_bytes})"
        )
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must normalize to an object")
    return normalized, encoded


def _canonical_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error
    return payload.encode("utf-8")


def _canonical_line_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _mean(total: float, count: int) -> float:
    return _rounded(total / count) if count else 0.0


def _fsync_directory_best_effort(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)
