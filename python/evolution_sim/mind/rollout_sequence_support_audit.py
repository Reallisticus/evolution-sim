from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.current_route_decision import (
    CURRENT_CLOSED_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V136_REPORT_PATH,
    MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
    NEXT_ALLOWED_RESEARCH_DIRECTION,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_INDEX_FIELD,
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
    TrajectoryJsonlDataset,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _resolve_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import (
    MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
    RolloutContextConfig,
    RolloutContextState,
    rollout_context_feature_contract,
)

MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION = (
    "mind_v3_v137_rollout_sequence_support_audit_v1"
)
MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_POLICY = (
    "diagnostics_only_v137_public_rollout_sequence_history_support_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v137-rollout-sequence-support-audit.json"
)

SEQUENCE_MODEL_ID = "public_rollout_sequence_history_lookup"
ACTION_ONLY_BASELINE_ID = "action_only_majority_baseline"
ACTION_ORDER_BASELINE_ID = "action_order_baseline"

STRICT_HELDOUT_SEEDS = (5, 13, 19, 29, 37, 41)
DEFAULT_HELDOUT_FRACTION = 0.25
DOMINANT_PREDICTED_ACTION_SHARE_MAX = 0.50

_SEED_PATTERN = re.compile(r"(?:seed[-_=]?|mind-v3-|heuristic-)(\d+)")
_TRAJECTORY_PATH_SUFFIXES = (".jsonl", ".jsonl.gz")

_FORBIDDEN_ROW_KEYS = frozenset(
    {
        "future_row",
        "future_rows",
        "future_trajectory",
        "future_trajectories",
        "fixture",
        "fixture_id",
        "fixture_identity",
        "fixture_name",
        "heuristic_action",
        "heuristic_recommendation",
        "private",
        "private_world_state",
        "provenance",
        "seed",
        "seed_identity",
        "simulation_world",
        "world",
        "world_state",
    }
)
_FORBIDDEN_TRAINABLE_FEATURE_TOKENS = (
    "future",
    "fixture",
    "heuristic",
    "private",
    "provenance",
    "seed",
    "world",
)


class RolloutSequenceSupportAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RolloutSequenceSupportAuditBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoadedTrajectories:
    datasets: tuple[TrajectoryJsonlDataset, ...]
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Example:
    row_id: str
    label: str
    valid_actions: tuple[str, ...]
    sequence_keys: tuple[str, ...]
    action_source: str
    source_path: str
    source_seed: int | None
    split: str


def build_rollout_sequence_support_audit_report(
    *,
    v136_report: Mapping[str, object] | None = None,
    v136_report_path: str | Path | None = DEFAULT_V136_REPORT_PATH,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    trajectory_paths: Sequence[str | Path] = (),
    report_paths: Sequence[str | Path] = (),
    heldout_seed_values: Sequence[int] = STRICT_HELDOUT_SEEDS,
    heldout_source_patterns: Sequence[str] = (),
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
    context_config: RolloutContextConfig | None = None,
) -> RolloutSequenceSupportAuditBuild:
    if heldout_fraction < 0.0 or heldout_fraction >= 1.0:
        raise RolloutSequenceSupportAuditError("heldout_fraction must be in [0.0, 1.0)")

    v136_payload, v136_evidence = _resolve_json_report(
        v136_report,
        v136_report_path,
        expected_schema=MIND_V3_CURRENT_ROUTE_DECISION_SCHEMA_VERSION,
    )
    trajectories = _resolve_trajectories(
        trajectory_datasets=trajectory_datasets,
        trajectory_paths=trajectory_paths,
        report_paths=report_paths,
    )
    config = context_config or RolloutContextConfig()
    records = _records_from_datasets(trajectories.datasets)
    leakage_scan = _leakage_scan(records)
    examples = _examples_from_records(
        records,
        config=config,
        heldout_seed_values=heldout_seed_values,
        heldout_source_patterns=heldout_source_patterns,
        heldout_fraction=heldout_fraction,
    )
    evaluation = _evaluate_examples(
        examples,
        heldout_seed_values=heldout_seed_values,
    )
    support_floors = _support_floors(
        source_integrity_passed=None,
        leakage_scan=leakage_scan,
        evaluation=evaluation,
    )
    source_integrity = _source_integrity(
        v136_report=v136_payload,
        v136_evidence=v136_evidence,
        trajectories=trajectories.evidence,
        leakage_scan=leakage_scan,
    )
    support_floors = _support_floors(
        source_integrity_passed=bool(source_integrity["passed"]),
        leakage_scan=leakage_scan,
        evaluation=evaluation,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
    )
    contract = _contract(config)
    report = {
        "schema_version": MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_POLICY,
        "contract": contract,
        "provenance": {"contract_digest": stable_payload_digest(contract)},
        "source_integrity": source_integrity,
        "source_evidence": {
            "v136_report": v136_evidence,
            "trajectories": trajectories.evidence,
        },
        "leakage_scan": leakage_scan,
        "train_heldout_split": evaluation["train_heldout_split"],
        "support_floors": support_floors,
        "baseline_comparisons": evaluation["baseline_comparisons"],
        "model_summaries": evaluation["model_summaries"],
        "action_collapse_diagnostics": evaluation["action_collapse_diagnostics"],
        "classification": classification,
        "recommendation": _recommendation(classification),
        "authorization_block": _authorization_block(),
        "non_promoted": True,
    }
    return RolloutSequenceSupportAuditBuild(report=report)


def write_rollout_sequence_support_audit_report(
    build: RolloutSequenceSupportAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract(config: RolloutContextConfig) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "report_only": True,
        "runtime_policy_effect": "none",
        "trainer_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "training_executed": False,
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "first_recovery_public_context_ranker_reopened": False,
        "required_v136_next_allowed_research_direction": (
            NEXT_ALLOWED_RESEARCH_DIRECTION
        ),
        "closed_first_recovery_path": CURRENT_CLOSED_PATH,
        "feature_policy": {
            "model": "deterministic_majority_lookup_diagnostic_only_v1",
            "sequence_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "rollout_context": rollout_context_feature_contract(config),
            "current_decision_inputs": ["action_mask"],
            "prior_history_inputs": [
                "same-agent finalized public trajectory rows before current decision"
            ],
            "excluded_trainable_inputs": [
                "private world state",
                "seed identity",
                "fixture identity",
                "heuristic recommendations",
                "future rows",
                "current-row outcome",
                "replay outcomes unavailable at decision time",
                "provenance",
            ],
        },
    }


def _source_integrity(
    *,
    v136_report: Mapping[str, object] | None,
    v136_evidence: Mapping[str, object],
    trajectories: Mapping[str, object],
    leakage_scan: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if v136_evidence.get("loaded") is not True:
        failures.append("missing_v136_report")
    elif v136_evidence.get("schema_matches") is not True:
        failures.append("v136_schema_mismatch")
    v136_source = _mapping(_mapping(v136_report).get("source_integrity"))
    if v136_source.get("passed") is not True:
        failures.append("v136_source_integrity_not_passed")
    if _mapping(v136_report).get("next_allowed_research_direction") != (
        NEXT_ALLOWED_RESEARCH_DIRECTION
    ):
        failures.append("v136_next_allowed_research_direction_mismatch")
    if _mapping(v136_report).get("current_closed_path") != CURRENT_CLOSED_PATH:
        failures.append("v136_current_closed_path_mismatch")
    if _int(trajectories.get("loaded_path_count")) <= 0:
        failures.append("no_trajectory_inputs_loaded")
    if _int(trajectories.get("record_count")) <= 0:
        failures.append("no_trajectory_records_loaded")
    if _int(trajectories.get("load_failure_count")) > 0:
        failures.append("trajectory_load_failures")
    if leakage_scan.get("passed") is not True:
        failures.append("leakage_scan_failed")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v136_source_integrity_passed": v136_source.get("passed"),
        "v136_next_allowed_research_direction": _mapping(v136_report).get(
            "next_allowed_research_direction"
        ),
        "required_v136_next_allowed_research_direction": (
            NEXT_ALLOWED_RESEARCH_DIRECTION
        ),
        "v136_current_closed_path": _mapping(v136_report).get("current_closed_path"),
        "required_current_closed_path": CURRENT_CLOSED_PATH,
        "trajectory_loaded_path_count": trajectories.get("loaded_path_count"),
        "trajectory_record_count": trajectories.get("record_count"),
        "leakage_scan_passed": leakage_scan.get("passed"),
    }


def _resolve_trajectories(
    *,
    trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None,
    trajectory_paths: Sequence[str | Path],
    report_paths: Sequence[str | Path],
) -> _LoadedTrajectories:
    datasets: list[TrajectoryJsonlDataset] = list(trajectory_datasets or ())
    failures: list[dict[str, object]] = []
    report_evidence = _report_path_evidence(report_paths)
    resolved_paths = list(trajectory_paths)
    for path in report_evidence["extracted_trajectory_paths"]:
        resolved_paths.append(Path(str(path)))

    for path in _unique_paths(resolved_paths):
        try:
            datasets.append(_load_trajectory_jsonl_lenient(path))
        except (OSError, json.JSONDecodeError, RolloutSequenceSupportAuditError) as exc:
            failures.append(
                {
                    "path": str(path),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )

    return _LoadedTrajectories(
        datasets=tuple(datasets),
        evidence={
            "requested_trajectory_paths": [str(path) for path in trajectory_paths],
            "requested_report_paths": [str(path) for path in report_paths],
            "report_path_evidence": report_evidence,
            "loaded_path_count": len(datasets),
            "loaded_paths": [str(dataset.path) for dataset in datasets],
            "record_count": sum(dataset.record_count for dataset in datasets),
            "load_failure_count": len(failures),
            "load_failures": failures,
        },
    )


def _report_path_evidence(report_paths: Sequence[str | Path]) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    extracted: list[str] = []
    for path_like in report_paths:
        path = Path(path_like)
        evidence: dict[str, object] = {"path": str(path), "loaded": False}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            evidence.update({"error": type(exc).__name__, "message": str(exc)})
            reports.append(evidence)
            continue
        evidence["loaded"] = isinstance(payload, Mapping)
        if not isinstance(payload, Mapping):
            evidence["error"] = "report_not_object"
            reports.append(evidence)
            continue
        paths = _extract_trajectory_paths(payload, report_dir=path.parent)
        evidence["schema_version"] = payload.get("schema_version")
        evidence["extracted_trajectory_path_count"] = len(paths)
        evidence["extracted_trajectory_paths"] = paths
        extracted.extend(paths)
        reports.append(evidence)
    return {
        "report_count": len(report_paths),
        "reports": reports,
        "extracted_trajectory_path_count": len(extracted),
        "extracted_trajectory_paths": sorted(set(extracted)),
    }


def _extract_trajectory_paths(
    value: object,
    *,
    report_dir: Path,
) -> list[str]:
    paths: list[str] = []

    def visit(item: object, key_path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, (*key_path, str(key)))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key_path)
            return
        if not isinstance(item, str):
            return
        if not item.endswith(_TRAJECTORY_PATH_SUFFIXES):
            return
        if not any("trajectory" in token for token in key_path):
            return
        path = Path(item)
        if not path.is_absolute() and not path.exists():
            candidate = report_dir / path
            path = candidate if candidate.exists() else path
        paths.append(str(path))

    visit(value, ())
    return sorted(set(paths))


def _load_trajectory_jsonl_lenient(path: str | Path) -> TrajectoryJsonlDataset:
    resolved = Path(path)
    records: list[dict[str, object]] = []
    header: dict[str, object] = {}
    footer: dict[str, object] = {}
    with _open_input(resolved) as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise RolloutSequenceSupportAuditError(
                    f"trajectory row {line_number} must be a JSON object"
                )
            if not header:
                header = dict(payload)
            footer = dict(payload)
            record = payload.get("record")
            if isinstance(record, Mapping):
                records.append(dict(record))
            elif payload.get("type") == "record":
                records.append(dict(payload))
    return TrajectoryJsonlDataset(
        path=resolved,
        header=header,
        records=tuple(records),
        footer=footer,
    )


def _records_from_datasets(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(datasets):
        episode_id = _dataset_episode_id(dataset, dataset_index=dataset_index)
        for record_index, record in enumerate(dataset.records):
            contextual = dict(record)
            contextual[TRAJECTORY_EPISODE_ID_FIELD] = episode_id
            contextual[TRAJECTORY_DATASET_INDEX_FIELD] = dataset_index
            contextual[TRAJECTORY_DATASET_RECORD_INDEX_FIELD] = record_index
            contextual[TRAJECTORY_SOURCE_PATH_FIELD] = str(dataset.path)
            records.append(contextual)
    return tuple(records)


def _dataset_episode_id(
    dataset: TrajectoryJsonlDataset,
    *,
    dataset_index: int,
) -> str:
    footer_provenance = _mapping(dataset.footer.get("provenance"))
    source_seeds = footer_provenance.get("source_seeds")
    split_id = footer_provenance.get("split_id")
    if isinstance(source_seeds, list) and isinstance(split_id, str) and split_id:
        seeds = ",".join(str(seed) for seed in source_seeds)
        return f"{dataset_index}:{split_id}:seed={seeds}:path={dataset.path}"
    return f"{dataset_index}:path={dataset.path}"


def _leakage_scan(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    leaks: list[dict[str, object]] = []
    for index, record in enumerate(records):
        for key in sorted(_FORBIDDEN_ROW_KEYS):
            if key in record:
                leaks.append(
                    {
                        "record_index": index,
                        "field": key,
                        "reason": "forbidden_public_sequence_row_field",
                    }
                )
        for container_key in (
            "features",
            "feature_input",
            "public_history_features",
            "trainable_features",
        ):
            container = record.get(container_key)
            for path in _forbidden_feature_paths(container, prefix=container_key):
                leaks.append(
                    {
                        "record_index": index,
                        "field": path,
                        "reason": "forbidden_trainable_feature_token",
                    }
                )
    return {
        "passed": not leaks,
        "leakage_count": len(leaks),
        "leaks": leaks[:24],
        "forbidden_row_keys": sorted(_FORBIDDEN_ROW_KEYS),
        "forbidden_trainable_feature_tokens": list(
            _FORBIDDEN_TRAINABLE_FEATURE_TOKENS
        ),
    }


def _forbidden_feature_paths(value: object, *, prefix: str) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN_TRAINABLE_FEATURE_TOKENS):
                paths.append(path)
            paths.extend(_forbidden_feature_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_feature_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _examples_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    config: RolloutContextConfig,
    heldout_seed_values: Sequence[int],
    heldout_source_patterns: Sequence[str],
    heldout_fraction: float,
) -> tuple[_Example, ...]:
    states: dict[tuple[str, int], RolloutContextState] = {}
    split_by_source = _split_by_source(
        records,
        heldout_seed_values=heldout_seed_values,
        heldout_source_patterns=heldout_source_patterns,
        heldout_fraction=heldout_fraction,
    )
    examples: list[_Example] = []
    for index, record in enumerate(records):
        episode_id = _episode_id(record, fallback_index=index)
        agent_id = _agent_id(record)
        state = states.setdefault((episode_id, agent_id), RolloutContextState(config))
        snapshot = state.snapshot()
        label = _label(record)
        valid_actions = _valid_actions(record)
        action_source = _action_source(record)
        source_path = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
        source_seed = _source_seed(record, source_path)
        if label in ACTION_NAMES and label in valid_actions and action_source != "passive":
            examples.append(
                _Example(
                    row_id=_row_id(record, fallback_index=index),
                    label=label,
                    valid_actions=valid_actions,
                    sequence_keys=_sequence_keys(
                        state=state,
                        snapshot=snapshot,
                        valid_actions=valid_actions,
                    ),
                    action_source=action_source,
                    source_path=source_path,
                    source_seed=source_seed,
                    split=split_by_source.get(source_path, "train"),
                )
            )
        state.update_from_record(record)
    return tuple(examples)


def _split_by_source(
    records: Sequence[Mapping[str, object]],
    *,
    heldout_seed_values: Sequence[int],
    heldout_source_patterns: Sequence[str],
    heldout_fraction: float,
) -> dict[str, str]:
    sources = sorted(
        {
            str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
            for record in records
        }
    )
    source_seed = _source_seed_by_source(records)
    strict = {int(seed) for seed in heldout_seed_values}
    split: dict[str, str] = {}
    for source in sources:
        seed = source_seed.get(source)
        is_heldout = seed in strict if seed is not None else False
        if not is_heldout:
            is_heldout = any(pattern in source for pattern in heldout_source_patterns)
        if not is_heldout:
            digest = stable_payload_digest(source)
            bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
            is_heldout = bucket < heldout_fraction
        split[source] = "heldout" if is_heldout else "train"
    values = set(split.values())
    if len(sources) >= 2 and values == {"train"}:
        split[sources[-1]] = "heldout"
    if len(sources) >= 2 and values == {"heldout"}:
        split[sources[0]] = "train"
    return split


def _source_seed_by_source(
    records: Sequence[Mapping[str, object]],
) -> dict[str, int | None]:
    source_seed: dict[str, int | None] = {}
    for record in records:
        source = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
        seed = _source_seed(record, source)
        if source not in source_seed or source_seed[source] is None:
            source_seed[source] = seed
    return source_seed


def _sequence_keys(
    *,
    state: RolloutContextState,
    snapshot: Mapping[str, object],
    valid_actions: Sequence[str],
) -> tuple[str, ...]:
    mask = _action_mask_token(valid_actions)
    recent_resolved = _recent_actions(snapshot.get("recent_resolved_actions"))
    recent_requested = _recent_actions(snapshot.get("recent_requested_actions"))
    last_resolved = recent_resolved[-1] if recent_resolved else "none"
    last_requested = recent_requested[-1] if recent_requested else "none"
    return (
        f"mask={mask}|ctx={state.context_key()}",
        f"mask={mask}|coarse={state.coarse_context_key()}",
        f"mask={mask}|last_resolved={last_resolved}|last_requested={last_requested}",
        f"mask={mask}|last_resolved={last_resolved}",
        f"mask={mask}|history_any={bool(recent_resolved or recent_requested)}",
    )


def _evaluate_examples(
    examples: Sequence[_Example],
    *,
    heldout_seed_values: Sequence[int],
) -> dict[str, object]:
    train = tuple(example for example in examples if example.split == "train")
    heldout = tuple(example for example in examples if example.split == "heldout")
    sequence_counts = _sequence_counts(train)
    action_counts = Counter(example.label for example in train)
    sequence_eval = _evaluate(
        heldout,
        predictor=lambda example: _predict_sequence(
            example,
            sequence_counts=sequence_counts,
            action_counts=action_counts,
        ),
    )
    action_only_eval = _evaluate(
        heldout,
        predictor=lambda example: _predict_action_only(
            example,
            action_counts=action_counts,
        ),
    )
    action_order_eval = _evaluate(heldout, predictor=_predict_action_order)
    return {
        "train_heldout_split": _split_summary(
            examples,
            train,
            heldout,
            heldout_seed_values=heldout_seed_values,
        ),
        "model_summaries": {
            SEQUENCE_MODEL_ID: sequence_eval,
            ACTION_ONLY_BASELINE_ID: action_only_eval,
            ACTION_ORDER_BASELINE_ID: action_order_eval,
        },
        "baseline_comparisons": {
            "heldout_accuracy_delta_vs_action_only": _round(
                _float(sequence_eval.get("accuracy"))
                - _float(action_only_eval.get("accuracy"))
            ),
            "heldout_accuracy_delta_vs_action_order": _round(
                _float(sequence_eval.get("accuracy"))
                - _float(action_order_eval.get("accuracy"))
            ),
            "beats_action_only_baseline": (
                _float(sequence_eval.get("accuracy"))
                > _float(action_only_eval.get("accuracy"))
            ),
            "beats_action_order_baseline": (
                _float(sequence_eval.get("accuracy"))
                > _float(action_order_eval.get("accuracy"))
            ),
        },
        "action_collapse_diagnostics": {
            "dominant_predicted_action": sequence_eval.get("dominant_predicted_action"),
            "dominant_predicted_action_share": sequence_eval.get(
                "dominant_predicted_action_share"
            ),
            "dominant_predicted_action_share_max": (
                DOMINANT_PREDICTED_ACTION_SHARE_MAX
            ),
            "passed": (
                _float(sequence_eval.get("dominant_predicted_action_share"))
                <= DOMINANT_PREDICTED_ACTION_SHARE_MAX
            ),
        },
    }


def _sequence_counts(examples: Sequence[_Example]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        for key in example.sequence_keys:
            counts[key][example.label] += 1
    return dict(counts)


def _evaluate(
    examples: Sequence[_Example],
    *,
    predictor,
) -> dict[str, object]:
    correct = 0
    unsupported = 0
    label_counts = Counter[str]()
    predicted_counts = Counter[str]()
    records: list[dict[str, object]] = []
    for example in examples:
        predicted = predictor(example)
        label_counts[example.label] += 1
        predicted_counts[predicted] += 1
        if predicted == example.label:
            correct += 1
        if predicted not in example.valid_actions:
            unsupported += 1
        records.append(
            {
                "row_id": example.row_id,
                "label": example.label,
                "predicted": predicted,
                "source_path": example.source_path,
                "source_seed": example.source_seed,
            }
        )
    total = len(examples)
    dominant_action, dominant_count = _dominant_count(predicted_counts)
    return {
        "heldout_record_count": total,
        "accuracy": _round(correct / float(total) if total else 0.0),
        "correct_count": correct,
        "label_counts": _ordered_counts(label_counts),
        "predicted_action_counts": _ordered_counts(predicted_counts),
        "dominant_predicted_action": {
            "action": dominant_action,
            "count": dominant_count,
            "share": _round(dominant_count / float(total) if total else 0.0),
            "total": total,
        },
        "dominant_predicted_action_share": _round(
            dominant_count / float(total) if total else 0.0
        ),
        "unsupported_action_count": unsupported,
        "unsupported_action_rate": _round(unsupported / float(total) if total else 0.0),
        "examples": records[:12],
    }


def _predict_sequence(
    example: _Example,
    *,
    sequence_counts: Mapping[str, Counter[str]],
    action_counts: Counter[str],
) -> str:
    valid = set(example.valid_actions)
    for key in example.sequence_keys:
        action = _best_counted_action(sequence_counts.get(key), valid)
        if action is not None:
            return action
    return _predict_action_only(example, action_counts=action_counts)


def _predict_action_only(example: _Example, *, action_counts: Counter[str]) -> str:
    action = _best_counted_action(action_counts, set(example.valid_actions))
    if action is not None:
        return action
    return _predict_action_order(example)


def _predict_action_order(example: _Example) -> str:
    valid = set(example.valid_actions)
    for action in ACTION_NAMES:
        if action in valid:
            return action
    return ACTION_NAMES[0]


def _best_counted_action(
    counts: Counter[str] | None,
    valid_actions: set[str],
) -> str | None:
    if not counts:
        return None
    best_action: str | None = None
    best_count = -1
    for action in ACTION_NAMES:
        if action not in valid_actions:
            continue
        count = int(counts.get(action, 0))
        if count > best_count:
            best_count = count
            best_action = action
    if best_action is None or best_count <= 0:
        return None
    return best_action


def _support_floors(
    *,
    source_integrity_passed: bool | None,
    leakage_scan: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    split = _mapping(evaluation.get("train_heldout_split"))
    comparisons = _mapping(evaluation.get("baseline_comparisons"))
    summaries = _mapping(evaluation.get("model_summaries"))
    sequence = _mapping(summaries.get(SEQUENCE_MODEL_ID))
    configured_leakage_count = _int(
        split.get("configured_heldout_seed_leakage_count")
    )
    floors = [
        _floor(
            "source_integrity_passed",
            bool(source_integrity_passed),
            observed=source_integrity_passed,
            required=True,
        ),
        _floor(
            "leakage_scan_passed",
            leakage_scan.get("passed") is True,
            observed=leakage_scan.get("leakage_count"),
            required=0,
        ),
        _floor(
            "heldout_source_split_available",
            _int(split.get("train_source_count")) > 0
            and _int(split.get("heldout_source_count")) > 0,
            observed={
                "train_source_count": split.get("train_source_count"),
                "heldout_source_count": split.get("heldout_source_count"),
            },
            required="train_source_count>0 and heldout_source_count>0",
        ),
        _floor(
            "beats_action_only_baseline",
            comparisons.get("beats_action_only_baseline") is True,
            observed=comparisons.get("heldout_accuracy_delta_vs_action_only"),
            required=">0",
        ),
        _floor(
            "beats_action_order_baseline",
            comparisons.get("beats_action_order_baseline") is True,
            observed=comparisons.get("heldout_accuracy_delta_vs_action_order"),
            required=">0",
        ),
        _floor(
            "dominant_predicted_action_share_lte_0_50",
            _float(sequence.get("dominant_predicted_action_share"))
            <= DOMINANT_PREDICTED_ACTION_SHARE_MAX,
            observed=sequence.get("dominant_predicted_action_share"),
            required=DOMINANT_PREDICTED_ACTION_SHARE_MAX,
        ),
        _floor(
            "unsupported_action_count_eq_0",
            _int(sequence.get("unsupported_action_count")) == 0,
            observed=sequence.get("unsupported_action_count"),
            required=0,
        ),
        _floor(
            "configured_heldout_seed_leakage_count_eq_0",
            configured_leakage_count == 0,
            observed=configured_leakage_count,
            required=0,
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "v137_rollout_sequence_support_floors_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "floors": floors,
    }


def _floor(
    name: str,
    passed: bool,
    *,
    observed: object,
    required: object,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    support_floors: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        primary = "rollout_sequence_support_audit_source_integrity_failed"
    elif support_floors.get("passed") is True:
        primary = "rollout_sequence_support_ready_for_future_sequence_world_model_scorer"
    else:
        primary = "rollout_sequence_support_blocked"
    return {
        "primary": primary,
        "labels": [primary],
        "first_failed_floor": support_floors.get("first_failed_floor"),
        "allowed_classifications": [
            "rollout_sequence_support_audit_source_integrity_failed",
            "rollout_sequence_support_blocked",
            "rollout_sequence_support_ready_for_future_sequence_world_model_scorer",
        ],
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    primary = classification.get("primary")
    ready = primary == "rollout_sequence_support_ready_for_future_sequence_world_model_scorer"
    return {
        "next_step": (
            "future_sequence_or_world_model_scorer_diagnostic_allowed"
            if ready
            else "stop_before_sequence_world_model_training"
        ),
        "future_sequence_world_model_scorer_justified": ready,
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "first_recovery_public_context_ranker_path_reopened": False,
        "first_failed_floor": classification.get("first_failed_floor"),
    }


def _authorization_block() -> dict[str, object]:
    return {
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "runtime_promotion_authorized": False,
        "first_recovery_public_context_ranker_reopened": False,
        "gate_change_authorized": False,
        "viewer_change_authorized": False,
        "replay_golden_change_authorized": False,
        "foundation_change_authorized": False,
    }


def _split_summary(
    examples: Sequence[_Example],
    train: Sequence[_Example],
    heldout: Sequence[_Example],
    *,
    heldout_seed_values: Sequence[int],
) -> dict[str, object]:
    configured_heldout_seeds = sorted({int(seed) for seed in heldout_seed_values})
    configured_heldout_seed_set = set(configured_heldout_seeds)
    train_sources = {example.source_path for example in train}
    heldout_sources = {example.source_path for example in heldout}
    train_configured_heldout = [
        example
        for example in train
        if example.source_seed in configured_heldout_seed_set
    ]
    return {
        "policy": "strict_seed_or_pattern_else_stable_source_hash_v1",
        "configured_heldout_seeds": configured_heldout_seeds,
        "heldout_seed_values": configured_heldout_seeds,
        "example_count": len(examples),
        "train_record_count": len(train),
        "heldout_record_count": len(heldout),
        "train_source_count": len(train_sources),
        "heldout_source_count": len(heldout_sources),
        "train_sources_sample": sorted(train_sources)[:12],
        "heldout_sources_sample": sorted(heldout_sources)[:12],
        "configured_heldout_seed_leakage_count": len(train_configured_heldout),
        "configured_heldout_seed_leakage_examples": [
            {
                "row_id": example.row_id,
                "source_path": example.source_path,
                "source_seed": example.source_seed,
            }
            for example in train_configured_heldout[:12]
        ],
        "strict_heldout_seeds": configured_heldout_seeds,
        "strict_seed_leakage_count": len(train_configured_heldout),
        "strict_seed_leakage_examples": [
            {
                "row_id": example.row_id,
                "source_path": example.source_path,
                "source_seed": example.source_seed,
            }
            for example in train_configured_heldout[:12]
        ],
    }


def _row_id(record: Mapping[str, object], *, fallback_index: int) -> str:
    source = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
    episode_id = _episode_id(record, fallback_index=fallback_index)
    record_index = record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD)
    if isinstance(record_index, int) and not isinstance(record_index, bool):
        return f"{source}:{episode_id}:record={record_index}"
    return f"{source}:{episode_id}:record={fallback_index}"


def _episode_id(record: Mapping[str, object], *, fallback_index: int) -> str:
    episode_id = record.get(TRAJECTORY_EPISODE_ID_FIELD)
    if isinstance(episode_id, str) and episode_id:
        return episode_id
    return f"inferred:{fallback_index}"


def _agent_id(record: Mapping[str, object]) -> int:
    agent_id = record.get("agent_id")
    if isinstance(agent_id, bool) or not isinstance(agent_id, int):
        return -1
    return int(agent_id)


def _label(record: Mapping[str, object]) -> str:
    requested = record.get("requested_action")
    resolved = record.get("resolved_action")
    if record.get("resolution_action_valid", True) is True and requested in ACTION_NAMES:
        return str(requested)
    if resolved in ACTION_NAMES:
        return str(resolved)
    return str(requested) if requested in ACTION_NAMES else "stay"


def _valid_actions(record: Mapping[str, object]) -> tuple[str, ...]:
    action_mask = record.get("action_mask")
    if not isinstance(action_mask, Mapping):
        return ACTION_NAMES
    actions = tuple(
        action for action in ACTION_NAMES if bool(action_mask.get(action, False))
    )
    return actions or ACTION_NAMES


def _action_source(record: Mapping[str, object]) -> str:
    source = record.get("action_source")
    return source if isinstance(source, str) and source else "unknown"


def _source_seed(record: Mapping[str, object], source_path: str) -> int | None:
    for value in (
        source_path,
        str(record.get(TRAJECTORY_EPISODE_ID_FIELD, "")),
    ):
        match = _SEED_PATTERN.search(value)
        if match is not None:
            return int(match.group(1))
    return None


def _action_mask_token(valid_actions: Sequence[str]) -> str:
    valid = set(valid_actions)
    return "".join("1" if action in valid else "0" for action in ACTION_NAMES)


def _recent_actions(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item) in ACTION_NAMES)


def _ordered_counts(counts: Counter[str]) -> dict[str, int]:
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    action = max(ACTION_NAMES, key=lambda item: (int(counts.get(item, 0)), item))
    return action, int(counts.get(action, 0))


def _unique_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(Path(path) for path in paths))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return int(value)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
