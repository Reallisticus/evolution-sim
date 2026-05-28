from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
    TrajectoryJsonlDataset,
    load_trajectory_jsonl,
    records_with_trajectory_context,
)
from evolution_sim.mind.feature_policy import feature_keys_from_record
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import (
    MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
    RolloutContextConfig,
    RolloutContextState,
    rollout_context_feature_contract,
)

MIND_V3_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION = "mind_v3_rollout_context_audit_v1"
MIND_V3_ROLLOUT_CONTEXT_AUDIT_POLICY = (
    "ecological_lookup_vs_rollout_context_lookup_v1"
)
ECOLOGICAL_ONLY_MODEL_ID = "ecological_only"
ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID = "ecological_plus_rollout_context"
DEFAULT_HELDOUT_FRACTION = 0.2
DEFAULT_MIN_EAT_OVERPREDICTION_RATE_REDUCTION = 0.05
_SEED_PATTERN = re.compile(r"(?:seed[-_=]|seed=)(\d+)")


class RolloutContextAuditError(ValueError):
    pass


def load_rollout_context_audit_datasets(
    paths: Sequence[str | Path],
) -> tuple[TrajectoryJsonlDataset, ...]:
    datasets = tuple(load_trajectory_jsonl(path) for path in paths)
    if not datasets:
        raise RolloutContextAuditError("at least one trajectory path is required")
    return datasets


def build_rollout_context_audit_report(
    records: Sequence[Mapping[str, object]],
    *,
    source_paths: Sequence[str | Path] | None = None,
    heldout_seed_values: Iterable[int] | None = None,
    heldout_source_patterns: Sequence[str] = (),
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
    min_eat_overprediction_rate_reduction: float = (
        DEFAULT_MIN_EAT_OVERPREDICTION_RATE_REDUCTION
    ),
    context_config: RolloutContextConfig | None = None,
) -> dict[str, object]:
    config = context_config or RolloutContextConfig()
    resolved_records = [dict(record) for record in records]
    if not resolved_records:
        raise RolloutContextAuditError("audit records must not be empty")
    heldout_seeds = (
        {int(seed) for seed in heldout_seed_values}
        if heldout_seed_values is not None
        else set()
    )
    if heldout_fraction < 0.0 or heldout_fraction >= 1.0:
        raise RolloutContextAuditError("heldout_fraction must be in [0.0, 1.0)")
    if min_eat_overprediction_rate_reduction < 0.0:
        raise RolloutContextAuditError(
            "min_eat_overprediction_rate_reduction must be non-negative"
        )

    contextual_rows = _contextual_rows(resolved_records, config=config)
    examples = [
        row
        for row in contextual_rows
        if row.label in ACTION_NAMES
        and row.label in row.valid_actions
        and row.action_source != "passive"
    ]
    if len(examples) < 2:
        raise RolloutContextAuditError("audit needs at least two policy action rows")

    split = _assign_split(
        examples,
        heldout_seeds=heldout_seeds,
        heldout_source_patterns=heldout_source_patterns,
        heldout_fraction=heldout_fraction,
    )
    train_examples = [example for example in examples if split[example.row_id] == "train"]
    heldout_examples = [
        example for example in examples if split[example.row_id] == "heldout"
    ]
    if not train_examples or not heldout_examples:
        raise RolloutContextAuditError(
            "train/held-out split must include at least one row on each side"
        )

    ecological_counts = _train_counts(
        train_examples,
        key_policy=ECOLOGICAL_ONLY_MODEL_ID,
    )
    context_counts = _train_counts(
        train_examples,
        key_policy=ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
    )
    ecological_eval = _evaluate_counts(
        heldout_examples,
        ecological_counts,
        key_policy=ECOLOGICAL_ONLY_MODEL_ID,
    )
    context_eval = _evaluate_counts(
        heldout_examples,
        context_counts,
        key_policy=ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID,
    )
    eat_delta = _eat_overprediction_delta(ecological_eval, context_eval)
    material = _material_improvement(
        eat_delta,
        ecological_eval,
        context_eval,
        min_rate_reduction=min_eat_overprediction_rate_reduction,
    )
    return {
        "schema_version": MIND_V3_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_ROLLOUT_CONTEXT_AUDIT_POLICY,
        "source_trajectories": _source_summary(
            resolved_records,
            source_paths=source_paths,
        ),
        "feature_contract": {
            "ecological_feature_policy": "mind_feature_policy_v2",
            "rollout_context_feature_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "rollout_context": rollout_context_feature_contract(config),
            "classifier": {
                "policy": "deterministic_majority_lookup_with_backoff_v1",
                "tie_break": "ACTION_NAMES order",
                "label": (
                    "requested_action when resolution_action_valid is true, "
                    "otherwise resolved_action"
                ),
            },
        },
        "train_heldout_split": _split_summary(
            examples,
            split,
            heldout_seeds=heldout_seeds,
            heldout_source_patterns=heldout_source_patterns,
            heldout_fraction=heldout_fraction,
        ),
        "confusion_matrices": {
            ECOLOGICAL_ONLY_MODEL_ID: ecological_eval["confusion_matrix"],
            ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID: context_eval[
                "confusion_matrix"
            ],
        },
        "model_summaries": {
            ECOLOGICAL_ONLY_MODEL_ID: ecological_eval,
            ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID: context_eval,
        },
        "eat_overprediction_delta": eat_delta,
        "failure_mode_assessment": material,
    }


def build_rollout_context_audit_report_from_datasets(
    datasets: Sequence[TrajectoryJsonlDataset],
    **kwargs: object,
) -> dict[str, object]:
    records = tuple(records_with_trajectory_context(datasets))
    return build_rollout_context_audit_report(
        records,
        source_paths=[dataset.path for dataset in datasets],
        **kwargs,
    )


def write_rollout_context_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


class _ContextualRow:
    __slots__ = (
        "row_id",
        "record",
        "label",
        "action_source",
        "valid_actions",
        "ecological_keys",
        "context_keys",
        "source_path",
        "source_seed",
        "context_snapshot",
    )

    def __init__(
        self,
        *,
        row_id: str,
        record: Mapping[str, object],
        label: str,
        action_source: str,
        valid_actions: tuple[str, ...],
        ecological_keys: tuple[str, ...],
        context_keys: tuple[str, ...],
        source_path: str,
        source_seed: int | None,
        context_snapshot: Mapping[str, object],
    ):
        self.row_id = row_id
        self.record = record
        self.label = label
        self.action_source = action_source
        self.valid_actions = valid_actions
        self.ecological_keys = ecological_keys
        self.context_keys = context_keys
        self.source_path = source_path
        self.source_seed = source_seed
        self.context_snapshot = context_snapshot


def _contextual_rows(
    records: Sequence[Mapping[str, object]],
    *,
    config: RolloutContextConfig,
) -> list[_ContextualRow]:
    rows: list[_ContextualRow] = []
    states: dict[tuple[str, int], RolloutContextState] = {}
    for index, record in enumerate(records):
        episode_id = _episode_id(record, fallback_index=index)
        agent_id = _agent_id(record)
        key = (episode_id, agent_id)
        state = states.setdefault(key, RolloutContextState(config))
        snapshot = state.snapshot()
        context_keys = (
            state.context_key(),
            state.coarse_context_key(),
            f"phase:{snapshot.get('post_carrion_contact')}|last:{_last_resolved(snapshot)}",
            "rollout_context:any",
        )
        source_path = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
        rows.append(
            _ContextualRow(
                row_id=_row_id(record, fallback_index=index),
                record=record,
                label=_label(record),
                action_source=_action_source(record),
                valid_actions=_valid_actions(record),
                ecological_keys=feature_keys_from_record(record),
                context_keys=context_keys,
                source_path=source_path,
                source_seed=_source_seed(record, source_path),
                context_snapshot=snapshot,
            )
        )
        state.update_from_record(record)
    return rows


def _train_counts(
    examples: Sequence[_ContextualRow],
    *,
    key_policy: str,
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        for key in _keys_for_policy(example, key_policy=key_policy):
            counts[key][example.label] += 1
        counts["__global__"][example.label] += 1
    return dict(counts)


def _evaluate_counts(
    examples: Sequence[_ContextualRow],
    counts: Mapping[str, Counter[str]],
    *,
    key_policy: str,
) -> dict[str, object]:
    matrix = {action: {predicted: 0 for predicted in ACTION_NAMES} for action in ACTION_NAMES}
    label_counts = Counter[str]()
    prediction_counts = Counter[str]()
    records: list[dict[str, object]] = []
    correct = 0
    for example in examples:
        predicted = _predict(example, counts, key_policy=key_policy)
        label_counts[example.label] += 1
        prediction_counts[predicted] += 1
        matrix[example.label][predicted] += 1
        if predicted == example.label:
            correct += 1
        records.append(
            {
                "row_id": example.row_id,
                "source_path": example.source_path,
                "source_seed": example.source_seed,
                "tick": example.record.get("tick"),
                "agent_id": example.record.get("agent_id"),
                "label": example.label,
                "predicted": predicted,
                "post_carrion_contact": bool(
                    example.context_snapshot.get("post_carrion_contact", False)
                ),
            }
        )
    total = len(examples)
    return {
        "record_count": total,
        "accuracy": _round(correct / float(total) if total else 0.0),
        "label_counts": _ordered_counts(label_counts),
        "prediction_counts": _ordered_counts(prediction_counts),
        "confusion_matrix": matrix,
        "eat_overprediction": _eat_overprediction_summary(records),
        "top_confusions": _top_confusions(matrix),
    }


def _predict(
    example: _ContextualRow,
    counts: Mapping[str, Counter[str]],
    *,
    key_policy: str,
) -> str:
    valid = set(example.valid_actions)
    for key in _keys_for_policy(example, key_policy=key_policy):
        action = _best_counted_action(counts.get(key), valid)
        if action is not None:
            return action
    action = _best_counted_action(counts.get("__global__"), valid)
    if action is not None:
        return action
    for candidate in ACTION_NAMES:
        if candidate in valid:
            return candidate
    return ACTION_NAMES[0]


def _keys_for_policy(example: _ContextualRow, *, key_policy: str) -> tuple[str, ...]:
    if key_policy == ECOLOGICAL_ONLY_MODEL_ID:
        return example.ecological_keys
    if key_policy != ECOLOGICAL_PLUS_ROLLOUT_CONTEXT_MODEL_ID:
        raise RolloutContextAuditError(f"unsupported key policy: {key_policy}")
    keys: list[str] = []
    for ecological_key in example.ecological_keys:
        for context_key in example.context_keys:
            keys.append(f"{ecological_key}|{context_key}")
    keys.extend(example.ecological_keys)
    return tuple(keys)


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


def _eat_overprediction_summary(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups = {
        "movement": lambda action: action in MOVEMENT_ACTIONS or action.startswith("move_"),
        "drink": lambda action: action == "drink",
        "stay": lambda action: action == "stay",
        "movement_drink_stay": lambda action: (
            action in MOVEMENT_ACTIONS or action.startswith("move_") or action in ("drink", "stay")
        ),
    }
    summary: dict[str, object] = {}
    for name, predicate in groups.items():
        total = 0
        predicted_eat = 0
        post_contact_total = 0
        post_contact_predicted_eat = 0
        examples: list[dict[str, object]] = []
        for record in records:
            label = str(record.get("label"))
            if not predicate(label):
                continue
            total += 1
            if bool(record.get("post_carrion_contact", False)):
                post_contact_total += 1
            if record.get("predicted") == "eat":
                predicted_eat += 1
                if len(examples) < 12:
                    examples.append(dict(record))
                if bool(record.get("post_carrion_contact", False)):
                    post_contact_predicted_eat += 1
        summary[name] = {
            "record_count": total,
            "predicted_as_eat_count": predicted_eat,
            "predicted_as_eat_rate": _round(
                predicted_eat / float(total) if total else 0.0
            ),
            "post_carrion_contact_record_count": post_contact_total,
            "post_carrion_contact_predicted_as_eat_count": (
                post_contact_predicted_eat
            ),
            "post_carrion_contact_predicted_as_eat_rate": _round(
                post_contact_predicted_eat / float(post_contact_total)
                if post_contact_total
                else 0.0
            ),
            "examples": examples,
        }
    return summary


def _eat_overprediction_delta(
    ecological_eval: Mapping[str, object],
    context_eval: Mapping[str, object],
) -> dict[str, object]:
    ecological = _mapping(ecological_eval.get("eat_overprediction"))
    context = _mapping(context_eval.get("eat_overprediction"))
    result: dict[str, object] = {}
    for group in ("movement", "drink", "stay", "movement_drink_stay"):
        ecological_group = _mapping(ecological.get(group))
        context_group = _mapping(context.get(group))
        ecological_rate = _float(ecological_group.get("predicted_as_eat_rate"))
        context_rate = _float(context_group.get("predicted_as_eat_rate"))
        ecological_post = _float(
            ecological_group.get("post_carrion_contact_predicted_as_eat_rate")
        )
        context_post = _float(
            context_group.get("post_carrion_contact_predicted_as_eat_rate")
        )
        result[group] = {
            "ecological_only_predicted_as_eat_rate": ecological_rate,
            "ecological_plus_context_predicted_as_eat_rate": context_rate,
            "absolute_rate_reduction": _round(ecological_rate - context_rate),
            "ecological_only_post_carrion_predicted_as_eat_rate": ecological_post,
            "ecological_plus_context_post_carrion_predicted_as_eat_rate": context_post,
            "post_carrion_absolute_rate_reduction": _round(
                ecological_post - context_post
            ),
        }
    return result


def _material_improvement(
    eat_delta: Mapping[str, object],
    ecological_eval: Mapping[str, object],
    context_eval: Mapping[str, object],
    *,
    min_rate_reduction: float,
) -> dict[str, object]:
    movement_delta = _mapping(eat_delta.get("movement_drink_stay"))
    rate_reduction = _float(movement_delta.get("absolute_rate_reduction"))
    post_contact_reduction = _float(
        movement_delta.get("post_carrion_absolute_rate_reduction")
    )
    accuracy_delta = _round(
        _float(context_eval.get("accuracy")) - _float(ecological_eval.get("accuracy"))
    )
    materially_improves = (
        rate_reduction >= min_rate_reduction
        and post_contact_reduction >= 0.0
        and accuracy_delta >= -0.01
    )
    blockers: list[dict[str, object]] = []
    if rate_reduction < min_rate_reduction:
        blockers.append(
            {
                "reason": "movement_drink_stay_eat_overprediction_delta_below_floor",
                "required_absolute_rate_reduction": _round(min_rate_reduction),
                "observed_absolute_rate_reduction": rate_reduction,
            }
        )
    if post_contact_reduction < 0.0:
        blockers.append(
            {
                "reason": "post_carrion_eat_overprediction_worse",
                "observed_post_carrion_delta": post_contact_reduction,
            }
        )
    if accuracy_delta < -0.01:
        blockers.append(
            {
                "reason": "heldout_accuracy_regression",
                "accuracy_delta": accuracy_delta,
            }
        )
    return {
        "materially_improves_v62_failure_mode": materially_improves,
        "required_absolute_rate_reduction": _round(min_rate_reduction),
        "movement_drink_stay_absolute_rate_reduction": rate_reduction,
        "post_carrion_absolute_rate_reduction": post_contact_reduction,
        "heldout_accuracy_delta": accuracy_delta,
        "blockers": blockers,
    }


def _assign_split(
    examples: Sequence[_ContextualRow],
    *,
    heldout_seeds: set[int],
    heldout_source_patterns: Sequence[str],
    heldout_fraction: float,
) -> dict[str, str]:
    split: dict[str, str] = {}
    for example in examples:
        is_heldout = False
        if heldout_seeds and example.source_seed in heldout_seeds:
            is_heldout = True
        if not is_heldout:
            is_heldout = any(pattern in example.source_path for pattern in heldout_source_patterns)
        if not is_heldout and not heldout_seeds and not heldout_source_patterns:
            digest = stable_payload_digest([example.source_path, example.source_seed])
            bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
            is_heldout = bucket < heldout_fraction
        split[example.row_id] = "heldout" if is_heldout else "train"
    values = set(split.values())
    if values == {"train"}:
        for example in _fallback_heldout_examples(examples, heldout_fraction):
            split[example.row_id] = "heldout"
    if values == {"heldout"}:
        first = sorted(examples, key=lambda example: example.row_id)[0]
        split[first.row_id] = "train"
    return split


def _fallback_heldout_examples(
    examples: Sequence[_ContextualRow],
    heldout_fraction: float,
) -> tuple[_ContextualRow, ...]:
    count = max(1, int(round(len(examples) * max(heldout_fraction, 0.1))))
    sorted_examples = sorted(
        examples,
        key=lambda example: stable_payload_digest(example.row_id),
    )
    return tuple(sorted_examples[:count])


def _split_summary(
    examples: Sequence[_ContextualRow],
    split: Mapping[str, str],
    *,
    heldout_seeds: set[int],
    heldout_source_patterns: Sequence[str],
    heldout_fraction: float,
) -> dict[str, object]:
    train_count = sum(1 for example in examples if split[example.row_id] == "train")
    heldout_count = len(examples) - train_count
    train_sources = {
        example.source_path for example in examples if split[example.row_id] == "train"
    }
    heldout_sources = {
        example.source_path
        for example in examples
        if split[example.row_id] == "heldout"
    }
    return {
        "policy": "source_seed_or_source_pattern_else_stable_source_hash_v1",
        "heldout_seed_values": sorted(heldout_seeds),
        "heldout_source_patterns": list(heldout_source_patterns),
        "heldout_fraction": _round(heldout_fraction),
        "train_record_count": train_count,
        "heldout_record_count": heldout_count,
        "train_source_count": len(train_sources),
        "heldout_source_count": len(heldout_sources),
        "train_sources_sample": sorted(train_sources)[:12],
        "heldout_sources_sample": sorted(heldout_sources)[:12],
    }


def _source_summary(
    records: Sequence[Mapping[str, object]],
    *,
    source_paths: Sequence[str | Path] | None,
) -> dict[str, object]:
    counts = Counter(str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown")) for record in records)
    paths = sorted(str(path) for path in (source_paths or counts.keys()))
    return {
        "trajectory_count": len(paths),
        "record_count": len(records),
        "paths": [
            {"path": path, "record_count": int(counts.get(path, 0))}
            for path in paths
        ],
    }


def _top_confusions(
    matrix: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    confusions: list[tuple[int, str, str]] = []
    for label, row in matrix.items():
        for predicted, count in row.items():
            if label == predicted or count <= 0:
                continue
            confusions.append((int(count), str(label), str(predicted)))
    confusions.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {"label": label, "predicted": predicted, "count": count}
        for count, label, predicted in confusions[:12]
    ]


def _ordered_counts(counts: Counter[str]) -> dict[str, int]:
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


def _episode_id(record: Mapping[str, object], *, fallback_index: int) -> str:
    episode_id = record.get(TRAJECTORY_EPISODE_ID_FIELD)
    if isinstance(episode_id, str) and episode_id:
        return episode_id
    return f"inferred:{fallback_index}"


def _row_id(record: Mapping[str, object], *, fallback_index: int) -> str:
    source = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
    episode_id = _episode_id(record, fallback_index=fallback_index)
    record_index = record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD)
    if isinstance(record_index, int) and not isinstance(record_index, bool):
        return f"{source}:{episode_id}:record={record_index}"
    return f"{source}:{episode_id}:record={fallback_index}"


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


def _action_source(record: Mapping[str, object]) -> str:
    source = record.get("action_source")
    return source if isinstance(source, str) and source else "unknown"


def _valid_actions(record: Mapping[str, object]) -> tuple[str, ...]:
    action_mask = record.get("action_mask")
    if not isinstance(action_mask, Mapping):
        return ACTION_NAMES
    actions = tuple(
        action for action in ACTION_NAMES if bool(action_mask.get(action, False))
    )
    return actions or ACTION_NAMES


def _source_seed(record: Mapping[str, object], source_path: str) -> int | None:
    for payload in (
        source_path,
        str(record.get(TRAJECTORY_EPISODE_ID_FIELD, "")),
    ):
        match = _SEED_PATTERN.search(payload)
        if match is not None:
            return int(match.group(1))
    return None


def _last_resolved(snapshot: Mapping[str, object]) -> str:
    recent = snapshot.get("recent_resolved_actions")
    if isinstance(recent, Sequence) and recent:
        action = str(recent[-1])
        if action.startswith("move_"):
            return "move"
        return action
    return "none"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
