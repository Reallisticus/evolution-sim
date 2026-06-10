from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.current_route_decision import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V136_REPORT_PATH,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
    TrajectoryJsonlDataset,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _resolve_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import RolloutContextConfig
from evolution_sim.mind.rollout_sequence_support_audit import (
    ACTION_ONLY_BASELINE_ID,
    ACTION_ORDER_BASELINE_ID,
    DEFAULT_OUTPUT_PATH as DEFAULT_V137_REPORT_PATH,
    DOMINANT_PREDICTED_ACTION_SHARE_MAX,
    MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
    SEQUENCE_MODEL_ID,
    STRICT_HELDOUT_SEEDS,
    _evaluate,
    _examples_from_records,
    _predict_action_only,
    _predict_action_order,
    _predict_sequence,
    _records_from_datasets,
    _resolve_trajectories,
    _sequence_counts,
    build_rollout_sequence_support_audit_report,
)

MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION = (
    "mind_v3_v138_rollout_sequence_strict_seed_support_recheck_v1"
)
MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_POLICY = (
    "diagnostics_only_v138_strict_seed_rollout_sequence_support_recheck_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v138-rollout-sequence-strict-seed-support-recheck.json"
)

READY_V137_CLASSIFICATION = (
    "rollout_sequence_support_ready_for_future_sequence_world_model_scorer"
)
_SEED_LIST_PATTERN = re.compile(r"seed[-_=]?([0-9][0-9,\s]*)")
_PREFIX_SEED_PATTERN = re.compile(r"(?:mind-v3-|heuristic-)(\d+)")


class RolloutSequenceStrictSeedSupportRecheckError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RolloutSequenceStrictSeedSupportRecheckBuild:
    report: dict[str, object]


def build_rollout_sequence_strict_seed_support_recheck_report(
    *,
    v137_report: Mapping[str, object] | None = None,
    v137_report_path: str | Path | None = DEFAULT_V137_REPORT_PATH,
    v136_report: Mapping[str, object] | None = None,
    v136_report_path: str | Path | None = DEFAULT_V136_REPORT_PATH,
    train_trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    strict_heldout_trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    train_trajectory_paths: Sequence[str | Path] = (),
    strict_heldout_trajectory_paths: Sequence[str | Path] = (),
    strict_seed_values: Sequence[int] = STRICT_HELDOUT_SEEDS,
    context_config: RolloutContextConfig | None = None,
) -> RolloutSequenceStrictSeedSupportRecheckBuild:
    strict_seeds = tuple(sorted({int(seed) for seed in strict_seed_values}))
    if not strict_seeds:
        raise RolloutSequenceStrictSeedSupportRecheckError(
            "strict_seed_values must not be empty"
        )

    v137_payload, v137_evidence = _resolve_json_report(
        v137_report,
        v137_report_path,
        expected_schema=MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
    )
    train = _resolve_trajectories(
        trajectory_datasets=train_trajectory_datasets,
        trajectory_paths=train_trajectory_paths,
        report_paths=(),
    )
    strict_heldout = _resolve_trajectories(
        trajectory_datasets=strict_heldout_trajectory_datasets,
        trajectory_paths=strict_heldout_trajectory_paths,
        report_paths=(),
    )
    datasets = (*train.datasets, *strict_heldout.datasets)
    config = context_config or RolloutContextConfig()
    v137_style = build_rollout_sequence_support_audit_report(
        v136_report=v136_report,
        v136_report_path=v136_report_path,
        trajectory_datasets=datasets,
        heldout_seed_values=strict_seeds,
        heldout_fraction=0.0,
        context_config=config,
    ).report
    records = _records_from_datasets(datasets)
    examples = _examples_from_records(
        records,
        config=config,
        heldout_seed_values=strict_seeds,
        heldout_source_patterns=(),
        heldout_fraction=0.0,
    )
    source_roles = _source_role_summary(
        train.datasets,
        strict_heldout.datasets,
        strict_seed_values=strict_seeds,
    )
    strict_metrics = _strict_seed_metrics(
        examples,
        strict_seed_values=strict_seeds,
        source_seed_sets=source_roles["source_seed_sets_by_path"],
    )
    source_integrity = _source_integrity(
        v137_report=v137_payload,
        v137_evidence=v137_evidence,
        train_evidence=train.evidence,
        strict_heldout_evidence=strict_heldout.evidence,
        source_roles=source_roles,
        v137_style_report=v137_style,
    )
    support_floors = _support_floors(
        source_integrity=source_integrity,
        v137_style_report=v137_style,
        source_roles=source_roles,
        strict_metrics=strict_metrics,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
    )
    contract = _contract(strict_seeds)
    report = {
        "schema_version": (
            MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_POLICY,
        "contract": contract,
        "provenance": {"contract_digest": stable_payload_digest(contract)},
        "source_integrity": source_integrity,
        "source_evidence": {
            "v137_report": v137_evidence,
            "train_trajectories": train.evidence,
            "strict_heldout_trajectories": strict_heldout.evidence,
            "v137_style_recheck": _v137_style_evidence(v137_style),
        },
        "strict_seed_source_integrity": source_roles,
        "leakage_scan": v137_style.get("leakage_scan"),
        "train_heldout_split": v137_style.get("train_heldout_split"),
        "strict_seed_metrics": strict_metrics,
        "support_floors": support_floors,
        "classification": classification,
        "authorization_block": _authorization_block(),
        "non_promoted": True,
    }
    return RolloutSequenceStrictSeedSupportRecheckBuild(report=report)


def write_rollout_sequence_strict_seed_support_recheck_report(
    build: RolloutSequenceStrictSeedSupportRecheckBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        import json

        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract(strict_seed_values: Sequence[int]) -> dict[str, object]:
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
        "shadow_scorer_execution_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "runtime_promotion_authorized": False,
        "required_v137_schema_version": MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
        "required_v137_classification": READY_V137_CLASSIFICATION,
        "strict_seed_values": list(strict_seed_values),
        "train_source_role": "v98_numeric_train_bank",
        "heldout_source_role": "strict_mind_v3_seed_heldout_trajectories",
        "heldout_fraction": 0.0,
    }


def _source_integrity(
    *,
    v137_report: Mapping[str, object] | None,
    v137_evidence: Mapping[str, object],
    train_evidence: Mapping[str, object],
    strict_heldout_evidence: Mapping[str, object],
    source_roles: Mapping[str, object],
    v137_style_report: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if v137_evidence.get("loaded") is not True:
        failures.append("missing_v137_report")
    elif v137_evidence.get("schema_matches") is not True:
        failures.append("v137_schema_mismatch")
    if _mapping(_mapping(v137_report).get("source_integrity")).get("passed") is not True:
        failures.append("v137_source_integrity_not_passed")
    if _mapping(_mapping(v137_report).get("classification")).get("primary") != (
        READY_V137_CLASSIFICATION
    ):
        failures.append("v137_classification_not_ready")
    if _int(train_evidence.get("loaded_path_count")) <= 0:
        failures.append("no_train_trajectory_inputs_loaded")
    if _int(strict_heldout_evidence.get("loaded_path_count")) <= 0:
        failures.append("no_strict_heldout_trajectory_inputs_loaded")
    if _int(train_evidence.get("load_failure_count")) > 0:
        failures.append("train_trajectory_load_failures")
    if _int(strict_heldout_evidence.get("load_failure_count")) > 0:
        failures.append("strict_heldout_trajectory_load_failures")
    if source_roles.get("strict_seed_presence_only_heldout") is not True:
        failures.append("strict_seeds_not_present_only_in_heldout")
    if _int(source_roles.get("source_path_overlap_count")) > 0:
        failures.append("train_strict_heldout_source_path_overlap")
    if _mapping(v137_style_report.get("source_integrity")).get("passed") is not True:
        failures.append("v137_style_recheck_source_integrity_not_passed")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v137_source_integrity_passed": _mapping(
            _mapping(v137_report).get("source_integrity")
        ).get("passed"),
        "v137_classification": _mapping(_mapping(v137_report).get("classification")).get(
            "primary"
        ),
        "required_v137_classification": READY_V137_CLASSIFICATION,
        "v137_style_recheck_source_integrity_passed": _mapping(
            v137_style_report.get("source_integrity")
        ).get("passed"),
        "train_loaded_path_count": train_evidence.get("loaded_path_count"),
        "strict_heldout_loaded_path_count": strict_heldout_evidence.get(
            "loaded_path_count"
        ),
        "strict_seed_presence_only_heldout": source_roles.get(
            "strict_seed_presence_only_heldout"
        ),
        "source_path_overlap_count": source_roles.get("source_path_overlap_count"),
        "source_path_overlap_examples": source_roles.get(
            "source_path_overlap_examples"
        ),
    }


def _source_role_summary(
    train_datasets: Sequence[TrajectoryJsonlDataset],
    strict_heldout_datasets: Sequence[TrajectoryJsonlDataset],
    *,
    strict_seed_values: Sequence[int],
) -> dict[str, object]:
    strict_seeds = set(int(seed) for seed in strict_seed_values)
    train_sources = _source_entries(train_datasets)
    heldout_sources = _source_entries(strict_heldout_datasets)
    train_strict_sources = [
        source
        for source in train_sources
        if set(source["source_seed_set"]).intersection(strict_seeds)
    ]
    heldout_strict_sources = [
        source
        for source in heldout_sources
        if set(source["source_seed_set"]).intersection(strict_seeds)
    ]
    heldout_seed_set = {
        int(seed)
        for source in heldout_strict_sources
        for seed in source["source_seed_set"]
        if seed in strict_seeds
    }
    missing = sorted(strict_seeds.difference(heldout_seed_set))
    train_leakage_seed_set = {
        int(seed)
        for source in train_strict_sources
        for seed in source["source_seed_set"]
        if seed in strict_seeds
    }
    counts_by_seed = {
        str(seed): sum(
            1 for source in heldout_strict_sources if seed in source["source_seed_set"]
        )
        for seed in sorted(strict_seeds)
    }
    train_paths = {str(source["source_path"]) for source in train_sources}
    heldout_paths = {str(source["source_path"]) for source in heldout_sources}
    overlap_paths = sorted(train_paths.intersection(heldout_paths))
    seed_sets_by_path = {
        str(source["source_path"]): list(source["source_seed_set"])
        for source in (*train_sources, *heldout_sources)
    }
    return {
        "policy": "v138_strict_seed_sources_train_vs_heldout_integrity_v1",
        "strict_seed_values": sorted(strict_seeds),
        "train_source_count": len(train_sources),
        "strict_heldout_source_count": len(heldout_sources),
        "strict_seeds_present_in_heldout": sorted(heldout_seed_set),
        "strict_seeds_present_in_train": sorted(train_leakage_seed_set),
        "missing_strict_heldout_seeds": missing,
        "strict_heldout_source_count_by_seed": counts_by_seed,
        "strict_seed_train_source_leakage_count": len(train_strict_sources),
        "strict_seed_train_source_leakage_examples": train_strict_sources[:12],
        "source_path_overlap_count": len(overlap_paths),
        "source_path_overlap_examples": overlap_paths[:12],
        "train_sources": train_sources,
        "strict_heldout_sources": heldout_sources,
        "source_seed_sets_by_path": seed_sets_by_path,
        "strict_seed_presence_only_heldout": (
            not missing and not train_strict_sources
        ),
    }


def _source_entries(
    datasets: Sequence[TrajectoryJsonlDataset],
) -> list[dict[str, object]]:
    by_source: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        source = str(dataset.path)
        entry = by_source.setdefault(
            source,
            {
                "source_path": source,
                "source_seed_set": set(),
                "record_count": 0,
            },
        )
        entry["source_seed_set"].update(_seed_set_from_dataset(dataset))
        entry["record_count"] = _int(entry["record_count"]) + dataset.record_count
    return [
        {
            "source_path": source,
            "source_seed": (
                seeds[0]
                if len(seeds := sorted(entry["source_seed_set"])) == 1
                else None
            ),
            "source_seed_set": seeds,
            "record_count": entry["record_count"],
        }
        for source, entry in sorted(by_source.items())
    ]


def _seed_set_from_dataset(dataset: TrajectoryJsonlDataset) -> set[int]:
    seeds: set[int] = set()
    seeds.update(_seed_tokens(str(dataset.path)))
    seeds.update(_seed_tokens(str(dataset.header.get("run_id", ""))))
    footer_provenance = _mapping(dataset.footer.get("provenance"))
    seeds.update(_seed_values_from_provenance(footer_provenance))
    for record in _records_from_datasets((dataset,)):
        seeds.update(_seed_tokens(str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, ""))))
        seeds.update(_seed_tokens(str(record.get(TRAJECTORY_EPISODE_ID_FIELD, ""))))
    return seeds


def _seed_values_from_provenance(value: object, *, key_path: tuple[str, ...] = ()) -> set[int]:
    seeds: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            seeds.update(_seed_values_from_provenance(child, key_path=(*key_path, str(key))))
        return seeds
    key_text = ".".join(key_path).lower()
    if "seed" in key_text:
        seeds.update(_numeric_seed_values(value))
    if any(token in key_text for token in ("path", "trajectory", "source")):
        seeds.update(_path_seed_values(value))
    return seeds


def _numeric_seed_values(value: object) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, str):
        return {int(token) for token in re.findall(r"\\d+", value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seeds: set[int] = set()
        for item in value:
            seeds.update(_numeric_seed_values(item))
        return seeds
    return set()


def _path_seed_values(value: object) -> set[int]:
    if isinstance(value, str):
        return _seed_tokens(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seeds: set[int] = set()
        for item in value:
            seeds.update(_path_seed_values(item))
        return seeds
    return set()


def _seed_tokens(value: str) -> set[int]:
    seeds: set[int] = set()
    for match in _SEED_LIST_PATTERN.finditer(value):
        seeds.update(int(token) for token in re.findall(r"\\d+", match.group(1)))
    seeds.update(int(match.group(1)) for match in _PREFIX_SEED_PATTERN.finditer(value))
    return seeds


def _strict_seed_metrics(
    examples: Sequence[object],
    *,
    strict_seed_values: Sequence[int],
    source_seed_sets: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    strict_seeds = tuple(sorted({int(seed) for seed in strict_seed_values}))
    train = tuple(example for example in examples if example.split == "train")
    strict_heldout = tuple(
        example
        for example in examples
        if example.split == "heldout"
        and set(source_seed_sets.get(example.source_path, ())).intersection(
            strict_seeds
        )
    )
    sequence_counts = _sequence_counts(train)
    action_counts = Counter(example.label for example in train)
    sequence_eval = _evaluate(
        strict_heldout,
        predictor=lambda example: _predict_sequence(
            example,
            sequence_counts=sequence_counts,
            action_counts=action_counts,
        ),
    )
    action_only_eval = _evaluate(
        strict_heldout,
        predictor=lambda example: _predict_action_only(
            example,
            action_counts=action_counts,
        ),
    )
    action_order_eval = _evaluate(strict_heldout, predictor=_predict_action_order)
    per_seed: dict[str, object] = {}
    for seed in strict_seeds:
        seed_examples = tuple(
            example
            for example in strict_heldout
            if seed in set(source_seed_sets.get(example.source_path, ()))
        )
        seed_sequence = _evaluate(
            seed_examples,
            predictor=lambda example: _predict_sequence(
                example,
                sequence_counts=sequence_counts,
                action_counts=action_counts,
            ),
        )
        seed_action_only = _evaluate(
            seed_examples,
            predictor=lambda example: _predict_action_only(
                example,
                action_counts=action_counts,
            ),
        )
        seed_action_order = _evaluate(seed_examples, predictor=_predict_action_order)
        per_seed[str(seed)] = {
            "heldout_record_count": seed_sequence.get("heldout_record_count"),
            "accuracy": seed_sequence.get("accuracy"),
            "correct_count": seed_sequence.get("correct_count"),
            "action_only_accuracy": seed_action_only.get("accuracy"),
            "action_order_accuracy": seed_action_order.get("accuracy"),
            "accuracy_delta_vs_action_only": _round(
                _float(seed_sequence.get("accuracy"))
                - _float(seed_action_only.get("accuracy"))
            ),
            "accuracy_delta_vs_action_order": _round(
                _float(seed_sequence.get("accuracy"))
                - _float(seed_action_order.get("accuracy"))
            ),
            "dominant_predicted_action": seed_sequence.get(
                "dominant_predicted_action"
            ),
            "dominant_predicted_action_share": seed_sequence.get(
                "dominant_predicted_action_share"
            ),
            "unsupported_action_count": seed_sequence.get("unsupported_action_count"),
        }
    counts_by_seed = {
        seed: _int(_mapping(per_seed[str(seed)]).get("heldout_record_count"))
        for seed in strict_seeds
    }
    missing_evaluated = [
        seed for seed, count in counts_by_seed.items() if int(count) <= 0
    ]
    return {
        "policy": "v138_strict_seed_sequence_support_metrics_v1",
        "strict_seed_values": list(strict_seeds),
        "aggregate_strict_heldout": sequence_eval,
        "baselines": {
            ACTION_ONLY_BASELINE_ID: action_only_eval,
            ACTION_ORDER_BASELINE_ID: action_order_eval,
        },
        "baseline_comparisons": {
            "strict_heldout_accuracy_delta_vs_action_only": _round(
                _float(sequence_eval.get("accuracy"))
                - _float(action_only_eval.get("accuracy"))
            ),
            "strict_heldout_accuracy_delta_vs_action_order": _round(
                _float(sequence_eval.get("accuracy"))
                - _float(action_order_eval.get("accuracy"))
            ),
            "beats_action_only_baseline": _float(sequence_eval.get("accuracy"))
            > _float(action_only_eval.get("accuracy")),
            "beats_action_order_baseline": _float(sequence_eval.get("accuracy"))
            > _float(action_order_eval.get("accuracy")),
        },
        "per_strict_seed": per_seed,
        "strict_seed_evaluated_record_counts": {
            str(seed): count for seed, count in counts_by_seed.items()
        },
        "strict_seed_missing_evaluated_record_seeds": missing_evaluated,
        "all_strict_seeds_have_evaluated_records": not missing_evaluated,
        "dominant_predicted_action_share": sequence_eval.get(
            "dominant_predicted_action_share"
        ),
        "unsupported_action_count": sequence_eval.get("unsupported_action_count"),
    }


def _support_floors(
    *,
    source_integrity: Mapping[str, object],
    v137_style_report: Mapping[str, object],
    source_roles: Mapping[str, object],
    strict_metrics: Mapping[str, object],
) -> dict[str, object]:
    leakage_scan = _mapping(v137_style_report.get("leakage_scan"))
    split = _mapping(v137_style_report.get("train_heldout_split"))
    aggregate = _mapping(strict_metrics.get("aggregate_strict_heldout"))
    comparisons = _mapping(strict_metrics.get("baseline_comparisons"))
    floors = [
        _floor(
            "source_integrity_passed",
            source_integrity.get("passed") is True,
            observed=source_integrity.get("passed"),
            required=True,
        ),
        _floor(
            "strict_seed_presence_only_heldout",
            source_roles.get("strict_seed_presence_only_heldout") is True,
            observed={
                "present_in_heldout": source_roles.get(
                    "strict_seeds_present_in_heldout"
                ),
                "present_in_train": source_roles.get("strict_seeds_present_in_train"),
                "missing": source_roles.get("missing_strict_heldout_seeds"),
            },
            required="all configured strict seeds present in heldout and absent from train",
        ),
        _floor(
            "train_strict_heldout_source_path_overlap_count_eq_0",
            _int(source_roles.get("source_path_overlap_count")) == 0,
            observed=source_roles.get("source_path_overlap_count"),
            required=0,
        ),
        _floor(
            "all_strict_seeds_have_evaluated_records",
            strict_metrics.get("all_strict_seeds_have_evaluated_records") is True,
            observed=strict_metrics.get("strict_seed_evaluated_record_counts"),
            required="heldout_record_count>0 for every configured strict seed",
        ),
        _floor(
            "leakage_scan_passed",
            leakage_scan.get("passed") is True,
            observed=leakage_scan.get("leakage_count"),
            required=0,
        ),
        _floor(
            "configured_heldout_seed_leakage_count_eq_0",
            _int(split.get("configured_heldout_seed_leakage_count")) == 0,
            observed=split.get("configured_heldout_seed_leakage_count"),
            required=0,
        ),
        _floor(
            "beats_action_only_baseline",
            comparisons.get("beats_action_only_baseline") is True,
            observed=comparisons.get("strict_heldout_accuracy_delta_vs_action_only"),
            required=">0",
        ),
        _floor(
            "beats_action_order_baseline",
            comparisons.get("beats_action_order_baseline") is True,
            observed=comparisons.get("strict_heldout_accuracy_delta_vs_action_order"),
            required=">0",
        ),
        _floor(
            "dominant_predicted_action_share_lte_0_50",
            _float(aggregate.get("dominant_predicted_action_share"))
            <= DOMINANT_PREDICTED_ACTION_SHARE_MAX,
            observed=aggregate.get("dominant_predicted_action_share"),
            required=DOMINANT_PREDICTED_ACTION_SHARE_MAX,
        ),
        _floor(
            "unsupported_action_count_eq_0",
            _int(aggregate.get("unsupported_action_count")) == 0,
            observed=aggregate.get("unsupported_action_count"),
            required=0,
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "v138_rollout_sequence_strict_seed_support_floors_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "floors": floors,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    support_floors: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        primary = "rollout_sequence_strict_seed_recheck_source_integrity_failed"
    elif support_floors.get("passed") is True:
        primary = "rollout_sequence_strict_seed_support_ready_for_future_sequence_world_model_scorer_diagnostic"
    else:
        primary = "rollout_sequence_strict_seed_support_blocked"
    return {
        "primary": primary,
        "labels": [primary],
        "first_failed_floor": support_floors.get("first_failed_floor"),
        "allowed_classifications": [
            "rollout_sequence_strict_seed_recheck_source_integrity_failed",
            "rollout_sequence_strict_seed_support_blocked",
            "rollout_sequence_strict_seed_support_ready_for_future_sequence_world_model_scorer_diagnostic",
        ],
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


def _v137_style_evidence(report: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": report.get("schema_version"),
        "source_integrity": report.get("source_integrity"),
        "classification": report.get("classification"),
        "train_heldout_split": report.get("train_heldout_split"),
        "support_floors": report.get("support_floors"),
        "baseline_comparisons": report.get("baseline_comparisons"),
        "model_summaries": {
            SEQUENCE_MODEL_ID: _mapping(report.get("model_summaries")).get(
                SEQUENCE_MODEL_ID
            )
        },
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


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _round(value: float) -> float:
    return round(float(value), 6)
