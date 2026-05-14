from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION = (
    "mind_v3_standard_progress_ledger_v1"
)
MIND_V3_STANDARD_PROGRESS_LEDGER_ROW_SCHEMA_VERSION = (
    "mind_v3_standard_progress_ledger_row_v1"
)
MIND_V3_PROGRESS_RULE_VERSION = (
    "strict_policy_pass_or_predeclared_support_floor_v1"
)

_VERSION_PATTERN = re.compile(r"\bv(\d{1,3})\b")
_ARTIFACT_VERSION_PATTERN = re.compile(r"mind-v3-v(\d{1,3})")
_AUTHORITATIVE_SUPPORT_PROBE_KEYS = frozenset(
    {
        "catastrophe_sensitive_utility_support_probe",
        "broad_branch_residual_oracle_support_probe",
        "broad_branch_residual_constrained_support_probe",
        "broad_transfer_residual_support_probe",
        "constrained_planning_support_probe",
        "depleted_resource_trap_support_probe",
        "planner_distillation_runtime_feasibility_support_probe",
        "sequence_continuation_support_probe",
    }
)


class MindV3ProgressLedgerError(ValueError):
    pass


def build_standard_progress_ledger_report(
    *,
    docs_path: str | Path,
    legacy_ledger_path: str | Path,
    output_dir: str | Path,
    start_version: int = 1,
    through_version: int | None = None,
) -> dict[str, object]:
    docs = _read_docs(Path(docs_path))
    artifacts = _artifact_paths(Path(output_dir))
    legacy_records = _read_legacy_ledger(Path(legacy_ledger_path))
    observed_versions = (
        set(docs)
        | set(artifacts)
        | {
            version
            for record in legacy_records
            for version in _record_versions(record)
        }
    )
    if through_version is None:
        through_version = max(observed_versions, default=start_version)
    if start_version <= 0:
        raise MindV3ProgressLedgerError("start_version must be positive")
    if through_version < start_version:
        raise MindV3ProgressLedgerError(
            "through_version must be greater than or equal to start_version"
        )

    rows = [
        _standard_progress_row(
            version=version,
            docs=docs.get(version, []),
            artifacts=artifacts.get(version, []),
            legacy_records=[
                record
                for record in legacy_records
                if version in _record_versions(record)
            ],
        )
        for version in range(start_version, through_version + 1)
    ]
    summary = _summary(rows)
    contract = {
        "schema_version": MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION,
        "row_schema_version": MIND_V3_STANDARD_PROGRESS_LEDGER_ROW_SCHEMA_VERSION,
        "progress_rule_version": MIND_V3_PROGRESS_RULE_VERSION,
        "progress_definition": (
            "A v-slice counts as progress only when it has either a strict "
            "policy pass or a diagnostic support probe with a predeclared "
            "floor that passed. Data milestones and accepted audits without a "
            "support floor are tracked separately."
        ),
        "start_version": int(start_version),
        "through_version": int(through_version),
    }
    return {
        "schema_version": MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "docs_path": str(docs_path),
            "legacy_ledger_path": str(legacy_ledger_path),
            "output_dir": str(output_dir),
            "contract_digest": stable_payload_digest(contract),
        },
        "summary": summary,
        "rows": rows,
    }


def write_standard_progress_ledger_report(
    report: Mapping[str, object],
    *,
    jsonl_output_path: str | Path,
    summary_output_path: str | Path | None = None,
) -> None:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise MindV3ProgressLedgerError("report rows must be a list")
    jsonl_path = Path(jsonl_output_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
    if summary_output_path is not None:
        summary_path = Path(summary_output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")


def _standard_progress_row(
    *,
    version: int,
    docs: Sequence[Mapping[str, object]],
    artifacts: Sequence[Path],
    legacy_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    artifact_payloads = [_load_json_artifact(path) for path in artifacts]
    authoritative_support_probes = [
        probe
        for payload in artifact_payloads
        for probe in _authoritative_support_probes(payload)
    ]
    support_probes = authoritative_support_probes or [
        probe
        for payload in artifact_payloads
        for probe in _support_probes(payload)
    ]
    strict_policy_observed = any(
        _has_key(payload, "promotion_candidate_passed")
        for payload in artifact_payloads
    ) or any(_has_key(record, "promotion_candidate_passed") for record in legacy_records)
    strict_policy_passed = any(
        _truthy_key(payload, "promotion_candidate_passed")
        for payload in artifact_payloads
    ) or any(
        _truthy_key(record, "promotion_candidate_passed")
        for record in legacy_records
    )
    diagnostic_support_floor_passed = any(
        probe.get("passed") is True for probe in support_probes
    )
    best_probe = _best_support_probe(support_probes)
    metrics = _version_metrics(
        artifact_payloads=artifact_payloads,
        legacy_records=legacy_records,
    )
    status = _row_status(
        strict_policy_observed=strict_policy_observed,
        strict_policy_passed=strict_policy_passed,
        support_probe_count=len(support_probes),
        diagnostic_support_floor_passed=diagnostic_support_floor_passed,
        artifact_count=len(artifacts),
        doc_mentioned=bool(docs),
    )
    progress_kind = "none"
    if strict_policy_passed:
        progress_kind = "strict_policy_pass"
    elif diagnostic_support_floor_passed:
        progress_kind = "diagnostic_support_floor_pass"
    source_paths = {
        "artifacts": [str(path) for path in artifacts],
        "docs": [
            {
                "path": str(item["path"]),
                "line": int(item["line"]),
                "text": str(item["text"]),
            }
            for item in docs[:8]
        ],
        "legacy_ledger_record_count": len(legacy_records),
    }
    row = {
        "schema_version": MIND_V3_STANDARD_PROGRESS_LEDGER_ROW_SCHEMA_VERSION,
        "progress_rule_version": MIND_V3_PROGRESS_RULE_VERSION,
        "version": int(version),
        "version_label": f"v{version}",
        "status": status,
        "progress_kind": progress_kind,
        "progress_passed": bool(strict_policy_passed or diagnostic_support_floor_passed),
        "strict_policy_observed": bool(strict_policy_observed),
        "strict_policy_passed": bool(strict_policy_passed),
        "diagnostic_support_probe_count": len(support_probes),
        "diagnostic_support_floor_passed": bool(diagnostic_support_floor_passed),
        "best_support_probe": best_probe,
        "doc_mentioned": bool(docs),
        "artifact_count": len(artifacts),
        "legacy_ledger_record_count": len(legacy_records),
        "metrics": metrics,
        "source_paths": source_paths,
    }
    row["row_digest"] = stable_payload_digest(
        {
            "version": row["version"],
            "status": row["status"],
            "progress_passed": row["progress_passed"],
            "metrics": row["metrics"],
            "source_paths": row["source_paths"],
        }
    )
    return row


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    progress_rows = [
        row for row in rows if row.get("progress_passed") is True
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "row_count": len(rows),
        "progress_passed_count": len(progress_rows),
        "progress_passed_versions": [
            int(row.get("version", 0)) for row in progress_rows
        ],
        "strict_policy_pass_versions": [
            int(row.get("version", 0))
            for row in rows
            if row.get("strict_policy_passed") is True
        ],
        "diagnostic_support_floor_pass_versions": [
            int(row.get("version", 0))
            for row in rows
            if row.get("diagnostic_support_floor_passed") is True
        ],
        "status_counts": dict(sorted(status_counts.items())),
    }


def _row_status(
    *,
    strict_policy_observed: bool,
    strict_policy_passed: bool,
    support_probe_count: int,
    diagnostic_support_floor_passed: bool,
    artifact_count: int,
    doc_mentioned: bool,
) -> str:
    if strict_policy_passed:
        return "strict_policy_pass"
    if strict_policy_observed:
        return "strict_policy_fail"
    if diagnostic_support_floor_passed:
        return "diagnostic_support_floor_pass"
    if support_probe_count:
        return "diagnostic_support_floor_fail"
    if artifact_count:
        return "artifact_or_data_milestone"
    if doc_mentioned:
        return "documented_only"
    return "missing_evidence"


def _support_probes(payload: Mapping[str, object]) -> list[dict[str, object]]:
    probes: list[dict[str, object]] = []
    _collect_support_probes(payload, path="$", probes=probes)
    return probes


def _authoritative_support_probes(
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    probes: list[dict[str, object]] = []
    for key in sorted(_AUTHORITATIVE_SUPPORT_PROBE_KEYS):
        value = payload.get(key)
        if isinstance(value, Mapping):
            _collect_support_probes(value, path=f"$.{key}", probes=probes)
    return probes


def _collect_support_probes(
    value: object,
    *,
    path: str,
    probes: list[dict[str, object]],
) -> None:
    if isinstance(value, Mapping):
        floor = _numeric_key(
            value,
            (
                "material_support_accuracy_floor",
                "support_accuracy_floor",
                "material_support_mode_accuracy_floor",
            ),
        )
        metric = _numeric_key(
            value,
            (
                "best_accuracy",
                "accuracy",
                "best_mode_accuracy",
                "terminal_oracle_accuracy",
            ),
        )
        passed_key = next(
            (
                key
                for key, item in value.items()
                if key.startswith("materially_supports_") and isinstance(item, bool)
            ),
            None,
        )
        if floor is not None and metric is not None and passed_key is not None:
            probes.append(
                {
                    "path": path,
                    "policy": value.get("policy"),
                    "metric": _round(metric),
                    "floor": _round(floor),
                    "passed_key": passed_key,
                    "passed": bool(value.get(passed_key)),
                    "runtime_policy_status": value.get("runtime_policy_status"),
                }
            )
        for key, item in value.items():
            if key in {
                "labels",
                "branch_points",
                "branch_results",
                "action_runs",
                "trajectory_records",
                "samples",
            }:
                continue
            _collect_support_probes(item, path=f"{path}.{key}", probes=probes)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_support_probes(item, path=f"{path}[{index}]", probes=probes)


def _best_support_probe(
    probes: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not probes:
        return None
    return dict(
        max(
            probes,
            key=lambda probe: (
                1 if probe.get("passed") is True else 0,
                _float(probe.get("metric")),
                -_float(probe.get("floor")),
                str(probe.get("path", "")),
            ),
        )
    )


def _version_metrics(
    *,
    artifact_payloads: Sequence[Mapping[str, object]],
    legacy_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    merged: dict[str, object] = {}
    for source in [*legacy_records, *artifact_payloads]:
        for key in (
            "candidate_alive_delta_vs_linear",
            "candidate_alive_regression_vs_linear",
            "candidate_births_delta_vs_linear",
            "candidate_min_seed_alive_delta_vs_linear",
            "candidate_min_seed_births_delta_vs_linear",
            "candidate_dominant_requested_action_share",
            "candidate_heuristic_action_source_count",
            "promotion_blocker_count",
            "blocker_count",
            "label_count",
            "material_oracle_gain_label_count",
            "terminal_alive_gain_total_vs_logged",
            "birth_gain_total_vs_logged",
            "best_accuracy",
        ):
            if key in source:
                merged[key] = source[key]
        aggregate = source.get("aggregate")
        if isinstance(aggregate, Mapping):
            for key in (
                "label_count",
                "material_oracle_gain_label_count",
                "terminal_alive_gain_total_vs_logged",
                "birth_gain_total_vs_logged",
                "heuristic_action_source_count",
                "oracle_changed_action_count",
            ):
                if key in aggregate:
                    merged[key] = aggregate[key]
        acceptance = source.get("acceptance")
        if isinstance(acceptance, Mapping):
            blockers = acceptance.get("blockers")
            if isinstance(blockers, list):
                merged["acceptance_blocker_count"] = len(blockers)
            strict_blockers = acceptance.get("strict_blockers")
            if isinstance(strict_blockers, list):
                merged["acceptance_blocker_count"] = len(strict_blockers)
            best_rule = acceptance.get("best_rule_for_diagnostics")
            if isinstance(best_rule, Mapping):
                _merge_best_rule_metrics(merged, best_rule)
    return dict(sorted(merged.items()))


def _merge_best_rule_metrics(
    merged: dict[str, object],
    best_rule: Mapping[str, object],
) -> None:
    rule = best_rule.get("rule")
    if isinstance(rule, str):
        merged["best_rule"] = rule
    for source_key, target_key in (
        ("blocker_count", "best_rule_blocker_count"),
        ("mean_target_local_score_delta", "best_rule_mean_target_local_score_delta"),
        ("target_alive_delta_negative_count", "best_rule_target_alive_delta_negative_count"),
        ("mean_terminal_alive_delta", "best_rule_mean_terminal_alive_delta"),
        ("mean_birth_delta", "best_rule_mean_birth_delta"),
        ("dominant_predicted_action_share", "best_rule_dominant_predicted_action_share"),
        ("dominant_predicted_mode_share", "best_rule_dominant_predicted_mode_share"),
    ):
        value = best_rule.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[target_key] = value
    for source_key in (
        "seed_41_tick_113_avoided",
        "seed_41_tick_114_avoided",
    ):
        value = best_rule.get(source_key)
        if isinstance(value, bool):
            merged[source_key] = value


def _read_docs(path: Path) -> dict[int, list[dict[str, object]]]:
    if not path.exists():
        return {}
    docs: dict[int, list[dict[str, object]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for version in _line_versions(line):
            docs.setdefault(version, []).append(
                {
                    "path": str(path),
                    "line": line_number,
                    "text": line.strip()[:240],
                }
            )
    return docs


def _line_versions(line: str) -> set[int]:
    versions: set[int] = set()
    for match in _VERSION_PATTERN.finditer(line):
        prefix = line[max(0, match.start() - 6) : match.start()].lower()
        if prefix.endswith("mind "):
            continue
        versions.add(int(match.group(1)))
    return versions


def _artifact_paths(output_dir: Path) -> dict[int, list[Path]]:
    artifacts: dict[int, list[Path]] = {}
    if not output_dir.exists():
        return artifacts
    for path in sorted(output_dir.glob("mind-v3-v*.json")):
        match = _ARTIFACT_VERSION_PATTERN.search(path.name)
        if match is None:
            continue
        artifacts.setdefault(int(match.group(1)), []).append(path)
    return artifacts


def _read_legacy_ledger(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    records: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MindV3ProgressLedgerError(
                f"legacy ledger is not valid JSONL at line {len(records) + 1}"
            ) from exc
        if isinstance(payload, Mapping):
            records.append(payload)
    return records


def _record_versions(record: Mapping[str, object]) -> set[int]:
    versions: set[int] = set()
    for value in record.values():
        if isinstance(value, str):
            for match in _ARTIFACT_VERSION_PATTERN.finditer(value):
                versions.add(int(match.group(1)))
    return versions


def _load_json_artifact(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _has_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


def _truthy_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        if value.get(key) is True:
            return True
        return any(_truthy_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_truthy_key(item, key) for item in value)
    return False


def _numeric_key(
    value: Mapping[str, object],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
    return None


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _round(value: float) -> float:
    return round(float(value), 6)
