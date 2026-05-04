from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from evolution_sim.env.runtime.mating import (
    ASEXUAL_REPRODUCTION_MODE,
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    REPRODUCTIVE_STAGE_ORDER,
    SEXUAL_EXPRESSION,
    SEXUAL_REPRODUCTION_MODE,
    STAGE2_PROTO_ROLES,
    STAGE3_X_Y_Z,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
    stage_rank,
)
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTION_EVENT_SCHEMA_VERSION,
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
)
from evolution_sim.cli.foundation_gate_validators.common import (
    _as_int_count_mapping,
    _as_optional_int,
    _flag,
)


REPRODUCTIVE_EXPRESSION_VALUES = {
    ASEXUAL_REPRODUCTION_MODE,
    SEXUAL_EXPRESSION,
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
}


def _reproductive_role_readiness_flags(
    *,
    scope: str,
    reproduction: object,
) -> list[dict[str, object]]:
    if not isinstance(reproduction, Mapping):
        return []
    stage_counts = _as_int_count_mapping(reproduction.get("reproductive_stage_counts"))
    expression_counts = _as_int_count_mapping(
        reproduction.get("reproductive_expression_counts")
    )
    capability_counts = _as_int_count_mapping(
        reproduction.get("reproductive_capability_counts")
    )
    role_stage_total = stage_counts.get(STAGE2_PROTO_ROLES, 0) + stage_counts.get(
        STAGE3_X_Y_Z,
        0,
    )
    if (
        role_stage_total <= 0
        and capability_counts.get("proto_role_differentiation", 0) <= 0
        and capability_counts.get("xyz_expression", 0) <= 0
    ):
        return []

    flags: list[dict[str, object]] = []
    proto_x_count = expression_counts.get(PROTO_X_EXPRESSION, 0)
    proto_y_count = expression_counts.get(PROTO_Y_EXPRESSION, 0)
    proto_z_count = expression_counts.get(PROTO_Z_EXPRESSION, 0)
    x_count = expression_counts.get(X_EXPRESSION, 0)
    y_count = expression_counts.get(Y_EXPRESSION, 0)
    z_count = expression_counts.get(Z_EXPRESSION, 0)
    has_x_like = proto_x_count + x_count > 0
    has_y_like = proto_y_count + y_count > 0
    plastic_count = proto_z_count + z_count
    if role_stage_total > 0 and not ((has_x_like and has_y_like) or plastic_count > 0):
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.reproductive_expression_counts",
                (
                    "Role-stage reproduction is present without complementary "
                    "X/Y-like or Z-plastic expression coverage."
                ),
            )
        )
    if stage_counts.get(STAGE3_X_Y_Z, 0) > 0 and z_count <= 0:
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.reproductive_expression_counts.z_plastic",
                "Stage 3 X/Y/Z reproduction is present without alive Z-plastic expression.",
            )
        )

    ready_stage_counts = _as_int_count_mapping(
        reproduction.get("ready_by_reproductive_stage")
    )
    ready_role_stage_total = ready_stage_counts.get(
        STAGE2_PROTO_ROLES,
        0,
    ) + ready_stage_counts.get(STAGE3_X_Y_Z, 0)
    if role_stage_total > 0 and ready_role_stage_total <= 0:
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.ready_by_reproductive_stage",
                "Role-stage agents are alive but none are currently reproduction-ready.",
            )
        )

    mate_search_counts = _as_int_count_mapping(reproduction.get("mate_search_run_counts"))
    sexual_searches = mate_search_counts.get("sexual_searches", 0)
    sexual_successes = mate_search_counts.get("sexual_successes", 0)
    if sexual_searches > 0 and sexual_successes <= 0:
        fallback_counts = {
            key: mate_search_counts.get(key, 0)
            for key in (
                "fallback_expression_incompatible",
                "fallback_no_compatible_partner",
                "fallback_no_same_group_partner",
            )
        }
        if any(count > 0 for count in fallback_counts.values()):
            flags.append(
                _flag(
                    "warning",
                    scope,
                    "reproduction.mate_search_run_counts",
                    (
                        "Role-stage sexual searches occurred without successes; "
                        f"fallbacks={fallback_counts}."
                    ),
                )
            )
    if mate_search_counts.get("constraint_expression_incompatible", 0) > 0:
        flags.append(
            _flag(
                "warning",
                scope,
                "reproduction.mate_search_run_counts.constraint_expression_incompatible",
                "Mate-search diagnostics observed role expression incompatibility.",
            )
        )
    return flags

def _reproductive_group_catalog_flags(
    *,
    scope: str,
    summary: dict[str, object],
    viewer: dict[str, object],
    reproductive_group_catalog: dict[str, object],
    events: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    groups = reproductive_group_catalog.get("groups")
    agent_catalog = viewer.get("agent_catalog")
    if not isinstance(groups, dict):
        return [
            _flag(
                "error",
                scope,
                "viewer.reproductive_group_catalog.groups",
                "Viewer reproductive group catalog does not declare groups.",
            )
        ]
    if not isinstance(agent_catalog, dict):
        return [
            _flag(
                "error",
                scope,
                "viewer.agent_catalog",
                "Viewer agent catalog is missing while validating reproductive groups.",
            )
        ]

    event_birth_counts, event_flags = _reproductive_group_birth_counts_from_events(
        scope=scope,
        events=events,
        groups=groups,
    )
    flags.extend(event_flags)

    member_counts: dict[str, int] = {}
    alive_member_counts: dict[str, int] = {}
    alive_stage_counts: dict[str, dict[str, int]] = {}
    alive_expression_counts: dict[str, dict[str, int]] = {}
    for raw_agent_id, agent_payload in agent_catalog.items():
        if not isinstance(agent_payload, dict):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog",
                    f"Agent catalog entry {raw_agent_id} is not an object.",
                )
            )
            continue
        group_id = agent_payload.get("reproductive_group_id")
        reproductive_stage = agent_payload.get("reproductive_stage")
        reproductive_expression = agent_payload.get("reproductive_expression")
        if "death_tick" not in agent_payload:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.death_tick",
                    f"Agent catalog entry {raw_agent_id} is missing death_tick.",
                )
            )
        elif (
            agent_payload.get("death_tick") is not None
            and (
                _as_optional_int(agent_payload.get("death_tick")) is None
                or int(agent_payload["death_tick"]) < 0
            )
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.death_tick",
                    (
                        f"Agent catalog entry {raw_agent_id} death_tick must "
                        "be a nonnegative integer or null."
                    ),
                )
            )
        if reproductive_stage not in REPRODUCTIVE_STAGE_ORDER:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_stage",
                    (
                        f"Agent catalog entry {raw_agent_id} has unknown "
                        "reproductive stage."
                    ),
                )
            )
        if reproductive_expression not in REPRODUCTIVE_EXPRESSION_VALUES:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_expression",
                    (
                        f"Agent catalog entry {raw_agent_id} has unknown "
                        "reproductive expression."
                    ),
                )
            )
        if group_id is None:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_group_id",
                    f"Agent catalog entry {raw_agent_id} is missing a reproductive group.",
                )
            )
            continue
        parsed_group_id = _as_optional_int(group_id)
        if parsed_group_id is None or parsed_group_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_group_id",
                    (
                        f"Agent catalog entry {raw_agent_id} reproductive_group_id "
                        "must be a nonnegative integer."
                    ),
                )
            )
            continue
        group_key = str(group_id)
        if group_key not in groups:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.agent_catalog.reproductive_group_id",
                    (
                        f"Agent catalog entry {raw_agent_id} references missing "
                        f"reproductive group {group_key}."
                    ),
                )
            )
            continue
        member_counts[group_key] = member_counts.get(group_key, 0) + 1
        if agent_payload.get("death_tick") is None:
            alive_member_counts[group_key] = alive_member_counts.get(group_key, 0) + 1
            _increment_nested_count(
                alive_stage_counts,
                group_key,
                str(reproductive_stage),
            )
            _increment_nested_count(
                alive_expression_counts,
                group_key,
                str(reproductive_expression),
            )

    for group_key, group_payload in groups.items():
        if not isinstance(group_payload, dict):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups",
                    f"Reproductive group {group_key} is not an object.",
                )
            )
            continue
        parsed_payload_group_id = _as_optional_int(group_payload.get("group_id"))
        if parsed_payload_group_id is None or parsed_payload_group_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.group_id",
                    f"Reproductive group {group_key} group_id must be a nonnegative integer.",
                )
            )
        elif str(parsed_payload_group_id) != str(group_key):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.group_id",
                    f"Reproductive group {group_key} group_id does not match its key.",
                )
            )
        expected_member_count = member_counts.get(str(group_key), 0)
        expected_alive_member_count = alive_member_counts.get(str(group_key), 0)
        member_count = _as_optional_int(group_payload.get("member_count"))
        alive_member_count = _as_optional_int(group_payload.get("alive_member_count"))
        actual_stage_counts = _as_count_dict(group_payload.get("alive_stage_counts"))
        actual_expression_counts = _as_count_dict(
            group_payload.get("alive_expression_counts")
        )
        if member_count != expected_member_count:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.member_count",
                    (
                        f"Reproductive group {group_key} member_count does not "
                        "match the agent catalog."
                    ),
                )
            )
        if alive_member_count != expected_alive_member_count:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.alive_member_count",
                    (
                        f"Reproductive group {group_key} alive_member_count does "
                        "not match the agent catalog."
                    ),
                )
            )
        expected_stage_counts = alive_stage_counts.get(str(group_key), {})
        expected_expression_counts = alive_expression_counts.get(str(group_key), {})
        if actual_stage_counts != expected_stage_counts:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.alive_stage_counts",
                    (
                        f"Reproductive group {group_key} alive_stage_counts does "
                        "not match the agent catalog."
                    ),
                )
            )
        if actual_expression_counts != expected_expression_counts:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.alive_expression_counts",
                    (
                        f"Reproductive group {group_key} alive_expression_counts "
                        "does not match the agent catalog."
                    ),
                )
            )
        birth_counts = {
            field: _as_optional_int(group_payload.get(field))
            for field in ("asexual_births", "sexual_births", "hybrid_births")
        }
        group_stage = group_payload.get("stage")
        if group_stage not in REPRODUCTIVE_STAGE_ORDER:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.stage",
                    f"Reproductive group {group_key} has unknown stage.",
                )
            )
        for field, count in birth_counts.items():
            if count is None or count < 0:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        f"viewer.reproductive_group_catalog.groups.{field}",
                        (
                            f"Reproductive group {group_key} {field} must be a "
                            "nonnegative integer."
                        ),
                    )
                )
        if event_birth_counts is not None:
            expected_event_births = event_birth_counts.get(str(group_key), {})
            for field, count in birth_counts.items():
                if count is None or count < 0:
                    continue
                expected_count = expected_event_births.get(field, 0)
                if count != expected_count:
                    flags.append(
                        _flag(
                            "error",
                            scope,
                            f"viewer.reproductive_group_catalog.groups.{field}",
                            (
                                f"Reproductive group {group_key} {field} does "
                                "not match agent_reproduced events."
                            ),
                        )
                    )
        sexual_births = birth_counts.get("sexual_births")
        hybrid_births = birth_counts.get("hybrid_births")
        if (
            sexual_births is not None
            and hybrid_births is not None
            and hybrid_births > sexual_births
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.hybrid_births",
                    (
                        f"Reproductive group {group_key} hybrid_births exceeds "
                        "sexual_births."
                    ),
                )
            )
        highest_alive_stage = _highest_counted_stage(expected_stage_counts)
        if (
            highest_alive_stage is not None
            and isinstance(group_stage, str)
            and stage_rank(group_stage) < stage_rank(highest_alive_stage)
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.reproductive_group_catalog.groups.stage",
                    (
                        f"Reproductive group {group_key} stage is below an "
                        "alive member stage."
                    ),
                )
            )
    if event_birth_counts is not None:
        flags.extend(
            _reproductive_summary_birth_flags(
                scope=scope,
                summary=summary,
                event_birth_counts=event_birth_counts,
            )
        )
    flags.extend(
        _reproductive_summary_catalog_flags(
            scope=scope,
            summary=summary,
            groups=groups,
            alive_member_counts=alive_member_counts,
            alive_stage_counts=alive_stage_counts,
            alive_expression_counts=alive_expression_counts,
        )
    )
    return flags


def _reproductive_summary_birth_flags(
    *,
    scope: str,
    summary: dict[str, object],
    event_birth_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    event_totals = {
        field: sum(
            int(group_counts.get(field, 0))
            for group_counts in event_birth_counts.values()
        )
        for field in ("asexual_births", "sexual_births", "hybrid_births")
    }
    event_birth_total = event_totals["asexual_births"] + event_totals["sexual_births"]
    summary_births = _as_optional_int(summary.get("births"))
    if summary_births is None or summary_births != event_birth_total:
        flags.append(
            _flag(
                "error",
                scope,
                "summary.births",
                "Summary births does not match agent_reproduced events.",
            )
        )
    reproductive_groups_end = summary.get("reproductive_groups_end")
    if not isinstance(reproductive_groups_end, Mapping):
        flags.append(
            _flag(
                "error",
                scope,
                "summary.reproductive_groups_end",
                "Full replay summary is missing reproductive group totals.",
            )
        )
        return flags
    if (
        reproductive_groups_end.get("schema_version")
        != REPRODUCTIVE_GROUP_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "summary.reproductive_groups_end.schema_version",
                "Summary reproductive group totals are missing or stale.",
            )
        )
    for field, expected_count in event_totals.items():
        summary_count = _as_optional_int(reproductive_groups_end.get(field))
        if summary_count is None or summary_count != expected_count:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"summary.reproductive_groups_end.{field}",
                    (
                        f"Summary reproductive group {field} does not match "
                        "agent_reproduced events."
                    ),
                )
            )
    return flags


def _reproductive_summary_catalog_flags(
    *,
    scope: str,
    summary: dict[str, object],
    groups: Mapping[object, object],
    alive_member_counts: Mapping[str, int],
    alive_stage_counts: Mapping[str, Mapping[str, int]],
    alive_expression_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, object]]:
    reproductive_groups_end = summary.get("reproductive_groups_end")
    if not isinstance(reproductive_groups_end, Mapping):
        return []

    stage_counts: dict[str, int] = {}
    for group_payload in groups.values():
        if not isinstance(group_payload, Mapping):
            continue
        stage = group_payload.get("stage")
        if isinstance(stage, str):
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    catalog_totals = {
        "group_count": len(groups),
        "alive_group_count": sum(
            1 for count in alive_member_counts.values() if count > 0
        ),
        "stage_counts": stage_counts,
        "alive_stage_counts": _sum_grouped_counts(alive_stage_counts),
        "alive_expression_counts": _sum_grouped_counts(alive_expression_counts),
    }
    flags: list[dict[str, object]] = []
    for field in ("group_count", "alive_group_count"):
        summary_count = _as_optional_int(reproductive_groups_end.get(field))
        if summary_count is None or summary_count != catalog_totals[field]:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"summary.reproductive_groups_end.{field}",
                    (
                        f"Summary reproductive group {field} does not match "
                        "the viewer catalog."
                    ),
                )
            )
    for field in ("stage_counts", "alive_stage_counts", "alive_expression_counts"):
        summary_counts = _as_count_dict(reproductive_groups_end.get(field))
        if summary_counts != catalog_totals[field]:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"summary.reproductive_groups_end.{field}",
                    (
                        f"Summary reproductive group {field} does not match "
                        "the viewer catalog."
                    ),
                )
            )
    return flags


def _reproductive_group_birth_counts_from_events(
    *,
    scope: str,
    events: Sequence[object] | None,
    groups: Mapping[object, object],
) -> tuple[dict[str, dict[str, int]] | None, list[dict[str, object]]]:
    flags: list[dict[str, object]] = []
    if events is None:
        return None, flags
    if isinstance(events, (str, bytes, bytearray)):
        return None, [
            _flag(
                "error",
                scope,
                "events",
                "Full replay events must be a sequence of event objects.",
            )
        ]

    group_keys = {str(group_key) for group_key in groups}
    counts: dict[str, dict[str, int]] = {
        group_key: {
            "asexual_births": 0,
            "sexual_births": 0,
            "hybrid_births": 0,
        }
        for group_key in group_keys
    }
    multi_offspring_groups: dict[tuple[int, ...], dict[str, object]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events",
                    f"Full replay event {index} is not an object.",
                )
            )
            continue
        if event.get("type") != "agent_reproduced":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.data",
                    f"Reproduction event {index} is missing a data object.",
                )
            )
            continue
        if data.get("schema_version") != REPRODUCTION_EVENT_SCHEMA_VERSION:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.schema_version",
                    (
                        f"Reproduction event {index} must declare "
                        f"{REPRODUCTION_EVENT_SCHEMA_VERSION}."
                    ),
                )
            )
        raw_group_id = data.get("child_reproductive_group_id")
        group_id = _as_optional_int(raw_group_id)
        if group_id is None or group_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.child_reproductive_group_id",
                    (
                        f"Reproduction event {index} child_reproductive_group_id "
                        "must be a nonnegative integer."
                    ),
                )
            )
            continue
        group_key = str(group_id)
        if group_key not in group_keys:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.child_reproductive_group_id",
                    (
                        f"Reproduction event {index} references missing "
                        f"reproductive group {group_key}."
                    ),
                )
            )
            continue
        reproduction_mode = data.get("reproduction_mode")
        hybrid = data.get("hybrid")
        if not isinstance(hybrid, bool):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.hybrid",
                    f"Reproduction event {index} hybrid must be boolean.",
                )
            )
            hybrid = False
        flags.extend(
            _reproduction_event_offspring_metadata_flags(
                scope=scope,
                index=index,
                data=data,
                reproduction_mode=reproduction_mode,
                multi_offspring_groups=multi_offspring_groups,
            )
        )
        if reproduction_mode == ASEXUAL_REPRODUCTION_MODE:
            counts[group_key]["asexual_births"] += 1
            if hybrid:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        "events.agent_reproduced.hybrid",
                        (
                            f"Reproduction event {index} cannot mark an "
                            "asexual birth as hybrid."
                        ),
                    )
                )
        elif reproduction_mode == SEXUAL_REPRODUCTION_MODE:
            counts[group_key]["sexual_births"] += 1
            if hybrid:
                counts[group_key]["hybrid_births"] += 1
        else:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.reproduction_mode",
                    (
                        f"Reproduction event {index} has unsupported "
                        f"reproduction_mode {reproduction_mode!r}."
                    ),
                )
            )
    flags.extend(
        _multi_offspring_group_flags(
            scope=scope,
            multi_offspring_groups=multi_offspring_groups,
        )
    )
    return counts, flags


def _reproduction_event_offspring_metadata_flags(
    *,
    scope: str,
    index: int,
    data: Mapping[object, object],
    reproduction_mode: object,
    multi_offspring_groups: dict[tuple[int, ...], dict[str, object]],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    child_id = _as_optional_int(data.get("child_id"))
    if child_id is None or child_id < 0:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.child_id",
                f"Reproduction event {index} child_id must be a nonnegative integer.",
            )
        )
        return flags
    offspring_count = _as_optional_int(data.get("offspring_count"))
    if offspring_count is None or offspring_count <= 0:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.offspring_count",
                (
                    f"Reproduction event {index} offspring_count must be a "
                    "positive integer."
                ),
            )
        )
        return flags
    flags.extend(
        _multi_offspring_attempt_metadata_flags(
            scope=scope,
            index=index,
            data=data,
            reproduction_mode=reproduction_mode,
            offspring_count=offspring_count,
        )
    )
    if offspring_count == 1:
        if data.get("multi_offspring") is True:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.multi_offspring",
                    (
                        f"Reproduction event {index} single-child birth cannot "
                        "be marked multi_offspring."
                    ),
                )
            )
        for field in (
            "offspring_index",
            "sibling_child_ids",
            "parent_energy_costs_total",
        ):
            if field in data:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        f"events.agent_reproduced.{field}",
                        (
                            f"Reproduction event {index} single-child birth "
                            f"cannot carry {field}."
                        ),
                    )
                )
        return flags
    if reproduction_mode != SEXUAL_REPRODUCTION_MODE:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.reproduction_mode",
                f"Reproduction event {index} multi-offspring birth must be sexual.",
            )
        )
    if data.get("multi_offspring") is not True:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring",
                (
                    f"Reproduction event {index} multi-offspring birth must set "
                    "multi_offspring true."
                ),
            )
        )
    offspring_index = _as_optional_int(data.get("offspring_index"))
    if (
        offspring_index is None
        or offspring_index <= 0
        or offspring_index > offspring_count
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.offspring_index",
                (
                    f"Reproduction event {index} offspring_index must be within "
                    "offspring_count."
                ),
            )
        )
        return flags
    raw_sibling_ids = data.get("sibling_child_ids")
    if not isinstance(raw_sibling_ids, Sequence) or isinstance(
        raw_sibling_ids, (str, bytes, bytearray)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.sibling_child_ids",
                f"Reproduction event {index} sibling_child_ids must be a sequence.",
            )
        )
        return flags
    sibling_ids: list[int] = []
    for raw_child_id in raw_sibling_ids:
        parsed_child_id = _as_optional_int(raw_child_id)
        if parsed_child_id is None or parsed_child_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.sibling_child_ids",
                    (
                        f"Reproduction event {index} sibling_child_ids must "
                        "contain nonnegative integers."
                    ),
                )
            )
            return flags
        sibling_ids.append(parsed_child_id)
    if len(sibling_ids) != offspring_count:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.sibling_child_ids",
                (
                    f"Reproduction event {index} sibling_child_ids must contain "
                    "offspring_count entries."
                ),
            )
        )
        return flags
    if len(set(sibling_ids)) != len(sibling_ids):
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.sibling_child_ids",
                (
                    f"Reproduction event {index} sibling_child_ids cannot "
                    "contain duplicates."
                ),
            )
        )
        return flags
    if child_id not in sibling_ids:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.sibling_child_ids",
                (
                    f"Reproduction event {index} sibling_child_ids must include "
                    "child_id."
                ),
            )
        )
        return flags
    if sibling_ids[offspring_index - 1] != child_id:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.offspring_index",
                (
                    f"Reproduction event {index} offspring_index must identify "
                    "child_id within sibling_child_ids."
                ),
            )
        )
    flags.extend(
        _parent_energy_costs_total_flags(
            scope=scope,
            index=index,
            data=data,
        )
    )
    sibling_key = tuple(sibling_ids)
    group = multi_offspring_groups.setdefault(
        sibling_key,
        {
            "offspring_count": offspring_count,
            "child_ids": set(),
            "offspring_indexes": set(),
        },
    )
    if group["offspring_count"] != offspring_count:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring",
                (
                    f"Reproduction event {index} sibling group has inconsistent "
                    "offspring_count values."
                ),
            )
        )
    child_ids = group["child_ids"]
    offspring_indexes = group["offspring_indexes"]
    if isinstance(child_ids, set):
        if child_id in child_ids:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.child_id",
                    (
                        f"Reproduction event {index} repeats child_id {child_id} "
                        "within a multi-offspring sibling group."
                    ),
                )
            )
        child_ids.add(child_id)
    if isinstance(offspring_indexes, set):
        if offspring_index in offspring_indexes:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.offspring_index",
                    (
                        f"Reproduction event {index} repeats offspring_index "
                        f"{offspring_index} within a multi-offspring sibling group."
                    ),
                )
            )
        offspring_indexes.add(offspring_index)
    return flags


def _multi_offspring_attempt_metadata_flags(
    *,
    scope: str,
    index: int,
    data: Mapping[object, object],
    reproduction_mode: object,
    offspring_count: int,
) -> list[dict[str, object]]:
    fields = (
        "multi_offspring_desired_count",
        "multi_offspring_actual_count",
        "multi_offspring_limit_reasons",
    )
    if not any(field in data for field in fields):
        return []
    flags: list[dict[str, object]] = []
    if reproduction_mode != SEXUAL_REPRODUCTION_MODE:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.reproduction_mode",
                (
                    f"Reproduction event {index} multi-offspring attempt "
                    "metadata must be sexual."
                ),
            )
        )
    desired_count = _as_optional_int(data.get("multi_offspring_desired_count"))
    actual_count = _as_optional_int(data.get("multi_offspring_actual_count"))
    if desired_count is None or desired_count <= 1:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_desired_count",
                (
                    f"Reproduction event {index} multi_offspring_desired_count "
                    "must be an integer above one."
                ),
            )
        )
    if actual_count is None or actual_count <= 0:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_actual_count",
                (
                    f"Reproduction event {index} multi_offspring_actual_count "
                    "must be a positive integer."
                ),
            )
        )
    if desired_count is None or actual_count is None:
        return flags
    if actual_count != offspring_count:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_actual_count",
                (
                    f"Reproduction event {index} multi_offspring_actual_count "
                    "must match offspring_count."
                ),
            )
        )
    if desired_count < actual_count:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_desired_count",
                (
                    f"Reproduction event {index} multi_offspring_desired_count "
                    "cannot be below actual count."
                ),
            )
        )
    raw_limit_reasons = data.get("multi_offspring_limit_reasons")
    if not isinstance(raw_limit_reasons, Sequence) or isinstance(
        raw_limit_reasons,
        (str, bytes, bytearray),
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_limit_reasons",
                (
                    f"Reproduction event {index} multi_offspring_limit_reasons "
                    "must be a sequence."
                ),
            )
        )
        return flags
    limit_reasons = [reason for reason in raw_limit_reasons if isinstance(reason, str)]
    if len(limit_reasons) != len(raw_limit_reasons):
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_limit_reasons",
                (
                    f"Reproduction event {index} multi_offspring_limit_reasons "
                    "must contain strings."
                ),
            )
        )
        return flags
    if desired_count > actual_count and not limit_reasons:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_limit_reasons",
                (
                    f"Reproduction event {index} clamped multi-offspring attempt "
                    "must include a limit reason."
                ),
            )
        )
    if desired_count == actual_count and limit_reasons:
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.multi_offspring_limit_reasons",
                (
                    f"Reproduction event {index} unclamped multi-offspring "
                    "attempt cannot include limit reasons."
                ),
            )
        )
    return flags


def _parent_energy_costs_total_flags(
    *,
    scope: str,
    index: int,
    data: Mapping[object, object],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    raw_costs = data.get("parent_energy_costs_total")
    if not isinstance(raw_costs, Sequence) or isinstance(
        raw_costs, (str, bytes, bytearray)
    ):
        return [
            _flag(
                "error",
                scope,
                "events.agent_reproduced.parent_energy_costs_total",
                (
                    f"Reproduction event {index} parent_energy_costs_total must "
                    "be a sequence."
                ),
            )
        ]
    parent_ids = data.get("parent_ids")
    if (
        isinstance(parent_ids, Sequence)
        and not isinstance(parent_ids, (str, bytes, bytearray))
        and len(raw_costs) != len(parent_ids)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "events.agent_reproduced.parent_energy_costs_total",
                (
                    f"Reproduction event {index} parent_energy_costs_total must "
                    "match parent_ids length."
                ),
            )
        )
    for entry in raw_costs:
        if not isinstance(entry, Mapping):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.parent_energy_costs_total",
                    (
                        f"Reproduction event {index} parent_energy_costs_total "
                        "entries must be objects."
                    ),
                )
            )
            continue
        agent_id = _as_optional_int(entry.get("agent_id"))
        energy_cost = entry.get("energy_cost")
        if agent_id is None or agent_id < 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.parent_energy_costs_total",
                    (
                        f"Reproduction event {index} parent_energy_costs_total "
                        "agent_id must be a nonnegative integer."
                    ),
                )
            )
        if (
            not isinstance(energy_cost, (int, float))
            or isinstance(energy_cost, bool)
            or not math.isfinite(float(energy_cost))
            or energy_cost < 0
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.parent_energy_costs_total",
                    (
                        f"Reproduction event {index} parent_energy_costs_total "
                        "energy_cost must be a nonnegative number."
                    ),
                )
            )
    return flags


def _multi_offspring_group_flags(
    *,
    scope: str,
    multi_offspring_groups: Mapping[tuple[int, ...], Mapping[str, object]],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for sibling_ids, group in multi_offspring_groups.items():
        offspring_count = group.get("offspring_count")
        child_ids = group.get("child_ids")
        offspring_indexes = group.get("offspring_indexes")
        if not isinstance(offspring_count, int):
            continue
        if (
            not isinstance(child_ids, set)
            or not isinstance(offspring_indexes, set)
            or len(child_ids) != offspring_count
            or len(offspring_indexes) != offspring_count
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "events.agent_reproduced.multi_offspring",
                    (
                        "Multi-offspring sibling group "
                        f"{list(sibling_ids)} has incomplete child events."
                    ),
                )
            )
    return flags


def _increment_nested_count(
    counts: dict[str, dict[str, int]],
    group_key: str,
    field_value: str,
) -> None:
    group_counts = counts.setdefault(group_key, {})
    group_counts[field_value] = group_counts.get(field_value, 0) + 1


def _sum_grouped_counts(
    grouped_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counts in grouped_counts.values():
        for key, count in counts.items():
            totals[key] = totals.get(key, 0) + int(count)
    return totals


def _highest_counted_stage(stage_counts: dict[str, int]) -> str | None:
    stages = [stage for stage, count in stage_counts.items() if count > 0]
    if not stages:
        return None
    return max(stages, key=stage_rank)


def _as_count_dict(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    counts: dict[str, int] = {}
    for key, count in value.items():
        parsed_count = _as_optional_int(count)
        if parsed_count is None or parsed_count < 0:
            return None
        counts[str(key)] = parsed_count
    return counts
