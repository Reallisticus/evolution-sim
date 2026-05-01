from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from evolution_sim.config import ReproductionConfig
from evolution_sim.env.runtime.state import Agent
from evolution_sim.genome.schema import Genome

STAGE1_FACULTATIVE_SEX = "stage1_facultative_sex"
STAGE2_PROTO_ROLES = "stage2_proto_roles"
STAGE3_X_Y_Z = "stage3_x_y_z"
REPRODUCTIVE_STAGE_ORDER = (
    "stage0_asexual",
    STAGE1_FACULTATIVE_SEX,
    STAGE2_PROTO_ROLES,
    STAGE3_X_Y_Z,
    "stage4_hybridization",
)
SEXUAL_EXPRESSION = "same_group_compatible"
PROTO_X_EXPRESSION = "proto_x_like"
PROTO_Y_EXPRESSION = "proto_y_like"
PROTO_Z_EXPRESSION = "proto_z_plastic"
X_EXPRESSION = "x"
Y_EXPRESSION = "y"
Z_EXPRESSION = "z_plastic"
SEXUAL_REPRODUCTION_MODE = "same_group_sexual"
ASEXUAL_REPRODUCTION_MODE = "asexual"
MATE_SEARCH_BLOCK_REASON_KEYS = (
    "different_group",
    "partner_sexual_locked",
    "partner_out_of_radius",
    "partner_not_ready",
    "expression_incompatible",
)


@dataclass(frozen=True, slots=True)
class MateCandidate:
    agent: Agent
    distance: int
    compatibility_score: float
    inbreeding_penalty: float


@dataclass(frozen=True, slots=True)
class MateSearchReport:
    selected: MateCandidate | None
    scanned_agents: int
    same_group_candidates: int
    same_group_sexual_candidates: int
    in_radius_candidates: int
    biologically_ready_candidates: int
    reason_counts: dict[str, int]
    constraint_counts: dict[str, int]
    expression_compatible_candidates: int
    expression_incompatible_candidates: int


def sexual_reproduction_unlocked(
    genome: Genome,
    config: ReproductionConfig,
) -> bool:
    if not config.sexual_reproduction_enabled:
        return False
    reproductive = genome.reproductive
    return (
        reproductive.sexual_reproduction_drive >= config.sexual_drive_threshold
        and reproductive.recombination_affinity
        >= config.sexual_recombination_threshold
    )


def role_differentiation_unlocked(
    genome: Genome,
    config: ReproductionConfig,
) -> bool:
    return (
        sexual_reproduction_unlocked(genome, config)
        and genome.reproductive.role_differentiation_drive
        >= config.role_differentiation_threshold
    )


def xyz_expression_unlocked(
    genome: Genome,
    config: ReproductionConfig,
) -> bool:
    return (
        role_differentiation_unlocked(genome, config)
        and genome.reproductive.role_differentiation_drive
        >= config.xyz_expression_threshold
    )


def reproductive_capabilities_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> dict[str, bool]:
    return {
        "sexual_reproduction": sexual_reproduction_unlocked(genome, config),
        "proto_role_differentiation": role_differentiation_unlocked(genome, config),
        "xyz_expression": xyz_expression_unlocked(genome, config),
        "hybridization": False,
        "multi_offspring": False,
    }


def reproductive_stage_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> str:
    if xyz_expression_unlocked(genome, config):
        return STAGE3_X_Y_Z
    if role_differentiation_unlocked(genome, config):
        return STAGE2_PROTO_ROLES
    if sexual_reproduction_unlocked(genome, config):
        return STAGE1_FACULTATIVE_SEX
    return "stage0_asexual"


def reproductive_expression_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> str:
    if xyz_expression_unlocked(genome, config):
        return _xyz_expression_for_genome(genome, config)
    if role_differentiation_unlocked(genome, config):
        return _proto_role_expression_for_genome(genome, config)
    if sexual_reproduction_unlocked(genome, config):
        return SEXUAL_EXPRESSION
    return ASEXUAL_REPRODUCTION_MODE


def stage_rank(stage: str) -> int:
    try:
        return REPRODUCTIVE_STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def max_reproductive_stage(stages: Iterable[str]) -> str:
    return max(stages, key=stage_rank, default="stage0_asexual")


def choose_same_group_mate(
    parent: Agent,
    agents: Iterable[Agent],
    *,
    config: ReproductionConfig,
    biologically_ready: Callable[[Agent], bool],
) -> MateCandidate | None:
    return same_group_mate_search_report(
        parent,
        agents,
        config=config,
        biologically_ready=biologically_ready,
    ).selected


def same_group_mate_search_report(
    parent: Agent,
    agents: Iterable[Agent],
    *,
    config: ReproductionConfig,
    biologically_ready: Callable[[Agent], bool],
) -> MateSearchReport:
    if not sexual_reproduction_unlocked(parent.genome, config):
        return MateSearchReport(
            selected=None,
            scanned_agents=0,
            same_group_candidates=0,
            same_group_sexual_candidates=0,
            in_radius_candidates=0,
            biologically_ready_candidates=0,
            reason_counts=_empty_mate_search_reason_counts(),
            constraint_counts=_empty_mate_search_reason_counts(),
            expression_compatible_candidates=0,
            expression_incompatible_candidates=0,
        )
    candidates: list[MateCandidate] = []
    reason_counts = _empty_mate_search_reason_counts()
    constraint_counts = _empty_mate_search_reason_counts()
    scanned_agents = 0
    same_group_candidates = 0
    same_group_sexual_candidates = 0
    in_radius_candidates = 0
    biologically_ready_candidates = 0
    expression_compatible_candidates = 0
    for agent in agents:
        candidate, block_reason, constraints = _candidate_for_with_reason(
            parent,
            agent,
            config,
            biologically_ready,
        )
        if block_reason == "ignored":
            continue
        scanned_agents += 1
        for constraint in constraints:
            constraint_counts[constraint] += 1
        if "different_group" not in constraints:
            same_group_candidates += 1
            if "partner_sexual_locked" not in constraints:
                same_group_sexual_candidates += 1
                if "expression_incompatible" not in constraints:
                    expression_compatible_candidates += 1
                if "partner_out_of_radius" not in constraints:
                    in_radius_candidates += 1
                    if "partner_not_ready" not in constraints:
                        biologically_ready_candidates += 1
        if candidate is not None:
            candidates.append(candidate)
            continue
        if block_reason in reason_counts:
            reason_counts[block_reason] += 1

    selected = None
    if candidates:
        selected = sorted(
            candidates,
            key=lambda item: (
                -item.compatibility_score,
                item.distance,
                item.agent.agent_id,
            ),
        )[0]
    return MateSearchReport(
        selected=selected,
        scanned_agents=scanned_agents,
        same_group_candidates=same_group_candidates,
        same_group_sexual_candidates=same_group_sexual_candidates,
        in_radius_candidates=in_radius_candidates,
        biologically_ready_candidates=biologically_ready_candidates,
        reason_counts=reason_counts,
        constraint_counts=constraint_counts,
        expression_compatible_candidates=expression_compatible_candidates,
        expression_incompatible_candidates=constraint_counts["expression_incompatible"],
    )


def _empty_mate_search_reason_counts() -> dict[str, int]:
    return {reason: 0 for reason in MATE_SEARCH_BLOCK_REASON_KEYS}


def _candidate_for(
    parent: Agent,
    agent: Agent,
    config: ReproductionConfig,
    biologically_ready: Callable[[Agent], bool],
) -> MateCandidate | None:
    candidate, _, _ = _candidate_for_with_reason(
        parent,
        agent,
        config,
        biologically_ready,
    )
    return candidate


def _candidate_for_with_reason(
    parent: Agent,
    agent: Agent,
    config: ReproductionConfig,
    biologically_ready: Callable[[Agent], bool],
) -> tuple[MateCandidate | None, str, tuple[str, ...]]:
    if agent.agent_id == parent.agent_id or not agent.alive:
        return None, "ignored", ()
    if (agent.reproductive_group_id or agent.lineage_id) != (
        parent.reproductive_group_id or parent.lineage_id
    ):
        return None, "different_group", ("different_group",)
    constraints: list[str] = []
    partner_sexual_unlocked = sexual_reproduction_unlocked(agent.genome, config)
    if not partner_sexual_unlocked:
        constraints.append("partner_sexual_locked")
    distance = abs(agent.x - parent.x) + abs(agent.y - parent.y)
    if distance > config.sexual_partner_radius:
        constraints.append("partner_out_of_radius")
    if not biologically_ready(agent):
        constraints.append("partner_not_ready")
    if partner_sexual_unlocked:
        expression_allowed, _ = expression_compatibility(parent, agent, config)
        if not expression_allowed:
            constraints.append("expression_incompatible")
    if constraints:
        return None, _primary_candidate_block_reason(constraints), tuple(constraints)
    penalty = close_kinship_penalty(parent, agent)
    return (
        MateCandidate(
            agent=agent,
            distance=distance,
            compatibility_score=_compatibility_score(parent, agent, distance, config),
            inbreeding_penalty=penalty,
        ),
        "compatible",
        (),
    )


def _primary_candidate_block_reason(constraints: Iterable[str]) -> str:
    constraint_set = set(constraints)
    for reason in MATE_SEARCH_BLOCK_REASON_KEYS:
        if reason in constraint_set:
            return reason
    return "partner_not_ready"


def close_kinship_penalty(left: Agent, right: Agent) -> float:
    left_parents = _known_parent_ids(left)
    right_parents = _known_parent_ids(right)
    if left.agent_id in right_parents or right.agent_id in left_parents:
        return 1.0
    if left_parents and right_parents and left_parents & right_parents:
        return 0.85
    if left.lineage_id == right.lineage_id:
        return 0.25
    return 0.0


def _known_parent_ids(agent: Agent) -> set[int]:
    parent_ids = {agent.parent_id, agent.secondary_parent_id}
    return {int(parent_id) for parent_id in parent_ids if parent_id is not None}


def _compatibility_score(
    left: Agent,
    right: Agent,
    distance: int,
    config: ReproductionConfig,
) -> float:
    left_reproductive = left.genome.reproductive
    right_reproductive = right.genome.reproductive
    drive = (
        left_reproductive.sexual_reproduction_drive
        + right_reproductive.sexual_reproduction_drive
    ) / 2.0
    affinity = (
        left_reproductive.recombination_affinity
        + right_reproductive.recombination_affinity
    ) / 2.0
    tolerance = (
        left_reproductive.hybridization_tolerance
        + right_reproductive.hybridization_tolerance
    ) / 2.0
    distance_penalty = distance / max(config.sexual_partner_radius, 1) * 0.08
    kinship_penalty = close_kinship_penalty(left, right) * 0.04
    _, expression_modifier = expression_compatibility(left, right, config)
    return round(
        drive * 0.45
        + affinity * 0.45
        + tolerance * 0.1
        + expression_modifier
        - distance_penalty
        - kinship_penalty,
        6,
    )


def expression_compatibility(
    left: Agent,
    right: Agent,
    config: ReproductionConfig,
) -> tuple[bool, float]:
    left_expression = left.reproductive_expression
    right_expression = right.reproductive_expression
    if ASEXUAL_REPRODUCTION_MODE in {left_expression, right_expression}:
        return False, 0.0
    if SEXUAL_EXPRESSION in {left_expression, right_expression}:
        return True, 0.0

    left_role = _expression_role(left_expression)
    right_role = _expression_role(right_expression)
    if left_role is None or right_role is None:
        return False, 0.0
    if left_role == "plastic" and right_role == "plastic":
        return True, -float(config.z_z_pairing_penalty)
    if "plastic" in {left_role, right_role}:
        return True, float(config.role_complementarity_bonus) * 0.5
    if left_role != right_role:
        return True, float(config.role_complementarity_bonus)
    return False, 0.0


def _proto_role_expression_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> str:
    if genome.reproductive.sex_plasticity >= config.z_plasticity_threshold:
        return PROTO_Z_EXPRESSION
    if genome.reproductive.sex_expression_bias < 0:
        return PROTO_X_EXPRESSION
    return PROTO_Y_EXPRESSION


def _xyz_expression_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> str:
    if genome.reproductive.sex_plasticity >= config.z_plasticity_threshold:
        return Z_EXPRESSION
    if genome.reproductive.sex_expression_bias < 0:
        return X_EXPRESSION
    return Y_EXPRESSION


def _expression_role(expression: str) -> str | None:
    if expression in {PROTO_X_EXPRESSION, X_EXPRESSION}:
        return "x"
    if expression in {PROTO_Y_EXPRESSION, Y_EXPRESSION}:
        return "y"
    if expression in {PROTO_Z_EXPRESSION, Z_EXPRESSION}:
        return "plastic"
    return None
