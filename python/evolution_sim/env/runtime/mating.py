from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from evolution_sim.config import ReproductionConfig
from evolution_sim.env.runtime.state import Agent
from evolution_sim.genome.schema import Genome

STAGE1_FACULTATIVE_SEX = "stage1_facultative_sex"
SEXUAL_EXPRESSION = "same_group_compatible"
SEXUAL_REPRODUCTION_MODE = "same_group_sexual"
ASEXUAL_REPRODUCTION_MODE = "asexual"


@dataclass(frozen=True, slots=True)
class MateCandidate:
    agent: Agent
    distance: int
    compatibility_score: float
    inbreeding_penalty: float


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


def reproductive_stage_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> str:
    if sexual_reproduction_unlocked(genome, config):
        return STAGE1_FACULTATIVE_SEX
    return "stage0_asexual"


def reproductive_expression_for_genome(
    genome: Genome,
    config: ReproductionConfig,
) -> str:
    if sexual_reproduction_unlocked(genome, config):
        return SEXUAL_EXPRESSION
    return ASEXUAL_REPRODUCTION_MODE


def choose_same_group_mate(
    parent: Agent,
    agents: Iterable[Agent],
    *,
    config: ReproductionConfig,
    biologically_ready: Callable[[Agent], bool],
) -> MateCandidate | None:
    if not sexual_reproduction_unlocked(parent.genome, config):
        return None
    candidates = [
        candidate
        for candidate in (
            _candidate_for(parent, agent, config, biologically_ready)
            for agent in agents
        )
        if candidate is not None
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (-item.compatibility_score, item.distance, item.agent.agent_id),
    )[0]


def _candidate_for(
    parent: Agent,
    agent: Agent,
    config: ReproductionConfig,
    biologically_ready: Callable[[Agent], bool],
) -> MateCandidate | None:
    if agent.agent_id == parent.agent_id or not agent.alive:
        return None
    if (agent.reproductive_group_id or agent.lineage_id) != (
        parent.reproductive_group_id or parent.lineage_id
    ):
        return None
    if not sexual_reproduction_unlocked(agent.genome, config):
        return None
    distance = abs(agent.x - parent.x) + abs(agent.y - parent.y)
    if distance > config.sexual_partner_radius:
        return None
    if not biologically_ready(agent):
        return None
    penalty = close_kinship_penalty(parent, agent)
    return MateCandidate(
        agent=agent,
        distance=distance,
        compatibility_score=_compatibility_score(parent, agent, distance, config),
        inbreeding_penalty=penalty,
    )


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
    return round(drive * 0.45 + affinity * 0.45 + tolerance * 0.1 - distance_penalty - kinship_penalty, 6)
