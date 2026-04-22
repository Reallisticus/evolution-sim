from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    TICK_STARTED = "tick_started"
    AGENT_MOVED = "agent_moved"
    AGENT_ATE = "agent_ate"
    CARCASS_DEPOSITED = "carcass_deposited"
    AGENT_DRANK = "agent_drank"
    AGENT_ATTACKED = "agent_attacked"
    AGENT_DAMAGED = "agent_damaged"
    AGENT_HEALED = "agent_healed"
    AGENT_REPRODUCED = "agent_reproduced"
    AGENT_DIED = "agent_died"
    TICK_COMPLETED = "tick_completed"
    RUN_COMPLETED = "run_completed"


@dataclass(slots=True)
class Event:
    tick: int
    type: EventType
    agent_id: int | None = None
    data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "type": self.type.value,
            "agent_id": self.agent_id,
            "data": self.data,
        }
