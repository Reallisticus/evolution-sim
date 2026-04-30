from __future__ import annotations

from dataclasses import asdict, dataclass

SIGNAL_CONTRACT_VERSION = "foundation_signal_contract_v1"
REPRODUCTIVE_SIGNAL_FIELD = "reproductive_signal"
COMMUNICATION_SIGNAL_FIELD = "communication_signal"


@dataclass(frozen=True, slots=True)
class SignalProfile:
    profile_id: str
    intensity: float
    radius: int
    duration_ticks: int
    energy_cost: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inert_signal_profile(profile_id: str) -> SignalProfile:
    return SignalProfile(
        profile_id=profile_id,
        intensity=0.0,
        radius=0,
        duration_ticks=0,
        energy_cost=0.0,
    )


def signal_contract() -> dict[str, object]:
    return {
        "schema_version": SIGNAL_CONTRACT_VERSION,
        "substrate": "continuous_field_discrete_emission_profiles",
        "policy_semantics": "opaque",
        "fields": [REPRODUCTIVE_SIGNAL_FIELD, COMMUNICATION_SIGNAL_FIELD],
        "default_profiles": [
            inert_signal_profile(REPRODUCTIVE_SIGNAL_FIELD).to_dict(),
            inert_signal_profile(COMMUNICATION_SIGNAL_FIELD).to_dict(),
        ],
        "enabled_in_scaffold": True,
        "emission_enabled_by_default": False,
    }
