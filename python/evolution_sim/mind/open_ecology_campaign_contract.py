"""Torch-free shared errors for persistent open-ecology campaign tooling."""

from __future__ import annotations


class OpenEcologyCampaignCoordinatorError(RuntimeError):
    """A campaign scheduling, resume, or frontier invariant failed closed."""


__all__ = ["OpenEcologyCampaignCoordinatorError"]
