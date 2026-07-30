"""Torch-free scalar contracts shared by Phase-A authority surfaces."""

from __future__ import annotations

import re


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OpenEcologyPhaseAError(ValueError):
    """Raised when Phase A differs from its sealed causal-ablation contract."""


def require_lowercase_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OpenEcologyPhaseAError(f"{field} must be lowercase SHA-256")
    return value


__all__ = [
    "OpenEcologyPhaseAError",
    "require_lowercase_sha256",
]
