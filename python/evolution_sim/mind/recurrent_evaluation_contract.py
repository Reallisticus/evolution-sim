from __future__ import annotations


RECURRENT_EVALUATION_SCHEMA_VERSION = "mind_public_recurrent_evaluation_v4"


class RecurrentEvaluationError(ValueError):
    """Raised when a recurrent evaluation contract fails closed."""


__all__ = [
    "RECURRENT_EVALUATION_SCHEMA_VERSION",
    "RecurrentEvaluationError",
]
