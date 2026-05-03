from __future__ import annotations

from random import Random
from typing import Sequence


def deterministic_seed_split(
    seeds: Sequence[int],
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
) -> dict[str, list[int]]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    deduped = list(dict.fromkeys(int(seed) for seed in seeds))
    shuffled = list(deduped)
    Random(split_seed).shuffle(shuffled)
    validation_count = int(round(len(shuffled) * validation_fraction))
    if validation_fraction > 0 and validation_count == 0 and len(shuffled) > 1:
        validation_count = 1
    validation = sorted(shuffled[:validation_count])
    validation_set = set(validation)
    train = sorted(seed for seed in deduped if seed not in validation_set)
    return {"train": train, "validation": validation}
