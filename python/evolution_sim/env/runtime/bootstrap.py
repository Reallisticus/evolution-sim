from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StaticTopology:
    adjacent_to_water: tuple[tuple[bool, ...], ...]
    water_distance: tuple[tuple[int | None, ...], ...]
    forest_neighbor_ratio_r1: tuple[tuple[float, ...], ...]


def build_static_topology(terrain_map: list[list[str]]) -> StaticTopology:
    height = len(terrain_map)
    width = len(terrain_map[0]) if height else 0

    distances: list[list[int | None]] = [[None for _ in range(width)] for _ in range(height)]
    frontier: deque[tuple[int, int]] = deque()
    for y, row in enumerate(terrain_map):
        for x, terrain in enumerate(row):
            if terrain == "water":
                distances[y][x] = 0
                frontier.append((x, y))

    while frontier:
        x, y = frontier.popleft()
        current_distance = distances[y][x]
        if current_distance is None:
            continue
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < width and 0 <= ny < height) or distances[ny][nx] is not None:
                continue
            distances[ny][nx] = current_distance + 1
            frontier.append((nx, ny))

    adjacent_to_water: list[list[bool]] = [[False for _ in range(width)] for _ in range(height)]
    forest_neighbor_ratio_r1: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            total = 0
            forest_matches = 0
            adjacent = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    total += 1
                    if terrain_map[ny][nx] == "forest":
                        forest_matches += 1
                    if abs(dx) + abs(dy) == 1 and terrain_map[ny][nx] == "water":
                        adjacent = True
            adjacent_to_water[y][x] = adjacent
            forest_neighbor_ratio_r1[y][x] = forest_matches / total if total else 0.0

    return StaticTopology(
        adjacent_to_water=tuple(tuple(row) for row in adjacent_to_water),
        water_distance=tuple(tuple(row) for row in distances),
        forest_neighbor_ratio_r1=tuple(tuple(row) for row in forest_neighbor_ratio_r1),
    )


def terrain_neighbor_ratio(
    *,
    terrain_map: list[list[str]],
    topology: StaticTopology,
    x: int,
    y: int,
    terrain_filter: set[str],
    radius: int,
) -> float:
    if radius == 1 and terrain_filter == {"forest"}:
        return topology.forest_neighbor_ratio_r1[y][x]

    height = len(terrain_map)
    width = len(terrain_map[0]) if height else 0
    total = 0
    matches = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            total += 1
            if terrain_map[ny][nx] in terrain_filter:
                matches += 1
    return matches / total if total else 0.0
