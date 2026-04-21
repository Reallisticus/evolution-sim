from __future__ import annotations

from dataclasses import dataclass
from random import Random


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_grid(values: list[list[float]]) -> list[list[float]]:
    flattened = [value for row in values for value in row]
    minimum = min(flattened)
    maximum = max(flattened)
    if maximum - minimum <= 1e-9:
        return [[0.5 for _ in row] for row in values]
    return [
        [round((value - minimum) / (maximum - minimum), 6) for value in row]
        for row in values
    ]


def _interpolated_value(
    control: list[list[float]],
    sample_x: float,
    sample_y: float,
) -> float:
    width = len(control[0])
    height = len(control)
    fx = sample_x * (width - 1)
    fy = sample_y * (height - 1)
    x0 = int(fx)
    y0 = int(fy)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = fx - x0
    ty = fy - y0

    top = control[y0][x0] * (1.0 - tx) + control[y0][x1] * tx
    bottom = control[y1][x0] * (1.0 - tx) + control[y1][x1] * tx
    return top * (1.0 - ty) + bottom * ty


def _generate_control_grid(rng: Random, width: int, height: int) -> list[list[float]]:
    return [[rng.random() for _ in range(width)] for _ in range(height)]


def _generate_fractal_field(
    width: int,
    height: int,
    seed: int,
    coarse_width: int,
    coarse_height: int,
) -> list[list[float]]:
    octave_specs = (
        (coarse_width, coarse_height, 0.58),
        (max(3, coarse_width - 2), max(3, coarse_height - 1), 0.29),
        (max(2, coarse_width - 4), max(2, coarse_height - 2), 0.13),
    )
    values = [[0.0 for _ in range(width)] for _ in range(height)]

    for octave_index, (grid_width, grid_height, weight) in enumerate(octave_specs):
        octave_rng = Random(seed + (octave_index + 1) * 7_919)
        control = _generate_control_grid(octave_rng, grid_width, grid_height)
        for y in range(height):
            sample_y = y / max(height - 1, 1)
            for x in range(width):
                sample_x = x / max(width - 1, 1)
                values[y][x] += _interpolated_value(control, sample_x, sample_y) * weight

    return _normalize_grid(values)


def _gradient_field(
    width: int,
    height: int,
    seed: int,
) -> list[list[float]]:
    rng = Random(seed)
    x_weight = rng.uniform(-1.0, 1.0)
    y_weight = rng.uniform(-1.0, 1.0)
    offset = rng.uniform(-0.25, 0.25)
    values: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        y_norm = y / max(height - 1, 1)
        for x in range(width):
            x_norm = x / max(width - 1, 1)
            row.append(x_norm * x_weight + y_norm * y_weight + offset)
        values.append(row)
    return _normalize_grid(values)


def _blend_fields(
    primary: list[list[float]],
    secondary: list[list[float]],
    primary_weight: float,
) -> list[list[float]]:
    secondary_weight = 1.0 - primary_weight
    return _normalize_grid(
        [
            [
                primary[y][x] * primary_weight + secondary[y][x] * secondary_weight
                for x in range(len(primary[0]))
            ]
            for y in range(len(primary))
        ]
    )


@dataclass(slots=True)
class EnvironmentFieldMaps:
    fertility: list[list[float]]
    moisture: list[list[float]]
    heat: list[list[float]]

    def to_serializable(self) -> dict[str, list[list[float]]]:
        return {
            "fertility": [[round(value, 4) for value in row] for row in self.fertility],
            "moisture": [[round(value, 4) for value in row] for row in self.moisture],
            "heat": [[round(value, 4) for value in row] for row in self.heat],
        }


def generate_environment_fields(
    width: int,
    height: int,
    seed: int,
    coarse_width: int,
    coarse_height: int,
) -> EnvironmentFieldMaps:
    moisture_noise = _generate_fractal_field(
        width=width,
        height=height,
        seed=seed * 101 + 11,
        coarse_width=coarse_width,
        coarse_height=coarse_height,
    )
    moisture_gradient = _gradient_field(width=width, height=height, seed=seed * 101 + 19)
    moisture = _blend_fields(moisture_noise, moisture_gradient, primary_weight=0.76)

    heat_noise = _generate_fractal_field(
        width=width,
        height=height,
        seed=seed * 101 + 23,
        coarse_width=coarse_width,
        coarse_height=coarse_height,
    )
    heat_gradient = _gradient_field(width=width, height=height, seed=seed * 101 + 29)
    inverse_moisture = [[1.0 - value for value in row] for row in moisture]
    heat = _normalize_grid(
        [
            [
                heat_noise[y][x] * 0.42
                + heat_gradient[y][x] * 0.18
                + inverse_moisture[y][x] * 0.4
                for x in range(width)
            ]
            for y in range(height)
        ]
    )

    fertility_noise = _generate_fractal_field(
        width=width,
        height=height,
        seed=seed * 101 + 31,
        coarse_width=coarse_width,
        coarse_height=coarse_height,
    )
    fertility = _normalize_grid(
        [
            [
                fertility_noise[y][x] * 0.52
                + moisture[y][x] * 0.3
                + (1.0 - heat[y][x]) * 0.18
                for x in range(width)
            ]
            for y in range(height)
        ]
    )

    return EnvironmentFieldMaps(
        fertility=[
            [_clamp01(value) for value in row]
            for row in fertility
        ],
        moisture=[
            [_clamp01(value) for value in row]
            for row in moisture
        ],
        heat=[
            [_clamp01(value) for value in row]
            for row in heat
        ],
    )
