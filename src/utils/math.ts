// src/utils/math.ts
export interface Vector2D {
  x: number;
  y: number;
}

export function createVector(x: number = 0, y: number = 0): Vector2D {
  return { x, y };
}

export function addVectors(v1: Vector2D, v2: Vector2D): Vector2D {
  return {
    x: v1.x + v2.x,
    y: v1.y + v2.y,
  };
}

export function subtractVectors(v1: Vector2D, v2: Vector2D): Vector2D {
  return {
    x: v1.x - v2.x,
    y: v1.y - v2.y,
  };
}

export function multiplyVector(v: Vector2D, scalar: number): Vector2D {
  return {
    x: v.x * scalar,
    y: v.y * scalar,
  };
}

export function divideVector(v: Vector2D, scalar: number): Vector2D {
  return {
    x: v.x / scalar,
    y: v.y / scalar,
  };
}

export function magnitudeVector(v: Vector2D): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function normalizeVector(v: Vector2D): Vector2D {
  const mag = magnitudeVector(v);
  if (mag === 0) {
    return { x: 0, y: 0 };
  }
  return divideVector(v, mag);
}

export function limitVector(v: Vector2D, max: number): Vector2D {
  const mag = magnitudeVector(v);
  if (mag > max) {
    return multiplyVector(normalizeVector(v), max);
  }
  return v;
}

export function distanceBetween(v1: Vector2D, v2: Vector2D): number {
  return magnitudeVector(subtractVectors(v1, v2));
}

export function randomInRange(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function randomVector(
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
): Vector2D {
  return {
    x: randomInRange(minX, maxX),
    y: randomInRange(minY, maxY),
  };
}
