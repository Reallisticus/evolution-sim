// src/entities/obstacle.ts
import { Vector2D, createVector } from '../utils/math';
import config from '../cfg';

export class Obstacle {
  public position: Vector2D;
  public size: number;
  public color: number = 0x555555; // Gray

  constructor(position?: Vector2D, size: number = 30) {
    this.position =
      position ||
      createVector(
        Math.random() * config.worldWidth,
        Math.random() * config.worldHeight
      );
    this.size = size;
  }
}
