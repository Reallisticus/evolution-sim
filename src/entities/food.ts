// src/entities/food.ts - Enhanced food types
import { Vector2D, createVector } from '../utils/math';
import config from '../cfg';

export enum FoodType {
  BASIC = 'basic',
  SUPER = 'super',
  POISON = 'poison',
}

export class Food {
  public position: Vector2D;
  public energy: number;
  public size: number;
  public isConsumed: boolean = false;
  public type: FoodType;
  public color: number;

  constructor(position?: Vector2D, type: FoodType = FoodType.BASIC) {
    this.position =
      position ||
      createVector(
        Math.random() * config.worldWidth,
        Math.random() * config.worldHeight
      );

    this.type = type;

    // Set properties based on food type
    switch (this.type) {
      case FoodType.BASIC:
        this.energy = 20;
        this.size = 5;
        this.color = 0x88aaff; // Blue
        break;
      case FoodType.SUPER:
        this.energy = 50;
        this.size = 8;
        this.color = 0xffaa00; // Orange
        break;
      case FoodType.POISON:
        this.energy = -30;
        this.size = 4;
        this.color = 0xaa00aa; // Purple
        break;
    }
  }

  public update(): void {
    // Food doesn't do anything for now
  }
}
