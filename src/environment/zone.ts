// src/environment/zone.ts
import { Vector2D } from '../utils/math';
import config from '../cfg';

export enum ZoneType {
  NORMAL = 'normal',
  HARSH = 'harsh', // Higher metabolism cost
  FERTILE = 'fertile', // More food spawns
  BARREN = 'barren', // Less food spawns
}

export class Zone {
  public position: Vector2D;
  public radius: number;
  public type: ZoneType;

  constructor(position: Vector2D, radius: number, type: ZoneType) {
    this.position = position;
    this.radius = radius;
    this.type = type;
  }

  public contains(point: Vector2D): boolean {
    const dx = point.x - this.position.x;
    const dy = point.y - this.position.y;
    return dx * dx + dy * dy <= this.radius * this.radius;
  }

  public getMetabolismMultiplier(): number {
    switch (this.type) {
      case ZoneType.HARSH:
        return 1.5; // Higher energy cost
      case ZoneType.FERTILE:
        return 0.8; // Lower energy cost
      default:
        return 1.0;
    }
  }

  public getFoodSpawnMultiplier(): number {
    switch (this.type) {
      case ZoneType.FERTILE:
        return 2.0; // More food
      case ZoneType.BARREN:
        return 0.3; // Less food
      default:
        return 1.0;
    }
  }
}
