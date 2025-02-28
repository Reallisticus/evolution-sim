// src/utils/spatial-grid.ts
import { Vector2D } from './math';

export class SpatialGrid<T extends { position: Vector2D }> {
  private cells: Map<string, T[]> = new Map();
  private cellSize: number;

  constructor(cellSize: number = 50) {
    this.cellSize = cellSize;
  }

  private getCellKey(x: number, y: number): string {
    const cellX = Math.floor(x / this.cellSize);
    const cellY = Math.floor(y / this.cellSize);
    return `${cellX},${cellY}`;
  }

  public insert(item: T): void {
    const key = this.getCellKey(item.position.x, item.position.y);
    if (!this.cells.has(key)) {
      this.cells.set(key, []);
    }
    this.cells.get(key)!.push(item);
  }

  public clear(): void {
    this.cells.clear();
  }

  public getNearby(position: Vector2D, radius: number): T[] {
    const result: T[] = [];

    // Get cells that might contain items within radius
    const minCellX = Math.floor((position.x - radius) / this.cellSize);
    const maxCellX = Math.floor((position.x + radius) / this.cellSize);
    const minCellY = Math.floor((position.y - radius) / this.cellSize);
    const maxCellY = Math.floor((position.y + radius) / this.cellSize);

    // Check each potential cell
    for (let cellX = minCellX; cellX <= maxCellX; cellX++) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY++) {
        const key = `${cellX},${cellY}`;
        const cell = this.cells.get(key);

        if (cell) {
          result.push(...cell);
        }
      }
    }

    return result;
  }
}
