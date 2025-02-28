// src/core/time.ts
import config from '../cfg';

export class TimeController {
  private lastTime: number = 0;
  private accumulator: number = 0;
  private timeStep: number = 1000 / config.tickRate;
  private running: boolean = false;
  private tickCallbacks: Array<() => void> = [];
  private renderCallbacks: Array<() => void> = [];

  constructor() {}

  public start(): void {
    if (!this.running) {
      this.running = true;
      this.lastTime = performance.now();
      requestAnimationFrame(this.loop.bind(this));
    }
  }

  public pause(): void {
    this.running = false;
  }

  public onTick(callback: () => void): void {
    this.tickCallbacks.push(callback);
  }

  public onRender(callback: () => void): void {
    this.renderCallbacks.push(callback);
  }

  private loop(currentTime: number): void {
    if (!this.running) return;

    const deltaTime = currentTime - this.lastTime;
    this.lastTime = currentTime;

    // Add delta time to accumulator
    this.accumulator += deltaTime * config.simulationSpeed;

    // Process fixed time steps
    while (this.accumulator >= this.timeStep) {
      // Execute tick callbacks
      for (const callback of this.tickCallbacks) {
        callback();
      }

      this.accumulator -= this.timeStep;
    }

    // Execute render callbacks
    for (const callback of this.renderCallbacks) {
      callback();
    }

    requestAnimationFrame(this.loop.bind(this));
  }
}
