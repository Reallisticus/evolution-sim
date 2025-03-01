// src/environment/cycle.ts

export enum TimeOfDay {
  DAY,
  NIGHT,
}

export class EnvironmentCycle {
  private dayLength: number = 500; // Ticks
  private totalTicks: number = 0;

  public update(): void {
    this.totalTicks++;
  }

  public getTimeOfDay(): TimeOfDay {
    const cyclePosition = this.totalTicks % (this.dayLength * 2);
    return cyclePosition < this.dayLength ? TimeOfDay.DAY : TimeOfDay.NIGHT;
  }

  public getDayNightRatio(): number {
    // Returns 0.0 (night) to 1.0 (day)
    const cyclePosition = this.totalTicks % (this.dayLength * 2);

    if (cyclePosition < this.dayLength) {
      // Day: 0 -> dayLength maps to 0.0 -> 1.0
      return cyclePosition / this.dayLength;
    } else {
      // Night: dayLength -> dayLength*2 maps to 1.0 -> 0.0
      return 1.0 - (cyclePosition - this.dayLength) / this.dayLength;
    }
  }

  public getVisibilityMultiplier(): number {
    // How well agents can see
    const time = this.getTimeOfDay();
    return time === TimeOfDay.DAY ? 1.0 : 0.3;
  }

  public getMovementMultiplier(): number {
    // How efficiently agents can move
    const time = this.getTimeOfDay();
    return time === TimeOfDay.DAY ? 1.0 : 0.7;
  }
  public setTimeOfDay(timeOfDay: TimeOfDay): void {
    // Calculate appropriate totalTicks value to match the desired time of day
    if (timeOfDay === TimeOfDay.DAY) {
      // Set to middle of day
      this.totalTicks = Math.floor(this.dayLength / 2);
    } else {
      // Set to middle of night
      this.totalTicks = Math.floor(this.dayLength * 1.5);
    }
  }

  // Add this method to set totalTicks directly (if needed)
  public setTotalTicks(ticks: number): void {
    this.totalTicks = ticks;
  }
}
