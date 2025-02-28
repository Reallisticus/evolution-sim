// src/visualization/renderer.ts
import { Simulation } from '../core/sim'; // Adjust path if needed
import { Agent } from '../entities/agent';
import { Food } from '../entities/food';
import config from '../cfg'; // Adjust path if needed
import { Obstacle } from '../entities/obstacles';
import { Zone, ZoneType } from '../environment/zone';
import { EnvironmentCycle, TimeOfDay } from '../environment/cycle';
import { FoodType } from '../entities/food';

export class Renderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private simulation: Simulation;
  private lastPositions: Map<Agent, { x: number; y: number }> = new Map();
  private resizeObserver: ResizeObserver;
  private lastFrameTime: number = 0;
  private frameRate: number = 0;

  constructor(simulation: Simulation, container: HTMLElement) {
    this.simulation = simulation;

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.setCanvasSize();
    this.canvas.style.background =
      'linear-gradient(to bottom, #0f0f1e, #1f1f3e)';
    container.appendChild(this.canvas);

    // Get context
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    this.ctx = ctx;

    // Make responsive
    this.resizeObserver = new ResizeObserver(() => this.setCanvasSize());
    this.resizeObserver.observe(container);
    window.addEventListener('resize', () => this.setCanvasSize());

    // Set up render callback
    this.simulation.onRender(this.render.bind(this));
  }

  private setCanvasSize(): void {
    // Set to full window size
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;

    // Update simulation world bounds
    config.worldWidth = window.innerWidth;
    config.worldHeight = window.innerHeight;
  }

  private render(): void {
    // Get environment data
    const agents = this.simulation.getAgents();
    const foods = this.simulation.getFoods();
    const obstacles = this.simulation.getObstacles();
    const zones = this.simulation.getZones();
    const environmentCycle = this.simulation.getEnvironmentCycle();

    // Calculate frame rate
    const now = performance.now();
    this.frameRate = 1000 / (now - this.lastFrameTime);
    this.lastFrameTime = now;

    const dayNightRatio = environmentCycle.getDayNightRatio();

    // Clear canvas with gradient
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const skyColorTop = this.interpolateColor(
      [15, 15, 30], // Night color
      [100, 150, 255], // Day color
      dayNightRatio
    );

    const skyColorBottom = this.interpolateColor(
      [30, 30, 60], // Night color
      [150, 200, 255], // Day color
      dayNightRatio
    );

    const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
    gradient.addColorStop(0, `rgb(${skyColorTop.join(',')})`);
    gradient.addColorStop(1, `rgb(${skyColorBottom.join(',')})`);
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    if (agents.length < 100) {
      this.drawGrid();
    }

    // Draw zones
    this.drawZones(zones);

    // Set performance threshold
    const highDetailThreshold = 30;
    const useHighDetail = agents.length <= highDetailThreshold;

    // Draw obstacles
    this.drawObstacles(obstacles);

    // Draw food (optimized)
    this.drawFood(foods, useHighDetail);

    // Draw agents with conditional detail level
    this.drawAgents(agents, useHighDetail);

    // Draw stats with nice styling
    this.drawStats(agents.length, foods.length, environmentCycle);
  }

  private interpolateColor(
    color1: number[],
    color2: number[],
    ratio: number
  ): number[] {
    return [
      Math.round(color1[0] + (color2[0] - color1[0]) * ratio),
      Math.round(color1[1] + (color2[1] - color1[1]) * ratio),
      Math.round(color1[2] + (color2[2] - color1[2]) * ratio),
    ];
  }

  // Add method to draw zones
  private drawZones(zones: Zone[]): void {
    for (const zone of zones) {
      // Set style based on zone type
      let fillColor;
      switch (zone.type) {
        case ZoneType.FERTILE:
          fillColor = 'rgba(0, 200, 100, 0.1)';
          break;
        case ZoneType.HARSH:
          fillColor = 'rgba(200, 50, 50, 0.1)';
          break;
        case ZoneType.BARREN:
          fillColor = 'rgba(150, 150, 150, 0.1)';
          break;
        default:
          fillColor = 'rgba(100, 100, 200, 0.1)';
      }

      // Draw zone
      this.ctx.fillStyle = fillColor;
      this.ctx.beginPath();
      this.ctx.arc(
        zone.position.x,
        zone.position.y,
        zone.radius,
        0,
        Math.PI * 2
      );
      this.ctx.fill();

      // Draw zone border
      this.ctx.strokeStyle = fillColor.replace('0.1', '0.3');
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Add zone label
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      this.ctx.font = '14px "Segoe UI", Arial, sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(
        zone.type.toUpperCase(),
        zone.position.x,
        zone.position.y
      );
      this.ctx.textAlign = 'left'; // Reset
    }
  }

  // Add method to draw obstacles
  private drawObstacles(obstacles: Obstacle[]): void {
    for (const obstacle of obstacles) {
      // Draw obstacle
      const gradient = this.ctx.createRadialGradient(
        obstacle.position.x,
        obstacle.position.y,
        0,
        obstacle.position.x,
        obstacle.position.y,
        obstacle.size
      );
      gradient.addColorStop(0, 'rgba(100, 100, 100, 0.9)');
      gradient.addColorStop(1, 'rgba(50, 50, 50, 0.7)');

      this.ctx.fillStyle = gradient;
      this.ctx.beginPath();
      this.ctx.arc(
        obstacle.position.x,
        obstacle.position.y,
        obstacle.size,
        0,
        Math.PI * 2
      );
      this.ctx.fill();

      // Draw obstacle border
      this.ctx.strokeStyle = 'rgba(80, 80, 80, 0.8)';
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }
  }

  private drawFood(foods: Food[], highDetail: boolean): void {
    // Skip decorative effects if many agents to improve performance
    if (highDetail) {
      this.ctx.shadowBlur = 10;
    } else {
      this.ctx.shadowBlur = 0;
    }

    // Group foods by type for batch rendering
    const foodsByType = {
      [FoodType.BASIC]: [] as Food[],
      [FoodType.SUPER]: [] as Food[],
      [FoodType.POISON]: [] as Food[],
    };

    foods.forEach((food) => foodsByType[food.type].push(food));

    // Draw basic food
    this.ctx.shadowColor = '#4488ff';
    this.ctx.fillStyle = '#88aaff';
    for (const food of foodsByType[FoodType.BASIC]) {
      this.ctx.beginPath();
      this.ctx.arc(food.position.x, food.position.y, food.size, 0, Math.PI * 2);
      this.ctx.fill();

      if (highDetail) {
        this.ctx.fillStyle = '#aaddff';
        this.ctx.beginPath();
        this.ctx.arc(
          food.position.x - food.size / 3,
          food.position.y - food.size / 3,
          food.size / 3,
          0,
          Math.PI * 2
        );
        this.ctx.fill();
        this.ctx.fillStyle = '#88aaff';
      }
    }

    // Draw super food
    this.ctx.shadowColor = '#ffaa00';
    this.ctx.fillStyle = '#ffcc44';
    for (const food of foodsByType[FoodType.SUPER]) {
      this.ctx.beginPath();
      this.ctx.arc(food.position.x, food.position.y, food.size, 0, Math.PI * 2);
      this.ctx.fill();

      if (highDetail) {
        this.ctx.fillStyle = '#ffee88';
        this.ctx.beginPath();
        this.ctx.arc(
          food.position.x - food.size / 3,
          food.position.y - food.size / 3,
          food.size / 3,
          0,
          Math.PI * 2
        );
        this.ctx.fill();
        this.ctx.fillStyle = '#ffcc44';
      }
    }

    // Draw poison food
    this.ctx.shadowColor = '#aa00aa';
    this.ctx.fillStyle = '#cc55cc';
    for (const food of foodsByType[FoodType.POISON]) {
      this.ctx.beginPath();
      this.ctx.arc(food.position.x, food.position.y, food.size, 0, Math.PI * 2);
      this.ctx.fill();

      if (highDetail) {
        // Add skull or X mark to poison
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();

        // Draw X
        const offset = food.size * 0.4;
        this.ctx.moveTo(food.position.x - offset, food.position.y - offset);
        this.ctx.lineTo(food.position.x + offset, food.position.y + offset);
        this.ctx.moveTo(food.position.x + offset, food.position.y - offset);
        this.ctx.lineTo(food.position.x - offset, food.position.y + offset);
        this.ctx.stroke();
      }
    }

    // Reset shadow
    this.ctx.shadowBlur = 0;
  }

  private drawAgents(agents: Agent[], highDetail: boolean): void {
    // Skip non-critical agents if framerate drops significantly
    const frameRateThreshold = 30;
    const skipFactor =
      this.frameRate < frameRateThreshold && agents.length > 50
        ? Math.floor(agents.length / 30)
        : 1;

    // Process agents
    for (let i = 0; i < agents.length; i++) {
      // Skip some agents when framerate is low
      if (skipFactor > 1 && i % skipFactor !== 0 && i !== 0) continue;

      const agent = agents[i];

      // Get or initialize last position for interpolation
      let lastPos = this.lastPositions.get(agent);
      if (!lastPos) {
        lastPos = { x: agent.position.x, y: agent.position.y };
        this.lastPositions.set(agent, lastPos);
      }

      // Interpolate position for smoother movement (LERP)
      // Faster interpolation when there are many agents
      const lerpFactor = highDetail ? 0.1 : 0.2;
      const interpX = lastPos.x + (agent.position.x - lastPos.x) * lerpFactor;
      const interpY = lastPos.y + (agent.position.y - lastPos.y) * lerpFactor;

      // Update last position for next frame
      lastPos.x = interpX;
      lastPos.y = interpY;

      // Calculate energy percentage for color
      const energyPercent = agent.energy / config.agentMaxEnergy;

      if (highDetail) {
        // HIGH DETAIL RENDERING

        // Set glow based on energy
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = `rgba(0, ${Math.floor(
          255 * energyPercent
        )}, 0, 0.8)`;

        // Draw agent body with gradient
        const gradient = this.ctx.createRadialGradient(
          interpX,
          interpY,
          0,
          interpX,
          interpY,
          agent.size
        );
        gradient.addColorStop(0, `rgba(100, 255, 100, ${energyPercent + 0.2})`);
        gradient.addColorStop(1, `rgba(0, 180, 0, ${energyPercent * 0.8})`);

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(interpX, interpY, agent.size, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw direction indicator
        if (agent.velocity.x !== 0 || agent.velocity.y !== 0) {
          this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
          this.ctx.lineWidth = 2;
          this.ctx.beginPath();
          this.ctx.moveTo(interpX, interpY);
          this.ctx.lineTo(
            interpX + agent.velocity.x * 8,
            interpY + agent.velocity.y * 8
          );
          this.ctx.stroke();

          // Draw "eyes" in direction of movement
          const angle = Math.atan2(agent.velocity.y, agent.velocity.x);
          const eyeOffset = agent.size * 0.6;
          const eyeSize = agent.size * 0.3;

          this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
          this.ctx.beginPath();
          this.ctx.arc(
            interpX + Math.cos(angle) * eyeOffset,
            interpY + Math.sin(angle) * eyeOffset,
            eyeSize,
            0,
            Math.PI * 2
          );
          this.ctx.fill();
        }
      } else {
        // LOW DETAIL RENDERING - Simpler and faster

        // No shadows for better performance
        this.ctx.shadowBlur = 0;

        // Simple color based on energy
        this.ctx.fillStyle = `rgba(0, ${
          Math.floor(200 * energyPercent) + 55
        }, 0, 0.8)`;
        this.ctx.beginPath();
        this.ctx.arc(interpX, interpY, agent.size, 0, Math.PI * 2);
        this.ctx.fill();

        // Simplified direction indicator
        if (agent.velocity.x !== 0 || agent.velocity.y !== 0) {
          this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
          this.ctx.lineWidth = 1;
          this.ctx.beginPath();
          this.ctx.moveTo(interpX, interpY);
          this.ctx.lineTo(
            interpX + agent.velocity.x * 5,
            interpY + agent.velocity.y * 5
          );
          this.ctx.stroke();
        }
      }

      // Clean up deleted agents
      if (agent.dead) {
        this.lastPositions.delete(agent);
      }
    }

    // Reset shadow
    this.ctx.shadowBlur = 0;
  }

  private drawGrid(): void {
    const gridSize = 50;
    const gridColor = 'rgba(80, 80, 160, 0.1)';

    this.ctx.strokeStyle = gridColor;
    this.ctx.lineWidth = 1;

    // Draw vertical lines
    for (let x = 0; x < this.canvas.width; x += gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.canvas.height);
      this.ctx.stroke();
    }

    // Draw horizontal lines
    for (let y = 0; y < this.canvas.height; y += gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.canvas.width, y);
      this.ctx.stroke();
    }
  }

  private drawStats(
    agentCount: number,
    foodCount: number,
    environmentCycle: EnvironmentCycle
  ): void {
    const timeOfDay = environmentCycle.getTimeOfDay();

    // Create semi-transparent panel
    this.ctx.fillStyle = 'rgba(0, 0, 30, 0.7)';
    this.ctx.roundRect(10, 10, 200, 180, 5);
    this.ctx.fill();

    // Draw border
    this.ctx.strokeStyle = 'rgba(100, 150, 255, 0.5)';
    this.ctx.lineWidth = 2;
    this.ctx.roundRect(10, 10, 200, 180, 5);
    this.ctx.stroke();

    // Draw stats text with text shadow
    this.ctx.font = '16px "Segoe UI", Arial, sans-serif';
    this.ctx.textBaseline = 'top';

    const textY = [20, 45, 70, 95, 120, 145];
    const labels = [
      `Generation: ${this.simulation.getGeneration()}`,
      `Tick: ${this.simulation.getTickCount()}`,
      `Agents: ${agentCount}`,
      `Food: ${foodCount}`,
      `Time: ${timeOfDay === TimeOfDay.DAY ? 'Day' : 'Night'}`,
      `FPS: ${Math.round(this.frameRate)}`,
    ];

    // Draw text shadow
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    labels.forEach((text, i) => {
      this.ctx.fillText(text, 21, textY[i] + 1);
    });

    // Draw text
    this.ctx.fillStyle = 'rgba(180, 220, 255, 1.0)';
    labels.forEach((text, i) => {
      this.ctx.fillText(text, 20, textY[i]);
    });
  }
}
