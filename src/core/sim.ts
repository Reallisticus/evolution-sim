// src/core/sim.ts
import config from '../cfg';
import { TimeController } from './time';
import { Agent } from '../entities/agent';
import { Food, FoodType } from '../entities/food';
import { Obstacle } from '../entities/obstacles';
import { Zone, ZoneType } from '../environment/zone';
import { EnvironmentCycle, TimeOfDay } from '../environment/cycle';
import { createVector, Vector2D } from '../utils/math';
import { SpeciesManager } from '../evolution/species';
import { LineageTracker } from '../evolution/lineage';

export class Simulation {
  private timeController: TimeController;
  private generation: number = 1;
  private tickCount: number = 0;
  private agents: Agent[] = [];
  private foods: Food[] = [];
  private obstacles: Obstacle[] = [];
  private zones: Zone[] = [];
  private environmentCycle: EnvironmentCycle = new EnvironmentCycle();
  private speciesManager: SpeciesManager = new SpeciesManager();
  private lineageTracker: LineageTracker = new LineageTracker();
  private deadAgents: Agent[] = [];

  constructor() {
    this.timeController = new TimeController();

    // Set up simulation loop
    this.timeController.onTick(this.update.bind(this));
  }

  public start(): void {
    console.log('Starting simulation...');
    this.timeController.start();
  }

  public pause(): void {
    console.log('Pausing simulation...');
    this.timeController.pause();
  }

  public reset(): void {
    console.log('Resetting simulation...');

    // Clean up existing agents
    for (const agent of this.agents) {
      agent.dispose();
    }

    // Reset collections
    this.agents = [];
    this.foods = [];
    this.obstacles = [];
    this.zones = [];
    this.deadAgents = [];
    this.generation = 1;
    this.tickCount = 0;

    // Reset tracking systems
    this.speciesManager = new SpeciesManager();
    this.lineageTracker = new LineageTracker();

    // Initialize environment
    this.initializeEnvironment();

    // Initialize agents
    for (let i = 0; i < config.initialAgentCount; i++) {
      const position = createVector(
        Math.random() * config.worldWidth,
        Math.random() * config.worldHeight
      );

      // Create agent with reference to this simulation
      const agent = new Agent(position, undefined, this);

      // Assign ID from lineage tracker
      agent.id = this.lineageTracker.registerBirth([], 0);

      // Assign species
      agent.species = this.speciesManager.assignSpecies(agent.genome, agent.id);

      this.agents.push(agent);
    }

    // Initialize food
    this.spawnFood(config.initialFoodCount);
  }

  private initializeEnvironment(): void {
    // Create zones
    this.zones.push(
      new Zone(
        createVector(config.worldWidth * 0.25, config.worldHeight * 0.25),
        config.worldWidth * 0.2,
        ZoneType.FERTILE
      ),
      new Zone(
        createVector(config.worldWidth * 0.75, config.worldHeight * 0.75),
        config.worldWidth * 0.2,
        ZoneType.HARSH
      ),
      new Zone(
        createVector(config.worldWidth * 0.75, config.worldHeight * 0.25),
        config.worldWidth * 0.15,
        ZoneType.BARREN
      )
    );

    // Create obstacles
    const obstacleCount = Math.floor(
      Math.sqrt(config.worldWidth * config.worldHeight) / 30
    );
    for (let i = 0; i < obstacleCount; i++) {
      const size = Math.random() * 30 + 20;
      const position = createVector(
        Math.random() * config.worldWidth,
        Math.random() * config.worldHeight
      );
      this.obstacles.push(new Obstacle(position, size));
    }
  }

  public onRender(callback: () => void): void {
    this.timeController.onRender(callback);
  }

  public getAgents(): Agent[] {
    return this.agents;
  }

  public getFoods(): Food[] {
    return this.foods;
  }

  public getObstacles(): Obstacle[] {
    return this.obstacles;
  }

  public getZones(): Zone[] {
    return this.zones;
  }

  public getEnvironmentCycle(): EnvironmentCycle {
    return this.environmentCycle;
  }

  public getGeneration(): number {
    return this.generation;
  }

  public getTickCount(): number {
    return this.tickCount;
  }

  public setGeneration(generation: number): void {
    this.generation = generation;
  }

  public setTickCount(tick: number): void {
    this.tickCount = tick;
  }

  public clearZones(): void {
    this.zones = [];
  }

  public setZones(zones: Zone[]): void {
    this.zones = zones;
  }

  public clearObstacles(): void {
    this.obstacles = [];
  }

  public setObstacles(obstacles: Obstacle[]): void {
    this.obstacles = obstacles;
  }

  public clearFood(): void {
    this.foods = [];
  }

  public setFood(foods: Food[]): void {
    this.foods = foods;
  }

  public clearAgents(): void {
    // Dispose all agents
    for (const agent of this.agents) {
      agent.dispose();
    }
    this.agents = [];
  }

  public setAgents(agents: Agent[]): void {
    this.agents = agents;
  }

  public getSpeciesManager(): SpeciesManager {
    return this.speciesManager;
  }

  public getLineageTracker(): LineageTracker {
    return this.lineageTracker;
  }

  private update(): void {
    this.tickCount++;

    // Update lineage tracker
    this.lineageTracker.updateTick(this.tickCount);

    // Update environment cycle
    this.environmentCycle.update();
    const timeOfDay = this.environmentCycle.getTimeOfDay();
    const movementMultiplier = this.environmentCycle.getMovementMultiplier();

    // Only process a subset of agents each frame if there are many
    const agentBatchSize = 10; // Process 10 agents per frame when there are many

    if (this.agents.length <= 20) {
      // Few agents - process all at once
      for (const agent of this.agents) {
        // Apply environment effects based on agent position
        this.applyEnvironmentEffects(agent);

        // Update agent
        agent.update(this.foods, this.obstacles, movementMultiplier);
      }
    } else {
      // Many agents - process in batches
      const startIdx = (this.tickCount * agentBatchSize) % this.agents.length;
      const endIdx = Math.min(startIdx + agentBatchSize, this.agents.length);

      for (let i = 0; i < this.agents.length; i++) {
        const agent = this.agents[i];

        // Detailed update (NN + full collision) for current batch
        if (i >= startIdx && i < endIdx) {
          // Apply environment effects
          this.applyEnvironmentEffects(agent);

          // Full update
          agent.update(this.foods, this.obstacles, movementMultiplier);
        } else {
          // Simple update (physics only) for other agents
          agent.updatePhysics(movementMultiplier);
        }
      }
    }

    // Update all food
    for (const food of this.foods) {
      food.update();
    }

    // Respawn consumed food
    const consumedCount = this.foods.filter((food) => food.isConsumed).length;
    if (consumedCount > 0) {
      this.spawnFood(consumedCount);
      this.foods = this.foods.filter((food) => !food.isConsumed);
    }

    // Update fitness tracking for all agents
    for (const agent of this.agents) {
      this.lineageTracker.updateFitness(agent.id, agent.fitness);
      if (agent.species) {
        agent.species.updateFitness(agent.fitness);
      }
    }

    // Process dead agents
    this.deadAgents = [];
    this.agents = this.agents.filter((agent) => {
      if (agent.dead) {
        this.deadAgents.push(agent);
        // Register death in lineage tracker
        this.lineageTracker.registerDeath(agent.id, agent.fitness);
        return false;
      }
      return true;
    });

    // Check for end of generation
    this.checkGenerationEnd();
  }

  private applyEnvironmentEffects(agent: Agent): void {
    // Find zone that contains the agent
    for (const zone of this.zones) {
      if (zone.contains(agent.position)) {
        // Apply zone effects
        agent.metabolismMultiplier = zone.getMetabolismMultiplier();
        return; // Only apply one zone's effects
      }
    }

    // Default if not in any zone
    agent.metabolismMultiplier = 1.0;
  }

  private checkGenerationEnd(): void {
    // End generation if no agents left or max time reached
    if (this.agents.length === 0 || this.tickCount >= 1000) {
      this.startNewGeneration();
    }
  }

  private startNewGeneration(): void {
    this.generation++;
    this.tickCount = 0;
    console.log(`Starting generation ${this.generation}`);

    // Advance generation in tracking systems
    this.speciesManager.advanceGeneration();
    this.lineageTracker.advanceGeneration();

    // If no agents survived, reset with new random agents
    if (this.agents.length === 0) {
      for (let i = 0; i < config.initialAgentCount; i++) {
        const position = createVector(
          Math.random() * config.worldWidth,
          Math.random() * config.worldHeight
        );

        // Create agent with simulation reference
        const agent = new Agent(position, undefined, this);

        // Register birth
        agent.id = this.lineageTracker.registerBirth([], 0);

        // Assign species
        agent.species = this.speciesManager.assignSpecies(
          agent.genome,
          agent.id
        );

        this.agents.push(agent);
      }
    } else {
      // Get sorted agents by fitness
      const sortedAgents = [...this.agents].sort(
        (a, b) => b.fitness - a.fitness
      );

      // Keep only the top performers
      const survivors = sortedAgents.slice(
        0,
        Math.ceil(sortedAgents.length / 4)
      );

      // Clean up non-survivors
      for (const agent of this.agents) {
        if (!survivors.includes(agent)) {
          agent.dispose();
        }
      }

      // Create new generation
      const newAgents: Agent[] = [];

      // Keep survivors
      for (const survivor of survivors) {
        newAgents.push(survivor);
      }

      // Fill up to initial count with offspring
      while (newAgents.length < config.initialAgentCount) {
        // Select random parent from survivors
        const parentIndex = Math.floor(Math.random() * survivors.length);
        const parent = survivors[parentIndex];

        // Have a chance to select a second parent for sexual reproduction
        let partner = null;
        if (Math.random() < 0.7 && survivors.length > 1) {
          let partnerIndex;
          do {
            partnerIndex = Math.floor(Math.random() * survivors.length);
          } while (partnerIndex === parentIndex);

          partner = survivors[partnerIndex];
        }

        // Create offspring
        const child = parent.reproduce(partner);

        // Register birth
        child.id = this.lineageTracker.registerBirth(
          partner ? [parent.id, partner.id] : [parent.id],
          0 // Initial species ID, will be set next
        );

        // Assign species
        child.species = this.speciesManager.assignSpecies(
          child.genome,
          child.id
        );

        newAgents.push(child);
      }

      this.agents = newAgents;
    }

    // Reset food
    this.foods = [];
    this.spawnFood(config.initialFoodCount);
  }

  public spawnFood(count: number): void {
    const baseCount = count;

    // Distribute food based on zones
    for (let i = 0; i < baseCount; i++) {
      // Random position
      let position = createVector(
        Math.random() * config.worldWidth,
        Math.random() * config.worldHeight
      );

      // Determine food type based on probabilities
      let foodType = FoodType.BASIC;
      const roll = Math.random();

      if (roll < 0.1) {
        foodType = FoodType.SUPER; // 10% chance of super food
      } else if (roll < 0.2) {
        foodType = FoodType.POISON; // 10% chance of poison
      }

      this.foods.push(new Food(position, foodType));
    }
  }
}
