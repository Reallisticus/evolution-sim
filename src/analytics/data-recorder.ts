// src/analytics/data-recorder.ts - Updated with history store
import { Simulation } from '../core/sim';
import { HistoryStore } from './history-store';
import { SimulationStats, AgentStats } from './data-types';
import { FoodType } from '../entities/food';
import { ZoneType } from '../environment/zone';
import { DeathCause } from '../entities/agent'; // We'll add this to agent.ts

export class DataRecorder {
  private simulation: Simulation;
  private recordInterval: number = 500; // ms
  private intervalId?: number;
  private historyStore: HistoryStore;
  private agentTracker: Map<number, AgentStats> = new Map();
  private currentGenStats: SimulationStats | null = null;

  constructor(simulation: Simulation) {
    this.simulation = simulation;
    this.historyStore = new HistoryStore();

    // Initialize agent tracker
    this.resetAgentTracker();
  }

  private resetAgentTracker(): void {
    this.agentTracker.clear();

    // Initialize tracking for all current agents
    const agents = this.simulation.getAgents();
    for (const agent of agents) {
      this.initializeAgentTracking(agent);
    }
  }

  private initializeAgentTracking(agent: any): void {
    if (this.agentTracker.has(agent.id)) return;

    this.agentTracker.set(agent.id, {
      id: agent.id,
      generation: this.simulation.getGeneration(),
      parentIds: agent.parentIds,
      age: 0,
      energy: agent.energy,
      fitness: 0,
      speciesId: agent.species?.id || 0,

      distanceTraveled: 0,
      avgSpeed: 0,

      foodConsumed: {
        [FoodType.BASIC]: 0,
        [FoodType.SUPER]: 0,
        [FoodType.POISON]: 0,
      },

      timeInZones: {
        [ZoneType.NORMAL]: 0,
        [ZoneType.HARSH]: 0,
        [ZoneType.FERTILE]: 0,
        [ZoneType.BARREN]: 0,
        outside: 0,
      },

      collisions: 0,

      avgResponsesToFood: {
        [FoodType.BASIC]: [0, 0, 0],
        [FoodType.SUPER]: [0, 0, 0],
        [FoodType.POISON]: [0, 0, 0],
      },

      avgResponsesToObstacles: [0, 0, 0],

      mutationRate: agent.genome.mutationRate,
      weightStats: {
        mean: 0,
        stdDev: 0,
      },
    });
  }

  public start(): void {
    // Clear any existing interval
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }

    // Start recording at regular intervals
    this.intervalId = window.setInterval(() => {
      this.recordData();
    }, this.recordInterval);

    // Record immediately
    this.recordData();

    // Add event listeners for agent lifecycle events
    this.simulation.on('agentCreated', this.handleAgentCreated.bind(this));
    this.simulation.on('agentDied', this.handleAgentDied.bind(this));
    this.simulation.on('generationEnd', this.handleGenerationEnd.bind(this));
    this.simulation.on('agentAteFood', this.handleAgentAteFood.bind(this));
    this.simulation.on('agentCollision', this.handleAgentCollision.bind(this));

    // Record initial generation stats
    this.recordGenerationStats();
  }

  public stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }

    // Save final state
    this.historyStore.saveToLocalStorage();
  }

  private handleAgentCreated(agent: any): void {
    this.initializeAgentTracking(agent);
  }

  private handleAgentDied(agent: any, cause: DeathCause): void {
    // Record final stats before removing
    const agentStats = this.agentTracker.get(agent.id);
    if (agentStats) {
      // Update current generation death counts
      if (this.currentGenStats) {
        this.currentGenStats.deathCauses[cause]++;
      }

      // Remove from tracker
      this.agentTracker.delete(agent.id);
    }
  }

  private handleGenerationEnd(): void {
    // Record final generation stats
    this.recordGenerationStats();

    // Reset agent tracking for new generation
    this.resetAgentTracker();
  }

  private handleAgentAteFood(agent: any, foodType: FoodType): void {
    const agentStats = this.agentTracker.get(agent.id);
    if (agentStats) {
      agentStats.foodConsumed[foodType]++;
    }
  }

  private handleAgentCollision(agent: any, isObstacle: boolean): void {
    const agentStats = this.agentTracker.get(agent.id);
    if (agentStats) {
      agentStats.collisions++;
    }
  }

  private recordData(): void {
    // Record full simulation data
    this.recordSimulationData();

    // Update agent tracking
    this.updateAgentTracking();

    // Store in localStorage
    localStorage.setItem(
      'evolutionSimData',
      JSON.stringify(this.currentGenStats)
    );
  }

  private recordSimulationData(): void {
    const agents = this.simulation.getAgents();
    const foods = this.simulation.getFoods();
    const zones = this.simulation.getZones();
    const activeSpecies = this.simulation
      .getSpeciesManager()
      .getActiveSpecies();

    // Cache common calculations
    const fitnessValues = agents.map((a) => a.fitness);
    const mutationRates = agents.map((a) => a.genome.mutationRate);

    // Create simulation data object
    const data = {
      timestamp: Date.now(),
      generation: this.simulation.getGeneration(),
      tick: this.simulation.getTickCount(),

      agents: {
        count: agents.length,
        fitness: this.calculateDistributionStats(fitnessValues),
        energyLevels: this.calculateEnergyStats(),
        ageSummary: this.calculateAgeStats(),
        topAgents: this.getTopAgentData(10),
        mutationRates: this.calculateDistributionStats(mutationRates),
      },

      species: {
        count: activeSpecies.length,
        distribution: this.getSpeciesDistribution(),
      },

      environment: {
        timeOfDay: this.simulation.getEnvironmentCycle().getTimeOfDay(),
        foodCounts: this.getFoodCounts(),
        zonePopulation: this.getZonePopulation(),
      },

      neuralNetworks: {
        weightDistribution: this.calculateWeightDistribution(),
        inputActivations: this.calculateInputActivations(),
        outputActivations: this.calculateOutputActivations(),
      },

      performance: {
        energyEfficiency: this.calculateEnergyEfficiency(),
        causeOfDeath: this.getDeathCauses(),
        behaviorMetrics: this.calculateBehaviorMetrics(),
      },
    };

    // Update current generation stats
    this.updateCurrentGenStats(data);

    // Store in localStorage
    localStorage.setItem('evolutionSimData', JSON.stringify(data));
  }

  private updateCurrentGenStats(data: any): void {
    if (!this.currentGenStats) {
      // Initialize if first time
      this.currentGenStats = {
        generation: data.generation,
        timestamp: Date.now(),
        tickCount: data.tick,

        agentCount: data.agents.count,
        avgFitness: data.agents.fitness.avg,
        maxFitness: data.agents.fitness.max,
        minFitness: data.agents.fitness.min,

        speciesCount: data.species.count,
        speciesDistribution: this.convertSpeciesDistribution(
          data.species.distribution
        ),

        avgMutationRate: data.agents.mutationRates.avg,
        mutationRateDistribution: this.calculateMutationRateDistribution(),

        energyEfficiency: data.performance.energyEfficiency,
        avgLifespan: this.calculateAvgLifespan(),
        deathCauses: {
          starvation: 0,
          collision: 0,
          other: 0,
        },

        foodPreferences: this.calculateFoodPreferences(),
        zoneDistribution: data.environment.zonePopulation,

        networkStats: {
          inputActivations: data.neuralNetworks.inputActivations,
          outputActivations: data.neuralNetworks.outputActivations,
          weightDistribution: data.neuralNetworks.weightDistribution,
        },
      };
    } else {
      // Update existing stats
      this.currentGenStats.timestamp = Date.now();
      this.currentGenStats.tickCount = data.tick;
      this.currentGenStats.agentCount = data.agents.count;
      this.currentGenStats.avgFitness = data.agents.fitness.avg;
      this.currentGenStats.maxFitness = data.agents.fitness.max;
      this.currentGenStats.minFitness = data.agents.fitness.min;
      this.currentGenStats.speciesCount = data.species.count;
      this.currentGenStats.speciesDistribution =
        this.convertSpeciesDistribution(data.species.distribution);
      this.currentGenStats.avgMutationRate = data.agents.mutationRates.avg;
      this.currentGenStats.mutationRateDistribution =
        this.calculateMutationRateDistribution();
      this.currentGenStats.energyEfficiency = data.performance.energyEfficiency;
      this.currentGenStats.foodPreferences = this.calculateFoodPreferences();
      this.currentGenStats.zoneDistribution = data.environment.zonePopulation;
      this.currentGenStats.networkStats = {
        inputActivations: data.neuralNetworks.inputActivations,
        outputActivations: data.neuralNetworks.outputActivations,
        weightDistribution: data.neuralNetworks.weightDistribution,
      };
    }
  }

  private recordGenerationStats(): void {
    if (this.currentGenStats) {
      // Save to history store
      this.historyStore.addGenerationStats(this.simulation.getGeneration(), {
        ...this.currentGenStats,
      });

      // Reset for next generation
      this.currentGenStats = null;
    }
  }

  // Additional analytical methods...

  private calculateDistributionStats(values: number[]) {
    if (values.length === 0) return { avg: 0, max: 0, min: 0, stdDev: 0 };

    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
    const max = Math.max(...values);
    const min = Math.min(...values);

    // Calculate standard deviation
    const squaredDiffs = values.map((val) => Math.pow(val - avg, 2));
    const variance =
      squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return { avg, max, min, stdDev };
  }

  private calculateEnergyStats() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return { avg: 0, max: 0, min: 0 };

    const energies = agents.map((a) => a.energy);
    return this.calculateDistributionStats(energies);
  }

  private calculateAgeStats() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return { avg: 0, max: 0, min: 0 };

    const ages = agents.map((a) => a.age);
    return this.calculateDistributionStats(ages);
  }

  private getTopAgentData(count: number) {
    const agents = this.simulation.getAgents();
    return agents
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, count)
      .map((agent) => ({
        id: agent.id,
        fitness: agent.fitness,
        energy: agent.energy,
        age: agent.age,
        speciesId: agent.species?.id,
        speciesName: agent.species?.name,
        mutationRate: agent.genome.mutationRate,
        stats: this.agentTracker.get(agent.id) || null,
      }));
  }

  private getSpeciesDistribution() {
    const species = this.simulation.getSpeciesManager().getActiveSpecies();
    return species.map((s) => ({
      id: s.id,
      name: s.name,
      count: s.members.length,
      color: s.color,
      bestFitness: s.bestFitness,
      creationGen: s.creationGeneration,
    }));
  }

  private convertSpeciesDistribution(
    distribution: any[]
  ): Record<number, number> {
    const result: Record<number, number> = {};
    distribution.forEach((s) => {
      result[s.id] = s.count;
    });
    return result;
  }

  private getFoodCounts() {
    const foods = this.simulation.getFoods();
    return {
      total: foods.length,
      byType: {
        [FoodType.BASIC]: foods.filter((f) => f.type === FoodType.BASIC).length,
        [FoodType.SUPER]: foods.filter((f) => f.type === FoodType.SUPER).length,
        [FoodType.POISON]: foods.filter((f) => f.type === FoodType.POISON)
          .length,
      },
    };
  }

  private getZonePopulation() {
    const zones = this.simulation.getZones();
    const agents = this.simulation.getAgents();

    const result: Record<string, number> = {};

    zones.forEach((zone) => {
      const zoneAgents = agents.filter((agent) =>
        zone.contains(agent.position)
      );
      result[zone.type] = zoneAgents.length;

      // Update agent zone tracking
      zoneAgents.forEach((agent) => {
        const stats = this.agentTracker.get(agent.id);
        if (stats) {
          stats.timeInZones[zone.type]++;
        }
      });
    });

    // Count agents not in any zone
    const agentsOutsideZones = agents.filter(
      (agent) => !zones.some((zone) => zone.contains(agent.position))
    ).length;

    result['outside'] = agentsOutsideZones;

    // Update agent "outside" zone tracking
    agents.forEach((agent) => {
      const stats = this.agentTracker.get(agent.id);
      if (stats && !zones.some((zone) => zone.contains(agent.position))) {
        stats.timeInZones['outside']++;
      }
    });

    return result;
  }

  private calculateWeightDistribution() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return { mean: 0, stdDev: 0, min: 0, max: 0 };

    // Collect all weights across all agents
    let allWeights: number[] = [];
    agents.forEach((agent) => {
      allWeights = allWeights.concat(agent.genome.weights);

      // Update agent's weight stats
      const stats = this.agentTracker.get(agent.id);
      if (stats) {
        const weights = agent.genome.weights;
        const mean = weights.reduce((sum, w) => sum + w, 0) / weights.length;

        const squaredDiffs = weights.map((w) => Math.pow(w - mean, 2));
        const variance =
          squaredDiffs.reduce((sum, diff) => sum + diff, 0) / weights.length;
        const stdDev = Math.sqrt(variance);

        stats.weightStats = { mean, stdDev };
      }
    });

    return this.calculateDistributionStats(allWeights);
  }

  private calculateInputActivations() {
    // Calculate average activation per input neuron
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return Array(12).fill(0);

    // This would require modifying agent.ts to track input activations
    // For now, return a placeholder
    return Array(12)
      .fill(0)
      .map(() => Math.random());
  }

  private calculateOutputActivations() {
    // Calculate average activation per output neuron
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return Array(3).fill(0);

    // This would require modifying agent.ts to track output activations
    // For now, return a placeholder
    return Array(3)
      .fill(0)
      .map(() => Math.random() * 0.5 + 0.25);
  }

  private calculateEnergyEfficiency() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return 0;

    // Calculate energy gained per distance traveled
    let totalEfficiency = 0;
    let count = 0;

    for (const agent of agents) {
      const stats = this.agentTracker.get(agent.id);
      if (stats && stats.distanceTraveled > 0) {
        const energyGained =
          stats.foodConsumed[FoodType.BASIC] * 20 +
          stats.foodConsumed[FoodType.SUPER] * 50 -
          stats.foodConsumed[FoodType.POISON] * 30;

        const efficiency = energyGained / stats.distanceTraveled;
        totalEfficiency += efficiency;
        count++;
      }
    }

    return count > 0 ? totalEfficiency / count : 0;
  }

  private getDeathCauses() {
    // This would be populated from handleAgentDied
    return (
      this.currentGenStats?.deathCauses || {
        starvation: 0,
        collision: 0,
        other: 0,
      }
    );
  }

  private calculateAvgLifespan() {
    // Get from lineage tracker
    const currentGen = this.simulation.getGeneration();
    const records = this.simulation
      .getLineageTracker()
      .getGenerationRecords(currentGen - 1) // Previous generation
      .filter((r) => r.deathTick !== null);

    if (records.length === 0) return 0;

    const lifespans = records.map((r) => (r.deathTick || 0) - r.birthTick);

    return lifespans.reduce((sum, span) => sum + span, 0) / lifespans.length;
  }

  private calculateMutationRateDistribution() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) {
      return { min: 0, max: 0, buckets: Array(10).fill(0) };
    }

    const rates = agents.map((a) => a.genome.mutationRate);
    const min = Math.min(...rates);
    const max = Math.max(...rates);

    // Create 10 buckets
    const bucketSize = (max - min) / 10;
    const buckets = Array(10).fill(0);

    rates.forEach((rate) => {
      const bucketIndex = Math.min(9, Math.floor((rate - min) / bucketSize));
      buckets[bucketIndex]++;
    });

    return { min, max, buckets };
  }

  private calculateFoodPreferences() {
    const result: Record<FoodType, number> = {
      [FoodType.BASIC]: 0,
      [FoodType.SUPER]: 0,
      [FoodType.POISON]: 0,
    };

    // Count food consumed this generation from agent tracker
    let totalFood = 0;

    for (const stats of this.agentTracker.values()) {
      result[FoodType.BASIC] += stats.foodConsumed[FoodType.BASIC];
      result[FoodType.SUPER] += stats.foodConsumed[FoodType.SUPER];
      result[FoodType.POISON] += stats.foodConsumed[FoodType.POISON];
    }

    totalFood =
      result[FoodType.BASIC] + result[FoodType.SUPER] + result[FoodType.POISON];

    // Convert to percentages
    if (totalFood > 0) {
      result[FoodType.BASIC] = parseFloat(
        ((result[FoodType.BASIC] / totalFood) * 100).toFixed(1)
      );
      result[FoodType.SUPER] = parseFloat(
        ((result[FoodType.SUPER] / totalFood) * 100).toFixed(1)
      );
      result[FoodType.POISON] = parseFloat(
        ((result[FoodType.POISON] / totalFood) * 100).toFixed(1)
      );
    }

    return result;
  }

  private calculateBehaviorMetrics() {
    // Calculate additional behavior metrics
    const agents = this.simulation.getAgents();

    // Calculate zone preferences (percentage of time spent in each zone)
    const zonePreferences: Record<string, number> = {};
    let totalZoneTime = 0;

    // Sum up total time in all zones across all agents
    for (const stats of this.agentTracker.values()) {
      for (const [zone, time] of Object.entries(stats.timeInZones)) {
        totalZoneTime += time;
        zonePreferences[zone] = (zonePreferences[zone] || 0) + time;
      }
    }

    // Convert to percentages
    if (totalZoneTime > 0) {
      for (const zone in zonePreferences) {
        zonePreferences[zone] = parseFloat(
          ((zonePreferences[zone] / totalZoneTime) * 100).toFixed(1)
        );
      }
    }

    return {
      zonePreferences,
      avgCollisions: this.calculateAvgCollisions(),
    };
  }

  private calculateAvgCollisions() {
    let totalCollisions = 0;
    let agentCount = 0;

    for (const stats of this.agentTracker.values()) {
      totalCollisions += stats.collisions;
      agentCount++;
    }

    return agentCount > 0 ? totalCollisions / agentCount : 0;
  }

  private updateAgentTracking() {
    const agents = this.simulation.getAgents();

    for (const agent of agents) {
      const stats = this.agentTracker.get(agent.id);
      if (stats) {
        // Update basic stats
        stats.age = agent.age;
        stats.energy = agent.energy;
        stats.fitness = agent.fitness;

        // Update distance traveled (requires tracking previous position)
        const lastPos = (stats as any).lastPosition;
        if (lastPos) {
          const dx = agent.position.x - lastPos.x;
          const dy = agent.position.y - lastPos.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          stats.distanceTraveled += distance;
        }

        // Store current position for next update
        (stats as any).lastPosition = { ...agent.position };

        // Update speed tracking
        const speed = Math.sqrt(
          agent.velocity.x * agent.velocity.x +
            agent.velocity.y * agent.velocity.y
        );

        // Exponential moving average for speed
        const alpha = 0.1; // Smoothing factor
        stats.avgSpeed = alpha * speed + (1 - alpha) * stats.avgSpeed;
      }
    }
  }

  public getHistory(): HistoryStore {
    return this.historyStore;
  }

  public exportHistory(format: 'json' | 'csv' = 'json'): string {
    if (format === 'csv') {
      return this.generateCSV();
    }

    return JSON.stringify(this.historyStore.exportHistory());
  }

  private generateCSV(): string {
    const stats = this.historyStore.getAllStats();
    if (stats.length === 0) return '';

    // Generate headers
    const headers = [
      'Generation',
      'Timestamp',
      'AgentCount',
      'AvgFitness',
      'MaxFitness',
      'SpeciesCount',
      'AvgMutationRate',
      'EnergyEfficiency',
      'AvgLifespan',
      'DeathByStarvation',
      'DeathByCollision',
      'BasicFoodPreference',
      'SuperFoodPreference',
      'PoisonFoodPreference',
    ];

    // Generate rows
    const rows = stats.map((stat) => [
      stat.generation,
      stat.timestamp,
      stat.agentCount,
      stat.avgFitness.toFixed(2),
      stat.maxFitness.toFixed(2),
      stat.speciesCount,
      stat.avgMutationRate.toFixed(4),
      stat.energyEfficiency.toFixed(2),
      stat.avgLifespan.toFixed(1),
      stat.deathCauses.starvation,
      stat.deathCauses.collision,
      stat.foodPreferences[FoodType.BASIC],
      stat.foodPreferences[FoodType.SUPER],
      stat.foodPreferences[FoodType.POISON],
    ]);

    // Join into CSV string
    return [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
  }
}
