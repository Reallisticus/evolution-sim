// src/analytics/data-recorder.ts
// Create this new file for recording simulation data

import { Simulation } from '../core/sim';
import { Agent } from '../entities/agent';
import { Species } from '../evolution/species';
import { FoodType } from '../entities/food';

export class DataRecorder {
  private simulation: Simulation;
  private recordInterval: number = 500; // ms
  private intervalId?: number;

  constructor(simulation: Simulation) {
    this.simulation = simulation;
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
  }

  public stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  private recordData(): void {
    const data = {
      timestamp: Date.now(),
      generation: this.simulation.getGeneration(),
      tick: this.simulation.getTickCount(),

      agents: {
        count: this.simulation.getAgents().length,
        fitness: this.calculateFitnessStats(),
        energyLevels: this.calculateEnergyStats(),
        ageSummary: this.calculateAgeStats(),
        topAgents: this.getTopAgentData(10),
      },

      species: {
        count: this.simulation.getSpeciesManager().getActiveSpecies().length,
        distribution: this.getSpeciesDistribution(),
      },

      environment: {
        timeOfDay: this.simulation.getEnvironmentCycle().getTimeOfDay(),
        foodCounts: this.getFoodCounts(),
        zonePopulation: this.getZonePopulation(),
      },
    };

    // Store in localStorage
    localStorage.setItem('evolutionSimData', JSON.stringify(data));
  }

  private calculateFitnessStats() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return { avg: 0, max: 0, min: 0 };

    const fitnesses = agents.map((a) => a.fitness);
    return {
      avg: fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length,
      max: Math.max(...fitnesses),
      min: Math.min(...fitnesses),
    };
  }

  private calculateEnergyStats() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return { avg: 0, max: 0, min: 0 };

    const energies = agents.map((a) => a.energy);
    return {
      avg: energies.reduce((sum, e) => sum + e, 0) / energies.length,
      max: Math.max(...energies),
      min: Math.min(...energies),
    };
  }

  private calculateAgeStats() {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return { avg: 0, max: 0, min: 0 };

    const ages = agents.map((a) => a.age);
    return {
      avg: ages.reduce((sum, a) => sum + a, 0) / ages.length,
      max: Math.max(...ages),
      min: Math.min(...ages),
    };
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
    });

    // Count agents not in any zone
    const agentsOutsideZones = agents.filter(
      (agent) => !zones.some((zone) => zone.contains(agent.position))
    ).length;

    result['outside'] = agentsOutsideZones;

    return result;
  }
}
