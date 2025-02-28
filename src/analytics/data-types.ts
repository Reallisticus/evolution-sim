// src/analytics/data-types.ts - Common data types
import { FoodType } from '../entities/food';
import { ZoneType } from '../environment/zone';

export interface SimulationStats {
  generation: number;
  timestamp: number;
  tickCount: number;

  // Agent stats
  agentCount: number;
  avgFitness: number;
  maxFitness: number;
  minFitness: number;

  // Species stats
  speciesCount: number;
  speciesDistribution: Record<number, number>;

  // Mutation stats
  avgMutationRate: number;
  mutationRateDistribution: {
    min: number;
    max: number;
    buckets: number[];
  };

  // Performance stats
  energyEfficiency: number;
  avgLifespan: number;
  deathCauses: {
    starvation: number;
    collision: number;
    other: number;
  };

  // Behavior stats
  foodPreferences: Record<FoodType, number>;
  zoneDistribution: Record<ZoneType | 'outside', number>;

  // Neural network stats
  networkStats: {
    inputActivations: number[];
    outputActivations: number[];
    weightDistribution: {
      mean: number;
      stdDev: number;
      min: number;
      max: number;
    };
  };
}

// Detailed agent stats for individual tracking
export interface AgentStats {
  id: number;
  generation: number;
  parentIds: number[];
  age: number;
  energy: number;
  fitness: number;
  speciesId: number;

  // Movement
  distanceTraveled: number;
  avgSpeed: number;

  // Behavior
  foodConsumed: Record<FoodType, number>;
  timeInZones: Record<ZoneType | 'outside', number>;
  collisions: number;

  // Neural
  avgResponsesToFood: Record<FoodType, number[]>;
  avgResponsesToObstacles: number[];

  // Genetic
  mutationRate: number;
  weightStats: {
    mean: number;
    stdDev: number;
  };
}
