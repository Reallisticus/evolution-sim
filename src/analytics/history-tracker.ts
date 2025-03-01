// src/analytics/history-tracker.ts
import { FoodType } from '../entities/food';
import { ZoneType } from '../environment/zone';
import { TimeOfDay } from '../environment/cycle';

export interface GenerationMetrics {
  id: number; // Generation number
  timestamp: number; // When this generation ended
  duration: number; // How many ticks this generation lasted

  // Population metrics
  agentCount: {
    initial: number; // Number of agents at start of generation
    final: number; // Number of agents at end of generation
    survived: number; // Number that survived (not died during generation)
    reproduced: number; // Number that successfully reproduced
  };

  // Fitness metrics
  fitness: {
    min: number;
    max: number;
    avg: number;
    median: number;
    stdDev: number; // Standard deviation of fitness
  };

  // Species metrics
  species: {
    count: number; // Number of species
    diversity: number; // Measure of genetic diversity
    dominant: {
      // Info about the most populous species
      id: number;
      name: string;
      count: number;
      avgFitness: number;
    };
    new: number; // New species formed this generation
    extinct: number; // Species that went extinct this generation
  };

  // Neural metrics
  neural: {
    avgWeights: number[]; // Average weights for key neural pathways
    avgMutationRate: number;
    minMutationRate: number;
    maxMutationRate: number;
  };

  // Environment interactions
  environment: {
    foodConsumed: Record<FoodType, number>;
    zoneDistribution: Record<ZoneType, number>;
    timeOfDayEfficiency: Record<TimeOfDay, number>; // Performance by time of day
    obstacleCollisions: number;
  };

  // Morphological metrics (if we add physical evolution)
  morphology?: {
    avgSize: number;
    avgSensorRange: number;
    avgMetabolicRate: number;
  };
}

export class HistoryTracker {
  private generationHistory: GenerationMetrics[] = [];
  private currentGeneration: Partial<GenerationMetrics> = {};
  private lastSaveTime: number = 0;
  private autoSaveInterval: number = 5; // Save every 5 generations
  private filename: string = 'evolution-history';

  constructor(loadFromStorage: boolean = true) {
    if (loadFromStorage) {
      this.loadFromLocalStorage();
    }
  }

  // Start tracking a new generation
  public startGeneration(generationId: number, agentCount: number): void {
    this.currentGeneration = {
      id: generationId,
      timestamp: Date.now(),
      agentCount: {
        initial: agentCount,
        final: 0,
        survived: 0,
        reproduced: 0,
      },
      fitness: {
        min: Infinity,
        max: 0,
        avg: 0,
        median: 0,
        stdDev: 0,
      },
      species: {
        count: 0,
        diversity: 0,
        dominant: {
          id: 0,
          name: '',
          count: 0,
          avgFitness: 0,
        },
        new: 0,
        extinct: 0,
      },
      neural: {
        avgWeights: [],
        avgMutationRate: 0,
        minMutationRate: Infinity,
        maxMutationRate: 0,
      },
      environment: {
        foodConsumed: {
          [FoodType.BASIC]: 0,
          [FoodType.SUPER]: 0,
          [FoodType.POISON]: 0,
        },
        zoneDistribution: {
          [ZoneType.NORMAL]: 0,
          [ZoneType.HARSH]: 0,
          [ZoneType.FERTILE]: 0,
          [ZoneType.BARREN]: 0,
        },
        timeOfDayEfficiency: {
          [TimeOfDay.DAY]: 0,
          [TimeOfDay.NIGHT]: 0,
        },
        obstacleCollisions: 0,
      },
    };
  }

  // Finish the current generation and save its metrics
  public endGeneration(metrics: Partial<GenerationMetrics>): void {
    // Merge the provided metrics with current generation tracking
    const completeMetrics = {
      ...this.currentGeneration,
      ...metrics,
      duration: Date.now() - (this.currentGeneration.timestamp || Date.now()),
    } as GenerationMetrics;

    // Add to history
    this.generationHistory.push(completeMetrics);

    // Auto-save periodically
    if (completeMetrics.id % this.autoSaveInterval === 0) {
      this.saveToLocalStorage();
    }

    this.currentGeneration = {};
  }

  // Record a food consumption event
  public recordFoodConsumption(foodType: FoodType): void {
    if (this.currentGeneration.environment?.foodConsumed) {
      this.currentGeneration.environment.foodConsumed[foodType]++;
    }
  }

  // Record obstacle collision
  public recordObstacleCollision(): void {
    if (this.currentGeneration.environment) {
      this.currentGeneration.environment.obstacleCollisions++;
    }
  }

  // Update agent zone distribution
  public updateZoneDistribution(distribution: Record<ZoneType, number>): void {
    if (this.currentGeneration.environment) {
      this.currentGeneration.environment.zoneDistribution = distribution;
    }
  }

  // Update time of day efficiency
  public updateTimeEfficiency(
    timeDistribution: Record<TimeOfDay, number>
  ): void {
    if (this.currentGeneration.environment) {
      this.currentGeneration.environment.timeOfDayEfficiency = timeDistribution;
    }
  }

  // Update neural metrics
  public updateNeuralMetrics(
    mutationRates: number[],
    keyWeights: number[]
  ): void {
    if (this.currentGeneration.neural) {
      // Calculate mutation rate statistics
      this.currentGeneration.neural.avgMutationRate =
        mutationRates.reduce((sum, rate) => sum + rate, 0) /
        mutationRates.length;
      this.currentGeneration.neural.minMutationRate = Math.min(
        ...mutationRates
      );
      this.currentGeneration.neural.maxMutationRate = Math.max(
        ...mutationRates
      );

      // Store average weights for key neural connections
      this.currentGeneration.neural.avgWeights = keyWeights;
    }
  }

  // Update species metrics
  public updateSpeciesMetrics(
    speciesCount: number,
    diversity: number,
    dominant: { id: number; name: string; count: number; avgFitness: number },
    newSpecies: number,
    extinctSpecies: number
  ): void {
    if (this.currentGeneration.species) {
      this.currentGeneration.species.count = speciesCount;
      this.currentGeneration.species.diversity = diversity;
      this.currentGeneration.species.dominant = dominant;
      this.currentGeneration.species.new = newSpecies;
      this.currentGeneration.species.extinct = extinctSpecies;
    }
  }

  // Get entire history
  public getHistory(): GenerationMetrics[] {
    return [...this.generationHistory];
  }

  // Get history for a specific generation
  public getGeneration(id: number): GenerationMetrics | undefined {
    return this.generationHistory.find((gen) => gen.id === id);
  }

  // Get a range of generations
  public getGenerationRange(
    startId: number,
    endId: number
  ): GenerationMetrics[] {
    return this.generationHistory.filter(
      (gen) => gen.id >= startId && gen.id <= endId
    );
  }

  // Get statistics over a range of generations
  public getStatistics(
    startGen: number = 0,
    endGen: number = Number.MAX_SAFE_INTEGER
  ): any {
    const range = this.getGenerationRange(startGen, endGen);

    if (range.length === 0) return null;

    // Calculate fitness progression
    const fitnessProgression = range.map((gen) => ({
      generation: gen.id,
      min: gen.fitness.min,
      max: gen.fitness.max,
      avg: gen.fitness.avg,
    }));

    // Calculate species diversity progression
    const speciesProgression = range.map((gen) => ({
      generation: gen.id,
      count: gen.species.count,
      diversity: gen.species.diversity,
      newSpecies: gen.species.new,
      extinctSpecies: gen.species.extinct,
    }));

    // Calculate food consumption patterns
    const foodConsumption = {
      basic: range.reduce(
        (sum, gen) => sum + gen.environment.foodConsumed[FoodType.BASIC],
        0
      ),
      super: range.reduce(
        (sum, gen) => sum + gen.environment.foodConsumed[FoodType.SUPER],
        0
      ),
      poison: range.reduce(
        (sum, gen) => sum + gen.environment.foodConsumed[FoodType.POISON],
        0
      ),
    };

    // Mutation rate progression
    const mutationRateProgression = range.map((gen) => ({
      generation: gen.id,
      avg: gen.neural.avgMutationRate,
      min: gen.neural.minMutationRate,
      max: gen.neural.maxMutationRate,
    }));

    return {
      generations: range.length,
      fitnessProgression,
      speciesProgression,
      foodConsumption,
      mutationRateProgression,
      // Add more aggregated statistics as needed
    };
  }

  // Save history to localStorage
  public saveToLocalStorage(): boolean {
    try {
      const historyJson = JSON.stringify(this.generationHistory);
      localStorage.setItem('evolutionHistory', historyJson);
      this.lastSaveTime = Date.now();
      console.log(
        `Saved evolution history with ${this.generationHistory.length} generations`
      );
      return true;
    } catch (error) {
      console.error('Failed to save evolution history:', error);
      return false;
    }
  }

  // Load history from localStorage
  public loadFromLocalStorage(): boolean {
    try {
      const historyJson = localStorage.getItem('evolutionHistory');
      if (!historyJson) return false;

      this.generationHistory = JSON.parse(historyJson);
      console.log(
        `Loaded evolution history with ${this.generationHistory.length} generations`
      );
      return true;
    } catch (error) {
      console.error('Failed to load evolution history:', error);
      return false;
    }
  }

  // Export history to file
  public exportToFile(): void {
    try {
      const historyJson = JSON.stringify(this.generationHistory, null, 2);
      const blob = new Blob([historyJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = `${this.filename}-${Date.now()}.json`;
      link.click();

      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export evolution history:', error);
    }
  }

  // Import history from file
  public importFromFile(file: File): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        try {
          const historyJson = event.target?.result as string;
          const importedHistory = JSON.parse(historyJson);

          // Validate imported data
          if (!Array.isArray(importedHistory)) {
            throw new Error('Invalid history format');
          }

          // Merge with existing history or replace it
          if (confirm('Append to existing history? Click Cancel to replace.')) {
            // Find highest generation ID to avoid duplicates
            const maxId = this.generationHistory.reduce(
              (max, gen) => Math.max(max, gen.id),
              0
            );

            // Filter out generations we already have
            const newGenerations = importedHistory.filter(
              (gen) =>
                !this.generationHistory.some(
                  (existing) => existing.id === gen.id
                )
            );

            this.generationHistory = [
              ...this.generationHistory,
              ...newGenerations,
            ];
            this.generationHistory.sort((a, b) => a.id - b.id);
          } else {
            this.generationHistory = importedHistory;
          }

          // Save the merged/replaced history
          this.saveToLocalStorage();

          resolve(true);
        } catch (error) {
          console.error('Error parsing history file:', error);
          reject(error);
        }
      };

      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };

      reader.readAsText(file);
    });
  }

  // Clear history
  public clearHistory(): void {
    if (confirm('Are you sure you want to clear all evolution history?')) {
      this.generationHistory = [];
      localStorage.removeItem('evolutionHistory');
    }
  }
}
