// src/persistence/simulation-saver.ts
import { Simulation } from '../core/sim';
import { Agent } from '../entities/agent';
import { Genome } from '../evolution/genome';
import { createVector } from '../utils/math';
import { Obstacle } from '../entities/obstacles';
import { Zone } from '../environment/zone';
import { Food } from '../entities/food';
import config from '../cfg';

export class SimulationPersistence {
  private simulation: Simulation;

  constructor(simulation: Simulation) {
    this.simulation = simulation;
  }

  public saveToLocalStorage(): boolean {
    try {
      // Create serializable state object
      const state = {
        timestamp: Date.now(),
        generation: this.simulation.getGeneration(),
        tickCount: this.simulation.getTickCount(),

        // Agent data (lightweight version)
        agents: this.simulation.getAgents().map((agent) => ({
          id: agent.id,
          fitness: agent.fitness,
          energy: agent.energy,
          age: agent.age,
          position: agent.position,
          velocity: agent.velocity,
          speciesId: agent.species?.id || 0,
          parentIds: agent.parentIds,
          metabolismMultiplier: agent.metabolismMultiplier,
          genome: {
            weights: agent.genome.weights,
            mutationRate: agent.genome.mutationRate,
          },
        })),

        // Species data
        species: this.simulation
          .getSpeciesManager()
          .getSpecies()
          .map((species) => ({
            id: species.id,
            name: species.name,
            color: species.color,
            creationGeneration: species.creationGeneration,
            lastImprovedGeneration: species.lastImprovedGeneration,
            bestFitness: species.bestFitness,
            members: species.members,
            representative: {
              weights: species.representative.weights,
              mutationRate: species.representative.mutationRate,
            },
          })),

        // Environment data
        environment: {
          timeOfDay: this.simulation.getEnvironmentCycle().getTimeOfDay(),
          zones: this.simulation.getZones().map((zone) => ({
            position: zone.position,
            radius: zone.radius,
            type: zone.type,
          })),
          obstacles: this.simulation.getObstacles().map((obstacle) => ({
            position: obstacle.position,
            size: obstacle.size,
          })),
        },

        // Lineage data
        lineage: Array.from(
          this.simulation.getLineageTracker().getAllRecords()
        ),

        // History data
        generationHistory: this.getGenerationHistory(),
      };

      // Save to localStorage
      localStorage.setItem('evolutionSimState', JSON.stringify(state));
      console.log(`Simulation saved at generation ${state.generation}`);

      return true;
    } catch (error) {
      console.error('Error saving simulation:', error);
      return false;
    }
  }

  private getGenerationHistory() {
    // Get historical data from data recorder if available
    // For now, create a minimal history from current state
    return [
      {
        generation: this.simulation.getGeneration(),
        timestamp: Date.now(),
        agentCount: this.simulation.getAgents().length,
        speciesCount: this.simulation.getSpeciesManager().getActiveSpecies()
          .length,
        avgFitness: this.calculateAvgFitness(),
        maxFitness: this.calculateMaxFitness(),
      },
    ];
  }

  private calculateAvgFitness(): number {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return 0;
    return (
      agents.reduce((sum, agent) => sum + agent.fitness, 0) / agents.length
    );
  }

  private calculateMaxFitness(): number {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return 0;
    return Math.max(...agents.map((agent) => agent.fitness));
  }

  public exportToFile(): void {
    try {
      // Create serializable state - same structure as in saveToLocalStorage
      const state = {
        timestamp: Date.now(),
        generation: this.simulation.getGeneration(),
        tickCount: this.simulation.getTickCount(),

        agents: this.simulation.getAgents().map((agent) => ({
          id: agent.id,
          fitness: agent.fitness,
          energy: agent.energy,
          age: agent.age,
          position: agent.position,
          velocity: agent.velocity,
          speciesId: agent.species?.id || 0,
          parentIds: agent.parentIds,
          metabolismMultiplier: agent.metabolismMultiplier,
          genome: {
            weights: agent.genome.weights,
            mutationRate: agent.genome.mutationRate,
          },
        })),

        species: this.simulation
          .getSpeciesManager()
          .getSpecies()
          .map((species) => ({
            id: species.id,
            name: species.name,
            color: species.color,
            creationGeneration: species.creationGeneration,
            lastImprovedGeneration: species.lastImprovedGeneration,
            bestFitness: species.bestFitness,
            members: species.members,
            representative: {
              weights: species.representative.weights,
              mutationRate: species.representative.mutationRate,
            },
          })),

        environment: {
          timeOfDay: this.simulation.getEnvironmentCycle().getTimeOfDay(),
          zones: this.simulation.getZones().map((zone) => ({
            position: zone.position,
            radius: zone.radius,
            type: zone.type,
          })),
          obstacles: this.simulation.getObstacles().map((obstacle) => ({
            position: obstacle.position,
            size: obstacle.size,
          })),
          // Add this food data
          food: this.simulation.getFoods().map((food) => ({
            position: food.position,
            type: food.type,
            energy: food.energy,
            size: food.size,
            isConsumed: food.isConsumed,
          })),
        },

        lineage: Array.from(
          this.simulation.getLineageTracker().getAllRecords()
        ),
        generationHistory: this.getGenerationHistory(),
      };

      // Convert to JSON string
      const stateJson = JSON.stringify(state);

      // Create download link
      const blob = new Blob([stateJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      // Create download link
      const a = document.createElement('a');
      a.href = url;
      a.download = `evolution-sim-gen${state.generation}.json`;
      document.body.appendChild(a);
      a.click();

      // Clean up
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting simulation:', error);
    }
  }

  public loadFromLocalStorage(): boolean {
    try {
      const stateJson = localStorage.getItem('evolutionSimState');
      if (!stateJson) {
        console.log('No saved simulation found');
        return false;
      }

      // Parse state
      const state = JSON.parse(stateJson);

      // Reset simulation first
      this.simulation.pause();

      // Restore generation and tick count
      this.simulation.setGeneration(state.generation);
      this.simulation.setTickCount(state.tickCount);

      // Restore species first (needed for agent reconstruction)
      this.restoreSpecies(state.species);

      // Restore agents
      this.restoreAgents(state.agents);

      // Restore environment state if available
      if (state.environment) {
        this.restoreEnvironment(state.environment);
      }

      // Restore lineage data if available
      if (state.lineage) {
        this.restoreLineage(state.lineage);
      }

      console.log(`Loaded simulation from generation ${state.generation}`);
      return true;
    } catch (error) {
      console.error('Error loading simulation:', error);
      return false;
    }
  }

  private restoreSpecies(speciesData: any[]): void {
    const speciesManager = this.simulation.getSpeciesManager();

    // Clear existing species
    speciesManager.clearAllSpecies();

    // Restore each species
    speciesData.forEach((speciesInfo) => {
      // Create representative genome
      const representativeGenome = new Genome(
        speciesInfo.representative.weights,
        speciesInfo.representative.mutationRate
      );

      // Create and configure species
      const species = speciesManager.createSpecies(
        speciesInfo.id,
        representativeGenome,
        speciesInfo.creationGeneration
      );

      // Set additional properties
      species.name = speciesInfo.name;
      species.color = speciesInfo.color;
      species.lastImprovedGeneration = speciesInfo.lastImprovedGeneration;
      species.bestFitness = speciesInfo.bestFitness;

      // Note: members will be added when agents are restored
    });
  }

  private restoreAgents(agentData: any[]): void {
    // Clear existing agents
    this.simulation.clearAgents();

    // Create new agents from saved data
    const newAgents: Agent[] = [];
    const speciesManager = this.simulation.getSpeciesManager();

    agentData.forEach((agentInfo) => {
      // Create genome from saved weights
      const genome = new Genome(
        agentInfo.genome.weights,
        agentInfo.genome.mutationRate
      );

      // Create position vector
      const position = createVector(agentInfo.position.x, agentInfo.position.y);

      // Create new agent
      const agent = new Agent(position, genome, this.simulation, agentInfo.id);

      // Restore agent properties
      agent.energy = agentInfo.energy;
      agent.fitness = agentInfo.fitness;
      agent.age = agentInfo.age;
      agent.parentIds = agentInfo.parentIds || [];
      agent.metabolismMultiplier = agentInfo.metabolismMultiplier || 1.0;

      // Restore velocity if available
      if (agentInfo.velocity) {
        agent.velocity = createVector(
          agentInfo.velocity.x,
          agentInfo.velocity.y
        );
      }

      // Assign species
      if (agentInfo.speciesId) {
        agent.species = speciesManager.getSpeciesById(agentInfo.speciesId);

        // Add agent to species members list
        if (agent.species) {
          agent.species.addMember(agent.id);
        }
      }

      newAgents.push(agent);
    });

    // Add restored agents to simulation
    this.simulation.setAgents(newAgents);
  }

  private restoreEnvironment(environmentData: any): void {
    // Restore environment cycle state
    if (environmentData.timeOfDay !== undefined) {
      this.simulation
        .getEnvironmentCycle()
        .setTimeOfDay(environmentData.timeOfDay);
    }

    // Restore zones if available
    if (environmentData.zones && Array.isArray(environmentData.zones)) {
      // Clear existing zones
      this.simulation.clearZones();

      // Create new zones from saved data
      const newZones = environmentData.zones.map((zoneData: any) => {
        const position = createVector(zoneData.position.x, zoneData.position.y);
        return new Zone(position, zoneData.radius, zoneData.type);
      });

      // Set zones in simulation
      this.simulation.setZones(newZones);
    }

    // Restore obstacles if available
    if (environmentData.obstacles && Array.isArray(environmentData.obstacles)) {
      // Clear existing obstacles
      this.simulation.clearObstacles();

      // Create new obstacles from saved data
      const newObstacles = environmentData.obstacles.map(
        (obstacleData: any) => {
          const position = createVector(
            obstacleData.position.x,
            obstacleData.position.y
          );
          return new Obstacle(position, obstacleData.size);
        }
      );

      // Set obstacles in simulation
      this.simulation.setObstacles(newObstacles);
    }

    // Restore food if available
    if (environmentData.food && Array.isArray(environmentData.food)) {
      // Clear existing food
      this.simulation.clearFood();

      // Create new food from saved data
      const newFood = environmentData.food.map((foodData: any) => {
        const position = createVector(foodData.position.x, foodData.position.y);
        return new Food(position, foodData.type);
      });

      // Set food in simulation
      this.simulation.setFood(newFood);
    } else {
      // If no food data, respawn food based on config
      this.simulation.spawnFood(config.initialFoodCount);
    }
  }

  private restoreLineage(lineageData: any[]): void {
    const lineageTracker = this.simulation.getLineageTracker();

    // Clear existing lineage data
    lineageTracker.clearAllRecords();

    // Restore lineage records
    lineageData.forEach((record) => {
      lineageTracker.addRecordDirect(record);
    });
  }

  public importFromFile(file: File): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        try {
          const stateJson = event.target?.result as string;
          const state = JSON.parse(stateJson);

          // Pause simulation
          this.simulation.pause();

          // Restore generation and tick count
          this.simulation.setGeneration(state.generation);
          this.simulation.setTickCount(state.tickCount);

          // Restore species first (needed for agent reconstruction)
          this.restoreSpecies(state.species);

          // Restore agents
          this.restoreAgents(state.agents);

          // Restore environment state if available
          if (state.environment) {
            this.restoreEnvironment(state.environment);
          }

          // Restore lineage data if available
          if (state.lineage) {
            this.restoreLineage(state.lineage);
          }

          console.log(
            `Imported simulation from generation ${state.generation}`
          );
          resolve(true);
        } catch (error) {
          console.error('Error parsing imported simulation:', error);
          reject(error);
        }
      };

      reader.onerror = () => {
        reject(new Error('Error reading file'));
      };

      reader.readAsText(file);
    });
  }
}
