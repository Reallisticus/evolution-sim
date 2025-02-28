// src/entities/agent.ts
import {
  Vector2D,
  createVector,
  normalizeVector,
  addVectors,
  subtractVectors,
  limitVector,
  distanceBetween,
  magnitudeVector,
  multiplyVector,
} from '../utils/math';
import { NeuralNetwork } from '../neural/network';
import { Genome } from '../evolution/genome';
import { Food, FoodType } from './food';
import config from '../cfg';
import { mutate, crossover, CrossoverType } from '../evolution/mutation'; // Add this at the top with other imports
import { Obstacle } from './obstacles';
import { Species } from '../evolution/species';
import { Simulation } from '../core/sim'; // Adjust path if needed

export class Agent {
  public position: Vector2D;
  public velocity: Vector2D;
  public acceleration: Vector2D;
  public size: number;
  public energy: number;
  public genome: Genome;
  public brain: NeuralNetwork;
  public age: number = 0;
  public dead: boolean = false;
  public fitness: number = 0;
  public color: number = config.agentColor;
  public metabolismMultiplier: number = 1.0;
  public id: number = 0; // Initialize with default value
  public species: Species | null = null;
  public parentIds: number[] = [];
  public simulation: Simulation; // Add simulation property

  constructor(
    position?: Vector2D,
    genome?: Genome,
    simulation?: Simulation,
    id: number = 0
  ) {
    // Store simulation reference and ID
    this.simulation = simulation as Simulation;
    this.id = id;

    // Initialize position
    this.position =
      position ||
      createVector(
        Math.random() * config.worldWidth,
        Math.random() * config.worldHeight
      );

    // Initialize physics
    this.velocity = createVector(0, 0);
    this.acceleration = createVector(0, 0);
    this.size = config.agentSize;

    // Initialize stats
    this.energy = config.agentMaxEnergy / 2;

    // Calculate expected genome size based on network architecture
    // This is important for compatibility when we change network size
    const inputSize = config.inputNeurons;
    const hiddenSizes = config.hiddenNeurons;
    const outputSize = config.outputNeurons;

    // Calculate weights and biases for each layer
    let expectedWeights = 0;
    let layerSizes = [inputSize, ...hiddenSizes, outputSize];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      // Weights: inputs * outputs for this layer
      expectedWeights += layerSizes[i] * layerSizes[i + 1];
      // Biases: one per output neuron
      expectedWeights += layerSizes[i + 1];
    }

    // Initialize genome with correct size
    if (genome && genome.weights.length !== expectedWeights) {
      // Genome size doesn't match - create a new random one with correct size
      console.warn('Genome size mismatch, creating new random genome');
      this.genome = Genome.random(expectedWeights);
    } else {
      // Use provided genome or create random one
      this.genome = genome || Genome.random(expectedWeights);
    }

    // Initialize neural network
    this.brain = this.genome.createNetwork(
      config.inputNeurons,
      config.hiddenNeurons,
      config.outputNeurons
    );
  }

  public update(
    foods: Food[],
    obstacles: Obstacle[] = [],
    movementMultiplier: number = 1.0
  ): void {
    this.age++;

    // Reduce energy based on metabolism
    this.energy -= config.agentMetabolismRate * this.metabolismMultiplier;

    if (this.energy <= 0) {
      this.dead = true;
      return;
    }
    // Get sensory input
    // Get sensory input (now includes obstacles)
    const inputs = this.getSensoryInputs(foods, obstacles);

    // Get neural network outputs
    const outputs = this.brain.predict(inputs);

    // Apply outputs to movement
    this.applyOutputs(outputs);

    // Update physics with movement multiplier
    this.updatePhysics(movementMultiplier);

    // Check for food consumption
    this.checkFoodConsumption(foods);

    // Check for obstacle collisions
    this.checkObstacleCollisions(obstacles);

    // Update fitness
    this.fitness = this.age + this.energy;
  }

  private getSensoryInputs(foods: Food[], obstacles: Obstacle[]): number[] {
    const inputs: number[] = [];

    // Add position (normalized to 0-1)
    inputs.push(this.position.x / config.worldWidth);
    inputs.push(this.position.y / config.worldHeight);

    // Add velocity (normalized)
    const speed = magnitudeVector(this.velocity) / config.agentMaxSpeed;
    inputs.push(speed);

    // Add energy (normalized)
    inputs.push(this.energy / config.agentMaxEnergy);

    // Find closest food
    let closestFood: Food | null = null;
    let closestDistance = Infinity;

    for (const food of foods) {
      if (!food.isConsumed) {
        const distance = distanceBetween(this.position, food.position);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestFood = food;
        }
      }
    }

    // Add closest food direction and distance (or 0 if none)
    if (closestFood) {
      const directionVector = normalizeVector(
        subtractVectors(closestFood.position, this.position)
      );
      inputs.push(directionVector.x);
      inputs.push(directionVector.y);
      inputs.push(
        closestDistance /
          Math.sqrt(config.worldWidth ** 2 + config.worldHeight ** 2)
      );

      // Add food type as numeric input
      inputs.push(
        closestFood.type === FoodType.BASIC
          ? 0.5
          : closestFood.type === FoodType.SUPER
          ? 1.0
          : 0.0
      );
    } else {
      inputs.push(0);
      inputs.push(0);
      inputs.push(1); // Max distance
      inputs.push(0.5); // Default food type
    }

    // Find closest obstacle
    let closestObstacle: Obstacle | null = null;
    closestDistance = Infinity;

    for (const obstacle of obstacles) {
      const distance = distanceBetween(this.position, obstacle.position);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestObstacle = obstacle;
      }
    }

    // Add closest obstacle direction and distance
    if (closestObstacle && closestDistance < 150) {
      const directionVector = normalizeVector(
        subtractVectors(closestObstacle.position, this.position)
      );
      inputs.push(directionVector.x);
      inputs.push(directionVector.y);
      inputs.push(closestDistance / 150); // Normalized to detection range
    } else {
      inputs.push(0);
      inputs.push(0);
      inputs.push(1); // Max distance
    }

    // Add random input for some unpredictability
    inputs.push(Math.random());

    return inputs;
  }

  private applyOutputs(outputs: number[]): void {
    // Reset acceleration
    this.acceleration = createVector(0, 0);

    // First output: Forward force
    const forwardForce = outputs[0] * config.agentMaxForce;
    const forwardVector = normalizeVector(
      this.velocity.x === 0 && this.velocity.y === 0
        ? createVector(1, 0)
        : this.velocity
    );

    // Second output: Turn (rotation)
    const turnForce = (outputs[1] * 2 - 1) * config.agentMaxForce;
    const turnVector = createVector(-forwardVector.y, forwardVector.x);

    // Third output: Speed multiplier
    const speedMultiplier = outputs[2] * 2; // 0 to 2

    // Apply forces
    const movementForce = addVectors(
      { x: forwardVector.x * forwardForce, y: forwardVector.y * forwardForce },
      { x: turnVector.x * turnForce, y: turnVector.y * turnForce }
    );

    this.acceleration = {
      x: movementForce.x * speedMultiplier,
      y: movementForce.y * speedMultiplier,
    };
  }

  public updatePhysics(movementMultiplier: number = 1.0): void {
    // Update velocity
    this.velocity = addVectors(this.velocity, this.acceleration);

    // Limit velocity
    this.velocity = limitVector(
      this.velocity,
      config.agentMaxSpeed * movementMultiplier
    );

    // Update position
    this.position = addVectors(
      this.position,
      multiplyVector(this.velocity, movementMultiplier)
    );

    // Wrap around edges
    if (this.position.x < 0) this.position.x = config.worldWidth;
    if (this.position.x > config.worldWidth) this.position.x = 0;
    if (this.position.y < 0) this.position.y = config.worldHeight;
    if (this.position.y > config.worldHeight) this.position.y = 0;
  }

  private checkObstacleCollisions(obstacles: Obstacle[]): void {
    for (const obstacle of obstacles) {
      const distance = distanceBetween(this.position, obstacle.position);

      // If colliding with obstacle, bounce back
      if (distance < this.size + obstacle.size) {
        // Calculate bounce direction (away from obstacle)
        const awayVector = normalizeVector(
          subtractVectors(this.position, obstacle.position)
        );

        // Move agent outside of obstacle
        const overlapDistance = this.size + obstacle.size - distance;
        this.position = addVectors(
          this.position,
          multiplyVector(awayVector, overlapDistance * 1.1)
        );

        // Reverse velocity component in direction of obstacle
        const dotProduct =
          this.velocity.x * awayVector.x + this.velocity.y * awayVector.y;

        this.velocity = subtractVectors(
          this.velocity,
          multiplyVector(awayVector, 2 * dotProduct)
        );

        // Reduce energy due to collision
        this.energy -= 5;
      }
    }
  }

  private checkFoodConsumption(foods: Food[]): void {
    // Only check nearby food
    const nearbyFoods = foods.filter((food) => {
      if (food.isConsumed) return false;

      // Quick distance check first (square distance for speed)
      const dx = food.position.x - this.position.x;
      const dy = food.position.y - this.position.y;
      const squareDistance = dx * dx + dy * dy;

      // Only do precise check if roughly in range
      return (
        squareDistance <
        (this.size + food.size + 10) * (this.size + food.size + 10)
      );
    });

    // Now do precise checks only on nearby food
    for (const food of nearbyFoods) {
      const distance = distanceBetween(this.position, food.position);

      // If close enough, consume the food
      if (distance < this.size + food.size) {
        this.energy += food.energy;

        // If poison made energy negative, cap at 1 to prevent immediate death
        if (this.energy < 1 && food.type === FoodType.POISON) {
          this.energy = 1;
        }

        // Cap energy at max
        if (this.energy > config.agentMaxEnergy) {
          this.energy = config.agentMaxEnergy;
        }

        food.isConsumed = true;
      }
    }
  }

  public canReproduce(): boolean {
    return this.energy > config.agentReproductionEnergy;
  }

  public reproduce(partner?: Agent | null): Agent {
    // Reduce energy due to reproduction
    this.energy = this.energy / 2;

    let childGenome: Genome;

    if (partner && partner.canReproduce()) {
      // Sexual reproduction (with crossover)
      partner.energy = partner.energy / 2;

      // Select crossover type - prefer more complex crossover as generations progress
      let crossoverType: CrossoverType;
      const generation = this.simulation.getGeneration();

      if (generation < 5) {
        crossoverType = CrossoverType.UNIFORM;
      } else if (generation < 15) {
        crossoverType = CrossoverType.SINGLE_POINT;
      } else {
        crossoverType = CrossoverType.MULTI_POINT;
      }

      // Use enhanced crossover with selected type
      childGenome = crossover(this.genome, partner.genome, crossoverType);

      // Child gets mutation rate that's the average of parents with slight variation
      childGenome.mutationRate =
        (this.genome.mutationRate + partner.genome.mutationRate) / 2;
    } else {
      // Asexual reproduction (clone with mutation)
      childGenome = this.genome.copy();
    }

    // Apply mutations to the genome
    childGenome = mutate(childGenome);

    // Create child with position near parent
    const childPosition = addVectors(this.position, {
      x: Math.random() * 20 - 10,
      y: Math.random() * 20 - 10,
    });

    // Create the child agent
    const childAgent = new Agent(childPosition, childGenome, this.simulation);

    // Set parent IDs for lineage tracking
    if (partner) {
      childAgent.parentIds = [this.id, partner.id];
    } else {
      childAgent.parentIds = [this.id];
    }

    return childAgent;
  }

  public dispose(): void {
    if (this.brain) {
      this.brain.dispose();
    }
  }
}
