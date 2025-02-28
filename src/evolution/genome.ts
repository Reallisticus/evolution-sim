import { NeuralNetwork } from '../neural/network';

export class Genome {
  public weights: number[];
  public mutationRate: number;

  constructor(weights?: number[], mutationRate: number = 0.05) {
    this.weights = weights || [];
    this.mutationRate = mutationRate;
  }

  public static fromNetwork(network: NeuralNetwork): Genome {
    return new Genome(network.getWeights());
  }

  public static random(size: number): Genome {
    const weights = Array(size)
      .fill(0)
      .map(() => Math.random() * 2 - 1);
    return new Genome(weights);
  }

  public createNetwork(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number
  ): NeuralNetwork {
    const network = new NeuralNetwork(inputSize, hiddenSizes, outputSize);

    // If we have weights, set them in the network
    if (this.weights.length > 0) {
      network.setWeights(this.weights);
    }

    return network;
  }

  public copy(): Genome {
    return new Genome([...this.weights], this.mutationRate);
  }
}

// Update the mutation function
export function mutate(genome: Genome): Genome {
  const newGenome = genome.copy();

  // First, possibly mutate the mutation rate itself
  if (Math.random() < 0.1) {
    // 10% chance to mutate mutation rate
    // Mutation rate changes by up to ±50% of its current value
    const rateChange = (Math.random() * 2 - 1) * 0.5 * genome.mutationRate;
    newGenome.mutationRate = Math.max(
      0.001,
      Math.min(0.3, genome.mutationRate + rateChange)
    );
  }

  // Now mutate the weights using the (possibly new) mutation rate
  for (let i = 0; i < newGenome.weights.length; i++) {
    if (Math.random() < newGenome.mutationRate) {
      // Use a normal distribution for more realistic mutations
      // (centered on current value, most changes are small, few are large)
      const change = randomNormal() * 0.2; // Standard deviation of 0.2
      newGenome.weights[i] += change;
    }
  }

  return newGenome;
}

// Helper function for normal distribution
function randomNormal(): number {
  // Box-Muller transform
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
