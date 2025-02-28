// src/evolution/mutation.ts
import { Genome } from './genome';
import config from '../cfg';

export enum CrossoverType {
  UNIFORM, // Random gene selection (current method)
  SINGLE_POINT, // Single crossover point
  MULTI_POINT, // Multiple crossover points
}

export function mutate(
  genome: Genome,
  rate: number = config.mutationRate,
  amount: number = config.mutationAmount
): Genome {
  const newGenome = genome.copy();

  for (let i = 0; i < newGenome.weights.length; i++) {
    if (Math.random() < rate) {
      // Apply mutation
      newGenome.weights[i] += (Math.random() * 2 - 1) * amount;
    }
  }

  return newGenome;
}

export function crossover(
  parent1: Genome,
  parent2: Genome,
  crossoverType: CrossoverType = CrossoverType.MULTI_POINT
): Genome {
  const child = new Genome();

  // Ensure parents have same length
  if (parent1.weights.length !== parent2.weights.length) {
    throw new Error('Parents must have the same genome length');
  }

  const length = parent1.weights.length;

  switch (crossoverType) {
    case CrossoverType.UNIFORM:
      // Random gene selection (50% chance from each parent)
      child.weights = parent1.weights.map((gene, i) =>
        Math.random() < 0.5 ? gene : parent2.weights[i]
      );
      break;

    case CrossoverType.SINGLE_POINT:
      // Single crossover point
      const crossPoint = Math.floor(Math.random() * length);
      child.weights = [
        ...parent1.weights.slice(0, crossPoint),
        ...parent2.weights.slice(crossPoint),
      ];
      break;

    case CrossoverType.MULTI_POINT:
      // Multiple crossover points (2-4 points)
      const numPoints = Math.floor(Math.random() * 3) + 2;
      const points = Array(numPoints)
        .fill(0)
        .map(() => Math.floor(Math.random() * length))
        .sort((a, b) => a - b);

      // Start with first parent
      child.weights = [...parent1.weights];
      let currentParent = 2; // Switch to second parent

      // Apply crossover at each point
      points.forEach((point) => {
        if (currentParent === 2) {
          // Switch to parent2's genes
          for (let i = point; i < length; i++) {
            child.weights[i] = parent2.weights[i];
          }
          currentParent = 1;
        } else {
          // Switch to parent1's genes
          for (let i = point; i < length; i++) {
            child.weights[i] = parent1.weights[i];
          }
          currentParent = 2;
        }
      });
      break;
  }

  return child;
}
