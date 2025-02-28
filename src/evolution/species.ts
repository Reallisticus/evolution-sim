// src/evolution/species.ts
import { Genome } from './genome';

export class Species {
  public id: number;
  public name: string;
  public members: number[] = []; // Agent IDs
  public representative: Genome;
  public color: string;
  public creationGeneration: number;
  public lastImprovedGeneration: number;
  public bestFitness: number = 0;

  constructor(
    id: number,
    representative: Genome,
    creationGeneration: number,
    firstMemberId: number
  ) {
    this.id = id;
    this.representative = representative.copy();
    this.creationGeneration = creationGeneration;
    this.lastImprovedGeneration = creationGeneration;
    this.members.push(firstMemberId);

    // Generate a name and color
    this.name = `Species-${id}`;

    // Random HSL color (more visually distinct than RGB)
    const hue = Math.floor(Math.random() * 360);
    this.color = `hsl(${hue}, 70%, 60%)`;
  }

  public addMember(agentId: number): void {
    if (!this.members.includes(agentId)) {
      this.members.push(agentId);
    }
  }

  public updateFitness(fitness: number): void {
    if (fitness > this.bestFitness) {
      this.bestFitness = fitness;
    }
  }
}

// Calculate genetic distance between genomes
export function geneticDistance(genomeA: Genome, genomeB: Genome): number {
  if (genomeA.weights.length !== genomeB.weights.length) {
    throw new Error('Genomes must have same length for distance calculation');
  }

  // Calculate Euclidean distance between weight vectors
  let sumSquaredDiff = 0;
  for (let i = 0; i < genomeA.weights.length; i++) {
    const diff = genomeA.weights[i] - genomeB.weights[i];
    sumSquaredDiff += diff * diff;
  }

  return Math.sqrt(sumSquaredDiff);
}

export class SpeciesManager {
  private species: Species[] = [];
  private nextSpeciesId: number = 1;
  private distanceThreshold: number = 4.0; // Adjust this to control species formation
  private currentGeneration: number = 1;

  constructor() {}

  // Find species for a genome, or create a new one if it doesn't match any existing species
  public assignSpecies(genome: Genome, agentId: number): Species {
    let species: Species | null = null;

    // Try to find a matching species
    for (const existingSpecies of this.species) {
      const distance = geneticDistance(genome, existingSpecies.representative);

      if (distance < this.distanceThreshold) {
        species = existingSpecies;
        break;
      }
    }

    // Create new species if no match found
    if (!species) {
      species = new Species(
        this.nextSpeciesId++,
        genome,
        this.currentGeneration,
        agentId
      );
      this.species.push(species);
    } else {
      species.addMember(agentId);
    }

    return species;
  }

  public advanceGeneration(): void {
    this.currentGeneration++;

    // Reset members for each species
    for (const species of this.species) {
      species.members = [];
    }
  }

  public getSpecies(): Species[] {
    return this.species;
  }

  public getCurrentGeneration(): number {
    return this.currentGeneration;
  }

  // Get active species (those with members)
  public getActiveSpecies(): Species[] {
    return this.species.filter((s) => s.members.length > 0);
  }

  public clearAllSpecies(): void {
    this.species = [];
  }

  public createSpecies(
    id: number,
    representative: Genome,
    creationGeneration: number
  ): Species {
    const species = new Species(id, representative, creationGeneration, 0);
    this.species.push(species);
    return species;
  }

  public getSpeciesById(id: number): Species | null {
    return this.species.find((s) => s.id === id) || null;
  }
}
