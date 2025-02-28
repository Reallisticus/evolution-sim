// src/evolution/lineage.ts
export interface AgentAncestry {
  id: number;
  parentIds: number[];
  generation: number;
  speciesId: number;
  birthTick: number;
  deathTick: number | null;
  maxFitness: number;
}

export class LineageTracker {
  private ancestryRecords: Map<number, AgentAncestry> = new Map();
  private nextAgentId: number = 1;
  private currentGeneration: number = 1;
  private currentTick: number = 0;

  constructor() {}

  public registerBirth(parentIds: number[], speciesId: number): number {
    const id = this.nextAgentId++;

    this.ancestryRecords.set(id, {
      id,
      parentIds,
      generation: this.currentGeneration,
      speciesId,
      birthTick: this.currentTick,
      deathTick: null,
      maxFitness: 0,
    });

    return id;
  }

  public registerDeath(agentId: number, finalFitness: number): void {
    const record = this.ancestryRecords.get(agentId);
    if (record) {
      record.deathTick = this.currentTick;
      record.maxFitness = Math.max(record.maxFitness, finalFitness);
    }
  }

  public updateFitness(agentId: number, fitness: number): void {
    const record = this.ancestryRecords.get(agentId);
    if (record) {
      record.maxFitness = Math.max(record.maxFitness, fitness);
    }
  }

  public advanceGeneration(): void {
    this.currentGeneration++;
  }

  public updateTick(tick: number): void {
    this.currentTick = tick;
  }

  public getAncestryRecord(agentId: number): AgentAncestry | undefined {
    return this.ancestryRecords.get(agentId);
  }

  public getAllRecords(): any[] {
    return Array.from(this.ancestryRecords.values());
  }

  public getGenerationRecords(generation: number): AgentAncestry[] {
    return Array.from(this.ancestryRecords.values()).filter(
      (record) => record.generation === generation
    );
  }

  // Get all ancestors of an agent
  public getAncestors(agentId: number): AgentAncestry[] {
    const ancestors: AgentAncestry[] = [];
    const record = this.ancestryRecords.get(agentId);

    if (!record) return ancestors;

    // Queue for breadth-first search
    const queue: number[] = [...record.parentIds];

    while (queue.length > 0) {
      const parentId = queue.shift()!;
      const parentRecord = this.ancestryRecords.get(parentId);

      if (parentRecord) {
        ancestors.push(parentRecord);
        queue.push(...parentRecord.parentIds);
      }
    }

    return ancestors;
  }

  public clearAllRecords(): void {
    this.ancestryRecords.clear();
  }

  public addRecordDirect(record: any): void {
    this.ancestryRecords.set(record.id, record);
  }
}
