// src/analytics/history-store.ts

import { SimulationStats } from './data-types.ts';

export class HistoryStore {
  private readonly HISTORY_KEY = 'evolution-sim-history';
  private readonly MAX_SESSIONS = 50;
  private readonly MAX_GENERATIONS_PER_SESSION = 1000;

  private sessionId: string;
  private sessionStats: Map<number, SimulationStats> = new Map();
  private sessionsIndex: string[] = [];

  constructor() {
    // Generate unique session ID
    this.sessionId =
      new Date().toISOString() +
      '-' +
      Math.random().toString(36).substring(2, 9);
    this.loadSessionsIndex();
    this.addSessionToIndex();
  }

  private loadSessionsIndex(): void {
    const index = localStorage.getItem('evolution-sim-sessions');
    if (index) {
      this.sessionsIndex = JSON.parse(index);
    }
  }

  private saveSessionsIndex(): void {
    localStorage.setItem(
      'evolution-sim-sessions',
      JSON.stringify(this.sessionsIndex)
    );
  }

  private addSessionToIndex(): void {
    this.sessionsIndex.push(this.sessionId);
    if (this.sessionsIndex.length > this.MAX_SESSIONS) {
      // Remove oldest session data
      const oldestSession = this.sessionsIndex.shift();
      if (oldestSession) {
        localStorage.removeItem(`${this.HISTORY_KEY}-${oldestSession}`);
      }
    }
    this.saveSessionsIndex();
  }

  public addGenerationStats(generation: number, stats: SimulationStats): void {
    this.sessionStats.set(generation, { ...stats, timestamp: Date.now() });

    // Save to localStorage periodically (every 5 generations)
    if (generation % 5 === 0) {
      this.saveToLocalStorage();
    }
  }

  public saveToLocalStorage(): void {
    try {
      const sessionData = Array.from(this.sessionStats.entries());

      // Break into chunks if needed (localStorage has size limits)
      const chunkSize = 50; // generations per chunk
      const chunks = Math.ceil(sessionData.length / chunkSize);

      for (let i = 0; i < chunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, sessionData.length);
        const chunk = sessionData.slice(start, end);
        localStorage.setItem(
          `${this.HISTORY_KEY}-${this.sessionId}-chunk-${i}`,
          JSON.stringify(chunk)
        );
      }

      // Store metadata
      localStorage.setItem(
        `${this.HISTORY_KEY}-${this.sessionId}-meta`,
        JSON.stringify({
          chunks,
          lastGeneration: Math.max(...this.sessionStats.keys()),
          startTime: sessionData[0]?.[1].timestamp,
          endTime: Date.now(),
        })
      );
    } catch (error) {
      console.error('Error saving history to localStorage:', error);
    }
  }

  public loadFromLocalStorage(sessionId?: string): SimulationStats[] {
    const targetSession = sessionId || this.sessionId;
    const result: SimulationStats[] = [];

    try {
      // Load metadata
      const metaString = localStorage.getItem(
        `${this.HISTORY_KEY}-${targetSession}-meta`
      );
      if (!metaString) return result;

      const meta = JSON.parse(metaString);

      // Load all chunks
      for (let i = 0; i < meta.chunks; i++) {
        const chunkString = localStorage.getItem(
          `${this.HISTORY_KEY}-${targetSession}-chunk-${i}`
        );
        if (chunkString) {
          const chunk = JSON.parse(chunkString);
          for (const [gen, stats] of chunk) {
            result.push({ ...stats, generation: Number(gen) });
          }
        }
      }

      // Sort by generation
      result.sort((a, b) => a.generation - b.generation);
    } catch (error) {
      console.error('Error loading history from localStorage:', error);
    }

    return result;
  }

  public loadAllSessionsHistory(): Map<string, SimulationStats[]> {
    const allHistory = new Map<string, SimulationStats[]>();

    for (const sessionId of this.sessionsIndex) {
      const sessionStats = this.loadFromLocalStorage(sessionId);
      if (sessionStats.length > 0) {
        allHistory.set(sessionId, sessionStats);
      }
    }

    return allHistory;
  }

  public exportHistory(): any {
    return {
      sessionId: this.sessionId,
      stats: Array.from(this.sessionStats.entries()),
      exportTime: Date.now(),
    };
  }

  public importHistory(data: any): boolean {
    try {
      if (!data.sessionId || !Array.isArray(data.stats)) {
        return false;
      }

      this.sessionId = data.sessionId;
      this.sessionStats = new Map(data.stats);
      this.saveToLocalStorage();

      // Add to index if not exists
      if (!this.sessionsIndex.includes(this.sessionId)) {
        this.addSessionToIndex();
      }

      return true;
    } catch (error) {
      console.error('Error importing history:', error);
      return false;
    }
  }

  public getStats(generation: number): SimulationStats | undefined {
    return this.sessionStats.get(generation);
  }

  public getAllStats(): SimulationStats[] {
    return Array.from(this.sessionStats.values()).sort(
      (a, b) => a.generation - b.generation
    );
  }

  public getLatestGenerations(count: number): SimulationStats[] {
    const allStats = this.getAllStats();
    return allStats.slice(Math.max(0, allStats.length - count));
  }
}
