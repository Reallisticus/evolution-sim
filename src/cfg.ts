// src/config.ts
export default {
  // Simulation settings
  simulationSpeed: 1,
  tickRate: 60, // Ticks per second

  // World settings
  worldWidth: 800,
  worldHeight: 600,
  initialAgentCount: 50,
  initialFoodCount: 100,

  // Agent settings
  agentSize: 10,
  agentMaxSpeed: 2,
  agentMaxForce: 0.1,
  agentMaxEnergy: 100,
  agentMetabolismRate: 0.1, // Energy consumed per tick
  agentReproductionEnergy: 70, // Energy threshold for reproduction

  inputNeurons: 12, // Updated from 8 to match our new sensory inputs
  hiddenNeurons: [16, 12], // Enhanced with two layers for more complex behaviors
  outputNeurons: 3, // Movement outputs

  // Evolution settings
  mutationRate: 0.05,
  mutationAmount: 0.1,

  // Visualization
  renderScale: 1,
  agentColor: 0x00ff00,
  foodColor: 0x0000ff,

  // GPU acceleration
  useGPU: true,
  batchSize: 64, // For batch processing on GPU
};
