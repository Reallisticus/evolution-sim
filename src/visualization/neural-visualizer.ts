// src/visualization/neural-visualizer.ts
import * as tf from '@tensorflow/tfjs';
import { NeuralNetwork } from '../neural/network';
import { Agent } from '../entities/agent';
import config from '../cfg';

export class NeuralVisualizer {
  private canvasWidth: number = 800;
  private canvasHeight: number = 600;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private neuronRadius: number = 15;
  private layerSpacing: number = 120;

  constructor(
    container: HTMLElement,
    width: number = 800,
    height: number = 600
  ) {
    this.canvasWidth = width;
    this.canvasHeight = height;

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvasWidth;
    this.canvas.height = this.canvasHeight;
    container.appendChild(this.canvas);

    // Get context
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    this.ctx = ctx;
  }

  public visualizeAgent(agent: Agent): void {
    this.clearCanvas();
    this.drawNetworkStructure(agent);
    this.drawWeights(agent);
    this.drawActivations(agent);
    this.drawLegend();
  }

  public compareTwoAgents(agent1: Agent, agent2: Agent): void {
    this.clearCanvas();

    // Split canvas in half
    this.ctx.save();
    this.ctx.translate(0, 0);
    this.ctx.scale(0.5, 1);
    this.drawNetworkStructure(agent1, true);
    this.drawWeightsDiff(agent1, agent2);
    this.ctx.restore();

    this.ctx.save();
    this.ctx.translate(this.canvasWidth / 2, 0);
    this.ctx.scale(0.5, 1);
    this.drawNetworkStructure(agent2, true);
    this.ctx.restore();

    // Draw labels
    this.ctx.font = '16px Arial';
    this.ctx.fillStyle = '#fff';
    this.ctx.fillText(
      `Agent ${agent1.id} (Fitness: ${agent1.fitness.toFixed(1)})`,
      10,
      20
    );
    this.ctx.fillText(
      `Agent ${agent2.id} (Fitness: ${agent2.fitness.toFixed(1)})`,
      this.canvasWidth / 2 + 10,
      20
    );

    // Draw difference legend
    this.drawDiffLegend();
  }

  public visualizeSpecies(agents: Agent[], maxAgents: number = 5): void {
    this.clearCanvas();

    // Group agents by species
    const speciesMap = new Map<number, Agent[]>();
    for (const agent of agents) {
      if (!agent.species) continue;

      const speciesId = agent.species.id;
      if (!speciesMap.has(speciesId)) {
        speciesMap.set(speciesId, []);
      }
      speciesMap.get(speciesId)!.push(agent);
    }

    // Sort species by size
    const sortedSpecies = Array.from(speciesMap.entries())
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 3); // Show top 3 species

    if (sortedSpecies.length === 0) {
      this.drawText('No species data available', 20, 30);
      return;
    }

    // Draw representative networks for each species
    const speciesHeight = this.canvasHeight / sortedSpecies.length;

    sortedSpecies.forEach(([speciesId, speciesAgents], index) => {
      // Sort by fitness and get top agent
      const topAgent = [...speciesAgents].sort(
        (a, b) => b.fitness - a.fitness
      )[0];

      // Draw in assigned section
      this.ctx.save();
      this.ctx.translate(0, index * speciesHeight);
      this.ctx.scale(1, speciesHeight / this.canvasHeight);

      this.drawNetworkStructure(topAgent, true);
      this.drawWeights(topAgent);

      // Label
      this.ctx.font = '16px Arial';
      this.ctx.fillStyle = topAgent.species!.color;
      this.ctx.fillText(
        `Species ${topAgent.species!.name} (${speciesAgents.length} members)`,
        10,
        20
      );

      this.ctx.restore();
    });
  }

  private clearCanvas(): void {
    this.ctx.fillStyle = '#1a1a2e';
    this.ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);
  }

  private drawNetworkStructure(agent: Agent, compact: boolean = false): void {
    const layerSizes = [
      config.inputNeurons,
      ...config.hiddenNeurons,
      config.outputNeurons,
    ];

    const layerCount = layerSizes.length;
    const layerSpacing = compact ? this.layerSpacing * 0.8 : this.layerSpacing;
    const neuronRadius = compact ? this.neuronRadius * 0.8 : this.neuronRadius;

    // Calculate positions for each layer
    const layerPositions: Array<Array<{ x: number; y: number }>> = [];

    for (let l = 0; l < layerCount; l++) {
      const layerSize = layerSizes[l];
      const layer: Array<{ x: number; y: number }> = [];

      // Calculate x-position for this layer
      const x = 50 + l * layerSpacing;

      // Calculate y-positions for neurons in this layer
      const totalHeight = (layerSize - 1) * neuronRadius * 3;
      const startY = (this.canvasHeight - totalHeight) / 2;

      for (let n = 0; n < layerSize; n++) {
        const y = startY + n * neuronRadius * 3;
        layer.push({ x, y });
      }

      layerPositions.push(layer);
    }

    // Draw connections
    this.ctx.strokeStyle = 'rgba(150, 150, 200, 0.2)';
    this.ctx.lineWidth = 1;

    for (let l = 0; l < layerCount - 1; l++) {
      const currentLayer = layerPositions[l];
      const nextLayer = layerPositions[l + 1];

      for (let i = 0; i < currentLayer.length; i++) {
        for (let j = 0; j < nextLayer.length; j++) {
          this.ctx.beginPath();
          this.ctx.moveTo(currentLayer[i].x, currentLayer[i].y);
          this.ctx.lineTo(nextLayer[j].x, nextLayer[j].y);
          this.ctx.stroke();
        }
      }
    }

    // Draw neurons
    for (let l = 0; l < layerCount; l++) {
      const layer = layerPositions[l];

      for (let n = 0; n < layer.length; n++) {
        const { x, y } = layer[n];

        // Draw neuron circle
        this.ctx.beginPath();
        this.ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);

        // Different colors for different layer types
        if (l === 0) {
          this.ctx.fillStyle = '#4488ff'; // Input layer
        } else if (l === layerCount - 1) {
          // Output layer colors by function
          const colors = ['#44ff88', '#ff8844', '#ff44aa'];
          this.ctx.fillStyle = colors[n % colors.length];
        } else {
          this.ctx.fillStyle = '#aaaaff'; // Hidden layer
        }

        this.ctx.fill();

        // Draw border
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();

        // Draw neuron label if not compact mode
        if (!compact) {
          this.ctx.fillStyle = '#ffffff';
          this.ctx.font = '10px Arial';
          this.ctx.textAlign = 'center';

          let label = '';
          if (l === 0) {
            label = `I${n}`;
          } else if (l === layerCount - 1) {
            const outputLabels = ['FWD', 'TURN', 'SPD'];
            label = outputLabels[n];
          } else {
            label = `H${l}-${n}`;
          }

          this.ctx.fillText(label, x, y + 3);
        }
      }
    }

    // Store positions for later use
    (this as any).neuronPositions = layerPositions;
  }

  private drawWeights(agent: Agent): void {
    const neuronPositions = (this as any).neuronPositions;
    if (!neuronPositions) return;

    // Get weights from agent's neural network
    const weights = agent.genome.weights;

    // We need to extract the weight matrices from the flattened array
    // This is complex and requires knowing the exact structure of weights in the network
    // For demonstration, we'll use a simplified approach

    // Calculate weight indexes for each layer pair
    const layerSizes = [
      config.inputNeurons,
      ...config.hiddenNeurons,
      config.outputNeurons,
    ];

    let weightIndex = 0;

    // For each layer pair (e.g., input->hidden1, hidden1->hidden2, etc.)
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const currentLayerSize = layerSizes[l];
      const nextLayerSize = layerSizes[l + 1];

      // For each connection between these layers
      for (let i = 0; i < currentLayerSize; i++) {
        for (let j = 0; j < nextLayerSize; j++) {
          const weight = weights[weightIndex++];

          // Skip bias weights for simplicity

          // Draw weight as line color and thickness
          this.ctx.beginPath();
          this.ctx.moveTo(neuronPositions[l][i].x, neuronPositions[l][i].y);
          this.ctx.lineTo(
            neuronPositions[l + 1][j].x,
            neuronPositions[l + 1][j].y
          );

          // Line width based on weight magnitude
          const magnitude = Math.abs(weight);
          this.ctx.lineWidth = Math.min(5, magnitude * 3);

          // Color based on sign
          if (weight > 0) {
            // Green for excitatory connections
            this.ctx.strokeStyle = `rgba(0, 255, 0, ${Math.min(1, magnitude)})`;
          } else {
            // Red for inhibitory connections
            this.ctx.strokeStyle = `rgba(255, 0, 0, ${Math.min(1, magnitude)})`;
          }

          this.ctx.stroke();
        }
      }

      // Skip bias weights for simplicity
      weightIndex += nextLayerSize;
    }
  }

  private drawWeightsDiff(agent1: Agent, agent2: Agent): void {
    const neuronPositions = (this as any).neuronPositions;
    if (!neuronPositions) return;

    // Get weights from both agents
    const weights1 = agent1.genome.weights;
    const weights2 = agent2.genome.weights;

    // Calculate layer sizes and indexes
    const layerSizes = [
      config.inputNeurons,
      ...config.hiddenNeurons,
      config.outputNeurons,
    ];

    let weightIndex = 0;

    // For each layer pair
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const currentLayerSize = layerSizes[l];
      const nextLayerSize = layerSizes[l + 1];

      // For each connection between these layers
      for (let i = 0; i < currentLayerSize; i++) {
        for (let j = 0; j < nextLayerSize; j++) {
          const weight1 = weights1[weightIndex];
          const weight2 = weights2[weightIndex];
          weightIndex++;

          // Calculate difference
          const diff = weight1 - weight2;

          // Draw connection with difference
          this.ctx.beginPath();
          this.ctx.moveTo(neuronPositions[l][i].x, neuronPositions[l][i].y);
          this.ctx.lineTo(
            neuronPositions[l + 1][j].x,
            neuronPositions[l + 1][j].y
          );

          // Line width based on difference magnitude
          const magnitude = Math.abs(diff);
          this.ctx.lineWidth = Math.min(5, magnitude * 3);

          // Color intensity based on magnitude
          const intensity = Math.min(1, magnitude);

          // Blue for agent1 > agent2, Purple for agent2 > agent1
          if (diff > 0) {
            this.ctx.strokeStyle = `rgba(0, 128, 255, ${intensity})`;
          } else if (diff < 0) {
            this.ctx.strokeStyle = `rgba(255, 0, 255, ${intensity})`;
          } else {
            this.ctx.strokeStyle = `rgba(128, 128, 128, 0.2)`;
          }

          this.ctx.stroke();
        }
      }

      // Skip bias weights
      weightIndex += nextLayerSize;
    }
  }

  private drawActivations(agent: Agent): void {
    // This would require modifications to the Agent class to track activations
    // We'll implement a placeholder for demonstration

    const neuronPositions = (this as any).neuronPositions;
    if (!neuronPositions) return;

    // Simulate activations for demonstration
    const inputActivations = Array(config.inputNeurons)
      .fill(0)
      .map(() => Math.random());

    const outputActivations = Array(config.outputNeurons)
      .fill(0)
      .map(() => Math.random());

    // Draw input activations
    for (let i = 0; i < config.inputNeurons; i++) {
      const { x, y } = neuronPositions[0][i];
      const activation = inputActivations[i];

      // Draw activation ring
      this.ctx.beginPath();
      this.ctx.arc(x, y, this.neuronRadius * 1.3, 0, Math.PI * 2);
      this.ctx.strokeStyle = `rgba(255, 255, 255, ${activation})`;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }

    // Draw output activations
    for (let i = 0; i < config.outputNeurons; i++) {
      const { x, y } = neuronPositions[neuronPositions.length - 1][i];
      const activation = outputActivations[i];

      // Draw activation ring
      this.ctx.beginPath();
      this.ctx.arc(x, y, this.neuronRadius * 1.3, 0, Math.PI * 2);
      this.ctx.strokeStyle = `rgba(255, 255, 255, ${activation})`;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Draw activation value
      this.ctx.fillStyle = '#ffffff';
      this.ctx.font = '12px Arial';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(activation.toFixed(2), x + this.neuronRadius * 1.5, y);
    }
  }

  private drawLegend(): void {
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(this.canvasWidth - 200, 10, 190, 140);

    this.ctx.font = '14px Arial';
    this.ctx.fillStyle = '#ffffff';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Connection Weight', this.canvasWidth - 190, 30);

    // Positive weight example
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 50);
    this.ctx.lineTo(this.canvasWidth - 140, 50);
    this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    this.ctx.lineWidth = 3;
    this.ctx.stroke();
    this.ctx.fillText('Positive', this.canvasWidth - 130, 55);

    // Negative weight example
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 75);
    this.ctx.lineTo(this.canvasWidth - 140, 75);
    this.ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    this.ctx.lineWidth = 3;
    this.ctx.stroke();
    this.ctx.fillText('Negative', this.canvasWidth - 130, 80);

    // Weight magnitude example
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 100);
    this.ctx.lineTo(this.canvasWidth - 140, 100);
    this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();
    this.ctx.fillText('Low Magnitude', this.canvasWidth - 130, 105);

    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 125);
    this.ctx.lineTo(this.canvasWidth - 140, 125);
    this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    this.ctx.lineWidth = 5;
    this.ctx.stroke();
    this.ctx.fillText('High Magnitude', this.canvasWidth - 130, 130);
  }

  private drawDiffLegend(): void {
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(this.canvasWidth - 200, 10, 190, 120);

    this.ctx.font = '14px Arial';
    this.ctx.fillStyle = '#ffffff';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Weight Difference', this.canvasWidth - 190, 30);

    // Agent 1 higher
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 50);
    this.ctx.lineTo(this.canvasWidth - 140, 50);
    this.ctx.strokeStyle = 'rgba(0, 128, 255, 0.8)';
    this.ctx.lineWidth = 3;
    this.ctx.stroke();
    this.ctx.fillText('Left Agent Higher', this.canvasWidth - 130, 55);

    // Agent 2 higher
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 75);
    this.ctx.lineTo(this.canvasWidth - 140, 75);
    this.ctx.strokeStyle = 'rgba(255, 0, 255, 0.8)';
    this.ctx.lineWidth = 3;
    this.ctx.stroke();
    this.ctx.fillText('Right Agent Higher', this.canvasWidth - 130, 80);

    // No difference
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvasWidth - 190, 100);
    this.ctx.lineTo(this.canvasWidth - 140, 100);
    this.ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)';
    this.ctx.lineWidth = 1;
    this.ctx.stroke();
    this.ctx.fillText('No Difference', this.canvasWidth - 130, 105);
  }

  private drawText(text: string, x: number, y: number): void {
    this.ctx.font = '16px Arial';
    this.ctx.fillStyle = '#ffffff';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(text, x, y);
  }

  // Additional utility methods for specialized visualizations
  public drawWeightDistribution(agent: Agent): void {
    this.clearCanvas();

    const weights = agent.genome.weights;

    // Calculate distribution
    const min = Math.min(...weights);
    const max = Math.max(...weights);
    const range = max - min;

    // Create histogram
    const bucketCount = 20;
    const buckets = Array(bucketCount).fill(0);

    weights.forEach((weight) => {
      const bucketIndex = Math.min(
        bucketCount - 1,
        Math.floor(((weight - min) / range) * bucketCount)
      );
      buckets[bucketIndex]++;
    });

    // Find maximum bucket value for scaling
    const maxBucketValue = Math.max(...buckets);

    // Draw histogram
    const barWidth = this.canvasWidth / bucketCount;
    const maxBarHeight = this.canvasHeight - 100;

    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '16px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(
      `Weight Distribution (Agent ${agent.id})`,
      this.canvasWidth / 2,
      30
    );

    // Draw x-axis
    this.ctx.beginPath();
    this.ctx.moveTo(0, this.canvasHeight - 50);
    this.ctx.lineTo(this.canvasWidth, this.canvasHeight - 50);
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.stroke();

    // Draw bars
    for (let i = 0; i < bucketCount; i++) {
      const barHeight = (buckets[i] / maxBucketValue) * maxBarHeight;
      const x = i * barWidth;
      const y = this.canvasHeight - 50 - barHeight;

      // Bar color based on weight value
      const normalizedValue = i / bucketCount;
      let color;

      if (normalizedValue < 0.5) {
        // Red for negative weights
        const intensity = 1 - normalizedValue * 2;
        color = `rgb(${Math.round(255 * intensity)}, 0, 0)`;
      } else {
        // Green for positive weights
        const intensity = (normalizedValue - 0.5) * 2;
        color = `rgb(0, ${Math.round(255 * intensity)}, 0)`;
      }

      this.ctx.fillStyle = color;
      this.ctx.fillRect(x, y, barWidth - 2, barHeight);
    }

    // Draw axis labels
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '12px Arial';
    this.ctx.textAlign = 'center';

    // x-axis labels
    this.ctx.fillText(min.toFixed(2), 10, this.canvasHeight - 30);
    this.ctx.fillText('0', this.canvasWidth / 2, this.canvasHeight - 30);
    this.ctx.fillText(
      max.toFixed(2),
      this.canvasWidth - 10,
      this.canvasHeight - 30
    );

    // Stats
    const avg = weights.reduce((sum, w) => sum + w, 0) / weights.length;
    const stdDev = Math.sqrt(
      weights.reduce((sum, w) => sum + Math.pow(w - avg, 2), 0) / weights.length
    );

    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Mean: ${avg.toFixed(4)}`, 10, 60);
    this.ctx.fillText(`Std Dev: ${stdDev.toFixed(4)}`, 10, 80);
    this.ctx.fillText(`Min: ${min.toFixed(4)}`, 10, 100);
    this.ctx.fillText(`Max: ${max.toFixed(4)}`, 10, 120);
  }

  public drawSensitivityAnalysis(agent: Agent): void {
    this.clearCanvas();

    // This would analyze how changes to each input affect outputs
    // For demonstration, we'll show a simplified visualization

    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '16px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(
      `Input Sensitivity Analysis (Agent ${agent.id})`,
      this.canvasWidth / 2,
      30
    );

    // Simulate input sensitivities
    const inputLabels = [
      'Pos X',
      'Pos Y',
      'Speed',
      'Energy',
      'Food X',
      'Food Y',
      'Food Dist',
      'Food Type',
      'Obstacle X',
      'Obstacle Y',
      'Obstacle Dist',
      'Random',
    ];

    const outputLabels = ['Forward', 'Turn', 'Speed'];

    // Simulate sensitivity matrix (would be computed by analyzing network)
    const sensitivities = Array(inputLabels.length)
      .fill(0)
      .map(() =>
        Array(outputLabels.length)
          .fill(0)
          .map(() => Math.random() * 2 - 1)
      );

    // Draw matrix
    const cellWidth = 60;
    const cellHeight = 30;
    const startX = (this.canvasWidth - outputLabels.length * cellWidth) / 2;
    const startY = 70;

    // Draw column headers (outputs)
    for (let j = 0; j < outputLabels.length; j++) {
      this.ctx.fillStyle = '#aaaaff';
      this.ctx.fillText(
        outputLabels[j],
        startX + j * cellWidth + cellWidth / 2,
        startY - 10
      );
    }

    // Draw rows
    for (let i = 0; i < inputLabels.length; i++) {
      // Row label (input)
      this.ctx.fillStyle = '#ffffff';
      this.ctx.textAlign = 'right';
      this.ctx.fillText(
        inputLabels[i],
        startX - 10,
        startY + i * cellHeight + cellHeight / 2 + 5
      );

      // Sensitivity cells
      for (let j = 0; j < outputLabels.length; j++) {
        const sensitivity = sensitivities[i][j];

        // Cell color based on sensitivity
        const intensity = Math.min(1, Math.abs(sensitivity));
        let color;

        if (sensitivity > 0) {
          // Green for positive correlation
          color = `rgba(0, ${Math.round(200 * intensity + 55)}, 0, 0.8)`;
        } else {
          // Red for negative correlation
          color = `rgba(${Math.round(200 * intensity + 55)}, 0, 0, 0.8)`;
        }

        // Draw cell
        this.ctx.fillStyle = color;
        this.ctx.fillRect(
          startX + j * cellWidth,
          startY + i * cellHeight,
          cellWidth - 2,
          cellHeight - 2
        );

        // Draw value
        this.ctx.fillStyle = '#ffffff';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
          sensitivity.toFixed(2),
          startX + j * cellWidth + cellWidth / 2,
          startY + i * cellHeight + cellHeight / 2 + 5
        );
      }
    }

    // Draw legend
    this.drawSensitivityLegend();
  }

  private drawSensitivityLegend(): void {
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(this.canvasWidth - 200, 10, 190, 120);

    this.ctx.font = '14px Arial';
    this.ctx.fillStyle = '#ffffff';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Input Sensitivity', this.canvasWidth - 190, 30);

    // Positive sensitivity
    this.ctx.fillStyle = 'rgba(0, 200, 0, 0.8)';
    this.ctx.fillRect(this.canvasWidth - 190, 40, 30, 20);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText('Positive Effect', this.canvasWidth - 150, 55);

    // Negative sensitivity
    this.ctx.fillStyle = 'rgba(200, 0, 0, 0.8)';
    this.ctx.fillRect(this.canvasWidth - 190, 70, 30, 20);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText('Negative Effect', this.canvasWidth - 150, 85);

    // Neutral
    this.ctx.fillStyle = 'rgba(100, 100, 100, 0.8)';
    this.ctx.fillRect(this.canvasWidth - 190, 100, 30, 20);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText('No Effect', this.canvasWidth - 150, 115);
  }
}
