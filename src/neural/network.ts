// src/neural/network.ts (fixed version)
import * as tf from '@tensorflow/tfjs';
import config from '../cfg';

export class NeuralNetwork {
  private model: tf.Sequential;

  constructor(inputSize: number, hiddenSizes: number[], outputSize: number) {
    // Set up TensorFlow.js to use GPU if available
    if (config.useGPU) {
      tf.setBackend('webgl');
    }

    // Create sequential model (use Sequential type instead of LayersModel)
    this.model = tf.sequential();

    // Add input layer
    this.model.add(
      tf.layers.dense({
        units: hiddenSizes[0],
        inputShape: [inputSize],
        activation: 'relu',
      })
    );

    // Add hidden layers
    for (let i = 1; i < hiddenSizes.length; i++) {
      this.model.add(
        tf.layers.dense({
          units: hiddenSizes[i],
          activation: 'relu',
        })
      );
    }

    // Add output layer
    this.model.add(
      tf.layers.dense({
        units: outputSize,
        activation: 'sigmoid', // Outputs between 0 and 1
      })
    );

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(),
      loss: 'meanSquaredError',
    });
  }

  public predict(inputs: number[]): number[] {
    // Convert inputs to tensor
    const inputTensor = tf.tensor2d([inputs]);

    // Run prediction
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;

    // Convert back to array
    const outputs = outputTensor.dataSync();

    // Clean up tensors to prevent memory leaks
    inputTensor.dispose();
    outputTensor.dispose();

    return Array.from(outputs);
  }

  public copy(): NeuralNetwork {
    const inputSize = this.model.inputs[0].shape[1] as number;
    const outputSize = this.model.outputs[0].shape[1] as number;
    const hiddenSizes: number[] = [];

    // Get hidden layer sizes (fix the units property access)
    for (let i = 1; i < this.model.layers.length - 1; i++) {
      // Use config.units to get the units safely
      const layerConfig = this.model.layers[i].getConfig() as any;
      hiddenSizes.push(layerConfig.units);
    }

    // Create new network with same architecture
    const newNetwork = new NeuralNetwork(inputSize, hiddenSizes, outputSize);

    // Copy weights
    for (let i = 0; i < this.model.layers.length; i++) {
      const weights = this.model.layers[i].getWeights();
      newNetwork.model.layers[i].setWeights(weights);
    }

    return newNetwork;
  }

  public dispose(): void {
    this.model.dispose();
  }

  // Get weights as a flat array (for genome representation)
  public getWeights(): number[] {
    const weights: number[] = [];

    // Loop through each layer
    for (const layer of this.model.layers) {
      const layerWeights = layer.getWeights();

      // Loop through each weight tensor in the layer
      for (const weightTensor of layerWeights) {
        // Convert tensor to array and add to flat array
        const weightArray = weightTensor.dataSync();
        weights.push(...Array.from(weightArray));
      }
    }

    return weights;
  }

  // Set weights from a flat array
  public setWeights(weights: number[]): void {
    let weightIndex = 0;

    // Loop through each layer
    for (const layer of this.model.layers) {
      const layerWeights = layer.getWeights();
      const newLayerWeights = [];

      // Loop through each weight tensor in the layer
      for (const weightTensor of layerWeights) {
        const shape = weightTensor.shape;
        const length = weightTensor.size;

        // Extract weights for this tensor from flat array
        const tensorWeights = weights.slice(weightIndex, weightIndex + length);
        weightIndex += length;

        // Create new tensor with the extracted weights
        const newWeightTensor = tf.tensor(tensorWeights, shape);
        newLayerWeights.push(newWeightTensor);
      }

      // Set the new weights for the layer
      layer.setWeights(newLayerWeights);
    }
  }
}
