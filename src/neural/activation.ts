// src/neural/activation.ts
export enum ActivationFunction {
  SIGMOID,
  TANH,
  RELU,
  LEAKY_RELU,
}

export function activate(
  value: number,
  activationFunction: ActivationFunction
): number {
  switch (activationFunction) {
    case ActivationFunction.SIGMOID:
      return 1 / (1 + Math.exp(-value));
    case ActivationFunction.TANH:
      return Math.tanh(value);
    case ActivationFunction.RELU:
      return Math.max(0, value);
    case ActivationFunction.LEAKY_RELU:
      return value > 0 ? value : 0.01 * value;
    default:
      return value;
  }
}
