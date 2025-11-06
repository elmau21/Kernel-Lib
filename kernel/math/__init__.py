"""
Módulo de algoritmos matemáticos avanzados.

Implementa algoritmos desde cero sin usar librerías de deep learning.
"""

from kernel.math.activations import (
    ActivationFunction, Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softmax, Linear,
    LossFunction, MeanSquaredError, CrossEntropy, BinaryCrossEntropy
)

from kernel.math.neural_network import Layer, NeuralNetwork

from kernel.math.optimizers import (
    Optimizer, SGD, RMSprop, Adam, AdaGrad, Nesterov
)

from kernel.math.optimization import (
    GradientDescent, NewtonMethod, ConjugateGradient, BFGS
)

__all__ = [
    # Activations
    "ActivationFunction", "Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", 
    "Softmax", "Linear",
    "LossFunction", "MeanSquaredError", "CrossEntropy", "BinaryCrossEntropy",
    # Neural Networks
    "Layer", "NeuralNetwork",
    # Optimizers
    "Optimizer", "SGD", "RMSprop", "Adam", "AdaGrad", "Nesterov",
    # Optimization
    "GradientDescent", "NewtonMethod", "ConjugateGradient", "BFGS",
]

