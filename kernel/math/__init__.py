"""
Módulo de algoritmos matemáticos avanzados.

Implementa algoritmos desde cero sin usar librerías de deep learning.
"""

from kernel.math.activations import (
    ActivationFunction, 
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softmax, Linear,
    Softplus, Swish, HardSigmoid, HardSwish, GELU, PReLU, SELU, Mish,
    LossFunction, MeanSquaredError, CrossEntropy, BinaryCrossEntropy
)

from kernel.math.neural_network import Layer, NeuralNetwork

from kernel.math.optimizers import (
    Optimizer, SGD, Momentum, RMSprop, Adam, AdaGrad, Nesterov,
    AdamW, Nadam, Adadelta, Rprop, SignSGD, Yogi, Adafactor, NovoGrad,
    AdaMax, Lion, Lookahead, Lamb, QHM, Fromage, AddSign, PowerSign,
    RAdam, AdaBelief, AMSBound, AdaBound, Ranger, RangerQH, AggMo,
    AdaMod, SMORMS3, AdaShift, ExtendedRprop
)

from kernel.math.optimization import (
    GradientDescent, NewtonMethod, ConjugateGradient, BFGS
)

__all__ = [
    # Activations
    "ActivationFunction", 
    "Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", "Softmax", "Linear",
    "Softplus", "Swish", "HardSigmoid", "HardSwish", "GELU", "PReLU", "SELU", "Mish",
    "LossFunction", "MeanSquaredError", "CrossEntropy", "BinaryCrossEntropy",
    # Neural Networks
    "Layer", "NeuralNetwork",
    # Optimizers - Clásicos
    "Optimizer", "SGD", "Momentum", "RMSprop", "Adam", "AdaGrad", "Nesterov",
    # Optimizers - Avanzados
    "AdamW", "Nadam", "Adadelta", "Rprop", "SignSGD", "Yogi", "Adafactor", 
    "NovoGrad", "AdaMax", "Lion", "Lookahead", "Lamb", "QHM", "Fromage",
    "AddSign", "PowerSign", "RAdam", "AdaBelief", "AMSBound", "AdaBound",
    "Ranger", "RangerQH", "AggMo", "AdaMod", "SMORMS3", "AdaShift", "ExtendedRprop",
    # Optimization
    "GradientDescent", "NewtonMethod", "ConjugateGradient", "BFGS",
]

