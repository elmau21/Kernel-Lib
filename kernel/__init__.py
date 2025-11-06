"""
Kernel - Motor de Métodos de Kernel para Machine Learning e Ingeniería

Una biblioteca de alto rendimiento para métodos de kernel en machine learning.
"""

__version__ = "1.0.0"
__author__ = "Kernel Team"

# Core
from kernel.core.kernel_base import KernelBase, KernelCache

# Kernels
from kernel.kernels.rbf import RBFKernel
from kernel.kernels.polynomial import PolynomialKernel
from kernel.kernels.linear import LinearKernel
from kernel.kernels.matern import MaternKernel
from kernel.kernels.laplacian import LaplacianKernel
from kernel.kernels.composite import CompositeKernel, ScaledKernel, TransformedKernel

# Methods
from kernel.methods.svm import KernelSVM
from kernel.methods.kpca import KernelPCA
from kernel.methods.gaussian_process import GaussianProcess

# Math (Algoritmos matemáticos avanzados)
from kernel.math import (
    NeuralNetwork, Layer,
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softmax, Linear,
    Softplus, Swish, HardSigmoid, HardSwish, GELU, PReLU, SELU, Mish,
    MeanSquaredError, CrossEntropy, BinaryCrossEntropy,
    SGD, Momentum, Adam, RMSprop, AdaGrad, Nesterov,
    AdamW, Nadam, RAdam, AdaBelief, Lion, Ranger,
    GradientDescent, NewtonMethod, ConjugateGradient, BFGS
)

__all__ = [
    # Core
    "KernelBase",
    "KernelCache",
    # Kernels
    "RBFKernel",
    "PolynomialKernel",
    "LinearKernel",
    "MaternKernel",
    "LaplacianKernel",
    "CompositeKernel",
    "ScaledKernel",
    "TransformedKernel",
    # Methods
    "KernelSVM",
    "KernelPCA",
    "GaussianProcess",
    # Math
    "NeuralNetwork", "Layer",
    "Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", "Softmax", "Linear",
    "Softplus", "Swish", "HardSigmoid", "HardSwish", "GELU", "PReLU", "SELU", "Mish",
    "MeanSquaredError", "CrossEntropy", "BinaryCrossEntropy",
    "SGD", "Momentum", "Adam", "RMSprop", "AdaGrad", "Nesterov",
    "AdamW", "Nadam", "RAdam", "AdaBelief", "Lion", "Ranger",
    "GradientDescent", "NewtonMethod", "ConjugateGradient", "BFGS",
]

