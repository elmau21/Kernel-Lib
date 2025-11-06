"""
Kernel RBF (Radial Basis Function) - También conocido como Gaussian Kernel.

K(x, y) = exp(-gamma * ||x - y||²)

Implementación avanzada con optimizaciones numéricas.
"""

import numpy as np
from typing import Optional
from kernel.core.kernel_base import KernelBase


class RBFKernel(KernelBase):
    """
    Kernel RBF (Radial Basis Function) con optimizaciones avanzadas.
    
    K(x, y) = exp(-gamma * ||x - y||²)
    
    donde gamma = 1 / (2 * sigma²) es el parámetro de escala.
    
    Propiedades matemáticas:
    - Universal: puede aproximar cualquier función continua (RKHS denso)
    - Infinitamente diferenciable (C^∞)
    - Parámetro gamma controla el ancho del kernel (bandwidth)
    - Mercer kernel: garantiza existencia de espacio de características
    
    Optimizaciones:
    - Cálculo eficiente usando broadcasting vectorizado
    - Soporte para GPU (CuPy)
    - Caching inteligente
    - Estabilidad numérica mejorada
    """
    
    def __init__(self, gamma: float = 1.0, sigma: Optional[float] = None,
                 use_cache: bool = True, enable_gpu: bool = False,
                 numerical_stability: float = 1e-12):
        """
        Inicializa el kernel RBF.
        
        Parámetros:
        -----------
        gamma : float, default=1.0
            Parámetro de escala. Si se proporciona sigma, gamma se calcula como
            1 / (2 * sigma²).
        sigma : float, opcional
            Desviación estándar. Si se proporciona, sobrescribe gamma.
        use_cache : bool, default=True
            Habilita caching de matrices de kernel.
        enable_gpu : bool, default=False
            Habilita cálculos en GPU.
        numerical_stability : float, default=1e-12
            Tolerancia para estabilidad numérica.
        """
        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma debe ser positivo")
            gamma = 1.0 / (2.0 * sigma ** 2)
        
        if gamma <= 0:
            raise ValueError("gamma debe ser positivo")
        
        super().__init__(gamma=gamma, use_cache=use_cache, enable_gpu=enable_gpu,
                        numerical_stability=numerical_stability)
        self.gamma = gamma
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de kernel RBF.
        
        Implementación optimizada usando identidad:
        ||x - y||² = ||x||² + ||y||² - 2*x·y
        
        Complejidad: O(n*m*d) donde n=|X|, m=|Y|, d=dimensiones
        """
        # Calcula ||x - y||² de forma eficiente usando broadcasting
        # Evita crear matriz completa de diferencias
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
        XY = np.dot(X, Y.T)
        
        squared_distances = X_norm + Y_norm.T - 2 * XY
        
        # Evita underflow numérico
        squared_distances = np.maximum(squared_distances, 0.0)
        
        # Aplica el kernel: exp(-gamma * ||x - y||²)
        K = np.exp(-self.gamma * squared_distances)
        
        return K
    
    def __repr__(self) -> str:
        return f"RBFKernel(gamma={self.gamma:.4f})"

