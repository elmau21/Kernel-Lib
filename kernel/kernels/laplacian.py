"""
Kernel Laplaciano - También conocido como kernel exponencial.

K(x, y) = exp(-gamma * ||x - y||_1)

donde ||·||_1 es la norma L1 (distancia Manhattan).

Útil para datos con características dispersas y robusto a outliers.
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist
from kernel.core.kernel_base import KernelBase


class LaplacianKernel(KernelBase):
    """
    Kernel Laplaciano usando distancia L1.
    
    K(x, y) = exp(-gamma * ||x - y||_1)
    
    Propiedades:
    - Más robusto a outliers que RBF
    - Útil para características dispersas
    - Produce funciones menos suaves que RBF
    - Computacionalmente eficiente
    
    Casos de uso:
    - Text mining y NLP
    - Análisis de imágenes con características dispersas
    - Datos con outliers significativos
    """
    
    def __init__(self, gamma: float = 1.0, sigma: Optional[float] = None,
                 use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel Laplaciano.
        
        Parámetros:
        -----------
        gamma : float, default=1.0
            Parámetro de escala.
        sigma : float, opcional
            Parámetro alternativo. Si se proporciona, gamma = 1 / sigma.
        use_cache : bool, default=True
            Habilita caching.
        enable_gpu : bool, default=False
            Habilita GPU.
        """
        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma debe ser positivo")
            gamma = 1.0 / sigma
        
        if gamma <= 0:
            raise ValueError("gamma debe ser positivo")
        
        super().__init__(gamma=gamma, use_cache=use_cache, enable_gpu=enable_gpu)
        self.gamma = gamma
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calcula el kernel Laplaciano usando distancia L1."""
        # Distancia Manhattan (L1)
        distances = cdist(X, Y, metric='cityblock')
        
        # Aplica el kernel
        K = np.exp(-self.gamma * distances)
        
        return K
    
    def __repr__(self) -> str:
        return f"LaplacianKernel(gamma={self.gamma:.4f})"

