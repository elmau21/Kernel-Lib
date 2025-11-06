"""
Kernel Lineal optimizado.

K(x, y) = x^T y

Implementación altamente optimizada para el caso más simple.
"""

import numpy as np
from typing import Optional
from kernel.core.kernel_base import KernelBase


class LinearKernel(KernelBase):
    """
    Kernel Lineal con optimizaciones.
    
    K(x, y) = x^T y
    
    Propiedades matemáticas:
    - Más rápido de calcular (O(n*m*d))
    - Equivalente a trabajar en el espacio original (no hay feature mapping)
    - Mercer kernel trivial
    - Útil cuando los datos ya son linealmente separables
    
    Optimizaciones:
    - Usa BLAS optimizado (np.dot)
    - Sin overhead de transformaciones
    - Ideal para datasets grandes
    """
    
    def __init__(self, use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel lineal.
        
        Parámetros:
        -----------
        use_cache : bool, default=True
            Habilita caching.
        enable_gpu : bool, default=False
            Habilita GPU.
        """
        super().__init__(use_cache=use_cache, enable_gpu=enable_gpu)
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de kernel lineal (producto punto).
        
        Optimización: usa BLAS optimizado directamente.
        """
        K = np.dot(X, Y.T)
        return K
    
    def __repr__(self) -> str:
        return "LinearKernel()"

