"""
Kernel Polinomial avanzado.

K(x, y) = (gamma * x^T y + coef0)^degree

Implementación con validaciones y optimizaciones.
"""

import numpy as np
from typing import Optional
from kernel.core.kernel_base import KernelBase


class PolynomialKernel(KernelBase):
    """
    Kernel Polinomial con optimizaciones.
    
    K(x, y) = (gamma * x^T y + coef0)^degree
    
    Propiedades matemáticas:
    - Captura interacciones polinomiales de orden 'degree'
    - El grado controla la complejidad del modelo (VC dimension)
    - Homogéneo si coef0=0, inhomogéneo si coef0≠0
    - Útil para relaciones no lineales con estructura polinomial
    
    Teoría:
    - Mapea a espacio de características de dimensión C(n+d, d)
    - Para degree=2: mapea a espacio de monomios de grado ≤2
    """
    
    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 0.0,
                 use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel polinomial.
        
        Parámetros:
        -----------
        degree : int, default=3
            Grado del polinomio. Debe ser entero positivo.
        gamma : float, default=1.0
            Parámetro de escala.
        coef0 : float, default=0.0
            Término independiente (bias).
        use_cache : bool, default=True
            Habilita caching.
        enable_gpu : bool, default=False
            Habilita GPU.
        """
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree debe ser un entero positivo")
        if gamma <= 0:
            raise ValueError("gamma debe ser positivo")
        
        super().__init__(degree=degree, gamma=gamma, coef0=coef0,
                        use_cache=use_cache, enable_gpu=enable_gpu)
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de kernel polinomial.
        
        Optimización: calcula producto punto una vez y luego eleva.
        """
        # Calcula el producto punto: x^T y
        XY = np.dot(X, Y.T)
        
        # Aplica la transformación polinomial
        # Maneja casos especiales para estabilidad numérica
        if self.coef0 == 0.0:
            # Polinomio homogéneo
            K = (self.gamma * XY) ** self.degree
        else:
            # Polinomio inhomogéneo
            K = (self.gamma * XY + self.coef0) ** self.degree
        
        return K
    
    def __repr__(self) -> str:
        return f"PolynomialKernel(degree={self.degree}, gamma={self.gamma:.4f}, coef0={self.coef0:.4f})"

