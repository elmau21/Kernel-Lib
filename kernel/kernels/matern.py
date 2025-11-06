"""
Kernel Matern - Generalización del kernel RBF con parámetro de suavidad.

K(x, y) = (2^(1-ν)/Γ(ν)) * (√(2ν) * r / ℓ)^ν * K_ν(√(2ν) * r / ℓ)

donde r = ||x - y||, ℓ es el length-scale, ν controla la suavidad,
y K_ν es la función de Bessel modificada.

Usado extensivamente en Gaussian Processes y análisis espacial.
"""

import numpy as np
from typing import Optional
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from kernel.core.kernel_base import KernelBase


class MaternKernel(KernelBase):
    """
    Kernel Matern con parámetro de suavidad ν.
    
    Propiedades:
    - ν = 1/2: Equivalente a kernel exponencial (no diferenciable)
    - ν = 3/2: Una vez diferenciable
    - ν = 5/2: Dos veces diferenciable
    - ν → ∞: Converge a RBF (infinitamente diferenciable)
    
    Útil para:
    - Gaussian Processes con diferentes niveles de suavidad
    - Análisis espacial y geoestadística
    - Modelado de funciones con ruido controlado
    """
    
    def __init__(self, length_scale: float = 1.0, nu: float = 1.5, 
                 use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel Matern.
        
        Parámetros:
        -----------
        length_scale : float, default=1.0
            Parámetro de escala (ℓ).
        nu : float, default=1.5
            Parámetro de suavidad. Valores comunes: 0.5, 1.5, 2.5, ∞
        use_cache : bool, default=True
            Habilita caching.
        enable_gpu : bool, default=False
            Habilita GPU.
        """
        if nu <= 0:
            raise ValueError("nu debe ser positivo")
        if length_scale <= 0:
            raise ValueError("length_scale debe ser positivo")
        
        super().__init__(length_scale=length_scale, nu=nu, 
                        use_cache=use_cache, enable_gpu=enable_gpu)
        self.length_scale = length_scale
        self.nu = nu
        
        # Precomputa constantes
        self._sqrt_2nu = np.sqrt(2 * nu)
        self._coef = 2 ** (1 - nu) / gamma(nu)
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calcula el kernel Matern."""
        # Distancias euclidianas
        distances = cdist(X, Y, metric='euclidean')
        
        # Evita división por cero
        distances = np.maximum(distances, 1e-12)
        
        # Calcula r / ℓ
        scaled_distances = self._sqrt_2nu * distances / self.length_scale
        
        # Casos especiales para valores comunes de nu
        if self.nu == 0.5:
            # Matern 1/2: exponencial
            K = np.exp(-scaled_distances / self._sqrt_2nu)
        elif self.nu == 1.5:
            # Matern 3/2: (1 + √3r) * exp(-√3r)
            K = (1 + scaled_distances) * np.exp(-scaled_distances)
        elif self.nu == 2.5:
            # Matern 5/2: (1 + √5r + 5r²/3) * exp(-√5r)
            K = (1 + scaled_distances + scaled_distances**2 / 3) * np.exp(-scaled_distances)
        else:
            # Caso general usando función de Bessel
            # K(x, y) = coef * (scaled_dist)^nu * K_nu(scaled_dist)
            scaled_distances_safe = np.maximum(scaled_distances, 1e-12)
            bessel_term = kv(self.nu, scaled_distances_safe)
            K = self._coef * (scaled_distances_safe ** self.nu) * bessel_term
            # Maneja casos donde bessel_term puede ser inf
            K = np.where(np.isfinite(K), K, 0.0)
        
        return K
    
    def __repr__(self) -> str:
        return f"MaternKernel(length_scale={self.length_scale:.4f}, nu={self.nu:.2f})"

