"""
Kernels compuestos - Combinaciones y transformaciones de kernels.

Permite crear kernels complejos mediante:
- Suma: K1 + K2
- Producto: K1 * K2
- Escalado: α * K
- Transformaciones: f(K)
"""

import numpy as np
from typing import Optional, Callable, List, Union
from kernel.core.kernel_base import KernelBase


class CompositeKernel(KernelBase):
    """
    Kernel compuesto que combina múltiples kernels.
    
    Soporta operaciones:
    - Suma: K = K1 + K2 + ... + Kn
    - Producto: K = K1 * K2 * ... * Kn
    - Combinación lineal: K = α1*K1 + α2*K2 + ... + αn*Kn
    """
    
    def __init__(self, kernels: List[KernelBase], 
                 operation: str = "sum",
                 weights: Optional[List[float]] = None,
                 use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel compuesto.
        
        Parámetros:
        -----------
        kernels : List[KernelBase]
            Lista de kernels a combinar.
        operation : str, default="sum"
            Operación: "sum", "product", o "weighted_sum".
        weights : List[float], opcional
            Pesos para combinación ponderada (requerido si operation="weighted_sum").
        use_cache : bool, default=True
            Habilita caching.
        enable_gpu : bool, default=False
            Habilita GPU.
        """
        if not kernels:
            raise ValueError("Debe proporcionar al menos un kernel")
        
        if operation == "weighted_sum" and weights is None:
            raise ValueError("weights es requerido para weighted_sum")
        
        if weights and len(weights) != len(kernels):
            raise ValueError("weights debe tener la misma longitud que kernels")
        
        self.kernels = kernels
        self.operation = operation
        self.weights = weights if weights else [1.0] * len(kernels)
        
        super().__init__(operation=operation, use_cache=use_cache, enable_gpu=enable_gpu)
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calcula el kernel compuesto."""
        if self.operation == "sum":
            K = sum(kernel(X, Y) for kernel in self.kernels)
        elif self.operation == "product":
            K = np.ones((X.shape[0], Y.shape[0]))
            for kernel in self.kernels:
                K *= kernel(X, Y)
        elif self.operation == "weighted_sum":
            K = sum(w * kernel(X, Y) for w, kernel in zip(self.weights, self.kernels))
        else:
            raise ValueError(f"Operación desconocida: {self.operation}")
        
        return K
    
    def __repr__(self) -> str:
        op_str = self.operation
        if self.operation == "weighted_sum":
            op_str += f" (weights={self.weights})"
        return f"CompositeKernel({op_str}, {len(self.kernels)} kernels)"


class ScaledKernel(KernelBase):
    """
    Kernel escalado: K_scaled = α * K.
    
    Útil para ajustar la escala de un kernel sin cambiar sus propiedades.
    """
    
    def __init__(self, kernel: KernelBase, scale: float = 1.0,
                 use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel escalado.
        
        Parámetros:
        -----------
        kernel : KernelBase
            Kernel base a escalar.
        scale : float, default=1.0
            Factor de escala.
        """
        if scale <= 0:
            raise ValueError("scale debe ser positivo")
        
        super().__init__(scale=scale, use_cache=use_cache, enable_gpu=enable_gpu)
        self.kernel = kernel
        self.scale = scale
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calcula el kernel escalado."""
        return self.scale * self.kernel(X, Y)
    
    def __repr__(self) -> str:
        return f"ScaledKernel(scale={self.scale:.4f}, kernel={self.kernel})"


class TransformedKernel(KernelBase):
    """
    Kernel con transformación: K_transformed = f(K).
    
    Permite aplicar transformaciones no lineales a un kernel base.
    """
    
    def __init__(self, kernel: KernelBase, transform: Callable[[np.ndarray], np.ndarray],
                 use_cache: bool = True, enable_gpu: bool = False):
        """
        Inicializa el kernel transformado.
        
        Parámetros:
        -----------
        kernel : KernelBase
            Kernel base.
        transform : Callable
            Función de transformación f: R -> R aplicada elemento a elemento.
        """
        super().__init__(use_cache=use_cache, enable_gpu=enable_gpu)
        self.kernel = kernel
        self.transform = transform
    
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calcula el kernel transformado."""
        K_base = self.kernel(X, Y)
        return self.transform(K_base)
    
    def __repr__(self) -> str:
        return f"TransformedKernel(kernel={self.kernel})"

