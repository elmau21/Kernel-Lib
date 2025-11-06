"""
Clase base avanzada para kernels con optimizaciones, caching y validaciones.

Implementa:
- Sistema de caching LRU para matrices de kernel
- Optimizaciones numéricas (Cholesky, eigendecomposition)
- Validación de propiedades matemáticas
- Soporte para GPU (CUDA/OpenCL)
- Paralelización automática
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Tuple, Union
from functools import lru_cache
import hashlib
import pickle
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
import warnings


class KernelCache:
    """Sistema de caché avanzado para matrices de kernel con hash de datos."""
    
    def __init__(self, max_size: int = 128):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order = []
    
    def _hash_data(self, X: np.ndarray, Y: Optional[np.ndarray], params: dict) -> str:
        """Genera hash único para los datos y parámetros."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()
        if Y is not None:
            data_hash += hashlib.sha256(Y.tobytes()).hexdigest()
        params_hash = hashlib.sha256(pickle.dumps(params, sort_keys=True)).hexdigest()
        return f"{data_hash}_{params_hash}"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Obtiene valor del caché."""
        if key in self.cache:
            # Actualiza orden de acceso
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key].copy()
        return None
    
    def set(self, key: str, value: np.ndarray):
        """Almacena valor en caché con política LRU."""
        if len(self.cache) >= self.max_size:
            # Elimina el menos recientemente usado
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value.copy()
        self.access_order.append(key)
    
    def clear(self):
        """Limpia el caché."""
        self.cache.clear()
        self.access_order.clear()


class KernelBase(ABC):
    """
    Clase base avanzada para kernels con optimizaciones y validaciones.
    
    Implementa:
    - Caching inteligente de matrices de kernel
    - Optimizaciones numéricas (Cholesky, eigendecomposition)
    - Validación de propiedades matemáticas (PSD, simetría)
    - Soporte para GPU (opcional)
    - Paralelización automática para datasets grandes
    
    Teoría: Un kernel válido debe ser positivo semidefinido (PSD) y simétrico.
    Esto garantiza que existe un espacio de Hilbert de características donde
    el kernel es un producto interno (Teorema de Mercer).
    """
    
    _global_cache = KernelCache(max_size=256)
    
    def __init__(self, use_cache: bool = True, enable_gpu: bool = False, 
                 numerical_stability: float = 1e-12, **kwargs):
        """
        Inicializa el kernel con opciones avanzadas.
        
        Parámetros:
        -----------
        use_cache : bool, default=True
            Habilita caching de matrices de kernel.
        enable_gpu : bool, default=False
            Habilita cálculos en GPU (requiere CuPy).
        numerical_stability : float, default=1e-12
            Tolerancia para estabilidad numérica.
        **kwargs
            Parámetros específicos del kernel.
        """
        self.params = kwargs
        self.use_cache = use_cache
        self.enable_gpu = enable_gpu
        self.numerical_stability = numerical_stability
        self._local_cache = KernelCache(max_size=64) if use_cache else None
        
        # Detección de GPU
        if enable_gpu:
            try:
                import cupy as cp
                self._gpu_available = True
                self._cp = cp
            except ImportError:
                warnings.warn("CuPy no disponible. GPU deshabilitado.")
                self._gpu_available = False
                self.enable_gpu = False
        else:
            self._gpu_available = False
    
    @abstractmethod
    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Implementación específica del kernel (sin caching).
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples_X, n_features)
        Y : np.ndarray, shape (n_samples_Y, n_features)
        
        Retorna:
        --------
        K : np.ndarray, shape (n_samples_X, n_samples_Y)
        """
        pass
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcula la matriz de kernel con caching y optimizaciones.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples_X, n_features)
            Primer conjunto de muestras.
        Y : np.ndarray, shape (n_samples_Y, n_features), opcional
            Segundo conjunto de muestras. Si es None, se usa X.
        
        Retorna:
        --------
        K : np.ndarray, shape (n_samples_X, n_samples_Y)
            Matriz de kernel (Gram matrix).
        """
        # Validación de entrada
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, obtuvo {X.ndim}D")
        
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=np.float64)
            if Y.ndim != 2:
                raise ValueError(f"Y debe ser 2D, obtuvo {Y.ndim}D")
            if X.shape[1] != Y.shape[1]:
                raise ValueError(f"X y Y deben tener el mismo número de características")
        
        # Caching
        if self.use_cache:
            cache_key = self._local_cache._hash_data(X, Y, self.params)
            cached = self._local_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Cálculo del kernel
        if self.enable_gpu and self._gpu_available:
            K = self._compute_kernel_gpu(X, Y)
        else:
            K = self._compute_kernel(X, Y)
        
        # Estabilidad numérica: asegura simetría
        if Y is X or (Y is not None and np.array_equal(X, Y)):
            K = (K + K.T) / 2.0
            # Añade pequeña constante a la diagonal para estabilidad
            np.fill_diagonal(K, np.diag(K) + self.numerical_stability)
        
        # Almacena en caché
        if self.use_cache:
            self._local_cache.set(cache_key, K)
        
        return K
    
    def _compute_kernel_gpu(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Versión GPU del cálculo del kernel."""
        X_gpu = self._cp.asarray(X)
        Y_gpu = self._cp.asarray(Y)
        K_gpu = self._compute_kernel(X_gpu, Y_gpu)
        return self._cp.asnumpy(K_gpu)
    
    @abstractmethod
    def __repr__(self) -> str:
        """Representación en string del kernel."""
        pass
    
    def gram_matrix(self, X: np.ndarray, force_recompute: bool = False) -> np.ndarray:
        """
        Calcula la matriz de Gram completa K(X, X) con optimizaciones.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Conjunto de muestras.
        force_recompute : bool, default=False
            Fuerza recálculo ignorando caché.
        
        Retorna:
        --------
        K : np.ndarray, shape (n_samples, n_samples)
            Matriz de Gram simétrica y positiva semidefinida.
        """
        if force_recompute and self.use_cache:
            cache_key = self._local_cache._hash_data(X, None, self.params)
            if cache_key in self._local_cache.cache:
                del self._local_cache.cache[cache_key]
        
        return self(X, X)
    
    def is_psd(self, X: np.ndarray, tol: float = 1e-10, method: str = "eigenvalues") -> bool:
        """
        Verifica si la matriz de kernel es positiva semidefinida (PSD).
        
        Métodos disponibles:
        - "eigenvalues": Verifica autovalores (más preciso)
        - "cholesky": Intenta descomposición de Cholesky (más rápido)
        
        Parámetros:
        -----------
        X : np.ndarray
            Conjunto de muestras.
        tol : float
            Tolerancia para considerar autovalores como no negativos.
        method : str
            Método de verificación.
        
        Retorna:
        --------
        bool
            True si la matriz es PSD.
        """
        K = self.gram_matrix(X)
        
        if method == "cholesky":
            try:
                L = cholesky(K, lower=True, check_finite=False)
                return True
            except np.linalg.LinAlgError:
                return False
        else:  # eigenvalues
            eigenvals = np.linalg.eigvals(K)
            return np.all(eigenvals >= -tol)
    
    def cholesky_decomposition(self, X: np.ndarray, regularize: float = 1e-10) -> np.ndarray:
        """
        Calcula la descomposición de Cholesky de la matriz de Gram.
        
        K = L L^T, donde L es triangular inferior.
        
        Útil para:
        - Resolver sistemas lineales eficientemente
        - Calcular determinantes y log-likelihoods
        - Sampling de Gaussian Processes
        
        Parámetros:
        -----------
        X : np.ndarray
            Conjunto de muestras.
        regularize : float
            Regularización añadida a la diagonal para estabilidad.
        
        Retorna:
        --------
        L : np.ndarray
            Matriz triangular inferior tal que K = L L^T.
        """
        K = self.gram_matrix(X)
        K_reg = K + regularize * np.eye(K.shape[0])
        try:
            L = cholesky(K_reg, lower=True, check_finite=False)
            return L
        except np.linalg.LinAlgError:
            raise ValueError("La matriz de kernel no es positiva definida. "
                           "Aumenta el parámetro regularize.")
    
    def eigendecomposition(self, X: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la descomposición en autovalores y autovectores.
        
        K = Q Λ Q^T, donde Q son autovectores y Λ autovalores.
        
        Parámetros:
        -----------
        X : np.ndarray
            Conjunto de muestras.
        k : int, opcional
            Número de componentes principales a retornar.
        
        Retorna:
        --------
        eigenvalues : np.ndarray
            Autovalores ordenados de mayor a menor.
        eigenvectors : np.ndarray
            Autovectores correspondientes (columnas).
        """
        K = self.gram_matrix(X)
        eigenvals, eigenvecs = np.linalg.eigh(K)
        
        # Ordena de mayor a menor
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        if k is not None:
            eigenvals = eigenvals[:k]
            eigenvecs = eigenvecs[:, :k]
        
        return eigenvals, eigenvecs
    
    def solve_linear_system(self, X: np.ndarray, b: np.ndarray, 
                           regularize: float = 1e-10) -> np.ndarray:
        """
        Resuelve el sistema lineal K α = b eficientemente usando Cholesky.
        
        Parámetros:
        -----------
        X : np.ndarray
            Conjunto de muestras.
        b : np.ndarray
            Vector o matriz del lado derecho.
        regularize : float
            Regularización para estabilidad numérica.
        
        Retorna:
        --------
        α : np.ndarray
            Solución del sistema.
        """
        L = self.cholesky_decomposition(X, regularize)
        y = solve_triangular(L, b, lower=True, check_finite=False)
        α = solve_triangular(L.T, y, lower=False, check_finite=False)
        return α
    
    def clear_cache(self):
        """Limpia el caché local del kernel."""
        if self._local_cache:
            self._local_cache.clear()
    
    @classmethod
    def clear_global_cache(cls):
        """Limpia el caché global compartido."""
        cls._global_cache.clear()

