"""
Kernel Principal Component Analysis (KPCA) - Análisis de Componentes Principales con Kernels.

Implementa reducción de dimensionalidad no lineal usando kernels.

Teoría:
- Mapea datos a espacio de características de alta dimensión
- Aplica PCA en ese espacio
- Permite reducción de dimensionalidad no lineal
"""

import numpy as np
from typing import Optional, Tuple
from kernel.core.kernel_base import KernelBase
from kernel.kernels.rbf import RBFKernel
import warnings


class KernelPCA:
    """
    Kernel Principal Component Analysis con optimizaciones avanzadas.
    
    Implementa:
    - Reducción de dimensionalidad no lineal
    - Centrado de kernel (centering)
    - Selección automática de número de componentes
    - Transformación inversa aproximada (pre-image)
    
    Referencias:
    - Schölkopf et al. (1998): "Nonlinear Component Analysis as a Kernel Eigenvalue Problem"
    """
    
    def __init__(self, kernel: Optional[KernelBase] = None, n_components: Optional[int] = None,
                 center_kernel: bool = True, alpha: float = 1e-6, 
                 eigen_solver: str = "auto"):
        """
        Inicializa KPCA.
        
        Parámetros:
        -----------
        kernel : KernelBase, opcional
            Kernel a usar. Por defecto RBF.
        n_components : int, opcional
            Número de componentes a retornar. Si None, retorna todos.
        center_kernel : bool, default=True
            Centra la matriz de kernel (recomendado).
        alpha : float, default=1e-6
            Regularización para estabilidad numérica.
        eigen_solver : str, default="auto"
            Solver para eigendecomposition: "auto", "dense", "arpack".
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.n_components = n_components
        self.center_kernel = center_kernel
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        
        # Parámetros del modelo
        self.X_fit_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.dual_coef_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Ajusta el modelo KPCA.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de entrenamiento.
        y : ignorado
            No usado, presente por compatibilidad con sklearn.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X debe ser 2D")
        
        self.X_fit_ = X
        n_samples = X.shape[0]
        
        # Calcula matriz de kernel
        K = self.kernel.gram_matrix(X)
        
        # Centra el kernel si es necesario
        if self.center_kernel:
            K = self._center_kernel(K)
        
        # Añade regularización para estabilidad
        K_reg = K + self.alpha * np.eye(n_samples)
        
        # Eigendecomposition
        if self.eigen_solver == "auto":
            # Usa eigendecomposition densa para matrices pequeñas/medianas
            if n_samples < 1000:
                eigen_solver = "dense"
            else:
                eigen_solver = "arpack"
        else:
            eigen_solver = self.eigen_solver
        
        if eigen_solver == "dense":
            eigenvals, eigenvecs = np.linalg.eigh(K_reg)
        else:  # arpack
            from scipy.sparse.linalg import eigs
            k = min(self.n_components or n_samples, n_samples - 1)
            eigenvals, eigenvecs = eigs(K_reg, k=k, which='LM')
            eigenvals = eigenvals.real
            eigenvecs = eigenvecs.real
        
        # Ordena de mayor a menor
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Normaliza autovectores
        # En KPCA, los autovectores deben normalizarse por sqrt(eigenval)
        # para que tengan norma 1 en el espacio de características
        norms = np.sqrt(np.abs(eigenvals))
        norms[norms == 0] = 1.0  # Evita división por cero
        eigenvecs = eigenvecs / norms[np.newaxis, :]
        
        # Selecciona componentes
        if self.n_components is not None:
            eigenvals = eigenvals[:self.n_components]
            eigenvecs = eigenvecs[:, :self.n_components]
        
        self.eigenvalues_ = eigenvals
        self.eigenvectors_ = eigenvecs
        
        # Coeficientes duales para transformación
        self.dual_coef_ = eigenvecs
        
        return self
    
    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """
        Centra la matriz de kernel.
        
        K_centered = K - 1/n * K * 1 - 1/n * 1 * K + 1/n² * 1 * K * 1
        
        donde 1 es matriz de unos.
        """
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        
        K_centered = (K - np.dot(one_n, K) - 
                     np.dot(K, one_n) + 
                     np.dot(one_n, np.dot(K, one_n)))
        
        return K_centered
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforma datos al espacio de componentes principales.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos a transformar.
        
        Retorna:
        --------
        X_transformed : np.ndarray, shape (n_samples, n_components)
            Datos transformados.
        """
        if self.X_fit_ is None:
            raise ValueError("El modelo debe ser entrenado primero (fit)")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Calcula kernel entre datos nuevos y datos de entrenamiento
        K_test = self.kernel(X, self.X_fit_)
        
        # Centra si es necesario
        if self.center_kernel:
            # Centra usando la media del kernel de entrenamiento
            n_train = self.X_fit_.shape[0]
            n_test = X.shape[0]
            
            # Calcula medias
            K_train_mean = self.kernel.gram_matrix(self.X_fit_).mean(axis=0)
            K_test_mean = K_test.mean(axis=1, keepdims=True)
            K_train_mean_all = self.kernel.gram_matrix(self.X_fit_).mean()
            
            # Centra
            K_test_centered = (K_test - 
                              np.tile(K_train_mean, (n_test, 1)) -
                              np.tile(K_test_mean, (1, n_train)) +
                              K_train_mean_all)
        else:
            K_test_centered = K_test
        
        # Proyecta: X_transformed = K_test_centered @ dual_coef_
        X_transformed = np.dot(K_test_centered, self.dual_coef_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ajusta el modelo y transforma los datos.
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrenamiento.
        y : ignorado
        
        Retorna:
        --------
        X_transformed : np.ndarray
            Datos transformados.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray, 
                         method: str = "preimage") -> np.ndarray:
        """
        Transformación inversa aproximada (pre-image).
        
        Nota: La transformación inversa exacta no existe en general.
        Esta es una aproximación.
        
        Parámetros:
        -----------
        X_transformed : np.ndarray
            Datos en el espacio transformado.
        method : str, default="preimage"
            Método: "preimage" (optimización) o "nearest" (vecino más cercano).
        
        Retorna:
        --------
        X_original : np.ndarray
            Aproximación de datos originales.
        """
        if self.X_fit_ is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        if method == "nearest":
            # Método simple: encuentra el punto de entrenamiento más cercano
            distances = np.sum((X_transformed[:, np.newaxis, :] - 
                              self.transform(self.X_fit_)[np.newaxis, :, :]) ** 2, 
                             axis=2)
            nearest_indices = np.argmin(distances, axis=1)
            return self.X_fit_[nearest_indices]
        
        else:  # preimage
            # Optimización para encontrar pre-image
            from scipy.optimize import minimize
            
            n_samples = X_transformed.shape[0]
            n_features = self.X_fit_.shape[1]
            X_reconstructed = np.zeros((n_samples, n_features))
            
            for i in range(n_samples):
                x_target = X_transformed[i]
                
                def objective(x):
                    x = x.reshape(1, -1)
                    x_transformed = self.transform(x)
                    return np.sum((x_transformed - x_target) ** 2)
                
                # Inicializa con el punto de entrenamiento más cercano
                distances = np.sum((x_target[np.newaxis, :] - 
                                  self.transform(self.X_fit_)) ** 2, 
                                 axis=1)
                x0 = self.X_fit_[np.argmin(distances)]
                
                result = minimize(objective, x0, method='BFGS')
                X_reconstructed[i] = result.x
            
            return X_reconstructed
    
    def explained_variance_ratio_(self) -> np.ndarray:
        """
        Calcula la proporción de varianza explicada por cada componente.
        
        Retorna:
        --------
        ratio : np.ndarray
            Proporción de varianza explicada.
        """
        if self.eigenvalues_ is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        total_variance = np.sum(self.eigenvalues_)
        if total_variance == 0:
            return np.zeros_like(self.eigenvalues_)
        
        return self.eigenvalues_ / total_variance

