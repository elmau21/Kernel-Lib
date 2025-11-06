"""
Gaussian Process (GP) con kernels - Implementación completa.

Implementa regresión y clasificación usando procesos gaussianos.

Teoría:
p(f|X, y) ∝ p(y|f) p(f|X)

donde f ~ GP(m, K) es un proceso gaussiano con media m y covarianza K.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
from kernel.core.kernel_base import KernelBase
from kernel.kernels.rbf import RBFKernel
import warnings


class GaussianProcess:
    """
    Gaussian Process para regresión con kernels.
    
    Implementa:
    - Regresión GP con ruido gaussiano
    - Predicción con incertidumbre (varianza)
    - Optimización de hiperparámetros (máxima verosimilitud)
    - Sampling de funciones del proceso
    
    Referencias:
    - Rasmussen & Williams (2006): "Gaussian Processes for Machine Learning"
    """
    
    def __init__(self, kernel: Optional[KernelBase] = None,
                 alpha: float = 1e-6, optimizer: Optional[str] = "fmin_l_bfgs_b",
                 n_restarts_optimizer: int = 0, normalize_y: bool = False):
        """
        Inicializa el Gaussian Process.
        
        Parámetros:
        -----------
        kernel : KernelBase, opcional
            Kernel a usar. Por defecto RBF.
        alpha : float, default=1e-6
            Ruido observacional (varianza).
        optimizer : str, opcional
            Optimizador para hiperparámetros: "fmin_l_bfgs_b" o None.
        n_restarts_optimizer : int, default=0
            Número de reinicios para optimización.
        normalize_y : bool, default=False
            Normaliza las salidas y.
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        
        # Parámetros del modelo
        self.X_train_ = None
        self.y_train_ = None
        self.y_train_mean_ = None
        self.y_train_std_ = None
        self.L_ = None
        self.alpha_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo GP.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de entrenamiento.
        y : np.ndarray, shape (n_samples,)
            Valores objetivo.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X debe ser 2D")
        if y.ndim != 1:
            raise ValueError("y debe ser 1D")
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud")
        
        self.X_train_ = X
        
        # Normaliza y si es necesario
        if self.normalize_y:
            self.y_train_mean_ = np.mean(y)
            self.y_train_std_ = np.std(y)
            if self.y_train_std_ == 0:
                self.y_train_std_ = 1.0
            self.y_train_ = (y - self.y_train_mean_) / self.y_train_std_
        else:
            self.y_train_mean_ = 0.0
            self.y_train_std_ = 1.0
            self.y_train_ = y
        
        # Optimiza hiperparámetros si es necesario
        if self.optimizer is not None and self.n_restarts_optimizer >= 0:
            self._optimize_hyperparameters()
        
        # Calcula matriz de covarianza
        K = self.kernel.gram_matrix(X)
        K += self.alpha * np.eye(K.shape[0])
        
        # Descomposición de Cholesky: K = L L^T
        try:
            self.L_ = cholesky(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # Si falla, añade más regularización
            K += 1e-6 * np.eye(K.shape[0])
            self.L_ = cholesky(K, lower=True, check_finite=False)
        
        # Resuelve: K α = y
        # Usando Cholesky: L L^T α = y
        # Primero: L v = y, luego: L^T α = v
        v = solve_triangular(self.L_, self.y_train_, lower=True, check_finite=False)
        self.alpha_ = solve_triangular(self.L_.T, v, lower=False, check_finite=False)
        
        return self
    
    def _optimize_hyperparameters(self):
        """Optimiza hiperparámetros usando máxima verosimilitud."""
        from scipy.optimize import fmin_l_bfgs_b
        
        def objective(params):
            """Función objetivo: log-likelihood negativo."""
            # Extrae parámetros del kernel
            if hasattr(self.kernel, 'gamma'):
                self.kernel.gamma = params[0]
                if len(params) > 1:
                    self.alpha = params[1]
            else:
                self.alpha = params[0]
            
            try:
                return -self._log_marginal_likelihood()
            except:
                return np.inf
        
        # Parámetros iniciales
        if hasattr(self.kernel, 'gamma'):
            x0 = [self.kernel.gamma, self.alpha]
            bounds = [(1e-5, 1e5), (1e-10, 1e2)]
        else:
            x0 = [self.alpha]
            bounds = [(1e-10, 1e2)]
        
        # Optimización con múltiples reinicios
        best_params = x0
        best_ll = -objective(x0)
        
        for _ in range(self.n_restarts_optimizer + 1):
            if _ > 0:
                # Reinicio aleatorio
                if hasattr(self.kernel, 'gamma'):
                    x0 = [np.random.uniform(1e-3, 10), 
                          np.random.uniform(1e-6, 1e-2)]
                else:
                    x0 = [np.random.uniform(1e-6, 1e-2)]
            
            try:
                result = fmin_l_bfgs_b(objective, x0, bounds=bounds, 
                                      approx_grad=True, maxiter=50)
                if -result[1] > best_ll:
                    best_ll = -result[1]
                    best_params = result[0]
            except:
                continue
        
        # Aplica mejores parámetros
        if hasattr(self.kernel, 'gamma'):
            self.kernel.gamma = best_params[0]
            self.alpha = best_params[1]
        else:
            self.alpha = best_params[0]
    
    def _log_marginal_likelihood(self) -> float:
        """
        Calcula el log-likelihood marginal.
        
        log p(y|X) = -0.5 * (y^T K^-1 y + log|K| + n*log(2π))
        """
        if self.L_ is None:
            K = self.kernel.gram_matrix(self.X_train_)
            K += self.alpha * np.eye(K.shape[0])
            try:
                self.L_ = cholesky(K, lower=True, check_finite=False)
            except:
                return -np.inf
        
        # log|K| = 2 * sum(log(diag(L)))
        log_det_K = 2 * np.sum(np.log(np.diag(self.L_)))
        
        # y^T K^-1 y = y^T α (donde K α = y)
        quad_form = np.dot(self.y_train_, self.alpha_)
        
        n = len(self.y_train_)
        log_likelihood = (-0.5 * quad_form - 
                         0.5 * log_det_K - 
                         0.5 * n * np.log(2 * np.pi))
        
        return log_likelihood
    
    def predict(self, X: np.ndarray, return_std: bool = True, 
                return_cov: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Predice usando el GP.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de prueba.
        return_std : bool, default=True
            Retorna desviación estándar de predicciones.
        return_cov : bool, default=False
            Retorna matriz de covarianza.
        
        Retorna:
        --------
        y_mean : np.ndarray
            Media de predicciones.
        y_std : np.ndarray, opcional
            Desviación estándar.
        y_cov : np.ndarray, opcional
            Matriz de covarianza.
        """
        if self.X_train_ is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Media: μ* = K(X*, X) α
        K_star = self.kernel(X, self.X_train_)
        y_mean = np.dot(K_star, self.alpha_)
        
        # Desnormaliza
        y_mean = y_mean * self.y_train_std_ + self.y_train_mean_
        
        if not return_std and not return_cov:
            return y_mean
        
        # Varianza: σ²* = K(X*, X*) - K(X*, X) K^-1 K(X, X*)
        K_star_star = self.kernel.gram_matrix(X)
        
        # v = L^-1 K(X, X*)^T
        v = solve_triangular(self.L_, K_star.T, lower=True, check_finite=False)
        
        # Varianza: diag(K**) - diag(v^T v)
        y_var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
        y_var = np.maximum(y_var, 0)  # Asegura no negatividad
        
        y_std = np.sqrt(y_var) * self.y_train_std_
        
        if return_cov:
            # Covarianza completa
            y_cov = K_star_star - np.dot(v.T, v)
            y_cov *= self.y_train_std_ ** 2
            return y_mean, y_std, y_cov
        
        if return_std:
            return y_mean, y_std
        
        return y_mean
    
    def sample_y(self, X: np.ndarray, n_samples: int = 1, 
                 random_state: Optional[int] = None) -> np.ndarray:
        """
        Muestra funciones del proceso gaussiano.
        
        Parámetros:
        -----------
        X : np.ndarray
            Puntos donde muestrear.
        n_samples : int, default=1
            Número de muestras.
        random_state : int, opcional
            Semilla aleatoria.
        
        Retorna:
        --------
        samples : np.ndarray, shape (n_samples, n_points)
            Muestras de funciones.
        """
        if self.X_train_ is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Predice media y covarianza
        y_mean, _, y_cov = self.predict(X, return_std=False, return_cov=True)
        
        # Muestra de distribución gaussiana multivariada
        # y ~ N(μ, Σ)
        samples = np.random.multivariate_normal(
            y_mean, y_cov, size=n_samples
        )
        
        return samples
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula R² score.
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de prueba.
        y : np.ndarray
            Valores verdaderos.
        
        Retorna:
        --------
        score : float
            R² score.
        """
        y_pred = self.predict(X, return_std=False)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

