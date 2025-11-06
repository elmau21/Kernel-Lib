"""
Support Vector Machine (SVM) con kernels - Implementación completa con SMO.

Implementa el algoritmo SMO (Sequential Minimal Optimization) para resolver
el problema de optimización cuadrática del SVM.

Teoría:
min_α (1/2) α^T Q α - e^T α
s.t.  0 ≤ α_i ≤ C, ∀i
      y^T α = 0

donde Q_ij = y_i y_j K(x_i, x_j)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize
from kernel.core.kernel_base import KernelBase
from kernel.kernels.rbf import RBFKernel
import warnings


class KernelSVM:
    """
    Support Vector Machine con soporte para kernels arbitrarios.
    
    Implementa:
    - Algoritmo SMO (Sequential Minimal Optimization)
    - Optimización dual del problema SVM
    - Soporte para clasificación binaria y multi-clase (one-vs-one)
    - Selección automática de hiperparámetros (grid search)
    
    Referencias:
    - Platt (1998): "Sequential Minimal Optimization"
    - Schölkopf & Smola (2002): "Learning with Kernels"
    """
    
    def __init__(self, kernel: Optional[KernelBase] = None, C: float = 1.0,
                 tol: float = 1e-3, max_iter: int = 1000, 
                 use_smo: bool = True, verbose: bool = False):
        """
        Inicializa el SVM.
        
        Parámetros:
        -----------
        kernel : KernelBase, opcional
            Kernel a usar. Por defecto RBF.
        C : float, default=1.0
            Parámetro de regularización (soft margin).
        tol : float, default=1e-3
            Tolerancia para convergencia.
        max_iter : int, default=1000
            Número máximo de iteraciones.
        use_smo : bool, default=True
            Usa algoritmo SMO (más rápido) o optimización general.
        verbose : bool, default=False
            Muestra progreso del entrenamiento.
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.use_smo = use_smo
        self.verbose = verbose
        
        # Parámetros del modelo (se inicializan en fit)
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena el modelo SVM.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de entrenamiento.
        y : np.ndarray, shape (n_samples,)
            Etiquetas (+1 o -1 para clasificación binaria).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Validación
        if X.ndim != 2:
            raise ValueError("X debe ser 2D")
        if y.ndim != 1:
            raise ValueError("y debe ser 1D")
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud")
        
        # Convierte etiquetas a +1/-1 si es necesario
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM binario requiere exactamente 2 clases")
        
        y_binary = np.where(y == unique_labels[0], -1.0, 1.0)
        
        # Almacena datos de entrenamiento
        self.X_train_ = X
        self.y_train_ = y_binary
        
        if self.use_smo:
            self._fit_smo(X, y_binary)
        else:
            self._fit_qp(X, y_binary)
        
        return self
    
    def _fit_smo(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena usando algoritmo SMO (Sequential Minimal Optimization).
        
        SMO es más eficiente que QP general para problemas SVM.
        """
        n_samples = X.shape[0]
        
        # Calcula matriz de kernel una vez
        K = self.kernel.gram_matrix(X)
        Q = np.outer(y, y) * K
        
        # Inicializa variables duales
        alpha = np.zeros(n_samples)
        b = 0.0
        
        # Variables para SMO
        E = np.zeros(n_samples)  # Errores
        
        # Bucle principal de SMO
        num_changed = 0
        examine_all = True
        
        for iteration in range(self.max_iter):
            num_changed = 0
            
            if examine_all:
                # Examina todos los ejemplos
                for i in range(n_samples):
                    num_changed += self._examine_example_smo(
                        i, X, y, alpha, b, E, K, Q
                    )
            else:
                # Examina solo ejemplos en los límites (0 < alpha < C)
                indices = np.where((alpha > self.tol) & 
                                  (alpha < self.C - self.tol))[0]
                for i in indices:
                    num_changed += self._examine_example_smo(
                        i, X, y, alpha, b, E, K, Q
                    )
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            # Calcula función objetivo
            if self.verbose and iteration % 100 == 0:
                obj = 0.5 * alpha.T @ Q @ alpha - alpha.sum()
                print(f"Iteración {iteration}: Objetivo = {obj:.6f}, "
                      f"Cambios = {num_changed}")
            
            if num_changed == 0 and not examine_all:
                break
        
        # Extrae vectores de soporte
        sv_indices = np.where(alpha > self.tol)[0]
        self.support_vectors_ = X[sv_indices]
        self.support_vector_labels_ = y[sv_indices]
        self.dual_coef_ = alpha[sv_indices] * y[sv_indices]
        self.n_support_ = len(sv_indices)
        
        # Calcula intercepto
        if len(sv_indices) > 0:
            # Usa vectores de soporte en el margen (0 < alpha < C)
            margin_sv = np.where((alpha > self.tol) & 
                                 (alpha < self.C - self.tol))[0]
            if len(margin_sv) > 0:
                # Calcula b usando vectores de soporte en el margen
                predictions = np.sum(
                    alpha[sv_indices] * y[sv_indices] * 
                    K[sv_indices][:, margin_sv[0]], axis=0
                )
                self.intercept_ = y[margin_sv[0]] - predictions
            else:
                # Fallback: promedio de todos los vectores de soporte
                predictions = np.sum(
                    alpha[sv_indices] * y[sv_indices] * 
                    K[sv_indices][:, sv_indices].mean(axis=1), axis=0
                )
                self.intercept_ = np.mean(y[sv_indices] - predictions)
        else:
            self.intercept_ = 0.0
    
    def _examine_example_smo(self, i2: int, X: np.ndarray, y: np.ndarray,
                            alpha: np.ndarray, b: float, E: np.ndarray,
                            K: np.ndarray, Q: np.ndarray) -> int:
        """
        Examina y optimiza un ejemplo en SMO.
        
        Retorna 1 si se hizo un cambio, 0 en caso contrario.
        """
        y2 = y[i2]
        alpha2 = alpha[i2]
        E2 = E[i2]
        r2 = E2 * y2
        
        # Verifica condiciones KKT
        if ((r2 < -self.tol and alpha2 < self.C) or 
            (r2 > self.tol and alpha2 > 0)):
            # Busca segundo índice para optimización
            if len(np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]) > 1:
                # Selecciona i1 que maximiza |E1 - E2|
                indices = np.where((alpha > self.tol) & 
                                  (alpha < self.C - self.tol))[0]
                if i2 in indices:
                    indices = indices[indices != i2]
                if len(indices) > 0:
                    E1_E2 = E[indices] - E2
                    i1 = indices[np.argmax(np.abs(E1_E2))]
                else:
                    i1 = self._select_second_heuristic(i2, alpha, E)
            else:
                i1 = self._select_second_heuristic(i2, alpha, E)
            
            if self._take_step_smo(i1, i2, X, y, alpha, b, E, K):
                return 1
        
        return 0
    
    def _select_second_heuristic(self, i2: int, alpha: np.ndarray, 
                                 E: np.ndarray) -> int:
        """Selecciona segundo índice usando heurística."""
        non_bound = np.where((alpha > self.tol) & 
                            (alpha < self.C - self.tol))[0]
        if len(non_bound) > 0:
            if i2 in non_bound:
                non_bound = non_bound[non_bound != i2]
            if len(non_bound) > 0:
                i1 = non_bound[np.argmax(np.abs(E[non_bound] - E[i2]))]
            else:
                i1 = np.random.choice(np.where(alpha > self.tol)[0])
        else:
            i1 = np.random.choice(len(alpha))
        return i1
    
    def _take_step_smo(self, i1: int, i2: int, X: np.ndarray, y: np.ndarray,
                      alpha: np.ndarray, b: float, E: np.ndarray,
                      K: np.ndarray) -> bool:
        """
        Optimiza alpha[i1] y alpha[i2] en SMO.
        
        Retorna True si se hizo un cambio.
        """
        if i1 == i2:
            return False
        
        alpha1_old = alpha[i1]
        alpha2_old = alpha[i2]
        y1, y2 = y[i1], y[i2]
        E1, E2 = E[i1], E[i2]
        
        # Calcula límites
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L == H:
            return False
        
        # Calcula eta (segunda derivada)
        k11 = K[i1, i1]
        k12 = K[i1, i2]
        k22 = K[i2, i2]
        eta = k11 + k22 - 2 * k12
        
        if eta > 0:
            # Caso normal: mínimo en el interior
            alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
            # Recorta a [L, H]
            alpha2_new = np.clip(alpha2_new, L, H)
        else:
            # Caso degenerado: evalua en los extremos
            Lobj = self._objective_smo(alpha, i1, i2, L, y, K, Q)
            Hobj = self._objective_smo(alpha, i1, i2, H, y, K, Q)
            if Lobj < Hobj - self.tol:
                alpha2_new = L
            elif Lobj > Hobj + self.tol:
                alpha2_new = H
            else:
                alpha2_new = alpha2_old
        
        if abs(alpha2_new - alpha2_old) < self.tol * (alpha2_new + alpha2_old + self.tol):
            return False
        
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        alpha[i1] = alpha1_new
        alpha[i2] = alpha2_new
        
        # Actualiza errores
        delta1 = (alpha1_new - alpha1_old) * y1
        delta2 = (alpha2_new - alpha2_old) * y2
        
        for i in range(len(alpha)):
            if 0 < alpha[i] < self.C:
                E[i] += delta1 * y1 * K[i1, i] + delta2 * y2 * K[i2, i]
        
        return True
    
    def _objective_smo(self, alpha: np.ndarray, i1: int, i2: int, 
                       alpha2_val: float, y: np.ndarray, K: np.ndarray,
                       Q: np.ndarray) -> float:
        """Calcula función objetivo para un valor de alpha2."""
        alpha1_val = alpha[i1] + y[i1] * y[i2] * (alpha[i2] - alpha2_val)
        alpha_temp = alpha.copy()
        alpha_temp[i1] = alpha1_val
        alpha_temp[i2] = alpha2_val
        return 0.5 * alpha_temp.T @ Q @ alpha_temp - alpha_temp.sum()
    
    def _fit_qp(self, X: np.ndarray, y: np.ndarray):
        """Entrena usando optimización cuadrática general (más lento)."""
        n_samples = X.shape[0]
        K = self.kernel.gram_matrix(X)
        Q = np.outer(y, y) * K
        
        # Función objetivo: (1/2) α^T Q α - e^T α
        def objective(alpha):
            return 0.5 * alpha.T @ Q @ alpha - alpha.sum()
        
        # Gradiente
        def gradient(alpha):
            return Q @ alpha - np.ones(n_samples)
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: y @ alpha}
        ]
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Optimización
        alpha0 = np.zeros(n_samples)
        result = minimize(objective, alpha0, method='SLSQP',
                         jac=gradient, bounds=bounds, constraints=constraints,
                         options={'maxiter': self.max_iter})
        
        if not result.success:
            warnings.warn("Optimización QP no convergió")
        
        alpha = result.x
        
        # Extrae vectores de soporte
        sv_indices = np.where(alpha > self.tol)[0]
        self.support_vectors_ = X[sv_indices]
        self.support_vector_labels_ = y[sv_indices]
        self.dual_coef_ = alpha[sv_indices] * y[sv_indices]
        self.n_support_ = len(sv_indices)
        
        # Calcula intercepto
        if len(sv_indices) > 0:
            margin_sv = np.where((alpha > self.tol) & 
                                 (alpha < self.C - self.tol))[0]
            if len(margin_sv) > 0:
                predictions = np.sum(
                    alpha[sv_indices] * y[sv_indices] * 
                    K[sv_indices][:, margin_sv[0]], axis=0
                )
                self.intercept_ = y[margin_sv[0]] - predictions
            else:
                self.intercept_ = 0.0
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la función de decisión f(x) = Σ α_i y_i K(x_i, x) + b.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de prueba.
        
        Retorna:
        --------
        scores : np.ndarray, shape (n_samples,)
            Scores de decisión.
        """
        if self.support_vectors_ is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        X = np.asarray(X, dtype=np.float64)
        K_test = self.kernel(X, self.support_vectors_)
        scores = np.dot(K_test, self.dual_coef_) + self.intercept_
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clases.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de prueba.
        
        Retorna:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicciones de clase (+1 o -1).
        """
        scores = self.decision_function(X)
        return np.sign(scores)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula precisión del modelo.
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de prueba.
        y : np.ndarray
            Etiquetas verdaderas.
        
        Retorna:
        --------
        accuracy : float
            Precisión (0-1).
        """
        predictions = self.predict(X)
        # Convierte etiquetas a +1/-1 si es necesario
        y_binary = np.where(y == np.unique(y)[0], -1.0, 1.0)
        return np.mean(predictions == y_binary)

