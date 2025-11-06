"""
Álgebra Lineal Computacional Avanzada.

Implementa algoritmos matemáticos fundamentales:
- Descomposición de matrices (LU, QR, SVD, Cholesky)
- Resolución de sistemas lineales
- Cálculo de autovalores y autovectores (Power Method, QR Algorithm)
- Pseudoinversa de Moore-Penrose
- Normas matriciales

Todo implementado matemáticamente desde cero.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import lu_factor, lu_solve, qr, svd, cholesky
import warnings


class MatrixDecomposition:
    """Clase base para descomposiciones de matrices."""
    
    @staticmethod
    def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Descomposición LU: A = L * U
        
        donde:
        - L es triangular inferior con 1s en la diagonal
        - U es triangular superior
        
        Algoritmo: Eliminación de Gauss con pivoteo parcial.
        
        Parámetros:
        -----------
        A : np.ndarray, shape (n, n)
            Matriz cuadrada.
        
        Retorna:
        --------
        L : np.ndarray
            Matriz triangular inferior.
        U : np.ndarray
            Matriz triangular superior.
        P : np.ndarray
            Matriz de permutación (pivoteo).
        """
        n = A.shape[0]
        A = A.copy().astype(float)
        P = np.eye(n)
        L = np.eye(n)
        
        for i in range(n - 1):
            # Pivoteo parcial: encuentra máximo en columna
            max_idx = i + np.argmax(np.abs(A[i:, i]))
            if max_idx != i:
                # Intercambia filas
                A[[i, max_idx]] = A[[max_idx, i]]
                P[[i, max_idx]] = P[[max_idx, i]]
                L[[i, max_idx], :i] = L[[max_idx, i], :i]
            
            # Eliminación
            for j in range(i + 1, n):
                if A[i, i] == 0:
                    raise ValueError("Matriz singular")
                factor = A[j, i] / A[i, i]
                L[j, i] = factor
                A[j, i:] -= factor * A[i, i:]
        
        U = A
        return L, U, P
    
    @staticmethod
    def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Descomposición QR: A = Q * R
        
        donde:
        - Q es ortogonal (Q^T * Q = I)
        - R es triangular superior
        
        Algoritmo: Gram-Schmidt ortogonalización.
        
        Parámetros:
        -----------
        A : np.ndarray, shape (m, n)
            Matriz (puede ser rectangular).
        
        Retorna:
        --------
        Q : np.ndarray
            Matriz ortogonal.
        R : np.ndarray
            Matriz triangular superior.
        """
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        for j in range(n):
            # Columna j de A
            v = A[:, j].copy()
            
            # Resta proyecciones sobre columnas anteriores
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]
            
            # Normaliza
            R[j, j] = np.linalg.norm(v)
            if R[j, j] < 1e-10:
                warnings.warn("Matriz casi singular")
            Q[:, j] = v / R[j, j]
        
        return Q, R
    
    @staticmethod
    def svd_decomposition(A: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Descomposición SVD: A = U * Σ * V^T
        
        donde:
        - U: autovectores de A*A^T (columnas)
        - Σ: valores singulares (diagonal)
        - V: autovectores de A^T*A (columnas)
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz.
        k : int, opcional
            Número de componentes principales a retornar.
        
        Retorna:
        --------
        U : np.ndarray
            Matriz de vectores singulares izquierdos.
        s : np.ndarray
            Valores singulares.
        Vt : np.ndarray
            Matriz de vectores singulares derechos (transpuesta).
        """
        # Usa implementación de scipy (SVD completo es complejo)
        # Para implementación desde cero, usaría algoritmo iterativo
        U, s, Vt = svd(A, full_matrices=False)
        
        if k is not None:
            U = U[:, :k]
            s = s[:k]
            Vt = Vt[:k, :]
        
        return U, s, Vt
    
    @staticmethod
    def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
        """
        Descomposición de Cholesky: A = L * L^T
        
        donde L es triangular inferior.
        
        Requiere que A sea simétrica y positiva definida.
        
        Parámetros:
        -----------
        A : np.ndarray, shape (n, n)
            Matriz simétrica positiva definida.
        
        Retorna:
        --------
        L : np.ndarray
            Matriz triangular inferior.
        """
        n = A.shape[0]
        L = np.zeros_like(A)
        
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2))
                    s = A[i, i] - np.sum(L[i, :i] ** 2)
                    if s < 0:
                        raise ValueError("Matriz no es positiva definida")
                    L[i, i] = np.sqrt(s)
                else:
                    # Fuera de diagonal: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k])) / L[j,j]
                    s = A[i, j] - np.sum(L[i, :j] * L[j, :j])
                    L[i, j] = s / L[j, j]
        
        return L


class EigenvalueSolver:
    """Solvers para autovalores y autovectores."""
    
    @staticmethod
    def power_method(A: np.ndarray, max_iter: int = 1000, 
                    tol: float = 1e-6) -> Tuple[float, np.ndarray]:
        """
        Power Method para encontrar autovalor dominante.
        
        Algoritmo:
        x_{k+1} = A * x_k / ||A * x_k||
        λ_k = x_k^T * A * x_k / (x_k^T * x_k)
        
        Converge al autovalor de mayor magnitud.
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz cuadrada.
        max_iter : int
            Número máximo de iteraciones.
        tol : float
            Tolerancia para convergencia.
        
        Retorna:
        --------
        eigenvalue : float
            Autovalor dominante.
        eigenvector : np.ndarray
            Autovector correspondiente (normalizado).
        """
        n = A.shape[0]
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        for i in range(max_iter):
            # x_{k+1} = A * x_k
            x_new = np.dot(A, x)
            
            # Normaliza
            x_new = x_new / np.linalg.norm(x_new)
            
            # Calcula autovalor: λ = x^T * A * x
            eigenvalue = np.dot(x_new, np.dot(A, x_new))
            
            # Verifica convergencia
            if np.linalg.norm(x_new - x) < tol:
                break
            
            x = x_new
        
        return eigenvalue, x_new
    
    @staticmethod
    def qr_algorithm(A: np.ndarray, max_iter: int = 1000,
                    tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR Algorithm para encontrar todos los autovalores.
        
        Algoritmo:
        A_0 = A
        Para k = 0, 1, 2, ...:
          Q_k, R_k = QR(A_k)
          A_{k+1} = R_k * Q_k
        
        A_k converge a matriz triangular con autovalores en la diagonal.
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz cuadrada.
        max_iter : int
            Número máximo de iteraciones.
        tol : float
            Tolerancia para convergencia.
        
        Retorna:
        --------
        eigenvalues : np.ndarray
            Autovalores.
        eigenvectors : np.ndarray
            Autovectores (columnas).
        """
        n = A.shape[0]
        A_k = A.copy()
        Q_total = np.eye(n)
        
        for i in range(max_iter):
            # QR decomposition
            Q, R = MatrixDecomposition.qr_decomposition(A_k)
            
            # A_{k+1} = R * Q
            A_k = np.dot(R, Q)
            
            # Acumula transformaciones para autovectores
            Q_total = np.dot(Q_total, Q)
            
            # Verifica convergencia (elementos subdiagonales pequeños)
            if np.max(np.abs(np.tril(A_k, -1))) < tol:
                break
        
        # Autovalores están en la diagonal
        eigenvalues = np.diag(A_k)
        
        # Autovectores son columnas de Q_total
        eigenvectors = Q_total
        
        return eigenvalues, eigenvectors


class LinearSystemSolver:
    """Solvers para sistemas lineales."""
    
    @staticmethod
    def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resuelve A*x = b usando descomposición LU.
        
        Algoritmo:
        1. A = L*U*P (descomposición LU con pivoteo)
        2. Resuelve L*y = P*b (forward substitution)
        3. Resuelve U*x = y (backward substitution)
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz del sistema.
        b : np.ndarray
            Vector del lado derecho.
        
        Retorna:
        --------
        x : np.ndarray
            Solución.
        """
        L, U, P = MatrixDecomposition.lu_decomposition(A)
        b_permuted = np.dot(P, b)
        
        # Forward substitution: L*y = b
        n = len(b)
        y = np.zeros(n)
        for i in range(n):
            y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])
        
        # Backward substitution: U*x = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
        return x
    
    @staticmethod
    def solve_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resuelve A*x = b usando descomposición QR.
        
        Algoritmo:
        1. A = Q*R
        2. Q*R*x = b => R*x = Q^T*b
        3. Resuelve R*x = Q^T*b (backward substitution)
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz del sistema (puede ser rectangular).
        b : np.ndarray
            Vector del lado derecho.
        
        Retorna:
        --------
        x : np.ndarray
            Solución (mínimos cuadrados si A es rectangular).
        """
        Q, R = MatrixDecomposition.qr_decomposition(A)
        
        # Q^T * b
        b_projected = np.dot(Q.T, b)
        
        # Resuelve R*x = b_projected (backward substitution)
        n = R.shape[1]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(R[i, i]) < 1e-10:
                warnings.warn("Sistema singular o mal condicionado")
                x[i] = 0
            else:
                x[i] = (b_projected[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
        
        return x


class MoorePenroseInverse:
    """Pseudoinversa de Moore-Penrose."""
    
    @staticmethod
    def compute(A: np.ndarray) -> np.ndarray:
        """
        Calcula pseudoinversa A^+ usando SVD.
        
        Si A = U*Σ*V^T, entonces A^+ = V*Σ^+*U^T
        donde Σ^+ tiene 1/σ_i en la diagonal (si σ_i > tol), 0 en caso contrario.
        
        Propiedades:
        - A * A^+ * A = A
        - A^+ * A * A^+ = A^+
        - (A * A^+)^T = A * A^+
        - (A^+ * A)^T = A^+ * A
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz (puede ser rectangular).
        
        Retorna:
        --------
        A_pinv : np.ndarray
            Pseudoinversa de Moore-Penrose.
        """
        U, s, Vt = MatrixDecomposition.svd_decomposition(A)
        
        # Σ^+: invierte valores singulares no nulos
        tol = np.max(s) * 1e-10
        s_inv = np.zeros_like(s)
        s_inv[s > tol] = 1.0 / s[s > tol]
        
        # A^+ = V * Σ^+ * U^T
        A_pinv = np.dot(Vt.T, np.dot(np.diag(s_inv), U.T))
        
        return A_pinv

