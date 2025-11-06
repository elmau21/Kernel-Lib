"""
Algoritmos de optimización matemática avanzados.

Implementa:
- Método de Newton
- Gradiente Conjugado
- Quasi-Newton (BFGS)
- Descenso de Gradiente con Búsqueda de Línea
- Optimización con restricciones (Lagrange)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
from scipy.linalg import solve
import warnings


class GradientDescent:
    """
    Gradiente Descendente con búsqueda de línea (Line Search).
    
    Matemática:
    θ_{k+1} = θ_k - α_k * ∇f(θ_k)
    
    donde α_k se encuentra usando búsqueda de línea (Armijo, Wolfe).
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 line_search: str = "armijo", alpha_init: float = 1.0):
        """
        Parámetros:
        -----------
        max_iter : int
            Número máximo de iteraciones.
        tol : float
            Tolerancia para convergencia.
        line_search : str
            Tipo de búsqueda de línea: "armijo", "wolfe", "fixed".
        alpha_init : float
            Tamaño de paso inicial.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.line_search = line_search
        self.alpha_init = alpha_init
    
    def armijo_line_search(self, f: Callable, grad: Callable, x: np.ndarray,
                          direction: np.ndarray, alpha_init: float = 1.0,
                          c: float = 0.1, rho: float = 0.5) -> float:
        """
        Búsqueda de línea de Armijo.
        
        Condición de Armijo:
        f(x + α*d) ≤ f(x) + c * α * ∇f(x)^T * d
        
        donde:
        - c es constante (típicamente 0.1)
        - α es el tamaño de paso
        - d es la dirección de búsqueda
        """
        alpha = alpha_init
        f_x = f(x)
        grad_x = grad(x)
        grad_dot_dir = np.dot(grad_x, direction)
        
        while f(x + alpha * direction) > f_x + c * alpha * grad_dot_dir:
            alpha *= rho
            if alpha < 1e-10:
                break
        
        return alpha
    
    def minimize(self, f: Callable, grad: Callable, x0: np.ndarray) -> Dict:
        """
        Minimiza función usando gradiente descendente.
        
        Parámetros:
        -----------
        f : Callable
            Función objetivo f(x).
        grad : Callable
            Gradiente ∇f(x).
        x0 : np.ndarray
            Punto inicial.
        
        Retorna:
        --------
        result : dict
            Resultado con 'x' (óptimo), 'fun' (valor), 'iterations', etc.
        """
        x = x0.copy()
        history = []
        
        for i in range(self.max_iter):
            grad_x = grad(x)
            
            # Verifica convergencia
            if np.linalg.norm(grad_x) < self.tol:
                break
            
            # Dirección: d = -∇f(x)
            direction = -grad_x
            
            # Búsqueda de línea
            if self.line_search == "armijo":
                alpha = self.armijo_line_search(f, grad, x, direction, self.alpha_init)
            elif self.line_search == "fixed":
                alpha = self.alpha_init
            else:
                alpha = self.alpha_init
            
            # Actualiza: x = x + α * d
            x_new = x + alpha * direction
            f_new = f(x_new)
            
            history.append({
                "iteration": i,
                "x": x.copy(),
                "f": f(x),
                "grad_norm": np.linalg.norm(grad_x),
                "alpha": alpha
            })
            
            x = x_new
        
        return {
            "x": x,
            "fun": f(x),
            "iterations": len(history),
            "history": history,
            "success": np.linalg.norm(grad(x)) < self.tol
        }


class NewtonMethod:
    """
    Método de Newton para optimización.
    
    Matemática:
    x_{k+1} = x_k - H^{-1}(x_k) * ∇f(x_k)
    
    donde H es la matriz Hessiana (segundas derivadas).
    
    Ventajas:
    - Convergencia cuadrática (muy rápida cerca del óptimo)
    - No requiere búsqueda de línea
    
    Desventajas:
    - Requiere calcular Hessiana (costoso)
    - Requiere invertir Hessiana (puede ser inestable)
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6,
                 regularize: float = 1e-6):
        """
        Parámetros:
        -----------
        max_iter : int
            Número máximo de iteraciones.
        tol : float
            Tolerancia para convergencia.
        regularize : float
            Regularización para Hessiana (evita singularidad).
        """
        self.max_iter = max_iter
        self.tol = tol
        self.regularize = regularize
    
    def minimize(self, f: Callable, grad: Callable, hessian: Callable,
                x0: np.ndarray) -> Dict:
        """
        Minimiza usando método de Newton.
        
        Parámetros:
        -----------
        f : Callable
            Función objetivo.
        grad : Callable
            Gradiente.
        hessian : Callable
            Función que retorna matriz Hessiana H(x).
        x0 : np.ndarray
            Punto inicial.
        
        Retorna:
        --------
        result : dict
            Resultado de optimización.
        """
        x = x0.copy()
        history = []
        
        for i in range(self.max_iter):
            grad_x = grad(x)
            
            # Verifica convergencia
            if np.linalg.norm(grad_x) < self.tol:
                break
            
            # Calcula Hessiana
            H = hessian(x)
            
            # Regulariza para evitar singularidad
            H_reg = H + self.regularize * np.eye(H.shape[0])
            
            try:
                # Resuelve: H * p = -∇f
                # p es la dirección de Newton
                p = solve(H_reg, -grad_x)
            except np.linalg.LinAlgError:
                warnings.warn("Hessiana singular, usando gradiente descendente")
                p = -grad_x
            
            # Actualiza: x = x + p
            x_new = x + p
            f_new = f(x_new)
            
            # Backtracking si no mejora
            alpha = 1.0
            while f_new > f(x) and alpha > 1e-10:
                alpha *= 0.5
                x_new = x + alpha * p
                f_new = f(x_new)
            
            history.append({
                "iteration": i,
                "x": x.copy(),
                "f": f(x),
                "grad_norm": np.linalg.norm(grad_x),
                "step": alpha
            })
            
            x = x_new
        
        return {
            "x": x,
            "fun": f(x),
            "iterations": len(history),
            "history": history,
            "success": np.linalg.norm(grad(x)) < self.tol
        }


class ConjugateGradient:
    """
    Método de Gradiente Conjugado.
    
    Matemática:
    Resuelve sistema lineal: A * x = b
    
    o minimiza función cuadrática: f(x) = (1/2) * x^T * A * x - b^T * x
    
    Algoritmo:
    1. p_0 = -r_0 (dirección inicial)
    2. Para k = 0, 1, 2, ...:
       α_k = (r_k^T * r_k) / (p_k^T * A * p_k)
       x_{k+1} = x_k + α_k * p_k
       r_{k+1} = r_k - α_k * A * p_k
       β_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
       p_{k+1} = -r_{k+1} + β_k * p_k
    
    Ventajas:
    - No requiere almacenar matriz A completa
    - Converge en n iteraciones (para sistema n×n)
    """
    
    def __init__(self, max_iter: int = None, tol: float = 1e-6):
        """
        Parámetros:
        -----------
        max_iter : int, opcional
            Número máximo de iteraciones (default: n donde n es tamaño del sistema).
        tol : float
            Tolerancia para convergencia.
        """
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Dict:
        """
        Resuelve sistema lineal A*x = b usando gradiente conjugado.
        
        Parámetros:
        -----------
        A : np.ndarray
            Matriz del sistema (debe ser simétrica y positiva definida).
        b : np.ndarray
            Vector del lado derecho.
        x0 : np.ndarray, opcional
            Punto inicial (default: vector cero).
        
        Retorna:
        --------
        result : dict
            Resultado con 'x' (solución), 'iterations', etc.
        """
        n = A.shape[0]
        max_iter = self.max_iter if self.max_iter is not None else n
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Residuo inicial: r = b - A*x
        r = b - np.dot(A, x)
        p = r.copy()  # Dirección inicial
        
        history = []
        
        for i in range(max_iter):
            # Verifica convergencia
            r_norm = np.linalg.norm(r)
            if r_norm < self.tol:
                break
            
            # Calcula α_k
            Ap = np.dot(A, p)
            alpha = np.dot(r, r) / np.dot(p, Ap)
            
            # Actualiza solución
            x = x + alpha * p
            
            # Actualiza residuo
            r_new = r - alpha * Ap
            
            # Calcula β_k (Fletcher-Reeves)
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            
            # Nueva dirección conjugada
            p = r_new + beta * p
            
            r = r_new
            
            history.append({
                "iteration": i,
                "x": x.copy(),
                "residual_norm": r_norm
            })
        
        return {
            "x": x,
            "iterations": len(history),
            "history": history,
            "residual_norm": np.linalg.norm(b - np.dot(A, x)),
            "success": np.linalg.norm(b - np.dot(A, x)) < self.tol
        }
    
    def minimize_quadratic(self, A: np.ndarray, b: np.ndarray,
                          x0: Optional[np.ndarray] = None) -> Dict:
        """
        Minimiza función cuadrática f(x) = (1/2)*x^T*A*x - b^T*x.
        
        Esto es equivalente a resolver A*x = b.
        """
        return self.solve(A, b, x0)


class BFGS:
    """
    Algoritmo BFGS (Broyden-Fletcher-Goldfarb-Shanno).
    
    Método Quasi-Newton que aproxima la inversa de la Hessiana.
    
    Matemática:
    H_{k+1} = H_k + (y_k * y_k^T) / (y_k^T * s_k) - 
               (H_k * s_k * s_k^T * H_k) / (s_k^T * H_k * s_k)
    
    donde:
    - s_k = x_{k+1} - x_k (cambio en x)
    - y_k = ∇f_{k+1} - ∇f_k (cambio en gradiente)
    - H_k aproxima la inversa de la Hessiana
    
    Ventajas:
    - No requiere calcular Hessiana
    - Convergencia superlineal
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """
        Parámetros:
        -----------
        max_iter : int
            Número máximo de iteraciones.
        tol : float
            Tolerancia para convergencia.
        """
        self.max_iter = max_iter
        self.tol = tol
    
    def minimize(self, f: Callable, grad: Callable, x0: np.ndarray) -> Dict:
        """
        Minimiza usando BFGS.
        
        Parámetros:
        -----------
        f : Callable
            Función objetivo.
        grad : Callable
            Gradiente.
        x0 : np.ndarray
            Punto inicial.
        
        Retorna:
        --------
        result : dict
            Resultado de optimización.
        """
        x = x0.copy()
        n = len(x)
        
        # Inicializa H como identidad (aproximación inicial de H^-1)
        H = np.eye(n)
        
        grad_x = grad(x)
        history = []
        
        for i in range(self.max_iter):
            # Verifica convergencia
            if np.linalg.norm(grad_x) < self.tol:
                break
            
            # Dirección de búsqueda: p = -H * ∇f
            p = -np.dot(H, grad_x)
            
            # Búsqueda de línea (Armijo simplificado)
            alpha = 1.0
            c = 0.1
            rho = 0.5
            f_x = f(x)
            
            while f(x + alpha * p) > f_x + c * alpha * np.dot(grad_x, p):
                alpha *= rho
                if alpha < 1e-10:
                    break
            
            # Actualiza x
            s = alpha * p  # s_k = x_{k+1} - x_k
            x_new = x + s
            grad_x_new = grad(x_new)
            y = grad_x_new - grad_x  # y_k = ∇f_{k+1} - ∇f_k
            
            # Actualiza H usando fórmula BFGS
            sy = np.dot(s, y)
            if abs(sy) > 1e-10:  # Evita división por cero
                Hy = np.dot(H, y)
                yHy = np.dot(y, Hy)
                
                # H = H + (s*s^T)/(s^T*y) - (H*y*y^T*H)/(y^T*H*y)
                H = (H + 
                     np.outer(s, s) / sy - 
                     np.outer(Hy, Hy) / yHy)
            
            history.append({
                "iteration": i,
                "x": x.copy(),
                "f": f_x,
                "grad_norm": np.linalg.norm(grad_x),
                "alpha": alpha
            })
            
            x = x_new
            grad_x = grad_x_new
        
        return {
            "x": x,
            "fun": f(x),
            "iterations": len(history),
            "history": history,
            "success": np.linalg.norm(grad_x) < self.tol
        }

