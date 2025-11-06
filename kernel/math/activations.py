"""
Funciones de activación implementadas matemáticamente desde cero.

Todas las funciones incluyen:
- Forward pass
- Derivada (backward pass)
- Propiedades matemáticas
"""

import numpy as np
from typing import Callable, Tuple


class ActivationFunction:
    """Clase base para funciones de activación."""
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa la función de activación."""
        raise NotImplementedError
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Calcula la derivada: d(activation)/dx * grad_output.
        
        Parámetros:
        -----------
        x : np.ndarray
            Input original (antes de activación).
        grad_output : np.ndarray
            Gradiente que viene de la capa siguiente.
        
        Retorna:
        --------
        grad_input : np.ndarray
            Gradiente con respecto al input.
        """
        raise NotImplementedError
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función."""
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """
    Función sigmoide: σ(x) = 1 / (1 + exp(-x))
    
    Propiedades:
    - Rango: (0, 1)
    - Derivada: σ'(x) = σ(x)(1 - σ(x))
    - Suave y diferenciable
    - Útil para probabilidades
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa sigmoide con estabilidad numérica."""
        # Evita overflow: clip x a rango seguro
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada: σ'(x) = σ(x)(1 - σ(x))"""
        s = self.forward(x)
        return s * (1.0 - s)
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: grad_input = σ'(x) * grad_output"""
        return self.derivative(x) * grad_output


class Tanh(ActivationFunction):
    """
    Tangente hiperbólica: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Propiedades:
    - Rango: (-1, 1)
    - Derivada: tanh'(x) = 1 - tanh²(x)
    - Centrada en cero (mejor que sigmoid para algunas tareas)
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa tanh."""
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada: tanh'(x) = 1 - tanh²(x)"""
        t = self.forward(x)
        return 1.0 - t ** 2
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return self.derivative(x) * grad_output


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit: ReLU(x) = max(0, x)
    
    Propiedades:
    - No diferenciable en x=0 (usamos subgradiente)
    - Derivada: 1 si x > 0, 0 si x <= 0
    - Muy eficiente computacionalmente
    - Soluciona problema de vanishing gradients
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa ReLU."""
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada: 1 si x > 0, 0 si x <= 0"""
        return (x > 0).astype(np.float64)
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return self.derivative(x) * grad_output


class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU: LeakyReLU(x) = max(αx, x) donde α es pequeño (típicamente 0.01)
    
    Propiedades:
    - Soluciona problema de "dying ReLU"
    - Permite gradientes pequeños para valores negativos
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Parámetros:
        -----------
        alpha : float, default=0.01
            Pendiente para valores negativos.
        """
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa LeakyReLU."""
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada: 1 si x > 0, α si x <= 0"""
        return np.where(x > 0, 1.0, self.alpha)
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return self.derivative(x) * grad_output


class ELU(ActivationFunction):
    """
    Exponential Linear Unit: ELU(x) = x si x > 0, α(exp(x) - 1) si x <= 0
    
    Propiedades:
    - Suave y diferenciable
    - Mejor que ReLU para algunas tareas
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Parámetros:
        -----------
        alpha : float, default=1.0
            Parámetro de escala.
        """
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa ELU."""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1.0))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada: 1 si x > 0, α*exp(x) si x <= 0"""
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return self.derivative(x) * grad_output


class Softmax(ActivationFunction):
    """
    Softmax: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Propiedades:
    - Normaliza a distribución de probabilidad
    - Útil para clasificación multiclase
    - Derivada más compleja (matriz Jacobiana)
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evalúa softmax con estabilidad numérica.
        
        Usa: softmax(x) = softmax(x - max(x))
        """
        # Estabilidad numérica: resta el máximo
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivada de softmax.

        Para softmax: ∂softmax_i/∂x_j = softmax_i(δ_ij - softmax_j)
        donde δ_ij es delta de Kronecker.

        Retorna matriz Jacobiana para cada muestra.
        Gestiona casos límite y posibles sesgos numéricos.
        """
        if x is None or np.size(x) == 0:
            raise ValueError("La entrada x a softmax.derivative no puede ser None ni vacía.")

        s = self.forward(x)
        # Fuerza tipo float64 para evitar errores inesperados por tipo
        s = np.asarray(s, dtype=np.float64)

        # Si solo una muestra (vector 1D)
        scalar_input = False
        if x.ndim == 0 or (x.ndim == 1 and s.shape[0] == 1):
            # Caso escalar, trivial
            jacobian = np.array([[0.0]])
            return jacobian
        if x.ndim == 1:
            s = s.reshape(1, -1)
            x = x.reshape(1, -1)
            scalar_input = True

        batch_size, n_classes = s.shape
        # Edge case: clases únicas, jacobiano trivial
        if n_classes == 1:
            jacobian = np.zeros((batch_size, 1, 1), dtype=np.float64)
            if scalar_input:
                return jacobian[0]
            return jacobian

        jacobian = np.zeros((batch_size, n_classes, n_classes), dtype=np.float64)
        for i in range(batch_size):
            # Vector softmax (s_i)
            s_vec = s[i]
            # Evita sesgo numérico forzando suma = 1 por estabilidad
            s_vec_sum = s_vec.sum()
            if not np.isclose(s_vec_sum, 1.0):
                s_vec = s_vec / (s_vec_sum + 1e-12)
            # Jacobiano: diag(s) - s.T @ s
            outer = np.outer(s_vec, s_vec)
            jacobian[i] = np.diag(s_vec) - outer
        print(jacobian)

        if scalar_input:
            return jacobian[0]
        return jacobian
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass para softmax.
        
        grad_input = J^T @ grad_output
        donde J es el Jacobiano.
        """
        jacobian = self.derivative(x)
        
        if x.ndim == 1:
            return jacobian.T @ grad_output
        else:
            # Para batch: (batch_size, n_classes, n_classes) @ (batch_size, n_classes)
            batch_size = x.shape[0]
            grad_input = np.zeros_like(x)
            for i in range(batch_size):
                grad_input[i] = jacobian[i].T @ grad_output[i]
            return grad_input


class Linear(ActivationFunction):
    """
    Función lineal (identidad): f(x) = x
    
    Propiedades:
    - Derivada: 1
    - Útil para regresión
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evalúa función lineal."""
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada: 1"""
        return np.ones_like(x)
    
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad_output


# Funciones de pérdida matemáticas
class LossFunction:
    """Clase base para funciones de pérdida."""
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula la pérdida."""
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula la pérdida."""
        raise NotImplementedError
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la pérdida."""
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error: MSE = (1/n) Σ(y_pred - y_true)²
    
    Propiedades:
    - Diferenciable
    - Penaliza errores grandes más que pequeños
    - Útil para regresión
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula MSE."""
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradiente: ∂MSE/∂y_pred = (2/n)(y_pred - y_true)"""
        return 2.0 * (y_pred - y_true) / y_pred.size


class CrossEntropy(LossFunction):
    """
    Cross-Entropy: CE = -Σ y_true * log(y_pred)
    
    Propiedades:
    - Útil para clasificación
    - Penaliza predicciones incorrectas fuertemente
    - Combinado con softmax es muy eficiente
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula cross-entropy con estabilidad numérica."""
        # Evita log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Gradiente de cross-entropy.
        
        Si se usa con softmax: grad = y_pred - y_true
        (esto es una simplificación matemática importante)
        """
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred_clipped / y_pred.shape[0]


class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy: BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
    Para clasificación binaria.
    """
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula BCE."""
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped) + 
                       (1 - y_true) * np.log(1 - y_pred_clipped))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradiente: ∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))"""
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / y_pred.size

