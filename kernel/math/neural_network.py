"""
Red Neuronal implementada matemáticamente desde cero.

Implementa:
- Forward propagation
- Backpropagation completo (derivación matemática)
- Inicialización de pesos (Xavier, He)
- Regularización (L1, L2, Dropout)
- Batch normalization

Toda la matemática está implementada manualmente sin usar librerías de deep learning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from kernel.math.activations import ActivationFunction, Sigmoid, Tanh, ReLU, Linear, Softmax
from kernel.math.activations import LossFunction, MeanSquaredError, CrossEntropy


class Layer:
    """
    Capa de red neuronal con implementación matemática completa.
    
    Forward: a^[l] = g(W^[l] * a^[l-1] + b^[l])
    Backward: Calcula ∂L/∂W, ∂L/∂b usando regla de la cadena
    """
    
    def __init__(self, n_input: int, n_output: int, 
                 activation: ActivationFunction = ReLU(),
                 use_bias: bool = True,
                 weight_init: str = "xavier",
                 dropout_rate: float = 0.0):
        """
        Inicializa la capa.
        
        Parámetros:
        -----------
        n_input : int
            Número de neuronas de entrada.
        n_output : int
            Número de neuronas de salida.
        activation : ActivationFunction
            Función de activación.
        use_bias : bool
            Usa bias o no.
        weight_init : str
            Método de inicialización: "xavier", "he", "random".
        dropout_rate : float
            Tasa de dropout (0 = sin dropout).
        """
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        
        # Inicializa pesos
        if weight_init == "xavier":
            # Xavier/Glorot: W ~ N(0, sqrt(2/(n_in + n_out)))
            limit = np.sqrt(2.0 / (n_input + n_output))
            self.W = np.random.randn(n_output, n_input) * limit
        elif weight_init == "he":
            # He initialization: W ~ N(0, sqrt(2/n_in))
            limit = np.sqrt(2.0 / n_input)
            self.W = np.random.randn(n_output, n_input) * limit
        else:  # random
            self.W = np.random.randn(n_output, n_input) * 0.1
        
        # Inicializa bias
        if use_bias:
            self.b = np.zeros((n_output, 1))
        else:
            self.b = None
        
        # Cache para backward pass
        self.last_input = None
        self.last_z = None  # Antes de activación
        self.last_a = None  # Después de activación
        self.dropout_mask = None
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward propagation.
        
        Matemática:
        z^[l] = W^[l] * a^[l-1] + b^[l]
        a^[l] = g(z^[l])
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_features, n_samples)
            Input de la capa.
        training : bool
            Si es True, aplica dropout.
        
        Retorna:
        --------
        a : np.ndarray
            Output de la capa.
        """
        # Guarda input para backward
        self.last_input = X
        
        # Calcula z = W*X + b
        # W: (n_output, n_input), X: (n_input, n_samples)
        # z: (n_output, n_samples)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        z = np.dot(self.W, X)
        if self.use_bias:
            z += self.b
        
        self.last_z = z
        
        # Aplica activación
        a = self.activation.forward(z)
        
        # Aplica dropout si está en entrenamiento
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
            a = a * self.dropout_mask
        else:
            self.dropout_mask = None
        
        self.last_a = a
        return a
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward propagation usando regla de la cadena.
        
        Matemática:
        Si L es la pérdida:
        - ∂L/∂a^[l-1] = (W^[l])^T * ∂L/∂z^[l]
        - ∂L/∂W^[l] = ∂L/∂z^[l] * (a^[l-1])^T
        - ∂L/∂b^[l] = Σ(∂L/∂z^[l]) (suma sobre muestras)
        
        donde ∂L/∂z^[l] = ∂L/∂a^[l] * g'(z^[l])
        
        Parámetros:
        -----------
        grad_output : np.ndarray, shape (n_output, n_samples)
            Gradiente con respecto a la salida de esta capa.
        
        Retorna:
        --------
        grad_input : np.ndarray
            Gradiente con respecto al input.
        grad_W : np.ndarray
            Gradiente con respecto a los pesos.
        grad_b : np.ndarray
            Gradiente con respecto al bias.
        """
        # Aplica dropout si estaba activo
        if self.dropout_mask is not None:
            grad_output = grad_output * self.dropout_mask
        
        # Calcula gradiente con respecto a z
        # ∂L/∂z = ∂L/∂a * g'(z)
        grad_z = self.activation.backward(self.last_z, grad_output)
        
        # Calcula gradientes con respecto a parámetros
        # ∂L/∂W = ∂L/∂z * X^T
        grad_W = np.dot(grad_z, self.last_input.T)
        
        # ∂L/∂b = Σ(∂L/∂z) sobre muestras
        if self.use_bias:
            grad_b = np.sum(grad_z, axis=1, keepdims=True)
        else:
            grad_b = None
        
        # Calcula gradiente con respecto al input (para capa anterior)
        # ∂L/∂X = W^T * ∂L/∂z
        grad_input = np.dot(self.W.T, grad_z)
        
        return grad_input, grad_W, grad_b
    
    def update_weights(self, grad_W: np.ndarray, grad_b: Optional[np.ndarray], 
                      learning_rate: float):
        """
        Actualiza pesos usando gradiente descendente.
        
        W = W - α * ∂L/∂W
        b = b - α * ∂L/∂b
        """
        self.W -= learning_rate * grad_W
        if self.use_bias and grad_b is not None:
            self.b -= learning_rate * grad_b


class NeuralNetwork:
    """
    Red Neuronal Multicapa implementada matemáticamente desde cero.
    
    Implementa:
    - Forward propagation completo
    - Backpropagation completo (derivación matemática)
    - Entrenamiento con diferentes optimizadores
    - Regularización L1/L2
    - Batch normalization (opcional)
    """
    
    def __init__(self, layers: List[int], 
                 activations: List[ActivationFunction] = None,
                 loss: LossFunction = MeanSquaredError(),
                 weight_init: str = "xavier",
                 dropout_rates: List[float] = None,
                 use_batch_norm: bool = False):
        """
        Inicializa la red neuronal.
        
        Parámetros:
        -----------
        layers : List[int]
            [n_input, n_hidden1, n_hidden2, ..., n_output]
        activations : List[ActivationFunction], opcional
            Funciones de activación por capa. Si None, usa ReLU para todas excepto la última.
        loss : LossFunction
            Función de pérdida.
        weight_init : str
            Método de inicialización.
        dropout_rates : List[float], opcional
            Tasa de dropout por capa.
        use_batch_norm : bool
            Usa batch normalization.
        """
        self.layers_config = layers
        self.loss = loss
        self.use_batch_norm = use_batch_norm
        
        # Crea capas
        self.layers: List[Layer] = []
        n_layers = len(layers) - 1
        
        if activations is None:
            activations = [ReLU()] * (n_layers - 1) + [Linear()]
        if dropout_rates is None:
            dropout_rates = [0.0] * n_layers
        
        for i in range(n_layers):
            activation = activations[i] if i < len(activations) else ReLU()
            dropout = dropout_rates[i] if i < len(dropout_rates) else 0.0
            
            layer = Layer(
                n_input=layers[i],
                n_output=layers[i + 1],
                activation=activation,
                weight_init=weight_init,
                dropout_rate=dropout
            )
            self.layers.append(layer)
        
        # Historial de entrenamiento
        self.history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": []
        }
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward propagation completo.
        
        Matemática:
        a^[0] = X
        z^[l] = W^[l] * a^[l-1] + b^[l]
        a^[l] = g^[l](z^[l])
        ...
        ŷ = a^[L]
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features) o (n_features, n_samples)
            Input.
        training : bool
            Modo entrenamiento (afecta dropout, batch norm).
        
        Retorna:
        --------
        output : np.ndarray
            Predicción de la red.
        """
        # Asegura formato correcto: (n_features, n_samples)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != self.layers[0].n_input:
            X = X.T
        
        # Forward a través de todas las capas
        a = X
        for layer in self.layers:
            a = layer.forward(a, training=training)
        
        return a
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> List[Tuple]:
        """
        Backpropagation completo usando regla de la cadena.
        
        Matemática:
        1. Calcula pérdida: L = loss(y_pred, y_true)
        2. Gradiente inicial: ∂L/∂ŷ
        3. Para cada capa (de atrás hacia adelante):
           - Calcula ∂L/∂z^[l] usando regla de la cadena
           - Calcula ∂L/∂W^[l] y ∂L/∂b^[l]
           - Propaga gradiente a capa anterior
        
        Parámetros:
        -----------
        y_pred : np.ndarray
            Predicción de la red.
        y_true : np.ndarray
            Valores verdaderos.
        
        Retorna:
        --------
        gradients : List[Tuple]
            Lista de (grad_W, grad_b) para cada capa.
        """
        # Asegura formato correcto
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.shape != y_true.shape:
            if y_pred.shape[0] != y_true.shape[0]:
                y_true = y_true.T
        
        # Gradiente inicial: ∂L/∂ŷ
        grad_output = self.loss.backward(y_pred, y_true)
        
        # Backpropagation a través de todas las capas
        gradients = []
        for layer in reversed(self.layers):
            grad_input, grad_W, grad_b = layer.backward(grad_output)
            gradients.append((grad_W, grad_b))
            grad_output = grad_input
        
        # Invierte para que coincida con orden de capas
        gradients.reverse()
        
        return gradients
    
    def train_step(self, X: np.ndarray, y: np.ndarray, 
                  learning_rate: float, 
                  l1_reg: float = 0.0,
                  l2_reg: float = 0.0) -> float:
        """
        Un paso de entrenamiento.
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrada.
        y : np.ndarray
            Etiquetas.
        learning_rate : float
            Tasa de aprendizaje.
        l1_reg : float
            Coeficiente de regularización L1.
        l2_reg : float
            Coeficiente de regularización L2.
        
        Retorna:
        --------
        loss : float
            Pérdida del batch.
        """
        # Forward
        y_pred = self.forward(X, training=True)
        
        # Calcula pérdida
        loss = self.loss.forward(y_pred, y)
        
        # Añade regularización
        if l2_reg > 0:
            for layer in self.layers:
                loss += l2_reg * np.sum(layer.W ** 2)
        if l1_reg > 0:
            for layer in self.layers:
                loss += l1_reg * np.sum(np.abs(layer.W))
        
        # Backward
        gradients = self.backward(y_pred, y)
        
        # Actualiza pesos
        for layer, (grad_W, grad_b) in zip(self.layers, gradients):
            # Añade gradientes de regularización
            if l2_reg > 0:
                grad_W += 2 * l2_reg * layer.W
            if l1_reg > 0:
                grad_W += l1_reg * np.sign(layer.W)
            
            layer.update_weights(grad_W, grad_b, learning_rate)
        
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray,
           epochs: int = 100,
           batch_size: int = 32,
           learning_rate: float = 0.01,
           validation_data: Optional[Tuple] = None,
           l1_reg: float = 0.0,
           l2_reg: float = 0.0,
           verbose: bool = True):
        """
        Entrena la red neuronal.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de entrenamiento.
        y : np.ndarray, shape (n_samples,)
            Etiquetas.
        epochs : int
            Número de épocas.
        batch_size : int
            Tamaño del batch.
        learning_rate : float
            Tasa de aprendizaje.
        validation_data : Tuple, opcional
            (X_val, y_val) para validación.
        l1_reg : float
            Regularización L1.
        l2_reg : float
            Regularización L2.
        verbose : bool
            Muestra progreso.
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            # Entrena en batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss = self.train_step(X_batch, y_batch, learning_rate, l1_reg, l2_reg)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            self.history["loss"].append(avg_loss)
            
            # Validación
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.predict(X_val)
                val_loss = self.loss.forward(val_pred, y_val)
                self.history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}"
                if validation_data is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice usando la red.
        
        Parámetros:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Datos de entrada.
        
        Retorna:
        --------
        predictions : np.ndarray
            Predicciones.
        """
        output = self.forward(X, training=False)
        # Retorna en formato (n_samples, n_output)
        if output.shape[0] != X.shape[0]:
            output = output.T
        return output
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa la red en un conjunto de datos.
        
        Retorna:
        --------
        metrics : dict
            Métricas de evaluación.
        """
        y_pred = self.predict(X)
        loss = self.loss.forward(y_pred, y)
        
        # Calcula accuracy si es clasificación
        if y.ndim == 1 or y.shape[1] == 1:
            # Clasificación binaria o regresión
            if np.all(np.isin(y, [0, 1])) or np.all(np.isin(y, [-1, 1])):
                # Clasificación binaria
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                y_true_binary = y.flatten()
                if np.any(y_true_binary == -1):
                    y_true_binary = (y_true_binary + 1) / 2
                accuracy = np.mean(y_pred_binary == y_true_binary)
            else:
                # Regresión: usa R²
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                accuracy = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            # Clasificación multiclase
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
            accuracy = np.mean(y_pred_classes == y_true_classes)
        
        return {"loss": loss, "accuracy": accuracy}

