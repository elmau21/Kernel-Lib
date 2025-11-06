"""
Ejemplo completo de Red Neuronal implementada matemáticamente desde cero.

Demuestra:
- Forward propagation
- Backpropagation completo
- Entrenamiento con diferentes optimizadores
- Comparación de funciones de activación
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernel.math.neural_network import NeuralNetwork
from kernel.math.activations import Sigmoid, Tanh, ReLU, Softmax, Linear
from kernel.math.activations import MeanSquaredError, CrossEntropy, BinaryCrossEntropy


def example_binary_classification():
    """Ejemplo de clasificación binaria con red neuronal."""
    print("=" * 60)
    print("Ejemplo: Clasificación Binaria")
    print("=" * 60)
    
    # Genera datos
    X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0,
                               n_informative=10, n_clusters_per_class=1,
                               random_state=42)
    
    # Normaliza
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convierte a formato correcto
    y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crea red neuronal
    # Arquitectura: 20 -> 64 -> 32 -> 1
    nn = NeuralNetwork(
        layers=[20, 64, 32, 1],
        activations=[ReLU(), ReLU(), Sigmoid()],
        loss=BinaryCrossEntropy(),
        weight_init="he"
    )
    
    print("\nArquitectura de la red:")
    print(f"  Capas: {nn.layers_config}")
    print(f"  Total de parámetros: {sum(l.W.size + (l.b.size if l.b is not None else 0) for l in nn.layers)}")
    
    # Entrena
    print("\nEntrenando...")
    nn.fit(X_train, y_train, epochs=100, batch_size=32, 
          learning_rate=0.01, validation_data=(X_test, y_test), verbose=True)
    
    # Evalúa
    results = nn.evaluate(X_test, y_test)
    print(f"\nResultados en test:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    
    # Visualiza pérdida
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nn.history['loss'], label='Train Loss')
    if nn.history['val_loss']:
        plt.plot(nn.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pérdida durante Entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predicciones
    y_pred = nn.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.flatten(), 
               cmap='RdYlBu', alpha=0.6, label='True')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_binary.flatten(),
               cmap='RdYlBu', marker='x', s=100, label='Predicted')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Predicciones')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nn_classification.png', dpi=150)
    print("\nVisualización guardada en 'nn_classification.png'")


def example_regression():
    """Ejemplo de regresión con red neuronal."""
    print("\n" + "=" * 60)
    print("Ejemplo: Regresión")
    print("=" * 60)
    
    # Genera datos
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    
    # Normaliza
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crea red neuronal para regresión
    nn = NeuralNetwork(
        layers=[10, 50, 25, 1],
        activations=[ReLU(), ReLU(), Linear()],
        loss=MeanSquaredError(),
        weight_init="xavier"
    )
    
    print("\nEntrenando red neuronal para regresión...")
    nn.fit(X_train, y_train, epochs=200, batch_size=32,
          learning_rate=0.001, validation_data=(X_test, y_test), verbose=True)
    
    # Evalúa
    results = nn.evaluate(X_test, y_test)
    print(f"\nResultados en test:")
    print(f"  Loss (MSE): {results['loss']:.4f}")
    print(f"  R² Score: {results['accuracy']:.4f}")
    
    # Visualiza
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn.history['loss'], label='Train Loss')
    if nn.history['val_loss']:
        plt.plot(nn.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Pérdida durante Entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    y_pred = nn.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Valores Verdaderos')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Verdaderos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_regression.png', dpi=150)
    print("\nVisualización guardada en 'nn_regression.png'")


def example_backpropagation_math():
    """Demuestra la matemática del backpropagation."""
    print("\n" + "=" * 60)
    print("Demostración: Matemática del Backpropagation")
    print("=" * 60)
    
    # Red simple: 2 -> 3 -> 1
    nn = NeuralNetwork(
        layers=[2, 3, 1],
        activations=[Sigmoid(), Sigmoid()],
        loss=MeanSquaredError()
    )
    
    # Datos de ejemplo
    X = np.array([[0.5, 0.3]]).T  # (2, 1)
    y = np.array([[0.8]])  # (1, 1)
    
    print("\nForward Propagation:")
    print(f"  Input X: shape {X.shape}")
    
    # Forward manual para mostrar matemática
    a0 = X
    print(f"  a^[0] (input):\n{a0}")
    
    # Capa 1
    W1 = nn.layers[0].W
    b1 = nn.layers[0].b
    z1 = np.dot(W1, a0) + b1
    a1 = nn.layers[0].activation.forward(z1)
    print(f"\n  Capa 1:")
    print(f"    z^[1] = W^[1] * a^[0] + b^[1]:\n{z1}")
    print(f"    a^[1] = g^[1](z^[1]):\n{a1}")
    
    # Capa 2
    W2 = nn.layers[1].W
    b2 = nn.layers[1].b
    z2 = np.dot(W2, a1) + b2
    a2 = nn.layers[1].activation.forward(z2)
    print(f"\n  Capa 2:")
    print(f"    z^[2] = W^[2] * a^[1] + b^[2]:\n{z2}")
    print(f"    a^[2] (output):\n{a2}")
    
    # Pérdida
    loss = nn.loss.forward(a2, y)
    print(f"\n  Loss: {loss:.6f}")
    
    print("\nBackward Propagation:")
    print("  Usando regla de la cadena:")
    print("    ∂L/∂W^[l] = ∂L/∂z^[l] * (a^[l-1])^T")
    print("    ∂L/∂b^[l] = Σ(∂L/∂z^[l])")
    print("    ∂L/∂a^[l-1] = (W^[l])^T * ∂L/∂z^[l]")
    
    # Backward
    gradients = nn.backward(a2, y)
    
    print(f"\n  Gradientes calculados para {len(gradients)} capas")
    for i, (grad_W, grad_b) in enumerate(gradients):
        print(f"    Capa {i+1}:")
        print(f"      ∂L/∂W^[{i+1}]: shape {grad_W.shape}, norm = {np.linalg.norm(grad_W):.6f}")
        if grad_b is not None:
            print(f"      ∂L/∂b^[{i+1}]: shape {grad_b.shape}, norm = {np.linalg.norm(grad_b):.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EJEMPLOS DE REDES NEURONALES - IMPLEMENTACIÓN MATEMÁTICA")
    print("=" * 60)
    
    try:
        example_binary_classification()
    except Exception as e:
        print(f"Error en clasificación: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_regression()
    except Exception as e:
        print(f"Error en regresión: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_backpropagation_math()
    except Exception as e:
        print(f"Error en demostración matemática: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Ejemplos completados")
    print("=" * 60)

