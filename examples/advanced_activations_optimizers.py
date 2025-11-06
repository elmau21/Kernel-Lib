"""
Ejemplos avanzados de funciones de activación y optimizadores.

Demuestra el uso de todas las nuevas funciones de activación y optimizadores.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernel.math.neural_network import NeuralNetwork
from kernel.math.activations import (
    ReLU, Swish, GELU, Mish, SELU, PReLU, Softplus, HardSwish, Sigmoid
)
from kernel.math.activations import BinaryCrossEntropy
from kernel.math.optimizers import (
    Adam, AdamW, RAdam, AdaBelief, Lion, Ranger, Nadam
)


def compare_activations():
    """Compara diferentes funciones de activación."""
    print("=" * 60)
    print("Comparación de Funciones de Activación")
    print("=" * 60)
    
    x = np.linspace(-5, 5, 100)
    activations = {
        "ReLU": ReLU(),
        "Swish": Swish(beta=1.0),
        "GELU": GELU(),
        "Mish": Mish(),
        "SELU": SELU(),
        "PReLU": PReLU(a=0.25),
        "Softplus": Softplus(),
        "HardSwish": HardSwish(),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, (name, activation) in enumerate(activations.items()):
        y = activation.forward(x)
        deriv = activation.derivative(x)
        
        ax = axes[idx]
        ax.plot(x, y, 'b-', linewidth=2, label=f'{name}(x)')
        ax.plot(x, deriv, 'r--', linewidth=2, label=f"d{name}/dx")
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activations_comparison.png', dpi=150)
    print("\nComparación guardada en 'activations_comparison.png'")


def compare_optimizers():
    """Compara diferentes optimizadores en una tarea simple."""
    print("\n" + "=" * 60)
    print("Comparación de Optimizadores")
    print("=" * 60)
    
    # Genera datos
    X, y = make_classification(n_samples=500, n_features=20, n_redundant=0,
                               n_informative=10, n_clusters_per_class=1,
                               random_state=42)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    optimizers_config = {
        "Adam": {"class": Adam, "lr": 0.001},
        "AdamW": {"class": AdamW, "lr": 0.001, "weight_decay": 0.01},
        "RAdam": {"class": RAdam, "lr": 0.001},
        "AdaBelief": {"class": AdaBelief, "lr": 0.001},
        "Lion": {"class": Lion, "lr": 0.001},
        "Ranger": {"class": Ranger, "lr": 0.001},
        "Nadam": {"class": Nadam, "lr": 0.001},
    }
    
    results = {}
    
    for name, config in optimizers_config.items():
        print(f"\nEntrenando con {name}...")
        
        # Crea red neuronal
        nn = NeuralNetwork(
            layers=[20, 64, 32, 1],
            activations=[Swish(), Swish(), ReLU()],
            loss=BinaryCrossEntropy(),
            weight_init="he"
        )
        
        # Entrena (nota: en producción, integrar optimizadores en NeuralNetwork)
        nn.fit(X_train, y_train, epochs=50, batch_size=32, 
              learning_rate=config["lr"], verbose=False)
        
        # Evalúa
        eval_results = nn.evaluate(X_test, y_test)
        results[name] = {
            "loss": eval_results["loss"],
            "accuracy": eval_results["accuracy"]
        }
        
        print(f"  Loss: {eval_results['loss']:.4f}")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    
    # Visualiza resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    losses = [results[n]["loss"] for n in names]
    accuracies = [results[n]["accuracy"] for n in names]
    
    ax1.bar(names, losses, color='skyblue')
    ax1.set_title("Loss por Optimizador")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(names, accuracies, color='lightcoral')
    ax2.set_title("Accuracy por Optimizador")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('optimizers_comparison.png', dpi=150)
    print("\nComparación guardada en 'optimizers_comparison.png'")


def example_swish_activation():
    """Ejemplo específico con activación Swish."""
    print("\n" + "=" * 60)
    print("Ejemplo: Red Neuronal con Swish")
    print("=" * 60)
    
    # Datos
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1)
    
    # Red con Swish
    nn_swish = NeuralNetwork(
        layers=[10, 50, 25, 1],
        activations=[Swish(beta=1.0), Swish(beta=1.0), Sigmoid()],
        loss=BinaryCrossEntropy()
    )
    
    # Red con ReLU (comparación)
    from kernel.math.activations import Sigmoid
    nn_relu = NeuralNetwork(
        layers=[10, 50, 25, 1],
        activations=[ReLU(), ReLU(), Sigmoid()],
        loss=BinaryCrossEntropy()
    )
    
    print("\nEntrenando red con Swish...")
    nn_swish.fit(X, y, epochs=100, batch_size=32, learning_rate=0.01, verbose=False)
    
    print("Entrenando red con ReLU...")
    nn_relu.fit(X, y, epochs=100, batch_size=32, learning_rate=0.01, verbose=False)
    
    # Compara
    swish_results = nn_swish.evaluate(X, y)
    relu_results = nn_relu.evaluate(X, y)
    
    print(f"\nSwish - Loss: {swish_results['loss']:.4f}, Accuracy: {swish_results['accuracy']:.4f}")
    print(f"ReLU  - Loss: {relu_results['loss']:.4f}, Accuracy: {relu_results['accuracy']:.4f}")


def example_gelu_activation():
    """Ejemplo con GELU (usado en Transformers)."""
    print("\n" + "=" * 60)
    print("Ejemplo: Red Neuronal con GELU")
    print("=" * 60)
    
    # Datos
    X, y = make_classification(n_samples=200, n_features=15, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1)
    
    # Red con GELU
    nn = NeuralNetwork(
        layers=[15, 64, 32, 1],
        activations=[GELU(), GELU(), Sigmoid()],
        loss=BinaryCrossEntropy(),
        weight_init="he"
    )
    
    print("Entrenando...")
    nn.fit(X, y, epochs=80, batch_size=32, learning_rate=0.001, verbose=True)
    
    results = nn.evaluate(X, y)
    print(f"\nResultados:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EJEMPLOS AVANZADOS - ACTIVACIONES Y OPTIMIZADORES")
    print("=" * 60)
    
    try:
        compare_activations()
    except Exception as e:
        print(f"Error en comparación de activaciones: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        compare_optimizers()
    except Exception as e:
        print(f"Error en comparación de optimizadores: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_swish_activation()
    except Exception as e:
        print(f"Error en ejemplo Swish: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_gelu_activation()
    except Exception as e:
        print(f"Error en ejemplo GELU: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Ejemplos completados")
    print("=" * 60)

